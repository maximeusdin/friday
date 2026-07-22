# 2-Tier Alias Universe Implementation Plan
## Precision-First Strategy Without Losing Recall

## Overview

This plan implements a 2-tier alias system:
- **Tier A**: High-precision auto-match aliases (only these produce mentions automatically)
- **Tier B**: Searchable but not auto-match (preserves recall for interactive search/adjudication)

---

## Phase 1: Immediate Fixes (Biggest Wins)

### 1.1 Fix Alias Ingest: Set `is_auto_match` Based on `alias_type`

**File**: `concordance/ingest_concordance_tab_aware.py`

**Location**: `ensure_alias()` function (around line 2672)

**Changes**:
```python
def ensure_alias(cur, source_id: int, entry_id: Optional[int], entity_id: int, al: ParsedAlias) -> Optional[int]:
    filtered = filter_alias_to_capitalized_tokens(al.alias)
    if filtered is None:
        return None

    alias_norm = normalize_alias_for_db(filtered)
    
    # Tier A: Auto-match only for high-precision alias types
    AUTO_MATCH_ALIAS_TYPES = {
        "canonical", "original_form", "bracket_variant", 
        "cover_name", "covername_from_body"
    }
    
    # Conditionally allow head_syn (only if not generic word)
    is_auto_match = al.alias_type in AUTO_MATCH_ALIAS_TYPES
    if al.alias_type == "head_syn":
        # Check if it's a generic word (add logic here)
        if filtered.lower() not in GENERIC_WORDS_TO_EXCLUDE:
            is_auto_match = True
    
    # Never auto-match these types
    NEVER_AUTO_MATCH_TYPES = {
        "definition", "scoped_label", "see", "work_name"
    }
    if al.alias_type in NEVER_AUTO_MATCH_TYPES:
        is_auto_match = False
    
    # Override: if DB already has is_auto_match set, respect it (for manual overrides)
    # But apply our rules as defaults
    
    # ... rest of function
```

**Impact**: Eliminates "soviet", "american", "light bomber" etc. from auto-matching (these are `definition` type).

---

### 1.2 Ban Single-Letter and 2-Letter Covernames (Unless Quoted/Bracketed)

**File**: `scripts/extract_entity_mentions.py`

**Location**: `load_all_aliases()` function (around line 1012)

**Changes**:
```python
# After entity type gate, add single/2-letter covername gate
if alias_class == "covername" and len(tokens) == 1:
    token = tokens[0]
    # Single letter: only allow if quoted/bracketed in original alias
    if len(token) == 1:
        original_has_quotes = '"' in alias or "'" in alias or '[' in alias or '(' in alias
        if not original_has_quotes:
            is_auto_match = False
    
    # 2-letter: only allow if ALLCAPS acronym (KGB, GRU, MGB, etc.)
    elif len(token) == 2:
        known_acronyms = {"kgb", "gru", "mgb", "nkvd", "sis", "mi6", "fbi", "cia", "nsa"}
        if token.lower() not in known_acronyms or not alias.isupper():
            is_auto_match = False
```

**Impact**: Removes "I", "A", "M", "US", "PM" (unless they're legitimate acronyms or quoted).

---

### 1.3 Stop Auto-Matching Generic Label Entities

**File**: `scripts/extract_entity_mentions.py`

**Location**: `load_all_aliases()` function (around line 1012)

**Changes**:
```python
# Add after entity type gate
GENERIC_LABEL_ALIASES = {
    "president", "general", "group", "ref", "minister", "chief", 
    "doctor", "agent", "officer", "director", "secretary"
}

if alias_norm.lower() in GENERIC_LABEL_ALIASES:
    # Only allow if it's part of a multi-word alias (e.g., "President Roosevelt")
    if len(tokens) == 1:
        is_auto_match = False
```

**Impact**: Removes standalone "president", "general", "group" from auto-matching.

---

## Phase 2: Derived Alias Types (Surname + Acronym)

### 2.1 Add Derived Surname Aliases for People

**New File**: `scripts/derive_surname_aliases.py` (or add to ingest script)

**Logic**:
```python
def derive_surname_aliases(conn, source_slug: str):
    """
    For each person entity with multi-token canonical/original name,
    create a derived alias for the last token (surname).
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT e.id, e.canonical_name, ea.alias_norm
            FROM entities e
            JOIN entity_aliases ea ON ea.entity_id = e.id
            JOIN concordance_entries ce ON ce.id = ea.entry_id
            JOIN concordance_sources cs ON cs.id = ce.source_id
            WHERE e.entity_type = 'person'
              AND cs.slug = %s
              AND ea.alias_type IN ('canonical', 'original_form')
              AND array_length(string_to_array(ea.alias_norm, ' '), 1) > 1
        """, (source_slug,))
        
        for entity_id, canonical, alias_norm in cur.fetchall():
            tokens = alias_norm.split()
            surname = tokens[-1]
            
            # Only derive if surname is >= 4 chars and not generic
            if len(surname) >= 4 and surname.lower() not in GENERIC_WORDS_TO_EXCLUDE:
                # Check if surname alias already exists
                cur.execute("""
                    SELECT id FROM entity_aliases
                    WHERE entity_id = %s AND alias_norm = %s
                """, (entity_id, surname))
                
                if not cur.fetchone():
                    cur.execute("""
                        INSERT INTO entity_aliases 
                        (entity_id, alias, alias_norm, alias_type, alias_class, 
                         is_auto_match, is_matchable, min_chars, match_case)
                        VALUES (%s, %s, %s, 'derived_last_name', 'person_last',
                                true, true, 4, 'titlecase_only')
                    """, (entity_id, surname, surname))
```

**Integration**: Run this after concordance ingest, before extraction.

---

### 2.2 Add Derived Acronym Aliases for Orgs/Agencies

**New File**: `scripts/derive_acronym_aliases.py`

**Logic**:
```python
KNOWN_ACRONYMS = {
    "kgb": "Komitet Gosudarstvennoy Bezopasnosti",
    "gru": "Glavnoye Razvedyvatelnoye Upravleniye",
    "mgb": "Ministerstvo Gosudarstvennoy Bezopasnosti",
    "nkvd": "Narodnyy Komissariat Vnutrennikh Del",
    "fbi": "Federal Bureau of Investigation",
    "cia": "Central Intelligence Agency",
    # ... expand list
}

def derive_acronym_aliases(conn, source_slug: str):
    """
    For org entities, create derived acronym aliases if the canonical name
    matches a known acronym expansion pattern.
    """
    with conn.cursor() as cur:
        for acronym, expansion_pattern in KNOWN_ACRONYMS.items():
            # Find org entities whose canonical name contains the expansion
            cur.execute("""
                SELECT DISTINCT e.id
                FROM entities e
                JOIN entity_aliases ea ON ea.entity_id = e.id
                JOIN concordance_entries ce ON ce.id = ea.entry_id
                JOIN concordance_sources cs ON cs.id = ce.source_id
                WHERE e.entity_type = 'org'
                  AND cs.slug = %s
                  AND ea.alias_type = 'canonical'
                  AND LOWER(ea.alias) LIKE %s
            """, (source_slug, f"%{expansion_pattern.lower()}%"))
            
            for (entity_id,) in cur.fetchall():
                # Check if acronym alias already exists
                cur.execute("""
                    SELECT id FROM entity_aliases
                    WHERE entity_id = %s AND alias_norm = %s
                """, (entity_id, acronym))
                
                if not cur.fetchone():
                    cur.execute("""
                        INSERT INTO entity_aliases
                        (entity_id, alias, alias_norm, alias_type, alias_class,
                         is_auto_match, is_matchable, min_chars, match_case)
                        VALUES (%s, %s, %s, 'derived_acronym', 'org',
                                true, true, 2, 'upper_only')
                    """, (entity_id, acronym.upper(), acronym))
```

**Integration**: Run after concordance ingest, before extraction.

---

## Phase 3: Real `requires_context` Implementation

### 3.1 Mark Ambiguous Covernames with `requires_context`

**File**: `concordance/ingest_concordance_tab_aware.py`

**Location**: `ensure_alias()` function

**Changes**:
```python
# After filtering, check if covername is ambiguous
AMBIGUOUS_COVERNAMES = {
    "link", "achievement", "master", "group", "general", "president",
    "information", "foreign", "neighbour", "neighbors", "neighbours"
}

if alias_class == "covername" and filtered.lower() in AMBIGUOUS_COVERNAMES:
    requires_context = "codename_like"
else:
    requires_context = None
```

---

### 3.2 Implement Context Checks

**File**: `scripts/extract_entity_mentions.py`

**Location**: `find_matches_for_chunk()` function, policy checks section (around line 1323)

**Changes**:
```python
def check_codename_context(chunk_text: str, start_pos: int, end_pos: int, surface: str) -> bool:
    """
    Check if ambiguous codename has context signals:
    - Quoted/bracketed: "LINK", ['LINK'], [LINK], (LINK)
    - Covername markers nearby: cover name, codenamed, cryptonym, alias, aka
    - ALLCAPS
    """
    # Check if surface is quoted/bracketed
    context_window_start = max(0, start_pos - 5)
    context_window_end = min(len(chunk_text), end_pos + 5)
    context_window = chunk_text[context_window_start:context_window_end]
    
    # Check quotes/brackets
    if any(marker in context_window for marker in ['"', "'", '[', ']', '(', ')']):
        return True
    
    # Check ALLCAPS
    if surface.isupper() and len(surface) >= 3:
        return True
    
    # Check for codename markers (±60 chars)
    extended_start = max(0, start_pos - 60)
    extended_end = min(len(chunk_text), end_pos + 60)
    extended_context = chunk_text[extended_start:extended_end].lower()
    
    markers = [
        "cover name", "codenamed", "cryptonym", "alias", 
        "also known as", "aka", "code name"
    ]
    if any(marker in extended_context for marker in markers):
        return True
    
    return False

# In policy checks section:
if ai.requires_context == "codename_like":
    if not check_codename_context(chunk_text, s, e, surface):
        rejection_stats.setdefault("context_gate", {})
        key = f"{ai.original_alias} (requires_context={ai.requires_context})"
        rejection_stats["context_gate"][key] = rejection_stats["context_gate"].get(key, 0) + 1
        # Optionally enqueue instead of rejecting
        if CONTEXT_GATE_FAILURE_ENQUEUE:
            collision_queue.append({...})
        continue
```

---

## Phase 4: Frequency-Based Auto-Downgrade

### 4.1 Compute Document Frequency (DF) for Aliases

**New File**: `scripts/compute_alias_frequency.py`

**Logic**:
```python
def compute_alias_frequency(conn, collection_slug: str = None):
    """
    Compute document frequency (DF) for each alias_norm:
    - How many chunks contain this token/span
    - Store in alias_stats table
    """
    with conn.cursor() as cur:
        # Create alias_stats table if not exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS alias_stats (
                alias_norm TEXT PRIMARY KEY,
                df_chunks INTEGER,
                df_percent NUMERIC,
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)
        
        # Get total chunk count
        if collection_slug:
            cur.execute("""
                SELECT COUNT(*) FROM chunks c
                JOIN chunk_metadata cm ON cm.chunk_id = c.id
                JOIN documents d ON d.id = cm.document_id
                JOIN collections col ON col.id = d.collection_id
                WHERE col.slug = %s
            """, (collection_slug,))
        else:
            cur.execute("SELECT COUNT(*) FROM chunks")
        
        total_chunks = cur.fetchone()[0]
        
        # Compute DF for each alias_norm
        # Use text search to find occurrences
        cur.execute("""
            SELECT ea.alias_norm, COUNT(DISTINCT c.id) as chunk_count
            FROM entity_aliases ea
            CROSS JOIN chunks c
            WHERE c.text ILIKE '%' || ea.alias_norm || '%'
            GROUP BY ea.alias_norm
        """)
        
        for alias_norm, chunk_count in cur.fetchall():
            df_percent = (chunk_count / total_chunks) * 100 if total_chunks > 0 else 0
            
            cur.execute("""
                INSERT INTO alias_stats (alias_norm, df_chunks, df_percent)
                VALUES (%s, %s, %s)
                ON CONFLICT (alias_norm) 
                DO UPDATE SET df_chunks = EXCLUDED.df_chunks,
                             df_percent = EXCLUDED.df_percent,
                             updated_at = NOW()
            """, (alias_norm, chunk_count, df_percent))
```

---

### 4.2 Apply DF-Based Rules in `load_all_aliases`

**File**: `scripts/extract_entity_mentions.py`

**Location**: `load_all_aliases()` function

**Changes**:
```python
# After loading aliases, check DF stats
with conn.cursor() as cur:
    cur.execute("""
        SELECT alias_norm, df_percent 
        FROM alias_stats
        WHERE alias_norm = ANY(%s)
    """, (list(aliases_by_norm.keys()),))
    
    df_map = {row[0]: row[1] for row in cur.fetchall()}
    
    # Apply DF-based rules
    for alias_norm, alias_infos in aliases_by_norm.items():
        df_percent = df_map.get(alias_norm, 0)
        
        for ai in alias_infos:
            tokens = alias_norm.split()
            
            # Rule: if DF > 0.5% and single-token and not ALLCAPS-acronym
            if df_percent > 0.5 and len(tokens) == 1:
                if not (ai.alias_class == "covername" and ai.original_alias.isupper() and len(ai.original_alias) <= 6):
                    ai.is_auto_match = False
            
            # Rule: if DF > 2%, disable matching entirely (unless whitelisted)
            if df_percent > 2.0:
                # Whitelist: known valuable acronyms
                whitelist = {"kgb", "gru", "mgb", "nkvd", "ussr", "fbi", "cia"}
                if alias_norm.lower() not in whitelist:
                    ai.is_auto_match = False
                    # Could also set is_matchable=False here
```

---

## Phase 5: Refine Partial Matching

### 5.1 Update Partial Matching to Use Derived Surnames Only

**File**: `scripts/extract_entity_mentions.py`

**Location**: `build_partial_match_index()` function (already updated, but refine)

**Changes**:
```python
def build_partial_match_index(aliases_by_norm: Dict[str, List[AliasInfo]], min_token_len: int = 4) -> Dict[str, List[str]]:
    """
    Build partial match index ONLY for:
    - Derived surname aliases (alias_type='derived_last_name')
    - Derived acronym aliases (alias_type='derived_acronym')
    - Last token of person_full aliases (if >= 4 chars)
    """
    idx: DefaultDict[str, List[str]] = defaultdict(list)
    
    for alias_norm, alias_infos in aliases_by_norm.items():
        # Only index derived aliases or last tokens of person_full
        for ai in alias_infos:
            if ai.alias_type == "derived_last_name":
                # This is already a surname alias
                tokens = alias_norm.split()
                if len(tokens) == 1 and len(tokens[0]) >= min_token_len:
                    idx[tokens[0]].append(alias_norm)
            
            elif ai.alias_type == "derived_acronym":
                # Index acronyms
                if len(alias_norm) >= 2 and len(alias_norm) <= 6:
                    idx[alias_norm].append(alias_norm)
            
            elif ai.alias_class == "person_full" and ai.is_auto_match:
                # Index last token only
                tokens = alias_norm.split()
                if len(tokens) > 1:
                    surname = tokens[-1]
                    if len(surname) >= min_token_len and surname.lower() not in GENERIC_WORDS_TO_EXCLUDE:
                        idx[surname].append(alias_norm)
    
    return dict(idx)
```

---

## Implementation Order

1. **Week 1**: Phase 1 (Immediate Fixes)
   - 1.1: Fix alias_type-based `is_auto_match`
   - 1.2: Ban single/2-letter covernames
   - 1.3: Stop generic label entities

2. **Week 2**: Phase 2 (Derived Aliases)
   - 2.1: Add surname derivation script
   - 2.2: Add acronym derivation script
   - Test with small dataset

3. **Week 3**: Phase 3 (Context Gating)
   - 3.1: Mark ambiguous covernames
   - 3.2: Implement context checks
   - Test with known ambiguous cases

4. **Week 4**: Phase 4 (Frequency-Based)
   - 4.1: Create DF computation script
   - 4.2: Integrate DF rules into extraction
   - Monitor impact on precision/recall

5. **Week 5**: Phase 5 (Refine Partial)
   - 5.1: Update partial matching logic
   - End-to-end testing
   - Tune thresholds

---

## Testing Strategy

### Precision Metrics
- Count of junk matches (generic words, single letters, etc.)
- Measure before/after each phase

### Recall Metrics
- Count of legitimate entities that are now missed
- Use known entity list for validation

### Validation Dataset
- Create gold-standard set of ~100-200 entity mentions
- Measure precision/recall after each phase

---

## Rollback Plan

Each phase should be:
1. **Reversible**: Keep old logic commented out
2. **Configurable**: Add feature flags (e.g., `--enable-df-filtering`)
3. **Auditable**: Log all changes to `is_auto_match` decisions

---

## Notes

- **Whitelist Management**: Maintain a small whitelist of truly common-but-valuable tokens (KGB, GRU, USSR, etc.)
- **Manual Overrides**: Allow manual `is_auto_match` overrides in DB for edge cases
- **Monitoring**: Track rejection stats to identify new patterns that need rules

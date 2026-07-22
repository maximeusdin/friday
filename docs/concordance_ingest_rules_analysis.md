# Concordance Ingest Rules Analysis

## Current Rules in `ingest_concordance_tab_aware.py`

### 1. Entry Segmentation Rules

**Layout-based (preferred when pdfplumber available):**
- Uses x0 indentation: headword = line with `x0 ≤ (left_margin + indent_delta)`
- Guards against false positives:
  - Not "As/Translated" lines
  - Not footnote numbering (`^\s*\d+\.\s+`)
  - Not lowercase continuations
  - Not bare small integers
  - Not noise (page numbers, separators)
- Footnote mode: enters on dash separators, skips until new headword

**Regex-based (fallback):**
- Colon-start: `^[\"\"']?[A-Z0-9]\S.{0,220}:\s`
- Person-dot-start: `^[A-Z][A-Za-z\'\-]+,\s+[A-Z][A-Za-z\'\-]+...\.\s+\S`
- Period-start: `^[\"\"']?[^.\n]{1,220}\.\s+(?:Unidentified|Venona|Vassiliev|See|Cover|Translated)\b`
- Special patterns: "Undeciphered Name No.", "Source No."

### 2. Entity Type Classification (`classify_entity_type`)

**Rules (in order):**
1. **cover_name** if:
   - Headword contains "cover name"
   - Headword starts with "Undeciphered Name No."
   - Headword contains "Source No."
   - Body contains "cover name in venona/vassiliev"

2. **person** if:
   - Headword looks like "Lastname, Firstname" (contains comma)

3. **topic** if:
   - Headword/body contains "related subjects", "references in", "all of vassiliev", "devoted to"

4. **other** (default)

**Problem:** Too conservative. Many covernames (like "DOUGLAS", "JACK") don't have explicit markers, so they're classified as "other".

### 3. Alias Extraction Rules (`parse_entry_block`)

**Current aliases extracted (regardless of entity type):**

1. **Canonical** (line 816-817)
   - From headword: quoted string, or first synonym, or cleaned headword

2. **Head synonyms** (lines 819-825)
   - Split headword on "and"
   - Added as aliases if different from canonical

3. **Bracket variants** (lines 827-833)
   - Extract `[X]` tokens from headword
   - Added as aliases

4. **Quoted variants** (lines 835-841)
   - Additional quoted strings beyond first
   - Added as aliases

5. **Definitional expansion** (lines 843-856)
   - For acronyms (2-10 chars, all caps): extracts definition from body
   - Example: "AAF: Army Air Force, U.S." → alias "Army Air Force, U.S."
   - **PROBLEM:** If body starts with person name, this creates person name alias

6. **Work name** (lines 858-862)
   - Pattern: `work name "X"`

7. **Name spelled** (lines 864-869)
   - Pattern: `Name spelled X`

8. **Scoped labels** (line 925)
   - "As X:" or "Translated as X:" → alias "X"
   - **PROBLEM:** If entry is covername and X is person name, creates wrong alias

**Key Issue:** All aliases are extracted without filtering by entity type. Person names from body become aliases of covername entities.

### 4. Link Extraction Rules

**Current links extracted:**

1. **cover_name_of** (lines 872-885)
   - Pattern: `Cover name in Venona/Vassiliev: X`
   - Creates link: `X --cover_name_of--> entity_canonical`

2. **changed_to** (lines 887-897)
   - Pattern: `X was changed to Y`
   - Creates link: `X --changed_to--> Y`

3. **Referent inference** (lines 899-909)
   - For cover_name entries: infers person referent from body start
   - Creates link: `covername --cover_name_of--> person`
   - **GOOD:** This creates relationships, not aliases

**Problem:** Links are created, but person names are ALSO extracted as aliases (via definitional expansion, scoped labels, etc.), causing double encoding.

### 5. Person Referent Inference (`infer_referent_from_body_start`)

**Current logic:**
- Takes first sentence before period
- Filters out bad starts ("unidentified", "likely", etc.)
- Strips temporal qualifiers ("prior to", "after", etc.)
- Strips trailing parentheticals
- Returns first sentence as referent

**Problem:** Too simple. Only handles "X." pattern. Misses "X, who...", "X (person)", etc.

## Root Cause Analysis

### Why Person Names Become Aliases of Covernames

**Example entry:**
```
DOUGLAS: Joseph Katz. Venona decrypt...
```

**Current parsing:**
1. Entity type: classified as "other" (no explicit "cover name" marker)
2. Canonical: "DOUGLAS"
3. Definitional expansion (line 843-856):
   - Detects "DOUGLAS" is all-caps acronym (2-10 chars)
   - Extracts body first sentence: "Joseph Katz"
   - Creates alias: "Joseph Katz" on DOUGLAS entity
4. Referent inference (line 901):
   - Also creates link: `DOUGLAS --cover_name_of--> Joseph Katz`
5. **Result:** Both alias AND link created → alias causes collision

**The Fix:** Don't extract person names as aliases for covername entities. Only extract covername-like strings.

## Proposed Improvements

### Improvement 1: Entity-Type-Aware Alias Extraction

**Add helper functions:**

```python
def is_covername_like(text: str) -> bool:
    """Check if text looks like a covername (surface form)."""
    t = text.strip()
    # All caps, short (2-6 chars)
    if t.isupper() and 2 <= len(t) <= 6:
        return True
    # All caps with dots (U.S., F.B.I.)
    if re.match(r'^[A-Z](?:\.[A-Z])+\.?$', t):
        return True
    # All caps with hyphens (ANGLO-AMERICAN)
    if re.match(r'^[A-Z]+(?:-[A-Z]+)+$', t):
        return True
    return False

def is_person_name_like(text: str) -> bool:
    """Check if text looks like a person name."""
    t = text.strip()
    # Has comma (Lastname, Firstname)
    if ',' in t:
        return True
    # Title case, has space, 2-4 words
    words = t.split()
    if 2 <= len(words) <= 4:
        if all(w and w[0].isupper() for w in words):
            return True
    return False
```

**Modify alias extraction to filter by entity type:**

```python
# In parse_entry_block, after extracting all candidate aliases:

# Filter aliases based on entity type
filtered_aliases = []
for al in pe.aliases:
    if pe.entity_type == "cover_name":
        # For covernames: only covername-like aliases
        if is_covername_like(al.alias):
            filtered_aliases.append(al)
        # Skip person names (they become relationships, not aliases)
    elif pe.entity_type == "person":
        # For persons: only person-like aliases
        if is_person_name_like(al.alias) or al.alias_type == "canonical":
            filtered_aliases.append(al)
        # Skip covernames (they become relationships, not aliases)
    else:
        # For other/topic: keep all (less restrictive)
        filtered_aliases.append(al)

pe.aliases = filtered_aliases
```

### Improvement 2: Better Entity Type Classification

**Add heuristics:**

```python
def classify_entity_type(entry_key: str, body: str) -> str:
    # ... existing rules ...
    
    key_core = remove_trailing_descriptor_paren(entry_key).strip()
    
    # NEW: Short all-caps headword + person referent in body → covername
    if (key_core.isupper() and 2 <= len(key_core) <= 6):
        # Check if body mentions person name
        if re.search(r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)', body):
            # Check if it's not just a definition (e.g., "AAF: Army Air Force")
            if not re.search(r'^[A-Z]{2,6}\s*:\s*[A-Z]', body):
                return "cover_name"
    
    # NEW: "covername for X" or "codename for X"
    if re.search(r'\b(cover\s+name|codename)\s+for\s+([A-Z][a-z]+\s+[A-Z][a-z]+)', body, re.IGNORECASE):
        return "cover_name"
    
    # ... rest of existing rules ...
```

### Improvement 3: Fix Definitional Expansion

**Current code (lines 843-856) extracts definition without checking if it's person-like:**

```python
# Current (PROBLEMATIC):
if pe.entity_canonical.isupper() and 2 <= len(pe.entity_canonical) <= 10:
    defn = body_first.split(".", 1)[0].strip()
    if defn and not defn.lower().startswith(("cover name", "see ", "as ")):
        pe.aliases.append(ParsedAlias(alias=defn2, alias_type="definition"))
```

**Fix: Only extract if it matches entity type:**

```python
# Fixed:
if pe.entity_canonical.isupper() and 2 <= len(pe.entity_canonical) <= 10:
    defn = body_first.split(".", 1)[0].strip()
    if defn and not defn.lower().startswith(("cover name", "see ", "as ")):
        defn2 = defn.strip().strip(";").strip()
        if len(defn2.split()) >= 2:
            # Only add if it matches entity type
            if pe.entity_type == "cover_name" and is_covername_like(defn2):
                pe.aliases.append(ParsedAlias(alias=defn2, alias_type="definition"))
            elif pe.entity_type == "person" and is_person_name_like(defn2):
                pe.aliases.append(ParsedAlias(alias=defn2, alias_type="definition"))
            elif pe.entity_type not in ("cover_name", "person"):
                # For other/topic: allow (less restrictive)
                pe.aliases.append(ParsedAlias(alias=defn2, alias_type="definition"))
```

### Improvement 4: Fix Scoped Labels ("As X:")

**Current code (line 925) always creates alias:**

```python
# Current (PROBLEMATIC):
if label:
    pe.aliases.append(ParsedAlias(alias=label, alias_type="scoped_label"))
```

**Fix: Create relationship if appropriate:**

```python
# Fixed:
if label:
    if pe.entity_type == "cover_name" and is_person_name_like(label):
        # Covername entry with person label → relationship, not alias
        pe.links.append(ParsedLink(
            link_type="covername_of",
            from_name=pe.entity_canonical,
            to_name=label,
            confidence="certain",
            notes=f"From '{m.group(1)} {label}' block"
        ))
    elif pe.entity_type == "person" and is_covername_like(label):
        # Person entry with covername label → relationship, not alias
        pe.links.append(ParsedLink(
            link_type="covername_of",
            from_name=label,
            to_name=pe.entity_canonical,
            confidence="certain",
            notes=f"From '{m.group(1)} {label}' block"
        ))
    else:
        # Same type or ambiguous → alias (surface form)
        pe.aliases.append(ParsedAlias(alias=label, alias_type="scoped_label"))
```

### Improvement 5: Better Person Referent Inference

**Current is too simple. Improve:**

```python
def infer_referent_from_body_start(body: str) -> Optional[str]:
    """Improved person referent inference with multiple patterns."""
    b = body.strip()
    if not b:
        return None
    
    b_norm = _normalize_quotes(b)
    
    # Try multiple patterns (in order of specificity)
    patterns = [
        # "Joseph Katz," or "Joseph Katz."
        r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s*[,\.]',
        # "Joseph Katz ("
        r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s+\(',
        # "Joseph Katz was"
        r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s+was',
        # "Joseph Katz who"
        r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s+who',
        # "Joseph Katz a "
        r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s+a\s+',
        # Fallback: first sentence before period
        r'^([^\.]+)\.',
    ]
    
    for pattern in patterns:
        m = re.match(pattern, b_norm)
        if m:
            referent = m.group(1).strip()
            # Validate: looks like person name
            if is_person_name_like(referent):
                # Clean up
                referent = re.sub(r'\s*\([^)]*\)\s*$', '', referent)  # Remove trailing parens
                referent = re.sub(r'\s*,\s*[^,]+$', '', referent)  # Remove trailing comma phrase
                return referent.strip()
    
    return None
```

## Implementation Plan

### Phase 1: Add Helper Functions
1. Add `is_covername_like()` and `is_person_name_like()` helpers
2. Add improved `infer_referent_from_body_start()` with multiple patterns

### Phase 2: Entity-Type-Aware Alias Filtering
1. Filter aliases after extraction based on entity type
2. Covernames: only covername-like aliases
3. Persons: only person-like aliases

### Phase 3: Fix Specific Extraction Points
1. Fix definitional expansion (check entity type)
2. Fix scoped labels (create relationships when appropriate)
3. Improve entity type classification (add heuristics)

### Phase 4: Test and Validate
1. Re-run ingest on sample entries
2. Check that person names are NOT aliases of covernames
3. Check that relationships are created correctly
4. Verify entity type classification improved

## Expected Impact

**Before:**
- Entry: `DOUGLAS: Joseph Katz. Venona...`
- Creates: covername entity "DOUGLAS" with alias "Joseph Katz"
- Result: "joseph katz" → 18 entities (collision)

**After:**
- Entry: `DOUGLAS: Joseph Katz. Venona...`
- Creates: covername entity "DOUGLAS" with aliases ["DOUGLAS"] (only covername-like)
- Creates: relationship `DOUGLAS --covername_of--> Joseph Katz`
- Result: "joseph katz" → 1 entity (person), "DOUGLAS" → 1 entity (covername)

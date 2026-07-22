# Concordance Ingest Improvements Plan

## Current Rules Analysis

### 1. Entry Segmentation Rules

**Layout-based (preferred):**
- Uses x0 indentation to detect headwords
- Headword = line with x0 ≤ (left_margin + indent_delta)
- Guards: not "As/Translated", not footnote numbering, not lowercase, not bare ints
- Handles footnote mode: enters on dash separators, skips until new headword

**Regex-based (fallback):**
- Colon-start: `^[\"\"']?[A-Z0-9]\S.{0,220}:\s`
- Person-dot-start: `^[A-Z][A-Za-z\'\-]+,\s+[A-Z][A-Za-z\'\-]+...\.\s+\S`
- Period-start: `^[\"\"']?[^.\n]{1,220}\.\s+(?:Unidentified|Venona|Vassiliev|See|Cover|Translated)\b`
- Undeciphered/Source No. patterns

### 2. Entity Type Classification Rules

**Current logic (`classify_entity_type`):**
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

**Problem:** Too conservative. Many covernames don't have explicit "cover name" marker.

### 3. Alias Extraction Rules

**Current aliases extracted:**
1. **Canonical** from headword (quoted string, or first synonym, or cleaned headword)
2. **Head synonyms** (split on "and")
3. **Bracket variants** `[X]`
4. **Quoted strings** in headword
5. **Definitional expansion** for acronyms (e.g., "AAF: Army Air Force")
6. **Work name** patterns: `work name "X"`
7. **Name spelled** patterns: `Name spelled X`
8. **Scoped labels**: "As X:", "Translated as X:"

**Problem:** Person names from body are being attached as aliases to covername entities.

### 4. Link Extraction Rules

**Current links extracted:**
1. **cover_name_of**: "Cover name in Venona/Vassiliev: X" → creates link from covername to person
2. **changed_to**: "X was changed to Y" → creates link
3. **Referent inference**: For cover_name entries, infers referent from body start

**Problem:** Links are being created, but person names are ALSO being attached as aliases, causing collisions.

## Key Problems Identified

### Problem 1: Person Names as Aliases of Covernames

**Current behavior:**
- Entry: `DOUGLAS: Joseph Katz. Venona decrypt...`
- Creates: covername entity "DOUGLAS" with alias "Joseph Katz"
- **Result:** "joseph katz" maps to 18 covername entities → collision

**Root cause:** The script extracts "Joseph Katz" from body and attaches it as an alias to the covername entity.

### Problem 2: Entity Type Misclassification

**Current behavior:**
- Entry: `JACK: ...` (no explicit "cover name" marker)
- Classified as: `other` or `topic`
- **Result:** Wrong entity type, wrong policy defaults

**Root cause:** Classification is too conservative, relies on explicit markers.

### Problem 3: Links vs Aliases Confusion

**Current behavior:**
- Entry: `DOUGLAS: Joseph Katz. Cover name in Venona: "DOUGLAS".`
- Creates: link `DOUGLAS --cover_name_of--> Joseph Katz`
- ALSO creates: alias "Joseph Katz" on DOUGLAS entity
- **Result:** Double encoding (link + alias), alias causes collision

**Root cause:** Script doesn't distinguish between "what appears in text" (alias) vs "who this is" (relationship).

## Proposed Improvements

### Improvement 1: Better Entity Type Classification

**Add heuristics for covername detection:**

```python
def classify_entity_type(entry_key: str, body: str) -> str:
    # ... existing rules ...
    
    # NEW: Covername heuristics
    # - Short all-caps headword (2-6 chars) + body mentions person name
    # - Headword is all-caps acronym-like + body has person referent
    # - Body pattern: "covername for X" or "codename for X"
    
    key_core = remove_trailing_descriptor_paren(entry_key).strip()
    
    # Heuristic: Short all-caps + person referent in body
    if (key_core.isupper() and 2 <= len(key_core) <= 6 and 
        re.search(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+', body)):
        # Check if body mentions person name pattern
        if re.search(r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)', body):
            return "cover_name"
    
    # Heuristic: "covername for X" or "codename for X"
    if re.search(r'\b(cover\s+name|codename)\s+for\s+([A-Z][a-z]+\s+[A-Z][a-z]+)', body, re.IGNORECASE):
        return "cover_name"
    
    # ... rest of existing rules ...
```

### Improvement 2: Separate Surface Forms from Identity Claims

**Key principle:** Only extract aliases that actually appear in text as surface forms.

**For covername entries:**
- **Aliases:** Only covername-like strings (DOUGLAS, DUGLAS, DZHEK, etc.)
- **NOT aliases:** Person names mentioned in body (these become relationships)

**For person entries:**
- **Aliases:** Person name variants (Joseph Katz, J. Katz, Katz, Joseph, etc.)
- **NOT aliases:** Covernames mentioned in body (these become relationships)

**Implementation:**

```python
def extract_aliases_for_entity_type(pe: ParsedEntry) -> List[ParsedAlias]:
    """Extract aliases based on entity type - only surface forms."""
    aliases = []
    
    # Always include canonical
    if pe.entity_canonical:
        aliases.append(ParsedAlias(alias=pe.entity_canonical, alias_type="canonical"))
    
    if pe.entity_type == "cover_name":
        # For covernames: only covername-like aliases
        # - Headword variants (if covername-like)
        # - Bracket variants (if covername-like)
        # - Quoted strings (if covername-like)
        # - NOT person names from body
        
        for syn in extract_head_synonyms(pe.entry_key):
            if is_covername_like(syn):  # All caps, short, etc.
                aliases.append(ParsedAlias(alias=syn, alias_type="head_syn"))
        
        for bt in extract_bracket_tokens(pe.entry_key):
            if is_covername_like(bt):
                aliases.append(ParsedAlias(alias=bt, alias_type="bracket_variant"))
    
    elif pe.entity_type == "person":
        # For persons: only person name variants
        # - Headword variants
        # - Bracket variants (if person-like)
        # - "Name spelled X"
        # - Work name
        # - NOT covernames mentioned in body
        
        for syn in extract_head_synonyms(pe.entry_key):
            if is_person_name_like(syn):
                aliases.append(ParsedAlias(alias=syn, alias_type="head_syn"))
        
        # ... existing person alias extraction ...
    
    return aliases
```

### Improvement 3: Extract Relationships Instead of Aliases

**When parsing covername entries:**
- Extract person referent from body
- Create relationship: `covername --covername_of--> person`
- Do NOT attach person name as alias

**When parsing person entries:**
- Extract covernames mentioned in body
- Create relationships: `covername --covername_of--> person`
- Do NOT attach covername as alias to person

**Implementation:**

```python
def extract_relationships(pe: ParsedEntry, body: str) -> List[ParsedLink]:
    """Extract relationships (not aliases) from entry body."""
    links = []
    
    if pe.entity_type == "cover_name":
        # Extract person referent from body
        referent = infer_referent_from_body_start(body)
        if referent and is_person_name_like(referent):
            links.append(ParsedLink(
                link_type="covername_of",
                from_name=pe.entity_canonical,
                to_name=referent,
                confidence="certain",
                notes="Inferred from entry body"
            ))
    
    elif pe.entity_type == "person":
        # Extract covernames mentioned in body
        # Pattern: "Cover name: X" or "Codename: X" or "Also known as X"
        for m in re.finditer(
            r'\b(cover\s+name|codename|also\s+known\s+as)\s*:\s*["\']?([A-Z0-9][A-Z0-9\s]{1,10})["\']?',
            body,
            re.IGNORECASE
        ):
            covername = m.group(2).strip()
            if is_covername_like(covername):
                links.append(ParsedLink(
                    link_type="covername_of",
                    from_name=covername,
                    to_name=pe.entity_canonical,
                    confidence="certain",
                    notes=f"From entry: {m.group(1)}"
                ))
    
    return links
```

### Improvement 4: Better Person Referent Inference

**Current `infer_referent_from_body_start` is too conservative.**

**Improvements:**
- Handle more patterns: "X (person)", "X, who...", "X, a...", "X was..."
- Better temporal qualifier handling
- Handle parentheticals better

```python
def infer_referent_from_body_start(body: str) -> Optional[str]:
    """Improved person referent inference."""
    b = body.strip()
    if not b:
        return None
    
    # Try multiple patterns
    patterns = [
        r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s*[,\.]',  # "Joseph Katz,"
        r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s+\(',     # "Joseph Katz ("
        r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s+was',    # "Joseph Katz was"
        r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s+who',     # "Joseph Katz who"
        r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s+a\s+',   # "Joseph Katz a "
    ]
    
    for pattern in patterns:
        m = re.match(pattern, b)
        if m:
            referent = m.group(1).strip()
            # Validate: looks like person name (has space, title case)
            if ' ' in referent and referent[0].isupper():
                # Strip trailing descriptors
                referent = re.sub(r'\s*\([^)]*\)\s*$', '', referent)
                referent = re.sub(r'\s*,\s*[^,]+$', '', referent)  # Remove trailing comma phrase
                return referent.strip()
    
    return None
```

### Improvement 5: Handle "As X:" Blocks Correctly

**Current behavior:**
- "As X:" creates alias "X" and scoped citations
- **Problem:** If X is a person name and entry is covername, this creates wrong alias

**Fix:**
- For covername entries: "As X:" where X is person-like → relationship, not alias
- For person entries: "As X:" where X is covername-like → relationship, not alias

```python
def handle_scoped_blocks(pe: ParsedEntry, body: str):
    """Handle 'As X:' blocks - create relationships, not aliases."""
    scoped_pat = re.compile(r'\b(As|Translated as)\s+["\']?([^"\':]+)["\']?\s*:', flags=re.IGNORECASE)
    
    for m in scoped_pat.finditer(body):
        label = m.group(2).strip()
        
        if pe.entity_type == "cover_name" and is_person_name_like(label):
            # Covername entry with person label → relationship
            pe.links.append(ParsedLink(
                link_type="covername_of",
                from_name=pe.entity_canonical,
                to_name=label,
                confidence="certain",
                notes=f"From '{m.group(1)} {label}' block"
            ))
        elif pe.entity_type == "person" and is_covername_like(label):
            # Person entry with covername label → relationship
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

### Improvement 6: Better Covername Detection

**Add more patterns:**

```python
def is_covername_like(text: str) -> bool:
    """Check if text looks like a covername."""
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
        if all(w[0].isupper() for w in words if w):
            return True
    
    return False
```

## Implementation Priority

### Phase 1: Critical Fixes (Do First)

1. **Separate alias extraction by entity type**
   - Covernames: only covername-like aliases
   - Persons: only person-like aliases
   - This prevents person names from being aliases of covernames

2. **Better entity type classification**
   - Add heuristics for covername detection
   - Reduces misclassification

3. **Extract relationships, not aliases**
   - Person referents in covername entries → relationships
   - Covernames in person entries → relationships

### Phase 2: Refinements

4. **Better person referent inference**
   - Handle more patterns
   - More robust extraction

5. **Handle "As X:" blocks correctly**
   - Create relationships when appropriate
   - Not just aliases

6. **Better covername/person detection helpers**
   - `is_covername_like()` and `is_person_name_like()`
   - Used throughout parsing

## Expected Impact

**After improvements:**
- Covername entities: aliases = only covername-like strings (DOUGLAS, JACK, etc.)
- Person entities: aliases = only person name variants (Joseph Katz, etc.)
- Relationships: covername --covername_of--> person (stored in entity_relationships)
- **Result:** "joseph katz" maps to 1 entity (person), not 18 covernames

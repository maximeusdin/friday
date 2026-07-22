# How `extract_entity_mentions.py` Works Now

## Overview

The script extracts entity mentions from text chunks using a **three-tier matching system** (exact, partial, fuzzy) with **systematic collision resolution** and **citation-based disambiguation**.

## Key Architecture Changes

### 1. **Simplified Collision Resolution Pipeline**

The collision resolution now follows a **clear, prioritized sequence**:

```
Collision Detected
    ↓
Step 1: Duplicate Detection (NEW - highest priority)
    ↓ (if duplicates found)
    → Try citation-based resolution to pick best duplicate
    → Fallback to first auto_match candidate
    → Fallback to first candidate
    ↓
Step 2: Citation-Based Resolution (if not already resolved)
    ↓ (if citation match found)
    → Use entity with best citation match
    ↓
Step 3: Dominance Rules
    ↓ (if dominant candidate found)
    → Use dominant candidate (e.g., covername > generic_word)
    ↓
Step 4: Harmless Check (if still unresolved)
    ↓ (if harmless)
    → Skip (don't enqueue)
    ↓ (if high-value)
    → Enqueue for review
```

### 2. **Duplicate Detection (Lines 367-386)**

**New Functions:**
- `get_canonical_names(cur, entity_ids)`: Batch-loads canonical names for entities
- `are_candidates_duplicates(alias_infos, canonical_names, cur)`: Checks if all candidates have the same normalized canonical name

**Why it matters:**
- Detects when multiple entity records refer to the same real-world entity
- Prevents marking duplicate collisions as "harmless"
- Allows picking the best duplicate using citations or other heuristics

### 3. **Systematic `is_collision_harmless` Logic (Lines 672-701)**

**Old approach:** Arbitrary thresholds (e.g., "if > 5 candidates, mark harmless")

**New approach:** Data-driven hierarchy:

1. **Eligibility check first**: If alias is not eligible (stopword, etc.), return `True` (harmless)
2. **Duplicate check**: If all candidates are duplicates, return `False` (never harmless - data quality issue)
3. **Entity value hierarchy**:
   - `covername` or `person_full`: Never harmless (`False`)
   - `generic_word`: Always harmless (`True`)
   - `person_given` (single token): Only harmless if `> 10` **different** entities (not duplicates)
4. **Default**: Try to resolve (`False`)

### 4. **Policy Checks After Resolution (Lines 1060-1114)**

**Key change:** Policy checks now happen **after** collision resolution, not before.

**Flow:**
1. Resolve collision (duplicates → citations → dominance → harmless)
2. **Then** check policy:
   - `is_auto_match` (line 1080)
   - Purely numeric (line 1086)
   - `min_chars` (lines 1092-1104)
   - Case matching (line 1111)

**Why:** This ensures we track "collision resolved but blocked by policy" separately from "no collision but blocked by policy".

### 5. **Removed `collision_resolved_policy_blocked` Tracking**

**Observation:** The current script doesn't track `collision_resolved_policy_blocked` in the statistics, but the CSV shows it. This suggests the tracking was removed or simplified.

**Current behavior:** If a collision is resolved but then blocked by policy, it's just counted as `rejected,policy` (not separately tracked).

## Processing Flow

### Phase 1: Alias Loading (Lines 708-862)

1. **Safe column detection**: Uses `information_schema.columns` (no aborted transactions)
2. **Load all matchable aliases** from `entity_aliases`
3. **Apply policy rules**:
   - `generic_word`: Disable auto-match unless it's a short ALLCAPS codeword
   - `person_given` (single token): Disable auto-match unless `allow_ambiguous_person_token=true`
   - `covername`: Default to `upper_only` case matching
4. **Override for "unidentified" entities**: Re-enable auto-match for entities with "unidentified" aliases

### Phase 2: Chunk Processing (Lines 869-1219)

For each chunk:

#### Tier 1: Exact Matching (Lines 903-1122)

1. **Tokenize** chunk text
2. **N-gram matching** (1-5 tokens, longest-first)
3. **For each n-gram**:
   - Check if normalized form exists in alias set
   - Get all candidate entities
   - **Eligibility check**: Stopwords, single letters, etc.
   - **Collision resolution** (if multiple candidates):
     - Duplicate detection → Citation → Dominance → Harmless
   - **Policy checks** (after resolution):
     - `is_auto_match`, `min_chars`, case matching
   - **Insert mention** if all checks pass

#### Tier 2: Partial Matching (Lines 1127-1134, if enabled)

- Match individual tokens ≥ 4 chars against alias tokens
- Lower confidence (0.5-0.7)
- Skip spans already matched exactly

#### Tier 3: Fuzzy Matching (Lines 1136-1137, if enabled)

- Levenshtein distance (max 2 edits)
- Minimum length 4 chars
- Confidence based on edit distance (0.8 for 1 edit, 0.6 for 2 edits)

### Phase 3: Batch Insertion (Lines 1226-1399)

1. **Preload caches**:
   - `chunk_metadata_cache`: Page IDs, PDF pages, document IDs
   - `document_cache`: Document name → ID mapping
   - `entity_citations_cache`: All citations for all entities

2. **Process chunks in batches** (default: 100)

3. **Idempotent insertion**:
   - Check existing mentions (temp table)
   - Insert only new mentions
   - Uses `execute_values` for bulk inserts

## Statistics Tracking

### Categories:

1. **`auto_matched`**: Successfully inserted mentions
2. **`rejected`**:
   - `policy`: Blocked by `is_auto_match`, `min_chars`, etc.
   - `not_eligible`: Stopwords, single letters, etc.
   - `case_mismatch`: Case policy enforcement
3. **`unresolved`**:
   - `collision_high_value_enqueued`: High-value collisions enqueued for review
   - `collision_high_value_too_many`: Too many candidates or case mismatch
   - `collision_dominance_none`: No dominant candidate found
   - `collision_harmless`: Too common/ambiguous (skipped)
4. **`auto_resolved`**: Collisions automatically resolved (duplicates, citations, dominance)

## Key Improvements

1. **No aborted transactions**: Uses `information_schema` instead of failing SELECTs
2. **Systematic collision resolution**: Duplicate detection first, then citations, then dominance
3. **Data-driven harmless detection**: Based on entity value, not arbitrary thresholds
4. **Batch caching**: Preloads all citations and metadata to minimize DB queries
5. **Clear separation**: Policy checks after resolution, not before

## Example: "yakubovich" Processing

1. **Text match**: "yakubovich" appears in chunk
2. **Collision**: Matches 2 entities (45240, 56335)
3. **Duplicate detection**: Both have canonical name "Yakubovich" → **Duplicates identified**
4. **Resolution**: Picks one entity (via citation or first candidate)
5. **Policy check**: `is_auto_match=False` (line 830: person_given single token)
6. **Result**: 
   - Counted in `auto_resolved,yakubovich (duplicate_resolved),2` ✅
   - Counted in `rejected,policy,Yakubovich (auto_match disabled),2` ❌

**The collision is resolved correctly, but the entity is blocked by policy.**

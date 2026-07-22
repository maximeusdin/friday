# How "yakubovich" is Handled - Explanation

## Summary

"yakubovich" appears in **3 different categories** in the match summary, representing different stages of the processing pipeline:

1. **`rejected,policy`** (Line 8): 2 occurrences - "Yakubovich (auto_match disabled)"
2. **`rejected,collision_resolved_policy_blocked`** (Line 47): 2 occurrences - "yakubovich (is_auto_match=false)"
3. **`auto_resolved,total`** (Line 153): 2 occurrences - "yakubovich (duplicate_resolved)"

## The Processing Flow

### Step 1: Text Matching
"yakubovich" appears **2 times** in the chunks being processed.

### Step 2: Alias Loading & Policy Application
When aliases are loaded, the system applies policy rules:

**Line 875-883 in extract_entity_mentions.py**:
```python
# Constrain single-token person names aggressively
if (entity_type == 'person' 
    and alias_class == 'person_given' 
    and len(tokens) == 1
    and not allow_ambiguous_person_token):
    # Default to is_auto_match=false for single-token person given names
    is_auto_match = False
```

**What happens to "yakubovich"**:
- It's a single token: ✅
- Entity type is 'person': ✅
- Alias class is inferred as 'person_given' (single token person name): ✅
- `allow_ambiguous_person_token` is False: ✅
- **Result**: `is_auto_match` is **forced to False** (even if DB has it as True)

### Step 3: Collision Detection
Each occurrence matches **2 candidate entities** (45240 and 56335), both with canonical name "Yakubovich" → **Collision detected**

### Step 4: Duplicate Detection (NEW)
The system detects that both candidates have the same canonical name → **Duplicates identified**

### Step 5: Collision Resolution
The system resolves the collision by picking one entity (using duplicate resolution logic):
- Tries citation-based resolution first
- Falls back to picking first `auto_match` candidate
- If no `auto_match`, picks first candidate

**Result**: Collision is resolved → Counted in `auto_resolved,total,yakubovich (duplicate_resolved),2`

### Step 6: Policy Check (After Resolution)
After resolving the collision, the system checks if the resolved entity can be auto-matched:
- **Problem**: `is_auto_match` was forced to False in Step 2 (line 883)
- **Result**: The resolved entity is blocked by policy → Counted in `rejected,collision_resolved_policy_blocked,yakubovich (is_auto_match=false),2`

### Step 7: Direct Policy Rejection (Alternative Path)
In some cases, "Yakubovich" might match entities directly without a collision (if only one entity matches in that chunk), but `is_auto_match` is still False:
- **Result**: Direct policy rejection → Counted in `rejected,policy,Yakubovich (auto_match disabled),2`

## Why It Appears in Multiple Categories

The same 2 occurrences are being counted at different stages:

1. **`auto_resolved`**: The collision resolution **succeeded** (duplicates detected and resolved) ✅
2. **`collision_resolved_policy_blocked`**: But the resolved entity **failed** the policy check (`is_auto_match=false`) ❌
3. **`rejected,policy`**: Some occurrences might also be rejected directly (without collision) ❌

## The Root Cause

**Line 883 forces `is_auto_match=False` for single-token person given names**

This is a conservative policy to prevent false positives from common names like "john", "smith", etc. However, "yakubovich" is a legitimate, specific entity that should be matched.

The policy logic:
- Single token + person + person_given + `allow_ambiguous_person_token=False` → `is_auto_match=False`

## Solutions

### Option 1: Set `allow_ambiguous_person_token=true` (Recommended)
Update the database to set `allow_ambiguous_person_token=true` for the "Yakubovich" alias:
```sql
UPDATE entity_aliases 
SET allow_ambiguous_person_token = true 
WHERE alias_norm = 'yakubovich';
```

### Option 2: Change Alias Class
If "Yakubovich" is a last name (not a given name), change the alias_class:
```sql
UPDATE entity_aliases 
SET alias_class = 'person_full'  -- or NULL
WHERE alias_norm = 'yakubovich';
```

### Option 3: Merge Duplicate Entities
Merge entities 45240 and 56335 into one, then set `allow_ambiguous_person_token=true` on the merged entity.

## Statistics Interpretation

- **`auto_resolved,yakubovich (duplicate_resolved),2`**: ✅ Good - collision resolution is working
- **`rejected,collision_resolved_policy_blocked,yakubovich (is_auto_match=false),2`**: ⚠️ Policy issue - resolved entity can't be auto-matched due to line 883
- **`rejected,policy,Yakubovich (auto_match disabled),2`**: ⚠️ Policy issue - same policy rule blocking it

## The Issue

The system is working correctly:
1. ✅ Duplicate detection works (identifies 45240 and 56335 as duplicates)
2. ✅ Collision resolution works (picks one entity)
3. ❌ But the policy at line 883 disables auto-match for single-token person given names

This is a **policy issue**, not a collision resolution issue. The collision is being resolved correctly, but the resolved entity is blocked by the conservative policy for person given names.

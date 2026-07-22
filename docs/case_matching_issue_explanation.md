# Case Matching Issue: Why "yakubovich" Works But "Yakubovich" Doesn't

## The Problem

In the match summary CSV:
- **"yakubovich"** (lowercase): `auto_resolved,total,yakubovich (duplicate_resolved),2` ✅
- **"Yakubovich"** (titlecase): `rejected,policy,Yakubovich (auto_match disabled),2` ❌

Same entity, different capitalization, different outcomes.

## Root Cause

The issue is in **two places** where case matching is checked:

### 1. Dominance Rule Filtering (Lines 607-609)

```python
if ai.alias_class in ("covername", "person_given"):
    if not check_case_match(surface, ai):
        continue  # Filter out this candidate
```

For `person_given` aliases (which "yakubovich" likely is), the case matching is **strictly enforced** during dominance rule filtering. If the surface text doesn't match the alias's case requirements, the candidate is filtered out.

### 2. Final Policy Check (Line 1111)

```python
if not check_case_match(surface, ai):
    rejection_stats.setdefault("case_mismatch", {})
    rejection_stats["case_mismatch"][ai.original_alias] = ...
    continue
```

After collision resolution, there's another case matching check. If it fails, the mention is rejected.

## Why It Happens

Both "yakubovich" and "Yakubovich" normalize to the same `alias_norm` ("yakubovich"), so they:
1. Find the same candidates
2. Go through the same collision resolution
3. **But then** the case matching check differs:

**Scenario A: Alias stored as "yakubovich" (lowercase) with `match_case = 'case_sensitive'`**
- Surface "yakubovich" → `check_case_match("yakubovich", ai)` → `"yakubovich" == "yakubovich"` → ✅ PASS
- Surface "Yakubovich" → `check_case_match("Yakubovich", ai)` → `"Yakubovich" != "yakubovich"` → ❌ FAIL

**Scenario B: Alias stored as "Yakubovich" (titlecase) with `match_case = 'titlecase_only'`**
- Surface "yakubovich" → `check_case_match("yakubovich", ai)` → First char is lowercase → ❌ FAIL
- Surface "Yakubovich" → `check_case_match("Yakubovich", ai)` → First uppercase, rest lowercase → ✅ PASS

But the CSV shows the opposite (lowercase passes, titlecase fails), so **Scenario A is more likely**.

## The Fix

The issue is that `person_given` aliases are being treated too strictly. For person names, we should be more lenient with case matching, especially if the alias has `match_case = 'any'`.

**Option 1: Make `person_given` case matching more lenient**

If `match_case = 'any'`, don't enforce strict case matching for `person_given`:

```python
if ai.alias_class in ("covername", "person_given"):
    # For person_given with match_case='any', be lenient
    if ai.alias_class == "person_given" and ai.match_case == "any":
        pass  # Allow through
    elif not check_case_match(surface, ai):
        continue
```

**Option 2: Normalize case before comparison for `person_given`**

For `person_given` aliases, compare normalized forms:

```python
if ai.alias_class in ("covername", "person_given"):
    if ai.alias_class == "person_given" and ai.match_case == "any":
        # For person names with 'any' case, compare normalized
        if surface.lower() != ai.original_alias.lower():
            continue
    elif not check_case_match(surface, ai):
        continue
```

**Option 3: Update database aliases**

Set `match_case = 'any'` for person name aliases that should match regardless of capitalization.

## Recommended Solution

**Option 1** is the cleanest - if an alias has `match_case = 'any'`, it should match regardless of case, even for `person_given` aliases. The current code is being too strict.

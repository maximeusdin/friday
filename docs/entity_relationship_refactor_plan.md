# Entity Relationship Refactor Plan

## Problem Statement

### Core Problem

**Many `cover_name` entities contain person-name aliases, causing high-collision surface strings that should instead resolve to a person entity.**

The exact-alias matcher needs `alias_norm → entity_id` to be close to functional (ideally 1:1 or 1:few) if you want deterministic high-precision emission. When person names are attached as aliases to covername entities, this breaks down.

### Current Problem Pattern

**Before (with current data model):**

- Text contains: "Joseph Katz met with ..."
- Alias-exact matcher sees `alias_norm = "joseph katz"`
- Returns 18 candidate entities (all covername entities that have "Joseph Katz" as alias)
- Unresolved collision (no deterministic resolution)
- **Result: No mention emitted** (or enqueued for review)

**The Data Modeling Choice:**
- Having "Joseph Katz" attached to 18 covername entities is a valid data modeling choice
- It encodes the fact: "Joseph Katz is the real name for these 18 covernames"
- **But it breaks surface-form matching** because aliases are meant to match what appears in text

### Root Cause

**Aliases are doing double-duty:**
1. Surface-form matching (what appears in text) ✓ Correct use
2. Identity claims (who this entity is) ✗ Should be relationships, not aliases

## Solution: Separate Surface Matching from Graph Knowledge

### Core Principle

**Aliases = only what appears in text as surface forms**
**Relationships = explicit, auditable edges between entities**

### After Refactor

#### Surface-Form Matching (Aliases) - Unchanged Concept

**Aliases remain "surface forms" for both kinds of entities:**

**Person entity "Joseph Katz":**
- Aliases: "Joseph Katz", "J. Katz", "Katz, Joseph", OCR variants, etc.
- These are strings that actually appear in text referring to this person

**Covername entity "DOUGLAS":**
- Aliases: "DOUGLAS", "DUGLAS", "DZHEK", "JACK", "IKS", etc.
- These are the covername tokens that appear in text as codenames

**Key point:** Aliases are still surface forms. We're not changing what aliases are—we're removing identity claims from the alias table.

#### Graph Knowledge (Relationships)

**New relationship table:**
```
DOUGLAS --covername_of--> Joseph Katz
JACK --covername_of--> Joseph Katz
IKS --covername_of--> Joseph Katz
```

## What This Fix Does in Practice

### Before (Current State)

**Text contains "Joseph Katz":**
- `alias_norm = "joseph katz"`
- 18 candidates (all covername entities)
- Unresolved collision
- **No mention emitted** (or enqueued)

### After (With Relationships)

**Text contains "Joseph Katz":**
- `alias_norm = "joseph katz"`
- 1 candidate (the person entity)
- **Mention emitted deterministically** ✓

**Text contains "DOUGLAS":**
- Matches the covername entity DOUGLAS
- Mention emitted
- UI/downstream can show: `DOUGLAS --covername_of→ Joseph Katz`

**Net Effect:**
- Preserve "many covernames per person" (via relationships)
- Make matching clean (aliases map to correct entities)
- Deterministic high-precision emission

## Benefits

### 1. Collisions Drop Dramatically

- "Joseph Katz" → maps to 1 entity (the person)
- "DOUGLAS" → maps to 1 entity (the covername)
- Eliminates pathological "one alias_norm → 18 entities" pattern
- Alias-exact extractor becomes high precision and mostly deterministic
- **Enables deterministic mention emission**

### 2. Preserves Important Facts

- Still captures "one person may have multiple cover names"
- Just stores it in the right place (relationships, not aliases)
- More auditable: relationships can have provenance, confidence, sources

### 3. UI/Research Experience Improves

**Before:**
- Highlight "Joseph Katz" → system can't tell which of 18 covername entities you meant

**After:**
- Highlight "DOUGLAS" in text
- UI shows: "DOUGLAS (covername) → linked to Joseph Katz (person) [evidence]"
- Clicking Joseph Katz shows all related covernames and mentions
- Can navigate bidirectionally: person → covernames, covername → person

### 4. Cleaner NER/Adjudication

- NER proposes "Joseph Katz" → lands on person entity (no collision)
- NER proposes "DOUGLAS" → lands on covername entity (no collision)
- Relationships handle the "who is this" part separately

## Implementation Plan

### Phase 1: Schema Design

#### New Table: `entity_relationships`

**Note:** This is a lightweight "graph-shaped" structure in Postgres. You don't need Neo4j or a full graph engine—just a simple edges table.

**What you get:**
- Covername → person navigation
- Person → covernames listing
- Provenance for those claims
- Stop forcing identity claims into aliases

```sql
CREATE TABLE entity_relationships (
  id BIGSERIAL PRIMARY KEY,
  source_entity_id BIGINT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
  target_entity_id BIGINT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
  relationship_type TEXT NOT NULL CHECK (relationship_type IN (
    'covername_of',      -- covername → person
    'alias_of',          -- alias entity → canonical entity (for future use)
    'member_of',         -- person → org
    'located_in',        -- place → place (hierarchy)
    'same_as',           -- entity → entity (merger/deduplication)
    'derived_from'       -- entity → entity (provenance)
  )),
  confidence REAL DEFAULT 1.0 CHECK (confidence >= 0.0 AND confidence <= 1.0),
  source TEXT,           -- Where this relationship came from (concordance, manual, NER, etc.)
  notes TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  
  -- Prevent duplicate relationships
  UNIQUE (source_entity_id, target_entity_id, relationship_type),
  
  -- Prevent self-loops
  CHECK (source_entity_id != target_entity_id)
);

CREATE INDEX entity_relationships_source_idx ON entity_relationships(source_entity_id);
CREATE INDEX entity_relationships_target_idx ON entity_relationships(target_entity_id);
CREATE INDEX entity_relationships_type_idx ON entity_relationships(relationship_type);
```

### Phase 2: Backfill Migration

#### Identify Pattern: "Person name as alias of covername"

```sql
-- Find covername entities that have person names as aliases
-- Pattern: entity_type = 'cover_name' AND alias matches person name pattern
WITH covername_person_aliases AS (
  SELECT DISTINCT
    e_covername.id AS covername_entity_id,
    e_covername.canonical_name AS covername_name,
    ea.alias AS person_alias,
    ea.alias_norm AS person_alias_norm,
    -- Try to find matching person entity
    e_person.id AS person_entity_id,
    e_person.canonical_name AS person_name
  FROM entities e_covername
  JOIN entity_aliases ea ON ea.entity_id = e_covername.id
  LEFT JOIN entities e_person ON e_person.entity_type = 'person'
    AND (
      -- Exact match on canonical name
      LOWER(TRIM(e_person.canonical_name)) = LOWER(TRIM(ea.alias))
      OR
      -- Match on alias_norm (normalized)
      EXISTS (
        SELECT 1 FROM entity_aliases ea_person
        WHERE ea_person.entity_id = e_person.id
        AND ea_person.alias_norm = ea.alias_norm
      )
    )
  WHERE e_covername.entity_type = 'cover_name'
    AND ea.alias_norm NOT IN (
      -- Exclude covername-like aliases (all caps, short)
      SELECT alias_norm FROM entity_aliases ea2
      WHERE ea2.entity_id = e_covername.id
      AND (ea2.alias = UPPER(ea2.alias) OR LENGTH(TRIM(ea2.alias)) <= 4)
    )
    -- Person name heuristics: contains space, title case, etc.
    AND (
      LENGTH(TRIM(ea.alias)) - LENGTH(REPLACE(TRIM(ea.alias), ' ', '')) > 0  -- Has space
      OR LENGTH(TRIM(ea.alias)) > 4  -- Longer than typical covername
    )
)
SELECT * FROM covername_person_aliases
WHERE person_entity_id IS NOT NULL
ORDER BY covername_entity_id;
```

#### Create Relationships

```sql
-- Insert covername_of relationships
INSERT INTO entity_relationships (
  source_entity_id,
  target_entity_id,
  relationship_type,
  confidence,
  source,
  notes
)
SELECT
  covername_entity_id,
  person_entity_id,
  'covername_of',
  1.0,  -- High confidence for concordance-derived
  'concordance_backfill',
  'Migrated from alias: ' || person_alias
FROM covername_person_aliases
WHERE person_entity_id IS NOT NULL
ON CONFLICT (source_entity_id, target_entity_id, relationship_type) DO NOTHING;
```

#### Remove Person Name Aliases from Covername Entities

```sql
-- Delete person name aliases from covername entities
-- (Keep only covername-like aliases: all caps, short, etc.)
DELETE FROM entity_aliases ea
USING entities e
WHERE ea.entity_id = e.id
  AND e.entity_type = 'cover_name'
  AND ea.alias_norm IN (
    -- Person aliases that now have relationships
    SELECT DISTINCT person_alias_norm
    FROM covername_person_aliases
    WHERE person_entity_id IS NOT NULL
  )
  -- But keep covername-like aliases
  AND NOT (
    ea.alias = UPPER(ea.alias)  -- All caps
    OR LENGTH(TRIM(ea.alias)) <= 4  -- Short
  );
```

### Phase 3: Update Extraction Logic

#### No Changes Needed (If Aliases Are Correct)

The extraction logic (`extract_entity_mentions.py`) doesn't need changes if:
- Aliases only contain surface forms that appear in text
- Relationships are separate and don't affect matching

#### Optional: Relationship-Aware Display

When displaying mentions, can optionally show related entities:

```python
def get_related_entities(entity_id: int, relationship_type: str = 'covername_of') -> List[Dict]:
    """Get entities related via a specific relationship type."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 
              er.target_entity_id,
              e.canonical_name,
              e.entity_type
            FROM entity_relationships er
            JOIN entities e ON e.id = er.target_entity_id
            WHERE er.source_entity_id = %s
              AND er.relationship_type = %s
        """, (entity_id, relationship_type))
        return [{'id': row[0], 'name': row[1], 'type': row[2]} for row in cur.fetchall()]
```

### Phase 4: Validation Queries

#### Check for Remaining Collisions

```sql
-- Find alias_norms that still map to multiple entities
-- (Should be much fewer after refactor)
SELECT 
  alias_norm,
  COUNT(DISTINCT entity_id) AS entity_count,
  array_agg(DISTINCT entity_type) AS entity_types,
  array_agg(DISTINCT canonical_name) AS entity_names
FROM entity_aliases ea
JOIN entities e ON e.id = ea.entity_id
GROUP BY alias_norm
HAVING COUNT(DISTINCT entity_id) > 1
ORDER BY entity_count DESC
LIMIT 50;
```

#### Verify Relationships Created

```sql
-- Count relationships by type
SELECT 
  relationship_type,
  COUNT(*) AS count
FROM entity_relationships
GROUP BY relationship_type;

-- Sample covername_of relationships
SELECT 
  e1.canonical_name AS covername,
  e2.canonical_name AS person,
  er.source,
  er.created_at
FROM entity_relationships er
JOIN entities e1 ON e1.id = er.source_entity_id
JOIN entities e2 ON e2.id = er.target_entity_id
WHERE er.relationship_type = 'covername_of'
ORDER BY er.created_at DESC
LIMIT 20;
```

## Migration Script Structure

```python
#!/usr/bin/env python3
"""
Backfill entity relationships from existing alias patterns.

Converts: "Joseph Katz" as alias of DOUGLAS covername
Into: DOUGLAS --covername_of--> Joseph Katz relationship
"""

def identify_covername_person_patterns(conn):
    """Find covername entities with person name aliases."""
    # Query to find pattern
    pass

def create_relationships(conn, patterns):
    """Create entity_relationships from identified patterns."""
    pass

def remove_person_aliases_from_covernames(conn, patterns):
    """Remove person name aliases from covername entities."""
    pass

def validate_migration(conn):
    """Check that collisions decreased and relationships created."""
    pass

def main():
    conn = get_conn()
    try:
        patterns = identify_covername_person_patterns(conn)
        create_relationships(conn, patterns)
        remove_person_aliases_from_covernames(conn, patterns)
        validate_migration(conn)
    finally:
        conn.close()
```

## How to Proceed: Implementation Plan

### Step 1: Analyze Current State

**Run diagnostic query to understand the problem:**

```sql
-- Find alias_norms that map to multiple entities (collisions)
SELECT 
  alias_norm,
  COUNT(DISTINCT entity_id) AS entity_count,
  array_agg(DISTINCT entity_type) AS entity_types,
  array_agg(DISTINCT canonical_name ORDER BY canonical_name) AS entity_names
FROM entity_aliases ea
JOIN entities e ON e.id = ea.entity_id
GROUP BY alias_norm
HAVING COUNT(DISTINCT entity_id) > 1
ORDER BY entity_count DESC
LIMIT 50;
```

**Expected findings:**
- Many person names (like "joseph katz") mapping to multiple `cover_name` entities
- These are the high-collision surface strings causing problems

### Step 2: Create Schema Migration

**Create migration file:** `migrations/0025_entity_relationships.sql`

- Creates `entity_relationships` table
- Adds indexes
- Adds comments/documentation

**Run migration:**
```bash
make entity-relationships  # or psql -f migrations/0025_entity_relationships.sql
```

### Step 3: Create Backfill Script

**Create script:** `scripts/backfill_entity_relationships.py`

**Script does:**
1. **Identify pattern:** Find covername entities with person name aliases
2. **Match to person entities:** Find corresponding person entities
3. **Create relationships:** Insert `covername_of` relationships
4. **Remove person aliases:** Delete person name aliases from covername entities
5. **Validate:** Check collisions decreased, relationships created

**Run in dry-run mode first:**
```bash
python scripts/backfill_entity_relationships.py --dry-run
```

**Review output:**
- How many relationships will be created
- Which person aliases will be removed
- Expected collision reduction

### Step 4: Execute Backfill

**Run for real:**
```bash
python scripts/backfill_entity_relationships.py
```

**Verify results:**
- Check relationship counts
- Re-run collision diagnostic query
- Compare before/after collision counts

### Step 5: Test Extraction

**Re-run entity extraction:**
```bash
python scripts/extract_entity_mentions.py --collection venona --dry-run
```

**Expected improvements:**
- Fewer collisions
- More mentions emitted (previously blocked by collisions)
- "Joseph Katz" now resolves to person entity deterministically

### Step 6: (Optional) Update UI/Display Logic

**Add relationship-aware display:**
- When showing mention of covername, display linked person
- When showing mention of person, display related covernames
- Add navigation between related entities

## Implementation Decision: To Relationships or Not?

### If You Implement Relationships

**What you get:**
- Collisions drop dramatically (deterministic mention emission)
- Clean separation: aliases = surface forms, relationships = identity claims
- Covername → person navigation
- Person → covernames listing
- Provenance for claims
- Better UI/research experience

**What it costs:**
- One new table (`entity_relationships`)
- One backfill migration
- Simple JOIN queries when displaying related entities

### If You Don't Implement Relationships

**You can still proceed, but you'll keep paying the tax forever:**

1. **Collisions stay high**
   - "Joseph Katz" → 18 candidates
   - Need more and more dominance hacks / preferred maps
   - Harder to achieve deterministic emission

2. **UI will struggle**
   - Can't explain why "Joseph Katz" maps to 18 entities
   - Can't navigate from covername to person cleanly
   - Research experience is confusing

3. **Maintenance burden**
   - Need to keep adding dominance rules
   - Need to maintain preferred_entity_id maps
   - System becomes more complex over time

**Recommendation:** Implement relationships. It's a lightweight change (one table, one migration) that solves the problem at the root.

## Summary

**One Sentence:**
This refactor fixes the core problem: many `cover_name` entities contain person-name aliases, causing high-collision surface strings. By moving identity claims from aliases to relationships, we enable deterministic mention emission while preserving "many covernames per person" knowledge.

**Key Changes:**
1. New `entity_relationships` table (lightweight Postgres edges, not a full graph engine)
2. Backfill migration to convert existing patterns
3. Remove person name aliases from covername entities
4. Relationships handle "who is this" separately from "what appears in text"

**Result:**
- Collisions drop dramatically
- Deterministic mention emission ("Joseph Katz" → 1 entity)
- System becomes more maintainable
- Better UI/research experience
- Cleaner separation of concerns

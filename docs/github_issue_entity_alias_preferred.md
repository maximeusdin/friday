# Enhancement: Human Override Table for Entity Alias Collisions

## Problem Statement

When extracting entity mentions, certain tokens frequently cause unresolved collisions that require manual adjudication. Examples include:
- **"doctor"** - Could refer to a person name or a role title
- **"ussr"** - Could refer to an organization entity or a covername
- **"moscow"** - Could refer to a place or a covername
- **"viktor"** - Could refer to a person name or a covername

Currently, fixing these requires:
1. Modifying algorithm code
2. Re-running extraction
3. Testing changes
4. Potentially breaking other cases

This is slow and error-prone for quick fixes.

## Proposed Solution

Create a database table `entity_alias_preferred` that allows human operators to specify preferred entity mappings for specific alias collisions. These overrides are applied automatically during extraction at the highest priority (Rule 0 in collision resolution).

### Benefits

- **Fast fixes**: Fix the 10 worst tokens in 5 minutes via SQL INSERT statements
- **No code changes**: No need to modify algorithms or re-deploy code
- **Auditable**: All overrides are tracked with timestamps and optional notes
- **Scoped**: Supports both global overrides and collection-specific overrides
- **Reversible**: Easy to update or remove overrides as needed

## Implementation Details

### Database Schema

```sql
CREATE TABLE entity_alias_preferred (
    id BIGSERIAL PRIMARY KEY,
    scope TEXT,  -- Optional: collection slug (NULL = global override)
    alias_norm TEXT NOT NULL,  -- Normalized alias string
    preferred_entity_id BIGINT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    note TEXT,  -- Optional explanation
    
    UNIQUE (scope, alias_norm)
);

CREATE INDEX idx_entity_alias_preferred_lookup 
    ON entity_alias_preferred(scope, alias_norm);
```

### Integration Points

1. **Migration**: `migrations/0029_entity_alias_preferred.sql`
2. **Script**: `scripts/extract_entity_mentions.py`
   - `load_preferred_mappings()` function automatically loads from this table
   - Applied in `find_dominant_candidate()` as Rule 0 (highest priority)
3. **Makefile**: `make entity-alias-preferred` target to run migration

### Usage Examples

```sql
-- Fix "doctor" globally (prefer person entity over role)
INSERT INTO entity_alias_preferred (scope, alias_norm, preferred_entity_id, note)
VALUES (NULL, 'doctor', 12345, 'Person name, not role title');

-- Fix "ussr" for specific collection (prefer org over covername)
INSERT INTO entity_alias_preferred (scope, alias_norm, preferred_entity_id, note)
VALUES ('venona', 'ussr', 67890, 'Organization, not codename in Venona context');

-- Fix "moscow" globally (prefer place over covername)
INSERT INTO entity_alias_preferred (scope, alias_norm, preferred_entity_id, note)
VALUES (NULL, 'moscow', 99999, 'Place name, not codename');

-- Update an existing override
UPDATE entity_alias_preferred
SET preferred_entity_id = 11111, note = 'Updated: correct entity'
WHERE scope IS NULL AND alias_norm = 'viktor';

-- Remove an override
DELETE FROM entity_alias_preferred
WHERE scope IS NULL AND alias_norm = 'doctor';
```

### Priority Order

Overrides are applied in this priority order (highest to lowest):
1. **Scoped override** (`scope` matches current collection)
2. **Global override** (`scope` is NULL)
3. **CSV file** (if `--preferred-mappings-csv` provided)
4. **Custom table** (if `--preferred-mappings-table` provided)
5. **Algorithm-based resolution** (citation, dominance, etc.)

## Acceptance Criteria

- [ ] Migration file creates `entity_alias_preferred` table with correct schema
- [ ] Indexes are created for fast lookups
- [ ] `load_preferred_mappings()` automatically loads from table
- [ ] Overrides are applied in `find_dominant_candidate()` as Rule 0
- [ ] Scoped overrides take precedence over global overrides
- [ ] Makefile target exists to run migration
- [ ] Documentation includes usage examples
- [ ] Script handles missing table gracefully (for backward compatibility)

## Testing

1. **Create test overrides**:
   ```sql
   INSERT INTO entity_alias_preferred (scope, alias_norm, preferred_entity_id, note)
   VALUES (NULL, 'test_alias', 1, 'Test override');
   ```

2. **Run extraction** and verify override is applied:
   ```bash
   python scripts/extract_entity_mentions.py --collection venona --summary-csv test_summary.csv
   ```

3. **Check match_summary.csv** - should show collision resolved with override

4. **Verify scoped overrides** work correctly (collection-specific vs global)

## Future Enhancements (Out of Scope)

- UI for managing overrides (can be added later)
- Bulk import/export of overrides
- Override suggestions based on collision frequency
- Override expiration/versioning

## Related Issues

- Related to collision resolution improvements
- Complements 2-tier alias system implementation
- Works with preferred mappings CSV/table support

## Notes

- This is a **read-only** table from the extraction script's perspective
- Overrides can be managed via SQL directly or future UI
- No breaking changes - script works without table (graceful degradation)
- Table is automatically loaded - no CLI flags needed (unlike CSV/table options)

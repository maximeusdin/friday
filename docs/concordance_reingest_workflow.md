# Concordance Re-Ingest and Downstream Workflow

This guide walks through re-ingesting the concordance with a **new source slug**, then running the full downstream pipeline. Use this when you want a clean slate (e.g. after discovering data loss) without affecting the old source.

---

## Where Jacob Golos (and Others) May Have Been Deleted

Several scripts and migrations can remove entities or aliases. These are the main suspects:

### 1. **Alias cleanup** (historical: migrations 0065/0066 — now removed)

Migrations 0065 and 0066 were removed. Alias cleanup is now done by `scripts/cleanup_aliases.py` (reversible, JSON-logged). It only removes case-only and word-order duplicates.

**Impact**: If aliases were deleted, `entry_document_pages` loses rows; entities can become orphans (no aliases).

### 2. **Ingest `_is_garbage_alias`** (during ingest)

In `concordance/ingest_concordance_tab_aware.py`, `_is_garbage_alias()` drops aliases whose **raw text** contains: `unidentified`, `unknown`, `ussr`, `venona`, `vassiliev`.

**Impact**: Aliases like `"SOUND [ZVUK] (cover name in Venona)"` are dropped because the string contains "venona". The main tokens "Sound" and "ZVUK" are usually added separately, so they may still be present. Worth checking whether any Jacob Golos aliases were dropped here.

### 3. **Cleanup `delete-orphans`**

`python scripts/cleanup_concordance.py --delete-orphans --confirm` deletes entities that have **no aliases**.

**Impact**: If 0065/0066 removed all of Jacob Golos’s aliases, the entity becomes an orphan and gets deleted here.

### 4. **Cleanup `delete-garbage-entities`** and `--apply-file`

- `--delete-garbage-entities` deletes entities whose `canonical_name` matches:
  - > 3 words, or
  - Contains `;`, `–`, `—`, or digits.
- That run is logged to `garbage_entity_runs` in the session JSON.
- `--apply-file` with `delete_entities_from_garbage_runs: true` deletes all entities in `garbage_entity_runs` (except those in `keep_entity_names`, which we removed).

**Impact**: "Jacob Golos" (2 words) does not match the garbage rule, so he should not be in `garbage_entity_runs`. If he was deleted, it is more likely via orphan deletion or another path.

### 5. **Export filters** (`_is_garbage_alias`, `_is_boring_alias`)

`scripts/export_concordance_data.py` skips rows when building `entry_document_pages` if `_is_garbage_alias(alias_norm)` or `_is_boring_alias(alias_norm, canonical_norm)` returns true.

**Impact**: Even if data exists in the DB, it may be dropped during export.

---

## Re-Ingest with New Source Slug

Using a new slug creates a new `concordance_sources` row and isolates this run from the old one.

### Prerequisites

- PDF: `data/raw/index/Vassiliev_Notebooks_and_Venona_Index-Concordance.pdf` (or `_med.pdf`, `_small.pdf`)
- `DATABASE_URL` set (e.g. `source friday_env.sh`)

### Step 1: Re-ingest concordance (new slug)

```bash
python concordance/ingest_concordance_tab_aware.py \
  --pdf data/raw/index/Vassiliev_Notebooks_and_Venona_Index-Concordance.pdf \
  --source-slug vassiliev_venona_index_20260211 \
  --source-title "Vassiliev and Venona Index (re-ingest 20260211)"
```

- Creates (or updates) `concordance_sources` with the new slug.
- Parses the PDF and populates `concordance_entries`, `entities`, `entity_aliases`, `entity_links`, `entity_citations`.
- Does not run migrations 0065 or 0066 (removed). Use `scripts/cleanup_aliases.py` for reversible alias cleanup.

### Step 2: Apply cleanup from session JSON

```bash
python scripts/cleanup_concordance.py \
  --apply-file cleanup_session_vassiliev_venona_index_20260210.json \
  --slug vassiliev_venona_index_20260211 \
  --confirm
```

**Note**: The session was created for `vassiliev_venona_index_20260210`. With a new slug you get a new `source_id`, so `garbage_entity_runs` and `orphan_runs` from the old source are skipped. Entity/alias merges/deletes in the JSON may not apply if IDs differ.

### Step 3: Export concordance data

```bash
python scripts/export_concordance_data.py \
  -o concordance_export \
  --source-slug vassiliev_venona_index_20260211
```

Produces (among others):

- `concordance_export/entry_document_pages.csv`
- `concordance_export/entity_aliases.csv`
- `concordance_export/entities.csv`

### Step 4: Populate PEM

```bash
python scripts/populate_page_entity_mentions.py --truncate
```

- Reads `concordance_export/entry_document_pages.csv`.
- Resolves `alias_norm` / `canonical_name` to `entity_id` via `entities` and `entity_aliases`.
- Inserts into `page_entity_mentions`.

### Step 5: Verify Jacob Golos

```bash
psql $DATABASE_URL -c "
SELECT e.id, e.canonical_name, e.entity_type,
       (SELECT COUNT(*) FROM entity_aliases ea WHERE ea.entity_id = e.id) AS alias_count,
       (SELECT COUNT(*) FROM page_entity_mentions pem WHERE pem.entity_id = e.id) AS pem_count
FROM entities e
JOIN concordance_sources cs ON cs.id = e.source_id
WHERE cs.slug = 'vassiliev_venona_index_20260211'
  AND e.canonical_name ILIKE '%Jacob Golos%';
"
```

If Jacob Golos is present, you should see one row with `alias_count` > 0 and `pem_count` > 0. For the same source, you can also inspect aliases:

```bash
psql $DATABASE_URL -c "
SELECT ea.alias, ea.alias_norm, ea.alias_type
FROM entity_aliases ea
JOIN entities e ON e.id = ea.entity_id
JOIN concordance_sources cs ON cs.id = e.source_id
WHERE cs.slug = 'vassiliev_venona_index_20260211'
  AND e.canonical_name ILIKE '%Jacob Golos%';
"
```

### Step 6: Update downstream config (if needed)

If other tools use the concordance source slug:

- **`app_kv`**: `UPDATE app_kv SET value = 'vassiliev_venona_index_20260211' WHERE key = 'concordance_source_slug';`
- **`retrieval/ops.py`**: Default `source_slug` in `concordance_expand_terms` and similar functions.
- **`scripts/sync_entities_to_rds.py`**: `CONCORDANCE_SOURCE_SLUG` env var.
- **`scripts/build_alias_lexicon.py`**: `--source-slug vassiliev_venona_index_20260211`

---

## Data Flow Summary

```
PDF
  → ingest_concordance_tab_aware.py
    → concordance_sources, concordance_entries, entities, entity_aliases, entity_links, entity_citations

DB (entity_aliases + concordance_entries)
  → export_concordance_data.py
    → concordance_export/entry_document_pages.csv

entry_document_pages.csv + entities
  → populate_page_entity_mentions.py
    → page_entity_mentions
```


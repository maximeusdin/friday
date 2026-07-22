# Integrating cleanup_concordance.py Results into the Database

`scripts/cleanup_concordance.py` removes garbage entities and aliases from a concordance source. **Integration is direct**: the script updates the database in place. There is no separate "import" step; the database is the single source of truth before and after cleanup.

---

## How integration works today

### 1. The script writes directly to the database

- **Bulk mode** (default or `--confirm`): Deletes all rows matching the garbage rules (alias/entity with >3 words or containing `;`, `–`, `—`, digits). No session file.
- **Adjudicate mode** (`--adjudicate`): Interactive (d)elete / (s)kip / (e)dit / (m)erge per item; decisions are saved to `cleanup_session_{slug}.json`. When you answer "Apply? [y]", the script applies those decisions in the same run (UPDATEs and DELETEs), then deletes the session file.

In both modes, changes are **committed to the same database** you connect to via `--db` or `DATABASE_URL`. No intermediate CSV or patch file is required to "integrate" — the script **is** the integration.

### 2. Tables the script touches (order matters)

| Step | What the script does |
|------|----------------------|
| 1 | **Delete** garbage `entity_aliases` (by garbage condition on alias) |
| 2 | **Null out** `mention_candidates.resolved_entity_id` for garbage entity IDs (batched) |
| 3 | **Delete** remaining `entity_aliases` for garbage entities |
| 4 | **Delete** `entity_citations` for garbage entities |
| 5 | **Delete** `entity_links` where from/to is a garbage entity |
| 6 | **Null out** `ocr_variant_clusters.canonical_entity_id` for garbage entities (batched) |
| 7 | **Delete** garbage `entities` |

**Adjudicate-only:** Before the deletes, the script may **UPDATE** `entity_aliases` (edit) or `entities` (edit), and **INSERT** new entities/aliases for "merge" decisions.

### 3. Cascades and related data

- **entity_mentions**: Has `entity_id REFERENCES entities(id) ON DELETE CASCADE`. When an entity is deleted, Postgres automatically deletes all `entity_mentions` rows for that entity. The script does not need to delete them explicitly.
- **Other FKs to entities**: Any table with `REFERENCES entities(id) ON DELETE CASCADE` will lose rows when an entity is deleted; `ON DELETE SET NULL` will get nulled. The script only explicitly handles `entity_aliases`, `entity_citations`, `entity_links`, `mention_candidates`, and `ocr_variant_clusters` where needed.

So after a run:

- **DB state**: Fewer entities/aliases for that concordance source; mentions and other dependent rows are either cascade-deleted or explicitly updated.
- **Concordance export CSVs**: Still reflect the **pre-cleanup** state until you re-export.

---

## Recommended workflow: cleanup then re-export

1. **Run cleanup** (same concordance source slug you use elsewhere):
   ```bash
   python scripts/cleanup_concordance.py --slug vassiliev_venona_index_20260130 [--dry-run] [--confirm or --adjudicate]
   ```
2. **Re-export** so CSV snapshots match the cleaned DB:
   ```bash
   python scripts/export_concordance_data.py --source-slug vassiliev_venona_index_20260130 -o concordance_export
   ```

Use the **same slug** for both (or the "most recent" source used by the export script). That way:

- **Database**: Single source of truth; cleanup has already integrated (entities/aliases and cascades are updated).
- **concordance_export/*.csv**: Refreshed to match the cleaned DB for downstream tools (e.g. analysis, external systems that read the CSVs).

---

## Using the session JSON file to modify the DB

When you run **adjudicate** mode (`--adjudicate`), the script saves your decisions after each prompt to a **session file**: `cleanup_session_{slug}.json`. If you answer "n" at "Apply? [y/N]", no changes are written to the DB and the file is kept. You can then apply that file later so the JSON is the only record of what to do, and the DB is updated in a separate step.

### Session file format

The file is JSON with two top-level keys:

- **alias_decisions**: Object keyed by alias id (string). Each value: `{"action": "delete"}`, `{"action": "skip"}`, `{"action": "edit", "new_value": "..."}`, or `{"action": "merge", "entity_name": "...", "alias_text": "...", "entity_type": "org"}`.
- **entity_decisions**: Object keyed by entity id (string). Each value: `{"action": "delete"}`, `{"action": "skip"}`, or `{"action": "edit", "new_value": "..."}`.

### Apply the session file to the database

Use **`--apply-file`** with the path to the session JSON and the same **`--slug`** used when the session was created:

```bash
python scripts/cleanup_concordance.py --apply-file cleanup_session_vassiliev_venona_index_20260130.json --slug vassiliev_venona_index_20260130
```

- **`--confirm`**: Prompt "Apply to database? [y/N]" before making changes.
- **`--keep-session`**: Do not delete the session file after a successful apply.

The script loads the JSON, derives the same actions as in interactive apply, applies them (edits, merges, alias deletes, entity deletes), and optionally removes the session file if it is the default path and `--keep-session` is not set.

#!/bin/bash
# Full local DB sync: migrations, concordance cleanup, alias cleanup, PEM, canonical re-embed.
#
# Prerequisites:
#   - DATABASE_URL set (or DB_HOST, DB_NAME, DB_USER, DB_PASS)
#   - Concordance already ingested (vassiliev_venona_index_20260210)
#   - cleanup_session_vassiliev_venona_index_20260210.json in repo root
#
# Run from repo root:
#   bash scripts/local_db_sync_full.sh
#
# Or run steps individually if you need to inspect between steps.

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

SLUG="vassiliev_venona_index_20260210"
CLEANUP_SESSION="cleanup_session_${SLUG}.json"
CONCURRENCY=4

echo "=== 1. Migrations ==="
python scripts/run_migrations.py

echo ""
echo "=== 2. JSON file cleanup (apply cleanup_session) ==="
if [ -f "$CLEANUP_SESSION" ]; then
  python scripts/cleanup_concordance.py --apply-file "$CLEANUP_SESSION" --slug "$SLUG"
else
  echo "  Skipping: $CLEANUP_SESSION not found"
fi

echo ""
echo "=== 3. Alias cleanup ==="
python scripts/cleanup_aliases.py --slug "$SLUG" --confirm

echo ""
echo "=== 4. Concordance export ==="
python scripts/export_concordance_data.py --source-slug "$SLUG" -o concordance_export

echo ""
echo "=== 5. Populate page_entity_mentions ==="
python scripts/populate_page_entity_mentions.py --truncate

echo ""
echo "=== 6. Re-embed canonical chunks (all collections) ==="
python scripts/embed_canonical_chunks.py --all-collections --rebuild --concurrency "$CONCURRENCY"

echo ""
echo "=== Done ==="

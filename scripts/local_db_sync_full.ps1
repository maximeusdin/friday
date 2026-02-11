# Full local DB sync: migrations, concordance cleanup, alias cleanup, PEM, canonical re-embed.
#
# Prerequisites:
#   - DATABASE_URL set (or DB_HOST, DB_NAME, DB_USER, DB_PASS)
#   - Concordance already ingested (vassiliev_venona_index_20260210)
#   - cleanup_session_vassiliev_venona_index_20260210.json in repo root
#
# Run from repo root:
#   .\scripts\local_db_sync_full.ps1
#
# Or run steps individually if you need to inspect between steps.

$ErrorActionPreference = "Stop"
$REPO_ROOT = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $REPO_ROOT

$SLUG = "vassiliev_venona_index_20260210"
$CLEANUP_SESSION = "cleanup_session_$SLUG.json"
$CONCURRENCY = 4

Write-Host "=== 1. Migrations ===" -ForegroundColor Cyan
python scripts/run_migrations.py

Write-Host ""
Write-Host "=== 2. JSON file cleanup (apply cleanup_session) ===" -ForegroundColor Cyan
if (Test-Path $CLEANUP_SESSION) {
    python scripts/cleanup_concordance.py --apply-file $CLEANUP_SESSION --slug $SLUG
} else {
    Write-Host "  Skipping: $CLEANUP_SESSION not found" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== 3. Alias cleanup ===" -ForegroundColor Cyan
python scripts/cleanup_aliases.py --slug $SLUG --confirm

Write-Host ""
Write-Host "=== 4. Concordance export ===" -ForegroundColor Cyan
python scripts/export_concordance_data.py --source-slug $SLUG -o concordance_export

Write-Host ""
Write-Host "=== 5. Populate page_entity_mentions ===" -ForegroundColor Cyan
python scripts/populate_page_entity_mentions.py --truncate

Write-Host ""
Write-Host "=== 6. Re-embed canonical chunks (all collections) ===" -ForegroundColor Cyan
python scripts/embed_canonical_chunks.py --all-collections --rebuild --concurrency $CONCURRENCY

Write-Host ""
Write-Host "=== Done ===" -ForegroundColor Green

# Full searchable-PDF run: all image-only collections, small first.
# Geometry checkpoints to ocr_cache_geom/ every 50 pages - safe to re-run anytime.
$ErrorActionPreference = 'Continue'
Set-Location C:\Users\maxim\friday

$collections = @(
    'data/raw/pravdin',
    'data/raw/Winton Burdett',
    'data/raw/Jack Childs',
    'data/raw/Morris Childs',
    'data/raw/fbi_solo'
)

foreach ($dir in $collections) {
    Write-Output "==== START $dir ===="
    python -u scripts/make_searchable_pdfs.py --dir $dir --workers 8
    Write-Output "==== FINISHED $dir (exit $LASTEXITCODE) ===="
}
Write-Output "==== ALL COLLECTIONS DONE ===="

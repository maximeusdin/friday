# Workflow script to ingest concordance and extract entity mentions from citations
# PowerShell version
#
# Usage:
#   .\concordance\run_citation_based_extraction.ps1 <pdf_path> [options]
#
# Example:
#   .\concordance\run_citation_based_extraction.ps1 data\concordance_index.pdf

param(
    [Parameter(Mandatory=$true)]
    [string]$PdfPath,
    
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$IngestOptions
)

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Step 1: Ingesting concordance index" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

$ingestArgs = @($PdfPath) + $IngestOptions
& python concordance/ingest_concordance_tab_aware.py $ingestArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host "Ingest failed. Aborting." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Step 2: Extracting entity mentions from citations" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "This will extract mentions for all entities that have citations."
Write-Host "Run with --dry-run first to preview:"
Write-Host "  python concordance/extract_entity_mentions_from_citations.py --all-entities --dry-run"
Write-Host ""

$response = Read-Host "Continue with extraction? (y/n)"
if ($response -eq 'y' -or $response -eq 'Y') {
    & python concordance/extract_entity_mentions_from_citations.py --all-entities
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Extraction failed." -ForegroundColor Red
        exit $LASTEXITCODE
    }
} else {
    Write-Host "Skipping extraction. Run manually when ready:"
    Write-Host "  python concordance/extract_entity_mentions_from_citations.py --all-entities"
}

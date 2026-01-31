#!/bin/bash
# Workflow script to ingest concordance and extract entity mentions from citations
#
# Usage:
#   ./concordance/run_citation_based_extraction.sh <pdf_path> [options]
#
# Example:
#   ./concordance/run_citation_based_extraction.sh data/concordance_index.pdf

set -e

PDF_PATH="$1"
if [ -z "$PDF_PATH" ]; then
    echo "Error: PDF path required"
    echo "Usage: $0 <pdf_path> [ingest_options]"
    exit 1
fi

shift  # Remove PDF path from arguments

echo "=========================================="
echo "Step 1: Ingesting concordance index"
echo "=========================================="
python concordance/ingest_concordance_tab_aware.py "$PDF_PATH" "$@"

echo ""
echo "=========================================="
echo "Step 2: Extracting entity mentions from citations"
echo "=========================================="
echo "This will extract mentions for all entities that have citations."
echo "Run with --dry-run first to preview:"
echo "  python concordance/extract_entity_mentions_from_citations.py --all-entities --dry-run"
echo ""
read -p "Continue with extraction? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python concordance/extract_entity_mentions_from_citations.py --all-entities
else
    echo "Skipping extraction. Run manually when ready:"
    echo "  python concordance/extract_entity_mentions_from_citations.py --all-entities"
fi

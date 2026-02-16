#!/bin/bash
# sync_to_aws.sh
#
# Sync missing data to AWS RDS:
#   1. Ingest Silvermaster collection (documents + pages)
#   2. Chunk Venona collection
#   3. Chunk Vassiliev collection
#
# Usage (from WSL):
#   cd /mnt/c/Users/maxim/friday
#   bash scripts/sync_to_aws.sh [--silvermaster] [--venona] [--vassiliev] [--all]
#
# Default: run all three if no flags specified

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
echo_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
echo_err() { echo -e "${RED}[ERROR]${NC} $1"; }

# Parse arguments
DO_SILVERMASTER=false
DO_VENONA=false
DO_VASSILIEV=false

if [[ $# -eq 0 ]] || [[ "$1" == "--all" ]]; then
    DO_SILVERMASTER=true
    DO_VENONA=true
    DO_VASSILIEV=true
else
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --silvermaster) DO_SILVERMASTER=true ;;
            --venona) DO_VENONA=true ;;
            --vassiliev) DO_VASSILIEV=true ;;
            --all) DO_SILVERMASTER=true; DO_VENONA=true; DO_VASSILIEV=true ;;
            *) echo_err "Unknown option: $1"; exit 1 ;;
        esac
        shift
    done
fi

echo "=============================================="
echo "AWS Database Sync Script"
echo "=============================================="
echo "Operations to run:"
echo "  - Silvermaster ingest: $DO_SILVERMASTER"
echo "  - Venona chunking: $DO_VENONA"
echo "  - Vassiliev chunking: $DO_VASSILIEV"
echo ""

# Source AWS credentials
echo_info "Loading AWS credentials..."
if [[ ! -f "./friday_env.sh" ]]; then
    echo_err "friday_env.sh not found. Run from project root: /mnt/c/Users/maxim/friday"
    exit 1
fi

source ./friday_env.sh

# Verify AWS connection
echo_info "Testing AWS RDS connection..."
if ! psql "$DATABASE_URL" -c "SELECT 1" > /dev/null 2>&1; then
    echo_err "Cannot connect to AWS RDS. Check credentials and network."
    exit 1
fi
echo_info "AWS RDS connection OK (DATABASE_URL set)"

# DATABASE_URL is used by psql and Python scripts

echo ""

# ==============================================
# 1. Silvermaster Ingest
# ==============================================
if [[ "$DO_SILVERMASTER" == "true" ]]; then
    echo "=============================================="
    echo "1. SILVERMASTER INGEST"
    echo "=============================================="
    
    SILVERMASTER_PDF_DIR="/mnt/c/Users/maxim/friday/data/raw/silvermaster/pdf"
    
    if [[ ! -d "$SILVERMASTER_PDF_DIR" ]]; then
        echo_err "Silvermaster PDF directory not found: $SILVERMASTER_PDF_DIR"
        echo_warn "Skipping Silvermaster ingest"
    else
        PDF_COUNT=$(find "$SILVERMASTER_PDF_DIR" -name "*.pdf" 2>/dev/null | wc -l)
        echo_info "Found $PDF_COUNT PDF files in $SILVERMASTER_PDF_DIR"
        
        echo_info "Running Silvermaster PDF ingest..."
        python3 scripts/ingest_silvermaster_pdfs.py \
            --input-dir "$SILVERMASTER_PDF_DIR" \
            --pipeline-version "silvermaster_pages_v1"
        
        echo ""
        echo_info "Running Silvermaster chunking..."
        python3 scripts/chunk_corpus.py \
            --collection silvermaster \
            --pipeline-version "chunk_v1_silvermaster_structured_4k" \
            --max-chars 4000
        
        echo_info "Silvermaster ingest complete!"
    fi
    echo ""
fi

# ==============================================
# 2. Venona Chunking
# ==============================================
if [[ "$DO_VENONA" == "true" ]]; then
    echo "=============================================="
    echo "2. VENONA CHUNKING"
    echo "=============================================="
    
    # Check if Venona documents exist in AWS
    VENONA_DOCS=$(psql "$DATABASE_URL" -t -c "SELECT COUNT(*) FROM documents d JOIN collections c ON c.id = d.collection_id WHERE c.slug = 'venona'")
    VENONA_DOCS=$(echo "$VENONA_DOCS" | tr -d '[:space:]')
    
    if [[ "$VENONA_DOCS" == "0" ]]; then
        echo_warn "No Venona documents found in AWS. Skipping chunking."
        echo_warn "You may need to run the Venona ingest first."
    else
        echo_info "Found $VENONA_DOCS Venona documents in AWS"
        
        # Check current chunk count
        VENONA_CHUNKS=$(psql "$DATABASE_URL" -t -c "
            SELECT COUNT(*) FROM chunks ch
            JOIN chunk_pages cp ON cp.chunk_id = ch.id
            JOIN pages p ON p.id = cp.page_id
            JOIN documents d ON d.id = p.document_id
            JOIN collections c ON c.id = d.collection_id
            WHERE c.slug = 'venona'
        ")
        VENONA_CHUNKS=$(echo "$VENONA_CHUNKS" | tr -d '[:space:]')
        echo_info "Current Venona chunks in AWS: $VENONA_CHUNKS"
        
        echo_info "Running Venona chunking..."
        python3 scripts/chunk_corpus.py \
            --collection venona \
            --pipeline-version "chunk_v1_full" \
            --venona-max-chars 24000
        
        echo_info "Venona chunking complete!"
    fi
    echo ""
fi

# ==============================================
# 3. Vassiliev Chunking
# ==============================================
if [[ "$DO_VASSILIEV" == "true" ]]; then
    echo "=============================================="
    echo "3. VASSILIEV CHUNKING"
    echo "=============================================="
    
    # Check if Vassiliev documents exist in AWS
    VASS_DOCS=$(psql "$DATABASE_URL" -t -c "SELECT COUNT(*) FROM documents d JOIN collections c ON c.id = d.collection_id WHERE c.slug = 'vassiliev'")
    VASS_DOCS=$(echo "$VASS_DOCS" | tr -d '[:space:]')
    
    if [[ "$VASS_DOCS" == "0" ]]; then
        echo_warn "No Vassiliev documents found in AWS. Skipping chunking."
        echo_warn "You may need to run the Vassiliev ingest first."
    else
        echo_info "Found $VASS_DOCS Vassiliev documents in AWS"
        
        # Check current chunk count
        VASS_CHUNKS=$(psql "$DATABASE_URL" -t -c "
            SELECT COUNT(*) FROM chunks ch
            JOIN chunk_pages cp ON cp.chunk_id = ch.id
            JOIN pages p ON p.id = cp.page_id
            JOIN documents d ON d.id = p.document_id
            JOIN collections c ON c.id = d.collection_id
            WHERE c.slug = 'vassiliev'
        ")
        VASS_CHUNKS=$(echo "$VASS_CHUNKS" | tr -d '[:space:]')
        echo_info "Current Vassiliev chunks in AWS: $VASS_CHUNKS"
        
        echo_info "Running Vassiliev chunking..."
        python3 scripts/chunk_corpus.py \
            --collection vassiliev \
            --pipeline-version "chunk_v1_full" \
            --max-chars 4000
        
        echo_info "Vassiliev chunking complete!"
    fi
    echo ""
fi

# ==============================================
# Summary
# ==============================================
echo "=============================================="
echo "SYNC COMPLETE - Summary"
echo "=============================================="

echo_info "Final counts in AWS:"
psql "$DATABASE_URL" -c "
SELECT 
    c.slug as collection,
    COUNT(DISTINCT d.id) as documents,
    COUNT(DISTINCT p.id) as pages,
    COUNT(DISTINCT ch.id) as chunks
FROM collections c
LEFT JOIN documents d ON d.collection_id = c.id
LEFT JOIN pages p ON p.document_id = d.id
LEFT JOIN chunk_pages cp ON cp.page_id = p.id
LEFT JOIN chunks ch ON ch.id = cp.chunk_id
GROUP BY c.slug
ORDER BY c.slug;
"

echo ""
echo_info "Done!"

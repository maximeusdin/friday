#!/usr/bin/env bash
set -euo pipefail

# =========================================
# Bulk ingest: Venona PDFs (PDF pages only)
# =========================================
#
# What this does:
# - Iterates over *.pdf in VENONA_DIR
# - Uses stable source_key = "venona:<stem>"
# - Sets volume = "Venona – <stem>"
#
# Usage (host shell):
#   export DB_HOST=localhost DB_PORT=5432 DB_NAME=neh DB_USER=neh DB_PASS=neh
#   ./scripts/ingest_all_venona.sh /app/data/venona
#

export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=neh
export DB_USER=neh
export DB_PASS=neh

cd /mnt/c/Users/maxim/friday

PYTHON_BIN="${PYTHON_BIN:-python3}"
INGEST_SCRIPT="${INGEST_SCRIPT:-scripts/ingest_venona_pdf.py}"
LANGUAGE="${LANGUAGE:-en}"

# Default Venona PDF directory (override by passing arg1)
VENONA_DIR="./data/raw/venona"

VOLUME_PREFIX="${VOLUME_PREFIX:-Venona}"
SOURCE_PREFIX="${SOURCE_PREFIX:-venona}"

# -----------------------------------------
# Sanity checks
# -----------------------------------------
if [[ ! -d "$VENONA_DIR" ]]; then
  echo "ERROR: VENONA_DIR does not exist: $VENONA_DIR" >&2
  exit 1
fi

if [[ ! -f "$INGEST_SCRIPT" ]]; then
  echo "ERROR: ingest script not found: $INGEST_SCRIPT" >&2
  echo "Set INGEST_SCRIPT=... or place it at scripts/ingest_venona_pdf.py" >&2
  exit 1
fi

# DB env vars should be set (script will read them)
if [[ -z "${DB_HOST:-}" || -z "${DB_NAME:-}" || -z "${DB_USER:-}" ]]; then
  echo "ERROR: DB environment variables not set (DB_HOST, DB_NAME, DB_USER)" >&2
  echo "Example:" >&2
  echo "  export DB_HOST=localhost DB_PORT=5432 DB_NAME=neh DB_USER=neh DB_PASS=neh" >&2
  exit 1
fi

shopt -s nullglob
pdfs=( "$VENONA_DIR"/*.pdf )
shopt -u nullglob

if [[ ${#pdfs[@]} -eq 0 ]]; then
  echo "No PDFs found in: $VENONA_DIR"
  exit 0
fi

echo "Starting bulk ingest of Venona PDFs"
echo "Directory: $VENONA_DIR"
echo "Using script: $INGEST_SCRIPT"
echo "Found ${#pdfs[@]} PDFs"
echo

# -----------------------------------------
# Ingest loop
# -----------------------------------------
count=0
failures=0

for pdf in "${pdfs[@]}"; do
  stem="$(basename "$pdf" .pdf)"
  source_key="${SOURCE_PREFIX}:${stem}"
  volume="${VOLUME_PREFIX} – ${stem}"

  echo "----------------------------------------"
  echo "Ingesting: $pdf"
  echo "  source_key = $source_key"
  echo "  volume     = $volume"
  echo

  if "$PYTHON_BIN" "$INGEST_SCRIPT" \
        --pdf "$pdf" \
        --source-key "$source_key" \
        --volume "$volume" \
        --language "$LANGUAGE"
  then
    ((count+=1))
  else
    echo "ERROR: ingest failed for $pdf" >&2
    ((failures+=1))
  fi
done

# -----------------------------------------
# Summary
# -----------------------------------------
echo
echo "========================================"
echo "Venona ingest complete"
echo "  Successes: $count"
echo "  Failures:  $failures"
echo "========================================"

if [[ "$failures" -gt 0 ]]; then
  exit 1
fi

#!/usr/bin/env bash
set -euo pipefail

# ============================
# Configuration
# ============================

export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=neh
export DB_USER=neh
export DB_PASS=neh


# Directory containing Vassiliev PDFs
VASSILIEV_DIR="../data/raw/vassiliev"

# Python interpreter (adjust if needed)
PYTHON_BIN="python3"

# Ingest script path
INGEST_SCRIPT="ingest_vassiliev_pdf.py"

# Language code for pages
LANGUAGE="en"

# Optional: fixed volume label prefix (useful for grouping)
VOLUME_PREFIX="Vassiliev"

# ============================
# Environment sanity checks
# ============================

if [[ ! -d "$VASSILIEV_DIR" ]]; then
  echo "ERROR: VASSILIEV_DIR does not exist: $VASSILIEV_DIR" >&2
  exit 1
fi

if [[ ! -f "$INGEST_SCRIPT" ]]; then
  echo "ERROR: ingest script not found: $INGEST_SCRIPT" >&2
  exit 1
fi

if [[ -z "${DB_HOST:-}" || -z "${DB_NAME:-}" || -z "${DB_USER:-}" ]]; then
  echo "ERROR: DB environment variables not set (DB_HOST, DB_NAME, DB_USER)" >&2
  exit 1
fi

echo "Starting bulk ingest of Vassiliev PDFs"
echo "Directory: $VASSILIEV_DIR"
echo "Using script: $INGEST_SCRIPT"
echo

# ============================
# Ingest loop
# ============================

count=0
failures=0

for pdf in "$VASSILIEV_DIR"/*.pdf; do
  if [[ ! -f "$pdf" ]]; then
    echo "No PDFs found in $VASSILIEV_DIR"
    exit 0
  fi

  stem="$(basename "$pdf" .pdf)"
  source_key="vassiliev:${stem}"
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

# ============================
# Summary
# ============================

echo
echo "========================================"
echo "Vassiliev ingest complete"
echo "  Successes: $count"
echo "  Failures:  $failures"
echo "========================================"

if [[ "$failures" -gt 0 ]]; then
  exit 1
fi

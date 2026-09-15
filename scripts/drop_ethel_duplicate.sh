#!/usr/bin/env bash
# Drop the duplicate 'Rosenberg, Ethel HQ see references.pdf' from rosenberg_ethel:
# DB rows (migration 0077) -> S3 object -> collection zip rebuild.
#
# Verified duplicate of 'ROSENBERG, ETHEL HQ SEE REF A.pdf' pp. 103-179; the local
# PDF and its OCR cache are kept, so re-ingest is free if this ever needs undoing.
#
# Usage:  bash scripts/drop_ethel_duplicate.sh
set -euo pipefail
cd "$(dirname "$0")/.."

export AWS_SHARED_CREDENTIALS_FILE="$PWD/.aws/credentials"
export PATH="/opt/anaconda3/envs/friday/bin:$PATH"
export DATABASE_URL="$(aws secretsmanager get-secret-value --region us-west-1 \
  --secret-id friday/DATABASE_URL --query SecretString --output text)"

DUP="Rosenberg, Ethel HQ see references.pdf"

echo "=== before"
python scripts/check_collection.py rosenberg_ethel || true

echo "=== 1/3 DB rows (migration 0077)"
python scripts/apply_sql.py migrations/0077_drop_ethel_duplicate_doc.sql

echo "=== 2/3 S3 object"
aws s3 rm "s3://fridayarchive.org/data/raw/rosenberg_ethel/$DUP" --region us-west-1

echo "=== 3/3 collection zip"
python scripts/build_collection_zips.py --collections rosenberg_ethel --invalidate

echo "=== after (expect 10 docs, 343 pages, 160 chunks)"
python scripts/check_collection.py rosenberg_ethel

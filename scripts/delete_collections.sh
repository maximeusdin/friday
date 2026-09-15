#!/usr/bin/env bash
# Delete collections from prod: DB rows (FK-ordered), S3 PDFs + per-collection
# zip, then rebuild data/zips/manifest.json from what's left in the DB.
#
# The inverse of scripts/restore_kv_collections.sh — a deleted collection can be
# brought back from data/raw/<slug>/ + ocr_cache/<slug>/ (OCR is free from cache;
# only the embeddings cost anything).
#
# Usage:  bash scripts/delete_collections.sh [--execute] slug [slug ...]
#         (without --execute it only reports what would go)
set -euo pipefail
cd "$(dirname "$0")/.."

export AWS_SHARED_CREDENTIALS_FILE="$PWD/.aws/credentials"
export PATH="/opt/anaconda3/envs/friday/bin:$PATH"
export DATABASE_URL="$(aws secretsmanager get-secret-value --region us-west-1 \
  --secret-id friday/DATABASE_URL --query SecretString --output text)"

EXECUTE=""
SLUGS=()
for a in "$@"; do
  case "$a" in
    --execute) EXECUTE="--execute" ;;
    -*) echo "unknown flag: $a" >&2; exit 2 ;;
    *) SLUGS+=("$a") ;;
  esac
done
[ ${#SLUGS[@]} -eq 0 ] && { echo "usage: $0 [--execute] slug [slug ...]" >&2; exit 2; }

echo "=== DB rows"
python scripts/delete_collection_rows.py $EXECUTE "${SLUGS[@]}"

if [ -z "$EXECUTE" ]; then
  echo
  echo "=== S3 (dry run)"
  for slug in "${SLUGS[@]}"; do
    aws s3 rm "s3://fridayarchive.org/data/raw/$slug/" --recursive --region us-west-1 --dryrun
    echo "(would delete) s3://fridayarchive.org/data/zips/$slug.zip"
  done
  echo
  echo "Re-run with --execute to apply."
  exit 0
fi

echo
echo "=== S3 PDFs + zips"
for slug in "${SLUGS[@]}"; do
  aws s3 rm "s3://fridayarchive.org/data/raw/$slug/" --recursive --region us-west-1
  aws s3 rm "s3://fridayarchive.org/data/zips/$slug.zip" --region us-west-1 || true
done

echo
echo "=== Rebuild downloads manifest (every surviving collection is unchanged, so nothing rebuilds)"
python scripts/build_collection_zips.py --invalidate

echo
echo "=== Verify"
python scripts/check_collection.py --list

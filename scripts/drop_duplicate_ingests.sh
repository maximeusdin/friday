#!/usr/bin/env bash
# Drop the 26 duplicate re-ingests found by the 2026-09-15 corpus sweep:
# DB rows (migration 0078) -> S3 objects -> collection zip rebuilds.
#
# 23 solo documents (the zero-padded/unpadded double ingest of 2026-02-01/02),
# rosenberg 'Julius 48' (a copy of 47), harry_gold '65-57449-13_Part2' (a copy of
# Part1) and one re-downloaded fbi_hiskey fragment. See
# docs/DUPLICATE_DOCUMENTS_2026-09-15.md and the header of migration 0078.
#
# Unlike the ethel drop, data/raw/solo is EMPTY locally and ocr_cache/solo holds
# nothing, so a solo document restored from S3 would have to be OCR'd again. The S3
# step is therefore held behind a confirmation and runs last.
#
# Usage:  bash scripts/drop_duplicate_ingests.sh          # DB + zips, leaves S3 alone
#         bash scripts/drop_duplicate_ingests.sh --s3     # also delete the S3 objects
set -euo pipefail
cd "$(dirname "$0")/.."

export AWS_SHARED_CREDENTIALS_FILE="$PWD/.aws/credentials"
export PATH="/opt/anaconda3/envs/friday/bin:$PATH"
export DATABASE_URL="$(aws secretsmanager get-secret-value --region us-west-1 \
  --secret-id friday/DATABASE_URL --query SecretString --output text)"

COLLECTIONS="solo rosenberg harry_gold fbi_hiskey"

DROPPED_OBJECTS=(
  "data/raw/solo/100-HQ-428091-EBF0099_text.pdf"
  "data/raw/solo/100-HQ-428091-EBF1405_text.pdf"
  "data/raw/solo/100-HQ-428091-Serial01-44_text.pdf"
  "data/raw/solo/100-HQ-428091-Serial0045-0069_text.pdf"
  "data/raw/solo/100-HQ-428091-Serial0070-0076_text.pdf"
  "data/raw/solo/100-HQ-428091-Serial0077-0162_text.pdf"
  "data/raw/solo/100-HQ-428091-Serial0163-0206_text.pdf"
  "data/raw/solo/100-HQ-428091-Serial0206-0228_text.pdf"
  "data/raw/solo/100-HQ-428091-Serial0229-0316_text.pdf"
  "data/raw/solo/100-HQ-428091-Serial317-320_text.pdf"
  "data/raw/solo/100-HQ-428091-Serial0321-0431_text.pdf"
  "data/raw/solo/100-HQ-428091-Serial0432-0509_text.pdf"
  "data/raw/solo/100-HQ-428091-Serial509-514_text.pdf"
  "data/raw/solo/100-HQ-428091-Serial0518-0585_text.pdf"
  "data/raw/solo/100-HQ-428091-Serial0586-0599_text.pdf"
  "data/raw/solo/100-HQ-428091-Serial0601-0711_text.pdf"
  "data/raw/solo/100-HQ-428091-Serial0712-0725_text.pdf"
  "data/raw/solo/100-HQ-428091-Serial0726-0828_text.pdf"
  "data/raw/solo/100-HQ-428091-Serial0829-0907_text.pdf"
  "data/raw/solo/100-HQ-428091-Serial0909-0958_text.pdf"
  "data/raw/solo/100-HQ-428091-Serial0958-0997_text.pdf"
  "data/raw/solo/100-HQ-428091-Serial0998-1065_text.pdf"
  "data/raw/solo/SOLO-045_text.pdf"
  "data/raw/rosenberg/Rosenberg, Julius 48_text.pdf"
  "data/raw/harry_gold/Gold- Harry-HQ-65-57449-13_Part2.pdf"
  "data/raw/fbi_hiskey/20220318_2, Previous investigation on Clarence F, Hiskey was  co.pdf"
)

echo "=== 0/4 dry run (read-only) — every guard must pass"
python scripts/dryrun_0078.py

echo
echo "=== before"
for c in $COLLECTIONS; do python scripts/check_collection.py "$c" || true; done

echo
echo "=== 1/4 DB rows (migration 0078)"
python scripts/apply_sql.py migrations/0078_drop_duplicate_ingests.sql

echo
echo "=== 2/4 collection zips"
python scripts/build_collection_zips.py --collections $(echo $COLLECTIONS | tr ' ' ',') --invalidate

echo
echo "=== 3/4 after (expect solo 101, rosenberg 187, harry_gold 104, fbi_hiskey 62 docs)"
for c in $COLLECTIONS; do python scripts/check_collection.py "$c" || true; done

echo
if [[ "${1:-}" != "--s3" ]]; then
  echo "=== 4/4 S3 objects: SKIPPED"
  echo "    The ${#DROPPED_OBJECTS[@]} source PDFs are still in s3://fridayarchive.org/, which is the"
  echo "    only rollback path for the solo documents (no local copy, no OCR cache)."
  echo "    Re-run with --s3 once you are satisfied with the result."
  exit 0
fi

echo "=== 4/4 S3 objects — deleting ${#DROPPED_OBJECTS[@]} source PDFs"
read -r -p "    This is not reversible. Type 'delete' to continue: " confirm
[[ "$confirm" == "delete" ]] || { echo "    aborted; DB changes above are already applied."; exit 1; }
for key in "${DROPPED_OBJECTS[@]}"; do
  aws s3 rm "s3://fridayarchive.org/$key" --region us-west-1
done
echo "    done. The zips built in step 2/4 are already correct — their contents come"
echo "    from the documents table, not from what is left in the bucket."

#!/usr/bin/env bash
#
# Upload a collection's source PDFs to S3 so the prod document viewer can serve them.
#
# Prod serves PDFs from s3://<bucket>/data/raw/<slug>/[<subfolder>/]<filename>, mirroring
# the local data/raw layout (see backend/app/routes/documents.py _build_s3_url). Text
# search does NOT need this; only "view the original scan" does.
#
# Usage (run in WSL, where the aws CLI + credentials live):
#   bash scripts/upload_pdfs_to_s3.sh --slug siss_scope_soviet \
#        --src-dir /mnt/c/Users/maxim/Downloads \
#        --glob 'Scope_of_Soviet_Activity_in_the_U_S part*.pdf'
#
#   # collection that uses a pdf/ subfolder (vassiliev, silvermaster):
#   bash scripts/upload_pdfs_to_s3.sh --slug vassiliev --src-dir ... --subfolder pdf
#
# Flags:
#   --slug       collection slug -> S3 prefix data/raw/<slug>/
#   --src-dir    local directory containing the PDFs (use /mnt/c/... from WSL)
#   --glob       filename glob within src-dir (default: *.pdf)
#   --subfolder  optional extra prefix segment (e.g. pdf) for slugs that use data/raw/<slug>/pdf/
#   --bucket     S3 bucket (default: fridayarchive.org)
#   --region     AWS region (default: us-west-1)
#   --dry-run    list what would be uploaded, do nothing
#
set -euo pipefail

SLUG=""
SRC_DIR=""
GLOB="*.pdf"
SUBFOLDER=""
BUCKET="fridayarchive.org"
REGION="us-west-1"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --slug) SLUG="$2"; shift 2;;
    --src-dir) SRC_DIR="$2"; shift 2;;
    --glob) GLOB="$2"; shift 2;;
    --subfolder) SUBFOLDER="$2"; shift 2;;
    --bucket) BUCKET="$2"; shift 2;;
    --region) REGION="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    *) echo "Unknown arg: $1" >&2; exit 2;;
  esac
done

[[ -n "$SLUG" ]] || { echo "ERROR: --slug required" >&2; exit 2; }
[[ -n "$SRC_DIR" ]] || { echo "ERROR: --src-dir required" >&2; exit 2; }
[[ -d "$SRC_DIR" ]] || { echo "ERROR: src-dir not found: $SRC_DIR" >&2; exit 2; }

PREFIX="data/raw/${SLUG}"
[[ -n "$SUBFOLDER" ]] && PREFIX="${PREFIX}/${SUBFOLDER}"
DEST="s3://${BUCKET}/${PREFIX}"

echo "Bucket:    $BUCKET ($REGION)"
echo "Source:    $SRC_DIR  (glob: $GLOB)"
echo "Dest:      $DEST/"
echo "Dry run:   $DRY_RUN"
echo

count=0
uploaded=0
while IFS= read -r -d '' f; do
  count=$((count+1))
  base="$(basename "$f")"
  key="${PREFIX}/${base}"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] $base  ->  s3://${BUCKET}/${key}"
  else
    aws s3 cp "$f" "s3://${BUCKET}/${key}" \
      --region "$REGION" \
      --content-type application/pdf \
      --only-show-errors
    echo "uploaded: $base"
    uploaded=$((uploaded+1))
  fi
done < <(find "$SRC_DIR" -maxdepth 1 -type f -name "$GLOB" -print0)

echo
echo "Matched $count file(s); uploaded $uploaded."
if [[ "$DRY_RUN" -eq 0 && "$uploaded" -gt 0 ]]; then
  echo "Verify (one key): https://${BUCKET}/${PREFIX}/<filename>.pdf"
fi

#!/usr/bin/env bash
# Restore the three KV/MI5 collections deleted from prod on 2026-08-03
# (gouzenko, de_valera, colepaugh_gimpel) from the local archives.
#
#   PDFs      data/raw/<slug>/          (already present)
#   OCR cache ocr_cache/<slug>/         (so --ocr is free and fast)
#
# Usage:  bash scripts/restore_kv_collections.sh [slug ...]      (default: all three)
# Safe to re-run: ingest upserts on (collection, source_name).
set -euo pipefail
cd "$(dirname "$0")/.."

export AWS_SHARED_CREDENTIALS_FILE="$PWD/.aws/credentials"
export PATH="/opt/anaconda3/envs/friday/bin:$PATH"
export DATABASE_URL="$(aws secretsmanager get-secret-value --region us-west-1 \
  --secret-id friday/DATABASE_URL --query SecretString --output text)"
export EMBED_PROVIDER=openai
: "${OPENAI_API_KEY:="$(grep -E '^OPENAI_API_KEY=' .env | cut -d= -f2-)"}"
export OPENAI_API_KEY

title_gouzenko="Gouzenko, Igor, Canadian and British security service files"
desc_gouzenko="MI5 files on the CORBY case: the September 1945 defection of Igor Gouzenko, GRU cipher clerk at the Soviet Embassy in Ottawa, whose documents exposed Soviet military-intelligence networks in Canada, the UK and the US, including Alan Nunn May. Telegrams, RCMP and Royal Commission material, interrogation reports and correspondence, 1945 onward."

title_de_valera="De Valera, Eamon British Security Service-MI5 files (KV 2/514-515)"
desc_de_valera="MI5 personal files (PF 1, volumes 1-2) on Eamon de Valera: surveillance reports and correspondence on de Valera and Irish republican activity, including wartime Irish neutrality and contacts of intelligence interest. Note: part 4 of KV 2/515 was not included in the source set."

title_colepaugh_gimpel="Colepaugh, William and Gimpel, Erich British Security Service-MI5 files (KV 2/564)"
desc_colepaugh_gimpel="MI5 file on William Curtis Colepaugh and Erich Gimpel, German agents landed on the Maine coast from U-boat U-1230 in November 1944: telegrams, press cuttings, interrogation material, and liaison correspondence with SIS, BSC and the FBI."

slugs=("$@"); [ ${#slugs[@]} -eq 0 ] && slugs=(gouzenko de_valera colepaugh_gimpel)

for slug in "${slugs[@]}"; do
  t="title_$slug"; d="desc_$slug"; pv="${slug}_v1"
  echo "=== $slug — ingest + chunk"
  python -m scripts.ingest_dir_collection --dir "data/raw/$slug" --slug "$slug" \
    --title "${!t}" --description "${!d}" --pipeline-version "$pv" --ocr

  echo "=== $slug — chunk metadata"
  python scripts/build_chunk_metadata.py --chunk-pipeline "$pv" --collection-slug "$slug"

  echo "=== $slug — embeddings"
  python -m scripts.embed_silvermaster_chunks --chunk-pv "$pv" --collection-slug "$slug" \
    --prefer-clean-text --fill-missing-only

  echo "=== $slug — chunk metadata (annotate embeddings)"
  python scripts/build_chunk_metadata.py --chunk-pipeline "$pv" --collection-slug "$slug"

  echo "=== $slug — S3 upload"
  aws s3 cp "data/raw/$slug/" "s3://fridayarchive.org/data/raw/$slug/" \
    --recursive --region us-west-1 --content-type application/pdf

  echo "=== $slug — sizes + download zip"
  python scripts/backfill_document_sizes.py --only-missing
  python scripts/build_collection_zips.py --collections "$slug" --invalidate
done

echo "=== done; verifying"
python scripts/check_collection.py --list

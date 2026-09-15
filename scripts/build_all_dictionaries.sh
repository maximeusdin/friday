#!/bin/bash
# Build OCR-variant dictionaries (with skeletons) for every scanned collection.
# Skips: venona + vassiliev (born-digital transcriptions, no OCR damage),
#        silvermaster (already built 2026-08-04).
# Requires DATABASE_URL in the environment. Idempotent-ish: each run creates a
# new build; fuzzy_lex always uses the latest per (chunk_pv, slug, norm_version).
set -uo pipefail

PY=/opt/anaconda3/envs/friday/bin/python

PAIRS=(
  "solo solo_v1_memo"
  "solo solo_finding_aids_v1"
  "rosenberg rosenberg_v1"
  "siss_scope_soviet siss_scope_v1_turns"
  "judith_coplon judith_coplon_v1"
  "hiss_chambers hiss_chambers_v1"
  "thomas_black thomas_black_v1"
  "elizabeth_bentley elizabeth_bentley_v1"
  "mccarthy mccarthy_v2_turns"
  "morris_childs morris_childs_v1"
  "jack_childs jack_childs_v1"
  "david_greenglass david_greenglass_v1"
  "david_ruth_greenglass david_ruth_greenglass_v1"
  "huac_reports huac_reports_v1_pages"
  "huac_hearings huac_hearings_v1_turns"
  "oscar_seborer oscar_seborer_v1"
  "albertson albertson_v1"
  "smedley smedley_v1"
  "joel_barr joel_barr_v1"
  "winton_burdett winton_burdett_v1"
  "koval koval_pages_v1"
  "fbicomrap fbicomrap_v1"
  "emanuel_bloch emanuel_bloch_v1"
  "eva_childs eva_childs_v1"
  "fbi_hiskey fbi_hiskey_v1"
  "golos golos_v1"
  "rosenberg_grand_jury rosenberg_gj_v1"
  "brothman_moskowitz_grand_jury brothman_moskowitz_gj_v1"
  "soviet_atomic_espionage_1951 soviet_atomic_espionage_v1"
  "volodarsky volodarsky_v1"
  "witzak witzak_v1"
  "rosenberg_trial_transcripts rosenberg_trial_v1"
  "pravdin pravdin_v1"
  "soviet_intel_travel_techniques soviet_intel_travel_v1"
  "ruth_alscher ruth_alscher_v1"
  "fbi_cinrad fbi_cinrad_v1"
  "mink mink_v1"
  "arthur_barr arthur_barr_v1"
)

fail=0
for pair in "${PAIRS[@]}"; do
  read -r slug pv <<< "$pair"
  echo "=== $slug ($pv) ==="
  if ! "$PY" scripts/build_corpus_dictionary.py \
      --chunk-pv "$pv" --collection-slug "$slug" --include-raw-transcript; then
    echo "FAILED: $slug ($pv)" >&2
    fail=1
  fi
done
exit $fail

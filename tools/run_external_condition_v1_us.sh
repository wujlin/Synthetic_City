#!/usr/bin/env bash
set -euo pipefail

TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
RAW_ROOT="${RAW_ROOT:-/home/jinlin/data/geoexplicit_data}"
DATA_ROOT="${SYNTHCITY_DATA_ROOT:-$RAW_ROOT/synthetic_city/data}"
OUT_ROOT="${OUT_ROOT:-outputs}"
ACS_YEAR="${ACS_YEAR:-2022}"
PYTHON_BIN="${PYTHON_BIN:-python}"
STATEFPS="${STATEFPS:-}"
SCOPE_TAG="${SCOPE_TAG:-}"

RUN_DIR="${RUN_DIR:-$OUT_ROOT/_extcond_v1_build_us_puma_${TS}}"
if [[ -z "$SCOPE_TAG" ]]; then
  if [[ -n "$STATEFPS" ]]; then
    SCOPE_TAG="state${STATEFPS//,/_}"
  else
    SCOPE_TAG="us"
  fi
fi
OUT_CSV="${OUT_CSV:-$DATA_ROOT/us/processed/external_conditions/extcond_v1_acs5_${ACS_YEAR}_puma_${SCOPE_TAG}.csv}"
mkdir -p "$RUN_DIR"

echo ">>> [extcond-v1-us] DATA_ROOT=$DATA_ROOT"
echo ">>> [extcond-v1-us] RUN_DIR=$RUN_DIR"
echo ">>> [extcond-v1-us] OUT_CSV=$OUT_CSV"
echo ">>> [extcond-v1-us] PYTHON_BIN=$PYTHON_BIN"
echo ">>> [extcond-v1-us] ACS_YEAR=$ACS_YEAR"
echo ">>> [extcond-v1-us] STATEFPS=${STATEFPS:-ALL_50_STATES}"

if [[ -n "$STATEFPS" ]]; then
  "$PYTHON_BIN" -u tools/build_external_condition_v1_acs_puma.py \
    --acs_year "$ACS_YEAR" \
    --statefps "$STATEFPS" \
    --out_path "$OUT_CSV" \
    --overwrite 2>&1 | tee "$RUN_DIR/run.log"
else
  "$PYTHON_BIN" -u tools/build_external_condition_v1_acs_puma.py \
    --acs_year "$ACS_YEAR" \
    --all_states \
    --out_path "$OUT_CSV" \
    --overwrite 2>&1 | tee "$RUN_DIR/run.log"
fi

"$PYTHON_BIN" - <<PY
import json
from pathlib import Path

run_dir = Path("$RUN_DIR")
out_csv = Path("$OUT_CSV")
summary = {
    "task": "external_condition_v1_build",
    "schema": "external_condition_v1",
    "geo_level": "puma",
    "scope": "subset" if "$STATEFPS" else "us",
    "acs_year": int("$ACS_YEAR"),
    "python_bin": "$PYTHON_BIN",
    "data_root": "$DATA_ROOT",
    "statefps": "$STATEFPS".split(",") if "$STATEFPS" else None,
    "out_csv": str(out_csv),
    "metadata_json": str(out_csv.with_suffix(out_csv.suffix + ".metadata.json")),
}
run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
PY

echo ">>> [extcond-v1-us] artifacts"
ls -lh "$OUT_CSV" "$OUT_CSV.metadata.json" "$RUN_DIR/run_summary.json"

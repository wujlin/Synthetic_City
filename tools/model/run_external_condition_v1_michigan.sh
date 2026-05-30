#!/usr/bin/env bash
set -euo pipefail

# External-condition v1 builder for Michigan.
# Purpose:
# - freeze a clean, wsA-friendly build unit for the first condition-led experiment
# - produce a PUMA-level external condition file from ACS tables

TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
RAW_ROOT="${RAW_ROOT:-/home/jinlin/data/geoexplicit_data}"
DATA_ROOT="${SYNTHCITY_DATA_ROOT:-$RAW_ROOT/synthetic_city/data}"
OUT_ROOT="${OUT_ROOT:-outputs}"

RUN_DIR="${RUN_DIR:-$OUT_ROOT/_extcond_v1_build_mi_puma_${TS}}"
mkdir -p "$RUN_DIR"

ACS_YEAR="${ACS_YEAR:-2022}"
STATEFP="${STATEFP:-26}"
PYTHON_BIN="${PYTHON_BIN:-python}"

TRACT_PUMA_CSV="${TRACT_PUMA_CSV:-}"
TRACT_ZIP="${TRACT_ZIP:-$DATA_ROOT/detroit/raw/geo/tiger/TIGER2023/tl_2023_26_tract.zip}"
PUMA_ZIP="${PUMA_ZIP:-$DATA_ROOT/detroit/raw/geo/tiger/TIGER2023/tl_2023_26_puma20.zip}"

OUT_CSV="${OUT_CSV:-$DATA_ROOT/detroit/processed/external_conditions/extcond_v1_acs5_${ACS_YEAR}_puma_state${STATEFP}_michigan.csv}"

echo ">>> [extcond-v1] DATA_ROOT=$DATA_ROOT"
echo ">>> [extcond-v1] RUN_DIR=$RUN_DIR"
echo ">>> [extcond-v1] OUT_CSV=$OUT_CSV"
echo ">>> [extcond-v1] PYTHON_BIN=$PYTHON_BIN"

if [[ -n "$TRACT_PUMA_CSV" ]]; then
  echo ">>> [extcond-v1] use tract->puma csv: $TRACT_PUMA_CSV"
  "$PYTHON_BIN" -u tools/data/build_external_condition_v1_michigan.py \
    --data_root "$DATA_ROOT" \
    --acs_year "$ACS_YEAR" \
    --statefp "$STATEFP" \
    --aggregate_to puma \
    --tract_puma_csv "$TRACT_PUMA_CSV" \
    --out_path "$OUT_CSV" \
    --overwrite 2>&1 | tee "$RUN_DIR/run.log"
else
  echo ">>> [extcond-v1] build tract->puma map from TIGER zips"
  "$PYTHON_BIN" -u tools/data/build_external_condition_v1_michigan.py \
    --data_root "$DATA_ROOT" \
    --acs_year "$ACS_YEAR" \
    --statefp "$STATEFP" \
    --aggregate_to puma \
    --tract_zip "$TRACT_ZIP" \
    --puma_zip "$PUMA_ZIP" \
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
    "acs_year": int("$ACS_YEAR"),
    "statefp": "$STATEFP",
    "python_bin": "$PYTHON_BIN",
    "data_root": "$DATA_ROOT",
    "out_csv": str(out_csv),
    "metadata_json": str(out_csv.with_suffix(out_csv.suffix + ".metadata.json")),
    "tract_puma_csv": "$TRACT_PUMA_CSV" if "$TRACT_PUMA_CSV" else None,
    "tract_zip": "$TRACT_ZIP" if not "$TRACT_PUMA_CSV" else None,
    "puma_zip": "$PUMA_ZIP" if not "$TRACT_PUMA_CSV" else None,
}
run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
PY

echo ">>> [extcond-v1] artifacts"
ls -lh "$OUT_CSV" "$OUT_CSV.metadata.json" "$RUN_DIR/run_summary.json"

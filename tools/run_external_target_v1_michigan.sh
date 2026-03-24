#!/usr/bin/env bash
set -euo pipefail

RAW_ROOT="${RAW_ROOT:-/home/jinlin/data/geoexplicit_data}"
DATA_ROOT="${SYNTHCITY_DATA_ROOT:-$RAW_ROOT/synthetic_city/data}"
OUT_ROOT="${OUT_ROOT:-outputs}"
STATEFP="${STATEFP:-26}"
PUMS_YEAR="${PUMS_YEAR:-2023}"
PUMS_PERIOD="${PUMS_PERIOD:-5-Year}"
ALPHA="${ALPHA:-0.0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

PUMS_DIR="${PUMS_DIR:-$DATA_ROOT/detroit/raw/pums/pums_${PUMS_YEAR}_${PUMS_PERIOD}}"
CONDITION_CSV="${CONDITION_CSV:-$DATA_ROOT/detroit/processed/external_conditions/extcond_v1_acs5_2022_puma_state${STATEFP}_michigan.csv}"
TARGET_OUT_DIR="${TARGET_OUT_DIR:-$DATA_ROOT/detroit/processed/external_targets}"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$OUT_ROOT/_exttarget_v1_build_mi_puma_${TS}"
mkdir -p "$RUN_DIR"

{
  echo "[info] RAW_ROOT=$RAW_ROOT"
  echo "[info] DATA_ROOT=$DATA_ROOT"
  echo "[info] PUMS_DIR=$PUMS_DIR"
  echo "[info] CONDITION_CSV=$CONDITION_CSV"
  echo "[info] TARGET_OUT_DIR=$TARGET_OUT_DIR"
  echo "[info] PYTHON_BIN=$PYTHON_BIN"
  "$PYTHON_BIN" -u tools/build_external_target_v1_michigan.py \
    --statefp "$STATEFP" \
    --pums_year "$PUMS_YEAR" \
    --pums_period "$PUMS_PERIOD" \
    --pums_dir "$PUMS_DIR" \
    --alpha "$ALPHA" \
    --condition_csv "$CONDITION_CSV" \
    --out_dir "$TARGET_OUT_DIR"
} 2>&1 | tee "$RUN_DIR/run.log"

cat > "$RUN_DIR/run_summary.json" <<JSON
{
  "task": "external_target_v1_build",
  "schema": "external_target_v1",
  "statefp": "${STATEFP}",
  "pums_year": ${PUMS_YEAR},
  "pums_period": "${PUMS_PERIOD}",
  "alpha": ${ALPHA},
  "python_bin": "${PYTHON_BIN}",
  "raw_root": "${RAW_ROOT}",
  "data_root": "${DATA_ROOT}",
  "pums_dir": "${PUMS_DIR}",
  "condition_csv": "${CONDITION_CSV}",
  "target_out_dir": "${TARGET_OUT_DIR}",
  "run_dir": "${RUN_DIR}"
}
JSON

echo "[ok] run summary written to $RUN_DIR/run_summary.json"

#!/usr/bin/env bash
set -euo pipefail

RAW_ROOT="${RAW_ROOT:-/home/jinlin/data/geoexplicit_data}"
DATA_ROOT="${SYNTHCITY_DATA_ROOT:-$RAW_ROOT/synthetic_city/data}"
OUT_ROOT="${OUT_ROOT:-outputs}"
PUMS_YEAR="${PUMS_YEAR:-2023}"
PUMS_PERIOD="${PUMS_PERIOD:-5-Year}"
ALPHA="${ALPHA:-0.0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
STATEFPS="${STATEFPS:-}"
SCOPE_TAG="${SCOPE_TAG:-}"

if [[ -z "$SCOPE_TAG" ]]; then
  if [[ -n "$STATEFPS" ]]; then
    SCOPE_TAG="state${STATEFPS//,/_}"
  else
    SCOPE_TAG="us"
  fi
fi

PUMS_DIR="${PUMS_DIR:-$DATA_ROOT/us/raw/pums/pums_${PUMS_YEAR}_${PUMS_PERIOD}}"
CONDITION_CSV="${CONDITION_CSV:-$DATA_ROOT/us/processed/external_conditions/extcond_v1_acs5_2022_puma_${SCOPE_TAG}.csv}"
TARGET_OUT_DIR="${TARGET_OUT_DIR:-$DATA_ROOT/us/processed/external_targets}"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${RUN_DIR:-$OUT_ROOT/_exttarget_v1_build_us_puma_${TS}}"
mkdir -p "$RUN_DIR"

{
  echo "[info] RAW_ROOT=$RAW_ROOT"
  echo "[info] DATA_ROOT=$DATA_ROOT"
  echo "[info] PUMS_DIR=$PUMS_DIR"
  echo "[info] CONDITION_CSV=$CONDITION_CSV"
  echo "[info] TARGET_OUT_DIR=$TARGET_OUT_DIR"
  echo "[info] PYTHON_BIN=$PYTHON_BIN"
  echo "[info] STATEFPS=${STATEFPS:-ALL_50_STATES}"
  if [[ -n "$STATEFPS" ]]; then
    "$PYTHON_BIN" -u tools/data/build_external_target_v1_us.py \
      --statefps "$STATEFPS" \
      --pums_year "$PUMS_YEAR" \
      --pums_period "$PUMS_PERIOD" \
      --pums_dir "$PUMS_DIR" \
      --alpha "$ALPHA" \
      --condition_csv "$CONDITION_CSV" \
      --out_dir "$TARGET_OUT_DIR"
  else
    "$PYTHON_BIN" -u tools/data/build_external_target_v1_us.py \
      --all_states \
      --pums_year "$PUMS_YEAR" \
      --pums_period "$PUMS_PERIOD" \
      --pums_dir "$PUMS_DIR" \
      --alpha "$ALPHA" \
      --condition_csv "$CONDITION_CSV" \
      --out_dir "$TARGET_OUT_DIR"
  fi
} 2>&1 | tee "$RUN_DIR/run.log"

"$PYTHON_BIN" - <<PY
import json
from pathlib import Path

run_dir = Path("$RUN_DIR")
summary = {
    "task": "external_target_v1_build",
    "schema": "external_target_v1",
    "geo_level": "puma",
    "scope": "$SCOPE_TAG",
    "statefps": "$STATEFPS".split(",") if "$STATEFPS" else None,
    "pums_year": int("$PUMS_YEAR"),
    "pums_period": "$PUMS_PERIOD",
    "alpha": float("$ALPHA"),
    "python_bin": "$PYTHON_BIN",
    "raw_root": "$RAW_ROOT",
    "data_root": "$DATA_ROOT",
    "pums_dir": "$PUMS_DIR",
    "condition_csv": "$CONDITION_CSV",
    "target_out_dir": "$TARGET_OUT_DIR",
    "run_dir": "$RUN_DIR",
}
(run_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
PY

echo "[ok] run summary written to $RUN_DIR/run_summary.json"

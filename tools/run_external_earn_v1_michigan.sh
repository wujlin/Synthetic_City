#!/usr/bin/env bash
set -euo pipefail

TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_ROOT="${RAW_ROOT:-/home/jinlin/data/geoexplicit_data}"
DATA_ROOT="${SYNTHCITY_DATA_ROOT:-$RAW_ROOT/synthetic_city/data}"
OUT_ROOT="${OUT_ROOT:-outputs}"
PYTHON_BIN="${PYTHON_BIN:-python}"

ACS_YEAR="${ACS_YEAR:-2022}"
STATEFP="${STATEFP:-26}"
PUMS_YEAR="${PUMS_YEAR:-2023}"
PUMS_PERIOD="${PUMS_PERIOD:-5-Year}"
PUMS_DIR="${PUMS_DIR:-$DATA_ROOT/detroit/raw/pums/pums_${PUMS_YEAR}_5-Year}"

TRACT_PUMA_CSV="${TRACT_PUMA_CSV:-}"
TRACT_ZIP="${TRACT_ZIP:-$DATA_ROOT/detroit/raw/geo/tiger/TIGER2023/tl_2023_${STATEFP}_tract.zip}"
PUMA_ZIP="${PUMA_ZIP:-$DATA_ROOT/detroit/raw/geo/tiger/TIGER2023/tl_2023_${STATEFP}_puma20.zip}"

COND_OUT="${COND_OUT:-$DATA_ROOT/detroit/processed/external_conditions/extcond_earn_v1_acs5_${ACS_YEAR}_puma_state${STATEFP}_michigan.csv}"
TARGET_OUT_DIR="${TARGET_OUT_DIR:-$DATA_ROOT/detroit/processed/external_targets}"
RUN_DIR="${RUN_DIR:-$OUT_ROOT/_earn_v1_build_mi_puma_${TS}}"

mkdir -p "$RUN_DIR"
cd "$ROOT_DIR"

{
  echo "[info] ACS_YEAR=$ACS_YEAR"
  echo "[info] PUMS_YEAR=$PUMS_YEAR"
  echo "[info] DATA_ROOT=$DATA_ROOT"
  echo "[info] PYTHON_BIN=$PYTHON_BIN"
  echo "[info] PUMS_DIR=$PUMS_DIR"
  echo "[info] COND_OUT=$COND_OUT"
  echo "[info] TARGET_OUT_DIR=$TARGET_OUT_DIR"

  if [[ -n "$TRACT_PUMA_CSV" ]]; then
    "$PYTHON_BIN" -u tools/build_external_condition_earn_v1_michigan.py \
      --data_root "$DATA_ROOT" \
      --acs_year "$ACS_YEAR" \
      --statefp "$STATEFP" \
      --aggregate_to puma \
      --tract_puma_csv "$TRACT_PUMA_CSV" \
      --out_path "$COND_OUT" \
      --overwrite
  else
    "$PYTHON_BIN" -u tools/build_external_condition_earn_v1_michigan.py \
      --data_root "$DATA_ROOT" \
      --acs_year "$ACS_YEAR" \
      --statefp "$STATEFP" \
      --aggregate_to puma \
      --tract_zip "$TRACT_ZIP" \
      --puma_zip "$PUMA_ZIP" \
      --out_path "$COND_OUT" \
      --overwrite
  fi

  "$PYTHON_BIN" -u tools/build_external_target_earn_v1_michigan.py \
    --statefp "$STATEFP" \
    --pums_year "$PUMS_YEAR" \
    --pums_period "$PUMS_PERIOD" \
    --pums_dir "$PUMS_DIR" \
    --condition_csv "$COND_OUT" \
    --out_dir "$TARGET_OUT_DIR" \
    --overwrite
} 2>&1 | tee "$RUN_DIR/run.log"

"$PYTHON_BIN" - <<PY
import json
from pathlib import Path

run_dir = Path("$RUN_DIR")
cond_out = Path("$COND_OUT")
target_out_dir = Path("$TARGET_OUT_DIR")
statefp = str("$STATEFP").zfill(2)
pums_year = int("$PUMS_YEAR")
stem = f"exttarget_earn_v1_pums_{pums_year}_puma_state{statefp}_michigan"
summary = {
    "task": "external_earn_v1_build_michigan",
    "acs_year": int("$ACS_YEAR"),
    "pums_year": pums_year,
    "statefp": statefp,
    "data_root": "$DATA_ROOT",
    "pums_dir": "$PUMS_DIR",
    "condition_csv": str(cond_out),
    "condition_metadata_json": str(cond_out.with_suffix(cond_out.suffix + ".metadata.json")),
    "target_wide_csv": str(target_out_dir / f"{stem}.csv"),
    "target_long_csv": str(target_out_dir / f"{stem}_long.csv"),
    "target_schema_json": str(target_out_dir / f"{stem}.schema.json"),
    "target_metadata_json": str(target_out_dir / f"{stem}.metadata.json"),
    "tract_puma_csv": "$TRACT_PUMA_CSV" if "$TRACT_PUMA_CSV" else None,
    "tract_zip": "$TRACT_ZIP" if not "$TRACT_PUMA_CSV" else None,
    "puma_zip": "$PUMA_ZIP" if not "$TRACT_PUMA_CSV" else None,
}
run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
PY

echo ">>> [external-earn-v1] artifacts"
ls -lh "$RUN_DIR/run.log" "$RUN_DIR/run_summary.json" "$COND_OUT" "$COND_OUT.metadata.json"

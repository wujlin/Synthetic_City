#!/usr/bin/env bash
set -euo pipefail

TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
RAW_ROOT="${RAW_ROOT:-/home/jinlin/data/geoexplicit_data}"
DATA_ROOT="${SYNTHCITY_DATA_ROOT:-$RAW_ROOT/synthetic_city/data}"
OUT_ROOT="${OUT_ROOT:-outputs}"
PYTHON_BIN="${PYTHON_BIN:-python}"

VARIANT="${VARIANT:-age_refine}"
FULL_CONDITION_CSV="${FULL_CONDITION_CSV:-$DATA_ROOT/us/processed/external_conditions/extcond_v1_acs5_2022_puma_us.csv}"
FULL_TARGET_WIDE_CSV="${FULL_TARGET_WIDE_CSV:-$DATA_ROOT/us/processed/external_targets/exttarget_v1_pums_2023_puma_us_joint_wide.csv}"

RUN_DIR="${RUN_DIR:-$OUT_ROOT/_extv1_variant_build_${VARIANT}_${TS}}"
COND_OUT="${COND_OUT:-$DATA_ROOT/us/processed/external_conditions/extcond_v1_${VARIANT}_acs5_2022_puma_us.csv}"
TARGET_OUT_DIR="${TARGET_OUT_DIR:-$DATA_ROOT/us/processed/external_targets}"
mkdir -p "$RUN_DIR"

{
  echo "[info] VARIANT=$VARIANT"
  echo "[info] DATA_ROOT=$DATA_ROOT"
  echo "[info] PYTHON_BIN=$PYTHON_BIN"
  echo "[info] FULL_CONDITION_CSV=$FULL_CONDITION_CSV"
  echo "[info] FULL_TARGET_WIDE_CSV=$FULL_TARGET_WIDE_CSV"
  echo "[info] COND_OUT=$COND_OUT"
  echo "[info] TARGET_OUT_DIR=$TARGET_OUT_DIR"

  "$PYTHON_BIN" -u tools/data/build_external_condition_v1_variant.py \
    --condition_csv "$FULL_CONDITION_CSV" \
    --variant "$VARIANT" \
    --out_path "$COND_OUT" \
    --overwrite

  "$PYTHON_BIN" -u tools/data/build_external_target_v1_variant.py \
    --joint_wide_csv "$FULL_TARGET_WIDE_CSV" \
    --variant "$VARIANT" \
    --condition_csv "$COND_OUT" \
    --out_dir "$TARGET_OUT_DIR" \
    --overwrite
} 2>&1 | tee "$RUN_DIR/run.log"

"$PYTHON_BIN" - <<PY
import json
from pathlib import Path

run_dir = Path("$RUN_DIR")
cond_out = Path("$COND_OUT")
variant = "$VARIANT"
target_out_dir = Path("$TARGET_OUT_DIR")
target_stem = Path("$FULL_TARGET_WIDE_CSV").name.replace("exttarget_v1_", f"exttarget_v1_{variant}_").replace("_joint_wide.csv", "")
summary = {
    "task": "external_v1_variant_build",
    "variant": variant,
    "full_condition_csv": "$FULL_CONDITION_CSV",
    "full_target_wide_csv": "$FULL_TARGET_WIDE_CSV",
    "condition_csv": str(cond_out),
    "condition_metadata_json": str(cond_out.with_suffix(cond_out.suffix + ".metadata.json")),
    "target_wide_csv": str(target_out_dir / f"{target_stem}_joint_wide.csv"),
    "target_long_csv": str(target_out_dir / f"{target_stem}_joint_long.csv"),
    "schema_json": str(target_out_dir / f"{target_stem}.schema.json"),
    "target_metadata_json": str(target_out_dir / f"{target_stem}.metadata.json"),
}
run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
PY

echo ">>> [extv1-variant-us] artifacts"
ls -lh "$COND_OUT" "$COND_OUT.metadata.json" "$RUN_DIR/run_summary.json"


#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW_ROOT="${RAW_ROOT:-/home/jinlin/data/geoexplicit_data}"
DATA_ROOT="${DATA_ROOT:-$RAW_ROOT/synthetic_city/data}"
PYTHON_BIN="${PYTHON_BIN:-python}"

FULL_CONDITION_CSV="${FULL_CONDITION_CSV:-$DATA_ROOT/us/processed/external_conditions/extcond_v1_acs5_2022_puma_us.csv}"
FULL_TARGET_WIDE_CSV="${FULL_TARGET_WIDE_CSV:-$DATA_ROOT/us/processed/external_targets/exttarget_v1_pums_2023_puma_us_joint_wide.csv}"

COND_DIR="${COND_DIR:-$DATA_ROOT/us/processed/external_conditions}"
TARGET_DIR="${TARGET_DIR:-$DATA_ROOT/us/processed/external_targets}"
COND_VARIANT_CSV="${COND_VARIANT_CSV:-$COND_DIR/extcond_v1_full_acs5_2022_puma_us.csv}"
TARGET_VARIANT_WIDE_CSV="${TARGET_VARIANT_WIDE_CSV:-$TARGET_DIR/exttarget_v1_full_pums_2023_puma_us_joint_wide.csv}"
TARGET_VARIANT_SCHEMA_JSON="${TARGET_VARIANT_SCHEMA_JSON:-$TARGET_DIR/exttarget_v1_full_pums_2023_puma_us.schema.json}"

EPOCHS="${EPOCHS:-4000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
ENCODER_HIDDEN_DIMS="${ENCODER_HIDDEN_DIMS:-256,256}"
COARSE_HIDDEN_DIMS="${COARSE_HIDDEN_DIMS:-256}"
FINE_HIDDEN_DIMS="${FINE_HIDDEN_DIMS:-512,512}"
LATENT_DIM="${LATENT_DIM:-128}"
FINE_INPUT_MODE="${FINE_INPUT_MODE:-z_only}"
RUN_PREFIX="${RUN_PREFIX:-_us_puma_external_joint_hier_full_batch}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
COARSE_WEIGHT="${COARSE_WEIGHT:-0.5}"
CONSISTENCY_WEIGHT="${CONSISTENCY_WEIGHT:-1.0}"
LOG_EVERY="${LOG_EVERY:-200}"
IPF_ITERS="${IPF_ITERS:-200}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-0 1 2}"
BUILD_ONCE="${BUILD_ONCE:-1}"

cd "$ROOT_DIR"

read -r -a SEED_ARR <<< "$SEEDS"
RUN_DIRS=()

echo "[info] FULL_CONDITION_CSV=$FULL_CONDITION_CSV"
echo "[info] FULL_TARGET_WIDE_CSV=$FULL_TARGET_WIDE_CSV"
echo "[info] COND_VARIANT_CSV=$COND_VARIANT_CSV"
echo "[info] TARGET_VARIANT_WIDE_CSV=$TARGET_VARIANT_WIDE_CSV"
echo "[info] TARGET_VARIANT_SCHEMA_JSON=$TARGET_VARIANT_SCHEMA_JSON"
echo "[info] PYTHON_BIN=$PYTHON_BIN"
echo "[info] DEVICE=$DEVICE"
echo "[info] FINE_INPUT_MODE=$FINE_INPUT_MODE"
echo "[info] SEEDS=${SEED_ARR[*]}"

mkdir -p "$COND_DIR" "$TARGET_DIR"

if [[ "$BUILD_ONCE" == "1" ]]; then
  echo "[info] batch build once for full"
  "$PYTHON_BIN" -u tools/data/build_external_condition_v1_variant.py \
    --condition_csv "$FULL_CONDITION_CSV" \
    --variant full \
    --out_path "$COND_VARIANT_CSV" \
    --overwrite

  "$PYTHON_BIN" -u tools/data/build_external_target_v1_variant.py \
    --joint_wide_csv "$FULL_TARGET_WIDE_CSV" \
    --variant full \
    --condition_csv "$COND_VARIANT_CSV" \
    --out_dir "$TARGET_DIR" \
    --overwrite
fi

for seed in "${SEED_ARR[@]}"; do
  run_dir="outputs/${RUN_PREFIX}_${FINE_INPUT_MODE}_seed${seed}"
  RUN_DIRS+=("$run_dir")
  echo "[info] running seed=$seed -> $run_dir"
  RAW_ROOT="$RAW_ROOT" \
  DATA_ROOT="$DATA_ROOT" \
  PYTHON_BIN="$PYTHON_BIN" \
  FULL_CONDITION_CSV="$FULL_CONDITION_CSV" \
  FULL_TARGET_WIDE_CSV="$FULL_TARGET_WIDE_CSV" \
  COND_DIR="$COND_DIR" \
  TARGET_DIR="$TARGET_DIR" \
  COND_VARIANT_CSV="$COND_VARIANT_CSV" \
  TARGET_VARIANT_WIDE_CSV="$TARGET_VARIANT_WIDE_CSV" \
  TARGET_VARIANT_SCHEMA_JSON="$TARGET_VARIANT_SCHEMA_JSON" \
  EPOCHS="$EPOCHS" \
  BATCH_SIZE="$BATCH_SIZE" \
  ENCODER_HIDDEN_DIMS="$ENCODER_HIDDEN_DIMS" \
  COARSE_HIDDEN_DIMS="$COARSE_HIDDEN_DIMS" \
  FINE_HIDDEN_DIMS="$FINE_HIDDEN_DIMS" \
  LATENT_DIM="$LATENT_DIM" \
  FINE_INPUT_MODE="$FINE_INPUT_MODE" \
  RUN_DIR="$run_dir" \
  LR="$LR" \
  WEIGHT_DECAY="$WEIGHT_DECAY" \
  COARSE_WEIGHT="$COARSE_WEIGHT" \
  CONSISTENCY_WEIGHT="$CONSISTENCY_WEIGHT" \
  LOG_EVERY="$LOG_EVERY" \
  IPF_ITERS="$IPF_ITERS" \
  DEVICE="$DEVICE" \
  SEED="$seed" \
  SKIP_VARIANT_BUILD="1" \
  bash tools/model/run_external_joint_hier_full.sh
done

SUMMARY_OUT="outputs/${RUN_PREFIX}_summary"
"$PYTHON_BIN" -u tools/model/summarize_external_joint_hier_runs.py \
  --label external_joint_hier_full_batch \
  --run_dirs "${RUN_DIRS[@]}" \
  --out_dir "$SUMMARY_OUT"

echo "[ok] summary=$SUMMARY_OUT/summary.json"

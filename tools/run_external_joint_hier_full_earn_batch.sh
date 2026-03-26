#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_ROOT="${RAW_ROOT:-/home/jinlin/data/geoexplicit_data}"
DATA_ROOT="${DATA_ROOT:-$RAW_ROOT/synthetic_city/data}"
PYTHON_BIN="${PYTHON_BIN:-python}"

FULL_CONDITION_CSV="${FULL_CONDITION_CSV:-$DATA_ROOT/us/processed/external_conditions/extcond_v1_earn_v1_acs5_2022_puma_us.csv}"
COND_EARN_TARGET_CSV="${COND_EARN_TARGET_CSV:-$DATA_ROOT/us/processed/external_targets/exttarget_earn_cond_v1_pums_2023_puma_us.csv}"

TARGET_DIR="${TARGET_DIR:-$DATA_ROOT/us/processed/external_targets}"
TARGET_FULL5_WIDE_CSV="${TARGET_FULL5_WIDE_CSV:-$TARGET_DIR/exttarget_v1_full_earn_pums_2023_puma_us_joint_wide.csv}"
TARGET_FULL5_SCHEMA_JSON="${TARGET_FULL5_SCHEMA_JSON:-$TARGET_DIR/exttarget_v1_full_earn_pums_2023_puma_us.schema.json}"

EPOCHS="${EPOCHS:-1500}"
BATCH_SIZE="${BATCH_SIZE:-256}"
ENCODER_HIDDEN_DIMS="${ENCODER_HIDDEN_DIMS:-256,256}"
COARSE_HIDDEN_DIMS="${COARSE_HIDDEN_DIMS:-256}"
FINE_HIDDEN_DIMS="${FINE_HIDDEN_DIMS:-768,768}"
LATENT_DIM="${LATENT_DIM:-128}"
FINE_INPUT_MODE="${FINE_INPUT_MODE:-z_only}"
RUN_PREFIX="${RUN_PREFIX:-_us_puma_external_joint_hier_full_earn_batch}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
COARSE_WEIGHT="${COARSE_WEIGHT:-0.5}"
CONSISTENCY_WEIGHT="${CONSISTENCY_WEIGHT:-1.0}"
LOG_EVERY="${LOG_EVERY:-100}"
IPF_ITERS="${IPF_ITERS:-150}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-0 1 2}"
BUILD_ONCE="${BUILD_ONCE:-0}"

cd "$ROOT_DIR"

read -r -a SEED_ARR <<< "$SEEDS"
RUN_DIRS=()

echo "[info] FULL_CONDITION_CSV=$FULL_CONDITION_CSV"
echo "[info] COND_EARN_TARGET_CSV=$COND_EARN_TARGET_CSV"
echo "[info] TARGET_FULL5_WIDE_CSV=$TARGET_FULL5_WIDE_CSV"
echo "[info] TARGET_FULL5_SCHEMA_JSON=$TARGET_FULL5_SCHEMA_JSON"
echo "[info] PYTHON_BIN=$PYTHON_BIN"
echo "[info] DEVICE=$DEVICE"
echo "[info] FINE_INPUT_MODE=$FINE_INPUT_MODE"
echo "[info] SEEDS=${SEED_ARR[*]}"

mkdir -p "$TARGET_DIR"

if [[ "$BUILD_ONCE" == "1" ]]; then
  echo "[info] batch build once for full_earn"
  "$PYTHON_BIN" -u tools/build_external_target_v1_full_earn.py \
    --conditional_target_csv "$COND_EARN_TARGET_CSV" \
    --condition_csv "$FULL_CONDITION_CSV" \
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
  COND_EARN_TARGET_CSV="$COND_EARN_TARGET_CSV" \
  TARGET_DIR="$TARGET_DIR" \
  TARGET_FULL5_WIDE_CSV="$TARGET_FULL5_WIDE_CSV" \
  TARGET_FULL5_SCHEMA_JSON="$TARGET_FULL5_SCHEMA_JSON" \
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
  SKIP_TARGET_BUILD="1" \
  bash tools/run_external_joint_hier_full_earn.sh
done

SUMMARY_OUT="outputs/${RUN_PREFIX}_summary"
"$PYTHON_BIN" -u tools/summarize_external_joint_hier_runs.py \
  --label external_joint_hier_full_earn_batch \
  --run_dirs "${RUN_DIRS[@]}" \
  --out_dir "$SUMMARY_OUT"

echo "[ok] summary=$SUMMARY_OUT/summary.json"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW_ROOT="${RAW_ROOT:-/home/jinlin/data/geoexplicit_data}"
DATA_ROOT="${DATA_ROOT:-$RAW_ROOT/synthetic_city/data}"
PYTHON_BIN="${PYTHON_BIN:-python}"

FULL_CONDITION_CSV="${FULL_CONDITION_CSV:-$DATA_ROOT/us/processed/external_conditions/extcond_v1_acs5_2022_puma_us.csv}"
FULL_TARGET_WIDE_CSV="${FULL_TARGET_WIDE_CSV:-$DATA_ROOT/us/processed/external_targets/exttarget_v1_pums_2023_puma_us_joint_wide.csv}"
FULL_SCHEMA_JSON="${FULL_SCHEMA_JSON:-$DATA_ROOT/us/processed/external_targets/exttarget_v1_pums_2023_puma_us.schema.json}"

EARN_CONDITION_CSV="${EARN_CONDITION_CSV:-$DATA_ROOT/us/processed/external_conditions/extcond_earn_v1_acs5_2022_puma_us.csv}"
MERGED_CONDITION_CSV="${MERGED_CONDITION_CSV:-$DATA_ROOT/us/processed/external_conditions/extcond_v1_earn_v1_acs5_2022_puma_us.csv}"
EARN_TARGET_CSV="${EARN_TARGET_CSV:-$DATA_ROOT/us/processed/external_targets/exttarget_earn_v1_pums_2023_puma_us.csv}"
EARN_TARGET_OUT_DIR="${EARN_TARGET_OUT_DIR:-$DATA_ROOT/us/processed/external_targets}"
PUMS_YEAR="${PUMS_YEAR:-2023}"
PUMS_DIR="${PUMS_DIR:-$DATA_ROOT/us/raw/pums/pums_${PUMS_YEAR}_5-Year}"
ACS_YEAR="${ACS_YEAR:-2022}"

EPOCHS="${EPOCHS:-4000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
ENCODER_HIDDEN_DIMS="${ENCODER_HIDDEN_DIMS:-256,256}"
COARSE_HIDDEN_DIMS="${COARSE_HIDDEN_DIMS:-256}"
FINE_HIDDEN_DIMS="${FINE_HIDDEN_DIMS:-512,512}"
EARN_HIDDEN_DIMS="${EARN_HIDDEN_DIMS:-128,128}"
LATENT_DIM="${LATENT_DIM:-128}"
FINE_INPUT_MODE="${FINE_INPUT_MODE:-z_only}"
RUN_PREFIX="${RUN_PREFIX:-_us_puma_external_joint_hier_full_earn_aux_batch}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
COARSE_WEIGHT="${COARSE_WEIGHT:-0.5}"
CONSISTENCY_WEIGHT="${CONSISTENCY_WEIGHT:-1.0}"
EARN_WEIGHT="${EARN_WEIGHT:-1.0}"
LOG_EVERY="${LOG_EVERY:-200}"
IPF_ITERS="${IPF_ITERS:-200}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-0 1 2}"
BUILD_EARN_CONDITION_ONCE="${BUILD_EARN_CONDITION_ONCE:-0}"
BUILD_EARN_TARGET_ONCE="${BUILD_EARN_TARGET_ONCE:-0}"
BUILD_MERGED_CONDITION_ONCE="${BUILD_MERGED_CONDITION_ONCE:-0}"

cd "$ROOT_DIR"
read -r -a SEED_ARR <<< "$SEEDS"
RUN_DIRS=()

echo "[info] PYTHON_BIN=$PYTHON_BIN"
echo "[info] FULL_CONDITION_CSV=$FULL_CONDITION_CSV"
echo "[info] FULL_TARGET_WIDE_CSV=$FULL_TARGET_WIDE_CSV"
echo "[info] FULL_SCHEMA_JSON=$FULL_SCHEMA_JSON"
echo "[info] EARN_CONDITION_CSV=$EARN_CONDITION_CSV"
echo "[info] MERGED_CONDITION_CSV=$MERGED_CONDITION_CSV"
echo "[info] EARN_TARGET_CSV=$EARN_TARGET_CSV"
echo "[info] FINE_INPUT_MODE=$FINE_INPUT_MODE"
echo "[info] DEVICE=$DEVICE"
echo "[info] SEEDS=${SEED_ARR[*]}"

for seed in "${SEED_ARR[@]}"; do
  run_dir="outputs/${RUN_PREFIX}_${FINE_INPUT_MODE}_seed${seed}"
  RUN_DIRS+=("$run_dir")
  echo "[info] running seed=$seed -> $run_dir"
  RAW_ROOT="$RAW_ROOT" \
  DATA_ROOT="$DATA_ROOT" \
  PYTHON_BIN="$PYTHON_BIN" \
  FULL_CONDITION_CSV="$FULL_CONDITION_CSV" \
  FULL_TARGET_WIDE_CSV="$FULL_TARGET_WIDE_CSV" \
  FULL_SCHEMA_JSON="$FULL_SCHEMA_JSON" \
  EARN_CONDITION_CSV="$EARN_CONDITION_CSV" \
  MERGED_CONDITION_CSV="$MERGED_CONDITION_CSV" \
  EARN_TARGET_CSV="$EARN_TARGET_CSV" \
  EARN_TARGET_OUT_DIR="$EARN_TARGET_OUT_DIR" \
  PUMS_YEAR="$PUMS_YEAR" \
  PUMS_DIR="$PUMS_DIR" \
  ACS_YEAR="$ACS_YEAR" \
  EPOCHS="$EPOCHS" \
  BATCH_SIZE="$BATCH_SIZE" \
  ENCODER_HIDDEN_DIMS="$ENCODER_HIDDEN_DIMS" \
  COARSE_HIDDEN_DIMS="$COARSE_HIDDEN_DIMS" \
  FINE_HIDDEN_DIMS="$FINE_HIDDEN_DIMS" \
  EARN_HIDDEN_DIMS="$EARN_HIDDEN_DIMS" \
  LATENT_DIM="$LATENT_DIM" \
  FINE_INPUT_MODE="$FINE_INPUT_MODE" \
  RUN_DIR="$run_dir" \
  LR="$LR" \
  WEIGHT_DECAY="$WEIGHT_DECAY" \
  COARSE_WEIGHT="$COARSE_WEIGHT" \
  CONSISTENCY_WEIGHT="$CONSISTENCY_WEIGHT" \
  EARN_WEIGHT="$EARN_WEIGHT" \
  LOG_EVERY="$LOG_EVERY" \
  IPF_ITERS="$IPF_ITERS" \
  DEVICE="$DEVICE" \
  SEED="$seed" \
  BUILD_EARN_CONDITION_ONCE="$BUILD_EARN_CONDITION_ONCE" \
  BUILD_EARN_TARGET_ONCE="$BUILD_EARN_TARGET_ONCE" \
  BUILD_MERGED_CONDITION_ONCE="$BUILD_MERGED_CONDITION_ONCE" \
  bash tools/model/run_external_joint_hier_full_earn_aux.sh
done

SUMMARY_OUT="outputs/${RUN_PREFIX}_summary"
"$PYTHON_BIN" -u tools/model/summarize_external_joint_hier_full_earn_aux_runs.py \
  --label external_joint_hier_full_earn_aux_batch \
  --run_dirs "${RUN_DIRS[@]}" \
  --out_dir "$SUMMARY_OUT"

echo "[ok] summary=$SUMMARY_OUT/summary.json"

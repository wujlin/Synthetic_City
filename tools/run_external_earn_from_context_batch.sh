#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_ROOT="${RAW_ROOT:-/home/jinlin/data/geoexplicit_data}"
DATA_ROOT="${DATA_ROOT:-$RAW_ROOT/synthetic_city/data}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CONDITION_CSV="${CONDITION_CSV:-$DATA_ROOT/us/processed/external_conditions/extcond_v1_acs5_2022_puma_us.csv}"
TARGET_CSV="${TARGET_CSV:-$DATA_ROOT/us/processed/external_targets/exttarget_earn_v1_pums_2023_puma_us.csv}"
TARGET_OUT_DIR="${TARGET_OUT_DIR:-$DATA_ROOT/us/processed/external_targets}"
PUMS_YEAR="${PUMS_YEAR:-2023}"
PUMS_DIR="${PUMS_DIR:-$DATA_ROOT/us/raw/pums/pums_${PUMS_YEAR}_5-Year}"

EPOCHS="${EPOCHS:-3000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
ENCODER_HIDDEN_DIMS="${ENCODER_HIDDEN_DIMS:-256,256}"
HEAD_HIDDEN_DIMS="${HEAD_HIDDEN_DIMS:-128,128}"
LATENT_DIM="${LATENT_DIM:-64}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
LOG_EVERY="${LOG_EVERY:-200}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-0 1 2}"
RUN_PREFIX="${RUN_PREFIX:-_us_puma_external_earn_from_context_batch}"
BUILD_TARGET_ONCE="${BUILD_TARGET_ONCE:-0}"

cd "$ROOT_DIR"
read -r -a SEED_ARR <<< "$SEEDS"
RUN_DIRS=()

echo "[info] PYTHON_BIN=$PYTHON_BIN"
echo "[info] CONDITION_CSV=$CONDITION_CSV"
echo "[info] TARGET_CSV=$TARGET_CSV"
echo "[info] PUMS_DIR=$PUMS_DIR"
echo "[info] BUILD_TARGET_ONCE=$BUILD_TARGET_ONCE"
echo "[info] SEEDS=${SEED_ARR[*]}"

if [[ "$BUILD_TARGET_ONCE" == "1" ]]; then
  "$PYTHON_BIN" -u tools/build_external_target_earn_v1_us.py \
    --all_states \
    --pums_year "$PUMS_YEAR" \
    --pums_dir "$PUMS_DIR" \
    --out_dir "$TARGET_OUT_DIR"
fi

for seed in "${SEED_ARR[@]}"; do
  run_dir="outputs/${RUN_PREFIX}_seed${seed}"
  RUN_DIRS+=("$run_dir")
  echo "[info] running seed=$seed -> $run_dir"
  BUILD_TARGET_ONCE=0 \
  CONDITION_CSV="$CONDITION_CSV" \
  TARGET_CSV="$TARGET_CSV" \
  TARGET_OUT_DIR="$TARGET_OUT_DIR" \
  PUMS_YEAR="$PUMS_YEAR" \
  PUMS_DIR="$PUMS_DIR" \
  EPOCHS="$EPOCHS" \
  BATCH_SIZE="$BATCH_SIZE" \
  ENCODER_HIDDEN_DIMS="$ENCODER_HIDDEN_DIMS" \
  HEAD_HIDDEN_DIMS="$HEAD_HIDDEN_DIMS" \
  LATENT_DIM="$LATENT_DIM" \
  LR="$LR" \
  WEIGHT_DECAY="$WEIGHT_DECAY" \
  LOG_EVERY="$LOG_EVERY" \
  DEVICE="$DEVICE" \
  SEED="$seed" \
  RUN_DIR="$run_dir" \
  PYTHON_BIN="$PYTHON_BIN" \
  bash tools/run_external_earn_from_context.sh
done

SUMMARY_OUT="outputs/${RUN_PREFIX}_summary"
"$PYTHON_BIN" -u tools/summarize_external_earn_from_context_runs.py \
  --label external_earn_from_context_batch \
  --run_dirs "${RUN_DIRS[@]}" \
  --out_dir "$SUMMARY_OUT"

echo "[ok] summary=$SUMMARY_OUT/summary.json"

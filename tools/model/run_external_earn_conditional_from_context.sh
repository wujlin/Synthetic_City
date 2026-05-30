#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW_ROOT="${RAW_ROOT:-/home/jinlin/data/geoexplicit_data}"
DATA_ROOT="${DATA_ROOT:-$RAW_ROOT/synthetic_city/data}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CONDITION_CSV="${CONDITION_CSV:-$DATA_ROOT/us/processed/external_conditions/extcond_v1_earn_v1_acs5_2022_puma_us.csv}"
TARGET_CSV="${TARGET_CSV:-$DATA_ROOT/us/processed/external_targets/exttarget_earn_cond_v1_pums_2023_puma_us.csv}"
TARGET_OUT_DIR="${TARGET_OUT_DIR:-$DATA_ROOT/us/processed/external_targets}"
PUMS_YEAR="${PUMS_YEAR:-2023}"
PUMS_DIR="${PUMS_DIR:-$DATA_ROOT/us/raw/pums/pums_${PUMS_YEAR}_5-Year}"

CONDITION_MODE="${CONDITION_MODE:-merged5}"
EPOCHS="${EPOCHS:-3000}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
ENCODER_HIDDEN_DIMS="${ENCODER_HIDDEN_DIMS:-256,256}"
HEAD_HIDDEN_DIMS="${HEAD_HIDDEN_DIMS:-256,256}"
LATENT_DIM="${LATENT_DIM:-64}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
LOG_EVERY="${LOG_EVERY:-200}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
RUN_DIR="${RUN_DIR:-outputs/_us_puma_external_earn_conditional_${CONDITION_MODE}_$(date -u +%Y%m%dT%H%M%SZ)}"
BUILD_TARGET_ONCE="${BUILD_TARGET_ONCE:-0}"

cd "$ROOT_DIR"

echo "[info] PYTHON_BIN=$PYTHON_BIN"
echo "[info] CONDITION_CSV=$CONDITION_CSV"
echo "[info] TARGET_CSV=$TARGET_CSV"
echo "[info] CONDITION_MODE=$CONDITION_MODE"
echo "[info] PUMS_DIR=$PUMS_DIR"
echo "[info] DEVICE=$DEVICE"

mkdir -p "$TARGET_OUT_DIR"

if [[ "$BUILD_TARGET_ONCE" == "1" ]]; then
  "$PYTHON_BIN" -u tools/data/build_external_target_earn_conditional_v1_us.py \
    --all_states \
    --pums_year "$PUMS_YEAR" \
    --pums_dir "$PUMS_DIR" \
    --out_dir "$TARGET_OUT_DIR"
fi

"$PYTHON_BIN" -u tools/model/train_external_earn_conditional_from_context.py \
  --target_csv "$TARGET_CSV" \
  --condition_csv "$CONDITION_CSV" \
  --condition_mode "$CONDITION_MODE" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --encoder_hidden_dims "$ENCODER_HIDDEN_DIMS" \
  --head_hidden_dims "$HEAD_HIDDEN_DIMS" \
  --latent_dim "$LATENT_DIM" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --log_every "$LOG_EVERY" \
  --save_final_model \
  --out_dir "$RUN_DIR"

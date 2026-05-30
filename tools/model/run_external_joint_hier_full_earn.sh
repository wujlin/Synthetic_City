#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
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
RUN_DIR="${RUN_DIR:-outputs/_us_puma_external_joint_hier_full_earn_${FINE_INPUT_MODE}_$(date -u +%Y%m%dT%H%M%SZ)}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
COARSE_WEIGHT="${COARSE_WEIGHT:-0.5}"
CONSISTENCY_WEIGHT="${CONSISTENCY_WEIGHT:-1.0}"
LOG_EVERY="${LOG_EVERY:-100}"
IPF_ITERS="${IPF_ITERS:-150}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
SKIP_TARGET_BUILD="${SKIP_TARGET_BUILD:-0}"

cd "$ROOT_DIR"

echo "[info] FULL_CONDITION_CSV=$FULL_CONDITION_CSV"
echo "[info] COND_EARN_TARGET_CSV=$COND_EARN_TARGET_CSV"
echo "[info] TARGET_FULL5_WIDE_CSV=$TARGET_FULL5_WIDE_CSV"
echo "[info] TARGET_FULL5_SCHEMA_JSON=$TARGET_FULL5_SCHEMA_JSON"
echo "[info] PYTHON_BIN=$PYTHON_BIN"
echo "[info] DEVICE=$DEVICE"
echo "[info] FINE_INPUT_MODE=$FINE_INPUT_MODE"
echo "[info] SKIP_TARGET_BUILD=$SKIP_TARGET_BUILD"

mkdir -p "$TARGET_DIR"

if [[ "$SKIP_TARGET_BUILD" != "1" ]]; then
  "$PYTHON_BIN" -u tools/data/build_external_target_v1_full_earn.py \
    --conditional_target_csv "$COND_EARN_TARGET_CSV" \
    --condition_csv "$FULL_CONDITION_CSV" \
    --out_dir "$TARGET_DIR" \
    --overwrite
fi

"$PYTHON_BIN" -u tools/model/train_external_joint_hier_full_earn.py \
  --joint_wide_csv "$TARGET_FULL5_WIDE_CSV" \
  --condition_csv "$FULL_CONDITION_CSV" \
  --schema_json "$TARGET_FULL5_SCHEMA_JSON" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --encoder_hidden_dims "$ENCODER_HIDDEN_DIMS" \
  --coarse_hidden_dims "$COARSE_HIDDEN_DIMS" \
  --fine_hidden_dims "$FINE_HIDDEN_DIMS" \
  --latent_dim "$LATENT_DIM" \
  --fine_input_mode "$FINE_INPUT_MODE" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --coarse_weight "$COARSE_WEIGHT" \
  --consistency_weight "$CONSISTENCY_WEIGHT" \
  --log_every "$LOG_EVERY" \
  --ipf_iters "$IPF_ITERS" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --save_final_model \
  --run_label "external_joint_hier_full_earn" \
  --out_dir "$RUN_DIR"

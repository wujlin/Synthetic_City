#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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
RUN_DIR="${RUN_DIR:-outputs/_us_puma_external_joint_hier_full_earn_aux_${FINE_INPUT_MODE}_$(date -u +%Y%m%dT%H%M%SZ)}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
COARSE_WEIGHT="${COARSE_WEIGHT:-0.5}"
CONSISTENCY_WEIGHT="${CONSISTENCY_WEIGHT:-1.0}"
EARN_WEIGHT="${EARN_WEIGHT:-1.0}"
LOG_EVERY="${LOG_EVERY:-200}"
IPF_ITERS="${IPF_ITERS:-200}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
BUILD_EARN_CONDITION_ONCE="${BUILD_EARN_CONDITION_ONCE:-1}"
BUILD_EARN_TARGET_ONCE="${BUILD_EARN_TARGET_ONCE:-0}"
BUILD_MERGED_CONDITION_ONCE="${BUILD_MERGED_CONDITION_ONCE:-1}"

cd "$ROOT_DIR"

echo "[info] FULL_CONDITION_CSV=$FULL_CONDITION_CSV"
echo "[info] EARN_CONDITION_CSV=$EARN_CONDITION_CSV"
echo "[info] MERGED_CONDITION_CSV=$MERGED_CONDITION_CSV"
echo "[info] FULL_TARGET_WIDE_CSV=$FULL_TARGET_WIDE_CSV"
echo "[info] FULL_SCHEMA_JSON=$FULL_SCHEMA_JSON"
echo "[info] EARN_TARGET_CSV=$EARN_TARGET_CSV"
echo "[info] PYTHON_BIN=$PYTHON_BIN"
echo "[info] DEVICE=$DEVICE"
echo "[info] FINE_INPUT_MODE=$FINE_INPUT_MODE"

mkdir -p "$(dirname "$MERGED_CONDITION_CSV")" "$EARN_TARGET_OUT_DIR"

if [[ "$BUILD_EARN_CONDITION_ONCE" == "1" ]]; then
  "$PYTHON_BIN" -u tools/build_external_condition_earn_v1_acs_puma.py \
    --all_states \
    --acs_year "$ACS_YEAR" \
    --out_path "$EARN_CONDITION_CSV" \
    --overwrite
fi

if [[ "$BUILD_EARN_TARGET_ONCE" == "1" ]]; then
  "$PYTHON_BIN" -u tools/build_external_target_earn_v1_us.py \
    --all_states \
    --pums_year "$PUMS_YEAR" \
    --pums_dir "$PUMS_DIR" \
    --out_dir "$EARN_TARGET_OUT_DIR"
fi

if [[ "$BUILD_MERGED_CONDITION_ONCE" == "1" ]]; then
  "$PYTHON_BIN" -u tools/merge_external_condition_v1_with_earn.py \
    --base_condition_csv "$FULL_CONDITION_CSV" \
    --earn_condition_csv "$EARN_CONDITION_CSV" \
    --out_path "$MERGED_CONDITION_CSV" \
    --overwrite
fi

"$PYTHON_BIN" -u tools/train_external_joint_hier_full_earn_aux.py \
  --joint_wide_csv "$FULL_TARGET_WIDE_CSV" \
  --condition_csv "$MERGED_CONDITION_CSV" \
  --schema_json "$FULL_SCHEMA_JSON" \
  --earn_target_csv "$EARN_TARGET_CSV" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --encoder_hidden_dims "$ENCODER_HIDDEN_DIMS" \
  --coarse_hidden_dims "$COARSE_HIDDEN_DIMS" \
  --fine_hidden_dims "$FINE_HIDDEN_DIMS" \
  --earn_hidden_dims "$EARN_HIDDEN_DIMS" \
  --latent_dim "$LATENT_DIM" \
  --fine_input_mode "$FINE_INPUT_MODE" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --coarse_weight "$COARSE_WEIGHT" \
  --consistency_weight "$CONSISTENCY_WEIGHT" \
  --earn_weight "$EARN_WEIGHT" \
  --log_every "$LOG_EVERY" \
  --ipf_iters "$IPF_ITERS" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --save_final_model \
  --out_dir "$RUN_DIR"

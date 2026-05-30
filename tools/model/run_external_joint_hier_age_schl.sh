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
COND_VARIANT_CSV="${COND_VARIANT_CSV:-$COND_DIR/extcond_v1_age_schl_refine_acs5_2022_puma_us.csv}"
TARGET_VARIANT_WIDE_CSV="${TARGET_VARIANT_WIDE_CSV:-$TARGET_DIR/exttarget_v1_age_schl_refine_pums_2023_puma_us_joint_wide.csv}"
TARGET_VARIANT_SCHEMA_JSON="${TARGET_VARIANT_SCHEMA_JSON:-$TARGET_DIR/exttarget_v1_age_schl_refine_pums_2023_puma_us.schema.json}"

EPOCHS="${EPOCHS:-4000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
ENCODER_HIDDEN_DIMS="${ENCODER_HIDDEN_DIMS:-256,256}"
COARSE_HIDDEN_DIMS="${COARSE_HIDDEN_DIMS:-256}"
FINE_HIDDEN_DIMS="${FINE_HIDDEN_DIMS:-512,512}"
LATENT_DIM="${LATENT_DIM:-128}"
FINE_INPUT_MODE="${FINE_INPUT_MODE:-z_coarse_prob}"
RUN_DIR="${RUN_DIR:-outputs/_us_puma_external_joint_hier_age_schl_${FINE_INPUT_MODE}_$(date -u +%Y%m%dT%H%M%SZ)}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
COARSE_WEIGHT="${COARSE_WEIGHT:-0.5}"
CONSISTENCY_WEIGHT="${CONSISTENCY_WEIGHT:-1.0}"
LOG_EVERY="${LOG_EVERY:-200}"
IPF_ITERS="${IPF_ITERS:-200}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
SKIP_VARIANT_BUILD="${SKIP_VARIANT_BUILD:-0}"

cd "$ROOT_DIR"

echo "[info] FULL_CONDITION_CSV=$FULL_CONDITION_CSV"
echo "[info] FULL_TARGET_WIDE_CSV=$FULL_TARGET_WIDE_CSV"
echo "[info] COND_VARIANT_CSV=$COND_VARIANT_CSV"
echo "[info] TARGET_VARIANT_WIDE_CSV=$TARGET_VARIANT_WIDE_CSV"
echo "[info] TARGET_VARIANT_SCHEMA_JSON=$TARGET_VARIANT_SCHEMA_JSON"
echo "[info] PYTHON_BIN=$PYTHON_BIN"
echo "[info] DEVICE=$DEVICE"
echo "[info] FINE_INPUT_MODE=$FINE_INPUT_MODE"
echo "[info] SKIP_VARIANT_BUILD=$SKIP_VARIANT_BUILD"

mkdir -p "$COND_DIR" "$TARGET_DIR"

if [[ "$SKIP_VARIANT_BUILD" != "1" ]]; then
  "$PYTHON_BIN" -u tools/data/build_external_condition_v1_variant.py \
    --condition_csv "$FULL_CONDITION_CSV" \
    --variant age_schl_refine \
    --out_path "$COND_VARIANT_CSV" \
    --overwrite

  "$PYTHON_BIN" -u tools/data/build_external_target_v1_variant.py \
    --joint_wide_csv "$FULL_TARGET_WIDE_CSV" \
    --variant age_schl_refine \
    --condition_csv "$COND_VARIANT_CSV" \
    --out_dir "$TARGET_DIR" \
    --overwrite
fi

"$PYTHON_BIN" -u tools/model/train_external_joint_hier_age_schl.py \
  --joint_wide_csv "$TARGET_VARIANT_WIDE_CSV" \
  --condition_csv "$COND_VARIANT_CSV" \
  --schema_json "$TARGET_VARIANT_SCHEMA_JSON" \
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
  --out_dir "$RUN_DIR"

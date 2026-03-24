#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_ROOT="${RAW_ROOT:-/home/jinlin/data/geoexplicit_data}"
DATA_ROOT="${DATA_ROOT:-$RAW_ROOT/synthetic_city/data}"
PYTHON_BIN="${PYTHON_BIN:-python}"

STAGE1_JOINT_WIDE_CSV="${STAGE1_JOINT_WIDE_CSV:-$DATA_ROOT/us/processed/external_targets/exttarget_v1_lite_pums_2023_puma_us_joint_wide.csv}"
STAGE1_SCHEMA_JSON="${STAGE1_SCHEMA_JSON:-$DATA_ROOT/us/processed/external_targets/exttarget_v1_lite_pums_2023_puma_us.schema.json}"
STAGE1_CONDITION_CSV="${STAGE1_CONDITION_CSV:-$DATA_ROOT/us/processed/external_conditions/extcond_v1_lite_acs5_2022_puma_us.csv}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-outputs/_us_puma_external_v1_lite_retry_20260323T075526Z/checkpoints/external/leave_mi_out/final.pt}"

FINAL_TARGET_WIDE_CSV="${FINAL_TARGET_WIDE_CSV:-$DATA_ROOT/us/processed/external_targets/exttarget_v1_age_schl_refine_pums_2023_puma_us_joint_wide.csv}"
FINAL_TARGET_SCHEMA_JSON="${FINAL_TARGET_SCHEMA_JSON:-$DATA_ROOT/us/processed/external_targets/exttarget_v1_age_schl_refine_pums_2023_puma_us.schema.json}"

C2F_DIR="${C2F_DIR:-$DATA_ROOT/us/processed/external_c2f}"
C2F_WIDE_CSV="${C2F_WIDE_CSV:-$C2F_DIR/extc2f_age_schl_exposure_pums_2023_puma_us_wide.csv}"
C2F_SCHEMA_JSON="${C2F_SCHEMA_JSON:-$C2F_DIR/extc2f_age_schl_exposure_pums_2023_puma_us.schema.json}"
RUN_DIR="${RUN_DIR:-outputs/_us_puma_external_c2f_age_schl_exposure_$(date -u +%Y%m%dT%H%M%SZ)}"

EPOCHS="${EPOCHS:-3000}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
LOG_EVERY="${LOG_EVERY:-200}"
N_EVAL_JOINT_SAMPLES="${N_EVAL_JOINT_SAMPLES:-64}"
HIDDEN_DIMS="${HIDDEN_DIMS:-256,256}"
CONDITION_INJECTION="${CONDITION_INJECTION:-concat}"
FILM_HIDDEN_DIM="${FILM_HIDDEN_DIM:-128}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
IPF_ITERS="${IPF_ITERS:-200}"

cd "$ROOT_DIR"

echo "[info] STAGE1_JOINT_WIDE_CSV=$STAGE1_JOINT_WIDE_CSV"
echo "[info] STAGE1_SCHEMA_JSON=$STAGE1_SCHEMA_JSON"
echo "[info] STAGE1_CONDITION_CSV=$STAGE1_CONDITION_CSV"
echo "[info] STAGE1_CHECKPOINT=$STAGE1_CHECKPOINT"
echo "[info] FINAL_TARGET_WIDE_CSV=$FINAL_TARGET_WIDE_CSV"
echo "[info] FINAL_TARGET_SCHEMA_JSON=$FINAL_TARGET_SCHEMA_JSON"
echo "[info] C2F_DIR=$C2F_DIR"
echo "[info] PYTHON_BIN=$PYTHON_BIN"
echo "[info] DEVICE=$DEVICE"

mkdir -p "$C2F_DIR"

"$PYTHON_BIN" -u tools/build_external_c2f_age_schl_exposure.py \
  --stage1_joint_wide_csv "$STAGE1_JOINT_WIDE_CSV" \
  --stage1_schema_json "$STAGE1_SCHEMA_JSON" \
  --stage1_condition_csv "$STAGE1_CONDITION_CSV" \
  --stage1_checkpoint "$STAGE1_CHECKPOINT" \
  --final_target_wide_csv "$FINAL_TARGET_WIDE_CSV" \
  --final_target_schema_json "$FINAL_TARGET_SCHEMA_JSON" \
  --n_eval_joint_samples "$N_EVAL_JOINT_SAMPLES" \
  --ipf_iters "$IPF_ITERS" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --out_dir "$C2F_DIR" \
  --overwrite

"$PYTHON_BIN" -u tools/train_external_c2f_age_schl_teacher.py \
  --wide_csv "$C2F_WIDE_CSV" \
  --schema_json "$C2F_SCHEMA_JSON" \
  --timesteps 1000 \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --hidden_dims "$HIDDEN_DIMS" \
  --condition_injection "$CONDITION_INJECTION" \
  --film_hidden_dim "$FILM_HIDDEN_DIM" \
  --seed "$SEED" \
  --log_every "$LOG_EVERY" \
  --n_eval_joint_samples "$N_EVAL_JOINT_SAMPLES" \
  --device "$DEVICE" \
  --save_final_model \
  --out_dir "$RUN_DIR"

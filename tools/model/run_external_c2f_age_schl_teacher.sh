#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW_ROOT="${RAW_ROOT:-/home/jinlin/data/geoexplicit_data}"
DATA_ROOT="${DATA_ROOT:-$RAW_ROOT/synthetic_city/data}"
PYTHON_BIN="${PYTHON_BIN:-python}"

FULL_TARGET_WIDE_CSV="${FULL_TARGET_WIDE_CSV:-$DATA_ROOT/us/processed/external_targets/exttarget_v1_pums_2023_puma_us_joint_wide.csv}"
C2F_DIR="${C2F_DIR:-$DATA_ROOT/us/processed/external_c2f}"
C2F_WIDE_CSV="${C2F_WIDE_CSV:-$C2F_DIR/extc2f_age_schl_teacher_pums_2023_puma_us_wide.csv}"
C2F_SCHEMA_JSON="${C2F_SCHEMA_JSON:-$C2F_DIR/extc2f_age_schl_teacher_pums_2023_puma_us.schema.json}"
RUN_DIR="${RUN_DIR:-outputs/_us_puma_external_c2f_age_schl_teacher_$(date -u +%Y%m%dT%H%M%SZ)}"

EPOCHS="${EPOCHS:-3000}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
LOG_EVERY="${LOG_EVERY:-200}"
N_EVAL_JOINT_SAMPLES="${N_EVAL_JOINT_SAMPLES:-64}"
HIDDEN_DIMS="${HIDDEN_DIMS:-256,256}"
CONDITION_INJECTION="${CONDITION_INJECTION:-concat}"
FILM_HIDDEN_DIM="${FILM_HIDDEN_DIM:-128}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"

cd "$ROOT_DIR"

echo "[info] FULL_TARGET_WIDE_CSV=$FULL_TARGET_WIDE_CSV"
echo "[info] C2F_DIR=$C2F_DIR"
echo "[info] C2F_WIDE_CSV=$C2F_WIDE_CSV"
echo "[info] C2F_SCHEMA_JSON=$C2F_SCHEMA_JSON"
echo "[info] PYTHON_BIN=$PYTHON_BIN"
echo "[info] DEVICE=$DEVICE"

mkdir -p "$C2F_DIR"

"$PYTHON_BIN" -u tools/model/build_external_c2f_age_schl_teacher.py \
  --joint_wide_csv "$FULL_TARGET_WIDE_CSV" \
  --out_dir "$C2F_DIR" \
  --overwrite

"$PYTHON_BIN" -u tools/model/train_external_c2f_age_schl_teacher.py \
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

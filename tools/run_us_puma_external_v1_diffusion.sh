#!/usr/bin/env bash
set -euo pipefail

RAW_ROOT="${RAW_ROOT:-/home/jinlin/data/geoexplicit_data}"
DATA_ROOT="${SYNTHCITY_DATA_ROOT:-$RAW_ROOT/synthetic_city/data}"
OUT_ROOT="${OUT_ROOT:-outputs}"
PYTHON_BIN="${PYTHON_BIN:-python}"

TARGET_WIDE_CSV="${TARGET_WIDE_CSV:-$DATA_ROOT/us/processed/external_targets/exttarget_v1_pums_2023_puma_us_joint_wide.csv}"
CONDITION_CSV="${CONDITION_CSV:-$DATA_ROOT/us/processed/external_conditions/extcond_v1_acs5_2022_puma_us.csv}"
SCHEMA_JSON="${SCHEMA_JSON:-}"

CONDITIONS="${CONDITIONS:-none,external}"
EVAL_MODE="${EVAL_MODE:-leave_mi_out}"
N_FOLDS="${N_FOLDS:-5}"
TIMESTEPS="${TIMESTEPS:-1000}"
EPOCHS="${EPOCHS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
HIDDEN_DIMS="${HIDDEN_DIMS:-512,512}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
CONDITION_INJECTION="${CONDITION_INJECTION:-concat}"
FILM_HIDDEN_DIM="${FILM_HIDDEN_DIM:-128}"
DEVICE="${DEVICE:-}"
SEED="${SEED:-0}"
LOG_EVERY="${LOG_EVERY:-200}"
N_EVAL_JOINT_SAMPLES="${N_EVAL_JOINT_SAMPLES:-128}"
IPF_ITERS="${IPF_ITERS:-200}"
POSTHOC_IPF_POLICY="${POSTHOC_IPF_POLICY:-external}"
SAVE_FINAL_MODEL="${SAVE_FINAL_MODEL:-0}"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${RUN_DIR:-$OUT_ROOT/_us_puma_external_v1_diffusion_${TS}}"
mkdir -p "$RUN_DIR"

CMD=(
  "$PYTHON_BIN" -u tools/train_us_puma_external_v1_diffusion.py
  --joint_wide_csv "$TARGET_WIDE_CSV"
  --condition_csv "$CONDITION_CSV"
  --conditions "$CONDITIONS"
  --eval_mode "$EVAL_MODE"
  --n_folds "$N_FOLDS"
  --timesteps "$TIMESTEPS"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --hidden_dims "$HIDDEN_DIMS"
  --lr "$LR"
  --weight_decay "$WEIGHT_DECAY"
  --condition_injection "$CONDITION_INJECTION"
  --film_hidden_dim "$FILM_HIDDEN_DIM"
  --seed "$SEED"
  --log_every "$LOG_EVERY"
  --n_eval_joint_samples "$N_EVAL_JOINT_SAMPLES"
  --ipf_iters "$IPF_ITERS"
  --posthoc_ipf_policy "$POSTHOC_IPF_POLICY"
  --out_dir "$RUN_DIR"
)

if [[ -n "$DEVICE" ]]; then
  CMD+=(--device "$DEVICE")
fi
if [[ -n "$SCHEMA_JSON" ]]; then
  CMD+=(--schema_json "$SCHEMA_JSON")
fi
if [[ "$SAVE_FINAL_MODEL" == "1" ]]; then
  CMD+=(--save_final_model)
fi

{
  echo "[info] RAW_ROOT=$RAW_ROOT"
  echo "[info] DATA_ROOT=$DATA_ROOT"
  echo "[info] TARGET_WIDE_CSV=$TARGET_WIDE_CSV"
  echo "[info] CONDITION_CSV=$CONDITION_CSV"
  echo "[info] SCHEMA_JSON=${SCHEMA_JSON:-<default-v1>}"
  echo "[info] PYTHON_BIN=$PYTHON_BIN"
  echo "[info] DEVICE=${DEVICE:-auto}"
  printf '[info] CMD='
  printf '%q ' "${CMD[@]}"
  printf '\n'
  "${CMD[@]}"
} 2>&1 | tee "$RUN_DIR/run.log"

echo "[ok] run dir: $RUN_DIR"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW_ROOT="${RAW_ROOT:-$ROOT_DIR/data}"
DATA_ROOT="${DATA_ROOT:-$RAW_ROOT/synthetic_city/data}"
PYTHON_BIN="${PYTHON_BIN:-python}"

WIDE_CSV="${WIDE_CSV:-$DATA_ROOT/us/processed/external_c2f/extc2f_full_earn_teacher_pums_2023_puma_us_wide.csv}"
SCHEMA_JSON="${SCHEMA_JSON:-$DATA_ROOT/us/processed/external_c2f/extc2f_full_earn_teacher_pums_2023_puma_us.schema.json}"

TIMESTEPS="${TIMESTEPS:-200}"
EPOCHS="${EPOCHS:-600}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
HIDDEN_DIMS="${HIDDEN_DIMS:-256,256}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
CONDITION_INJECTION="${CONDITION_INJECTION:-concat}"
FILM_HIDDEN_DIM="${FILM_HIDDEN_DIM:-128}"
LATENT_DIM="${LATENT_DIM:-256}"
ENCODER_HIDDEN_DIMS="${ENCODER_HIDDEN_DIMS:-256,256}"
HEAD_HIDDEN_DIMS="${HEAD_HIDDEN_DIMS:-256}"
CLEAN_HEAD_WEIGHT="${CLEAN_HEAD_WEIGHT:-0.0}"
CONSISTENCY_WEIGHT="${CONSISTENCY_WEIGHT:-0.0}"
AUX_T_GATE="${AUX_T_GATE:--1}"
PREDICT_MODE="${PREDICT_MODE:-diffusion}"
BLEND_ALPHA="${BLEND_ALPHA:-0.0}"
N_EVAL_JOINT_SAMPLES="${N_EVAL_JOINT_SAMPLES:-128}"
DIFF_LOSS_REWEIGHT_ALPHA="${DIFF_LOSS_REWEIGHT_ALPHA:-0.0}"
DIFF_LOSS_REWEIGHT_FLOOR="${DIFF_LOSS_REWEIGHT_FLOOR:-0.05}"
DIFF_LOSS_REWEIGHT_CAP="${DIFF_LOSS_REWEIGHT_CAP:-5.0}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
SAVE_FINAL_MODEL="${SAVE_FINAL_MODEL:-0}"
SAVE_BEST_MODEL="${SAVE_BEST_MODEL:-0}"
EVAL_EVERY_EPOCHS="${EVAL_EVERY_EPOCHS:-0}"
STRUCTURED_LOW_MEMORY="${STRUCTURED_LOW_MEMORY:-0}"
READ_CHUNKSIZE="${READ_CHUNKSIZE:-10000}"
EVAL_MODE="${EVAL_MODE:-leave_mi_out}"
HELDOUT_STATEFP="${HELDOUT_STATEFP:-26}"
RUN_DIR="${RUN_DIR:-outputs/_us_puma_external_c2f_full_earn_teacher_$(date -u +%Y%m%dT%H%M%SZ)}"

cd "$ROOT_DIR"

echo "[info] WIDE_CSV=$WIDE_CSV"
echo "[info] SCHEMA_JSON=$SCHEMA_JSON"
echo "[info] PYTHON_BIN=$PYTHON_BIN"
echo "[info] DEVICE=$DEVICE"
echo "[info] LATENT_DIM=$LATENT_DIM"
echo "[info] ENCODER_HIDDEN_DIMS=$ENCODER_HIDDEN_DIMS"
echo "[info] HEAD_HIDDEN_DIMS=$HEAD_HIDDEN_DIMS"
echo "[info] CLEAN_HEAD_WEIGHT=$CLEAN_HEAD_WEIGHT"
echo "[info] CONSISTENCY_WEIGHT=$CONSISTENCY_WEIGHT"
echo "[info] AUX_T_GATE=$AUX_T_GATE"
echo "[info] PREDICT_MODE=$PREDICT_MODE"
echo "[info] BLEND_ALPHA=$BLEND_ALPHA"
echo "[info] DIFF_LOSS_REWEIGHT_ALPHA=$DIFF_LOSS_REWEIGHT_ALPHA"
echo "[info] DIFF_LOSS_REWEIGHT_FLOOR=$DIFF_LOSS_REWEIGHT_FLOOR"
echo "[info] DIFF_LOSS_REWEIGHT_CAP=$DIFF_LOSS_REWEIGHT_CAP"
echo "[info] SAVE_BEST_MODEL=$SAVE_BEST_MODEL"
echo "[info] EVAL_EVERY_EPOCHS=$EVAL_EVERY_EPOCHS"
echo "[info] STRUCTURED_LOW_MEMORY=$STRUCTURED_LOW_MEMORY"
echo "[info] READ_CHUNKSIZE=$READ_CHUNKSIZE"
CMD=(
  "$PYTHON_BIN" -u tools/model/train_external_c2f_full_earn_teacher.py
  --wide_csv "$WIDE_CSV"
  --schema_json "$SCHEMA_JSON"
  --eval_mode "$EVAL_MODE"
  --heldout_statefp "$HELDOUT_STATEFP"
  --timesteps "$TIMESTEPS"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --hidden_dims "$HIDDEN_DIMS"
  --lr "$LR"
  --weight_decay "$WEIGHT_DECAY"
  --condition_injection "$CONDITION_INJECTION"
  --film_hidden_dim "$FILM_HIDDEN_DIM"
  --latent_dim "$LATENT_DIM"
  --encoder_hidden_dims "$ENCODER_HIDDEN_DIMS"
  --head_hidden_dims "$HEAD_HIDDEN_DIMS"
  --clean_head_weight "$CLEAN_HEAD_WEIGHT"
  --consistency_weight "$CONSISTENCY_WEIGHT"
  --aux_t_gate "$AUX_T_GATE"
  --predict_mode "$PREDICT_MODE"
  --blend_alpha "$BLEND_ALPHA"
  --n_eval_joint_samples "$N_EVAL_JOINT_SAMPLES"
  --diff_loss_reweight_alpha "$DIFF_LOSS_REWEIGHT_ALPHA"
  --diff_loss_reweight_floor "$DIFF_LOSS_REWEIGHT_FLOOR"
  --diff_loss_reweight_cap "$DIFF_LOSS_REWEIGHT_CAP"
  --device "$DEVICE"
  --seed "$SEED"
  --eval_every_epochs "$EVAL_EVERY_EPOCHS"
  --out_dir "$RUN_DIR"
)
if [[ "$STRUCTURED_LOW_MEMORY" == "1" ]]; then
  CMD+=(--structured_low_memory --read_chunksize "$READ_CHUNKSIZE")
fi
if [[ "$SAVE_FINAL_MODEL" == "1" ]]; then
  CMD+=(--save_final_model)
fi
if [[ "$SAVE_BEST_MODEL" == "1" ]]; then
  CMD+=(--save_best_model)
fi

printf '[info] CMD='
printf '%q ' "${CMD[@]}"
printf '\n'
"${CMD[@]}"

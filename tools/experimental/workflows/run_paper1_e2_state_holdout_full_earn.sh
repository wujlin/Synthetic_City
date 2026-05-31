#!/usr/bin/env bash
set -euo pipefail

# Extra state-held-out validation for Paper 1 full-earn coarse-to-fine pipeline.
# Default values reproduce the mainline recipe while replacing Michigan with STATEFP.

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
STATEFP="${STATEFP:-55}"
SEED="${SEED:-0}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-3000}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-600}"
STAGE1_EVAL_EVERY="${STAGE1_EVAL_EVERY:-200}"
STAGE2_EVAL_EVERY="${STAGE2_EVAL_EVERY:-50}"
STAGE1_N_VAL_SAMPLES="${STAGE1_N_VAL_SAMPLES:-16}"
STAGE1_N_EVAL_SAMPLES="${STAGE1_N_EVAL_SAMPLES:-32}"
STAGE2_N_EVAL_SAMPLES="${STAGE2_N_EVAL_SAMPLES:-128}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
PUMS_YEAR="${PUMS_YEAR:-2023}"
ACS_YEAR="${ACS_YEAR:-2022}"
SCOPE_TAG="${SCOPE_TAG:-us}"
DATASET_LABEL="${DATASET_LABEL:-pums${PUMS_YEAR}_acs${ACS_YEAR}}"

DATA_ROOT="${DATA_ROOT:-/home/jinlin/data/geoexplicit_data/synthetic_city/data}"
TARGET_DIR="${TARGET_DIR:-$DATA_ROOT/us/processed/external_targets}"
COND_DIR="${COND_DIR:-$DATA_ROOT/us/processed/external_conditions}"
C2F_BASE_DIR="${C2F_BASE_DIR:-$DATA_ROOT/us/processed/external_c2f}"

JOINT_WIDE_CSV="${JOINT_WIDE_CSV:-$TARGET_DIR/exttarget_v1_full_earn_pums_${PUMS_YEAR}_puma_${SCOPE_TAG}_joint_wide.csv}"
SCHEMA_JSON="${SCHEMA_JSON:-$TARGET_DIR/exttarget_v1_full_earn_pums_${PUMS_YEAR}_puma_${SCOPE_TAG}.schema.json}"
CONDITION_CSV="${CONDITION_CSV:-$COND_DIR/extcond_v1_earn_v1_acs5_${ACS_YEAR}_puma_${SCOPE_TAG}.csv}"
CONDITION_SCHEMA_JSON="${CONDITION_SCHEMA_JSON:-$SCHEMA_JSON}"

FOLD_LABEL="leave_state_${STATEFP}_out"
if [[ "$STATEFP" == "26" ]]; then
  FOLD_LABEL="leave_mi_out"
fi

RUN_STEM="paper1_E2_full_earn_${DATASET_LABEL}_state${STATEFP}_seed${SEED}_${TIMESTAMP}"
STAGE1_RUN_DIR="${STAGE1_RUN_DIR:-outputs/_${RUN_STEM}_stage1}"
C2F_DIR="${C2F_DIR:-$C2F_BASE_DIR/e2_state${STATEFP}_seed${SEED}_${TIMESTAMP}_stage1ipfcondmix}"
STAGE2_RUN_DIR="${STAGE2_RUN_DIR:-outputs/_${RUN_STEM}_stage2}"
EVAL_RUN_DIR="${EVAL_RUN_DIR:-outputs/_${RUN_STEM}_eval}"

mkdir -p "$(dirname "$STAGE1_RUN_DIR")" "$C2F_DIR" "$(dirname "$STAGE2_RUN_DIR")" "$(dirname "$EVAL_RUN_DIR")"

echo "[E2] state=$STATEFP seed=$SEED timestamp=$TIMESTAMP"
echo "[E2] dataset=$DATASET_LABEL pums_year=$PUMS_YEAR acs_year=$ACS_YEAR scope=$SCOPE_TAG"
echo "[E2] stage1 -> $STAGE1_RUN_DIR"
"$PYTHON_BIN" -u tools/model/train_external_joint_hier_diffusion_full_earn.py \
  --joint_wide_csv "$JOINT_WIDE_CSV" \
  --condition_csv "$CONDITION_CSV" \
  --schema_json "$SCHEMA_JSON" \
  --condition_schema_json "$CONDITION_SCHEMA_JSON" \
  --eval_mode leave_state_out \
  --heldout_statefp "$STATEFP" \
  --timesteps 200 \
  --epochs "$STAGE1_EPOCHS" \
  --batch_size 512 \
  --encoder_hidden_dims 256,256 \
  --coarse_hidden_dims 256 \
  --diffusion_hidden_dims 512,512 \
  --latent_dim 128 \
  --condition_injection concat \
  --coarse_weight 0.5 \
  --consistency_weight 1.0 \
  --marginal_weight 1.0 \
  --selection_metric val_tvd_coarse_head \
  --selection_raw_weight 0.25 \
  --aux_t_gate 50 \
  --detach_coarse_encoder \
  --diff_loss_reweight_alpha 0.5 \
  --support_mask_mode none \
  --n_val_joint_samples "$STAGE1_N_VAL_SAMPLES" \
  --n_eval_joint_samples "$STAGE1_N_EVAL_SAMPLES" \
  --val_ipf_iters 200 \
  --ipf_iters 200 \
  --eval_every "$STAGE1_EVAL_EVERY" \
  --val_frac 0.05 \
  --val_min_groups 96 \
  --ema_decay 0.999 \
  --save_best_checkpoint \
  --save_final_model \
  --device "$DEVICE" \
  --seed "$SEED" \
  --run_label "external_joint_hier_diffusion_full_earn_v2_e2_${DATASET_LABEL}_state${STATEFP}_seed${SEED}" \
  --out_dir "$STAGE1_RUN_DIR"

STAGE1_CKPT="$STAGE1_RUN_DIR/checkpoints/$FOLD_LABEL/best.pt"
if [[ ! -f "$STAGE1_CKPT" ]]; then
  STAGE1_CKPT="$STAGE1_RUN_DIR/checkpoints/$FOLD_LABEL/final.pt"
fi
if [[ ! -f "$STAGE1_CKPT" ]]; then
  echo "[E2][err] missing stage1 checkpoint under $STAGE1_RUN_DIR/checkpoints/$FOLD_LABEL" >&2
  exit 2
fi

echo "[E2] build c2f teacher -> $C2F_DIR"
"$PYTHON_BIN" -u tools/model/build_external_c2f_full_earn_teacher.py \
  --joint_wide_csv "$JOINT_WIDE_CSV" \
  --out_dir "$C2F_DIR" \
  --use_stage1_coarse_ipf_for_condition \
  --append_true_coarse_rows \
  --stage1_checkpoint "$STAGE1_CKPT" \
  --stage1_schema_json "$SCHEMA_JSON" \
  --stage1_condition_csv "$CONDITION_CSV" \
  --stage1_condition_schema_json "$CONDITION_SCHEMA_JSON" \
  --stage1_timesteps 200 \
  --stage1_ipf_iters 200 \
  --stage1_seed "$SEED" \
  --stage1_device "$DEVICE" \
  --overwrite

C2F_WIDE="$C2F_DIR/extc2f_full_earn_stage1ipfcondmix_pums_${PUMS_YEAR}_puma_${SCOPE_TAG}_wide.csv"
C2F_SCHEMA="$C2F_DIR/extc2f_full_earn_stage1ipfcondmix_pums_${PUMS_YEAR}_puma_${SCOPE_TAG}.schema.json"

echo "[E2] stage2 -> $STAGE2_RUN_DIR"
"$PYTHON_BIN" -u tools/model/train_external_c2f_full_earn_teacher.py \
  --wide_csv "$C2F_WIDE" \
  --schema_json "$C2F_SCHEMA" \
  --eval_mode leave_state_out \
  --heldout_statefp "$STATEFP" \
  --timesteps 200 \
  --epochs "$STAGE2_EPOCHS" \
  --batch_size 4096 \
  --hidden_dims 256,256 \
  --latent_dim 256 \
  --encoder_hidden_dims 256,256 \
  --head_hidden_dims 256 \
  --condition_injection concat \
  --clean_head_weight 1.0 \
  --consistency_weight 0.5 \
  --aux_t_gate 50 \
  --predict_mode blend \
  --blend_alpha 0.25 \
  --diff_loss_reweight_alpha 0.5 \
  --n_eval_joint_samples "$STAGE2_N_EVAL_SAMPLES" \
  --eval_every_epochs "$STAGE2_EVAL_EVERY" \
  --save_best_model \
  --save_final_model \
  --device "$DEVICE" \
  --seed "$SEED" \
  --out_dir "$STAGE2_RUN_DIR"

STAGE2_CKPT="$STAGE2_RUN_DIR/checkpoints/external_c2f_full_earn_teacher/$FOLD_LABEL/best.pt"
if [[ ! -f "$STAGE2_CKPT" ]]; then
  STAGE2_CKPT="$STAGE2_RUN_DIR/checkpoints/external_c2f_full_earn_teacher/$FOLD_LABEL/final.pt"
fi
if [[ ! -f "$STAGE2_CKPT" ]]; then
  echo "[E2][err] missing stage2 checkpoint under $STAGE2_RUN_DIR/checkpoints/external_c2f_full_earn_teacher/$FOLD_LABEL" >&2
  exit 3
fi

echo "[E2] eval -> $EVAL_RUN_DIR"
"$PYTHON_BIN" -u tools/model/eval_external_c2f_full_earn_pipeline.py \
  --stage1_joint_wide_csv "$JOINT_WIDE_CSV" \
  --stage1_schema_json "$SCHEMA_JSON" \
  --stage1_condition_csv "$CONDITION_CSV" \
  --stage1_condition_schema_json "$CONDITION_SCHEMA_JSON" \
  --stage1_checkpoint "$STAGE1_CKPT" \
  --stage1_timesteps 200 \
  --stage2_wide_csv "$C2F_WIDE" \
  --stage2_schema_json "$C2F_SCHEMA" \
  --stage2_checkpoint "$STAGE2_CKPT" \
  --stage2_n_eval_joint_samples "$STAGE2_N_EVAL_SAMPLES" \
  --ipf_iters 200 \
  --heldout_statefp "$STATEFP" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --out_dir "$EVAL_RUN_DIR"

"$PYTHON_BIN" - <<PY
import json, pathlib
summary_path = pathlib.Path("$EVAL_RUN_DIR") / "metrics" / "coarse_to_fine_summary.json"
summary = json.loads(summary_path.read_text())
payload = {
    "statefp": "$STATEFP",
    "seed": int("$SEED"),
    "dataset_label": "$DATASET_LABEL",
    "pums_year": int("$PUMS_YEAR"),
    "acs_year": int("$ACS_YEAR"),
    "scope": "$SCOPE_TAG",
    "stage1_run_dir": "$STAGE1_RUN_DIR",
    "stage2_run_dir": "$STAGE2_RUN_DIR",
    "eval_run_dir": "$EVAL_RUN_DIR",
    "pipeline_tvd": summary["pipeline_stage1_coarse_ipf"]["tvd_joint"]["mean"],
    "ipf_tvd": summary["ipf_train_seed_external"]["tvd_joint"]["mean"],
    "oracle_stage2_tvd": summary["oracle_stage2_true_coarse"]["tvd_joint"]["mean"],
}
(pathlib.Path("$EVAL_RUN_DIR") / "run_manifest.json").write_text(json.dumps(payload, indent=2) + "\\n")
print(json.dumps(payload, indent=2))
PY

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW_ROOT="${RAW_ROOT:-/home/jinlin/data/geoexplicit_data}"
DATA_ROOT="${DATA_ROOT:-$RAW_ROOT/synthetic_city/data}"
PYTHON_BIN="${PYTHON_BIN:-python}"

STAGE1_JOINT_WIDE_CSV="${STAGE1_JOINT_WIDE_CSV:-$DATA_ROOT/us/processed/external_targets/exttarget_v1_lite_pums_2023_puma_us_joint_wide.csv}"
STAGE1_SCHEMA_JSON="${STAGE1_SCHEMA_JSON:-$DATA_ROOT/us/processed/external_targets/exttarget_v1_lite_pums_2023_puma_us.schema.json}"
STAGE1_CONDITION_CSV="${STAGE1_CONDITION_CSV:-$DATA_ROOT/us/processed/external_conditions/extcond_v1_lite_acs5_2022_puma_us.csv}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-outputs/_us_puma_external_v1_lite_retry_20260323T075526Z/checkpoints/external/leave_mi_out/final.pt}"

STAGE2_WIDE_CSV="${STAGE2_WIDE_CSV:-$DATA_ROOT/us/processed/external_c2f/extc2f_age_schl_teacher_pums_2023_puma_us_wide.csv}"
STAGE2_SCHEMA_JSON="${STAGE2_SCHEMA_JSON:-$DATA_ROOT/us/processed/external_c2f/extc2f_age_schl_teacher_pums_2023_puma_us.schema.json}"
STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:-outputs/_us_puma_external_c2f_age_schl_teacher_20260323T103646Z/checkpoints/external_c2f_age_schl_teacher/leave_mi_out/final.pt}"

FINAL_TARGET_WIDE_CSV="${FINAL_TARGET_WIDE_CSV:-$DATA_ROOT/us/processed/external_targets/exttarget_v1_age_schl_refine_pums_2023_puma_us_joint_wide.csv}"
FINAL_TARGET_SCHEMA_JSON="${FINAL_TARGET_SCHEMA_JSON:-$DATA_ROOT/us/processed/external_targets/exttarget_v1_age_schl_refine_pums_2023_puma_us.schema.json}"
ONE_SHOT_SUMMARY_JSON="${ONE_SHOT_SUMMARY_JSON:-outputs/_us_puma_external_v1_age_schl_refine_20260323T084421Z/metrics/ablation_summary.json}"

STAGE1_N_EVAL_JOINT_SAMPLES="${STAGE1_N_EVAL_JOINT_SAMPLES:-64}"
STAGE2_N_EVAL_JOINT_SAMPLES="${STAGE2_N_EVAL_JOINT_SAMPLES:-64}"
IPF_ITERS="${IPF_ITERS:-200}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
RUN_DIR="${RUN_DIR:-outputs/_us_puma_external_c2f_age_schl_eval_$(date -u +%Y%m%dT%H%M%SZ)}"

cd "$ROOT_DIR"

echo "[info] STAGE1_JOINT_WIDE_CSV=$STAGE1_JOINT_WIDE_CSV"
echo "[info] STAGE1_SCHEMA_JSON=$STAGE1_SCHEMA_JSON"
echo "[info] STAGE1_CONDITION_CSV=$STAGE1_CONDITION_CSV"
echo "[info] STAGE1_CHECKPOINT=$STAGE1_CHECKPOINT"
echo "[info] STAGE2_WIDE_CSV=$STAGE2_WIDE_CSV"
echo "[info] STAGE2_SCHEMA_JSON=$STAGE2_SCHEMA_JSON"
echo "[info] STAGE2_CHECKPOINT=$STAGE2_CHECKPOINT"
echo "[info] FINAL_TARGET_WIDE_CSV=$FINAL_TARGET_WIDE_CSV"
echo "[info] FINAL_TARGET_SCHEMA_JSON=$FINAL_TARGET_SCHEMA_JSON"
echo "[info] ONE_SHOT_SUMMARY_JSON=$ONE_SHOT_SUMMARY_JSON"
echo "[info] PYTHON_BIN=$PYTHON_BIN"
echo "[info] DEVICE=$DEVICE"

"$PYTHON_BIN" -u tools/model/eval_external_c2f_age_schl_pipeline.py \
  --stage1_joint_wide_csv "$STAGE1_JOINT_WIDE_CSV" \
  --stage1_schema_json "$STAGE1_SCHEMA_JSON" \
  --stage1_condition_csv "$STAGE1_CONDITION_CSV" \
  --stage1_checkpoint "$STAGE1_CHECKPOINT" \
  --stage2_wide_csv "$STAGE2_WIDE_CSV" \
  --stage2_schema_json "$STAGE2_SCHEMA_JSON" \
  --stage2_checkpoint "$STAGE2_CHECKPOINT" \
  --final_target_wide_csv "$FINAL_TARGET_WIDE_CSV" \
  --final_target_schema_json "$FINAL_TARGET_SCHEMA_JSON" \
  --one_shot_summary_json "$ONE_SHOT_SUMMARY_JSON" \
  --stage1_n_eval_joint_samples "$STAGE1_N_EVAL_JOINT_SAMPLES" \
  --stage2_n_eval_joint_samples "$STAGE2_N_EVAL_JOINT_SAMPLES" \
  --ipf_iters "$IPF_ITERS" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --out_dir "$RUN_DIR"

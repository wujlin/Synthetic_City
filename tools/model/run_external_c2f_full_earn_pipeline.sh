#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW_ROOT="${RAW_ROOT:-$ROOT_DIR/data}"
DATA_ROOT="${DATA_ROOT:-$RAW_ROOT/synthetic_city/data}"
OUT_ROOT="${OUT_ROOT:-outputs}"
PYTHON_BIN="${PYTHON_BIN:-python}"

STAGE1_JOINT_WIDE_CSV="${STAGE1_JOINT_WIDE_CSV:-$DATA_ROOT/us/processed/external_targets/exttarget_v1_full_earn_pums_2023_puma_us_joint_wide.csv}"
STAGE1_SCHEMA_JSON="${STAGE1_SCHEMA_JSON:-$DATA_ROOT/us/processed/external_targets/exttarget_v1_full_earn_pums_2023_puma_us.schema.json}"
STAGE1_CONDITION_CSV="${STAGE1_CONDITION_CSV:-$DATA_ROOT/us/processed/external_conditions/extcond_v1_earn_v1_acs5_2022_puma_us.csv}"
STAGE1_CONDITION_SCHEMA_JSON="${STAGE1_CONDITION_SCHEMA_JSON:-$STAGE1_SCHEMA_JSON}"
STAGE1_CONDITION_SCALE_MODE="${STAGE1_CONDITION_SCALE_MODE:-none}"
STAGE1_CONDITION_EXTRA_CSV="${STAGE1_CONDITION_EXTRA_CSV:-}"
STAGE1_CONDITION_EXTRA_STANDARDIZE="${STAGE1_CONDITION_EXTRA_STANDARDIZE:-none}"
STAGE1_CONDITION_EXTRA_MISSING_POLICY="${STAGE1_CONDITION_EXTRA_MISSING_POLICY:-require}"

STAGE2_WIDE_CSV="${STAGE2_WIDE_CSV:-$DATA_ROOT/us/processed/external_c2f/extc2f_full_earn_teacher_pums_2023_puma_us_wide.csv}"
STAGE2_SCHEMA_JSON="${STAGE2_SCHEMA_JSON:-$DATA_ROOT/us/processed/external_c2f/extc2f_full_earn_teacher_pums_2023_puma_us.schema.json}"

STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:?STAGE1_CHECKPOINT is required}"
STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:?STAGE2_CHECKPOINT is required}"
ONE_SHOT_SUMMARY_JSON="${ONE_SHOT_SUMMARY_JSON:-}"

STAGE1_TIMESTEPS="${STAGE1_TIMESTEPS:-200}"
STAGE2_N_EVAL_JOINT_SAMPLES="${STAGE2_N_EVAL_JOINT_SAMPLES:-64}"
IPF_ITERS="${IPF_ITERS:-200}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
HELDOUT_STATEFP="${HELDOUT_STATEFP:-26}"

RUN_DIR="${RUN_DIR:-$OUT_ROOT/_us_puma_external_c2f_full_earn_eval_$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$RUN_DIR"

cd "$ROOT_DIR"

CMD=(
  "$PYTHON_BIN" -u tools/model/eval_external_c2f_full_earn_pipeline.py
  --stage1_joint_wide_csv "$STAGE1_JOINT_WIDE_CSV"
  --stage1_schema_json "$STAGE1_SCHEMA_JSON"
  --stage1_condition_csv "$STAGE1_CONDITION_CSV"
  --stage1_condition_schema_json "$STAGE1_CONDITION_SCHEMA_JSON"
  --stage1_condition_scale_mode "$STAGE1_CONDITION_SCALE_MODE"
  --stage1_condition_extra_standardize "$STAGE1_CONDITION_EXTRA_STANDARDIZE"
  --stage1_condition_extra_missing_policy "$STAGE1_CONDITION_EXTRA_MISSING_POLICY"
  --stage1_checkpoint "$STAGE1_CHECKPOINT"
  --stage1_timesteps "$STAGE1_TIMESTEPS"
  --stage2_wide_csv "$STAGE2_WIDE_CSV"
  --stage2_schema_json "$STAGE2_SCHEMA_JSON"
  --stage2_checkpoint "$STAGE2_CHECKPOINT"
  --stage2_n_eval_joint_samples "$STAGE2_N_EVAL_JOINT_SAMPLES"
  --ipf_iters "$IPF_ITERS"
  --device "$DEVICE"
  --seed "$SEED"
  --heldout_statefp "$HELDOUT_STATEFP"
  --out_dir "$RUN_DIR"
)

if [[ -n "$ONE_SHOT_SUMMARY_JSON" ]]; then
  CMD+=(--one_shot_summary_json "$ONE_SHOT_SUMMARY_JSON")
fi
if [[ -n "$STAGE1_CONDITION_EXTRA_CSV" ]]; then
  CMD+=(--stage1_condition_extra_csv "$STAGE1_CONDITION_EXTRA_CSV")
fi

printf '[info] CMD='
printf '%q ' "${CMD[@]}"
printf '\n'
"${CMD[@]}" 2>&1 | tee "$RUN_DIR/run.log"

echo "[ok] run dir: $RUN_DIR"

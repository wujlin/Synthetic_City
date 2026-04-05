#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_ROOT="${RAW_ROOT:-}"
if [[ -n "${SYNTHCITY_DATA_ROOT:-}" ]]; then
  DATA_ROOT="${SYNTHCITY_DATA_ROOT}"
elif [[ -n "$RAW_ROOT" ]]; then
  DATA_ROOT="$RAW_ROOT/synthetic_city/data"
else
  DATA_ROOT="$ROOT_DIR/data"
fi
PYTHON_BIN="${PYTHON_BIN:-python}"

JOINT_WIDE_CSV="${JOINT_WIDE_CSV:-$DATA_ROOT/us/processed/external_targets/exttarget_v1_full_income_v2_pums_2023_puma_us_joint_wide.csv}"
OUT_DIR="${OUT_DIR:-$DATA_ROOT/us/processed/external_c2f}"

USE_STAGE1_COARSE_IPF_FOR_CONDITION="${USE_STAGE1_COARSE_IPF_FOR_CONDITION:-0}"
APPEND_TRUE_COARSE_ROWS="${APPEND_TRUE_COARSE_ROWS:-0}"
CHILD_MASK_MODE="${CHILD_MASK_MODE:-parent_all}"
CHILD_SUPPORT_EPS="${CHILD_SUPPORT_EPS:-0.0}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-}"
STAGE1_SCHEMA_JSON="${STAGE1_SCHEMA_JSON:-$DATA_ROOT/us/processed/external_targets/exttarget_v1_full_income_v2_pums_2023_puma_us.schema.json}"
STAGE1_CONDITION_CSV="${STAGE1_CONDITION_CSV:-$DATA_ROOT/us/processed/external_conditions/extcond_v1_income_v2_acs5_2022_puma_us.csv}"
STAGE1_CONDITION_SCHEMA_JSON="${STAGE1_CONDITION_SCHEMA_JSON:-$STAGE1_CONDITION_CSV.schema.json}"
STAGE1_TIMESTEPS="${STAGE1_TIMESTEPS:-200}"
STAGE1_IPF_ITERS="${STAGE1_IPF_ITERS:-200}"
STAGE1_SEED="${STAGE1_SEED:-0}"
STAGE1_DEVICE="${STAGE1_DEVICE:-cuda}"
OVERWRITE="${OVERWRITE:-0}"

cd "$ROOT_DIR"

CMD=(
  "$PYTHON_BIN" -u tools/build_external_c2f_full_income_v2_teacher.py
  --joint_wide_csv "$JOINT_WIDE_CSV"
  --out_dir "$OUT_DIR"
  --child_mask_mode "$CHILD_MASK_MODE"
  --child_support_eps "$CHILD_SUPPORT_EPS"
  --stage1_timesteps "$STAGE1_TIMESTEPS"
  --stage1_ipf_iters "$STAGE1_IPF_ITERS"
  --stage1_seed "$STAGE1_SEED"
  --stage1_device "$STAGE1_DEVICE"
)

if [[ "$USE_STAGE1_COARSE_IPF_FOR_CONDITION" == "1" ]]; then
  CMD+=(--use_stage1_coarse_ipf_for_condition)
  CMD+=(--stage1_checkpoint "$STAGE1_CHECKPOINT")
  CMD+=(--stage1_schema_json "$STAGE1_SCHEMA_JSON")
  CMD+=(--stage1_condition_csv "$STAGE1_CONDITION_CSV")
  CMD+=(--stage1_condition_schema_json "$STAGE1_CONDITION_SCHEMA_JSON")
fi
if [[ "$APPEND_TRUE_COARSE_ROWS" == "1" ]]; then
  CMD+=(--append_true_coarse_rows)
fi
if [[ "$OVERWRITE" == "1" ]]; then
  CMD+=(--overwrite)
fi

printf '[info] CMD='
printf '%q ' "${CMD[@]}"
printf '\n'
"${CMD[@]}"

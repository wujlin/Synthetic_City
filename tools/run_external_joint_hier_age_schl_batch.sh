#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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

RUN_PREFIX="${RUN_PREFIX:-_us_puma_external_joint_hier_age_schl_batch}"
EPOCHS="${EPOCHS:-4000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
ENCODER_HIDDEN_DIMS="${ENCODER_HIDDEN_DIMS:-256,256}"
COARSE_HIDDEN_DIMS="${COARSE_HIDDEN_DIMS:-256}"
FINE_HIDDEN_DIMS="${FINE_HIDDEN_DIMS:-512,512}"
LATENT_DIM="${LATENT_DIM:-128}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
COARSE_WEIGHT="${COARSE_WEIGHT:-0.5}"
CONSISTENCY_WEIGHT="${CONSISTENCY_WEIGHT:-1.0}"
LOG_EVERY="${LOG_EVERY:-200}"
IPF_ITERS="${IPF_ITERS:-200}"
DEVICE="${DEVICE:-cuda}"

cd "$ROOT_DIR"

echo "[info] batch build once for age_schl_refine"
"$PYTHON_BIN" -u tools/build_external_condition_v1_variant.py \
  --condition_csv "$FULL_CONDITION_CSV" \
  --variant age_schl_refine \
  --out_path "$COND_VARIANT_CSV" \
  --overwrite

"$PYTHON_BIN" -u tools/build_external_target_v1_variant.py \
  --joint_wide_csv "$FULL_TARGET_WIDE_CSV" \
  --variant age_schl_refine \
  --condition_csv "$COND_VARIANT_CSV" \
  --out_dir "$TARGET_DIR" \
  --overwrite

declare -a MODES=("z_coarse_prob" "z_coarse_prob" "z_coarse_prob" "z_only" "z_coarse_latent")
declare -a SEEDS=("0" "1" "2" "0" "0")
declare -a TAGS=("seed0" "seed1" "seed2" "zonly_seed0" "zlatent_seed0")
declare -a RUN_DIRS=()

for idx in "${!MODES[@]}"; do
  mode="${MODES[$idx]}"
  seed="${SEEDS[$idx]}"
  tag="${TAGS[$idx]}"
  run_dir="outputs/${RUN_PREFIX}_${tag}"
  RUN_DIRS+=("$run_dir")
  echo "[run] mode=$mode seed=$seed run_dir=$run_dir"
  SKIP_VARIANT_BUILD=1 \
  FINE_INPUT_MODE="$mode" \
  SEED="$seed" \
  RUN_DIR="$run_dir" \
  EPOCHS="$EPOCHS" \
  BATCH_SIZE="$BATCH_SIZE" \
  ENCODER_HIDDEN_DIMS="$ENCODER_HIDDEN_DIMS" \
  COARSE_HIDDEN_DIMS="$COARSE_HIDDEN_DIMS" \
  FINE_HIDDEN_DIMS="$FINE_HIDDEN_DIMS" \
  LATENT_DIM="$LATENT_DIM" \
  LR="$LR" \
  WEIGHT_DECAY="$WEIGHT_DECAY" \
  COARSE_WEIGHT="$COARSE_WEIGHT" \
  CONSISTENCY_WEIGHT="$CONSISTENCY_WEIGHT" \
  LOG_EVERY="$LOG_EVERY" \
  IPF_ITERS="$IPF_ITERS" \
  DEVICE="$DEVICE" \
  PYTHON_BIN="$PYTHON_BIN" \
  bash tools/run_external_joint_hier_age_schl.sh
done

"$PYTHON_BIN" -u tools/summarize_external_joint_hier_runs.py \
  --label external_joint_hier_age_schl_batch \
  --out_dir "outputs/${RUN_PREFIX}_summary" \
  --run_dirs "${RUN_DIRS[@]}"

echo "[ok] batch finished"

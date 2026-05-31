#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/jinlin/miniconda3/envs/dpl/bin/python}"
RAW_ROOT="${RAW_ROOT:-/home/jinlin/data/geoexplicit_data}"
DATA_ROOT="${DATA_ROOT:-$RAW_ROOT/synthetic_city/data}"
OUT_ROOT="${OUT_ROOT:-outputs}"

JOINT_WIDE_CSV="${JOINT_WIDE_CSV:-$DATA_ROOT/us/processed/external_targets/exttarget_v1_full_earn_pums_2023_puma_us_joint_wide.csv}"
SCHEMA_JSON="${SCHEMA_JSON:-$DATA_ROOT/us/processed/external_targets/exttarget_v1_full_earn_pums_2023_puma_us.schema.json}"
CONDITION_CSV="${CONDITION_CSV:-$DATA_ROOT/us/processed/external_conditions/extcond_v1_earn_v1_acs5_2022_puma_us.csv}"
CONDITION_SCHEMA_JSON="${CONDITION_SCHEMA_JSON:-$SCHEMA_JSON}"
CONDITION_EXTRA_CSV="${CONDITION_EXTRA_CSV:-}"
CONDITION_EXTRA_STANDARDIZE="${CONDITION_EXTRA_STANDARDIZE:-none}"
CONDITION_EXTRA_MISSING_POLICY="${CONDITION_EXTRA_MISSING_POLICY:-require}"

TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
ROOT_RUN="${ROOT_RUN:-$OUT_ROOT/_coarse_dim_sensitivity_formal_${TS}}"
PRESETS="${PRESETS:-main_288}"
SEEDS="${SEEDS:-0}"
GPU_ID="${GPU_ID:-0}"

STAGE1_EPOCHS="${STAGE1_EPOCHS:-3000}"
STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-512}"
STAGE1_EVAL_EVERY="${STAGE1_EVAL_EVERY:-200}"
STAGE1_LOG_EVERY="${STAGE1_LOG_EVERY:-200}"
STAGE1_N_VAL_JOINT_SAMPLES="${STAGE1_N_VAL_JOINT_SAMPLES:-16}"
STAGE1_N_EVAL_JOINT_SAMPLES="${STAGE1_N_EVAL_JOINT_SAMPLES:-32}"

STAGE2_EPOCHS="${STAGE2_EPOCHS:-600}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-4096}"
STAGE2_EVAL_EVERY_EPOCHS="${STAGE2_EVAL_EVERY_EPOCHS:-50}"
STAGE2_N_EVAL_JOINT_SAMPLES="${STAGE2_N_EVAL_JOINT_SAMPLES:-128}"

PIPELINE_N_EVAL_JOINT_SAMPLES="${PIPELINE_N_EVAL_JOINT_SAMPLES:-64}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda}"
HELDOUT_STATEFP="${HELDOUT_STATEFP:-26}"

cd "$ROOT_DIR"
mkdir -p "$ROOT_RUN"

if [[ "${WRITE_MANIFEST:-1}" == "1" ]]; then
cat > "$ROOT_RUN/run_manifest.json" <<JSON
{
  "created_utc": "${TS}",
  "purpose": "Formal coarse-stage dimension sensitivity rerun with mainline-scale training budget.",
  "joint_wide_csv": "${JOINT_WIDE_CSV}",
  "schema_json": "${SCHEMA_JSON}",
  "condition_csv": "${CONDITION_CSV}",
  "condition_schema_json": "${CONDITION_SCHEMA_JSON}",
  "condition_extra_csv": "${CONDITION_EXTRA_CSV}",
  "condition_extra_standardize": "${CONDITION_EXTRA_STANDARDIZE}",
  "condition_extra_missing_policy": "${CONDITION_EXTRA_MISSING_POLICY}",
  "presets": "${PRESETS}",
  "seeds": "${SEEDS}",
  "heldout_statefp": "${HELDOUT_STATEFP}",
  "stage1": {
    "epochs": ${STAGE1_EPOCHS},
    "batch_size": ${STAGE1_BATCH_SIZE},
    "predict_mode": "head",
    "coarse_head_weight": 0.5,
    "marginal_weight": 1.0,
    "consistency_weight": 1.0,
    "aux_t_gate": 50,
    "selection_metric": "val_tvd_joint"
  },
  "stage2": {
    "epochs": ${STAGE2_EPOCHS},
    "batch_size": ${STAGE2_BATCH_SIZE},
    "clean_head_weight": 1.0,
    "consistency_weight": 0.5,
    "aux_t_gate": 50,
    "predict_mode": "blend",
    "blend_alpha": 0.25,
    "diff_loss_reweight_alpha": 0.5
  }
}
JSON
fi

run_one() {
  local preset="$1"
  local seed="$2"
  local combo_dir="$ROOT_RUN/${preset}/seed${seed}"
  local c2f_dir="$combo_dir/c2f_data"
  local stage1_dir="$combo_dir/stage1"
  local stage2_dir="$combo_dir/stage2"
  local eval_dir="$combo_dir/eval"
  local output_stem="extc2f_full_earn_${preset}_stage1ipfcondmix_pums_2023_puma_us"
  local stage2_wide_csv="$c2f_dir/${output_stem}_wide.csv"
  local stage2_schema_json="$c2f_dir/${output_stem}.schema.json"
  local flush_rows="${FLUSH_ROWS:-20000}"
  local fold_label="leave_state_${HELDOUT_STATEFP}_out"
  if [[ "$HELDOUT_STATEFP" == "26" ]]; then
    fold_label="leave_mi_out"
  fi
  local stage1_ckpt="$stage1_dir/checkpoints/${fold_label}/best.pt"
  local stage2_ckpt="$stage2_dir/checkpoints/external_c2f_full_earn_teacher/${fold_label}/best.pt"
  local -a stage1_condition_extra_args=()
  if [[ -n "$CONDITION_EXTRA_CSV" ]]; then
    stage1_condition_extra_args=(--stage1_condition_extra_csv "$CONDITION_EXTRA_CSV")
  fi

  if [[ "$preset" == "fine_720" && -n "${FLUSH_ROWS_FINE_720:-}" ]]; then
    flush_rows="$FLUSH_ROWS_FINE_720"
  fi
  if [[ "$preset" == "fine_2400" && -n "${FLUSH_ROWS_FINE_2400:-}" ]]; then
    flush_rows="$FLUSH_ROWS_FINE_2400"
  fi

  mkdir -p "$combo_dir" "$c2f_dir"
  {
    echo "[combo] preset=$preset seed=$seed gpu=$GPU_ID"
    echo "[combo] combo_dir=$combo_dir"
    if [[ -f "$stage1_ckpt" && "${FORCE_RERUN_STAGE1:-0}" != "1" ]]; then
      echo "[stage1] skip existing checkpoint: $stage1_ckpt"
    else
      echo "[stage1] start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
      SYNTHETIC_CITY_C2F_COARSE_PRESET="$preset" \
      CUDA_VISIBLE_DEVICES="$GPU_ID" \
      PYTHON_BIN="$PYTHON_BIN" \
      JOINT_WIDE_CSV="$JOINT_WIDE_CSV" \
      SCHEMA_JSON="$SCHEMA_JSON" \
      CONDITION_CSV="$CONDITION_CSV" \
      CONDITION_SCHEMA_JSON="$CONDITION_SCHEMA_JSON" \
      CONDITION_EXTRA_CSV="$CONDITION_EXTRA_CSV" \
      CONDITION_EXTRA_STANDARDIZE="$CONDITION_EXTRA_STANDARDIZE" \
      CONDITION_EXTRA_MISSING_POLICY="$CONDITION_EXTRA_MISSING_POLICY" \
      EPOCHS="$STAGE1_EPOCHS" \
      BATCH_SIZE="$STAGE1_BATCH_SIZE" \
      EVAL_EVERY="$STAGE1_EVAL_EVERY" \
      LOG_EVERY="$STAGE1_LOG_EVERY" \
      N_VAL_JOINT_SAMPLES="$STAGE1_N_VAL_JOINT_SAMPLES" \
      N_EVAL_JOINT_SAMPLES="$STAGE1_N_EVAL_JOINT_SAMPLES" \
      MARGINAL_WEIGHT="1.0" \
      COARSE_HEAD_WEIGHT="0.5" \
      CONSISTENCY_WEIGHT="1.0" \
      AUX_T_GATE="50" \
      PREDICT_MODE="head" \
      SELECTION_METRIC="val_tvd_joint" \
      EVAL_MODE="leave_state_out" \
      HELDOUT_STATEFP="$HELDOUT_STATEFP" \
      DEVICE="cuda" \
      SEED="$seed" \
      RUN_DIR="$stage1_dir" \
      bash tools/model/run_external_c2f_full_earn_stage1_coarse.sh
    fi
    if [[ ! -f "$stage1_ckpt" ]]; then
      echo "[error] missing stage1 checkpoint: $stage1_ckpt"
      exit 2
    fi

    if [[ -f "$stage2_wide_csv" && -f "$stage2_schema_json" && "${FORCE_REBUILD_STAGE2_DATA:-0}" != "1" ]]; then
      echo "[stage2-data] skip existing data: $stage2_wide_csv"
    else
      echo "[stage2-data] start $(date -u +%Y-%m-%dT%H:%M:%SZ) flush_rows=$flush_rows"
      SYNTHETIC_CITY_C2F_COARSE_PRESET="$preset" \
      CUDA_VISIBLE_DEVICES="$GPU_ID" \
      "$PYTHON_BIN" -u tools/model/build_external_c2f_full_earn_teacher.py \
        --joint_wide_csv "$JOINT_WIDE_CSV" \
        --out_dir "$c2f_dir" \
        --use_stage1_coarse_ipf_for_condition \
        --append_true_coarse_rows \
        --stage1_checkpoint "$stage1_ckpt" \
        --stage1_schema_json "$SCHEMA_JSON" \
        --stage1_condition_csv "$CONDITION_CSV" \
        --stage1_condition_schema_json "$CONDITION_SCHEMA_JSON" \
        "${stage1_condition_extra_args[@]}" \
        --stage1_condition_extra_standardize "$CONDITION_EXTRA_STANDARDIZE" \
        --stage1_condition_extra_missing_policy "$CONDITION_EXTRA_MISSING_POLICY" \
        --stage1_timesteps 200 \
        --stage1_ipf_iters 200 \
        --stage1_seed "$seed" \
        --stage1_device cuda \
        --output_stem "$output_stem" \
        --flush_rows "$flush_rows" \
        --overwrite
    fi

    if [[ -f "$stage2_ckpt" && "${FORCE_RERUN_STAGE2:-0}" != "1" ]]; then
      echo "[stage2] skip existing checkpoint: $stage2_ckpt"
    else
      echo "[stage2] start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
      SYNTHETIC_CITY_C2F_COARSE_PRESET="$preset" \
      CUDA_VISIBLE_DEVICES="$GPU_ID" \
      PYTHON_BIN="$PYTHON_BIN" \
      WIDE_CSV="$stage2_wide_csv" \
      SCHEMA_JSON="$stage2_schema_json" \
      EPOCHS="$STAGE2_EPOCHS" \
      BATCH_SIZE="$STAGE2_BATCH_SIZE" \
      CLEAN_HEAD_WEIGHT="1.0" \
      CONSISTENCY_WEIGHT="0.5" \
      AUX_T_GATE="50" \
      PREDICT_MODE="blend" \
      BLEND_ALPHA="0.25" \
      DIFF_LOSS_REWEIGHT_ALPHA="0.5" \
      SAVE_BEST_MODEL="1" \
      SAVE_FINAL_MODEL="0" \
      EVAL_EVERY_EPOCHS="$STAGE2_EVAL_EVERY_EPOCHS" \
      N_EVAL_JOINT_SAMPLES="$STAGE2_N_EVAL_JOINT_SAMPLES" \
      STRUCTURED_LOW_MEMORY="${STAGE2_STRUCTURED_LOW_MEMORY:-${STRUCTURED_LOW_MEMORY:-0}}" \
      READ_CHUNKSIZE="${STAGE2_READ_CHUNKSIZE:-${READ_CHUNKSIZE:-10000}}" \
      EVAL_MODE="leave_state_out" \
      HELDOUT_STATEFP="$HELDOUT_STATEFP" \
      DEVICE="cuda" \
      SEED="$seed" \
      RUN_DIR="$stage2_dir" \
      bash tools/model/run_external_c2f_full_earn_teacher.sh
    fi
    if [[ ! -f "$stage2_ckpt" ]]; then
      echo "[error] missing stage2 checkpoint: $stage2_ckpt"
      exit 3
    fi

    if [[ "${SKIP_EVAL:-0}" == "1" ]]; then
      echo "[eval] skip because SKIP_EVAL=1"
    elif [[ -f "$eval_dir/run_summary.json" && "${FORCE_RERUN_EVAL:-0}" != "1" ]]; then
      echo "[eval] skip existing summary: $eval_dir/run_summary.json"
    else
      echo "[eval] start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
      SYNTHETIC_CITY_C2F_COARSE_PRESET="$preset" \
      CUDA_VISIBLE_DEVICES="$GPU_ID" \
      PYTHON_BIN="$PYTHON_BIN" \
      STAGE1_JOINT_WIDE_CSV="$JOINT_WIDE_CSV" \
      STAGE1_SCHEMA_JSON="$SCHEMA_JSON" \
      STAGE1_CONDITION_CSV="$CONDITION_CSV" \
      STAGE1_CONDITION_SCHEMA_JSON="$CONDITION_SCHEMA_JSON" \
      STAGE1_CONDITION_EXTRA_CSV="$CONDITION_EXTRA_CSV" \
      STAGE1_CONDITION_EXTRA_STANDARDIZE="$CONDITION_EXTRA_STANDARDIZE" \
      STAGE1_CONDITION_EXTRA_MISSING_POLICY="$CONDITION_EXTRA_MISSING_POLICY" \
      STAGE1_CHECKPOINT="$stage1_ckpt" \
      STAGE2_WIDE_CSV="$stage2_wide_csv" \
      STAGE2_SCHEMA_JSON="$stage2_schema_json" \
      STAGE2_CHECKPOINT="$stage2_ckpt" \
      STAGE2_N_EVAL_JOINT_SAMPLES="$PIPELINE_N_EVAL_JOINT_SAMPLES" \
      HELDOUT_STATEFP="$HELDOUT_STATEFP" \
      DEVICE="$EVAL_DEVICE" \
      SEED="$seed" \
      RUN_DIR="$eval_dir" \
      bash tools/model/run_external_c2f_full_earn_pipeline.sh
    fi

    echo "[combo] done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } 2>&1 | tee "$combo_dir/run.log"
}

for preset in $PRESETS; do
  for seed in $SEEDS; do
    run_one "$preset" "$seed"
  done
done

echo "[ok] root run: $ROOT_RUN"

#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/jinlin/miniconda3/envs/dpl/bin/python}"
OUT_ROOT="${OUT_ROOT:-/mnt/data_hdd/synthetic_city_runs}"
TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
ROOT="${ROOT:-$OUT_ROOT/_one_stage_poi_spatial_mi_3seed_${TS}}"
SESSION="${SESSION:-one_stage_poi_spatial_3seed}"
FEATURE_CSV="${FEATURE_CSV:-data/us/processed/features/puma_poi_landuse_dataplor_us_v1.csv}"
FEATURE_STANDARDIZE="${FEATURE_STANDARDIZE:-zscore}"

run_seed() {
  local gpu="$1"
  local seed="$2"
  local run_dir="$ROOT/seed${seed}"
  mkdir -p "$run_dir"
  echo "[run] seed=${seed} gpu=${gpu} run_dir=${run_dir}"
  (
    cd "$REPO_ROOT"
    CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHON_BIN="$PYTHON_BIN" \
    OUT_ROOT="$ROOT" \
    RUN_DIR="$run_dir" \
    CONDITION_EXTRA_CSV="$FEATURE_CSV" \
    CONDITION_EXTRA_STANDARDIZE="$FEATURE_STANDARDIZE" \
    CONDITION_EXTRA_MISSING_POLICY="require" \
    EVAL_MODE="leave_mi_out" \
    HELDOUT_STATEFP="26" \
    TIMESTEPS="200" \
    EPOCHS="3000" \
    BATCH_SIZE="512" \
    ENCODER_HIDDEN_DIMS="256,256" \
    COARSE_HIDDEN_DIMS="256" \
    DIFFUSION_HIDDEN_DIMS="512,512" \
    LATENT_DIM="128" \
    LR="1e-3" \
    WEIGHT_DECAY="1e-4" \
    CONDITION_INJECTION="concat" \
    COARSE_WEIGHT="0.5" \
    CONSISTENCY_WEIGHT="1.0" \
    MARGINAL_WEIGHT="1.0" \
    SELECTION_METRIC="val_tvd_coarse_head" \
    SELECTION_RAW_WEIGHT="0.25" \
    LOGP_CLIP_QUANTILE_LOW="0.001" \
    LOGP_CLIP_QUANTILE_HIGH="0.999" \
    AUX_T_GATE="50" \
    DETACH_COARSE_ENCODER="1" \
    DIFF_LOSS_REWEIGHT_ALPHA="0.5" \
    DIFF_LOSS_REWEIGHT_FLOOR="0.05" \
    DIFF_LOSS_REWEIGHT_CAP="5.0" \
    SUPPORT_MASK_MODE="none" \
    SUPPORT_MASK_EPS="1e-12" \
    DEVICE="cuda" \
    SEED="$seed" \
    LOG_EVERY="200" \
    EVAL_EVERY="200" \
    VAL_FRAC="0.05" \
    VAL_MIN_GROUPS="96" \
    N_VAL_JOINT_SAMPLES="16" \
    VAL_IPF_ITERS="200" \
    N_EVAL_JOINT_SAMPLES="32" \
    IPF_ITERS="200" \
    EMA_DECAY="0.999" \
    SAVE_BEST_CHECKPOINT="1" \
    SAVE_FINAL_MODEL="0" \
    SAVE_EVAL_CHECKPOINT_EVERY="0" \
    bash tools/model/run_external_joint_hier_diffusion_full_earn.sh
  )
}

wait_gpu_empty() {
  local gpu="$1"
  if [[ "${WAIT_FOR_GPU_EMPTY:-0}" != "1" ]]; then
    return 0
  fi
  echo "[wait] waiting for GPU ${gpu} to become free"
  while nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -Eq '[0-9]'; do
    nvidia-smi --id="$gpu" --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || true
    sleep 300
  done
}

worker() {
  local gpu="$1"
  shift
  mkdir -p "$ROOT"
  {
    echo "[worker] root=${ROOT}"
    echo "[worker] feature_csv=${FEATURE_CSV}"
    echo "[worker] feature_standardize=${FEATURE_STANDARDIZE}"
    echo "[worker] gpu=${gpu} seeds=$*"
  } | tee -a "$ROOT/launch.log"
  wait_gpu_empty "$gpu"
  for seed in "$@"; do
    run_seed "$gpu" "$seed"
  done
}

quote_worker_cmd() {
  local wait_flag="$1"
  local gpu="$2"
  shift 2
  printf 'cd %q && ROOT=%q PYTHON_BIN=%q FEATURE_CSV=%q FEATURE_STANDARDIZE=%q WAIT_FOR_GPU_EMPTY=%q bash %q --worker %q' \
    "$REPO_ROOT" "$ROOT" "$PYTHON_BIN" "$FEATURE_CSV" "$FEATURE_STANDARDIZE" "$wait_flag" "$0" "$gpu"
  local seed
  for seed in "$@"; do
    printf ' %q' "$seed"
  done
}

launch_tmux() {
  mkdir -p "$ROOT"
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[error] tmux session already exists: $SESSION" >&2
    exit 1
  fi
  local cmd_gpu1 cmd_gpu0
  cmd_gpu1="$(quote_worker_cmd 0 1 0 2)"
  cmd_gpu0="$(quote_worker_cmd 1 0 1)"
  tmux new-session -d -s "$SESSION" -n gpu1 "$cmd_gpu1"
  tmux split-window -h -t "$SESSION:0" "$cmd_gpu0"
  tmux select-layout -t "$SESSION:0" tiled >/dev/null
  {
    echo "[launch] session=${SESSION}"
    echo "[launch] root=${ROOT}"
    echo "[launch] gpu1 seeds=0,2"
    echo "[launch] gpu0 seeds=1 wait_for_empty=1"
    echo "[launch] feature_csv=${FEATURE_CSV}"
  } | tee "$ROOT/launch.log"
}

case "${1:-}" in
  --worker)
    shift
    worker "$@"
    ;;
  --tmux|"")
    launch_tmux
    ;;
  *)
    echo "Usage: $0 [--tmux | --worker GPU SEED...]" >&2
    exit 2
    ;;
esac

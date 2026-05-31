#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
ROOT_RUN="${ROOT_RUN:-outputs/_coarse_dim_sensitivity_formal_${TS}}"
PYTHON_BIN="${PYTHON_BIN:-/home/jinlin/miniconda3/envs/dpl/bin/python}"

WAIT_FOR_SESSIONS="${WAIT_FOR_SESSIONS:-}"
mkdir -p "$ROOT_RUN"

cat > "$ROOT_RUN/queue_manifest.json" <<JSON
{
  "created_utc": "${TS}",
  "root_run": "${ROOT_RUN}",
  "purpose": "Two-GPU queue for formal coarse-stage dimension sensitivity.",
  "workers": {
    "gpu0": "${GPU0_COMBOS-main_288:0 coarse_108:1 fine_720:1}",
    "gpu1": "${GPU1_COMBOS-main_288:1 coarse_108:0 coarse_108:2 fine_720:0 fine_720:2}"
  },
  "wait_for_sessions": "${WAIT_FOR_SESSIONS}"
}
JSON

wait_for_sessions() {
  if [[ -z "$WAIT_FOR_SESSIONS" ]]; then
    return 0
  fi
  echo "[queue] waiting for sessions: $WAIT_FOR_SESSIONS"
  while true; do
    local any_alive=0
    for sess in $WAIT_FOR_SESSIONS; do
      if tmux has-session -t "$sess" 2>/dev/null; then
        any_alive=1
        echo "[queue] still running: $sess"
      fi
    done
    if [[ "$any_alive" == "0" ]]; then
      echo "[queue] waited sessions have finished"
      return 0
    fi
    sleep 120
  done
}

run_worker() {
  local gpu="$1"
  shift
  local combos=("$@")
  local log="$ROOT_RUN/queue_worker_gpu${gpu}.log"
  {
    echo "[worker gpu${gpu}] start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    for combo in "${combos[@]}"; do
      local preset="${combo%%:*}"
      local seed="${combo##*:}"
      echo "[worker gpu${gpu}] combo preset=$preset seed=$seed start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
      WRITE_MANIFEST=0 \
      ROOT_RUN="$ROOT_RUN" \
      PRESETS="$preset" \
      SEEDS="$seed" \
      GPU_ID="$gpu" \
      PYTHON_BIN="$PYTHON_BIN" \
      bash tools/experimental/workflows/run_paper1_coarse_dim_sensitivity_formal.sh
      echo "[worker gpu${gpu}] combo preset=$preset seed=$seed done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    done
    echo "[worker gpu${gpu}] done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } 2>&1 | tee "$log"
}

wait_for_sessions

read -r -a gpu0_combos <<< "${GPU0_COMBOS-main_288:0 coarse_108:1 fine_720:1}"
read -r -a gpu1_combos <<< "${GPU1_COMBOS-main_288:1 coarse_108:0 coarse_108:2 fine_720:0 fine_720:2}"

run_worker 0 "${gpu0_combos[@]}" &
pid0=$!
run_worker 1 "${gpu1_combos[@]}" &
pid1=$!

wait "$pid0"
wait "$pid1"

echo "[queue] all workers finished: $ROOT_RUN"

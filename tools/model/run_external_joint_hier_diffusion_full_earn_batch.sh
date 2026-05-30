#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_ROOT="${OUT_ROOT:-outputs}"
RUN_PREFIX="${RUN_PREFIX:-_us_puma_external_joint_hier_diffusion_full_earn_v2_batch}"
SUMMARY_JSON="${SUMMARY_JSON:-$OUT_ROOT/${RUN_PREFIX}_summary.json}"
SEEDS="${SEEDS:-0 1 2}"

cd "$ROOT_DIR"

declare -a RUN_DIRS=()
for SEED in $SEEDS; do
  TS="$(date -u +%Y%m%dT%H%M%SZ)"
  RUN_DIR="${OUT_ROOT}/${RUN_PREFIX}_seed${SEED}_${TS}"
  echo "[info] launching seed=$SEED -> $RUN_DIR"
  SEED="$SEED" RUN_DIR="$RUN_DIR" bash tools/model/run_external_joint_hier_diffusion_full_earn.sh
  RUN_DIRS+=("$RUN_DIR")
done

"${PYTHON_BIN:-python}" -u tools/model/summarize_external_joint_hier_diffusion_runs.py \
  --label "external_joint_hier_diffusion_full_earn_v2_batch" \
  --run_dirs "${RUN_DIRS[@]}" \
  --out_json "$SUMMARY_JSON"

echo "[ok] summary: $SUMMARY_JSON"

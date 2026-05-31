#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
ROOT_RUN="${ROOT_RUN:-/mnt/data_hdd/synthetic_city_runs/_paper1_k960_poilodes_state_holdout_FL_TX_WI_${TS}}"
PYTHON_BIN="${PYTHON_BIN:-/home/jinlin/miniconda3/envs/dpl/bin/python}"
CONDITION_EXTRA_CSV="${CONDITION_EXTRA_CSV:-/home/jinlin/projects/Synthetic_City/data/us/processed/features/puma_spatial_poi_lodes_us_v1.csv}"
CONDITION_EXTRA_STANDARDIZE="${CONDITION_EXTRA_STANDARDIZE:-zscore}"
CONDITION_EXTRA_MISSING_POLICY="${CONDITION_EXTRA_MISSING_POLICY:-require}"

PRESET="${PRESET:-fine_960}"
SEED="${SEED:-0}"

STAGE1_EPOCHS="${STAGE1_EPOCHS:-3000}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-600}"
STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-512}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-4096}"
STAGE2_STRUCTURED_LOW_MEMORY="${STAGE2_STRUCTURED_LOW_MEMORY:-1}"
STAGE2_READ_CHUNKSIZE="${STAGE2_READ_CHUNKSIZE:-5000}"
FLUSH_ROWS="${FLUSH_ROWS:-5000}"

PIPELINE_N_EVAL_JOINT_SAMPLES="${PIPELINE_N_EVAL_JOINT_SAMPLES:-64}"
STAGE2_N_EVAL_JOINT_SAMPLES="${STAGE2_N_EVAL_JOINT_SAMPLES:-128}"
STAGE1_N_EVAL_JOINT_SAMPLES="${STAGE1_N_EVAL_JOINT_SAMPLES:-32}"
STAGE1_N_VAL_JOINT_SAMPLES="${STAGE1_N_VAL_JOINT_SAMPLES:-16}"

mkdir -p "$ROOT_RUN"
cat > "$ROOT_RUN/run_manifest.json" <<JSON
{
  "created_utc": "${TS}",
  "purpose": "K=960 POI+LODES state-holdout rerun for Florida, Texas, and Wisconsin.",
  "root_run": "${ROOT_RUN}",
  "preset": "${PRESET}",
  "seed": ${SEED},
  "heldout_statefps": ["12", "48", "55"],
  "condition_extra_csv": "${CONDITION_EXTRA_CSV}",
  "condition_extra_standardize": "${CONDITION_EXTRA_STANDARDIZE}",
  "condition_extra_missing_policy": "${CONDITION_EXTRA_MISSING_POLICY}",
  "gpu_plan": {
    "gpu0": ["12", "55"],
    "gpu1": ["48"]
  }
}
JSON

run_state() {
  local statefp="$1"
  local gpu="$2"
  local state_root="$ROOT_RUN/state${statefp}"
  local log="$ROOT_RUN/state${statefp}_gpu${gpu}.log"
  mkdir -p "$state_root"
  {
    echo "[state-holdout] state=${statefp} preset=${PRESET} seed=${SEED} gpu=${gpu} start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    ROOT_RUN="$state_root" \
    PRESETS="$PRESET" \
    SEEDS="$SEED" \
    GPU_ID="$gpu" \
    PYTHON_BIN="$PYTHON_BIN" \
    HELDOUT_STATEFP="$statefp" \
    EVAL_MODE="leave_state_out" \
    CONDITION_EXTRA_CSV="$CONDITION_EXTRA_CSV" \
    CONDITION_EXTRA_STANDARDIZE="$CONDITION_EXTRA_STANDARDIZE" \
    CONDITION_EXTRA_MISSING_POLICY="$CONDITION_EXTRA_MISSING_POLICY" \
    STAGE1_EPOCHS="$STAGE1_EPOCHS" \
    STAGE2_EPOCHS="$STAGE2_EPOCHS" \
    STAGE1_BATCH_SIZE="$STAGE1_BATCH_SIZE" \
    STAGE2_BATCH_SIZE="$STAGE2_BATCH_SIZE" \
    STAGE2_STRUCTURED_LOW_MEMORY="$STAGE2_STRUCTURED_LOW_MEMORY" \
    STAGE2_READ_CHUNKSIZE="$STAGE2_READ_CHUNKSIZE" \
    FLUSH_ROWS="$FLUSH_ROWS" \
    STAGE1_N_VAL_JOINT_SAMPLES="$STAGE1_N_VAL_JOINT_SAMPLES" \
    STAGE1_N_EVAL_JOINT_SAMPLES="$STAGE1_N_EVAL_JOINT_SAMPLES" \
    STAGE2_N_EVAL_JOINT_SAMPLES="$STAGE2_N_EVAL_JOINT_SAMPLES" \
    PIPELINE_N_EVAL_JOINT_SAMPLES="$PIPELINE_N_EVAL_JOINT_SAMPLES" \
    bash tools/experimental/workflows/run_paper1_coarse_dim_sensitivity_formal.sh
    echo "[state-holdout] state=${statefp} done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } 2>&1 | tee "$log"
}

run_worker() {
  local gpu="$1"
  shift
  for statefp in "$@"; do
    run_state "$statefp" "$gpu"
  done
}

run_worker 0 12 55 &
pid0=$!
run_worker 1 48 &
pid1=$!

wait "$pid0"
wait "$pid1"

"$PYTHON_BIN" - <<PY
import json
import pathlib

root = pathlib.Path("${ROOT_RUN}")
rows = []
for state in ["12", "48", "55"]:
    summary_path = root / f"state{state}" / "${PRESET}" / "seed${SEED}" / "eval" / "metrics" / "coarse_to_fine_summary.json"
    if not summary_path.exists():
        rows.append({"statefp": state, "status": "missing", "summary_path": str(summary_path)})
        continue
    d = json.loads(summary_path.read_text())
    rows.append({
        "statefp": state,
        "status": "completed",
        "pipeline_tvd": d["pipeline_stage1_coarse_ipf"]["tvd_joint"]["mean"],
        "pipeline_puma_sd": d["pipeline_stage1_coarse_ipf"]["tvd_joint"]["std"],
        "pipeline_p90": d["pipeline_stage1_coarse_ipf"]["tvd_joint"]["p90"],
        "ipf_tvd": d["ipf_train_seed_external"]["tvd_joint"]["mean"],
        "oracle_stage2_tvd": d["oracle_stage2_true_coarse"]["tvd_joint"]["mean"],
        "n_heldout_pumas": d["pipeline_stage1_coarse_ipf"]["tvd_joint"]["n"],
        "summary_path": str(summary_path),
    })
out = {"root_run": str(root), "rows": rows}
(root / "state_holdout_summary.json").write_text(json.dumps(out, indent=2) + "\\n")
print(json.dumps(out, indent=2))
PY

echo "[ok] state-holdout root: $ROOT_RUN"

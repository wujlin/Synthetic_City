#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RAW_ROOT="${RAW_ROOT:-/home/jinlin/data/geoexplicit_data}"
DATA_ROOT="${DATA_ROOT:-$RAW_ROOT/synthetic_city/data}"
OUT_ROOT="${OUT_ROOT:-outputs}"
PYTHON_BIN="${PYTHON_BIN:-/home/jinlin/miniconda3/envs/dpl/bin/python}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
HELDOUT_STATEFP="${HELDOUT_STATEFP:-26}"
TAG="${TAG:-oracle_role_state${HELDOUT_STATEFP}_seed${SEED}_$(date -u +%Y%m%dT%H%M%SZ)}"

FEATURE_CSV="${FEATURE_CSV:-outputs/_ssl_external_view_jepa_func_20260511T000000Z/inputs/puma_lodes_functional_role_summary_mi_fl_tx_wi.csv}"
FEATURE_STANDARDIZE="${FEATURE_STANDARDIZE:-zscore}"
FEATURE_MISSING_POLICY="${FEATURE_MISSING_POLICY:-zero}"

STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:-outputs/_us_puma_external_c2f_full_earn_teacher_stage1ipfcondmix_mainline_gate50_selcoarse_ep3000_seed0_maskw_a05_cleanheadw1_cons05_gate50_blend25_bestsel50_20260327T023511Z/checkpoints/external_c2f_full_earn_teacher/leave_mi_out/best.pt}"

STAGE1_RUN_DIR="$OUT_ROOT/_ssl_functional_condition_stage1_${TAG}"
C2F_RUN_DIR="$OUT_ROOT/_ssl_functional_condition_c2f_${TAG}"

echo "[info] TAG=$TAG"
echo "[info] FEATURE_CSV=$FEATURE_CSV"
echo "[info] STAGE1_RUN_DIR=$STAGE1_RUN_DIR"
echo "[info] C2F_RUN_DIR=$C2F_RUN_DIR"

RAW_ROOT="$RAW_ROOT" \
DATA_ROOT="$DATA_ROOT" \
OUT_ROOT="$OUT_ROOT" \
PYTHON_BIN="$PYTHON_BIN" \
DEVICE="$DEVICE" \
SEED="$SEED" \
HELDOUT_STATEFP="$HELDOUT_STATEFP" \
CONDITION_EXTRA_CSV="$FEATURE_CSV" \
CONDITION_EXTRA_STANDARDIZE="$FEATURE_STANDARDIZE" \
CONDITION_EXTRA_MISSING_POLICY="$FEATURE_MISSING_POLICY" \
EPOCHS="${EPOCHS:-3000}" \
TIMESTEPS="${TIMESTEPS:-200}" \
BATCH_SIZE="${BATCH_SIZE:-512}" \
COARSE_WEIGHT="${COARSE_WEIGHT:-0.5}" \
CONSISTENCY_WEIGHT="${CONSISTENCY_WEIGHT:-1.0}" \
MARGINAL_WEIGHT="${MARGINAL_WEIGHT:-1.0}" \
SELECTION_METRIC="${SELECTION_METRIC:-val_tvd_coarse_head}" \
SELECTION_RAW_WEIGHT="${SELECTION_RAW_WEIGHT:-0.25}" \
LOGP_CLIP_QUANTILE_LOW="${LOGP_CLIP_QUANTILE_LOW:-0.001}" \
LOGP_CLIP_QUANTILE_HIGH="${LOGP_CLIP_QUANTILE_HIGH:-0.999}" \
AUX_T_GATE="${AUX_T_GATE:-50}" \
DETACH_COARSE_ENCODER="${DETACH_COARSE_ENCODER:-1}" \
DIFF_LOSS_REWEIGHT_ALPHA="${DIFF_LOSS_REWEIGHT_ALPHA:-0.5}" \
DIFF_LOSS_REWEIGHT_FLOOR="${DIFF_LOSS_REWEIGHT_FLOOR:-0.05}" \
DIFF_LOSS_REWEIGHT_CAP="${DIFF_LOSS_REWEIGHT_CAP:-5.0}" \
SUPPORT_MASK_MODE="${SUPPORT_MASK_MODE:-none}" \
SAVE_BEST_CHECKPOINT="${SAVE_BEST_CHECKPOINT:-1}" \
SAVE_FINAL_MODEL="${SAVE_FINAL_MODEL:-1}" \
EVAL_EVERY="${EVAL_EVERY:-200}" \
LOG_EVERY="${LOG_EVERY:-200}" \
N_VAL_JOINT_SAMPLES="${N_VAL_JOINT_SAMPLES:-16}" \
N_EVAL_JOINT_SAMPLES="${N_EVAL_JOINT_SAMPLES:-32}" \
RUN_DIR="$STAGE1_RUN_DIR" \
bash tools/model/run_external_joint_hier_diffusion_full_earn.sh

STAGE1_CHECKPOINT="$STAGE1_RUN_DIR/checkpoints/leave_mi_out/best.pt"
if [[ ! -f "$STAGE1_CHECKPOINT" ]]; then
  echo "[error] missing stage1 checkpoint: $STAGE1_CHECKPOINT" >&2
  exit 1
fi

OUT_ROOT="$OUT_ROOT" \
DATA_ROOT="$DATA_ROOT" \
PYTHON_BIN="$PYTHON_BIN" \
DEVICE="$DEVICE" \
SEED="$SEED" \
HELDOUT_STATEFP="$HELDOUT_STATEFP" \
STAGE1_CHECKPOINT="$STAGE1_CHECKPOINT" \
STAGE2_CHECKPOINT="$STAGE2_CHECKPOINT" \
STAGE1_CONDITION_EXTRA_CSV="$FEATURE_CSV" \
STAGE1_CONDITION_EXTRA_STANDARDIZE="$FEATURE_STANDARDIZE" \
STAGE1_CONDITION_EXTRA_MISSING_POLICY="$FEATURE_MISSING_POLICY" \
STAGE2_N_EVAL_JOINT_SAMPLES="${STAGE2_N_EVAL_JOINT_SAMPLES:-64}" \
RUN_DIR="$C2F_RUN_DIR" \
bash tools/model/run_external_c2f_full_earn_pipeline.sh

"$PYTHON_BIN" - <<PY
import json
from pathlib import Path

stage1 = Path("$STAGE1_RUN_DIR") / "run_summary.json"
c2f = Path("$C2F_RUN_DIR") / "run_summary.json"
print("[summary] stage1", stage1)
print("[summary] c2f", c2f)
if c2f.exists():
    obj = json.loads(c2f.read_text())
    res = obj.get("results", {})
    pipe = res.get("pipeline_stage1_coarse_ipf", {}).get("tvd_joint", {})
    ipf = res.get("ipf_train_seed_external", {}).get("tvd_joint", {})
    coarse = res.get("stage1_coarse", {}).get("tvd_ipf", {})
    print("[summary] stage1_coarse_tvd", coarse.get("mean"))
    print("[summary] c2f_pipeline_tvd", pipe.get("mean"))
    print("[summary] ipf_tvd", ipf.get("mean"))
PY

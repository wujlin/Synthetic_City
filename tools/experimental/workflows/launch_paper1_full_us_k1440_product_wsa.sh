#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/jinlin/miniconda3/envs/dpl/bin/python}"
TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
COARSE_PRESET="${COARSE_PRESET:-fine_1440}"
COARSE_LABEL="${COARSE_LABEL:-${COARSE_PRESET#fine_}}"
if [[ "$COARSE_LABEL" =~ ^[0-9] ]]; then
  COARSE_LABEL="k${COARSE_LABEL}"
fi
ROOT_RUN="${ROOT_RUN:-/home/jinlin/projects/Synthetic_City/outputs/_paper1_full_us_spatial_population_2023_${COARSE_LABEL}_geoidfix_${TS}}"
SEED="${SEED:-2}"

RAW_ROOT="${RAW_ROOT:-/home/jinlin/data/geoexplicit_data}"
DATA_ROOT="${DATA_ROOT:-$RAW_ROOT/synthetic_city/data}"
BASE_JOINT="${BASE_JOINT:-$DATA_ROOT/us/processed/external_targets/exttarget_v1_full_earn_pums_2023_puma_us_joint_wide.csv}"
BASE_SCHEMA="${BASE_SCHEMA:-$DATA_ROOT/us/processed/external_targets/exttarget_v1_full_earn_pums_2023_puma_us.schema.json}"
BASE_CONDITION="${BASE_CONDITION:-$DATA_ROOT/us/processed/external_conditions/extcond_v1_earn_v1_acs5_2022_puma_us.csv}"
STAGE1_CONDITION_SCALE_MODE="${STAGE1_CONDITION_SCALE_MODE:-none}"
STAGE1_CONDITION_EXTRA_CSV="${STAGE1_CONDITION_EXTRA_CSV:-}"
STAGE1_CONDITION_EXTRA_STANDARDIZE="${STAGE1_CONDITION_EXTRA_STANDARDIZE:-}"
STAGE1_CONDITION_EXTRA_MISSING_POLICY="${STAGE1_CONDITION_EXTRA_MISSING_POLICY:-}"
DC_SOURCE_RUN="${DC_SOURCE_RUN:-/home/jinlin/projects/Synthetic_City/outputs/_paper1_dc_spatial_population_2023_20260517T024302Z}"
DC_JOINT="${DC_JOINT:-$DC_SOURCE_RUN/data/external_targets/exttarget_v1_full_earn_pums_2023_puma_state11_joint_wide.csv}"
DC_CONDITION="${DC_CONDITION:-$DC_SOURCE_RUN/data/external_conditions/extcond_v1_earn_v1_acs5_2022_puma_state11.csv}"
ASSET_INVENTORY="${ASSET_INVENTORY:-/home/jinlin/projects/Synthetic_City/outputs/_paper1_full_us_spatial_population_2023_available51_ak2016fallback_with_dc_20260517T101849Z/metrics/asset_inventory_ready_joint_intersection.csv}"
DC_ASSET_INVENTORY="${DC_ASSET_INVENTORY:-/home/jinlin/projects/Synthetic_City/outputs/_paper1_full_us_spatial_population_2023_available50_excl_ak_with_dc_20260517T024836Z/metrics/asset_inventory_ready_joint_intersection.csv}"

SWEEP_ROOT="${SWEEP_ROOT:-/mnt/data_hdd/synthetic_city_runs/_coarse_dim_sensitivity_wsa_poilodes_k_sweep_20260525T125854Z}"
if [[ "$COARSE_PRESET" == "fine_960" ]]; then
  DEFAULT_MODEL_RUN="$SWEEP_ROOT/fine_960/seed${SEED}"
  DEFAULT_STAGE1_CKPT="$DEFAULT_MODEL_RUN/stage1/checkpoints/leave_mi_out/best.pt"
  DEFAULT_STAGE2_WIDE="$DEFAULT_MODEL_RUN/c2f_data/extc2f_full_earn_fine_960_stage1ipfcondmix_pums_2023_puma_us_wide.csv"
  DEFAULT_STAGE2_SCHEMA="$DEFAULT_MODEL_RUN/c2f_data/extc2f_full_earn_fine_960_stage1ipfcondmix_pums_2023_puma_us.schema.json"
  DEFAULT_STAGE2_CKPT="$DEFAULT_MODEL_RUN/stage2/checkpoints/external_c2f_full_earn_teacher/leave_mi_out/best.pt"
  STAGE1_CONDITION_EXTRA_CSV="${STAGE1_CONDITION_EXTRA_CSV:-/home/jinlin/projects/Synthetic_City/data/us/processed/features/puma_spatial_poi_lodes_us_v1.csv}"
  STAGE1_CONDITION_EXTRA_STANDARDIZE="${STAGE1_CONDITION_EXTRA_STANDARDIZE:-zscore}"
  STAGE1_CONDITION_EXTRA_MISSING_POLICY="${STAGE1_CONDITION_EXTRA_MISSING_POLICY:-require}"
else
  DEFAULT_MODEL_RUN="/mnt/data_hdd/synthetic_city_runs/_coarse_dim_sensitivity_wsa_highK_fine1440_seed2_20260519T031023Z/fine_1440/seed2"
  DEFAULT_STAGE1_CKPT="$DEFAULT_MODEL_RUN/stage1/checkpoints/leave_mi_out/best.pt"
  DEFAULT_STAGE2_WIDE="$DEFAULT_MODEL_RUN/c2f_data/extc2f_full_earn_fine_1440_stage1ipfcondmix_pums_2023_puma_us_wide.csv"
  DEFAULT_STAGE2_SCHEMA="$DEFAULT_MODEL_RUN/c2f_data/extc2f_full_earn_fine_1440_stage1ipfcondmix_pums_2023_puma_us.schema.json"
  DEFAULT_STAGE2_CKPT="/mnt/data_hdd/synthetic_city_runs/_coarse_dim_sensitivity_wsa_highK_fine1440_seed2_structlowmem_20260520T011111Z/fine_1440/seed2/stage2/checkpoints/external_c2f_full_earn_teacher/leave_mi_out/best.pt"
fi
STAGE1_CONDITION_EXTRA_STANDARDIZE="${STAGE1_CONDITION_EXTRA_STANDARDIZE:-none}"
STAGE1_CONDITION_EXTRA_MISSING_POLICY="${STAGE1_CONDITION_EXTRA_MISSING_POLICY:-require}"
STAGE1_CKPT="${STAGE1_CKPT:-$DEFAULT_STAGE1_CKPT}"
STAGE2_WIDE="${STAGE2_WIDE:-$DEFAULT_STAGE2_WIDE}"
STAGE2_SCHEMA="${STAGE2_SCHEMA:-$DEFAULT_STAGE2_SCHEMA}"
STAGE2_CKPT="${STAGE2_CKPT:-$DEFAULT_STAGE2_CKPT}"

EXPORT_GPU_ID="${EXPORT_GPU_ID:-1}"
STAGE2_N_EVAL_JOINT_SAMPLES="${STAGE2_N_EVAL_JOINT_SAMPLES:-64}"
IPF_ITERS="${IPF_ITERS:-200}"
STATES="${STATES:-ready}"
PUMA_UIDS="${PUMA_UIDS:-}"
STATE_WORKERS="${STATE_WORKERS:-1}"
N_JOBS_PER_STATE="${N_JOBS_PER_STATE:-4}"
SAMPLE_N_PER_STATE="${SAMPLE_N_PER_STATE:-100000}"
RUN_NATIONAL_QC="${RUN_NATIONAL_QC:-1}"
RUN_RELEASE_EXPORT="${RUN_RELEASE_EXPORT:-1}"
RELEASE_CHUNKSIZE="${RELEASE_CHUNKSIZE:-1000000}"
BUILD_CROSS_STATE_HOME_OUTBOUND_CACHE="${BUILD_CROSS_STATE_HOME_OUTBOUND_CACHE:-1}"
CROSS_STATE_HOME_OUTBOUND_CACHE_DIR="${CROSS_STATE_HOME_OUTBOUND_CACHE_DIR:-$ROOT_RUN/lodes_home_outbound_cache}"
CT_LEGACY_TRACT_ZIP="${CT_LEGACY_TRACT_ZIP:-$DATA_ROOT/us/processed/spatial_inputs_2023/state=09/raw/geo/tl_2020_09_tract.zip}"
STAGE1_EXTRA_ARGS=()
if [[ -n "$STAGE1_CONDITION_EXTRA_CSV" ]]; then
  STAGE1_EXTRA_ARGS=(--stage1_condition_extra_csv "$STAGE1_CONDITION_EXTRA_CSV")
fi

MODEL_DIR="$ROOT_RUN/model"
METRICS_DIR="$ROOT_RUN/metrics"
LOG_DIR="$ROOT_RUN/logs"
mkdir -p "$MODEL_DIR" "$METRICS_DIR" "$LOG_DIR"

COMBINED_JOINT="$MODEL_DIR/reference_joint_wide_50states_plus_dc.csv"
COMBINED_CONDITION="$MODEL_DIR/condition_50states_plus_dc.csv"
PRED_CSV="$MODEL_DIR/predicted_joint_wide_seed${SEED}_${COARSE_PRESET}_50states_plus_dc.csv"
PRED_NPZ="$MODEL_DIR/predicted_joint_wide_seed${SEED}_${COARSE_PRESET}_50states_plus_dc.npz"
PRED_SUMMARY="$MODEL_DIR/predicted_joint_wide_seed${SEED}_${COARSE_PRESET}_50states_plus_dc.summary.json"
INV_COPY="$METRICS_DIR/asset_inventory_ready_joint_intersection.csv"

"$PYTHON_BIN" - <<PY
from __future__ import annotations
import datetime as dt
import json
import pathlib
import shutil
import pandas as pd

base_joint = pathlib.Path("$BASE_JOINT")
base_cond = pathlib.Path("$BASE_CONDITION")
dc_joint = pathlib.Path("$DC_JOINT")
dc_cond = pathlib.Path("$DC_CONDITION")
asset_inventory = pathlib.Path("$ASSET_INVENTORY")
dc_asset_inventory = pathlib.Path("$DC_ASSET_INVENTORY")
out_joint = pathlib.Path("$COMBINED_JOINT")
out_cond = pathlib.Path("$COMBINED_CONDITION")
inv_copy = pathlib.Path("$INV_COPY")
summary_path = pathlib.Path("$ROOT_RUN/run_manifest.json")
states_arg = "$STATES".strip().lower()
selected_statefps = None
if states_arg not in {"", "ready", "all"}:
    selected_statefps = {s.strip().zfill(2) for s in "$STATES".split(",") if s.strip()}
selected_puma_uids = None
if "$PUMA_UIDS".strip():
    selected_puma_uids = {
        "".join(ch for ch in s.strip() if ch.isdigit()).zfill(7)
        for s in "$PUMA_UIDS".split(",")
        if s.strip()
    }

for p in [base_joint, base_cond, dc_joint, dc_cond, asset_inventory]:
    if not p.exists():
        raise SystemExit(f"required input not found: {p}")

out_joint.parent.mkdir(parents=True, exist_ok=True)
inv_copy.parent.mkdir(parents=True, exist_ok=True)

if not out_joint.exists() or "${FORCE_REBUILD_INPUTS:-0}" == "1":
    base = pd.read_csv(base_joint, low_memory=False)
    dc = pd.read_csv(dc_joint, low_memory=False)
    missing = [c for c in base.columns if c not in dc.columns]
    if missing:
        raise SystemExit(f"DC joint missing base columns: {missing[:10]}")
    dc = dc[base.columns]
    joint = pd.concat([base, dc], ignore_index=True)
    joint["puma_uid"] = joint["puma_uid"].astype(str).str.replace(r"\\.0$", "", regex=True).str.zfill(7)
    joint["statefp"] = joint["puma_uid"].str.slice(0, 2)
    if selected_statefps is not None:
        joint = joint[joint["statefp"].isin(selected_statefps)].copy()
    if selected_puma_uids is not None:
        joint = joint[joint["puma_uid"].isin(selected_puma_uids)].copy()
    joint.to_csv(out_joint, index=False)

if not out_cond.exists() or "${FORCE_REBUILD_INPUTS:-0}" == "1":
    base_c = pd.read_csv(base_cond, low_memory=False)
    dc_c = pd.read_csv(dc_cond, low_memory=False)
    missing = [c for c in base_c.columns if c not in dc_c.columns]
    if missing:
        raise SystemExit(f"DC condition missing base columns: {missing[:10]}")
    dc_c = dc_c[base_c.columns]
    cond = pd.concat([base_c, dc_c], ignore_index=True)
    cond["puma_uid"] = cond["puma_uid"].astype(str).str.replace(r"\\.0$", "", regex=True).str.zfill(7)
    cond["statefp"] = cond["puma_uid"].str.slice(0, 2)
    if selected_statefps is not None:
        cond = cond[cond["statefp"].isin(selected_statefps)].copy()
    if selected_puma_uids is not None:
        cond = cond[cond["puma_uid"].isin(selected_puma_uids)].copy()
    cond.to_csv(out_cond, index=False)

required_asset_cols = [
    "tract_zip",
    "targets_long_csv",
    "tract_puma_csv",
    "roads_path",
    "lodes_main_path",
    "lodes_aux_path",
    "wac_path",
]

shutil.copy2(asset_inventory, inv_copy)
inv = pd.read_csv(inv_copy, dtype={"statefp": str})
inv["statefp"] = inv["statefp"].astype(str).str.replace(r"\\.0$", "", regex=True).str.zfill(2)
dc_missing_before_fix: list[str] = []
if (inv["statefp"] == "11").any():
    dc_row = inv.loc[inv["statefp"] == "11"].iloc[0]
    for col in required_asset_cols:
        val = dc_row.get(col, "")
        if col in inv.columns and (pd.isna(val) or not str(val).strip() or str(val).strip().lower() == "nan"):
            dc_missing_before_fix.append(col)
else:
    dc_missing_before_fix = required_asset_cols[:]

if dc_missing_before_fix:
    if not dc_asset_inventory.exists():
        raise SystemExit(f"DC asset inventory not found for repair: {dc_asset_inventory}")
    dc_inv = pd.read_csv(dc_asset_inventory, dtype={"statefp": str})
    dc_inv["statefp"] = dc_inv["statefp"].astype(str).str.replace(r"\\.0$", "", regex=True).str.zfill(2)
    dc_good = dc_inv.loc[dc_inv["statefp"] == "11"]
    if dc_good.empty:
        raise SystemExit(f"DC asset inventory has no statefp=11 row: {dc_asset_inventory}")
    if not (inv["statefp"] == "11").any():
        inv = pd.concat([inv, dc_good.iloc[[0]][inv.columns.intersection(dc_good.columns)]], ignore_index=True)
    for col in inv.columns:
        if col in dc_good.columns:
            inv.loc[inv["statefp"] == "11", col] = dc_good.iloc[0][col]
    inv.to_csv(inv_copy, index=False)

required_target_vars = {"AGEP_SEX_cross", "SCHL_25p", "ESR_16p", "PINCP_16p_bin"}
target_repairs: list[dict[str, str]] = []
target_failures: list[str] = []
inv_for_target_check = inv
if selected_statefps is not None:
    inv_for_target_check = inv[inv["statefp"].isin(selected_statefps)].copy()

def target_missing_vars(path: pathlib.Path) -> set[str]:
    if not path.exists():
        return set(required_target_vars)
    vars_seen = set(pd.read_csv(path, usecols=["variable"])["variable"].astype(str).unique().tolist())
    return required_target_vars - vars_seen

for idx, row in inv_for_target_check.iterrows():
    statefp = str(row.get("statefp", "")).replace(".0", "").zfill(2)
    target_path = pathlib.Path(str(row.get("targets_long_csv", "")))
    missing = target_missing_vars(target_path)
    if not missing:
        continue
    alt_path = target_path.with_name(target_path.stem + "_agesex" + target_path.suffix)
    alt_missing = target_missing_vars(alt_path)
    if not alt_missing:
        inv.loc[idx, "targets_long_csv"] = str(alt_path)
        target_repairs.append({"statefp": statefp, "from": str(target_path), "to": str(alt_path)})
        continue
    target_failures.append(
        f"state={statefp} target={target_path} missing={sorted(missing)}; "
        f"alt={alt_path} alt_missing={sorted(alt_missing)}"
    )

if target_failures:
    detail = "\\n".join(target_failures[:10])
    raise SystemExit(
        "Asset inventory is not compatible with joint age-sex tract constraints. "
        "Build spatial inputs with tools/data/build_full_us_spatial_inputs.py --include_age_sex_cross first. "
        f"Examples:\\n{detail}"
    )
if target_repairs:
    inv.to_csv(inv_copy, index=False)

joint_head = pd.read_csv(out_joint, usecols=["statefp", "puma_uid", "total_person_weight"], dtype={"statefp": str, "puma_uid": str})
payload = {
    "created_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    "run_dir": "$ROOT_RUN",
    "target_product": "Full-US point-level spatial synthetic population from $COARSE_PRESET coarse-to-fine model.",
    "coarse_preset": "$COARSE_PRESET",
    "joint_reference_csv": str(out_joint),
    "condition_csv": str(out_cond),
    "stage1_condition_scale_mode": "$STAGE1_CONDITION_SCALE_MODE",
    "stage1_condition_extra_csv": "$STAGE1_CONDITION_EXTRA_CSV",
    "stage1_condition_extra_standardize": "$STAGE1_CONDITION_EXTRA_STANDARDIZE",
    "stage1_condition_extra_missing_policy": "$STAGE1_CONDITION_EXTRA_MISSING_POLICY",
    "asset_inventory_csv": str(inv_copy),
    "asset_inventory_source_csv": str(asset_inventory),
    "dc_asset_inventory_repair_csv": str(dc_asset_inventory),
    "dc_inventory_missing_before_fix": dc_missing_before_fix,
    "target_constraint_required_variables": sorted(required_target_vars),
    "target_inventory_agesex_repairs": target_repairs,
    "stage1_checkpoint": "$STAGE1_CKPT",
    "stage2_wide_csv": "$STAGE2_WIDE",
    "stage2_schema_json": "$STAGE2_SCHEMA",
    "stage2_checkpoint": "$STAGE2_CKPT",
    "n_pumas_reference": int(joint_head.shape[0]),
    "n_statefps_reference": int(joint_head["statefp"].astype(str).str.zfill(2).nunique()),
    "reference_population_total": float(pd.to_numeric(joint_head["total_person_weight"], errors="coerce").fillna(0).sum()),
    "asset_inventory_rows": int(inv.shape[0]),
    "asset_inventory_ready_rows": int((inv.get("status", "ready").astype(str) == "ready").sum()) if "status" in inv.columns else int(inv.shape[0]),
    "export_gpu_id": "$EXPORT_GPU_ID",
    "state_workers": int("$STATE_WORKERS"),
    "n_jobs_per_state": int("$N_JOBS_PER_STATE"),
    "allow_cross_state_work": True,
    "build_cross_state_home_outbound_cache": "$BUILD_CROSS_STATE_HOME_OUTBOUND_CACHE" == "1",
    "cross_state_home_outbound_cache_dir": "$CROSS_STATE_HOME_OUTBOUND_CACHE_DIR",
    "work_destination_profile": "od_preserving",
}
summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
PY

if [[ ! -s "$PRED_CSV" || "${FORCE_EXPORT:-0}" == "1" ]]; then
  echo "[export] start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  CUDA_VISIBLE_DEVICES="$EXPORT_GPU_ID" \
  SYNTHETIC_CITY_C2F_COARSE_PRESET="$COARSE_PRESET" \
  "$PYTHON_BIN" -u tools/experimental/workflows/export_c2f_full_earn_all_puma_joint_wide.py \
    --coarse_preset "$COARSE_PRESET" \
    --stage1_joint_wide_csv "$COMBINED_JOINT" \
    --stage1_schema_json "$BASE_SCHEMA" \
    --stage1_condition_csv "$COMBINED_CONDITION" \
    --stage1_condition_schema_json "$BASE_SCHEMA" \
    --stage1_condition_scale_mode "$STAGE1_CONDITION_SCALE_MODE" \
    "${STAGE1_EXTRA_ARGS[@]}" \
    --stage1_condition_extra_standardize "$STAGE1_CONDITION_EXTRA_STANDARDIZE" \
    --stage1_condition_extra_missing_policy "$STAGE1_CONDITION_EXTRA_MISSING_POLICY" \
    --stage1_checkpoint "$STAGE1_CKPT" \
    --stage1_timesteps 200 \
    --stage1_batch_size 512 \
    --stage2_wide_csv "$STAGE2_WIDE" \
    --stage2_schema_json "$STAGE2_SCHEMA" \
    --stage2_checkpoint "$STAGE2_CKPT" \
    --stage2_n_eval_joint_samples "$STAGE2_N_EVAL_JOINT_SAMPLES" \
    --ipf_iters "$IPF_ITERS" \
    --heldout_statefp_for_scaler 26 \
    --device cuda \
    --seed "$SEED" \
    --out_csv "$PRED_CSV" \
    --out_npz "$PRED_NPZ" \
    --out_summary_json "$PRED_SUMMARY" \
    --progress_every 25 \
    2>&1 | tee "$LOG_DIR/export_${COARSE_PRESET}_joint_wide.log"
else
  echo "[export] skip existing $PRED_CSV"
fi

if [[ "${RUN_FINE2400_SEED1_AFTER_EXPORT:-0}" == "1" && -n "${FINE2400_ROOT_RUN:-}" ]]; then
  mkdir -p "$FINE2400_ROOT_RUN"
  echo "[fine2400-age8] launch seed1 on gpu ${EXPORT_GPU_ID} after k1440 export"
  (
    WRITE_MANIFEST=0 \
    ROOT_RUN="$FINE2400_ROOT_RUN" \
    PRESETS=fine_2400_age8 \
    SEEDS=1 \
    GPU_ID="$EXPORT_GPU_ID" \
    PYTHON_BIN="$PYTHON_BIN" \
    STAGE2_STRUCTURED_LOW_MEMORY="${STAGE2_STRUCTURED_LOW_MEMORY:-1}" \
    STRUCTURED_LOW_MEMORY="${STRUCTURED_LOW_MEMORY:-1}" \
    STAGE2_READ_CHUNKSIZE="${STAGE2_READ_CHUNKSIZE:-5000}" \
    READ_CHUNKSIZE="${READ_CHUNKSIZE:-5000}" \
    FLUSH_ROWS="${FLUSH_ROWS:-5000}" \
    STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-4096}" \
    STAGE2_N_EVAL_JOINT_SAMPLES="${FINE2400_STAGE2_N_EVAL_JOINT_SAMPLES:-128}" \
    PIPELINE_N_EVAL_JOINT_SAMPLES="${FINE2400_PIPELINE_N_EVAL_JOINT_SAMPLES:-64}" \
    EVAL_DEVICE="${EVAL_DEVICE:-cuda}" \
    HELDOUT_STATEFP="${HELDOUT_STATEFP:-26}" \
    bash tools/experimental/workflows/run_paper1_coarse_dim_sensitivity_formal.sh
  ) > "$FINE2400_ROOT_RUN/queue_worker_gpu${EXPORT_GPU_ID}_seed1_after_k1440_export.log" 2>&1 &
  echo "$!" > "$FINE2400_ROOT_RUN/seed1_after_k1440_export.pid"
fi

echo "[state-batch] start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
STATE_BATCH_EXTRA_ARGS=()
if [[ "$BUILD_CROSS_STATE_HOME_OUTBOUND_CACHE" == "1" ]]; then
  echo "[lodes-home-outbound-cache] start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  CT_CACHE_ARGS=()
  if [[ -s "$CT_LEGACY_TRACT_ZIP" ]]; then
    CT_CACHE_ARGS=(--ct_legacy_tract_zip "$CT_LEGACY_TRACT_ZIP")
  fi
  "$PYTHON_BIN" tools/data/build_lodes_home_outbound_cache.py \
    --asset_inventory_csv "$INV_COPY" \
    --cache_dir "$CROSS_STATE_HOME_OUTBOUND_CACHE_DIR" \
    --home_statefps "$STATES" \
    "${CT_CACHE_ARGS[@]}" \
    2>&1 | tee "$ROOT_RUN/lodes_home_outbound_cache.log"
  STATE_BATCH_EXTRA_ARGS=(--cross_state_home_outbound_cache_dir "$CROSS_STATE_HOME_OUTBOUND_CACHE_DIR")
elif [[ -n "$CROSS_STATE_HOME_OUTBOUND_CACHE_DIR" && -d "$CROSS_STATE_HOME_OUTBOUND_CACHE_DIR" ]]; then
  STATE_BATCH_EXTRA_ARGS=(--cross_state_home_outbound_cache_dir "$CROSS_STATE_HOME_OUTBOUND_CACHE_DIR")
fi

"$PYTHON_BIN" tools/experimental/workflows/launch_paper1_full_us_state_batch.py \
  --repo_root "$ROOT_DIR" \
  --run_dir "$ROOT_RUN" \
  --asset_inventory_csv "$INV_COPY" \
  --joint_wide_csv "$PRED_CSV" \
  --schema_json "$BASE_SCHEMA" \
  --states "$STATES" \
  --state_workers "$STATE_WORKERS" \
  --n_jobs_per_state "$N_JOBS_PER_STATE" \
  --sample_n_per_state "$SAMPLE_N_PER_STATE" \
  --seed "$SEED" \
  --allow_cross_state_work \
  "${STATE_BATCH_EXTRA_ARGS[@]}" \
  --work_destination_profile od_preserving \
  2>&1 | tee "$ROOT_RUN/state_batch.stdout.log"

if [[ "$RUN_NATIONAL_QC" == "1" ]]; then
  echo "[national-qc] start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$PYTHON_BIN" tools/spatial/aggregate_paper1_spatial_national_qc.py \
    --run_dir "$ROOT_RUN" \
    --sample_n 100000 \
    --seed "$SEED" \
    2>&1 | tee "$ROOT_RUN/national_qc_aggregate.log"
fi

if [[ "$RUN_RELEASE_EXPORT" == "1" ]]; then
  echo "[release-csv] start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$PYTHON_BIN" tools/release/export_paper1_release_csv.py \
    --run_dir "$ROOT_RUN" \
    --states "$STATES" \
    --chunksize "$RELEASE_CHUNKSIZE" \
    2>&1 | tee "$ROOT_RUN/release_csv_export.log"
fi

echo "[ok] ${COARSE_PRESET} full-US run: $ROOT_RUN"

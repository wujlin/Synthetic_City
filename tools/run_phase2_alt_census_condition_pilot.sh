#!/usr/bin/env bash
set -euo pipefail

# Alternate ACS-condition robustness pilot for Phase 2.
#
# Question answered by this run:
# - If the PUMA->tract allocator keeps AGEP/SEX as hard feasibility constraints,
#   but replaces the main ACS soft prior with a different Census universe
#   definition (SCHL_25p / ESR_16p / PINCP_16p_bin), does the allocation remain
#   stable and interpretable?
#
# Required inputs:
#   JOINT_WIDE_CSV
#   SCHEMA_JSON
#   GROUP_TO_REGION_CSV
#
# Optional:
#   TARGETS_LONG_CSV      If unset, build a tract-level ACS targets_long file on the fly.
#   DATA_ROOT             Defaults to repo-local data root logic through the Python script.
#   PUMA_UIDS             Optional comma-separated region slice.
#   RUN_DIR               Optional explicit output directory.
#   LABEL                 Optional run label.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

JOINT_WIDE_CSV="${JOINT_WIDE_CSV:-}"
SCHEMA_JSON="${SCHEMA_JSON:-}"
GROUP_TO_REGION_CSV="${GROUP_TO_REGION_CSV:-}"
TARGETS_LONG_CSV="${TARGETS_LONG_CSV:-}"
PUMA_UIDS="${PUMA_UIDS:-}"
RUN_DIR="${RUN_DIR:-}"
LABEL="${LABEL:-phase2_alt_census_condition_pilot}"
ACS_YEAR="${ACS_YEAR:-2022}"
STATEFP="${STATEFP:-26}"

if [[ -z "$JOINT_WIDE_CSV" || -z "$SCHEMA_JSON" || -z "$GROUP_TO_REGION_CSV" ]]; then
  echo "[err] JOINT_WIDE_CSV, SCHEMA_JSON, and GROUP_TO_REGION_CSV are required." >&2
  exit 1
fi

if [[ -z "$TARGETS_LONG_CSV" ]]; then
  TARGETS_LONG_CSV="$REPO_ROOT/data/detroit/processed/marginals/acs5_${ACS_YEAR}_marginals_long_tract_state${STATEFP}_michigan.csv"
  echo "[info] TARGETS_LONG_CSV not set; building $TARGETS_LONG_CSV"
  "$PYTHON_BIN" -u "$REPO_ROOT/tools/build_acs_targets_long_michigan.py" \
    --data_root "$REPO_ROOT/data" \
    --acs_year "$ACS_YEAR" \
    --statefp "$STATEFP" \
    --tables "B01001,B15003,B20001,B23025" \
    --overwrite
fi

CMD=(
  "$PYTHON_BIN" -u "$REPO_ROOT/tools/exp_phase2_puma_to_small_area.py"
  --joint_wide_csv "$JOINT_WIDE_CSV"
  --schema_json "$SCHEMA_JSON"
  --targets_long_csv "$TARGETS_LONG_CSV"
  --prior_targets_csv "$TARGETS_LONG_CSV"
  --group_to_region_csv "$GROUP_TO_REGION_CSV"
  --group_col "tract_geoid"
  --region_col "puma_uid"
  --statefp "$STATEFP"
  --hard_variables "AGEP_bin,SEX"
  --prior_variables "SCHL_25p,ESR_16p,PINCP_16p_bin"
  --label "$LABEL"
)

if [[ -n "$PUMA_UIDS" ]]; then
  CMD+=(--puma_uids "$PUMA_UIDS")
fi

if [[ -n "$RUN_DIR" ]]; then
  CMD+=(--run_dir "$RUN_DIR")
fi

echo "[info] running alternate ACS-condition pilot"
printf '  %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"

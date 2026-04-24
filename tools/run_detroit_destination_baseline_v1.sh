#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/jinlin/projects/Synthetic_City}"
cd "$PROJECT_ROOT"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate dpl

MODE="${MODE:-contextual}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"

PERSONS_PATH="${PERSONS_PATH:-/home/jinlin/projects/Synthetic_City/outputs/_phase25_detroitmsa_core_households_20260329T093404Z/synthetic/persons_with_households.parquet}"
AREAS_PATH="${AREAS_PATH:-/home/jinlin/data/geoexplicit_data/synthetic_city/data/tl_2023_26_tract.zip}"
ROADS_PATH="${ROADS_PATH:-/home/jinlin/data/geoexplicit_data/synthetic_city/data/reference/geo_synthetic_pop_usa/road/MI_road_cleaned.shp.zip}"
TRACT_OD_PATH="${TRACT_OD_PATH:-/home/jinlin/projects/Synthetic_City/outputs/_prepare_detroit_lodes_tract_od_detroitmsa_core_2020_homecenterfix_20260330T050954Z/tract_od.csv}"
MOBILITY_CSV="${MOBILITY_CSV:-/home/jinlin/data/Mobility_Data/place20190206.csv}"
TIGER_BG_ZIP="${TIGER_BG_ZIP:-/home/jinlin/data/geoexplicit_data/synthetic_city/data/detroit/raw/geo/tiger/TIGER2023/tl_2023_26_bg.zip}"

DEST_ACCESS_COL="${DEST_ACCESS_COL:-work_access_jobs_gravity}"
DEST_CENTER_COL="${DEST_CENTER_COL:-work_access_job_centers_gravity}"

DISTANCE_BETA="${DISTANCE_BETA:-0.08}"

DEST_SEGMENT_WEIGHT="0.0"
DEST_ACCESS_WEIGHT="0.0"
OD_PAIR_PRIOR_COL="${OD_PAIR_PRIOR_COL:-}"
OD_PAIR_PRIOR_WEIGHT="${OD_PAIR_PRIOR_WEIGHT:-0.0}"
DEST_CENTER_WEIGHT="0.0"
SAME_COUNTY_WEIGHT="0.0"
SAME_HOME_CENTER_WEIGHT="0.0"
ASSIGNMENT_MODE="independent"
DISTANCE_EARN_MAP=""
DEST_ACCESS_EARN_MAP=""
DEST_CENTER_EARN_MAP=""
SAME_COUNTY_EARN_MAP=""
SAME_HOME_CENTER_EARN_MAP=""

case "$MODE" in
  distance_only)
    LABEL_CORE="detroit_dest_distance_only"
    ;;
  gravity_lodes)
    LABEL_CORE="detroit_dest_gravity_lodes"
    DEST_SEGMENT_WEIGHT="1.0"
    DEST_ACCESS_WEIGHT="0.5"
    ASSIGNMENT_MODE="independent"
    ;;
  contextual)
    LABEL_CORE="detroit_dest_contextual"
    DEST_SEGMENT_WEIGHT="1.0"
    DEST_ACCESS_WEIGHT="0.5"
    DEST_CENTER_WEIGHT="0.5"
    SAME_COUNTY_WEIGHT="0.15"
    SAME_HOME_CENTER_WEIGHT="0.25"
    ASSIGNMENT_MODE="hierarchical_county_center"
    DISTANCE_EARN_MAP="CE01:1.15,CE02:1.0,CE03:0.9"
    DEST_ACCESS_EARN_MAP="CE01:0.95,CE02:1.0,CE03:1.05"
    DEST_CENTER_EARN_MAP="CE01:0.85,CE02:1.0,CE03:1.15"
    SAME_COUNTY_EARN_MAP="CE01:1.1,CE02:1.0,CE03:0.9"
    SAME_HOME_CENTER_EARN_MAP="CE01:1.2,CE02:1.0,CE03:0.9"
    ;;
  mobility_m1)
    LABEL_CORE="detroit_dest_mobility_m1"
    DEST_SEGMENT_WEIGHT="1.0"
    DEST_ACCESS_WEIGHT="0.5"
    OD_PAIR_PRIOR_COL="${OD_PAIR_PRIOR_COL:-mobility_od_residual}"
    OD_PAIR_PRIOR_WEIGHT="${OD_PAIR_PRIOR_WEIGHT:-0.5}"
    DEST_CENTER_WEIGHT="0.5"
    SAME_COUNTY_WEIGHT="0.15"
    SAME_HOME_CENTER_WEIGHT="0.25"
    ASSIGNMENT_MODE="hierarchical_county_center"
    DISTANCE_EARN_MAP="CE01:1.15,CE02:1.0,CE03:0.9"
    DEST_ACCESS_EARN_MAP="CE01:0.95,CE02:1.0,CE03:1.05"
    DEST_CENTER_EARN_MAP="CE01:0.85,CE02:1.0,CE03:1.15"
    SAME_COUNTY_EARN_MAP="CE01:1.1,CE02:1.0,CE03:0.9"
    SAME_HOME_CENTER_EARN_MAP="CE01:1.2,CE02:1.0,CE03:0.9"
    ;;
  mobility_county_m2)
    LABEL_CORE="detroit_dest_mobility_county_m2"
    DEST_SEGMENT_WEIGHT="1.0"
    DEST_ACCESS_WEIGHT="0.5"
    OD_PAIR_PRIOR_COL="${OD_PAIR_PRIOR_COL:-mobility_work_county_residual}"
    OD_PAIR_PRIOR_WEIGHT="${OD_PAIR_PRIOR_WEIGHT:-0.5}"
    DEST_CENTER_WEIGHT="0.5"
    SAME_COUNTY_WEIGHT="0.0"
    SAME_HOME_CENTER_WEIGHT="0.25"
    ASSIGNMENT_MODE="hierarchical_county"
    DISTANCE_EARN_MAP="CE01:1.15,CE02:1.0,CE03:0.9"
    DEST_ACCESS_EARN_MAP="CE01:0.95,CE02:1.0,CE03:1.05"
    DEST_CENTER_EARN_MAP="CE01:0.85,CE02:1.0,CE03:1.15"
    SAME_COUNTY_EARN_MAP="CE01:1.1,CE02:1.0,CE03:0.9"
    SAME_HOME_CENTER_EARN_MAP="CE01:1.2,CE02:1.0,CE03:0.9"
    ;;
  mobility_center_m3)
    LABEL_CORE="detroit_dest_mobility_center_m3"
    DEST_SEGMENT_WEIGHT="1.0"
    DEST_ACCESS_WEIGHT="0.5"
    OD_PAIR_PRIOR_COL="${OD_PAIR_PRIOR_COL:-mobility_work_center_residual}"
    OD_PAIR_PRIOR_WEIGHT="${OD_PAIR_PRIOR_WEIGHT:-0.5}"
    DEST_CENTER_WEIGHT="0.5"
    SAME_COUNTY_WEIGHT="0.0"
    SAME_HOME_CENTER_WEIGHT="0.0"
    ASSIGNMENT_MODE="hierarchical_county_center"
    DISTANCE_EARN_MAP="CE01:1.15,CE02:1.0,CE03:0.9"
    DEST_ACCESS_EARN_MAP="CE01:0.95,CE02:1.0,CE03:1.05"
    DEST_CENTER_EARN_MAP="CE01:0.85,CE02:1.0,CE03:1.15"
    SAME_COUNTY_EARN_MAP="CE01:1.1,CE02:1.0,CE03:0.9"
    SAME_HOME_CENTER_EARN_MAP="CE01:1.2,CE02:1.0,CE03:0.9"
    ;;
  mobility_center_topk_m4)
    LABEL_CORE="detroit_dest_mobility_center_topk_m4"
    DEST_SEGMENT_WEIGHT="1.0"
    DEST_ACCESS_WEIGHT="0.5"
    OD_PAIR_PRIOR_COL="${OD_PAIR_PRIOR_COL:-mobility_center_topk_bonus_k10}"
    OD_PAIR_PRIOR_WEIGHT="${OD_PAIR_PRIOR_WEIGHT:-0.5}"
    DEST_CENTER_WEIGHT="0.5"
    SAME_COUNTY_WEIGHT="0.15"
    SAME_HOME_CENTER_WEIGHT="0.25"
    ASSIGNMENT_MODE="hierarchical_county_center"
    DISTANCE_EARN_MAP="CE01:1.15,CE02:1.0,CE03:0.9"
    DEST_ACCESS_EARN_MAP="CE01:0.95,CE02:1.0,CE03:1.05"
    DEST_CENTER_EARN_MAP="CE01:0.85,CE02:1.0,CE03:1.15"
    SAME_COUNTY_EARN_MAP="CE01:1.1,CE02:1.0,CE03:0.9"
    SAME_HOME_CENTER_EARN_MAP="CE01:1.2,CE02:1.0,CE03:0.9"
    ;;
  *)
    echo "Unsupported MODE=$MODE. Use one of: distance_only, gravity_lodes, contextual, mobility_m1, mobility_county_m2, mobility_center_m3, mobility_center_topk_m4" >&2
    exit 2
    ;;
esac

PHASE3B_DIR="${PHASE3B_DIR:-/home/jinlin/projects/Synthetic_City/outputs/_phase3b_${LABEL_CORE}_${RUN_TAG}}"
PHASE3_DIR="${PHASE3_DIR:-/home/jinlin/projects/Synthetic_City/outputs/_phase3_${LABEL_CORE}_nj48_${RUN_TAG}}"
VALIDATE_DIR="${VALIDATE_DIR:-/home/jinlin/projects/Synthetic_City/outputs/_phase3_validate_${LABEL_CORE}_${RUN_TAG}}"

echo "[info] MODE=$MODE"
echo "[info] PERSONS_PATH=$PERSONS_PATH"
echo "[info] TRACT_OD_PATH=$TRACT_OD_PATH"
echo "[info] PHASE3B_DIR=$PHASE3B_DIR"
echo "[info] PHASE3_DIR=$PHASE3_DIR"
echo "[info] VALIDATE_DIR=$VALIDATE_DIR"

if [[ "$MODE" == "mobility_m1" || "$MODE" == "mobility_county_m2" || "$MODE" == "mobility_center_m3" || "$MODE" == "mobility_center_topk_m4" ]]; then
  python - <<PY
import pandas as pd
path = r"${TRACT_OD_PATH}"
col = r"${OD_PAIR_PRIOR_COL}"
df = pd.read_csv(path, nrows=5)
if col not in df.columns:
    raise SystemExit(f"${MODE} requires column '{col}' in {path}")
print(f"[info] confirmed mobility prior column: {col}")
PY
fi

python tools/exp_phase3b_assign_work_destinations.py \
  --persons_path "$PERSONS_PATH" \
  --tract_od_path "$TRACT_OD_PATH" \
  --work_eligible_col is_worker \
  --distance_col distance_km \
  --distance_beta "$DISTANCE_BETA" \
  --earn_col EARN_16p_bin \
  --age_col AGEP_bin \
  --schl_col SCHL_allpop \
  --destination_segment_weight "$DEST_SEGMENT_WEIGHT" \
  --destination_access_col "$DEST_ACCESS_COL" \
  --destination_access_weight "$DEST_ACCESS_WEIGHT" \
  --od_pair_prior_col "$OD_PAIR_PRIOR_COL" \
  --od_pair_prior_weight "$OD_PAIR_PRIOR_WEIGHT" \
  --destination_center_col "$DEST_CENTER_COL" \
  --destination_center_weight "$DEST_CENTER_WEIGHT" \
  --same_county_weight "$SAME_COUNTY_WEIGHT" \
  --same_home_center_weight "$SAME_HOME_CENTER_WEIGHT" \
  --distance_earn_multiplier_map "$DISTANCE_EARN_MAP" \
  --destination_access_earn_multiplier_map "$DEST_ACCESS_EARN_MAP" \
  --destination_center_earn_multiplier_map "$DEST_CENTER_EARN_MAP" \
  --same_county_earn_multiplier_map "$SAME_COUNTY_EARN_MAP" \
  --same_home_center_earn_multiplier_map "$SAME_HOME_CENTER_EARN_MAP" \
  --assignment_mode "$ASSIGNMENT_MODE" \
  --seed 0 \
  --run_dir "$PHASE3B_DIR" \
  --label "phase3b_${LABEL_CORE}"

python tools/exp_phase3_road_locations.py \
  --persons_path "$PHASE3B_DIR/synthetic/persons_with_worktract.parquet" \
  --areas_path "$AREAS_PATH" \
  --roads_path "$ROADS_PATH" \
  --group_col tract_geoid \
  --work_group_col work_tract_geoid \
  --areas_group_col GEOID \
  --person_id_col person_id \
  --household_col household_id \
  --work_eligible_col is_worker \
  --home_mode conservative \
  --work_mtfcc_values S1100,S1200 \
  --work_gap_exception_mtfcc_values S1400 \
  --legalization_fraction 1e-6 \
  --home_interpolation_density 0.0005 \
  --work_interpolation_density 0.0002 \
  --n_jobs 48 \
  --parallel_chunksize 32 \
  --seed 0 \
  --run_dir "$PHASE3_DIR" \
  --label "phase3_${LABEL_CORE}_nj48"

python tools/exp_phase3_validate_mobility_anchor.py \
  --mobility_csv "$MOBILITY_CSV" \
  --synthetic_person_locations "$PHASE3_DIR/synthetic/person_locations.csv" \
  --tiger_bg_zip "$TIGER_BG_ZIP" \
  --min_home_secs 21600 \
  --min_work_secs 10800 \
  --min_home_work_distance_m 500 \
  --run_dir "$VALIDATE_DIR" \
  --label "phase3_validate_${LABEL_CORE}"

echo "[done] baseline run finished for MODE=$MODE"

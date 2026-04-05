#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/jinlin/projects/Synthetic_City}"
cd "$PROJECT_ROOT"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate dpl

PERSONS_PATH="${PERSONS_PATH:-/home/jinlin/projects/Synthetic_City/outputs/_phase25_detroitmsa_core_households_20260329T093404Z/synthetic/persons_with_households.parquet}"
AREAS_PATH="${AREAS_PATH:-/home/jinlin/data/geoexplicit_data/synthetic_city/data/tl_2023_26_tract.zip}"
ROADS_PATH="${ROADS_PATH:-/home/jinlin/data/geoexplicit_data/synthetic_city/data/reference/geo_synthetic_pop_usa/road/MI_road_cleaned.shp.zip}"
TRACT_OD_PATH="${TRACT_OD_PATH:-/home/jinlin/projects/Synthetic_City/outputs/_prepare_detroit_poi_attractiveness_detroitmsa_core_homecenter_20260331_pilot/tract_od.csv}"
MOBILITY_CSV="${MOBILITY_CSV:-/home/jinlin/data/Mobility_Data/place20190206.csv}"
TIGER_BG_ZIP="${TIGER_BG_ZIP:-/home/jinlin/data/geoexplicit_data/synthetic_city/data/detroit/raw/geo/tiger/TIGER2023/tl_2023_26_bg.zip}"
DEST_ACCESS_COL="${DEST_ACCESS_COL:-work_access_jobs_poi_blend}"

PHASE3B_DIR="${PHASE3B_DIR:-/home/jinlin/projects/Synthetic_City/outputs/_phase3b_assign_work_destinations_detroitmsa_core_homecenter_poi_blend_20260331_pilot}"
PHASE3_DIR="${PHASE3_DIR:-/home/jinlin/projects/Synthetic_City/outputs/_phase3_roadloc_detroitmsa_core_homecenter_poi_blend_nj48_20260331_pilot}"
VALIDATE_DIR="${VALIDATE_DIR:-/home/jinlin/projects/Synthetic_City/outputs/_phase3_validate_mobility_anchor_detroitmsa_core_homecenter_poi_blend_20260331_pilot}"

python tools/exp_phase3b_assign_work_destinations.py \
  --persons_path "$PERSONS_PATH" \
  --tract_od_path "$TRACT_OD_PATH" \
  --work_eligible_col is_worker \
  --distance_col distance_km \
  --distance_beta 0.08 \
  --earn_col EARN_16p_bin \
  --age_col AGEP_bin \
  --schl_col SCHL_allpop \
  --destination_segment_weight 1.0 \
  --destination_access_col "$DEST_ACCESS_COL" \
  --destination_access_weight 0.5 \
  --destination_center_col work_access_job_centers_gravity \
  --destination_center_weight 0.5 \
  --same_county_weight 0.15 \
  --same_home_center_weight 0.25 \
  --distance_earn_multiplier_map CE01:1.15,CE02:1.0,CE03:0.9 \
  --destination_access_earn_multiplier_map CE01:0.95,CE02:1.0,CE03:1.05 \
  --destination_center_earn_multiplier_map CE01:0.85,CE02:1.0,CE03:1.15 \
  --same_county_earn_multiplier_map CE01:1.1,CE02:1.0,CE03:0.9 \
  --same_home_center_earn_multiplier_map CE01:1.2,CE02:1.0,CE03:0.9 \
  --assignment_mode hierarchical_county_center \
  --seed 0 \
  --run_dir "$PHASE3B_DIR" \
  --label phase3b_assign_work_destinations_detroitmsa_core_homecenter_poi_blend

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
  --label phase3_roadloc_detroitmsa_core_homecenter_poi_blend_nj48

python tools/exp_phase3_validate_mobility_anchor.py \
  --mobility_csv "$MOBILITY_CSV" \
  --synthetic_person_locations "$PHASE3_DIR/synthetic/person_locations.csv" \
  --tiger_bg_zip "$TIGER_BG_ZIP" \
  --min_home_secs 21600 \
  --min_work_secs 10800 \
  --min_home_work_distance_m 500 \
  --run_dir "$VALIDATE_DIR" \
  --label phase3_validate_mobility_anchor_detroitmsa_core_homecenter_poi_blend

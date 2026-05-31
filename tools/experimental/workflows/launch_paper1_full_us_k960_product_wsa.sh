#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export COARSE_PRESET="${COARSE_PRESET:-fine_960}"
export COARSE_LABEL="${COARSE_LABEL:-k960}"
export SEED="${SEED:-1}"
export STAGE1_CONDITION_EXTRA_CSV="${STAGE1_CONDITION_EXTRA_CSV:-/home/jinlin/projects/Synthetic_City/data/us/processed/features/puma_spatial_poi_lodes_us_v1.csv}"
export STAGE1_CONDITION_EXTRA_STANDARDIZE="${STAGE1_CONDITION_EXTRA_STANDARDIZE:-zscore}"
export STAGE1_CONDITION_EXTRA_MISSING_POLICY="${STAGE1_CONDITION_EXTRA_MISSING_POLICY:-require}"

exec "$SCRIPT_DIR/launch_paper1_full_us_k1440_product_wsa.sh"

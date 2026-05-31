#!/usr/bin/env bash
set -euo pipefail

# Dataset-level replication for Paper 1.
# This builds the 2022 ACS/PUMS full-earn artifacts and then runs the same
# state-held-out hierarchical evaluation used for the 2023 main results.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RAW_ROOT="${RAW_ROOT:-/home/jinlin/data/geoexplicit_data}"
DATA_ROOT="${SYNTHCITY_DATA_ROOT:-$RAW_ROOT/synthetic_city/data}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"

PUMS_YEAR="${PUMS_YEAR:-2022}"
PUMS_PERIOD="${PUMS_PERIOD:-5-Year}"
ACS_YEAR="${ACS_YEAR:-2022}"
SCOPE_TAG="${SCOPE_TAG:-us}"
DATASET_LABEL="${DATASET_LABEL:-pums${PUMS_YEAR}_acs${ACS_YEAR}}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"

STATEFP_LIST="${STATEFP_LIST:-26 12 48 55}"
SEED_LIST="${SEED_LIST:-0}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
DOWNLOAD_PUMS_IF_MISSING="${DOWNLOAD_PUMS_IF_MISSING:-1}"

PUMS_DIR="${PUMS_DIR:-$DATA_ROOT/us/raw/pums/pums_${PUMS_YEAR}_${PUMS_PERIOD}}"
COND_DIR="${COND_DIR:-$DATA_ROOT/us/processed/external_conditions}"
TARGET_DIR="${TARGET_DIR:-$DATA_ROOT/us/processed/external_targets}"

BASE_CONDITION_CSV="${BASE_CONDITION_CSV:-$COND_DIR/extcond_v1_acs5_${ACS_YEAR}_puma_${SCOPE_TAG}.csv}"
EARN_CONDITION_CSV="${EARN_CONDITION_CSV:-$COND_DIR/extcond_earn_v1_acs5_${ACS_YEAR}_puma_${SCOPE_TAG}.csv}"
MERGED_CONDITION_CSV="${MERGED_CONDITION_CSV:-$COND_DIR/extcond_v1_earn_v1_acs5_${ACS_YEAR}_puma_${SCOPE_TAG}.csv}"
EARN_TARGET_CSV="${EARN_TARGET_CSV:-$TARGET_DIR/exttarget_earn_cond_v1_pums_${PUMS_YEAR}_puma_${SCOPE_TAG}.csv}"
FULL_TARGET_WIDE_CSV="${FULL_TARGET_WIDE_CSV:-$TARGET_DIR/exttarget_v1_full_earn_pums_${PUMS_YEAR}_puma_${SCOPE_TAG}_joint_wide.csv}"
FULL_TARGET_SCHEMA_JSON="${FULL_TARGET_SCHEMA_JSON:-$TARGET_DIR/exttarget_v1_full_earn_pums_${PUMS_YEAR}_puma_${SCOPE_TAG}.schema.json}"

RUN_DIR="${RUN_DIR:-outputs/_paper1_2022_dataset_replication_prepare_${TIMESTAMP}}"
mkdir -p "$RUN_DIR" "$COND_DIR" "$TARGET_DIR"

export RAW_ROOT DATA_ROOT PYTHON_BIN ACS_YEAR PUMS_YEAR PUMS_PERIOD SCOPE_TAG
export OUT_CSV="$BASE_CONDITION_CSV"

run_or_skip() {
  local target_path="$1"
  shift
  if [[ "$FORCE_REBUILD" != "1" && -f "$target_path" ]]; then
    echo "[skip] exists: $target_path"
    return 0
  fi
  echo "[run] $*"
  "$@"
}

{
  echo "[replication] dataset=$DATASET_LABEL"
  echo "[replication] DATA_ROOT=$DATA_ROOT"
  echo "[replication] PUMS_DIR=$PUMS_DIR"
  echo "[replication] ACS_YEAR=$ACS_YEAR PUMS_YEAR=$PUMS_YEAR"
  echo "[replication] STATEFP_LIST=$STATEFP_LIST SEED_LIST=$SEED_LIST"

  if [[ "$SKIP_PREPARE" != "1" ]]; then
    if [[ ! -d "$PUMS_DIR" ]]; then
      if [[ "$DOWNLOAD_PUMS_IF_MISSING" == "1" ]]; then
        echo "[run] PUMS_DIR not found; downloading ACS PUMS ${PUMS_YEAR} ${PUMS_PERIOD} for all states"
        "$PYTHON_BIN" -u tools/detroit_fetch_public_data.py pums \
          --out_root "$DATA_ROOT" \
          --pums_year "$PUMS_YEAR" \
          --pums_period "$PUMS_PERIOD" \
          --all_states
      else
        echo "[err] PUMS_DIR not found: $PUMS_DIR" >&2
        echo "[hint] Set DOWNLOAD_PUMS_IF_MISSING=1 or stage ACS PUMS ${PUMS_YEAR} ${PUMS_PERIOD} under this directory." >&2
        exit 2
      fi
    fi

    if [[ ! -d "$PUMS_DIR" ]]; then
      echo "[err] PUMS_DIR still not found after download attempt: $PUMS_DIR" >&2
      exit 2
    fi

    run_or_skip "$BASE_CONDITION_CSV" \
      bash tools/model/run_external_condition_v1_us.sh

    run_or_skip "$EARN_CONDITION_CSV" \
      "$PYTHON_BIN" -u tools/data/build_external_condition_earn_v1_acs_puma.py \
        --all_states \
        --acs_year "$ACS_YEAR" \
        --out_path "$EARN_CONDITION_CSV" \
        --overwrite

    if [[ "$FORCE_REBUILD" == "1" || ! -f "$MERGED_CONDITION_CSV" ]]; then
      "$PYTHON_BIN" -u tools/data/merge_external_condition_v1_with_earn.py \
        --base_condition_csv "$BASE_CONDITION_CSV" \
        --earn_condition_csv "$EARN_CONDITION_CSV" \
        --out_path "$MERGED_CONDITION_CSV" \
        --overwrite
    else
      echo "[skip] exists: $MERGED_CONDITION_CSV"
    fi

    run_or_skip "$EARN_TARGET_CSV" \
      "$PYTHON_BIN" -u tools/data/build_external_target_earn_conditional_v1_us.py \
        --all_states \
        --pums_year "$PUMS_YEAR" \
        --pums_period "$PUMS_PERIOD" \
        --pums_dir "$PUMS_DIR" \
        --out_dir "$TARGET_DIR"

    run_or_skip "$FULL_TARGET_WIDE_CSV" \
      "$PYTHON_BIN" -u tools/data/build_external_target_v1_full_earn.py \
        --conditional_target_csv "$EARN_TARGET_CSV" \
        --condition_csv "$MERGED_CONDITION_CSV" \
        --pums_year "$PUMS_YEAR" \
        --scope_tag "$SCOPE_TAG" \
        --out_dir "$TARGET_DIR" \
        --overwrite
  fi

  "$PYTHON_BIN" - <<PY
import json
from pathlib import Path

payload = {
    "task": "paper1_2022_dataset_replication_prepare",
    "dataset_label": "$DATASET_LABEL",
    "acs_year": int("$ACS_YEAR"),
    "pums_year": int("$PUMS_YEAR"),
    "pums_period": "$PUMS_PERIOD",
    "scope": "$SCOPE_TAG",
    "statefp_list": "$STATEFP_LIST".split(),
    "seed_list": "$SEED_LIST".split(),
    "data_root": "$DATA_ROOT",
    "pums_dir": "$PUMS_DIR",
    "base_condition_csv": "$BASE_CONDITION_CSV",
    "earn_condition_csv": "$EARN_CONDITION_CSV",
    "merged_condition_csv": "$MERGED_CONDITION_CSV",
    "earn_target_csv": "$EARN_TARGET_CSV",
    "full_target_wide_csv": "$FULL_TARGET_WIDE_CSV",
    "full_target_schema_json": "$FULL_TARGET_SCHEMA_JSON",
    "run_dir": "$RUN_DIR",
    "prepare_only": bool(int("$PREPARE_ONLY")),
    "skip_prepare": bool(int("$SKIP_PREPARE")),
    "download_pums_if_missing": bool(int("$DOWNLOAD_PUMS_IF_MISSING")),
}
Path("$RUN_DIR").mkdir(parents=True, exist_ok=True)
(Path("$RUN_DIR") / "run_summary.json").write_text(json.dumps(payload, indent=2) + "\\n")
print(json.dumps(payload, indent=2))
PY

  if [[ "$PREPARE_ONLY" == "1" ]]; then
    echo "[ok] prepare_only=1; stopping before training/evaluation."
    exit 0
  fi

  PUMS_YEAR="$PUMS_YEAR" \
    ACS_YEAR="$ACS_YEAR" \
    SCOPE_TAG="$SCOPE_TAG" \
    DATASET_LABEL="$DATASET_LABEL" \
    STATEFP_LIST="$STATEFP_LIST" \
    SEED_LIST="$SEED_LIST" \
    TIMESTAMP="$TIMESTAMP" \
    PYTHON_BIN="$PYTHON_BIN" \
    DEVICE="$DEVICE" \
    JOINT_WIDE_CSV="$FULL_TARGET_WIDE_CSV" \
    SCHEMA_JSON="$FULL_TARGET_SCHEMA_JSON" \
    CONDITION_CSV="$MERGED_CONDITION_CSV" \
    bash tools/experimental/workflows/run_paper1_e2_state_holdout_batch.sh
} 2>&1 | tee "$RUN_DIR/run.log"

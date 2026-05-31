#!/usr/bin/env bash
set -euo pipefail

# Sequential batch wrapper for the Paper 1 E2 state-held-out validation.
# This intentionally runs one state at a time to avoid GPU contention.

STATEFP_LIST="${STATEFP_LIST:-55 12 48}"
SEED_LIST="${SEED_LIST:-0}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
PUMS_YEAR="${PUMS_YEAR:-2023}"
ACS_YEAR="${ACS_YEAR:-2022}"
DATASET_LABEL="${DATASET_LABEL:-pums${PUMS_YEAR}_acs${ACS_YEAR}}"
LOG_DIR="${LOG_DIR:-outputs/_paper1_E2_state_holdout_${DATASET_LABEL}_batch_${TIMESTAMP}}"
mkdir -p "$LOG_DIR"

for statefp in $STATEFP_LIST; do
  for seed in $SEED_LIST; do
    log_path="$LOG_DIR/state${statefp}_seed${seed}.log"
    echo "[batch] dataset=$DATASET_LABEL state=$statefp seed=$seed log=$log_path"
    STATEFP="$statefp" \
      SEED="$seed" \
      TIMESTAMP="${TIMESTAMP}" \
      PUMS_YEAR="$PUMS_YEAR" \
      ACS_YEAR="$ACS_YEAR" \
      DATASET_LABEL="$DATASET_LABEL" \
      SCOPE_TAG="${SCOPE_TAG:-us}" \
      DATA_ROOT="${DATA_ROOT:-/home/jinlin/data/geoexplicit_data/synthetic_city/data}" \
      JOINT_WIDE_CSV="${JOINT_WIDE_CSV:-}" \
      SCHEMA_JSON="${SCHEMA_JSON:-}" \
      CONDITION_CSV="${CONDITION_CSV:-}" \
      bash tools/experimental/workflows/run_paper1_e2_state_holdout_full_earn.sh 2>&1 | tee "$log_path"
  done
done

python - <<PY
import json, pathlib
root = pathlib.Path("$LOG_DIR")
rows = []
for p in sorted(pathlib.Path("outputs").glob(f"_paper1_E2_full_earn_state*_seed*_${TIMESTAMP}_eval/run_manifest.json")):
    rows.append(json.loads(p.read_text()))
for p in sorted(pathlib.Path("outputs").glob(f"_paper1_E2_full_earn_${DATASET_LABEL}_state*_seed*_${TIMESTAMP}_eval/run_manifest.json")):
    obj = json.loads(p.read_text())
    if obj not in rows:
        rows.append(obj)
payload = {
    "timestamp": "$TIMESTAMP",
    "dataset_label": "$DATASET_LABEL",
    "pums_year": int("$PUMS_YEAR"),
    "acs_year": int("$ACS_YEAR"),
    "scope": "${SCOPE_TAG:-us}",
    "statefp_list": "$STATEFP_LIST".split(),
    "seed_list": "$SEED_LIST".split(),
    "runs": rows,
}
(root / "batch_summary.json").write_text(json.dumps(payload, indent=2) + "\\n")
print(json.dumps({"timestamp": "$TIMESTAMP", "dataset_label": "$DATASET_LABEL", "n_runs": len(rows), "runs": rows}, indent=2))
PY

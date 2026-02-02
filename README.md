# Synthetic City

This repository supports a research project on **building-level, spatiotemporal synthetic population generation** using **diffusion models** and **explicit, reviewable constraint modules**.

The code is intentionally KISS: each module has a single responsibility, and the PoC scripts are designed to produce small, auditable outputs (JSON metrics + metadata) that can be synced via GitHub.

## Key documents

- NSFC writing drafts: `NSFC/`
- Detroit code/data blueprint: `docs/detroit_code_data_structure.md`
- Method-to-module mapping: `docs/synthpop_architecture.md`
- Data search notes: `docs/DATA_SEARCH.md`
- Workstation guide: `docs/WORKSTATION_GUIDE.md`

## Repository layout

```text
NSFC/        # proposal drafts
docs/        # literature notes, blueprints, procurement notes
src/         # reusable synthpop modules (library-style)
tools/       # runnable scripts (PoC + data utilities)
outputs/     # small run artifacts committed for review (see .gitignore)
data/        # optional symlink to an external data root (NOT committed)
```

## Model architecture (current)

### Scheme B (PoC, runnable)

**Goal:** learn realistic *attribute joint structure* from PUMS, then anchor to buildings via an explicit allocator.

1) **Attribute generator (diffusion)** learns:

- `P(attrs | macro_geo)` where `macro_geo = PUMA` for the PoC
- `attrs = {AGEP, PINCP, SEX}` (ESR is removed in PoC to avoid semantic conflicts for minors)
- PUMS data note: for `AGEP < 16`, `PINCP` can be missing (not in universe); the PoC imputes missing child `PINCP = 0.0` to avoid dropping all children during cleaning.

2) **Spatial anchoring (explicit post-processing)** assigns each generated person to a building within the same macro group:

- `bldg_id = f(attrs, buildings_in_group; method)`
- methods: `random`, `capacity_only`, `income_price_match` (ablation-ready)

3) **Validation**

- internal: generated vs PUMS holdout (grouped marginal TVD + key associations)
- external (optional): generated vs ACS-derived `targets_long` (currently age/sex only; income is household-level in ACS tables)

Code entrypoints:

- Orchestration: `tools/poc_tabddpm_pums_buildingcond.py`
- Diffusion model: `src/synthpop/model/diffusion_tabular.py`
- Building allocation: `src/synthpop/spatial/building_allocation.py`
- Metrics: `src/synthpop/validation/stats.py`

### ACS-supervised diffusion (PoC, runnable)

**Goal:** test whether **tract-level context (geo + built)** carries learnable signal for **age×sex** distributions, using **ACS B01001** as training supervision and **PUMS** as an external validation source (never used in training).

Core idea:

- Train on **pseudo-individuals sampled from ACS tract distributions**.
- Conditions are **tract_context** (ablations: `none`, `geo-only`, `built-only`, `geo+built`).
- Evaluation:
  - internal: tract-level TVD vs ACS on held-out tracts
  - external: aggregate tract predictions to PUMA and compare vs PUMS (plus an ACS→PUMA baseline gap)

Code entrypoint:

- `tools/poc_tabddpm_acs_supervised_b01001.py`

### Scheme C-v2 (WIP)

Shared latent encoders + alignment losses + joint diffusion guidance (skeleton + smoke tests exist, but not the default Detroit PoC yet). See `PI_Opinion.md` and `src/synthpop/model/joint_diffusion.py`.

## Setup

Recommended: use **conda** (geo stack + torch CUDA wheels are often painful with pure pip).

```bash
python -m pip install -r requirements.txt
```

Run unit tests:

```bash
python -m unittest discover -s tests
```

## Detroit data (workstation)

We recommend keeping raw/processed city data out of git and symlinking `data/` to an external disk path:

```bash
export RAW_ROOT=/home/jinlin/data/geoexplicit_data
mkdir -p "$RAW_ROOT/synthetic_city/data"
ln -snf "$RAW_ROOT/synthetic_city/data" data
```

Download P0 public data (TIGER/ACS/PUMS/OSM) and register SafeGraph (local symlink only):

```bash
python tools/detroit_fetch_public_data.py tiger --out_root "$RAW_ROOT/synthetic_city/data"
python tools/detroit_fetch_public_data.py acs --out_root "$RAW_ROOT/synthetic_city/data" --acs_year 2023 --tables "B01001,B19001,B23025"
python tools/detroit_fetch_public_data.py pums --out_root "$RAW_ROOT/synthetic_city/data" --pums_year 2023
python tools/detroit_fetch_public_data.py osm --out_root "$RAW_ROOT/synthetic_city/data" --region michigan
python tools/detroit_fetch_public_data.py safegraph --out_root "$RAW_ROOT/synthetic_city/data" --safegraph_dir "$RAW_ROOT/safegraph/safegraph_unzip"
```

## Build ACS targets_long (optional external validation)

This produces an **ACS-derived long table**: `(group, variable, category, target)`.

You can generate:
- **tract-level** targets: `group_col=tract_geoid` (recommended for spatial consistency checks)
- **PUMA-level** targets: `group_col=puma` (coarser, but more stable)

```bash
export DATA_ROOT="$RAW_ROOT/synthetic_city/data"

# (A) tract-level targets_long (group_col=tract_geoid)
python tools/build_acs_marginals_long.py \
  --out_root "$DATA_ROOT" \
  --acs_year 2023 \
  --tables "B01001,B19001,B23025" \
  --geo_level tract \
  --aggregate_to none

# (B) PUMA-level targets_long (tract -> puma aggregation)
python tools/build_acs_marginals_long.py \
  --out_root "$DATA_ROOT" \
  --acs_year 2023 \
  --tables "B01001,B19001,B23025" \
  --geo_level tract \
  --aggregate_to puma
```

Expected output (example for Wayne County, MI):

- `data/detroit/processed/marginals/acs5_2023_marginals_long_puma_state26_county163.csv`
- `data/detroit/processed/marginals/acs5_2023_marginals_long_tract_geoid_state26_county163.csv`

## Prepare building features (required for Scheme B allocation)

1) Build a Detroit building feature table (geometry-derived):

```bash
python tools/prepare_detroit_buildings_gba.py \
  --gba_tile "/path/to/LoD1.geojson" \
  --tiger_place_zip "$DATA_ROOT/detroit/raw/geo/tiger/TIGER2023/tl_2023_26_place.zip" \
  --tiger_puma_zip "$DATA_ROOT/detroit/raw/geo/tiger/TIGER2023/tl_2023_26_puma20.zip" \
  --tiger_tract_zip "$DATA_ROOT/detroit/raw/geo/tiger/TIGER2023/tl_2023_26_tract.zip" \
  --tiger_bg_zip "$DATA_ROOT/detroit/raw/geo/tiger/TIGER2023/tl_2023_26_bg.zip" \
  --out_csv "$DATA_ROOT/detroit/processed/buildings/buildings_detroit_features.csv"
```

2) Join parcels/assessment to get `price_tier` (used only by allocation, NOT by diffusion training):

```bash
python tools/join_detroit_buildings_parcel_assessment.py \
  --buildings_csv "$DATA_ROOT/detroit/processed/buildings/buildings_detroit_features.csv" \
  --parcels_path "$DATA_ROOT/detroit/raw/parcels/detroit_parcels_current" \
  --group_for_tier tract --n_tiers 5 \
  --out_csv "$DATA_ROOT/detroit/processed/buildings/buildings_detroit_features_price.csv"
```

## Run Scheme B PoC (train + sample + allocate + metrics)

```bash
export DATA_ROOT="$RAW_ROOT/synthetic_city/data"
export BLDG_CSV="$DATA_ROOT/detroit/processed/buildings/buildings_detroit_features_price.csv"
export ACS_LONG="$DATA_ROOT/detroit/processed/marginals/acs5_2023_marginals_long_puma_state26_county163.csv"
export ACS_LONG_TRACT="$DATA_ROOT/detroit/processed/marginals/acs5_2023_marginals_long_tract_geoid_state26_county163.csv"
export OUT_DIR="$DATA_ROOT/detroit/outputs/runs/_poc_tabddpm_income_price_match_$(date -u +%Y%m%dT%H%M%SZ)"

PYTHONUNBUFFERED=1 python -u tools/poc_tabddpm_pums_buildingcond.py \
  --mode train-sample \
  --data_root "$DATA_ROOT" \
  --buildings_csv "$BLDG_CSV" \
  --allocation_method income_price_match \
  --n_tiers 5 \
  --acs_marginals_long "$ACS_LONG" \
  --acs_marginals_long_tract "$ACS_LONG_TRACT" \
  --n_rows 200000 \
  --epochs 1000 \
  --batch_size 4096 \
  --timesteps 200 \
  --n_samples 50000 \
  --device cuda \
  --out_dir "$OUT_DIR" \
  |& tee "$OUT_DIR/run.log"
```

Outputs (in `OUT_DIR`):

- `model.pt`, `encoder.json`, `train_summary.json`
- `samples_building.csv` (large; gitignored by default)
- `building_portrait.csv` (large; gitignored by default)
- `sample_summary.json`
- `metrics/stats_metrics.json` (PUMS holdout reference)
- `metrics/stats_metrics_acs.json` (ACS targets_long reference, if provided)
- `metrics/stats_metrics_acs_tract.json` (ACS tract-level targets_long reference, if provided)

Optional diagnosis helper (workstation-only; uses large `samples_building.csv`):

```bash
python tools/diagnose_tract_validation.py \
  --run_dir "$OUT_DIR" \
  --buildings_csv "$BLDG_CSV" \
  --acs_targets_long_tract "$ACS_LONG_TRACT"
```

Optional analysis: check whether building features are clustered by tract (necessary for any tract-aware conditioning idea):

```bash
python tools/analyze_building_feature_clustering.py \
  --buildings_csv "$BLDG_CSV" \
  --group_col tract_geoid
```

## Run ACS-supervised PoC (tract_context ablation + 4-fold PUMA-block CV)

```bash
export DATA_ROOT="$RAW_ROOT/synthetic_city/data"
export ACS_B01001="$DATA_ROOT/detroit/raw/census/acs/acs5_2023/acs5_2023_B01001_tract_state26_county163.csv.gz"
export BLDG_CSV="$DATA_ROOT/detroit/processed/buildings/buildings_detroit_features_price.csv"
export TIGER_TRACT="$DATA_ROOT/detroit/raw/geo/tiger/TIGER2023/tl_2023_26_tract.zip"
export TIGER_PUMA="$DATA_ROOT/detroit/raw/geo/tiger/TIGER2023/tl_2023_26_puma20.zip"
export OUT_DIR="$DATA_ROOT/detroit/outputs/runs/_poc_acs_supervised_b01001_$(date -u +%Y%m%dT%H%M%SZ)"

PYTHONUNBUFFERED=1 python -u tools/poc_tabddpm_acs_supervised_b01001.py \
  --acs_b01001_csv_gz "$ACS_B01001" \
  --buildings_csv "$BLDG_CSV" \
  --tiger_tract_zip "$TIGER_TRACT" \
  --tiger_puma_zip "$TIGER_PUMA" \
  --data_root "$DATA_ROOT" \
  --conditions "none,geo-only,built-only,geo+built" \
  --puma_blocks "3202,3203;3208,3209;3210,3211;3212,3213" \
  --epochs 1000 \
  --batch_size 4096 \
  --timesteps 200 \
  --n_eval_per_tract 2000 \
  --device cuda \
  --out_dir "$OUT_DIR" \
  |& tee "$OUT_DIR/run.log"
```

Alias entrypoint (same CLI):

```bash
PYTHONUNBUFFERED=1 python -u tools/poc_tabddpm_acs_tract.py --help
```

Key outputs (small, commit-friendly):

- `run_summary.json`
- `metrics/acs_pums_baseline_gap.json`
- `metrics/ablation_summary.json` (mean±std across folds)
- `fold_*/**/metrics/internal_acs_holdout.json`
- `fold_*/**/metrics/external_pums_by_puma.json` (if `--data_root` provided)

## Results syncing strategy

- Large artifacts (model checkpoints, large CSV/parquet) are ignored via `.gitignore`.
- Small, reviewable outputs (JSON metrics/metadata/logs) can be committed under `outputs/` for PI review.

# Synthetic City

Research code for creating a multi-attribute geographically explicit synthetic population for the United States.

## Data Release

The release-format synthetic population dataset is hosted on OSF:

<https://osf.io/e7wp8/>

The OSF release contains 332,387,543 synthetic individuals across 2,462 Public Use Microdata Areas (PUMAs) in the 50 states and Washington, D.C. Files are organized by state under `synthetic_population/data/csv_by_state/`. Each compressed CSV uses the USPS abbreviation in the filename, for example `synthetic_individuals_CA.csv.gz`.

Release columns:

```text
person_id, age, gender, education, employment, income,
home_lon, home_lat, work_lon, work_lat
```

The dataset is released under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License (CC BY-NC-ND 4.0). See the OSF `license.txt` for the exact license statement.

## Framework

The proposed generative framework separates joint distribution reconstruction from location assignment. Step 1 creates the target and condition vectors for each PUMA. Step 2 reconstructs the PUMA-specific five-attribute joint distribution from aggregated-level census conditions and spatial representation vectors. Step 3 samples synthetic individuals from the predicted joint distribution and assigns geographically explicit home and workplace locations.

### Step 1: Create Target and Condition Vectors

Step 1 prepares the training target and conditional vectors required by Step 2. For each PUMA, it creates one target vector and two condition vectors.

- `p`: a PUMS-derived five-attribute joint distribution over age, gender, education, employment, and income.
- `c`: a PUMA-level census condition vector built from ACS Detailed Tables.
- `h`: a spatial representation vector built from POI and LODES-derived features.

PUMS provides the observed individual-level co-occurrence patterns used to construct the target joint distribution. ACS Detailed Tables provide aggregated-level census conditions. POI and LODES features provide spatial information beyond census marginals, allowing the model to represent spatial non-stationarity across PUMAs.

Primary manuscript-facing entrypoints:

- PUMA target distribution `p`: `tools/workflow/step1_build_puma_joint_targets.py`
- PUMA census condition `c`: `tools/workflow/step1_build_puma_census_conditions.py`
- PUMA spatial representation `h`: `tools/workflow/step1_build_puma_spatial_features.py`
- POI feature vocabulary used in `h`: `docs/STEP1_POI_FEATURES.md`
- Tract-level ACS constraints and state spatial assets: `tools/data/build_full_us_spatial_inputs.py`
- LODES and TIGER support data helpers: `tools/data/download_lodes_functional_assets.py`, `tools/data/download_tiger2023_cache.py`, `tools/data/build_lodes_home_outbound_cache.py`
- Shared data loading utilities: `synthpop/data/census.py`, `synthpop/data/lodes.py`, `synthpop/data/geo.py`

### Step 2: Hierarchical Generative Process

Step 2 is the core hierarchical diffusion-based generative process. It reconstructs the PUMA-level joint distribution with two diffusion stages.

- Stage 1 generates a coarse joint distribution using the aggregated-level condition `c` and the spatial representation `h`.
- Stage 2 generates the fine-grained joint distribution by refining each coarse group into its fine-grained combinations.

The current paper configuration uses `K=960` as the default coarse state-space size in Stage 1. Stage 2 maps the coarse distribution back to the full 3,000-cell joint distribution over the five attributes. This setting follows the manuscript sensitivity analysis: `K=960` preserves plateau-level held-out accuracy while keeping the Stage 1 prediction target compact.

Primary manuscript-facing entrypoints:

- Stage 1 coarse model: `tools/workflow/step2_train_coarse_diffusion.py`
- Stage 2 refinement target construction: `tools/workflow/step2_build_refinement_targets.py`
- Stage 2 fine refinement model: `tools/workflow/step2_train_refinement_diffusion.py`
- End-to-end joint recovery evaluation: `tools/workflow/step2_eval_joint_recovery.py`
- Coarse-to-fine schema and support logic: `tools/model/external_c2f_full_earn_schema.py`, `tools/model/external_c2f_full_earn_stage2_model.py`

### Step 3: Generate Synthetic Individuals with Geographically Explicit Locations

Step 3 turns the predicted joint distribution into synthetic individuals with five attributes and geographically explicit locations.

- Individuals are sampled from the predicted PUMA-level joint distribution.
- Home tracts are assigned with tract-level ACS constraints, including the age-gender joint constraint.
- Home locations are placed along residential road-network supports within the assigned tracts.
- Employed individuals receive destination work tracts from LODES commuting flows.
- Workplace locations are placed on secondary-road and residential-intersection supports within the assigned destination tracts.

The resulting product is a national synthetic population with five attributes and home and workplace latitude-longitude values, where workplace locations apply to employed individuals.

Primary manuscript-facing entrypoints:

- Predicted joint distribution export: `tools/model/export_predicted_joint_wide_from_npz.py`
- Person-level expansion from PUMA distributions: `tools/workflow/step3_expand_individuals.py`
- Home tract allocation under tract ACS constraints: `tools/workflow/step3_assign_home_tracts.py`, `synthpop/spatial/puma_to_small_area.py`
- Work-destination tract allocation from LODES: `tools/workflow/step3_assign_work_tracts.py`, `synthpop/spatial/work_destination_allocation.py`
- Road-supported home and workplace coordinates: `tools/workflow/step3_assign_road_locations.py`, `synthpop/spatial/road_location_allocation.py`
- National QC aggregation: `tools/workflow/step3_aggregate_spatial_qc.py`

### Release Export

The final product is exported as state-level 10-column CSV files and uploaded to OSF.

Primary manuscript-facing entrypoints:

- Release-format CSV export: `tools/workflow/release_export_state_csv.py`
- OSF file naming and upload utilities: `tools/release/osf_rename_state_files_to_postal.py`, `tools/workflow/release_upload_osf.py`
- Manuscript data-product summaries and figures: `tools/figures/make_sigspatial_data_product.py`, `tools/figures/make_sigspatial_national_spatial_product.py`

## Data Sources

| Source | Role in the framework |
|---|---|
| Public Use Microdata Sample (PUMS) | Constructs the PUMA-level target joint distribution `p` |
| American Community Survey (ACS) Detailed Tables | Constructs the condition vector `c` and tract-level allocation constraints |
| Point-of-interest (POI) data | Constructs part of the spatial representation vector `h` |
| Longitudinal Employer-Household Dynamics Origin-Destination Employment Statistics (LODES) | Constructs part of `h` and provides destination tracts for workplace assignment |
| Road network | Assigns home and workplace locations within assigned tracts |

## Repository Layout

This is a research-first repository rather than a packaged software library. Reusable logic is kept under `synthpop/`, while experiment entrypoints and release utilities are under `tools/`.

```text
synthpop/       reusable package modules for data loading, constraints, models, spatial allocation, and validation
tools/workflow/    stable manuscript-facing Step 1, Step 2, Step 3, and release entrypoints
tools/data/        data construction, schema, ACS/PUMS/LODES/POI preparation
tools/model/       model training, evaluation, and coarse-to-fine backend utilities
tools/spatial/     individual expansion, home and workplace assignment, road-location placement, and spatial QC
tools/figures/     manuscript figure, table, and validation-summary builders
tools/release/     public CSV export, file naming, and OSF upload helpers
docs/              method notes, run notes, data contracts, and experiment summaries
tests/             lightweight smoke tests and regression checks
outputs/           local run artifacts, gitignored by default
data/              optional local link to external data roots, gitignored by default
```

## Implementation Map

The current national pipeline is script-driven. The table below links each manuscript step to the corresponding code names in this repository.

| Manuscript step | Main task | Code entrypoints |
|---|---|---|
| Step 1 | Build PUMS targets, ACS conditions, POI/LODES spatial features, and tract-level constraints | `tools/workflow/step1_build_puma_joint_targets.py`; `tools/workflow/step1_build_puma_census_conditions.py`; `tools/workflow/step1_build_puma_spatial_features.py`; `tools/data/build_full_us_spatial_inputs.py`; `synthpop/data/lodes.py` |
| Step 2 | Train and evaluate the hierarchical generative process | `tools/workflow/step2_train_coarse_diffusion.py`; `tools/workflow/step2_build_refinement_targets.py`; `tools/workflow/step2_train_refinement_diffusion.py`; `tools/workflow/step2_eval_joint_recovery.py` |
| Step 3 | Expand predicted distributions into individuals and assign geographically explicit home and workplace locations | `tools/workflow/step3_expand_individuals.py`; `tools/workflow/step3_assign_home_tracts.py`; `tools/workflow/step3_assign_work_tracts.py`; `tools/workflow/step3_assign_road_locations.py`; `tools/workflow/step3_aggregate_spatial_qc.py` |
| Release | Export the public dataset and synchronize OSF files | `tools/workflow/release_export_state_csv.py`; `tools/release/osf_rename_state_files_to_postal.py`; `tools/workflow/release_upload_osf.py` |

Use the `step*` and `release*` entrypoints when citing code paths in manuscript notes, slides, or repository documentation. Some backend implementation modules retain historical experiment labels for reproducibility; `docs/IMPLEMENTATION_MANIFEST.md` records the mapping from manuscript-facing names to those backend files.

Large raw data, licensed data, model checkpoints, and generated state-level products are intentionally kept out of git. The public repository tracks code, documentation, lightweight tests, and manuscript-supporting assets.

Directory guides:

- `docs/README.md`
- `synthpop/README.md`
- `tools/README.md`
- `tests/README.md`

## Getting Started

Install the Python dependencies in an isolated environment, then run the smoke tests:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests
```

For the manuscript framework and data schema, start with:

- `docs/DATA_CONTRACT.md`
- `docs/synthpop_architecture.md`
- `docs/IMPLEMENTATION_MANIFEST.md`

Most full-scale runs require external data roots and workstation-scale compute. Local users should treat `data/` and `outputs/` as machine-specific paths rather than versioned repository content.

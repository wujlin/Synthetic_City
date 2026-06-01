# Synthetic Population Architecture

The current framework separates three problems that should not be collapsed into one step:

1. build region-specific targets and conditions;
2. recover the PUMA-level joint distribution of individual attributes;
3. convert that distribution into individuals with explicit home and workplace locations.

This separation keeps the model interpretable and makes failures easier to diagnose.

## Step 1: Target and Condition Construction

For each Public Use Microdata Area (PUMA), the pipeline builds:

- `p`: a PUMS-derived joint distribution over age, gender, education, employment, and income;
- `c`: an ACS-derived census condition vector;
- `h`: a spatial representation vector derived from POI and LODES features.

PUMS supplies observed co-occurrence patterns. ACS supplies accessible aggregate conditions. POI and LODES provide spatial context beyond census marginals.

The POI vocabulary used in `h` is documented in `docs/STEP1_POI_FEATURES.md`.

Manuscript-facing entrypoints:

- `tools/workflow/step1_build_puma_joint_targets.py`
- `tools/workflow/step1_build_puma_census_conditions.py`
- `tools/workflow/step1_build_puma_spatial_features.py`

## Step 2: Hierarchical Diffusion

The model predicts the joint distribution in two stages.

Stage 1 predicts a coarse distribution conditioned on `c` and `h`. In the current paper configuration, the coarse space has `K=960` combinations.

Stage 2 refines each coarse group into fine-grained combinations and reconstructs the full 3,000-cell joint distribution over the five attributes.

The hierarchy reduces the learning burden while retaining the full final attribute space.

Manuscript-facing entrypoints:

- `tools/workflow/step2_train_coarse_diffusion.py`
- `tools/workflow/step2_build_refinement_targets.py`
- `tools/workflow/step2_train_refinement_diffusion.py`
- `tools/workflow/step2_eval_joint_recovery.py`

## Step 3: Spatial Population Generation

The predicted PUMA-level joint distribution is sampled into individuals. Home tracts are assigned with tract-level ACS constraints, including age-gender consistency. Home coordinates are placed on residential road-supported candidate points.

Workers receive work destination tracts from LODES commuting flows. Workplace coordinates are then placed on road-supported workplace candidate points.

Manuscript-facing entrypoints:

- `tools/workflow/step3_expand_individuals.py`
- `tools/workflow/step3_assign_home_tracts.py`
- `tools/workflow/step3_assign_work_tracts.py`
- `tools/workflow/step3_assign_road_locations.py`
- `tools/workflow/step3_aggregate_spatial_qc.py`

## Data Roles

| Data source | Role |
|---|---|
| ACS PUMS | PUMA-level target joint distribution |
| ACS Detailed Tables | PUMA-level conditions and tract-level allocation constraints |
| POI | Spatial representation |
| LODES | Spatial representation and work destination flows |
| Road network | Candidate supports for home and workplace coordinates |

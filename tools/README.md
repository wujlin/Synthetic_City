# Tools Layout

The `tools` directory is organized by manuscript role.

| Directory | Purpose |
|---|---|
| `workflow/` | Stable Step 1, Step 2, Step 3, and release entrypoints used in documentation |
| `data/` | ACS, PUMS, LODES, POI, schema, and spatial-input construction |
| `model/` | Hierarchical diffusion training, evaluation, and legacy model-run wrappers |
| `spatial/` | Synthetic individual expansion, home/work tract assignment, road-location placement, and spatial QC |
| `figures/` | Manuscript figure, table, and validation-summary builders |
| `release/` | State CSV export, OSF file naming, and upload helpers |
| `experimental/` | Exploratory and historical scripts kept outside the manuscript-facing pipeline |

Use `tools/workflow/*.py` in manuscript-facing documentation. Backend files in the other folders keep historical names when renaming would break reproducibility of previous run manifests.

## Recommended Entrypoints

The stable entrypoints are under `tools/workflow/`:

| Manuscript step | Entrypoints |
|---|---|
| Step 1 | `step1_build_puma_joint_targets.py`, `step1_build_puma_census_conditions.py`, `step1_build_puma_spatial_features.py` |
| Step 2 | `step2_train_coarse_diffusion.py`, `step2_build_refinement_targets.py`, `step2_train_refinement_diffusion.py`, `step2_eval_joint_recovery.py` |
| Step 3 | `step3_expand_individuals.py`, `step3_assign_home_tracts.py`, `step3_assign_work_tracts.py`, `step3_assign_road_locations.py`, `step3_aggregate_spatial_qc.py` |
| Release | `release_export_state_csv.py`, `release_upload_osf.py` |

Use `docs/IMPLEMENTATION_MANIFEST.md` when a manuscript-facing entrypoint delegates to a backend script with a historical name.

# Implementation Manifest

This manifest keeps repository file names aligned with the manuscript framework.
Use the manuscript-facing entrypoints below in the README, slides, paper notes, and issue discussions.
Backend files may retain older experiment labels so that existing run artifacts remain reproducible.

## Naming Rules

- `step1_*`: constructs PUMA-level targets and condition vectors.
- `step2_*`: trains or evaluates the hierarchical diffusion model.
- `step3_*`: converts predicted joint distributions into located synthetic individuals.
- `release_*`: exports or uploads the public release dataset.
- Avoid citing backend names such as `external`, `teacher`, `paper1`, or `phase` in manuscript-facing text.
- Keep generated run directories under `outputs/_task_setting_split_YYYYMMDDThhmmssZ`.

## Manuscript-Facing Entrypoints

| Manuscript component | Public entrypoint | Backend implementation |
|---|---|---|
| Step 1 target `p` | `tools/workflow/step1_build_puma_joint_targets.py` | `tools/data/build_external_target_v1_full_earn.py` |
| Step 1 census condition `c` | `tools/workflow/step1_build_puma_census_conditions.py` | `tools/data/build_external_condition_earn_v1_acs_puma.py` |
| Step 1 spatial representation `h` | `tools/workflow/step1_build_puma_spatial_features.py` | `tools/data/build_puma_spatial_features.py` |
| Step 1 tract constraints and spatial assets | `tools/data/build_full_us_spatial_inputs.py` | `tools/data/build_full_us_spatial_inputs.py` |
| Step 2 Stage 1 coarse model | `tools/workflow/step2_train_coarse_diffusion.py` | `tools/model/train_external_c2f_full_earn_stage1_coarse.py` |
| Step 2 Stage 2 refinement target | `tools/workflow/step2_build_refinement_targets.py` | `tools/model/build_external_c2f_full_earn_teacher.py` |
| Step 2 Stage 2 refinement model | `tools/workflow/step2_train_refinement_diffusion.py` | `tools/model/train_external_c2f_full_earn_teacher.py` |
| Step 2 held-out and national evaluation | `tools/workflow/step2_eval_joint_recovery.py` | `tools/model/eval_external_c2f_full_earn_pipeline.py` |
| Step 3 individual expansion | `tools/workflow/step3_expand_individuals.py` | `tools/spatial/exp_phase2_expand_to_persons.py` |
| Step 3 home tract assignment | `tools/workflow/step3_assign_home_tracts.py` | `tools/spatial/exp_phase2_puma_to_small_area.py` |
| Step 3 work tract assignment | `tools/workflow/step3_assign_work_tracts.py` | `tools/spatial/exp_phase3b_assign_work_destinations.py` |
| Step 3 road-supported coordinates | `tools/workflow/step3_assign_road_locations.py` | `tools/spatial/exp_phase3_road_locations.py` |
| Step 3 national spatial QC | `tools/workflow/step3_aggregate_spatial_qc.py` | `tools/spatial/aggregate_paper1_spatial_national_qc.py` |
| Release CSV export | `tools/workflow/release_export_state_csv.py` | `tools/release/export_paper1_release_csv.py` |
| Release OSF upload | `tools/workflow/release_upload_osf.py` | `tools/release/upload_osf_release_incremental.py` |

## Public Release Schema

The public CSV files should expose exactly these columns:

```text
person_id, age, gender, education, employment, income,
home_lon, home_lat, work_lon, work_lat
```

Intermediate identifiers such as `statefp`, `puma_uid`, `tract_geoid`, and worker flags belong in internal QC files, not in the release CSVs.

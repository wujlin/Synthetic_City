# Tests

This directory contains lightweight smoke tests and regression checks for reusable package logic and manuscript-facing pipeline components.

## Running Tests

Install dependencies, then run:

```bash
python -m unittest discover -s tests
```

The tests are designed to run without national-scale raw data, model checkpoints, or generated OSF release files. They focus on schema behavior, allocation logic, metric calculations, and small synthetic examples.

## Test Coverage

| Area | Representative tests |
|---|---|
| Diffusion and model utilities | `test_diffusion_tabular_smoke.py`, `test_joint_diffusion_smoke.py` |
| PUMA-to-small-area allocation | `test_puma_to_small_area_smoke.py`, `test_allocation_expansion_smoke.py` |
| LODES and work-destination logic | `test_lodes_geoid_crosswalk_smoke.py`, `test_work_destination_allocation_smoke.py` |
| Road-location placement | `test_road_location_allocation_smoke.py` |
| Validation metrics | `test_stats_metrics_smoke.py`, `test_stats_metrics_targets_smoke.py` |

Full national runs should be validated through run manifests, QC summaries, and held-out evaluation artifacts under gitignored `outputs/` directories rather than through unit tests.

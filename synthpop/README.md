# `synthpop` Package

`synthpop` contains reusable Python modules for the synthetic population pipeline. The package is intentionally small and implementation-focused: manuscript-facing execution should start from `tools/workflow/`, while shared logic lives here.

## Module Map

| Module | Purpose |
|---|---|
| `data/` | ACS, PUMS, LODES, POI, geography, and state-code loading utilities |
| `model/` | Diffusion model components for categorical and tabular joint-distribution recovery |
| `constraints/` | Projection, hard-rule, and soft-guidance utilities for constrained distributions |
| `spatial/` | PUMA-to-tract allocation, individual expansion, work-destination assignment, and road-location placement |
| `validation/` | Statistical and spatial validation metrics |
| `paths.py` | Path-resolution helpers for local and external data roots |
| `plot_style.py` | Shared manuscript-figure plotting defaults |

## Design Contract

The package should remain reusable and free of workstation-specific paths. Full-scale pipeline scripts, experiment launchers, OSF upload code, and figure builders belong under `tools/`.

The public data release exposes only:

```text
person_id, age, gender, education, employment, income,
home_lon, home_lat, work_lon, work_lat
```

Intermediate geographic identifiers and QC flags should stay in internal files and validation artifacts.

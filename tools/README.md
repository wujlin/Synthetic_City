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

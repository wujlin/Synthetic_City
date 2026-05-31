# Experimental Tools

This folder keeps exploratory and historical scripts that are not part of the
manuscript-facing pipeline in `tools/workflow/`.

| Directory | Contents |
|---|---|
| `data/` | External-feature builders for ACS uncertainty, POI, LODES, OSM roads, VIIRS, building morphology, and imagery tiles |
| `representation/` | SSL, external-view, residual, and projection-aware representation probes |
| `model/` | Residual-prior and heteroskedastic coarse-diffusion model variants |
| `workflows/` | Historical Paper 1, WSA launch, state-holdout, replication, and product-generation wrappers |

These files are retained for reproducibility and future reference. Stable code
that supports the current paper should be promoted into `tools/data/`,
`tools/model/`, `tools/spatial/`, `tools/figures/`, `tools/release/`, or
`tools/workflow/` before being cited in public documentation.

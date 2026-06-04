# Documentation

This directory contains repository documentation that supports the SIGSPATIAL 2026 synthetic population manuscript and the public OSF data release.

## Reading Order

| File | Purpose |
|---|---|
| `synthpop_architecture.md` | Method-level overview of the three-step framework: target and condition construction, hierarchical diffusion, and spatial population generation |
| `IMPLEMENTATION_MANIFEST.md` | Mapping between manuscript-facing step names and the backend scripts that implement them |
| `DATA_CONTRACT.md` | Storage rules, identifier conventions, release schema, and public dataset contract |
| `STEP1_POI_FEATURES.md` | POI feature vocabulary used in the Step 1 spatial representation vector `h` |

## Documentation Scope

These files should use the same terminology as the manuscript:

- `p`: PUMS-derived PUMA-level target joint distribution;
- `c`: ACS-derived PUMA-level census condition vector;
- `h`: POI/LODES-derived spatial representation vector;
- Step 1: target and condition construction;
- Step 2: hierarchical generative process;
- Step 3: synthetic individual generation and spatial assignment.

Avoid using historical run names such as `external`, `teacher`, `paper1`, or `phase` in manuscript-facing documentation unless the text is explicitly describing an implementation mapping.

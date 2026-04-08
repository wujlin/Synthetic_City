# Synthetic City

Research codebase for **synthetic population generation and spatialization** with a current focus on the **Detroit metropolitan area**. The repository combines two tightly related threads:

- **distribution-level demographic generation** from PUMS / ACS constraints
- **explicit home/work spatialization** from small-area population mass to reviewable road-supported locations

The project is organized around a simple idea: do not jump directly from marginals to point locations. Instead, separate three inferential tasks that should not be mixed together:

1. recover **who lives in a PUMA**
2. disaggregate that joint mass to **tract / CBG**
3. assign explicit **home / work locations** within each small area

## Project Status

The current Detroit pipeline is mature enough to support manuscript writing and figure production.

- **Phase 1**: recover PUMA-level demographic joint structure
- **Phase 2**: disaggregate PUMA mass to tract / CBG with ACS as the main anchor
- **Phase 2.5**: synthesize households so that home assignment is household-semantic
- **Phase 3**: assign explicit home/work locations under road-constrained supports

Current empirical takeaways:

- **home** is the strongest result and already has stable external validation
- **work destination tract + commute shape** are reasonable and publishable as qualified results
- **OD pairing** remains the main unresolved problem

## What Is In This Repository

This repo is not a polished software package. It is a **research-first working repository** that keeps reusable modules, experiment entrypoints, documentation, and manuscript assets in one place.

```text
src/         reusable library-style modules
tools/       experiment entrypoints, data prep, validation, figure scripts
docs/        methods notes, architecture notes, experiment findings, data guides
Essay/       manuscript drafts and paper figures
tests/       lightweight tests and smoke checks
data/        optional local link to external data root (not committed)
outputs/     run artifacts and synced experiment products (gitignored)
figures/     current figure products for manuscript/presentation use
```

## Architecture At A Glance

### Phase 1: Demographic Joint Recovery

Phase 1 reconstructs the **joint attribute structure** of a population within a macro geography such as a PUMA. In the repo this appears as diffusion-based tabular generation, conditional encoders, and alignment utilities.

Representative modules:

- `src/synthpop/model/`
- `src/synthpop/alignment/`
- `src/synthpop/features/condition_vectors.py`

Representative experiment scripts:

- `tools/train_us_puma_5var_diffusion.py`
- `tools/train_us_puma_3var_diffusion.py`
- `tools/poc_tabddpm_pums.py`

### Phase 2: PUMA -> Tract / CBG Disaggregation

Phase 2 is a **constrained decomposition** problem, not a person-level location prediction problem. ACS provides the main small-area anchor; mobility signals are useful mainly as residual spatial heterogeneity evidence or validation, not as the primary demographic truth.

Representative modules:

- `src/synthpop/spatial/puma_to_small_area.py`
- `src/synthpop/constraints/`
- `src/synthpop/data/acs_crosstab.py`

Representative experiment scripts:

- `tools/exp_phase2_puma_to_small_area.py`
- `tools/exp5_tract_postalign.py`

### Phase 2.5: Household Synthesis

This stage turns person-level synthetic population into **household-consistent** population so that home assignment is semantically correct.

Representative modules:

- `src/synthpop/spatial/tract_householding.py`

Representative experiment scripts:

- `tools/exp_phase25_synthesize_households.py`

### Phase 3: Explicit Home / Work Spatialization

Phase 3 assigns explicit locations within each tract / CBG.

- **home**: road-constrained residential support, household-aware assignment
- **work**: worker -> destination tract -> explicit work point

Representative modules:

- `src/synthpop/spatial/road_location_allocation.py`
- `src/synthpop/spatial/work_destination_allocation.py`
- `src/synthpop/validation/mobility_anchor.py`

Representative experiment and figure scripts:

- `tools/exp_phase3b_assign_work_destinations.py`
- `tools/exp_phase3_validate_mobility_anchor.py`
- `tools/viz_phase3_detroit_overviews.py`
- `tools/viz_phase3_validation_detroit.py`

## Data Roles

The project uses each data source for a specific inferential role rather than treating all of them as interchangeable evidence.

| Data source | Role in pipeline |
|---|---|
| **PUMS** | microdata seed for demographic joint structure |
| **ACS / Census** | tract / CBG population anchors and household constraints |
| **LODES** | work destination mass and OD skeleton |
| **TIGER roads / MTFCC** | explicit home/work candidate supports |
| **Mobility anchors** | external validation, local pattern checks, commute diagnostics |
| **POI / visit products** | optional refinement or secondary diagnostics, not current mainline |

## Where To Start

If you are new to the repository, read these in order:

1. [`docs/synthpop_architecture.md`](docs/synthpop_architecture.md)  
   Project architecture, phase logic, and module-to-question mapping.
2. [`docs/detroit_code_data_structure.md`](docs/detroit_code_data_structure.md)  
   Detroit-specific data layout and run structure.
3. [`docs/phase3_small_area_to_road_constrained_locations_method.md`](docs/phase3_small_area_to_road_constrained_locations_method.md)  
   Current explicit spatialization logic.
4. [`docs/phase3_work_destination_detroit_2026-03-29.md`](docs/phase3_work_destination_detroit_2026-03-29.md)  
   Work destination findings.
5. [`docs/phase3_mobility_anchor_validation_detroit_2026-03-29.md`](docs/phase3_mobility_anchor_validation_detroit_2026-03-29.md)  
   Home/work validation summary.

For the earlier distribution-level diffusion thread, see:

- [`docs/methods.md`](docs/methods.md)
- [`docs/findings.md`](docs/findings.md)

## Running Code

This repository is script-driven. Most experiments are launched from `tools/`, while reusable logic lives in `src/`.

Basic setup:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests
```

Detroit public data bootstrap:

```bash
python tools/detroit_fetch_public_data.py tiger --out_root "$RAW_ROOT/synthetic_city/data"
python tools/detroit_fetch_public_data.py acs --out_root "$RAW_ROOT/synthetic_city/data" --acs_year 2023
python tools/detroit_fetch_public_data.py pums --out_root "$RAW_ROOT/synthetic_city/data" --pums_year 2022
python tools/detroit_fetch_public_data.py osm --out_root "$RAW_ROOT/synthetic_city/data" --region michigan
```

## Data and Storage Policy

Raw data, licensed data, synced experiment products, and large outputs are intentionally kept **out of git**.

- `data/` should point to an external data root or symlink
- `outputs/` is for experiment products and is gitignored
- `figures/` only keeps current manuscript-ready products, not every intermediate rendering

This repo tracks **code, documentation, and lightweight review artifacts**, not the full data lake.

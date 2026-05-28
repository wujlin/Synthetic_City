# Synthetic City

Research code for generating a multi-attribute, geographically explicit synthetic population for the United States.

The current mainline follows the framework in the SIGSPATIAL 2026 manuscript: it learns region-specific joint distributions of five demographic and socioeconomic attributes, then converts those distributions into synthetic individuals with explicit home and workplace coordinates.

## Data Release

The release-format synthetic population dataset is hosted on OSF:

<https://osf.io/e7wp8/>

The OSF release is organized by state under `synthetic_population/data/csv_by_state/`. Each compressed CSV contains one state or Washington, D.C., and uses the USPS abbreviation in the filename, for example `synthetic_individuals_CA.csv.gz`.

Release columns:

```text
person_id, age, gender, education, employment, income,
home_lon, home_lat, work_lon, work_lat
```

The dataset is released under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License (CC BY-NC-ND 4.0). See the OSF `license.txt` for the exact license statement.

## Framework

The framework separates joint-distribution recovery from spatial assignment. This separation is the central design choice: the model first learns who lives in each region, then assigns those synthetic people to tracts and road-supported locations under census and commuting-flow constraints.

### Step 1: Target and Condition Construction

Step 1 constructs one training target and two condition vectors for each Public Use Microdata Area (PUMA).

- `p`: a PUMS-derived five-attribute joint distribution over age, gender, education, employment, and income.
- `c`: a PUMA-level census condition vector built from ACS Detailed Tables.
- `h`: a spatial representation vector built from POI and LODES-derived features.

The PUMS target provides observed individual-level co-occurrence patterns, while ACS and spatial features provide the region-specific conditions used during inference.

### Step 2: Hierarchical Diffusion

Step 2 learns the PUMA-level joint distribution with a two-stage diffusion process.

- Stage 1 predicts a coarse joint distribution using `c` and `h`.
- Stage 2 refines each coarse group into fine-grained attribute combinations.

The current paper configuration uses `K=960` coarse combinations in Stage 1 and refines them back to the full 3,000-cell joint distribution over the five attributes. This keeps the learning target compact while preserving the full attribute space used to sample individuals.

### Step 3: Spatial Synthetic Population Generation

Step 3 turns the predicted joint distribution into individual records and explicit locations.

- Individuals are sampled from the predicted PUMA-level joint distribution.
- Home tracts are assigned with tract-level ACS constraints, including age-gender consistency.
- Home coordinates are placed on residential road-supported candidate locations.
- Workers receive destination work tracts from LODES commuting flows.
- Workplace coordinates are placed on road-supported workplace candidate locations.

The resulting product is a national synthetic population with five attributes and explicit home/work coordinates.

## Data Sources

| Source | Role in the framework |
|---|---|
| ACS PUMS | Constructs the PUMA-level target joint distribution `p` |
| ACS Detailed Tables | Constructs census condition `c` and tract-level allocation constraints |
| POI data | Contributes to the spatial representation `h` |
| LODES | Contributes to `h` and provides work-destination flows |
| Road network | Provides candidate supports for home and workplace coordinates |

## Repository Layout

This is a research-first repository rather than a packaged software library. Reusable logic is kept under `src/`, while experiment entrypoints and release utilities are under `tools/`.

```text
src/         reusable modules for data loading, constraints, models, spatial allocation, and validation
tools/       data preparation, training, evaluation, release export, and OSF upload scripts
docs/        method notes, run notes, data contracts, and experiment summaries
tests/       lightweight smoke tests and regression checks
figures/     manuscript and presentation figure assets
outputs/     local run artifacts, gitignored by default
data/        optional local link to external data roots, gitignored by default
```

## Pipeline Components

The current national pipeline is script-driven. The repository contains utilities for:

- building aligned PUMS/ACS targets;
- training and evaluating the hierarchical diffusion model;
- generating state-level spatial synthetic populations;
- exporting 10-column release-format CSV files;
- preparing upload manifests and checksums for the OSF release.

Large raw data, licensed data, model checkpoints, and generated state-level products are intentionally kept out of git. The public repository tracks code, documentation, lightweight tests, and manuscript-supporting assets.

## Getting Started

Install the Python dependencies in an isolated environment, then run the smoke tests:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests
```

For the manuscript framework and data schema, start with:

- `docs/DATA_CONTRACT.md`
- `docs/synthpop_architecture.md`

Most full-scale runs require external data roots and workstation-scale compute. Local users should treat `data/` and `outputs/` as machine-specific paths rather than versioned repository content.

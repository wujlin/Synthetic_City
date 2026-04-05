# Phase 2 ACS-Anchor + Mobility Residual Pilot (2026-03-28)

## Question

For a product-facing pipeline, tract/CBG-level ACS statistics are available and should be used.
The key question is whether mobility should still enter the same `PUMA -> tract` allocation
step as an additional residual prior, or whether ACS already dominates the demographic part of
the problem.

## Setup

We compare four runs on the same `10 PUMA` slice:

- ACS-only constrained allocation:
  `/home/jinlin/projects/Synthetic_City/outputs/_phase2_p2t_10puma_recon_20260328T100817Z`
- ACS + raw mobility residual `w=0.05`:
  `/home/jinlin/projects/Synthetic_City/outputs/_phase2_p2t_10puma_acsplusrawres_w005_20260328T151540Z`
- ACS + raw mobility residual `w=0.10`:
  `/home/jinlin/projects/Synthetic_City/outputs/_phase2_p2t_10puma_acsplusrawres_w010_20260328T151540Z`
- ACS + raw mobility residual `w=0.25`:
  `/home/jinlin/projects/Synthetic_City/outputs/_phase2_p2t_10puma_acsplusrawres_w025_20260328T151540Z`

All runs use:

- hard constraints: `AGEP_bin`, `SEX`
- ACS soft prior: `SCHL_allpop`, `ESR_allpop`, `EARN_16p_bin`
- only the mobility residual weight changes

So this is not an “ACS vs mobility” comparison. It directly asks whether mobility provides
additional value once the relevant tract-level ACS marginals are already present.

## Main Results

Mean TVD over `SCHL / ESR / EARN`:

- ACS-only: `0.03265`
- ACS + raw residual `w=0.05`: `0.03289`
- ACS + raw residual `w=0.10`: `0.03319`
- ACS + raw residual `w=0.25`: `0.03425`

Variable-level view:

- ACS-only:
  - `SCHL 0.01772`
  - `ESR 0.03854`
  - `EARN 0.04169`
- ACS + raw residual `w=0.05`:
  - `SCHL 0.01802`
  - `ESR 0.03875`
  - `EARN 0.04189`
- ACS + raw residual `w=0.25`:
  - `SCHL 0.02015`
  - `ESR 0.03969`
  - `EARN 0.04291`

The pattern is monotone: once tract-level ACS already anchors the same variables, adding
mobility residual degrades those demographic metrics, and stronger residual weights degrade
them more.

## What This Means

This result clarifies an important product distinction.

At the `PUMA -> tract/CBG` stage, ACS and mobility are not contributing the same type of
information:

- ACS already provides the demographic target for these variables
- mobility adds a different, behavior-driven spatial bias

When the evaluation target is still the same tract-level ACS marginals, that extra mobility
bias does not help. It pushes the allocation away from the ACS target rather than resolving an
unidentified demographic ambiguity.

So the right interpretation is not “mobility is useless”. The right interpretation is:

- mobility is useful when ACS does **not** already specify the target variable at the same
  spatial scale
- mobility is useful for behavioral and downstream spatial realism
- but mobility should not overwrite tract-level ACS on the same variables inside the
  demographic disaggregation step

## Product Takeaway

For a product-oriented pipeline, the clean strategy is:

- `PUMA -> tract/CBG`: use ACS as the main constraint and demographic prior
- do **not** add mobility residual on top of tract-level ACS for the same demographic variables
- reserve mobility for downstream steps such as:
  - `tract/CBG -> building`
  - activity / POI assignment
  - behavioral consistency checks

In other words, once tract-level ACS already tells us what the tract should look like
demographically, mobility should stop acting as a competing demographic prior and move to the
next layer of the pipeline.

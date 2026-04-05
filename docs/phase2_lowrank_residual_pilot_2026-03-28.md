# Phase 2 Low-Rank Residual Pilot (2026-03-28)

## Question

The residual mobility experiments showed that tract assignment benefits from a weak mobility
energy. The next question is whether that signal is mostly a shared low-rank neighborhood
structure, or whether it still depends on tract-specific detail.

## Runs

- `nosoft`:
  `/home/jinlin/projects/Synthetic_City/outputs/_phase2_p2t_10puma_nosoft_20260328T113745Z`
- raw residual `w=0.5`:
  `/home/jinlin/projects/Synthetic_City/outputs/_phase2_p2t_10puma_residual_w05_20260328T114957Z`
- low-rank residual `rank=2`:
  `/home/jinlin/projects/Synthetic_City/outputs/_phase2_p2t_10puma_residual_lowrank_r2_20260328T122145Z`
- low-rank residual `rank=4`:
  `/home/jinlin/projects/Synthetic_City/outputs/_phase2_p2t_10puma_residual_lowrank_r4_20260328T122145Z`

The low-rank layer compresses the mobility-predicted residual targets by SVD on the
`(variable, category) x tract` centered log-share matrix and then reconstructs a rank-`r`
version before building `q(g|k)`.

## Main Results

Mean TVD over `SCHL / ESR / EARN`:

- `nosoft`: `0.06535`
- raw residual `w=0.5`: `0.06449`
- low-rank `r=2`: `0.06534`
- low-rank `r=4`: `0.06474`

Variable-level view:

- raw residual `w=0.5`:
  - `SCHL 0.07583`
  - `ESR 0.04954`
  - `EARN 0.06809`
- low-rank `r=4`:
  - `SCHL 0.07634`
  - `ESR 0.04965`
  - `EARN 0.06822`

The `r=4` model still improves over `nosoft`, but it does not beat the raw residual version.
`r=2` is too compressed and nearly collapses back to the baseline.

## Interpretation

This result is useful because it separates two kinds of mobility signal:

- a shared low-rank component does exist
- but the full gain still depends on higher-rank, tract-specific detail

So the spatial signal is not purely “a few neighborhood archetypes”. It appears to contain:

- a coarse, shared compatibility structure
- plus local tract detail that still matters after the hard constraints are applied

This is a stronger scientific statement than “rank 4 is better than rank 2”. It says the
phase-2 assignment problem is only partially low-rank.

## Takeaway

The best current model remains the raw residual prior, not the compressed one. The low-rank
pilot still provides one clear insight:

- mobility enters phase 2 through a partly shared compatibility structure,
- but that structure is not sufficient by itself,
- which means tract-level idiosyncrasy remains part of the identifiable signal.

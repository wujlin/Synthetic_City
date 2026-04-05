# Phase 2 Low-Rank + Smooth Residual Pilot (2026-03-28)

## Question

The sparse pilot showed that mobility residuals are not pure low-rank structure, but the
remaining correction also did not look truly very sparse. The next question is whether the
tract-specific detail behaves more like a **smooth local variation** over mobility-similar
neighborhoods.

## Runs

- `nosoft`:
  `/home/jinlin/projects/Synthetic_City/outputs/_phase2_p2t_10puma_nosoft_20260328T113745Z`
- raw residual `w=0.5`:
  `/home/jinlin/projects/Synthetic_City/outputs/_phase2_p2t_10puma_residual_w05_20260328T114957Z`
- low-rank + sparse best previous run:
  `/home/jinlin/projects/Synthetic_City/outputs/_phase2_p2t_10puma_residual_lrsparse_r4_w10_t005_20260328T134040Z`
- low-rank + smooth `rank=4, smooth_weight=0.5, knn=8`:
  `/home/jinlin/projects/Synthetic_City/outputs/_phase2_p2t_10puma_residual_lrsmooth_r4_w05_k8_20260328T135956Z`
- low-rank + smooth `rank=4, smooth_weight=1.0, knn=8`:
  `/home/jinlin/projects/Synthetic_City/outputs/_phase2_p2t_10puma_residual_lrsmooth_r4_w10_k8_20260328T135956Z`
- low-rank + smooth `rank=4, smooth_weight=1.0, knn=4`:
  `/home/jinlin/projects/Synthetic_City/outputs/_phase2_p2t_10puma_residual_lrsmooth_r4_w10_k4_20260328T140125Z`

The smooth model uses the same low-rank core as before, but replaces the sparse correction
with a k-nearest-neighbor smoothing step in the mobility feature space, restricted within each
PUMA.

## Main Results

Mean TVD over `SCHL / ESR / EARN`:

- `nosoft`: `0.06535`
- raw residual `w=0.5`: `0.06449`
- low-rank + sparse best: `0.06450`
- low-rank + smooth `w=0.5, k=8`: `0.06460`
- low-rank + smooth `w=1.0, k=8`: `0.06451`
- low-rank + smooth `w=1.0, k=4`: `0.06451`

Variable-level view for the best smooth run (`w=1.0, k=4`):

- `SCHL 0.07585`
- `ESR 0.04962`
- `EARN 0.06806`

So the smooth correction is useful. It clearly improves over low-rank only, and it nearly
matches both the sparse variant and the raw residual baseline. But it still does not surpass
the raw residual prior.

## What This Reveals

This pilot sharpens the interpretation of the phase-2 mobility signal:

- the residual detail is not only global structure
- it is not well described as a tiny set of sparse outliers either
- part of it does behave like smooth local variation over mobility-similar tracts

The smoothing step reduces the mean absolute residual magnitude from about `0.0879` to:

- `0.0636` for `k=8`
- `0.0715` for `k=4`

Yet even after this structured smoothing, the final allocation is still just behind the raw
residual prior. This means the useful tract-specific correction is **partly smooth, but not
fully reducible to a simple neighborhood smoother**.

That is a more informative conclusion than saying “`k=8` is slightly better than `k=4`”. The
important point is that the local correction is structured enough to survive smoothing, but not
structured enough to make the raw residual redundant.

## Takeaway

The current evidence now points to a consistent picture:

- a low-rank core exists
- local tract detail matters
- that local detail is neither purely sparse nor purely smooth

This is why the raw residual prior remains the strongest empirical model. It preserves all
three pieces at once:

- shared compatibility structure
- local tract correction
- residual heterogeneity that is not fully captured by either sparse selection or smooth
  neighborhood averaging

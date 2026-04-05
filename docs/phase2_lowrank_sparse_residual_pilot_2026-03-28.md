# Phase 2 Low-Rank + Sparse Residual Pilot (2026-03-28)

## Question

The previous low-rank pilot showed that mobility residuals contain a shared neighborhood
structure, but `rank=4` still underperformed the raw residual prior. The next question is
whether the remaining gain comes from a sparse tract-specific correction, or whether the
useful detail is still too distributed to be captured by a simple sparse residual model.

## Runs

- `nosoft`:
  `/home/jinlin/projects/Synthetic_City/outputs/_phase2_p2t_10puma_nosoft_20260328T113745Z`
- raw residual `w=0.5`:
  `/home/jinlin/projects/Synthetic_City/outputs/_phase2_p2t_10puma_residual_w05_20260328T114957Z`
- low-rank only `rank=4`:
  `/home/jinlin/projects/Synthetic_City/outputs/_phase2_p2t_10puma_residual_lowrank_r4_20260328T122145Z`
- low-rank + sparse `rank=4, sparse_weight=0.5, threshold=0.05`:
  `/home/jinlin/projects/Synthetic_City/outputs/_phase2_p2t_10puma_residual_lrsparse_r4_w05_t005_20260328T133841Z`
- low-rank + sparse `rank=4, sparse_weight=0.75, threshold=0.05`:
  `/home/jinlin/projects/Synthetic_City/outputs/_phase2_p2t_10puma_residual_lrsparse_r4_w075_t005_20260328T133841Z`
- low-rank + sparse `rank=4, sparse_weight=1.0, threshold=0.05`:
  `/home/jinlin/projects/Synthetic_City/outputs/_phase2_p2t_10puma_residual_lrsparse_r4_w10_t005_20260328T134040Z`

All three sparse runs keep the same `10 PUMA` slice and the same hard/prior settings as the
best raw residual baseline. The only change is how the mobility residual target is decomposed
before entering `q(g|k)`.

## Main Results

Mean TVD over `SCHL / ESR / EARN`:

- `nosoft`: `0.06535`
- raw residual `w=0.5`: `0.06449`
- low-rank only `rank=4`: `0.06474`
- low-rank + sparse `w=0.5`: `0.06455`
- low-rank + sparse `w=0.75`: `0.06451`
- low-rank + sparse `w=1.0`: `0.06450`

Variable-level view for the best sparse run (`w=1.0, threshold=0.05`):

- `SCHL 0.07589`
- `ESR 0.04954`
- `EARN 0.06806`

Relative to low-rank only, the sparse correction clearly helps. Relative to raw residual, it
almost closes the gap but does not surpass it.

## What This Reveals

This pilot supports a more precise interpretation of the mobility signal:

- the low-rank core is real and useful
- the remaining gain is not pure noise, because adding back thresholded tract detail improves
  on low-rank only
- but the correction is not truly very sparse in a strong sense

At `threshold=0.05`, the retained sparse fraction is still about `0.523`. That is an important
clue: the tract-specific detail is not concentrated in a handful of extreme cells. The useful
residual structure remains fairly distributed.

This explains why the sparse variant nearly matches raw residual but does not cleanly beat it:
the mobility correction is partly structured and partly local, yet the local component is still
too broad to be reduced to a tiny set of outliers.

## Takeaway

The best empirical model remains the raw residual prior. The low-rank + sparse pilot still
sharpens the scientific story:

- phase 2 is not governed by tract marginals alone
- mobility contributes through a shared compatibility core plus local tract correction
- but that local correction is only moderately sparse, not a negligible afterthought

So the main insight is not that a particular decomposition wins. It is that the residual
mobility signal is **structured but not compressible to a purely low-rank or ultra-sparse
form**, which is why the raw residual prior remains hard to beat.

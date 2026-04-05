# Phase 2 Mobility Residual Pilot (2026-03-28)

## Question

The earlier `type-conditioned mobility prior` experiment used mobility to predict tract-level
`SCHL / ESR / EARN` marginals directly and then blended those marginals into `q(g|k)`. That
design was easy to run, but it overwrote the same ACS-style targets used for evaluation and
therefore could not cleanly test whether mobility adds useful within-PUMA structure.

This pilot tests a different design:

- keep `AGEP_bin` and `SEX` as the only base priors
- use mobility only as a `within-PUMA residual energy`
- let mobility re-rank tracts for each type, instead of rewriting tract marginals

## Runs

- `nosoft` baseline:
  `/home/jinlin/projects/Synthetic_City/outputs/_phase2_p2t_10puma_nosoft_20260328T113745Z`
- residual `w=0.5`:
  `/home/jinlin/projects/Synthetic_City/outputs/_phase2_p2t_10puma_residual_w05_20260328T114957Z`
- residual `w=1.0`:
  `/home/jinlin/projects/Synthetic_City/outputs/_phase2_p2t_10puma_residual_w10_20260328T114957Z`

All runs use the same 10 PUMA slice:

- `2600100, 2600200, 2600300, 2600400, 2600500, 2600600, 2600700, 2600801, 2600802, 2600900`

## Main Findings

`residual w=0.5` improves all three soft target metrics relative to `nosoft`:

- `SCHL_allpop mean TVD`: `0.07720 -> 0.07583`
- `ESR_allpop mean TVD`: `0.05040 -> 0.04954`
- `EARN_16p_bin mean TVD`: `0.06844 -> 0.06809`

`residual w=1.0` is more mixed:

- `SCHL_allpop mean TVD`: `0.07720 -> 0.07711`
- `ESR_allpop mean TVD`: `0.05040 -> 0.05002`
- `EARN_16p_bin mean TVD`: `0.06844 -> 0.06948`

Other constraints remain stable:

- `AGEP_bin mean TVD` stays at `~0.00964`
- `SEX mean TVD` stays at `~0.00152`
- all `10/10` PUMAs converge

The entropy pattern is also consistent with a mild residual effect:

- `nosoft weighted_mean_entropy`: `5.7991`
- residual `w=0.5`: `5.7968`
- residual `w=1.0`: `5.7918`

This means the residual prior is not collapsing mass; it is only making tract choices slightly
more selective.

## Interpretation

The key result is not that mobility beats the ACS soft-prior baseline. It does not.

The key result is that:

- direct mobility-to-marginal prediction is too blunt and degrades performance
- residual mobility energy is much better behaved
- with a moderate weight, mobility provides a small but real gain over the `nosoft` baseline

This supports the phase-2 design claim:

- mobility should enter `q(g|k)` as a spatial heterogeneity signal
- it should not be used as a replacement for tract ACS marginals

## Next Step

The next useful upgrade is not a larger linear regressor. It is a better residual scorer, for example:

- variable-specific feature blocks
- low-rank `type x neighborhood-profile` compatibility
- a mobility-side evaluation metric that does not recycle the same ACS targets

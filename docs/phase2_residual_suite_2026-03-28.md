# Phase 2 Residual Mobility Suite (2026-03-28)

## Question

The earlier pilot established that mobility helps more as a `within-PUMA residual energy`
than as a direct replacement for tract marginals. This suite tests three tighter questions:

- what residual strength works best?
- which variable axis benefits most from mobility?
- does the gain disappear if the spatial correspondence is broken?

## Runs

Reference runs:

- `nosoft`: `/home/jinlin/projects/Synthetic_City/outputs/_phase2_p2t_10puma_nosoft_20260328T113745Z`
- residual `w=0.5`: `/home/jinlin/projects/Synthetic_City/outputs/_phase2_p2t_10puma_residual_w05_20260328T114957Z`
- residual `w=1.0`: `/home/jinlin/projects/Synthetic_City/outputs/_phase2_p2t_10puma_residual_w10_20260328T114957Z`

New suite runs:

- residual `w=0.25`: `/home/jinlin/projects/Synthetic_City/outputs/_phase2_p2t_10puma_residual_w025_20260328T121250Z`
- residual `w=0.75`: `/home/jinlin/projects/Synthetic_City/outputs/_phase2_p2t_10puma_residual_w075_20260328T121250Z`
- residual `SCHL only`: `/home/jinlin/projects/Synthetic_City/outputs/_phase2_p2t_10puma_residual_schl_w05_20260328T121250Z`
- residual `ESR only`: `/home/jinlin/projects/Synthetic_City/outputs/_phase2_p2t_10puma_residual_esr_w05_20260328T121250Z`
- residual `EARN only`: `/home/jinlin/projects/Synthetic_City/outputs/_phase2_p2t_10puma_residual_earn_w05_20260328T121406Z`
- residual `shuffle control`: `/home/jinlin/projects/Synthetic_City/outputs/_phase2_p2t_10puma_residual_shuffle_w05_20260328T121406Z`

## Weight Sweep

Mean TVD over `SCHL / ESR / EARN`:

- `nosoft`: `0.06535`
- residual `w=0.25`: `0.06460`
- residual `w=0.50`: `0.06449`
- residual `w=0.75`: `0.06487`
- residual `w=1.00`: `0.06554`

The pattern is clear:

- weak residual guidance helps
- the gain peaks around `0.25 - 0.50`
- stronger residual energy starts to over-steer tract assignment

This is the expected behavior of a valid weak prior.

## Variable Ablation

Relative to `nosoft`, the most interpretable improvements are:

- `SCHL only`: `SCHL 0.07720 -> 0.07594`
- `ESR only`: `ESR 0.05040 -> 0.04993`
- `EARN only`: `EARN 0.06844 -> 0.06835`

These single-axis runs improve their own target most strongly, while also producing small
spillover gains on the other variables. This means mobility is not just acting as a generic
tract-mass smoother; it is carrying type-relevant neighborhood information.

## Negative Control

The shuffled control breaks the tract-feature correspondence within each PUMA while preserving
the feature distribution itself.

Compared with `nosoft`:

- `SCHL`: `0.07720 -> 0.07859`
- `ESR`: `0.05040 -> 0.05114`
- `EARN`: `0.06844 -> 0.06940`

All three metrics get worse.

This is the strongest evidence in the suite. The residual gain is not a solver artifact or a
generic entropy effect. It depends on the correct spatial alignment between tract and mobility
profile.

## Takeaway

The suite supports one clean claim:

- mobility contributes usable information at the level of `within-PUMA tract ranking`
- it should enter as a weak residual energy
- it should not be used to overwrite tract-level soft marginals

This is a problem-structure result, not just an implementation choice.

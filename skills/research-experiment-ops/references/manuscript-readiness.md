# Manuscript Readiness

## Core rule

A result is ready for the manuscript only when the text, figure, table, and underlying run all point to the same experiment.

## Readiness checklist

Before a run enters the manuscript, verify:

- the run is artifact-complete
- the cited numbers match the JSON or summary files
- the plotted figure uses the same run the text describes
- captions do not retain legacy scope or outdated labels
- old baselines or old settings have been fully removed from figure scripts, not just from prose

## Common failure modes

- updated table, stale figure
- updated prose, stale caption
- new experiment in the text, old experiment still plotted
- citing rounded values that hide a key comparison

## Practical rule

If a claim depends on a specific run, make that run identifiable and frozen before polishing the sentence.

## Comparison logic

Keep subsection roles clear:

- verification: intrinsic quality, convergence, stability
- comparison: advantage over baselines, cross-validation wins, robustness of comparison

Do not mix these by default.

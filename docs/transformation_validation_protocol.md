# Transformation validation protocol

Purpose: convert the assumed transformation semantics into an empirically
measured ontology before it is used to build any benchmark.

## Stimuli

For each anchor segment `A` and each candidate `(transform, parameter)` setting
`T`:

- render `B = T(A)` with the parameter grids in `phase1/data/transforms.py`;
- present `A` then `B` in the same trial;
- ask: "Rate how similar B's musical motion is to A's, from 1 (very different)
  to 7 (same motion)."

The prompt targets **perceived musical motion similarity**, not general
similarity, emotion, tension or liking. Only one construct is measured in this
study.

## Raters and agreement

- Minimum 5 raters; each rater hears all trials (complete design) so Fleiss'
  kappa is well-defined in addition to Krippendorff's alpha.
- Report Krippendorff's alpha (ordinal) per `(transform, parameter)` setting.
- Settings with `alpha < 0.2`, too few raters, or too few trials are `uncertain`.

## Classification

With midpoint `m = 4.0` and margin `0.5`, using the bootstrap 95% CI of the mean
rating:

- `preserving` if `ci_low >= m + margin`
- `altering` if `ci_high <= m - margin`
- `ambiguous` otherwise
- `uncertain` if the minimum rater/trial/alpha thresholds are not met

A transform whose settings disagree is labeled `mixed`. `time_stretch` is
expected to be `mixed` (e.g., small changes preserving, large changes altering),
which is precisely why the status is estimated per parameter value rather than
asserted per transform; it must not be forced into invariance.

## Output

`score_study.py --kind rating` writes `ontology.json` with, for every
`(transform, parameter)` entry, the mean, bootstrap CI, agreement, the response
distribution over `{preserving, ambiguous, altering}`, and the final status,
plus `by_key` and `by_transform` maps. Downstream experiments read this file;
they do not read the provisional hypotheses in `transforms.py`.

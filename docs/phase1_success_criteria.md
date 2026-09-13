# Phase 1 pre-registered success criteria

A representation is reported as evidence for a musical-motion trajectory ontology
only if **all** criteria below hold. Failure modes are reported explicitly.

## Data and protocol

- Domain: solo piano.
- Segment candidates are fixed windows at `[0.5, 1, 2, 4, 8]` s.
- The transformation ontology is established by the human rating study before any
  positive/negative use, at the `(transform, parameter)` level. Settings with
  insufficient raters, insufficient trials, or Krippendorff's alpha (ordinal)
  below 0.2 are `uncertain` and excluded from the decision; a transform whose
  settings disagree is `mixed`.

## Criteria

1. **Collapse (health).**
   `collapse_report(all_frames).healthy == true`:
   mean per-dimension variance above the floor and effective rank above
   `0.1 * dim`, for the representation under test.

2. **Invariance.**
   For every `(transform, parameter)` setting validated as `preserving`,
   `mean d(z_A, z_T(A)) < mean d(z_A, z_B)` where `z_B` is a different gesture,
   with primary distance `norm` (length-normalized). AUC of the preserving vs
   other distance split is `> 0.5`.

3. **Specificity.**
   Settings validated as `altering` produce distances clearly larger than
   preserving settings (reported as part of the same separation analysis).

4. **Human alignment.**
   On the human triplet benchmark, model accuracy (using the same distance) is
   strictly greater than MERT, CLAP and handcrafted features.

5. **Prediction.**
   Multi-step prediction MSE of the learned trajectory is lower than
   persistence, linear extrapolation, ridge on frozen features, and constant
   mean.

## Decision rule

- If criterion 1 fails, no other claim is made (representation is degenerate).
- If criterion 2 fails for a `preserving` setting, that setting is reclassified
  `ambiguous`, not treated as a representation failure, and the protocol is
  re-run.
- If criterion 4 fails, the gesture ontology is reported as **unsupported at this
  stage**; the failure is documented and the next iteration targets temporal
  structure, invariance objective, or architecture.

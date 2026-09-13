# MAESTRO pilot results

Produced from 40 composer-stratified MAESTRO excerpts (train split), segmented
at `{1, 2, 4}` s (50% hop, at most four segments per excerpt, 22.05 kHz mono).

| File | Contents |
|---|---|
| `segments.jsonl` | 160 segment manifests (`segment_id`, `source_id`, start/end, sample rate). Absolute paths removed. |
| `variants.jsonl` | 1,600 transformed variants (`transform`, `params`, sample rate). Absolute paths removed. |
| `transformation_validation_trials.json` | The 400 rating trials (40 per `(transform, parameter)` setting). Audio paths are relative to the study folder. |
| `transformation_validation.html` | The dependency-free study instrument (embeds the trials). Audio is not included. |
| `encoder_comparison.json` | Invariance/collapse summary for `mel` and `handcrafted` on a 40-segment subset. |

## Interpretation

**Everything here is provisional.** The transformation statuses were *not*
human-validated, so `encoder_comparison.json` uses the prior hypotheses in
`phase1/data/transforms.py` and is engineering validation, not evidence about
musical motion. In particular, a high separation AUC under an assumed ontology
quantifies the inflation described by threat T1 in the manuscript, and must not
be read as a result.

To regenerate: `docs/reproduction.md`.

# Contributing

Thanks for your interest. This project is a research framework; the most
valuable contributions are ones that make its claims harder to inflate.

## Ground rules

1. **Never let model results define the ontology.** Transformation statuses come
   only from human ratings (`phase1/benchmark/transformation_validation.py`).
   Code that infers `preserving`/`altering` from a model belongs in
   `phase1/experiments/`, never in the ontology path.
2. **Keep anti-collapse mandatory.** Any new objective must retain the variance
   and covariance terms in `phase1/representation/losses.py`. Do not add an L2
   penalty on adjacent latents as a proxy for continuity; use prediction error.
3. **Report all three distances.** If you add a distance, add it alongside
   `raw` / `norm` / `dtw` in `phase1/experiments/common.py`, and keep `norm` as
   the primary unless a preregistered change says otherwise.
4. **Do not commit data or secrets.** Audio is gitignored; only manifests are
   tracked.

## Development

```powershell
pip install -r requirements.txt
pip install -e ".[ssl,dev]"
pytest
```

- Python >= 3.11 (tested on 3.14).
- Style: type hints, short docstrings, no inline comments unless necessary.
- Add a test in `tests/` for every behavior change; tests use synthetic audio and
  must run offline.

## Adding a transform

Add a `TransformSpec` in `phase1/data/transforms.py` with an explicit parameter
grid and a `*_candidate` hypothesis (a prior only). It becomes usable as an
invariance positive only after the human rating study validates it.

## Pull requests

Describe (a) what changed, (b) which preregistered criterion it affects, and
(c) how you verified it. Note that changes to `docs/phase1_success_criteria.md`
are considered a protocol amendment and must be called out explicitly.

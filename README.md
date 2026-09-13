# mmt — Musical Motion Trajectory

**Repository:** https://github.com/Fengrru/mmt

Can musical intent be represented as a continuous perceptual trajectory?

We study whether musical motion can be represented as a continuous latent
trajectory `z_{1:T} = f(x_{1:T})` that is:

1. perceptually meaningful
2. invariant to realization
3. predictable over time
4. locally editable

This repository is **Phase 1 only**. Its goal is *not* to prove that "gesture"
exists. Its goal is to build an experimental framework that cannot produce a
false positive because of definition holes.

## Status

- **Framework:** complete and tested (`pytest`: 30 passed on synthetic audio).
- **Pilot on real data:** 40 MAESTRO solo-piano excerpts -> 160 segments ->
  1,600 controlled variants; a 400-trial human rating study is generated.
- **Human ontology:** **not yet collected.** Until >= 5 raters complete the
  rating study, no invariance claim is made and every representation number is
  marked `unvalidated`.
- **Manuscript:** not public yet. The protocol, criteria, and pilot are
  documented in [`docs/`](docs) and this README.

## Load-bearing decisions

- **Domain:** solo piano only.
- **Object under test:** a continuous trajectory `z_{1:T} = f(x_{1:T})`; a
  *gesture* is a temporally localized trajectory segment.
- **Transformation semantics are empirical and parameterized, not assumed.**
  Every `(transform, parameter)` setting is classified by humans into
  `preserving / altering / ambiguous / uncertain` *before* it is used as a
  positive or negative, so a perceptual boundary `alpha*` can be estimated
  instead of forcing a binary label
  (`phase1/benchmark/transformation_validation.py`).
- **Distances for variable-length trajectories:** `raw`, time-normalized
  (`norm`, primary), and `dtw` (robustness only) — `phase1/experiments/common.py`.
- **Anti-collapse is mandatory** from day one (VICReg-style variance +
  covariance); continuity is expressed as multi-step predictability, never as an
  L2 penalty on adjacent latents.
- **Identification is the bottleneck:** transformation validation and
  representation evaluation are strictly separated, so model results can never
  feed back into the ontology.
- **A representation is only claimed to carry musical motion if** the
  pre-registered criteria in [`docs/phase1_success_criteria.md`](docs/phase1_success_criteria.md)
  all hold.

## Pipeline

```
audio -> segments -> parameterized transforms -> [human transformation validation]
      -> frozen SSL / acoustic / learned trajectory
      -> human triplet benchmark  vs  model geometry
      -> invariance / specificity / collapse / multi-step prediction
      -> (deferred) local editing
```

## Layout

```
phase1/
  data/           manifest, segmentation, transforms, audio I/O
  representation/ SSL wrapper, trajectory encoder, losses
  benchmark/      human study runner, parameterized ontology, metrics
  baselines/      acoustic, MERT, CLAP
  experiments/    invariance, collapse, prediction, ablations
  visualization/  trajectories, similarity (RDM / MDS)
configs/          YAML configs (default, maestro_pilot)
scripts/          CLI entry points
tests/            synthetic-audio unit tests
docs/             success criteria, protocol, reproduction
datasets/          subset manifests only (audio is not redistributed)
artifacts/         generated data (gitignored)
```

## Reproduction

Full steps are in [`docs/reproduction.md`](docs/reproduction.md). Summary:

```powershell
git clone https://github.com/Fengrru/mmt.git
cd mmt
pip install -r requirements.txt
pytest

# 1. select a small composer-stratified MAESTRO subset (metadata only)
python scripts/prepare_pilot.py --metadata path\to\maestro-v3.0.0.csv `
  --audio-root path\to\maestro-v3.0.0 --out datasets\maestro_subset --n 40 --split train

# 2. fetch the selected audio (resumable; mirror stores extracted .wav)
python scripts/download_pilot.py
powershell -ExecutionPolicy Bypass -File datasets\maestro_subset\download_pilot.ps1

# 3. build stimuli
python scripts/make_stimuli.py --metadata datasets\maestro_subset\metadata.csv `
  --audio-root (Get-Content datasets\maestro_subset\audio_root.txt) `
  --out artifacts\maestro_pilot\stimuli --durations 1.0 2.0 4.0 --max-segments-per-source 4

# 4. build the human study and collect responses
python scripts/build_human_study.py --stimuli artifacts\maestro_pilot\stimuli `
  --out artifacts\maestro_pilot\study --mode rating --max-segments 40

# 5. score -> parameterized ontology, then compare
python scripts\score_study.py --kind rating --responses artifacts\maestro_pilot\study\responses\*.json `
  --trials artifacts\maestro_pilot\study\transformation_validation_trials.json `
  --out artifacts\maestro_pilot\stimuli\ontology.json --min-raters 5 --min-trials 10
python scripts\run_experiment.py compare --stimuli artifacts\maestro_pilot\stimuli `
  --encoders handcrafted mel mert clap
```

## Success criteria (pre-registered)

A trajectory representation is reported as evidence only if **all** hold on
held-out data:

- `HumanAlignment(trajectory) > HumanAlignment(MERT), HumanAlignment(CLAP), HumanAlignment(handcrafted)`
- `mean d(z_A, z_T(A)) < mean d(z_A, z_B)` for every setting validated as preserving
- collapse diagnostics in the healthy band
- multi-step prediction better than persistence, linear extrapolation, ridge on
  frozen features, and constant mean

If any key result fails, the gesture ontology is reported as unsupported at this
stage, and the failure mode is documented rather than hidden.

## Citation

See [`CITATION.cff`](CITATION.cff). This is a protocol/framework release; a
bibliographic entry will be updated when the study is published.

## License

- Code: [MIT](LICENSE).
- MAESTRO audio: released under CC BY-NC-SA 4.0 by its authors; **not**
  redistributed here. Only subset manifests are versioned. Users are responsible
  for obtaining the audio under its own terms.

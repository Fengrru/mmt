# Reproduction guide

Tested on Windows (PowerShell 5.1) and Python 3.14.6. Commands assume the repo
root as the working directory.

## 0. Environment

```powershell
pip install -r requirements.txt
pip install -e ".[ssl,dev]"   # optional: MERT / CLAP baselines
pytest                        # expect: all tests pass
```

Disk: the pilot needs ~3.5 GB for the 40 MAESTRO excerpts plus a few hundred MB
for stimuli and the study.

## 1. Select a pilot subset (metadata only)

```powershell
python scripts/prepare_pilot.py `
  --metadata path\to\maestro-v3.0.0.csv `
  --audio-root path\to\maestro-v3.0.0 `
  --out datasets\maestro_subset --n 40 --split train
```

This writes `datasets/maestro_subset/metadata.csv` (composer-stratified) and
`audio_root.txt`. It does **not** download audio.

## 2. Fetch the selected audio

```powershell
python scripts/download_pilot.py
powershell -ExecutionPolicy Bypass -File datasets\maestro_subset\download_pilot.ps1
```

The default mirror stores extracted `.wav` files matching MAESTRO's official
metadata paths, so no format conversion is needed. Downloads are resumable
(`curl -C -`). Audio is written under the path in `audio_root.txt`.

## 3. Build stimuli

```powershell
python scripts/make_stimuli.py `
  --metadata datasets\maestro_subset\metadata.csv `
  --audio-root (Get-Content datasets\maestro_subset\audio_root.txt) `
  --out artifacts\maestro_pilot\stimuli `
  --durations 1.0 2.0 4.0 --max-segments-per-source 4
```

Expected: 160 segments and 1,600 variants. Outputs `segments.jsonl` and
`variants.jsonl` in the stimuli directory.

## 4. Build and distribute the human study

```powershell
python scripts/build_human_study.py `
  --stimuli artifacts\maestro_pilot\stimuli `
  --out artifacts\maestro_pilot\study --mode rating --max-segments 40 --seed 0
```

Open `artifacts/maestro_pilot/study/transformation_validation.html` in a browser
(works from `file://`). Each rater enters an ID, completes the trials, and clicks
**Download responses**. Put the resulting JSON files in
`artifacts/maestro_pilot/study/responses/`.

## 5. Score the ontology and compare

```powershell
python scripts\score_study.py --kind rating `
  --responses artifacts\maestro_pilot\study\responses\*.json `
  --trials artifacts\maestro_pilot\study\transformation_validation_trials.json `
  --out artifacts\maestro_pilot\stimuli\ontology.json `
  --min-raters 5 --min-trials 10

python scripts\run_experiment.py compare `
  --stimuli artifacts\maestro_pilot\stimuli `
  --ontology artifacts\maestro_pilot\stimuli\ontology.json `
  --encoders handcrafted mel mert clap
```

`ontology.json` records, per `(transform, parameter)`, the mean, bootstrap CI,
agreement, and response distribution, plus `by_key` and `by_transform` statuses.
Until `--min-raters` is met, settings are `uncertain` and no invariance claim is
made.

## Self-supervised baselines (MERT / CLAP)

These download model weights from the Hugging Face Hub. In regions where the Hub
is slow, point transformers at a mirror:

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
python scripts\run_experiment.py compare --stimuli artifacts\maestro_pilot\stimuli `
  --ontology artifacts\maestro_pilot\stimuli\ontology.json --encoders mert clap
```

## Prediction experiment

```powershell
python scripts\run_experiment.py prediction `
  --encoder mel --stimuli artifacts\maestro_pilot\stimuli --k 4 --epochs 30
```

Splits are by source segment (no window leakage). The learned trajectory is
compared against persistence, linear extrapolation, ridge on frozen features,
and constant mean.

## Notes

- All generated data lives under `artifacts/` and is gitignored.
- MAESTRO audio is subject to its own license (CC BY-NC-SA 4.0) and is not
  redistributed. Verify the mirror's terms before relying on it.

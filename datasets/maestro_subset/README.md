# MAESTRO pilot subset

Expected layout after `scripts/prepare_pilot.py`:

```
datasets/maestro_subset/
  metadata.csv      # subset rows (audio_filename, composer, ...)
  audio_root.txt    # absolute path to the audio root (unless --copy)
  audio/            # only present with --copy
```

## How to populate

Download MAESTRO separately (the script does not fetch data), then:

```powershell
python scripts/prepare_pilot.py `
  --metadata path\to\maestro-v3.0.0\maestro-v3.0.0.csv `
  --audio-root path\to\maestro-v3.0.0 `
  --n 40 --split train --copy
```

Then build stimuli:

```powershell
python scripts/make_stimuli.py `
  --metadata datasets\maestro_subset\metadata.csv `
  --audio-root (Get-Content datasets\maestro_subset\audio_root.txt) `
  --out artifacts\maestro_pilot\stimuli
```

If you already have a folder of solo-piano recordings, skip the metadata route
and pass `--audio-dir <folder>` to `make_stimuli.py` directly.

Audio is gitignored; only `metadata.csv` and this README are tracked.

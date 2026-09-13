import numpy as np
import pytest

from phase1.data.audio_io import save_audio

SR = 22050


def _note_seq(sr, freqs, note_dur=0.4, amp=0.5):
    chunks = []
    for f in freqs:
        n = int(sr * note_dur)
        t = np.arange(n) / sr
        env = np.hanning(n)
        chunks.append(amp * np.sin(2 * np.pi * f * t) * env)
    return np.concatenate(chunks).astype(np.float32)


def _chromatic(start_hz, n, step=1.5):
    return [start_hz * (2 ** (step * i / 12)) for i in range(n)]


@pytest.fixture(scope="session")
def synth_audio_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth_audio")
    save_audio(d / "up.wav", _note_seq(SR, _chromatic(220.0, 12, 1.5)), SR)
    save_audio(d / "down.wav", _note_seq(SR, _chromatic(440.0, 12, -1.5)), SR)
    save_audio(d / "arp.wav", _note_seq(SR, [220, 277, 330, 277] * 3), SR)
    return d


@pytest.fixture()
def stimuli_dir(tmp_path, synth_audio_dir):
    from phase1.data.segments import SegmenterConfig, build_segments
    from phase1.data.transforms import materialize_variants

    cfg = SegmenterConfig(durations=(0.5, 1.0, 2.0), max_segments_per_source=6)
    segments = build_segments(synth_audio_dir, tmp_path, config=cfg)
    variants = materialize_variants(segments, tmp_path)
    return tmp_path, segments, variants

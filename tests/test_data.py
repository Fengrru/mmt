import numpy as np

from phase1.data.audio_io import load_audio
from phase1.data.manifest import Segment, Variant, read_jsonl
from phase1.data.segments import SegmenterConfig, build_segments
from phase1.data.transforms import DEFAULT_TRANSFORMS, apply_transform, materialize_variants


def test_segments_are_built_and_written(synth_audio_dir, tmp_path):
    cfg = SegmenterConfig(durations=(0.5, 1.0), max_segments_per_source=4)
    segments = build_segments(synth_audio_dir, tmp_path, config=cfg)
    assert segments
    assert (tmp_path / "segments.jsonl").exists()
    loaded = read_jsonl(tmp_path / "segments.jsonl", cls=Segment)
    assert len(loaded) == len(segments)
    for seg in segments:
        y, sr = load_audio(seg.path, sr=seg.sample_rate)
        assert sr == cfg.target_sr
        assert abs(seg.duration - 0.5) < 0.01 or abs(seg.duration - 1.0) < 0.01
        assert len(y) > 0


def test_variants_materialized(synth_audio_dir, tmp_path):
    cfg = SegmenterConfig(durations=(1.0,), max_segments_per_source=2)
    segments = build_segments(synth_audio_dir, tmp_path, config=cfg)
    variants = materialize_variants(segments, tmp_path)
    n_expected = len(segments) * sum(len(t.grid) for t in DEFAULT_TRANSFORMS)
    assert len(variants) == n_expected
    assert (tmp_path / "variants.jsonl").exists()
    assert len(read_jsonl(tmp_path / "variants.jsonl", cls=Variant)) == n_expected


def test_gain_scales_amplitude():
    sr = 22050
    y = (0.1 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)).astype(np.float32)
    louder = apply_transform(y, sr, "gain", db=6.0)
    assert np.sqrt(np.mean(louder**2)) > np.sqrt(np.mean(y**2))


def test_time_stretch_changes_duration(synth_audio_dir):
    y, sr = load_audio(next(iter(sorted(synth_audio_dir.glob("*.wav")))), sr=22050)
    slow = apply_transform(y, sr, "time_stretch", rate=0.8)
    fast = apply_transform(y, sr, "time_stretch", rate=1.2)
    assert len(slow) > len(y) > len(fast)


def test_pitch_shift_preserves_length_approx(synth_audio_dir):
    y, sr = load_audio(next(iter(sorted(synth_audio_dir.glob("*.wav")))), sr=22050)
    shifted = apply_transform(y, sr, "pitch_shift", semitones=2.0)
    assert abs(len(shifted) - len(y)) < 0.05 * len(y)

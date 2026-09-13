from phase1.benchmark.human_triplets import build_triplets, score_responses
from phase1.benchmark.transformation_validation import (
    ALTERING,
    MIXED,
    PRESERVING,
    aggregate_ratings,
    build_rating_trials,
    classify_ontology,
    ontology_key,
    ontology_lookup,
)
from phase1.data.manifest import write_json

PRESERVING_TRANSFORMS = {"pitch_shift", "gain", "brightness"}


def _fake_rating_responses(trials, n_raters=5, answer_fn=None):
    responses = []
    for r in range(n_raters):
        for t in trials:
            if answer_fn is not None:
                answer = answer_fn(t)
            else:
                answer = 7 if t["transform"] in PRESERVING_TRANSFORMS else 2
            responses.append({"trial_id": t["trial_id"], "rater_id": f"R{r}", "answer": answer})
    return responses


def test_rating_study_yields_parameterized_ontology(stimuli_dir):
    tmp_path, segments, variants = stimuli_dir
    trials = build_rating_trials(segments, variants, tmp_path / "rating")
    assert trials
    responses = _fake_rating_responses(trials)
    aggregated = aggregate_ratings(responses, trials)
    ontology = classify_ontology(aggregated, min_raters=5, min_trials=1, min_alpha=-1.0)

    assert ontology_lookup(ontology, "pitch_shift", {"semitones": 2.0}) == PRESERVING
    assert ontology_lookup(ontology, "time_stretch", {"rate": 0.8}) == ALTERING

    key = ontology_key("time_stretch", {"rate": 0.8})
    entry = next(e for e in ontology["entries"] if e["key"] == key)
    assert set(entry["judgment"]) == {"preserving", "ambiguous", "altering"}
    assert abs(sum(entry["judgment"].values()) - 1.0) < 1e-9
    assert entry["n_raters"] == 5
    assert "krippendorff_alpha_ordinal" in entry["agreement"]


def test_ontology_finds_tempo_boundary(stimuli_dir):
    tmp_path, segments, variants = stimuli_dir
    trials = build_rating_trials(segments, variants, tmp_path / "rating2")

    def answer_fn(t):
        if t["transform"] == "time_stretch":
            return 6 if t["params"]["rate"] == 0.9 else 2
        return 7

    responses = _fake_rating_responses(trials, answer_fn=answer_fn)
    aggregated = aggregate_ratings(responses, trials)
    ontology = classify_ontology(aggregated, min_raters=5, min_trials=1, min_alpha=-1.0)
    assert ontology_lookup(ontology, "time_stretch", {"rate": 0.9}) == PRESERVING
    assert ontology_lookup(ontology, "time_stretch", {"rate": 1.25}) == ALTERING
    assert ontology["by_transform"]["time_stretch"] == MIXED


def test_triplet_study_and_scoring(stimuli_dir):
    tmp_path, segments, variants = stimuli_dir
    ontology = {t: (PRESERVING if t in PRESERVING_TRANSFORMS else ALTERING) for t in
                ["pitch_shift", "gain", "brightness", "time_stretch"]}
    trials = build_triplets(segments, variants, ontology, tmp_path / "triplets", n_triplets=20, seed=0)
    assert trials
    for t in trials:
        assert {o["id"] for o in t["options"]} == {t["positive_id"], t["negative_id"]}

    responses = [
        {"trial_id": t["trial_id"], "rater_id": "R0", "answer": t["positive_id"]} for t in trials
    ]
    trials_path = write_json(tmp_path / "triplets_trials.json", {"trials": trials})
    responses_path = write_json(tmp_path / "triplets_responses.json", {"responses": responses})
    result = score_responses(responses_path, trials_path)
    assert result["accuracy"] == 1.0


def test_invariance_and_collapse_run(stimuli_dir):
    from phase1.experiments.collapse import run as run_collapse
    from phase1.experiments.invariance import run as run_invariance

    tmp_path, _, _ = stimuli_dir
    inv = run_invariance("mel", tmp_path, ontology_path=None, limit=6, max_seconds=2.0)
    assert "separation" in inv and "per_key" in inv and inv["validated_ontology"] is False
    col = run_collapse("mel", tmp_path, limit=6, max_seconds=2.0)
    assert "all_frames" in col and "effective_rank" in col["all_frames"]


def test_prediction_runs(stimuli_dir):
    from phase1.experiments.prediction import run as run_prediction

    tmp_path, _, _ = stimuli_dir
    res = run_prediction("mel", tmp_path, limit=18, k=2, epochs=2, max_seconds=2.0)
    assert res["learned_mse"] >= 0.0
    assert "persistence" in res["baseline_mse"]

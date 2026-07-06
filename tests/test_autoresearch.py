import random

from soapscope.config import PipelineConfig
from soapscope.autoresearch.loop import (
    evaluate_config, random_neighbor, autoresearch, _get,
)


def _flat(cfg):
    from soapscope.autoresearch.loop import KNOBS
    return {p: _get(cfg, p) for (p, *_r) in KNOBS}


def test_evaluate_config_returns_metrics():
    m = evaluate_config(PipelineConfig(), n_frames=30)
    assert 0.0 <= m.score <= 1.0


def test_random_neighbor_changes_exactly_one_knob():
    base = PipelineConfig()
    cand, change = random_neighbor(base, random.Random(0))
    before, after = _flat(base), _flat(cand)
    diffs = [k for k in before if before[k] != after[k]]
    assert len(diffs) == 1
    assert change.startswith(diffs[0])


def test_autoresearch_never_regresses_below_baseline():
    base = PipelineConfig()
    baseline = evaluate_config(base, n_frames=40)
    best_cfg, history = autoresearch(base, rounds=6, n_frames=40, seed=1,
                                     log=lambda *_: None)
    best = evaluate_config(best_cfg, n_frames=40)
    # Hill-climb only keeps improvements, so it can't end worse than baseline.
    assert best.score >= baseline.score - 1e-6
    assert history[0].name == "baseline"

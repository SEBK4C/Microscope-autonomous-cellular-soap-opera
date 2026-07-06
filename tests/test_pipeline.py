from soapscope.config import PipelineConfig
from soapscope.pipeline import run_synthetic


def test_pipeline_runs_end_to_end():
    cfg = PipelineConfig()
    cfg.n_frames = 40
    res = run_synthetic(cfg, collect_frames=True)
    assert len(res.frames) == 40
    assert len(res.records) == 40
    assert len(res.stage_commands) == 40
    assert res.transcript                      # someone said something
    # Annotated frames are RGB and larger-or-equal to the world (banner+bar).
    assert res.frames[0].shape[2] == 3


def test_metrics_are_sane():
    cfg = PipelineConfig()
    cfg.n_frames = 60
    res = run_synthetic(cfg, collect_frames=False)
    m = res.metrics
    assert 0.0 <= m.score <= 1.0
    assert 0.0 <= m.caption_variety <= 1.0
    assert m.gt_recall is not None and m.gt_recall > 0.4
    assert m.n_tracks_total > 0
    assert m.fps > 0


def test_stage_follows_a_star():
    cfg = PipelineConfig()
    cfg.n_frames = 50
    res = run_synthetic(cfg, collect_frames=False)
    # At least some frames issued a real move (the camera actually panned).
    moves = [c for c in res.stage_commands if c["cmd"] == "move_abs"]
    assert len(moves) > 0
    stars = {r.star_id for r in res.records if r.star_id is not None}
    assert stars                                # a star was chosen

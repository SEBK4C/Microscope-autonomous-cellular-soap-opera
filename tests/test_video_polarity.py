import numpy as np

import soapscope.video.io as vio
from soapscope.config import PipelineConfig, SegmentConfig, WorldConfig
from soapscope.pipeline import run_synthetic
from soapscope.video.synthetic import SyntheticWorld
from soapscope.vision.segment import ClassicalSegmenter


# ---- polarity auto-detection ------------------------------------------------

def test_autodetect_polarity_darkfield():
    world = SyntheticWorld(WorldConfig(style="darkfield", seed=7))
    frame, _ = next(world.frames(1))
    seg = ClassicalSegmenter(SegmentConfig(polarity="auto"))
    seg.segment(frame)
    assert seg.last_polarity == "bright"


def test_autodetect_polarity_brightfield():
    world = SyntheticWorld(WorldConfig(style="brightfield", seed=7))
    frame, _ = next(world.frames(1))
    seg = ClassicalSegmenter(SegmentConfig(polarity="auto"))
    seg.segment(frame)
    assert seg.last_polarity == "dark"


def test_wrong_polarity_hurts_brightfield_recall():
    """Sanity: bright-field needs dark polarity; the wrong one collapses recall."""
    def recall(polarity, adaptive):
        cfg = PipelineConfig()
        cfg.n_frames = 45
        cfg.world.style = "brightfield"
        cfg.segment.polarity = polarity
        cfg.segment.adaptive = adaptive
        return run_synthetic(cfg, collect_frames=False).metrics.gt_recall

    assert recall("dark", True) > 0.7
    assert recall("auto", True) > 0.7
    assert recall("bright", False) < 0.4        # wrong polarity fails, as it should


def test_default_darkfield_unchanged():
    """The original default path (bright, non-adaptive) must not regress."""
    cfg = PipelineConfig()
    cfg.n_frames = 45
    m = run_synthetic(cfg, collect_frames=False).metrics
    assert m.gt_recall is not None and m.gt_recall > 0.85


# ---- load_video plumbing (decoder monkeypatched; no imageio needed) ---------

def test_load_video_stride_and_cap(monkeypatch):
    fake = [np.full((20, 30, 3), i, np.uint8) for i in range(10)]
    monkeypatch.setattr(vio, "_open_video", lambda path: iter(fake))
    out = list(vio.load_video("x.mp4", stride=2, max_frames=3))
    assert len(out) == 3
    assert out[0][0, 0, 0] == 0 and out[1][0, 0, 0] == 2      # kept 0,2,4


def test_load_video_downscale_and_channels(monkeypatch):
    frames = [np.zeros((40, 80), np.uint8),                    # 2D grayscale
              np.zeros((40, 80, 4), np.uint8)]                 # RGBA
    monkeypatch.setattr(vio, "_open_video", lambda path: iter(frames))
    out = list(vio.load_video("x.mp4", max_width=40))
    assert out[0].shape == (20, 40, 3)                         # grayscale->RGB, aspect kept
    assert out[1].shape == (20, 40, 3)                         # alpha dropped

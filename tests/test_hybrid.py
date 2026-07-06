"""Temporal+spatial hybrid segmentation tests (pure numpy)."""

import numpy as np

from soapscope.config import SegmentConfig
from soapscope.vision.segment import ClassicalSegmenter


def _disc(cy, cx, r=6, size=90):
    yy, xx = np.mgrid[0:size, 0:size]
    img = np.full((size, size), 20, np.uint8)
    img[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = 220
    return np.stack([img] * 3, axis=-1)


def _seg(**kw):
    base = dict(temporal=True, min_area=5, threshold=0.3)
    base.update(kw)
    return ClassicalSegmenter(SegmentConfig(**base))


def test_pure_motion_splits_moving_blob_into_crescents():
    seg = _seg(hybrid=False)
    seg.segment(_disc(45, 25))          # frame 0 seeds the background
    res = seg.segment(_disc(45, 55))    # blob jumped 30px -> old+new positions
    assert len(res.detections) == 2     # two separate motion crescents
    assert seg.last_polarity == "temporal"


def test_hybrid_keeps_moving_blob_whole():
    seg = _seg(hybrid=True, motion_gate=0.12, motion_dilate=6)
    seg.segment(_disc(45, 25))
    res = seg.segment(_disc(45, 55))
    assert len(res.detections) == 1     # one solid body, no crescents
    assert seg.last_polarity == "hybrid"


def test_hybrid_suppresses_static_scene():
    seg = _seg(hybrid=True)
    frame = _disc(45, 45)
    seg.segment(frame)
    res = seg.segment(frame)            # no motion -> gate empty -> nothing
    assert len(res.detections) == 0


def test_default_path_unaffected_by_refactor():
    # Non-temporal spatial path must still find a bright disc.
    seg = ClassicalSegmenter(SegmentConfig(threshold=0.3, min_area=5))
    res = seg.segment(_disc(45, 45))
    assert len(res.detections) == 1
    assert seg.last_polarity in ("bright", "dark")

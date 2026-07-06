import numpy as np

from soapscope.config import SegmentConfig
from soapscope.vision.segment import (
    ClassicalSegmenter, _binary_open, _median3,
)


def _disc(cy, cx, r=6, size=64, bright=True):
    """A single bright (or dark) disc on the opposite background — RGB uint8."""
    yy, xx = np.mgrid[0:size, 0:size]
    d = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    base = 20 if bright else 220
    val = 220 if bright else 40
    img = np.full((size, size), base, np.uint8)
    img[d] = val
    return np.stack([img] * 3, axis=-1)


def test_binary_open_removes_speck_keeps_blob():
    mask = np.zeros((20, 20), bool)
    mask[2, 2] = True               # lone speck
    mask[8:14, 8:14] = True         # solid 6x6 blob
    opened = _binary_open(mask, 1)
    assert not opened[2, 2]         # speck removed
    assert opened[10, 11]           # blob core survives


def test_median3_removes_salt():
    a = np.full((10, 10), 50.0, np.float32)
    a[5, 5] = 250.0
    out = _median3(a)
    assert abs(float(out[5, 5]) - 50.0) < 1e-3


def test_temporal_segmentation_needs_motion():
    seg = ClassicalSegmenter(SegmentConfig(temporal=True, min_area=5, threshold=0.3))
    r1 = seg.segment(_disc(30, 20))
    assert seg.last_polarity == "temporal"
    assert len(r1.detections) == 0        # first frame builds the background only
    r2 = seg.segment(_disc(30, 44))       # the disc moved -> motion foreground
    assert len(r2.detections) >= 1


def test_temporal_static_scene_is_quiet():
    seg = ClassicalSegmenter(SegmentConfig(temporal=True, min_area=5, threshold=0.3))
    frame = _disc(32, 32)
    seg.segment(frame)
    r = seg.segment(frame)                # identical frame -> (almost) no motion
    assert len(r.detections) == 0


def test_vectorized_segment_counts_blobs():
    seg = ClassicalSegmenter(SegmentConfig(threshold=0.3, min_area=5))
    img = np.full((64, 64), 20, np.uint8)
    for (cy, cx) in [(16, 16), (16, 48), (48, 32)]:
        yy, xx = np.mgrid[0:64, 0:64]
        img[(yy - cy) ** 2 + (xx - cx) ** 2 <= 36] = 220
    res = seg.segment(np.stack([img] * 3, -1))
    assert len(res.detections) == 3
    assert res.labels.max() == 3

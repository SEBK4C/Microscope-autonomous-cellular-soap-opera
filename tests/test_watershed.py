import numpy as np

from soapscope.config import SegmentConfig
from soapscope.vision.segment import (
    ClassicalSegmenter, _distance_transform, label_components, watershed_split,
)


def _discs(sep, r=9, size=70):
    yy, xx = np.mgrid[0:size, 0:size]
    c = size // 2
    return (((yy - c) ** 2 + (xx - (c - sep // 2)) ** 2 <= r * r)
            | ((yy - c) ** 2 + (xx - (c + sep // 2)) ** 2 <= r * r))


def _ellipse(size=70, ry=8, rx=20):
    yy, xx = np.mgrid[0:size, 0:size]
    c = size // 2
    return ((yy - c) / ry) ** 2 + ((xx - c) / rx) ** 2 <= 1.0


def _disc_frame(sep):
    m = _discs(sep)
    img = np.full(m.shape, 20, np.uint8)
    img[m] = 220
    return np.stack([img] * 3, axis=-1)


def test_watershed_splits_just_touching_round_cells():
    m = _discs(18)                       # two discs that just touch
    assert label_components(m)[1] == 1   # connected-components sees ONE blob
    assert watershed_split(m, 24, 0.55)[1] == 2   # watershed splits them


def test_watershed_keeps_heavy_overlap_merged():
    # Heavily overlapping blobs are genuinely one shape — must not be forced apart.
    assert watershed_split(_discs(12), 24, 0.55)[1] == 1


def test_watershed_does_not_oversplit_elongated_single():
    assert watershed_split(_ellipse(), 24, 0.55)[1] == 1


def test_distance_transform_peaks_at_centre():
    d = _distance_transform(_discs(0), 24)   # sep 0 => a single centred disc
    c = 35
    assert d[c, c] == d.max() and d.max() > 3


def test_segmenter_watershed_flag_splits_touching():
    frame = _disc_frame(18)
    # blur=0: pre-threshold blur softens the neck and would re-merge the cells.
    # adaptive=False: isolate the watershed on the raw threshold mask.
    plain = ClassicalSegmenter(SegmentConfig(threshold=0.3, min_area=5, blur=0,
                                             adaptive=False, watershed=False))
    split = ClassicalSegmenter(SegmentConfig(threshold=0.3, min_area=5, blur=0,
                                             adaptive=False, watershed=True,
                                             watershed_seed_frac=0.55))
    assert len(plain.segment(frame).detections) == 1
    assert len(split.segment(frame).detections) == 2

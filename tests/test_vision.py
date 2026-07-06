import numpy as np

from soapscope.config import SegmentConfig, TrackConfig, WorldConfig
from soapscope.vision.segment import ClassicalSegmenter, label_components
from soapscope.vision.track import Tracker
from soapscope.video.synthetic import SyntheticWorld


def test_label_components_counts_blobs():
    m = np.zeros((10, 10), bool)
    m[1:3, 1:3] = True          # blob A
    m[6:9, 6:9] = True          # blob B
    labels, n = label_components(m)
    assert n == 2
    assert labels[m].min() >= 1
    assert set(np.unique(labels[m])) == {1, 2}


def test_label_components_diagonal_is_one_blob():
    m = np.zeros((6, 6), bool)
    m[1, 1] = m[2, 2] = m[3, 3] = True   # 8-connected chain
    _labels, n = label_components(m)
    assert n == 1


def test_segmenter_finds_microbes():
    world = SyntheticWorld(WorldConfig(seed=1))
    frame, gt = next(world.frames(1))
    seg = ClassicalSegmenter(SegmentConfig()).segment(frame)
    # Should detect a decent fraction of the ground-truth microbes.
    assert len(seg.detections) >= max(1, int(0.5 * len(gt.items)))
    for d in seg.detections:
        assert d.area >= SegmentConfig().min_area


def test_tracker_keeps_stable_ids_on_smooth_motion():
    from soapscope.vision.segment import Detection
    tk = Tracker(TrackConfig(min_hits=1))
    ids = []
    for step in range(6):
        det = Detection(label=1, centroid=(10.0 + step * 2, 20.0 + step * 2),
                        bbox=(0, 0, 5, 5), area=30)
        tracks = tk.update([det])
        ids.append(tracks[0].id)
    assert len(set(ids)) == 1       # one identity throughout

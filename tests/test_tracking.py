import numpy as np

from soapscope.config import TrackConfig
from soapscope.metrics import mota_and_idsw
from soapscope.vision.assign import linear_sum_assignment
from soapscope.vision.segment import Detection
from soapscope.vision.track import Tracker


def _det(cy, cx, label=1, area=30):
    return Detection(label=label, centroid=(cy, cx),
                     bbox=(int(cy - 3), int(cx - 3), int(cy + 3), int(cx + 3)), area=area)


class _GT:
    def __init__(self, items):
        self.items = items


# ---- Hungarian solver -------------------------------------------------------

def test_linear_sum_assignment_known_case():
    cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]], float)
    r, c = linear_sum_assignment(cost)
    assert cost[r, c].sum() == 5.0            # optimal
    assert sorted(r) == [0, 1, 2] and sorted(c.tolist()) == sorted(set(c.tolist()))


def test_linear_sum_assignment_rectangular():
    cost = np.array([[1, 9], [9, 1], [5, 5]], float)   # 3 rows, 2 cols
    r, c = linear_sum_assignment(cost)
    assert len(r) == 2 and len(set(c.tolist())) == 2   # every column used once


# ---- tracking behaviour -----------------------------------------------------

def test_kalman_tracker_keeps_stable_id():
    tk = Tracker(TrackConfig(min_hits=1, assignment="hungarian", use_kalman=True))
    ids = []
    for step in range(8):
        tracks = tk.update([_det(10 + 2 * step, 20 + 2 * step)])
        ids.append(tracks[0].id)
    assert len(set(ids)) == 1


def test_tracker_preserves_identity_through_crossing():
    tk = Tracker(TrackConfig(min_hits=1, assignment="hungarian", use_kalman=True,
                             max_dist=60))
    left_id = None
    for t in range(11):
        xa, xb = 10 + 8 * t, 90 - 8 * t       # A moves right, B moves left, they cross
        tracks = tk.update([_det(25, xa), _det(35, xb)])
        if t == 0:
            left_id = min(tracks, key=lambda tr: tr.cx).id
    left = [tr for tr in tk.confirmed_tracks() if tr.id == left_id][0]
    assert left.cx > 60          # the left-starting track ended on the right: no ID swap


# ---- MOTA metric ------------------------------------------------------------

def test_mota_perfect_tracking():
    gts = [_GT({7: (10, 10 + 5 * t, 6, "x")}) for t in range(4)]
    ftracks = [[(1, 10, 10 + 5 * t)] for t in range(4)]
    mota, idsw, fp, fn = mota_and_idsw(gts, ftracks, match_radius=15)
    assert (idsw, fp, fn) == (0, 0, 0) and mota == 1.0


def test_mota_counts_id_switch():
    gts = [_GT({7: (10, 10 + 5 * t, 6, "x")}) for t in range(3)]
    ftracks = [[(1, 10, 10)], [(1, 10, 15)], [(2, 10, 20)]]   # id 1 -> 2 at frame 2
    _mota, idsw, _fp, _fn = mota_and_idsw(gts, ftracks, match_radius=15)
    assert idsw == 1

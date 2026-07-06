"""Multi-object tracking: stitch per-frame detections into persistent identities.

Two matching strategies (``TrackConfig.assignment``): greedy nearest-neighbour
(default, dependency-free) or optimal Hungarian assignment (fewer ID switches on
crossings). Motion is coasted either by an EMA velocity (default) or a
constant-velocity **Kalman filter** (``use_kalman``). All behind one interface;
a SAM2 video-propagation tracker could replace it later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import TrackConfig
from .assign import linear_sum_assignment
from .segment import Detection


class KalmanFilter:
    """Constant-velocity 2D Kalman filter with state [y, x, vy, vx]."""

    def __init__(self, y: float, x: float, q: float = 1.0, r: float = 4.0):
        self.x = np.array([y, x, 0.0, 0.0], dtype=float)
        self.P = np.diag([r, r, 100.0, 100.0]).astype(float)
        self.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1],
                           [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.Q = np.diag([q * 0.25, q * 0.25, q, q]).astype(float)
        self.R = np.diag([r, r]).astype(float)

    def predict(self) -> None:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, y: float, x: float) -> None:
        z = np.array([y, x], dtype=float)
        innov = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innov
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def pos(self) -> tuple:
        return (float(self.x[0]), float(self.x[1]))

    def vel(self) -> tuple:
        return (float(self.x[2]), float(self.x[3]))


@dataclass
class Track:
    id: int
    cy: float
    cx: float
    vy: float = 0.0
    vx: float = 0.0
    area: float = 0.0
    bbox: tuple = (0, 0, 0, 0)
    age: int = 0                     # frames since creation
    hits: int = 1                    # detections matched
    missed: int = 0                  # consecutive frames without a detection
    label: Optional[int] = None      # detection label in the current frame
    history: List[tuple] = field(default_factory=list)
    kf: Optional[KalmanFilter] = None

    @property
    def centroid(self) -> tuple:
        return (self.cy, self.cx)

    @property
    def speed(self) -> float:
        return float(np.hypot(self.vy, self.vx))

    def confirmed(self, min_hits: int) -> bool:
        return self.hits >= min_hits and self.missed == 0


class Tracker:
    def __init__(self, cfg: TrackConfig):
        self.cfg = cfg
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1
        self._confirmed: set = set()
        # Populated each update() for the drama layer.
        self.born: List[int] = []      # raw new tracks (may be noise)
        self.entered: List[int] = []   # tracks that just became CONFIRMED
        self.exited: List[int] = []    # confirmed tracks that were just lost

    # ---------------------------------------------------------- prediction
    def _predict_track(self, t: Track) -> tuple:
        """Advance a track to this frame; return its predicted (cy, cx)."""
        if t.kf is not None:
            t.kf.predict()
            return t.kf.pos()
        return (t.cy + t.vy, t.cx + t.vx)   # EMA constant-velocity coast

    # ---------------------------------------------------------- matching
    def _greedy_match(self, track_ids, pred, dets) -> List[Tuple[int, int]]:
        cand = []
        for tid in track_ids:
            py, px = pred[tid]
            for j, d in enumerate(dets):
                dist = float(np.hypot(d.centroid[0] - py, d.centroid[1] - px))
                if dist <= self.cfg.max_dist:
                    cand.append((dist, tid, j))
        cand.sort(key=lambda c: c[0])
        mt, md, out = set(), set(), []
        for dist, tid, j in cand:
            if tid in mt or j in md:
                continue
            mt.add(tid)
            md.add(j)
            out.append((tid, j))
        return out

    def _hungarian_match(self, track_ids, pred, dets) -> List[Tuple[int, int]]:
        n, m = len(track_ids), len(dets)
        big = self.cfg.max_dist * 1000.0
        cost = np.full((n, m), big)
        for i, tid in enumerate(track_ids):
            py, px = pred[tid]
            for j, d in enumerate(dets):
                dist = float(np.hypot(d.centroid[0] - py, d.centroid[1] - px))
                if dist <= self.cfg.max_dist:
                    cost[i, j] = dist
        rows, cols = linear_sum_assignment(cost)
        return [(track_ids[i], j) for i, j in zip(rows, cols) if cost[i, j] < big]

    # ---------------------------------------------------------- update
    def update(self, dets: List[Detection]) -> List[Track]:
        cfg = self.cfg
        self.born, self.entered, self.exited = [], [], []
        track_ids = list(self.tracks.keys())

        pred = {tid: self._predict_track(self.tracks[tid]) for tid in track_ids}
        if cfg.assignment == "hungarian" and track_ids and dets:
            matches = self._hungarian_match(track_ids, pred, dets)
        else:
            matches = self._greedy_match(track_ids, pred, dets)

        matched_tracks, matched_dets = set(), set()
        for tid, j in matches:
            matched_tracks.add(tid)
            matched_dets.add(j)
            self._update_track(self.tracks[tid], dets[j])

        # Unmatched existing tracks: coast (already predicted), age out if lost.
        for tid in track_ids:
            if tid in matched_tracks:
                continue
            t = self.tracks[tid]
            t.missed += 1
            t.age += 1
            t.label = None
            t.cy, t.cx = pred[tid]
            t.history.append((t.cy, t.cx))
            if len(t.history) > 64:
                t.history.pop(0)
            if t.missed > cfg.max_missed:
                if tid in self._confirmed:
                    self.exited.append(tid)
                    self._confirmed.discard(tid)
                del self.tracks[tid]

        # Unmatched detections: spawn new tracks.
        for j, d in enumerate(dets):
            if j in matched_dets:
                continue
            self._new_track(d)

        # Promote tracks that have gathered enough hits to "confirmed".
        for t in self.tracks.values():
            if t.confirmed(cfg.min_hits) and t.id not in self._confirmed:
                self._confirmed.add(t.id)
                self.entered.append(t.id)

        return list(self.tracks.values())

    def _new_track(self, d: Detection) -> None:
        tid = self._next_id
        self._next_id += 1
        kf = None
        if self.cfg.use_kalman:
            kf = KalmanFilter(d.centroid[0], d.centroid[1],
                              q=self.cfg.kalman_q, r=self.cfg.kalman_r)
        t = Track(id=tid, cy=d.centroid[0], cx=d.centroid[1],
                  area=float(d.area), bbox=d.bbox, label=d.label,
                  history=[d.centroid], kf=kf)
        self.tracks[tid] = t
        self.born.append(tid)

    def _update_track(self, t: Track, d: Detection) -> None:
        ny, nx = d.centroid
        if t.kf is not None:
            t.kf.update(ny, nx)
            t.cy, t.cx = t.kf.pos()
            t.vy, t.vx = t.kf.vel()
        else:
            # Velocity via EMA of frame-to-frame displacement (the default path).
            a = 0.5
            t.vy = (1 - a) * t.vy + a * (ny - t.cy)
            t.vx = (1 - a) * t.vx + a * (nx - t.cx)
            t.cy, t.cx = ny, nx
        t.area = 0.6 * t.area + 0.4 * float(d.area)
        t.bbox = d.bbox
        t.label = d.label
        t.hits += 1
        t.age += 1
        t.missed = 0
        t.history.append((t.cy, t.cx))
        if len(t.history) > 64:
            t.history.pop(0)

    def confirmed_tracks(self) -> List[Track]:
        return [t for t in self.tracks.values() if t.confirmed(self.cfg.min_hits)]

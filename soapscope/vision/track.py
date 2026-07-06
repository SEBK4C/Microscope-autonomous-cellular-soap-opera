"""Multi-object tracking: stitch per-frame detections into persistent identities.

Greedy nearest-neighbour matching with a distance gate and simple constant-
velocity coasting. Dependency-free and good enough for the prototype; a
Kalman/Hungarian or SAM3 video-propagation tracker can replace it behind the
same interface later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..config import TrackConfig
from .segment import Detection


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

    def _predict(self, t: Track) -> tuple:
        return (t.cy + t.vy, t.cx + t.vx)

    def update(self, dets: List[Detection]) -> List[Track]:
        cfg = self.cfg
        self.born, self.entered, self.exited = [], [], []
        track_ids = list(self.tracks.keys())

        # Build candidate (distance, track_id, det_index) triples within the gate.
        candidates = []
        for tid in track_ids:
            py, px = self._predict(self.tracks[tid])
            for j, d in enumerate(dets):
                dy = d.centroid[0] - py
                dx = d.centroid[1] - px
                dist = float(np.hypot(dy, dx))
                if dist <= cfg.max_dist:
                    candidates.append((dist, tid, j))
        candidates.sort(key=lambda c: c[0])

        matched_tracks = set()
        matched_dets = set()
        for dist, tid, j in candidates:
            if tid in matched_tracks or j in matched_dets:
                continue
            matched_tracks.add(tid)
            matched_dets.add(j)
            self._update_track(self.tracks[tid], dets[j])

        # Unmatched existing tracks: coast forward, age out if lost too long.
        for tid in track_ids:
            if tid in matched_tracks:
                continue
            t = self.tracks[tid]
            t.missed += 1
            t.age += 1
            t.label = None
            t.cy, t.cx = self._predict(t)
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
            tid = self._next_id
            self._next_id += 1
            t = Track(id=tid, cy=d.centroid[0], cx=d.centroid[1],
                      area=float(d.area), bbox=d.bbox, label=d.label,
                      history=[d.centroid])
            self.tracks[tid] = t
            self.born.append(tid)

        # Promote tracks that have gathered enough hits to "confirmed" and
        # announce them once (the drama layer treats this as an ENTER).
        for t in self.tracks.values():
            if t.confirmed(cfg.min_hits) and t.id not in self._confirmed:
                self._confirmed.add(t.id)
                self.entered.append(t.id)

        return list(self.tracks.values())

    def _update_track(self, t: Track, d: Detection) -> None:
        ny, nx = d.centroid
        # Velocity via EMA of frame-to-frame displacement.
        mvy, mvx = ny - t.cy, nx - t.cx
        a = 0.5
        t.vy = (1 - a) * t.vy + a * mvy
        t.vx = (1 - a) * t.vx + a * mvx
        t.cy, t.cx = ny, nx
        t.area = 0.6 * t.area + 0.4 * float(d.area)
        t.bbox = d.bbox
        t.label = d.label
        t.hits += 1
        t.age += 1
        t.missed = 0
        t.history.append((ny, nx))
        if len(t.history) > 64:
            t.history.pop(0)

    def confirmed_tracks(self) -> List[Track]:
        return [t for t in self.tracks.values() if t.confirmed(self.cfg.min_hits)]

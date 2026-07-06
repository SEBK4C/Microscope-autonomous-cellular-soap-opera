"""Behaviour analysis: turn tracks into dramatic "beats".

A *beat* is a narratable event — an arrival, an exit, a chase, a collision, a
mitotic "birth", a stationary sulk. Each beat carries a drama ``score`` used
both by the captioner (what to talk about) and the stage director (whom to
follow). This is the bridge between cold trajectories and hot gossip.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .track import Track


@dataclass
class Beat:
    kind: str                        # ENTER|EXIT|DIVIDE|CHASE|FLEE|ENCOUNTER|SPEED_BURST|LINGER|WANDER
    subjects: List[int]              # track ids involved (protagonist first)
    score: float                     # how dramatic, higher = juicier
    data: dict = field(default_factory=dict)


@dataclass
class FrameFeatures:
    frame_idx: int
    beats: List[Beat]
    per_track: Dict[int, dict]       # tid -> {speed, heading, area, nearest, nearest_dist}

    def top(self, star_id: Optional[int] = None) -> Optional[Beat]:
        """The beat to narrate. With ``star_id`` (the microbe the camera is
        following), prefer a beat that microbe *leads* — then any beat it's in —
        so the narration is about who we're watching. Falls back to the juiciest
        beat overall when the star is idle or unset (backwards-compatible)."""
        if not self.beats:
            return None
        if star_id is not None:
            lead = [b for b in self.beats if b.subjects and b.subjects[0] == star_id]
            if lead:
                return max(lead, key=lambda b: b.score)
            involved = [b for b in self.beats if star_id in b.subjects]
            if involved:
                return max(involved, key=lambda b: b.score)
        return max(self.beats, key=lambda b: b.score)


def _radius(area: float) -> float:
    return max(3.0, math.sqrt(max(area, 1.0) / math.pi))


class FeatureAnalyzer:
    """Stateful per-frame analyser (holds the memory a soap opera needs)."""

    def __init__(self, height: int, width: int):
        self.H = height
        self.W = width
        self.prev_pos: Dict[int, Tuple[float, float]] = {}
        self.prev_pair: Dict[Tuple[int, int], float] = {}
        self.speed_ema: Dict[int, float] = {}
        self.still: Dict[int, int] = {}

    def _near_edge(self, cy: float, cx: float, margin: float = 28.0) -> bool:
        return (cx < margin or cx > self.W - margin
                or cy < margin or cy > self.H - margin)

    def step(self, frame_idx: int, tracks: List[Track],
             entered: List[int], exited: List[int]) -> FrameFeatures:
        beats: List[Beat] = []
        per_track: Dict[int, dict] = {}
        by_id = {t.id: t for t in tracks}

        # ---- per-track scalar features + linger/burst bookkeeping ----------
        for t in tracks:
            spd = t.speed
            ema = self.speed_ema.get(t.id, spd)
            ema = 0.7 * ema + 0.3 * spd
            self.speed_ema[t.id] = ema
            heading = math.degrees(math.atan2(t.vy, t.vx)) if spd > 1e-3 else 0.0
            per_track[t.id] = {
                "speed": spd, "ema": ema, "heading": heading,
                "area": t.area, "radius": _radius(t.area),
                "nearest": None, "nearest_dist": float("inf"),
            }
            self.still[t.id] = self.still.get(t.id, 0) + 1 if spd < 0.8 else 0

        # ---- pairwise interactions (approach / flee / encounter) -----------
        ids = list(by_id.keys())
        for a_idx in range(len(ids)):
            for b_idx in range(a_idx + 1, len(ids)):
                ia, ib = ids[a_idx], ids[b_idx]
                ta, tb = by_id[ia], by_id[ib]
                d = math.hypot(ta.cy - tb.cy, ta.cx - tb.cx)
                if d < per_track[ia]["nearest_dist"]:
                    per_track[ia].update(nearest=ib, nearest_dist=d)
                if d < per_track[ib]["nearest_dist"]:
                    per_track[ib].update(nearest=ia, nearest_dist=d)
                key = (ia, ib)
                prev = self.prev_pair.get(key)
                self.prev_pair[key] = d
                ra = per_track[ia]["radius"]
                rb = per_track[ib]["radius"]
                touch = (ra + rb) * 1.5
                if d <= touch:
                    beats.append(Beat("ENCOUNTER", [ia, ib],
                                      score=6.0 + max(0.0, (touch - d)) * 0.1,
                                      data={"dist": d}))
                elif prev is not None and d < 160:
                    closing = prev - d
                    fast = max(ta.speed, tb.speed)
                    if closing > 1.4 and fast > 1.2:
                        # Whoever is faster is doing the chasing.
                        chaser, chased = (ia, ib) if ta.speed >= tb.speed else (ib, ia)
                        beats.append(Beat("CHASE", [chaser, chased],
                                          score=5.0 + closing * 0.4 + fast * 0.3,
                                          data={"dist": d, "closing": closing}))
                    elif closing < -1.8 and fast > 1.2:
                        flee_from = ia if ta.speed >= tb.speed else ib
                        other = ib if flee_from == ia else ia
                        beats.append(Beat("FLEE", [flee_from, other],
                                          score=4.5 + (-closing) * 0.3,
                                          data={"dist": d}))

        # ---- arrivals: mitosis vs drifting in ------------------------------
        for tid in entered:
            t = by_id.get(tid)
            if t is None:
                continue
            # Nearest older confirmed neighbour at birth?
            parent, best = None, 1e9
            for o in tracks:
                if o.id == tid or o.age < 4:
                    continue
                dd = math.hypot(o.cy - t.cy, o.cx - t.cx)
                if dd < best:
                    parent, best = o.id, dd
            if parent is not None and best < _radius(t.area) * 4.5:
                beats.append(Beat("DIVIDE", [parent, tid], score=8.5,
                                  data={"dist": best}))
            elif self._near_edge(t.cy, t.cx):
                beats.append(Beat("ENTER", [tid], score=4.0, data={"edge": True}))
            else:
                beats.append(Beat("ENTER", [tid], score=3.2, data={"edge": False}))

        # ---- exits ---------------------------------------------------------
        for tid in exited:
            py, px = self.prev_pos.get(tid, (self.H / 2, self.W / 2))
            edge = self._near_edge(py, px, margin=40.0)
            beats.append(Beat("EXIT", [tid], score=4.2 if edge else 3.0,
                              data={"pos": (py, px), "edge": edge}))

        # ---- solo drama: speed bursts, lingering ---------------------------
        for t in tracks:
            pt = per_track[t.id]
            if t.speed > 3.2 and t.speed > pt["ema"] * 1.8:
                beats.append(Beat("SPEED_BURST", [t.id],
                                  score=3.5 + t.speed * 0.2, data={"speed": t.speed}))
            elif self.still.get(t.id, 0) >= 10:
                beats.append(Beat("LINGER", [t.id], score=2.2,
                                  data={"frames": self.still[t.id]}))

        if not beats and tracks:
            # Always have something to say — narrate the busiest drifter.
            star = max(tracks, key=lambda t: t.speed)
            beats.append(Beat("WANDER", [star.id], score=1.0,
                              data={"speed": star.speed}))

        # Remember positions for next frame (exits, deltas).
        self.prev_pos = {t.id: (t.cy, t.cx) for t in tracks}
        return FrameFeatures(frame_idx=frame_idx, beats=beats, per_track=per_track)

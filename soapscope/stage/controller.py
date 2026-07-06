"""Stage director: pick today's star and keep the CNC stage centred on it.

The "director" scores every track by how much drama it's currently generating
(from the beats), applies hysteresis so the camera doesn't twitch between
microbes, then emits a rate-limited move command to recentre the chosen star.
This is the closed loop a real microcontroller would run against live tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from ..config import StageConfig
from ..vision.features import FrameFeatures
from ..vision.track import Track
from .api import CNCStage, SimulatedStage, StageCommand, StageState


@dataclass
class StageStep:
    star_id: Optional[int]
    target: Optional[tuple]          # (cy, cx) the star, in frame px
    command: StageCommand
    state: StageState
    scores: Dict[int, float]


class StageController:
    def __init__(self, cfg: StageConfig, height: int, width: int,
                 stage: Optional[CNCStage] = None):
        self.cfg = cfg
        self.H, self.W = height, width
        self.cx0, self.cy0 = width / 2.0, height / 2.0
        self.stage = stage or SimulatedStage(x=0.0, y=0.0, max_step=cfg.max_step)
        self.drama_ema: Dict[int, float] = {}
        self.star_id: Optional[int] = None

    def _drama_scores(self, feats: FrameFeatures, live: set) -> Dict[int, float]:
        raw: Dict[int, float] = {tid: 0.0 for tid in live}
        for b in feats.beats:
            for i, tid in enumerate(b.subjects):
                if tid not in raw:
                    continue
                raw[tid] += b.score * (1.0 if i == 0 else 0.5)
        # Smooth so a single frame's fluke doesn't yank the camera.
        for tid in live:
            prev = self.drama_ema.get(tid, 0.0)
            self.drama_ema[tid] = 0.6 * prev + 0.4 * raw.get(tid, 0.0)
        # Forget departed tracks.
        for tid in list(self.drama_ema):
            if tid not in live:
                del self.drama_ema[tid]
        return dict(self.drama_ema)

    def _pick_star(self, scores: Dict[int, float], live: set) -> Optional[int]:
        mode = self.cfg.follow
        if mode == "none" or not live:
            return None
        if mode.startswith("id:"):
            want = int(mode.split(":", 1)[1])
            return want if want in live else None
        # "director": drama-weighted with hysteresis.
        if not scores:
            return None
        best = max(scores, key=lambda k: scores[k])
        if self.star_id is None or self.star_id not in live:
            return best
        if scores.get(best, 0.0) > scores.get(self.star_id, 0.0) * self.cfg.hysteresis:
            return best
        return self.star_id

    def pick_star(self, tracks: List[Track], feats: FrameFeatures) -> Optional[int]:
        """Drama-weighted star selection with hysteresis (reusable by the mover)."""
        live = {t.id for t in tracks}
        scores = self._drama_scores(feats, live)
        self.star_id = self._pick_star(scores, live)
        return self.star_id

    def step(self, tracks: List[Track], feats: FrameFeatures) -> StageStep:
        live = {t.id for t in tracks}
        by_id = {t.id: t for t in tracks}
        scores = self._drama_scores(feats, live)

        if not self.cfg.enabled:
            return StageStep(None, None, StageCommand("ping"),
                             self.stage.position(), scores)

        self.star_id = self._pick_star(scores, live)
        if self.star_id is None or self.star_id not in by_id:
            return StageStep(None, None, StageCommand("ping"),
                             self.stage.position(), scores)

        star = by_id[self.star_id]
        # Desired stage offset = where the star sits relative to FOV centre.
        want_x = star.cx - self.cx0
        want_y = star.cy - self.cy0
        pos = self.stage.position()
        off = ((want_x - pos.x) ** 2 + (want_y - pos.y) ** 2) ** 0.5
        if off < self.cfg.deadzone:
            cmd = StageCommand("ping")
        else:
            cmd = StageCommand("move_abs", x=want_x, y=want_y,
                               feed=self.cfg.max_step)
        state = self.stage.apply(cmd)
        return StageStep(star_id=self.star_id, target=(star.cy, star.cx),
                         command=cmd, state=state, scores=scores)

    def viewport_center(self) -> tuple:
        """Where the stage is currently framing, in frame px (cy, cx)."""
        p = self.stage.position()
        return (self.cy0 + p.y, self.cx0 + p.x)

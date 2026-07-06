"""True moving-crop CNC stage.

The slide (world) is larger than the camera (sensor). The stage physically pans
across the slide to keep today's star centred, so microbes actually enter and
leave the field of view — the brief's core premise made literal. Tracking runs
in WORLD coordinates (stage-motion-compensated) so a panning stage doesn't look
like microbe motion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

from ..config import StageConfig
from ..vision.features import FrameFeatures
from ..vision.track import Track
from .controller import StageController


@dataclass
class MovingStep:
    star_id: Optional[int]
    center: tuple                 # stage centre in world coords (cy, cx)
    crop_origin: tuple            # (y0, x0) top-left of the sensor window in world coords
    star_in_frame: bool


class MovingStageController:
    """Drives the sensor crop across a larger world to follow the star."""

    def __init__(self, cfg: StageConfig, world_h: int, world_w: int,
                 sensor_h: int, sensor_w: int):
        self.cfg = cfg
        self.WH, self.WW = world_h, world_w
        self.sh, self.sw = sensor_h, sensor_w
        self.cy = world_h / 2.0
        self.cx = world_w / 2.0
        self._picker = StageController(cfg, world_h, world_w)   # reuse star selection

    def crop_origin(self) -> tuple:
        y0 = min(max(int(round(self.cy - self.sh / 2)), 0), max(0, self.WH - self.sh))
        x0 = min(max(int(round(self.cx - self.sw / 2)), 0), max(0, self.WW - self.sw))
        return y0, x0

    def step(self, tracks: List[Track], feats: FrameFeatures) -> MovingStep:
        star_id = None
        if self.cfg.enabled:
            star_id = self._picker.pick_star(tracks, feats)
        by_id = {t.id: t for t in tracks}

        if star_id is not None and star_id in by_id:
            star = by_id[star_id]
            # Aim where the star is heading (velocity feedforward), not just
            # where it is — cancels the follow lag for a moving subject.
            tgt_y = star.cy + self.cfg.lead * star.vy
            tgt_x = star.cx + self.cfg.lead * star.vx
            dy, dx = tgt_y - self.cy, tgt_x - self.cx
            dist = math.hypot(dy, dx)
            if dist > self.cfg.deadzone:
                if dist > self.cfg.max_step and dist > 0:
                    s = self.cfg.max_step / dist
                    dy, dx = dy * s, dx * s
                self.cy += dy
                self.cx += dx
            # Keep the sensor window inside the slide.
            self.cy = min(max(self.cy, self.sh / 2), self.WH - self.sh / 2)
            self.cx = min(max(self.cx, self.sw / 2), self.WW - self.sw / 2)

        y0, x0 = self.crop_origin()
        in_frame = False
        if star_id is not None and star_id in by_id:
            s = by_id[star_id]
            in_frame = (y0 <= s.cy < y0 + self.sh) and (x0 <= s.cx < x0 + self.sw)
        return MovingStep(star_id=star_id, center=(self.cy, self.cx),
                          crop_origin=(y0, x0), star_in_frame=in_frame)

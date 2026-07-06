"""A synthetic microscope world.

Why synthetic first? Because it lets us prototype and *score* the ENTIRE stack
(segment -> track -> drama -> stage) with zero downloads and perfect ground
truth. Real microscopy clips from HuggingFace/Kaggle slot in via
``video.io``/a real source later; the pipeline downstream is identical.

The world is a shallow "flow cell": microbes drift with a gentle current plus
Brownian jitter, occasionally divide (mitosis), sometimes chase or flee each
other, and enter/exit the field of view. That churn is exactly the raw
material a soap opera needs — arrivals, departures, pursuits and births.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np

from ..config import WorldConfig


# A few "species": (label, body RGB, elongation, wiggle) — purely cosmetic but
# gives the captioner distinct-looking creatures to anthropomorphise.
SPECIES = [
    ("paramecium", (120, 210, 180), 1.9, 0.9),
    ("amoeba",     (210, 160, 120), 1.2, 1.6),
    ("rotifer",    (170, 150, 220), 1.5, 0.7),
    ("diatom",     (220, 205, 130), 1.0, 0.2),
    ("euglena",    (140, 220, 130), 2.3, 1.1),
    ("ciliate",    (215, 140, 180), 1.6, 1.0),
]


@dataclass
class _Microbe:
    gid: int                 # ground-truth id (stable for the microbe's whole life)
    y: float
    x: float
    vy: float
    vx: float
    radius: float
    species: int
    phase: float
    age: int = 0
    alive: bool = True
    role: str = "drifter"    # drifter | predator | prey
    born_frame: int = 0


@dataclass
class GroundTruth:
    """Per-frame ground truth: gid -> (cy, cx, radius, species)."""

    frame: int
    items: dict


class SyntheticWorld:
    """Deterministic generator of microscope frames + ground truth."""

    def __init__(self, cfg: WorldConfig):
        self.cfg = cfg
        self.rng = np.random.RandomState(cfg.seed)
        # Sensor = what the camera captures; world (slide) = sensor * world_scale.
        # Physics and rendering happen in WORLD coords (self.H/W); at scale 1.0
        # the world equals the sensor, so all non-moving behaviour is unchanged.
        self.sensor_h, self.sensor_w = cfg.height, cfg.width
        self.H = int(round(cfg.height * cfg.world_scale))
        self.W = int(round(cfg.width * cfg.world_scale))
        self._next_gid = 0
        self.microbes: List[_Microbe] = []
        self._bg = self._make_background()
        for _ in range(cfg.n_start):
            self.microbes.append(self._spawn(edge=False, frame=0))
        # Designate one predator and one prey for guaranteed chase scenes.
        if len(self.microbes) >= 2:
            self.microbes[0].role = "predator"
            self.microbes[1].role = "prey"

    # ------------------------------------------------------------------ spawn
    def _spawn(self, edge: bool, frame: int) -> _Microbe:
        c = self.cfg
        gid = self._next_gid
        self._next_gid += 1
        species = int(self.rng.randint(len(SPECIES)))
        radius = float(self.rng.uniform(7, 15))
        if edge:
            # Enter from a random edge heading inward, riding the current.
            side = self.rng.randint(4)
            if side == 0:      # left
                y, x = self.rng.uniform(0, self.H), -radius
                vy, vx = self.rng.uniform(-0.4, 0.4), abs(self.rng.uniform(0.4, 1.2))
            elif side == 1:    # right
                y, x = self.rng.uniform(0, self.H), self.W + radius
                vy, vx = self.rng.uniform(-0.4, 0.4), -abs(self.rng.uniform(0.4, 1.2))
            elif side == 2:    # top
                y, x = -radius, self.rng.uniform(0, self.W)
                vy, vx = abs(self.rng.uniform(0.4, 1.2)), self.rng.uniform(-0.4, 0.4)
            else:              # bottom
                y, x = self.H + radius, self.rng.uniform(0, self.W)
                vy, vx = -abs(self.rng.uniform(0.4, 1.2)), self.rng.uniform(-0.4, 0.4)
        else:
            y = float(self.rng.uniform(0.15 * self.H, 0.85 * self.H))
            x = float(self.rng.uniform(0.15 * self.W, 0.85 * self.W))
            vy = float(self.rng.uniform(-1, 1))
            vx = float(self.rng.uniform(-1, 1))
        return _Microbe(
            gid=gid, y=y, x=x, vy=vy, vx=vx, radius=radius,
            species=species, phase=float(self.rng.uniform(0, 6.28)),
            born_frame=frame,
        )

    # ---------------------------------------------------------------- physics
    def _step(self, frame: int) -> None:
        c = self.cfg
        rng = self.rng
        alive = [m for m in self.microbes if m.alive]

        # Predator/prey biasing for chase scenes.
        preds = [m for m in alive if m.role == "predator"]
        preys = [m for m in alive if m.role == "prey"]
        for p in preds:
            if preys:
                t = min(preys, key=lambda q: (q.y - p.y) ** 2 + (q.x - p.x) ** 2)
                d = np.hypot(t.y - p.y, t.x - p.x) + 1e-6
                if d < 160:
                    p.vy += 0.10 * (t.y - p.y) / d
                    p.vx += 0.10 * (t.x - p.x) / d
        for q in preys:
            if preds:
                p = min(preds, key=lambda r: (r.y - q.y) ** 2 + (r.x - q.x) ** 2)
                d = np.hypot(p.y - q.y, p.x - q.x) + 1e-6
                if d < 120:
                    q.vy -= 0.12 * (p.y - q.y) / d
                    q.vx -= 0.12 * (p.x - q.x) / d

        for m in alive:
            m.age += 1
            m.phase += 0.3
            # Brownian jitter + gentle rightward current (the flow cell).
            m.vy += rng.uniform(-1, 1) * c.brownian
            m.vx += rng.uniform(-1, 1) * c.brownian + c.flow * 0.15
            # Mild drag so speeds stay bounded.
            m.vy *= 0.86
            m.vx *= 0.86
            m.y += m.vy
            m.x += m.vx
            # Exit when it fully leaves the field of view.
            margin = m.radius + 4
            if (m.x < -margin or m.x > self.W + margin
                    or m.y < -margin or m.y > self.H + margin):
                if m.age > 3:            # ignore the frame it's spawning in
                    m.alive = False

        # Mitosis: a microbe splits into two (great soap-opera "birth").
        for m in list(alive):
            if m.alive and rng.rand() < c.divide_prob and len(self._live()) < c.max_microbes:
                child = _Microbe(
                    gid=self._next_gid, y=m.y + rng.uniform(-6, 6),
                    x=m.x + rng.uniform(-6, 6), vy=-m.vy, vx=-m.vx,
                    radius=max(6.0, m.radius * 0.8), species=m.species,
                    phase=m.phase + 3.14, born_frame=frame, role="drifter",
                )
                self._next_gid += 1
                m.radius = max(6.0, m.radius * 0.85)
                self.microbes.append(child)

        # New arrivals drifting in from the edges.
        if len(self._live()) < c.max_microbes and rng.rand() < c.spawn_prob:
            self.microbes.append(self._spawn(edge=True, frame=frame))

    def _live(self) -> List[_Microbe]:
        return [m for m in self.microbes if m.alive]

    # --------------------------------------------------------------- rendering
    def _make_background(self) -> np.ndarray:
        """A dim, unevenly-lit microscope field with a soft vignette."""
        H, W = self.H, self.W
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        # Low-frequency illumination blotches.
        illum = np.zeros((H, W), np.float32)
        for _ in range(4):
            cy, cx = self.rng.uniform(0, H), self.rng.uniform(0, W)
            s = self.rng.uniform(H * 0.3, H * 0.7)
            illum += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))
        illum = illum / (illum.max() + 1e-6)
        # Vignette.
        cy, cx = H / 2, W / 2
        vig = 1.0 - 0.55 * (((yy - cy) / (H / 2)) ** 2 + ((xx - cx) / (W / 2)) ** 2)
        vig = np.clip(vig, 0.25, 1.0)
        if self.cfg.style == "brightfield":
            # Bright, warm, gently-vignetted field (microbes will be darker).
            base = (150 + 45 * illum) * (0.72 + 0.28 * vig)
            bg = np.stack([base * 1.02, base * 1.0, base * 0.95], axis=-1)
        else:
            base = (18 + 26 * illum) * vig
            bg = np.stack([base * 0.9, base * 1.0, base * 1.08], axis=-1)  # cool tint
        return np.clip(bg, 0, 255).astype(np.float32)

    def _render(self) -> np.ndarray:
        frame = self._bg.copy()
        # Fine sensor noise.
        frame += self.rng.uniform(-3, 3, frame.shape).astype(np.float32)
        H, W = self.H, self.W
        for m in self._live():
            speed = np.hypot(m.vy, m.vx)
            ang = np.arctan2(m.vy, m.vx)
            elong = SPECIES[m.species][2] * (1.0 + 0.15 * min(speed, 3))
            wig = 1.0 + 0.12 * SPECIES[m.species][3] * np.sin(m.phase)
            ry = m.radius * wig
            rx = m.radius * wig
            # Elongate along the direction of travel.
            R = int(np.ceil(m.radius * max(elong, 1.4) + 3))
            y0, y1 = max(0, int(m.y) - R), min(H, int(m.y) + R + 1)
            x0, x1 = max(0, int(m.x) - R), min(W, int(m.x) + R + 1)
            if y0 >= y1 or x0 >= x1:
                continue
            ly, lx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            dy = ly - m.y
            dx = lx - m.x
            # Rotate into the microbe's frame to elongate along heading.
            ca, sa = np.cos(-ang), np.sin(-ang)
            u = ca * dx - sa * dy      # along heading
            v = sa * dx + ca * dy      # perpendicular
            dist = np.sqrt((u / (rx * elong)) ** 2 + (v / ry) ** 2)
            body = np.clip(1.0 - (dist - 0.6) / 0.5, 0, 1)   # soft edge
            rim = np.clip(1.0 - np.abs(dist - 1.0) / 0.28, 0, 1) * 0.7
            col = np.array(SPECIES[m.species][1], np.float32)
            patch = frame[y0:y1, x0:x1]
            nucleus = np.clip(1.0 - dist / 0.35, 0, 1)
            if self.cfg.style == "brightfield":
                # Dark, faintly-tinted body that casts a shadow, ringed by a
                # bright phase-contrast halo, with a slightly darker nucleus.
                muted = 0.35 * col + 0.65 * np.array([70, 70, 78], np.float32)
                for ch in range(3):
                    patch[..., ch] = patch[..., ch] * (1 - 0.7 * body) + muted[ch] * (0.7 * body)
                    patch[..., ch] = np.clip(patch[..., ch] + 55 * rim, 0, 255)     # bright halo
                    patch[..., ch] = np.clip(patch[..., ch] - 35 * nucleus, 0, 255)  # dark nucleus
            else:
                for ch in range(3):
                    patch[..., ch] = patch[..., ch] * (1 - body) + col[ch] * body
                    patch[..., ch] = np.clip(patch[..., ch] - 45 * rim, 0, 255)
                for ch in range(3):
                    patch[..., ch] = np.clip(patch[..., ch] + 60 * nucleus, 0, 255)  # bright nucleus
            frame[y0:y1, x0:x1] = patch
        return np.clip(frame, 0, 255).astype(np.uint8)

    def _ground_truth(self, frame: int) -> GroundTruth:
        items = {
            m.gid: (m.y, m.x, m.radius, SPECIES[m.species][0])
            for m in self._live()
        }
        return GroundTruth(frame=frame, items=items)

    # ------------------------------------------------------------------- API
    def frames(self, n: int) -> Iterator[Tuple[np.ndarray, GroundTruth]]:
        """Yield ``n`` (frame, ground_truth) pairs, advancing the physics."""
        for f in range(n):
            if f > 0:
                self._step(f)
            yield self._render(), self._ground_truth(f)

    def render_only(self, n: int) -> Iterator[np.ndarray]:
        for frame, _gt in self.frames(n):
            yield frame

"""Configuration objects for the whole pipeline.

Every tunable knob the autoresearch loop might sweep lives here, so an
experiment is just "make a PipelineConfig, run it, score it".
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class WorldConfig:
    """Synthetic microbe world (our deterministic, ground-truth testbed)."""

    height: int = 360
    width: int = 540
    n_start: int = 7          # microbes visible at t=0
    max_microbes: int = 14
    spawn_prob: float = 0.06  # chance/frame a new microbe drifts in from an edge
    divide_prob: float = 0.010  # chance/frame a microbe undergoes mitosis
    flow: float = 0.35        # global drift (the "current" in the flow cell), px/frame
    brownian: float = 0.9     # random-walk jitter magnitude
    seed: int = 7


@dataclass
class SegmentConfig:
    backend: str = "classical"   # classical | sam2 | sam3  (only classical implemented today)
    threshold: float = 0.28      # foreground threshold on the normalized "microbe-ness" map
    min_area: int = 25           # drop specks smaller than this (px)
    max_area: int = 20000
    blur: int = 1                # box-blur radius pre-threshold (denoise)


@dataclass
class TrackConfig:
    max_dist: float = 55.0       # gating distance for greedy nearest-neighbour matching (px)
    max_missed: int = 8          # frames a track survives without a detection before it dies
    min_hits: int = 2            # detections before a track is "confirmed" / eligible for drama


@dataclass
class DramaConfig:
    backend: str = "template"    # template | llm  (llm swapped in later)
    caption_every: int = 6       # emit a fresh narrator line every N frames
    seed: int = 7
    spice: float = 1.0           # 0..2, how melodramatic the templates get


@dataclass
class StageConfig:
    enabled: bool = True
    follow: str = "director"     # director (most dramatic) | none | id:<n>
    deadzone: float = 40.0       # don't move the stage until the star drifts this far off-centre
    max_step: float = 24.0       # max stage move per frame (px), models CNC feed-rate limit
    hysteresis: float = 1.4      # a challenger must be this much more dramatic to steal the camera


@dataclass
class RenderConfig:
    enabled: bool = True
    scale: int = 1               # upscale factor for the output
    show_masks: bool = True
    show_tracks: bool = True
    show_stage: bool = True
    caption_lines: int = 2
    fps: int = 12


@dataclass
class PipelineConfig:
    world: WorldConfig = field(default_factory=WorldConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    drama: DramaConfig = field(default_factory=DramaConfig)
    stage: StageConfig = field(default_factory=StageConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    n_frames: int = 140

    def to_json(self, **kw) -> str:
        return json.dumps(asdict(self), **kw)

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        return cls(
            world=WorldConfig(**d.get("world", {})),
            segment=SegmentConfig(**d.get("segment", {})),
            track=TrackConfig(**d.get("track", {})),
            drama=DramaConfig(**d.get("drama", {})),
            stage=StageConfig(**d.get("stage", {})),
            render=RenderConfig(**d.get("render", {})),
            n_frames=d.get("n_frames", 140),
        )

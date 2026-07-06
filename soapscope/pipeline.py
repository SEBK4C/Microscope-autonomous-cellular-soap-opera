"""The end-to-end pipeline: source -> segment -> track -> features -> drama -> stage -> render.

One ``Pipeline`` is built from one ``PipelineConfig``; ``run`` consumes an
iterable of frames (synthetic or real) and returns annotated frames, the full
narration transcript, a per-frame stage-command log (what a microcontroller
would receive), and the metrics the autoresearch loop scores.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

import numpy as np

from .config import PipelineConfig
from .drama.captioner import CaptionEvent, make_captioner
from .drama.characters import CharacterRegistry
from .metrics import Metrics, compute_metrics
from .render.overlay import render_frame
from .stage.api import StageCommand
from .stage.controller import StageController
from .vision.features import FeatureAnalyzer
from .vision.segment import make_segmenter
from .vision.track import Tracker


@dataclass
class FrameRecord:
    frame_idx: int
    n_detections: int
    n_tracks: int
    star_id: Optional[int]
    caption: str
    stage_cmd: dict


@dataclass
class PipelineResult:
    frames: List[np.ndarray]
    transcript: List[CaptionEvent]
    records: List[FrameRecord]
    stage_commands: List[dict]
    metrics: Metrics
    config: PipelineConfig
    segmenter_polarity: Optional[str] = None   # what the classical segmenter resolved


class Pipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.segmenter = make_segmenter(cfg.segment)
        self.tracker = Tracker(cfg.track)
        self.registry = CharacterRegistry(cfg.drama.seed)
        self.captioner = make_captioner(cfg.drama)
        self.analyzer: Optional[FeatureAnalyzer] = None
        self.controller: Optional[StageController] = None

    def run(self, frames: Iterable, collect_frames: bool = True) -> PipelineResult:
        cfg = self.cfg
        annotated: List[np.ndarray] = []
        records: List[FrameRecord] = []
        stage_cmds: List[dict] = []
        detections_per_frame: List[int] = []
        frame_tracks: List[List[Tuple[int, float, float]]] = []
        track_lengths: dict = {}
        gts: List[Optional[object]] = []

        t0 = time.perf_counter()
        for i, item in enumerate(frames):
            if isinstance(item, tuple):
                frame, gt = item
            else:
                frame, gt = item, None
            if self.analyzer is None:
                H, W = frame.shape[:2]
                self.analyzer = FeatureAnalyzer(H, W)
                self.controller = StageController(cfg.stage, H, W)

            seg = self.segmenter.segment(frame)
            self.tracker.update(seg.detections)
            confirmed = self.tracker.confirmed_tracks()
            feats = self.analyzer.step(
                i, confirmed, self.tracker.entered, self.tracker.exited)
            caption = self.captioner.update(i, feats, self.registry)
            step = self.controller.step(confirmed, feats)

            for t in confirmed:
                track_lengths[t.id] = track_lengths.get(t.id, 0) + 1
            frame_tracks.append([(t.id, t.cy, t.cx) for t in confirmed])
            detections_per_frame.append(len(seg.detections))
            gts.append(gt)

            cmd_dict = _cmd_to_dict(i, step.command)
            stage_cmds.append(cmd_dict)
            records.append(FrameRecord(
                frame_idx=i, n_detections=len(seg.detections),
                n_tracks=len(confirmed), star_id=step.star_id,
                caption=caption.headline if caption else "", stage_cmd=cmd_dict))

            if collect_frames and cfg.render.enabled:
                annotated.append(render_frame(
                    frame, seg, confirmed, self.registry, caption, step,
                    cfg.render, i))
        elapsed = time.perf_counter() - t0

        metrics = compute_metrics(
            n_frames=len(records), elapsed_s=elapsed,
            detections_per_frame=detections_per_frame,
            track_lengths=track_lengths, frame_tracks=frame_tracks, gts=gts,
            caption_headlines=[e.headline for e in self.captioner_transcript()],
            caption_kinds=[e.kind for e in self.captioner_transcript()],
        )
        return PipelineResult(
            frames=annotated, transcript=self.captioner_transcript(),
            records=records, stage_commands=stage_cmds, metrics=metrics,
            config=cfg,
            segmenter_polarity=getattr(self.segmenter, "last_polarity", None))

    def captioner_transcript(self) -> List[CaptionEvent]:
        return getattr(self.captioner, "transcript", [])


def _cmd_to_dict(frame_idx: int, cmd: StageCommand) -> dict:
    d = {"t": frame_idx, "cmd": cmd.cmd}
    if cmd.cmd == "move_rel":
        d.update(dx=round(cmd.dx, 2), dy=round(cmd.dy, 2))
    elif cmd.cmd == "move_abs":
        d.update(x=round(cmd.x, 2), y=round(cmd.y, 2))
    return d


def run_synthetic(cfg: PipelineConfig, collect_frames: bool = True) -> PipelineResult:
    """Convenience: build the synthetic world from cfg and run the pipeline."""
    from .video.synthetic import SyntheticWorld
    world = SyntheticWorld(cfg.world)
    return Pipeline(cfg).run(world.frames(cfg.n_frames), collect_frames=collect_frames)

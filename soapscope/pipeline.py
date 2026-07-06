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
    moving: Optional[dict] = None              # moving-stage stats (offset, travel, ...)


class Pipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.segmenter = make_segmenter(cfg.segment)
        self.tracker = Tracker(cfg.track)
        self.registry = CharacterRegistry(cfg.drama.seed)
        self.captioner = make_captioner(cfg.drama)
        self.analyzer: Optional[FeatureAnalyzer] = None
        self.controller: Optional[StageController] = None

    def _step_frame(self, i: int, frame):
        """Core per-frame compute (segment → track → features → drama → stage).

        Lazily sizes the analyzer/controller to the frame. Shared by ``run``
        (batch) and ``stream`` (live), so both paths stay identical.
        """
        cfg = self.cfg
        if self.analyzer is None:
            H, W = frame.shape[:2]
            self.analyzer = FeatureAnalyzer(H, W)
            self.controller = StageController(cfg.stage, H, W)
        seg = self.segmenter.segment(frame)
        self.tracker.update(seg.detections)
        confirmed = self.tracker.confirmed_tracks()
        feats = self.analyzer.step(
            i, confirmed, self.tracker.entered, self.tracker.exited)
        caption = self.captioner.update(i, feats, self.registry,
                                        frame=frame, tracks=confirmed)
        step = self.controller.step(confirmed, feats)
        return seg, confirmed, feats, caption, step

    def stream(self, frames: Iterable):
        """Yield ``(annotated_frame, caption, tracks, stage_step)`` per frame.

        For the live web viewer: same pipeline as ``run`` but a generator that
        emits each rendered frame as it's produced (near-real-time).
        """
        cfg = self.cfg
        for i, item in enumerate(frames):
            frame = item[0] if isinstance(item, tuple) else item
            seg, confirmed, _feats, caption, step = self._step_frame(i, frame)
            annotated = render_frame(frame, seg, confirmed, self.registry,
                                     caption, step, cfg.render, i)
            yield annotated, caption, confirmed, step

    def run(self, frames: Iterable, collect_frames: bool = True) -> PipelineResult:
        cfg = self.cfg
        annotated: List[np.ndarray] = []
        records: List[FrameRecord] = []
        stage_cmds: List[dict] = []
        detections_per_frame: List[int] = []
        frame_tracks: List[List[Tuple[int, float, float]]] = []
        track_lengths: dict = {}
        gts: List[Optional[object]] = []

        showrunner, recap_ev, last = None, None, None
        if cfg.drama.season_path:
            from .drama.season import SeasonMemory, Showrunner
            showrunner = Showrunner(SeasonMemory(cfg.drama.season_path),
                                    seed=cfg.drama.seed,
                                    observe_every=cfg.drama.caption_every)
            recap = showrunner.recap()
            if recap:
                recap_ev = CaptionEvent(0, recap,
                                        ["📺 PREVIOUSLY, on As the Slide Turns…", recap],
                                        "RECAP", [])

        t0 = time.perf_counter()
        for i, item in enumerate(frames):
            if isinstance(item, tuple):
                frame, gt = item
            else:
                frame, gt = item, None

            seg, confirmed, feats, caption, step = self._step_frame(i, frame)
            if showrunner is not None:
                showrunner.observe(feats, self.registry, i)
            shown = (recap_ev if recap_ev is not None and i < cfg.drama.recap_frames
                     else caption)

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
                caption=shown.headline if shown else "", stage_cmd=cmd_dict))

            if collect_frames and cfg.render.enabled:
                annotated.append(render_frame(
                    frame, seg, confirmed, self.registry, shown, step, cfg.render, i))
            last = (frame, seg, confirmed, step)
        elapsed = time.perf_counter() - t0

        cliff_ev = None
        if showrunner is not None:
            cliff = showrunner.finish()
            n = len(records)
            cliff_ev = CaptionEvent(n, cliff,
                                    ["📺 NEXT TIME, on As the Slide Turns…", cliff],
                                    "CLIFFHANGER", [])
            if collect_frames and cfg.render.enabled and last is not None:
                lf, lseg, ltr, lstep = last
                for k in range(cfg.drama.cliffhanger_frames):
                    annotated.append(render_frame(lf, lseg, ltr, self.registry,
                                                  cliff_ev, lstep, cfg.render, n + k))

        metrics = compute_metrics(
            n_frames=len(records), elapsed_s=elapsed,
            detections_per_frame=detections_per_frame,
            track_lengths=track_lengths, frame_tracks=frame_tracks, gts=gts,
            caption_headlines=[e.headline for e in self.captioner_transcript()],
            caption_kinds=[e.kind for e in self.captioner_transcript()],
        )
        transcript = (([recap_ev] if recap_ev else [])
                      + self.captioner_transcript()
                      + ([cliff_ev] if cliff_ev else []))
        return PipelineResult(
            frames=annotated, transcript=transcript,
            records=records, stage_commands=stage_cmds, metrics=metrics,
            config=cfg,
            segmenter_polarity=getattr(self.segmenter, "last_polarity", None))

    def run_moving(self, world_frames: Iterable,
                   collect_frames: bool = True) -> PipelineResult:
        """Moving-crop stage: pan a sensor window across a larger slide (world).

        Segments the sensor crop, lifts detections to WORLD coords, tracks there
        (stage-motion-compensated), and pans the stage to follow the star.
        """
        import math
        from .render.overlay import render_moving_frame
        from .stage.moving import MovingStageController
        from .vision.segment import Detection

        cfg = self.cfg
        annotated: List[np.ndarray] = []
        records: List[FrameRecord] = []
        stage_cmds: List[dict] = []
        detections_per_frame: List[int] = []
        frame_tracks: List = []
        track_lengths: dict = {}
        gts: List = []
        mover = None
        offsets: List[float] = []
        travel = 0.0
        star_frames = 0
        star_in = 0

        t0 = time.perf_counter()
        for i, item in enumerate(world_frames):
            wframe, wgt = item if isinstance(item, tuple) else (item, None)
            WH, WW = wframe.shape[:2]
            sh, sw = min(cfg.world.height, WH), min(cfg.world.width, WW)
            if mover is None:
                self.analyzer = FeatureAnalyzer(WH, WW)
                mover = MovingStageController(cfg.stage, WH, WW, sh, sw)

            y0, x0 = mover.crop_origin()
            sensor = wframe[y0:y0 + sh, x0:x0 + sw]
            seg = self.segmenter.segment(sensor)
            world_dets = [Detection(
                label=d.label, area=d.area,
                centroid=(d.centroid[0] + y0, d.centroid[1] + x0),
                bbox=(d.bbox[0] + y0, d.bbox[1] + x0, d.bbox[2] + y0, d.bbox[3] + x0),
            ) for d in seg.detections]

            self.tracker.update(world_dets)
            confirmed = self.tracker.confirmed_tracks()
            feats = self.analyzer.step(
                i, confirmed, self.tracker.entered, self.tracker.exited)
            caption = self.captioner.update(i, feats, self.registry,
                                            frame=sensor, tracks=confirmed)

            pre = (mover.cy, mover.cx)
            mstep = mover.step(confirmed, feats)
            travel += math.hypot(mover.cy - pre[0], mover.cx - pre[1])
            by_id = {t.id: t for t in confirmed}
            if mstep.star_id in by_id:
                s = by_id[mstep.star_id]
                offsets.append(math.hypot(s.cy - pre[0], s.cx - pre[1]))
                star_frames += 1
                if mstep.star_in_frame:
                    star_in += 1

            for t in confirmed:
                track_lengths[t.id] = track_lengths.get(t.id, 0) + 1
            frame_tracks.append([(t.id, t.cy, t.cx) for t in confirmed])
            gts.append(wgt)
            detections_per_frame.append(len(seg.detections))
            cmd = {"t": i, "cmd": "move_abs",
                   "x": round(mover.cx, 1), "y": round(mover.cy, 1)}
            stage_cmds.append(cmd)
            records.append(FrameRecord(
                frame_idx=i, n_detections=len(seg.detections),
                n_tracks=len(confirmed), star_id=mstep.star_id,
                caption=caption.headline if caption else "", stage_cmd=cmd))

            if collect_frames and cfg.render.enabled:
                annotated.append(render_moving_frame(
                    sensor, seg, confirmed, self.registry, caption, mstep,
                    (y0, x0), (WH, WW), (sh, sw), cfg.render, i))
        elapsed = time.perf_counter() - t0

        metrics = compute_metrics(
            n_frames=len(records), elapsed_s=elapsed,
            detections_per_frame=detections_per_frame, track_lengths=track_lengths,
            frame_tracks=frame_tracks, gts=gts,
            caption_headlines=[e.headline for e in self.captioner_transcript()],
            caption_kinds=[e.kind for e in self.captioner_transcript()])
        moving = {
            "mean_star_offset": (sum(offsets) / len(offsets)) if offsets else None,
            "stage_travel": travel,
            "star_in_frame_pct": (star_in / star_frames) if star_frames else None,
        }
        return PipelineResult(
            frames=annotated, transcript=self.captioner_transcript(),
            records=records, stage_commands=stage_cmds, metrics=metrics,
            config=cfg, segmenter_polarity=getattr(self.segmenter, "last_polarity", None),
            moving=moving)

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

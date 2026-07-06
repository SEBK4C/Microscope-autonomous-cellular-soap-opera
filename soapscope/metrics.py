"""Metrics — the objective the autoresearch loop optimises.

We want the show to be (a) well-tracked, (b) varied and funny, and (c) fast.
``compute_metrics`` rolls those into interpretable sub-scores plus a single
scalar ``score`` so an experiment is trivially comparable to the baseline —
exactly Karpathy's "measure val, keep if better, else discard" recipe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple


@dataclass
class Metrics:
    n_frames: int
    fps: float
    mean_detections: float
    n_tracks_total: int
    mean_track_len: float
    max_track_len: int
    caption_count: int
    caption_variety: float          # unique / total headlines (0..1)
    kind_variety: int               # distinct beat kinds narrated
    gt_recall: Optional[float]      # fraction of visible GT microbes covered
    fragmentation: Optional[float]  # tracks / GT-ids (1.0 ideal)
    score: float

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        rec = "n/a" if self.gt_recall is None else f"{self.gt_recall:.2f}"
        frag = "n/a" if self.fragmentation is None else f"{self.fragmentation:.2f}"
        return (
            f"score={self.score:.3f} | fps={self.fps:.1f} | "
            f"recall={rec} | frag={frag} | "
            f"tracks={self.n_tracks_total} meanlen={self.mean_track_len:.1f} | "
            f"captions={self.caption_count} variety={self.caption_variety:.2f} "
            f"kinds={self.kind_variety}"
        )


def compute_metrics(
    n_frames: int,
    elapsed_s: float,
    detections_per_frame: List[int],
    track_lengths: Dict[int, int],
    frame_tracks: List[List[Tuple[int, float, float]]],
    gts: List[Optional[object]],
    caption_headlines: List[str],
    caption_kinds: List[str],
    match_radius: float = 22.0,
) -> Metrics:
    fps = n_frames / elapsed_s if elapsed_s > 0 else 0.0
    mean_det = sum(detections_per_frame) / max(1, len(detections_per_frame))
    lens = list(track_lengths.values())
    n_tracks = len(lens)
    mean_len = sum(lens) / n_tracks if n_tracks else 0.0
    max_len = max(lens) if lens else 0

    n_caps = len(caption_headlines)
    variety = (len(set(caption_headlines)) / n_caps) if n_caps else 0.0
    kind_variety = len(set(caption_kinds))

    # Ground-truth-based scores (synthetic world only).
    gt_recall: Optional[float] = None
    fragmentation: Optional[float] = None
    if gts and any(g is not None for g in gts):
        covered, total = 0, 0
        gt_ids = set()
        for frame_i, gt in enumerate(gts):
            if gt is None:
                continue
            items = getattr(gt, "items", {})
            tks = frame_tracks[frame_i] if frame_i < len(frame_tracks) else []
            for gid, (gy, gx, gr, _sp) in items.items():
                gt_ids.add(gid)
                total += 1
                if any((ty - gy) ** 2 + (tx - gx) ** 2 <= match_radius ** 2
                       for (_id, ty, tx) in tks):
                    covered += 1
        gt_recall = covered / total if total else 0.0
        fragmentation = n_tracks / max(1, len(gt_ids))

    # ---- single scalar objective (higher is better) --------------------
    track_stability = _clip(mean_len / (0.5 * max(1, n_frames)), 0, 1)
    speed = _clip(fps / 30.0, 0, 1)
    recall = gt_recall if gt_recall is not None else 0.7
    frag_pen = _clip(abs((fragmentation or 1.0) - 1.0), 0, 1)
    score = (0.30 * recall + 0.25 * track_stability + 0.20 * variety
             + 0.15 * speed + 0.10 * (1 - frag_pen))

    return Metrics(
        n_frames=n_frames, fps=fps, mean_detections=mean_det,
        n_tracks_total=n_tracks, mean_track_len=mean_len, max_track_len=max_len,
        caption_count=n_caps, caption_variety=variety, kind_variety=kind_variety,
        gt_recall=gt_recall, fragmentation=fragmentation, score=score,
    )


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

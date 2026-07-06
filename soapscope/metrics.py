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

import numpy as np


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
    mota: Optional[float] = None    # Multi-Object Tracking Accuracy (GT-only; higher better)
    id_switches: Optional[int] = None  # identity switches over the clip (lower better)
    caption_coherence: Optional[float] = None  # fraction of captions about the followed star (higher better)

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        rec = "n/a" if self.gt_recall is None else f"{self.gt_recall:.2f}"
        frag = "n/a" if self.fragmentation is None else f"{self.fragmentation:.2f}"
        mota = "n/a" if self.mota is None else f"{self.mota:.2f}"
        idsw = "n/a" if self.id_switches is None else str(self.id_switches)
        coh = "n/a" if self.caption_coherence is None else f"{self.caption_coherence:.2f}"
        return (
            f"score={self.score:.3f} | fps={self.fps:.1f} | "
            f"recall={rec} | frag={frag} | mota={mota} idsw={idsw} | "
            f"tracks={self.n_tracks_total} meanlen={self.mean_track_len:.1f} | "
            f"captions={self.caption_count} variety={self.caption_variety:.2f} "
            f"kinds={self.kind_variety} coherence={coh}"
        )


def mota_and_idsw(gts, frame_tracks, match_radius: float):
    """Standard MOTA + identity-switch count against ground truth.

    MOTA = 1 - (FN + FP + IDSW) / GT_total, where GT<->track matching each frame
    is optimal (Hungarian) within ``match_radius``. Returns (mota, idsw, fp, fn).
    """
    from .vision.assign import linear_sum_assignment
    prev: Dict[int, int] = {}          # gt_id -> track_id from the last frame it matched
    idsw = fp = fn = gt_total = 0
    for frame_i, gt in enumerate(gts):
        if gt is None:
            continue
        items = getattr(gt, "items", {})
        gt_ids = list(items.keys())
        gt_pts = [(items[g][0], items[g][1]) for g in gt_ids]
        tks = frame_tracks[frame_i] if frame_i < len(frame_tracks) else []
        tk_ids = [t[0] for t in tks]
        tk_pts = [(t[1], t[2]) for t in tks]
        gt_total += len(gt_ids)
        matched: Dict[int, int] = {}
        if gt_ids and tk_ids:
            cost = np.zeros((len(gt_ids), len(tk_ids)))
            for i, (gy, gx) in enumerate(gt_pts):
                for j, (ty, tx) in enumerate(tk_pts):
                    cost[i, j] = math.hypot(gy - ty, gx - tx)
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if cost[i, j] <= match_radius:
                    matched[gt_ids[i]] = tk_ids[j]
        fn += len(gt_ids) - len(matched)
        fp += len(tk_ids) - len(set(matched.values()))
        for g, tk in matched.items():
            if g in prev and prev[g] != tk:
                idsw += 1
            prev[g] = tk
    mota = 1.0 - (fn + fp + idsw) / max(1, gt_total)
    return mota, idsw, fp, fn


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
    caption_stars: Optional[List[Tuple[Optional[int], Optional[int]]]] = None,
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

    # Coherence: of the captions with both a protagonist and a followed star,
    # what fraction narrate the microbe the camera is actually on. Diagnostic
    # only (like MOTA) — the objective doesn't price it, but a soap opera should.
    caption_coherence: Optional[float] = None
    if caption_stars:
        pairs = [(p, s) for (p, s) in caption_stars if p is not None and s is not None]
        if pairs:
            caption_coherence = sum(p == s for p, s in pairs) / len(pairs)

    # Ground-truth-based scores (synthetic world only).
    gt_recall: Optional[float] = None
    fragmentation: Optional[float] = None
    mota: Optional[float] = None
    id_switches: Optional[int] = None
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
        mota, id_switches, _fp, _fn = mota_and_idsw(gts, frame_tracks, match_radius)

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
        mota=mota, id_switches=id_switches, caption_coherence=caption_coherence,
    )


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

"""Segmentation: frame -> object masks + detections.

The default ``ClassicalSegmenter`` is pure numpy (CPU, no model download) so
the whole stack runs anywhere. It normalises illumination, thresholds a
"microbe-ness" map, then does connected-components labelling. SAM2/SAM3
backends implement the SAME ``Segmenter`` interface and drop in later — the
tracker and drama layers never know the difference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..config import SegmentConfig


@dataclass
class Detection:
    label: int                       # id in the frame's label map (0 = background)
    centroid: tuple                  # (cy, cx) float
    bbox: tuple                      # (y0, x0, y1, x1)
    area: int

    @property
    def size(self) -> float:
        return float(self.area) ** 0.5


@dataclass
class SegResult:
    labels: np.ndarray               # (H, W) int32 label map
    detections: List[Detection]


def _to_gray(frame: np.ndarray) -> np.ndarray:
    f = frame.astype(np.float32)
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def _box_blur(a: np.ndarray, r: int) -> np.ndarray:
    """Fast box blur via an integral image (separable, O(HW))."""
    if r <= 0:
        return a
    H, W = a.shape
    ii = np.zeros((H + 1, W + 1), np.float64)
    ii[1:, 1:] = np.cumsum(np.cumsum(a, axis=0), axis=1)
    ys = np.arange(H)
    xs = np.arange(W)
    y0 = np.clip(ys - r, 0, H)[:, None]
    y1 = np.clip(ys + r + 1, 0, H)[:, None]
    x0 = np.clip(xs - r, 0, W)[None, :]
    x1 = np.clip(xs + r + 1, 0, W)[None, :]
    total = (ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0])
    count = (y1 - y0) * (x1 - x0)
    return (total / np.maximum(count, 1)).astype(np.float32)


def label_components(mask: np.ndarray) -> "tuple[np.ndarray, int]":
    """8-connected connected components, pure numpy/python.

    Iterates only over foreground pixels (raster order), so cost scales with
    the microbe area, not the whole frame — fast for sparse microscopy fields.
    """
    H, W = mask.shape
    labels = np.zeros((H, W), np.int32)
    parent = [0]                     # union-find; index 0 is background

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    ys, xs = np.nonzero(mask)
    next_label = 1
    for y, x in zip(ys.tolist(), xs.tolist()):
        neigh = []
        if y > 0:
            if x > 0 and labels[y - 1, x - 1]:
                neigh.append(labels[y - 1, x - 1])
            if labels[y - 1, x]:
                neigh.append(labels[y - 1, x])
            if x < W - 1 and labels[y - 1, x + 1]:
                neigh.append(labels[y - 1, x + 1])
        if x > 0 and labels[y, x - 1]:
            neigh.append(labels[y, x - 1])
        if not neigh:
            labels[y, x] = next_label
            parent.append(next_label)
            next_label += 1
        else:
            m = min(neigh)
            labels[y, x] = m
            for n in neigh:
                union(m, n)

    # Second pass: flatten to compact, contiguous labels.
    remap: dict = {}
    count = 0
    out = np.zeros_like(labels)
    for y, x in zip(ys.tolist(), xs.tolist()):
        r = find(labels[y, x])
        lbl = remap.get(r)
        if lbl is None:
            count += 1
            lbl = count
            remap[r] = lbl
        out[y, x] = lbl
    return out, count


class Segmenter:
    """Interface shared by every segmentation backend."""

    def segment(self, frame: np.ndarray) -> SegResult:  # pragma: no cover
        raise NotImplementedError


class ClassicalSegmenter(Segmenter):
    def __init__(self, cfg: SegmentConfig):
        self.cfg = cfg

    def _score(self, frame: np.ndarray) -> np.ndarray:
        gray = _to_gray(frame)
        if self.cfg.blur:
            gray = _box_blur(gray, self.cfg.blur)
        # Robust illumination-invariant normalisation: microbes sit in the
        # bright tail above a dim, unevenly-lit background.
        lo = np.percentile(gray, 45)
        hi = np.percentile(gray, 99)
        score = np.clip((gray - lo) / max(hi - lo, 1e-3), 0.0, 1.0)
        return score

    def segment(self, frame: np.ndarray) -> SegResult:
        cfg = self.cfg
        score = self._score(frame)
        mask = score >= cfg.threshold
        labels, n = label_components(mask)
        dets: List[Detection] = []
        out = np.zeros_like(labels)
        keep = 0
        for lbl in range(1, n + 1):
            ys, xs = np.nonzero(labels == lbl)
            area = int(ys.size)
            if area < cfg.min_area or area > cfg.max_area:
                continue
            keep += 1
            cy, cx = float(ys.mean()), float(xs.mean())
            bbox = (int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1)
            out[ys, xs] = keep
            dets.append(Detection(label=keep, centroid=(cy, cx), bbox=bbox, area=area))
        return SegResult(labels=out, detections=dets)


class SamSegmenter(Segmenter):
    """Placeholder for SAM2 / SAM3.

    Later autoresearch iterations install ``torch`` + the SAM checkpoint and
    fill this in (automatic-mask-generation for the first frame, then video
    propagation for tracking). It intentionally raises until then so the
    classical path stays the tested default.
    """

    def __init__(self, cfg: SegmentConfig):
        self.cfg = cfg

    def segment(self, frame: np.ndarray) -> SegResult:  # pragma: no cover
        raise NotImplementedError(
            "SAM backend not installed. `pip install -e .[sam]`, add a checkpoint, "
            "and implement soapscope.vision.segment.SamSegmenter. Until then use "
            "backend='classical'."
        )


def make_segmenter(cfg: SegmentConfig) -> Segmenter:
    if cfg.backend == "classical":
        return ClassicalSegmenter(cfg)
    if cfg.backend in ("sam2", "sam3"):
        return SamSegmenter(cfg)
    raise ValueError(f"unknown segmentation backend: {cfg.backend!r}")

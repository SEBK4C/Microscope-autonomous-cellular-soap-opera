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


def _median3(a: np.ndarray) -> np.ndarray:
    """3x3 median filter (kills salt-and-pepper / compression speckle)."""
    p = np.pad(a, 1, mode="edge")
    H, W = a.shape
    stack = np.stack([p[i:i + H, j:j + W] for i in range(3) for j in range(3)], axis=0)
    return np.median(stack, axis=0).astype(np.float32)


def _binary_dilate(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    m = mask
    for _ in range(iters):
        m = _box_blur(m.astype(np.float32), 1) > 1e-6      # any neighbour set
    return m


def _binary_erode(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    m = mask
    for _ in range(iters):
        m = _box_blur(m.astype(np.float32), 1) >= 1.0 - 1e-6  # all neighbours set
    return m


def _binary_open(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    """Opening = erode then dilate: removes specks/thin bridges, keeps blobs."""
    return _binary_dilate(_binary_erode(mask, iters), iters)


def _distance_transform(mask: np.ndarray, max_iters: int) -> np.ndarray:
    """Approximate distance-to-edge via iterative erosion (Chebyshev-ish).

    A foreground pixel's value = how many 3x3 erosions it survives, so blob
    centres peak and thin necks between touching blobs stay low. Pure numpy;
    cost scales with the largest microbe radius, not the frame.
    """
    dist = np.zeros(mask.shape, np.float32)
    cur = mask
    for _ in range(max(1, max_iters)):
        cur = _binary_erode(cur, 1)
        if not cur.any():
            break
        dist += cur
    return dist


def watershed_split(mask: np.ndarray, max_dist: int,
                    seed_frac: float) -> "tuple[np.ndarray, int]":
    """Split touching blobs with distance-transform markers + region growing.

    1. distance transform → peaks mark blob centres;
    2. seeds = per-component cores (dist ≥ seed_frac · component-peak), so two
       touching microbes give two seeds but a single blob gives one;
    3. grow the seed labels outward over the mask (nearest-seed flood) so the
       shared blob is partitioned along the neck.
    Returns a compact (labels, count) like ``label_components``.
    """
    base, nb = label_components(mask)
    if nb == 0:
        return base, 0
    dist = _distance_transform(mask, max_dist)

    # Per-component peak, broadcast back to pixels.
    peak = np.zeros(nb + 1, np.float32)
    np.maximum.at(peak, base.ravel(), dist.ravel())
    peak_map = peak[base]
    seeds_mask = (base > 0) & (dist >= np.maximum(seed_frac * peak_map, 1e-3))
    seeds, ns = label_components(seeds_mask)
    if ns <= nb:
        return base, nb                     # nothing extra to split — keep CC result

    # Grow seeds over the foreground (vectorised nearest-seed flood).
    labels = seeds.copy()
    for _ in range(max(1, max_dist) + 2):
        unl = (mask) & (labels == 0)
        if not unl.any():
            break
        up = np.zeros_like(labels); up[:-1, :] = labels[1:, :]
        dn = np.zeros_like(labels); dn[1:, :] = labels[:-1, :]
        lf = np.zeros_like(labels); lf[:, :-1] = labels[:, 1:]
        rt = np.zeros_like(labels); rt[:, 1:] = labels[:, :-1]
        cand = np.where(up > 0, up, np.where(dn > 0, dn,
                        np.where(lf > 0, lf, rt)))
        take = unl & (cand > 0)
        if not take.any():
            break
        labels[take] = cand[take]
    labels[~mask] = 0
    # Compact labels to 1..k.
    uniq = np.unique(labels)
    uniq = uniq[uniq > 0]
    remap = np.zeros(int(labels.max()) + 1, np.int32)
    for i, u in enumerate(uniq, start=1):
        remap[u] = i
    return remap[labels], len(uniq)


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
        self.last_polarity: Optional[str] = None   # what "auto" resolved to
        self._auto_polarity: Optional[str] = None   # cached auto decision (per video)
        self._bg: Optional[np.ndarray] = None       # running background (temporal mode)

    def _detect_polarity(self, gray: np.ndarray) -> str:
        """Do the objects darken or brighten the frame vs their local background?

        Compares the total positive vs negative deviation from a local-mean
        background. Object *bodies* have far more area than phase halos, so the
        sign of the net deviation robustly reveals polarity — and being
        gradient-relative, it is not fooled by vignetting or uneven lighting.
        """
        # Cap the background radius to a fraction of the frame: on a large frame
        # (real footage) this stays at adaptive_radius, but on small frames /
        # blobs-large-relative-to-frame it prevents the blur "halo" from swamping
        # the objects and inverting the sign.
        H, W = gray.shape
        r = min(max(6, self.cfg.adaptive_radius), max(4, min(H, W) // 8))
        bg = _box_blur(gray, r)
        resid = gray - bg
        pos = float(np.clip(resid, 0.0, None).sum())
        neg = float(np.clip(-resid, 0.0, None).sum())
        return "dark" if neg > pos * 1.05 else "bright"

    def _spatial_score(self, gray: np.ndarray) -> np.ndarray:
        """Polarity-aware "microbe-ness" of the whole blobs (the original path)."""
        cfg = self.cfg
        pol = cfg.polarity
        if pol == "auto":
            # Decide once on the first frame and keep it — a clip's polarity is
            # fixed, and caching avoids per-frame flicker (and recomputation).
            if self._auto_polarity is None:
                self._auto_polarity = self._detect_polarity(gray)
            pol = self._auto_polarity
        self.last_polarity = pol

        if not cfg.adaptive:
            # Global robust normalisation (illumination-invariant). The bright
            # branch is the original, unchanged default path.
            if pol == "bright":
                lo = np.percentile(gray, 45)
                hi = np.percentile(gray, 99)
                return np.clip((gray - lo) / max(hi - lo, 1e-3), 0.0, 1.0)
            # dark: microbes are the low tail — invert.
            hi = np.percentile(gray, 55)
            lo = np.percentile(gray, 1)
            return np.clip((hi - gray) / max(hi - lo, 1e-3), 0.0, 1.0)

        # Adaptive local thresholding: score = deviation from the local mean,
        # which survives strong illumination gradients and phase-contrast halos.
        bg = _box_blur(gray, max(3, cfg.adaptive_radius))
        resid = gray - bg
        signal = resid if pol == "bright" else -resid
        signal = np.clip(signal, 0.0, None)
        hi = np.percentile(signal, 99.5)
        return np.clip(signal / max(hi, 1e-3), 0.0, 1.0)

    def _score(self, frame: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        gray = _to_gray(frame)
        if cfg.median >= 3:
            gray = _median3(gray)
        if cfg.blur:
            gray = _box_blur(gray, cfg.blur)

        if not cfg.temporal:
            return self._spatial_score(gray)

        # Motion foreground: deviation from a slowly-updated background.
        # Erases static texture / compression noise — ideal for a mostly-still
        # microscope field with drifting microbes.
        if self._bg is None:
            self._bg = gray.copy()
        diff = np.abs(gray - self._bg)
        self._bg = (1.0 - cfg.bg_alpha) * self._bg + cfg.bg_alpha * gray
        motion = np.clip(diff / max(float(np.percentile(diff, 99.5)), 1e-3), 0.0, 1.0)

        if not cfg.hybrid:
            self.last_polarity = "temporal"
            return motion

        # Hybrid: take the WHOLE-body spatial score but keep it only where there
        # is motion nearby. Motion gates out static texture/noise; the spatial
        # score fills in solid bodies — so a moving blob is one detection, not
        # the leading/trailing crescents pure motion produces.
        spatial = self._spatial_score(gray)
        active = _binary_dilate(motion >= cfg.motion_gate, max(1, cfg.motion_dilate))
        self.last_polarity = "hybrid"
        return spatial * active

    def segment(self, frame: np.ndarray) -> SegResult:
        cfg = self.cfg
        score = self._score(frame)
        mask = score >= cfg.threshold
        if cfg.open_iter > 0:
            mask = _binary_open(mask, cfg.open_iter)
        if cfg.watershed:
            labels, n = watershed_split(mask, cfg.watershed_max_dist,
                                        cfg.watershed_seed_frac)
        else:
            labels, n = label_components(mask)
        if n == 0:
            return SegResult(labels=labels, detections=[])

        # Vectorised per-component stats (fast even with many noise blobs).
        ys, xs = np.nonzero(labels)
        lab = labels[ys, xs]
        area = np.bincount(lab, minlength=n + 1).astype(np.int64)
        sum_y = np.bincount(lab, weights=ys, minlength=n + 1)
        sum_x = np.bincount(lab, weights=xs, minlength=n + 1)
        min_y = np.full(n + 1, labels.shape[0], np.int64)
        max_y = np.zeros(n + 1, np.int64)
        min_x = np.full(n + 1, labels.shape[1], np.int64)
        max_x = np.zeros(n + 1, np.int64)
        np.minimum.at(min_y, lab, ys)
        np.maximum.at(max_y, lab, ys)
        np.minimum.at(min_x, lab, xs)
        np.maximum.at(max_x, lab, xs)

        remap = np.zeros(n + 1, np.int32)
        dets: List[Detection] = []
        keep = 0
        for lbl in range(1, n + 1):
            a = int(area[lbl])
            if a < cfg.min_area or a > cfg.max_area:
                continue
            keep += 1
            remap[lbl] = keep
            cy, cx = float(sum_y[lbl] / a), float(sum_x[lbl] / a)
            bbox = (int(min_y[lbl]), int(min_x[lbl]),
                    int(max_y[lbl]) + 1, int(max_x[lbl]) + 1)
            dets.append(Detection(label=keep, centroid=(cy, cx), bbox=bbox, area=a))
        out = remap[labels]
        return SegResult(labels=out, detections=dets)


def masks_to_segresult(masks, shape, min_area: int, max_area: int) -> SegResult:
    """Convert a list of per-object boolean masks (SAM output) to a SegResult.

    Model-agnostic glue: any SAM-family backend produces (N, H, W) masks; this
    turns them into the exact same Detection/label-map contract the classical
    path emits, so the tracker/drama/render layers don't care which segmenter
    ran. Masks are painted largest-first so small objects stay visible on top.
    """
    H, W = shape
    labels = np.zeros((H, W), np.int32)
    ordered = sorted((np.asarray(m, dtype=bool) for m in masks),
                     key=lambda m: -int(m.sum()))
    dets: List[Detection] = []
    keep = 0
    for m in ordered:
        area = int(m.sum())
        if area < min_area or area > max_area:
            continue
        ys, xs = np.nonzero(m)
        if ys.size == 0:
            continue
        keep += 1
        labels[m] = keep
        bbox = (int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1)
        dets.append(Detection(label=keep, centroid=(float(ys.mean()), float(xs.mean())),
                              bbox=bbox, area=area))
    return SegResult(labels=labels, detections=dets)


class SamSegmenter(Segmenter):
    """SAM2 / SAM3 (or an equivalent SOTA model) via a lazily-loaded backend.

    Runs automatic "segment everything" mask generation per frame and adapts the
    masks to the standard ``SegResult``. The heavy torch/ultralytics import lives
    in ``sam_backend`` and is only touched on first use, so importing soapscope
    stays light and the classical path remains the tested default. If no backend
    is installed, ``segment`` raises a clear, actionable error.
    """

    def __init__(self, cfg: SegmentConfig):
        self.cfg = cfg
        self.last_polarity = "sam"
        self._backend = None

    def _ensure_backend(self):
        if self._backend is None:
            from .sam_backend import load_sam_backend
            self._backend = load_sam_backend(self.cfg)
        return self._backend

    @property
    def backend_name(self) -> str:
        return getattr(self._backend, "name", "unloaded")

    def segment(self, frame: np.ndarray) -> SegResult:
        backend = self._ensure_backend()
        masks = backend.generate(np.asarray(frame, dtype=np.uint8))
        return masks_to_segresult(masks, frame.shape[:2],
                                  self.cfg.min_area, self.cfg.max_area)


def make_segmenter(cfg: SegmentConfig) -> Segmenter:
    if cfg.backend == "classical":
        return ClassicalSegmenter(cfg)
    if cfg.backend in ("sam", "sam2", "sam3"):
        return SamSegmenter(cfg)
    raise ValueError(f"unknown segmentation backend: {cfg.backend!r}")

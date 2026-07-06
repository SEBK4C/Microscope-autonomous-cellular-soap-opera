"""Lazy SAM-family segmentation backends.

Kept out of ``segment.py`` so the heavy torch/ultralytics imports only happen
when a SAM backend is actually selected — importing soapscope stays light and
the pure-numpy classical path remains the tested default.

Each backend exposes ``generate(frame_rgb) -> list[bool (H, W) mask]`` doing
"segment everything". ``load_sam_backend`` tries the installed options in order
and raises a clear, actionable error if none is available.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

# Known SAM-family weights and where to pull them on HuggingFace. The agent proxy
# allows HF but blocks GitHub release assets (403), which is exactly where
# ultralytics tries to auto-download — so a bare "FastSAM-s.pt" fetch fails. We
# resolve to a local file in models/ first, then fetch from HF into models/.
_HF_WEIGHTS = {
    "fastsam-s.pt": [("Uminosachi/FastSAM", "FastSAM-s.pt"),
                     ("mkshing/FastSAM", "FastSAM-s.pt")],
    "fastsam-x.pt": [("Uminosachi/FastSAM", "FastSAM-x.pt"),
                     ("mkshing/FastSAM", "FastSAM-x.pt")],
    "mobile_sam.pt": [("dhkim2810/MobileSAM", "mobile_sam.pt"),
                      ("Uminosachi/MobileSAM", "mobile_sam.pt")],
}
_MODELS_DIR = "models"
_MIN_WEIGHT_BYTES = 100_000   # reject error-page stubs (e.g. the 403 JSON blob)


def resolve_weight(name: str) -> str:
    """Turn a bare weight name into a real local path, fetching from HF if needed.

    ultralytics would auto-download from GitHub releases (proxy-blocked, 403), so
    we (1) use ``name`` if it already points at a real file, (2) look in
    ``models/``, (3) fetch from HuggingFace into ``models/``. Falls through to the
    original name if all else fails (lets ultralytics raise its own error)."""
    if os.path.isfile(name) and os.path.getsize(name) >= _MIN_WEIGHT_BYTES:
        return name
    base = os.path.basename(name)
    local = os.path.join(_MODELS_DIR, base)
    if os.path.isfile(local) and os.path.getsize(local) >= _MIN_WEIGHT_BYTES:
        return local
    fetched = _fetch_from_hf(base)
    return fetched or name


def _fetch_from_hf(basename: str) -> Optional[str]:
    repos = _HF_WEIGHTS.get(basename.lower())
    if not repos:
        return None
    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        return None
    os.makedirs(_MODELS_DIR, exist_ok=True)
    dest = os.path.join(_MODELS_DIR, basename)
    for repo_id, filename in repos:
        try:
            path = hf_hub_download(repo_id=repo_id, filename=filename)
            if os.path.getsize(path) >= _MIN_WEIGHT_BYTES:
                import shutil
                shutil.copy(path, dest)
                return dest
        except Exception:
            continue
    return None


class SamBackend:
    name = "base"

    def generate(self, frame_rgb: np.ndarray) -> List[np.ndarray]:  # pragma: no cover
        raise NotImplementedError


def _extract_ultralytics_masks(results) -> List[np.ndarray]:
    """Pull boolean masks out of an ultralytics Results list."""
    out: List[np.ndarray] = []
    for r in results:
        m = getattr(r, "masks", None)
        if m is None or getattr(m, "data", None) is None:
            continue
        data = m.data
        try:
            data = data.cpu().numpy()
        except AttributeError:
            data = np.asarray(data)
        for mm in data:
            out.append(np.asarray(mm) > 0.5)
    return out


class UltralyticsBackend(SamBackend):
    """FastSAM / MobileSAM / SAM via the ultralytics package (auto-downloads weights).

    FastSAM segments *everything* by default and is the most CPU-friendly, so it
    is preferred. MobileSAM/SAM are promptable; called without prompts they
    return whatever the model proposes for the frame.
    """

    def __init__(self, cfg, kind: str):
        self.cfg = cfg
        self.name = kind
        model_name = cfg.sam_model
        if kind == "fastsam":
            from ultralytics import FastSAM
            if not str(model_name).lower().startswith("fastsam"):
                model_name = "FastSAM-s.pt"
            self.model = FastSAM(resolve_weight(model_name))
            self._everything = True
        else:
            from ultralytics import SAM
            if kind == "mobile_sam" and "mobile" not in str(model_name).lower():
                model_name = "mobile_sam.pt"
            self.model = SAM(resolve_weight(model_name))
            self._everything = False

    def generate(self, frame_rgb: np.ndarray) -> List[np.ndarray]:
        kw = dict(verbose=False, imgsz=int(self.cfg.sam_imgsz))
        if self._everything:
            results = self.model(frame_rgb, retina_masks=True,
                                 conf=float(self.cfg.sam_conf), **kw)
        else:
            results = self.model(frame_rgb, **kw)
        return _extract_ultralytics_masks(results)


class SegmentAnythingBackend(SamBackend):
    """Meta's original segment-anything automatic mask generator.

    ``cfg.sam_model`` is ``"<arch>:<checkpoint_path>"`` (e.g.
    ``"vit_b:/models/sam_vit_b.pth"``) or just a checkpoint path (arch=vit_b).
    """

    name = "segment_anything"

    def __init__(self, cfg):
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        spec = cfg.sam_model
        arch, ckpt = spec.split(":", 1) if ":" in spec else ("vit_b", spec)
        sam = sam_model_registry[arch](checkpoint=resolve_weight(ckpt))
        sam.to("cpu")
        self.amg = SamAutomaticMaskGenerator(sam, points_per_side=12)

    def generate(self, frame_rgb: np.ndarray) -> List[np.ndarray]:
        anns = self.amg.generate(np.asarray(frame_rgb))
        return [np.asarray(a["segmentation"], dtype=bool) for a in anns]


_KINDS = {
    "fastsam": lambda cfg: UltralyticsBackend(cfg, "fastsam"),
    "mobile_sam": lambda cfg: UltralyticsBackend(cfg, "mobile_sam"),
    "ultralytics_sam": lambda cfg: UltralyticsBackend(cfg, "ultralytics_sam"),
    "segment_anything": SegmentAnythingBackend,
}


def load_sam_backend(cfg) -> SamBackend:
    order = ([cfg.sam_backend] if cfg.sam_backend != "auto"
             else ["fastsam", "mobile_sam", "segment_anything"])
    errors: List[str] = []
    for name in order:
        factory = _KINDS.get(name)
        if factory is None:
            errors.append(f"{name}: unknown backend")
            continue
        try:
            return factory(cfg)
        except Exception as e:  # noqa: BLE001 - report every failed backend
            errors.append(f"{name}: {type(e).__name__}: {e}")
    raise ImportError(
        "No SAM backend available. Install one of:\n"
        "  pip install -e .[sam]                  # ultralytics: FastSAM/MobileSAM/SAM (auto weights)\n"
        "  pip install segment-anything torch     # Meta SAM + a checkpoint in segment.sam_model\n"
        "Backends tried:\n  " + "\n  ".join(errors)
    )

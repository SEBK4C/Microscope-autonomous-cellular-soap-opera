"""Frame I/O with only Pillow as a dependency (no ffmpeg required).

- ``save_gif``  : animated preview, viewable anywhere.
- ``save_png``  : crisp single-frame inspection.
- ``load_frames_dir`` : read a directory of PNG/JPG frames as a source, so a
  real microscopy clip exported to frames flows through the same pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Sequence

import numpy as np
from PIL import Image


def save_png(frame: np.ndarray, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(path)
    return path


def save_gif(frames: Sequence[np.ndarray], path: str | Path, fps: int = 12,
             loop: int = 0) -> Path:
    """Write frames to an animated GIF. Frames are (H, W, 3) uint8."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not len(frames):
        raise ValueError("no frames to save")
    imgs = [Image.fromarray(np.asarray(f, dtype=np.uint8)).convert("P",
            palette=Image.ADAPTIVE, colors=256) for f in frames]
    duration = max(1, int(round(1000.0 / max(1, fps))))
    imgs[0].save(
        path, save_all=True, append_images=imgs[1:],
        duration=duration, loop=loop, optimize=False, disposal=1,
    )
    return path


def load_frames_dir(directory: str | Path,
                    pattern: str = "*") -> Iterator[np.ndarray]:
    """Yield frames from an ordered directory of images (a real-video source)."""
    directory = Path(directory)
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    files: List[Path] = sorted(
        p for p in directory.glob(pattern) if p.suffix.lower() in exts
    )
    for p in files:
        with Image.open(p) as im:
            yield np.asarray(im.convert("RGB"), dtype=np.uint8)

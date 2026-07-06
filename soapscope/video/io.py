"""Frame I/O with only Pillow as a dependency (no ffmpeg required).

- ``save_gif``  : animated preview, viewable anywhere.
- ``save_png``  : crisp single-frame inspection.
- ``load_frames_dir`` : read a directory of PNG/JPG frames as a source, so a
  real microscopy clip exported to frames flows through the same pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional, Sequence

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


def save_mp4(frames: Sequence[np.ndarray], path: str | Path, fps: int = 12) -> Path:
    """Write frames to an H.264 mp4 (small + high quality; needs imageio-ffmpeg)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not len(frames):
        raise ValueError("no frames to save")
    try:
        import imageio.v3 as iio  # type: ignore
    except ImportError as e:
        raise ImportError(
            "mp4 output needs imageio + a decoder: `pip install -e .[video]` "
            "(imageio, imageio-ffmpeg). GIF output has no such dependency."
        ) from e
    stack = np.stack([np.asarray(f, dtype=np.uint8) for f in frames])
    iio.imwrite(path, stack, fps=fps, codec="libx264", pixelformat="yuv420p")
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


def _open_video(path: str):
    """Return an iterator of frames from a video file, via imageio (lazy)."""
    try:
        import imageio.v3 as iio  # type: ignore
        return iio.imiter(path)
    except ImportError:
        pass
    try:
        import imageio  # type: ignore
        return imageio.get_reader(path)
    except ImportError as e:
        raise ImportError(
            "Reading video files needs imageio + a decoder: "
            "`pip install -e .[video]` (imageio, imageio-ffmpeg). "
            "Alternatively export the clip to PNG frames and use load_frames_dir "
            "(see data/README.md)."
        ) from e


def load_video(path: str | Path, stride: int = 1, max_frames: Optional[int] = None,
               max_width: Optional[int] = None) -> Iterator[np.ndarray]:
    """Yield RGB uint8 frames from a video file (mp4/webm/ogv/… via imageio).

    ``stride`` keeps every Nth frame, ``max_frames`` caps the count, and
    ``max_width`` downscales (keeping aspect) so the pure-numpy pipeline stays
    near-real-time on large footage.
    """
    stride = max(1, int(stride))
    kept = 0
    for i, frame in enumerate(_open_video(str(path))):
        if i % stride:
            continue
        arr = np.asarray(frame)
        if arr.ndim == 2:                       # grayscale -> RGB
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[-1] == 4:                  # drop alpha
            arr = arr[..., :3]
        arr = arr.astype(np.uint8)
        if max_width and arr.shape[1] > max_width:
            h, w = arr.shape[:2]
            new_h = int(round(h * max_width / w))
            arr = np.asarray(
                Image.fromarray(arr).resize((max_width, new_h), Image.LANCZOS),
                dtype=np.uint8)
        yield arr
        kept += 1
        if max_frames and kept >= max_frames:
            break

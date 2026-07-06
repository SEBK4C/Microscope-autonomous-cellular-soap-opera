"""Draw the annotated frame with Pillow.

Everything the viewer needs to follow the plot: coloured masks per character,
name tags, motion trails, the CNC stage's viewport + reticle on today's star,
and a caption bar with the current narration.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..config import RenderConfig
from ..drama.captioner import CaptionEvent
from ..drama.characters import CharacterRegistry
from ..stage.controller import StageStep
from ..vision.features import FrameFeatures
from ..vision.segment import SegResult
from ..vision.track import Track

# Distinct, readable label colours.
COLORS = [
    (239, 71, 111), (255, 209, 102), (6, 214, 160), (17, 138, 178),
    (255, 127, 80), (131, 56, 236), (58, 134, 255), (251, 86, 7),
    (46, 196, 182), (231, 29, 54), (255, 159, 28), (162, 215, 78),
    (199, 125, 255), (0, 187, 249),
]


def color_for(track_id: int) -> tuple:
    return COLORS[track_id % len(COLORS)]


@lru_cache(maxsize=8)
def _font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)   # Pillow >= 10.1
    except TypeError:                              # pragma: no cover - old Pillow
        return ImageFont.load_default()


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_w: int) -> List[str]:
    words = text.split()
    lines, cur = [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if draw.textlength(trial, font=font) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _tint_masks(frame: np.ndarray, seg: SegResult, tracks: List[Track],
                alpha: float = 0.45) -> np.ndarray:
    out = frame.astype(np.float32)
    labels = seg.labels
    for t in tracks:
        if t.label is None:
            continue
        m = labels == t.label
        if not m.any():
            continue
        c = np.array(color_for(t.id), np.float32)
        out[m] = out[m] * (1 - alpha) + c * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def render_frame(frame: np.ndarray, seg: SegResult, tracks: List[Track],
                 reg: CharacterRegistry, caption: Optional[CaptionEvent],
                 stage: Optional[StageStep], cfg: RenderConfig,
                 frame_idx: int, episode: int = 1) -> np.ndarray:
    H, W = frame.shape[:2]
    base_arr = _tint_masks(frame, seg, tracks) if cfg.show_masks else frame
    base = Image.fromarray(base_arr).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    f_small = _font(13)
    f_med = _font(15)
    f_big = _font(18)

    star_id = stage.star_id if stage else None

    if cfg.show_tracks:
        for t in tracks:
            col = color_for(t.id)
            # Motion trail.
            if len(t.history) > 1:
                pts = [(cx, cy) for (cy, cx) in t.history[-24:]]
                d.line(pts, fill=col + (150,), width=2)
            y0, x0, y1, x1 = t.bbox
            is_star = (t.id == star_id)
            width = 3 if is_star else 1
            d.rectangle([x0, y0, x1, y1], outline=col + (255,), width=width)
            ch = reg.get(t.id)
            tag = f"{'★ ' if is_star else ''}{ch.short}"
            tw = d.textlength(tag, font=f_small)
            ty = max(0, y0 - 15)
            d.rectangle([x0, ty, x0 + tw + 6, ty + 14], fill=(0, 0, 0, 160))
            d.text((x0 + 3, ty + 1), tag, font=f_small, fill=col + (255,))

    # CNC stage viewport + reticle.
    if cfg.show_stage and stage is not None and stage.target is not None:
        vh, vw = H * 0.42, W * 0.42
        # Viewport follows the (rate-limited) stage position.
        vcy = H / 2 + stage.state.y
        vcx = W / 2 + stage.state.x
        d.rectangle([vcx - vw / 2, vcy - vh / 2, vcx + vw / 2, vcy + vh / 2],
                    outline=(255, 255, 255, 180), width=2)
        d.text((vcx - vw / 2 + 4, vcy - vh / 2 + 2), "CNC FOV",
               font=f_small, fill=(255, 255, 255, 200))
        # Reticle on the star's true position (the tracking target).
        ry, rx = stage.target
        r = 12
        d.line([rx - r, ry, rx + r, ry], fill=(255, 255, 255, 220), width=1)
        d.line([rx, ry - r, rx, ry + r], fill=(255, 255, 255, 220), width=1)
        d.ellipse([rx - r, ry - r, rx + r, ry + r], outline=(255, 255, 255, 220), width=1)

    # Top banner.
    d.rectangle([0, 0, W, 22], fill=(0, 0, 0, 120))
    d.text((6, 3), "AS THE SLIDE TURNS", font=f_med, fill=(255, 214, 102, 255))
    right = f"ep {episode}  •  frame {frame_idx:03d}  •  cast {len(tracks)}"
    d.text((W - d.textlength(right, font=f_small) - 6, 5), right,
           font=f_small, fill=(220, 220, 220, 230))

    # Caption bar (bottom).
    if caption is not None and caption.lines:
        lines: List[str] = []
        for ln in caption.lines[: max(1, cfg.caption_lines)]:
            lines.extend(_wrap(d, ln, f_med, W - 24))
        lines = lines[: max(1, cfg.caption_lines) + 1]
        bar_h = 12 + 20 * len(lines)
        d.rectangle([0, H - bar_h, W, H], fill=(0, 0, 0, 170))
        d.rectangle([0, H - bar_h, W, H - bar_h + 3], fill=(255, 214, 102, 220))
        for i, ln in enumerate(lines):
            fill = (255, 255, 255, 240) if i == 0 else (200, 220, 255, 230)
            d.text((12, H - bar_h + 8 + i * 20), ln, font=f_med, fill=fill)

    out = Image.alpha_composite(base, overlay).convert("RGB")
    if cfg.scale and cfg.scale != 1:
        out = out.resize((W * cfg.scale, H * cfg.scale), Image.NEAREST)
    return np.asarray(out, dtype=np.uint8)

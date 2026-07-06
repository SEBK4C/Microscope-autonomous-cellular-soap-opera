#!/usr/bin/env python3
"""Split a video file into an ordered directory of PNG frames.

The ffmpeg-free path (uses imageio): handy when you want to inspect/curate
frames, or feed them through ``soapscope run --input <dir>``.

    python scripts/frames_from_video.py data/videos/clip.webm data/frames/clip \
        --stride 2 --max-frames 120 --max-width 640
"""

from __future__ import annotations

import argparse
from pathlib import Path

from soapscope.video.io import load_video, save_png


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("out_dir")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--max-width", type=int, default=0)
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    n = 0
    for i, frame in enumerate(load_video(
            args.video, stride=args.stride,
            max_frames=args.max_frames or None,
            max_width=args.max_width or None)):
        save_png(frame, out / f"{i:05d}.png")
        n += 1
    print(f"[frames] wrote {n} frames to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

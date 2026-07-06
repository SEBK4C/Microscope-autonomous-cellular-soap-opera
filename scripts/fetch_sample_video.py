#!/usr/bin/env python3
"""Fetch a small, public-domain microscopy clip to prototype on.

Downloads the first reachable clip from a curated list of Wikimedia Commons
videos (public domain / CC) into ``data/videos/``. No API key needed. If the
network is unavailable, it prints the URLs so you can grab one by hand.

    python scripts/fetch_sample_video.py                 # grab the default small clip
    python scripts/fetch_sample_video.py --list          # show candidates
    python scripts/fetch_sample_video.py --url <URL>     # fetch a specific URL

Then run the pipeline on it:

    python -m soapscope.cli run --input data/videos/<file> --max-frames 80
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

# Curated public-domain / CC microscopy clips (smallest / most-decodable first).
# .webm = VP8/VP9, .ogv/.ogg = Theora — all decodable via imageio-ffmpeg.
CANDIDATES = [
    ("Swift_ciliate.webm",
     "https://upload.wikimedia.org/wikipedia/commons/3/3c/Swift_ciliate_DSC_1523.webm"),
    ("Rotifer_video.ogv",
     "https://upload.wikimedia.org/wikipedia/commons/1/1d/Rotifer_video.ogv"),
    ("Rugby_ball_ciliate.webm",
     "https://upload.wikimedia.org/wikipedia/commons/f/f4/Rugby_ball_ciliate_DSC_2291.webm"),
    ("Paramecium_microbescope.webm",
     "https://upload.wikimedia.org/wikipedia/commons/b/b3/"
     "Paramecium_recorded_with_Microbescope%2C_2014-10-14.webm"),
    ("Nervous_ciliate.webm",
     "https://upload.wikimedia.org/wikipedia/commons/0/02/Nervous_ciliate_DSC_6896.webm"),
]

DEST = Path("data/videos")
UA = "SoapScope/0.1 (autoresearch prototype; https://github.com/)"


def _download(url: str, path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=60) as r, open(path, "wb") as fh:
            fh.write(r.read())
        print(f"[fetch] saved {path} ({path.stat().st_size/1e6:.2f} MB)")
        return True
    except Exception as e:  # noqa: BLE001 - best-effort fetch
        print(f"[fetch] failed {url}: {e}")
        if path.exists():
            path.unlink(missing_ok=True)
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", help="download this exact URL instead of the defaults")
    ap.add_argument("--all", action="store_true", help="try to fetch every candidate")
    ap.add_argument("--list", action="store_true", help="list candidates and exit")
    args = ap.parse_args(argv)

    if args.list:
        for name, url in CANDIDATES:
            print(f"{name:28s} {url}")
        return 0

    if args.url:
        name = args.url.split("/")[-1].split("?")[0] or "clip.webm"
        ok = _download(args.url, DEST / name)
        return 0 if ok else 1

    targets = CANDIDATES if args.all else CANDIDATES
    for name, url in targets:
        if _download(url, DEST / name):
            print(f"\n[fetch] ready. Try:\n  "
                  f"python -m soapscope.cli run --input {DEST / name} --max-frames 80\n")
            if not args.all:
                return 0
    print("[fetch] no clip could be downloaded (offline?). URLs above — grab one "
          "manually into data/videos/ and see data/README.md.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

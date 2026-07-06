#!/usr/bin/env python3
"""Fetch SAM-family weights from HuggingFace into ``models/``.

The agent proxy (and many locked-down networks) allow HuggingFace but block
GitHub release assets — which is exactly where ultralytics tries to auto-download
FastSAM/MobileSAM, so a bare ``FastSAM(...)`` call 403s. This grabs the real
weights from HF mirrors instead, so ``segment.backend="sam"`` just works offline
afterwards (weights are git-ignored).

    python scripts/fetch_sam_weights.py                 # FastSAM-s (fast, near-real-time on CPU)
    python scripts/fetch_sam_weights.py --all           # + FastSAM-x, mobile_sam
    python scripts/fetch_sam_weights.py --name mobile_sam.pt

Then:

    python -m soapscope.cli demo --backend sam          # segment-everything with FastSAM
"""

from __future__ import annotations

import argparse
import sys

# Reuse the backend's resolver so there is one source of truth for HF mirrors.
from soapscope.vision.sam_backend import _fetch_from_hf, _HF_WEIGHTS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="FastSAM-s.pt",
                    help="weight filename to fetch (default: FastSAM-s.pt)")
    ap.add_argument("--all", action="store_true",
                    help="fetch every known weight (FastSAM-s/-x, mobile_sam)")
    ap.add_argument("--list", action="store_true", help="list known weights + HF sources")
    args = ap.parse_args()

    if args.list:
        for name, repos in _HF_WEIGHTS.items():
            srcs = ", ".join(f"{r}/{f}" for r, f in repos)
            print(f"{name:16s} <- {srcs}")
        return 0

    names = list(_HF_WEIGHTS) if args.all else [args.name]
    ok = True
    for name in names:
        dest = _fetch_from_hf(name)   # matches on basename.lower() internally
        if dest:
            print(f"✓ {name} -> {dest}")
        else:
            print(f"✗ {name}: no reachable HF source (or huggingface_hub missing)")
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

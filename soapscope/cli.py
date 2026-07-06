"""Command-line entry point.

    soapscope demo                 # synthetic world -> annotated gif + transcript
    soapscope run --input frames/  # a real clip exported to frames
    soapscope experiment           # Karpathy-style autoresearch hill-climb
    soapscope sweep --param ...     # grid-sweep one knob
    soapscope protocol             # print the CNC microcontroller wire protocol
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from .config import PipelineConfig
from .pipeline import Pipeline, run_synthetic


def _write_outputs(res, out_dir: Path, fps: int) -> None:
    from .video.io import save_gif, save_png
    out_dir.mkdir(parents=True, exist_ok=True)
    if res.frames:
        save_gif(res.frames, out_dir / "episode.gif", fps=fps)
        save_png(res.frames[len(res.frames) // 2], out_dir / "frame.png")
    with open(out_dir / "transcript.txt", "w", encoding="utf-8") as fh:
        for ev in res.transcript:
            fh.write(f"[{ev.frame_idx:04d}] ({ev.kind}) " + " / ".join(ev.lines) + "\n")
    with open(out_dir / "stage_commands.jsonl", "w", encoding="utf-8") as fh:
        for c in res.stage_commands:
            fh.write(json.dumps(c) + "\n")
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(res.metrics.as_dict(), fh, indent=2)


def cmd_demo(args) -> int:
    cfg = PipelineConfig()
    cfg.n_frames = args.frames
    cfg.world.seed = args.seed
    cfg.drama.seed = args.seed
    cfg.drama.spice = args.spice
    cfg.stage.enabled = not args.no_stage
    cfg.drama.backend = args.narrator
    cfg.drama.llm_model = args.llm_model
    cfg.world.style = args.style
    cfg.segment.polarity = args.polarity
    # Bright-field footage segments best with adaptive local thresholding.
    cfg.segment.adaptive = (args.adaptive if args.adaptive is not None
                            else args.style == "brightfield")
    if args.width:
        cfg.world.width = args.width
    if args.height:
        cfg.world.height = args.height
    print(f"[demo] rendering {cfg.n_frames} frames "
          f"({cfg.world.width}x{cfg.world.height}) style={args.style} "
          f"polarity={args.polarity} adaptive={cfg.segment.adaptive} …")
    res = run_synthetic(cfg, collect_frames=True)
    out_dir = Path(args.out)
    _write_outputs(res, out_dir, cfg.render.fps)
    print("[demo]", res.metrics.summary())
    print(f"[demo] wrote {out_dir}/episode.gif, transcript.txt, "
          f"stage_commands.jsonl, metrics.json")
    return 0


def cmd_run(args) -> int:
    import os
    from .video.io import load_frames_dir, load_video
    cfg = PipelineConfig()
    cfg.segment.min_area = args.min_area
    if args.backend == "sam":
        # SAM segments directly; classical denoising knobs don't apply.
        cfg.segment.backend = "sam3"
        cfg.segment.sam_backend = args.sam_backend
        cfg.segment.sam_model = args.sam_model
        cfg.segment.sam_imgsz = args.sam_imgsz
        print("[run] NOTE: SAM on CPU is far from real-time (~0.03 fps); "
              "use a GPU, or the default classical backend for speed.")
    else:
        cfg.segment.backend = "classical"
        cfg.segment.polarity = args.polarity  # real clips: auto-detect polarity
        cfg.segment.adaptive = args.adaptive
        # De-fragmentation bundle for noisy real footage (AUTORESEARCH_JOURNAL #1).
        cfg.segment.temporal = args.temporal  # motion foreground erases static noise
        cfg.segment.hybrid = args.hybrid      # whole-body gating (best for CLEAN footage)
        cfg.segment.open_iter = args.open     # despeckle the mask
        if args.temporal:                     # coast longer through fast motion
            cfg.track.max_missed = 12
            cfg.track.max_dist = 65.0
            cfg.track.min_hits = 3
    if os.path.isdir(args.input):
        frames = load_frames_dir(args.input, pattern=args.pattern)
        print(f"[run] processing frame directory {args.input} "
              f"(polarity={args.polarity}, adaptive={args.adaptive}) …")
    else:
        frames = load_video(args.input, stride=args.stride,
                            max_frames=args.max_frames or None,
                            max_width=args.max_width or None)
        print(f"[run] decoding video {args.input} "
              f"(stride={args.stride}, max_frames={args.max_frames}, "
              f"max_width={args.max_width}, polarity={args.polarity}, "
              f"adaptive={args.adaptive}) …")
    res = Pipeline(cfg).run(frames, collect_frames=True)
    out_dir = Path(args.out)
    _write_outputs(res, out_dir, cfg.render.fps)
    print("[run]", res.metrics.summary())
    if res.segmenter_polarity:
        print(f"[run] resolved polarity: {res.segmenter_polarity}")
    print(f"[run] wrote {out_dir}/episode.gif and friends")
    return 0


def cmd_experiment(args) -> int:
    from .autoresearch.loop import autoresearch, append_journal
    base = PipelineConfig()
    best, history = autoresearch(base, rounds=args.rounds, n_frames=args.frames,
                                 seed=args.seed)
    kept = [h for h in history if h.kept]
    print("\n[experiment] leaderboard (kept steps):")
    for h in kept:
        print(f"  {h.score:.3f}  {h.name}")
    if args.journal:
        lines = [f"- `{h.name}` → score **{h.score:.3f}** — {h.metrics.summary()}"
                 for h in history]
        body = (
            "Ran autoresearch hill-climb "
            f"(rounds={args.rounds}, frames={args.frames}, seed={args.seed}).\n\n"
            + "\n".join(lines)
            + f"\n\n**Best kept score: {kept[-1].score:.3f}**"
        )
        append_journal(args.journal, "autoresearch run", body)
        print(f"[experiment] appended results to {args.journal}")
    return 0


def cmd_sweep(args) -> int:
    from .autoresearch.loop import sweep
    base = PipelineConfig()
    raw = [v.strip() for v in args.values.split(",")]
    values: List = []
    for v in raw:
        try:
            values.append(int(v))
        except ValueError:
            try:
                values.append(float(v))
            except ValueError:
                values.append(v)
    sweep(base, args.param, values, n_frames=args.frames)
    return 0


def cmd_protocol(args) -> int:
    from .stage.api import MICROCONTROLLER_PROTOCOL
    print(MICROCONTROLLER_PROTOCOL)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="soapscope",
                                description="Autonomous cellular soap opera.")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="run the synthetic world end to end")
    d.add_argument("--frames", type=int, default=140)
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--spice", type=float, default=1.0)
    d.add_argument("--narrator", choices=["template", "llm"], default="template",
                   help="template = instant & offline; llm = small local model "
                        "(needs .[llm]; ~2.4 s/caption on CPU)")
    d.add_argument("--llm-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    d.add_argument("--style", choices=["darkfield", "brightfield"], default="darkfield",
                   help="darkfield = bright microbes/dark bg; brightfield = dark microbes/light bg")
    d.add_argument("--polarity", choices=["bright", "dark", "auto"], default="auto")
    d.add_argument("--adaptive", action=argparse.BooleanOptionalAction, default=None,
                   help="local adaptive thresholding (auto-on for brightfield)")
    d.add_argument("--width", type=int, default=0)
    d.add_argument("--height", type=int, default=0)
    d.add_argument("--no-stage", action="store_true")
    d.add_argument("--out", default="out")
    d.set_defaults(func=cmd_demo)

    r = sub.add_parser("run", help="run on a real clip (video file or frame directory)")
    r.add_argument("--input", required=True,
                   help="a video file (mp4/webm/ogv/…) or a directory of PNG/JPG frames")
    r.add_argument("--pattern", default="*")
    r.add_argument("--backend", choices=["classical", "sam"], default="classical",
                   help="classical = pure-numpy, near-real-time; sam = SAM/MobileSAM "
                        "(needs .[sam] + weights; slow on CPU)")
    r.add_argument("--sam-backend", default="auto",
                   help="fastsam | mobile_sam | segment_anything | auto")
    r.add_argument("--sam-model", default="FastSAM-s.pt",
                   help="SAM model name or path (e.g. models/mobile_sam.pt)")
    r.add_argument("--sam-imgsz", type=int, default=512, help="SAM inference size")
    r.add_argument("--polarity", choices=["bright", "dark", "auto"], default="auto")
    r.add_argument("--adaptive", action=argparse.BooleanOptionalAction, default=True)
    r.add_argument("--temporal", action=argparse.BooleanOptionalAction, default=True,
                   help="motion foreground (best for a static microscope field); "
                        "disable for a moving/panning stage")
    r.add_argument("--hybrid", action=argparse.BooleanOptionalAction, default=False,
                   help="temporal: keep whole bodies via spatial gating — best for "
                        "CLEAN footage; noisy compressed clips prefer pure motion")
    r.add_argument("--open", type=int, default=1, help="morphological opening iterations")
    r.add_argument("--min-area", type=int, default=45, help="min detection area (px)")
    r.add_argument("--stride", type=int, default=1, help="keep every Nth video frame")
    r.add_argument("--max-frames", type=int, default=240)
    r.add_argument("--max-width", type=int, default=640, help="downscale wide footage")
    r.add_argument("--out", default="out")
    r.set_defaults(func=cmd_run)

    e = sub.add_parser("experiment", help="autoresearch hill-climb")
    e.add_argument("--rounds", type=int, default=12)
    e.add_argument("--frames", type=int, default=90)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--journal", default="")
    e.set_defaults(func=cmd_experiment)

    s = sub.add_parser("sweep", help="grid-sweep one config knob")
    s.add_argument("--param", required=True, help="e.g. segment.threshold")
    s.add_argument("--values", required=True, help="comma-separated values")
    s.add_argument("--frames", type=int, default=90)
    s.set_defaults(func=cmd_sweep)

    pr = sub.add_parser("protocol", help="print the CNC microcontroller protocol")
    pr.set_defaults(func=cmd_protocol)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

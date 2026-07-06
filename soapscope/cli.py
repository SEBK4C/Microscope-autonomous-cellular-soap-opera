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
    if args.width:
        cfg.world.width = args.width
    if args.height:
        cfg.world.height = args.height
    print(f"[demo] rendering {cfg.n_frames} frames "
          f"({cfg.world.width}x{cfg.world.height}) seed={args.seed} …")
    res = run_synthetic(cfg, collect_frames=True)
    out_dir = Path(args.out)
    _write_outputs(res, out_dir, cfg.render.fps)
    print("[demo]", res.metrics.summary())
    print(f"[demo] wrote {out_dir}/episode.gif, transcript.txt, "
          f"stage_commands.jsonl, metrics.json")
    return 0


def cmd_run(args) -> int:
    from .video.io import load_frames_dir
    cfg = PipelineConfig()
    frames = load_frames_dir(args.input, pattern=args.pattern)
    print(f"[run] processing frames from {args.input} …")
    res = Pipeline(cfg).run(frames, collect_frames=True)
    out_dir = Path(args.out)
    _write_outputs(res, out_dir, cfg.render.fps)
    print("[run]", res.metrics.summary())
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
    d.add_argument("--width", type=int, default=0)
    d.add_argument("--height", type=int, default=0)
    d.add_argument("--no-stage", action="store_true")
    d.add_argument("--out", default="out")
    d.set_defaults(func=cmd_demo)

    r = sub.add_parser("run", help="run on a directory of real frames")
    r.add_argument("--input", required=True)
    r.add_argument("--pattern", default="*")
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

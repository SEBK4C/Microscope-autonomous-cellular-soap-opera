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


def _episode_stats(res):
    m = res.metrics
    return [
        ("recall", f"{m.gt_recall:.0%}" if m.gt_recall is not None else "n/a"),
        ("fps", f"{m.fps:.0f}"),
        ("cast", str(m.n_tracks_total)),
        ("MOTA", f"{m.mota:.2f}" if m.mota is not None else "n/a"),
        ("beats", str(m.caption_count)),
    ]


def _write_outputs(res, out_dir: Path, fps: int, fmt: str = "gif") -> None:
    from .video.io import save_gif, save_mp4, save_png
    from .render.episode_page import save_episode_html
    out_dir.mkdir(parents=True, exist_ok=True)
    media_path, media_mime = None, None
    if res.frames:
        save_png(res.frames[len(res.frames) // 2], out_dir / "frame.png")
        want_mp4 = fmt in ("mp4", "both")
        want_gif = fmt in ("gif", "both")
        if want_mp4:
            try:
                save_mp4(res.frames, out_dir / "episode.mp4", fps=fps)
                media_path, media_mime = out_dir / "episode.mp4", "video/mp4"
            except ImportError:
                print("[out] mp4 needs .[video]; falling back to gif")
                want_gif = True
        if want_gif:
            gif = save_gif(res.frames, out_dir / "episode.gif", fps=fps)
            if media_path is None:
                media_path, media_mime = gif, "image/gif"

    with open(out_dir / "transcript.txt", "w", encoding="utf-8") as fh:
        for ev in res.transcript:
            fh.write(f"[{ev.frame_idx:04d}] ({ev.kind}) " + " / ".join(ev.lines) + "\n")
    with open(out_dir / "stage_commands.jsonl", "w", encoding="utf-8") as fh:
        for c in res.stage_commands:
            fh.write(json.dumps(c) + "\n")
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(res.metrics.as_dict(), fh, indent=2)

    if media_path is not None:
        rows = [(ev.frame_idx, ev.kind, ev.headline) for ev in res.transcript]
        subtitle = (f"{len(res.records)} frames · score {res.metrics.score:.2f} · "
                    f"{res.metrics.fps:.0f} fps")
        save_episode_html(out_dir / "episode.html", media_path.read_bytes(),
                          media_mime, "As the Slide Turns", subtitle,
                          rows, _episode_stats(res))


def cmd_demo(args) -> int:
    cfg = PipelineConfig()
    cfg.n_frames = args.frames
    cfg.world.seed = args.seed
    cfg.drama.seed = args.seed
    cfg.drama.spice = args.spice
    cfg.stage.enabled = not args.no_stage
    cfg.drama.backend = args.narrator
    cfg.drama.llm_model = args.llm_model
    cfg.drama.season_path = args.season
    cfg.world.style = args.style
    cfg.segment.polarity = args.polarity
    # Adaptive local thresholding is the default (robust to gradients/halos and
    # slightly better on the bench); --no-adaptive opts out.
    cfg.segment.adaptive = (args.adaptive if args.adaptive is not None else True)
    if args.width:
        cfg.world.width = args.width
    if args.height:
        cfg.world.height = args.height

    if args.moving:
        from .pipeline import Pipeline
        from .video.synthetic import SyntheticWorld
        cfg.world.world_scale = args.world_scale
        cfg.world.n_start = max(cfg.world.n_start, 14)
        cfg.world.max_microbes = max(cfg.world.max_microbes, 24)
        sw = int(cfg.world.width * args.world_scale)
        sh = int(cfg.world.height * args.world_scale)
        print(f"[demo] moving stage: slide {sw}x{sh}, sensor "
              f"{cfg.world.width}x{cfg.world.height}, {cfg.n_frames} frames …")
        world = SyntheticWorld(cfg.world)
        res = Pipeline(cfg).run_moving(world.frames(cfg.n_frames), collect_frames=True)
    else:
        print(f"[demo] rendering {cfg.n_frames} frames "
              f"({cfg.world.width}x{cfg.world.height}) style={args.style} "
              f"polarity={args.polarity} adaptive={cfg.segment.adaptive} …")
        res = run_synthetic(cfg, collect_frames=True)

    out_dir = Path(args.out)
    _write_outputs(res, out_dir, cfg.render.fps, fmt=args.format)
    print("[demo]", res.metrics.summary())
    if res.moving and res.moving.get("mean_star_offset") is not None:
        print(f"[demo] moving: star_offset={res.moving['mean_star_offset']:.0f}px "
              f"stage_travel={res.moving['stage_travel']:.0f}px")
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
        note = ("FastSAM is near-real-time on CPU (~11 fps @ imgsz 384)"
                if str(args.sam_backend) in ("auto", "fastsam")
                else "MobileSAM/SAM everything-mode on CPU is ~0.03 fps (use a GPU)")
        print(f"[run] NOTE: {note}. Weights auto-fetch from HuggingFace into models/.")
    else:
        cfg.segment.backend = "classical"
        cfg.segment.polarity = args.polarity  # real clips: auto-detect polarity
        cfg.segment.adaptive = args.adaptive
        # De-fragmentation bundle for noisy real footage (AUTORESEARCH_JOURNAL #1).
        cfg.segment.temporal = args.temporal  # motion foreground erases static noise
        cfg.segment.hybrid = args.hybrid      # whole-body gating (best for CLEAN footage)
        cfg.segment.watershed = args.watershed  # split touching round cells
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
    _write_outputs(res, out_dir, cfg.render.fps, fmt=args.format)
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


def cmd_serve(args) -> int:
    from .web.viewer import serve
    cfg = PipelineConfig()
    cfg.n_frames = args.frames
    cfg.render.fps = args.fps
    cfg.drama.backend = args.narrator
    serve(cfg, host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="soapscope",
                                description="Autonomous cellular soap opera.")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="run the synthetic world end to end")
    d.add_argument("--frames", type=int, default=140)
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--spice", type=float, default=1.0)
    d.add_argument("--narrator", choices=["template", "llm", "vlm"], default="template",
                   help="template = instant & offline; llm = small local text model "
                        "(~2.4 s/caption); vlm = BLIP-grounded, mentions appearance "
                        "(~3 s/caption). llm/vlm need .[llm]/.[vlm]")
    d.add_argument("--llm-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    d.add_argument("--season", default="",
                   help="season-memory JSON file: recurring lore, a 'Previously on…' "
                        "recap and a cliffhanger across episodes (run repeatedly)")
    d.add_argument("--style", choices=["darkfield", "brightfield"], default="darkfield",
                   help="darkfield = bright microbes/dark bg; brightfield = dark microbes/light bg")
    d.add_argument("--polarity", choices=["bright", "dark", "auto"], default="auto")
    d.add_argument("--adaptive", action=argparse.BooleanOptionalAction, default=None,
                   help="local adaptive thresholding (auto-on for brightfield)")
    d.add_argument("--width", type=int, default=0)
    d.add_argument("--height", type=int, default=0)
    d.add_argument("--no-stage", action="store_true")
    d.add_argument("--format", choices=["gif", "mp4", "both"], default="gif",
                   help="episode media: gif (no deps), mp4 (needs .[video]), or both")
    d.add_argument("--moving", action="store_true",
                   help="moving-crop CNC stage: pan a sensor across a larger slide")
    d.add_argument("--world-scale", type=float, default=1.8,
                   help="slide size / sensor size for --moving")
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
    r.add_argument("--watershed", action=argparse.BooleanOptionalAction, default=False,
                   help="split touching round cells (slower; over-splits elongated microbes)")
    r.add_argument("--min-area", type=int, default=45, help="min detection area (px)")
    r.add_argument("--stride", type=int, default=1, help="keep every Nth video frame")
    r.add_argument("--max-frames", type=int, default=240)
    r.add_argument("--max-width", type=int, default=640, help="downscale wide footage")
    r.add_argument("--format", choices=["gif", "mp4", "both"], default="gif",
                   help="episode media: gif (no deps), mp4 (needs .[video]), or both")
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

    sv = sub.add_parser("serve", help="live web viewer — watch an episode stream")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--fps", type=int, default=12)
    sv.add_argument("--frames", type=int, default=240)
    sv.add_argument("--narrator", choices=["template", "llm", "vlm"], default="template")
    sv.set_defaults(func=cmd_serve)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

# Data — prototyping from real microscopy video

SoapScope prototypes the whole stack from **static video**, so no lab hardware
is needed to develop it. The synthetic world (`soapscope/video/synthetic.py`)
is the default benchmark; this folder is where real clips go.

> Videos are **git-ignored** (`data/videos/`, `*.mp4`) to keep the repo light.
> Download locally; commit only tiny derived samples if truly needed.

## Ingesting a clip today

Export the clip to an ordered folder of frames, then run the pipeline on it:

```bash
# with ffmpeg (if installed):
ffmpeg -i data/videos/pond.mp4 -vf fps=12 data/frames/pond/%05d.png

python -m soapscope.cli run --input data/frames/pond/
# → out/episode.gif, transcript.txt, stage_commands.jsonl, metrics.json
```

`soapscope run` reads a directory of PNG/JPG frames via
`soapscope.video.io.load_frames_dir`, which is the same source interface the
synthetic world satisfies — the rest of the pipeline is identical.

## Candidate public datasets (for the loop to pull from)

Backlog item #1 wires up an automated fetcher. Good starting points:

- **Kaggle** — search "microorganism", "microbes", "plankton", "cell tracking".
  The *WHOI-Plankton* and various *pond microorganism* video/image sets are apt.
- **Hugging Face** — `datasets` tagged microscopy / cell-tracking / bio-imaging.
- **Cell Tracking Challenge** (celltrackingchallenge.net) — canonical annotated
  microscopy time-lapses (great for scoring tracking against real GT).
- **GitHub** — motility/tracking repos often ship short sample microbe clips.

## Notes for real footage (known gotchas)

- **Contrast polarity.** Our classical segmenter assumes microbes are *brighter*
  than background (dark-field). Bright-field microbes are often *darker* —
  backlog item #1 adds a polarity/adaptive-threshold option.
- **Frame rate.** Downsample to ~10–15 fps; the drama beats read better and it
  keeps things near-real-time.
- **Resolution.** Very large frames slow the pure-numpy connected-components;
  downscale to ≤ ~720p for the prototype (SAM3 backend will lift this later).

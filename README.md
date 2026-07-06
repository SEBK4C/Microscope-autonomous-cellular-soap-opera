# 🔬 SoapScope — the Autonomous Cellular Soap Opera

> Point a microscope at some pond water. Let AI follow the microbes around and
> narrate their lives like a daytime soap / a Gary Larson cartoon — while a CNC
> stage keeps today's *star* centred in frame.

![SoapScope demo](docs/demo.gif)

*A synthetic episode of **As the Slide Turns**: tracked microbes get names,
archetypes and feuds; the white box is the CNC stage's field of view following
the ★ star; the bar narrates the drama.*

SoapScope is a full, **local, CPU-only** prototype of that pipeline. It runs
today with nothing but `numpy` + `Pillow`, and every heavy part (SAM3
segmentation, a local LLM narrator, a real CNC stage) slots in later behind a
stable interface. It's built and improved by a **Karpathy-style autoresearch
loop** — see [`AUTORESEARCH_JOURNAL.md`](AUTORESEARCH_JOURNAL.md).

## Why it works without a lab (yet)

The whole software stack is prototyped from **static video** (or a built-in
synthetic microbe world with ground truth). Swap the video source for a real
clip from HuggingFace/Kaggle/GitHub, or the simulated stage for a
microcontroller — nothing else changes.

```
video → segment → track → features → drama → stage → render
```

| Stage | Today (runs anywhere) | Drops in later |
|-------|-----------------------|----------------|
| **Video** | synthetic world + ground truth; PNG-frame folders | real microscopy clips, webcam |
| **Segment** | classical numpy connected-components | **SAM2 / SAM3** |
| **Track** | greedy nearest-neighbour, stable IDs | Kalman + Hungarian, SAM3 video propagation |
| **Drama** | offline template narrator with feud/romance memory | local **LLM / VLM** narrator |
| **Stage** | `SimulatedStage` + JSON-lines protocol | real **CNC over serial** |
| **Render** | annotated GIF + transcript | mp4, live web viewer |

## Quickstart

```bash
pip install -e .            # core deps: numpy + Pillow only
# or: pip install -r requirements.txt

python -m soapscope.cli demo
# → out/episode.gif, out/transcript.txt, out/stage_commands.jsonl, out/metrics.json
```

Sample narration (`out/transcript.txt`):

```
[0012] (ENCOUNTER) It's the confrontation we were promised: Madame Dmitri
       Euglenova vs Contessa Genevieve Micrococcus, no notes.
[0024] (CHASE) Count Dmitri Pseudopod pursues Duchess Ophelia Diatomsky
       across the slide. Is it love? Is it lunch? Yes.
[0066] (FLEE) Madame Dmitri Euglenova FLEES from Count Vesper Micrococcus!
       The betrayal! The velocity!
```

Each frame also emits a stage command a microcontroller can consume
(`out/stage_commands.jsonl`):

```json
{"t": 4, "cmd": "move_abs", "x": 67.39, "y": 40.21}
```

## CLI

```bash
soapscope demo --frames 140 --seed 7             # synthetic dark-field episode
soapscope demo --style brightfield               # dark microbes on a light field
soapscope run --input clip.webm --max-frames 80  # a real video file (mp4/webm/ogv/…)
soapscope run --input path/to/frames/            # …or a directory of PNG frames
soapscope experiment --rounds 12                 # autoresearch: hill-climb the config
soapscope sweep --param segment.threshold --values 0.2,0.3,0.4
soapscope protocol                               # print the CNC wire protocol
```

## Real microscopy video

![Bright-field episode](docs/brightfield.gif)

*Bright-field mode: dark microbes with phase halos on a light field — the
common real-microscopy look, auto-detected and segmented.*

Grab a small public-domain clip and narrate it in two commands:

```bash
pip install -e .[video]                 # imageio + a bundled ffmpeg (no system ffmpeg needed)
python scripts/fetch_sample_video.py    # → data/videos/<clip> (Wikimedia Commons, public domain)
python -m soapscope.cli run --input data/videos/Swift_ciliate.webm --max-frames 80
```

The segmenter **auto-detects contrast polarity** (bright microbes on dark
background vs dark microbes on a light one) and uses adaptive local thresholding
so vignetting and phase halos don't fool it — no per-clip tuning needed to get a
watchable first cut.

## The CNC stage as an API

The pipeline only ever talks to the abstract `CNCStage` interface. A real
microcontroller (Arduino / RP2040 / ESP32) speaks newline-delimited JSON over
USB serial — see [`hardware/`](hardware/) for the protocol and a firmware stub.
`SimulatedStage` implements the same interface so the entire control loop is
testable from static video.

## Autoresearch

Following [karpathy/autoresearch](https://github.com/karpathy/autoresearch):
propose a tweak, run the fixed benchmark, measure a single `score`, keep it only
if it improved, repeat. Here the "training script" is the whole
`PipelineConfig`, and the metric blends tracking accuracy, narration variety and
speed. A cron loop drives it every 15 minutes; results land in the journal.

## Status

End-to-end and running on **real internet-sourced microscopy video**, not just
the synthetic world. Baseline score ≈ 0.90 on the synthetic benchmark (94%
recall, ~17 fps CPU); auto polarity + adaptive thresholding reaches ≈ 0.96 on
bright-field. Real clips are watchable out of the box but still fragment into
short tracks — de-fragmenting them is the current focus. Roadmap and experiment
log: [`AUTORESEARCH_JOURNAL.md`](AUTORESEARCH_JOURNAL.md).

## License

MIT — see [LICENSE](LICENSE).

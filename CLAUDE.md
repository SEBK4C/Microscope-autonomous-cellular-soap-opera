# CLAUDE.md — SoapScope project guide & autoresearch loop protocol

**SoapScope** turns a microscope slide into a soap opera: segment + track the
microbes drifting across the field of view, then auto-caption their antics like
a Gary Larson cartoon / daytime soap, while a CNC stage keeps today's "star"
centred. The whole stack is prototyped from **static video** (and a synthetic
world) so it runs on a laptop with **no GPU**, then swaps in SAM3, a local LLM,
and real CNC hardware behind stable interfaces.

This repo is developed by an **autoresearch loop** (Karpathy-style): every 15
minutes an agent continues the work. **You are that agent.** Do not re-scaffold
— continue.

## The loop protocol (do this every iteration)

1. **Read** `AUTORESEARCH_JOURNAL.md` (state + backlog) and skim recent `git log`.
2. **Pick** the single highest-value item from *Backlog / Next Experiments*.
3. **Implement** it in a small, self-contained change. Keep the core runnable
   with only `numpy` + `Pillow` (heavy backends stay optional extras).
4. **Verify** it: `python -m pytest -q` and, for pipeline changes,
   `python -m soapscope.cli demo --frames 60` (inspect `out/` and metrics).
   For tuning, use `python -m soapscope.cli experiment`.
5. **Log** a dated entry in `AUTORESEARCH_JOURNAL.md`: what you changed, the
   metric before/after, what worked, what didn't, and update the backlog.
6. **Commit & push** to `claude/microbe-soap-opera-sam3-dixhwb`
   (`git push -u origin claude/microbe-soap-opera-sam3-dixhwb`, retry with
   backoff on network errors). Do **not** open a PR unless asked.

Keep each iteration green: never commit a broken `demo` or failing tests.

## Architecture (data flows left to right)

```
video → vision.segment → vision.track → vision.features → drama → stage → render
 (source)   (masks)        (identities)     (beats)      (captions)(CNC)  (overlay)
```

| Module | Role | Swap-in later |
|--------|------|---------------|
| `soapscope/video/synthetic.py` | deterministic dark/bright-field world + ground truth | real clips via `video/io.load_video` / `load_frames_dir` |
| `soapscope/vision/segment.py` | `ClassicalSegmenter` (numpy CC; polarity/adaptive/temporal/hybrid/watershed) + `SamSegmenter` (FastSAM/MobileSAM/SAM via `sam_backend.py`) | SAM2 video propagation; GPU |
| `soapscope/vision/track.py` | Kalman + Hungarian tracker (`assign.py`), stable ids, enter/exit | SAM3 video propagation |
| `soapscope/vision/features.py` | trajectories → dramatic *beats* | richer interaction model |
| `soapscope/drama/characters.py` | persistent names + soap archetypes | — |
| `soapscope/drama/captioner.py` | `TemplateCaptioner` (instant) + `LLMCaptioner` (local LLM) + `VLMCaptioner` (BLIP-grounded, `vlm_backend.py`) | bigger VLM / two-stage |
| `soapscope/drama/season.py` | `SeasonMemory` + `Showrunner`: cross-episode lore, "Previously on…" recap, cliffhanger | multi-season arcs |
| `soapscope/stage/api.py` | `CNCStage` interface, `SimulatedStage`, JSON-lines protocol | `SerialStage` → real microcontroller |
| `soapscope/stage/controller.py` | "director" star pick + rate-limited follow; `moving.py` pans a sensor across a larger slide (world-coord tracking) | closed-loop focus/zoom |
| `soapscope/render/overlay.py` + `episode_page.py` | masks/names/trails/viewport/caption bar; mp4 + a self-contained HTML episode page | live web viewer |
| `soapscope/pipeline.py` | wires it all together | — |
| `soapscope/metrics.py` | autoresearch `score` + MOTA / ID-switch (GT) | funniness metrics |
| `soapscope/autoresearch/loop.py` | hill-climb over `PipelineConfig` | smarter proposers |

Every tunable lives in `soapscope/config.py` (`PipelineConfig`), so an
experiment is just "make a config, run it, read `metrics.score`".

## The objective (`metrics.score`, higher = better)

`0.30·recall + 0.25·track_stability + 0.20·caption_variety + 0.15·speed + 0.10·(1−|frag−1|)`

- **recall** — fraction of ground-truth microbes covered by a track (synthetic only)
- **track_stability** — mean track length / (½·frames)
- **caption_variety** — unique headlines / total
- **speed** — fps / 30 (near-real-time target)
- **frag** — tracks / GT-ids (1.0 ideal; penalise over/under-segmentation)

Baseline today is ~**0.90**. Improvements must beat the current best on the
fixed synthetic benchmark (seed-locked) to be kept.

## Conventions

- **No mandatory GPU / large downloads.** Core = numpy + Pillow. Anything heavy
  (`torch`, `llama-cpp`, `opencv`, `imageio`, `pyserial`) is an optional extra
  and must degrade gracefully (fall back, or raise a clear "install X" error).
- **Interfaces first.** New backends implement the existing base classes
  (`Segmenter`, `Captioner`, `CNCStage`) so the pipeline never changes.
- **Determinism.** Seed everything; the benchmark must be reproducible.
- **Model weights: fetch from HuggingFace.** The agent proxy allows pip (pypi)
  and HF downloads but **blocks GitHub release assets (403)** — so ultralytics'
  auto-download fails; pull weights from HF into `models/` (git-ignored).
  Install CPU torch from `https://download.pytorch.org/whl/cpu`.
- Keep comments about *constraints*, not narration.

## Quickstart

```bash
pip install -e .                       # or: pip install -r requirements.txt
python -m soapscope.cli demo           # → out/episode.gif + transcript + metrics
python -m soapscope.cli experiment --rounds 12 --journal AUTORESEARCH_JOURNAL.md
python -m soapscope.cli protocol       # CNC microcontroller wire protocol
python -m pytest -q
```

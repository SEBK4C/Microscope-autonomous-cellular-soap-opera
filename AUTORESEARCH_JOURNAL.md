# Autoresearch Journal — SoapScope

A running lab notebook for the autonomous research loop (see `CLAUDE.md` for the
per-iteration protocol). Newest entries at the bottom. Each iteration: pick one
backlog item, implement, verify, record the metric delta here, update the
backlog, commit & push.

**Fixed benchmark:** `PipelineConfig()` defaults on the synthetic world,
seed-locked. Headline metric is `metrics.score` (definition in `CLAUDE.md`).
An experiment is *kept* only if it beats the current best.

---

## Backlog / Next Experiments  (prioritised — top item first)

Pull the **top** item each loop. Re-order as you learn. Mark done by moving a
line into a dated entry below.

1. **De-fragment real footage.** Real clips segment into many short tracks (the
   ciliate clip: 77 tracks / 60 frames, mean len 10) because compression
   noise/texture trips the adaptive threshold and the greedy tracker drops IDs.
   Try: temporal denoise (rolling-median background), morphological opening,
   `min_area` that scales with frame size, and a larger gate / Kalman predict.
   *Metric:* mean_track_len ↑ and track count ↓ on the real clip; keep the
   synthetic score ≥ 0.90.
2. **SAM2/SAM3 segmentation backend.** Implement `SamSegmenter.segment` behind
   the `[sam]` extra: automatic-mask-generation on the first frame, then video
   propagation for tracking. Fall back to classical if `torch`/checkpoint
   absent. *Metric:* recall & fragmentation vs classical; fps budget.
3. **Local-LLM narrator.** Implement `LLMCaptioner` with llama.cpp/Ollama, a
   soap-opera system prompt fed the frame's beats + character memory (optionally
   cropped microbe thumbnails via a small VLM). Keep template as fallback.
   *Metric:* caption_variety + a human spot-check; must stay near-real-time.
4. **Kalman + Hungarian tracking.** Constant-velocity Kalman predict + optimal
   assignment; add a proper MOTA / ID-switch metric using GT. (Pairs with #1.)
5. **Touching-microbe segmentation.** Distance-transform + watershed split so
   two collided microbes don't merge into one track. *Metric:* fragmentation.
6. **True moving-crop stage.** Make the world larger than the sensor so the CNC
   actually pans across a slide and microbes leave/enter the sensor; add
   stage-motion-compensated tracking. *Metric:* star stays in-frame % ; recall.
7. **Season memory.** Persist character bios + relationships to disk across
   episodes; "Previously, on…" recaps and end-of-episode cliffhangers.
8. **Real video output.** mp4 via `imageio-ffmpeg`; optional live web viewer
   that streams annotated frames + captions (near-real-time from webcam).
9. **Adaptive by default?** auto+adaptive beat the dark-field default on the
   synthetic bench (0.96 vs 0.93); consider making adaptive the default once
   validated on more real clips. *Metric:* synthetic score; real-clip frag.
10. **TTS narrator** (optional): speak the caption bar with an announcer voice.

---

## 2026-07-06 — Iteration 0: end-to-end stack from scratch

**Goal:** stand up the *entire* pipeline so every later iteration is an
incremental, measurable improvement rather than scaffolding.

**Built (all runnable on CPU with only numpy + Pillow):**

- **Synthetic microscope world** (`video/synthetic.py`) — a flow cell of
  drifting microbes with Brownian motion, a global current, mitosis, predator/
  prey chases, and edge arrivals/exits. Emits per-frame **ground truth** so
  tracking can be scored objectively. Deterministic (seeded).
- **Classical segmenter** (`vision/segment.py`) — illumination-normalised
  threshold + pure-numpy 8-connected component labelling (iterates only over
  foreground pixels, so it's fast on sparse fields). `SamSegmenter` stub behind
  the same interface for later.
- **Tracker** (`vision/track.py`) — greedy nearest-neighbour matching with a
  distance gate + constant-velocity coasting; stable IDs; clean `entered`
  (newly-confirmed) / `exited` events so arrivals/exits aren't triggered by
  one-frame noise.
- **Behaviour → beats** (`vision/features.py`) — ENTER / EXIT / DIVIDE / CHASE /
  FLEE / ENCOUNTER / SPEED_BURST / LINGER / WANDER, each with a drama `score`.
- **Drama layer** (`drama/`) — persistent characters (title + name + soap
  archetype) and a `TemplateCaptioner` that narrates the top beat with a memory
  of feuds/romances, so repeated encounters escalate into a running rivalry.
- **CNC stage** (`stage/`) — abstract `CNCStage`, a `SimulatedStage`, a
  `SerialStage` stub, and a documented **newline-JSON microcontroller protocol**
  (`soapscope protocol`). A `StageController` "director" scores tracks by live
  drama and keeps the star centred with hysteresis + a feed-rate cap.
- **Renderer** (`render/overlay.py`) — masks, character tags, motion trails, the
  CNC viewport + reticle, top banner, and a caption bar → annotated GIF.
- **Metrics + autoresearch** (`metrics.py`, `autoresearch/loop.py`) — a single
  `score` and a Karpathy-style hill-climb (`soapscope experiment`) that perturbs
  one `PipelineConfig` knob at a time and keeps improvements.
- **CLI** (`demo` / `run` / `experiment` / `sweep` / `protocol`) and a **13-test
  suite** (all green).

**Result (fixed benchmark, seed 7):**

```
score≈0.90 | fps≈16.8 | recall=0.94 | frag≈1.1 | tracks≈17 meanlen≈46
captions=15 variety=1.00 kinds=5
```

The narration is genuinely legible as a soap opera — characters keep their
names across the episode and rivalries build (e.g. "The rivalry deepens —
round 3"). Demo: `docs/demo.gif`.

**What worked:** the interface-first design — the pipeline is fully decoupled
from backends, so SAM3 / a local LLM / real hardware are drop-ins. Scoring
against synthetic ground truth makes the autoresearch loop meaningful on day 0.

**What's shaky / next:** classical segmentation assumes microbes are *brighter*
than background (true for our dark-field synthetic, often false for real
bright-field clips) → item #1 needs a polarity/adaptive option. Tracker
fragments slightly during collisions (frag > 1) → items #5/#6.

**Next:** backlog item #1 — real-video ingestion + a fetch script, and make the
segmenter polarity-aware so a real clip produces a watchable episode.

---

## 2026-07-06 — Iteration 1: real-video ingestion + polarity-aware segmentation

**Picked:** backlog #1 (real-video ingestion). Its true blocker was that the
classical segmenter assumed microbes are *brighter* than background — false for
real **bright-field** microscopy, where they're darker (with phase halos).

**Built & verified:**

- **Polarity-aware segmentation** (`vision/segment.py`) — `polarity` ∈
  {`bright`, `dark`, `auto`} + an `adaptive` local-mean thresholding mode
  (`SegmentConfig`). The default (bright, non-adaptive) path is byte-for-byte
  unchanged, so no regression. `auto` decides polarity once per clip from the
  **sign of the net deviation from a local-mean background** — robust to
  vignetting/halos (my first percentile-based heuristic mis-fired on halos;
  the residual-mass version is stable). Cached per video to avoid flicker.
- **Bright-field synthetic style** (`WorldConfig.style="brightfield"`) — a
  deterministic, ground-truth testbed for the dark-polarity path: dark bodies +
  bright phase halos on a light, vignetted field.
- **Video-file ingestion** — `video.io.load_video` (lazy `imageio`, with
  stride / max-frames / max-width) and `soapscope run` now accepts a **video
  file or a frames directory**, with `--polarity/--adaptive/--stride/…` flags.
- **Scripts** — `scripts/fetch_sample_video.py` (curated public-domain Wikimedia
  Commons microbe clips) and `scripts/frames_from_video.py`. `data/README.md`
  updated. New tests (polarity detection, brightfield recall, `load_video`
  transforms via a monkeypatched decoder). **19/19 tests green.**

**Results (fixed synthetic bench, seed 7, n≈45–60):**

| world | polarity | adaptive | recall | score |
|-------|----------|----------|--------|-------|
| dark-field | bright (default) | no | 0.93 | 0.93 |
| dark-field | **auto** | no | 0.93 | 0.94 |
| bright-field | bright (wrong) | no | **0.10** | 0.40 |
| bright-field | **auto** | **yes** | 0.95 | **0.96** |

`auto` resolves correctly on both worlds. Wrong-polarity collapse (0.10 recall)
confirms the fix is load-bearing. Bright-field demo: `docs/brightfield.gif`.

**Real footage — end to end.** Installed the `[video]` extra, fetched a 1.09 MB
public-domain ciliate clip from Wikimedia Commons, and ran the full pipeline:
it decoded the webm, **auto-detected `dark` polarity correctly**, tracked,
named, followed a star with the CNC viewport, and narrated — a genuinely
watchable episode from an internet clip with zero manual tuning. 🎉

**What's shaky / next:** real footage fragments badly — **77 tracks over 60
frames, mean length 10** (vs ~12 stable tracks on synthetic). Compression
noise/texture trips the adaptive threshold and the greedy tracker drops IDs
during fast ciliate motion. That's the new **backlog #1** (de-fragment real
footage: temporal denoise + morphology + size-scaled `min_area` + a
larger/Kalman tracker). Note: auto+adaptive also *beat* the dark-field default
on synthetic (0.96 vs 0.93) → possible new default (backlog #9).

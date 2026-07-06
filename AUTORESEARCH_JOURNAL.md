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

1. **Real-video ingestion.** Add a `FileVideoSource` (mp4 → frames via optional
   `imageio`/`opencv`) and `scripts/fetch_sample_video.py` to pull a small
   public microbe clip (HuggingFace/Kaggle/GitHub — candidates in
   `data/README.md`). Run `soapscope run` on it. Expect the classical segmenter
   to need a **polarity option** (bright-field microbes are often *darker* than
   background) and per-clip threshold tuning. *Metric:* qualitative + fps.
2. **SAM2/SAM3 segmentation backend.** Implement `SamSegmenter.segment` behind
   the `[sam]` extra: automatic-mask-generation on the first frame, then video
   propagation for tracking. Fall back to classical if `torch`/checkpoint
   absent. *Metric:* recall & fragmentation vs classical; fps budget.
3. **Local-LLM narrator.** Implement `LLMCaptioner` with llama.cpp/Ollama, a
   soap-opera system prompt fed the frame's beats + character memory (optionally
   cropped microbe thumbnails via a small VLM). Keep template as fallback.
   *Metric:* caption_variety + a human spot-check; must stay near-real-time.
4. **True moving-crop stage.** Make the world larger than the sensor so the CNC
   actually pans across a slide and microbes leave/enter the sensor; add
   stage-motion-compensated tracking. *Metric:* star stays in-frame % ; recall.
5. **Better tracking.** Kalman predict + Hungarian assignment; measure
   fragmentation drop and ID-switch count (add a proper MOTA metric using GT).
6. **Touching-microbe segmentation.** Distance-transform + watershed split so
   two collided microbes don't merge into one track. *Metric:* fragmentation.
7. **Season memory.** Persist character bios + relationships to disk across
   episodes; "Previously, on…" recaps and end-of-episode cliffhangers.
8. **Real video output.** mp4 via `imageio-ffmpeg`; optional live web viewer
   that streams annotated frames + captions (near-real-time from webcam).
9. **TTS narrator** (optional): speak the caption bar with an announcer voice.

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

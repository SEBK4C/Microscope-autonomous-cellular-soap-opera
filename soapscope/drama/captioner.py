"""Turn beats into soap-opera narration.

``TemplateCaptioner`` is offline, deterministic and genuinely funny-ish: it
keeps a memory of who is feuding and who is flirting, so a CHASE between the
same two microbes escalates over time into a proper rivalry. An ``LLMCaptioner``
(local Llama/VLM) implements the same interface and swaps in later for
free-form wit — the pipeline doesn't care which is running.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..config import DramaConfig
from ..vision.features import Beat, FrameFeatures
from .characters import Character, CharacterRegistry


@dataclass
class CaptionEvent:
    frame_idx: int
    headline: str                    # the punchy caption-bar line
    lines: List[str]                 # headline + any narrator flourish
    kind: str                        # beat kind that triggered it
    subjects: List[int] = field(default_factory=list)


# Announcer flourishes sprinkled in for that daytime-TV texture.
STINGERS = [
    "MEANWHILE, under the coverslip…",
    "SUDDENLY—",
    "Little did they know…",
    "But the current had other plans.",
    "Somewhere, a flagellum quivered.",
    "The plot, much like the pond water, thickens.",
    "Previously, on As the Slide Turns…",
    "Cue dramatic organelle music.",
    "And SCENE.",
]

TEMPLATES: Dict[str, List[str]] = {
    "ENTER": [
        "A stranger drifts into frame: {A}, {arch_a}, arrives on the current.",
        "The doors of the flow cell part — enter {A}, looking {trait_a}.",
        "New in town: {A} washes ashore and immediately judges everyone.",
        "{A} makes an entrance nobody asked for but everybody will remember.",
    ],
    "EXIT": [
        "{A} floats off into the great vignette. Written off the show, for now.",
        "Farewell, {A}. Gone, but never forgotten (until next frame).",
        "{A} exits stage left, chased only by regret and a faint current.",
        "And just like that, {A} is out of focus — literally and emotionally.",
    ],
    "DIVIDE": [
        "SCANDAL: {A} has split into two! {B} is born, and the resemblance is uncanny.",
        "{A} undergoes mitosis live on air. It's a girl! And also… another {a_short}.",
        "Against all advice, {A} divides. {B} inherits the drama and the debts.",
        "BREAKING: {A} becomes two. The custody battle over the nucleus begins.",
    ],
    "CHASE": [
        "{A} pursues {B} across the slide. Is it love? Is it lunch? Yes.",
        "The chase is ON: {A}, {arch_a}, will NOT let {b_short} get away.",
        "{A} closes in on {B}. Somewhere a diatom gasps.",
        "{A} tails {B} with the intensity of unpaid rent.",
    ],
    "FLEE": [
        "{A} FLEES from {B}! The betrayal! The velocity!",
        "{A} bolts, leaving {B} and a trail of unresolved tension.",
        "Not today: {A} peaces out on {B} at record speed.",
        "{A} runs. {B} pretends not to care. {B} cares.",
    ],
    "ENCOUNTER": [
        "{A} and {B} collide. Sparks? Cytoplasm? Both are exchanged.",
        "Face to face at last: {A} meets {B}, and neither will apologise.",
        "{A} bumps into {B}. The membranes touch. The tabloids rejoice.",
        "It's the confrontation we were promised: {A} vs {B}, no notes.",
    ],
    "SPEED_BURST": [
        "{A} storms off in a huff at {speed:.1f} px/frame. ICONIC.",
        "{A} suddenly accelerates — someone must have mentioned the ex.",
        "{A} makes a scene and a wake. Everyone is talking.",
        "Zoom! {A} exits the conversation the only way {pron_a} knows how.",
    ],
    "LINGER": [
        "{A} broods in the corner, motionless, misunderstood.",
        "{A} has not moved in {frames} frames. This is a cry for help.",
        "Still waters: {A} sulks photogenically while the plot waits.",
        "{A} contemplates the meaning of it all. And lunch.",
    ],
    "WANDER": [
        "{A} drifts, {trait_a} as ever, plotting nothing in particular.",
        "A quiet moment as {A} wanders the pond of broken dreams.",
        "{A} mills about. Even microbes have Mondays.",
        "Nothing happens, beautifully, starring {A}.",
    ],
}

# Escalation lines when a rivalry/romance has history.
RIVALRY_TAGS = [
    "Their feud enters its {n}th act.",
    "That's {n} confrontations now. Someone start counting.",
    "The rivalry deepens — round {n}.",
]
ROMANCE_TAGS = [
    "Will-they-won't-they tension: now at {n}.",
    "The shippers are FED, encounter number {n}.",
    "Chemistry rating: {n}/10 and climbing.",
]


class Captioner:
    """Interface shared by every captioning backend."""

    def update(self, frame_idx: int, feats: FrameFeatures,
               reg: CharacterRegistry) -> CaptionEvent:  # pragma: no cover
        raise NotImplementedError


class TemplateCaptioner(Captioner):
    def __init__(self, cfg: DramaConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.current: Optional[CaptionEvent] = None
        self.transcript: List[CaptionEvent] = []
        self.rivalry: Dict[Tuple[int, int], int] = {}
        self.romance: Dict[Tuple[int, int], int] = {}
        self._title_shown = False

    # -------------------------------------------------------------- helpers
    def _fields(self, beat: Beat, reg: CharacterRegistry) -> dict:
        chars = [reg.get(s) for s in beat.subjects]
        f: dict = {}
        if chars:
            a = chars[0]
            f.update(A=a.name, a_short=a.short, arch_a=a.archetype,
                     trait_a=a.trait, pron_a=a.pronoun)
        if len(chars) > 1:
            b = chars[1]
            f.update(B=b.name, b_short=b.short, arch_b=b.archetype,
                     trait_b=b.trait, pron_b=b.pronoun)
        f.update(beat.data)
        return f

    def _pair(self, a: int, b: int) -> Tuple[int, int]:
        return (a, b) if a <= b else (b, a)

    def _relationship_tag(self, beat: Beat) -> Optional[str]:
        if len(beat.subjects) < 2:
            return None
        key = self._pair(beat.subjects[0], beat.subjects[1])
        if beat.kind in ("CHASE", "FLEE", "ENCOUNTER"):
            if beat.kind == "ENCOUNTER" and self.romance.get(key, 0) >= self.rivalry.get(key, 0):
                self.romance[key] = self.romance.get(key, 0) + 1
                n = self.romance[key]
                if n >= 2:
                    return self.rng.choice(ROMANCE_TAGS).format(n=n)
            else:
                self.rivalry[key] = self.rivalry.get(key, 0) + 1
                n = self.rivalry[key]
                if n >= 2:
                    return self.rng.choice(RIVALRY_TAGS).format(n=n)
        return None

    def _render_beat(self, beat: Beat, reg: CharacterRegistry) -> List[str]:
        fields = self._fields(beat, reg)
        template = self.rng.choice(TEMPLATES.get(beat.kind, TEMPLATES["WANDER"]))
        try:
            headline = template.format(**fields)
        except (KeyError, IndexError):
            headline = template  # be robust to any missing field
        lines = [headline]
        tag = self._relationship_tag(beat)
        if tag:
            lines.append(tag)
        elif self.cfg.spice >= 1.0 and self.rng.random() < 0.35 * self.cfg.spice:
            lines.append(self.rng.choice(STINGERS))
        return lines

    def _is_tick(self, frame_idx: int) -> bool:
        return self.current is None or frame_idx % max(1, self.cfg.caption_every) == 0

    # ---------------------------------------------------------------- update
    def update(self, frame_idx: int, feats: FrameFeatures,
               reg: CharacterRegistry) -> CaptionEvent:
        if not self._is_tick(frame_idx):
            # Hold the current caption between ticks; keep its frame stamp fresh.
            return self.current  # type: ignore[return-value]

        beat = feats.top()
        if beat is None:
            headline = "The pond is calm. Suspiciously calm."
            ev = CaptionEvent(frame_idx=frame_idx, headline=headline,
                              lines=[headline], kind="IDLE", subjects=[])
        else:
            lines = self._render_beat(beat, reg)
            if not self._title_shown:
                lines = ["🎬 AS THE SLIDE TURNS — today's episode:"] + lines
                self._title_shown = True
            ev = CaptionEvent(frame_idx=frame_idx, headline=lines[0],
                              lines=lines, kind=beat.kind, subjects=beat.subjects)
        self.current = ev
        self.transcript.append(ev)
        return ev


LLM_SYSTEM = (
    "You are the narrator of 'As the Slide Turns', a melodramatic daytime soap "
    "opera about microbes on a microscope slide. Given a scene, write ONE short, "
    "funny caption (max 18 words) in the style of a Gary Larson cartoon: witty, "
    "deadpan, a little absurd. Output only the caption — no quotes, no preamble, "
    "no emoji, no stage directions."
)

SITUATIONS = {
    "ENTER": "{A} ({arch_a}) drifts into view",
    "EXIT": "{A} floats out of frame and is gone",
    "DIVIDE": "{A} just split in two — {B} was born from it",
    "CHASE": "{A} ({arch_a}) is chasing {B} across the slide",
    "FLEE": "{A} is fleeing from {B} at speed",
    "ENCOUNTER": "{A} and {B} collide face to face",
    "SPEED_BURST": "{A} suddenly races off in a hurry",
    "LINGER": "{A} sits perfectly still, brooding",
    "WANDER": "{A} ({arch_a}) drifts aimlessly",
}


class LLMCaptioner(Captioner):
    """Narrate with a small local LLM, falling back to templates if unavailable.

    The heavy model is loaded lazily on the first caption tick; any failure
    (missing transformers, download error, generation error) silently drops to
    the template captioner, so the pipeline never breaks. It keeps a short
    rolling memory of recent lines + character run-ins for story continuity.
    """

    def __init__(self, cfg: DramaConfig):
        self.cfg = cfg
        self._fallback = TemplateCaptioner(cfg)      # safety net + template flavour
        self.transcript: List[CaptionEvent] = []
        self.current: Optional[CaptionEvent] = None
        self._llm = None
        self._llm_tried = False
        self._title_shown = False
        self._recent: List[str] = []
        self._history: Dict[Tuple[int, int], int] = {}
        self.used_llm = 0
        self.used_fallback = 0

    # ------------------------------------------------------------ prompting
    def _situation(self, beat: Beat, reg: CharacterRegistry) -> str:
        fields = self._fallback._fields(beat, reg)
        template = SITUATIONS.get(beat.kind, SITUATIONS["WANDER"])
        try:
            return template.format(**fields)
        except (KeyError, IndexError):
            return "something stirs on the slide"

    def _user_prompt(self, beat: Beat, reg: CharacterRegistry) -> str:
        parts = [f"Scene: {self._situation(beat, reg)}."]
        if len(beat.subjects) >= 2:
            key = self._fallback._pair(beat.subjects[0], beat.subjects[1])
            n = self._history.get(key, 0)
            if n >= 2:
                parts.append(f"(They have history — this is run-in #{n}.)")
        if self._recent:
            parts.append(f"Previously: {self._recent[-1]}")
        parts.append("Caption:")
        return " ".join(parts)

    @staticmethod
    def _clean(text: str) -> str:
        line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
        for pre in ("Caption:", "CAPTION:", "caption:"):
            if line.startswith(pre):
                line = line[len(pre):].strip()
        line = line.split("#")[0].strip()          # drop hashtag spam the small model loves
        line = line.strip().strip('"').strip("'").strip()[:160]
        # If the tail was cut off mid-sentence, trim back to the last full one.
        if line and line[-1] not in ".!?":
            ends = [i for i, c in enumerate(line) if c in ".!?"]
            if ends and ends[-1] >= len(line) * 0.5:
                line = line[:ends[-1] + 1]
        return line

    def _is_tick(self, frame_idx: int) -> bool:
        # Uses THIS captioner's own state (not the fallback's, which never updates).
        return self.current is None or frame_idx % max(1, self.cfg.caption_every) == 0

    def _ensure_llm(self):
        if not self._llm_tried:
            self._llm_tried = True
            try:
                from .llm_backend import load_llm
                self._llm = load_llm(self.cfg)
            except Exception:
                self._llm = None
        return self._llm

    # --------------------------------------------------------------- update
    def update(self, frame_idx: int, feats: FrameFeatures,
               reg: CharacterRegistry) -> CaptionEvent:
        if not self._is_tick(frame_idx):
            return self.current  # type: ignore[return-value]

        beat = feats.top()
        if beat is not None and len(beat.subjects) >= 2:
            key = self._fallback._pair(beat.subjects[0], beat.subjects[1])
            self._history[key] = self._history.get(key, 0) + 1

        line = None
        llm = self._ensure_llm()
        if llm is not None and beat is not None:
            try:
                line = self._clean(llm.generate(LLM_SYSTEM, self._user_prompt(beat, reg)))
            except Exception:
                line = None

        if line:
            self.used_llm += 1
        else:
            self.used_fallback += 1
            line = ("The pond is calm. Suspiciously calm." if beat is None
                    else self._fallback._render_beat(beat, reg)[0])

        lines = [line]
        if not self._title_shown:
            lines = ["🎬 AS THE SLIDE TURNS — today's episode:"] + lines
            self._title_shown = True
        ev = CaptionEvent(frame_idx=frame_idx, headline=lines[0], lines=lines,
                          kind=(beat.kind if beat else "IDLE"),
                          subjects=(beat.subjects if beat else []))
        self.current = ev
        self.transcript.append(ev)
        self._recent.append(line)
        if len(self._recent) > 3:
            self._recent.pop(0)
        return ev


def make_captioner(cfg: DramaConfig) -> Captioner:
    if cfg.backend == "template":
        return TemplateCaptioner(cfg)
    if cfg.backend == "llm":
        return LLMCaptioner(cfg)
    raise ValueError(f"unknown drama backend: {cfg.backend!r}")

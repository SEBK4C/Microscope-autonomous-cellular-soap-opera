"""Season memory — the show's running lore across episodes.

Persists character bios, feuds and romances to a JSON file so successive
episodes (run with the same drama seed → a recurring cast) build a serialized
soap opera: a "Previously, on As the Slide Turns…" recap opens each new episode
and a cliffhanger closes it.

Returns plain strings (no captioner import) — the pipeline wraps them into
caption cards — so there's no import cycle.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SHOW = "As the Slide Turns"
_FEUD_MILESTONES = {3, 5, 8, 13, 21}


def _pair(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


class SeasonMemory:
    """Load/accumulate/save the show's persistent state (a JSON file)."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.state = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 - corrupt file -> start fresh
                pass
        return {"show": SHOW, "episodes": 0, "characters": {}, "events": []}

    @property
    def episodes_done(self) -> int:
        return int(self.state.get("episodes", 0))

    def _char(self, name: str, archetype: str, ep: int) -> dict:
        chars = self.state.setdefault("characters", {})
        c = chars.get(name)
        if c is None:
            c = {"archetype": archetype, "appearances": 0, "feuds": {},
                 "romances": {}, "births": 0, "exits": 0, "first_ep": ep, "last_ep": ep}
            chars[name] = c
        c["appearances"] += 1
        c["last_ep"] = ep
        c["archetype"] = archetype or c["archetype"]
        return c

    def note_appearance(self, name: str, archetype: str, ep: int) -> None:
        self._char(name, archetype, ep)

    def bump_relationship(self, a: str, b: str, kind: str, ep: int) -> int:
        """kind = 'feuds' | 'romances'. Returns the (symmetric) cumulative count."""
        ca = self._char(a, "", ep)
        cb = self._char(b, "", ep)
        ca[kind][b] = ca[kind].get(b, 0) + 1
        cb[kind][a] = cb[kind].get(a, 0) + 1
        return ca[kind][b]

    def add_event(self, ep: int, text: str, score: float) -> None:
        self.state.setdefault("events", []).append(
            {"ep": ep, "text": text, "score": float(score)})

    def finish_episode(self) -> None:
        self.state["episodes"] = self.episodes_done + 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")


class Showrunner:
    """Observes the drama each frame and produces the recap + cliffhanger."""

    def __init__(self, memory: SeasonMemory, seed: int = 7, observe_every: int = 6):
        self.mem = memory
        self.ep = memory.episodes_done + 1          # the episode now being filmed
        self.rng = random.Random(seed + self.ep)
        self.observe_every = max(1, observe_every)
        self._hot: Dict[Tuple[str, str], Tuple[str, int]] = {}   # this episode's threads
        self._births: List[Tuple[str, str]] = []

    # ---- observation ----------------------------------------------------
    def observe(self, feats, reg, frame_idx: int) -> None:
        if frame_idx % self.observe_every:
            return
        beat = feats.top()
        if beat is None or not beat.subjects:
            return
        chars = [reg.get(s) for s in beat.subjects]
        for c in chars:
            self.mem.note_appearance(c.name, c.archetype, self.ep)
        a = chars[0].name
        b = chars[1].name if len(chars) > 1 else None

        if beat.kind == "DIVIDE" and b:
            self.mem.state["characters"][a]["births"] += 1
            self._births.append((a, b))
            self.mem.add_event(self.ep, f"{a} gave birth to {b} live on the slide", 6.0)
        elif beat.kind == "EXIT":
            self.mem.state["characters"][a]["exits"] += 1
            self.mem.add_event(self.ep, f"{a} drifted out of frame, perhaps forever", 4.0)
        elif b and beat.kind in ("CHASE", "FLEE", "ENCOUNTER"):
            key = _pair(a, b)
            romance = beat.kind == "ENCOUNTER" and self.rng.random() < 0.5
            kind = "romances" if romance else "feuds"
            n = self.mem.bump_relationship(a, b, kind, self.ep)
            self._hot[key] = (kind, n)
            if kind == "feuds" and n in _FEUD_MILESTONES:
                self.mem.add_event(self.ep, f"the feud between {a} and {b} reached its {_ordinal(n)} act", 3.0 + n)
            elif kind == "romances" and n in _FEUD_MILESTONES:
                self.mem.add_event(self.ep, f"{a} and {b} grew closer — chemistry rating {n}", 3.0 + n)

    # ---- recap (opening) ------------------------------------------------
    def recap(self) -> Optional[str]:
        if self.mem.episodes_done == 0:
            return None                              # first episode: nothing to recap
        prior = [e for e in self.mem.state.get("events", []) if e["ep"] < self.ep]
        if not prior:
            return (f"Previously, on {SHOW}: the pond simmered with unspoken intrigue, "
                    f"and absolutely nobody minded their own business.")
        prior.sort(key=lambda e: (e["score"], e["ep"]), reverse=True)
        picks = [e["text"] for e in prior[:3]]
        return f"Previously, on {SHOW}: " + "; ".join(picks) + "."

    # ---- cliffhanger (closing) ------------------------------------------
    def cliffhanger(self) -> str:
        if self._births:
            a, b = self._births[-1]
            return (f"NEXT TIME on {SHOW}: can {a} handle single-celled parenthood, "
                    f"now that {b} has its own agenda? Tune in.")
        if self._hot:
            (a, b), (kind, n) = max(self._hot.items(), key=lambda kv: kv[1][1])
            if kind == "romances":
                return (f"NEXT TIME on {SHOW}: will {a} and {b} finally admit their "
                        f"chemistry (rating: {n})? The current only knows.")
            return (f"NEXT TIME on {SHOW}: will the feud between {a} and {b} "
                    f"(now {n} acts deep) ever end? Osmosis can only do so much.")
        return f"NEXT TIME on {SHOW}: more drifting, more drama. Same slide, same time."

    def finish(self) -> str:
        """Persist the episode and return the cliffhanger line."""
        cliff = self.cliffhanger()
        self.mem.finish_episode()
        return cliff

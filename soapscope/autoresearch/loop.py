"""The autoresearch experiment loop.

``evaluate_config``  : run the fixed synthetic benchmark, return Metrics.
``random_neighbor``  : perturb one knob within sane bounds (the "mutation").
``autoresearch``     : hill-climb — propose, evaluate, keep-if-better, repeat.
``sweep``            : exhaustively try a grid of values for named params.

All runs use ``collect_frames=False`` for speed; only the metrics matter here.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from ..config import PipelineConfig
from ..metrics import Metrics
from ..pipeline import run_synthetic


# (path.to.attr, low, high, is_int)
KNOBS: List[Tuple[str, float, float, bool]] = [
    ("segment.threshold", 0.15, 0.50, False),
    ("segment.min_area", 8, 80, True),
    ("segment.blur", 0, 2, True),
    ("track.max_dist", 25.0, 90.0, False),
    ("track.max_missed", 3, 15, True),
    ("track.min_hits", 1, 4, True),
    ("drama.caption_every", 3, 12, True),
    ("drama.spice", 0.0, 2.0, False),
    ("stage.deadzone", 10.0, 80.0, False),
    ("stage.hysteresis", 1.0, 2.0, False),
]


@dataclass
class ExperimentResult:
    name: str
    params: Dict[str, float]
    score: float
    metrics: Metrics
    kept: bool = False


def _get(cfg: PipelineConfig, path: str):
    obj = cfg
    parts = path.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    return getattr(obj, parts[-1])


def _set(cfg: PipelineConfig, path: str, value) -> None:
    obj = cfg
    parts = path.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], value)


def evaluate_config(cfg: PipelineConfig, n_frames: Optional[int] = None) -> Metrics:
    cfg = copy.deepcopy(cfg)
    if n_frames is not None:
        cfg.n_frames = n_frames
    cfg.render.enabled = False       # skip rendering; we only need metrics
    res = run_synthetic(cfg, collect_frames=False)
    return res.metrics


def random_neighbor(cfg: PipelineConfig, rng: random.Random) -> Tuple[PipelineConfig, str]:
    """Return a deep-copied config with exactly one knob nudged."""
    cfg = copy.deepcopy(cfg)
    path, lo, hi, is_int = rng.choice(KNOBS)
    cur = _get(cfg, path)
    span = (hi - lo)
    delta = rng.uniform(-0.25, 0.25) * span
    val = min(hi, max(lo, cur + delta))
    if is_int:
        val = int(round(val))
        if val == cur:               # ensure a real change
            val = int(min(hi, max(lo, cur + rng.choice([-1, 1]))))
    _set(cfg, path, val)
    return cfg, f"{path}={val}"


def autoresearch(base: Optional[PipelineConfig] = None, rounds: int = 12,
                 n_frames: int = 90, seed: int = 0,
                 log: Callable[[str], None] = print) -> Tuple[PipelineConfig, List[ExperimentResult]]:
    """Hill-climb the config. Returns (best_config, history)."""
    rng = random.Random(seed)
    cfg = copy.deepcopy(base) if base is not None else PipelineConfig()
    best_metrics = evaluate_config(cfg, n_frames)
    best_score = best_metrics.score
    history: List[ExperimentResult] = [
        ExperimentResult("baseline", {}, best_score, best_metrics, kept=True)
    ]
    log(f"[autoresearch] baseline {best_metrics.summary()}")
    for r in range(rounds):
        cand, change = random_neighbor(cfg, rng)
        m = evaluate_config(cand, n_frames)
        kept = m.score > best_score + 1e-4
        history.append(ExperimentResult(f"round{r+1}:{change}",
                       {change.split('=')[0]: change.split('=')[1]},
                       m.score, m, kept=kept))
        flag = "KEEP " if kept else "drop "
        log(f"[autoresearch] {flag} r{r+1:02d} {change:<26} "
            f"score={m.score:.3f} (best={best_score:.3f})")
        if kept:
            cfg, best_score, best_metrics = cand, m.score, m
    log(f"[autoresearch] done. best score={best_score:.3f} :: {best_metrics.summary()}")
    return cfg, history


def sweep(base: PipelineConfig, path: str, values: List, n_frames: int = 90,
          log: Callable[[str], None] = print) -> List[ExperimentResult]:
    """Grid-sweep one parameter; useful for building intuition / the journal."""
    out: List[ExperimentResult] = []
    best = -1.0
    for v in values:
        cfg = copy.deepcopy(base)
        _set(cfg, path, v)
        m = evaluate_config(cfg, n_frames)
        kept = m.score > best
        best = max(best, m.score)
        out.append(ExperimentResult(f"{path}={v}", {path: v}, m.score, m, kept))
        log(f"[sweep] {path}={v!s:<8} score={m.score:.3f}  {m.summary()}")
    return out


def append_journal(path: str | Path, title: str, body: str) -> None:
    """Append a dated section to the autoresearch journal."""
    path = Path(path)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = f"\n## {stamp} — {title}\n\n{body}\n"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(entry)

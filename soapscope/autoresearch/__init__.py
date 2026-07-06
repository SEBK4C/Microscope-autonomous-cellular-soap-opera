"""Autoresearch — Karpathy-style autonomous experimentation over the pipeline.

Instead of editing a training script, the "agent" here proposes tweaks to a
``PipelineConfig``, runs the fixed synthetic benchmark, measures ``score``, and
keeps the change only if it improved. The 15-minute cron loop is the overnight
agent; this module is the reproducible experiment runner it drives.
"""

from .loop import (
    evaluate_config, random_neighbor, autoresearch, sweep,
    ExperimentResult, append_journal,
)

__all__ = [
    "evaluate_config", "random_neighbor", "autoresearch", "sweep",
    "ExperimentResult", "append_journal",
]

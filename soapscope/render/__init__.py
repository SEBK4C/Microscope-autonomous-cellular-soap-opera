"""Rendering: composite masks, character labels, the stage viewport and the
soap-opera caption bar onto each frame for a shareable annotated video."""

from .overlay import render_frame, color_for

__all__ = ["render_frame", "color_for"]

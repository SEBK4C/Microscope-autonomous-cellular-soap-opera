"""Video sources and frame I/O.

A *source* yields RGB uint8 frames of shape (H, W, 3). The synthetic source
also yields ground-truth microbe positions so the autoresearch loop can score
tracking accuracy objectively. Real-video sources (mp4 / frame dirs) plug in
here later without the rest of the pipeline noticing.
"""

from .synthetic import SyntheticWorld, GroundTruth
from .io import save_gif, save_png, load_frames_dir

__all__ = ["SyntheticWorld", "GroundTruth", "save_gif", "save_png", "load_frames_dir"]

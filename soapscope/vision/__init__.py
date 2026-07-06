"""Computer-vision stage: turn pixels into tracked, described characters.

- ``segment`` : frame -> per-object masks (classical today; SAM2/SAM3 later).
- ``track``   : stitch detections across frames into persistent identities.
- ``features``: derive motion/interaction "beats" that the drama layer narrates.
"""

from .segment import Segmenter, ClassicalSegmenter, Detection, SegResult, make_segmenter
from .track import Tracker, Track
from .features import FrameFeatures, FeatureAnalyzer, Beat

__all__ = [
    "Segmenter", "ClassicalSegmenter", "Detection", "SegResult", "make_segmenter",
    "Tracker", "Track", "FrameFeatures", "FeatureAnalyzer", "Beat",
]

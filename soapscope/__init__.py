"""SoapScope — the autonomous cellular soap opera.

Track microbes drifting across a microscope slide and auto-caption their lives
like characters in a daytime soap / a Gary Larson cartoon. The whole software
stack is prototyped from static video (or a synthetic world) so it runs on a
laptop with no GPU, then swaps in SAM3 + a local LLM + a real CNC stage later.

Pipeline:  source -> segment -> track -> features -> drama -> stage -> render
"""

from .config import PipelineConfig

__version__ = "0.1.0"
__all__ = ["PipelineConfig", "__version__"]

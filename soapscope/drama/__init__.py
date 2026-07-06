"""The drama layer — where trajectories become tabloid.

- ``characters`` : give every track a persistent name, title and archetype.
- ``captioner``  : narrate the beats like a daytime-soap announcer / Gary
  Larson caption, keeping a memory of feuds and flirtations so the story
  actually develops across frames.
"""

from .characters import Character, CharacterRegistry
from .captioner import Captioner, TemplateCaptioner, CaptionEvent, make_captioner

__all__ = [
    "Character", "CharacterRegistry",
    "Captioner", "TemplateCaptioner", "CaptionEvent", "make_captioner",
]

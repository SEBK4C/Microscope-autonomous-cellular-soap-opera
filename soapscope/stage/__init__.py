"""The CNC microscope-stage layer.

An abstract stage API that a real microcontroller plugs into over serial, a
pure-software ``SimulatedStage`` for prototyping from static video, and a
``StageController`` "director" that decides which microbe is today's star and
keeps it centred. Swap ``SimulatedStage`` for ``SerialStage`` and the rest of
the stack is unchanged.
"""

from .api import (
    CNCStage, SimulatedStage, SerialStage, StageState, StageCommand,
    MICROCONTROLLER_PROTOCOL,
)
from .controller import StageController, StageStep

__all__ = [
    "CNCStage", "SimulatedStage", "SerialStage", "StageState", "StageCommand",
    "MICROCONTROLLER_PROTOCOL", "StageController", "StageStep",
]

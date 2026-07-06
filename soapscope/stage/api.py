"""CNC stage API — the contract a real microcontroller implements.

The pipeline only ever talks to the abstract ``CNCStage`` interface, so the
same tracking/framing logic drives a simulated stage today and real hardware
tomorrow. The wire protocol is newline-delimited JSON so an Arduino / RP2040 /
ESP32 running a few lines of firmware can parse it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional


MICROCONTROLLER_PROTOCOL = """\
SoapScope CNC Stage Protocol (v0)
=================================
Transport : serial (USB CDC), 115200 baud, newline-delimited JSON ("JSON lines")
Frame     : one JSON object per line, terminated with '\\n'
Units     : microns (host converts pixels->microns via steps_per_px * um_per_step)

Host -> MCU (commands)
  {"cmd":"move_rel","dx":<float>,"dy":<float>,"feed":<float>}   relative move (um), feed = um/s
  {"cmd":"move_abs","x":<float>,"y":<float>,"feed":<float>}     absolute move (um)
  {"cmd":"home"}                                                seek endstops, zero the origin
  {"cmd":"focus","dz":<float>}                                  optional Z / focus nudge (um)
  {"cmd":"stop"}                                                immediate halt
  {"cmd":"ping"}                                                liveness check

MCU -> Host (replies), one per command
  {"ok":true,"x":<float>,"y":<float>,"z":<float>,"moving":<bool>}
  {"ok":false,"err":"<message>"}

Firmware notes
  - Clamp all moves to the stage travel envelope; never lose steps.
  - Respect 'feed' as a max rate; ramp accel internally.
  - Report position after every move so the host closed-loop keeps the star centred.
"""


@dataclass
class StageState:
    x: float = 0.0        # stage position, frame-pixel units in the prototype
    y: float = 0.0
    z: float = 0.0        # focus axis (unused in the prototype)
    moving: bool = False


@dataclass
class StageCommand:
    cmd: str                      # move_rel | move_abs | home | focus | stop | ping
    dx: float = 0.0
    dy: float = 0.0
    x: float = 0.0
    y: float = 0.0
    dz: float = 0.0
    feed: float = 0.0

    def to_json_line(self) -> str:
        d = {"cmd": self.cmd}
        if self.cmd == "move_rel":
            d.update(dx=round(self.dx, 3), dy=round(self.dy, 3), feed=round(self.feed, 3))
        elif self.cmd == "move_abs":
            d.update(x=round(self.x, 3), y=round(self.y, 3), feed=round(self.feed, 3))
        elif self.cmd == "focus":
            d.update(dz=round(self.dz, 3))
        return json.dumps(d) + "\n"


class CNCStage:
    """Abstract stage. Implementations move a slide and report position."""

    def home(self) -> StageState:  # pragma: no cover - interface
        raise NotImplementedError

    def apply(self, cmd: StageCommand) -> StageState:  # pragma: no cover
        raise NotImplementedError

    def position(self) -> StageState:  # pragma: no cover
        raise NotImplementedError


class SimulatedStage(CNCStage):
    """Software stage for prototyping from static video.

    Integrates relative/absolute moves with a per-frame feed-rate cap so the
    virtual camera pans as smoothly as a real CNC would. Position is kept in
    frame-pixel units; ``origin`` is the field-of-view centre.
    """

    def __init__(self, x: float = 0.0, y: float = 0.0, max_step: float = 24.0):
        self.state = StageState(x=x, y=y)
        self.max_step = max_step
        self.history: List[StageState] = [StageState(x=x, y=y)]

    def home(self) -> StageState:
        self.state = StageState(x=0.0, y=0.0)
        self.history.append(StageState(**self.state.__dict__))
        return self.state

    def apply(self, cmd: StageCommand) -> StageState:
        if cmd.cmd == "move_rel":
            tx, ty = self.state.x + cmd.dx, self.state.y + cmd.dy
        elif cmd.cmd == "move_abs":
            tx, ty = cmd.x, cmd.y
        elif cmd.cmd in ("home",):
            return self.home()
        else:  # ping / stop / focus — no XY change in the prototype
            self.state.moving = False
            return self.state
        dx, dy = tx - self.state.x, ty - self.state.y
        dist = (dx * dx + dy * dy) ** 0.5
        if dist > self.max_step and dist > 0:
            s = self.max_step / dist
            dx, dy = dx * s, dy * s
        self.state.x += dx
        self.state.y += dy
        self.state.moving = dist > 0.5
        self.history.append(StageState(**self.state.__dict__))
        return self.state

    def position(self) -> StageState:
        return self.state


class SerialStage(CNCStage):
    """Real-hardware stage over serial (requires ``pyserial``; optional dep).

    Lazily imports pyserial so the core install stays dependency-light. Sends
    the JSON-line protocol above and parses the MCU's position replies.
    """

    def __init__(self, port: str, baud: int = 115200, timeout: float = 1.0):
        try:
            import serial  # type: ignore
        except ImportError as e:  # pragma: no cover - hardware path
            raise ImportError(
                "SerialStage needs pyserial: pip install -e .[hardware]"
            ) from e
        self._ser = serial.Serial(port, baud, timeout=timeout)
        self.state = StageState()

    def _txn(self, cmd: StageCommand) -> StageState:  # pragma: no cover - hardware
        self._ser.write(cmd.to_json_line().encode("utf-8"))
        raw = self._ser.readline().decode("utf-8").strip()
        if raw:
            rep = json.loads(raw)
            if rep.get("ok"):
                self.state = StageState(
                    x=rep.get("x", self.state.x), y=rep.get("y", self.state.y),
                    z=rep.get("z", self.state.z), moving=rep.get("moving", False),
                )
        return self.state

    def home(self) -> StageState:  # pragma: no cover - hardware
        return self._txn(StageCommand("home"))

    def apply(self, cmd: StageCommand) -> StageState:  # pragma: no cover - hardware
        return self._txn(cmd)

    def position(self) -> StageState:  # pragma: no cover - hardware
        return self.state

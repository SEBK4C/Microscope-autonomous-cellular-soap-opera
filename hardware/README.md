# CNC stage — the microcontroller side

SoapScope drives the slide through the abstract `CNCStage` interface
(`soapscope/stage/api.py`). In software that's `SimulatedStage`; on the bench
it's `SerialStage`, which talks to a microcontroller over USB serial using a
tiny **newline-delimited JSON** protocol. Print the full spec any time with:

```bash
python -m soapscope.cli protocol
```

## Wiring (reference design)

A minimal 2-axis stage:

- **MCU:** any Arduino-class board (Uno/Nano) or RP2040/ESP32 (USB CDC).
- **Motion:** 2× stepper (NEMA-8/11) on the X/Y micrometer knobs via belt or
  flexible coupling, driven by A4988/DRV8825 (or TMC2209 for quiet microstep).
- **Endstops:** one per axis for homing.
- **Optional Z:** a third stepper on the fine-focus knob (`focus` command).

The host closes the loop: it tracks microbes in each frame, decides where the
star is, and streams `move_abs`/`move_rel` commands; the MCU just executes and
reports position. Keep the MCU firmware dumb and fast.

## Protocol (v0)

Host → MCU, one JSON object per line (`\n`), 115200 baud, units = microns:

```json
{"cmd":"move_abs","x":120.0,"y":-45.0,"feed":800}
{"cmd":"move_rel","dx":10.0,"dy":0.0,"feed":800}
{"cmd":"home"}
{"cmd":"focus","dz":2.0}
{"cmd":"ping"}
```

MCU → Host, one reply per command:

```json
{"ok":true,"x":120.0,"y":-45.0,"z":0.0,"moving":false}
```

Pixel→micron conversion happens on the host (`steps_per_px · um_per_step`), so
the MCU only ever deals in microns/steps.

## Files

- `firmware_stub.ino` — a comp/uploadable Arduino sketch skeleton that parses
  the protocol and moves two steppers. Fill in the TODOs for your driver.

## Try it against the simulator

You can exercise the exact command stream the hardware would receive, straight
from a static-video run:

```bash
python -m soapscope.cli demo --frames 80
cat out/stage_commands.jsonl        # the JSON lines a microcontroller consumes
```

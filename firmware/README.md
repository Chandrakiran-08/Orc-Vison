# Running the brain on microcontrollers

The perception half of Orc-Vison (camera → model → tracker) **cannot** run on
a microcontroller — it needs CPython, OpenCV and an inference runtime. That
does not have to stop the *brain* from running there.

These ports let the decision layer live on the board, so the machine keeps
deciding — and keeps its memory — even when the link to the host drops.

```
host (camera + model) ──MQTT/serial──▶ MCU (brain) ──▶ motors / relay / servo
       perception                        autonomy            actuation
```

## Which port for which board

| Board | RAM | Port | Status |
|-------|-----|------|--------|
| **Arduino Uno R4 WiFi** (RA4M1) | 32 KB | `OrcVisionBrain/` (C++) | Logic verified on host; **not flash-tested** |
| ESP32 / S3 / C3 | 320–520 KB | `OrcVisionBrain/` (C++) or `micropython/` | Logic verified on host; **not flash-tested** |
| Raspberry Pi Pico W (RP2040) | 264 KB | `micropython/` (or C++ via Arduino core) | Logic verified on host; **not flash-tested** |
| STM32 / nRF52 / Teensy | varies | `OrcVisionBrain/` (C++) | Logic verified on host; **not flash-tested** |
| Raspberry Pi / Jetson (Linux) | — | use the full `orcvision.brain` Python package | Verified |

## Measured footprint (C++ port)

Compiled with `g++ -O2`, `sizeof(OrcVisionBrain)`:

| Tuning | Bytes | Share of Uno R4's 32 KB |
|--------|-------|------------------------|
| Uno R4 preset (6 objects / 8 labels / 10 traces) | **1284 B** | 3.9 % |
| Library default (8 / 12 / 12) | 1520 B | 4.6 % |

Component sizes: `OvObject` 40 B, `OvTrace` 24 B, `OvDecision` 268 B.

The brain is not what constrains you on a 32 KB board — the WiFi stack and
JSON buffer are. Tune with `OV_MAX_OBJECTS`, `OV_MAX_LABELS`, `OV_MAX_TRACES`
before including the header.

## Embedded discipline (enforced by tests)

- **No dynamic allocation.** No `malloc`/`new`, no `String`, no STL
  containers. Heap fragmentation on a long-running MCU is a field failure,
  not a warning. Asserted by `test_cpp_library_uses_no_dynamic_allocation`.
- **No exceptions, no RTTI, no recursion.** Bounded work per frame.
- `float`, not `double`. Only `expf`/`logf`/`sqrtf`.
- Compiles clean under `-Wall -Wextra -Werror`, C++11.

## Parity — all three ports decide identically

A robot that behaves differently depending on where the brain runs is a
nasty class of bug, so the ports are pinned to each other:

```bash
# regenerate golden vectors from the Python reference
python firmware/OrcVisionBrain/tests/generate_golden.py

# check C++ and MicroPython against them
python -m pytest tests/test_firmware_parity.py -v
```

`golden.h` is generated from `orcvision.brain` and committed. If you change
the Python decision logic, regenerate it — a diff there is the signal that
the firmware must change too. The C++ test compiles and runs on the host; it
is skipped where no compiler is present.

Currently **13 golden decisions across 8 scenarios**, including the headline
behaviour: identical perception input, `AVOID` first, `STOP` after feedback
that avoiding failed.

## Verification status — read this before trusting it

| What | Status |
|------|--------|
| Decision logic matches the Python reference | **Verified** (golden-vector parity test, runs in CI) |
| Compiles clean for host with `-Wall -Wextra -Werror` | **Verified** |
| No heap allocation / no STL | **Verified** (test-enforced) |
| Fits in Uno R4 SRAM | **Verified** (measured, test-enforced < 4 KB) |
| Runs correctly on real Uno R4 / ESP32 / Pico W hardware | **NOT verified** — no maintainer has flashed it |
| Cross-compiles with the Arduino toolchain | **NOT verified** — no `arduino-cli` in the dev environment |

The firmware is written carefully and its logic is tested, but nobody has
put it on a board. Treat on-device behaviour as unproven until you confirm
it yourself, and keep the actuator harmless (onboard LED) on first run.

## Quick start — Arduino (Uno R4 WiFi, ESP32, ...)

1. Copy `firmware/OrcVisionBrain/` into your Arduino `libraries/` folder
   (or zip it and use *Sketch → Include Library → Add .ZIP Library*).
2. Open `examples/UnoR4WiFiBrain/UnoR4WiFiBrain.ino`.
3. Install `ArduinoMqttClient` and `ArduinoJson` (v6+); `WiFiS3` ships with
   the UNO R4 board package.
4. Edit the WiFi/broker constants, select *Arduino UNO R4 WiFi*, upload.
5. On the host:

```bash
python -m orcvision run --source video.mp4 --model yolov8n \
    --track --depth --sink mqtt
```

The board subscribes to `orcvision/events`, decides locally, publishes to
`orcvision/actions`, and accepts outcomes on `orcvision/feedback`:

```bash
mosquitto_pub -t orcvision/feedback -m '{"success": false}'
```

That feedback is what makes the next decision different.

## Quick start — MicroPython (ESP32, Pico W)

```bash
mpremote cp firmware/micropython/orcvision_brain.py :
```

```python
from orcvision_brain import VisionBrain
brain = VisionBrain(goal="avoid_collision")
brain.begin_frame(t)
brain.observe("obstacle", 0.9, cx, cy, size, depth_m=1.2, track_id=1)
brain.end_frame()
d = brain.decide()
print(d.explain())
brain.feedback(False); brain.learn()
brain.save("brain.json")     # survives a power cycle
```

## What the MCU does *not* do

It does not run object detection, and this is not a claim that it could.
Detections are made on a host and shipped over the wire. What lives on the
board is state, memory, decision-making, the safety floor and learning —
which is the part that has to keep working when the network does not.

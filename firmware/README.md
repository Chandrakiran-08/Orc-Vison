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
| **Arduino Uno R4 WiFi** (RA4M1) | 32 KB | `OrcVisionBrain/` (C++) | ✅ **Verified on hardware** — self test passes on-device |
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
| Cross-compiles with the Arduino toolchain | **Verified** — arduino-cli 1.5.1, core `arduino:renesas_uno` |
| `BrainSelfTest.ino` runs correctly on a real Uno R4 WiFi | **Verified** — 5/5 on-device, 2026-09-03 |
| `UnoR4WiFiBrain.ino` cross-compiles for the Uno R4 | **Verified** — ~77 KB flash (29%), 9.1 KB globals (27%) |
| `UnoR4WiFiBrain.ino` full networked loop on hardware | **Verified** — WiFi + MQTT + JSON + memory flip, 2026-09-03 |
| ESP32 / Pico W / STM32 hardware | **NOT verified** — no maintainer has flashed those |
| MicroPython port on a real board | **NOT verified** |

### What the hardware run actually proved

Flashed to an Arduino Uno R4 WiFi on 2026-09-03:

```
Sketch uses 60124 bytes (22%) of program storage. Maximum is 262144 bytes.
Global variables use 8036 bytes (24%) of dynamic memory,
leaving 24732 bytes for local variables. Maximum is 32768 bytes.

Brain footprint: 1284 bytes of SRAM
  [PASS] object tracked across frames
  [PASS] motion detected as APPROACHING
  [PASS] fresh situation -> AVOID
  [PASS] remembered failure -> STOP instead of AVOID
  [PASS] MOVE refused with a hazard at 0.4 m
 SELF TEST: PASS (5/5)
```

The decisions are byte-identical to the Python reference, so the
golden-vector parity holds across the toolchain boundary — the same brain,
the same choices, on a 48 MHz microcontroller.

Note the memory line: the brain is 1284 B of the 8036 B of globals; the rest
is the Arduino core and serial buffers. There is 24 KB of headroom for the
WiFi stack and JSON parsing on top.

The networked `UnoR4WiFiBrain.ino` is also verified on hardware. On a real
Uno R4 WiFi it associates with the AP, connects to an MQTT broker,
subscribes, parses a `PerceptionEvent` off `orcvision/events`, decides, and
publishes the action to `orcvision/actions`. Driven through the memory-flip
sequence (`firmware/board_demo.sh`) it chose `AVOID`, took failure feedback,
then chose `STOP` for the identical message — the memory flip, over live
MQTT, on the board:

```
Decision: AVOID (steer right)
feedback: last action FAILED (will be avoided next time)
Decision: STOP
  - AVOID down-weighted: previously failed here
```

What remains unproven on hardware: a real camera feeding real detections (the
board was driven by synthetic events here), the platform-safety layer (absent
from the firmware — see the gap in the repo docs), and the ESP32 / Pico W /
STM32 / MicroPython targets. Keep the actuator harmless (onboard LED) until
you have watched it behave on your own bench.

### ArduinoJson version

Use **6.x**, which `flash_uno_r4.sh` pins. ArduinoJson 7 removed
`StaticJsonDocument` and allocates its document from the heap; doing that
inside an MQTT callback on a 32 KB board reintroduces exactly the
fragmentation risk the rest of this firmware avoids. The sketch has a
version guard so it compiles under either, but on the Uno R4 prefer 6.x.

## Security posture — read before wiring an actuator

The examples use **plain MQTT with no authentication and no TLS**, because
that is what a `mosquitto -v` bench setup gives you. Understand what that
means before this drives anything that moves:

**Anyone who can reach the broker can drive your hardware.** The board acts
on whatever arrives on `orcvision/events`, and a forged detection ("obstacle
at 0.3 m") makes it react. Feedback is worse: anyone publishing to
`orcvision/feedback` writes directly into the brain's long-term memory and
can teach it that a correct action is wrong. There is no message
authentication, no replay protection, and no sender identity anywhere in
this design.

The safety constraints still hold under forged input — they are evaluated
after scoring and cannot be learned away, so a poisoned policy still cannot
advance into a hazard the board can see. But a forged *detection* changes
what the board believes it can see, and no amount of local logic fixes that.

If this leaves the bench:

- Put the broker and boards on an isolated VLAN or a dedicated AP, not the
  house/office network.
- Turn on MQTT authentication (`password_file`) and TLS
  (`mosquitto` `cafile`/`certfile`/`keyfile`); set `allow_anonymous false`.
- Restrict topics per client with a mosquitto ACL so a compromised sensor
  cannot publish feedback.
- Keep a hardware interlock — an e-stop or current limit — that no firmware
  path can override. Treat the software safety floor as defence in depth,
  never as the only thing between a motor and a person.

None of this is exotic; it is just not on by default, and the examples do
not pretend otherwise.

## Quick start — Arduino (Uno R4 WiFi, ESP32, ...)

Scripted, using arduino-cli. Note that the Arduino IDE packaged in most
distro repositories is 1.8.x, which does **not** support the Uno R4 — use
this script, or install IDE 2.x from arduino.cc.

```bash
./firmware/flash_uno_r4.sh check      # install arduino-cli + the board core
./firmware/flash_uno_r4.sh compile    # build the self test — no board needed
./firmware/flash_uno_r4.sh upload     # flash it and open the serial monitor
```

Run the self test first: it needs no libraries and no network, so a failure
points at the brain or the toolchain rather than at WiFi. A good run ends
with `SELF TEST: PASS (5/5)` and a fast-blinking onboard LED.

Then the networked sketch:

```bash
./firmware/flash_uno_r4.sh upload wifi
```

Before that one, edit the WiFi/broker constants at the top of
`examples/UnoR4WiFiBrain/UnoR4WiFiBrain.ino`. It needs `ArduinoMqttClient`
and `ArduinoJson` (the script installs both); `WiFiS3` ships with the board
core. Then, on the host:

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
brain.feedback(False)
brain.learn()
brain.save("brain.json")  # survives a power cycle
```

## What the MCU does *not* do

It does not run object detection, and this is not a claim that it could.
Detections are made on a host and shipped over the wire. What lives on the
board is state, memory, decision-making, the safety floor and learning —
which is the part that has to keep working when the network does not.

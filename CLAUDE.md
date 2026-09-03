# Orc-Vison — working notes for Claude Code

## What this project is

A **perception pipeline** plus an **autonomous decision brain** for
vision-based machines. The guiding principle: *don't build another vision
model — build the brain that turns vision into autonomous decisions.*

```
sensor → model → tracker → rule engine → sink        (perception, orcvision/)
                                  ↓
      state → memory → decision → action → feedback  (brain, orcvision/brain/)
```

## Layout

| Path | What |
|------|------|
| `orcvision/` | Perception pipeline: sensors, models, trackers, sinks, safe-AST rule engine |
| `orcvision/brain/` | The decision brain (CPython reference implementation) |
| `firmware/OrcVisionBrain/` | C++ port for microcontrollers (Uno R4, ESP32, STM32) |
| `firmware/micropython/` | MicroPython port (ESP32, Pico W) |
| `examples/` | Runnable demos, no hardware needed |
| `tests/` | 150 tests, all hardware-free |
| `SPEC.md` | Canonical spec; the v0.2 section at the end governs the brain |

## Non-negotiable invariants

Break these and the project stops being what it is:

1. **The brain never imports a detector.** All perception enters through
   `brain/adapters.py`. YOLO is one input among many.
2. **`orcvision/brain/` is pure standard library.** No numpy, torch,
   pydantic, network or cloud. A test asserts this
   (`test_brain_pulls_in_no_third_party_dependencies`). `orcvision/__init__`
   is lazy (PEP 562) specifically to keep this true.
3. **Safety constraints are evaluated after scoring and can veto anything.**
   A learned policy can reorder preferences but can never unlock a forbidden
   action. If everything is vetoed, take the configured safe action — never
   the least-bad forbidden one.
4. **The firmware ports must stay decision-identical to Python.** Golden
   vectors in `firmware/OrcVisionBrain/tests/golden.h` are generated from the
   Python brain. Change the decision logic → regenerate them → the firmware
   test tells you the ports need updating.
5. **No dynamic allocation in the C++ firmware.** No malloc/new/String/STL.
   Test-enforced.
6. **Honesty about verification.** README and `firmware/README.md` carry
   status tables saying what is and is not tested. Never upgrade a claim
   without doing the work.

## Commands

```bash
pytest -q                                    # 150 tests
ruff check . && ruff format --check .        # what CI runs
python examples/autonomous_brain/demo.py     # the memory flip (AVOID -> STOP)
python examples/uav_obstacle/demo.py
python examples/industrial_safety/demo.py
python firmware/OrcVisionBrain/tests/generate_golden.py   # after brain changes
```

**On a machine with ROS 2 installed**, ROS's `launch_testing` pytest plugin
hijacks collection and fails on a missing `lark`. Use:

```bash
PYTHONPATH= pytest -q          # or PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

## Current state

Branch `claude/repo-capabilities-0yi9zx`, open as
[PR #1](https://github.com/Chandrakiran-08/Orc-Vison/pull/1) and **not yet
merged** — the GitHub landing page still shows the old perception-only
project. Pushing to this branch updates that PR.

Done: brain (state, temporal, memory, trainable policy, constraints,
feedback), C++ and MicroPython ports with parity tests, `PlatformState`
(battery/altitude/geofence/interlock), stale-perception failsafe, frozen
production mode, audit log, UAV and industrial presets.

## Known gaps — read before claiming otherwise

1. **Partly flashed.** `BrainSelfTest.ino` is verified on a real Arduino
   Uno R4 WiFi (5/5 on-device, 1284 B SRAM, decisions identical to Python).
   Still unproven on hardware: `UnoR4WiFiBrain.ino` (WiFiS3 /
   ArduinoMqttClient / ArduinoJson are a different library surface), plus
   ESP32, Pico W, STM32 and the MicroPython port.
2. **The firmware ports lack the platform-safety layer.** No battery,
   staleness, interlock or geofence handling in C++/MicroPython. Parity
   tests still pass only because the golden scenarios do not exercise
   `PlatformState`. This is a real divergence between the README and the
   board.
3. **No real-camera run.** Everything has been driven by synthetic
   detections. `disappear_after_misses`, `match_radius` and the proximity
   thresholds are educated guesses that real footage will correct.
4. **MQTT examples are unauthenticated plaintext.** Anyone on the network
   can forge detections or poison memory via feedback. Documented in
   `firmware/README.md`; fine on a bench, not beyond it.
5. **Not a certified safety system.** Belongs behind rated interlocks and
   flight-controller failsafes, never in place of them.

## Style

- Ruff, line length 100, rules `E,F,I,W,UP,B`.
- Comments explain *why*, especially non-obvious safety reasoning. Match the
  surrounding density; do not narrate what the code plainly says.
- Tests state the property being protected in the docstring, and regression
  tests say what previously went wrong.

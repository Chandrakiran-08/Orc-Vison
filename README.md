# Orc-Vison

**Turn vision sensors into a structured perception event stream — with a
rule-based decision layer — for vision-based autonomous systems.**

Bring a camera and a model; get structured JSON perception events out, with
optional rule-based alerts, over stdout, MQTT, or a file. Standalone,
open-source (Apache-2.0), and deliberately small: it hands you the
`sensor → model → tracker → decision → event` loop so you don't write it
from scratch.

```
sensor ──▶ model ──▶ tracker ──▶ decision rules ──▶ sink (stdout/mqtt/file)
 (rgb)     (onnx/     (track_id)   (alerts[])         JSON PerceptionEvent
           yolo)                        │
                                        ▼
                         ┌──────────────────────────────┐
                         │  autonomous brain (optional) │
                         │  state ▸ memory ▸ decision   │
                         │  ▸ action ▸ feedback ▸ learn │
                         └──────────────────────────────┘
```

Perception is the "eyes and reflexes"; the optional
[**brain layer**](#autonomous-brain) adds the part that remembers, decides
and learns. It is model-agnostic — YOLO is just one possible input.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install orcvision[cpu]          # base + ONNX Runtime (CPU)
# pip install orcvision[gpu]        # ONNX Runtime (CUDA)
# pip install orcvision[yolo]       # Ultralytics backend (.pt / known names, AGPL)
# pip install orcvision[depth]      # monocular depth (transformers + torch)
# pip install orcvision[realsense]  # Intel RealSense RGB-D
```

The base package is intentionally light — inference backends and hardware
SDKs live behind extras and are lazy-imported.

## 30-second quickstart

```bash
# See your environment (Python, ONNX providers, cameras, CUDA, extras)
python -m orcvision doctor

# Run a detector on a video file and print JSON events
python -m orcvision run --source path/to/video.mp4 --model yolov8n --sink stdout

# ...or your webcam
python -m orcvision run --source 0 --model yolov8n --sink stdout
```

Each frame prints one `PerceptionEvent` as JSON:

```json
{"timestamp":1788240924.72,"frame_id":1,"source":"camera:0","modality":"rgb",
 "frame_shape":[480,640],
 "detections":[{"label":"person","confidence":0.69,"bbox":[179,220,640,475],
                "track_id":null,"depth_m":null,"class_id":0}],
 "alerts":[]}
```

## Compatibility

**Verified (perception pipeline):**
- Ubuntu 22.04 / 24.04 and derivatives (Pop!_OS, Mint, elementary, Zorin)
- x86_64, CPU and NVIDIA GPU (CUDA via onnxruntime-gpu)
- RGB cameras: USB/CSI webcams, video files, RTSP streams

**Should work (untested, reports welcome):**
- Debian 12+, Fedora 38+, Arch, openSUSE
- Raspberry Pi 5, NVIDIA Jetson (Orin family)
- Intel RealSense cameras (code written, not verified on real hardware)

**Not implemented (extension points, contributions welcome):**
- Thermal cameras
- Stereo camera pairs
- Non-visual sensor fusion (IMU, lidar, etc.) — out of scope by design

**Microcontroller / hardware output side:**
- Arduino Uno R4 WiFi — firmware written, pending real-hardware
  verification by the maintainer (author will update this line once
  flash-tested)
- ESP32, Raspberry Pi Pico W — reference examples only, untested
- orcvision/mqtt_simulator.py — a software stand-in for any
  microcontroller, verified, usable right now to test the full
  pipeline without hardware

**Microcontroller brain ports (`firmware/`):**
- Arduino Uno R4 WiFi — **verified on hardware**. Both the self test (5/5,
  1284 B SRAM) and the full networked sketch (WiFi + MQTT + JSON + the
  AVOID→STOP memory flip over live MQTT) run on-device, decisions
  byte-identical to the Python reference
- Still not flash-tested: a real camera feeding it, and the ESP32, Pico W
  and MicroPython targets

## Sensor support

Any class implementing `SensorProtocol` is a valid sensor — no registration
needed. Return a `SensorFrame` from `read()` and clean up in `release()`.

| Sensor | Module | Status |
|--------|--------|--------|
| RGB (webcam / file / RTSP) | `sensors/rgb_camera.py`, `rtsp_camera.py` | **Tested** |
| Intel RealSense (RGB-D) | `sensors/realsense.py` | Code written, **unverified** (no hardware) |
| Thermal | `sensors/thermal_camera.py` | **Stub** — documented extension point |
| Stereo pair | `sensors/stereo_camera.py` | **Stub** — documented extension point |

**Adding a sensor:**

```python
from orcvision.sensors import SensorFrame


class MySensor:
    def read(self) -> SensorFrame | None:
        rgb = ...  # np.ndarray (H, W, 3)
        return SensorFrame(rgb=rgb, depth=None, timestamp=..., modality="rgb")

    def release(self) -> None: ...
```

See the docstrings in `thermal_camera.py` / `stereo_camera.py` for
step-by-step guides.

## Event schema

```python
class Detection(BaseModel):
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    track_id: int | None = None
    depth_m: float | None = None  # sensor depth, monocular estimate, or null
    class_id: int | None = None


class PerceptionEvent(BaseModel):
    timestamp: float
    frame_id: int
    source: str  # e.g. "camera:0", "realsense:0"
    modality: str  # rgb | rgbd | thermal | stereo
    frame_shape: tuple[int, int]  # (height, width)
    detections: list[Detection]
    alerts: list[str] = []  # populated by the decision layer
```

## Bring your own model

A model is anything with `infer(frame) -> list[Detection]`. Resolution
order:

1. `*.onnx` path → ONNX Runtime backend (`[cpu]` / `[gpu]`)
2. `*.pt` path → Ultralytics backend (`[yolo]`)
3. Known name (`rtdetr-l`, `yolov8n`, …) → Ultralytics auto-download to
   `~/.cache/orcvision/`

```bash
python -m orcvision run --source 0 --weights ./my_model.onnx --sink stdout
```

ONNX provider auto-detection prefers `CUDAExecutionProvider` and falls back
to `CPUExecutionProvider`. The Ultralytics backend probes the GPU and falls
back to CPU if the installed torch build has no kernels for your card.

## Decision rules

Config-driven, rule-based (**not** learned). Expressions are evaluated with
a safe AST walker over detection fields only — **no `eval()`/`exec()` of
arbitrary code**.

```yaml
decision:
  # Per-detection rules: evaluated once per detected object.
  rules:
    - when: "label == 'person' and confidence > 0.8"
      action: "alert"
      message: "Confident person detected"   # optional custom text
      severity: "warning"                     # optional; prefixes the alert
      cooldown_s: 30
    - when: "label == 'person' and track_id is not None"
      action: "alert"
      min_consecutive_frames: 5
  # Event-scope rules: evaluated once per frame over ALL detections.
  event_rules:
    - when: "count('person') >= 3"
      message: "Crowd forming"
      severity: "critical"
    - when: "exists('person') and exists('vehicle')"
      message: "Person near vehicle"
    - when: "min_depth('obstacle') < 1.5"
      message: "Obstacle within 1.5 m"
```

**Per-detection fields:** `label`, `confidence`, `bbox`, `track_id`,
`depth_m`, `class_id`.

**Event-scope helpers** (frame-level "how many" / "is there both X and Y"
decisions a single detection can't express):

| Helper | Returns |
|--------|---------|
| `count(label=None)` | number of detections (optionally of one label) |
| `exists(label)` | `True` if any detection has that label |
| `max_conf(label=None)` | highest confidence, or `0.0` if none match |
| `min_depth(label=None)` | nearest `depth_m`, or `null` if unknown |

Both scopes support `cooldown_s`, `min_consecutive_frames`, and optional
`name` / `message` / `severity`. Matched rules append to
`PerceptionEvent.alerts` (formatted `[severity] name: message`, or the
legacy `action: when` when none are set). The same safe AST evaluator backs
both — event-scope expressions may only call the four helpers above; no
other calls, attribute access, or arbitrary code is ever evaluated.

## Autonomous brain

The decision rules above are *reflexes* — they see one frame and fire. The
brain layer (`orcvision.brain`) is the intelligence above perception: it
keeps state, remembers outcomes, and decides.

```
Perception → State → Memory → Decision → Action → Feedback → (loop)
```

The difference in one line: **given identical perception input, the brain
can choose a different action, because it remembers how the last one
turned out.**

```bash
python examples/autonomous_brain/demo.py   # no camera/model/network needed
```

```
TRIAL 1   obstacle closing 4.0 m -> 1.2 m   ->  AVOID(direction=right)
   >>> feedback: that failed (clipped the obstacle)
TRIAL 2   identical input                   ->  STOP
```

### API

```python
from orcvision.brain import VisionBrain

brain = VisionBrain(goal="avoid_collision")
brain.observe(vision_output)  # PerceptionEvent, dicts, or a SceneState
decision = brain.decide()
print(decision.explain())  # every decision is inspectable
brain.execute(decision)  # Action -> your hardware executor
brain.feedback(success=False)  # outcome -> memory (changes next decision)
brain.learn()  # outcome -> policy weights
```

Attach it to the live pipeline instead:

```bash
python -m orcvision run --source video.mp4 --model yolov8n --track --depth \
    --brain --goal avoid_collision --explain
```

Each event then carries a `decision` field alongside `alerts` (existing
consumers, including the microcontroller sketches, are unaffected).

### Model-agnostic by construction

The brain never imports a detector. Any perception source works:

```python
from orcvision.brain import from_records

scene = from_records(
    [{"label": "obstacle", "confidence": 0.9, "bbox": (0.4, 0.4, 0.6, 0.6)}],
    timestamp=t,
    normalized=True,  # embedded pipelines often skip pixels
)
```

Pixel coordinates are normalized to 0..1, so a policy trained at 640×480
still works at 1920×1080 or on a 96×96 microcontroller camera.

### Layers (each independently replaceable)

| Module | Role |
|--------|------|
| `brain/adapters.py` | any vision output → normalized `SceneState` |
| `brain/state.py` | `ObjectState` / `SceneState` / `WorldState` |
| `brain/temporal.py` | appeared / moved / approaching / stopped / disappeared |
| `brain/memory.py` | bounded working memory + decaying long-term memory |
| `brain/decision.py` | feature extraction → utility scoring → explained choice |
| `brain/policy.py` | trainable `LinearPolicy` (imitation + reward) |
| `brain/constraints.py` | deterministic safety floor under the learned policy |
| `brain/actions.py` | generic `Action` vocabulary, no hardware coupling |
| `brain/feedback.py` | outcome → memory + policy update |

Swap the decision engine without touching memory; swap memory without
touching perception. Pass your own via the `VisionBrain(...)` constructor.

### Trainable, not a black box

The policy is a linear model over named features — a trained policy is a few
dozen floats — and every decision decomposes into readable terms:

```python
brain.policy.fit(dataset, epochs=10)  # imitation / supervised
brain.policy.reinforce(action, feats, reward)  # RL-compatible
brain.save("~/.cache/orcvision/brain")  # weights + memory persist
```

### Where it runs (honest version)

Perception needs a real computer. The brain does not — it has ports for
microcontrollers, in [`firmware/`](firmware/README.md).

| Half | Needs | Runs on |
|------|-------|---------|
| Perception (`sensor → model → tracker`) | CPython + OpenCV + ONNX/Ultralytics | Linux SBC or PC. **No MCU.** |
| Brain — Python (`orcvision.brain`) | CPython 3.11+, stdlib only | Any CPython board, incl. Pi Zero 2 W |
| Brain — C++ (`firmware/OrcVisionBrain/`) | C++11, ~1.3 KB RAM | **Uno R4 WiFi**, ESP32, STM32, nRF52, Teensy |
| Brain — MicroPython (`firmware/micropython/`) | MicroPython | ESP32, Pico W |

So the Arduino Uno R4 WiFi **can** run the brain: the C++ port uses
**1284 bytes** with its preset tuning — 3.9 % of the board's 32 KB SRAM. It
does *not* run detection; detections arrive over MQTT from a host, and the
board decides locally:

```
host (camera + model) ──MQTT──▶ MCU (brain) ──▶ motors / relay / servo
```

Putting the brain on the board means autonomy — and memory of what failed
last time — survives the link to the host going down.

All three ports are pinned to the same behaviour by a golden-vector parity
test (`tests/test_firmware_parity.py`): 13 decisions across 8 scenarios,
including the `AVOID → STOP` memory flip. The C++ builds clean under
`-Wall -Wextra -Werror` with no heap allocation, no STL and no `String`.

**Status:** the C++ brain is **verified on a real Arduino Uno R4 WiFi** —
both the on-device self test (5/5) and the full networked sketch (WiFi +
MQTT + JSON parse + the AVOID→STOP memory flip over live MQTT) run on
hardware, making the same decisions as the Python reference. Still unproven:
a real camera feeding it, and the ESP32 / Pico W / STM32 / MicroPython
targets. See [`firmware/README.md`](firmware/README.md) for the full status
table before trusting it near an actuator.

### Built for UAVs and industrial automation

The brain reasons about the **machine as well as the world**. Most real
incidents are energy, containment or health — not a missed detection — so
`PlatformState` carries battery, altitude, geofence, link and interlock
state, and the constraints act on it.

```bash
python examples/uav_obstacle/demo.py        # aerial platform
python examples/industrial_safety/demo.py   # machine cell guard
```

**UAV** (`examples/uav_obstacle/`) — mission → `RETURN_HOME` below the 25%
battery line → `DESCEND` below 10%; geofence breach returns home on a full
battery; a lost link stops the mission; `ASCEND` lets it climb over an
obstacle rather than only dodging. Four of the six demo decisions are made
on telemetry, not pixels.

**Industrial** (`examples/industrial_safety/`) — keep-out zones as a
software light curtain, `EMERGENCY_STOP` on a tripped interlock, and the
one most systems get wrong: **when the camera feed goes quiet, the cell
stops.** Perception going stale is itself a safety event, not a reason to
keep acting on a frozen snapshot.

Three deployment properties make that possible:

| | |
|---|---|
| **Stale-perception failsafe** | `StaleDataConstraint` forbids anything but a declared safe state once the world model is older than a limit |
| **Frozen policy** | `brain.freeze()` pins the weights so behaviour is deterministic and reproducible — you cannot certify a machine whose policy drifts |
| **Audit trail** | every decision appended as JSON Lines (action, weighted reasons, vetoes, platform state), rotating so it cannot fill the disk |

⚠️ **Not a certified safety system.** Functional safety for machinery
(IEC 62061, ISO 13849) and UAV airworthiness need rated, redundant,
assessed components. Use this as a supervisory layer *behind* proper
interlocks, light curtains, e-stops and flight-controller failsafes —
never in place of them. Nothing here has flown or guarded a real machine.

### Safety: learning cannot override the floor

A learned policy driving actuators has a real failure mode — if every
action accumulates failures, scores sink and something unsafe can float to
the top. So the brain is a **hybrid**: the trainable policy proposes,
deterministic constraints dispose. A constraint can veto an action outright
no matter how attractive the policy finds it, and if everything is vetoed
the brain falls back to `STOP` rather than the least-bad forbidden action.

```
MOVE forbidden: obstacle at 0.40 m in center (proximity 0.92 >= 0.60)
```

Constraints are plain, auditable Python with no weights — nothing that
drifts with training.

## Sinks

| `--sink` | Output |
|----------|--------|
| `stdout` | newline-delimited JSON to stdout |
| `mqtt`   | publishes JSON to an MQTT topic (paho-mqtt) |
| `file`   | appends JSON lines to a `.jsonl` file |

## Config file

Describe a whole run in YAML and pass `--config`:

```bash
python -m orcvision run --config examples/security_person_alert/config.yaml
```

See `examples/` for `security_person_alert` (decision rules),
`drone_obstacle` (monocular depth), and `arm_pick` (RGB-D).

## Microcontroller integration

The pipeline can publish over MQTT to any subscriber. Test the whole loop
with **zero hardware** using the software receiver:

```bash
mosquitto -v &                                             # a broker
python -m orcvision.mqtt_simulator --topic orcvision/events &
python -m orcvision run --source video.mp4 --sink mqtt --model yolov8n
```

Firmware examples live in `examples/microcontroller/` — an Arduino Uno R4
WiFi sketch (written, **pending maintainer flash-test**) plus ESP32 and
Pico W reference-only examples. See that folder's README for the exact
verification status of each.

## Training

Thin wrapper over Ultralytics (requires `[yolo]`). Bring your own labeled
dataset in Ultralytics YAML format:

```bash
python -m orcvision train --data data.yaml --model yolov8n --epochs 10
python -m orcvision test  --weights runs/detect/train/weights/best.pt --data data.yaml
```

## Roadmap

**Landed in v0.2 (the brain layer):** internal state + world model, temporal
reasoning, working/long-term memory, explainable utility decisions, generic
action abstraction, feedback loop, trainable policy, safety constraints.

**Next:**

- ROS 2 bridge (`orcvision-ros`)
- Real thermal / stereo sensor implementations
- Raspberry Pi 5 / Jetson verification of the brain loop
- Richer decision engines behind the same protocol (decision tree, tiny MLP,
  full RL policy)
- Multi-goal arbitration and goal stacks

## License

Apache-2.0. See [LICENSE](LICENSE). The optional `[yolo]` extra pulls in
Ultralytics, which is AGPL-licensed — it is kept isolated behind that extra
and never imported unless you opt in.

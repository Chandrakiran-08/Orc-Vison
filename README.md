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
           yolo)
```

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
  rules:
    - when: "label == 'person' and confidence > 0.8"
      action: "alert"
      cooldown_s: 30
    - when: "label == 'person' and track_id is not None"
      action: "alert"
      min_consecutive_frames: 5
```

Available fields: `label`, `confidence`, `bbox`, `track_id`, `depth_m`,
`class_id`. Matched rules append to `PerceptionEvent.alerts`.

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

## Roadmap (v0.2)

- ROS 2 bridge (`orcvision-ros`)
- Real thermal / stereo sensor implementations
- Raspberry Pi 5 / Jetson verification
- Learned decision layer (optional, alongside rules)

## License

Apache-2.0. See [LICENSE](LICENSE). The optional `[yolo]` extra pulls in
Ultralytics, which is AGPL-licensed — it is kept isolated behind that extra
and never imported unless you opt in.

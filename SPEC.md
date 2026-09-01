# Orc-Vison — v0.1 Specification

## What this is
A lightweight Python CLI + library that turns vision sensors into a
structured perception event stream, with a simple rule-based decision
layer, for vision-based autonomous systems (drones, UGVs, robotic arms,
security cameras, humanoids). Package name: `orcvision`.

Users bring a camera (or other vision sensor) and a model; they get
structured JSON perception events out, with optional rule-based
decisions/alerts, over stdout, MQTT, or a file. Standalone open-source
project — not tied to any other product.

## What this is NOT (v0.1 non-goals — do not build)
- ROS 2 integration (future orcvision-ros package)
- Non-visual sensor fusion (IMU, lidar, ultrasonic) — vision sensors only
- Learned/ML-based decision-making — rule-based only in v0.1
- Multi-camera simultaneous fusion (single active sensor at a time)
- SLAM, navigation, motion planning
- Docker
- Labeling / dataset annotation tools (bring your own labeled data)
- Model zoo hosting
- TensorRT-specific code paths
- Full implementations of thermal or stereo camera sensors (protocol
  only — see Sensor section)

## Target users
Solo devs and small teams building vision-based autonomous systems who
want to skip writing the sensor → model → tracker → decision → event
loop from scratch.

## Distro / hardware posture
Ubuntu 22.04/24.04 and derivatives (Pop!_OS, Mint, elementary, Zorin)
are the verified target for the Python pipeline. No distro-specific
code; no hardcoded paths; use platform.machine()/platform.system()
only if branching is truly required.

Raspberry Pi 5 and NVIDIA Jetson are designed-in (ARM64-compatible
dependencies) but NOT verified — say so honestly in the README.

## Tech stack (locked)
- Python 3.11+ (targeting 3.12)
- Typer (CLI)
- Pydantic v2 (event + sensor frame schemas)
- opencv-python-headless (camera I/O — NOT opencv-python)
- ONNX Runtime (default inference backend, NOT bundled in base deps)
- Ultralytics (opt-in only, `[yolo]` extra — AGPL, kept isolated)
- pyrealsense2 (opt-in only, `[realsense]` extra, lazy import)
- paho-mqtt (MQTT sink, Python side)
- pytest (tests)
- ruff (lint/format)

See the pasted build spec for full package layout, schemas, CLI
commands, and constraints. This file is the canonical reference; the
README compatibility section must be copied verbatim from it.

## Event schema
Detection: label, confidence, bbox (x1,y1,x2,y2), track_id?, depth_m?, class_id?
PerceptionEvent: timestamp, frame_id, source, modality, frame_shape,
detections[], alerts[].

## Decision layer
Config-driven condition/action rules, safe expression evaluation only —
no eval()/exec() of arbitrary code. Matched rules append to
PerceptionEvent.alerts.

## Local preview (--display)
Optional `--display` flag on `orcvision run` opens a local OpenCV window
showing the live feed with bounding boxes/labels (red box border when the
frame has alerts, green otherwise). Off by default; requires a GUI-capable
`opencv-python` install (not `opencv-python-headless`) on the machine
running it — a clear error should be raised if the GUI backend is missing.
'q' or Esc closes the window and stops the run. Not intended for headless
servers/CI.

## Explicit build constraints for Claude Code
- Do not add dependencies not listed above without asking first
- Do not create files outside the layout
- Do not implement anything on the non-goals list
- realsense/thermal/stereo must not block the rest of the build
- Decision layer uses safe expression evaluation only
- Lazy-import ultralytics, torch, transformers, pyrealsense2
- Every GPU code path falls back to CPU gracefully
- Never call cv2.imshow(); opencv-python-headless only
- Test suite must pass with zero physical hardware
- uno_r4_wifi_alert.ino must be written carefully but NEVER described
  as tested/verified
- License: Apache-2.0 throughout

## README compatibility section (exact honesty required — copy verbatim)
```markdown
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
```

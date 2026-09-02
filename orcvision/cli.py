"""Orc-Vison command-line interface (Typer)."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import typer

from orcvision import __version__

app = typer.Typer(
    add_completion=False,
    help="Turn vision sensors into a structured perception event stream.",
    no_args_is_help=True,
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _coerce_source(source: str):
    """Return an int device index when the source is a bare integer."""
    if isinstance(source, str) and source.isdigit():
        return int(source)
    return source


def _build_sensor(sensor_type: str, source, width, height):
    if sensor_type == "rgb":
        from orcvision.sensors.rgb_camera import RGBCamera

        return RGBCamera(source=source, width=width, height=height)
    if sensor_type == "realsense":
        from orcvision.sensors.realsense import RealSenseCamera

        return RealSenseCamera(width=width or 640, height=height or 480)
    raise typer.BadParameter(f"Unknown --sensor-type {sensor_type!r} (rgb|realsense)")


def _build_sink(sink_type: str, cfg):
    if sink_type == "stdout":
        from orcvision.sinks.stdout import StdoutSink

        return StdoutSink()
    if sink_type == "mqtt":
        from orcvision.sinks.mqtt import MQTTSink

        return MQTTSink(host=cfg.host, port=cfg.port, topic=cfg.topic)
    if sink_type == "file":
        from orcvision.sinks.file import FileSink

        return FileSink(cfg.path)
    raise typer.BadParameter(f"Unknown --sink {sink_type!r} (stdout|mqtt|file)")


def _source_name(sensor, fallback: str) -> str:
    return getattr(sensor, "source_name", fallback)


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------
@app.command()
def run(
    source: str = typer.Option("0", help="Device index, file path, or rtsp:// URL"),
    sensor_type: str = typer.Option("rgb", "--sensor-type", help="rgb | realsense"),
    model: str = typer.Option("rtdetr-l", help="Model name or path (.onnx/.pt)"),
    weights: str | None = typer.Option(None, help="Explicit weights path override"),
    sink: str = typer.Option("stdout", help="stdout | mqtt | file"),
    confidence: float = typer.Option(0.35, help="Detection confidence threshold"),
    depth: bool = typer.Option(False, "--depth", help="Monocular depth estimation"),
    track: bool = typer.Option(False, "--track", help="Assign track_ids across frames"),
    max_frames: int | None = typer.Option(
        None, help="Stop after N frames (useful for files / testing)"
    ),
    config: Path | None = typer.Option(None, help="YAML config file (overrides flags)"),
    display: bool = typer.Option(
        False, "--display", help="Show a local preview window with boxes/labels (needs a GUI)"
    ),
):
    """Run the perception pipeline: sensor -> model -> tracker -> decision -> sink."""
    from orcvision.config import RunConfig, SinkConfig, load_config
    from orcvision.decision.rules import RuleEngine
    from orcvision.events import PerceptionEvent
    from orcvision.models.resolve import resolve_model

    # Resolve effective settings from config file or CLI flags.
    if config is not None:
        cfg: RunConfig = load_config(config)
        sensor_type = cfg.sensor.type
        src = _coerce_source(str(cfg.sensor.source))
        width, height = cfg.sensor.width, cfg.sensor.height
        model_name = cfg.model.name
        weights = cfg.model.weights
        confidence = cfg.model.confidence
        track = cfg.tracker.enabled
        depth = cfg.depth.enabled
        sink_type = cfg.sink.type
        sink_cfg = cfg.sink
        rules = RuleEngine.from_config(cfg.decision.rules, cfg.decision.event_rules)
    else:
        src = _coerce_source(source)
        width = height = None
        model_name = model
        sink_type = sink
        sink_cfg = SinkConfig(type=sink)
        rules = RuleEngine()

    sensor = _build_sensor(sensor_type, src, width, height)
    detector = resolve_model(model_name, weights=weights, confidence=confidence, track=track)

    # The Ultralytics backend tracks natively (track=True passed through).
    # For the ONNX backend there is no built-in tracker, so attach the
    # lightweight IoU tracker instead.
    tracker = None
    if track:
        from orcvision.models.onnx_detector import ONNXDetector

        if isinstance(detector, ONNXDetector):
            from orcvision.trackers import IoUTracker

            tracker = IoUTracker()

    depth_estimator = None
    if depth:
        from orcvision.models.depth import DepthEstimator

        depth_estimator = DepthEstimator()

    out = _build_sink(sink_type, sink_cfg)
    source_name = _source_name(sensor, f"{sensor_type}:{src}")

    if display:
        import cv2  # local import: only pulled in when --display is used

    frame_id = 0
    try:
        while True:
            frame = sensor.read()
            if frame is None:
                break

            detections = detector.infer(frame.rgb)

            if tracker is not None:
                detections = tracker.update(detections)

            if depth_estimator is not None:
                depth_map = depth_estimator.estimate(frame.rgb)
                for det in detections:
                    det.depth_m = depth_estimator.sample_bbox(depth_map, det.bbox)
            elif frame.depth is not None:
                # Sensor-provided metric depth (e.g. RealSense).
                for det in detections:
                    x1, y1, x2, y2 = det.bbox
                    cy = min(max((y1 + y2) // 2, 0), frame.depth.shape[0] - 1)
                    cx = min(max((x1 + x2) // 2, 0), frame.depth.shape[1] - 1)
                    det.depth_m = float(frame.depth[cy, cx])

            event = PerceptionEvent(
                timestamp=frame.timestamp,
                frame_id=frame_id,
                source=source_name,
                modality=frame.modality,
                frame_shape=(frame.rgb.shape[0], frame.rgb.shape[1]),
                detections=detections,
            )
            rules.apply(event)
            out.emit(event)

            if display:
                vis = frame.rgb.copy()
                for det in detections:
                    x1, y1, x2, y2 = det.bbox
                    color = (0, 0, 255) if event.alerts else (0, 200, 0)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                    label = f"{det.label} {det.confidence:.2f}"
                    cv2.putText(
                        vis, label, (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
                    )
                cv2.imshow("Orc-Vison", vis)
                # 1ms poll so the window redraws; 'q' or Esc quits cleanly.
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            frame_id += 1
            if max_frames is not None and frame_id >= max_frames:
                break
    except KeyboardInterrupt:  # pragma: no cover - interactive
        typer.echo("Interrupted, shutting down.", err=True)
    finally:
        sensor.release()
        out.close()
        if hasattr(detector, "release"):
            detector.release()
        if display:
            cv2.destroyAllWindows()

    typer.echo(f"Processed {frame_id} frame(s).", err=True)


# --------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------
@app.command()
def doctor():
    """Report environment: Python, platform, ONNX providers, cameras, CUDA."""
    typer.echo(f"orcvision {__version__}")
    typer.echo(f"Python:        {platform.python_version()}")
    typer.echo(f"Platform:      {platform.system()} {platform.release()}")
    typer.echo(f"Architecture:  {platform.machine()}")

    # ONNX Runtime providers
    try:
        import onnxruntime as ort

        typer.echo(f"ONNX Runtime:  {ort.__version__}")
        typer.echo(f"  providers:   {', '.join(ort.get_available_providers())}")
    except ImportError:
        typer.echo("ONNX Runtime:  not installed (pip install orcvision[cpu]|[gpu])")

    # Cameras (/dev/video*)
    devices = sorted(Path("/dev").glob("video*")) if platform.system() == "Linux" else []
    if devices:
        typer.echo(f"Cameras:       {', '.join(str(d) for d in devices)}")
    else:
        typer.echo("Cameras:       none found at /dev/video*")

    # video group membership
    try:
        import grp
        import os

        video_gid = grp.getgrnam("video").gr_gid
        in_video = video_gid in os.getgroups()
        typer.echo(f"video group:   {'yes' if in_video else 'NO — camera access may fail'}")
    except (KeyError, ImportError):
        typer.echo("video group:   unknown")

    # CUDA driver presence (best-effort, no hard dependency)
    import shutil

    if shutil.which("nvidia-smi"):
        typer.echo("CUDA driver:   nvidia-smi present")
    else:
        typer.echo("CUDA driver:   nvidia-smi not found (CPU-only inference)")

    # pyrealsense2
    try:
        import pyrealsense2  # noqa: F401

        typer.echo("pyrealsense2:  installed")
    except ImportError:
        typer.echo("pyrealsense2:  not installed ([realsense] extra, optional)")

    # ultralytics
    try:
        import ultralytics  # noqa: F401

        typer.echo("ultralytics:   installed")
    except ImportError:
        typer.echo("ultralytics:   not installed ([yolo] extra, optional)")


# --------------------------------------------------------------------------
# info
# --------------------------------------------------------------------------
@app.command()
def info():
    """Show version, available models, sensors, and backends."""
    from orcvision.models.resolve import KNOWN_NAMES

    typer.echo(f"orcvision {__version__}")
    typer.echo("")
    typer.echo("Sensors:")
    typer.echo("  rgb        RGB camera / video file / RTSP  (tested)")
    typer.echo("  realsense  Intel RealSense RGB-D           (code written, unverified)")
    typer.echo("  thermal    stub / extension point          (not implemented)")
    typer.echo("  stereo     stub / extension point          (not implemented)")
    typer.echo("")
    typer.echo("Model backends:")
    typer.echo("  .onnx      ONNX Runtime      ([cpu] or [gpu] extra)")
    typer.echo("  .pt / name Ultralytics       ([yolo] extra)")
    typer.echo(f"  known names: {', '.join(sorted(KNOWN_NAMES))}")
    typer.echo("")
    typer.echo("Sinks: stdout, mqtt, file")


# --------------------------------------------------------------------------
# train / test (thin Ultralytics wrappers)
# --------------------------------------------------------------------------
@app.command()
def train(
    data: str = typer.Option(..., help="Dataset YAML (Ultralytics format)"),
    model: str = typer.Option("yolov8n", help="Base model to fine-tune"),
    epochs: int = typer.Option(10, help="Training epochs"),
    imgsz: int = typer.Option(640, help="Image size"),
):
    """Train a model (thin wrapper over Ultralytics; requires [yolo] extra)."""
    from orcvision.training.train import train_model

    train_model(data=data, model=model, epochs=epochs, imgsz=imgsz)


@app.command()
def test(
    weights: str = typer.Option(..., help="Weights to validate (.pt)"),
    data: str = typer.Option(..., help="Dataset YAML (Ultralytics format)"),
):
    """Validate a model (thin wrapper over Ultralytics; requires [yolo] extra)."""
    from orcvision.training.train import validate_model

    metrics = validate_model(weights=weights, data=data)
    typer.echo(str(metrics))


@app.command()
def version():
    """Print the orcvision version."""
    typer.echo(__version__)


def main() -> None:  # pragma: no cover - console-script entry
    app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())  # type: ignore[func-returns-value]

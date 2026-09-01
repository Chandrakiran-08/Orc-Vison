"""Thin wrappers over Ultralytics train/val (requires the ``[yolo]`` extra).

Training is intentionally a shallow pass-through — Orc-Vison does not
reimplement a training loop. Bring your own labeled dataset in Ultralytics
YAML format.
"""

from __future__ import annotations

from typing import Any


def _load_yolo(model: str):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Training needs the optional extra: `pip install orcvision[yolo]` "
            "(Ultralytics is AGPL-licensed)."
        ) from exc
    return YOLO(model)


def train_model(data: str, model: str = "yolov8n", epochs: int = 10, imgsz: int = 640) -> Any:
    """Fine-tune ``model`` on ``data`` for ``epochs`` epochs."""
    yolo = _load_yolo(model)
    return yolo.train(data=data, epochs=epochs, imgsz=imgsz)


def validate_model(weights: str, data: str) -> Any:
    """Validate ``weights`` against ``data`` and return the metrics object."""
    yolo = _load_yolo(weights)
    return yolo.val(data=data)

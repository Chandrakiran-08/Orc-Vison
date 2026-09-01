"""Model resolution — turn a name-or-path into a ready detector.

Resolution order (per SPEC):
1. ``*.onnx`` path  -> :class:`ONNXDetector` (ONNX Runtime backend)
2. ``*.pt`` path    -> :class:`YOLODetector` (Ultralytics, ``[yolo]`` extra)
3. known name       -> Ultralytics auto-download to ``~/.cache/orcvision``
                       (``.pt``), then :class:`YOLODetector`

Default model: ``rtdetr-l`` (Apache-2.0 licensed weights).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

KNOWN_NAMES = {"rtdetr-l", "rtdetr-x", "yolov8n", "yolov8s", "yolov8m", "yolo11n"}

CACHE_DIR = Path(os.path.expanduser("~/.cache/orcvision"))


def resolve_model(
    name_or_path: str,
    weights: str | None = None,
    confidence: float = 0.35,
    track: bool = False,
) -> Any:
    """Return a detector object exposing ``infer(frame) -> list[Detection]``."""
    target = weights or name_or_path
    # Expand ~ / env vars so weights paths from YAML configs resolve.
    if any(sep in target for sep in ("/", "~", "\\")):
        target = os.path.expanduser(os.path.expandvars(target))
    lower = target.lower()

    if lower.endswith(".onnx"):
        from orcvision.models.onnx_detector import ONNXDetector

        return ONNXDetector(target, confidence=confidence)

    if lower.endswith(".pt"):
        from orcvision.models.yolo import YOLODetector

        return YOLODetector(target, confidence=confidence, track=track)

    if target in KNOWN_NAMES:
        # Ultralytics downloads the weights on first use. Point its cache at
        # ours so downloads land in ~/.cache/orcvision.
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        from orcvision.models.yolo import YOLODetector

        return YOLODetector(str(CACHE_DIR / f"{target}.pt"), confidence=confidence, track=track)

    raise ValueError(
        f"Could not resolve model {target!r}. Provide a path to a .onnx or .pt "
        f"file, or one of the known names: {sorted(KNOWN_NAMES)}. For .onnx use "
        "`pip install orcvision[cpu]`; for .pt / known names use "
        "`pip install orcvision[yolo]`."
    )

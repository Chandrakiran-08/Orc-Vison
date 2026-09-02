"""Perception adapters — turn *any* vision output into a ``SceneState``.

This module is the brain's only contact surface with perception, and it is
deliberately thin. The brain never learns that YOLO exists: YOLO is simply
one adapter among many.

Three ways in, in increasing order of genericity:

``from_perception_event``
    Native Orc-Vison :class:`~orcvision.events.PerceptionEvent` (ONNX,
    Ultralytics, RealSense — anything the existing pipeline produces).

``from_records``
    Any sequence of plain dicts. Use this for OpenCV blob trackers, custom
    CNNs, segmentation masks reduced to boxes, optical-flow blobs, or a
    microcontroller sending compact JSON over serial.

``from_boxes``
    Bare ``(label, confidence, bbox)`` tuples — the minimum viable input.

All three normalize pixel coordinates to 0..1 fractions of the frame, so a
policy is portable across resolutions and camera modules.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from orcvision.brain.state import ObjectState, SceneState, clamp01, zone_of


@runtime_checkable
class PerceptionAdapter(Protocol):
    """Anything that can produce a normalized scene from a vision output."""

    def to_scene(self, raw: Any) -> SceneState:  # pragma: no cover - protocol
        ...


def _normalize_box(
    bbox: Sequence[float], frame_shape: tuple[int, int]
) -> tuple[tuple[float, float], float]:
    """Return ((cx, cy), area) normalized to 0..1 from a pixel bbox."""
    height, width = frame_shape
    if not width or not height:
        return (0.5, 0.5), 0.0
    x1, y1, x2, y2 = (float(v) for v in bbox)
    cx = clamp01(((x1 + x2) / 2.0) / width)
    cy = clamp01(((y1 + y2) / 2.0) / height)
    area = clamp01((abs(x2 - x1) * abs(y2 - y1)) / float(width * height))
    return (cx, cy), area


def _object_id(label: str, track_id: Any, index: int) -> str:
    """Stable identity for an object across frames.

    A ``track_id`` from an upstream tracker is authoritative. Without one we
    synthesize a per-frame id; the temporal reasoner then falls back to
    position-based association.
    """
    if track_id is not None:
        return f"{label}#{track_id}"
    return f"{label}~{index}"


def from_records(
    records: Iterable[Mapping[str, Any]],
    *,
    timestamp: float,
    frame_shape: tuple[int, int] = (1, 1),
    frame_id: int = 0,
    source: str = "records",
    modality: str = "rgb",
    normalized: bool = False,
) -> SceneState:
    """Build a scene from generic dicts — the model-agnostic entry point.

    Each record needs ``label`` and ``bbox``; ``confidence``, ``track_id``
    and ``depth_m`` are optional. Set ``normalized=True`` if the bboxes are
    already 0..1 fractions (common for embedded vision pipelines that never
    report pixels).
    """
    shape = (1, 1) if normalized else frame_shape
    objects: list[ObjectState] = []
    for index, rec in enumerate(records):
        label = str(rec.get("label", "object"))
        bbox = rec.get("bbox", (0, 0, 0, 0))
        position, size = _normalize_box(bbox, shape)
        depth = rec.get("depth_m")
        objects.append(
            ObjectState(
                object_id=_object_id(label, rec.get("track_id"), index),
                label=label,
                confidence=float(rec.get("confidence", 1.0)),
                position=position,
                size=size,
                depth_m=None if depth is None else float(depth),
                zone=zone_of(position[0]),
                last_seen=timestamp,
                first_seen=timestamp,
            )
        )
    return SceneState(
        timestamp=timestamp,
        objects=objects,
        frame_id=frame_id,
        source=source,
        modality=modality,
    )


def from_boxes(
    boxes: Iterable[Sequence[Any]],
    *,
    timestamp: float,
    frame_shape: tuple[int, int] = (1, 1),
    **kwargs: Any,
) -> SceneState:
    """Build a scene from ``(label, confidence, bbox)`` tuples."""
    records = [{"label": b[0], "confidence": b[1], "bbox": b[2]} for b in boxes]
    return from_records(records, timestamp=timestamp, frame_shape=frame_shape, **kwargs)


def from_perception_event(event: Any) -> SceneState:
    """Adapt a native Orc-Vison ``PerceptionEvent`` into a ``SceneState``.

    Typed as ``Any`` on purpose: the brain must not import the perception
    schema, so this only duck-types the attributes it needs.
    """
    frame_shape = tuple(event.frame_shape)  # (height, width)
    records = [
        {
            "label": d.label,
            "confidence": d.confidence,
            "bbox": d.bbox,
            "track_id": d.track_id,
            "depth_m": d.depth_m,
        }
        for d in event.detections
    ]
    scene = from_records(
        records,
        timestamp=event.timestamp,
        frame_shape=(frame_shape[0], frame_shape[1]),
        frame_id=event.frame_id,
        source=event.source,
        modality=event.modality,
    )
    # Carry any deterministic rule-engine alerts through as context; the
    # decision layer may weigh them, but never depends on them existing.
    scene.meta["alerts"] = list(getattr(event, "alerts", []))
    return scene


class EventAdapter:
    """Adapter object form of :func:`from_perception_event`."""

    def to_scene(self, raw: Any) -> SceneState:
        return from_perception_event(raw)


class RecordAdapter:
    """Adapter object form of :func:`from_records`, with fixed frame shape."""

    def __init__(self, frame_shape: tuple[int, int] = (1, 1), normalized: bool = False) -> None:
        self.frame_shape = frame_shape
        self.normalized = normalized

    def to_scene(self, raw: Any) -> SceneState:
        records = raw.get("detections", []) if isinstance(raw, Mapping) else raw
        timestamp = raw.get("timestamp", 0.0) if isinstance(raw, Mapping) else 0.0
        return from_records(
            records,
            timestamp=float(timestamp),
            frame_shape=self.frame_shape,
            normalized=self.normalized,
        )

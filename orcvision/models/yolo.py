"""Ultralytics YOLO / RT-DETR backend (lazy import, ``[yolo]`` extra).

Ultralytics is AGPL-licensed and heavy, so it is imported lazily and only
pulled in when a user opts into the ``[yolo]`` extra and selects a ``.pt``
weight or a known Ultralytics model name.
"""

from __future__ import annotations

from typing import Any

from orcvision.events import Detection


def _safe_device() -> str:
    """Return a usable device string, falling back to CPU.

    ``torch.cuda.is_available()`` can report True on a GPU whose compute
    capability this torch build has no kernels for (e.g. an older card with
    a newer CUDA wheel). We probe with a tiny op and fall back to CPU on any
    failure so inference never crashes.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return "cpu"
        (torch.zeros(1, device="cuda") + 1).cpu()
        return "cuda"
    except Exception:
        return "cpu"


class YOLODetector:
    """Wrapper over ``ultralytics.YOLO`` producing orcvision Detections."""

    def __init__(
        self,
        weights: str,
        confidence: float = 0.35,
        track: bool = False,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is not installed. Install the optional extra: "
                "`pip install orcvision[yolo]` (note: Ultralytics is AGPL-licensed)."
            ) from exc

        self.model = YOLO(weights)
        self.confidence = confidence
        self.track = track
        self.device = _safe_device()

    def infer(self, frame: Any) -> list[Detection]:
        if self.track:
            results = self.model.track(
                frame, conf=self.confidence, persist=True, verbose=False, device=self.device
            )
        else:
            results = self.model.predict(
                frame, conf=self.confidence, verbose=False, device=self.device
            )

        detections: list[Detection] = []
        if not results:
            return detections
        result = results[0]
        names = result.names
        boxes = result.boxes
        if boxes is None:
            return detections
        for box in boxes:
            cid = int(box.cls[0])
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            track_id = int(box.id[0]) if getattr(box, "id", None) is not None else None
            detections.append(
                Detection(
                    label=names.get(cid, str(cid)),
                    confidence=float(box.conf[0]),
                    bbox=(x1, y1, x2, y2),
                    track_id=track_id,
                    class_id=cid,
                )
            )
        return detections

    def release(self) -> None:
        self.model = None

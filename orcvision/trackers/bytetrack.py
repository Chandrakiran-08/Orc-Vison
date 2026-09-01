"""Lightweight IoU tracker for the ONNX detection path.

The Ultralytics backend (``models/yolo.py``) ships its own native
ByteTrack/BoT-SORT implementation and uses it directly when tracking is
enabled. For the framework-agnostic ONNX path there is no built-in
tracker, so this module provides a minimal greedy IoU associator that
assigns stable ``track_id``s across frames. It is intentionally simple
(no Kalman filter, no re-identification) — good enough to drive the
decision layer's per-track gating without extra dependencies.
"""

from __future__ import annotations

from orcvision.events import Detection


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class IoUTracker:
    """Greedy IoU tracker assigning ``track_id`` in place on detections."""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._next_id = 1
        # track_id -> {"bbox": tuple, "age": int}
        self._tracks: dict[int, dict] = {}

    def update(self, detections: list[Detection]) -> list[Detection]:
        assigned: set[int] = set()
        for det in detections:
            best_id, best_iou = None, self.iou_threshold
            for tid, track in self._tracks.items():
                if tid in assigned:
                    continue
                score = _iou(det.bbox, track["bbox"])
                if score >= best_iou:
                    best_id, best_iou = tid, score
            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
            det.track_id = best_id
            self._tracks[best_id] = {"bbox": det.bbox, "age": 0}
            assigned.add(best_id)

        # Age out unmatched tracks.
        for tid in list(self._tracks):
            if tid not in assigned:
                self._tracks[tid]["age"] += 1
                if self._tracks[tid]["age"] > self.max_age:
                    del self._tracks[tid]
        return detections

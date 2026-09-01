"""Default detector backed by ONNX Runtime.

Supports the two most common exported object-detection output layouts:

* **YOLOv8 / YOLO11** — output ``[1, 4+nc, N]`` (xywh + class scores,
  no separate objectness), coordinates in input-pixel space.
* **RT-DETR** — output ``[1, N, 4+nc]`` (cxcywh normalized 0..1 +
  class scores).

Provider auto-detection prefers ``CUDAExecutionProvider`` and falls back
to ``CPUExecutionProvider`` gracefully.
"""

from __future__ import annotations

import numpy as np

from orcvision.events import Detection
from orcvision.labels import COCO_CLASSES


def available_providers() -> list[str]:
    try:
        import onnxruntime as ort
    except ImportError:  # pragma: no cover - onnxruntime is in the [cpu] extra
        return []
    return list(ort.get_available_providers())


def _preferred_providers() -> list[str]:
    avail = available_providers()
    ordered = []
    if "CUDAExecutionProvider" in avail:
        ordered.append("CUDAExecutionProvider")
    ordered.append("CPUExecutionProvider")
    return ordered


class ONNXDetector:
    """ONNX Runtime object detector with YOLOv8 / RT-DETR postprocessing."""

    def __init__(
        self,
        weights: str,
        confidence: float = 0.35,
        iou: float = 0.45,
        labels: list[str] | None = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "onnxruntime is not installed. Install an inference backend, e.g. "
                "`pip install orcvision[cpu]` or `pip install orcvision[gpu]`."
            ) from exc

        self.confidence = confidence
        self.iou = iou
        self.labels = labels or COCO_CLASSES
        self.session = ort.InferenceSession(weights, providers=_preferred_providers())
        self._input = self.session.get_inputs()[0]
        self.input_name = self._input.name
        # Static square input side (fall back to 640 for dynamic axes).
        shape = self._input.shape
        h = shape[2] if isinstance(shape[2], int) else 640
        w = shape[3] if isinstance(shape[3], int) else 640
        self.input_h, self.input_w = h, w

    # -- preprocessing --------------------------------------------------
    def _letterbox(self, frame: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        import cv2

        ih, iw = frame.shape[:2]
        scale = min(self.input_h / ih, self.input_w / iw)
        nh, nw = int(round(ih * scale)), int(round(iw * scale))
        resized = cv2.resize(frame, (nw, nh))
        canvas = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)
        pad_y = (self.input_h - nh) // 2
        pad_x = (self.input_w - nw) // 2
        canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None]  # NCHW
        return blob, scale, float(pad_x), float(pad_y)

    # -- inference ------------------------------------------------------
    def infer(self, frame: np.ndarray) -> list[Detection]:
        blob, scale, pad_x, pad_y = self._letterbox(frame)
        outputs = self.session.run(None, {self.input_name: blob})
        pred = outputs[0]
        ih, iw = frame.shape[:2]
        boxes, scores, class_ids = self._decode(pred, scale, pad_x, pad_y, iw, ih)
        keep = self._nms(boxes, scores)

        detections: list[Detection] = []
        for i in keep:
            cid = int(class_ids[i])
            label = self.labels[cid] if 0 <= cid < len(self.labels) else str(cid)
            x1, y1, x2, y2 = (int(v) for v in boxes[i])
            detections.append(
                Detection(
                    label=label,
                    confidence=float(scores[i]),
                    bbox=(x1, y1, x2, y2),
                    class_id=cid,
                )
            )
        return detections

    def _decode(self, pred, scale, pad_x, pad_y, iw, ih):
        pred = np.asarray(pred)
        if pred.ndim == 3:
            pred = pred[0]
        # Normalize to (N, features).
        # YOLOv8 export is (4+nc, N); RT-DETR is (N, 4+nc).
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T  # -> (N, 4+nc)

        boxes_xywh = pred[:, :4]
        class_scores = pred[:, 4:]
        class_ids = class_scores.argmax(axis=1)
        scores = class_scores[np.arange(class_scores.shape[0]), class_ids]

        # RT-DETR outputs normalized cxcywh in 0..1; YOLOv8 in input pixels.
        normalized = boxes_xywh.max() <= 1.5
        cx, cy, bw, bh = boxes_xywh.T
        if normalized:
            cx, cy, bw, bh = (
                cx * self.input_w,
                cy * self.input_h,
                bw * self.input_w,
                bh * self.input_h,
            )
        x1 = (cx - bw / 2 - pad_x) / scale
        y1 = (cy - bh / 2 - pad_y) / scale
        x2 = (cx + bw / 2 - pad_x) / scale
        y2 = (cy + bh / 2 - pad_y) / scale
        x1 = np.clip(x1, 0, iw)
        y1 = np.clip(y1, 0, ih)
        x2 = np.clip(x2, 0, iw)
        y2 = np.clip(y2, 0, ih)
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        mask = scores >= self.confidence
        return boxes[mask], scores[mask], class_ids[mask]

    def _nms(self, boxes: np.ndarray, scores: np.ndarray) -> list[int]:
        if len(boxes) == 0:
            return []
        x1, y1, x2, y2 = boxes.T
        areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        order = scores.argsort()[::-1]
        keep: list[int] = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.clip(xx2 - xx1, 0, None)
            h = np.clip(yy2 - yy1, 0, None)
            inter = w * h
            union = areas[i] + areas[order[1:]] - inter
            iou = np.where(union > 0, inter / union, 0)
            order = order[1:][iou <= self.iou]
        return keep

    def release(self) -> None:  # symmetry with sensors; nothing to free
        self.session = None

"""Pydantic schemas for the perception event stream."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Detection(BaseModel):
    """A single detected object in a frame."""

    label: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    track_id: int | None = None
    depth_m: float | None = None  # sensor depth, monocular estimate, or null
    class_id: int | None = None


class PerceptionEvent(BaseModel):
    """One frame's worth of perception output, ready to serialize to a sink."""

    timestamp: float
    frame_id: int
    source: str  # e.g. "camera:0", "realsense:0"
    modality: str  # from SensorFrame: rgb | rgbd | thermal | stereo
    frame_shape: tuple[int, int]  # (height, width)
    detections: list[Detection] = Field(default_factory=list)
    alerts: list[str] = Field(default_factory=list)  # populated by decision layer
    # Optional autonomous-brain output (orcvision.brain). Null unless the
    # brain layer is enabled, so existing consumers — including the
    # microcontroller sketches, which read `alerts` — are unaffected.
    decision: dict | None = None

    def to_json(self) -> str:
        return self.model_dump_json()

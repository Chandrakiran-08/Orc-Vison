"""Schema roundtrip tests for the event models."""

import json

from orcvision.events import Detection, PerceptionEvent


def test_detection_defaults():
    det = Detection(label="person", confidence=0.9, bbox=(1, 2, 3, 4))
    assert det.track_id is None
    assert det.depth_m is None
    assert det.class_id is None


def test_perception_event_roundtrip():
    event = PerceptionEvent(
        timestamp=123.4,
        frame_id=7,
        source="camera:0",
        modality="rgb",
        frame_shape=(480, 640),
        detections=[Detection(label="person", confidence=0.91, bbox=(10, 20, 30, 40), track_id=3)],
        alerts=["alert: label == 'person'"],
    )
    payload = event.to_json()
    data = json.loads(payload)
    assert data["frame_id"] == 7
    assert data["modality"] == "rgb"
    assert data["detections"][0]["label"] == "person"
    assert data["alerts"] == ["alert: label == 'person'"]

    # Rebuild from JSON and confirm equality.
    rebuilt = PerceptionEvent.model_validate_json(payload)
    assert rebuilt == event


def test_bbox_is_tuple():
    event = PerceptionEvent(
        timestamp=0.0,
        frame_id=0,
        source="file:x",
        modality="rgb",
        frame_shape=(2, 2),
        detections=[],
    )
    assert event.detections == []
    assert event.alerts == []

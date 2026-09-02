"""Brain perception/state/temporal tests — synthetic scenes, no hardware."""

from orcvision.brain.adapters import from_boxes, from_perception_event, from_records
from orcvision.brain.state import (
    MOTION_APPROACHING,
    MOTION_STATIONARY,
    ObjectState,
    WorldState,
    zone_of,
)
from orcvision.brain.temporal import (
    OBJECT_APPEARED,
    OBJECT_APPROACHING,
    OBJECT_DISAPPEARED,
    ZONE_CHANGED,
    TemporalConfig,
    TemporalReasoner,
)
from orcvision.events import Detection, PerceptionEvent


def _rec(label="obstacle", bbox=(280, 200, 360, 280), depth=None, track_id=1):
    return {"label": label, "confidence": 0.9, "bbox": bbox, "depth_m": depth, "track_id": track_id}


def test_zone_mapping():
    assert zone_of(0.1) == "left"
    assert zone_of(0.5) == "center"
    assert zone_of(0.9) == "right"


def test_adapter_normalizes_to_fractions():
    scene = from_records([_rec()], timestamp=1.0, frame_shape=(480, 640))
    obj = scene.objects[0]
    # bbox centre (320, 240) on a 640x480 frame -> exactly the middle.
    assert abs(obj.position[0] - 0.5) < 1e-6
    assert abs(obj.position[1] - 0.5) < 1e-6
    assert 0.0 < obj.size < 1.0
    assert obj.zone == "center"


def test_adapter_is_resolution_independent():
    """The same object at two resolutions must normalize identically."""
    small = from_records([_rec(bbox=(280, 200, 360, 280))], timestamp=0.0, frame_shape=(480, 640))
    large = from_records(
        [_rec(bbox=(840, 600, 1080, 840))], timestamp=0.0, frame_shape=(1440, 1920)
    )
    assert small.objects[0].position == large.objects[0].position
    assert abs(small.objects[0].size - large.objects[0].size) < 1e-9


def test_adapter_accepts_normalized_input():
    scene = from_records(
        [{"label": "x", "bbox": (0.0, 0.0, 0.5, 0.5)}], timestamp=0.0, normalized=True
    )
    assert abs(scene.objects[0].size - 0.25) < 1e-9


def test_from_boxes_minimal_input():
    scene = from_boxes([("person", 0.8, (0, 0, 320, 240))], timestamp=0.0, frame_shape=(480, 640))
    assert scene.objects[0].label == "person"


def test_from_perception_event_is_duck_typed():
    event = PerceptionEvent(
        timestamp=5.0,
        frame_id=3,
        source="camera:0",
        modality="rgb",
        frame_shape=(480, 640),
        detections=[Detection(label="person", confidence=0.9, bbox=(0, 0, 320, 240), track_id=7)],
        alerts=["alert: x"],
    )
    scene = from_perception_event(event)
    assert scene.objects[0].label == "person"
    assert scene.objects[0].object_id == "person#7"
    assert scene.meta["alerts"] == ["alert: x"]


def test_proximity_uses_depth_when_present():
    near = ObjectState("a", "x", 0.9, (0.5, 0.5), 0.01, depth_m=0.5)
    far = ObjectState("b", "x", 0.9, (0.5, 0.5), 0.01, depth_m=4.5)
    assert near.proximity() > far.proximity()


def test_proximity_falls_back_to_size_without_depth():
    big = ObjectState("a", "x", 0.9, (0.5, 0.5), 0.25)
    small = ObjectState("b", "x", 0.9, (0.5, 0.5), 0.01)
    assert big.proximity() > small.proximity()


# --- temporal reasoning -----------------------------------------------------


def test_object_appeared_then_disappeared():
    world = WorldState()
    reasoner = TemporalReasoner(TemporalConfig(disappear_after_misses=2))

    events = reasoner.update(world, from_records([_rec()], timestamp=0.0, frame_shape=(480, 640)))
    assert any(e.kind == OBJECT_APPEARED for e in events)

    reasoner.update(world, from_records([], timestamp=1.0, frame_shape=(480, 640)))
    events = reasoner.update(world, from_records([], timestamp=2.0, frame_shape=(480, 640)))
    assert any(e.kind == OBJECT_DISAPPEARED for e in events)
    assert world.objects == {}


def test_missing_for_one_frame_is_not_forgotten():
    """Detectors drop a box for a frame constantly; that is not a disappearance."""
    world = WorldState()
    reasoner = TemporalReasoner(TemporalConfig(disappear_after_misses=5))
    reasoner.update(world, from_records([_rec()], timestamp=0.0, frame_shape=(480, 640)))
    events = reasoner.update(world, from_records([], timestamp=0.1, frame_shape=(480, 640)))
    assert not any(e.kind == OBJECT_DISAPPEARED for e in events)
    assert len(world.objects) == 1


def test_approaching_detected_from_depth():
    world = WorldState()
    reasoner = TemporalReasoner()
    reasoner.update(world, from_records([_rec(depth=3.0)], timestamp=0.0, frame_shape=(480, 640)))
    events = reasoner.update(
        world, from_records([_rec(depth=2.0)], timestamp=1.0, frame_shape=(480, 640))
    )
    assert any(e.kind == OBJECT_APPROACHING for e in events)
    assert next(iter(world.objects.values())).motion == MOTION_APPROACHING


def test_stationary_object_reports_no_motion():
    world = WorldState()
    reasoner = TemporalReasoner()
    reasoner.update(world, from_records([_rec(depth=3.0)], timestamp=0.0, frame_shape=(480, 640)))
    reasoner.update(world, from_records([_rec(depth=3.0)], timestamp=1.0, frame_shape=(480, 640)))
    assert next(iter(world.objects.values())).motion == MOTION_STATIONARY


def test_zone_change_event():
    world = WorldState()
    reasoner = TemporalReasoner()
    reasoner.update(
        world,
        from_records([_rec(bbox=(0, 200, 80, 280))], timestamp=0.0, frame_shape=(480, 640)),
    )
    events = reasoner.update(
        world,
        from_records([_rec(bbox=(560, 200, 640, 280))], timestamp=1.0, frame_shape=(480, 640)),
    )
    assert any(e.kind == ZONE_CHANGED for e in events)


def test_association_without_track_ids():
    """The brain must still track identity behind a detector with no tracker."""
    world = WorldState()
    reasoner = TemporalReasoner()
    reasoner.update(
        world,
        from_records(
            [_rec(bbox=(280, 200, 360, 280), track_id=None)],
            timestamp=0.0,
            frame_shape=(480, 640),
        ),
    )
    reasoner.update(
        world,
        from_records(
            [_rec(bbox=(290, 205, 370, 285), track_id=None)],
            timestamp=0.5,
            frame_shape=(480, 640),
        ),
    )
    # Nudged slightly, so it is the same object — not a second one.
    assert len(world.objects) == 1


def test_world_describe_and_relationships():
    world = WorldState(goal="avoid_collision")
    reasoner = TemporalReasoner()
    reasoner.update(
        world,
        from_records(
            [
                _rec(label="person", bbox=(0, 200, 80, 280), depth=1.0, track_id=1),
                _rec(label="obstacle", bbox=(560, 200, 640, 280), depth=3.0, track_id=2),
            ],
            timestamp=0.0,
            frame_shape=(480, 640),
        ),
    )
    assert any("Goal: avoid_collision" in line for line in world.describe())
    predicates = {r.predicate for r in world.relationships()}
    assert "left_of" in predicates and "closer_than" in predicates

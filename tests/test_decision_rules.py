"""Decision-layer tests against synthetic events (no hardware)."""

import pytest

from orcvision.decision.rules import Rule, RuleEngine, SafeExpressionError
from orcvision.events import Detection, PerceptionEvent


def _event(detections):
    return PerceptionEvent(
        timestamp=0.0,
        frame_id=0,
        source="test",
        modality="rgb",
        frame_shape=(10, 10),
        detections=detections,
    )


def test_simple_match():
    rule = Rule(when="label == 'person' and confidence > 0.8")
    assert rule.matches(Detection(label="person", confidence=0.9, bbox=(0, 0, 1, 1)))
    assert not rule.matches(Detection(label="person", confidence=0.5, bbox=(0, 0, 1, 1)))
    assert not rule.matches(Detection(label="car", confidence=0.99, bbox=(0, 0, 1, 1)))


def test_is_not_null():
    rule = Rule(when="track_id is not None")
    assert rule.matches(Detection(label="x", confidence=1.0, bbox=(0, 0, 1, 1), track_id=7))
    assert not rule.matches(Detection(label="x", confidence=1.0, bbox=(0, 0, 1, 1)))


def test_in_operator():
    rule = Rule(when="label in ['person', 'car']")
    assert rule.matches(Detection(label="car", confidence=1.0, bbox=(0, 0, 1, 1)))
    assert not rule.matches(Detection(label="dog", confidence=1.0, bbox=(0, 0, 1, 1)))


def test_engine_populates_alerts():
    engine = RuleEngine.from_config(
        [{"when": "label == 'person' and confidence > 0.8", "action": "alert"}]
    )
    event = _event([Detection(label="person", confidence=0.95, bbox=(0, 0, 1, 1))])
    engine.apply(event, now=0.0)
    assert len(event.alerts) == 1
    assert event.alerts[0].startswith("alert:")


def test_no_alert_when_no_match():
    engine = RuleEngine.from_config([{"when": "label == 'cat'"}])
    event = _event([Detection(label="person", confidence=0.9, bbox=(0, 0, 1, 1))])
    engine.apply(event, now=0.0)
    assert event.alerts == []


def test_cooldown_suppresses_repeat():
    engine = RuleEngine.from_config([{"when": "label == 'person'", "cooldown_s": 30}])
    det = [Detection(label="person", confidence=0.9, bbox=(0, 0, 1, 1))]
    e1 = _event(det)
    engine.apply(e1, now=100.0)
    assert len(e1.alerts) == 1
    e2 = _event(det)
    engine.apply(e2, now=110.0)  # within cooldown
    assert e2.alerts == []
    e3 = _event(det)
    engine.apply(e3, now=140.0)  # cooldown elapsed
    assert len(e3.alerts) == 1


def test_min_consecutive_frames():
    engine = RuleEngine.from_config([{"when": "label == 'person'", "min_consecutive_frames": 3}])
    for i in range(2):
        e = _event([Detection(label="person", confidence=0.9, bbox=(0, 0, 1, 1), track_id=1)])
        engine.apply(e, now=float(i))
        assert e.alerts == []
    e = _event([Detection(label="person", confidence=0.9, bbox=(0, 0, 1, 1), track_id=1)])
    engine.apply(e, now=3.0)
    assert len(e.alerts) == 1


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('echo hi')",
        "open('/etc/passwd').read()",
        "label.__class__",
        "(lambda: 1)()",
        "[x for x in range(3)]",
    ],
)
def test_unsafe_expressions_rejected(expr):
    with pytest.raises(SafeExpressionError):
        Rule(when=expr)


def test_unknown_variable_rejected():
    rule = Rule(when="foobar == 1")
    with pytest.raises(SafeExpressionError):
        rule.matches(Detection(label="x", confidence=1.0, bbox=(0, 0, 1, 1)))

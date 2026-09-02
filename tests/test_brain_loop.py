"""End-to-end brain loop tests.

The headline test here is :func:`test_memory_changes_the_decision` — the
property that separates this from a YOLO wrapper: identical perception
input, different action, because the brain remembers what happened last
time.
"""

import pytest

from orcvision.brain import VisionBrain
from orcvision.brain.actions import AVOID, MOVE, STOP, Action, ActionExecutor
from orcvision.brain.brain import BrainConfig
from orcvision.events import Detection, PerceptionEvent


def _frame(t: float, depth: float, label: str = "obstacle") -> dict:
    """One frame of an obstacle straight ahead at ``depth`` metres."""
    return {
        "timestamp": t,
        "frame_shape": (480, 640),
        "detections": [
            {
                "label": label,
                "confidence": 0.9,
                "bbox": (280, 200, 360, 300),
                "track_id": 1,
                "depth_m": depth,
            }
        ],
    }


def _approach(brain: VisionBrain, base: float = 0.0) -> None:
    """Drive a closing obstacle through the brain."""
    for i, depth in enumerate([4.0, 3.0, 2.0, 1.2]):
        brain.observe(_frame(base + i * 0.5, depth))


# --- the core claim ---------------------------------------------------------


def test_memory_changes_the_decision():
    """Same scene, different action — because the first choice failed before."""
    brain = VisionBrain(goal="avoid_collision")

    _approach(brain)
    first = brain.decide()
    assert first.action.type == AVOID

    # Report that avoiding did not work out.
    brain.feedback(success=False, decision=first, note="clipped the obstacle")
    brain.learn()

    # Identical perception input the second time around.
    _approach(brain, base=100.0)
    second = brain.decide()

    assert second.action.type != first.action.type
    assert second.action.type == STOP
    # And the brain can say why it changed its mind: the evidence sits on
    # the action it rejected, not on the one it picked.
    assert any(action == AVOID for action, _ in second.demoted)
    assert "previously failed" in second.explain()


def test_success_reinforces_the_same_choice():
    brain = VisionBrain(goal="avoid_collision")
    _approach(brain)
    first = brain.decide()
    brain.feedback(success=True, decision=first)
    brain.learn()

    _approach(brain, base=100.0)
    assert brain.decide().action.type == first.action.type


def test_experience_is_situation_scoped_not_global():
    """A lesson learned about one situation must not leak into a different one."""
    brain = VisionBrain(goal="avoid_collision")
    _approach(brain)
    failed = brain.decide()
    brain.feedback(success=False, decision=failed)
    brain.learn()

    # A different situation: nothing in view at all.
    brain.observe({"timestamp": 500.0, "frame_shape": (480, 640), "detections": []})
    decision = brain.decide()
    assert decision.situation != failed.situation
    assert not any("previously failed" in r.text for r in decision.reasons)


# --- the loop ---------------------------------------------------------------


def test_full_loop_runs_and_records():
    brain = VisionBrain(goal="avoid_collision")
    _approach(brain)
    decision = brain.decide()
    result = brain.execute(decision)
    assert result["ok"] is True
    assert brain.world.last_action == decision.action.type
    outcome = brain.feedback(success=True)
    assert outcome.action_type == decision.action.type
    assert brain.learn()["updated"] is True


def test_step_is_observe_decide_execute():
    brain = VisionBrain(goal="avoid_collision")
    decision = brain.step(_frame(0.0, 1.0))
    assert isinstance(decision.action, Action)
    assert brain.executor.history == [decision.action]


def test_brain_accepts_a_native_perception_event():
    """Model-agnostic in practice: the existing pipeline's event just works."""
    brain = VisionBrain(goal="avoid_collision")
    event = PerceptionEvent(
        timestamp=1.0,
        frame_id=1,
        source="camera:0",
        modality="rgb",
        frame_shape=(480, 640),
        detections=[
            Detection(label="person", confidence=0.95, bbox=(280, 200, 360, 300), depth_m=0.8)
        ],
    )
    brain.observe(event)
    assert brain.world.visible()[0].label == "person"
    assert brain.decide().action.type in {AVOID, STOP}


def test_brain_accepts_bare_normalized_records():
    """A microcontroller sending compact JSON needs no pixel dimensions."""
    brain = VisionBrain(goal="avoid_collision")
    brain.observe([{"label": "obstacle", "confidence": 0.9, "bbox": (0.4, 0.4, 0.6, 0.6)}])
    assert brain.world.visible()[0].label == "obstacle"


def test_unadaptable_input_is_rejected():
    with pytest.raises(TypeError):
        VisionBrain().observe(42)


def test_decide_without_observation_does_not_crash():
    assert VisionBrain().decide().action.type in {MOVE, STOP, "WAIT", "SIGNAL", "TRACK", AVOID}


def test_execute_before_decide_raises():
    with pytest.raises(RuntimeError):
        VisionBrain().execute()


def test_feedback_before_decide_raises():
    with pytest.raises(RuntimeError):
        VisionBrain().feedback(success=True)


def test_learn_without_outcome_is_a_noop():
    brain = VisionBrain()
    brain.observe(_frame(0.0, 1.0))
    brain.decide()
    assert brain.learn()["updated"] is False


# --- goals, executors, explainability, persistence --------------------------


def test_goal_switch_changes_the_situation_key():
    brain = VisionBrain(goal="patrol")
    brain.observe(_frame(0.0, 1.0))
    patrol = brain.decide().situation
    brain.set_goal("avoid_collision")
    brain.observe(_frame(1.0, 1.0))
    assert brain.decide().situation != patrol


def test_custom_executor_receives_the_action():
    class RecordingExecutor(ActionExecutor):
        def __init__(self):
            super().__init__()
            self.driven = []

        def execute(self, action):
            self.driven.append(action.type)
            return {"ok": True}

    executor = RecordingExecutor()
    brain = VisionBrain(goal="avoid_collision", executor=executor)
    brain.step(_frame(0.0, 1.0))
    assert executor.driven


def test_explain_is_available_and_readable():
    brain = VisionBrain(goal="avoid_collision")
    assert "No decision" in brain.explain()
    _approach(brain)
    brain.decide()
    assert "Decision:" in brain.explain()


def test_world_description_reads_like_a_world_model():
    brain = VisionBrain(goal="avoid_collision")
    _approach(brain)
    text = "\n".join(brain.describe_world())
    assert "Goal: avoid_collision" in text
    assert "obstacle" in text


def test_save_and_load_restores_experience(tmp_path):
    brain = VisionBrain(goal="avoid_collision")
    _approach(brain)
    decision = brain.decide()
    brain.feedback(success=False, decision=decision)
    brain.learn()
    brain.save(tmp_path)

    revived = VisionBrain(goal="avoid_collision")
    revived.load(tmp_path)
    _approach(revived)
    # The lesson survived the power cycle.
    assert revived.decide().action.type == STOP


def test_memory_footprint_stays_bounded_over_a_long_run():
    """Continuous operation must not grow without limit on an embedded board."""
    brain = VisionBrain(config=BrainConfig(working_capacity=32, longterm_capacity=16))
    for i in range(400):
        brain.observe(_frame(i * 0.1, 1.0 + (i % 5) * 0.5, label=f"obj{i % 40}"))
        decision = brain.decide()
        brain.feedback(success=i % 2 == 0, decision=decision)
        brain.learn()
    assert len(brain.memory.working) <= 32
    assert len(brain.memory.longterm) <= 16


# --- edge-first guarantees --------------------------------------------------


def test_brain_pulls_in_no_third_party_dependencies():
    """The brain must stay deployable on constrained hardware.

    Importing it may not drag in pydantic, numpy, torch, opencv or any
    other heavyweight dependency — those belong to the perception half.
    """
    import subprocess
    import sys

    probe = (
        "import sys, orcvision.brain;"
        "heavy=[m for m in ['numpy','torch','cv2','pydantic','yaml','typer',"
        "'onnxruntime','ultralytics','paho'] if m in sys.modules];"
        "print(','.join(heavy))"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "", f"brain leaked heavy imports: {out.stdout.strip()}"


def test_perception_schemas_still_importable_from_package_root():
    """Laziness must not break the existing public API."""
    from orcvision import Detection, PerceptionEvent

    assert Detection(label="x", confidence=1.0, bbox=(0, 0, 1, 1)).label == "x"
    assert (
        PerceptionEvent(
            timestamp=0.0, frame_id=0, source="s", modality="rgb", frame_shape=(1, 1)
        ).decision
        is None
    )

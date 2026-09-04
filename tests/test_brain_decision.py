"""Decision, policy, constraint and learning tests."""

import pytest

from orcvision.brain.actions import AVOID, MOVE, STOP, Action
from orcvision.brain.adapters import from_records
from orcvision.brain.constraints import ConfidenceConstraint, ProximityConstraint
from orcvision.brain.decision import (
    DecisionContext,
    UtilityDecisionEngine,
    default_weights,
    situation_key,
)
from orcvision.brain.memory import Memory
from orcvision.brain.policy import LinearPolicy
from orcvision.brain.state import SceneState, WorldState
from orcvision.brain.temporal import TemporalReasoner


def _ctx(depth=1.0, label="obstacle", bbox=(280, 200, 360, 280), goal="avoid_collision"):
    world = WorldState(goal=goal)
    scene = from_records(
        [{"label": label, "confidence": 0.9, "bbox": bbox, "depth_m": depth, "track_id": 1}],
        timestamp=0.0,
        frame_shape=(480, 640),
    )
    events = TemporalReasoner().update(world, scene)
    return DecisionContext(
        scene=scene, world=world, events=events, memory=Memory(), goal=goal, now=0.0
    )


# --- policy -----------------------------------------------------------------


def test_linear_policy_scores_and_explains():
    policy = LinearPolicy({"STOP|a": 2.0, "STOP|b": -1.0})
    score, contributions = policy.score("STOP", {"a": 1.0, "b": 0.5})
    assert score == pytest.approx(1.5)
    assert contributions == {"a": 2.0, "b": -0.5}


def test_unknown_feature_contributes_nothing():
    policy = LinearPolicy({})
    score, _ = policy.score("STOP", {"unseen": 1.0})
    assert score == 0.0


def test_reinforce_moves_weights_toward_reward():
    policy = LinearPolicy({}, learning_rate=0.5)
    policy.reinforce("STOP", {"x": 1.0}, reward=1.0)
    assert policy.weights["STOP|x"] == pytest.approx(0.5)
    policy.reinforce("STOP", {"x": 1.0}, reward=-1.0)
    assert policy.weights["STOP|x"] == pytest.approx(0.0)


def test_imitation_learning_corrects_only_when_wrong():
    policy = LinearPolicy({"MOVE|x": 1.0}, learning_rate=0.5)
    candidates = {"MOVE": {"x": 1.0}, "STOP": {"x": 1.0}}
    assert policy.learn_from_example(candidates, "STOP") is True  # was wrong
    # Repeat until the expert action wins, then no further correction.
    for _ in range(10):
        policy.learn_from_example(candidates, "STOP")
    assert policy.rank(candidates)[0][0] == "STOP"
    assert policy.learn_from_example(candidates, "STOP") is False


def test_fit_converges_on_separable_data():
    policy = LinearPolicy({}, learning_rate=0.2)
    dataset = [
        ({"MOVE": {"clear": 1.0}, "STOP": {"clear": 0.0}}, "MOVE"),
        ({"MOVE": {"clear": 0.0}, "STOP": {"clear": 1.0}}, "STOP"),
    ]
    report = policy.fit(dataset, epochs=25)
    assert report["history"][-1]["accuracy"] == 1.0


def test_unknown_expert_action_rejected():
    with pytest.raises(KeyError):
        LinearPolicy({}).learn_from_example({"MOVE": {}}, "FLY")


def test_policy_save_load_roundtrip(tmp_path):
    policy = LinearPolicy({"STOP|a": 1.25}, learning_rate=0.3)
    policy.save(tmp_path / "p.json")
    loaded = LinearPolicy.load(tmp_path / "p.json")
    assert loaded.weights == policy.weights
    assert loaded.learning_rate == 0.3


# --- decision engine --------------------------------------------------------


def test_situation_key_is_low_cardinality():
    near = situation_key(_ctx(depth=0.5))
    far = situation_key(_ctx(depth=4.8))
    assert near.endswith("near") and far.endswith("far")
    # Same situation from slightly different pixels must collapse to one key.
    assert situation_key(_ctx(depth=0.5)) == situation_key(_ctx(depth=0.6))


def test_engine_avoids_a_closing_hazard():
    engine = UtilityDecisionEngine()
    decision = engine.decide(_ctx(depth=1.0))
    assert decision.action.type == AVOID
    assert decision.action.parameters["direction"] in ("left", "right")
    assert decision.reasons


def test_engine_moves_when_scene_is_clear():
    world = WorldState(goal="patrol")
    ctx = DecisionContext(
        scene=SceneState(timestamp=0.0), world=world, memory=Memory(), goal="patrol", now=0.0
    )
    assert UtilityDecisionEngine().decide(ctx).action.type == MOVE


def test_avoid_steers_away_from_hazard_side():
    left = UtilityDecisionEngine().decide(_ctx(bbox=(0, 200, 80, 280)))
    right = UtilityDecisionEngine().decide(_ctx(bbox=(560, 200, 640, 280)))
    assert left.action.parameters["direction"] == "right"
    assert right.action.parameters["direction"] == "left"


def test_decision_is_explainable_and_serializable():
    decision = UtilityDecisionEngine().decide(_ctx(depth=1.0))
    text = decision.explain()
    assert "Decision:" in text and "Reason:" in text
    payload = decision.to_dict()
    assert payload["action"]["type"] == AVOID
    assert isinstance(payload["reasons"], list)


# --- safety constraints -----------------------------------------------------


def test_proximity_constraint_vetoes_advancing_into_a_hazard():
    ctx = _ctx(depth=0.5)
    assert ProximityConstraint().veto(MOVE, ctx) is not None
    assert ProximityConstraint().veto(STOP, ctx) is None  # stopping is always allowed


def test_proximity_constraint_allows_movement_when_far():
    assert ProximityConstraint().veto(MOVE, _ctx(depth=4.9)) is None


def test_confidence_constraint_blocks_low_confidence_targets():
    world = WorldState()
    scene = from_records(
        [{"label": "cup", "confidence": 0.1, "bbox": (0, 0, 10, 10), "track_id": 1}],
        timestamp=0.0,
        frame_shape=(480, 640),
    )
    TemporalReasoner().update(world, scene)
    ctx = DecisionContext(
        scene=scene,
        world=world,
        memory=Memory(),
        now=0.0,
        target_labels=frozenset({"cup"}),
    )
    assert ConfidenceConstraint().veto("GRASP", ctx) is not None


def test_learned_policy_cannot_unlock_a_vetoed_action():
    """The safety floor must survive arbitrarily bad training."""
    policy = LinearPolicy(default_weights())
    # Train the policy into believing MOVE is always best.
    for _ in range(50):
        policy.reinforce(MOVE, {"bias": 1.0, "path_clear": 1.0}, reward=5.0)
    engine = UtilityDecisionEngine(policy=policy)
    decision = engine.decide(_ctx(depth=0.4))  # hazard very close
    assert decision.action.type != MOVE
    assert any(action == MOVE for action, _ in decision.vetoed)


def test_safety_fallback_when_everything_is_vetoed():
    class VetoEverything:
        def veto(self, action_type, ctx):
            return f"{action_type} forbidden"

    engine = UtilityDecisionEngine(constraints=[VetoEverything()], safe_action=STOP)
    decision = engine.decide(_ctx(depth=1.0))
    assert decision.action.type == STOP
    assert decision.safety_fallback is True


def test_constraints_can_be_disabled_explicitly():
    engine = UtilityDecisionEngine(constraints=[])
    assert engine.decide(_ctx(depth=0.4)).vetoed == []


def test_action_is_serializable():
    action = Action(MOVE, {"direction": "forward", "speed": 0.5})
    assert action.to_dict() == {
        "type": "MOVE",
        "parameters": {"direction": "forward", "speed": 0.5},
    }
    assert "direction=forward" in str(action)

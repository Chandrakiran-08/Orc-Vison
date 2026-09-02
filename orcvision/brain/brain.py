"""``VisionBrain`` — the facade researchers actually touch.

The whole loop in the API's shape::

    brain = VisionBrain(goal="avoid_collision")
    brain.observe(vision_output)   # perception  -> state -> temporal -> memory
    decision = brain.decide()      # state + memory + goal -> action
    brain.execute(decision)        # action out to hardware (or a dry run)
    brain.feedback(success=False)  # outcome -> memory
    brain.learn()                  # outcome -> policy weights

Everything under this facade is replaceable: pass your own adapter,
decision engine, policy, memory or executor to the constructor. Nothing in
this file imports a detector, a framework, or a network client, so the
brain runs offline and stays cheap enough for edge hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from orcvision.brain.actions import DEFAULT_ACTIONS, Action, ActionExecutor
from orcvision.brain.adapters import from_perception_event, from_records
from orcvision.brain.constraints import Constraint
from orcvision.brain.decision import (
    Decision,
    DecisionContext,
    DecisionEngine,
    UtilityDecisionEngine,
)
from orcvision.brain.feedback import Outcome, record_outcome
from orcvision.brain.memory import KIND_DECISION, KIND_EVENT, Memory
from orcvision.brain.policy import LinearPolicy
from orcvision.brain.state import SceneState, WorldState
from orcvision.brain.temporal import BrainEvent, TemporalConfig, TemporalReasoner


@dataclass
class BrainConfig:
    """Tuning knobs, all with edge-sane defaults."""

    goal: str = "idle"
    actions: tuple[str, ...] = DEFAULT_ACTIONS
    hazard_labels: frozenset[str] = frozenset({"person", "obstacle", "vehicle", "car"})
    target_labels: frozenset[str] = frozenset()
    max_range_m: float = 5.0
    safe_action: str = "STOP"  # taken when every candidate is vetoed
    working_capacity: int = 64
    working_retention_s: float = 30.0
    longterm_capacity: int = 256
    longterm_half_life_s: float = 600.0
    learning_rate: float = 0.1
    temporal: TemporalConfig = field(default_factory=TemporalConfig)


class VisionBrain:
    """A lightweight autonomous decision layer over any perception source."""

    def __init__(
        self,
        config: BrainConfig | None = None,
        *,
        goal: str | None = None,
        engine: DecisionEngine | None = None,
        policy: LinearPolicy | None = None,
        memory: Memory | None = None,
        executor: ActionExecutor | None = None,
        constraints: list[Constraint] | None = None,
    ) -> None:
        self.config = config or BrainConfig()
        if goal is not None:
            self.config.goal = goal

        self.memory = memory or Memory()
        self.memory.working.capacity = self.config.working_capacity
        self.memory.working.retention_s = self.config.working_retention_s
        self.memory.longterm.capacity = self.config.longterm_capacity
        self.memory.longterm.half_life_s = self.config.longterm_half_life_s

        self.policy = policy or LinearPolicy(learning_rate=self.config.learning_rate)
        if not self.policy.weights:
            from orcvision.brain.decision import default_weights

            self.policy.weights = default_weights(self.config.actions)

        self.engine: DecisionEngine = engine or UtilityDecisionEngine(
            policy=self.policy,
            actions=self.config.actions,
            max_range_m=self.config.max_range_m,
            constraints=constraints,
            safe_action=self.config.safe_action,
        )
        self.executor = executor or ActionExecutor()
        self.temporal = TemporalReasoner(self.config.temporal)

        self.world = WorldState(goal=self.config.goal)
        self.scene: SceneState | None = None
        self.events: list[BrainEvent] = []
        self.last_decision: Decision | None = None
        self._pending: list[tuple[Decision, float]] = []

    # --- goals -------------------------------------------------------------

    @property
    def goal(self) -> str:
        return self.world.goal

    def set_goal(self, goal: str) -> None:
        """Change what the brain is trying to achieve."""
        self.world.goal = goal
        self.config.goal = goal

    # --- perception --------------------------------------------------------

    def observe(self, vision_output: Any, *, timestamp: float | None = None) -> SceneState:
        """Take any vision output and fold it into state, time and memory.

        Accepts a ``PerceptionEvent``, a ready-made ``SceneState``, or a
        plain list of detection dicts — whatever your perception stack
        produces.
        """
        scene = self._to_scene(vision_output, timestamp)
        self.scene = scene
        self.events = self.temporal.update(self.world, scene)
        self.update_memory()
        return scene

    def _to_scene(self, raw: Any, timestamp: float | None) -> SceneState:
        if isinstance(raw, SceneState):
            return raw
        if hasattr(raw, "detections") and hasattr(raw, "frame_shape"):
            return from_perception_event(raw)
        if isinstance(raw, dict):
            return from_records(
                raw.get("detections", []),
                timestamp=float(raw.get("timestamp", timestamp or 0.0)),
                frame_shape=tuple(raw.get("frame_shape", (1, 1))),
                normalized=bool(raw.get("normalized", False)),
            )
        if isinstance(raw, list):
            return from_records(raw, timestamp=timestamp or 0.0, normalized=True)
        raise TypeError(f"Cannot adapt {type(raw).__name__} into a SceneState")

    def update_memory(self) -> None:
        """Commit this frame's changes to working memory and age both stores."""
        if self.scene is None:
            return
        now = self.scene.timestamp
        for event in self.events:
            self.memory.working.add(KIND_EVENT, event.describe(), now, label=event.label)
        self.memory.prune(now)

    # --- decision ----------------------------------------------------------

    def context(self) -> DecisionContext:
        scene = self.scene or SceneState(timestamp=self.world.updated_at)
        return DecisionContext(
            scene=scene,
            world=self.world,
            events=self.events,
            memory=self.memory,
            goal=self.world.goal,
            now=scene.timestamp,
            hazard_labels=self.config.hazard_labels,
            target_labels=self.config.target_labels,
        )

    def decide(self) -> Decision:
        """Choose an action from current state, memory and goal."""
        decision = self.engine.decide(self.context())
        self.last_decision = decision
        now = self.context().now
        self.memory.working.add(
            KIND_DECISION,
            f"{decision.action} ({decision.score:+.2f})",
            now,
            label=decision.action.type,
        )
        self._pending.append((decision, now))
        return decision

    def execute(self, decision: Decision | None = None) -> dict[str, Any]:
        """Hand the action to the executor (hardware lives outside the brain)."""
        decision = decision or self.last_decision
        if decision is None:
            raise RuntimeError("no decision to execute — call decide() first")
        result = self.executor.execute(decision.action)
        self.world.last_action = decision.action.type
        return result

    # --- feedback & learning ----------------------------------------------

    def feedback(
        self,
        success: bool,
        *,
        decision: Decision | None = None,
        reward: float | None = None,
        note: str = "",
        timestamp: float | None = None,
    ) -> Outcome:
        """Report how an action turned out. This is what changes future choices."""
        decision = decision or self.last_decision
        if decision is None:
            raise RuntimeError("no decision to give feedback on")
        now = timestamp if timestamp is not None else self.context().now
        outcome = record_outcome(self.memory, decision, success, now, reward=reward, note=note)
        self._pending = [(d, t) for d, t in self._pending if d is not decision]
        self._last_outcome = (decision, outcome)
        return outcome

    def learn(self) -> dict[str, Any]:
        """Fold the most recent outcome into the policy weights.

        Memory already changed the next decision the instant feedback landed;
        this generalizes the lesson into the policy itself.
        """
        pair = getattr(self, "_last_outcome", None)
        if pair is None:
            return {"updated": False, "reason": "no outcome recorded"}
        decision, outcome = pair
        features = decision.features.get(outcome.action_type, {})
        self.policy.reinforce(outcome.action_type, features, outcome.reward)
        self._last_outcome = None
        return {
            "updated": True,
            "action": outcome.action_type,
            "reward": outcome.reward,
            "situation": outcome.situation,
        }

    def train(self, dataset: list[tuple[dict[str, dict[str, float]], str]], **kw: Any) -> Any:
        """Batch imitation training — see :meth:`LinearPolicy.fit`."""
        return self.policy.fit(dataset, **kw)

    # --- introspection & persistence --------------------------------------

    def explain(self) -> str:
        """Why did the brain do what it just did?"""
        if self.last_decision is None:
            return "No decision made yet."
        return self.last_decision.explain()

    def describe_world(self) -> list[str]:
        return self.world.describe()

    def step(self, vision_output: Any, *, execute: bool = True) -> Decision:
        """observe -> decide -> execute in one call, for a continuous loop."""
        self.observe(vision_output)
        decision = self.decide()
        if execute:
            self.execute(decision)
        return decision

    def save(self, directory: str | Path) -> None:
        """Persist policy weights and long-term memory across power cycles."""
        import json

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.policy.save(path / "policy.json")
        (path / "memory.json").write_text(
            json.dumps(self.memory.longterm.snapshot(), indent=2), encoding="utf-8"
        )

    def load(self, directory: str | Path) -> None:
        import json

        path = Path(directory)
        policy_file = path / "policy.json"
        memory_file = path / "memory.json"
        if policy_file.exists():
            self.policy = LinearPolicy.load(policy_file)
            if isinstance(self.engine, UtilityDecisionEngine):
                self.engine.policy = self.policy
        if memory_file.exists():
            self.memory.longterm.restore(json.loads(memory_file.read_text(encoding="utf-8")))


__all__ = ["Action", "BrainConfig", "VisionBrain"]

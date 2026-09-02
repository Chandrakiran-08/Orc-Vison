"""The decision engine — state + memory + goal + actions → one action.

Structure::

    Current State ─┐
    Memory ────────┤
    Goal ──────────┼─▶ feature extraction ─▶ policy scoring ─▶ argmax ─▶ Action
    Available acts ┘                                   │
                                                       └─▶ explanation

Feature extraction is the only part that knows about *this* problem domain
(proximity, approach, hazards). The policy scoring it feeds is generic and
trainable, and the two are separable — swap either without touching the
other, or replace the whole engine via the :class:`DecisionEngine` protocol.

The memory features (``mem_success`` / ``mem_failure``) are what make the
brain more than a rule table: they carry what happened the *last* time this
situation was handled this way, so identical pixels can yield a different
action once experience says the first choice did not work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from orcvision.brain.actions import (
    ASCEND,
    AVOID,
    DEFAULT_ACTIONS,
    DESCEND,
    EMERGENCY_STOP,
    HOVER,
    MOVE,
    RETURN_HOME,
    SIGNAL,
    STOP,
    TRACK,
    WAIT,
    Action,
)
from orcvision.brain.constraints import Constraint, default_constraints
from orcvision.brain.memory import KIND_OUTCOME, Memory
from orcvision.brain.policy import Features, LinearPolicy, Policy
from orcvision.brain.state import SceneState, WorldState
from orcvision.brain.temporal import BrainEvent

# Initial weights. Hand-set so the brain is useful *before* any training,
# then adjusted by learning. Safety-shaped: closing hazards dominate.
DEFAULT_WEIGHTS: dict[str, float] = {
    # Evade a hazard when there is room to; stop when there is not.
    "AVOID|hazard_proximity": 1.00,
    "AVOID|hazard_approaching": 0.90,
    "AVOID|bias": 0.05,
    "STOP|hazard_proximity": 0.90,
    "STOP|hazard_approaching": 0.80,
    "STOP|bias": 0.00,
    # Making progress is only attractive when the way is clear.
    "MOVE|path_clear": 0.70,
    "MOVE|bias": 0.10,
    "TRACK|target_present": 0.60,
    "TRACK|bias": 0.00,
    "WAIT|bias": 0.15,
    "SIGNAL|hazard_present": 0.30,
    "SIGNAL|bias": 0.00,
    # Climbing over an obstacle is a real option for an aerial platform and
    # often better than a lateral dodge, but sits just under AVOID so a
    # ground-style evasion stays the default when both are available.
    "ASCEND|hazard_proximity": 0.85,
    "ASCEND|hazard_approaching": 0.70,
    # Recovery actions are driven by the platform's own state, not the
    # scene. These are what make the machine *choose* to come home rather
    # than merely being forbidden from going further.
    "RETURN_HOME|battery_low": 1.20,
    "RETURN_HOME|outside_fence": 1.00,
    "RETURN_HOME|platform_unhealthy": 0.60,
    "DESCEND|battery_critical": 1.50,
    "EMERGENCY_STOP|platform_unhealthy": 1.20,
    "HOVER|bias": 0.20,
}

# Experience weights, applied to every action: what memory says about how
# this action worked out last time in a situation like this one.
MEMORY_WEIGHTS = {"mem_failure": -1.50, "mem_success": 0.50}


def default_weights(actions: tuple[str, ...] = DEFAULT_ACTIONS) -> dict[str, float]:
    """Starting weight table, including per-action memory terms."""
    weights = dict(DEFAULT_WEIGHTS)
    for action in actions:
        for feature, value in MEMORY_WEIGHTS.items():
            weights.setdefault(f"{action}|{feature}", value)
        weights.setdefault(f"{action}|bias", 0.0)
    return weights


@dataclass
class DecisionContext:
    """Everything the engine is allowed to look at."""

    scene: SceneState
    world: WorldState
    events: list[BrainEvent] = field(default_factory=list)
    memory: Memory = field(default_factory=Memory)
    goal: str = "idle"
    now: float = 0.0
    hazard_labels: frozenset[str] = frozenset({"person", "obstacle", "vehicle", "car"})
    target_labels: frozenset[str] = frozenset()
    # Battery thresholds the recovery features are scaled against; kept here
    # so they match whatever BatteryConstraint was configured with.
    battery_return_pct: float = 25.0
    battery_land_pct: float = 10.0


@dataclass(slots=True)
class Reason:
    """One weighted term behind a decision, in plain language."""

    text: str
    contribution: float

    def __str__(self) -> str:
        return f"{self.text} ({self.contribution:+.2f})"


@dataclass
class Decision:
    """The chosen action, its score, and why."""

    action: Action
    score: float
    reasons: list[Reason] = field(default_factory=list)
    alternatives: list[tuple[str, float]] = field(default_factory=list)
    features: dict[str, Features] = field(default_factory=dict)
    situation: str = ""
    # Actions the policy ranked higher but a safety constraint forbade.
    vetoed: list[tuple[str, str]] = field(default_factory=list)
    safety_fallback: bool = False
    # Actions pushed down the ranking by remembered failures. This is what
    # lets the brain answer "why did you change your mind?" — the evidence
    # lives on the *rejected* action, not the chosen one.
    demoted: list[tuple[str, float]] = field(default_factory=list)

    def explain(self) -> str:
        """Render an inspectable account of this decision."""
        lines = [f"Decision: {self.action}", "", "Reason:"]
        if self.safety_fallback:
            lines.append("  every candidate action was vetoed — falling back to safe action")
        if not self.reasons and not self.safety_fallback:
            lines.append("  (no contributing evidence — default action)")
        for reason in sorted(self.reasons, key=lambda r: -abs(r.contribution)):
            lines.append(f"  {reason}")
        if self.demoted:
            lines += ["", "Down-weighted by experience:"]
            lines += [
                f"  {action} previously failed in this situation ({penalty:+.2f})"
                for action, penalty in self.demoted
            ]
        if self.vetoed:
            lines += ["", "Vetoed by safety constraints:"]
            lines += [f"  {reason}" for _, reason in self.vetoed]
        if self.alternatives:
            alts = ", ".join(f"{name} {score:+.2f}" for name, score in self.alternatives)
            lines += ["", f"Considered: {alts}"]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.to_dict(),
            "score": round(self.score, 4),
            "situation": self.situation,
            "reasons": [
                {"text": r.text, "contribution": round(r.contribution, 4)} for r in self.reasons
            ],
            "alternatives": [[n, round(s, 4)] for n, s in self.alternatives],
            "vetoed": [{"action": a, "reason": r} for a, r in self.vetoed],
            "safety_fallback": self.safety_fallback,
            "demoted": [{"action": a, "penalty": round(p, 4)} for a, p in self.demoted],
        }


@runtime_checkable
class DecisionEngine(Protocol):
    """Replaceable decision mechanism."""

    def decide(self, ctx: DecisionContext) -> Decision:  # pragma: no cover - protocol
        ...


def _threshold_signal(value: float | None, threshold: float) -> float:
    """0 above the threshold; 0.5 at it, rising to 1.0 at zero.

    Used for battery limits, where "just below the line" must already be a
    decisive signal rather than a rounding error.
    """
    if value is None or threshold <= 0 or value >= threshold:
        return 0.0
    depth = (threshold - value) / threshold  # 0 at the line, 1 at empty
    return min(1.0, 0.5 + 0.5 * depth)


def situation_key(ctx: DecisionContext) -> str:
    """A coarse label for "situations like this one".

    Memory is indexed by this, so experience generalizes across frames
    rather than being pinned to exact pixel values. Deliberately low
    cardinality: hazard class, where it is, and roughly how close.
    """
    hazard = ctx.world.nearest(set(ctx.hazard_labels)) or ctx.scene.nearest(set(ctx.hazard_labels))
    if hazard is None:
        return f"{ctx.goal}|clear"
    proximity = hazard.proximity()
    band = "near" if proximity > 0.6 else "mid" if proximity > 0.3 else "far"
    return f"{ctx.goal}|{hazard.label}|{hazard.zone}|{band}"


def outcome_key(situation: str, action_type: str) -> str:
    return f"{KIND_OUTCOME}:{situation}:{action_type}"


class UtilityDecisionEngine:
    """Feature-based utility scoring over a swappable :class:`Policy`.

    Covers the V1 mechanisms in one place — state-based conditions, utility
    scoring, priority (via weights), memory-conditioned choice, and a
    learned policy — because they are all the same computation once the
    features are named.
    """

    def __init__(
        self,
        policy: Policy | None = None,
        actions: tuple[str, ...] = DEFAULT_ACTIONS,
        max_range_m: float = 5.0,
        constraints: list[Constraint] | None = None,
        safe_action: str = STOP,
    ) -> None:
        self.actions = actions
        self.policy = policy or LinearPolicy(default_weights(actions))
        self.max_range_m = max_range_m
        # Deterministic floor under the learned policy. Pass [] to disable.
        self.constraints = default_constraints() if constraints is None else constraints
        self.safe_action = safe_action

    # --- features ----------------------------------------------------------

    def scene_features(self, ctx: DecisionContext) -> dict[str, float]:
        """Domain signals shared by all candidate actions."""
        hazards = set(ctx.hazard_labels)
        nearest = ctx.world.nearest(hazards) or ctx.scene.nearest(hazards)
        proximity = nearest.proximity(self.max_range_m) if nearest else 0.0
        approaching = 1.0 if nearest and nearest.motion == "approaching" else 0.0
        # An explicit approach event is stronger evidence than the latched
        # motion state, so let it override.
        if any(e.kind == "object_approaching" for e in ctx.events):
            approaching = 1.0
        targets = [o for o in ctx.world.visible() if o.label in ctx.target_labels]

        # The platform's own state matters as much as the scene: most real
        # incidents are energy, containment or health, not a missed object.
        platform = ctx.world.platform
        # Crossing a battery threshold is a step change, not a gentle ramp.
        # A purely linear deficit is ~0 just below the threshold, which lets a
        # cheap action like HOVER outrank RETURN_HOME at 22% of 25% — the
        # aircraft would then hold station until the battery died. So the
        # signal steps to 0.5 the moment the threshold is crossed and ramps to
        # 1.0 at empty.
        battery_low = _threshold_signal(platform.battery_pct, ctx.battery_return_pct)
        battery_critical = _threshold_signal(platform.battery_pct, ctx.battery_land_pct)

        return {
            "hazard_proximity": proximity,
            "hazard_approaching": approaching,
            "hazard_present": 1.0 if nearest else 0.0,
            "path_clear": 1.0 - proximity,
            "target_present": 1.0 if targets else 0.0,
            "battery_low": battery_low,
            "battery_critical": battery_critical,
            "outside_fence": 1.0 if platform.outside_geofence() else 0.0,
            "platform_unhealthy": 0.0 if platform.healthy() else 1.0,
        }

    def memory_features(
        self, ctx: DecisionContext, situation: str, action_type: str
    ) -> dict[str, float]:
        """What experience says about this action in this situation."""
        trace = ctx.memory.longterm.recall(outcome_key(situation, action_type), ctx.now)
        if trace is None:
            return {"mem_success": 0.0, "mem_failure": 0.0}
        content = trace.content if isinstance(trace.content, dict) else {}
        successes = float(content.get("successes", 0))
        failures = float(content.get("failures", 0))
        total = successes + failures
        if total <= 0:
            return {"mem_success": 0.0, "mem_failure": 0.0}
        return {"mem_success": successes / total, "mem_failure": failures / total}

    def candidate_features(self, ctx: DecisionContext, situation: str) -> dict[str, Features]:
        """Per-action feature vectors. Only relevant signals are included."""
        s = self.scene_features(ctx)
        relevant: dict[str, list[str]] = {
            STOP: ["hazard_proximity", "hazard_approaching"],
            AVOID: ["hazard_proximity", "hazard_approaching"],
            MOVE: ["path_clear"],
            TRACK: ["target_present"],
            WAIT: [],
            SIGNAL: ["hazard_present"],
            ASCEND: ["hazard_proximity", "hazard_approaching"],
            DESCEND: ["battery_critical"],
            HOVER: [],
            RETURN_HOME: ["battery_low", "outside_fence", "platform_unhealthy"],
            EMERGENCY_STOP: ["platform_unhealthy"],
        }
        candidates: dict[str, Features] = {}
        for action in self.actions:
            feats: Features = {name: s[name] for name in relevant.get(action, []) if name in s}
            feats["bias"] = 1.0
            feats.update(self.memory_features(ctx, situation, action))
            candidates[action] = feats
        return candidates

    # --- decision ----------------------------------------------------------

    def decide(self, ctx: DecisionContext) -> Decision:
        situation = situation_key(ctx)
        candidates = self.candidate_features(ctx, situation)
        ranked = self.policy.rank(candidates) if hasattr(self.policy, "rank") else None
        if ranked is None:  # a custom Policy without rank()
            scored = [(a, *self.policy.score(a, f)) for a, f in candidates.items()]
            ranked = sorted(scored, key=lambda r: r[1], reverse=True)

        # Apply the deterministic safety floor: walk the ranking until an
        # action survives every constraint. A learned policy can reorder
        # preferences, but it can never unlock a forbidden action.
        vetoed: list[tuple[str, str]] = []
        chosen: tuple[str, float, dict[str, float]] | None = None
        for action_type, score, contributions in ranked:
            reason = self._first_veto(action_type, ctx)
            if reason is None:
                chosen = (action_type, score, contributions)
                break
            vetoed.append((action_type, reason))

        safety_fallback = chosen is None
        if chosen is None:
            # Everything was forbidden — take the safe action, not the
            # least-bad forbidden one.
            fallback_features = candidates.get(self.safe_action, {})
            score, contributions = self.policy.score(self.safe_action, fallback_features)
            chosen = (self.safe_action, score, contributions)

        best_action, best_score, contributions = chosen
        reasons = [
            Reason(self._phrase(name, best_action, ctx), value)
            for name, value in contributions.items()
            if abs(value) > 1e-9
        ]
        # Record which *other* actions memory pushed down, so a change of
        # mind is traceable even though the evidence sits on the loser.
        demoted: list[tuple[str, float]] = []
        for action_type, feats in candidates.items():
            if action_type == best_action or not feats.get("mem_failure"):
                continue
            penalty = self.policy.score(action_type, {"mem_failure": feats["mem_failure"]})[0]
            if penalty < 0:
                demoted.append((action_type, penalty))

        return Decision(
            action=self._build_action(best_action, ctx),
            score=best_score,
            reasons=reasons,
            alternatives=[(name, score) for name, score, _ in ranked if name != best_action],
            features=candidates,
            situation=situation,
            vetoed=vetoed,
            safety_fallback=safety_fallback,
            demoted=demoted,
        )

    def _first_veto(self, action_type: str, ctx: DecisionContext) -> str | None:
        for constraint in self.constraints:
            reason = constraint.veto(action_type, ctx)
            if reason is not None:
                return reason
        return None

    def _phrase(self, feature: str, action: str, ctx: DecisionContext) -> str:
        """Turn a feature name into something a researcher can read."""
        hazard = ctx.world.nearest(set(ctx.hazard_labels))
        what = hazard.label if hazard else "hazard"
        where = f" in {hazard.zone}" if hazard else ""
        return {
            "hazard_proximity": f"{what} at close range{where}",
            "hazard_approaching": f"{what} moving toward system",
            "hazard_present": f"{what} detected{where}",
            "path_clear": "path appears clear",
            "target_present": "tracking target visible",
            "bias": f"baseline preference for {action}",
            "mem_success": f"{action} previously succeeded here",
            "mem_failure": f"{action} previously failed here",
            "battery_low": "battery below return threshold",
            "battery_critical": "battery critically low",
            "outside_fence": "outside the permitted envelope",
            "platform_unhealthy": "platform unhealthy (link/interlock/emergency)",
        }.get(feature, feature)

    def _build_action(self, action_type: str, ctx: DecisionContext) -> Action:
        """Attach parameters — e.g. which way to turn when avoiding."""
        params: dict[str, Any] = {}
        hazard = ctx.world.nearest(set(ctx.hazard_labels))
        if action_type == AVOID and hazard is not None:
            # Steer away from the side the hazard is on.
            params["direction"] = {"left": "right", "right": "left"}.get(hazard.zone, "right")
            params["reason_object"] = hazard.label
        elif action_type == MOVE:
            params["direction"] = "forward"
            params["speed"] = 0.5
        elif action_type == TRACK:
            target = next((o for o in ctx.world.visible() if o.label in ctx.target_labels), None)
            if target is not None:
                params["target"] = target.object_id
                params["zone"] = target.zone
        elif action_type == SIGNAL:
            params["level"] = "warning"
        elif action_type in (ASCEND, DESCEND):
            params["axis"] = "vertical"
            if ctx.world.platform.altitude_m is not None:
                params["altitude_m"] = ctx.world.platform.altitude_m
        elif action_type == RETURN_HOME:
            if ctx.world.platform.battery_pct is not None:
                params["battery_pct"] = ctx.world.platform.battery_pct
        elif action_type == EMERGENCY_STOP:
            params["reason"] = (
                "emergency"
                if ctx.world.platform.emergency
                else ("interlock" if not ctx.world.platform.interlock_ok else "link")
            )
        return Action(type=action_type, parameters=params)

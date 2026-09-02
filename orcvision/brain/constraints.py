"""Safety constraints — the floor a learned policy is not allowed to cross.

A utility policy that learns from outcomes has a failure mode that matters
enormously for physical autonomy: if every action accumulates failures, the
scores all sink, and some unrelated action (``MOVE`` forward, say) can end
up on top *while a hazard is closing*. Learning has no built-in notion that
some mistakes are unrecoverable.

So the brain is a **hybrid**: a trainable policy proposes, and deterministic
constraints dispose. Constraints are evaluated after scoring and can veto an
action outright, no matter how attractive the policy finds it. They are
plain, auditable Python — no weights, nothing learned, nothing that drifts.

If every candidate is vetoed the brain falls back to its configured safe
action (``STOP`` by default) rather than picking the least-bad forbidden one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from orcvision.brain.actions import FOLLOW, GRASP, MOVE, TRACK, TURN

if TYPE_CHECKING:  # pragma: no cover - typing only
    from orcvision.brain.decision import DecisionContext


@runtime_checkable
class Constraint(Protocol):
    """A hard rule that can forbid an action."""

    def veto(self, action_type: str, ctx: DecisionContext) -> str | None:
        """Return a reason string to forbid the action, or ``None`` to allow."""
        ...


class ProximityConstraint:
    """Forbid advancing toward a hazard that is too close.

    This is the constraint that keeps a degraded or badly-trained policy
    from driving into something. It reads the same normalized proximity the
    policy does, but its verdict is absolute.
    """

    def __init__(
        self,
        threshold: float = 0.6,
        blocked_actions: frozenset[str] = frozenset({MOVE, FOLLOW, GRASP, TURN}),
        max_range_m: float = 5.0,
    ) -> None:
        self.threshold = threshold
        self.blocked_actions = blocked_actions
        self.max_range_m = max_range_m

    def veto(self, action_type: str, ctx: DecisionContext) -> str | None:
        if action_type not in self.blocked_actions:
            return None
        hazard = ctx.world.nearest(set(ctx.hazard_labels)) or ctx.scene.nearest(
            set(ctx.hazard_labels)
        )
        if hazard is None:
            return None
        proximity = hazard.proximity(self.max_range_m)
        if proximity >= self.threshold:
            distance = f"{hazard.depth_m:.2f} m" if hazard.depth_m is not None else "close range"
            return (
                f"{action_type} forbidden: {hazard.label} at {distance} "
                f"in {hazard.zone} (proximity {proximity:.2f} >= {self.threshold:.2f})"
            )
        return None


class ConfidenceConstraint:
    """Forbid acting on a target the perception layer is unsure about.

    Prevents a low-confidence false positive from triggering a physical
    response such as ``GRASP`` or ``FOLLOW``.
    """

    def __init__(
        self,
        min_confidence: float = 0.4,
        guarded_actions: frozenset[str] = frozenset({GRASP, FOLLOW, TRACK}),
    ) -> None:
        self.min_confidence = min_confidence
        self.guarded_actions = guarded_actions

    def veto(self, action_type: str, ctx: DecisionContext) -> str | None:
        if action_type not in self.guarded_actions:
            return None
        targets = [o for o in ctx.world.visible() if o.label in ctx.target_labels]
        if not targets:
            return None
        best = max(targets, key=lambda o: o.confidence)
        if best.confidence < self.min_confidence:
            return (
                f"{action_type} forbidden: {best.label} confidence "
                f"{best.confidence:.2f} < {self.min_confidence:.2f}"
            )
        return None


def default_constraints() -> list[Constraint]:
    """The safety floor applied unless the caller opts out."""
    return [ProximityConstraint(), ConfidenceConstraint()]

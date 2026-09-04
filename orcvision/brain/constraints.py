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

from orcvision.brain.actions import (
    ASCEND,
    DESCEND,
    EMERGENCY_STOP,
    FOLLOW,
    GRASP,
    HOVER,
    MOVE,
    RETURN_HOME,
    SIGNAL,
    STOP,
    TRACK,
    TURN,
    WAIT,
)

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


class StaleDataConstraint:
    """Forbid acting on perception that has gone quiet.

    This is the constraint that matters most in a real deployment and the
    one toy systems always omit. If the camera, the network or the detector
    process dies, the world model simply stops updating — and a brain with
    no notion of staleness keeps confidently acting on a frozen snapshot,
    driving into a world that has moved on.

    Anything other than the safe action is forbidden once perception is
    older than ``max_age_s``.
    """

    def __init__(
        self,
        max_age_s: float = 1.0,
        # Deliberately excludes WAIT: when the world model may be wrong the
        # machine should reach a declared safe state, not merely idle in
        # whatever pose it happens to be in.
        allowed_when_stale: frozenset[str] = frozenset({STOP, EMERGENCY_STOP, HOVER, RETURN_HOME}),
    ) -> None:
        self.max_age_s = max_age_s
        self.allowed_when_stale = allowed_when_stale

    def veto(self, action_type: str, ctx: DecisionContext) -> str | None:
        if action_type in self.allowed_when_stale:
            return None
        age = ctx.now - ctx.world.updated_at
        if ctx.world.tick > 0 and age > self.max_age_s:
            return (
                f"{action_type} forbidden: perception is {age:.1f} s stale "
                f"(limit {self.max_age_s:.1f} s) — the world model may no longer be true"
            )
        return None


class HealthConstraint:
    """Forbid everything but recovery when the platform is unhealthy.

    A tripped e-stop, an open guard door or a lost command link means the
    machine has no business continuing its task, no matter how clear the
    camera says the path is.
    """

    def __init__(
        self,
        allowed_when_unhealthy: frozenset[str] = frozenset(
            {STOP, EMERGENCY_STOP, HOVER, RETURN_HOME, WAIT, SIGNAL}
        ),
    ) -> None:
        self.allowed_when_unhealthy = allowed_when_unhealthy

    def veto(self, action_type: str, ctx: DecisionContext) -> str | None:
        if action_type in self.allowed_when_unhealthy:
            return None
        platform = ctx.world.platform
        if platform.emergency:
            return f"{action_type} forbidden: emergency declared"
        if not platform.interlock_ok:
            return f"{action_type} forbidden: safety interlock tripped"
        if not platform.link_ok:
            return f"{action_type} forbidden: command link down"
        return None


class BatteryConstraint:
    """Forbid continuing the mission on a battery that cannot finish it.

    Two thresholds, matching how flight controllers actually behave:
    below ``return_pct`` only recovery actions remain; below
    ``land_pct`` even returning is off the table and it must come down.
    """

    def __init__(
        self,
        return_pct: float = 25.0,
        land_pct: float = 10.0,
        mission_actions: frozenset[str] = frozenset({MOVE, FOLLOW, TRACK, ASCEND, GRASP}),
    ) -> None:
        self.return_pct = return_pct
        self.land_pct = land_pct
        self.mission_actions = mission_actions

    def veto(self, action_type: str, ctx: DecisionContext) -> str | None:
        platform = ctx.world.platform
        if platform.battery_pct is None:
            return None  # unknown telemetry: do not invent a limit
        if platform.battery_below(self.land_pct) and action_type not in {
            DESCEND,
            STOP,
            EMERGENCY_STOP,
        }:
            return (
                f"{action_type} forbidden: battery {platform.battery_pct:.0f}% "
                f"below land threshold {self.land_pct:.0f}% — must descend now"
            )
        if platform.battery_below(self.return_pct) and action_type in self.mission_actions:
            return (
                f"{action_type} forbidden: battery {platform.battery_pct:.0f}% "
                f"below return threshold {self.return_pct:.0f}%"
            )
        return None


class GeofenceConstraint:
    """Forbid travelling further out once outside the permitted envelope.

    Covers both the horizontal geofence (UAV, AGV work area) and the
    altitude band. Coming back is always allowed — otherwise a breach would
    be unrecoverable.
    """

    def __init__(
        self,
        outbound_actions: frozenset[str] = frozenset({MOVE, FOLLOW, ASCEND, TRACK}),
    ) -> None:
        self.outbound_actions = outbound_actions

    def veto(self, action_type: str, ctx: DecisionContext) -> str | None:
        if action_type not in self.outbound_actions:
            return None
        platform = ctx.world.platform
        if platform.outside_geofence():
            return (
                f"{action_type} forbidden: {platform.distance_from_home_m:.0f} m from home "
                f"exceeds the {platform.geofence_radius_m:.0f} m geofence"
            )
        if action_type == ASCEND and platform.altitude_m is not None:
            if (
                platform.max_altitude_m is not None
                and platform.altitude_m >= platform.max_altitude_m
            ):
                return (
                    f"ASCEND forbidden: at {platform.altitude_m:.0f} m, ceiling is "
                    f"{platform.max_altitude_m:.0f} m"
                )
        if action_type == DESCEND and platform.altitude_m is not None:
            if (
                platform.min_altitude_m is not None
                and platform.altitude_m <= platform.min_altitude_m
            ):
                return (
                    f"DESCEND forbidden: at {platform.altitude_m:.0f} m, floor is "
                    f"{platform.min_altitude_m:.0f} m"
                )
        return None


class KeepOutZoneConstraint:
    """Forbid motion while a hazard occupies a defined region of the frame.

    The software analogue of a light curtain: an industrial cell defines a
    danger zone, and any person inside it stops the machine regardless of
    measured distance. Zones are normalized (x1, y1, x2, y2) in 0..1 so
    they are independent of camera resolution.
    """

    def __init__(
        self,
        zones: list[tuple[float, float, float, float]],
        blocked_actions: frozenset[str] = frozenset({MOVE, FOLLOW, GRASP, TURN, ASCEND}),
        labels: frozenset[str] | None = None,
    ) -> None:
        self.zones = zones
        self.blocked_actions = blocked_actions
        # None means "any hazard label"; otherwise restrict to these classes.
        self.labels = labels

    def veto(self, action_type: str, ctx: DecisionContext) -> str | None:
        if action_type not in self.blocked_actions:
            return None
        watch = self.labels if self.labels is not None else ctx.hazard_labels
        for obj in ctx.world.visible():
            if obj.label not in watch:
                continue
            x, y = obj.position
            for index, (x1, y1, x2, y2) in enumerate(self.zones):
                if x1 <= x <= x2 and y1 <= y <= y2:
                    return (
                        f"{action_type} forbidden: {obj.label} inside keep-out zone "
                        f"{index} ({x1:.2f},{y1:.2f})-({x2:.2f},{y2:.2f})"
                    )
        return None


def uav_constraints(
    battery_return_pct: float = 25.0,
    battery_land_pct: float = 10.0,
    max_perception_age_s: float = 1.0,
) -> list[Constraint]:
    """A sane safety floor for an aerial platform."""
    return [
        HealthConstraint(),
        StaleDataConstraint(max_age_s=max_perception_age_s),
        BatteryConstraint(return_pct=battery_return_pct, land_pct=battery_land_pct),
        GeofenceConstraint(),
        ProximityConstraint(blocked_actions=frozenset({MOVE, FOLLOW, TRACK})),
    ]


def industrial_constraints(
    keep_out_zones: list[tuple[float, float, float, float]] | None = None,
    max_perception_age_s: float = 0.5,
) -> list[Constraint]:
    """A sane safety floor for a fixed industrial cell.

    Perception staleness is tighter than the UAV default: a stopped
    conveyor is cheap, a missed person is not.
    """
    constraints: list[Constraint] = [
        HealthConstraint(),
        StaleDataConstraint(max_age_s=max_perception_age_s),
        ProximityConstraint(),
    ]
    if keep_out_zones:
        constraints.insert(0, KeepOutZoneConstraint(keep_out_zones))
    return constraints

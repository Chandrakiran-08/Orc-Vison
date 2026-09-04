"""Normalized internal state — the brain's own view of the world.

This layer is deliberately **decoupled from any detector**. Nothing here
imports YOLO, ONNX, numpy or pydantic; a :class:`SceneState` is a plain
dataclass that any perception source can produce via
:mod:`orcvision.brain.adapters`.

Two representations:

``SceneState``
    What is visible *right now*, normalized. Positions are fractions of the
    frame (0..1) so a decision policy trained at 640x480 still works at
    1920x1080 or on a 96x96 microcontroller camera.

``WorldState``
    What the brain *believes* persists across frames: tracked objects with
    motion, the active goal, and the last action taken. This is what makes
    decisions temporal rather than per-frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Horizontal zones, as fractions of frame width. Coarse on purpose: a
# decision layer needs "which way do I turn", not sub-pixel accuracy.
ZONE_LEFT = "left"
ZONE_CENTER = "center"
ZONE_RIGHT = "right"

# Motion classes assigned by the temporal reasoner.
MOTION_UNKNOWN = "unknown"
MOTION_STATIONARY = "stationary"
MOTION_MOVING = "moving"
MOTION_APPROACHING = "approaching"
MOTION_RECEDING = "receding"


def zone_of(cx: float) -> str:
    """Map a normalized x-centre (0..1) to a coarse horizontal zone."""
    if cx < 1 / 3:
        return ZONE_LEFT
    if cx > 2 / 3:
        return ZONE_RIGHT
    return ZONE_CENTER


def clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


@dataclass(slots=True)
class ObjectState:
    """One object as the *brain* understands it, not as a detector emitted it."""

    object_id: str  # stable identity across frames (track id, or synthesized)
    label: str
    confidence: float
    position: tuple[float, float]  # normalized (cx, cy) in 0..1
    size: float  # normalized bbox area in 0..1
    depth_m: float | None = None
    zone: str = ZONE_CENTER
    last_seen: float = 0.0
    first_seen: float = 0.0
    # Filled in by the temporal reasoner:
    velocity: tuple[float, float] = (0.0, 0.0)  # normalized units per second
    approach_rate: float = 0.0  # metres/sec toward the sensor; >0 = closing in
    motion: str = MOTION_UNKNOWN
    misses: int = 0  # consecutive frames this object was not observed

    def proximity(self, max_range_m: float = 5.0) -> float:
        """How close this object is, as 0 (far) .. 1 (touching).

        Uses metric depth when available. Without depth, apparent size is a
        usable monocular proxy — a bigger box means closer, which is enough
        for a coarse "back off" decision.
        """
        if self.depth_m is not None and max_range_m > 0:
            return clamp01(1.0 - (self.depth_m / max_range_m))
        # sqrt(area) ~ linear extent; x2 so a box covering ~25% of the frame
        # already reads as "very close".
        return clamp01((self.size**0.5) * 2.0)

    def age(self, now: float) -> float:
        return max(0.0, now - self.first_seen)


@dataclass(slots=True)
class Relationship:
    """A lightweight spatial relation between two objects."""

    subject: str  # object_id
    predicate: str  # near | left_of | right_of | closer_than
    target: str  # object_id

    def describe(self) -> str:
        return f"{self.subject} {self.predicate} {self.target}"


@dataclass(slots=True)
class SceneState:
    """Normalized snapshot of one frame — the brain's perception input."""

    timestamp: float
    objects: list[ObjectState] = field(default_factory=list)
    frame_id: int = 0
    source: str = "unknown"
    modality: str = "rgb"
    meta: dict[str, Any] = field(default_factory=dict)

    def by_label(self, label: str) -> list[ObjectState]:
        return [o for o in self.objects if o.label == label]

    def labels(self) -> set[str]:
        return {o.label for o in self.objects}

    def count(self, label: str | None = None) -> int:
        return len(self.objects) if label is None else len(self.by_label(label))

    def nearest(self, labels: set[str] | None = None) -> ObjectState | None:
        """The most proximate object, optionally restricted to some labels."""
        pool = [o for o in self.objects if labels is None or o.label in labels]
        return max(pool, key=lambda o: o.proximity(), default=None)


@dataclass
class PlatformState:
    """What the machine knows about *itself*.

    Reasoning about obstacles is only half of autonomy. A UAV that cannot
    act on its own battery level, or a machine that cannot tell its safety
    interlock has tripped, is not deployable — most real incidents come from
    the platform's own state, not from a missed detection.

    Every field is optional and defaults to "unknown" (``None``), because a
    given platform reports only some of them. Constraints that depend on a
    field simply do not fire when it is unknown, so partial telemetry
    degrades cleanly instead of raising.
    """

    # Energy — the single most common cause of UAV loss.
    battery_pct: float | None = None  # 0..100
    # Vertical position and limits (UAV, gantry, lift).
    altitude_m: float | None = None
    max_altitude_m: float | None = None
    min_altitude_m: float | None = None
    # Horizontal containment (UAV geofence, AGV work area).
    distance_from_home_m: float | None = None
    geofence_radius_m: float | None = None
    # Motion.
    speed_mps: float | None = None
    max_speed_mps: float | None = None
    # Health and interlocks.
    link_ok: bool = True  # command/telemetry link up
    interlock_ok: bool = True  # e-stop / light curtain / guard door closed
    emergency: bool = False  # operator or supervisor declared an emergency
    # Free-form extras a specific platform wants constraints to see.
    extra: dict[str, Any] = field(default_factory=dict)

    def battery_below(self, threshold_pct: float) -> bool:
        """True only when the battery is known AND below the threshold."""
        return self.battery_pct is not None and self.battery_pct < threshold_pct

    def outside_geofence(self) -> bool:
        if self.distance_from_home_m is None or self.geofence_radius_m is None:
            return False
        return self.distance_from_home_m > self.geofence_radius_m

    def altitude_out_of_band(self) -> bool:
        if self.altitude_m is None:
            return False
        if self.max_altitude_m is not None and self.altitude_m > self.max_altitude_m:
            return True
        return self.min_altitude_m is not None and self.altitude_m < self.min_altitude_m

    def healthy(self) -> bool:
        """Whether it is safe to consider anything other than a safe action."""
        return self.link_ok and self.interlock_ok and not self.emergency

    def describe(self) -> list[str]:
        lines: list[str] = []
        if self.battery_pct is not None:
            lines.append(f"Battery: {self.battery_pct:.0f}%")
        if self.altitude_m is not None:
            lines.append(f"Altitude: {self.altitude_m:.1f} m")
        if self.distance_from_home_m is not None:
            lines.append(f"Distance from home: {self.distance_from_home_m:.1f} m")
        if not self.link_ok:
            lines.append("Link: DOWN")
        if not self.interlock_ok:
            lines.append("Interlock: TRIPPED")
        if self.emergency:
            lines.append("EMERGENCY declared")
        return lines


@dataclass
class WorldState:
    """Persistent internal model: what the brain believes is out there.

    Unlike :class:`SceneState` (one frame), this survives across frames and
    is what gives the brain continuity — object identity, motion, the goal
    it is pursuing, and what it last did.
    """

    objects: dict[str, ObjectState] = field(default_factory=dict)
    goal: str = "idle"
    last_action: str | None = None
    updated_at: float = 0.0
    tick: int = 0
    environment: dict[str, Any] = field(default_factory=dict)
    # The machine's model of itself — battery, altitude, interlocks. See
    # PlatformState: without this the brain can only reason about the world,
    # never about its own capacity to act in it.
    platform: PlatformState = field(default_factory=lambda: PlatformState())

    def visible(self) -> list[ObjectState]:
        """Objects observed in the most recent frame (not merely remembered)."""
        return [o for o in self.objects.values() if o.misses == 0]

    def nearest(self, labels: set[str] | None = None) -> ObjectState | None:
        pool = [o for o in self.visible() if labels is None or o.label in labels]
        return max(pool, key=lambda o: o.proximity(), default=None)

    def relationships(self, max_pairs: int = 12) -> list[Relationship]:
        """Derive coarse spatial relations between visible objects.

        Capped by ``max_pairs`` — this is O(n^2) and runs per frame on
        hardware that may not have cycles to spare.
        """
        objs = self.visible()
        out: list[Relationship] = []
        for i, a in enumerate(objs):
            for b in objs[i + 1 :]:
                if len(out) >= max_pairs:
                    return out
                ax, _ = a.position
                bx, _ = b.position
                if abs(ax - bx) < 0.15:
                    out.append(Relationship(a.object_id, "near", b.object_id))
                elif ax < bx:
                    out.append(Relationship(a.object_id, "left_of", b.object_id))
                else:
                    out.append(Relationship(a.object_id, "right_of", b.object_id))
                if a.depth_m is not None and b.depth_m is not None and a.depth_m < b.depth_m:
                    out.append(Relationship(a.object_id, "closer_than", b.object_id))
        return out

    def describe(self) -> list[str]:
        """Human-readable summary of the internal world model."""
        lines = [f"Goal: {self.goal}"]
        for obj in sorted(self.visible(), key=lambda o: -o.proximity()):
            depth = f"{obj.depth_m:.2f} m" if obj.depth_m is not None else "unknown depth"
            lines.append(f"Object: {obj.label} | Zone: {obj.zone} | State: {obj.motion} | {depth}")
        if self.last_action:
            lines.append(f"Last action: {self.last_action}")
        lines.extend(self.platform.describe())
        return lines

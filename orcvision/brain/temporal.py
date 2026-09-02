"""Temporal reasoning — turn a sequence of frames into *changes*.

A YOLO wrapper answers "what is in this frame". A brain answers "what just
happened". This module folds each new :class:`SceneState` into the
persistent :class:`WorldState`, maintains object identity, estimates motion,
and emits discrete :class:`BrainEvent`s:

    frame N   frame N+1   frame N+2
       └──────────┴───────────┘
                  ▼
         object approached / stopped / disappeared
                  ▼
              decision

Object association uses the upstream ``track_id`` when perception provided
one, and falls back to nearest-neighbour position matching when it did not —
so the brain still works behind a detector with no tracker at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from orcvision.brain.state import (
    MOTION_APPROACHING,
    MOTION_MOVING,
    MOTION_RECEDING,
    MOTION_STATIONARY,
    ObjectState,
    SceneState,
    WorldState,
    zone_of,
)

# Event kinds emitted by the temporal reasoner.
OBJECT_APPEARED = "object_appeared"
OBJECT_DISAPPEARED = "object_disappeared"
OBJECT_MOVED = "object_moved"
OBJECT_STOPPED = "object_stopped"
OBJECT_APPROACHING = "object_approaching"
OBJECT_RECEDING = "object_receding"
ZONE_CHANGED = "zone_changed"
SCENE_CHANGED = "scene_changed"


@dataclass(slots=True)
class BrainEvent:
    """A discrete change the brain noticed between frames."""

    kind: str
    timestamp: float
    object_id: str | None = None
    label: str | None = None
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        subject = self.label or self.object_id or "scene"
        return f"{subject} {self.kind.replace('object_', '').replace('_', ' ')}{self.detail}"


@dataclass
class TemporalConfig:
    """Thresholds for change detection. Tune per platform/frame rate."""

    move_threshold: float = 0.02  # normalized displacement counted as motion
    approach_threshold: float = 0.05  # m/s of depth closing counted as approach
    size_growth_threshold: float = 0.12  # relative bbox growth ~ approaching
    disappear_after_misses: int = 5  # frames before an object is dropped
    match_radius: float = 0.15  # normalized distance for id-less association


class TemporalReasoner:
    """Folds scenes into a world model and reports what changed."""

    def __init__(self, config: TemporalConfig | None = None) -> None:
        self.config = config or TemporalConfig()

    def update(self, world: WorldState, scene: SceneState) -> list[BrainEvent]:
        """Merge ``scene`` into ``world``; return the changes detected."""
        cfg = self.config
        events: list[BrainEvent] = []
        now = scene.timestamp
        # Use the tick counter, not the truthiness of ``updated_at`` — a
        # first frame whose timestamp is exactly 0.0 is still a first frame.
        dt = max(1e-6, now - world.updated_at) if world.tick else 0.0

        matched: dict[str, ObjectState] = {}
        unmatched_prev = dict(world.objects)

        for obs in scene.objects:
            prev = self._associate(obs, unmatched_prev)
            if prev is None:
                obs.first_seen = now
                obs.misses = 0
                matched[obs.object_id] = obs
                events.append(
                    BrainEvent(
                        OBJECT_APPEARED,
                        now,
                        obs.object_id,
                        obs.label,
                        f" in {obs.zone}",
                        {"zone": obs.zone},
                    )
                )
                continue

            unmatched_prev.pop(prev.object_id, None)
            events.extend(self._update_matched(prev, obs, now, dt))
            matched[prev.object_id] = prev

        # Objects not seen this frame: age them out rather than forgetting
        # instantly — detectors drop boxes for a frame or two all the time.
        for obj_id, missing in unmatched_prev.items():
            missing.misses += 1
            if missing.misses >= cfg.disappear_after_misses:
                events.append(
                    BrainEvent(
                        OBJECT_DISAPPEARED, now, obj_id, missing.label, f" from {missing.zone}"
                    )
                )
            else:
                matched[obj_id] = missing

        if len(matched) != len(world.objects):
            events.append(
                BrainEvent(
                    SCENE_CHANGED,
                    now,
                    detail=f" ({len(world.objects)} -> {len(matched)} objects)",
                    data={"before": len(world.objects), "after": len(matched)},
                )
            )

        world.objects = matched
        world.updated_at = now
        world.tick += 1
        return events

    def _associate(
        self, obs: ObjectState, candidates: dict[str, ObjectState]
    ) -> ObjectState | None:
        """Find the previously-known object this observation continues."""
        # Fast path: upstream tracker gave us a stable id.
        if obs.object_id in candidates:
            return candidates[obs.object_id]
        # Fallback: nearest same-label object within the match radius. This is
        # what lets the brain run behind a detector with no tracker.
        best, best_dist = None, self.config.match_radius
        for cand in candidates.values():
            if cand.label != obs.label:
                continue
            dx = cand.position[0] - obs.position[0]
            dy = cand.position[1] - obs.position[1]
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < best_dist:
                best, best_dist = cand, dist
        return best

    def _update_matched(
        self, prev: ObjectState, obs: ObjectState, now: float, dt: float
    ) -> list[BrainEvent]:
        """Update a known object in place and emit its change events."""
        cfg = self.config
        events: list[BrainEvent] = []

        dx = obs.position[0] - prev.position[0]
        dy = obs.position[1] - prev.position[1]
        displacement = (dx * dx + dy * dy) ** 0.5
        was_moving = prev.motion in (MOTION_MOVING, MOTION_APPROACHING, MOTION_RECEDING)

        # Depth closing rate (positive = getting nearer), or apparent-size
        # growth as the monocular fallback.
        approach_rate = 0.0
        if prev.depth_m is not None and obs.depth_m is not None and dt > 0:
            approach_rate = (prev.depth_m - obs.depth_m) / dt
        size_growth = (obs.size - prev.size) / prev.size if prev.size > 0 else 0.0

        approaching = approach_rate > cfg.approach_threshold or (
            obs.depth_m is None and size_growth > cfg.size_growth_threshold
        )
        receding = approach_rate < -cfg.approach_threshold or (
            obs.depth_m is None and size_growth < -cfg.size_growth_threshold
        )

        if approaching:
            motion = MOTION_APPROACHING
        elif receding:
            motion = MOTION_RECEDING
        elif displacement > cfg.move_threshold:
            motion = MOTION_MOVING
        else:
            motion = MOTION_STATIONARY

        old_zone = prev.zone
        # Mutate the persistent object rather than replacing it, so
        # first_seen / identity survive.
        prev.confidence = obs.confidence
        prev.position = obs.position
        prev.size = obs.size
        prev.depth_m = obs.depth_m
        prev.zone = zone_of(obs.position[0])
        prev.last_seen = now
        prev.misses = 0
        prev.velocity = (dx / dt, dy / dt) if dt > 0 else (0.0, 0.0)
        prev.approach_rate = approach_rate
        prev.motion = motion

        if motion == MOTION_APPROACHING:
            events.append(
                BrainEvent(
                    OBJECT_APPROACHING,
                    now,
                    prev.object_id,
                    prev.label,
                    f" ({approach_rate:.2f} m/s)" if approach_rate else "",
                    {"approach_rate": approach_rate, "proximity": prev.proximity()},
                )
            )
        elif motion == MOTION_RECEDING:
            events.append(BrainEvent(OBJECT_RECEDING, now, prev.object_id, prev.label))
        elif motion == MOTION_MOVING:
            events.append(BrainEvent(OBJECT_MOVED, now, prev.object_id, prev.label))
        elif was_moving:
            events.append(BrainEvent(OBJECT_STOPPED, now, prev.object_id, prev.label))

        if prev.zone != old_zone:
            events.append(
                BrainEvent(
                    ZONE_CHANGED,
                    now,
                    prev.object_id,
                    prev.label,
                    f" {old_zone} -> {prev.zone}",
                    {"from": old_zone, "to": prev.zone},
                )
            )
        return events

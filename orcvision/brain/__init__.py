"""Orc-Vison brain — the intelligence layer above perception.

    Perception → State → Memory → Decision → Action → Feedback

This package deliberately contains **no computer-vision model**. It consumes
structured perception output (from YOLO, ONNX, OpenCV, a depth camera, a
custom CNN, or a microcontroller sending compact JSON) and turns it into
autonomous decisions.

Quick start::

    from orcvision.brain import VisionBrain

    brain = VisionBrain(goal="avoid_collision")
    brain.observe(perception_event)      # any vision source
    decision = brain.decide()
    print(decision.explain())
    brain.execute(decision)
    brain.feedback(success=False)        # closes the loop
    brain.learn()

Every layer is replaceable — see the submodules: :mod:`adapters` (perception
in), :mod:`state`, :mod:`temporal`, :mod:`memory`, :mod:`decision`,
:mod:`policy` (trainable), :mod:`actions`, :mod:`feedback`.

Pure standard library: no numpy, no torch, no network, no cloud.
"""

from orcvision.brain.actions import (
    ACTION_TYPES,
    DEFAULT_ACTIONS,
    INDUSTRIAL_ACTIONS,
    UAV_ACTIONS,
    Action,
    ActionExecutor,
)
from orcvision.brain.adapters import from_boxes, from_perception_event, from_records
from orcvision.brain.audit import AuditLog
from orcvision.brain.brain import BrainConfig, VisionBrain
from orcvision.brain.constraints import (
    BatteryConstraint,
    Constraint,
    GeofenceConstraint,
    HealthConstraint,
    KeepOutZoneConstraint,
    StaleDataConstraint,
    industrial_constraints,
    uav_constraints,
)
from orcvision.brain.decision import (
    Decision,
    DecisionContext,
    DecisionEngine,
    UtilityDecisionEngine,
)
from orcvision.brain.feedback import Outcome
from orcvision.brain.memory import LongTermMemory, Memory, WorkingMemory
from orcvision.brain.policy import LinearPolicy, Policy
from orcvision.brain.state import ObjectState, PlatformState, SceneState, WorldState
from orcvision.brain.temporal import BrainEvent, TemporalConfig, TemporalReasoner

__all__ = [
    "ACTION_TYPES",
    "DEFAULT_ACTIONS",
    "INDUSTRIAL_ACTIONS",
    "UAV_ACTIONS",
    "AuditLog",
    "BatteryConstraint",
    "Constraint",
    "GeofenceConstraint",
    "HealthConstraint",
    "KeepOutZoneConstraint",
    "PlatformState",
    "StaleDataConstraint",
    "industrial_constraints",
    "uav_constraints",
    "Action",
    "ActionExecutor",
    "BrainConfig",
    "BrainEvent",
    "Decision",
    "DecisionContext",
    "DecisionEngine",
    "LinearPolicy",
    "LongTermMemory",
    "Memory",
    "ObjectState",
    "Outcome",
    "Policy",
    "SceneState",
    "TemporalConfig",
    "TemporalReasoner",
    "UtilityDecisionEngine",
    "VisionBrain",
    "WorkingMemory",
    "WorldState",
    "from_boxes",
    "from_perception_event",
    "from_records",
]

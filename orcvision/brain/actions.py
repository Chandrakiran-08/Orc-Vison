"""Generic action abstraction — hardware lives outside the brain.

The brain emits an :class:`Action`: a type plus parameters, serializable to
JSON. It never imports a motor driver, a GPIO library or a ROS message. A
thin executor at the edge (an MQTT subscriber, an Arduino sketch, a robot
SDK shim) is what turns ``MOVE {"direction": "forward"}`` into voltage.

That boundary is what makes one trained policy portable across a drone, an
arm and a security camera.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Canonical action vocabulary. CUSTOM carries anything project-specific in
# its parameters so the set never needs to grow for a one-off actuator.
MOVE = "MOVE"
STOP = "STOP"
TURN = "TURN"
FOLLOW = "FOLLOW"
AVOID = "AVOID"
TRACK = "TRACK"
SELECT = "SELECT"
GRASP = "GRASP"
WAIT = "WAIT"
SIGNAL = "SIGNAL"
CUSTOM = "CUSTOM"

ACTION_TYPES = (
    MOVE,
    STOP,
    TURN,
    FOLLOW,
    AVOID,
    TRACK,
    SELECT,
    GRASP,
    WAIT,
    SIGNAL,
    CUSTOM,
)

# A safe default repertoire for a mobile vision platform.
DEFAULT_ACTIONS = (STOP, AVOID, MOVE, TRACK, WAIT, SIGNAL)


@dataclass(slots=True)
class Action:
    """A machine-readable command with no hardware coupling."""

    type: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "parameters": dict(self.parameters)}

    def __str__(self) -> str:
        if not self.parameters:
            return self.type
        params = ", ".join(f"{k}={v}" for k, v in sorted(self.parameters.items()))
        return f"{self.type}({params})"


class ActionExecutor:
    """Base executor. Subclass to drive real hardware.

    The default implementation only records what it was asked to do, which
    is exactly what you want in simulation, tests and dry runs.
    """

    def __init__(self) -> None:
        self.history: list[Action] = []

    def execute(self, action: Action) -> dict[str, Any]:
        """Perform the action; return an arbitrary result dict."""
        self.history.append(action)
        return {"executed": action.type, "ok": True}

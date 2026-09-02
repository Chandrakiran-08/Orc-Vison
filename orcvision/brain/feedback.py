"""Feedback — how an action's outcome becomes tomorrow's decision.

Without this module the brain is a stateless scorer. With it, the loop
closes::

    decide ─▶ act ─▶ observe outcome ─▶ reward ─▶ memory + policy ─▶ decide

Feedback is written to two places, because they serve different horizons:

* **Long-term memory**, as success/failure counts keyed by
  ``(situation, action)``. This changes behaviour *immediately* on the next
  decision, with no training step — the decision engine reads those counts
  as features.
* **The policy**, as a reward-weighted weight update. This generalizes the
  lesson across situations, slowly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from orcvision.brain.decision import Decision, outcome_key
from orcvision.brain.memory import KIND_OUTCOME, Memory


@dataclass(slots=True)
class Outcome:
    """The observed result of a previously taken action."""

    action_type: str
    situation: str
    success: bool
    reward: float
    timestamp: float
    note: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        verdict = "succeeded" if self.success else "failed"
        suffix = f" — {self.note}" if self.note else ""
        return f"{self.action_type} {verdict} in [{self.situation}]{suffix}"


def record_outcome(
    memory: Memory,
    decision: Decision,
    success: bool,
    now: float,
    reward: float | None = None,
    note: str = "",
) -> Outcome:
    """Write an outcome into working and long-term memory.

    The long-term trace is keyed by ``(situation, action)`` and accumulates
    counts, so repeated attempts reinforce one trace instead of flooding
    memory with duplicates.
    """
    action_type = decision.action.type
    situation = decision.situation
    if reward is None:
        reward = 1.0 if success else -1.0

    key = outcome_key(situation, action_type)
    trace = memory.longterm.recall(key, now)
    content = (
        dict(trace.content)
        if trace and isinstance(trace.content, dict)
        else {
            "successes": 0,
            "failures": 0,
            "action": action_type,
            "situation": situation,
        }
    )
    content["successes" if success else "failures"] += 1
    content["last_reward"] = reward

    # Failures matter more than successes for safety-critical autonomy: a
    # bad outcome should be remembered longer and weigh on the next choice.
    importance = 0.85 if not success else 0.6
    memory.longterm.remember(key, content, now, kind=KIND_OUTCOME, importance=importance)

    outcome = Outcome(
        action_type=action_type,
        situation=situation,
        success=success,
        reward=reward,
        timestamp=now,
        note=note,
    )
    memory.working.add(KIND_OUTCOME, outcome.describe(), now, label=action_type)
    return outcome

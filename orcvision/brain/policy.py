"""The trainable decision policy — deliberately the simplest thing that learns.

Scoring is linear::

    score(action) = Σ  weight[action|feature] × feature_value

That choice is not a placeholder, it is the point:

* **Lightweight.** A dict of floats. No torch, no numpy, no matrix ops — a
  trained policy is a few dozen numbers you can print. (It needs CPython to
  run today; see "Where it runs" in the README for the MCU story.)
* **Explainable.** Every term is a named contribution, so "why did you
  stop?" is answered by sorting the terms, not by interpreting a black box.
* **Trainable.** A linear model has a well-defined gradient, so the same
  weights accept supervised/imitation updates *and* reward-based ones.
* **Replaceable.** Anything implementing :class:`Policy` can drop in — a
  decision tree, a tiny MLP, an RL policy — without touching memory, state
  or perception.

The learning rules here are textbook: a perceptron-style update for
imitation, and a REINFORCE-flavoured update for reward. Both nudge the same
weight table, so a policy can be bootstrapped from demonstrations and then
refined online from outcomes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

Features = dict[str, float]


@runtime_checkable
class Policy(Protocol):
    """Swappable scoring policy."""

    def score(self, action_type: str, features: Features) -> tuple[float, dict[str, float]]:
        """Return ``(score, per-feature contributions)``."""
        ...


def _key(action_type: str, feature: str) -> str:
    return f"{action_type}|{feature}"


class LinearPolicy:
    """Linear utility policy over named features, trainable online."""

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        learning_rate: float = 0.1,
    ) -> None:
        self.weights: dict[str, float] = dict(weights or {})
        self.learning_rate = learning_rate

    # --- inference ---------------------------------------------------------

    def score(self, action_type: str, features: Features) -> tuple[float, dict[str, float]]:
        contributions = {
            name: self.weights.get(_key(action_type, name), 0.0) * value
            for name, value in features.items()
        }
        return sum(contributions.values()), contributions

    def rank(self, candidates: dict[str, Features]) -> list[tuple[str, float, dict[str, float]]]:
        """Score every candidate action, best first."""
        scored = [(action, *self.score(action, feats)) for action, feats in candidates.items()]
        return sorted(scored, key=lambda row: row[1], reverse=True)

    # --- learning ----------------------------------------------------------

    def reinforce(
        self,
        action_type: str,
        features: Features,
        reward: float,
        learning_rate: float | None = None,
    ) -> None:
        """Reward-driven update: push the taken action's weights by ``reward``.

        Positive reward makes this action more likely in states that look
        like this one; negative reward makes it less likely. This is the
        hook an RL loop drives.
        """
        lr = self.learning_rate if learning_rate is None else learning_rate
        for name, value in features.items():
            k = _key(action_type, name)
            self.weights[k] = self.weights.get(k, 0.0) + lr * reward * value

    def learn_from_example(
        self,
        candidates: dict[str, Features],
        expert_action: str,
        learning_rate: float | None = None,
    ) -> bool:
        """Imitation learning: make ``expert_action`` outrank the alternatives.

        Perceptron update — only corrects when the policy would have chosen
        differently. Returns True if a correction was applied.
        """
        if expert_action not in candidates:
            raise KeyError(f"expert action {expert_action!r} not among candidates")
        lr = self.learning_rate if learning_rate is None else learning_rate
        ranked = self.rank(candidates)
        predicted = ranked[0][0]
        if predicted == expert_action:
            return False
        # Reward the expert's choice, penalize the one we wrongly preferred.
        self.reinforce(expert_action, candidates[expert_action], 1.0, lr)
        self.reinforce(predicted, candidates[predicted], -1.0, lr)
        return True

    def fit(
        self,
        dataset: list[tuple[dict[str, Features], str]],
        epochs: int = 5,
        learning_rate: float | None = None,
    ) -> dict[str, Any]:
        """Batch imitation training over ``(candidates, expert_action)`` pairs.

        Supervised decision learning: the training signal is *which action*
        an expert took in a state, never anything about perception.
        """
        history = []
        for epoch in range(epochs):
            corrections = sum(
                self.learn_from_example(cands, expert, learning_rate) for cands, expert in dataset
            )
            accuracy = 1.0 - (corrections / len(dataset)) if dataset else 1.0
            history.append({"epoch": epoch, "corrections": corrections, "accuracy": accuracy})
            if corrections == 0:
                break  # converged
        return {"epochs_run": len(history), "history": history}

    # --- persistence -------------------------------------------------------

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"learning_rate": self.learning_rate, "weights": self.weights}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> LinearPolicy:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(weights=data.get("weights", {}), learning_rate=data.get("learning_rate", 0.1))

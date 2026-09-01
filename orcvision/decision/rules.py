"""Config-driven rule engine with safe expression evaluation.

Rules are evaluated per-detection against a small, fixed set of fields.
Expressions are parsed with :mod:`ast` and evaluated by an explicit
whitelist walker — there is **no** ``eval()`` of arbitrary code and no
``exec()``. Only boolean/comparison logic over detection fields and
literals is permitted; names, attribute access, calls, comprehensions,
and subscripting are all rejected.

Supported per-detection variables:
    label (str), confidence (float), bbox (tuple), track_id (int|None),
    depth_m (float|None), class_id (int|None)

Example rule (YAML):
    - when: "label == 'person' and confidence > 0.8"
      action: "alert"
      cooldown_s: 30
"""

from __future__ import annotations

import ast
import time
from dataclasses import dataclass, field
from typing import Any

from orcvision.events import Detection, PerceptionEvent

ALLOWED_FIELDS = {"label", "confidence", "bbox", "track_id", "depth_m", "class_id"}


class SafeExpressionError(ValueError):
    """Raised when a rule expression contains a disallowed construct."""


# Node types the evaluator understands. Anything else is rejected at
# compile time, so untrusted config can never reach a call/import/etc.
_ALLOWED_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.USub,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Is,
    ast.IsNot,
    ast.In,
    ast.NotIn,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.List,
    ast.Tuple,
)


class _SafeEvaluator(ast.NodeVisitor):
    def __init__(self, variables: dict[str, Any]) -> None:
        self.vars = variables

    def visit(self, node: ast.AST) -> Any:
        if not isinstance(node, _ALLOWED_NODES):
            raise SafeExpressionError(f"Disallowed expression element: {type(node).__name__}")
        return super().visit(node)

    def generic_visit(self, node: ast.AST) -> Any:  # pragma: no cover - safety net
        raise SafeExpressionError(f"Unsupported node: {type(node).__name__}")

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        values = [self.visit(v) for v in node.values]
        if isinstance(node.op, ast.And):
            return all(values)
        return any(values)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.Not):
            return not operand
        return -operand  # USub

    def visit_Compare(self, node: ast.Compare) -> Any:
        left = self.visit(node.left)
        result = True
        for op, comparator in zip(node.ops, node.comparators, strict=False):
            right = self.visit(comparator)
            result = result and self._compare(op, left, right)
            left = right
        return result

    @staticmethod
    def _compare(op: ast.cmpop, left: Any, right: Any) -> bool:
        if isinstance(op, ast.Eq):
            return left == right
        if isinstance(op, ast.NotEq):
            return left != right
        if isinstance(op, ast.Lt):
            return left < right
        if isinstance(op, ast.LtE):
            return left <= right
        if isinstance(op, ast.Gt):
            return left > right
        if isinstance(op, ast.GtE):
            return left >= right
        if isinstance(op, ast.Is):
            return left is right
        if isinstance(op, ast.IsNot):
            return left is not right
        if isinstance(op, ast.In):
            return left in right
        if isinstance(op, ast.NotIn):
            return left not in right
        raise SafeExpressionError(f"Unsupported comparison: {type(op).__name__}")

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id in ("None", "True", "False"):  # handled as constants normally
            return {"None": None, "True": True, "False": False}[node.id]
        if node.id not in self.vars:
            raise SafeExpressionError(
                f"Unknown variable {node.id!r}; allowed: {sorted(ALLOWED_FIELDS)}"
            )
        return self.vars[node.id]

    def visit_Constant(self, node: ast.Constant) -> Any:
        return node.value

    def visit_List(self, node: ast.List) -> Any:
        return [self.visit(e) for e in node.elts]

    def visit_Tuple(self, node: ast.Tuple) -> Any:
        return tuple(self.visit(e) for e in node.elts)


def _compile(expression: str) -> ast.Expression:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise SafeExpressionError(f"Invalid rule expression: {expression!r}") from exc
    # Validate the whole tree up front so bad config fails fast.
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise SafeExpressionError(
                f"Disallowed expression element {type(node).__name__} in {expression!r}"
            )
    return tree


@dataclass
class Rule:
    """A single condition/action rule."""

    when: str
    action: str = "alert"
    cooldown_s: float = 0.0
    min_consecutive_frames: int = 1
    _tree: ast.Expression = field(init=False, repr=False)
    _last_fired: float = field(default=0.0, init=False, repr=False)
    # track_id -> consecutive match count
    _streaks: dict[Any, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._tree = _compile(self.when)

    def matches(self, detection: Detection) -> bool:
        evaluator = _SafeEvaluator(
            {
                "label": detection.label,
                "confidence": detection.confidence,
                "bbox": detection.bbox,
                "track_id": detection.track_id,
                "depth_m": detection.depth_m,
                "class_id": detection.class_id,
            }
        )
        return bool(evaluator.visit(self._tree))

    def _key(self, detection: Detection) -> Any:
        return detection.track_id if detection.track_id is not None else id(detection)

    def evaluate(self, detection: Detection, now: float) -> str | None:
        """Return an action string if the rule fires for this detection."""
        if not self.matches(detection):
            if detection.track_id is not None:
                self._streaks.pop(detection.track_id, None)
            return None

        # min_consecutive_frames gating (per track).
        if self.min_consecutive_frames > 1:
            key = self._key(detection)
            self._streaks[key] = self._streaks.get(key, 0) + 1
            if self._streaks[key] < self.min_consecutive_frames:
                return None

        # cooldown gating (rule-global).
        if self.cooldown_s > 0 and (now - self._last_fired) < self.cooldown_s:
            return None

        self._last_fired = now
        return self.action


class RuleEngine:
    """Evaluate a set of rules against each PerceptionEvent's detections."""

    def __init__(self, rules: list[Rule] | None = None) -> None:
        self.rules = rules or []

    @classmethod
    def from_config(cls, rules_config: list[dict[str, Any]] | None) -> RuleEngine:
        rules = []
        for rc in rules_config or []:
            rules.append(
                Rule(
                    when=rc["when"],
                    action=rc.get("action", "alert"),
                    cooldown_s=float(rc.get("cooldown_s", 0.0)),
                    min_consecutive_frames=int(rc.get("min_consecutive_frames", 1)),
                )
            )
        return cls(rules)

    def apply(self, event: PerceptionEvent, now: float | None = None) -> PerceptionEvent:
        """Mutate ``event.alerts`` in place with any fired rule actions."""
        now = time.monotonic() if now is None else now
        for rule in self.rules:
            for detection in event.detections:
                action = rule.evaluate(detection, now)
                if action:
                    msg = f"{action}: {rule.when}"
                    if msg not in event.alerts:
                        event.alerts.append(msg)
        return event

"""Config-driven rule engine with safe expression evaluation.

Two rule scopes are supported, both driven by the same safe evaluator:

**Per-detection rules** (``decision.rules``) are evaluated once per
detection against a small, fixed set of fields:
    label (str), confidence (float), bbox (tuple), track_id (int|None),
    depth_m (float|None), class_id (int|None)

**Event-scope rules** (``decision.event_rules``) are evaluated once per
frame over *all* detections, via a small whitelist of pure aggregate
helpers — useful for "how many" / "is there both X and Y" decisions that a
single detection cannot express:
    count(label=None)   -> number of detections (optionally of one label)
    exists(label)       -> True if any detection has that label
    max_conf(label=None)-> highest confidence (0.0 if none match)
    min_depth(label=None)-> nearest depth_m (None if unknown/no match)

Expressions are parsed with :mod:`ast` and evaluated by an explicit
whitelist walker — there is **no** ``eval()`` of arbitrary code and no
``exec()``. Only boolean/comparison logic over the exposed fields,
literals, and (for event rules) the whitelisted helper calls is permitted;
attribute access, arbitrary names, lambdas, comprehensions, and
subscripting are all rejected at compile time.

Example rules (YAML)::

    decision:
      rules:
        - when: "label == 'person' and confidence > 0.8"
          action: "alert"
          message: "Confident person detected"
          severity: "warning"
          cooldown_s: 30
      event_rules:
        - when: "count('person') >= 3"
          message: "Crowd forming"
          severity: "critical"
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
        # Ordering comparisons against an unknown value (e.g. a null
        # depth_m, or min_depth() with no matching detection) are treated
        # as "does not match" rather than raising, so a rule like
        # "min_depth('obstacle') < 1.5" quietly stays silent when depth is
        # unavailable instead of crashing the pipeline.
        if isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
            if left is None or right is None:
                return False
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


# Names of the aggregate helper functions callable from an event-scope
# expression. This is a *closed* set: any other call target is rejected at
# compile time, so untrusted config can only ever invoke these four pure,
# read-only functions over the frame's detections.
AGGREGATE_FUNCTIONS = frozenset({"count", "exists", "max_conf", "min_depth"})

# Event-scope expressions additionally permit calls (to the whitelisted
# helpers only) and keyword arguments to them.
_AGG_ALLOWED_NODES = (*_ALLOWED_NODES, ast.Call, ast.keyword)


def _aggregate_namespace(detections: list[Detection]) -> dict[str, Any]:
    """Build the whitelisted helper functions bound to one frame."""

    def _select(label: str | None) -> list[Detection]:
        if label is None:
            return list(detections)
        return [d for d in detections if d.label == label]

    def count(label: str | None = None) -> int:
        return len(_select(label))

    def exists(label: str) -> bool:
        return len(_select(label)) > 0

    def max_conf(label: str | None = None) -> float:
        return max((d.confidence for d in _select(label)), default=0.0)

    def min_depth(label: str | None = None) -> float | None:
        depths = [d.depth_m for d in _select(label) if d.depth_m is not None]
        return min(depths) if depths else None

    return {"count": count, "exists": exists, "max_conf": max_conf, "min_depth": min_depth}


class _AggregateEvaluator(_SafeEvaluator):
    """Safe evaluator for event-scope rules: adds whitelisted helper calls.

    Only bare-name calls to functions in :data:`AGGREGATE_FUNCTIONS` are
    permitted. Attribute-based calls (``x.y()``), starred/`**` unpacking,
    and any non-whitelisted callable are rejected — keeping the same
    no-arbitrary-code guarantee as the per-detection evaluator.
    """

    def visit(self, node: ast.AST) -> Any:
        if not isinstance(node, _AGG_ALLOWED_NODES):
            raise SafeExpressionError(f"Disallowed expression element: {type(node).__name__}")
        return super(_SafeEvaluator, self).visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        if not isinstance(node.func, ast.Name) or node.func.id not in AGGREGATE_FUNCTIONS:
            raise SafeExpressionError(
                f"Only these functions may be called: {sorted(AGGREGATE_FUNCTIONS)}"
            )
        func = self.vars[node.func.id]
        args = [self.visit(a) for a in node.args]
        kwargs = {kw.arg: self.visit(kw.value) for kw in node.keywords if kw.arg is not None}
        if any(kw.arg is None for kw in node.keywords):
            raise SafeExpressionError("`**kwargs` unpacking is not allowed")
        return func(*args, **kwargs)


def _compile(expression: str, allowed_nodes: tuple[type, ...] = _ALLOWED_NODES) -> ast.Expression:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise SafeExpressionError(f"Invalid rule expression: {expression!r}") from exc
    # Validate the whole tree up front so bad config fails fast.
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise SafeExpressionError(
                f"Disallowed expression element {type(node).__name__} in {expression!r}"
            )
        # For event rules, calls are only allowed to the whitelisted names.
        if isinstance(node, ast.Call) and (
            not isinstance(node.func, ast.Name) or node.func.id not in AGGREGATE_FUNCTIONS
        ):
            raise SafeExpressionError(
                f"Only these functions may be called: {sorted(AGGREGATE_FUNCTIONS)}"
            )
    return tree


def _format_alert(action: str, when: str, name: str, message: str, severity: str) -> str:
    """Build the alert string appended to ``PerceptionEvent.alerts``.

    Backward compatible: with no ``message``/``severity``/``name`` set the
    output is the historical ``"<action>: <when>"``. When provided, a custom
    message and an optional ``[severity]`` prefix make alerts human-readable
    and prioritizable downstream.
    """
    body = message or f"{action}: {when}"
    if name:
        body = f"{name}: {body}"
    if severity:
        return f"[{severity}] {body}"
    return body


@dataclass
class Rule:
    """A single per-detection condition/action rule."""

    when: str
    action: str = "alert"
    cooldown_s: float = 0.0
    min_consecutive_frames: int = 1
    name: str = ""
    message: str = ""
    severity: str = ""
    _tree: ast.Expression = field(init=False, repr=False)
    _last_fired: float = field(default=0.0, init=False, repr=False)
    # track_id -> consecutive match count
    _streaks: dict[Any, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._tree = _compile(self.when)

    def alert_message(self) -> str:
        return _format_alert(self.action, self.when, self.name, self.message, self.severity)

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


@dataclass
class EventRule:
    """A frame-scope rule evaluated once over *all* detections.

    Unlike :class:`Rule`, the expression may call the whitelisted aggregate
    helpers (``count``, ``exists``, ``max_conf``, ``min_depth``) to make
    decisions about the whole frame, e.g. ``"count('person') >= 3"`` or
    ``"exists('person') and exists('vehicle')"``.
    """

    when: str
    action: str = "alert"
    cooldown_s: float = 0.0
    min_consecutive_frames: int = 1
    name: str = ""
    message: str = ""
    severity: str = ""
    _tree: ast.Expression = field(init=False, repr=False)
    _last_fired: float = field(default=0.0, init=False, repr=False)
    _streak: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._tree = _compile(self.when, allowed_nodes=_AGG_ALLOWED_NODES)

    def alert_message(self) -> str:
        return _format_alert(self.action, self.when, self.name, self.message, self.severity)

    def matches(self, detections: list[Detection]) -> bool:
        evaluator = _AggregateEvaluator(_aggregate_namespace(detections))
        return bool(evaluator.visit(self._tree))

    def evaluate(self, detections: list[Detection], now: float) -> str | None:
        """Return an action string if the rule fires for this frame."""
        if not self.matches(detections):
            self._streak = 0
            return None

        # min_consecutive_frames gating (per frame streak).
        if self.min_consecutive_frames > 1:
            self._streak += 1
            if self._streak < self.min_consecutive_frames:
                return None

        # cooldown gating (rule-global).
        if self.cooldown_s > 0 and (now - self._last_fired) < self.cooldown_s:
            return None

        self._last_fired = now
        return self.action


class RuleEngine:
    """Evaluate per-detection and event-scope rules against each event."""

    def __init__(
        self,
        rules: list[Rule] | None = None,
        event_rules: list[EventRule] | None = None,
    ) -> None:
        self.rules = rules or []
        self.event_rules = event_rules or []

    @classmethod
    def from_config(
        cls,
        rules_config: list[dict[str, Any]] | None,
        event_rules_config: list[dict[str, Any]] | None = None,
    ) -> RuleEngine:
        rules = [
            Rule(
                when=rc["when"],
                action=rc.get("action", "alert"),
                cooldown_s=float(rc.get("cooldown_s", 0.0)),
                min_consecutive_frames=int(rc.get("min_consecutive_frames", 1)),
                name=str(rc.get("name", "")),
                message=str(rc.get("message", "")),
                severity=str(rc.get("severity", "")),
            )
            for rc in rules_config or []
        ]
        event_rules = [
            EventRule(
                when=rc["when"],
                action=rc.get("action", "alert"),
                cooldown_s=float(rc.get("cooldown_s", 0.0)),
                min_consecutive_frames=int(rc.get("min_consecutive_frames", 1)),
                name=str(rc.get("name", "")),
                message=str(rc.get("message", "")),
                severity=str(rc.get("severity", "")),
            )
            for rc in event_rules_config or []
        ]
        return cls(rules, event_rules)

    def apply(self, event: PerceptionEvent, now: float | None = None) -> PerceptionEvent:
        """Mutate ``event.alerts`` in place with any fired rule actions."""
        now = time.monotonic() if now is None else now
        for rule in self.rules:
            for detection in event.detections:
                if rule.evaluate(detection, now):
                    msg = rule.alert_message()
                    if msg not in event.alerts:
                        event.alerts.append(msg)
        for event_rule in self.event_rules:
            if event_rule.evaluate(event.detections, now):
                msg = event_rule.alert_message()
                if msg not in event.alerts:
                    event.alerts.append(msg)
        return event

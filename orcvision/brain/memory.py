"""Memory — bounded working memory and decaying long-term memory.

The design goal here is *useful* memory, not a database of everything the
camera has ever seen. Both stores have hard caps so the brain's footprint
stays flat during an indefinitely long run on an embedded board.

``WorkingMemory``
    A ring buffer of recent items (events, decisions, outcomes) with a
    capacity **and** a retention window. Answers "what just happened".

``LongTermMemory``
    Keyed traces with importance, reinforcement, exponential decay and
    eviction. Answers "what do I know about situations like this" — this is
    the store that makes the brain choose differently the second time.

Long-term traces are keyed, so repeat observations *reinforce* an existing
trace instead of appending a duplicate.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

# Trace kinds used by the default decision layer.
KIND_OUTCOME = "outcome"  # action X in situation Y succeeded/failed
KIND_OBJECT = "object"  # this object has been seen before
KIND_EVENT = "event"  # a notable thing happened
KIND_DECISION = "decision"  # a decision the brain made


@dataclass(slots=True)
class MemoryItem:
    """One entry in working memory."""

    kind: str
    timestamp: float
    content: Any
    label: str | None = None

    def describe(self) -> str:
        return f"[{self.kind}] {self.content}"


class WorkingMemory:
    """Short-term memory: bounded, time-windowed, cheap to scan."""

    def __init__(self, capacity: int = 64, retention_s: float = 30.0) -> None:
        self.retention_s = retention_s
        self._items: deque[MemoryItem] = deque(maxlen=capacity)

    @property
    def capacity(self) -> int:
        return self._items.maxlen or 0

    @capacity.setter
    def capacity(self, value: int) -> None:
        """Resize the ring buffer.

        A property rather than a plain attribute because ``deque.maxlen`` is
        read-only: assigning ``capacity`` after construction would otherwise
        silently do nothing and leave the footprint unbounded.
        """
        if value != self.capacity:
            self._items = deque(self._items, maxlen=value)

    def __len__(self) -> int:
        return len(self._items)

    def add(self, kind: str, content: Any, timestamp: float, label: str | None = None) -> None:
        self._items.append(MemoryItem(kind, timestamp, content, label))

    def prune(self, now: float) -> None:
        """Drop anything older than the retention window."""
        cutoff = now - self.retention_s
        while self._items and self._items[0].timestamp < cutoff:
            self._items.popleft()

    def recent(
        self, kind: str | None = None, since: float | None = None, limit: int | None = None
    ) -> list[MemoryItem]:
        """Most-recent-first view of working memory."""
        items = [
            it
            for it in reversed(self._items)
            if (kind is None or it.kind == kind) and (since is None or it.timestamp >= since)
        ]
        return items[:limit] if limit else items

    def count(self, kind: str, since: float | None = None) -> int:
        return len(self.recent(kind=kind, since=since))

    def clear(self) -> None:
        self._items.clear()


@dataclass(slots=True)
class MemoryTrace:
    """A long-term memory with an importance that decays unless reinforced."""

    key: str
    kind: str
    content: Any
    importance: float = 0.5
    created_at: float = 0.0
    last_reinforced: float = 0.0
    last_access: float = 0.0
    hits: int = 1

    def strength(self, now: float, half_life_s: float) -> float:
        """Current retrievability: importance decayed by time since reinforcement.

        Frequently reinforced traces decay from a higher base, so a lesson
        learned ten times outlives one learned once.
        """
        if half_life_s <= 0:
            return self.importance
        elapsed = max(0.0, now - self.last_reinforced)
        decay = math.exp(-elapsed * math.log(2) / half_life_s)
        # Repetition bonus, saturating so it can never dominate importance.
        repetition = 1.0 + math.log1p(self.hits - 1) * 0.25
        return self.importance * decay * repetition


class LongTermMemory:
    """Keyed, decaying, self-pruning memory.

    Supports the mechanisms the brain actually needs: relevance (strength),
    importance, decay, retrieval, updating, forgetting and deduplication.
    """

    def __init__(
        self,
        capacity: int = 256,
        half_life_s: float = 600.0,
        forget_below: float = 0.05,
    ) -> None:
        self.capacity = capacity
        self.half_life_s = half_life_s
        self.forget_below = forget_below
        self._traces: dict[str, MemoryTrace] = {}

    def __len__(self) -> int:
        return len(self._traces)

    def remember(
        self,
        key: str,
        content: Any,
        now: float,
        kind: str = KIND_EVENT,
        importance: float = 0.5,
    ) -> MemoryTrace:
        """Store or reinforce a trace. Repeat keys deduplicate by design."""
        trace = self._traces.get(key)
        if trace is None:
            trace = MemoryTrace(
                key=key,
                kind=kind,
                content=content,
                importance=importance,
                created_at=now,
                last_reinforced=now,
                last_access=now,
            )
            self._traces[key] = trace
        else:
            trace.content = content
            trace.hits += 1
            trace.last_reinforced = now
            trace.last_access = now
            # Reinforcement raises importance with diminishing returns.
            trace.importance = min(1.0, trace.importance + (1.0 - trace.importance) * 0.3)
            trace.importance = max(trace.importance, importance)
        self.forget(now)
        return trace

    def recall(self, key: str, now: float) -> MemoryTrace | None:
        """Retrieve a trace if it is still strong enough to be remembered."""
        trace = self._traces.get(key)
        if trace is None:
            return None
        if trace.strength(now, self.half_life_s) < self.forget_below:
            del self._traces[key]
            return None
        trace.last_access = now
        return trace

    def search(self, kind: str | None = None, prefix: str | None = None) -> list[MemoryTrace]:
        return [
            t
            for t in self._traces.values()
            if (kind is None or t.kind == kind) and (prefix is None or t.key.startswith(prefix))
        ]

    def forget(self, now: float) -> int:
        """Evict decayed traces, then the weakest if still over capacity."""
        removed = [
            k
            for k, t in self._traces.items()
            if t.strength(now, self.half_life_s) < self.forget_below
        ]
        for key in removed:
            del self._traces[key]

        overflow = len(self._traces) - self.capacity
        if overflow > 0:
            weakest = sorted(
                self._traces.items(), key=lambda kv: kv[1].strength(now, self.half_life_s)
            )[:overflow]
            for key, _ in weakest:
                del self._traces[key]
                removed.append(key)
        return len(removed)

    def snapshot(self) -> dict[str, Any]:
        """Serializable view — for persistence across power cycles."""
        return {
            k: {
                "kind": t.kind,
                "content": t.content,
                "importance": t.importance,
                "created_at": t.created_at,
                "last_reinforced": t.last_reinforced,
                "hits": t.hits,
            }
            for k, t in self._traces.items()
        }

    def restore(self, data: dict[str, Any]) -> None:
        """Load a snapshot produced by :meth:`snapshot`."""
        self._traces = {
            k: MemoryTrace(
                key=k,
                kind=v.get("kind", KIND_EVENT),
                content=v.get("content"),
                importance=float(v.get("importance", 0.5)),
                created_at=float(v.get("created_at", 0.0)),
                last_reinforced=float(v.get("last_reinforced", 0.0)),
                last_access=float(v.get("last_reinforced", 0.0)),
                hits=int(v.get("hits", 1)),
            )
            for k, v in data.items()
        }


@dataclass
class Memory:
    """Convenience bundle of both stores, as the brain uses them together."""

    working: WorkingMemory = field(default_factory=WorkingMemory)
    longterm: LongTermMemory = field(default_factory=LongTermMemory)

    def prune(self, now: float) -> None:
        self.working.prune(now)
        self.longterm.forget(now)

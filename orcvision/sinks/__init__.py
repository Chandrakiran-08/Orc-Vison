"""Event sinks — where PerceptionEvents go."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from orcvision.events import PerceptionEvent


@runtime_checkable
class SinkProtocol(Protocol):
    def emit(self, event: PerceptionEvent) -> None: ...

    def close(self) -> None: ...


__all__ = ["SinkProtocol"]

"""Model abstraction — a protocol, not a framework."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from orcvision.events import Detection


@runtime_checkable
class ModelProtocol(Protocol):
    """Any object that turns a frame into a list of detections."""

    def infer(self, frame: Any) -> list[Detection]:
        """Run inference on an RGB numpy frame and return detections."""
        ...


__all__ = ["ModelProtocol"]

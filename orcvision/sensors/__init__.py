"""Sensor abstraction — the extensibility story.

Any class implementing :class:`SensorProtocol` is a valid vision sensor.
No registration required: return a :class:`SensorFrame` from ``read()``
and release resources in ``release()``.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel


class SensorFrame(BaseModel):
    """A single frame from any vision sensor.

    ``rgb`` and ``depth`` hold numpy arrays; ``arbitrary_types_allowed``
    keeps Pydantic from trying to validate/serialize them.
    """

    model_config = {"arbitrary_types_allowed": True}

    rgb: Any  # np.ndarray (H, W, 3)
    depth: Any | None = None  # np.ndarray (H, W) or None
    timestamp: float
    modality: str  # "rgb", "rgbd", "thermal", "stereo"


@runtime_checkable
class SensorProtocol(Protocol):
    """Structural interface every sensor must satisfy."""

    def read(self) -> SensorFrame | None:
        """Return the next frame, or None when the stream is exhausted."""
        ...

    def release(self) -> None:
        """Release any underlying hardware / file handles."""
        ...


__all__ = ["SensorFrame", "SensorProtocol"]

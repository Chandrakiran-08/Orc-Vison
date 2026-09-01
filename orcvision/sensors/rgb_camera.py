"""RGB camera sensor backed by OpenCV VideoCapture.

Accepts an int device index (``0`` -> /dev/video0), a file path, or an
``rtsp://`` URL. Produces ``modality="rgb"`` frames.
"""

from __future__ import annotations

import os
import time

import cv2

from orcvision.sensors import SensorFrame


class RGBCamera:
    """OpenCV-backed RGB sensor for webcams, video files, and RTSP streams."""

    def __init__(
        self,
        source: int | str = 0,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        # Expand ~ / env vars for file-path sources (int indices / URLs pass through).
        if isinstance(source, str) and any(s in source for s in ("~", "$")):
            source = os.path.expanduser(os.path.expandvars(source))
        self.source = source
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open video source {source!r}. "
                "Check the device index / path / URL and permissions "
                "(is your user in the 'video' group?)."
            )
        if width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    @property
    def source_name(self) -> str:
        if isinstance(self.source, int):
            return f"camera:{self.source}"
        if str(self.source).startswith("rtsp://"):
            return f"rtsp:{self.source}"
        return f"file:{self.source}"

    def read(self) -> SensorFrame | None:
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return SensorFrame(rgb=frame, depth=None, timestamp=time.time(), modality="rgb")

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

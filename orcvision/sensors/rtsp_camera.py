"""RTSP network-stream sensor — a thin wrapper over :class:`RGBCamera`.

RTSP URLs already work through ``RGBCamera``; this exists as an explicit,
discoverable entry point and validates that the source is an RTSP URL.
"""

from __future__ import annotations

from orcvision.sensors.rgb_camera import RGBCamera


class RTSPCamera(RGBCamera):
    """RGB sensor for an ``rtsp://`` network stream."""

    def __init__(
        self,
        source: str,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        if not str(source).startswith("rtsp://"):
            raise ValueError(f"RTSPCamera expects an rtsp:// URL, got {source!r}")
        super().__init__(source=source, width=width, height=height)

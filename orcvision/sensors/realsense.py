"""Intel RealSense sensor (lazy import, ``[realsense]`` extra).

Produces ``modality="rgbd"`` frames with both ``rgb`` and aligned
``depth`` populated. ``depth`` is a float32 metre-scale map.

.. warning::
   This backend is written but **UNVERIFIED** — the maintainer has no
   RealSense hardware to test against. If pyrealsense2 is not installed,
   importing this module still succeeds; the failure is raised only when
   :class:`RealSenseCamera` is actually constructed, so the rest of the
   CLI keeps working regardless.
"""

from __future__ import annotations

import time

from orcvision.sensors import SensorFrame


class RealSenseCamera:
    """RGB-D sensor backed by pyrealsense2 with depth aligned to colour."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                "pyrealsense2 is not installed. Install the optional extra: "
                "`pip install orcvision[realsense]`."
            ) from exc

        import numpy as np  # noqa: F401

        self._rs = rs
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = self._pipeline.start(config)

        # Depth scale converts raw z16 units to metres.
        depth_sensor = profile.get_device().first_depth_sensor()
        self._depth_scale = depth_sensor.get_depth_scale()
        self._align = rs.align(rs.stream.color)

    def read(self) -> SensorFrame | None:
        import numpy as np

        frames = self._pipeline.wait_for_frames()
        frames = self._align.process(frames)
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if not color or not depth:
            return None
        rgb = np.asanyarray(color.get_data())
        depth_m = np.asanyarray(depth.get_data()).astype("float32") * self._depth_scale
        return SensorFrame(rgb=rgb, depth=depth_m, timestamp=time.time(), modality="rgbd")

    @property
    def source_name(self) -> str:
        return "realsense:0"

    def release(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None

"""Stereo camera pair — documented extension stub (NOT implemented in v0.1).

Stereo support is intentionally left as an extension point. To add it,
implement :class:`~orcvision.sensors.SensorProtocol` for your stereo rig
and return ``SensorFrame`` objects with ``modality="stereo"``.

Contributor guide
-----------------
1. Capture the left and right frames (two ``cv2.VideoCapture`` handles, a
   single side-by-side UVC stream you split, or a vendor SDK).
2. Rectify both views using your calibration (``cv2.stereoRectify`` +
   ``cv2.initUndistortRectifyMap``).
3. Compute a disparity map (``cv2.StereoSGBM_create``) and convert it to a
   metric depth map with ``depth = f * baseline / disparity``.
4. Put the rectified left image in ``rgb`` and the metric depth map in
   ``depth``; set ``modality="stereo"`` and a monotonic ``timestamp``.
5. Release both captures in ``release()``.

No registration is needed — any class satisfying ``SensorProtocol`` is a
valid sensor.
"""

from __future__ import annotations

from orcvision.sensors import SensorFrame


class StereoCamera:
    """Placeholder stereo sensor — see module docstring to implement."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "Stereo camera support is a documented extension point and is not "
            "implemented in orcvision v0.1. See the module docstring in "
            "orcvision/sensors/stereo_camera.py for a step-by-step guide to "
            "implementing SensorProtocol for your stereo rig."
        )

    def read(self) -> SensorFrame | None:  # pragma: no cover - never constructed
        raise NotImplementedError

    def release(self) -> None:  # pragma: no cover - never constructed
        raise NotImplementedError

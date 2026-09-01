"""Thermal camera — documented extension stub (NOT implemented in v0.1).

Thermal support is intentionally left as an extension point. To add it,
implement :class:`~orcvision.sensors.SensorProtocol` for your specific
thermal device and return ``SensorFrame`` objects with ``modality="thermal"``.

Contributor guide
-----------------
1. Open your device (many thermal cores expose a UVC video stream, so
   ``cv2.VideoCapture`` may work; others need a vendor SDK — e.g. FLIR
   Spinnaker, Seek Thermal, or a raw radiometric USB protocol).
2. Convert each raw radiometric frame to an 8-bit BGR image for the
   ``rgb`` field (apply a colormap such as ``cv2.applyColorMap`` with
   ``cv2.COLORMAP_INFERNO``). Optionally keep the calibrated temperature
   map in the ``depth`` field if a model can consume it.
3. Set ``modality="thermal"`` and a monotonic ``timestamp``.
4. Release the device in ``release()``.

No registration is needed — any class satisfying ``SensorProtocol`` is a
valid sensor. Wire it into the CLI by extending ``--sensor-type`` in
``orcvision/cli.py`` (or instantiate it directly from the library API).
"""

from __future__ import annotations

from orcvision.sensors import SensorFrame


class ThermalCamera:
    """Placeholder thermal sensor — see module docstring to implement."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "Thermal camera support is a documented extension point and is not "
            "implemented in orcvision v0.1. See the module docstring in "
            "orcvision/sensors/thermal_camera.py for a step-by-step guide to "
            "implementing SensorProtocol for your thermal device."
        )

    def read(self) -> SensorFrame | None:  # pragma: no cover - never constructed
        raise NotImplementedError

    def release(self) -> None:  # pragma: no cover - never constructed
        raise NotImplementedError

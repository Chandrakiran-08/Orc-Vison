"""Sensor tests using mock frames — no real camera / hardware required."""

import numpy as np

from orcvision.sensors import SensorFrame, SensorProtocol


class MockSensor:
    """A SensorProtocol implementation that yields synthetic frames."""

    def __init__(self, n_frames=3):
        self.n_frames = n_frames
        self._i = 0
        self.released = False

    def read(self):
        if self._i >= self.n_frames:
            return None
        self._i += 1
        rgb = np.zeros((48, 64, 3), dtype=np.uint8)
        return SensorFrame(rgb=rgb, depth=None, timestamp=float(self._i), modality="rgb")

    def release(self):
        self.released = True


def test_mock_sensor_satisfies_protocol():
    sensor = MockSensor()
    assert isinstance(sensor, SensorProtocol)


def test_mock_sensor_stream_and_release():
    sensor = MockSensor(n_frames=2)
    frames = []
    while True:
        f = sensor.read()
        if f is None:
            break
        frames.append(f)
    assert len(frames) == 2
    assert frames[0].modality == "rgb"
    assert frames[0].rgb.shape == (48, 64, 3)
    sensor.release()
    assert sensor.released is True


def test_sensor_frame_holds_depth():
    depth = np.ones((48, 64), dtype=np.float32)
    frame = SensorFrame(rgb=np.zeros((48, 64, 3)), depth=depth, timestamp=1.0, modality="rgbd")
    assert frame.modality == "rgbd"
    assert frame.depth.shape == (48, 64)


def test_thermal_and_stereo_are_stubs():
    from orcvision.sensors.stereo_camera import StereoCamera
    from orcvision.sensors.thermal_camera import ThermalCamera

    for stub in (ThermalCamera, StereoCamera):
        try:
            stub()
        except NotImplementedError as exc:
            assert "extension point" in str(exc)
        else:  # pragma: no cover
            raise AssertionError(f"{stub.__name__} should raise NotImplementedError")

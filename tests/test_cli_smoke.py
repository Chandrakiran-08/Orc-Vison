"""CLI smoke tests — info/doctor exit 0, and the run loop over a mock."""

from typer.testing import CliRunner

from orcvision.cli import app

runner = CliRunner()


def test_info_exit_zero():
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "orcvision" in result.stdout


def test_doctor_exit_zero():
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Python:" in result.stdout


def test_version_exit_zero():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0


def test_run_pipeline_with_mock(monkeypatch):
    """Drive the full run loop with a mock sensor + mock model, no hardware."""
    import numpy as np

    from orcvision import cli
    from orcvision.events import Detection
    from orcvision.sensors import SensorFrame

    class _MockSensor:
        source_name = "mock:0"

        def __init__(self):
            self._n = 0

        def read(self):
            if self._n >= 2:
                return None
            self._n += 1
            return SensorFrame(
                rgb=np.zeros((8, 8, 3), dtype=np.uint8),
                depth=None,
                timestamp=float(self._n),
                modality="rgb",
            )

        def release(self):
            pass

    class _MockModel:
        def infer(self, frame):
            return [Detection(label="person", confidence=0.99, bbox=(0, 0, 4, 4))]

    monkeypatch.setattr(cli, "_build_sensor", lambda *a, **k: _MockSensor())
    monkeypatch.setattr("orcvision.models.resolve.resolve_model", lambda *a, **k: _MockModel())

    result = runner.invoke(app, ["run", "--source", "mock", "--sink", "stdout"])
    assert result.exit_code == 0, result.output
    assert '"label":"person"' in result.stdout
    assert "Processed 2 frame(s)." in result.output

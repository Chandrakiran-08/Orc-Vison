# Contributing to Orc-Vison

Thanks for your interest! Orc-Vison is Apache-2.0 licensed.

## Dev setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[cpu,dev]"
```

Run the checks:

```bash
ruff check .
ruff format --check .
pytest
python -m orcvision doctor
```

## Adding a new sensor

Any class implementing `SensorProtocol` (see `orcvision/sensors/__init__.py`)
is a valid sensor — no registration required. Return a `SensorFrame` from
`read()` and clean up in `release()`. See `thermal_camera.py` /
`stereo_camera.py` for documented extension stubs.

## Ground rules

- Base dependencies stay light — heavy/optional deps go behind extras and
  are lazy-imported.
- No GUI backends: `opencv-python-headless` only, never `cv2.imshow()`.
- The decision layer uses safe expression evaluation only — no `eval()`
  of arbitrary code, no `exec()`.
- Tests must pass with zero physical hardware.

By contributing you agree your work is licensed under Apache-2.0.

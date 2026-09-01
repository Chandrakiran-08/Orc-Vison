# Example: security person alert

Watch a camera and raise an MQTT alert when a confident person appears, or
when a tracked person lingers across several frames. This is the reference
example for the **decision-rules** feature.

## Run

```bash
# Optional: a broker + the software receiver to see alerts land
mosquitto -v &
python -m orcvision.mqtt_simulator --topic orcvision/events &

# Run against your webcam using this config
python -m orcvision run --config examples/security_person_alert/config.yaml
```

To test without a camera, edit `config.yaml` and set
`sensor.source: tests/fixtures/sample.mp4`.

## What to notice

- `decision.rules` populate `PerceptionEvent.alerts`.
- `cooldown_s: 30` stops the first rule from firing more than once per 30s.
- `min_consecutive_frames: 5` requires a *tracked* person (so
  `tracker.enabled: true`) to persist before the second rule fires.
- Rules use safe expressions over detection fields only — no arbitrary
  code execution.

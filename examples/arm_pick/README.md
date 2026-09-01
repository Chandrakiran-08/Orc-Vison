# Example: robotic arm pick

Detect a graspable object (bottle/cup) and alert when it is within reach,
writing events to a JSONL file a motion controller can tail.

## Run

```bash
pip install orcvision[cpu,realsense]   # realsense provides metric depth
python -m orcvision run --config examples/arm_pick/config.yaml
```

> **Note:** the RealSense backend is written but **unverified** — the
> maintainer has no RealSense hardware. To try this with a plain webcam,
> set `sensor.type: rgb` and either drop the `depth_m` clause from the rule
> or add `depth.enabled: true` with `pip install orcvision[depth]` for a
> (relative) monocular estimate.

## Notes

- With a RealSense sensor, `depth_m` is metric (metres), so
  `depth_m < 0.5` means "within 50 cm".
- Events go to `arm_pick_events.jsonl` (one JSON object per line).
- Grasp planning / kinematics are out of scope (see SPEC non-goals) —
  Orc-Vison provides the perception + trigger, your controller does the
  motion.

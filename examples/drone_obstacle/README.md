# Example: drone obstacle awareness

Detect obstacles and estimate their distance with monocular depth, raising
alerts when something is close or a person is ahead.

## Run

```bash
pip install orcvision[cpu,depth]   # depth adds transformers + torch
python -m orcvision run --config examples/drone_obstacle/config.yaml
```

## Notes

- `depth.enabled: true` runs Depth Anything V2 and fills `depth_m` on each
  detection by sampling the bbox centre.
- **This is a relative monocular estimate, not metric range.** For metric
  depth use an RGB-D sensor (`sensor.type: realsense`), which populates
  `depth_m` in metres directly — no `[depth]` extra needed.
- Out of scope by design (see SPEC non-goals): navigation, motion
  planning, and IMU/lidar fusion. Orc-Vison emits perception + alerts;
  your flight stack decides what to do with them.

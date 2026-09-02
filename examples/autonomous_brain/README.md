# Autonomous brain — the decision layer above perception

This example demonstrates the intelligence layer: the part that turns
perception output into autonomous decisions, remembers what happened, and
changes its behaviour accordingly.

```
Vision output → State → Temporal tracking → Memory
     → Decision → Action → Feedback → Memory update → Next decision
```

## Run it (no hardware, no model weights, no network)

```bash
python examples/autonomous_brain/demo.py
```

The "vision output" is a hand-written list of detections — exactly the shape
any perception source hands over — so the demo runs anywhere Python does.

## What it shows

Two identical trials of an obstacle closing from 4.0 m to 1.2 m:

| Trial | Decision | Why |
|-------|----------|-----|
| 1 | `AVOID(direction=right)` | Hazard closing, no prior experience |
| 2 | `STOP` | `AVOID` is remembered to have failed here |

The perception input is byte-for-byte identical between the two trials.
**Only memory differs.** A YOLO wrapper produces the same output both times;
a brain does not. That is the whole point of the layer.

The brain also explains itself:

```
Decision: STOP

Reason:
  obstacle moving toward system (+0.80)
  obstacle at close range in center (+0.68)

Down-weighted by experience:
  AVOID previously failed in this situation (-1.50)
```

## Using it with the real pipeline

The brain consumes the perception events the existing pipeline already
emits, so nothing about your detector changes:

```bash
# Attach the brain to a live run and print its reasoning
python -m orcvision run --source video.mp4 --model yolov8n \
    --track --depth --brain --goal avoid_collision --explain
```

Each emitted event gains a `decision` field:

```json
{"timestamp": 1788240924.72, "frame_id": 42, "detections": [...], "alerts": [],
 "decision": {"action": {"type": "STOP", "parameters": {}}, "score": 1.48,
              "situation": "avoid_collision|obstacle|center|near",
              "reasons": [{"text": "obstacle moving toward system", "contribution": 0.8}],
              "vetoed": [], "safety_fallback": false}}
```

Or drive it yourself from any vision source:

```python
from orcvision.brain import VisionBrain

brain = VisionBrain(goal="avoid_collision")
brain.observe(detections)  # PerceptionEvent, dicts, or a SceneState
decision = brain.decide()
print(decision.explain())
brain.execute(decision)  # hands the Action to your executor
brain.feedback(success=False)  # closes the loop
brain.learn()
```

See `config.yaml` in this folder for the config-file form.

## Safety note

The learned policy sits **under** a deterministic constraint layer
(`orcvision/brain/constraints.py`). Constraints can veto an action outright
no matter how attractive the policy finds it — so a badly-trained or
degraded policy still cannot drive the platform forward into a close
obstacle. If every candidate is vetoed the brain falls back to `STOP`
rather than the least-bad forbidden action.

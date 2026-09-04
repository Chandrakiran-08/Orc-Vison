"""Orc-Vison autonomous brain — end-to-end demo, no hardware required.

Proves the V1 loop::

    Vision output -> State -> Temporal tracking -> Memory
        -> Decision -> Action -> Feedback -> Memory update -> Next decision

The scenario is a mobile platform with an obstacle closing in front of it.
The point of the demo is the *difference between trial 1 and trial 2*: the
perception input is byte-for-byte identical, but the brain acts differently
the second time because it remembers that its first choice failed.

Run::

    python examples/autonomous_brain/demo.py

No camera, no model weights, no broker, no network — the "vision output" is
a hand-written list of detections, exactly as any perception source would
hand it over.
"""

from __future__ import annotations

from orcvision.brain import VisionBrain

RULE = "─" * 72


def approach_sequence(base_t: float) -> list[dict]:
    """A synthetic 'obstacle closing in' clip, as a vision system would emit it."""
    return [
        {
            "timestamp": base_t + i * 0.5,
            "frame_shape": (480, 640),
            "detections": [
                {
                    "label": "obstacle",
                    "confidence": 0.92,
                    "bbox": (280, 200, 360, 300),
                    "track_id": 1,
                    "depth_m": depth,
                }
            ],
        }
        for i, depth in enumerate([4.0, 3.0, 2.0, 1.2])
    ]


def run_trial(brain: VisionBrain, trial: int, base_t: float):
    print(f"\n{RULE}\nTRIAL {trial} — an obstacle closes from 4.0 m to 1.2 m\n{RULE}")

    for frame in approach_sequence(base_t):
        brain.observe(frame)
        for event in brain.events:
            print(f"  [temporal] {event.describe()}")

    print("\n  Internal world model:")
    for line in brain.describe_world():
        print(f"    {line}")

    decision = brain.decide()
    brain.execute(decision)

    print("\n  " + brain.explain().replace("\n", "\n  "))
    return decision


def main() -> None:
    brain = VisionBrain(goal="avoid_collision")

    # --- Trial 1: no experience yet -------------------------------------
    first = run_trial(brain, 1, base_t=0.0)

    # The action did not work out — the platform clipped the obstacle.
    print(f"\n  >>> FEEDBACK: {first.action.type} failed (clipped the obstacle)")
    brain.feedback(success=False, decision=first, note="clipped the obstacle")
    brain.learn()

    # --- Trial 2: identical input, now with experience ------------------
    second = run_trial(brain, 2, base_t=100.0)

    # --- The point ------------------------------------------------------
    print(f"\n{RULE}\nRESULT\n{RULE}")
    print(f"  Trial 1 chose: {first.action}")
    print(f"  Trial 2 chose: {second.action}")
    print(
        f"\n  Identical perception input, different action: "
        f"{'YES' if first.action.type != second.action.type else 'NO'}"
    )
    print("\n  That difference is the whole point: the decision came from")
    print("  remembered experience, not from the pixels. A YOLO wrapper")
    print("  would have produced the same output both times.")

    print("\n  Working memory (most recent first):")
    for item in brain.memory.working.recent(limit=6):
        print(f"    {item.describe()}")

    print("\n  Long-term memory:")
    for trace in brain.memory.longterm.search():
        print(f"    {trace.key} -> {trace.content} (hits={trace.hits})")


if __name__ == "__main__":
    main()

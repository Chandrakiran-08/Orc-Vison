"""UAV inspection flight — the decision layer, no hardware required.

Shows what a vision brain has to get right on an aerial platform, which is
mostly *not* about seeing obstacles:

    1. mission flight with a clear path
    2. an obstacle closes in — evade, or climb over it
    3. the battery falls below the return threshold — come home
    4. the battery hits critical — land now, returning is no longer an option
    5. a geofence breach on a full battery — still come home
    6. the telemetry link drops — recover, do not continue the mission

Every one of 3-6 is a *platform* condition, not something in the camera
frame. A brain that only reasons about pixels cannot make any of these
calls, which is why UAV autonomy needs a self-model.

Run::

    python examples/uav_obstacle/demo.py

Nothing here talks to a flight controller. The brain emits actions like
RETURN_HOME and DESCEND; wiring those to MAVLink/ArduPilot/PX4 is the
integrator's job and deliberately lives outside the brain.
"""

from __future__ import annotations

from orcvision.brain import VisionBrain
from orcvision.brain.actions import UAV_ACTIONS
from orcvision.brain.brain import BrainConfig
from orcvision.brain.constraints import uav_constraints

RULE = "─" * 74


def build_uav() -> VisionBrain:
    brain = VisionBrain(
        BrainConfig(
            goal="inspect",
            actions=UAV_ACTIONS,
            safe_action="HOVER",  # a UAV cannot simply "stop" mid-air
            hazard_labels=frozenset({"obstacle", "person", "vehicle", "bird"}),
        ),
        constraints=uav_constraints(battery_return_pct=25.0, battery_land_pct=10.0),
    )
    brain.update_platform(
        battery_pct=95.0,
        altitude_m=40.0,
        max_altitude_m=120.0,  # typical regulatory ceiling
        min_altitude_m=5.0,
        distance_from_home_m=30.0,
        geofence_radius_m=250.0,
    )
    return brain


def frame(t: float, depth: float | None = None) -> dict:
    dets = []
    if depth is not None:
        dets.append(
            {
                "label": "obstacle",
                "confidence": 0.92,
                "bbox": (280, 200, 360, 300),
                "depth_m": depth,
                "track_id": 1,
            }
        )
    return {"timestamp": t, "frame_shape": (480, 640), "detections": dets}


def show(brain: VisionBrain, title: str, t: float) -> None:
    decision = brain.decide(now=t)
    platform = brain.platform
    telemetry = (
        f"batt {platform.battery_pct:>3.0f}%  alt {platform.altitude_m:>5.1f} m  "
        f"home {platform.distance_from_home_m:>5.1f} m  "
        f"link {'up' if platform.link_ok else 'DOWN'}"
    )
    print(f"\n{RULE}\n{title}\n  {telemetry}\n{RULE}")
    print("  " + decision.explain().replace("\n", "\n  "))


def main() -> None:
    brain = build_uav()

    # 1. Clear sky, healthy platform.
    brain.observe(frame(0.0))
    show(brain, "1. Mission flight, path clear", 0.0)

    # 2. An obstacle closes in.
    for i, depth in enumerate([5.0, 4.0, 3.0, 1.5]):
        brain.observe(frame(1.0 + i * 0.5, depth))
    show(brain, "2. Obstacle closing head-on", 2.5)

    # 3. Battery falls below the return threshold, mid-mission.
    brain.observe(frame(10.0))
    brain.update_platform(battery_pct=22.0, distance_from_home_m=140.0)
    show(brain, "3. Battery below return threshold", 10.0)

    # 4. Critical battery — returning is no longer viable.
    brain.observe(frame(20.0))
    brain.update_platform(battery_pct=7.0)
    show(brain, "4. Battery critical — land now", 20.0)

    # 5. Geofence breach on a healthy battery.
    brain.observe(frame(30.0))
    brain.update_platform(battery_pct=88.0, distance_from_home_m=300.0)
    show(brain, "5. Outside the 250 m geofence", 30.0)

    # 6. Telemetry link drops.
    brain.observe(frame(40.0))
    brain.update_platform(distance_from_home_m=60.0, link_ok=False)
    show(brain, "6. Command link lost", 40.0)

    print(f"\n{RULE}\nWhy this is not a YOLO wrapper\n{RULE}")
    print("  Four of those six decisions had nothing to do with the camera.")
    print("  Battery, geofence and link state decided them — which is exactly")
    print("  how UAV incidents actually happen. The brain reasons about the")
    print("  aircraft as well as the world in front of it.")


if __name__ == "__main__":
    main()

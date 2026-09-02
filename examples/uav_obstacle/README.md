# UAV inspection — decision layer for an aerial platform

```bash
python examples/uav_obstacle/demo.py
```

Six situations, no hardware:

| # | Situation | Decision |
|---|-----------|----------|
| 1 | Clear path, healthy platform | `MOVE` |
| 2 | Obstacle closing head-on | `AVOID` (with `ASCEND` ranked second) |
| 3 | Battery below the 25% return threshold | `RETURN_HOME` |
| 4 | Battery below the 10% land threshold | `DESCEND` — returning is no longer viable |
| 5 | Outside the 250 m geofence, full battery | `RETURN_HOME` |
| 6 | Telemetry link lost | recovery, not mission |

**Four of those six had nothing to do with the camera.** Battery, geofence
and link state decided them — which is how UAV incidents actually happen. A
brain that only reasons about pixels cannot make any of those calls, which
is why `PlatformState` exists: the aircraft reasons about *itself* as well as
about the world in front of it.

`ASCEND` matters too: climbing over an obstacle is an option a ground robot
does not have, and the action vocabulary has to be able to express it.

## Wiring it to a real aircraft

The brain emits `RETURN_HOME`, `DESCEND`, `HOVER` as plain
`Action(type, parameters)` objects. Mapping those to MAVLink
(`ArduPilot`/`PX4`) modes is the integrator's job and deliberately lives
outside the brain — that boundary is what keeps one policy portable across
airframes.

```python
def execute(action):
    if action.type == "RETURN_HOME":
        vehicle.mode = VehicleMode("RTL")
    elif action.type == "DESCEND":
        vehicle.mode = VehicleMode("LAND")
    elif action.type == "HOVER":
        vehicle.mode = VehicleMode("LOITER")
```

## Honest status

Simulated telemetry only. Nothing here has flown, and the thresholds
(25% return, 10% land, 5 m proximity) are sane defaults, not tuned values
for your airframe. Treat this as the decision layer to integrate and tune,
not a flight-ready autopilot. The safety constraints are a software floor —
they are not a substitute for the flight controller's own failsafes, which
should remain configured and independent.

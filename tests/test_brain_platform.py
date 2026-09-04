"""Platform self-model, safety constraints, production mode and audit trail.

These cover the properties that decide whether the brain is deployable on a
UAV or in an industrial cell, as opposed to demonstrable on a desk:

* it knows its own battery / altitude / interlocks, and acts on them
* it fails safe when perception goes stale
* it can be frozen so field behaviour does not drift
* every decision can be reconstructed after an incident
"""

import json

import pytest

from orcvision.brain import VisionBrain
from orcvision.brain.actions import (
    ASCEND,
    DESCEND,
    INDUSTRIAL_ACTIONS,
    MOVE,
    RETURN_HOME,
    STOP,
    UAV_ACTIONS,
)
from orcvision.brain.audit import AuditLog
from orcvision.brain.brain import BrainConfig
from orcvision.brain.constraints import (
    BatteryConstraint,
    GeofenceConstraint,
    HealthConstraint,
    KeepOutZoneConstraint,
    StaleDataConstraint,
    industrial_constraints,
    uav_constraints,
)
from orcvision.brain.state import PlatformState

CLEAR = {"timestamp": 0.0, "frame_shape": (480, 640), "detections": []}


def _person_at(bbox, t=0.0, depth=3.0):
    return {
        "timestamp": t,
        "frame_shape": (480, 640),
        "detections": [
            {
                "label": "person",
                "confidence": 0.9,
                "bbox": bbox,
                "depth_m": depth,
                "track_id": 1,
            }
        ],
    }


# --- PlatformState -----------------------------------------------------------


def test_unknown_telemetry_never_invents_a_limit():
    """Partial telemetry must degrade cleanly, not raise or guess."""
    p = PlatformState()
    assert p.battery_below(20) is False
    assert p.outside_geofence() is False
    assert p.altitude_out_of_band() is False
    assert p.healthy() is True


def test_platform_thresholds():
    p = PlatformState(
        battery_pct=18.0,
        altitude_m=130.0,
        max_altitude_m=120.0,
        distance_from_home_m=300.0,
        geofence_radius_m=250.0,
    )
    assert p.battery_below(25) and not p.battery_below(10)
    assert p.outside_geofence()
    assert p.altitude_out_of_band()


def test_health_reflects_interlock_link_and_emergency():
    assert PlatformState().healthy()
    assert not PlatformState(interlock_ok=False).healthy()
    assert not PlatformState(link_ok=False).healthy()
    assert not PlatformState(emergency=True).healthy()


def test_update_platform_rejects_unknown_fields():
    brain = VisionBrain()
    brain.update_platform(battery_pct=50.0)
    assert brain.platform.battery_pct == 50.0
    with pytest.raises(AttributeError):
        brain.update_platform(fuel_level=3)


# --- stale perception (the failsafe that matters most) ----------------------


def test_stale_perception_forbids_motion():
    brain = VisionBrain(
        BrainConfig(goal="cell", actions=INDUSTRIAL_ACTIONS),
        constraints=industrial_constraints(max_perception_age_s=0.5),
    )
    brain.observe({**CLEAR, "timestamp": 100.0})
    assert brain.decide(now=100.1).action.type == MOVE  # fresh
    stale = brain.decide(now=103.0)  # feed died 3 s ago
    assert stale.action.type == STOP
    assert any("stale" in reason for _, reason in stale.vetoed)


def test_stale_constraint_allows_only_declared_safe_states():
    """WAIT is not a safe state — idling in place is not the same as stopping."""
    constraint = StaleDataConstraint(max_age_s=0.5)
    assert "WAIT" not in constraint.allowed_when_stale
    assert STOP in constraint.allowed_when_stale


def test_no_staleness_before_the_first_frame():
    """A brain that has never seen a frame must not report stale data."""
    brain = VisionBrain(constraints=[StaleDataConstraint(max_age_s=0.1)])
    assert brain.decide(now=9999.0).action.type  # does not raise or veto everything


# --- platform health --------------------------------------------------------


def test_tripped_interlock_stops_the_machine():
    brain = VisionBrain(
        BrainConfig(goal="cell", actions=INDUSTRIAL_ACTIONS),
        constraints=industrial_constraints(),
    )
    brain.observe(CLEAR)
    assert brain.decide(now=0.0).action.type == MOVE
    brain.update_platform(interlock_ok=False)
    decision = brain.decide(now=0.0)
    # It *chooses* the emergency stop rather than merely being blocked from
    # moving — the platform-health feature outranks the mission actions — and
    # names the interlock as the cause.
    assert decision.action.type == "EMERGENCY_STOP"
    assert decision.action.parameters["reason"] == "interlock"
    assert any("unhealthy" in r.text for r in decision.reasons)


def test_declared_emergency_blocks_mission_actions():
    brain = VisionBrain(
        BrainConfig(goal="cell", actions=INDUSTRIAL_ACTIONS), constraints=[HealthConstraint()]
    )
    brain.observe(CLEAR)
    brain.declare_emergency()
    assert brain.decide(now=0.0).action.type != MOVE


# --- UAV: battery, geofence, altitude ---------------------------------------


def _uav():
    brain = VisionBrain(
        BrainConfig(goal="inspect", actions=UAV_ACTIONS, safe_action="HOVER"),
        constraints=uav_constraints(),
    )
    brain.observe(CLEAR)
    brain.update_platform(
        altitude_m=40.0, max_altitude_m=120.0, distance_from_home_m=20.0, geofence_radius_m=250.0
    )
    return brain


@pytest.mark.parametrize(
    "battery,expected",
    [(90.0, MOVE), (20.0, RETURN_HOME), (6.0, DESCEND)],
)
def test_battery_drives_the_recovery_ladder(battery, expected):
    """Mission -> come home -> land, as the battery falls."""
    brain = _uav()
    brain.update_platform(battery_pct=battery)
    assert brain.decide(now=0.0).action.type == expected


def test_geofence_breach_sends_it_home_even_on_a_full_battery():
    brain = _uav()
    brain.update_platform(battery_pct=95.0, distance_from_home_m=300.0)
    assert brain.decide(now=0.0).action.type == RETURN_HOME


def test_ceiling_blocks_ascend():
    brain = _uav()
    brain.update_platform(battery_pct=90.0, altitude_m=120.0, max_altitude_m=120.0)
    assert GeofenceConstraint().veto(ASCEND, brain.context(0.0)) is not None


def test_battery_constraint_is_inert_without_telemetry():
    """No battery reading must not mean 'assume empty'."""
    brain = _uav()
    brain.update_platform(battery_pct=None)
    assert BatteryConstraint().veto(MOVE, brain.context(0.0)) is None


def test_ascend_is_a_real_option_for_an_aerial_platform():
    """A UAV can climb over an obstacle; that must be ranked, not absent."""
    brain = _uav()
    brain.update_platform(battery_pct=90.0)
    for t, depth in [(1.0, 4.0), (1.5, 3.0), (2.0, 2.0), (2.5, 1.2)]:
        brain.observe(
            {
                "timestamp": t,
                "frame_shape": (480, 640),
                "detections": [
                    {
                        "label": "obstacle",
                        "confidence": 0.9,
                        "bbox": (280, 200, 360, 300),
                        "depth_m": depth,
                        "track_id": 1,
                    }
                ],
            }
        )
    decision = brain.decide(now=2.5)
    ranked = [decision.action.type] + [name for name, _ in decision.alternatives]
    assert ASCEND in ranked[:2]


# --- industrial: keep-out zones ---------------------------------------------


def test_person_in_keep_out_zone_stops_the_machine():
    zones = [(0.3, 0.2, 0.7, 1.0)]
    brain = VisionBrain(
        BrainConfig(goal="cell", actions=INDUSTRIAL_ACTIONS),
        constraints=industrial_constraints(keep_out_zones=zones),
    )
    brain.observe(CLEAR)
    assert brain.decide(now=0.0).action.type == MOVE

    brain.observe(_person_at((280, 200, 360, 400), t=0.1))  # centre of frame
    decision = brain.decide(now=0.1)
    assert decision.action.type == STOP
    assert any("keep-out zone" in r for _, r in decision.vetoed)


def test_person_outside_the_zone_does_not_stop_the_machine():
    """A zone guard that trips on everything is useless."""
    brain = VisionBrain(
        BrainConfig(goal="cell", actions=INDUSTRIAL_ACTIONS),
        constraints=[KeepOutZoneConstraint([(0.4, 0.4, 0.6, 0.6)])],
    )
    brain.observe(_person_at((0, 0, 60, 60), depth=8.0))  # far corner
    assert brain.decide(now=0.0).action.type == MOVE


def test_keep_out_zones_are_resolution_independent():
    zones = [(0.3, 0.2, 0.7, 1.0)]
    hd = VisionBrain(
        BrainConfig(goal="cell", actions=INDUSTRIAL_ACTIONS),
        constraints=[KeepOutZoneConstraint(zones)],
    )
    hd.observe(
        {
            "timestamp": 0.0,
            "frame_shape": (1080, 1920),
            "detections": [
                {"label": "person", "confidence": 0.9, "bbox": (840, 450, 1080, 900)},
            ],
        }
    )
    assert hd.decide(now=0.0).action.type == STOP


# --- production mode --------------------------------------------------------


def test_frozen_policy_does_not_drift():
    """Certifiable behaviour: feedback still recorded, weights pinned."""
    brain = VisionBrain(goal="avoid_collision")
    brain.observe(_person_at((280, 200, 360, 300), depth=1.0))
    before = dict(brain.policy.weights)

    brain.freeze()
    assert brain.frozen
    decision = brain.decide(now=0.0)
    brain.feedback(success=False, decision=decision)
    report = brain.learn()

    assert report["updated"] is False
    assert "frozen" in report["reason"]
    assert brain.policy.weights == before  # not one weight moved


def test_frozen_brain_still_records_outcomes_for_analysis():
    brain = VisionBrain(goal="avoid_collision", config=BrainConfig(frozen=True))
    brain.observe(_person_at((280, 200, 360, 300), depth=1.0))
    decision = brain.decide(now=0.0)
    brain.feedback(success=False, decision=decision)
    assert len(brain.memory.longterm) >= 1  # memory still learns; weights do not


def test_unfreeze_restores_learning():
    brain = VisionBrain(goal="avoid_collision", config=BrainConfig(frozen=True))
    brain.observe(_person_at((280, 200, 360, 300), depth=1.0))
    brain.unfreeze()
    decision = brain.decide(now=0.0)
    brain.feedback(success=False, decision=decision)
    assert brain.learn()["updated"] is True


# --- audit trail ------------------------------------------------------------


def test_audit_log_captures_enough_to_reconstruct_a_decision(tmp_path):
    path = tmp_path / "audit.jsonl"
    brain = VisionBrain(
        BrainConfig(goal="cell", actions=INDUSTRIAL_ACTIONS, audit_path=str(path)),
        constraints=industrial_constraints(keep_out_zones=[(0.3, 0.2, 0.7, 1.0)]),
    )
    brain.update_platform(battery_pct=77.0, interlock_ok=True)
    brain.observe(_person_at((280, 200, 360, 400)))
    brain.decide(now=0.0)

    records = brain.audit.read()
    assert len(records) == 1
    entry = records[0]
    assert entry["decision"]["action"]["type"] == STOP
    assert entry["goal"] == "cell"
    assert entry["platform"]["battery_pct"] == 77.0
    assert entry["objects"][0]["label"] == "person"
    assert any("keep-out" in v["reason"] for v in entry["decision"]["vetoed"])
    # Must be valid JSON Lines, one object per line.
    for line in path.read_text(encoding="utf-8").splitlines():
        json.loads(line)


def test_audit_log_rotates_and_stays_bounded(tmp_path):
    log = AuditLog(tmp_path / "a.jsonl", max_bytes=800, keep=2)
    brain = VisionBrain(goal="avoid_collision")
    brain.observe(_person_at((280, 200, 360, 300), depth=1.0))
    decision = brain.decide(now=0.0)
    for _ in range(200):
        log.record(decision, brain.world, timestamp=0.0)
    assert (tmp_path / "a.jsonl").stat().st_size < 800 * 3
    assert (tmp_path / "a.jsonl.1").exists()


def test_audit_write_failure_never_stops_the_machine(tmp_path):
    """A full disk must not take the robot down."""
    log = AuditLog(tmp_path / "a.jsonl")
    brain = VisionBrain(goal="avoid_collision")
    brain.observe(_person_at((280, 200, 360, 300), depth=1.0))
    decision = brain.decide(now=0.0)
    log.path = tmp_path / "no-such-dir" / "deeper" / "a.jsonl"  # unwritable
    assert log.record(decision, brain.world) is False
    assert log.write_errors == 1


def test_audit_tolerates_a_torn_final_line(tmp_path):
    """Power cut mid-write must cost one record, not the whole log."""
    path = tmp_path / "a.jsonl"
    path.write_text('{"t":0,"decision":{}}\n{"t":1,"dec', encoding="utf-8")
    assert len(AuditLog(path).read()) == 1


def test_battery_threshold_is_decisive_not_a_gentle_ramp():
    """Just below the return threshold must already beat cheap alternatives.

    Regression: with a purely linear deficit, battery_low at 22% of a 25%
    threshold scored 0.14 — less than HOVER's 0.20 bias — so the aircraft
    held station until the battery died instead of coming home.
    """
    brain = _uav()
    for battery in (24.0, 22.0, 15.0, 11.0):
        brain.update_platform(battery_pct=battery)
        assert brain.decide(now=0.0).action.type == RETURN_HOME, (
            f"at {battery}% the aircraft must come home, not idle"
        )
    # And above the line the mission continues.
    brain.update_platform(battery_pct=26.0)
    assert brain.decide(now=0.0).action.type == MOVE

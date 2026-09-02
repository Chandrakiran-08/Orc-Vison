"""Industrial cell guard — the decision layer, no hardware required.

A camera watches a machine cell. The brain decides whether the cell may keep
running, and every decision is written to an audit trail.

    1. cell clear — run
    2. a person enters the keep-out zone — stop (a software light curtain)
    3. the person is near but outside the zone — keep running
    4. the camera feed dies — stop, because the world model may be stale
    5. the guard door opens (interlock) — emergency stop
    6. after the incident, replay the audit log

Two properties matter more here than clever decisions:

**Fail safe, not fail silent.** Step 4 is the one most systems get wrong. If
the detector crashes or the network stalls, a brain with no staleness check
keeps acting on a frozen snapshot of a cell that has since had someone walk
into it. Perception going quiet is itself a safety event.

**Deterministic behaviour.** The policy is frozen, so the cell behaves
identically today and in six months. You cannot certify a machine whose
decision policy drifts in the field; train offline, freeze, deploy.

Run::

    python examples/industrial_safety/demo.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from orcvision.brain import VisionBrain
from orcvision.brain.actions import INDUSTRIAL_ACTIONS
from orcvision.brain.brain import BrainConfig
from orcvision.brain.constraints import industrial_constraints

RULE = "─" * 74

# Normalized (x1, y1, x2, y2) — the dangerous half of the frame, in front of
# the machine. Normalized coordinates mean the same zone works if the camera
# is swapped for a different resolution.
KEEP_OUT_ZONES = [(0.30, 0.20, 0.70, 1.00)]


def build_cell(audit_path: Path) -> VisionBrain:
    brain = VisionBrain(
        BrainConfig(
            goal="cell_guard",
            actions=INDUSTRIAL_ACTIONS,
            safe_action="STOP",
            hazard_labels=frozenset({"person"}),
            audit_path=str(audit_path),
            frozen=True,  # deterministic: no field drift
        ),
        constraints=industrial_constraints(
            keep_out_zones=KEEP_OUT_ZONES,
            max_perception_age_s=0.5,  # tighter than a UAV: a stopped line is cheap
        ),
    )
    brain.update_platform(interlock_ok=True, link_ok=True)
    return brain


def frame(t: float, person_bbox=None, depth: float = 2.5) -> dict:
    dets = []
    if person_bbox is not None:
        dets.append(
            {
                "label": "person",
                "confidence": 0.94,
                "bbox": person_bbox,
                "depth_m": depth,
                "track_id": 1,
            }
        )
    return {"timestamp": t, "frame_shape": (480, 640), "detections": dets}


def show(brain: VisionBrain, title: str, t: float) -> str:
    decision = brain.decide(now=t)
    print(f"\n{RULE}\n{title}\n{RULE}")
    print("  " + decision.explain().replace("\n", "\n  "))
    return decision.action.type


def main() -> None:
    audit_path = Path(tempfile.mkdtemp()) / "cell_audit.jsonl"
    brain = build_cell(audit_path)

    # 1. Cell clear.
    brain.observe(frame(0.0))
    show(brain, "1. Cell clear", 0.0)

    # 2. Person steps into the keep-out zone.
    brain.observe(frame(1.0, (280, 200, 360, 400)))
    show(brain, "2. Person INSIDE the keep-out zone", 1.0)

    # 3. Person visible but well outside the zone — a guard that trips on
    #    everything gets bypassed by operators within a week.
    brain.observe(frame(2.0, (10, 30, 70, 120), depth=4.5))
    show(brain, "3. Person present but OUTSIDE the zone, at 4.5 m", 2.0)

    # 4. Camera feed dies. Clear the cell first so nothing in view could
    #    explain a stop — the only reason left is that perception went quiet.
    for t in (2.5, 3.0, 3.5, 4.0, 4.5):
        brain.observe(frame(t))
    show(brain, "4. Cell empty, then camera feed lost (no frame for 3 s)", 7.5)

    # 5. Guard door opens.
    brain.observe(frame(6.0))
    brain.update_platform(interlock_ok=False)
    show(brain, "5. Guard door open — interlock tripped", 6.0)

    # 6. Incident review.
    print(f"\n{RULE}\n6. Audit trail — what did it decide, and on what evidence?\n{RULE}")
    for record in brain.audit.read():
        action = record["decision"]["action"]["type"]
        seen = ", ".join(f"{o['label']}@{o['zone']}" for o in record["objects"]) or "nothing"
        vetoes = len(record["decision"]["vetoed"])
        interlock = "ok" if record["platform"]["interlock_ok"] else "TRIPPED"
        print(
            f"  t={record['t']:>4.1f}  {action:<15} saw: {seen:<20} "
            f"vetoes: {vetoes}  interlock: {interlock}"
        )
    print(f"\n  Written to {audit_path}")
    print("  One JSON object per line — replayable, greppable, and complete")
    print("  enough to reconstruct each decision without the running process.")

    print(f"\n{RULE}\nWhy this is deployable rather than a demo\n{RULE}")
    print("  * Step 4 stopped the cell because perception went QUIET, not")
    print("    because it saw something. Fail-safe, not fail-silent.")
    print("  * The policy is frozen, so behaviour is reproducible and")
    print("    certifiable — no drift between commissioning and next year.")
    print("  * Step 3 kept running: a guard that trips on everything gets")
    print("    disabled by operators, which is worse than no guard at all.")


if __name__ == "__main__":
    main()

"""Generate golden decision vectors from the Python brain.

The C++ port must reproduce these exactly. Run from the repo root::

    python firmware/OrcVisionBrain/tests/generate_golden.py

Writes ``golden.h`` next to the parity test. Regenerate whenever the Python
decision logic changes — a diff in that file is the signal that the firmware
needs to change too.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from orcvision.brain import VisionBrain  # noqa: E402

FRAME_W, FRAME_H = 640, 480

# Each scenario is a list of steps. A step is either a frame of detections or
# a feedback event, so a scenario can exercise the whole loop.
SCENARIOS: list[tuple[str, list]] = [
    (
        "closing_obstacle_center",
        [
            ("frame", 0.0, [("obstacle", 0.9, (280, 200, 360, 300), 4.0, 1)]),
            ("frame", 0.5, [("obstacle", 0.9, (280, 200, 360, 300), 3.0, 1)]),
            ("frame", 1.0, [("obstacle", 0.9, (280, 200, 360, 300), 2.0, 1)]),
            ("frame", 1.5, [("obstacle", 0.9, (280, 200, 360, 300), 1.2, 1)]),
            ("decide",),
        ],
    ),
    (
        "memory_flips_choice",
        [
            ("frame", 0.0, [("obstacle", 0.9, (280, 200, 360, 300), 4.0, 1)]),
            ("frame", 0.5, [("obstacle", 0.9, (280, 200, 360, 300), 3.0, 1)]),
            ("frame", 1.0, [("obstacle", 0.9, (280, 200, 360, 300), 2.0, 1)]),
            ("frame", 1.5, [("obstacle", 0.9, (280, 200, 360, 300), 1.2, 1)]),
            ("decide",),
            ("feedback", False),
            ("frame", 100.0, [("obstacle", 0.9, (280, 200, 360, 300), 4.0, 1)]),
            ("frame", 100.5, [("obstacle", 0.9, (280, 200, 360, 300), 3.0, 1)]),
            ("frame", 101.0, [("obstacle", 0.9, (280, 200, 360, 300), 2.0, 1)]),
            ("frame", 101.5, [("obstacle", 0.9, (280, 200, 360, 300), 1.2, 1)]),
            ("decide",),
        ],
    ),
    (
        "clear_scene_moves",
        [
            ("frame", 0.0, []),
            ("decide",),
        ],
    ),
    (
        "hazard_on_the_left",
        [
            ("frame", 0.0, [("obstacle", 0.9, (0, 200, 80, 300), 2.0, 1)]),
            ("frame", 0.5, [("obstacle", 0.9, (0, 200, 80, 300), 1.0, 1)]),
            ("decide",),
        ],
    ),
    (
        "distant_hazard_allows_move",
        [
            ("frame", 0.0, [("obstacle", 0.9, (280, 200, 300, 220), 4.9, 1)]),
            ("frame", 0.5, [("obstacle", 0.9, (280, 200, 300, 220), 4.9, 1)]),
            ("decide",),
        ],
    ),
    (
        "person_hazard_no_depth",
        [
            ("frame", 0.0, [("person", 0.9, (200, 100, 440, 400), None, 1)]),
            ("frame", 0.5, [("person", 0.9, (180, 80, 460, 420), None, 1)]),
            ("decide",),
        ],
    ),
    (
        "repeated_failures_degrade_safely",
        [
            ("frame", 0.0, [("obstacle", 0.9, (280, 200, 360, 300), 1.0, 1)]),
            ("decide",),
            ("feedback", False),
            ("frame", 10.0, [("obstacle", 0.9, (280, 200, 360, 300), 1.0, 1)]),
            ("decide",),
            ("feedback", False),
            ("frame", 20.0, [("obstacle", 0.9, (280, 200, 360, 300), 1.0, 1)]),
            ("decide",),
            ("feedback", False),
            ("frame", 30.0, [("obstacle", 0.9, (280, 200, 360, 300), 1.0, 1)]),
            ("decide",),
        ],
    ),
    (
        "success_keeps_choice",
        [
            ("frame", 0.0, [("obstacle", 0.9, (280, 200, 360, 300), 1.5, 1)]),
            ("decide",),
            ("feedback", True),
            ("frame", 10.0, [("obstacle", 0.9, (280, 200, 360, 300), 1.5, 1)]),
            ("decide",),
        ],
    ),
]


def run_scenario(steps: list) -> list[dict]:
    brain = VisionBrain(goal="avoid_collision")
    decisions = []
    for step in steps:
        if step[0] == "frame":
            _, ts, dets = step
            brain.observe(
                {
                    "timestamp": ts,
                    "frame_shape": (FRAME_H, FRAME_W),
                    "detections": [
                        {
                            "label": label,
                            "confidence": conf,
                            "bbox": bbox,
                            "depth_m": depth,
                            "track_id": tid,
                        }
                        for label, conf, bbox, depth, tid in dets
                    ],
                }
            )
        elif step[0] == "decide":
            d = brain.decide()
            decisions.append({"action": d.action.type, "score": d.score})
        elif step[0] == "feedback":
            brain.feedback(success=step[1])
            brain.learn()
    return decisions


def c_literal(value) -> str:
    if value is None:
        return "OV_UNKNOWN_DEPTH"
    return f"{value:.6f}f"


def main() -> None:
    out = Path(__file__).with_name("golden.h")
    lines = [
        "// GENERATED FILE — do not edit by hand.",
        "// Regenerate with: python firmware/OrcVisionBrain/tests/generate_golden.py",
        "//",
        "// Golden decision vectors captured from the Python reference brain",
        "// (orcvision.brain). The C++ port must reproduce every action below.",
        "#ifndef ORCVISION_GOLDEN_H",
        "#define ORCVISION_GOLDEN_H",
        "",
        f"#define GOLDEN_FRAME_W {FRAME_W}",
        f"#define GOLDEN_FRAME_H {FRAME_H}",
        "",
        "struct GoldenDet {",
        "  const char* label; float conf; float x1, y1, x2, y2; float depth; int16_t track;",
        "};",
        "struct GoldenStep {",
        "  int kind;  // 0 = frame, 1 = decide, 2 = feedback",
        "  float timestamp;",
        "  int det_count;",
        "  const GoldenDet* dets;",
        "  int success;              // for feedback steps",
        "  const char* expect_action; // for decide steps",
        "};",
        "struct GoldenScenario {",
        "  const char* name; int step_count; const GoldenStep* steps;",
        "};",
        "",
    ]

    scenario_entries = []
    for name, steps in SCENARIOS:
        decisions = run_scenario(steps)
        decision_iter = iter(decisions)
        det_arrays = []
        step_entries = []
        for idx, step in enumerate(steps):
            if step[0] == "frame":
                _, ts, dets = step
                arr_name = f"{name}_dets_{idx}"
                if dets:
                    items = ", ".join(
                        '{{"{}", {:.6f}f, {:.1f}f, {:.1f}f, {:.1f}f, {:.1f}f, {}, {}}}'.format(
                            label, conf, *bbox, c_literal(depth), tid
                        )
                        for label, conf, bbox, depth, tid in dets
                    )
                    det_arrays.append(
                        f"static const GoldenDet {arr_name}[] = {{{items}}};"
                    )
                    step_entries.append(
                        f"  {{0, {ts:.6f}f, {len(dets)}, {arr_name}, 0, nullptr}},"
                    )
                else:
                    step_entries.append(f"  {{0, {ts:.6f}f, 0, nullptr, 0, nullptr}},")
            elif step[0] == "decide":
                expected = next(decision_iter)
                step_entries.append(
                    f'  {{1, 0.0f, 0, nullptr, 0, "{expected["action"]}"}},'
                    f"  // score {expected['score']:+.4f}"
                )
            elif step[0] == "feedback":
                step_entries.append(
                    f"  {{2, 0.0f, 0, nullptr, {1 if step[1] else 0}, nullptr}},"
                )

        lines.extend(det_arrays)
        lines.append(f"static const GoldenStep {name}_steps[] = {{")
        lines.extend(step_entries)
        lines.append("};")
        lines.append("")
        scenario_entries.append(
            f'  {{"{name}", (int)(sizeof({name}_steps) / sizeof(GoldenStep)), {name}_steps}},'
        )

    lines.append("static const GoldenScenario GOLDEN_SCENARIOS[] = {")
    lines.extend(scenario_entries)
    lines.append("};")
    lines.append(
        "static const int GOLDEN_SCENARIO_COUNT = "
        "(int)(sizeof(GOLDEN_SCENARIOS) / sizeof(GoldenScenario));"
    )
    lines.append("")
    lines.append("#endif  // ORCVISION_GOLDEN_H")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    total = sum(len(run_scenario(s)) for _, s in SCENARIOS)
    print(f"wrote {out} — {len(SCENARIOS)} scenarios, {total} decisions")


if __name__ == "__main__":
    main()

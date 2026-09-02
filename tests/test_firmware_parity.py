"""Keep the three brain implementations in lockstep.

There are now three ports of the same decision logic:

* ``orcvision/brain/``            — CPython reference (host / Linux SBC)
* ``firmware/OrcVisionBrain/``    — C++ for MCUs (Uno R4, ESP32, STM32, ...)
* ``firmware/micropython/``       — MicroPython (ESP32, Pico W)

Divergence between them is a silent, nasty class of bug: the robot would
behave differently depending on where the brain runs. These tests pin all
three to the same golden scenarios.

The C++ test compiles and runs on the host and is skipped where no compiler
is available. It verifies *logic*, not on-device behaviour — flashing real
hardware remains a separate, unperformed step.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FIRMWARE = REPO / "firmware"
CPP_DIR = FIRMWARE / "OrcVisionBrain"
GOLDEN_GEN = CPP_DIR / "tests" / "generate_golden.py"

sys.path.insert(0, str(FIRMWARE / "micropython"))
sys.path.insert(0, str(CPP_DIR / "tests"))


def _reference_decisions(steps):
    """Run a scenario through the CPython reference brain."""
    from orcvision.brain import VisionBrain

    brain = VisionBrain(goal="avoid_collision")
    out = []
    for step in steps:
        if step[0] == "frame":
            _, ts, dets = step
            brain.observe(
                {
                    "timestamp": ts,
                    "frame_shape": (480, 640),
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
            out.append(brain.decide().action.type)
        elif step[0] == "feedback":
            brain.feedback(success=step[1])
            brain.learn()
    return out


def _micropython_decisions(steps):
    """Run the same scenario through the MicroPython port (under CPython)."""
    import orcvision_brain as mp

    brain = mp.VisionBrain(goal="avoid_collision")
    out = []
    for step in steps:
        if step[0] == "frame":
            _, ts, dets = step
            brain.begin_frame(ts)
            for label, conf, bbox, depth, tid in dets:
                brain.observe_pixels(
                    label, conf, bbox[0], bbox[1], bbox[2], bbox[3], 640, 480, depth, tid
                )
            brain.end_frame()
        elif step[0] == "decide":
            out.append(brain.decide().action)
        elif step[0] == "feedback":
            brain.feedback(step[1])
            brain.learn()
    return out


def _scenarios():
    import generate_golden

    return generate_golden.SCENARIOS


# --- MicroPython port -------------------------------------------------------


@pytest.mark.parametrize("name", [n for n, _ in _scenarios()])
def test_micropython_port_matches_reference(name):
    scenarios = dict(_scenarios())
    steps = scenarios[name]
    assert _micropython_decisions(steps) == _reference_decisions(steps)


def test_micropython_port_avoids_cpython_only_imports():
    """The port is useless on-device if it needs modules MicroPython lacks.

    Checks the actual import statements via AST — a docstring that merely
    *mentions* ``dataclasses`` is not a dependency on it.
    """
    import ast

    source = (FIRMWARE / "micropython" / "orcvision_brain.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                imported.add(node.module.split(".")[0])
            if node.module == "__future__":
                imported.add("__future__")

    unavailable = {"dataclasses", "typing", "pathlib", "__future__", "enum", "abc"}
    assert not (imported & unavailable), (
        f"MicroPython does not provide: {sorted(imported & unavailable)}"
    )


def test_micropython_port_reproduces_the_memory_flip():
    """The headline behaviour must survive the port to constrained hardware."""
    import orcvision_brain as mp

    brain = mp.VisionBrain(goal="avoid_collision")

    def approach(base):
        for i, depth in enumerate([4.0, 3.0, 2.0, 1.2]):
            brain.begin_frame(base + i * 0.5)
            brain.observe_pixels(
                "obstacle", 0.9, 280, 200, 360, 300, 640, 480, depth, 1
            )
            brain.end_frame()

    approach(0.0)
    first = brain.decide()
    assert first.action == mp.AVOID

    brain.feedback(False)
    brain.learn()

    approach(100.0)
    second = brain.decide()
    assert second.action == mp.STOP
    assert any(action == mp.AVOID for action, _ in second.demoted)


def test_micropython_safety_floor_holds():
    import orcvision_brain as mp

    brain = mp.VisionBrain(goal="avoid_collision")
    # Train it to love MOVE.
    for _ in range(50):
        brain.weights["MOVE|bias"] = brain.weights.get("MOVE|bias", 0.0) + 1.0
    brain.begin_frame(0.0)
    brain.observe_pixels("obstacle", 0.9, 280, 200, 360, 300, 640, 480, 0.4, 1)
    brain.end_frame()
    decision = brain.decide()
    assert decision.action != mp.MOVE
    assert any(a == mp.MOVE for a, _ in decision.vetoed)


# --- C++ port ---------------------------------------------------------------


@pytest.mark.skipif(shutil.which("g++") is None, reason="no C++ compiler available")
def test_cpp_port_matches_reference(tmp_path):
    """Compile the MCU library on the host and check every golden decision."""
    # Regenerate goldens first so a drifted Python brain fails loudly here
    # rather than silently shipping a stale firmware expectation.
    subprocess.run([sys.executable, str(GOLDEN_GEN)], check=True, capture_output=True)

    binary = tmp_path / "parity"
    compile_result = subprocess.run(
        [
            "g++", "-std=c++11", "-O2", "-Wall", "-Wextra", "-Werror",
            "-I", str(CPP_DIR / "src"),
            "-I", str(CPP_DIR / "tests"),
            str(CPP_DIR / "tests" / "parity_test.cpp"),
            "-o", str(binary),
        ],
        capture_output=True,
        text=True,
    )
    assert compile_result.returncode == 0, f"compile failed:\n{compile_result.stderr}"

    run = subprocess.run([str(binary)], capture_output=True, text=True)
    assert run.returncode == 0, f"parity failed:\n{run.stdout}\n{run.stderr}"
    assert "PASS" in run.stdout


@pytest.mark.skipif(shutil.which("g++") is None, reason="no C++ compiler available")
def test_cpp_brain_fits_in_uno_r4_sram(tmp_path):
    """The Uno R4 WiFi has 32 KB of SRAM; the brain must be a small part of it."""
    probe = tmp_path / "size.cpp"
    probe.write_text(
        "#define OV_MAX_OBJECTS 6\n"
        "#define OV_MAX_LABELS 8\n"
        "#define OV_MAX_TRACES 10\n"
        '#include "OrcVisionBrain.h"\n'
        "#include <stdio.h>\n"
        "int main(){ printf(\"%u\\n\", (unsigned)sizeof(OrcVisionBrain)); return 0; }\n",
        encoding="utf-8",
    )
    binary = tmp_path / "size"
    subprocess.run(
        ["g++", "-std=c++11", "-O2", "-I", str(CPP_DIR / "src"), str(probe), "-o", str(binary)],
        check=True,
        capture_output=True,
    )
    size = int(subprocess.run([str(binary)], capture_output=True, text=True).stdout.strip())
    # Comfortably inside 32 KB, leaving room for WiFi/MQTT/JSON buffers.
    assert size < 4096, f"brain grew to {size} bytes — too large for a 32 KB board"


@pytest.mark.skipif(shutil.which("g++") is None, reason="no C++ compiler available")
def test_cpp_library_uses_no_dynamic_allocation():
    """No heap on an MCU: fragmentation there is a field failure, not a warning."""
    source = (CPP_DIR / "src" / "OrcVisionBrain.h").read_text(encoding="utf-8")
    code = "\n".join(
        line for line in source.splitlines()
        if not line.strip().startswith(("*", "//", "/*"))
    )
    for banned in ("malloc(", "calloc(", "realloc(", "new ", "std::", "String "):
        assert banned not in code, f"{banned} must not appear in MCU firmware"

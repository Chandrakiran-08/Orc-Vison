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
            brain.observe_pixels("obstacle", 0.9, 280, 200, 360, 300, 640, 480, depth, 1)
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
            "g++",
            "-std=c++11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(CPP_DIR / "src"),
            "-I",
            str(CPP_DIR / "tests"),
            str(CPP_DIR / "tests" / "parity_test.cpp"),
            "-o",
            str(binary),
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
        'int main(){ printf("%u\\n", (unsigned)sizeof(OrcVisionBrain)); return 0; }\n',
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
        line for line in source.splitlines() if not line.strip().startswith(("*", "//", "/*"))
    )
    for banned in ("malloc(", "calloc(", "realloc(", "new ", "std::", "String "):
        assert banned not in code, f"{banned} must not appear in MCU firmware"


# --- security / robustness regressions --------------------------------------


@pytest.mark.skipif(shutil.which("g++") is None, reason="no C++ compiler available")
def test_cpp_does_not_collapse_two_same_label_objects(tmp_path):
    """Two nearby same-label detections must stay two objects.

    Regression: association could match a slot already claimed by an earlier
    detection in the same frame, silently merging two people into one. On a
    safety system that under-counts what is in front of the machine.
    """
    probe = tmp_path / "collapse.cpp"
    probe.write_text(
        '#include "OrcVisionBrain.h"\n'
        "#include <stdio.h>\n"
        "int main(){\n"
        '  OrcVisionBrain b; b.begin(); b.addHazardLabel("person");\n'
        "  b.beginFrame(0.0f);\n"
        '  b.observe("person",0.9f,0.30f,0.5f,0.02f,3.0f,-1);\n'
        '  b.observe("person",0.9f,0.34f,0.5f,0.02f,3.0f,-1);\n'
        "  b.endFrame();\n"
        '  printf("%u\\n", b.visibleCount());\n'
        "  return 0; }\n",
        encoding="utf-8",
    )
    binary = tmp_path / "collapse"
    subprocess.run(
        ["g++", "-std=c++11", "-O2", "-I", str(CPP_DIR / "src"), str(probe), "-o", str(binary)],
        check=True,
        capture_output=True,
    )
    out = subprocess.run([str(binary)], capture_output=True, text=True).stdout.strip()
    assert out == "2", f"two people collapsed into {out} object(s)"


@pytest.mark.skipif(shutil.which("g++") is None, reason="no C++ compiler available")
def test_cpp_explain_survives_non_finite_scores(tmp_path):
    """Casting a NaN/huge float to int is UB; explain() must not do that.

    Runaway reward updates can genuinely drive a weight to inf, and the
    formatter runs on a machine with no fault handler to catch it.
    """
    probe = tmp_path / "nonfinite.cpp"
    probe.write_text(
        '#include "OrcVisionBrain.h"\n'
        "#include <stdio.h>\n"
        "#include <math.h>\n"
        "int main(){\n"
        '  OrcVisionBrain b; b.begin(); b.addHazardLabel("obstacle");\n'
        '  b.beginFrame(0.0f); b.observe("obstacle",0.9f,0.5f,0.5f,0.05f,1.0f,1); b.endFrame();\n'
        "  char buf[256];\n"
        "  const float vals[3] = {1e30f, NAN, -INFINITY};\n"
        "  for (int i=0;i<3;i++){\n"
        "    b.setWeight(OV_STOP, OV_F_BIAS, vals[i]);\n"
        "    OvDecision d = b.decide();\n"
        "    b.explain(d, buf, sizeof(buf));\n"
        # explain() legitimately emits newlines, so allow them.
        "    for (size_t j=0; buf[j]; ++j) {\n"
        "      char c = buf[j];\n"
        "      if (c != '\\n' && (c < 32 || c > 126)) { printf(\"BAD\\n\"); return 1; }\n"
        "    }\n"
        "  }\n"
        '  printf("OK\\n"); return 0; }\n',
        encoding="utf-8",
    )
    binary = tmp_path / "nonfinite"
    subprocess.run(
        ["g++", "-std=c++11", "-O2", "-I", str(CPP_DIR / "src"), str(probe), "-o", str(binary)],
        check=True,
        capture_output=True,
    )
    result = subprocess.run([str(binary)], capture_output=True, text=True)
    assert result.stdout.strip() == "OK", "explain() emitted non-printable bytes"


@pytest.mark.skipif(shutil.which("g++") is None, reason="no C++ compiler available")
def test_cpp_rejects_label_table_larger_than_the_bitmask(tmp_path):
    """A >16 label table would silently stop marking hazards. Fail the build."""
    probe = tmp_path / "toomany.cpp"
    probe.write_text(
        '#define OV_MAX_LABELS 20\n#include "OrcVisionBrain.h"\nint main(){ return 0; }\n',
        encoding="utf-8",
    )
    result = subprocess.run(
        ["g++", "-std=c++11", "-I", str(CPP_DIR / "src"), str(probe), "-o", str(tmp_path / "x")],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, "OV_MAX_LABELS > 16 must not compile"
    assert "OV_MAX_LABELS" in result.stderr


def test_corrupt_policy_file_does_not_crash_the_brain(tmp_path):
    """Persisted state is untrusted input: corrupt weights must not propagate."""
    import json

    from orcvision.brain.policy import LinearPolicy

    (tmp_path / "policy.json").write_text(
        json.dumps(
            {
                "learning_rate": "fast",
                "weights": {
                    "STOP|bias": "not-a-number",
                    "MOVE|bias": None,
                    "AVOID|bias": 0.5,
                    "WAIT|bias": float("inf"),
                },
            }
        ),
        encoding="utf-8",
    )
    policy = LinearPolicy.load(tmp_path / "policy.json")
    assert policy.weights == {"AVOID|bias": 0.5}  # only the valid entry survives
    assert policy.learning_rate == 0.1
    score, _ = policy.score("AVOID", {"bias": 1.0})
    assert score == 0.5


def test_corrupt_memory_file_does_not_crash_the_brain(tmp_path):
    """A truncated or hand-edited memory file must be survivable, not fatal."""
    import json

    from orcvision.brain import VisionBrain

    (tmp_path / "policy.json").write_text(json.dumps({"weights": {}}), encoding="utf-8")
    (tmp_path / "memory.json").write_text(
        json.dumps(
            {
                "good": {
                    "kind": "outcome",
                    "content": {"successes": 1},
                    "importance": 0.5,
                    "hits": 1,
                },
                "bad_str": "not-a-dict",
                "bad_num": 42,
            }
        ),
        encoding="utf-8",
    )
    brain = VisionBrain(goal="avoid_collision")
    brain.load(tmp_path)  # must not raise
    brain.observe([{"label": "obstacle", "confidence": 0.9, "bbox": (0.4, 0.4, 0.6, 0.6)}])
    assert brain.decide().action.type  # still operable
    assert len(brain.memory.longterm) == 1  # malformed entries dropped


@pytest.mark.skipif(shutil.which("g++") is None, reason="no C++ compiler available")
def test_epoch_timestamps_corrupt_motion_rates(tmp_path):
    """Document, executably, why the sketch uses a local time base.

    float32 spacing near a Unix epoch (~1.79e9) is about 128 s, so passing
    epoch timestamps collapses dt and corrupts every derived rate. Motion
    classification survives by luck, which is what makes this dangerous: it
    degrades quietly. This pins the failure mode so nobody "simplifies" the
    sketch back into using the host's timestamp.
    """
    probe = tmp_path / "epoch.cpp"
    probe.write_text(
        '#include "OrcVisionBrain.h"\n'
        "#include <stdio.h>\n"
        "static float rate(float t0, float step){\n"
        '  OrcVisionBrain b; b.begin(); b.addHazardLabel("obstacle");\n'
        "  float depths[4] = {4.0f,3.0f,2.0f,1.2f};\n"
        "  for (int i=0;i<4;i++){\n"
        "    b.beginFrame(t0 + (float)i*step);\n"
        '    b.observe("obstacle",0.9f,0.5f,0.5f,0.05f,depths[i],1);\n'
        "    b.endFrame();\n"
        "  }\n"
        "  const OvObject* o = b.objectAt(0);\n"
        "  return o ? o->approach_rate : 0.0f;\n"
        "}\n"
        'int main(){ printf("%.6g %.6g\\n", rate(12.0f,0.5f), rate(1788240924.0f,0.5f));\n'
        "  return 0; }\n",
        encoding="utf-8",
    )
    binary = tmp_path / "epoch"
    subprocess.run(
        ["g++", "-std=c++11", "-O2", "-I", str(CPP_DIR / "src"), str(probe), "-o", str(binary)],
        check=True,
        capture_output=True,
    )
    out = subprocess.run([str(binary)], capture_output=True, text=True).stdout.split()
    local_rate, epoch_rate = float(out[0]), float(out[1])

    # A sane time base recovers the true closing speed (0.8 m per 0.5 s).
    assert 1.0 < local_rate < 3.0, f"local time base gave {local_rate} m/s"
    # An epoch time base produces physically absurd kinematics.
    assert epoch_rate > 1000.0, (
        f"epoch time base gave {epoch_rate} m/s — if float32 behaviour changed, "
        "revisit the time-base comments in the sketch and header"
    )


@pytest.mark.skipif(shutil.which("g++") is None, reason="no C++ compiler available")
def test_on_device_self_test_sketch_passes(tmp_path):
    """Compile and run the board self-test sketch against a stubbed Arduino API.

    BrainSelfTest.ino is what a user flashes first: it exercises the whole
    loop on-device with no networking, so a failure there points at the
    brain or the toolchain rather than at WiFi. Running it here means the
    sketch cannot rot between releases — the only thing this cannot cover
    is the Arduino cross-compiler itself.
    """
    stub_dir = CPP_DIR / "tests" / "arduino_stub"
    sketch_dir = CPP_DIR / "examples" / "BrainSelfTest"

    main = tmp_path / "main.cpp"
    main.write_text(
        '#include "Arduino.h"\n'
        f'#include "{sketch_dir / "BrainSelfTest.ino"}"\n'
        "int main(){ setup(); return (checksPassed == checksRun) ? 0 : 1; }\n",
        encoding="utf-8",
    )
    binary = tmp_path / "selftest"
    compiled = subprocess.run(
        [
            "g++",
            "-std=c++11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-x",
            "c++",
            str(main),
            "-I",
            str(stub_dir),
            "-I",
            str(CPP_DIR / "src"),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
    )
    assert compiled.returncode == 0, f"sketch does not compile:\n{compiled.stderr}"

    run = subprocess.run([str(binary)], capture_output=True, text=True)
    assert "SELF TEST: PASS (5/5)" in run.stdout, run.stdout
    assert run.returncode == 0

// Parity test: the C++ brain must decide exactly what the Python brain decides.
//
// Build and run on the host (no board required):
//     g++ -std=c++11 -O2 -Wall -Wextra -I../src parity_test.cpp -o parity && ./parity
//
// This verifies the *decision logic*, which is the part that must not drift
// between the two implementations. It does not and cannot verify on-device
// behaviour — flashing and running on real hardware is a separate step.

#include <stdio.h>
#include <string.h>

#include "OrcVisionBrain.h"
#include "golden.h"

static int failures = 0;
static int checks = 0;

static void check(bool ok, const char* scenario, const char* what, const char* got,
                  const char* want) {
  ++checks;
  if (ok) return;
  ++failures;
  printf("  FAIL [%s] %s: got %s, expected %s\n", scenario, what, got, want);
}

// Configure a brain exactly as VisionBrain(goal="avoid_collision") does.
static void configure(OrcVisionBrain& brain) {
  brain.begin();
  // Interning order must cover the Python default hazard set; only labels
  // actually seen matter for behaviour, but interning up front keeps the
  // label ids stable and the hazard mask correct.
  brain.addHazardLabel("person");
  brain.addHazardLabel("obstacle");
  brain.addHazardLabel("vehicle");
  brain.addHazardLabel("car");
  brain.setGoal(1);  // any stable id; Python uses the goal string
}

int main() {
  printf("OrcVisionBrain parity test (C++ vs Python reference)\n");
  printf("sizeof(OrcVisionBrain) = %u bytes\n", (unsigned)sizeof(OrcVisionBrain));
  printf("sizeof(OvDecision)     = %u bytes\n", (unsigned)sizeof(OvDecision));
  printf("---------------------------------------------------------------\n");

  for (int s = 0; s < GOLDEN_SCENARIO_COUNT; ++s) {
    const GoldenScenario& sc = GOLDEN_SCENARIOS[s];
    OrcVisionBrain brain;
    configure(brain);

    int decisions = 0;
    for (int i = 0; i < sc.step_count; ++i) {
      const GoldenStep& st = sc.steps[i];
      if (st.kind == 0) {  // frame
        brain.beginFrame(st.timestamp);
        for (int d = 0; d < st.det_count; ++d) {
          const GoldenDet& det = st.dets[d];
          brain.observePixels(det.label, det.conf, det.x1, det.y1, det.x2, det.y2,
                              GOLDEN_FRAME_W, GOLDEN_FRAME_H, det.depth, det.track);
        }
        brain.endFrame();
      } else if (st.kind == 1) {  // decide
        OvDecision dec = brain.decide();
        const char* got = OrcVisionBrain::actionName(dec.action);
        check(strcmp(got, st.expect_action) == 0, sc.name, "action", got, st.expect_action);
        ++decisions;
      } else {  // feedback
        brain.feedback(st.success != 0);
        brain.learn();
      }
    }
    printf("  %-34s %d decision(s) checked\n", sc.name, decisions);
  }

  printf("---------------------------------------------------------------\n");
  if (failures == 0) {
    printf("PASS: %d/%d decisions match the Python reference exactly\n", checks, checks);
    return 0;
  }
  printf("FAIL: %d of %d checks diverged\n", failures, checks);
  return 1;
}

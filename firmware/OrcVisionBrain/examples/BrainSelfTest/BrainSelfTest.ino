/*
 * OrcVisionBrain — on-device self test
 * ---------------------------------------------------------------------------
 * Flash this FIRST, before the WiFi/MQTT sketch.
 *
 * It runs the brain entirely from synthetic detections held in flash: no
 * network, no broker, no JSON, no camera. So if it fails, the problem is the
 * brain or the toolchain — and if it passes, every later failure is
 * networking, which is a much smaller haystack.
 *
 * What it checks, on the board:
 *   1. the library compiles and links for your target
 *   2. temporal reasoning detects a closing obstacle
 *   3. the brain decides AVOID on a fresh situation
 *   4. after feedback that AVOID failed, the SAME input yields STOP
 *   5. the safety floor still forbids advancing into a close hazard
 *   6. it reports its own RAM footprint
 *
 * Step 4 is the real test: identical perception, different action, because
 * the board remembered. That is the whole point of the library.
 *
 * Libraries required: none.
 * Board: any — Uno R4 WiFi, ESP32, RP2040, STM32...
 *
 * Expected output ends with:  SELF TEST: PASS (5/5)
 */

#define OV_MAX_OBJECTS 6
#define OV_MAX_LABELS 8
#define OV_MAX_TRACES 10

#include <OrcVisionBrain.h>

OrcVisionBrain brain;

int checksRun = 0;
int checksPassed = 0;

void check(bool ok, const char* what) {
  ++checksRun;
  if (ok) ++checksPassed;
  Serial.print(ok ? F("  [PASS] ") : F("  [FAIL] "));
  Serial.println(what);
}

// Drive one "obstacle closing from 4.0 m to 1.2 m" sequence through the
// brain. `base` offsets the timestamps so each run is a separate encounter.
//
// Note the time base: small seconds, not a Unix epoch. float32 cannot
// resolve half-second gaps up at 1.79e9, which would silently flatten every
// motion estimate.
void approach(float base) {
  const float depths[4] = {4.0f, 3.0f, 2.0f, 1.2f};
  for (int i = 0; i < 4; ++i) {
    brain.beginFrame(base + (float)i * 0.5f);
    brain.observe("obstacle", 0.92f, 0.5f, 0.5f, 0.05f, depths[i], 1);
    brain.endFrame();
  }
}

void printDecision(const OvDecision& d) {
  char buf[256];
  brain.explain(d, buf, sizeof(buf));
  Serial.println(buf);
}

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 4000) { /* wait briefly for USB serial */ }

  Serial.println();
  Serial.println(F("======================================================"));
  Serial.println(F(" OrcVisionBrain — on-device self test"));
  Serial.println(F("======================================================"));
  Serial.print(F("Brain footprint: "));
  Serial.print((unsigned)sizeof(OrcVisionBrain));
  Serial.println(F(" bytes of SRAM"));

  brain.begin();
  brain.addHazardLabel("obstacle");
  brain.addHazardLabel("person");
  brain.setGoal(1);  // avoid_collision

  // --- 1 & 2: perception -> state -> temporal reasoning ------------------
  Serial.println(F("\n-- Encounter 1: obstacle closes head-on --"));
  approach(0.0f);
  const OvObject* obj = brain.objectAt(0);
  check(obj != nullptr, "object tracked across frames");
  check(obj && obj->motion == OV_MOTION_APPROACHING, "motion detected as APPROACHING");

  // --- 3: first decision, no experience ----------------------------------
  OvDecision first = brain.decide();
  printDecision(first);
  check(first.action == OV_AVOID, "fresh situation -> AVOID");
  brain.markExecuted(first);

  // --- 4: the headline behaviour -----------------------------------------
  Serial.println(F("\n>>> FEEDBACK: that AVOID failed (clipped the obstacle)"));
  brain.feedback(false);
  brain.learn();

  Serial.println(F("\n-- Encounter 2: IDENTICAL input, now with experience --"));
  approach(100.0f);
  OvDecision second = brain.decide();
  printDecision(second);
  check(second.action == OV_STOP, "remembered failure -> STOP instead of AVOID");

  // --- 5: the safety floor holds -----------------------------------------
  Serial.println(F("\n-- Safety floor: may it drive into a close obstacle? --"));
  brain.beginFrame(200.0f);
  brain.observe("obstacle", 0.95f, 0.5f, 0.5f, 0.09f, 0.4f, 2);
  brain.endFrame();
  OvDecision close = brain.decide();
  printDecision(close);
  check(close.action != OV_MOVE, "MOVE refused with a hazard at 0.4 m");

  // --- verdict ------------------------------------------------------------
  Serial.println(F("\n======================================================"));
  Serial.print(checksPassed == checksRun ? F(" SELF TEST: PASS (") : F(" SELF TEST: FAIL ("));
  Serial.print(checksPassed);
  Serial.print(F("/"));
  Serial.print(checksRun);
  Serial.println(F(")"));
  Serial.println(F("======================================================"));

  if (checksPassed == checksRun) {
    Serial.println(F("The brain runs correctly on this board."));
    Serial.println(F("Next: flash UnoR4WiFiBrain for the networked version."));
  } else {
    Serial.println(F("Something differs from the host reference — please"));
    Serial.println(F("report the output above as a GitHub issue."));
  }

  pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {
  // Heartbeat: fast blink = pass, slow blink = fail. Lets you read the
  // result without a serial monitor attached.
  const int period = (checksPassed == checksRun) ? 200 : 1000;
  digitalWrite(LED_BUILTIN, HIGH);
  delay(period);
  digitalWrite(LED_BUILTIN, LOW);
  delay(period);
}

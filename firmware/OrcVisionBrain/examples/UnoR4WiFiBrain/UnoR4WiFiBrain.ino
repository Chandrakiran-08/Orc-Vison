/*
 * Orc-Vison — Arduino Uno R4 WiFi autonomous brain
 * ---------------------------------------------------------------------------
 * The board does NOT run object detection. A host running the Orc-Vison
 * perception pipeline publishes detections over MQTT; this sketch runs the
 * *decision* layer locally and drives an actuator.
 *
 *   host (camera + model)  --MQTT-->  Uno R4 (brain)  -->  LED / relay / servo
 *
 * Why put the brain on the board at all? Because it keeps deciding — and
 * keeps its memory — when the link to the host drops. The host supplies
 * perception; autonomy stays local.
 *
 * The brain remembers outcomes, so an action that failed in a given
 * situation is chosen less readily next time, and a deterministic safety
 * floor forbids advancing toward a close hazard regardless of what the
 * policy has learned.
 *
 * Libraries (Arduino Library Manager):
 *   - WiFiS3            (bundled with the Arduino UNO R4 board package)
 *   - ArduinoMqttClient (by Arduino)
 *   - ArduinoJson       (by Benoit Blanchon, v6+ — StaticJsonDocument keeps
 *                        parsing memory bounded, which matters at 32 KB)
 *
 * Board: "Arduino UNO R4 WiFi" (Renesas RA4M1, 32 KB SRAM).
 *
 * RAM: the brain itself is ~1.5 KB with the tuning below. The JSON document
 * is the larger consumer; shrink JSON_CAPACITY if you are tight.
 *
 * SECURITY: this example uses plain MQTT with no authentication and no TLS.
 * Anyone who can reach the broker can publish detections (making the board
 * react to a hazard that is not there) or publish feedback (writing directly
 * into the brain's long-term memory). There is no sender identity, message
 * authentication or replay protection here. Fine on an isolated bench;
 * enable MQTT auth + TLS and per-topic ACLs before this drives anything that
 * moves, and keep a hardware interlock no firmware path can override.
 * See firmware/README.md for the full posture.
 *
 * NOTE: This sketch has NOT been flash-tested on hardware by the project
 * maintainers. The decision logic it calls into is verified against the
 * Python reference by a host-side parity test, but on-device behaviour is
 * unverified — review it and confirm on your own board before wiring
 * anything beyond the onboard LED.
 */

// Size the brain before including it. Defaults are already small; these
// values suit a single-hazard obstacle-avoidance role on a 32 KB board.
#define OV_MAX_OBJECTS 6
#define OV_MAX_LABELS 8
#define OV_MAX_TRACES 10

#include <ArduinoJson.h>
#include <ArduinoMqttClient.h>
#include <OrcVisionBrain.h>
#include <WiFiS3.h>

// ArduinoJson 6 and 7 spell the fixed-capacity document differently, and 7
// removed StaticJsonDocument entirely. Support both rather than silently
// failing to compile for whichever one the Library Manager installed.
#if ARDUINOJSON_VERSION_MAJOR >= 7
typedef JsonDocument OvJsonDoc;
#define OV_MAKE_JSON_DOC(name, cap) OvJsonDoc name
#else
#define OV_MAKE_JSON_DOC(name, cap) StaticJsonDocument<cap> name
#endif

// ---------------------------------------------------------------------------
// User configuration
// ---------------------------------------------------------------------------
const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

const char* MQTT_BROKER = "192.168.1.100";
const int MQTT_PORT = 1883;
const char* MQTT_CLIENTID = "orcvision-uno-r4-brain";

const char* TOPIC_EVENTS = "orcvision/events";      // detections in
const char* TOPIC_FEEDBACK = "orcvision/feedback";  // outcomes in
const char* TOPIC_ACTIONS = "orcvision/actions";    // decisions out

const int ALERT_PIN = LED_BUILTIN;
const unsigned long ACT_HOLD_MS = 1500;
const size_t JSON_CAPACITY = 1024;

// ---------------------------------------------------------------------------
WiFiClient wifiClient;
MqttClient mqttClient(wifiClient);
OrcVisionBrain brain;

unsigned long actUntil = 0;
uint32_t framesSeen = 0;

// ---------------------------------------------------------------------------
void connectWiFi() {
  // Sanity-check the radio before blaming the network.
  if (WiFi.status() == WL_NO_MODULE) {
    Serial.println("FATAL: WiFi module not responding.");
    while (true) delay(1000);
  }
  String fv = WiFi.firmwareVersion();
  if (fv < WIFI_FIRMWARE_LATEST_VERSION) {
    Serial.print("NOTE: WiFi firmware ");
    Serial.print(fv);
    Serial.print(" is older than ");
    Serial.print(WIFI_FIRMWARE_LATEST_VERSION);
    Serial.println(" — update via the UNO R4 firmware updater if WiFi misbehaves.");
  }

  // Call WiFi.begin() ONCE per attempt, then poll. Calling it inside the
  // loop condition restarts the association on every iteration, so DHCP
  // never completes and the board reports "connected" with IP 0.0.0.0 —
  // associated to the AP but with no usable address.
  for (int attempt = 1;; ++attempt) {
    Serial.print("Connecting to WiFi: ");
    Serial.println(WIFI_SSID);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    unsigned long deadline = millis() + 20000UL;
    while ((long)(millis() - deadline) < 0) {
      // Associated is not enough: wait for a real DHCP lease.
      if (WiFi.status() == WL_CONNECTED && (uint32_t)WiFi.localIP() != 0) {
        Serial.print("\nWiFi connected, IP: ");
        Serial.println(WiFi.localIP());
        Serial.print("Gateway: ");
        Serial.println(WiFi.gatewayIP());
        return;
      }
      Serial.print('.');
      delay(500);
    }

    Serial.print("\n  attempt ");
    Serial.print(attempt);
    Serial.print(" timed out (status=");
    Serial.print(WiFi.status());
    Serial.print(", ip=");
    Serial.print(WiFi.localIP());
    Serial.println("). Check the password, and that the AP is 2.4 GHz.");
    WiFi.disconnect();
    delay(3000);
  }
}

void connectMqtt() {
  mqttClient.setId(MQTT_CLIENTID);
  Serial.print("Connecting to MQTT broker ");
  Serial.print(MQTT_BROKER);
  Serial.print(":");
  Serial.println(MQTT_PORT);
  while (!mqttClient.connect(MQTT_BROKER, MQTT_PORT)) {
    Serial.print("  MQTT connect failed, error = ");
    Serial.print(mqttClient.connectError());
    Serial.println("  (-2 = cannot reach the broker)");
    Serial.println("  Check: broker IP correct? mosquitto listening on");
    Serial.println("  0.0.0.0:1883, not 127.0.0.1? firewall open? Some phone");
    Serial.println("  hotspots isolate clients, which blocks this entirely.");
    delay(3000);
  }
  Serial.println("MQTT connected.");
  mqttClient.subscribe(TOPIC_EVENTS);
  mqttClient.subscribe(TOPIC_FEEDBACK);
  Serial.println("Subscribed to events + feedback.");
}

// Drive the actuator for the chosen action. Everything hardware-specific
// lives here; the brain never knows about pins.
void driveActuator(const OvDecision& d) {
  switch (d.action) {
    case OV_STOP:
    case OV_AVOID:
    case OV_SIGNAL:
      // Replace with relay/servo/motor control as needed. For AVOID,
      // d.avoid_zone says which way to steer.
      digitalWrite(ALERT_PIN, HIGH);
      actUntil = millis() + ACT_HOLD_MS;
      break;
    default:  // MOVE / TRACK / WAIT — nothing to assert here
      break;
  }
}

void publishDecision(const OvDecision& d) {
  if (!mqttClient.connected()) return;
  mqttClient.beginMessage(TOPIC_ACTIONS);
  mqttClient.print("{\"action\":\"");
  mqttClient.print(OrcVisionBrain::actionName(d.action));
  mqttClient.print("\",\"score\":");
  mqttClient.print(d.score, 3);
  if (d.action == OV_AVOID) {
    mqttClient.print(",\"direction\":\"");
    mqttClient.print(OrcVisionBrain::zoneName(d.avoid_zone));
    mqttClient.print("\"");
  }
  mqttClient.print(",\"safety_fallback\":");
  mqttClient.print(d.safety_fallback ? "true" : "false");
  mqttClient.print("}");
  mqttClient.endMessage();
}

void handleEvent(const JsonDocument& doc) {
  // Frame dimensions let us accept pixel bboxes; normalized input also works.
  uint16_t h = doc["frame_shape"][0] | 0;
  uint16_t w = doc["frame_shape"][1] | 0;

  // Time base: the board's own clock, NOT the host's timestamp.
  //
  // The host sends a Unix epoch (~1.79e9). float32 has 24 bits of mantissa,
  // so representable values up there are ~128 seconds apart: every
  // frame-to-frame delta rounds to zero and dt collapses onto the library's
  // minimum-dt clamp. Measured effect on a 4-frame approach — approach_rate
  // reads 8e5 m/s instead of 1.6, and memory decay quantises into 128 s
  // steps. Motion *classification* happens to survive (a huge rate still
  // exceeds the threshold), so this fails quietly rather than obviously.
  // millis()/1000 is small, monotonic and millisecond-exact, which is all
  // the temporal reasoning needs.
  float ts = (float)millis() / 1000.0f;

  brain.beginFrame(ts);
  JsonArrayConst dets = doc["detections"];
  for (JsonObjectConst det : dets) {
    const char* label = det["label"] | "object";
    float conf = det["confidence"] | 1.0f;
    JsonArrayConst bbox = det["bbox"];
    if (bbox.size() < 4) continue;
    // depth_m is null when the host has no depth; map that to the sentinel.
    float depth =
        det["depth_m"].isNull() ? OV_UNKNOWN_DEPTH : det["depth_m"].as<float>();
    int16_t track =
        det["track_id"].isNull() ? (int16_t)-1 : (int16_t)det["track_id"].as<int>();

    if (w > 0 && h > 0) {
      brain.observePixels(label, conf, bbox[0].as<float>(), bbox[1].as<float>(),
                          bbox[2].as<float>(), bbox[3].as<float>(), w, h, depth, track);
    } else {
      float x1 = bbox[0].as<float>(), y1 = bbox[1].as<float>();
      float x2 = bbox[2].as<float>(), y2 = bbox[3].as<float>();
      brain.observe(label, conf, (x1 + x2) * 0.5f, (y1 + y2) * 0.5f,
                    (x2 - x1) * (y2 - y1), depth, track);
    }
  }
  brain.endFrame();

  OvDecision decision = brain.decide();
  brain.markExecuted(decision);
  driveActuator(decision);
  publishDecision(decision);

  ++framesSeen;
  char buf[220];
  brain.explain(decision, buf, sizeof(buf));
  Serial.println(buf);
}

// The host (or an operator, or a bumper switch) tells the board how the last
// action turned out. This is what makes the next decision different.
void handleFeedback(const JsonDocument& doc) {
  bool success = doc["success"] | false;
  brain.feedback(success);
  brain.learn();
  Serial.print("feedback: last action ");
  Serial.println(success ? "succeeded" : "FAILED (will be avoided next time)");
}

void onMqttMessage(int messageSize) {
  (void)messageSize;
  String topic = mqttClient.messageTopic();

  OV_MAKE_JSON_DOC(doc, JSON_CAPACITY);
  DeserializationError err = deserializeJson(doc, mqttClient);
  if (err) {
    Serial.print("JSON parse failed: ");
    Serial.println(err.c_str());
    return;
  }
  if (topic == TOPIC_FEEDBACK) {
    handleFeedback(doc);
  } else {
    handleEvent(doc);
  }
}

// ---------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000) { /* brief wait for USB serial */ }

  pinMode(ALERT_PIN, OUTPUT);
  digitalWrite(ALERT_PIN, LOW);

  // Teach the brain which classes matter. Order fixes the label ids.
  brain.begin();
  brain.addHazardLabel("person");
  brain.addHazardLabel("obstacle");
  brain.addHazardLabel("vehicle");
  brain.setGoal(1);  // avoid_collision
  brain.config().max_range_m = 5.0f;
  brain.config().veto_proximity = 0.6f;  // never advance closer than this

  Serial.print("OrcVisionBrain ready, using ");
  Serial.print((unsigned)sizeof(OrcVisionBrain));
  Serial.println(" bytes of SRAM");

  connectWiFi();
  mqttClient.onMessage(onMqttMessage);
  connectMqtt();
}

void loop() {
  if (!mqttClient.connected()) connectMqtt();
  mqttClient.poll();

  // Rollover-safe: millis() wraps after ~49 days, so compare a signed
  // difference rather than the raw values.
  if (actUntil != 0 && (long)(millis() - actUntil) >= 0) {
    digitalWrite(ALERT_PIN, LOW);
    actUntil = 0;
  }
}

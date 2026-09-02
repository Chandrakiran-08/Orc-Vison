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
  Serial.print("Connecting to WiFi: ");
  Serial.println(WIFI_SSID);
  while (WiFi.begin(WIFI_SSID, WIFI_PASSWORD) != WL_CONNECTED) {
    Serial.println("  ...retrying in 3s");
    delay(3000);
  }
  Serial.print("WiFi connected, IP: ");
  Serial.println(WiFi.localIP());
}

void connectMqtt() {
  mqttClient.setId(MQTT_CLIENTID);
  while (!mqttClient.connect(MQTT_BROKER, MQTT_PORT)) {
    Serial.print("  MQTT connect failed, error = ");
    Serial.println(mqttClient.connectError());
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
  float ts = doc["timestamp"] | (float)(millis() / 1000.0f);

  brain.beginFrame(ts);
  JsonArrayConst dets = doc["detections"];
  for (JsonObjectConst det : dets) {
    const char* label = det["label"] | "object";
    float conf = det["confidence"] | 1.0f;
    JsonArrayConst bbox = det["bbox"];
    if (bbox.size() < 4) continue;
    // depth_m is null when the host has no depth; map that to the sentinel.
    float depth = det["depth_m"].isNull() ? OV_UNKNOWN_DEPTH : (float)det["depth_m"];
    int16_t track = det["track_id"].isNull() ? -1 : (int16_t)det["track_id"];

    if (w > 0 && h > 0) {
      brain.observePixels(label, conf, bbox[0], bbox[1], bbox[2], bbox[3], w, h, depth,
                          track);
    } else {
      float cx = ((float)bbox[0] + (float)bbox[2]) * 0.5f;
      float cy = ((float)bbox[1] + (float)bbox[3]) * 0.5f;
      float size = ((float)bbox[2] - (float)bbox[0]) * ((float)bbox[3] - (float)bbox[1]);
      brain.observe(label, conf, cx, cy, size, depth, track);
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

  StaticJsonDocument<JSON_CAPACITY> doc;
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

  if (actUntil != 0 && millis() > actUntil) {
    digitalWrite(ALERT_PIN, LOW);
    actUntil = 0;
  }
}

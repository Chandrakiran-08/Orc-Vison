/*
 * Orc Vision — Arduino Uno R4 WiFi alert receiver
 * ------------------------------------------------
 * Subscribes to the Orc Vision perception-events MQTT topic, parses each
 * JSON event, and drives the onboard LED when the decision layer raised an
 * alert (i.e. the event's "alerts" array is non-empty).
 *
 * Libraries (install via Arduino Library Manager):
 *   - WiFiS3            (bundled with the Arduino UNO R4 board package)
 *   - ArduinoMqttClient (by Arduino)
 *   - Arduino_JSON      (by Arduino)  <-- simple JSON parser used below
 *
 * Board: "Arduino UNO R4 WiFi" (Renesas RA4M1 core).
 *
 * NOTE: This sketch is provided as-is and has NOT been flash-tested by the
 * project maintainers at the time of writing. Review the wiring/actuator
 * section before connecting anything beyond the onboard LED.
 */

#include <WiFiS3.h>
#include <ArduinoMqttClient.h>
#include <Arduino_JSON.h>

// ---------------------------------------------------------------------------
// User configuration — edit these for your network / broker.
// ---------------------------------------------------------------------------
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

const char* MQTT_BROKER   = "192.168.1.100";  // host running mosquitto
const int   MQTT_PORT     = 1883;
const char* MQTT_TOPIC    = "orcvision/events";
const char* MQTT_CLIENTID = "orcvision-uno-r4";

// Actuator pin. LED_BUILTIN is the onboard LED. To drive a relay or servo
// instead, wire it here and adapt triggerActuator() below.
const int ALERT_PIN       = LED_BUILTIN;
const unsigned long ALERT_HOLD_MS = 2000;  // how long to hold the LED on

// ---------------------------------------------------------------------------
WiFiClient   wifiClient;
MqttClient   mqttClient(wifiClient);

unsigned long alertUntil = 0;  // millis() timestamp to turn the LED off

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
  Serial.print("Connecting to MQTT broker ");
  Serial.print(MQTT_BROKER);
  Serial.print(":");
  Serial.println(MQTT_PORT);
  while (!mqttClient.connect(MQTT_BROKER, MQTT_PORT)) {
    Serial.print("  MQTT connect failed, error = ");
    Serial.println(mqttClient.connectError());
    delay(3000);
  }
  Serial.println("MQTT connected.");
  mqttClient.subscribe(MQTT_TOPIC);
  Serial.print("Subscribed to ");
  Serial.println(MQTT_TOPIC);
}

void triggerActuator() {
  // Onboard LED. Replace with relay/servo logic as needed:
  //   * Relay module: digitalWrite(ALERT_PIN, HIGH) to close the contact.
  //   * Servo: #include <Servo.h>, attach a Servo, and servo.write(angle).
  digitalWrite(ALERT_PIN, HIGH);
  alertUntil = millis() + ALERT_HOLD_MS;
}

// Called for every incoming MQTT message.
void onMqttMessage(int messageSize) {
  String payload;
  payload.reserve(messageSize);
  while (mqttClient.available()) {
    payload += (char)mqttClient.read();
  }

  JSONVar event = JSON.parse(payload);
  if (JSON.typeof(event) == "undefined") {
    Serial.println("  (could not parse JSON payload)");
    return;
  }

  int frameId = (int)(double)event["frame_id"];
  JSONVar alerts = event["alerts"];
  int alertCount = alerts.length();

  Serial.print("frame ");
  Serial.print(frameId);
  Serial.print(": alerts=");
  Serial.println(alertCount);

  if (alertCount > 0) {
    for (int i = 0; i < alertCount; i++) {
      Serial.print("  ALERT: ");
      Serial.println((const char*)alerts[i]);
    }
    triggerActuator();
  }
}

// ---------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000) { /* wait briefly for USB serial */ }

  pinMode(ALERT_PIN, OUTPUT);
  digitalWrite(ALERT_PIN, LOW);

  connectWiFi();
  mqttClient.onMessage(onMqttMessage);
  connectMqtt();
}

void loop() {
  // Reconnect if the broker connection dropped.
  if (!mqttClient.connected()) {
    connectMqtt();
  }

  // Keep the MQTT client alive and process incoming messages.
  mqttClient.poll();

  // Auto-clear the actuator after the hold window.
  if (alertUntil != 0 && millis() > alertUntil) {
    digitalWrite(ALERT_PIN, LOW);
    alertUntil = 0;
  }
}

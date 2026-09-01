/*
 * Orc Vision — ESP32 alert receiver (REFERENCE ONLY, UNTESTED)
 * -----------------------------------------------------------
 * Reference sketch showing the same idea as uno_r4_wifi_alert.ino using the
 * common ESP32 stack (WiFi.h + PubSubClient). This is provided as a
 * starting point and has NOT been tested on hardware. Adapt to your board.
 *
 * Libraries:
 *   - WiFi.h        (bundled with the ESP32 Arduino core)
 *   - PubSubClient  (by Nick O'Leary)
 *   - ArduinoJson   (by Benoit Blanchon, v6+)
 *
 * Board: any ESP32 dev module.
 */

#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char* MQTT_BROKER   = "192.168.1.100";
const int   MQTT_PORT     = 1883;
const char* MQTT_TOPIC    = "orcvision/events";
const char* MQTT_CLIENTID = "orcvision-esp32";
const int   ALERT_PIN     = 2;  // onboard LED on many ESP32 dev boards

WiFiClient   wifiClient;
PubSubClient mqtt(wifiClient);

void connectWiFi() {
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.print("\nWiFi connected, IP: ");
  Serial.println(WiFi.localIP());
}

void onMessage(char* topic, byte* payload, unsigned int length) {
  StaticJsonDocument<2048> doc;
  DeserializationError err = deserializeJson(doc, payload, length);
  if (err) {
    Serial.print("JSON parse failed: ");
    Serial.println(err.c_str());
    return;
  }
  JsonArray alerts = doc["alerts"].as<JsonArray>();
  Serial.print("frame ");
  Serial.print((int)doc["frame_id"]);
  Serial.print(": alerts=");
  Serial.println(alerts.size());
  if (alerts.size() > 0) {
    digitalWrite(ALERT_PIN, HIGH);
    delay(2000);
    digitalWrite(ALERT_PIN, LOW);
  }
}

void connectMqtt() {
  mqtt.setServer(MQTT_BROKER, MQTT_PORT);
  mqtt.setCallback(onMessage);
  while (!mqtt.connected()) {
    if (mqtt.connect(MQTT_CLIENTID)) {
      mqtt.subscribe(MQTT_TOPIC);
      Serial.println("MQTT connected + subscribed.");
    } else {
      Serial.print("MQTT failed, rc=");
      Serial.println(mqtt.state());
      delay(3000);
    }
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(ALERT_PIN, OUTPUT);
  digitalWrite(ALERT_PIN, LOW);
  connectWiFi();
  connectMqtt();
}

void loop() {
  if (!mqtt.connected()) connectMqtt();
  mqtt.loop();
}

# Orc-Vison - Raspberry Pi Pico W alert receiver (REFERENCE ONLY, UNTESTED)
# --------------------------------------------------------------------------
# MicroPython reference sketch. Subscribes to the Orc-Vison perception-events
# MQTT topic and lights the onboard LED when an event carries alerts.
#
# This is provided as a starting point and has NOT been tested on hardware.
#
# Requirements:
#   - MicroPython firmware for Raspberry Pi Pico W
#   - umqtt.simple  (copy umqtt/simple.py onto the board, or use mip)
#
# Copy this file onto the board as main.py, edit the config, and reset.

import json
import time

import network
from machine import Pin
from umqtt.simple import MQTTClient

# --- user configuration ----------------------------------------------------
WIFI_SSID = "YOUR_WIFI_SSID"
WIFI_PASSWORD = "YOUR_WIFI_PASSWORD"

MQTT_BROKER = "192.168.1.100"
MQTT_PORT = 1883
MQTT_TOPIC = b"orcvision/events"
MQTT_CLIENT_ID = b"orcvision-pico-w"
ALERT_HOLD_S = 2

led = Pin("LED", Pin.OUT)  # onboard LED on the Pico W


def connect_wifi():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    wlan.connect(WIFI_SSID, WIFI_PASSWORD)
    while not wlan.isconnected():
        print("connecting to WiFi...")
        time.sleep(1)
    print("WiFi connected:", wlan.ifconfig()[0])


def on_message(topic, msg):
    try:
        event = json.loads(msg)
    except ValueError:
        print("could not parse JSON payload")
        return
    alerts = event.get("alerts", [])
    print("frame", event.get("frame_id"), "alerts=", len(alerts))
    if alerts:
        for a in alerts:
            print("  ALERT:", a)
        led.on()
        time.sleep(ALERT_HOLD_S)
        led.off()


def main():
    connect_wifi()
    client = MQTTClient(MQTT_CLIENT_ID, MQTT_BROKER, port=MQTT_PORT, keepalive=60)
    client.set_callback(on_message)
    client.connect()
    client.subscribe(MQTT_TOPIC)
    print("subscribed to", MQTT_TOPIC)
    try:
        while True:
            client.wait_msg()  # blocking; use check_msg() for a non-blocking loop
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()

# Microcontroller integration

Orc-Vison publishes perception events as JSON over MQTT (`--sink mqtt`).
Any device that can subscribe to the topic and parse JSON can react to
alerts. These examples flash the onboard LED when an event's `alerts`
array is non-empty — swap the LED for a relay/servo/buzzer as needed.

## Verification status

| File | Device | Status |
|------|--------|--------|
| `uno_r4_wifi_alert.ino` | Arduino Uno R4 WiFi | **Firmware written, pending real-hardware verification by the maintainer.** Not yet flash-tested. |
| `esp32_reference.ino` | ESP32 | Reference only — **untested**. |
| `pico_w_reference.py` | Raspberry Pi Pico W (MicroPython) | Reference only — **untested**. |
| `../../orcvision/mqtt_simulator.py` | (software) | **Verified** — run it right now to test the full pipeline with no hardware. |

> The three firmware examples are provided as correct-by-construction
> starting points. Treat them as untested until you flash and confirm on
> your own device. Start with `mqtt_simulator.py` to validate the pipeline,
> then move to real hardware.

## Test the whole loop without hardware first

```bash
# 1. Start a broker (Ubuntu/Pop!_OS: sudo apt install mosquitto)
mosquitto -v

# 2. Start the software receiver (stands in for the microcontroller)
python -m orcvision.mqtt_simulator --host localhost --topic orcvision/events

# 3. Run the pipeline into MQTT
python -m orcvision run \
    --source tests/fixtures/sample.mp4 \
    --sink mqtt --model yolov8n
```

The simulator prints each event and a `>>> [ALERT]` line whenever the
decision layer fired — exactly what a real MCU sketch reacts to.

## Arduino Uno R4 WiFi (`uno_r4_wifi_alert.ino`)

**Libraries** (Arduino Library Manager):
- `WiFiS3` — bundled with the *Arduino UNO R4 Boards* package
- `ArduinoMqttClient`
- `Arduino_JSON`

**Steps:**
1. Board Manager → install **Arduino UNO R4 Boards**.
2. Select board *Arduino UNO R4 WiFi* and the serial port.
3. Edit `WIFI_SSID`, `WIFI_PASSWORD`, `MQTT_BROKER`, `MQTT_TOPIC`.
4. Upload, then open Serial Monitor at 115200 baud.
5. Run the pipeline with `--sink mqtt --host <broker-ip>`.

**Wiring the onboard LED** needs nothing extra. To drive a **relay**:
connect the module's IN pin to `ALERT_PIN`, VCC/GND to 5V/GND, and use the
relay's COM/NO contacts to switch your load — adapt `triggerActuator()`.
For a **servo**, add `#include <Servo.h>`, attach it, and call
`servo.write(angle)` from `triggerActuator()`.

## ESP32 (`esp32_reference.ino`) — reference only

Libraries: `WiFi.h` (ESP32 core), `PubSubClient`, `ArduinoJson` (v6+).
Onboard LED is usually GPIO 2. Untested — adapt to your board.

## Raspberry Pi Pico W (`pico_w_reference.py`) — reference only

Flash MicroPython for the Pico W, copy `umqtt/simple.py` onto the board,
save this file as `main.py`, edit the config, and reset. Untested.

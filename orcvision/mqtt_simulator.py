"""Software stand-in for a microcontroller receiver.

Subscribes to the same MQTT topic the pipeline publishes to and logs every
perception event it receives, highlighting any that carry alerts. Use this
to exercise the full pipeline end-to-end without any physical hardware.

Usage
-----
Terminal 1 — start a broker (Ubuntu: ``sudo apt install mosquitto``)::

    mosquitto -v

Terminal 2 — start the simulator::

    python -m orcvision.mqtt_simulator --host localhost --topic orcvision/events

Terminal 3 — run the pipeline into MQTT::

    python -m orcvision run --source tests/fixtures/sample.mp4 \\
        --sink mqtt --model yolov8n

The simulator prints each event and flashes a "[ALERT]" line whenever the
decision layer populated ``alerts`` — exactly what an Arduino sketch would
react to by driving an LED or relay.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys


def _on_connect(client, userdata, flags, reason_code, properties=None):
    topic = userdata["topic"]
    print(f"[sim] connected (rc={reason_code}); subscribing to {topic!r}")
    client.subscribe(topic, qos=userdata["qos"])


def _on_message(client, userdata, msg):
    try:
        event = json.loads(msg.payload.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        print(f"[sim] non-JSON payload on {msg.topic}: {msg.payload!r}")
        return

    n = len(event.get("detections", []))
    alerts = event.get("alerts", [])
    frame_id = event.get("frame_id", "?")
    labels = ", ".join(sorted({d["label"] for d in event.get("detections", [])})) or "-"
    print(f"[sim] frame {frame_id}: {n} detection(s) [{labels}]")
    for alert in alerts:
        # This is where an MCU would drive an LED / relay / servo.
        print(f"[sim]   >>> [ALERT] {alert}  (would trigger actuator)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Simulated MCU MQTT receiver.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--topic", default="orcvision/events")
    parser.add_argument("--qos", type=int, default=0)
    args = parser.parse_args(argv)

    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        print("paho-mqtt is required. Install orcvision (paho-mqtt is a base dep).")
        return 1

    userdata = {"topic": args.topic, "qos": args.qos}
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, userdata=userdata)
    except (AttributeError, TypeError):  # pragma: no cover - older paho
        client = mqtt.Client(userdata=userdata)
    client.on_connect = _on_connect
    client.on_message = _on_message

    try:
        client.connect(args.host, args.port, keepalive=60)
    except OSError as exc:
        print(f"[sim] could not connect to broker at {args.host}:{args.port}: {exc}")
        print("[sim] is a broker running? Try: mosquitto -v")
        return 1

    signal.signal(signal.SIGINT, lambda *_: (client.disconnect(), sys.exit(0)))
    print(f"[sim] listening on {args.host}:{args.port} topic={args.topic!r} (Ctrl-C to stop)")
    client.loop_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

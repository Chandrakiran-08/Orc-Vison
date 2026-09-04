#!/usr/bin/env bash
#
# One-command memory-flip demo for the Uno R4 brain — no camera, no YOLO.
#
# It publishes three MQTT messages to the board and narrates what to watch
# for on the serial monitor:
#
#   1. an obstacle, close and centre   -> the board decides (AVOID)
#   2. feedback that the choice failed -> the board remembers
#   3. the SAME obstacle again         -> the board decides differently (STOP)
#
# That is the whole point of the project: identical input, different action,
# because the microcontroller remembered. You do not need the perception
# pipeline or torch for this — the board only cares about the JSON.
#
# Usage (with the serial monitor already open in another terminal):
#   ./firmware/board_demo.sh              # broker on localhost
#   ./firmware/board_demo.sh 10.192.244.19  # or name the broker host

set -euo pipefail

BROKER="${1:-localhost}"
EVENTS="orcvision/events"
FEEDBACK="orcvision/feedback"

OBSTACLE='{"timestamp":1.0,"frame_shape":[480,640],"detections":[{"label":"obstacle","confidence":0.9,"bbox":[280,200,360,300],"depth_m":1.0,"track_id":1}]}'

command -v mosquitto_pub >/dev/null || {
  echo "mosquitto_pub not found. Install: sudo apt install mosquitto-clients" >&2
  exit 1
}

pub() { mosquitto_pub -h "$BROKER" -t "$1" -m "$2"; }

echo "Broker: $BROKER"
echo "Watch the serial monitor as each step runs."
echo

echo "STEP 1  obstacle appears (close, centre) — the board decides..."
pub "$EVENTS" "$OBSTACLE"
sleep 3

echo "STEP 2  telling the board that decision FAILED..."
pub "$FEEDBACK" '{"success":false}'
sleep 2

echo "STEP 3  the SAME obstacle again — watch it choose differently..."
pub "$EVENTS" "$OBSTACLE"
sleep 1

echo
echo "Done. On the serial monitor you should see two decisions:"
echo "  step 1 -> AVOID (or STOP)"
echo "  step 3 -> a DIFFERENT action, with 'previously failed here' in the reason."
echo
echo "Same message both times. Different decision. That is the memory flip,"
echo "running on the microcontroller."

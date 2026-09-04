#!/usr/bin/env bash
#
# Build and flash the Orc-Vison brain onto an Arduino Uno R4 WiFi.
#
# Uses arduino-cli rather than the GUI, because every step is then explicit
# and reproducible — and because the Arduino IDE packaged in most distro
# repositories is 1.8.x, which does not support the Uno R4 at all.
#
#   ./firmware/flash_uno_r4.sh check      # toolchain only, no board needed
#   ./firmware/flash_uno_r4.sh compile    # compile the self test (no board)
#   ./firmware/flash_uno_r4.sh upload     # compile + flash + open the monitor
#   ./firmware/flash_uno_r4.sh upload wifi  # the networked sketch instead
#
# Start with `compile`. It needs no hardware and settles whether the sketch
# builds for the real target, which is the one thing host tests cannot prove.

set -euo pipefail

FQBN="arduino:renesas_uno:unor4wifi"
CORE="arduino:renesas_uno"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="$HERE/OrcVisionBrain"
BAUD=115200

SKETCH_NAME="${2:-selftest}"
case "$SKETCH_NAME" in
  selftest) SKETCH="$LIB_DIR/examples/BrainSelfTest" ; NEEDS_LIBS=0 ;;
  wifi)     SKETCH="$LIB_DIR/examples/UnoR4WiFiBrain"; NEEDS_LIBS=1 ;;
  *) echo "Unknown sketch '$SKETCH_NAME' (use: selftest | wifi)" >&2; exit 2 ;;
esac

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33m    %s\033[0m\n' "$*"; }
die()  { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# --- arduino-cli ------------------------------------------------------------
ensure_cli() {
  if command -v arduino-cli >/dev/null 2>&1; then
    say "arduino-cli found: $(arduino-cli version)"
    return
  fi
  say "Installing arduino-cli into ~/.local/bin"
  mkdir -p "$HOME/.local/bin"
  curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh \
    | BINDIR="$HOME/.local/bin" sh
  export PATH="$HOME/.local/bin:$PATH"
  command -v arduino-cli >/dev/null 2>&1 || die "install failed; add ~/.local/bin to PATH"
  warn "Add this to ~/.bashrc:  export PATH=\"\$HOME/.local/bin:\$PATH\""
}

ensure_core() {
  if arduino-cli core list 2>/dev/null | grep -q "^${CORE}"; then
    say "Board core '$CORE' already installed"
    return
  fi
  say "Installing board core '$CORE' (a few hundred MB, one time)"
  arduino-cli core update-index
  arduino-cli core install "$CORE"
}

ensure_libs() {
  [ "$NEEDS_LIBS" -eq 1 ] || return 0
  say "Installing sketch libraries (ArduinoMqttClient, ArduinoJson 6.x)"
  arduino-cli lib install "ArduinoMqttClient" >/dev/null

  # Pin ArduinoJson to 6.x deliberately. Version 7 removed
  # StaticJsonDocument, and its JsonDocument allocates from the heap — a heap
  # allocation inside an MQTT callback on a 32 KB board is the fragmentation
  # risk this project avoids everywhere else. 6.x keeps the document in a
  # fixed-size buffer. The sketch compiles under either (see the version
  # guard at the top of it), but 6.x is the right one for this board.
  if arduino-cli lib list 2>/dev/null | grep -qi "^ArduinoJson[[:space:]]\+7\."; then
    warn "ArduinoJson 7.x is installed; downgrading to 6.x for a heap-free document"
  fi
  arduino-cli lib install "ArduinoJson@6.21.5" >/dev/null
}

find_port() {
  # Prefer a port arduino-cli itself matches to the board.
  local detected
  detected="$(arduino-cli board list 2>/dev/null | awk -v f="$FQBN" '$0 ~ f {print $1; exit}')"
  if [ -n "$detected" ]; then echo "$detected"; return; fi
  for candidate in /dev/ttyACM0 /dev/ttyACM1 /dev/ttyUSB0; do
    [ -e "$candidate" ] && { echo "$candidate"; return; }
  done
  echo ""
}

check_permissions() {
  local port="$1"
  [ -w "$port" ] && return 0
  warn "No write permission on $port."
  warn "Fix with:  sudo usermod -aG dialout \$USER   then LOG OUT and back in."
  warn "To test right now without logging out:  sudo chmod a+rw $port"
  return 1
}

# --- actions ----------------------------------------------------------------
do_compile() {
  say "Compiling $(basename "$SKETCH") for $FQBN"
  arduino-cli compile --fqbn "$FQBN" --library "$LIB_DIR" --warnings all "$SKETCH"
  say "COMPILE OK — the sketch builds for the real target"
}

do_upload() {
  local port; port="$(find_port)"
  [ -n "$port" ] || die "No board found. Plug in the Uno R4 and check: arduino-cli board list"
  say "Board port: $port"
  check_permissions "$port" || die "Cannot write to $port (see above)"

  do_compile
  say "Uploading to $port"
  arduino-cli upload -p "$port" --fqbn "$FQBN" "$SKETCH"
  say "UPLOAD OK — opening serial monitor at $BAUD (Ctrl-C to exit)"
  sleep 2
  arduino-cli monitor -p "$port" -c baudrate=$BAUD
}

case "${1:-check}" in
  check)
    ensure_cli; ensure_core
    say "Detected boards:"; arduino-cli board list || true
    say "Toolchain ready. Next:  $0 compile"
    ;;
  compile) ensure_cli; ensure_core; ensure_libs; do_compile ;;
  upload)  ensure_cli; ensure_core; ensure_libs; do_upload ;;
  *) echo "Usage: $0 {check|compile|upload} [selftest|wifi]" >&2; exit 2 ;;
esac

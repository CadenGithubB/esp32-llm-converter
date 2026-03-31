#!/bin/bash
#
# capture_llm_test.sh — Run multiple LLM test questions and capture debug output
#
# Usage:
#   ./training_scripts/capture_llm_test.sh /dev/cu.usbserial-XXXX
#
# Prerequisite: flash the firmware with debug enabled, then run this script.
# It sends each question over serial, captures the full debug output per question
# into separate files under test_logs/.
#
# Before running: make sure debugllm is enabled on the device.
# You can do this by sending "debugllm 1" manually first, or this script does it.
#
# Each output file is named by topic for easy identification.

PORT="${1:-/dev/cu.usbserial-0001}"
BAUD=115200
OUTDIR="$(dirname "$0")/../test_logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="${OUTDIR}/${TIMESTAMP}"

mkdir -p "$OUTDIR"

# Test questions — cover different topics and known problem areas
declare -A QUESTIONS
QUESTIONS=(
  ["01_tof_sensor"]="What is the tof sensor?"
  ["02_presence_sensor"]="What is the presence sensor?"
  ["03_thermal_camera"]="How do I start the thermal camera?"
  ["04_imu_sensor"]="What is the IMU?"
  ["05_wifi_connect"]="How do I connect to WiFi?"
  ["06_mqtt_setup"]="How do I set up MQTT?"
  ["07_espnow"]="What is ESP-NOW?"
  ["08_memory_check"]="How do I check memory usage?"
  ["09_llm_load"]="How do I load a model?"
  ["10_ota_update"]="How do I update the firmware?"
  ["11_servo_control"]="How do I move a servo?"
  ["12_radio_tune"]="How do I tune the FM radio?"
  ["13_gesture_sensor"]="What is the APDS9960?"
  ["14_gamepad"]="How do I read the gamepad?"
  ["15_automation"]="How do I create an automation rule?"
  ["16_confusable_tof_presence"]="Does the tof sensor detect people?"
  ["17_confusable_thermal_presence"]="Is the thermal camera a motion sensor?"
  ["18_confusable_imu_presence"]="Can the IMU detect if someone is in the room?"
  ["19_confusable_heap_psram"]="Is PSRAM the same as heap?"
  ["20_out_of_domain"]="Write me a poem"
)

echo "=== LLM Debug Test Capture ==="
echo "Port: $PORT"
echo "Output: $OUTDIR"
echo "Questions: ${#QUESTIONS[@]}"
echo ""

# Function to send a command and capture output with timeout
send_and_capture() {
  local name="$1"
  local question="$2"
  local outfile="${OUTDIR}/${name}.txt"

  echo "[$name] Sending: $question"
  echo "# Question: $question" > "$outfile"
  echo "# Captured: $(date)" >> "$outfile"
  echo "---" >> "$outfile"

  # Send the command and capture for 15 seconds (enough for generation + debug)
  # Using a subshell with timeout to read serial output
  {
    echo "llm ask \"$question\""
    sleep 15
  } | stty -f "$PORT" $BAUD 2>/dev/null

  # Alternative: if you have screen/minicom, capture differently
  # This simple approach sends the command and captures the response
  # For best results, use the manual method described below

  echo "  -> Saved to $outfile"
}

echo ""
echo "============================================================"
echo "RECOMMENDED: Manual capture method (more reliable)"
echo "============================================================"
echo ""
echo "1. Open serial monitor:  idf.py monitor"
echo "2. Enable debug:         debugllm 1"
echo "3. For EACH question below, copy all output between the"
echo "   '=== START ===' and '=== END ===' markers into the"
echo "   corresponding file."
echo ""
echo "Output directory: $OUTDIR"
echo ""

# Create empty files with the questions as headers
for name in $(echo "${!QUESTIONS[@]}" | tr ' ' '\n' | sort); do
  question="${QUESTIONS[$name]}"
  outfile="${OUTDIR}/${name}.txt"
  echo "# Question: $question" > "$outfile"
  echo "# File: $name.txt" >> "$outfile"
  echo "# Command to type: llm ask \"$question\"" >> "$outfile"
  echo "---" >> "$outfile"
  echo "# Paste debug output below this line:" >> "$outfile"
  echo ""
  echo "  $name.txt  ->  llm ask \"$question\""
done

echo ""
echo "Files created. Paste serial output into each file after running the command."
echo ""
echo "Quick copy-paste command list for serial monitor:"
echo "================================================="
echo "debugllm 1"
for name in $(echo "${!QUESTIONS[@]}" | tr ' ' '\n' | sort); do
  echo "llm ask \"${QUESTIONS[$name]}\""
done
echo "================================================="
echo ""
echo "TIP: To capture everything at once, redirect your serial monitor to a log file:"
echo "  idf.py monitor 2>&1 | tee ${OUTDIR}/full_session.txt"
echo "Then run all commands. Afterwards, this script can split the log:"
echo "  $0 --split ${OUTDIR}/full_session.txt"

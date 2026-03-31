#!/usr/bin/env python3
"""Check for references to sensors/chips that don't exist in the firmware.

Scans the training file for known-bad sensor names and chip references
that were previously hallucinated. Also checks for any unknown chip
part numbers that might have crept in.

Update KNOWN_BAD and VALID_CHIPS as the firmware evolves.

Usage:
    python training_scripts/check_hallucinated_sensors.py [FILE]
"""
import re
import sys
from pathlib import Path


# Sensors/chips that do NOT exist in Hardware One firmware
KNOWN_BAD = [
    "BME280",
    "BME680",
    "BME688",
    "ICM-42688",
    "ICM42688",
    "MPU6050",
    "MPU9250",
    "MLX90614",     # not in firmware; only MLX90640
    "SHT31",
    "SHT40",
    "DHT11",
    "DHT22",
    "BMP280",
    "BMP388",
    "LIS3DH",
    "LSM6DS",
    "AHT20",
]

# Valid chips in the firmware (for reference)
VALID_CHIPS = [
    "BNO055",       # IMU
    "VL53L4CX",     # ToF distance
    "MLX90640",     # Thermal camera
    "STHS34PF80",   # IR presence/motion
    "PA1010D",      # GPS
    "DS3231",       # RTC
    "APDS9960",     # Gesture/proximity/color
    "APDS-9960",    # Alternate formatting
    "RDA5807",      # FM radio
    "PCA9685",      # Servo PWM driver
    "ESP32-S3",     # Main MCU
    "SSD1306",      # OLED display
]

# Pattern to catch unknown chip part numbers (letter+digit combos)
PART_NUMBER_RE = re.compile(r'\b[A-Z]{2,5}\d{3,5}[A-Z]?\b')


def main():
    default = Path(__file__).resolve().parent.parent / "training_data" / "hardwareone_rich.txt"
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default

    with open(path) as f:
        lines = f.readlines()

    valid_set = {v.upper() for v in VALID_CHIPS}
    bad_set = {b.upper() for b in KNOWN_BAD}
    problems = []
    unknown_parts = set()

    for i, line in enumerate(lines, 1):
        # Check known-bad sensors
        for bad in KNOWN_BAD:
            if bad.lower() in line.lower():
                problems.append((i, bad, line.strip()))

        # Check for unknown part numbers
        for match in PART_NUMBER_RE.findall(line):
            if match.upper() not in valid_set and match.upper() not in bad_set:
                # Filter out common false positives
                if match not in ("TEMP", "THEN", "HTTP", "JSON", "MQTT",
                                 "SSID", "OLED", "HTML", "PSRAM", "UART",
                                 "GPIO", "USB", "RGB", "LED", "OTA",
                                 "NTP", "PWM", "I2C", "SPI", "BLE",
                                 "TCP", "UDP", "CLI", "MAC", "IRAM",
                                 "DRAM", "INT8", "FF00FF"):
                    unknown_parts.add((match, i, line.strip()[:80]))

    if problems:
        print(f"FOUND {len(problems)} references to hallucinated/invalid sensors:\n")
        for lineno, bad, text in problems:
            print(f"  Line {lineno}: [{bad}] {text[:90]}")
    else:
        print("No hallucinated sensor references found. Clean.")

    if unknown_parts:
        print(f"\nFound {len(unknown_parts)} unknown part numbers (review manually):\n")
        for part, lineno, text in sorted(unknown_parts):
            print(f"  Line {lineno}: [{part}] {text}")

    print(f"\nValid chips for reference: {', '.join(VALID_CHIPS)}")


if __name__ == "__main__":
    main()

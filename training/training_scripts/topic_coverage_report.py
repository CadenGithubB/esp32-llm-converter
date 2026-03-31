#!/usr/bin/env python3
"""Report QA pair counts per topic/sensor to find thin coverage.

Groups questions by keyword matching and reports how many QA pairs
each topic has. Topics with fewer than 20 pairs are flagged as thin
and may need expansion to prevent hallucination.

Usage:
    python training_scripts/topic_coverage_report.py [FILE]
"""
import re
import sys
from collections import defaultdict
from pathlib import Path


# Topic definitions: name -> list of keywords to match in Q: lines
TOPICS = {
    "tof/distance":     ["tof", "vl53", "distance", "time-of-flight", "time of flight"],
    "imu/motion":       ["imu", "bno055", "acceleration", "gyroscope", "inertial"],
    "thermal":          ["thermal", "mlx90640", "heat map", "heatmap"],
    "presence":         ["presence", "sths34", "ir sensor", "infrared presence"],
    "gps":              ["gps", "pa1010", "latitude", "longitude", "coordinates"],
    "rtc":              ["rtc", "ds3231", "real-time clock", "realtime clock"],
    "gesture/apds":     ["apds", "gesture", "apds9960", "colour sensor", "color sensor"],
    "radio/fm":         ["radio", "rda5807", "fm ", "tune", "frequency"],
    "servo/pca":        ["servo", "pca9685", "pwm driver"],
    "gamepad/seesaw":   ["gamepad", "seesaw", "joystick"],
    "wifi":             ["wifi", "wi-fi", "access point", "ssid", "network connect"],
    "mqtt":             ["mqtt", "broker", "subscribe", "publish topic"],
    "espnow":           ["espnow", "esp-now", "peer", "esp now"],
    "bluetooth/ble":    ["bluetooth", "ble "],
    "llm":              ["llm", "language model", "model.bin", "inference"],
    "memory":           ["memory", "psram", "heap", "memsum", "ram usage"],
    "automation":       ["automation", "if.*then", "rule", "trigger"],
    "ota/firmware":     ["ota", "firmware", "update.*firmware", "flash"],
    "files/storage":    ["file", "littlefs", "sd card", "upload", "download", "ls "],
    "users/auth":       ["user", "login", "password", "role", "auth"],
    "led/neopixel":     ["led", "neopixel", "ledcolor"],
    "oled":             ["oled", "display", "screen"],
    "debug":            ["debug"],
    "web dashboard":    ["dashboard", "web ui", "web interface", "browser"],
    "cli":              ["cli", "serial", "command line", "terminal"],
    "i2c":              ["i2c", "i2cscan"],
    "negatives/ood":    ["only answer questions about", "only know about"],
}


def main():
    default = Path(__file__).resolve().parent.parent / "training_data" / "hardwareone_rich.txt"
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default

    qa_pairs = []
    with open(path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Q: ") and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if next_line.startswith("A: "):
                qa_pairs.append((line[3:], next_line[3:]))
                i += 2
                continue
        i += 1

    topic_counts = defaultdict(int)
    uncategorised = []

    for q, a in qa_pairs:
        matched = False
        combined = (q + " " + a).lower()
        for topic, keywords in TOPICS.items():
            for kw in keywords:
                if re.search(kw.lower(), combined):
                    topic_counts[topic] += 1
                    matched = True
                    break
        if not matched:
            uncategorised.append(q)

    print(f"=== TOPIC COVERAGE REPORT ({len(qa_pairs)} total QA pairs) ===\n")
    print(f"{'Topic':<22} {'Count':>5}  Status")
    print("-" * 45)

    for topic in sorted(topic_counts, key=topic_counts.get, reverse=True):
        count = topic_counts[topic]
        flag = " *** THIN ***" if count < 20 else ""
        print(f"{topic:<22} {count:5d}{flag}")

    if uncategorised:
        print(f"\n{'(uncategorised)':<22} {len(uncategorised):5d}")
        print("\nSample uncategorised questions:")
        for q in uncategorised[:10]:
            print(f"  Q: {q}")


if __name__ == "__main__":
    main()

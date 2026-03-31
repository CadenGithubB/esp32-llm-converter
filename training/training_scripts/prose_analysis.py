#!/usr/bin/env python3
"""Analyze prose paragraphs in the training corpus.

Reports:
  - Total prose paragraph count and word lengths
  - Any prose exceeding the token danger zone (60+ words ~ 75+ tokens)
  - Prose vs QA ratio
  - Topics covered by prose vs missing from prose

Prose paragraphs are lines that don't start with Q: or A: and aren't blank.

Usage:
    python training_scripts/prose_analysis.py [FILE]
"""
import sys
from pathlib import Path


def main():
    default = Path(__file__).resolve().parent.parent / "training_data" / "hardwareone_rich.txt"
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default

    with open(path) as f:
        lines = f.readlines()

    prose = []
    qa_count = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("Q: ") or stripped.startswith("A: "):
            if stripped.startswith("Q: "):
                qa_count += 1
            continue
        # It's prose
        prose.append(stripped)

    if not prose:
        print("No prose paragraphs found.")
        return

    word_counts = [len(p.split()) for p in prose]

    print(f"=== PROSE ANALYSIS ===\n")
    print(f"Prose paragraphs: {len(prose)}")
    print(f"QA pairs: {qa_count}")
    print(f"Prose:QA ratio: 1:{qa_count // max(len(prose), 1)}")
    print(f"Average prose length: {sum(word_counts) / len(word_counts):.1f} words")
    print(f"Shortest: {min(word_counts)} words")
    print(f"Longest: {max(word_counts)} words")

    # Flag long prose (risk of truncation at seq=128)
    long_prose = [(wc, p) for wc, p in zip(word_counts, prose) if wc > 55]
    if long_prose:
        print(f"\n=== PROSE NEAR TOKEN LIMIT (>55 words, ~70+ tokens) ===")
        for wc, p in sorted(long_prose, reverse=True):
            print(f"\n  [{wc} words] {p[:120]}...")
    else:
        print(f"\nAll prose paragraphs are under 55 words. No truncation risk at seq=128.")

    # Check which topics have prose coverage
    topic_keywords = {
        "tof/distance": ["tof", "vl53", "distance"],
        "imu": ["imu", "bno055", "acceleration"],
        "thermal": ["thermal", "mlx90640", "heat"],
        "presence": ["presence", "sths34"],
        "gps": ["gps", "pa1010", "satellite"],
        "rtc": ["rtc", "ds3231", "clock"],
        "apds/gesture": ["apds", "gesture"],
        "radio": ["radio", "rda5807", "fm"],
        "servo": ["servo", "pca9685"],
        "gamepad": ["gamepad", "seesaw", "joystick"],
        "wifi": ["wifi", "access point"],
        "mqtt": ["mqtt", "broker"],
        "espnow": ["espnow", "esp-now"],
        "llm": ["llm", "language model"],
        "memory": ["psram", "heap", "memory"],
    }

    print(f"\n=== PROSE TOPIC COVERAGE ===")
    for topic, keywords in topic_keywords.items():
        found = any(
            any(kw in p.lower() for kw in keywords)
            for p in prose
        )
        status = "covered" if found else "*** MISSING ***"
        print(f"  {topic:<20} {status}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Check that facts are stated consistently across answers.

Looks for specific known facts (chip names, numbers, specs) and ensures
they appear the same way everywhere. Catches things like:
  - "32 by 24 pixel" vs "32 by 24" (missing "pixel")
  - "BNO055" vs "ICM-42688-P" (wrong chip name)
  - "8 MB" vs "8MB" vs "eight megabytes" (inconsistent formatting)

Update FACT_CHECKS as training data evolves.

Usage:
    python training_scripts/check_answer_consistency.py [FILE]
"""
import re
import sys
from pathlib import Path


# Each check: (description, correct_pattern_regex, wrong_patterns_list)
FACT_CHECKS = [
    (
        "IMU chip should be BNO055",
        r"BNO055",
        ["ICM-42688", "ICM42688", "MPU6050", "MPU9250", "LSM6DS"],
    ),
    (
        "Thermal resolution should include 'pixel'",
        r"32 by 24 pixel",
        ["32 by 24 infrared", "32 by 24 heat"],  # missing "pixel"
    ),
    (
        "No BME280 references",
        None,
        ["BME280", "bme280", "openbme", "closebme", "debugbme"],
    ),
    (
        "Presence sensor should be STHS34PF80",
        r"STHS34PF80",
        ["STHS34PF8[^0]"],  # typos
    ),
    (
        "ToF sensor should be VL53L4CX",
        r"VL53L4CX",
        ["VL53L1X", "VL53L0X", "VL6180"],
    ),
    (
        "GPS should be PA1010D",
        r"PA1010D",
        ["NEO-6M", "NEO6M", "UBLOX"],
    ),
    (
        "Help command phrasing should be consistent",
        r"Type help to see available commands",
        ["Type help for a list of commands"],
    ),
    (
        "'followed by' should have been replaced with 'then'/'with'",
        None,
        ["followed by the", "followed by a"],
    ),
]


def main():
    default = Path(__file__).resolve().parent.parent / "training_data" / "hardwareone_rich.txt"
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default

    with open(path) as f:
        lines = f.readlines()

    answer_lines = [(i + 1, line.strip()) for i, line in enumerate(lines)
                    if line.strip().startswith("A: ")]

    issues_found = 0

    for desc, correct_pat, wrong_pats in FACT_CHECKS:
        hits = []
        for lineno, line in answer_lines:
            for wp in wrong_pats:
                if re.search(wp, line, re.IGNORECASE):
                    hits.append((lineno, line, wp))

        if hits:
            issues_found += len(hits)
            print(f"\n[ISSUE] {desc}")
            for lineno, line, wp in hits[:5]:
                print(f"  Line {lineno}: matched '{wp}'")
                print(f"    {line[:100]}")
            if len(hits) > 5:
                print(f"  ... and {len(hits) - 5} more")
        else:
            print(f"[OK] {desc}")

    if issues_found == 0:
        print(f"\nAll consistency checks passed.")
    else:
        print(f"\n{issues_found} issues found across all checks.")


if __name__ == "__main__":
    main()

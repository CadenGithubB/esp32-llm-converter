#!/usr/bin/env python3
"""Run all training data quality checks in sequence.

Usage:
    python training_scripts/run_all_checks.py [FILE]

Defaults to hardwareone_rich.txt in the parent directory.
"""
import subprocess
import sys
from pathlib import Path


SCRIPTS = [
    "check_hallucinated_sensors.py",
    "check_answer_consistency.py",
    "answer_frequency_balance.py",
    "find_near_duplicate_answers.py",
    "answer_fluff_analysis.py",
    "topic_coverage_report.py",
    "prose_analysis.py",
]


def main():
    script_dir = Path(__file__).resolve().parent
    default_file = script_dir.parent / "training_data" / "hardwareone_rich.txt"
    target = sys.argv[1] if len(sys.argv) > 1 else str(default_file)

    for script_name in SCRIPTS:
        script_path = script_dir / script_name
        if not script_path.exists():
            print(f"\n{'=' * 60}")
            print(f"SKIP: {script_name} (not found)")
            continue

        print(f"\n{'=' * 60}")
        print(f"RUNNING: {script_name}")
        print(f"{'=' * 60}")

        result = subprocess.run(
            [sys.executable, str(script_path), target],
            capture_output=False,
        )

        if result.returncode != 0:
            print(f"  *** {script_name} exited with code {result.returncode} ***")

    print(f"\n{'=' * 60}")
    print(f"ALL CHECKS COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

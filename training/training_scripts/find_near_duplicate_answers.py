#!/usr/bin/env python3
"""Find near-duplicate answers that differ by only a few words.

These cause the model to flip between two phrasings at inference time,
wasting FFN capacity on redundant patterns. Pairs scoring 75%+ word
overlap should be reviewed and merged into a single canonical answer.

Usage:
    python training_scripts/find_near_duplicate_answers.py [FILE]

Defaults to hardwareone_rich.txt in the same parent directory.
"""
import sys
from itertools import combinations
from pathlib import Path


def word_overlap(a: str, b: str) -> float:
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0
    return len(wa & wb) / min(len(wa), len(wb))


def main():
    default = Path(__file__).resolve().parent.parent / "training_data" / "hardwareone_rich.txt"
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default

    answers = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("A: "):
                answers.append(line[3:])

    unique_answers = list(set(answers))
    print(f"Total answer lines: {len(answers)}, Unique: {len(unique_answers)}")

    near_dupes = []
    for a, b in combinations(unique_answers, 2):
        score = word_overlap(a, b)
        if score >= 0.75 and a != b:
            near_dupes.append((score, a, b))

    near_dupes.sort(reverse=True)

    if not near_dupes:
        print("\nNo near-duplicate answers found (75%+ overlap). Looks clean.")
        return

    print(f"\nFound {len(near_dupes)} near-duplicate pairs (75%+ word overlap):\n")
    for score, a, b in near_dupes:
        print(f"  [{score:.0%}]")
        print(f"  A: {a}")
        print(f"  B: {b}")
        print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Check answer frequency balance across the training corpus.

Reports:
  - Top 30 most-repeated answers with their exact count
  - Distribution histogram (how many answers appear N times)
  - Max/min/median repetition counts
  - Flags any answer appearing 30+ times as potentially dominant

High repetition imbalance causes dominant answers to overwrite rare ones
during training. Ideal max is ~20-25x for a corpus of this size.

Usage:
    python training_scripts/answer_frequency_balance.py [FILE]
"""
import sys
from collections import Counter
from pathlib import Path


def main():
    default = Path(__file__).resolve().parent.parent / "training_data" / "hardwareone_rich.txt"
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default

    answers = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("A: "):
                answers.append(line[3:])

    counts = Counter(answers)
    freq_list = sorted(counts.values(), reverse=True)

    print(f"=== ANSWER FREQUENCY BALANCE ===\n")
    print(f"Total answer lines: {len(answers)}")
    print(f"Unique answers: {len(counts)}")
    print(f"Max repetition: {freq_list[0]}x")
    print(f"Median repetition: {freq_list[len(freq_list) // 2]}x")
    print(f"Min repetition: {freq_list[-1]}x")

    # Top 30 most repeated
    print(f"\n=== TOP 30 MOST REPEATED ANSWERS ===")
    for ans, cnt in counts.most_common(30):
        flag = " *** HIGH ***" if cnt >= 30 else ""
        display = ans[:85] + "..." if len(ans) > 85 else ans
        print(f"  {cnt:4d}x  {display}{flag}")

    # Distribution histogram
    print(f"\n=== FREQUENCY DISTRIBUTION ===")
    dist = Counter(counts.values())
    for freq in sorted(dist.keys()):
        bar = "#" * min(dist[freq], 50)
        print(f"  {freq:3d}x: {dist[freq]:3d} answers  {bar}")


if __name__ == "__main__":
    main()

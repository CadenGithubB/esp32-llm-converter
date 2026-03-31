#!/usr/bin/env python3
"""Analyze answer fluff: repeated phrases, verbose openers, and word budget.

Reports:
  - Most common answer openers (first 4 words)
  - Most common 3-grams and 4-grams across all answers
  - Top 30 answers ranked by total word budget (count x word_count)
  - Percentage of answers starting with "Type "
  - Average answer length

Use this to find boilerplate phrases consuming FFN capacity.

Usage:
    python training_scripts/answer_fluff_analysis.py [FILE]
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
    unique = len(counts)
    total_words = sum(len(a.split()) * c for a, c in counts.items())
    avg_words = sum(len(a.split()) for a in answers) / len(answers)

    # Openers
    openers = Counter()
    for a in answers:
        words = a.split()[:4]
        openers[" ".join(words)] += 1

    print("=== MOST COMMON ANSWER OPENERS (first 4 words) ===")
    for phrase, count in openers.most_common(20):
        print(f"  {count:4d}x  {phrase}")

    # 3-grams
    trigrams = Counter()
    for a in answers:
        words = a.lower().split()
        for i in range(len(words) - 2):
            trigrams[" ".join(words[i:i + 3])] += 1

    print("\n=== MOST COMMON 3-GRAMS ACROSS ALL ANSWERS ===")
    for phrase, count in trigrams.most_common(30):
        print(f"  {count:4d}x  {phrase}")

    # 4-grams
    fourgrams = Counter()
    for a in answers:
        words = a.lower().split()
        for i in range(len(words) - 3):
            fourgrams[" ".join(words[i:i + 4])] += 1

    print("\n=== MOST COMMON 4-GRAMS ACROSS ALL ANSWERS ===")
    for phrase, count in fourgrams.most_common(25):
        print(f"  {count:4d}x  {phrase}")

    # Top answers by word budget
    budget = []
    for ans, cnt in counts.items():
        wc = len(ans.split())
        budget.append((cnt * wc, cnt, wc, ans))
    budget.sort(reverse=True)

    print(f"\n=== TOP 30 ANSWERS BY TOTAL WORD BUDGET (count x words) ===")
    print(f"{'Budget':>6} {'Cnt':>4} {'Wds':>4}  Answer")
    print("-" * 100)
    for b, cnt, wc, ans in budget[:30]:
        display = ans[:90] + "..." if len(ans) > 90 else ans
        print(f"{b:6d} {cnt:4d} {wc:4d}  {display}")

    # Summary
    type_count = sum(1 for a in answers if a.startswith("Type "))
    print(f"\n=== SUMMARY ===")
    print(f"Total answers: {len(answers)}, Unique: {unique}")
    print(f"Total word budget: {total_words:,}")
    print(f"Average answer length: {avg_words:.1f} words")
    print(f"Answers starting with 'Type ': {type_count}/{len(answers)} ({type_count * 100 // len(answers)}%)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Shuffle Q&A paragraphs in the training corpus for mixed-topic training blocks.

When QA pairs are grouped by topic, pack_qa_blocks creates single-topic blocks
and the model never learns to disambiguate confusable topics within a context
window. Shuffling ensures each training block contains a mix of topics.

This modifies the file in-place. Run once after adding new content, or whenever
the topic ordering needs refreshing.

Usage:
    python training_scripts/shuffle_training_data.py hardwareone_rich.txt [--seed 42]

The seed defaults to 42 for reproducibility. Use a different seed to get a
different ordering.
"""
import argparse
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Shuffle training data paragraphs")
    parser.add_argument("file", type=Path, help="Training data file to shuffle")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without modifying file")
    args = parser.parse_args()

    if not args.file.is_file():
        print(f"Error: {args.file} not found")
        return 1

    raw = args.file.read_text(encoding="utf-8")
    paragraphs = [blk.strip() for blk in raw.split("\n\n") if blk.strip()]

    # Count types
    qa_count = sum(1 for p in paragraphs if p.startswith("Q:"))
    prose_count = len(paragraphs) - qa_count

    print(f"File: {args.file}")
    print(f"Paragraphs: {len(paragraphs)} ({qa_count} Q&A, {prose_count} prose)")
    print(f"Seed: {args.seed}")

    # Check if already shuffled (rough heuristic: count topic runs)
    # A "run" is consecutive paragraphs about the same topic keyword
    def topic_runs(paras):
        """Count how many times the topic changes between consecutive paragraphs."""
        changes = 0
        prev_words = set()
        for p in paras:
            words = set(p.lower().split()[:20])  # first 20 words as topic signal
            overlap = len(words & prev_words)
            if prev_words and overlap < 3:
                changes += 1
            prev_words = words
        return changes

    runs_before = topic_runs(paragraphs)

    random.Random(args.seed).shuffle(paragraphs)

    runs_after = topic_runs(paragraphs)

    print(f"Topic transitions: {runs_before} -> {runs_after} (higher = more mixed)")

    if args.dry_run:
        print("(dry run — file not modified)")
        return 0

    # Write back with double-newline separators
    output = "\n\n".join(paragraphs) + "\n"
    args.file.write_text(output, encoding="utf-8")
    print(f"Wrote shuffled file: {args.file} ({len(output):,} bytes)")
    return 0


if __name__ == "__main__":
    exit(main())

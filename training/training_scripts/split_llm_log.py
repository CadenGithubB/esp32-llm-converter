#!/usr/bin/env python3
"""Split a full serial monitor log into per-question files.

When you capture an entire debug session with:
    idf.py monitor 2>&1 | tee test_logs/full_session.txt

Then run all your 'llm ask' questions in sequence, this script splits
the log into one file per question for easier analysis.

It detects question boundaries by looking for 'llm ask' or 'llm generate'
commands in the serial output.

Usage:
    python training_scripts/split_llm_log.py test_logs/full_session.txt [OUTPUT_DIR]

Output: Creates test_logs/split/ with numbered files like 01_what_is_the_tof.txt
"""
import re
import sys
from pathlib import Path


def slugify(text: str, max_len: int = 50) -> str:
    """Turn a question into a safe filename slug."""
    text = text.lower().strip().rstrip("?").strip()
    text = re.sub(r'[^a-z0-9 ]', '', text)
    text = re.sub(r'\s+', '_', text)
    return text[:max_len]


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <log_file> [output_dir]")
        sys.exit(1)

    log_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else log_path.parent / "split"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(log_path) as f:
        lines = f.readlines()

    # Find lines that look like LLM commands
    cmd_pattern = re.compile(r'llm\s+(ask|generate)\s+"([^"]+)"', re.IGNORECASE)

    sections = []
    current_question = None
    current_lines = []

    for line in lines:
        match = cmd_pattern.search(line)
        if match:
            # Save previous section
            if current_question:
                sections.append((current_question, current_lines))
            current_question = match.group(2)
            current_lines = [line]
        elif current_question:
            current_lines.append(line)

    # Save last section
    if current_question:
        sections.append((current_question, current_lines))

    if not sections:
        print("No 'llm ask' or 'llm generate' commands found in log file.")
        sys.exit(1)

    print(f"Found {len(sections)} question(s) in {log_path.name}:\n")

    for i, (question, content) in enumerate(sections, 1):
        slug = slugify(question)
        filename = f"{i:02d}_{slug}.txt"
        filepath = out_dir / filename

        with open(filepath, "w") as f:
            f.write(f"# Question: {question}\n")
            f.write(f"# Lines: {len(content)}\n")
            f.write("---\n")
            f.writelines(content)

        # Count debug lines
        debug_lines = sum(1 for l in content if "[LLM]" in l)
        print(f"  {filename:<55} {len(content):5d} lines ({debug_lines} debug)")

    print(f"\nSplit into {len(sections)} files in {out_dir}/")


if __name__ == "__main__":
    main()

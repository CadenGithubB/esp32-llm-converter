#!/usr/bin/env python3
"""Scan the hardwareone-idf firmware project and list all registered CLI commands.

Parses command registration arrays in .cpp files. Commands are registered as
struct entries with the command string as the first field, e.g.:
  { "opentof", "Start ToF sensor.", false, cmd_tof_start, ... }

Usage:
    python training_scripts/list_firmware_commands.py [firmware_path]

firmware_path defaults to ../../../hardwareone-idf relative to this script,
or pass an explicit path to the repo root.
"""
import argparse
import re
import sys
from pathlib import Path


def find_commands(firmware_root: Path) -> list[str]:
    """Scan all .cpp files under firmware_root and extract command strings."""
    components = firmware_root / "components" / "hardwareone"
    if not components.is_dir():
        sys.exit(f"Cannot find components directory at {components}")

    # Pattern: first field of a command registration struct is a quoted string
    # e.g.  { "opentof", "description", ...
    # Capture any { "word", pattern (command name is first quoted string in braces)
    cmd_pattern = re.compile(r'\{\s*"([a-z][a-z0-9_ ]+)"')

    # Also capture registerCommand("name", ...) style
    reg_pattern = re.compile(r'registerCommand\s*\(\s*"([a-z][a-z0-9_ ]+)"')

    commands: set[str] = set()

    for cpp_file in sorted(components.glob("**/*.cpp")):
        text = cpp_file.read_text(encoding="utf-8", errors="replace")
        for m in cmd_pattern.finditer(text):
            cmd = m.group(1).strip()
            # Filter: real commands are lowercase, no leading spaces, not too long
            if len(cmd) >= 2 and len(cmd) <= 40 and not cmd.startswith(" "):
                commands.add(cmd)
        for m in reg_pattern.finditer(text):
            cmd = m.group(1).strip()
            if len(cmd) >= 2 and len(cmd) <= 40:
                commands.add(cmd)

    return sorted(commands)


def find_training_commands(corpus_path: Path) -> set[str]:
    """Extract commands referenced in training data answers (Type <cmd> patterns)."""
    if not corpus_path.is_file():
        return set()
    raw = corpus_path.read_text(encoding="utf-8", errors="replace")
    commands: set[str] = set()
    for line in raw.splitlines():
        if line.startswith("A: "):
            for m in re.finditer(r'\btype\s+(\S+)', line, re.IGNORECASE):
                cmd = m.group(1).rstrip(".,;:")
                if cmd and cmd.lower() not in {"the", "if", "a", "an"}:
                    commands.add(cmd)
    return commands


def main():
    parser = argparse.ArgumentParser(description="List all HardwareOne firmware CLI commands")
    parser.add_argument(
        "firmware_path",
        nargs="?",
        type=Path,
        default=Path(__file__).parent.parent.parent.parent / "hardwareone-idf",
        help="Path to hardwareone-idf repo root",
    )
    parser.add_argument(
        "--training-file",
        type=Path,
        default=Path(__file__).parent.parent / "training_data" / "hardwareone_rich.txt",
        help="Training corpus to cross-reference (default: hardwareone_rich.txt)",
    )
    parser.add_argument(
        "--in-training-only",
        action="store_true",
        help="Only show commands that appear in training data",
    )
    parser.add_argument(
        "--not-in-training",
        action="store_true",
        help="Only show firmware commands NOT in training data",
    )
    args = parser.parse_args()

    firmware_commands = find_commands(args.firmware_path)
    training_commands = find_training_commands(args.training_file)
    training_lower = {c.lower() for c in training_commands}

    print(f"Firmware root:   {args.firmware_path}")
    print(f"Training file:   {args.training_file}")
    print(f"Firmware commands found: {len(firmware_commands)}")
    print(f"Commands in training answers: {len(training_commands)}")
    print()

    if args.in_training_only:
        print("=== COMMANDS IN TRAINING DATA ===")
        for cmd in sorted(training_commands, key=str.lower):
            print(f"  {cmd}")
    elif args.not_in_training:
        print("=== FIRMWARE COMMANDS NOT IN TRAINING DATA ===")
        for cmd in firmware_commands:
            if cmd.lower() not in training_lower:
                print(f"  {cmd}")
    else:
        print("=== ALL FIRMWARE COMMANDS ===")
        for cmd in firmware_commands:
            in_training = "  [in training]" if cmd.lower() in training_lower else ""
            print(f"  {cmd}{in_training}")

    print()
    print("=== TRAINING COMMANDS NOT MATCHING ANY FIRMWARE COMMAND ===")
    firmware_lower = {c.lower() for c in firmware_commands}
    mismatches = [c for c in sorted(training_commands) if c.lower() not in firmware_lower]
    if mismatches:
        for cmd in mismatches:
            print(f"  {cmd}  <-- not found in firmware")
    else:
        print("  None — all training commands match firmware commands.")


if __name__ == "__main__":
    main()

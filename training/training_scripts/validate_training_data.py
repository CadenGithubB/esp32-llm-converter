#!/usr/bin/env python3
"""Comprehensive validation of HardwareOne training data against the firmware.

Runs a suite of checks and reports issues so you know exactly what needs
updating when either the training corpus or the firmware changes.

Checks performed
----------------
1. CLI command cross-reference
   - All "Type <cmd>" / "type <cmd>" patterns extracted from answers
   - Compared against actual CLI commands registered in the firmware
   - Reports commands in training that don't exist in firmware (fabricated)
   - Reports firmware commands that appear nowhere in training (coverage gaps)

2. Deprecated / renamed command audit
   - Hard-coded list of known bad names and their correct replacements
   - Flags any occurrence in the corpus

3. Corpus quality checks
   - "Hardware One" (two words) — must be "HardwareOne"
   - Broken Q/A pairs (Q without A, or A without Q)
   - Answer duplication: reports any answer appearing more than MAX_COPIES times
   - Q/A count summary

4. Special-token coverage
   - Lists domain tokens from trainer files that are NOT in training data at all
     (wasted vocab slots)
   - Lists domain tokens from training data that are NOT in the trainer special
     tokens list (candidates for addition)

Usage
-----
    python training_scripts/validate_training_data.py [options]

    --corpus FILE        Training corpus (default: training_data/hardwareone_rich.txt)
    --firmware PATH      hardwareone-idf repo root (default: ../../../../hardwareone-idf)
    --trainer FILE       GPU trainer to read special tokens from (default: train_tiny_model_gpu.py)
    --max-copies N       Warn when an answer appears more than N times (default: 15)
    --strict             Exit with code 1 if any issues are found
    --summary-only       Skip per-item details, show counts only
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Known deprecated / renamed commands.
# Key   = bad string that should NOT appear in training answers
# Value = suggested replacement (or None if it should just be removed)
# ---------------------------------------------------------------------------
# Each entry: bad_pattern -> (replacement_hint, use_word_boundary)
# use_word_boundary=True: match only when surrounded by non-word chars (avoids
# substring false-positives like "radioautostart" inside "fmradioautostart").
DEPRECATED_COMMANDS: dict[str, tuple[str | None, bool]] = {
    "openradio":            ("openfmradio",                     True),
    "closeradio":           ("closefmradio",                    True),
    "radioautostart":       ("fmradioautostart",                True),
    "openservo":            ("servo <channel> <angle>",         True),
    "closeservo":           ("(no close command for PCA9685)",  True),
    "servoautostart":       ("(no autostart for servo)",        True),
    "synctime":             ("ntpsync",                         True),
    "memsum":               ("memsample or memreport",          True),
    "Type memory":          ("memsample or memreport",          False),  # phrase check
    "Type automations":     ("Type automationlist",             False),  # command-use only
    "Hardware One":         ("HardwareOne",                     False),
    "ESP-32":               ("ESP32-S3",                        False),
}

# Words that the regex captures but are not real standalone commands —
# they are prefix hints, generic terms, or prose words.
COMMAND_PATTERN_ALLOWLIST = {
    "open",      # used as prefix hint: "Type open plus the sensor name"
    "the",
    "a",
    "an",
    "if",
}

# Pattern to capture standalone "Type X" command references from answer lines.
# Matches "Type word", "type word", "typing word" etc.
TYPE_CMD_RE = re.compile(r'\btype\s+([a-z][a-z0-9_]+)', re.IGNORECASE)

# Pattern for commands registered in firmware struct entries.
STRUCT_CMD_RE = re.compile(r'\{\s*"([a-z][a-z0-9_ ]+)"')
REG_CMD_RE    = re.compile(r'registerCommand\s*\(\s*"([a-z][a-z0-9_ ]+)"')

# Pattern to find special_tokens list entries in Python trainer source.
SPECIAL_TOKEN_RE = re.compile(r'"([A-Za-z][A-Za-z0-9_\-<|>:]+)"')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_corpus_pairs(path: Path) -> list[tuple[str, str]]:
    """Parse Q:/A: blocks; return list of (question, answer) tuples."""
    pairs: list[tuple[str, str]] = []
    pending_q: str | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("Q: "):
            if pending_q is not None:
                pairs.append((pending_q, ""))  # orphaned Q
            pending_q = line[3:].strip()
        elif line.startswith("A: ") or line.startswith("Do: "):
            answer = line[line.index(":")+1:].strip()
            prefix = line[:line.index(":")]  # "A" or "Do"
            if pending_q is not None:
                pairs.append((pending_q, answer, prefix))
                pending_q = None
            else:
                pairs.append(("", answer, prefix))  # orphaned response
    if pending_q is not None:
        pairs.append((pending_q, "", ""))
    return pairs


def load_corpus_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def extract_training_commands(corpus_text: str) -> set[str]:
    """Extract all 'Type X' command names from answer lines."""
    cmds: set[str] = set()
    for line in corpus_text.splitlines():
        if not line.startswith("A: "):
            continue
        for m in TYPE_CMD_RE.finditer(line):
            cmd = m.group(1).lower().rstrip(".,;:")
            if cmd not in COMMAND_PATTERN_ALLOWLIST and len(cmd) >= 2:
                cmds.add(cmd)
    return cmds


def find_firmware_commands(firmware_root: Path) -> set[str]:
    """Scan all .cpp files in components/hardwareone for CLI command names."""
    components = firmware_root / "components" / "hardwareone"
    if not components.is_dir():
        return set()
    cmds: set[str] = set()
    for cpp in sorted(components.glob("**/*.cpp")):
        text = cpp.read_text(encoding="utf-8", errors="replace")
        for m in STRUCT_CMD_RE.finditer(text):
            cmd = m.group(1).strip().lower()
            if 2 <= len(cmd) <= 40 and not cmd.startswith(" "):
                cmds.add(cmd)
        for m in REG_CMD_RE.finditer(text):
            cmd = m.group(1).strip().lower()
            if 2 <= len(cmd) <= 40:
                cmds.add(cmd)
    return cmds


def load_special_tokens(trainer_path: Path) -> list[str]:
    """Extract strings from the special_tokens=[...] block in a trainer."""
    if not trainer_path.is_file():
        return []
    text = trainer_path.read_text(encoding="utf-8", errors="replace")
    # Find the special_tokens=[...] block
    m = re.search(r'special_tokens\s*=\s*\[(.*?)\]', text, re.DOTALL)
    if not m:
        return []
    block = m.group(1)
    tokens = SPECIAL_TOKEN_RE.findall(block)
    # Filter out the infrastructure tokens that aren't domain terms
    infra = {"<|endoftext|>", "<pad>", "<unk>", "Q:", "A:"}
    return [t for t in tokens if t not in infra]


# ---------------------------------------------------------------------------
# Check functions  (each returns list[str] of issue messages)
# ---------------------------------------------------------------------------

def check_command_crossref(
    training_cmds: set[str],
    firmware_cmds: set[str],
    summary_only: bool,
) -> tuple[list[str], list[str]]:
    """Returns (fabricated_issues, gap_issues)."""
    fabricated: list[str] = []
    gaps: list[str] = []

    fw_lower = {c.lower() for c in firmware_cmds}
    tr_lower  = {c.lower() for c in training_cmds}

    for cmd in sorted(training_cmds):
        if cmd.lower() not in fw_lower:
            fabricated.append(f"  ✗  '{cmd}'  — in training answers but NOT in firmware")

    # Coverage gaps: firmware sensor open/close commands not in training
    # Only report open*/close* commands, excluding debug commands (those are
    # developer diagnostics that end-users don't need to know about).
    for cmd in sorted(firmware_cmds):
        if cmd.lower() not in tr_lower:
            cl = cmd.lower()
            is_sensor_cmd = (cl.startswith("open") or cl.startswith("close")) and not cl.startswith("debug")
            is_autostart   = cl.endswith("autostart") and not cl.startswith("debug")
            if is_sensor_cmd or is_autostart:
                gaps.append(f"  ?  '{cmd}'  — firmware command, not in training")

    return fabricated, gaps


def check_deprecated(corpus_text: str, summary_only: bool) -> list[str]:
    """Check for known deprecated strings anywhere in the corpus."""
    issues: list[str] = []
    # Pre-compile word-boundary patterns for efficiency
    patterns = {
        bad: (re.compile(r'\b' + re.escape(bad) + r'\b') if wb else None)
        for bad, (_, wb) in DEPRECATED_COMMANDS.items()
    }
    for line_no, line in enumerate(corpus_text.splitlines(), 1):
        for bad, (replacement, use_wb) in DEPRECATED_COMMANDS.items():
            if use_wb:
                if not patterns[bad].search(line):
                    continue
            else:
                if bad not in line:
                    continue
            repl_str = f"  → use: {replacement}" if replacement else "  → remove"
            if not summary_only:
                issues.append(f"  line {line_no:4d}: [{bad}]{repl_str}")
            else:
                issues.append(bad)
    return issues


def check_corpus_quality(
    corpus_path: Path,
    pairs: list[tuple],
    max_copies: int,
    summary_only: bool,
    firmware_cmds: set[str] | None = None,
) -> list[str]:
    """Check pair integrity, duplication, HardwareOne spelling, and Do: validity."""
    issues: list[str] = []
    corpus_text = corpus_path.read_text(encoding="utf-8", errors="replace")

    # Spelling
    count = corpus_text.count("Hardware One")
    if count:
        issues.append(f"  'Hardware One' (two words) appears {count} times — should be 'HardwareOne'")

    # Broken pairs
    orphan_q = sum(1 for p in pairs if p[1] == "")
    orphan_a = sum(1 for p in pairs if p[0] == "")
    if orphan_q:
        issues.append(f"  {orphan_q} Q: line(s) with no following A:/Do:")
    if orphan_a:
        issues.append(f"  {orphan_a} A:/Do: line(s) with no preceding Q:")

    # Do: pair validation
    do_pairs = [p for p in pairs if len(p) > 2 and p[2] == "Do"]
    a_pairs  = [p for p in pairs if len(p) <= 2 or p[2] != "Do"]
    if do_pairs:
        print(f"  Do: pairs: {len(do_pairs)},  A: pairs: {len(a_pairs)}")
        if firmware_cmds:
            fw_lower = {c.lower() for c in firmware_cmds}
            invalid_do = []
            for q, cmd, _ in do_pairs:
                # The command might have args — check first word
                first_word = cmd.split()[0].lower() if cmd else ""
                if first_word and first_word not in fw_lower:
                    invalid_do.append(f"    '{cmd}' (Q: {q})")
            if invalid_do:
                issues.append(f"  {len(invalid_do)} Do: response(s) reference invalid commands:")
                for msg in invalid_do[:10]:
                    issues.append(msg)

    # Duplication ceiling
    answer_counts = Counter(p[1] for p in pairs if p[1])
    over_limit = {a: n for a, n in answer_counts.items() if n > max_copies}
    if over_limit:
        issues.append(f"  {len(over_limit)} answer(s) appear more than {max_copies} times:")
        if not summary_only:
            for ans, n in sorted(over_limit.items(), key=lambda x: -x[1])[:10]:
                issues.append(f"    ({n}x)  {ans[:80]}")

    return issues


def check_special_tokens(
    trainer_path: Path,
    corpus_text: str,
    summary_only: bool,
) -> tuple[list[str], list[str]]:
    """Returns (wasted_slot_issues, missing_from_trainer_issues)."""
    tokens = load_special_tokens(trainer_path)
    if not tokens:
        return [], [f"  Could not read special_tokens from {trainer_path}"]

    wasted:  list[str] = []
    missing: list[str] = []

    for tok in tokens:
        if tok not in corpus_text:
            wasted.append(f"  '{tok}'  — in trainer special_tokens but NEVER appears in corpus")

    # Extract domain-ish words from training answers that aren't in special tokens
    # (heuristic: all-lowercase 5+ char runs that appear as "Type X")
    training_cmds_raw: set[str] = set()
    for line in corpus_text.splitlines():
        if not line.startswith("A: "):
            continue
        for m in TYPE_CMD_RE.finditer(line):
            cmd = m.group(1).rstrip(".,;:")
            if len(cmd) >= 4 and cmd not in COMMAND_PATTERN_ALLOWLIST:
                training_cmds_raw.add(cmd)

    # Filter to compound-ish tokens only — short common words (≤6 chars) that
    # BPE handles fine are noise; also skip tokens with uppercase (placeholders).
    common_short = {
        "close", "debug", "help", "reboot", "radio", "servo",
        "files", "oled", "open", "battery", "settings",
    }
    token_set = set(tokens)
    for cmd in sorted(training_cmds_raw):
        if cmd in common_short:
            continue
        if not cmd.islower():          # skip placeholders like SENSORautostart
            continue
        if len(cmd) < 5:               # too short to worry about BPE fragmentation
            continue
        if cmd not in token_set and cmd.lower() not in {t.lower() for t in token_set}:
            missing.append(f"  '{cmd}'  — appears in training answers but NOT in special_tokens")

    return wasted, missing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    script_dir = Path(__file__).parent
    training_dir = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Validate HardwareOne training data against firmware",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=training_dir / "training_data" / "hardwareone_rich.txt",
        metavar="FILE",
        help="Primary training corpus (default: training_data/hardwareone_rich.txt)",
    )
    parser.add_argument(
        "--firmware",
        type=Path,
        default=script_dir.parent.parent.parent / "hardwareone-idf",
        metavar="PATH",
        help="hardwareone-idf repo root",
    )
    parser.add_argument(
        "--trainer",
        type=Path,
        default=training_dir / "train_tiny_model_gpu.py",
        metavar="FILE",
        help="Trainer script to read special_tokens from",
    )
    parser.add_argument(
        "--max-copies",
        type=int,
        default=15,
        metavar="N",
        help="Warn when an answer appears more than N times (default: 15)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if any issues are found",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Show counts only, skip per-item detail",
    )
    args = parser.parse_args()

    total_issues = 0

    # -- Load corpus -----------------------------------------------------------
    if not args.corpus.is_file():
        sys.exit(f"Corpus not found: {args.corpus}")

    corpus_text = load_corpus_text(args.corpus)
    pairs       = load_corpus_pairs(args.corpus)
    qa_count    = sum(1 for p in pairs if p[0] and p[1])
    do_count    = sum(1 for p in pairs if len(p) > 2 and p[2] == "Do")
    a_count     = qa_count - do_count

    print("=" * 70)
    print("HardwareOne Training Data Validator")
    print("=" * 70)
    print(f"  Corpus:       {args.corpus}")
    print(f"  Firmware:     {args.firmware}")
    print(f"  Trainer:      {args.trainer}")
    print(f"  Q/A pairs:    {qa_count}  (A: {a_count}, Do: {do_count})")
    print()

    # -- 1. CLI command cross-reference ----------------------------------------
    print("─" * 70)
    print("1. CLI COMMAND CROSS-REFERENCE")
    print("─" * 70)
    training_cmds = extract_training_commands(corpus_text)
    firmware_cmds = find_firmware_commands(args.firmware) if args.firmware.is_dir() else set()

    if not firmware_cmds:
        print("  ⚠  Firmware path not found — skipping command cross-reference")
        print(f"     Pass --firmware to: {args.firmware}")
    else:
        print(f"  Firmware commands found: {len(firmware_cmds)}")
        print(f"  Training answer commands: {len(training_cmds)}")
        fabricated, gaps = check_command_crossref(training_cmds, firmware_cmds, args.summary_only)
        if fabricated:
            print(f"\n  FABRICATED (in training, not in firmware) — {len(fabricated)} issue(s):")
            for msg in fabricated:
                print(msg)
            total_issues += len(fabricated)
        else:
            print("  ✓  No fabricated commands found.")

        if gaps:
            print(f"\n  COVERAGE GAPS (sensor commands not in training) — {len(gaps)} item(s):")
            for msg in gaps:
                print(msg)
        else:
            print("  ✓  All sensor open/close commands appear in training.")
    print()

    # -- 2. Deprecated command audit -------------------------------------------
    print("─" * 70)
    print("2. DEPRECATED / RENAMED COMMAND AUDIT")
    print("─" * 70)
    dep_issues = check_deprecated(corpus_text, args.summary_only)
    if dep_issues:
        if args.summary_only:
            counts = Counter(dep_issues)
            for term, n in counts.most_common():
                print(f"  ✗  '{term}' appears {n} time(s)")
        else:
            for msg in dep_issues[:50]:
                print(msg)
            if len(dep_issues) > 50:
                print(f"  ... and {len(dep_issues) - 50} more")
        total_issues += len(set(dep_issues))
    else:
        print("  ✓  No deprecated command strings found.")

    print()

    # -- 3. Corpus quality -----------------------------------------------------
    print("─" * 70)
    print("3. CORPUS QUALITY CHECKS")
    print("─" * 70)
    q_issues = check_corpus_quality(args.corpus, pairs, args.max_copies, args.summary_only, firmware_cmds or None)
    if q_issues:
        for msg in q_issues:
            print(msg)
        total_issues += len(q_issues)
    else:
        print("  ✓  Corpus quality looks good.")
    print()

    # -- 4. Special-token coverage ---------------------------------------------
    print("─" * 70)
    print("4. SPECIAL-TOKEN COVERAGE")
    print("─" * 70)
    wasted, missing = check_special_tokens(args.trainer, corpus_text, args.summary_only)
    if wasted:
        print(f"  WASTED SLOTS (in special_tokens but never in corpus) — {len(wasted)} item(s):")
        for msg in wasted[:20]:
            print(msg)
        total_issues += len(wasted)
    else:
        print("  ✓  All special tokens appear in corpus.")

    if missing:
        print(f"\n  MISSING FROM TRAINER (in training answers but not special_tokens) — {len(missing)} item(s):")
        for msg in missing[:20]:
            print(msg)
    print()

    # -- Summary ---------------------------------------------------------------
    print("=" * 70)
    if total_issues == 0:
        print("✓  All checks passed — training data looks clean.")
    else:
        print(f"✗  {total_issues} issue(s) found. Review output above.")
    print("=" * 70)

    return 1 if (args.strict and total_issues > 0) else 0


if __name__ == "__main__":
    sys.exit(main())

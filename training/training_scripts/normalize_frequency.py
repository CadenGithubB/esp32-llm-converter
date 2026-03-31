#!/usr/bin/env python3
"""Normalize answer frequencies in the training corpus to ~5x each.

For answers appearing MORE than 5 times: keeps the 5 most diverse questions.
For answers appearing LESS than 5 times: generates paraphrased questions.
Preserves prose paragraphs unchanged.

Usage:
    python training_scripts/normalize_frequency.py [FILE]
"""
import re
import sys
import random
import hashlib
from collections import defaultdict
from pathlib import Path

TARGET = 5

# ---------------------------------------------------------------------------
# Paraphrase generation
# ---------------------------------------------------------------------------

# Templates for factual / descriptive answers (A: lines)
QUESTION_TEMPLATES = [
    # Direct question forms
    "What is {topic}?",
    "Tell me about {topic}",
    "How does {topic} work?",
    "Explain {topic}",
    "Can you describe {topic}?",
    "What does {topic} do?",
    "Give me info on {topic}",
    "I want to know about {topic}",
    "Describe {topic} for me",
    "What can you tell me about {topic}?",
]

# Templates for command-oriented answers ("Type X to ...")
COMMAND_TEMPLATES = [
    "How do I {action}?",
    "What command {action_verb} {action}?",
    "Show me how to {action}",
    "I want to {action}",
    "Command to {action}?",
    "I need to {action}",
    "What's the command for {action_gerund}?",
    "How to {action}?",
    "Can I {action}?",
    "Help me {action}",
    "Way to {action}?",
    "Tell me how to {action}",
]

# Templates for Do: pairs (action commands)
DO_TEMPLATES = [
    "{action_verb} the {noun}",
    "start {noun}",
    "turn on {noun}",
    "launch {noun}",
    "enable {noun}",
    "activate {noun}",
    "open {noun}",
    "run {noun}",
    "begin {noun}",
    "fire up {noun}",
    "I want to {action_verb} {noun}",
    "please {action_verb} {noun}",
    "{action_verb} {noun} now",
    "can you {action_verb} {noun}",
    "go ahead and {action_verb} {noun}",
]

# Map common command prefixes to human-readable actions
COMMAND_ACTION_MAP = {
    "opentof": ("open", "tof", "the tof sensor", "distance measurement"),
    "openimu": ("open", "imu", "the IMU sensor", "motion sensing"),
    "opengps": ("open", "gps", "the GPS module", "GPS"),
    "openpresence": ("open", "presence", "the presence sensor", "presence detection"),
    "openapds": ("open", "apds", "the gesture sensor", "gesture detection"),
    "opengamepad": ("open", "gamepad", "the gamepad", "the gamepad controller"),
    "openfmradio": ("open", "radio", "the FM radio", "the radio"),
    "openservo": ("open", "servo", "the servo controller", "servos"),
    "closetof": ("close", "tof", "the tof sensor", "distance measurement"),
    "closeimu": ("close", "imu", "the IMU sensor", "motion sensing"),
    "closegps": ("close", "gps", "the GPS", "GPS"),
    "closepresence": ("close", "presence", "the presence sensor", "presence detection"),
    "closeapds": ("close", "apds", "the gesture sensor", "gesture detection"),
    "closegamepad": ("close", "gamepad", "the gamepad", "the gamepad controller"),
    "closefmradio": ("close", "radio", "the FM radio", "the radio"),
    "tofread": ("read", "tof", "the tof sensor", "distance data"),
    "presenceread": ("read", "presence", "the presence sensor", "presence data"),
    "imu": ("read", "imu", "the IMU", "motion data"),
    "gps": ("read", "gps", "the GPS", "location data"),
    "gamepad": ("read", "gamepad", "the gamepad", "gamepad input"),
    "reboot": ("reboot", "device", "the device", "the system"),
    "uptime": ("check", "uptime", "device uptime", "how long it's been running"),
    "battery": ("check", "battery", "the battery", "battery status"),
    "wifistatus": ("check", "wifi status", "wifi", "the wifi connection"),
    "i2cscan": ("scan", "i2c", "the I2C bus", "connected sensors"),
    "help": ("show", "help", "the help menu", "available commands"),
}

# Verb synonyms for Do: paraphrases
VERB_SYNONYMS = {
    "open": ["start", "enable", "activate", "turn on", "launch", "initialize", "fire up"],
    "close": ["stop", "disable", "deactivate", "turn off", "shut down", "end"],
    "read": ["get", "check", "show", "display", "fetch", "pull up", "see"],
    "reboot": ["restart", "reset", "reboot", "power cycle"],
    "check": ["show", "display", "get", "see", "what is", "tell me"],
    "scan": ["scan", "search", "find", "detect", "discover", "look for"],
    "show": ["display", "list", "print", "get", "see"],
}


def is_good_question(q: str) -> bool:
    """Reject obviously bad generated questions."""
    q_lower = q.lower().rstrip("?").strip()
    # Too short (less than 3 real words)
    words = q_lower.split()
    if len(words) < 2:
        return False
    # Ends with a function word / stop word
    bad_endings = {"the", "a", "an", "to", "of", "for", "with", "in", "on",
                   "is", "are", "do", "can", "i", "my", "not", "this", "that"}
    if words[-1] in bad_endings:
        return False
    # Contains repeated words suggesting template glitch
    if len(words) != len(set(words)) and len(words) < 5:
        return False
    # Nonsensical "describe/explain" + common word patterns
    if re.match(r"^(explain|describe|can you describe|tell me about)\s+(alive|day|many|only|there|this feature|here|where|much|very|just|also|even)\b", q_lower):
        return False
    # "Can I <single-word-not-a-verb>?"
    m = re.match(r"^can i (\w+)\??$", q_lower)
    if m and m.group(1) in ("wifi", "mqtt", "ble", "ota", "gps", "imu", "tof", "led",
                             "there", "this", "that", "here", "only", "many"):
        return False
    return True


def extract_topic_from_answer(answer: str) -> str:
    """Try to extract a clean, short topic noun phrase from an answer."""
    # "The XYZ is ..." - extract noun + optional descriptor
    m = re.match(r"(?:The |HardwareOne'?s? )?([A-Z][\w-]{1,20}(?:\s+(?:sensor|chip|module|controller|camera|radio))?)\s+(?:is |are |uses |has |provides |detects |measures )", answer)
    if m:
        return m.group(1).strip()
    # "X is a/an/the ..."
    m = re.match(r"([A-Z][\w-]{1,20}) is (?:a |an |the )", answer)
    if m:
        return m.group(1).strip()
    # "Type X to ..."
    m = re.match(r"Type (\w+)", answer)
    if m:
        return m.group(1)
    # Look for capitalized proper nouns (sensor/chip names)
    caps = re.findall(r'\b([A-Z][A-Z0-9]{2,}(?:[-]?\w+)?)\b', answer)
    skip = {"The", "Type", "Use", "HardwareOne", "Hardware", "One", "WiFi", "MQTT",
            "ESP", "LED", "OTA", "USB", "SD", "I2C", "BLE", "JSON", "OR", "AND",
            "IF", "THEN", "TEMP", "NOT", "MHz", "Yes", "No", "You", "It", "They",
            "This", "That", "There", "Check", "Make"}
    caps = [c for c in caps if c not in skip]
    if caps:
        return caps[0]
    # Fallback: look for a descriptive noun phrase
    # "... the X ..." where X is a meaningful noun
    nouns = re.findall(r'the (\w{4,})', answer.lower())
    noun_skip = {"device", "command", "current", "other", "same", "first", "last",
                 "next", "most", "some", "that", "this", "which", "each", "every",
                 "serial", "internal", "available", "following"}
    nouns = [n for n in nouns if n not in noun_skip]
    if nouns:
        return nouns[0]
    # Last resort
    return "this feature"


def extract_action_from_answer(answer: str) -> str:
    """Extract the action/goal from a command-oriented answer."""
    # "Type X to Y"
    m = re.match(r"Type \w[\w\s]* to (.+?)(?:\.|,|$)", answer)
    if m:
        return m.group(1).strip().lower()
    # "Use X to Y"
    m = re.match(r"Use \w[\w\s]* to (.+?)(?:\.|,|$)", answer)
    if m:
        return m.group(1).strip().lower()
    # "There is no X command" / "no X command"
    m = re.search(r'no (\w[\w\s]{2,20}?) command', answer)
    if m:
        return m.group(1).strip().lower()
    # Try to get something from the topic
    topic = extract_topic_from_answer(answer)
    if topic and topic != "this feature":
        return topic.lower()
    # Extract from question-style keywords in the answer
    m = re.search(r'(?:clear|delete|remove|reset|update|change|set|configure|enable|disable) (?:the )?(\w[\w\s]{2,15})', answer.lower())
    if m:
        return m.group(0).strip()
    return "use this feature"


def extract_topic_from_questions(questions: list) -> str:
    """Extract the most common meaningful noun from existing questions."""
    stop = {"what", "how", "do", "does", "did", "i", "the", "a", "an", "is", "to",
            "my", "me", "can", "you", "tell", "about", "are", "in", "on", "for",
            "with", "of", "it", "its", "this", "that", "which", "where", "when",
            "why", "show", "explain", "describe", "use", "get", "need", "want",
            "command", "not", "won", "doesn", "isn", "don", "didn", "has", "have",
            "been", "work", "working", "going", "there", "only", "help", "see",
            "set", "run", "way", "much", "many", "most", "some", "all", "any",
            "start", "stop", "open", "close", "turn", "check", "read", "write"}
    words = []
    for q in questions:
        for w in re.findall(r'\w+', q.lower()):
            if w not in stop and len(w) > 2:
                words.append(w)
    if words:
        from collections import Counter
        common = Counter(words).most_common(3)
        top_word = common[0][0]
        # Reject single words that don't make sense as topics
        if len(top_word) < 3:
            return "this"
        return top_word
    return "this"


def generate_question_for_answer(answer: str, existing_questions: list, n_needed: int) -> list:
    """Generate n_needed new question paraphrases for a given answer."""
    generated = []
    existing_lower = {q.lower().rstrip("?").strip() for q in existing_questions}

    is_command_answer = answer.startswith("Type ") or "command" in answer.lower()
    is_offtopic = ("I can only answer" in answer or "I can only help" in answer
                   or "I have no internet" in answer or "I cannot" in answer
                   or "I run entirely" in answer)

    if is_offtopic:
        # For off-topic rejection answers, generate various off-topic questions
        offtopic_qs = [
            "Write me a poem", "Tell me a joke", "What's the weather?",
            "How tall is the Eiffel Tower?", "What year was Python created?",
            "Help me with my homework", "Calculate 42 times 17",
            "Translate hello to French", "Who won the World Cup?",
            "What is machine learning?", "Can you play music?",
            "Tell me a fun fact", "What's the meaning of life?",
            "Write some code for me", "What's 2 plus 2?",
            "Can you browse the web?", "Search for something",
            "What is quantum computing?", "Explain neural networks",
            "Write a haiku", "Do my taxes", "Book a flight",
            "Order me food", "What time is it in Tokyo?",
            "Who is the president?", "Summarize this article",
            "Debug my Python script", "Make me a website",
            "What's trending on Twitter?", "Read my email",
        ]
        random.shuffle(offtopic_qs)
        for q in offtopic_qs:
            if q.lower().rstrip("?").strip() not in existing_lower:
                generated.append(q)
                existing_lower.add(q.lower().rstrip("?").strip())
                if len(generated) >= n_needed:
                    break
        return generated

    # Prefer question-derived topic (reflects what users actually ask) over answer-derived topic
    q_topic = extract_topic_from_questions(existing_questions)
    a_topic = extract_topic_from_answer(answer)
    # Use question topic if it looks meaningful, otherwise fall back to answer topic
    topic = q_topic if q_topic not in ("this", "") and len(q_topic) > 2 else a_topic

    if is_command_answer:
        action = extract_action_from_answer(answer)
        templates = list(COMMAND_TEMPLATES)
        random.shuffle(templates)
        for tmpl in templates:
            q = tmpl.format(
                action=action,
                action_verb="does" if "{action_verb}" in tmpl else "",
                action_gerund=action + "ing" if not action.endswith("ing") else action,
            )
            q = q.replace("  ", " ").strip()
            if q.lower().rstrip("?").strip() not in existing_lower and is_good_question(q):
                generated.append(q)
                existing_lower.add(q.lower().rstrip("?").strip())
                if len(generated) >= n_needed:
                    break
    else:
        templates = list(QUESTION_TEMPLATES)
        random.shuffle(templates)
        for tmpl in templates:
            q = tmpl.format(topic=topic)
            if q.lower().rstrip("?").strip() not in existing_lower and is_good_question(q):
                generated.append(q)
                existing_lower.add(q.lower().rstrip("?").strip())
                if len(generated) >= n_needed:
                    break

    # If we still need more, generate keyword-based variants using topic and question-derived topic
    q_topic = extract_topic_from_questions(existing_questions)
    topics_to_try = [topic, q_topic] if q_topic != topic else [topic]
    while len(generated) < n_needed:
        extra_templates = []
        for t in topics_to_try:
            extra_templates.extend([
                f"What about {t}?",
                f"Info on {t}?",
                f"Details on {t}?",
                f"I have a question about {t}",
                f"What do I need to know about {t}?",
                f"Help with {t}",
                f"{t}?",
                f"How about {t}?",
                f"What's {t} for?",
                f"Quick question about {t}",
                f"Need help with {t}",
                f"Can you help with {t}?",
                f"I want to understand {t}",
                f"More info about {t}",
            ])
        random.shuffle(extra_templates)
        added_any = False
        for q in extra_templates:
            if q.lower().rstrip("?").strip() not in existing_lower and is_good_question(q):
                generated.append(q)
                existing_lower.add(q.lower().rstrip("?").strip())
                added_any = True
                if len(generated) >= n_needed:
                    break
        if not added_any:
            # Absolute fallback with index
            idx = len(generated)
            q = f"Tell me about {topic} (variant {idx})"
            generated.append(q)
            existing_lower.add(q.lower().strip())
            if len(generated) >= n_needed:
                break

    return generated[:n_needed]


def generate_do_question(command: str, existing_questions: list, n_needed: int) -> list:
    """Generate question paraphrases for a Do: command pair."""
    generated = []
    existing_lower = {q.lower().strip() for q in existing_questions}

    info = COMMAND_ACTION_MAP.get(command.strip())
    if info:
        verb, noun, long_noun, alt_noun = info
        synonyms = VERB_SYNONYMS.get(verb, [verb])
        candidates = []
        for syn in synonyms:
            candidates.append(f"{syn} {long_noun}")
            candidates.append(f"{syn} {noun}")
            candidates.append(f"{syn} {alt_noun}")
            candidates.append(f"I want to {syn} {long_noun}")
            candidates.append(f"please {syn} {noun}")
            candidates.append(f"can you {syn} {long_noun}")
            candidates.append(f"go ahead and {syn} {noun}")
            candidates.append(f"{syn} {noun} now")
            candidates.append(f"let's {syn} {long_noun}")
        random.shuffle(candidates)
        for q in candidates:
            q_clean = q.lower().strip()
            if q_clean not in existing_lower:
                generated.append(q)
                existing_lower.add(q_clean)
                if len(generated) >= n_needed:
                    break
    else:
        # Unknown command - generate generic variants
        cmd_words = re.findall(r'[a-z]+', command.lower())
        base = " ".join(cmd_words) if cmd_words else command
        candidates = [
            f"run {base}", f"execute {base}", f"do {base}",
            f"I want to {base}", f"please {base}", f"can you {base}",
            f"{base} now", f"go ahead and {base}", f"start {base}",
            f"perform {base}", f"trigger {base}", f"initiate {base}",
        ]
        random.shuffle(candidates)
        for q in candidates:
            if q.lower().strip() not in existing_lower:
                generated.append(q)
                existing_lower.add(q.lower().strip())
                if len(generated) >= n_needed:
                    break

    return generated[:n_needed]


def select_diverse_questions(questions: list, n: int) -> list:
    """Select the n most diverse questions from a list by maximizing word-set differences."""
    if len(questions) <= n:
        return list(questions)

    def word_set(q):
        return set(re.findall(r'\w+', q.lower()))

    # Start with the shortest and longest questions for diversity
    sorted_by_len = sorted(questions, key=len)
    selected = [sorted_by_len[0]]
    if len(sorted_by_len) > 1:
        selected.append(sorted_by_len[-1])

    remaining = [q for q in questions if q not in selected]

    while len(selected) < n and remaining:
        # Pick the question with the least overlap with already-selected questions
        selected_words = set()
        for q in selected:
            selected_words |= word_set(q)

        best = None
        best_score = -1
        for q in remaining:
            qw = word_set(q)
            # Score = unique words not in selected set + length diversity bonus
            unique = len(qw - selected_words)
            len_diff = abs(len(q) - sum(len(s) for s in selected) / len(selected))
            score = unique * 10 + len_diff
            if score > best_score:
                best_score = score
                best = q
        if best:
            selected.append(best)
            remaining.remove(best)
        else:
            break

    return selected[:n]


# ---------------------------------------------------------------------------
# Parsing and writing
# ---------------------------------------------------------------------------

def parse_file(path: str) -> list:
    """Parse the training file into blocks.

    Returns a list of dicts:
      {"type": "qa", "question": str, "answer": str, "answer_type": "A"|"Do"}
      {"type": "prose", "text": str}
    """
    blocks = []
    with open(path) as f:
        content = f.read()

    # Split into paragraphs separated by blank lines
    raw_blocks = re.split(r'\n\n+', content.strip())

    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue
        lines = block.split('\n')
        if len(lines) >= 2 and lines[0].startswith('Q: '):
            question = lines[0][3:].strip()
            second = lines[1].strip()
            if second.startswith('A: '):
                blocks.append({
                    "type": "qa",
                    "question": question,
                    "answer": second[3:].strip(),
                    "answer_type": "A",
                })
            elif second.startswith('Do: '):
                blocks.append({
                    "type": "qa",
                    "question": question,
                    "answer": second[4:].strip(),
                    "answer_type": "Do",
                })
            else:
                # Q: line but answer doesn't have A:/Do: prefix - treat as prose
                blocks.append({"type": "prose", "text": block})
        elif len(lines) == 1 and lines[0].startswith('Q: '):
            # Orphan question line - keep as prose
            blocks.append({"type": "prose", "text": block})
        else:
            blocks.append({"type": "prose", "text": block})

    return blocks


def write_file(path: str, blocks: list):
    """Write blocks back to the training file."""
    parts = []
    for b in blocks:
        if b["type"] == "qa":
            prefix = "A" if b["answer_type"] == "A" else "Do"
            parts.append(f"Q: {b['question']}\n{prefix}: {b['answer']}")
        else:
            parts.append(b["text"])
    with open(path, 'w') as f:
        f.write('\n\n'.join(parts) + '\n')


def main():
    default = Path(__file__).resolve().parent.parent / "training_data" / "hardwareone_rich.txt"
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default

    random.seed(42)  # Reproducible

    blocks = parse_file(str(path))

    # Separate prose from QA
    prose_blocks = [b for b in blocks if b["type"] == "prose"]
    qa_blocks = [b for b in blocks if b["type"] == "qa"]

    print(f"Parsed {len(qa_blocks)} Q&A pairs and {len(prose_blocks)} prose blocks")

    # Group by (answer_type, answer_text)
    groups = defaultdict(list)
    for b in qa_blocks:
        key = (b["answer_type"], b["answer"])
        groups[key].append(b["question"])

    unique_answers = len(groups)
    print(f"Unique answers: {unique_answers}")

    # Stats tracking
    trimmed_count = 0
    expanded_count = 0
    trimmed_pairs_removed = 0
    expanded_pairs_added = 0

    # Normalize each group
    new_qa_blocks = []
    for (answer_type, answer), questions in groups.items():
        orig_count = len(questions)

        if orig_count > TARGET:
            # Trim: keep most diverse unique questions
            unique_qs = list(dict.fromkeys(questions))  # deduplicate preserving order
            kept = select_diverse_questions(unique_qs, TARGET)
            trimmed_count += 1
            trimmed_pairs_removed += (orig_count - len(kept))
            # If fewer unique questions than TARGET, generate paraphrases to fill
            if len(kept) < TARGET:
                n_fill = TARGET - len(kept)
                if answer_type == "Do":
                    fill_qs = generate_do_question(answer, kept, n_fill)
                else:
                    fill_qs = generate_question_for_answer(answer, kept, n_fill)
                kept.extend(fill_qs)
                expanded_count += 1
                expanded_pairs_added += len(fill_qs)
            for q in kept:
                new_qa_blocks.append({
                    "type": "qa",
                    "question": q,
                    "answer": answer,
                    "answer_type": answer_type,
                })
        elif orig_count < TARGET:
            # Expand: generate paraphrases
            n_needed = TARGET - orig_count
            if answer_type == "Do":
                new_qs = generate_do_question(answer, questions, n_needed)
            else:
                new_qs = generate_question_for_answer(answer, questions, n_needed)
            expanded_count += 1
            expanded_pairs_added += len(new_qs)
            # Keep all originals
            for q in questions:
                new_qa_blocks.append({
                    "type": "qa",
                    "question": q,
                    "answer": answer,
                    "answer_type": answer_type,
                })
            # Add generated
            for q in new_qs:
                new_qa_blocks.append({
                    "type": "qa",
                    "question": q,
                    "answer": answer,
                    "answer_type": answer_type,
                })
        else:
            # Already at target
            for q in questions:
                new_qa_blocks.append({
                    "type": "qa",
                    "question": q,
                    "answer": answer,
                    "answer_type": answer_type,
                })

    # Shuffle QA blocks
    random.shuffle(new_qa_blocks)

    # Interleave prose back in at roughly their original density
    total_blocks = len(new_qa_blocks) + len(prose_blocks)
    if prose_blocks:
        # Distribute prose evenly
        interval = max(1, len(new_qa_blocks) // (len(prose_blocks) + 1))
        final_blocks = []
        prose_idx = 0
        for i, qa in enumerate(new_qa_blocks):
            if prose_idx < len(prose_blocks) and i > 0 and i % interval == 0:
                final_blocks.append(prose_blocks[prose_idx])
                prose_idx += 1
            final_blocks.append(qa)
        # Append remaining prose
        while prose_idx < len(prose_blocks):
            final_blocks.append(prose_blocks[prose_idx])
            prose_idx += 1
    else:
        final_blocks = new_qa_blocks

    write_file(str(path), final_blocks)

    # Final stats
    final_qa = [b for b in final_blocks if b["type"] == "qa"]
    print(f"\n=== NORMALIZATION RESULTS ===")
    print(f"Original Q&A pairs:    {len(qa_blocks)}")
    print(f"Final Q&A pairs:       {len(final_qa)}")
    print(f"Unique answers:        {unique_answers}")
    print(f"Target frequency:      {TARGET}x")
    print(f"Answers trimmed (>{TARGET}x): {trimmed_count} (removed {trimmed_pairs_removed} pairs)")
    print(f"Answers expanded (<{TARGET}x): {expanded_count} (added {expanded_pairs_added} pairs)")
    already_at_target = unique_answers - trimmed_count - expanded_count
    # Some answers may be counted in both trimmed and expanded (had duplicates needing fill)
    if already_at_target < 0:
        already_at_target = 0
    print(f"Answers already at {TARGET}x: {already_at_target}")
    print(f"Prose blocks preserved: {len(prose_blocks)}")
    print(f"Total blocks written:  {len(final_blocks)}")

    # Verify
    verify_groups = defaultdict(int)
    for b in final_blocks:
        if b["type"] == "qa":
            verify_groups[(b["answer_type"], b["answer"])] += 1
    counts = list(verify_groups.values())
    not_five = [(k, v) for k, v in verify_groups.items() if v != TARGET]
    if not_five:
        print(f"\nWARNING: {len(not_five)} answers not at {TARGET}x:")
        for (atype, ans), cnt in sorted(not_five, key=lambda x: -x[1])[:10]:
            print(f"  {cnt}x [{atype}] {ans[:70]}...")
    else:
        print(f"\nAll {len(verify_groups)} unique answers are at exactly {TARGET}x.")


if __name__ == "__main__":
    main()

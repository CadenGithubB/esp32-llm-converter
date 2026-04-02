#!/usr/bin/env python3
“””
Train the HardwareOne on-device LLM — CPU edition.

CPU trainer for the HW1HelpAgent192_deep model (dim=192, 16 layers, 6 heads,
FFN=512, seq=128, 4K vocab). Slow — use the GPU trainer if possible.

──────────────────────────────────────────────────────────────────────────────
SETUP (run once):
  pip install -r requirements.txt

──────────────────────────────────────────────────────────────────────────────
USAGE:

  # HardwareOne help agent (recommended):
  python train_tiny_model.py \\
      --preset HW1HelpAgent192_deep \\
      --text training_data/hardwareone_rich.txt \\
      --epochs 400 --lr 3e-4 --batch-size 8 \\
      --out ./out_HW1HelpAgent192_deep

  # Parameter count / PSRAM estimate only (no training):
  python train_tiny_model.py --preset HW1HelpAgent192_deep --estimate-only

──────────────────────────────────────────────────────────────────────────────
AFTER TRAINING:

  1) Open index.html in Chrome/Edge (no server needed — open directly from disk)
  2) Drop the output folder onto the converter page
  3) Select INT8 quantization, group size 128 (defaults)
  4) Click Convert → Download → saves model.bin
  5) Copy model.bin to SD card at /sd/llm/ or upload via the web Files tab
  6) From the CLI: llm load /sd/llm/model.bin

  Loss should fall from ~7 to ~0.1–0.3.

──────────────────────────────────────────────────────────────────────────────
PRESET REFERENCE (ESP32-S3 8 MB PSRAM, INT8 + group_size=128):

  HW1HelpAgent192_deep  4K vocab, dim=192, 16 layers, FFN=512  ~7.7 MB  ← USE THIS
  HW1HelpAgent          4K vocab, dim=128, 22 layers, FFN=768  ~7.5 MB
  HW1HelpAgent192       4K vocab, dim=192, 12 layers, FFN=768  ~7.6 MB
  HW1HelpAgent256       4K vocab, dim=256,  8 layers, FFN=768  ~7.8 MB
“””

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

# Prevent HuggingFace libraries from making network calls when using local text files.
# Must be set before any transformers/datasets import.
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Named architectures: vocab_size, n_embd, n_layer, n_head, seq_len (n_inner = 4 * n_embd unless --n-inner)
PRESETS: dict[str, dict[str, int]] = {
    "micro": {"vocab_size": 512, "n_embd": 32, "n_layer": 2, "n_head": 4, "seq_len": 64},
    "tiny": {"vocab_size": 2048, "n_embd": 64, "n_layer": 4, "n_head": 8, "seq_len": 128},
    "small": {"vocab_size": 4096, "n_embd": 128, "n_layer": 6, "n_head": 8, "seq_len": 256},
    "medium": {"vocab_size": 8192, "n_embd": 256, "n_layer": 8, "n_head": 8, "seq_len": 512},
    "large": {"vocab_size": 12288, "n_embd": 384, "n_layer": 12, "n_head": 12, "seq_len": 512},
    "xlarge": {"vocab_size": 16384, "n_embd": 512, "n_layer": 16, "n_head": 16, "seq_len": 1024},
    # ── ESP32-S3 8 MB PSRAM targets (INT8, dim=128) ────────────────────────────
    # All three fit comfortably in 8 MB PSRAM when converted with INT8 + group_size=128.
    # Use ctx=64 on device. Train on TinyStories or similar simple English text.
    # "baseline": 16K vocab / 12 layers — most expressive vocabulary, moderate depth (~6.4 MB)
    "baseline": {"vocab_size": 16384, "n_embd": 128, "n_layer": 12, "n_head": 8, "seq_len": 128},
    # "leaner":   8K vocab / 15 layers — freed embedding RAM reinvested in depth (~6.4 MB)
    "leaner":   {"vocab_size": 8192,  "n_embd": 128, "n_layer": 15, "n_head": 8, "seq_len": 128},
    # "stretch":  8K vocab / 18 layers — maximum depth that fits in 8 MB PSRAM (~7.4 MB)
    "stretch":  {"vocab_size": 8192,  "n_embd": 128, "n_layer": 18, "n_head": 8, "seq_len": 128},
    # "stretch2": 8K vocab / 20 layers / hidden=640 — deeper + wider FFN than stretch (~8.1 MB)
    "stretch2": {"vocab_size": 8192,  "n_embd": 128, "n_layer": 20, "n_head": 8, "seq_len": 128, "n_inner": 640},
    # "narrow":   4K vocab / dim=256 / 8 layers — optimized for specialized domain Q&A (~7.3 MB)
    # head_size=32 (vs 16 in dim=128 models) gives much richer attention. Train on domain-specific
    # text only. Faster inference than stretch (fewer layers, same ops/token).
    "narrow":   {"vocab_size": 4096,  "n_embd": 256, "n_layer": 8,  "n_head": 8, "seq_len": 128, "n_inner": 512},
    # "narrow2":  4K vocab / dim=128 / 20 layers / n_head=4 — deep domain Q&A model.
    # dim=128 frees PSRAM for 20 layers (2.5× more than narrow's 8). n_head=4 gives head_size=32,
    # the same rich per-head attention as narrow. n_inner=640 (5×dim) widens the FFN.
    # PSRAM budget at ctx=64: ~4.7 MB weights + ~1.3 MB KV cache = ~6.9 MB total (~1 MB headroom).
    # Use this when narrow fails to associate answers with the right topic.
    "narrow2":  {"vocab_size": 4096,  "n_embd": 128, "n_layer": 20, "n_head": 4, "seq_len": 128, "n_inner": 640},
    # "narrow3":  4K vocab / dim=128 / 18 layers / n_head=4 / n_inner=768 — wider FFN variant.
    # Trades 2 layers of depth for 20% wider FFN (768 vs 640). Each layer stores more distinct
    # facts in its feed-forward network, reducing answer bleed between similar topics.
    # PSRAM budget at ctx=64: roughly same as narrow2 (~6.9 MB) — wider FFN offsets fewer layers.
    # Recommended over narrow2 for domain Q&A where factual precision matters more than depth.
    "narrow3":  {"vocab_size": 4096,  "n_embd": 128, "n_layer": 18, "n_head": 4, "seq_len": 128, "n_inner": 768},
    # "narrow3_short": 4K vocab / dim=128 / 18 layers / n_head=4 / n_inner=768 / seq_len=64
    # Optimized for single-turn Q&A with reduced context window (64 tokens vs 128).
    # Saves 15-20% PSRAM on KV cache, faster inference, same quality for one-shot Q&A.
    # PSRAM budget at ctx=64: ~4.7 MB weights + ~0.65 MB KV cache = ~5.4 MB total (~2.6 MB headroom).
    # Use this when you don't need multi-turn conversation history.
    "narrow3_short":  {"vocab_size": 4096,  "n_embd": 128, "n_layer": 18, "n_head": 4, "seq_len": 64, "n_inner": 768},
    # "HW1HelpAgent": 4K vocab / dim=128 / 22 layers / n_head=4 / n_inner=768 — Hardware One help agent.
    # 22 layers fits ctx=64 in 8 MB PSRAM. Wide FFN (768, 6×dim) for factual precision.
    # head_size=32 (dim/n_head) for rich attention. Designed for domain Q&A with casual paraphrases.
    "HW1HelpAgent":   {"vocab_size": 4096,  "n_embd": 128, "n_layer": 22, "n_head": 4, "seq_len": 128, "n_inner": 768},
    # "HW1HelpAgent_slim": same as HW1HelpAgent but FFN trimmed 768->720 (~264KB smaller).
    "HW1HelpAgent_slim": {"vocab_size": 4096, "n_embd": 128, "n_layer": 22, "n_head": 4, "seq_len": 128, "n_inner": 720},
    # "HW1HelpAgent192": dim=192 / 12 layers / 6 heads / FFN=768 — middle ground between 128 and 256.
    "HW1HelpAgent192": {"vocab_size": 4096, "n_embd": 192, "n_layer": 12, "n_head": 6, "seq_len": 128, "n_inner": 768},
    # "HW1HelpAgent192_deep": dim=192 / 16 layers / 6 heads / FFN=512 — recommended for Hardware One.
    # FFN=512 (2.67×dim) balances factual storage width with 16 layers of routing depth.
    # Est. PSRAM at ctx=64: ~7688 KB with ~504 KB headroom on 8 MB ESP32-S3.
    "HW1HelpAgent192_deep": {"vocab_size": 3328, "n_embd": 192, "n_layer": 16, "n_head": 6, "seq_len": 128, "n_inner": 512},
    # "HW1HelpAgent256": dim=256 / 8 layers / 8 heads / FFN=768 — wider representation for better topic separation.
    "HW1HelpAgent256": {"vocab_size": 4096, "n_embd": 256, "n_layer": 8, "n_head": 8, "seq_len": 128, "n_inner": 768},
}

# argparse dest → CLI flag (for “only override if user did not pass this flag”)
_ARG_TO_FLAG = {
    "vocab_size": "--vocab-size",
    "n_embd": "--n-embd",
    "n_layer": "--n-layer",
    "n_head": "--n-head",
    "n_inner": "--n-inner",
    "seq_len": "--seq-len",
}


def _argv_provides_flag(flag: str) -> bool:
    for a in sys.argv[1:]:
        if a == flag or a.startswith(flag + "="):
            return True
    return False


def apply_preset(ns: argparse.Namespace) -> None:
    if not ns.preset:
        return
    if ns.preset not in PRESETS:
        sys.exit(f"Unknown preset: {ns.preset}")
    for key, val in PRESETS[ns.preset].items():
        flag = _ARG_TO_FLAG[key]
        if not _argv_provides_flag(flag):
            setattr(ns, key, val)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train GPT-2 for esp32-llm-converter (tiny → xlarge presets)")
    p.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default=None,
        help="Architecture bundle (optional). CLI flags like --n-embd still override.",
    )
    p.add_argument(
        "--estimate-only",
        action="store_true",
        help="Print parameter count for the chosen architecture and exit (no data / no training).",
    )
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument(
        "--text",
        type=Path,
        nargs="+",
        metavar="FILE",
        help="One or more plain-text files (UTF-8). All files are concatenated for training.",
    )
    src.add_argument(
        "--dataset",
        choices=("tiny_stories",),
        help="Built-in dataset name (downloads via Hugging Face datasets)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for checkpoint (required unless --estimate-only)",
    )
    p.add_argument("--vocab-size", type=int, default=2048, help="BPE vocab size (smaller → smaller model.bin)")
    p.add_argument("--n-embd", type=int, default=64, help="Hidden size (dim)")
    p.add_argument("--n-layer", type=int, default=4, help="Transformer blocks")
    p.add_argument("--n-head", type=int, default=8, help="Attention heads (must divide n-embd)")
    p.add_argument("--n-inner", type=int, default=None, help="FFN hidden (default: 4 * n-embd)")
    p.add_argument("--seq-len", type=int, default=128, help="Context length (blocks for LM)")
    p.add_argument("--epochs", type=float, default=400.0,
                   help="Training epochs (default 400 for HW1HelpAgent192_deep on hardwareone_rich.txt)")
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Stop after N optimizer steps (overrides --epochs). Use for smoke runs, e.g. --max-steps 15",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-samples", type=int, default=None, help="Cap training rows (TinyStories)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        metavar="N",
        help="Gradient accumulation steps (useful for large presets + small GPU memory)",
    )
    p.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Trade compute for VRAM (recommended for medium+ on one GPU)",
    )
    return p.parse_args()


def train_bpe_tokenizer(text_paths: list[Path], vocab_size: int, out_dir: Path) -> "GPT2TokenizerFast":
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import GPT2TokenizerFast

    tokenizer_core = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer_core.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[
            # Infrastructure
            "<|endoftext|>", "<pad>", "<unk>", "Q:", "A:", "Do:",
            # Sensor open commands — compound words BPE would fragment
            "opentof", "openimu", "opengps", "openthermal",
            "openpresence", "openapds", "openfmradio", "opengamepad",
            "openrtc", "opencamera", "openmic", "openespnow", "opensr",
            # WiFi commands
            "openwifi", "closewifi", "wifiadd", "wifiscan", "wifistatus",
            "wifilist", "wifipromote",
            # BLE commands
            "openble", "closeble", "bleinfo", "blestatus",
            # MQTT commands
            "openmqtt", "mqttstatus",
            # ESP-NOW commands
            "espnowsend", "espnowsendfile", "espnowstats",
            "espnowpair", "espnowunpair", "espnowdevices",
            "espnowsetname", "espnowbroadcast", "espnowpairsecure",
            "espnowlist",
            # Bond commands
            "bondconnect", "bonddisconnect", "bondstatus",
            # User commands
            "useradd", "userdelete", "userlist",
            "userchangepassword", "userresetpassword",
            # Session commands
            "sessionlist", "sessionrevoke", "pendinglist",
            # Battery and system
            "batterystatus", "savesettings", "lightsleep",
            # OLED commands
            "oledbrightness", "oledmode", "oledstatus", "oledclear",
            # LED commands
            "ledcolor", "ledclear", "ledeffect", "ledbrightness",
            # File commands
            "filecreate", "filedelete", "filerename", "fileview",
            # SD card commands
            "sdinfo", "sdformat",
            # Camera commands
            "cameracapture", "camerasave",
            # Microphone commands
            "micrecord", "micdelete",
            # GPS commands
            "gpsread",
            # FM radio commands
            "fmradiotune", "fmradioseek", "fmradiovolume",
            "fmradiomute", "fmradiounmute", "fmradioread",
            # RTC commands
            "rtcread", "rtcsync",
            # Presence sensor
            "presenceread", "presencestatus",
            # APDS sensor
            "apdscolor", "apdsproximity", "apdsgesture",
            # Servo commands
            "servoprofile", "servocalibrate",
            # LLM commands
            "llmload", "llmunload",
            # Edge Impulse commands
            "eienable", "eidetect", "eicontinuous", "eiconfidence",
            # Speech recognition commands
            "srconfidence", "srautotune", "srcmdslist",
            # Sensor reads and diagnostics
            "thermalread", "gamepadread", "sensorinfo", "i2cscan",
            "memsample", "memreport",
            # Misc
            "ntpsync", "automationlist", "automationadd",
        ],
    )
    tokenizer_core.train([str(p) for p in text_paths], trainer)
    tok_path = out_dir / "tokenizer.json"
    tokenizer_core.save(str(tok_path))

    hf_tok = GPT2TokenizerFast(tokenizer_file=str(tok_path))
    if hf_tok.pad_token is None:
        hf_tok.pad_token = hf_tok.eos_token
    hf_tok.save_pretrained(out_dir)
    return hf_tok


def run_qa_test(model, tokenizer, label: str) -> None:
    """Run generation test with Q&A prompts and print results.

    Stops when the model emits the atomic ``Q:`` token (same heuristic as ESP32 firmware).
    """
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    class StopOnQTokenAfterPrompt(StoppingCriteria):
        def __init__(self, prompt_len: int, q_token_id: int) -> None:
            self.prompt_len = prompt_len
            self.q_token_id = q_token_id

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            if input_ids.shape[1] <= self.prompt_len:
                return False
            return int(input_ids[0, -1].item()) == self.q_token_id

    prompts = [
        "Q: What is WiFi?\nA:",
        "Q: What is ESP-NOW?\nA:",
        "Q: What sensors does Hardware One have?\nA:",
        "Q: What is MQTT?\nA:",
        "Q: Is ESP-NOW the same as WiFi?\nA:",
        "Q: How do I update the firmware?\nA:",
        "Once upon a time",
    ]

    q_enc = tokenizer.encode("Q:", add_special_tokens=False)
    q_token_id = q_enc[0] if q_enc else None

    print()
    print(f"  === {label} ===")
    if q_token_id is not None:
        print(f"  (generation stops if model emits Q: token id={q_token_id}, same as device firmware)")
    model.eval()
    for prompt_text in prompts:
        try:
            device = next(model.parameters()).device
            input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
            prompt_len = int(input_ids.shape[1])
            attn = torch.ones_like(input_ids, dtype=torch.long)
            # Cap new tokens well below seq_len — the prompt already occupies
            # some positions, and n_positions == seq_len == 128.  If total
            # context exceeds n_positions the GPU positional-embedding lookup
            # goes out of bounds and triggers a CUDA device-side assertion that
            # poisons the context for the rest of the process (killing Phase 2).
            safe_max_new = max(16, model.config.n_positions - prompt_len - 4)
            gen_kw: dict = dict(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=safe_max_new,
                do_sample=True,
                temperature=0.5,
                top_p=0.8,
                repetition_penalty=1.3,
                pad_token_id=tokenizer.eos_token_id,
            )
            if q_token_id is not None:
                gen_kw["stopping_criteria"] = StoppingCriteriaList(
                    [StopOnQTokenAfterPrompt(prompt_len, q_token_id)]
                )
            with torch.no_grad():
                output = model.generate(**gen_kw)
            new_ids = output[0, prompt_len:]
            ended_on_q = (
                q_token_id is not None
                and new_ids.numel() > 0
                and int(new_ids[-1].item()) == q_token_id
            )
            to_dec = new_ids[:-1] if ended_on_q else new_ids
            answer = tokenizer.decode(to_dec, skip_special_tokens=False)
            if len(answer) > 280:
                answer = answer[:280] + "..."
            tail = "  [stopped: Q:]" if ended_on_q else ""
            print(f"    {prompt_text}")
            print(f"      -> {answer}{tail}")
            print()
        except Exception as e:
            print(f"    {prompt_text}")
            print(f"      -> ERROR: {e}")
            print()


# ── Filler prefix detection ──────────────────────────────────────────────────
# The firmware strips these phrases from user questions before the model sees
# them.  Training data must not contain them, or the model learns patterns it
# will never encounter at inference time.
_FILLER_PREFIXES = [
    # Polite/indirect frames
    "Can you tell me how to ",    "Could you tell me how to ",
    "Can you show me how to ",    "Could you show me how to ",
    "I was wondering how to ",    "What is the best way to ",
    "How would I go about ",      "How do I go about ",
    "What is the way to ",        "Can you tell me ",
    "Could you tell me ",         "Please tell me ",
    "Do you know how to ",        "I was wondering ",
    "Is there a way to ",         "Is it possible to ",
    "I would like to ",           "Please help me ",
    "Tell me how to ",            "I'd like to ",
    "Do you know ",               "I want to ",
    "I need to ",                 "Could you ",
    "Can you ",                   "Help me ",
    "Tell me ",                   "Please ",
    # "How do I" — verb+object IS the topic
    "How do I use the ",          "How do I use ",
    "How do I check the ",        "How do I check ",
    "How do I see the ",          "How do I see ",
    "How do I set up the ",       "How do I set up ",
    "How do I set the ",          "How do I set ",
    "How do I get the ",          "How do I get ",
    "How do I turn on the ",      "How do I turn off the ",
    "How do I turn on ",          "How do I turn off ",
    "How do I change the ",       "How do I change ",
    "How do I enable the ",       "How do I enable ",
    "How do I disable the ",      "How do I disable ",
    "How do I connect to ",       "How do I connect ",
    "How do I create ",           "How do I send ",
    "How do I add ",              "How do I update ",
    "How do I delete ",           "How do I remove ",
    "How do I read ",             "How do I list ",
    "How do I view ",             "How do I open ",
    "How do I start ",            "How do I stop ",
    "How do I save ",             "How do I find ",
    "How do I configure ",        "How do I measure ",
    "How do I detect ",           "How do I access ",
    "How do I make ",             "How do I scan ",
    "How do I pair ",             "How do I join ",
    "How do I leave ",            "How do I build ",
    "How do I flash ",            "How do I assign ",
    "How do I clear ",            "How do I run ",
    "How do I log ",              "How do I record ",
    "How do I schedule ",         "How do I broadcast ",
    "How do I transfer ",         "How do I switch ",
    "How do I sync ",             "How do I tune ",
    "How do I reset ",            "How do I reboot ",
    "How do I do ",               "How do I ",
    # "How does" / "What is/are/does" — remainder is topic
    "How does the ",              "How does ",
    "What is the ",               "What is a ",
    "What is an ",                "What is ",
    "What are the ",              "What are ",
    "What does the ",             "What does ",
    # Quantity
    "How much ",                  "How many ",
    "How long ",                  "How fast ",
    "How far ",                   "How often ",
    "How accurate ",
]


def _check_filler_prefixes(rows: list[str], source_files: list[str]) -> None:
    """Warn if any Q: lines start with filler phrases the firmware strips."""
    hits: list[tuple[str, str, str]] = []  # (source_hint, original_q, filler)
    for row in rows:
        for line in row.split("\n"):
            if not line.startswith("Q: "):
                continue
            body = line[3:]
            for filler in _FILLER_PREFIXES:
                if body.lower().startswith(filler.lower()):
                    hits.append((", ".join(source_files), line.strip(), filler.strip()))
                    break
    if hits:
        print(f"\n{'='*72}")
        print(f"WARNING: {len(hits)} Q: line(s) contain filler prefixes that the")
        print(f"firmware strips at inference time.  The model will never see these")
        print(f"phrases, so they waste training capacity.  Remove them from your")
        print(f"training data so it matches what the model sees on-device.")
        print(f"{'='*72}")
        for src, q, filler in hits[:20]:  # show first 20
            print(f"  {q}")
            print(f"    -> strip \"{filler}\"")
        if len(hits) > 20:
            print(f"  ... and {len(hits) - 20} more")
        print(f"{'='*72}\n")


def load_text_dataset(args: argparse.Namespace) -> tuple[list[Path], "Dataset"]:
    """Returns (temp_files_to_cleanup_or_empty, hf_dataset)."""
    from datasets import Dataset

    if args.text:
        rows: list[str] = []
        for p in args.text:
            if not p.is_file():
                sys.exit(f"Not a file: {p}")
            raw = p.read_text(encoding="utf-8", errors="replace")
            paragraphs = [blk.strip() for blk in raw.split("\n\n") if blk.strip()]
            rows.extend(paragraphs)
        print(f"Loaded {len(rows)} text paragraphs from {len(args.text)} file(s).")

        # ── Check for filler prefixes in Q: lines ──────────────────────────
        # The firmware strips these phrases before the model sees them, so
        # training data should not contain them — the model would learn
        # patterns it never encounters at inference time.
        _check_filler_prefixes(rows, [str(p) for p in args.text])

        return [], Dataset.from_dict({"text": rows})

    from datasets import load_dataset
    assert args.dataset == "tiny_stories"
    os.environ["HF_DATASETS_OFFLINE"] = "0"
    ds = load_dataset("roneneldan/TinyStories", split="train")
    if args.max_samples:
        n = min(len(ds), args.max_samples)
        ds = ds.select(range(n))
    return [], ds


def run_estimate_only(args: argparse.Namespace) -> None:
    """Print architecture size without tokenizer data or training."""
    try:
        from transformers import GPT2Config, GPT2LMHeadModel
    except ImportError as e:
        sys.exit(f"Missing dependency: {e}\nInstall: pip install torch transformers")

    if args.n_embd % args.n_head != 0:
        sys.exit(f"n-embd ({args.n_embd}) must be divisible by n-head ({args.n_head})")

    n_inner = args.n_inner if args.n_inner is not None else 4 * args.n_embd
    config = GPT2Config(
        vocab_size=args.vocab_size,
        n_positions=args.seq_len,
        n_ctx=args.seq_len,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_inner=n_inner,
        bos_token_id=0,
        eos_token_id=0,
    )
    model = GPT2LMHeadModel(config)
    n_params = sum(p.numel() for p in model.parameters())
    preset_note = f"preset={args.preset}" if args.preset else "custom (no --preset)"
    print("Architecture estimate (vocab_size from CLI; real training uses len(tokenizer) ≈ vocab_size)")
    print(f"  {preset_note}")
    print(f"  n_embd={args.n_embd}  n_layer={args.n_layer}  n_head={args.n_head}  n_inner={n_inner}")
    print(f"  seq_len={args.seq_len}  vocab_size={args.vocab_size}")
    print(f"  Parameters: {n_params:,}  (~{n_params / 1e6:.2f}M)")
    print(f"  Rough FP32 weight bytes (if fully loaded in RAM): ~{n_params * 4 / 1024 / 1024:.0f} MiB")
    print("  On-device budgets are often a few MiB PSRAM — use a smaller preset or INT8 + smaller dim.")


def main() -> None:
    args = parse_args()
    apply_preset(args)
    random.seed(args.seed)

    if args.estimate_only:
        run_estimate_only(args)
        return

    if args.out is None:
        sys.exit("--out is required unless you pass --estimate-only")
    if not args.text and not args.dataset:
        sys.exit("Provide --text PATH or --dataset tiny_stories")

    if args.n_embd % args.n_head != 0:
        sys.exit(f"n-embd ({args.n_embd}) must be divisible by n-head ({args.n_head})")

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from transformers import (
            DataCollatorForLanguageModeling,
            GPT2Config,
            GPT2LMHeadModel,
            Trainer,
            TrainingArguments,
        )
    except ImportError as e:
        sys.exit(f"Missing dependency: {e}\nInstall: pip install torch transformers datasets tokenizers accelerate")

    temp_files: list[Path] = []
    text_paths: list[Path] = []

    if args.text:
        text_paths = [p.resolve() for p in args.text]
        _, ds_raw = load_text_dataset(args)
    elif args.dataset:
        _, hf_train = load_text_dataset(args)
        chunk_path = out_dir / "_train_corpus.txt"
        with chunk_path.open("w", encoding="utf-8") as f:
            for row in hf_train:
                f.write(row["text"].strip() + "\n")
        temp_files.append(chunk_path)
        text_paths = [chunk_path]
        ds_raw = hf_train

    print("Training BPE tokenizer…")
    tokenizer = train_bpe_tokenizer(text_paths, args.vocab_size, out_dir)

    # ── Tokenizer debug info ─────────────────────────────────────────────
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    for special in ["Q:", "A:", "<|endoftext|>", "<pad>", "<unk>"]:
        tok_ids = tokenizer.encode(special, add_special_tokens=False)
        print(f"  '{special}' -> token IDs: {tok_ids}  (atomic={len(tok_ids)==1})")
    sample_qa = "Q: What is WiFi?\nA: WiFi connects to a router."
    sample_ids = tokenizer.encode(sample_qa, add_special_tokens=False)
    print(f"  Sample Q&A tokenization ({len(sample_ids)} tokens):")
    for i, tid in enumerate(sample_ids[:30]):
        piece = tokenizer.decode([tid])
        print(f"    [{i}] {tid} = {repr(piece)}")
    if len(sample_ids) > 30:
        print(f"    ... ({len(sample_ids) - 30} more)")
    print("─" * 60)

    n_inner = args.n_inner if args.n_inner is not None else 4 * args.n_embd
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=args.seq_len,
        n_ctx=args.seq_len,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_inner=n_inner,
        bos_token_id=eos_id,
        eos_token_id=eos_id,
    )
    model = GPT2LMHeadModel(config)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.seq_len,
            padding=False,
        )

    # Pre-compute marker token ids for question masking
    _a_marker_ids = tokenizer.encode("\nA:", add_special_tokens=False)
    _a_marker_len = len(_a_marker_ids)
    _do_marker_ids = tokenizer.encode("\nDo:", add_special_tokens=False)
    _do_marker_len = len(_do_marker_ids)
    _q_marker_ids = tokenizer.encode("\nQ:", add_special_tokens=False)
    _q_marker_len = len(_q_marker_ids)
    _q_start_ids = tokenizer.encode("Q:", add_special_tokens=False)
    _q_start_len = len(_q_start_ids)

    def _build_label_mask(ids: list[int]) -> list[int]:
        """Mask question tokens (-100), keep answer/Do:/prose tokens for training."""
        n = len(ids)
        answer_starts: list[int] = []
        for i in range(n - max(_a_marker_len, _do_marker_len) + 1):
            if ids[i:i + _a_marker_len] == _a_marker_ids:
                answer_starts.append(i + _a_marker_len)
            elif ids[i:i + _do_marker_len] == _do_marker_ids:
                answer_starts.append(i + _do_marker_len)
        if not answer_starts:
            return list(ids)  # prose — train on all tokens
        question_starts: list[int] = []
        if _q_start_len <= n and ids[:_q_start_len] == _q_start_ids:
            question_starts.append(0)
        for i in range(n - _q_marker_len + 1):
            if ids[i:i + _q_marker_len] == _q_marker_ids:
                question_starts.append(i)
        question_starts.sort()
        labels = [-100] * n
        for ans_pos in answer_starts:
            next_q = n
            for qs in question_starts:
                if qs > ans_pos:
                    next_q = qs
                    break
            for j in range(ans_pos, min(next_q, n)):
                labels[j] = ids[j]
        return labels

    if args.text:
        # Q&A boundary-aware mode: each paragraph (one Q: ... A: ... block) is
        # tokenized independently and truncated to seq_len. group_texts() is NOT
        # used, so Q→A pairs are never split across training block boundaries.
        print(f"Mode: Q&A boundary-aware (each paragraph = one training block, truncated to {args.seq_len})")
        tok_ds = ds_raw.map(tokenize, batched=True, remove_columns=["text"])
        tok_ds = tok_ds.filter(lambda ex: len(ex["input_ids"]) > 0)

        # Apply question masking: mask Q tokens, train only on A/Do tokens
        block_size = args.seq_len
        def pack_qa_blocks(examples):
            blocks = []
            labels_list = []
            for ids in examples["input_ids"]:
                if not ids:
                    continue
                entry = list(ids[:block_size]) if len(ids) > block_size else list(ids)
                entry_labels = _build_label_mask(entry)
                pad_len = block_size - len(entry)
                blocks.append(entry + [eos_id] * pad_len)
                labels_list.append(entry_labels + [-100] * pad_len)
            return {"input_ids": blocks, "labels": labels_list}

        lm_ds = tok_ds.map(pack_qa_blocks, batched=True, remove_columns=tok_ds.column_names)

        # ── Q&A boundary debug ────────────────────────────────────────────────
        q_id = tokenizer.convert_tokens_to_ids("Q:")
        a_id = tokenizer.convert_tokens_to_ids("A:")
        print(f"\n[BLOCK DEBUG] Q: token id={q_id}  A: token id={a_id}")
        print(f"[BLOCK DEBUG] Total training blocks: {len(lm_ds)}")

        lengths = [len(ex["input_ids"]) for ex in lm_ds]
        if lengths:
            print(f"[BLOCK DEBUG] Block lengths — min={min(lengths)}  max={max(lengths)}  "
                  f"avg={sum(lengths)/len(lengths):.1f}  "
                  f"truncated_to_{args.seq_len}={sum(1 for l in lengths if l == args.seq_len)}")

        n_has_q  = sum(1 for ex in lm_ds if q_id in ex["input_ids"])
        n_has_a  = sum(1 for ex in lm_ds if a_id in ex["input_ids"])
        n_has_both = sum(1 for ex in lm_ds if q_id in ex["input_ids"] and a_id in ex["input_ids"])
        n_missing_a = sum(1 for ex in lm_ds if q_id in ex["input_ids"] and a_id not in ex["input_ids"])
        print(f"[BLOCK DEBUG] Blocks with Q: token : {n_has_q}/{len(lm_ds)}")
        print(f"[BLOCK DEBUG] Blocks with A: token : {n_has_a}/{len(lm_ds)}")
        print(f"[BLOCK DEBUG] Blocks with both Q:+A: : {n_has_both}/{len(lm_ds)}")
        if n_missing_a:
            print(f"[BLOCK DEBUG] WARNING: {n_missing_a} blocks have Q: but no A: "
                  f"(pair was truncated — increase --seq-len or shorten QA pairs)")

        print(f"\n[BLOCK DEBUG] Sample decoded blocks (first 5):")
        for i, ex in enumerate(lm_ds.select(range(min(5, len(lm_ds))))):
            decoded = tokenizer.decode(ex["input_ids"], skip_special_tokens=False)
            decoded_short = decoded[:200].replace("\n", "\\n")
            ids_preview = ex["input_ids"][:12]
            print(f"  [{i}] len={len(ex['input_ids'])}  ids={ids_preview}...")
            print(f"       text: {repr(decoded_short)}")
        print()
        # ─────────────────────────────────────────────────────────────────────
    else:
        nproc = max(1, min(8, (args.max_samples or 50000) // 1000 + 1))
        tok_ds = ds_raw.map(
            tokenize,
            batched=True,
            remove_columns=ds_raw.column_names,
            num_proc=nproc,
        )

        block_size = args.seq_len

        def group_texts(examples):
            # Dataset mode: concatenate all tokens and slice into fixed blocks.
            # Only used for --dataset (non-Q&A data). Never used with --text.
            all_ids: list[int] = []
            for ids in examples["input_ids"]:
                all_ids.extend(ids)
            total_length = len(all_ids)
            if total_length < block_size:
                return {"input_ids": [], "labels": []}
            total_length = (total_length // block_size) * block_size
            chunks = [all_ids[i : i + block_size] for i in range(0, total_length, block_size)]
            return {"input_ids": chunks, "labels": [c[:] for c in chunks]}

        rm_cols = [c for c in tok_ds.column_names if c != "input_ids"]
        lm_ds = tok_ds.map(
            group_texts,
            batched=True,
            batch_size=10_000,
            remove_columns=rm_cols,
        )
        lm_ds = lm_ds.filter(lambda ex: len(ex["input_ids"]) > 0)
    if len(lm_ds) == 0:
        sys.exit(
            "No training blocks after grouping — use longer text, more TinyStories samples, or lower --seq-len."
        )

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    use_cuda = torch.cuda.is_available()

    # Compute warmup steps: ramp LR from ~0 over the first 6% of training
    _steps_per_epoch = max(1, len(lm_ds) // args.batch_size)
    _total_steps = int(_steps_per_epoch * args.epochs // args.grad_accum)
    _warmup_steps = max(1, int(_total_steps * 0.06))

    ta_kw: dict = dict(
        output_dir=str(out_dir / "trainer_ckpt"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",                 # smooth ease-in/ease-out LR curve
        warmup_steps=_warmup_steps,                # ramp LR from ~0 over first 6% of steps
        weight_decay=0.01,                         # L2 regularization
        logging_steps=20,
        save_steps=10_000,
        save_total_limit=1,
        prediction_loss_only=True,
        fp16=use_cuda,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="none",
    )
    if args.max_steps is not None and args.max_steps > 0:
        ta_kw["max_steps"] = args.max_steps
    training_args = TrainingArguments(**ta_kw)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters  |  batch={args.batch_size}  |  epochs={args.epochs}")
    print(f"Dataset: {len(lm_ds):,} blocks of {args.seq_len} tokens")
    print(f"Params-to-blocks ratio: {n_params / max(1, len(lm_ds)):.0f}:1")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_ds,
        data_collator=collator,
    )

    print("Training…")
    trainer.train()

    # ── Post-training diagnostics ─────────────────────────────────────────────
    print("─" * 60)
    print("POST-TRAINING DIAGNOSTICS")
    print("─" * 60)

    log_history = trainer.state.log_history
    losses = [(entry["step"], entry["loss"]) for entry in log_history if "loss" in entry]
    if losses:
        first_loss = losses[0][1]
        last_loss = losses[-1][1]
        min_loss = min(l for _, l in losses)
        print(f"  Loss: first={first_loss:.4f}  last={last_loss:.4f}  min={min_loss:.4f}  steps={losses[-1][0]}")
        if last_loss > first_loss * 0.95:
            print("  WARNING: Loss barely decreased — model may not have learned.")

    model.eval()
    print("  Key weight stats:")
    for name, param in model.named_parameters():
        if any(k in name for k in ["wte.weight", "wpe.weight", "ln_f.weight", "ln_f.bias",
                                     "h.0.ln_1", "h.0.attn.c_attn", "h.0.mlp.c_fc"]):
            data = param.detach().float().cpu()
            nan_count = int(data.isnan().sum())
            nan_warn = " NAN!" if nan_count > 0 else ""
            print(f"    {name:45s} [{float(data.min()):+.6f}, {float(data.max()):+.6f}] "
                  f"mean={float(data.mean()):+.6f} std={float(data.std()):.6f}{nan_warn}")

    # Domain Q&A generation test
    run_qa_test(model, tokenizer, "Post-training Q&A Test")
    print("─" * 60)

    print(f"Saving to {out_dir} …")
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)

    for p in temp_files:
        try:
            p.unlink()
        except OSError:
            pass

    print("Done. Contents:")
    for f in sorted(out_dir.iterdir()):
        if f.is_file():
            print(f"  {f.name} ({f.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()

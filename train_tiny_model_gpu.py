#!/usr/bin/env python3
"""
Train a GPT-2–compatible LM for the esp32-llm-converter — NVIDIA GPU edition.

This is a GPU-optimized copy of train_tiny_model.py. It auto-detects CUDA and
enables bf16 (Ampere+), torch.compile, and larger batch sizes for maximum throughput.
On a modern GPU (RTX 3060/4060+), expect 5-10x faster than CPU.

──────────────────────────────────────────────────────────────────────────────
SETUP (run once on the GPU machine):

  # 1) Install Python deps (CUDA 12.x):
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  pip install transformers datasets tokenizers accelerate

  # 2) Verify GPU is visible:
  python -c "import torch; print(torch.cuda.get_device_name(0))"

──────────────────────────────────────────────────────────────────────────────
USAGE:

  # ESP32-S3 stretch model (8K vocab, 18 layers) — recommended first run:
  python train_tiny_model_gpu.py --preset stretch \\
      --dataset tiny_stories --max-samples 200000 --epochs 5 \\
      --out ./out_stretch

  # Leaner (8K vocab, 15 layers):
  python train_tiny_model_gpu.py --preset leaner \\
      --dataset tiny_stories --max-samples 200000 --epochs 5 \\
      --out ./out_leaner

  # Baseline (16K vocab, 12 layers):
  python train_tiny_model_gpu.py --preset baseline \\
      --dataset tiny_stories --max-samples 200000 --epochs 5 \\
      --out ./out_baseline

  # Parameter count only (no training):
  python train_tiny_model_gpu.py --preset stretch --estimate-only

  # Custom data:
  python train_tiny_model_gpu.py --preset stretch --text mydata.txt --out ./out_custom

──────────────────────────────────────────────────────────────────────────────
AFTER TRAINING:

  1) Copy the output folder to the machine with the converter
  2) Open index.html in the esp32-llm-converter directory
  3) Drag the output folder into the converter
  4) Select INT8 output format
  5) Download model.bin → upload to ESP32 SD card

ESP32-S3 8 MB PSRAM targets (all use INT8 + group_size=128, ctx=64 on device):
  narrow2  →  4K vocab, dim=128, 20 layers, ~4.7 MB wts + ~1.3 MB KV = ~6.9 MB  ← deep domain Q&A, head_size=32
  narrow   →  4K vocab, dim=256,  8 layers, ~5.1 MB wts + ~1.0 MB KV = ~6.1 MB  ← wide attention domain, head_size=32
  stretch2 →  8K vocab, dim=128, 20 layers, ~5.4 MB wts + ~1.3 MB KV = ~6.7 MB  ← general, wider FFN
  stretch  →  8K vocab, dim=128, 18 layers, ~4.4 MB wts + ~1.1 MB KV = ~5.5 MB  ← general depth
  leaner   →  8K vocab, dim=128, 15 layers, ~3.8 MB wts + ~0.9 MB KV = ~4.7 MB  ← lean general
  baseline → 16K vocab, dim=128, 12 layers, ~4.5 MB wts + ~0.8 MB KV = ~5.3 MB  ← broadest vocab
"""

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

# Named architectures: vocab_size, n_embd, n_layer, n_head, seq_len
PRESETS: dict[str, dict[str, int]] = {
    "micro":    {"vocab_size": 512,   "n_embd": 32,  "n_layer": 2,  "n_head": 4,  "seq_len": 64},
    "tiny":     {"vocab_size": 2048,  "n_embd": 64,  "n_layer": 4,  "n_head": 8,  "seq_len": 128},
    "small":    {"vocab_size": 4096,  "n_embd": 128, "n_layer": 6,  "n_head": 8,  "seq_len": 256},
    "medium":   {"vocab_size": 8192,  "n_embd": 256, "n_layer": 8,  "n_head": 8,  "seq_len": 512},
    "large":    {"vocab_size": 12288, "n_embd": 384, "n_layer": 12, "n_head": 12, "seq_len": 512},
    "xlarge":   {"vocab_size": 16384, "n_embd": 512, "n_layer": 16, "n_head": 16, "seq_len": 1024},
    # ── ESP32-S3 8 MB PSRAM targets (INT8, dim=128) ────────────────────────────
    "baseline": {"vocab_size": 16384, "n_embd": 128, "n_layer": 12, "n_head": 8,  "seq_len": 128},
    "leaner":   {"vocab_size": 8192,  "n_embd": 128, "n_layer": 15, "n_head": 8,  "seq_len": 128},
    "stretch":  {"vocab_size": 8192,  "n_embd": 128, "n_layer": 18, "n_head": 8,  "seq_len": 128},
    # "stretch2": 8K vocab / 20 layers / hidden=640 — deeper + wider FFN than stretch (~8.1 MB INT8)
    "stretch2": {"vocab_size": 8192,  "n_embd": 128, "n_layer": 20, "n_head": 8,  "seq_len": 128, "n_inner": 640},
    # "narrow":   4K vocab / dim=256 / 8 layers — optimized for specialized domain Q&A (~7.3 MB INT8)
    # head_size=32 (vs 16 in dim=128 models) gives much richer attention per token.
    # Train on domain-specific text only. Faster inference than stretch with better per-token quality.
    "narrow":   {"vocab_size": 4096,  "n_embd": 256, "n_layer": 8,  "n_head": 8,  "seq_len": 128, "n_inner": 512},
    # "narrow2":  4K vocab / dim=128 / 20 layers / n_head=4 — deep domain Q&A model.
    # dim=128 frees PSRAM for 20 layers (2.5× more than narrow's 8). n_head=4 gives head_size=32,
    # the same rich per-head attention as narrow. n_inner=640 (5×dim) widens the FFN.
    # PSRAM budget at ctx=64: ~4.7 MB weights + ~1.3 MB KV cache = ~6.9 MB total (~1 MB headroom).
    # Use this when narrow fails to associate answers with the right topic.
    "narrow2":  {"vocab_size": 4096,  "n_embd": 128, "n_layer": 20, "n_head": 4,  "seq_len": 128, "n_inner": 640},
}

_ARG_TO_FLAG = {
    "vocab_size": "--vocab-size",
    "n_embd":     "--n-embd",
    "n_layer":    "--n-layer",
    "n_head":     "--n-head",
    "n_inner":    "--n-inner",
    "seq_len":    "--seq-len",
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


def detect_gpu() -> tuple[bool, bool, str]:
    """Returns (has_cuda, supports_bf16, description)."""
    try:
        import torch
    except ImportError:
        return False, False, "torch not installed"

    if not torch.cuda.is_available():
        return False, False, "no CUDA GPU detected"

    name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

    # bf16 supported on Ampere (sm_80+) — RTX 3000+, A100, H100, etc.
    major = torch.cuda.get_device_properties(0).major
    supports_bf16 = major >= 8

    return True, supports_bf16, f"{name} ({vram_gb:.1f} GB VRAM)"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train GPT-2 for esp32-llm-converter — NVIDIA GPU edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default=None,
        help="Architecture bundle. ESP32-S3 targets: stretch / leaner / baseline",
    )
    p.add_argument(
        "--estimate-only",
        action="store_true",
        help="Print parameter count and GPU info, then exit (no training).",
    )
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--text",    type=Path, nargs="+", metavar="FILE",
                     help="One or more plain-text files (UTF-8). All files are concatenated for training.")
    src.add_argument("--dataset", choices=("tiny_stories",), help="Built-in dataset")
    p.add_argument("--out",       type=Path, default=None, help="Output directory")
    p.add_argument("--vocab-size",type=int,  default=2048)
    p.add_argument("--n-embd",    type=int,  default=64)
    p.add_argument("--n-layer",   type=int,  default=4)
    p.add_argument("--n-head",    type=int,  default=8)
    p.add_argument("--n-inner",   type=int,  default=None)
    p.add_argument("--seq-len",   type=int,  default=128)
    p.add_argument("--epochs",    type=float, default=3.0,
                   help="Training epochs (default 3 — more than CPU default for better quality)")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Stop after N steps (smoke test). Omit for full training.")
    p.add_argument("--batch-size", type=int, default=128,
                   help="Per-GPU batch size (default 128; try 512-2048 on 16GB+ VRAM)")
    p.add_argument("--lr",         type=float, default=5e-4)
    p.add_argument("--max-samples",type=int,   default=None)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--grad-accum", type=int,   default=1, metavar="N",
                   help="Gradient accumulation steps (reduce if OOM)")
    p.add_argument("--gradient-checkpointing", action="store_true",
                   help="Trade compute for VRAM (rarely needed for ESP32-size models)")
    p.add_argument("--no-bf16",    action="store_true",
                   help="Disable bf16 even if GPU supports it (force fp16)")
    p.add_argument("--compile",    action="store_true",
                   help="Enable torch.compile (PyTorch 2.0+ only — ~10-30%% speedup, slower start)")
    p.add_argument("--workers",    type=int, default=4,
                   help="DataLoader worker processes (default 4)")
    p.add_argument("--resume",        type=Path, default=None, metavar="CKPT_DIR",
                   help="Resume training from a checkpoint directory (e.g. ./out_stretch/trainer_ckpt/checkpoint-5000)")
    p.add_argument("--finetune-from", type=Path, default=None, metavar="MODEL_DIR",
                   help="Fine-tune from an existing model directory (loads weights + tokenizer; skips BPE training). "
                        "Use with --text and a lower --lr (e.g. 1e-4). "
                        "Example: --finetune-from ./out_stretch --text hardwareone_qa.txt")
    p.add_argument("--negatives",     type=Path, default=None, metavar="FILE",
                   help="Negative reinforcement text file. When provided, training runs in two phases: "
                        "(1) train on --text files only with test output, (2) add negatives and continue training. "
                        "This lets you verify the model before negatives are applied.")
    p.add_argument("--neg-epochs",    type=float, default=None,
                   help="Epochs for phase 2 (negatives). Defaults to same as --epochs.")
    p.add_argument("--neg-lr",        type=float, default=None,
                   help="Learning rate for phase 2 (negatives). Defaults to --lr * 0.5.")
    p.add_argument("--qa-test-prompts", type=Path, default=None, metavar="FILE",
                   help="File with Q&A test prompts (one Q: per line). Used for domain-specific generation tests.")
    return p.parse_args()


def train_bpe_tokenizer(text_paths: list[Path], vocab_size: int, out_dir: Path) -> "GPT2TokenizerFast":
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import GPT2TokenizerFast

    tokenizer_core = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer_core.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<|endoftext|>", "<pad>", "<unk>", "Q:", "A:"],
    )
    tokenizer_core.train([str(p) for p in text_paths], trainer)
    tok_path = out_dir / "tokenizer.json"
    tokenizer_core.save(str(tok_path))

    hf_tok = GPT2TokenizerFast(tokenizer_file=str(tok_path))
    if hf_tok.pad_token is None:
        hf_tok.pad_token = hf_tok.eos_token
    hf_tok.save_pretrained(out_dir)
    return hf_tok


def run_qa_test(model, tokenizer, device, label: str, prompts: list[str] | None = None) -> None:
    """Run generation test with Q&A prompts and print results.

    Uses the same stop-on-Q: rule as the ESP32 firmware (halt when the model emits the
    atomic ``Q:`` token after the prompt) so console tests resemble on-device behavior.
    """
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    class StopOnQTokenAfterPrompt(StoppingCriteria):
        """Stop when the last generated token is the ``Q:`` special (id matches tokenizer)."""

        def __init__(self, prompt_len: int, q_token_id: int) -> None:
            self.prompt_len = prompt_len
            self.q_token_id = q_token_id

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            if input_ids.shape[1] <= self.prompt_len:
                return False
            return int(input_ids[0, -1].item()) == self.q_token_id

    if prompts is None:
        prompts = [
            "Q: What is WiFi?\nA:",
            "Q: What is ESP-NOW?\nA:",
            "Q: What sensors does Hardware One have?\nA:",
            "Q: What is MQTT?\nA:",
            "Q: What is the BME280?\nA:",
            "Q: Is ESP-NOW the same as WiFi?\nA:",
            "Q: How do I update the firmware?\nA:",
            "Q: What is BLE?\nA:",
            "Once upon a time",
            "The cat sat on",
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
            input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
            prompt_len = int(input_ids.shape[1])
            attn = torch.ones_like(input_ids, dtype=torch.long, device=device)
            gen_kw: dict = dict(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=128,
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


def load_text_dataset(args: argparse.Namespace) -> tuple[list[Path], "Dataset"]:
    from datasets import Dataset

    if args.text:
        # Multiple files: validate, read, split into paragraphs so each becomes
        # a separate dataset row. This prevents truncation from eating the whole file.
        rows: list[str] = []
        for p in args.text:
            if not p.is_file():
                sys.exit(f"Not a file: {p}")
            raw = p.read_text(encoding="utf-8", errors="replace")
            # Split on blank lines so each paragraph is one row; tokenizer then
            # processes each independently and group_texts chunks into seq_len blocks.
            paragraphs = [blk.strip() for blk in raw.split("\n\n") if blk.strip()]
            rows.extend(paragraphs)
        print(f"Loaded {len(rows)} text paragraphs from {len(args.text)} file(s).")
        return [], Dataset.from_dict({"text": rows})

    from datasets import load_dataset
    assert args.dataset == "tiny_stories"
    os.environ["HF_DATASETS_OFFLINE"] = "0"   # re-enable for intentional download
    ds = load_dataset("roneneldan/TinyStories", split="train")
    if args.max_samples:
        n = min(len(ds), args.max_samples)
        ds = ds.select(range(n))
    return [], ds


def run_estimate_only(args: argparse.Namespace) -> None:
    try:
        from transformers import GPT2Config, GPT2LMHeadModel
    except ImportError as e:
        sys.exit(f"Missing dependency: {e}\nInstall: pip install torch transformers")

    if args.n_embd % args.n_head != 0:
        sys.exit(f"n-embd ({args.n_embd}) must be divisible by n-head ({args.n_head})")

    has_cuda, supports_bf16, gpu_desc = detect_gpu()

    print("─" * 60)
    print("GPU INFO")
    print(f"  CUDA available:  {has_cuda}")
    print(f"  GPU:             {gpu_desc}")
    if has_cuda:
        print(f"  bf16 support:    {supports_bf16} ({'Ampere+, will use bf16' if supports_bf16 else 'older GPU, will use fp16'})")
    print("─" * 60)

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
    preset_note = f"preset={args.preset}" if args.preset else "custom"
    print("ARCHITECTURE")
    print(f"  {preset_note}")
    print(f"  n_embd={args.n_embd}  n_layer={args.n_layer}  n_head={args.n_head}  n_inner={n_inner}")
    print(f"  seq_len={args.seq_len}  vocab_size={args.vocab_size}")
    print(f"  Parameters: {n_params:,}  (~{n_params / 1e6:.2f}M)")
    print(f"  FP32 size: ~{n_params * 4 / 1024 / 1024:.0f} MiB  |  INT8 size: ~{n_params / 1024 / 1024:.0f} MiB")
    print("─" * 60)


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

    # ── GPU detection ─────────────────────────────────────────────────────────
    has_cuda, supports_bf16, gpu_desc = detect_gpu()
    print("─" * 60)
    if has_cuda:
        print(f"GPU: {gpu_desc}")
        use_bf16 = supports_bf16 and not args.no_bf16
        use_fp16 = not use_bf16
        print(f"Precision: {'bf16 (Ampere+)' if use_bf16 else 'fp16'}")
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM total: {total_vram_gb:.1f} GB")
    else:
        print("WARNING: No CUDA GPU detected — falling back to CPU.")
        print("Training will be slow. Use the regular train_tiny_model.py for CPU.")
        use_bf16 = False
        use_fp16 = False
        total_vram_gb = 0.0
    print("─" * 60)

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

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

    # ── Tokenizer: load existing or train fresh ────────────────────────────
    if args.finetune_from:
        # Fine-tune path: reuse the tokenizer from the base model.
        # CRITICAL — the token IDs in the weights must match the vocab.
        # Retraining the tokenizer on new data would reassign IDs and break everything.
        from transformers import GPT2TokenizerFast
        ft_dir = args.finetune_from.resolve()
        if not ft_dir.is_dir():
            sys.exit(f"--finetune-from path not found: {ft_dir}")
        print(f"Fine-tuning from: {ft_dir}")
        print("Loading tokenizer from base model (NOT retraining)…")
        tokenizer = GPT2TokenizerFast.from_pretrained(str(ft_dir))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Copy tokenizer files to out_dir so the output is self-contained
        tokenizer.save_pretrained(out_dir)
    else:
        print("Training BPE tokenizer…")
        tokenizer = train_bpe_tokenizer(text_paths, args.vocab_size, out_dir)

    # ── Tokenizer debug info ─────────────────────────────────────────────
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    # Verify special tokens are atomic
    for special in ["Q:", "A:", "<|endoftext|>", "<pad>", "<unk>"]:
        tok_ids = tokenizer.encode(special, add_special_tokens=False)
        print(f"  '{special}' -> token IDs: {tok_ids}  (atomic={len(tok_ids)==1})")
    # Test a sample Q&A prompt to see tokenization
    sample_qa = "Q: What is WiFi?\nA: WiFi connects to a router."
    sample_ids = tokenizer.encode(sample_qa, add_special_tokens=False)
    print(f"  Sample Q&A tokenization ({len(sample_ids)} tokens):")
    for i, tid in enumerate(sample_ids[:30]):
        piece = tokenizer.decode([tid])
        print(f"    [{i}] {tid} = {repr(piece)}")
    if len(sample_ids) > 30:
        print(f"    ... ({len(sample_ids) - 30} more)")
    print("─" * 60)

    # ── Model: load existing or create fresh ──────────────────────────────
    n_inner = args.n_inner if args.n_inner is not None else 4 * args.n_embd
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    if args.finetune_from:
        print("Loading model weights from base model…")
        model = GPT2LMHeadModel.from_pretrained(str(ft_dir))
        # Apply any architecture overrides from preset (usually none when fine-tuning)
        n_inner = model.config.n_inner if model.config.n_inner else 4 * model.config.n_embd
    else:
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

    # ── VRAM usage after model creation ───────────────────────────────────────
    if has_cuda:
        model = model.cuda()
        torch.cuda.synchronize()
        allocated_mb = torch.cuda.memory_allocated() / 1024**2
        reserved_mb  = torch.cuda.memory_reserved()  / 1024**2
        n_params_now = sum(p.numel() for p in model.parameters())
        bytes_per_param = 2 if (use_bf16 or use_fp16) else 4
        model_mb = n_params_now * bytes_per_param / 1024**2
        print(f"Model weights: ~{model_mb:.0f} MB ({bytes_per_param}-byte {'bf16' if use_bf16 else 'fp16' if use_fp16 else 'fp32'})")
        print(f"VRAM after model load: {allocated_mb:.0f} MB allocated / {reserved_mb:.0f} MB reserved / {total_vram_gb*1024:.0f} MB total")
        headroom_mb = total_vram_gb * 1024 - allocated_mb
        # rough guidance: each batch of 128-token sequences needs ~headroom/bs MB
        suggested_bs = max(32, min(1024, int(headroom_mb * 0.6 / max(1, model_mb / 128))))
        suggested_bs = (suggested_bs // 32) * 32  # round to multiple of 32
        if suggested_bs > args.batch_size:
            print(f"Tip: {total_vram_gb:.0f} GB VRAM available — try --batch-size {suggested_bs} for faster training")
        print("─" * 60)

    # torch.compile — PyTorch 2.0+ only, ~10-30% speedup after warm-up
    if args.compile:
        if hasattr(torch, "compile"):
            print("Applying torch.compile (first batch will be slow — this is normal)…")
            model = torch.compile(model)
        else:
            print("WARNING: --compile requested but torch.compile not available (need PyTorch 2.0+). Skipping.")

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.seq_len,
            padding=False,
        )

    def tokenize_no_trunc(examples):
        # For local text files: don't truncate — group_texts will chunk into seq_len blocks.
        return tokenizer(
            examples["text"],
            truncation=False,
            padding=False,
        )

    if args.text:
        tok_ds = ds_raw.map(tokenize_no_trunc, batched=True, remove_columns=["text"])
    else:
        nproc = max(1, min(8, (args.max_samples or 200000) // 1000 + 1))
        tok_ds = ds_raw.map(
            tokenize,
            batched=True,
            remove_columns=ds_raw.column_names,
            num_proc=nproc,
        )

    block_size = args.seq_len

    def group_texts(examples):
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
        sys.exit("No training blocks after grouping — use more samples or lower --seq-len.")

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    ta_kw: dict = dict(
        output_dir=str(out_dir / "trainer_ckpt"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=50,
        save_steps=5_000,
        save_total_limit=2,
        prediction_loss_only=True,
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.workers,       # parallel data loading
        dataloader_pin_memory=has_cuda,            # faster CPU→GPU transfers
        torch_compile=args.compile,
        report_to="none",                          # disable wandb/tensorboard by default
        remove_unused_columns=False,               # transformers 5.x compatibility: keep input_ids/labels
    )
    if args.max_steps is not None and args.max_steps > 0:
        ta_kw["max_steps"] = args.max_steps

    training_args = TrainingArguments(**ta_kw)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters  |  batch={args.batch_size}  |  epochs={args.epochs}")
    print(f"Dataset: {len(lm_ds):,} blocks of {args.seq_len} tokens")
    steps_per_epoch = max(1, len(lm_ds) // args.batch_size)
    total_steps = int(steps_per_epoch * args.epochs // args.grad_accum)
    print(f"Estimated steps: ~{total_steps:,}")
    print(f"Params-to-blocks ratio: {n_params / max(1, len(lm_ds)):.0f}:1")
    if args.negatives:
        print(f"Two-phase training: Phase 1 (positive only) -> test -> Phase 2 (+ negatives)")
        print(f"  Negatives file: {args.negatives}")
    phase_label = "Phase 1 (positive only)" if args.negatives else "Training"
    print(f"{phase_label}…")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_ds,
        data_collator=collator,
    )
    resume_ckpt = str(args.resume) if args.resume else None
    if resume_ckpt:
        print(f"Resuming from checkpoint: {resume_ckpt}")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    if has_cuda:
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"Peak VRAM used during training: {peak_mb:.0f} MB / {total_vram_gb*1024:.0f} MB total")

    # ── Post-training diagnostics ─────────────────────────────────────────────
    print("─" * 60)
    print("POST-TRAINING DIAGNOSTICS")
    print("─" * 60)

    # Training loss from trainer state
    log_history = trainer.state.log_history
    losses = [(entry["step"], entry["loss"]) for entry in log_history if "loss" in entry]
    if losses:
        first_loss = losses[0][1]
        last_loss = losses[-1][1]
        min_loss = min(l for _, l in losses)
        print(f"  Loss: first={first_loss:.4f}  last={last_loss:.4f}  min={min_loss:.4f}  steps={losses[-1][0]}")
        if last_loss > first_loss * 0.95:
            print("  WARNING: Loss barely decreased — model may not have learned. Try more steps/data.")
        elif last_loss < 0.5:
            print("  Loss < 0.5 — model is fitting well. Check for overfitting if dataset is small.")

    # Weight statistics per tensor type
    model.eval()
    if hasattr(model, '_orig_mod'):
        inspect_model = model._orig_mod  # unwrap torch.compile
    else:
        inspect_model = model

    print()
    print("  Weight statistics (compare with converter SPOT values):")
    weight_stats = {}
    for name, param in inspect_model.named_parameters():
        data = param.detach().float().cpu()
        stats = {
            "shape": list(data.shape),
            "min": float(data.min()),
            "max": float(data.max()),
            "mean": float(data.mean()),
            "std": float(data.std()),
            "nan": int(data.isnan().sum()),
            "zero_frac": float((data.abs() < 1e-8).sum()) / data.numel(),
        }
        weight_stats[name] = stats

    # Print summary for key tensors
    key_tensors = [
        ("transformer.wte.weight", "embedding"),
        ("transformer.wpe.weight", "pos_embedding"),
        ("transformer.h.0.ln_1.weight", "L0_attn_norm"),
        ("transformer.h.0.ln_1.bias", "L0_attn_norm_bias"),
        ("transformer.h.0.attn.c_attn.weight", "L0_c_attn"),
        ("transformer.h.0.ln_2.weight", "L0_ffn_norm"),
        ("transformer.h.0.ln_2.bias", "L0_ffn_norm_bias"),
        ("transformer.h.0.mlp.c_fc.weight", "L0_mlp_up"),
        ("transformer.ln_f.weight", "final_norm"),
        ("transformer.ln_f.bias", "final_norm_bias"),
    ]
    for tensor_name, label in key_tensors:
        if tensor_name in weight_stats:
            s = weight_stats[tensor_name]
            nan_warn = " NAN!" if s["nan"] > 0 else ""
            zero_warn = f" {s['zero_frac']*100:.0f}%zero" if s["zero_frac"] > 0.5 else ""
            print(f"    {label:25s} {str(s['shape']):20s} [{s['min']:+.6f}, {s['max']:+.6f}] "
                  f"mean={s['mean']:+.6f} std={s['std']:.6f}{nan_warn}{zero_warn}")

    # Check for dead/degenerate layers
    print()
    n_layer = inspect_model.config.n_layer
    for l_idx in range(n_layer):
        ln1_name = f"transformer.h.{l_idx}.ln_1.weight"
        ln2_name = f"transformer.h.{l_idx}.ln_2.weight"
        attn_name = f"transformer.h.{l_idx}.attn.c_attn.weight"
        mlp_name = f"transformer.h.{l_idx}.mlp.c_fc.weight"

        issues = []
        for tname, label in [(attn_name, "attn"), (mlp_name, "mlp")]:
            if tname in weight_stats:
                s = weight_stats[tname]
                if s["nan"] > 0:
                    issues.append(f"{label} has NaN")
                if s["std"] < 1e-6:
                    issues.append(f"{label} near-zero (std={s['std']:.2e})")
                if s["zero_frac"] > 0.9:
                    issues.append(f"{label} {s['zero_frac']*100:.0f}% dead")
        if issues:
            print(f"  WARNING Layer {l_idx}: {', '.join(issues)}")

    # Domain Q&A generation test
    device = next(inspect_model.parameters()).device

    # Load custom test prompts if provided
    qa_prompts = None
    if args.qa_test_prompts and args.qa_test_prompts.is_file():
        raw_lines = args.qa_test_prompts.read_text(encoding="utf-8").strip().splitlines()
        qa_prompts = [line.strip() for line in raw_lines if line.strip() and line.strip().startswith("Q:")]
        # Format as "Q: ...\nA:" prompts
        qa_prompts = [f"{q}\nA:" if "\nA:" not in q else q for q in qa_prompts]
        print(f"  Loaded {len(qa_prompts)} custom test prompts from {args.qa_test_prompts}")

    phase1_label = "Phase 1 (positive only)" if args.negatives else "Post-training"
    run_qa_test(inspect_model, tokenizer, device, f"{phase1_label} Q&A Test", qa_prompts)

    print("─" * 60)

    # ══════════════════════════════════════════════════════════════════════
    # Phase 2: Negative reinforcement (optional)
    # ══════════════════════════════════════════════════════════════════════
    if args.negatives:
        neg_path = args.negatives.resolve()
        if not neg_path.is_file():
            print(f"WARNING: Negatives file not found: {neg_path} — skipping phase 2")
        else:
            print("═" * 60)
            print("PHASE 2: ADDING NEGATIVE REINFORCEMENT DATA")
            print("═" * 60)

            # Read negative data and combine with original text data
            neg_raw = neg_path.read_text(encoding="utf-8", errors="replace")
            neg_paragraphs = [blk.strip() for blk in neg_raw.split("\n\n") if blk.strip()]
            print(f"  Negative data: {len(neg_paragraphs)} paragraphs from {neg_path.name}")

            # Rebuild dataset: original text + negatives
            from datasets import Dataset as _Dataset
            combined_rows: list[str] = []
            for p_path in text_paths:
                if p_path.is_file():
                    raw = p_path.read_text(encoding="utf-8", errors="replace")
                    combined_rows.extend([blk.strip() for blk in raw.split("\n\n") if blk.strip()])
            combined_rows.extend(neg_paragraphs)
            print(f"  Combined dataset: {len(combined_rows)} paragraphs (original + negatives)")

            ds_combined = _Dataset.from_dict({"text": combined_rows})
            tok_combined = ds_combined.map(
                lambda ex: tokenizer(ex["text"], truncation=False, padding=False),
                batched=True, remove_columns=["text"],
            )
            rm_cols2 = [c for c in tok_combined.column_names if c != "input_ids"]
            lm_combined = tok_combined.map(group_texts, batched=True, batch_size=10_000, remove_columns=rm_cols2)
            lm_combined = lm_combined.filter(lambda ex: len(ex["input_ids"]) > 0)
            print(f"  Combined blocks: {len(lm_combined):,} of {args.seq_len} tokens")

            neg_epochs = args.neg_epochs if args.neg_epochs is not None else args.epochs
            neg_lr = args.neg_lr if args.neg_lr is not None else args.lr * 0.5
            print(f"  Phase 2 config: epochs={neg_epochs} lr={neg_lr:.1e}")

            ta_kw2 = dict(ta_kw)  # copy phase 1 args
            ta_kw2["output_dir"] = str(out_dir / "trainer_ckpt_phase2")
            ta_kw2["num_train_epochs"] = neg_epochs
            ta_kw2["learning_rate"] = neg_lr
            ta_kw2.pop("max_steps", None)  # no step limit for phase 2

            training_args2 = TrainingArguments(**ta_kw2)
            trainer2 = Trainer(
                model=model,  # continues from phase 1 weights
                args=training_args2,
                train_dataset=lm_combined,
                data_collator=collator,
            )

            print("Phase 2 training…")
            trainer2.train()

            # Phase 2 loss report
            log2 = trainer2.state.log_history
            losses2 = [(e["step"], e["loss"]) for e in log2 if "loss" in e]
            if losses2:
                print(f"  Phase 2 loss: first={losses2[0][1]:.4f}  last={losses2[-1][1]:.4f}  "
                      f"min={min(l for _, l in losses2):.4f}  steps={losses2[-1][0]}")

            # Test after negatives
            run_qa_test(inspect_model, tokenizer, device, "Phase 2 (+ negatives) Q&A Test", qa_prompts)
            print("─" * 60)

    print(f"Saving to {out_dir} …")
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)

    for p in temp_files:
        try:
            p.unlink()
        except OSError:
            pass

    print("─" * 60)
    print("Done. Output files:")
    for f in sorted(out_dir.iterdir()):
        if f.is_file():
            print(f"  {f.name}  ({f.stat().st_size / 1024:.1f} KB)")
    print("─" * 60)
    print("Next steps:")
    print("  1) Copy this folder to the machine with the converter")
    print("  2) Open index.html → drag folder in → select INT8 → download model.bin")
    print("  3) Upload model.bin to ESP32 SD card at /sd/llm/")


if __name__ == "__main__":
    main()

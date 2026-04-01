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
    # "narrow3":  4K vocab / dim=128 / 18 layers / n_head=4 / n_inner=768 — wider FFN variant.
    "narrow3":  {"vocab_size": 4096,  "n_embd": 128, "n_layer": 18, "n_head": 4,  "seq_len": 128, "n_inner": 768},
    # "narrow3_short": same as narrow3 but seq_len=64 for single-turn Q&A.
    "narrow3_short": {"vocab_size": 4096, "n_embd": 128, "n_layer": 18, "n_head": 4, "seq_len": 64, "n_inner": 768},
    # "HW1HelpAgent": 4K vocab / dim=128 / 22 layers / n_head=4 / n_inner=768 — Hardware One help agent.
    # 22 layers fits ctx=64 in 8 MB PSRAM. Wide FFN (768, 6x dim) for factual precision.
    "HW1HelpAgent": {"vocab_size": 4096, "n_embd": 128, "n_layer": 22, "n_head": 4, "seq_len": 128, "n_inner": 768},
    # "HW1HelpAgent_slim": same as HW1HelpAgent but FFN trimmed 768->720 (~264KB smaller).
    # Use when the full 768 model is too tight on PSRAM.
    "HW1HelpAgent_slim": {"vocab_size": 4096, "n_embd": 128, "n_layer": 22, "n_head": 4, "seq_len": 128, "n_inner": 720},
    # "HW1HelpAgent192": dim=192 / 12 layers / 6 heads / FFN=768 — middle ground between 128 and 256.
    # 2.25× representational capacity vs dim=128 while keeping 12 layers of depth.
    # Fits 8 MB PSRAM with ctx=64: ~6142KB weights + 325KB fp + 1152KB kv + 35KB act ≈ 7654KB.
    "HW1HelpAgent192": {"vocab_size": 4096, "n_embd": 192, "n_layer": 12, "n_head": 6, "seq_len": 128, "n_inner": 768},
    # "HW1HelpAgent192_deep": dim=192 / 16 layers / 6 heads / FFN=512 — recommended for Hardware One.
    # FFN=512 (2.67×dim) balances factual storage width with 16 layers of routing depth.
    # Est. PSRAM at ctx=64: ~7688 KB with ~504 KB headroom on 8 MB ESP32-S3.
    "HW1HelpAgent192_deep": {"vocab_size": 4096, "n_embd": 192, "n_layer": 16, "n_head": 6, "seq_len": 128, "n_inner": 512},
    # "HW1HelpAgent256": dim=256 / 8 layers / 8 heads / FFN=768 — wider representation for better topic separation.
    # 4× representational capacity vs dim=128 at the cost of depth (8 vs 22 layers).
    # Fits 8 MB PSRAM with ctx=64: ~6336KB weights + 234KB fp + 1024KB kv + 46KB act ≈ 7640KB.
    "HW1HelpAgent256": {"vocab_size": 4096, "n_embd": 256, "n_layer": 8, "n_head": 8, "seq_len": 128, "n_inner": 768},
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
        special_tokens=[
            # Infrastructure
            "<|endoftext|>", "<pad>", "<unk>", "Q:", "A:", "Do:",
            # Sensor open/close commands — BPE would otherwise split these
            "opentof", "closetof",
            "openimu", "closeimu",
            "opengps", "closegps",
            "openthermal", "closethermal",
            "openpresence", "closepresence",
            "openapds", "closeapds",
            "openfmradio", "closefmradio",
            "opengamepad", "closegamepad",
            "openrtc", "closertc",
            "opencamera", "closecamera",
            "openmic", "closemic",
            "openespnow", "closeespnow",
            "opensr", "closesr",
            # WiFi commands
            "openwifi", "closewifi", "wifiadd", "wifiscan", "wifistatus",
            "wifilist", "wifirm", "wifipromote",
            "wifiautoreconnect", "wifitxpower", "wifigettxpower",
            # BLE commands
            "openble", "closeble",
            "bleinfo", "blesend", "blestatus", "bleautostart",
            # MQTT commands
            "openmqtt", "closemqtt", "mqttstatus", "mqttautostart",
            "mqttHost", "mqttUser", "mqttPassword",
            "mqttSubscribeTopics", "mqttPublishIMU", "mqttPublishThermal",
            # ESP-NOW commands
            "espnowsend", "espnowsendfile", "espnowstats",
            "espnowpair", "espnowunpair", "espnowdevices",
            "espnowsetname", "espnowmode", "espnowremote",
            "espnowbroadcast", "espnowpairsecure", "espnowlist",
            # Bond commands
            "bondconnect", "bonddisconnect", "bondstatus",
            # User commands
            "useradd", "userdelete", "userlist",
            "userchangepassword", "userresetpassword",
            "userapprove", "userdeny", "userpromote", "userdemote",
            "userrequest", "usersync",
            # Session commands
            "sessionlist", "sessionrevoke", "pendinglist",
            # Battery and system
            "batterystatus", "status", "reboot",
            "cpufreq", "temperature", "voltage", "uptime", "time",
            "sleep", "lightsleep",
            "login", "logout", "savesettings", "taskstats",
            # OLED commands
            "oledbrightness", "oledmode", "oledstatus", "oledtext", "oledclear",
            # LED commands
            "ledcolor", "ledclear", "ledeffect", "ledbrightness",
            "ledstartupeffect", "ledstartupcolor", "ledstartupcolor2",
            "ledstartupduration", "ledstartupenabled",
            # File commands
            "files", "filecreate", "filedelete", "filerename", "fileview", "mkdir",
            "fsusage",
            # SD card commands
            "sdinfo", "sdformat", "sdmount", "sdunmount", "sddiag",
            # Camera commands
            "cameracapture", "camerasave", "cameraread", "cameraautostart",
            # Microphone commands
            "micrecord", "micread", "miclevel", "miclist", "micdelete",
            "micgain", "micautostart",
            # GPS commands
            "gpsread", "gpsautostart",
            # FM radio commands
            "fmradiotune", "fmradioseek", "fmradiovolume",
            "fmradiomute", "fmradiounmute", "fmradioread",
            # RTC commands
            "rtcread", "rtcset", "rtcsync", "rtcautostart",
            # Presence sensor
            "presenceread", "presencestatus", "presenceautostart",
            # APDS sensor
            "apdsread", "apdscolor", "apdsproximity", "apdsgesture", "apdsmode",
            # Servo commands
            "servolist", "servoprofile", "servocalibrate",
            # LLM commands
            "llmload", "llmunload", "llmmodels", "llmstatus",
            # Edge Impulse commands
            "eienable", "eidetect", "eifile", "eicontinuous",
            "eiconfidence", "eistatus",
            "eimodellist", "eimodelload", "eimodelinfo", "eimodelunload",
            "eitrackstatus", "eitrackenable", "eitrackclear",
            # G2 glasses commands
            "g2show", "g2scan", "g2init", "g2deinit",
            "g2clear", "g2status", "g2verbose", "g2nav",
            # Speech recognition commands
            "srstatus", "srcmdslist",
            "srconfidence", "sraccept", "srdyngain", "srraw",
            "srautotune", "srtuning",
            "srtuningswgain", "srtuninggain", "srtuningagc",
            "srtuningvad", "srtuningfilters",
            "srsnipon", "srsnipoff", "srsnipstart", "srsnipstop",
            "srsnipstatus", "srsnipconfig",
            "srdebuglevel", "srdebugstats", "srdebugtelem", "srdebugreset",
            # Voice commands
            "voicearm", "voicedisarm", "voicestatus",
            "voicecancel", "voicehelp",
            # Certificate commands
            "certgen", "certinfo",
            # Logging commands
            "log", "loglevel", "autolog",
            # Automation commands
            "automationlist", "automationadd", "automationrun",
            # Sensor and diagnostics
            "sensors", "sensorinfo", "i2cscan",
            "memsample", "memreport",
            "help",
            # Autostart commands
            "imuautostart", "fmradioautostart", "apdsautostart",
            "thermalautostart", "gamepadautostart", "tofautostart",
            # Standalone domain keywords
            "tof", "imu", "mqtt", "apds", "espnow",
            "thermalread", "tofread", "imuread", "gamepadread",
            "ntpsync",
            "debugtof", "debugwifi", "debugespnow",
            # Platform names — hyphens and mixed case cause bad splits
            "HardwareOne", "ESP-NOW", "ESP-IDF", "ESP32-S3",
            # Chip part numbers — BPE fragments these into meaningless pieces
            "VL53L4CX", "BNO055", "MLX90640", "STHS34PF80",
            "PA1010D", "DS3231", "APDS9960", "RDA5807", "PCA9685",
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


def run_qa_test(model, tokenizer, device, label: str, prompts: list[str] | None = None) -> None:
    """Run generation test with Q&A prompts and print results.

    Uses the same stop rules as the ESP32 firmware: halt when the model emits
    ``Q:`` (token 3) or ``A:`` (token 4) after the prompt.  Repetition penalty
    matches the device default (1.5).

    Note: the device also enforces sentence_limit=2 and hard_cap=80 which
    would truncate output further.  This test intentionally omits those limits
    so you can see the full model output for debugging.
    """
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    class StopOnSpecialTokenAfterPrompt(StoppingCriteria):
        """Stop when the model emits Q: (id 3) or A: (id 4) after the prompt.

        Matches the ESP32 firmware behaviour which halts on both tokens to
        prevent runaway generation of additional Q&A pairs.
        """

        def __init__(self, prompt_len: int, stop_ids: list[int]) -> None:
            self.prompt_len = prompt_len
            self.stop_ids = set(stop_ids)

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            if input_ids.shape[1] <= self.prompt_len:
                return False
            return int(input_ids[0, -1].item()) in self.stop_ids

    if prompts is None:
        prompts = [
            "Q: What is WiFi?\nA:",
            "Q: What is ESP-NOW?\nA:",
            "Q: What sensors does Hardware One have?\nA:",
            "Q: What is MQTT?\nA:",
            "Q: What is the presence sensor?\nA:",
            "Q: Is ESP-NOW the same as WiFi?\nA:",
            "Q: How do I update the firmware?\nA:",
            "Q: What is BLE?\nA:",
            "Once upon a time",
            "The cat sat on",
        ]

    q_enc = tokenizer.encode("Q:", add_special_tokens=False)
    a_enc = tokenizer.encode("A:", add_special_tokens=False)
    stop_ids: list[int] = []
    if q_enc:
        stop_ids.append(q_enc[0])
    if a_enc:
        stop_ids.append(a_enc[0])

    print()
    print(f"  === {label} ===")
    if stop_ids:
        print(f"  (generation stops on Q:/A: token ids={stop_ids}, same as device firmware)")
    model.eval()
    for prompt_text in prompts:
        try:
            input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
            prompt_len = int(input_ids.shape[1])
            attn = torch.ones_like(input_ids, dtype=torch.long, device=device)
            # Cap generation so prompt + output never exceeds the position table
            max_positions = getattr(model.config, 'n_positions', 128)
            max_new = max(1, max_positions - prompt_len)
            gen_kw: dict = dict(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=max_new,
                do_sample=False,
                repetition_penalty=1.5,  # match device firmware LLM_DEFAULT_REP_PENALTY
                pad_token_id=tokenizer.eos_token_id,
            )
            if stop_ids:
                gen_kw["stopping_criteria"] = StoppingCriteriaList(
                    [StopOnSpecialTokenAfterPrompt(prompt_len, stop_ids)]
                )
            with torch.no_grad():
                output = model.generate(**gen_kw)
            new_ids = output[0, prompt_len:]
            ended_on_stop = (
                stop_ids
                and new_ids.numel() > 0
                and int(new_ids[-1].item()) in stop_ids
            )
            stop_name = ""
            if ended_on_stop:
                last_id = int(new_ids[-1].item())
                stop_name = "Q:" if (q_enc and last_id == q_enc[0]) else "A:"
            to_dec = new_ids[:-1] if ended_on_stop else new_ids
            answer = tokenizer.decode(to_dec, skip_special_tokens=False)
            if len(answer) > 280:
                answer = answer[:280] + "..."
            tail = f"  [stopped: {stop_name}]" if ended_on_stop else ""
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
            # processes each independently and pack_qa_blocks packs into seq_len blocks.
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
        # For local text files: don't truncate — pack_qa_blocks will pack into seq_len blocks.
        return tokenizer(
            examples["text"],
            truncation=False,
            padding=False,
        )

    # Pre-compute the token ids for "\nA:" and "\nDo:" so we can find the
    # answer/command boundary.  In Q&A paragraphs the format is either
    # "Q: question\nA: answer" or "Q: intent\nDo: command".  We mask the
    # question portion so the model only trains on predicting the response.
    # Prose paragraphs (no Q:/A:/Do: markers) are left fully unmasked.
    _a_marker_ids = tokenizer.encode("\nA:", add_special_tokens=False)
    _a_marker_len = len(_a_marker_ids)
    _do_marker_ids = tokenizer.encode("\nDo:", add_special_tokens=False)
    _do_marker_len = len(_do_marker_ids)

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

    # Pre-compute Q: token id for multi-turn masking
    _q_marker_ids = tokenizer.encode("\nQ:", add_special_tokens=False)
    _q_marker_len = len(_q_marker_ids)
    # Also detect Q: at position 0 (start of block, no leading newline)
    _q_start_ids = tokenizer.encode("Q:", add_special_tokens=False)
    _q_start_len = len(_q_start_ids)

    def _build_label_mask(ids: list[int]) -> list[int]:
        """Build a per-token label mask for Q&A training blocks.

        For single-turn Q&A: mask the question, train on the answer.
        For multi-turn Q&A: mask ALL question portions, train on ALL answers.
        For prose (no markers): train on everything.

        Returns a list of labels where -100 = masked (question/padding),
        and the actual token id = trainable (answer content).
        """
        n = len(ids)

        # Find all answer/command start positions (after \nA: or \nDo:)
        answer_starts: list[int] = []
        for i in range(n - max(_a_marker_len, _do_marker_len) + 1):
            if ids[i:i + _a_marker_len] == _a_marker_ids:
                answer_starts.append(i + _a_marker_len)
            elif ids[i:i + _do_marker_len] == _do_marker_ids:
                answer_starts.append(i + _do_marker_len)

        if not answer_starts:
            return list(ids)  # prose — train on all tokens

        # Find all question start positions (\nQ: within the block)
        question_starts: list[int] = []
        # Check for Q: at the very start of the block (no \n prefix)
        if _q_start_len <= n and ids[:_q_start_len] == _q_start_ids:
            question_starts.append(0)
        # Find all \nQ: markers within the block (subsequent questions)
        for i in range(n - _q_marker_len + 1):
            if ids[i:i + _q_marker_len] == _q_marker_ids:
                question_starts.append(i)  # include the \n as part of question
        question_starts.sort()  # ensure ascending order

        # Build mask: default to masked (-100), then unmask answer regions
        labels = [-100] * n

        for ans_pos in answer_starts:
            # Find the next question start after this answer
            next_q = n  # default: answer runs to end of block
            for qs in question_starts:
                if qs > ans_pos:
                    next_q = qs
                    break
            # Unmask the answer region
            for j in range(ans_pos, min(next_q, n)):
                labels[j] = ids[j]

        return labels

    def pack_qa_blocks(examples):
        """One Q&A pair (or multi-turn block) per training block.

        Each entry gets its own training block so the model never attends
        across unrelated pair boundaries.  Question tokens are masked (-100)
        so the loss only trains on answer prediction.  For multi-turn blocks,
        ALL question portions are masked and ALL answer portions are trained.
        Prose paragraphs have no Q:/A: markers and are trained on fully.
        Padding is also masked.
        """
        blocks = []
        labels_list = []

        _n_empty = 0
        _n_total = len(examples["input_ids"])
        for ids in examples["input_ids"]:
            if not ids:
                _n_empty += 1
                continue
            entry = list(ids[:block_size]) if len(ids) > block_size else list(ids)

            # Build labels: mask all question tokens, keep all answer tokens
            entry_labels = _build_label_mask(entry)

            # Pad single entry to block_size
            pad_len = block_size - len(entry)
            blocks.append(entry + [eos_id] * pad_len)
            labels_list.append(entry_labels + [-100] * pad_len)

        if _n_empty > 0:
            print(f"  pack_qa_blocks: {_n_empty}/{_n_total} inputs had empty input_ids")
        if len(blocks) != _n_total:
            print(f"  pack_qa_blocks: {_n_total} inputs → {len(blocks)} blocks ({_n_total - len(blocks)} dropped)")

        return {"input_ids": blocks, "labels": labels_list}

    # Pre-packing diagnostic: check how many tokenized entries are empty
    _pre_empty = sum(1 for ex in tok_ds if not ex["input_ids"])
    _pre_total = len(tok_ds)
    print(f"Pre-packing: {_pre_total} tokenized paragraphs, {_pre_empty} empty")
    if _pre_empty > 0:
        for i, ex in enumerate(tok_ds):
            if not ex["input_ids"]:
                print(f"  Empty at index {i}")
                if i >= 10:
                    print(f"  ... (showing first 10 of {_pre_empty})")
                    break

    rm_cols = [c for c in tok_ds.column_names if c != "input_ids"]
    lm_ds = tok_ds.map(
        pack_qa_blocks,
        batched=True,
        batch_size=10_000,
        remove_columns=rm_cols,
    )
    lm_ds = lm_ds.filter(lambda ex: len(ex["input_ids"]) > 0)
    if len(lm_ds) == 0:
        sys.exit("No training blocks after packing — use more samples or lower --seq-len.")

    # Diagnostic: report data loss if significant
    _n_input = len(tok_ds)
    _n_output = len(lm_ds)
    if _n_output < _n_input:
        _dropped = _n_input - _n_output
        print(f"  WARNING: {_dropped}/{_n_input} paragraphs dropped during packing "
              f"({_dropped/_n_input*100:.0f}% loss). Check for empty tokenizations.")
        # Sample some dropped entries to help debug
        _empty_count = 0
        for i, ex in enumerate(tok_ds):
            if not ex["input_ids"]:
                _empty_count += 1
                if _empty_count <= 5:
                    print(f"    Empty input_ids at index {i}")

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

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
        weight_decay=0.01,                         # L2 regularization — prevents weight explosion
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
    print(f"Dataset: {len(lm_ds):,} blocks (up to {args.seq_len} tokens, complete Q&A pairs)")
    steps_per_epoch = max(1, len(lm_ds) // args.batch_size)
    total_steps = int(steps_per_epoch * args.epochs // args.grad_accum)
    print(f"Estimated steps: ~{total_steps:,}")
    print(f"Params-to-blocks ratio: {n_params / max(1, len(lm_ds)):.0f}:1")
    print("Training…")

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

    run_qa_test(inspect_model, tokenizer, device, "Post-training Q&A Test", qa_prompts)

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

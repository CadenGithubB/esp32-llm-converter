#!/usr/bin/env bash
# Train one checkpoint per preset into generated_presets/<preset>/ (GPT-2 safetensors + tokenizer).
# Requires: .venv with torch/transformers/datasets, network for TinyStories on first run.
# Short run: --max-steps limits training (smoke / layout test). For real quality, remove --max-steps
# and raise --max_samples / --epochs.
#
# ESP32-S3 8 MB PSRAM targets (all INT8-convertible, dim=128, ctx=64 on device):
#   baseline  — 16K vocab, dim=128, 12 layers (~6.4 MB)   broadest vocabulary, moderate depth
#   leaner    —  8K vocab, dim=128, 15 layers (~6.4 MB)   freed embedding RAM reinvested in depth
#   stretch   —  8K vocab, dim=128, 18 layers (~7.4 MB)   maximum depth that fits 8 MB PSRAM
#   stretch2  —  8K vocab, dim=128, 20 layers, hidden=640 (~8.1 MB)  deeper + wider FFN, pushes PSRAM limit
#   narrow    —  4K vocab, dim=256,  8 layers (~7.3 MB)   specialized domain, head_size=32, fast inference

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export HF_HOME="${ROOT}/.hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
mkdir -p "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"
PY="${ROOT}/.venv/bin/python"
SCRIPT="${ROOT}/train_tiny_model.py"
OUT="${ROOT}/generated_presets"
mkdir -p "$OUT"

COMMON=(
  --dataset tiny_stories
  --max-samples 8000
  --max-steps 12
  --epochs 1
  --seed 42
)

run_one() {
  local name="$1"
  shift
  echo ""
  echo "========== preset: ${name} =========="
  "$PY" "$SCRIPT" --preset "$name" --out "${OUT}/${name}" "${COMMON[@]}" "$@"
}

# ── Original presets (general size ladder) ───────────────────────────────────
run_one micro --batch-size 4
run_one tiny --batch-size 4
run_one small --batch-size 2
run_one medium --batch-size 1 --grad-accum 2 --gradient-checkpointing
run_one large --batch-size 1 --grad-accum 4 --gradient-checkpointing
run_one xlarge --batch-size 1 --grad-accum 8 --gradient-checkpointing

# ── ESP32-S3 8 MB PSRAM targets (dim=128, INT8 + group_size=128) ─────────────
# These are smoke runs (--max-steps 12). For a trained model remove --max-steps
# and set --max-samples 200000 or higher, e.g.:
#   python train_tiny_model.py --preset leaner --dataset tiny_stories \
#       --max-samples 200000 --epochs 3 --out ./out_leaner
run_one baseline --batch-size 2                                          # 16K vocab / 12 layers
run_one leaner   --batch-size 2                                          #  8K vocab / 15 layers
run_one stretch  --batch-size 1 --grad-accum 2 --gradient-checkpointing  #  8K vocab / 18 layers
run_one stretch2 --batch-size 1 --grad-accum 2 --gradient-checkpointing  #  8K vocab / 20 layers / hidden=640
run_one narrow   --batch-size 1 --grad-accum 2                           #  4K vocab / dim=256 / 8 layers

echo ""
echo "Done. Checkpoints under: ${OUT}/<preset>/ (model.safetensors, tokenizer.json, config.json)"
echo ""
echo "ESP32-S3 8 MB targets:"
echo "  baseline → ${OUT}/baseline/  (16K vocab, dim=128, 12 layers, ~6.4 MB INT8)"
echo "  leaner   → ${OUT}/leaner/    ( 8K vocab, dim=128, 15 layers, ~6.4 MB INT8)"
echo "  stretch  → ${OUT}/stretch/   ( 8K vocab, dim=128, 18 layers, ~7.4 MB INT8)"
echo "  stretch2 → ${OUT}/stretch2/  ( 8K vocab, dim=128, 20 layers, hidden=640, ~8.1 MB INT8)"
echo "  narrow   → ${OUT}/narrow/    ( 4K vocab, dim=256,  8 layers, ~7.3 MB INT8, domain-specialized)"

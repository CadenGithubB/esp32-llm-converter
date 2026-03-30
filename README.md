# HardwareOne LLM Tool

**Train tiny language models on your PC and deploy them to ESP32-S3**

Part of the [Hardware One](https://github.com/CadenGithubB/HardwareOne) ecosystem — a self-contained IoT platform with WiFi, sensors, ESP-NOW mesh networking, MQTT, and local AI inference.

---

## What This Tool Does

HardwareOne LLM Tool trains ultra-compact GPT-2 style language models on a PC and converts them to run entirely on the ESP32-S3 microcontroller using only 8MB of PSRAM. No cloud, no internet required at runtime — the model runs locally on the device.

**Training happens on your PC. The model runs on the ESP32. Nothing is trained on the device.**

### Key Features

- **Tiny Models**: 4K vocab, 18 layers, ~7.3 MB quantized
- **PC-Based Training**: Train on any machine with Python and PyTorch (GPU strongly recommended)
- **INT8 Quantization**: Browser-based converter produces a single `model.bin` for the device
- **Q&A Optimized**: Boundary-aware packing prevents answer bleed across training blocks
- **Two-Phase Training**: Positive Q&A examples followed by negative out-of-domain corrections
- **Hardware One Integration**: Drop the converted `model.bin` on the SD card and it runs

---

## Quick Start

### 1. Install Dependencies

```bash
cd training
pip install -r requirements.txt
```

For GPU training (recommended):
```bash
# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### 2. Train Your Model

```bash
python training/train_tiny_model_gpu.py \
    --preset HW1HelpAgent192_deep \
    --text training/hardwareone_rich.txt \
    --negatives training/training_data/hardwareone_qa_negatives.txt \
    --epochs 150 --lr 3e-4 --batch-size 16 \
    --out ./out_HW1HelpAgent192_deep
```

Training takes ~30-60 minutes on a modern GPU. CPU training is possible but slow (many hours).

See `training/INSTRUCTIONS.txt` for full details including CPU training commands.

### 3. Convert to ESP32 Format

1. Open `index.html` in Chrome/Edge/Firefox
2. Drag the output folder (`./out_HW1HelpAgent192_deep`) onto the page
3. Select **INT8 quantization**, group size **128**
4. Click **Convert**, then **Download** — saves `model.bin`

### 4. Deploy to Hardware One

Copy `model.bin` to `/sd/llm/` on the SD card or upload via the web Files page. Load it from the LLM tab or CLI.

---

## What's Included

### Training (`training/`)
- `train_tiny_model_gpu.py` — GPU training script (recommended)
- `train_tiny_model.py` — CPU training script
- `hardwareone_rich.txt` — Complete training corpus (Q&A pairs, passages, conversations)
- `training_data/hardwareone_qa_negatives.txt` — Out-of-domain corrections for Phase 2
- `requirements.txt` — Python dependencies
- `INSTRUCTIONS.txt` — Detailed training guide and preset reference

### Converter (root)
- `index.html` — Browser-based INT8 quantization converter
- `tokenizer.js` — Tokenizer used by the converter

### Technical Docs (`technical_docs/`)
- `architecture_comparison.html` — Visual comparison of model architectures
- `transformer_deep_dive.html` — Detailed transformer internals reference

### Download
- `training/hardwareone_training_package.zip` — Everything in `training/` bundled for easy download

---

## Model Presets

| Preset | Vocab | Layers | Dim | FFN | PSRAM (INT8) | Notes |
|--------|-------|--------|-----|-----|--------------|-------|
| **HW1HelpAgent192_deep** | 4K | 18 | 192 | 320 | ~7.3 MB | **Recommended** — best depth/width balance, 733KB headroom |
| HW1HelpAgent | 4K | 22 | 128 | 768 | ~7.5 MB | Proven fallback, wide FFN |
| HW1HelpAgent192 | 4K | 12 | 192 | 768 | ~7.5 MB | Wider per-layer but shallower |
| narrow3 | 4K | 18 | 128 | 768 | ~6.9 MB | Previous default |

All presets target 8MB PSRAM on ESP32-S3. See `training/INSTRUCTIONS.txt` for full list.

---

## Training Philosophy

### Boundary-Aware Q&A Packing
Q&A pairs are packed into fixed 128-token training blocks without splitting any pair across a boundary. Before this fix, ~39% of pairs were corrupted by being split mid-answer. The model now learns clean, complete Q&A associations.

### Two-Phase Learning
1. **Phase 1** (~150 epochs): Learn positive Q&A associations from `hardwareone_rich.txt`
2. **Phase 2**: Apply negative corrections from `hardwareone_qa_negatives.txt` to prevent conflating similar topics (ESP-NOW vs WiFi, MQTT vs direct, etc.)

---

## Performance

**Typical results on ESP32-S3 @ 240MHz with HW1HelpAgent192_deep:**
- **Inference speed**: 2-4 tokens/second (INT8)
- **Model size**: ~7.3 MB (fits in 8 MB PSRAM with ~733 KB headroom)
- **Context window**: 128 tokens
- **Use case**: Single-turn domain Q&A — "What is ESP-NOW?", "How do I set the MQTT broker?"

---

## Integration with Hardware One Firmware

The trained model integrates with [Hardware One firmware](https://github.com/CadenGithubB/HardwareOne):

```bash
# CLI usage
llm load              # Load model.bin from SD card
llm generate What is ESP-NOW?
llm models            # List available models
llm status            # Check model state
```

**Web UI**: Navigate to the LLM tab for a chat interface with token-per-second stats.

---

## Requirements

- **Python 3.8+**
- **PyTorch 2.0+** (CPU or CUDA)
- **8GB+ RAM** (16GB recommended for GPU training)
- **Modern browser** (Chrome/Edge/Firefox for the converter)
- **ESP32-S3** with 8MB PSRAM (for deployment)

---

## License

MIT License — See LICENSE file for details.

---

## Related

- **[Hardware One Firmware](https://github.com/CadenGithubB/HardwareOne)** — ESP32-S3 IoT platform with LLM inference support

---

**Built for Hardware One — Local AI, No Cloud Required**

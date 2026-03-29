# Hardware One — Training Commands

## Step 1: Verify environment

```bash
python3 --version
```

```bash
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

```bash
python3 -c "import transformers, tokenizers, accelerate; print('deps ok')"
```

---

## Step 2: Install dependencies (if not done)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

```bash
pip install transformers datasets tokenizers accelerate
```

---

## Step 3: Train the narrow2 model (recommended — rich multi-format dataset)

### Step 3a: Generate the rich training corpus

The rich dataset includes Q&A pairs, descriptive passages, semantic clusters, sentence
completions, corrective pairs, command references, and multi-turn conversations. Each
data type teaches a different aspect of language quality.

`--repeat 3` writes every sample 3 times into the file (varied context positions).
`--epochs 8` loops the full dataset 8 times during training. Total exposure: 24× per sample.

```bash
python3 make_rich_dataset.py --repeat 3 --out hardwareone_rich.txt
```

### Step 3b: Train on the rich corpus

```bash
python3 train_tiny_model_gpu.py --preset narrow2 \
  --text hardwareone_rich.txt hardwareone_overview.txt \
  --negatives hardwareone_qa_negatives.txt \
  --epochs 200 \
  --batch-size 16 \
  --lr 3e-4 \
  --out ./out_narrow2
```

**Without negatives** (positive data only — simpler, test first):

```bash
python3 train_tiny_model_gpu.py --preset narrow2 \
  --text hardwareone_rich.txt hardwareone_overview.txt \
  --epochs 200 \
  --batch-size 16 \
  --lr 3e-4 \
  --out ./out_narrow2
```

> **`--repeat` vs `--epochs`**: `--repeat` controls how many times each sample is written
> into the training file (preprocessing). `--epochs` controls how many times the trainer
> loops over the file. They are NOT equivalent to simply multiplying — `--epochs` is what
> actually drives convergence. The model needs enough **gradient steps** to converge: with
> ~230 training blocks at batch=16, each epoch is only ~15 steps. You need 200 epochs to
> reach ~3000 steps and get loss below 0.1. `--repeat` adds variety in how data is chunked,
> but does not replace epochs. Rule of thumb: always train until loss is below 0.5, ideally below 0.1.

> You should see output within seconds: GPU info, "Loaded N text paragraphs...", "Training BPE tokenizer...".
> If nothing appears after 30 seconds, kill and try the CPU script (Step 6).
>
> **New debug output**: The script now shows tokenizer verification (Q:/A: as atomic tokens),
> domain Q&A test generations after each phase, and loss curves.

> You should see output within seconds: GPU info, "Loaded N text paragraphs...", "Training BPE tokenizer...".
> If nothing appears after 30 seconds, kill and try the CPU script (Step 5).
>
> **New debug output**: The script now shows tokenizer verification (Q:/A: as atomic tokens),
> domain Q&A test generations after each phase, and loss curves.

### Two-phase training flags

| Flag | Default | Description |
|------|---------|-------------|
| `--negatives FILE` | none | Negative reinforcement data file. Enables two-phase training. |
| `--neg-epochs N` | same as `--epochs` | Epochs for phase 2 (with negatives). |
| `--neg-lr RATE` | `--lr * 0.5` | Learning rate for phase 2 (lower to avoid catastrophic forgetting). |
| `--qa-test-prompts FILE` | built-in | Custom Q&A test prompts file (one `Q:` per line). |

---

## Step 4: Train the narrow model (shallower, faster inference)

Uses only the Hardware One training data. Optimized for domain Q&A with richer per-head attention (head_size=32).
Use this if you need faster inference on the device or want to compare against narrow2.

```bash
python3 train_tiny_model_gpu.py --preset narrow \
  --text hardwareone_qa.txt hardwareone_qa_expanded.txt hardwareone_overview.txt hardwareone_qa_v2.txt hardwareone_qa_troubleshooting.txt hardwareone_qa_comprehensive.txt \
  --epochs 200 \
  --batch-size 8 \
  --lr 3e-4 \
  --out ./out_narrow
```

---

## Step 5: Train the stretch2 model (more layers, general purpose)

```bash
python3 train_tiny_model_gpu.py --preset stretch2 \
  --text hardwareone_rich.txt hardwareone_overview.txt \
  --epochs 200 \
  --batch-size 16 \
  --lr 3e-4 \
  --out ./out_stretch2
```

---

## Step 6 (old Step 5): CPU fallback (if GPU script hangs)

```bash
python3 train_tiny_model.py --preset narrow2 \
  --text hardwareone_qa.txt hardwareone_qa_expanded.txt hardwareone_overview.txt hardwareone_qa_v2.txt hardwareone_qa_troubleshooting.txt hardwareone_qa_comprehensive.txt \
  --epochs 200 \
  --out ./out_narrow2
```

---

## Step 7: Estimate parameter count without training

```bash
python3 train_tiny_model_gpu.py --preset narrow2 --estimate-only
```

```bash
python3 train_tiny_model_gpu.py --preset narrow --estimate-only
```

```bash
python3 train_tiny_model_gpu.py --preset stretch2 --estimate-only
```

---

## Troubleshooting

**Hangs immediately / no output at all:**
```bash
python3 -c "import torch; print(torch.zeros(10))"
```
If this hangs, torch install is broken. Reinstall deps.

**Hangs after "Training BPE tokenizer...":**
Normal — training tokenizer on your text files. Wait 30 seconds.

**Hangs after "torch.compile...":**
Normal on first run — compiles CUDA kernels. Wait up to 2 minutes.

**CUDA out of memory:**
Add `--batch-size 16` or `--batch-size 8` to the command.

**"No module named X":**
```bash
pip install transformers datasets tokenizers accelerate
```

**Check what's happening (run with verbose output):**
```bash
python3 -u train_tiny_model_gpu.py --preset narrow2 \
  --text hardwareone_qa.txt hardwareone_qa_expanded.txt hardwareone_overview.txt hardwareone_qa_v2.txt \
  --epochs 200 --batch-size 8 --lr 3e-4 \
  --out ./out_narrow2 2>&1 | tee training.log
```

---

## After training

Copy the output folder back to the Mac converter machine:

```bash
scp -r ./out_narrow2 user@macbook:/Users/morgan/esp/llm-converter/
```

Then open `index.html` in the converter, drag in the output folder, select INT8, download `model.bin`, copy to SD card.

---

## Preset comparison

| Preset  | Vocab | dim | Layers | n_head | head_size | Wts   | +KV@64 | Total  | Notes                              |
|---------|-------|-----|--------|--------|-----------|-------|--------|--------|------------------------------------|
| narrow2 | 4K    | 128 | 20     | 4      | 32        | ~4.7  | ~1.3   | ~6.9MB | ★ Deep domain Q&A, rich attention  |
| narrow  | 4K    | 256 | 8      | 8      | 32        | ~5.1  | ~1.0   | ~6.1MB | Wide attention, fast inference     |
| stretch2| 8K    | 128 | 20     | 8      | 16        | ~5.4  | ~1.3   | ~6.7MB | General, wider FFN                 |
| stretch | 8K    | 128 | 18     | 8      | 16        | ~4.4  | ~1.1   | ~5.5MB | General depth                      |

`narrow2` uses ~6.9 MB total PSRAM at ctx=64 (~1 MB headroom). **Load with ctx=64 or 96 max —
ctx=128 will exceed 8 MB and fail to load.** 2.5× more layers than `narrow`, head_size=32,
wider FFN (n_inner=640). **Use narrow2 for Hardware One domain Q&A.**

---

## Training data files

### Primary corpus (use these)

| File | Purpose | Content |
|------|---------|---------|
| `hardwareone_rich.txt` | **Rich multi-format corpus (generated)** | 170 Q&A pairs + passages + clusters + completions + corrections + command refs + conversations |
| `hardwareone_overview.txt` | Narrative overview | Prose description of the whole system |
| `hardwareone_qa_negatives.txt` | Negative reinforcement | "No." answers, wrong-premise corrections |

Generate `hardwareone_rich.txt` with: `python3 make_rich_dataset.py --repeat 3`

### Legacy files (superseded by rich corpus)

| File | Purpose |
|------|---------|
| `hardwareone_qa.txt` | Original core Q&A pairs |
| `hardwareone_qa_expanded.txt` | Extended Q&A |
| `hardwareone_qa_v2.txt` | Topic-anchored Q&A |
| `hardwareone_qa_comprehensive.txt` | Wide coverage Q&A |
| `hardwareone_qa_troubleshooting.txt` | Troubleshooting Q&A |

**Q: and A: are special tokens** — the tokenizer treats them as atomic single tokens (not split by BPE).
This gives the model clean question/answer boundary signals during training.

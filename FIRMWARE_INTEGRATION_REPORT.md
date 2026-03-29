# ESP32 LLM Converter — integration report for firmware / downstream tools

This document describes **what [esp32-llm-converter](https://github.com/CadenGithubB/esp32-llm-converter)** produces, how it relates to Hugging Face models, and **why it is not interchangeable** with **llama2.c**-style checkpoints or other random `.bin` files.

Feed this file to an AI or engineer implementing **on-device loading and inference** so the runtime matches the **exact** binary layout below.

---

## 1. Purpose of the web tool

| The tool does | The tool does **not** do |
|----------------|---------------------------|
| Load HF-style checkpoints (**safetensors** and/or **pytorch_model.bin** in-browser) | Train models |
| Parse `config.json` + `tokenizer.json` | Export llama2.c / Karpathy checkpoints |
| Optionally quantize weight matrices to **INT8** with per-group **FP32 scales** | Produce GGUF, ONNX, or raw PyTorch `.pt` |
| Pack a **single file** `model.bin` for upload to a device | Match the memory layout of **llama2.c** `checkpoint.bin` |

The converter is a **browser-based packer** from “HF causal LM weights + tokenizer” → **one self-describing binary** (`model.bin`).

---

## 2. Critical: not llama2.c format

Many ESP32 LLM ports (e.g. **llama2.c**, **DaveBben/esp32-llm** stories checkpoints) expect:

- A **small fixed header** (often a handful of `int32` fields in a **specific order**), then
- A **flat float32 weight blob** laid out exactly as the C `transformer()` code expects.

**This project’s `model.bin` is a different format entirely:**

- **Magic:** `0x4C4C4D31` (`"LLM1"` in big-endian uint32) at offset **0**
- **Header length:** **64 bytes** (not the llama2.c struct size)
- **Quantization:** weights may be **INT8 + FP32 scales** (see §4), not “all FP32”
- **Tokenizer:** embedded **inside the same file** after the header (not a separate `tokenizer.bin` unless you split it yourself)

If firmware reads the first bytes as **llama2.c `LLMConfig`**, values like `dim=827149388` are **misinterpreted garbage**—those integers are actually **magic/version/quant fields**, not `dim`/`seq_len`.

**Action for the firmware project:** either

1. **Implement a loader for this document’s format** (recommended if you control both sides), or  
2. **Write a separate PC-side tool** that converts `model.bin` (this format) → whatever your C stack expects (llama2.c layout), or  
3. **Drop** llama2.c layout assumptions and port **inference** to this tensor ordering and INT8 scheme.

There is **no** automatic compatibility: **renaming a file to `model.bin` is not enough.**

---

## 3. File layout overview

**Endianness:** multi-byte integers in the **header** use **explicit** endianness in the implementation: `DataView` writes **big-endian** for the magic; several fields use **little-endian** (`true` in `setUint16` / `setUint32`). **Treat the table below as authoritative only when cross-checking the source** (`index.html` / `format.js` in this repo).

**Practical approach for implementers:** mirror the **JavaScript** `buildOutputBin` / `tensorBlock` functions byte-for-byte.

### 3.1 Header (64 bytes)

| Offset | Size | Field | Notes |
|--------|------|--------|------|
| 0 | 4 | `magic` | **Big-endian** uint32: `0x4C4C4D31` |
| 4 | 1 | `version` | `1` |
| 5 | 1 | `quant_type` | `0` = FP32 weights, `1` = INT8 weights (embedding + matrices where quantized) |
| 6 | 2 | `group_size` | UINT16 LE — INT8 group size (e.g. 128) |
| 8 | 2 | `dim` | UINT16 LE — model width (`hidden_size` / `n_embd`) |
| 10 | 2 | `hidden_dim` | UINT16 LE — FFN hidden (`intermediate_size` / MLP width) |
| 12 | 1 | `n_layers` | |
| 13 | 1 | `n_heads` | |
| 14 | 1 | `n_kv_heads` | GQA: for MHA, equals `n_heads` |
| 15 | 4 | `vocab_size` | UINT32 LE — careful: starts at byte 15 (not 16-byte aligned) |
| 19 | 2 | `seq_len` | UINT16 LE — max sequence length from config |
| 21 | 1 | `arch_type` | `0` = Llama-style, `1` = GPT-2–family (includes GPT-Neo in this toolchain) |
| 22 | 42 | `reserved` | **Zero padding** to 64 bytes |

**Important:** Offsets 15–18 are `vocab_size` (4 bytes). Some readers wrongly assume 4-byte alignment before `vocab_size`; that will mis-parse the header.

### 3.2 Tokenizer block

| Field | Meaning |
|--------|---------|
| UINT32 LE | `tok_byte_len` — length in bytes of the following blob **only** |
| `tok_byte_len` bytes | Custom **packed tokenizer** (see §5), **not** raw `tokenizer.json` text |

### 3.3 Weights (sequential)

All tensors are prefixed with **UINT32 LE `n_elements`** = logical scalar count (number of FP32 values **before** quantizing to int8; i.e. matrix element count).

For each tensor:

- **FP32 mode (`quant_type` global is F32, or tensor stored as F32):**  
  `n_elements` then **`n_elements * 4` bytes** of float32 data.

- **INT8 mode:**  
  `n_elements` then  
  **`ceil(n_elements / group_size)` float32 scales** (4 bytes each), then  
  **`n_elements` signed int8 values**.

Order of tensors:

1. **Embedding** — shape conceptually `[vocab_size, dim]` flattened row-major (layout convention must match the device’s `matmul`).

2. **Per layer** `i = 0 .. n_layers-1`, fixed order of **9** slots:

   `attn_norm`, `q`, `k`, `v`, `o`, `ffn_norm`, `gate`, `up`, `down`

   - **Llama:** maps from `input_layernorm`, `q_proj`, `k_proj`, `v_proj`, `o_proj`, `post_attention_layernorm`, `gate_proj`, `up_proj`, `down_proj`.
   - **GPT-2:** uses combined attention + MLP weights; **gate** is a **dummy** (tiny / zero) tensor in the pack—still present so the layer struct size is uniform.
   - **GPT-Neo:** separate Q/K/V; same 9-slot layout as GPT-2 after export.

3. **Final norm** — FP32, length `dim`.

4. **LM head** — UINT8 flag: `1` = weights tied to embedding (no extra matrix), `0` = followed by another tensor block for `lm_head` (same INT8/FP32 rules).

---

## 4. INT8 dequantization

For each INT8 group of `group_size` elements sharing one scale `s` (float32):

```text
w_float[i] = quant[i] * s
```

Norm tensors (`attn_norm`, `ffn_norm`, final norm) stay **FP32** in the current exporter.

---

## 5. Embedded tokenizer binary (custom)

The blob after `tok_byte_len` is **not** SentencePiece `.model` and not raw HF JSON. It is produced by `encodeTokenizerBinary()` in the converter:

| Part | Format |
|------|--------|
| Header | UINT32 LE `vocab_size`, UINT32 LE `merge_count` |
| Vocab | For `id` in `0 .. vocab_size-1`: UINT8 `byte_len` (0–255), then `byte_len` UTF-8 bytes for that token string |
| Merges | `merge_count` times: UINT32 LE `left_id`, `right_id`, `merged_id` |

HF BPE special characters (`Ġ` → space, etc.) are normalized when encoding.

The device must implement **encode** (and optionally decode) consistent with this table + merge rules, or you must embed a different tokenizer pipeline.

---

## 6. Supported source architectures (training export)

The converter recognizes **three** families when reading HF `config.json` + tensors:

| Family | `arch_type` in header | Notes |
|--------|------------------------|--------|
| Llama-style (Llama, Mistral, many causal LMs) | `0` | `model.embed_tokens`, `model.layers.*` |
| GPT-2 | `1` | `transformer.wte`, `transformer.h.*.attn.c_attn`, etc. |
| GPT-Neo | `1` | `transformer.h.*.attn.attention.q_proj` etc. |

Other architectures (T5, encoder-only, etc.) are **not** exported unless you extend the importer.

---

## 7. Typical workflow for users

1. Obtain a compatible HF model (or train with `train_tiny_model.py` — presets `micro` … `xlarge`, plus `--estimate-only` for parameter counts; **GPT-2–compatible** checkpoint).
2. Open `index.html` locally (file or HTTP; see project README for mixed-content rules if fetching from HF).
3. Drop folder or **Fetch** from Hugging Face.
4. Choose **INT8** (or F32) and group size; set **LittleFS KB** estimate if shown.
5. **Convert & Download** → `model.bin`.
6. Upload **`model.bin`** to the device (e.g. `/system/llm/model.bin`).

**There is no separate tokenizer file** in the default pipeline—the tokenizer is inside `model.bin`. If firmware expects `tokenizer.json` on disk, either split the format or change the device to read the embedded block.

---

## 8. Sanity checks for firmware (recommended)

Before allocating KV cache or large buffers:

1. **Verify magic** `0x4C4C4D31` at offset 0.
2. **Parse header** exactly as §3.1; reject if `version != 1` (until a new spec exists).
3. **Range-check** `dim`, `n_layers`, `vocab_size`, `seq_len`, etc.
4. **Compute expected file size** from header + known tensor layout (including INT8 scale overhead) and compare to **actual file size** before trusting any dimension.
5. **Do not** log “model config” fields **before** validation succeeds—garbage files will produce nonsense numbers.

---

## 9. Relation to “44 KB” / tiny files

A **valid** `model.bin` for a tiny model is usually **hundreds of KB to a few MB** after INT8 (tokenizer + embeddings dominate small models). A **very small** file (e.g. tens of KB) together with **nonsense header fields** usually means:

- Wrong file uploaded (HTML error page, truncated download, different tool’s `.bin`), or  
- Firmware reading the file with the **wrong struct** (e.g. llama2.c header on this format).

---

## 10. Summary for the “other” AI / firmware engineer

| Question | Answer |
|----------|--------|
| Is this llama2.c? | **No.** Different header, quantization, tokenizer embedding, tensor order. |
| Can I load it with llama2.c `checkpoint` code? | **Not without a full adapter** that implements §3–5. |
| What must the device implement? | Parser for this header + tokenizer blob + tensor blocks + INT8 dequant + **forward pass** for `arch_type` 0 or 1. |
| Where is the reference? | `buildOutputBin` / `tensorBlock` / `encodeTokenizerBinary` in this repo’s `index.html` (and `format.js`). |

---

## 11. Repo pointers

- **Converter UI + packer:** `index.html`  
- **Format documentation (comments):** `format.js`  
- **Optional Python trainer (GPT-2–compatible export):** `train_tiny_model.py` (`--preset`, `--estimate-only`, `--grad-accum`, `--gradient-checkpointing`)  
- **Worker (INT8 quant):** `worker.js`

---

*Document generated for cross-project alignment. Version: converter `VERSION == 1` in header.*

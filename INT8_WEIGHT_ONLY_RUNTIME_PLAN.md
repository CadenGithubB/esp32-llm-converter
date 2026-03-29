# Plan: INT8 weights in PSRAM (weight-only quant at runtime)

This document is for **firmware / on-device inference** (e.g. ESP32-S3 + PSRAM). The **browser converter** (`index.html`) already emits **LLM1** `model.bin` with **INT8 matrix weights + per-group FP32 scales** when quantization is INT8 — see `FIRMWARE_INTEGRATION_REPORT.md` §3–4.

The gap is **not** the file format. The gap is **how the loader and forward pass use those bytes after load**.

---

## 1. Problem statement

### Current behavior (as described)

- Weights are stored **INT8 on disk** (plus scales).
- **Loader expands everything to FP32 in PSRAM** at load time.
- **Rough RAM ≈ 4×** the size of the INT8 weight payload for those tensors (plus scales, which are small).
- Example intuition: **~5 MB** of INT8 weights on disk → **~20 MB** FP32 buffer if fully expanded — **before** KV cache and activations.
- Available PSRAM after Wi-Fi / HTTP / system (e.g. **~5–6 MB free**) is easy to exceed even when the **file** looks “small.”

### What we want

- **Keep matrix weights as INT8 (+ scales) in PSRAM** for the lifetime of the model.
- **Dequantize only where needed for math**, ideally **per group or per tile** during **matmul / fused ops**, then discard temporaries — same *idea* as **llama.cpp**-style weight-only quantization (not identical layout).
- **Norms, biases, activations, KV cache** can remain **FP32** (or FP16 later); those are smaller than weights for many small models and are **not** the 4× expansion problem.

### Why this helps “broader hardware”

Same **LLM1** blobs become usable on **more boards** without shrinking architecture as aggressively — the dominant term stops being “4× all weights in RAM.”

---

## 2. What the format already provides (no converter change required)

For **global INT8** (`quant_type == 1` in the 64-byte header):

- Each quantized tensor: **UINT32 `n_elements`**, then **`ceil(n_elements / group_size)` FP32 scales**, then **`n_elements` int8 values**.
- Dequant rule per group (already in report): `w_float[i] = quant[i] * scale[group(i)]`.

**Firmware implication:** You already have everything needed to implement **fused dequant × matmul** without ever allocating a full FP32 copy of the weight matrix.

**Optional future:** Converter changes only if you add **new** packing (e.g. Q4_K) — **out of scope** for this plan.

---

## 3. Non-goals (v1 of this refactor)

- **Quantized KV cache** (INT8/FP16 KV): valuable later; not required to fix the **weight expansion** problem.
- **Changing token limits in the converter:** user-facing caps are **device policy** and **model `seq_len`**, not the packer.
- **INT8 norms:** current exporter keeps norms FP32; tiny compared to weights; can stay FP32 in v1.

---

## 4. Implementation phases (firmware)

### Phase A — Inventory & hot spots

- Trace **`loadWeights()`** (or equivalent): where **INT8 → FP32** bulk buffers are allocated.
- List **every consumer** of weight tensors: `matmul`, attention projections, MLP, embedding lookup, optional `lm_head`.
- Confirm **GPT-2** paths: dummy **gate** tensor is **FP32** in the file even when global quant is INT8 — loader must **not** treat it as INT8 (see integration report + manifest `firmware_hint`).

### Phase B — Data structures

- For each **quantized** weight tensor, store:
  - `int8_t *weights` (or `uint8_t *` + sign convention),
  - `float *scales` (length `ceil(n_elements / group_size)`),
  - `uint32_t n_elements`, `uint16_t group_size` (from header or fixed at load).
- **Remove** or bypass `float *` pointers that point at **full-size FP32 copies** of those matrices.

### Phase C — Math: fused dequant + matmul

- Replace “dequant full matrix then `sgemm`” with one of:
  - **Per-group:** For output row/column blocks, accumulate  
    `dot(int8, int8 or fp32 activation) * scale`  
    into FP32 accumulators (details depend on layout **row-major vs column** and which side is quantized).
  - **Micro-kernel / tile:** Process a **tile** of the output; load only the INT8 weights for that tile + corresponding scales.

**Attention / QKV:** Same pattern for `q`, `k`, `v`, `o`, MLP `up`/`down` — wherever a large matrix multiply sits.

**Embedding (`wte`):** If stored INT8, use **gather + per-row scale** (or per-group along flattened row) instead of expanding full embedding table to FP32.

**Tied LM head:** If tied to embedding, no separate matrix; if separate INT8 block, same matmul path.

### Phase D — Memory accounting

- After Phase B–C, **recompute peak PSRAM**:
  - **Weights:** ~`n_int8 + n_scales*4` (+ small alignment), not `n_params*4`.
  - **KV + activations:** unchanged formula (still FP32 for v1).
- Surface a **single “estimated load”** in logs (you already print “need X KB, have Y KB”) — align that formula with the new layout.

### Phase E — Validation

- **Numerical:** Compare logits / a few token IDs against **reference** (current FP32-expanded path or PC float) on **short** prompts — expect small drift; large drift = bug in grouping or layout.
- **Stress:** Long generation, **watchdog** / stack high-water — ensure no new allocs in the hot loop.
- **Performance:** Expect some **CPU** overhead; often acceptable on S3 if bandwidth-bound; measure **tok/s** before/after.

---

## 5. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Layout mismatch (group order vs row-major) | Unit-test one known **small** `model.bin` from this converter; compare to FP32 golden |
| GPT-2 gate / norms mis-read as INT8 | Branch on tensor **type** + **arch**; follow manifest hints |
| Speed regression | Tile size tuning; optional ESP-NN / SIMD later |
| Fragmentation | Allocate weight buffers once at load; pool scratch |

---

## 6. Relation to this repo

| Piece | Responsibility |
|--------|----------------|
| **`index.html` / LLM1 packer** | Already writes INT8 + scales; documents norms FP32 |
| **Training** | `train_tiny_model.py` — unrelated to runtime INT8 |
| **Device C++** (`System_LLM.cpp` etc.) | **This plan applies here** |

No **required** converter change for “weight-only INT8 in PSRAM” if you already load **LLM1** correctly.

---

## 7. Per-request access patterns — flash vs PSRAM (for implementers)

This section captures a **separate** RAM strategy from §4: not “how weights are multiplied,” but **what data must stay hot for every token** vs what is **only needed in phases** of a single user request (encode → forward loop → decode).

### 7.1 What is *not* “read once per interaction”

Assume a typical pipeline: **tokenize prompt → prefill + autoregressive decode → decode token IDs to text for the UI**.

| Data | Typical access pattern |
|------|-------------------------|
| **Transformer weights** | Used on **every forward** for **every new token** (and in prefill). **Not** one-shot. |
| **KV cache** | **Updated every step**; must be fast RAM if performance matters. |
| **Positional encoding** (WPE or RoPE tables / logic) | Used **every forward** as the sequence grows. **Not** one-shot. |
| **Embedding table (`wte`)** | Used repeatedly (prefill + last-token path in decode). **Not** one-shot. |

So the **large** blobs are **not** candidates for “load once at start of request, never touch again during generation.” If they are moved off PSRAM, it is **mmap / layer streaming** (different design), not because they are cold after step one.

### 7.2 What *may* be “mostly cold” during the token-generation loop

Depends on **exact** firmware structure; **trace** the code paths.

| Data | Why it might leave PSRAM or be loaded only in phases |
|------|--------------------------------------------------------|
| **Embedded tokenizer** (`model.bin` tokenizer block after header — see `FIRMWARE_INTEGRATION_REPORT.md` §5) | **Encode:** BPE / vocab / merges are hot when converting **user text → token IDs**. **Steady decode loop** is often **pure tensor ops on IDs** — tokenizer may be **idle** until you **decode** logits to strings for display (or decode incrementally). If the pipeline separates these phases, the tokenizer table is a candidate to **keep in flash** (mmap, or read into a **small** scratch only for encode/decode windows) instead of **resident in PSRAM for the entire generation**. |
| **Parsed header / tiny config** | Read once per model load; negligible size; moving them barely matters. |
| **Anything only used in prefill** | If your implementation has structures used **only** when processing the **prompt** and not in the per-token decode loop, they could be **released after prefill** — **highly implementation-specific**; requires profiling. |

### 7.3 Design implications

- **Do not confuse** (a) **streaming weights from flash** every layer/token with (b) **tokenizer / one-shot** data. Different bottlenecks: bandwidth vs peak RAM.
- **Validate** by instrumenting: tag allocations with “prefill only,” “decode loop,” “encode only,” and confirm the **hot loop** does not touch tokenizer bytes if you move them to flash.
- **Risk:** If **decode** runs every token for streaming UI, tokenizer **decode** may be hot more often than “once”; still might be **smaller working set** than keeping the **full** vocab/merge tables in PSRAM.

### 7.4 For another AI / reviewer

When auditing `System_LLM.cpp` (or equivalent), explicitly answer:

1. Where does **encode(prompt)** end and **generate()** begin?
2. Does the **inner per-token loop** call into tokenizer code, or only **tensor forward + sampling**?
3. Can **tokenizer** be **mmap’d** from the same `model.bin` on flash without a full RAM copy?

---

## 8. Summary

- **Today:** INT8 in the file is wasted if the device **materializes FP32 for all weights** at load — **~4×** RAM vs INT8 payload.
- **Target:** **Keep INT8 weights + scales in PSRAM**; **dequantize inside matmul** (group- or tile-wise). KV/activations FP32 in v1.
- **Also consider:** **Tokenizer and other phase-local data** may be candidates for **flash-backed** storage if the **decode loop** does not need them resident — §7.
- **Format:** Already sufficient; work is **firmware forward pass + loader refactor** and validation.

This is a **meaningful** refactor of the hot path but is the standard way to make the same **LLM1** binaries usable on **more** embedded hardware without shrinking models to extreme sizes.

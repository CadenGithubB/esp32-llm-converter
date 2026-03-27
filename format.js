/**
 * format.js — write the ESP32 LLM binary format
 *
 * ┌─────────────────────────────────────────────────────────┐
 * │  HEADER  (64 bytes)                                     │
 * │  magic        uint32  0x4C4C4D31  "LLM1"               │
 * │  version      uint8   1                                 │
 * │  quant_type   uint8   0=f32  1=int8                     │
 * │  group_size   uint16  quantization group size           │
 * │  dim          uint16  embedding dimension               │
 * │  hidden_dim   uint16  FFN hidden dimension              │
 * │  n_layers     uint8   transformer layers                │
 * │  n_heads      uint8   attention heads                   │
 * │  n_kv_heads   uint8   key/value heads (GQA)             │
 * │  vocab_size   uint32  vocabulary size                   │
 * │  seq_len      uint16  max sequence length               │
 * │  arch_type    uint8   0=Llama  1=GPT-2                  │
 * │  pad to 64 bytes                                        │
 * ├─────────────────────────────────────────────────────────┤
 * │  TOKENIZER BLOCK                                        │
 * │  uint32: block_byte_size (not including this uint32)    │
 * │  ... tokenizer binary (see tokenizer.js) ...            │
 * ├─────────────────────────────────────────────────────────┤
 * │  EMBEDDING TABLE                                        │
 * │  uint32: n_elements                                     │
 * │  float32[n_elements]                                    │
 * ├─────────────────────────────────────────────────────────┤
 * │  LAYERS  (repeated n_layers times)                      │
 * │  For each of: attn_norm, q, k, v, o,                   │
 * │               ffn_norm, gate, up, down                  │
 * │    uint32: n_elements                                   │
 * │    if int8:                                             │
 * │      float32[ceil(n_elements/group_size)]  scales       │
 * │      int8[n_elements]                      quant data   │
 * │    if f32:                                              │
 * │      float32[n_elements]                               │
 * ├─────────────────────────────────────────────────────────┤
 * │  FINAL NORM                                             │
 * │  uint32: n_elements                                     │
 * │  float32[n_elements]                                    │
 * ├─────────────────────────────────────────────────────────┤
 * │  LM HEAD                                                │
 * │  uint8: 1=tied to embedding, 0=separate                 │
 * │  if separate: uint32 + data (same format as other mats) │
 * └─────────────────────────────────────────────────────────┘
 */

export const MAGIC   = 0x4C4C4D31;
export const VERSION = 1;

export const QUANT = {
  F32:  0,
  INT8: 1,
};

/**
 * Build the complete output binary.
 *
 * @param {object}     config      - architecture params
 * @param {ArrayBuffer} tokBuf     - tokenizer binary from tokenizer.js
 * @param {Float32Array} embedData - embedding table weights
 * @param {Array}       layers     - array of layer objects, each:
 *   { attn_norm, q, k, v, o, ffn_norm, gate, up, down }
 *   each value is either:
 *     { mode:'f32', data:Float32Array }
 *     { mode:'int8', scales:Float32Array, quant:Int8Array }
 * @param {Float32Array} finalNorm - final RMSNorm weights
 * @param {object}       lmHead    - { tied:true } or { tied:false, ...same as layer weight }
 * @param {number}       quantType - QUANT.F32 or QUANT.INT8
 * @param {number}       groupSize - quantization group size
 */
export function buildOutputBin({
  config,
  tokBuf,
  embedData,
  layers,
  finalNorm,
  lmHead,
  quantType,
  groupSize,
}) {
  const parts = [];

  // ── Header ──────────────────────────────────────────────
  const hdr = new ArrayBuffer(64);
  const dv  = new DataView(hdr);
  dv.setUint32( 0, MAGIC,              false); // big-endian magic
  dv.setUint8 ( 4, VERSION);
  dv.setUint8 ( 5, quantType);
  dv.setUint16( 6, groupSize,          true);
  dv.setUint16( 8, config.dim,         true);
  dv.setUint16(10, config.hidden_dim,  true);
  dv.setUint8 (12, config.n_layers);
  dv.setUint8 (13, config.n_heads);
  dv.setUint8 (14, config.n_kv_heads);
  dv.setUint32(15, config.vocab_size,  true);
  dv.setUint16(19, config.seq_len,     true);
  dv.setUint8 (21, config.arch_type ?? 0);  // 0=Llama, 1=GPT-2
  // bytes 22-63 remain zero (pad)
  parts.push(hdr);

  // ── Tokenizer block ─────────────────────────────────────
  const tokSize = new Uint32Array([tokBuf.byteLength]);
  parts.push(tokSize.buffer);
  parts.push(tokBuf);

  // ── Embedding table (F32 or INT8 depending on quantType) ──
  const embedN = embedData.mode === 'int8' ? embedData.quant.length : embedData.data.length;
  parts.push(...tensorBlock(embedN, embedData));

  // ── Layers ──────────────────────────────────────────────
  const LAYER_KEYS = ['attn_norm', 'q', 'k', 'v', 'o', 'ffn_norm', 'gate', 'up', 'down'];
  for (const layer of layers) {
    for (const key of LAYER_KEYS) {
      const w = layer[key];
      if (!w) throw new Error(`Layer missing weight: ${key}`);
      parts.push(...tensorBlock(w.mode === 'int8' ? w.quant.length : w.data.length, w));
    }
  }

  // ── Final norm (always F32) ──────────────────────────────
  parts.push(...tensorBlock(finalNorm.length, { mode: 'f32', data: finalNorm }));

  // ── LM Head ─────────────────────────────────────────────
  if (lmHead.tied) {
    parts.push(new Uint8Array([1]).buffer);
  } else {
    parts.push(new Uint8Array([0]).buffer);
    const n = lmHead.mode === 'int8' ? lmHead.quant.length : lmHead.data.length;
    parts.push(...tensorBlock(n, lmHead));
  }

  // ── Assemble ─────────────────────────────────────────────
  return assembleBlob(parts);
}

function tensorBlock(nElements, weight) {
  const parts = [];
  const countBuf = new Uint32Array([nElements]);
  parts.push(countBuf.buffer);

  if (weight.mode === 'int8') {
    parts.push(weight.scales.buffer);
    parts.push(weight.quant.buffer);
  } else {
    parts.push(weight.data.buffer);
  }

  return parts;
}

function assembleBlob(parts) {
  const total = parts.reduce((n, b) => n + b.byteLength, 0);
  const out   = new Uint8Array(total);
  let offset  = 0;
  for (const buf of parts) {
    out.set(new Uint8Array(buf), offset);
    offset += buf.byteLength;
  }
  return new Blob([out], { type: 'application/octet-stream' });
}

/** Estimate output size in bytes before actually building */
export function estimateSize({ config, tokBufSize, quantType, groupSize }) {
  const { dim, hidden_dim, n_layers, n_kv_heads, vocab_size, seq_len } = config;
  const n_heads    = config.n_heads;
  const head_dim   = dim / n_heads;
  const kv_dim     = n_kv_heads * head_dim;

  function matSize(rows, cols) {
    if (quantType === QUANT.F32) return rows * cols * 4;
    const numel  = rows * cols;
    const nGroups = Math.ceil(numel / groupSize);
    return numel * 1 + nGroups * 4;  // int8 + scales
  }

  let size = 64;                              // header
  size += 4 + tokBufSize;                     // tokenizer
  size += 4 + matSize(vocab_size, dim);        // embedding

  for (let i = 0; i < n_layers; i++) {
    size += 4 + dim * 4;                      // attn_norm (f32)
    size += 4 + matSize(dim, dim);            // q
    size += 4 + matSize(kv_dim, dim);         // k
    size += 4 + matSize(kv_dim, dim);         // v
    size += 4 + matSize(dim, dim);            // o
    size += 4 + dim * 4;                      // ffn_norm (f32)
    size += 4 + matSize(hidden_dim, dim);     // gate
    size += 4 + matSize(hidden_dim, dim);     // up
    size += 4 + matSize(dim, hidden_dim);     // down
  }

  size += 4 + dim * 4;                        // final norm (f32)
  size += 1;                                  // lm_head tied flag
  // assume tied; if not: + 4 + matSize(vocab_size, dim)

  return size;
}

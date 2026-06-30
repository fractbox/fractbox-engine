// Compact share codec — a tight binary encoding for share links, replacing the
// verbose JSON-then-base64 form. Versioned and SECTION-BASED so it stays
// forward-compatible: a decoder skips sections (and op params) it doesn't
// understand, and new data rides in NEW sections without breaking old links.
//
// Wire format (all bytes, then base64url for the URL):
//
//   CONTAINER
//     u8        version (CODEC_VERSION)
//     repeat until end of buffer:
//       u8      section tag
//       varint  payload length (bytes)
//       bytes   payload
//     → unknown tags are skipped by length (forward-compat)
//
//   primitives
//     varint    unsigned LEB128
//     zigzag    signed int encoded as varint( (n<<1) ^ (n>>31) )
//
//   TAG_FORMULA payload (engine data — encoded/decoded here)
//     u8        flags: bit0 addC · bit1 julia · bits2..3 deOption (0..3)
//     varint    iters
//     zigzag    camera.yawDeg   ×10
//     zigzag    camera.pitchDeg ×10
//     zigzag    camera.dist     ×100
//     zigzag    camera.fovDeg   ×10
//     [if julia] zigzag juliaC[0..2] ×1000
//     varint    opCount
//     repeat opCount:
//       varint  opcode (operator id)
//       varint  paramCount
//       repeat paramCount: zigzag value ×100
//
// App-defined sections (theme, sound matrix) live in the app; they reuse the
// ByteWriter/ByteReader/pack helpers exported here. Their tags are reserved
// below so the registry stays in one place.

import { OPERATORS } from './operators.js';

export const CODEC_VERSION = 1;

export const TAG = {
  FORMULA: 0x01,
  THEME: 0x02, // app: a single varint theme index
  MATRIX: 0x03, // app: the sound matrix
  COLORING: 0x04, // app: the full coloring (mode/colors/palette/light) — overrides THEME
  SPIN: 0x05, // app: auto-spin state (on + speed + axis tilt)
  ASCII: 0x06, // app: ASCII view state (on + density)
  MUTED: 0x07, // app: indices of muted ops (kept in the op-list but not rendered)
};

// Fixed-point scales (kept here so encode/decode can never disagree).
const CAM_ANGLE = 10; // 0.1° precision for yaw/pitch/fov
const CAM_DIST = 100; // 0.01 precision for distance
const JULIA = 1000; // 0.001 precision for the Julia seed
const PARAM = 100; // 0.01 precision for operator params (matches slider steps)

const q = (v, scale) => Math.round((v || 0) * scale);
const dq = (n, scale) => n / scale;

// ── byte primitives ────────────────────────────────────────────────────────

export class ByteWriter {
  constructor() {
    this.bytes = [];
  }
  u8(n) {
    this.bytes.push(n & 0xff);
    return this;
  }
  varint(n) {
    n = n >>> 0;
    while (n >= 0x80) {
      this.bytes.push((n & 0x7f) | 0x80);
      n >>>= 7;
    }
    this.bytes.push(n);
    return this;
  }
  zigzag(n) {
    n = n | 0;
    return this.varint((n << 1) ^ (n >> 31));
  }
  raw(u8arr) {
    for (const b of u8arr) this.bytes.push(b);
    return this;
  }
  take() {
    return Uint8Array.from(this.bytes);
  }
}

export class ByteReader {
  constructor(u8arr) {
    this.b = u8arr;
    this.pos = 0;
  }
  get done() {
    return this.pos >= this.b.length;
  }
  u8() {
    return this.b[this.pos++];
  }
  varint() {
    let result = 0,
      shift = 0,
      byte;
    do {
      byte = this.b[this.pos++];
      result |= (byte & 0x7f) << shift;
      shift += 7;
    } while (byte & 0x80);
    return result >>> 0;
  }
  zigzag() {
    const n = this.varint();
    return (n >>> 1) ^ -(n & 1);
  }
  raw(len) {
    const out = this.b.subarray(this.pos, this.pos + len);
    this.pos += len;
    return out;
  }
}

// ── base64url over raw bytes ─────────────────────────────────────────────────

export function bytesToB64url(u8arr) {
  let bin = '';
  for (const b of u8arr) bin += String.fromCharCode(b);
  return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

export function b64urlToBytes(str) {
  const bin = atob(str.replace(/-/g, '+').replace(/_/g, '/'));
  return Uint8Array.from(bin, (c) => c.charCodeAt(0));
}

// ── container: version + length-delimited sections ──────────────────────────

export function packContainer(sections) {
  const w = new ByteWriter().u8(CODEC_VERSION);
  for (const { tag, bytes } of sections) {
    w.u8(tag).varint(bytes.length).raw(bytes);
  }
  return bytesToB64url(w.take());
}

// Returns { version, sections: Map<tag, Uint8Array> }. Unknown tags are kept in
// the map too (so callers can choose to handle them); truncated input throws.
export function unpackContainer(str) {
  const r = new ByteReader(b64urlToBytes(str));
  const version = r.u8();
  const sections = new Map();
  while (!r.done) {
    const tag = r.u8();
    const len = r.varint();
    sections.set(tag, r.raw(len));
  }
  return { version, sections };
}

// ── FORMULA section ─────────────────────────────────────────────────────────

const idToKey = new Map(OPERATORS.map((o) => [o.id, o.key]));
const keyToId = new Map(OPERATORS.map((o) => [o.key, o.id]));

export function encodeFormula(f) {
  const w = new ByteWriter();
  const flags =
    (f.addC ? 1 : 0) | (f.julia ? 2 : 0) | (((f.deOption ?? 0) & 0x3) << 2);
  w.u8(flags).varint(f.iters ?? 8);
  const cam = f.camera || {};
  w.zigzag(q(cam.yawDeg, CAM_ANGLE))
    .zigzag(q(cam.pitchDeg, CAM_ANGLE))
    .zigzag(q(cam.dist, CAM_DIST))
    .zigzag(q(cam.fovDeg, CAM_ANGLE));
  if (f.julia) {
    const c = f.juliaC || [0, 0, 0];
    w.zigzag(q(c[0], JULIA)).zigzag(q(c[1], JULIA)).zigzag(q(c[2], JULIA));
  }
  const ops = f.ops || [];
  w.varint(ops.length);
  for (const op of ops) {
    const id = keyToId.get(op.key);
    // Unknown key can't be encoded — skip it (the op-list stays valid).
    if (id === undefined) continue;
    const vals = op.values || [];
    w.varint(id).varint(vals.length);
    for (const v of vals) w.zigzag(q(v, PARAM));
  }
  return w.take();
}

export function decodeFormula(bytes) {
  const r = new ByteReader(bytes);
  const flags = r.u8();
  const f = {
    addC: !!(flags & 1),
    julia: !!(flags & 2),
    deOption: (flags >> 2) & 0x3,
    iters: r.varint(),
    camera: {
      yawDeg: dq(r.zigzag(), CAM_ANGLE),
      pitchDeg: dq(r.zigzag(), CAM_ANGLE),
      dist: dq(r.zigzag(), CAM_DIST),
      fovDeg: dq(r.zigzag(), CAM_ANGLE),
    },
    ops: [],
  };
  if (f.julia) {
    f.juliaC = [dq(r.zigzag(), JULIA), dq(r.zigzag(), JULIA), dq(r.zigzag(), JULIA)];
  }
  const opCount = r.varint();
  for (let i = 0; i < opCount; i++) {
    const id = r.varint();
    const n = r.varint();
    const values = [];
    for (let j = 0; j < n; j++) values.push(dq(r.zigzag(), PARAM));
    const key = idToKey.get(id);
    // Unknown opcode (e.g. a newer encoder): we still consumed its params via
    // the stored count, so we can safely drop just this op and keep going.
    if (key !== undefined) f.ops.push({ key, values });
  }
  return f;
}

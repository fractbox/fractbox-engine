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

import { OPERATORS } from "./operators.js";

export const CODEC_VERSION = 1;

export const TAG = {
  FORMULA: 0x01,
  THEME: 0x02, // app: a single varint theme index
  MATRIX: 0x03, // app: the sound matrix
  COLORING: 0x04, // app: the full coloring (mode/colors/palette/light) — overrides THEME
  SPIN: 0x05, // app: auto-spin state (on + speed + axis tilt)
  ASCII: 0x06, // app: ASCII view state (on + density)
  MUTED: 0x07, // app: indices of muted ops (kept in the op-list but not rendered)
  OBJECTS: 0x08, // CSG scene: the objects[] array (multi-object) — see app/src/share.ts.
  //               Rides ALONGSIDE the (unchanged) FORMULA section; old decoders skip
  //               it by length and decode a correct single-object scene (§3.5, B3).
  VIEW: 0x09, // deep zoom: pan target + full-precision dist — see app/src/share.ts.
  //             Rides ALONGSIDE FORMULA (which keeps its own coarse dist, floored
  //             away from 0 on encode so an old decoder still gets a valid, if
  //             shallow, camera instead of a degenerate eye-at-target one) —
  //             docs/design/DEEP_ZOOM.md §7.1.
  SHAPES: 0x0b, // D0 shape leaves: per-object shapeId/params/iterShape — see app/src/share.ts.
  //              Rides ALONGSIDE OBJECTS (v1 body unchanged); old decoders skip it by
  //              length and decode a valid leaf-less scene (mixed objects degrade to
  //              their bare op chain) — PRIMITIVE_DIFS_D0.md §2.5. Emitted ONLY when
  //              some object carries a leaf the v1 body can't express, so legacy
  //              scenes' links stay byte-identical.
  SHAPES2: 0x0c, // #627 leaf-param overflow (sp4..sp7, the objAux lane): per-object
  //              extraCount + zigzag(×1000) extras, index-aligned with OBJECTS. A
  //              SIBLING tag (never wider SHAPES records): shipped applyShapes reads
  //              min(paramCount, 4) WITHOUT skipping extras, so widening SHAPES would
  //              byte-misalign every follow-on record in an old build. Old decoders
  //              skip this tag by container length and get the first-4-params scene.
  //              Emitted ONLY when some object's overflow slot differs from the
  //              leaf's own defaults, so ≤4-param scenes' links stay byte-identical.
  HYBRID: 0x0a, // hybrid iteration: slot B's ops + addC + schedule — see app/src/share.ts.
  //              Rides ALONGSIDE FORMULA (slot A, unchanged); old decoders skip it
  //              by length and decode a valid slot-A-only single-object formula
  //              (graceful degradation) — docs/design/HYBRID_ITERATION.md §3.7.
  //              NOTE: the design doc originally proposed 0x09 for this tag, but
  //              deep zoom's TAG.VIEW claimed it first in this session — 0x0a is
  //              the actual next-free tag as of this implementation.
  HYBRID_N: 0x0c, // N-slot (≥3) hybrid: the FULL slot list (incl. A) + schedule
  //              counts — see kit/share.ts. Emitted INSTEAD of TAG.HYBRID when a
  //              formula has ≥3 slots; a 2-slot formula still rides TAG.HYBRID
  //              byte-for-byte. Old decoders skip this unknown tag by length and
  //              decode a valid FLAT (slot-A-only) formula — HYBRID_NSLOT_SPEC.md
  //              §2.2. (0x0b is SHAPES; 0x0c is SHAPES2; 0x0d is the next free id.)
  MODULATORS: 0x0d, // app: the LFO modulator bank + the timeline bpm — see kit/share.ts.
  //              A modulator is a property of the LOOK (a slowly-breathing
  //              Mandelbox is a thing you send someone), so it rides links;
  //              keyframe easing does NOT, because an ease curve is a property
  //              of a FLIGHT and flights stay file-shaped — ANIMATION_DEPTH.md
  //              §8.2. Emitted ONLY when some modulator is on, so every link
  //              for the feature nobody touched stays byte-identical (§11.8).
  //              Old decoders skip it by length and render the un-modulated
  //              formula: correct, static, not broken. (0x0e is next free.)
};

// Fixed-point scales (kept here so encode/decode can never disagree).
const CAM_ANGLE = 10; // 0.1° precision for yaw/pitch/fov
const CAM_DIST = 100; // 0.01 precision for distance
const JULIA = 1000; // 0.001 precision for the Julia seed
const PARAM = 100; // 0.01 precision for operator params (matches slider steps)

// Hard caps on decoded counts. Counts ride the wire as varints (up to 2^32-1),
// so an untrusted payload must never be trusted to bound a loop/allocation. Set
// generously above any real formula — sanitize.js applies the true engine limits.
export const MAX_DECODE_OPS = 512;
export const MAX_DECODE_PARAMS = 64;

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
  // Full-precision f64 (deep zoom §7.1 — the recenter offset/dist need more
  // range than any fixed-point scale can give without a ~14-digit integer).
  f64(v) {
    const buf = new ArrayBuffer(8);
    new DataView(buf).setFloat64(0, v || 0, false);
    return this.raw(new Uint8Array(buf));
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
  f64() {
    const bytes = this.raw(8);
    const buf = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + 8);
    return new DataView(buf).getFloat64(0, false);
  }
}

// ── base64url over raw bytes ─────────────────────────────────────────────────

export function bytesToB64url(u8arr) {
  let bin = "";
  for (const b of u8arr) bin += String.fromCharCode(b);
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

export function b64urlToBytes(str) {
  const bin = atob(str.replace(/-/g, "+").replace(/_/g, "/"));
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
// the map too (so callers can choose to handle them). Truncated input does NOT
// throw: ByteReader reads past the end yield undefined (→ zeros) and raw()
// returns a short subarray, so a clipped payload decodes to zero-filled fields —
// harmless, since every decode is fed through sanitize.js before use.
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
  // Deep zoom (§7.1): this section's fixed-point dist (×100) rounds a deep
  // dist to 0, which an old (pre-TAG.VIEW) decoder reads as eye===target — a
  // degenerate camera, not a graceful shallow fallback. Floor what's written
  // HERE (TAG.VIEW, below, carries the real value at full f64 precision) so an
  // old decoder always gets a valid, if wrong-depth, picture.
  const legacyDist = Math.max(cam.dist || 24, 1.2);
  w.zigzag(q(cam.yawDeg, CAM_ANGLE))
    .zigzag(q(cam.pitchDeg, CAM_ANGLE))
    .zigzag(q(legacyDist, CAM_DIST))
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
    f.juliaC = [
      dq(r.zigzag(), JULIA),
      dq(r.zigzag(), JULIA),
      dq(r.zigzag(), JULIA),
    ];
  }
  // Counts come straight off an untrusted URL payload; a varint can declare up
  // to 2^32-1, so cap before looping/allocating (a crafted ~20-byte link could
  // otherwise spin billions of iterations and hang the tab before sanitize runs).
  const opCount = Math.min(r.varint(), MAX_DECODE_OPS);
  for (let i = 0; i < opCount && !r.done; i++) {
    const id = r.varint();
    const n = Math.min(r.varint(), MAX_DECODE_PARAMS);
    const values = [];
    for (let j = 0; j < n && !r.done; j++) values.push(dq(r.zigzag(), PARAM));
    const key = idToKey.get(id);
    // Unknown opcode (e.g. a newer encoder): we still consumed its params via
    // the stored count, so we can safely drop just this op and keep going.
    if (key !== undefined) f.ops.push({ key, values });
  }
  return f;
}

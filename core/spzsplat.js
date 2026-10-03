// Niantic .spz v2 compressed 3D-Gaussian-splat writer (issue #368 — the 102 MB
// solve). The uncompressed INRIA .ply at the 1.5 M cap is ≈ 102 MB — unusable on
// mobile SuperSplat. .spz quantizes every attribute to 1–3 bytes and gzips the
// whole buffer (≈ 10–15× smaller than the float32 .ply), and the format-survey
// spike (docs/planning/SPLAT_COMPRESS_SPIKE.md) found SuperSplat ingests it.
//
// Chosen over SOG because .spz is writable DEPENDENCY-FREE: gzip via the
// platform `CompressionStream` (Node ≥18 + browsers), quantization via pure-JS
// bit-packing — no image codec, so the SAME writer serves the CLI and the app
// (SOG's WebP encode has no dep-free Node path, which would strand the CLI).
// Raw ES, DOM-free — the core invariant. UE takes .ply ONLY (spike row 2), so
// .spz is the WEB/MOBILE compact path; UE keeps the uncompressed .ply.
//
// Byte layout (Niantic spz, format version 2 — the widely-read single-gzip form;
// v3 smallest-three quat and v4 zstd multi-stream are NOT dep-free / not needed):
//   16-byte header, then per-attribute arrays IN THIS ORDER, then gzip the whole:
//     header | positions | alphas | colors | scales | rotations | [sh]
//   header: magic u32 "NGSP" | version u32 =2 | numPoints u32 |
//           shDegree u8 | fractionalBits u8 | flags u8 | reserved u8   (all LE)
//   positions:  3 × 24-bit signed fixed-point, value = round(p · 2^fractionalBits)
//   alphas:     u8 = round(sigmoid(logit) · 255)
//   colors:     u8 = round(f_dc · SH_C0 · 255 + 127.5)   (= round(sRGB · 255))
//   scales:     u8 = round((log-scale + 10) · 16)
//   rotations:  3 × u8 = round(q.xyz · (±127.5) + 127.5)  (w≥0 enforced; w dropped)
//   sh:         degree-0 default ⇒ none (SH0-by-fiat, the authored-material size cut)
//
// pack/unpack are PURE + synchronous (byte-level Node-testable); write/read wrap
// them with the async gzip container.

import { SH_C0 } from "./splatfit.js";

export const SPZ_MAGIC = 0x5053474e; // "NGSP" as a little-endian u32
export const SPZ_VERSION = 2;
export const SPZ_FLAG_ANTIALIASED = 0x1;
const HEADER_BYTES = 16;
const INT24_MAX = 0x7fffff;
const INT24_MIN = -0x800000;

const clampByte = (v) => (v < 0 ? 0 : v > 255 ? 255 : v) | 0;
const sigmoid = (x) => 1 / (1 + Math.exp(-x));
// Inverse sigmoid for the reader; guarded off the 0/1 poles (a byte 0 or 255
// would otherwise decode logit ±Infinity).
const invSigmoid = (y) => {
  const a = y <= 0 ? 0.5 / 255 : y >= 1 ? 254.5 / 255 : y;
  return Math.log(a / (1 - a));
};

// Pick fractionalBits (position fixed-point precision) so no coordinate overflows
// the signed 24-bit range. 12 (⇒ 1/4096 world-unit resolution) is the spz default
// and the common case; only huge exports (|coord| > 2^23/4096 ≈ 2048) step down.
// The value is written into the header, so any conforming reader honours it.
function chooseFractionalBits(pos, count, override) {
  if (override != null) return override | 0;
  let maxAbs = 0;
  for (let i = 0, n = count * 3; i < n; i++) {
    const a = Math.abs(pos[i]);
    if (a > maxAbs) maxAbs = a;
  }
  if (!(maxAbs > 0)) return 12;
  const fb = Math.floor(Math.log2(INT24_MAX / maxAbs));
  return Math.max(0, Math.min(12, fb));
}

// splats: fitSplats output { count, pos, fdc, opacity, scaleLog, rot } (rot = [w,x,y,z]).
// opts: { fractionalBits?, antialiased? }.  Returns the UNCOMPRESSED bytes (Uint8Array):
// header + quantized attribute arrays, ready to gzip. Throws on count === 0 (the §5.3a
// backstop, matching writeSplatPly — never emit a degenerate file).
export function packSpz(splats, opts = {}) {
  const count = splats.count | 0;
  if (count <= 0) throw new RangeError("packSpz: refusing to pack 0 splats");
  const { pos, fdc, opacity, scaleLog, rot } = splats;
  const shDegree = 0; // SH0-by-fiat: the compact default (authored material, no view-dep)
  const fractionalBits = chooseFractionalBits(pos, count, opts.fractionalBits);
  const flags = opts.antialiased ? SPZ_FLAG_ANTIALIASED : 0;
  const scaleFP = Math.pow(2, fractionalBits);

  const posBytes = count * 9;
  const alphaBytes = count;
  const colorBytes = count * 3;
  const scaleBytes = count * 3;
  const rotBytes = count * 3; // v2: 3 bytes (xyz), w reconstructed on read
  const total =
    HEADER_BYTES + posBytes + alphaBytes + colorBytes + scaleBytes + rotBytes;

  const buf = new Uint8Array(total);
  const dv = new DataView(buf.buffer);
  dv.setUint32(0, SPZ_MAGIC, true);
  dv.setUint32(4, SPZ_VERSION, true);
  dv.setUint32(8, count, true);
  buf[12] = shDegree;
  buf[13] = fractionalBits & 0xff;
  buf[14] = flags & 0xff;
  buf[15] = 0; // reserved

  // Array cursors, in spz v2 order: positions, alphas, colors, scales, rotations.
  let po = HEADER_BYTES;
  let ao = po + posBytes;
  let co = ao + alphaBytes;
  let so = co + colorBytes;
  let ro = so + scaleBytes;

  for (let i = 0; i < count; i++) {
    const j3 = 3 * i,
      j4 = 4 * i;
    // positions: 24-bit signed fixed-point, little-endian (two's complement).
    for (let c = 0; c < 3; c++) {
      let fx = Math.round(pos[j3 + c] * scaleFP);
      if (fx > INT24_MAX) fx = INT24_MAX;
      else if (fx < INT24_MIN) fx = INT24_MIN;
      const u = fx & 0xffffff;
      buf[po++] = u & 0xff;
      buf[po++] = (u >> 8) & 0xff;
      buf[po++] = (u >> 16) & 0xff;
    }
    // alpha: sigmoid of the stored logit, ×255.
    buf[ao++] = clampByte(Math.round(sigmoid(opacity[i]) * 255));
    // color: SH0 DC → 8-bit. f_dc · SH_C0 = (sRGB − 0.5), so this is round(sRGB·255).
    buf[co++] = clampByte(Math.round(fdc[j3] * SH_C0 * 255 + 127.5));
    buf[co++] = clampByte(Math.round(fdc[j3 + 1] * SH_C0 * 255 + 127.5));
    buf[co++] = clampByte(Math.round(fdc[j3 + 2] * SH_C0 * 255 + 127.5));
    // scale: log-scale, (x + 10)·16.
    buf[so++] = clampByte(Math.round((scaleLog[j3] + 10) * 16));
    buf[so++] = clampByte(Math.round((scaleLog[j3 + 1] + 10) * 16));
    buf[so++] = clampByte(Math.round((scaleLog[j3 + 2] + 10) * 16));
    // rotation: normalize [w,x,y,z], force w≥0 (q and −q are the same rotation),
    // store xyz as ±127.5·comp + 127.5; the reader rebuilds w = √(1 − |xyz|²).
    let qw = rot[j4],
      qx = rot[j4 + 1],
      qy = rot[j4 + 2],
      qz = rot[j4 + 3];
    const ql = Math.hypot(qw, qx, qy, qz) || 1;
    qw /= ql;
    qx /= ql;
    qy /= ql;
    qz /= ql;
    const s = qw < 0 ? -127.5 : 127.5;
    buf[ro++] = clampByte(Math.round(qx * s + 127.5));
    buf[ro++] = clampByte(Math.round(qy * s + 127.5));
    buf[ro++] = clampByte(Math.round(qz * s + 127.5));
  }
  return buf;
}

// Inverse of packSpz — parse UNCOMPRESSED spz v2 bytes back to a fit-bundle-ish
// shape (Float32Arrays). For --verify and byte-level round-trip tests; NOT the
// gzip container (use readSplatSpz for a whole file). rot is returned [w,x,y,z].
export function unpackSpz(bytes) {
  const buf = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const magic = dv.getUint32(0, true);
  if (magic !== SPZ_MAGIC)
    throw new Error(`unpackSpz: bad magic 0x${magic.toString(16)}`);
  const version = dv.getUint32(4, true);
  if (version !== SPZ_VERSION)
    throw new Error(
      `unpackSpz: unsupported version ${version} (this writer emits ${SPZ_VERSION})`,
    );
  const count = dv.getUint32(8, true);
  const shDegree = buf[12];
  const fractionalBits = buf[13];
  const flags = buf[14];
  const invFP = 1 / Math.pow(2, fractionalBits);

  let po = HEADER_BYTES;
  let ao = po + count * 9;
  let co = ao + count;
  let so = co + count * 3;
  let ro = so + count * 3;

  const pos = new Float32Array(count * 3);
  const fdc = new Float32Array(count * 3);
  const opacity = new Float32Array(count);
  const scaleLog = new Float32Array(count * 3);
  const rot = new Float32Array(count * 4);

  for (let i = 0; i < count; i++) {
    const j3 = 3 * i,
      j4 = 4 * i;
    for (let c = 0; c < 3; c++) {
      let u = buf[po] | (buf[po + 1] << 8) | (buf[po + 2] << 16);
      po += 3;
      if (u & 0x800000) u |= 0xff000000; // sign-extend 24 → 32 bit
      pos[j3 + c] = u * invFP;
    }
    opacity[i] = invSigmoid(buf[ao++] / 255);
    fdc[j3] = (buf[co++] / 255 - 0.5) / SH_C0;
    fdc[j3 + 1] = (buf[co++] / 255 - 0.5) / SH_C0;
    fdc[j3 + 2] = (buf[co++] / 255 - 0.5) / SH_C0;
    scaleLog[j3] = buf[so++] / 16 - 10;
    scaleLog[j3 + 1] = buf[so++] / 16 - 10;
    scaleLog[j3 + 2] = buf[so++] / 16 - 10;
    const rx = buf[ro++] / 127.5 - 1;
    const ry = buf[ro++] / 127.5 - 1;
    const rz = buf[ro++] / 127.5 - 1;
    const rw = Math.sqrt(Math.max(0, 1 - (rx * rx + ry * ry + rz * rz)));
    rot[j4] = rw;
    rot[j4 + 1] = rx;
    rot[j4 + 2] = ry;
    rot[j4 + 3] = rz;
  }
  return {
    version,
    numPoints: count,
    shDegree,
    fractionalBits,
    flags,
    pos,
    fdc,
    opacity,
    scaleLog,
    rot,
  };
}

// gzip / gunzip via the platform CompressionStream (Node ≥18 + browsers) — no
// npm, no node:zlib import, so it holds in the raw-ESM core AND the app worker.
async function gzip(u8) {
  const cs = new CompressionStream("gzip");
  const w = cs.writable.getWriter();
  w.write(u8);
  w.close();
  const ab = await new Response(cs.readable).arrayBuffer();
  return new Uint8Array(ab);
}
async function gunzip(u8) {
  const ds = new DecompressionStream("gzip");
  const w = ds.writable.getWriter();
  w.write(u8);
  w.close();
  const ab = await new Response(ds.readable).arrayBuffer();
  return new Uint8Array(ab);
}

// The whole-file writer: pack → gzip. Returns an ArrayBuffer (what host.deliver /
// writeFileSync take). Async because the gzip container is stream-based.
export async function writeSplatSpz(splats, opts = {}) {
  const packed = packSpz(splats, opts);
  const gz = await gzip(packed);
  return gz.buffer.slice(gz.byteOffset, gz.byteOffset + gz.byteLength);
}

// Whole-file reader: gunzip → unpack. For CLI --verify and tests.
export async function readSplatSpz(buffer) {
  const u8 = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
  const packed = await gunzip(u8);
  return unpackSpz(packed);
}

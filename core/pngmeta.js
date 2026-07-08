// PNG text-chunk read/write — pure, dependency-free (no build, no deflate).
//
// Lets a saved PNG carry the formula/scene as standard tEXt/iTXt chunks so the
// image is self-describing (see docs/design/PNG_METADATA.md). We only ever write
// short, uncompressed text, so there is no zTXt / deflate path — keeping the core
// no-dependency invariant intact.
//
// Contract:
//   embedChunks(png, chunks) — png MUST be a valid PNG (our own canvas.toBlob
//     output); throws on a bad signature.
//   readChunks(png) / readText(png, keyword) — parse UNTRUSTED input (arbitrary
//     dropped files) and MUST NEVER throw: a bad/truncated/non-PNG buffer yields
//     [] / null.

const PNG_SIG = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

// ── CRC-32 (IEEE 802.3, poly 0xEDB88320) over chunk type+data ───────────────
const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

export function crc32(bytes) {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

// ── text helpers ────────────────────────────────────────────────────────────
// PNG keywords and tEXt values are Latin-1; iTXt values are UTF-8.
function latin1Bytes(s) {
  const b = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    if (c > 0xff) throw new Error(`non-Latin-1 char in "${s}"`);
    b[i] = c;
  }
  return b;
}
const latin1Str = (bytes) => {
  let s = '';
  for (let i = 0; i < bytes.length; i++) s += String.fromCharCode(bytes[i]);
  return s;
};
const utf8Bytes = (s) => new TextEncoder().encode(s);
const utf8Str = (bytes) => new TextDecoder().decode(bytes);

// Keyword rules (PNG spec): 1–79 Latin-1 printable chars, no leading/trailing space.
function assertKeyword(kw) {
  if (typeof kw !== 'string' || kw.length < 1 || kw.length > 79)
    throw new Error(`keyword must be 1–79 chars: "${kw}"`);
  if (kw[0] === ' ' || kw[kw.length - 1] === ' ')
    throw new Error(`keyword has leading/trailing space: "${kw}"`);
  for (let i = 0; i < kw.length; i++) {
    const c = kw.charCodeAt(i);
    const printable = (c >= 32 && c <= 126) || (c >= 161 && c <= 255);
    if (!printable) throw new Error(`non-printable char in keyword: "${kw}"`);
  }
}

// ── chunk serialization ─────────────────────────────────────────────────────
function chunkBytes(type, data) {
  const typeBytes = latin1Bytes(type);
  const body = new Uint8Array(typeBytes.length + data.length);
  body.set(typeBytes, 0);
  body.set(data, typeBytes.length);
  const out = new Uint8Array(4 + body.length + 4);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, data.length, false); // length excludes type + crc
  out.set(body, 4);
  dv.setUint32(4 + body.length, crc32(body), false);
  return out;
}

function textChunkData({ type, keyword, text, lang = '', translated = '' }) {
  assertKeyword(keyword);
  const kw = latin1Bytes(keyword);
  if (type === 'tEXt') {
    const val = latin1Bytes(text ?? '');
    const d = new Uint8Array(kw.length + 1 + val.length);
    d.set(kw, 0); // kw already followed by the zeroed separator byte
    d.set(val, kw.length + 1);
    return d;
  }
  if (type === 'iTXt') {
    // keyword \0 compFlag(0) compMethod(0) langTag \0 translatedKeyword \0 utf8text
    const langB = latin1Bytes(lang);
    const transB = utf8Bytes(translated);
    const val = utf8Bytes(text ?? '');
    const d = new Uint8Array(kw.length + 1 + 2 + langB.length + 1 + transB.length + 1 + val.length);
    let o = 0;
    d.set(kw, o); o += kw.length + 1; // \0
    d[o++] = 0; // compression flag: uncompressed
    d[o++] = 0; // compression method
    d.set(langB, o); o += langB.length + 1; // \0
    d.set(transB, o); o += transB.length + 1; // \0
    d.set(val, o);
    return d;
  }
  throw new Error(`unsupported chunk type: ${type}`);
}

function isPng(bytes) {
  if (!bytes || bytes.length < 8) return false;
  for (let i = 0; i < 8; i++) if (bytes[i] !== PNG_SIG[i]) return false;
  return true;
}

// Walk chunks, invoking cb(type, dataStart, len). Stops at IEND. Tolerant: a
// truncated/garbage tail just ends the walk (readers never throw).
function walk(bytes, cb) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let off = 8;
  while (off + 8 <= bytes.length) {
    const len = dv.getUint32(off, false);
    const type = latin1Str(bytes.subarray(off + 4, off + 8));
    const dataStart = off + 8;
    if (dataStart + len + 4 > bytes.length) break; // truncated — bail cleanly
    cb(type, dataStart, len);
    if (type === 'IEND') return;
    off = dataStart + len + 4; // skip data + crc
  }
}

// ── public API ──────────────────────────────────────────────────────────────

// Insert text chunks before the first IDAT (falling back to before IEND if a
// PNG somehow has no IDAT). Placing them ahead of the pixel data keeps the
// metadata where every reader looks — some tools skip text chunks that appear
// after IDAT. Returns a NEW Uint8Array.
export function embedChunks(pngBytes, chunks) {
  if (!isPng(pngBytes)) throw new Error('embedChunks: not a PNG');
  let insertAt = -1;
  let iendStart = -1;
  walk(pngBytes, (type, dataStart) => {
    const chunkStart = dataStart - 8; // back up to the length field
    if (type === 'IDAT' && insertAt < 0) insertAt = chunkStart;
    if (type === 'IEND') iendStart = chunkStart;
  });
  if (insertAt < 0) insertAt = iendStart; // no IDAT — degenerate; sit before IEND
  if (insertAt < 0) throw new Error('embedChunks: no IDAT/IEND chunk');

  const additions = chunks.map((c) => chunkBytes(c.type, textChunkData(c)));
  const addLen = additions.reduce((n, a) => n + a.length, 0);
  const out = new Uint8Array(pngBytes.length + addLen);
  out.set(pngBytes.subarray(0, insertAt), 0);
  let o = insertAt;
  for (const a of additions) { out.set(a, o); o += a.length; }
  out.set(pngBytes.subarray(insertAt), o); // the rest (IDAT…IEND), unchanged
  return out;
}

// All tEXt/iTXt chunks as {type, keyword, text}. [] if not a PNG. Never throws.
export function readChunks(pngBytes) {
  const found = [];
  try {
    if (!isPng(pngBytes)) return found;
    walk(pngBytes, (type, dataStart, len) => {
      if (type !== 'tEXt' && type !== 'iTXt') return;
      const data = pngBytes.subarray(dataStart, dataStart + len);
      const nul = data.indexOf(0);
      if (nul < 0) return;
      const keyword = latin1Str(data.subarray(0, nul));
      if (type === 'tEXt') {
        found.push({ type, keyword, text: latin1Str(data.subarray(nul + 1)) });
        return;
      }
      // iTXt: after keyword\0 come compFlag, compMethod, langTag\0, translated\0, text
      let p = nul + 1;
      const compFlag = data[p++];
      p++; // compression method
      const langEnd = data.indexOf(0, p);
      if (langEnd < 0) return;
      const transEnd = data.indexOf(0, langEnd + 1);
      if (transEnd < 0) return;
      const val = data.subarray(transEnd + 1);
      // We never write compressed iTXt; skip any we can't read rather than guess.
      if (compFlag !== 0) return;
      found.push({ type, keyword, text: utf8Str(val) });
    });
  } catch {
    /* untrusted input — return whatever we gathered, never throw */
  }
  return found;
}

// First tEXt/iTXt value matching `keyword`, or null. Never throws.
export function readText(pngBytes, keyword) {
  for (const c of readChunks(pngBytes)) if (c.keyword === keyword) return c.text;
  return null;
}

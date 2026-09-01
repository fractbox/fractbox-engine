// Zero-tooling test for the PNG text-chunk read/write. Run: node core/pngmeta.test.mjs
// (Named *.test.mjs so it stays out of the apps' served `core/*.js` surface —
// the test stays at the source of truth, never shipped into an app's core copy.)
import assert from 'node:assert/strict';
import { embedChunks, readChunks, readText, crc32 } from './pngmeta.js';

let pass = 0;
const test = (name, fn) => {
  try {
    fn();
    pass++;
  } catch (e) {
    console.error(`✗ ${name}\n  ${e.message}`);
    process.exitCode = 1;
  }
};

const SIG = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
const concat = (...arrs) => {
  const out = new Uint8Array(arrs.reduce((n, a) => n + a.length, 0));
  let o = 0;
  for (const a of arrs) { out.set(a, o); o += a.length; }
  return out;
};

// Build a chunk INDEPENDENTLY of the module (readers don't verify CRC, so a
// placeholder crc is fine for base fixtures).
function rawChunk(type, data = new Uint8Array(0), crc = 0) {
  const t = new TextEncoder().encode(type);
  const out = new Uint8Array(4 + 4 + data.length + 4);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, data.length, false);
  out.set(t, 4);
  out.set(data, 8);
  dv.setUint32(8 + data.length, crc, false);
  return out;
}
// A minimal, structurally-valid PNG: sig + IHDR + IDAT + IEND.
const basePng = () => concat(
  new Uint8Array(SIG),
  rawChunk('IHDR', new Uint8Array(13)),
  rawChunk('IDAT', new Uint8Array([1, 2, 3, 4])),
  rawChunk('IEND'),
);

// First byte offset of a 4-char chunk type in the stream (test data has no
// collisions, so a plain scan is fine).
function indexOfType(png, type) {
  const t = new TextEncoder().encode(type);
  outer: for (let i = 0; i + t.length <= png.length; i++) {
    for (let j = 0; j < t.length; j++) if (png[i + j] !== t[j]) continue outer;
    return i;
  }
  return -1;
}

// Hand-built iTXt data payload (independent of the module's writer), so the
// reader is tested against a chunk it did not itself produce.
function itxtData(keyword, text) {
  const kw = new TextEncoder().encode(keyword);
  const val = new TextEncoder().encode(text);
  const d = new Uint8Array(kw.length + 1 + 2 + 1 + 1 + val.length); // kw \0 flag method lang\0 trans\0 text
  let o = 0;
  d.set(kw, o); o += kw.length + 1; // keyword + \0
  d[o++] = 0; // compression flag: uncompressed
  d[o++] = 0; // compression method
  o += 1;     // empty langTag + \0
  o += 1;     // empty translatedKeyword + \0
  d.set(val, o);
  return d;
}

// ── CRC known vector ────────────────────────────────────────────────────────
test('crc32 matches the well-known empty-IEND vector (0xAE426082)', () => {
  assert.equal(crc32(new TextEncoder().encode('IEND')), 0xae426082);
});

// ── round-trips ─────────────────────────────────────────────────────────────
test('embedChunks → readText round-trips a tEXt value', () => {
  const url = 'https://fractbox.com/#c=ABC-_123';
  const png = embedChunks(basePng(), [{ type: 'tEXt', keyword: 'Fractbox', text: url }]);
  assert.equal(readText(png, 'Fractbox'), url);
});

test('embedChunks → readText round-trips a UTF-8 iTXt value', () => {
  const name = 'Wörld ★ bulb — café';
  const png = embedChunks(basePng(), [{ type: 'iTXt', keyword: 'Title', text: name }]);
  assert.equal(readText(png, 'Title'), name);
});

test('readText finds a keyword written across BOTH tEXt and iTXt (the app usage)', () => {
  const png = embedChunks(basePng(), [
    { type: 'tEXt', keyword: 'Software', text: 'Fractbox (engine 0.2.0)' },
    { type: 'tEXt', keyword: 'Fractbox', text: 'https://fractbox.com/#c=Z' },
    { type: 'iTXt', keyword: 'Title', text: 'My Formula' },
  ]);
  assert.equal(readText(png, 'Software'), 'Fractbox (engine 0.2.0)');
  assert.equal(readText(png, 'Fractbox'), 'https://fractbox.com/#c=Z');
  assert.equal(readText(png, 'Title'), 'My Formula'); // iTXt — a tEXt-only reader fails here
});

// A tEXt-only readText would fail this: the Title lives in a hand-built iTXt
// chunk the module never wrote itself.
test('readText reads a hand-constructed iTXt chunk (not our own output)', () => {
  const png = concat(
    new Uint8Array(SIG),
    rawChunk('IHDR', new Uint8Array(13)),
    rawChunk('iTXt', itxtData('Title', 'héllo ★')),
    rawChunk('IEND'),
  );
  assert.equal(readText(png, 'Title'), 'héllo ★');
});

test('unknown keyword reads null', () => {
  const png = embedChunks(basePng(), [{ type: 'tEXt', keyword: 'Fractbox', text: 'x' }]);
  assert.equal(readText(png, 'Nope'), null);
});

// ── structure preserved ─────────────────────────────────────────────────────
test('embedded image keeps the PNG signature and ends with IEND', () => {
  const png = embedChunks(basePng(), [{ type: 'tEXt', keyword: 'Fractbox', text: 'x' }]);
  for (let i = 0; i < 8; i++) assert.equal(png[i], SIG[i]);
  const tail = new TextDecoder().decode(png.subarray(png.length - 8, png.length - 4));
  assert.equal(tail, 'IEND');
  const kinds = readChunks(png).map((c) => c.keyword);
  assert.deepEqual(kinds, ['Fractbox']);
});

test('text chunk is inserted BEFORE the first IDAT (reader compatibility)', () => {
  const png = embedChunks(basePng(), [{ type: 'tEXt', keyword: 'Fractbox', text: 'x' }]);
  const t = indexOfType(png, 'tEXt');
  const idat = indexOfType(png, 'IDAT');
  assert.ok(t > 0 && idat > 0 && t < idat, `tEXt(${t}) must precede IDAT(${idat})`);
});

// ── untrusted-input contract: never throw ───────────────────────────────────
test('readChunks on non-PNG returns [] (no throw)', () => {
  assert.deepEqual(readChunks(new Uint8Array([1, 2, 3])), []);
  assert.deepEqual(readChunks(new Uint8Array(0)), []);
  assert.equal(readText(new Uint8Array([9, 9, 9]), 'Fractbox'), null);
});

test('readChunks on a truncated PNG returns cleanly (no throw)', () => {
  const png = embedChunks(basePng(), [{ type: 'tEXt', keyword: 'Fractbox', text: 'hello' }]);
  const cut = png.subarray(0, 20); // sig + partial IHDR
  assert.deepEqual(readChunks(cut), []);
});

// ── embed contract: bad input throws ────────────────────────────────────────
test('embedChunks throws on a bad signature', () => {
  assert.throws(() => embedChunks(new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]), []));
});

test('embedChunks rejects an out-of-range keyword', () => {
  assert.throws(() => embedChunks(basePng(), [{ type: 'tEXt', keyword: '', text: 'x' }]));
  assert.throws(() => embedChunks(basePng(), [{ type: 'tEXt', keyword: ' lead', text: 'x' }]));
});

console.log(`pngmeta: ${pass} passed`);

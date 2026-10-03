// core/pngstream.js — the streaming PNG encoder (TILED_EXPORT.md §2.4.3).
//
// We author this file format ourselves, so nothing else in the tree would
// notice if it drifted: a wrong filter byte, a mis-framed IDAT or a bad CRC all
// produce a file that still *looks* like a PNG. The gate is therefore a real
// DECODE — inflate the IDATs with the platform DecompressionStream, un-filter
// the scanlines by the spec's own algorithm, and compare pixels — plus a
// walk that verifies every chunk's CRC independently of the writer.
//
// The un-filter below is written from the PNG spec, NOT from pngstream.js's
// filter table, so a sign error in one side cannot be mirrored in the other.

import test from "node:test";
import assert from "node:assert/strict";
import { createPngStream, memorySink, IDAT_MAX } from "./pngstream.js";
import { readChunks, crc32 } from "./pngmeta.js";

const SIG = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

// ── an independent PNG reader ───────────────────────────────────────────────
function walkChunks(bytes) {
  assert.deepEqual([...bytes.subarray(0, 8)], SIG, "PNG signature");
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = [];
  let p = 8;
  while (p < bytes.length) {
    const len = dv.getUint32(p, false);
    const type = String.fromCharCode(...bytes.subarray(p + 4, p + 8));
    const data = bytes.subarray(p + 8, p + 8 + len);
    const crc = dv.getUint32(p + 8 + len, false);
    assert.equal(
      crc,
      crc32(bytes.subarray(p + 4, p + 8 + len)),
      `CRC of ${type} @${p}`,
    );
    out.push({ type, data, len });
    p += 12 + len;
  }
  assert.equal(p, bytes.length, "chunks exactly cover the file");
  return out;
}

async function inflate(bytes) {
  const ds = new DecompressionStream("deflate");
  const w = ds.writable.getWriter();
  w.write(bytes);
  w.close();
  const parts = [];
  const r = ds.readable.getReader();
  for (;;) {
    const { value, done } = await r.read();
    if (done) break;
    parts.push(value);
  }
  const n = parts.reduce((a, b) => a + b.length, 0);
  const out = new Uint8Array(n);
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.length;
  }
  return out;
}

// Un-filter per PNG spec §9.2, transcribed from the spec's equations rather
// than from the encoder.
function unfilter(raw, W, H, bpp) {
  const stride = W * bpp;
  const out = new Uint8Array(H * stride);
  let p = 0;
  for (let y = 0; y < H; y++) {
    const ft = raw[p++];
    const row = out.subarray(y * stride, (y + 1) * stride);
    const up = y > 0 ? out.subarray((y - 1) * stride, y * stride) : null;
    for (let i = 0; i < stride; i++) {
      const x = raw[p + i];
      const a = i >= bpp ? row[i - bpp] : 0;
      const b = up ? up[i] : 0;
      const c = up && i >= bpp ? up[i - bpp] : 0;
      let v;
      if (ft === 0) v = x;
      else if (ft === 1) v = x + a;
      else if (ft === 2) v = x + b;
      else if (ft === 3) v = x + ((a + b) >> 1);
      else if (ft === 4) {
        const pr = a + b - c;
        const pa = Math.abs(pr - a),
          pb = Math.abs(pr - b),
          pc = Math.abs(pr - c);
        v = x + (pa <= pb && pa <= pc ? a : pb <= pc ? b : c);
      } else throw new Error(`bad filter type ${ft} on row ${y}`);
      row[i] = v & 0xff;
    }
    p += stride;
  }
  assert.equal(p, raw.length, "inflated data is exactly H filtered scanlines");
  return out;
}

async function decode(bytes) {
  const chunks = walkChunks(bytes);
  assert.equal(chunks[0].type, "IHDR", "IHDR is first");
  assert.equal(chunks[chunks.length - 1].type, "IEND", "IEND is last");
  const dv = new DataView(chunks[0].data.buffer, chunks[0].data.byteOffset, 13);
  const W = dv.getUint32(0, false),
    H = dv.getUint32(4, false);
  const depth = chunks[0].data[8],
    ctype = chunks[0].data[9];
  assert.equal(depth, 8);
  const bpp = ctype === 6 ? 4 : 3;
  const idats = chunks.filter((c) => c.type === "IDAT");
  assert.ok(idats.length > 0, "has IDAT");
  // Text before pixels — some readers skip trailing text chunks.
  const firstIdat = chunks.findIndex((c) => c.type === "IDAT");
  for (const [i, c] of chunks.entries())
    if (c.type === "tEXt" || c.type === "iTXt")
      assert.ok(i < firstIdat, `${c.type} precedes IDAT`);
  const z = new Uint8Array(idats.reduce((a, c) => a + c.len, 0));
  let o = 0;
  for (const c of idats) {
    z.set(c.data, o);
    o += c.len;
  }
  return { W, H, ctype, px: unfilter(await inflate(z), W, H, bpp), bpp };
}

async function encode(W, H, rgba, opts = {}, bands = null) {
  const sink = memorySink();
  const s = await createPngStream({ W, H, sink, ...opts });
  if (bands) {
    let y = 0;
    for (const n of bands) {
      await s.writeRows(rgba.subarray(y * W * 4, (y + n) * W * 4), n);
      y += n;
    }
    assert.equal(y, H, "bands cover the image");
  } else {
    await s.writeRows(rgba, H);
  }
  const blob = await s.finish();
  return new Uint8Array(await blob.arrayBuffer());
}

// A deterministic image with structure in both axes (so Sub/Up/Paeth all get
// picked somewhere) plus a noisy patch (so None wins somewhere).
function testImage(W, H) {
  const px = new Uint8Array(W * H * 4);
  let seed = 12345;
  const rnd = () =>
    ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) >>> 16) & 0xff;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = (y * W + x) * 4;
      const noisy = x > W * 0.6 && y > H * 0.6;
      px[i] = noisy ? rnd() : (x * 3) & 0xff;
      px[i + 1] = noisy ? rnd() : (y * 5) & 0xff;
      px[i + 2] = noisy ? rnd() : ((x + y) * 2) & 0xff;
      px[i + 3] = 255 - ((x * 7) & 0x7f);
    }
  }
  return px;
}

for (const filter of ["none", "adaptive"]) {
  test(`pngstream: RGB round-trip through a real decode (filter=${filter})`, async () => {
    const W = 37,
      H = 23; // deliberately odd + prime-ish: no accidental alignment
    const rgba = testImage(W, H);
    const dec = await decode(await encode(W, H, rgba, { filter }));
    assert.equal(dec.W, W);
    assert.equal(dec.H, H);
    assert.equal(dec.ctype, 2, "alpha:false → colour type 2 (RGB)");
    for (let i = 0, j = 0; i < W * H * 4; i += 4, j += 3) {
      assert.equal(dec.px[j], rgba[i], `R @${i / 4}`);
      assert.equal(dec.px[j + 1], rgba[i + 1], `G @${i / 4}`);
      assert.equal(dec.px[j + 2], rgba[i + 2], `B @${i / 4}`);
    }
  });

  test(`pngstream: RGBA round-trip keeps alpha (filter=${filter})`, async () => {
    const W = 16,
      H = 9;
    const rgba = testImage(W, H);
    const dec = await decode(await encode(W, H, rgba, { filter, alpha: true }));
    assert.equal(dec.ctype, 6, "alpha:true → colour type 6 (RGBA)");
    assert.deepEqual([...dec.px], [...rgba]);
  });
}

test("pngstream: a multi-row-band write produces the SAME file as one shot", async () => {
  // The whole point of the encoder. If band boundaries leaked into the filter
  // state (a `prev` row reset at a band edge, say) the pixels would still
  // decode — Up/Paeth would just be predicted against zeros — and only a
  // byte-compare against the single-shot file would notice.
  const W = 40,
    H = 30;
  const rgba = testImage(W, H);
  const one = await encode(W, H, rgba, { filter: "adaptive" });
  for (const bands of [[30], [1, 29], [10, 10, 10], [7, 1, 13, 9], [29, 1]]) {
    const many = await encode(W, H, rgba, { filter: "adaptive" }, bands);
    assert.deepEqual(
      [...many],
      [...one],
      `bands ${bands.join("+")} == single shot`,
    );
  }
});

test("pngstream: text chunks are written inline and read back by pngmeta", async () => {
  const W = 8,
    H = 4;
  const text = [
    { type: "tEXt", keyword: "Software", text: "Fractbox" },
    {
      type: "tEXt",
      keyword: "Fractbox",
      text: "https://fractbox.com/#c=abc123",
    },
    { type: "iTXt", keyword: "Title", text: "Mandelbulb — 16K plate ✦" },
  ];
  const bytes = await encode(W, H, testImage(W, H), { text });
  const got = readChunks(bytes);
  assert.deepEqual(
    got.map((c) => [c.type, c.keyword, c.text]),
    text.map((c) => [c.type, c.keyword, c.text]),
    "the existing reader sees exactly what we wrote",
  );
  await decode(bytes); // and the file still decodes with them present
});

test("pngstream: IDATs are split at the cap and re-join into one zlib stream", async () => {
  // A zlib stream may be split at any byte boundary; this pins that we actually
  // split (multiple IDATs) and that the pieces re-join losslessly.
  const W = 64,
    H = 64;
  const rgba = testImage(W, H);
  const small = await encode(W, H, rgba, { filter: "none", idatMax: 512 });
  const chunks = walkChunks(small).filter((c) => c.type === "IDAT");
  assert.ok(chunks.length > 1, `expected several IDATs, got ${chunks.length}`);
  for (const c of chunks) assert.ok(c.len <= 512, `IDAT ${c.len} <= 512`);
  const dec = await decode(small);
  for (let i = 0, j = 0; i < W * H * 4; i += 4, j += 3)
    assert.equal(dec.px[j], rgba[i]);
  assert.equal(IDAT_MAX, 1 << 20, "the shipping cap is 1 MiB");
});

test("pngstream: adaptive filtering actually beats none on structured pixels", async () => {
  const W = 128,
    H = 128;
  const rgba = testImage(W, H);
  const a = await encode(W, H, rgba, { filter: "adaptive" });
  const n = await encode(W, H, rgba, { filter: "none" });
  assert.ok(
    a.length < n.length,
    `adaptive ${a.length} should beat none ${n.length}`,
  );
});

test("pngstream: refuses a short file, a long file, and a short band", async () => {
  const W = 8,
    H = 4;
  const rgba = testImage(W, H);
  const mk = async () =>
    createPngStream({ W, H, sink: memorySink(), filter: "none" });

  let s = await mk();
  await s.writeRows(rgba.subarray(0, W * 2 * 4), 2);
  await assert.rejects(() => s.finish(), /wrote 2 of 4 rows/);

  s = await mk();
  await s.writeRows(rgba, 4);
  await assert.rejects(() => s.writeRows(rgba, 1), /5 rows written/);

  s = await mk();
  await assert.rejects(
    () => s.writeRows(rgba.subarray(0, 8), 4),
    /row band is 8 bytes/,
  );
});

test("pngstream: abort discards and never throws over the original failure", async () => {
  const sink = memorySink();
  const s = await createPngStream({ W: 8, H: 4, sink });
  await s.writeRows(testImage(8, 4).subarray(0, 8 * 2 * 4), 2);
  await s.abort();
  assert.equal((await sink.close()).size, 0, "sink emptied");
  await s.abort(); // idempotent
});

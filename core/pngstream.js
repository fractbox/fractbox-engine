// Streaming PNG encoder — writes a file of ARBITRARY size without ever holding
// it, or any full-size pixel buffer, in memory. TILED_EXPORT.md §2.4.3.
//
// Why this exists: the shipping save path is `canvas.toBlob` → `createImageBitmap`
// → a full-size 2D canvas → a SECOND PNG encode → `embedChunks` (which needs the
// whole file as one Uint8Array). Every one of those steps is O(output) resident,
// and the 2D canvas additionally hits a per-engine canvas-AREA ceiling that
// varies by browser, OS and device memory (§1.9). At 8K/A0/16K none of it works.
//
// Here the caller pushes row bands as they are assembled and we push bytes at the
// sink; peak resident is one row band plus the deflate backlog, flat in the
// output size. Metadata is written INLINE right after IHDR, which is what kills
// the whole-file re-read: we author the file, so we never have to parse it back.
//
// DEPENDENCY-FREE, and it must stay that way (the core no-build invariant):
// deflate comes from the platform `CompressionStream("deflate")` — RFC 1950
// zlib, which is exactly what an IDAT payload is — following the precedent in
// core/spzsplat.js. `crc32` and the chunk framing come from core/pngmeta.js.
//
// The sink is the caller's: any `{ write(u8), close?(), abort?() }` (a File
// System Access writable, an OPFS file, or an array of chunks). Core does not
// choose one — the app does, because the choice is a browser-capability and
// user-consent question, not an engine one.

import { PNG_SIGNATURE, pngChunk, pngTextChunk } from "./pngmeta.js";

// A zlib stream may be split at ANY byte boundary, so IDAT chunk size is purely
// a framing choice: big enough that the per-chunk CRC and 12 bytes of framing
// are noise, small enough that we never hold much.
export const IDAT_MAX = 1 << 20; // 1 MiB
// Filtered scanlines are batched to this before being handed to the compressor —
// one `writer.write()` per row would pay the stream machinery 14 044 times on an
// A0 plate for no benefit.
const FEED_CHUNK = 1 << 20;

// `CompressionStream("deflate")` is Chrome 80+ / Firefox 113+ / Safari 16.4+.
// Absent → the tiled rows are disabled up front rather than failing mid-export
// (TILED_EXPORT.md §4).
export function pngStreamSupported() {
  return typeof CompressionStream === "function";
}

// ── row filters (PNG spec §9) ───────────────────────────────────────────────
// Filtering is what makes deflate effective on photographic-ish data: it turns
// a scanline into residuals against its left/upper neighbours. `none` is the
// correctness baseline the spec mandates; `adaptive` is the standard
// minimum-sum-of-absolute-differences heuristic over None/Sub/Up/Paeth and
// typically halves the file, which is 100+ MB at these sizes. Both are pinned
// by a decode round-trip test, which is the condition §2.4.3 attaches to
// enabling the heuristic.
const paeth = (a, b, c) => {
  const p = a + b - c;
  const pa = Math.abs(p - a),
    pb = Math.abs(p - b),
    pc = Math.abs(p - c);
  return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
};
// Sum of |signed byte| — the standard cost proxy for "how well will this
// deflate", cheap and famously good enough.
const sabs = (v) => (v < 128 ? v : 256 - v);

// Write filter type + filtered bytes of `cur` (against `prev`) into `line`.
function filterInto(line, cur, prev, bpp, mode) {
  const n = cur.length;
  if (mode === "none") {
    line[0] = 0;
    line.set(cur, 1);
    return;
  }
  // One pass to score all four candidates …
  let cNone = 0,
    cSub = 0,
    cUp = 0,
    cPaeth = 0;
  for (let i = 0; i < n; i++) {
    const x = cur[i];
    const a = i >= bpp ? cur[i - bpp] : 0;
    const b = prev[i];
    const c = i >= bpp ? prev[i - bpp] : 0;
    cNone += x;
    cSub += sabs((x - a) & 0xff);
    cUp += sabs((x - b) & 0xff);
    cPaeth += sabs((x - paeth(a, b, c)) & 0xff);
  }
  // … then one pass applying the winner. Cheaper than materialising four rows,
  // and the extra read pass is trivial next to the deflate that follows.
  let best = 0,
    bestCost = cNone;
  if (cSub < bestCost) ((best = 1), (bestCost = cSub));
  if (cUp < bestCost) ((best = 2), (bestCost = cUp));
  if (cPaeth < bestCost) ((best = 4), (bestCost = cPaeth));
  line[0] = best;
  if (best === 0) {
    line.set(cur, 1);
  } else if (best === 1) {
    for (let i = 0; i < n; i++)
      line[i + 1] = (cur[i] - (i >= bpp ? cur[i - bpp] : 0)) & 0xff;
  } else if (best === 2) {
    for (let i = 0; i < n; i++) line[i + 1] = (cur[i] - prev[i]) & 0xff;
  } else {
    for (let i = 0; i < n; i++)
      line[i + 1] =
        (cur[i] -
          paeth(
            i >= bpp ? cur[i - bpp] : 0,
            prev[i],
            i >= bpp ? prev[i - bpp] : 0,
          )) &
        0xff;
  }
}

function ihdrData(W, H, alpha) {
  const d = new Uint8Array(13);
  const dv = new DataView(d.buffer);
  dv.setUint32(0, W, false);
  dv.setUint32(4, H, false);
  d[8] = 8; // bit depth
  d[9] = alpha ? 6 : 2; // colour type: 6 = RGBA, 2 = RGB
  d[10] = 0; // compression: deflate
  d[11] = 0; // filter method: adaptive (per-scanline filter byte)
  d[12] = 0; // interlace: none
  return d;
}

/**
 * Open a streaming PNG. Writes the signature, IHDR and any text chunks
 * immediately, then accepts row bands until `H` rows have been written.
 *
 * @param {object}   o
 * @param {number}   o.W, o.H       image size in pixels
 * @param {boolean}  o.alpha        true → RGBA (colour type 6); false → RGB,
 *                                  which drops 25% of the bytes before deflate
 * @param {Array}    o.text         {type,keyword,text} specs (see pngmeta.js),
 *                                  written inline after IHDR
 * @param {object}   o.sink         { write(u8), close?(), abort?() }
 * @param {string}   o.filter       'adaptive' (default) | 'none'
 * @returns {Promise<{writeRows, finish, abort, rowsWritten}>}
 *
 * `writeRows(rgba, rows)` takes **RGBA** input always (that is what a GPU
 * readback and a 2D canvas both produce); the RGB conversion happens here.
 */
export async function createPngStream({
  W,
  H,
  alpha = false,
  text = [],
  sink,
  filter = "adaptive",
  idatMax = IDAT_MAX,
}) {
  if (!Number.isInteger(W) || !Number.isInteger(H) || W < 1 || H < 1)
    throw new Error(`pngstream: bad size ${W}×${H}`);
  if (!sink || typeof sink.write !== "function")
    throw new Error("pngstream: sink must provide write()");
  if (!pngStreamSupported())
    throw new Error('pngstream: CompressionStream("deflate") is unavailable');
  if (filter !== "adaptive" && filter !== "none")
    throw new Error(`pngstream: unknown filter mode "${filter}"`);

  const bpp = alpha ? 4 : 3;
  const stride = W * bpp;

  await sink.write(Uint8Array.from(PNG_SIGNATURE));
  await sink.write(pngChunk("IHDR", ihdrData(W, H, alpha)));
  for (const t of text) await sink.write(pngTextChunk(t));

  const cs = new CompressionStream("deflate");
  const writer = cs.writable.getWriter();
  const reader = cs.readable.getReader();

  // ── the drain pump ────────────────────────────────────────────────────────
  // The ONLY writer of IDATs, which is what keeps the file's chunk order sound
  // without a sink lock: header (above, before the pump starts) → IDAT* (pump)
  // → IEND (finish, after the pump has ended).
  let pend = new Uint8Array(idatMax);
  let pendLen = 0;
  let failed = null;
  const emit = async () => {
    if (!pendLen) return;
    await sink.write(pngChunk("IDAT", pend.subarray(0, pendLen)));
    pendLen = 0;
  };
  const pump = (async () => {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      let off = 0;
      while (off < value.length) {
        const n = Math.min(idatMax - pendLen, value.length - off);
        pend.set(value.subarray(off, off + n), pendLen);
        pendLen += n;
        off += n;
        if (pendLen === idatMax) await emit();
      }
    }
    await emit();
  })().catch((e) => {
    failed = failed || e;
  });

  // ── the feed side ─────────────────────────────────────────────────────────
  let prev = new Uint8Array(stride);
  let cur = new Uint8Array(stride);
  const line = new Uint8Array(1 + stride);
  const feedCap = Math.max(FEED_CHUNK, line.length);
  let feed = new Uint8Array(feedCap);
  let feedLen = 0;
  let rowsWritten = 0;

  async function flushFeed() {
    if (!feedLen) return;
    // Hand the buffer over rather than copying: allocate a fresh one so the
    // compressor's view can never be overwritten under it.
    const buf = feed.subarray(0, feedLen);
    feed = new Uint8Array(feedCap);
    feedLen = 0;
    await writer.ready;
    await writer.write(buf);
  }

  async function writeRows(rgba, rows) {
    if (failed) throw failed;
    if (!Number.isInteger(rows) || rows < 0)
      throw new Error(`pngstream: bad row count ${rows}`);
    if (rows === 0) return;
    if (rowsWritten + rows > H)
      throw new Error(
        `pngstream: ${rowsWritten + rows} rows written for a ${H}-row image`,
      );
    if (rgba.length < rows * W * 4)
      throw new Error(
        `pngstream: row band is ${rgba.length} bytes, need ${rows * W * 4}`,
      );
    for (let r = 0; r < rows; r++) {
      const src = r * W * 4;
      if (alpha) {
        cur.set(rgba.subarray(src, src + stride));
      } else {
        for (let x = 0, s = src, d = 0; x < W; x++, s += 4, d += 3) {
          cur[d] = rgba[s];
          cur[d + 1] = rgba[s + 1];
          cur[d + 2] = rgba[s + 2];
        }
      }
      filterInto(line, cur, prev, bpp, filter);
      if (feedLen + line.length > feed.length) await flushFeed();
      feed.set(line, feedLen);
      feedLen += line.length;
      // Swap, don't copy: `prev` must be the RAW row, and `cur` is rebuilt from
      // scratch on every iteration, so the two buffers can just trade places.
      const t = prev;
      prev = cur;
      cur = t;
    }
    rowsWritten += rows;
    if (failed) throw failed;
  }

  async function finish() {
    if (failed) throw failed;
    if (rowsWritten !== H)
      throw new Error(`pngstream: wrote ${rowsWritten} of ${H} rows`);
    await flushFeed();
    await writer.close();
    await pump;
    if (failed) throw failed;
    await sink.write(pngChunk("IEND", new Uint8Array(0)));
    return sink.close ? await sink.close() : null;
  }

  // Discard everything. Deliberately swallows: abort runs on a path that is
  // already failing (user cancel, OOM, device lost) and must not add a second
  // exception on top of the first.
  async function abort() {
    try {
      await writer.abort();
    } catch {
      /* the stream may already be errored */
    }
    try {
      await pump;
    } catch {
      /* ditto */
    }
    try {
      if (sink.abort) await sink.abort();
    } catch {
      /* the partial file is best-effort to remove */
    }
  }

  return {
    writeRows,
    finish,
    abort,
    get rowsWritten() {
      return rowsWritten;
    },
  };
}

// An in-memory sink: keeps the emitted chunks as separate Uint8Arrays and lets
// the Blob constructor do the joining, so the file is never concatenated into
// one giant typed array (that copy is one of the three blockers §1.9 names).
// This is the LAST-RESORT sink — the app gates it on an estimated size.
export function memorySink({ type = "image/png" } = {}) {
  let chunks = [];
  return {
    name: "memory",
    write(u8) {
      chunks.push(u8);
    },
    close() {
      const b = new Blob(chunks, { type });
      chunks = [];
      return b;
    },
    abort() {
      chunks = [];
    },
  };
}

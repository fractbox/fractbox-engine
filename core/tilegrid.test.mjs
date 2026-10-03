// Run: node --test core/tilegrid.test.mjs
// TILED_EXPORT.md §3/PR-1 — tile geometry and the off-axis camera window.
//
// The window's vertical term is the part that bites: `ry0` is measured from the
// TOP while ndc.y points UP, so `by` is NOT the mirror of `bx`, and BOTH biases
// vanish for the full frame. A sign error there maps the top tile onto the
// bottom half of the image and still passes every symmetric/identity test — the
// spec carried exactly that bug through a full draft. Hence the orientation and
// off-centre-quadrant assertions below, and hence the reference side of the ray
// test transcribes the SHADER, never the spec's own closed form.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  tileGrid,
  tileWindow,
  rowBandBytes,
  readbackToRGBA,
  evenDims,
  TILE_PX_MAX,
  TILE_PAD,
  TILE_AREA_MAX,
} from "./tilegrid.js";

// ── the reference side: a literal transcription of core/shader.js ────────────
// Read straight off the WGSL (shader.js — `ndc = uv*2-1`, `aspect = res.x/res.y`,
// `rd = fwd + (ndc.x*aspect*tanF)*right + (ndc.y*tanF)*up`) with the vertex
// stage's uv = ((i+0.5)/W, 1 − (j+0.5)/H) substituted in. It knows nothing about
// Lemma 1, about `s = 2·tanF/H`, or about tileWindow(). That independence is the
// point: a reference derived from the same closed form as the implementation
// proves only that the algebra was copied consistently, and would have passed
// with the `by` sign error intact.
function shaderPlane(X, Y, W, H, tanF, win = null) {
  const uvx = (X + 0.5) / W;
  const uvy = 1 - (Y + 0.5) / H;
  const ndcx = uvx * 2 - 1;
  const ndcy = uvy * 2 - 1;
  const aspect = W / H;
  const wx = win ? ndcx * aspect * win.sx + win.bx : ndcx * aspect;
  const wy = win ? ndcy * win.sy + win.by : ndcy;
  return { px: wx * tanF, py: wy * tanF };
}

const TAN_F = Math.tan((0.5 * (42 * Math.PI)) / 180); // core/camera.js default fov

// ── identity ────────────────────────────────────────────────────────────────

test("tileWindow(full frame) is EXACTLY (1,1,0,0) — the bit-identity anchor", () => {
  for (const [W, H] of [
    [1920, 1080],
    [2048, 2048],
    [1080, 1920],
    [9934, 14044],
    [96, 64],
    [2, 2],
  ]) {
    const w = tileWindow(0, 0, W, H, W, H);
    // Object.is, not ==: a −0 bias would multiply through as −0 and is not the
    // same float as the +0 the untiled path never computes at all.
    assert.ok(Object.is(w.sx, 1), `sx for ${W}x${H}`);
    assert.ok(Object.is(w.sy, 1), `sy for ${W}x${H}`);
    assert.ok(Object.is(w.bx, 0), `bx for ${W}x${H}`);
    assert.ok(Object.is(w.by, 0), `by for ${W}x${H}`);
  }
});

test("a 1x1 grid renders the whole frame and its window is the identity", () => {
  const g = tileGrid(2048, 2048);
  assert.equal(g.cols, 1);
  assert.equal(g.rows, 1);
  assert.equal(g.tw, 2048);
  assert.equal(g.th, 2048);
  const t = g.tiles[0];
  assert.deepEqual([t.rx0, t.ry0], [0, 0]);
  assert.deepEqual(tileWindow(t.rx0, t.ry0, g.tw, g.th, 2048, 2048), {
    sx: 1,
    sy: 1,
    bx: 0,
    by: 0,
  });
});

// ── ray identity ────────────────────────────────────────────────────────────

test("ray identity: every tile pixel reproduces the full-frame ray (<1e-12 rel)", () => {
  const cases = [
    { W: 1000, H: 700, cols: 3, rows: 2 },
    { W: 2048, H: 2048, cols: 2, rows: 2 },
    { W: 640, H: 480, cols: 4, rows: 3 },
    { W: 300, H: 900, cols: 1, rows: 5 }, // tall + single column
  ];
  let worstAbs = 0;
  for (const { W, H, cols, rows } of cases) {
    // Hand-built rects here (not tileGrid) so the ray algebra is tested on its
    // own, including rects tileGrid would never emit.
    for (let j = 0; j < rows; j++) {
      for (let i = 0; i < cols; i++) {
        const rw = Math.floor(W / cols);
        const rh = Math.floor(H / rows);
        const rx0 = i * rw;
        const ry0 = j * rh;
        const win = tileWindow(rx0, ry0, rw, rh, W, H);
        // corners, both seam-adjacent columns/rows, and a sampled interior
        const xs = [0, 1, rw - 2, rw - 1, (rw >> 1) - 1, rw >> 1, 7];
        const ys = [0, 1, rh - 2, rh - 1, (rh >> 1) - 1, rh >> 1, 5];
        for (const y of ys) {
          for (const x of xs) {
            const tileSide = shaderPlane(x, y, rw, rh, TAN_F, win);
            const fullSide = shaderPlane(rx0 + x, ry0 + y, W, H, TAN_F);
            const scale = Math.max(
              1e-6,
              Math.abs(fullSide.px),
              Math.abs(fullSide.py),
            );
            const ax = Math.abs(tileSide.px - fullSide.px);
            const ay = Math.abs(tileSide.py - fullSide.py);
            worstAbs = Math.max(worstAbs, ax, ay);
            assert.ok(
              ax / scale < 1e-12,
              `px @${W}x${H} tile(${i},${j}) px(${x},${y}) rel=${ax / scale}`,
            );
            assert.ok(
              ay / scale < 1e-12,
              `py @${W}x${H} tile(${i},${j}) px(${x},${y}) rel=${ay / scale}`,
            );
          }
        }
      }
    }
  }
  // ABSOLUTE, which is what the spec quotes (max |p_full − p_tile| = 1.7e-16 on
  // its 1000x700 case). The relative figure above is inflated near p ≈ 0 by
  // cancellation and is not the honest scale for "is the window exact". If this
  // ever leaves f64 round-off territory the window grew a real error — the
  // pre-correction `by` disagreed by 0.44 on the very first tile.
  assert.ok(worstAbs < 1e-15, `worst absolute disagreement ${worstAbs}`);
});

test("orientation: the TOP tile looks up (p_y > 0), the BOTTOM tile looks down", () => {
  const W = 800;
  const H = 600;
  const rh = 300;
  const top = tileWindow(0, 0, W, rh, W, H); // ry0 = 0 → the TOP half
  const bot = tileWindow(0, 300, W, rh, W, H); // ry0 = 300 → the BOTTOM half
  // Mid-tile pixel: p_y is dominated by the bias, whose sign is the whole
  // question. (It is not the bias alone — the pixel CENTRE sits half a pixel off
  // the tile's midline, so ndc.y is a small non-zero. That half-pixel is exactly
  // what the identity below accounts for and a naive mirror assertion would not.)
  const pyTop = shaderPlane(W / 2, rh / 2, W, rh, TAN_F, top).py;
  const pyBot = shaderPlane(W / 2, rh / 2, W, rh, TAN_F, bot).py;
  assert.ok(pyTop > 0, `top tile must aim ABOVE the axis, got p_y=${pyTop}`);
  assert.ok(pyBot < 0, `bottom tile must aim BELOW the axis, got p_y=${pyBot}`);
  // Each equals the full-frame ray of the SAME absolute pixel — the claim the
  // signs are only a proxy for.
  assert.ok(
    Math.abs(pyTop - shaderPlane(W / 2, rh / 2, W, H, TAN_F).py) < 1e-15,
  );
  assert.ok(
    Math.abs(pyBot - shaderPlane(W / 2, 300 + rh / 2, W, H, TAN_F).py) < 1e-15,
  );
});

test("orientation: an off-centre, NON-SQUARE tile in each quadrant lands right", () => {
  const W = 1000;
  const H = 800;
  const rw = 300;
  const rh = 200; // deliberately not square and not a divisor of W or H
  const quadrants = [
    { name: "top-left", rx0: 40, ry0: 30, sx: -1, sy: +1 },
    { name: "top-right", rx0: 640, ry0: 30, sx: +1, sy: +1 },
    { name: "bottom-left", rx0: 40, ry0: 560, sx: -1, sy: -1 },
    { name: "bottom-right", rx0: 640, ry0: 560, sx: +1, sy: -1 },
  ];
  for (const q of quadrants) {
    const win = tileWindow(q.rx0, q.ry0, rw, rh, W, H);
    const { px, py } = shaderPlane(rw / 2, rh / 2, rw, rh, TAN_F, win);
    assert.equal(Math.sign(px), q.sx, `${q.name}: p_x sign (got ${px})`);
    assert.equal(Math.sign(py), q.sy, `${q.name}: p_y sign (got ${py})`);
    // and the centre ray matches the full-frame ray of the same absolute pixel
    const ref = shaderPlane(q.rx0 + rw / 2, q.ry0 + rh / 2, W, H, TAN_F);
    assert.ok(Math.abs(px - ref.px) < 1e-12, `${q.name}: p_x vs full frame`);
    assert.ok(Math.abs(py - ref.py) < 1e-12, `${q.name}: p_y vs full frame`);
  }
});

test("the PRE-CORRECTION vertical bias would fail the orientation test", () => {
  // Guard on the guard: if by ever reverts to bx's naive mirror,
  // (2·ry0 + rh − H)/H, this is what the top tile does. Recorded so the
  // orientation assertion above can never be "simplified" into a tautology.
  const W = 800;
  const H = 600;
  const rh = 300;
  const naive = { sx: rh / H, sy: rh / H, bx: 0, by: (2 * 0 + rh - H) / H };
  const py = shaderPlane(W / 2, rh / 2, W, rh, TAN_F, naive).py;
  assert.ok(py < 0, "the naive mirror aims the TOP tile downward — the bug");
  assert.ok(
    Math.abs(py) > 0.1,
    "and it is off by 0.44 of a plane unit, not an ulp",
  );
});

// ── committed rects: an exact partition ─────────────────────────────────────

test("committed rects tile [0,W)x[0,H) exactly — no gap, no overlap, exhaustive", () => {
  const cases = [
    [7680, 4320],
    [9934, 14044],
    [15360, 8640],
    [16384, 16384],
    [3840, 2160],
    [4096, 4096],
    [2, 2],
  ];
  for (const [W, H] of cases) {
    const g = tileGrid(W, H);
    const seen = new Uint8Array(g.cols * g.rows);
    // 1-D boundary chains first (cheap, and it is the bandRect contract).
    const xs = [...new Set(g.tiles.map((t) => t.x0))].sort((a, b) => a - b);
    const ys = [...new Set(g.tiles.map((t) => t.y0))].sort((a, b) => a - b);
    assert.equal(xs[0], 0);
    assert.equal(ys[0], 0);
    for (const t of g.tiles) {
      assert.ok(t.x1 > t.x0 && t.y1 > t.y0, `${W}x${H}: empty slice`);
      seen[t.j * g.cols + t.i] = 1;
      if (t.i === g.cols - 1)
        assert.equal(t.x1, W, `${W}x${H}: last col closes on W`);
      if (t.j === g.rows - 1)
        assert.equal(t.y1, H, `${W}x${H}: last row closes on H`);
      else {
        const next = g.tiles.find((u) => u.i === t.i && u.j === t.j + 1);
        assert.equal(t.y1, next.y0, `${W}x${H}: row boundary is shared`);
      }
      if (t.i < g.cols - 1) {
        const next = g.tiles.find((u) => u.j === t.j && u.i === t.i + 1);
        assert.equal(t.x1, next.x0, `${W}x${H}: col boundary is shared`);
      }
    }
    assert.ok(
      seen.every((v) => v === 1),
      `${W}x${H}: every (i,j) present`,
    );
    // total committed area == the frame, which no gap+no overlap alone implies
    const area = g.tiles.reduce((a, t) => a + (t.x1 - t.x0) * (t.y1 - t.y0), 0);
    assert.equal(area, W * H, `${W}x${H}: committed area`);
  }
});

test("committed boundaries and slice sizes are EVEN (the bloom half-res phase)", () => {
  for (const [W, H] of [
    [7680, 4320],
    [9934, 14044],
    [16384, 16384],
  ]) {
    const g = tileGrid(W, H);
    for (const t of g.tiles) {
      assert.equal(t.x0 & 1, 0, "x0 even");
      assert.equal(t.y0 & 1, 0, "y0 even");
      assert.equal(t.x1 & 1, 0, "x1 even");
      assert.equal(t.y1 & 1, 0, "y1 even");
    }
  }
});

// ── rendered rects: uniform, clamped inside, apron respected ────────────────

test("rendered rects: uniform size, even, inside the image, containing their commit", () => {
  for (const [W, H] of [
    [7680, 4320],
    [9934, 14044],
    [15360, 8640],
    [16384, 16384],
    [4096, 4096],
    [3840, 2160],
  ]) {
    const g = tileGrid(W, H);
    assert.equal(g.tw & 1, 0, `${W}x${H}: tw even`);
    assert.equal(g.th & 1, 0, `${W}x${H}: th even`);
    assert.ok(
      g.tw <= TILE_PX_MAX && g.th <= TILE_PX_MAX,
      `${W}x${H}: within the cap`,
    );
    assert.ok(
      g.tw * g.th <= TILE_AREA_MAX,
      `${W}x${H}: tile area under the cap`,
    );
    for (const t of g.tiles) {
      assert.equal(t.rx0 & 1, 0, "rx0 even");
      assert.equal(t.ry0 & 1, 0, "ry0 even");
      assert.ok(
        t.rx0 >= 0 && t.rx0 + g.tw <= W,
        `${W}x${H}: rendered rect inside W`,
      );
      assert.ok(
        t.ry0 >= 0 && t.ry0 + g.th <= H,
        `${W}x${H}: rendered rect inside H`,
      );
      assert.ok(
        t.rx0 <= t.x0 && t.x1 <= t.rx0 + g.tw,
        `${W}x${H}: commit inside render (x)`,
      );
      assert.ok(
        t.ry0 <= t.y0 && t.y1 <= t.ry0 + g.th,
        `${W}x${H}: commit inside render (y)`,
      );
      // Every committed pixel is >= pad from the rendered edge UNLESS that edge
      // is the image edge, where the full frame has the same zero-clamp.
      if (t.x0 - g.pad >= 0 && t.x1 + g.pad <= W) {
        assert.ok(t.x0 - t.rx0 >= g.pad, `${W}x${H}: left apron`);
        assert.ok(t.rx0 + g.tw - t.x1 >= g.pad, `${W}x${H}: right apron`);
      }
      if (t.y0 - g.pad >= 0 && t.y1 + g.pad <= H) {
        assert.ok(t.y0 - t.ry0 >= g.pad, `${W}x${H}: top apron`);
        assert.ok(t.ry0 + g.th - t.y1 >= g.pad, `${W}x${H}: bottom apron`);
      }
    }
  }
});

test("pad 0 (bloom off): the rendered rect IS the committed rect", () => {
  const g = tileGrid(7680, 4320, { pad: 0 });
  for (const t of g.tiles) {
    assert.equal(t.rx0, t.x0);
    assert.equal(t.ry0, t.y0);
    assert.equal(t.x1 - t.x0, g.tw);
    assert.equal(t.y1 - t.y0, g.th);
  }
});

// ── the §2.4.1 table ────────────────────────────────────────────────────────

test("tileGrid reproduces the §2.4.1 table exactly (all five preset rows)", () => {
  const table = [
    // name, W, H, cols, rows, tiles, peak row band MB (2 s.f. as printed)
    ["4K", 3840, 2160, 1, 1, 1, 33],
    ["8K UHD", 7680, 4320, 2, 2, 4, 66],
    ["A0 @300dpi", 9934, 14044, 3, 9, 27, 62],
    ["16K", 15360, 8640, 4, 8, 32, 66],
    ["16384²", 16384, 16384, 5, 16, 80, 67],
  ];
  for (const [name, W, H, cols, rows, tiles, bandMB] of table) {
    const g = tileGrid(W, H);
    assert.equal(g.cols, cols, `${name}: cols`);
    assert.equal(g.rows, rows, `${name}: rows`);
    assert.equal(g.tiles.length, tiles, `${name}: tile count`);
    assert.equal(
      Math.round(rowBandBytes(g) / 1e6),
      bandMB,
      `${name}: peak row band MB`,
    );
  }
  // The spec quotes 8K's largest tile as 3872x2192 = 8.5 Mpx — the biggest any
  // shipping preset produces, and the number TILE_AREA_MAX has to clear.
  const g8k = tileGrid(7680, 4320);
  assert.equal(g8k.tw, 3872);
  assert.equal(g8k.th, 2192);
  assert.ok(g8k.tw * g8k.th < TILE_AREA_MAX);
});

test("A0 @300dpi rounds UP to even: 9933x14043 → 9934x14044", () => {
  assert.deepEqual(evenDims(9933, 14043), { W: 9934, H: 14044 });
  assert.deepEqual(evenDims(3840, 2160), { W: 3840, H: 2160 }); // already even
});

// ── the degenerate case TILE_AREA_MAX exists for ────────────────────────────

test("TILE_AREA_MAX keeps the readback canvas UNDER 2^24, not exactly on it", () => {
  // cols = 1 and a fat row-band budget is the shape that reaches 4096x~4094 —
  // which is 2^24, precisely the iOS/macOS Safari canvas-area ceiling.
  const g = tileGrid(4096, 4096, { rowBandBudget: 1 << 30 });
  assert.ok(
    g.tw * g.th <= TILE_AREA_MAX,
    `tile ${g.tw}x${g.th} over the area cap`,
  );
  assert.ok(g.tw * g.th < 2 ** 24, "and strictly under 2^24");
  assert.ok(g.rows >= 2, "which it achieves by adding a row, not by cropping");
});

test("every tuning constant is an overridable test seam", () => {
  // The seam gate drives the PRODUCTION selector at test scale; with a hard
  // TILE_MIN_H a 64-row image could only ever produce rows = 1 — the shape
  // where the vertical window term is identically zero.
  const g = tileGrid(96, 64, {
    pad: 2,
    tileMax: 64,
    minH: 8,
    rowBandBudget: 8192,
  });
  assert.ok(g.rows >= 2, `minH override must reach 2+ rows, got ${g.rows}`);
  assert.ok(g.cols >= 2, `tileMax override must reach 2+ cols, got ${g.cols}`);
  assert.equal(g.pad, 2);
  const dflt = tileGrid(96, 64);
  assert.equal(dflt.cols, 1, "and the defaults would have collapsed it to 1x1");
  assert.equal(dflt.rows, 1);
});

test("an explicit cols/rows override forces the shape but keeps every invariant", () => {
  // The dev tile probe needs a seam at a chosen place and a size where the real
  // selector answers 1x1. Forcing the SHAPE must not fork the geometry.
  const W = 2048;
  const H = 2048;
  const g = tileGrid(W, H, { cols: 2, rows: 2 });
  assert.equal(g.cols, 2);
  assert.equal(g.rows, 2);
  assert.equal(g.tiles.length, 4);
  assert.equal(g.tw, 1056); // 1024 commit + 2×16 apron
  assert.equal(g.th, 1056);
  const area = g.tiles.reduce((a, t) => a + (t.x1 - t.x0) * (t.y1 - t.y0), 0);
  assert.equal(area, W * H, "still an exact partition");
  for (const t of g.tiles) {
    assert.ok(t.rx0 >= 0 && t.rx0 + g.tw <= W, "still inside the image");
    assert.ok(t.ry0 >= 0 && t.ry0 + g.th <= H);
    assert.ok(
      t.rx0 <= t.x0 && t.x1 <= t.rx0 + g.tw,
      "commit still inside render",
    );
    assert.ok(t.ry0 <= t.y0 && t.y1 <= t.ry0 + g.th);
    assert.equal(t.rx0 & 1, 0);
    assert.equal(t.ry0 & 1, 0);
  }
  // The last row/col's rendered origin is pulled back off the image edge, which
  // is what makes every tile the same size.
  const last = g.tiles.find((t) => t.i === 1 && t.j === 1);
  assert.deepEqual([last.rx0, last.ry0], [992, 992]);
});

test("a forced shape that cannot contain its commit THROWS, it does not crop", () => {
  // Silently returning a tile smaller than its committed rect would drop pixels
  // in the stitch and look like a window bug.
  assert.throws(
    () => tileGrid(2048, 2048, { cols: 1, rows: 1, tileMax: 1024 }),
    /needs a/,
  );
});

test("tileGrid rejects odd or degenerate dimensions rather than silently fixing them", () => {
  assert.throws(() => tileGrid(9933, 14044), /EVEN/);
  assert.throws(() => tileGrid(9934, 14043), /EVEN/);
  assert.throws(() => tileGrid(0, 100), /integers/);
  assert.throws(() => tileGrid(100.5, 100), /integers/);
  assert.throws(() => tileGrid(100, 100, { pad: 15 }), /even/);
});

test("TILE_PAD clears the derived ±11 px bloom support with room to spare", () => {
  assert.equal(
    TILE_PAD & 1,
    0,
    "pad must be even for the half-res phase argument",
  );
  assert.ok(
    TILE_PAD >= 11 + 5,
    "and clear the tightest candidate bound by 5+ px",
  );
});

// ── readbackToRGBA (PR-2) ───────────────────────────────────────────────────
// The tile readback's pixel half. It lives here rather than in renderer.js
// precisely so this can run: nothing in CI has a GPU, and the invariant below is
// the one PR-1's measured failure demands — createImageBitmap on a WebGPU canvas
// silently handed back a fully TRANSPARENT tile for 1 tile in 4, which
// composites as nothing and leaves a missing rectangle in a print plate.

// `ch` rows of `bytesPerRow`, of which the first cw*4 bytes are pixels.
function packRows(rows, bytesPerRow) {
  const buf = new Uint8Array(rows.length * bytesPerRow);
  rows.forEach((r, i) => buf.set(r, i * bytesPerRow));
  return buf;
}

test("readbackToRGBA: an OPAQUE readback is never all-transparent", () => {
  // The load-bearing one. Even where the shader wrote alpha 0 for every pixel
  // (an all-sky tile — perfectly legitimate), the opaque path must hand back
  // 255, so "all transparent" can only ever mean the copy landed nothing.
  const cw = 5,
    ch = 3,
    bpr = 256;
  const rows = [];
  for (let r = 0; r < ch; r++) {
    const row = new Uint8Array(bpr);
    for (let c = 0; c < cw; c++) {
      row[c * 4] = 10 + c; // some real colour…
      row[c * 4 + 1] = 20;
      row[c * 4 + 2] = 30;
      row[c * 4 + 3] = 0; // …with zero alpha throughout
    }
    rows.push(row);
  }
  const { data, blank } = readbackToRGBA(
    packRows(rows, bpr),
    bpr,
    cw,
    ch,
    false,
    false,
  );
  assert.equal(blank, false, "real pixels are not blank");
  for (let i = 0; i < cw * ch; i++)
    assert.equal(data[i * 4 + 3], 255, `pixel ${i} forced opaque`);
});

test("readbackToRGBA: `blank` fires ONLY on an all-zero raw copy", () => {
  const cw = 4,
    ch = 2,
    bpr = 256;
  const zero = packRows([new Uint8Array(bpr), new Uint8Array(bpr)], bpr);
  assert.equal(readbackToRGBA(zero, bpr, cw, ch, false, false).blank, true);

  // A single non-zero byte anywhere in the copied region is enough — including
  // one hiding in the alpha channel of one pixel.
  for (const [row, off] of [
    [0, 0],
    [0, 3],
    [1, cw * 4 - 1],
  ]) {
    const b = zero.slice();
    b[row * bpr + off] = 1;
    assert.equal(
      readbackToRGBA(b, bpr, cw, ch, false, false).blank,
      false,
      `byte ${off} of row ${row} defeats blank`,
    );
  }

  // …but a byte in the ROW PADDING must not, or the guard would never fire on a
  // real buffer (the padding is uninitialised).
  const padded = zero.slice();
  padded[cw * 4] = 0xff; // first padding byte of row 0
  padded[bpr - 1] = 0xff;
  assert.equal(
    readbackToRGBA(padded, bpr, cw, ch, false, false).blank,
    true,
    "256-byte row padding is not pixel data",
  );
});

test("readbackToRGBA: drops row padding, swizzles BGRA, keeps alpha on request", () => {
  const cw = 2,
    ch = 2,
    bpr = 256;
  const mk = (base) => {
    const row = new Uint8Array(bpr).fill(0xee); // padding poison
    for (let c = 0; c < cw; c++) row.set([base + c, 100, 200, 40], c * 4);
    return row;
  };
  const src = packRows([mk(1), mk(50)], bpr);

  // RGBA source, alpha wanted → straight through, tightly packed.
  const rgba = readbackToRGBA(src, bpr, cw, ch, false, true);
  assert.equal(rgba.data.length, cw * ch * 4, "output is tightly packed");
  assert.deepEqual(
    [...rgba.data.subarray(0, 8)],
    [1, 100, 200, 40, 2, 100, 200, 40],
  );
  assert.deepEqual(
    [...rgba.data.subarray(8, 16)],
    [50, 100, 200, 40, 51, 100, 200, 40],
  );

  // BGRA source → R and B trade places, G and A untouched.
  const bgra = readbackToRGBA(src, bpr, cw, ch, true, true);
  assert.deepEqual([...bgra.data.subarray(0, 4)], [200, 100, 1, 40]);
});

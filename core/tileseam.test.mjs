// Run: node --test core/tileseam.test.mjs
// TILED_EXPORT.md §3/PR-1 — THE seam gate.
//
// WGSL is compiled nowhere in CI (#206), so no automated test can execute the
// shader's tile window. The CPU tier mirrors the same algebra (cpu.js traceGrid
// `tile` option), which makes the one claim that matters testable without a GPU:
// render a frame whole, render it again as an N×M tiling, and demand the two be
// BYTE-identical.
//
// Two properties of this test are load-bearing, both of them fixes for ways an
// earlier draft of it could not have failed:
//
//  1. The grid comes from the PRODUCTION tileGrid() selector, driven at test
//     scale through its option seams — not from hand-written rects. A test that
//     hand-rolls its rects proves nothing about the geometry that ships.
//  2. It asserts rows >= 2 AND cols >= 2 BEFORE comparing a single pixel. With
//     a hard TILE_MIN_H = 512 a 64-row image collapses to one row, where the
//     vertical bias `by` is identically zero; the test would then exercise only
//     the horizontal term — which was never wrong — and pass green over a live
//     `by` sign error. That is not hypothetical; it is what the spec's first
//     draft specified.
import { test } from "node:test";
import assert from "node:assert/strict";
import { shadeGrid } from "./cpu.js";
import { tileGrid, tileWindow } from "./tilegrid.js";

// A formula with real structure across the whole frame: a Mandelbox fills the
// field of view and its folds put high-frequency detail on every seam a 3×2
// grid can draw. A sphere or an empty scene would leave most tile boundaries in
// blank background, where any window bug is invisible.
const FORMULA = {
  name: "seam",
  ops: [
    { key: "boxFold", values: [1] },
    { key: "sphereFold", values: [0.5, 1] },
    { key: "scale", values: [2.1] },
  ],
  iters: 9,
  deOption: 2,
  addC: true,
  camera: { yawDeg: 24, pitchDeg: -13, dist: 6.5, fovDeg: 42 },
};

// Everything that is NOT a pure function of the individual ray is off or frozen
// (§2.2 / the shadeGrid header): autoLevels is a whole-grid reduction, edges and
// structure are neighbourhood operators, and dither keys off the LOCAL row/col.
// All four default off — named here so a future default flip fails loudly rather
// than quietly making this test a lie.
const BASE = {
  ss: 1,
  edges: false,
  structure: false,
  dither: false,
  coloring: { mode: 1, autoLevels: false },
};

// Render one rect of the W×H frame. `aspect` is passed EXPLICITLY on both sides:
// cpu.js defaults to the character-cell cols/(2*rows), but the window is derived
// against the GPU tiers' square-pixel res.x/res.y (see traceGrid's `tile` doc).
function renderRect(W, H, rx0, ry0, rw, rh) {
  const tile =
    rw === W && rh === H && rx0 === 0 && ry0 === 0
      ? null
      : tileWindow(rx0, ry0, rw, rh, W, H);
  return shadeGrid(FORMULA, {
    ...BASE,
    cols: rw,
    rows: rh,
    aspect: rw / rh,
    tile,
  });
}

// Stitch the committed sub-rect of each tile into a full-frame buffer.
function stitch(W, H, grid) {
  const chars = new Array(W * H).fill(null);
  const rgb = new Array(W * H).fill(undefined);
  for (const t of grid.tiles) {
    const g = renderRect(W, H, t.rx0, t.ry0, grid.tw, grid.th);
    for (let y = t.y0; y < t.y1; y++) {
      for (let x = t.x0; x < t.x1; x++) {
        const src = (y - t.ry0) * grid.tw + (x - t.rx0);
        const dst = y * W + x;
        chars[dst] = g.chars[src];
        rgb[dst] = g.rgb[src];
      }
    }
  }
  return { chars, rgb };
}

test("CPU-tier seam identity: a 3x2 tiling is BYTE-identical to the full frame", () => {
  const W = 96;
  const H = 64;
  // The production selector, driven at test scale. pad 0 because the CPU tier
  // has no bloom, so there is no neighbourhood operator to apron against — and
  // with pad 0 the rendered rects ARE the committed rects, which makes any
  // mismatch a window error rather than a crop error.
  // tileMax 34, not 32, to land 3 columns of 32: `usable` subtracts 2 for the
  // even-snap slack (a snapped slice can be 2 px wider than W/cols), so a bare
  // 32 would give usable = 30 and four columns. Same arithmetic that keeps a
  // 16K tile under 4096 — exercised here at 1/100 the scale.
  const grid = tileGrid(W, H, {
    pad: 0,
    tileMax: 34,
    minH: 8,
    rowBandBudget: 32 * W * 4,
  });

  // ── the guard that makes the rest of this test capable of failing ──────────
  assert.ok(grid.rows >= 2, `seam gate needs >= 2 ROWS, got ${grid.rows}`);
  assert.ok(grid.cols >= 2, `seam gate needs >= 2 COLS, got ${grid.cols}`);
  assert.equal(grid.cols, 3, "the 3x2 grid the spec names");
  assert.equal(grid.rows, 2);

  const full = renderRect(W, H, 0, 0, W, H);
  const tiled = stitch(W, H, grid);

  // Glyphs first: a single index into the ramp, so a mismatch localises fast.
  let firstBad = -1;
  for (let i = 0; i < W * H; i++) {
    if (full.chars[i] !== tiled.chars[i]) {
      firstBad = i;
      break;
    }
  }
  assert.equal(
    firstBad,
    -1,
    firstBad < 0
      ? ""
      : `glyph mismatch at (${firstBad % W},${Math.floor(firstBad / W)}): ` +
          `full ${JSON.stringify(full.chars[firstBad])} vs tiled ${JSON.stringify(tiled.chars[firstBad])}`,
  );

  // Then colour, which is the real pixel compare: 8 bits per channel off the
  // shaded normal, so it moves when a ray moves even where the glyph does not.
  let diffs = 0;
  let worst = 0;
  for (let i = 0; i < W * H; i++) {
    const a = full.rgb[i];
    const b = tiled.rgb[i];
    if (a === null || b === null) {
      if (a !== b) diffs++;
      continue;
    }
    for (let k = 0; k < 3; k++) {
      const d = Math.abs(a[k] - b[k]);
      if (d) {
        diffs++;
        worst = Math.max(worst, d);
      }
    }
  }
  assert.equal(
    diffs,
    0,
    `${diffs} channel mismatches, worst |Δ| = ${worst} / 255`,
  );
});

test("the seam gate is not vacuous: a WRONG window makes it fail", () => {
  // Same machinery, one sign flipped — the exact defect the spec shipped in its
  // first draft (`by` as the naive mirror of `bx`). If this test ever passes,
  // the comparison above has stopped comparing anything.
  const W = 96;
  const H = 64;
  const rw = 96;
  const rh = 32;
  const good = tileWindow(0, 0, rw, rh, W, H); // TOP half
  const bad = { ...good, by: -good.by };
  const g1 = shadeGrid(FORMULA, {
    ...BASE,
    cols: rw,
    rows: rh,
    aspect: rw / rh,
    tile: good,
  });
  const g2 = shadeGrid(FORMULA, {
    ...BASE,
    cols: rw,
    rows: rh,
    aspect: rw / rh,
    tile: bad,
  });
  assert.notDeepEqual(
    g1.chars,
    g2.chars,
    "a flipped vertical bias must change the image",
  );
});

test("tile-window ray pins: interior, corner and edge agree to < 1e-15", () => {
  // A second, narrower check on the same window the render above used — this one
  // reads the plane coordinate directly instead of inferring it from shaded
  // pixels, so it localises a failure to the algebra rather than the renderer.
  // The reference transcribes the shader's own steps (uv → ndc → ×aspect×tanF),
  // independently of tileWindow's closed form.
  const W = 1000;
  const H = 700;
  const tanF = Math.tan(0.5 * ((42 * Math.PI) / 180));
  const plane = (X, Y, w, h, win) => {
    const ndcx = ((X + 0.5) / w) * 2 - 1;
    const ndcy = (1 - (Y + 0.5) / h) * 2 - 1;
    const asp = w / h;
    return {
      px: (win ? ndcx * asp * win.sx + win.bx : ndcx * asp) * tanF,
      py: (win ? ndcy * win.sy + win.by : ndcy) * tanF,
    };
  };
  const rects = [
    { name: "interior", rx0: 320, ry0: 210, rw: 340, rh: 240 },
    { name: "top-left corner", rx0: 0, ry0: 0, rw: 340, rh: 240 },
    { name: "bottom-right corner", rx0: 660, ry0: 460, rw: 340, rh: 240 },
    { name: "top edge, off-centre", rx0: 137, ry0: 0, rw: 340, rh: 240 },
    { name: "left edge, off-centre", rx0: 0, ry0: 173, rw: 340, rh: 240 },
    { name: "full frame", rx0: 0, ry0: 0, rw: W, rh: H },
  ];
  let worst = 0;
  for (const r of rects) {
    const win = tileWindow(r.rx0, r.ry0, r.rw, r.rh, W, H);
    for (const [x, y] of [
      [0, 0],
      [r.rw - 1, 0],
      [0, r.rh - 1],
      [r.rw - 1, r.rh - 1],
      [r.rw >> 1, r.rh >> 1],
      [1, r.rh - 2],
    ]) {
      const t = plane(x, y, r.rw, r.rh, win);
      const f = plane(r.rx0 + x, r.ry0 + y, W, H, null);
      worst = Math.max(worst, Math.abs(t.px - f.px), Math.abs(t.py - f.py));
    }
  }
  assert.ok(
    worst < 1e-15,
    `worst |p_full − p_tile| = ${worst} (spec quotes 1.7e-16)`,
  );
});

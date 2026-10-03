// Tiled high-res export — tile geometry + the off-axis camera window.
// docs/planning/TILED_EXPORT.md §2.1 (camera math), §2.3 (geometry), §2.4.4.
//
// PURE: no DOM, no GPU, no imports. Node-testable, and deliberately so — the
// WGSL that consumes this is compiled nowhere in CI (#206), so the algebra
// below is the only part of the tile path a test can reach without a GPU.
//
// The one fact everything rests on (§2.1.1, Lemma 1): the map from an integer
// pixel index to its image-plane coordinate is affine with a SINGLE pitch
// s = 2·tan(fov/2)/H on both axes —
//
//     p_x = +s·(X + 0.5 − W/2)      p_y = −s·(Y + 0.5 − H/2)
//
// with X,Y framebuffer indices (origin top-left, +Y DOWN) while uv.y/ndc.y
// point UP. `s` depends on fov and H only — not on W, not on any sub-rectangle.
// That is why an off-axis sub-frame can reproduce the parent's rays exactly,
// and the sign is why the vertical window term is NOT the mirror of the
// horizontal one.

// ── tuning constants ────────────────────────────────────────────────────────
// EVERY one of these is overridable per call (see tileGrid's options). Not for
// flexibility's sake: the CPU seam test must drive the PRODUCTION selector at
// test scale, and a hard TILE_MIN_H = 512 would collapse a 64-row test image to
// a single row — the shape where the vertical window term is identically zero,
// i.e. exactly where the bug that survived the spec's first draft lived
// (TILED_EXPORT.md §3/PR-1). A gate that cannot reach the failure is not a gate.
export const TILE_PX_MAX = 4096; // per-dimension render cap (= today's STILL_PX_CAP)
export const TILE_PAD = 16; // bloom apron, even, ≥ the ±11 px support (§2.2.1(b))
export const TILE_MIN_H = 512; // below this the apron overhead stops being negligible
export const TILE_AREA_MAX = 12_000_000; // readback canvas stays UNDER 2^24 (§2.4.4)
export const ROW_BAND_BUDGET_BYTES = 64 * 1024 * 1024; // assembly row band (§2.4.3)
export const STILL_PX_CAP_TILED = 16384; // the tiled ceiling; STILL_PX_CAP is unchanged

const evenUp = (n) => n + (n & 1);
const evenDown = (n) => n - (n & 1);
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);

// Tiled exports round each requested dimension UP to even (§2.3.3) so the
// half-res bloom parity argument is unconditional. A0 @300dpi 9933×14043 →
// 9934×14044; every shipping preset is already even.
export function evenDims(w, h) {
  return {
    W: evenUp(Math.max(2, Math.round(w))),
    H: evenUp(Math.max(2, Math.round(h))),
  };
}

// ── the off-axis window ─────────────────────────────────────────────────────
// The rendered rect (rx0, ry0, rw, rh) is in full-frame FRAMEBUFFER coordinates:
// (rx0, ry0) is its TOP-LEFT corner, +Y down — matching bandRect's y0 and
// setScissorRect's origin (renderer.js). Returns the (sx, sy, bx, by) applied as
//
//     wx = ndc.x · aspect · sx + bx        wy = ndc.y · sy + by
//
// in place of `ndc.x · aspect` and `ndc.y`, where `aspect = rw/rh` — the shader's
// own convention. Derivation and the seam-identity proof: §2.1.3–§2.1.4.
//
// NOTE the `by` sign. `ry0` is measured from the TOP and ndc.y points UP, so the
// vertical term carries the flip of Lemma 1. Equivalently by = (2·B + rh − H)/H
// with B = H − ry0 − rh, the rect's offset from the BOTTOM. Both biases vanish
// for the full frame, which is precisely why a sign error here survives every
// identity test and has to be caught by an off-centre orientation assertion.
//
// A caller on the CPU tier must pass the SQUARE-PIXEL aspect (cols/rows), not
// cpu.js's character-cell default cols/(2·rows): sy/by are aspect-independent,
// but bx is expressed in units of `ndc.x · aspect` and would need halving under
// the cell convention. See traceGrid's `tile` option.
export function tileWindow(rx0, ry0, rw, rh, W, H) {
  return {
    // sx = sy because the shader's aspect = rw/rh already carries the
    // horizontal stretch; the window only undoes it and re-scales to the
    // parent frame's pitch.
    sx: rh / H,
    sy: rh / H,
    bx: (2 * rx0 + rw - W) / H,
    by: (H - 2 * ry0 - rh) / H,
  };
}

// Exactly (1,1,0,0) for the full frame — and ×1.0 + 0.0 is exact in IEEE-754,
// so the untiled render stays BIT-identical, an invariant we pin rather than
// hope for (§2.1.3).
export const TILE_WINDOW_IDENTITY = Object.freeze({
  sx: 1,
  sy: 1,
  bx: 0,
  by: 0,
});

// ── the grid ────────────────────────────────────────────────────────────────
// tileGrid(W, H, opts) → { cols, rows, tw, th, pad, W, H, tiles[] }
//
//   tiles[k] = { i, j, x0, y0, x1, y1, rx0, ry0 }
//     [x0,x1) × [y0,y1)  the COMMITTED rect — what lands in the output. The
//                        committed rects tile [0,W)×[0,H) exactly: no gap, no
//                        overlap, exhaustive (the bandRect contract, one
//                        dimension up), with boundaries snapped to even.
//     (rx0, ry0)         the RENDERED origin — the committed origin pushed out
//                        by the apron and clamped inside the image. Every tile
//                        renders at the SAME size (tw, th).
//
// Uniform tile size is not a nicety: the HDR+bloom+accum bundle cache is keyed
// by "w×h" and holds two slots (renderer.js BUNDLES_MAX), so a per-tile size
// would evict and reallocate ~44 B/px of GPU textures on every tile.
export function tileGrid(W, H, opts = {}) {
  const {
    pad = TILE_PAD,
    tileMax = TILE_PX_MAX,
    minH = TILE_MIN_H,
    areaMax = TILE_AREA_MAX,
    rowBandBudget = ROW_BAND_BUDGET_BYTES,
    // Force the grid SHAPE, bypassing the selector but keeping every piece of
    // geometry below (even-snapped boundaries, uniform size, apron, clamped
    // origins). For the dev tile probe, which needs a seam in a specific place
    // at a size where the real selector would answer 1×1 — and for tests that
    // want a shape without reverse-engineering the budget that produces it.
    // The selector itself is exercised by the seam test's option seams, not by
    // these; do NOT use them to skip it.
    cols: colsOpt,
    rows: rowsOpt,
  } = opts;
  if (!Number.isInteger(W) || !Number.isInteger(H) || W < 2 || H < 2)
    throw new Error(`tileGrid: W,H must be integers ≥ 2 (got ${W}×${H})`);
  if (W & 1 || H & 1)
    throw new Error(
      `tileGrid: W,H must be EVEN (got ${W}×${H}) — see evenDims/§2.3.3`,
    );
  if (pad & 1 || pad < 0)
    throw new Error(`tileGrid: pad must be even and ≥ 0 (got ${pad})`);

  // `usable` is the widest COMMITTED slice a tile can hold: the cap minus the
  // apron on both sides, minus 2 for the even-snap slack below. Even-snapping a
  // boundary can make one slice up to 2 px wider than W/cols, so without the −2
  // a maximal grid would push tw past the cap.
  const usable = Math.max(2, tileMax - 2 * pad - 2);
  const cols = colsOpt ?? (W <= tileMax ? 1 : Math.ceil(W / usable));

  // Committed boundaries, snapped to even (§2.3.1). Even boundaries are what
  // make every rendered origin automatically even, which the half-res bloom
  // grid's phase alignment needs (§2.2.1(b)).
  const bounds = (n, total) => {
    const b = new Array(n + 1);
    for (let k = 0; k < n; k++) b[k] = 2 * Math.floor((k * total) / (2 * n));
    b[n] = total;
    return b;
  };
  const xb = bounds(cols, W);
  let maxCommitW = 0;
  for (let i = 0; i < cols; i++)
    maxCommitW = Math.max(maxCommitW, xb[i + 1] - xb[i]);
  const tw = Math.min(W, evenUp(maxCommitW + 2 * pad), tileMax);

  // Rows are a MEMORY decision, not a texture-size one: the assembly holds one
  // W-wide row band at a time (§2.4.3). Two further ceilings apply — `usable`
  // (so th stays under the cap) and the area cap (so the readback canvas stays
  // under 2^24, §2.4.4). The −2·pad−2 mirrors the width slack above.
  const byBand = Math.floor(rowBandBudget / (W * 4));
  const thByArea = evenDown(Math.floor(areaMax / tw));
  const rowCap = Math.max(
    2,
    Math.min(clamp(byBand, minH, usable), thByArea - 2 * pad - 2),
  );
  const rows = rowsOpt ?? Math.max(1, Math.ceil(H / rowCap));

  const yb = bounds(rows, H);
  let maxCommitH = 0;
  for (let j = 0; j < rows; j++)
    maxCommitH = Math.max(maxCommitH, yb[j + 1] - yb[j]);
  const th = Math.min(H, evenUp(maxCommitH + 2 * pad), tileMax);

  // Containment is the invariant everything else rests on: if a committed rect
  // does not fit inside its rendered rect, the stitch silently drops pixels.
  // The selector's arithmetic guarantees it; a FORCED shape can violate it, so
  // fail loudly here rather than produce a grid whose tiles crop each other.
  if (tw < maxCommitW || th < maxCommitH)
    throw new Error(
      `tileGrid: ${cols}×${rows} needs a ${maxCommitW}×${maxCommitH} tile but the ` +
        `${tileMax} cap allows only ${tw}×${th} — raise tileMax or add tiles`,
    );

  // The apron is spent on the INTERIOR side at an image edge: a tile whose
  // apron hung outside the image would render real content where the full
  // frame's blur reads zero, and its border pixels would differ (§2.2.1(b)).
  const tiles = [];
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < cols; i++) {
      tiles.push({
        i,
        j,
        x0: xb[i],
        y0: yb[j],
        x1: xb[i + 1],
        y1: yb[j + 1],
        rx0: clamp(xb[i] - pad, 0, W - tw),
        ry0: clamp(yb[j] - pad, 0, H - th),
      });
    }
  }
  return { cols, rows, tw, th, pad, W, H, tiles };
}

// ── tile readback → tightly-packed RGBA ─────────────────────────────────────
// The pixel half of renderer.createTileTarget().read(), lifted out here because
// it is pure and because the invariant it carries is worth a CI test that no
// GPU can provide.
//
// `src` is a mapped copyTextureToBuffer range: `ch` rows of `bytesPerRow`, of
// which only the first `cw*4` bytes are pixels (WebGPU requires 256-byte row
// alignment, so the tail is padding and must be dropped, not copied).
//
// Returns { data, blank }:
//   • `data` is cw×ch tightly-packed RGBA. `wantAlpha` false forces alpha to
//     255 — the shader writes real hit/miss alpha (#428) and every opaque
//     consumer expects opaque.
//   • `blank` is true when every byte of the RAW copy was zero.
//
// WHY `blank` EXISTS. PR-1 measured `createImageBitmap` on a WebGPU canvas
// returning a fully TRANSPARENT bitmap for 1 tile in 4 — a silently missing
// rectangle, the worst possible defect in a print plate. Reading an owned
// texture removes that failure class, but the guard is kept and made loud.
// Note what makes it sound on the opaque path: alpha is forced to 255 there, so
// a real readback can NEVER be all-transparent, and an all-zero raw buffer can
// only mean the copy landed nothing. On the ALPHA path a fully transparent
// black region is a legitimate render, so the caller must not treat it as an
// error — that asymmetry is the whole reason `blank` reports rather than throws.
export function readbackToRGBA(src, bytesPerRow, cw, ch, bgra, wantAlpha) {
  const out = new Uint8Array(cw * ch * 4);
  let any = 0;
  for (let r = 0; r < ch; r++) {
    for (let c = 0; c < cw; c++) {
      const s = r * bytesPerRow + c * 4,
        d = (r * cw + c) * 4;
      const a = src[s + 3];
      any |= src[s] | src[s + 1] | src[s + 2] | a;
      out[d] = bgra ? src[s + 2] : src[s];
      out[d + 1] = src[s + 1];
      out[d + 2] = bgra ? src[s] : src[s + 2];
      out[d + 3] = wantAlpha ? a : 255;
    }
  }
  return { data: out, blank: any === 0 };
}

// Peak assembly row band in bytes — the tallest committed slice, full width.
// Reported rather than assumed: the even-snap can push one slice up to 2 rows
// past ROW_BAND_BUDGET_BYTES, which is a budget, not a hard cap.
export function rowBandBytes(grid) {
  let maxH = 0;
  for (const t of grid.tiles) maxH = Math.max(maxH, t.y1 - t.y0);
  return grid.W * maxH * 4;
}

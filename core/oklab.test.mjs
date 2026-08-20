// OKLab conversion + N-stop palette sampling (COLORING P0). These pin the JS
// reference that core/cpu.js uses directly and that core/frameparams.js uploads
// to the GPU tiers — the WGSL (shader.js oklabToSrgb/albedoStops) and GLSL
// (shader_gl.js) mirrors use the SAME matrices/walk, so a regression here is a
// three-tier divergence. Run: node --test core/oklab.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";
import { srgbToOklab, oklabToSrgb, sampleStops } from "./oklab.js";

const close = (a, b, eps = 1e-4) => Math.abs(a - b) <= eps;
const rgbClose = (a, b, eps = 1e-4) =>
  a.every((v, i) => close(v, b[i], eps));

test("srgbToOklab ↔ oklabToSrgb round-trips across the cube", () => {
  const colors = [
    [0, 0, 0],
    [1, 1, 1],
    [0.5, 0.5, 0.5],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0.86, 0.46, 0.18], // default colA
    [0.18, 0.62, 0.74], // default colB
    [0.03, 0.03, 0.03], // near-black (the piecewise curve matters here)
  ];
  for (const c of colors) {
    const back = oklabToSrgb(srgbToOklab(c));
    assert.ok(rgbClose(c, back, 1e-4), `round-trip ${c} → ${back}`);
  }
});

test("OKLab midpoint of blue→yellow stays chromatic (beats the sRGB lerp)", () => {
  // The whole point of OKLab: a plain sRGB lerp of pure blue↔yellow passes
  // through dead gray ([.5,.5,.5], spread 0). The OKLab midpoint must retain
  // real chroma — asserted comparatively so it's threshold-independent.
  const stops = [
    { c: [0, 0, 1], p: 0 }, // pure blue
    { c: [1, 1, 0], p: 1 }, // pure yellow
  ];
  const oklabMid = sampleStops(stops, 0.5, false);
  const spread = (v) => Math.max(...v) - Math.min(...v);
  const srgbMidSpread = 0; // (0+1)/2 = .5 in every channel → gray
  assert.ok(
    spread(oklabMid) > srgbMidSpread + 0.08,
    `OKLab midpoint ${oklabMid} (spread ${spread(oklabMid)}) should beat the gray sRGB lerp`,
  );
});

test("endpoints are exact (t=0 → first, t=1 → last)", () => {
  const stops = [
    { c: [0.9, 0.1, 0.2], p: 0 },
    { c: [0.1, 0.3, 0.8], p: 0.5 },
    { c: [0.2, 0.9, 0.4], p: 1 },
  ];
  assert.ok(rgbClose(sampleStops(stops, 0, false), stops[0].c, 1e-4));
  assert.ok(rgbClose(sampleStops(stops, 1, false), stops[2].c, 1e-4));
});

test("unsorted stops are sorted by position before sampling", () => {
  const sorted = [
    { c: [0.9, 0.1, 0.2], p: 0 },
    { c: [0.2, 0.9, 0.4], p: 1 },
  ];
  const shuffled = [sorted[1], sorted[0]];
  assert.ok(
    rgbClose(sampleStops(shuffled, 0.3, false), sampleStops(sorted, 0.3, false)),
  );
});

test("non-cyclic clamps outside the stop range to the endpoints", () => {
  const stops = [
    { c: [0.9, 0.1, 0.2], p: 0.25 },
    { c: [0.2, 0.9, 0.4], p: 0.75 },
  ];
  assert.ok(rgbClose(sampleStops(stops, 0.0, false), stops[0].c, 1e-4));
  assert.ok(rgbClose(sampleStops(stops, 1.0, false), stops[1].c, 1e-4));
});

test("cyclic wrap is continuous across the t=1→0 seam", () => {
  // A cyclic palette must approach the same color from just below 1 and just
  // above 0 (the wrap segment blends last→first both ways).
  const stops = [
    { c: [0.9, 0.1, 0.2], p: 0 },
    { c: [0.2, 0.9, 0.4], p: 0.6 },
  ];
  const nearOne = sampleStops(stops, 0.999, true);
  const nearZero = sampleStops(stops, 0.001, true);
  // Both sit on the wrap segment near its ends → very close to each other.
  assert.ok(rgbClose(nearOne, nearZero, 5e-3), `${nearOne} vs ${nearZero}`);
});

test("cyclic full-span palette (stops at 0 and 1) never returns NaN", () => {
  // Regression: the cyclic wrap divided by (first.p − pl), which is 0 when the
  // stops span the full [0,1] → 0/0 = NaN (the WGSL/GLSL guarded, the JS didn't;
  // caught by the app's cyclic swatch + timeline crossfade — PR-C).
  const stops = [
    { c: [0.9, 0.1, 0.2], p: 0 },
    { c: [0.2, 0.9, 0.4], p: 1 },
  ];
  for (let t = 0; t <= 1; t += 0.05) {
    const c = sampleStops(stops, t, true);
    assert.ok(
      c.every((v) => Number.isFinite(v)),
      `NaN at t=${t}: ${c}`,
    );
  }
});

test("fewer than 2 stops → the single color, empty → black", () => {
  assert.deepEqual(sampleStops([{ c: [0.3, 0.6, 0.9], p: 0 }], 0.5), [0.3, 0.6, 0.9]);
  assert.deepEqual(sampleStops([], 0.5), [0, 0, 0]);
});

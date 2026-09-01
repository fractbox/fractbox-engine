// Pins deriveFrameParams (frameparams.js) — the shared per-frame derivation
// both live renderers pack from. This is the regression net the WGSL⟷GLSL
// hand-mirror never had: the derived values below are the shipped look, and a
// change here is a RENDER change on both tiers, so it must be deliberate.
import { test } from "node:test";
import assert from "node:assert/strict";
import { deriveFrameParams } from "./frameparams.js";

const near = (a, b, eps = 1e-12) => Math.abs(a - b) <= eps;
const near3 = (a, b, eps = 1e-12) =>
  a.length === b.length && a.every((x, i) => near(x, b[i], eps));

test("empty payload → the shipped defaults (old saved colorings upgrade)", () => {
  const d = deriveFrameParams({});
  assert.deepEqual(d.colA, [0.86, 0.46, 0.18]);
  assert.deepEqual(d.colB, [0.18, 0.62, 0.74]);
  assert.deepEqual(d.bg, [0.07, 0.09, 0.15]);
  assert.deepEqual(d.juliaC, [0, 0, 0]);
  assert.equal(d.addGate, 0);
  assert.equal(d.julia, 0);
  assert.equal(d.colorMode, 0);
  assert.equal(d.deScale, 0.85);
  assert.equal(d.deOption, 2);
  assert.equal(d.tNear, 0.02);
  assert.equal(d.tFar, 80.0);
  assert.deepEqual(d.palA, [0.5, 0.5, 0.5]);
  assert.deepEqual(d.palB, [0.5, 0.5, 0.5]);
  assert.deepEqual(d.palC, [1, 1, 1]);
  assert.deepEqual(d.palD, [0, 0.33, 0.67]);
  assert.equal(d.palOn, 0);
  // COLORING P0 — absent palette.stops → the legacy cosine/ramp path (count 0);
  // defaults, built-in presets, and old saves all land here (#239 finding 4).
  assert.deepEqual(d.palStops, []);
  assert.equal(d.palStopCount, 0);
  assert.equal(d.palCyclic, 0);
  // #160 Z-up default key (az 31°, el 40°) + the derived rig off it.
  assert.deepEqual(d.lightDir, [0.395, 0.657, 0.643]);
  assert.equal(d.ambient, 0.16);
  assert.equal(d.rim, 0.45);
  assert.equal(d.gloss, 0);
  assert.equal(d.intensity, 1.0);
  assert.deepEqual(d.keyColor, [1, 1, 1]);
  assert.equal(d.metallic, 0);
  assert.equal(d.shadowK, 17); // shadow 0.5 → 30 − 26·0.5
  assert.equal(d.shadowOn, 1);
  assert.equal(d.ao, 0.55);
  assert.ok(near3(d.fillDir, [-0.395, -0.657, 0.643 * 0.35]));
  assert.equal(d.fill, 0);
  assert.ok(near3(d.backDir, [-0.395, -0.657, -0.643]));
  assert.equal(d.back, 0);
  // P3/P4 macros all opt-in.
  assert.equal(d.sky, 0);
  assert.equal(d.sunGlow, 0);
  assert.equal(d.ground, 0.35);
  assert.equal(d.ibl, 0);
  assert.equal(d.fog, 0);
  assert.equal(d.inScatter, 0);
  assert.equal(d.bloomStrength, 0);
  assert.equal(d.bloomThreshold, 1.0);
  assert.equal(d.bloomOn, false);
  assert.equal(d.exposure, 0);
});

test("top-lit key with fill: fill/back dirs derive from the key (Z-up frame)", () => {
  // Nearly-overhead key: fill flips azimuth (negate x/y) and flattens
  // elevation (z × 0.35); back mirrors the key fully. Shaders normalize.
  const d = deriveFrameParams({
    light: { dir: [0.1, 0.2, 0.97], fill: 0.6, back: 0.25 },
  });
  assert.ok(near3(d.fillDir, [-0.1, -0.2, 0.97 * 0.35]));
  assert.ok(near3(d.backDir, [-0.1, -0.2, -0.97]));
  assert.equal(d.fill, 0.6);
  assert.equal(d.back, 0.25);
});

test("#462: keyColor (+ fillColor/backColor) pass through undefaulted", () => {
  // The reporter saw a RED key-light swatch with no visible tint and asked
  // whether the color even reaches the shader. It does — deriveFrameParams is
  // the one place both GPU tiers read it from (renderer.js gF[100..102],
  // renderer_gl.js uKeyC), so pin the pass-through here: a future refactor
  // that renames/drops the field, or reads `g.keyColor` instead of
  // `g.light.keyColor`, breaks this test before it reaches either shader.
  const d = deriveFrameParams({
    light: {
      keyColor: [1, 0, 0],
      fillColor: [0, 1, 0],
      backColor: [0, 0, 1],
    },
  });
  assert.deepEqual(d.keyColor, [1, 0, 0]);
  assert.deepEqual(d.fillColor, [0, 1, 0]);
  assert.deepEqual(d.backColor, [0, 0, 1]);
});

test("shadow slider → penumbra k (30 hard … 4 very soft) and the on-gate", () => {
  assert.equal(deriveFrameParams({ light: { shadow: 0 } }).shadowK, 30);
  assert.equal(deriveFrameParams({ light: { shadow: 0 } }).shadowOn, 0);
  assert.equal(deriveFrameParams({ light: { shadow: 1 } }).shadowK, 4);
  assert.equal(deriveFrameParams({ light: { shadow: 1 } }).shadowOn, 1);
});

test("sunGlow === false gates the glow disc but keeps sky/IBL (#160)", () => {
  const d = deriveFrameParams({ light: { sky: 0.8, sunGlow: false } });
  assert.equal(d.sky, 0.8);
  assert.equal(d.sunGlow, 0);
  assert.equal(d.ibl, 0.8);
  // …and the default (no sunGlow field) rides the sky macro.
  assert.equal(deriveFrameParams({ light: { sky: 0.8 } }).sunGlow, 0.8);
});

test("lightIndicator (#391) shows the glow disc independently of Sky (sky stays 0)", () => {
  // Sky untouched (default 0, its own atmosphere feature stays opted-out) —
  // the indicator still lights up the shared sunGlow amount on its own.
  const d = deriveFrameParams({ light: { lightIndicator: true } });
  assert.equal(d.sky, 0);
  assert.ok(d.sunGlow > 0);
  // Off by default → byte-identical to a payload that never heard of #391.
  assert.equal(deriveFrameParams({ light: {} }).sunGlow, 0);
  // Whichever source is stronger wins (max, not additive/double-counted): a
  // high Sky value isn't dimmed by the indicator also being off.
  const withSky = deriveFrameParams({
    light: { sky: 0.9, lightIndicator: true },
  });
  assert.equal(withSky.sunGlow, 0.9);
});

test("fog macro expands to density + in-scatter; glow to bloom (×0.8, on-flag)", () => {
  const d = deriveFrameParams({ light: { fog: 0.4, glow: 0.5 } });
  assert.equal(d.fog, 0.4);
  assert.equal(d.inScatter, 0.4);
  assert.ok(near(d.bloomStrength, 0.4)); // 0.5 × 0.8
  assert.equal(d.bloomOn, true);
});

test("deep zoom boosts fog DENSITY past the TFAR_MIN floor; in-scatter stays slider-true", () => {
  // dist = 0.0024 (zoom ×10⁴): floor ratio TFAR_MIN/(dist·TFAR_K) = 8/0.008 = 1000.
  const deep = deriveFrameParams({
    light: { fog: 0.4 },
    tNear: 0.0024 * (0.02 / 24), // dist·TNEAR_K — how the settle passes it
  });
  // comp 1000 > 4 → LOG-DEPTH mode: negative word, |value| = the raw slider
  assert.ok(near(deep.fog, -0.4));
  // in-scatter carries the boosted MAGNITUDE (the lit-haze look), sign-free
  assert.ok(near(deep.inScatter, 400));
  // …and the boost is inert when the floor isn't binding (normal zoom).
  const shallow = deriveFrameParams({
    light: { fog: 0.4 },
    tNear: 24 * (0.02 / 24),
  });
  assert.equal(shallow.fog, 0.4);
  assert.equal(shallow.inScatter, 0.4);
});

test("julia folds into the add-gate (fixed c always added)", () => {
  assert.equal(deriveFrameParams({ addC: false, julia: true }).addGate, 1);
  assert.equal(deriveFrameParams({ addC: true, julia: false }).addGate, 1);
  assert.equal(deriveFrameParams({ addC: false, julia: false }).addGate, 0);
  assert.equal(deriveFrameParams({ julia: true }).julia, 1);
});

test("explicit payload values pass through undefaulted", () => {
  const d = deriveFrameParams({
    colA: [1, 0, 0],
    colorMode: 2,
    deScale: 0.3,
    deOption: 3,
    tNear: 0.001,
    tFar: 5,
    light: { ambient: 0.3, exposure: -1.5, ao: 0 },
  });
  assert.deepEqual(d.colA, [1, 0, 0]);
  assert.equal(d.colorMode, 2);
  assert.equal(d.deScale, 0.3);
  assert.equal(d.deOption, 3);
  assert.equal(d.tNear, 0.001);
  assert.equal(d.tFar, 5);
  assert.equal(d.ambient, 0.3);
  assert.equal(d.exposure, -1.5);
  assert.equal(d.ao, 0); // 0 is a value, not a missing field (?? not ||)
});

test("COLORING P0 — palette.stops derives sorted OKLab words + count/cyclic", () => {
  const d = deriveFrameParams({
    palette: {
      on: true, // `on` is the master switch (PR-C on-gate)
      cyclic: true,
      stops: [
        { c: [0.2, 0.9, 0.4], p: 1 }, // deliberately out of order
        { c: [0.9, 0.1, 0.2], p: 0 },
        { c: [0.1, 0.3, 0.8], p: 0.5 },
      ],
    },
  });
  assert.equal(d.palStopCount, 3);
  assert.equal(d.palCyclic, 1);
  // sorted by position; each packed [L,a,b,p], position preserved in .w
  assert.deepEqual(
    d.palStops.map((s) => s[3]),
    [0, 0.5, 1],
  );
  // first stop's L (OKLab lightness) is finite and in a sane range
  assert.ok(Number.isFinite(d.palStops[0][0]) && d.palStops[0][0] > 0);
});

test("COLORING P0 — a single stop is NOT enough (needs ≥2) → legacy path", () => {
  const d = deriveFrameParams({ palette: { stops: [{ c: [1, 0, 0], p: 0 }] } });
  assert.equal(d.palStopCount, 0);
  assert.deepEqual(d.palStops, []);
});

test("COLORING P0 — palette.on is the master switch: off → stops ignored", () => {
  const stops = [
    { c: [0.9, 0.1, 0.2], p: 0 },
    { c: [0.2, 0.9, 0.4], p: 1 },
  ];
  assert.equal(
    deriveFrameParams({ palette: { on: true, stops } }).palStopCount,
    2,
  );
  assert.equal(
    deriveFrameParams({ palette: { on: false, stops } }).palStopCount,
    0,
  );
});

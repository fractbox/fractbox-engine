// IMGTEX (#631) — user-image environment map + triplanar surface texture.
//
// The load-bearing invariant is the envx contract: both texture flags are
// CODEGEN-gated, so a session that never loads an image emits shader text
// with NOT ONE texture token — byte-identity is the perf doctrine's "prove
// it's free" standard (the #125 lesson). These tests fence the gate, the
// bind-index reservation (9-11; 12-13 stay free for #630), the globals-layout
// append, the variant keying, the shared derivation, sanitize, and the pure
// JS mirrors of the WGSL math (triplanar weights / equirect UV).
//
// Run: node --test core/envmap.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  buildWGSL,
  GLOBALS_WORDS,
  ENVX_WORD,
  EMAP_WORD,
  GLOBALS_WORDS_ALLOC,
} from "./shader.js";
import { buildFragGL, buildSceneFragGL } from "./shader_gl.js";
import { keyFor, wgslOf } from "./renderer.js";
import { deriveFrameParams } from "./frameparams.js";
import { frameFeaturesFor } from "./capturesettle.js";
import { sanitizeColoring } from "./sanitize.js";
import { defaultColoring } from "./coloring.js";
import {
  fitImageDims,
  triplanarWeights,
  equirectUV,
  IMG_MAX_DIM,
  IMG_MAX_DIM_MOBILE,
} from "./envmap.js";

// Code tokens are per-half exclusive; the STRUCT rows (emapU/triU) are shared
// on purpose — both rows emit whenever either flag is on, keeping the uniform
// offsets fixed against the one buffer layout.
const ENVMAP_TOKENS = ["envTex", "envImage", "@binding(9)"];
const SURFTEX_TOKENS = ["triTex", "triplanarImage", "@binding(11)"];
const SAMPLER_TOKEN = "@binding(10)";
const A = [{ key: "boxFold", values: [1] }];
const SCENE = [{ shapeId: 1, objType: "sphere", combine: "union" }];

// ── The gate: off emits nothing, on emits everything ─────────────────────────
test("WGSL: default build carries no IMGTEX token (the free-when-off fence)", () => {
  const src = buildWGSL();
  for (const tok of [
    ...ENVMAP_TOKENS,
    ...SURFTEX_TOKENS,
    SAMPLER_TOKEN,
    "textureSampleLevel(envTex",
    "sampler",
  ])
    assert.ok(!src.includes(tok), `default WGSL leaked ${tok}`);
});

test("WGSL: envx-only build carries no IMGTEX token either", () => {
  const src = buildWGSL({ envx: true });
  for (const tok of [...ENVMAP_TOKENS, ...SURFTEX_TOKENS, SAMPLER_TOKEN])
    assert.ok(!src.includes(tok), `envx WGSL leaked ${tok}`);
});

test("WGSL: envMap build carries the env half (and only it)", () => {
  for (const opts of [
    { envMap: true },
    { envMap: true, ops: [1, 2] },
    { envMap: true, scene: true },
    { envMap: true, envx: true },
  ]) {
    const src = buildWGSL(opts);
    for (const tok of [...ENVMAP_TOKENS, SAMPLER_TOKEN, "emapU"])
      assert.ok(
        src.includes(tok),
        `envMap WGSL (${JSON.stringify(opts)}) missing ${tok}`,
      );
    for (const tok of SURFTEX_TOKENS)
      assert.ok(!src.includes(tok), `envMap WGSL leaked surface token ${tok}`);
  }
});

test("WGSL: surfTex build carries the surface half (and only it)", () => {
  const src = buildWGSL({ surfTex: true });
  for (const tok of [...SURFTEX_TOKENS, SAMPLER_TOKEN, "triU"])
    assert.ok(src.includes(tok), `surfTex WGSL missing ${tok}`);
  for (const tok of ENVMAP_TOKENS)
    assert.ok(!src.includes(tok), `surfTex WGSL leaked env token ${tok}`);
});

test("WGSL: both flags emit both halves and one shared sampler", () => {
  const src = buildWGSL({ envMap: true, surfTex: true });
  for (const tok of [...ENVMAP_TOKENS, ...SURFTEX_TOKENS])
    assert.ok(src.includes(tok), `dual WGSL missing ${tok}`);
  assert.equal(src.split(SAMPLER_TOKEN).length - 1, 1, "sampler declared once");
});

// ── Binding reservation (parallel catoptron sessions partition 9-13) ─────────
test("bindings: 9-11 only; 12-13 stay free for #630", () => {
  const src = buildWGSL({
    envMap: true,
    surfTex: true,
    perturb: true, // 6
    ops: [40], // gnarl carries >3 params → opAux at 7
    scene: true,
    leaves: true,
  });
  assert.ok(src.includes("@binding(9) var envTex"), "env texture at 9");
  assert.ok(src.includes("@binding(10) var imgSamp"), "sampler at 10");
  assert.ok(src.includes("@binding(11) var triTex"), "surface texture at 11");
  assert.ok(!src.includes("@binding(12)"), "12 is reserved for #630");
  assert.ok(!src.includes("@binding(13)"), "13 is reserved for #630");
});

// ── Sampling correctness fences (headless-provable properties) ───────────────
test("WGSL: image taps use textureSampleLevel, never textureSample", () => {
  // envColor is called from non-uniform control flow (hit/miss branches, the
  // reflection term) — implicit-derivative sampling there is invalid WGSL and
  // some drivers only warn. Pin the explicit-LOD form on every tap.
  const src = buildWGSL({ envMap: true, surfTex: true });
  assert.equal(src.match(/textureSampleLevel\((envTex|triTex)/g)?.length, 4);
  assert.ok(!/textureSample\((envTex|triTex)/.test(src), "no implicit-LOD tap");
});

test("WGSL: envMap decouples from the Sky blend (the ENVX stars precedent)", () => {
  const src = buildWGSL({ envMap: true });
  assert.ok(
    src.includes(
      "mix(bgOut, envImage(rd), clamp(G.emapU.x, 0.0, 1.0) * (1.0 - clamp(G.env.x, 0.0, 1.0)))",
    ),
    "missing the (1 − sky) complement on the miss path",
  );
});

// ── Globals layout: append-only tail after the ENVX rows ─────────────────────
test("globals layout: IMGTEX rows append after the ENVX tail", () => {
  assert.equal(GLOBALS_WORDS, 48); // base struct + post PG size — frozen
  assert.equal(ENVX_WORD, 48);
  assert.equal(EMAP_WORD, 51); // emapU=51, triU=52
  // base 48 + envx 3 + imgtex 2 + aurora 3 (P6, 53..55) +
  // CINE GRADE 4 (56..59 — shader.js GRADE_WORD) + CLIP 2 (60..61 —
  // shader.js CLIP_WORD).
  assert.equal(GLOBALS_WORDS_ALLOC, 62);
});

test("struct offsets: an envMap-only build still carries the ENVX rows as padding", () => {
  // emapU sits at word 51 in the BUFFER; the struct can only agree if the
  // three ENVX rows (48-50) are present even when envx is off.
  const src = buildWGSL({ envMap: true });
  for (const row of ["starsU", "bandU", "zenU"])
    assert.ok(src.includes(row), `envMap struct missing padding row ${row}`);
});

// ── Variant keying (renderer.js) ─────────────────────────────────────────────
test("keyFor: envMap/surfTex fork the variant key; wgslOf forwards the flags", () => {
  const base = { numericDE: false, coloring: false, ops: [1] };
  const k0 = keyFor(base);
  const kE = keyFor({ ...base, envMap: true });
  const kS = keyFor({ ...base, surfTex: true });
  const kB = keyFor({ ...base, envMap: true, surfTex: true });
  assert.equal(new Set([k0, kE, kS, kB]).size, 4, "four distinct keys");
  const o = wgslOf({ ...base, envMap: true, surfTex: true });
  assert.equal(o.envMap, true);
  assert.equal(o.surfTex, true);
  assert.equal(wgslOf(base).envMap, false);
});

test("frameFeaturesFor: flags default false (headless capture) and pass through", () => {
  const f = { ops: A, iters: 8 };
  const c = defaultColoring();
  assert.equal(frameFeaturesFor(f, c).envMap, false);
  assert.equal(frameFeaturesFor(f, c).surfTex, false);
  const on = frameFeaturesFor(f, c, { envMap: true, surfTex: true });
  assert.equal(on.envMap, true);
  assert.equal(on.surfTex, true);
});

// ── Shared derivation ────────────────────────────────────────────────────────
test("deriveFrameParams: defaults are amount 1 / bright 1 / rot 0 / scale 1", () => {
  const d = deriveFrameParams({ light: defaultColoring().light });
  assert.equal(d.emapAmt, 1); // inert without a texture — the renderer ANDs presence
  assert.equal(d.emapBright, 1);
  assert.equal(d.emapRot, 0);
  assert.equal(d.triAmt, 1);
  assert.equal(d.triScale, 1);
});

test("deriveFrameParams: slider clamps (amounts 0-1, bright 0-3, scale 0.02-3)", () => {
  const d = deriveFrameParams({
    light: {
      envMapAmount: 7,
      envMapBright: -1,
      envMapRot: 2,
      surfTexAmount: -3,
      surfTexScale: 0,
    },
  });
  assert.equal(d.emapAmt, 1);
  assert.equal(d.emapBright, 0);
  assert.equal(d.emapRot, 1);
  assert.equal(d.triAmt, 0);
  assert.equal(d.triScale, 0.02);
});

// ── Sanitize ─────────────────────────────────────────────────────────────────
test("sanitizeColoring: the five sliders survive with their bounds", () => {
  const c = sanitizeColoring({
    ...defaultColoring(),
    light: {
      ...defaultColoring().light,
      envMapAmount: 0.5,
      envMapBright: 9,
      envMapRot: 0.25,
      surfTexAmount: 0.7,
      surfTexScale: -1,
    },
  });
  assert.equal(c.light.envMapAmount, 0.5);
  assert.equal(c.light.envMapBright, 3); // clamped to the slider max
  assert.equal(c.light.envMapRot, 0.25);
  assert.equal(c.light.surfTexAmount, 0.7);
  assert.equal(c.light.surfTexScale, 0.02); // clamped to the slider min
});

// ── GLSL tier: deliberately untouched (WebGL2 renders with the feature off,
// joining its documented no-accum/DOF/bloom reduced set) ─────────────────────
test("GLSL: no emitter grows a user-image sampler", () => {
  for (const src of [
    buildFragGL(A),
    buildFragGL(A, undefined, undefined, { envx: true }),
    buildSceneFragGL(SCENE),
  ])
    for (const tok of ["envImage", "triplanarImage", "uEnvTex", "uTriTex"])
      assert.ok(!src.includes(tok), `GLSL leaked ${tok}`);
});

// ── Pure helpers (envmap.js) — the JS mirrors of the WGSL math ───────────────
test("triplanarWeights: normalized, non-negative, axis-dominant", () => {
  for (const n of [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0.577, 0.577, 0.577],
    [0.9, 0.3, -0.316],
    [-0.6, 0.64, 0.48],
  ]) {
    const w = triplanarWeights(n);
    assert.ok(Math.abs(w[0] + w[1] + w[2] - 1) < 1e-9, "sums to 1");
    for (const x of w) assert.ok(x >= 0, "non-negative");
    // The dominant normal axis owns the dominant weight.
    const ax = n.map(Math.abs);
    assert.equal(w.indexOf(Math.max(...w)), ax.indexOf(Math.max(...ax)));
  }
  // Sharpening: a 45° edge normal splits evenly; a face normal commits fully.
  const edge = triplanarWeights([Math.SQRT1_2, Math.SQRT1_2, 0]);
  assert.ok(Math.abs(edge[0] - 0.5) < 1e-9 && edge[2] === 0);
  assert.deepEqual(triplanarWeights([0, -1, 0]), [0, 1, 0]);
});

test("triplanarWeights matches the emitted WGSL formula", () => {
  // The WGSL is w = (n²)² normalized — pin the exact expression so the JS
  // mirror and the shader cannot drift apart silently.
  const src = buildWGSL({ surfTex: true });
  assert.ok(src.includes("var w = n * n;"), "WGSL first squaring");
  assert.ok(src.includes("w = w * w;"), "WGSL second squaring");
  assert.ok(src.includes("w = w / (w.x + w.y + w.z);"), "WGSL normalization");
});

test("equirectUV: poles, seam wrap, and the rotation slider", () => {
  // +Z (zenith) → v at the clamped top; −Z → clamped bottom.
  assert.equal(equirectUV([0, 0, 1])[1], 0.001);
  assert.equal(equirectUV([0, 0, -1])[1], 0.999);
  // Horizon → v = 0.5; +X azimuth → u = 0.5 by the atan2 convention.
  const h = equirectUV([1, 0, 0]);
  assert.ok(Math.abs(h[0] - 0.5) < 1e-9 && Math.abs(h[1] - 0.5) < 1e-9);
  // Rotation is a full turn per unit and wraps.
  const r = equirectUV([1, 0, 0], 0.25);
  assert.ok(Math.abs(r[0] - 0.75) < 1e-9);
  assert.ok(
    Math.abs(equirectUV([1, 0, 0], 1.5)[0] - 0.0) < 1e-9 ||
      equirectUV([1, 0, 0], 1.5)[0] > 0.999,
  );
});

test("fitImageDims: caps the long edge, keeps aspect, never upscales", () => {
  assert.deepEqual(fitImageDims(4000, 3000, 2048), { w: 2048, h: 1536 });
  assert.deepEqual(fitImageDims(3000, 4000, 1024), { w: 768, h: 1024 });
  assert.deepEqual(fitImageDims(800, 600, 2048), { w: 800, h: 600 }); // no upscale
  assert.deepEqual(fitImageDims(0, 0, 2048), { w: 1, h: 1 }); // degenerate input
  assert.ok(IMG_MAX_DIM === 2048 && IMG_MAX_DIM_MOBILE === 1024); // the #476 memory budget
});

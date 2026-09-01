// AURORA (ENVX P6) — colored fbm sky: aurora curtains + nebula clouds
// (IDEAS 2026-08-21 "Aurora/nebula ENVX layer" — the roadmap's named
// next-background candidate).
//
// The load-bearing invariant: the layer is CODEGEN-gated, not uniform-gated.
// A look with both amounts at 0 must emit shader text with NOT ONE aurora
// token — that byte-identity is the perf doctrine's "prove it's free" standard
// (the #125 lesson: a never-executing uniform branch still cost Mandelbulb
// +31%). These tests fence the gate on every emitter (WGSL, GLSL flat/hybrid/
// scene, standalone bake), the shared derivation (incl. the hue→color pair),
// the WGSL⟷GLSL fbm parity, the bloom-safe peak constants, sanitize, and the
// share-layout constants.
//
// Run: node --test core/aurora.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  buildWGSL,
  ENVX_WORD,
  EMAP_WORD,
  AUR_WORD,
  GLOBALS_WORDS,
  GLOBALS_WORDS_ALLOC,
} from "./shader.js";
import { buildFragGL, buildSceneFragGL } from "./shader_gl.js";
import { deriveFrameParams } from "./frameparams.js";
import { sanitizeColoring } from "./sanitize.js";
import { exportStandaloneGLSL } from "./exportStandalone.js";
import { defaultColoring } from "./coloring.js";
import { frameFeaturesFor } from "./capturesettle.js";

const AURORA_TOKENS_WGSL = ["aurU", "aurA", "aurB", "auroraSky", "hashAu", "fbmAu"];
const AURORA_TOKENS_GL = ["uAurora", "uNebula", "uAurA", "uAurB", "auroraSky", "hashAu"];
const A = [{ key: "boxFold", values: [1] }];
const B = [{ key: "scale", values: [2] }];
const SCENE = [{ shapeId: 1, objType: "sphere", combine: "union" }];

// ── The gate: off emits nothing, on emits everything ─────────────────────────
test("WGSL: default build carries no AURORA token (the free-when-off fence)", () => {
  const src = buildWGSL();
  for (const tok of AURORA_TOKENS_WGSL)
    assert.ok(!src.includes(tok), `default WGSL leaked ${tok}`);
});

test("WGSL: envx/imgtex builds without aurora carry no AURORA token", () => {
  for (const opts of [
    { envx: true },
    { envMap: true },
    { surfTex: true },
    { envx: true, envMap: true, surfTex: true },
  ]) {
    const src = buildWGSL(opts);
    for (const tok of AURORA_TOKENS_WGSL)
      assert.ok(
        !src.includes(tok),
        `WGSL (${JSON.stringify(opts)}) leaked ${tok}`,
      );
  }
});

test("WGSL: aurora build carries the full layer", () => {
  for (const opts of [
    { aurora: true },
    { aurora: true, ops: [1, 2] },
    { aurora: true, scene: true },
    { aurora: true, hybrid: true },
    { aurora: true, envx: true }, // hashAu must not collide with hashE
    { aurora: true, envMap: true, surfTex: true },
    { aurora: true, capture: true },
    { aurora: true, neon: true },
  ]) {
    const src = buildWGSL(opts);
    for (const tok of AURORA_TOKENS_WGSL)
      assert.ok(
        src.includes(tok),
        `aurora WGSL (${JSON.stringify(opts)}) missing ${tok}`,
      );
  }
});

test("WGSL: aurora+envx defines both hash helpers exactly once each", () => {
  const src = buildWGSL({ aurora: true, envx: true });
  const count = (needle) => src.split(needle).length - 1;
  assert.equal(count("fn hashE("), 1, "hashE must be defined once");
  assert.equal(count("fn hashAu("), 1, "hashAu must be defined once");
});

test("GLSL: default builds carry no AURORA token (flat/hybrid/scene)", () => {
  for (const src of [
    buildFragGL([]),
    buildFragGL(A),
    buildFragGL(A, [{ ops: B }]),
    buildFragGL(A, undefined, undefined, { envx: true }),
    buildSceneFragGL(SCENE),
    buildSceneFragGL(SCENE, { envx: true }),
  ])
    for (const tok of AURORA_TOKENS_GL)
      assert.ok(!src.includes(tok), `default GLSL leaked ${tok}`);
});

test("GLSL: aurora builds carry the full layer (flat/hybrid/scene)", () => {
  for (const src of [
    buildFragGL(A, undefined, undefined, { aurora: true }),
    buildFragGL(A, [{ ops: B }], undefined, { aurora: true }),
    buildFragGL(A, undefined, undefined, { aurora: true, envx: true }),
    buildSceneFragGL(SCENE, { aurora: true }),
    buildSceneFragGL(SCENE, { aurora: true, envx: true }),
  ])
    for (const tok of AURORA_TOKENS_GL)
      assert.ok(src.includes(tok), `aurora GLSL missing ${tok}`);
});

// ── Globals layout — append-only tail contract ───────────────────────────────
test("globals layout: AURORA tail appends after the IMGTEX tail", () => {
  assert.equal(GLOBALS_WORDS, 48); // the base struct + post-shader PG size — frozen
  assert.equal(ENVX_WORD, 48); // untouched
  assert.equal(EMAP_WORD, 51); // untouched
  assert.equal(AUR_WORD, 53); // aurU=53, aurA=54, aurB=55
  // base 48 + envx 3 + imgtex 2 + aurora 3 + CINE GRADE 4 (56..59 — #671's
  // rows land AFTER ours; shader.js GRADE_WORD) + CLIP 2 (60..61 — the
  // cross-section rows land after the grade's; shader.js CLIP_WORD).
  assert.equal(GLOBALS_WORDS_ALLOC, 62);
});

test("aurora variants carry the ENVX+IMGTEX rows as dormant padding", () => {
  // The aurU row sits at fixed word 53 against the ONE buffer layout, so an
  // aurora-only variant must still declare starsU..triU above it.
  const src = buildWGSL({ aurora: true });
  for (const row of ["starsU", "bandU", "zenU", "emapU", "triU", "aurU"])
    assert.ok(src.includes(row), `aurora struct missing padding row ${row}`);
});

// ── WGSL ⟷ GLSL fbm/curtain parity ──────────────────────────────────────────
// The two tiers must paint the same sky. Pin the magic constants in BOTH
// bodies; a change to one emitter without the other fails here.
const AURORA_MAGIC = [
  "157.31, 113.97, 271.89", // hashAu basis (≠ hashE's — no collision by name or value)
  "43758.5453", // hash scale
  "* 2.03 + ", // fbm octave 2 lacunarity…
  "* 4.07 + ", // …and octave 3
  "1.1428571", // octave-sum normalizer (÷0.875)
  "* 4.0,", // curtain fold frequency (azimuth-major anisotropy)
  "* 13.0,", // fine streak-ripple frequency (what makes curtains read vertical)
  "0.32, 0.72,", // fold separation threshold
  "0.25 + 0.75 *", // streak ripple floor/gain
  "0.3, 0.7,", // streak threshold window
  "0.35 + 0.75 *", // fold reach
  "+ 0.3) /", // curtain base BELOW the horizon (stays in frame at shallow pitch)
  "-1.7 *", // vertical exp falloff (sharp floor, diffuse top)
  "* 1.55, 1.0)", // profile renormalizer, capped at 1 (bloom-safe core)
  "rd.z * 2.2", // floor→tip color ramp by height above the horizon
  "* 0.8 * ", // curtain peak 0.8 — UNDER the bloom threshold 1.0 (8e38804)
  "* 2.3 + ", // nebula domain frequency
  "0.3, 0.8,", // nebula cloud threshold
  "* 2.2 - 1.1", // nebula accent tint only in dense cores (mid-blends sag gray)
  "* 0.45 * ", // nebula peak 0.45 — subtler than curtains, bloom-safe
  "-0.35, 0.12,", // the starfield's horizon shape, shared
];
test("aurora math is pinned identically in WGSL and GLSL", () => {
  const wgsl = buildWGSL({ aurora: true });
  const glsl = buildFragGL(A, undefined, undefined, { aurora: true });
  for (const magic of AURORA_MAGIC) {
    assert.ok(wgsl.includes(magic), `WGSL missing aurora constant "${magic}"`);
    assert.ok(glsl.includes(magic), `GLSL missing aurora constant "${magic}"`);
  }
});

// ── Background-composite placement (the perf divergence from the starfield) ──
test("aurora is added at bgOut, never inside envColor", () => {
  const wgsl = buildWGSL({ aurora: true });
  assert.ok(
    wgsl.includes("bgOut = bgOut + auroraSky(rd);"),
    "WGSL missing the bgOut add",
  );
  const envBody = wgsl.slice(
    wgsl.indexOf("fn envColor"),
    wgsl.indexOf("fn ggxSpec"),
  );
  assert.ok(
    !envBody.includes("auroraSky"),
    "auroraSky leaked into envColor — it would then run per IBL/reflection tap",
  );
  const glsl = buildFragGL(A, undefined, undefined, { aurora: true });
  assert.ok(glsl.includes("bgOut += auroraSky(rd);"), "GLSL missing the bgOut add");
  const envBodyGL = glsl.slice(
    glsl.indexOf("vec3 envColor"),
    glsl.indexOf("vec3 ggxSpec"),
  );
  assert.ok(!envBodyGL.includes("auroraSky"), "GLSL auroraSky leaked into envColor");
});

// ── Shared derivation ────────────────────────────────────────────────────────
test("deriveFrameParams: default look derives aurora off with inert words", () => {
  const d = deriveFrameParams({ light: defaultColoring().light });
  assert.equal(d.aurora, false);
  assert.equal(d.auroraAmt, 0);
  assert.equal(d.nebulaAmt, 0);
  assert.equal(d.auroraDrift, 0);
});

test("deriveFrameParams: each amount alone flips the latch; the hue never does", () => {
  assert.equal(deriveFrameParams({ light: { aurora: 0.4 } }).aurora, true);
  assert.equal(deriveFrameParams({ light: { nebula: 0.4 } }).aurora, true);
  assert.equal(deriveFrameParams({ light: { auroraHue: 0.9 } }).aurora, false);
});

test("deriveFrameParams: the default hue derives the green→purple pair", () => {
  const d = deriveFrameParams({ light: { aurora: 1 } });
  const [rA, gA, bA] = d.auroraColA;
  // Floor at hue 0.36 — a green (G dominant, R/B suppressed by saturation).
  assert.ok(gA > rA && gA > bA, `floor not green: ${d.auroraColA}`);
  const [rB, gB, bB] = d.auroraColB;
  // Tip at hue 0.78 — a violet/purple (B ≥ R > G).
  assert.ok(bB > gB && rB > gB, `tip not purple: ${d.auroraColB}`);
  for (const c of [...d.auroraColA, ...d.auroraColB])
    assert.ok(c >= 0 && c <= 1, "derived color out of sRGB range");
});

test("deriveFrameParams: amounts and hue clamp to slider domains", () => {
  const d = deriveFrameParams({ light: { aurora: 9, nebula: -2, auroraHue: 7 } });
  assert.equal(d.auroraAmt, 1);
  assert.equal(d.nebulaAmt, 0);
  // hue clamped to 1 → floor is the hue-1.0 (red) color, still in range
  for (const c of d.auroraColA) assert.ok(c >= 0 && c <= 1);
});

// ── Prewarm prediction mirrors the renderer latch ────────────────────────────
test("frameFeaturesFor predicts the aurora bit from the look", () => {
  const f = { ops: A, iters: 8 };
  assert.equal(frameFeaturesFor(f, defaultColoring(), {}).aurora, false);
  assert.equal(
    frameFeaturesFor(f, { light: { aurora: 0.5 } }, {}).aurora,
    true,
  );
  assert.equal(
    frameFeaturesFor(f, { light: { nebula: 0.5 } }, {}).aurora,
    true,
  );
});

// ── Sanitize (the hostile-wire fence) ────────────────────────────────────────
test("sanitize: aurora scalars clamp to slider domains", () => {
  const out = sanitizeColoring({
    light: { aurora: 99, auroraHue: -5, nebula: 2 },
  });
  assert.equal(out.light.aurora, 1);
  assert.equal(out.light.auroraHue, 0);
  assert.equal(out.light.nebula, 1);
});

test("sanitize: absent aurora fields stay absent (shape-preserving)", () => {
  const out = sanitizeColoring({ light: { ambient: 0.2 } });
  for (const k of ["aurora", "auroraHue", "nebula"])
    assert.ok(!(k in out.light), `sanitize invented light.${k}`);
});

// ── Standalone export ────────────────────────────────────────────────────────
const FORMULA = { ops: A };
test("standalone export: an aurora look bakes the layer consts", () => {
  const g = exportStandaloneGLSL(FORMULA, {
    light: { aurora: 0.6, nebula: 0.3 },
  });
  for (const tok of ["const float uAurora", "const vec3 uAurA", "auroraSky"])
    assert.ok(g.includes(tok), `aurora export missing ${tok}`);
});

test("standalone export: an aurora-off look emits no AURORA token", () => {
  const g = exportStandaloneGLSL(FORMULA, { light: defaultColoring().light });
  for (const tok of AURORA_TOKENS_GL)
    assert.ok(!g.includes(tok), `aurora-off export leaked ${tok}`);
});

// ENVX (backgrounds P5) — starfield / Milky-Way band / zenith color.
//
// The load-bearing invariant: the extension is CODEGEN-gated, not uniform-
// gated. A look with stars/band/zenith all off must emit shader text with NOT
// ONE extension token — that byte-identity is the perf doctrine's "prove it's
// free" standard (the #125 lesson: a never-executing uniform branch still cost
// Mandelbulb +31%). These tests fence the gate on every emitter (WGSL, GLSL
// flat/hybrid/scene, standalone bake), the shared derivation, sanitize, and
// the WGSL⟷GLSL star-math parity.
//
// Run: node --test core/envx.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  buildWGSL,
  ENVX_WORD,
  GLOBALS_WORDS,
  GLOBALS_WORDS_ALLOC,
} from "./shader.js";
import { buildFragGL, buildSceneFragGL } from "./shader_gl.js";
import { deriveFrameParams } from "./frameparams.js";
import { sanitizeColoring } from "./sanitize.js";
import { exportStandaloneGLSL } from "./exportStandalone.js";
import { defaultColoring } from "./coloring.js";

const ENVX_TOKENS_WGSL = ["starsU", "bandU", "zenU", "starField", "hashE"];
const ENVX_TOKENS_GL = ["uStars", "uBand", "uZenC", "uZenOn", "starField", "hashE"];
const A = [{ key: "boxFold", values: [1] }];
const B = [{ key: "scale", values: [2] }];
const SCENE = [{ shapeId: 1, objType: "sphere", combine: "union" }];

// ── The gate: off emits nothing, on emits everything ─────────────────────────
test("WGSL: default build carries no ENVX token (the free-when-off fence)", () => {
  const src = buildWGSL();
  for (const tok of ENVX_TOKENS_WGSL)
    assert.ok(!src.includes(tok), `default WGSL leaked ${tok}`);
});

test("WGSL: envx build carries the full extension", () => {
  for (const opts of [
    { envx: true },
    { envx: true, ops: [1, 2] },
    { envx: true, scene: true },
    { envx: true, hybrid: true },
  ]) {
    const src = buildWGSL(opts);
    for (const tok of ENVX_TOKENS_WGSL)
      assert.ok(src.includes(tok), `envx WGSL (${JSON.stringify(opts)}) missing ${tok}`);
  }
});

test("GLSL: default builds carry no ENVX token", () => {
  for (const src of [
    buildFragGL([]),
    buildFragGL(A),
    buildFragGL(A, [{ ops: B }]),
    buildSceneFragGL(SCENE),
  ])
    for (const tok of ENVX_TOKENS_GL)
      assert.ok(!src.includes(tok), `default GLSL leaked ${tok}`);
});

test("GLSL: envx builds carry the full extension (flat/hybrid/scene)", () => {
  for (const src of [
    buildFragGL(A, undefined, undefined, { envx: true }),
    buildFragGL(A, [{ ops: B }], undefined, { envx: true }),
    buildSceneFragGL(SCENE, { envx: true }),
  ])
    for (const tok of ENVX_TOKENS_GL)
      assert.ok(src.includes(tok), `envx GLSL missing ${tok}`);
});

// ── Layout constants ─────────────────────────────────────────────────────────
test("globals layout: ENVX tail appends after the base struct", () => {
  assert.equal(GLOBALS_WORDS, 48); // the base struct + post-shader PG size — frozen
  assert.equal(ENVX_WORD, 48); // starsU=48, bandU=49, zenU=50
  assert.equal(GLOBALS_WORDS_ALLOC, 51); // buffer ceiling = base + 3 tail rows
});

// ── WGSL ⟷ GLSL star-math parity ────────────────────────────────────────────
// The two tiers must place every star identically. Pin the magic constants in
// BOTH bodies; a change to one emitter without the other fails here.
const STAR_MAGIC = [
  "127.1, 311.7, 74.7", // hash basis
  "43758.5453", // hash scale
  "* 0.5 - sOff", // jitter margin (0.25 cells clear of the disc support)
  "* kAng", // pixel-fixed disc sharpness…
  "/ (2.0 *", // …k = H²/(2.0·fov²), σ ≈ 1.0 px at any zoom
  "step(0.3, hashE(cell + 17.3))", // ~30% presence cull — breaks the carpet
  "0.05 + 0.75 * m2 * m2 * m2", // steep magnitude curve, peak 0.8 < bloom threshold
  "* 1.7, 150.0)", // second dust layer: 1.7× density, capped
  "* 0.55", // …at 55% brightness — the depth layer
  "0.0, 0.45", // band smoothstep width
  "0.25 + 1.75 * bandW", // band density lift
  "0.72, 0.78, 1.0", // band haze color
  "cell + 5.19", // temperature hash offset
];
test("star math is pinned identically in WGSL and GLSL", () => {
  const wgsl = buildWGSL({ envx: true });
  const glsl = buildFragGL(A, undefined, undefined, { envx: true });
  for (const magic of STAR_MAGIC) {
    assert.ok(wgsl.includes(magic), `WGSL missing star constant "${magic}"`);
    assert.ok(glsl.includes(magic), `GLSL missing star constant "${magic}"`);
  }
});

// ── Shared derivation ────────────────────────────────────────────────────────
test("deriveFrameParams: all-off look derives envx false with inert words", () => {
  const d = deriveFrameParams({ light: defaultColoring().light });
  assert.equal(d.envx, false);
  assert.equal(d.stars, 0);
  assert.equal(d.band, 0);
  assert.equal(d.zenithOn, 0);
});

test("deriveFrameParams: each macro alone flips envx on", () => {
  assert.equal(deriveFrameParams({ light: { stars: 0.4 } }).envx, true);
  assert.equal(deriveFrameParams({ light: { band: 0.4 } }).envx, true);
  assert.equal(deriveFrameParams({ light: { zenith: [0, 0, 0.1] } }).envx, true);
});

test("deriveFrameParams: density slider maps to the 14..80 cell range", () => {
  assert.equal(deriveFrameParams({ light: { starDensity: 0 } }).starDensity, 14);
  assert.equal(deriveFrameParams({ light: { starDensity: 1 } }).starDensity, 80);
  assert.equal(deriveFrameParams({ light: {} }).starDensity, 47); // default 0.5
});

// ── Sky-blend decoupling ─────────────────────────────────────────────────────
// Field report (2026-08-09): Sky 0.09 crushed the star field to 9% — stars
// must not be hostage to the blend. Each emitter adds the (1 − sky) complement
// of envColor's sky-scaled star term in its miss path.
test("stars decouple from the Sky blend in both emitters", () => {
  const wgsl = buildWGSL({ envx: true });
  assert.ok(
    wgsl.includes("starField(rd) * smoothstep(-0.35, 0.12, rd.z) * (1.0 - clamp(G.env.x, 0.0, 1.0))"),
    "WGSL missing the (1 − sky) star complement",
  );
  const glsl = buildFragGL(A, undefined, undefined, { envx: true });
  assert.ok(
    glsl.includes("starField(rd) * smoothstep(-0.35, 0.12, rd.z) * (1.0 - clamp(uSky, 0.0, 1.0))"),
    "GLSL missing the (1 − sky) star complement",
  );
});

test("deriveFrameParams: band normal is unit and tilt sweeps horizon→overhead", () => {
  const flat = deriveFrameParams({ light: { bandTilt: 0 } }).bandDir;
  const vert = deriveFrameParams({ light: { bandTilt: 1 } }).bandDir;
  assert.deepEqual(flat, [0, 0, 1]); // tilt 0: plane normal +Z (band on the horizon)
  assert.ok(Math.abs(vert[1] - 1) < 1e-12 && Math.abs(vert[2]) < 1e-12); // tilt 1: overhead arc
  for (const n of [flat, vert])
    assert.ok(Math.abs(Math.hypot(...n) - 1) < 1e-12, "band normal not unit");
});

// ── Sanitize (the hostile-wire fence) ────────────────────────────────────────
test("sanitize: ENVX scalars clamp to slider domains, zenith keeps 3 channels", () => {
  const out = sanitizeColoring({
    light: { stars: 99, starDensity: -5, band: 0.5, bandTilt: 2, zenith: [9, -1, 0.5] },
  });
  assert.equal(out.light.stars, 1);
  assert.equal(out.light.starDensity, 0);
  assert.equal(out.light.band, 0.5);
  assert.equal(out.light.bandTilt, 1);
  assert.deepEqual(out.light.zenith, [1, 0, 0.5]);
});

test("sanitize: absent ENVX fields stay absent (shape-preserving)", () => {
  const out = sanitizeColoring({ light: { ambient: 0.2 } });
  for (const k of ["stars", "starDensity", "starSeed", "band", "bandTilt", "zenith"])
    assert.ok(!(k in out.light), `sanitize invented light.${k}`);
});

// ── Standalone export ────────────────────────────────────────────────────────
const FORMULA = { ops: A };
test("standalone export: starred look bakes the ENVX consts", () => {
  const g = exportStandaloneGLSL(FORMULA, {
    light: { stars: 0.6, band: 0.3, zenith: [0.01, 0.02, 0.08] },
  });
  for (const tok of ["const float uStars", "const vec3 uZenC", "starField"])
    assert.ok(g.includes(tok), `starred export missing ${tok}`);
});

test("standalone export: unstarred look emits no ENVX token", () => {
  const g = exportStandaloneGLSL(FORMULA, { light: defaultColoring().light });
  for (const tok of ENVX_TOKENS_GL)
    assert.ok(!g.includes(tok), `unstarred export leaked ${tok}`);
});

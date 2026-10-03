// THIN FILM — angle-driven interference material (IDEAS 2026-08-21 wave):
// the soap-bubble / beetle-shell sheen. NOT the S6 "Iridescence" slider (an
// orbit-trap modulator on the Glow SIGNAL): this term is pure geometry —
// phase ∝ cos(view, normal) — so its spectral bands sweep as the camera
// orbits, and the two stack.
//
// The load-bearing invariant: the feature is CODEGEN-gated, not uniform-gated.
// A look with the Thin film slider at 0 must emit shader text with NOT ONE
// film token — that byte-identity is the perf doctrine's "prove it's free"
// standard (the #125 lesson: a never-executing uniform branch still cost
// Mandelbulb +31%). These tests fence the gate on every emitter (WGSL, GLSL
// flat/hybrid, the film-free scene builder, standalone bake), the term parity
// between the GPU tiers, the shared derivation, the prewarm predictor,
// sanitize, and the CPU tier's LDR mirror.
//
// Run: node --test core/thinfilm.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { buildWGSL } from "./shader.js";
import { buildFragGL, buildSceneFragGL } from "./shader_gl.js";
import { deriveFrameParams } from "./frameparams.js";
import { sanitizeColoring } from "./sanitize.js";
import { exportStandaloneGLSL } from "./exportStandalone.js";
import { defaultColoring } from "./coloring.js";
import { frameFeaturesFor } from "./capturesettle.js";
import { shadeGrid } from "./cpu.js";

const FILM_TOKENS_WGSL = ["filmCos", "filmPh", "filmCol", "filmW", "G.morphX.w"];
const FILM_TOKENS_GL = ["uFilm", "filmCos", "filmPh", "filmCol", "filmW"];
const A = [{ key: "boxFold", values: [1] }];
const B = [{ key: "scale", values: [2] }];
const SCENE = [{ shapeId: 1, objType: "sphere", combine: "union" }];

// ── The gate: off emits nothing, on emits everything ─────────────────────────
test("WGSL: default build carries no THIN FILM token (the free-when-off fence)", () => {
  const src = buildWGSL();
  for (const tok of FILM_TOKENS_WGSL)
    assert.ok(!src.includes(tok), `default WGSL leaked ${tok}`);
});

test("WGSL: neon/envx/sreflect builds without thinFilm carry no film token", () => {
  for (const opts of [
    { neon: true },
    { envx: true },
    { sreflect: true },
    { neon: true, envx: true, sreflect: true },
  ]) {
    const src = buildWGSL(opts);
    for (const tok of FILM_TOKENS_WGSL)
      assert.ok(
        !src.includes(tok),
        `WGSL (${JSON.stringify(opts)}) leaked ${tok}`,
      );
  }
});

test("WGSL: thinFilm build carries the full feature", () => {
  for (const opts of [
    { thinFilm: true },
    { thinFilm: true, ops: [1, 2] },
    { thinFilm: true, coloring: true },
    { thinFilm: true, sreflect: true },
    { thinFilm: true, envx: true },
    { thinFilm: true, capture: true },
    { thinFilm: true, neon: true },
  ]) {
    const src = buildWGSL(opts);
    for (const tok of FILM_TOKENS_WGSL)
      assert.ok(
        src.includes(tok),
        `thinFilm WGSL (${JSON.stringify(opts)}) missing ${tok}`,
      );
  }
});

test("GLSL: default builds carry no THIN FILM token (flat/hybrid/scene)", () => {
  for (const src of [
    buildFragGL([]),
    buildFragGL(A),
    buildFragGL(A, [{ ops: B }]),
    buildFragGL(A, undefined, undefined, { neon: true, envx: true }),
    buildSceneFragGL(SCENE),
  ])
    for (const tok of FILM_TOKENS_GL)
      assert.ok(!src.includes(tok), `default GLSL leaked ${tok}`);
});

test("GLSL: thinFilm builds carry the full feature (flat/hybrid)", () => {
  for (const src of [
    buildFragGL(A, undefined, undefined, { thinFilm: true }),
    buildFragGL(A, [{ ops: B }], undefined, { thinFilm: true }),
    buildFragGL(A, undefined, undefined, {
      thinFilm: true,
      neon: true,
      envx: true,
    }),
  ]) {
    for (const tok of FILM_TOKENS_GL)
      assert.ok(src.includes(tok), `thinFilm GLSL missing ${tok}`);
  }
});

// ── Term parity — the same math in both GPU tiers ────────────────────────────
test("interference term is pinned identically in WGSL and GLSL", () => {
  const wgsl = buildWGSL({ thinFilm: true, neon: true });
  const glsl = buildFragGL(A, undefined, undefined, {
    thinFilm: true,
    neon: true,
  });
  // Phase ∝ cosθ at the inverse-λ ratios; modulation via mix toward 2·filmCol.
  assert.ok(wgsl.includes("let filmPh = 12.0 * filmCos;"));
  assert.ok(glsl.includes("float filmPh = 12.0 * filmCos;"));
  assert.ok(
    wgsl.includes("0.5 + 0.5 * cos(vec3f(1.0, 1.2218, 1.4444) * filmPh)"),
  );
  assert.ok(
    glsl.includes("0.5 + 0.5 * cos(vec3(1.0, 1.2218, 1.4444) * filmPh)"),
  );
  assert.ok(
    wgsl.includes("G.morphX.w * (0.3 + 0.7 * pow(1.0 - filmCos, 2.0))"),
  );
  assert.ok(glsl.includes("uFilm * (0.3 + 0.7 * pow(1.0 - filmCos, 2.0))"));
  assert.ok(wgsl.includes("col = col * mix(vec3f(1.0), filmCol * 2.0, filmW);"));
  assert.ok(glsl.includes("col *= mix(vec3(1.0), filmCol * 2.0, filmW);"));
  // Splice order: film modulates REFLECTED light, so it must precede the NEON
  // emission add and the fog blocks in both tiers.
  assert.ok(
    wgsl.indexOf("filmCol * 2.0") < wgsl.indexOf("G.p3ctl.w * neonSig"),
    "WGSL film must precede the neon emission",
  );
  assert.ok(
    wgsl.indexOf("filmCol * 2.0") < wgsl.indexOf("// P3 fog:"),
    "WGSL film must precede the fog block",
  );
  assert.ok(
    glsl.indexOf("filmCol * 2.0") < glsl.indexOf("uNeon * nSig"),
    "GLSL film must precede the neon emission",
  );
  assert.ok(
    glsl.indexOf("filmCol * 2.0") < glsl.indexOf("P3 fog + sun in-scatter"),
    "GLSL film must precede the fog block",
  );
});

// ── Shared derivation ────────────────────────────────────────────────────────
test("deriveFrameParams: default look derives thinFilm off with inert words", () => {
  const d = deriveFrameParams({ light: defaultColoring().light });
  assert.equal(d.thinFilm, false);
  assert.equal(d.filmAmt, 0);
});

test("deriveFrameParams: the slider IS the word (no premap) + the latch", () => {
  const d = deriveFrameParams({ light: { thinFilm: 0.5 } });
  assert.equal(d.thinFilm, true);
  assert.equal(d.filmAmt, 0.5);
  assert.equal(deriveFrameParams({ light: { thinFilm: 1 } }).filmAmt, 1);
  assert.equal(deriveFrameParams({ light: { thinFilm: 2 } }).filmAmt, 1); // clamped
  assert.equal(deriveFrameParams({ light: { thinFilm: -1 } }).filmAmt, 0);
});

// ── Prewarm predictor (capturesettle) — mirrors the renderer latch ───────────
test("frameFeaturesFor: slider > 0 predicts thinFilm on flat, never on scenes", () => {
  const flat = { ops: [{ key: "boxFold", values: [1] }] };
  const coloringOn = { mode: 1, light: { thinFilm: 0.5 } };
  assert.equal(frameFeaturesFor(flat, coloringOn).thinFilm, true);
  assert.equal(
    frameFeaturesFor(flat, { mode: 1, light: {} }).thinFilm,
    false,
  );
  const scene = { objects: SCENE };
  assert.equal(frameFeaturesFor(scene, coloringOn).thinFilm, false);
});

// ── Sanitize (the hostile-wire fence) ────────────────────────────────────────
test("sanitize: thinFilm clamps to the slider domain, absent stays absent", () => {
  const out = sanitizeColoring({ light: { thinFilm: 99 } });
  assert.equal(out.light.thinFilm, 1);
  assert.equal(sanitizeColoring({ light: { thinFilm: -3 } }).light.thinFilm, 0);
  const abs = sanitizeColoring({ light: { ambient: 0.2 } });
  assert.ok(!("thinFilm" in abs.light), "sanitize invented light.thinFilm");
});

// ── Standalone export ────────────────────────────────────────────────────────
const FORMULA = { ops: A };
test("standalone export: a filmed look bakes the uFilm const + the term", () => {
  const g = exportStandaloneGLSL(FORMULA, { light: { thinFilm: 0.5 } });
  assert.ok(g.includes("const float uFilm = 0.5"), "missing baked uFilm");
  assert.ok(
    g.includes("mix(vec3(1.0), filmCol * 2.0, filmW)"),
    "missing the modulation",
  );
});

test("standalone export: a film-off look emits no THIN FILM token", () => {
  const g = exportStandaloneGLSL(FORMULA, { light: defaultColoring().light });
  for (const tok of FILM_TOKENS_GL)
    assert.ok(!g.includes(tok), `film-off export leaked ${tok}`);
});

// ── CPU/ASCII tier — the LDR mirror ──────────────────────────────────────────
const CPU_FORMULA = {
  name: "film-cpu",
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
const CPU_OPTS = {
  cols: 32,
  rows: 20,
  ss: 1,
  edges: false,
  structure: false,
  dither: false,
};
const gridRGB = (light) =>
  shadeGrid(CPU_FORMULA, {
    ...CPU_OPTS,
    coloring: { mode: 1, autoLevels: false, light: { ...light } },
  }).rgb;

test("cpu: thinFilm 0 (and the absent field) render byte-identically", () => {
  const base = gridRGB({});
  const zero = gridRGB({ thinFilm: 0 });
  assert.deepEqual(zero, base, "thinFilm: 0 must not change one CPU pixel");
});

test("cpu: thinFilm > 0 tints covered cells (per-channel, LDR-clamped)", () => {
  const base = gridRGB({});
  const filmed = gridRGB({ thinFilm: 0.9 });
  let changed = 0;
  let differential = 0; // cells where the channels moved by DIFFERENT amounts
  for (let i = 0; i < base.length; i++) {
    if (!base[i]) {
      assert.equal(filmed[i], base[i], "background cells must be untouched");
      continue;
    }
    const d = [0, 1, 2].map((c) => filmed[i][c] - base[i][c]);
    for (let c = 0; c < 3; c++) {
      assert.ok(filmed[i][c] >= 0 && filmed[i][c] <= 255, "LDR clamp violated");
    }
    if (d.some((x) => x !== 0)) changed++;
    if (d[0] !== d[1] || d[1] !== d[2]) differential++;
  }
  assert.ok(changed > 0, "thinFilm 0.9 changed nothing — the CPU mirror is dead");
  assert.ok(
    differential > 0,
    "film moved all channels equally — a gray wash, not interference",
  );
});

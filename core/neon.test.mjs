// NEON — emissive surface glow (IDEAS 2026-08-21 wave).
//
// The load-bearing invariant: the feature is CODEGEN-gated, not uniform-gated.
// A look with the Neon slider at 0 must emit shader text with NOT ONE neon
// token — that byte-identity is the perf doctrine's "prove it's free" standard
// (the #125 lesson: a never-executing uniform branch still cost Mandelbulb
// +31%). These tests fence the gate on every emitter (WGSL, GLSL flat/hybrid,
// the neon-free scene builder, standalone bake), the shared derivation, the
// bloom arming, sanitize, and the CPU tier's LDR approximation.
//
// Run: node --test core/neon.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { buildWGSL } from "./shader.js";
import { buildFragGL, buildSceneFragGL } from "./shader_gl.js";
import { deriveFrameParams } from "./frameparams.js";
import { sanitizeColoring } from "./sanitize.js";
import { exportStandaloneGLSL } from "./exportStandalone.js";
import { defaultColoring } from "./coloring.js";
import { shadeGrid } from "./cpu.js";

const NEON_TOKENS_WGSL = ["neonSig", "G.p3ctl.w * neonSig"];
const NEON_TOKENS_GL = ["uNeon", "nSig"];
const A = [{ key: "boxFold", values: [1] }];
const B = [{ key: "scale", values: [2] }];
const SCENE = [{ shapeId: 1, objType: "sphere", combine: "union" }];

// ── The gate: off emits nothing, on emits everything ─────────────────────────
test("WGSL: default build carries no NEON token (the free-when-off fence)", () => {
  const src = buildWGSL();
  for (const tok of NEON_TOKENS_WGSL)
    assert.ok(!src.includes(tok), `default WGSL leaked ${tok}`);
});

test("WGSL: neon build carries the full feature", () => {
  for (const opts of [
    { neon: true },
    { neon: true, ops: [1, 2] },
    { neon: true, coloring: true },
    { neon: true, sreflect: true },
    { neon: true, envx: true },
    { neon: true, capture: true },
  ]) {
    const src = buildWGSL(opts);
    for (const tok of NEON_TOKENS_WGSL)
      assert.ok(
        src.includes(tok),
        `neon WGSL (${JSON.stringify(opts)}) missing ${tok}`,
      );
    // The signal records: default arm + the Painter arm.
    assert.ok(src.includes("neonSig = m;"), "missing the default-arm record");
    assert.ok(src.includes("neonSig = 1.0;"), "missing the Painter record");
  }
});

test("GLSL: default builds carry no NEON token (flat/hybrid/scene)", () => {
  for (const src of [
    buildFragGL([]),
    buildFragGL(A),
    buildFragGL(A, [{ ops: B }]),
    buildSceneFragGL(SCENE),
  ])
    for (const tok of NEON_TOKENS_GL)
      assert.ok(!src.includes(tok), `default GLSL leaked ${tok}`);
});

test("GLSL: neon builds carry the full feature (flat/hybrid)", () => {
  for (const src of [
    buildFragGL(A, undefined, undefined, { neon: true }),
    buildFragGL(A, [{ ops: B }], undefined, { neon: true }),
    buildFragGL(A, undefined, undefined, { neon: true, envx: true }),
  ]) {
    for (const tok of NEON_TOKENS_GL)
      assert.ok(src.includes(tok), `neon GLSL missing ${tok}`);
    assert.ok(src.includes("uNeon * nSig * nSig"), "missing the emission add");
  }
});

// ── Emission parity — the same math in both GPU tiers ────────────────────────
test("emission term is pinned identically in WGSL and GLSL", () => {
  const wgsl = buildWGSL({ neon: true });
  const glsl = buildFragGL(A, undefined, undefined, { neon: true });
  // albedo · gain · sig², added after the object-intensity multiply.
  assert.ok(
    wgsl.includes("col = col + albedo * (G.p3ctl.w * neonSig * neonSig);"),
  );
  assert.ok(glsl.includes("col += albedo * (uNeon * nSig * nSig);"));
  // Both sit BEFORE the fog blocks (haze must attenuate the glow).
  assert.ok(
    wgsl.indexOf("G.p3ctl.w * neonSig") < wgsl.indexOf("// P3 fog:"),
    "WGSL emission must precede the fog block",
  );
  assert.ok(
    glsl.indexOf("uNeon * nSig") < glsl.indexOf("P3 fog + sun in-scatter"),
    "GLSL emission must precede the fog block",
  );
});

// ── Shared derivation ────────────────────────────────────────────────────────
test("deriveFrameParams: default look derives neon off with inert words", () => {
  const d = deriveFrameParams({ light: defaultColoring().light });
  assert.equal(d.neon, false);
  assert.equal(d.neonGain, 0);
  assert.equal(d.bloomOn, false); // and it does NOT arm the bloom
});

test("deriveFrameParams: the slider maps to the ×8 HDR gain + the latch", () => {
  const d = deriveFrameParams({ light: { neon: 0.5 } });
  assert.equal(d.neon, true);
  assert.equal(d.neonGain, 4);
  assert.equal(deriveFrameParams({ light: { neon: 1 } }).neonGain, 8);
  assert.equal(deriveFrameParams({ light: { neon: 2 } }).neonGain, 8); // clamped
});

test("deriveFrameParams: neon arms the bloom composite at Glow 0", () => {
  const d = deriveFrameParams({ light: { neon: 1, glow: 0 } });
  assert.equal(d.bloomOn, true);
  assert.ok(Math.abs(d.bloomStrength - 0.4) < 1e-12); // the 0.5 floor × 0.8
  // Glow keeps full authority above the floor.
  const g = deriveFrameParams({ light: { neon: 1, glow: 0.9 } });
  assert.ok(Math.abs(g.bloomStrength - 0.72) < 1e-12);
  assert.equal(d.bloomThreshold, 1.0); // the fixed pre-tonemap threshold
});

// ── Sanitize (the hostile-wire fence) ────────────────────────────────────────
test("sanitize: neon clamps to the slider domain, absent stays absent", () => {
  const out = sanitizeColoring({ light: { neon: 99 } });
  assert.equal(out.light.neon, 1);
  assert.equal(sanitizeColoring({ light: { neon: -3 } }).light.neon, 0);
  const abs = sanitizeColoring({ light: { ambient: 0.2 } });
  assert.ok(!("neon" in abs.light), "sanitize invented light.neon");
});

// ── Standalone export ────────────────────────────────────────────────────────
const FORMULA = { ops: A };
test("standalone export: a neon look bakes the uNeon const + the emission", () => {
  const g = exportStandaloneGLSL(FORMULA, { light: { neon: 0.5 } });
  assert.ok(g.includes("const float uNeon = 4"), "missing baked uNeon gain");
  assert.ok(g.includes("uNeon * nSig * nSig"), "missing the emission add");
});

test("standalone export: a neon-off look emits no NEON token", () => {
  const g = exportStandaloneGLSL(FORMULA, { light: defaultColoring().light });
  for (const tok of NEON_TOKENS_GL)
    assert.ok(!g.includes(tok), `neon-off export leaked ${tok}`);
});

// ── CPU/ASCII tier — the LDR approximation (documented divergence) ───────────
const CPU_FORMULA = {
  name: "neon-cpu",
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

test("cpu: neon 0 (and the absent field) render byte-identically", () => {
  const base = gridRGB({});
  const zero = gridRGB({ neon: 0 });
  assert.deepEqual(zero, base, "neon: 0 must not change one CPU pixel");
});

test("cpu: neon > 0 brightens covered cells, never darkens", () => {
  const base = gridRGB({});
  const lit = gridRGB({ neon: 0.8 });
  let brighter = 0;
  for (let i = 0; i < base.length; i++) {
    if (!base[i]) {
      assert.equal(lit[i], base[i], "background cells must be untouched");
      continue;
    }
    for (let c = 0; c < 3; c++) {
      assert.ok(lit[i][c] >= base[i][c], `cell ${i} channel ${c} darkened`);
      assert.ok(lit[i][c] <= 255, "LDR clamp violated");
    }
    if (lit[i].some((v, c) => v > base[i][c])) brighter++;
  }
  assert.ok(brighter > 0, "neon 0.8 changed nothing — the CPU add is dead");
});

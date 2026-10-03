// Zero-tooling guard for the SHARED operator math (cpuorbit.js `applyOp`, now
// the one source consumed by both the CPU/ASCII renderer (cpu.js re-imports
// it) and evaluate.js — see REFACTORING.md item 1, and #266 for the split).
// Named *.test.mjs so it stays out of the apps' served *.js surface.
//
// The point: evaluate.SUPPORTED is derived from the IR registry, on the promise
// that applyOp has a real transform for every operator. If an op were added to
// the registry but not to applyOp's switch, it would hit the (silent) default
// and measure() would grade formulas using it as if the op did nothing. These
// tests fail loudly in that case.
//
// Run: node --test core/opmath.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";
import { OPERATORS, byKey } from "./operators.js";
import { applyOp } from "./cpuorbit.js";

// Deterministic, dependency-free PRNG so trials are reproducible.
function mulberry32(seed) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const clone = (s) => ({ x: s.x, y: s.y, z: s.z, w: s.w });
const changed = (a, b) =>
  a.x !== b.x || a.y !== b.y || a.z !== b.z || a.w !== b.w;

test("applyOp implements every registry operator (no silent no-op)", () => {
  const rnd = mulberry32(0x9e3779b9);
  const rn = (lo, hi) => lo + (hi - lo) * rnd();
  for (const op of OPERATORS) {
    const n = op.params.length;
    let everChanged = false;
    // Many asymmetric (point, params) trials: a real op transforms space for
    // SOME of them; a missing op (default case) never changes the point.
    for (let trial = 0; trial < 200 && !everChanged; trial++) {
      const s = { x: rn(-2, 2), y: rn(-2, 2), z: rn(-2, 2), w: 1 };
      const before = clone(s);
      const v = Array.from({ length: Math.max(n, 3) }, () => rn(-2.5, 2.5));
      applyOp(op.key, v, s);
      if (
        Number.isFinite(s.x) &&
        Number.isFinite(s.y) &&
        Number.isFinite(s.z) &&
        Number.isFinite(s.w) &&
        changed(before, s)
      )
        everChanged = true;
    }
    assert.ok(
      everChanged,
      `applyOp("${op.key}") never transformed a point over 200 trials — ` +
        `op is missing from the switch (silent no-op) or is a pure identity.`,
    );
  }
});

test("applyOp is deterministic (same input → same output)", () => {
  for (const op of OPERATORS) {
    const v = op.params.map((_, i) => 0.37 * (i + 1) - 0.5);
    const s1 = { x: 0.6, y: -0.4, z: 0.9, w: 1 };
    const s2 = { x: 0.6, y: -0.4, z: 0.9, w: 1 };
    applyOp(op.key, v, s1);
    applyOp(op.key, v, s2);
    assert.deepEqual(s1, s2, `applyOp("${op.key}") is not deterministic`);
  }
});

test("applyOp keeps finite input finite for sane params", () => {
  const rnd = mulberry32(42);
  const rn = (lo, hi) => lo + (hi - lo) * rnd();
  for (const op of OPERATORS) {
    for (let trial = 0; trial < 50; trial++) {
      const s = { x: rn(-1.5, 1.5), y: rn(-1.5, 1.5), z: rn(-1.5, 1.5), w: 1 };
      // Sane params: positive scales/sizes, moderate angles — the ranges the
      // sliders actually expose. (mandelbulbPower etc. can legitimately blow up
      // far from origin; keep the point inside the unit-ish region above.)
      const v = op.params.map((p) => {
        const d = Number.isFinite(p.default) ? p.default : 1;
        return d + rn(-0.3, 0.3);
      });
      applyOp(op.key, v, s);
      assert.ok(
        Number.isFinite(s.x) && Number.isFinite(s.y) && Number.isFinite(s.z),
        `applyOp("${op.key}") produced non-finite output from finite input`,
      );
    }
  }
});

// ── kaleido Mirror param (pre-step pass unlock) ──────────────────────────────
// Mirror ON (default, and the value old 2-value op lists sanitize to) is the
// historic reflecting kaleidooscope; Mirror OFF is the pure rotational
// sector-snap the MB3D PolyFold/Koch pre-steps use: every point at angle
// θ0 + k·wedge maps back to θ0 exactly (radius kept, no reflection).
test("kaleido mirror OFF = rotational sector-snap (no reflection)", () => {
  const N = 5;
  const wedge = (2 * Math.PI) / N;
  const r = 1.3;
  const th0 = -0.3 * wedge; // negative side of the base wedge
  for (let k = -2; k <= 2; k++) {
    const th = th0 + k * wedge;
    const s = { x: r * Math.cos(th), y: r * Math.sin(th), z: 0.7, w: 1 };
    applyOp("kaleido", [N, 0, 0], s);
    const ang = Math.atan2(s.y, s.x);
    assert.ok(
      Math.abs(ang - th0) < 1e-6,
      `sector ${k}: got ${ang}, want ${th0}`,
    );
    assert.ok(Math.abs(Math.hypot(s.x, s.y) - r) < 1e-6, "radius changed");
    assert.equal(s.z, 0.7);
  }
});

// ── #426 w-bookkeeping (the analytic IFS DE = r/|w|) ─────────────────────────
// #426 hypothesized a WGSL icosaFold/kaleido `w` divergence made a deOption-2
// GPU capture march a uniformly-tiny DE. A numeric CPU↔WGSL chain compare
// (app/scripts/chaindiff-426.mjs) DISPROVED it: the tiers agree on w and DE to
// f32 precision. These pins lock in WHY — icosaFold/kaleido are isometries
// (|Jacobian| = 1 ⇒ w untouched) and scale is the only w-multiplier — so any
// future edit that lets one of these folds touch w (breaking r/|w|) fails loudly.
test("#426: icosaFold and kaleido are isometries — w is untouched (|Jacobian|=1)", () => {
  const rnd = mulberry32(0xc0ffee);
  const rn = (lo, hi) => lo + (hi - lo) * rnd();
  for (let trial = 0; trial < 200; trial++) {
    const w0 = rn(0.1, 50);
    for (const [key, v] of [
      ["icosaFold", []],
      ["kaleido", [5, 0, 1]],
      ["kaleido", [6, 30, 0]],
    ]) {
      const s = { x: rn(-2, 2), y: rn(-2, 2), z: rn(-2, 2), w: w0 };
      applyOp(key, v, s);
      assert.equal(
        s.w,
        w0,
        `applyOp("${key}") changed w ${w0}→${s.w} — a fold that touches w breaks the r/|w| DE`,
      );
    }
  }
});

test("#426: scale is the r/|w| derivative multiplier — w *= |k| (translate leaves w alone)", () => {
  const s = { x: 0.3, y: -0.7, z: 0.4, w: 1 };
  applyOp("scale", [1.16], s);
  assert.ok(Math.abs(s.w - 1.16) < 1e-12, `scale 1.16: w=${s.w}, want 1.16`);
  applyOp("scale", [-2], s);
  assert.ok(
    Math.abs(s.w - 1.16 * 2) < 1e-12,
    `scale -2: w=${s.w}, want |k| product`,
  );
  const before = s.w;
  applyOp("translate", [-1.17, -0.98, 0.05], s);
  assert.equal(s.w, before, "translate must not touch w (|Jacobian|=1)");
});

test("#426: the repro chain accumulates w = |scale|^iters (only scale grows w)", () => {
  // icosaFold + kaleido(5,0,1) + translate(-1.17,-0.98,0.05) + scale(1.16), ×24.
  const chain = [
    ["icosaFold", []],
    ["kaleido", [5, 0, 1]],
    ["translate", [-1.17, -0.98, 0.05]],
    ["scale", [1.16]],
  ];
  const s = { x: 0.2, y: 0.1, z: -0.3, w: 1 };
  for (let i = 0; i < 24; i++) for (const [k, v] of chain) applyOp(k, v, s);
  const expected = Math.pow(1.16, 24);
  assert.ok(
    Math.abs(s.w - expected) / expected < 1e-9,
    `w=${s.w}, want ${expected} — icosaFold/kaleido/translate must not perturb the derivative`,
  );
});

test("kaleido mirror ON and missing 3rd value both keep the legacy reflection", () => {
  const N = 5;
  const wedge = (2 * Math.PI) / N;
  const th0 = -0.3 * wedge;
  const mk = () => ({ x: Math.cos(th0), y: Math.sin(th0), z: 0, w: 1 });
  const on = mk();
  const legacy = mk();
  const off = mk();
  applyOp("kaleido", [N, 0, 1], on);
  applyOp("kaleido", [N, 0], legacy); // pre-mirror-param op list
  applyOp("kaleido", [N, 0, 0], off);
  assert.deepEqual(on, legacy, "2-value list must behave as mirror ON");
  // Reflection flips the negative-side angle to +0.3·wedge...
  assert.ok(Math.abs(Math.atan2(on.y, on.x) - 0.3 * wedge) < 1e-6);
  // ...which mirror OFF must NOT do.
  assert.ok(Math.abs(Math.atan2(off.y, off.x) - th0) < 1e-6);
});

// ── #553: kaleido Mirror was dead on every GLSL tier ─────────────────────────
// operators.js:424 used to compare the GLSL *variable-name string* v[2]
// ("uP[2]" on WebGL2/standalone-export, "p2" on the MB3D desktop export)
// against 0.5 in JAVASCRIPT at emit time — a NaN comparison, always false, so
// the abs(ang) reflection branch never reached any GLSL tier regardless of
// the authored Mirror value. The CPU-only tests above (`applyOp`) cannot see
// this class of bug at all — it lives entirely in the `glsl` string template,
// which they never exercise. These pin the fix at the emission level.
test("#553: kaleido Mirror's v[2] reference actually reaches the emitted GLSL", () => {
  const kaleido = byKey("kaleido");
  // Every real consumer feeds v[2] as a variable-name-like string — "uP[2]"
  // on WebGL2 and the standalone/Shadertoy/Compushady export (both build on
  // shader_gl.js's iterBodyGL), "p2" on the MB3D desktop export
  // (shader.js's getGenericParam path) — never a number. Pre-fix, NEITHER
  // form ever appeared anywhere in the output: the JS ternary silently
  // swallowed whichever string came in, so Mirror was unreachable in GLSL
  // on all three tiers regardless of its authored value.
  for (const ref of ["uP[2]", "p2"]) {
    const body = kaleido.glsl(["uP[0]", "radians(uP[1])", ref]);
    assert.ok(
      body.includes(ref),
      `kaleido.glsl(): the Mirror reference "${ref}" never reached the output`,
    );
  }
});

test("#553: kaleido Mirror ON vs OFF emit DIFFERENT GLSL (a real runtime branch, not a dead one)", () => {
  const kaleido = byKey("kaleido");
  // v[2] as an emission-time constant is as legitimate an input to the
  // template as a uP[]/p-slot runtime reference (both are just GLSL
  // float-valued expressions) — literal 1.0/0.0 makes "ON" vs "OFF" a
  // concrete, legible byte-diff. (The previous test already pins that a REAL
  // uP[]/p-slot reference also participates, which is what production feeds.)
  const on = kaleido.glsl(["6.0", "radians(30.0)", "1.0"]);
  const off = kaleido.glsl(["6.0", "radians(30.0)", "0.0"]);
  assert.notEqual(
    on,
    off,
    "Mirror ON and OFF must not emit byte-identical GLSL",
  );
  assert.match(on, /if \(1\.0 > 0\.5\) \{ ang = abs\(ang\); \}/);
  assert.match(off, /if \(0\.0 > 0\.5\) \{ ang = abs\(ang\); \}/);
});

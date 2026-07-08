// Zero-tooling guard for the SHARED operator math (cpu.js `applyOp`, now the one
// source consumed by both the CPU/ASCII renderer and evaluate.js — see
// REFACTORING.md item 1). Named *.test.mjs so sync_web_core.sh skips it.
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
import { OPERATORS } from "./operators.js";
import { applyOp } from "./cpu.js";

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

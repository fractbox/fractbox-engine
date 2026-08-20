// Deep zoom Phase 4 (DEEP_ZOOM_DF64.md, plan PR-3) — the CPU f64 oracle.
//
// cpu.js marches in plain JS numbers (f64 end-to-end, no fround) — it is
// wall-free to ~×10¹⁴ and therefore the CI-able reference for "structure
// still exists at ×10⁹". This test finds a surface point on the shipped
// TOURBILLON preset by marching its own DE, then samples a nanometer-scale
// grid around it and pins NON-DEGENERACY: the field varies at 1e-9 scale
// (f32 would flatline — its quantum at these coordinates is ~1e-7). The
// GPU-df64-vs-CPU crop compare at matched cameras is the manual half
// (plan PR-4 ladder); this pin guards the oracle itself.
//
// Run: node --test core/deepzoom.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";
import { makeDE } from "./cpu.js";
import { TOURBILLON } from "./oplist.js";

test("CPU DE resolves structure at ×10⁹ (the df64 correctness oracle)", () => {
  const de = makeDE(TOURBILLON, TOURBILLON.iters);
  // Walk near the surface (d ~ 1e-4 suffices — see below): march the
  // engine's own surface probe direction, then damped descent. The pin does
  // NOT need surface contact; it needs a point where the field has O(1)
  // gradient, because the claim under test is that inputs 1e-9 apart —
  // INDISTINGUISHABLE in f32 (quantum ~6e-8 at |p|≈O(1)) — produce
  // distinct outputs in the f64 tier.
  let p = [3, 1.1, 0.7];
  let d = de(...p);
  assert.ok(Number.isFinite(d) && d > 0, "start point must be outside");
  for (let i = 0; i < 600 && d > 1e-4; i++) {
    const n = Math.hypot(...p) || 1;
    p = [p[0] - (p[0] / n) * d * 0.7, p[1] - (p[1] / n) * d * 0.7, p[2] - (p[2] / n) * d * 0.7];
    d = de(...p);
    assert.ok(Number.isFinite(d), `DE went non-finite at step ${i}`);
  }
  assert.ok(d <= 1e-3, `must get near the surface (got ${d})`);

  // nanometer grid: 8×8 samples spaced 1e-9 in the surface's tangent-ish
  // plane. At |p| ≈ O(1) the f32 quantum is ~6e-8 — a 1e-9 grid is BELOW
  // the f32 wall, so any variation seen here is f64 doing real work.
  const h = 1e-9;
  const vals = [];
  for (let i = 0; i < 8; i++)
    for (let j = 0; j < 8; j++)
      vals.push(de(p[0] + (i - 3.5) * h, p[1] + (j - 3.5) * h, p[2]));
  assert.ok(
    vals.every(Number.isFinite),
    "all sub-wall samples finite",
  );
  const lo = Math.min(...vals);
  const hi = Math.max(...vals);
  // non-degenerate: the field VARIES across the sub-f32 grid…
  assert.ok(hi - lo > 1e-12, `field must vary at 1e-9 scale (spread ${hi - lo})`);
  // …and every sample is still near-surface (the grid didn't jump rooms)
  assert.ok(hi < 1e-2, `samples must stay in the near field (max ${hi})`);
  // distinct values: quantized-flat output (the f32 signature) would repeat
  const distinct = new Set(vals.map((v) => v.toPrecision(12))).size;
  assert.ok(distinct > 16, `expected many distinct values, got ${distinct}`);
});

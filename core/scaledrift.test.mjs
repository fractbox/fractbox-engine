// Exponent guard for the scaleDrift op (SCALE_VARY.md §8) — the load-bearing
// test the degeneracy anchor (scaledrift-anchor in scripts/oracle/manifest.json)
// CANNOT provide: at ScaleVary=0 every exponent gives m=1, so only a NON-zero-v
// case pinned against the hand-computed sequence proves the exponent is i+1
// (MB3D updates Scale BEFORE its first use — §2 indexing), not i.
//
// Named *.test.mjs so sync_web_core.sh skips it. Run: node --test core/scaledrift.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";
import { applyOp } from "./cpu.js";

// m recovered from an x=1 unit point: scaleDrift multiplies pos by m.
function driftM(scale, vary, i) {
  const s = { x: 1, y: 0, z: 0, w: 1 };
  if (i !== undefined) s.i = i;
  applyOp("scaleDrift", [scale, vary], s);
  return s;
}

test("scaleDrift ramps as S_{i+1} = 1+(S0-1)*(1+v)^(i+1) — the i+1 exponent", () => {
  // Scale=2, ScaleVary=0.05 → S1,S2,S3 of SCALE_VARY.md §2.
  const expect = { 0: 2.05, 1: 2.1025, 2: 2.157625 };
  for (const i of [0, 1, 2]) {
    const s = driftM(2.0, 0.05, i);
    assert.ok(
      Math.abs(s.x - expect[i]) < 1e-6,
      `scaleDrift m at i=${i} = ${s.x}, want ${expect[i]} (exponent must be i+1, not i)`,
    );
    // Conformal: w tracks |m| exactly like the scale op.
    assert.ok(Math.abs(s.w - Math.abs(expect[i])) < 1e-6, `w must track |m| at i=${i}`);
  }
  // i=0 giving 2.05 (not 2.0) is the whole point: exponent i would give (1.05)^0=1 → m=2.0.
  assert.ok(Math.abs(driftM(2.0, 0.05, 0).x - 2.0) > 0.04, "i=0 must NOT be the i-exponent value 2.0");
});

test("scaleDrift ScaleVary=0 degenerates to a constant scale (the oracle anchor)", () => {
  for (const i of [0, 1, 5, 11]) {
    const s = driftM(2.0, 0.0, i);
    assert.ok(Math.abs(s.x - 2.0) < 1e-12, `ScaleVary=0 must be constant scale(2.0) at i=${i}, got ${s.x}`);
  }
});

test("scaleDrift missing s.i defaults to 0 (belt-and-suspenders floor)", () => {
  const s = driftM(2.0, 0.05, undefined); // no s.i set
  assert.ok(Math.abs(s.x - 2.05) < 1e-6, `undefined s.i must floor to i=0 (m=2.05), got ${s.x}`);
});

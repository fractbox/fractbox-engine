// Guards for magnetXYZ / magnetXYZAbs (MagVsXYZ family) — pins the per-axis
// closed form against an independent derivation, the two deliberate variant
// deltas (abs fold + A1-driven angle), and the W_BULB_NUMERIC contract.
//
// Named *.test.mjs so sync_web_core.sh skips it. Run: node --test core/magnet.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";
import { applyOp } from "./cpuorbit.js";

const P0 = { x: 0.5, y: 0.4, z: 0.7, w: 1 };
const run = (key, values, pt = P0) => {
  const s = { ...pt };
  applyOp(key, values, s);
  return s;
};

// Independent re-derivation of one lane:
//   base: a' =  cos(P ·atan2(r, |a|·P·A1)) · (r² + A2·a²)^(P/2)
//   abs:  a' = |cos(A1·atan2(r, |a|·P·A1)) · (r² + A2·a²)^(P/2)|
function lane(pt, a, P, A1, A2, isAbs) {
  const r2 = pt.x * pt.x + pt.y * pt.y + pt.z * pt.z;
  const r = Math.sqrt(r2);
  const ang = (isAbs ? A1 : P) * Math.atan2(r, Math.abs(a) * P * A1);
  const out = Math.cos(ang) * Math.pow(r2 + A2 * a * a, P / 2);
  return isAbs ? Math.abs(out) : out;
}

test("magnetXYZ matches the closed form on every lane", () => {
  for (const [P, A1, A2] of [
    [2, Math.PI / 2, Math.PI / 4],
    [3.5, 0.8, 1.2],
  ]) {
    const got = run("magnetXYZ", [P, A1, A2]);
    for (const [k, a] of [
      ["x", P0.x],
      ["y", P0.y],
      ["z", P0.z],
    ]) {
      const want = lane(P0, a, P, A1, A2, false);
      assert.ok(
        Math.abs(got[k] - want) < 1e-12,
        `${k}: got ${got[k]}, want ${want}`,
      );
    }
  }
});

test("magnetXYZAbs matches its closed form (abs fold + A1-driven angle)", () => {
  const [P, A1, A2] = [2, Math.PI / 2, Math.PI / 2];
  const got = run("magnetXYZAbs", [P, A1, A2]);
  for (const [k, a] of [
    ["x", P0.x],
    ["y", P0.y],
    ["z", P0.z],
  ]) {
    const want = lane(P0, a, P, A1, A2, true);
    assert.ok(
      Math.abs(got[k] - want) < 1e-12,
      `${k}: got ${got[k]}, want ${want}`,
    );
    assert.ok(got[k] >= 0, `${k} must be abs-folded (non-negative)`);
  }
});

test("the variants are genuinely different maps (not just an abs of each other)", () => {
  // Same params for both: at P ≠ A1 the angle multipliers differ, so the abs
  // variant must NOT equal |base|.
  const vals = [3, 0.9, 1.1];
  const base = run("magnetXYZ", vals);
  const folded = run("magnetXYZAbs", vals);
  const absOfBase = {
    x: Math.abs(base.x),
    y: Math.abs(base.y),
    z: Math.abs(base.z),
  };
  const differs = ["x", "y", "z"].some(
    (k) => Math.abs(folded[k] - absOfBase[k]) > 1e-6,
  );
  assert.ok(differs, "magnetXYZAbs must differ from |magnetXYZ| when P ≠ A1");
});

test("origin guard + W_BULB_NUMERIC contract (w untouched)", () => {
  const zero = run("magnetXYZ", [2, 1.5, 0.7], { x: 0, y: 0, z: 0, w: 1.3 });
  assert.deepEqual([zero.x, zero.y, zero.z, zero.w], [0, 0, 0, 1.3]);
  const s = run("magnetXYZAbs", [2, 1.5, 0.7], { ...P0, w: 1.7 });
  assert.equal(s.w, 1.7);
});

// Guard for the riemannBulb op — pins the hardened (m-rescaled) stereographic
// combine against the plain closed form, checks the pole guards keep float
// math finite, and asserts the W_BULB_NUMERIC contract (w untouched).
//
// Named *.test.mjs so sync_web_core.sh skips it. Run: node --test core/riemann.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";
import { applyOp } from "./cpuorbit.js";

const run = (values, pt) => {
  const s = { ...pt };
  applyOp("riemannBulb", values, s);
  return s;
};

// Independent plain-form re-derivation (no m-rescale hardening): project the
// unit direction stereographically from the +Y pole, tan-half-angle power the
// two coordinates, invert the projection, scale by r^P.
function plainRiemann(pt, P) {
  const r = Math.hypot(pt.x, pt.y, pt.z);
  const d = Math.min(pt.y / r - 1, -1e-7);
  const a = pt.x / r / d;
  const b = pt.z / r / d;
  const tp = (t) => {
    const ang = Math.atan2(2 * t, t * t - 1) * P;
    return Math.sin(ang) / Math.max(Math.cos(ang) + 1, 1e-30);
  };
  const ta = tp(a),
    tb = tp(b);
  const S = ta * ta + tb * tb;
  const rp = Math.pow(r, P);
  return {
    x: ((2 * ta) / (S + 1)) * rp,
    y: ((S - 1) / (S + 1)) * rp,
    z: ((2 * tb) / (S + 1)) * rp,
  };
}

test("hardened combine equals the plain closed form (generic points)", () => {
  const pts = [
    { x: 0.5, y: 0.4, z: 0.7, w: 1 },
    { x: -0.31, y: 0.62, z: -0.44, w: 1 },
    { x: 1.1, y: -0.2, z: 0.05, w: 1 },
  ];
  for (const P of [2, 8, 11.5]) {
    for (const pt of pts) {
      const got = run([P], pt);
      const want = plainRiemann(pt, P);
      for (const k of ["x", "y", "z"]) {
        const scale = Math.max(1, Math.abs(want[k]));
        assert.ok(
          Math.abs(got[k] - want[k]) / scale < 1e-9,
          `P=${P} ${k}: got ${got[k]}, want ${want[k]}`,
        );
      }
    }
  }
});

test("pole guards: +Y axis, origin-adjacent, and near-pole points stay finite", () => {
  const hard = [
    { x: 0, y: 1, z: 0, w: 1 }, // exactly the projection pole
    { x: 1e-12, y: 0.9999999, z: -1e-12, w: 1 }, // hugging the pole
    { x: 1e-10, y: 1e-10, z: 1e-10, w: 1 }, // near the origin (R > guard)
    { x: 0, y: 0, z: 0, w: 1 }, // origin: guarded no-op
  ];
  for (const pt of hard) {
    const s = run([8], pt);
    for (const k of ["x", "y", "z", "w"])
      assert.ok(
        Number.isFinite(s[k]),
        `${k} must stay finite for ${JSON.stringify(pt)}, got ${s[k]}`,
      );
  }
});

test("W_BULB_NUMERIC contract: w is untouched", () => {
  const s = run([8], { x: 0.5, y: 0.4, z: 0.7, w: 1.7 });
  assert.equal(s.w, 1.7, "riemannBulb must not modify w (numeric DE owns it)");
});

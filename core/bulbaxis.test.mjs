// Convention guard for the bulbAxis op (TRIGBULB_SPIKE.md) — pins the three
// trig conventions against independently hand-derived closed forms, the
// back-compat degeneracy (Convention absent/0 ≡ the pre-Convention op ≡
// mandelbulbPower at axis 0), and the w-update invariance across conventions.
//
// Named *.test.mjs so sync_web_core.sh skips it. Run: node --test core/bulbaxis.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";
import { applyOp } from "./cpuorbit.js";

const P = { x: 0.5, y: 0.4, z: 0.7, w: 1.0 };
const run = (key, values, pt = P) => {
  const s = { ...pt };
  applyOp(key, values, s);
  return s;
};
const close = (a, b, eps = 1e-12) => Math.abs(a - b) < eps;

// Independent re-derivation (z-polar layout): r→rⁿ, φ = atan2(y,x)·n, with the
// polar angle θ from acos or asin of z/r and the convention choosing which of
// sin/cos lands on the pole. Written from the published bulb construction, not
// from the op body.
function bulbZ(pt, n, conv) {
  const r = Math.hypot(pt.x, pt.y, pt.z);
  const u = Math.max(-1, Math.min(1, pt.z / r));
  const th = (conv === 2 ? Math.asin(u) : Math.acos(u)) * n;
  const ph = Math.atan2(pt.y, pt.x) * n;
  const rn = Math.pow(r, n);
  const eq = conv === 0 ? Math.sin(th) : Math.cos(th);
  const pole = conv === 0 ? Math.cos(th) : Math.sin(th);
  return {
    x: rn * eq * Math.cos(ph),
    y: rn * eq * Math.sin(ph),
    z: rn * pole,
    w: (n * rn) / r + 1, // starting w = 1
  };
}

test("Convention 0/1/2 match the hand-derived closed forms (z-polar)", () => {
  for (const conv of [0, 1, 2]) {
    const got = run("bulbAxis", [8.0, 0.0, conv]);
    const want = bulbZ(P, 8.0, conv);
    for (const k of ["x", "y", "z", "w"])
      assert.ok(
        close(got[k], want[k]),
        `conv=${conv} ${k}: got ${got[k]}, want ${want[k]}`,
      );
  }
});

test("conventions are genuinely distinct maps (1 and 2 differ from 0 and from each other)", () => {
  const [c0, c1, c2] = [0, 1, 2].map((c) => run("bulbAxis", [8.0, 0.0, c]));
  assert.ok(!close(c0.z, c1.z, 1e-6), "conv 1 must differ from conv 0");
  assert.ok(!close(c0.z, c2.z, 1e-6), "conv 2 must differ from conv 0");
  assert.ok(!close(c1.z, c2.z, 1e-6), "conv 2 must differ from conv 1");
});

test("back-compat: 2-value payload ≡ Convention 0 (the degeneracy anchor)", () => {
  const legacy = run("bulbAxis", [8.0, 1.0]); // pre-Convention arity
  const conv0 = run("bulbAxis", [8.0, 1.0, 0.0]);
  for (const k of ["x", "y", "z", "w"])
    assert.ok(close(legacy[k], conv0[k]), `legacy ${k} must equal conv 0`);
});

test("axis 0 + Convention 0 reproduces mandelbulbPower exactly", () => {
  const viaAxis = run("bulbAxis", [8.0, 0.0, 0.0]);
  const viaBulb = run("mandelbulbPower", [8.0]);
  for (const k of ["x", "y", "z", "w"])
    assert.ok(close(viaAxis[k], viaBulb[k]), `${k} must match mandelbulbPower`);
});

test("w-update is convention-invariant (same analytic dr, TRIGBULB_SPIKE §w-rule)", () => {
  const start = { x: 0.31, y: -0.62, z: 0.44, w: 1.7 };
  const ws = [0, 1, 2].map((c) => run("bulbAxis", [7.0, 1.0, c], start).w);
  assert.ok(
    close(ws[0], ws[1]) && close(ws[0], ws[2]),
    `w must not depend on Convention, got ${ws}`,
  );
});

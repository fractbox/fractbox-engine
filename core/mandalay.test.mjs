// Guards for mandalayFold (parity wave 1).
//
// The op is built clean-room from the authors' prose + pseudocode in the 2015
// Fractal Forums Mandalay thread (DarkBeam, with knighty) — see the derivation
// note in operators.js. What these tests defend is the property that made
// knighty's formulation the one worth shipping: it is a GLOBALLY CONTINUOUS
// piecewise isometry, so W_UNCHANGED is exact and it needs no deApprox flag.
// DarkBeam's branched KIFS variant reaches the same tower family but tears
// space at its branch boundaries; a tear is an infinite Lipschitz constant and
// the marcher steps through it.
//
// Named *.test.mjs so sync_web_core.sh skips it.
// Run: node --test core/mandalay.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";
import { applyOp } from "./cpuorbit.js";
import { isApproxDE, isDeSound, byKey } from "./operators.js";

const run = (values, pt) => {
  const s = { x: pt.x, y: pt.y, z: pt.z, w: pt.w ?? 1 };
  applyOp("mandalayFold", values, s);
  return s;
};
const V = [0.5, 0.1, 0.0]; // sourced defaults (fo 0.5, g 0.1)
// Deterministic PRNG so a failure is reproducible.
function mulberry32(seed) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

test("|Jacobian| = 1 — the fold is a local isometry", () => {
  // THE load-bearing claim behind wRule W_UNCHANGED. Central differences over
  // random points; branch boundaries are measure-zero so a stray sample there
  // would show up as a large outlier, not a small one.
  const rnd = mulberry32(0xa17d);
  const h = 1e-6;
  let worst = 0;
  for (let i = 0; i < 4000; i++) {
    const p = {
      x: (rnd() - 0.5) * 4,
      y: (rnd() - 0.5) * 4,
      z: (rnd() - 0.5) * 4,
    };
    const J = ["x", "y", "z"].map((ax) => {
      const a = { ...p },
        b = { ...p };
      a[ax] -= h;
      b[ax] += h;
      const ra = run(V, a),
        rb = run(V, b);
      return [
        (rb.x - ra.x) / (2 * h),
        (rb.y - ra.y) / (2 * h),
        (rb.z - ra.z) / (2 * h),
      ];
    });
    const det =
      J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1]) -
      J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0]) +
      J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);
    worst = Math.max(worst, Math.abs(Math.abs(det) - 1));
  }
  assert.ok(
    worst < 1e-5,
    `max ||J|−1| = ${worst} — the fold is not an isometry`,
  );
});

test("the fold never EXPANDS distance (no tears — this is why it ships unflagged)", () => {
  // A discontinuity would show up here as a ratio far above 1: two points a
  // hair apart landing far apart. Non-expansion is what makes an unchanged w a
  // sound DE bound.
  const rnd = mulberry32(0x5eed);
  let maxRatio = 0;
  for (const params of [V, [0.9, 0.5, 0.0], [0.3, 1.2, 0.4]]) {
    for (let i = 0; i < 40000; i++) {
      const p = {
        x: (rnd() - 0.5) * 4,
        y: (rnd() - 0.5) * 4,
        z: (rnd() - 0.5) * 4,
      };
      const q = {
        x: p.x + (rnd() - 0.5) * 1e-4,
        y: p.y + (rnd() - 0.5) * 1e-4,
        z: p.z + (rnd() - 0.5) * 1e-4,
      };
      const a = run(params, p),
        b = run(params, q);
      const dIn = Math.hypot(p.x - q.x, p.y - q.y, p.z - q.z);
      if (dIn === 0) continue;
      maxRatio = Math.max(
        maxRatio,
        Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z) / dIn,
      );
    }
  }
  assert.ok(
    maxRatio < 1 + 1e-9,
    `expansion ratio ${maxRatio} > 1 — the fold tears space`,
  );
});

test("abs + descending sort: the octahedral wedge (Fold = Gap = 0 exposes it)", () => {
  // With both offsets at 0 the offset stage degenerates to a negation of the
  // middle lane, leaving the raw "Kifs Octahedral fold" visible: the output
  // magnitudes are the input magnitudes, sorted descending.
  for (const p of [
    { x: 0.3, y: -1.2, z: 0.8 },
    { x: -2.0, y: 0.1, z: -0.05 },
    { x: 1, y: 1, z: 1 },
  ]) {
    const o = run([0, 0, 0], p);
    const got = [Math.abs(o.x), Math.abs(o.y), Math.abs(o.z)];
    const want = [Math.abs(p.x), Math.abs(p.y), Math.abs(p.z)].sort(
      (a, b) => b - a,
    );
    for (let i = 0; i < 3; i++)
      assert.ok(Math.abs(got[i] - want[i]) < 1e-12, `${got} vs ${want}`);
  }
});

test("the offset pair matches knighty's DBKNFold closed form", () => {
  // x ← |x−fo| − fo ; clamped odd fold t = min(g, max(0, x−y)) ; y ← fo − |y−fo|
  const fo = 0.5,
    g = 0.1;
  const p = { x: 1.7, y: -0.9, z: 0.2 };
  let [X, Y, Z] = [Math.abs(p.x), Math.abs(p.y), Math.abs(p.z)].sort(
    (a, b) => b - a,
  );
  X = Math.abs(X - fo) - fo;
  const t = Math.min(g, Math.max(0, X - Y));
  X -= t;
  Y += t;
  Y = fo - Math.abs(Y - fo);
  const o = run([fo, g, 0], p);
  assert.ok(Math.abs(o.x - X) < 1e-12, `x ${o.x} vs ${X}`);
  assert.ok(Math.abs(o.y - Y) < 1e-12, `y ${o.y} vs ${Y}`);
  assert.ok(Math.abs(o.z - Z) < 1e-12, `z ${o.z} vs ${Z}`);
});

test("the clamped diagonal fold is continuous through BOTH of its regimes", () => {
  // t = min(g, max(0, x−y)) has three regimes (identity / transposition /
  // translation). The junctions are where a naive `if` would tear.
  const fo = 0.5,
    g = 0.25,
    eps = 1e-9;
  // Sweep a point across the whole fold structure and demand no jump.
  let prev = null;
  for (let u = -3; u <= 3; u += 0.0005) {
    const o = run([fo, g, 0], { x: u, y: 0.37, z: 0.11 });
    if (prev) {
      const jump = Math.hypot(o.x - prev.x, o.y - prev.y, o.z - prev.z);
      assert.ok(jump < 0.002 + eps, `jump ${jump} at u = ${u.toFixed(4)}`);
    }
    prev = o;
  }
});

test("ZFold folds the short axis only when it bites, and 0 leaves it alone", () => {
  const p = { x: 2.0, y: 1.4, z: 0.9 };
  assert.equal(run([0.5, 0.1, 0], p).z, 0.9, "ZFold 0 = untouched");
  // z (0.9) above the plane 0.4 → reflected to 2·0.4 − 0.9 = −0.1
  assert.ok(Math.abs(run([0.5, 0.1, 0.4], p).z - -0.1) < 1e-12, "reflected");
  // z below the plane → untouched
  assert.equal(run([0.5, 0.1, 1.5], p).z, 0.9, "plane above z = untouched");
});

test("registry contract: an exact isometry — w untouched, DE stays sound", () => {
  const def = byKey("mandalayFold");
  assert.equal(def.wRule, "unchanged");
  assert.ok(
    !def.deApprox,
    "a continuous isometry must NOT be deApprox-flagged",
  );
  assert.equal(def.params.length, 3);
  const f = { ops: [{ key: "mandalayFold", values: V }] };
  assert.equal(isApproxDE(f), false);
  assert.equal(isDeSound(f), true, "must not taint DE-soundness");
  // #426 contract: an isometry leaves w exactly alone.
  for (const p of [
    { x: 0.3, y: -1.2, z: 0.8, w: 1.7 },
    { x: -2, y: 0.4, z: 1.1, w: 0.25 },
  ])
    assert.equal(run(V, p).w, p.w, "isometry must not touch w");
});

test("finite in ⇒ finite out, including the degenerate corners", () => {
  for (const p of [
    { x: 0, y: 0, z: 0 },
    { x: 1, y: 1, z: 1 },
    { x: -1, y: 1, z: -1 },
  ])
    for (const params of [V, [0, 0, 0], [2, 2, 2]]) {
      const o = run(params, p);
      for (const k of ["x", "y", "z", "w"])
        assert.ok(
          Number.isFinite(o[k]),
          `${k} = ${o[k]} at ${JSON.stringify(p)}`,
        );
    }
});

// Guards for complexMap (parity wave 2) — the murl / cayley family.
//
// The corpus gap this op exists to close: corpus_coverage_2026-07-13.json
// classes `murl`, `murl2_fast` and `cayley2IFS` as needs_op —
//   murl        "complex-power Moebius composite (z^N, Cayley-style rational,
//                N-th root, z/Order)"
//   murl2_fast  "fast Order=2 specialization of murl; same gap"
//   cayley2IFS  "degree-4 complex rational map (Cayley transform of z^2) in
//                the rotated xy-plane; z kept, w *= |P7|"
// The op is a Möbius map conjugated by ζ↦ζ^N on the XY plane. Because the map
// was DERIVED from those structural notes rather than transcribed from any
// source (see the clean-room block in operators.js id 61), the pins below are
// against independently computable closed forms — textbook Cayley values and
// the published `curl` variation — not against a reference implementation.
//
// Named *.test.mjs so it stays out of the apps' served *.js surface.
// Run: node --test core/complexmap.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";
import { applyOp } from "./cpuorbit.js";
import { isApproxDE, isDeSound, byKey } from "./operators.js";

const CAYLEY = 0;
const MURL = 1;
const MURL2 = 2;

const run = (values, pt) => {
  const s = { x: 0, y: 0, z: 0, w: 1, ...pt };
  applyOp("complexMap", values, s);
  return s;
};
const close = (a, b, eps = 1e-9) => Math.abs(a - b) < eps;
const at = (values, x, y) => {
  const q = run(values, { x, y });
  return [q.x, q.y];
};

test("Cayley: the textbook transform (z−i)/(z+i) at Order 1, C 1", () => {
  // Ahlfors/Rudin's Cayley transform, the conformal map carrying the upper
  // half-plane onto the unit disc. Three values anyone can verify by hand:
  //   0 ↦ (−i)/(i) = −1,   i ↦ 0,   1 ↦ (1−i)/(1+i) = −i
  const V = [1, 1, CAYLEY];
  const [ax, ay] = at(V, 0, 0);
  assert.ok(close(ax, -1) && close(ay, 0), `0 ↦ (${ax}, ${ay}), want (−1, 0)`);
  const [bx, by] = at(V, 0, 1);
  assert.ok(close(bx, 0) && close(by, 0), `i ↦ (${bx}, ${by}), want (0, 0)`);
  const [cx, cy] = at(V, 1, 0);
  assert.ok(close(cx, 0) && close(cy, -1), `1 ↦ (${cx}, ${cy}), want (0, −1)`);
});

test("Cayley maps the upper half-plane INTO the unit disc (its defining job)", () => {
  const V = [1, 1, CAYLEY];
  for (const [x, y] of [
    [0, 0.3],
    [2, 1],
    [-4, 0.05],
    [0.1, 9],
    [-0.7, 2.2],
  ]) {
    const [u, v] = at(V, x, y);
    assert.ok(
      Math.hypot(u, v) < 1 + 1e-9,
      `(${x},${y}) in the upper half-plane mapped outside the disc: |${u}+${v}i|`,
    );
  }
  // and the real axis lands ON the unit circle
  for (const x of [-3, -0.5, 0, 1, 7]) {
    const [u, v] = at(V, x, 0);
    assert.ok(
      close(Math.hypot(u, v), 1),
      `real ${x} ↦ |·| = ${Math.hypot(u, v)}`,
    );
  }
});

test("Cayley at Order 2 is the corpus's cayley2 — (z²−i)/(z²+i)", () => {
  // Independent reference: square first in complex arithmetic, then apply the
  // Order-1 Cayley. The op must agree with the composition.
  const sq = (x, y) => [x * x - y * y, 2 * x * y];
  for (const [x, y] of [
    [1, 0],
    [0.6, -1.3],
    [-2.2, 0.4],
    [0.05, 0.05],
  ]) {
    const [ux, uy] = sq(x, y);
    const want = at([1, 1, CAYLEY], ux, uy); // Cayley applied to z²
    const got = at([2, 1, CAYLEY], x, y); // the op at Order 2
    assert.ok(
      close(got[0], want[0], 1e-8) && close(got[1], want[1], 1e-8),
      `(${x},${y}): got ${got}, want ${want}`,
    );
  }
});

test("Murl at Order 1 IS the published curl variation z/(1+Cz)", () => {
  // flam3 variation 39 `curl` (Draves & Reckase, The Fractal Flame Algorithm,
  // Appendix) is z/(1 + c₁z + c₂z²). At N = 1 the internal rescale c′ = C is
  // the identity and murl collapses to curl's c₁ = C, c₂ = 0 slice — times the
  // (1 + C) normalisation. That equivalence is why the name reads as "Möbius
  // curl"; it is derived, not cited (no published source states it).
  for (const C of [1, 0.5, -0.75, 2]) {
    for (const [x, y] of [
      [1, 0],
      [3, 0],
      [0.4, 0.9],
      [-1.5, 0.2],
    ]) {
      const dr = 1 + C * x,
        di = C * y;
      const den = dr * dr + di * di;
      const A = 1 + C;
      const want = [
        (A * (x * dr + y * di)) / den,
        (A * (y * dr - x * di)) / den,
      ];
      const got = at([1, C, MURL], x, y);
      assert.ok(
        close(got[0], want[0], 1e-8) && close(got[1], want[1], 1e-8),
        `C=${C} (${x},${y}): got ${got}, want ${want}`,
      );
    }
  }
});

test("Murl rescales C by 1/(Order−1) — the easy-to-miss internal step", () => {
  // f(ζ) = (1+c′)ζ/(1+c′ζ^N) with c′ = C/(N−1). Pinned directly: the op at
  // (N, C) must equal the hand-rolled reciprocal built from c′, NOT from C.
  for (const N of [2, 3, 5, 8])
    for (const C of [1, -0.6, 2]) {
      const cp = C / (N - 1);
      for (const [x, y] of [
        [0.7, 0.2],
        [-1.1, 0.9],
        [0.3, -0.45],
      ]) {
        // ζ^N by de Moivre
        const r = Math.hypot(x, y) ** N,
          th = Math.atan2(y, x) * N;
        const ur = r * Math.cos(th),
          ui = r * Math.sin(th);
        const dr = 1 + cp * ur,
          di = cp * ui;
        const den = dr * dr + di * di;
        const A = 1 + cp;
        const want = [
          (A * (x * dr + y * di)) / den,
          (A * (y * dr - x * di)) / den,
        ];
        const got = at([N, C, MURL], x, y);
        assert.ok(
          close(got[0], want[0], 1e-8) && close(got[1], want[1], 1e-8),
          `N=${N} C=${C} (${x},${y}): got ${got}, want ${want}`,
        );
      }
    }
});

test("Murl and Murl2 COINCIDE at Order 2 — the murl2_fast ruling", () => {
  // c′ = C/(2−1) = C and 2/N = 1, so the two variants are the same map at
  // N = 2. This is the independent confirmation that the corpus's
  // `murl2_fast` really is "a fast Order=2 specialization of murl" — one
  // Order value, not a separate formula. (They part company for C < −1, where
  // murl2's |1+C| normalisation differs in sign from murl's 1+C.)
  for (const C of [-0.9, -0.25, 0, 0.5, 1, 2])
    for (const [x, y] of [
      [1, 0],
      [0.4, 0.9],
      [-1.5, 0.2],
      [2.2, -1.7],
    ]) {
      const a = at([2, C, MURL], x, y);
      const b = at([2, C, MURL2], x, y);
      assert.ok(
        close(a[0], b[0], 1e-9) && close(a[1], b[1], 1e-9),
        `C=${C} (${x},${y}): murl ${a} ≠ murl2 ${b}`,
      );
    }
});

test("Murl2 is the (2/N)-power reciprocal on the UNRESCALED C", () => {
  // f(ζ) = |1+C|^(2/N)·ζ/(1+C·ζ^N)^(2/N). Real-axis closed form at C = 1,
  // N = 4: 2^(1/2)·x/(1+x⁴)^(1/2).
  for (const x of [0.5, 1, 2]) {
    const [u, v] = at([4, 1, MURL2], x, 0);
    const want = (Math.SQRT2 * x) / Math.sqrt(1 + x ** 4);
    assert.ok(close(u, want, 1e-9), `x=${x}: got ${u}, want ${want}`);
    assert.ok(close(v, 0), `x=${x} must stay real, got ${v}`);
  }
});

test("the normalisation pins f = identity on the unit circle where ζ^N = 1", () => {
  // Both swirls carry a normalisation ((1+c′), |1+C|^(2/N)) whose whole job is
  // to make the denominator cancel at ζ^N = 1 — so ζ = 1 is a fixed point for
  // every C. Without it, moving the C slider rescales the figure instead of
  // reshaping it.
  for (const C of [0.25, 1, 2, -0.5])
    for (const N of [1, 2, 3, 6]) {
      const [u, v] = at([N, C, MURL], 1, 0);
      assert.ok(
        close(u, 1, 1e-9) && close(v, 0, 1e-9),
        `murl N=${N} C=${C} ↦ ${u},${v}`,
      );
      const [p, q] = at([N, C, MURL2], 1, 0);
      assert.ok(
        close(p, 1, 1e-9) && close(q, 0, 1e-9),
        `murl2 N=${N} C=${C} ↦ ${p},${q}`,
      );
    }
});

test("both swirls with C = 0 are the identity on the plane", () => {
  // 1 + 0·z^N = 1 and the normalisation is 1 — a real degeneracy, since C = 0
  // sits inside the declared slider range and must not blow up.
  for (const variant of [MURL, MURL2])
    for (const N of [1, 2, 3, 8])
      for (const [x, y] of [
        [0.3, -1.2],
        [4, 4],
        [0, 0],
      ]) {
        const [u, v] = at([N, 0, variant], x, y);
        assert.ok(
          close(u, x) && close(v, y),
          `v${variant} N=${N} (${x},${y}) ↦ (${u},${v})`,
        );
      }
});

test("all three variants stay bounded as |z| → ∞ (why they suit an IFS)", () => {
  // Cayley sends ∞ ↦ 1; both swirls decay. None needs a bounding companion the
  // way an inversion does — that is what makes this op usable on its own.
  for (const N of [1, 2, 5]) {
    const [cx, cy] = at([N, 1, CAYLEY], 1e6, 0);
    assert.ok(
      Math.hypot(cx, cy) < 1.001,
      `cayley N=${N}: |·| = ${Math.hypot(cx, cy)}`,
    );
    for (const variant of [MURL, MURL2]) {
      const [mx, my] = at([N, 1, variant], 1e6, 0);
      assert.ok(
        Math.hypot(mx, my) < 2,
        `v${variant} N=${N}: |·| = ${Math.hypot(mx, my)}`,
      );
    }
  }
});

test("the Z lane rides through untouched, always", () => {
  // "z kept" is explicit in the corpus note for cayley2, and it is what makes
  // the 3-D map non-conformal (hence deApprox).
  for (const V of [
    [1, 1, CAYLEY],
    [2, 1, CAYLEY],
    [3, -0.6, MURL],
    [8, 2, MURL],
  ])
    for (const h of [-4.25, 0, 7])
      assert.equal(run(V, { x: 1.3, y: -0.8, z: h }).z, h, `Z changed at ${V}`);
});

test("REGRESSION: w is never touched — no pointwise |f′| in the bookkeeping", () => {
  // The wave-1 torusInvert lesson. |f′| diverges at the poles of the rational
  // map, and since Z is untouched each pole is a whole VERTICAL LINE in space
  // — pushing it into w would explode w, collapse DE = r/w and render every
  // scene as a flat wall. Sample right up against a pole and demand w is
  // still exactly the input.
  // Cayley pole: z^N = −iC  ⇒  at N=1, C=1 that is z = −i.
  for (const p of [
    { x: 0, y: -1, w: 1.7 }, // exactly the pole
    { x: 1e-7, y: -1 + 1e-7, w: 0.25 }, // just off it
    { x: 0.5, y: 0.4, w: 3.0 },
  ]) {
    assert.equal(run([1, 1, CAYLEY], p).w, p.w, "cayley must not touch w");
    assert.equal(run([1, -1, MURL], p).w, p.w, "murl must not touch w");
  }
});

test("poles are guarded — finite in ⇒ finite out everywhere", () => {
  const pts = [
    { x: 0, y: 0 },
    { x: 0, y: -1 }, // cayley pole at N=1, C=1
    { x: -1, y: 0 }, // murl pole at N=1, C=1
    { x: 1, y: 1 },
    { x: -3.3, y: 2.7 },
    { x: 1e-9, y: -1e-9 },
  ];
  for (const N of [1, 2, 3, 8])
    for (const C of [-2, -0.05, 0, 1, 2])
      for (const variant of [CAYLEY, MURL])
        for (const p of pts) {
          const q = run([N, C, variant], p);
          for (const k of ["x", "y", "z", "w"])
            assert.ok(
              Number.isFinite(q[k]),
              `${k}=${q[k]} at N=${N} C=${C} v=${variant} ${JSON.stringify(p)}`,
            );
        }
});

test("an out-of-range Variant falls back to variant 0 (the terminal else)", () => {
  const base = { x: 0.7, y: -0.35, z: 0.2, w: 1 };
  const v0 = run([2, 1, CAYLEY], base);
  for (const bogus of [9, -4, 2.6]) {
    const q = run([2, 1, bogus], base);
    assert.deepEqual(
      [q.x, q.y, q.z],
      [v0.x, v0.y, v0.z],
      `Variant ${bogus} must land on variant 0`,
    );
  }
});

test("registry contract: approximate DE, w-untouched warp", () => {
  const def = byKey("complexMap");
  assert.equal(def.wRule, "unchanged");
  assert.equal(def.deApprox, true, "non-conformal + a branch cut ⇒ deApprox");
  assert.equal(def.params.length, 3);
  assert.equal(def.category, "warp");
  for (const p of def.params) {
    assert.ok(p.step >= 0.01, `${p.name} step ${p.step} is finer than 0.01`);
    assert.ok(
      close(Math.round(p.default / p.step) * p.step, p.default, 1e-9),
      `${p.name} default ${p.default} is off its ${p.step} grid`,
    );
  }
  const f = { ops: [{ key: "complexMap", values: [2, 1, CAYLEY] }] };
  assert.equal(isApproxDE(f), true, "must tighten the step policy");
  assert.equal(
    isDeSound(f),
    false,
    "an approximate warp must not vouch as sound",
  );
});

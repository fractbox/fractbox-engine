// Deep zoom Phase 4 (DEEP_ZOOM_DF64.md, plan PR-1) — df64 library + law pins.
//
// The WGSL df64 library (shader.js DF64_LIB_WGSL) can't run in Node, so this
// file carries a STEP-FOR-STEP JS mirror of every function with Math.fround
// after each arithmetic op — exact f32 emulation, which IS the semantic
// reference the WGSL is written against. If an algorithm changes in the WGSL,
// its mirror here must change identically (they are kept in the same order,
// same temporaries). The WGSL's df_launder barriers are the IDENTITY in
// exact arithmetic, so the mirror is just the fround chain; whether a GPU
// toolchain actually honors per-op rounding is the device probe's job
// (harness/df64.html — which is how the D5 hazard was CONFIRMED on
// Chrome/Metal and the laundering became mandatory), not Node's.
//
// Run: node --test core/df64.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";
import { lambdaHat, kStarFor, df64Eligible, K_STAR_MAX } from "./stability.js";
import { TOURBILLON } from "./oplist.js";
import { OPERATORS } from "./operators.js";
import {
  F32_QUANTUM,
  DF64_QUANTUM,
  zoomHeadroom,
  zoomMag,
} from "./recenter.js";

const f = Math.fround;

// ── the mirror (1:1 with DF64_LIB_WGSL — same steps, same temporaries) ─────
const two_sum = (a, b) => {
  const s = f(a + b);
  const bb = f(s - a);
  const ea = f(a - f(s - bb));
  const eb = f(b - bb);
  return [s, f(ea + eb)];
};
const quick_two_sum = (a, b) => {
  const s = f(a + b);
  return [s, f(b - f(s - a))];
};
const df_split = (a) => {
  const t = f(a * 4097.0);
  const hi = f(t - f(t - a));
  return [hi, f(a - hi)];
};
const two_prod = (a, b) => {
  const p = f(a * b);
  const aa = df_split(a);
  const bb = df_split(b);
  const e1 = f(f(aa[0] * bb[0]) - p);
  const e2 = f(e1 + f(aa[0] * bb[1]));
  const e3 = f(e2 + f(aa[1] * bb[0]));
  return [p, f(e3 + f(aa[1] * bb[1]))];
};
const df_add = (a, b) => {
  const s = two_sum(a[0], b[0]);
  const t = two_sum(a[1], b[1]);
  const s2 = quick_two_sum(s[0], f(s[1] + t[0]));
  return quick_two_sum(s2[0], f(s2[1] + t[1]));
};
const df_neg = (a) => [-a[0], -a[1]];
const df_sub = (a, b) => df_add(a, df_neg(b));
const df_add_f32 = (a, b) => {
  const s = two_sum(a[0], b);
  return quick_two_sum(s[0], f(s[1] + a[1]));
};
const df_mul = (a, b) => {
  const p = two_prod(a[0], b[0]);
  const c1 = f(a[0] * b[1]);
  const c2 = f(a[1] * b[0]);
  const e = f(p[1] + f(c1 + c2));
  return quick_two_sum(p[0], e);
};
const df_mul_f32 = (a, b) => {
  const p = two_prod(a[0], b);
  const e = f(p[1] + f(a[1] * b));
  return quick_two_sum(p[0], e);
};
const df_abs = (a) => {
  let neg = a[0] < 0;
  if (a[0] === 0) neg = a[1] < 0;
  return neg ? df_neg(a) : a;
};
const df_lt = (a, b) => {
  const d = df_sub(a, b);
  return d[0] < 0 || (d[0] === 0 && d[1] < 0);
};
const df_div = (a, b) => {
  const q1 = f(a[0] / b[0]);
  const r = df_sub(a, df_mul_f32(b, q1));
  const q2 = f(f(r[0] + r[1]) / b[0]);
  return quick_two_sum(q1, q2);
};

// f64 → normalized df pair (hi = fround(v), lo = fround(v − hi); lo exact
// for the magnitudes in play). Same split writeGlobals will use (plan PR-2).
const toDf = (v) => {
  const hi = f(v);
  return [hi, f(v - hi)];
};
const val = (a) => a[0] + a[1]; // exact in f64 (both components are f32)

// ── EFT exactness — these are EXACT identities, not error bounds ───────────

test("two_sum is an exact error-free transformation", () => {
  const pairs = [
    [f(1), f(2 ** -24)], // lo at the ulp boundary
    [f(0.1), f(0.2)], // both inexact decimals
    [f(1e8), f(-1e8 + 4096)], // near-total cancellation
    [f(3.14159265), f(-3.1415)], // partial cancellation
    [f(1e-30), f(1)], // |a| ≪ |b| (order matters to the algorithm)
    [f(-7.5), f(7.5)], // exact zero
    [f(2 ** 100), f(-(2 ** 76))], // wide exponent gap
  ];
  for (const [a, b] of pairs) {
    const [s, e] = two_sum(a, b);
    // f32+f32 is exact in f64, and TwoSum's error term is exactly the
    // rounding error — so s + e must equal a + b with NO tolerance.
    assert.equal(s + e, a + b, `two_sum(${a}, ${b})`);
    assert.equal(s, f(a + b), "hi must be the f32 rounding of the sum");
  }
});

test("two_prod is an exact error-free transformation", () => {
  const pairs = [
    [f(0.1), f(0.3)],
    [f(1 + 2 ** -12), f(1 - 2 ** -12)], // exercises the split boundary
    [f(12345.678), f(-0.0009765625)],
    [f(1e10), f(3e-10)],
    [f(4097), f(4097)], // the splitter constant itself
    [f(1.5), f(2 ** -100)], // tiny but normal product path
  ];
  for (const [a, b] of pairs) {
    const [p, e] = two_prod(a, b);
    // f32×f32 needs ≤48 mantissa bits — exact in f64; Dekker's error term
    // recovers the f32 rounding error exactly (no overflow at these ranges).
    assert.equal(p + e, a * b, `two_prod(${a}, ${b})`);
    assert.equal(p, f(a * b), "hi must be the f32 rounding of the product");
  }
});

// ── df64 accuracy vs native f64 (the ~49-bit claim) ────────────────────────

// Deep-zoom-regime operands: O(1) orbit coordinates with structure far below
// f32's 24 bits — exactly what the op loop will feed these functions.
const OPERANDS = [
  1.000000123456789,
  -0.7853981633974483,
  2.718281828459045,
  1e-9 + 1e-16,
  -3.0000000001,
  0.1234567890123456,
  1.999999999999,
  -1e6 * Math.PI,
  6.62607015e-2,
];

test("df_add / df_sub track f64 to ~2⁻⁴⁶ relative", () => {
  for (const va of OPERANDS) {
    for (const vb of OPERANDS) {
      const ref = va + vb;
      const got = val(df_add(toDf(va), toDf(vb)));
      const tol = Math.max(Math.abs(ref), 1e-12) * 2 ** -46;
      assert.ok(
        Math.abs(got - ref) <= tol,
        `df_add(${va}, ${vb}): got ${got}, ref ${ref}`,
      );
      const refS = va - vb;
      const gotS = val(df_sub(toDf(va), toDf(vb)));
      const tolS = Math.max(Math.abs(refS), 1e-12) * 2 ** -46;
      assert.ok(Math.abs(gotS - refS) <= tolS, `df_sub(${va}, ${vb})`);
    }
  }
});

test("df_mul / df_mul_f32 / df_add_f32 track f64 to ~2⁻⁴⁶ relative", () => {
  for (const va of OPERANDS) {
    for (const vb of OPERANDS) {
      const ref = va * vb;
      const got = val(df_mul(toDf(va), toDf(vb)));
      const tol = Math.max(Math.abs(ref), 1e-12) * 2 ** -46;
      assert.ok(
        Math.abs(got - ref) <= tol,
        `df_mul(${va}, ${vb}): got ${got}, ref ${ref}`,
      );
      // the f32-scalar forms (op params, fold constants are plain f32)
      const s = f(vb);
      const refM = va * s;
      const gotM = val(df_mul_f32(toDf(va), s));
      assert.ok(
        Math.abs(gotM - refM) <= Math.max(Math.abs(refM), 1e-12) * 2 ** -46,
        `df_mul_f32(${va}, ${s})`,
      );
      const refA = va + s;
      const gotA = val(df_add_f32(toDf(va), s));
      assert.ok(
        Math.abs(gotA - refA) <= Math.max(Math.abs(refA), 1e-12) * 2 ** -46,
        `df_add_f32(${va}, ${s})`,
      );
    }
  }
});

test("df64 beats plain f32 by orders of magnitude on a fold-like chain", () => {
  // A miniature Mandelbox-ish recurrence: x ← |x·s + c|, 20 iterations —
  // the shape of error accumulation the marcher will see.
  const s = f(-1.9),
    c = f(0.4321);
  let x64 = 1.0000000123,
    x32 = f(x64),
    xdf = toDf(x64);
  for (let i = 0; i < 20; i++) {
    x64 = Math.abs(x64 * s + c);
    x32 = f(Math.abs(f(f(x32 * s) + c)));
    xdf = df_abs(df_add_f32(df_mul_f32(xdf, s), c));
  }
  const errDf = Math.abs(val(xdf) - x64);
  const err32 = Math.abs(x32 - x64);
  assert.ok(
    errDf < err32 / 1e4,
    `df64 chain error ${errDf} should be ≪ f32's ${err32}`,
  );
});

test("df_abs consults lo when hi == 0 (exact fold boundary)", () => {
  // hi exactly 0, sign carried entirely by lo — the spec's §4a-1 edge.
  assert.deepEqual(df_abs([0, -(2 ** -30)]), [-0, 2 ** -30]);
  assert.deepEqual(df_abs([0, 2 ** -30]), [0, 2 ** -30]);
  assert.deepEqual(df_abs([-1.5, 2 ** -30]), [1.5, -(2 ** -30)]);
  assert.deepEqual(df_abs([1.5, -(2 ** -30)]), [1.5, -(2 ** -30)]);
  // the recovered VALUE is |v| in both hi==0 branches
  assert.equal(val(df_abs([0, -(2 ** -30)])), 2 ** -30);
});

test("hi/lo split reconstructs deep-zoom offsets to the df64 quantum", () => {
  // A two-term f32 split carries ~48 mantissa bits — NOT the f64's 53, so
  // reconstruction is |v|·2⁻⁴⁸-accurate, not exact. That residual IS the
  // df64 wall (the spec's 2⁻⁴⁹ quantum, ~×10¹³); pin it as a bound, and pin
  // exactness for values that genuinely fit in 48 bits.
  const offsets = [1e6 * Math.PI, -123456.789012345, 0.1, 1.2345678901];
  for (const v of offsets) {
    const [hi, lo] = toDf(v);
    assert.equal(hi, f(v));
    assert.ok(
      Math.abs(hi + lo - v) <= Math.abs(v) * 2 ** -47,
      `split of ${v} must reconstruct to the df64 quantum`,
    );
    // lo really is sub-f32 information
    assert.ok(Math.abs(lo) <= Math.abs(hi) * 2 ** -23 || hi === 0);
  }
  // 48-bit-representable values split EXACTLY.
  const exact = [2 ** 20 + 2 ** -20, 1.5, -4096.03125];
  for (const v of exact) {
    const [hi, lo] = toDf(v);
    assert.equal(hi + lo, v, `48-bit value ${v} must split exactly`);
  }
});

// ── the k* / λ̂ laws (stability.js) ─────────────────────────────────────────

test("kStarFor: zero above the wall, clamped, monotone in magnification", () => {
  assert.equal(kStarFor(1, 3, 64), 0); // no magnification
  assert.equal(kStarFor(2 ** 20, 3, 64), 0); // still above the noise floor
  // the field-measured boundary case: headroom 0.85 (mag·2⁻²⁴ ≈ 0.59) MUST
  // yield k* ≥ 1 — the unmargined law returned 0 here and shipped shattered
  // pixels with the df64 chip dark (the PR-4 preview report)
  assert.ok(kStarFor(9.92e6, 2, 27) >= 1);
  assert.equal(kStarFor(2 ** 26, 2, 64), 6); // 2 bits deficit + 4 margin at λ̂=2
  // ×10¹³, λ̂=3 — the spec's worked band (§"Where the wall is", 11–17ish)
  const k13 = kStarFor(1e13, 3, 64);
  assert.equal(k13, 15); // (19.2 + 4 margin bits) / log2(3)
  assert.ok(k13 >= 11 && k13 <= 17);
  // monotone in mag
  assert.ok(kStarFor(1e10, 3, 64) <= k13);
  // clamped to iters
  assert.equal(kStarFor(1e30, 1.01, 20), 20);
  // λ̂ ≤ 1: signal never outgrows the noise — all-df64
  assert.equal(kStarFor(1e13, 1, 24), 24);
  assert.equal(kStarFor(1e13, 0.5, 24), 24);
  // degenerate inputs
  assert.equal(kStarFor(0, 3, 64), 0);
  assert.equal(kStarFor(1e13, 3, 0), 0);
});

test("lambdaHat is a LOWER bound: scale exact, sphereFold min-factor, isometries 1", () => {
  const F = (ops) => ({ ops });
  // isometries only → 1
  assert.equal(
    lambdaHat(F([{ key: "boxFold", values: [1] }, { key: "absFold" }])),
    1,
  );
  // scale is exact: |−2.5|
  assert.equal(lambdaHat(F([{ key: "scale", values: [-2.5] }])), 2.5);
  // normal sphereFold (fixedR > minR) amplification is CONDITIONAL — its
  // guaranteed factor is 1, so it must NOT raise λ̂ (raising λ̂ cuts k*
  // below safe — the direction the spec's PR-1 correction pins down).
  assert.equal(
    lambdaHat(
      F([
        { key: "scale", values: [-2.5] },
        { key: "sphereFold", values: [0.5, 1.0] },
        { key: "boxFold", values: [1] },
      ]),
    ),
    2.5,
  );
  // INVERTED sphereFold params (fixedR < minR) contract — that lowers λ̂.
  assert.equal(lambdaHat(F([{ key: "sphereFold", values: [1.0, 0.5] }])), 0.25);
  // muted ops don't count
  assert.equal(
    lambdaHat(F([{ key: "scale", values: [-2.5], muted: true }])),
    1,
  );
  // registry defaults fill absent values (scale default is 2 in the registry
  // — pin via the no-values form staying > 1 rather than a magic number)
  assert.ok(lambdaHat(F([{ key: "scale" }])) > 1);
});

test("kStarFor: K_STAR_MAX bounds EVERY branch (unbounded-budget guard)", () => {
  // λ̂ ≤ 1 (e.g. the 4b inversions) otherwise returns the FULL auto-detail
  // iteration count, which reaches 51+ at ×10¹² — a budget set by another
  // subsystem entirely. Bound it. (NB: df64 frame cost was MEASURED equal to
  // f32 at depth, so this is insurance, not a fix for a measured stall.)
  for (const lam of [0, 0.25, 1, 1.01, 2, 3, 8]) {
    for (const iters of [12, 28, 51, 200]) {
      for (const mag of [1e3, 1e7, 1e13, 1e30]) {
        const k = kStarFor(mag, lam, iters);
        assert.ok(
          k <= K_STAR_MAX,
          `k*=${k} exceeded ceiling (λ̂=${lam}, iters=${iters}, mag=${mag})`,
        );
        assert.ok(k <= iters, "still clamped to iters");
        assert.ok(k >= 0 && Number.isInteger(k), "k* is a non-negative int");
      }
    }
  }
  // the exact regression: λ̂ = 0 at ×10¹² with iters 51 was 51, now the ceiling
  assert.equal(kStarFor(4.88e13, 0, 51), K_STAR_MAX);
  // a shorter loop still wins — the ceiling must not INFLATE k* past iters
  assert.equal(kStarFor(4.88e13, 0, 5), 5);
  // and the verified-good 4a case is untouched: Tourbillon (λ̂=2) @ ×10¹²
  assert.equal(kStarFor(4.88e13, 2, 51), 26);
  assert.ok(26 <= K_STAR_MAX, "ceiling must not regress shipped 4a depth");
});

test("4b inversions are NOT twinned or eligible (dropped — issue #459)", () => {
  // The Tier B₁ twins were dropped after the PR #422 post-mortem: their
  // algebra was verified correct, but λ̂ = 0 demands FULL-loop df64
  // (K_STAR_MAX truncation = measured f32-grade quantization plates) at a
  // frame cost no renderpolicy governor bounds yet (= the GPU-wedge crash).
  // Do NOT re-add wgslDf to either op without issue #459's prerequisites:
  // full-loop k* for λ̂ = 0 AND a df64 cost governor, gated by an
  // uncapped-k* render diff.
  for (const key of ["sphereInv", "radialInvert"]) {
    const def = OPERATORS.find((o) => o.key === key);
    assert.ok(def, `${key} still ships as an f32 op`);
    assert.ok(!def.wgslDf, `${key} must NOT carry a df64 twin (issue #459)`);
    assert.equal(
      df64Eligible({ ops: [{ key }, { key: "boxFold", values: [1] }] }),
      false,
      `${key} formulas stay df64-ineligible`,
    );
    // λ̂ falls through to the generic rules (the branch went with the twins)
    assert.equal(lambdaHat({ ops: [{ key }] }), 1, `${key} λ̂ fallthrough`);
  }
  // The λ̂ = 0 CONTRACT stays supported (degenerate minR→0 sphereFold can
  // still approach it): every iteration must run df64.
  assert.equal(kStarFor(1e9, 0, 14), 14);
});

test("df_div tracks f64 to ~2⁻⁴⁶ relative (sphereFold's data-dependent k)", () => {
  for (const va of OPERANDS) {
    for (const vb of OPERANDS) {
      const ref = va / vb;
      const got = val(df_div(toDf(va), toDf(vb)));
      const tol = Math.max(Math.abs(ref), 1e-12) * 2 ** -45;
      assert.ok(
        Math.abs(got - ref) <= tol,
        `df_div(${va}, ${vb}): got ${got}, ref ${ref}`,
      );
    }
  }
});

test("df_lt resolves orderings f32 cannot (the fold-decision compare)", () => {
  const a = 1.000000123456789; // differs from b only below f32
  const b = 1.000000123456999;
  assert.equal(f(a), f(b), "premise: f32 cannot tell these apart");
  assert.ok(df_lt(toDf(a), toDf(b)));
  assert.ok(!df_lt(toDf(b), toDf(a)));
  assert.ok(!df_lt(toDf(a), toDf(a)));
  // hi ties, sign lives in lo
  assert.ok(df_lt([1.5, -1e-10], [1.5, 1e-10]));
});

test("df64Eligible: flat + analytic-IFS + all-ops-twinned (TOURBILLON is the fixture)", () => {
  // the SHIPPED preset — the non-Julia addC case the plan's D2 fix exists for
  assert.equal(df64Eligible(TOURBILLON), true);
  // any active op without a twin disqualifies (kaleido: data-dependent atan2)
  assert.equal(
    df64Eligible({
      ...TOURBILLON,
      ops: [...TOURBILLON.ops, { key: "kaleido", values: [6] }],
    }),
    false,
  );
  // …but MUTED untwinned ops are fine (muted ops never run)
  assert.equal(
    df64Eligible({
      ...TOURBILLON,
      ops: [...TOURBILLON.ops, { key: "kaleido", values: [6], muted: true }],
    }),
    true,
  );
  // hybrid / scene / non-IFS DE / empty are all out (plan D1)
  assert.equal(
    df64Eligible({ ...TOURBILLON, hybrid: { b: { ops: [] } } }),
    false,
  );
  assert.equal(df64Eligible({ ...TOURBILLON, objects: [{}] }), false);
  assert.equal(
    df64Eligible({
      ...TOURBILLON,
      deOption: 0,
      ops: [{ key: "mandelbulb", values: [8] }],
    }),
    false,
  );
  assert.equal(df64Eligible({ ...TOURBILLON, ops: [] }), false);
  assert.equal(df64Eligible(null), false);
});

// ── the headroom law (recenter.js) ──────────────────────────────────────────

test("zoomHeadroom reproduces the shipped f32 wall law and links to zoomMag", () => {
  const cam = {
    target: [1.2, -0.4, 0.7],
    dist: 1e-5,
    fovDeg: 42,
    heightPx: 900,
  };
  const h = zoomHeadroom(cam);
  // identity: headroom = 1 / (quantum · mag)
  assert.ok(Math.abs(h - 1 / (F32_QUANTUM * zoomMag(cam))) < h * 1e-12);
  // df64 moves the wall by exactly the quantum ratio (2²⁶)
  const h64 = zoomHeadroom(cam, DF64_QUANTUM);
  assert.ok(Math.abs(h64 / h - F32_QUANTUM / DF64_QUANTUM) < 1e-6 * (h64 / h));
  assert.equal(F32_QUANTUM / DF64_QUANTUM, 2 ** 26);
  // near-origin targets floor the orbit scale at 1 (matches headroomFor)
  const near = { ...cam, target: [1e-9, 0, 0] };
  const far = { ...cam, target: [0, 0, 0] };
  assert.equal(zoomHeadroom(near), zoomHeadroom(far));
  // deeper (smaller dist) → less headroom, linearly
  const deeper = { ...cam, dist: cam.dist / 10 };
  assert.ok(Math.abs(zoomHeadroom(deeper) - h / 10) < h * 1e-9);
});

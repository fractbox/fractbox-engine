// Guard for the ruckerBulb op (id 62) — the first operator to spend the opAux
// overflow lane (docs/planning/OP_PARAM_ENCODING.md).
//
// It closes the MB3D corpus file `Ruckerbulb`, whose entry in
// docs/planning/data/corpus_coverage_2026-07-13.json reads:
//
//   "Trig bulb with azimuth atan(-x,z)*zAnglePow, polar atan(y,x)*Power,
//    negated x/y outputs, separate angle powers (P=2 default). bulbAxis
//    approximates the shape family; exact needs per-angle powers + con[vention]"
//
// TRIGBULB_SPIKE.md:65-72 measured ~5 dof and, against a 3-slot ABI, pinned the
// TRUNCATED encoding ruckerBulb(Power, AziPow, ZMul), recording the radial-power
// selector and the 4th angle flavor as ACCEPTED LOSSES. This op restores both.
//
// The closed forms below are re-derived here from the published spherical-power
// bulb construction — deliberately NOT read back out of the op body, so the two
// can disagree. Named *.test.mjs so sync_web_core.sh skips it.
// Run: node --test core/ruckerbulb.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";
import { applyOp } from "./cpuorbit.js";
import { byKey, isEscapeTime, isDeSound, W_BULB } from "./operators.js";
import { MAX_OP_PARAMS_INLINE } from "./limits.js";

const P = { x: 0.5, y: 0.4, z: 0.7, w: 1.0 };
const run = (key, values, pt = P) => {
  const s = { ...pt };
  applyOp(key, values, s);
  return s;
};
const close = (a, b, eps = 1e-12) => Math.abs(a - b) < eps;
const same = (a, b, eps = 1e-12) =>
  ["x", "y", "z", "w"].every((k) => close(a[k], b[k], eps));

// Independent re-derivation. z-polar spherical power: the radius maps r → r^q
// with q the SELECTED exponent, the azimuth φ = atan2(y,x) is multiplied by the
// azimuth's own power, and the polar angle θ is multiplied by Power. Which of
// sin/cos lands on the pole, and how θ is measured, is the convention. ZMul
// scales the pole component on the way out.
function rucker(pt, power, aziPow, zMul, radialSel, conv) {
  const r = Math.hypot(pt.x, pt.y, pt.z);
  const u = Math.max(-1, Math.min(1, pt.z / r));
  // conv 3 is the flavor the spike could not afford: a SIGNED planar angle from
  // the two-argument arctangent, range (-π, π], instead of the unsigned acos.
  const base =
    conv === 2
      ? Math.asin(u)
      : conv === 3
        ? Math.atan2(-pt.x, pt.z)
        : Math.acos(u);
  const th = base * power;
  const ph = Math.atan2(pt.y, pt.x) * power * aziPow;
  const q = radialSel === 1 ? power * aziPow : power;
  const rn = Math.pow(r, q);
  const eq = conv === 0 ? Math.sin(th) : Math.cos(th);
  const pole = conv === 0 ? Math.cos(th) : Math.sin(th);
  return {
    x: rn * eq * Math.cos(ph),
    y: rn * eq * Math.sin(ph),
    z: rn * pole * zMul,
    // ‖J‖ <= max(1,|ZMul|)·q·r^(q-1) — the diag(1,1,ZMul) post-scale bound.
    w: ((q * rn) / r) * Math.max(1, Math.abs(zMul)) + 1, // starting w = 1
  };
}

// ── Closed forms ─────────────────────────────────────────────────────────────

test("matches the hand-derived closed form across the whole enum cross", () => {
  for (const conv of [0, 1, 2, 3])
    for (const rsel of [0, 1])
      for (const [pw, az, zm] of [
        [8, 1, 1],
        [8, 1.5, 1],
        [6, 0.75, -1.25],
        [2, 2, 0.4],
      ]) {
        const got = run("ruckerBulb", [pw, az, zm, rsel, conv]);
        const want = rucker(P, pw, az, zm, rsel, conv);
        for (const k of ["x", "y", "z", "w"])
          assert.ok(
            close(got[k], want[k]),
            `conv=${conv} rsel=${rsel} P=${pw} az=${az} zm=${zm} ${k}: got ${got[k]}, want ${want[k]}`,
          );
      }
});

// ── Degeneracy anchors (the whole reason this encoding is reviewable) ────────

test("ruckerBulb(P,1,1,0,0) IS bulbAxis(P,0,0) — the convention-0 anchor", () => {
  // The brief's degeneracy requirement: at Convention 0 / RadialSel 0 the op
  // must reduce to bulbAxis's cos convention. Pinned numerically, not by eye.
  for (const pw of [2, 5.5, 8, 16])
    assert.ok(
      same(run("ruckerBulb", [pw, 1, 1, 0, 0]), run("bulbAxis", [pw, 0, 0])),
      `Power ${pw} must match bulbAxis's cos convention exactly`,
    );
});

test("...and therefore mandelbulbPower(P) too", () => {
  assert.ok(
    same(run("ruckerBulb", [8, 1, 1, 0, 0]), run("mandelbulbPower", [8])),
    "the classic bulb must be reachable from this op",
  );
});

test("conventions 0/1/2 agree with bulbAxis's, so the two ops never diverge", () => {
  // bulbAxis (id 29) is the convention-semantics reference: 0 cos-polar,
  // 1 sin-polar (NormBulb), 2 asin-latitude (sine bulb). ruckerBulb reproduces
  // all three at its own defaults; only flavor 3 is new.
  for (const conv of [0, 1, 2])
    assert.ok(
      same(
        run("ruckerBulb", [8, 1, 1, 0, conv]),
        run("bulbAxis", [8, 0, conv]),
      ),
      `convention ${conv} must match bulbAxis exactly`,
    );
});

test("back-compat: short payloads read as the defaults (arity migration)", () => {
  const full = run("ruckerBulb", [8, 1, 1, 0, 0]);
  assert.ok(same(run("ruckerBulb", [8]), full), "1-value payload");
  assert.ok(same(run("ruckerBulb", [8, 1]), full), "2-value payload");
  assert.ok(
    same(run("ruckerBulb", [8, 1, 1]), full),
    "the spike's 3-value pin",
  );
  assert.ok(same(run("ruckerBulb", [8, 1, 1, 0]), full), "4-value payload");
});

// ── The enums are real ───────────────────────────────────────────────────────

test("all four conventions are genuinely distinct maps", () => {
  // Flavor 3 must not collapse onto 0/1/2. Any UNSIGNED reformulation —
  // atan2(hypot(x,y), z) or atan2(z, hypot(x,y)) — is algebraically just
  // acos/asin again and would silently be a duplicate; the SIGN is the content.
  const out = [0, 1, 2, 3].map((c) => run("ruckerBulb", [8, 1, 1, 0, c]));
  for (let i = 0; i < 4; i++)
    for (let j = i + 1; j < 4; j++)
      assert.ok(
        !close(out[i].z, out[j].z, 1e-6) || !close(out[i].x, out[j].x, 1e-6),
        `convention ${i} and ${j} produce the same point — not a real enum value`,
      );
});

test("an out-of-range enum falls back to its 0 branch", () => {
  assert.ok(
    same(
      run("ruckerBulb", [8, 1, 1, 0, 9]),
      run("ruckerBulb", [8, 1, 1, 0, 0]),
    ),
  );
  assert.ok(
    same(
      run("ruckerBulb", [8, 1.5, 1, 7, 0]),
      run("ruckerBulb", [8, 1.5, 1, 0, 0]),
    ),
  );
});

test("RadialSel is inert at AziPow 1 and a real choice otherwise", () => {
  // Honest property, worth pinning both ways: the selector picks between Power
  // and Power·AziPow, which COINCIDE at AziPow = 1.
  assert.ok(
    same(
      run("ruckerBulb", [8, 1, 1, 0, 0]),
      run("ruckerBulb", [8, 1, 1, 1, 0]),
    ),
    "the two powers coincide at AziPow 1, so the selector must do nothing",
  );
  const a = run("ruckerBulb", [8, 1.5, 1, 0, 0]);
  const b = run("ruckerBulb", [8, 1.5, 1, 1, 0]);
  assert.ok(!close(a.x, b.x, 1e-6) || !close(a.z, b.z, 1e-6));
});

test("AziPow winds the azimuth independently of the lobe count", () => {
  const tied = run("ruckerBulb", [8, 1, 1, 0, 0]);
  const wound = run("ruckerBulb", [8, 2, 1, 0, 0]);
  assert.ok(!close(tied.x, wound.x, 1e-6) || !close(tied.y, wound.y, 1e-6));
  // With RadialSel 0 the radius is untouched by AziPow, so |output| is unchanged.
  assert.ok(
    close(
      Math.hypot(tied.x, tied.y, tied.z),
      Math.hypot(wound.x, wound.y, wound.z),
      1e-9,
    ),
    "a pure azimuth rewind must be radius-preserving",
  );
});

test("ZMul scales the pole and flips it when negative", () => {
  const base = run("ruckerBulb", [8, 1, 1, 0, 0]);
  const half = run("ruckerBulb", [8, 1, 0.5, 0, 0]);
  const flip = run("ruckerBulb", [8, 1, -1, 0, 0]);
  assert.ok(close(half.z, base.z * 0.5));
  assert.ok(close(flip.z, -base.z));
  assert.ok(close(half.x, base.x) && close(flip.y, base.y), "x/y untouched");
});

// ── The w-rule (TRIGBULB_SPIKE.md §w-rule) ───────────────────────────────────

test("w is convention-invariant — the angle flavor never touches the radial dr", () => {
  const start = { x: 0.31, y: -0.62, z: 0.44, w: 1.7 };
  const ws = [0, 1, 2, 3].map(
    (c) => run("ruckerBulb", [7, 1, 1, 0, c], start).w,
  );
  assert.ok(
    ws.every((x) => close(x, ws[0])),
    `w must not depend on Convention, got ${ws}`,
  );
});

test("w tracks the SELECTED radial exponent, and equals bulbAxis's at defaults", () => {
  assert.ok(
    close(run("ruckerBulb", [8, 1, 1, 0, 0]).w, run("bulbAxis", [8, 0, 0]).w),
    "at the defaults the bound must be the plain W_BULB update, undiluted",
  );
  // RadialSel 1 exponentiates by Power·AziPow, so the derivative must follow.
  const r = Math.hypot(P.x, P.y, P.z);
  const q = 8 * 1.5;
  assert.ok(
    close(run("ruckerBulb", [8, 1.5, 1, 1, 0]).w, (q * Math.pow(r, q)) / r + 1),
  );
});

test("w carries the ZMul Lipschitz factor, and only when it exceeds 1", () => {
  const base = run("ruckerBulb", [8, 1, 1, 0, 0]).w - 1;
  // |ZMul| <= 1 is a CONTRACTION on the pole — it cannot increase ‖J‖, so the
  // bound must not be loosened (that would inflate the DE and soften the surface).
  assert.ok(close(run("ruckerBulb", [8, 1, 0.25, 0, 0]).w - 1, base));
  assert.ok(close(run("ruckerBulb", [8, 1, -1, 0, 0]).w - 1, base));
  // Past 1 it genuinely stretches, so the bound must grow with it.
  assert.ok(close(run("ruckerBulb", [8, 1, 2, 0, 0]).w - 1, base * 2));
  assert.ok(close(run("ruckerBulb", [8, 1, -1.5, 0, 0]).w - 1, base * 1.5));
});

// ── Registry contract ────────────────────────────────────────────────────────

test("registry contract: 5 params, escape-time bulb, spends the overflow lane", () => {
  const def = byKey("ruckerBulb");
  assert.equal(def.id, 62);
  assert.equal(def.wRule, W_BULB);
  assert.equal(def.category, "power");
  assert.equal(def.params.length, 5);
  assert.ok(
    def.params.length > MAX_OP_PARAMS_INLINE,
    "this op exists to exercise the lane — if it fits inline it proves nothing",
  );
  assert.ok(!("deApprox" in def), "the radial dr is analytic and bounded");
  assert.ok(typeof def.blurb === "string" && def.blurb.trim());

  const f = { ops: [{ key: "ruckerBulb", values: [8, 1, 1, 0, 0] }] };
  assert.equal(
    isEscapeTime(f),
    true,
    "must route the formula to escape-time DE",
  );
  assert.equal(isDeSound(f), false, "a bulb is not the analytic IFS family");

  for (const p of def.params) {
    assert.ok(
      p.step >= 0.01,
      `${p.name} step ${p.step} is finer than the wire grid`,
    );
    assert.ok(
      close(Math.round(p.default / p.step) * p.step, p.default, 1e-9),
      `${p.name} default ${p.default} is off its ${p.step} grid`,
    );
    assert.ok(
      p.default >= p.min && p.default <= p.max,
      `${p.name} default out of range`,
    );
    assert.ok(
      typeof p.tip === "string" && p.tip.trim(),
      `${p.name} needs a tip`,
    );
  }
  // AziPow must stay positive: RadialSel can route it into the RADIAL exponent,
  // where a negative power sends r→0 to a non-finite f32. Negative windings are
  // the mirror isometry (a recipe card), per TRIGBULB_SPIKE.md:39-45.
  assert.ok(
    byKey("ruckerBulb").params[1].min > 0,
    "AziPow must be positive-only",
  );
});

test("finite in, finite out across the declared param ranges", () => {
  const def = byKey("ruckerBulb");
  const grid = (p, n) =>
    Array.from(
      { length: n },
      (_, i) => p.min + ((p.max - p.min) * i) / (n - 1),
    );
  const pts = [
    P,
    { x: 1e-7, y: 1e-7, z: 1e-7, w: 1 }, // the r ≈ 0 guard
    { x: 0, y: 0, z: 0, w: 1 }, // exactly the origin
    { x: -2.4, y: 1.9, z: -3.1, w: 1 },
    { x: 0, y: 0, z: 2, w: 1 }, // on the pole
    { x: 2, y: 0, z: 0, w: 1 }, // on the equator
  ];
  for (const pw of grid(def.params[0], 4))
    for (const az of grid(def.params[1], 4))
      for (const zm of grid(def.params[2], 4))
        for (const rsel of [0, 1])
          for (const conv of [0, 1, 2, 3])
            for (const pt of pts) {
              const s = run("ruckerBulb", [pw, az, zm, rsel, conv], pt);
              for (const k of ["x", "y", "z", "w"])
                assert.ok(
                  Number.isFinite(s[k]),
                  `non-finite ${k} at P=${pw} az=${az} zm=${zm} rsel=${rsel} conv=${conv} pt=${JSON.stringify(pt)}`,
                );
            }
});

test("the origin is a fixed point — the r guard leaves the state alone", () => {
  const s = run("ruckerBulb", [8, 1, 1, 0, 0], { x: 0, y: 0, z: 0, w: 1 });
  assert.deepEqual(s, { x: 0, y: 0, z: 0, w: 1 });
});

// ── 3-emitter mirror ─────────────────────────────────────────────────────────

test("all three emitters read all five params, and only the WGSL uses the lane", () => {
  const def = byKey("ruckerBulb");
  // WGSL author-facing syntax is uniform: p0..p2 inline, p3/p4 rewritten to
  // opAux[o].x/.y by the splicer at emit time (OP_PARAM_ENCODING.md §5.2).
  for (const p of ["op.p0", "op.p1", "op.p2", "op.p3", "op.p4"])
    assert.ok(def.wgsl.includes(p), `wgsl body must read ${p}`);
  // GLSL gets variable-name STRINGS; every param must reach the emitted text.
  const emitted = def.glsl(["A0", "A1", "A2", "A3", "A4"]);
  for (const v of ["A0", "A1", "A2", "A3", "A4"])
    assert.ok(emitted.includes(v), `glsl body must interpolate ${v}`);
  // #553 class-fence, restated locally: a `${...}` must never resolve a branch
  // in JavaScript — v holds strings, so the comparison has to be GLSL text.
  assert.equal(
    /\$\{[^}]*[<>][^}]*\}/.test(def.glsl.toString()),
    false,
    "a JS-side comparison on a param reference is always-false dead code (#553)",
  );
  // The CPU tier takes an untyped array — the lane is invisible there.
  assert.ok(
    !same(
      run("ruckerBulb", [8, 1, 1, 0, 0]),
      run("ruckerBulb", [8, 1, 1, 1, 3]),
    ),
  );
});

// Convention + split-power guard for the bulbAxis op (TRIGBULB_SPIKE.md,
// OP_PARAM_ENCODING.md) — pins the three trig conventions and the two
// independent angle-power multipliers against independently hand-derived
// closed forms, the back-compat degeneracies (short payloads ≡ tied powers ≡
// the pre-wave op ≡ mandelbulbPower at axis 0), the subsumption of
// sphericalTwoStage, and the w-update's invariance to both convention and
// angle powers.
//
// The oracle below is deliberately re-derived from the published bulb
// construction rather than read back out of the op body, so the two can
// disagree.
//
// Named *.test.mjs so it stays out of the apps' served *.js surface. Run: node --test core/bulbaxis.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";
import { applyOp } from "./cpuorbit.js";
import { byKey, W_BULB } from "./operators.js";
import { MAX_OP_PARAMS_INLINE } from "./limits.js";

const P = { x: 0.5, y: 0.4, z: 0.7, w: 1.0 };
const run = (key, values, pt = P) => {
  const s = { ...pt };
  applyOp(key, values, s);
  return s;
};
const close = (a, b, eps = 1e-12) => Math.abs(a - b) < eps;
const KEYS = ["x", "y", "z", "w"];
// Points chosen to exercise every branch: generic, pole, equator, negatives.
const PTS = [
  { x: 0.5, y: 0.4, z: 0.7, w: 1.0 },
  { x: -0.62, y: 0.31, z: -0.44, w: 1.7 },
  { x: 0.0, y: 0.0, z: 0.9, w: 1.0 },
  { x: 0.8, y: -0.6, z: 0.0, w: 1.0 },
  { x: -0.3, y: -0.7, z: -0.2, w: 2.4 },
];

// Independent re-derivation. The op permutes the axis triple so that `up` is
// the pole; the polar angle comes from acos (conventions 0/1) or asin
// (convention 2) of up/r; the convention chooses which of sin/cos lands on the
// pole; and the two angles carry their own power multipliers on top of the
// shared Power:  r→rⁿ,  θ' = n·ThetaMul·θ,  φ' = n·PhiMul·φ.
function bulbRef(pt, n, axis, conv, tm = 1, pm = 1) {
  const r = Math.hypot(pt.x, pt.y, pt.z);
  // axis 0 → pole z, plane (x,y); 1 → pole y, plane (z,x); 2 → pole x, plane (y,z)
  const [up, a, b] =
    axis === 1
      ? [pt.y, pt.z, pt.x]
      : axis === 2
        ? [pt.x, pt.y, pt.z]
        : [pt.z, pt.x, pt.y];
  const u = Math.max(-1, Math.min(1, up / r));
  const th = (conv === 2 ? Math.asin(u) : Math.acos(u)) * n * tm;
  const ph = Math.atan2(b, a) * n * pm;
  const rn = Math.pow(r, n);
  const eq = conv === 0 ? Math.sin(th) : Math.cos(th);
  const pole = conv === 0 ? Math.cos(th) : Math.sin(th);
  const na = rn * eq * Math.cos(ph);
  const nb = rn * eq * Math.sin(ph);
  const nup = rn * pole;
  const out =
    axis === 1
      ? { x: nb, y: nup, z: na }
      : axis === 2
        ? { x: nup, y: na, z: nb }
        : { x: na, y: nb, z: nup };
  // The radial derivative ignores the angle multipliers entirely.
  out.w = ((n * rn) / r) * pt.w + 1;
  return out;
}

const expectMatch = (got, want, label) => {
  for (const k of KEYS)
    assert.ok(
      close(got[k], want[k]),
      `${label} ${k}: got ${got[k]}, want ${want[k]}`,
    );
};
// Degeneracy anchors are bit-exact, not merely close — a multiplier of 1 is an
// IEEE-754 identity, so any drift means the body stopped being a pure widening.
const expectExact = (got, want, label) => {
  for (const k of KEYS)
    assert.equal(
      got[k],
      want[k],
      `${label} ${k}: got ${got[k]}, want ${want[k]}`,
    );
};

// ── Closed forms ──────────────────────────────────────────────────────────

test("every axis × convention × split-power cell matches the hand-derived closed form", () => {
  for (const axis of [0, 1, 2])
    for (const conv of [0, 1, 2])
      for (const [tm, pm] of [
        [1, 1],
        [0.5, 2],
        [-1.5, 0.25],
        [2, -3],
      ])
        for (const pt of PTS)
          expectMatch(
            run("bulbAxis", [8.0, axis, conv, tm, pm], pt),
            bulbRef(pt, 8.0, axis, conv, tm, pm),
            `axis=${axis} conv=${conv} tm=${tm} pm=${pm}`,
          );
});

test("conventions are genuinely distinct maps (1 and 2 differ from 0 and from each other)", () => {
  const [c0, c1, c2] = [0, 1, 2].map((c) =>
    run("bulbAxis", [8.0, 0.0, c, 1.0, 1.0]),
  );
  assert.ok(!close(c0.z, c1.z, 1e-6), "conv 1 must differ from conv 0");
  assert.ok(!close(c0.z, c2.z, 1e-6), "conv 2 must differ from conv 0");
  assert.ok(!close(c1.z, c2.z, 1e-6), "conv 2 must differ from conv 1");
});

// ── Degeneracy anchors ────────────────────────────────────────────────────
// The whole back-compat argument for widening a SHIPPED op: at the tied-power
// default the new op must be the old op bit-for-bit, in every existing cell.

test("ThetaMul = PhiMul = 1 is the pre-wave op EXACTLY, for every axis × convention", () => {
  for (const axis of [0, 1, 2])
    for (const conv of [0, 1, 2])
      for (const pt of PTS)
        expectExact(
          run("bulbAxis", [8.0, axis, conv, 1.0, 1.0], pt),
          bulbRef(pt, 8.0, axis, conv), // oracle at its own tied-power default
          `axis=${axis} conv=${conv}`,
        );
});

test("arity migration: 2-, 3- and 4-value payloads all ≡ the full 5-value default", () => {
  const full = (pt) => run("bulbAxis", [8.0, 1.0, 0.0, 1.0, 1.0], pt);
  for (const pt of PTS) {
    // pre-Convention arity (the pin the Convention add shipped)
    expectExact(run("bulbAxis", [8.0, 1.0], pt), full(pt), "2-value");
    // the arity this wave widens from
    expectExact(run("bulbAxis", [8.0, 1.0, 0.0], pt), full(pt), "3-value");
    // a payload that stopped inside the overflow lane
    expectExact(run("bulbAxis", [8.0, 1.0, 0.0, 1.0], pt), full(pt), "4-value");
  }
});

test("axis 0 + Convention 0 at tied powers reproduces mandelbulbPower exactly", () => {
  for (const pt of PTS)
    expectExact(
      run("bulbAxis", [8.0, 0.0, 0.0, 1.0, 1.0], pt),
      run("mandelbulbPower", [8.0], pt),
      "mandelbulbPower",
    );
});

test("SUBSUMPTION: axis 0 + Convention 0 is sphericalTwoStage(Power, ThetaMul, PhiMul) exactly", () => {
  // The capability TRIGBULB_SPIKE.md:58 rejected on slots ("37 has no free
  // slot"). Widening THIS op delivers the cross instead, so op 37 is a strict
  // special case and the two can never disagree where they overlap.
  for (const [tm, pm] of [
    [1, 1],
    [0.5, 2],
    [-1.5, 0.25],
    [2, -3],
    [0, 1],
    [1, 0],
  ])
    for (const pt of PTS)
      expectExact(
        run("bulbAxis", [6.0, 0.0, 0.0, tm, pm], pt),
        run("sphericalTwoStage", [6.0, tm, pm], pt),
        `twoStage tm=${tm} pm=${pm}`,
      );
});

test("the ruckerBulb cross-op anchors survive the widening", () => {
  // core/ruckerbulb.test.mjs pins ruckerBulb against 3-value bulbAxis calls;
  // this is the same identity restated at the new arity, so a future edit that
  // breaks one breaks both.
  for (const conv of [0, 1, 2])
    for (const pt of PTS)
      expectExact(
        run("ruckerBulb", [8.0, 1.0, 1.0, 0.0, conv], pt),
        run("bulbAxis", [8.0, 0.0, conv, 1.0, 1.0], pt),
        `rucker conv=${conv}`,
      );
});

// ── The split powers are real ─────────────────────────────────────────────

test("ThetaMul and PhiMul each move the map, and move it independently", () => {
  const base = run("bulbAxis", [8.0, 0.0, 0.0, 1.0, 1.0]);
  const tOnly = run("bulbAxis", [8.0, 0.0, 0.0, 1.6, 1.0]);
  const pOnly = run("bulbAxis", [8.0, 0.0, 0.0, 1.0, 1.6]);
  const both = run("bulbAxis", [8.0, 0.0, 0.0, 1.6, 1.6]);
  assert.ok(
    !close(base.z, tOnly.z, 1e-6),
    "ThetaMul must change the pole component",
  );
  assert.ok(!close(base.y, pOnly.y, 1e-6), "PhiMul must change the azimuth");
  assert.ok(
    !close(tOnly.y, both.y, 1e-6),
    "the two must not collapse into one dof",
  );
  // PhiMul is a pure azimuth rotation+winding: it cannot touch the pole.
  assert.ok(
    close(base.z, pOnly.z),
    "PhiMul must leave the pole component alone",
  );
});

test("the multipliers reach any absolute angle power (the encoding claim)", () => {
  // θ' = Power·ThetaMul·θ, so an exemplar wanting an absolute θ-power of 3 on
  // a radial power of 8 is ThetaMul = 3/8 — the property that makes the
  // radial/θ/φ exponents fully independent and RadialSel unnecessary here.
  for (const pt of PTS)
    expectMatch(
      run("bulbAxis", [8.0, 0.0, 0.0, 3 / 8, 5 / 8], pt),
      bulbRef(pt, 8.0, 0, 0, 3 / 8, 5 / 8),
      "absolute powers (θ=3, φ=5, r=8)",
    );
});

test("a signed multiplier is legal and never reaches the radial exponent", () => {
  // Why these are ±4 like sphericalTwoStage's and not positive-only like
  // ruckerBulb's AziPow: nothing here couples an angle power to pow(br, bp).
  for (const [tm, pm] of [
    [-4, -4],
    [-1, 3.2],
    [4, -0.05],
  ])
    for (const pt of PTS) {
      const g = run("bulbAxis", [8.0, 1.0, 2.0, tm, pm], pt);
      for (const k of KEYS)
        assert.ok(
          Number.isFinite(g[k]),
          `tm=${tm} pm=${pm} ${k} must stay finite`,
        );
    }
});

test("finite in, finite out across the param grid and the edge points", () => {
  const edges = [
    ...PTS,
    { x: 0, y: 0, z: 0, w: 1 }, // origin (guarded by br > 1e-9)
    { x: 1e-12, y: 0, z: 0, w: 1 }, // just inside the guard
  ];
  for (const n of [2, 5.5, 8, 16])
    for (const axis of [0, 1, 2])
      for (const conv of [0, 1, 2])
        for (const [tm, pm] of [
          [1, 1],
          [-4, 4],
          [0, 0],
        ])
          for (const pt of edges) {
            const g = run("bulbAxis", [n, axis, conv, tm, pm], pt);
            for (const k of KEYS)
              assert.ok(
                Number.isFinite(g[k]),
                `n=${n} axis=${axis} conv=${conv} tm=${tm} pm=${pm} pt=${JSON.stringify(pt)} ${k}`,
              );
          }
});

// ── The w-rule ────────────────────────────────────────────────────────────

test("w-update is invariant to BOTH convention and the angle powers (W_BULB)", () => {
  // TRIGBULB_SPIKE.md:77-82 — all conventions leave r→rⁿ alone; an angle
  // multiplier is radius-preserving for the same reason. So the analytic dr
  // is one formula for the whole 3×3×ℝ² cell space.
  const start = { x: 0.31, y: -0.62, z: 0.44, w: 1.7 };
  const ws = new Set();
  for (const conv of [0, 1, 2])
    for (const [tm, pm] of [
      [1, 1],
      [3, -2],
      [0, 0.5],
      [-4, 4],
    ])
      ws.add(run("bulbAxis", [7.0, 1.0, conv, tm, pm], start).w);
  assert.equal(
    ws.size,
    1,
    `w must not depend on Convention/ThetaMul/PhiMul, got ${[...ws]}`,
  );
  // …and it is the plain W_BULB update.
  const r = Math.hypot(start.x, start.y, start.z);
  assert.ok(close([...ws][0], ((7 * Math.pow(r, 7)) / r) * start.w + 1));
});

// ── Registry contract ─────────────────────────────────────────────────────

test("the registry entry says what the encoding says", () => {
  const def = byKey("bulbAxis");
  assert.equal(def.id, 29);
  assert.equal(def.wRule, W_BULB);
  assert.equal(def.category, "power");
  assert.equal(def.params.length, 5);
  // If it fit inline it would prove nothing about the overflow lane.
  assert.ok(
    def.params.length > MAX_OP_PARAMS_INLINE,
    "bulbAxis must actually spend the opAux lane",
  );
  assert.equal(def.deApprox, undefined);
  assert.ok(def.blurb && def.blurb.length > 0);
  const [, , , tmp, pmp] = def.params;
  for (const p of [tmp, pmp]) {
    assert.equal(p.default, 1.0, `${p.name} must default to the tied power`);
    assert.ok(p.min < 0 && p.max > 0, `${p.name} must be signed`);
    assert.ok(
      p.step >= 0.01,
      `${p.name} step must survive the 0.01 share grid`,
    );
    assert.ok(p.default >= p.min && p.default <= p.max);
    assert.ok(p.tip && p.tip.length > 0, `${p.name} needs a tip`);
  }
  // The multipliers must match sphericalTwoStage's range, or the subsumption
  // above is only partial.
  const two = byKey("sphericalTwoStage").params;
  assert.equal(tmp.min, two[1].min);
  assert.equal(tmp.max, two[1].max);
  assert.equal(pmp.min, two[2].min);
  assert.equal(pmp.max, two[2].max);
});

// ── 3-emitter mirror ──────────────────────────────────────────────────────

test("all three emitters read all five params", () => {
  const def = byKey("bulbAxis");
  // WGSL keeps the AUTHOR-facing syntax; core/shader.js's splicer rewrites
  // p3/p4 to opAux[o].x/.y at emit time (pinned in core/opaux.test.mjs).
  for (const p of ["op.p0", "op.p1", "op.p2", "op.p3", "op.p4"])
    assert.ok(def.wgsl.includes(p), `wgsl must read ${p}`);
  // GLSL is a string template over variable NAMES — feed sentinels.
  const emitted = def.glsl(["A0", "A1", "A2", "A3", "A4"]);
  for (const s of ["A0", "A1", "A2", "A3", "A4"])
    assert.ok(emitted.includes(s), `glsl must read ${s}`);
  // #553: a JS-side comparison on a param reference is always-false dead code
  // (the kaleido Mirror class of bug — OP_PARAM_ENCODING.md Appendix).
  assert.equal(
    /\$\{[^}]*[<>][^}]*\}/.test(def.glsl.toString()),
    false,
    "a JS-side comparison on a param reference is always-false dead code (#553)",
  );
  // CPU: the two extremes must not collapse.
  const lo = run("bulbAxis", [8.0, 0.0, 0.0, -4.0, -4.0]);
  const hi = run("bulbAxis", [8.0, 0.0, 0.0, 4.0, 4.0]);
  assert.ok(!close(lo.x, hi.x, 1e-6) || !close(lo.z, hi.z, 1e-6));
});

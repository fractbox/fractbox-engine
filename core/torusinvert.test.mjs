// Guards for torusInvert (parity wave 1) — the _toruspinv family.
//
// The corpus ruling this op exists to satisfy: PRIMITIVE_GAPS §1 #2 listed
// _toruspinv1/2/3 under radialInvert, but op 28 is a pure SPHERE inversion, so
// the family was NOT covered (PRIMITIVE_COVERAGE_PLAN.md §3 corrects it). These
// tests pin the closed forms that make the distinction real, plus the
// approximate-DE contract.
//
// Named *.test.mjs so it stays out of the apps' served *.js surface.
// Run: node --test core/torusinvert.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";
import { applyOp } from "./cpuorbit.js";
import { isApproxDE, isDeSound, byKey } from "./operators.js";

const run = (values, pt) => {
  const s = { ...pt };
  applyOp("torusInvert", values, s);
  return s;
};
const P = { x: 0.5, y: 0.4, z: 0.7, w: 1.3 };
const close = (a, b, eps = 1e-12) => Math.abs(a - b) < eps;
// distance from a point to the torus core circle of radius R
const coreDist = (p, R) => Math.hypot(R - Math.hypot(p.x, p.y), p.z);

test("variant 0 is a true inversion: OP·OP′ = Radius² in the meridian plane", () => {
  // The defining property of inversion in a circle (MathWorld, "Inversion").
  // This is what separates torusInvert from radialInvert: the reference circle
  // is the torus CORE CIRCLE, not a point.
  for (const [rad, R] of [
    [1.0, 2.0],
    [0.7, 2.3],
    [1.4, 3.1],
  ]) {
    const q = run([rad, R, 0], P);
    assert.ok(
      close(coreDist(P, R) * coreDist(q, R), rad * rad, 1e-12),
      `Radius ${rad} R ${R}: ${coreDist(P, R) * coreDist(q, R)} ≠ ${rad * rad}`,
    );
  }
});

test("every variant preserves the azimuth (surface of revolution)", () => {
  const th = Math.atan2(P.y, P.x);
  for (const v of [0, 1, 2, 3]) {
    const q = run([1.0, 2.0, v], P);
    assert.ok(
      close(Math.atan2(q.y, q.x), th, 1e-12),
      `variant ${v} rotated the azimuth`,
    );
  }
});

test("variant 0 is involutive where the image stays at ρ′ ≥ 0", () => {
  // Inversion is its own inverse — but only inside the ρ ≥ 0 meridian chart
  // (see the operators.js note: an image thrown past the axis comes back with
  // an unsigned hypot). Defaults sit well inside that region.
  for (const [rad, R] of [
    [1.0, 2.0],
    [0.7, 2.3],
  ]) {
    const back = run([rad, R, 0], run([rad, R, 0], P));
    for (const k of ["x", "y", "z"])
      assert.ok(close(back[k], P[k], 1e-9), `Radius ${rad} R ${R} ${k}`);
  }
});

test("variant 0 fixes the inverting circle itself (d = Radius ⇒ identity)", () => {
  // A point exactly on the reference circle must not move.
  const R = 2.0,
    rad = 0.6,
    a = 0.9;
  const p = {
    x: R + rad * Math.cos(a),
    y: 0,
    z: rad * Math.sin(a),
    w: 1,
  };
  const q = run([rad, R, 0], p);
  for (const k of ["x", "y", "z"])
    assert.ok(close(q[k], p[k], 1e-12), `${k}: ${q[k]} vs ${p[k]}`);
});

test("the pseudo variants match their published closed forms", () => {
  const rad = 1.1,
    R = 2.0;
  const rho = Math.hypot(P.x, P.y),
    u = R - rho,
    d2 = u * u + P.z * P.z,
    d = Math.sqrt(d2);
  // variant 1: xy ·= Radius/d, z ·= d/Radius
  const v1 = run([rad, R, 1], P);
  assert.ok(close(Math.hypot(v1.x, v1.y), rho * (rad / d), 1e-12), "v1 xy");
  assert.ok(close(v1.z, P.z * (d / rad), 1e-12), "v1 z");
  // variant 2: z ·= d·Radius (multiply, not divide)
  const v2 = run([rad, R, 2], P);
  assert.ok(close(Math.hypot(v2.x, v2.y), rho * (rad / d), 1e-12), "v2 xy");
  assert.ok(close(v2.z, P.z * (d * rad), 1e-12), "v2 z");
  // variant 3: the unrooted form — d² wherever variant 1 uses d
  const v3 = run([rad, R, 3], P);
  assert.ok(close(Math.hypot(v3.x, v3.y), rho * (rad / d2), 1e-12), "v3 xy");
  assert.ok(close(v3.z, P.z * (d2 / rad), 1e-12), "v3 z");
});

test("torusInvert is NOT radialInvert (the ruling the op exists to encode)", () => {
  const t = run([1.0, 2.0, 0], P);
  const s = { ...P };
  applyOp("radialInvert", [0, 0, 0], s);
  assert.ok(
    Math.hypot(t.x - s.x, t.y - s.y, t.z - s.z) > 0.5,
    "torus inversion collapsed onto the sphere inversion",
  );
});

test("w tracks the MERIDIAN factor, never max()'d with the azimuthal one", () => {
  // REGRESSION. The first cut used w *= max(k, ρ′/ρ) "to be conservative".
  // ρ′/ρ diverges on the rotation axis (ρ → 0, ρ′ finite), so w exploded for
  // any orbit passing near the axis, DE = r/w collapsed, and every preset
  // rendered as a flat wall. w must be the meridian factor Radius²/d² alone.
  const rad = 1.0,
    R = 2.0;
  // A point very close to the z-axis is where the bad version blew up.
  const nearAxis = { x: 1e-7, y: 0, z: 0.5, w: 1 };
  const q = run([rad, R, 0], nearAxis);
  const rho = Math.hypot(nearAxis.x, nearAxis.y),
    u = R - rho,
    d2 = u * u + nearAxis.z * nearAxis.z;
  assert.ok(
    Math.abs(q.w - (rad * rad) / d2) < 1e-12,
    `w ${q.w} ≠ meridian factor ${(rad * rad) / d2}`,
  );
  assert.ok(q.w < 10, `w = ${q.w} blew up near the axis (the wall bug)`);
  // …and the pseudo variants track their xy factor Radius/d, not the z lane.
  const p = { x: 0.5, y: 0.4, z: 0.7, w: 1 };
  const d = Math.hypot(R - Math.hypot(p.x, p.y), p.z);
  assert.ok(Math.abs(run([rad, R, 1], p).w - rad / d) < 1e-12, "v1 w = Rad/d");
  assert.ok(Math.abs(run([rad, R, 2], p).w - rad / d) < 1e-12, "v2 w = Rad/d");
});

test("w takes a positive factor ≥ 1 near the core circle (conservative DE)", () => {
  // Points near the core circle blow up; w must grow with them so DE = r/w
  // shrinks rather than overshooting.
  const near = { x: 1.99, y: 0, z: 0.02, w: 1 };
  for (const v of [0, 1, 2, 3]) {
    const q = run([1.0, 2.0, v], near);
    assert.ok(Number.isFinite(q.w) && q.w > 0, `variant ${v} w = ${q.w}`);
  }
  assert.ok(run([1.0, 2.0, 0], near).w > 1, "expansion must raise w");
});

test("finite in ⇒ finite out on the axis, at the origin and on the core circle", () => {
  const probes = [
    { x: 0, y: 0, z: 0, w: 1 }, // origin
    { x: 0, y: 0, z: 1, w: 1 }, // rotation axis
    { x: 2, y: 0, z: 0, w: 1 }, // exactly on the core circle (d = 0)
    { x: -2, y: 0, z: 0, w: 1 },
  ];
  for (const p of probes)
    for (const v of [0, 1, 2, 3]) {
      const q = run([1.0, 2.0, v], p);
      for (const k of ["x", "y", "z", "w"])
        assert.ok(
          Number.isFinite(q[k]),
          `variant ${v} at (${p.x},${p.y},${p.z}) → ${k} = ${q[k]}`,
        );
    }
});

test("registry contract: W_MUL_K + deApprox, and it taints DE-soundness", () => {
  const def = byKey("torusInvert");
  assert.equal(def.wRule, "mul_k");
  assert.equal(def.deApprox, true);
  assert.equal(def.params.length, 3);
  const f = { ops: [{ key: "torusInvert", values: [1, 2, 0] }] };
  assert.equal(isApproxDE(f), true, "must tighten the step policy");
  assert.equal(isDeSound(f), false, "best-effort w must not vouch as sound");
});

test("out-of-range Variant falls through to the true inversion", () => {
  // The terminal `else` is the safety net for links from a newer encoder.
  assert.deepEqual(run([1, 2, 9], P), run([1, 2, 0], P));
  assert.deepEqual(run([1, 2, -4], P), run([1, 2, 0], P));
});

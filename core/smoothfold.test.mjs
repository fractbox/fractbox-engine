// Guards for smoothBoxFold/smoothBallFold (Phase C) — closed forms vs the
// hand-crafted ABoxSmoothFold source, the MinRsq≥0.99 degenerate branch, the
// ball fold's conformal w tracking, and the deApprox classification.
// Run: node --test core/smoothfold.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";
import { applyOp } from "./cpuorbit.js";
import { isApproxDE } from "./operators.js";

const run = (key, values, pt) => {
  const s = { ...pt };
  applyOp(key, values, s);
  return s;
};
const P = { x: 0.5, y: -0.4, z: 0.7, w: 1.3 };
const close = (a, b) => Math.abs(a - b) < 1e-12;

test("smoothBoxFold: per-axis C¹ blend into the positive octant, w untouched", () => {
  const [F, Sh, Fx] = [1, 6, 1];
  const f = (a) => {
    const t = Math.pow(Math.abs(a), Sh) * Fx;
    return (Math.abs(a) + (2 * F - Math.abs(a)) * t) / (t + 1);
  };
  const s = run("smoothBoxFold", [F, Sh, Fx], P);
  assert.ok(close(s.x, f(P.x)) && close(s.y, f(P.y)) && close(s.z, f(P.z)));
  assert.ok(
    s.y > 0,
    "negative components map positive (no sign restore, like the source)",
  );
  assert.equal(s.w, P.w);
});

test("smoothBallFold: blended radial multiplier, w tracks k exactly", () => {
  const [Mr, Sh, Fx] = [0.25, 4, 0.3];
  const r2 = P.x * P.x + P.y * P.y + P.z * P.z;
  const c = (1 + Mr) / 2;
  const h = (1 - Mr) / 2;
  const n = Math.abs(r2 - c) / h;
  const b = Math.pow(Math.sqrt(n), Sh) * Fx;
  const m = c - (h * (b + n)) / (1 + b);
  const k = 1 / Math.max(Math.abs(m), 1e-20);
  const s = run("smoothBallFold", [Mr, Sh, Fx], P);
  assert.ok(close(s.x, P.x * k) && close(s.y, P.y * k) && close(s.z, P.z * k));
  assert.ok(close(s.w, P.w * k), "w must track the conformal multiplier");
});

test("smoothBallFold MinRsq ≥ 0.99: m = 1, the identity (degenerate branch)", () => {
  const s = run("smoothBallFold", [0.99, 4, 0.3], P);
  for (const k of ["x", "y", "z", "w"]) assert.ok(close(s[k], P[k]), k);
});

test("both ops are deApprox-tagged", () => {
  for (const key of ["smoothBoxFold", "smoothBallFold"])
    assert.equal(isApproxDE({ ops: [{ key, values: [1, 4, 1] }] }), true, key);
});

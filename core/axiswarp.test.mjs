// Guards for the axis-warp quartet (Phase C): closed forms per axis selector,
// corpus-default anchors, and the deApprox/w contracts.
// Run: node --test core/axiswarp.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";
import { applyOp } from "./cpuorbit.js";
import { isApproxDE } from "./operators.js";

const run = (key, values, pt) => {
  const s = { ...pt };
  applyOp(key, values, s);
  return s;
};
const P = { x: 0.5, y: 0.4, z: 0.7, w: 1.3 };
const close = (a, b) => Math.abs(a - b) < 1e-12;

test("asinhWarp: a' = Base·log2(t+√(t²+1)), axis-selected, others untouched", () => {
  const t = P.y * -2;
  const s = run("asinhWarp", [1, -2, 0.37], P);
  assert.ok(close(s.y, 0.37 * Math.log2(t + Math.sqrt(t * t + 1))));
  assert.ok(s.x === P.x && s.z === P.z);
});

test("logWarp: a' = Base·log2(|Mul·a|+0.01) on the selected axis", () => {
  const s = run("logWarp", [0, 1, 0.37], P);
  assert.ok(close(s.z, 0.37 * Math.log2(Math.abs(P.z) + 0.01)));
  assert.ok(s.x === P.x && s.y === P.y);
});

test("neoSqrWarp: signed parabola branches on sign(t)", () => {
  const pos = run("neoSqrWarp", [1, 1, 1], P); // t = 0.4 ≥ 0
  assert.ok(close(pos.y, 0.4 * (1 - 0.4)));
  const neg = run("neoSqrWarp", [1, 1, 1], { ...P, y: -0.4 }); // t < 0
  assert.ok(close(neg.y, -0.4 * (-0.4 - 1)));
});

test("sinShear: pair 0 at defaults is exactly _YplusSinZ (y += sin z)", () => {
  const s = run("sinShear", [0, 1, 1], P);
  assert.ok(close(s.y, P.y + Math.sin(P.z)));
  assert.ok(s.x === P.x && s.z === P.z);
});

test("all four are deApprox and leave w untouched", () => {
  for (const key of ["asinhWarp", "logWarp", "neoSqrWarp", "sinShear"]) {
    assert.equal(isApproxDE({ ops: [{ key, values: [0, 1, 1] }] }), true, key);
    assert.equal(run(key, [0, 1, 1], P).w, P.w, key);
  }
});

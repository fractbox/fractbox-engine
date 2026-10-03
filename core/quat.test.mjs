// Guards the one shared eulerToQuat (CSG rotation convention) hoisted out of
// renderer.js / renderer_gl.js / cpu.js. The three used to carry byte-identical
// copies; this locks the convention so the WGSL/GLSL/CPU tiers stay in agreement.
import { test } from "node:test";
import assert from "node:assert/strict";
import { eulerToQuat } from "./quat.js";

const near = (a, b, eps = 1e-12) =>
  a.length === b.length && a.every((x, i) => Math.abs(x - b[i]) <= eps);

test("identity euler → identity quaternion", () => {
  assert.ok(near(eulerToQuat([0, 0, 0]), [0, 0, 0, 1]));
});

test("90° about X (Three.js XYZ intrinsic)", () => {
  const s = Math.SQRT1_2;
  assert.ok(near(eulerToQuat([90, 0, 0]), [s, 0, 0, s], 1e-9));
});

test("length-4 input is treated as a quaternion and normalized", () => {
  assert.ok(near(eulerToQuat([3, 4, 0, 0]), [0.6, 0.8, 0, 0]));
  // already-unit quat passes through unchanged
  assert.ok(near(eulerToQuat([0, 0, 0, 1]), [0, 0, 0, 1]));
});

test("missing / nullish components default to 0, output is a unit quat", () => {
  const q = eulerToQuat(undefined);
  assert.ok(near(q, [0, 0, 0, 1]));
  const q2 = eulerToQuat([45, undefined, null]);
  const len = Math.hypot(...q2);
  assert.ok(Math.abs(len - 1) < 1e-12, `expected unit length, got ${len}`);
});

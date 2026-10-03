// Guards for makinTri / makinFuzzy (Makin3D family) — pins both variant maps
// and the fuzzy damping against hand-derived closed forms, the tri-state→
// boolean Fuzzy encoding, and the W_BULB_NUMERIC contract.
//
// Named *.test.mjs so it stays out of the apps' served *.js surface. Run: node --test core/makin.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";
import { applyOp } from "./cpuorbit.js";

const P = { x: 0.5, y: 0.4, z: -0.7, w: 1 };
const run = (key, values, pt = P) => {
  const s = { ...pt };
  applyOp(key, values, s);
  return s;
};
const close = (a, b) => Math.abs(a - b) < 1e-12;

test("makinTri variant 0 = Makin3D-1 closed form", () => {
  const { x, y, z } = P;
  const s = run("makinTri", [0]);
  assert.ok(close(s.x, x * x - y * y - z * z), `x: ${s.x}`);
  assert.ok(close(s.y, 2 * x * y), `y: ${s.y}`);
  assert.ok(close(s.z, 2 * z * (x - y)), `z: ${s.z}`);
});

test("makinTri variant 1 = Makin3D-2 closed form", () => {
  const { x, y, z } = P;
  const s = run("makinTri", [1]);
  assert.ok(close(s.x, x * x + 2 * y * z), `x: ${s.x}`);
  assert.ok(close(s.y, -(y * y + 2 * z * x)), `y: ${s.y}`);
  assert.ok(close(s.z, -(z * z) + 2 * y * x), `z: ${s.z}`);
});

test("makinFuzzy closed form + signed-square flags flip only their lane's damping", () => {
  const { x, y, z } = P; // z < 0 so FuzzyY's signed square actually differs
  const L = 0.05;
  const lane = (mzOrMy, a, b) => 2 * x * a * (1 - mzOrMy / (x * x + a * a + L));
  // plain squares (flags 0)
  const plain = run("makinFuzzy", [0, 0, L]);
  assert.ok(close(plain.x, x * x - y * y - z * z));
  assert.ok(close(plain.y, lane(z * z, y)));
  assert.ok(close(plain.z, lane(y * y, z)));
  // FuzzyY=1: the y lane's damping uses z·|z| (= −z² here); z lane unchanged
  const fy = run("makinFuzzy", [1, 0, L]);
  assert.ok(close(fy.y, lane(-(z * z), y)), `signed-z damping: ${fy.y}`);
  assert.ok(close(fy.z, plain.z), "FuzzyY must not touch the z lane");
  // FuzzyZ=1 with y > 0: y·|y| = y², so nothing changes — the sign gate matters
  const fz = run("makinFuzzy", [0, 1, L]);
  assert.ok(
    close(fz.z, plain.z),
    "FuzzyZ with y>0 must equal the plain square",
  );
});

test("W_BULB_NUMERIC contract: w untouched on both ops", () => {
  assert.equal(run("makinTri", [1], { ...P, w: 1.7 }).w, 1.7);
  assert.equal(run("makinFuzzy", [1, 1, 0.01], { ...P, w: 2.3 }).w, 2.3);
});

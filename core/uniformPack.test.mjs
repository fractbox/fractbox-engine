// Run: node --test core/uniformPack.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { packOpParams } from "./uniformPack.js";
import { MAX_PARAMS } from "./limits.js";

// Real op defs (param counts asserted here so the test breaks loudly if an op's
// arity ever changes): boxFold=1, sphereFold=2, scale=1, mengerFold=0.
const box = (v) => ({ key: "boxFold", values: [v] });
const sph = (a, b) => ({ key: "sphereFold", values: [a, b] });
const scl = (v) => ({ key: "scale", values: [v] });

test("packs one list contiguously from slot 0", () => {
  const out = packOpParams([box(0.5), scl(2)]);
  assert.equal(out.length, MAX_PARAMS);
  assert.equal(out[0], 0.5); // boxFold param
  assert.equal(out[1], 2); // scale param
  assert.equal(out[2], 0); // untouched → zero-padded
});

test("multi-param ops consume consecutive slots in order", () => {
  const out = packOpParams([sph(1, 2), box(3)]);
  assert.deepEqual([out[0], out[1], out[2]], [1, 2, 3]);
});

test("two lists pack A-then-B into the SAME array (hybrid concat)", () => {
  const a = packOpParams([box(1), scl(2)]); // slots 0,1
  const b = packOpParams([box(1), scl(2)], [sph(3, 4)]); // + slots 2,3
  assert.deepEqual([b[0], b[1]], [a[0], a[1]]); // A half unchanged
  assert.deepEqual([b[2], b[3]], [3, 4]); // B continues after A
});

test("zero-param op contributes nothing", () => {
  const out = packOpParams([{ key: "mengerFold", values: [] }, box(7)]);
  assert.equal(out[0], 7); // boxFold lands at slot 0, mengerFold took none
});

test("missing values default to 0", () => {
  const out = packOpParams([sph(1)]); // values[1] absent
  assert.deepEqual([out[0], out[1]], [1, 0]);
});

test("empty input → full-width zero array", () => {
  const out = packOpParams([]);
  assert.equal(out.length, MAX_PARAMS);
  assert.ok(out.every((v) => v === 0));
});

test("unknown op key throws the engine's exact message", () => {
  assert.throws(
    () => packOpParams([{ key: "notAnOp", values: [1] }]),
    /^Error: writeOps: unknown op key notAnOp$/,
  );
});

test("returns a fresh array each call (no shared mutation)", () => {
  const a = packOpParams([box(1)]);
  const b = packOpParams([box(2)]);
  assert.equal(a[0], 1);
  assert.equal(b[0], 2); // b didn't clobber a
});

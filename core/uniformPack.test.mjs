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

// ── >3-param ops (docs/planning/OP_PARAM_ENCODING.md) ────────────────────────
// The uP[] table has always been arity-driven, so the WebGL2 tier needed no
// change for the overflow lane. Pinned because "needed no change" is a claim.

test("a >3-param op consumes all its slots on the GL tier", () => {
  const out = packOpParams([
    { key: "ruckerBulb", values: [8, 1.5, -1, 1, 3] },
    box(9),
  ]);
  assert.deepEqual([...out.slice(0, 6)], [8, 1.5, -1, 1, 3, 9]);
});

test("overrunning the pool THROWS instead of silently truncating", () => {
  // This used to be `slot < MAX_PARAMS` in the loop guard: a flat formula past
  // the pool packed a prefix and rendered a different — but plausible —
  // fractal, with no error on any tier. The scene path always threw
  // (renderer_gl.js writeScene); the flat path now agrees.
  const one = { key: "ruckerBulb", values: [8, 1, 1, 0, 0] };
  const fits = Array.from({ length: Math.floor(MAX_PARAMS / 5) }, () => one);
  assert.equal(packOpParams(fits).length, MAX_PARAMS);
  assert.throws(
    () => packOpParams([...fits, one, one]),
    new RegExp(`packOpParams: param slot \\d+ > cap ${MAX_PARAMS}`),
  );
});

test("the throw fires on the boundary op, not one op late", () => {
  const b = (n) => Array.from({ length: n }, () => box(1)); // 1 param each
  assert.equal(packOpParams(b(MAX_PARAMS)).length, MAX_PARAMS); // exactly full
  assert.throws(() => packOpParams(b(MAX_PARAMS + 1)), /> cap/);
});

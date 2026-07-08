// Zero-tooling guard for the untrusted-input sanitizer. sanitize.js is the funnel
// every share link / dropped PNG / pasted JSON passes through before it reaches
// the renderer, but until now it was only exercised from app/test/*.ts — which
// does NOT travel with the raw-ESM engine (the OSS mirror ships core/ alone).
// These run in plain Node so the hardening is guarded build-lessly.
//
// Run: node --test core/sanitize.test.mjs   (*.test.mjs → sync_web_core skips it)
import assert from "node:assert/strict";
import { test } from "node:test";
import { sanitizeFormula, sanitizeScene, sanitizeHybrid, BLANK } from "./sanitize.js";
import { MAX_FLAT_OPS } from "./limits.js";

const flat = (over = {}) => ({
  name: "T",
  ops: [{ key: "boxFold", values: [1] }, { key: "scale", values: [2] }],
  camera: { yawDeg: 35, pitchDeg: 22, dist: 14, fovDeg: 42 },
  ...over,
});

test("valid flat formula round-trips its ops and a finite camera", () => {
  const f = sanitizeFormula(flat());
  assert.deepEqual(f.ops.map((o) => o.key), ["boxFold", "scale"]);
  assert.ok(Object.values(f.camera).every((n) => Number.isFinite(n)));
});

test("unknown operator throws (fails closed, never silently drops)", () => {
  assert.throws(() => sanitizeFormula(flat({ ops: [{ key: "__nope__", values: [] }] })), /unknown operator/);
});

test("garbage camera can't produce a NaN (black-render) camera", () => {
  const f = sanitizeFormula(flat({ camera: { dist: "xxx", fovDeg: null } }));
  assert.ok(Object.values(f.camera).every((n) => Number.isFinite(n)));
  assert.equal(f.camera.dist, BLANK.camera.dist); // defaulted, not NaN
});

test("short juliaC is padded to exactly 3 (no undefined into the Float32Array)", () => {
  const f = sanitizeFormula(flat({ julia: true, juliaC: [1] }));
  assert.deepEqual(f.juliaC, [1, 0, 0]);
});

test("garbage deOption is coerced to the 0..3 range", () => {
  assert.equal(sanitizeFormula(flat({ deOption: "9" })).deOption, 3);
  assert.equal(sanitizeFormula(flat({ deOption: -5 })).deOption, 0);
  assert.equal(sanitizeFormula(flat({ deOption: "abc" })).deOption, 2); // default
});

test("name/note strip control chars (GLSL-export injection) and length-cap", () => {
  const f = sanitizeFormula(flat({ name: "line1\n} evil {", note: "x".repeat(500) }));
  assert.ok(!/[\n\r]/.test(f.name), "newline stripped from name");
  assert.equal(f.name, "line1 } evil {");
  assert.ok(f.note.length <= 120);
});

test("flat op-count is capped at MAX_FLAT_OPS (unified to the smaller tier — item 2)", () => {
  const many = Array.from({ length: 1000 }, () => ({ key: "boxFold", values: [1] }));
  const f = sanitizeFormula(flat({ ops: many }));
  assert.ok(
    f.ops.length <= MAX_FLAT_OPS,
    `expected ≤${MAX_FLAT_OPS} ops, got ${f.ops.length}`,
  );
});

test("scene: objType clamped, object count capped, multi-object forces surface color", () => {
  const objs = Array.from({ length: 20 }, () => ({ objType: 99, ops: [] }));
  const f = sanitizeScene({ ...flat({ ops: [] }), objects: objs, coloring: { mode: 3 } });
  assert.ok(f.objects.length <= 8, "MAX_OBJECTS cap");
  assert.ok(f.objects.every((o) => o.objType >= 0 && o.objType <= 6), "objType clamped");
  assert.equal(f.coloring.mode, 0, "multi-object scene forced to surface coloring");
});

test("hybrid: schedule clamps so a,b ≥ 1 and a+b ≤ 12", () => {
  const f = sanitizeHybrid(flat({ hybrid: { b: { ops: [{ key: "scale", values: [2] }] }, schedule: { a: 99, b: 99 } } }));
  assert.ok(f.hybrid.schedule.a >= 1 && f.hybrid.schedule.b >= 1);
  assert.ok(f.hybrid.schedule.a + f.hybrid.schedule.b <= 12);
});

test("BLANK is a valid, sanitizable empty slate", () => {
  const f = sanitizeFormula({ ...BLANK });
  assert.deepEqual(f.ops, []);
  assert.ok(Object.values(f.camera).every((n) => Number.isFinite(n)));
});

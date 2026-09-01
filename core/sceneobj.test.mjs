// Unit tests for the shared scene-object normalizer (sceneobj.js) — the one
// place the CSG fallback chains live now that renderer.js / renderer_gl.js /
// cpu.js all pack from it. Pins the canonical defaults so a tier can't quietly
// re-grow its own divergent chain.
import { test } from "node:test";
import assert from "node:assert/strict";
import { normalizeSceneObject } from "./sceneobj.js";
import { eulerToQuat } from "./quat.js";

test("empty object → full canonical defaults", () => {
  const n = normalizeSceneObject({});
  assert.deepEqual(n, {
    shapeId: 0,
    shapeParams: [1, 0, 0, 0],
    iterShape: false,
    ops: [],
    iters: 1,
    addC: false,
    julia: false,
    juliaC: [0, 0, 0],
    deOption: 2,
    looseDE: false,
    combine: 0,
    blendK: 0,
    origin: [0, 0, 0],
    uscale: 1,
    quat: [0, 0, 0, 1],
    color: [0.86, 0.46, 0.18],
    objType: 0,
    primParam: 1,
    primParam2: 0,
    boxBase: false,
  });
});

test("transform block wins over flat-shape fields", () => {
  const n = normalizeSceneObject({
    transform: { origin: [1, 2, 3], uscale: 2, rot: [90, 0, 0] },
    origin: [9, 9, 9],
    uscale: 5,
    rot: [0, 0, 90],
  });
  assert.deepEqual(n.origin, [1, 2, 3]);
  assert.equal(n.uscale, 2);
  assert.deepEqual(n.quat, eulerToQuat([90, 0, 0]));
});

test("flat-shape fallbacks apply when transform is absent", () => {
  const n = normalizeSceneObject({ origin: [4, 5, 6], uscale: 3, rot: [0, 45, 0] });
  assert.deepEqual(n.origin, [4, 5, 6]);
  assert.equal(n.uscale, 3);
  assert.deepEqual(n.quat, eulerToQuat([0, 45, 0]));
});

test("length-4 rot is taken as a quaternion (normalized), euler otherwise", () => {
  const q = normalizeSceneObject({ transform: { rot: [3, 4, 0, 0] } }).quat;
  assert.deepEqual(q, [0.6, 0.8, 0, 0]);
});

test("short origin/juliaC/color pad per-component (no undefined → NaN poison)", () => {
  const n = normalizeSceneObject({
    origin: [7],
    objType: 0,
    julia: true,
    juliaC: [1],
    color: [0.5],
  });
  assert.deepEqual(n.origin, [7, 0, 0]);
  assert.deepEqual(n.juliaC, [1, 0, 0]);
  assert.deepEqual(n.color, [0.5, 0.46, 0.18]);
});

test("primParam fallback chain: primParam ?? halfExtent ?? radius ?? 1", () => {
  assert.equal(normalizeSceneObject({ objType: 1, primParam: 0.6 }).primParam, 0.6);
  assert.equal(normalizeSceneObject({ objType: 1, halfExtent: 0.7 }).primParam, 0.7);
  assert.equal(normalizeSceneObject({ objType: 2, radius: 0.8 }).primParam, 0.8);
  assert.equal(
    normalizeSceneObject({ objType: 1, halfExtent: 0.7, radius: 0.8, primParam: 0.6 }).primParam,
    0.6,
  );
  assert.equal(normalizeSceneObject({ objType: 2 }).primParam, 1);
  assert.equal(normalizeSceneObject({ objType: 3, primParam2: 0.25 }).primParam2, 0.25);
  assert.equal(normalizeSceneObject({ objType: 3 }).primParam2, 0);
});

test("combine falls back to legacy combineType and masks to 2 bits", () => {
  assert.equal(normalizeSceneObject({ combineType: 2 }).combine, 2);
  assert.equal(normalizeSceneObject({ combine: 1, combineType: 3 }).combine, 1);
  assert.equal(normalizeSceneObject({ combine: 7 }).combine, 3); // & 3
});

test("legacy pure shapes canonicalize to the loop-is-identity form (D0 §2.4)", () => {
  // Old links carry deOption 0 for primitives; the unified loop must NOT read
  // that as the escape finalize — pure leaves force deOption 2 / addC false /
  // julia false / iters 1 so the (empty) loop reproduces the legacy
  // skip-the-loop primitive exactly.
  const prim = normalizeSceneObject({
    objType: 2,
    deOption: 0,
    iters: 9,
    addC: true,
    julia: true,
    juliaC: [1, 2, 3],
    boxBase: true,
    ops: [{ key: "scale", values: [2] }],
  });
  assert.equal(prim.shapeId, 2);
  assert.equal(prim.deOption, 2);
  assert.equal(prim.iters, 1);
  assert.equal(prim.addC, false);
  assert.equal(prim.julia, false);
  assert.equal(prim.boxBase, false);
  assert.deepEqual(prim.ops, []);
  assert.deepEqual(prim.juliaC, [1, 2, 3]); // seed survives (harmless, unused)
  const ifs = normalizeSceneObject({
    objType: 0,
    deOption: 0,
    julia: true,
    boxBase: true,
    ops: [{ key: "scale", values: [2] }],
  });
  assert.equal(ifs.deOption, 0);
  assert.equal(ifs.julia, true);
  assert.equal(ifs.shapeId, 1); // legacy boxBase → box-leaf finalize
  assert.equal(ifs.iterShape, false);
  assert.equal(ifs.boxBase, true); // pattern alias for the codec redundancy rule
  assert.equal(ifs.objType, 0); // chain objects present as 0
});

test("objType is masked to 4 bits and coerced from strings", () => {
  assert.equal(normalizeSceneObject({ objType: "2" }).objType, 2);
  assert.equal(normalizeSceneObject({ objType: 17 }).objType, 1); // & 0xf
});

test("new form: shapeId + shapeParams + ops = a mixed object (D0)", () => {
  const n = normalizeSceneObject({
    shapeId: 3,
    shapeParams: [1.2, 0.3, 0, 0],
    ops: [{ key: "boxFold", values: [1] }, { key: "scale", values: [2], muted: true }],
    iters: 7,
    addC: true,
    deOption: 2,
  });
  assert.equal(n.shapeId, 3);
  assert.deepEqual(n.shapeParams, [1.2, 0.3, 0, 0]);
  assert.equal(n.ops.length, 1); // muted dropped, chain KEPT alongside the leaf
  assert.equal(n.iters, 7);
  assert.equal(n.addC, true);
  assert.equal(n.objType, 0); // conservative alias: mixed ⇒ 0, never 3
  assert.equal(n.primParam, 1.2); // aliases mirror shapeParams
  assert.equal(n.primParam2, 0.3);
  assert.equal(n.boxBase, false);
});

test("iterShape rides shapeId and presents as objType 0 (D3)", () => {
  const pure = normalizeSceneObject({ shapeId: 2, shapeParams: [0.5], iterShape: true, iters: 5 });
  assert.equal(pure.iterShape, true);
  assert.equal(pure.iters, 5); // iterated pure leaf keeps its iteration count
  assert.equal(pure.objType, 0);
  const noLeaf = normalizeSceneObject({ shapeId: 0, iterShape: true });
  assert.equal(noLeaf.iterShape, false); // meaningless without a leaf
});

test("explicit shapeId wins over legacy objType; params fall back to primParam", () => {
  const n = normalizeSceneObject({ objType: 5, shapeId: 2, primParam: 0.7 });
  assert.equal(n.shapeId, 2);
  assert.deepEqual(n.shapeParams, [0.7, 0, 0, 0]);
  const legacy = normalizeSceneObject({ objType: 4, primParam: 0.4, primParam2: 0.9 });
  assert.equal(legacy.shapeId, 4);
  assert.deepEqual(legacy.shapeParams, [0.4, 0.9, 0, 0]);
  assert.equal(legacy.objType, 4); // true legacy pure shape keeps its alias
});

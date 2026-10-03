// Regression: shape/scene formulas keep their geometry in objects[], not the
// top-level op-list, so exportGLSL used to walk an empty `ops` and emit an empty
// iterateJIT_ body ("IFS/shapes produce no GLSL"). It now emits the leaf's DE for
// a single shape and an explanatory stub for a scene; op-list export is unchanged.
import { test } from "node:test";
import assert from "node:assert/strict";
import { exportGLSL } from "./shader.js";

const shapeFormula = {
  name: "Gyroid",
  deOption: 2,
  ops: [],
  objects: [
    {
      shapeId: 7, // gyroid leaf
      shapeParams: [6, 0.05, 0, 1.4],
      ops: [],
      transform: { origin: [0, 0, 0], uscale: 1, rot: [0, 0, 0] },
    },
  ],
};

const sceneFormula = {
  name: "Spiral Walls",
  deOption: 2,
  ops: [],
  objects: [
    { shapeId: 53, shapeParams: [0.25, 2, 0.5, 0.06], combine: 0 },
    { shapeId: 2, shapeParams: [2, 0, 0, 0], combine: 3 },
  ],
};

const opFormula = {
  name: "Amazing Surf 2",
  addC: true,
  deOption: 2,
  ops: [
    { key: "surfFold", values: [1] },
    { key: "sphereFold", values: [0.05, 1] },
    { key: "scale", values: [2] },
  ],
};

test("single-leaf shape exports a real DE, not an empty body", () => {
  const g = exportGLSL(shapeFormula);
  assert.match(g, /float shapeDE_Gyroid\(vec3 p\)/);
  assert.match(g, /sin\(q\.x\)/); // the actual gyroid DE math is present
  assert.match(g, /vec4 prm = vec4\(6\.0, 0\.05, 0\.0, 1\.4\)/); // params baked
  assert.doesNotMatch(g, /void iterateJIT_/); // no op-list body (the word may appear in a comment)
  // the pre-fix bug was an empty function body
  const body = g.slice(g.indexOf("{") + 1, g.lastIndexOf("}"));
  assert.ok(body.trim().length > 0, "shape DE body must not be empty");
});

test("multi-object scene exports an honest CSG stub, not empty", () => {
  const s = exportGLSL(sceneFormula);
  assert.match(s, /SCENE: Spiral Walls — 2 objects \(CSG composition\)/);
  assert.match(s, /gyroid|logspiral|leaf 53|shape #53/i); // names its objects
  assert.doesNotMatch(s, /void iterateJIT_/); // no op-list body (the word may appear in a comment)
});

test("op-list formula still exports the iterateJIT_ body (unchanged)", () => {
  const o = exportGLSL(opFormula);
  assert.match(
    o,
    /void iterateJIT_Amazing_Surf_2\(int slot, vec3 c, inout vec3 pos, inout float w\)/,
  );
  assert.match(o, /getGenericParam\(slot, 0\)/);
  assert.match(o, /\/\/ Composed from 3 primitive/);
});

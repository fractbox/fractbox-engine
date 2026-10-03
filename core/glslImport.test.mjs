// Zero-tooling guard for the .glsl / op-list JSON importer. It parses arbitrary
// pasted/loaded text into a formula, so its detection + failure modes matter.
// Round-trips against the real exporter so the emit↔import pair stays consistent.
//
// Run: node --test core/glslImport.test.mjs   (*.test.mjs → sync skips it)
import assert from "node:assert/strict";
import { test } from "node:test";
import { looksLikeGlsl, glslToFormula, importFormula } from "./glslImport.js";
import { glslFor } from "./exporter.js";
import { sanitizeFormula } from "./sanitize.js";

const formula = sanitizeFormula({
  name: "Round Trip",
  ops: [
    { key: "boxFold", values: [1] },
    { key: "sphereFold", values: [0.5, 1] },
    { key: "scale", values: [2] },
  ],
  camera: { yawDeg: 35, pitchDeg: 22, dist: 14, fovDeg: 42 },
});

test("exported .glsl round-trips its operator recipe through the importer", () => {
  const glsl = glslFor(formula);
  const back = glslToFormula(glsl);
  assert.deepEqual(back.ops.map((o) => o.key), ["boxFold", "sphereFold", "scale"]);
  assert.deepEqual(back.ops[1].values, [0.5, 1]);
});

test("looksLikeGlsl distinguishes exported .glsl from op-list JSON", () => {
  assert.equal(looksLikeGlsl(glslFor(formula)), true);
  assert.equal(looksLikeGlsl('{"ops":[]}'), false);
  assert.equal(looksLikeGlsl("[1,2,3]"), false);
});

test("importFormula dispatches JSON vs GLSL to the same formula", () => {
  const fromJson = importFormula(JSON.stringify({ ...formula }));
  const fromGlsl = importFormula(glslFor(formula));
  assert.deepEqual(fromJson.ops.map((o) => o.key), fromGlsl.ops.map((o) => o.key));
});

test("an unknown operator in the recipe throws a clear error", () => {
  const bad = glslFor(formula).replace("boxFold", "__not_an_op__");
  assert.throws(() => glslToFormula(bad), /unknown operator/);
});

test("native / decompiled .glsl (no Composed-from line) is rejected clearly", () => {
  assert.throws(() => glslToFormula("float map(vec3 p){ return length(p)-1.0; }"), /not a web-exported/);
});

test("empty / whitespace input throws instead of producing junk", () => {
  assert.throws(() => importFormula("   "), /nothing to import/);
});

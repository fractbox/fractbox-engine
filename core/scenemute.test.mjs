// Zero-tooling parity guard: a MUTED op inside a scene object must be skipped
// by ALL THREE emitters (Formula Outline PR 0.2 — the pre-existing divergence
// was CPU honoring it while WebGPU/WebGL2 uploaded o.ops raw).
// Run: node core/scenemute.test.mjs
//
// The CPU (cpu.js makeSceneDE) and WebGL2 codegen (shader_gl.js) are pure and
// tested directly, as is the shared normalizer (sceneobj.js) all three packers
// now consume. The two renderers' writeScene live inside device-bound factories
// (createRenderer needs a GPUDevice, createRendererGL a WebGL2 context), so each
// gets a source-level tripwire that it routes through normalizeSceneObject —
// crude, but it can't silently re-diverge without tripping.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { makeDE } from "./cpu.js";
import { buildSceneFragGL, sceneParamLayout, activeSceneOps } from "./shader_gl.js";
import { normalizeSceneObject } from "./sceneobj.js";

let pass = 0;
const test = (name, fn) => {
  try {
    fn();
    pass++;
  } catch (e) {
    console.error(`✗ ${name}\n  ${e.message}`);
    process.exitCode = 1;
  }
};

// A 2-object scene whose first (IFS) object carries a muted op, and its twin
// with that op REMOVED — every tier must treat the two identically.
const ident = { origin: [0, 0, 0], uscale: 1, rot: [0, 0, 0] };
const obj = (ops) => ({
  objType: 0,
  ops,
  iters: 8,
  addC: false,
  deOption: 2,
  transform: { ...ident },
  combine: 0,
  blendK: 0,
});
const OPS_MUTED = [
  { key: "sierpinskiFold", values: [] },
  { key: "scale", values: [2], muted: true },
  { key: "translate", values: [-1, -1, -1] },
];
const OPS_STRIPPED = OPS_MUTED.filter((o) => !o.muted);
const box = { objType: 1, primParam: 0.6, transform: { ...ident }, combine: 0, blendK: 0, iters: 1 };
const scene = (ops) => ({ name: "T", ops: [], iters: 8, objects: [obj(ops), box] });

test("cpu: makeSceneDE skips a muted op (DE matches the stripped twin)", () => {
  const deMuted = makeDE(scene(OPS_MUTED));
  const deStripped = makeDE(scene(OPS_STRIPPED));
  const deRaw = makeDE(scene(OPS_MUTED.map(({ key, values }) => ({ key, values }))));
  let differs = false;
  for (const q of [0.1, 0.4, 0.9, 1.3, 2.1]) {
    const p = [q, q * 0.7, -q * 0.4];
    assert.equal(deMuted(...p), deStripped(...p), `DE diverges at ${p}`);
    if (deMuted(...p) !== deRaw(...p)) differs = true;
  }
  // The muted op is not a no-op — un-muting it must actually change the field.
  assert.ok(differs, "muting the scale op changed nothing (weak fixture)");
});

test("webgl2: codegen + param layout skip a muted op (source matches the stripped twin)", () => {
  assert.equal(buildSceneFragGL(scene(OPS_MUTED).objects), buildSceneFragGL(scene(OPS_STRIPPED).objects));
  assert.notEqual(
    buildSceneFragGL(scene(OPS_MUTED).objects),
    buildSceneFragGL(scene(OPS_MUTED.map(({ key, values }) => ({ key, values }))).objects),
  );
  assert.deepEqual(
    sceneParamLayout(scene(OPS_MUTED).objects),
    sceneParamLayout(scene(OPS_STRIPPED).objects),
  );
});

test("sceneobj: normalizeSceneObject's active-op slice drops muted ops (the real unit)", () => {
  // The slice every packer consumes — must match shader_gl.js activeSceneOps
  // (the codegen side) exactly, or uP[] values shift against the program.
  assert.deepEqual(normalizeSceneObject(obj(OPS_MUTED)).ops, OPS_STRIPPED);
  assert.deepEqual(activeSceneOps(obj(OPS_MUTED)), OPS_STRIPPED);
  // Primitives carry no op slice at all.
  assert.deepEqual(normalizeSceneObject({ ...box, ops: OPS_MUTED }).ops, []);
});

test("webgl2: renderer_gl.js writeScene packs from normalizeSceneObject (source tripwire)", () => {
  const src = readFileSync(fileURLToPath(new URL("./renderer_gl.js", import.meta.url)), "utf8");
  const body = src.slice(src.indexOf("function writeScene"), src.indexOf("function writeGlobals"));
  assert.ok(
    /map\(normalizeSceneObject\)/.test(body),
    "renderer_gl.js writeScene must normalize objects via sceneobj.js " +
      "normalizeSceneObject (its .ops IS the active slice — muted ops dropped)",
  );
});

test("webgpu: renderer.js writeScene packs from normalizeSceneObject (source tripwire)", () => {
  const src = readFileSync(fileURLToPath(new URL("./renderer.js", import.meta.url)), "utf8");
  const body = src.slice(src.indexOf("function writeScene"), src.indexOf("// Cached HDR intermediate"));
  assert.ok(
    /map\(normalizeSceneObject\)/.test(body),
    "renderer.js writeScene must normalize objects via sceneobj.js " +
      "normalizeSceneObject (the shared active-op slice / fallback chains)",
  );
});

console.log(`scenemute.test.mjs: ${pass} passed`);

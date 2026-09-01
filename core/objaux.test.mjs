// #627 — the LEAF-param overflow lane (objAux), the leaf mirror of opAux
// (core/opaux.test.mjs, OP_PARAM_ENCODING.md §5). Same three pins, one level
// down: (1) the pay-per-use guarantee is structural — a thin leaf set emits a
// shader with not one objAux token; (2) the emission predicate and the
// renderer's bind predicate are the same exported function, so they cannot
// disagree (auto pipeline layouts prune declared-but-unread bindings, and
// binding one that isn't in the layout is a hard validation error); (3) the
// fat variant actually threads the lane end to end (binding → dispatch →
// per-object read at the SAME index that reads objects[]).
import { test } from "node:test";
import assert from "node:assert/strict";
import { buildWGSL, usesObjAux } from "./shader.js";
import { buildSceneFragGL, bulkLayout } from "./shader_gl.js";
import { LEAVES, leafById } from "./leaves.js";
import { normalizeSceneObject } from "./sceneobj.js";
import { sanitizeScene } from "./sanitize.js";
import { makeDE } from "./cpu.js";
import { MAX_LEAF_PARAMS, MAX_LEAF_PARAMS_INLINE } from "./limits.js";

// The registry's fat set, derived — today that's the Room rider (id 25), but
// these tests must keep holding as more leaves go fat.
const FAT_IDS = LEAVES.filter(
  (l) => l.params.length > MAX_LEAF_PARAMS_INLINE,
).map((l) => l.id);
const THIN_IDS = LEAVES.filter(
  (l) => l.params.length <= MAX_LEAF_PARAMS_INLINE,
).map((l) => l.id);

test("#627: at least one fat leaf exists (the Room rider) and none exceed the cap", () => {
  assert.ok(FAT_IDS.includes(25), "Room (25) declares a 5th param (Door)");
  for (const l of LEAVES)
    assert.ok(
      l.params.length <= MAX_LEAF_PARAMS,
      `${l.key}: ${l.params.length} ≤ ${MAX_LEAF_PARAMS}`,
    );
});

test("#627: a thin leaf set emits NO objAux token; predicates mirror", () => {
  const wgsl = buildWGSL({ leaves: THIN_IDS.slice(0, 6), ops: [] });
  assert.equal(
    (wgsl.match(/objAux/g) || []).length,
    0,
    "thin sets must not even DECLARE the lane",
  );
  assert.equal(usesObjAux(THIN_IDS), false);
  assert.equal(usesObjAux(false), false);
  assert.equal(usesObjAux([]), false);
  assert.equal(usesObjAux(FAT_IDS), true);
  assert.equal(usesObjAux(true), true, "the general all-leaves build is fat");
  // thin signatures are the exact legacy 2-arg shape
  assert.match(wgsl, /fn leafDist\(id: u32, p: vec3f, prm: vec4f\) -> f32/);
  assert.match(wgsl, /fn objIterDE\(p0: vec3f, ob: Obj\) -> f32/);
});

test("#627: the fat variant threads the lane end to end", () => {
  const wgsl = buildWGSL({ leaves: [25], ops: [] });
  assert.match(
    wgsl,
    /@group\(0\) @binding\(8\) var<storage, read> objAux : array<vec4f>;/,
  );
  assert.match(
    wgsl,
    /fn leaf_room\(p: vec3f, prm: vec4f, prm2: vec4f\) -> f32/,
  );
  assert.match(
    wgsl,
    /fn leafDist\(id: u32, p: vec3f, prm: vec4f, prm2: vec4f\) -> f32/,
  );
  assert.match(wgsl, /fn objIterDE\(p0: vec3f, ob: Obj, prm2: vec4f\) -> f32/);
  assert.match(wgsl, /fn objDist\(p0: vec3f, ob: Obj, aux: vec4f\) -> f32/);
  // index-derived addressing: the same k that indexes objects[] (all 3 walks —
  // mapDE, sceneTint, sceneOrbit — share the one call-site text)
  assert.equal((wgsl.match(/objDist\(p0, ob, objAux\[k\]\)/g) || []).length, 3);
});

test("#627 GL: uObjPrimP2 + prm2 appear only for fat scenes; Bulk layout gated", () => {
  const room = {
    shapeId: 25,
    shapeParams: [1, 0.8, 1, 0.05, 0.3],
    ops: [],
    iters: 1,
    transform: { origin: [0, 0, 0], uscale: 1, rot: [0, 0, 0] },
    combine: 0,
    blendK: 0,
  };
  const thin = { ...room, shapeId: 26, shapeParams: [0.6, 0.6, 0.6, 0.1] };
  const fatGL = buildSceneFragGL([room], {});
  const thinGL = buildSceneFragGL([thin], {});
  assert.match(fatGL, /float leaf_room\(vec3 p, vec4 prm, vec4 prm2\)/);
  assert.match(fatGL, /leaf_room\(pos, uObjPrimP\[0\], uObjPrimP2\[0\]\)/);
  assert.equal((thinGL.match(/prm2|uObjPrimP2/g) || []).length, 0);
  // The 2026-08-31 field find (the "black bar / ASCII on GL" trigger): the
  // fragment's OWN Bulk block — the declaration the leaf call sites compile
  // against — must carry uObjPrimP2 for a fat scene. The emitter passed a
  // bare {scene:true} here, so every fat-leaf scene USED the uniform without
  // DECLARING it, failed to compile on the GL tier, and the glHealth gate
  // demoted the whole app to ASCII the moment a city/heightfield thumbnail
  // baked. Uses-implies-declares, in the same source text:
  assert.match(
    fatGL,
    /uObjPrimP2\[8\];/,
    "fat scene DECLARES uObjPrimP2 in its own Bulk block",
  );
  assert.ok(
    !/uObjPrimP2\s*\[\s*8\s*\]\s*;/.test(thinGL),
    "thin scene stays byte-clean of the lane",
  );
  const names = (o) => bulkLayout(o).map((m) => m.name);
  assert.ok(!names({ scene: true }).includes("uObjPrimP2"));
  assert.ok(names({ scene: true, leafAux: true }).includes("uObjPrimP2"));
});

test("#627: normalize preserves overflow params; thin objects keep length 4", () => {
  const n = normalizeSceneObject({
    shapeId: 25,
    shapeParams: [1, 0.8, 1, 0.05, 0.4],
  });
  assert.equal(n.shapeParams.length, 5);
  assert.equal(n.shapeParams[4], 0.4);
  const t = normalizeSceneObject({ shapeId: 26, shapeParams: [1, 1, 1, 0.1] });
  assert.equal(t.shapeParams.length, 4, "≤4-param objects keep the vec4 shape");
});

test("#627: sanitize widens to the declared count, clamps the lane, keeps thin at 4", () => {
  const f = sanitizeScene({
    name: "s",
    iters: 8,
    ops: [],
    objects: [
      { shapeId: 25, shapeParams: [1, 0.8, 1, 0.05, 99999], ops: [], iters: 1 },
      { shapeId: 26, shapeParams: [1, 1, 1, 0.1], ops: [], iters: 1 },
    ],
  });
  const [room, rbox] = f.objects;
  assert.equal(room.shapeParams.length, 5, "fat leaf: declared width");
  assert.equal(room.shapeParams[4], 1e4, "lane values clamp like inline ones");
  assert.equal(rbox.shapeParams.length, 4, "thin leaf: legacy width, exactly");
  // and a missing lane slot falls back to the declared default (Door = 0)
  const g = sanitizeScene({
    name: "s",
    iters: 8,
    ops: [],
    objects: [
      { shapeId: 25, shapeParams: [1, 0.8, 1, 0.05], ops: [], iters: 1 },
    ],
  });
  assert.equal(g.objects[0].shapeParams.length, 5);
  assert.equal(g.objects[0].shapeParams[4], 0);
});

test("#627: CPU tier reads the lane — Room's Door carves the back wall", () => {
  const mk = (door) =>
    makeDE(
      {
        name: "probe",
        iters: 1,
        addC: false,
        ops: [],
        objects: [
          {
            objType: 0,
            ops: [],
            iters: 1,
            addC: false,
            shapeId: 25,
            shapeParams: [1, 0.8, 1, 0.05, door],
            transform: { origin: [0, 0, 0], uscale: 1, rot: [0, 0, 0] },
            combine: 0,
            blendK: 0,
          },
        ],
      },
      1,
    );
  // dead center of the back wall (z = -Depth + Wall), at floor level +0.1:
  const x = 0,
    y = -0.5,
    z = -1 + 0.05;
  assert.ok(mk(0)(x, y, z) < 0, "Door=0: solid wall (legacy behavior)");
  assert.ok(mk(0.35)(x, y, z) > 0, "Door=0.35: the doorway is carved out");
  // outside the door's half-width the wall is still solid
  assert.ok(mk(0.35)(0.7, y, z) < 0, "wall beyond the door stays solid");
  // and above the door top (y > Height/2) too
  assert.ok(mk(0.35)(0, 0.6, z) < 0, "lintel above the door stays solid");
});

// DEFAULT-uniform-block budget fence for the WebGL2 fragment shader.
//
// GLSL ES 3.0 guarantees only MAX_FRAGMENT_UNIFORM_VECTORS == 224 vec4 slots in
// the DEFAULT uniform block (the spec floor — real minimum-spec devices report
// exactly this). A field dump from an iOS 15.8 iPad on this very branch showed
//   gl-link-fail "FRAGMENT shader uniforms count exceeds MAX_FRAGMENT_UNIFORM_VECTORS(224)"
// → the WebGL2 tier failed to LINK and auto-fell back to ASCII. The cause: the
// op-param array uP[192] (192 vec4s), the palette stops, and the per-object CSG
// scene arrays (~88 vec4s) all lived in the default block — worst case ~311.
//
// The fix moved those arrays into a std140 uniform BUFFER (shader_gl.js
// bulkLayout / bulkBlockGL), which lives under the far larger
// MAX_UNIFORM_BLOCK_SIZE floor (≥ 16384 bytes = 1024 vec4s). This test is the
// regression fence: it emits the HEAVIEST shader configurations, counts the
// vec4-equivalents left in the DEFAULT block, and fails if any exceeds 200 —
// a 24-slot margin under the 224 floor. GLSL is never compiled in CI, so a fat
// uniform array sneaking back into the default block would otherwise reach live
// unseen (the same silent-dark class as #206).
//
// Run: node --test core/uniformbudget.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { OPERATORS, W_BULB_NUMERIC } from "./operators.js";
import { buildFragGL, buildSceneFragGL, bulkLayout } from "./shader_gl.js";
import { MAX_OBJECTS } from "./limits.js";

// vec4-equivalents an emitted shader consumes in the DEFAULT uniform block.
// Uniform BUFFER blocks (`... uniform Name { ... };`) do NOT count against
// MAX_FRAGMENT_UNIFORM_VECTORS, so they're excluded. Under GLSL ES packing:
//  • every array element pads to a full vec4 (float uP[192] ⇒ 192), the practical
//    driver behaviour that produced the 224 overflow;
//  • non-array scalars/vectors share vec4 columns — modelled by first-fit-
//    decreasing bin-packing into rows of 4 columns (float/int 1, vec2 2, vec3 3,
//    vec4 4), the tight packing the spec guarantees.
const COMPS = {
  float: 1,
  int: 1,
  uint: 1,
  bool: 1,
  vec2: 2,
  vec3: 3,
  vec4: 4,
  mat3: 3,
  mat4: 4,
};
const stripComments = (s) =>
  s.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/[^\n]*/g, "");

export function defaultBlockVec4s(rawSrc) {
  const src = stripComments(rawSrc);
  // Remove std140 uniform BUFFER blocks — they're budgeted separately.
  const noBlocks = src.replace(/uniform\s+\w+\s*\{[^}]*\}\s*;/g, "");
  let arrays = 0;
  const scalars = [];
  for (const m of noBlocks.matchAll(/\buniform\s+(\w+)\s+([^;{]+);/g)) {
    const comps = COMPS[m[1]] ?? 4;
    for (const raw of m[2].split(",")) {
      const arr = raw.trim().match(/\[(\d+)\]/);
      if (arr) arrays += parseInt(arr[1], 10) * Math.ceil(comps / 4);
      else scalars.push(comps);
    }
  }
  scalars.sort((a, b) => b - a);
  const bins = [];
  for (const c of scalars) {
    let placed = false;
    for (let i = 0; i < bins.length; i++) {
      if (bins[i] + c <= 4) {
        bins[i] += c;
        placed = true;
        break;
      }
    }
    if (!placed) bins.push(c);
  }
  return { total: arrays + bins.length, arrays, scalarRows: bins.length };
}

// std140 byte size of a bulkLayout — every member is an array, so 16-byte stride.
function bulkBytes(members) {
  return members.reduce((n, m) => n + m.count * 16, 0);
}

const BUDGET = 200; // ≤ 224 spec floor, with a 24-slot margin (target design ≤160)
const BLOCK_FLOOR = 16384; // MAX_UNIFORM_BLOCK_SIZE spec floor (bytes)

const allOps = OPERATORS.map((o) => ({ key: o.key, values: [0, 0, 0] }));
const numericOp = OPERATORS.find((o) => o.wRule === W_BULB_NUMERIC);
// All-leaves CSG scene at the object cap — the historical worst case (uObj*[8]
// per-object arrays + uP[192]).
const sceneMax = Array.from({ length: MAX_OBJECTS }, (_, k) => ({
  shapeId: (k % 4) + 1,
  combine: "union",
  ops: [{ key: "boxFold", values: [1] }],
  iters: 6,
}));

// The heaviest emissions buildFragGL / buildSceneFragGL can produce. The default
// block is invariant to op COUNT (uP[MAX_PARAMS] / uObj*[MAX_OBJECTS] are
// fixed-size), so these cover the true worst case for each shader shape.
const WORST_CASE = {
  "flat (all ops)": buildFragGL(allOps),
  "scene (all leaves, 8 objects)": buildSceneFragGL(sceneMax),
  "hybrid (2-slot dual switch)": buildFragGL(allOps.slice(0, 32), [
    { ops: allOps.slice(32) },
  ]),
  "hybrid (8-slot, max schedule arrays)": buildFragGL(allOps.slice(0, 8), [
    { ops: [allOps[8]] },
    { ops: [allOps[9]] },
    { ops: [allOps[10]] },
    { ops: [allOps[11]] },
    { ops: [allOps[12]] },
    { ops: [allOps[13]] },
    { ops: [allOps[14]] },
  ]),
  "numeric-DE variant": buildFragGL(
    numericOp ? [{ key: numericOp.key, values: [0, 0, 0] }] : [],
  ),
};

test("no worst-case shader exceeds the default-block vec4 budget", () => {
  for (const [label, src] of Object.entries(WORST_CASE)) {
    const b = defaultBlockVec4s(src);
    assert.ok(
      b.total <= BUDGET,
      `${label}: default block is ${b.total} vec4s (arrays ${b.arrays} + ` +
        `${b.scalarRows} scalar rows) — over the ${BUDGET} budget. GLSL ES ` +
        `guarantees only 224 (MAX_FRAGMENT_UNIFORM_VECTORS); move fat arrays ` +
        `into the std140 Bulk block (shader_gl.js bulkLayout).`,
    );
    // No plain-uniform ARRAY may remain in the default block — arrays are the
    // budget hogs (each element pads to a full vec4) and belong in the UBO.
    assert.equal(
      b.arrays,
      0,
      `${label}: ${b.arrays} array-vec4s still in the default block — every ` +
        `uniform ARRAY must ride the std140 Bulk block.`,
    );
  }
});

test("the std140 Bulk block fits the MAX_UNIFORM_BLOCK_SIZE floor", () => {
  const layouts = {
    flat: bulkLayout({}),
    "hybrid-8": bulkLayout({ hybrid: 8 }),
    scene: bulkLayout({ scene: true }),
  };
  for (const [label, members] of Object.entries(layouts)) {
    const bytes = bulkBytes(members);
    assert.ok(
      bytes <= BLOCK_FLOOR,
      `${label}: Bulk block is ${bytes} bytes — over the ${BLOCK_FLOOR}-byte ` +
        `MAX_UNIFORM_BLOCK_SIZE floor.`,
    );
  }
});

// Prove the fence actually bites: a fat uniform array injected into the default
// block must push the count over budget (guards against the counter silently
// mis-parsing and passing everything).
test("the budget counter catches a fat default-block array (fence self-test)", () => {
  const poisoned = buildFragGL([]) + "\nuniform vec4 uEvilBudgetHog[192];\n";
  const b = defaultBlockVec4s(poisoned);
  assert.ok(b.total > BUDGET, `fence failed to trip: got ${b.total}`);
  assert.equal(b.arrays, 192);
});

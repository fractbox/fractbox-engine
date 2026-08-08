// D0 3-emitter parity gates (PRIMITIVE_DIFS_D0 §4): the unified op-chain+leaf
// object path. CPU is checked NUMERICALLY (legacy-vs-new-form equivalence +
// an independent iterated-shape reference); the GPU emitters are checked by
// STRING emission + a bit-layout tripwire between shader.js (reads) and
// renderer.js (writes) — WebGPU render output isn't CI-able here.
// Run: node --test core/d0parity.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { makeDE } from "./cpu.js";
import { applyOp } from "./cpuorbit.js";
import { buildWGSL } from "./shader.js";
import { buildSceneFragGL } from "./shader_gl.js";

const close = (a, b, eps = 1e-12) => Math.abs(a - b) < eps;
const scene = (objects) => ({
  name: "t",
  ops: [],
  iters: 8,
  deOption: 2,
  addC: false,
  objects,
});
const TF = { origin: [0, 0, 0], uscale: 1, rot: [0, 0, 0] };
const PTS = [
  [0.3, -0.2, 0.5],
  [1.1, 0.7, -0.4],
  [-0.8, 1.3, 0.2],
  [2.0, -1.5, 0.9],
];

test("CPU: new-form pure leaves ≡ legacy primitives (unification pin)", () => {
  const cases = [
    [
      { objType: 2, primParam: 0.5 },
      { shapeId: 2, shapeParams: [0.5, 0, 0, 0] },
    ],
    [
      { objType: 3, primParam: 0.8, primParam2: 0.25 },
      { shapeId: 3, shapeParams: [0.8, 0.25, 0, 0] },
    ],
    [
      { objType: 5, primParam: 0.3, primParam2: 0.6 },
      { shapeId: 5, shapeParams: [0.3, 0.6, 0, 0] },
    ],
  ];
  for (const [legacy, form] of cases) {
    const a = makeDE(
      scene([
        { ...legacy, ops: [], iters: 1, transform: TF, combine: 0, blendK: 0 },
      ]),
    );
    const b = makeDE(
      scene([
        { ...form, ops: [], iters: 1, transform: TF, combine: 0, blendK: 0 },
      ]),
    );
    for (const [x, y, z] of PTS)
      assert.ok(close(a(x, y, z), b(x, y, z)), JSON.stringify(form));
  }
});

test("CPU: new-form box-final ≡ legacy boxBase (subsumption pin)", () => {
  const ops = [
    { key: "boxFold", values: [1] },
    { key: "scale", values: [2] },
  ];
  const legacy = makeDE(
    scene([
      {
        objType: 0,
        boxBase: true,
        primParam: 0.75,
        ops,
        iters: 6,
        addC: true,
        deOption: 2,
        transform: TF,
        combine: 0,
        blendK: 0,
      },
    ]),
  );
  const form = makeDE(
    scene([
      {
        shapeId: 1,
        shapeParams: [0.75, 0, 0, 0],
        ops,
        iters: 6,
        addC: true,
        deOption: 2,
        transform: TF,
        combine: 0,
        blendK: 0,
      },
    ]),
  );
  for (const [x, y, z] of PTS) assert.ok(close(legacy(x, y, z), form(x, y, z)));
});

test("CPU: iterated-shape (D3) matches an independent min-over-iterations reference", () => {
  const ops = [
    { key: "boxFold", values: [1] },
    { key: "scale", values: [2] },
  ];
  const de = makeDE(
    scene([
      {
        shapeId: 2,
        shapeParams: [0.5, 0, 0, 0],
        iterShape: true,
        ops,
        iters: 5,
        addC: true,
        deOption: 2,
        transform: TF,
        combine: 0,
        blendK: 0,
      },
    ]),
  );
  // Reference: replay the loop with applyOp directly (the §2.1 semantics —
  // sample after ops + addC, before the bail check; min of leaf/|w|).
  const BAIL = 1e6;
  const ref = (px, py, pz) => {
    const s = { x: px, y: py, z: pz, w: 1 };
    let dmin = 1e9;
    for (let i = 0; i < 5; i++) {
      s.i = i;
      for (const op of ops) applyOp(op.key, op.values, s);
      s.x += px;
      s.y += py;
      s.z += pz;
      const leaf = Math.hypot(s.x, s.y, s.z) - 0.5;
      dmin = Math.min(dmin, leaf / Math.max(Math.abs(s.w), 1e-9));
      if (s.x * s.x + s.y * s.y + s.z * s.z > BAIL) break;
    }
    return dmin;
  };
  for (const [x, y, z] of PTS)
    assert.ok(close(de(x, y, z), ref(x, y, z)), `${x},${y},${z}`);
});

test("WGSL: leaf registry fns + flag decode are emitted; old dispatch is gone", () => {
  const src = buildWGSL({ numericDE: false });
  for (const n of [
    "fn leaf_box",
    "fn leaf_sphere",
    "fn leaf_torus",
    "fn leaf_cylinder",
    "fn leaf_capsule",
    "fn leaf_plane",
    "fn leafDist",
    "(ob.flags >> 12u) & 0xFFu", // shapeId bits
    "(1u << 20u)", // iterShape bit
  ])
    assert.ok(src.includes(n), n);
  for (const gone of ["2048u", "objType == 1u"])
    assert.ok(!src.includes(gone), `stale: ${gone}`);
});

test("GLSL: unified codegen — pure bake, mixed finalize, iterated min, used-leaves only", () => {
  const src = buildSceneFragGL([
    { objType: 1, primParam: 0.6 },
    {
      shapeId: 3,
      shapeParams: [1.2, 0.3, 0, 0],
      ops: [{ key: "boxFold", values: [1] }],
      iters: 5,
      combine: 1,
    },
    {
      shapeId: 2,
      shapeParams: [0.5, 0, 0, 0],
      iterShape: true,
      ops: [{ key: "scale", values: [2] }],
      iters: 6,
      combine: 2,
    },
  ]);
  for (const n of [
    "float leaf_box",
    "float leaf_torus",
    "float leaf_sphere",
    "return leaf_box(pos, uObjPrimP[0]);", // pure leaf bakes to a single call
    "return leaf_torus(pos, uObjPrimP[1]) / max(abs(w), 1e-9);", // final mode
    "dmin = min(dmin, leaf_sphere(pos, uObjPrimP[2]) / max(abs(w), 1e-9));", // D3 in-loop
    "return dmin;",
    // uObjPrimP now rides the std140 Bulk block (GLES uniform-budget fix) —
    // declared as a block member, not a plain `uniform`.
    "vec4 uObjPrimP[8];",
  ])
    assert.ok(src.includes(n), n);
  assert.ok(!src.includes("leaf_cylinder"), "unused leaves are not emitted");
  assert.ok(!src.includes("uObjPrim["), "old scalar prim uniforms are gone");
});

test("bit-layout tripwire: renderer.js writes the bits shader.js reads", () => {
  const rend = readFileSync(new URL("./renderer.js", import.meta.url), "utf8");
  const shad = readFileSync(new URL("./shader.js", import.meta.url), "utf8");
  // shapeId bits 12-19 and iterShape bit 20 — both sides must name the same
  // shifts, or a silent flag skew renders every leaf as dust.
  assert.ok(rend.includes("<< 12"), "renderer must pack shapeId at bit 12");
  assert.ok(shad.includes(">> 12u"), "shader must read shapeId at bit 12");
  assert.ok(rend.includes("1 << 20"), "renderer must pack iterShape at bit 20");
  assert.ok(shad.includes("1u << 20u"), "shader must read iterShape at bit 20");
  assert.ok(
    rend.includes("o.shapeParams[3]"),
    "renderer must pack all 4 shapeParams",
  );
});

// ── First object is the BASE (field report 2026-08-01) ─────────────────────
// subtract/intersect against the empty 1e9 accumulator are degenerate: a
// first-object subtract vanished, and a following intersect then kept 1e9
// everywhere — the whole scene rendered as empty sky after a simple reorder.
// Object 0's combine is forced to union in all three tiers; for union/smooth
// first objects that is byte-identical (min(1e9, dk) == dk).
test("first-object subtract/intersect acts as the base, not empty sky", () => {
  const sphere = (combine, origin = [0, 0, 0]) => ({
    objType: 2,
    primParam: 1,
    combine,
    blendK: 0,
    transform: { origin, uscale: 1, rot: [0, 0, 0] },
  });
  // subtract-first: the sphere IS the base — de(2,0,0) ≈ 1, not 1e9.
  const deSub = makeDE({ objects: [sphere(2)] });
  assert.ok(Math.abs(deSub(2, 0, 0) - 1) < 1e-6, `got ${deSub(2, 0, 0)}`);
  // intersect-second still works on that base: two unit spheres offset 1 —
  // the lens region around (0.5,0,0) is inside (negative DE).
  const deLens = makeDE({ objects: [sphere(2), sphere(3, [1, 0, 0])] });
  assert.ok(deLens(0.5, 0, 0) < 0, `lens interior, got ${deLens(0.5, 0, 0)}`);
  // ...and far outside stays a sane positive distance, not 1e9.
  assert.ok(deLens(4, 0, 0) > 0 && deLens(4, 0, 0) < 100);
  // GL emitter bakes the same rule: object 0 emits plain union.
  const gl = buildSceneFragGL([sphere(2), sphere(3, [1, 0, 0])]);
  const body = gl.slice(gl.indexOf("float mapDE")); // the fold, not the leaf defs
  // base: union despite combine=2, then obj1's intersect — and NO subtract
  // anywhere (obj0's was neutralized; obj1 is intersect).
  assert.match(body, /d = min\(d, dk\);[\s\S]*smaxGL\(d, dk/);
  assert.doesNotMatch(body, /smaxGL\(d, -dk/);
});

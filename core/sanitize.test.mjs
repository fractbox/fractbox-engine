// Zero-tooling guard for the untrusted-input sanitizer. sanitize.js is the funnel
// every share link / dropped PNG / pasted JSON passes through before it reaches
// the renderer, but until now it was only exercised from app/test/*.ts — which
// does NOT travel with the raw-ESM engine (the OSS mirror ships core/ alone).
// These run in plain Node so the hardening is guarded build-lessly.
//
// Run: node --test core/sanitize.test.mjs   (*.test.mjs → sync_web_core skips it)
import assert from "node:assert/strict";
import { test } from "node:test";
import {
  sanitizeFormula,
  sanitizeScene,
  sanitizeHybrid,
  BLANK,
} from "./sanitize.js";
import { MAX_FLAT_OPS } from "./limits.js";
import { HYBRID_MAX_SLOTS } from "./hybridmodel.js";
import { COLOR_MODE_MAX } from "./coloring.js";
import { byKey } from "./operators.js";

const flat = (over = {}) => ({
  name: "T",
  ops: [
    { key: "boxFold", values: [1] },
    { key: "scale", values: [2] },
  ],
  camera: { yawDeg: 35, pitchDeg: 22, dist: 14, fovDeg: 42 },
  ...over,
});

test("valid flat formula round-trips its ops and a finite camera", () => {
  const f = sanitizeFormula(flat());
  assert.deepEqual(
    f.ops.map((o) => o.key),
    ["boxFold", "scale"],
  );
  assert.ok(Object.values(f.camera).every((n) => Number.isFinite(n)));
});

test("unknown operator throws (fails closed, never silently drops)", () => {
  assert.throws(
    () => sanitizeFormula(flat({ ops: [{ key: "__nope__", values: [] }] })),
    /unknown operator/,
  );
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
  const f = sanitizeFormula(
    flat({ name: "line1\n} evil {", note: "x".repeat(500) }),
  );
  assert.ok(!/[\n\r]/.test(f.name), "newline stripped from name");
  assert.equal(f.name, "line1 } evil {");
  assert.ok(f.note.length <= 120);
});

test("flat op-count is capped at MAX_FLAT_OPS (unified to the smaller tier — item 2)", () => {
  const many = Array.from({ length: 1000 }, () => ({
    key: "boxFold",
    values: [1],
  }));
  const f = sanitizeFormula(flat({ ops: many }));
  assert.ok(
    f.ops.length <= MAX_FLAT_OPS,
    `expected ≤${MAX_FLAT_OPS} ops, got ${f.ops.length}`,
  );
});

test("scene: objType clamped, object count capped, coloring mode survives", () => {
  const objs = Array.from({ length: 20 }, () => ({ objType: 99, ops: [] }));
  const f = sanitizeScene({
    ...flat({ ops: [] }),
    objects: objs,
    coloring: { mode: 1 },
  });
  assert.ok(f.objects.length <= 8, "MAX_OBJECTS cap");
  assert.ok(
    f.objects.every((o) => o.objType >= 0 && o.objType <= 6),
    "objType clamped",
  );
  // SCENES.md §Coloring (amends CSG §3.8): Glow/Bands render on scenes via
  // orbit-free signals now, so a shared scene KEEPS its chosen mode.
  assert.equal(f.coloring.mode, 1, "scene coloring mode survives sanitize");
});

test("hybrid: schedule clamps so a,b ≥ 1 and a+b ≤ 12", () => {
  const f = sanitizeHybrid(
    flat({
      hybrid: {
        b: { ops: [{ key: "scale", values: [2] }] },
        schedule: { a: 99, b: 99 },
      },
    }),
  );
  assert.ok(f.hybrid.schedule.a >= 1 && f.hybrid.schedule.b >= 1);
  assert.ok(f.hybrid.schedule.a + f.hybrid.schedule.b <= 12);
});

test("BLANK is a valid, sanitizable empty slate", () => {
  const f = sanitizeFormula({ ...BLANK });
  assert.deepEqual(f.ops, []);
  assert.ok(Object.values(f.camera).every((n) => Number.isFinite(n)));
});

// ── D0 new-form objects (PRIMITIVE_DIFS_D0 §2.4) ────────────────────────────

test("sanitizeScene: new-form shapeId keeps ops (mixed object), registry-clamps params", () => {
  const f = sanitizeScene({
    name: "d0",
    ops: [],
    objects: [
      {
        shapeId: 4, // cylinder: 2 declared params → slots 2/3 must zero
        shapeParams: [99999, -5, 7, 1],
        ops: [{ key: "boxFold", values: [1] }],
        iters: 6,
        addC: true,
      },
    ],
  });
  const o = f.objects[0];
  assert.equal(o.shapeId, 4);
  assert.equal(o.ops.length, 1, "mixed object keeps its chain");
  assert.equal(o.shapeParams[0], 1e4, "clamped to engine max");
  assert.equal(o.shapeParams[1], 1e-4, "positive-size param floored");
  assert.equal(o.shapeParams[2], 0, "undeclared slot zeroed");
  assert.equal(o.shapeParams[3], 0, "undeclared slot zeroed");
  assert.equal(
    o.objType,
    0,
    "mixed presents as objType 0 (conservative alias)",
  );
});

test("sanitizeScene: shapeId clamps to the registry range; iterShape needs a leaf", () => {
  const f = sanitizeScene({
    name: "d0",
    ops: [],
    objects: [
      { shapeId: 240, shapeParams: [1] },
      { shapeId: 0, iterShape: true, ops: [{ key: "scale", values: [2] }] },
    ],
  });
  assert.equal(
    f.objects[0].shapeId,
    240,
    "out-of-registry id PRESERVED (render degrades, re-share intact)",
  );
  assert.deepEqual(
    f.objects[0].shapeParams,
    [1, 0, 0, 0],
    "unknown-leaf params engine-clamped",
  );
  assert.equal(
    f.objects[1].iterShape,
    undefined,
    "iterShape dropped without a leaf",
  );
});

test("sanitizeScene: new-form pure leaf mirrors primParam aliases for evaluate.js", () => {
  const f = sanitizeScene({
    name: "d0",
    ops: [],
    objects: [{ shapeId: 4, shapeParams: [0.5, 0.9] }],
  });
  const o = f.objects[0];
  assert.equal(
    o.objType,
    4,
    "true legacy-range pure leaf presents its shapeId",
  );
  assert.equal(o.primParam, 0.5);
  assert.equal(o.primParam2, 0.9);
});

test("sanitizeScene: legacy objects are untouched by the new branch (pin)", () => {
  const f = sanitizeScene({
    name: "legacy",
    ops: [],
    objects: [{ objType: 2, primParam: 0.5, deOption: 3, julia: true }],
  });
  const o = f.objects[0];
  assert.equal(o.objType, 2);
  assert.equal(
    o.deOption,
    0,
    "legacy primitives still force deOption 0 (wire pin)",
  );
  assert.equal(o.shapeId, undefined, "legacy JSON gains no new fields");
});

// ── #538 hardening: slice-then-map, slot op caps, coloring, registry clamps ──
// Hostile-input set (upgraded from #537's paste-from-AI pre-flight). The rule
// these encode: a formula that is merely OUT OF RANGE is clamped and kept, a
// formula that is STRUCTURALLY wrong (unknown operator, wrong container type)
// still fails closed. Sanitize repairs values; it never invents operators.

test("#538/1 oversized hybrid.slots[] is sliced BEFORE it is mapped", () => {
  // The proof is the throw that does NOT happen: garbage in a slot past the cap
  // can only be silent if that slot was dropped before sanitizeOps ever saw it.
  const slot = () => ({ ops: [{ key: "scale", values: [2] }] });
  const f = sanitizeFormula(
    flat({
      hybrid: {
        slots: [
          slot(),
          slot(),
          slot(),
          { ops: [{ key: "__nope__", values: [] }] }, // past A+3 — never validated
          ...Array.from({ length: 5000 }, slot),
        ],
        schedule: { counts: [1, 1, 1, 1] },
      },
    }),
  );
  assert.equal(f.hybrid.slots.length, HYBRID_MAX_SLOTS - 1, "A + 3 kept");
  assert.equal(f.hybrid.schedule.counts.length, HYBRID_MAX_SLOTS);
});

test("#538/2 per-slot ops are capped exactly like slot A (legacy + N-slot)", () => {
  // absFold takes ZERO params, so a huge slot slips past the MAX_PARAMS check —
  // op COUNT was the ungoverned dimension.
  const many = Array.from({ length: 4000 }, () => ({ key: "absFold" }));
  const legacy = sanitizeFormula(flat({ hybrid: { b: { ops: many } } }));
  assert.equal(legacy.hybrid.b.ops.length, MAX_FLAT_OPS, "slot B capped");

  const n = sanitizeFormula(
    flat({
      hybrid: {
        slots: [{ ops: many }, { ops: many }],
        schedule: { counts: [1, 1, 1] },
      },
    }),
  );
  assert.ok(
    n.hybrid.slots.every((s) => s.ops.length === MAX_FLAT_OPS),
    "every N-slot capped",
  );
});

test("#538/2 an unknown op INSIDE the cap still throws (structural ≠ range)", () => {
  assert.throws(
    () =>
      sanitizeFormula(
        flat({ hybrid: { b: { ops: [{ key: "__nope__", values: [] }] } } }),
      ),
    /unknown operator/,
  );
});

test("#538/3 scene coloring is clamped, not passed through raw", () => {
  const f = sanitizeScene({
    ...flat({ ops: [] }),
    objects: [{ objType: 1 }],
    coloring: {
      mode: 99, // renders as Address on WGSL but Surface on GLSL if unclamped
      stripeFreq: -4,
      iridescence: 12,
      palettePhase: -3,
      colA: [5, -1], // short AND out of gamut
      light: { dir: [0, 0, 0], intensity: 1e9, exposure: -50, ao: NaN },
      palette: {
        on: true,
        stops: [
          ...Array.from({ length: 255 }, () => ({ c: [1, 0, 0], p: 0.5 })),
          { p: 0.5 }, // no `c` — srgbToOklab destructures this and throws
        ],
      },
    },
  });
  const c = f.coloring;
  assert.equal(c.mode, COLOR_MODE_MAX, "mode clamped to the emitter's ceiling");
  assert.equal(c.stripeFreq, 1);
  assert.equal(c.iridescence, 1);
  assert.equal(c.palettePhase, 0);
  assert.equal(c.colA.length, 3, "uniform writers index [0..2] blind");
  assert.ok(c.colA.every((x) => x >= 0 && x <= 1));
  assert.ok(
    Math.hypot(...c.light.dir) > 0.5,
    "a zero light dir would normalize() to NaN in both shaders",
  );
  assert.equal(c.light.intensity, 2.5);
  assert.equal(c.light.exposure, -1);
  assert.equal(c.light.ao, 0.55, "NaN falls back to the default, not through");
  assert.equal(c.palette.stops.length, 8, "decoder never capped the u8 count");
  assert.ok(
    c.palette.stops.every((s) => Array.isArray(s.c) && s.c.length === 3),
    "no stop reaches oklab.js without its color",
  );
});

test("#538/3 coloring sanitize is SHAPE-PRESERVING (byte stability)", () => {
  // The 11 shipped themes carry only these four keys, and encodeColoring keys
  // its defaults off ABSENCE — materializing light/palette here would rewrite
  // every themed link on re-share.
  const theme = { mode: 1, colA: [0.9, 0.5, 0.2], colB: [0.2, 0.6, 0.9], bg: [0, 0, 0] }; // prettier-ignore
  const f = sanitizeScene({
    ...flat({ ops: [] }),
    objects: [{ objType: 1 }],
    coloring: theme,
  });
  assert.deepEqual(f.coloring, theme, "in-range coloring round-trips exactly");
  assert.deepEqual(Object.keys(f.coloring).sort(), ["bg", "colA", "colB", "mode"]); // prettier-ignore

  const empty = sanitizeScene({
    ...flat({ ops: [] }),
    objects: [{ objType: 1 }],
    coloring: "not an object",
  });
  assert.equal(
    empty.coloring,
    undefined,
    "junk coloring is dropped, not faked",
  );
});

test("#538/4 op values clamp to the registry [min,max], never rejected", () => {
  const f = sanitizeFormula(
    flat({
      ops: [
        { key: "scale", values: [1e30] },
        { key: "surfFold", values: [-1e9] },
        { key: "translate", values: [1e12, "junk", -1e12] },
      ],
    }),
  );
  const p = (k) => byKey(k).params;
  assert.equal(f.ops[0].values[0], p("scale")[0].max);
  assert.equal(f.ops[1].values[0], p("surfFold")[0].min);
  assert.deepEqual(f.ops[2].values, [
    p("translate")[0].max,
    p("translate")[1].default, // non-numeric → default, as before
    p("translate")[2].min,
  ]);
  assert.ok(
    f.ops.every((o) => o.values.every(Number.isFinite)),
    "no inf/NaN survives into the op buffer",
  );
});

test("#538/4 in-range op values are untouched (the clamp is not a quantizer)", () => {
  // Values that ride BETWEEN slider steps must survive — the share codec
  // quantizes to 0.01 on its own; sanitize must not re-grid them.
  const vals = [1.23456789, -0.000123, 3.14159];
  const f = sanitizeFormula(
    flat({ ops: [{ key: "translate", values: vals }] }),
  );
  assert.deepEqual(f.ops[0].values, vals);
});

// The op-param OVERFLOW LANE — docs/planning/OP_PARAM_ENCODING.md.
//
// `struct Op` stays 16 B forever; params past the third ride a parallel
// `opAux : array<vec4f>` storage buffer indexed by the SAME `o` that indexes
// ops[]. This file pins the three things that make that safe, none of which a
// renderer test could reach (there is no GPU in CI, and WGSL is compiled
// nowhere in CI either — see core/shaderlint.test.mjs:8-11):
//
//   1. PAY-PER-USE. A shader built over only ≤3-param ops must not contain the
//      lane at all — not one read, and not even the binding declaration. This
//      is the #125 obligation: the numeric-DE probe was ALSO behind a branch
//      that "never executes" and still cost Mandelbulb +31%.
//   2. LANE ORDERING across a multi-object concatenation — the `cursor`-vs-`i`
//      hazard of §5.5, the mechanism's single likeliest bug.
//   3. The standing assumptions the addressing rests on: 17 op-read sites all
//      spelled `let op = ops[o];`, and a byte-identical `struct Op`.
//
// Named *.test.mjs so sync_web_core.sh skips it.
// Run: node --test core/opaux.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";
import { OPERATORS, byKey } from "./operators.js";
import { buildWGSL, usesOpAux } from "./shader.js";
import { packOpAuxLanes } from "./uniformPack.js";
import {
  MAX_OP_PARAMS,
  MAX_OP_PARAMS_INLINE,
  OP_AUX_F32,
  MAX_PARAMS,
  MAX_FLAT_OPS,
} from "./limits.js";

const FAT = OPERATORS.filter((o) => o.params.length > MAX_OP_PARAMS_INLINE);
const THIN = OPERATORS.filter((o) => o.params.length <= MAX_OP_PARAMS_INLINE);

// ── 1. Pay-per-use ───────────────────────────────────────────────────────────

test("a build over only <=3-param ops contains NO opAux tokens at all", () => {
  const thinIds = THIN.slice(0, 12).map((o) => o.id);
  const wgsl = buildWGSL({ ops: thinIds });
  assert.equal(
    (wgsl.match(/opAux/g) || []).length,
    0,
    "a thin op-set must not even DECLARE the lane — the binding is gated on " +
      "the same predicate as the reads, because an `auto` pipeline layout " +
      "prunes a declared-but-unread storage binding and binding one that " +
      "isn't in the layout is a hard validation error",
  );
  assert.equal(usesOpAux(thinIds), false);
});

test("the empty op-set (#265's zero-case shader) has no lane either", () => {
  assert.equal((buildWGSL({ ops: [] }).match(/opAux/g) || []).length, 0);
  assert.equal(usesOpAux([]), false);
});

test("a build including a fat op declares the lane exactly once and reads it", () => {
  assert.ok(FAT.length, "this test needs at least one >3-param op registered");
  const wgsl = buildWGSL({ ops: [FAT[0].id] });
  assert.equal(
    (wgsl.match(/@group\(0\) @binding\(7\) var<storage, read> opAux/g) || [])
      .length,
    1,
    "exactly one binding declaration",
  );
  assert.ok((wgsl.match(/opAux\[o\]/g) || []).length > 0, "and real reads");
  assert.equal(usesOpAux([FAT[0].id]), true);
});

test("usesOpAux mirrors emission for every single-op set (the bind-group contract)", () => {
  // The renderer derives its bind-group entry from usesOpAux over the SAME ops
  // list handed to buildWGSL. If the two ever disagree, every frame using that
  // variant dies with a WebGPU validation error — so pin the agreement per op.
  for (const op of OPERATORS) {
    const emitted = /var<storage, read> opAux/.test(
      buildWGSL({ ops: [op.id] }),
    );
    assert.equal(
      usesOpAux([op.id]),
      emitted,
      `usesOpAux disagrees with emission for "${op.key}"`,
    );
  }
});

test("the full switch (ops: null) declares the lane while any fat op exists", () => {
  assert.equal(usesOpAux(null), FAT.length > 0);
  assert.equal(
    /var<storage, read> opAux/.test(buildWGSL({ ops: null })),
    FAT.length > 0,
  );
});

// ── 2. The splicer ───────────────────────────────────────────────────────────

test("op.p3/p4/p5 are rewritten to opAux swizzles, and none leak through", () => {
  const wgsl = buildWGSL({ ops: null });
  assert.equal(
    /\bop\.p[3-9]\b/.test(wgsl),
    false,
    "an unspliced op.p3+ would reference a field struct Op does not have",
  );
  for (const op of FAT) {
    const body = buildWGSL({ ops: [op.id] });
    const declared = op.params.length - MAX_OP_PARAMS_INLINE;
    const want = ["x", "y", "z"].slice(0, declared);
    for (const sw of want)
      assert.ok(
        body.includes(`opAux[o].${sw}`),
        `${op.key} declares ${op.params.length} params but never reads opAux[o].${sw}`,
      );
  }
});

test("the aux read is index-derived from the SAME loop var as ops[]", () => {
  const wgsl = buildWGSL({ ops: null });
  // Every aux read must be opAux[o] — never a literal, never another variable.
  // Index-derived addressing is what lets all 17 op-read sites stay untouched.
  for (const m of wgsl.matchAll(/opAux\[([^\]]*)\]/g))
    assert.equal(m[1], "o", `aux read indexed by "${m[1]}", expected "o"`);
});

test("struct Op is byte-identical, and the 17 op-read sites are unchanged", () => {
  const wgsl = buildWGSL({ ops: null });
  assert.match(
    wgsl,
    /struct Op \{ opType: u32, p0: f32, p1: f32, p2: f32 \};/,
    "the whole design is that the struct never moves",
  );
  assert.equal(
    (wgsl.match(/let op = ops\[o\];/g) || []).length,
    17,
    "opAux[o] borrows this loop variable — if a refactor renames it, the lane " +
      "silently reads the wrong slot, so fail loudly here instead",
  );
});

test("an op referencing a param past the cap is a build-time error", () => {
  const bogus = {
    id: 9999,
    key: "__bogusOverflow",
    wgsl: `let x = op.p${MAX_OP_PARAMS};`,
    params: [],
  };
  OPERATORS.push(bogus);
  try {
    assert.throws(() => buildWGSL({ ops: [9999] }), /cap is 6 params/);
  } finally {
    OPERATORS.pop();
  }
});

// ── 3. Lane packing / ordering — the cursor-vs-i hazard ──────────────────────

const op = (key, ...values) => ({ key, values });

test("packOpAuxLanes writes one 4-float lane per op slot, .w reserved", () => {
  const out = packOpAuxLanes([op("boxFold", 1), op("scale", 2)]);
  assert.equal(out.length, 8);
  assert.deepEqual([...out], [0, 0, 0, 0, 0, 0, 0, 0]);
});

test("a thin op's lane is zero even when it follows a fat op", () => {
  // A stale lane from a previous frame's fat op in the same slot would render a
  // silently different fractal, so every slot is written, not just fat ones.
  const out = packOpAuxLanes([op("ruckerBulb", 8, 1, 1, 1, 3), op("scale", 2)]);
  assert.deepEqual([...out.slice(0, 4)], [1, 3, 0, 0]);
  assert.deepEqual([...out.slice(4, 8)], [0, 0, 0, 0]);
});

test("⚠ the fat op's lane lands at its GLOBAL slot in a multi-object scene", () => {
  // THE §5.5 REGRESSION. Three objects; the fat op is in the LAST one, at
  // global slot 3. Indexing the lane by the per-object `i` (which restarts at 0
  // for every object) would put it at slot 0 instead — no compile error, no CI
  // signal, and the bug only surfaces when a fat op finally lands in a scene.
  const objA = [op("boxFold", 1), op("scale", 2)];
  const objB = [op("absFold")];
  const objC = [op("ruckerBulb", 8, 1.5, 1, 1, 2)];
  const out = packOpAuxLanes(objA, objB, objC);

  assert.equal(out.length, 4 * 4, "4 ops across 3 objects = 4 lanes");
  assert.deepEqual(
    [...out.slice(12, 16)],
    [1, 2, 0, 0],
    "RadialSel=1, Convention=2 must land at global slot 3",
  );
  for (let s = 0; s < 3; s++)
    assert.deepEqual(
      [...out.slice(s * 4, s * 4 + 4)],
      [0, 0, 0, 0],
      `slot ${s} must be untouched — an i-indexed write would alias here`,
    );
});

test("lane order matches op order across the concatenation, fat op first", () => {
  const out = packOpAuxLanes(
    [op("ruckerBulb", 8, 1, 1, 1, 1)],
    [op("ruckerBulb", 8, 1, 1, 0, 3)],
  );
  assert.deepEqual([...out.slice(0, 3)], [1, 1, 0]);
  assert.deepEqual([...out.slice(4, 7)], [0, 3, 0]);
});

test("an empty op-list still yields a non-empty buffer (WebGPU rejects 0-byte writes)", () => {
  assert.equal(packOpAuxLanes([]).length, 4);
});

test("a fat op inside a NON-FIRST hybrid slot lands at its global slot", () => {
  // bulbAxis is the first fat op to appear inside a SHIPPED hybrid preset
  // (Triune Bulb, core/oplist.js), a path ruckerBulb never exercised.
  // writeHybrid routes a PRE-CONCATENATED array through writeOps, so the
  // cursor-vs-i hazard of §5.5 cannot bite here — this pins that it stays so,
  // because a future slot-wise packer would reintroduce it silently.
  const slot0 = [op("boxFold", 1), op("scale", 2)];
  const slot1 = [op("bulbAxis", 5, 1, 0, 0.75, -2)];
  const slot2 = [op("bulbAxis", 8, 0, 2, 1, 1)];
  const out = packOpAuxLanes(slot0, slot1, slot2);

  assert.equal(out.length, 4 * 4, "4 ops across 3 slots = 4 lanes");
  assert.deepEqual(
    [...out.slice(8, 12)],
    [0.75, -2, 0, 0],
    "ThetaMul/PhiMul must land at global slot 2",
  );
  assert.deepEqual(
    [...out.slice(12, 16)],
    [1, 1, 0, 0],
    "the tied-power default is written, not left stale",
  );
  for (let s = 0; s < 2; s++)
    assert.deepEqual(
      [...out.slice(s * 4, s * 4 + 4)],
      [0, 0, 0, 0],
      `slot ${s} must be untouched`,
    );
});

test("two fat ops coexist: each reads its own lane, neither aliases the other", () => {
  // The registry now has more than one op above the inline cap, so the lane is
  // shared. bulbAxis (5) and ruckerBulb (5) adjacent is the cheapest proof that
  // the swizzle assignment is per-op-position, not per-op-kind.
  const out = packOpAuxLanes([
    op("bulbAxis", 8, 2, 1, -1.5, 0.25),
    op("ruckerBulb", 8, 1, 1, 1, 3),
  ]);
  assert.deepEqual([...out.slice(0, 4)], [-1.5, 0.25, 0, 0]);
  assert.deepEqual([...out.slice(4, 8)], [1, 3, 0, 0]);
});

// ── 4. Registry / budget contract ────────────────────────────────────────────

test("no op declares more params than one inline block + one aux lane", () => {
  for (const o of OPERATORS)
    assert.ok(
      o.params.length <= MAX_OP_PARAMS,
      `${o.key} declares ${o.params.length} > ${MAX_OP_PARAMS}`,
    );
  assert.equal(MAX_OP_PARAMS, MAX_OP_PARAMS_INLINE + OP_AUX_F32);
  // The pool must fit a full flat formula of the widest op on BOTH tiers.
  assert.equal(MAX_PARAMS, MAX_FLAT_OPS * MAX_OP_PARAMS);
});

test("validateOperators warns (never fails) on ops that spend the lane", () => {
  // Arity creep stays visible in CI and reviewable, per §5.6 — it must not be a
  // hard failure, or the lane could never be used.
  return import("./invariants.js").then(({ validateOperators }) => {
    const { failures, warnings } = validateOperators();
    assert.deepEqual(failures, []);
    for (const o of FAT)
      assert.ok(
        warnings.some((w) => w.includes(o.key) && w.includes("opAux")),
        `expected an arity-creep warning naming "${o.key}"`,
      );
    assert.equal(
      warnings.filter((w) => w.includes("opAux")).length,
      FAT.length,
      "one warning per fat op, and none for a thin one",
    );
  });
});

test("the GL tier needs no lane — uP[] has always been arity-driven", async () => {
  const { iterBodyGL } = await import("./shader_gl.js");
  const fat = byKey("ruckerBulb");
  const { body, paramCount } = iterBodyGL([op("ruckerBulb", 8, 1, 1, 1, 2)]);
  assert.equal(paramCount, fat.params.length, "all 5 params get a uP[] slot");
  for (let i = 0; i < fat.params.length; i++)
    assert.ok(body.includes(`uP[${i}]`), `param ${i} must reach the shader`);
  assert.equal(/opAux/.test(body), false, "the lane is a WebGPU-tier concept");
});

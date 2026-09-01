// #265 — march-variant specialization: the op-set descriptor must distinguish
// "this formula dispatches NO ops" ([] → emit zero switch cases) from "the
// caller isn't specializing" (null → emit the full ~58-case switch).
//
// The bug this pins: renderer.js's wgslOf/keyFor used `f.ops && f.ops.length`,
// which collapses [] into the null sentinel — so a pure-leaf scene (23 of the
// 90 shipped presets carry zero ops) compiled the WORST-case full-op shader
// (116 KB / ~2.4 s on an M-series Mac) instead of the smallest possible one
// (49 KB / ~0.33 s). See docs/rca/rca-issue-265-op-specialized-march.md.
//
// These helpers are pure and module-scope precisely so this gate can run with
// NO GPU — createRenderer is never called here.
import test from "node:test";
import assert from "node:assert/strict";
import { keyFor, wgslOf } from "./renderer.js";
import { buildWGSL } from "./shader.js";
import { OPERATORS } from "./operators.js";
import { PRESETS } from "./oplist.js";
import { frameFeaturesFor, formulaOpSet } from "./capturesettle.js";
import { normalizeSceneObject } from "./sceneobj.js";
import { activeOps, byKey } from "./operators.js";

const feat = (over = {}) => ({
  numericDE: false,
  leaves: null,
  coloring: false,
  scene: false,
  hybrid: false,
  morph: false,
  df64: false,
  ops: [],
  ...over,
});
// Count emitted `case <id>u: {` clauses in the WGSL op switch.
const caseIds = (wgsl) => {
  const s = new Set();
  for (const m of wgsl.matchAll(/case (\d+)u: \{/g)) s.add(Number(m[1]));
  return s;
};

test("#265 wgslOf: an empty op array specializes to zero ops, not to null", () => {
  assert.deepEqual(wgslOf(feat({ ops: [] })).ops, []);
  assert.notEqual(
    wgslOf(feat({ ops: [] })).ops,
    null,
    "[] must NOT be coerced to the full-switch sentinel — that was the bug",
  );
});

test("#265 wgslOf: a non-empty op array is passed through unchanged", () => {
  const ops = [1, 5, 9];
  assert.deepEqual(wgslOf(feat({ ops })).ops, ops);
});

test("#265 wgslOf: null/undefined ops still mean 'emit the full switch'", () => {
  assert.equal(wgslOf(feat({ ops: null })).ops, null);
  assert.equal(wgslOf(feat({ ops: undefined })).ops, null);
});

test("#265 keyFor: empty-op and unspecialized keys never collide", () => {
  const empty = keyFor(feat({ ops: [] }));
  const full = keyFor(feat({ ops: null }));
  assert.notEqual(
    empty,
    full,
    "[] and null emit DIFFERENT shaders, so they must not share a cache entry",
  );
  assert.match(empty, /^0:-:-$/, "empty op-set keeps the legacy '-' marker");
  assert.match(full, /^0:\*:-$/, "unspecialized ops get their own '*' marker");
});

test("#265 keyFor: existing specialized key strings are unchanged", () => {
  // The scene form quoted throughout the compile-freeze diag notes.
  assert.equal(
    keyFor(feat({ scene: true, leaves: [1, 2], ops: [] })),
    "10:-:1.2",
  );
  assert.equal(keyFor(feat({ ops: [2, 7] })), "0:2.7:-");
});

test("#265 buildWGSL: ops:[] emits ZERO op cases but keeps the default clause", () => {
  const wgsl = buildWGSL({ ...wgslOf(feat({ ops: [] })), leaves: false });
  assert.equal(caseIds(wgsl).size, 0, "no operator case should be emitted");
  assert.match(wgsl, /default: \{\}/, "the switch must still have a default");
});

test("#265 buildWGSL: ops:null emits every operator case", () => {
  const wgsl = buildWGSL({ ...wgslOf(feat({ ops: null })), leaves: false });
  assert.equal(caseIds(wgsl).size, OPERATORS.length);
});

test("#265 buildWGSL: ops:[ids] emits exactly those cases", () => {
  const ids = OPERATORS.slice(0, 3).map((o) => o.id);
  const wgsl = buildWGSL({ ...wgslOf(feat({ ops: ids })), leaves: false });
  assert.deepEqual(
    [...caseIds(wgsl)].sort((a, b) => a - b),
    [...ids].sort((a, b) => a - b),
  );
});

test("#265 buildWGSL: an empty op-set is strictly smaller than the full switch", () => {
  // Measured on the SCENE feature set — the population this bug actually hit
  // (pure-leaf scenes). Scenes inline the op switch at the most sites, so the
  // saving is largest there: ~116 KB -> ~49 KB in the shipped roster.
  const sceneFeat = feat({ scene: true, leaves: [1] });
  const small = buildWGSL(wgslOf({ ...sceneFeat, ops: [] })).length;
  const big = buildWGSL(wgslOf({ ...sceneFeat, ops: null })).length;
  assert.ok(
    big > small * 2,
    `full switch (${big}B) should be >2x the zero-op shader (${small}B)`,
  );
  // And the flat set still shrinks meaningfully (fewer inline sites).
  const fSmall = buildWGSL(wgslOf(feat({ ops: [] }))).length;
  const fBig = buildWGSL(wgslOf(feat({ ops: null }))).length;
  assert.ok(fBig > fSmall * 1.4, `flat: ${fBig}B vs ${fSmall}B`);
});

// The end-to-end regression: every zero-op preset must now emit the SMALL
// shader. Before the fix these 23 scenes emitted the full-switch 116 KB one.
test("#265 zero-op presets emit a specialized (not full-switch) shader", () => {
  const zero = PRESETS.filter((p) => {
    const ff = frameFeaturesFor(p, { mode: 0 });
    return Array.isArray(ff.ops) && ff.ops.length === 0;
  });
  assert.ok(zero.length > 0, "roster should still contain zero-op presets");
  for (const p of zero) {
    const ff = frameFeaturesFor(p, { mode: 0 });
    const wgsl = buildWGSL(wgslOf(ff));
    assert.equal(
      caseIds(wgsl).size,
      0,
      `${p.name}: a zero-op formula must emit zero op cases`,
    );
  }
});

test("#265 every preset emits exactly the op cases its formula uses", () => {
  for (const p of PRESETS) {
    const ff = frameFeaturesFor(p, { mode: 0 });
    const got = [...caseIds(buildWGSL(wgslOf(ff)))].sort((a, b) => a - b);
    assert.deepEqual(
      got,
      ff.ops,
      `${p.name}: emitted cases must equal its op-set`,
    );
  }
});

// SAFETY INVARIANT behind dropping cases: the predicted op-set must never
// UNDER-count what the write path actually uploads, or a dropped case could be
// dispatched and a fold would silently become a no-op (wrong image, no crash).
// Scenes: normalizeSceneObject only ever REMOVES ops (muted / legacyPure), it
// never synthesizes one. Flat: activeOps applies the same muted filter.
test("#265 formulaOpSet never under-counts the ops the write path uploads", () => {
  for (const p of PRESETS) {
    const predicted = new Set(formulaOpSet(p));
    const actual = new Set();
    if (Array.isArray(p.objects) && p.objects.length) {
      for (const o of p.objects.map(normalizeSceneObject))
        for (const op of o.ops) actual.add(byKey(op.key).id);
    } else if (p.hybrid) {
      for (const op of activeOps(p)) actual.add(byKey(op.key).id);
      for (const op of (p.hybrid.b?.ops || []).filter((o) => !o.muted))
        actual.add(byKey(op.key).id);
    } else {
      for (const op of activeOps(p)) actual.add(byKey(op.key).id);
    }
    for (const id of actual)
      assert.ok(
        predicted.has(id),
        `${p.name}: op ${id} is uploaded but was NOT predicted — its case would be dropped`,
      );
  }
});

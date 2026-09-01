// Canonical preset/operator gate — node --test core/presets.test.mjs
//
// The same invariants formula-creator/check.mjs enforces (through its synced
// core copy), asserted against the CANONICAL top-level core/ — so the gate
// survives the legacy app's retirement (LAUNCH_PLAN F4) and runs wherever the
// core suite runs (CI's `node --test core/*.test.mjs` step). No GPU: validates
// everything checkable statically — operator opcodes contiguous with <= 3
// params each, every preset resolves + exports engine-conformant GLSL.
// Warnings stay warnings (logged, not fatal), matching check.mjs semantics.

import test from "node:test";
import assert from "node:assert/strict";
import { OPERATORS, byKey } from "./operators.js";
import { PRESETS } from "./oplist.js";
import { validateOperators, validateFormula } from "./invariants.js";

test("operator registry passes validateOperators", () => {
  const { failures, warnings } = validateOperators();
  for (const w of warnings) console.warn("WARN:", w);
  assert.deepEqual(failures, []);
  assert.ok(OPERATORS.length > 0, "registry is non-empty");
});

test("every preset resolves and exports clean GLSL", () => {
  assert.ok(PRESETS.length > 0, "preset catalog is non-empty");
  for (const p of PRESETS) {
    const { failures, warnings } = validateFormula(p);
    for (const w of warnings) console.warn(`WARN (${p.name}):`, w);
    assert.deepEqual(failures, [], `preset "${p.name}"`);
  }
});

test("#538: every preset op value lies inside its own registry [min,max]", () => {
  // Since #538 sanitize CLAMPS op values to the registry range, so a preset
  // authored outside its own sliders no longer round-trips: it renders one way
  // when loaded flat (main.ts skips sanitize for flat formulas) and another way
  // once shared, imported, or saved to ★ Mine. Two presets were already in that
  // state when the clamp landed — Surf Coral's surfFold(5) against a max of 3,
  // and Cantor Rotations' translate(-5.77) against a min of -2 — and both were
  // resolved by WIDENING the operator's range to match the shipped art, never
  // by editing the art. This gate keeps the two in agreement from here on: if it
  // fails, decide which is wrong (usually the range) rather than clamping.
  const viol = [];
  const walk = (ops, where) => {
    for (const o of ops || []) {
      const def = byKey(o.key);
      if (!def) continue; // unknown keys are validateFormula's job, not ours
      def.params.forEach((p, i) => {
        const v = o.values?.[i];
        if (typeof v !== "number") return;
        if (v < p.min || v > p.max)
          viol.push(`${where} ${o.key}.${p.name}=${v} ∉ [${p.min}, ${p.max}]`);
      });
    }
  };
  for (const p of PRESETS) {
    walk(p.ops, `"${p.name}"`);
    for (const o of p.objects || []) walk(o.ops, `"${p.name}" object`);
    walk(p.hybrid?.b?.ops, `"${p.name}" slot B`);
    for (const s of p.hybrid?.slots || []) walk(s.ops, `"${p.name}" slot`);
  }
  assert.deepEqual(viol, []);
});

test("#116: Surf Mushroom is a BOUNDED fold shape, not an infinite surf sheet", () => {
  // The original recipe used `surfFold` (box fold on X,Y only, Z free), whose
  // attractor is an infinite horizontal SHEET — at the default camera it read
  // as a flat splat, and orbiting flew the camera through the endless slab,
  // which the reporter saw as a "clipping plane cutting through the shape".
  // The fix ships the compact negative-scale Amazing Box (boxFold on ALL axes)
  // instead. Guard: this preset must NOT reintroduce the free-Z surf/cylinder
  // fold, and must fold with the all-axis boxFold that keeps the object finite.
  const p = PRESETS.find((x) => x.name === "Surf Mushroom");
  assert.ok(p, "Surf Mushroom preset exists");
  const keys = p.ops.map((o) => o.key);
  assert.ok(
    keys.includes("boxFold"),
    "must use all-axis boxFold (bounded solid), not the free-Z surfFold",
  );
  assert.ok(
    !keys.includes("surfFold") && !keys.includes("cylinderFold"),
    "must NOT use surfFold/cylinderFold — those leave Z free and make an infinite sheet",
  );
  // The Scale Drift op is what organic-izes the lattice (the #116 ask); keep it.
  assert.ok(
    keys.includes("scaleDrift"),
    "keeps Scale Drift for the organic mass gradient",
  );
  // Camera must frame the whole (finite) object, not sit far away on a splat
  // (old default was dist 24 → a tiny flat blob).
  assert.ok(
    p.camera.dist <= 20,
    `camera should frame the object (dist ${p.camera.dist} <= 20)`,
  );
});

test("#116: Surf Coral uses Julia mode to bound its surfFold chain", () => {
  // Surf Coral DOES use surfFold (twice, with a rotate between), which alone
  // would be the free-Z infinite-sheet attractor the Surf Mushroom test above
  // guards against. What makes THIS recipe bounded instead is the fixed Julia
  // seed (`julia: true` + `juliaC`) replacing the per-point orbit constant —
  // verified by rendering with the camera pulled back to dist 45-60, where the
  // whole lumpy coral/mushroom-cluster blob sits in frame with no sheet
  // clipping. Guard: keep julia mode on (dropping it would silently regress
  // this preset back into an infinite sheet) and keep the camera framed wide
  // enough to show the whole bounded object, not a close-up crop.
  const p = PRESETS.find((x) => x.name === "Surf Coral");
  assert.ok(p, "Surf Coral preset exists");
  assert.ok(
    p.julia,
    "must keep Julia mode on — it's what bounds the surfFold chain",
  );
  assert.ok(
    Array.isArray(p.juliaC) && p.juliaC.length === 3,
    "carries a fixed juliaC seed",
  );
  const keys = p.ops.map((o) => o.key);
  assert.equal(
    keys.filter((k) => k === "surfFold").length,
    2,
    "keeps both surf folds",
  );
  assert.ok(
    p.camera.dist >= 30,
    `camera should pull back far enough to frame the whole bounded blob (dist ${p.camera.dist} >= 30)`,
  );
});

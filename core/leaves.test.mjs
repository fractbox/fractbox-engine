// Registry gate for the shape-leaf table (leaves.js) — the D0 counterpart of
// operators.test.mjs' validateOperators gate. Contiguous append-only ids,
// 1-4 params each, every UI step on the TAG.SHAPES ×1000 fixed-point grid.
// Run: node --test core/leaves.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  LEAVES,
  leafById,
  leafByKey,
  MAX_LEAF_ID,
  validateLeaves,
} from "./leaves.js";

test("registry self-checks pass (contiguous ids, param budgets, step grid)", () => {
  assert.deepEqual(validateLeaves(), []);
});

test("the 6 launch leaves hold their shipped ids (codec stability)", () => {
  const want = {
    box: 1,
    sphere: 2,
    torus: 3,
    cylinder: 4,
    capsule: 5,
    plane: 6,
  };
  for (const [key, id] of Object.entries(want))
    assert.equal(leafByKey(key)?.id, id, key);
  assert.equal(MAX_LEAF_ID, LEAVES.length);
});

test("lookups are total and null-safe", () => {
  assert.equal(leafById(3)?.key, "torus");
  assert.equal(leafById("3")?.key, "torus");
  assert.equal(leafById(99), null);
  assert.equal(leafByKey("nope"), null);
});

test("exact/approx split is deliberate per leaf (deApprox is an opt-in tag)", () => {
  // Exact: the launch SDFs + per-cell lattice constructions + IQ menger.
  // Approx: TPMS/implicit |f|/|∇f| bounds, polar unrolls, voxel quantization.
  const approx = new Set([
    "gyroid",
    "schwarzP",
    "lidinoid",
    "scherk",
    "gear",
    "knotPQ",
    "loresVoxel",
    "helix",
    "helixStairs",
    "sphereCage",
    "sliceCage",
    "waveSurface",
    "kleinBagel",
    "seashell",
    "dini",
    "heartSurf",
    "citrus",
    "piriform",
    "kissSurf",
    "dingDong",
    "devilSurf",
    "trifoliumSurf",
    "decoCube",
    "cayleyCubic",
    "gumdropTorus",
    "quadricSurf",
    // batch 3 heightfields: iterated/Lipschitz-foreshortened gaps, not exact.
    "gnarlyField",
    "ducksField",
    "mandelPlate",
    "checkerField",
    "riemannSqrt",
    // batch 4: fluting steepens near the axis / conformal-chart or sampled
    // distances — foreshortened, not exact.
    "tower",
    "loxodrome",
    "logSpiral",
    "pseudoSphere",
    // batch 5: Taubin-quotient surfaces (randomCells stays exact — integer
    // hash + 3×3×3 scan + half-cell clamp is a true bound).
    "umbrella",
    "kleinBottle",
    // kleinian-limit: first-order conformal-pullback estimate.
    "kleinianLimit",
    // #634 city stays EXACT — per-cell exact box tiers + 3×3 scan + cell-pitch
    // cap is a true bound (footprints never leave their cell, so nothing
    // outside the window beats the cap; brute-force proof in city.test.mjs).
  ]);
  for (const l of LEAVES)
    assert.equal(!!l.deApprox, approx.has(l.key), `${l.key} deApprox mismatch`);
});

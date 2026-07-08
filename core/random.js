// "Surprise me" — guaranteed non-blank random formula. Canonical source shared
// by every frontend (card creator + Blockly). Two generation paths, weighted:
//
//   - RECIPES (the original path): each family is a starter-preset recipe
//     varied only by DE-safe knobs (rigid rotations, bounded box params, tail
//     kaleido/translate, or view-only for the IFS sponges). Hand-proven, no
//     verification needed — but only 4 skeletons over ~13 of the operators.
//
//   - PRESET-SEEDED (vary.js): clone a random plain starter, wide-jitter every
//     param inside its declared operator range, and let the numeric oracle
//     (isSound → evaluate.measure) reject the rare degenerate roll. This walks
//     the WHOLE shipped op gamut — every family a preset covers becomes a
//     Surprise family, including the escape-time maps and any future preset —
//     while the proven recipes remain the guaranteed fallback.

import { PRESETS } from "./oplist.js";
import { jitterParams, soundCandidate } from "./vary.js";

// Starters the seeded path may draw from: plain op-stacks only. Scenes vary by
// object transforms (not op params), and hybrids would drag slot-B round-trip
// support into every randomFormula consumer (e.g. the Blockly checker) — both
// stay recipe/preset territory for now.
const seedablePresets = () => PRESETS.filter((p) => !p.objects?.length && !p.hybrid);

// One wide-jittered take on a random starter, oracle-verified; falls back to
// the classic recipes if five rolls in a row read as blank (rare).
function seededFormula() {
  const pool = seedablePresets();
  const seed = pool[Math.floor(Math.random() * pool.length)];
  const candidate = (attempt) => {
    // Narrow the throw a little on each retry — wild first, safer later.
    const f = jitterParams(seed, { spread: 0.45 - attempt * 0.06 });
    f.name = "Random";
    f.note = `randomized ${seed.name} — tweak it, Save it, or Reset`;
    if (typeof f.iters === "number")
      f.iters = Math.max(3, Math.min(24, f.iters + Math.floor(Math.random() * 5) - 2));
    if (f.camera) {
      // The seed's tuned distance stays (it frames this family right); the
      // angle is where the variety is.
      f.camera = { ...f.camera };
      f.camera.yawDeg = +(Math.random() * 360).toFixed(2);
      f.camera.pitchDeg = +(-25 + Math.random() * 60).toFixed(2);
    }
    return f;
  };
  return soundCandidate(candidate, recipeFormula);
}

export function randomFormula() {
  // Half the draws walk the full preset gamut; half stay on the proven recipes
  // (they carry their own charm — pure rolls no preset param-space contains).
  return Math.random() < 0.5 ? seededFormula() : recipeFormula();
}

function recipeFormula() {
  const rnd = (a, b) => a + Math.random() * (b - a);
  const r2 = (a, b) => +rnd(a, b).toFixed(2);
  const pick = (arr) => arr[Math.floor(Math.random() * arr.length)];
  const chance = (p) => Math.random() < p;
  const rots = ['rotateXY', 'rotateYZ', 'rotateXZ'];
  const family = pick(['box', 'box', 'box', 'bulb', 'bulb', 'menger', 'sierpinski']);

  // Escape-time Mandelbulb: a single spherical power is always a solid bulb.
  // Power 2–8 stays clear of the fp32 bailout cliff; a leading rotation spins it.
  if (family === 'bulb') {
    const ops = [];
    if (chance(0.5)) ops.push({ key: pick(rots), values: [r2(-90, 90)] });
    ops.push({ key: 'mandelbulbPower', values: [r2(2, 8)] });
    return {
      name: 'Random', note: 'randomized bulb — tweak it, Save it, or Reset',
      addC: true, iters: Math.floor(rnd(7, 11)), deOption: 0, ops,
      camera: { yawDeg: r2(0, 360), pitchDeg: r2(-20, 25), dist: 5.0, fovDeg: 42 },
    };
  }

  // Structural IFS sponges: keep the exact proven recipe (params interlock),
  // vary only camera + depth.
  if (family === 'menger' || family === 'sierpinski') {
    const r = family === 'menger'
      ? { ops: [
            { key: 'absFold', values: [] }, { key: 'mengerFold', values: [] },
            { key: 'scale', values: [3.0] }, { key: 'translate', values: [-2.0, -2.0, 0.0] },
            { key: 'zFold', values: [1.0, 2.0] } ],
          iters: Math.floor(rnd(4, 7)), dist: 9.0, note: 'randomized sponge' }
      : { ops: [
            { key: 'sierpinskiFold', values: [] }, { key: 'scale', values: [2.0] },
            { key: 'translate', values: [-1.0, -1.0, -1.0] } ],
          iters: Math.floor(rnd(11, 16)), dist: 8.0, note: 'randomized tetra' };
    return {
      name: 'Random', note: `${r.note} — tweak it, Save it, or Reset`,
      addC: false, iters: r.iters, deOption: 2, ops: r.ops,
      camera: { yawDeg: r2(0, 360), pitchDeg: r2(-30, 40), dist: r.dist, fovDeg: 42 },
    };
  }

  // Mandelbox (default, weighted): bounded box core + DE-safe extras.
  const minR = r2(0.35, 0.6);
  const ops = [
    { key: 'boxFold',    values: [r2(0.9, 1.3)] },
    { key: 'sphereFold', values: [minR, r2(minR + 0.4, 1.4)] },
  ];
  const nRot = Math.floor(rnd(0, 3));
  for (let k = 0; k < nRot; k++) ops.push({ key: pick(rots), values: [r2(-90, 90)] });
  // Positive scales only: negative-scale boxes escape past the bailout at the
  // fixed dist=24 / iter range → blank sky (~10%). Set one by hand if wanted.
  ops.push({ key: 'scale', values: [pick([2, 2, 2.2, 2.5, 3])] });
  // At most one tail flourish (proven safe atop the bounded box).
  if (chance(0.5)) {
    if (chance(0.5)) ops.push({ key: 'kaleido',   values: [pick([3, 4, 5, 6, 8]), r2(0, 30)] });
    else             ops.push({ key: 'translate', values: [r2(-0.4, 0.4), r2(-0.4, 0.4), r2(-0.4, 0.4)] });
  }
  return {
    name: 'Random', note: 'randomized box — tweak it, Save it, or Reset',
    addC: true, iters: Math.floor(rnd(9, 14)), deOption: 2, ops,
    camera: { yawDeg: r2(0, 360), pitchDeg: r2(-25, 35), dist: 24.0, fovDeg: 42 },
  };
}

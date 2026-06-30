// "Surprise me" — guaranteed non-blank random formula. Canonical source shared
// by every frontend (card creator + Blockly). Each family is a starter-preset
// recipe varied only by DE-safe knobs (rigid rotations, bounded box params,
// tail kaleido/translate, or view-only for the IFS sponges). See the card app's
// project notes for the full rationale.

export function randomFormula() {
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

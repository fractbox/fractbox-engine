// Zero-tooling test for the Surprise generator — both the classic recipes and
// the preset-seeded wide-jitter path must emit sanitizable, visibly-rendering
// formulas every single time (this is the button a first-time visitor mashes).
// Run: node core/random.test.mjs
import assert from 'node:assert/strict';
import { randomFormula } from './random.js';
import { sanitizeFormula } from './sanitize.js';
import { isSound } from './vary.js';

let pass = 0;
const test = (name, fn) => {
  try { pass++; fn(); } catch (e) { console.error(`✗ ${name}\n  ${e.message}`); process.exitCode = 1; }
};

test('60 draws: every formula sanitizes, frames, and reads as sound', () => {
  for (let i = 0; i < 60; i++) {
    const f = randomFormula();
    const s = sanitizeFormula(f); // throws on any structural problem
    assert.ok(s.ops.length > 0 || s.objects?.length, `draw ${i} is empty`);
    assert.ok(f.camera && Number.isFinite(f.camera.dist) && f.camera.dist > 0, `draw ${i}: bad camera`);
    assert.ok(Number.isFinite(f.iters) && f.iters >= 2, `draw ${i}: bad iters`);
    // The generator's own oracle must agree with what it shipped.
    assert.ok(isSound(f), `draw ${i} ("${f.note}") reads as blank`);
  }
});

test('draws reach beyond the four classic recipe families', () => {
  // The seeded path (50% of draws) walks the preset catalog — across 80 draws
  // the odds of never leaving the classic recipes are ~2^-80. Detect that via
  // op keys the recipes never emit.
  const RECIPE_KEYS = new Set([
    'rotateXY', 'rotateYZ', 'rotateXZ', 'mandelbulbPower', 'absFold', 'mengerFold',
    'scale', 'translate', 'zFold', 'sierpinskiFold', 'boxFold', 'sphereFold', 'kaleido',
  ]);
  let beyond = false;
  for (let i = 0; i < 80 && !beyond; i++) {
    const f = randomFormula();
    beyond = f.ops.some((op) => !RECIPE_KEYS.has(op.key));
  }
  assert.ok(beyond, 'no draw ever used an op outside the classic recipe vocabulary');
});

console.log(`random: ${pass} tests, ${process.exitCode ? 'FAILURES' : 'all green'}`);

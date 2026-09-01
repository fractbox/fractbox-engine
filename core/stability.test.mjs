// Zero-tooling test for the stability predicate. Run: node core/stability.test.mjs
// (Named *.test.mjs so it stays out of the apps' served `core/*.js` surface —
// the test stays at the source of truth, never shipped into an app's core copy.)
import assert from 'node:assert/strict';
import { byKey } from './operators.js';
import {
  stability,
  stands,
  deFamily,
  scaleProduct,
  hybridDeFamily,
  hybridLooseDE,
  BOUNDING_FOLDS,
  NEEDS_RADIUS_BOUND,
} from './stability.js';

let pass = 0;
const test = (name, fn) => {
  try {
    fn();
    pass++;
  } catch (e) {
    console.error(`✗ ${name}\n  ${e.message}`);
    process.exitCode = 1;
  }
};

// Helper: build a formula from [key, ...values] tuples.
const F = (...ops) => ({ name: 'T', ops: ops.map(([key, ...values]) => ({ key, values })) });

// ── drift guard: the curated key-sets must all resolve in the real IR ──
test('curated key-sets resolve against the operator IR (no drift)', () => {
  for (const k of [...BOUNDING_FOLDS, ...NEEDS_RADIUS_BOUND])
    assert.ok(byKey(k), `key "${k}" is not a real operator`);
});

// ── DE-family classification (exact tier) ──
test('empty stack → family empty, does not stand', () => {
  assert.equal(deFamily(F()), 'empty');
  assert.equal(stands(F()), false);
  assert.equal(stability(F()).reasons[0].code, 'empty-stack');
});

test('Mandelbox recipe → ifs family, stands', () => {
  const v = stability(F(['boxFold', 1.0], ['sphereFold', 0.5, 1.0], ['scale', 2.0]));
  assert.equal(v.family, 'ifs');
  assert.equal(v.stands, true);
  assert.deepEqual(v.reasons, []);
});

test('pure bulb → escape family, stands', () => {
  const v = stability(F(['mandelbulbPower', 8.0]));
  assert.equal(v.family, 'escape');
  assert.equal(v.stands, true);
});

test('bulb + w-moving IFS fold → mixed, does not stand (certain)', () => {
  const v = stability(F(['boxFold', 1.0], ['sphereFold', 0.5, 1.0], ['scale', 2.0], ['mandelbulbPower', 8.0]));
  assert.equal(v.family, 'mixed');
  assert.equal(v.stands, false);
  assert.equal(v.certain, true);
  assert.equal(v.reasons[0].code, 'mixed-de');
});

test('bulb + only reflections → escape (not mixed)', () => {
  // absFold is W_UNCHANGED (does not move w), so no family conflict.
  assert.equal(deFamily(F(['absFold'], ['mandelbulbPower', 8.0])), 'escape');
});

// ── documented pairing rule (heuristic tier) ──
test('lone kaleido + scale → escapes (needs a box/sphere fold)', () => {
  const v = stability(F(['kaleido', 6, 0], ['scale', 2.0]));
  assert.equal(v.stands, false);
  const r = v.reasons.find((r) => r.code === 'unbounded-decorator');
  assert.ok(r && r.exact === false, 'expected a heuristic unbounded-decorator fail');
});

test('kaleido WITH a box fold → stands', () => {
  assert.equal(stands(F(['boxFold', 1.0], ['kaleido', 6, 0], ['scale', 2.0])), true);
});

test('lone inversion → escapes; paired with a sphere fold → stands', () => {
  assert.equal(stands(F(['radialInvert', 0, 0, 0], ['scale', 2.0])), false);
  assert.equal(stands(F(['sphereFold', 0.5, 1.0], ['radialInvert', 0, 0, 0], ['scale', 2.0])), true);
});

// ── scale magnitude (invariants.js threshold) ──
test('scale < 2 → stands but warns loose-de', () => {
  const v = stability(F(['boxFold', 1.0], ['sphereFold', 0.5, 1.0], ['scale', 1.5]));
  assert.equal(v.stands, true);
  assert.ok(v.reasons.some((r) => r.code === 'loose-de' && r.severity === 'warn'));
});

test('scaleProduct multiplies |scale| across active scales, ignores sign', () => {
  assert.equal(scaleProduct(F(['scale', -2.0], ['scale', 1.5])), 3.0);
  assert.equal(scaleProduct(F(['boxFold', 1.0])), 1.0);
});

// ── muted ops are excluded everywhere ──
test('a muted bounding fold does not count as present', () => {
  const f = {
    name: 'T',
    ops: [
      { key: 'boxFold', values: [1.0], muted: true },
      { key: 'kaleido', values: [6, 0] },
      { key: 'scale', values: [2.0] },
    ],
  };
  assert.equal(stands(f), false); // the only box fold is muted → kaleido is unpaired
});

// ── hybrid iteration: DE family (union rule) + loose-DE step tightening ──
// Helper: a hybrid formula {slot-A ops, slot-B ops, addC per slot}.
const HYB = (aOps, bOps, { addCA = false, addCB = false } = {}) => ({
  name: 'H',
  ops: aOps.map(([key, ...values]) => ({ key, values })),
  addC: addCA,
  hybrid: {
    b: { ops: bOps.map(([key, ...values]) => ({ key, values })), addC: addCB },
    schedule: { a: 1, b: 1 },
  },
});

test('hybridDeFamily is the union of both slots (fold-only slot is family-neutral)', () => {
  // fold-only × bulb → escape (a W_UNCHANGED slot conflicts with neither family)
  assert.equal(
    hybridDeFamily(HYB([['mengerFold'], ['absFold']], [['mandelbulbPower', 8.0]])),
    'escape',
  );
  // IFS × IFS → ifs
  assert.equal(hybridDeFamily(HYB([['mengerFold'], ['scale', 3.0]], [['boxFold', 1.0], ['scale', 2.0]])), 'ifs');
  // w-moving scale slot × bulb slot → mixed (no shared valid w)
  assert.equal(hybridDeFamily(HYB([['scale', 2.0]], [['mandelbulbPower', 8.0]])), 'mixed');
});

// Regression guard for the §3.3 "ship-row" overstep hazard (the fold-only-slot
// dr under-count): an escape-family hybrid with one fold-only slot must render
// on the LOOSE (tighter) deScale, else the marcher oversteps and drops ~half the
// surface at grazing angles (measured — HYBRID_ITERATION.md §5, sweep 2026-07-01).
test('hybridLooseDE fires for an escape hybrid with a fold-only slot (fold×bulb+addC)', () => {
  const foldBulb = HYB([['absFold'], ['mengerFold'], ['translate', -1, -1, 0]], [['mandelbulbPower', 8.0]], {
    addCA: true,
    addCB: true,
  });
  assert.equal(hybridDeFamily(foldBulb), 'escape');
  assert.equal(hybridLooseDE(foldBulb), true);
});

test('hybridLooseDE stays FALSE for a bulb×bulb escape hybrid (both slots carry dr growth)', () => {
  const bulbBulb = HYB([['mandelbulbPower', 8.0]], [['mandelbulbPower', 3.0]], { addCA: true, addCB: true });
  assert.equal(hybridDeFamily(bulbBulb), 'escape');
  assert.equal(hybridLooseDE(bulbBulb), false);
});

test('hybridLooseDE still fires for a loose-IFS hybrid (|scale| < 2 in either slot)', () => {
  // slot B has scale 1.5 (< 2) → loose IFS, must tighten (pre-existing union rule)
  assert.equal(hybridLooseDE(HYB([['mengerFold'], ['scale', 3.0]], [['boxFold', 1.0], ['scale', 1.5]])), true);
});

test('hybridLooseDE FALSE for a tight IFS×IFS hybrid (all |scale| ≥ 2)', () => {
  assert.equal(hybridLooseDE(HYB([['mengerFold'], ['scale', 3.0]], [['boxFold', 1.0], ['scale', 2.0]])), false);
});

console.log(`stability.test.mjs: ${pass} passed`);

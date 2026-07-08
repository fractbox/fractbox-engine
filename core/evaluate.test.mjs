// Zero-tooling test for the CPU evaluator + wobble/lean measure.
// Run: node core/evaluate.test.mjs   (named *.test.mjs so sync skips it)
import assert from 'node:assert/strict';
import { byKey, OPERATORS } from './operators.js';
import { measure, SUPPORTED } from './evaluate.js';

let pass = 0;
const test = (name, fn) => {
  try { fn(); pass++; } catch (e) { console.error(`✗ ${name}\n  ${e.message}`); process.exitCode = 1; }
};
const F = (...ops) => ({ name: 'T', ops: ops.map(([key, ...values]) => ({ key, values })) });

const MANDELBOX = F(['boxFold', 1.0], ['sphereFold', 0.5, 1.0], ['scale', 2.0]);
const LONE_KALEIDO = F(['kaleido', 6, 0], ['scale', 2.0]);
const KALEIDO_BOX = F(['boxFold', 1.0], ['kaleido', 6, 0], ['scale', 2.0]);
// A real bulb adds +c every iteration (z = z^p + c) — gated on addC, like every
// escape-time preset. Without addC it would be the trivial z = z^p map.
const BULB = { ...F(['mandelbulbPower', 8.0]), addC: true };
const MIXED = F(['boxFold', 1.0], ['sphereFold', 0.5, 1.0], ['scale', 2.0], ['mandelbulbPower', 8.0]);

// ── drift guard ──
test('every SUPPORTED key resolves in the operator IR', () => {
  for (const k of SUPPORTED) assert.ok(byKey(k), `"${k}" is not a real operator`);
});

// ── exact short-circuits ──
test('empty stack → wobble 1', () => {
  const m = measure(F());
  assert.equal(m.family, 'empty');
  assert.equal(m.wobble, 1);
});
test('mixed DE → wobble 1 (certain break)', () => {
  const m = measure(MIXED);
  assert.equal(m.family, 'mixed');
  assert.equal(m.wobble, 1);
});
test('unsupported op → supported:false (no silent mis-measure)', () => {
  // A key not in SUPPORTED must fail closed rather than silently mis-measure.
  // (Uses a synthetic key — every real operator is now ported to the evaluator.)
  const m = measure(F(['__unported_op__', 0.005], ['scale', 3.0]));
  assert.equal(m.supported, false);
  assert.equal(m.wobble, null);
});

test('SUPPORTED covers every real operator (evaluator parity with the IR)', () => {
  // Guards the drift that hid the `menger` gap: measure() only grades a formula
  // when every op is in SUPPORTED, so a real op missing here silently downgrades
  // presets that use it. If you add an operator, port it to applyOp + SUPPORTED.
  const sup = new Set(SUPPORTED);
  const missing = OPERATORS.map((o) => o.key).filter((k) => !sup.has(k));
  assert.deepEqual(missing, [], `operators missing from evaluate.SUPPORTED: ${missing.join(', ')}`);
});

// ── the core continuum: stable < broken ──
test('Mandelbox stands: low wobble, real coverage', () => {
  const m = measure(MANDELBOX);
  assert.ok(m.supported);
  assert.ok(m.coverage > 0.1, `expected coverage > 0.1, got ${m.coverage.toFixed(3)}`);
  assert.ok(m.wobble < 0.35, `expected wobble < 0.35, got ${m.wobble.toFixed(3)}`);
});
test('lone kaleido escapes: high wobble, ~no coverage', () => {
  const m = measure(LONE_KALEIDO);
  assert.ok(m.coverage < 0.03, `expected coverage < 0.03, got ${m.coverage.toFixed(3)}`);
  assert.ok(m.wobble > 0.8, `expected wobble > 0.8, got ${m.wobble.toFixed(3)}`);
});
test('pure bulb renders: low-ish wobble', () => {
  const m = measure(BULB);
  assert.equal(m.family, 'escape');
  assert.ok(m.coverage > 0.05, `expected coverage > 0.05, got ${m.coverage.toFixed(3)}`);
});

// ── +c parity with the renderer: addC must actually feed into the iteration ──
// (regression guard: the proxy used to gate +c on escape-time, so it silently
// ignored addC on IFS presets — every Mandelbox-family formula was mis-measured.)
test('addC is honored: toggling +c changes the measured iteration', () => {
  const withC = measure({ ...MANDELBOX, addC: true });
  const without = measure({ ...MANDELBOX, addC: false });
  // Coverage saturates for a dense Mandelbox either way, but the +c add reshapes
  // the attractor — meanSharp must move. (Was identical when +c gated on escape.)
  assert.ok(
    Math.abs(withC.meanSharp - without.meanSharp) > 0.5,
    `addC should alter the attractor: ${withC.meanSharp.toFixed(2)} vs ${without.meanSharp.toFixed(2)}`,
  );
});

// ── monotonicity: adding a bounding fold reduces wobble ──
test('kaleido+box wobbles LESS than lone kaleido', () => {
  assert.ok(measure(KALEIDO_BOX).wobble < measure(LONE_KALEIDO).wobble);
});
test('over-expansion raises wobble: scale 3.5 ≥ scale 2.0', () => {
  const lo = measure(F(['boxFold', 1.0], ['sphereFold', 0.5, 1.0], ['scale', 2.0]));
  const hi = measure(F(['boxFold', 1.0], ['sphereFold', 0.5, 1.0], ['scale', 3.5]));
  assert.ok(hi.wobble >= lo.wobble, `scale3.5 ${hi.wobble.toFixed(3)} should be ≥ scale2.0 ${lo.wobble.toFixed(3)}`);
});

// ── surface geometry: this palette re-centers attractors, so lean (COM) is ~0
// for standing fractals; the directional cue is anisotropy (elongation axis). ──
test('a standing centered fractal has ~0 lean (folds re-center it)', () => {
  assert.ok(measure(MANDELBOX).leanMag < 0.1, `lean ${measure(MANDELBOX).leanMag.toFixed(3)}`);
});
test('elongation detects anisotropy: surfFold (Z free) stretches more than an isotropic Mandelbox', () => {
  const iso = measure(MANDELBOX);
  const sheet = measure(F(['surfFold', 1.0], ['sphereFold', 0.5, 1.0], ['scale', 2.0]));
  assert.ok(iso.elong < 1.6, `Mandelbox should be ~isotropic, got elong ${iso.elong.toFixed(2)}`);
  assert.ok(sheet.elong > iso.elong, `surfFold elong ${sheet.elong.toFixed(2)} should exceed Mandelbox ${iso.elong.toFixed(2)}`);
  assert.equal(sheet.axis, 'z', `surfFold should stretch along z, got ${sheet.axis}`);
});

// ── determinism (Halton sampling, no RNG) ──
test('measure is deterministic', () => {
  const a = measure(MANDELBOX), b = measure(MANDELBOX);
  assert.equal(a.wobble, b.wobble);
  assert.equal(a.coverage, b.coverage);
});

console.log(`evaluate.test.mjs: ${pass} passed`);

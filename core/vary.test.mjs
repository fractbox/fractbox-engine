// Zero-tooling test for the metadata jitter + numeric soundness gate.
// Run: node core/vary.test.mjs   (named *.test.mjs so sync skips it)
import assert from 'node:assert/strict';
import { OPERATORS } from './operators.js';
import { PRESETS } from './oplist.js';
import { jitterParams, isSound, soundCandidate, TOUCHY } from './vary.js';

let pass = 0;
const test = (name, fn) => {
  try { pass++; fn(); } catch (e) { console.error(`✗ ${name}\n  ${e.message}`); process.exitCode = 1; }
};

// A deterministic rng that walks the extremes — worst case for range respect.
const extremes = () => { let i = 0; return () => (i++ % 2 === 0 ? 0 : 1); };

const plainPresets = PRESETS.filter((p) => !p.objects?.length && !p.hybrid);

// ── jitterParams stays inside every op's declared param range ──
test('jittered params respect declared min/max for every operator', () => {
  for (const op of OPERATORS) {
    if (!op.params.length) continue;
    const f = {
      name: 'T',
      ops: [{ key: op.key, values: op.params.map((p) => p.default) }],
    };
    for (const spread of [0.15, 0.5, 1.0]) {
      const j = jitterParams(f, { spread, rng: extremes() });
      j.ops[0].values.forEach((v, i) => {
        const m = op.params[i];
        assert.ok(v >= m.min && v <= m.max, `${op.key}[${i}] = ${v} outside [${m.min}, ${m.max}]`);
      });
    }
  }
});

test('integer-stepped params stay on their step grid', () => {
  const f = { name: 'T', ops: [{ key: 'kaleido', values: [6, 0] }] };
  for (let n = 0; n < 20; n++) {
    const v = jitterParams(f, { spread: 1.0 }).ops[0].values[0];
    assert.equal(v % 1, 0, `kaleido sectors = ${v} is not an integer`);
  }
});

test('jitter does not mutate the source and covers hybrid slot B', () => {
  const f = {
    name: 'T',
    ops: [{ key: 'boxFold', values: [1.0] }],
    hybrid: { schedule: 'alternate', b: { ops: [{ key: 'scale', values: [2.0] }], addC: false } },
  };
  const j = jitterParams(f, { spread: 1.0, rng: () => 1 });
  assert.equal(f.ops[0].values[0], 1.0, 'source op mutated');
  assert.equal(f.hybrid.b.ops[0].values[0], 2.0, 'source hybrid op mutated');
  assert.notEqual(j.ops[0].values[0], 1.0, 'op not jittered');
  assert.notEqual(j.hybrid.b.ops[0].values[0], 2.0, 'hybrid slot-B op not jittered');
});

test('TOUCHY movers are jittered on a shorter leash', () => {
  const f = { name: 'T', ops: [{ key: 'translate', values: [0, 0, 0] }] };
  assert.ok(TOUCHY.has('translate'));
  const j = jitterParams(f, { spread: 1.0, touchyScale: 0.25, rng: () => 1 });
  // translate span is [-2,2] → full-spread nudge would be 4; leashed = 1.
  for (const v of j.ops[0].values) assert.ok(Math.abs(v) <= 1.01, `translate leash broken: ${v}`);
});

// ── the soundness gate agrees with the shipped catalog ──
test('every plain preset passes isSound', () => {
  for (const p of plainPresets) {
    assert.ok(isSound(p), `"${p.name}" read as unsound — gate would reject a shipped preset`);
  }
});

test('a collapsed IFS (contraction scale) fails isSound', () => {
  // Regression: a preset-seeded roll of Tetra VS with scale 0.01 shipped as a
  // share link that rendered empty. Contracting IFS collapse to a point with
  // escaped=0 — the escape-time "something converged" accept signal must not
  // apply to the ifs family (wobble=1 is the truth there).
  const collapsed = {
    name: 'T',
    addC: false,
    iters: 15,
    deOption: 2,
    ops: [
      { key: 'sierpinskiFold', values: [] },
      { key: 'varyScale', values: [0.76, 0.32, 1.05] },
      { key: 'scale', values: [0.01] },
      { key: 'translate', values: [-0.8, -0.92, -0.95] },
    ],
    camera: { yawDeg: 30, pitchDeg: 20, dist: 8, fovDeg: 42 },
  };
  assert.equal(isSound(collapsed), false, 'collapsed IFS passed the gate');
});

test('a near-unit-scale IFS (space-filling solid) fails isSound', () => {
  // Regression: a preset-seeded roll of Octahedron with scale 2.0 → 1.13
  // shipped as a share link that rendered blank. The sampler honestly reads
  // it as SOLID (wobble 0.30) — the figure balloons to ~|t|·s/(s−1) ≈ 10 and
  // the seed's tuned camera (dist 4.5) sits inside it. The expansion guard
  // (|scaleProduct| ≥ 1.6) rejects the whole degenerate class structurally.
  const drifting = {
    name: 'T',
    addC: false,
    iters: 12,
    deOption: 2,
    ops: [
      { key: 'octaFold', values: [] },
      { key: 'scale', values: [1.13] },
      { key: 'translate', values: [-1.02, 0.27, -0.47] },
    ],
    camera: { yawDeg: 359.2, pitchDeg: -19.1, dist: 4.5, fovDeg: 42 },
  };
  assert.equal(isSound(drifting), false, 'space-filling IFS passed the gate');
});

test('a broken menger lattice (everything escapes) fails isSound', () => {
  // Regression: a preset-seeded roll of Rounded Menger with scale 3 → 4 broke
  // the lattice's scale/translate interlock — every probe sample escapes
  // (escaped = 1.000), yet measure() reports a neutral LOW wobble (0.30) for
  // the empty inner region. The evidence rule (a region may only accept when
  // something in it converged) rejects it.
  const brokenLattice = {
    name: 'T',
    addC: false,
    iters: 9,
    deOption: 2,
    ops: [
      { key: 'menger', values: [0.07] },
      { key: 'scale', values: [4] },
      { key: 'translate', values: [-1.53, -2, -0.28] },
    ],
    camera: { yawDeg: 0, pitchDeg: 0, dist: 8.5, fovDeg: 42 },
  };
  assert.equal(isSound(brokenLattice), false, 'all-escaped lattice passed the gate');
});

test('empty and mixed-DE formulas fail isSound', () => {
  assert.equal(isSound({ name: 'T', ops: [] }), false, 'empty passed');
  const mixed = {
    name: 'T',
    ops: [
      { key: 'boxFold', values: [1.0] },
      { key: 'sphereFold', values: [0.5, 1.0] },
      { key: 'scale', values: [2.0] },
      { key: 'mandelbulbPower', values: [8.0] },
    ],
  };
  assert.equal(isSound(mixed), false, 'mixed DE passed');
});

// ── soundCandidate: accepts a good roll, falls back after bad ones ──
test('soundCandidate returns the first sound draw, or the fallback', () => {
  const good = plainPresets[0];
  const bad = { name: 'T', ops: [] };
  assert.equal(soundCandidate(() => good, () => bad), good);
  let draws = 0;
  const out = soundCandidate(
    () => { draws++; return bad; },
    () => good,
    { attempts: 3 },
  );
  assert.equal(draws, 3, 'attempt budget not honored');
  assert.equal(out, good, 'fallback not used');
});

console.log(`vary: ${pass} tests, ${process.exitCode ? 'FAILURES' : 'all green'}`);

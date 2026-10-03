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

// ── hybrid honesty (single-orbit-runner refactor) ────────────────────────────
// measure() used to iterate slot A only — a jitter that broke a hybrid's slot B
// passed the vary.js ship-gate and shipped a blank share link. The probes now
// derive from cpu.js makeOrbit, which runs the real A/B schedule.
const HYBRID = (bOps, sched = { a: 1, b: 1 }) => ({
  ...MANDELBOX,
  hybrid: { schedule: sched, b: { ops: bOps.map(([key, ...values]) => ({ key, values })) } },
});

test('hybrid: benign slot B measures solid', () => {
  const m = measure(HYBRID([['rotateXYZ', 15, 10, 5]]));
  assert.ok(m.supported, 'benign hybrid must be supported');
  assert.ok(m.coverage > 0.5, `expected coverage > 0.5, got ${m.coverage.toFixed(3)}`);
  assert.ok(m.wobble < 0.35, `expected wobble < 0.35, got ${m.wobble.toFixed(3)}`);
});
test('hybrid: catastrophic slot B raises wobble (gate blindspot regression)', () => {
  // scale 100 every other iteration blows every orbit past the bailout — the
  // render is blank. Blind-to-B measure() read this as the solid slot-A box.
  const broken = measure(HYBRID([['scale', 100]]));
  const benign = measure(HYBRID([['rotateXYZ', 15, 10, 5]]));
  assert.ok(broken.wobble > 0.6, `broken slot B must wobble hard, got ${broken.wobble.toFixed(3)}`);
  assert.ok(broken.escaped > 0.9, `broken slot B must escape, got ${broken.escaped.toFixed(3)}`);
  assert.ok(broken.wobble > benign.wobble + 0.3,
    `broken ${broken.wobble.toFixed(3)} must clearly exceed benign ${benign.wobble.toFixed(3)}`);
  // And the old blind behavior — hybrid measured identical to flat slot A — is gone.
  const flatA = measure(MANDELBOX);
  assert.notEqual(broken.coverage, flatA.coverage, 'slot B must actually participate in the orbit');
});
test('hybrid: slot B actually runs on the benign path too', () => {
  const withB = measure(HYBRID([['rotateXYZ', 15, 10, 5]]));
  const flatA = measure(MANDELBOX);
  assert.ok(Math.abs(withB.meanSharp - flatA.meanSharp) > 0.05,
    `slot B must reshape the attractor: hybrid ${withB.meanSharp.toFixed(3)} vs flat ${flatA.meanSharp.toFixed(3)}`);
});
test('hybrid: family is the slot-UNION verdict (mixed across slots → certain break)', () => {
  // Slot A moves w (sphereFold/scale), slot B adds a bulb — per-slot both look
  // fine; the union is the render-tier 'mixed' conflict and must read wobble 1.
  const m = measure(HYBRID([['mandelbulbPower', 8.0]]));
  assert.equal(m.family, 'mixed');
  assert.equal(m.wobble, 1);
});
test('hybrid: unsupported op in slot B fails closed', () => {
  const m = measure(HYBRID([['__unported_op__', 0.5]]));
  assert.equal(m.supported, false);
  assert.equal(m.wobble, null);
});

// ── flat numerics are frozen (single-orbit-runner refactor) ──────────────────
// The tuned wobble thresholds (vary.js isSound, the games) must not shift for
// non-hybrid formulas. Snapshots captured from the pre-refactor evaluator
// (deAt/probePoint private loops) at commit 93b2f6a — the runner-derived
// probes must reproduce them bit-for-bit (asserted to 1e-12).
test('flat measure snapshot: pre-refactor values reproduced', () => {
  const JULIA_BOX = { ...MANDELBOX, julia: true, juliaC: [0.2, 0.1, -0.3] };
  const SNAP = [
    [MANDELBOX,   { wobble: 0,                   coverage: 1,           meanSharp: 13.916792036915641, escaped: 0 }],
    [BULB,        { wobble: 0.25722932535847515, coverage: 0.994140625, meanSharp: 3.425689154717496,  escaped: 0.98046875 }],
    [KALEIDO_BOX, { wobble: 0.08430488006634342, coverage: 1,           meanSharp: 9.189837331121886,  escaped: 0 }],
    [JULIA_BOX,   { wobble: 0,                   coverage: 1,           meanSharp: 12.893357465550377, escaped: 0 }],
  ];
  for (const [f, want] of SNAP) {
    const m = measure(f);
    for (const k of Object.keys(want)) {
      assert.ok(Math.abs(m[k] - want[k]) < 1e-12,
        `${f.name ?? '?'}.${k}: expected ${want[k]}, got ${m[k]}`);
    }
  }
});

// ── scenes: analytic primitives grade solid (objType mask regression) ────────
// evaluate.js used to mask objType & 3 (everywhere else in core it's & 0xf, types
// 0–6), so cylinder (4 & 3 = 0) fell into the IFS branch as an op-less object →
// wobble 1 → false "toppling/blank" reject in the vary.js ship-gate; torus (3)
// misrouted; capsule (5) / plane (6) only passed by aliasing onto 1/2.
const SCENE = (objType, extra = {}) => ({
  objects: [{ objType, origin: [0, 0, 0], ...extra }],
});
test('scene: every analytic primitive type 1–6 measures solid', () => {
  const cases = [
    [1, { primParam: 1 }],            // box
    [2, { primParam: 1 }],            // sphere
    [3, { primParam: 1, primParam2: 0.25 }], // torus
    [4, { primParam: 0.5, primParam2: 0.5 }], // cylinder
    [5, { primParam: 0.3, primParam2: 0.5 }], // capsule
    [6, { primParam: 0.1 }],          // plane
  ];
  for (const [t, extra] of cases) {
    const m = measure(SCENE(t, extra));
    assert.ok(m.supported, `type ${t}: expected supported`);
    assert.equal(m.wobble, 0, `type ${t}: expected wobble 0, got ${m.wobble}`);
    assert.ok(m.coverage >= 1, `type ${t}: expected full coverage, got ${m.coverage}`);
  }
});
test('scene: torus + cylinder were the false rejects (pinned regression)', () => {
  // With the & 3 mask these graded wobble 1 (certain topple) and the vary.js
  // gate rejected the scene as blank/toppling. They are exact solids: wobble 0.
  for (const t of [3, 4]) {
    const m = measure(SCENE(t, { primParam: 1, primParam2: 0.3 }));
    assert.equal(m.wobble, 0, `objType ${t} must not false-reject (got wobble ${m.wobble})`);
  }
});

test('D0 guard: a shapeId ≥ 7 / iterShape scene measures without crashing', () => {
  // evaluate.js is leaf-unaware by design (PRIMITIVE_DIFS_D0 §2.8): the
  // conservative objType alias routes new-leaf/mixed/iterShape objects into
  // its existing op-chain path. This pins "no crash, sane shape" — not values.
  const scene = {
    name: 'd0',
    ops: [],
    iters: 8,
    deOption: 2,
    addC: false,
    objects: [
      { shapeId: 7, shapeParams: [1, 0, 0, 0], ops: [], iters: 1, combine: 0, blendK: 0 },
      {
        shapeId: 1,
        shapeParams: [0.75, 0, 0, 0],
        iterShape: true,
        ops: [{ key: 'boxFold', values: [1] }, { key: 'scale', values: [2] }],
        iters: 6,
        combine: 0,
        blendK: 0,
      },
    ],
  };
  const m = measure(scene, { region: 2.5, samples: 64 });
  assert.ok(m && typeof m.wobble === 'number' && !Number.isNaN(m.wobble), 'measureScene crashed or NaNed');
  pass++;
});

console.log(`evaluate.test.mjs: ${pass} passed`);

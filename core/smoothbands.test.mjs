// COLORING P0 — smooth escape bands (S1, PR-B). Pins the smoothEscFrac
// heuristic that de-staircases the "bands" color mode, and the makeIterMeasure
// "escape" output that consumes it. The WGSL/GLSL mirrors (shader.js /
// shader_gl.js smoothEscFrac) use the identical formula + guards, so this is
// the cross-tier regression net (CI has no GPU). Run:
//   node --test core/smoothbands.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";
import {
  smoothEscFrac,
  makeIterMeasure,
  makeDE,
  curvatureMixT,
  robustRange,
  sampleSignalMixT,
  signalRange,
  makePainterMeasure,
} from "./cpu.js";
import { BAILOUT_ESCAPE } from "./limits.js";

test("smoothEscFrac stays in [0,1] and never NaN across the full sweep", () => {
  const bailSq = 256; // rBail = 16
  // rEsc from below the bailout radius, across it, to far past its square.
  for (let rEsc = 0; rEsc <= 5000; rEsc += 0.37) {
    const f = smoothEscFrac(rEsc, bailSq);
    assert.ok(Number.isFinite(f), `NaN/Inf at rEsc=${rEsc}`);
    assert.ok(f >= 0 && f <= 1, `out of range ${f} at rEsc=${rEsc}`);
  }
});

test("smoothEscFrac endpoints: ≈1 at the bailout radius, →0 by its square", () => {
  const bailSq = 256; // rBail = 16
  assert.ok(smoothEscFrac(16.0001, bailSq) > 0.999); // just escaped → ~full band
  assert.ok(smoothEscFrac(256, bailSq) < 0.001); // rBail² → next band edge
  assert.equal(smoothEscFrac(1e6, bailSq), 0); // hard escape → clamped
});

test("smoothEscFrac is monotonic decreasing within [rBail, rBail²]", () => {
  const bailSq = 256;
  let prev = Infinity;
  for (let rEsc = 16; rEsc <= 256; rEsc += 1) {
    const f = smoothEscFrac(rEsc, bailSq);
    assert.ok(f <= prev + 1e-9, `not monotonic at rEsc=${rEsc}`);
    prev = f;
  }
});

test("#239 finding 8 — a bailout < 1 (rBail<1) can't produce NaN", () => {
  // The squared-radius escape test means bailSq<1 → rBail<1 → log(rBail)<0 →
  // the log ratio would flip sign. The guard returns 0 instead of NaN.
  for (const bailSq of [0.25, 0.5, 0.9]) {
    for (let rEsc = 0; rEsc < 50; rEsc += 0.5) {
      const f = smoothEscFrac(rEsc, bailSq);
      assert.ok(Number.isFinite(f) && f >= 0 && f <= 1, `bad at bailSq=${bailSq} rEsc=${rEsc}`);
    }
  }
});

test("makeIterMeasure escape output is bounded [0,1] and de-staircased", () => {
  // A power-8 escape formula so 'bands' mode has real escape structure.
  const formula = {
    ops: [{ key: "mandelbulbPower", values: [8, 0, 0] }],
    addC: true,
    iters: 12,
    deOption: 0,
  };
  const measure = makeIterMeasure(formula, "escape", 12);
  const quantum = 1 / 12; // the old integer step
  let offGrid = 0,
    n = 0;
  // sample a slab through the bulb where points escape at varying iterations
  for (let x = -1.4; x <= 1.4; x += 0.05) {
    for (let z = -1.4; z <= 1.4; z += 0.05) {
      const v = measure(x, 0.2, z);
      assert.ok(v >= 0 && v <= 1 && Number.isFinite(v), `bands ${v} @(${x},${z})`);
      // count samples that are NOT on the integer grid → proof of smoothing
      const frac = (v / quantum) % 1;
      if (frac > 1e-3 && frac < 1 - 1e-3) offGrid++;
      n++;
    }
  }
  // The pre-S1 measure returned only multiples of 1/iters; the smooth version
  // must put a substantial share of escaped samples strictly between bands.
  assert.ok(offGrid > n * 0.1, `too few smoothed samples: ${offGrid}/${n}`);
});

test("Silk (S2) — makeIterMeasure 'silk' is bounded [0,1] and frequency-sensitive", () => {
  const formula = {
    ops: [{ key: "mandelbulbPower", values: [8, 0, 0] }],
    addC: true,
    iters: 12,
    deOption: 0,
  };
  const lowK = makeIterMeasure(formula, "silk", 12, 2);
  const highK = makeIterMeasure(formula, "silk", 12, 12);
  let differ = 0,
    n = 0;
  for (let x = -1.4; x <= 1.4; x += 0.1) {
    for (let z = -1.4; z <= 1.4; z += 0.1) {
      const a = lowK(x, 0.2, z);
      const b = highK(x, 0.2, z);
      assert.ok(a >= 0 && a <= 1 && Number.isFinite(a), `silk ${a} @(${x},${z})`);
      assert.ok(b >= 0 && b <= 1 && Number.isFinite(b), `silk ${b} @(${x},${z})`);
      if (Math.abs(a - b) > 1e-3) differ++;
      n++;
    }
  }
  // A different stripe frequency must change the pattern on a real fraction.
  assert.ok(differ > n * 0.2, `stripeFreq had too little effect: ${differ}/${n}`);
});

test("Pinwheel (S3) — makeIterMeasure 'pin' is bounded [0,1) and angle-sensitive", () => {
  const formula = {
    ops: [{ key: "mandelbulbPower", values: [8, 0, 0] }],
    addC: true,
    iters: 12,
    deOption: 0,
  };
  const pin = makeIterMeasure(formula, "pin", 12);
  const vals = [];
  for (let x = -1.4; x <= 1.4; x += 0.1) {
    for (let z = -1.4; z <= 1.4; z += 0.1) {
      const a = pin(x, 0.2, z);
      // fract() output is [0,1); every sample must be finite and in range.
      assert.ok(a >= 0 && a < 1 && Number.isFinite(a), `pin ${a} @(${x},${z})`);
      vals.push(a);
    }
  }
  // The trap angle sweeps a real spread across the field — not a flat constant.
  const min = Math.min(...vals),
    max = Math.max(...vals);
  assert.ok(max - min > 0.5, `pinwheel angle range too narrow: ${max - min}`);
});

test("Curvature (S4) — curvatureMixT is bounded [0,1] and shape-sensitive", () => {
  const formula = {
    ops: [{ key: "mandelbulbPower", values: [8, 0, 0] }],
    addC: true,
    iters: 10,
    deOption: 0,
  };
  const de = makeDE(formula, 10);
  const vals = [];
  for (let x = -1.3; x <= 1.3; x += 0.13) {
    for (let z = -1.3; z <= 1.3; z += 0.13) {
      const a = curvatureMixT(de, x, 0.15, z);
      assert.ok(a >= 0 && a <= 1 && Number.isFinite(a), `curv ${a} @(${x},${z})`);
      vals.push(a);
    }
  }
  // A flat DE field would give a constant 0.5 everywhere; a real fractal's
  // varying curvature must spread the signal off that midpoint.
  const min = Math.min(...vals),
    max = Math.max(...vals);
  assert.ok(max - min > 0.05, `curvature signal too flat: range ${max - min}`);
});

test("Curvature (S4) — a scene DE (CSG objects) also yields a real signal", () => {
  const scene = {
    objects: [
      { ops: [{ key: "mandelbulbPower", values: [8, 0, 0] }], addC: true, iters: 8 },
      { ops: [{ key: "boxFold", values: [1, 0, 0] }], iters: 8 },
    ],
  };
  const de = makeDE(scene, 8);
  let spread = 0,
    n = 0,
    off = 0;
  for (let x = -1.5; x <= 1.5; x += 0.15) {
    for (let z = -1.5; z <= 1.5; z += 0.15) {
      const a = curvatureMixT(de, x, 0.1, z);
      assert.ok(a >= 0 && a <= 1 && Number.isFinite(a), `scene curv ${a}`);
      if (Math.abs(a - 0.5) > 1e-3) off++;
      n++;
    }
  }
  // The scene-aware DE must produce a non-trivial curvature field (not all 0.5).
  assert.ok(off > n * 0.1, `scene curvature too flat: ${off}/${n} off-midpoint`);
});

test("Auto-levels (P2) — robustRange rejects outliers and floors a flat signal", () => {
  // A tight cluster near 0.5 with two far outliers → the p3/p97 range ignores
  // the outliers; a truly constant signal returns the floor, not span 0.
  const clustered = [];
  for (let i = 0; i < 100; i++) clustered.push(0.5 + (i % 5) * 0.002);
  clustered.push(0.0, 1.0); // outliers
  const { lo, span } = robustRange(clustered);
  assert.ok(lo > 0.4 && lo < 0.55, `lo ${lo} should sit in the cluster`);
  assert.ok(span >= 0.06 && span < 0.3, `span ${span} floored, outliers rejected`);
  const flat = new Array(50).fill(0.7);
  assert.ok(robustRange(flat).span >= 0.06, "constant signal floored, not zero");
  // Too few samples → identity.
  assert.deepEqual(robustRange([0.1, 0.2]), { lo: 0, span: 1 });
});

test("Auto-levels (P2) — signalRange gates off / normalizes the right modes", () => {
  const formula = {
    ops: [{ key: "mandelbulbPower", values: [8, 0, 0] }],
    addC: true,
    iters: 10,
    deOption: 0,
    camera: { dist: 4 }, // frame the bulb (real presets always carry a camera)
  };
  // Off → identity regardless of mode.
  assert.deepEqual(signalRange(formula, { mode: 1, autoLevels: false }, 10), {
    lo: 0,
    span: 1,
  });
  // Surface (0) and pinwheel (4) are never normalized even with autoLevels on.
  assert.deepEqual(signalRange(formula, { mode: 0, autoLevels: true }, 10), {
    lo: 0,
    span: 1,
  });
  assert.deepEqual(signalRange(formula, { mode: 4, autoLevels: true }, 10), {
    lo: 0,
    span: 1,
  });
  // Curvature (5) with autoLevels on → a real, bounded, non-identity range.
  const r = signalRange(formula, { mode: 5, autoLevels: true }, 10);
  assert.ok(r.span > 0 && r.span <= 1, `curvature span ${r.span}`);
  assert.ok(r.lo >= 0 && r.lo + r.span <= 1.0001, `range within [0,1]: ${r.lo}..${r.lo + r.span}`);
});

test("Auto-levels (P2) — sampleSignalMixT yields in-range samples for a signal mode", () => {
  const formula = {
    ops: [{ key: "mandelbulbPower", values: [8, 0, 0] }],
    addC: true,
    iters: 10,
    deOption: 0,
    camera: { dist: 4 },
  };
  const vals = sampleSignalMixT(formula, { mode: 5 }, { iters: 10 });
  assert.ok(vals.length > 20, `expected a populated sample, got ${vals.length}`);
  for (const v of vals) assert.ok(v >= 0 && v <= 1 && Number.isFinite(v), `sample ${v}`);
});

test("Auto-levels (P2) — a formula the canonical view can't frame fails safe to identity", () => {
  // No hits → < 8 samples → robustRange identity, so auto-levels is a safe no-op
  // rather than dividing by a bogus range.
  const unframed = {
    ops: [{ key: "mandelbulbPower", values: [8, 0, 0] }],
    addC: true,
    iters: 10,
    deOption: 0,
    camera: { dist: 400 }, // object is a sub-pixel speck → no coverage
  };
  assert.deepEqual(signalRange(unframed, { mode: 5, autoLevels: true }, 10), {
    lo: 0,
    span: 1,
  });
});

test("Painter (R S7) — makePainterMeasure returns a valid sRGB albedo that varies", () => {
  const formula = {
    ops: [{ key: "mandelbulbPower", values: [8, 0, 0] }],
    addC: true,
    iters: 12,
    deOption: 0,
  };
  const coloring = {
    mode: 6,
    palette: {
      on: true,
      stops: [
        { c: [0.9, 0.1, 0.1], p: 0 },
        { c: [0.1, 0.9, 0.2], p: 0.5 },
        { c: [0.1, 0.2, 0.9], p: 1 },
      ],
      cyclic: true,
    },
  };
  const paint = makePainterMeasure(formula, coloring, 12);
  const seen = new Set();
  for (let x = -1.3; x <= 1.3; x += 0.13) {
    for (let z = -1.3; z <= 1.3; z += 0.13) {
      const rgb = paint(x, 0.15, z);
      assert.equal(rgb.length, 3);
      for (const ch of rgb)
        assert.ok(ch >= 0 && ch <= 1 && Number.isFinite(ch), `channel ${ch}`);
      seen.add(rgb.map((v) => Math.round(v * 8)).join(","));
    }
  }
  assert.ok(seen.size > 10, `painter too uniform: ${seen.size} distinct colors`);
});

test("Address (R S8) — makeIterMeasure 'address' returns discrete octant bands", () => {
  const formula = {
    ops: [{ key: "boxFold", values: [1, 0, 0] }, { key: "scale", values: [2, 0, 0] }],
    addC: true,
    iters: 10,
    deOption: 0,
  };
  const addr = makeIterMeasure(formula, "address", 10);
  const bands = new Set();
  for (let x = -1.4; x <= 1.4; x += 0.1) {
    for (let z = -1.4; z <= 1.4; z += 0.1) {
      const v = addr(x, 0.2, z);
      assert.ok(v >= 0 && v < 1 && Number.isFinite(v), `address ${v}`);
      // must land on an octant band center: (oct + 0.5)/8 for oct 0..7.
      const oct = v * 8 - 0.5;
      assert.ok(Math.abs(oct - Math.round(oct)) < 1e-9, `not an octant band: ${v}`);
      bands.add(Math.round(oct));
    }
  }
  // A folded formula should reach several distinct octants across the field.
  assert.ok(bands.size >= 3, `too few address regions: ${bands.size}`);
});

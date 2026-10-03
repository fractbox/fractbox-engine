import { test } from "node:test";
import assert from "node:assert/strict";
import {
  splatMetrics,
  metricOnSurface,
  metricCoverage,
  metricOverdraw,
  metricNormalAgreement,
  metricColorDrift,
} from "./splatmetrics.js";
import { captureSplats, aoScale } from "./splatcapture.js";
import { makeDE, makePointAlbedo } from "./cpu.js";
import { PRESETS } from "./oplist.js";
import { defaultColoring } from "./coloring.js";

const F = (a) => Float32Array.from(a);
const close = (a, b, e = 1e-9) => Math.abs(a - b) <= e;

// ── Unit tests: hand-placed clouds pin each metric's arithmetic exactly ──────

test("metricOnSurface: fraction of centers within k·r0 of the surface", () => {
  const de = (x, y, z) => Math.hypot(x, y, z) - 1; // unit sphere
  const points = {
    count: 4,
    // three on the sphere (|de|=0), one at radius 2 (|de|=1)
    pos: F([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 2]),
    normal: F([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]),
    albedo: F([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
  };
  assert.ok(
    close(metricOnSurface(de, points, 0.1, 1), 0.75),
    "3 of 4 on-surface",
  );
  assert.ok(
    close(metricOnSurface(de, points, 2, 1), 1.0),
    "loose threshold catches all",
  );
});

test("splatMetrics #536: onSurface floors r0 against captureEps — a floored DE with a sub-eps r0 still verifies", () => {
  // #536 (post-#583): #583 fixed the reported Menger corner crop by raising
  // the crop's iteration count, which tightens the DE's MEASURED convergence
  // floor. But the object's own scale-relative floor (splatcapture.js
  // EPS_SCALE·radius) is a hard bound no iteration count pushes eps below,
  // while r0 (computeR0 = radiusScale·diag/√hits) keeps shrinking without
  // limit as a capture volume keeps shrinking — so a tight-enough crop still
  // crosses eps > r0. Measured on dev post-#583 (Menger corner crop,
  // magnification swept against the object frame): crossover at ×150
  // (eps 3.30e-4 vs r0 3.27e-4, onSurface already down to 0.9943), degrading
  // further at deeper crops (×1500: eps/r0 = 11.1, onSurface 0.5624) — the
  // "eps > r0, onSurface unverifiable" failure this issue describes, just at
  // a tighter close-up than #583's own regression pins.
  //
  // Reproduced synthetically here (no capture pipeline needed): a floored DE
  // that never reports smaller than `eps` even exactly at a splat center —
  // exactly what a DE that has bottomed out at its convergence floor does —
  // observed through an r0 finer than that eps.
  const eps = 5e-3;
  const de = () => 4.9e-3; // a floored DE: bottoms out just under eps
  const points = {
    count: 4,
    pos: F([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]),
    normal: F([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]),
    albedo: F(new Array(12).fill(0.5)),
  };
  const r0 = 4e-4; // finer than eps — a reachable computeR0 output on a tight crop
  const frame = { radius: 1, epsFloor: eps }; // captureEps(frame) = eps here

  // Premise: the RAW metric (as #535/#536 originally reported) misreads a
  // fully-converged floored DE as off-surface, because r0 alone is too tight.
  const raw = metricOnSurface(de, points, r0);
  assert.equal(raw, 0, `premise: unfloored r0 misreports (${raw})`);

  // splatMetrics (the reduce/metrics layer this issue names) floors the
  // metric's OWN r0 reference at the frame's eps before comparing — capture
  // behavior (computeR0/reducePoints) is not called here and is untouched.
  const m = splatMetrics({
    de,
    albedoAt: () => [0, 0, 0],
    aoScale: () => 1,
    sample: null,
    points,
    r0,
    frame,
  });
  assert.equal(
    m.onSurface,
    1,
    `#536: r0 floored against captureEps restores a verifiable onSurface (${m.onSurface})`,
  );
});

test("metricCoverage / metricOverdraw: Euclidean radius cover + mean overlap count", () => {
  // Two splats at each of ±1 (doubled), radius 0.5.
  const points = {
    count: 4,
    pos: F([-1, 0, 0, -1, 0, 0, 1, 0, 0, 1, 0, 0]),
    normal: F([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]),
    albedo: F(new Array(12).fill(0.5)),
    radius: F([0.5, 0.5, 0.5, 0.5]),
  };
  // hits exactly on the two splat clusters ⇒ every hit covered by 2
  const onCluster = { count: 2, pos: F([-1, 0, 0, 1, 0, 0]) };
  assert.ok(close(metricCoverage(onCluster, points), 1.0), "all hits covered");
  assert.ok(
    close(metricOverdraw(onCluster, points), 2.0),
    "doubled cloud ⇒ overdraw 2",
  );
  // add two far hits ⇒ half covered, overdraw (2+2+0+0)/4
  const mixed = { count: 4, pos: F([-1, 0, 0, 1, 0, 0, -5, 0, 0, 5, 0, 0]) };
  assert.ok(
    close(metricCoverage(mixed, points), 0.5),
    "half the hits fall in holes",
  );
  assert.ok(
    close(metricOverdraw(mixed, points), 1.0),
    "mean overlap over all hits",
  );
});

test("metricNormalAgreement: outward normals on a sphere align with ∇DE (≈1); flipped ≈ −1", () => {
  const de = (x, y, z) => Math.hypot(x, y, z) - 1;
  const frame = { radius: 1.5 };
  const out = {
    count: 2,
    pos: F([1, 0, 0, 0, 1, 0]),
    normal: F([1, 0, 0, 0, 1, 0]), // outward = ∇DE direction
    albedo: F([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
  };
  assert.ok(
    metricNormalAgreement(de, out, frame) > 0.999,
    "aligned normals ⇒ ≈ +1",
  );
  const flipped = { ...out, normal: F([-1, 0, 0, 0, -1, 0]) };
  assert.ok(
    metricNormalAgreement(de, flipped, frame) < -0.999,
    "flipped ⇒ ≈ −1",
  );
});

test("metricColorDrift: max-channel |stored − reeval| (AO reproduced when aoStrength>0)", () => {
  const de = () => 0;
  const frame = { radius: 1 };
  const albedoAt = () => [0.5, 0.6, 0.7];
  const noAo = () => 1;
  const match = {
    count: 1,
    pos: F([0, 0, 0]),
    normal: F([0, 0, 1]),
    albedo: F([0.5, 0.6, 0.7]),
    radius: F([0.1]),
  };
  // tol 1e-6: albedo is stored Float32 but reeval is f64 (SH0 chain is f64).
  assert.ok(
    close(metricColorDrift(de, albedoAt, noAo, match, frame, 0), 0, 1e-6),
    "exact match ⇒ 0 drift",
  );
  const off = { ...match, albedo: F([0.4, 0.6, 0.7]) };
  assert.ok(
    close(metricColorDrift(de, albedoAt, noAo, off, frame, 0), 0.1, 1e-6),
    "max channel Δ = 0.1",
  );
  // aoStrength>0 ⇒ reeval multiplied by aoScale (here 0.5) before compare
  const halfAo = () => 0.5;
  const baked = { ...match, albedo: F([0.25, 0.3, 0.35]) }; // = albedoAt·0.5
  assert.ok(
    close(metricColorDrift(de, albedoAt, halfAo, baked, frame, 0.5), 0, 1e-6),
    "AO reproduced ⇒ measures drift not the baked darkening",
  );
});

// ── The S2 reference pins — the falsifiability anchor (SPLAT_GAP_IMPL §1.3d) ──
// Full captureSplats at fixed CI-sized settings, defaultColoring (reproducibility
// over representativeness — colorDrift is the one coloring-dependent metric).
// Deterministic (seeded reservoir + no Math.random), verified reproducing across
// runs. Later phases MUST move a number or explain why not.
//
// Per-metric tolerance (R-6): the bounded [0,1]/[-1,1] metrics hold to 2e-3 across
// engines, but `overdraw` is an unbounded mean overlap COUNT (~4-5) whose value hinges
// on borderline points flipping in/out of the neighbor radius — the most sensitive to
// cross-engine Math.hypot/pow rounding (a Linux CI runner drifts ~3.4e-3 vs the macOS
// pin). It gets 2e-2 (still ~50× tighter than a real regression, which moves overdraw
// by whole units — the P2 aniso bug pushed Carved Cube 4.0→20.7).
const TOL = {
  onSurface: 2e-3,
  coverage: 2e-3,
  overdraw: 2e-2,
  normalAgreement: 2e-3,
  colorDrift: 2e-3,
};
const PIN_OPTS = {
  views: 12,
  res: 64,
  cap: 40_000,
  layers: 2,
  aoStrength: 0.5,
  sampleHits: 120_000,
};
// CAPTURE_VOLUME_SHAPES re-pin: the capture now samples the
// frame's VOLUME — support-sized per-view windows plus an inside() test — where
// before it marched a constant ±radius window and kept every hit, giving a
// union-of-slabs blob reaching ~1.74-2.06·radius. Clipping to the box drops the
// out-of-volume overshoot (~20% of raw hits on Carved Cube), so the surviving
// cloud's local density changes and `overdraw` (a mean overlap COUNT) shifts
// with it. What moved, and by how little: overdraw Mandelbulb 4.8508 -> 4.8579,
// Carved Cube 3.9764 -> 4.0945, Gnarl Dunes 4.3703 -> 4.3755; normalAgreement
// Mandelbulb 0.7958 -> 0.7771 and Gnarl Dunes 0.7428 -> 0.7615; colorDrift
// Mandelbulb 0.0811 -> 0.0875. Both directions,
// on both metrics — a redistribution of where splats sit, not a systematic loss.
// onSurface and coverage do NOT move (0.999+/1.0 throughout), and those are the
// pins a real regression trips — it moves overdraw by whole UNITS (the P2 aniso
// bug put Carved Cube at 20.7).
//
// An earlier draft padded ext per-axis by the 1.10 that radius carries. It is
// NOT in the tree: it cost quality on exactly these fixtures (Gnarl Dunes
// normalAgreement 0.7615 -> 0.7241) because the extra shell is sparse outer
// structure that coarsens the grid for the geometry that matters.
// #450 (ray-clip to the capture volume) moved exactly ONE number here: Gnarl
// Dunes normalAgreement 0.7615 -> 0.7816, and UP. Clipping stops a ray from
// spending its march budget crossing material outside the volume, so fewer rays
// die mid-crossing and record a grazing, badly-normalled hit. Every other pin on
// every fixture is unchanged (these are whole-object frames — a uniform box the
// rays mostly start outside — so the clip has little left to remove; the volumes
// it rescues are the user-drawn ones buried inside geometry).
const PINS = {
  Mandelbulb: {
    onSurface: 0.9989,
    coverage: 1.0,
    overdraw: 4.8579, // re-pinned: CAPTURE_VOLUME_SHAPES box clip (see note above PINS)
    // re-pinned 2026-07-22: the probe moved +h off-surface along the splat
    // normal (S-2 snapped centers straddled thin walls exactly at the zero set)
    normalAgreement: 0.7771, // re-pinned: CAPTURE_VOLUME_SHAPES box clip (see note above PINS)
    colorDrift: 0.0875, // re-pinned: CAPTURE_VOLUME_SHAPES box clip
  },
  "Carved Cube": {
    onSurface: 1.0,
    coverage: 1.0,
    overdraw: 4.0945, // re-pinned: CAPTURE_VOLUME_SHAPES box clip (see note above PINS)
    normalAgreement: 0.9978,
    colorDrift: 0.0195,
  },
  // #507 re-pin. Gnarl Dunes is one of only two presets in the library whose DE
  // has a convergence floor above the default eps (measured: it sticks on 11.7%
  // of probe rays at 5.6e-3, so eps 6.5e-4 → 1.68e-2), so it is the one fixture
  // here the floor probe fires on — Mandelbulb (0.0% stuck) and Carved Cube
  // (0.5%) are untouched, byte-identical, and their pins below/above did not
  // move. Marching at an epsilon the DE can actually reach moved three numbers:
  //   normalAgreement 0.7816 → 0.9245  — the point of the fix. This fixture has
  //     been the harness's worst normal score through three prior re-pins
  //     (0.7428 → 0.7615 → 0.7816); it clears 0.92 in one step because the ∇DE
  //     is now sampled on the smooth part of the DE's ramp instead of inside
  //     its noise floor.
  //   colorDrift     0.0457 → 0.0071  — 6.4× better, and the direct #507
  //     symptom: a stored color only drifts from a re-evaluation at the same
  //     point when the signal driving it is noise.
  //   overdraw       4.3755 → 5.9712  — the real cost, and understood: the
  //     captured shell is now eps-thick (1.7e-2) against an r0 pitch of 4.1e-2,
  //     so splats at different depths in it overlap where before the shell was
  //     essentially 2D. onSurface and coverage both stay at 1.0. The shipped
  //     paths already re-solve this (autoRadius defaults ON, and --refine snaps
  //     centers onto the zero set); tightening it in the capture itself wants
  //     its own measurement, not a guess bolted onto a color fix.
  "Gnarl Dunes": {
    onSurface: 1.0,
    coverage: 1.0,
    overdraw: 5.9712, // re-pinned: #507 eps clears the DE's convergence floor
    normalAgreement: 0.9245, // re-pinned: #507 (was 0.7816, 0.7615, 0.7428)
    colorDrift: 0.0071, // re-pinned: #507
  },
};

for (const [name, pin] of Object.entries(PINS)) {
  test(`splatMetrics S2 baseline pin: ${name}`, () => {
    const f = PRESETS.find((p) => p.name === name);
    assert.ok(f, `fixture ${name} present`);
    const r = captureSplats(f, defaultColoring(), PIN_OPTS);
    assert.ok(r && r.sample, "captured with a hit sample");
    const it = r.stats.iters;
    const m = splatMetrics(
      {
        de: makeDE(f, it),
        albedoAt: makePointAlbedo(f, defaultColoring(), it),
        aoScale,
        sample: r.sample,
        points: r.points,
        r0: r.r0,
        frame: r.frame,
      },
      { aoStrength: 0.5 },
    );
    for (const key of [
      "onSurface",
      "coverage",
      "overdraw",
      "normalAgreement",
      "colorDrift",
    ])
      assert.ok(
        Math.abs(m[key] - pin[key]) <= TOL[key],
        `${name}.${key}: got ${m[key].toFixed(4)}, pin ${pin[key]} (Δ ${Math.abs(m[key] - pin[key]).toExponential(2)}, tol ${TOL[key]})`,
      );
  });
}

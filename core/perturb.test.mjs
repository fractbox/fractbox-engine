// Perturbation deep zoom PR-1 — the P0 spike ladder as permanent CI pins
// (PERTURBATION_ZOOM.md §9 P0/P1 as-run; PERTURBATION_ZOOM_IMPL.md PR-1).
//
// The delta kernel is plain arithmetic (no EFTs), so these fround mirrors
// are FAITHFUL to the GPU computation — a green run here is real evidence
// (P1 measured the WGSL kernel within ~2e-6 of this kernel on Metal).
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  ptSupported,
  buildOrbit,
  deltaRun,
  deltaDE,
  truthRun,
  targetToFx,
  fxNudge,
  fxFromF64,
  fxToF64,
  PT_TAU,
  PT_SLOT_F32,
} from "./perturb.js";
import { TOURBILLON, PRESETS } from "./oplist.js";

const byName = (n) => {
  const p = PRESETS.find((x) => x.name === n);
  assert.ok(p, "preset exists: " + n);
  return p;
};
const FIX = [
  TOURBILLON, // addC, expanding, rotations
  byName("Amazing Box"), // addC, negative scale
  byName("Kleinian Drop"), // no addC, contracting, radialInvert (λ̂=0)
];
const ITERS = 30;
// the measured P0 envelope: ≤ iters·2⁻²¹ with heavy straddles; gate at
// iters·2⁻²⁰ so CI never flakes on a lucky margin draw
const GATE = ITERS * 2 ** -20;

function mulberry32(a) {
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// relative error of the kernel's reconstruction vs the exact truth,
// measured in the reference frame (the P0 metric): both sides express the
// sample as frame + δ; compare the δs.
function sampleErr(fix, orbit, T, d0, iters = ITERS) {
  const truth = truthRun(fix, T, d0, iters);
  const res = deltaRun(orbit, d0);
  if (truth.escapedAt >= 0 || res.switched >= 0) return null; // f32-class regime, pinned elsewhere
  const dT = [
    fxToF64(truth.x - orbit.finalZfx[0]),
    fxToF64(truth.y - orbit.finalZfx[1]),
    fxToF64(truth.z - orbit.finalZfx[2]),
  ];
  const dR = res.d; // the frame residual — NEVER pos − finalZ (absorption)
  const mag = Math.max(Math.hypot(...dT), 1e-300);
  return Math.hypot(dR[0] - dT[0], dR[1] - dT[1], dR[2] - dT[2]) / mag;
}

test("ptSupported names only registry ops", () => {
  for (const k of [
    "boxFold",
    "sphereFold",
    "sphereInv",
    "radialInvert",
    "rotateXYZ",
    "cylinderFold",
  ])
    assert.ok(ptSupported(k), k);
  assert.ok(!ptSupported("kaleido") && !ptSupported("mandelbulbPower"));
});

test("fx target helpers: exact nudge far below f64", () => {
  const Tfx = targetToFx([0.31, 0.22, 0.48]);
  const nudged = fxNudge(Tfx, [1e-20, -2e-21, 0]);
  // the nudge is exact in fx even though it vanishes in f64…
  assert.equal(nudged[0] - Tfx[0], fxFromF64(1e-20));
  // …and the f64 mirror is unchanged (this is WHY the target must widen)
  assert.equal(fxToF64(nudged[0]), 0.31);
});

test("packed layout: slot count and trailer", () => {
  const fix = byName("Kleinian Drop");
  const o = buildOrbit(fix, [0.31, 0.22, 0.48], 12);
  assert.equal(o.packed.length, (12 * o.opCount + 1) * PT_SLOT_F32);
  const t = 12 * o.opCount * PT_SLOT_F32;
  assert.equal(o.packed[t], o.finalZ[0]);
  assert.equal(o.packed[t + 3], o.escapeAt);
});

// ── the ladder: 1e-6 → 1e-32, three real presets ─────────────────────────
for (const fix of FIX) {
  test(`ladder — ${fix.name} tracks the exact map to 1e-32 (gate ${GATE.toExponential(1)})`, () => {
    // Expanding stacks legitimately hand shallow rungs to the f32 tail
    // (λ̂^iters·δ₀ > τ — the P0 tables' blank cells): the gate applies
    // where the delta actually tracks, and the deep rungs must track.
    let trackedRungs = 0;
    for (const N of [6, 12, 20, 26, 32]) {
      const dist = 10 ** -N;
      const rand = mulberry32(0xc0ffee + N);
      const T = fix.camera?.target ?? [0.41, 0.33, 0.52];
      const orbit = buildOrbit(fix, T, ITERS);
      let max = 0,
        compared = 0;
      for (let s = 0; s < 24; s++) {
        const mag = dist * 1e-3 * (0.5 + rand());
        let d0 = [rand() - 0.5, rand() - 0.5, rand() - 0.5];
        const nn = Math.hypot(...d0);
        d0 = d0.map((v) => (v / nn) * mag);
        const e = sampleErr(fix, orbit, T, d0);
        if (e === null) {
          // switched/escaped: the f32-class regime — still must be sane
          const r = deltaRun(orbit, d0);
          assert.ok(r.pos.every(Number.isFinite), "finite switched state");
          continue;
        }
        compared++;
        max = Math.max(max, e);
      }
      if (compared >= 12) {
        trackedRungs++;
        assert.ok(
          max <= GATE,
          `rung 1e-${N}: relErr ${max.toExponential(2)} ≤ ${GATE.toExponential(1)}`,
        );
      }
    }
    assert.ok(trackedRungs >= 3, `deep rungs track (${trackedRungs}/5)`);
  });
}

// ── adversarial near-margin camera (the diffabs case-split under fire) ───
test("adversarial margins — reference within 0.3·pixel of a fold plane", () => {
  const fix = byName("Kleinian Drop");
  for (const N of [8, 20, 30]) {
    const dist = 10 ** -N;
    const T = [1 - 0.3 * dist * 1e-3, 0.22, 0.48]; // boxFold plane at x=1
    const orbit = buildOrbit(fix, T, ITERS);
    const rand = mulberry32(0xbeef + N);
    let max = 0,
      compared = 0;
    for (let s = 0; s < 24; s++) {
      const mag = dist * 1e-3 * (0.5 + rand());
      let d0 = [rand() - 0.5, rand() - 0.5, rand() - 0.5];
      const nn = Math.hypot(...d0);
      d0 = d0.map((v) => (v / nn) * mag);
      const e = sampleErr(fix, orbit, T, d0);
      if (e === null) continue;
      compared++;
      max = Math.max(max, e);
    }
    assert.ok(compared >= 12, `1e-${N}: comparable samples (${compared})`);
    assert.ok(
      max <= GATE,
      `1e-${N}: adversarial relErr ${max.toExponential(2)}`,
    );
  }
});

// ── τ-boundary (plan D3): straddling the switchover must be seamless ─────
test("τ boundary — samples across the switchover stay finite and sane", () => {
  const fix = byName("Kleinian Drop");
  const T = fix.camera?.target ?? [0.31, 0.22, 0.48];
  const orbit = buildOrbit(fix, T, ITERS);
  const rand = mulberry32(0x7a7);
  let switchedSeen = 0,
    trackedSeen = 0;
  for (let s = 0; s < 60; s++) {
    // |δ0| swept through τ·O(1): from 0.1·τ to 10·τ
    const mag = PT_TAU * 10 ** (rand() * 2 - 1);
    let d0 = [rand() - 0.5, rand() - 0.5, rand() - 0.5];
    const nn = Math.hypot(...d0);
    d0 = d0.map((v) => (v / nn) * mag);
    const res = deltaRun(orbit, d0);
    const r = Math.hypot(...res.pos);
    assert.ok(Number.isFinite(r) && Number.isFinite(res.w), "finite state");
    assert.ok(r < 2 * 1e6 + 1e3, "bounded by bailout scale");
    const de = deltaDE(orbit, d0);
    assert.ok(Number.isFinite(de) && de >= 0, "finite DE");
    if (res.switched >= 0) switchedSeen++;
    else trackedSeen++;
  }
  assert.ok(
    switchedSeen > 5 && trackedSeen > 5,
    `both regimes exercised (${switchedSeen}/${trackedSeen})`,
  );
});

// ── straddle envelope at high iterations (the measured P0 caveat) ────────
// P0 stressed 60 iterations bailout-free; under production semantics the
// Amazing Box reference bails at iteration 45, so the honest stress is the
// longest bailout-clean run (40) at the rung whose δ growth parks samples
// just under τ — the heavy-straddle band.
test("straddle envelope — Amazing Box @40 iters stays ≤ N·2⁻¹⁹", () => {
  const fix = byName("Amazing Box");
  const T = [0.62, 0.41, 0.53];
  const iters = 40;
  const orbit = buildOrbit(fix, T, iters);
  assert.equal(orbit.escapeAt, iters, "reference is bailout-clean at 40");
  const rand = mulberry32(0x60);
  const dist = 1e-13; // λ_eff^40·pixel ≈ 0.1·τ — tracks, straddle-rich
  let max = 0,
    compared = 0;
  for (let s = 0; s < 24; s++) {
    const mag = dist * 1e-3 * (0.5 + rand());
    let d0 = [rand() - 0.5, rand() - 0.5, rand() - 0.5];
    const nn = Math.hypot(...d0);
    d0 = d0.map((v) => (v / nn) * mag);
    const truth = truthRun(fix, T, d0, iters);
    const res = deltaRun(orbit, d0);
    if (truth.escapedAt >= 0 || res.switched >= 0) continue;
    const dT = [
      fxToF64(truth.x - orbit.finalZfx[0]),
      fxToF64(truth.y - orbit.finalZfx[1]),
      fxToF64(truth.z - orbit.finalZfx[2]),
    ];
    const dR = res.d;
    const e =
      Math.hypot(dR[0] - dT[0], dR[1] - dT[1], dR[2] - dT[2]) /
      Math.max(Math.hypot(...dT), 1e-300);
    compared++;
    max = Math.max(max, e);
  }
  assert.ok(compared >= 8, `comparable samples (${compared})`);
  assert.ok(max <= 40 * 2 ** -19, `straddle envelope ${max.toExponential(2)}`);
});

// ── per-op pins: every v1 op tracks its own exact map ────────────────────
const OP_FIXTURES = [
  { key: "boxFold", values: [1.0] },
  { key: "boxFoldXYZ", values: [1.0, 0.8, 1.2] },
  { key: "surfFold", values: [1.0] },
  { key: "absFold", values: [] },
  { key: "absXYZ", values: [1, 0, 1] },
  { key: "sphereFold", values: [0.5, 1.0] },
  { key: "cylinderFold", values: [0.5, 1.0] },
  { key: "sphereInv", values: [1.2] },
  { key: "radialInvert", values: [0.0, 0.0, 0.5] },
  { key: "scale", values: [2.0] },
  { key: "translate", values: [0.3, -0.2, 0.1] },
  { key: "rotateXY", values: [14.0] },
  { key: "rotateYZ", values: [7.0] },
  { key: "rotateXZ", values: [21.0] },
  { key: "rotateXYZ", values: [11.0, 5.0, 17.0] },
];
for (const op of OP_FIXTURES) {
  test(`per-op pin — ${op.key}`, () => {
    // a bounded companion stack (box + sphere fold cap the radius) so no
    // op under test can walk the orbit past the bailout
    const fix = {
      addC: false,
      iters: 8,
      deOption: 2,
      ops: [
        op,
        { key: "boxFold", values: [1.0] },
        { key: "sphereFold", values: [0.5, 1.0] },
      ],
    };
    const T = [0.37, 0.29, 0.44];
    const orbit = buildOrbit(fix, T, 8);
    const rand = mulberry32(0x0b + op.key.length);
    let max = 0,
      compared = 0;
    for (const N of [8, 20, 28]) {
      const dist = 10 ** -N;
      for (let s = 0; s < 8; s++) {
        let d0 = [rand() - 0.5, rand() - 0.5, rand() - 0.5];
        const nn = Math.hypot(...d0);
        d0 = d0.map((v) => (v / nn) * dist);
        const e = sampleErr(fix, orbit, T, d0, 8);
        if (e === null) continue;
        compared++;
        max = Math.max(max, e);
      }
    }
    assert.ok(compared >= 12, `comparable (${compared})`);
    assert.ok(max <= 8 * 2 ** -20, `${op.key} relErr ${max.toExponential(2)}`);
  });
}

// ── reference-escape semantics: the forced switchover at escapeAt ────────
test("reference escape forces the f32 tail (no NaN, plausible state)", () => {
  // an escaping stack: expander with no bounding fold
  const fix = {
    addC: true,
    iters: 20,
    deOption: 2,
    ops: [{ key: "scale", values: [2.0] }],
  };
  const T = [0.9, 0.7, 0.8];
  const orbit = buildOrbit(fix, T, 20);
  assert.ok(orbit.escapeAt < 20, "reference escaped");
  const res = deltaRun(orbit, [1e-9, -1e-9, 1e-9]);
  assert.ok(Number.isFinite(res.pos[0]) && Number.isFinite(res.w));
});

// ── unsupported op rejection ─────────────────────────────────────────────
test("buildOrbit rejects unsupported ops loudly", () => {
  const fix = { addC: true, iters: 8, ops: [{ key: "kaleido", values: [6] }] };
  assert.throws(() => buildOrbit(fix, [0, 0, 0], 8), /unsupported op/);
});

// ── D11 (impl plan / review blocker B1): the residual seed must not round-
// trip through the absolute eye. cam.basis() exposes roRel = dir·dist
// directly; the classic eye − target subtraction absorbs it at depth.
test("camera basis exposes an absorption-free residual (D11)", async () => {
  const { makeCamera } = await import("./camera.js");
  const cam = makeCamera({
    target: [9.7, 2.1, 1.4],
    dist: 1,
    yawDeg: 35,
    pitchDeg: 22,
  });
  cam.dist = 1e-20; // deep zoom at a high-|T| target — the B1 regime
  const b = cam.basis();
  assert.ok(Array.isArray(b.roRel), "basis exposes roRel");
  const mag = Math.hypot(...b.roRel);
  assert.ok(Math.abs(mag - 1e-20) < 1e-35, "roRel magnitude is exactly dist");
  // the round trip loses EVERYTHING at this depth (the documented ceiling):
  const rt = [
    b.eye[0] - cam.target[0],
    b.eye[1] - cam.target[1],
    b.eye[2] - cam.target[2],
  ];
  const rtErr = Math.hypot(
    rt[0] - b.roRel[0],
    rt[1] - b.roRel[1],
    rt[2] - b.roRel[2],
  );
  assert.ok(
    rtErr / mag > 0.5,
    `eye−target is garbage here (err ${(rtErr / mag).toExponential(1)}× the residual)`,
  );
});

// ── PR-4 pins ────────────────────────────────────────────────────────────
// D7 drift guard: the registry's wgslPt membership and this module's JS
// stepper membership must be the SAME set — ptEligible requires both, so a
// drift would disable the tier loudly, and this pin catches it in CI.
test("wgslPt registry membership === ptSupported membership", async () => {
  const { OPERATORS } = await import("./operators.js");
  const reg = OPERATORS.filter((o) => o.wgslPt)
    .map((o) => o.key)
    .sort();
  const sup = OPERATORS.filter((o) => ptSupported(o.key))
    .map((o) => o.key)
    .sort();
  assert.deepEqual(reg, sup);
});

test("ptEligible: the inversion family is finally deep-eligible", async () => {
  const { ptEligible } = await import("./stability.js");
  const kd = byName("Kleinian Drop");
  assert.ok(
    ptEligible(kd, ptSupported),
    "Kleinian Drop (radialInvert) eligible",
  );
  assert.ok(ptEligible(TOURBILLON, ptSupported), "Tourbillon eligible");
  assert.ok(
    !ptEligible(
      { addC: true, iters: 8, ops: [{ key: "kaleido", values: [6] }] },
      ptSupported,
    ),
    "kaleido not in v1",
  );
  assert.ok(
    !ptEligible({ ...kd, hybrid: { b: { ops: [] } } }, ptSupported),
    "hybrid excluded",
  );
  assert.ok(
    !ptEligible({ ...kd, objects: [{ id: 1 }] }, ptSupported),
    "scenes excluded",
  );
});

// Review M1 grep-gate: the zoom floor is ptMinDist() at every preview clamp
// site — a raw MIN_DIST clamp reappearing would make max depth
// gesture-dependent again.
test("preview.js has no raw MIN_DIST clamps (M1 grep-gate)", async () => {
  const { readFileSync } = await import("node:fs");
  const src = readFileSync(new URL("./preview.js", import.meta.url), "utf8");
  assert.equal(
    (src.match(/Math\.max\(MIN_DIST/g) || []).length,
    0,
    "all preview clamps route through ptMinDist()",
  );
  assert.ok(
    src.includes("cam.zoom(factor, ptMinDist())"),
    "cam.zoom callers pass the floor",
  );
});

// #489 grep-gate: an orthographic render is invariant to camera DISTANCE (the
// definition of the projection), so every zoom entry point (wheel, pinch,
// keyboard +/-, the zoom()/zoomBy() API) must fork to scaling orthoH, not
// fall through to a cam.dist dolly the render would silently ignore — which
// is exactly what shipped: "zoom doesn't work in Top/Side/Front, works fine
// in Perspective." preview.js has no DOM/GPU-free unit-test harness (real
// canvas + WebGPU only), so this pins the source the same way the MIN_DIST
// gate above does — a regression here is a silent behavior loss, not a type
// or lint error.
test("preview.js: every zoomAtCenter path forks on orthoH before touching cam.dist (#489 grep-gate)", async () => {
  const { readFileSync } = await import("node:fs");
  const src = readFileSync(new URL("./preview.js", import.meta.url), "utf8");
  const zoomAtCenterBody = src.slice(
    src.indexOf("const zoomAtCenter = (factor) => {"),
    src.indexOf("const zoomAtCenter = (factor) => {") + 1100,
  );
  assert.match(
    zoomAtCenterBody,
    /if \(orthoH > 0\) \{\s*orthoH = Math\.max\(ptMinDist\(\), orthoH \* braked\);/,
    "zoomAtCenter scales orthoH (not cam.dist) under ortho, floored at ptMinDist()",
  );
  const wheelListenerIdx = src.indexOf('canvas.addEventListener(\n    "wheel"');
  const wheelBody = src.slice(wheelListenerIdx, wheelListenerIdx + 900);
  assert.match(
    wheelBody,
    /if \(orthoH > 0\) \{[\s\S]*?zoomAtCenter\(factor\);\s*return;\s*\}/,
    "the wheel handler forks to zoomAtCenter under ortho BEFORE the perspective-only cursor probe",
  );
});

// #489 round 4: middle-mouse-button drag on the canvas should PAN (usual 3D
// app flow), not orbit. Source-pin (preview.js has no DOM-free harness, same
// convention as the M1 grep-gate above): pin that the pointermove drag
// handler branches to pan on the middle button, and that this new branch
// exits BEFORE the orbit branch's #441 ortho-drop — routing middle-drag to
// pan must NOT drop orthographic (only ORBIT does), which is the whole
// reason the reporter wants it in Top/Side/Front views.
test("preview.js: middle-button drag pans, not orbits (#489)", async () => {
  const { readFileSync } = await import("node:fs");
  const src = readFileSync(new URL("./preview.js", import.meta.url), "utf8");
  assert.ok(
    src.includes('if (e.shiftKey || (e.buttons & 4) !== 0) {'),
    "pan branch triggers on Shift OR the middle mouse button (buttons bit 4)",
  );
});

test("preview.js: middle-button pan exits before the orbit ortho-drop (#489, #441 stays orbit-only)", async () => {
  const { readFileSync } = await import("node:fs");
  const src = readFileSync(new URL("./preview.js", import.meta.url), "utf8");
  const panGuard = 'if (e.shiftKey || (e.buttons & 4) !== 0) {';
  const panStart = src.indexOf(panGuard);
  const orbitStart = src.indexOf('dragMode = "orbit";');
  assert.ok(panStart >= 0 && orbitStart > panStart, "pan branch precedes the orbit branch");
  const panBranch = src.slice(panStart, orbitStart);
  assert.ok(panBranch.includes("return;"), "pan branch returns early, never reaching orbit code");
  assert.ok(
    !panBranch.includes("orthoH = 0"),
    "middle-button/Shift pan never drops orthoH — only the orbit branch below does",
  );
  assert.ok(
    src.slice(orbitStart).includes("orthoH = 0"),
    "the #441 ortho-drop is still present, gated inside the orbit branch",
  );
});

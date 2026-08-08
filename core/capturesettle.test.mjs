// Globals-parity pins for the shared frame settle (EXPORT_P1 PR-A). A mock
// renderer records what settleFrame writes; these assertions pin the payload
// shape for flat/scene/hybrid (+ the live-only morph/colorBlend/kStar seams),
// so a policy/coloring edit that diverges the settle trips CI — the settle is
// the LIVE render path too (true delegation), not just capture.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  settleFrame,
  frameFeaturesFor,
  formulaOpSet,
  formulaLeafIds,
  ensureCpu,
} from "./capturesettle.js";
import { TOURBILLON } from "./oplist.js";
import { defaultColoring } from "./coloring.js";
import { bailoutFor, sceneDeScale } from "./renderpolicy.js";
import { BAILOUT_IFS } from "./limits.js";

// #266: cpu.js (signalRange) now loads lazily behind capturesettle's
// ensureCpu() — every settleFrame caller calls it once first (see its doc
// comment in capturesettle.js). Top-level await so every test below runs
// against the real signalRange, not the load race.
await ensureCpu();

const mock = () => {
  const calls = { globals: [], ops: [], scene: [], hybrid: [], morph: [] };
  return {
    calls,
    writeGlobals: (p) => calls.globals.push(p),
    writeOps: (ops) => (calls.ops.push(ops), ops.length),
    writeScene: (objs) => (calls.scene.push(objs), objs.length),
    writeHybrid: (...a) => calls.hybrid.push(a),
    writeMorph: (...a) => calls.morph.push(a),
  };
};
const CAM = { dist: 3, yawDeg: 0, pitchDeg: 0, fovDeg: 42 };
const Q = { steps: 128, eps: 1e-3, itersOverride: 24 };

test("flat: writeOps + payload basics; capture defaults (no live bag)", () => {
  const r = mock();
  const col = defaultColoring();
  settleFrame(r, TOURBILLON, col, Q, [64, 64], CAM);
  assert.equal(r.calls.ops.length, 1, "flat path wrote ops");
  assert.equal(r.calls.globals.length, 1);
  const p = r.calls.globals[0];
  assert.equal(p.opCount, r.calls.ops[0].length);
  assert.equal(p.iters, 24, "itersOverride wins");
  assert.equal(p.kStar, 0, "no live kStarFor ⇒ 0 (capture overrides after)");
  assert.equal(p.colorBlend, null, "no live colorBlend ⇒ null");
  assert.equal(p.bailout, bailoutFor(TOURBILLON));
  assert.equal(p.cam, CAM);
  assert.ok(p.tNear > 0 && p.tFar > p.tNear);
  assert.ok("sigLo" in p && "sigSpan" in p, "auto-levels range present");
});

test("flat: live bag — effIters, kStarFor(iters), onFlatPayload tap, colorBlend", () => {
  const r = mock();
  const col = defaultColoring();
  let tapped = null;
  const blend = { modeB: 2, t: 0.5 };
  settleFrame(r, TOURBILLON, col, { steps: 64, eps: 1e-3 }, [64, 64], CAM, {
    effIters: 99,
    kStarFor: (iters) => (assert.equal(iters, 99), 7),
    onFlatPayload: (p) => (tapped = p),
    colorBlend: blend,
  });
  const p = r.calls.globals[0];
  assert.equal(p.iters, 99, "no itersOverride ⇒ live effIters");
  assert.equal(p.kStar, 7, "live kStarFor rides the flat payload");
  assert.equal(p.colorBlend, blend);
  assert.equal(tapped, p, "onFlatPayload sees the exact written payload");
});

test("scene: writeScene + the scene overrides (colorBlend null REGARDLESS)", () => {
  const r = mock();
  const col = defaultColoring();
  const scene = {
    name: "s",
    ops: [],
    objects: [
      {
        objType: 1,
        ops: [],
        iters: 4,
        transform: { origin: [0, 0, 0], uscale: 1, rot: [0, 0, 0] },
        combine: 0,
        blendK: 0,
      },
      {
        objType: 2,
        ops: [],
        iters: 4,
        transform: { origin: [1, 0, 0], uscale: 1, rot: [0, 0, 0] },
        combine: 0,
        blendK: 0,
      },
    ],
  };
  settleFrame(r, scene, col, Q, [64, 64], CAM, { colorBlend: { modeB: 2 } });
  assert.equal(r.calls.scene.length, 1);
  const p = r.calls.globals[0];
  assert.equal(p.objectCount, 2);
  assert.equal(p.iters, 0, "per-object iters ride Obj descriptors");
  assert.equal(p.addC, false);
  assert.equal(p.bailout, BAILOUT_IFS);
  assert.equal(p.deOption, 2);
  assert.equal(p.deScale, sceneDeScale(scene.objects));
  assert.equal(p.colorBlend, null, "scenes never crossfade");
});

test("hybrid (legacy 2-slot): writeHybrid(slotOps, counts, addC) + the packed hyb word", () => {
  const r = mock();
  const col = defaultColoring();
  const hyb = {
    name: "h",
    addC: true,
    ops: [{ key: "boxFold", values: [1, 0, 0] }],
    hybrid: {
      b: { ops: [{ key: "scale", values: [2.5, 0, 0] }], addC: false },
      schedule: { a: 2, b: 1 },
    },
  };
  settleFrame(r, hyb, col, Q, [64, 64], CAM);
  assert.equal(r.calls.hybrid.length, 1);
  // writeHybrid now takes (slotOps, counts, addC) — one op-list per slot.
  const [slotOps, counts, addC] = r.calls.hybrid[0];
  assert.equal(slotOps.length, 2, "2 slots (A, B)");
  assert.equal(slotOps[0][0].key, "boxFold"); // slot A
  assert.equal(slotOps[1][0].key, "scale"); // slot B
  assert.deepEqual(counts, [2, 1]);
  assert.deepEqual(addC, [true, false]);
  const p = r.calls.globals[0];
  assert.equal(p.opCount, 0, "hyb word carries the per-slot counts");
  assert.equal(p.addC, false, "superseded by the packed per-slot addC");
  // The globals hybrid payload is the accessor-derived N-slot descriptor packHyb reads.
  assert.deepEqual(p.hybrid, {
    opCounts: [1, 1],
    counts: [2, 1],
    addC: [true, false],
  });
});

test("hybrid (3-slot N-slot shape): every slot rides writeHybrid + the packed word", () => {
  const r = mock();
  const col = defaultColoring();
  const hyb = {
    name: "h3",
    addC: true,
    ops: [{ key: "boxFold", values: [1, 0, 0] }], // slot A
    hybrid: {
      slots: [
        { ops: [{ key: "scale", values: [2.5, 0, 0] }], addC: false }, // slot B
        { ops: [{ key: "sphereFold", values: [0.5, 1, 0] }], addC: true }, // slot C
      ],
      schedule: { counts: [2, 1, 3] },
    },
  };
  settleFrame(r, hyb, col, Q, [64, 64], CAM);
  assert.equal(r.calls.hybrid.length, 1);
  const [slotOps, counts, addC] = r.calls.hybrid[0];
  assert.equal(slotOps.length, 3, "3 slots (A, B, C)");
  assert.equal(slotOps[2][0].key, "sphereFold"); // slot C reaches the renderer
  assert.deepEqual(counts, [2, 1, 3]);
  assert.deepEqual(addC, [true, false, true]);
  const p = r.calls.globals[0];
  assert.deepEqual(p.hybrid, {
    opCounts: [1, 1, 1],
    counts: [2, 1, 3],
    addC: [true, false, true],
  });
});

test("morph is LIVE-only: runs with live.morph on plain formulas, never without", () => {
  const r = mock();
  const col = defaultColoring();
  const b = {
    name: "b",
    ops: [{ key: "scale", values: [2.2, 0, 0] }],
    iters: 9,
  };
  settleFrame(r, TOURBILLON, col, Q, [64, 64], CAM, {
    morph: { f: b, t: 0.3, swell: 0.1 },
  });
  assert.equal(r.calls.morph.length, 1, "live morph path taken");
  assert.equal(r.calls.globals[0].morph.bIters, 9);
  const r2 = mock();
  settleFrame(r2, TOURBILLON, col, Q, [64, 64], CAM); // capture shape: no bag
  assert.equal(r2.calls.morph.length, 0, "no live bag ⇒ flat path");
});

test("#370: debugView defaults OFF (0) and rides the base payload on every path", () => {
  const col = defaultColoring();
  // No live bag (capture / default) ⇒ 0, i.e. byte-identical, overlay off.
  const rFlat0 = mock();
  settleFrame(rFlat0, TOURBILLON, col, Q, [64, 64], CAM);
  assert.equal(
    rFlat0.calls.globals[0].debugView,
    0,
    "no live bag ⇒ overlay off",
  );

  // Flat path threads a live debugView through to writeGlobals.
  const rFlat = mock();
  settleFrame(rFlat, TOURBILLON, col, Q, [64, 64], CAM, { debugView: 2 });
  assert.equal(
    rFlat.calls.globals[0].debugView,
    2,
    "flat: live debugView rides base",
  );

  // Scene path (spread from the SAME base) carries it too.
  const scene = {
    name: "s",
    ops: [],
    objects: [
      {
        objType: 1,
        ops: [],
        iters: 4,
        transform: { origin: [0, 0, 0], uscale: 1, rot: [0, 0, 0] },
        combine: 0,
        blendK: 0,
      },
    ],
  };
  const rScene = mock();
  settleFrame(rScene, scene, col, Q, [64, 64], CAM, { debugView: 3 });
  assert.equal(
    rScene.calls.globals[0].debugView,
    3,
    "scene: live debugView rides base",
  );

  // Hybrid path (also a base spread) carries it.
  const hyb = {
    name: "h",
    ops: [{ key: "boxFold", values: [1, 0, 0] }],
    hybrid: {
      b: { ops: [{ key: "scale", values: [2.5, 0, 0] }] },
      schedule: { a: 1, b: 1 },
    },
  };
  const rHyb = mock();
  settleFrame(rHyb, hyb, col, Q, [64, 64], CAM, { debugView: 1 });
  assert.equal(
    rHyb.calls.globals[0].debugView,
    1,
    "hybrid: live debugView rides base",
  );
});

test("frameFeaturesFor: morph/df64/blend are explicit live params; opSet honors morphF", () => {
  const col = defaultColoring();
  const base = frameFeaturesFor(TOURBILLON, col);
  assert.equal(base.morph, false);
  assert.equal(base.df64, false);
  const b = { name: "b", ops: [{ key: "rotateXYZ", values: [10, 0, 0] }] };
  const live = frameFeaturesFor(TOURBILLON, col, { morphF: b, df64: true });
  assert.equal(live.morph, true);
  assert.equal(live.df64, true);
  assert.ok(
    formulaOpSet(TOURBILLON, b).length > formulaOpSet(TOURBILLON).length,
    "morph B ops join the predicted set only when morphF is passed",
  );
  assert.equal(formulaLeafIds(TOURBILLON), null, "non-scene ⇒ no leaves");
});

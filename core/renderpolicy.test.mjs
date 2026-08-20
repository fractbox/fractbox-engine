// Pins the pure render-policy layer (renderpolicy.js) extracted from
// preview.js — auto-detail iteration law, quality tiers, scene march scale.
// These decisions previously lived in closures with zero coverage; this pins
// their CURRENT behavior so a policy change must be deliberate.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  classifyTier,
  bootPredictMs,
  BOOT_PREDICT_MS_DEFAULT,
  BOOT_PREDICT_MS_FAST,
  shouldRaceGeneralAtBoot,
  bailoutFor,
  REF_DIST,
  TNEAR_K,
  TFAR_K,
  ITER_CEIL,
  ITER_PER_OCTAVE,
  itersForMagnification,
  effectiveIters,
  qualityParams,
  stillQualityParams,
  resolveStillDims,
  bandRect,
  exportSampleCount,
  EXPORT_SAMPLES_FULL,
  EXPORT_SAMPLES_HEAVY,
  STILL_PX_CAP,
  sceneDeScale,
  shadeLight,
  asciiMoveParams,
  ASCII_MOVE_COL_SCALE,
  opEvalsPerIter,
  formulaCostScore,
  isMobileClass,
  entryDetailClamp,
  MOBILE_ENTRY_WORK_BUDGET,
  MOBILE_ENTRY_MIN_DETAIL,
  makeEntryClampArm,
  governorInit,
  governorStep,
  GOV_BUDGET_MS,
  GOV_SCALE_FLOOR,
  GOV_OVER_FRAMES,
  GOV_UNDER_FRAMES,
  GOV_DOWN_MIN,
  magnificationFor,
  classifyDeviceLoss,
  DEVICE_LOST_REASON_MAX,
  readbackBudgetMs,
  READBACK_MS_PER_SUBMIT,
  READBACK_MS_FLOOR,
} from "./renderpolicy.js";
import { PT_MIN_DIST } from "./camera.js";
import { MAX_ITERS, BAILOUT_ESCAPE, BAILOUT_IFS } from "./limits.js";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

test("near/far constants: the old fixed [0.02, 80] at the default framing", () => {
  assert.equal(REF_DIST * TNEAR_K, 0.02);
  assert.equal(REF_DIST * TFAR_K, 80.0);
  assert.equal(ITER_CEIL, MAX_ITERS);
});

test("bailoutFor: escape-time gets the small bailout, IFS the huge one", () => {
  assert.equal(
    bailoutFor({ ops: [{ key: "mandelbulbPower", values: [8] }] }),
    BAILOUT_ESCAPE,
  );
  assert.equal(
    bailoutFor({ ops: [{ key: "boxFold", values: [1] }] }),
    BAILOUT_IFS,
  );
});

// ── effectiveIters (auto-detail §6) ─────────────────────────────────────────

test("effectiveIters: default framing (M≈1) adds nothing", () => {
  assert.equal(effectiveIters(12, { dist: REF_DIST }), 12);
  assert.equal(effectiveIters(12, { dist: REF_DIST * 2 }), 12); // zoomed OUT: no boost
});

test("effectiveIters: ~one extra iteration per zoom octave, clamped to ITER_CEIL", () => {
  assert.equal(effectiveIters(12, { dist: REF_DIST / 2 }), 13); // 1 octave
  assert.equal(effectiveIters(12, { dist: REF_DIST / 16 }), 16); // 4 octaves
  assert.equal(effectiveIters(60, { dist: REF_DIST / 1024 }), ITER_CEIL); // clamped
});

// ── magnificationFor — the scale-free zoom law ───────────────────────────────
// The badge, the auto-detail law and the depth march law all read magnification
// from here. Every one of them used to hand-type its own floor, and a stale
// floor does not fail loudly — it silently STOPS COUNTING, which is how a
// working perturbation descent came to look like a hard pin at ×2.4·10¹³.

test("magnificationFor is exactly REF_DIST/dist in the ordinary range", () => {
  assert.equal(magnificationFor(REF_DIST), 1);
  assert.equal(magnificationFor(REF_DIST / 2), 2);
  assert.equal(magnificationFor(1e-6), REF_DIST / 1e-6);
  assert.equal(magnificationFor(48), 0.5); // zoomed OUT — below ×1, not clamped
});

test("magnificationFor keeps counting through the perturbation range", () => {
  // THE REGRESSION PIN. The old badge floor (1e-12) capped this at
  // REF_DIST/1e-12 = 2.4e13 for EVERY deeper camera — reproduced in-app as a
  // descent that read ×2.4·10¹³ for 13 straight wheel bursts while the true
  // dist fell to 2.95e-27 (×8.1e27).
  const PINNED = REF_DIST / 1e-12;
  for (const dist of [1e-13, 1e-16, 1e-20, 1e-24, 1e-27]) {
    assert.equal(magnificationFor(dist), REF_DIST / dist, `dist=${dist}`);
    assert.ok(magnificationFor(dist) > PINNED, `dist=${dist} still saturating`);
  }
  assert.ok(magnificationFor(1e-20) > 1e20); // the pt tier's advertised range
});

test("magnificationFor floors only at the camera's own absolute clamp", () => {
  // PT_MIN_DIST is the deepest a camera can legally sit (camera.js clampDist),
  // so the law saturates exactly there and nowhere shallower.
  const ceiling = REF_DIST / PT_MIN_DIST;
  assert.equal(magnificationFor(PT_MIN_DIST), ceiling);
  assert.equal(magnificationFor(PT_MIN_DIST / 1000), ceiling);
  // Degenerate input is a division guard, never a wall: 0/negative/non-finite
  // report the ceiling instead of Infinity or NaN reaching the DOM.
  for (const bad of [0, -1, NaN, undefined, null])
    assert.equal(magnificationFor(bad), ceiling, `input=${bad}`);
});

test("the depth-driven laws keep responding below the old 1e-12 floor", () => {
  // #480 fixed these two consumers; they now share the law rather than
  // re-deriving it, so the fix cannot come apart again.
  assert.ok(
    effectiveIters(12, { dist: 1e-20 }) > effectiveIters(12, { dist: 1e-12 }),
    "auto-detail must still rise past the df64-era floor",
  );
});

test("effectiveIters: autoDetail off returns the base verbatim", () => {
  assert.equal(
    effectiveIters(12, { dist: REF_DIST / 16, autoDetail: false }),
    12,
  );
  assert.equal(effectiveIters(0, { autoDetail: false }), 0);
  assert.equal(effectiveIters(undefined, { autoDetail: false }), 0); // base || 0
});

test("effectiveIters: Detail-slider override is absolute and wins (2..ITER_CEIL)", () => {
  assert.equal(
    effectiveIters(12, { dist: REF_DIST / 16, detailOverride: 5 }),
    5,
  );
  assert.equal(effectiveIters(12, { detailOverride: 1 }), 2); // floor
  assert.equal(effectiveIters(12, { detailOverride: 999 }), ITER_CEIL); // ceil
});

// ── qualityParams (#32 tiers + deep zoom §6 depth) ──────────────────────────

const TIGHT = { ops: [{ key: "boxFold", values: [1] }], iters: 12 }; // not loose
const LOOSE = { ops: [{ key: "scale", values: [1.5] }], iters: 12 }; // |scale| < 2

test("qualityParams: settled full — tight vs loose march budgets", () => {
  assert.deepEqual(qualityParams(TIGHT, { dprCap: 2 }), {
    scale: 1.0,
    steps: 200,
    eps: 0.001,
    deScale: 0.5,
  });
  assert.deepEqual(qualityParams(LOOSE, { dprCap: 2 }), {
    scale: 1.0,
    steps: 320,
    eps: 0.001,
    deScale: 0.3,
  });
  // Touch-class devices (dprCap 1) get the smaller step budgets.
  assert.equal(qualityParams(TIGHT, { dprCap: 1 }).steps, 140);
  assert.equal(qualityParams(LOOSE, { dprCap: 1 }).steps, 220);
});

test("qualityParams: deep zoom raises steps and tightens deScale (single-object only)", () => {
  const deep = qualityParams(TIGHT, { dprCap: 2, dist: REF_DIST / 1e4 }); // M = 10⁴
  const depth = Math.min(1 + 4 * 0.35, 3); // 2.4
  assert.equal(deep.steps, Math.round(200 * depth));
  assert.equal(deep.deScale, Math.max(0.25, 0.5 / Math.sqrt(depth)));
  // DEPTH_CAP caps the boost; STEP_CEIL caps the steps.
  const deepest = qualityParams(LOOSE, { dprCap: 2, dist: REF_DIST / 1e9 });
  assert.equal(deepest.steps, 512); // min(STEP_CEIL, 320·3)
  // CSG scenes are not depth-boosted (sceneDeScale is their knob). Scenes
  // always carry ops: [] post-sanitize.
  const scene = { ops: [], objects: [{ objType: 1 }] };
  assert.equal(
    qualityParams(scene, { dprCap: 2, dist: REF_DIST / 1e4 }).steps,
    200,
  );
});

test("qualityParams: interactive tiers — full / balanced / smooth", () => {
  const full = qualityParams(TIGHT, { dprCap: 2 });
  assert.deepEqual(
    qualityParams(TIGHT, { quality: "low", moveQuality: "full", dprCap: 2 }),
    full,
  );
  const bal = qualityParams(TIGHT, {
    quality: "low",
    moveQuality: "balanced",
    dprCap: 2,
  });
  assert.deepEqual(bal, { ...full, scale: 0.8, cheap: true }); // full march budget, lower res
  const balTouch = qualityParams(TIGHT, {
    quality: "low",
    isTouch: true,
    dprCap: 2,
  });
  assert.equal(balTouch.scale, 0.7);
  assert.deepEqual(
    qualityParams(TIGHT, { quality: "low", moveQuality: "smooth", dprCap: 2 }),
    {
      scale: 0.7,
      steps: 48,
      eps: 0.003,
      deScale: 0.65,
      cheap: true,
    },
  );
  assert.equal(
    qualityParams(TIGHT, {
      quality: "low",
      moveQuality: "smooth",
      isTouch: true,
      dprCap: 2,
    }).scale,
    0.6,
  );
});

// ── sceneDeScale (CSG march scale) ──────────────────────────────────────────

test("sceneDeScale: pure-union primitive/tight scenes keep 0.5", () => {
  assert.equal(
    sceneDeScale([
      { objType: 1, combine: 0 },
      { objType: 2, combine: 0 },
    ]),
    0.5,
  );
  assert.equal(sceneDeScale([{ objType: 0, ops: TIGHT.ops, combine: 0 }]), 0.5);
});

test("sceneDeScale: a loose IFS child tightens the whole scene to 0.3", () => {
  assert.equal(
    sceneDeScale([{ objType: 1 }, { objType: 0, ops: LOOSE.ops }]),
    0.3,
  );
  // …but a MUTED loose op must not vouch for a bound it isn't providing.
  const muted = [{ key: "scale", values: [1.5], muted: true }, ...TIGHT.ops];
  assert.equal(sceneDeScale([{ objType: 0, ops: muted }]), 0.5);
  // …and a primitive carrying stray ops stays exact.
  assert.equal(sceneDeScale([{ objType: 2, ops: LOOSE.ops }]), 0.5);
});

test("sceneDeScale: any carving object (subtract/intersect) forces 0.25", () => {
  assert.equal(
    sceneDeScale([{ objType: 1 }, { objType: 2, combine: 2 }]),
    0.25,
  );
  assert.equal(
    sceneDeScale([{ objType: 1 }, { objType: 2, combine: 3 }]),
    0.25,
  );
  assert.equal(
    sceneDeScale([{ objType: 1 }, { objType: 2, combineType: 2 }]),
    0.25,
  ); // legacy field
  assert.equal(sceneDeScale([{ objType: 1 }, { objType: 2, combine: 1 }]), 0.5); // smooth-union is safe
});

// ── shadeLight (P1 cheap-tier shading) ──────────────────────────────────────

test("shadeLight: cheap tier zeroes shadow+AO; settled passes the light through", () => {
  const light = { dir: [0, 0, 1], shadow: 0.5, ao: 0.55, fill: 0.2 };
  assert.equal(shadeLight({ cheap: false }, light), light); // same object, untouched
  assert.equal(shadeLight(undefined, light), light);
  assert.deepEqual(shadeLight({ cheap: true }, light), {
    ...light,
    shadow: 0,
    ao: 0,
  });
});

// ── D0 leaf-aware sceneDeScale (PRIMITIVE_DIFS_D0 §2.7) ─────────────────────

test("sceneDeScale: a mixed object (op chain + leaf) is as loose as its chain", () => {
  // Loose chain finalized by a torus leaf → the chain's 0.3 governs.
  assert.equal(
    sceneDeScale([
      { shapeId: 3, shapeParams: [1, 0.25, 0, 0], ops: LOOSE.ops, iters: 12 },
    ]),
    0.3,
  );
  // Tight chain + leaf → exact 0.5; iterShape adds nothing.
  assert.equal(
    sceneDeScale([
      {
        shapeId: 1,
        shapeParams: [1, 0, 0, 0],
        ops: TIGHT.ops,
        iters: 12,
        iterShape: true,
      },
    ]),
    0.5,
  );
});

test("sceneDeScale: a deApprox LEAF tightens ×0.5, composing with the carve clamp", async () => {
  const { LEAVES } = await import("./leaves.js");
  // Synthetic D2-style approximate leaf (registry scans are live — see leaves.js).
  LEAVES.push({
    id: 99,
    key: "_testApprox",
    name: "T",
    deApprox: true,
    params: [{ name: "p", def: 1, min: 0.05, max: 4, step: 0.01 }],
    wgsl: "return length(p) - prm.x;",
    glsl: "return length(p) - prm.x;",
  });
  try {
    assert.equal(
      sceneDeScale([{ shapeId: 99, shapeParams: [1, 0, 0, 0] }]),
      0.25,
    ); // 0.5 × 0.5
    assert.equal(
      sceneDeScale([
        { objType: 1 },
        { shapeId: 99, shapeParams: [1, 0, 0, 0], combine: 2 },
      ]),
      0.125, // carve 0.25 × approx 0.5
    );
  } finally {
    LEAVES.pop();
  }
});

test("sceneDeScale: an approx OP in a mixed object still tightens ×0.5", () => {
  const ops = [{ key: "polygonFold", values: [6, 1, 0] }, ...TIGHT.ops];
  assert.equal(
    sceneDeScale([{ shapeId: 2, shapeParams: [0.5, 0, 0, 0], ops, iters: 12 }]),
    0.25,
  );
});

// ── Interactive pixel budget (#212) ─────────────────────────────────────────
// The interactive tiers cap their render scale so scale²·devicePx stays under
// INTERACT_PX_BUDGET — a 5K display interacts at laptop cost. Settled frames
// and the explicit moveQuality 'full' pledge are never clamped.
test("qualityParams: interactive scale honors the pixel budget on huge canvases", () => {
  const px5k = 5120 * 2660;
  const q = qualityParams(TIGHT, { quality: "low", dprCap: 2, devicePx: px5k });
  assert.ok(q.scale < 0.8, "balanced tier clamps below 0.8 at 5K");
  const rendered = q.scale * q.scale * px5k;
  assert.ok(
    rendered <= 4.3e6,
    `rendered px ${Math.round(rendered)} within budget`,
  );
  // Laptop-class canvas: 0.8× of 2560×1600 is under budget → unchanged.
  const lap = qualityParams(TIGHT, {
    quality: "low",
    dprCap: 2,
    devicePx: 2560 * 1600,
  });
  assert.equal(lap.scale, 0.8);
  // Settled frames are never clamped (banding bounds their dispatch instead).
  const full = qualityParams(TIGHT, {
    quality: "full",
    dprCap: 2,
    devicePx: px5k,
  });
  assert.equal(full.scale, 1.0);
  // devicePx absent → exactly the old behavior.
  assert.equal(qualityParams(TIGHT, { quality: "low", dprCap: 2 }).scale, 0.8);
});

test("qualityParams: interactive time budget clamps heavy-formula scale", () => {
  const laptop = 2560 * 1600;
  // Heavy scene (predicted 1400 ms settled): balanced tier drops resolution
  // until the predicted tier cost sits at the budget.
  const q = qualityParams(TIGHT, {
    quality: "low",
    dprCap: 2,
    devicePx: laptop,
    predictedFullMs: 1400,
  });
  assert.ok(q.scale < 0.4, `heavy clamps hard (got ${q.scale})`);
  assert.ok(q.scale >= 0.25, "never below the legibility floor");
  // Cheap scene (60 ms): under budget at 0.8× — untouched.
  const cheap = qualityParams(TIGHT, {
    quality: "low",
    dprCap: 2,
    devicePx: laptop,
    predictedFullMs: 60,
  });
  assert.equal(cheap.scale, 0.8);
  // The explicit moveQuality 'full' pledge is never clamped.
  const pledge = qualityParams(TIGHT, {
    quality: "low",
    moveQuality: "full",
    dprCap: 2,
    devicePx: laptop,
    predictedFullMs: 1400,
  });
  assert.equal(pledge.scale, 1.0);
});

// ── Unbounded scenes (grazing-ray starvation fix) ────────────────────────────
test("sceneUnbounded folds the combine chain like the marcher", async () => {
  const { sceneUnbounded, unboundedScene } = await import("./renderpolicy.js");
  const gnarl = {
    shapeId: 39,
    shapeParams: [0.3, 3, 0.35, 4],
    ops: [],
    iters: 1,
    transform: {},
  };
  const ball = { objType: 2, primParam: 2.1, ops: [], iters: 1, transform: {} };
  // union with a bounded ball stays unbounded; intersect bounds it.
  assert.equal(sceneUnbounded([gnarl, { ...ball, combine: 0 }]), true);
  assert.equal(sceneUnbounded([gnarl, { ...ball, combine: 3 }]), false);
  // subtracting a bounded ball keeps the field's extent.
  assert.equal(sceneUnbounded([gnarl, { ...ball, combine: 2 }]), true);
  // intersecting two unbounded fields is still unbounded.
  assert.equal(sceneUnbounded([gnarl, { ...gnarl, combine: 3 }]), true);
  // a muted unbounded object doesn't render, so it doesn't widen the scene.
  assert.equal(
    sceneUnbounded([ball, { ...gnarl, muted: true, combine: 0 }]),
    false,
  );
  assert.equal(unboundedScene({ objects: [gnarl] }), true);
  assert.equal(unboundedScene({ ops: [] }), false);
});

test("qualityParams doubles the march budget for an unbounded scene", async () => {
  const { qualityParams } = await import("./renderpolicy.js");
  const gnarl = {
    shapeId: 39,
    shapeParams: [0.3, 3, 0.35, 4],
    ops: [],
    iters: 1,
    transform: {},
  };
  const boundedF = {
    ops: [],
    iters: 8,
    objects: [
      gnarl,
      {
        objType: 2,
        primParam: 2,
        ops: [],
        iters: 1,
        transform: {},
        combine: 3,
      },
    ],
  };
  const unboundedF = { ops: [], iters: 8, objects: [gnarl] };
  const bounded = qualityParams(boundedF, "high", { dprCap: 2 });
  const unbounded = qualityParams(unboundedF, "high", { dprCap: 2 });
  assert.equal(unbounded.steps, 512); // settled tier goes to the ceiling
});

// ── Offline still / export budget matches the settled view (#281/#282/#283) ──
// The saved PNG (preview.stillBlob) and exported frames (captureFrame) must
// march at the on-screen SETTLED quality, or "saved == what you see" breaks.
// The regression these pin: the export paths substituted a TIGHTER eps (0.0006)
// while keeping the settled step budget, which starved the grazing silhouette
// rays of the deApprox bounded TPMS leaves and eroded the saved shape inward
// (gyroid "clipping"; lidinoid/scherk "different shape"). stillQualityParams
// now sources steps, eps AND deScale from the settled policy — one truth.
const leafScene = (shapeId, shapeParams) => ({
  ops: [],
  iters: 8,
  objects: [
    {
      objType: 0,
      shapeId,
      shapeParams,
      ops: [],
      iters: 1,
      transform: {},
      combine: 0,
    },
  ],
});
// gyroid 7, schwarzP 8, lidinoid 9, scherk 10 — all deApprox bounded leaves.
const TPMS = [
  ["gyroid", leafScene(7, [3, 0.06, 0, 1.4])],
  ["schwarzP", leafScene(8, [3, 0, 0.06, 1.4])],
  ["lidinoid", leafScene(9, [3, 0, 0.05, 1.4])],
  ["scherk", leafScene(10, [2.5, 0.04, 1.4])],
];

test("stillQualityParams: still eps == the settled view's eps (no tighter offline eps)", () => {
  for (const [name, f] of TPMS) {
    const settled = qualityParams(f, { quality: "full", dprCap: 2 });
    const still = stillQualityParams(f);
    // The exact divergence #281/#282/#283: eps MUST equal the settled eps, and
    // MUST NOT be the old hand-picked tighter literal that eroded the silhouette.
    assert.equal(
      still.eps,
      settled.eps,
      `${name}: still eps must match settled`,
    );
    assert.notEqual(
      still.eps,
      0.0006,
      `${name}: must not use the old tighter still eps`,
    );
    assert.ok(
      still.eps >= 0.001,
      `${name}: settled eps is the looser 0.001, not tighter`,
    );
  }
});

test("stillQualityParams: steps + deScale also track the settled policy", () => {
  for (const [name, f] of TPMS) {
    const settled = qualityParams(f, { quality: "full", dprCap: 2 });
    const still = stillQualityParams(f);
    assert.equal(
      still.steps,
      settled.steps,
      `${name}: still steps match settled`,
    );
    assert.equal(
      still.deScale,
      settled.deScale,
      `${name}: still deScale matches settled`,
    );
  }
});

test("stillQualityParams: deep-zoom camera depth carries into the still budget", () => {
  // A closer camera (deep zoom) boosts the settled budget; the still must honor
  // the same boost, else the saved PNG under-marches versus the zoomed screen.
  const f = { ops: [{ key: "boxFold", values: [1] }], iters: 10 };
  const shallow = stillQualityParams(f, REF_DIST);
  const deep = stillQualityParams(f, REF_DIST / 1000);
  assert.ok(
    deep.steps >= shallow.steps,
    "deeper zoom marches at least as many steps",
  );
});

// ── bandRect (banded march band geometry, #514) ──────────────────────────────
// The regression fence #514 never had: its banded-vs-single-dispatch identity
// check was run at VIEWPORT dims only, so nothing pinned band geometry against
// the EXPLICIT export heights the #513 size picker introduced. A band grid
// derived from the live canvas height instead of the render target's would
// leave strips of an explicit-size export un-marched — exactly the "off-centre
// / missing content" failure mode. These pin the tiling at the real export
// heights.
const tileCovers = (n, h) => {
  const rects = Array.from({ length: n }, (_, i) => bandRect(i, n, h));
  // exact tiling: contiguous, no gap, no overlap, closes on h
  assert.equal(rects[0].y0, 0, "first band starts at row 0");
  assert.equal(rects[n - 1].y1, h, "last band closes on the full height");
  for (let i = 1; i < n; i++) {
    assert.equal(rects[i].y0, rects[i - 1].y1, `band ${i} abuts band ${i - 1}`);
  }
  const rows = rects.reduce((s, r) => s + r.h, 0);
  assert.equal(rows, h, "the bands cover exactly h rows in total");
  return rects;
};

test("bandRect: bands tile the EXPORT height exactly at every preset size", () => {
  // The heights the EXPORT_SIZE presets actually render at (4K / Square /
  // Portrait / 1080p), across the band counts settleBands() can pick.
  for (const h of [2160, 2048, 1920, 1080]) {
    for (const n of [1, 2, 3, 7, 24, 64]) tileCovers(n, h);
  }
});

test("bandRect: tiles heights that do NOT divide evenly (no sliver left behind)", () => {
  // 2160/64 = 33.75 and 2048/24 = 85.33 — the rounding cases where a naive
  // floor on the last band would drop rows off the bottom of the frame.
  for (const [n, h] of [
    [64, 2160],
    [24, 2048],
    [7, 1920],
    [13, 1117],
    [64, 1],
  ]) {
    tileCovers(n, h);
  }
});

test("bandRect: band geometry follows the height it is GIVEN, not any other", () => {
  // The whole point of the fence: the same band index over an export-sized
  // target must span export rows, not live-canvas rows. A 4K export banded
  // over the on-screen 1800-row canvas would stop 360 rows short.
  const exportH = 2160;
  const liveH = 1800;
  assert.equal(bandRect(63, 64, exportH).y1, exportH);
  assert.notEqual(bandRect(63, 64, liveH).y1, exportH);
  // mid-band offsets scale with the target height too
  assert.equal(bandRect(32, 64, exportH).y0, Math.floor((2160 * 32) / 64));
});

test("bandRect: degenerate slices come back empty for the caller to skip", () => {
  // n > h: most bands are zero-height; the renderer skips them rather than
  // issuing an empty scissor (a 0-height scissor rect is a WebGPU error).
  const rects = Array.from({ length: 8 }, (_, i) => bandRect(i, 8, 3));
  assert.equal(
    rects.filter((r) => r.h > 0).length,
    3,
    "only as many non-empty bands as there are rows",
  );
  assert.equal(
    rects.reduce((s, r) => s + r.h, 0),
    3,
  );
  // a zero-height target yields nothing at all
  assert.deepEqual(bandRect(0, 4, 0), { y0: 0, y1: 0, h: 0 });
});

test("bandRect: clamps a garbage band index/count instead of scissoring off-target", () => {
  assert.deepEqual(bandRect(-3, 4, 2160), bandRect(0, 4, 2160));
  assert.deepEqual(bandRect(99, 4, 2160), bandRect(3, 4, 2160));
  assert.deepEqual(bandRect(0, 0, 2160), { y0: 0, y1: 2160, h: 2160 });
});

// ── resolveStillDims (EXPORT_SIZE picker — deterministic PNG export size) ─────
test("resolveStillDims: no override reproduces the legacy rect × dpr path exactly", () => {
  // Byte-for-byte the pre-picker computation: round(rect*dpr), long-edge clamp.
  assert.deepEqual(resolveStillDims({ rectW: 900, rectH: 700, dpr: 2 }), {
    W: 1800,
    H: 1400,
  });
  // The rect fallbacks (4 / 3) when a headless/zero rect is passed.
  assert.deepEqual(resolveStillDims({ rectW: 0, rectH: 0, dpr: 1 }), {
    W: 4,
    H: 3,
  });
});

test("resolveStillDims: no override clamps the LONG EDGE to the cap (aspect kept)", () => {
  // 5000×3000 @2× = 10000×6000; long edge 10000 → ×0.4096 → 4096×2458.
  assert.deepEqual(resolveStillDims({ rectW: 5000, rectH: 3000, dpr: 2 }), {
    W: 4096,
    H: 2458,
  });
});

test("resolveStillDims: an explicit {width,height} override wins over the rect", () => {
  // Preset dims flow straight through when under the cap, ignoring rect/dpr.
  assert.deepEqual(
    resolveStillDims({
      width: 3840,
      height: 2160,
      rectW: 900,
      rectH: 700,
      dpr: 2,
    }),
    { W: 3840, H: 2160 },
  );
  assert.deepEqual(resolveStillDims({ width: 1080, height: 1920 }), {
    W: 1080,
    H: 1920,
  });
});

test("resolveStillDims: override clamps EACH dimension independently (square 4096 legal)", () => {
  // Per-DIMENSION clamp, not long-edge: a 4096×4096 square is legal.
  assert.deepEqual(resolveStillDims({ width: 4096, height: 4096 }), {
    W: 4096,
    H: 4096,
  });
  // Over-cap in one or both dimensions clamps only the offender(s).
  assert.deepEqual(resolveStillDims({ width: 8000, height: 8000 }), {
    W: STILL_PX_CAP,
    H: STILL_PX_CAP,
  });
  assert.deepEqual(resolveStillDims({ width: 9000, height: 1080 }), {
    W: STILL_PX_CAP,
    H: 1080,
  });
  // Non-integers round.
  assert.deepEqual(resolveStillDims({ width: 1920.6, height: 1080.4 }), {
    W: 1921,
    H: 1080,
  });
});

test("resolveStillDims: a lone width/height (not both) falls back to the rect path", () => {
  // The override requires BOTH dimensions together; one alone is ignored.
  const rect = { rectW: 900, rectH: 700, dpr: 2 };
  assert.deepEqual(resolveStillDims({ width: 3840, ...rect }), {
    W: 1800,
    H: 1400,
  });
  assert.deepEqual(resolveStillDims({ height: 2160, ...rect }), {
    W: 1800,
    H: 1400,
  });
});

test("STILL_PX_CAP is the documented 4096 ceiling", () => {
  assert.equal(STILL_PX_CAP, 4096);
});

test("preview.js stillBlob threads opts.{width,height} through resolveStillDims", () => {
  // Source-level pin (CI has no GPU — stillBlob can't run): the export path must
  // route dims through the pure resolver AND pass the picker's override in, so
  // the no-override path stays byte-identical and the override is actually honored.
  const src = readFileSync(
    fileURLToPath(new URL("./preview.js", import.meta.url)),
    "utf8",
  );
  assert.match(src, /resolveStillDims\(/, "stillBlob calls resolveStillDims");
  assert.match(
    src,
    /width:\s*opts\.width/,
    "opts.width is threaded into resolveStillDims",
  );
  assert.match(
    src,
    /height:\s*opts\.height/,
    "opts.height is threaded into resolveStillDims",
  );
  assert.doesNotMatch(
    src,
    /STILL_LONG_EDGE_CAP/,
    "the inline long-edge clamp moved into resolveStillDims",
  );
});

// ── exportSampleCount (#save-latency — adaptive export AA) ───────────────────
test("exportSampleCount: full decision table (heavy/light × adaptive/full)", () => {
  // Adaptive (default): heavy → the reduced count; light → the full count.
  assert.equal(
    exportSampleCount({ heavy: true, mode: "adaptive" }),
    EXPORT_SAMPLES_HEAVY,
  );
  assert.equal(
    exportSampleCount({ heavy: false, mode: "adaptive" }),
    EXPORT_SAMPLES_FULL,
  );
  // Full: always the full count regardless of weight.
  assert.equal(
    exportSampleCount({ heavy: true, mode: "full" }),
    EXPORT_SAMPLES_FULL,
  );
  assert.equal(
    exportSampleCount({ heavy: false, mode: "full" }),
    EXPORT_SAMPLES_FULL,
  );
});

test("exportSampleCount: defaults to adaptive, and light-adaptive == full (no change)", () => {
  // No mode given → adaptive; no heavy given → light. So the zero-arg call and a
  // light-adaptive call both yield the full count — a light save is untouched.
  assert.equal(exportSampleCount(), EXPORT_SAMPLES_FULL);
  assert.equal(exportSampleCount({}), EXPORT_SAMPLES_FULL);
  assert.equal(
    exportSampleCount({ heavy: false }),
    exportSampleCount({ heavy: false, mode: "full" }),
    "light formulas render identically in adaptive and full modes",
  );
});

test("exportSampleCount: the reduced count is a real, ~3× speedup and stays legible", () => {
  // The whole point: heavy-adaptive is materially fewer samples than full (the
  // ~3× wall win) but not so few the AA collapses.
  assert.equal(EXPORT_SAMPLES_FULL, 24);
  assert.equal(EXPORT_SAMPLES_HEAVY, 8);
  assert.ok(
    EXPORT_SAMPLES_HEAVY >= 4 && EXPORT_SAMPLES_HEAVY < EXPORT_SAMPLES_FULL,
    "heavy count is a real reduction that keeps usable AA",
  );
  assert.ok(
    EXPORT_SAMPLES_FULL / EXPORT_SAMPLES_HEAVY >= 2.5,
    "heavy-adaptive is at least ~3× fewer marches than full",
  );
});

test("preview.js stillBlob threads opts.aaMode through exportSampleCount", () => {
  // Source-level pin (CI has no GPU): the export path must derive its sample
  // count from renderpolicy.exportSampleCount, fed the accumCap()==0 heavy signal
  // and opts.aaMode — so core owns the DECISION but stays pref-agnostic (the mode
  // arrives from the app). Guards against a future edit re-hardcoding "24".
  const src = readFileSync(
    fileURLToPath(new URL("./preview.js", import.meta.url)),
    "utf8",
  );
  assert.match(src, /exportSampleCount\(/, "stillBlob calls exportSampleCount");
  assert.match(src, /mode:\s*opts\.aaMode/, "opts.aaMode is threaded in");
  assert.match(
    src,
    /heavy:\s*!bakeDOF/,
    "heavy is the accumCap()==0 signal (bakeDOF's inverse)",
  );
});

// ── classifyTier (CAPABILITY_PROBE.md) — static machine class ────────────────
test("classifyTier: no backend → ascii floor", () => {
  assert.equal(classifyTier({ backend: "none" }), "ascii");
  assert.equal(classifyTier({}), "ascii");
});
test("classifyTier: software rasterizers → software (WARP/llvmpipe/SwiftShader)", () => {
  assert.equal(
    classifyTier({ backend: "webgpu", isSoftware: true }),
    "software",
  );
  assert.equal(
    classifyTier({
      backend: "webgpu",
      description: "Microsoft Basic Render Driver",
    }),
    "software",
  );
  assert.equal(
    classifyTier({ backend: "webgl2", vendor: "Google SwiftShader" }),
    "software",
  );
  assert.equal(
    classifyTier({ backend: "webgpu", architecture: "llvmpipe" }),
    "software",
  );
});
test("classifyTier: Apple M-series → fast", () => {
  assert.equal(
    classifyTier({ backend: "webgpu", vendor: "apple", architecture: "m1" }),
    "fast",
  );
  assert.equal(
    classifyTier({ backend: "webgpu", description: "Apple M4 Max" }),
    "fast",
  );
});
test("classifyTier: Pascal / GTX 10-series and ≤2 cores → slow", () => {
  assert.equal(
    classifyTier({ backend: "webgpu", description: "NVIDIA GeForce GTX 1080" }),
    "slow",
  );
  assert.equal(
    classifyTier({ backend: "webgpu", architecture: "pascal" }),
    "slow",
  );
  assert.equal(
    classifyTier({ backend: "webgpu", vendor: "intel", cores: 2 }),
    "slow",
  );
});
test("classifyTier: everything else → medium", () => {
  assert.equal(
    classifyTier({
      backend: "webgpu",
      vendor: "intel",
      description: "Intel Iris Xe",
      cores: 8,
    }),
    "medium",
  );
  assert.equal(
    classifyTier({
      backend: "webgpu",
      description: "NVIDIA GeForce RTX 4090",
      cores: 24,
    }),
    "medium",
  );
});

// ── bootPredictMs / shouldRaceGeneralAtBoot (DEFERRED_FORMULA_SWAP.md Phase 2a)
test("bootPredictMs: only 'fast' seeds lower; everything else keeps the conservative default", () => {
  assert.equal(bootPredictMs("fast"), BOOT_PREDICT_MS_FAST);
  assert.equal(bootPredictMs("medium"), BOOT_PREDICT_MS_DEFAULT);
  assert.equal(bootPredictMs("slow"), BOOT_PREDICT_MS_DEFAULT);
  assert.equal(bootPredictMs("software"), BOOT_PREDICT_MS_DEFAULT);
  assert.equal(bootPredictMs("ascii"), BOOT_PREDICT_MS_DEFAULT);
  assert.equal(bootPredictMs(undefined), BOOT_PREDICT_MS_DEFAULT);
  assert.ok(BOOT_PREDICT_MS_FAST < BOOT_PREDICT_MS_DEFAULT);
});
test("shouldRaceGeneralAtBoot: only fast/medium race; slow/software/unclassified keep today's single attempt", () => {
  assert.equal(shouldRaceGeneralAtBoot("fast"), true);
  assert.equal(shouldRaceGeneralAtBoot("medium"), true);
  assert.equal(shouldRaceGeneralAtBoot("slow"), false);
  assert.equal(shouldRaceGeneralAtBoot("software"), false);
  assert.equal(shouldRaceGeneralAtBoot("ascii"), false);
  assert.equal(shouldRaceGeneralAtBoot(undefined), false);
});

// ── ASCII/CPU motion tier (#32 — "When moving, I'm seeing different result
// completely…"). The reporter asked for the "Quality while moving" preference
// to apply to the ASCII view too; before this it was GPU-only and the ASCII
// tier's own motion drop was hardcoded with no opt-out.
//
// COLS is ascii.ts's own ladder, mirrored here ONLY to prove the byte-identical
// claim below. asciiMoveParams deliberately does NOT return a column floor —
// COLS[0] has exactly one owner (ascii.ts grid()), so a future density change
// cannot silently break `balanced`.
const COLS = [40, 80, 120, 240];
const asciiCols = (base, mq) => {
  const s = asciiMoveParams(mq).colScale;
  return s < 1 ? Math.max(COLS[0], Math.round(base * s)) : base;
};

test("#32 asciiMoveParams: 'balanced' is byte-identical to the pre-#32 hardcoded tier", () => {
  // ascii.ts grid() used to read, verbatim:
  //   Math.max(COLS[0], Math.round(base * 0.6))
  assert.equal(ASCII_MOVE_COL_SCALE, 0.6);
  const { colScale, fx, spinCoarse } = asciiMoveParams("balanced");
  assert.equal(colScale, 0.6);
  assert.equal(fx, false); // the overlay passes paused during motion, as before
  assert.equal(spinCoarse, false); // spin kept full density, as before
  // Table-driven across EVERY density index, not spot-checked — including the
  // degenerate index 0, where round(40*0.6)=24 clamps back to 40 and the coarse
  // tier is already a no-op today.
  for (const base of COLS)
    assert.equal(
      asciiCols(base, "balanced"),
      Math.max(COLS[0], Math.round(base * 0.6)),
    );
  assert.equal(asciiCols(40, "balanced"), 40);
  assert.equal(asciiCols(80, "balanced"), 48);
  assert.equal(asciiCols(120, "balanced"), 72);
  assert.equal(asciiCols(240, "balanced"), 144);
});

test("#32 asciiMoveParams: 'full' means no motion tier at all — the moving frame equals the settled one", () => {
  const q = asciiMoveParams("full");
  assert.equal(q.colScale, 1); // ascii.ts skips touchMotion entirely at >= 1
  assert.equal(q.fx, true); // edge/structure/dither overlays keep running
  assert.equal(q.spinCoarse, false);
  for (const base of COLS) assert.equal(asciiCols(base, "full"), base);
});

test("#32 asciiMoveParams: 'smooth' is monotonically cheaper than 'balanced', never below the coarsest density", () => {
  const q = asciiMoveParams("smooth");
  assert.ok(q.colScale < asciiMoveParams("balanced").colScale);
  assert.equal(q.fx, false);
  // Spin bounds the ~600 ms/frame main-thread block ONLY on this rung — the
  // deliberate divergence from the GPU tier (preview.js coarsens every
  // autoRotate frame). Pinned in BOTH directions so neither can drift.
  assert.equal(q.spinCoarse, true);
  assert.equal(asciiMoveParams("balanced").spinCoarse, false);
  assert.equal(asciiMoveParams("full").spinCoarse, false);
  for (const base of COLS) {
    assert.ok(asciiCols(base, "smooth") <= asciiCols(base, "balanced"));
    assert.ok(asciiCols(base, "smooth") >= COLS[0]); // never below the user's coarsest choice
  }
});

test("#32 asciiMoveParams: unknown / absent input falls back to balanced", () => {
  for (const bad of [undefined, null, "", "BALANCED", "low", 3, {}])
    assert.deepEqual(asciiMoveParams(bad), asciiMoveParams("balanced"));
});

test("#32 asciiMoveParams: no rung exposes a march knob — the ASCII shape can never be a function of motion", () => {
  // The whole point of the ASCII half of #32: unlike qualityParams (whose
  // `smooth` rung cuts steps/eps/deScale and genuinely moves the geometry),
  // this policy may only ever change SAMPLING DENSITY and GLYPH VOCABULARY.
  for (const mq of ["full", "balanced", "smooth"]) {
    const keys = Object.keys(asciiMoveParams(mq)).sort();
    assert.deepEqual(keys, ["colScale", "fx", "spinCoarse"]);
    for (const banned of [
      "steps",
      "eps",
      "deScale",
      "iters",
      "maxSteps",
      "iterCap",
    ])
      assert.equal(
        banned in asciiMoveParams(mq),
        false,
        `${mq} must not carry ${banned}`,
      );
  }
});

// ── the GPU half of #32, pinned against regression. `balanced` must keep the
// ENTIRE march budget — that is what makes the moving silhouette identical to
// the settled one (measured this session on real WebGPU: IoU 1.0000, 0 XOR
// pixels, on Amazing Surf / Cantor Rotations / Menger Cloud / the reporter's
// own formula). Only `scale` and `cheap` may differ.
test("#32 qualityParams: 'balanced' differs from settled ONLY in scale + cheap shading", () => {
  const formulas = [
    {
      ops: [
        { key: "boxFold", values: [1] },
        { key: "scale", values: [2] },
      ],
      iters: 12,
      deOption: 2,
    },
    // loose DE (scale < 2) — the Amazing Surf / Cantor Rotations class that
    // collapsed most visibly while moving
    {
      ops: [
        { key: "surfFold", values: [1] },
        { key: "scale", values: [1.9] },
      ],
      iters: 12,
      deOption: 2,
    },
  ];
  for (const f of formulas) {
    const settled = qualityParams(f, { quality: "full", dprCap: 2 });
    const bal = qualityParams(f, {
      quality: "low",
      moveQuality: "balanced",
      dprCap: 2,
    });
    assert.equal(
      bal.steps,
      settled.steps,
      "march budget must survive the drag",
    );
    assert.equal(bal.eps, settled.eps, "hit threshold must survive the drag");
    assert.equal(
      bal.deScale,
      settled.deScale,
      "step size must survive the drag",
    );
    assert.equal(bal.cheap, true); // shadow/AO are shading, not shape
    assert.ok(bal.scale <= settled.scale);
    // 'full' really is a no-op tier
    const fullTier = qualityParams(f, {
      quality: "low",
      moveQuality: "full",
      dprCap: 2,
    });
    assert.deepEqual(fullTier, settled);
    // 'smooth' is the ONLY rung allowed to move the geometry
    const sm = qualityParams(f, {
      quality: "low",
      moveQuality: "smooth",
      dprCap: 2,
    });
    assert.ok(
      sm.steps < settled.steps &&
        sm.eps > settled.eps &&
        sm.deScale > settled.deScale,
    );
  }
});

// ── #476 static formula cost score ──────────────────────────────────────────
const op = (key, ...values) => ({ key, values });

test("opEvalsPerIter: flat formula = op count", () => {
  assert.equal(opEvalsPerIter({ ops: [op("boxFold", 1), op("scale", 2)] }), 2);
  assert.equal(opEvalsPerIter({ ops: [] }), 0);
  assert.equal(opEvalsPerIter(null), 0);
  assert.equal(opEvalsPerIter({}), 0);
});

test("opEvalsPerIter: legacy 2-slot hybrid weights each slot by its schedule", () => {
  const f = {
    ops: [op("boxFold", 1), op("scale", 2)], // slot A: 2 ops
    hybrid: { b: { ops: [op("sphereFold", 1)] }, schedule: { a: 1, b: 3 } }, // slot B: 1 op ×3
  };
  // 2×1 (A) + 1×3 (B) = 5
  assert.equal(opEvalsPerIter(f), 5);
});

test("opEvalsPerIter: a muted hybrid slot contributes zero", () => {
  const base = {
    ops: [op("boxFold", 1), op("scale", 2)],
    hybrid: { b: { ops: [op("sphereFold", 1)] }, schedule: { a: 2, b: 2 } },
  };
  assert.equal(opEvalsPerIter(base), 2 * 2 + 1 * 2); // 6
  // slot B muted → only A
  assert.equal(
    opEvalsPerIter({
      ...base,
      hybrid: { ...base.hybrid, b: { ...base.hybrid.b, muted: true } },
    }),
    4,
  );
  // slot A muted → only B
  assert.equal(
    opEvalsPerIter({ ...base, hybrid: { ...base.hybrid, aMuted: true } }),
    2,
  );
});

test("opEvalsPerIter: future N-slot shape (slots[]/schedule.counts[]) is counted defensively", () => {
  // The Deep Cage class: 4 phases, 3 ops each, schedule 1:2:2:1 → 3×(1+2+2+1)=18.
  const f = {
    ops: [op("a", 1), op("b", 1), op("c", 1)], // slot A also mirrored at top level (old-build FLAT degrade)
    hybrid: {
      slots: [
        { ops: [op("a", 1), op("b", 1), op("c", 1)] },
        { ops: [op("d", 1), op("e", 1), op("f", 1)] },
        { ops: [op("g", 1), op("h", 1), op("i", 1)] },
        { ops: [op("j", 1), op("k", 1), op("l", 1)] },
      ],
      schedule: { counts: [1, 2, 2, 1] },
    },
  };
  assert.equal(opEvalsPerIter(f), 18);
  // slots[] takes precedence over any legacy .b — no double-count of slot A.
  const withStrayB = {
    ...f,
    hybrid: { ...f.hybrid, b: { ops: [op("z", 1)] } },
  };
  assert.equal(opEvalsPerIter(withStrayB), 18);
  // a muted slot drops out; missing count defaults to 1.
  const muted = {
    hybrid: {
      slots: [
        { ops: [op("a", 1), op("b", 1)] },
        { ops: [op("c", 1)], muted: true },
      ],
      schedule: { counts: [2] }, // count for slot 1 absent → defaults to 1 (but it's muted anyway)
    },
  };
  assert.equal(opEvalsPerIter(muted), 4);
});

test("formulaCostScore: hybrid > flat at equal iters; pixels multiply", () => {
  const flat = { ops: [op("boxFold", 1), op("scale", 2)], iters: 12 };
  const hybrid = {
    ops: [op("boxFold", 1), op("scale", 2)],
    hybrid: { b: { ops: [op("sphereFold", 1)] }, schedule: { a: 2, b: 2 } },
    iters: 12,
  };
  assert.equal(formulaCostScore(flat), 2 * 12);
  assert.equal(formulaCostScore(hybrid), 6 * 12);
  assert.ok(formulaCostScore(hybrid) > formulaCostScore(flat));
  assert.equal(formulaCostScore(flat, { pixels: 1000 }), 2 * 12 * 1000);
  // explicit iters override (e.g. auto-detail boosted count)
  assert.equal(formulaCostScore(flat, { iters: 30 }), 2 * 30);
});

test("formulaCostScore: a scene sums ops×iters over its objects (objects multiply cost)", () => {
  const scene = {
    objects: [
      { ops: [op("boxFold", 1), op("scale", 2)], iters: 10 }, // 20
      { ops: [op("sphereFold", 1)], iters: 8 }, // 8
      { ops: [op("x", 1)], iters: 5, muted: true }, // skipped
    ],
  };
  assert.equal(formulaCostScore(scene), 28);
  // two identical objects cost twice one
  const one = { objects: [{ ops: [op("a", 1), op("b", 1)], iters: 10 }] };
  const two = {
    objects: [...one.objects, { ops: [op("a", 1), op("b", 1)], iters: 10 }],
  };
  assert.equal(formulaCostScore(two), 2 * formulaCostScore(one));
});

// ── #476 mobile-class detection ─────────────────────────────────────────────
test("isMobileClass: iPad (CriOS UA) and coarse+touch qualify; touch laptop + desktop do not", () => {
  // iPad Pro, Chrome iOS — the field-report device
  assert.equal(
    isMobileClass({
      maxTouchPoints: 5,
      coarsePointer: true,
      ua: "Mozilla/5.0 (iPad; CPU OS 17_0 like Mac OS X) CriOS/120 Mobile/15E148",
    }),
    true,
  );
  // iPadOS Safari desktop-mode UA ("Macintosh") — no UA marker, but coarse+touch catches it
  assert.equal(
    isMobileClass({
      maxTouchPoints: 5,
      coarsePointer: true,
      ua: "Macintosh; Intel Mac OS X",
    }),
    true,
  );
  assert.equal(isMobileClass({ ua: "Linux; Android 14; Pixel 8" }), true);
  // touch LAPTOP: touch present, but the PRIMARY pointer is fine (mouse) → not mobile
  assert.equal(
    isMobileClass({
      maxTouchPoints: 10,
      coarsePointer: false,
      ua: "Windows NT 10.0",
    }),
    false,
  );
  // plain desktop
  assert.equal(
    isMobileClass({ maxTouchPoints: 0, coarsePointer: false, ua: "Macintosh" }),
    false,
  );
  assert.equal(isMobileClass({}), false);
});

// ── #476 cost-aware entry clamp ─────────────────────────────────────────────
test("entryDetailClamp: desktop never clamps; a cheap formula never clamps", () => {
  const heavy = {
    ops: [op("a", 1), op("b", 1), op("c", 1)],
    hybrid: {
      slots: [
        { ops: [op("a", 1), op("b", 1), op("c", 1)] },
        { ops: [op("d", 1), op("e", 1), op("f", 1)] },
        { ops: [op("g", 1), op("h", 1), op("i", 1)] },
        { ops: [op("j", 1), op("k", 1), op("l", 1)] },
      ],
      schedule: { counts: [1, 2, 2, 1] },
    },
    iters: 18,
  };
  assert.equal(entryDetailClamp(heavy, { coarseMobile: false }), null); // desktop
  const cheap = { ops: [op("boxFold", 1), op("scale", 2)], iters: 12 }; // 2×12=24 ≤ 160
  assert.equal(entryDetailClamp(cheap, { coarseMobile: true }), null);
});

test("entryDetailClamp: a heavy formula on mobile clamps starting detail down, floored, only lower", () => {
  const heavy = {
    hybrid: {
      slots: [
        { ops: [op("a", 1), op("b", 1), op("c", 1)] },
        { ops: [op("d", 1), op("e", 1), op("f", 1)] },
        { ops: [op("g", 1), op("h", 1), op("i", 1)] },
        { ops: [op("j", 1), op("k", 1), op("l", 1)] },
      ],
      schedule: { counts: [1, 2, 2, 1] },
    },
    iters: 18,
  };
  // opEvals/iter = 18, base 18 → cost 324 > 160 → clamp to floor(160/18)=8
  const clamp = entryDetailClamp(heavy, { coarseMobile: true });
  assert.equal(clamp, 8);
  assert.ok(clamp < heavy.iters);
  assert.ok(clamp >= MOBILE_ENTRY_MIN_DETAIL);
  // clamp never below the min-detail floor even for an extreme formula
  const extreme = { ...heavy, iters: 64 };
  assert.ok(
    entryDetailClamp(extreme, { coarseMobile: true, budget: 20 }) >=
      MOBILE_ENTRY_MIN_DETAIL,
  );
  // scenes are out of scope (detailOverride is single-object)
  assert.equal(
    entryDetailClamp(
      { objects: [{ ops: [op("a", 1)], iters: 40 }] },
      { coarseMobile: true },
    ),
    null,
  );
});

test("entryDetailClamp: never RAISES — a base already under the implied cap is left alone", () => {
  // opEvals/iter 18, base 6 → cost 108 ≤ 160 → no clamp (would-be clamp 8 > base 6)
  const f = {
    hybrid: {
      slots: [
        { ops: [op("a", 1), op("b", 1), op("c", 1)] },
        { ops: [op("d", 1), op("e", 1), op("f", 1)] },
        { ops: [op("g", 1), op("h", 1), op("i", 1)] },
        { ops: [op("j", 1), op("k", 1), op("l", 1)] },
      ],
      schedule: { counts: [1, 2, 2, 1] },
    },
    iters: 6,
  };
  assert.equal(entryDetailClamp(f, { coarseMobile: true }), null);
});

// ── #562 — entry clamp survives the load's own frameTo retarget-reset ──────
// preview.js's frameTo (#551/#560) nulls detailOverride on every retarget, and
// every load path (preset/share/import/Surprise/Remix/Wander) calls setFormula
// then frameTo back-to-back — so the #476 clamp setFormula just applied was
// nulled by that SAME load's frameTo before a single frame ever rendered.
// makeEntryClampArm is the fix's pure seam: these tests pin its one-shot
// arm/consume contract directly, and the last one replays preview.js's exact
// call sequence to pin the end-to-end fix.
test("makeEntryClampArm: an armed clamp survives exactly the next consume, then resets", () => {
  const arm = makeEntryClampArm();
  arm.arm();
  assert.equal(arm.consumeSurvives(), true); // the load's own frameTo preserves it
  assert.equal(arm.consumeSurvives(), false); // one-shot — a LATER retarget resets normally
});

test("makeEntryClampArm: never armed → consume is always false", () => {
  const arm = makeEntryClampArm();
  assert.equal(arm.consumeSurvives(), false);
  assert.equal(arm.consumeSurvives(), false);
});

test("makeEntryClampArm: disarm cancels a pending arm (a real zoom/manual write beats a stale arm)", () => {
  const arm = makeEntryClampArm();
  arm.arm();
  arm.disarm();
  assert.equal(arm.consumeSurvives(), false);
});

test("#562 regression: the mobile preset-load sequence (setFormula → frameTo) preserves the #476 clamp through to the drawn frame", () => {
  // Models preview.js's exact call order for every load path. setFormula
  // computes+applies the entry clamp and arms; the load's frameTo (a
  // retarget — new shape, cont=false) used to unconditionally null
  // detailOverride right back out — the #562 defect.
  const heavy = {
    hybrid: {
      slots: [
        { ops: [op("a", 1), op("b", 1), op("c", 1)] },
        { ops: [op("d", 1), op("e", 1), op("f", 1)] },
        { ops: [op("g", 1), op("h", 1), op("i", 1)] },
        { ops: [op("j", 1), op("k", 1), op("l", 1)] },
      ],
      schedule: { counts: [1, 2, 2, 1] },
    },
    iters: 18,
  };
  const arm = makeEntryClampArm();
  let detailOverride = null;

  // setFormula(heavy) on a coarse/mobile device:
  arm.disarm(); // every setFormula call starts clean (no stale prior arm)
  const clamp = entryDetailClamp(heavy, { coarseMobile: true });
  assert.ok(clamp != null, "heavy formula must actually clamp on mobile");
  detailOverride = clamp;
  arm.arm();

  // frameTo(preset's saved camera) — a retarget (new shape), cont=false:
  const cont = false;
  const clampSurvives = arm.consumeSurvives();
  if (!cont && !clampSurvives) detailOverride = null;
  assert.equal(
    detailOverride,
    clamp,
    "#562: the entry clamp must survive the load's own frameTo",
  );

  // A LATER, unrelated retarget (no intervening setFormula) resets normally —
  // the clamp protects only the first unmeasured settle, not a standing floor.
  const laterSurvives = arm.consumeSurvives();
  if (!cont && !laterSurvives) detailOverride = null;
  assert.equal(
    detailOverride,
    null,
    "a later retarget must still hand detail back to auto-detail",
  );
});

test("#562 regression: a load that does NOT clamp (cheap formula / desktop) never arms — frameTo nulls as before", () => {
  const cheap = { ops: [op("boxFold", 1), op("scale", 2)], iters: 12 };
  const arm = makeEntryClampArm();
  let detailOverride = 99; // pretend a stale value was sitting here

  arm.disarm();
  const clamp = entryDetailClamp(cheap, { coarseMobile: true });
  assert.equal(clamp, null);
  // setFormula only writes detailOverride/arms when clamp != null — unchanged here.

  const cont = false;
  const clampSurvives = arm.consumeSurvives();
  if (!cont && !clampSurvives) detailOverride = null;
  assert.equal(
    detailOverride,
    null,
    "no clamp means no arm — the retarget resets normally",
  );
});

// ── #476 per-frame governor state machine ───────────────────────────────────
test("governorInit: starts unengaged at full scale", () => {
  assert.deepEqual(governorInit(), { scale: 1, over: 0, under: 0 });
});

test("governorStep: a single over-budget frame does NOT act; it acts on GOV_OVER_FRAMES", () => {
  let s = governorInit();
  s = governorStep(s, 950); // over #1
  assert.equal(
    s.scale,
    1,
    "one slow frame is not enough (no cliff from a spike)",
  );
  assert.equal(s.over, 1);
  s = governorStep(s, 950); // over #2 → act
  assert.ok(s.scale < 1, "downshifts within GOV_OVER_FRAMES frames");
  assert.equal(GOV_OVER_FRAMES, 2);
});

test("governorStep: the iPad case — 950ms recovers under the 250ms budget in a few steps", () => {
  let s = governorInit();
  // Simulate the settle cost tracking scale² as the governor shrinks it.
  const fullMs = 950;
  const cost = () => fullMs * s.scale * s.scale;
  for (let i = 0; i < 12; i++) s = governorStep(s, cost(), { budgetMs: 250 });
  const finalMs = cost();
  assert.ok(
    finalMs <= 250,
    `final frame ${Math.round(finalMs)}ms must be under budget`,
  );
  assert.ok(s.scale >= GOV_SCALE_FLOOR, "never below the floor");
  assert.ok(s.scale < 1, "definitely engaged");
});

test("governorStep: one downshift halves at most (GOV_DOWN_MIN) even for a catastrophic frame", () => {
  let s = governorInit();
  s = governorStep(s, 100000); // absurdly slow
  s = governorStep(s, 100000); // acts on the 2nd
  assert.ok(
    s.scale >= GOV_DOWN_MIN - 1e-9,
    "a single step can't overcorrect past half",
  );
  assert.ok(s.scale <= GOV_DOWN_MIN + 1e-9);
});

test("governorStep: scale never drops below the floor no matter how many kills", () => {
  let s = governorInit();
  for (let i = 0; i < 100; i++) s = governorStep(s, 100000);
  assert.equal(s.scale, GOV_SCALE_FLOOR);
});

test("governorStep: hysteresis dead-band — a frame parked near budget does NOT oscillate", () => {
  // Downshift once, then feed a frame that lands in [recover, budget]. It must
  // neither downshift further nor recover — stable.
  let s = governorInit();
  s = governorStep(s, 400, { budgetMs: 250 });
  s = governorStep(s, 400, { budgetMs: 250 }); // acts
  const engaged = s.scale;
  assert.ok(engaged < 1);
  // A steady 200ms frame is in the dead-band [150, 250] → hold for many frames.
  for (let i = 0; i < 50; i++) s = governorStep(s, 200, { budgetMs: 250 });
  assert.equal(
    s.scale,
    engaged,
    "dead-band frames neither shrink nor grow the scale",
  );
});

test("governorStep: recovers only after sustained comfortable frames, and gently", () => {
  let s = governorInit();
  s = governorStep(s, 950, { budgetMs: 250 });
  s = governorStep(s, 950, { budgetMs: 250 }); // engaged
  const engaged = s.scale;
  // A handful of fast frames is NOT enough to recover (needs GOV_UNDER_FRAMES).
  for (let i = 0; i < GOV_UNDER_FRAMES - 1; i++)
    s = governorStep(s, 40, { budgetMs: 250 });
  assert.equal(s.scale, engaged, "recovery waits for sustained comfort");
  s = governorStep(s, 40, { budgetMs: 250 }); // the GOV_UNDER_FRAMES-th
  assert.ok(s.scale > engaged, "then recovers a gentle step");
  assert.ok(s.scale <= 1);
});

test("governorStep: a custom budget scales the recover threshold proportionally (test hook)", () => {
  // ?govbudget=80 → recover at 0.6×80=48. A 60ms frame must sit in the dead-band
  // (not recover), a 40ms frame must be comfortable.
  let s = { scale: 0.5, over: 0, under: 0 };
  const held = governorStep(s, 60, { budgetMs: 80 });
  assert.equal(
    held.under,
    0,
    "60ms is inside the dead-band for an 80ms budget",
  );
  const comfy = governorStep(s, 40, { budgetMs: 80 });
  assert.equal(
    comfy.under,
    1,
    "40ms is below the proportional recover threshold",
  );
});

test("governorStep: ignores non-measurements (0 / negative / NaN)", () => {
  let s = { scale: 0.5, over: 1, under: 2 };
  assert.deepEqual(governorStep(s, 0), s);
  assert.deepEqual(governorStep(s, -5), s);
  assert.deepEqual(governorStep(s, NaN), s);
});

// ── classifyDeviceLoss (#473 — a lost device must fall a tier, not freeze) ───
test("#473: a lost device demotes — the field case is {reason:'unknown'}", () => {
  // The 2026-08-01 iPad/CriOS dump verbatim: device-lost with reason 'unknown'
  // and no message. This is THE case that used to leave a frozen canvas.
  const v = classifyDeviceLoss({ reason: "unknown", message: "" });
  assert.equal(v.demote, true, "an unknown-reason loss is a crash → demote");
  assert.match(v.reason, /^device-lost: unknown$/);
});

test("#473: a reason-less / undefined loss still demotes", () => {
  // Implementations vary in what they populate; absence must never read as
  // "fine". Both the no-argument and the empty-detail call must demote.
  for (const arg of [undefined, {}, { reason: undefined }, { reason: null }]) {
    const v = classifyDeviceLoss(arg);
    assert.equal(v.demote, true, `demotes for ${JSON.stringify(arg)}`);
    assert.equal(v.reason, "device-lost: unknown");
  }
});

test("#473: reason 'destroyed' is OUR teardown — never demote", () => {
  // device.destroy() / page teardown resolves device.lost with 'destroyed'.
  // Demoting there would swap the view (and toast) on a user who is leaving.
  for (const reason of ["destroyed", "Destroyed", "DESTROYED"]) {
    const v = classifyDeviceLoss({ reason });
    assert.equal(v.demote, false, `'${reason}' is intentional, not a fault`);
    assert.equal(v.reason, null);
  }
});

test("#473: demotion fires exactly once per session", () => {
  // The app's own guard is belt-and-braces; the policy is the authority, so a
  // second loss (or the drain of losses after one) cannot re-enter the fallback.
  const first = classifyDeviceLoss({ reason: "unknown", demoted: false });
  assert.equal(first.demote, true);
  const second = classifyDeviceLoss({ reason: "unknown", demoted: true });
  assert.equal(second.demote, false, "already demoted → no second transition");
  assert.equal(second.reason, null);
});

test("#473: the message is carried but the reason string stays bounded", () => {
  const v = classifyDeviceLoss({
    reason: "unknown",
    message: "Validation failure",
  });
  assert.equal(v.reason, "device-lost: unknown — Validation failure");
  // The string reaches a toast and the /stats beacon — a driver dumping a wall
  // of text must not blow either up.
  const long = classifyDeviceLoss({
    reason: "unknown",
    message: "x".repeat(999),
  });
  assert.ok(
    long.reason.length <= DEVICE_LOST_REASON_MAX,
    `reason capped at ${DEVICE_LOST_REASON_MAX}, got ${long.reason.length}`,
  );
});

test("#473: preview.js routes a lost device into the onGpuDead demotion", () => {
  // Source-level pin (CI has no GPU — device-lost cannot be provoked here).
  // THE regression: before the fix, preview.js's device-lost handler only
  // latched `deviceLost = true` and stopped, so the canvas froze half-rendered
  // and the ONLY runtime demotion (onGpuDead) was reachable from the dead-GL
  // gate alone. Assert the event is wired to the ladder, and that the pump
  // guard which made the freeze survivable is still there.
  const src = readFileSync(
    fileURLToPath(new URL("./preview.js", import.meta.url)),
    "utf8",
  );
  assert.match(
    src,
    /classifyDeviceLoss\(/,
    "the device-lost verdict comes from the pure policy layer",
  );
  // The demote branch must set hasGPU=false AND call opts.onGpuDead — the same
  // pair the dead-GL tier uses. Matched together so neither can drift alone.
  assert.match(
    src,
    /gpuDeadFired\s*=\s*true;\s*\n\s*hasGPU\s*=\s*false;[\s\S]{0,400}?opts\.onGpuDead\?\.\(/,
    "the demote branch latches, drops hasGPU and hands the app onGpuDead",
  );
  assert.match(
    src,
    /noteDiag\("device-lost-fallback"/,
    "the demotion is recorded as its own diag event",
  );
  // #476 Part C must survive: the pump still refuses to submit on a dead device.
  assert.match(
    src,
    /if \(deviceLost\) \{\s*\n\s*lastSkip = "device-lost";/,
    "the pump still short-circuits on a lost device",
  );
});

// ── readbackBudgetMs (#460 residue — mapAsync is the fence's twin) ───────────
// #477 raced every queue fence against a timeout. It left `buffer.mapAsync()`
// bare, and a watchdog-killed submit hangs a map exactly as it hangs a fence —
// re-creating the permanent freeze on the readback paths (three of which hold
// preview.js's `busy`, two the exporters' `offline`).
test("#460: a single-submit map waits the floor, not the fence's 10 s", () => {
  // The floor is deliberately far above #477's 10 s: a hung map is PERMANENT,
  // so recovering in 60 s costs nothing, while tripping on a healthy 40 s
  // export would cost the user their render.
  assert.equal(readbackBudgetMs(1), READBACK_MS_FLOOR);
  assert.ok(
    READBACK_MS_FLOOR >= 60000,
    `floor must stay generous, got ${READBACK_MS_FLOOR}`,
  );
});

test("#460: the budget scales with the submits queued before the map", () => {
  // A 24-sample export queues 24 accumulate submits before mapping; the map
  // cannot resolve until all of them retire, so a fixed ceiling would abort
  // real work.
  assert.equal(readbackBudgetMs(24), 24 * READBACK_MS_PER_SUBMIT);
  assert.ok(readbackBudgetMs(24) > readbackBudgetMs(4));
  // Monotonic — more queued work can never mean a tighter deadline.
  let prev = 0;
  for (const n of [1, 2, 8, 24, 256]) {
    const b = readbackBudgetMs(n);
    assert.ok(b >= prev, `budget must not shrink at submits=${n}`);
    prev = b;
  }
});

test("#460: a nonsense submit count falls back to the floor, never to 0", () => {
  // A zero/NaN budget would make setTimeout fire immediately and abort every
  // readback — a false timeout is the one outcome worse than the hang.
  for (const bad of [0, -3, NaN, Infinity, undefined, null, "many"]) {
    assert.equal(
      readbackBudgetMs(bad),
      READBACK_MS_FLOOR,
      `submits=${String(bad)} must degrade to the floor`,
    );
  }
});

test("#460: every renderer.js readback maps through the watchdog guard", () => {
  // Source-level pin (CI has no GPU — a watchdog kill cannot be provoked here).
  // THE regression: a bare `await buf.mapAsync(...)` never settles when the
  // platform watchdog kills the submit, so preview.js's `busy` / the exporters'
  // `offline` stay latched and the canvas freezes until reload.
  const src = readFileSync(
    fileURLToPath(new URL("./renderer.js", import.meta.url)),
    "utf8",
  );
  // No mapAsync may be awaited directly — the guard must own every one.
  assert.doesNotMatch(
    src,
    /await\s+(?:Promise\.all\(\s*\[\s*)?[A-Za-z_$][\w$]*\.mapAsync\(/,
    "a bare awaited mapAsync is the #460 hang — route it through mapGuarded()",
  );
  assert.match(
    src,
    /async function mapGuarded\(where, submits, maps\)/,
    "the readback guard exists",
  );
  // It must actually REJECT on timeout: the whole recovery is that callers'
  // existing try/finally releases their latch, and only a throw runs those.
  assert.match(
    src,
    /setTimeout\(\s*\(\)\s*=>\s*\{[\s\S]{0,300}?rej\(/,
    "the guard rejects on timeout so the callers' finally blocks run",
  );
  assert.match(
    src,
    /note\("readback-timeout"/,
    "a hung readback is recorded as its own diag event",
  );
  // Each of the three await sites keeps a distinct tag, so the diag event says
  // WHICH readback hung.
  for (const where of ["tile-read", "render-to-image", "splat-capture"]) {
    assert.match(
      src,
      new RegExp(`mapGuarded\\("${where}"`),
      `the ${where} readback is guarded`,
    );
  }
  // The multi-sample path must pass its real submit count, not a bare 1 — a
  // 24-sample export legitimately outruns the floor.
  assert.match(
    src,
    /mapGuarded\("render-to-image",\s*\(ac \? samples : 0\) \+ 1,/,
    "the accumulate path budgets for every per-sample submit it queued",
  );
});


// ── #496: the auto-detail law, reusable without a camera ────────────────────
// The splat export zooms by FRAMING (a user-drawn capture volume, or the S-5a
// view frame) — a zoom no `cam.dist` describes. itersForMagnification states
// the same law in bare magnification and effectiveIters now delegates to it,
// so the count a zoomed CAMERA marches and the count a zoomed BOX marches
// cannot drift apart. #496's residue was exactly that drift: #415 removed
// `iters: views.effectiveIters()` from the export, and every capture since has
// marched the flat base count however far the framing zoomed in.
test("#496: itersForMagnification is one iteration per zoom octave, clamped", () => {
  assert.equal(ITER_PER_OCTAVE, 1.0, "the law is one iteration per octave");
  assert.equal(itersForMagnification(5, 1), 5, "no magnification ⇒ base");
  assert.equal(itersForMagnification(5, 2), 6, "×2 ⇒ +1");
  assert.equal(itersForMagnification(5, 4), 7, "×4 ⇒ +2");
  assert.equal(itersForMagnification(5, 8.7), 8, "the reporter's ×8.7 box ⇒ +3");
  assert.equal(itersForMagnification(5, 20), 9, "×20 ⇒ +4");
  assert.equal(
    itersForMagnification(5, 1e30),
    ITER_CEIL,
    "clamped at the cross-backend ceiling",
  );
});

test("#496: itersForMagnification never LOWERS the count (zoomed out ⇒ inert)", () => {
  for (const m of [1, 0.5, 0.01, 0, -3, NaN, undefined, Infinity])
    assert.equal(
      itersForMagnification(12, m),
      12,
      `magnification ${m} leaves the base count alone`,
    );
  // Infinity is deliberately in that list. magnificationFor can never produce
  // it (dist is floored at PT_MIN_DIST), so a non-finite magnification is a
  // CALLER BUG — and the safe response to a bug is the base count, never a
  // silent jump to a 64-iteration DE that would cost seconds per frame.
});

test("#496: effectiveIters is behavior-preserving over the delegation", () => {
  assert.equal(
    effectiveIters(9, { dist: REF_DIST }),
    9,
    "default framing ⇒ base",
  );
  assert.equal(
    effectiveIters(9, { dist: REF_DIST / 2 }),
    10,
    "one octave in ⇒ +1",
  );
  assert.equal(
    effectiveIters(9, { dist: REF_DIST / 8 }),
    12,
    "three octaves in ⇒ +3",
  );
  assert.equal(effectiveIters(9, { dist: REF_DIST * 4 }), 9, "zoomed OUT ⇒ base");
  assert.equal(
    effectiveIters(9, { autoDetail: false, dist: REF_DIST / 1024 }),
    9,
    "auto-detail off still bypasses the law entirely",
  );
  assert.equal(
    effectiveIters(9, { detailOverride: 30, dist: REF_DIST / 1024 }),
    30,
    "an absolute Detail override still wins over the law",
  );
  // The delegation is total: for any distance the camera can hold, the two
  // spellings agree exactly.
  for (const dist of [REF_DIST, REF_DIST / 3, 1e-6, 1e-20, 1e-30, 1e3])
    assert.equal(
      effectiveIters(9, { dist }),
      itersForMagnification(9, magnificationFor(dist)),
      `effectiveIters == itersForMagnification∘magnificationFor at dist ${dist}`,
    );
});

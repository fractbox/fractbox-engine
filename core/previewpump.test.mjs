// #709 / #710 — pump correctness + export concurrency regression tests.
//
// Two kinds of test here, matching the repo's CI reality ("WGSL is compiled
// nowhere in CI", no GPU, no DOM):
//
//   1. HEADLESS BEHAVIORAL: createPreview boots fine in Node with a tiny
//      window/canvas shim and a forced backend of "none" (no WebGPU, no
//      WebGL2 — the ladder's ASCII rung). Everything that doesn't need a GPU
//      — the camera, frameTo, setCruise's release probe, the #710 render
//      lock, setOffline's drain — is exercised for real. The cruise-release
//      TypeError and the NaN-share-link throw both reproduced under exactly
//      this harness before their fixes.
//
//   2. SOURCE-LEVEL PINS for the pump internals only a live GPU reaches
//      (settle hot-loop predicate, measuredTier epilogue, drewOk error
//      guards, band-abort channels) — the same pinning strategy
//      alphaexport.test.mjs / renderpolicy.test.mjs already use on this file.
//
// Run: node --test core/previewpump.test.mjs
import assert from "node:assert/strict";
import { test, after } from "node:test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

// ── Headless browser-globals shim (before createPreview is CALLED; the module
// itself is import-safe in Node — clipplane.test.mjs already imports it bare).
// CI's node has NO `navigator` (it landed as a Node global only in v21), so
// stub exactly what createPreview's boot path reads: navigator.maxTouchPoints
// (isTouch/coarseMobile) + userAgent (isMobileClass) + hardwareConcurrency /
// deviceMemory (the capability probe); window.matchMedia / devicePixelRatio /
// addEventListener; rAF. backend:"none" throws before createRenderer, so
// navigator.gpu is never consulted. Every stub is CREATED-only (a real global
// is never replaced) and tracked, then removed in after() — no pollution if
// the runner ever shares a process across test files.
const stubbedGlobals = [];
const stubGlobal = (key, value) => {
  if (globalThis[key] === undefined) {
    globalThis[key] = value;
    stubbedGlobals.push(key);
  }
};
stubGlobal("performance", { now: () => Date.now() });
stubGlobal("navigator", {
  userAgent: "node",
  maxTouchPoints: 0,
  hardwareConcurrency: 4,
  deviceMemory: 0,
});
stubGlobal("window", {});
const stubbedWindowKeys = [];
const stubWindow = (key, value) => {
  if (window[key] === undefined) {
    window[key] = value;
    stubbedWindowKeys.push(key);
  }
};
stubWindow("matchMedia", () => ({ matches: false }));
stubWindow("devicePixelRatio", 1);
stubWindow("addEventListener", () => {});
stubGlobal("requestAnimationFrame", (cb) =>
  setTimeout(() => cb(performance.now()), 0),
);
stubGlobal("cancelAnimationFrame", clearTimeout);
after(() => {
  for (const k of stubbedWindowKeys) delete globalThis.window[k];
  for (const k of stubbedGlobals) delete globalThis[k];
});

const { createPreview } = await import("./preview.js");
const { PRESETS, clone } = await import("./oplist.js");

const bulb = () => clone(PRESETS.find((f) => f.name === "Mandelbulb"));
const makeCanvas = () => ({
  width: 300,
  height: 150,
  clientHeight: 150,
  addEventListener: () => {},
  getBoundingClientRect: () => ({ width: 300, height: 150, left: 0, top: 0 }),
});
const boot = async () => {
  const p = await createPreview(makeCanvas(), { backend: "none" });
  p.setFormula(bulb());
  return p;
};

const src = readFileSync(
  fileURLToPath(new URL("./preview.js", import.meta.url)),
  "utf8",
);

// ── #709 finding 1 — cruise release ─────────────────────────────────────────

test("#709: setCruise(true) then setCruise(false) with a formula loaded does not throw", async () => {
  const p = await boot();
  p.setCruise(true, 1);
  assert.equal(p.cruising(), true);
  // Pre-fix: the release path called surfaceAhead(b.eye, b.fwd) against the
  // PR-4 `(b)` signature — probeFrame read `.eye` off the eye ARRAY and threw
  // "Cannot read properties of undefined (reading 'slice')" on every flight
  // release since 2026-07-31 (reproduced under this exact harness).
  p.setCruise(false);
  assert.equal(p.cruising(), false);
  assert.ok(
    Number.isFinite(p.cam.dist) && p.cam.dist > 0,
    "release leaves a sane camera distance",
  );
  assert.ok(
    p.cam.target.every(Number.isFinite),
    "release leaves a finite orbit target",
  );
});

test("#709: the cruise release re-pin routes through nudgeTarget (D6 bookkeeping), not an absolute target write", () => {
  // The pump's own cruise hit path re-pins via nudgeTarget(roRel + fwd·h);
  // the release must be the same residual-space delta so a deep flight's
  // exact-target (ptTfx) survives instead of being rebased to f64.
  const rel = src.match(
    /cruise = null;[\s\S]{0,1600}?nudgeTarget\(\[\s*b\.roRel\[0\] \+ b\.fwd\[0\] \* h,/,
  );
  assert.ok(rel, "setCruise(false) re-pins through nudgeTarget(roRel + fwd·h)");
});

// ── #709 finding 4 — frameTo screens crafted share-link floats ──────────────

test("#709: frameTo survives non-finite floats in every VIEW field", async () => {
  const p = await boot();
  // Pre-fix: fxFromF64 THROWS on the NaN targetLo (load left half-applied),
  // and a NaN target landed verbatim (every ray NaN → black screen).
  p.frameTo({
    yawDeg: NaN,
    pitchDeg: 22,
    dist: NaN,
    fovDeg: Infinity,
    target: [NaN, 1, 2],
    targetLo: [NaN, 0, 0],
    targetLo2: [Infinity, 0, 0],
  });
  const c = p.camObj();
  for (const k of ["yawDeg", "pitchDeg", "dist", "fovDeg"])
    assert.ok(Number.isFinite(c[k]), `${k} finite after a crafted load`);
  assert.deepEqual(
    p.cam.target,
    [0, 1, 2],
    "non-finite target words default to 0; finite ones survive",
  );
  p.frameTo({ target: [NaN, NaN, NaN], dist: 5 });
  assert.deepEqual(p.cam.target, [0, 0, 0]);
  assert.ok(Number.isFinite(p.cam.dist));
});

test("#709: frameTo's retarget branch re-clamps dist AFTER the target lands", () => {
  // The dist accessor's wall floor carries a max(|target|,1) orbit term for
  // the df64/f32 classes; writing dist before the target evaluated the floor
  // against the PREVIOUS view's orbit scale. The raw value is re-written once
  // the new target (and its ptTfx seed) are in place.
  assert.match(
    src,
    /cam\.target = nextT;[\s\S]{0,2200}?cam\.dist = fin\(c\.dist, 24\);\s*\n\s*ptOrbitKey = "";/,
    "cam.dist must be re-assigned after cam.target/ptTfx in the retarget branch",
  );
});

// ── #710 — the render-ownership lock ────────────────────────────────────────

test("#710: render lock is single-owner with token-guarded release", async () => {
  const p = await boot();
  assert.equal(p.renderOwner(), null);
  const t1 = p.tryAcquireRender("test-a");
  assert.ok(t1 != null, "free lock acquires");
  assert.equal(p.renderOwner(), "test-a");
  assert.equal(p.renderInfo().renderOwner, "test-a", "owner on renderInfo");
  assert.equal(p.tryAcquireRender("test-b"), null, "second claimant refused");
  assert.equal(p.releaseRender(t1 + 999), false, "a stale token is a no-op");
  assert.equal(p.renderOwner(), "test-a", "still held after the stale release");
  assert.equal(p.releaseRender(t1), true);
  assert.equal(p.renderOwner(), null);
  const t2 = p.tryAcquireRender("test-c");
  assert.notEqual(t2, t1, "tokens are never reused");
  assert.equal(
    p.releaseRender(t1),
    false,
    "an old token cannot release a newer hold — the #710 stomp, prevented",
  );
  assert.equal(p.renderOwner(), "test-c");
  assert.equal(p.releaseRender(t2), true);
});

test("#710: drainRender resolves immediately when free, and only on release when held", async () => {
  const p = await boot();
  await p.drainRender(); // free — must not hang
  const tok = p.tryAcquireRender("in-flight-frame");
  let drained = false;
  const d = p.drainRender().then(() => {
    drained = true;
  });
  await new Promise((r) => setTimeout(r, 10));
  assert.equal(drained, false, "held — drain must still be pending");
  p.releaseRender(tok);
  await d;
  assert.equal(drained, true);
});

test("#710: setOffline(true) returns a promise that resolves only after the in-flight hold releases", async () => {
  const p = await boot();
  const tok = p.tryAcquireRender("in-flight-frame");
  let drained = false;
  const pr = p.setOffline(true);
  assert.equal(typeof pr?.then, "function", "setOffline returns a thenable");
  assert.equal(p.isOffline(), true, "the offline gate itself flips SYNCHRONOUSLY");
  pr.then(() => {
    drained = true;
  });
  await new Promise((r) => setTimeout(r, 10));
  assert.equal(drained, false, "must not resolve while a frame is in flight");
  p.releaseRender(tok);
  await pr;
  assert.equal(drained, true);
  await p.setOffline(true); // already offline — resolves immediately
  p.setOffline(false);
  assert.equal(p.isOffline(), false);
});

test("#710: beginThumbs/endThumbs ride the lock (owner-tagged, released on end)", async () => {
  const p = await boot();
  await p.beginThumbs();
  assert.equal(p.renderOwner(), "thumbs");
  assert.equal(p.tryAcquireRender("test"), null, "thumb batch excludes others");
  p.endThumbs();
  assert.equal(p.renderOwner(), null);
});

test("#710: `busy` is a mirror — only the lock functions write it", () => {
  // Nine independent `busy = true/false` sites were the defect: any release
  // freed any hold. Post-fix the ONLY assignments live inside
  // tryAcquireRender/releaseRenderToken (plus the declaration).
  assert.equal(
    src.match(/busy = true/g).length,
    1,
    "exactly one `busy = true` (inside tryAcquireRender)",
  );
  assert.equal(
    src.match(/busy = false/g).length,
    2,
    "exactly two `busy = false` (declaration + releaseRenderToken)",
  );
  assert.match(src, /releaseRenderToken\(frameTok\)/, "pump releases by token");
  assert.match(src, /releaseRenderToken\(accumTok\)/, "refine releases by token");
  assert.match(
    src,
    /await acquireRenderExclusive\("still"\)/,
    "stillBlob acquires (drains the pump) instead of stomping",
  );
  assert.match(src, /await acquireRenderExclusive\("still-tiled"\)/);
  assert.match(src, /await acquireRenderExclusive\("capture-frame"\)/);
});

test("#710: setOffline's drain + the band loops' offline aborts are wired", () => {
  assert.match(
    src,
    /return offline && !was \? drainRender\(\) : Promise\.resolve\(\);/,
    "setOffline(true) returns the drain promise",
  );
  // Both banded loops (settle + refine tick) abort on offline AND on the
  // #709 settle-generation channel, so the drain lands within ~one band and
  // a gesture can never be swallowed by the self-restoring quality flag.
  assert.match(
    src,
    /quality !== "full" \|\|\s*offline \|\|\s*settleGen !== frameGen \|\|/,
    "banded settle aborts on offline + settleGen",
  );
  assert.match(
    src,
    /needsDraw \|\|\s*quality !== "full" \|\|\s*offline \|\|\s*settleGen !== tickGen/,
    "banded refine tick aborts on offline + settleGen",
  );
  assert.match(
    src,
    /function bumpInteract\(\) \{\s*quality = "low";\s*settleGen\+\+;/,
    "bumpInteract bumps the generation",
  );
});

// ── #709 finding 2 — settle hot-loop after a fence timeout (source pin: the
// predicate lives mid-pump, unreachable without a GPU) ──────────────────────

test("#709: the coarse-first upgrade predicate excludes settleScaleCap like it excludes governorScale", () => {
  assert.match(
    src,
    /governorScale >= 1 &&[\s\S]{0,900}?settleScaleCap >= 1\s*\n\s*\) \{/,
    "settleScaleCap < 1 must not re-trigger the upgrade settle — the " +
      "post-fence-timeout rAF-cadence settle→upgrade hot-loop",
  );
  // …and the cap RECOVERS after N clean settles instead of persisting for
  // the session (one early watchdog kill no longer means half-res forever).
  assert.match(src, /settleScaleCap = Math\.min\(1, settleScaleCap \* 2\);/);
  assert.match(
    src,
    /function onFenceTimeout\(where\) \{\s*fenceTimeouts\+\+;\s*settleCleanRuns = 0;/,
    "a timeout restarts the clean-run count",
  );
});

// ── #709 finding 3 — measuredTier on the single-dispatch settle path ────────

test("#709: the single-dispatch settle epilogue tags measuredTier (both settle paths agree)", () => {
  const tags = src.match(
    /measuredTier = lastPt \? "pt" : lastDf64 \? "df64" : "f32";/g,
  );
  assert.equal(
    tags?.length,
    2,
    "banded AND single-dispatch epilogues both tag the measurement's tier",
  );
  assert.match(
    src,
    /if \(drewOk && tFull && !settleHung\) \{[\s\S]{0,1200}?measuredTier = lastPt \? "pt" : lastDf64 \? "df64" : "f32";/,
    "the tFull epilogue records the tier alongside lastFullMs",
  );
});

// ── #709 finding 4b — a draw-error frame neither reports drawn nor feeds the
// cost model ────────────────────────────────────────────────────────────────

test('#709: a throwing draw reports skip:"error" and records no settle measurement', () => {
  assert.match(
    src,
    /if \(drewOk\) \{\s*framesDrawn\+\+;[\s\S]{0,200}?lastSkip = "drew";\s*\} else \{\s*lastSkip = "error";\s*\}/,
    "framesDrawn/skip:drew only on a completed frame; the catch path says error",
  );
  assert.match(
    src,
    /if \(drewOk && tFull && !settleHung\) \{\s*lastFullMs = dt;/,
    "the cost-model record is gated on drewOk — a 1-5 ms time-to-throw can no " +
      "longer masquerade as the settled cost (the unbanded-giant-dispatch setup)",
  );
});

// ── #709 finding 7 — wall law honors the morph override ─────────────────────

// ── #724 — interactive hot-path perf (source pins: the pump/draw internals
// only a live GPU reaches; the policy-level convergence behavior is tested
// for real in renderpolicy.test.mjs) ────────────────────────────────────────

test("#724: the interactive budget is fed a FULL-RES prediction, decoupled from the just-sized canvas", () => {
  // The oscillator's mechanism was a feedback loop: sizeCanvas had already
  // multiplied the canvas by last frame's q.scale, predictFullMs() read that
  // canvas, and budgetScale applied the candidate scale² a second time.
  assert.match(
    src,
    /predictedFullMs: predictFullMsAtPx\(devicePx\)/,
    "qualityParams feeds the rect-derived devicePx prediction",
  );
  // The px-explicit model must not read canvas state — that was the loop.
  const fn = src.match(/function predictFullMsAtPx\(px\) \{[\s\S]*?\n {2}\}/);
  assert.ok(fn, "predictFullMsAtPx exists");
  assert.ok(
    !/canvas\./.test(fn[0]),
    "predictFullMsAtPx reads no canvas state",
  );
  // …and the unmeasured fallback is FLAT (scaling the boot guess by a canvas
  // ratio would re-import the same feedback).
  assert.match(fn[0], /return UNMEASURED_MS;/);
  // The canvas-px consumers (settle banding, fence timeouts, backpressure)
  // keep their basis via the thin wrapper.
  assert.match(
    src,
    /function predictFullMs\(\) \{\s*return predictFullMsAtPx\(canvas\.width \* canvas\.height\);\s*\}/,
  );
});

test("#724: q.scale rides the renderpolicy ladder — snapped before the snapshot AND in sizeCanvas", () => {
  // Pump: quantized before lastRender records it (lastRender.scale must be
  // exactly the scale the canvas rendered at — the settle cap and the idle
  // upgrade reconstruct full-res cost as lastFullMs / scale²).
  assert.match(
    src,
    /q\.scale = quantizeScale\(q\.scale\);[\s\S]{0,700}?lastRender = \{/,
    "pump snaps q.scale before the diagnostics snapshot",
  );
  // sizeCanvas re-quantizes defensively for any other caller.
  assert.match(
    src,
    /Math\.min\(window\.devicePixelRatio \|\| 1, DPR_CAP\) \* quantizeScale\(scale\)/,
    "sizeCanvas snaps too",
  );
});

test("#724: the settled accum base stands down when refinement cannot run", () => {
  // A measured-heavy formula (liveAccumCap() === 0) settles via renderer.draw
  // — no rgba32float accum allocation, no full-canvas blend — instead of
  // seeding an average that never gets a second sample. Gated on a
  // measurement OF THIS FORMULA so unmeasured first settles keep refining.
  assert.match(
    src,
    /const refineOff =\s*measuredFor === formula && lastFullPx > 0 && liveAccumCap\(\) === 0;/,
    "refineOff requires a measurement of the live formula",
  );
  assert.match(
    src,
    /const wantAccum =\s*quality === "full" &&\s*!autoRotate &&\s*!q\.cheap &&\s*!refineOff &&\s*!!renderer\.drawAccum;/,
    "wantAccum carries the refineOff guard",
  );
  // Both settle shapes must have a non-accum fallthrough for the skip:
  assert.match(
    src,
    /if \(wantAccum\) renderer\.drawAccum\(\{ skipMarch: true \}\);\s*else renderer\.draw\(\{ skipMarch: true \}\);/,
    "banded settle resolves via draw() when accum is off",
  );
  assert.match(
    src,
    /\} else if \(wantAccum\) \{[\s\S]{0,300}?\} else \{\s*renderer\.draw\(\);/,
    "single-dispatch settle falls through to draw() when accum is off",
  );
  // The settled-frame fence (the cost measurement) survives the skip: it is
  // keyed on tFull OR wantAccum, and tFull does not depend on refineOff.
  assert.match(src, /tFull = quality === "full" && !q\.cheap;/);
  assert.match(src, /\} else if \(tFull \|\| wantAccum\) \{/);
});

test("#724: renderThumbTile caches a blankness verdict beside the URL (no app-side re-decode)", () => {
  assert.match(
    src,
    /thumbCacheSet\(key, \{ url, blank: tileBlankVerdict\(\) \}\);/,
    "the verdict is computed at render time from pixels the engine already has",
  );
  assert.match(
    src,
    /const thumbTileBlank = \(p, look, opts\) => \{/,
    "thumbTileBlank is the public read",
  );
  assert.match(src, /\n    thumbTileBlank,/, "exported on the preview object");
  // The verdict must never over-hide: a failed sample (null) reports NOT
  // blank. The pixel rule itself lives in blankverdict.js (node-pinned there);
  // the sampler is shared with the still-export guard below.
  assert.match(
    src,
    /function sampleBlank\(src\) \{\s*try \{[\s\S]*?return classifyBlank\([\s\S]*?\} catch \{\s*return null;\s*\}\s*\}/,
  );
  assert.match(
    src,
    /function tileBlankVerdict\(\) \{[\s\S]*?return sampleBlank\(thumbCanvas\)\?\.uniform \?\? false;\s*\}/,
  );
});

test("beta black-PNG: stillBlob refuses a dead-device / never-presented capture instead of saving it", () => {
  // The ≤4096 still renders into the LIVE canvas (a resize discards the backing
  // store) and toBlob encodes whatever is there. Pin the guard: the device-lost
  // latch is read either side of the capture, the capture is sampled through
  // the shared sampler, all-zero always fails, opaque uniform black fails only
  // beside a GPU signal, and the error names the size + the remedy.
  const body = src.slice(src.indexOf("async function stillBlob("), src.indexOf("async function stillBlobTiled("));
  assert.match(body, /onFence: \(ok\) => \{\s*if \(!ok\) fenceFailed = true;/, "fence timeouts are gathered, not discarded");
  assert.match(body, /if \(deviceLost\) throw failStill\("failed"\);\s*let blob = await new Promise/, "device-lost checked before the capture");
  assert.match(body, /if \(!blob\) throw new Error\("canvas capture returned null"\);\s*if \(deviceLost\) throw failStill\("failed"\);/, "…and after it");
  assert.match(body, /const blank = await sampleBlankBlob\(blob\);/, "the ENCODED capture is decoded and sampled (a 2D read of the WebGPU canvas is all-zero after toBlob)");
  assert.match(src, /async function sampleBlankBlob\(blob\) \{[\s\S]*?createImageBitmap\(blob\)[\s\S]*?return sampleBlank\(bmp\);[\s\S]*?\} catch \{\s*return null;/, "decode failure never blocks a save");
  assert.match(
    body,
    /blank\.allZero \|\|\s*\(blank\.black &&\s*\(deviceLost \|\| fenceFailed \|\| uncapturedErrors > errsBefore\)\)/,
    "all-zero always fails; opaque black only beside a GPU signal (an empty black view still saves)",
  );
  assert.match(body, /`Export at \$\{W\}×\$\{H\} \$\{what\}: \$\{why\} — try a smaller export size`/, "the error names the size and the remedy");
  assert.match(body, /noteDiag\("still-failed", \{/, "the failure lands in the diag ring");
  // The guard is inside the try whose finally restores the live canvas size.
  const guardAt = body.indexOf("const blank = await sampleBlankBlob(blob);");
  const finallyAt = body.indexOf("} finally {\n      canvas.width = prevW;");
  assert.ok(guardAt > 0 && finallyAt > guardAt, "guard runs under the canvas-restoring finally");
});

test("still export: a save larger than the live canvas trims the bundle cache before AND after", () => {
  const body = src.slice(src.indexOf("async function stillBlob("), src.indexOf("async function stillBlobTiled("));
  assert.match(body, /const biggerThanLive = \(prevW, prevH\) => W \* H > prevW \* prevH;/, "keyed on pixel count, not device class");
  assert.match(body, /const n = renderer\.releaseBundles\?\.\(\);/, "optional — the GL tier has no bundle cache");
  assert.match(body, /if \(big\) trimBundles\("before"\);\s*canvas\.width = W;/, "trimmed before the resize");
  assert.match(body, /canvas\.height = prevH;[\s\S]{0,300}?if \(big\) trimBundles\("after"\);\s*releaseRenderToken\(tok\);/, "…and after the restore, still under the finally");
  assert.match(body, /if \(big\) trimBundles\("before-alpha"\);\s*return await stillBlobAlpha/, "the alpha path gets the same room");
});

test("#709: brakeStop/wallHeadroomAt treat a live morph as deep-tier-unavailable", () => {
  assert.match(
    src,
    /const deepTier =\s*!!formula &&\s*!morph &&/,
    "brakeStop: a melt renders the f32 twin, so the hard stop is the f32 one",
  );
  assert.match(
    src,
    /if \(!morph && ptMode !== "off" && ptElig\(formula\)\)/,
    "wallHeadroomAt: pt clause stands down while morph is set",
  );
  assert.match(
    src,
    /!morph && df64Mode !== "off" && df64Eligible\(formula\)/,
    "wallHeadroomAt: df64 clause stands down while morph is set",
  );
});

test("thumb tiles key their warm-up off cheap-frame features (no df64/perturb), and the batch is counted", () => {
  // A tile draws with q.cheap, and writeFrame drops the df64 + perturbation
  // latches on a cheap frame — so the readiness check / warm-up must NOT read
  // the live latches. Deep-zoomed, that read made every "＋ Add a move" tile
  // warm one specialized variant it never drew with (68 compiles per open).
  assert.match(
    src,
    /df64: over && "df64" in over \? over\.df64 : df64Now\(\)/,
    "frameFeatures honours an explicit df64 override (presence-checked)",
  );
  assert.match(
    src,
    /perturb: over && "perturb" in over \? over\.perturb : ptNow\(\)/,
    "…and an explicit perturb override",
  );
  const body = src.slice(src.indexOf("async function renderThumbTile("), src.indexOf("function beginThumbs("));
  assert.match(body, /const tileFeat = \(\) => frameFeatures\(p, \{ df64: false, perturb: false \}\);/);
  assert.match(body, /!renderer\.marchReadyFor\(tileFeat\(\)\)\) \{\s*const ff = tileFeat\(\);/, "both the check and the warm-up use the tile features");
  assert.doesNotMatch(body, /marchReadyFor\(frameFeatures\(p\)\)/, "the live-latch form is gone");
  assert.match(body, /cheap: true,\s*iterCap,/, "tiles still draw cheap (that is what makes the keying correct)");
  // Telemetry: one `thumbs` diag event per beginThumbs/endThumbs bracket.
  assert.match(src, /function beginThumbs\(\) \{[\s\S]{0,120}?thumbBatchReset\(\);/);
  assert.match(src, /thumbBatchNote\(\);\s*scheduleDraw\(\);\s*\}/, "endThumbs notes the batch before re-arming the pump");
  assert.match(src, /noteDiag\("thumbs", \{\s*tiles: thumbBatch\.tiles,\s*hits: thumbBatch\.hits,\s*compiles: thumbBatch\.compiles,/);
});

test("thumb LITE tier: keyed in the cache, touch-class march budget, finite loose-DE cap", () => {
  assert.match(src, /const thumbKey = \(p, look, opts\) =>\s*JSON\.stringify\(\[[\s\S]{0,400}?!!opts\?\.lite,/, "lite is part of the cache key");
  assert.match(src, /const thumbTileCached = \(p, look, opts\) =>\s*thumbCacheGet\(thumbKey\(p, look, opts\)\)/);
  assert.match(src, /const thumbTileBlank = \(p, look, opts\) => \{\s*const hit = thumbCacheGet\(thumbKey\(p, look, opts\)\);/);
  const body = src.slice(src.indexOf("async function renderThumbTile("), src.indexOf("function beginThumbs("));
  assert.match(body, /const lite = !!opts\?\.lite;/);
  assert.match(body, /\.\.\.\(lite \? \{ dprCap: 1 \} : \{\}\),/, "lite = the touch-class settled budget (140/220 steps)");
  assert.match(body, /hybridLooseDE\(p\)\s*\? lite\s*\? LITE_LOOSE_ITER_CAP\s*: Infinity\s*: 12;/, "a loose base gets a finite cap on lite, Infinity otherwise");
  assert.match(src, /const LITE_LOOSE_ITER_CAP = 24;/);
});

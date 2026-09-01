// §S2 GPU splat G-buffer capture DRIVE — the view loop extracted from
// preview.captureSplatGBuffer (EXPORT_P1_HEADLESS_CAPTURE PR-B). ONE
// implementation for both consumers: preview's captureSplatGBuffer is now a
// thin wrapper (settleFrame + this), and the headless CLI (PR-C) calls it
// directly on a canvas-free renderer. Canvas/DOM-free by construction.
//
// The caller SETTLES first (capturesettle.settleFrame with the still policy +
// itersOverride and NO live bag — capture ignores morph/colorBlend/the df64
// zoom latch by design), then this drive: forces the capture offset (with its
// OWN capK df64 decision — capture geometry, not the live camera's), compiles
// the capture pipeline for the formula's exact feature signature, renders one
// ortho view per submit, and STREAM-REDUCES each view into a bounded
// accumulator (makeStreamingReduce) so peak memory ≈ final splat count, not
// Σ hits (the OOM fix).
//
// Return contract (preserved verbatim from preview — callers rely on it):
//   { noPipeline: true } — the capture pipeline failed to build (validation/
//                          compile error). NOT "no surface": the caller falls
//                          back to the CPU workers, which share DE + framing
//                          and WILL find the surface (#365 honest-toast fix).
//   null                 — the capture RAN and found zero hits, OR onView
//                          returned false (cancelled). Disambiguated only by
//                          the caller's own cancelled flag — deliberate.
//   StreamResult         — the finalized survivor cloud { points, r0, bbox,
//                          sample, stats } (Fractbox frame, pre-mirror). BARE:
//                          no `frame`, no `stats.iters` — the caller holds
//                          those itself (they're its inputs).
//   throws               — mid-capture device failure ⇒ caller falls back too.
import { df64Eligible, lambdaHat, kStarFor } from "./stability.js";
import { fibonacciDir, makeStreamingReduce } from "./splatcapture.js";
import { frameFeaturesFor } from "./capturesettle.js";

export async function driveSplatCapture(renderer, formula, coloring, opts) {
  if (!renderer?.createSplatCapture) return null;
  const {
    iters,
    frame,
    O,
    views = 64,
    res = 256,
    deScale = 1,
    layers = 2,
    aoStrength = 0.5,
    // Stream-reduce knobs (§6): the reduce runs HERE, per view, so the
    // exporter threads its cap + tuning through (radiusScale/aniso match the
    // app's fitSplats consumer; the rest default to reducePoints' defaults).
    cap = 1_500_000,
    radiusScale = 1.6,
    alphaBase = 0.95, // sharp-end opacity (SPLAT_SHARPNESS S-1): slider-driven by the app
    aniso = 0,
    anisoMax = 3,
    sampleN = 0, // >0 ⇒ attach a reservoir subsample (Auto-radius + P0 harness)
    // March safety (§6): a full-step DE march (deScale 1) OVERSHOOTS thin, high-
    // frequency surfaces (kaleido/octaFold stacks) → the capture finds ZERO hits
    // while the live render (conservative policy) shows them. The caller retries
    // with a smaller deScale + more steps when the first pass comes back empty.
    maxSteps = 200,
    onView,
  } = opts;
  // 2. Force the capture offset O (§2.5) — overrides the camera-derived
  //    word. Deep zoom P4: k* for the CAPTURE geometry (ortho pixel size =
  //    2·radius/res against the orbit scale |O|), not the live camera's —
  //    overrideCaptureOffset writes the hi/lo pair + kStar as one unit.
  const capPix = (2 * (frame?.radius ?? 1)) / Math.max(res, 1);
  const capK = df64Eligible(formula)
    ? kStarFor(
        Math.max(Math.hypot(O[0], O[1], O[2]), 1) / Math.max(capPix, 1e-30),
        lambdaHat(formula),
        iters,
      )
    : 0;
  renderer.overrideCaptureOffset(O, capK);
  // 3. Compile the capture pipeline for this formula's exact feature
  //    signature — df64 bit from the CAPTURE decision, not the live latch.
  //    No morphF/blendModeB: capture never carries live morph or colorBlend
  //    (plan §2 captureFeatures), matching the no-live settle the caller ran.
  const session = await renderer.createSplatCapture(
    frameFeaturesFor(formula, coloring, { df64: capK > 0 }),
    res,
  );
  // No capture pipeline (validation/compile error) is NOT "no surface": return a
  // distinct sentinel so the caller falls back to CPU (which shares DE + framing
  // and WILL find the surface) instead of misreporting an empty capture. A plain
  // null from here on means the capture RAN and found zero hits (or was cancelled).
  if (!session) return { noPipeline: true };
  // 4. One view per submit + mapAsync ⇒ the compositor breathes between views
  //    (naturally banded, per the settle-banding lesson). MERGE each view's
  //    hits into the streaming accumulator and discard the view immediately —
  //    no whole-cloud concat (that + the caller's Float32Array.from tripled
  //    peak memory and OOM'd big exports). GPU hits are world coords (the pass
  //    adds O back), so frame.center is a valid fixed grid origin.
  const reduce = makeStreamingReduce({
    frame,
    cap,
    radiusScale,
    alphaBase,
    aniso,
    anisoMax,
    sampleN,
    views,
  });
  for (let k = 0; k < views; k++) {
    let v;
    try {
      v = await session.captureView(fibonacciDir(k, views), frame, O, res, {
        deScale,
        layers,
        aoStrength,
        maxSteps,
      });
    } catch (e) {
      session.dispose();
      throw e; // device lost / mapAsync reject ⇒ caller falls back to CPU
    }
    reduce.addChunk(v.pos, v.normal, v.albedo);
    v = null; // free this view's raw hits before the next submit
    if (onView && onView(k + 1, views) === false) {
      session.dispose();
      return null; // cancelled — no fallback
    }
  }
  session.dispose();
  // The finalized survivor cloud (survivors ≤ cap) + running bbox + reservoir
  // sample — the caller consumes these directly (no worldBBox/takeMetricsSample
  // pass over a raw cloud that no longer exists). null ⇒ zero hits.
  return reduce.finalize();
}

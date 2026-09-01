// Render policy — the pure decision layer extracted from preview.js: which
// bailout a formula needs, the deep-zoom near/far constants, the iteration
// auto-detail law, the interactive/settled quality tiers, the CSG scene march
// scale, and the cheap-tier shading override. No DOM, no GPU, no closure state
// — every input is explicit, so this layer is unit-testable in Node while
// preview.js (which owns the canvas, camera, and pump) just feeds it its
// current state. cpu.js shares TNEAR_K/TFAR_K too (its traceGrid used to carry
// a fourth hand-typed 0.02/24 · 80/24 copy).

import { isEscapeTime, isNumericDE, isApproxDE, byKey } from "./operators.js";
import { looseDE, hybridLooseDE } from "./stability.js";
import { MAX_ITERS, BAILOUT_ESCAPE, BAILOUT_IFS } from "./limits.js";
import { leafById } from "./leaves.js";
import { normalizeSceneObject } from "./sceneobj.js";
// The camera's own absolute dist clamp — the deepest distance any tier allows.
// camera.js imports nothing, so this cannot cycle.
import { PT_MIN_DIST } from "./camera.js";

// Capability tier (docs/planning/CAPABILITY_PROBE.md) — a MACHINE class derived
// from static, zero-cost signals available the moment the backend is chosen
// (adapter identity + host), NOT a per-formula predictor. Pure + GPU-free so it's
// unit-testable in Node and shared by preview.js (classify at boot) and the
// analytics beacon (report to /stats for threshold calibration). This slice only
// OBSERVES the class; acting on it (cheaper boot variant for 'slow', sharper
// first frame for 'fast') is a separate change through the soak ladder. The
// thresholds below are best-guess starting points — the fleet /stats panel is
// what calibrates them, which is the whole point of collecting the data first.
export function classifyTier({
  backend = "none",
  vendor = "",
  architecture = "",
  description = "",
  isSoftware = false,
  cores = 0,
} = {}) {
  if (backend === "none") return "ascii"; // no GPU at all → CPU/ASCII floor
  const s = `${vendor} ${architecture} ${description}`.toLowerCase();
  // Software rasterizers (WARP/llvmpipe/SwiftShader) — the march runs
  // catastrophically slow; mirror renderer.js's software-adapter regex.
  if (isSoftware || /warp|basic render|software|llvmpipe|swiftshader/.test(s))
    return "software";
  // Apple silicon (M-series): fast unified-memory GPUs, quick shader compiles.
  if (/apple|\bm[1-9]\b/.test(s)) return "fast";
  // Known SLOW SHADER COMPILERS (renderer.js:107): Nvidia Pascal (GTX 10-series)
  // on D3D12 compiles the march shader in 2–55 s. Also treat ≤2 logical cores as
  // a low-end proxy. These want the cheapest boot variant (acted on later).
  if (/pascal|geforce gtx 10|gtx 10[0-9]0|\b10[5-8]0\b/.test(s)) return "slow";
  if (cores && cores <= 2) return "slow";
  return "medium";
}

// Phase 2a (docs/planning/DEFERRED_FORMULA_SWAP.md) — the boot-time "no real
// measurement yet" seed and the boot-variant-race gate. Both are pure
// functions of `tier` alone so they're Node-testable without a GPU; preview.js
// calls them once `capability.tier` is known.
//
// bootPredictMs: preview.js's predictFullMs() falls back to this when no
// settled frame has been measured for the current formula yet (UNMEASURED_MS).
// Only 'fast' seeds lower — a wrong "assume light" guess on anything less than
// confidently fast risks under-banding a heavy formula's first settle (the
// exact freeze this fallback exists to prevent); medium/slow/software/
// unclassified keep the original conservative default. Never touches
// lastFullMs itself — measuredFor's hard cutover to a real per-formula
// measurement is unaffected (CAPABILITY_PROBE.md's resolved Q1).
export const BOOT_PREDICT_MS_DEFAULT = 1600;
export const BOOT_PREDICT_MS_FAST = 700;
export function bootPredictMs(tier) {
  return tier === "fast" ? BOOT_PREDICT_MS_FAST : BOOT_PREDICT_MS_DEFAULT;
}

// shouldRaceGeneralAtBoot: whether the boot hero should ALSO kick the GENERAL
// variant's async prewarm alongside its usual SPECIALIZED attempt (renderer.js
// already prefers general-if-warm and marchReadyFor already ORs both keys, so
// whichever lands first wins for free — no new gating logic needed on the
// caller's side). Deliberately excludes 'slow'/'software': a spec review found
// racing two concurrent compiles risks GPU-process/driver-JIT contention that
// a slow-compiler machine can least afford (and could even trip the unrelated
// markSlowGpu() heuristic as a false positive), while a blind "just try
// general" fallback for that tier can regress a returning user whose
// SPECIALIZED variant was already disk-cached from a prior session. Neither
// risk applies where compiles are cheap anyway, so the race is scoped to
// exactly the tiers where it's low-risk; slow/software keep today's unchanged
// single-specialized-attempt boot path (see DEFERRED_FORMULA_SWAP.md Phase 2a).
export function shouldRaceGeneralAtBoot(tier) {
  return tier === "fast" || tier === "medium";
}

// Escape-time power maps need the small bailout or rᵖᵒʷᵉʳ overflows fp32; IFS
// folds stay bounded so a huge radius is harmless (values live in limits.js).
export const bailoutFor = (f) =>
  isEscapeTime(f) || isNumericDE(f) ? BAILOUT_ESCAPE : BAILOUT_IFS;

// Deep zoom (§5): the default framing distance the old fixed near/far (0.02/80)
// were tuned for — near/far now scale off cam.dist so a deep zoom doesn't start
// inside the surface (near too far) or get far-capped against nearby geometry
// (far too close). Also the HUD's magnification reference (M = REF_DIST/dist).
export const REF_DIST = 24;
export const TNEAR_K = 0.02 / REF_DIST;
export const TFAR_K = 80.0 / REF_DIST;
// The magnification law itself — ONE source of truth for "how far in are we",
// shared by the depth-driven control laws below and the app's zoom badge.
//
// It exists because the same absolute floor was hand-typed at each call site
// and went stale when a tier deepened the reachable range. #480 lifted the two
// floors in this file (1e-12 → the pt-era floor) but the app's badge kept its
// own copy, so every descent past dist 1e-12 SATURATED at REF_DIST/1e-12 =
// ×2.4·10¹³ — the camera kept going (the pt tier reaches ×10³⁰, verified on
// the wheel path), the readout just stopped counting, and the deep-zoom
// feature looked hard-pinned. The floor is PT_MIN_DIST, the camera's own
// absolute clamp (camera.js clampDist), so the badge can express every
// distance the camera can legally hold and can never divide by zero; it is a
// guard against 0/negative/non-finite input, NEVER a tier's wall (the wall is
// the brake's job — recenter.js zoomHeadroom/ptHeadroom).
export function magnificationFor(dist) {
  return REF_DIST / Math.max(Number(dist) || 0, PT_MIN_DIST);
}
// Deep zoom P4 field fix: the dist-proportional far plane is a savage clip
// at depth — at zoom ×4.6·10³ on the Mandelbulb it sat at 7·10⁻³ world
// units while the bulb spans ~2, and a CPU-f64 probe measured 26% of the
// frame hitting REAL geometry beyond it (rendered black — "disappears to
// black" report). Floor tFar at the object scale so the far field never
// vanishes; sphere tracing with relative eps costs only LOG in the t-range,
// so the floor is cheap. 8 covers every shipped preset's extent (unbounded
// scenes additionally keep their ×4 multiplier).
export const TFAR_MIN = 8.0;

// Unbounded scenes — plane/lattice/heightfield leaves (leaves.js `unbounded`)
// left uncut by an intersect: grazing rays cross far more surface than a
// framable object, so the default march budget starves (hole artifacts,
// floating slivers, geometry "clipped" at grazing angles) and the
// dist-scaled tFar chops the field mid-view. Both knobs widen when the scene
// is judged unbounded; bounded scenes pay nothing. The judgment folds the
// combine chain exactly like the marcher: union/smooth widen the extent,
// intersect bounds it unless BOTH sides are unbounded, subtract keeps the
// left side's extent. Muted objects are skipped (they don't render).
export const TFAR_UNBOUNDED_MUL = 4;
export const STEPS_UNBOUNDED_MUL = 2;
export function sceneUnbounded(objects) {
  let unb = false;
  (objects || []).forEach((o, i) => {
    if (!o || o.muted) return;
    const leafUnb = !!leafById(normalizeSceneObject(o).shapeId)?.unbounded;
    const c = i === 0 ? 0 : (o.combine ?? 0) & 3;
    if (c === 3) unb = unb && leafUnb;
    else if (c !== 2) unb = unb || leafUnb;
  });
  return unb;
}
export const unboundedScene = (f) =>
  !!(f && f.objects && sceneUnbounded(f.objects));

// Deep zoom §6 (Phase 3) — depth-adaptive step budget bounds. STEP_CEIL must not
// exceed the GLSL literal loop cap (shader_gl.js's `for (int i = 0; i < 512; …)`,
// both fragment shaders) — pushing past it needs raising that literal (a shader
// recompile), out of scope for v1. DEPTH_CAP (×3 steps / ÷√3 deScale at the
// deepest) is a starting point tuned against fps on the Mac mini reference GPU;
// revisit if deep zoom feels sluggish or thin features still drop out.
export const STEP_CEIL = 512;
export const DEPTH_CAP = 3;
// Auto-detail (§6) — a distance-estimated fractal has a FIXED finest scale for a
// given iteration count, so zooming past it just smooths out (the DE becomes a
// bound, not the surface). Raise the iteration count as you zoom in — roughly one
// extra iteration per zoom octave (log2 of magnification), the natural fractal
// law — so fine structure keeps resolving. ITER_CEIL is the GLSL literal loop cap
// (shader_gl.js interpolates limits.js MAX_ITERS); WGSL's bound is a uniform, but
// we clamp to the same value for cross-backend parity. Iteration count is the
// dominant DE cost, so this is the opt-out "Detail with zoom" toggle, not an
// always-on cost.
export const ITER_CEIL = MAX_ITERS;
export const ITER_PER_OCTAVE = 1.0;

// The auto-detail law itself, stated in terms of a bare MAGNIFICATION rather
// than a camera distance — because not every consumer has a camera. The splat
// capture zooms by FRAMING (a user-drawn capture volume, or the S-5a view
// frame): a box 1/8.7 the object's size is a ×8.7 zoom that no `cam.dist`
// describes. Extracted so effectiveIters below and the capture path
// (splatcapture.js cropCaptureIters) apply ONE law and cannot drift apart —
// #496 was in the end exactly that drift, the export having lost its link to
// this law when #415 removed `iters: views.effectiveIters()`.
//
// `mag <= 1` (or non-finite) ⇒ the base count, unchanged: zoomed OUT, or not
// zoomed at all, needs no extra iterations. That is what makes every caller
// inert on a whole-object framing.
export function itersForMagnification(base, mag) {
  const b = base || 0;
  const m = Number(mag);
  if (!Number.isFinite(m) || m <= 1) return b;
  const extra = Math.max(0, Math.round(Math.log2(m) * ITER_PER_OCTAVE));
  return Math.min(ITER_CEIL, b + extra);
}

// Auto-detail (§6): the iteration count actually sent to the shader — the
// formula's base iters, plus ~one per zoom octave once zoomed in, clamped to
// ITER_CEIL. At the default framing (M≈1) the boost is 0, so nothing changes
// until you zoom. autoDetail off → the base count, exactly as before.
// detailOverride: the user dragged the Detail slider to an ABSOLUTE count —
// honor it verbatim (2..ITER_CEIL) so they can go BELOW auto-detail's floor.
export function effectiveIters(
  base,
  { dist = REF_DIST, autoDetail = true, detailOverride = null } = {},
) {
  const b = base || 0;
  if (detailOverride != null)
    return Math.min(ITER_CEIL, Math.max(2, detailOverride));
  if (!autoDetail) return b;
  const M = magnificationFor(dist); // scale-free (#480: a hand-typed 1e-12 floor here froze auto-detail/steps below x10^13, "detail stuck at 55")
  return itersForMagnification(b, M);
}

// Interactive pixel budget (#212): the interactive tiers used a FIXED scale
// (0.8×/0.7×) — on a 5K display that's still ~8.7 Mpx per interactive frame,
// ~1 s of GPU on a mid formula, and a dispatch that long starves Chrome's
// compositor (the whole browser freezes, not just this tab). Cap the
// interactive tiers at a fixed device-pixel budget instead: laptop-class
// canvases are unaffected (0.8× of a 2560×1600 canvas is already under it);
// huge displays interact at the same absolute cost as a laptop and pop to
// full resolution on settle exactly as before. Settled frames are unclamped
// — their dispatch size is bounded by BANDING (preview.js settle loop).
export const INTERACT_PX_BUDGET = 4.2e6;

// Interactive TIME budget (#212, part 2): the pixel budget alone still lets a
// heavy formula freeze — a loose-DE preset's balanced-tier frame (full march
// budget, 0.8× res) is ~900 ms of GPU on a laptop canvas, and those frames
// pipeline unfenced during drags/load-tweens. Clamp the interactive scale so
// the PREDICTED frame cost (the settle measurement, scaled by resolution and
// a cheap-shading discount) stays near the budget. Cheap scenes never clamp;
// heavy scenes drag chunky-but-live and pop sharp on settle — the never-
// freeze bias. The floor keeps the picture legible even for extreme scenes.
export const INTERACT_MS_BUDGET = 100;
const INTERACT_SCALE_FLOOR = 0.25;
const CHEAP_SHADE_DISCOUNT = 0.6; // interactive tiers skip shadow/AO marches
const budgetScale = (scale, devicePx, predictedFullMs) => {
  let s =
    devicePx > 0
      ? Math.min(scale, Math.sqrt(INTERACT_PX_BUDGET / devicePx))
      : scale;
  if (predictedFullMs > 0) {
    // predicted cost of THIS tier's frame ≈ settled ms × scale² × discount
    const tierMs = predictedFullMs * s * s * CHEAP_SHADE_DISCOUNT;
    if (tierMs > INTERACT_MS_BUDGET)
      s = Math.max(
        INTERACT_SCALE_FLOOR,
        s * Math.sqrt(INTERACT_MS_BUDGET / tierMs),
      );
  }
  return s;
};

// The march-quality knobs {scale, steps, eps, deScale, cheap?} for a formula in
// the current interaction state.
//   quality      'full' (settled) | 'low' (interacting)
//   moveQuality  #32 — render tier while moving: 'smooth'|'balanced'|'full'
//   isTouch/dprCap — device class (touch renders at a lower supersample)
//   dist         cam.dist, the deep-zoom depth input (§6 Phase 3)
//   devicePx     the UNSCALED canvas device-pixel count (w×h at DPR, scale 1) —
//                feeds the interactive pixel budget above; 0/absent = no clamp
//   predictedFullMs  the caller's estimate of a settled frame's cost at this
//                resolution (preview.js measures it per formula) — feeds the
//                interactive time budget; 0/absent = no clamp
export function qualityParams(
  f,
  {
    quality = "full",
    moveQuality = "balanced",
    isTouch = false,
    dprCap = 2,
    dist = REF_DIST,
    devicePx = 0,
    predictedFullMs = 0,
  } = {},
) {
  // Settled (full) quality — extra march steps + a tight step size so the
  // marcher doesn't overstep THIN SURFACES (Amazing Surf etc.) at grazing
  // angles. A LOOSE analytic DE (scale < 2, see stability.looseDE) over-
  // estimates distance, so it needs a tighter step (0.3) + more steps or it
  // flies past its thin surface and renders blank when still (#14). Only
  // loose-DE formulas pay that cost.
  // Hybrid iteration (§3.4) — union rule: loose if EITHER slot's own ops
  // would be loose alone (hybridLooseDE falls back to plain looseDE when
  // f.hybrid is absent, so this is a strict superset of the old check).
  const loose = !!(f && hybridLooseDE(f));
  const full = loose
    ? { scale: 1.0, steps: dprCap < 2 ? 220 : 320, eps: 0.001, deScale: 0.3 }
    : {
        scale: 1.0,
        steps: dprCap < 2 ? 140 : 200,
        eps: 0.001,
        deScale: 0.5,
      };
  // Deep zoom §6 (Phase 3) — depth is an INPUT to the same {steps, deScale}
  // knobs, not a new axis: deeper zoom needs more march steps and a tighter
  // step size to resolve thin deep features (the reconstruction keeps eps·t
  // sane since t is residual-scaled — §3). Single-object path only in v1;
  // CSG scenes use sceneDeScale, a separate knob, and aren't recentered (§14).
  if (f && !f.objects) {
    const M = magnificationFor(dist); // scale-free (#480: a hand-typed 1e-12 floor here froze auto-detail/steps below x10^13, "detail stuck at 55")
    const depth = Math.min(1 + Math.log10(Math.max(M, 1)) * 0.35, DEPTH_CAP);
    if (depth > 1) {
      full.steps = Math.min(STEP_CEIL, Math.round(full.steps * depth));
      full.deScale = Math.max(0.25, full.deScale / Math.sqrt(depth));
    }
  }
  // Approximate-DE (APPROX_DE.md §3): a deApprox op's analytic DE is not a
  // true bound — halve the step AFTER the depth clamp (approx floor 0.125)
  // and double the march budget so deep thin surfaces stay reachable. All
  // tiers below spread from `full`, and the smooth tier applies it itself.
  const approx = !!(f && isApproxDE(f));
  if (approx) {
    full.deScale *= APPROX_DESCALE_MUL;
    full.steps = Math.min(STEP_CEIL, Math.round(full.steps * 2));
  }
  // Unbounded scenes (see sceneUnbounded): grazing rays over an infinite
  // field need a bigger budget or they starve into sky holes. The tangent
  // band (rays skimming the field toward an object behind it) consumes
  // ~deScale-sized steps for many units, so the settled tier goes straight
  // to the ceiling — anything less leaves reach-limit slivers at near-level
  // pitch. Only unbounded scenes pay; interactive tiers derive from this.
  if (unboundedScene(f))
    full.steps = Math.max(
      Math.min(STEP_CEIL, Math.round(full.steps * STEPS_UNBOUNDED_MUL)),
      STEP_CEIL,
    );
  if (quality !== "low") return full;
  // Interactive (while orbiting/zooming). Which tier depends on the user's
  // "while moving" preference (#32 — the moving-vs-still shift was distracting):
  //   full     — no drop; geometry identical to still (heaviest).
  //   balanced — keep the full MARCH BUDGET (steps/deScale/eps) so the SHAPE
  //              stays consistent; drop only RESOLUTION. This kills the swimming
  //              "cutting plane" (a too-small step budget runs the ray out before
  //              it reaches far surfaces) and the thin-surface dropouts, at the
  //              cost of softer pixels while you drag.
  //   smooth   — the cheap coarse tier: low res + few steps + loose step/eps.
  //              Max framerate on weak GPUs, but the shape visibly shifts.
  if (moveQuality === "full") return full;
  if (moveQuality === "smooth")
    return {
      // smooth's few-steps march is far cheaper than the settled frame the
      // prediction measures — no time clamp, just the pixel budget.
      scale: budgetScale(isTouch ? 0.6 : 0.7, devicePx, 0),
      steps: approx ? 96 : 48,
      eps: 0.003,
      deScale: approx ? 0.65 * APPROX_DESCALE_MUL : 0.65,
      cheap: true, // P1: drop shadow + AO marches while dragging (word flip, no recompile)
    };
  // balanced (default): full march budget (shape-stable), lower resolution,
  // and CHEAP SHADING — shadow/AO are shading, not shape, and their marches
  // dominate frame cost on heavy scenes (2 fps on an 8-object CSG while a
  // light slider drags). The shading pop on settle coincides with the
  // existing resolution pop.
  return {
    ...full,
    scale: budgetScale(isTouch ? 0.7 : 0.8, devicePx, predictedFullMs),
    cheap: true,
  };
}

// ASCII/CPU motion tier (#32). The ASCII view runs its own interactive tier —
// `ascii.ts` drops the COLUMN COUNT while you drag and pauses the edge/
// structure/dither overlay passes — but it was hardcoded and single-level, so
// the "Quality while moving" preference did nothing there (the control was
// disabled outright). #32's reporter asked for the setting to apply to the
// ASCII view too; this is the ASCII half of qualityParams' three-rung ladder.
//
// THE ONE INVARIANT: no rung touches the ASCII MARCH (maxSteps / deScale / eps
// / iters live in cpu.js traceGrid and are never a function of motion). So the
// ASCII tier can only ever change SAMPLING DENSITY and GLYPH VOCABULARY —
// never the surface the rays find. That is strictly stronger than the GPU
// ladder, where `smooth` genuinely does move the geometry, and it is the point:
// the fix for "while moving I see a different result" must not import the very
// defect it is fixing.
//
//   colScale — multiplier on the chosen column count. The FLOOR is deliberately
//              NOT returned here: it is ascii.ts's own COLS[0], which this
//              module cannot see, and duplicating the literal would silently
//              break the balanced-is-unchanged guarantee the day that ladder
//              changes. ascii.ts applies Math.max(COLS[0], …) itself.
//   fx       — whether the edge / structure / dither overlay passes run. These
//              are extra per-cell orbit work, so they are the ASCII analogue of
//              qualityParams' `cheap` (which drops the shadow/AO marches).
//   spinCoarse — whether AUTO-ROTATE renders on the motion tier. Deliberately
//              NOT plain GPU parity (preview.js puts every autoRotate frame in
//              the interactive tier): an 0.8x GPU render-scale drop is nearly
//              imperceptible, but an 0.6x COLUMN drop turns a 80x30 grid of
//              contoured detail into a 48x18 blob, and putting the default
//              showcase spin behind that would introduce a brand-new, highly
//              visible moving-vs-still difference — regressing #32. But a weak
//              machine must not be stuck with an unbounded ~600 ms/frame
//              main-thread block on spin with no way out, so the opt-out lands
//              on `smooth`, the rung whose whole documented purpose is "max
//              framerate, accept the visible drop".
export const ASCII_MOVE_COL_SCALE = 0.6; // balanced — today's literal
export function asciiMoveParams(moveQuality = "balanced") {
  if (moveQuality === "full")
    return { colScale: 1, fx: true, spinCoarse: false };
  if (moveQuality === "smooth")
    return { colScale: 0.4, fx: false, spinCoarse: true };
  return { colScale: ASCII_MOVE_COL_SCALE, fx: false, spinCoarse: false };
}

// Offline still / frame-export march budget (#281/#282/#283). A saved PNG
// (preview.stillBlob) or an exported video frame (captureFrame) must render at
// the SAME effective march quality as the on-screen SETTLED view, or "saved ==
// what you see" breaks. The single source of truth is the settled (full)
// policy — steps, eps AND deScale all come straight from qualityParams, not
// hand-picked literals at the call site.
//
// The bug this centralizes out: the export paths kept the policy's steps/deScale
// but substituted a TIGHTER eps (0.0006 vs the settled 0.001). A tighter surface
// epsilon reads as "more detail", but against the SAME step budget it starves
// the grazing rays that skim a surface tangentially — and for the deApprox
// bounded TPMS leaves (gyroid/schwarzP/lidinoid/scherk, d = max(field, |p|−R))
// the entire outer silhouette IS a tangent band. Those rays never reached the
// tighter threshold within maxSteps, dropped to background, and the saved image
// eroded a ring inward of the live silhouette (the gyroid "clipping", the
// lidinoid/scherk "different shape"). eps = q.eps keeps them identical.
//
// dprCap 2 → the desktop supersample tier (the still renders at device px). The
// caller supplies the live camera depth so deep-zoom boosts carry through.
export function stillQualityParams(f, dist = REF_DIST) {
  const q = qualityParams(f, { quality: "full", dprCap: 2, dist });
  return {
    steps: q.steps,
    eps: q.eps,
    deScale: q.deScale ?? resolveDeScale(0.5, f),
  };
}

// Export AA sample count (#save-latency). A PNG still accumulates N jittered
// full-frame marches for offline-grade anti-aliasing — but each sample is a
// full march, so N is the direct multiplier on save time. A HEAVY formula
// (one whose live settled view never accumulates DOF — accumCap()==0, the
// same signal stillBlob already uses to gate the lens offsets to pure AA)
// pays 7.5–15 s for the full 24 samples, exactly the saves that hurt.
//
//   mode 'adaptive' (default): heavy → EXPORT_SAMPLES_HEAVY (8, ~3× faster
//       with slightly softer AA — pure AA grain, no shape/DOF change since a
//       heavy formula's samples are already lens-centered); light → the full
//       count (light saves are fast and converge DOF, so keep them pristine).
//   mode 'full': always EXPORT_SAMPLES_FULL, regardless of weight.
//
// Pure (no GPU/DOM) so the decision is unit-testable in Node; preview.js feeds
// it {heavy: accumCap()==0, mode: opts.aaMode} and threads the mode in via the
// stillBlob opts so core stays pref-agnostic (the app owns the pref).
export const EXPORT_SAMPLES_FULL = 24;
export const EXPORT_SAMPLES_HEAVY = 8;
export function exportSampleCount({ heavy = false, mode = "adaptive" } = {}) {
  if (mode === "full") return EXPORT_SAMPLES_FULL;
  return heavy ? EXPORT_SAMPLES_HEAVY : EXPORT_SAMPLES_FULL;
}

// PNG-still export dimensions (EXPORT_SIZE picker). Two modes:
//   • explicit {width,height} override — deterministic output independent of
//     the viewport shape. BOTH are required together; each dimension is clamped
//     INDEPENDENTLY to STILL_PX_CAP (per-dimension, so a 4096×4096 square is
//     legal — this is NOT the long-edge clamp).
//   • no override — derive from the on-screen canvas rect × dpr and clamp the
//     LONG EDGE to the same cap. This is the legacy "saved == what you see"
//     path and is kept byte-identical.
// Pure (no GPU/DOM) so preview.js's export sizing is unit-testable in Node.
export const STILL_PX_CAP = 4096;
export function resolveStillDims({
  width,
  height,
  rectW,
  rectH,
  dpr,
  cap = STILL_PX_CAP,
} = {}) {
  if (width && height) {
    return {
      W: Math.max(1, Math.min(cap, Math.round(width))),
      H: Math.max(1, Math.min(cap, Math.round(height))),
    };
  }
  let W = Math.round((rectW || 4) * dpr);
  let H = Math.round((rectH || 3) * dpr);
  const longEdge = Math.max(W, H);
  if (longEdge > cap) {
    const s = cap / longEdge;
    W = Math.max(1, Math.round(W * s));
    H = Math.max(1, Math.round(H * s));
  }
  return { W, H };
}

// Banded-march band geometry (#514). Band `i` of `n` covers rows [y0, y1) of a
// target `h` rows tall; the caller scissors exactly that strip. The ONE source
// of band geometry — the renderer must never re-derive it from a cached or
// live-canvas height, because a still export renders at dimensions that differ
// from the on-screen canvas (EXPORT_SIZE, #513) and a band grid keyed off the
// wrong height would leave strips unmarched or scissor them off the target.
// Contract (pinned by renderpolicy.test.mjs): the n bands TILE [0, h) exactly —
// no gap, no overlap, band n-1 always closes on h. Degenerate slices (n > h)
// come back empty (y1 === y0) and the caller skips them.
// Pure (no GPU) so the tiling is unit-testable in Node.
export function bandRect(i, n, h) {
  const rows = Math.max(0, Math.floor(h));
  const bands = Math.max(1, Math.floor(n));
  const idx = Math.min(Math.max(0, Math.floor(i)), bands - 1);
  const y0 = Math.floor((rows * idx) / bands);
  // The LAST band closes on h rather than on its floored share, so rounding
  // never leaves a sliver of un-marched rows at the bottom of the frame.
  const y1 = idx === bands - 1 ? rows : Math.floor((rows * (idx + 1)) / bands);
  return { y0, y1, h: Math.max(0, y1 - y0) };
}

// Approximate-DE step policy (APPROX_DE.md §3): the single multiplier a
// deApprox formula's deScale pays, and the shared resolver every deScale
// producer routes through — qualityParams above, sceneDeScale below, the
// preview.js literals, and cpu.js's own ASCII-tier default (the one tier that
// bypasses this module's qualityParams entirely).
export const APPROX_DESCALE_MUL = 0.5;
export const resolveDeScale = (base, f) =>
  f && isApproxDE(f) ? base * APPROX_DESCALE_MUL : base;

// CSG Phase 1a — scene deScale is one global marcher parameter, so the scene
// marches at the TIGHTEST child's value: min over objects. A box primitive is
// exact (0.5); an IFS object is loose (0.3) iff its sub-formula's looseDE fires.
//
// CARVE_DESCALE (subtract/intersect): max(a,b) and max(a,-b) are NOT valid
// distance bounds for the result — they OVER-estimate distance, so the
// sphere-marcher oversteps the carved cavity walls and punches holes/dropouts.
// Union (min) is safe; subtract/intersect aren't. Mitigation (spec §3.3):
// march more conservatively whenever any object carves.
//
// Empirically tuned on the real GPU (puppeteer + ANGLE/Metal WebGPU & WebGL2,
// reading back the canvas for hole pixels):
//   - PRIMITIVE carves (box−sphere bite, box∩sphere) render clean even at 0.5
//     — the analytic SDFs are exact, so the max() overstep stays bounded and
//     the marcher's adaptive eps still catches the walls.
//   - a FRACTAL carve (Menger sponge − sphere) is the real stress: at 0.5/0.35
//     the carved face is grainy with speckle and a dropout slot; at 0.25 it
//     resolves crisp and solid (WebGL2 readback: 0 hole pixels). The approximate
//     IFS DE compounds the max() over-estimate, so it needs the tighter step.
// Settled on 0.25 — crisp on the fractal carve, clean on primitive carves, with
// margin. ONLY carving scenes pay it; a pure-union scene keeps 0.5 (no slowdown).
export function sceneDeScale(objects) {
  const CARVE_DESCALE = 0.25;
  let m = 0.5;
  let carving = false;
  let anyApprox = false;
  for (const o of objects) {
    // D0 (PRIMITIVE_DIFS_D0 §2.7): the gate is a function of the compiled DE's
    // COMPONENTS, not the retired op-chain-XOR-shape split — so judge the SAME
    // canonical form the emitters compile (normalizeSceneObject: active-op
    // slice, legacy pure shapes drop stray ops, muted ops excluded). A mixed
    // object is as loose as its chain; a pure leaf is exact; iterShape adds
    // nothing (min of the same per-iteration bounds).
    const n = normalizeSceneObject(o);
    const chainLoose =
      n.ops.length > 0 &&
      looseDE({
        ops: n.ops,
        deOption: n.deOption,
        iters: n.iters,
      });
    m = Math.min(m, chainLoose ? 0.3 : 0.5);
    if (n.combine === 2 || n.combine === 3) carving = true; // subtract / intersect
    // Approximate components: an approx op anywhere in the chain, or a leaf
    // whose own bound is approximate (leaves.js deApprox — heightfield/Taubin
    // D2 leaves; the 6 launch leaves are exact).
    if (n.ops.some((op) => byKey(op.key)?.deApprox)) anyApprox = true;
    if (leafById(n.shapeId)?.deApprox) anyApprox = true;
  }
  if (carving) m = Math.min(m, CARVE_DESCALE);
  // Approximate-DE (APPROX_DE.md §3): scene deScale is ONE global knob, so an
  // approx component in ANY object tightens the whole scene — same shape as
  // the carve rule, and the same known v1 conservatism for mixed scenes.
  if (anyApprox) m *= APPROX_DESCALE_MUL;
  return m;
}

// P1 — the cheap interactive tier renders shadowless / AO-less by overriding
// the two light words (the shader skips both marches entirely at 0). The
// settled frame passes the coloring's light through untouched.
// #630 — reflBounces rides the same override: a cheap frame renders the plain
// (mirror-off) variant — deriveFrameParams sees bounces 0 and the sreflect
// latch drops, so drags don't pay the bounce marches and the settle pops the
// mirrors back in, exactly the shadows/AO precedent (spec review 2g option i).
export function shadeLight(q, light) {
  if (!q?.cheap) return light;
  return { ...light, shadow: 0, ao: 0, reflBounces: 0 };
}

// ── Static formula cost + per-frame GPU governor (#476) ────────────────────
// A passively-cooled mobile GPU dies not from ONE slow frame but from SUSTAINED
// heavy ones: the 2026-08-01 iPad death was a FULL-tier settle holding scale 1
// at ~950 ms/frame for ~7 min until the OS evicted the GPU process — UNDER the
// desktop settle budget (SETTLE_MS_BUDGET = 1200 ms), so nothing capped it, and
// the device classified "fast" (adapter reports "apple", identical to a Mac).
// These pure helpers give preview.js two composable levers, both Node-testable
// without a GPU: a reactive scale governor keyed off the MEASURED settled ms,
// and a cost-aware STARTING-detail clamp for the first (unmeasured) frames.

// Scheduled op-EVALUATIONS per (super-)iteration — the per-ray-step work a
// formula pays. Duck-typed across THREE shapes so it already counts the N-slot
// stack (#488/#491) correctly the day it lands, WITHOUT importing from that
// unmerged branch:
//   flat          → f.ops.length
//   legacy 2-slot → f.ops (slot A ×schedule.a) + f.hybrid.b.ops (×schedule.b)
//   N-slot        → f.hybrid.slots[{ops}…incl A] ×f.hybrid.schedule.counts[]
// Muted slots (b.muted / aMuted / slot.muted) contribute 0 — a muted slot is
// never emitted, so it is never marched.
export function opEvalsPerIter(f) {
  if (!f || typeof f !== "object") return 0;
  const opCount = (o) => (Array.isArray(o?.ops) ? o.ops.length : 0);
  const h = f.hybrid;
  if (h && typeof h === "object") {
    // Future N-slot shape: slots[] carries EVERY slot (incl. A); counts[] parallel.
    if (Array.isArray(h.slots) && h.slots.length) {
      const counts = Array.isArray(h.schedule?.counts) ? h.schedule.counts : [];
      return h.slots.reduce((sum, slot, i) => {
        if (!slot || slot.muted) return sum;
        const c = Number.isFinite(counts[i]) ? Math.max(0, counts[i]) : 1;
        return sum + opCount(slot) * c;
      }, 0);
    }
    // Legacy 2-slot: slot A = f.ops (top-level), slot B = h.b.ops.
    const a = Number.isFinite(h.schedule?.a) ? Math.max(0, h.schedule.a) : 1;
    const b = Number.isFinite(h.schedule?.b) ? Math.max(0, h.schedule.b) : 1;
    const aOps = h.aMuted ? 0 : opCount(f);
    const bOps = h.b?.muted ? 0 : opCount(h.b);
    return aOps * a + bOps * b;
  }
  return opCount(f);
}

// A STATIC cost score ≈ op-evals/iter × iterations × pixels — the task's simple
// model. `pixels` defaults to 1 so the score is the intrinsic per-ray march work
// (the entry clamp keys off the op-evals×iters term and holds pixels roughly
// constant — mobile DPR is already capped). A SCENE marches EVERY object each
// map step, so its cost is the SUM over objects of (object ops × object iters):
// objects multiply mapDE cost, and the score reflects that, not one op-list.
export function formulaCostScore(f, { pixels = 1, iters = null } = {}) {
  if (!f || typeof f !== "object") return 0;
  const px = Math.max(1, pixels);
  if (Array.isArray(f.objects)) {
    let work = 0;
    for (const o of f.objects) {
      if (!o || o.muted) continue;
      const n = Array.isArray(o.ops) ? o.ops.length : 0;
      work += n * Math.max(1, o.iters || f.iters || 1);
    }
    return work * px;
  }
  const it = iters != null ? iters : Math.max(1, f.iters || 1);
  return opEvalsPerIter(f) * it * px;
}

// Coarse/mobile-class device gate for the governor + entry clamp. Adapter
// strings are USELESS here — an iPad reports vendor/architecture "apple"
// (identical to a Mac) and classifyTier() returns "fast" — so key off the
// pointer/touch signals and UA markers instead (the 2026-08-01 iPad had
// slowGpu=false yet died). A touch LAPTOP (a fine PRIMARY pointer + a strong
// GPU) must NOT qualify — it won't sustain slow frames and shouldn't be nerfed —
// so require a COARSE primary pointer ALONGSIDE touch, or an explicit mobile UA
// marker (which also catches iPadOS Safari's desktop-mode "Macintosh" UA via the
// coarse+touch fallback).
export function isMobileClass({
  maxTouchPoints = 0,
  coarsePointer = false,
  ua = "",
} = {}) {
  if (/crios|iphone|ipad|ipod|android|\bmobile\b/i.test(ua)) return true;
  return !!coarsePointer && maxTouchPoints > 0;
}

// Cost-aware entry clamp: the STARTING detail (iterations) for a freshly
// loaded/imported/preset-applied formula on a coarse/mobile device, clamped so
// the FIRST unmeasured settle can't be fatal. Only ever LOWERS (the user drags
// Detail back up; a zoom hands detail to auto-detail). Scenes are out of scope
// (they carry per-object iters; detailOverride is single-object). Returns the
// clamped iter count, or null for "no clamp" (desktop, cheap formula, or a base
// already at/below the floor).
export const MOBILE_ENTRY_WORK_BUDGET = 160; // ceiling on op-evals/iter × iters
export const MOBILE_ENTRY_MIN_DETAIL = 8; // never clamp below this — stay legible
export function entryDetailClamp(
  f,
  { coarseMobile = false, budget = MOBILE_ENTRY_WORK_BUDGET } = {},
) {
  if (!coarseMobile || !f || typeof f !== "object" || f.objects) return null;
  const w = opEvalsPerIter(f);
  const base = f.iters || 0;
  if (!(w > 0) || !(base > 0)) return null;
  if (w * base <= budget) return null; // cheap enough — no clamp
  const clamped = Math.max(MOBILE_ENTRY_MIN_DETAIL, Math.floor(budget / w));
  return clamped < base ? clamped : null; // only ever lower the start
}

// #562 — the entry clamp above set detailOverride, but preview.js's frameTo
// (#551/#560) nulls detailOverride on EVERY retarget, and a load's own frameTo
// (share/preset/import/Surprise/Remix/Wander all funnel through setFormula
// then frameTo) always fires right after setFormula — so the clamp used to be
// dead before a single frame rendered. This tiny pure state machine is the fix's
// seam: setFormula arms it the instant it clamps; frameTo consumes it on the
// very next call (whichever way that call classifies) and learns whether IT is
// allowed to null detailOverride. One-shot by construction — consuming always
// disarms, so only the load's OWN frameTo is affected, never a later, unrelated
// retarget. disarm() is also called at the TOP of every setFormula (even one
// that doesn't re-clamp) and by every other detailOverride writer (a real zoom,
// a manual slider drag), so a stale arm can never survive to the wrong call.
export function makeEntryClampArm() {
  let armed = false;
  return {
    disarm() {
      armed = false;
    },
    arm() {
      armed = true;
    },
    // Read-and-clear: true only for the ONE call immediately following arm().
    consumeSurvives() {
      const wasArmed = armed;
      armed = false;
      return wasArmed;
    },
  };
}

// ── Per-frame scale governor (#476) ────────────────────────────────────────
// Reactive companion to the fence-timeout recovery in preview.js (#460/#473):
// that fires only after a 10 s+ watchdog kill; this watches the MEASURED
// settled-frame time and shrinks the render scale within 2-3 frames once frames
// SUSTAIN past a budget, so a heavy formula never REACHES the multi-second frame
// that kills a mobile GPU process. Scale is the shape-preserving lever (GPU cost
// ∝ scale²); preview.js composes it with settleScaleCap by taking the min.
export const GOV_BUDGET_MS = 250; // sustained frame time that triggers a downshift
export const GOV_RECOVER_FRAC = 0.6; // recover only below this fraction of budget (hysteresis gap)
export const GOV_SCALE_FLOOR = 0.25; // never shrink below this (matches the interactive floor)
export const GOV_OVER_FRAMES = 2; // consecutive over-budget frames before acting
export const GOV_UNDER_FRAMES = 24; // consecutive comfortable frames before recovering a step
export const GOV_DOWN_MIN = 0.5; // one downshift halves at most (no cliff from a noisy sample)
export const GOV_UP_STEP = 1.25; // gentle recovery multiplier

export function governorInit() {
  return { scale: 1, over: 0, under: 0 };
}

// Fold one MEASURED frame time into the governor. `frameMs` must be honest GPU
// time — preview.js feeds only FENCED settled/banded frames (interactive frames
// are unfenced and under-measure the GPU). Returns a NEW state (pure); read
// state.scale. Hysteresis: downshift above budgetMs, recover only below
// recoverMs (default 0.6×budget), hold in the dead-band between — so a frame
// parked near budget after a downshift can't oscillate.
export function governorStep(state, frameMs, opts = {}) {
  const budgetMs = opts.budgetMs ?? GOV_BUDGET_MS;
  const recoverMs = opts.recoverMs ?? budgetMs * GOV_RECOVER_FRAC;
  const floor = opts.floor ?? GOV_SCALE_FLOOR;
  const overFrames = opts.overFrames ?? GOV_OVER_FRAMES;
  const underFrames = opts.underFrames ?? GOV_UNDER_FRAMES;
  let { scale, over, under } = state;
  if (!(frameMs > 0)) return { scale, over, under }; // ignore non-measurements
  if (frameMs > budgetMs) {
    over += 1;
    under = 0;
    if (over >= overFrames && scale > floor) {
      // Aim near budget (cost ∝ scale²) but never overcorrect from one sample:
      // at most halve per step. frameMs was measured at the CURRENT scale, so
      // the ratio is relative — successive steps compose toward budget.
      const target = Math.sqrt(budgetMs / frameMs);
      scale = Math.max(floor, scale * Math.max(GOV_DOWN_MIN, target));
      over = 0; // observe the effect of this step before stepping again
    }
  } else if (frameMs < recoverMs) {
    under += 1;
    over = 0;
    if (under >= underFrames && scale < 1) {
      scale = Math.min(1, scale * GOV_UP_STEP);
      under = 0;
    }
  } else {
    over = 0; // dead-band [recoverMs, budgetMs]: stable — decay both counters
    under = 0;
  }
  return { scale, over, under };
}

// ── Device-loss verdict (#473) ─────────────────────────────────────────────
// A lost WebGPU device used to be TERMINAL: preview.js latched the pump shut so
// it couldn't storm the dead device (#476 Part C) and stopped there, leaving the
// canvas frozen mid-render with only a "reload" toast. The 2026-08-01 iPad dump
// (CriOS/Metal, Kleinian Drop at the f32 wall) is the field case — and, unlike
// the desktop Metal wedge in #460, CriOS DOES fire device-lost, so the event is
// actionable. The fallback ladder already has a runtime demotion rung — the one
// the dead-GL tier uses (hasGPU=false + onGpuDead → the app's ASCII view) — it
// simply was never wired to this event. This is the verdict layer for that.
//
// Why demote rather than re-create the device: the canvas cannot change context
// type (a canvas that got "webgpu" can never getContext("webgl2")), so the only
// rungs below WebGPU here are a fresh WebGPU device or the CPU/ASCII tier. A
// re-init would have to re-push every renderer-bound resource AND arrive with
// wall-regime cost governance, or it just re-kills the device on the next settle
// — that is issue #473's half (2) and stays a separate change. Demotion is the
// bounded move that turns a dead canvas into a live view.
//
// Pure + GPU-free so it is unit-testable in Node (core/renderpolicy.test.mjs).
export const DEVICE_LOST_REASON_MAX = 160; // keep the beacon/toast string sane

// classifyDeviceLoss({ reason, message, demoted }) → { demote, reason }
//
//   reason  : GPUDeviceLostInfo.reason ("destroyed" | "unknown" | undefined)
//   message : GPUDeviceLostInfo.message (often empty — the iPad dump had none)
//   demoted : has a demotion already fired this session? (fire-exactly-once)
//
// "destroyed" means WE called device.destroy() — an intentional teardown (page
// unload, an explicit dispose), NOT a crash. Demoting on it would toast a user
// who is already leaving. Every other reason, including the field case's bare
// {reason:"unknown"} and a reason-less loss, is a crash: demote.
export function classifyDeviceLoss({
  reason = "",
  message = "",
  demoted = false,
} = {}) {
  if (demoted) return { demote: false, reason: null }; // already fell a tier
  if (String(reason || "").toLowerCase() === "destroyed")
    return { demote: false, reason: null }; // our own teardown, not a fault
  let why = "device-lost: " + (reason || "unknown");
  const msg = String(message || "").trim();
  if (msg) why += " — " + msg;
  return { demote: true, reason: why.slice(0, DEVICE_LOST_REASON_MAX) };
}

// ── Readback watchdog budget (#460 residue) ─────────────────────────────────
// #477 raced every queue fence in preview.js against a timeout, because a GPU
// dispatch killed by the platform watchdog fires NO device.lost and NO error —
// the promise simply never settles. `buffer.mapAsync()` is the SAME primitive
// with the same failure: the map for a killed submit never settles either, and
// renderer.js awaits one on every readback path (tiled save, alpha still,
// thumbnail, video frame, splat capture). Three of those await under
// preview.js's `busy` and two under the exporters' `offline` — so an unguarded
// map re-creates #460's permanent freeze on a different primitive.
//
// The budget is per SUBMIT and deliberately huge. The platform watchdog kills
// an individual dispatch in seconds, so no single submit can legitimately
// outrun READBACK_MS_PER_SUBMIT — but an N-sample export queues N of them
// before mapping, so a fixed ceiling would abort real work. Hence the floor is
// a full minute rather than #477's 10 s: a hung map is PERMANENT, so recovering
// in 60 s instead of 10 s costs the user nothing, while tripping on a healthy
// 40 s export would cost them the render. False negatives are cheap here; false
// positives are not.
//
// Pure + GPU-free so it is unit-testable in Node (core/renderpolicy.test.mjs).
export const READBACK_MS_PER_SUBMIT = 20000; // no watchdog lets one submit near this
export const READBACK_MS_FLOOR = 60000; // generous floor for a single-submit map

// readbackBudgetMs(submits) → ms to wait before declaring a map hung.
// `submits` is how many queue submits were enqueued BEFORE the map — the map
// cannot resolve until all of them retire, so the wait scales with them.
export function readbackBudgetMs(submits = 1) {
  const n = Number.isFinite(submits) && submits > 0 ? Math.ceil(submits) : 1;
  return Math.max(READBACK_MS_FLOOR, n * READBACK_MS_PER_SUBMIT);
}

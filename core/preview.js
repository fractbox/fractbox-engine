// Shared WebGPU preview controller — the whole render engine both frontends
// use, so they differ ONLY in their formula editor. Owns the renderer, camera,
// coloring, the busy-gated pump with interactive quality tiers, canvas
// orbit/zoom/pinch gestures, auto-rotate, and the GPU side of PNG export +
// preset thumbnails. It builds NO app markup: PNG export renders to a Blob
// (stillBlob) and a thumbnail renders to a data URL (renderThumbTile) — these
// are exposed as raw primitives; the download <a> and the clickable grid are
// each app's own concern (REFACTORING.md #3 / issue #77 — core owns no DOM).
// The only DOM the engine itself touches is its render surface (the canvas) +
// offscreen pixel buffers.
//
// The app feeds it a formula (setFormula) + coloring (setColoring); the editor
// is entirely the app's business. DOM-side chrome (badge text, hints) is left
// to the app via the onFrame / onFrameStart callbacks.

import { createRenderer, lensSample, r2jitter } from "./renderer.js";
// renderer_gl.js (+ its shader_gl.js ≈ 63 KB) is the WebGL2 FALLBACK — only
// reached when WebGPU is unavailable or forced off. Load it lazily so the
// WebGPU-capable majority never pays its download/parse on the boot chunk.
// Dynamic import is legal in raw ESM, so core stays build-less.
import {
  makeCamera,
  MIN_DIST,
  PT_MIN_DIST,
  isContinuousPush,
} from "./camera.js";
import {
  hybridLooseDE,
  df64Eligible,
  ptEligible,
  lambdaHat,
  kStarFor,
} from "./stability.js";
// Perturbation deep zoom (PERTURBATION_ZOOM_IMPL.md PR-4): the reference-
// orbit builder + JS delta kernel (the D8 deep probe) + the BigInt target
// helpers (D6 — past ~×10¹⁶ a re-pinned target exceeds f64).
import {
  buildOrbit,
  deltaDE,
  ptSupported,
  targetToFx,
  fxNudge,
  fxFromF64,
  fxToF64,
} from "./perturb.js";
// The shared frame settle + feature derivation (EXPORT_P1 PR-A): writeFrame /
// frameFeatures below are thin live-state bindings over these — the flat/scene/
// hybrid/morph Globals construction has NO copy in this file anymore.
// ensureCpu: capturesettle.js lazy-loads cpu.js internally (#266) — called
// below, early in createPreview's async boot work.
import { settleFrame, frameFeaturesFor, ensureCpu } from "./capturesettle.js";
import {
  zoomHeadroom,
  zoomMag,
  ptHeadroom,
  F32_QUANTUM,
  DF64_QUANTUM,
} from "./recenter.js";
import { defaultColoring } from "./coloring.js";
import { surfaceHitDist } from "./zoomsurface.js";
import {
  wheelZoomFactor,
  makeVelocityTracker,
  makeGlide,
  GLIDE_MIN_SPEED,
  twoFingerDelta,
} from "./gestures.js";
import { cruiseAdvance } from "./cruise.js";
import { embedChunks } from "./pngmeta.js";
// TILED_EXPORT.md — the tile geometry + off-axis camera window (PR-1) and the
// streaming PNG encoder (PR-2), consumed by stillBlobTiled below.
import {
  tileGrid,
  tileWindow,
  evenDims,
  rowBandBytes,
  TILE_PAD,
  TILE_WINDOW_IDENTITY,
  STILL_PX_CAP_TILED,
} from "./tilegrid.js";
import {
  createPngStream,
  memorySink,
  pngStreamSupported,
} from "./pngstream.js";
// The pure render-policy layer (bailout choice, near/far constants, auto-detail
// law, quality tiers, scene march scale, cheap-tier shading) lives in
// renderpolicy.js — unit-tested in Node; this module feeds it its live state.
// (The settle-path consumers — bailoutFor, sceneDeScale, resolveDeScale,
// unboundedScene, shadeLight, TFAR_UNBOUNDED_MUL — moved with the settle to
// capturesettle.js in PR-A.)
import {
  REF_DIST,
  TNEAR_K,
  TFAR_K,
  ITER_CEIL,
  effectiveIters as policyEffectiveIters,
  qualityParams as policyQualityParams,
  stillQualityParams,
  resolveStillDims,
  exportSampleCount,
  STILL_PX_CAP,
  classifyTier,
  bootPredictMs,
  shouldRaceGeneralAtBoot,
  isMobileClass,
  entryDetailClamp,
  makeEntryClampArm,
  governorInit,
  governorStep,
  GOV_BUDGET_MS,
  GOV_SCALE_FLOOR,
  classifyDeviceLoss,
} from "./renderpolicy.js";

// Re-export the HUD's magnification reference (M = REF_DIST/dist) — the app
// imports it from here (app/src/main.ts), and it predates renderpolicy.js.
export { REF_DIST };

const D2R = Math.PI / 180;

// cpu.js is the whole CPU/ASCII render tier (~117KB raw) — not needed for the
// WebGPU boot path itself, only for the zoom-to-surface probe below (getCpuDE).
// A static (or unconditional top-level dynamic) import pinned it in the app's
// boot chunk (#266) — Vite/Rolldown modulepreloads a module-scope `import()`
// right alongside the entry (it can prove it always fires), which is no
// improvement. Wrapping it in a function only reached by an explicit call
// (ensureCpuMod, invoked early in createPreview below) is what actually keeps
// it off the eager/preloaded set.
let _cpuMod = null;
let _cpuModLoading = null;
function ensureCpuMod() {
  if (!_cpuModLoading)
    _cpuModLoading = import("./cpu.js").then((m) => {
      _cpuMod = m;
      return m;
    });
  return _cpuModLoading;
}

// capturedrive.js (the splat GPU-capture view loop) is only reached from
// captureSplatGBuffer below — an explicit, on-demand export action, never on
// the render boot path. Load it on first use instead of statically (#266).
let _capturedriveMod = null;
async function getDriveSplatCapture() {
  if (!_capturedriveMod) _capturedriveMod = await import("./capturedrive.js");
  return _capturedriveMod.driveSplatCapture;
}

export async function createPreview(canvas, opts = {}) {
  const isTouch = navigator.maxTouchPoints > 0;
  const DPR_CAP = isTouch ? 1.0 : 2.0;
  // #476 — coarse/mobile-class gate for the frame governor + cost-aware entry
  // clamp. Adapter strings are useless (an iPad reports "apple" and classifies
  // "fast"), so key off pointer/touch + UA markers instead (isMobileClass).
  const coarseMobile = isMobileClass({
    maxTouchPoints: navigator.maxTouchPoints || 0,
    coarsePointer: !!(
      typeof window !== "undefined" &&
      window.matchMedia &&
      window.matchMedia("(pointer: coarse)").matches
    ),
    ua: (typeof navigator !== "undefined" && navigator.userAgent) || "",
  });
  // Test/opt-in hook: opts.governorBudgetMs (app threads ?govbudget=<ms>) forces
  // the governor ON at a custom budget so headless verification can trip it on a
  // desktop GPU that would never otherwise qualify as mobile-class.
  const govBudgetOverride = Number.isFinite(opts.governorBudgetMs)
    ? opts.governorBudgetMs
    : null;
  // #476 Part C: latched when the WebGPU device is lost — the pump stops
  // submitting so it can't storm "device mismatch" errors on the dead device.
  let deviceLost = false;
  // #473 — one-shot latch for the runtime tier demotion (onGpuDead). Shared by
  // the device-lost path below and the dead-GL gate in pump(), so whichever
  // fires first owns the transition and the app is told exactly once.
  let gpuDeadFired = false;
  const onFrame = opts.onFrame || (() => {}); // (ms) after each frame
  const onFrameStart = opts.onFrameStart || (() => {});
  // (active, kind) — fires when the pump HOLDS a frame waiting for a variant to
  // compile (post-boot browse), so the app can show a "compiling shader" hint.
  // Announced one frame BEFORE the compile is kicked, so the hint paints before
  // the GPU-process compile starves the compositor. kind ∈ shape|hybrid|morph|
  // formula. onCompileHold(false) when the compile lands and the frame draws.
  const onCompileHold = opts.onCompileHold || (() => {});
  let compileHolding = false;
  // (where, error) when a frame fails. The pump already catches + logs so a
  // throw can't latch it shut, but console-only errors are invisible off-device
  // — a broken tier ran 9 days in production unreported (issue #206). The app
  // hooks this into its render-health beacon.
  const onRenderError = opts.onRenderError || (() => {});

  let renderer = null,
    hasGPU = false,
    backend = "none",
    capability = null; // CAPABILITY_PROBE.md — static machine class, set at boot
  let formula = null;
  // Last globals payload actually sent to the renderer on the FLAT path — the
  // standalone GLSL exporter (#291) reads it so its baked consts are the exact
  // values the on-screen render used (auto-levels sigLo/sigSpan included). Only
  // the flat branch sets it; hybrid/scene leave it stale (the exporter guards
  // those out and forces a fresh flat frame before reading — see getLastFrameState).
  let lastFrameState = null;
  // Formula morph (VIDEO_EXPORT_DRAWER_V2 tier-2 spike): when set, writeFrame
  // blends the current formula's DE toward morph.f by morph.t (WebGPU only —
  // other tiers render formula A). Null = every path exactly as before.
  let morph = null;
  // Coloring-mode crossfade (timeline transitions between views with different
  // color modes / palette toggles): {t, modeB, palOnB} — the shader shades
  // under both modes and mixes. Null = legacy shading. WebGPU only (the GL
  // writer ignores the extra word — those tiers keep the midpoint snap).
  let colorBlend = null;
  // Debug surface-quality overlay (#370, SPIRULAE_LEARNINGS Plan C3): a dev-only
  // heat view of a per-pixel march metric (0 off | 1 step-count | 2 overshoot |
  // 3 ∇DE instability). WebGPU tier ONLY; gated in the UI behind ?diag / the
  // showDiag pref. 0 renders byte-identically to before the field existed.
  let debugView = 0;
  // ORTHOGRAPHIC_VIEWS (#441) — the ortho half-height; 0 = perspective. Lives
  // HERE, as closure state, deliberately NOT on `cam`: the orbit model is a
  // tight {yaw,pitch,dist,fov,target} whose consumers (sharecodec's fixed-width
  // TAG.VIEW, interp's lerp, recenter, the ASCII tier's parallel copy) all break
  // on a new camera field. Keeping it here also means ONE place decides what
  // writeGlobals, stillBlob and the standalone bundles see. Transient by
  // design: not in the #c= share link, not across a formula switch.
  let orthoH = 0;
  // Zoom-to-surface (§5 navigation): a CPU distance field for the CURRENT
  // formula, built lazily and reused across zoom events, so a scroll/pinch can
  // cheaply probe where the surface is straight ahead. Invalidated whenever the
  // formula changes (setFormula). Null until first needed / on build failure.
  let cpuDE = null,
    cpuDEFor = null;
  let coloring = defaultColoring();
  const cam = makeCamera(
    opts.camera || { yawDeg: 35, pitchDeg: 22, dist: 24, fovDeg: 42 },
  );
  let needsDraw = false,
    busy = false,
    scheduled = false,
    offline = false; // video export: while true the live rAF pump is suspended so
  // the offline render loop (captureFrame) fully owns the canvas/device.
  let quality = "full",
    moveQuality = "balanced", // #32 — render tier while moving: smooth|balanced|full
    df64Engaged = false, // deep zoom P4 — hysteresis latch (see df64Update)
    df64Mode = "auto", // deep zoom P4 — setDf64 override ('auto' | 'off')
    lastDf64 = false, // what the last written frame actually sent (renderInfo)
    // Perturbation tier (PR-4): its own latch + lever, superseding df64 when
    // both are eligible (impl plan D10). The reference orbit is cached by a
    // (iters, target-generation, formula-signature) key and re-uploaded on
    // any staleness (re-pin, auto-detail boost, formula edit — plan D5).
    // Field report 2026-08-01 (iPad, PR-4 preview): "kills gpu easily" —
    // mobile-class GPUs sit under much tighter compositor watchdogs, and a
    // deep pt settle (512 steps × 50+ iters × the dual-switch kernel) can
    // blow one before the band predictor has a pt-tier measurement. Until
    // the fleet pass (plan PR-5) proves the tier per device class, pt
    // defaults OFF on touch devices — setPt('auto') re-enables for testing,
    // and the tier-tagged predictor below right-sizes the first settle
    // everywhere else.
    ptMode = isTouch ? "off" : "auto", // setPt override ('auto' | 'off')
    ptEngaged = false,
    lastPt = false, // did the last written frame carry the pt variant?
    ptOrbit = null,
    ptOrbitKey = "",
    // #551 — how many times the reference orbit was actually REBUILT this
    // session. D5's law is "at re-pin/boost rate, never per frame", and the
    // only way to see a violation is to count: a flythrough whose rebuild
    // count tracks its frame count is the defect, one that tracks its re-pins
    // is the law. Surfaced on renderInfo beside the other tier diagnostics.
    ptOrbitBuilds = 0,
    lastPtWhy = "", // #551 — which gate clause last denied the pt tier
    ptTfx = null, // BigInt[3] — the exact target (D6); null until first sync
    ptTfxGen = 0, // bumped on every exact nudge/rebase (cache key component)
    ptMirror = null, // the f64 target ptTfx corresponds to (drift detection)
    autoDetail = opts.autoDetail ?? true, // raise iters with zoom depth (§6)
    detailOverride = null, // §6 — manual ABSOLUTE iters set from the Detail slider;
    // one-shot, lets the user drop below auto-detail's floor; cleared on next zoom
    autoRotate = false,
    settleTimer = null,
    spinSpeed = 0.7,
    spinTilt = 0; // 0° = turntable (spin around +Z) … 90° = tumble (around +X)
  // #562 — one-shot arm so the #476 entry clamp setFormula just applied
  // survives the load's own immediately-following frameTo retarget-reset; see
  // makeEntryClampArm's doc in renderpolicy.js for the full sequencing.
  const entryClampArm = makeEntryClampArm();
  // Boot uses the hero's specialized march variant for a fast first paint; once
  // it's on screen we flip to the GENERAL (compile-once, reused) variant for all
  // subsequent formulas so browsing never fires a per-preset compile (renderer
  // prewarmGeneralFor). Flipped true after the first successful draw.
  let browseMode = false;
  // Orbit inertia (§3.3): a glide step-function while coasting after a flick
  // (advanced inside pump — see there), plus the release-velocity tracker fed
  // by the orbit drag. Orbit drags only: pan/pinch never glide (pan inertia at
  // depth flings the user across the structure), nor does reduced-motion.
  let glide = null,
    glideT = 0,
    dragMode = null; // 'orbit' | 'pan' | 'pinch' — what the current drag is doing
  // DE-scaled cruise (§4) — hold-to-fly, advanced in pump like glide. dir +1
  // in / −1 out; the ahead-probe is throttled to every 3rd frame; hasHit
  // routes between asymptotic approach and open-space drift (cruise.js).
  let cruise = null; // { dir, t, tick, hasHit }
  const dragVel = makeVelocityTracker();
  const reducedMotion = () =>
    window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches ?? false;
  // Spin axis from the tilt angle: tilts from vertical (+Z) toward +X.
  const spinAxis = () => [
    Math.sin((spinTilt * Math.PI) / 180),
    0,
    Math.cos((spinTilt * Math.PI) / 180),
  ];

  // Renderer fallback ladder: WebGPU → WebGL2 → (none → app shows the ASCII view).
  // opts.backend forces a tier for testing: 'webgl2' skips WebGPU; 'ascii'/'none'
  // skips both (so the app falls to the ASCII view even with a GPU present).
  const force = opts.backend;
  try {
    if (force === "webgl2" || force === "ascii" || force === "none")
      throw new Error("forced " + force);
    renderer = await createRenderer(canvas, {
      // #476 Part C: renderer.js only NOTES device-lost (it fires onTrouble);
      // nothing stopped the pump, so on the 2026-08-01 iPad the loop kept
      // submitting resources bound to the DEAD device → a 5 ms-later "device
      // mismatch" uncaptured-error storm. Latch the guard here and forward the
      // event to the app so it's logged + toasted, not silent. #477 handles the
      // DIFFERENT watchdog-fence hang (which fires NO device-lost), so this
      // composes rather than overlaps.
      onTrouble: (kind, detail) => {
        if (kind === "device-lost") {
          deviceLost = true;
          console.error(
            "preview: WebGPU device lost —",
            detail?.reason ?? "unknown",
            detail?.message ?? "",
          );
        }
        try {
          opts.onGpuTrouble?.(kind, detail);
        } catch {
          /* reporting must never break the frame */
        }
        // #473 — and then FALL A TIER instead of stopping here. The latch above
        // only keeps the pump off the dead device; on the 2026-08-01 iPad that
        // left a half-rendered frozen canvas and a "reload" toast. The ladder's
        // runtime demotion rung already exists (it is how a dead WebGL2 tier
        // reaches the ASCII view — see the glHealth gate in pump() below); this
        // event was simply never wired to it. Same verdict → same handler → the
        // app attaches the CPU/ASCII renderer and the user keeps a LIVE view.
        // Fires after the forward above so the diag ring reads in event order.
        // Why demote and not re-create: the canvas cannot change context type,
        // so ASCII is the next reachable rung, and a fresh device would have to
        // arrive with wall-regime cost governance or it just re-kills on the
        // next settle (#473 half 2 — its own change).
        if (kind === "device-lost") {
          const v = classifyDeviceLoss({
            reason: detail?.reason,
            message: detail?.message,
            demoted: gpuDeadFired,
          });
          if (v.demote) {
            gpuDeadFired = true;
            hasGPU = false; // scheduleDraw() no-ops from here
            noteDiag("device-lost-fallback", { reason: v.reason });
            try {
              opts.onGpuDead?.(v.reason);
            } catch {
              /* the app's fallback must never re-enter the pump */
            }
          }
        }
      },
    });
    hasGPU = true;
    backend = "webgpu";
    // Surface async WebGPU validation errors (P3) — without this a bad bind
    // group / pipeline silently drops every pass in the submit: black canvas,
    // no exception anywhere. Costs nothing when healthy. Also route to the diag
    // reporter so it's visible on-page (crash triage on Brave/Windows).
    renderer.device.addEventListener?.("uncapturederror", (e) => {
      const msg = e.error?.message || String(e);
      // #476 Part C: once the device is lost, every in-flight submit trips a
      // "device mismatch" error — don't storm the reporter with them (the pump
      // has already stopped; these are the last queued frames draining).
      if (deviceLost) return;
      console.error("WebGPU uncaptured:", msg);
      try {
        opts.onGpuTrouble?.("uncaptured-error", { message: msg });
      } catch {
        /* reporting must never break the frame */
      }
    });
  } catch (e) {
    if (force !== "ascii" && force !== "none") {
      try {
        const { createRendererGL } = await import("./renderer_gl.js");
        // Give the GL tier the SAME diag hook the WebGPU tier gets (compile/link
        // /context/draw faults → app diag.ts), plus the ?glfail test hook.
        renderer = await createRendererGL(canvas, {
          onTrouble: opts.onGpuTrouble,
          glFail: opts.glFail,
        });
        hasGPU = true;
        backend = "webgl2";
      } catch (e2) {
        console.error("WebGL2 unavailable:", e2?.message || e2);
        hasGPU = false;
      }
    }
    if (!hasGPU && !force) console.warn("WebGPU unavailable:", e?.message || e);
  }
  // Detect-broken-GL auto-fallback state. The WebGL2 tier fails DARK (#206): a
  // dead tier (frag compile/link fail, or a GL error on the first draws) draws
  // BLACK while the pump reports frames advancing + skip:"drew" + impossibly
  // fast ms. glHealth() (pure classifier) is polled for the first few frames;
  // once dead we stop the GPU pump and hand the app the reason so it falls to
  // the ASCII view. `glDead` guards the transition to fire exactly once.
  let glDead = false;
  const GL_HEALTH_WINDOW = 8; // poll only the first frames — getError is first-draws-only
  // Capability probe (docs/planning/CAPABILITY_PROBE.md) — classify the MACHINE
  // from the static signals now known (adapter identity + host), the moment the
  // backend is chosen. Zero-cost, OBSERVE-ONLY: the app reports it to /stats so
  // the tier thresholds can be calibrated on the real fleet. Acting on the tier
  // (cheaper boot variant for 'slow', sharper first frame for 'fast') is a
  // separate change through the soak ladder — this slice only measures + exposes.
  {
    const ad = renderer?.getDiag?.()?.adapter ?? null;
    const isSoftware = !!renderer?.getDiag?.()?.isFallback;
    const cores = navigator.hardwareConcurrency || 0;
    capability = {
      backend,
      adapter: ad,
      isSoftware,
      cores,
      deviceMemory: navigator.deviceMemory || 0,
      dpr: Math.min(window.devicePixelRatio || 1, DPR_CAP),
      tier: classifyTier({
        backend,
        vendor: ad?.vendor ?? "",
        architecture: ad?.architecture ?? "",
        description: ad?.description ?? "",
        isSoftware,
        cores,
      }),
    };
  }
  // Feature-gated march variants (renderer.js) build lazily on first use. The
  // boot variant is flat + Surface; the pump/tile paths below prewarm whatever
  // variant a frame needs (async) and HOLD the frame until it's ready, so the
  // first shape/color-mode/numeric use is a background compile, not a freeze.
  // Nudge the current formula's variant warm shortly after boot settles, in
  // case the hero itself needs one (e.g. a Glow or scene hero).
  if (renderer?.prewarmMarchFor)
    setTimeout(() => {
      if (formula) renderer.prewarmMarchFor(frameFeatures(formula));
    }, 2000);

  function scheduleDraw() {
    if (!hasGPU || offline || deviceLost) return; // #476 Part C: dead device — don't re-arm
    needsDraw = true;
    if (!scheduled && !busy) {
      scheduled = true;
      requestAnimationFrame(pump);
    }
  }
  // Coarse quality while interacting, settle to full once the user pauses.
  function bumpInteract() {
    quality = "low";
    clearTimeout(settleTimer);
    settleTimer = setTimeout(() => {
      quality = "full";
      scheduleDraw();
    }, 200);
    scheduleDraw();
  }
  // Auto-detail (§6) — policy lives in renderpolicy.js; this wrapper feeds it
  // the live camera depth + the user's autoDetail/Detail-slider state.
  // Single-object and hybrid only; CSG scenes carry per-object iters (§14).
  function effectiveIters(base) {
    return policyEffectiveIters(base, {
      dist: cam.dist,
      autoDetail,
      detailOverride,
    });
  }
  // Does this formula render a shape leaf? Only leaf objects need the
  // leaves-included march variant (renderer.js boots leaf-free). Mirrors
  // normalizeSceneObject's shapeId sources (explicit shapeId · legacy objType
  // 1-6 · boxBase → box) — conservative: over-detection only holds a frame for
  // an already-warmed variant, it never renders wrong.
  // Predict the leaf-id set a scene uses (lever #3 for leaves), so the pump
  // prewarms the SAME specialized variant the draw selects. Mirrors
  // normalizeSceneObject's shapeId sources (explicit shapeId · legacy objType
  // 1-6 · boxBase → box 1). Prediction only — the renderer latches the EXACT set
  // from writeScene, so a mismatch costs a resync, never a wrong render.
  // (formulaLeafIds/formulaOpSet moved to capturesettle.js — EXPORT_P1 PR-A;
  // the closure `morph` read became an explicit parameter there.)
  // The march-variant descriptor a frame of (formula, current coloring) needs —
  // feature flags + the op-set — must mirror renderer.js's per-frame activeFeat
  // latch so the pump prewarms (and holds for) the SAME variant the draw selects.
  // A mismatch only costs an extra prewarm, never correctness.
  // ── Deep zoom P4 (DEEP_ZOOM_DF64.md) — df64 engagement ──
  // Eligibility is static per formula (stability.js df64Eligible: flat +
  // analytic-IFS + all ops twinned); ENGAGEMENT is a continuous per-frame
  // signal with wide hysteresis (engage when headroom < 4 — the wall
  // indicator's "approaching" threshold; release only past 16) so zoom
  // oscillation can't thrash variants. The smooth (drag) tier stays f32 by
  // decision D8 — the pump falls back to the pinned f32 twin, no compile.
  function df64Args() {
    // DISPLAY resolution, never the backing store: the settle budget drops
    // the render scale under load, and a backing-store heightPx INFLATES
    // headroom exactly when the device is struggling — on the 2026-08-01
    // iPad report the budget's 0.36× scale turned the h=3 detail-limit stop
    // into an effective ~1, and the gate "blew past". This is the same bug
    // the HUD fixed in b288e36; it now holds for every consumer of the law
    // (brake, re-pin guards, df64/pt engagement — which also stops
    // flickering with the budget). Headless canvases have no layout
    // (clientHeight 0) → fall back to the backing height as before.
    const hp = Math.round(
      (canvas.clientHeight
        ? canvas.clientHeight * Math.min(window.devicePixelRatio || 1, DPR_CAP)
        : canvas.height) || 1,
    );
    return {
      target: cam.target,
      dist: cam.dist,
      fovDeg: (cam.fov * 180) / Math.PI,
      heightPx: hp,
    };
  }
  // ── Perturbation tier (PERTURBATION_ZOOM_IMPL.md PR-4) ──────────────────
  // Eligibility requires BOTH the registry twin (wgslPt) and the JS orbit
  // stepper — perturb.test.mjs pins the memberships equal (D7).
  const ptElig = (f) => ptEligible(f, ptSupported);
  // The zoom floor this formula can actually use (review M1: FOUR clamp
  // sites share this — cam.zoom calls, cruise, double-click, wheel).
  const ptMinDist = () =>
    ptMode !== "off" && formula && ptElig(formula) ? PT_MIN_DIST : MIN_DIST;
  // Engagement mirrors df64Update: the SAME f32-law headroom signal, engage
  // h < 8 / release h > 32 — pt supersedes df64 when both eligible (D10;
  // df64Update below stands down while the pt latch holds).
  function ptUpdate(f) {
    if (ptMode === "off" || !f || morph || !ptElig(f)) {
      ptEngaged = false;
      return;
    }
    const h = zoomHeadroom(df64Args());
    if (ptEngaged) {
      if (h > 32) ptEngaged = false;
    } else if (h < 8) {
      ptEngaged = true;
    }
  }
  const ptNow = () => ptEngaged;
  // Exact target bookkeeping (D6). cam.target stays plain f64 for every
  // existing consumer; ptTfx carries the sub-f64 truth. nudgeTarget applies
  // a RESIDUAL-SPACE delta to both (each re-pin's delta is f64-exact at its
  // own scale — it is absolute-target assignment that loses bits, review B1).
  function nudgeTarget(delta) {
    if (ptTfx) {
      ptTfx = fxNudge(ptTfx, delta);
      ptTfxGen++;
    }
    cam.target = [
      cam.target[0] + delta[0],
      cam.target[1] + delta[1],
      cam.target[2] + delta[2],
    ];
    ptMirror = cam.target.slice();
  }
  // Any target mutation that bypassed nudgeTarget (pan, orbit reset, a v1
  // share load…) rebases the exact target at f64 precision — correct above
  // ~×10¹⁶, and the fallback below it (a foreign mutation there has no
  // sub-f64 information to preserve anyway).
  function syncTfx() {
    const t = cam.target;
    if (
      !ptTfx ||
      !ptMirror ||
      t[0] !== ptMirror[0] ||
      t[1] !== ptMirror[1] ||
      t[2] !== ptMirror[2]
    ) {
      ptTfx = targetToFx(t);
      ptMirror = t.slice();
      ptTfxGen++;
    }
  }
  // The reference orbit for the CURRENT (formula, effective iters, exact
  // target) — rebuilt + re-uploaded on any staleness (plan D5), and rebuilt
  // SYNCHRONOUSLY here so offline exports can never ride a stale orbit (the
  // df64 stale-latch lesson, plan D9). ~1-3 ms of BigInt per rebuild, at
  // re-pin/boost rate, never per frame.
  function ptOpsSig(f) {
    return JSON.stringify([
      !!f.addC,
      !!f.julia,
      f.juliaC ?? 0,
      (f.ops ?? []).map((o) => [o.key, o.enabled === false ? 0 : 1, o.values]),
    ]);
  }
  function ensurePtOrbit(f, iters) {
    syncTfx();
    const key = iters + "|" + ptTfxGen + "|" + ptOpsSig(f);
    if (ptOrbitKey === key && ptOrbit) return true;
    ptOrbitBuilds++; // #551 — renderInfo diagnostic: re-pin cadence, not per-frame
    try {
      ptOrbit = buildOrbit(f, ptTfx, iters);
    } catch {
      ptOrbit = null; // unsupported shape → the frame falls back to f32/df64
      return false;
    }
    ptOrbitKey = key;
    renderer.writePerturbOrbit(ptOrbit.packed);
    return true;
  }
  // D8 — the depth-aware probe frame: ONE wrapper, four consumers (cruise/
  // zoomAtCenter ahead-probe, burial dEye, double-click, wheel). Past the
  // CPU f64 probe's validity (~×10¹⁴ — and its absolute-eye inputs hit the
  // B1 round trip well before), the probe evaluates the JS delta kernel in
  // RESIDUAL space; the origin the caller marches from switches with it.
  function probeFrame(b) {
    if (
      ptNow() &&
      cam.dist < 1e-12 &&
      formula &&
      ensurePtOrbit(formula, effectiveIters(formula.iters))
    ) {
      const orbit = ptOrbit;
      return {
        o: b.roRel.slice(),
        de: (x, y, z) => deltaDE(orbit, [x, y, z]),
      };
    }
    return { o: b.eye.slice(), de: getCpuDE() };
  }
  function df64Update(f) {
    if (df64Mode === "off" || !f || morph || ptEngaged || !df64Eligible(f)) {
      df64Engaged = false;
      return;
    }
    const h = zoomHeadroom(df64Args());
    // Engage at h < 8 (field-measured: at h≈4.9 the f32 render shows false
    // quantization plates vs the f64 ground truth — artifacts are visible
    // well above the h=1 wall). The k* margin makes k* ≥ 1 exactly below
    // h=8, so the latch and the law agree at the boundary. Release at 32
    // keeps the hysteresis band wide (4×).
    if (df64Engaged) {
      if (h > 32) df64Engaged = false;
    } else if (h < 8) {
      df64Engaged = true;
    }
  }
  // The engagement latch alone. Per-FRAME tier gating happens at the kStar
  // write (writeFrame): cheap interactive frames render the pinned f32 twin,
  // settled frames get df64. This was `&& moveQuality !== "smooth"` — a
  // GLOBAL pref gate that silently disabled df64 EVERYWHERE, settles
  // included, for exactly the slow machines that set the smooth pref (field
  // report 2026-07-31: M1, deep zoom stuck at f32 mush, diag showed a clean
  // full-tier settle with df64:false and no df64 compile ever attempted).
  const df64Now = () => df64Engaged;

  function frameFeatures(f, over) {
    // Thin wrapper binding the LIVE state (morph, df64 zoom latch, colorBlend
    // crossfade) onto the shared derivation (capturesettle.js — PR-A).
    // `over.morphF` (#609) lets needsCompile/prewarmFor key a morph PAIR
    // before setMorph installs it — Wander warms the from→to variant ahead of
    // the melt. Presence-checked, not truthiness: an explicit null must BEAT
    // the live morph (warming the destination's plain post-commit variant
    // while the previous melt's morph is still installed).
    return frameFeaturesFor(f, coloring, {
      morphF: over && "morphF" in over ? over.morphF : (morph?.f ?? null),
      df64: df64Now(), // latch updated by the pump BEFORE any consumer
      perturb: ptNow(), // perturbation tier latch (PR-4) — same contract
      blendModeB: colorBlend ? colorBlend.modeB : null,
    });
  }
  // Quality tiers (#32, deep zoom §6) — policy in renderpolicy.js; wrapper
  // feeds the interaction state + device class + camera depth.
  function qualityParams(f, rect) {
    // devicePx: the unscaled canvas pixel count — feeds the interactive
    // pixel budget (#212) so a 5K display interacts at laptop cost.
    // `rect` is threaded from pump so qualityParams + sizeCanvas share ONE
    // getBoundingClientRect per frame instead of forcing two reflows.
    const r = rect || canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, DPR_CAP);
    return policyQualityParams(f, {
      quality,
      moveQuality,
      isTouch,
      dprCap: DPR_CAP,
      dist: cam.dist,
      devicePx: r.width * r.height * dpr * dpr,
      predictedFullMs: predictFullMs(), // interactive time budget (#212)
    });
  }
  function sizeCanvas(scale, rect) {
    const r = rect || canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, DPR_CAP) * scale;
    const w = Math.max(1, Math.floor(r.width * dpr)),
      h = Math.max(1, Math.floor(r.height * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
  }
  function writeFrame(f, q, res, c) {
    // Touch wall-band work cap (#473, iPad field reports #1-#4): parked in
    // the warning band (h < 8) a passively-cooled tablet dies not on ONE
    // frame but on 30-60 s of sustained 300-500 ms near-wall grinding
    // (steps ~314 × iters ~23 measured at each death) until the OS evicts
    // the GPU process. In the band — where the badge already says detail
    // is ending — cap the ray work: the render is coarser, the device
    // survives. Desktop is untouched.
    if (isTouch && f && !f.objects && wallHeadroom() < ZOOM_BRAKE_START) {
      q = {
        ...q,
        steps: Math.min(q.steps ?? 512, 180),
        iterCap: Math.min(q.iterCap ?? Infinity, (f.iters ?? 12) + 8),
      };
    }
    // Delegates to the SHARED settle (capturesettle.js — EXPORT_P1 PR-A, true
    // delegation: the flat/scene/hybrid/morph Globals construction has no
    // second copy). The live-only state rides the options bag; the headless
    // capture driver calls settleFrame directly with none of it.
    settleFrame(renderer, f, coloring, q, res, c || cam, {
      orthoH, // #441 — 0 = perspective (the byte-identical path)
      morph,
      colorBlend,
      debugView, // #370 debug surface-quality overlay (WebGPU tier only)
      effIters: effectiveIters(f.iters),
      // Deep zoom P4 — the flat-path zoom latch. kStar > 0 is BOTH the shader
      // switchover uniform AND the variant-selection latch (activeFeat.df64).
      // df64Eligible re-checked HERE, not just in the pump's df64Update: the
      // offline export paths (stillBlob/captureFrame under setOffline) call
      // writeFrame without a pump, so a timeline that swaps formulas
      // mid-export rides a STALE df64Engaged latch. With a stale latch an
      // ineligible formula would write kStar > 0 and the df64 op switch
      // (casesDf has only twinned ops) would silently SKIP its untwinned ops
      // for the first k* iterations — a structurally wrong frame, not noise.
      // Tier gate is PER-FRAME: cheap frames (smooth/balanced interactive)
      // write kStar 0 and render the pinned f32 twin — that twin is pinned
      // for exactly this fallback — while full-quality frames (every settle,
      // and interaction under the 'full' pref) carry df64. Gating on the
      // moveQuality PREF instead used to kill df64 on settles too.
      kStarFor: (iters) =>
        df64Now() && !q.cheap && df64Eligible(f)
          ? kStarFor(zoomMag(df64Args()), lambdaHat(f), iters)
          : 0,
      // Perturbation tier (PR-4): eligibility re-checked HERE for the same
      // offline-export reason as kStarFor above, and the reference orbit is
      // (re)built + uploaded synchronously before the frame commits to the
      // pt variant — a frame can never render against a stale orbit (D5/D9).
      // Cheap interactive frames stay on the pinned f32 twin, like df64.
      // #551 — the gate records WHICH clause turned the tier away. "pt is off"
      // has four different causes with four different fixes, and none of them
      // is visible in the pixels; renderInfo.ptWhy names the one that fired.
      perturbFor: (iters) => {
        lastPtWhy = !ptNow()
          ? "latch"
          : q.cheap
            ? "cheap"
            : !ptElig(f)
              ? "ineligible"
              : !ensurePtOrbit(f, iters)
                ? "orbit"
                : "";
        return !lastPtWhy;
      },
      onFlatPayload: (payload) => {
        lastDf64 = payload.kStar > 0;
        lastPt = !!payload.perturb;
        lastFrameState = payload; // #291 standalone export reads this exact payload
      },
    });
  }

  // P2 progressive accumulation (RENDER_QUALITY §5): once a settled FULL frame
  // lands and nothing else wants to draw, keep rendering R2-jittered samples
  // into the renderer's running average — free supersampling that converges in
  // a fraction of a second, then the GPU goes fully idle at the cap. Any
  // invalidation funnels through scheduleDraw/bumpInteract (needsDraw), which
  // re-runs the normal draw below — that frame writes with weight 1, replacing
  // the average outright, so there is no stale-ghost state to clear.
  // Adaptive sample budget (post-P4 perf feedback: "everything got sluggish"):
  // a fixed 64-frame cap meant several seconds of pegged GPU after EVERY pause
  // on heavy scenes — and a new drag stalled behind the in-flight heavy frame.
  // Budget by the measured settled-frame cost instead: cheap scenes get many
  // samples (they converge fast anyway), heavy scenes get few, and very heavy
  // scenes skip refinement entirely (the base frame is already the best
  // interactive tradeoff there).
  let lastFullMs = 16; // measured settled-frame GPU+encode time
  // Diagnostics (#): the ACTUAL render tier + march params of the last drawn
  // frame, so the diag report can reveal whether a spin/drag is hitting the
  // cheap interactive tier or accidentally paying full settled cost (shadows +
  // AO + full resolution + full step count). Surfaced via renderInfo().
  let lastRender = null;
  let framesDrawn = 0; // real tier frames drawn this session (0 ⇒ pump never drew)
  let lastSkip = null; // why pump last bailed before drawing (or "drew")
  function accumCap() {
    if (lastFullMs < 20) return 32;
    if (lastFullMs < 45) return 12;
    if (lastFullMs < 100) return 6;
    return 0;
  }
  // Banded settle (#212): a settled frame that would take seconds of GPU in
  // ONE dispatch starves Chrome's compositor — the whole browser freezes.
  // Predict this frame's cost from the last measured settled frame (scaled by
  // pixel-count change) and split the march into ceil(predicted/100 ms) bands,
  // one submit per band with a rAF between (the pump loop below). A formula
  // whose cost was never measured is assumed HEAVY: banding a cheap scene
  // costs a few frames of settle latency; not banding a heavy one freezes
  // every tab in the browser.
  // Target GPU-ms per band. Each band is a fenced submit with a rAF between, so
  // this bounds how long a single settle chunk can starve the compositor (the
  // cursor). Kept small so even a multi-second settle stays interactive.
  const SETTLE_BAND_MS = 60;
  // Ceiling on band count. This was 16 — which CAPPED a heavy settle: a 5.3 s
  // frame wants ceil(5300/60) ≈ 88 bands but got 16, i.e. ~330 ms/band, and
  // because the center screen-strips carry most of the fractal the worst band
  // hit ~900 ms (measured: 6 stalls, worst 917 ms, on a "Julia Whirl Cage"
  // full settle). Raised so a multi-second frame can actually reach the 60 ms
  // target. Row-banding is non-uniform (dense center strips cost more than
  // empty ones), so the worst band is still a few× the average — a truly
  // several-second settle wants a resolution cap too, not just finer bands.
  const SETTLE_BANDS_MAX = 64;
  // Unmeasured = assume HEAVY (loose-DE presets measure ~1.4 s settled on a
  // laptop canvas): over-banding a cheap scene wastes a few frames of settle
  // latency, under-banding a heavy one freezes the browser. Capability probe
  // Phase 2a (DEFERRED_FORMULA_SWAP.md) seeds this lower ONLY on a confidently
  // fast machine — bootPredictMs() is a pure lookup, still conservative for
  // medium/slow/software/unclassified.
  const UNMEASURED_MS = bootPredictMs(capability?.tier);
  // Heavy-settle resolution cap (#): a full-detail settled frame for a very heavy
  // formula costs SECONDS at full Retina resolution (measured: 5.8 s on a 6.6 Mpx
  // canvas for "Julia Whirl Cage") — banding stops the ONE freeze but can't make
  // 5.8 s of GPU fast. When a formula's MEASURED full-res settle blows this
  // budget, render its settle at reduced resolution (floored so it never drops
  // below 1× effective on a 2× Retina display → stays crisp) instead of the full
  // 2×. Light formulas never measure over budget, so they keep full resolution.
  const SETTLE_MS_BUDGET = 1200;
  const SETTLE_SCALE_FLOOR = 0.5; // 0.5 × 2× DPR = 1× effective — Retina-crisp
  let lastFullPx = 0; // device px lastFullMs was measured at
  let lastFullWork = 0; // steps×iters lastFullMs was measured at (march work)
  let measuredFor = null; // the formula object that measurement belongs to
  // Which precision tier the measurement was taken on. A cross-tier
  // prediction (an f32-measured baseline predicting a pt frame) undercounts
  // by the pt kernel's cost factor — ~1.2-1.8× measured on M-series Metal,
  // unknown-worse on mobile TBDR — which under-bands the FIRST pt settle
  // into exactly the watchdog a weak GPU kills on (the 2026-08-01 iPad
  // field report; same shape as the df64-era "bands undercounted" bug).
  let measuredTier = "f32";
  const PT_COST_GUESS = 2.5; // conservative until a pt-tier sample lands
  // March work the CURRENT settled frame would pay. Cost is ~px·steps·iters,
  // and a per-PIXEL model alone chronically under-predicts during a deep-zoom
  // descent: depth raises steps (200→512) and auto-detail raises iters
  // (19→55+), so a measurement taken shallow lags reality by up to ~7× and
  // the settle cap never catches up (field report: 2 s settles at ×10¹³ with
  // the cap sitting at scale 0.93). Scenes keep iters factor 1 — they aren't
  // recentered, so their per-iteration cost is constant and cancels in the
  // measurement/prediction ratio.
  function marchWorkNow() {
    const qF = policyQualityParams(formula, {
      quality: "full",
      dprCap: DPR_CAP,
      dist: cam.dist,
    });
    const it =
      formula && !formula.objects
        ? Math.min(qF.iterCap ?? Infinity, effectiveIters(formula.iters))
        : 1;
    return qF.steps * Math.max(1, it);
  }
  // Predicted settled-frame cost at the CURRENT canvas size and march depth —
  // the one number both never-freeze mechanisms key off: settle banding below,
  // and the interactive time budget in renderpolicy (drag/tween frames of a
  // heavy formula are ~60% of a settled frame at scale² — see budgetScale).
  function predictFullMs() {
    const px = canvas.width * canvas.height;
    if (!(measuredFor === formula && lastFullPx > 0)) return UNMEASURED_MS;
    // Cross-tier guard: predicting a pt frame from a non-pt measurement
    // multiplies in the conservative kernel-cost factor (see measuredTier).
    const tierK = ptNow() && measuredTier !== "pt" ? PT_COST_GUESS : 1;
    return (
      ((lastFullMs * px) / lastFullPx) *
      (marchWorkNow() / Math.max(1, lastFullWork)) *
      tierK
    );
  }
  // The still path had no AbortSignal at all before the tiled export (§1.12).
  // DOMException('','AbortError') is the platform's own cancel shape, so the app
  // can tell a user cancel from a real failure with the standard `e.name` check
  // rather than a bespoke sentinel.
  function throwIfAborted(signal) {
    if (!signal || !signal.aborted) return;
    throw signal.reason || new DOMException("Export cancelled", "AbortError");
  }
  // Predicted settled-frame cost at an ARBITRARY pixel count rather than the
  // canvas's — the tiled export needs the cost of one TILE before it has
  // resized anything, so the estimate can be shown while the popover is still
  // open (TILED_EXPORT §7 Q4, decided yes). predictFullMs is linear in px by
  // construction, so this is the same prediction re-based, not a second model.
  function predictMsForPx(px) {
    const cur = Math.max(1, canvas.width * canvas.height);
    return predictFullMs() * (Math.max(1, px) / cur);
  }
  // True when predictFullMs is backed by a real measurement of THIS formula.
  // The estimate is worth much less without one (it falls back to a flat boot
  // guess), and the UI says so rather than quoting a fabricated number.
  function haveMeasurement() {
    return measuredFor === formula && lastFullPx > 0;
  }
  function settleBands() {
    if (!renderer.drawMarchBand) return 1; // GL tier: single dispatch as before
    // First pt settle (no pt-tier sample yet): slice as finely as possible
    // regardless of prediction — a wrong guess costs fence overhead; an
    // under-banded first frame costs the GPU process on a weak device.
    if (ptNow() && measuredTier !== "pt") return SETTLE_BANDS_MAX;
    return Math.max(
      1,
      Math.min(SETTLE_BANDS_MAX, Math.ceil(predictFullMs() / SETTLE_BAND_MS)),
    );
  }
  let accumN = -1; // -1 = not accumulating; ≥1 = samples in the average
  let accumStartT = 0; // refinement begins 250 ms after the settled frame
  let accumTick = 0; // refinement draws every OTHER rAF (compositor headroom)
  // Interactive frames don't fence (see pump) — they pipeline on the GPU. Cap
  // the submits in flight at 2 so a heavy scene can't queue unbounded work
  // (latency + memory) when the GPU runs slower than rAF submits arrive.
  let inFlight = 0;
  // ── Fence watchdog recovery (#460 / #473) ────────────────────────────────
  // A GPU dispatch killed by the platform watchdog fires NO device.lost and NO
  // error — queue.onSubmittedWorkDone() simply never settles (measured 2026-07-30
  // on Chrome/Metal M1 with a multi-second frame; the mobile manifestation #473
  // escalates to device-lost at the f32 wall). The fence then wedges the pump
  // FOREVER: an unresolved SETTLE await holds `busy` (its `busy = false` never
  // runs → skip:"busy"), and an unresolved interactive fence never decrements
  // inFlight → the backpressure gate latches skip:"fenced-inflight" on every
  // subsequent frame. Either way the canvas freezes until reload. So race every
  // fence against a generous timeout: on a hang the caller's cleanup still runs
  // and the render policy degrades a tier, so the retry is cheaper instead of
  // re-wedging on the same too-expensive frame.
  let fenceTimeouts = 0; // diagnostics: how many fences the watchdog killed
  // Session multiplier on the SETTLE resolution (≤1). Halved on each fence
  // timeout — the one big cost lever we own without touching the formula (device
  // px, hence GPU cost, scale as scale²) — and it also relaxes SETTLE_SCALE_FLOOR
  // so the settle can go coarser than the normal Retina floor. Persists for the
  // session: the watchdog keeps killing the same frame until it is cheap enough
  // to finish.
  let settleScaleCap = 1;
  // ── Per-frame GPU cost governor (#476) ────────────────────────────────────
  // Reactive companion to the fence-timeout recovery above: THAT fires only
  // after a 10 s+ watchdog kill; THIS watches the MEASURED settled-frame time
  // and shrinks the render scale within 2-3 frames once frames SUSTAIN past a
  // budget — so a heavy formula never REACHES the multi-second frame that kills a
  // passively-cooled mobile GPU. The 2026-08-01 iPad death: a full-tier settle
  // held scale 1 at ~950 ms/frame (steps 244 · iters 18 at dist 0.18) for ~7 min
  // — UNDER SETTLE_MS_BUDGET's desktop 1200 ms cap, so settleScaleCap stayed 1
  // and fenceTimeouts stayed 0. governorScale is the shape-preserving lever
  // (device px, hence GPU cost, scale as scale²); the settle-cap math takes the
  // min of it and settleScaleCap. Gated to coarse/mobile devices (a desktop GPU
  // doesn't die from a slow frame — no reason to nerf its heavy settles) unless a
  // test budget forces it on. Pure state machine lives in renderpolicy.js.
  const governorActive = govBudgetOverride != null || coarseMobile;
  const GOV_BUDGET = govBudgetOverride ?? GOV_BUDGET_MS;
  let governor = governorInit();
  let governorScale = 1; // mirror of governor.scale for the pump + renderInfo
  // Fold one HONEST settled-frame measurement into the governor. Interactive
  // frames are unfenced (their dt under-measures GPU) so they are NOT fed —
  // only fenced settled/banded frames, whose dt is real GPU time. On a downshift
  // record it (diag ring + onGpuTrouble → the app's "reduced detail" toast) and
  // reschedule so the now-cheaper frame paints promptly.
  function observeSettleMs(ms) {
    if (!governorActive || !(ms > 0)) return;
    const prev = governorScale;
    governor = governorStep(governor, ms, { budgetMs: GOV_BUDGET });
    governorScale = governor.scale;
    if (governorScale < prev - 1e-6) {
      noteDiag("governor", {
        frameMs: Math.round(ms),
        budgetMs: Math.round(GOV_BUDGET),
        scale: Math.round(governorScale * 100) / 100,
      });
      scheduleDraw();
    }
  }
  // Mirror renderer.js note(): append to the SAME diag ring the on-page report
  // reads, and fire onGpuTrouble so the app surfaces it live. Best-effort — the
  // WebGL2/ASCII tiers have no GPU diag object (and their fence is a synchronous
  // gl.finish() that can't hang anyway).
  function noteDiag(kind, detail) {
    try {
      const d = renderer?.getDiag?.();
      if (d && Array.isArray(d.events))
        d.events.push({ kind, detail, at: Math.round(performance.now()) });
    } catch {
      /* diag is best-effort */
    }
    try {
      opts.onGpuTrouble?.(kind, detail);
    } catch {
      /* reporting must never break the frame */
    }
  }
  // Generous by design: 4× the predicted settled cost, floored at 10 s. A
  // merely-slow-but-healthy frame must never trip it (a false timeout would
  // needlessly degrade quality and, worse, corrupt the measurement — dt would be
  // the timeout, not GPU time), which is exactly why every release below is
  // token-guarded: a late REAL completion after a timeout must not double-act.
  function fenceTimeoutMs() {
    return Math.max(10000, 4 * predictFullMs());
  }
  function onFenceTimeout(where) {
    fenceTimeouts++;
    // Halve the settle resolution (floored well below the Retina floor so a
    // genuinely too-heavy frame keeps getting cheaper each kill).
    settleScaleCap = Math.max(SETTLE_SCALE_FLOOR / 4, settleScaleCap * 0.5);
    noteDiag("fence-timeout", {
      where,
      timeoutMs: Math.round(fenceTimeoutMs()),
      settleScaleCap: Math.round(settleScaleCap * 100) / 100,
      n: fenceTimeouts,
    });
    scheduleDraw(); // RESUME: re-arm the pump so the now-cheaper frame retries
  }
  // Await a settle fence but never hang on it. Resolves true when the GPU
  // signals done (the common case — a REJECTED fence still counts as "done",
  // not a hang) and false when the timeout wins, after recording the timeout and
  // degrading policy. The lost fence promise is left pending (harmless: at most
  // one dangling microtask per kill, a rare event).
  async function fencedSettle(where) {
    let timer = 0;
    const timeout = new Promise((res) => {
      timer = setTimeout(() => res(false), fenceTimeoutMs());
    });
    const fence = renderer.device.queue.onSubmittedWorkDone().then(
      () => true,
      () => true,
    );
    const ok = await Promise.race([fence, timeout]);
    clearTimeout(timer);
    if (!ok) onFenceTimeout(where);
    return ok;
  }
  // Sample streams (subpixel R2 + P4 golden-ratio lens disk) are imported from
  // renderer.js — ONE source, no hand-mirrored copies to drift.

  async function pump() {
    scheduled = false;
    // #476 Part C: the WebGPU device is gone — STOP submitting. Every draw below
    // creates/binds resources on the dead device and throws "device mismatch"
    // (2026-08-01 iPad: device-lost, then a 5 ms-later error storm as the loop
    // kept pumping). Recovery/reinit is deferred; #477 covers the different
    // watchdog-fence hang (which fires no device-lost at all).
    if (deviceLost) {
      lastSkip = "device-lost";
      return;
    }
    // Video export: while an offline render owns the canvas, the live pump must
    // not draw — otherwise a queued rAF pump fires during captureFrame's await,
    // resizes + redraws the canvas (spin-advanced, live size) between the frame's
    // draw and its readback, so that frame captures the live view → export flicker.
    if (offline) {
      lastSkip = "offline";
      return;
    }
    if (autoRotate) {
      cam.spinAround(spinAxis(), spinSpeed);
      // Auto-rotate is continuous MOTION, so it must render in the interactive
      // tier — exactly like a manual drag (bumpInteract) or its inertia (glide
      // below). Without this the spin inherited the last SETTLED quality ("full")
      // and paid full resolution + full march steps + shadow/AO marches on every
      // frame, forever — a heavy formula spun at a few fps even on a strong GPU,
      // and the user's moveQuality pref (smooth/balanced) was silently ignored.
      // moveQuality:"full" still opts back into a full-quality showcase spin.
      quality = "low";
      needsDraw = true;
    }
    // Orbit inertia (§3.3) — the glide advances HERE, inside pump, exactly like
    // autoRotate above: an external rAF loop calling bumpInteract() would
    // double-schedule this async pump between its await and its self-reschedule
    // (double render rate). Glide frames render in the interactive tier
    // (quality low) and hand back a settled full frame when the decay ends.
    if (glide) {
      const now = performance.now();
      const g = glide.step(Math.min(100, now - glideT)); // clamp hidden-tab gaps
      glideT = now;
      cam.orbit(g.dYawDeg, g.dPitchDeg);
      if (g.active) {
        quality = "low";
      } else {
        glide = null;
        quality = "full"; // decay done → this frame IS the settled frame
      }
      needsDraw = true;
    }
    // DE-scaled cruise (§4) — also advanced in pump. Probe the surface ahead
    // every 3rd frame (CPU-DE marches aren't free at 120 Hz); between probes
    // the cached outcome keeps flying. A hit re-pins the pivot on the surface
    // ahead and approaches asymptotically; open space drifts eye+target
    // together at k·dist; no CPU DE at all (CSG scenes, unbuildable formulas)
    // degrades to a plain dolly, exactly like zoomAtCenter's probe miss.
    if (cruise && formula) {
      const now = performance.now();
      const dt = Math.min(100, now - cruise.t);
      cruise.t = now;
      if (cruise.dir > 0 && cruise.tick++ % 3 === 0) {
        const b = cam.basis();
        const h = surfaceAhead(b);
        cruise.hasHit = h != null && h > 0;
        if (cruise.hasHit) {
          // exact re-pin (D6/D11): the delta is residual-space, f64-exact
          nudgeTarget([
            b.roRel[0] + b.fwd[0] * h,
            b.roRel[1] + b.fwd[1] * h,
            b.roRel[2] + b.fwd[2] * h,
          ]);
          cam.dist = h; // eye stays put (|eye − target| = h) — seamless repin
        }
      }
      const s = cruiseAdvance(
        {
          dir: cruise.dir,
          hasHit: cruise.hasHit || getCpuDE() == null,
          dist: cam.dist,
        },
        dt,
      );
      if (s.distFactor !== 1) {
        const nd = Math.max(
          ptMinDist(),
          Math.min(200, cam.dist * s.distFactor),
        );
        // Cruise-in respects the detail-limit gate too (the one zoom path
        // the 2ad0646 brake left unbraked — recorded then as a residual
        // gap): never fly below the wall the badge declares.
        if (!(s.distFactor < 1) || wallHeadroomAt(nd) > brakeStop())
          cam.dist = nd;
      }
      if (s.drift) {
        const b = cam.basis();
        nudgeTarget([
          b.fwd[0] * s.drift,
          b.fwd[1] * s.drift,
          b.fwd[2] * s.drift,
        ]);
      }
      quality = "low";
      needsDraw = true;
    }
    // #464 — eased zoom-in chase. cam.dist and its coupled pivot glide toward
    // the goal a big re-pin set, so a large surface correction no longer snaps
    // the readout. Advanced HERE, like glide/cruise, so the ease rides the same
    // frame budget and never double-schedules. Continuous-motion inputs own the
    // camera outright — drop the chase rather than fight them.
    if (zoomChase && (glide || cruise || autoRotate || !formula)) {
      zoomChase = null;
    } else if (zoomChase) {
      const now = performance.now();
      const dt = Math.min(100, now - zoomChase.t);
      zoomChase.t = now;
      const f = 1 - Math.exp(-dt / ZOOM_EASE_TAU);
      const td = zoomChase.tDelta;
      nudgeTarget([td[0] * f, td[1] * f, td[2] * f]);
      zoomChase.tDelta = [td[0] * (1 - f), td[1] * (1 - f), td[2] * (1 - f)];
      cam.dist += (zoomChase.distGoal - cam.dist) * f;
      const near =
        Math.abs(zoomChase.distGoal - cam.dist) <= zoomChase.distGoal * 0.01 &&
        Math.hypot(...zoomChase.tDelta) <=
          Math.max(cam.dist, ptMinDist()) * 0.01;
      if (near) {
        finishZoomChase(); // land it exactly
        quality = "full"; // this frame IS the settled frame (like glide's tail)
      } else {
        quality = "low";
      }
      needsDraw = true;
    }
    if (busy || !formula) {
      lastSkip = busy ? "busy" : "no-formula";
      return;
    }
    // Accumulation branch — the third loop-continuation predicate: nothing
    // changed (needsDraw false), a settled base frame is in the average, and
    // the sample cap isn't reached.
    if (!needsDraw && accumN >= 1 && accumN < accumCap()) {
      // UI-smoothness guards ("even menu opening feels slow"): refinement is
      // strictly lower priority than EVERYTHING the user does. Start only
      // after 250 ms of true idle (an immediate next interaction never fights
      // it), and draw on every other rAF so the compositor — which needs the
      // same GPU to animate menus — always has free slots between our
      // 40–80 ms frames.
      accumTick++;
      if (performance.now() < accumStartT || (accumTick & 1) === 1) {
        lastSkip = "accum-defer";
        scheduled = true;
        requestAnimationFrame(pump);
        return;
      }
      busy = true;
      const [jx, jy] = r2jitter(accumN);
      // P4: each accumulation sample also gets a lens point (DOF converges
      // with the AA); the golden-ratio disk stream lives renderer-side.
      const [lx, ly] = lensSample(accumN);
      renderer.writeJitter(jx, jy, 1 / (accumN + 1), lx, ly);
      try {
        renderer.drawAccum();
        // Fenced, but watchdog-safe (#460): a hung refine fence would otherwise
        // never reach `busy = false` below (it's outside the try, past the
        // await) and wedge the pump. On timeout, stop accumulating; the recovery
        // reschedules the base frame.
        if (await fencedSettle("accum")) accumN++;
        else accumN = -1;
      } catch (e) {
        console.error("accum:", e);
        onRenderError("accum", e);
        accumN = -1;
      }
      busy = false;
      lastSkip = "accum-refine";
      if ((needsDraw || (accumN >= 1 && accumN < accumCap())) && !scheduled) {
        scheduled = true;
        requestAnimationFrame(pump);
      }
      return;
    }
    if (!needsDraw) {
      lastSkip = "idle-no-draw";
      return;
    }
    // The boot march variant is flat + Surface (renderer.js). A frame that needs
    // a gated feature (a color mode, a scene, a leaf, numeric-DE, hybrid, morph)
    // can't draw until that variant's pipeline exists, and compiling the big
    // variants SYNCHRONOUSLY froze the whole browser for seconds (#222). Kick the
    // async build and HOLD the frame — the previous frame stays on screen and
    // the page keeps breathing; the held frame renders the moment the compile
    // lands. Falls through to the sync compile (activeMarch) only when async
    // pipelines are unsupported (prewarmMarchFor returns null).
    if (formula) {
      ptUpdate(formula); // perturbation tier first — it supersedes df64 (D10)
      df64Update(formula); // deep zoom P4 — BEFORE frameFeatures
    }
    if (
      formula &&
      renderer.marchReadyFor &&
      !renderer.marchReadyFor(frameFeatures(formula))
    ) {
      const ff = frameFeatures(formula);
      // Announce the hold ONE frame before kicking the compile so the app's
      // "compiling" hint paints before the compositor freezes (browse only — the
      // boot hero has the splash loader). Next pump kicks the actual compile.
      if (browseMode && !compileHolding) {
        compileHolding = true;
        onCompileHold(
          true,
          ff.scene
            ? "shape"
            : ff.hybrid
              ? "hybrid"
              : ff.morph
                ? "morph"
                : "formula",
        );
        lastSkip = "compile-announce";
        scheduled = true;
        requestAnimationFrame(pump);
        return;
      }
      // Which shader to compile-and-hold on? BEFORE the first frame the boot hero
      // takes its SPECIALIZED variant (few ops → fast first paint, #271/#273).
      // AFTER that, browsing takes the GENERAL variant for FLAT (compiled once,
      // cached), and the per-formula SPECIALIZED variant for scene/hybrid/morph
      // (prewarmGeneralFor returns null → falls through) — those blobs are too
      // big for Chrome to cache as one general shader, so specialization keeps
      // them small enough to cache and far faster to compile.
      //
      // Phase 2a (DEFERRED_FORMULA_SWAP.md) — on a confidently fast/medium
      // machine, ALSO race the GENERAL variant alongside the boot hero's
      // SPECIALIZED attempt: marchReadyFor already ORs both keys and
      // activeMarch() already prefers general, so if it happens to be
      // disk-cached from a prior session (PR #278), the hero can paint from
      // that instead of waiting out a cold specialized compile — no downside
      // if it isn't (specialized was already the faster cold path). The
      // result is fire-and-forget: it doesn't gate this tick's `warm` check,
      // it just races in the background for a later tick to pick up.
      // Deliberately NOT done for slow/software — see shouldRaceGeneralAtBoot.
      if (
        !browseMode &&
        renderer.prewarmGeneralFor &&
        shouldRaceGeneralAtBoot(capability?.tier)
      ) {
        renderer.prewarmGeneralFor(ff);
      }
      const warm =
        browseMode && renderer.prewarmGeneralFor
          ? renderer.prewarmGeneralFor(ff) || renderer.prewarmMarchFor(ff)
          : renderer.prewarmMarchFor(ff);
      if (warm) {
        lastSkip = "compiling";
        scheduled = true;
        requestAnimationFrame(pump); // re-check next frame; needsDraw stays set
        return;
      }
    }
    if (compileHolding) {
      compileHolding = false;
      onCompileHold(false);
    }
    // Backpressure: the GPU is behind — encoding more would only grow the
    // queue. Keep needsDraw and retry next rAF. Heavy formulas (predicted
    // settled cost > 400 ms — their clamped interactive frames still run
    // ~100 ms each) allow only ONE frame in flight, so the compositor is at
    // most one bounded dispatch away from a slot; cheap scenes keep the
    // 2-deep pipeline for full-rate drags.
    if (inFlight >= (predictFullMs() > 400 ? 1 : 2)) {
      lastSkip = "fenced-inflight";
      scheduled = true;
      requestAnimationFrame(pump);
      return;
    }
    needsDraw = false;
    busy = true;
    onFrameStart();
    let t0 = performance.now();
    let tFull = false;
    let settleHung = false; // a single-dispatch settle fence timed out (#460)
    // The WHOLE frame — writeFrame (shader link, writeScene overflow, unknown
    // op) as well as draw/submit — is protected: a throw anywhere here used to
    // skip the `busy = false` below and latch the pump shut forever
    // (scheduleDraw no-ops while busy → permanently frozen canvas). The finally
    // guarantees busy release; the catch surfaces the cause.
    try {
      const rect = canvas.getBoundingClientRect(); // one reflow, shared below
      const q = qualityParams(formula, rect);
      // #476 governor: cap EVERY tier at the session governor scale. The settled
      // full frame is additionally floored below in the settle-cap block — which
      // includes governorScale in its floor, so it can take a settle BELOW the
      // Retina floor (the killer was a full-res settle) — but interactive/cheap
      // frames skip that block, so apply the cap here for them.
      if (governorScale < 1)
        q.scale = Math.max(GOV_SCALE_FLOOR, Math.min(q.scale, governorScale));
      // Heavy-settle resolution cap (see SETTLE_MS_BUDGET), coarse-FIRST (§A).
      // Only the SETTLED full-quality frame. UNMEASURED formula ⇒ treat as heavy
      // (cost = Infinity) so the very FIRST settle renders COARSE (the floor)
      // instead of eating a multi-second full frame before we know its cost — a
      // heavy formula's first paint drops from ~4 s to ~1 s. It then measures at
      // that scale (perPx is scale-invariant, so the full-res prediction is
      // stable and the cap converges to the floor rather than oscillating) and,
      // if it turns out UNDER budget, upgrades to full res one settle later (see
      // the tFull block). Light formulas thus end up crisp; heavy ones stay
      // capped at 1× effective on a 2× display.
      if (quality === "full" && !q.cheap) {
        const eff = Math.min(window.devicePixelRatio || 1, DPR_CAP);
        const fullPx = rect.width * eff * rect.height * eff;
        const measured = measuredFor === formula && lastFullPx > 0;
        // px AND march-work aware (see marchWorkNow): a shallow measurement
        // must not under-cap a deep settle whose steps/iters have grown.
        const fullMs = measured
          ? ((lastFullMs / lastFullPx) * fullPx * marchWorkNow()) /
            Math.max(1, lastFullWork)
          : Infinity;
        // Post-fence-timeout degrade (#460/#473): each watchdog kill halves
        // settleScaleCap and RELAXES the floor to match, so the retried settle
        // marches far fewer pixels and can actually finish. When no timeout has
        // fired (settleScaleCap === 1) this is byte-identical to the old cap:
        // budgetScale is the only term < 1, floored at SETTLE_SCALE_FLOOR.
        // #476: governorScale joins settleScaleCap as a min term AND relaxes the
        // floor to match, so a governed settle can drop below the Retina floor —
        // exactly what the iPad needed (the fatal frame was scale 1, under the
        // desktop 1200 ms budget so `over` was false; governorScale is the only
        // term that would have capped it). governorScale === 1 ⇒ byte-identical.
        const over = fullMs > SETTLE_MS_BUDGET;
        if (over || settleScaleCap < 1 || governorScale < 1) {
          const floor = Math.min(
            SETTLE_SCALE_FLOOR,
            settleScaleCap,
            governorScale,
          );
          const budgetScale = over ? Math.sqrt(SETTLE_MS_BUDGET / fullMs) : 1;
          q.scale = Math.max(
            floor,
            Math.min(q.scale, budgetScale, settleScaleCap, governorScale),
          );
        }
      }
      // Diagnostics: snapshot the tier this frame actually renders at. `ms` is
      // filled in below once the frame time is measured.
      lastRender = {
        tier: quality, // "full" (settled) | "low" (interactive/moveQuality tier)
        moveQuality, // the user's while-moving pref: smooth|balanced|full
        spin: autoRotate,
        scale: Math.round(q.scale * 100) / 100, // resolution multiplier
        steps: q.steps, // march steps per ray
        iters: formula.objects
          ? 0
          : Math.min(q.iterCap ?? Infinity, effectiveIters(formula.iters)),
        cheap: !!q.cheap, // shadow + AO marches dropped this frame?
        ms: 0,
      };
      sizeCanvas(q.scale, rect);
      writeFrame(formula, q, [canvas.width, canvas.height]);
      t0 = performance.now();
      // Accumulate only from a settled, still, full-quality frame (structural
      // WebGPU-only: the GL renderer has no drawAccum — PR-#55 fallback pattern).
      const wantAccum =
        quality === "full" && !autoRotate && !q.cheap && !!renderer.drawAccum;
      tFull = quality === "full" && !q.cheap;
      const bands = tFull ? settleBands() : 1;
      if (bands > 1) {
        // Banded settle (#212, see settleBands): one bounded submit per band,
        // fenced, with a rAF between so the compositor gets a slot. A gesture
        // arriving mid-settle flips quality to "low" — abandon the half-built
        // frame (its HDR rows are stale the moment the camera moves); the
        // interactive frames take over and the settle re-runs on the next
        // pause. Cost is measured as the SUM of band fences, not wall time —
        // wall includes our own rAF waits and would poison the accum budget.
        if (wantAccum) renderer.writeJitter(0, 0, 1, 0, 0);
        let bandMs = 0;
        let aborted = false;
        for (let i = 0; i < bands; i++) {
          if (i) await new Promise((r) => requestAnimationFrame(r));
          if (quality !== "full") {
            aborted = true;
            break;
          }
          const b0 = performance.now();
          renderer.drawMarchBand(i, bands);
          // A hung band fence (#460) aborts the whole settle: recovery has
          // already dropped the scale cap and rescheduled, so the retry re-bands
          // the frame coarser instead of submitting more bands that also hang.
          if (!(await fencedSettle("band"))) {
            aborted = true;
            break;
          }
          bandMs += performance.now() - b0;
        }
        if (aborted) {
          tFull = false; // no honest measurement — don't feed lastFullMs
          accumN = -1;
        } else {
          if (wantAccum) renderer.drawAccum({ skipMarch: true });
          else renderer.draw({ skipMarch: true });
          const r0 = performance.now();
          if (await fencedSettle("band-composite")) {
            bandMs += performance.now() - r0;
            lastFullMs = bandMs;
            lastFullPx = canvas.width * canvas.height;
            lastFullWork = lastRender
              ? lastRender.steps * Math.max(1, lastRender.iters)
              : 0;
            measuredFor = formula;
            measuredTier = lastPt ? "pt" : lastDf64 ? "df64" : "f32";
            observeSettleMs(bandMs); // #476 governor: honest banded-settle GPU time
            accumN = wantAccum ? 1 : -1;
            accumStartT = performance.now() + 250;
          } else {
            // Hung composite fence → dt is the timeout, not GPU time; don't
            // poison lastFullMs. Recovery has already rescheduled a coarser one.
            accumN = -1;
          }
        }
        tFull = false; // measurement already recorded from band fences
      } else if (wantAccum) {
        renderer.writeJitter(0, 0, 1, 0, 0); // base sample: pixel center, lens center
        renderer.drawAccum();
      } else {
        renderer.draw();
      }
      if (bands > 1) {
        // handled above — banded settle fenced per band and resolved.
      } else if (tFull || wantAccum) {
        // Settled frames keep the fence: `dt` below must be honest GPU time —
        // it feeds lastFullMs, the adaptive accumulation budget. Watchdog-safe
        // (#460): a hung fence here would hold `busy` (released only in the
        // finally, which the await would never reach) → frozen canvas. On
        // timeout, flag it so `dt` (now the timeout, not GPU time) isn't
        // recorded as the settled cost.
        if (!(await fencedSettle("settle"))) settleHung = true;
      } else if (renderer.drawAccum) {
        // Interactive tier (WebGPU — drawAccum is the tier marker, PR-#55
        // pattern): DON'T fence. Fencing every live frame serializes
        // encode → GPU → fence → next rAF, so a heavy scene degrades to
        // seconds-per-update while the page stays "smooth". Let frames
        // pipeline; count them so the gate above caps the queue at 2. The
        // callback also re-kicks the pump — when the gate deferred a frame,
        // needsDraw is still up but nothing may be scheduled.
        //
        // Watchdog-safe (#460): a hung interactive fence would never decrement
        // inFlight → the backpressure gate latches skip:"fenced-inflight" on
        // every later frame. Race it against a timeout and force the decrement.
        // Token-guarded (`settled`): the real completion and the timeout can
        // both fire, but only the first acts — a late real completion after a
        // timeout must NOT double-decrement inFlight.
        inFlight++;
        let settled = false;
        let fenceTimer = 0;
        const done = (timedOut) => {
          if (settled) return;
          settled = true;
          if (fenceTimer) clearTimeout(fenceTimer);
          inFlight--;
          if (timedOut) onFenceTimeout("interactive");
          if (needsDraw && !scheduled && !busy) {
            scheduled = true;
            requestAnimationFrame(pump);
          }
        };
        renderer.device.queue.onSubmittedWorkDone().then(
          () => done(false),
          () => done(false),
        );
        fenceTimer = setTimeout(() => done(true), fenceTimeoutMs());
      }
      // WebGL2 interactive frames get NO fence at all: the GL shim's fence is
      // a synchronous gl.finish() (a main-thread block), and GL's swap chain
      // already backpressures unfenced work.
      if (bands <= 1) {
        // (banded settles did their own accum bookkeeping above)
        // A hung settle fence must not seed accumulation from an unfinished
        // base frame (#460) — the recovery will redraw a coarser one.
        accumN = !settleHung && wantAccum ? 1 : -1;
        accumStartT = performance.now() + 250;
      }
    } catch (e) {
      console.error("draw:", e);
      onRenderError("draw", e);
      accumN = -1;
    } finally {
      busy = false;
    }
    const dt = performance.now() - t0;
    if (lastRender) lastRender.ms = Math.round(dt); // diagnostics: this frame's cost
    framesDrawn++; // diagnostics: a real tier frame reached the GPU
    lastSkip = "drew";
    // WebGL2 tier health gate: a dead GL tier draws BLACK while the lines above
    // report "drew" with fast ms — exactly the iOS-15 field signature. Ask the
    // pure classifier; on death, stop the GPU pump (hasGPU=false → scheduleDraw
    // no-ops) and hand the app the reason to fall to the ASCII view instead of a
    // permanent black canvas. GL tier only, first few frames only — zero cost on
    // WebGPU and on a healthy GL device.
    if (backend === "webgl2" && !glDead && framesDrawn <= GL_HEALTH_WINDOW) {
      const h = renderer.glHealth?.();
      if (h?.dead) {
        glDead = true;
        hasGPU = false;
        gpuDeadFired = true; // #473 — one demotion per session, either path
        noteDiag("gl-fallback", { reason: h.reason });
        try {
          opts.onGpuDead?.(h.reason);
        } catch {
          /* the app's fallback must never re-enter the pump */
        }
      }
    }
    if (tFull && !settleHung) {
      lastFullMs = dt; // feeds the adaptive accumulation budget
      lastFullPx = canvas.width * canvas.height;
      lastFullWork = lastRender
        ? lastRender.steps * Math.max(1, lastRender.iters)
        : 0;
      measuredFor = formula;
      observeSettleMs(dt); // #476 governor: honest single-settle GPU time
    }
    // Coarse-first upgrade (§A) — runs for BOTH single (tFull) and BANDED settles.
    // A banded settle sets tFull=false but records its OWN lastFullMs/lastFullPx/
    // measuredFor from the band fences (above), so gating this on `if (tFull)`
    // silently skipped it for every settle longer than ~one band (~60 ms) — which
    // is nearly all of them — leaving borderline/light formulas stuck at the coarse
    // floor (measured: M4 Julia pinned at scale 0.5 when full-res was under budget).
    // Gate on the RECORDED measurement instead: a full-quality settle that rendered
    // capped (scale < 1) whose full-res cost is under budget was a light formula
    // rendered needlessly coarse → schedule ONE more settle, which the cap (now
    // measured, under budget → no cap) renders at full resolution. Heavy formulas
    // (full-res cost > budget) fail the check → no re-trigger, no loop; a full-res
    // settle (scale 1) never enters.
    if (
      measuredFor === formula &&
      lastFullPx > 0 &&
      quality === "full" &&
      lastRender &&
      !lastRender.cheap &&
      lastRender.scale < 1 &&
      // #476: don't fight the governor. When it holds scale < 1 the re-settle
      // would render capped again (scale < 1) and re-trigger forever — an
      // upgrade loop the desktop path never hits (its cap only engages when
      // fullResMs > budget, which fails this check). Yield the upgrade to it.
      governorScale >= 1
    ) {
      const fullResMs = lastFullMs / (lastRender.scale * lastRender.scale);
      if (fullResMs <= SETTLE_MS_BUDGET) {
        needsDraw = true;
        scheduleDraw();
      }
    }
    onFrame(dt);
    // The hero is on screen (specialized, fast). From here every new formula
    // takes the compile-once GENERAL variant so browsing never compiles per
    // preset (the guard above). Only reached after a real draw — held frames
    // return earlier.
    browseMode = true;
    if (
      needsDraw ||
      autoRotate ||
      glide ||
      cruise ||
      zoomChase || // #464 — keep pumping while an eased re-pin is settling
      (accumN >= 1 && accumN < accumCap())
    ) {
      scheduled = true;
      requestAnimationFrame(pump);
    }
  }

  // ── Zoom-to-surface (§5) ──────────────────────────────────────────────────
  // Plain zoom just shrinks cam.dist, moving the eye toward the orbit target —
  // which for an unpanned camera is the object's CENTROID. Zoom in far enough
  // and the eye flies THROUGH the surface into the interior, where every ray
  // hits an interior wall at a near-uniform angle and the frame washes out to a
  // flat colour (the "why does it go single-colour so fast" report). Deep zoom's
  // Shift+drag pan fixes it manually, but nobody discovers that — so plain zoom
  // must Just Work: on zoom IN, glide the orbit target onto the surface point
  // straight ahead, so zoom dollies TOWARD the surface (asymptotically, never
  // through it) instead of toward the centroid.
  // Deep zoom P4 fix (plan PR-3 as-built notes): the probe must march the
  // SAME surface the GPU draws. Auto-detail boosts iters with depth, and the
  // base-iters surface diverges from the boosted one by more than a deep
  // viewport spans (past ~×10⁸ the old probe pinned the pivot onto a surface
  // the render doesn't show — zooming steered into empty space). Memoize on
  // (formula, effective iters) so the probe tracks the boost as it climbs.
  let cpuDEIters = 0;
  const getCpuDE = () => {
    const eff =
      formula && !formula.objects
        ? effectiveIters(formula.iters)
        : (formula?.iters ?? 0);
    if (cpuDEFor === formula && cpuDEIters === eff) return cpuDE;
    cpuDEFor = formula;
    cpuDEIters = eff;
    try {
      cpuDE = formula ? _cpuMod.makeDE(formula, eff || undefined) : null;
    } catch {
      cpuDE = null; // some formula shape cpu.js can't build → fall back to plain zoom
    }
    return cpuDE;
  };
  // Distance from `eye` along `fwd` to the surface straight ahead (null if the
  // ray misses or the eye is already inside) — the probe lives in the pure,
  // unit-tested zoomsurface.js; here we just feed it the current DE and the
  // dist-scaled near/far bounds (matching the shader's tNear/tFar).
  // Depth-aware since PR-4 (plan D8): probeFrame switches the evaluator AND
  // the marching origin to residual space once the pt tier owns the depth.
  const surfaceAhead = (b) => {
    const P = probeFrame(b);
    return surfaceHitDist(
      P.de,
      P.o,
      b.fwd,
      cam.dist * TNEAR_K,
      cam.dist * (TFAR_K * 1.5),
    );
  };
  // One zoom step. On zoom IN, re-pin the orbit target to the surface straight
  // ahead first (keeps the eye exactly put — target = eye + fwd·h, dist = h —
  // so it's visually seamless), THEN apply the factor so the eye moves a
  // fraction of the way toward that surface point. Zoom OUT is left as a plain
  // dolly so backing away never drifts the pivot unexpectedly.
  //
  // The ahead-probe is THROTTLED to one march per 150 ms: this fires per input
  // event (pinch pointermove at 60–120 Hz, keyboard auto-repeat via zoomBy),
  // and surfaceAhead is a full CPU-DE march — the same cost the wheel path
  // already caches (wheelProbe) and cruise already throttles (every 3rd
  // frame). Skipped events stay exact: a hit re-pins the pivot ONTO the
  // surface, so the next zoom-in dollies toward that same point; staleness
  // (e.g. a simultaneous two-finger pan turning the view) is bounded by the
  // 150 ms window because the throttle does NOT slide.
  // Deep zoom — progressive zoom-IN brake at the precision wall (field
  // report 2026-07-31: with df64 ineligible, wheel zoom free-fell to
  // ×2.6·10¹¹ of featureless mush at 1 fps — "feels almost blocked" — the
  // render can't show anything real past the wall, so the ZOOM stops
  // instead). The law is the badge's: headroom vs the wall THIS formula can
  // actually render — the df64 quantum when df64 can engage, f32 otherwise.
  // Zoom-out is never braked; scenes are exempt (no recenter, no law).
  // Braking starts at h = 8 (the measured false-plate onset band, also the
  // df64 engage threshold) and hard-stops just above wall-grade mush.
  const ZOOM_BRAKE_START = 8;
  // The hard stop is FORMULA-CLASS dependent, matching the badge's wall
  // thresholds exactly (field request 2026-08-01: "gate going over the
  // detail limit"). With a deep tier engageable (pt or df64) the wall law
  // already IS that tier's quantum, so stopping at 1.2 sits just above the
  // real wall. Without one (f32-only — including pt-eligible formulas on a
  // pt-dark device), the badge declares "past detail limit" at h ≤ 3 (the
  // measured false-plate onset): stopping at 1.2 left a reachable band
  // (3 → 1.2) that renders only quantization noise — and on mobile GPUs
  // grinds full-tier settles into the compositor watchdog (the iPad
  // device-lost report, #473). Zoom-in now stops where the badge says the
  // detail ends; zoom-out is never braked.
  const brakeStop = () => {
    const deepTier =
      !!formula &&
      ((ptMode !== "off" && ptElig(formula)) ||
        (df64Mode !== "off" && df64Eligible(formula)));
    return deepTier ? 1.2 : 3;
  };
  // Headroom against the wall THIS formula can actually render (df64 quantum
  // when df64 can engage, f32 otherwise) — the brake's and the re-pin
  // guards' shared law.
  function wallHeadroom() {
    return wallHeadroomAt(cam.dist);
  }
  // Headroom the camera WOULD have at distance d — the re-pin guards ask
  // this about a candidate distance before accepting it.
  function wallHeadroomAt(d) {
    if (!formula || formula.objects) return Infinity;
    // Perturbation tier (PR-4): its wall is an ABSOLUTE pixel-footprint
    // floor with no |T| term (recenter.js PT_FLOOR) — the |T|-relative
    // quantum laws apply only when pt cannot engage.
    if (ptMode !== "off" && ptElig(formula))
      return ptHeadroom({ ...df64Args(), dist: d });
    const q =
      df64Mode !== "off" && df64Eligible(formula) ? DF64_QUANTUM : F32_QUANTUM;
    return zoomHeadroom({ ...df64Args(), dist: d }, q);
  }
  function brakeZoomIn(factor) {
    if (!(factor < 1) || !formula || formula.objects) return factor;
    const h = wallHeadroom();
    if (!(h > 0) || h >= ZOOM_BRAKE_START) return factor;
    if (h <= brakeStop()) return 1; // at the wall: zoom-in is a no-op
    // log-space ease: full speed at the band edge, asymptotically gentle
    // approaching the stop — the "progressively slow down" feel.
    const s =
      Math.log(h / brakeStop()) / Math.log(ZOOM_BRAKE_START / brakeStop());
    return Math.pow(factor, s * s);
  }
  // The deepest dist this formula may reach: the wall distance (headroom ==
  // brakeStop, the SAME law the soft brake stops at), floored at the numeric
  // tier floor. It is the HARD backstop the brake can't be — the brake gates on
  // PRE-zoom headroom, so one oversized step (a big anchored pinch from far out,
  // a stale-resume gesture) computes a goal PAST the wall and eases there. Wall
  // headroom is linear in dist (recenter.js / ptHeadroom), so the wall distance
  // is dist·brakeStop/headroom(dist) — independent of the current dist, exact
  // from any probe. Scenes / pre-formula / no-wall → just the numeric floor.
  // Installed as the camera's distFloor so EVERY cam.dist write is clamped, and
  // reused at the zoom-chase goal so the ease settles AT the wall (not past it,
  // and never stuck below its own clamped goal).
  const wallFloorDist = () => {
    const h = wallHeadroom();
    if (!formula || formula.objects || !(h > 0) || !Number.isFinite(h))
      return ptMinDist();
    return Math.max(ptMinDist(), (cam.dist * brakeStop()) / h);
  };
  cam.distFloor = wallFloorDist;
  // #464 — controlled zoom-IN. A single wheel/pinch/key notch used to write
  // cam.dist the instant it fired, with two failure modes at depth:
  //   1. probe MISS → the eye free-fell toward the centroid, un-anchored, so a
  //      fast trackpad/pinch could descend with no surface feedback ("framing
  //      rides on luck"); the wheel is already ≤ ×1.25/notch (gestures.js) but
  //      pinch/keyboard are unclamped, so an un-anchored notch is now CAPPED.
  //   2. probe HIT after a miss streak → `cam.dist = h` SNAPPED the readout by
  //      orders in one event (the re-pin "lurch"). A jump that big now EASES to
  //      its goal over a few frames (advanced in pump(), like glide/cruise — the
  //      camera-controls rule) instead of snapping. A normal shallow re-pin
  //      (the probe agrees with where we already are) still lands instantly, so
  //      the common gesture is byte-for-byte unchanged.
  const UNANCHORED_MAX = 1.5; // cap effective zoom-IN per un-anchored notch
  const ZOOM_EASE_TAU = 45; // ms — exp-smoothing constant for eased re-pins (~2 frames)
  // A re-pin lands INSTANTLY while the probe distance is within this ratio of
  // the current dist (no visible jump to smooth); outside it the correction is
  // a "big jump" and rides the chase.
  const REPIN_INSTANT_LO = 0.6,
    REPIN_INSTANT_HI = 1.6;
  // Eased zoom-in state: cam.dist glides toward distGoal while the residual
  // pivot delta tDelta drains, both at the same per-frame rate so the EYE moves
  // smoothly forward (target and dist stay coupled — no lurch). Advanced in
  // pump(); any non-zoom camera input finishes it (finishZoomChase).
  let zoomChase = null; // { distGoal, tDelta:[x,y,z], t } | null
  function finishZoomChase() {
    if (!zoomChase) return;
    nudgeTarget(zoomChase.tDelta);
    cam.dist = zoomChase.distGoal;
    zoomChase = null;
  }
  let centerProbeT = 0;
  const zoomAtCenter = (factor) => {
    const braked = brakeZoomIn(factor);
    if (braked === 1) return; // braked to a stop — keep detailOverride intact
    detailOverride = null; // §6 — a zoom hands detail back to auto-detail
    entryClampArm.disarm(); // #562 — a real interactive zoom cancels any pending arm
    // #489 — under ortho, "zoom" is the frustum half-height (orthoH), not
    // camera distance: an orthographic render is BY DEFINITION invariant to
    // eye distance, so every branch below (dolly, anchored eye-march) is
    // silently invisible — the reported "zoom doesn't work in Top/Side/Front,
    // works fine in Perspective". Scale orthoH by the same wall-braked factor
    // instead, center-anchored (cursor-anchoring, like the wheel handler's
    // probe below, is a reasonable follow-up — not required to fix "does
    // nothing at all").
    if (orthoH > 0) {
      orthoH = Math.max(ptMinDist(), orthoH * braked);
      bumpInteract();
      return;
    }
    if (!(braked < 1) || !formula) {
      // Zoom-OUT (or no formula): a plain dolly, never anchored or eased —
      // backing away must stay crisp and immediate (#464: zoom-out is instant).
      finishZoomChase();
      cam.zoom(braked, ptMinDist());
      bumpInteract();
      return;
    }
    const b = cam.basis();
    let anchor = null; // surface distance the eye should approach, or null (miss)
    // At/past the wall the ahead-probe must NOT re-pin: in the dust regime the
    // eye is always within microns of SOME grain, so a re-pin would teleport
    // the readout decades deeper in one event, and past the wall nothing the
    // probe anchors to is real anyway. The probe is a full CPU-DE march, so it
    // is throttled to one per 150 ms (#364/#464).
    const now = performance.now();
    if (now - centerProbeT >= 150 && wallHeadroom() > brakeStop()) {
      centerProbeT = now;
      const h = surfaceAhead(b);
      // A re-pin may only LAND above the wall stop (measured: 5.4e-4 → 3.6e-7
      // in a single burial re-pin at headroom 1.35 — #464's lurch at its worst).
      const landsAboveWall = (d) => wallHeadroomAt(d) > brakeStop();
      if (h != null && h > 0 && landsAboveWall(h)) {
        anchor = h;
      } else if (h == null || !(h > 0)) {
        // Deep zoom P4 burial recovery (#364): a fast zoom can outrun the
        // 150 ms throttle and carry the eye to within the probe's NEAR bound of
        // the surface, from where surfaceAhead's inside-guard returns null on
        // every later call and zoom degrades to a blind dolly INTO the wall.
        // One extra CPU-DE eval detects it — the "re-probe before a big jump"
        // reuse (#464): a surface closer than ~the near plane but still ahead
        // re-pins to the TRUE distance so the asymptotic approach resumes.
        const P = probeFrame(b);
        const dEye = P.de ? P.de(P.o[0], P.o[1], P.o[2]) : null;
        if (
          dEye != null &&
          dEye > 0 &&
          dEye <= cam.dist * TNEAR_K * 4 &&
          landsAboveWall(dEye)
        )
          anchor = dEye;
      }
    }
    if (anchor != null) {
      // Approach the surface point straight ahead. The re-pin delta
      // (roRel + fwd·anchor) moves the pivot onto the surface with the eye put;
      // the factor then dollies the eye a fraction of the way in — goal dist
      // anchor·braked. Delta is residual-space, f64-exact at its own scale (D6).
      const delta = [
        b.roRel[0] + b.fwd[0] * anchor,
        b.roRel[1] + b.fwd[1] * anchor,
        b.roRel[2] + b.fwd[2] * anchor,
      ];
      // Floor the goal at the wall too (not just the numeric tier floor): a
      // large factor when far from the wall (h ≥ START, so brakeZoomIn passed it
      // through) makes anchor·braked land PAST the wall, and the ease would
      // otherwise chase it there. Clamping here also keeps the chase's own goal
      // in agreement with cam.dist's hard clamp, so it settles instead of
      // sticking one clamp above an unreachable goal.
      const goalDist = Math.max(
        wallFloorDist(),
        Math.min(200, anchor * braked),
      );
      const corr = anchor / cam.dist; // how far the probe is from where we are
      if (!zoomChase && corr > REPIN_INSTANT_LO && corr < REPIN_INSTANT_HI) {
        // Small correction (the probe agrees with the current dist) → land it
        // now, exactly as before the ease existed: eye stays put, pivot rides
        // the surface, dist becomes anchor·braked.
        nudgeTarget(delta);
        cam.dist = goalDist;
      } else {
        // Big jump (free-fall recovery / dust regime), or a chase already in
        // flight → ease toward the anchored goal instead of snapping. tDelta is
        // the WHOLE remaining move measured from the current camera, so a fresh
        // probe simply replaces it.
        zoomChase = zoomChase || {
          distGoal: goalDist,
          tDelta: [0, 0, 0],
          t: now,
        };
        zoomChase.tDelta = delta;
        zoomChase.distGoal = goalDist;
      }
    } else {
      // Un-anchored: cap the per-notch step so a fast trackpad/pinch can't
      // free-fall between the throttled surface probes (#464 point 1).
      const cf = Math.max(braked, 1 / UNANCHORED_MAX);
      if (zoomChase)
        zoomChase.distGoal = Math.max(wallFloorDist(), zoomChase.distGoal * cf);
      else cam.dist = Math.max(ptMinDist(), cam.dist * cf);
    }
    bumpInteract();
  };
  // The world-space ray through a client pixel (the shader's ray generation,
  // CPU-side) + the basis it was built from — shared by double-click zoom and
  // the cursor-anchored wheel zoom (§3.2).
  const pixelRay = (clientX, clientY) => {
    const rect = canvas.getBoundingClientRect();
    const ndcX = (2 * (clientX - rect.left)) / (rect.width || 1) - 1;
    const ndcY = 1 - (2 * (clientY - rect.top)) / (rect.height || 1);
    const aspect = (rect.width || 1) / (rect.height || 1);
    const tanF = Math.tan(0.5 * cam.fov);
    const b = cam.basis();
    let rx =
      b.fwd[0] + ndcX * aspect * tanF * b.right[0] + ndcY * tanF * b.up[0];
    let ry =
      b.fwd[1] + ndcX * aspect * tanF * b.right[1] + ndcY * tanF * b.up[1];
    let rz =
      b.fwd[2] + ndcX * aspect * tanF * b.right[2] + ndcY * tanF * b.up[2];
    // #441 — under ortho every pixel has its OWN origin and a shared direction.
    // Callers must use `ro`, not b.eye: marching from the single eye would probe
    // screen-centre wherever the user actually clicked, silently.
    if (orthoH > 0) {
      const sxw = ndcX * aspect * orthoH,
        syw = ndcY * orthoH;
      const roOfs = [
        b.right[0] * sxw + b.up[0] * syw,
        b.right[1] * sxw + b.up[1] * syw,
        b.right[2] * sxw + b.up[2] * syw,
      ];
      const ro = [
        b.eye[0] + roOfs[0],
        b.eye[1] + roOfs[1],
        b.eye[2] + roOfs[2],
      ];
      // roOfs: the origin's offset FROM THE EYE — probe consumers add it to
      // probeFrame's origin so the march works in whichever space (absolute
      // or residual, plan D8) the depth calls for.
      return { b, ray: b.fwd.slice(), ro, roOfs };
    }
    const rl = Math.hypot(rx, ry, rz) || 1;
    return {
      b,
      ray: [rx / rl, ry / rl, rz / rl],
      ro: b.eye.slice(),
      roOfs: [0, 0, 0],
    };
  };
  // Probe the surface along a pixel's ray and re-pin the orbit pivot onto the
  // hit point (eye stays put: target = eye + ray·h ⇒ |eye−target| = h = dist).
  // Returns true on a hit. Double-click only — the off-axis repin ROTATES the
  // view onto the feature, which is that gesture's intent; wheel zoom must
  // stay seamless and uses the translate-along-ray anchor below instead.
  const repinToPixel = (clientX, clientY) => {
    const { b, ray, roOfs } = pixelRay(clientX, clientY);
    const P = probeFrame(b);
    const h = surfaceHitDist(
      P.de,
      // #441: the PER-PIXEL origin — probe-space eye + the pixel offset
      [P.o[0] + roOfs[0], P.o[1] + roOfs[1], P.o[2] + roOfs[2]],
      ray,
      cam.dist * TNEAR_K,
      cam.dist * (TFAR_K * 1.5),
    );
    if (h == null || h <= 0) return false;
    // exact re-pin: newTarget − target = roRel + pixel offset + ray·h
    nudgeTarget([
      b.roRel[0] + roOfs[0] + ray[0] * h,
      b.roRel[1] + roOfs[1] + ray[1] * h,
      b.roRel[2] + roOfs[2] + ray[2] * h,
    ]);
    cam.dist = h;
    return true;
  };
  // Double-click / double-tap to zoom toward the clicked POINT (map-style),
  // reusing the same surface probe but along the ray through the clicked pixel
  // instead of the centre. On a surface hit: recenter the orbit target on that
  // feature AND pull the eye `factor`× closer to it in one step. A miss (clicked
  // empty space) falls back to a plain centre zoom so the gesture never no-ops.
  const zoomToPixel = (clientX, clientY, factor) => {
    factor = brakeZoomIn(factor);
    if (factor === 1) return; // braked to a stop — see zoomAtCenter
    finishZoomChase(); // a double-click owns the pivot — end any eased zoom (#464)
    detailOverride = null; // §6 — see zoomAtCenter
    entryClampArm.disarm(); // #562 — see zoomAtCenter
    const before = cam.dist; // repin overwrites dist with the hit distance
    if (repinToPixel(clientX, clientY)) {
      // Pivot re-pinned on the clicked surface point; zoom by a FIXED factor of
      // the PRIOR distance (not h) — so every double-click is a predictable ~2×,
      // whether you clicked a near or far feature, and the pivot rides the
      // surface (like wheel zoom) so the eye can't cross into it.
      cam.dist = Math.max(ptMinDist(), before * factor);
      bumpInteract();
    } else {
      cam.zoom(factor, ptMinDist()); // clicked empty space → plain zoom, never a dead gesture
      bumpInteract();
    }
  };

  // ── Orbit / zoom / pinch gestures ─────────────────────────────────────────
  const ptrs = new Map();
  canvas.addEventListener("pointerdown", (e) => {
    ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY, t: e.timeStamp });
    glide = null; // grabbing the canvas stops a coast dead (§3.3)
    finishZoomChase(); // a new gesture (orbit/pan/pinch) ends any eased zoom (#464)
    if (ptrs.size === 1) {
      dragVel.reset();
      dragMode = null;
    }
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointerup", (e) => {
    ptrs.delete(e.pointerId);
    canvas.releasePointerCapture?.(e.pointerId);
    // Release of the LAST pointer after an orbit drag: coast if it was a flick
    // (§3.3). Pan/pinch drags and reduced-motion release dead, as today.
    if (ptrs.size === 0) {
      if (
        dragMode === "orbit" &&
        !cruise && // steering releases stop dead — the cruise owns the motion
        dragVel.speed() >= GLIDE_MIN_SPEED &&
        !reducedMotion()
      ) {
        const { vx, vy } = dragVel.velocity();
        glide = makeGlide(vx, vy);
        glideT = performance.now();
        clearTimeout(settleTimer); // pump owns quality until the glide ends
        scheduleDraw();
      }
      dragMode = null;
    }
  });
  // iOS/Safari fires pointercancel (NOT pointerup) when the system steals an
  // active touch — a second finger landing, an edge-swipe, a scroll or OS
  // gesture. Without handling it the pointer stays in `ptrs` and dragMode never
  // resets, so orbit/pan/pinch go dead until reload ("lost focus"). Treat it as a
  // hard release: drop the pointer, no coast (a cancel isn't a flick).
  const cancelPointer = (e) => {
    ptrs.delete(e.pointerId);
    canvas.releasePointerCapture?.(e.pointerId);
    if (ptrs.size === 0) {
      dragMode = null;
      glide = null;
      dragVel.reset();
    }
  };
  canvas.addEventListener("pointercancel", cancelPointer);
  // Screen px → world units in the view plane (2·dist·tan(fov/2) / height), so
  // a pan tracks the surface under the pointer regardless of zoom depth. Shared
  // by Shift+drag and the two-finger pan.
  const panByPixels = (dxPx, dyPx) => {
    finishZoomChase(); // a pan moves the pivot directly — end any eased zoom (#464)
    const rect = canvas.getBoundingClientRect();
    const wPerPxY = (2 * cam.dist * Math.tan(cam.fov / 2)) / (rect.height || 1);
    const wPerPxX = wPerPxY * ((rect.width || 1) / (rect.height || 1));
    cam.pan(-dxPx * wPerPxX, dyPx * wPerPxY);
  };
  canvas.addEventListener("pointermove", (e) => {
    if (!ptrs.has(e.pointerId)) return;
    if (ptrs.size >= 2) {
      // Two fingers = pan + zoom SIMULTANEOUSLY (§3.4, standard map behavior):
      // centroid translation pans the orbit target, pinch distance zooms. This
      // is the only pan touch has (Shift doesn't exist on phones).
      dragMode = "pinch";
      const [a0, b0] = [...ptrs.values()];
      ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY, t: e.timeStamp });
      const [a1, b1] = [...ptrs.values()];
      const g = twoFingerDelta(a0, b0, a1, b1);
      if (g.panX || g.panY) panByPixels(g.panX, g.panY);
      if (g.zoom !== 1) zoomAtCenter(g.zoom);
      else bumpInteract(); // pan-only move still needs the interactive redraw
      return;
    }
    const p = ptrs.get(e.pointerId);
    // Pan the orbit target (deep zoom §5 navigation) — Shift+drag, so it doesn't
    // collide with the default plain-drag orbit. #489 round 4: middle-button
    // drag (`e.buttons & 4`) is the other standard-3D-app pan trigger — check
    // it here too. `buttons` bit 4 is mouse-only (pen tip/eraser and touch
    // contact both report bit 1), so primary-button drag and touch/pen drags
    // are unaffected and still fall through to orbit below. Routing middle-
    // drag to pan (not orbit) also means it does NOT drop orthographic (#441
    // — only ORBIT drops ortho), which is the point: Top/Side/Front views can
    // now be panned with the middle button without losing the ortho projection.
    if (e.shiftKey || (e.buttons & 4) !== 0) {
      dragMode = "pan";
      panByPixels(e.clientX - p.x, e.clientY - p.y);
      ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY, t: e.timeStamp });
      bumpInteract();
      return;
    }
    // Orbit speed scales with zoom (dist): finer when zoomed in, never
    // sluggish. Halved while cruising (§4) — the drag is steering the nose.
    dragMode = "orbit";
    // #441 — reaching for ORBIT is asking for 3D space back, so an orbit drops
    // orthographic. Only orbit: pan, wheel and pinch are inspecting within the
    // projection you chose and keep it. (One-finger drag is orbit, so touch
    // always has this gesture to hand — no need to make pinch revert too.)
    if (orthoH > 0) {
      orthoH = 0;
      scheduleDraw();
    }
    const s =
      (cruise ? 0.2 : 0.4) * Math.max(0.4, Math.min(1.3, cam.dist / 18));
    const dx = (e.clientX - p.x) * s,
      dy = (e.clientY - p.y) * s;
    cam.orbit(dx, dy);
    dragVel.push(dx, dy, e.timeStamp - p.t); // release velocity for the glide (§3.3)
    ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY, t: e.timeStamp });
    bumpInteract();
  });
  // Wheel zoom (§3.1/§3.2): delta-proportional factor (trackpads and notched
  // wheels finally agree — see gestures.js), anchored on the CURSOR when
  // zooming in. NO-JUMP anchor: an off-centre pivot repin (à la double-click)
  // would translate the derived eye sideways on the first tick — the orbit eye
  // is target + u(yaw,pitch)·dist, so moving the pivot off the view axis with
  // fixed angles MOVES the eye. Instead, slide the EYE straight down the
  // cursor ray by δ = h·(1−factor) with orientation untouched, which in orbit
  // state is exactly:  target += ray·δ + u·dist·(1−factor);  dist ×= factor
  // (⇒ eye′ = eye + ray·δ — pure translation, seamless, and asymptotic: the
  // remaining ray distance is h·factor > 0, so the eye never crosses in).
  // Zoom OUT stays a plain dolly — backing away must never drift the pivot.
  // The probe is cached per gesture (events < 150 ms apart, cursor within
  // 8 px; h decremented by each δ, which stays exact along one ray) — CPU-DE
  // marches aren't free at trackpad event rates; a MISS is cached too, so
  // empty-space scrolling degrades to the centre probe without re-marching.
  let wheelProbe = null; // { t, x, y, ray, h } — cursor probe; h null = miss
  canvas.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const factor = wheelZoomFactor(e.deltaY, e.deltaMode, rect.height || 800);
      // #489 — the cursor-anchored eye-march below is perspective-only math
      // (it slides the EYE down the cursor ray, which an ortho render can't
      // see); route ortho straight to zoomAtCenter's orthoH scaling instead
      // of probing for a surface hit it would just have to discard. Skips
      // the CPU-DE probe entirely under ortho, too — it was never going to
      // be used.
      if (orthoH > 0) {
        wheelProbe = null;
        zoomAtCenter(factor);
        return;
      }
      if (factor >= 1 || !formula) {
        wheelProbe = null;
        zoomAtCenter(factor);
        return;
      }
      // Wall stop, checked BEFORE the cursor probe: at the precision wall the
      // brake discards the zoom anyway, and the probe is a full CPU-DE march
      // per wheel event — at depth that march creeps through mush and stalls
      // the MAIN THREAD (the "whole machine sluggish" half of the 2026-07-31
      // report). No zoom ⇒ no probe ⇒ no cost.
      if (brakeZoomIn(factor) === 1) return;
      const now = performance.now();
      const fresh =
        wheelProbe &&
        now - wheelProbe.t < 150 &&
        Math.hypot(e.clientX - wheelProbe.x, e.clientY - wheelProbe.y) < 8;
      if (!fresh) {
        const { b, ray, roOfs } = pixelRay(e.clientX, e.clientY);
        const P = probeFrame(b);
        const h = surfaceHitDist(
          P.de,
          [P.o[0] + roOfs[0], P.o[1] + roOfs[1], P.o[2] + roOfs[2]], // #441 per-pixel origin
          ray,
          cam.dist * TNEAR_K,
          cam.dist * (TFAR_K * 1.5),
        );
        wheelProbe = {
          t: now,
          x: e.clientX,
          y: e.clientY,
          ray,
          h: h != null && h > 0 ? h : null,
        };
      } else {
        wheelProbe.t = now; // sliding window: a continuous scroll keeps the anchor
      }
      if (wheelProbe.h != null) {
        // The cursor found the surface — hand off from any eased center chase
        // (from a prior miss streak) to this seamless per-notch slide (#464).
        finishZoomChase();
        detailOverride = null; // §6 — a zoom hands detail back to auto-detail
        entryClampArm.disarm(); // #562 — see zoomAtCenter
        const k = 1 - factor;
        const delta = wheelProbe.h * k;
        const b = cam.basis();
        const back = cam.dist * k;
        // already delta-form — exact through the BigInt target (D6)
        nudgeTarget([
          wheelProbe.ray[0] * delta - b.fwd[0] * back,
          wheelProbe.ray[1] * delta - b.fwd[1] * back,
          wheelProbe.ray[2] * delta - b.fwd[2] * back,
        ]);
        cam.dist = Math.max(ptMinDist(), cam.dist * factor);
        wheelProbe.h -= delta; // the eye advanced δ along the ray — keep it exact
        bumpInteract();
      } else {
        zoomAtCenter(factor); // cursor over empty space → centre-probe zoom
      }
    },
    { passive: false },
  );
  // Double-click to zoom toward the clicked point (~2× per double-click). Alt/⌥
  // double-click zooms back OUT around the same point, so it's a reversible probe.
  canvas.addEventListener("dblclick", (e) => {
    e.preventDefault();
    zoomToPixel(e.clientX, e.clientY, e.altKey ? 2.0 : 0.5);
  });
  // Resize rides the interactive tier: a live window-drag fires resize per
  // event, and each size change also reallocates the HDR/accum textures — at
  // settled quality that's a full-cost frame per tick. bumpInteract renders
  // them cheap and settles to full 200 ms after the last one.
  window.addEventListener("resize", bumpInteract);
  // App-switch / tab-hide hardening (iPad field report: an out-of-bounds zoom
  // "when switching out or swipe"). Drop ALL in-flight gesture + momentum state
  // the instant we go hidden, so a resume can't replay a stale delta as one
  // oversized step: a pinch computed against pre-hide pointer positions, a flick
  // glide, a hold-to-fly cruise, or an eased zoom-chase carrying a stale goal.
  // (The per-step dt clamps in pump and the dist wall clamp are the deeper
  // backstops; this keeps the gesture layer itself from ever seeing the gap.)
  // ownerDocument, not a bare `document`, so core stays DOM-import-clean.
  canvas.ownerDocument?.addEventListener("visibilitychange", () => {
    if (!canvas.ownerDocument.hidden) return;
    glide = null;
    cruise = null;
    zoomChase = null;
    dragMode = null;
    wheelProbe = null;
    dragVel.reset();
    ptrs.clear();
  });

  // #551 — the SHAPE a camera push is judged against. Op VALUES are
  // deliberately absent: a param-morphing flight moves them every frame and
  // that is a continuous change (the reference orbit keys on values itself, in
  // ptOpsSig, and rebuilds when they move). What must force a retarget is a
  // different shape — another preset, an op added/removed/muted, a scene or
  // hybrid — because that is when "auto-detail from its base" means a
  // different base.
  function deStructSig(f) {
    if (!f) return "";
    return JSON.stringify([
      !!f.objects,
      !!f.hybrid,
      !!f.addC,
      !!f.julia,
      f.deOption ?? 0,
      (f.ops ?? []).map((o) => [o.key, o.enabled === false ? 0 : 1]),
    ]);
  }
  let camPushSig = null; // the shape the LAST push was judged against

  function frameTo(c) {
    if (!c) return;
    zoomChase = null; // a camera load snaps — abandon any eased zoom (#464)
    // #551 — classify BEFORE touching cam: the question is about the step FROM
    // where we are TO where `c` asks us to be. A continuation of the same shape
    // and the same view keeps the descent state; anything else is a retarget
    // and resets it exactly as it always did.
    const sig = deStructSig(formula);
    const cont =
      camPushSig === sig &&
      isContinuousPush(
        { dist: cam.dist, fovDeg: cam.fov / D2R, target: cam.target },
        c,
      );
    camPushSig = sig;
    // #562 — consume the arm on THIS call regardless of how it classifies: it
    // was set by the setFormula that immediately precedes every load's first
    // frameTo, so this is that call whether it lands as a retarget (the usual
    // case — a new shape) or, rarely, as a continuation (isContinuousPush can
    // still say yes for two same-shaped formulas). One-shot by construction —
    // leaving it armed past this call would risk it wrongly surviving into a
    // LATER, unrelated retarget.
    const clampSurvives = entryClampArm.consumeSurvives();
    // §6 — fresh camera/formula → auto-detail from its base. EXCEPT: #562 — if
    // setFormula just armed the #476 entry clamp for THIS load, this retarget
    // IS that load's own frameTo, and nulling here is what silently killed the
    // clamp before a single frame rendered. Preserve it exactly this once; a
    // later, unrelated retarget finds the arm already consumed and resets
    // normally, same as always.
    if (!cont && !clampSurvives) detailOverride = null;
    cam.yaw = (c.yawDeg ?? 35) * D2R;
    cam.pitch = (c.pitchDeg ?? 22) * D2R;
    cam.dist = c.dist ?? 24;
    cam.fov = (c.fovDeg ?? 42) * D2R;
    const nextT = Array.isArray(c.target) ? c.target.slice(0, 3) : [0, 0, 0];
    if (cont) {
      // A continuation moves the pivot through nudgeTarget — the sanctioned
      // residual-space path (D6), f64-exact at its own scale — so the exact
      // target survives the step instead of being rebased, and ptTfxGen bumps
      // ONLY when the pivot actually moved. A step that does not move the pivot
      // at all (a dolly, an orbit, a fov change — the shape of every deep-zoom
      // flythrough) therefore leaves the reference orbit valid, and the frame
      // reuses it. That reuse is the whole of #551.
      const d = [
        nextT[0] - cam.target[0],
        nextT[1] - cam.target[1],
        nextT[2] - cam.target[2],
      ];
      if (d[0] || d[1] || d[2]) nudgeTarget(d);
      return;
    }
    // Pan target (§5) — absent on every existing saved camera, so defaults to
    // the origin (today's implicit behavior, unchanged for all current presets).
    cam.target = nextT;
    // TAG.VIEW v2 (PR-4/D6): sub-f64 target words from a deep share — seed
    // the exact target so the descent resumes from the link's true point.
    if (Array.isArray(c.targetLo)) {
      ptTfx = cam.target.map(
        (hi, i) =>
          fxFromF64(hi) +
          fxFromF64(c.targetLo[i] ?? 0) +
          fxFromF64(c.targetLo2?.[i] ?? 0),
      );
      ptMirror = cam.target.slice();
      ptTfxGen++;
    } else {
      ptTfx = null; // v1/preset camera: rebase lazily from cam.target
      ptMirror = null;
    }
    ptOrbitKey = ""; // any restored camera invalidates the reference orbit
  }
  const camObj = () => {
    const o = {
      yawDeg: cam.yaw / D2R,
      pitchDeg: cam.pitch / D2R,
      dist: cam.dist,
      fovDeg: cam.fov / D2R,
      target: cam.target.slice(),
    };
    // D6: surface the sub-f64 words when the exact target holds more than
    // the f64 mirror (a re-pin below ~×10¹⁶) — TAG.VIEW v2 carries them.
    if (ptTfx) {
      const lo = [0, 1, 2].map((i) =>
        fxToF64(ptTfx[i] - fxFromF64(cam.target[i])),
      );
      if (lo.some((v) => v !== 0)) {
        o.targetLo = lo;
        o.targetLo2 = [0, 1, 2].map((i) =>
          fxToF64(ptTfx[i] - fxFromF64(cam.target[i]) - fxFromF64(lo[i])),
        );
      }
    }
    return o;
  };

  // Splice opts.metadata into a PNG blob (see pngmeta.js) if present, else
  // return it untouched. Shared by stillBlob's normal + alpha (#509) paths so
  // the embed step stays byte-identical between them.
  async function embedMetaIfAny(blob, metadata) {
    if (!metadata || !metadata.length) return blob;
    const bytes = embedChunks(
      new Uint8Array(await blob.arrayBuffer()),
      metadata,
    );
    return new Blob([bytes], { type: "image/png" });
  }

  // #509 (follow-up to #428/#482): the plain single-frame Save's transparent-
  // background path. stillBlob's normal path reads back the PRESENTED WebGPU
  // canvas (canvas.toBlob), but that canvas is configured alphaMode:"opaque"
  // (createRenderer) — the swap-chain forces full opacity on present no matter
  // what the shader writes, so real alpha can never survive that readback.
  // renderToImage is the SAME offscreen mechanism #428/#482 already use for the
  // flythrough's transparent PNG frames (an owned texture, copied back
  // directly — never touches the swap-chain), so this reuses it wholesale
  // instead of teaching the banded/presented-canvas pipeline a second exit.
  // One consequence: no watchdog banding here (#460's fix is presented-canvas-
  // specific) — same tradeoff the flythrough's per-frame alpha export already
  // ships with, not a new risk this feature introduces.
  async function stillBlobAlpha(W, H, opts) {
    // bakeDOF mirrors the normal path's DOF gate (see the comment above
    // STILL_SAMPLES below): a heavy formula that never converges DOF on
    // screen must not bake lens-jittered blur into its transparent still.
    const bakeDOF = accumCap() > 0;
    const STILL_SAMPLES = exportSampleCount({
      heavy: !bakeDOF,
      mode: opts.aaMode || "adaptive",
    });
    writeFrame(
      formula,
      stillQualityParams(formula, (cam && cam.dist) || REF_DIST),
      [W, H],
    );
    // Single offscreen render, no banding — see the tradeoff note above.
    opts.onStart && opts.onStart({ w: W, h: H, bands: 1 });
    const imgData = await renderer.renderToImage(
      W,
      H,
      STILL_SAMPLES,
      true,
      bakeDOF,
    );
    opts.onProgress && opts.onProgress(1);
    const cv = document.createElement("canvas");
    cv.width = W;
    cv.height = H;
    cv.getContext("2d").putImageData(imgData, 0, 0);
    const blob = await new Promise((res) => cv.toBlob(res, "image/png"));
    if (!blob) throw new Error("canvas capture returned null");
    return embedMetaIfAny(blob, opts.metadata);
  }

  // The banded, fenced export accumulation loop — ONE copy, shared by stillBlob
  // and the dev-only tile probe below. Assumes the canvas is already sized and
  // writeFrame() has run; leaves the pixels in the canvas for the caller to read
  // back. `bands` comes from settleBands() AFTER the resize, so it keys off the
  // export pixel count.
  //
  // Extracted verbatim (#212/#460 behaviour preserved to the letter): a rAF
  // between every band so the compositor keeps a slot, one fence per submit, and
  // the deliberate discarding of the fence result — on a watchdog kill the
  // export keeps going and captures whatever rendered, unlike the live settle
  // loop which aborts. (TILED_EXPORT §2.6 makes surfacing that a PR-2
  // requirement; PR-1 changes nothing about it.)
  // `target` (PR-2): resolve into a caller-owned offscreen texture instead of
  // the swap chain — the tiled export's readback (renderer.createTileTarget).
  // `onFence(ok)` (PR-2): the fence booleans this loop otherwise DISCARDS. At
  // one tile a watchdog kill degrades the whole image and the user sees it; at
  // 80 tiles it degrades one rectangle of a print plate, which is both likelier
  // and far easier to miss (TILED_EXPORT §2.6). The boolean already exists;
  // handing it to the caller costs nothing and the default stays "ignore".
  // `signal`: polled where the loop already yields, so a cancel lands within
  // one band rather than one export.
  async function accumulateStill({
    bakeDOF,
    samples,
    bands,
    onUnit,
    target,
    onFence,
    signal,
  }) {
    const resolveAccum = (o) =>
      renderer.drawAccum(target ? { ...o, target } : o);
    const fence = async (where) => {
      const ok = await fencedSettle(where);
      onFence && onFence(ok, where);
      return ok;
    };
    if (renderer.drawAccum) {
      for (let s = 0; s < samples; s++) {
        const first = s === 0;
        const [jx, jy] = first ? [0, 0] : r2jitter(s);
        const [lx, ly] = first || !bakeDOF ? [0, 0] : lensSample(s);
        renderer.writeJitter(jx, jy, first ? 1 : 1 / (s + 1), lx, ly);
        if (bands > 1) {
          for (let i = 0; i < bands; i++) {
            // Yield a rAF between every band (across samples too): the fence
            // alone runs in a microtask and never hands the compositor a slot.
            if (i || s) await new Promise((r) => requestAnimationFrame(r));
            throwIfAborted(signal);
            renderer.drawMarchBand(i, bands);
            await fence("export-band");
            onUnit && onUnit();
          }
          resolveAccum({ skipMarch: true });
          await fence("export-composite");
        } else {
          resolveAccum();
          onUnit && onUnit();
        }
      }
      renderer.writeJitter(0, 0, 0, 0, 0);
      // bands>1 fenced every submit already. bands==1 submitted all 24
      // dispatches unfenced (the original light-save fast path) → one settle
      // before the pixels are readable. Watchdog-safe (#460): a hung fence
      // would never reach the finally that restores canvas size / releases
      // `busy`; on timeout fall through and capture what rendered.
      if (bands === 1) await fence("export");
    } else {
      renderer.draw();
      await fence("export");
    }
  }

  // Render the current view to a high-res PNG Blob (engine side of PNG export;
  // turning it into a download is the app's job). Returns null when there's
  // nothing to render (no GPU / no formula); a failed canvas capture throws
  // (the caller restores nothing — the finally here does). opts.metadata: PNG
  // text chunks (see pngmeta.js) spliced in so the saved image carries the
  // formula/share-URL and can re-open (docs/design/PNG_METADATA.md).
  async function stillBlob(opts = {}) {
    if (!hasGPU || !formula) return null;
    const rect = canvas.getBoundingClientRect();
    // Dimensions come from renderpolicy.resolveStillDims (pure, node-tested):
    //   • opts.{width,height} (the EXPORT size picker) — an explicit override,
    //     BOTH required together, each clamped INDEPENDENTLY to 4096 (so a
    //     4096×4096 square is legal), giving deterministic output sizes that
    //     don't inherit the viewport's shape.
    //   • no override — the legacy path: render at the SAME device resolution
    //     the settled screen uses (dpr, capped — see sizeCanvas/qualityParams),
    //     not a fixed 900px, so the saved PNG stays as crisp as the on-screen
    //     Retina/2x canvas ("saved == what you see"). The 4096 LONG-edge clamp
    //     bounds render time / PNG size on huge/5K displays, where dpr*rect
    //     would otherwise demand a slow 24-sample accumulate at an enormous
    //     pixel count for no visible benefit. This branch is byte-identical to
    //     before the picker existed.
    const dpr = Math.min(window.devicePixelRatio || 1, DPR_CAP);
    const { W, H } = resolveStillDims({
      width: opts.width,
      height: opts.height,
      rectW: rect.width,
      rectH: rect.height,
      dpr,
    });
    // #509: transparent-background export never touches the presented canvas
    // (renderToImage owns its own offscreen texture at W×H) — an entirely
    // separate path, so the normal branch below stays byte-for-byte unchanged
    // when opts.alpha is falsy or the tier can't support it (WebGL2 exposes no
    // renderToImage-with-real-alpha; see stillBlobAlpha's header comment).
    if (opts.alpha && renderer.renderToImage) {
      busy = true;
      try {
        return await stillBlobAlpha(W, H, opts);
      } finally {
        busy = false;
        scheduleDraw();
      }
    }
    const prevW = canvas.width,
      prevH = canvas.height;
    busy = true;
    try {
      canvas.width = W;
      canvas.height = H;
      // Derive the WHOLE march budget from the SETTLED policy (renderpolicy
      // stillQualityParams — steps, eps AND deScale), so the saved PNG marches
      // at the live settled view's EXACT quality ("saved == what you see").
      // The old fixed 220 steps starved loose-DE/approx/unbounded formulas
      // (darker/grainier report); a hand-picked tighter eps (0.0006 vs the
      // settled 0.001) then eroded the silhouette of the deApprox bounded TPMS
      // leaves — see stillQualityParams for the full #281/#282/#283 write-up.
      writeFrame(
        formula,
        stillQualityParams(formula, (cam && cam.dist) || REF_DIST),
        [W, H],
      );
      // P2: PNG stills accumulate 24 jittered samples (offline-grade AA) when
      // the renderer supports it; the GL tier keeps the single-sample draw.
      //
      // DOF (the per-sample lens offsets) is GATED on whether the on-screen
      // SETTLED view actually shows depth-of-field, so "saved PNG == what you
      // see on screen". The live view only converges DOF through the pump's
      // accumulation loop (each accum sample carries lensSample(accumN)), and
      // that loop runs at most accumCap() samples — which is 0 for a heavy
      // formula (lastFullMs ≥ 100 ms; a loose-DE deep-zoom frame is seconds).
      // Such a frame therefore settles on screen to the lens-CENTERED base
      // frame (writeJitter(0,0,1,0,0) in the banded/base path) — crisp, no
      // bokeh. The still USED to bake all 24 lens-jittered samples regardless,
      // so its foreground blurred into DOF blobs the screen never showed (the
      // "saved PNG is blurry" report — NOT the earlier resolution bug). So:
      //   • accumCap() > 0  (light enough that the live view converges DOF)
      //       → keep the lens offsets; the still shows the same bokeh, better
      //         converged. Matches a light formula's on-screen DOF.
      //   • accumCap() == 0 (heavy → live view never accumulates DOF)
      //       → render lens-CENTERED (lens offsets dropped); the 24 samples are
      //         pure sub-pixel AA (r2jitter kept) so the still stays crisp AND
      //         high-AA, matching the on-screen settled frame.
      // accumCap() is read from the last on-screen settled measurement, so it is
      // exactly the live view's own DOF-accumulate decision, not a re-derivation.
      const bakeDOF = accumCap() > 0;
      // Export AA sample count (#save-latency) — renderpolicy owns the decision;
      // `heavy` is the SAME accumCap()==0 signal `bakeDOF` reads, so an adaptive
      // heavy save takes fewer samples (~3× faster) AND those samples are already
      // lens-centered (bakeDOF false) — pure AA, no shape/DOF change. The mode
      // comes in via opts.aaMode (default 'adaptive'), so core stays pref-agnostic.
      const STILL_SAMPLES = exportSampleCount({
        heavy: !bakeDOF,
        mode: opts.aaMode || "adaptive",
      });
      // Banded export (#212 pattern, extended to the still). The 24-sample
      // accumulate is 24 FULL-frame marches; at export resolution each is a
      // multi-second single GPU dispatch, and 24 of them back-to-back with one
      // trailing fence starved Chrome's compositor for the WHOLE save (the
      // "frozen export" report) AND, on a heavy formula, blew the single export
      // fence past its watchdog timeout (~4×predictFullMs) so the save fell
      // through to toBlob mid-render and captured a PARTIAL accumulate (#460).
      // Reuse the live settled path's banding: split EACH sample's march into
      // settleBands() scissored strips (drawMarchBand), one fenced submit per
      // band with a rAF between, then resolve that sample with
      // drawAccum({skipMarch:true}). This is PIXEL-IDENTICAL to the unbanded
      // drawAccum() per sample — the bands are scissored regions of the SAME
      // march pass into the SAME HDR, the skipMarch resolve runs the identical
      // accum/bloom/post, and the accum ping-pong flips exactly once per sample
      // as before — but no single dispatch exceeds one band, so the UI stays
      // alive and each small fence clears the watchdog comfortably. canvas is
      // already W×H, so settleBands()/predictFullMs key off the EXPORT pixel
      // count: light/small saves get bands==1 (the original fast path, kept
      // byte-for-byte) and only heavy/large saves band.
      const exportBands = renderer.drawMarchBand ? settleBands() : 1;
      opts.onStart && opts.onStart({ w: W, h: H, bands: exportBands });
      const totalUnits = STILL_SAMPLES * exportBands;
      let unitsDone = 0;
      await accumulateStill({
        bakeDOF,
        samples: STILL_SAMPLES,
        bands: exportBands,
        onUnit: () => {
          unitsDone++;
          opts.onProgress && opts.onProgress(unitsDone / totalUnits);
        },
      });
      opts.onProgress && opts.onProgress(1);
      let blob = await new Promise((res) => canvas.toBlob(res, "image/png"));
      if (!blob) throw new Error("canvas capture returned null");
      if (opts.metadata && opts.metadata.length) {
        const bytes = embedChunks(
          new Uint8Array(await blob.arrayBuffer()),
          opts.metadata,
        );
        blob = new Blob([bytes], { type: "image/png" });
      }
      return blob;
    } finally {
      canvas.width = prevW;
      canvas.height = prevH;
      busy = false;
      scheduleDraw();
    }
  }

  // ── TILED HIGH-RES EXPORT (TILED_EXPORT.md §2.3–§2.5, PR-2) ───────────────
  //
  // PNG stills above the 4096 cap — 8K, A0 @300dpi, 24×36in, 16K — by rendering
  // an N×M grid of ≤4096 off-axis sub-frames (PR-1's camera window makes each
  // one reproduce the parent frame's rays EXACTLY) and streaming them into one
  // PNG as they land.
  //
  // THE RULE THIS PATH IS BUILT AROUND: never allocate anything larger than one
  // tile. No full-size canvas, no full-size ImageBitmap, no whole-image RGBA
  // buffer, no second PNG encode, no whole-file re-read for metadata. Peak
  // resident is one row band (≤64 MiB) plus one tile readback — FLAT in the
  // output size. That is what makes every per-engine canvas-area ceiling (§1.9:
  // Firefox ~125 Mpx, iOS Safari 16.8 Mpx) structurally irrelevant instead of a
  // compatibility matrix to probe.
  //
  // Everything that is not a function of the individual ray is decided ONCE,
  // before the loop, and passed down unchanged (§2.2) — above all bakeDOF and
  // STILL_SAMPLES, which read a TIMING measurement and would otherwise let a
  // heavy formula flip between 8 and 24 samples mid-image and put visibly
  // different grain in different tiles.
  //
  // opts:
  //   width, height   requested size; rounded UP to even (§2.3.3) and capped at
  //                   STILL_PX_CAP_TILED. STILL_PX_CAP (4096) is NOT touched —
  //                   the untiled path keeps its own ceiling.
  //   alpha           real transparent background. Free here, unlike the ≤4096
  //                   path: we render into an owned texture, never the
  //                   alphaMode:"opaque" swap chain, so alpha survives readback
  //                   AND keeps the banding/watchdog protection stillBlobAlpha
  //                   has to give up (#509).
  //   aaMode          'adaptive' | 'full' — as stillBlob.
  //   metadata        PNG text chunks, written INLINE after IHDR.
  //   sink            { write, close?, abort? } — the app picks (FS Access →
  //                   OPFS → in-memory); defaults to in-memory.
  //   signal          AbortSignal. Polled between bands and between tiles.
  //   onStart/onProgress/onTile/onRowBand — progress + the watermark hook.
  //
  // Returns { blob, W, H, tiles, degraded, grid }. `blob` is null when the sink
  // wrote straight to disk; `degraded` lists the tiles whose fence hit the
  // watchdog (§2.6) so the app can badge the result instead of shipping a
  // quietly defective plate.
  async function stillBlobTiled(opts = {}) {
    if (!hasGPU || !formula) return null;
    // Hard refusals, all BEFORE anything is allocated or rendered (§2.6). Each
    // is also enforced in the UI, which disables the row with the reason — this
    // is the backstop, not the message the user is meant to hit.
    if (!renderer.createTileTarget)
      throw new Error("Tiled export needs the WebGPU renderer");
    if (coarseMobile) throw new Error("Tiled export needs a desktop GPU");
    if (!pngStreamSupported())
      throw new Error(
        "This browser cannot stream PNG (needs CompressionStream)",
      );

    const { W, H } = evenDims(
      Math.min(STILL_PX_CAP_TILED, Math.max(2, opts.width | 0)),
      Math.min(STILL_PX_CAP_TILED, Math.max(2, opts.height | 0)),
    );
    // pad is TILE_PAD unconditionally. §2.2.1(b) allows pad = 0 when bloom is
    // off, but `bloomOn` is renderer-internal state derived from the coloring,
    // and getting that wrong produces a faint seam only visible on bright edges
    // — the exact failure the apron exists to prevent. ~3–4% extra marched
    // pixels is the right price for an unconditional guarantee.
    const grid = tileGrid(W, H, { pad: TILE_PAD });
    const alpha = !!opts.alpha;
    const sink = opts.sink || memorySink();

    const prevW = canvas.width,
      prevH = canvas.height;
    let target = null;
    let stream = null;
    busy = true;
    try {
      throwIfAborted(opts.signal);
      // ONE canvas resize for the whole export (§2.3.2). The bundle cache is
      // keyed "w×h" with two slots, so a per-tile size would evict and
      // reallocate ~44 B/px of GPU textures on every tile; `busy` above keeps
      // the pump from drawing at the on-screen size and evicting it mid-export.
      // Resizing the canvas (rather than passing dimensions around) is also
      // what makes settleBands()/predictFullMs key off the TILE's pixel count,
      // exactly as they key off the export's today.
      canvas.width = grid.tw;
      canvas.height = grid.th;
      writeFrame(
        formula,
        stillQualityParams(formula, (cam && cam.dist) || REF_DIST),
        [grid.tw, grid.th],
      );
      // Frozen for the whole export (§2.2 / decision 6).
      const bakeDOF = accumCap() > 0;
      const STILL_SAMPLES = exportSampleCount({
        heavy: !bakeDOF,
        mode: opts.aaMode || "adaptive",
      });
      const bands = renderer.drawMarchBand ? settleBands() : 1;
      const total = grid.tiles.length;

      target = renderer.createTileTarget(grid.tw, grid.th);
      stream = await createPngStream({
        W,
        H,
        alpha,
        text: opts.metadata || [],
        sink,
      });

      opts.onStart &&
        opts.onStart({
          w: W,
          h: H,
          bands,
          tiles: total,
          cols: grid.cols,
          rows: grid.rows,
        });

      // One row-band buffer for the WHOLE export, sized to the tallest committed
      // slice; shorter rows use a subarray. Re-allocating per row would churn
      // tens of MB nine to sixteen times over.
      const maxBandH = Math.max(...grid.tiles.map((t) => t.y1 - t.y0));
      const bandBuf = new Uint8Array(W * maxBandH * 4);

      const totalUnits = total * STILL_SAMPLES * bands;
      let unitsDone = 0;
      let done = 0;
      const degraded = [];

      for (let j = 0; j < grid.rows; j++) {
        throwIfAborted(opts.signal);
        const rowTiles = grid.tiles.filter((t) => t.j === j);
        const y0 = rowTiles[0].y0,
          bandH = rowTiles[0].y1 - y0;
        const band = bandBuf.subarray(0, W * bandH * 4);
        band.fill(0);

        for (const t of rowTiles) {
          throwIfAborted(opts.signal);
          // The off-axis window + the tile's absolute rect. tilepx is what keeps
          // the background gradient, the vignette and the dither addressed in
          // PARENT-frame space — without it each tile grows its own vignette and
          // restarts the sky gradient, which is the most visible seam of all.
          renderer.writeTile(tileWindow(t.rx0, t.ry0, grid.tw, grid.th, W, H), [
            t.rx0,
            t.ry0,
            W,
            H,
          ]);
          let ok = true;
          await accumulateStill({
            bakeDOF,
            samples: STILL_SAMPLES,
            bands,
            signal: opts.signal,
            target,
            onFence: (good) => {
              ok = ok && good;
            },
            onUnit: () => {
              unitsDone++;
              opts.onProgress && opts.onProgress(unitsDone / totalUnits);
            },
          });
          if (!ok) degraded.push({ i: t.i, j: t.j });

          // Commit ONLY the committed sub-rect — the apron is rendered and
          // thrown away, which is what makes the crop exact rather than a blend.
          const cw = t.x1 - t.x0,
            ch = t.y1 - t.y0;
          const { data, blank } = await target.read(
            t.x0 - t.rx0,
            t.y0 - t.ry0,
            cw,
            ch,
            alpha,
          );
          // The PR-1 failure signature, kept as a named error. On the OPAQUE
          // path every pixel of a real readback carries alpha 255, so an
          // all-zero buffer can only mean the copy landed nothing — a silently
          // missing rectangle in a print plate is the worst possible outcome.
          // Not applied on the alpha path, where a fully transparent black
          // region is a legitimate render.
          if (blank && !alpha)
            throw new Error(
              `tiled export: tile (${t.i},${t.j}) read back blank`,
            );
          for (let r = 0; r < ch; r++)
            band.set(
              data.subarray(r * cw * 4, (r + 1) * cw * 4),
              ((t.y0 - y0 + r) * W + t.x0) * 4,
            );
          done++;
          opts.onTile &&
            opts.onTile({
              index: done,
              total,
              i: t.i,
              j: t.j,
              x0: t.x0,
              y0: t.y0,
            });
        }

        // The watermark hook (§2.5): composited into the row bands it overlaps,
        // ONCE — not onto a stitched image (there is none) and not per tile
        // (which would repeat it). Core owns no branding, so the app supplies it.
        if (opts.onRowBand) await opts.onRowBand(band, y0, bandH, W, H);
        await stream.writeRows(band, bandH);
      }

      const blob = await stream.finish();
      stream = null;
      return { blob, W, H, tiles: total, degraded, grid };
    } catch (e) {
      // A partial PNG on the user's disk is worse than no PNG: abort unlinks it
      // (FS Access truncates on close, OPFS removes the temp file).
      if (stream) await stream.abort();
      throw e;
    } finally {
      target && target.destroy();
      // Restore the window BEFORE anything can draw again. writeFrame would do
      // it too (every full write restores the identity), but leaving a live
      // renderer holding a tile window is the kind of state a later bug reads.
      renderer.writeTile &&
        renderer.writeTile(TILE_WINDOW_IDENTITY, [0, 0, 0, 0]);
      canvas.width = prevW;
      canvas.height = prevH;
      busy = false;
      scheduleDraw();
    }
  }

  // What the popover needs to answer "how long will this take?" BEFORE the user
  // commits (TILED_EXPORT §7 Q4 — decided yes, with a confirm above ~30 s).
  // Pure read of the same frozen decisions stillBlobTiled will make, so the
  // estimate and the export cannot disagree about tile count or sample count.
  //
  // `measured` is the honest part: predictFullMs falls back to a flat boot guess
  // when this formula has never had a settled frame timed, and an estimate built
  // on that is a guess about a guess. The UI must say so rather than quote a
  // fabricated minute count — a wrong estimate on a heavy formula is worse than
  // none, which is exactly why the spec left this as an open question.
  function tiledExportPlan({ width, height, aaMode } = {}) {
    const { W, H } = evenDims(
      Math.min(STILL_PX_CAP_TILED, Math.max(2, width | 0)),
      Math.min(STILL_PX_CAP_TILED, Math.max(2, height | 0)),
    );
    const grid = tileGrid(W, H, { pad: TILE_PAD });
    const heavy = !(accumCap() > 0);
    const samples = exportSampleCount({ heavy, mode: aaMode || "adaptive" });
    // Band count is decided AFTER the canvas is resized to the tile, so predict
    // it the same way settleBands() will: from the tile's pixel count.
    const tileMs = predictMsForPx(grid.tw * grid.th);
    const bands = renderer.drawMarchBand
      ? Math.max(
          1,
          Math.min(SETTLE_BANDS_MAX, Math.ceil(tileMs / SETTLE_BAND_MS)),
        )
      : 1;
    // Per tile: `samples` full marches of the tile, plus one rAF yield per band
    // per sample (the loop hands the compositor a slot between every band — real
    // wall-clock time, and at 64 bands × 24 samples it is not a rounding error).
    const RAF_MS = 16.7;
    const renderMs = grid.tiles.length * samples * (tileMs + bands * RAF_MS);
    // Filter + deflate, measured at ~25–40 Mpx/s on this machine's Node build.
    const encodeMs = ((W * H) / 25e6) * 1000;
    return {
      W,
      H,
      cols: grid.cols,
      rows: grid.rows,
      tiles: grid.tiles.length,
      tw: grid.tw,
      th: grid.th,
      samples,
      bands,
      ms: renderMs + encodeMs,
      rowBandBytes: rowBandBytes(grid),
      measured: haveMeasurement(),
    };
  }

  // Video export (docs/design/VIDEO_EXPORT.md §7.1) — render ONE frame of the
  // current formula/coloring/camera at a chosen resolution and resolve when the
  // GPU is done, returning an ImageBitmap the caller can feed to a VideoEncoder.
  // Generalizes exportPNG; used by the offline HQ flythrough render. The caller
  // wraps setOffline(true)/(false) around its loop so the live pump stays out of
  // the way (this method draws directly, not via the pump). Restores canvas size
  // each call; does NOT scheduleDraw (the loop owns redraws — see setOffline).
  async function captureFrame(opts = {}) {
    if (!hasGPU || !formula) return null;
    // #345: opts.cam frames a SPECIFIC view offline (e.g. the splat-export's own
    // object-framing) WITHOUT touching the live camera — writeFrame's 4th arg fully
    // overrides the camera. A plain {yawDeg,pitchDeg,dist,fovDeg,target} literal
    // becomes a camera instance here.
    const useCam = opts.cam ? makeCamera(opts.cam) : null;
    const rect = canvas.getBoundingClientRect();
    const H = opts.h || 900;
    const W =
      opts.w || Math.round(H * ((rect.width || 4) / (rect.height || 3)));
    // Derive the march budget from the SETTLED policy (shared with stillBlob —
    // renderpolicy stillQualityParams): the old fixed 220 steps bypassed every
    // policy boost (loose-DE/approx/unbounded scenes exported starved budgets
    // while the live settled view had 320-512 — the "GIF render is still poor"
    // report), and a hand-picked tighter eps eroded the deApprox bounded TPMS
    // leaves' silhouette (#281/#282/#283). steps, eps AND deScale all track the
    // settled view so the exported frame matches what's on screen.
    const quality =
      opts.quality ||
      stillQualityParams(
        formula,
        ((useCam || cam) && (useCam || cam).dist) || REF_DIST,
      );
    // P2: opts.samples > 1 → renderToImage runs N jittered march+accumulate
    // rounds (movie-grade AA, offline so frame time is free).
    const samples = Math.max(1, opts.samples | 0);
    writeFrame(formula, quality, [W, H], useCam);
    // WebGPU: render into an OFFSCREEN texture and copy the pixels straight back
    // (renderToImage — the deterministic path the thumbnail gallery uses). Reading
    // back the PRESENTED canvas instead suffers swap-chain double-buffering: the
    // read alternates between buffers, some still holding the uninitialised clear
    // colour → the "green frame" bug in offline export. This never touches the
    // visible canvas, so it also can't flicker the live view.
    if (renderer.renderToImage) {
      // opts.alpha (issue #428): real transparent-background PNG export,
      // WebGPU-tier only — the WebGL2 fallback below always reads back an
      // opaque canvas.
      const img = await renderer.renderToImage(W, H, samples, !!opts.alpha);
      return img ? await createImageBitmap(img) : null;
    }
    // WebGL2 fallback: canvas readback is reliable here (preserveDrawingBuffer),
    // so draw at the target size and grab it; restore live size for one-shot use.
    const prevW = canvas.width,
      prevH = canvas.height;
    busy = true;
    try {
      if (canvas.width !== W || canvas.height !== H) {
        canvas.width = W;
        canvas.height = H;
        writeFrame(formula, quality, [W, H], useCam);
      }
      renderer.draw();
      return await createImageBitmap(canvas);
    } finally {
      busy = false;
      if (!offline) {
        canvas.width = prevW;
        canvas.height = prevH;
        scheduleDraw();
      }
    }
  }

  // Suspend/resume the live rAF pump for an offline export. While offline, the
  // app can setFormula/setColoring/frameTo per frame without triggering live
  // draws; captureFrame renders each frame explicitly. Resuming kicks one draw.
  function setOffline(b) {
    offline = !!b;
    if (!offline) scheduleDraw();
  }

  // Thumbnail dataURL cache, keyed by what actually drives the render (ops +
  // camera + the escape/IFS knobs, AND the scene/hybrid structure — without
  // objects/hybrid a CSG scene and a hybrid formula sharing the same base ops
  // collided in the cache and one showed the other's picture). Lets the SAME
  // formula be tiled in more than one place — e.g. a filtered strip AND an
  // expand-to-grid sheet, or the same preset across category switches — while
  // paying the GPU cost only once.
  //
  // Bounded LRU: the live camera is part of the key (movepicker stamps it so
  // tiles match the stage framing), so every picker open at a fresh orbit angle
  // minted ~41 permanent dataURLs — tens of MB over a long session with
  // auto-spin. Cap comfortably above one gallery + one picker's worth; on a
  // hit, re-insert to refresh recency; on overflow, evict the oldest key.
  const THUMB_CACHE_MAX = 256;
  const thumbCache = new Map();
  const thumbCacheGet = (key) => {
    const hit = thumbCache.get(key);
    if (hit !== undefined) {
      thumbCache.delete(key);
      thumbCache.set(key, hit); // refresh recency
    }
    return hit;
  };
  const thumbCacheSet = (key, url) => {
    thumbCache.delete(key);
    thumbCache.set(key, url);
    while (thumbCache.size > THUMB_CACHE_MAX)
      thumbCache.delete(thumbCache.keys().next().value);
  };
  const thumbKey = (p, look) =>
    JSON.stringify([
      p.ops,
      p.camera,
      p.iters,
      !!p.addC,
      !!p.julia,
      p.juliaC || null,
      p.deOption ?? null,
      p.objects || null,
      p.hybrid || null,
      // Per-tile coloring override (curated preset looks) when given; else
      // the LIVE coloring — a look-less tile (the picker's contextual cards
      // and its "Your fractal now" reference) renders under the current
      // colors, so the fingerprint must carry them or a theme change serves
      // the previous colors from the cache on the next picker open.
      look || coloring,
    ]);
  // Offscreen pixel buffer for reading a rendered tile back to a data URL
  // (renderToImage → ImageData → toDataURL). Reused across tiles; resized to the
  // requested tile size. Not app markup — an engine render target (could be an
  // OffscreenCanvas), so it stays engine-side.
  let thumbCanvas = null,
    thumbCtx = null;
  function thumbScratch(w, h) {
    if (!thumbCanvas) {
      thumbCanvas = document.createElement("canvas");
      thumbCtx = thumbCanvas.getContext("2d");
    }
    if (thumbCanvas.width !== w) thumbCanvas.width = w;
    if (thumbCanvas.height !== h) thumbCanvas.height = h;
    return thumbCtx;
  }
  // Engine side of thumbnails (the caller builds the grid markup itself):
  //   thumbTileCached — cache lookup only, no GPU (instant tiles).
  //   renderThumbTile — render ONE preset to an offscreen texture and read it
  //     straight back (renderToImage) — deterministic, so a tile never shows a
  //     stale/other preset (the canvas-present race that made e.g. Slab Box show
  //     the Pseudo-Kleinian picture). Full MARCH quality (deScale 0.5) so thin/
  //     complex surfaces don't render holey, but CHEAP SHADING (no shadow/AO
  //     marches) — at tile size the depth cues are invisible and the P1 marches
  //     made gallery loads sluggish. Caches by fingerprint.
  //   beginThumbs/endThumbs — the caller brackets a batch of misses so the live
  //     pump stays paused across the whole grid (one busy window, one settle).
  const thumbTileCached = (p, look) => thumbCacheGet(thumbKey(p, look)) || null;
  async function renderThumbTile(p, W, H, look) {
    const key = thumbKey(p, look);
    const hit = thumbCacheGet(key);
    if (hit) return hit;
    // A tile whose preset needs a gated march variant (numeric-DE, a scene/leaf,
    // a color mode, …) must not trigger that variant's SYNC compile — that was
    // the "opening + Add a move freezes the browser" report. Await the async
    // build instead: this tile (and the serialized drain behind it) arrives once
    // the background compile lands.
    if (renderer.marchReadyFor && !renderer.marchReadyFor(frameFeatures(p))) {
      const ff = frameFeatures(p);
      // Thumbnails ride the GENERAL (compile-once, reused) variant, NOT a
      // per-preset specialized compile — otherwise browsing a category
      // recompiles a shader per tile (scenes ~40 s each on Pascal/D3D12: the
      // "creates thumbnails" freeze). One general flat + one general scene shader
      // now serve every tile of that kind (built-in tiles are baked and skip this
      // entirely; ★ Mine and any un-baked tile still land here).
      const warm =
        (renderer.prewarmGeneralFor && renderer.prewarmGeneralFor(ff)) ||
        renderer.prewarmMarchFor(ff);
      if (warm) await warm;
    }
    const s2d = thumbScratch(W, H);
    // Per-tile coloring override (curated preset looks): swap the coloring in
    // for this one render and restore it after. Palette cycling is forced off
    // first so the look's flat colors are what the tile shows — unless the
    // look itself carries a palette (a saved formula's colors). The restore is
    // guarded: if the app called setColoring while we awaited the readback,
    // keep ITS coloring, not our snapshot.
    const prev = coloring;
    const temp = look
      ? {
          ...coloring,
          palette: { ...(coloring.palette || {}), on: false },
          ...look,
        }
      : null;
    if (temp) coloring = temp;
    try {
      const tileCam = makeCamera(p.camera);
      // A LOOSE analytic DE (scale < 2 — stability.js looseDE/#14) over-
      // estimates distance: the default march (steps 200, deScale 0.5) flies
      // past the (often thin) surface and the tile comes back solid black —
      // the same failure the live full-quality tier already guards against
      // (renderpolicy.js qualityParams). Route through it here too instead of
      // hardcoding the non-loose numbers, so a loose preset gets the tighter
      // step/more steps it needs. quality:"full"/moveQuality:"full" selects
      // that tier unconditionally (a tile is never "interactive").
      const q = policyQualityParams(p, {
        quality: "full",
        moveQuality: "full",
        dist: tileCam.dist,
      });
      // iterCap 12: at tile size (~168 px) detail past ~12 iterations is
      // sub-pixel FOR A WELL-CONVERGED formula, but a loose-DE formula (scale
      // < 2) hasn't converged yet at 12 — the orbit's |w| is still small, so
      // r/|w| is a bad distance bound and the march above misses the surface
      // entirely (issue #228: Icokaletra_Jolie/Katica, scale 1.16, went solid
      // black at iterCap 12 even with the march fix above). Only a loose
      // formula pays the extra iterations; the common case keeps the cap.
      const iterCap = hybridLooseDE(p) ? Infinity : 12;
      writeFrame(
        p,
        {
          steps: q.steps,
          eps: q.eps,
          deScale: q.deScale,
          cheap: true,
          iterCap,
        },
        [W, H],
        tileCam,
      );
      s2d.putImageData(await renderer.renderToImage(W, H), 0, 0);
    } finally {
      if (temp && coloring === temp) coloring = prev;
    }
    const url = thumbCanvas.toDataURL("image/png");
    thumbCacheSet(key, url);
    return url;
  }
  function beginThumbs() {
    busy = true;
  }
  function endThumbs() {
    busy = false;
    scheduleDraw();
  }

  // Does `formula` still need a march-variant compile? (true ⇒ selecting or
  // precompiling it would hit the GPU). Lets the app's idle background compile
  // queue peek without kicking a compile, so it can paint a hint first.
  // `over` (#609): optional frameFeatures override — `{morphF}` keys a morph
  // pair (or, with an explicit null, the plain variant) independent of the
  // live setMorph state.
  function needsCompile(formula, over) {
    return !!(
      hasGPU &&
      formula &&
      renderer.marchReadyFor &&
      !renderer.marchReadyFor(frameFeatures(formula, over))
    );
  }
  // Prewarm the march variant `formula` will use — general for FLAT, specialized
  // for scene/hybrid/morph (prewarmGeneralFor returns null → falls through).
  // Returns the in-flight compile promise, or null if already warm / unsupported.
  // Powers the app's idle background compile queue: per-scene compiles spread
  // across the time the user spends looking around instead of hitting on every
  // gallery click. The user's actual selection still compiles on-demand (the pump
  // hold path) — this just warms ahead. Wander (#609) passes `over.morphF` to
  // warm the from→to morph variants BEFORE starting a melt, so the pump never
  // holds mid-ambience on "Compiling morph shader…".
  function prewarmFor(formula, over) {
    if (!needsCompile(formula, over)) return null;
    const ff = frameFeatures(formula, over);
    return (
      (renderer.prewarmGeneralFor && renderer.prewarmGeneralFor(ff)) ||
      renderer.prewarmMarchFor(ff)
    );
  }

  // §S2 GPU splat G-buffer capture (docs/planning/UE_SPLAT_S2_IMPL.md §3.2 +
  // SPLAT_STREAMING_REDUCE §6) — thin wrapper since EXPORT_P1 PR-B: settle,
  // then delegate the whole view loop to core/capturedrive.js (shared with the
  // headless CLI — its return contract, incl. the {noPipeline} sentinel and
  // bare-null cancel/zero-hit, is documented THERE). The APP wraps this in
  // setOffline(true)/(false), like the video exporter wraps captureFrame, so no
  // live frame interleaves and rewrites Globals mid-capture.
  async function captureSplatGBuffer(opts) {
    if (!hasGPU || !renderer.createSplatCapture) return null;
    const { formula, iters, res = 256 } = opts;
    // 1. Settle Globals/ops/objects with the still policy + the EXPLICIT iters
    //    (threaded via itersOverride, not re-derived — §3.2 iters box). This also
    //    latches the feature signature createSplatCapture compiles for. Camera
    //    words are written but unused by fsCapture (ortho rays come from CaptureU).
    //    Direct settleFrame with NO live bag — the capture is a still of the
    //    FORMULA, so the live morph/colorBlend/df64-latch state (and the
    //    lastFrameState tap) deliberately doesn't apply; this is byte-for-byte
    //    the settle the headless CLI runs (PR-B parity by construction).
    settleFrame(
      renderer,
      formula,
      coloring,
      {
        ...stillQualityParams(formula, (cam && cam.dist) || REF_DIST),
        itersOverride: iters,
      },
      [res, res],
      cam,
    );
    // 2–5. The drive: capture-offset override (+ its OWN capK df64 decision),
    //    pipeline compile, per-view render → streaming reduce.
    const driveSplatCapture = await getDriveSplatCapture();
    return driveSplatCapture(renderer, formula, coloring, opts);
  }

  // Kick off cpu.js now (getCpuDE's makeDE + capturesettle's signalRange) —
  // after the GPU adapter negotiation + shader compiles above, which take
  // longer, so this mostly just confirms it's already in. Await here, once,
  // so createPreview's caller never sees a "not loaded yet" window: behavior
  // is identical to the old static import, just off the boot chunk's
  // parse/eval (#266).
  await Promise.all([ensureCpuMod(), ensureCpu()]);

  return {
    // Live getter (not a boot-time snapshot): a mid-session WebGL2 death flips
    // this false so views.hasGPU() and the hasGPU-gated app paths (save, camera
    // carry-back) see the demotion, not a stale `true`.
    get hasGPU() {
      return hasGPU;
    },
    backend,
    getCapability: () => capability, // CAPABILITY_PROBE.md — static machine class

    needsCompile,
    prewarmFor,
    // GPU diagnostics (adapter, per-variant compile timings, device-lost /
    // validation events) — null on the WebGL2/ASCII tiers.
    getDiag: () => renderer?.getDiag?.() ?? { backend, adapter: null },
    // #291 standalone export — the globals payload from the last FLAT frame (null
    // until a flat frame has rendered). The exporter forces a fresh frame first.
    getLastFrameState: () => lastFrameState,
    // #441 — set the orthographic half-height (0 = perspective). Transient
    // inspection state: an orbit clears it, and it is never encoded in a share
    // link or carried across a formula switch.
    setOrtho(h) {
      const v = Number.isFinite(h) && h > 0 ? h : 0;
      if (v === orthoH) return;
      orthoH = v;
      scheduleDraw();
    },
    isOrtho: () => orthoH > 0,
    cam,
    isTouch,
    camObj,
    frameTo,
    bumpInteract,
    setFormula(f) {
      const isNew = f !== formula;
      formula = f;
      cpuDE = cpuDEFor = null; // invalidate the zoom-to-surface distance field
      // #562 — a fresh setFormula call always starts a clean slate for the arm:
      // a load that doesn't (re-)clamp must not let a PRIOR load's stale arm
      // survive into this one's frameTo.
      entryClampArm.disarm();
      // #476 cost-aware entry clamp: a heavy formula freshly loaded/imported/
      // preset-applied on a coarse/mobile device gets its STARTING detail
      // clamped so the FIRST (unmeasured) settle can't be fatal — the governor
      // above is reactive (needs 2-3 frames), this protects the very first paint.
      // Only lowers, single-object only (detailOverride is single-object); the
      // user can still drag Detail up, and any zoom hands detail to auto-detail.
      // #562 — every load path (preset/share/import/Surprise/Remix/Wander) calls
      // frameTo (directly or via a camera tween) immediately after this, and
      // that call classifies as a retarget (new shape) and used to unconditionally
      // null detailOverride right back out — so the clamp set here never survived
      // to a single rendered frame. Arming here tells frameTo's retarget branch
      // to preserve (not null) the value we just set, exactly once.
      if (isNew && coarseMobile && f && !f.objects) {
        const clamp = entryDetailClamp(f, { coarseMobile: true });
        if (clamp != null) {
          detailOverride = clamp;
          entryClampArm.arm();
          noteDiag("entry-clamp", { iters: clamp, base: f.iters ?? null });
        }
      }
      scheduleDraw();
    },
    // Formula-morph spike: blend the CURRENT formula's DE toward `f` by t
    // (0 = pure current … 1 = pure f). Pass null to switch it off. WebGPU
    // only, plain formulas only — writeFrame falls back to the normal path
    // otherwise. Both live drawing and captureFrame/renderToImage honor it
    // (they share writeFrame).
    // `swell` = PEAK mid-blend dilation in world units (default 0): the engine
    // shapes it as swell·4t(1−t) — zero at both endpoints, so t=0/t=1 stay
    // exact — to counteract the level-set erosion ("waist") where the two
    // fields don't overlap.
    setMorph(f, t = 0, swell = 0) {
      const tc = Math.max(0, Math.min(1, t));
      morph = f ? { f, t: tc, swell: swell * 4 * tc * (1 - tc) } : null;
      scheduleDraw();
    },
    // Coloring-mode crossfade: b = {t, modeB, palOnB} or null (off). Enums
    // (color mode, palette toggle) can't lerp — the shader shades under both
    // views' modes and mixes by t instead of snapping at the midpoint.
    setColorBlend(b) {
      colorBlend = b || null;
      scheduleDraw();
    },
    // Debug surface-quality overlay (#370) — dev diagnostic, WebGPU tier only.
    // n: 0 off | 1 march step-count heat | 2 overshoot/bracket | 3 ∇DE instability.
    // Remaps a per-pixel march metric to a heat ramp instead of shading, so
    // discontinuity / precision hot-spots are visible (spirulae's red cue). The
    // UI exposes this only when the GPU diagnostics panel is up (?diag / showDiag).
    setDebugView(n) {
      debugView = Math.max(0, Math.min(3, Math.round(Number(n) || 0)));
      scheduleDraw();
    },
    getDebugView: () => debugView,
    getFormula() {
      return formula;
    },
    setColoring(c) {
      coloring = c;
      scheduleDraw();
    },
    getColoring() {
      return coloring;
    },
    setAutoRotate(b) {
      autoRotate = b;
      // Spin stopped → settle back to a full-detail frame (the interactive tier
      // the spin rendered in leaves the canvas at reduced resolution otherwise).
      if (!b) quality = "full";
      scheduleDraw();
    },
    setSpinSpeed(degPerFrame) {
      spinSpeed = degPerFrame;
      if (autoRotate) scheduleDraw();
    },
    setSpinTilt(deg) {
      spinTilt = deg;
      if (autoRotate) scheduleDraw();
    },
    // Deep zoom P4 — df64 override: 'auto' (default, hysteresis-engaged) or
    // 'off' (pin f32 — the A/B lever the ladder harness and a future pref
    // use; PR-4 may surface it). Off also releases the engagement latch.
    setDf64(mode) {
      df64Mode = mode === "off" ? "off" : "auto";
      if (df64Mode === "off") df64Engaged = false;
      scheduleDraw();
    },
    // Perturbation tier A/B lever (PR-4) — same contract as setDf64.
    setPt(mode) {
      ptMode = mode === "off" ? "off" : "auto";
      if (ptMode === "off") ptEngaged = false;
      scheduleDraw();
    },
    // #32 — render quality while moving: 'smooth' | 'balanced' | 'full'.
    setMoveQuality(mode) {
      moveQuality = mode === "full" || mode === "smooth" ? mode : "balanced";
      scheduleDraw();
    },
    // Auto-detail (§6) — raise iterations with zoom. Off restores the base count.
    setAutoDetail(b) {
      autoDetail = !!b;
      scheduleDraw();
    },
    getAutoDetail: () => autoDetail,
    // Manual Detail override (§6) — set the ABSOLUTE effective iteration count
    // from the slider, bypassing auto-detail's zoom boost so the user can go
    // below its floor (base + boost). Pass null to clear. One-shot: any zoom
    // (zoomAtCenter/zoomToPixel) and camera loads (frameTo) clear it, handing
    // detail back to auto-detail — see §6.
    setDetailOverride(n) {
      entryClampArm.disarm(); // #562 — a manual write cancels any pending arm
      detailOverride =
        n == null ? null : Math.min(ITER_CEIL, Math.max(2, Math.round(n)));
      scheduleDraw();
    },
    // The iteration count actually being rendered right now (base + zoom boost) —
    // lets the HUD show "Detail N" so the auto-raise is visible, not magic. Scenes
    // (per-object iters) report 0, matching their writeGlobals.
    currentIters: () =>
      formula && !formula.objects ? effectiveIters(formula.iters) : 0,
    // Diagnostics (#): the actual render tier + march params of the last drawn
    // frame, PLUS why the pump last bailed and how many frames it has drawn.
    // Feeds the diag report's `render=…` line: `frames:0 skip:"compiling"` means
    // no GPU frame ever drew (and why); a real spin frame shows its tier + ms so
    // a slow spin/drag reveals exactly what it paid for instead of us guessing.
    renderInfo: () => ({
      frames: framesDrawn,
      glDead, // WebGL2 tier died (black-canvas class) → app fell to ASCII
      df64: lastDf64, // deep zoom P4 — did the last written frame engage df64?
      perturb: lastPt, // pt tier (PR-4) — did the last frame ride the delta variant?
      ptOrbitBuilds, // #551 — reference-orbit rebuilds this session (re-pin cadence)
      ptWhy: lastPtWhy, // #551 — '' when engaged, else latch|cheap|ineligible|orbit
      ptEligible: !!(formula && ptElig(formula)),
      // Whether the tier is allowed to engage AT ALL on this device/session
      // (touch defaults off until the fleet pass). The HUD must gate its
      // wall law on this, not eligibility alone — an eligible formula on a
      // pt-dark device is still f32/df64-walled (2026-08-01 iPad report #2:
      // the badge stayed silent through a wall-regime grind).
      ptAvailable: ptMode !== "off",
      skip: lastSkip,
      autoRotate,
      cw: canvas.width, // GPU canvas backing store (300×150 ⇒ never sized/drawn)
      ch: canvas.height,
      inFlight,
      // Fence watchdog recovery (#460/#473): count of watchdog-killed fences
      // recovered this session, and the current session settle-resolution cap
      // (1 = never tripped). Nonzero fenceTimeouts with frames still advancing =
      // the recovery working; a permanently high inFlight would be the old wedge.
      fenceTimeouts,
      settleScaleCap,
      // Per-frame cost governor (#476): the session render-scale cap driven by
      // MEASURED settled ms (1 = never tripped), whether it's armed on this
      // device, and its budget. governorScale < 1 with frames advancing = the
      // governor holding a heavy formula under budget instead of grinding fatal.
      governorScale,
      governorActive,
      governorBudgetMs: GOV_BUDGET,
      lastFullMs, // last MEASURED settled-frame GPU time — what the governor keys off
      deviceLost,
      last: lastRender,
    }),
    requestDraw: scheduleDraw,
    zoom(factor) {
      zoomAtCenter(factor);
    },
    // Keyboard nudges (§3.5) — the app drives whichever tier is visible through
    // these three (plus zoomBy) instead of reaching into cam directly. The
    // ASCII view exposes the same trio.
    orbitBy(dxDeg, dyDeg) {
      glide = null; // a deliberate nudge stops a coast
      finishZoomChase(); // an orbit changes fwd — end any eased zoom (#464)
      cam.orbit(dxDeg, dyDeg);
      bumpInteract();
    },
    panPx(dxPx, dyPx) {
      panByPixels(dxPx, dyPx);
      bumpInteract();
    },
    zoomBy(factor) {
      zoomAtCenter(factor);
    },
    // DE-scaled cruise (§4): hold-to-fly. dir +1 = in, −1 = back out. On
    // release, one final probe-and-repin leaves the user orbiting whatever
    // they flew up to; a miss keeps target/dist untouched (void orbit is
    // accepted — double-click or the zoom-HUD reset recover, never a silent
    // dist snap that would teleport the eye).
    setCruise(on, dir = 1) {
      if (!on) {
        if (!cruise) return;
        cruise = null;
        if (formula) {
          const b = cam.basis();
          const h = surfaceAhead(b.eye, b.fwd);
          if (h != null && h > 0) {
            cam.target = [
              b.eye[0] + b.fwd[0] * h,
              b.eye[1] + b.fwd[1] * h,
              b.eye[2] + b.fwd[2] * h,
            ];
            cam.dist = h;
          }
        }
        quality = "full";
        needsDraw = true;
        scheduleDraw();
        return;
      }
      const d = dir < 0 ? -1 : 1;
      if (cruise && cruise.dir === d) return; // key auto-repeat
      cruise = { dir: d, t: performance.now(), tick: 0, hasHit: false };
      glide = null;
      zoomChase = null; // cruise owns the motion — abandon any eased zoom (#464)
      detailOverride = null; // continuous zoom — hand detail back to auto (§6)
      entryClampArm.disarm(); // #562 — see zoomAtCenter
      clearTimeout(settleTimer); // pump owns quality while cruising
      scheduleDraw();
    },
    cruising: () => !!cruise,
    captureFrame,
    captureSplatGBuffer,
    setOffline,
    // Render primitives (issue #77 — core owns no DOM): the engine renders
    // pixels only. stillBlob → a PNG Blob (app turns it into a download);
    // thumbTileCached/renderThumbTile/beginThumbs/endThumbs → the thumbnail
    // grid's per-tile pixels (app builds + owns the grid markup).
    stillBlob,
    stillBlobTiled, // TILED_EXPORT PR-2 — >4096 via an N×M tile grid
    tiledExportPlan, // grid + time estimate for the picker, before committing
    thumbTileCached,
    renderThumbTile,
    beginThumbs,
    endThumbs,
  };
}

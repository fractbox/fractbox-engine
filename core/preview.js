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

import { createRenderer } from "./renderer.js";
import { createRendererGL } from "./renderer_gl.js";
import { makeCamera } from "./camera.js";
import { isEscapeTime, isNumericDE, effectiveDeOption, activeOps } from "./operators.js";
import { looseDE, hybridLooseDE, hybridDeFamily } from "./stability.js";
import { defaultColoring } from "./coloring.js";
import { makeDE } from "./cpu.js";
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

// Escape-time power maps need a small bailout (r=8) or rᵖᵒʷᵉʳ overflows fp32;
// IFS folds stay bounded so a huge radius is harmless.
const bailoutFor = (f) => (isEscapeTime(f) || isNumericDE(f) ? 64.0 : 1.0e6);
const D2R = Math.PI / 180;

// Deep zoom (§5): the default framing distance the old fixed near/far (0.02/80)
// were tuned for — near/far now scale off cam.dist so a deep zoom doesn't start
// inside the surface (near too far) or get far-capped against nearby geometry
// (far too close). Also the HUD's magnification reference (M = REF_DIST/dist).
export const REF_DIST = 24;
const TNEAR_K = 0.02 / REF_DIST;
const TFAR_K = 80.0 / REF_DIST;

// Deep zoom §6 (Phase 3) — depth-adaptive step budget bounds. STEP_CEIL must not
// exceed the GLSL literal loop cap (shader_gl.js's `for (int i = 0; i < 512; …)`,
// both fragment shaders) — pushing past it needs raising that literal (a shader
// recompile), out of scope for v1. DEPTH_CAP (×3 steps / ÷√3 deScale at the
// deepest) is a starting point tuned against fps on the Mac mini reference GPU;
// revisit if deep zoom feels sluggish or thin features still drop out.
const STEP_CEIL = 512;
const DEPTH_CAP = 3;
// Auto-detail (§6) — a distance-estimated fractal has a FIXED finest scale for a
// given iteration count, so zooming past it just smooths out (the DE becomes a
// bound, not the surface). Raise the iteration count as you zoom in — roughly one
// extra iteration per zoom octave (log2 of magnification), the natural fractal
// law — so fine structure keeps resolving. ITER_CEIL is the GLSL literal loop cap
// (shader_gl.js `for (int i = 0; i < 64; …)`); WGSL's bound is a uniform, but we
// clamp to 64 for cross-backend parity. Iteration count is the dominant DE cost,
// so this is the opt-out "Detail with zoom" toggle, not an always-on cost.
const ITER_CEIL = 64;
const ITER_PER_OCTAVE = 1.0;

export async function createPreview(canvas, opts = {}) {
  const isTouch = navigator.maxTouchPoints > 0;
  const DPR_CAP = isTouch ? 1.0 : 2.0;
  const onFrame = opts.onFrame || (() => {}); // (ms) after each frame
  const onFrameStart = opts.onFrameStart || (() => {});

  let renderer = null,
    hasGPU = false,
    backend = "none";
  let formula = null;
  // Formula morph (VIDEO_EXPORT_DRAWER_V2 tier-2 spike): when set, writeFrame
  // blends the current formula's DE toward morph.f by morph.t (WebGPU only —
  // other tiers render formula A). Null = every path exactly as before.
  let morph = null;
  // Coloring-mode crossfade (timeline transitions between views with different
  // color modes / palette toggles): {t, modeB, palOnB} — the shader shades
  // under both modes and mixes. Null = legacy shading. WebGPU only (the GL
  // writer ignores the extra word — those tiers keep the midpoint snap).
  let colorBlend = null;
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
    autoDetail = opts.autoDetail ?? true, // raise iters with zoom depth (§6)
    detailOverride = null, // §6 — manual ABSOLUTE iters set from the Detail slider;
    // one-shot, lets the user drop below auto-detail's floor; cleared on next zoom
    autoRotate = false,
    settleTimer = null,
    spinSpeed = 0.7,
    spinTilt = 0; // 0° = turntable (spin around +Z) … 90° = tumble (around +X)
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
    renderer = await createRenderer(canvas);
    hasGPU = true;
    backend = "webgpu";
    // Surface async WebGPU validation errors (P3) — without this a bad bind
    // group / pipeline silently drops every pass in the submit: black canvas,
    // no exception anywhere. Costs nothing when healthy.
    renderer.device.addEventListener?.("uncapturederror", (e) =>
      console.error("WebGPU uncaptured:", e.error?.message || e),
    );
  } catch (e) {
    if (force !== "ascii" && force !== "none") {
      try {
        renderer = await createRendererGL(canvas);
        hasGPU = true;
        backend = "webgl2";
      } catch (e2) {
        console.error("WebGL2 unavailable:", e2?.message || e2);
        hasGPU = false;
      }
    }
    if (!hasGPU && !force) console.warn("WebGPU unavailable:", e?.message || e);
  }

  function scheduleDraw() {
    if (!hasGPU || offline) return;
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
  // Auto-detail (§6): the iteration count actually sent to the shader — the
  // formula's base iters, plus ~one per zoom octave once zoomed in, clamped to
  // ITER_CEIL. At the default framing (M≈1) the boost is 0, so nothing changes
  // until you zoom. Off → the base count, exactly as before. Single-object and
  // hybrid only; CSG scenes carry per-object iters and stay shallow (§14).
  function effectiveIters(base) {
    const b = base || 0;
    // Manual override (§6): the user dragged the Detail slider to an ABSOLUTE
    // count — honor it verbatim (2..ITER_CEIL) so they can go BELOW auto-detail's
    // floor (base + zoom boost). One-shot: the next zoom clears it and auto-detail
    // resumes from the base.
    if (detailOverride != null) return Math.min(ITER_CEIL, Math.max(2, detailOverride));
    if (!autoDetail) return b;
    const M = REF_DIST / Math.max(cam.dist, 1e-12);
    const extra = Math.max(0, Math.round(Math.log2(Math.max(M, 1)) * ITER_PER_OCTAVE));
    return Math.min(ITER_CEIL, b + extra);
  }
  function qualityParams(f) {
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
      ? { scale: 1.0, steps: DPR_CAP < 2 ? 220 : 320, eps: 0.001, deScale: 0.3 }
      : {
          scale: 1.0,
          steps: DPR_CAP < 2 ? 140 : 200,
          eps: 0.001,
          deScale: 0.5,
        };
    // Deep zoom §6 (Phase 3) — depth is an INPUT to the same {steps, deScale}
    // knobs, not a new axis: deeper zoom needs more march steps and a tighter
    // step size to resolve thin deep features (the reconstruction keeps eps·t
    // sane since t is residual-scaled — §3). Single-object path only in v1;
    // CSG scenes use sceneDeScale, a separate knob, and aren't recentered (§14).
    if (f && !f.objects) {
      const M = REF_DIST / Math.max(cam.dist, 1e-12);
      const depth = Math.min(1 + Math.log10(Math.max(M, 1)) * 0.35, DEPTH_CAP);
      if (depth > 1) {
        full.steps = Math.min(STEP_CEIL, Math.round(full.steps * depth));
        full.deScale = Math.max(0.25, full.deScale / Math.sqrt(depth));
      }
    }
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
        scale: isTouch ? 0.6 : 0.7,
        steps: 48,
        eps: 0.003,
        deScale: 0.65,
        cheap: true, // P1: drop shadow + AO marches while dragging (word flip, no recompile)
      };
    // balanced (default): full march budget (shape-stable), lower resolution,
    // and CHEAP SHADING — shadow/AO are shading, not shape, and their marches
    // dominate frame cost on heavy scenes (2 fps on an 8-object CSG while a
    // light slider drags). The shading pop on settle coincides with the
    // existing resolution pop.
    return { ...full, scale: isTouch ? 0.7 : 0.8, cheap: true };
  }
  function sizeCanvas(scale) {
    const r = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, DPR_CAP) * scale;
    const w = Math.max(1, Math.floor(r.width * dpr)),
      h = Math.max(1, Math.floor(r.height * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
  }
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
  function sceneDeScale(objects) {
    const CARVE_DESCALE = 0.25;
    let m = 0.5;
    let carving = false;
    for (const o of objects) {
      // Primitives (box/sphere, objType > 0) are exact; only an IFS op-slice can
      // be loose (scale < 2, see stability.looseDE) and tighten the scene deScale.
      // Judge the ACTIVE slice only — the emitters skip muted ops (per-op scene
      // mute), so a muted scale op must not vouch for a bound it isn't providing.
      const loose =
        Number(o.objType) > 0
          ? false
          : looseDE({
              ops: (o.ops || []).filter((op) => !op.muted),
              deOption: o.deOption,
              iters: o.iters,
            });
      m = Math.min(m, loose ? 0.3 : 0.5);
      const cmb = (o.combine ?? o.combineType ?? 0) & 3;
      if (cmb === 2 || cmb === 3) carving = true; // subtract / intersect
    }
    if (carving) m = Math.min(m, CARVE_DESCALE);
    return m;
  }

  // P1 — the cheap interactive tier renders shadowless / AO-less by overriding
  // the two light words (the shader skips both marches entirely at 0). The
  // settled frame passes coloring.light through untouched.
  function shadeLight(q) {
    if (!q?.cheap) return coloring.light;
    return { ...coloring.light, shadow: 0, ao: 0 };
  }

  function writeFrame(f, q, res, c) {
    const dist = (c || cam).dist;
    const tNear = dist * TNEAR_K,
      tFar = dist * TFAR_K;
    // Hybrid iteration (IDEAS ①, docs/design/HYBRID_ITERATION.md §3.5) —
    // mutually exclusive with f.objects (§3.8). Objects-first tie-break,
    // matching sanitize.js/cpu.js (Formula Outline Step 3 §4a): a dual-set
    // formula is malformed input, and every layer must break the tie the same
    // way or the badge and the pixels disagree. `.length`-based (not truthy)
    // so an `objects: []` + hybrid input dispatches here instead of falling
    // through to writeScene([]), which throws. Slot A is the formula's own
    // ops/addC (unchanged); writeHybrid concatenates slot B's ops onto the
    // SAME shared ops buffer writeOps already uses (WGSL), or codegens a
    // hybrid-aware program (GLSL) — see renderer.js/renderer_gl.js.
    if (f.hybrid && !f.objects?.length && renderer.writeHybrid) {
      const opsA = activeOps(f);
      const opsB = (f.hybrid.b?.ops || []).filter((o) => !o.muted);
      renderer.writeHybrid(
        opsA,
        opsB,
        f.hybrid.schedule,
        !!f.addC,
        !!f.hybrid.b?.addC,
      );
      const family = hybridDeFamily(f);
      renderer.writeGlobals({
        res,
        cam: c || cam,
        iters: effectiveIters(f.iters), // auto-detail (§6)
        opCount: 0, // unused by mapDE_hybrid — the hyb word carries aOpCount/bOpCount
        addC: false, // superseded by the per-slot addCA/addCB below
        maxSteps: q.steps,
        bailout: family === "ifs" ? 1.0e6 : 64.0,
        eps: q.eps,
        deScale: q.deScale ?? 0.85,
        colA: coloring.colA,
        colB: coloring.colB,
        bg: coloring.bg,
        colorMode: coloring.mode,
        tNear,
        tFar,
        deOption: family === "ifs" ? (f.deOption ?? 2) : 0,
        julia: f.julia, // formula-level (§3.8) — one seed, not per-slot
        juliaC: f.juliaC,
        palette: coloring.palette,
        light: shadeLight(q),
        objectCount: 0,
        hybrid: {
          aOpCount: opsA.length,
          bOpCount: opsB.length,
          scheduleA: f.hybrid.schedule?.a ?? 1,
          scheduleB: f.hybrid.schedule?.b ?? 1,
          // julia is ORed in separately at the shader level (mapDE_hybrid reads
          // G.jc.w) — not folded in here, so this stays exactly "this slot's
          // own addC", matching what §3.8 calls "the per-slot addC already
          // decides which iterations fold it in" (the formula-level Julia OR
          // is orthogonal, applied uniformly to both slots by the shader).
          addCA: !!f.addC,
          addCB: !!f.hybrid.b?.addC,
        },
        colorBlend,
      });
      return;
    }
    // Multi-object scene path (additive — only when f.objects is present).
    if (f.objects && renderer.writeScene) {
      const n = renderer.writeScene(f.objects);
      renderer.writeGlobals({
        res,
        cam: c || cam,
        iters: 0,
        opCount: 0,
        addC: false,
        maxSteps: q.steps,
        bailout: 1.0e6, // IFS folds stay bounded; box doesn't iterate
        eps: q.eps,
        deScale: sceneDeScale(f.objects),
        colA: coloring.colA,
        colB: coloring.colB,
        bg: coloring.bg,
        colorMode: 0, // multi-object is surface-coloring only (§3.8)
        deOption: 2,
        julia: false,
        palette: coloring.palette,
        light: shadeLight(q),
        objectCount: n,
        tNear,
        tFar,
      });
      return;
    }
    // Formula morph (VIDEO_EXPORT_DRAWER_V2 tier-2 spike, WebGPU only): blend
    // two PLAIN formulas' distance fields — d = mix(dA, dB, t) — a true 3D
    // morph for the flythrough's shape hard-cuts. Plain-only on both ends
    // (no objects/hybrid); callers fall back to the hard cut otherwise. A is
    // the current formula (it also drives orbit-trap coloring — the app lerps
    // palettes separately); B rides the concatenated op slice + morph words.
    if (
      morph &&
      renderer.writeMorph &&
      !f.objects?.length &&
      !f.hybrid &&
      !morph.f.objects?.length &&
      !morph.f.hybrid
    ) {
      const opsA = activeOps(f);
      const opsB = activeOps(morph.f);
      renderer.writeMorph(opsA, opsB);
      renderer.writeGlobals({
        res,
        cam: c || cam,
        iters: effectiveIters(f.iters), // auto-detail (§6)
        opCount: opsA.length,
        addC: f.addC,
        maxSteps: q.steps,
        bailout: bailoutFor(f), // A's own; B carries its own in the morph word
        eps: q.eps,
        deScale: q.deScale ?? 0.85,
        colA: coloring.colA,
        colB: coloring.colB,
        bg: coloring.bg,
        colorMode: coloring.mode,
        tNear,
        tFar,
        deOption: effectiveDeOption(f),
        julia: f.julia,
        juliaC: f.juliaC,
        palette: coloring.palette,
        light: shadeLight(q),
        objectCount: 0,
        morph: {
          bOpCount: opsB.length,
          bIters: morph.f.iters ?? 12,
          bAddC: !!morph.f.addC,
          bJulia: !!morph.f.julia,
          bJuliaC: morph.f.juliaC,
          bDeOption: effectiveDeOption(morph.f),
          bailB: bailoutFor(morph.f), // B's OWN escape bound (see renderer)
          t: morph.t,
          swell: morph.swell,
        },
        colorBlend,
      });
      return;
    }
    const oc = renderer.writeOps(activeOps(f));
    renderer.writeGlobals({
      res,
      cam: c || cam,
      iters: effectiveIters(f.iters), // auto-detail (§6)
      opCount: oc,
      addC: f.addC,
      maxSteps: q.steps,
      bailout: bailoutFor(f),
      eps: q.eps,
      deScale: q.deScale ?? 0.85,
      colA: coloring.colA,
      colB: coloring.colB,
      bg: coloring.bg,
      colorMode: coloring.mode,
      tNear,
      tFar,
      deOption: effectiveDeOption(f),
      julia: f.julia,
      juliaC: f.juliaC,
      palette: coloring.palette,
      light: shadeLight(q),
      objectCount: 0,
      colorBlend,
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
  function accumCap() {
    if (lastFullMs < 20) return 32;
    if (lastFullMs < 45) return 12;
    if (lastFullMs < 100) return 6;
    return 0;
  }
  let accumN = -1; // -1 = not accumulating; ≥1 = samples in the average
  let accumStartT = 0; // refinement begins 250 ms after the settled frame
  let accumTick = 0; // refinement draws every OTHER rAF (compositor headroom)
  function r2(i) {
    const f = (x) => x - Math.floor(x);
    return [f(0.5 + i * 0.7548776662466927) - 0.5, f(0.5 + i * 0.5698402909980532) - 0.5];
  }
  // P4 lens disk samples (mirrors renderer.js lensSample — golden-ratio pair).
  function lensOf(i) {
    const f = (x) => x - Math.floor(x);
    const r = Math.sqrt(f(0.5 + i * 0.6180339887498949));
    const th = 2 * Math.PI * f(0.5 + i * 0.38196601125010515);
    return [r * Math.cos(th), r * Math.sin(th)];
  }

  async function pump() {
    scheduled = false;
    // Video export: while an offline render owns the canvas, the live pump must
    // not draw — otherwise a queued rAF pump fires during captureFrame's await,
    // resizes + redraws the canvas (spin-advanced, live size) between the frame's
    // draw and its readback, so that frame captures the live view → export flicker.
    if (offline) return;
    if (autoRotate) {
      cam.spinAround(spinAxis(), spinSpeed);
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
        const h = surfaceAhead(b.eye, b.fwd);
        cruise.hasHit = h != null && h > 0;
        if (cruise.hasHit) {
          cam.target = [
            b.eye[0] + b.fwd[0] * h,
            b.eye[1] + b.fwd[1] * h,
            b.eye[2] + b.fwd[2] * h,
          ];
          cam.dist = h; // eye stays put (|eye − target| = h) — seamless repin
        }
      }
      const s = cruiseAdvance(
        { dir: cruise.dir, hasHit: cruise.hasHit || getCpuDE() == null, dist: cam.dist },
        dt,
      );
      if (s.distFactor !== 1)
        cam.dist = Math.max(1e-9, Math.min(200, cam.dist * s.distFactor));
      if (s.drift) {
        const b = cam.basis();
        cam.target[0] += b.fwd[0] * s.drift;
        cam.target[1] += b.fwd[1] * s.drift;
        cam.target[2] += b.fwd[2] * s.drift;
      }
      quality = "low";
      needsDraw = true;
    }
    if (busy || !formula) return;
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
        scheduled = true;
        requestAnimationFrame(pump);
        return;
      }
      busy = true;
      const [jx, jy] = r2(accumN);
      // P4: each accumulation sample also gets a lens point (DOF converges
      // with the AA); the golden-ratio disk stream lives renderer-side.
      const [lx, ly] = lensOf(accumN);
      renderer.writeJitter(jx, jy, 1 / (accumN + 1), lx, ly);
      try {
        renderer.drawAccum();
        await renderer.device.queue.onSubmittedWorkDone();
        accumN++;
      } catch (e) {
        console.error("accum:", e);
        accumN = -1;
      }
      busy = false;
      if ((needsDraw || (accumN >= 1 && accumN < accumCap())) && !scheduled) {
        scheduled = true;
        requestAnimationFrame(pump);
      }
      return;
    }
    if (!needsDraw) return;
    needsDraw = false;
    busy = true;
    onFrameStart();
    const q = qualityParams(formula);
    sizeCanvas(q.scale);
    writeFrame(formula, q, [canvas.width, canvas.height]);
    const t0 = performance.now();
    // Accumulate only from a settled, still, full-quality frame (structural
    // WebGPU-only: the GL renderer has no drawAccum — PR-#55 fallback pattern).
    const wantAccum =
      quality === "full" && !autoRotate && !q.cheap && !!renderer.drawAccum;
    const tFull = quality === "full" && !q.cheap;
    try {
      if (wantAccum) {
        renderer.writeJitter(0, 0, 1, 0, 0); // base sample: pixel center, lens center
        renderer.drawAccum();
      } else {
        renderer.draw();
      }
      await renderer.device.queue.onSubmittedWorkDone();
      accumN = wantAccum ? 1 : -1;
      accumStartT = performance.now() + 250;
    } catch (e) {
      console.error("draw:", e);
      accumN = -1;
    }
    busy = false;
    const dt = performance.now() - t0;
    if (tFull) lastFullMs = dt; // feeds the adaptive accumulation budget
    onFrame(dt);
    if (
      needsDraw ||
      autoRotate ||
      glide ||
      cruise ||
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
  const getCpuDE = () => {
    if (cpuDEFor === formula) return cpuDE; // cache hit (formula unchanged)
    cpuDEFor = formula;
    try {
      cpuDE = formula ? makeDE(formula) : null;
    } catch {
      cpuDE = null; // some formula shape cpu.js can't build → fall back to plain zoom
    }
    return cpuDE;
  };
  // Distance from `eye` along `fwd` to the surface straight ahead (null if the
  // ray misses or the eye is already inside) — the probe lives in the pure,
  // unit-tested zoomsurface.js; here we just feed it the current DE and the
  // dist-scaled near/far bounds (matching the shader's tNear/tFar).
  const surfaceAhead = (eye, fwd) =>
    surfaceHitDist(
      getCpuDE(),
      eye,
      fwd,
      cam.dist * TNEAR_K,
      cam.dist * (TFAR_K * 1.5),
    );
  // One zoom step. On zoom IN, re-pin the orbit target to the surface straight
  // ahead first (keeps the eye exactly put — target = eye + fwd·h, dist = h —
  // so it's visually seamless), THEN apply the factor so the eye moves a
  // fraction of the way toward that surface point. Zoom OUT is left as a plain
  // dolly so backing away never drifts the pivot unexpectedly.
  const zoomAtCenter = (factor) => {
    detailOverride = null; // §6 — a zoom hands detail back to auto-detail
    if (factor < 1 && formula) {
      const b = cam.basis();
      const h = surfaceAhead(b.eye, b.fwd);
      if (h != null && h > 0) {
        cam.target = [
          b.eye[0] + b.fwd[0] * h,
          b.eye[1] + b.fwd[1] * h,
          b.eye[2] + b.fwd[2] * h,
        ];
        cam.dist = h; // eye unchanged (see above); pivot now rides the surface
      }
    }
    cam.zoom(factor);
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
    let rx = b.fwd[0] + ndcX * aspect * tanF * b.right[0] + ndcY * tanF * b.up[0];
    let ry = b.fwd[1] + ndcX * aspect * tanF * b.right[1] + ndcY * tanF * b.up[1];
    let rz = b.fwd[2] + ndcX * aspect * tanF * b.right[2] + ndcY * tanF * b.up[2];
    const rl = Math.hypot(rx, ry, rz) || 1;
    return { b, ray: [rx / rl, ry / rl, rz / rl] };
  };
  // Probe the surface along a pixel's ray and re-pin the orbit pivot onto the
  // hit point (eye stays put: target = eye + ray·h ⇒ |eye−target| = h = dist).
  // Returns true on a hit. Double-click only — the off-axis repin ROTATES the
  // view onto the feature, which is that gesture's intent; wheel zoom must
  // stay seamless and uses the translate-along-ray anchor below instead.
  const repinToPixel = (clientX, clientY) => {
    const { b, ray } = pixelRay(clientX, clientY);
    const h = surfaceHitDist(
      getCpuDE(),
      b.eye,
      ray,
      cam.dist * TNEAR_K,
      cam.dist * (TFAR_K * 1.5),
    );
    if (h == null || h <= 0) return false;
    cam.target = [b.eye[0] + ray[0] * h, b.eye[1] + ray[1] * h, b.eye[2] + ray[2] * h];
    cam.dist = h;
    return true;
  };
  // Double-click / double-tap to zoom toward the clicked POINT (map-style),
  // reusing the same surface probe but along the ray through the clicked pixel
  // instead of the centre. On a surface hit: recenter the orbit target on that
  // feature AND pull the eye `factor`× closer to it in one step. A miss (clicked
  // empty space) falls back to a plain centre zoom so the gesture never no-ops.
  const zoomToPixel = (clientX, clientY, factor) => {
    detailOverride = null; // §6 — see zoomAtCenter
    const before = cam.dist; // repin overwrites dist with the hit distance
    if (repinToPixel(clientX, clientY)) {
      // Pivot re-pinned on the clicked surface point; zoom by a FIXED factor of
      // the PRIOR distance (not h) — so every double-click is a predictable ~2×,
      // whether you clicked a near or far feature, and the pivot rides the
      // surface (like wheel zoom) so the eye can't cross into it.
      cam.dist = Math.max(1e-9, before * factor);
      bumpInteract();
    } else {
      cam.zoom(factor); // clicked empty space → plain zoom, never a dead gesture
      bumpInteract();
    }
  };

  // ── Orbit / zoom / pinch gestures ─────────────────────────────────────────
  const ptrs = new Map();
  canvas.addEventListener("pointerdown", (e) => {
    ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY, t: e.timeStamp });
    glide = null; // grabbing the canvas stops a coast dead (§3.3)
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
    // collide with the default plain-drag orbit.
    if (e.shiftKey) {
      dragMode = "pan";
      panByPixels(e.clientX - p.x, e.clientY - p.y);
      ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY, t: e.timeStamp });
      bumpInteract();
      return;
    }
    // Orbit speed scales with zoom (dist): finer when zoomed in, never
    // sluggish. Halved while cruising (§4) — the drag is steering the nose.
    dragMode = "orbit";
    const s = (cruise ? 0.2 : 0.4) * Math.max(0.4, Math.min(1.3, cam.dist / 18));
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
      if (factor >= 1 || !formula) {
        wheelProbe = null;
        zoomAtCenter(factor);
        return;
      }
      const now = performance.now();
      const fresh =
        wheelProbe &&
        now - wheelProbe.t < 150 &&
        Math.hypot(e.clientX - wheelProbe.x, e.clientY - wheelProbe.y) < 8;
      if (!fresh) {
        const { b, ray } = pixelRay(e.clientX, e.clientY);
        const h = surfaceHitDist(
          getCpuDE(),
          b.eye,
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
        detailOverride = null; // §6 — a zoom hands detail back to auto-detail
        const k = 1 - factor;
        const delta = wheelProbe.h * k;
        const b = cam.basis();
        const back = cam.dist * k;
        cam.target = [
          cam.target[0] + wheelProbe.ray[0] * delta - b.fwd[0] * back,
          cam.target[1] + wheelProbe.ray[1] * delta - b.fwd[1] * back,
          cam.target[2] + wheelProbe.ray[2] * delta - b.fwd[2] * back,
        ];
        cam.dist = Math.max(1e-9, cam.dist * factor);
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
  window.addEventListener("resize", scheduleDraw);

  function frameTo(c) {
    if (!c) return;
    detailOverride = null; // §6 — fresh camera/formula → auto-detail from its base
    cam.yaw = (c.yawDeg ?? 35) * D2R;
    cam.pitch = (c.pitchDeg ?? 22) * D2R;
    cam.dist = c.dist ?? 24;
    cam.fov = (c.fovDeg ?? 42) * D2R;
    // Pan target (§5) — absent on every existing saved camera, so defaults to
    // the origin (today's implicit behavior, unchanged for all current presets).
    cam.target = Array.isArray(c.target) ? c.target.slice(0, 3) : [0, 0, 0];
  }
  const camObj = () => ({
    yawDeg: cam.yaw / D2R,
    pitchDeg: cam.pitch / D2R,
    dist: cam.dist,
    fovDeg: cam.fov / D2R,
    target: cam.target.slice(),
  });

  // Render the current view to a high-res PNG Blob (engine side of PNG export;
  // turning it into a download is the app's job). Returns null when there's
  // nothing to render (no GPU / no formula); a failed canvas capture throws
  // (the caller restores nothing — the finally here does). opts.metadata: PNG
  // text chunks (see pngmeta.js) spliced in so the saved image carries the
  // formula/share-URL and can re-open (docs/design/PNG_METADATA.md).
  async function stillBlob(opts = {}) {
    if (!hasGPU || !formula) return null;
    const rect = canvas.getBoundingClientRect();
    const H = 900,
      W = Math.round(H * ((rect.width || 4) / (rect.height || 3)));
    const prevW = canvas.width,
      prevH = canvas.height;
    busy = true;
    try {
      canvas.width = W;
      canvas.height = H;
      writeFrame(formula, { steps: 220, eps: 0.0006, deScale: 0.5 }, [W, H]);
      // P2: PNG stills accumulate 24 jittered samples (offline-grade AA) when
      // the renderer supports it; the GL tier keeps the single-sample draw.
      if (renderer.drawAccum) {
        renderer.writeJitter(0, 0, 1);
        renderer.drawAccum();
        for (let i = 1; i < 24; i++) {
          const f = (x) => x - Math.floor(x);
          renderer.writeJitter(
            f(0.5 + i * 0.7548776662466927) - 0.5,
            f(0.5 + i * 0.5698402909980532) - 0.5,
            1 / (i + 1),
          );
          renderer.drawAccum();
        }
        renderer.writeJitter(0, 0, 0);
      } else {
        renderer.draw();
      }
      await renderer.device.queue.onSubmittedWorkDone();
      let blob = await new Promise((res) => canvas.toBlob(res, "image/png"));
      if (!blob) throw new Error("canvas capture returned null");
      if (opts.metadata && opts.metadata.length) {
        const bytes = embedChunks(new Uint8Array(await blob.arrayBuffer()), opts.metadata);
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

  // Video export (docs/design/VIDEO_EXPORT.md §7.1) — render ONE frame of the
  // current formula/coloring/camera at a chosen resolution and resolve when the
  // GPU is done, returning an ImageBitmap the caller can feed to a VideoEncoder.
  // Generalizes exportPNG; used by the offline HQ flythrough render. The caller
  // wraps setOffline(true)/(false) around its loop so the live pump stays out of
  // the way (this method draws directly, not via the pump). Restores canvas size
  // each call; does NOT scheduleDraw (the loop owns redraws — see setOffline).
  async function captureFrame(opts = {}) {
    if (!hasGPU || !formula) return null;
    const rect = canvas.getBoundingClientRect();
    const H = opts.h || 900;
    const W = opts.w || Math.round(H * ((rect.width || 4) / (rect.height || 3)));
    const quality = opts.quality || { steps: 220, eps: 0.0006, deScale: 0.5 };
    // P2: opts.samples > 1 → renderToImage runs N jittered march+accumulate
    // rounds (movie-grade AA, offline so frame time is free).
    const samples = Math.max(1, opts.samples | 0);
    writeFrame(formula, quality, [W, H]);
    // WebGPU: render into an OFFSCREEN texture and copy the pixels straight back
    // (renderToImage — the deterministic path the thumbnail gallery uses). Reading
    // back the PRESENTED canvas instead suffers swap-chain double-buffering: the
    // read alternates between buffers, some still holding the uninitialised clear
    // colour → the "green frame" bug in offline export. This never touches the
    // visible canvas, so it also can't flicker the live view.
    if (renderer.renderToImage) {
      const img = await renderer.renderToImage(W, H, samples);
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
        writeFrame(formula, quality, [W, H]);
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
  const thumbCache = new Map();
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
      // Optional per-tile coloring override (curated preset looks) — part of
      // the fingerprint so the same shape can be tiled under different looks.
      look || null,
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
  const thumbTileCached = (p, look) => thumbCache.get(thumbKey(p, look)) || null;
  async function renderThumbTile(p, W, H, look) {
    const key = thumbKey(p, look);
    const hit = thumbCache.get(key);
    if (hit) return hit;
    const s2d = thumbScratch(W, H);
    // Per-tile coloring override (curated preset looks): swap the coloring in
    // for this one render and restore it after. Palette cycling is forced off
    // first so the look's flat colors are what the tile shows — unless the
    // look itself carries a palette (a saved formula's colors). The restore is
    // guarded: if the app called setColoring while we awaited the readback,
    // keep ITS coloring, not our snapshot.
    const prev = coloring;
    const temp = look
      ? { ...coloring, palette: { ...(coloring.palette || {}), on: false }, ...look }
      : null;
    if (temp) coloring = temp;
    try {
      writeFrame(
        p,
        { steps: 200, eps: 0.001, deScale: 0.5, cheap: true },
        [W, H],
        makeCamera(p.camera),
      );
      s2d.putImageData(await renderer.renderToImage(W, H), 0, 0);
    } finally {
      if (temp && coloring === temp) coloring = prev;
    }
    const url = thumbCanvas.toDataURL("image/png");
    thumbCache.set(key, url);
    return url;
  }
  function beginThumbs() {
    busy = true;
  }
  function endThumbs() {
    busy = false;
    scheduleDraw();
  }

  return {
    hasGPU,
    backend,
    cam,
    isTouch,
    camObj,
    frameTo,
    bumpInteract,
    setFormula(f) {
      formula = f;
      cpuDE = cpuDEFor = null; // invalidate the zoom-to-surface distance field
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
      if (b) scheduleDraw();
    },
    setSpinSpeed(degPerFrame) {
      spinSpeed = degPerFrame;
      if (autoRotate) scheduleDraw();
    },
    setSpinTilt(deg) {
      spinTilt = deg;
      if (autoRotate) scheduleDraw();
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
      detailOverride =
        n == null ? null : Math.min(ITER_CEIL, Math.max(2, Math.round(n)));
      scheduleDraw();
    },
    // The iteration count actually being rendered right now (base + zoom boost) —
    // lets the HUD show "Detail N" so the auto-raise is visible, not magic. Scenes
    // (per-object iters) report 0, matching their writeGlobals.
    currentIters: () =>
      formula && !formula.objects ? effectiveIters(formula.iters) : 0,
    requestDraw: scheduleDraw,
    zoom(factor) {
      zoomAtCenter(factor);
    },
    // Keyboard nudges (§3.5) — the app drives whichever tier is visible through
    // these three (plus zoomBy) instead of reaching into cam directly. The
    // ASCII view exposes the same trio.
    orbitBy(dxDeg, dyDeg) {
      glide = null; // a deliberate nudge stops a coast
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
      detailOverride = null; // continuous zoom — hand detail back to auto (§6)
      clearTimeout(settleTimer); // pump owns quality while cruising
      scheduleDraw();
    },
    cruising: () => !!cruise,
    captureFrame,
    setOffline,
    // Render primitives (issue #77 — core owns no DOM): the engine renders
    // pixels only. stillBlob → a PNG Blob (app turns it into a download);
    // thumbTileCached/renderThumbTile/beginThumbs/endThumbs → the thumbnail
    // grid's per-tile pixels (app builds + owns the grid markup).
    stillBlob,
    thumbTileCached,
    renderThumbTile,
    beginThumbs,
    endThumbs,
  };
}

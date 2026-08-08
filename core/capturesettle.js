// Frame settle — the Globals/ops/objects write path, extracted from preview.js
// (EXPORT_P1_HEADLESS_CAPTURE PR-A). ONE implementation for both consumers:
// preview.writeFrame delegates here for every live frame (its live-only state —
// morph, colorBlend, auto-detail iters, the df64 zoom latch, lastFrameState —
// arrives via the `live` options bag), and the headless capture driver (PR-B)
// calls it directly with no live bag at all. Canvas/DOM-free by construction;
// Globals packing itself stays single-sourced in renderer.writeGlobals (the
// #362 offsetLo lesson — this module never packs words).
import { effectiveDeOption, activeOps, byKey } from "./operators.js";
import {
  TNEAR_K,
  TFAR_K,
  TFAR_MIN,
  TFAR_UNBOUNDED_MUL,
  unboundedScene,
  bailoutFor,
  resolveDeScale,
  sceneDeScale,
  shadeLight as policyShadeLight,
} from "./renderpolicy.js";
import { BAILOUT_ESCAPE, BAILOUT_IFS } from "./limits.js";
import { hybridDeFamily } from "./stability.js";
import { activeHybridSlots } from "./hybridmodel.js";
// cpu.js is the whole CPU/ASCII render tier (~117KB raw) — signalRange (auto-
// levels range) is the one export settleFrame needs from it, called every
// frame below. A static (or unconditional top-level dynamic) import here
// pinned all of cpu.js in the app's boot chunk even though nothing on the
// WebGPU first-paint path calls into it (issue #266) — Vite/Rolldown treats a
// module-scope `import()` as "always executes at eval time" and modulepreloads
// it right alongside the entry, which is no improvement at all. Wrapping the
// import() in a function that only runs on an explicit call is what actually
// gets it off the eager/preloaded set.
// This is NOT a "degrade until ready" seam — every caller of settleFrame
// (preview.js at createPreview, export-splat.mjs's --gpu path, this file's
// own tests) calls `ensureCpu()` once before its first settleFrame call, so
// the signalRange() call below is never racing the load — behavior is
// identical to the old static import, just off the boot chunk's parse/eval.
let _signalRange = null;
let _cpuLoading = null;
export function ensureCpu() {
  if (!_cpuLoading)
    _cpuLoading = import("./cpu.js").then((m) => {
      _signalRange = m.signalRange;
      return m;
    });
  return _cpuLoading;
}

// D0 leaf ids present in a scene (variant specialization) — null for non-scenes
// or a scene with no leaves. (Moved verbatim from preview.js.)
export function formulaLeafIds(f) {
  if (!Array.isArray(f?.objects)) return null;
  const s = new Set();
  for (const o of f.objects) {
    if (!o) continue;
    const id =
      (o.shapeId ?? 0) > 0
        ? o.shapeId
        : (o.objType ?? 0) > 0
          ? o.objType
          : o.boxBase
            ? 1
            : 0;
    if (id > 0) s.add(id & 0xff);
  }
  return s.size ? [...s].sort((a, b) => a - b) : null;
}

// Predict the op-set the write path will upload (prewarm lever #3). Mirrors
// writeOps/writeScene's byKey(op.key).id, muted filtered. Only a PREDICTION —
// the renderer latches the EXACT set from what it wrote. `morphF` is the morph
// B-formula when a live morph is active (its ops ride the concatenated slice);
// headless capture never passes one. (Moved from preview.js; the closure
// `morph` read became this explicit parameter — review finding.)
function opIdsInto(set, ops) {
  for (const o of ops || []) {
    if (o?.muted) continue;
    const d = byKey(o.key);
    if (d) set.add(d.id);
  }
}
export function formulaOpSet(f, morphF = null) {
  const s = new Set();
  if (Array.isArray(f?.objects) && f.objects.length) {
    for (const ob of f.objects) opIdsInto(s, ob.ops);
  } else if (f?.hybrid && !f.objects?.length) {
    opIdsInto(s, f.ops);
    opIdsInto(s, f.hybrid.b?.ops);
  } else {
    opIdsInto(s, f.ops);
    if (morphF) opIdsInto(s, morphF.ops); // morph concatenates B's ops
  }
  return [...s].sort((a, b) => a - b);
}

// The march-variant descriptor a frame of (formula, coloring) needs — feature
// flags + op-set — mirroring renderer.js's per-frame activeFeat latch. Live
// state arrives explicitly: `morphF` (active morph B-formula), `df64` (the
// zoom latch — capture passes its OWN capK decision), `blendModeB` (colorBlend
// crossfade target mode). (Extracted from preview.frameFeatures.)
export function frameFeaturesFor(
  f,
  coloring,
  { morphF = null, df64 = false, perturb = false, blendModeB = null } = {},
) {
  const scene = Array.isArray(f?.objects) && f.objects.length > 0;
  const isColorMode = (m) => m > 0.5 && Math.round(m) !== 5; // not surface/curvature
  return {
    numericDE: !scene && effectiveDeOption(f) >= 2.5,
    leaves: scene ? formulaLeafIds(f) : null,
    coloring:
      !scene &&
      (isColorMode(coloring.mode) ||
        (blendModeB != null && isColorMode(blendModeB))),
    scene,
    hybrid: !!f?.hybrid && !scene,
    morph: !!morphF && !scene && !f?.hybrid,
    df64,
    perturb, // perturbation tier latch (PERTURBATION_ZOOM_IMPL.md PR-4)
    ops: formulaOpSet(f, morphF),
  };
}

// Settle one frame's Globals + ops/objects/hybrid/morph uploads through the
// given renderer. `q` is a quality params object (renderpolicy stillQualityParams
// shape + optional itersOverride/iterCap/deScale); `cam` supplies dist (and the
// camera words — unused by fsCapture, which uses CaptureU ortho rays).
// `live` (all optional — headless capture passes none):
//   morph        — active live morph {f, t, swell} (WebGPU flythrough blend)
//   colorBlend   — live coloring-mode crossfade (null on scenes regardless)
//   effIters     — the auto-detail-resolved iters (defaults to f.iters)
//   kStarFor(iters) — the df64 zoom latch for the FLAT path (defaults to 0;
//                  capture overrides offset+kStar separately AFTER settle)
//   onFlatPayload(payload) — flat-path payload tap (preview stores it as
//                  lastFrameState for the #291 standalone export)
// (Body moved verbatim from preview.writeFrame; only the closure reads became
// parameters. Any edit here IS an edit to the live render path — it has no
// second copy.)
export function settleFrame(renderer, f, coloring, q, res, cam, live = {}) {
  const dist = cam.dist;
  const tNear = dist * TNEAR_K,
    // Unbounded scenes reach 4x further before the far cut (renderpolicy);
    // fog scales with tFar, so the horizon fades instead of cliffing.
    // TFAR_MIN floors the window at the object scale — see renderpolicy.js
    // (deep zoom P4 field fix, #364: proportional-only tFar clipped 26% of a
    // zoomed Mandelbulb frame to black).
    tFar =
      Math.max(dist * TFAR_K, TFAR_MIN) *
      (unboundedScene(f) ? TFAR_UNBOUNDED_MUL : 1);
  // One base writeGlobals payload for all four paths (flat is exactly this),
  // so each path below spreads it and overrides ONLY what genuinely differs —
  // a deliberate omission is a visible `key:` override, not a missing line
  // in a 25-line retyped literal.
  const base = {
    res,
    cam,
    // #441 — orthographic half-height (0 = perspective). Rides the options bag
    // exactly like debugView/colorBlend; absent ⇒ 0 ⇒ today's projection.
    orthoH: live.orthoH ?? 0,
    // Auto-detail (§6); q.iterCap bounds it for tiny render targets
    // (thumbnail tiles) where extra iterations are sub-pixel anyway.
    // §S2: q.itersOverride threads the CPU path's exact iters into GPU capture
    // (don't re-derive from this view's detail state — #181 sync bug class).
    iters:
      q.itersOverride ??
      Math.min(q.iterCap ?? Infinity, live.effIters ?? f.iters ?? 12),
    opCount: 0,
    addC: f.addC,
    maxSteps: q.steps,
    bailout: bailoutFor(f),
    eps: q.eps,
    deScale: q.deScale ?? resolveDeScale(0.85, f),
    colA: coloring.colA,
    colB: coloring.colB,
    bg: coloring.bg,
    colorMode: coloring.mode,
    stripeFreq: coloring.stripeFreq, // COLORING S2 — Silk stripe frequency
    iridescence: coloring.iridescence, // COLORING P3 S6 — Glow trap-XYZ modulator
    palettePhase: coloring.palettePhase, // COLORING P3 — palette phase/cycling
    // COLORING P2 — auto-levels. Range computed here (formula in hand) and fed
    // to both GPU tiers via pctl.zw / uSig*. Memoized on formula identity +
    // mode, sampled with the STABLE f.iters (not the zoom-boosted base.iters)
    // so orbiting/zooming doesn't re-sample every frame. Identity {0,1} when
    // off or a non-normalizable (surface/pinwheel) mode.
    ...(() => {
      const { lo, span } = _signalRange(f, coloring, f.iters);
      return { sigLo: lo, sigSpan: span };
    })(),
    tNear,
    tFar,
    deOption: effectiveDeOption(f),
    julia: f.julia,
    juliaC: f.juliaC,
    palette: coloring.palette,
    light: policyShadeLight(q, coloring.light),
    objectCount: 0,
    colorBlend: live.colorBlend ?? null,
    // Debug surface-quality overlay (#370) — dev-only, WebGPU tier; 0 = off
    // (default, byte-identical). Rides all four paths via the base spread.
    debugView: live.debugView ?? 0,
  };
  // Hybrid iteration (IDEAS ①, docs/design/HYBRID_ITERATION.md §3.5) —
  // mutually exclusive with f.objects (§3.8). Objects-first tie-break,
  // matching sanitize.js/cpu.js (Formula Outline Step 3 §4a).
  if (f.hybrid && !f.objects?.length && renderer.writeHybrid) {
    // N-slot hybrid (HYBRID_NSLOT_SPEC.md §2.3) — the ONE upload path (live +
    // capture/splat export). Read EVERY slot through the canonical accessor, then
    // hand the renderer the per-slot op lists (WebGPU concatenates onto the shared
    // buffer; GL codegens one iterStep per slot) plus the schedule counts + per-
    // slot addC for the packed `hyb` word (writeGlobals → packHyb).
    // activeHybridSlots drops MUTED phases (the eye toggle) — the ONE choke point
    // shared with cpuorbit, so the WebGPU concat buffer / GL codegen only ever see
    // the slots that actually run.
    const { slots, counts } = activeHybridSlots(f);
    if (!slots.length) {
      // Every phase muted → render nothing (mirror the legacy both-muted rule):
      // upload zero ops instead of falling back to slot A. In-app engineView
      // already collapses this to empty-ops flat before we get here; this guards a
      // raw all-muted formula reaching capture (e.g. an export skipping engineView).
      renderer.writeGlobals({ ...base, opCount: renderer.writeOps([]) });
      return;
    }
    const slotOps = slots.map((s) => s.ops.filter((o) => !o.muted));
    const addC = slots.map((s) => !!s.addC);
    renderer.writeHybrid(slotOps, counts, addC);
    const family = hybridDeFamily(f);
    renderer.writeGlobals({
      ...base,
      opCount: 0, // unused by mapDE_hybrid — the hyb word carries per-slot counts
      addC: false, // superseded by the per-slot addC packed below
      bailout: family === "ifs" ? BAILOUT_IFS : BAILOUT_ESCAPE,
      deOption: family === "ifs" ? (f.deOption ?? 2) : 0,
      // julia is ORed in separately at the shader level (hybWalk reads G.jc.w) —
      // addC[] stays exactly "each slot's own addC" (§3.8).
      hybrid: {
        opCounts: slotOps.map((o) => o.length),
        counts,
        addC,
      },
    });
    return;
  }
  // Multi-object scene path (additive — only when f.objects is present).
  if (f.objects && renderer.writeScene) {
    const n = renderer.writeScene(f.objects);
    renderer.writeGlobals({
      ...base,
      iters: 0, // per-object iters ride the Obj descriptors, not this word
      addC: false, // ditto — per-object addC/julia flags
      bailout: BAILOUT_IFS, // IFS folds stay bounded; box doesn't iterate
      deScale: sceneDeScale(f.objects),
      // Scene coloring (SCENES.md §Coloring): Glow/Bands render on scenes via
      // orbit-free signals, so the chosen mode passes through.
      colorMode: base.colorMode,
      deOption: 2,
      julia: false,
      juliaC: null, // per-object seeds ride the Obj descriptors
      objectCount: n,
      colorBlend: null, // DELIBERATE: scenes are surface-mode — no color crossfade
    });
    return;
  }
  // Formula morph (VIDEO_EXPORT_DRAWER_V2 tier-2, WebGPU-only, LIVE-only):
  // blend two PLAIN formulas' distance fields. Headless capture never passes
  // live.morph, so this path is unreachable there.
  const morph = live.morph;
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
      ...base, // bailout stays A's own; B carries its own in the morph word
      opCount: opsA.length,
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
    });
    return;
  }
  // Flat single-formula path — the base payload verbatim, plus the op count.
  const oc = renderer.writeOps(activeOps(f));
  // Deep zoom P4 — k* rides the flat path only (eligibility is flat-only).
  // kStar > 0 is BOTH the shader's switchover uniform AND the renderer's
  // variant-selection latch, so the value and the pipeline can never disagree.
  // Clamped to this frame's EFFECTIVE iters. Headless capture leaves the
  // default 0 — it writes its OWN capK via overrideCaptureOffset after settle.
  const kStar = live.kStarFor ? live.kStarFor(base.iters) : 0;
  // Perturbation tier (PR-4): the flat-path pt latch — the closure rebuilds
  // and uploads the reference orbit synchronously before saying yes (D5/D9),
  // so payload.perturb true GUARANTEES a matching orbit is in the buffer.
  const perturb = live.perturbFor ? !!live.perturbFor(base.iters) : false;
  const payload = { ...base, opCount: oc, kStar, perturb };
  live.onFlatPayload?.(payload); // preview stores lastFrameState (#291 export)
  renderer.writeGlobals(payload);
}

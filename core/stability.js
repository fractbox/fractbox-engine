// ─────────────────────────────────────────────────────────────────────────
// Stability predicate — "will this op-stack render something, or blank sky?"
// ─────────────────────────────────────────────────────────────────────────
// The games series (Cairn / Spire / Quarry) all need one shared, GPU-free
// answer to: *does this tower stand?* This module derives that from the operator
// IR alone — no render, no engine call — so daily puzzles can be generated and
// graded in Node (check.mjs) and judged live in the browser without a frame.
//
// HONESTY ABOUT WHAT'S KNOWABLE. Whether an IFS fractal stays bounded is a
// dynamical-systems property; only the live WebGPU render is the true oracle.
// So this predicate layers three signals, each tagged with its confidence:
//
//   1. DE-FAMILY COHERENCE  (EXACT, from wRule).  An op either tracks its own
//      DE bookkeeping (the IFS fold family: W_UNCHANGED/W_MUL_K/W_MUL_SCALE) or
//      flips to escape-time (W_BULB). Mixing a bulb with w-moving IFS folds is
//      the genuine "Mixed DE" failure the flagship's health badge warns about
//      (main.ts updateHealth) — it renders blank/wrong. This tier is certain.
//
//   2. DOCUMENTED PAIRING   (HEURISTIC, quoted from operators.js).  A handful of
//      ops carry an explicit caveat in their own source comment: kaleido,
//      polyAngleFold ("bound the radius with a box/sphere fold"), sphereInv and
//      radialInvert ("unbounded alone — pair with a box/sphere fold or the
//      attractor escapes (blank sky)"). We encode exactly that rule: such an op
//      with no box/sphere-family fold present in the stack → escapes.
//
//   3. SCALE MAGNITUDE      (from invariants.js).  The analytic IFS DE wants
//      |scale| >= 2 or it "may render blank" (the same threshold invariants.js
//      warns on). A soft warning, IFS family only.
//
// The two curated key-sets below are the only place we name operators by string.
// stability.test.mjs guards them against drift: every key must resolve in
// OPERATORS, so a renamed/removed op fails CI rather than silently mis-grading.
// ─────────────────────────────────────────────────────────────────────────

import {
  byKey,
  activeOps,
  effectiveDeOption,
  isEscapeTime,
  isNumericDE,
  W_MUL_K,
  W_MUL_SCALE,
} from "./operators.js";
import { hybridSlots } from "./hybridmodel.js";

// Box/sphere-family folds — the "bound the radius with a box/sphere fold" that
// the pairing-required decorators (below) name. Box-family reflections fold
// coordinates inward; sphere-family ball-folds cap the radial term. Presence of
// any one satisfies the documented pairing requirement.
export const BOUNDING_FOLDS = Object.freeze([
  "boxFold", // id 0  — the Mandelbox box fold
  "boxFoldXYZ", // id 18 — per-axis box fold
  "surfFold", // id 15 — Amazing-Surf box fold (X,Y)
  "sphereFold", // id 1  — Mandelbox radius cap
  "cylinderFold", // id 27 — sphere fold in XY
]);

// Decorators whose own source comment says they are unbounded alone and must be
// paired with a box/sphere fold or they escape to blank sky. Angle folds bound
// DIRECTION only; inversions blow up near their center. Both need a radius cap
// somewhere in the active stack.
export const NEEDS_RADIUS_BOUND = Object.freeze([
  "kaleido", // id 6  — "pair with a box/sphere fold ... or renders blank sky"
  "polyAngleFold", // id 26 — "Bound the radius with a box/sphere fold."
  "sphereInv", // id 13 — "inversion alone is unbounded — pair with a box/sphere fold"
  "radialInvert", // id 28 — "Unbounded alone — pair with a box/sphere fold ... (blank sky)"
]);

const _BOUNDING = new Set(BOUNDING_FOLDS);
const _NEEDS_BOUND = new Set(NEEDS_RADIUS_BOUND);

// The product of |scale| across active Scale ops — the stack's net radial gain
// from the one cleanly-defined expander. 1.0 when there are no scales. (Sphere
// folds / inversions also move the radius, but conditionally and not in closed
// form, so they're intentionally excluded from this number.)
export function scaleProduct(formula) {
  let p = 1.0;
  for (const op of activeOps(formula)) {
    if (byKey(op.key)?.wRule === W_MUL_SCALE)
      p *= Math.abs(op.values?.[0] ?? 1);
  }
  return p;
}

// ── Deep zoom Phase 4 (DEEP_ZOOM_DF64.md §4a-3) — the k* precision laws ────
//
// λ̂ — a conservative LOWER bound on the op stack's per-iteration expansion
// of pixel-to-pixel separation. DIRECTION MATTERS: k* = ⌈log(2⁻²⁴·M)/log λ̂⌉
// DECREASES as λ̂ grows, so an OVERestimate of the expansion under-runs k*
// (f32 noise survives at depth — a correctness bug), while an underestimate
// merely over-runs it (more df64 iterations than needed — a cost, not a bug).
// Hence every factor is the op's GUARANTEED minimum |Jacobian|:
//   scale         |s|                       — exact (constant Jacobian)
//   sphereFold    min(1, (fixedR/minR)²)    — its amplification k is
//                 CONDITIONAL, k ∈ [min(1,(fixedR/minR)²), max(1,(fixedR/minR)²)];
//                 the guaranteed factor is the min: 1 in the normal
//                 fixedR > minR case, (fixedR/minR)² < 1 for inverted params
//                 (a contracting fold — it slows signal growth, so it must
//                 lower λ̂).
//   isometries    1                          — folds/rotates/translate
// minR is guarded to ≥ 1e-3 (a degenerate minR→0 makes the INVERTED-params
// ratio collapse to 0 → λ̂→0 → k* = iters: safe, just all-df64-slow).
// Param defaults come from the op registry so an absent `values` matches
// what the shader would actually run.
export function lambdaHat(formula) {
  let p = 1.0;
  for (const op of activeOps(formula)) {
    const def = byKey(op.key);
    if (!def) continue;
    const v = (i) => op.values?.[i] ?? def.params?.[i]?.default ?? 0;
    if (def.wRule === W_MUL_SCALE) {
      p *= Math.abs(v(0)) || 1;
    } else if (op.key === "sphereFold") {
      const minR = Math.max(Math.abs(v(0)), 1e-3);
      const fixedR = Math.abs(v(1));
      p *= Math.min(1, (fixedR / minR) ** 2);
    }
    // The Tier B₁ inversions (sphereInv/radialInvert) briefly had a λ̂ = 0
    // branch here (PR #422). Dropped with their twins: λ̂ = 0 means k* =
    // iters, and K_STAR_MAX truncated that into measured corruption, while
    // honoring it costs ~3× f32 with no renderpolicy governor — the two
    // halves of the "crashing graphics card" report. Untwinned ops are
    // ineligible (df64Eligible), so no df64 consumer reaches λ̂ for them;
    // re-landing needs the full-loop + cost-governor work tracked in the
    // 4b re-land issue.
  }
  return p;
}

// df64 eligibility (plan D1/D3): FLAT formulas only (no scene objects, no
// hybrid slot-B — those loops don't carry the df64 codegen; morph is a
// RENDERER-time state the caller must gate separately), analytic-IFS DE
// only (deOption 2 — escape-time and numeric-FD DEs are out of 4a), and
// every active op must have a df64 twin (wgslDf presence IS subset
// membership — one source of truth, no list to drift).
export function df64Eligible(formula) {
  if (!formula) return false;
  if (Array.isArray(formula.objects) && formula.objects.length) return false;
  if (formula.hybrid) return false;
  if (effectiveDeOption(formula) !== 2) return false;
  const active = activeOps(formula);
  if (!active.length) return false;
  return active.every((op) => !!byKey(op.key)?.wgslDf);
}

// Perturbation-tier eligibility (PERTURBATION_ZOOM_IMPL.md D7): the same
// structural line df64 draws (flat, deOption-2, non-hybrid), and every
// active op must have BOTH sides of the delta machinery — the wgslPt twin
// in the registry (the GPU kernel) and core/perturb.js's JS stepper (the
// reference orbit + CPU mirror). perturb.test.mjs pins the two memberships
// equal, so neither can drift silently; requiring both here means a drift
// disables the tier loudly instead of rendering a structurally wrong frame.
// A formula eligible for both tiers prefers pt (plan D10 — cheaper, deeper,
// no λ̂ law, and it is the ONLY deep tier for the λ̂ = 0 inversions).
export function ptEligible(formula, ptSupported) {
  if (!formula) return false;
  if (Array.isArray(formula.objects) && formula.objects.length) return false;
  if (formula.hybrid) return false;
  if (effectiveDeOption(formula) !== 2) return false;
  const active = activeOps(formula);
  if (!active.length) return false;
  return active.every(
    (op) => !!byKey(op.key)?.wgslPt && (!ptSupported || ptSupported(op.key)),
  );
}

// K_STAR_MAX — a conservative bound on df64 iterations.
//
// HONEST PROVENANCE: this was added believing df64 iterations had blown a GPU
// watchdog at ×10¹². Direct measurement on Chrome/Metal (headless probe,
// 2026-07-25) FALSIFIED that: at a fixed deep camera the df64 frame costs the
// SAME as the f32 frame — 1.5s/frame at both ×10¹⁰ and ×10¹² with df64 on and
// off alike (the render policy absorbs the load by dropping scale to ~0.68).
// So this constant does NOT fix a measured rendering cost.
//
// It is kept for one narrower reason: `lambdaHat` returns 0 for the Phase 4b
// inversions, and the λ̂ ≤ 1 branch below otherwise returns the FULL iteration
// count, which auto-detail pushes to 51+ at depth. That is an unbounded budget
// set by a different subsystem, and bounding it is cheap insurance. 28 sits
// above the largest k* any verified configuration asks for (Tourbillon needs
// 26 at ×10¹²), so shipped 4a behaviour is unchanged at every depth.
//
// Truncating k* below the precision law's ask is benign ONLY for λ̂ > 1: the
// grown signal dominates the re-injected f32 noise, so the wall merely
// returns far deeper than with no df64 at all. For a λ̂ = 0 op class it is a
// CORRECTNESS bug, measured, not theoretical (Tier B₁ post-mortem, 2026-07-30):
// contracting/parabolic dynamics never wash the noise out, and the render
// collapses to f32-grade quantization plates at full df64 cost — that was the
// entire "corrupt twins" defect of PR #422 (the twins were right; see
// DF64_4B_TIER_B1_HANDOFF.md). Any future λ̂ = 0 class (the degenerate
// minR→0 sphereFold is the surviving example) must run FULL-loop df64 —
// k* = iters, this cap not binding — or stay ineligible.
export const K_STAR_MAX = 28;

// k* — how many leading iterations of the op loop must run in df64 before
// the pixel-scale signal has outgrown freshly-injected f32 rounding noise
// (~2⁻²⁴ absolute at O(1) orbit magnitudes). mag M = orbit-scale / pixel
// world size (recenter.js zoomMag). Signal starts at ~1/M and grows ≥ λ̂×
// per iteration; it clears the noise floor once λ̂^k ≥ 2⁻²⁴·M. Clamped to
// [0, min(iters, K_STAR_MAX)]; λ̂ ≤ 1 never outgrows → the ceiling.
export function kStarFor(mag, lam, iters) {
  // The cost ceiling binds BEFORE the precision law — every return below is
  // already clamped by it, so no branch can hand back an unbounded budget.
  const n = Math.min(Math.max(0, Math.floor(iters ?? 0)), K_STAR_MAX);
  if (!(mag > 0)) return 0;
  // +4 bits of safety margin (field-measured, PR-4 preview): the raw noise
  // metric (half-ULP, 2⁻²⁴) is one bit MORE optimistic than the formation
  // quantum the wall indicator uses (full ULP, 2⁻²³), and artifacts are
  // visible slightly ABOVE the wall — a real report rendered shattered at
  // headroom 0.85 while the unmargined law said k* = 0 (df64 latched but
  // never ran an iteration). With the margin, k* ≥ 1 across the whole
  // engaged band and the two laws can no longer disagree at the boundary.
  const needBits = Math.log2(2 ** -24 * mag) + 4;
  if (needBits <= 0) return 0; // comfortably above the noise floor
  if (!(lam > 1)) return n;
  return Math.min(n, Math.ceil(needBits / Math.log2(lam)));
}

// Loose analytic DE: the IFS estimator r/|w| is only a sound distance bound
// when every expanding op grows |w| fast enough. A scale op with |scale| < 2
// (the codebase caveat — see invariants.js) loosens it: the DE over-estimates
// distance, so the marcher OVERSTEPS the (often thin) surface and the static
// full-quality pass renders blank/banded while the coarse interactive pass —
// looser hit eps — still catches it ("renders only when moving", issue #14).
// The renderer reads this to march such formulas with a tighter step (smaller
// deScale) so the static pass resolves them too.
//
// A SECOND loosener (issue #14, astiglic's Menger + Abs Fold (offset)): an
// `absOffsetFold` with a non-zero offset is a pure reflection (|Jacobian| = 1,
// w untouched — the op math is sound), but `abs(p + off) − off` NET-TRANSLATES
// the orbit by a constant every iteration. With no box/sphere fold to cap the
// radius, that walks the IFS attractor off-centre and thins it out of frame,
// and the analytic r/|w| DE (measured from the origin) stops bottoming out on
// the surface → the static pass renders blank ("adding Abs fold just killed the
// fractal, nothing renders"). A bounding fold folds the drift back in, so a
// Mandelbox + offset fold stays sound and is NOT flagged. We warn only for the
// unbounded case so the health badge tells the user why it went blank.
export function looseDE(formula) {
  if (deFamily(formula) !== "ifs") return false;
  const active = activeOps(formula);
  const hasBoundingFold = active.some((op) => _BOUNDING.has(op.key));
  return active.some((op) => {
    const rule = byKey(op.key)?.wRule;
    if (rule === W_MUL_SCALE && Math.abs(op.values?.[0] ?? 0) < 2) return true;
    // Net-translating offset fold with nothing to bound the drift.
    if (
      op.key === "absOffsetFold" &&
      !hasBoundingFold &&
      (op.values || []).some((v) => Math.abs(v ?? 0) > 1e-6)
    )
      return true;
    return false;
  });
}

// The DE family the stack falls into, from wRule alone (exact):
//   'empty'  — no active ops (nothing to render)
//   'ifs'    — fold family only (analytic r/|w| DE)
//   'escape' — has a bulb op and only isometries otherwise (escape-time DE)
//   'mixed'  — a bulb op AND a w-moving IFS fold (scale/sphere/inversion):
//              the two DE families conflict → blank/wrong render
export function deFamily(formula) {
  const active = activeOps(formula);
  if (active.length === 0) return "empty";
  // Numeric finite-difference DE ignores w entirely, so a numeric op composes
  // with ANY mix of folds/scales/bulbs — it never produces a 'mixed' conflict.
  if (isNumericDE(formula)) return "numeric";
  const bulb = isEscapeTime(formula);
  const movesW = active.some((op) => {
    const r = byKey(op.key)?.wRule;
    return r === W_MUL_K || r === W_MUL_SCALE;
  });
  if (bulb && movesW) return "mixed";
  if (bulb) return "escape";
  return "ifs";
}

// Full verdict. `reasons` carries every signal that fired, each tagged exact vs
// heuristic; `stands` is false iff any 'fail' reason fired. `certain` is true
// only when the verdict rests entirely on the exact DE-family tier (empty /
// mixed) — a clean IFS/escape verdict is a confident *heuristic*, not a proof
// (the GPU is the final oracle). Callers that need a hard answer should treat
// `certain:false && stands:true` as "very likely renders".
export function stability(formula) {
  const reasons = [];
  const family = deFamily(formula);
  const active = activeOps(formula);

  if (family === "empty") {
    reasons.push({
      code: "empty-stack",
      severity: "fail",
      exact: true,
      message: "No active operators — nothing renders.",
    });
    return { stands: false, certain: true, family, scaleProduct: 1, reasons };
  }

  if (family === "mixed") {
    reasons.push({
      code: "mixed-de",
      severity: "fail",
      exact: true,
      message:
        "A Mandelbulb/escape-time power is stacked with w-moving IFS folds — " +
        'the DE families conflict (the engine’s "Mixed DE" case) and it renders blank/wrong.',
    });
    return {
      stands: false,
      certain: true,
      family,
      scaleProduct: scaleProduct(formula),
      reasons,
    };
  }

  // family is 'ifs' or 'escape' — apply the documented heuristics.
  const hasBoundingFold = active.some((op) => _BOUNDING.has(op.key));
  const unpaired = active
    .filter((op) => _NEEDS_BOUND.has(op.key))
    .map((op) => op.key);
  if (unpaired.length && !hasBoundingFold) {
    const names = [...new Set(unpaired)].join(", ");
    reasons.push({
      code: "unbounded-decorator",
      severity: "fail",
      exact: false,
      message:
        `${names} bound direction / invert only and need a box or sphere fold ` +
        "(boxFold, sphereFold, cylinderFold, …) in the stack, or the attractor escapes (blank sky).",
    });
  }

  const sp = scaleProduct(formula);
  if (family === "ifs") {
    // The codebase's own analytic-DE caveat (invariants.js): |scale| < 2 → loose
    // DE (see looseDE above; the renderer compensates with a tighter step).
    if (looseDE(formula)) {
      for (const op of active) {
        if (
          byKey(op.key)?.wRule === W_MUL_SCALE &&
          Math.abs(op.values?.[0] ?? 0) < 2
        ) {
          reasons.push({
            code: "loose-de",
            severity: "warn",
            exact: false,
            message: `scale ${op.values?.[0]} < 2 — analytic IFS DE may render blank or banded.`,
          });
        }
        // Net-translating offset fold with no box/sphere fold to cap the drift
        // (issue #14) — walks the attractor off-centre → static pass blanks.
        if (
          op.key === "absOffsetFold" &&
          !hasBoundingFold &&
          (op.values || []).some((v) => Math.abs(v ?? 0) > 1e-6)
        ) {
          reasons.push({
            code: "loose-de",
            severity: "warn",
            exact: false,
            message:
              "Abs Fold (offset) shifts the fold off-centre with no box/sphere fold to " +
              "bound it — the attractor drifts out of frame and may render blank. Add a " +
              "box or sphere fold, or set the offset back toward 0.",
          });
        }
      }
    }
    // An expander with nothing to fold it back tends to escape.
    if (sp > 1 && !hasBoundingFold) {
      reasons.push({
        code: "no-cap",
        severity: "warn",
        exact: false,
        message:
          "Scale expands the stack but no box/sphere fold is present to fold it back in.",
      });
    }
  }

  const stands = !reasons.some((r) => r.severity === "fail");
  return { stands, certain: false, family, scaleProduct: sp, reasons };
}

// Convenience boolean: does the tower stand? (See stability() for the full
// verdict with reasons and confidence.)
export function stands(formula) {
  return stability(formula).stands;
}

// Hybrid iteration (IDEAS ①, docs/design/HYBRID_ITERATION.md §3.3-3.4) — the DE
// family of a hybrid formula is deFamily over the UNION of both slots' ops, not
// each slot classified separately: the question is whether the combined stream
// of ops that actually runs (either slot can execute in any given outer
// iteration) agrees on one w bookkeeping, not whether each slot is internally
// consistent alone. This is deliberately MORE PERMISSIVE than a naive per-slot
// check — a W_UNCHANGED-only slot (a pure fold: menger/box/rotate/…) touches
// neither w's multiplicative nor additive bookkeeping, so it combines safely
// with EITHER an IFS slot or an escape slot; a per-slot table would wrongly
// flag that case. sanitizeHybrid (sanitize.js) rejects 'mixed'; updateHealth
// (app/src/main.ts) shows the Mixed-DE badge only when this returns 'mixed' —
// the two must never disagree (§3.9 review finding).
export function hybridDeFamily(formula) {
  if (!formula?.hybrid) return deFamily(formula);
  // UNION over EVERY slot's ops (via the one accessor — legacy 2-slot or N-slot).
  const { slots } = hybridSlots(formula);
  return deFamily({ ops: slots.flatMap((s) => s.ops) });
}

// Union rule, mirroring hybridDeFamily: loose if EITHER slot's own ops would be
// loose on their own (the renderer's step-tightening is a per-scene/per-formula
// global, so the tighter requirement wins — same pattern as CSG's sceneDeScale).
export function hybridLooseDE(formula) {
  if (!formula?.hybrid) return looseDE(formula);
  // Per-slot union via the one accessor: loose if ANY slot's ops are loose alone.
  const { slots } = hybridSlots(formula);
  if (slots.some((s) => looseDE({ ops: s.ops }))) return true;
  // Escape-family hybrid with a FOLD-ONLY slot (no W_BULB): in the iterations
  // that slot runs, the analytic escape derivative dr gets no growth term
  // (no +n·rⁿ⁻¹·dr), so the shared escape DE 0.5·ln(r)·r/|w| OVER-estimates the
  // safe step and the marcher oversteps — the §3.3 "ship-row" hazard the family
  // table alone doesn't surface. A WebGPU hole-readback sweep (2026-07-01,
  // docs/design/HYBRID_ITERATION.md §5) measured a fold×bulb hybrid losing ~51%
  // of its surface at the worst camera angle at the default deScale 0.5, dropping
  // to ~13% (in-band with the safe controls) at the loose 0.3 tier. Tighten to
  // that proven loose step for this class — a bulb×bulb escape hybrid (both slots
  // carry the +1 term) stays on the fast 0.5 path.
  if (hybridDeFamily(formula) === "escape") {
    if (slots.some((s) => !isEscapeTime({ ops: s.ops }))) return true;
  }
  return false;
}

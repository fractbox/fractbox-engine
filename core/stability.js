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
  isEscapeTime,
  W_MUL_K,
  W_MUL_SCALE,
} from './operators.js';

// Box/sphere-family folds — the "bound the radius with a box/sphere fold" that
// the pairing-required decorators (below) name. Box-family reflections fold
// coordinates inward; sphere-family ball-folds cap the radial term. Presence of
// any one satisfies the documented pairing requirement.
export const BOUNDING_FOLDS = Object.freeze([
  'boxFold', // id 0  — the Mandelbox box fold
  'boxFoldXYZ', // id 18 — per-axis box fold
  'surfFold', // id 15 — Amazing-Surf box fold (X,Y)
  'sphereFold', // id 1  — Mandelbox radius cap
  'cylinderFold', // id 27 — sphere fold in XY
]);

// Decorators whose own source comment says they are unbounded alone and must be
// paired with a box/sphere fold or they escape to blank sky. Angle folds bound
// DIRECTION only; inversions blow up near their center. Both need a radius cap
// somewhere in the active stack.
export const NEEDS_RADIUS_BOUND = Object.freeze([
  'kaleido', // id 6  — "pair with a box/sphere fold ... or renders blank sky"
  'polyAngleFold', // id 26 — "Bound the radius with a box/sphere fold."
  'sphereInv', // id 13 — "inversion alone is unbounded — pair with a box/sphere fold"
  'radialInvert', // id 28 — "Unbounded alone — pair with a box/sphere fold ... (blank sky)"
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
    if (byKey(op.key)?.wRule === W_MUL_SCALE) p *= Math.abs(op.values?.[0] ?? 1);
  }
  return p;
}

// Loose analytic DE: the IFS estimator r/|w| is only a sound distance bound
// when every expanding op grows |w| fast enough. A scale op with |scale| < 2
// (the codebase caveat — see invariants.js) loosens it: the DE over-estimates
// distance, so the marcher OVERSTEPS the (often thin) surface and the static
// full-quality pass renders blank/banded while the coarse interactive pass —
// looser hit eps — still catches it ("renders only when moving", issue #14).
// The renderer reads this to march such formulas with a tighter step (smaller
// deScale) so the static pass resolves them too.
export function looseDE(formula) {
  if (deFamily(formula) !== 'ifs') return false;
  return activeOps(formula).some(
    (op) => byKey(op.key)?.wRule === W_MUL_SCALE && Math.abs(op.values?.[0] ?? 0) < 2,
  );
}

// The DE family the stack falls into, from wRule alone (exact):
//   'empty'  — no active ops (nothing to render)
//   'ifs'    — fold family only (analytic r/|w| DE)
//   'escape' — has a bulb op and only isometries otherwise (escape-time DE)
//   'mixed'  — a bulb op AND a w-moving IFS fold (scale/sphere/inversion):
//              the two DE families conflict → blank/wrong render
export function deFamily(formula) {
  const active = activeOps(formula);
  if (active.length === 0) return 'empty';
  const bulb = isEscapeTime(formula);
  const movesW = active.some((op) => {
    const r = byKey(op.key)?.wRule;
    return r === W_MUL_K || r === W_MUL_SCALE;
  });
  if (bulb && movesW) return 'mixed';
  if (bulb) return 'escape';
  return 'ifs';
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

  if (family === 'empty') {
    reasons.push({
      code: 'empty-stack',
      severity: 'fail',
      exact: true,
      message: 'No active operators — nothing renders.',
    });
    return { stands: false, certain: true, family, scaleProduct: 1, reasons };
  }

  if (family === 'mixed') {
    reasons.push({
      code: 'mixed-de',
      severity: 'fail',
      exact: true,
      message:
        'A Mandelbulb/escape-time power is stacked with w-moving IFS folds — ' +
        'the DE families conflict (the engine’s "Mixed DE" case) and it renders blank/wrong.',
    });
    return { stands: false, certain: true, family, scaleProduct: scaleProduct(formula), reasons };
  }

  // family is 'ifs' or 'escape' — apply the documented heuristics.
  const hasBoundingFold = active.some((op) => _BOUNDING.has(op.key));
  const unpaired = active
    .filter((op) => _NEEDS_BOUND.has(op.key))
    .map((op) => op.key);
  if (unpaired.length && !hasBoundingFold) {
    const names = [...new Set(unpaired)].join(', ');
    reasons.push({
      code: 'unbounded-decorator',
      severity: 'fail',
      exact: false,
      message:
        `${names} bound direction / invert only and need a box or sphere fold ` +
        '(boxFold, sphereFold, cylinderFold, …) in the stack, or the attractor escapes (blank sky).',
    });
  }

  const sp = scaleProduct(formula);
  if (family === 'ifs') {
    // The codebase's own analytic-DE caveat (invariants.js): |scale| < 2 → loose
    // DE (see looseDE above; the renderer compensates with a tighter step).
    if (looseDE(formula)) {
      for (const op of active) {
        if (byKey(op.key)?.wRule === W_MUL_SCALE && Math.abs(op.values?.[0] ?? 0) < 2) {
          reasons.push({
            code: 'loose-de',
            severity: 'warn',
            exact: false,
            message: `scale ${op.values?.[0]} < 2 — analytic IFS DE may render blank or banded.`,
          });
        }
      }
    }
    // An expander with nothing to fold it back tends to escape.
    if (sp > 1 && !hasBoundingFold) {
      reasons.push({
        code: 'no-cap',
        severity: 'warn',
        exact: false,
        message: 'Scale expands the stack but no box/sphere fold is present to fold it back in.',
      });
    }
  }

  const stands = !reasons.some((r) => r.severity === 'fail');
  return { stands, certain: false, family, scaleProduct: sp, reasons };
}

// Convenience boolean: does the tower stand? (See stability() for the full
// verdict with reasons and confidence.)
export function stands(formula) {
  return stability(formula).stands;
}

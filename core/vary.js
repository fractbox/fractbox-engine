// Metadata-driven formula variation + numeric soundness gate.
//
// Every operator already declares its parameter ranges (operators.js params:
// {min, max, step}), and evaluate.js can score any op-stack on the CPU
// (measure() → wobble, where ~1 means toppling/BLANK). Together they replace
// the old hand-maintained "which knobs are safe to touch" tables: jitter any
// param inside its declared range, then let the numeric oracle reject the
// rare degenerate roll. Shared by Surprise (core/random.js wide jitter) and
// the app's Remix (narrow jitter for ops its tuned table doesn't know).

import { OPERATORS } from "./operators.js";
import { clone } from "./oplist.js";
import { measure } from "./evaluate.js";
import { deFamily, scaleProduct } from "./stability.js";
import { hybridSlots } from "./hybridmodel.js";

const PARAM_META = new Map(OPERATORS.map((op) => [op.key, op.params]));

// Overlap-sensitive movers: their params place the figure relative to itself
// (tile pitch, mirror seams, absolute offsets), so a full-range roll usually
// tears the shape apart. They still get jittered — just on a short leash
// (touchyScale), with the soundness gate as the backstop.
export const TOUCHY = new Set([
  "translate",
  "modFold",
  "tentFold",
  "brickFold",
  "absOffsetFold",
  "zFold",
]);

/**
 * Nudge ONE op's values inside their declared ranges (the per-op unit of
 * jitterParams) — exported so the app's Remix can metadata-jitter exactly the
 * ops its hand-tuned table doesn't know, without re-rolling the ones it does.
 */
export function jitterOpValues(op, spread, touchyScale, rng) {
  const meta = PARAM_META.get(op.key);
  if (!meta || !Array.isArray(op.values)) return;
  const s = TOUCHY.has(op.key) ? spread * touchyScale : spread;
  for (let i = 0; i < op.values.length && i < meta.length; i++) {
    const m = meta[i];
    const span = m.max - m.min;
    if (!(span > 0)) continue;
    let v = op.values[i] + (rng() * 2 - 1) * s * span;
    v = Math.min(m.max, Math.max(m.min, v));
    // Respect declared granularity: integer-stepped params (kaleido sectors…)
    // snap to their step; continuous ones just get tidied for share links.
    v = m.step >= 1 ? Math.round(v / m.step) * m.step : +v.toFixed(3);
    op.values[i] = Math.min(m.max, Math.max(m.min, v));
  }
}

/**
 * Return a copy of `formula` with every op's params nudged inside their
 * declared ranges. `spread` is the max nudge as a fraction of each param's
 * full span (0.15 = a firm remix, 0.5 = a wild new take). EVERY hybrid slot's
 * ops ride along (A + B/C/…, via the one accessor — Remix jitters the whole
 * formula or none, never a half-mutated N-slot result, §2.8); scene objects are
 * left to the caller (their params live in transforms, not op values). Pass
 * `rng` for reproducible tests.
 */
export function jitterParams(formula, opts = {}) {
  const { spread = 0.15, touchyScale = 0.3, rng = Math.random } = opts;
  const f = clone(formula);
  // The accessor's op arrays reference the clone, so nudging them in place
  // mutates `f` directly (flat → just slot A; hybrid → every slot).
  for (const slot of hybridSlots(f).slots)
    for (const op of slot.ops) jitterOpValues(op, spread, touchyScale, rng);
  return f;
}

/**
 * Numeric blank-detector: does this formula converge to a visible figure?
 * Two accept signals per probe region, either suffices:
 *   - wobble ≤ maxWobble — measure()'s 0 (solid) → 1 (toppling/blank) score;
 *   - ESCAPE-TIME FAMILIES ONLY: escaped ≤ escapedCap — SOME samples
 *     converged. Thin escape-time figures (Newton Mix converges ~2% of points
 *     yet renders fine) legitimately run wobble ~0.9+, but a truly blank
 *     escape-time roll escapes at 100.0% in EVERY region. The same signal is
 *     MEANINGLESS for the ifs family, whose degenerate mode is the opposite:
 *     a contraction roll (e.g. scale 0.01 on a Sierpinski) collapses the
 *     lattice to an invisible point and NOTHING escapes — escaped=0 with
 *     wobble=1, which this signal would have waved through (it did once: a
 *     shared "Wandering Tetra" that rendered empty). IFS lives on wobble alone.
 * The empty/mixed-DE families are structural, region-free rejects (note their
 * short-circuit reports escaped=0, so that check must come first). The region
 * ladder matters at both ends: escape-time mass sits inside |x| ≲ 1.2, while
 * box-family attractors are an order of magnitude larger than the unit-ish
 * default probe. Unsupported inputs (scenes with exotic ops, future keys)
 * return true — the gate only vetoes what the oracle can actually see.
 */
export function isSound(formula, opts = {}) {
  const {
    regions = [1.2, 2.5, 8, 20],
    maxWobble = 0.85,
    escapedCap = 0.995,
    samples = 384,
    minExpansion = 1.6,
  } = opts;
  // Structural guard for the ifs family: without real per-iteration expansion
  // the lattice degenerates BEFORE the sampler can tell — |scale| ≪ 1 collapses
  // to an invisible point, and |scale| ≈ 1 balloons into a space-filling solid
  // that measure() honestly reports as "solid" (wobble 0.3) while the camera
  // sits INSIDE it (a shared "Drifting Octa" at scale 1.13 rendered blank
  // exactly this way). Every shipped IFS preset runs |scaleProduct| ≥ 2.0;
  // 1.6 keeps a wide margin on both sides.
  if (
    deFamily(formula) === "ifs" &&
    Math.abs(scaleProduct(formula)) < minExpansion
  )
    return false;
  let sawVerdict = false;
  for (const region of regions) {
    const m = measure(formula, { region, samples });
    if (!m || m.wobble === null || m.wobble === undefined) continue; // unsupported here
    if (m.family === "empty" || m.family === "mixed") return false; // structural, region-free
    sawVerdict = true;
    // Evidence rule: a region may only accept if SOMETHING in it converged.
    // measure() reports a neutral low wobble when a probe region contains no
    // mass at all (escaped = 1.000 exactly) — a shared "Wandering Form" (a
    // menger roll whose scale/translate interlock broke, everything escapes)
    // read wobble 0.30 in the empty inner region and shipped blank.
    // But escaped === 1 alone is NOT absence: a bounded-CYCLIC orbit (the
    // "Smooth Box" recipe — smoothBoxFold re-clamps every iteration while the
    // ball fold + scale push |p| past the bail radius between clamps) reads
    // "all escaped" in every region yet holds a real marchable surface.
    // coverage can't discriminate (the broken lattice fakes 0.22 from sharp-
    // probe artifacts a march never reaches) — but measure()'s surfaceLean
    // sphere-march can: real surface ⇒ hits > 0 (Smooth Box 147, broken
    // lattice 0). Presence = bounded orbits OR march-verified surface.
    const present =
      (typeof m.escaped === "number" ? m.escaped < 1 : true) ||
      (m.hits ?? 0) > 0;
    if (!present) continue;
    if (m.wobble <= maxWobble) return true;
    const escapeTime = m.family === "escape" || m.family === "numeric";
    if (escapeTime && typeof m.escaped === "number" && m.escaped <= escapedCap)
      return true;
  }
  return !sawVerdict; // no region could judge it → don't block
}

/**
 * Generate-and-test: draw candidates from `make()` until one passes the
 * soundness gate (≤ `attempts` draws), else return `fallback()`. The tiny
 * loop every caller of jitterParams was about to hand-roll.
 */
export function soundCandidate(make, fallback, opts = {}) {
  const { attempts = 5, ...soundOpts } = opts;
  for (let i = 0; i < attempts; i++) {
    const f = make(i);
    if (isSound(f, soundOpts)) return f;
  }
  return fallback();
}

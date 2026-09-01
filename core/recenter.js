// Deep zoom §3.1 — the residual/offset split, shared by renderer.js (WGSL) and
// renderer_gl.js (GLSL) so the two tiers can never disagree on how it's computed.
// Pure arithmetic, no GPU — kept in its own module so it's unit-testable without
// a WebGPU/WebGL context (see recenter.test.ts).
//
// O is the recenter offset (the pan target, kept in f64 by the caller — these are
// plain JS numbers, which already are f64). roRel = eye − O is the small residual
// sent to the GPU as camPos; O is sent separately as the offset uniform. The
// shader reconstructs p_world = p_rel + offset once per DE entry (§3.2) — an
// identity in exact arithmetic, so roRel + O === eye always.
//
// CSG scenes are out of scope for v1 (§14 — objDist's per-object placement isn't
// recentered), so `isScene` forces O to (0,0,0): the shader's reconstruction is
// then an exact no-op and scenes render byte-identically to before recenter.
export function computeRecenter(eye, target, isScene) {
  const O = isScene ? [0, 0, 0] : target || [0, 0, 0];
  const roRel = [eye[0] - O[0], eye[1] - O[1], eye[2] - O[2]];
  return { O, roRel };
}

// ── Deep zoom Phase 4 (DEEP_ZOOM_DF64.md) — the hi/lo offset split ─────────
// O lives in f64 JS-side; the GPU gets it as an f32 PAIR: hi = fround(O),
// lo = fround(O − hi) (exact for these magnitudes — lo is the part the f32
// truncation used to throw away). The df64 marcher reconstructs
// p0 = (p_rel + hi) ⊕ lo to ~49-bit precision. ONE function so writeGlobals,
// overrideCaptureOffset, and any future capture path can never disagree on
// the split (an internally inconsistent pair — hi from one O, lo from
// another — is worse than no lo at all).
export function splitHiLo(v) {
  const hi = [Math.fround(v[0]), Math.fround(v[1]), Math.fround(v[2])];
  return {
    hi,
    lo: [
      Math.fround(v[0] - hi[0]),
      Math.fround(v[1] - hi[1]),
      Math.fround(v[2] - hi[2]),
    ],
  };
}

// ── Deep zoom Phase 4 (DEEP_ZOOM_DF64.md) — the precision-wall law ─────────
// One source of truth for "how close is this camera to the precision wall",
// shared by the app HUD (main.ts headroomFor becomes a wrapper, plan PR-4)
// and the df64 engagement decision in preview.js (plan PR-3). Pure
// arithmetic, unit-tested in df64.test.mjs.
//
// The wall model (§8, verified against the shipped indicator): the offset
// reconstruction quantizes world positions at |O|·quantum, and a pixel stops
// resolving when its world-space footprint (dist · pixelAngle) sinks to that
// quantum. quantum = 2⁻²³ for f32 (mantissa spacing), 2⁻⁴⁹ for df64.
export const F32_QUANTUM = 2 ** -23;
export const DF64_QUANTUM = 2 ** -49;

// ── Perturbation tier (PERTURBATION_ZOOM_IMPL.md PR-4) — the pt wall law ──
// Unlike the quantum laws above, the perturbation noise floor has NO |T|
// term: the delta kernel's precision is relative to the residual itself, so
// the wall is an ABSOLUTE pixel-footprint floor — WGSL flushes f32
// subnormals, and δ·δ intermediates need headroom above ~1.2e-38, giving a
// practical floor around dist·pixelAngle ≈ 1e-35 (~×10³⁰; the P0 ladder is
// clean at 1e-32 and the assessment doc §7 carries the margin math).
export const PT_FLOOR = 1e-35;

// headroom for a perturbation-engaged camera: pixel footprint over the
// absolute floor. Mirrors zoomHeadroom's shape so HUD/brake consumers can
// swap laws by eligibility without new plumbing.
export function ptHeadroom(args) {
  const { signal } = zoomScales(args);
  return signal / PT_FLOOR;
}

// The two scales everything derives from: `signal` = one pixel's world-space
// footprint at the target; `orbit` = the orbit-magnitude scale the rounding
// noise is relative to (|O|, floored at the O(1) attractor scale).
export function zoomScales({ target, dist, fovDeg = 42, heightPx }) {
  const t = target ?? [0, 0, 0];
  const orbit = Math.max(Math.hypot(t[0] ?? 0, t[1] ?? 0, t[2] ?? 0), 1);
  const fov = ((fovDeg ?? 42) * Math.PI) / 180;
  const pixelAngle = (2 * Math.tan(fov / 2)) / Math.max(heightPx ?? 1, 1);
  return { signal: dist * pixelAngle, orbit };
}

// headroom = pixel footprint / wall quantum: >1 above the wall, ≤1 at it.
// With the default quantum this reproduces the shipped f32 indicator law
// (app/src/main.ts headroomFor).
export function zoomHeadroom(args, quantum = F32_QUANTUM) {
  const { signal, orbit } = zoomScales(args);
  return signal / (orbit * quantum);
}

// M — the magnification the k* law consumes (stability.js kStarFor):
// orbit scale over pixel footprint. Identity: zoomHeadroom(a, q) ===
// 1 / (q · zoomMag(a)) — pinned in df64.test.mjs.
export function zoomMag(args) {
  const { signal, orbit } = zoomScales(args);
  return orbit / signal;
}

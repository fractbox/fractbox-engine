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

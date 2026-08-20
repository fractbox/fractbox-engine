// Pure WebGL2 health classification — the verdict layer for the GL fallback
// tier (renderer_gl.js). Dependency-free and side-effect-free so it can be
// unit-tested (core/gldiag.test.mjs) with no GL context, and so the live
// renderer, preview.js and the diag layer share ONE definition of "the GL tier
// is dead". The renderer only COLLECTS raw signals (compile/link status,
// context-lost, the gl.getError() codes sampled on the first few draws) into a
// plain object; this function turns them into a verdict.
//
// Why this exists: the WebGL2 tier fails DARK. A reserved `gl_` local once
// blacked out every draw for 9 days in prod with zero telemetry (#206); the
// 2026 iOS-15 field dump showed the pump "succeeding" (frames advancing,
// skip:"drew", impossibly fast ms) while the canvas stayed pure black. Only
// HARD signals mark the tier dead here — never slowness (the frame governor owns
// that).

// WebGL/WebGL2 error enum → short name, for compact diag text (the dump is
// user-pasted). Anything not listed renders as its hex code.
export const GL_ERROR_NAMES = Object.freeze({
  0x0500: "INVALID_ENUM",
  0x0501: "INVALID_VALUE",
  0x0502: "INVALID_OPERATION",
  0x0505: "OUT_OF_MEMORY",
  0x0506: "INVALID_FRAMEBUFFER_OPERATION",
  0x9242: "CONTEXT_LOST_WEBGL",
});

export function glErrorName(code) {
  return GL_ERROR_NAMES[code] || "0x" + Number(code).toString(16);
}

// classifyGlHealth(signals) → { dead: boolean, reason: string | null }
//
// signals (all optional; absent = "not observed yet", i.e. healthy so far):
//   contextCreationError : the canvas fired webglcontextcreationerror
//   contextLost          : a webglcontextlost event fired
//   compileFailed        : a shader compile returned COMPILE_STATUS false
//   linkFailed           : a program link returned LINK_STATUS false
//   drawErrors           : array of gl.getError() codes sampled on the first draws
//
// Verdict is worst-first; the reason is a short, diag-friendly string. A zero
// (NO_ERROR) inside drawErrors is ignored — only a nonzero code is a hard fault.
export function classifyGlHealth(signals = {}) {
  const s = signals || {};
  if (s.contextCreationError)
    return { dead: true, reason: "webglcontextcreationerror" };
  if (s.contextLost) return { dead: true, reason: "webgl2 context lost" };
  if (s.compileFailed) return { dead: true, reason: "gl-compile-fail" };
  if (s.linkFailed) return { dead: true, reason: "gl-link-fail" };
  const errs = Array.isArray(s.drawErrors) ? s.drawErrors.filter((c) => c) : [];
  if (errs.length)
    return { dead: true, reason: "gl-draw-error " + glErrorName(errs[0]) };
  return { dead: false, reason: null };
}

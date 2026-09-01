// IMGTEX (#631) — user-image environment map + triplanar surface texture:
// the PURE helpers. The GPU work lives in shader.js (codegen, gated on the
// `envMap`/`surfTex` build flags) and renderer.js (texture upload + bind
// groups); everything here is plain math shared by the app's decode path and
// the tests — no GPU, no DOM beyond what a caller passes in.
//
// Clean-room note: this feature is idea-level parity with Catoptron3D (no
// license file there — see issue #631). Every formula below is derived
// independently: equirectangular mapping and triplanar projection are
// textbook techniques, and the constants are ours.

// ── Upload size budget ───────────────────────────────────────────────────────
// A phone camera photo is routinely 4000×3000+ — ~48 MB as RGBA8, on devices
// whose render pipeline is already cost-governed (#476; the 2026-08-01 iPad
// death). Decode-time downscale caps the texture at a dimension that keeps
// the worst case bounded: 2048² RGBA8 = 16 MB on desktop, 1024² = 4 MB on
// mobile-class devices (the caller picks via isMobileClass — renderpolicy.js).
export const IMG_MAX_DIM = 2048;
export const IMG_MAX_DIM_MOBILE = 1024;

// Fit (w, h) inside a `cap`-sized square, preserving aspect, never upscaling.
// Returns integer dims ≥ 1. Mirrors what the app feeds createImageBitmap's
// resizeWidth/resizeHeight so the policy is testable without a browser.
export function fitImageDims(w, h, cap = IMG_MAX_DIM) {
  const W = Math.max(1, Math.floor(w || 1));
  const H = Math.max(1, Math.floor(h || 1));
  const s = Math.min(1, cap / Math.max(W, H));
  return {
    w: Math.max(1, Math.round(W * s)),
    h: Math.max(1, Math.round(H * s)),
  };
}

// ── Triplanar blend weights ──────────────────────────────────────────────────
// JS mirror of the WGSL in shader.js (triplanarImage): per-axis weights are
// the normal components raised to the 4th power ((n²)² — two squarings, no
// pow), normalized to sum 1. The sharpening keeps each face committed to one
// projection so the three taps only cross-fade in narrow edge bands. Pinned
// against the emitted WGSL by envmap.test.mjs.
export function triplanarWeights(n) {
  const x2 = n[0] * n[0],
    y2 = n[1] * n[1],
    z2 = n[2] * n[2];
  const wx = x2 * x2,
    wy = y2 * y2,
    wz = z2 * z2;
  const s = wx + wy + wz || 1;
  return [wx / s, wy / s, wz / s];
}

// ── Equirectangular direction → UV ───────────────────────────────────────────
// JS mirror of the WGSL (envImage): u = azimuth around the world +Z axis
// (the frame's up — #160) mapped to a full turn, plus the rotation slider
// (0..1 = one turn; the sampler wraps, so u is returned un-fract'd modulo 1
// here only for testability); v = polar angle from the +Z pole, clamped a
// hair off the poles so a repeat-addressed sampler can't wrap the top row
// onto the bottom.
export function equirectUV(rd, rot = 0) {
  const u = Math.atan2(rd[1], rd[0]) / (2 * Math.PI) + 0.5 + rot;
  const v = Math.acos(Math.max(-1, Math.min(1, rd[2]))) / Math.PI;
  return [u - Math.floor(u), Math.max(0.001, Math.min(0.999, v))];
}

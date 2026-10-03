// OKLab color space (Björn Ottosson, 2020) + N-stop palette sampling.
// Raw ES module, zero deps — the CORE-invariant JS reference for coloring.
//
// Why OKLab: interpolating a gradient in sRGB (or linear RGB) desaturates the
// midpoint — blue→yellow passes through gray. OKLab is perceptually even and
// the straight-line path between two colors stays vivid, so N-stop palettes
// blend cleanly (COLORING.md). It is the SINGLE SOURCE of the conversion math:
// core/cpu.js samples palettes through here directly, core/frameparams.js
// converts stops sRGB→OKLab here before upload, and the WGSL `oklabToSrgb`
// (core/shader.js) + GLSL mirror decode with the SAME matrices — the round
// trip is pinned by core/oklab.test.mjs.
//
// All sRGB values are [0,1] triples. OKLab L is ~[0,1] for displayable colors;
// a,b are small signed (~±0.4).

// sRGB → linear (exact piecewise EOTF — matches shader s2l / post l2s inverse).
function s2l(c) {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}
// linear → sRGB (exact piecewise OETF).
function l2s(c) {
  const cc = c < 0 ? 0 : c > 1 ? 1 : c;
  return cc <= 0.0031308 ? cc * 12.92 : 1.055 * Math.pow(cc, 1 / 2.4) - 0.055;
}

// sRGB [0,1]³ → OKLab.
export function srgbToOklab([r, g, b]) {
  const lr = s2l(r),
    lg = s2l(g),
    lb = s2l(b);
  const l = 0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb;
  const m = 0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb;
  const s = 0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb;
  const l_ = Math.cbrt(l),
    m_ = Math.cbrt(m),
    s_ = Math.cbrt(s);
  return [
    0.2104542553 * l_ + 0.793617785 * m_ - 0.0040720468 * s_,
    1.9779984951 * l_ - 2.428592205 * m_ + 0.4505937099 * s_,
    0.0259040371 * l_ + 0.7827717662 * m_ - 0.808675766 * s_,
  ];
}

// OKLab → sRGB [0,1]³ (clamped to the displayable cube). Inverse of
// srgbToOklab; the WGSL/GLSL `oklabToSrgb` use these same coefficients.
export function oklabToSrgb([L, a, b]) {
  const l_ = L + 0.3963377774 * a + 0.2158037573 * b;
  const m_ = L - 0.1055613458 * a - 0.0638541728 * b;
  const s_ = L - 0.0894841775 * a - 1.291485548 * b;
  const l = l_ * l_ * l_,
    m = m_ * m_ * m_,
    s = s_ * s_ * s_;
  return [
    l2s(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s),
    l2s(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s),
    l2s(-0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s),
  ];
}

// Sample an N-stop palette at t ∈ [0,1]. `stops` is [{c: sRGB[3], p: pos}], any
// order (sorted here); `cyclic` wraps the last stop to the first across the
// seam. Returns sRGB. Mirrors the WGSL `albedoStops` walk exactly (core/cpu.js
// and the GPU tiers must agree). Fewer than 2 stops → the single/absent color.
export function sampleStops(stops, t, cyclic = false) {
  if (!stops || stops.length === 0) return [0, 0, 0];
  if (stops.length === 1) return stops[0].c.slice();
  const s = stops
    .map((x) => ({ lab: srgbToOklab(x.c), p: x.p }))
    .sort((a, b) => a.p - b.p);
  const n = s.length;
  const tt = t < 0 ? 0 : t > 1 ? 1 : t;
  const first = s[0],
    last = s[n - 1];
  const lerp = (A, B, f) => [
    A[0] + (B[0] - A[0]) * f,
    A[1] + (B[1] - A[1]) * f,
    A[2] + (B[2] - A[2]) * f,
  ];
  const clamp01 = (f) => (f < 0 ? 0 : f > 1 ? 1 : f);
  if (tt <= first.p) {
    if (cyclic) {
      const pl = last.p - 1;
      // max()-guard the span: a full-span cyclic palette (stops at exactly 0
      // and 1) makes first.p−pl == 0 → 0/0 = NaN without it. Mirrors the WGSL/
      // GLSL albedoStops (which already guard with max(…, 1e-6)).
      return oklabToSrgb(lerp(last.lab, first.lab, clamp01((tt - pl) / Math.max(first.p - pl, 1e-6))));
    }
    return oklabToSrgb(first.lab);
  }
  if (tt >= last.p) {
    if (cyclic) {
      const ph = first.p + 1;
      return oklabToSrgb(lerp(last.lab, first.lab, clamp01((tt - last.p) / Math.max(ph - last.p, 1e-6))));
    }
    return oklabToSrgb(last.lab);
  }
  for (let i = 0; i < n - 1; i++) {
    if (tt >= s[i].p && tt <= s[i + 1].p) {
      const span = s[i + 1].p - s[i].p || 1e-6;
      return oklabToSrgb(lerp(s[i].lab, s[i + 1].lab, (tt - s[i].p) / span));
    }
  }
  return oklabToSrgb(last.lab); // unreachable
}

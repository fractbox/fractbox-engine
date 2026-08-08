#version 300 es
precision highp float;
out vec4 fragColor;
// COLORING P0 — smooth escape fraction (S1 bands, #239 D6). Mirror of
// shader.js smoothEscFrac: adds a fractional offset to the integer escape
// count so "bands" mode stops stair-stepping. HEURISTIC — an op-list has no
// single power, so log2 is a chosen constant, not log(power). Guarded rBail>1
// && rEsc>1 (a bailout<1 flips the log ratio to NaN, which clamp won't fix).
float smoothEscFrac(float rEsc, float bailSq) {
  float rBail = sqrt(bailSq);
  if (rBail <= 1.0 || rEsc <= 1.0) return 0.0;
  return clamp(1.0 - log2(log(rEsc) / log(rBail)), 0.0, 1.0);
}

uniform vec2 uRes;
uniform float uFov;
uniform vec3 uCamPos, uCamFwd, uCamRight, uCamUp;
// #441 ORTHOGRAPHIC half-height; 0 = perspective. A NEW uniform, not a mirror
// of the WGSL edit: the WGSL tier hides this in camFwd.w, but GLSL declares
// plain vec3s with no padding component to reuse.
uniform float uOrthoH;
uniform vec3 uOffset; // deep zoom §3.1 — (0,0,0) for scenes (§14, exact no-op)
uniform int uIters, uMaxSteps, uColorMode;
uniform float uBailout, uEps, uDeScale, uDeOption;
uniform vec3 uColA, uColB, uBg, uJc;
uniform float uJulia;
uniform vec3 uPalA, uPalB, uPalC, uPalD;
uniform float uPalOn;
uniform float uPalCount, uPalCyclic;
uniform float uStripeFreq;
uniform float uSigLo, uSigSpan;
uniform float uIridescence, uPalettePhase;
uniform vec3 uLightDir;
uniform float uAmbient, uRim, uGloss, uIntensity;
uniform vec3 uKeyC, uFillDir, uFillC, uBackDir, uBackC; // P1 light rig
uniform float uMetallic, uShadowK, uShadowOn, uAoStr, uFill, uBack; // P1
uniform float uSky, uSunGlow, uGround, uIbl, uFogAmt, uInScatter; // P3 (bloom = WebGPU-only)
uniform float uExposure; // whole-frame EV (mirrors the WGSL post word)
uniform float uNear, uFar; // deep zoom §5 — was hardcoded 0.02 / 80.0

layout(std140) uniform Bulk {
  int uHyb[2];
  int uAddGate[2];
  vec4 uPalStops[8];
  float uP[192];
};

float g_wq; // desktop 4D scratch the shared op bodies write to; unused here

void iterStep0(inout vec3 pos, inout float w, int i) {
  g_wq = 1.0;

    // box fold (reflection: |Jacobian| = 1, w untouched)
    pos.x = abs(pos.x + uP[0]) - abs(pos.x - uP[0]) - pos.x;
    pos.y = abs(pos.y + uP[0]) - abs(pos.y - uP[0]) - pos.y;
    pos.z = abs(pos.z + uP[0]) - abs(pos.z - uP[0]) - pos.z;
}
void iterStep1(inout vec3 pos, inout float w, int i) {
  g_wq = 1.0;

    // conformal scale (the expanding map → |scale| onto w)
    pos  *= uP[1];
    w    *= abs(uP[1]);
    g_wq *= abs(uP[1]);
}
// N-slot schedule walk — mirrors shader.js hybWalk / cpuorbit.hybridSlotAt (same
// period/phase math), baked as a GLSL chain (GLSL can't index functions).
void hybStep(inout vec3 pos, inout float w, inout vec3 c, int i) {
  int period = uHyb[0] + uHyb[1];
  int ph = period > 0 ? (i % period) : 0; // GLSL ES 3.00 (#version 300 es) has int %
  int acc = 0;
  acc += uHyb[0]; if (ph < acc) { iterStep0(pos, w, i); if (uAddGate[0] == 1) pos += c; return; }
  acc += uHyb[1]; if (ph < acc) { iterStep1(pos, w, i); if (uAddGate[1] == 1) pos += c; return; }
}

float mapDE(vec3 p_rel) {
  vec3 p0 = p_rel + uOffset;
  vec3 pos = p0; float w = 1.0;
  vec3 c = (uJulia > 0.5) ? uJc : p0;
  for (int i = 0; i < 64; i++) {
    if (i >= uIters) break;
    hybStep(pos, w, c, i);
    if (dot(pos, pos) > uBailout) break;
  }
  float r = length(pos);
  if (uDeOption < 1.0) return 0.5 * log(max(r, 1e-9)) * r / max(abs(w), 1e-9);
  return r / max(abs(w), 1e-9);
}

float orbitTrap(vec3 p_rel) {
  vec3 p0 = p_rel + uOffset;
  vec3 pos = p0; float w = 1.0; vec3 c = (uJulia > 0.5) ? uJc : p0; float tr = 1e9;
  for (int i = 0; i < 64; i++) {
    if (i >= uIters) break;
    hybStep(pos, w, c, i);
    tr = min(tr, length(pos));
    if (dot(pos, pos) > uBailout) break;
  }
  return tr;
}

// COLORING R S8 — IFS address: sign-octant of the FINAL orbit point → 8 colors.
float orbitAddressGL(vec3 p_rel) {
  vec3 p0 = p_rel + uOffset;
  vec3 pos = p0; float w = 1.0; vec3 c = (uJulia > 0.5) ? uJc : p0;
  for (int i = 0; i < 64; i++) {
    if (i >= uIters) break;
    hybStep(pos, w, c, i);
    if (dot(pos, pos) > uBailout) break;
  }
  float oct = float(int(pos.x > 0.0) + 2 * int(pos.y > 0.0) + 4 * int(pos.z > 0.0));
  return (oct + 0.5) / 8.0;
}

// COLORING S2 — Silk (stripe average). Mirror of shader.js orbitSilk: mean over
// the orbit of 0.5 + 0.5·sin(k·θ), θ = atan(pos.y, pos.x), k = uStripeFreq.
float orbitSilk(vec3 p_rel) {
  vec3 p0 = p_rel + uOffset;
  vec3 pos = p0; float w = 1.0; vec3 c = (uJulia > 0.5) ? uJc : p0;
  float k = max(uStripeFreq, 1.0); float acc = 0.0; float cnt = 0.0;
  for (int i = 0; i < 64; i++) {
    if (i >= uIters) break;
    hybStep(pos, w, c, i);
    acc += 0.5 + 0.5 * sin(k * atan(pos.y, pos.x));
    cnt += 1.0;
    if (dot(pos, pos) > uBailout) break;
  }
  return acc / max(cnt, 1.0);
}

// COLORING S3 — Pinwheel (trap angle). Mirror of shader.js orbitPin: the angle
// atan(pos.y, pos.x) at the orbit's closest approach to the origin, → [0,1).
float orbitPin(vec3 p_rel) {
  vec3 p0 = p_rel + uOffset;
  vec3 pos = p0; float w = 1.0; vec3 c = (uJulia > 0.5) ? uJc : p0;
  float tr = 1e9; float ang = 0.0;
  for (int i = 0; i < 64; i++) {
    if (i >= uIters) break;
    hybStep(pos, w, c, i);
    float r2 = dot(pos, pos);
    if (r2 < tr) { tr = r2; ang = atan(pos.y, pos.x); }
    if (r2 > uBailout) break;
  }
  return fract(ang * 0.15915494 + 0.5); // 1/(2π); [-π,π] → [0,1)
}

// COLORING P3 — iridescence (S6). Per-axis orbit minima → axis asymmetry
// (min|x| − min|z|). Mirror of shader.js orbitIrid (hybrid step).
float orbitIrid(vec3 p_rel) {
  vec3 p0 = p_rel + uOffset;
  vec3 pos = p0; float w = 1.0; vec3 c = (uJulia > 0.5) ? uJc : p0;
  vec3 mn = vec3(1e9);
  for (int i = 0; i < 64; i++) {
    if (i >= uIters) break;
    hybStep(pos, w, c, i);
    mn = min(mn, abs(pos));
    if (dot(pos, pos) > uBailout) break;
  }
  return (mn.x - mn.z) / (mn.x + mn.z + 1e-6); // scale-invariant asymmetry
}

float escapeIter(vec3 p_rel) {
  vec3 p0 = p_rel + uOffset;
  vec3 pos = p0; float w = 1.0; vec3 c = (uJulia > 0.5) ? uJc : p0; int esc = uIters; float rEsc = 0.0;
  for (int i = 0; i < 64; i++) {
    if (i >= uIters) break;
    hybStep(pos, w, c, i);
    if (dot(pos, pos) > uBailout) { esc = i; rEsc = length(pos); break; }
  }
  return (float(esc) + smoothEscFrac(rEsc, uBailout)) / float(max(uIters, 1)); // S1 smooth bands
}

// e scales with hit distance t (deep zoom §3.4/§5) — a fixed epsilon straddles
// unrelated geometry once the near/far range is no longer a fixed [0.02, 80].
// p_rel is the RESIDUAL hit point (§3.4) — perturbing it here, before each
// mapDE call reconstructs, is what keeps the 4-tap difference precision-correct.
vec3 calcNormal(vec3 p_rel, float t) {
  vec2 e = vec2(1.0, -1.0) * clamp(t * 3e-5, 1e-6, 6e-4);
  vec3 g = e.xyy * mapDE(p_rel + e.xyy) + e.yyx * mapDE(p_rel + e.yyx) +
           e.yxy * mapDE(p_rel + e.yxy) + e.xxx * mapDE(p_rel + e.xxx);
  // The four tetrahedron offsets sum to 0, so a locally-flat DE field (all taps
  // equal) gives an exactly-zero gradient and normalize() is NaN → the lighting
  // math collapses to pure black. Fall back to a stable normal. Mirror of the
  // WGSL guard in shader.js. length > 1e-20 also rejects a NaN/Inf gradient.
  float L = length(g);
  if (L > 1e-20) { return g / L; }
  float pl = length(p_rel);
  return pl > 1e-6 ? p_rel / pl : vec3(0.0, 0.0, 1.0);
}

// COLORING S4 — Curvature tint. Discrete Laplacian of the DE field via the same
// tetrahedron taps as calcNormal (gradient cancels; residual is second-order).
// Mirror of shader.js curvatureAt. ⚠ eps is ~100× the normal's: curvature is
// O(e²) and sinks below f32 precision at the normal eps — the coarser probe is
// load-bearing. The 0.15 gain is a chosen heuristic; tanh bounds to [0,1].
float curvatureAt(vec3 p_rel, float t) {
  float ce = clamp(t * 3e-3, 1e-4, 1e-2);
  vec2 e = vec2(1.0, -1.0) * ce;
  float lap = mapDE(p_rel + e.xyy) + mapDE(p_rel + e.yyx)
            + mapDE(p_rel + e.yxy) + mapDE(p_rel + e.xxx) - 4.0 * mapDE(p_rel);
  return clamp(0.5 + 0.5 * tanh(lap / (ce * ce) * 0.15), 0.0, 1.0);
}

// COLORING P2 — auto-levels: remap the raw signal by its per-formula range
// (uSigLo, uSigSpan). Identity (0,1) when off/cyclic. Mirror of shader.js normSig.
float normSig(float x) {
  if (uSigSpan <= 1e-4) return x;
  return clamp((x - uSigLo) / uSigSpan, 0.0, 1.0);
}

// COLORING P3 — palette phase (mirror of shader.js albedoFor's rotation). Exact
// identity at phase 0; fract wraps for cyclic palettes → seamless color flow.
float palPhase(float t) {
  return uPalettePhase != 0.0 ? fract(t + uPalettePhase) : t;
}

// sRGB → linear, exact piecewise (P0 — paired with the exact l2s encode below;
// mixing curves would break the picker round trip near black, issue #6).
vec3 s2l(vec3 c) {
  vec3 cc = max(c, vec3(0.0));
  vec3 lo = cc / 12.92;
  vec3 hi = pow((cc + 0.055) / 1.055, vec3(2.4));
  return mix(hi, lo, vec3(lessThanEqual(cc, vec3(0.04045))));
}
// linear → sRGB, exact piecewise (P0).
vec3 l2s(vec3 c) {
  vec3 cc = clamp(c, vec3(0.0), vec3(1.0));
  vec3 lo = cc * 12.92;
  vec3 hi = 1.055 * pow(cc, vec3(1.0 / 2.4)) - 0.055;
  return mix(hi, lo, vec3(lessThanEqual(cc, vec3(0.0031308))));
}
// COLORING P0 — OKLab → sRGB (mirror of shader.js oklabToSrgb / oklab.js). The
// N-stop palette blends in OKLab (perceptually even), then decodes here.
vec3 oklabToSrgb(vec3 lab) {
  float l_ = lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z;
  float m_ = lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z;
  float s_ = lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z;
  vec3 lms = vec3(l_ * l_ * l_, m_ * m_ * m_, s_ * s_ * s_);
  vec3 lin = vec3(
     4.0767416621 * lms.x - 3.3077115913 * lms.y + 0.2309699292 * lms.z,
    -1.2684380046 * lms.x + 2.6097574011 * lms.y - 0.3413193965 * lms.z,
    -0.0041960863 * lms.x - 0.7034186147 * lms.y + 1.7076147010 * lms.z);
  return l2s(lin); // reuse the exact piecewise encode + [0,1] clamp
}
// Walk the ≤8 sorted stops, lerp the bracketing pair in OKLab, decode once.
// Constant loop bound (ES 3.0-safe) + dynamic break; cyclic wraps last→first.
vec3 albedoStopsGL(float t) {
  int n = int(uPalCount);
  bool cyclic = uPalCyclic > 0.5;
  float tt = clamp(t, 0.0, 1.0);
  vec4 first = uPalStops[0];
  vec4 last = uPalStops[n - 1];
  if (tt <= first.w) {
    if (cyclic) {
      float pl = last.w - 1.0;
      return oklabToSrgb(mix(last.xyz, first.xyz, clamp((tt - pl) / max(first.w - pl, 1e-6), 0.0, 1.0)));
    }
    return oklabToSrgb(first.xyz);
  }
  if (tt >= last.w) {
    if (cyclic) {
      float ph = first.w + 1.0;
      return oklabToSrgb(mix(last.xyz, first.xyz, clamp((tt - last.w) / max(ph - last.w, 1e-6), 0.0, 1.0)));
    }
    return oklabToSrgb(last.xyz);
  }
  for (int i = 0; i < 8; i++) {
    if (i >= n - 1) break;
    vec4 a = uPalStops[i];
    vec4 b = uPalStops[i + 1];
    if (tt >= a.w && tt <= b.w) {
      return oklabToSrgb(mix(a.xyz, b.xyz, (tt - a.w) / max(b.w - a.w, 1e-6)));
    }
  }
  return oklabToSrgb(last.xyz);
}
// Filmic soft-shoulder (P0) — identity below the shoulder (picker round trip by
// construction), tanh rolloff above. Mirrors shader.js buildPostWGSL tone1/3.
float tone1(float x) {
  const float S = 0.75;
  return x <= S ? x : S + (1.0 - S) * tanh((x - S) / (1.0 - S));
}
vec3 tone3(vec3 c) { return vec3(tone1(c.r), tone1(c.g), tone1(c.b)); }
// Interleaved-gradient noise dither (P0) — mirrors buildPostWGSL ign().
float ign(vec2 px) {
  return fract(52.9829189 * fract(dot(px, vec2(0.06711056, 0.00583715))));
}
// ── P1 shading helpers (mirror shader.js softShadow/calcAO/ggxSpec) ─────────
float softShadow(vec3 p0, vec3 n, vec3 ldir, float tHit, float k) {
  // Normal lift-off — mirrors shader.js (light-dir offsets self-shadow at grazing angles).
  vec3 org = p0 + n * max(uEps * tHit * 12.0, 2e-3);
  float sh = 1.0;
  float s = max(uEps * tHit * 8.0, 1e-3);
  for (int i = 0; i < 32; i++) {
    float d = mapDE(org + ldir * s) * uDeScale;
    sh = min(sh, k * d / s);
    s += clamp(d, 0.01, 0.5);
    if (sh < 0.005 || s > 12.0) break;
  }
  return clamp(sh, 0.0, 1.0);
}
float calcAO(vec3 p0, vec3 n, float tHit) {
  float r = clamp(tHit * 0.3, 0.1, 1.5);
  float occ = 0.0, sca = 1.0;
  for (int i = 1; i <= 5; i++) {
    float h = r * (0.01 + 0.12 * float(i) / 5.0);
    occ += (h - mapDE(p0 + n * h)) * sca;
    sca *= 0.85;
  }
  return clamp(1.0 - 3.0 * uAoStr * (occ / r), 0.0, 1.0);
}
// P3 directional environment — mirrors shader.js envColor().
// Up is world +Z (the #160 / 31e2253 lighting frame): horizon in the XY plane.
vec3 envColor(vec3 rd) {
  float tg = clamp(rd.z * 0.5 + 0.5, 0.0, 1.0);
  vec3 sky = s2l(mix(uBg * 0.35, uBg, tg));
  sky = mix(sky * (1.0 - uGround), sky, smoothstep(-0.35, 0.12, rd.z));
  float sunAmt = pow(max(dot(rd, normalize(uLightDir)), 0.0), 24.0);
  // Neutral (white) sun glow, not tinted by the light albedo (#160 item 2) — mirrors WGSL.
  return sky + vec3(1.0) * (sunAmt * uSunGlow);
}
vec3 ggxSpec(vec3 n, vec3 v, vec3 l, float rough, vec3 f0) {
  vec3 h = normalize(v + l);
  float ndl = max(dot(n, l), 0.0);
  float ndv = max(dot(n, v), 1e-3);
  float ndh = max(dot(n, h), 0.0);
  float a = rough * rough;
  float a2 = a * a;
  float dd = ndh * ndh * (a2 - 1.0) + 1.0;
  float D = a2 / (3.14159265 * dd * dd);
  float kk = (rough + 1.0) * (rough + 1.0) / 8.0;
  float gv = ndv / (ndv * (1.0 - kk) + kk);
  float gLit = ndl / (ndl * (1.0 - kk) + kk);
  vec3 F = f0 + (vec3(1.0) - f0) * pow(1.0 - max(dot(h, v), 0.0), 5.0);
  return D * gv * gLit * F / max(4.0 * ndv, 1e-3);
}

vec3 palLookupGL(float t) {
  float pt = palPhase(t);
  if (uPalCount > 1.5) return albedoStopsGL(pt);
  if (uPalOn > 0.5) return clamp(uPalA + uPalB * cos(6.2831853 * (uPalC * pt + uPalD)), vec3(0.0), vec3(1.0));
  return mix(uColA, uColB, pt);
}
vec3 orbitPainterGL(vec3 p_rel) {
  vec3 p0 = p_rel + uOffset;
  vec3 pos = p0; float w = 1.0; vec3 c = (uJulia > 0.5) ? uJc : p0;
  vec3 col = vec3(0.0); float wsum = 0.0;
  for (int i = 0; i < 64; i++) {
    if (i >= uIters) break;
    hybStep(pos, w, c, i);
    float d = length(pos);
    float ti = fract(d * 0.35 + float(i) * 0.03); // per-iteration palette coordinate
    float wt = exp(-1.5 * d);                      // trap-proximity weight
    col += wt * palLookupGL(ti);
    wsum += wt;
    if (dot(pos, pos) > uBailout) break;
  }
  return col / max(wsum, 1e-4);
}

void main() {
  vec2 uv = gl_FragCoord.xy / uRes;
  vec2 ndc = uv * 2.0 - 1.0;
  float aspect = uRes.x / uRes.y;
  float tanF = tan(0.5 * uFov);
  // #441 — perspective fans directions from one origin; orthographic spreads
  // the ORIGIN and holds the direction. Mirrors core/shader.js exactly; this
  // tier is NEVER compiled in CI (#206), so verify with ?renderer=webgl2.
  vec3 rd = uOrthoH > 0.0
    ? normalize(uCamFwd)
    : normalize(uCamFwd + (ndc.x * aspect * tanF) * uCamRight + (ndc.y * tanF) * uCamUp);
  // Deep zoom §3.1 — uCamPos is the RESIDUAL ro_rel, not the absolute eye; every
  // p below is p_rel, reconstructed once per call inside mapDE/calcNormal/
  // orbitTrap/escapeIter/sceneTintGL (§3.2). uOffset=(0,0,0) for scenes (§14).
  vec3 ro = uCamPos;
  // World-space displacement, so it adds straight to the RESIDUAL (§3.1).
  if (uOrthoH > 0.0) {
    ro = ro + ((ndc.x * aspect * uOrthoH) * uCamRight) + ((ndc.y * uOrthoH) * uCamUp);
  }
  vec3 bg = s2l(mix(uBg * 0.35, uBg, clamp(uv.y, 0.0, 1.0)));

  float t = uNear; float tPrev = t; int steps = 0; bool hit = false;
  float lastD = 1e9; bool exhausted = true;
  for (int i = 0; i < 512; i++) {
    if (i >= uMaxSteps) break;
    steps = i;
    float d = mapDE(ro + rd * t) * uDeScale;
    lastD = d;
    if (d < uEps * t) { hit = true; exhausted = false; break; }
    tPrev = t;
    t += d;
    if (t > uFar) { exhausted = false; break; }
  }
  // Hit refinement (mirrors shader.js): the loop lands up to a full march step
  // past the eps·t crossing, and that per-ray overshoot terraces grazing
  // silhouettes into staircases. Bisect [tPrev, t] 8× → overshoot ÷256,
  // sub-pixel at any zoom, ≤8 extra mapDE calls per hit ray. tPrev==t guards
  // the first-sample hit (no bracket); the exhausted softA pseudo-hit below
  // sets its hit flag after this and never refines.
  if (hit && t > tPrev) {
    float lo = tPrev; float hi = t;
    for (int r = 0; r < 8; r++) {
      float mid = 0.5 * (lo + hi);
      if (mapDE(ro + rd * mid) * uDeScale < uEps * mid) { hi = mid; } else { lo = mid; }
    }
    t = hi;
  }
  // Budget exhausted while still hugging geometry: shade at the last position
  // instead of dropping to sky, CONFIDENCE-WEIGHTED by how close the ray got
  // (mirrors shader.js — silhouette-grazing rays must fade to sky, not paint
  // phantom surface that bulges the horizon around objects).
  float softA = 1.0;
  if (!hit && exhausted && lastD < uEps * t * 8.0) {
    hit = true;
    softA = 1.0 - clamp(lastD / (uEps * t * 8.0), 0.0, 1.0);
    softA *= softA;
  }
  vec3 bgOut = uSky > 0.0 ? mix(bg, envColor(rd), clamp(uSky, 0.0, 1.0)) : bg; // P3 sky blend
  // Encode the miss like the hit path (kills the double-dark bg quirk — the
  // displayed background now matches the picked theme colors; mirrors WGSL).
  // Sky rays get the full-fog-column sun in-scatter (mirrors the WGSL miss
  // path) — else fogged geometry meets raw sky at a hard seam.
  if (!hit) {
    vec3 skyOut = bgOut;
    if (uFogAmt > 0.0) {
      float scatM = pow(max(dot(rd, normalize(uLightDir)), 0.0), 8.0) * uInScatter;
      skyOut += s2l(uKeyC) * scatM * (1.0 - exp(-3.0 * uFogAmt));
    } else if (uFogAmt < 0.0) { // log-depth mode: sky spans all decades to uFar
      float t0M = max(length(uCamPos), 1e-12);
      float decM = log2(1.0 + uFar / t0M) * 0.30103;
      float scatM = pow(max(dot(rd, normalize(uLightDir)), 0.0), 8.0) * uInScatter;
      skyOut += s2l(uKeyC) * scatM * (1.0 - exp(-2.3 * (-uFogAmt) * decM));
    }
    fragColor = vec4(l2s(tone3(skyOut * exp2(uExposure))), 1.0); return;
  }

  vec3 p = ro + rd * t;
  vec3 nrm = calcNormal(p, t);
  float stepAO = 1.0 - float(steps) / float(uMaxSteps); // cavity dust (pre-P1 AO)
  float cav = 0.7 + 0.3 * stepAO;
  vec3 lightDir = normalize(uLightDir);
  float diff = max(dot(nrm, lightDir), 0.0);
  float rim = pow(1.0 - max(dot(nrm, -rd), 0.0), 2.0);
  float amb = uAmbient;
  float occ = uAoStr > 0.0 ? calcAO(p, nrm, t) : 1.0;               // P1 real AO
  float sh = uShadowOn > 0.5 ? 0.15 + 0.85 * softShadow(p, nrm, lightDir, t, uShadowK) : 1.0; // P1 (15% floor)

  vec3 albedo;
  if (uColorMode == 6) {
    albedo = orbitPainterGL(p); // COLORING R S7 — per-iteration palette blend (direct color)
  } else {
    float mixT;
    if (uColorMode == 7) mixT = orbitAddressGL(p); // COLORING R S8 — IFS sign-octant
    else if (uColorMode == 5) mixT = curvatureAt(p, t);
    else if (uColorMode == 4) mixT = orbitPin(p);
    else if (uColorMode == 3) mixT = orbitSilk(p);
    else if (uColorMode == 2) mixT = escapeIter(p);
    else if (uColorMode == 1) mixT = clamp(orbitTrap(p) / 1.5, 0.0, 1.0);
    else mixT = 0.5 + 0.5 * nrm.z;
    mixT = normSig(mixT); // COLORING P2 auto-levels (identity for surface/pinwheel)
    // COLORING P3 iridescence (S6) — Glow-only per-pixel palette-phase shift.
    if (uColorMode == 1 && uIridescence > 0.001) mixT = fract(mixT + uIridescence * orbitIrid(p));
    mixT = palPhase(mixT); // COLORING P3 — palette phase/cycling
    if (uPalCount > 1.5)
      albedo = albedoStopsGL(mixT); // N-stop OKLab palette (count ≥ 2)
    else if (uPalOn > 0.5)
      albedo = clamp(uPalA + uPalB * cos(6.2831853 * (uPalC * mixT + uPalD)), vec3(0.0), vec3(1.0));
    else albedo = mix(uColA, uColB, mixT);
  }
  albedo = s2l(albedo); // sRGB→linear (issue #6)

  // P1 GGX composition (mirrors shader.js — see that file for rationale).
  float rough = clamp(1.0 - uGloss, 0.05, 1.0);
  float metal = clamp(uMetallic, 0.0, 1.0);
  vec3 f0 = mix(vec3(0.04), albedo, metal);
  float kd = 1.0 - metal;
  float specAmt = clamp(max(uGloss, metal), 0.0, 1.0);
  vec3 keyC = s2l(uKeyC);
  float ambT = amb * (0.3 + 0.7 * occ); // AO eases ambient to a 30% floor (mirrors WGSL)
  vec3 ambC = vec3(ambT); // P3 IBL: hue-only env tint, luminance preserved
  if (uIbl > 0.0) {
    vec3 e = envColor(nrm);
    float elum = max(dot(e, vec3(0.2126, 0.7152, 0.0722)), 1e-3);
    ambC = ambT * mix(vec3(1.0), e / elum, clamp(uIbl, 0.0, 1.0));
  }
  vec3 col = albedo * kd * (ambC + 1.25 * (1.0 - amb) * diff * sh * keyC) * cav; // key energy 1.25 (mirrors WGSL)
  col += ggxSpec(nrm, -rd, lightDir, rough, f0) * keyC * (1.25 * sh * occ * specAmt);
  if (uFill > 0.0) {
    vec3 fd = normalize(uFillDir);
    vec3 fc = s2l(uFillC) * uFill;
    col += albedo * kd * max(dot(nrm, fd), 0.0) * fc * cav;
    col += ggxSpec(nrm, -rd, fd, rough, f0) * fc * (occ * specAmt);
  }
  if (uBack > 0.0) {
    vec3 bd = normalize(uBackDir);
    vec3 bc = s2l(uBackC) * uBack;
    col += albedo * kd * max(dot(nrm, bd), 0.0) * bc * cav;
    col += ggxSpec(nrm, -rd, bd, rough, f0) * bc * (occ * specAmt);
  }
  col += s2l(uColB) * (rim * uRim * occ);
  if (uSky > 0.0 && specAmt > 0.0) { // P3 env reflections (mirrors WGSL)
    vec3 er = mix(envColor(reflect(rd, nrm)), s2l(uBg) * 0.6, rough);
    vec3 fenv = f0 + (vec3(1.0) - f0) * pow(1.0 - max(dot(nrm, -rd), 0.0), 5.0);
    col += er * fenv * (uSky * specAmt * occ);
  }
  col *= uIntensity; // object intensity (kept object-only, matches WebGPU)
  // Far fade — mirrors WGSL: the last stretch before uFar blends fully into
  // the sky so the finite render boundary never shows on unbounded fields.
  float farF = max(smoothstep(0.62 * uFar, 0.95 * uFar, t), 1.0 - softA);
  if (uFogAmt > 0.0) { // P3 fog + sun in-scatter (mirrors WGSL); 0 = legacy fade
    float ff = max(1.0 - exp(-3.0 * uFogAmt * t / uFar), farF);
    float scat = pow(max(dot(rd, normalize(uLightDir)), 0.0), 8.0) * uInScatter;
    col = mix(col, bgOut + s2l(uKeyC) * scat, ff);
  } else if (uFogAmt < 0.0) { // deep-zoom log-depth fog (mirrors WGSL; |x| = slider)
    float t0 = max(length(uCamPos), 1e-12);
    float dec = log2(1.0 + t / t0) * 0.30103;
    float ff = max(1.0 - exp(-2.3 * (-uFogAmt) * dec), farF);
    float scat = pow(max(dot(rd, normalize(uLightDir)), 0.0), 8.0) * uInScatter;
    col = mix(col, bgOut + s2l(uKeyC) * scat, ff);
  } else {
    col = mix(col, bgOut, max(clamp(t / uFar, 0.0, 1.0) * 0.6, farF));
  }
  // P0 tail (single-pass — WebGL2 has no HDR intermediate, but the math is the
  // same as the WebGPU post pass): tone map → exact sRGB encode → dither.
  col = l2s(tone3(max(col, vec3(0.0)) * exp2(uExposure)));
  col += vec3((ign(gl_FragCoord.xy) - 0.5) / 255.0);
  fragColor = vec4(col, 1.0);
}
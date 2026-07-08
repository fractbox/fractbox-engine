// WebGL2 (GLSL ES 3.00) backend — the SECOND live renderer, for the large
// population that has WebGL2 but not WebGPU. Same raymarch + DE + coloring as
// the WGSL shader (shader.js), so it's near-full-fidelity, not a degraded view.
//
// Op math reuse: instead of a third hand-written copy, the per-iteration body is
// generated from the SAME operators.js `glsl()` emitter the desktop export uses,
// with params bound to a `uP[]` uniform array. So editing param VALUES is pure
// uniform upload; only adding/removing/reordering ops (a STRUCTURE change)
// regenerates + recompiles the fragment shader.

import { byKey, W_BULB_NUMERIC } from "./operators.js";
// Capacity caps live once in limits.js; re-export so renderer_gl's existing
// import site keeps working while the values are single-sourced.
import { MAX_PARAMS, MAX_OBJECTS } from "./limits.js";
export { MAX_PARAMS, MAX_OBJECTS };

// Fullscreen triangle from gl_VertexID (no vertex buffer needed in WebGL2).
export const VERT_GL = `#version 300 es
void main() {
  vec2 p = vec2(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0);
  gl_Position = vec4(p, 0.0, 1.0);
}`;

// One iteration of the op stack, params bound to uP[] (angles → radians()).
// Returns { body, paramCount }; paramCount drives how much of uP we upload.
// `slotBase` offsets the uP[] indices so several objects can SHARE one uP[]
// array (each object's params concatenated at a distinct base — see
// sceneParamLayout). slotBase defaults to 0 → the single-object path unchanged.
export function iterBodyGL(ops, slotBase = 0) {
  let body = "";
  let slot = 0;
  for (const op of ops) {
    const def = byKey(op.key);
    if (!def) continue;
    const v = def.params.map((pm) => {
      const ref = `uP[${slotBase + slot++}]`;
      return pm.type === "angle" ? `radians(${ref})` : ref;
    });
    body += def.glsl(v) + "\n";
  }
  return { body, paramCount: slot };
}

// Per-op mute inside a scene object — the ACTIVE slice every scene consumer
// (codegen, layout, param upload, structural signature) must share, mirroring
// the CPU tier (cpu.js makeSceneDE activeOps) and WebGPU (renderer.js
// writeScene). 3-emitter mirror discipline, guarded by core/scenemute.test.mjs.
export function activeSceneOps(o) {
  return (o.ops || []).filter((op) => !op.muted);
}

// CSG scene — uP[] layout across objects. IFS objects (objType 0) each consume
// `paramCount` slots (their ACTIVE ops' params, concatenated); primitives
// consume none. buildSceneFragGL AND renderer_gl.writeScene both call this so
// the codegen'd uP[] indices and the uploaded values stay in lockstep.
export function sceneParamLayout(objects) {
  let cursor = 0;
  return objects.map((o) => {
    const objType = Number(o.objType) & 0xf;
    if (objType !== 0) return { objType, slotBase: cursor, paramCount: 0 };
    const { paramCount } = iterBodyGL(activeSceneOps(o));
    const slotBase = cursor;
    cursor += paramCount;
    return { objType, slotBase, paramCount };
  });
}

// Build the full fragment shader for a given op-list. The op bodies from
// operators.js are already GLSL (the desktop dialect), so they drop straight in.
// hybridB, when present ({ops:[...]}) — IDEAS ①, docs/design/HYBRID_ITERATION.md
// §3.5 — makes this hybrid-aware: emits iterStepA/iterStepB (params share one
// uP[] via the CSG sceneParamLayout slot-base trick) and mapDE/orbitTrap/
// escapeIter alternate them by a {uHybA,uHybB} SCHEDULE UNIFORM (not baked into
// the source — only editing either op-stack's STRUCTURE rebuilds the program,
// same rule as the plain single-object path). Everything else (uniforms,
// calcNormal, main()) is shared verbatim — hybrid has no effect on them.
export function buildFragGL(ops, hybridB) {
  const { body, paramCount } = iterBodyGL(ops);
  const coreUniforms = hybridB
    ? `uniform int uHybA, uHybB; // schedule counts — Phase 1a/1b (IDEAS ①)
uniform int uAddGateA, uAddGateB; // per-slot addC (||julia folded in on upload)`
    : `uniform int uAddGate;`;
  let core;
  if (hybridB) {
    const { body: bodyB } = iterBodyGL(hybridB.ops, paramCount);
    core = `
void iterStepA(inout vec3 pos, inout float w, int i) {
  g_wq = 1.0;
${body}
}
void iterStepB(inout vec3 pos, inout float w, int i) {
  g_wq = 1.0;
${bodyB}
}
// Schedule branch — mirrors shader.js mapDE_hybrid EXACTLY (same period/phase
// math), just as a runtime int loop instead of a WGSL bit-packed word.
void hybStep(inout vec3 pos, inout float w, inout vec3 c, int i) {
  int period = uHybA + uHybB;
  int ph = period > 0 ? (i % period) : 0; // GLSL ES 3.00 (#version 300 es) has int %
  if (ph < uHybA) { iterStepA(pos, w, i); if (uAddGateA == 1) pos += c; }
  else            { iterStepB(pos, w, i); if (uAddGateB == 1) pos += c; }
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

float escapeIter(vec3 p_rel) {
  vec3 p0 = p_rel + uOffset;
  vec3 pos = p0; float w = 1.0; vec3 c = (uJulia > 0.5) ? uJc : p0; int esc = uIters;
  for (int i = 0; i < 64; i++) {
    if (i >= uIters) break;
    hybStep(pos, w, c, i);
    if (dot(pos, pos) > uBailout) { esc = i; break; }
  }
  return float(esc) / float(max(uIters, 1));
}`;
  } else {
    // Emit the numeric-DE probe only when this op-list actually contains a
    // W_BULB_NUMERIC op — the GLSL program recompiles per structural edit
    // anyway, and analytic formulas shouldn't carry (or pay register pressure
    // for) a dead orbitR + branch (mirrors the WGSL variant split).
    const wantsNumeric = ops.some((op) => byKey(op.key)?.wRule === W_BULB_NUMERIC);
    const numericChunk = !wantsNumeric ? "" : `
// Orbit escape radius only — the numeric-DE probe (deOption 3, mirrors
// shader.js orbitR). Sampled 4× per DE for the finite-difference gradient.
float orbitR(vec3 p0) {
  vec3 pos = p0; float w = 1.0;
  vec3 c = (uJulia > 0.5) ? uJc : p0;
  for (int i = 0; i < 64; i++) {
    if (i >= uIters) break;
    iterStep(pos, w, i);
    if (uAddGate == 1) pos += c;
    if (dot(pos, pos) > uBailout) break;
  }
  return length(pos);
}
`;
    const numericBranch = !wantsNumeric ? "" : `
  // DEoption 3 — numeric finite-difference DE (mirrors shader.js mapDE_single).
  if (uDeOption >= 2.5) {
    float R = orbitR(p0);
    float eps = 1e-4 * max(1.0, length(p0));
    vec3 g = vec3(orbitR(p0 + vec3(eps, 0.0, 0.0)) - R,
                  orbitR(p0 + vec3(0.0, eps, 0.0)) - R,
                  orbitR(p0 + vec3(0.0, 0.0, eps)) - R) / eps;
    float gl = length(g);
    float de = R * log(max(R, 1.0)) / (gl + 0.06);
    return (gl > 1e-3) ? de : 0.5 * sqrt(max(R, 0.0));
  }
`;
    core = `
void iterStep(inout vec3 pos, inout float w, int i) {
  g_wq = 1.0;
${body}
}
${numericChunk}
float mapDE(vec3 p_rel) {
  vec3 p0 = p_rel + uOffset; // deep zoom §3.2 — the ONE reconstruction point
${numericBranch}
  vec3 pos = p0; float w = 1.0;
  vec3 c = (uJulia > 0.5) ? uJc : p0;
  for (int i = 0; i < 64; i++) {
    if (i >= uIters) break;
    iterStep(pos, w, i);
    if (uAddGate == 1) pos += c;
    if (dot(pos, pos) > uBailout) break;
  }
  float r = length(pos);
  if (uDeOption < 1.0) return 0.5 * log(max(r, 1e-9)) * r / max(abs(w), 1e-9);
  return r / max(abs(w), 1e-9);
}

float orbitTrap(vec3 p_rel) {
  vec3 p0 = p_rel + uOffset; // deep zoom §3.2
  vec3 pos = p0; float w = 1.0; vec3 c = (uJulia > 0.5) ? uJc : p0; float tr = 1e9;
  for (int i = 0; i < 64; i++) {
    if (i >= uIters) break;
    iterStep(pos, w, i);
    if (uAddGate == 1) pos += c;
    tr = min(tr, length(pos));
    if (dot(pos, pos) > uBailout) break;
  }
  return tr;
}

float escapeIter(vec3 p_rel) {
  vec3 p0 = p_rel + uOffset; // deep zoom §3.2
  vec3 pos = p0; float w = 1.0; vec3 c = (uJulia > 0.5) ? uJc : p0; int esc = uIters;
  for (int i = 0; i < 64; i++) {
    if (i >= uIters) break;
    iterStep(pos, w, i);
    if (uAddGate == 1) pos += c;
    if (dot(pos, pos) > uBailout) { esc = i; break; }
  }
  return float(esc) / float(max(uIters, 1));
}`;
  }
  return `#version 300 es
precision highp float;
out vec4 fragColor;

uniform vec2 uRes;
uniform float uFov;
uniform vec3 uCamPos, uCamFwd, uCamRight, uCamUp;
uniform vec3 uOffset; // deep zoom §3.1 — (0,0,0) for scenes (§14, exact no-op)
uniform int uIters, uMaxSteps, uColorMode;
${coreUniforms}
uniform float uBailout, uEps, uDeScale, uDeOption;
uniform vec3 uColA, uColB, uBg, uJc;
uniform float uJulia;
uniform vec3 uPalA, uPalB, uPalC, uPalD;
uniform float uPalOn;
uniform vec3 uLightDir;
uniform float uAmbient, uRim, uGloss, uIntensity;
uniform vec3 uKeyC, uFillDir, uFillC, uBackDir, uBackC; // P1 light rig
uniform float uMetallic, uShadowK, uShadowOn, uAoStr, uFill, uBack; // P1
uniform float uSky, uSunGlow, uGround, uIbl, uFogAmt, uInScatter; // P3 (bloom = WebGPU-only)
uniform float uExposure; // whole-frame EV (mirrors the WGSL post word)
uniform float uNear, uFar; // deep zoom §5 — was hardcoded 0.02 / 80.0
uniform float uP[${MAX_PARAMS}];

float g_wq; // desktop 4D scratch the shared op bodies write to; unused here
${core}

// e scales with hit distance t (deep zoom §3.4/§5) — a fixed epsilon straddles
// unrelated geometry once the near/far range is no longer a fixed [0.02, 80].
// p_rel is the RESIDUAL hit point (§3.4) — perturbing it here, before each
// mapDE call reconstructs, is what keeps the 4-tap difference precision-correct.
vec3 calcNormal(vec3 p_rel, float t) {
  vec2 e = vec2(1.0, -1.0) * clamp(t * 3e-5, 1e-6, 6e-4);
  return normalize(
    e.xyy * mapDE(p_rel + e.xyy) + e.yyx * mapDE(p_rel + e.yyx) +
    e.yxy * mapDE(p_rel + e.yxy) + e.xxx * mapDE(p_rel + e.xxx));
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
vec3 envColor(vec3 rd) {
  float tg = clamp(rd.y * 0.5 + 0.5, 0.0, 1.0);
  vec3 sky = s2l(mix(uBg * 0.35, uBg, tg));
  sky = mix(sky * (1.0 - uGround), sky, smoothstep(-0.35, 0.12, rd.y));
  float sunAmt = pow(max(dot(rd, normalize(uLightDir)), 0.0), 24.0);
  return sky + s2l(uKeyC) * (sunAmt * uSunGlow);
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
  float gl_ = ndl / (ndl * (1.0 - kk) + kk);
  vec3 F = f0 + (vec3(1.0) - f0) * pow(1.0 - max(dot(h, v), 0.0), 5.0);
  return D * gv * gl_ * F / max(4.0 * ndv, 1e-3);
}

void main() {
  vec2 uv = gl_FragCoord.xy / uRes;
  vec2 ndc = uv * 2.0 - 1.0;
  float aspect = uRes.x / uRes.y;
  float tanF = tan(0.5 * uFov);
  vec3 rd = normalize(uCamFwd + (ndc.x * aspect * tanF) * uCamRight + (ndc.y * tanF) * uCamUp);
  // Deep zoom §3.1 — uCamPos is the RESIDUAL ro_rel, not the absolute eye; every
  // p below is p_rel, reconstructed once per call inside mapDE/calcNormal/
  // orbitTrap/escapeIter/sceneTintGL (§3.2). uOffset=(0,0,0) for scenes (§14).
  vec3 ro = uCamPos;
  vec3 bg = s2l(mix(uBg * 0.35, uBg, clamp(uv.y, 0.0, 1.0)));

  float t = uNear; int steps = 0; bool hit = false;
  for (int i = 0; i < 512; i++) {
    if (i >= uMaxSteps) break;
    steps = i;
    float d = mapDE(ro + rd * t) * uDeScale;
    if (d < uEps * t) { hit = true; break; }
    t += d;
    if (t > uFar) break;
  }
  vec3 bgOut = uSky > 0.0 ? mix(bg, envColor(rd), clamp(uSky, 0.0, 1.0)) : bg; // P3 sky blend
  // Encode the miss like the hit path (kills the double-dark bg quirk — the
  // displayed background now matches the picked theme colors; mirrors WGSL).
  if (!hit) { fragColor = vec4(l2s(tone3(bgOut * exp2(uExposure))), 1.0); return; }

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

  float mixT;
  if (uColorMode == 2) mixT = escapeIter(p);
  else if (uColorMode == 1) mixT = clamp(orbitTrap(p) / 1.5, 0.0, 1.0);
  else mixT = 0.5 + 0.5 * nrm.z;

  vec3 albedo;
  if (uPalOn > 0.5)
    albedo = clamp(uPalA + uPalB * cos(6.2831853 * (uPalC * mixT + uPalD)), vec3(0.0), vec3(1.0));
  else albedo = mix(uColA, uColB, mixT);
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
  if (uFogAmt > 0.0) { // P3 fog + sun in-scatter (mirrors WGSL); 0 = legacy fade
    float ff = 1.0 - exp(-3.0 * uFogAmt * t / uFar);
    float scat = pow(max(dot(rd, normalize(uLightDir)), 0.0), 8.0) * uInScatter;
    col = mix(col, bgOut + s2l(uKeyC) * scat, ff);
  } else {
    col = mix(col, bgOut, clamp(t / uFar, 0.0, 1.0) * 0.6);
  }
  // P0 tail (single-pass — WebGL2 has no HDR intermediate, but the math is the
  // same as the WebGPU post pass): tone map → exact sRGB encode → dither.
  col = l2s(tone3(max(col, vec3(0.0)) * exp2(uExposure)));
  col += vec3((ign(gl_FragCoord.xy) - 0.5) / 255.0);
  fragColor = vec4(col, 1.0);
}`;
}

// ── CSG Phase 1b/C — multi-object scene fragment shader (the GLSL mirror) ─────
// The WGSL backend (shader.js mapDE) loops `objects[]` at runtime indexing a
// storage buffer; GLSL can't, so we CODEGEN one objDE_k() per object (IFS slice
// via iterBodyGL, analytic box/sphere for primitives) + a static `mapDE()` that
// unrolls the combine with per-object placement. Structure (object count, each
// object's op sequence, objType, combine) is baked into the source — the program
// rebuilds on a structural scene change; per-object scalars (origin/uscale/quat/
// blendK/juliaC/primParam/iters/flags) ride uniform ARRAYS so value tweaks don't
// recompile. Mirrors shader.js qrot/boxDE/sphereDE/sminP/objIterDE EXACTLY.
// Scenes are surface-mode only (§3.8) — no orbit-trap/escape codegen.
export function buildSceneFragGL(objects) {
  const layout = sceneParamLayout(objects);

  // Per-object DE functions.
  let funcs = "";
  for (let k = 0; k < objects.length; k++) {
    const o = objects[k];
    const objType = Number(o.objType) & 0xf;
    if (objType === 1) {
      // Analytic box SDF (exact). Mirrors shader.js boxDE().
      funcs += `
float objDE_${k}(vec3 p0) {
  vec3 q = abs(p0) - vec3(uObjPrim[${k}]);
  return length(max(q, vec3(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}`;
    } else if (objType === 2) {
      // Analytic sphere SDF. Mirrors shader.js sphereDE().
      funcs += `
float objDE_${k}(vec3 p0) {
  return length(p0) - uObjPrim[${k}];
}`;
    } else if (objType === 3) {
      // Analytic torus SDF. Mirrors shader.js torusDE() (R=prim, r=prim2).
      funcs += `
float objDE_${k}(vec3 p0) {
  vec2 q = vec2(length(p0.xz) - uObjPrim[${k}], p0.y);
  return length(q) - uObjPrim2[${k}];
}`;
    } else if (objType === 4) {
      // Capped-cylinder SDF. Mirrors shader.js cylinderDE() (r=prim, h=prim2).
      funcs += `
float objDE_${k}(vec3 p0) {
  vec2 d = vec2(length(p0.xz) - uObjPrim[${k}], abs(p0.y) - uObjPrim2[${k}]);
  return min(max(d.x, d.y), 0.0) + length(max(d, vec2(0.0)));
}`;
    } else if (objType === 5) {
      // Vertical-capsule SDF. Mirrors shader.js capsuleDE() (r=prim, h=prim2).
      funcs += `
float objDE_${k}(vec3 p0) {
  vec3 q = p0; q.y -= clamp(q.y, -uObjPrim2[${k}], uObjPrim2[${k}]);
  return length(q) - uObjPrim[${k}];
}`;
    } else if (objType === 6) {
      // Slab/ground-plane SDF. Mirrors shader.js planeDE() (thick=prim).
      funcs += `
float objDE_${k}(vec3 p0) {
  return abs(p0.y) - uObjPrim[${k}];
}`;
    } else {
      // IFS op-slice — mirrors shader.js objIterDE(): per-object ops, iters,
      // addC(||julia), deOption, Julia constant. Params bound at this object's
      // uP[] base so several objects share one uP[] array.
      const { body } = iterBodyGL(activeSceneOps(o), layout[k].slotBase);
      // Box-DE base (boxBase) — structural, so it's baked into the codegen (the
      // program already rebuilds on a structural signature change). he = uObjPrim[k]
      // (primParam repurposed for IFS). Mirrors shader.js objIterDE bit11 branch.
      const finalDE = o.boxBase
        ? `vec3 bq = abs(pos) - vec3(uObjPrim[${k}]);
  float bd = length(max(bq, vec3(0.0))) + min(max(bq.x, max(bq.y, bq.z)), 0.0);
  return bd / max(abs(w), 1e-9);`
        : `return r / max(abs(w), 1e-9);`;
      funcs += `
float objDE_${k}(vec3 p0) {
  vec3 pos = p0; float w = 1.0;
  vec3 c = (uObjJulia[${k}] > 0.5) ? uObjJc[${k}] : p0;
  int it = uObjIters[${k}];
  for (int i = 0; i < 64; i++) {
    if (i >= it) break;
    g_wq = 1.0;
${body}    if (uObjAddGate[${k}] == 1) pos += c;
    if (dot(pos, pos) > uBailout) break;
  }
  float r = length(pos);
  if (uObjDeOption[${k}] < 1.0) return 0.5 * log(max(r, 1e-9)) * r / max(abs(w), 1e-9);
  ${finalDE}
}`;
    }
  }

  // Scene mapDE — combine over objects (unrolled). Mirrors shader.js mapDE():
  // pk = qrot(conj(q), p−origin)/uscale; dk = objDE_k(pk)·uscale; combine.
  // mapBody = scene DE (combine over objects). tintBody mirrors it but tracks the
  // winning object id (§3.8 per-object color) — both share the same placement so
  // they can't drift. Mirrors shader.js mapDE() / sceneTint().
  let mapBody = "  float d = 1.0e9;\n";
  let tintBody = "  float d = 1.0e9; int win = 0;\n";
  for (let k = 0; k < objects.length; k++) {
    const o = objects[k];
    // subtract/intersect use (smooth-)max() — not a valid distance bound (oversteps
    // the carved walls); the scene compensates with a tighter deScale (preview.js).
    const combine = (o.combine ?? o.combineType ?? 0) & 3;
    const comb =
      combine === 1
        ? `d = sminGL(d, dk, uObjBlendK[${k}]);` // smooth-union
        : combine === 2
          ? `d = smaxGL(d, -dk, uObjBlendK[${k}]);` // subtract: carve k out (blendK rounds the cut)
          : combine === 3
            ? `d = smaxGL(d, dk, uObjBlendK[${k}]);` // intersect: keep overlap (blendK rounds the seam)
            : `d = min(d, dk);`; // union
    const tint =
      combine === 2
        ? `float nd = smaxGL(d, -dk, uObjBlendK[${k}]); if (-dk > d) win = ${k}; d = nd;`
        : combine === 3
          ? `float nd = smaxGL(d, dk, uObjBlendK[${k}]); if (dk > d) win = ${k}; d = nd;`
          : combine === 1
            ? `if (dk < d) win = ${k}; d = sminGL(d, dk, uObjBlendK[${k}]);`
            : `if (dk < d) win = ${k}; d = min(d, dk);`;
    const place = `vec4 q = uObjQuat[${k}]; vec4 qinv = vec4(-q.x, -q.y, -q.z, q.w);
    vec3 pk = qrotGL(qinv, p0 - uObjOrigin[${k}]) / uObjUscale[${k}];
    float dk = objDE_${k}(pk) * uObjUscale[${k}];`;
    mapBody += `  {
    ${place}
    ${comb}
  }
`;
    tintBody += `  {
    ${place}
    ${tint}
  }
`;
  }
  mapBody += "  return d;\n";
  tintBody += "  return uObjColor[win];\n";

  return `#version 300 es
precision highp float;
out vec4 fragColor;

uniform vec2 uRes;
uniform float uFov;
uniform vec3 uCamPos, uCamFwd, uCamRight, uCamUp;
uniform vec3 uOffset; // deep zoom §3.1 — (0,0,0) for scenes (§14, exact no-op)
uniform int uIters, uAddGate, uMaxSteps, uColorMode;
uniform float uBailout, uEps, uDeScale, uDeOption;
uniform vec3 uColA, uColB, uBg, uJc;
uniform float uJulia;
uniform vec3 uPalA, uPalB, uPalC, uPalD;
uniform float uPalOn;
uniform vec3 uLightDir;
uniform float uAmbient, uRim, uGloss, uIntensity;
uniform vec3 uKeyC, uFillDir, uFillC, uBackDir, uBackC; // P1 light rig
uniform float uMetallic, uShadowK, uShadowOn, uAoStr, uFill, uBack; // P1
uniform float uSky, uSunGlow, uGround, uIbl, uFogAmt, uInScatter; // P3 (bloom = WebGPU-only)
uniform float uExposure; // whole-frame EV (mirrors the WGSL post word)
uniform float uNear, uFar; // deep zoom §5 — was hardcoded 0.02 / 80.0
uniform float uP[${MAX_PARAMS}];

// CSG scene per-object uniforms (cap ${MAX_OBJECTS}). Structure is codegen'd;
// these scalars ride uniforms so value tweaks don't force a recompile.
uniform vec3  uObjOrigin[${MAX_OBJECTS}];
uniform float uObjUscale[${MAX_OBJECTS}];
uniform vec4  uObjQuat[${MAX_OBJECTS}];
uniform float uObjBlendK[${MAX_OBJECTS}];
uniform vec3  uObjJc[${MAX_OBJECTS}];
uniform float uObjPrim[${MAX_OBJECTS}];
uniform float uObjPrim2[${MAX_OBJECTS}];
uniform int   uObjIters[${MAX_OBJECTS}];
uniform int   uObjAddGate[${MAX_OBJECTS}];
uniform float uObjJulia[${MAX_OBJECTS}];
uniform float uObjDeOption[${MAX_OBJECTS}];
uniform vec3  uObjColor[${MAX_OBJECTS}];  // per-object albedo (sRGB) — §3.8

float g_wq; // desktop 4D scratch the shared op bodies write to; unused here

// Rotate v by quaternion q (q = local→world). Mirrors shader.js qrot().
vec3 qrotGL(vec4 q, vec3 v) {
  vec3 u = q.xyz; float s = q.w;
  return 2.0 * dot(u, v) * u + (s * s - dot(u, u)) * v + 2.0 * s * cross(u, v);
}
// Polynomial smooth-min (under-estimates ≤ k/4 → DE-safe). Mirrors shader.js sminP().
float sminGL(float a, float b, float k) {
  if (k <= 0.0) return min(a, b);
  float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
  return mix(b, a, h) - k * h * (1.0 - h);
}
// Smooth max — carve dual of sminGL (k=0 → hard max). Mirrors shader.js smaxP().
float smaxGL(float a, float b, float k) { return -sminGL(-a, -b, k); }
${funcs}

// Deep zoom §3.1: uOffset is always (0,0,0) for scenes (§14, objDist's per-
// object placement isn't recentered in v1) — this reconstruction is therefore
// an exact no-op today, kept only so the pattern matches the single-object
// shader and needs no change when CSG recenter is eventually designed.
float mapDE(vec3 p_rel) {
  vec3 p0 = p_rel + uOffset;
${mapBody}}

// Per-object albedo at p0 — the winning object's color (§3.8). Mirrors shader.js sceneTint().
vec3 sceneTintGL(vec3 p_rel) {
  vec3 p0 = p_rel + uOffset;
${tintBody}}

// Scenes are surface-mode only (§3.8); stubs satisfy main()'s references.
float orbitTrap(vec3 p0) { return 0.0; }
float escapeIter(vec3 p0) { return 0.0; }

// e scales with hit distance t (deep zoom §3.4/§5) — a fixed epsilon straddles
// unrelated geometry once the near/far range is no longer a fixed [0.02, 80].
// p_rel is the RESIDUAL hit point (§3.4) — perturbing it here, before each
// mapDE call reconstructs, is what keeps the 4-tap difference precision-correct.
vec3 calcNormal(vec3 p_rel, float t) {
  vec2 e = vec2(1.0, -1.0) * clamp(t * 3e-5, 1e-6, 6e-4);
  return normalize(
    e.xyy * mapDE(p_rel + e.xyy) + e.yyx * mapDE(p_rel + e.yyx) +
    e.yxy * mapDE(p_rel + e.yxy) + e.xxx * mapDE(p_rel + e.xxx));
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
vec3 envColor(vec3 rd) {
  float tg = clamp(rd.y * 0.5 + 0.5, 0.0, 1.0);
  vec3 sky = s2l(mix(uBg * 0.35, uBg, tg));
  sky = mix(sky * (1.0 - uGround), sky, smoothstep(-0.35, 0.12, rd.y));
  float sunAmt = pow(max(dot(rd, normalize(uLightDir)), 0.0), 24.0);
  return sky + s2l(uKeyC) * (sunAmt * uSunGlow);
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
  float gl_ = ndl / (ndl * (1.0 - kk) + kk);
  vec3 F = f0 + (vec3(1.0) - f0) * pow(1.0 - max(dot(h, v), 0.0), 5.0);
  return D * gv * gl_ * F / max(4.0 * ndv, 1e-3);
}

void main() {
  vec2 uv = gl_FragCoord.xy / uRes;
  vec2 ndc = uv * 2.0 - 1.0;
  float aspect = uRes.x / uRes.y;
  float tanF = tan(0.5 * uFov);
  vec3 rd = normalize(uCamFwd + (ndc.x * aspect * tanF) * uCamRight + (ndc.y * tanF) * uCamUp);
  // Deep zoom §3.1 — uCamPos is the RESIDUAL ro_rel, not the absolute eye; every
  // p below is p_rel, reconstructed once per call inside mapDE/calcNormal/
  // orbitTrap/escapeIter/sceneTintGL (§3.2). uOffset=(0,0,0) for scenes (§14).
  vec3 ro = uCamPos;
  vec3 bg = s2l(mix(uBg * 0.35, uBg, clamp(uv.y, 0.0, 1.0)));

  float t = uNear; int steps = 0; bool hit = false;
  for (int i = 0; i < 512; i++) {
    if (i >= uMaxSteps) break;
    steps = i;
    float d = mapDE(ro + rd * t) * uDeScale;
    if (d < uEps * t) { hit = true; break; }
    t += d;
    if (t > uFar) break;
  }
  vec3 bgOut = uSky > 0.0 ? mix(bg, envColor(rd), clamp(uSky, 0.0, 1.0)) : bg; // P3 sky blend
  // Encode the miss like the hit path (kills the double-dark bg quirk — the
  // displayed background now matches the picked theme colors; mirrors WGSL).
  if (!hit) { fragColor = vec4(l2s(tone3(bgOut * exp2(uExposure))), 1.0); return; }

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

  vec3 albedo = s2l(sceneTintGL(p)); // per-object color (§3.8); sRGB→linear (issue #6)

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
  if (uFogAmt > 0.0) { // P3 fog + sun in-scatter (mirrors WGSL); 0 = legacy fade
    float ff = 1.0 - exp(-3.0 * uFogAmt * t / uFar);
    float scat = pow(max(dot(rd, normalize(uLightDir)), 0.0), 8.0) * uInScatter;
    col = mix(col, bgOut + s2l(uKeyC) * scat, ff);
  } else {
    col = mix(col, bgOut, clamp(t / uFar, 0.0, 1.0) * 0.6);
  }
  // P0 tail (single-pass — WebGL2 has no HDR intermediate, but the math is the
  // same as the WebGPU post pass): tone map → exact sRGB encode → dither.
  col = l2s(tone3(max(col, vec3(0.0)) * exp2(uExposure)));
  col += vec3((ign(gl_FragCoord.xy) - 0.5) / 255.0);
  fragColor = vec4(col, 1.0);
}`;
}

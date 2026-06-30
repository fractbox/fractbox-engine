// WebGL2 (GLSL ES 3.00) backend — the SECOND live renderer, for the large
// population that has WebGL2 but not WebGPU. Same raymarch + DE + coloring as
// the WGSL shader (shader.js), so it's near-full-fidelity, not a degraded view.
//
// Op math reuse: instead of a third hand-written copy, the per-iteration body is
// generated from the SAME operators.js `glsl()` emitter the desktop export uses,
// with params bound to a `uP[]` uniform array. So editing param VALUES is pure
// uniform upload; only adding/removing/reordering ops (a STRUCTURE change)
// regenerates + recompiles the fragment shader.

import { byKey } from "./operators.js";

export const MAX_PARAMS = 192; // 64 ops × 3 params — matches renderer MAX_OPS

// Fullscreen triangle from gl_VertexID (no vertex buffer needed in WebGL2).
export const VERT_GL = `#version 300 es
void main() {
  vec2 p = vec2(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0);
  gl_Position = vec4(p, 0.0, 1.0);
}`;

// One iteration of the op stack, params bound to uP[] (angles → radians()).
// Returns { body, paramCount }; paramCount drives how much of uP we upload.
export function iterBodyGL(ops) {
  let body = "";
  let slot = 0;
  for (const op of ops) {
    const def = byKey(op.key);
    if (!def) continue;
    const v = def.params.map((pm) => {
      const ref = `uP[${slot++}]`;
      return pm.type === "angle" ? `radians(${ref})` : ref;
    });
    body += def.glsl(v) + "\n";
  }
  return { body, paramCount: slot };
}

// Build the full fragment shader for a given op-list. The op bodies from
// operators.js are already GLSL (the desktop dialect), so they drop straight in.
export function buildFragGL(ops) {
  const { body } = iterBodyGL(ops);
  return `#version 300 es
precision highp float;
out vec4 fragColor;

uniform vec2 uRes;
uniform float uFov;
uniform vec3 uCamPos, uCamFwd, uCamRight, uCamUp;
uniform int uIters, uAddGate, uMaxSteps, uColorMode;
uniform float uBailout, uEps, uDeScale, uDeOption;
uniform vec3 uColA, uColB, uBg, uJc;
uniform float uJulia;
uniform vec3 uPalA, uPalB, uPalC, uPalD;
uniform float uPalOn;
uniform vec3 uLightDir;
uniform float uAmbient, uRim, uGloss, uIntensity;
uniform float uP[${MAX_PARAMS}];

float g_wq; // desktop 4D scratch the shared op bodies write to; unused here

void iterStep(inout vec3 pos, inout float w) {
  g_wq = 1.0;
${body}
}

float mapDE(vec3 p0) {
  vec3 pos = p0; float w = 1.0;
  vec3 c = (uJulia > 0.5) ? uJc : p0;
  for (int i = 0; i < 64; i++) {
    if (i >= uIters) break;
    iterStep(pos, w);
    if (uAddGate == 1) pos += c;
    if (dot(pos, pos) > uBailout) break;
  }
  float r = length(pos);
  if (uDeOption < 1.0) return 0.5 * log(max(r, 1e-9)) * r / max(abs(w), 1e-9);
  return r / max(abs(w), 1e-9);
}

float orbitTrap(vec3 p0) {
  vec3 pos = p0; float w = 1.0; vec3 c = (uJulia > 0.5) ? uJc : p0; float tr = 1e9;
  for (int i = 0; i < 64; i++) {
    if (i >= uIters) break;
    iterStep(pos, w);
    if (uAddGate == 1) pos += c;
    tr = min(tr, length(pos));
    if (dot(pos, pos) > uBailout) break;
  }
  return tr;
}

float escapeIter(vec3 p0) {
  vec3 pos = p0; float w = 1.0; vec3 c = (uJulia > 0.5) ? uJc : p0; int esc = uIters;
  for (int i = 0; i < 64; i++) {
    if (i >= uIters) break;
    iterStep(pos, w);
    if (uAddGate == 1) pos += c;
    if (dot(pos, pos) > uBailout) { esc = i; break; }
  }
  return float(esc) / float(max(uIters, 1));
}

vec3 calcNormal(vec3 p) {
  vec2 e = vec2(1.0, -1.0) * 0.0006;
  return normalize(
    e.xyy * mapDE(p + e.xyy) + e.yyx * mapDE(p + e.yyx) +
    e.yxy * mapDE(p + e.yxy) + e.xxx * mapDE(p + e.xxx));
}

vec3 s2l(vec3 c) { return pow(max(c, vec3(0.0)), vec3(2.2)); } // sRGB→linear (issue #6)

void main() {
  vec2 uv = gl_FragCoord.xy / uRes;
  vec2 ndc = uv * 2.0 - 1.0;
  float aspect = uRes.x / uRes.y;
  float tanF = tan(0.5 * uFov);
  vec3 rd = normalize(uCamFwd + (ndc.x * aspect * tanF) * uCamRight + (ndc.y * tanF) * uCamUp);
  vec3 ro = uCamPos;
  vec3 bg = s2l(mix(uBg * 0.35, uBg, clamp(uv.y, 0.0, 1.0)));

  float t = 0.02; int steps = 0; bool hit = false;
  for (int i = 0; i < 512; i++) {
    if (i >= uMaxSteps) break;
    steps = i;
    float d = mapDE(ro + rd * t) * uDeScale;
    if (d < uEps * t) { hit = true; break; }
    t += d;
    if (t > 80.0) break;
  }
  if (!hit) { fragColor = vec4(bg, 1.0); return; }

  vec3 p = ro + rd * t;
  vec3 nrm = calcNormal(p);
  float ao = 1.0 - float(steps) / float(uMaxSteps);
  vec3 lightDir = normalize(uLightDir);
  float diff = max(dot(nrm, lightDir), 0.0);
  float rim = pow(1.0 - max(dot(nrm, -rd), 0.0), 2.0);
  float amb = uAmbient;

  float mixT;
  if (uColorMode == 2) mixT = escapeIter(p);
  else if (uColorMode == 1) mixT = clamp(orbitTrap(p) / 1.5, 0.0, 1.0);
  else mixT = 0.5 + 0.5 * nrm.z;

  vec3 albedo;
  if (uPalOn > 0.5)
    albedo = clamp(uPalA + uPalB * cos(6.2831853 * (uPalC * mixT + uPalD)), vec3(0.0), vec3(1.0));
  else albedo = mix(uColA, uColB, mixT);
  albedo = s2l(albedo); // sRGB→linear (issue #6)

  vec3 halfv = normalize(lightDir - rd);
  float spec = uGloss * pow(max(dot(nrm, halfv), 0.0), 32.0);
  vec3 col = albedo * (amb + (1.0 - amb) * diff) * (0.35 + 0.65 * ao);
  col += vec3(spec) * ao;
  col += s2l(uColB) * (rim * uRim * ao);
  col *= uIntensity; // light intensity (exposure)
  col = mix(col, bg, clamp(t / 80.0, 0.0, 1.0) * 0.6);
  col = pow(max(col, vec3(0.0)), vec3(1.0 / 2.2));
  fragColor = vec4(col, 1.0);
}`;
}

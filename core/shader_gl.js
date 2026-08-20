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
import { normalizeSceneObject } from "./sceneobj.js";
import { LEAVES } from "./leaves.js";
// Capacity caps live once in limits.js; re-export so renderer_gl's existing
// import site keeps working while the values are single-sourced.
import { MAX_PARAMS, MAX_OBJECTS, MAX_ITERS } from "./limits.js";
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
    // Unknown keys THROW, matching the WebGPU tier (renderer.js writeOps) —
    // silently skipping an op here rendered a different fractal on this tier.
    if (!def) throw new Error(`iterBodyGL: unknown op key ${op.key}`);
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

// CSG scene — uP[] layout across objects. Objects with an op chain each
// consume `paramCount` slots (their ACTIVE ops' params, concatenated); pure
// leaves consume none. buildSceneFragGL AND renderer_gl.writeScene both call
// this so the codegen'd uP[] indices and the uploaded values stay in lockstep.
export function sceneParamLayout(objects) {
  let cursor = 0;
  const layout = objects.map((o) => {
    const n = normalizeSceneObject(o);
    if (!n.ops.length) return { slotBase: cursor, paramCount: 0 };
    const { paramCount } = iterBodyGL(n.ops);
    const slotBase = cursor;
    cursor += paramCount;
    return { slotBase, paramCount };
  });
  // Explicit capacity check: the packed per-object params share ONE uP[] array,
  // and a sanitize-legal scene (8 objects × 24 ops × 6 params = 1152) can exceed
  // it. Without this the overflow only surfaced later as an opaque GLSL compile
  // error (out-of-range uP[] index) thrown from linkProgram.
  if (cursor > MAX_PARAMS)
    throw new Error(
      `scene needs ${cursor} op params across ${objects.length} objects — ` +
        `over the WebGL2 uniform budget uP[${MAX_PARAMS}]; ` +
        `reduce ops/params on the scene's IFS objects`,
    );
  return layout;
}

// ── Shared fragment-shader chunks ─────────────────────────────────────────────
// buildFragGL (flat/hybrid) and buildSceneFragGL (CSG scenes) emit the SAME
// raymarcher around different mapDE cores. Everything the two programs share —
// prelude, the uniform block's head/tail, the shade-helper library, main() —
// lives here ONCE so a lighting/tonemap change is a 2-way edit (WGSL +
// this file) instead of a 3-way one. The REAL divergences stay explicit at
// each call site: the iteration/coloring uniform middle (flat/hybrid carry the
// addC-gate uniforms) and main()'s albedo block (palette/orbit-trap mix vs
// per-object scene tint) — see `commonUniformsGL(variantUniforms)` and
// `mainGL(albedoBlock)` below.

const FRAG_PRELUDE_GL = `#version 300 es
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
}`;

// Uniform block. The camera/marcher head and the light-rig/post tail are
// identical in both builders; the middle (iteration + coloring plumbing)
// differs — flat/hybrid carries the addC-gate uniform(s) — so each builder
// passes its own `variantUniforms` lines explicitly.
function commonUniformsGL(variantUniforms, envx = false) {
  return `uniform vec2 uRes;
uniform float uFov;
uniform vec3 uCamPos, uCamFwd, uCamRight, uCamUp;
// #441 ORTHOGRAPHIC half-height; 0 = perspective. A NEW uniform, not a mirror
// of the WGSL edit: the WGSL tier hides this in camFwd.w, but GLSL declares
// plain vec3s with no padding component to reuse.
uniform float uOrthoH;
uniform vec3 uOffset; // deep zoom §3.1 — (0,0,0) for scenes (§14, exact no-op)
${variantUniforms}
uniform vec3 uLightDir;
uniform float uAmbient, uRim, uGloss, uIntensity;
uniform vec3 uKeyC, uFillDir, uFillC, uBackDir, uBackC; // P1 light rig
uniform float uMetallic, uShadowK, uShadowOn, uAoStr, uFill, uBack; // P1
uniform float uSky, uSunGlow, uGround, uIbl, uFogAmt, uInScatter; // P3 (bloom = WebGPU-only)
${
  envx
    ? `uniform float uStars, uStarDensity, uStarSeed, uBand; // ENVX (backgrounds P5)
uniform vec3 uBandDir, uZenC; // ENVX — band plane normal, zenith color
uniform float uZenOn; // ENVX — zenith blend (0 = legacy 0.35·bg zenith)
`
    : ""
}uniform float uExposure; // whole-frame EV (mirrors the WGSL post word)
uniform float uNear, uFar; // deep zoom §5 — was hardcoded 0.02 / 80.0`;
}

// ── std140 bulk UBO (GLES-minimum uniform budget) ─────────────────────────────
// The FAT arrays — op params (uP), palette stops, the hybrid schedule, and the
// per-object scene arrays — do NOT live in the default uniform block. On a
// GLES-3.0-minimum device (MAX_FRAGMENT_UNIFORM_VECTORS == 224, the spec floor)
// uP[] alone is MAX_PARAMS vec4s and the scene per-object arrays add ~88 more, so
// the default block overflowed and the WHOLE WebGL2 tier failed to LINK — a
// field dump (iOS 15.8 iPad) reported exactly
//   gl-link-fail "FRAGMENT shader uniforms count exceeds MAX_FRAGMENT_UNIFORM_VECTORS(224)".
// A std140 uniform BUFFER is core WebGL2 with a much larger floor
// (MAX_UNIFORM_BLOCK_SIZE ≥ 16384 bytes = 1024 vec4s), so the arrays move here.
// Array indexing in the shader is byte-identical (`uP[i]`, `uObjPrimP[k]`, …) —
// only the DECLARATION moves — so all op/coloring/scene math is untouched.
//
// bulkLayout() is the SINGLE SOURCE of the member set + order, consumed by both
// this emitter and renderer_gl.js's std140 upload (which computes byte offsets
// from the same descriptor), the way sceneParamLayout keeps uP[] in lockstep.
// Every member is an ARRAY, so under std140 each has a 16-byte element stride and
// a 16-aligned base — the CPU offset math is a running sum of count·16.
// `comps` = active components (float/int 1, vec3 3, vec4 4); `baseType` picks the
// CPU view (Int32Array vs Float32Array). See core/uniformbudget.test.mjs.
export function bulkLayout({ hybrid = 0, scene = false } = {}) {
  const M = [];
  const add = (name, glslType, comps, count) =>
    M.push({
      name,
      glslType,
      comps,
      count,
      baseType: glslType === "int" ? "int" : "float",
    });
  // Hybrid schedule (per-slot counts + addC gates) — arrays only when hybrid;
  // the flat path's single addC gate stays a scalar in the default block.
  if (hybrid) {
    add("uHyb", "int", 1, hybrid);
    add("uAddGate", "int", 1, hybrid);
  }
  add("uPalStops", "vec4", 4, 8); // COLORING P0 — 8-stop OKLab palette
  if (scene) {
    add("uObjOrigin", "vec3", 3, MAX_OBJECTS);
    add("uObjUscale", "float", 1, MAX_OBJECTS);
    add("uObjQuat", "vec4", 4, MAX_OBJECTS);
    add("uObjBlendK", "float", 1, MAX_OBJECTS);
    add("uObjJc", "vec3", 3, MAX_OBJECTS);
    add("uObjPrimP", "vec4", 4, MAX_OBJECTS); // leaf shapeParams (leaves.js)
    add("uObjIters", "int", 1, MAX_OBJECTS);
    add("uObjAddGate", "int", 1, MAX_OBJECTS);
    add("uObjJulia", "float", 1, MAX_OBJECTS);
    add("uObjDeOption", "float", 1, MAX_OBJECTS);
    add("uObjColor", "vec3", 3, MAX_OBJECTS); // per-object albedo (sRGB) — §3.8
  }
  add("uP", "float", 1, MAX_PARAMS); // op-param array (the biggest single hog)
  return M;
}

// Emit the std140 block declaration for a bulkLayout() member list. No instance
// name → members are referenced by bare name (`uP[i]`), so nothing downstream of
// the declaration changes. `layout(binding=)` is ES 3.1+, so the CPU side binds
// via glUniformBlockBinding after link (renderer_gl.setupBulk).
export function bulkBlockGL(members) {
  const lines = members
    .map((m) => `  ${m.glslType} ${m.name}[${m.count}];`)
    .join("\n");
  return `layout(std140) uniform Bulk {
${lines}
};`;
}

// ── Standalone export seam (M0, #291) ────────────────────────────────────────
// The FLAT fragment shader's complete uniform surface, as a table. The engine
// path above (commonUniformsGL + the variant block in buildFragGL) is the
// canonical hand-written source and is left byte-for-byte untouched — a golden
// test guards it. This table is the SAME surface in structured form, consumed by
// the standalone exporter to bake const values; a drift test (standalone.test.mjs)
// asserts the table's names equal the names parsed out of the engine block, so the
// two can't silently diverge. Only the flat surface is tabulated — hybrid/scene
// export is deferred (the exporter guards those out), so their extra uniforms
// (uHyb*, uObj*) are intentionally absent.
export const UNIFORM_TABLE_FLAT = [
  { type: "vec2", name: "uRes" },
  { type: "float", name: "uFov" },
  { type: "vec3", name: "uCamPos" },
  { type: "vec3", name: "uCamFwd" },
  { type: "vec3", name: "uCamRight" },
  { type: "vec3", name: "uCamUp" },
  { type: "float", name: "uOrthoH" },
  { type: "vec3", name: "uOffset" },
  { type: "int", name: "uIters" },
  { type: "int", name: "uMaxSteps" },
  { type: "int", name: "uColorMode" },
  { type: "int", name: "uAddGate" },
  { type: "float", name: "uBailout" },
  { type: "float", name: "uEps" },
  { type: "float", name: "uDeScale" },
  { type: "float", name: "uDeOption" },
  { type: "vec3", name: "uColA" },
  { type: "vec3", name: "uColB" },
  { type: "vec3", name: "uBg" },
  { type: "vec3", name: "uJc" },
  { type: "float", name: "uJulia" },
  { type: "vec3", name: "uPalA" },
  { type: "vec3", name: "uPalB" },
  { type: "vec3", name: "uPalC" },
  { type: "vec3", name: "uPalD" },
  { type: "float", name: "uPalOn" },
  { type: "vec4", name: "uPalStops", array: 8 },
  { type: "float", name: "uPalCount" },
  { type: "float", name: "uPalCyclic" },
  { type: "float", name: "uStripeFreq" },
  { type: "float", name: "uSigLo" },
  { type: "float", name: "uSigSpan" },
  { type: "float", name: "uIridescence" },
  { type: "float", name: "uPalettePhase" },
  { type: "vec3", name: "uLightDir" },
  { type: "float", name: "uAmbient" },
  { type: "float", name: "uRim" },
  { type: "float", name: "uGloss" },
  { type: "float", name: "uIntensity" },
  { type: "vec3", name: "uKeyC" },
  { type: "vec3", name: "uFillDir" },
  { type: "vec3", name: "uFillC" },
  { type: "vec3", name: "uBackDir" },
  { type: "vec3", name: "uBackC" },
  { type: "float", name: "uMetallic" },
  { type: "float", name: "uShadowK" },
  { type: "float", name: "uShadowOn" },
  { type: "float", name: "uAoStr" },
  { type: "float", name: "uFill" },
  { type: "float", name: "uBack" },
  { type: "float", name: "uSky" },
  { type: "float", name: "uSunGlow" },
  { type: "float", name: "uGround" },
  { type: "float", name: "uIbl" },
  { type: "float", name: "uFogAmt" },
  { type: "float", name: "uInScatter" },
  { type: "float", name: "uExposure" },
  { type: "float", name: "uNear" },
  { type: "float", name: "uFar" },
  { type: "float", name: "uP", array: MAX_PARAMS },
];

// ENVX (backgrounds P5) — the extension uniforms, present only when the look
// uses them (commonUniformsGL's envx splice). Kept OUT of UNIFORM_TABLE_FLAT so
// the drift test's default surface (and the golden fixtures) stay untouched;
// the bake path appends this table exactly when the emitted shader declares it.
export const UNIFORM_TABLE_ENVX = [
  { type: "float", name: "uStars" },
  { type: "float", name: "uStarDensity" },
  { type: "float", name: "uStarSeed" },
  { type: "float", name: "uBand" },
  { type: "vec3", name: "uBandDir" },
  { type: "vec3", name: "uZenC" },
  { type: "float", name: "uZenOn" },
];

// Render the flat uniform surface for the STANDALONE export: each entry is asked
// of the bake strategy `emit.valueFor(name, type, array)`. A returned literal
// string → a baked `const`; `null` → the uniform is KEPT live (camera/view). The
// strategy owns the keep/bake classification (single source with the exporter's
// TAXONOMY); this function owns only the surface (types/arrays).
export function bakeUniformBlock(emit, { envx = false } = {}) {
  return (envx ? [...UNIFORM_TABLE_FLAT, ...UNIFORM_TABLE_ENVX] : UNIFORM_TABLE_FLAT).map(({ type, name, array }) => {
    const suffix = array ? `[${array}]` : "";
    const v = emit.valueFor(name, type, array);
    return v == null
      ? `uniform ${type} ${name}${suffix};`
      : `const ${type} ${name}${suffix} = ${v};`;
  }).join("\n");
}

// Shade-helper library — normals, sRGB codec, tone map, dither, P1 shadows/AO/
// GGX, P3 environment. Needs only mapDE() + the common uniforms above, so both
// builders splice it after their mapDE core. `envx` (ENVX backgrounds P5)
// splices the starfield/zenith/band extension — false emits the pre-ENVX text
// byte-for-byte (the golden fixtures pin it), mirroring shader.js buildWGSL.
const shadeLibGL = (envx = false) => `// e scales with hit distance t (deep zoom §3.4/§5) — a fixed epsilon straddles
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
${
  envx
    ? `// ENVX (backgrounds P5) — procedural starfield + Milky-Way band, the GLSL
// mirror of shader.js starField (same cell hash, same constants — the two
// tiers must agree on where every star sits).
float hashE(vec3 p) {
  return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453);
}
vec3 starLayer(vec3 rd, float D, vec3 sOff, float kAng) {
  vec3 cell = floor(rd * D + sOff);
  vec3 j = vec3(hashE(cell + 0.13), hashE(cell + 7.31), hashE(cell + 3.77));
  vec3 s = normalize(cell + 0.5 + (j - 0.5) * 0.5 - sOff);
  float m = hashE(cell + 11.7);
  float live = step(0.3, hashE(cell + 17.3));
  float a2 = dot(rd - s, rd - s);
  float m2 = m * m;
  float b = exp(-a2 * kAng) * (0.05 + 0.75 * m2 * m2 * m2) * live;
  vec3 tint = mix(vec3(1.0), mix(vec3(1.0, 0.84, 0.66), vec3(0.68, 0.82, 1.0), hashE(cell + 5.19)), 0.5);
  return tint * b;
}
vec3 starField(vec3 rd) {
  float kAng = uRes.y * uRes.y / (2.0 * uFov * uFov);
  vec3 sOff = vec3(uStarSeed);
  float bandW = 1.0 - smoothstep(0.0, 0.45, abs(dot(rd, uBandDir)));
  float bandLift = mix(1.0, 0.25 + 1.75 * bandW, uBand);
  vec3 c = starLayer(rd, uStarDensity, sOff, kAng);
  c = c + starLayer(rd, min(uStarDensity * 1.7, 150.0), sOff + vec3(31.7), kAng) * 0.55;
  return c * (bandLift * uStars)
       + vec3(0.72, 0.78, 1.0) * (bandW * bandW * uBand * 0.12);
}
`
    : ""
}// P3 directional environment — mirrors shader.js envColor().
// Up is world +Z (the #160 / 31e2253 lighting frame): horizon in the XY plane.
vec3 envColor(vec3 rd) {
  float tg = clamp(rd.z * 0.5 + 0.5, 0.0, 1.0);
  vec3 sky = s2l(mix(${envx ? "mix(uBg * 0.35, uZenC, uZenOn)" : "uBg * 0.35"}, uBg, tg));
  sky = mix(sky * (1.0 - uGround), sky, smoothstep(-0.35, 0.12, rd.z));
${envx ? "  sky += starField(rd) * smoothstep(-0.35, 0.12, rd.z); // stars set below the horizon\n" : ""}  float sunAmt = pow(max(dot(rd, normalize(uLightDir)), 0.0), 24.0);
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
}`;
// ^ ggxSpec's Smith term for the light direction is `gLit`, NOT the obvious
// short name: identifiers with that reserved GLSL-ES prefix (or containing a
// double underscore) make ANGLE reject the WHOLE shader, and generated GLSL
// never compiles in CI (no GL context) — the previous name blacked out the
// entire WebGL2 tier all the way to live until a device report caught it
// (issue #206). shaderlint.test.mjs regex-gates the emitted source now; this
// comment sits OUTSIDE the template so the lint stays clean.

// main()'s albedo block — the ONE line-for-line divergence between the two
// mains. Flat/hybrid mixes palette/theme colors by the selected color mode
// (needs orbitTrap/escapeIter); scenes take the winning object's tint (§3.8).
const ALBEDO_FLAT_GL = `  vec3 albedo;
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
  albedo = s2l(albedo); // sRGB→linear (issue #6)`;
// Scene coloring (docs/design/SCENES.md §Coloring): Surface (mode 0) = the
// winning object's tint; Curvature is geometry-space; Pinwheel keys world
// azimuth; and COLORING P3 S5 gives Glow/Bands/Silk the winning object's REAL
// orbit (sceneOrbitGL). All arms stops-aware, mirroring shader.js albedoFor.
const SCENE_PAL_GL = `if (uPalCount > 1.5) albedo = albedoStopsGL(mixT);
    else if (uPalOn > 0.5)
      albedo = clamp(uPalA + uPalB * cos(6.2831853 * (uPalC * mixT + uPalD)), vec3(0.0), vec3(1.0));
    else albedo = mix(uColA, uColB, mixT);`;
const ALBEDO_SCENE_GL = `  vec3 albedo;
  if (uColorMode == 5) {
    float mixT = palPhase(normSig(curvatureAt(p, t))); // Curvature (geometry-space)
    ${SCENE_PAL_GL}
  } else if (uColorMode == 4) {
    float mixT = palPhase(fract(atan(p.y, p.x) * 0.15915494 + 0.5)); // Pinwheel → world azimuth
    ${SCENE_PAL_GL}
  } else if (uColorMode >= 1) {
    // COLORING P3 S5 — Glow/Bands/Silk run the winning object's real orbit.
    float mixT = palPhase(normSig(sceneOrbitGL(p, uColorMode)));
    ${SCENE_PAL_GL}
  } else {
    albedo = sceneTintGL(p); // mode 0 — per-object color (§3.8)
  }
  albedo = s2l(albedo); // sRGB→linear (issue #6)`;

// The raymarch + shade main(), shared verbatim except the albedo block above.
function mainGL(albedoBlock, envx = false) {
  return `void main() {
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
${
  envx
    ? `  // ENVX: stars are NOT hostage to the Sky blend — mirrors shader.js: the
  // complement of envColor's sky-scaled star term, so total star brightness
  // is the Stars slider alone at any Sky.
  bgOut += starField(rd) * smoothstep(-0.35, 0.12, rd.z) * (1.0 - clamp(uSky, 0.0, 1.0));
`
    : ""
}  // Encode the miss like the hit path (kills the double-dark bg quirk — the
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

${albedoBlock}

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
}`;
}

// Build the full fragment shader for a given op-list. The op bodies from
// operators.js are already GLSL (the desktop dialect), so they drop straight in.
// `extraSlots`, when a non-empty array ([{ops:[...]}…]) — HYBRID_NSLOT_SPEC.md
// §2.4 — makes this hybrid-aware: emits one iterStepK per slot (A + extras;
// params share one uP[] via chained slot-base offsets) and mapDE/orbitTrap/
// escapeIter alternate them via hybStep, driven by the uHyb[]/uAddGate[] SCHEDULE
// UNIFORM ARRAYS (not baked — only editing a slot's op-stack STRUCTURE rebuilds
// the program, same rule as the plain single-object path). Everything else
// (uniforms, calcNormal, main()) is shared verbatim — hybrid has no effect there.
// COLORING R S7 — the Painter chunk (one palette read + the per-iteration blend).
// Emitted AFTER SHADE_LIB_GL (needs albedoStopsGL) but the orbit uses the core's
// hybStep/iterStep (declared before it). GLSL requires declaration-before-use, so
// this can't live in SHADE_LIB_GL (before core) like the WGSL twin does.
function painterLibGL(hybrid) {
  const step = hybrid
    ? "hybStep(pos, w, c, i);"
    : "iterStep(pos, w, i);\n    if (uAddGate == 1) pos += c;";
  return `vec3 palLookupGL(float t) {
  float pt = palPhase(t);
  if (uPalCount > 1.5) return albedoStopsGL(pt);
  if (uPalOn > 0.5) return clamp(uPalA + uPalB * cos(6.2831853 * (uPalC * pt + uPalD)), vec3(0.0), vec3(1.0));
  return mix(uColA, uColB, pt);
}
vec3 orbitPainterGL(vec3 p_rel) {
  vec3 p0 = p_rel + uOffset;
  vec3 pos = p0; float w = 1.0; vec3 c = (uJulia > 0.5) ? uJc : p0;
  vec3 col = vec3(0.0); float wsum = 0.0;
  for (int i = 0; i < ${MAX_ITERS}; i++) {
    if (i >= uIters) break;
    ${step}
    float d = length(pos);
    float ti = fract(d * 0.35 + float(i) * 0.03); // per-iteration palette coordinate
    float wt = exp(-1.5 * d);                      // trap-proximity weight
    col += wt * palLookupGL(ti);
    wsum += wt;
    if (dot(pos, pos) > uBailout) break;
  }
  return col / max(wsum, 1e-4);
}`;
}

export function buildFragGL(ops, extraSlots, emit, { envx = false } = {}) {
  const { body, paramCount } = iterBodyGL(ops);
  // `extraSlots` (HYBRID_NSLOT_SPEC.md §2.4) — an array of the hybrid's slots
  // BEYOND A ([{ ops }, …]); falsy = flat. Slot A is `ops`; the extras codegen
  // one iterStepK each, chaining their uP[] param bases after A's.
  const hybrid = Array.isArray(extraSlots) && extraSlots.length > 0;
  const nSlots = hybrid ? 1 + extraSlots.length : 1;
  // Hybrid moves uHyb[]/uAddGate[] into the std140 Bulk block (arrays); the flat
  // path keeps its single addC gate as a default-block scalar.
  let core;
  if (hybrid) {
    // One iterStepK per slot, param bases chained (A first, then B, C…) so all
    // slots share the single uP[] array — the WebGL2 mirror of the WebGPU tier's
    // concatenated op buffer.
    let base = paramCount;
    const stepDefs = [
      `void iterStep0(inout vec3 pos, inout float w, int i) {\n  g_wq = 1.0;\n${body}}`,
    ];
    extraSlots.forEach((slot, k) => {
      const { body: b, paramCount: pc } = iterBodyGL(slot.ops || [], base);
      base += pc;
      stepDefs.push(
        `void iterStep${k + 1}(inout vec3 pos, inout float w, int i) {\n  g_wq = 1.0;\n${b}}`,
      );
    });
    const periodExpr = Array.from(
      { length: nSlots },
      (_, k) => `uHyb[${k}]`,
    ).join(" + ");
    const chain = Array.from(
      { length: nSlots },
      (_, k) =>
        `  acc += uHyb[${k}]; if (ph < acc) { iterStep${k}(pos, w, i); if (uAddGate[${k}] == 1) pos += c; return; }`,
    ).join("\n");
    core = `
${stepDefs.join("\n")}
// N-slot schedule walk — mirrors shader.js hybWalk / cpuorbit.hybridSlotAt (same
// period/phase math), baked as a GLSL chain (GLSL can't index functions).
void hybStep(inout vec3 pos, inout float w, inout vec3 c, int i) {
  int period = ${periodExpr};
  int ph = period > 0 ? (i % period) : 0; // GLSL ES 3.00 (#version 300 es) has int %
  int acc = 0;
${chain}
}

float mapDE(vec3 p_rel) {
  vec3 p0 = p_rel + uOffset;
  vec3 pos = p0; float w = 1.0;
  vec3 c = (uJulia > 0.5) ? uJc : p0;
  for (int i = 0; i < ${MAX_ITERS}; i++) {
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
  for (int i = 0; i < ${MAX_ITERS}; i++) {
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
  for (int i = 0; i < ${MAX_ITERS}; i++) {
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
  for (int i = 0; i < ${MAX_ITERS}; i++) {
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
  for (int i = 0; i < ${MAX_ITERS}; i++) {
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
  for (int i = 0; i < ${MAX_ITERS}; i++) {
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
  for (int i = 0; i < ${MAX_ITERS}; i++) {
    if (i >= uIters) break;
    hybStep(pos, w, c, i);
    if (dot(pos, pos) > uBailout) { esc = i; rEsc = length(pos); break; }
  }
  return (float(esc) + smoothEscFrac(rEsc, uBailout)) / float(max(uIters, 1)); // S1 smooth bands
}`;
  } else {
    // Emit the numeric-DE probe only when this op-list actually contains a
    // W_BULB_NUMERIC op — the GLSL program recompiles per structural edit
    // anyway, and analytic formulas shouldn't carry (or pay register pressure
    // for) a dead orbitR + branch (mirrors the WGSL variant split).
    const wantsNumeric = ops.some(
      (op) => byKey(op.key)?.wRule === W_BULB_NUMERIC,
    );
    const numericChunk = !wantsNumeric
      ? ""
      : `
// Orbit escape radius only — the numeric-DE probe (deOption 3, mirrors
// shader.js orbitR). Sampled 4× per DE for the finite-difference gradient.
float orbitR(vec3 p0) {
  vec3 pos = p0; float w = 1.0;
  vec3 c = (uJulia > 0.5) ? uJc : p0;
  for (int i = 0; i < ${MAX_ITERS}; i++) {
    if (i >= uIters) break;
    iterStep(pos, w, i);
    if (uAddGate == 1) pos += c;
    if (dot(pos, pos) > uBailout) break;
  }
  return length(pos);
}
`;
    const numericBranch = !wantsNumeric
      ? ""
      : `
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
  for (int i = 0; i < ${MAX_ITERS}; i++) {
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
  for (int i = 0; i < ${MAX_ITERS}; i++) {
    if (i >= uIters) break;
    iterStep(pos, w, i);
    if (uAddGate == 1) pos += c;
    tr = min(tr, length(pos));
    if (dot(pos, pos) > uBailout) break;
  }
  return tr;
}

// COLORING R S8 — IFS address (sign-octant of the final orbit point), non-hybrid twin.
float orbitAddressGL(vec3 p_rel) {
  vec3 p0 = p_rel + uOffset;
  vec3 pos = p0; float w = 1.0; vec3 c = (uJulia > 0.5) ? uJc : p0;
  for (int i = 0; i < ${MAX_ITERS}; i++) {
    if (i >= uIters) break;
    iterStep(pos, w, i);
    if (uAddGate == 1) pos += c;
    if (dot(pos, pos) > uBailout) break;
  }
  float oct = float(int(pos.x > 0.0) + 2 * int(pos.y > 0.0) + 4 * int(pos.z > 0.0));
  return (oct + 0.5) / 8.0;
}

// COLORING S2 — Silk (stripe average), non-hybrid twin (mirrors orbitSilk above).
float orbitSilk(vec3 p_rel) {
  vec3 p0 = p_rel + uOffset;
  vec3 pos = p0; float w = 1.0; vec3 c = (uJulia > 0.5) ? uJc : p0;
  float k = max(uStripeFreq, 1.0); float acc = 0.0; float cnt = 0.0;
  for (int i = 0; i < ${MAX_ITERS}; i++) {
    if (i >= uIters) break;
    iterStep(pos, w, i);
    if (uAddGate == 1) pos += c;
    acc += 0.5 + 0.5 * sin(k * atan(pos.y, pos.x));
    cnt += 1.0;
    if (dot(pos, pos) > uBailout) break;
  }
  return acc / max(cnt, 1.0);
}

// COLORING S3 — Pinwheel (trap angle), non-hybrid twin (mirrors orbitPin above).
float orbitPin(vec3 p_rel) {
  vec3 p0 = p_rel + uOffset;
  vec3 pos = p0; float w = 1.0; vec3 c = (uJulia > 0.5) ? uJc : p0;
  float tr = 1e9; float ang = 0.0;
  for (int i = 0; i < ${MAX_ITERS}; i++) {
    if (i >= uIters) break;
    iterStep(pos, w, i);
    if (uAddGate == 1) pos += c;
    float r2 = dot(pos, pos);
    if (r2 < tr) { tr = r2; ang = atan(pos.y, pos.x); }
    if (r2 > uBailout) break;
  }
  return fract(ang * 0.15915494 + 0.5); // 1/(2π); [-π,π] → [0,1)
}

// COLORING P3 — iridescence (S6), non-hybrid twin (mirrors orbitIrid above).
float orbitIrid(vec3 p_rel) {
  vec3 p0 = p_rel + uOffset;
  vec3 pos = p0; float w = 1.0; vec3 c = (uJulia > 0.5) ? uJc : p0;
  vec3 mn = vec3(1e9);
  for (int i = 0; i < ${MAX_ITERS}; i++) {
    if (i >= uIters) break;
    iterStep(pos, w, i);
    if (uAddGate == 1) pos += c;
    mn = min(mn, abs(pos));
    if (dot(pos, pos) > uBailout) break;
  }
  return (mn.x - mn.z) / (mn.x + mn.z + 1e-6); // scale-invariant asymmetry
}

float escapeIter(vec3 p_rel) {
  vec3 p0 = p_rel + uOffset; // deep zoom §3.2
  vec3 pos = p0; float w = 1.0; vec3 c = (uJulia > 0.5) ? uJc : p0; int esc = uIters; float rEsc = 0.0;
  for (int i = 0; i < ${MAX_ITERS}; i++) {
    if (i >= uIters) break;
    iterStep(pos, w, i);
    if (uAddGate == 1) pos += c;
    if (dot(pos, pos) > uBailout) { esc = i; rEsc = length(pos); break; }
  }
  return (float(esc) + smoothEscFrac(rEsc, uBailout)) / float(max(uIters, 1)); // S1 smooth bands
}`;
  }
  // Iteration/coloring uniforms — the flat/hybrid-specific middle of the default
  // uniform block: the single addC gate (flat only; hybrid's per-slot uHyb[]/
  // uAddGate[] ride the Bulk block) plus the palette/orbit coloring scalars.
  // uPalStops (an array) also lives in the Bulk block now — see bulkLayout().
  const variantUniforms = [
    `uniform int uIters, uMaxSteps, uColorMode;`,
    hybrid ? null : `uniform int uAddGate;`,
    `uniform float uBailout, uEps, uDeScale, uDeOption;`,
    `uniform vec3 uColA, uColB, uBg, uJc;`,
    `uniform float uJulia;`,
    `uniform vec3 uPalA, uPalB, uPalC, uPalD;`,
    `uniform float uPalOn;`,
    `uniform float uPalCount, uPalCyclic;`,
    `uniform float uStripeFreq;`,
    `uniform float uSigLo, uSigSpan;`,
    `uniform float uIridescence, uPalettePhase;`,
  ]
    .filter(Boolean)
    .join("\n");
  // Standalone export (#291): `emit` swaps the whole uniform surface for baked
  // consts (camera/view stay live) — NO Bulk block (uP/uPalStops bake to const
  // arrays). Engine path (emit undefined) declares the scalars + the std140 Bulk
  // block; the golden fixtures pin it. Only the flat path is bakeable; the
  // exporter guards hybrid out, so an emit + extraSlots combination never occurs.
  const uniformSection = emit
    ? bakeUniformBlock(emit, { envx })
    : `${commonUniformsGL(variantUniforms, envx)}

${bulkBlockGL(bulkLayout({ hybrid: hybrid ? nSlots : 0 }))}`;
  return `${FRAG_PRELUDE_GL}

${uniformSection}

float g_wq; // desktop 4D scratch the shared op bodies write to; unused here
${core}

${shadeLibGL(envx)}

${painterLibGL(hybrid)}

${mainGL(ALBEDO_FLAT_GL, envx)}`;
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
export function buildSceneFragGL(objects, { envx = false } = {}) {
  const layout = sceneParamLayout(objects);
  const norm = objects.map(normalizeSceneObject);

  // Leaf SDFs (registry-sourced, D0 §2.2) — emitted once each, only the ids
  // this scene actually uses (the codegen is per-scene anyway).
  const usedIds = [
    ...new Set(norm.map((n) => n.shapeId).filter((id) => id > 0)),
  ];
  let funcs = "";
  for (const l of LEAVES) {
    if (!usedIds.includes(l.id)) continue;
    funcs += `
float leaf_${l.key}(vec3 p, vec4 prm) {
  ${l.glsl}
}`;
  }

  // Per-object DE functions — ONE unified template (op chain + leaf finalize,
  // spec §2.2.1). Pure leaves bake to a single call (the loop and w fold away
  // at codegen time); the structural signature (renderer_gl writeScene) covers
  // shapeId/iterShape so a shape change relinks like any structure change.
  for (let k = 0; k < norm.length; k++) {
    const n = norm[k];
    const leaf = n.shapeId > 0 ? LEAVES.find((l) => l.id === n.shapeId) : null;
    const leafCall = leaf ? `leaf_${leaf.key}(pos, uObjPrimP[${k}])` : null;
    // COLORING P3 S5 — per-object orbit signal (glow/bands/silk), mirror of
    // shader.js objOrbitSignal. Emitted for EVERY object (even pure leaves —
    // the empty op loop still tracks the trap from the addC walk, matching WGSL).
    const { body: orbitBody } = iterBodyGL(n.ops, layout[k].slotBase);
    funcs += `
float objOrbit_${k}(vec3 p0, int mode) {
  vec3 pos = p0; float w = 1.0;
  vec3 c = (uObjJulia[${k}] > 0.5) ? uObjJc[${k}] : p0;
  int it = uObjIters[${k}];
  float sk = max(uStripeFreq, 1.0);
  float tr = 1e9; int esc = it; float rEsc = 0.0; float acc = 0.0; float cnt = 0.0;
  for (int i = 0; i < ${MAX_ITERS}; i++) {
    if (i >= it) break;
    g_wq = 1.0;
${orbitBody}    if (uObjAddGate[${k}] == 1) pos += c;
    tr = min(tr, length(pos));
    acc += 0.5 + 0.5 * sin(sk * atan(pos.y, pos.x));
    cnt += 1.0;
    float r2 = dot(pos, pos);
    if (r2 > uBailout) { esc = i; rEsc = sqrt(r2); break; }
  }
  if (mode == 7) { // COLORING R S8 — sign-octant of the final orbit point
    float oct = float(int(pos.x > 0.0) + 2 * int(pos.y > 0.0) + 4 * int(pos.z > 0.0));
    return (oct + 0.5) / 8.0;
  }
  if (mode == 3) return acc / max(cnt, 1.0);
  if (mode == 2) return (float(esc) + smoothEscFrac(rEsc, uBailout)) / float(max(it, 1));
  return clamp(tr / 1.5, 0.0, 1.0);
}`;
    if (!n.ops.length && leaf && !n.iterShape) {
      // Pure leaf: identical to the legacy primitive codegen (w = 1).
      funcs += `
float objDE_${k}(vec3 p0) {
  vec3 pos = p0;
  return ${leafCall};
}`;
      continue;
    }
    const { body } = iterBodyGL(n.ops, layout[k].slotBase);
    const iterMin =
      n.iterShape && leaf
        ? `    dmin = min(dmin, ${leafCall} / max(abs(w), 1e-9));\n`
        : "";
    const finalDE =
      n.iterShape && leaf
        ? `return dmin;`
        : leaf
          ? `return ${leafCall} / max(abs(w), 1e-9);`
          : `return r / max(abs(w), 1e-9);`;
    funcs += `
float objDE_${k}(vec3 p0) {
  vec3 pos = p0; float w = 1.0;${n.iterShape && leaf ? " float dmin = 1.0e9;" : ""}
  vec3 c = (uObjJulia[${k}] > 0.5) ? uObjJc[${k}] : p0;
  int it = uObjIters[${k}];
  for (int i = 0; i < ${MAX_ITERS}; i++) {
    if (i >= it) break;
    g_wq = 1.0;
${body}    if (uObjAddGate[${k}] == 1) pos += c;
${iterMin}    if (dot(pos, pos) > uBailout) break;
  }
  float r = length(pos);
  if (uObjDeOption[${k}] < 1.0) return 0.5 * log(max(r, 1e-9)) * r / max(abs(w), 1e-9);
  ${finalDE}
}`;
  }

  // Scene mapDE — combine over objects (unrolled). Mirrors shader.js mapDE():
  // pk = qrot(conj(q), p−origin)/uscale; dk = objDE_k(pk)·uscale; combine.
  // mapBody = scene DE (combine over objects). tintBody mirrors it but tracks the
  // winning object id (§3.8 per-object color) — both share the same placement so
  // they can't drift. Mirrors shader.js mapDE() / sceneTint().
  let mapBody = "  float d = 1.0e9;\n";
  let tintBody = "  float d = 1.0e9; int win = 0;\n";
  for (let k = 0; k < norm.length; k++) {
    // subtract/intersect use (smooth-)max() — not a valid distance bound (oversteps
    // the carved walls); the scene compensates with a tighter deScale (preview.js).
    // Object 0 is the BASE (combine forced to union) — mirrors shader.js mapDE:
    // subtract/intersect against the empty accumulator are degenerate and a
    // first-object carve blanked the whole scene on reorder.
    const combine = k === 0 ? 0 : norm[k].combine;
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
  // tintBody now ends after the combine walk with `win` set; sceneTintGL and
  // sceneOrbitGL (COLORING P3 S5) both reuse it, then diverge on the finalize.
  // orbitSwitch: place the hit into the winning object's local space and run its
  // orbit (GLSL has no function pointers → an unrolled if-chain on `win`).
  let orbitSwitch = "";
  for (let k = 0; k < norm.length; k++) {
    orbitSwitch += `  if (win == ${k}) {
    vec4 q = uObjQuat[${k}]; vec4 qi = vec4(-q.x, -q.y, -q.z, q.w);
    vec3 pk = qrotGL(qi, p0 - uObjOrigin[${k}]) / uObjUscale[${k}];
    return objOrbit_${k}(pk, mode);
  }
`;
  }

  // Iteration/coloring uniforms — the scene-specific middle of the uniform
  // block: only what the emitted scene source references. Scenes are
  // surface-mode (§3.8) — no palette/orbit coloring — and iteration plumbing
  // (iters/addC gate/julia/deOption) rides the per-object uObj* arrays, so the
  // flat builder's uIters/uAddGate/uColorMode/uDeOption/uColA/uJc/uJulia/uPal*
  // are never read here and aren't declared (they were inactive uniforms —
  // compiled out, null locations, renderer_gl uploads already no-ops).
  const variantUniforms = `uniform int uMaxSteps, uColorMode;
uniform float uBailout, uEps, uDeScale, uPalOn;
uniform vec3 uColA, uColB, uBg;
uniform vec3 uPalA, uPalB, uPalC, uPalD;
// uPalStops rides the std140 Bulk block (below) so the shared SHADE_LIB_GL's
// albedoStopsGL compiles here too; the scene albedo path keeps per-object tint /
// cosine (N-stop is flat-only in P0), so it stays an inactive member (its
// renderer_gl upload is a no-op for scenes).
uniform float uPalCount, uPalCyclic;
uniform float uSigLo, uSigSpan;
uniform float uPalettePhase;
uniform float uStripeFreq;`;
  return `${FRAG_PRELUDE_GL}

${commonUniformsGL(variantUniforms, envx)}

// CSG scene per-object arrays + op params + palette stops (cap ${MAX_OBJECTS}
// objects). They ride a std140 uniform BUFFER, not the default uniform block —
// on a GLES-3.0-minimum device the per-object arrays + uP[${MAX_PARAMS}] overflowed the
// default block (MAX_FRAGMENT_UNIFORM_VECTORS ≥ 224) and the tier failed to
// LINK. Structure is codegen'd; these per-object scalars ride the buffer so a
// value tweak doesn't force a recompile. See bulkLayout() for the shared layout.
${bulkBlockGL(bulkLayout({ scene: true }))}

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
${tintBody}  return uObjColor[win];
}

// COLORING P3 S5 — the winning object's own orbit signal (glow/bands/silk).
// Mirrors shader.js sceneOrbit(): same combine walk, then run the winner's orbit.
float sceneOrbitGL(vec3 p_rel, int mode) {
  vec3 p0 = p_rel + uOffset;
${tintBody}${orbitSwitch}  return 0.5;
}

${shadeLibGL(envx)}

${mainGL(ALBEDO_SCENE_GL, envx)}`;
}

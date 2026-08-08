// Standalone GLSL export (#291, milestone M0). Packages the WebGL2 raymarcher
// (shader_gl.js) into a self-contained GLSL ES 3.0 fragment shader that runs
// OUTSIDE the engine: camera/view stay uniforms, everything else is baked const.
// Distinct from shader.js's exportGLSL (the desktop iterateJIT_ fragment).
//
// See docs/planning/EXPORT_M0_STANDALONE_GLSL.md. M0 = FLAT formulas only;
// hybrid/scene are guarded out (deferred).
import { activeOps } from "./operators.js";
import { buildFragGL, UNIFORM_TABLE_FLAT } from "./shader_gl.js";
import { deriveFrameParams } from "./frameparams.js";
import { computeRecenter } from "./recenter.js";
import { packOpParams } from "./uniformPack.js";
import { glslNum, glslVec, glslArr } from "./glslfmt.js";

export const STANDALONE_GLSL_VERSION = 1;

// Null-safe, verbatim copies of evaluate.js:127/176 (unexported there).
const isScene = (f) => Array.isArray(f?.objects) && f.objects.length > 0;
const isHybrid = (f) => !!f?.hybrid && !isScene(f);

// CAMERA/VIEW — kept as live uniforms in the export (single source of the keep
// list; every host drives the view). Everything else in UNIFORM_TABLE_FLAT bakes.
export const TAXONOMY = {
  keep: new Set([
    "uRes",
    "uFov",
    "uCamPos",
    "uCamFwd",
    "uCamRight",
    "uCamUp",
    "uOffset",
    "uNear",
    "uFar",
  ]),
};

// Baked uniform → field in the merged value bag. Names don't match 1:1 (uJc←
// juliaC, uKeyC←keyColor, the light rig flattens), so the map is explicit. uP is
// attached separately (packOpParams). If a baked uniform is MISSING here the
// exporter throws rather than silently baking 0 (the blank-render trap).
const UNIFORM_SOURCES = {
  uOrthoH: "orthoH", // #441 — always 0 in a bundle (see bakeValues)
  uIters: "iters",
  uMaxSteps: "maxSteps",
  uColorMode: "colorMode",
  uAddGate: "addGate",
  uBailout: "bailout",
  uEps: "eps",
  uDeScale: "deScale",
  uDeOption: "deOption",
  uColA: "colA",
  uColB: "colB",
  uBg: "bg",
  uJc: "juliaC",
  uJulia: "julia",
  uPalA: "palA",
  uPalB: "palB",
  uPalC: "palC",
  uPalD: "palD",
  uPalOn: "palOn",
  uPalStops: "palStops",
  uPalCount: "palStopCount",
  uPalCyclic: "palCyclic",
  uStripeFreq: "stripeFreq",
  uSigLo: "sigLo",
  uSigSpan: "sigSpan",
  uIridescence: "iridescence",
  uPalettePhase: "palettePhase",
  uLightDir: "lightDir",
  uAmbient: "ambient",
  uRim: "rim",
  uGloss: "gloss",
  uIntensity: "intensity",
  uKeyC: "keyColor",
  uFillDir: "fillDir",
  uFillC: "fillColor",
  uBackDir: "backDir",
  uBackC: "backColor",
  uMetallic: "metallic",
  uShadowK: "shadowK",
  uShadowOn: "shadowOn",
  uAoStr: "ao",
  uFill: "fill",
  uBack: "back",
  uSky: "sky",
  uSunGlow: "sunGlow",
  uGround: "ground",
  uIbl: "ibl",
  uFogAmt: "fog",
  uInScatter: "inScatter",
  uExposure: "exposure",
  uP: "uP",
};

// Format a raw value as a GLSL literal for the declared type. int → bare integer
// (glslNum would wrongly add ".0"); float → glslNum; vec3 → glslVec; arrays →
// glslArr (padded to `array` so an empty palette can't emit vec4[8]()).
function fmt(type, value, array) {
  if (array) {
    if (type === "vec4")
      return glslArr("vec4", array, value, (s) => glslVec("vec4", s));
    return glslArr(type, array, value); // uP: float[MAX_PARAMS]
  }
  if (type === "int") return String(Math.trunc(Number(value) || 0));
  if (type === "vec3") return glslVec("vec3", value || [0, 0, 0]);
  if (type === "vec4") return glslVec("vec4", value || [0, 0, 0, 0]);
  return glslNum(value); // float
}

// Merge the three value sources (see the plan's "Baked-uniform value sources"):
//  1. deriveFrameParams(state) — coloring/palette/light-rig subset
//  2. raw state (writeGlobals payload) — iters/maxSteps/bailout/eps (NOT in #1)
//  3. packOpParams(ops) — uP[]
function bakeValues(state, ops) {
  return {
    ...state,
    ...deriveFrameParams(state),
    // #441 — a standalone bundle always bakes PERSPECTIVE. Orthographic is a
    // transient inspection mode in the studio, not part of a look, and a
    // shareable artifact should not inherit it. (The bake guard below would
    // throw without a source, which is how this got noticed rather than
    // shipping a bundle with an undefined uniform.)
    orthoH: 0,
    uP: packOpParams(ops), // ops already muted-filtered by the caller
  };
}

// The `emit` strategy buildFragGL's bakeUniformBlock consults: null → keep live.
function bakeStrategy(vals) {
  return {
    valueFor(name, type, array) {
      if (TAXONOMY.keep.has(name)) return null; // camera/view stays a uniform
      const field = UNIFORM_SOURCES[name];
      if (!field)
        throw new Error(
          `exportStandalone: baked uniform ${name} has no value source ` +
            `(add it to UNIFORM_SOURCES or TAXONOMY.keep)`,
        );
      return fmt(type, vals[field], array);
    },
  };
}

function standaloneHeader() {
  return `// STANDALONE_GLSL_VERSION: ${STANDALONE_GLSL_VERSION}
// Fractbox standalone GLSL raymarcher (#291) — self-contained, runs OUTSIDE the
// engine (Shadertoy/TouchDesigner/Godot/UE-via-Compushady). NOT the desktop
// iterateJIT_ fragment (exportGLSL). Supply uRes, uFov, uCamPos/Fwd/Right/Up,
// uOffset, uNear, uFar as uniforms; everything else is baked. Entry: main()→fragColor.
// Anti-aliased via per-pixel supersampling (#define FBX_AA below — raise for a
// cleaner still). Fidelity: GL tier — no DOF / bloom / HDR-post (WebGPU-only).`;
}

// ── Anti-aliasing (shared by every export target) ────────────────────────────
// Single-pass exports have no frame accumulation (the app's smoothing), so a
// marginal DE reads as all-over grain, not just aliased edges. Each target turns
// its per-pixel render into a vec4-returning `fbxRender(fragCoord)` and averages
// an FBX_AA×FBX_AA jittered grid — reproducing accumulation in one pass.
const AA_DEFINE = `// Anti-aliasing: FBX_AA×FBX_AA samples/pixel (1 = off). Raise to 3–4 for a clean
// final still — the fractbox app gets this smoothness from frame accumulation.
#define FBX_AA 2`;

// Loop body that accumulates fbxRender(<coord> + off) into `aaCol`.
const aaAccum = (coord) => `  vec3 acc = vec3(0.0);
  for (int j = 0; j < FBX_AA; j++)
    for (int i = 0; i < FBX_AA; i++) {
      vec2 off = (vec2(float(i), float(j)) + 0.5) / float(FBX_AA) - 0.5;
      acc += fbxRender(${coord} + off).rgb;
    }
  vec3 aaCol = acc / float(FBX_AA * FBX_AA);`;

// Convert the standalone body's fullscreen `void main()` into a vec4-returning
// `fbxRender(fragCoord)` (both fragColor writes → returns; gl_FragCoord → the arg).
function renderAsFunction(g) {
  return g
    .replace("void main() {", "vec4 fbxRender(in vec2 fragCoord) {")
    .replace(
      "fragColor = vec4(l2s(tone3(skyOut * exp2(uExposure))), 1.0); return;",
      "return vec4(l2s(tone3(skyOut * exp2(uExposure))), 1.0);",
    )
    .replace("fragColor = vec4(col, 1.0);", "return vec4(col, 1.0);")
    .replaceAll("gl_FragCoord.xy", "fragCoord");
}

// Raw self-contained shader (camera as uniforms, `void main()`) — the base every
// target builds on. Internal: bakedCameraGLSL slices this before re-wrapping,
// and exportStandaloneGLSL AA-wraps it.
// Raw body only — starts with `#version 300 es`, NO leading header comment.
// `#version` MUST be the first token in an ES shader (strict parsers —
// glslangValidator, some WebGL2 drivers — reject even comments/whitespace before
// it), so any header is injected AFTER the directive by the caller.
function standaloneRaw(formula, state) {
  const ops = activeOps(formula);
  const emit = bakeStrategy(bakeValues(state || {}, ops));
  return buildFragGL(ops, null, emit);
}

// Insert a header comment block right after the version directive (never before).
const afterVersion = (glsl, version, header) =>
  glsl.replace(`${version}\n`, `${version}\n${header}\n`);

// formula: an op-list Formula; state: the live globals payload
// (preview.getLastFrameState()). Returns a self-contained, ANTI-ALIASED GLSL ES
// 3.0 shader, or a one-line comment stub for the deferred hybrid/scene cases.
export function exportStandaloneGLSL(formula, state) {
  if (isHybrid(formula) || isScene(formula))
    return "// Standalone export: hybrid/scene not yet supported (M0 flat-only) — #291\n";
  const raw = afterVersion(
    standaloneRaw(formula, state),
    "#version 300 es",
    standaloneHeader(),
  );
  const g = renderAsFunction(raw);
  return `${g}

${AA_DEFINE}
void main() {
${aaAccum("gl_FragCoord.xy")}
  fragColor = vec4(aaCol, 1.0);
}`;
}

// ── M1 — Shadertoy target (#291) ─────────────────────────────────────────────
// Shadertoy has NO custom uniforms, so the standalone shader's kept CAMERA/VIEW
// uniforms must be baked too (a snapshot of the current view — live orbit via
// iMouse is deferred). Derived from state.cam exactly as renderer_gl.applyUniforms
// does (basis + computeRecenter), so the Shadertoy view matches the app.
function cameraConsts(state) {
  const cam = state && state.cam;
  if (!cam || typeof cam.basis !== "function") return null; // no live camera
  const b = cam.basis();
  const { O, roRel } = computeRecenter(b.eye, cam.target, false); // flat → not scene
  const d = deriveFrameParams(state);
  return {
    uFov: glslNum(cam.fov),
    uCamPos: glslVec("vec3", roRel),
    uOffset: glslVec("vec3", O),
    uCamFwd: glslVec("vec3", b.fwd),
    uCamRight: glslVec("vec3", b.right),
    uCamUp: glslVec("vec3", b.up),
    uNear: glslNum(d.tNear),
    uFar: glslNum(d.tFar),
  };
}

function shadertoyHeader() {
  return `// Fractbox → Shadertoy (#291 M1). Paste into a new Shadertoy shader.
// Self-contained: params, coloring, light AND camera are baked (a snapshot of
// the fractbox view — live orbit via iMouse is a later pass). Resolution tracks
// iResolution. Anti-aliased via per-pixel supersampling (#define FBX_AA below —
// raise for a cleaner still). Fidelity: GL tier — no DOF / bloom / HDR-post.\n`;
}

// The standalone shader from `#version` onward, with the CAMERA/VIEW uniforms
// baked to consts (Shadertoy/UE have no fractbox uniform binding). Returns null
// when there's no live camera to bake. Shared by the Shadertoy + Compushady rewraps.
function bakedCameraGLSL(formula, state) {
  const cam = cameraConsts(state);
  if (!cam) return null;
  let g = standaloneRaw(formula, state); // raw base — starts with #version, no header
  for (const [name, lit] of Object.entries(cam)) {
    g = g.replace(
      new RegExp(`uniform (\\w+) ${name};`),
      `const $1 ${name} = ${lit};`,
    );
  }
  return g;
}

// Transform the standalone shader into a Shadertoy `mainImage` shader. The kept
// camera uniforms become baked consts; uRes → iResolution; the GLSL-ES headers
// Shadertoy supplies itself are stripped; the entry becomes mainImage. Returns a
// stub for hybrid/scene / when there is no live camera to bake.
export function exportShadertoy(formula, state) {
  if (isHybrid(formula) || isScene(formula))
    return "// Shadertoy export: hybrid/scene not yet supported (flat-only) — #291\n";
  let g = bakedCameraGLSL(formula, state);
  if (!g)
    return "// Shadertoy export needs a live GPU preview (camera) — render first. #291\n";
  // Resolution is Shadertoy's iResolution; strip the headers Shadertoy provides.
  g = g
    .replace("uniform vec2 uRes;", "#define uRes iResolution.xy")
    .replace("#version 300 es\n", "")
    .replace("precision highp float;\n", "")
    .replace("out vec4 fragColor;\n", "");
  // Shared: render → fbxRender(fragCoord); mainImage supersamples it (FBX_AA).
  g = renderAsFunction(g);
  g += `

${AA_DEFINE}
void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
${aaAccum("fragCoord")}
  fragColor = vec4(aaCol, 1.0);
}`;
  return shadertoyHeader() + g;
}

// ── M2 — Compushady / Unreal (GLSL compute, no transpiler) ───────────────────
// Compushady (github.com/rdeioris/CompushadyUnreal) compiles GLSL directly, so
// the SAME body runs in UE with only an entry/output rewrap: fragment main() →
// a compute shader that imageStore()s into a UAV texture Compushady blits. This
// is EXPERIMENTAL — it targets desktop GLSL 4.60 (glslang), which this repo can't
// compile in CI; validate in UE per docs/planning/EXPORT_M2_COMPUSHADY_UE.md.
function compushadyHeader() {
  return `// Fractbox → Unreal via Compushady (#291 M2) — EXPERIMENTAL, unverified in CI.
// A GLSL 4.60 COMPUTE shader: dispatch over the output texture; the raymarch/
// shade body is shared with the standalone export, camera baked as a snapshot.
// Setup (RenderTarget2D, dispatch, screen material): see
// docs/planning/EXPORT_M2_COMPUSHADY_UE.md. Anti-aliased (#define FBX_AA).
// Fidelity: GL tier (no DOF / bloom / HDR-post).\n`;
}

export function exportCompushady(formula, state) {
  if (isHybrid(formula) || isScene(formula))
    return "// Compushady export: hybrid/scene not yet supported (flat-only) — #291\n";
  let g = bakedCameraGLSL(formula, state);
  if (!g)
    return "// Compushady export needs a live GPU preview (camera) — render first. #291\n";
  // Desktop GLSL for glslang; strip the ES precision/out (compute writes a UAV).
  // uRes becomes the output image size, declared as a global #define BEFORE
  // fbxRender uses it (fbxOut must precede the #define's imageSize call).
  g = g
    .replace(
      "#version 300 es\n",
      "#version 460\n" +
        "layout(rgba8, binding = 0) uniform writeonly image2D fbxOut;\n" +
        "#define uRes vec2(imageSize(fbxOut))\n",
    )
    .replace("precision highp float;\n", "")
    .replace("out vec4 fragColor;\n", "")
    .replace("uniform vec2 uRes;\n", ""); // uRes is the #define above now
  // Shared: render → fbxRender(fragCoord) (uRes global via the #define).
  g = renderAsFunction(g);
  // Compute entry: one invocation per pixel, supersampled → imageStore the UAV.
  g += `

${AA_DEFINE}
layout(local_size_x = 8, local_size_y = 8) in;
void main() {
  ivec2 pc = ivec2(gl_GlobalInvocationID.xy);
  ivec2 sz = imageSize(fbxOut);
  if (pc.x >= sz.x || pc.y >= sz.y) return;
${aaAccum("(vec2(pc) + 0.5)")}
  imageStore(fbxOut, pc, vec4(aaCol, 1.0));
}`;
  // Header AFTER #version (keep the directive the first token, as for standalone).
  return afterVersion(g, "#version 460", compushadyHeader());
}

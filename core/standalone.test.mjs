// Run: node --test core/standalone.test.mjs
// M0 standalone-GLSL export (#291). PR 2 covers the decl/bake seam: the engine
// (emit-undefined) path must stay byte-identical, and the exported uniform table
// must match the engine's actual uniform surface. Exporter-shape assertions land
// in PR 4.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import {
  buildFragGL,
  buildSceneFragGL,
  bakeUniformBlock,
  UNIFORM_TABLE_FLAT,
} from "./shader_gl.js";
import {
  exportStandaloneGLSL,
  exportShadertoy,
  exportCompushady,
  STANDALONE_GLSL_VERSION,
} from "./exportStandalone.js";

const fx = (name) =>
  readFileSync(
    fileURLToPath(new URL(`./__fixtures__/${name}`, import.meta.url)),
    "utf8",
  );

const A = [{ key: "boxFold", values: [1] }];
const B = [{ key: "scale", values: [2] }];
const C = [{ key: "sphereFold", values: [0.5, 1] }];
const SCENE_OBJ = { shapeId: 0, objType: "sphere", combine: "union" };

// ── Golden: the engine path is byte-for-byte unchanged by the decl seam ───────
test("flat frag shader unchanged (emit undefined)", () => {
  assert.equal(buildFragGL([]), fx("frag_flat.glsl"));
});
test("hybrid (2-slot) frag shader unchanged", () => {
  assert.equal(buildFragGL(A, [{ ops: B }]), fx("frag_hybrid.glsl"));
});
test("hybrid (3-slot) frag shader — N-slot golden", () => {
  assert.equal(
    buildFragGL(A, [{ ops: B }, { ops: C }]),
    fx("frag_hybrid3.glsl"),
  );
});
test("scene frag shader unchanged", () => {
  assert.equal(buildSceneFragGL([SCENE_OBJ]), fx("frag_scene.glsl"));
});

// ── Drift guard: the table IS the flat uniform surface ────────────────────────
// Parse every uniform NAME from the real engine block — both the plain
// default-block `uniform T names;` decls AND the std140 uniform-BUFFER block
// members (`uniform Bulk { T name[N]; … };`, where the fat arrays uP/uPalStops
// now live) — and confirm the set equals UNIFORM_TABLE_FLAT, so a uniform added
// to one but not the other fails loudly (the silent-divergence M0 exists to
// prevent). The std140 migration must not drop a name from the export surface.
function uniformNamesOf(src) {
  const names = new Set();
  const clean = src.replace(/\/\/[^\n]*/g, ""); // drop line comments first
  // std140 uniform-BUFFER block members
  for (const blk of clean.matchAll(/uniform\s+\w+\s*\{([^}]*)\}\s*;/g)) {
    for (const member of blk[1].split(";")) {
      const m = member.trim().match(/^\w+\s+([A-Za-z_]\w*)/);
      if (m) names.add(m[1]);
    }
  }
  // plain default-block uniforms (with the blocks removed so they don't re-hit)
  const noBlocks = clean.replace(/uniform\s+\w+\s*\{[^}]*\}\s*;/g, "");
  for (const m of noBlocks.matchAll(/uniform\s+\w+\s+([^;{]+);/g)) {
    for (const decl of m[1].split(",")) {
      names.add(
        decl
          .trim()
          .replace(/\[.*\]$/, "")
          .trim(),
      );
    }
  }
  return names;
}
test("UNIFORM_TABLE_FLAT matches the engine's flat uniform surface", () => {
  const engine = uniformNamesOf(buildFragGL([]));
  const table = new Set(UNIFORM_TABLE_FLAT.map((u) => u.name));
  const missing = [...engine].filter((n) => !table.has(n));
  const extra = [...table].filter((n) => !engine.has(n));
  assert.deepEqual(missing, [], `uniforms in shader but not table: ${missing}`);
  assert.deepEqual(extra, [], `uniforms in table but not shader: ${extra}`);
});

// ── bakeUniformBlock: keep vs bake, driven by the strategy ────────────────────
test("bakeUniformBlock keeps camera uniforms, bakes the rest", () => {
  // Stub strategy: keep camera/view (null), bake everything else to a literal.
  const KEEP = new Set([
    "uRes",
    "uFov",
    "uCamPos",
    "uCamFwd",
    "uCamRight",
    "uCamUp",
    "uOffset",
    "uNear",
    "uFar",
  ]);
  const emit = {
    valueFor: (name, type, array) =>
      KEEP.has(name) ? null : array ? `${type}[${array}](/*…*/)` : `${type}(0)`,
  };
  const out = bakeUniformBlock(emit);
  assert.match(out, /uniform vec3 uCamPos;/); // kept live
  assert.match(out, /const float uExposure = /); // baked
  assert.match(out, /const vec4 uPalStops\[8\] = /); // array baked with size
  assert.doesNotMatch(out, /uniform float uExposure/); // not left as uniform
});

// ── PR 4: the exporter ────────────────────────────────────────────────────────
const FLAT = {
  ops: [
    { key: "boxFold", values: [1] },
    { key: "scale", values: [2] },
  ],
};
const STATE = {
  iters: 17,
  maxSteps: 128,
  bailout: 42,
  eps: 0.001,
  colorMode: 2,
  colA: [0.8, 0.4, 0.1],
  colB: [0.1, 0.6, 0.7],
  bg: [0.05, 0.07, 0.1],
  julia: 0,
  juliaC: [0, 0, 0],
  deScale: 0.85,
  deOption: 2,
  tNear: 0.02,
  tFar: 80,
  sigLo: 0.1,
  sigSpan: 0.8,
  stripeFreq: 5,
  iridescence: 0,
  palettePhase: 0,
  palette: { on: false },
  light: {},
};

test("exports a self-contained shader (not the iterateJIT_ fragment)", () => {
  const glsl = exportStandaloneGLSL(FLAT, STATE);
  assert.match(
    glsl,
    new RegExp(`STANDALONE_GLSL_VERSION: ${STANDALONE_GLSL_VERSION}`),
  );
  assert.match(glsl, /#version 300 es/);
  assert.match(glsl, /void main\(/);
  assert.match(glsl, /uniform vec3 uCamPos;/); // camera KEPT live
  assert.doesNotMatch(glsl, /uniform float uExposure/); // baked, not a uniform
  assert.doesNotMatch(glsl, /iterateJIT_\w*\(/); // not the desktop fragment
  assert.doesNotMatch(glsl, /getGenericParam/);
});

// The blank-render trap: iters/maxSteps/bailout/eps are NOT in deriveFrameParams,
// so if the exporter fed only that they'd bake to 0 → empty render. Assert the
// ACTUAL values survive.
test("value-source guard — iters/bailout/eps bake real values, not 0", () => {
  const glsl = exportStandaloneGLSL(FLAT, STATE);
  assert.match(glsl, /const int uIters = 17;/);
  assert.match(glsl, /const int uMaxSteps = 128;/);
  assert.match(glsl, /const float uBailout = 42\.0;/);
  assert.match(glsl, /const float uEps = 0\.001;/);
  assert.doesNotMatch(glsl, /const int uIters = 0;/);
});

test("every baked uniform resolves (full export never throws)", () => {
  assert.doesNotThrow(() => exportStandaloneGLSL(FLAT, STATE));
});

// #version MUST be the first token — strict ES parsers reject even a comment
// before it (glslangValidator: "must occur first in shader"). The header goes
// AFTER the directive.
test("#version is the first line of the standalone (nothing before it)", () => {
  assert.equal(
    exportStandaloneGLSL(FLAT, STATE).split("\n")[0],
    "#version 300 es",
  );
});

// uPalStops must always emit 8 initializers — AND each must be a full vec4 with
// components. vec4[8]() (empty array) OR vec4() (empty element) both fail to
// compile ("constructor does not have any arguments").
test("palette-off bakes uPalStops padded to 8 full vec4s, no empty constructors", () => {
  const glsl = exportStandaloneGLSL(FLAT, STATE); // palette.on false → 0 stops
  assert.doesNotMatch(glsl, /vec4\[8\]\(\)/); // no empty array
  assert.doesNotMatch(glsl, /vec[234]\(\)/); // no empty element constructor (the #291 bug)
  const m = glsl.match(/const vec4 uPalStops\[8\] = vec4\[8\]\((.*?)\);/s);
  assert.ok(m, "uPalStops[8] initializer present");
  assert.equal(m[1].split("vec4(").length - 1, 8); // exactly 8 vec4 initializers
});

// No export target may ever emit an argument-less vecN() constructor.
test("no export emits an empty vecN() constructor (any palette state)", () => {
  for (const st of [STATE, { ...STATE, palette: { on: true, stops: [] } }])
    for (const out of [
      exportStandaloneGLSL(FLAT, st),
      exportShadertoy(FLAT, { ...st, cam: CAM }),
    ])
      assert.doesNotMatch(out, /vec[234]\(\)/);
});

test("≥3-stop palette bakes (dynamic-index path)", () => {
  const state = {
    ...STATE,
    palette: {
      on: true,
      stops: [
        { c: [1, 0, 0], p: 0 },
        { c: [0, 1, 0], p: 0.5 },
        { c: [0, 0, 1], p: 1 },
      ],
    },
  };
  const glsl = exportStandaloneGLSL(FLAT, state);
  assert.match(glsl, /const float uPalCount = 3\.0;/);
  assert.doesNotMatch(glsl, /vec4\[8\]\(\)/);
});

test("hybrid/scene → guard stub, not a broken flat shader", () => {
  assert.match(
    exportStandaloneGLSL({ objects: [{ shapeId: 1 }] }, STATE),
    /not yet supported/,
  );
  assert.match(
    exportStandaloneGLSL({ hybrid: true, ops: [] }, STATE),
    /not yet supported/,
  );
});

// ── M1: Shadertoy target ──────────────────────────────────────────────────────
const CAM = {
  fov: 0.9,
  target: [0, 0, 0],
  basis: () => ({
    eye: [0, 0, 2.6],
    fwd: [0, 0, -1],
    right: [1, 0, 0],
    up: [0, 1, 0],
  }),
};
const STATE_CAM = { ...STATE, cam: CAM };

test("Shadertoy export uses mainImage + iResolution, strips GLSL-ES headers", () => {
  const g = exportShadertoy(FLAT, STATE_CAM);
  assert.match(g, /void mainImage\( out vec4 fragColor, in vec2 fragCoord \)/);
  assert.match(g, /#define uRes iResolution\.xy/);
  assert.doesNotMatch(g, /#version 300 es/);
  assert.doesNotMatch(g, /precision highp float;/);
  assert.doesNotMatch(g, /uniform vec2 uRes;/);
  assert.doesNotMatch(g, /\bvoid main\(\)/);
  assert.doesNotMatch(g, /gl_FragCoord/); // → fragCoord
});

test("Shadertoy export supersamples (FBX_AA over a fbxRender function)", () => {
  const g = exportShadertoy(FLAT, STATE_CAM);
  assert.match(g, /vec4 fbxRender\(in vec2 fragCoord\)/); // render is a function
  assert.match(g, /#define FBX_AA 2/); // tunable AA
  assert.match(g, /acc \+= fbxRender\(fragCoord \+ off\)/); // averaged samples
  // the render's two fragColor writes became returns → only mainImage assigns it
  assert.equal((g.match(/fragColor = /g) || []).length, 1);
});

test("Shadertoy export bakes the camera (no uniforms left) matching the view", () => {
  const g = exportShadertoy(FLAT, STATE_CAM);
  assert.match(g, /const vec3 uCamPos = vec3\(0\.0, 0\.0, 2\.6\);/); // eye−target
  assert.match(g, /const float uFov = 0\.9;/);
  assert.doesNotMatch(g, /uniform vec3 uCamPos;/); // camera no longer a uniform
});

test("Shadertoy export stubs when no live camera / hybrid / scene", () => {
  assert.match(exportShadertoy(FLAT, STATE), /render first/); // STATE has no .cam
  assert.match(
    exportShadertoy({ objects: [{}] }, STATE_CAM),
    /not yet supported/,
  );
});

// ── M2: Compushady / Unreal (GLSL compute) ────────────────────────────────────
test("Compushady export is a GLSL 4.60 compute shader writing a UAV", () => {
  const g = exportCompushady(FLAT, STATE_CAM);
  assert.equal(g.split("\n")[0], "#version 460"); // directive first, header after
  assert.match(g, /layout\(local_size_x = 8, local_size_y = 8\) in;/);
  assert.match(
    g,
    /layout\(rgba8, binding = 0\) uniform writeonly image2D fbxOut;/,
  );
  assert.match(g, /#define uRes vec2\(imageSize\(fbxOut\)\)/); // res = output size
  assert.match(g, /vec4 fbxRender\(in vec2 fragCoord\)/); // shared render fn
  assert.match(g, /#define FBX_AA 2/); // shared AA
  assert.match(g, /imageStore\(fbxOut, pc, vec4\(aaCol, 1\.0\)\)/); // AA'd write
});

test("Compushady export leaves no fragment-stage leftovers", () => {
  const g = exportCompushady(FLAT, STATE_CAM);
  assert.doesNotMatch(g, /#version 300 es/);
  assert.doesNotMatch(g, /precision highp float;/);
  assert.doesNotMatch(g, /out vec4 fragColor;/);
  assert.doesNotMatch(g, /gl_FragCoord/); // → fragCoord param
  assert.doesNotMatch(g, /fragColor =/); // both writes → returns
  assert.doesNotMatch(g, /uniform vec2 uRes;/); // resolution = imageSize
});

// Every target routes through the SAME base + shared AA (fbxRender + FBX_AA loop
// → aaCol). Guards against a fix landing on one target but not the others.
test("all three exports supersample via the shared fbxRender/FBX_AA base", () => {
  for (const g of [
    exportStandaloneGLSL(FLAT, STATE),
    exportShadertoy(FLAT, STATE_CAM),
    exportCompushady(FLAT, STATE_CAM),
  ]) {
    assert.match(g, /vec4 fbxRender\(in vec2 fragCoord\)/);
    assert.match(g, /#define FBX_AA 2/);
    assert.match(g, /acc \+= fbxRender\(.*\+ off\)\.rgb/);
    assert.match(g, /vec4\(aaCol, 1\.0\)/);
  }
});

test("Compushady export stubs for no-camera / hybrid / scene", () => {
  assert.match(exportCompushady(FLAT, STATE), /render first/);
  assert.match(
    exportCompushady({ hybrid: true, ops: [] }, STATE_CAM),
    /not yet supported/,
  );
});

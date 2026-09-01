// Scene coloring (SCENES.md §Coloring, amends CSG §3.8) — Glow/Bands render on
// scenes via orbit-free signals in ALL THREE tiers: WGSL + GLSL emission checks
// and an ASCII-tier smoke (behavioral CPU values are visual-verified; the
// signals are trivial — 0.5+0.5·nz and fract(|p|·0.75)).
// Run: node --test core/scenecolor.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { buildWGSL } from "./shader.js";
import { buildSceneFragGL } from "./shader_gl.js";
import { renderAsciiColored } from "./cpu.js";

const SCENE = {
  name: "t",
  ops: [],
  iters: 8,
  deOption: 2,
  addC: false,
  camera: { yawDeg: 20, pitchDeg: 15, dist: 5, fovDeg: 42 },
  objects: [
    {
      objType: 0,
      shapeId: 2,
      shapeParams: [0.8, 0, 0, 0],
      ops: [],
      iters: 1,
      transform: { origin: [0, 0, 0], uscale: 1, rot: [0, 0, 0] },
      combine: 0,
      blendK: 0,
      color: [0.9, 0.4, 0.2],
    },
  ],
};

test("WGSL: scene albedo — mode-0 tint + P3 S5 per-object orbit signal", () => {
  const src = buildWGSL({ numericDE: false });
  assert.ok(src.includes("albedo = sceneTint(p);"), "mode-0 per-object tint");
  // COLORING P3 S5 — Glow/Bands/Silk now run the winning object's REAL orbit
  // (auto-levelled), replacing the old nz/radial stand-ins.
  assert.ok(
    src.includes("normSig(sceneOrbit(p, G.prm.w))"),
    "per-object scene orbit signal",
  );
  assert.ok(src.includes("fn objOrbitSignal("), "per-object orbit fn emitted");
  assert.ok(src.includes("fn sceneOrbit("), "scene orbit dispatcher emitted");
});

test("GLSL: scene fragment declares palette uniforms + P3 S5 per-object orbit", () => {
  const src = buildSceneFragGL(SCENE.objects);
  for (const n of [
    "uColorMode",
    "uPalOn",
    "uPalA",
    "sceneTintGL(p)",
    // COLORING P3 S5 — real per-object orbit replaces the nz/radial stand-ins.
    "sceneOrbitGL(p, uColorMode)",
    "float sceneOrbitGL(",
    "float objOrbit_0(",
  ])
    assert.ok(src.includes(n), n);
});

test("ASCII: a scene renders under every color mode (bands uses the radial key)", () => {
  for (const mode of [0, 1, 2]) {
    const out = renderAsciiColored(SCENE, {
      cols: 32,
      rows: 16,
      coloring: {
        mode,
        palette: { on: true },
        colA: [0.9, 0.4, 0.2],
        colB: [0.2, 0.5, 0.9],
      },
    });
    assert.ok(
      out && typeof out.text === "string" && out.text.length > 0,
      `mode ${mode}`,
    );
  }
});

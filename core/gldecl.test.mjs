// #708 — decl-vs-reference lint over the EMITTED GLSL of every shipped preset.
//
// The field failure this pins: the fat-leaf scene path emitted leaf calls
// reading uObjPrimP2[k] while the std140 Bulk block declaration omitted the
// member (bulkLayout({scene:true}) without the leafAux predicate), so the two
// shipped fat-leaf presets — City Blocks and Whorl City — failed to COMPILE on
// the whole WebGL2 tier, and the glHealth gate demoted the session to ASCII.
// The GL tier is never compiled in CI, so a use-without-declaration slip is
// invisible to every text-golden test unless checked for EXPLICITLY: this lint
// asserts every u-identifier referenced in the emitted text is declared in the
// SAME text (default-block `uniform` statements, comma lists included, or a
// Bulk std140 block member). Run against the pre-#708 emitter it flags exactly
// City Blocks + Whorl City with ["uObjPrimP2"].
//
// Coverage: all 109 presets (flat / hybrid via the canonical activeHybridSlots
// / scene), each in the default build AND the everything-on look-variant build
// (~20 ms total). Named *.test.mjs so sync skips it.
// Run: node --test core/gldecl.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";
import { buildFragGL, buildSceneFragGL } from "./shader_gl.js";
import { PRESETS } from "./oplist.js";
import { activeHybridSlots } from "./hybridmodel.js";

// Uniform-looking identifiers (the codebase convention: `u` + UpperCamel; `uv`
// and friends never match) referenced but not declared anywhere in the text.
function undeclaredUniforms(glsl) {
  const declared = new Set();
  // Default-block statements — comma lists included
  // (`uniform vec3 uCamPos, uCamFwd, …;`).
  for (const stmt of glsl.matchAll(/\buniform\s+\w+\s+([^;]+);/g))
    for (const n of stmt[1].matchAll(/\bu[A-Z]\w*/g)) declared.add(n[0]);
  // Bulk std140 block members (`  vec4 uObjPrimP2[8];`).
  for (const m of glsl.matchAll(/^\s*(?:float|int|vec[234])\s+(u[A-Z]\w*)\[/gm))
    declared.add(m[1]);
  const bad = new Set();
  for (const ref of glsl.matchAll(/\b(u[A-Z]\w*)\b/g))
    if (!declared.has(ref[1])) bad.add(ref[1]);
  return [...bad];
}

// Build a preset's fragment exactly as renderer_gl's write* paths do.
function buildPreset(p, variant) {
  if (p.objects) return buildSceneFragGL(p.objects, variant);
  if (p.hybrid) {
    // Canonical slot accessor (both the legacy `hybrid.b` and the N-slot
    // `hybrid.slots` shapes) — the same choke point capturesettle reads.
    const { slots } = activeHybridSlots(p);
    return buildFragGL(
      slots[0].ops,
      slots.slice(1).map((s) => ({ ops: s.ops })),
      undefined,
      variant,
    );
  }
  return buildFragGL(p.ops, undefined, undefined, variant);
}

// Every look-driven codegen bit ON at once — the widest uniform surface each
// shape can emit. Scene builders never take neon/thinFilm (V1 flat-only);
// planet+equirect are exclusive, so equirect gets its own leg below.
const ALL_FLAT = {
  envx: true,
  neon: true,
  aurora: true,
  grade: true,
  thinFilm: true,
  clip: true,
  clipJag: true,
  planet: true,
};
const ALL_SCENE = {
  envx: true,
  aurora: true,
  grade: true,
  clip: true,
  clipJag: true,
  planet: true,
};

test("#708 every shipped preset's emitted GLSL declares every uniform it references", () => {
  for (const p of PRESETS) {
    const variants = p.objects
      ? [{}, ALL_SCENE, { equirect: true }]
      : [{}, ALL_FLAT, { equirect: true }];
    for (const v of variants) {
      const bad = undeclaredUniforms(buildPreset(p, v));
      assert.deepEqual(
        bad,
        [],
        `"${p.name}" ${JSON.stringify(v)} references undeclared uniform(s) ` +
          `${JSON.stringify(bad)} — a guaranteed WebGL2 compile failure ` +
          `(the #708 City Blocks / Whorl City class)`,
      );
    }
  }
});

test("#708 the shipped fat-leaf scenes declare AND reference the uObjPrimP2 lane", () => {
  // Belt-and-braces for the exact #708 pair: the lint above would pass if a
  // regression dropped the REFERENCES too (fat leaves silently losing their
  // sp4..sp7 params) — pin that both presets still read the overflow lane.
  for (const name of ["City Blocks", "Whorl City"]) {
    const p = PRESETS.find((x) => x.name === name);
    assert.ok(p, `preset "${name}" missing`);
    const glsl = buildSceneFragGL(p.objects);
    assert.match(
      glsl,
      /uObjPrimP2\[\d+\]/,
      `"${name}" must pass the overflow lane to its fat leaf`,
    );
    assert.match(
      glsl,
      /^\s*vec4 uObjPrimP2\[/m,
      `"${name}" must declare uObjPrimP2 in its Bulk block`,
    );
  }
});

// CINE GRADE — cinematic post looks (IDEAS 2026-08-21 wave).
//
// The load-bearing invariant, same as NEON's: the feature is CODEGEN-gated,
// not uniform-gated. Look = None must emit shader text with NOT ONE grade
// token — that byte-identity is the perf doctrine's "prove it's free" standard
// (the #125 lesson: a never-executing uniform branch still cost Mandelbulb
// +31%). The march shaders are untouched by construction (the grade lives in
// the POST pass / the GL fragment tail); these tests fence:
//   - the WGSL post pass: default build carries no grade token and is
//     byte-identical (sha256) to an explicit { grade: false } build,
//   - every GLSL emitter (flat / hybrid / scene) across the ENTIRE preset
//     catalog: { grade: false } is sha256-identical to the default build,
//   - the WGSL march shader never sees the grade at all,
//   - grade math parity between the two GPU tiers (the neon parity method),
//   - the shared derivation (frameparams) and sanitize (the hostile wire).
//
// Run: node --test core/gradegate.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import {
  buildWGSL,
  buildPostWGSL,
  GRADE_WORD,
  GLOBALS_WORDS,
  GLOBALS_WORDS_ALLOC,
} from "./shader.js";
import { buildFragGL, buildSceneFragGL } from "./shader_gl.js";
import { deriveFrameParams } from "./frameparams.js";
import { sanitizeColoring } from "./sanitize.js";
import { PRESETS } from "./oplist.js";
import { hybridSlots } from "./hybridmodel.js";

const sha = (s) => createHash("sha256").update(s).digest("hex");
const GRADE_TOKENS_WGSL = ["CINE GRADE", "gA = G.w["];
const GRADE_TOKENS_GL = ["cineGrade", "uGradeA"];
const A = [{ key: "boxFold", values: [1] }];
const B = [{ key: "scale", values: [2] }];
const SCENE = [{ shapeId: 1, objType: "sphere", combine: "union" }];

// ── The gate: off emits nothing, on emits everything ─────────────────────────
test("WGSL post: default build carries no grade token and byte-matches { grade: false }", () => {
  const base = buildPostWGSL();
  for (const tok of GRADE_TOKENS_WGSL)
    assert.ok(!base.includes(tok), `default post WGSL leaked ${tok}`);
  assert.equal(sha(base), sha(buildPostWGSL({ grade: false })));
  // PG stays at the frozen base size — the graded variant alone grows it.
  assert.ok(base.includes(`array<vec4f, ${GLOBALS_WORDS}>`));
});

test("WGSL post: the graded build carries the full feature", () => {
  const g = buildPostWGSL({ grade: true });
  for (const tok of GRADE_TOKENS_WGSL)
    assert.ok(g.includes(tok), `graded post WGSL missing ${tok}`);
  // Reads exactly the four tail rows, through a PG sized to reach them.
  for (let i = 0; i < 4; i++)
    assert.ok(g.includes(`G.w[${GRADE_WORD + i}]`), `missing row ${i}`);
  assert.ok(g.includes(`array<vec4f, ${GRADE_WORD + 4}>`));
  // The vignette must use the ABSOLUTE coordinates (gpos/gres — the tiled-
  // export §2.2.1(a) contract), never pos.xy directly.
  assert.ok(g.includes("distance(gpos / gres, vec2f(0.5))"));
});

test("WGSL march: no build variant ever sees the grade (post-pass-only feature)", () => {
  for (const opts of [
    {},
    { coloring: true },
    { envx: true },
    { neon: true },
    { capture: true },
  ]) {
    const src = buildWGSL(opts);
    for (const tok of [...GRADE_TOKENS_WGSL, "uGrade", "gradeA"])
      assert.ok(!src.includes(tok), `march WGSL (${JSON.stringify(opts)}) leaked ${tok}`);
  }
});

test("globals layout: grade rows append at 56 (53..55 reserved for aurora)", () => {
  assert.equal(GRADE_WORD, 56);
  // The ALLOC ceiling moved past GRADE_WORD+4 when the CLIP rows appended at
  // 60..61 (core/clipplane.test.mjs pins the full tail); the grade's own rows
  // are frozen where they were.
  assert.ok(GLOBALS_WORDS_ALLOC >= GRADE_WORD + 4);
});

// ── GLSL: byte-identity across the ENTIRE preset catalog ─────────────────────
// The aurora/neon method, extended to a sha256 sweep: every preset's emitted
// fragment (flat, hybrid, scene — whichever the preset is) must hash
// identically with the grade off vs. absent. One counterexample = a leak.
test("GLSL: grade-off is sha256-identical to default for every preset", () => {
  let flat = 0,
    hyb = 0,
    scene = 0;
  for (const p of PRESETS) {
    if (p.objects?.length) {
      assert.equal(
        sha(buildSceneFragGL(p.objects)),
        sha(buildSceneFragGL(p.objects, { grade: false })),
        `scene preset ${p.name} leaked grade bytes`,
      );
      scene++;
    } else if (p.hybrid) {
      const slots = hybridSlots(p).slots;
      const extras = slots.slice(1).map((s) => ({ ops: s.ops }));
      assert.equal(
        sha(buildFragGL(slots[0].ops, extras)),
        sha(buildFragGL(slots[0].ops, extras, undefined, { grade: false })),
        `hybrid preset ${p.name} leaked grade bytes`,
      );
      hyb++;
    } else {
      assert.equal(
        sha(buildFragGL(p.ops)),
        sha(buildFragGL(p.ops, undefined, undefined, { grade: false })),
        `flat preset ${p.name} leaked grade bytes`,
      );
      flat++;
    }
  }
  assert.equal(flat + hyb + scene, PRESETS.length);
  // No grade token anywhere in a default build either (spot-check one of each).
  for (const src of [buildFragGL(A), buildFragGL(A, [{ ops: B }]), buildSceneFragGL(SCENE)])
    for (const tok of GRADE_TOKENS_GL)
      assert.ok(!src.includes(tok), `default GLSL leaked ${tok}`);
});

test("GLSL: graded builds carry the full feature (flat/hybrid/scene)", () => {
  for (const src of [
    buildFragGL(A, undefined, undefined, { grade: true }),
    buildFragGL(A, [{ ops: B }], undefined, { grade: true }),
    buildFragGL(A, undefined, undefined, { grade: true, envx: true, neon: true }),
    buildSceneFragGL(SCENE, { grade: true }),
  ]) {
    for (const tok of GRADE_TOKENS_GL)
      assert.ok(src.includes(tok), `graded GLSL missing ${tok}`);
    // Applied at BOTH encode points: the miss/sky return and the hit tail.
    assert.ok(src.includes("l2s(cineGrade(tone3(skyOut * exp2(uExposure))))"));
    assert.ok(src.includes("l2s(cineGrade(tone3(max(col, vec3(0.0)) * exp2(uExposure))))"));
  }
});

// ── Cross-tier parity — the same grade math in both GPU tiers ────────────────
test("grade math is pinned identically in WGSL and GLSL", () => {
  const wgsl = buildPostWGSL({ grade: true });
  const glsl = buildFragGL(A, undefined, undefined, { grade: true });
  // The load-bearing lines, modulo dialect (vec3f/vec3, let/float).
  for (const [w, g] of [
    ["dot(g, vec3f(0.2126, 0.7152, 0.0722))", "dot(g, vec3(0.2126, 0.7152, 0.0722))"],
    ["(1.0 - smoothstep(0.0, 0.45, gLum))", "(1.0 - smoothstep(0.0, 0.45, lum))"],
    ["smoothstep(0.25, 0.65, gLum)", "smoothstep(0.25, 0.65, lum)"],
    ["g * gTone * 2.0", "g * tone * 2.0"],
    ["mix(gB.rgb, gC.rgb, pow(gLum, 0.6))", "mix(uGradeB.rgb, uGradeC.rgb, pow(lum, 0.6))"],
    ["smoothstep(0.30, 1.05, gvd)", "smoothstep(0.30, 1.05, vd)"],
  ]) {
    assert.ok(wgsl.includes(w), `WGSL missing ${w}`);
    assert.ok(glsl.includes(g), `GLSL missing ${g}`);
  }
  // Both grade post-tonemap: WGSL after the tone3 gate, GLSL wrapping tone3.
  assert.ok(wgsl.indexOf("c = tone3(c);") < wgsl.indexOf("CINE GRADE"));
});

// ── Shared derivation ────────────────────────────────────────────────────────
test("deriveFrameParams: no grade derives the latch off with neutral words", () => {
  const d = deriveFrameParams({ light: {} });
  assert.equal(d.gradeOn, false);
  assert.deepEqual(d.gradeA, [0, 0, 1, 0]);
  assert.deepEqual(d.gradeB, [0.5, 0.5, 0.5, 0]);
  assert.deepEqual(d.gradeC, [0.5, 0.5, 0.5, 0]);
  assert.deepEqual(d.gradeD, [0, 0, 0, 0]);
});

test("deriveFrameParams: a grade block maps to the words, clamped", () => {
  const d = deriveFrameParams({
    light: {
      grade: {
        strength: 0.8,
        contrast: 9, // clamps to 1
        saturation: 3, // clamps to 2
        shadowDesat: -1, // clamps to 0
        shadowTint: [0.3, 0.5, 2], // channel clamps to 1
        splitAmt: 0.9,
        hiTint: [0.7, 0.5, 0.4],
        duoAmt: 0.5,
        vignette: 0.2,
      },
    },
  });
  assert.equal(d.gradeOn, true);
  assert.deepEqual(d.gradeA, [0.8, 1, 2, 0]);
  assert.deepEqual(d.gradeB, [0.3, 0.5, 1, 0.9]);
  assert.deepEqual(d.gradeC, [0.7, 0.5, 0.4, 0.5]);
  assert.deepEqual(d.gradeD, [0.2, 0, 0, 0]);
});

test("deriveFrameParams: strength 0 keeps the latch OFF (no graded pipeline)", () => {
  const d = deriveFrameParams({ light: { grade: { strength: 0, contrast: 1 } } });
  assert.equal(d.gradeOn, false);
});

// ── Sanitize (the hostile-wire fence) ────────────────────────────────────────
test("sanitize: look/strength clamp to their domains, absent stays absent", () => {
  const out = sanitizeColoring({ light: { gradeLook: 99, gradeStrength: 7 } });
  assert.equal(out.light.gradeLook, 15);
  assert.equal(out.light.gradeStrength, 1);
  const abs = sanitizeColoring({ light: { ambient: 0.2 } });
  assert.ok(!("gradeLook" in abs.light), "sanitize invented light.gradeLook");
  assert.ok(!("gradeStrength" in abs.light), "sanitize invented light.gradeStrength");
});

test("sanitize: the DERIVED grade block never survives an import", () => {
  const out = sanitizeColoring({
    light: { gradeLook: 2, grade: { strength: 1, contrast: 1 } },
  });
  assert.ok(!("grade" in out.light), "derived light.grade must be stripped");
});

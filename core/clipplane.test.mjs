// CLIP — clipping-plane cross-section + the MRI sweep (IDEAS catalog row +
// the 2026-08-21 wave's last idea).
//
// The load-bearing invariant, same as tiny planet's: the feature is
// CODEGEN-gated, not uniform-gated. The plane term lives INSIDE the march
// loop — exactly the #125 / d00d9a7 register-pressure surface — so clip off
// must emit shader text with NOT ONE clip token, byte for byte (sha256), on
// BOTH GPU tiers, across the feature matrix AND the whole preset catalog.
// The CPU tier is the one copy of the clipped-march algebra a node test can
// execute (WGSL/GLSL compile nowhere in CI), so the geometry — the CSG max,
// the clipped bisection, the cut-face classification — is proven there
// against a closed-form sphere.
//
// Allocation ledger (2026-08-26/27): F_CLIP = 65536, globals tail rows
// 60..61 (62 spare from clip's 60-62 claim; 63 reserved for the concurrent
// 360°/equirect arm — which landed on 131072 + camUp.w instead), share codec
// P3 bit 9. The JAGGED follow-on adds F_CLIPJAG = 262144 and ZERO new rows
// (amp/freq/invD ride clipS.yzw — lanes inside clip's row-61 claim); the
// bit-9 sub-block widened 2 → 3 bytes pre-release. Pinned below (§4, §11).
//
// Run: node --test core/clipplane.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";

import {
  buildWGSL,
  ENVX_WORD,
  EMAP_WORD,
  AUR_WORD,
  GRADE_WORD,
  CLIP_WORD,
  GLOBALS_WORDS,
  GLOBALS_WORDS_ALLOC,
} from "./shader.js";
import { buildFragGL, buildSceneFragGL } from "./shader_gl.js";
import { keyFor, wgslOf } from "./renderer.js";
import { frameFeaturesFor } from "./capturesettle.js";
import {
  deriveFrameParams,
  CLIP_OFFSET_LIMIT,
  CLIP_JAG_WORLD,
  CLIP_JAG_FREQ,
  CLIP_JAG_LIP,
} from "./frameparams.js";
import { sanitizeColoring } from "./sanitize.js";
import { measureWorldExtents } from "./preview.js";
import { exportStandaloneGLSL } from "./exportStandalone.js";
import { defaultColoring } from "./coloring.js";
import { shadeGrid, clipJagNoise } from "./cpu.js";
import { makeCamera } from "./camera.js";
import { PRESETS } from "./oplist.js";
import { hybridSlots } from "./hybridmodel.js";

const readSrc = async (f) =>
  (await import("node:fs/promises")).readFile(
    new URL(f, import.meta.url),
    "utf8",
  );
const sha = (s) => createHash("sha256").update(s).digest("hex");

// Every token below exists ONLY in the clip splices — one leaking into an
// off build is a byte-identity failure with a name attached.
const CLIP_TOKENS_WGSL = ["clipU", "clipS", "cutFace", "clipPl", "clipN"];
const CLIP_TOKENS_GL = ["uClip", "uClipS", "cutFace", "clipPl"];

const A = [{ key: "boxFold", values: [1] }];
const B = [{ key: "scale", values: [2] }];
const SCENE = [{ shapeId: 1, objType: "sphere", combine: "union" }];

// ── 1. the gate ─────────────────────────────────────────────────────────────

// The SAME matrix the envx/neon/planet gates use, with the features that
// landed since appended — every combination must be clip-free when the flag
// is off.
const WGSL_MATRIX = [
  ["default", {}],
  [
    "minimal",
    {
      numericDE: false,
      leaves: false,
      coloring: false,
      scene: false,
      hybrid: false,
      morph: false,
    },
  ],
  ["capture", { capture: true }],
  ["df64", { df64: true }],
  ["perturb", { perturb: true }],
  ["envx", { envx: true }],
  ["neon", { neon: true }],
  ["sreflect", { sreflect: true }],
  ["envMap", { envMap: true }],
  ["surfTex", { surfTex: true }],
  ["aurora", { aurora: true }],
  ["thinFilm", { thinFilm: true }],
  ["planet", { planet: true }],
  ["scene", { scene: true, leaves: true }],
  ["hybrid", { hybrid: true }],
  ["morph", { morph: true }],
  ["ops:[]", { ops: [] }],
  ["ops:[1,2,3]", { ops: [1, 2, 3] }],
];

test("WGSL: clip off emits NOT ONE clip token, across the feature matrix", () => {
  for (const [name, opts] of WGSL_MATRIX) {
    const off = buildWGSL(opts);
    for (const tok of CLIP_TOKENS_WGSL)
      assert.ok(!off.includes(tok), `${name}: clip-off text leaked ${tok}`);
  }
});

test("WGSL: clip:false is byte-identical to omitting the flag", () => {
  for (const [name, opts] of WGSL_MATRIX) {
    assert.equal(
      sha(buildWGSL({ ...opts, clip: false })),
      sha(buildWGSL(opts)),
      `${name}: an explicit clip:false must not perturb one byte`,
    );
  }
});

test("WGSL: clip on actually clips the march (not a no-op flag)", () => {
  const off = buildWGSL();
  const on = buildWGSL({ clip: true });
  assert.notEqual(sha(off), sha(on), "clip:true emitted identical text");
  // The plane term reaches the march LOOP and the BISECTION refinement — the
  // refinement against an unclipped mapDE would walk the bracket back through
  // the cut face, so both call sites must carry the max.
  assert.ok(
    on.includes(
      "max(mapDE(p) * G.prm.z, dot(G.clipU.xyz, p + G.offset.xyz) - G.clipU.w)",
    ),
    "march loop lost the clipped distance",
  );
  assert.ok(
    on.includes(
      "max(mapDE(ro + rd * mid) * G.prm.z, dot(G.clipU.xyz, ro + rd * mid + G.offset.xyz) - G.clipU.w)",
    ),
    "bisection refinement lost the clipped distance",
  );
  // Cut-face shading: classification + the flat MRI shade + the AO/shadow
  // cut guards (both march the UNCLIPPED mapDE from inside the body and
  // would black the face).
  assert.ok(on.includes("let cutFace = hit && clipPl >= mapDE(p) * G.prm.z;"));
  assert.ok(on.includes("if (G.mat.w > 0.0 && !cutFace)"), "AO cut guard");
  assert.ok(on.includes("if (G.mat.z > 0.5 && !cutFace)"), "shadow cut guard");
  assert.ok(on.includes("if (cutFace) {"), "cut shade missing");
  // …and the off arm keeps its own text (the plain expressions).
  assert.ok(off.includes("let d = mapDE(p) * G.prm.z;"));
  assert.ok(off.includes("let nrm = calcNormal(p, t);"));
});

test("GLSL: clip off emits NOT ONE clip token (flat, hybrid and scene)", () => {
  const builds = [
    ["flat", buildFragGL(A)],
    ["flat-explicit", buildFragGL(A, undefined, undefined, { clip: false })],
    ["hybrid", buildFragGL(A, [{ ops: B }])],
    ["scene", buildSceneFragGL(SCENE)],
    ["scene-explicit", buildSceneFragGL(SCENE, { clip: false })],
  ];
  for (const [name, src] of builds)
    for (const tok of CLIP_TOKENS_GL)
      assert.ok(!src.includes(tok), `${name}: clip-off GLSL leaked ${tok}`);
});

test("GLSL: the clip variant declares uClip, the off variant never does", () => {
  for (const src of [
    buildFragGL(A, undefined, undefined, { clip: true }),
    buildFragGL(A, [{ ops: B }], undefined, { clip: true }),
    buildSceneFragGL(SCENE, { clip: true }),
  ]) {
    assert.ok(src.includes("uniform vec4 uClip;"), "no uClip declaration");
    assert.ok(src.includes("uniform float uClipS;"), "no uClipS declaration");
    assert.ok(
      src.includes(
        "max(mapDE(ro + rd * t) * uDeScale, dot(uClip.xyz, ro + rd * t + uOffset) - uClip.w)",
      ),
      "GL march loop lost the clipped distance",
    );
    assert.ok(
      src.includes(
        "max(mapDE(ro + rd * mid) * uDeScale, dot(uClip.xyz, ro + rd * mid + uOffset) - uClip.w)",
      ),
      "GL bisection lost the clipped distance",
    );
    assert.ok(src.includes("bool cutFace = hit && clipPl >= mapDE(p) * uDeScale;")); // prettier-ignore
  }
});

// ── 2. byte-identity across the ENTIRE preset catalog (both tiers) ──────────
// The gradegate method: every preset's emitted shader, clip:false vs default,
// sha256-identical — with the flat/hybrid/scene tally proving no preset fell
// out of the sweep.

test("GLSL: clip-off is sha256-identical to default for every preset", () => {
  let flat = 0,
    hyb = 0,
    scene = 0;
  for (const p of PRESETS) {
    if (p.objects?.length) {
      assert.equal(
        sha(buildSceneFragGL(p.objects)),
        sha(buildSceneFragGL(p.objects, { clip: false })),
        `scene preset ${p.name} leaked clip bytes`,
      );
      scene++;
    } else if (p.hybrid) {
      const slots = hybridSlots(p).slots;
      const extras = slots.slice(1).map((s) => ({ ops: s.ops }));
      assert.equal(
        sha(buildFragGL(slots[0].ops, extras)),
        sha(buildFragGL(slots[0].ops, extras, undefined, { clip: false })),
        `hybrid preset ${p.name} leaked clip bytes`,
      );
      hyb++;
    } else {
      assert.equal(
        sha(buildFragGL(p.ops)),
        sha(buildFragGL(p.ops, undefined, undefined, { clip: false })),
        `flat preset ${p.name} leaked clip bytes`,
      );
      flat++;
    }
  }
  assert.equal(flat + hyb + scene, PRESETS.length);
});

test("WGSL: clip-off is sha256-identical to default for every preset's variant", () => {
  let n = 0;
  for (const p of PRESETS) {
    // The exact descriptor the renderer would latch for this preset (mode 1
    // exercises the coloring lever too), through the same wgslOf mapping.
    const ff = frameFeaturesFor(p, { mode: 1 });
    const o = wgslOf(ff);
    assert.equal(
      sha(buildWGSL({ ...o, clip: false })),
      sha(buildWGSL(o)),
      `preset ${p.name}: clip:false perturbed the WGSL variant`,
    );
    n++;
  }
  assert.equal(n, PRESETS.length);
});

// ── 3. the two tiers can't drift — shared math tokens ───────────────────────
// The aurora AURORA_MAGIC method: the cut-shade curve and the plane compare
// must be pinned identically in both emitters.
test("the cut-face math is pinned identically in WGSL and GLSL", async () => {
  const wgsl = buildWGSL({ clip: true });
  const gl = buildFragGL(A, undefined, undefined, { clip: true });
  for (const term of ["(0.42 + 0.58 * max(dot(nrm, lightDir), 0.0))"]) {
    assert.ok(wgsl.includes(term), `WGSL lost the cut shade term ${term}`);
    assert.ok(gl.includes(term), `GLSL lost the cut shade term ${term}`);
  }
  // Both classifications compare the plane distance against the SCALED DE.
  assert.ok(wgsl.includes("clipPl >= mapDE(p) * G.prm.z"));
  assert.ok(gl.includes("clipPl >= mapDE(p) * uDeScale"));
  // The CPU mirror carries the same max — pinned as source text, since node
  // executes it below but a token drift would still un-mirror the tiers.
  // (The three sites share ONE clipPlaneAt helper since the jagged cut; the
  // pin follows the helper's flat expression + its use in the march and the
  // bisection.)
  const cpu = await readSrc("./cpu.js");
  assert.ok(cpu.includes("clip.nx * x + clip.ny * y + clip.nz * z - clip.w")); // prettier-ignore
  assert.ok(cpu.includes("d = Math.max(d, clipPlaneAt(sx, sy, sz))")); // prettier-ignore
  assert.ok(cpu.includes("dm = Math.max(dm, clipPlaneAt(mx, my, mz))")); // prettier-ignore
});

// ── 4. variant keying — F_CLIP = 65536 (the allocation ledger) ──────────────

test("renderer: clip is its own variant bit and reaches buildWGSL", () => {
  const base = { ops: null, leaves: null };
  assert.notEqual(
    keyFor({ ...base, clip: true }),
    keyFor({ ...base, clip: false }),
    "clip must not share a variant with the plain march",
  );
  // It must not collide with any other single feature bit either. (`leaves`
  // is excluded — it is an id array, not a boolean bit.)
  const bits = [
    "numericDE",
    "coloring",
    "scene",
    "hybrid",
    "morph",
    "df64",
    "perturb",
    "envx",
    "envMap",
    "surfTex",
    "sreflect",
    "neon",
    "aurora",
    "thinFilm",
    "planet",
  ];
  const clipKey = keyFor({ ...base, clip: true });
  for (const b of bits)
    assert.notEqual(clipKey, keyFor({ ...base, [b]: true }), `collides: ${b}`);
  assert.equal(wgslOf({ ...base, clip: true }).clip, true);
  assert.equal(wgslOf(base).clip, false);
  const bitOf = (b) => Number(keyFor({ ...base, [b]: true }).split(":")[0]);
  const clipBit = Number(clipKey.split(":")[0]);
  // The ledger's pre-assignment, verified not assumed: 65536, above the sum
  // of every existing bit (so it is a genuinely new power of two)…
  assert.equal(clipBit, 65536, "the ledger says F_CLIP = 65536");
  const others = bits.reduce((n, b) => n + bitOf(b), 0);
  assert.ok(
    clipBit > others,
    `F_CLIP (${clipBit}) must sit above every other bit (sum ${others})`,
  );
  // …and independent of each (key = the SUM of two bits, never an alias).
  for (const b of bits)
    assert.equal(
      Number(keyFor({ ...base, clip: true, [b]: true }).split(":")[0]),
      clipBit + bitOf(b),
      `clip + ${b} must be the SUM of two bits`,
    );
  // The OFF key string is literally the pre-clip key string — the cheap
  // byte-identity analogue at the keying layer (marchvariant's format pin).
  assert.match(keyFor({ ...base, ops: null }), /^0:\*:-$/);
});

// ── 5. globals layout — the append-only tail contract ───────────────────────

test("globals layout: CLIP rows append at 60 (the ledger's 60-62 claim)", () => {
  assert.equal(GLOBALS_WORDS, 48); // base struct + post PG size — frozen
  assert.equal(ENVX_WORD, 48); // untouched
  assert.equal(EMAP_WORD, 51); // untouched
  assert.equal(AUR_WORD, 53); // untouched
  assert.equal(GRADE_WORD, 56); // untouched
  assert.equal(CLIP_WORD, 60); // clipU=60, clipS=61
  // Ceiling: 62 (clip's spare row 62 is NOT allocated; row 63 is reserved
  // for the 360°/equirect arm and must not be claimed here).
  assert.equal(GLOBALS_WORDS_ALLOC, 62);
});

test("clip variants carry EVERY earlier tail row as dormant padding", () => {
  // clipU sits at fixed word 60 against the ONE buffer layout, so a
  // clip-only variant must declare starsU..gradeD above it — including the
  // grade rows no march shader reads.
  const src = buildWGSL({ clip: true });
  for (const row of [
    "starsU",
    "bandU",
    "zenU",
    "emapU",
    "triU",
    "aurU",
    "aurA",
    "aurB",
    "gradeA",
    "gradeB",
    "gradeC",
    "gradeD",
    "clipU",
    "clipS",
  ])
    assert.ok(src.includes(row), `clip struct missing padding row ${row}`);
  // …and clip does NOT drag the aurora/envx CODE in, only the rows.
  assert.ok(!src.includes("auroraSky"), "clip variant emitted the aurora sky");
  assert.ok(!src.includes("starField"), "clip variant emitted the star field");
});

// ── 6. the shared derivation (frameparams) ──────────────────────────────────

test("frameparams: clipOn is the latch; axis+flip derive the plane", () => {
  const d0 = deriveFrameParams({});
  assert.equal(d0.clip, false);
  assert.deepEqual(d0.clipN, [1, 0, 0]);
  assert.equal(d0.clipW, 0);
  assert.equal(d0.clipShade, 1);

  const on = deriveFrameParams({ light: { clipOn: true } });
  assert.equal(on.clip, true);

  // The latch is the EXPLICIT boolean — a truthy non-boolean does not latch
  // (sanitize/decode normalize to real booleans before this runs).
  assert.equal(deriveFrameParams({ light: { clipOn: 1 } }).clip, false);

  // Axis 2, flipped, offset 0.7: normal −Z, constant −0.7 — the plane HOLDS
  // its position at z = 0.7 and only the kept half swaps.
  const f = deriveFrameParams({
    light: { clipOn: true, clipAxis: 2, clipFlip: true, clipOffset: 0.7 },
  });
  assert.deepEqual(f.clipN, [0, 0, -1]);
  assert.ok(Math.abs(f.clipW - -0.7) < 1e-12);

  // Hostile numbers: axis rounds and clamps, offset clamps to the wire
  // fence (±CLIP_OFFSET_LIMIT — wider than the World slider's −2..2 because
  // the MRI sweep drives measured world-sized offsets; Mandelbox needs ±7).
  const h = deriveFrameParams({
    light: { clipOn: true, clipAxis: 9.4, clipOffset: 1e9 },
  });
  assert.deepEqual(h.clipN, [0, 0, 1]);
  assert.equal(h.clipW, CLIP_OFFSET_LIMIT);
  // …and a sweep-sized offset passes through UNCLAMPED (the defect the wide
  // fence fixes: the old ±2 clamp parked the plane inside big formulas).
  assert.equal(
    deriveFrameParams({ light: { clipOn: true, clipOffset: 7.2 } }).clipW,
    7.2,
  );
  assert.deepEqual(
    deriveFrameParams({ light: { clipOn: true, clipAxis: -3 } }).clipN,
    [1, 0, 0],
  );
});

test("capturesettle: frameFeaturesFor mirrors the latch (no scene guard)", () => {
  const flat = { ops: [{ key: "scale", values: [2] }], iters: 4 };
  const scene = { objects: SCENE, ops: [], iters: 1 };
  assert.equal(frameFeaturesFor(flat, { mode: 0 }).clip, false);
  assert.equal(
    frameFeaturesFor(flat, { mode: 0, light: { clipOn: true } }).clip,
    true,
  );
  // March geometry — scenes carry it too (the planet/aurora contract).
  assert.equal(
    frameFeaturesFor(scene, { mode: 0, light: { clipOn: true } }).clip,
    true,
  );
  assert.equal(frameFeaturesFor(scene, { mode: 0 }).clip, false);
});

// ── 7. sanitize — the hostile wire ──────────────────────────────────────────

test("sanitize: clip fields clamp to the wire domain and stay shape-preserving", () => {
  const c = sanitizeColoring({
    mode: 1,
    light: { clipOn: 2, clipFlip: "yes", clipAxis: 250, clipOffset: -1e9 },
  });
  assert.equal(c.light.clipOn, true); // boolean-coerced
  assert.equal(c.light.clipFlip, true);
  assert.equal(c.light.clipAxis, 2); // clamped to 0..2
  assert.equal(c.light.clipOffset, -CLIP_OFFSET_LIMIT); // clamped to the wire fence
  // A measured sweep-sized offset survives the wire (frameparams' fence and
  // sanitize's must agree — one CLIP_OFFSET_LIMIT source).
  assert.equal(
    sanitizeColoring({ mode: 1, light: { clipOffset: -6.9 } }).light.clipOffset,
    -6.9,
  );
  // Absent fields stay absent (the shape-preserving contract — re-sharing a
  // pre-clip look must not materialize clip fields onto it).
  const abs = sanitizeColoring({ mode: 1, light: { ambient: 0.2 } });
  assert.ok(!("clipOn" in abs.light), "sanitize invented light.clipOn");
  assert.ok(!("clipAxis" in abs.light), "sanitize invented light.clipAxis");
  assert.ok(!("clipOffset" in abs.light), "sanitize invented light.clipOffset");
});

// ── 8. the CPU tier — the one executable copy of the algebra ────────────────
// One sphere of radius R at the origin: every clipped-march property has a
// closed form. The camera basis is read off makeCamera, so the test aims the
// plane relative to the ACTUAL eye rather than assuming a yaw convention.
const R = 0.5;
const D = 4;
const SPHERE = {
  name: "clip-probe-sphere",
  objects: [
    {
      shapeId: 2,
      shapeParams: [R, 0, 0, 0],
      combine: "union",
      ops: [],
      iters: 1,
    },
  ],
  ops: [],
  iters: 1,
  camera: { yawDeg: 0, pitchDeg: 0, dist: D, fovDeg: 42 },
};
const GRID = {
  cols: 81,
  rows: 81,
  aspect: 1,
  ss: 1,
  edges: false,
  structure: false,
  dither: false,
  coloring: { mode: 1, autoLevels: false },
};
const eyeOf = () => makeCamera(SPHERE.camera).basis().eye;
// shadeGrid returns chars/rgb; the geometric assertions need the raw grid —
// drive traceGrid through shadeGrid's opts.clip escape hatch and read the
// coverage off the chars (space = miss), the tinyplanet method.
const gridOf = (clip) => {
  const { cols, rows, chars } = shadeGrid(SPHERE, { ...GRID, clip });
  let covered = 0;
  for (const ch of chars) if (ch !== " ") covered++;
  return { cols, rows, chars, covered };
};

test("cpu: clip null (and the absent field) render byte-identically", () => {
  const base = shadeGrid(SPHERE, GRID).rgb;
  assert.deepEqual(
    shadeGrid(SPHERE, { ...GRID, clip: null }).rgb,
    base,
    "clip: null must not change one CPU pixel",
  );
});

test("cpu: a plane past the sphere hides nothing; through the centre, half", () => {
  const eye = eyeOf();
  // Slice along the axis most PERPENDICULAR to the view ray, so the cut is
  // seen edge-on and coverage genuinely halves.
  const a = [0, 1, 2].reduce((m, i) => (Math.abs(eye[i]) < Math.abs(eye[m]) ? i : m), 0); // prettier-ignore
  const n = [0, 0, 0];
  n[a] = 1;
  const clipAt = (w) => ({ nx: n[0], ny: n[1], nz: n[2], w });
  const base = gridOf(null).covered;
  const clear = gridOf(clipAt(1.5)).covered; // keep n·p <= 1.5 ⊇ the whole sphere
  const half = gridOf(clipAt(0)).covered; // through the centre
  assert.equal(clear, base, "a non-binding plane changed the silhouette");
  assert.ok(
    half > 0.3 * base && half < 0.7 * base,
    `centre plane should halve coverage (${half}/${base})`,
  );
  // Flip keeps the other half; together they tile the disc (edge cells on
  // the cut line may land in both — allow slack).
  const flipped = gridOf({ nx: -n[0], ny: -n[1], nz: -n[2], w: 0 }).covered;
  assert.ok(
    half + flipped > 0.9 * base && half + flipped < 1.15 * base,
    `the two halves should tile the sphere (${half}+${flipped} vs ${base})`,
  );
});

test("cpu: viewed from the clipped side, the visible face IS the cut", () => {
  const eye = eyeOf();
  // Slice along the axis most ALIGNED with the eye, keeping the far half —
  // the camera sits in the removed half-space and looks straight at the flat
  // cross-section (the MRI view).
  const a = [0, 1, 2].reduce((m, i) => (Math.abs(eye[i]) > Math.abs(eye[m]) ? i : m), 0); // prettier-ignore
  const s = Math.sign(eye[a]) || 1;
  const n = [0, 0, 0];
  n[a] = s; // keep n·p <= 0 — the half AWAY from the camera
  const { cols, rows } = GRID;
  const g = shadeGrid(SPHERE, {
    ...GRID,
    clip: { nx: n[0], ny: n[1], nz: n[2], w: 0 },
    scratch: {},
  });
  let covered = 0;
  for (const ch of g.chars) if (ch !== " ") covered++;
  assert.ok(covered > 0, "the clipped sphere vanished");
  // The raw grid: re-trace via the same opts to read the cut plane. (shadeGrid
  // has no grid return; assert on the CENTRE cell's character instead — the
  // centre of the cut disc must be a lit, non-space glyph, and the lit
  // intensity across the middle row must be FLAT: the plane normal shades
  // every cut cell identically, which is the tier's lightbox.)
  const mid = (rows - 1) / 2;
  const row = g.chars.slice(mid * cols, (mid + 1) * cols);
  const lit = row.filter((ch) => ch !== " ");
  assert.ok(lit.length > 5, "no cut face across the centre row");
  // Flat shade ⇒ one glyph dominates the row (the disc interior is a single
  // intensity; only the rim cells may differ).
  const counts = {};
  for (const ch of lit) counts[ch] = (counts[ch] || 0) + 1;
  const domFrac = Math.max(...Object.values(counts)) / lit.length;
  assert.ok(
    domFrac > 0.7,
    `cut face should shade flat (dominant glyph ${(domFrac * 100) | 0}%)`,
  );
});

// ── 9. standalone export + capture never carry the cut ──────────────────────

test("standalone export: a bundle is always UNCLIPPED", () => {
  const g = exportStandaloneGLSL(
    { name: "x", ops: [{ key: "scale", values: [2] }], iters: 6 },
    {
      light: {
        ...defaultColoring().light,
        clipOn: true,
        clipAxis: 1,
        clipOffset: 0.4,
      },
    },
  );
  for (const tok of CLIP_TOKENS_GL)
    assert.ok(!g.includes(tok), `standalone bundle leaked ${tok}`);
});

test("renderer.js source: the capture pipeline forces clip off", async () => {
  const src = await readSrc("./renderer.js");
  assert.match(
    src,
    /clip: false,\s*clipJag: false,\s*planet: false,\s*equirect: false,\s*capture: true/,
    "splat capture marches its own rays — it must not fork on clip (nor its jagged sub-variant)",
  );
});

// ── 10. world extents (the MRI sweep's amplitude measurement) ───────────────
//
// preview.js measureWorldExtents — the seam that replaced the sweep's fixed
// 1.3 amplitude (the "doesn't go through the whole fractal" defect: formula
// world size varies from Menger's ±1 to Mandelbox's ±7, and Whorl Citadel's
// porous spiral defeats even a radial ray probe). Closed-form DEs prove the
// measurement here; the live-browser probe proves the traversal.

test("measureWorldExtents: closed-form sphere — per-axis extents, off-centre", () => {
  // Sphere radius 3 centred at (1, 0, 0): extents [4, 3, 3].
  const de = (x, y, z) => Math.hypot(x - 1, y, z) - 3;
  const m = measureWorldExtents(de);
  assert.ok(m, "sphere measured null");
  assert.ok(Math.abs(m.ext[0] - 4) < 0.5, `x extent ${m.ext[0]} ≠ ~4`);
  assert.ok(Math.abs(m.ext[1] - 3) < 0.5, `y extent ${m.ext[1]} ≠ ~3`);
  assert.ok(Math.abs(m.ext[2] - 3) < 0.5, `z extent ${m.ext[2]} ≠ ~3`);
  // The safe direction: never UNDER the true extent (a plane at ±ext must be
  // fully clear of the body — the whole point of the measurement).
  assert.ok(
    m.ext[0] >= 4 - 1e-9 && m.ext[1] >= 3 - 1e-9 && m.ext[2] >= 3 - 1e-9,
  );
  assert.ok(m.r >= Math.hypot(m.ext[0], m.ext[1], m.ext[2]) - 1e-9);
});

test("measureWorldExtents: anisotropic box — each axis measured separately", () => {
  // Axis-aligned box half-extents (0.6, 2, 5) — a slab: the sweep amplitude
  // along X must NOT inherit Z's 5 (dead travel was the original 1.3-vs-2.0
  // probe complaint, restated per-axis).
  const de = (x, y, z) => {
    const qx = Math.abs(x) - 0.6;
    const qy = Math.abs(y) - 2;
    const qz = Math.abs(z) - 5;
    const ox = Math.max(qx, 0);
    const oy = Math.max(qy, 0);
    const oz = Math.max(qz, 0);
    return Math.hypot(ox, oy, oz) + Math.min(Math.max(qx, qy, qz), 0);
  };
  const m = measureWorldExtents(de);
  assert.ok(m, "box measured null");
  assert.ok(m.ext[0] >= 0.6 && m.ext[0] < 1.4, `x ${m.ext[0]} not ~0.6..1.4`);
  assert.ok(m.ext[1] >= 2 && m.ext[1] < 2.8, `y ${m.ext[1]} not ~2..2.8`);
  assert.ok(m.ext[2] >= 5 && m.ext[2] < 5.8, `z ${m.ext[2]} not ~5..5.8`);
});

test("measureWorldExtents: no body / no DE / throwing DE → null (fallback)", () => {
  assert.equal(
    measureWorldExtents((x, y, z) => Math.hypot(x, y, z) + 100),
    null,
  );
  assert.equal(measureWorldExtents(null), null);
  assert.equal(
    measureWorldExtents(() => {
      throw new Error("mid-scan");
    }),
    null,
  );
});

test("measureWorldExtents: acceptance presets measure world-sized (the defect numbers)", async () => {
  // The CPU DE of the real catalog entries — the same evaluator the preview
  // seam hands over. Menger ~±1 (unit scale), Mandelbox ~±6-7 (the old ±2
  // offset clamp couldn't even reach its midriff), Whorl Citadel ~±3-4 (the
  // owner's GIF: the fixed 1.3 started the sweep already 13% deep).
  const { makeDE } = await import("./cpu.js");
  const ext0 = (name) => {
    const p = PRESETS.find((x) => x.name === name);
    assert.ok(p, `${name} missing from PRESETS`);
    const m = measureWorldExtents(makeDE(p));
    assert.ok(m, `${name} measured null`);
    return m.ext[0];
  };
  const menger = ext0("Menger");
  assert.ok(menger > 0.95 && menger < 1.5, `Menger x ${menger}`);
  const mbox = ext0("Mandelbox");
  assert.ok(mbox > 5.5 && mbox < 8, `Mandelbox x ${mbox}`);
  const whorl = ext0("Whorl Citadel");
  assert.ok(whorl > 2.6 && whorl < 4.5, `Whorl Citadel x ${whorl}`);
  // …and every one exceeds what the old fixed 1.3 could clear except the
  // unit-scale Menger — the per-formula measurement is not optional.
  assert.ok(mbox > 1.3 && whorl > 1.3);
});

// ── 11. CLIP JAGGED — the noised/eroded cut ─────────────────────────────────
//
// The owner's "noisy/jagged plane, so it kind of consumes the shape", and per
// the follow-up clarification a property of the STATIC cross-section itself:
// the World pane's Jagged slider erodes a still cut, and the sweep merely
// animates the plane offset through the same static world-space field. Its
// own codegen latch (F_CLIPJAG) on the #125 rationale — the value noise runs
// once per march step, so a FLAT cut must not carry one token of it — and its
// own DE-safety argument: the noised plane term is divided by
// 1 + CLIP_JAG_LIP·amp·freq, the worst-case Lipschitz bound of the noised
// field, restoring the march's lower-bound property. amp/freq/invD are
// runtime words in clipS.yzw — lanes INSIDE clip's existing row-61
// allocation, so the jag claims one feature bit (262144) and zero new rows.

const JAG_TOKENS = ["clipJagNoise", "clipJagHash", "clipJagVal"];
const JAG_TERM_WGSL = (pt) =>
  `(dot(G.clipU.xyz, ${pt} + G.offset.xyz) - G.clipU.w - G.clipS.y * clipJagNoise((${pt} + G.offset.xyz) * G.clipS.z)) * G.clipS.w`;
const JAG_TERM_GL = (pt) =>
  `(dot(uClip.xyz, ${pt} + uOffset) - uClip.w - uClipJ.x * clipJagNoise((${pt} + uOffset) * uClipJ.y)) * uClipJ.z`;

test("WGSL: the FLAT clip carries not one jag token (its own gate)", () => {
  // The jag noise must never ride a flat cut — the numeric-DE lesson (#125):
  // per-step work behind a "never true" condition is not free.
  for (const opts of [{}, { clip: true }, { clip: true, clipJag: false }]) {
    const src = buildWGSL(opts);
    for (const tok of JAG_TOKENS)
      assert.ok(
        !src.includes(tok),
        `${JSON.stringify(opts)} leaked jag token ${tok}`,
      );
  }
  // clipJag:false is byte-identical to omitting the flag, on the flat-clip
  // variant AND across the whole off matrix.
  assert.equal(
    sha(buildWGSL({ clip: true, clipJag: false })),
    sha(buildWGSL({ clip: true })),
  );
  for (const [name, opts] of WGSL_MATRIX)
    assert.equal(
      sha(buildWGSL({ ...opts, clipJag: false })),
      sha(buildWGSL(opts)),
      `${name}: clipJag:false perturbed the off text`,
    );
});

test("WGSL: the jag variant erodes march, bisection AND classification with ONE expression", () => {
  const src = buildWGSL({ clip: true, clipJag: true });
  // The identical scaled term at all three call sites — classification must
  // compare the operand that actually bound the max, or cut pixels
  // misclassify along the jagged boundary.
  assert.ok(
    src.includes(`max(mapDE(p) * G.prm.z, ${JAG_TERM_WGSL("p")})`),
    "march loop lost the noised term",
  );
  assert.ok(
    src.includes(
      `max(mapDE(ro + rd * mid) * G.prm.z, ${JAG_TERM_WGSL("ro + rd * mid")})`,
    ),
    "bisection lost the noised term",
  );
  assert.ok(
    src.includes(`let clipPl = ${JAG_TERM_WGSL("p")};`),
    "classification diverged from the march term",
  );
  // The noise functions are declared, 2 octaves at 2.03 lattice detune.
  for (const tok of JAG_TOKENS) assert.ok(src.includes(`fn ${tok}`), tok);
  assert.ok(src.includes("q * 2.03 + vec3f(13.5, 7.2, 3.1)"), "octave 2");
  // The cut-face shade itself is untouched (v1: plane-normal lightbox — flat
  // lighting over jagged contours reads fine; a noise-gradient normal is the
  // documented stretch goal).
  assert.ok(src.includes("if (cutFace) {"), "cut shade missing on jag");
});

test("GLSL: jag mirrors WGSL — uClipJ + the same substitution (flat and scene)", () => {
  for (const src of [
    buildFragGL(A, undefined, undefined, { clip: true, clipJag: true }),
    buildSceneFragGL(SCENE, { clip: true, clipJag: true }),
  ]) {
    assert.ok(src.includes("uniform vec3 uClipJ;"), "no uClipJ declaration");
    assert.ok(
      src.includes(
        `max(mapDE(ro + rd * t) * uDeScale, ${JAG_TERM_GL("ro + rd * t")})`,
      ),
      "GL march lost the noised term",
    );
    assert.ok(
      src.includes(
        `max(mapDE(ro + rd * mid) * uDeScale, ${JAG_TERM_GL("ro + rd * mid")})`,
      ),
      "GL bisection lost the noised term",
    );
    assert.ok(
      src.includes(`float clipPl = ${JAG_TERM_GL("p")};`),
      "GL classification diverged",
    );
    for (const tok of JAG_TOKENS)
      assert.ok(src.includes(`float ${tok}(vec3 q)`), tok);
  }
  // …and the FLAT clip variant carries none of it (sha-identical to the
  // pre-jag flat clip build).
  for (const [name, src] of [
    ["flat clip", buildFragGL(A, undefined, undefined, { clip: true })],
    ["scene clip", buildSceneFragGL(SCENE, { clip: true })],
  ]) {
    assert.ok(!src.includes("uClipJ"), `${name} leaked uClipJ`);
    for (const tok of JAG_TOKENS)
      assert.ok(!src.includes(tok), `${name} leaked ${tok}`);
  }
  assert.equal(
    sha(buildFragGL(A, undefined, undefined, { clip: true, clipJag: false })),
    sha(buildFragGL(A, undefined, undefined, { clip: true })),
  );
});

test("renderer: clipJag is its own bit (262144), summing cleanly with clip", () => {
  const base = { ops: null, leaves: null };
  const bitOf = (f) => Number(keyFor({ ...base, ...f }).split(":")[0]);
  assert.equal(bitOf({ clip: true }), 65536);
  assert.equal(
    bitOf({ clip: true, clipJag: true }),
    65536 + 262144,
    "clip+jag must be the SUM of the two bits",
  );
  // Above every earlier bit incl. equirect's 131072 — the next free power of
  // two after the two 2026-08-26 ledger claims.
  assert.equal(bitOf({ clipJag: true }), 262144);
  assert.equal(bitOf({ equirect: true }), 131072);
  assert.equal(wgslOf({ ...base, clip: true, clipJag: true }).clipJag, true);
  assert.equal(wgslOf({ ...base, clip: true }).clipJag, false);
});

test("frameparams: the jag words derive from the slider; presence is amount-keyed", () => {
  // Off (absent / 0): neutral words — amp 0, invD exactly 1, latch false.
  for (const L of [{}, { clipOn: true }, { clipOn: true, clipJag: 0 }]) {
    const d = deriveFrameParams({ light: L });
    assert.equal(d.clipJagAmp, 0);
    assert.equal(d.clipJagInv, 1);
    assert.equal(d.clipJag, false);
  }
  // Slider 1 with the plane on: amp = CLIP_JAG_WORLD, freq fixed, invD is
  // the full Lipschitz divisor, latch true.
  const on = deriveFrameParams({ light: { clipOn: true, clipJag: 1 } });
  assert.equal(on.clipJagAmp, CLIP_JAG_WORLD);
  assert.equal(on.clipJagFreq, CLIP_JAG_FREQ);
  assert.ok(
    Math.abs(
      on.clipJagInv - 1 / (1 + CLIP_JAG_LIP * CLIP_JAG_WORLD * CLIP_JAG_FREQ),
    ) < 1e-12,
  );
  assert.equal(on.clipJag, true);
  // Jag WITHOUT the plane never latches the variant (clip gates jag).
  assert.equal(deriveFrameParams({ light: { clipJag: 1 } }).clipJag, false);
  // Hostile slider clamps 0..1.
  assert.equal(
    deriveFrameParams({ light: { clipOn: true, clipJag: 9 } }).clipJagAmp,
    CLIP_JAG_WORLD,
  );
  assert.equal(
    deriveFrameParams({ light: { clipOn: true, clipJag: -3 } }).clipJagAmp,
    0,
  );
});

test("capturesettle: the jag latch mirrors frameparams (clip on AND amount > 0)", () => {
  const flat = { ops: [{ key: "scale", values: [2] }], iters: 4 };
  const at = (L) => frameFeaturesFor(flat, { mode: 0, light: L }).clipJag;
  assert.equal(at({ clipOn: true, clipJag: 0.4 }), true);
  assert.equal(at({ clipOn: true }), false);
  assert.equal(at({ clipOn: true, clipJag: 0 }), false);
  assert.equal(at({ clipJag: 0.4 }), false); // jag without the plane
  assert.equal(frameFeaturesFor(flat, { mode: 0 }).clipJag, false);
});

test("sanitize: clipJag clamps 0..1 and stays shape-preserving", () => {
  assert.equal(
    sanitizeColoring({ mode: 1, light: { clipJag: 7 } }).light.clipJag,
    1,
  );
  assert.equal(
    sanitizeColoring({ mode: 1, light: { clipJag: -2 } }).light.clipJag,
    0,
  );
  assert.equal(
    sanitizeColoring({ mode: 1, light: { clipJag: 0.35 } }).light.clipJag,
    0.35,
  );
  const abs = sanitizeColoring({ mode: 1, light: { ambient: 0.2 } });
  assert.ok(!("clipJag" in abs.light), "sanitize invented light.clipJag");
});

test("cpu: clipJagNoise is static, bounded and inside the Lipschitz claim", () => {
  // Deterministic (static world-space field — the bite pattern must hold
  // still while the sweep advances the plane through it)…
  assert.equal(clipJagNoise(0.3, -1.2, 2.7), clipJagNoise(0.3, -1.2, 2.7));
  // …bounded in [-1, 1]…
  let mx = 0;
  const rand = (() => {
    let s = 42;
    return () => (s = (s * 1103515245 + 12345) % 2147483648) / 2147483648;
  })();
  const pts = [];
  for (let i = 0; i < 4000; i++) {
    const p = [rand() * 8 - 4, rand() * 8 - 4, rand() * 8 - 4];
    const v = clipJagNoise(p[0], p[1], p[2]);
    assert.ok(v >= -1 && v <= 1, `noise out of range: ${v}`);
    pts.push([p, v]);
    mx = Math.max(mx, Math.abs(v));
  }
  assert.ok(mx > 0.3, "noise is degenerate (near-constant)");
  // …and its empirical slope stays under the CLIP_JAG_LIP bound the divisor
  // uses (the bound is worst-case analytic; sampled slopes must sit below it
  // or the DE-safety division is a fiction).
  const h = 1e-3;
  let steep = 0;
  for (const [p] of pts.slice(0, 1500)) {
    const v0 = clipJagNoise(p[0], p[1], p[2]);
    for (const d of [
      [h, 0, 0],
      [0, h, 0],
      [0, 0, h],
      [h * 0.577, h * 0.577, h * 0.577],
    ]) {
      const v1 = clipJagNoise(p[0] + d[0], p[1] + d[1], p[2] + d[2]);
      steep = Math.max(steep, Math.abs(v1 - v0) / Math.hypot(d[0], d[1], d[2]));
    }
  }
  assert.ok(
    steep < CLIP_JAG_LIP,
    `sampled slope ${steep.toFixed(2)} exceeds the CLIP_JAG_LIP bound ${CLIP_JAG_LIP}`,
  );
  assert.ok(steep > 0.5, "noise slope degenerate — the erosion would be mush");
});

test("cpu: jamp 0 degenerates to the exact flat cut, byte for byte", () => {
  const eye = eyeOf();
  const a = [0, 1, 2].reduce((m, i) => (Math.abs(eye[i]) < Math.abs(eye[m]) ? i : m), 0); // prettier-ignore
  const n = [0, 0, 0];
  n[a] = 1;
  const flat = shadeGrid(SPHERE, {
    ...GRID,
    clip: { nx: n[0], ny: n[1], nz: n[2], w: 0 },
  }).rgb;
  const jag0 = shadeGrid(SPHERE, {
    ...GRID,
    clip: { nx: n[0], ny: n[1], nz: n[2], w: 0, jamp: 0, jfreq: 3, jinv: 1 },
  }).rgb;
  assert.deepEqual(jag0, flat, "jamp 0 must not change one CPU pixel");
});

test("cpu: a jagged edge-on cut is ROUGH — the boundary varies row to row", () => {
  const eye = eyeOf();
  const a = [0, 1, 2].reduce((m, i) => (Math.abs(eye[i]) < Math.abs(eye[m]) ? i : m), 0); // prettier-ignore
  const n = [0, 0, 0];
  n[a] = 1;
  const amp = CLIP_JAG_WORLD; // slider 1
  const jag = {
    nx: n[0],
    ny: n[1],
    nz: n[2],
    w: 0,
    jamp: amp,
    jfreq: CLIP_JAG_FREQ,
    jinv: 1 / (1 + CLIP_JAG_LIP * amp * CLIP_JAG_FREQ),
  };
  const flatG = shadeGrid(SPHERE, {
    ...GRID,
    clip: { ...jag, jamp: 0, jinv: 1 },
  });
  const jagG = shadeGrid(SPHERE, { ...GRID, clip: jag });
  const perRow = (g) => {
    const out = [];
    for (let r = 0; r < GRID.rows; r++) {
      let c = 0;
      for (let x = 0; x < GRID.cols; x++)
        if (g.chars[r * GRID.cols + x] !== " ") c++;
      out.push(c);
    }
    return out;
  };
  const fr = perRow(flatG);
  const jr = perRow(jagG);
  // The noise moves the cut line by a DIFFERENT amount on different rows —
  // that per-row variation IS the roughness (a flat cut differs from another
  // flat cut only by a constant shift). Count distinct per-row deltas over
  // the rows that show the body.
  const deltas = [];
  for (let r = 0; r < GRID.rows; r++)
    if (fr[r] > 0 || jr[r] > 0) deltas.push(jr[r] - fr[r]);
  assert.ok(deltas.length > 10, "sphere missing from the grid");
  const distinct = new Set(deltas).size;
  assert.ok(
    distinct >= 3,
    `jagged boundary should vary row to row (distinct deltas: ${distinct})`,
  );
  // …while staying inside the amp envelope: the total silhouette change is a
  // fraction of the body, not a different picture (amp 0.3 on an R=0.5
  // sphere erodes the boundary band, never the far hemisphere).
  const covF = fr.reduce((x, y) => x + y, 0);
  const covJ = jr.reduce((x, y) => x + y, 0);
  assert.ok(covJ > 0.4 * covF && covJ < 1.6 * covF, `${covJ} vs ${covF}`);
});

test("renderer.js source: writeGlobals feeds the jag lanes (clipS.yzw)", async () => {
  const src = await readSrc("./renderer.js");
  assert.match(src, /gF\[CL \+ 5\] = d\.clipJagAmp/);
  assert.match(src, /gF\[CL \+ 6\] = d\.clipJagFreq/);
  assert.match(src, /gF\[CL \+ 7\] = d\.clipJagInv/);
});

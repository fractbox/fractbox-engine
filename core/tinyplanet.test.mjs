// TINY PLANET (stereographic projection) — the standing regression suite.
//
// Two obligations, both of them the reason this file exists rather than a
// comment somewhere:
//
//  1. THE GATE IS FREE. Ray generation lives at the top of the fragment shader
//     that also holds the march, so a never-executing uniform branch there is
//     exactly the shape the perf doctrine warns about (d00d9a7 / core/shader.js
//     header: the numeric-DE probe cost Mandelbox +7%, Mandelbulb +31% behind a
//     branch that never ran). The projection is therefore CODEGEN-gated, and
//     "off" must emit shader text with NOT ONE planet token — byte-identity is
//     the doctrine's "prove it's free" standard, and unlike a timing it does
//     not have a noise floor. Same contract as ENVX (envx.test.mjs) and NEON
//     (neon.test.mjs), tested the same way, across the same feature matrix.
//
//  2. THE FOUR MIRRORS AGREE. The map is hand-copied into core/shader.js
//     (WGSL), core/shader_gl.js (GLSL), core/cpu.js (JS) and core/preview.js
//     (pixelRay, for the click/zoom gestures). CI compiles neither shader tier,
//     so the CPU copy is the only one a node test can EXECUTE — it is the lever
//     for the geometry, and the two shader tiers are pinned by their text.
//
// The map, once, so a reader can check any mirror against this file:
//
//     u = wx·k , v = wy·k , q = u² + v² , k = tan(planetFov/4)
//     dir = ( 2u·right + 2v·up + (1−q)·fwd ) / (1 + q)
//
// the inverse stereographic projection from the image plane onto the sphere of
// directions, projected from the antipode of `fwd`. Landmarks: (0,0) → +fwd
// (the pole — screen centre looks at the orbit target, which is what makes the
// subject read as a planet), |(u,v)| = 1 → the equator (90° off-axis), and
// |(u,v)| → ∞ → −fwd (behind the camera).
import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";

import { buildWGSL } from "./shader.js";
import { buildFragGL, buildSceneFragGL } from "./shader_gl.js";
import { keyFor, wgslOf } from "./renderer.js";
import { frameFeaturesFor } from "./capturesettle.js";
import { exportStandaloneGLSL } from "./exportStandalone.js";
import { defaultColoring } from "./coloring.js";
import { shadeGrid } from "./cpu.js";
import {
  PLANET_FOV_MIN,
  PLANET_FOV_MAX,
  PLANET_FOV_DEFAULT,
} from "./preview.js";

const readSrc = async (f) =>
  (await import("node:fs/promises")).readFile(
    new URL(f, import.meta.url),
    "utf8",
  );
const sha = (s) => createHash("sha256").update(s).digest("hex");

// Tokens that exist ONLY in the planet variant. `pInv`/`pq` are the map's own
// locals; uPlanetK / camRight.w are how each tier carries the parameter.
const PLANET_TOKENS_WGSL = ["planetK", "pInv", "pq", "TINY PLANET"];
const PLANET_TOKENS_GL = ["uPlanetK", "pInv", "pq", "TINY PLANET"];

const SCENE = [{ shapeId: 1, objType: "sphere", combine: "union" }];

// ── 1. the gate ─────────────────────────────────────────────────────────────

// The SAME matrix envx/neon use, so a feature added to one is added to all
// three: every combination must be planet-free when the flag is off.
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
  ["scene", { scene: true, leaves: true }],
  ["hybrid", { hybrid: true }],
  ["morph", { morph: true }],
  ["ops:[]", { ops: [] }],
  ["ops:[1,2,3]", { ops: [1, 2, 3] }],
];

test("WGSL: planet off emits NOT ONE planet token, across the feature matrix", () => {
  for (const [name, opts] of WGSL_MATRIX) {
    const off = buildWGSL(opts);
    for (const tok of PLANET_TOKENS_WGSL)
      assert.ok(!off.includes(tok), `${name}: planet-off text leaked ${tok}`);
  }
});

test("WGSL: planet:false is byte-identical to omitting the flag", () => {
  for (const [name, opts] of WGSL_MATRIX) {
    assert.equal(
      sha(buildWGSL({ ...opts, planet: false })),
      sha(buildWGSL(opts)),
      `${name}: an explicit planet:false must not perturb one byte`,
    );
  }
});

test("WGSL: planet on actually swaps the ray generation (not a no-op flag)", () => {
  const off = buildWGSL();
  const on = buildWGSL({ planet: true });
  assert.notEqual(sha(off), sha(on), "planet:true emitted identical text");
  // The perspective arm's operands are GONE, not merely joined by a branch —
  // that is the difference between a codegen gate and a uniform one.
  assert.ok(on.includes("let planetK = G.camRight.w;"), "no planetK read");
  assert.ok(!on.includes("let orthoH = G.camFwd.w;"), "ortho arm survived");
  assert.ok(
    !on.includes("let tanF = tan(0.5 * G.res.z);"),
    "the planet variant must not emit the dead perspective tangent",
  );
  // …and the ortho arm keeps its own operands when planet is off.
  assert.ok(off.includes("let orthoH = G.camFwd.w;"), "ortho arm missing");
  assert.ok(off.includes("let tanF = tan(0.5 * G.res.z);"), "tanF missing");
});

test("GLSL: planet off emits NOT ONE planet token (flat, hybrid and scene)", () => {
  const shapes = [
    ["flat", buildFragGL([], null, undefined, {})],
    ["flat+envx", buildFragGL([], null, undefined, { envx: true })],
    ["flat+neon", buildFragGL([], null, undefined, { neon: true })],
    [
      "hybrid",
      buildFragGL(
        [{ key: "scale", values: [2] }],
        [{ ops: [] }],
        undefined,
        {},
      ),
    ],
    ["scene", buildSceneFragGL(SCENE, {})],
    ["scene+envx", buildSceneFragGL(SCENE, { envx: true })],
  ];
  for (const [name, src] of shapes)
    for (const tok of PLANET_TOKENS_GL)
      assert.ok(!src.includes(tok), `${name}: planet-off GLSL leaked ${tok}`);
});

test("GLSL: the planet variant declares uPlanetK, the off variant never does", () => {
  const on = buildFragGL([], null, undefined, { planet: true });
  assert.ok(on.includes("uniform float uPlanetK;"), "uPlanetK not declared");
  assert.ok(
    !on.includes("float tanF = tan(0.5 * uFov);"),
    "the planet variant must not emit the dead perspective tangent",
  );
  // Scenes get it too — ray generation happens before anything knows whether
  // the formula is flat or CSG, so there is no flat-only guard here (unlike
  // NEON, whose scene builder deliberately never emits it).
  assert.ok(
    buildSceneFragGL(SCENE, { planet: true }).includes(
      "uniform float uPlanetK;",
    ),
    "the scene builder must honour planet too",
  );
});

// ── 2. the mirrors ──────────────────────────────────────────────────────────

// Pin the map's ALGEBRA in both shader texts. Not a formatting check: these are
// the terms that decide where every pixel looks, and the two tiers are compiled
// nowhere in CI, so a hand-edit that "fixes" one and not the other would ship
// two different projections. (The ENVX suite pins its star math the same way.)
const MAP_TERMS_WGSL = [
  "let pq = (pu * pu) + (pv * pv);",
  "let pInv = 1.0 / (1.0 + pq);",
  "((2.0 * pu) * pInv) * G.camRight.xyz",
  "((2.0 * pv) * pInv) * G.camUp.xyz",
  "((1.0 - pq) * pInv) * G.camFwd.xyz",
];
const MAP_TERMS_GL = [
  "float pq = (pu * pu) + (pv * pv);",
  "float pInv = 1.0 / (1.0 + pq);",
  "((2.0 * pu) * pInv) * uCamRight",
  "((2.0 * pv) * pInv) * uCamUp",
  "((1.0 - pq) * pInv) * uCamFwd",
];

test("the stereographic map is pinned identically in WGSL and GLSL", () => {
  const w = buildWGSL({ planet: true });
  const g = buildFragGL([], null, undefined, { planet: true });
  for (const t of MAP_TERMS_WGSL)
    assert.ok(w.includes(t), `WGSL missing: ${t}`);
  for (const t of MAP_TERMS_GL) assert.ok(g.includes(t), `GLSL missing: ${t}`);
  // Both tiers scale the plane coordinate by k on BOTH axes, and the x axis
  // carries the aspect — the classic place a mirror drifts into a squashed
  // planet on one tier only.
  assert.ok(w.includes("let pu = wx * planetK;"), "WGSL pu");
  assert.ok(w.includes("let pv = wy * planetK;"), "WGSL pv");
  assert.ok(g.includes("float pu = (ndc.x * aspect) * uPlanetK;"), "GLSL pu");
  assert.ok(g.includes("float pv = ndc.y * uPlanetK;"), "GLSL pv");
});

// Collapse every balanced parenthesised group to `_`, innermost first. What is
// left is the line's TOP-LEVEL structure — so counting `*` in it counts the
// multiplications nobody grouped.
const topLevel = (line) => {
  let p = line,
    prev;
  do {
    prev = p;
    p = p.replace(/\([^()]*\)/g, "_");
  } while (p !== prev);
  return p;
};

test("every product in the emitted map is parenthesised (the blackout trap)", () => {
  // core/shader.js's own header records that an unparenthesised multiply has
  // blacked out every scene in this repo before, and WGSL/GLSL are compiled
  // nowhere in CI — so the grouping is checked here or not at all.
  const lines = [
    ...buildWGSL({ planet: true }).split("\n"),
    ...buildFragGL([], null, undefined, { planet: true }).split("\n"),
  ].filter((l) => /\bp(u|v|q|Inv)\b/.test(l) && !l.trim().startsWith("//"));
  assert.ok(lines.length >= 8, "map lines not found — did the emitter change?");
  for (const l of lines) {
    const chain = (topLevel(l).match(/\*/g) || []).length;
    assert.ok(chain <= 1, `ungrouped product chain: ${l.trim()}`);
  }
});

// ── 3. the CPU tier EXECUTES the geometry ───────────────────────────────────
//
// One sphere of radius R, camera at distance D looking at the origin, so the
// silhouette is exact: it subtends half-angle α = asin(R/D), and under the map
// its edge lands at plane radius tan(α/2). As a fraction of the frame's
// vertical half-extent that is tan(α/2)/k — a closed form to check the CPU
// mirror against, with no fractal in the way.
const R = 0.5;
const D = 4;
const SPHERE = {
  name: "planet-probe-sphere",
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
  aspect: 1, // square pixels — see traceGrid's `tile` note on the cell default
  ss: 1,
  edges: false,
  structure: false,
  dither: false,
  coloring: { mode: 1, autoLevels: false },
};

// The silhouette's radius in units of the frame's vertical half-extent, read
// off the rendered grid: scan the centre row outward for the last covered cell.
// A blank cell is a space; anything else is the sphere.
function discRadius(planetK) {
  const { cols, rows, chars } = shadeGrid(SPHERE, { ...GRID, planetK });
  const mid = (rows - 1) / 2;
  const row = chars.slice(mid * cols, (mid + 1) * cols);
  let last = -1;
  for (let c = 0; c < cols; c++) if (row[c] !== " ") last = c;
  if (last < 0) return 0;
  // cell centre → ndc, then |ndc| is already the fraction of the half-extent.
  return Math.abs(-1 + (2 * (last + 0.5)) / cols);
}

test("cpu: the disc lands where the closed form says (the map is the map)", () => {
  const alpha = Math.asin(R / D);
  const edge = Math.tan(alpha / 2); // plane radius of the silhouette
  for (const fovDeg of [90, 150, 220]) {
    const k = Math.tan((fovDeg * Math.PI) / 720);
    const want = edge / k;
    const got = discRadius(k);
    // One cell is 2/cols ≈ 0.025 of the half-extent; allow two.
    assert.ok(
      Math.abs(got - want) < 0.06,
      `fov ${fovDeg}: disc at ${got.toFixed(3)}, closed form ${want.toFixed(3)}`,
    );
  }
});

test("cpu: widening the planet FOV shrinks the disc, monotonically", () => {
  const fovs = [60, 90, 120, 150, 180, 240, 300];
  const radii = fovs.map((f) => discRadius(Math.tan((f * Math.PI) / 720)));
  for (let i = 1; i < radii.length; i++)
    assert.ok(
      radii[i] < radii[i - 1],
      `fov ${fovs[i]} did not wrap more than ${fovs[i - 1]} ` +
        `(${radii[i].toFixed(3)} vs ${radii[i - 1].toFixed(3)})`,
    );
  // …and the wrap is real: at 300° the subject is a small world in a big sky.
  assert.ok(radii.at(-1) < 0.25, "300° should leave mostly sky");
});

test("cpu: planetK 0 (and the absent field) render byte-identically", () => {
  const base = shadeGrid(SPHERE, GRID).rgb;
  assert.deepEqual(
    shadeGrid(SPHERE, { ...GRID, planetK: 0 }).rgb,
    base,
    "planetK: 0 must not change one CPU pixel",
  );
});

test("cpu: ortho wins over planet if both somehow arrive (deterministic)", () => {
  // The setters make this unreachable; the tier still has to be defined rather
  // than compounding two projections into nonsense.
  const both = shadeGrid(SPHERE, { ...GRID, orthoH: 1.2, planetK: 0.8 }).rgb;
  const orthoOnly = shadeGrid(SPHERE, { ...GRID, orthoH: 1.2 }).rgb;
  assert.deepEqual(both, orthoOnly);
});

// ── 4. variant keying ───────────────────────────────────────────────────────

test("renderer: planet is its own variant bit and reaches buildWGSL", () => {
  const base = { ops: null, leaves: null };
  assert.notEqual(
    keyFor({ ...base, planet: true }),
    keyFor({ ...base, planet: false }),
    "planet must not share a variant with perspective",
  );
  // It must not collide with any other single feature bit either.
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
    // AURORA (#667) and THIN FILM (#669) landed while this branch was open —
    // aurora took 8192, the bit this feature was originally written against,
    // and thin film then took 16384. F_PLANET renumbered to 32768; these two
    // entries are what makes that a CHECKED fact rather than a comment.
    "aurora",
    "thinFilm",
  ];
  const planetKey = keyFor({ ...base, planet: true });
  for (const b of bits)
    assert.notEqual(
      planetKey,
      keyFor({ ...base, [b]: true }),
      `collides: ${b}`,
    );
  assert.equal(wgslOf({ ...base, planet: true }).planet, true);
  assert.equal(wgslOf(base).planet, false);
  // The bit is its own power of two ABOVE every neighbour, so no combination
  // of the others can alias it — the property the #631 partition rule exists
  // to protect, checked rather than left to a comment.
  const bitOf = (b) => Number(keyFor({ ...base, [b]: true }).split(":")[0]);
  const planetBit = Number(planetKey.split(":")[0]);
  const others = bits.reduce((n, b) => n + bitOf(b), 0);
  assert.ok(
    planetBit > others,
    `F_PLANET (${planetBit}) must sit above every other bit (sum ${others})`,
  );
  // …and it COMPOSES with each of them rather than replacing it: an aurora
  // sky under a tiny planet is two bits, not one.
  for (const b of bits)
    assert.equal(
      Number(keyFor({ ...base, planet: true, [b]: true }).split(":")[0]),
      planetBit + bitOf(b),
      `planet + ${b} must be the SUM of two bits`,
    );
});

test("capturesettle: frameFeaturesFor mirrors the latch, and defaults off", () => {
  const f = { ops: [], iters: 8 };
  const c = defaultColoring();
  assert.equal(frameFeaturesFor(f, c).planet, false, "capture is perspective");
  assert.equal(frameFeaturesFor(f, c, { planet: true }).planet, true);
  // No scene guard: ray-gen runs before anything knows which kind it is.
  const scene = { objects: SCENE, ops: [], iters: 8 };
  assert.equal(frameFeaturesFor(scene, c, { planet: true }).planet, true);
});

test("standalone export: a bundle is always PERSPECTIVE", () => {
  const g = exportStandaloneGLSL(
    { name: "x", ops: [{ key: "scale", values: [2] }], iters: 6 },
    { light: defaultColoring().light },
  );
  for (const tok of PLANET_TOKENS_GL)
    assert.ok(!g.includes(tok), `standalone bundle leaked ${tok}`);
});

// ── 5. the source-level contracts (grep gates) ──────────────────────────────
//
// The #489 precedent: some of this feature's correctness lives in WHICH branch
// of preview.js does what, and those branches are DOM-bound (no node harness).
// Gate them on the source text, the way perturb.test.mjs gates the ortho zoom
// fork it could not execute either.

test("preview.js: the two projections are mutually exclusive, in the setters", async () => {
  const src = await readSrc("./preview.js");
  assert.match(
    src,
    /setPlanet\(fovDeg, \{ frame = false \} = \{\}\)\s*\{[\s\S]*?orthoH = 0;/,
    "setPlanet must clear orthoH",
  );
  assert.match(
    src,
    /setOrtho\(h\)\s*\{[\s\S]*?if \(v > 0\) planetFov = 0;/,
    "setOrtho must clear planetFov",
  );
});

test("preview.js: an ORBIT clears ortho and KEEPS the planet (#441 divergence)", async () => {
  const src = await readSrc("./preview.js");
  const orbit = src.slice(src.indexOf('dragMode = "orbit";'));
  const body = orbit.slice(0, orbit.indexOf("cam.orbit(dx, dy);"));
  assert.ok(body.includes("orthoH = 0;"), "the orbit branch must drop ortho");
  assert.ok(
    !body.includes("planetFov = 0"),
    "the orbit branch must NOT drop the planet — orbiting IS how you spin it",
  );
});

test("preview.js: the wheel forks to the plain dolly under the planet", async () => {
  const src = await readSrc("./preview.js");
  assert.match(
    src,
    /if \(planetFov > 0\) \{\s*wheelProbe = null;\s*zoomAtCenter\(factor\);\s*return;\s*\}/,
    "cursor-anchored zoom is perspective-only math; the planet must route to " +
      "zoomAtCenter and skip the CPU-DE probe it would discard",
  );
});

test("preview.js: pixelRay knows the projection (or clicks land on the wrong pixel)", async () => {
  const src = await readSrc("./preview.js");
  const ray = src.slice(src.indexOf("const pixelRay = ("));
  const body = ray.slice(0, ray.indexOf("// Probe the surface along"));
  assert.ok(body.includes("if (planetFov > 0)"), "pixelRay has no planet arm");
  assert.ok(
    body.includes("const pInv = 1 / (1 + pq);"),
    "pixelRay's arm must use the same map as the shaders",
  );
  assert.ok(
    /if \(planetFov > 0\)[\s\S]*?if \(orthoH > 0\)/.test(body),
    "the planet arm must precede the ortho arm (they are exclusive)",
  );
});

test("preview.js: planetK reaches the settle payload and the prewarm prediction", async () => {
  const src = await readSrc("./preview.js");
  assert.ok(
    src.includes("planetK: planetK(), // TINY PLANET"),
    "settleFrame's live bag must carry planetK",
  );
  assert.ok(
    src.includes("planet: planetK() > 0,"),
    "frameFeatures must predict the variant, or the pump holds the wrong one",
  );
});

test("stills, Record and offline exports inherit the projection", async () => {
  const src = await readSrc("./preview.js");
  // The projection is in the RENDER, not a post-pass, so nothing has to be
  // arranged per export path — but only because there is ONE place the live
  // options bag is built. A new settleFrame call would carry its own bag, and
  // some export would silently go perspective.
  //
  // Two call sites exactly, and the second is a deliberate exception: the
  // splat G-buffer capture settles with NO live bag at all, because it is a
  // still of the FORMULA rather than of the view (fsCapture marches CaptureU's
  // own rays; renderer.js forces planet:false on that pipeline for the same
  // reason). If this count moves, decide which kind the new one is.
  const settles = src.match(/^\s*settleFrame\(/gm) || [];
  assert.equal(
    settles.length,
    2,
    "settleFrame call sites moved — re-check which ones carry the live bag",
  );
  assert.match(
    src,
    /Direct settleFrame with NO live bag/,
    "the splat-capture exception must stay documented at its call site",
  );
  // …and every still/export path renders through writeFrame with NO camera
  // override, i.e. through that bag.
  const wf = src.match(/writeFrame\(/g) || [];
  assert.ok(wf.length > 3, "writeFrame call sites not found — did they move?");
  // Signature re-pinned 2026-08-31: `col` is the thumbnail path's per-tile
  // look, threaded through to settleFrame (`col || coloring`) instead of the
  // old swap-and-restore of the module coloring (a data race that parked a
  // TILE's look as the session coloring — the "same link, sometimes blue
  // sometimes green" field report). Fall-through re-checked: every
  // still/export caller omits it, so they render the live coloring, the live
  // cam (`c || cam`) and the ONE live options bag exactly as before.
  assert.ok(
    /function writeFrame\(f, q, res, c, col\) \{/.test(src),
    "writeFrame's signature changed; re-check that stills still fall through " +
      "to the live cam AND the live options bag",
  );
});

test("renderer.js: planetK rides camRight.w and drives the codegen latch", async () => {
  const src = await readSrc("./renderer.js");
  assert.match(src, /gF\[15\] = planetK;/, "camRight.w must carry planetK");
  assert.match(src, /planet: planetK > 0,/, "activeFeat must latch off it");
  assert.match(
    src,
    // 360° equirect joined the same forced-off list when it landed; the
    // planet half of the contract is unchanged.
    /planet: false,\s*equirect: false,\s*capture: true/,
    "splat capture marches its own rays — it must not fork on planet",
  );
});

test("renderer_gl.js: the planet relinks in BOTH directions", async () => {
  const src = await readSrc("./renderer_gl.js");
  // ENVX/NEON relink upward only (their on-variant degrades to the off
  // picture). The planet variant has no perspective arm, so leaving it linked
  // with uPlanetK = 0 would collapse every ray onto +fwd — one flat colour.
  assert.match(src, /planetWant !== progPlanet/, "upward-only would strand it");
  assert.match(src, /if \(progPlanet\) gl\.uniform1f\(U\("uPlanetK"\)/);
});

test("streamlines.js: the overlay projects through the same map", async () => {
  const src = await readSrc("./streamlines.js");
  // A particle overlay that stayed perspective under the planet would scatter
  // its sprites across the wrong pixels — silently, like pixelRay would.
  assert.match(src, /let planetK = G\.camRight\.w;/);
  assert.match(src, /let den = 1\.0 \+ dz;/, "forward map missing");
  // The cull line grew a third clause when 360° equirect landed (its rays
  // reach behind the eye too) — the planet contract is unchanged: z <= 0 must
  // not cull while planetK > 0.
  assert.match(
    src,
    /if \(z <= 1\.0e-4 && planetK <= 0\.0 && equirectS <= 0\.0\) \{ return o; \}/,
    "the in-front cull must not eat the planet's sky half",
  );
});

// ── 6. the range ────────────────────────────────────────────────────────────

test("preview.js: the framing assist declines rather than burying the camera", async () => {
  const src = await readSrc("./preview.js");
  const fn = src.slice(src.indexOf("const dropToPlanetSurface = () => {"));
  const body = fn.slice(0, fn.indexOf("\n  };"));
  // Three guards, each of which a cruder earlier rule got wrong on some
  // preset. They are the difference between an assist that is safe to run on
  // EVERY switch-on and one that has to be opt-in.
  assert.ok(
    body.includes("if (!(alpha > 0)) return false;"),
    "empty space ahead must DECLINE, not dive at nothing",
  );
  assert.ok(
    body.includes("if (alpha >= PLANET_TARGET_ALPHA) return false;"),
    "a subject that already fills the view must be left alone — diving into a " +
      "lattice preset buries the camera (measured: Menger 9.0 → 1.3, edgeSky 0.115)",
  );
  assert.ok(
    body.includes("want = Math.max(want, cam.dist - h * 0.95);"),
    "the eye must stay outside the surface straight ahead",
  );
  assert.ok(
    body.includes("Math.max(ptMinDist(), want)"),
    "the precision floor still applies — this is a zoom like any other",
  );
  // And it must be one-way: nothing here restores a remembered distance.
  assert.ok(
    !/restore|previousDist|distBefore/.test(body),
    "the drop is deliberately not undone on switch-off (see its header)",
  );
});

test("preview.js: the framing assist runs on switch-ON only", async () => {
  const src = await readSrc("./preview.js");
  // A re-dive on every slider input would make the FOV control unusable, so
  // the caller has to ask for it and the app only asks from the toggle.
  assert.match(src, /if \(frame\) dropToPlanetSurface\(\);/);
});

test("the FOV range stops short of the map's singularity", () => {
  assert.ok(PLANET_FOV_MAX < 360, "tan(360/4) is infinite — 360 is unusable");
  assert.ok(PLANET_FOV_MIN > 0);
  assert.ok(
    PLANET_FOV_DEFAULT >= PLANET_FOV_MIN &&
      PLANET_FOV_DEFAULT <= PLANET_FOV_MAX,
  );
  // The frame's vertical edge looks out at planetFov/2 from the pole, so the
  // default has to be wide enough that the picture genuinely WRAPS rather than
  // merely crops. 120° puts the edge 60° off-axis (two thirds of the way to
  // the horizon); the exact value is pinned by app/scripts/planet-probe.mjs,
  // which measures disc radius and sky coverage on real WebGPU frames.
  assert.ok(
    PLANET_FOV_DEFAULT >= 120,
    "the default must wrap, not just crop — see the probe sweep in the PR",
  );
});

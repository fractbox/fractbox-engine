// 360° EQUIRECTANGULAR (mono v1) — the standing regression suite. The second
// arm on tiny planet's ray-gen seam, tested to the tinyplanet.test.mjs
// standard and structure — the two files are deliberately siblings:
//
//  1. THE GATE IS FREE. Ray generation sits at the top of the fragment shader
//     that also holds the march (the #125 / d00d9a7 register-pressure
//     surface), so the projection is CODEGEN-gated and "off" must emit shader
//     text with NOT ONE equirect token — byte-identity across the feature
//     matrix, planet included, is the doctrine's "prove it's free" standard.
//
//  2. THE FOUR MIRRORS AGREE. The map is hand-copied into core/shader.js
//     (WGSL), core/shader_gl.js (GLSL), core/cpu.js (JS) and core/preview.js
//     (pixelRay). CI compiles neither shader tier, so the CPU copy is the one
//     a node test can EXECUTE — the geometry (pole rows, the closed-form disc
//     extents, and above all the WRAPAROUND: a subject behind the camera must
//     split across the left and right edges) is checked there.
//
// The map, once, so a reader can check any mirror against this file:
//
//     lon = wx · eqS          eqS = π·H/W of the FULL image (WGSL camUp.w;
//     lat = wy · π/2               the GL tier's compile-time constant π/aspect)
//     dir = cos(lat)·( sin(lon)·right + cos(lon)·fwd ) + sin(lat)·up
//
// Landmarks: (0,0) → +fwd (image centre = the orbit target — orbiting rotates
// the panorama), wy = ±1 → ±up (the poles: whole rows converge on one
// direction), the frame's left and right edges → −fwd from both sides.
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

const readSrc = async (f) =>
  (await import("node:fs/promises")).readFile(
    new URL(f, import.meta.url),
    "utf8",
  );
const sha = (s) => createHash("sha256").update(s).digest("hex");

// Tokens that exist ONLY in the equirect variant. eqLon/eqLat/eqCl are the
// map's own locals; eqS / camUp.w is how the WGSL tier carries the scale (the
// GL tier bakes it, so its tokens are the locals alone).
const EQ_TOKENS_WGSL = ["eqS", "eqLon", "eqLat", "eqCl", "360 EQUIRECT"];
const EQ_TOKENS_GL = ["eqLon", "eqLat", "eqCl", "360 EQUIRECT"];

const SCENE = [{ shapeId: 1, objType: "sphere", combine: "union" }];

// ── 1. the gate ─────────────────────────────────────────────────────────────

// The tinyplanet matrix PLUS the planet itself — a feature that landed on the
// same seam is exactly the neighbour a splice regression would hide in.
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
  ["planet", { planet: true }],
  ["planet+aurora+film", { planet: true, aurora: true, thinFilm: true }],
];

test("WGSL: equirect off emits NOT ONE equirect token, across the matrix", () => {
  for (const [name, opts] of WGSL_MATRIX) {
    const off = buildWGSL(opts);
    for (const tok of EQ_TOKENS_WGSL)
      assert.ok(!off.includes(tok), `${name}: equirect-off text leaked ${tok}`);
  }
});

test("WGSL: equirect:false is byte-identical to omitting the flag", () => {
  for (const [name, opts] of WGSL_MATRIX) {
    assert.equal(
      sha(buildWGSL({ ...opts, equirect: false })),
      sha(buildWGSL(opts)),
      `${name}: an explicit equirect:false must not perturb one byte`,
    );
  }
});

test("WGSL: equirect on actually swaps the ray generation (not a no-op flag)", () => {
  const off = buildWGSL();
  const on = buildWGSL({ equirect: true });
  assert.notEqual(sha(off), sha(on), "equirect:true emitted identical text");
  // The perspective/ortho arm's operands are GONE, not merely joined by a
  // branch — the codegen-gate contract, same as the planet's.
  assert.ok(on.includes("let eqS = G.camUp.w;"), "no eqS read");
  assert.ok(!on.includes("let orthoH = G.camFwd.w;"), "ortho arm survived");
  assert.ok(
    !on.includes("let tanF = tan(0.5 * G.res.z);"),
    "the equirect variant must not emit the dead perspective tangent",
  );
  // …and no PLANET tokens either: one projection per module.
  assert.ok(!on.includes("planetK"), "planet tokens leaked into equirect");
});

test("WGSL/GLSL: planet and equirect together THROW (exclusive projections)", () => {
  assert.throws(() => buildWGSL({ planet: true, equirect: true }));
  assert.throws(() =>
    buildFragGL([], null, undefined, { planet: true, equirect: true }),
  );
  assert.throws(() =>
    buildSceneFragGL(SCENE, { planet: true, equirect: true }),
  );
});

test("GLSL: equirect off emits NOT ONE equirect token (flat, hybrid, scene, planet)", () => {
  const shapes = [
    ["flat", buildFragGL([], null, undefined, {})],
    ["flat+envx", buildFragGL([], null, undefined, { envx: true })],
    ["flat+planet", buildFragGL([], null, undefined, { planet: true })],
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
    ["scene+planet", buildSceneFragGL(SCENE, { planet: true })],
  ];
  for (const [name, src] of shapes)
    for (const tok of EQ_TOKENS_GL)
      assert.ok(!src.includes(tok), `${name}: equirect-off GLSL leaked ${tok}`);
});

test("GLSL: the equirect variant carries the map and drops the tangent", () => {
  const on = buildFragGL([], null, undefined, { equirect: true });
  assert.ok(on.includes("float eqLon = ndc.x * 3.141592653589793;"));
  assert.ok(
    !on.includes("float tanF = tan(0.5 * uFov);"),
    "the equirect variant must not emit the dead perspective tangent",
  );
  // Unlike the planet it declares NO uniform: the GL tier never tiles, so the
  // longitude scale is a compile-time constant, and a stray uniform here
  // would be a drift from that design.
  assert.ok(!on.includes("uEquirect"), "equirect must not declare a uniform");
  // Scenes get it too — ray generation runs before flat/CSG matters.
  assert.ok(
    buildSceneFragGL(SCENE, { equirect: true }).includes("float eqLon"),
    "the scene builder must honour equirect too",
  );
});

// ── 2. the mirrors ──────────────────────────────────────────────────────────

const MAP_TERMS_WGSL = [
  "let eqLon = wx * eqS;",
  "let eqLat = wy * 1.5707963267948966;",
  "let eqCl = cos(eqLat);",
  "(eqCl * sin(eqLon)) * G.camRight.xyz",
  "(sin(eqLat)) * G.camUp.xyz",
  "(eqCl * cos(eqLon)) * G.camFwd.xyz",
];
const MAP_TERMS_GL = [
  "float eqLon = ndc.x * 3.141592653589793;",
  "float eqLat = ndc.y * 1.5707963267948966;",
  "float eqCl = cos(eqLat);",
  "(eqCl * sin(eqLon)) * uCamRight",
  "(sin(eqLat)) * uCamUp",
  "(eqCl * cos(eqLon)) * uCamFwd",
];

test("the lat-long map is pinned identically in WGSL and GLSL", () => {
  const w = buildWGSL({ equirect: true });
  const g = buildFragGL([], null, undefined, { equirect: true });
  for (const t of MAP_TERMS_WGSL)
    assert.ok(w.includes(t), `WGSL missing: ${t}`);
  for (const t of MAP_TERMS_GL) assert.ok(g.includes(t), `GLSL missing: ${t}`);
});

// Collapse balanced parens innermost-first; what is left is the line's
// top-level structure, so counting `*` counts the ungrouped products (the
// blackout trap — see tinyplanet.test.mjs and core/shader.js's own header).
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
  const lines = [
    ...buildWGSL({ equirect: true }).split("\n"),
    ...buildFragGL([], null, undefined, { equirect: true }).split("\n"),
  ].filter((l) => /\beq(Lon|Lat|Cl|S)\b/.test(l) && !l.trim().startsWith("//"));
  assert.ok(lines.length >= 8, "map lines not found — did the emitter change?");
  for (const l of lines) {
    const chain = (topLevel(l).match(/\*/g) || []).length;
    assert.ok(chain <= 1, `ungrouped product chain: ${l.trim()}`);
  }
});

// ── 3. the CPU tier EXECUTES the geometry ───────────────────────────────────
//
// Camera convention (core/camera.js): world up is +Z; yaw 0 / pitch 0 / dist D
// targeting the origin puts the eye at (0, D, 0) with fwd (0,−1,0),
// up (0,0,1), right (−1,0,0). A sphere of radius R at a known offset from the
// EYE therefore lands at a closed-form (lon, lat) — the "known direction →
// known pixel" validation the visual probe can't do mathematically.
const R = 0.5;
const D = 4;
const sphereAt = (origin, r = R) => ({
  name: "equirect-probe-sphere",
  objects: [
    {
      shapeId: 2,
      shapeParams: [r, 0, 0, 0],
      origin,
      combine: "union",
      ops: [],
      iters: 1,
    },
  ],
  ops: [],
  iters: 1,
  camera: { yawDeg: 0, pitchDeg: 0, dist: D, fovDeg: 42 },
});
const GRID = {
  cols: 81,
  rows: 81,
  aspect: 1, // square grid — lon spans ±π across it, lat ±π/2 (see below)
  ss: 1,
  edges: false,
  structure: false,
  dither: false,
  coloring: { mode: 1, autoLevels: false },
};

const covered = (chars, cols, r, c) => chars[r * cols + c] !== " ";
// Horizontal / vertical half-extent of the coverage through the grid centre,
// in ndc units (fractions of the half-extent), like tinyplanet's discRadius.
function extents(formula) {
  const { cols, rows, chars } = shadeGrid(formula, { ...GRID, equirect: true });
  const midR = (rows - 1) / 2;
  const midC = (cols - 1) / 2;
  let lastC = -1;
  for (let c = 0; c < cols; c++)
    if (covered(chars, cols, Math.round(midR), c)) lastC = c;
  let lastRUp = -1; // rows above centre (toward +lat)
  for (let r = Math.floor(midR); r >= 0; r--)
    if (covered(chars, cols, r, Math.round(midC))) lastRUp = midR - r;
  return {
    cols,
    rows,
    chars,
    h: lastC < 0 ? 0 : Math.abs(-1 + (2 * (lastC + 0.5)) / cols),
    v: lastRUp < 0 ? 0 : (2 * lastRUp) / rows,
  };
}

test("cpu: the on-axis disc lands where the closed form says — on BOTH axes", () => {
  // Sphere at the orbit target: α = asin(R/D) off-axis. Under the map the
  // silhouette's horizontal half-extent is lon/π = α/π of the half-width and
  // the vertical is lat/(π/2) = 2α/π of the half-height — the 2:1 angular
  // anisotropy on a square grid is a FINGERPRINT of the equirect map (no
  // other projection in this repo produces that pair).
  const alpha = Math.asin(R / D);
  const { h, v } = extents(sphereAt([0, 0, 0]));
  const cell = 2 / GRID.cols;
  assert.ok(
    Math.abs(h - alpha / Math.PI) < 2.5 * cell,
    `horizontal extent ${h.toFixed(3)}, closed form ${(alpha / Math.PI).toFixed(3)}`,
  );
  assert.ok(
    Math.abs(v - (2 * alpha) / Math.PI) < 2.5 * cell,
    `vertical extent ${v.toFixed(3)}, closed form ${((2 * alpha) / Math.PI).toFixed(3)}`,
  );
});

test("cpu: straight up covers the ENTIRE top row (pole convergence), and only the poleward rows", () => {
  // Sphere directly above the EYE (eye = (0, D, 0), up = +Z): every longitude
  // converges on the pole, so the top row must be covered at EVERY column —
  // "a known direction lands at the correct pixel row", and the no-pinch
  // pole behaviour, in one check.
  const f = sphereAt([0, D, 2], 0.4);
  const { cols, rows, chars } = shadeGrid(f, { ...GRID, equirect: true });
  for (let c = 0; c < cols; c++)
    assert.ok(covered(chars, cols, 0, c), `top row gap at column ${c}`);
  // …and the equator row is empty (the sphere subtends nowhere near 90°).
  const midR = Math.round((rows - 1) / 2);
  for (let c = 0; c < cols; c++)
    assert.ok(!covered(chars, cols, midR, c), `equator hit at column ${c}`);
});

test("cpu: a known 45°-up direction lands at the closed-form row", () => {
  // Sphere centred at eye + (fwd+up)/√2 · d ⇒ lat = 45°, lon = 0 ⇒ disc
  // centre at ndcY = 0.5, ndcX = 0. Eye = (0, D, 0), fwd = (0,−1,0),
  // up = (0,0,1) ⇒ origin = (0, D − d/√2, d/√2).
  const d = 3;
  const s = d / Math.SQRT2;
  const f = sphereAt([0, D - s, s], 0.25);
  const { cols, rows, chars } = shadeGrid(f, { ...GRID, equirect: true });
  // Find the covered centroid.
  let sr = 0,
    sc = 0,
    n = 0;
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++)
      if (covered(chars, cols, r, c)) {
        sr += r;
        sc += c;
        n++;
      }
  assert.ok(n > 0, "45°-up sphere not visible at all");
  const ndcY = 1 - (2 * (sr / n + 0.5)) / rows;
  const ndcX = -1 + (2 * (sc / n + 0.5)) / cols;
  assert.ok(
    Math.abs(ndcY - 0.5) < 0.05,
    `centroid row at ndcY ${ndcY.toFixed(3)}, want 0.5`,
  );
  assert.ok(
    Math.abs(ndcX) < 0.05,
    `centroid col at ndcX ${ndcX.toFixed(3)}, want 0`,
  );
});

test("cpu: a subject BEHIND the camera splits across the left and right edges (wraparound)", () => {
  // Sphere behind the eye (direction −fwd = (0,1,0) from the eye): lon = ±π,
  // so it must appear at BOTH horizontal edges of the frame and nowhere near
  // the centre — the killer correctness check for a 360 photo (the two edges
  // are the same seam and must agree).
  const f = sphereAt([0, D + 3, 0], 0.8);
  const { cols, rows, chars } = shadeGrid(f, { ...GRID, equirect: true });
  const midR = Math.round((rows - 1) / 2);
  assert.ok(covered(chars, cols, midR, 0), "left edge empty");
  assert.ok(covered(chars, cols, midR, cols - 1), "right edge empty");
  assert.ok(
    !covered(chars, cols, midR, Math.round((cols - 1) / 2)),
    "centre should be empty — the subject is behind the camera",
  );
  // Symmetry of the split: the two half-widths agree to a cell (the seam is
  // one direction seen from both sides).
  let leftW = 0;
  while (leftW < cols && covered(chars, cols, midR, leftW)) leftW++;
  let rightW = 0;
  while (rightW < cols && covered(chars, cols, midR, cols - 1 - rightW))
    rightW++;
  assert.ok(
    Math.abs(leftW - rightW) <= 1,
    `wrap halves disagree: left ${leftW} vs right ${rightW} cells`,
  );
});

test("cpu: equirect false (and the absent field) render byte-identically", () => {
  const f = sphereAt([0, 0, 0]);
  const base = shadeGrid(f, GRID).rgb;
  assert.deepEqual(
    shadeGrid(f, { ...GRID, equirect: false }).rgb,
    base,
    "equirect: false must not change one CPU pixel",
  );
});

test("cpu: ortho and planet each win over equirect (deterministic precedence)", () => {
  const f = sphereAt([0, 0, 0]);
  assert.deepEqual(
    shadeGrid(f, { ...GRID, orthoH: 1.2, equirect: true }).rgb,
    shadeGrid(f, { ...GRID, orthoH: 1.2 }).rgb,
  );
  assert.deepEqual(
    shadeGrid(f, { ...GRID, planetK: 0.8, equirect: true }).rgb,
    shadeGrid(f, { ...GRID, planetK: 0.8 }).rgb,
  );
});

// ── 4. variant keying ───────────────────────────────────────────────────────

test("renderer: equirect is its own variant bit and reaches buildWGSL", () => {
  const base = { ops: null, leaves: null };
  assert.notEqual(
    keyFor({ ...base, equirect: true }),
    keyFor({ ...base, equirect: false }),
    "equirect must not share a variant with perspective",
  );
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
  const eqKey = keyFor({ ...base, equirect: true });
  for (const b of bits)
    assert.notEqual(eqKey, keyFor({ ...base, [b]: true }), `collides: ${b}`);
  assert.equal(wgslOf({ ...base, equirect: true }).equirect, true);
  assert.equal(wgslOf(base).equirect, false);
  // Its own power of two above every neighbour — no combination can alias it.
  // NOTE the deliberate GAP: 65536 is claimed IN FLIGHT by the concurrent
  // clipping-plane branch (allocation ledger, PR body), so the bit here must
  // be MORE than double the highest landed bit (F_PLANET 32768).
  const bitOf = (b) => Number(keyFor({ ...base, [b]: true }).split(":")[0]);
  const eqBit = Number(eqKey.split(":")[0]);
  assert.equal(eqBit, 131072, "F_EQUIRECT moved off its pre-assigned bit");
  const others = bits.reduce((n, b) => n + bitOf(b), 0);
  assert.ok(
    eqBit > others + 65536,
    `F_EQUIRECT (${eqBit}) must clear every landed bit AND the in-flight 65536`,
  );
  // …and it COMPOSES with each of them (an aurora sky in a panorama is two
  // bits, not one) — except planet, which is exclusive by construction.
  for (const b of bits.filter((x) => x !== "planet"))
    assert.equal(
      Number(keyFor({ ...base, equirect: true, [b]: true }).split(":")[0]),
      eqBit + bitOf(b),
      `equirect + ${b} must be the SUM of two bits`,
    );
});

test("capturesettle: frameFeaturesFor mirrors the latch, and defaults off", () => {
  const f = { ops: [], iters: 8 };
  const c = defaultColoring();
  assert.equal(
    frameFeaturesFor(f, c).equirect,
    false,
    "capture is perspective",
  );
  assert.equal(frameFeaturesFor(f, c, { equirect: true }).equirect, true);
  // No scene guard: ray-gen runs before anything knows which kind it is.
  const scene = { objects: SCENE, ops: [], iters: 8 };
  assert.equal(frameFeaturesFor(scene, c, { equirect: true }).equirect, true);
});

test("standalone export: a bundle is always PERSPECTIVE", () => {
  const g = exportStandaloneGLSL(
    { name: "x", ops: [{ key: "scale", values: [2] }], iters: 6 },
    { light: defaultColoring().light },
  );
  for (const tok of EQ_TOKENS_GL)
    assert.ok(!g.includes(tok), `standalone bundle leaked ${tok}`);
});

// ── 5. the source-level contracts (grep gates) ──────────────────────────────
//
// The #489 precedent, via tinyplanet.test.mjs: the DOM-bound branches of
// preview.js have no node harness, so their contracts are gated on the source.

test("preview.js: the THREE projections are mutually exclusive, in the setters", async () => {
  const src = await readSrc("./preview.js");
  assert.match(
    src,
    /setEquirect\(on, \{ frame = false \} = \{\}\)\s*\{[\s\S]*?orthoH = 0;[\s\S]*?planetFov = 0;/,
    "setEquirect must clear ortho AND planet",
  );
  assert.match(
    src,
    /setPlanet\(fovDeg[\s\S]*?equirectOn = false;/,
    "setPlanet must clear equirect",
  );
  assert.match(
    src,
    /setOrtho\(h\)\s*\{[\s\S]*?equirectOn = false;/,
    "setOrtho must clear equirect",
  );
});

test("preview.js: an ORBIT keeps the panorama (the planet divergence, again)", async () => {
  const src = await readSrc("./preview.js");
  const orbit = src.slice(src.indexOf('dragMode = "orbit";'));
  const body = orbit.slice(0, orbit.indexOf("cam.orbit(dx, dy);"));
  assert.ok(body.includes("orthoH = 0;"), "the orbit branch must drop ortho");
  assert.ok(
    !body.includes("equirectOn = false"),
    "the orbit branch must NOT drop equirect — orbiting IS looking around",
  );
});

test("preview.js: the wheel forks to the plain dolly under equirect", async () => {
  const src = await readSrc("./preview.js");
  assert.match(
    src,
    /if \(equirectOn\) \{\s*wheelProbe = null;\s*zoomAtCenter\(factor\);\s*return;\s*\}/,
    "equirect zoom is a centre dolly (there is no FOV to crop, deliberately)",
  );
});

test("preview.js: pixelRay knows the projection (or clicks land on the wrong pixel)", async () => {
  const src = await readSrc("./preview.js");
  const ray = src.slice(src.indexOf("const pixelRay = ("));
  const body = ray.slice(0, ray.indexOf("// Probe the surface along"));
  assert.ok(body.includes("if (equirectOn)"), "pixelRay has no equirect arm");
  assert.ok(
    body.includes("const lon = ndcX * Math.PI,"),
    "pixelRay's arm must use the same map as the shaders",
  );
});

test("preview.js: equirectS reaches the settle payload, the prewarm, and the tiled plate dims", async () => {
  const src = await readSrc("./preview.js");
  assert.ok(
    src.includes("equirectS: equirectS(res), // 360° EQUIRECT"),
    "settleFrame's live bag must carry equirectS",
  );
  assert.ok(
    src.includes("equirect: equirectOn,"),
    "frameFeatures must predict the variant, or the pump holds the wrong one",
  );
  // The tiled path must park the PLATE's dims — writeFrame sees the tile's
  // res, and π·H/W of a tile is the wrong longitude scale (a silently
  // mis-scaled panorama, the worst kind of wrong).
  assert.match(src, /tiledFullDims = \[W, H\];[\s\S]*?writeFrame\(/);
  assert.ok(
    src.includes("tiledFullDims = null; // 360° EQUIRECT"),
    "…and clear them in the finally",
  );
});

test("renderer.js: equirectS rides camUp.w and drives the codegen latch", async () => {
  const src = await readSrc("./renderer.js");
  assert.match(src, /gF\[19\] = equirectS;/, "camUp.w must carry equirectS");
  assert.match(src, /equirect: equirectS > 0,/, "activeFeat must latch off it");
  assert.match(
    src,
    /planet: false,\s*equirect: false,\s*capture: true/,
    "splat capture marches its own rays — it must not fork on equirect",
  );
});

test("renderer_gl.js: equirect relinks in BOTH directions", async () => {
  const src = await readSrc("./renderer_gl.js");
  // Like the planet (and unlike envx/neon): the variant has no perspective
  // arm, so a downgrade cannot wait for the next structural rebuild.
  assert.match(src, /equirectWant !== progEquirect/, "upward-only strands it");
});

test("streamlines.js: the overlay projects through the same map", async () => {
  const src = await readSrc("./streamlines.js");
  assert.match(src, /let equirectS = G\.camUp\.w;/);
  assert.match(
    src,
    /let lon = atan2\(dot\(d, G\.camRight\.xyz\), dot\(d, G\.camFwd\.xyz\)\);/,
    "forward map missing",
  );
  assert.match(
    src,
    /if \(z <= 1\.0e-4 && planetK <= 0\.0 && equirectS <= 0\.0\) \{ return o; \}/,
    "the in-front cull must not eat the panorama's rear half",
  );
});

test("stills, Record and offline exports inherit the projection (structurally)", async () => {
  const src = await readSrc("./preview.js");
  // The tinyplanet suite pins the settleFrame call-site count and the
  // writeFrame funnel; this only needs the equirect word to ride the SAME
  // one live bag (asserted above) — re-pin the funnel so a refactor that
  // splits it re-runs this reasoning.
  const settles = src.match(/^\s*settleFrame\(/gm) || [];
  assert.equal(
    settles.length,
    2,
    "settleFrame call sites moved — re-check which ones carry the live bag",
  );
});

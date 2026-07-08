// Tourbillon expressed purely as an op-list — the smoke test.
//
// This is the EXACT decomposition of formulas/glsl/Tourbillon.glsl:
//   box fold(1.0) → sphere fold(0.5, 1.0) → scale(2.0) → rotXY(14°) → rotYZ(7°)
// with AddC (the dispatch adds `c` after each iteration). Its native DEFAULTS
// line is  2.0,0.5,1.0,1.0,14.0,7.0  =  Scale,MinRadius,FixedRadius,FoldLimit,
// TwistXY,TwistYZ — the same six numbers appear below, just grouped by the
// operator that consumes them.
//
// If the WebGPU interpreter renders this op-list and the hand-written GLSL
// renders the same shape, the operator IR + DE rules are validated end-to-end.

export const TOURBILLON = {
  name: "Tourbillon",
  note: "screw-folded Mandelbox · re-derived from primitives",
  addC: true,
  iters: 12,
  deOption: 2, // IFS analytic DE: r/|w|
  ops: [
    { key: "boxFold", values: [1.0] }, // FoldLimit
    { key: "sphereFold", values: [0.5, 1.0] }, // MinRadius, FixedRadius
    { key: "scale", values: [2.0] }, // Scale
    { key: "rotateXY", values: [14.0] }, // TwistXY (degrees)
    { key: "rotateYZ", values: [7.0] }, // TwistYZ (degrees)
  ],
  // Orbit camera pulled back far enough that the whole body is in frame on load
  // (radius ≈ dist·tan(fov/2) ≈ 0.38·24 ≈ 9 units of the target plane visible).
  camera: { yawDeg: 35, pitchDeg: 22, dist: 24.0, fovDeg: 42 },
};

// ── Starter gallery ────────────────────────────────────────────────────────
// All built on the proven box-fold → sphere-fold → scale Mandelbox core (the
// sphere fold bounds the radius, so the attractor never escapes to blank sky),
// then varied by scale sign, rotation, or an extra angle/abs fold. deOption 2 =
// the analytic IFS r/|w| distance estimate; AddC re-adds the seed each iter.

const MANDELBOX = {
  name: "Mandelbox",
  note: "the classic box-fold · sphere-fold · scale ×2",
  addC: true,
  iters: 12,
  deOption: 2,
  ops: [
    { key: "boxFold", values: [1.0] },
    { key: "sphereFold", values: [0.5, 1.0] },
    { key: "scale", values: [2.0] },
  ],
  camera: { yawDeg: 35, pitchDeg: 22, dist: 24.0, fovDeg: 42 },
};

const STAR_BOX = {
  name: "Star Box",
  note: "Mandelbox core + 5-fold kaleidoscope and a YZ tilt",
  addC: true,
  iters: 11,
  deOption: 2,
  ops: [
    { key: "boxFold", values: [1.0] },
    { key: "sphereFold", values: [0.5, 1.0] },
    { key: "scale", values: [2.0] },
    { key: "kaleido", values: [5.0, 0.0] },
    { key: "rotateYZ", values: [30.0] },
  ],
  camera: { yawDeg: 40, pitchDeg: 18, dist: 24.0, fovDeg: 42 },
};

const KALEIDO_BOX = {
  name: "Kaleido Box",
  note: "Mandelbox core + 8-fold kaleidoscope twist",
  addC: true,
  iters: 11,
  deOption: 2,
  ops: [
    { key: "boxFold", values: [1.0] },
    { key: "sphereFold", values: [0.5, 1.0] },
    { key: "scale", values: [2.0] },
    { key: "kaleido", values: [8.0, 20.0] },
  ],
  camera: { yawDeg: 30, pitchDeg: 25, dist: 24.0, fovDeg: 42 },
};

const TWIST_TOWER = {
  name: "Twist Tower",
  note: "box core with stronger XY/YZ rotation than Tourbillon",
  addC: true,
  iters: 12,
  deOption: 2,
  ops: [
    { key: "boxFold", values: [1.0] },
    { key: "sphereFold", values: [0.5, 1.0] },
    { key: "scale", values: [2.0] },
    { key: "rotateXY", values: [45.0] },
    { key: "rotateYZ", values: [20.0] },
  ],
  camera: { yawDeg: 35, pitchDeg: 28, dist: 24.0, fovDeg: 42 },
};

const DRIFT_BOX = {
  name: "Drift Box",
  note: "Mandelbox core with a constant translate — breaks the symmetry",
  addC: true,
  iters: 12,
  deOption: 2,
  ops: [
    { key: "boxFold", values: [1.0] },
    { key: "sphereFold", values: [0.5, 1.0] },
    { key: "scale", values: [2.0] },
    { key: "translate", values: [0.3, 0.0, 0.0] },
  ],
  camera: { yawDeg: 35, pitchDeg: 22, dist: 24.0, fovDeg: 42 },
};

const MANDELBOX_X3 = {
  name: "Mandelbox ×3",
  note: "higher scale — finer, more filigreed shell",
  addC: true,
  iters: 10,
  deOption: 2,
  ops: [
    { key: "boxFold", values: [1.0] },
    { key: "sphereFold", values: [0.5, 1.0] },
    { key: "scale", values: [3.0] },
  ],
  camera: { yawDeg: 35, pitchDeg: 22, dist: 24.0, fovDeg: 42 },
};

// Pure IFS (no +c). DE = length(pos) / 2^iters via w ×|scale| — the standard
// Sierpinski tetrahedron distance estimate.
const SIERPINSKI = {
  name: "Sierpinski",
  note: "tetrahedral fold · scale ×2 · classic IFS",
  addC: false,
  iters: 14,
  deOption: 2,
  ops: [
    { key: "sierpinskiFold", values: [] },
    { key: "scale", values: [2.0] },
    { key: "translate", values: [-1.0, -1.0, -1.0] },
  ],
  camera: { yawDeg: 30, pitchDeg: 20, dist: 8.0, fovDeg: 42 },
};

// Octahedron IFS: the octahedral fold is a pure symmetry op (abs into the
// positive octant, then sort) — ALONE it just mirrors space onto the origin and
// renders blank. Like every KIFS fold (Sierpinski/Menger) it needs scale ×2 +
// translate to grow the gasket. The six octahedron vertices (±1,0,0)/(0,±1,0)/
// (0,0,±1) all collapse under the fold to the single representative (1,0,0), so
// the offset rides ONE axis — translate(-1,0,0), not the diagonal (-1,-1,-1)
// (which space-fills into a solid block). This is the canonical Sierpinski-
// octahedron, the working answer to "is there an octahedral-fold example?".
const OCTAHEDRON = {
  name: "Octahedron",
  note: "octahedral fold · scale ×2 · the Sierpinski octahedron",
  addC: false,
  iters: 14,
  deOption: 2,
  ops: [
    { key: "octaFold", values: [] },
    { key: "scale", values: [2.0] },
    { key: "translate", values: [-1.0, 0.0, 0.0] },
  ],
  camera: { yawDeg: 30, pitchDeg: 20, dist: 4.5, fovDeg: 42 },
};

// Corner Cube IFS: the plain abs fold (no sort) maps all 8 octants into the
// positive octant — unlike octaFold (which additionally sorts components and
// thereby funnels all 6 octahedron vertices to one representative), absFold
// alone keeps the eight-corner cube geometry intact. Scale ×2 + translate
// (-1,-1,-1) then produces 8 self-similar copies at the 8 corners of the unit
// cube: the 3-D Sierpinski cube / Cantor dust. (For octaFold the diagonal
// offset (-1,-1,-1) would overconstrain the sorted domain and space-fill into
// a solid block — here it is the correct IFS vertex.)
const CORNER_CUBE = {
  name: "Corner Cube",
  note: "abs fold · scale ×2 · corner IFS — 8-fold Sierpinski cube toward all corners",
  addC: false,
  iters: 14,
  deOption: 2,
  ops: [
    { key: "absFold", values: [] },
    { key: "scale", values: [2.0] },
    { key: "translate", values: [-1.0, -1.0, -1.0] },
  ],
  camera: { yawDeg: 35, pitchDeg: 22, dist: 4.0, fovDeg: 42 },
};

// The canonical Menger sponge IFS: abs → sort → scale ×3 → translate → z-fold.
// The z-fold (if z>1, z-=2) is what closes the sponge in the third axis.
const MENGER = {
  name: "Menger",
  note: "abs · sort · scale ×3 · z-fold — the sponge",
  addC: false,
  iters: 5,
  deOption: 2,
  ops: [
    { key: "absFold", values: [] },
    { key: "mengerFold", values: [] },
    { key: "scale", values: [3.0] },
    { key: "translate", values: [-2.0, -2.0, 0.0] },
    { key: "zFold", values: [1.0, 2.0] },
  ],
  camera: { yawDeg: 30, pitchDeg: 25, dist: 9.0, fovDeg: 42 },
};

// Rounded Menger: the same sponge built from the single Menger op, whose
// Smoothness rounds the edges (sqrt mode at +s) for an organic look. The op
// folds abs + sort + the z-wrap itself, so it's just menger · scale ×3 ·
// translate — no separate absFold/mengerFold/zFold. Fewer iters than the sharp
// MENGER since the rounding compounds and washes out fine detail at depth.
const ROUNDED_MENGER = {
  name: "Rounded Menger",
  note: "smoothed Menger fold (sqrt-rounded) · scale ×3 — the organic sponge",
  addC: false,
  iters: 8,
  deOption: 2,
  ops: [
    { key: "menger", values: [0.01] },
    { key: "scale", values: [3.0] },
    { key: "translate", values: [-2.0, -2.0, 0.0] },
  ],
  camera: { yawDeg: 30, pitchDeg: 20, dist: 8.5, fovDeg: 42 },
};

// Menger Cloud: the Menger op in POLYNOMIAL smoothing (negative Smoothness),
// driven deep (24 iters) with an off-canonical translate so the IFS overlaps
// into a soft, melted body rather than the crisp sponge.
const MENGER_CLOUD = {
  name: "Menger Cloud",
  note: "polynomial-smoothed menger (−s) · scale ×3 · 24 iters — soft melted body",
  addC: false,
  iters: 24,
  deOption: 2,
  ops: [
    { key: "menger", values: [-0.05] },
    { key: "scale", values: [3.0] },
    { key: "translate", values: [-1.0, -1.0, -0.5] },
  ],
  camera: { yawDeg: 30, pitchDeg: -20, dist: 5.0, fovDeg: 42 },
};

// Mandelbox core + a diagonal plane fold — an extra mirror plane on top of the
// bounded box, so it stays in frame but gains KIFS-style facets.
const MIRROR_BOX = {
  name: "Mirror Box",
  note: "box · sphere · plane-fold mirror · scale ×2",
  addC: true,
  iters: 12,
  deOption: 2,
  ops: [
    { key: "boxFold", values: [1.0] },
    { key: "sphereFold", values: [0.5, 1.0] },
    { key: "planeFold", values: [1.0, -1.0, 0.0] },
    { key: "scale", values: [2.0] },
  ],
  camera: { yawDeg: 35, pitchDeg: 22, dist: 24.0, fovDeg: 42 },
};

// Pseudo-Kleinian (Knighty): a Mandelbox fold core whose +c seed is replaced
// by a FIXED offset, with no AddC — so it's a Kleinian-group IFS (a self-
// inverse foam) rather than an escape fractal. box fold + sphere fold bound the
// radius (so it stays in frame); scale ×2 keeps the analytic r/|w| DE crisp;
// the constant offset after the scale is what knits the gasket together.
const PSEUDO_KLEINIAN = {
  name: "Pseudo-Kleinian",
  note: "box · sphere fold · scale ×2 · fixed offset (no +c) — Kleinian foam",
  addC: false,
  iters: 11,
  deOption: 2,
  ops: [
    { key: "boxFold", values: [1.0] },
    { key: "sphereFold", values: [0.5, 2.51] },
    { key: "scale", values: [2.0] },
    { key: "translate", values: [0.08, -0.21, -1.07] },
  ],
  camera: { yawDeg: 35, pitchDeg: 22, dist: 24.0, fovDeg: 42 },
};

// The classic escape-time Mandelbulb: spherical power z→z⁸ + c. Not an IFS —
// deOption 0 selects the escape-time DE (0.5·ln r·r/dr) in the preview, and the
// engine drives it via numDiff. AddC re-adds the world seed each iteration.
const MANDELBULB = {
  name: "Mandelbulb",
  note: "spherical power z→z⁸ + c — the classic escape-time bulb",
  addC: true,
  iters: 8,
  deOption: 0,
  ops: [{ key: "mandelbulbPower", values: [8.0] }],
  camera: { yawDeg: 35, pitchDeg: 12, dist: 5.0, fovDeg: 42 },
};

// The Juliabulb: same spherical power, but Julia mode replaces the per-point
// seed with a FIXED constant c — so instead of the Mandelbulb's "map of all
// bulbs" you get one connected Julia body. Tune cx/cy/cz live to morph it.
const JULIABULB = {
  name: "Juliabulb",
  note: "Mandelbulb power with a fixed Julia seed — one connected body",
  addC: true,
  iters: 9,
  deOption: 0,
  julia: true,
  juliaC: [0.35, 0.3, -0.2],
  ops: [{ key: "mandelbulbPower", values: [8.0] }],
  camera: { yawDeg: 35, pitchDeg: 12, dist: 5.0, fovDeg: 42 },
};

// Amazing Surf: the box fold acts on X,Y only (Surf Fold), so the bounded solid
// collapses into thin folded sheets — a fractal landscape of surfaces.
const AMAZING_SURF = {
  name: "Amazing Surf",
  note: "X/Y-only box fold (surf) + sphere fold + scale — folded sheets",
  addC: true,
  iters: 12,
  deOption: 2,
  ops: [
    { key: "surfFold", values: [1.0] },
    { key: "sphereFold", values: [0.5, 1.0] },
    { key: "scale", values: [2.0] },
  ],
  camera: { yawDeg: 40, pitchDeg: 28, dist: 24.0, fovDeg: 42 },
};

// Slab Box: anisotropic box fold (thin Z limit) flattens the Mandelbox into a
// slab — shows off the per-axis Box Fold XYZ.
const SLAB_BOX = {
  name: "Slab Box",
  note: "anisotropic box fold (thin Z) + sphere fold + scale — flattened box",
  addC: true,
  iters: 12,
  deOption: 2,
  ops: [
    { key: "boxFoldXYZ", values: [1.0, 1.0, 0.4] },
    { key: "sphereFold", values: [0.5, 1.0] },
    { key: "scale", values: [2.0] },
  ],
  camera: { yawDeg: 35, pitchDeg: 25, dist: 24.0, fovDeg: 42 },
};

// ── Phase 1 primitive demos — one per new op, each on a bounded core so it
//    stays in frame (see docs/planning/PRIMITIVE_PRIORITIES.md). ──

// Poly Angle Fold: Mandelbox core + a 7-fold rotational sector snap. Unlike the
// kaleido reflection, this ROTATES the point into one sector — rotational, not
// mirrored, symmetry.
const SECTOR_BOX = {
  name: "Sector Box",
  note: "Mandelbox core + 7-fold poly-angle sector fold — rotational symmetry",
  addC: true,
  iters: 11,
  deOption: 2,
  ops: [
    { key: "boxFold", values: [1.0] },
    { key: "sphereFold", values: [0.5, 1.0] },
    { key: "scale", values: [2.0] },
    { key: "polyAngleFold", values: [7.0, 0.0, 0.0] },
  ],
  camera: { yawDeg: 35, pitchDeg: 22, dist: 24.0, fovDeg: 42 },
};

// Hex Fold: Mandelbox core folded into a 60° wedge — 6-fold honeycomb symmetry.
const HEX_BOX = {
  name: "Hex Box",
  note: "Mandelbox core + hexagonal fold — 6-fold honeycomb symmetry",
  addC: true,
  iters: 11,
  deOption: 2,
  ops: [
    { key: "boxFold", values: [1.0] },
    { key: "sphereFold", values: [0.5, 1.0] },
    { key: "scale", values: [2.0] },
    { key: "hexFold", values: [] },
  ],
  camera: { yawDeg: 35, pitchDeg: 22, dist: 24.0, fovDeg: 42 },
};

// Cylinder Fold: the Amazing-Surf cylinder ball-fold (radius in XY, Z free)
// after an X/Y-only box fold — tubular folded surfaces.
const CYLINDER_SURF = {
  name: "Cylinder Surf",
  note: "X/Y box fold + cylinder fold (Z free) + scale — tubular surfaces",
  addC: true,
  iters: 12,
  deOption: 2,
  ops: [
    { key: "surfFold", values: [1.0] },
    { key: "cylinderFold", values: [0.5, 1.0] },
    { key: "scale", values: [2.0] },
  ],
  camera: { yawDeg: 40, pitchDeg: 28, dist: 24.0, fovDeg: 42 },
};

// Surf Mushroom: the Amazing-Surf mushroom family — the Cylinder Surf chain with
// the constant scale replaced by Scale Drift (the running-Scale feedback,
// closed-form per SCALE_VARY.md), plus a gentle in-loop rotation. The drift
// organic-izes the tubes into blob/mushroom character. iters ≤ 12 and a small
// ScaleVary keep the closed-form pow() well clear of the ±1e5 clamp.
const SURF_MUSHROOM = {
  name: "Surf Mushroom",
  note: "MB3D Amazing-Surf mushroom: surf + cylinder fold + Scale Drift (running-scale feedback) + tilt",
  addC: true,
  iters: 12,
  deOption: 2,
  ops: [
    { key: "surfFold", values: [1.0] },
    { key: "cylinderFold", values: [0.5, 1.0] },
    { key: "scaleDrift", values: [2.0, 0.05] },
    { key: "rotateXYZ", values: [15.0, 0.0, 0.0] },
  ],
  camera: { yawDeg: 40, pitchDeg: 28, dist: 24.0, fovDeg: 42 },
};

// Bulb Power (axis): the Mandelbulb power taken around the Y axis (IQ
// convention) — a re-oriented escape-time bulb. Axis 0 would match Mandelbulb.
const BULB_Y = {
  name: "Bulb (Y-axis)",
  note: "Mandelbulb power around the Y axis (IQ convention) — a re-oriented bulb",
  addC: true,
  iters: 8,
  deOption: 0,
  ops: [{ key: "bulbAxis", values: [8.0, 1.0] }],
  camera: { yawDeg: 35, pitchDeg: 12, dist: 5.0, fovDeg: 42 },
};

// Inversion (shifted): a Kleinian-style IFS — bounded box/sphere fold core plus a
// unit inversion about an off-center point (no +c), knitting a foam of spheres.
const KLEINIAN_DROP = {
  name: "Kleinian Drop",
  note: "box · sphere fold · shifted inversion · scale ×2 (no +c) — Kleinian foam",
  addC: false,
  iters: 11,
  deOption: 2,
  ops: [
    { key: "boxFold", values: [1.0] },
    { key: "sphereFold", values: [0.5, 1.0] },
    { key: "radialInvert", values: [0.0, 0.0, 0.5] },
    { key: "scale", values: [2.0] },
  ],
  camera: { yawDeg: 35, pitchDeg: 22, dist: 24.0, fovDeg: 42 },
};

// Abs XYZ: a Mandelbox core with an extra PER-AXIS abs (X and Z folded, Y left
// free) — independent abs X / abs Y / abs Z toggles. Folding only two axes adds
// an asymmetric mirror the all-axes absFold can't make; the box/sphere fold core
// keeps every axis bounded so the unfolded Y axis still renders a finite body.
const ABS_BOX = {
  name: "Abs Box",
  note: "Mandelbox + per-axis abs (X,Z only) — asymmetric mirror",
  addC: true,
  iters: 11,
  deOption: 2,
  ops: [
    { key: "boxFold", values: [1.0] },
    { key: "sphereFold", values: [0.5, 1.0] },
    { key: "absXYZ", values: [1.0, 0.0, 1.0] },
    { key: "scale", values: [2.0] },
  ],
  camera: { yawDeg: 35, pitchDeg: 22, dist: 24.0, fovDeg: 42 },
};

const VARY_BOX = {
  name: "Vary Box",
  note: "Mandelbox whose ball fold tests r²·⁸ (Vary Scale Fold) — denser, busier carving",
  addC: true,
  iters: 12,
  deOption: 2,
  ops: [
    { key: "boxFold", values: [1.0] },
    { key: "varyScale", values: [0.5, 1.0, 1.4] },
    { key: "scale", values: [2.0] },
  ],
  camera: { yawDeg: 35, pitchDeg: 22, dist: 24.0, fovDeg: 42 },
};

const BRISTORBROT = {
  name: "BristorBrot",
  note: "the classic Bristorbrot triplex square — first numeric-DE (deOption 3) formula",
  addC: true,
  iters: 10,
  deOption: 3,
  ops: [{ key: "bristorBrot", values: [2.0, 1.0, -1.0] }],
  camera: { yawDeg: 35, pitchDeg: 18, dist: 5.0, fovDeg: 42 },
};

// NewtonTri raw Mandelbrot-mode renders as a banded slab (numDiff shells —
// MB3D's own numeric-DE class); the GOOD look is the mix, which only the
// numeric DE family allows (a bulb-class map + a w-moving fold would be
// 'mixed'/broken under the analytic families).
const NEWTON_MIX = {
  name: "Newton Mix",
  note: "NewtonTri triplex square × sphere fold — a mix only the numeric DE allows",
  addC: true,
  iters: 10,
  deOption: 3,
  ops: [
    { key: "newtonTri2", values: [] },
    { key: "sphereFold", values: [0.5, 1.0] },
  ],
  camera: { yawDeg: 35, pitchDeg: 18, dist: 5.0, fovDeg: 42 },
};

const MSLTOE_SYM = {
  name: "Msltoe Sym",
  note: "msltoe's symmetric quadratic bulb (MB3D MsltoeSym3) — a folded-pair complex square",
  addC: true,
  iters: 12,
  deOption: 0,
  ops: [{ key: "msltoeSym3", values: [1.0, 0.0] }],
  camera: { yawDeg: 35, pitchDeg: 18, dist: 5.0, fovDeg: 42 },
};

// ── #86 fold-chain classics — recipes from the corpus buckets, math
//    re-derived from the published formulas (names credited in notes). The two
//    "Tilted" presets are the AFFINEFOLD_SPIKE.md rotated-fold recipe: enter a
//    rotated frame, fold ONE axis, back out in reverse order with negated
//    angles (op-lists compose in reverse; see the spike doc). ──
const INVERTED_BOX = {
  name: "Amazing Box",
  note: "the classic negative-scale Amazing Box (Tglad; MB3D ABoxMod1/AmazingBox2 family) — the scale −2 hollow-temple attractor",
  addC: true,
  iters: 12,
  deOption: 2,
  ops: [
    { key: "boxFold", values: [1.0] },
    { key: "sphereFold", values: [0.5, 1.0] },
    { key: "scale", values: [-2.0] },
  ],
  camera: { yawDeg: 35, pitchDeg: 22, dist: 24.0, fovDeg: 42 },
};

const AMAZING_SURF_2 = {
  name: "Amazing Surf 2",
  note: "MB3D 'Amazing Surf 2' — a tighter surf fold with the ball fold in its pure-inversion regime",
  addC: true,
  iters: 12,
  deOption: 2,
  ops: [
    { key: "surfFold", values: [1.0] },
    { key: "sphereFold", values: [0.05, 1.0] },
    { key: "scale", values: [2.0] },
  ],
  camera: { yawDeg: 35, pitchDeg: 22, dist: 24.0, fovDeg: 42 },
};

const ICOSAHEDRON = {
  name: "Icosahedron",
  note: "icosahedral KIFS (MB3D IcosahedronIFS) — golden-ratio offset grows the 5-fold gasket",
  addC: false,
  iters: 14,
  deOption: 2,
  ops: [
    { key: "icosaFold", values: [] },
    { key: "scale", values: [2.0] },
    // -0.81 not the golden 0.809: the share codec quantizes translate to its
    // 0.01 step, and presets must round-trip exactly (share.test enforces).
    { key: "translate", values: [-0.81, -0.5, 0.0] },
  ],
  camera: { yawDeg: 30, pitchDeg: 20, dist: 8.0, fovDeg: 42 },
};

const CANTOR_DUST = {
  name: "Cantor Dust",
  note: "3D Cantor set (MB3D CantorIFS) — abs · scale ×3 · corner offset, disconnected by design",
  addC: false,
  iters: 10,
  deOption: 2,
  ops: [
    { key: "absFold", values: [] },
    { key: "scale", values: [3.0] },
    { key: "translate", values: [-2.0, -2.0, -2.0] },
  ],
  camera: { yawDeg: 30, pitchDeg: 20, dist: 8.0, fovDeg: 42 },
};

const TETRA_VS = {
  name: "Tetra VS",
  note: "Sierpinski gasket warped by a vary-scale ball fold (after MB3D ATetraVS)",
  addC: false,
  iters: 14,
  deOption: 2,
  ops: [
    { key: "sierpinskiFold", values: [] },
    { key: "varyScale", values: [0.5, 1.0, 1.2] },
    { key: "scale", values: [2.0] },
    { key: "translate", values: [-1.0, -1.0, -1.0] },
  ],
  camera: { yawDeg: 30, pitchDeg: 20, dist: 8.0, fovDeg: 42 },
};

const TWISTED_SIERPINSKI = {
  name: "Twisted Sierpinski",
  note: "Sierpinski gasket with an in-loop rotation (MB3D Sierpinski3's Rotation2) — the spiral tetrahedron",
  addC: false,
  iters: 14,
  deOption: 2,
  ops: [
    { key: "sierpinskiFold", values: [] },
    { key: "scale", values: [2.0] },
    { key: "translate", values: [-1.0, -1.0, -1.0] },
    { key: "rotateXYZ", values: [8.0, 0.0, 12.0] },
  ],
  camera: { yawDeg: 30, pitchDeg: 20, dist: 8.0, fovDeg: 42 },
};

const TILTED_BOX = {
  name: "Tilted Box",
  note: "Mandelbox whose box fold runs in a tilted frame (MB3D _RotatedFolding family, via the #85 recipe) — Add c stays unrotated, so it isn't just a turned Mandelbox",
  addC: true,
  iters: 12,
  deOption: 2,
  ops: [
    { key: "rotateXY", values: [30.0] },
    { key: "rotateYZ", values: [20.0] },
    { key: "boxFold", values: [1.0] },
    { key: "rotateYZ", values: [-20.0] },
    { key: "rotateXY", values: [-30.0] },
    { key: "sphereFold", values: [0.5, 1.0] },
    { key: "scale", values: [2.0] },
  ],
  camera: { yawDeg: 35, pitchDeg: 22, dist: 40.0, fovDeg: 42 },
};

const TWIN_BULB = {
  name: "Twin Bulb",
  note: "generalized bulb with split angle powers (θ×½, φ×2 — the two-stage family, #83); 1×/1× equals Mandelbulb",
  addC: true,
  iters: 8,
  deOption: 0,
  ops: [{ key: "sphericalTwoStage", values: [8.0, 0.5, 2.0] }],
  camera: { yawDeg: 35, pitchDeg: 12, dist: 5.0, fovDeg: 42 },
};

const BOX_BULB = {
  name: "Box Bulb",
  note: "the L4-norm boxy Mandelbulb (MB3D BoxBulb, from the readable C++ ref)",
  addC: true,
  iters: 8,
  deOption: 0,
  ops: [{ key: "boxBulb", values: [8.0] }],
  camera: { yawDeg: 35, pitchDeg: 12, dist: 5.0, fovDeg: 42 },
};

const SLONOBROT = {
  name: "SlonoBrot",
  note: "z-folded quadratic (MB3D SlonoBrot2, from the readable C++ ref)",
  addC: true,
  iters: 10,
  deOption: 0,
  ops: [{ key: "slonoBrot2", values: [] }],
  camera: { yawDeg: 35, pitchDeg: 18, dist: 5.0, fovDeg: 42 },
};

const HEX_TILT = {
  name: "Hex Tilt",
  note: "Hex Box pitched 30° out of the fold plane — the #81 'h-tilt' recipe, approved on #91",
  addC: true,
  iters: 11,
  deOption: 2,
  ops: [
    { key: "boxFold", values: [1.0] },
    { key: "sphereFold", values: [0.5, 1.0] },
    { key: "scale", values: [2.0] },
    { key: "hexFold", values: [] },
    { key: "rotateYZ", values: [30.0] },
  ],
  camera: { yawDeg: 35, pitchDeg: 22, dist: 36.0, fovDeg: 42 },
};

const MSLTOE_SYM2 = {
  name: "Msltoe Sym II",
  note: "the Sym2 sign-rule variant of msltoe's quadratic bulb (MB3D MsltoeSym2)",
  addC: true,
  iters: 12,
  deOption: 0,
  ops: [{ key: "msltoeSym3", values: [1.0, 1.0] }],
  camera: { yawDeg: 35, pitchDeg: 18, dist: 5.0, fovDeg: 42 },
};

// Tourbillon stays first — it's the smoke-test default.
const COOKIES = {
  name: "Cookies",
  note: "kaleido-folded Menger — a ring of cookies",
  addC: false,
  iters: 9,
  deOption: 2,
  ops: [
    { key: "absFold", values: [] },
    { key: "mengerFold", values: [] },
    { key: "scale", values: [3.35] },
    { key: "translate", values: [-1.79, -2.0, -0.04] },
    { key: "kaleido", values: [12, 38.5] },
  ],
  camera: { yawDeg: 30, pitchDeg: 25, dist: 5.5, fovDeg: 42 },
};

// ── CSG showcase scenes — multi-object formulas (objects[]). Each one shows a
// headline capability of the scene engine; see docs/design/CSG_MULTI_OBJECT.md.
// (Multi-object → surface coloring only, enforced by sanitizeScene.)
const CUBE_CLUSTER = {
  name: "Cube Cluster",
  note: "central cube ∪ flat-cube fractal satellites (CSG)",
  ops: [],
  iters: 8,
  deOption: 2,
  addC: false,
  camera: { yawDeg: 28, pitchDeg: 18, dist: 8.0, fovDeg: 42 },
  objects: [
    {
      objType: 1,
      primParam: 0.8,
      color: [0.9, 0.52, 0.2],
      transform: { origin: [0, 0, 0], uscale: 1.0, rot: [14, 20, 0] },
      combine: 0,
    },
    {
      objType: 0,
      ops: [
        { key: "absFold", values: [] },
        { key: "scale", values: [3] },
        { key: "translate", values: [-2, -2, -2] },
      ],
      iters: 5,
      addC: false,
      deOption: 2,
      boxBase: true,
      primParam: 1.0,
      color: [0.3, 0.55, 0.85],
      transform: { origin: [0, 0, 0], uscale: 2.0, rot: [0, 0, 0] },
      combine: 0,
    },
  ],
};
const CARVED_CUBE = {
  name: "Carved Cube",
  note: "a cube with spheres bitten out (CSG subtract)",
  ops: [],
  iters: 8,
  deOption: 2,
  addC: false,
  camera: { yawDeg: 32, pitchDeg: 20, dist: 4.2, fovDeg: 42 },
  objects: [
    {
      objType: 1,
      primParam: 0.9,
      color: [0.6, 0.48, 0.85],
      transform: { origin: [0, 0, 0], uscale: 1.0, rot: [12, 18, 0] },
      combine: 0,
    },
    {
      objType: 2,
      primParam: 0.75,
      color: [0.92, 0.58, 0.26],
      transform: { origin: [0.85, 0.85, 0.0], uscale: 1.0, rot: [0, 0, 0] },
      combine: 2,
    },
    {
      objType: 2,
      primParam: 0.5,
      color: [0.92, 0.58, 0.26],
      transform: { origin: [-0.7, -0.6, 0.7], uscale: 1.0, rot: [0, 0, 0] },
      combine: 2,
    },
  ],
};
const RING_STONES = {
  name: "Ring & Stones",
  note: "torus, sphere and box composed (CSG primitives)",
  ops: [],
  iters: 8,
  deOption: 2,
  addC: false,
  camera: { yawDeg: 26, pitchDeg: 24, dist: 4.5, fovDeg: 42 },
  objects: [
    {
      objType: 3,
      primParam: 1.1,
      primParam2: 0.34,
      color: [0.34, 0.78, 0.72],
      transform: { origin: [0, 0, 0], uscale: 1.0, rot: [68, 0, 0] },
      combine: 0,
    },
    {
      objType: 2,
      primParam: 0.55,
      color: [0.9, 0.5, 0.2],
      transform: { origin: [0, 0, 0], uscale: 1.0, rot: [0, 0, 0] },
      combine: 0,
    },
    {
      objType: 1,
      primParam: 0.4,
      color: [0.82, 0.38, 0.54],
      transform: { origin: [1.5, 0.2, 0.3], uscale: 1.0, rot: [20, 30, 0] },
      combine: 0,
    },
  ],
};

// Hybrid iteration showcases (IDEAS ①, docs/design/HYBRID_ITERATION.md Phase 4)
// — a formula that alternates TWO op-lists across outer iterations (slot A =
// the top-level ops, slot B = `hybrid.b.ops`), Mandelbulb3D's signature
// "hybrid" mode. v1 is same-family only (§3.3) — all three below are either
// IFS×IFS or escape×escape, the two DE-safe combinations.
const HYBRID_MENGER_BOX = {
  name: "Menger x Mandelbox",
  note: "alternates a Menger fold with a Mandelbox fold each iteration (hybrid, IFS×IFS)",
  ops: [
    { key: "absFold", values: [] },
    { key: "mengerFold", values: [] },
    { key: "scale", values: [3.0] },
    { key: "translate", values: [-2.0, -2.0, 0.0] },
  ],
  addC: false,
  iters: 12,
  deOption: 2,
  camera: { yawDeg: 14, pitchDeg: 14, dist: 9.2, fovDeg: 42 },
  hybrid: {
    b: {
      ops: [
        { key: "boxFold", values: [1.0] },
        { key: "sphereFold", values: [0.5, 1.0] },
        { key: "scale", values: [2.0] },
      ],
      addC: true,
    },
    schedule: { a: 1, b: 1 },
  },
};
const HYBRID_BULB = {
  name: "Bulb Hybrid",
  note: "power-8 and power-3 Mandelbulb alternating each iteration (hybrid, escape×escape)",
  ops: [{ key: "mandelbulbPower", values: [8.0] }],
  addC: true,
  iters: 10,
  deOption: 0,
  camera: { yawDeg: 35, pitchDeg: 15, dist: 5.0, fovDeg: 42 },
  hybrid: {
    b: { ops: [{ key: "mandelbulbPower", values: [3.0] }], addC: true },
    schedule: { a: 1, b: 1 },
  },
};
const HYBRID_SIERPINSKI_OCTA = {
  name: "Sierpinski x Octahedral",
  note: "Sierpinski tetrahedral fold alternating with an octahedral fold (hybrid, IFS×IFS)",
  ops: [
    { key: "sierpinskiFold", values: [] },
    { key: "scale", values: [2.0] },
    { key: "translate", values: [-1.0, -1.0, -1.0] },
  ],
  addC: false,
  iters: 14,
  deOption: 2,
  camera: { yawDeg: 18, pitchDeg: 14, dist: 5.5, fovDeg: 42 },
  hybrid: {
    b: {
      ops: [
        { key: "octaFold", values: [] },
        { key: "scale", values: [2.0] },
        { key: "translate", values: [-1.0, 0.0, 0.0] },
      ],
      addC: false,
    },
    schedule: { a: 1, b: 1 },
  },
};

export const PRESETS = [
  TOURBILLON,
  MANDELBOX,
  COOKIES,
  MANDELBOX_X3,
  KALEIDO_BOX,
  STAR_BOX,
  TWIST_TOWER,
  DRIFT_BOX,
  MIRROR_BOX,
  PSEUDO_KLEINIAN,
  AMAZING_SURF,
  SLAB_BOX,
  MANDELBULB,
  JULIABULB,
  SIERPINSKI,
  OCTAHEDRON,
  CORNER_CUBE,
  MENGER,
  ROUNDED_MENGER,
  MENGER_CLOUD,
  SECTOR_BOX,
  HEX_BOX,
  CYLINDER_SURF,
  SURF_MUSHROOM,
  BULB_Y,
  KLEINIAN_DROP,
  ABS_BOX,
  VARY_BOX,
  BRISTORBROT,
  NEWTON_MIX,
  MSLTOE_SYM,
  TWIN_BULB,
  BOX_BULB,
  SLONOBROT,
  HEX_TILT,
  MSLTOE_SYM2,
  INVERTED_BOX,
  AMAZING_SURF_2,
  ICOSAHEDRON,
  CANTOR_DUST,
  TETRA_VS,
  TWISTED_SIERPINSKI,
  TILTED_BOX,
  CUBE_CLUSTER,
  CARVED_CUBE,
  RING_STONES,
  HYBRID_MENGER_BOX,
  HYBRID_BULB,
  HYBRID_SIERPINSKI_OCTA,
];

// Empty slate for the "New" button — no ops, sane defaults. Renders nothing
// until the first operator is added (see AUTHORING.md for the build-up). IFS DE
// by default; adding a Mandelbulb Power op auto-switches to escape-time.
export const BLANK = {
  name: "Untitled",
  note: "blank slate — add an operator to begin (see AUTHORING.md)",
  addC: false,
  iters: 8,
  deOption: 2,
  ops: [],
  camera: { yawDeg: 35, pitchDeg: 22, dist: 14.0, fovDeg: 42 },
};

// Deep copy so the UI can mutate freely and "Reset" restores the original.
export function clone(formula) {
  return JSON.parse(JSON.stringify(formula));
}

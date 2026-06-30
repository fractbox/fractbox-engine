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
  name: 'Tourbillon',
  note: 'screw-folded Mandelbox · re-derived from primitives',
  addC: true,
  iters: 12,
  deOption: 2,            // IFS analytic DE: r/|w|
  ops: [
    { key: 'boxFold',    values: [1.0] },        // FoldLimit
    { key: 'sphereFold', values: [0.5, 1.0] },   // MinRadius, FixedRadius
    { key: 'scale',      values: [2.0] },        // Scale
    { key: 'rotateXY',   values: [14.0] },       // TwistXY (degrees)
    { key: 'rotateYZ',   values: [7.0] },        // TwistYZ (degrees)
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
  name: 'Mandelbox',
  note: 'the classic box-fold · sphere-fold · scale ×2',
  addC: true, iters: 12, deOption: 2,
  ops: [
    { key: 'boxFold',    values: [1.0] },
    { key: 'sphereFold', values: [0.5, 1.0] },
    { key: 'scale',      values: [2.0] },
  ],
  camera: { yawDeg: 35, pitchDeg: 22, dist: 24.0, fovDeg: 42 },
};

const STAR_BOX = {
  name: 'Star Box',
  note: 'Mandelbox core + 5-fold kaleidoscope and a YZ tilt',
  addC: true, iters: 11, deOption: 2,
  ops: [
    { key: 'boxFold',    values: [1.0] },
    { key: 'sphereFold', values: [0.5, 1.0] },
    { key: 'scale',      values: [2.0] },
    { key: 'kaleido',    values: [5.0, 0.0] },
    { key: 'rotateYZ',   values: [30.0] },
  ],
  camera: { yawDeg: 40, pitchDeg: 18, dist: 24.0, fovDeg: 42 },
};

const KALEIDO_BOX = {
  name: 'Kaleido Box',
  note: 'Mandelbox core + 8-fold kaleidoscope twist',
  addC: true, iters: 11, deOption: 2,
  ops: [
    { key: 'boxFold',    values: [1.0] },
    { key: 'sphereFold', values: [0.5, 1.0] },
    { key: 'scale',      values: [2.0] },
    { key: 'kaleido',    values: [8.0, 20.0] },
  ],
  camera: { yawDeg: 30, pitchDeg: 25, dist: 24.0, fovDeg: 42 },
};

const TWIST_TOWER = {
  name: 'Twist Tower',
  note: 'box core with stronger XY/YZ rotation than Tourbillon',
  addC: true, iters: 12, deOption: 2,
  ops: [
    { key: 'boxFold',    values: [1.0] },
    { key: 'sphereFold', values: [0.5, 1.0] },
    { key: 'scale',      values: [2.0] },
    { key: 'rotateXY',   values: [45.0] },
    { key: 'rotateYZ',   values: [20.0] },
  ],
  camera: { yawDeg: 35, pitchDeg: 28, dist: 24.0, fovDeg: 42 },
};

const DRIFT_BOX = {
  name: 'Drift Box',
  note: 'Mandelbox core with a constant translate — breaks the symmetry',
  addC: true, iters: 12, deOption: 2,
  ops: [
    { key: 'boxFold',    values: [1.0] },
    { key: 'sphereFold', values: [0.5, 1.0] },
    { key: 'scale',      values: [2.0] },
    { key: 'translate',  values: [0.3, 0.0, 0.0] },
  ],
  camera: { yawDeg: 35, pitchDeg: 22, dist: 24.0, fovDeg: 42 },
};

const MANDELBOX_X3 = {
  name: 'Mandelbox ×3',
  note: 'higher scale — finer, more filigreed shell',
  addC: true, iters: 10, deOption: 2,
  ops: [
    { key: 'boxFold',    values: [1.0] },
    { key: 'sphereFold', values: [0.5, 1.0] },
    { key: 'scale',      values: [3.0] },
  ],
  camera: { yawDeg: 35, pitchDeg: 22, dist: 24.0, fovDeg: 42 },
};

// Pure IFS (no +c). DE = length(pos) / 2^iters via w ×|scale| — the standard
// Sierpinski tetrahedron distance estimate.
const SIERPINSKI = {
  name: 'Sierpinski',
  note: 'tetrahedral fold · scale ×2 · classic IFS',
  addC: false, iters: 14, deOption: 2,
  ops: [
    { key: 'sierpinskiFold', values: [] },
    { key: 'scale',          values: [2.0] },
    { key: 'translate',      values: [-1.0, -1.0, -1.0] },
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
  name: 'Octahedron',
  note: 'octahedral fold · scale ×2 · the Sierpinski octahedron',
  addC: false, iters: 14, deOption: 2,
  ops: [
    { key: 'octaFold',  values: [] },
    { key: 'scale',     values: [2.0] },
    { key: 'translate', values: [-1.0, 0.0, 0.0] },
  ],
  camera: { yawDeg: 30, pitchDeg: 20, dist: 4.5, fovDeg: 42 },
};

// The canonical Menger sponge IFS: abs → sort → scale ×3 → translate → z-fold.
// The z-fold (if z>1, z-=2) is what closes the sponge in the third axis.
const MENGER = {
  name: 'Menger',
  note: 'abs · sort · scale ×3 · z-fold — the sponge',
  addC: false, iters: 5, deOption: 2,
  ops: [
    { key: 'absFold',    values: [] },
    { key: 'mengerFold', values: [] },
    { key: 'scale',      values: [3.0] },
    { key: 'translate',  values: [-2.0, -2.0, 0.0] },
    { key: 'zFold',      values: [1.0, 2.0] },
  ],
  camera: { yawDeg: 30, pitchDeg: 25, dist: 9.0, fovDeg: 42 },
};

// Rounded Menger: the same sponge built from the single Menger op, whose
// Smoothness rounds the edges (sqrt mode at +s) for an organic look. The op
// folds abs + sort + the z-wrap itself, so it's just menger · scale ×3 ·
// translate — no separate absFold/mengerFold/zFold. Fewer iters than the sharp
// MENGER since the rounding compounds and washes out fine detail at depth.
const ROUNDED_MENGER = {
  name: 'Rounded Menger',
  note: 'smoothed Menger fold (sqrt-rounded) · scale ×3 — the organic sponge',
  addC: false, iters: 8, deOption: 2,
  ops: [
    { key: 'menger',    values: [0.01] },
    { key: 'scale',     values: [3.0] },
    { key: 'translate', values: [-2.0, -2.0, 0.0] },
  ],
  camera: { yawDeg: 30, pitchDeg: 20, dist: 8.5, fovDeg: 42 },
};

// Menger Cloud: the Menger op in POLYNOMIAL smoothing (negative Smoothness),
// driven deep (24 iters) with an off-canonical translate so the IFS overlaps
// into a soft, melted body rather than the crisp sponge.
const MENGER_CLOUD = {
  name: 'Menger Cloud',
  note: 'polynomial-smoothed menger (−s) · scale ×3 · 24 iters — soft melted body',
  addC: false, iters: 24, deOption: 2,
  ops: [
    { key: 'menger',    values: [-0.05] },
    { key: 'scale',     values: [3.0] },
    { key: 'translate', values: [-1.0, -1.0, -0.5] },
  ],
  camera: { yawDeg: 30, pitchDeg: -20, dist: 5.0, fovDeg: 42 },
};

// Mandelbox core + a diagonal plane fold — an extra mirror plane on top of the
// bounded box, so it stays in frame but gains KIFS-style facets.
const MIRROR_BOX = {
  name: 'Mirror Box',
  note: 'box · sphere · plane-fold mirror · scale ×2',
  addC: true, iters: 12, deOption: 2,
  ops: [
    { key: 'boxFold',    values: [1.0] },
    { key: 'sphereFold', values: [0.5, 1.0] },
    { key: 'planeFold',  values: [1.0, -1.0, 0.0] },
    { key: 'scale',      values: [2.0] },
  ],
  camera: { yawDeg: 35, pitchDeg: 22, dist: 24.0, fovDeg: 42 },
};

// Pseudo-Kleinian (Knighty): a Mandelbox fold core whose +c seed is replaced
// by a FIXED offset, with no AddC — so it's a Kleinian-group IFS (a self-
// inverse foam) rather than an escape fractal. box fold + sphere fold bound the
// radius (so it stays in frame); scale ×2 keeps the analytic r/|w| DE crisp;
// the constant offset after the scale is what knits the gasket together.
const PSEUDO_KLEINIAN = {
  name: 'Pseudo-Kleinian',
  note: 'box · sphere fold · scale ×2 · fixed offset (no +c) — Kleinian foam',
  addC: false, iters: 11, deOption: 2,
  ops: [
    { key: 'boxFold',    values: [1.0] },
    { key: 'sphereFold', values: [0.5, 2.51] },
    { key: 'scale',      values: [2.0] },
    { key: 'translate',  values: [0.08, -0.21, -1.07] },
  ],
  camera: { yawDeg: 35, pitchDeg: 22, dist: 24.0, fovDeg: 42 },
};

// The classic escape-time Mandelbulb: spherical power z→z⁸ + c. Not an IFS —
// deOption 0 selects the escape-time DE (0.5·ln r·r/dr) in the preview, and the
// engine drives it via numDiff. AddC re-adds the world seed each iteration.
const MANDELBULB = {
  name: 'Mandelbulb',
  note: 'spherical power z→z⁸ + c — the classic escape-time bulb',
  addC: true, iters: 8, deOption: 0,
  ops: [
    { key: 'mandelbulbPower', values: [8.0] },
  ],
  camera: { yawDeg: 35, pitchDeg: 12, dist: 5.0, fovDeg: 42 },
};

// The Juliabulb: same spherical power, but Julia mode replaces the per-point
// seed with a FIXED constant c — so instead of the Mandelbulb's "map of all
// bulbs" you get one connected Julia body. Tune cx/cy/cz live to morph it.
const JULIABULB = {
  name: 'Juliabulb',
  note: 'Mandelbulb power with a fixed Julia seed — one connected body',
  addC: true, iters: 9, deOption: 0,
  julia: true, juliaC: [0.35, 0.30, -0.20],
  ops: [
    { key: 'mandelbulbPower', values: [8.0] },
  ],
  camera: { yawDeg: 35, pitchDeg: 12, dist: 5.0, fovDeg: 42 },
};

// Amazing Surf: the box fold acts on X,Y only (Surf Fold), so the bounded solid
// collapses into thin folded sheets — a fractal landscape of surfaces.
const AMAZING_SURF = {
  name: 'Amazing Surf',
  note: 'X/Y-only box fold (surf) + sphere fold + scale — folded sheets',
  addC: true, iters: 12, deOption: 2,
  ops: [
    { key: 'surfFold',   values: [1.0] },
    { key: 'sphereFold', values: [0.5, 1.0] },
    { key: 'scale',      values: [2.0] },
  ],
  camera: { yawDeg: 40, pitchDeg: 28, dist: 24.0, fovDeg: 42 },
};

// Slab Box: anisotropic box fold (thin Z limit) flattens the Mandelbox into a
// slab — shows off the per-axis Box Fold XYZ.
const SLAB_BOX = {
  name: 'Slab Box',
  note: 'anisotropic box fold (thin Z) + sphere fold + scale — flattened box',
  addC: true, iters: 12, deOption: 2,
  ops: [
    { key: 'boxFoldXYZ', values: [1.0, 1.0, 0.4] },
    { key: 'sphereFold', values: [0.5, 1.0] },
    { key: 'scale',      values: [2.0] },
  ],
  camera: { yawDeg: 35, pitchDeg: 25, dist: 24.0, fovDeg: 42 },
};

// ── Phase 1 primitive demos — one per new op, each on a bounded core so it
//    stays in frame (see docs/planning/PRIMITIVE_PRIORITIES.md). ──

// Poly Angle Fold: Mandelbox core + a 7-fold rotational sector snap. Unlike the
// kaleido reflection, this ROTATES the point into one sector — rotational, not
// mirrored, symmetry.
const SECTOR_BOX = {
  name: 'Sector Box',
  note: 'Mandelbox core + 7-fold poly-angle sector fold — rotational symmetry',
  addC: true, iters: 11, deOption: 2,
  ops: [
    { key: 'boxFold',       values: [1.0] },
    { key: 'sphereFold',    values: [0.5, 1.0] },
    { key: 'scale',         values: [2.0] },
    { key: 'polyAngleFold', values: [7.0, 0.0, 0.0] },
  ],
  camera: { yawDeg: 35, pitchDeg: 22, dist: 24.0, fovDeg: 42 },
};

// Hex Fold: Mandelbox core folded into a 60° wedge — 6-fold honeycomb symmetry.
const HEX_BOX = {
  name: 'Hex Box',
  note: 'Mandelbox core + hexagonal fold — 6-fold honeycomb symmetry',
  addC: true, iters: 11, deOption: 2,
  ops: [
    { key: 'boxFold',    values: [1.0] },
    { key: 'sphereFold', values: [0.5, 1.0] },
    { key: 'scale',      values: [2.0] },
    { key: 'hexFold',    values: [] },
  ],
  camera: { yawDeg: 35, pitchDeg: 22, dist: 24.0, fovDeg: 42 },
};

// Cylinder Fold: the Amazing-Surf cylinder ball-fold (radius in XY, Z free)
// after an X/Y-only box fold — tubular folded surfaces.
const CYLINDER_SURF = {
  name: 'Cylinder Surf',
  note: 'X/Y box fold + cylinder fold (Z free) + scale — tubular surfaces',
  addC: true, iters: 12, deOption: 2,
  ops: [
    { key: 'surfFold',     values: [1.0] },
    { key: 'cylinderFold', values: [0.5, 1.0] },
    { key: 'scale',        values: [2.0] },
  ],
  camera: { yawDeg: 40, pitchDeg: 28, dist: 24.0, fovDeg: 42 },
};

// Bulb Power (axis): the Mandelbulb power taken around the Y axis (IQ
// convention) — a re-oriented escape-time bulb. Axis 0 would match Mandelbulb.
const BULB_Y = {
  name: 'Bulb (Y-axis)',
  note: 'Mandelbulb power around the Y axis (IQ convention) — a re-oriented bulb',
  addC: true, iters: 8, deOption: 0,
  ops: [
    { key: 'bulbAxis', values: [8.0, 1.0] },
  ],
  camera: { yawDeg: 35, pitchDeg: 12, dist: 5.0, fovDeg: 42 },
};

// Inversion (shifted): a Kleinian-style IFS — bounded box/sphere fold core plus a
// unit inversion about an off-center point (no +c), knitting a foam of spheres.
const KLEINIAN_DROP = {
  name: 'Kleinian Drop',
  note: 'box · sphere fold · shifted inversion · scale ×2 (no +c) — Kleinian foam',
  addC: false, iters: 11, deOption: 2,
  ops: [
    { key: 'boxFold',      values: [1.0] },
    { key: 'sphereFold',   values: [0.5, 1.0] },
    { key: 'radialInvert', values: [0.0, 0.0, 0.5] },
    { key: 'scale',        values: [2.0] },
  ],
  camera: { yawDeg: 35, pitchDeg: 22, dist: 24.0, fovDeg: 42 },
};

// Abs XYZ: a Mandelbox core with an extra PER-AXIS abs (X and Z folded, Y left
// free) — independent abs X / abs Y / abs Z toggles. Folding only two axes adds
// an asymmetric mirror the all-axes absFold can't make; the box/sphere fold core
// keeps every axis bounded so the unfolded Y axis still renders a finite body.
const ABS_BOX = {
  name: 'Abs Box',
  note: 'Mandelbox + per-axis abs (X,Z only) — asymmetric mirror',
  addC: true, iters: 11, deOption: 2,
  ops: [
    { key: 'boxFold',    values: [1.0] },
    { key: 'sphereFold', values: [0.5, 1.0] },
    { key: 'absXYZ',     values: [1.0, 0.0, 1.0] },
    { key: 'scale',      values: [2.0] },
  ],
  camera: { yawDeg: 35, pitchDeg: 22, dist: 24.0, fovDeg: 42 },
};

// Tourbillon stays first — it's the smoke-test default.
const COOKIES = {
  name: 'Cookies',
  note: 'kaleido-folded Menger — a ring of cookies',
  addC: false, iters: 9, deOption: 2,
  ops: [
    { key: 'absFold',    values: [] },
    { key: 'mengerFold', values: [] },
    { key: 'scale',      values: [3.35] },
    { key: 'translate',  values: [-1.79, -2.0, -0.04] },
    { key: 'kaleido',    values: [12, 38.5] },
  ],
  camera: { yawDeg: 30, pitchDeg: 25, dist: 5.5, fovDeg: 42 },
};

export const PRESETS = [
  TOURBILLON, MANDELBOX, COOKIES, MANDELBOX_X3, KALEIDO_BOX, STAR_BOX,
  TWIST_TOWER, DRIFT_BOX, MIRROR_BOX, PSEUDO_KLEINIAN, AMAZING_SURF,
  SLAB_BOX, MANDELBULB, JULIABULB, SIERPINSKI, OCTAHEDRON, MENGER, ROUNDED_MENGER,
  MENGER_CLOUD, SECTOR_BOX, HEX_BOX, CYLINDER_SURF, BULB_Y, KLEINIAN_DROP, ABS_BOX,
];

// Empty slate for the "New" button — no ops, sane defaults. Renders nothing
// until the first operator is added (see AUTHORING.md for the build-up). IFS DE
// by default; adding a Mandelbulb Power op auto-switches to escape-time.
export const BLANK = {
  name: 'Untitled',
  note: 'blank slate — add an operator to begin (see AUTHORING.md)',
  addC: false, iters: 8, deOption: 2,
  ops: [],
  camera: { yawDeg: 35, pitchDeg: 22, dist: 14.0, fovDeg: 42 },
};

// Deep copy so the UI can mutate freely and "Reset" restores the original.
export function clone(formula) {
  return JSON.parse(JSON.stringify(formula));
}

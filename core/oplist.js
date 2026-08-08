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
    { key: "kaleido", values: [5.0, 0.0, 1.0] },
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
    { key: "kaleido", values: [8.0, 20.0, 1.0] },
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

// Abs Menger: the requested showcase for issue #14 — astiglic asked for the
// MB3D TransformIFS ABS X/Y/Z primitive (which we ship as absXYZ + the
// offset/"fix" form absOffsetFold — see the issue thread and the Abs Box
// preset) demonstrated together with the Menger fold, "one with menger for
// example". The canonical MENGER preset is absFold · mengerFold · scale ×3 ·
// translate · zFold; swapping the plain absFold for absOffsetFold with a
// nonzero Y offset is exactly the "_AbsY, fixY" move from astiglic's animated
// clip (github.com/fractbox/fractbox/issues/14#issuecomment-4835987192) —
// abs(p+offset)-offset instead of abs(p) — now folded into the Menger
// lattice instead of the plain Mandelbox core. Because the offset breaks the
// z<=y<=x>=0 ordering the plain sponge relies on before mengerFold's sort, the
// lattice grows asymmetrically: the square cavities of the plain sponge become
// lopsided TRIANGULAR cavities (GPU-verified, app/.shots/abs_menger.png) — a
// genuinely distinct member of the Menger family, not a re-skin of
// MENGER/CANTOR_ROTATIONS/COOKIES/HYBRID_MENGER_BOX. Offset 0.15 was picked
// over larger offsets (tried up to 0.4) by GPU render: 0.4 over-filled the
// cavities into a near-solid block, while 0.15 keeps the sponge airy and
// clearly carved. Scale stays at 3 (no loose-DE march needed). Measured on the
// CPU evaluate() oracle: wobble 0.230 / coverage 0.168 — close to the
// canonical Menger's own 0.255/0.156, so the offset does not destabilize the
// fold.
const ABS_MENGER = {
  name: "Abs Menger",
  note: "abs fold (offset) · menger fold · scale ×3 · z-fold — the ABS X/Y/Z showcase from #14, a sponge with asymmetric triangular cavities",
  addC: false,
  iters: 10,
  deOption: 2,
  ops: [
    { key: "absOffsetFold", values: [0.0, 0.15, 0.0] },
    { key: "mengerFold", values: [] },
    { key: "scale", values: [3.0] },
    { key: "translate", values: [-2.0, -2.0, 0.0] },
    { key: "zFold", values: [1.0, 2.0] },
  ],
  camera: { yawDeg: 38, pitchDeg: 22, dist: 9.5, fovDeg: 42 },
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

// Surf Mushroom: a BOUNDED member of the Amazing-Surf / Mandelbox fold family
// (issue #116). The earlier recipe used `surfFold` (box fold on X,Y only, Z
// free), whose attractor is an INFINITE horizontal sheet — at the default
// camera it read as a flat splat, and up close, orbiting the camera flew it
// through the endless slab, which looked exactly like a "clipping plane cutting
// through the shape" (the #116 report). No fold in the palette makes a literal
// cap-and-stem mushroom (folds give self-similar sheets or symmetric temples,
// not a single asymmetric macro-feature), so the honest fix is to ship the
// closest COMPACT organic shape instead of a mislabelled sheet: the negative-
// scale Amazing Box (boxFold on ALL three axes → bounded solid) with the
// constant scale replaced by Scale Drift (running-scale feedback, closed-form
// per SCALE_VARY.md). The drift organic-izes the hollow-temple lattice; the
// object is finite, so orbiting shows a whole 3D form with no sheet artifact.
// iters = 12 with ScaleVary = 0.05 keeps the closed-form pow() clear of the
// ±1e5 clamp. Framed at dist 18 so the entire object sits in view.
const SURF_MUSHROOM = {
  name: "Surf Mushroom",
  note: "Bounded Amazing-Surf/Box shape: box fold + sphere fold + Scale Drift (running-scale feedback) — a compact organic solid (#116)",
  addC: true,
  iters: 12,
  deOption: 2,
  ops: [
    { key: "boxFold", values: [1.0] },
    { key: "sphereFold", values: [0.5, 1.0] },
    { key: "scaleDrift", values: [-2.0, 0.05] },
  ],
  camera: { yawDeg: 35, pitchDeg: 22, dist: 18.0, fovDeg: 42 },
};

// Surf Coral: found by @astiglic (#116) hunting for the MB3D "Amazing Surf"
// mushroom/coral look — TWO surf folds with a rotate between them, under a
// fixed Julia seed instead of the per-point orbit. Julia mode is the piece the
// earlier searches in this issue never tried: every constant-parameter fold
// chain (surfFold/boxFold/sphereFold/cylinderFold, with or without an outer
// translate+rotate) is scale-invariant by construction, so it either collapses
// to an infinite sheet (surfFold, Z free) or a symmetric bounded temple
// (boxFold) — never a single dominant, non-self-similar macro blob. Swapping
// in a FIXED juliaC breaks that invariance: the filled-Julia-set boundary of
// this map is a bounded, lumpy, coral/cauliflower-head solid (verified by
// pulling the camera back to dist 45-60 — the whole attractor sits in frame,
// no infinite-sheet clipping). iters=16 matches visual fidelity of the
// original 64-iter recipe at this framing (extra iterations added no visible
// detail here) at a fraction of the cost.
const SURF_CORAL = {
  name: "Surf Coral",
  note: "Two surf folds + a rotate, under a fixed Julia seed — a bounded coral/mushroom-cluster head (#116, recipe from @astiglic)",
  addC: true,
  julia: true,
  juliaC: [0.06, -0.01, 1.11],
  iters: 16,
  deOption: 2,
  ops: [
    { key: "surfFold", values: [1.36] },
    { key: "scale", values: [1.91] },
    { key: "rotateXYZ", values: [21, -8, 23.5] },
    { key: "surfFold", values: [5] },
  ],
  camera: { yawDeg: 40, pitchDeg: 25, dist: 45.0, fovDeg: 42 },
};

// Bulb Power (axis): the Mandelbulb power taken around the Y axis (IQ
// convention) — a re-oriented escape-time bulb. Axis 0 would match Mandelbulb.
const BULB_Y = {
  name: "Bulb (Y-axis)",
  note: "Mandelbulb power around the Y axis (IQ convention) — a re-oriented bulb",
  addC: true,
  iters: 8,
  deOption: 0,
  ops: [{ key: "bulbAxis", values: [8.0, 1.0, 0.0] }],
  camera: { yawDeg: 35, pitchDeg: 12, dist: 5.0, fovDeg: 42 },
};

// bulbAxis Convention 1 (sin-polar): the NormBulb trig flavor — sin/cos swapped
// on the multiplied polar angle, x-polar like the MB3D exemplar (TRIGBULB_SPIKE.md).
const NORM_BULB = {
  name: "Norm Bulb",
  note: "sin-polar bulb around the X axis — after MB3D IQ_NormBulb's convention",
  addC: true,
  iters: 8,
  deOption: 0,
  ops: [{ key: "bulbAxis", values: [8.0, 2.0, 1.0] }],
  camera: { yawDeg: 35, pitchDeg: 12, dist: 5.0, fovDeg: 42 },
};

// bulbAxis Convention 2 (asin-latitude): the classic "sine bulb" — the latitude
// convention MB3D's StandartBulb family uses (TRIGBULB_SPIKE.md).
const SINE_BULB = {
  name: "Sine Bulb",
  note: "asin-latitude bulb — the classic sine-bulb convention (MB3D StandartBulb family)",
  addC: true,
  iters: 8,
  deOption: 0,
  ops: [{ key: "bulbAxis", values: [8.0, 0.0, 2.0] }],
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

// Riemann Bulb: stereographic Riemann-sphere power — after MB3D's Riemann
// (hand-crafted, oracle-validated source; TRIGBULB era B2 wave 4). Numeric DE
// like BristorBrot.
const RIEMANN = {
  name: "Riemann",
  note: "stereographic Riemann-sphere power ×8 — after MB3D Riemann; numeric DE",
  addC: true,
  iters: 10,
  deOption: 3,
  ops: [{ key: "riemannBulb", values: [8.0] }],
  camera: { yawDeg: 35, pitchDeg: 18, dist: 5.0, fovDeg: 42 },
};

// Klein Poly Map (Tglad family, B2 wave 4): Riemann-sphere warp + Klein
// tetrahedral / dihedral rational map. Numeric DE like BristorBrot.
const KLEIN_TETRA = {
  name: "Klein Tetra",
  note: "Riemann-sphere warp + Klein tetrahedral rational map — after MB3D TgladTetra; numeric DE",
  addC: true,
  iters: 10,
  deOption: 3,
  ops: [{ key: "kleinPolyMap", values: [0.0, 0.0, 0.0] }],
  camera: { yawDeg: 35, pitchDeg: 18, dist: 5.0, fovDeg: 42 },
};

const KLEIN_DIHEDRAL = {
  name: "Klein Dihedral",
  note: "Riemann-sphere warp + Klein dihedral rational map — after MB3D TgladDihed; numeric DE",
  addC: true,
  iters: 10,
  deOption: 3,
  ops: [{ key: "kleinPolyMap", values: [0.0, 0.0, 2.0] }],
  camera: { yawDeg: 35, pitchDeg: 18, dist: 5.0, fovDeg: 42 },
};

// Magnet XYZ (MagVsXYZ family, B2 wave 4): per-axis magnet rational map.
// Numeric DE like BristorBrot.
const MAGNET = {
  name: "Magnet",
  note: "per-axis magnet rational map — after MB3D MagVsXYZ; numeric DE",
  addC: true,
  iters: 10,
  deOption: 3,
  ops: [{ key: "magnetXYZ", values: [2.0, 1.57, 0.79] }],
  camera: { yawDeg: 35, pitchDeg: 18, dist: 5.0, fovDeg: 42 },
};

const MAGNET_ABS = {
  name: "Magnet Abs",
  note: "abs-folded magnet map — after MB3D MagVsXYZabs; numeric DE",
  addC: true,
  iters: 10,
  deOption: 3,
  ops: [{ key: "magnetXYZAbs", values: [2.0, 1.57, 1.57] }],
  camera: { yawDeg: 35, pitchDeg: 18, dist: 5.0, fovDeg: 42 },
};

// Makin triplex family (B2 wave 4): David Makin's bilinear squares.
// Numeric DE like BristorBrot.
const MAKIN = {
  name: "Makin 3D",
  note: "Makin's twisted triplex square — after MB3D Makin3D-1; numeric DE",
  addC: true,
  iters: 10,
  deOption: 3,
  ops: [{ key: "makinTri", values: [0.0] }],
  camera: { yawDeg: 35, pitchDeg: 18, dist: 5.0, fovDeg: 42 },
};

const MAKIN_FUZZY = {
  name: "Makin Fuzzy",
  note: "Makin square with rational damping — after MB3D Makin3D-3-4; numeric DE",
  addC: true,
  iters: 10,
  deOption: 3,
  ops: [{ key: "makinFuzzy", values: [0.0, 0.0, 0.01] }],
  camera: { yawDeg: 35, pitchDeg: 18, dist: 5.0, fovDeg: 42 },
};

// Polygon Fold (Phase C, first approximate-DE warp): Mandelbox core with the
// cross-section pushed toward a hexagon — after MB3D's _BPolygonFromCircle
// family. The approx flag tightens the march step (APPROX_DE.md).
const POLYGON_BOX = {
  name: "Polygon Box",
  note: "Mandelbox core + hexagon radial remap — after MB3D _BPolygon* (approximate DE)",
  addC: true,
  iters: 12,
  deOption: 2,
  ops: [
    { key: "boxFold", values: [1.0] },
    { key: "sphereFold", values: [0.5, 1.0] },
    { key: "scale", values: [2.0] },
    { key: "polygonFold", values: [6.0, -1.0, 0.0] },
  ],
  camera: { yawDeg: 35, pitchDeg: 22, dist: 24.0, fovDeg: 42 },
};

// ── Parity wave 1 ──────────────────────────────────────────────────────────
// The KIFS side of the 2015 Mandalay thread — no +c, so the fold alone builds
// the structure DarkBeam was chasing ("it kind of extrudes Mandelbox adding
// nice towers at the edges"). Both presets are the same three moves at two Gap
// settings and they read as two different objects: knighty's narrow Gap 0.1
// gives smooth cut facets, a wide Gap plus the independent ZFold plane gives
// the terraced tower stack. Camera distances are MEASURED, not copied — the
// body's surface sits at r ≈ 3.3, and ≈ 3× that frames it (the same ratio the
// shipped Mandelbox/Polygon Box cameras use).
//
// ⚠ The +c "Mandalay Box" recipes (mandalayFold standing in for boxFold on a
// Mandelbox core) were CULLED on the #86 rule. At scale 2 the body is
// unbounded — the fold TRANSLATES a large component down by 2·Fold rather than
// clamping it, so unlike boxFold it cannot bound on its own — and the scale-3
// version that is bounded renders as an off-centre crumbly mass, because
// abs+sort drives the attractor into a single octant and it never centres on
// the camera. The op is sound; that recipe is not.
const MANDALAY_TOWERS = {
  name: "Mandalay Towers",
  note: "Mandalay fold (wide gap + Z plane) · scale ×2 · corner translate (no +c) — terraced towers",
  addC: false,
  iters: 14,
  deOption: 2,
  ops: [
    { key: "mandalayFold", values: [0.5, 1.0, 1.0] },
    { key: "scale", values: [2.0] },
    { key: "translate", values: [-1.0, -1.0, 0.0] },
  ],
  camera: { yawDeg: 35, pitchDeg: 22, dist: 10.0, fovDeg: 42 },
};

const MANDALAY_GEM = {
  name: "Mandalay Gem",
  note: "Mandalay fold at knighty's narrow gap · scale ×2 · corner translate (no +c) — cut facets",
  addC: false,
  iters: 14,
  deOption: 2,
  ops: [
    { key: "mandalayFold", values: [0.5, 0.1, 0.0] },
    { key: "scale", values: [2.0] },
    { key: "translate", values: [-1.0, -1.0, 0.0] },
  ],
  camera: { yawDeg: 35, pitchDeg: 22, dist: 10.0, fovDeg: 42 },
};

// ⚠ NO PRESET SHIPS FOR torusInvert OR FOR THE LOG-POLAR coordMap MODE — the
// #86 rule again. Both ops are correct, pinned in all three emitters, and
// reachable from the Build picker; what is missing is a recipe that frames:
//   • torusInvert expands hard close to the torus core circle, and its
//     azimuthal lane stretches without bound on the rotation axis (the map
//     opens the axis out into a cylinder). Every core tried this wave —
//     Kleinian-style box+sphere, inversion before AND after the scale,
//     variants 0/1/3, Radius 0.3–1.0, R 0.6–2.0 — rendered as either a flat
//     wall or empty sky. It wants a bounding companion this wave didn't find.
//   • mod-folding the log-radius lane inside a to/fromCoord sandwich collapses
//     EVERY radius into one annulus of width e^cell. That is mathematically
//     exactly what a log-polar tiling does, and visually a featureless shell;
//     folding the angle lane instead goes unbounded.
// Recorded as open preset work rather than shipped half-good.

// ⚠ NO PRESET SHIPS FOR brickFold EITHER (parity wave 2) — the same #86 rule,
// and for a reason that is structural rather than a failure to search hard
// enough. Every op in this engine runs ONCE PER ITERATION. A domain repetition
// is therefore not "tile the finished body"; it is re-folded on every pass, so
// what the marcher sees is a lattice that is dense everywhere, with no sky and
// no silhouette. That is exactly why modFold (id 17) — shipped since the first
// palette, and the direct sibling of this op — also carries ZERO presets.
// Measured this wave on real WebGPU, 13 recipes: Mandelbox/Sierpinski/octa/
// Menger bodies, cells 4–8 (8 is the declared max), stagger 0–4, iters 4–12,
// cameras from dist 12 to 34, single-axis and both-axis tilings, and an
// X/Z-plane variant. Two came close and still missed:
//   • CellY 0 (rows off, a colonnade rather than a wall) is the only recipe
//     that read as architecture — receding towers against black sky — but with
//     the rows off there is no stagger, so it is a modFold preset wearing this
//     op's name. Not worth a gallery slot.
//   • iters 5, cell 8, stagger 4 over a Sierpinski body genuinely SHOWS the
//     running bond (offset triangles course to course), but washed out to a
//     low-contrast grey field at 1–2 fps.
// The op is correct, pinned in all three emitters (core/brickfold.test.mjs) and
// reachable from the Build picker; it closes the corpus gap on its own terms.
// A framing recipe is recorded as open preset work rather than shipped
// half-good — most likely it wants a bounding companion that caps the free
// axis, or an eventual "apply once, outside the loop" op modifier.

// ⚠ NO PRESET SHIPS FOR complexMap EITHER (parity wave 2), and the reason is
// worth writing down because it is the op's own defining property biting back.
// ALL THREE VARIANTS ARE CONTRACTIVE AT INFINITY — Cayley sends ∞ → 1, and
// both swirls decay. That is exactly what makes the op safe to drop into a
// stack without a bounding companion (see the id 61 note), and it is also what
// defeats BOTH distance estimators:
//   • the escape-time family never fires, because no orbit can reach the
//     bailout radius once the map keeps pulling infinity back to a finite
//     point — so an addC Mandelbox core goes SOLID (space-filling) when the
//     map is placed before the fold chain, and dissolves to a low-contrast
//     haze when placed after it;
//   • the analytic IFS estimate r/|w| loses its bound for the same reason.
// Measured, not assumed. On the CPU DE: ~90 Mandelbox combinations (3 variants
// × 3 Orders × 5 C values × before/after) and ~216 over contraction-defined
// bodies (Sierpinski, Menger, octahedral), scored by what fraction of ray
// directions hit a surface. The plain-Mandelbox control scores 100% coverage
// at radius ≈ 8.4; every complexMap variant on it fell to 15–25% (haze) or
// pinned the probe at its far plane (solid). On the octahedral control — the
// one body that is BOTH fully covering and compact — not one of 120
// configurations held even 60%. Switching to the numeric DE (deOption 3),
// which tolerates any op mix, finally produced four candidates above 50%;
// all four were rendered on real WebGPU and all four are grainy, low-contrast
// masses with the camera inside them.
// The op is correct, pinned in all three emitters (core/complexmap.test.mjs)
// and reachable from the Build picker. What it wants is a companion that
// RESTORES an escape mechanism after the contraction — most likely a bounding
// fold with a hard radius cut, which is its own piece of work. Recorded as
// open preset work rather than shipped half-good.

// Gnarl (Phase C): the iconic gnarled Mandelbox — nested-sine warp riding the
// classic core. Approximate DE.
const GNARLED_BOX = {
  name: "Gnarled Box",
  note: "Mandelbox core + 3D gnarl warp — after MB3D _gnarl3D (approximate DE)",
  addC: true,
  iters: 12,
  deOption: 2,
  ops: [
    { key: "boxFold", values: [1.0] },
    { key: "sphereFold", values: [0.5, 1.0] },
    { key: "scale", values: [2.0] },
    { key: "gnarl3D", values: [0.1, 3.0, 3.0] },
  ],
  camera: { yawDeg: 35, pitchDeg: 22, dist: 24.0, fovDeg: 42 },
};

// Sine shear (Phase C): the classic _YplusSinZ ripple riding the Mandelbox.
const WAVY_BOX = {
  name: "Wavy Box",
  note: "Mandelbox core + cross-axis sine shear — after MB3D _YplusSinZ (approximate DE)",
  addC: true,
  iters: 12,
  deOption: 2,
  ops: [
    { key: "boxFold", values: [1.0] },
    { key: "sphereFold", values: [0.5, 1.0] },
    { key: "scale", values: [2.0] },
    { key: "sinShear", values: [0.0, 1.0, 1.0] },
  ],
  camera: { yawDeg: 35, pitchDeg: 22, dist: 24.0, fovDeg: 42 },
};

// Smooth folds (Phase C): the ABoxSmoothFold recipe — rounded Mandelbox.
const SMOOTH_BOX = {
  name: "Smooth Box",
  note: "smooth box fold + smooth ball fold + scale — the MB3D ABoxSmoothFold recipe (approximate DE)",
  addC: true,
  iters: 12,
  deOption: 2,
  ops: [
    { key: "smoothBoxFold", values: [1.0, 6.0, 1.0] },
    { key: "smoothBallFold", values: [0.25, 4.0, 0.3] },
    { key: "scale", values: [2.0] },
  ],
  camera: { yawDeg: 35, pitchDeg: 22, dist: 24.0, fovDeg: 42 },
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

// Cantor Rotations: a FAITHFUL parity build of MB3D's CantorIFS_rotations,
// composed from stock MIT operators against decoded numeric params (the user's
// own MB3D file, parsed externally — no GPL code was read):
//   Scale=1.83526, CScale=(6.90514,0.05055,-0.08),
//   Rotation1=(x,y,z)=(-135.54, 27, -36)°, Rotation2=(0, -21, -7)°, iters=12.
// MB3D CantorIFS's per-iteration transform is:
//   abs(p) → R1·p → magnitude-sort fold → R2·p → ×Scale → −(Scale−1)·CScale
// which maps onto: absFold · rotateXYZ(R1) · mengerFold · rotateXYZ(R2) · scale ·
// translate(−(Scale−1)·CScale).
//
// WHY mengerFold, NOT octaFold (the #234 fix). The airy, DISCONNECTED look of
// the real formula comes from R1's rotational asymmetry surviving into the fold.
// octaFold does `pos = abs(pos)` as its FIRST step, which folds every octant
// onto the positive one and thereby CANCELS most of the preceding R1 — the
// result collapses into a dense cauliflower/foam. mengerFold is the SAME
// descending component sort WITHOUT that leading abs (it is sign-preserving), so
// the rotated signs from R1 pass through the sort untouched and the dust stays
// airy and disconnected. The single abs the real formula applies is the explicit
// absFold BEFORE R1 (exactly once, as MB3D does it); re-absing inside the fold is
// the cauliflower bug.
//
// Rotation mapping: fractbox rotateXYZ params are (AngleXY[about Z], AngleYZ
// [about X], AngleXZ[about Y]) applied Z→X→Y; MB3D angles are (x,y,z)=(about
// X,Y,Z). We map AngleXY=z, AngleYZ=x, AngleXZ=y.
//
// Scale is the DECODED 1.835 (kept faithful, NOT nudged to 2; quantized to 1.84
// so the 0.01 share codec round-trips). A scale < 2 leaves a LOOSE analytic IFS
// DE, but that is a first-class, handled case: stability.looseDE() flags it and
// the renderer compensates with a tighter step (renderpolicy deScale 0.5→0.3 +
// more march steps), exactly as the shipped Amazing-Surf / scale<2 presets
// render. The invariants gate emits a WARN (not a failure) for scale<2 under
// deOption 2, and Remix re-clamps scale to ≥2 on its own output, so no gate
// blocks the faithful value. The consistent offset is −(Scale−1)·CScale =
// (-5.77,-0.04,0.07). Angles/translate sit at 0.01 granularity so the share
// codec round-trips exactly.
const CANTOR_ROTATIONS = {
  name: "Cantor Rotations",
  note: "MB3D CantorIFS (rotations) — abs · rot · mengerFold · rot · scale ×1.84 · offset; a rotated, disconnected Cantor dust from decoded params (mengerFold keeps R1's asymmetry — airy, not cauliflower)",
  addC: false,
  iters: 12,
  deOption: 2,
  ops: [
    { key: "absFold", values: [] },
    { key: "rotateXYZ", values: [-36.0, -135.54, 27.0] },
    { key: "mengerFold", values: [] },
    { key: "rotateXYZ", values: [-7.0, 0.0, -21.0] },
    { key: "scale", values: [1.84] },
    { key: "translate", values: [-5.77, -0.04, 0.07] },
  ],
  camera: { yawDeg: 30, pitchDeg: 20, dist: 9.0, fovDeg: 42 },
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
    { key: "kaleido", values: [12, 38.5, 1.0] },
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
// Sierpinski Cube (#21): the 8-corner octant IFS (absFold · scale ×2 ·
// translate -1) rendered as a box LEAF instead of the classic radial-dust
// finalize — a plain "Corner Cube" formula (same ops, no leaf) folds into a
// SOLID faceted cube at any zoom (the 8 octants tile exactly edge-to-edge, no
// gaps), reading as one rounded blob, not eight shrinking corner copies. The
// leaf is sampled with iterShape (D3: min leaf-distance across every
// iteration, not just the final one) so BOTH levels are visible at once — 8
// corner cubes (iter 1) with 64 smaller ones nested at their own corners
// (iter 2). iterShape needs the explicit shapeId/leaf D0 form (the legacy
// boxBase alias doesn't carry it through sanitize). A separate central box
// object supplies the dominant middle cube the pure octant recursion leaves
// hollow (there's a real gap at the origin — no level ever lands there).
//
// Tuning note: the octant map fixes level-k copies at radius growing from
// 0.87 toward a limit of 1.73×primParam as k→∞ (never past that shell), so
// iters mainly controls DENSITY, not spread — cranking it past 2-3 makes the
// union of ALL levels' copies so numerous (8+64+512+…) that from outside they
// read as one bumpy solid blob again, losing the discrete-cubes look (tested
// up to iters 6). iters:2 + a half-extent well under the 0.5 corner-spacing
// (0.2, ~40% fill) is what actually reads as "shrinking cubes with real gaps,
// scattering away from center in 8 directions" — unlike Cube Cluster's
// Menger-corner (scale ×3, single final level) cluster of many same-size
// small cubes.
const SIERPINSKI_CUBE = {
  name: "Sierpinski Cube",
  note: "8-corner Sierpinski/Cantor cube IFS as flat boxes, not dust (#21)",
  ops: [],
  iters: 8,
  deOption: 2,
  addC: false,
  camera: { yawDeg: 32, pitchDeg: 24, dist: 9.0, fovDeg: 42 },
  objects: [
    {
      // The dominant middle cube — the pure octant IFS below has a real gap
      // at the origin (no level ever lands there), so it never supplies one.
      objType: 1,
      primParam: 0.32,
      color: [0.95, 0.55, 0.22],
      transform: { origin: [0, 0, 0], uscale: 1.0, rot: [18, 24, 0] },
      combine: 0,
    },
    {
      objType: 0,
      shapeId: 1,
      ops: [
        { key: "absFold", values: [] },
        { key: "scale", values: [2.0] },
        { key: "translate", values: [-1.0, -1.0, -1.0] },
      ],
      iters: 2,
      addC: false,
      deOption: 2,
      iterShape: true,
      primParam: 0.2,
      color: [0.86, 0.42, 0.16],
      transform: { origin: [0, 0, 0], uscale: 1.0, rot: [18, 24, 0] },
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

// D2 batch-1 leaf showcases (PRIMITIVE_DIFS_LEAVES.md) — new-form scene
// objects (shapeId + shapeParams). The unbounded lattices (hex/tri grids)
// ship intersected with a sphere so the camera sees an object, not a wall.
const GYROID_SHELL = {
  name: "Gyroid",
  note: "Schoen gyroid minimal surface (TPMS shell, approximate DE)",
  ops: [],
  iters: 8,
  deOption: 2,
  addC: false,
  camera: { yawDeg: 28, pitchDeg: 18, dist: 4.4, fovDeg: 42 },
  objects: [
    {
      shapeId: 7,
      shapeParams: [6.0, 0.05, 0.0, 1.4],
      ops: [],
      iters: 1,
      color: [0.36, 0.78, 0.74],
      transform: { origin: [0, 0, 0], uscale: 1.0, rot: [0, 0, 0] },
      combine: 0,
      blendK: 0,
    },
  ],
};
const SCHERK_TOWER = {
  name: "Scherk Tower",
  note: "Scherk saddle tower — sin z = sinh x sinh y (minimal surface, approximate DE)",
  ops: [],
  iters: 8,
  deOption: 2,
  addC: false,
  camera: { yawDeg: 40, pitchDeg: 12, dist: 4.6, fovDeg: 42 },
  objects: [
    {
      shapeId: 10,
      shapeParams: [4.0, 0.04, 1.4, 0],
      ops: [],
      iters: 1,
      color: [0.85, 0.74, 0.3],
      transform: { origin: [0, 0, 0], uscale: 1.0, rot: [90, 0, 0] },
      combine: 0,
      blendK: 0,
    },
  ],
};
const HEX_REACTOR = {
  name: "Hex Reactor",
  note: "honeycomb pipe lattice ∩ sphere (hex-grid leaf + intersect combine)",
  ops: [],
  iters: 8,
  deOption: 2,
  addC: false,
  camera: { yawDeg: 20, pitchDeg: 32, dist: 4.6, fovDeg: 42 },
  objects: [
    {
      shapeId: 11,
      shapeParams: [0.28, 1.5, 0.035, 0.85],
      ops: [],
      iters: 1,
      color: [0.56, 0.45, 0.82],
      transform: { origin: [0, 0, 0], uscale: 1.0, rot: [0, 0, 0] },
      combine: 0,
      blendK: 0,
    },
    {
      shapeId: 2,
      shapeParams: [1.4, 0, 0, 0],
      ops: [],
      iters: 1,
      color: [0.56, 0.45, 0.82],
      transform: { origin: [0, 0, 0], uscale: 1.0, rot: [0, 0, 0] },
      combine: 3,
      blendK: 0,
    },
  ],
};
const GEARWORKS = {
  name: "Gearworks",
  note: "gear + (2,3)-torus knot threaded through it (D2 leaves)",
  ops: [],
  iters: 8,
  deOption: 2,
  addC: false,
  camera: { yawDeg: 30, pitchDeg: 16, dist: 5.2, fovDeg: 42 },
  objects: [
    {
      shapeId: 13,
      shapeParams: [12, 1.0, 0.1, 0.22],
      ops: [],
      iters: 1,
      color: [0.8, 0.52, 0.34],
      transform: { origin: [0, 0, 0], uscale: 1.0, rot: [0, 0, 0] },
      combine: 0,
      blendK: 0,
    },
    {
      shapeId: 15,
      shapeParams: [2, 3, 1.0, 0.09],
      ops: [],
      iters: 1,
      color: [0.3, 0.55, 0.85],
      transform: { origin: [0, 0, 0], uscale: 1.0, rot: [0, 0, 0] },
      combine: 0,
      blendK: 0,
    },
  ],
};
const VOXEL_ORB = {
  name: "Voxel Orb",
  note: "sphere quantized to voxels (lores leaf, approximate DE)",
  ops: [],
  iters: 8,
  deOption: 2,
  addC: false,
  camera: { yawDeg: 32, pitchDeg: 22, dist: 3.8, fovDeg: 42 },
  objects: [
    {
      shapeId: 16,
      shapeParams: [0, 1.0, 0.22, 0.01],
      ops: [],
      iters: 1,
      color: [0.45, 0.78, 0.4],
      transform: { origin: [0, 0, 0], uscale: 1.0, rot: [0, 0, 0] },
      combine: 0,
      blendK: 0,
    },
  ],
};
const MENGER_PLATE = {
  name: "Menger Plate",
  note: "IQ menger sponge on a plate (leaf with internal detail loop)",
  ops: [],
  iters: 8,
  deOption: 2,
  addC: false,
  camera: { yawDeg: 24, pitchDeg: 30, dist: 3.9, fovDeg: 42 },
  objects: [
    {
      shapeId: 14,
      shapeParams: [1.0, 0.33, 4, 0],
      ops: [],
      iters: 1,
      color: [0.82, 0.36, 0.52],
      transform: { origin: [0, 0, 0], uscale: 1.0, rot: [0, 0, 0] },
      combine: 0,
      blendK: 0,
    },
  ],
};
const TORUS_CASCADE = {
  name: "Torus Cascade",
  note: "iterated-shape dIFS — a torus sampled at every fold scale (D3)",
  ops: [],
  iters: 8,
  deOption: 2,
  addC: false,
  camera: { yawDeg: 30, pitchDeg: 20, dist: 5.0, fovDeg: 42 },
  objects: [
    {
      shapeId: 3,
      shapeParams: [0.55, 0.12, 0, 0],
      iterShape: true,
      ops: [
        { key: "sierpinskiFold", values: [] },
        { key: "scale", values: [2.0] },
        { key: "translate", values: [-1.0, -1.0, -1.0] },
      ],
      iters: 6,
      addC: false,
      deOption: 2,
      color: [0.86, 0.46, 0.18],
      transform: { origin: [0, 0, 0], uscale: 1.0, rot: [0, 0, 0] },
      combine: 0,
      blendK: 0,
    },
  ],
};

// D2 batch-2 leaf showcases — cages, shells and frames.
const SPHERE_CAGE = {
  name: "Sphere Cage",
  note: "lat/long wireframe sphere (cage leaf)",
  ops: [],
  iters: 8,
  deOption: 2,
  addC: false,
  camera: { yawDeg: 26, pitchDeg: 20, dist: 3.8, fovDeg: 42 },
  objects: [
    {
      shapeId: 19,
      shapeParams: [1.0, 0.035, 6, 8],
      ops: [],
      iters: 1,
      color: [0.85, 0.74, 0.3],
      transform: { origin: [0, 0, 0], uscale: 1.0, rot: [0, 0, 0] },
      combine: 0,
      blendK: 0,
    },
    {
      shapeId: 2,
      shapeParams: [0.55, 0, 0, 0],
      ops: [],
      iters: 1,
      color: [0.3, 0.55, 0.85],
      transform: { origin: [0, 0, 0], uscale: 1.0, rot: [0, 0, 0] },
      combine: 0,
      blendK: 0,
    },
  ],
};
const KLEIN_BAGEL = {
  name: "Klein Bagel",
  note: "figure-8 Klein bottle immersion (published parametric, approximate DE)",
  ops: [],
  iters: 8,
  deOption: 2,
  addC: false,
  camera: { yawDeg: 30, pitchDeg: 46, dist: 4.8, fovDeg: 42 },
  objects: [
    {
      shapeId: 22,
      shapeParams: [0.9, 0.5, 1, 0.03],
      ops: [],
      iters: 1,
      color: [0.36, 0.78, 0.74],
      transform: { origin: [0, 0, 0], uscale: 1.0, rot: [0, 0, 0] },
      combine: 0,
      blendK: 0,
    },
  ],
};
const SEASHELL = {
  name: "Seashell",
  note: "logarithmic conch spiral (shell math, approximate DE)",
  ops: [],
  iters: 8,
  deOption: 2,
  addC: false,
  camera: { yawDeg: 40, pitchDeg: 26, dist: 9.0, fovDeg: 42 },
  objects: [
    {
      shapeId: 23,
      shapeParams: [0.05, 0.5, 0.06, 4],
      ops: [],
      iters: 1,
      color: [0.8, 0.52, 0.34],
      transform: { origin: [0, -1, 0], uscale: 0.9, rot: [0, 0, 0] },
      combine: 0,
      blendK: 0,
    },
  ],
};
const WAVE_POOL = {
  name: "Wave Pool",
  note: "circular sine heightfield ∩ cylinder (wave leaf)",
  ops: [],
  iters: 8,
  deOption: 2,
  addC: false,
  camera: { yawDeg: 30, pitchDeg: 30, dist: 5.2, fovDeg: 42 },
  objects: [
    {
      shapeId: 21,
      shapeParams: [9.0, 0.12, 1, 0.04],
      ops: [],
      iters: 1,
      color: [0.3, 0.55, 0.85],
      transform: { origin: [0, 0, 0], uscale: 1.0, rot: [0, 0, 0] },
      combine: 0,
      blendK: 0,
    },
    {
      shapeId: 4,
      shapeParams: [1.6, 0.5, 0, 0],
      ops: [],
      iters: 1,
      color: [0.3, 0.55, 0.85],
      transform: { origin: [0, 0, 0], uscale: 1.0, rot: [0, 0, 0] },
      combine: 3,
      blendK: 0,
    },
  ],
};
const DOUBLE_HELIX = {
  name: "Double Helix",
  note: "two-strand helix ∩ cylinder (helix leaf)",
  ops: [],
  iters: 8,
  deOption: 2,
  addC: false,
  camera: { yawDeg: 24, pitchDeg: 10, dist: 5.4, fovDeg: 42 },
  objects: [
    {
      shapeId: 17,
      shapeParams: [1.1, 2, 0.11, 0.7],
      ops: [],
      iters: 1,
      color: [0.36, 0.78, 0.74],
      transform: { origin: [0, 0, 0], uscale: 1.0, rot: [0, 0, 0] },
      combine: 0,
      blendK: 0,
    },
    {
      shapeId: 4,
      shapeParams: [1.0, 1.5, 0, 0],
      ops: [],
      iters: 1,
      color: [0.85, 0.74, 0.3],
      transform: { origin: [0, 0, 0], uscale: 1.0, rot: [0, 0, 0] },
      combine: 3,
      blendK: 0,
    },
  ],
};
const FRAME_NEST = {
  name: "Frame Nest",
  note: "iterated-shape box frame — a frame at every fold scale (D3)",
  ops: [],
  iters: 8,
  deOption: 2,
  addC: false,
  camera: { yawDeg: 32, pitchDeg: 22, dist: 5.4, fovDeg: 42 },
  objects: [
    {
      shapeId: 27,
      shapeParams: [0.9, 0.04, 0, 0],
      iterShape: true,
      ops: [
        { key: "absFold", values: [] },
        { key: "mengerFold", values: [] },
        { key: "scale", values: [3.0] },
        { key: "translate", values: [-2.0, -2.0, 0.0] },
      ],
      iters: 4,
      addC: false,
      deOption: 2,
      color: [0.45, 0.78, 0.4],
      transform: { origin: [0, 0, 0], uscale: 1.0, rot: [0, 0, 0] },
      combine: 0,
      blendK: 0,
    },
  ],
};

// D2 Taubin-wave showcases — signed algebraic surfaces.
const HEART_PRESET = {
  name: "Heart",
  note: "the Taubin heart sextic (algebraic surface, approximate DE)",
  ops: [],
  iters: 8,
  deOption: 2,
  addC: false,
  camera: { yawDeg: 8, pitchDeg: 2, dist: 4.6, fovDeg: 42 },
  objects: [
    {
      shapeId: 28,
      shapeParams: [1.0, 1.6, 0, 0],
      ops: [],
      iters: 1,
      color: [0.82, 0.26, 0.34],
      transform: { origin: [0, 0, 0], uscale: 1.0, rot: [-90, 0, 0] },
      combine: 0,
      blendK: 0,
    },
  ],
};
const DECOCUBE_PRESET = {
  name: "Decocube",
  note: "3D-XplorMath deco-cube — tubes around six face circles (approximate DE)",
  ops: [],
  iters: 8,
  deOption: 2,
  addC: false,
  camera: { yawDeg: 28, pitchDeg: 22, dist: 6.2, fovDeg: 42 },
  objects: [
    {
      shapeId: 35,
      shapeParams: [0.8, 0.02, 1.0, 2.2],
      ops: [],
      iters: 1,
      color: [0.56, 0.45, 0.82],
      transform: { origin: [0, 0, 0], uscale: 1.0, rot: [0, 0, 0] },
      combine: 0,
      blendK: 0,
    },
  ],
};
const GUMDROP_PRESET = {
  name: "Gumdrop",
  note: "Bourke gumdrop torus quartic (algebraic surface, approximate DE)",
  ops: [],
  iters: 8,
  deOption: 2,
  addC: false,
  camera: { yawDeg: 30, pitchDeg: 20, dist: 4.6, fovDeg: 42 },
  objects: [
    {
      shapeId: 37,
      shapeParams: [0.6, 1.8, 0, 0],
      ops: [],
      iters: 1,
      color: [0.85, 0.6, 0.3],
      transform: { origin: [0, 0, 0], uscale: 1.0, rot: [0, 0, 90] },
      combine: 0,
      blendK: 0,
    },
  ],
};

// Hybrid iteration showcases (IDEAS ①, docs/design/HYBRID_ITERATION.md Phase 4)
// — a formula that alternates TWO op-lists across outer iterations (slot A =
// the top-level ops, slot B = `hybrid.b.ops`), Mandelbulb3D's signature
// "hybrid" mode. v1 is same-family only (§3.3) — all three below are either
// IFS×IFS or escape×escape, the two DE-safe combinations.
// D2 batch-3 leaf showcases — heightfield terrains.
const GNARL_DUNES = {
  name: "Gnarl Dunes",
  note: "Pickover gnarl height field, cut to a dune disc (approximate DE)",
  ops: [],
  iters: 8,
  deOption: 2,
  addC: false,
  camera: { yawDeg: 30, pitchDeg: 34, dist: 6.5, fovDeg: 42 },
  objects: [
    {
      shapeId: 39,
      shapeParams: [0.3, 3, 0.35, 4],
      // ×2.5 before the leaf shrinks the dune wavelength (2π in field space)
      // to fit the disc; w tracks the scale so the DE stays sound.
      ops: [{ key: "scale", values: [2.5] }],
      iters: 1,
      color: [0.8, 0.68, 0.4],
      transform: { origin: [0, 0, 0], uscale: 1.0, rot: [-90, 0, 0] },
      combine: 0,
      blendK: 0,
    },
    {
      // The Frame-Nest rule: an unbounded field needs an intersect bound to
      // read as an object (and to frame/zoom sanely).
      shapeId: 2,
      shapeParams: [2.1, 0, 0, 0],
      ops: [],
      iters: 1,
      color: [0.8, 0.68, 0.4],
      transform: { origin: [0, 0, 0], uscale: 1.0, rot: [0, 0, 0] },
      combine: 3,
      blendK: 0,
    },
  ],
};
const RIEMANN_SWIRL = {
  name: "Riemann Swirl",
  note: "Riemann sqrt-branch sheet with a radial phase spiral",
  ops: [],
  iters: 8,
  deOption: 2,
  addC: false,
  camera: { yawDeg: 20, pitchDeg: 26, dist: 4.4, fovDeg: 42 },
  objects: [
    {
      shapeId: 43,
      shapeParams: [2, 0.7, 1.5, 1.6],
      ops: [],
      iters: 1,
      color: [0.34, 0.7, 0.72],
      transform: { origin: [0, 0, 0], uscale: 1.0, rot: [0, 0, 0] },
      combine: 0,
      blendK: 0,
    },
  ],
};

// D2 batch-4 leaf showcases — the geometric tail.
const CUT_GEM = {
  name: "Cut Gem",
  note: "brilliant cut — facet fold + crown/pavilion/table planes",
  ops: [],
  iters: 8,
  deOption: 2,
  addC: false,
  camera: { yawDeg: 24, pitchDeg: 16, dist: 4.6, fovDeg: 42 },
  objects: [
    {
      shapeId: 51,
      shapeParams: [8, 1, 0.35, 0.9],
      ops: [],
      iters: 1,
      color: [0.55, 0.8, 0.95],
      transform: { origin: [0, 0, 0], uscale: 1.0, rot: [0, 0, 0] },
      combine: 0,
      blendK: 0,
    },
  ],
};
const LOXODROME_PRESET = {
  name: "Loxodrome",
  note: "two rhumb-line strands winding pole to pole (approximate DE)",
  ops: [],
  iters: 8,
  deOption: 2,
  addC: false,
  camera: { yawDeg: 30, pitchDeg: 24, dist: 4.4, fovDeg: 42 },
  objects: [
    {
      shapeId: 52,
      shapeParams: [4, 1, 2, 0.09],
      ops: [],
      iters: 1,
      color: [0.85, 0.55, 0.3],
      transform: { origin: [0, 0, 0], uscale: 1.0, rot: [0, 0, 0] },
      combine: 0,
      blendK: 0,
    },
  ],
};
const SPIRAL_WALLS = {
  name: "Spiral Walls",
  note: "log-spiral arm walls cut to a disc (approximate DE)",
  ops: [],
  iters: 8,
  deOption: 2,
  addC: false,
  camera: { yawDeg: 30, pitchDeg: 42, dist: 6.2, fovDeg: 42 },
  objects: [
    {
      shapeId: 53,
      shapeParams: [0.25, 2, 0.5, 0.06],
      ops: [],
      iters: 1,
      color: [0.75, 0.6, 0.45],
      transform: { origin: [0, 0, 0], uscale: 1.0, rot: [0, 0, 0] },
      combine: 0,
      blendK: 0,
    },
    {
      // Frame-Nest rule: the radially unbounded arms need an intersect bound.
      shapeId: 2,
      shapeParams: [2.0, 0, 0, 0],
      ops: [],
      iters: 1,
      color: [0.75, 0.6, 0.45],
      transform: { origin: [0, 0, 0], uscale: 1.0, rot: [0, 0, 0] },
      combine: 3,
      blendK: 0,
    },
  ],
};

// D2 batch-5 leaf showcases.
const KLEIN_BOTTLE = {
  name: "Klein Bottle",
  note: "the classic sextic immersion as a thin shell (approximate DE)",
  ops: [],
  iters: 8,
  deOption: 2,
  addC: false,
  camera: { yawDeg: 30, pitchDeg: 24, dist: 5.0, fovDeg: 42 },
  objects: [
    {
      shapeId: 57,
      shapeParams: [0.4, 0.03, 1.4, 0],
      ops: [],
      iters: 1,
      color: [0.5, 0.75, 0.6],
      transform: { origin: [0, 0, 0], uscale: 1.0, rot: [0, 0, 0] },
      combine: 0,
      blendK: 0,
    },
  ],
};
const TOY_BOX = {
  name: "Toy Box",
  note: "hash-grid cells dispatching random primitives, cut to a ball",
  ops: [],
  iters: 8,
  deOption: 2,
  addC: false,
  camera: { yawDeg: 28, pitchDeg: 30, dist: 6.0, fovDeg: 42 },
  objects: [
    {
      shapeId: 55,
      shapeParams: [0.55, 0.72, 7, 2],
      ops: [],
      iters: 1,
      color: [0.82, 0.62, 0.4],
      transform: { origin: [0, 0, 0], uscale: 1.0, rot: [0, 0, 0] },
      combine: 0,
      blendK: 0,
    },
    {
      // Frame-Nest rule: the infinite shape field needs an intersect bound.
      shapeId: 2,
      shapeParams: [1.9, 0, 0, 0],
      ops: [],
      iters: 1,
      color: [0.82, 0.62, 0.4],
      transform: { origin: [0, 0, 0], uscale: 1.0, rot: [0, 0, 0] },
      combine: 3,
      blendK: 0,
    },
  ],
};

// Kleinian limit set (leaf 58, docs/planning/KLEINIAN_LIMIT.md) — the camera
// sits INSIDE the corridor between the two tangent-ball solids; that's where
// the limit-set garlands live (from outside the shape reads as a slab).
const KLEINIAN_PEARLS = {
  name: "Kleinian Pearls",
  note: "Maskit limit set from inside the corridor (approximate DE)",
  ops: [],
  iters: 8,
  deOption: 2,
  addC: false,
  // dist 1.0 keeps the orbit inside the corridor for pitches up to ~77°
  // (walls sit at y = ±u/2 ≈ ±0.98) — at 1.4 the safe range was only ±44°
  // and dragging felt like bumping into walls (user feedback).
  camera: { yawDeg: 80, pitchDeg: 10, dist: 1.0, fovDeg: 70 },
  objects: [
    {
      shapeId: 58,
      shapeParams: [1.95, 0.07, 0.12, 48],
      ops: [],
      iters: 1,
      color: [0.75, 0.68, 0.55],
      transform: { origin: [0, 0, 0], uscale: 1.0, rot: [0, 0, 0] },
      combine: 0,
      blendK: 0,
    },
  ],
};

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

// ── N-slot hybrid showcases (≥3 slots — the new `hybrid.slots[]` stored shape,
// HYBRID_NSLOT_SPEC.md). Slot A is the formula body; the extra slots ride in
// hybrid.slots[]; schedule.counts is the FULL per-slot period incl. A at index
// 0. Both keep ONE DE family across every slot (hybridDeFamily === the family,
// NOT 'mixed') so the health chip stays green and the analytic DE holds. ──

// Triune Bulb: three escape-time bulb personalities on an uneven 2:1:3 period —
// the power-8 Mandelbulb (spiky), the power-2 square (round, few lobes), then a
// Y-axis power-5 bulb (bulbAxis, re-oriented). Alternating the powers stacks the
// per-power shells into a tiered, floret-crusted body no single-power bulb makes
// — the 3-slot successor to Bulb Hybrid (which is a 1:1 power-8×power-3 pair).
// deOption 0 (escape-time) across all three slots → escape family, not mixed.
const HYBRID_BULB_TRIO = {
  name: "Triune Bulb",
  note: "power-8, power-2 and Y-axis power-5 bulbs on a 2:1:3 schedule (hybrid, three escape-time phases)",
  ops: [{ key: "mandelbulbPower", values: [8.0] }],
  addC: true,
  iters: 10,
  deOption: 0,
  camera: { yawDeg: 35, pitchDeg: 14, dist: 5.0, fovDeg: 42 },
  hybrid: {
    slots: [
      { ops: [{ key: "mandelbulbPower", values: [2.0] }], addC: true },
      { ops: [{ key: "bulbAxis", values: [5.0, 1.0, 0.0] }], addC: true },
    ],
    schedule: { counts: [2, 1, 3] },
  },
};

// Fourfold Citadel: four distinct IFS fold personalities woven on a 1:2:1:2
// period — a Menger fold (square-cavity sponge), the Mandelbox fold (box+sphere,
// organic +c), the Sierpinski tetrahedral fold, then a 6-fold kaleido+rotate
// symmetry snap. The alternation carves a monumental banded lattice: Menger
// cavities and box-fold detail read together as one structure, the kaleido phase
// injects the radial symmetry, the Sierpinski phase re-consolidates it. All four
// slots are analytic IFS (deOption 2) → ifs family, not mixed.
const HYBRID_QUAD_CITADEL = {
  name: "Fourfold Citadel",
  note: "Menger · Mandelbox · Sierpinski · kaleido folds on a 1:2:1:2 schedule (hybrid, four IFS phases)",
  ops: [
    { key: "absFold", values: [] },
    { key: "mengerFold", values: [] },
    { key: "scale", values: [3.0] },
    { key: "translate", values: [-2.0, -2.0, 0.0] },
  ],
  addC: false,
  iters: 11,
  deOption: 2,
  camera: { yawDeg: 20, pitchDeg: 16, dist: 9.5, fovDeg: 42 },
  hybrid: {
    slots: [
      {
        ops: [
          { key: "boxFold", values: [1.0] },
          { key: "sphereFold", values: [0.5, 1.0] },
          { key: "scale", values: [2.0] },
        ],
        addC: true,
      },
      {
        ops: [
          { key: "sierpinskiFold", values: [] },
          { key: "scale", values: [2.0] },
          { key: "translate", values: [-1.0, -1.0, -1.0] },
        ],
        addC: false,
      },
      {
        ops: [
          { key: "kaleido", values: [6.0, 18.0, 1.0] },
          { key: "rotateXY", values: [24.0] },
        ],
        addC: false,
      },
    ],
    schedule: { counts: [1, 2, 1, 2] },
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
  ABS_MENGER,
  SECTOR_BOX,
  HEX_BOX,
  CYLINDER_SURF,
  SURF_MUSHROOM,
  SURF_CORAL,
  BULB_Y,
  NORM_BULB,
  SINE_BULB,
  KLEINIAN_DROP,
  ABS_BOX,
  VARY_BOX,
  BRISTORBROT,
  RIEMANN,
  KLEIN_TETRA,
  KLEIN_DIHEDRAL,
  MAGNET,
  MAGNET_ABS,
  MAKIN,
  MAKIN_FUZZY,
  POLYGON_BOX,
  MANDALAY_TOWERS,
  MANDALAY_GEM,
  GNARLED_BOX,
  WAVY_BOX,
  SMOOTH_BOX,
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
  CANTOR_ROTATIONS,
  TETRA_VS,
  TWISTED_SIERPINSKI,
  TILTED_BOX,
  CUBE_CLUSTER,
  SIERPINSKI_CUBE,
  CARVED_CUBE,
  RING_STONES,
  GYROID_SHELL,
  SCHERK_TOWER,
  HEX_REACTOR,
  GEARWORKS,
  VOXEL_ORB,
  MENGER_PLATE,
  TORUS_CASCADE,
  SPHERE_CAGE,
  KLEIN_BAGEL,
  SEASHELL,
  WAVE_POOL,
  DOUBLE_HELIX,
  FRAME_NEST,
  HEART_PRESET,
  DECOCUBE_PRESET,
  GUMDROP_PRESET,
  GNARL_DUNES,
  RIEMANN_SWIRL,
  CUT_GEM,
  LOXODROME_PRESET,
  SPIRAL_WALLS,
  KLEIN_BOTTLE,
  TOY_BOX,
  KLEINIAN_PEARLS,
  HYBRID_MENGER_BOX,
  HYBRID_BULB,
  HYBRID_SIERPINSKI_OCTA,
  HYBRID_BULB_TRIO,
  HYBRID_QUAD_CITADEL,
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

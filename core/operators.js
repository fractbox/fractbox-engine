// ─────────────────────────────────────────────────────────────────────────
// Operator IR — the single source of truth for the primitive palette.
// ─────────────────────────────────────────────────────────────────────────
// A "formula" is an ordered list of operators applied to a point each
// iteration. Each operator entry declares everything the rest of the system
// needs, so adding a new primitive means adding ONE entry here:
//
//   id      stable numeric opcode (the value packed into the GPU op buffer
//           and matched by the WGSL interpreter's `switch`). Never renumber.
//   key     stable string id used by op-lists / JSON interchange.
//   name    human label for the UI.
//   wRule   how this op affects the running derivative `w` (the DE bookkeeping).
//           Folds/rotations are isometries (w ×1) → DE-free. Only sphere-fold
//           and scale move w. This is *the* reason primitives compose into a
//           correct distance estimate without any global re-derivation.
//   params  [{name, default, min, max, step, type}]  (type: 'double' | 'angle')
//           Drives the UI sliders AND the GPU op-buffer packing (positional).
//   wgsl    the interpreter switch-case body. Mutates `pos` / `w`; reads its
//           params from `op.p0`, `op.p1`, `op.p2`.
//   glsl(v) emits the native iterateJIT_ body fragment. `v` is the array of
//           local variable names the exporter assigned to this op's params.
//
// The WGSL and GLSL emitters are kept side-by-side on purpose: they are the
// same math in two shading languages, and a divergence between them is exactly
// the kind of bug a reviewer can catch by eye.
// ─────────────────────────────────────────────────────────────────────────

export const W_UNCHANGED = 'unchanged'; // isometry — |Jacobian| = 1
export const W_MUL_K     = 'mul_k';     // sphere fold — multiplies w by fold factor k
export const W_MUL_SCALE = 'mul_scale'; // conformal scale — multiplies w by |scale|
export const W_BULB      = 'bulb';      // escape-time power — tracks the analytic
                                        // derivative dr; flips the DE family (not IFS)

export const OPERATORS = [
  {
    id: 0, key: 'boxFold', name: 'Box Fold', wRule: W_UNCHANGED,
    params: [{ name: 'FoldLimit', default: 1.0, min: 0.1, max: 3.0, step: 0.01, type: 'double' }],
    wgsl: `
        let fold = op.p0;
        pos.x = abs(pos.x + fold) - abs(pos.x - fold) - pos.x;
        pos.y = abs(pos.y + fold) - abs(pos.y - fold) - pos.y;
        pos.z = abs(pos.z + fold) - abs(pos.z - fold) - pos.z;`,
    glsl: (v) => `
    // box fold (reflection: |Jacobian| = 1, w untouched)
    pos.x = abs(pos.x + ${v[0]}) - abs(pos.x - ${v[0]}) - pos.x;
    pos.y = abs(pos.y + ${v[0]}) - abs(pos.y - ${v[0]}) - pos.y;
    pos.z = abs(pos.z + ${v[0]}) - abs(pos.z - ${v[0]}) - pos.z;`,
  },
  {
    id: 1, key: 'sphereFold', name: 'Sphere Fold', wRule: W_MUL_K,
    params: [
      { name: 'MinRadius',   default: 0.5, min: 0.05, max: 2.0, step: 0.01, type: 'double' },
      { name: 'FixedRadius', default: 1.0, min: 0.1,  max: 3.0, step: 0.01, type: 'double' },
    ],
    wgsl: `
        let minR2 = op.p0 * op.p0;
        let fixedR2 = op.p1 * op.p1;
        let r2 = dot(pos, pos);
        var k = 1.0;
        if (r2 < minR2) { k = fixedR2 / minR2; }
        else if (r2 < fixedR2) { k = fixedR2 / r2; }
        pos = pos * k;
        w = w * k;`,
    glsl: (v) => `
    // sphere fold (uniform scale by k → tracked onto w)
    {
        float minR2 = ${v[0]} * ${v[0]};
        float fixedR2 = ${v[1]} * ${v[1]};
        float r2 = dot(pos, pos);
        float k;
        if      (r2 < minR2)   k = fixedR2 / minR2;
        else if (r2 < fixedR2) k = fixedR2 / r2;
        else                   k = 1.0;
        pos  *= k;
        w    *= k;
        g_wq *= k;
    }`,
  },
  {
    id: 2, key: 'scale', name: 'Scale', wRule: W_MUL_SCALE,
    params: [{ name: 'Scale', default: 2.0, min: -4.0, max: 4.0, step: 0.01, type: 'double' }],
    wgsl: `
        let s = op.p0;
        pos = pos * s;
        w = w * abs(s);`,
    glsl: (v) => `
    // conformal scale (the expanding map → |scale| onto w)
    pos  *= ${v[0]};
    w    *= abs(${v[0]});
    g_wq *= abs(${v[0]});`,
  },
  {
    id: 3, key: 'rotateXY', name: 'Rotate XY', wRule: W_UNCHANGED,
    params: [{ name: 'AngleXY', default: 0.0, min: -180.0, max: 180.0, step: 0.5, type: 'angle' }],
    wgsl: `
        let a = radians(op.p0);
        let ca = cos(a); let sa = sin(a);
        let nx = pos.x * ca - pos.y * sa;
        let ny = pos.x * sa + pos.y * ca;
        pos.x = nx; pos.y = ny;`,
    glsl: (v) => `
    // rotate in XY (orthogonal: w untouched)
    {
        float ca = cos(${v[0]}), sa = sin(${v[0]});
        float nx = pos.x * ca - pos.y * sa;
        float ny = pos.x * sa + pos.y * ca;
        pos.x = nx; pos.y = ny;
    }`,
  },
  {
    id: 4, key: 'rotateYZ', name: 'Rotate YZ', wRule: W_UNCHANGED,
    params: [{ name: 'AngleYZ', default: 0.0, min: -180.0, max: 180.0, step: 0.5, type: 'angle' }],
    wgsl: `
        let a = radians(op.p0);
        let ca = cos(a); let sa = sin(a);
        let ny = pos.y * ca - pos.z * sa;
        let nz = pos.y * sa + pos.z * ca;
        pos.y = ny; pos.z = nz;`,
    glsl: (v) => `
    // rotate in YZ (orthogonal: w untouched)
    {
        float ca = cos(${v[0]}), sa = sin(${v[0]});
        float ny = pos.y * ca - pos.z * sa;
        float nz = pos.y * sa + pos.z * ca;
        pos.y = ny; pos.z = nz;
    }`,
  },
  // ── Extra fold-family primitives (not in Tourbillon, here to show the
  //    palette grows by data alone — adding these needed no engine change) ──
  {
    id: 5, key: 'absFold', name: 'Abs Fold', wRule: W_UNCHANGED,
    params: [],
    wgsl: `        pos = abs(pos);`,
    glsl: () => `
    // abs fold into the positive octant (reflection)
    pos = abs(pos);`,
  },
  {
    id: 6, key: 'kaleido', name: 'Kaleidoscope', wRule: W_UNCHANGED,
    params: [
      { name: 'Symmetry', default: 6.0, min: 2.0, max: 16.0, step: 1.0, type: 'double' },
      { name: 'Twist',    default: 0.0, min: -180.0, max: 180.0, step: 0.5, type: 'angle' },
    ],
    wgsl: `
        let wedge = 6.2831853 / max(op.p0, 2.0);
        var ang = atan2(pos.y, pos.x);
        ang = ang - wedge * floor(ang / wedge + 0.5);
        ang = abs(ang) + radians(op.p1);
        let rad = length(pos.xy);
        pos.x = cos(ang) * rad;
        pos.y = sin(ang) * rad;`,
    glsl: (v) => `
    // N-fold kaleidoscope angle fold (reflection: w untouched).
    // NOTE: angle-folds bound DIRECTION only — pair with a box/sphere fold to
    // bound radius or the attractor escapes (renders blank sky).
    {
        float wedge = 6.2831853 / max(${v[0]}, 2.0);
        float ang = atan(pos.y, pos.x);
        ang = ang - wedge * floor(ang / wedge + 0.5);
        ang = abs(ang) + ${v[1]};
        float rad = length(pos.xy);
        pos.x = cos(ang) * rad;
        pos.y = sin(ang) * rad;
    }`,
  },
  {
    id: 7, key: 'translate', name: 'Translate', wRule: W_UNCHANGED,
    params: [
      { name: 'TransX', default: 0.0, min: -2.0, max: 2.0, step: 0.01, type: 'double' },
      { name: 'TransY', default: 0.0, min: -2.0, max: 2.0, step: 0.01, type: 'double' },
      { name: 'TransZ', default: 0.0, min: -2.0, max: 2.0, step: 0.01, type: 'double' },
    ],
    wgsl: `
        pos = pos + vec3f(op.p0, op.p1, op.p2);`,
    glsl: (v) => `
    // constant offset (translation: |Jacobian| = 1, w untouched)
    pos += vec3(${v[0]}, ${v[1]}, ${v[2]});`,
  },
  {
    id: 8, key: 'rotateXZ', name: 'Rotate XZ', wRule: W_UNCHANGED,
    params: [{ name: 'AngleXZ', default: 0.0, min: -180.0, max: 180.0, step: 0.5, type: 'angle' }],
    wgsl: `
        let a = radians(op.p0);
        let ca = cos(a); let sa = sin(a);
        let nx = pos.x * ca - pos.z * sa;
        let nz = pos.x * sa + pos.z * ca;
        pos.x = nx; pos.z = nz;`,
    glsl: (v) => `
    // rotate in XZ (orthogonal: w untouched)
    {
        float ca = cos(${v[0]}), sa = sin(${v[0]});
        float nx = pos.x * ca - pos.z * sa;
        float nz = pos.x * sa + pos.z * ca;
        pos.x = nx; pos.z = nz;
    }`,
  },
  {
    id: 9, key: 'mengerFold', name: 'Menger Fold', wRule: W_UNCHANGED,
    params: [],
    wgsl: `
        if (pos.x < pos.y) { let t = pos.x; pos.x = pos.y; pos.y = t; }
        if (pos.x < pos.z) { let t = pos.x; pos.x = pos.z; pos.z = t; }
        if (pos.y < pos.z) { let t = pos.y; pos.y = pos.z; pos.z = t; }`,
    glsl: () => `
    // menger sort fold — order components descending (permutation, w untouched).
    // Pair with an abs fold to sort by magnitude (the Menger sponge recipe).
    if (pos.x < pos.y) { float t = pos.x; pos.x = pos.y; pos.y = t; }
    if (pos.x < pos.z) { float t = pos.x; pos.x = pos.z; pos.z = t; }
    if (pos.y < pos.z) { float t = pos.y; pos.y = pos.z; pos.z = t; }`,
  },
  {
    id: 10, key: 'sierpinskiFold', name: 'Sierpinski Fold', wRule: W_UNCHANGED,
    params: [],
    wgsl: `
        if (pos.x + pos.y < 0.0) { let t = -pos.y; pos.y = -pos.x; pos.x = t; }
        if (pos.x + pos.z < 0.0) { let t = -pos.z; pos.z = -pos.x; pos.x = t; }
        if (pos.y + pos.z < 0.0) { let t = -pos.z; pos.z = -pos.y; pos.y = t; }`,
    glsl: () => `
    // sierpinski tetrahedral fold — reflect across x+y, x+z, y+z planes
    // (reflections: w untouched). Pair with scale ×2 + translate for the IFS.
    if (pos.x + pos.y < 0.0) { float t = -pos.y; pos.y = -pos.x; pos.x = t; }
    if (pos.x + pos.z < 0.0) { float t = -pos.z; pos.z = -pos.x; pos.x = t; }
    if (pos.y + pos.z < 0.0) { float t = -pos.z; pos.z = -pos.y; pos.y = t; }`,
  },
  {
    id: 11, key: 'zFold', name: 'Z Fold', wRule: W_UNCHANGED,
    params: [
      { name: 'Threshold', default: 1.0, min: -3.0, max: 3.0, step: 0.01, type: 'double' },
      { name: 'Shift',     default: 2.0, min: -4.0, max: 4.0, step: 0.01, type: 'double' },
    ],
    wgsl: `
        if (pos.z > op.p0) { pos.z = pos.z - op.p1; }`,
    glsl: (v) => `
    // conditional z fold — the Menger sponge z-wrap (translation: w untouched)
    if (pos.z > ${v[0]}) pos.z -= ${v[1]};`,
  },
  {
    id: 12, key: 'planeFold', name: 'Plane Fold', wRule: W_UNCHANGED,
    params: [
      { name: 'NormalX', default: 1.0, min: -1.0, max: 1.0, step: 0.01, type: 'double' },
      { name: 'NormalY', default: 1.0, min: -1.0, max: 1.0, step: 0.01, type: 'double' },
      { name: 'NormalZ', default: 0.0, min: -1.0, max: 1.0, step: 0.01, type: 'double' },
    ],
    // Knighty's conditional reflection: any point on the negative side of the
    // plane through the origin (normal n) is mirrored to the positive side.
    // box/sierpinski folds are special cases — this is the general KIFS fold.
    wgsl: `
        var nv = vec3f(op.p0, op.p1, op.p2);
        if (dot(nv, nv) < 1e-12) { nv = vec3f(1.0, 0.0, 0.0); }
        let n = normalize(nv);
        let d = dot(pos, n);
        if (d < 0.0) { pos = pos - 2.0 * d * n; }`,
    glsl: (v) => `
    // plane fold — reflect across the plane through origin with normal n
    // (reflection: |Jacobian| = 1, w untouched). The general KIFS fold.
    {
        vec3 nv = vec3(${v[0]}, ${v[1]}, ${v[2]});
        if (dot(nv, nv) < 1e-12) nv = vec3(1.0, 0.0, 0.0);
        vec3 n = normalize(nv);
        float d = dot(pos, n);
        if (d < 0.0) pos -= 2.0 * d * n;
    }`,
  },
  {
    id: 14, key: 'mandelbulbPower', name: 'Mandelbulb Power', wRule: W_BULB,
    params: [{ name: 'Power', default: 8.0, min: 2.0, max: 16.0, step: 0.1, type: 'double' }],
    // Spherical power z→zⁿ (White/Nylander Mandelbulb). This is an ESCAPE-TIME
    // map, not an IFS fold: set the formula's DE option to 0 (escape) and turn
    // AddC on. w accumulates the analytic derivative dr = n·rⁿ⁻¹·dr + 1, which
    // the preview's escape-time DE (0.5·ln r·r/dr) consumes. Adding this op
    // flips the formula off the IFS r/|w| estimate (the badge shows "escape").
    wgsl: `
        let bp = op.p0;
        let br = length(pos);
        if (br > 1e-9) {
          let bth = acos(clamp(pos.z / br, -1.0, 1.0)) * bp;
          let bph = atan2(pos.y, pos.x) * bp;
          let brn = pow(br, bp);
          w = bp * brn / br * w + 1.0;
          let bst = sin(bth);
          pos = brn * vec3f(bst * cos(bph), bst * sin(bph), cos(bth));
        }`,
    glsl: (v) => `
    // Mandelbulb spherical power z→z^${v[0]} (White/Nylander) — escape-time.
    // Engine DEoption 0 (numDiff) drives the surface DE; w carries the analytic
    // derivative for any consumer that wants it. Pair with AddC (the dispatch
    // re-adds c after this returns).
    {
        float bp = ${v[0]};
        float br = length(pos);
        if (br > 1e-9) {
            float bth = acos(clamp(pos.z / br, -1.0, 1.0)) * bp;
            float bph = atan(pos.y, pos.x) * bp;
            float brn = pow(br, bp);
            w = bp * brn / br * w + 1.0;
            float bst = sin(bth);
            pos = brn * vec3(bst * cos(bph), bst * sin(bph), cos(bth));
        }
    }`,
  },
  {
    id: 13, key: 'sphereInv', name: 'Sphere Inversion', wRule: W_MUL_K,
    params: [{ name: 'Radius', default: 1.0, min: 0.1, max: 3.0, step: 0.01, type: 'double' }],
    // Unconditional sphere inversion  p -> r²·p/|p|².  Conformal: the local
    // scale factor k = r²/|p|² is isotropic, so it tracks cleanly onto w (same
    // DE bookkeeping as the sphere fold). NOTE: inversion alone is unbounded —
    // pair with a box/sphere fold or the attractor escapes (blank sky).
    wgsl: `
        let r2 = op.p0 * op.p0;
        let d = max(dot(pos, pos), 1e-6);
        let k = r2 / d;
        pos = pos * k;
        w = w * k;`,
    glsl: (v) => `
    // sphere inversion (conformal: isotropic scale k → tracked onto w)
    {
        float r2 = ${v[0]} * ${v[0]};
        float d = max(dot(pos, pos), 1e-6);
        float k = r2 / d;
        pos  *= k;
        w    *= k;
        g_wq *= k;
    }`,
  },
  {
    id: 15, key: 'surfFold', name: 'Surf Fold', wRule: W_UNCHANGED,
    params: [{ name: 'FoldLimit', default: 1.0, min: 0.1, max: 3.0, step: 0.01, type: 'double' }],
    // Amazing Surf box fold: fold X and Y only, leaving Z free. The unfolded Z
    // axis turns the Mandelbox's solid into thin sheets / surfaces.
    wgsl: `
        let sl = op.p0;
        pos.x = abs(pos.x + sl) - abs(pos.x - sl) - pos.x;
        pos.y = abs(pos.y + sl) - abs(pos.y - sl) - pos.y;`,
    glsl: (v) => `
    // Amazing Surf fold — box fold on X,Y only (Z free): builds sheets/surfaces.
    pos.x = abs(pos.x + ${v[0]}) - abs(pos.x - ${v[0]}) - pos.x;
    pos.y = abs(pos.y + ${v[0]}) - abs(pos.y - ${v[0]}) - pos.y;`,
  },
  {
    id: 16, key: 'octaFold', name: 'Octahedral Fold', wRule: W_UNCHANGED,
    params: [],
    // Fold into the octahedral fundamental domain: abs into the positive octant,
    // then sort the components descending. Eight-fold symmetric KIFS bodies.
    // Pure symmetry fold — ALONE it just mirrors space (the only bounded point
    // is the origin, so it renders blank). Pair with Scale + Translate (like
    // Menger/Sierpinski/icosa) to grow the gasket, e.g. Scale 2 + Translate(-1,-1,-1).
    wgsl: `
        pos = abs(pos);
        if (pos.x < pos.y) { let t = pos.x; pos.x = pos.y; pos.y = t; }
        if (pos.x < pos.z) { let t = pos.x; pos.x = pos.z; pos.z = t; }
        if (pos.y < pos.z) { let t = pos.y; pos.y = pos.z; pos.z = t; }`,
    glsl: () => `
    // octahedral fold — abs into the positive octant, then sort x>=y>=z
    // (reflection + permutation: |Jacobian| = 1, w untouched). Pair with
    // scale + translate (e.g. ×2, (-1,-1,-1)) to grow the gasket.
    pos = abs(pos);
    if (pos.x < pos.y) { float t = pos.x; pos.x = pos.y; pos.y = t; }
    if (pos.x < pos.z) { float t = pos.x; pos.x = pos.z; pos.z = t; }
    if (pos.y < pos.z) { float t = pos.y; pos.y = pos.z; pos.z = t; }`,
  },
  {
    id: 17, key: 'modFold', name: 'Mod Fold (Tile)', wRule: W_UNCHANGED,
    params: [
      { name: 'CellX', default: 4.0, min: 0.0, max: 8.0, step: 0.05, type: 'double' },
      { name: 'CellY', default: 4.0, min: 0.0, max: 8.0, step: 0.05, type: 'double' },
      { name: 'CellZ', default: 0.0, min: 0.0, max: 8.0, step: 0.05, type: 'double' },
    ],
    // Domain repetition: wrap each axis into a cell, tiling space into an
    // infinite lattice of copies (a cell size of 0 leaves that axis alone).
    // Per-cell translation, so the DE stays sound AS LONG AS the bounded body
    // fits inside the cell — too small a cell and neighbouring copies overlap.
    wgsl: `
        if (op.p0 > 0.0) { pos.x = pos.x - op.p0 * floor(pos.x / op.p0 + 0.5); }
        if (op.p1 > 0.0) { pos.y = pos.y - op.p1 * floor(pos.y / op.p1 + 0.5); }
        if (op.p2 > 0.0) { pos.z = pos.z - op.p2 * floor(pos.z / op.p2 + 0.5); }`,
    glsl: (v) => `
    // mod fold — domain repetition / tiling (cell of 0 = axis off). Per-cell
    // translation; keep the cell larger than the body or copies overlap.
    if (${v[0]} > 0.0) pos.x -= ${v[0]} * floor(pos.x / ${v[0]} + 0.5);
    if (${v[1]} > 0.0) pos.y -= ${v[1]} * floor(pos.y / ${v[1]} + 0.5);
    if (${v[2]} > 0.0) pos.z -= ${v[2]} * floor(pos.z / ${v[2]} + 0.5);`,
  },
  {
    id: 18, key: 'boxFoldXYZ', name: 'Box Fold XYZ', wRule: W_UNCHANGED,
    params: [
      { name: 'LimitX', default: 1.0, min: 0.1, max: 3.0, step: 0.01, type: 'double' },
      { name: 'LimitY', default: 1.0, min: 0.1, max: 3.0, step: 0.01, type: 'double' },
      { name: 'LimitZ', default: 1.0, min: 0.1, max: 3.0, step: 0.01, type: 'double' },
    ],
    // Box fold with an independent limit per axis → anisotropic (stretched /
    // slab) boxes instead of the uniform cube. Equals Box Fold when X=Y=Z.
    wgsl: `
        pos.x = abs(pos.x + op.p0) - abs(pos.x - op.p0) - pos.x;
        pos.y = abs(pos.y + op.p1) - abs(pos.y - op.p1) - pos.y;
        pos.z = abs(pos.z + op.p2) - abs(pos.z - op.p2) - pos.z;`,
    glsl: (v) => `
    // per-axis box fold (anisotropic): independent fold limit on X, Y, Z.
    pos.x = abs(pos.x + ${v[0]}) - abs(pos.x - ${v[0]}) - pos.x;
    pos.y = abs(pos.y + ${v[1]}) - abs(pos.y - ${v[1]}) - pos.y;
    pos.z = abs(pos.z + ${v[2]}) - abs(pos.z - ${v[2]}) - pos.z;`,
  },
  {
    id: 19, key: 'absOffsetFold', name: 'Abs Fold (offset)', wRule: W_UNCHANGED,
    params: [
      { name: 'OffsetX', default: 0.0, min: -2.0, max: 2.0, step: 0.01, type: 'double' },
      { name: 'OffsetY', default: 0.0, min: -2.0, max: 2.0, step: 0.01, type: 'double' },
      { name: 'OffsetZ', default: 0.0, min: -2.0, max: 2.0, step: 0.01, type: 'double' },
    ],
    // Abs fold across SHIFTED planes: reflect across x = -OffsetX (etc.) instead
    // of the origin. Off-centre mirrors break the symmetry of a plain abs fold.
    // Equals Abs Fold at offset 0. (Reflection + shift: |Jacobian| = 1.)
    wgsl: `
        pos = abs(pos + vec3f(op.p0, op.p1, op.p2)) - vec3f(op.p0, op.p1, op.p2);`,
    glsl: (v) => `
    // offset abs fold — mirror across planes shifted by the offset (reflection).
    pos = abs(pos + vec3(${v[0]}, ${v[1]}, ${v[2]})) - vec3(${v[0]}, ${v[1]}, ${v[2]});`,
  },
  {
    id: 20, key: 'tentFold', name: 'Tent Fold', wRule: W_UNCHANGED,
    params: [
      { name: 'PeriodX', default: 0.0, min: 0.0, max: 8.0, step: 0.05, type: 'double' },
      { name: 'PeriodY', default: 0.0, min: 0.0, max: 8.0, step: 0.05, type: 'double' },
      { name: 'PeriodZ', default: 0.0, min: 0.0, max: 8.0, step: 0.05, type: 'double' },
    ],
    // Periodic triangle-wave (tent) fold per axis: mirror space into repeating
    // ridges (period of 0 = axis off). Like Mod Fold but MIRRORED rather than
    // wrapped — same "keep the body inside one period" DE caveat.
    wgsl: `
        if (op.p0 > 0.0) { pos.x = abs(pos.x - op.p0 * round(pos.x / op.p0)); }
        if (op.p1 > 0.0) { pos.y = abs(pos.y - op.p1 * round(pos.y / op.p1)); }
        if (op.p2 > 0.0) { pos.z = abs(pos.z - op.p2 * round(pos.z / op.p2)); }`,
    glsl: (v) => `
    // tent fold — periodic mirrored ridges per axis (0 = axis off).
    if (${v[0]} > 0.0) pos.x = abs(pos.x - ${v[0]} * round(pos.x / ${v[0]}));
    if (${v[1]} > 0.0) pos.y = abs(pos.y - ${v[1]} * round(pos.y / ${v[1]}));
    if (${v[2]} > 0.0) pos.z = abs(pos.z - ${v[2]} * round(pos.z / ${v[2]}));`,
  },
  {
    id: 21, key: 'rotateXYZ', name: 'Rotate XYZ', wRule: W_UNCHANGED,
    params: [
      { name: 'AngleXY', default: 0.0, min: -180.0, max: 180.0, step: 0.5, type: 'angle' },
      { name: 'AngleYZ', default: 0.0, min: -180.0, max: 180.0, step: 0.5, type: 'angle' },
      { name: 'AngleXZ', default: 0.0, min: -180.0, max: 180.0, step: 0.5, type: 'angle' },
    ],
    // One Euler rotation = the three plane rotations in sequence (about Z, then
    // X, then Y). Orthogonal: |Jacobian| = 1, w untouched.
    wgsl: `
        let ra = radians(op.p0); let rb = radians(op.p1); let rd = radians(op.p2);
        { let ca = cos(ra); let sa = sin(ra); let nx = pos.x*ca - pos.y*sa; let ny = pos.x*sa + pos.y*ca; pos.x = nx; pos.y = ny; }
        { let ca = cos(rb); let sa = sin(rb); let ny = pos.y*ca - pos.z*sa; let nz = pos.y*sa + pos.z*ca; pos.y = ny; pos.z = nz; }
        { let ca = cos(rd); let sa = sin(rd); let nx = pos.x*ca - pos.z*sa; let nz = pos.x*sa + pos.z*ca; pos.x = nx; pos.z = nz; }`,
    glsl: (v) => `
    // Euler rotation (XY, then YZ, then XZ) — orthogonal: w untouched.
    { float ca = cos(${v[0]}), sa = sin(${v[0]}); float nx = pos.x*ca - pos.y*sa, ny = pos.x*sa + pos.y*ca; pos.x = nx; pos.y = ny; }
    { float ca = cos(${v[1]}), sa = sin(${v[1]}); float ny = pos.y*ca - pos.z*sa, nz = pos.y*sa + pos.z*ca; pos.y = ny; pos.z = nz; }
    { float ca = cos(${v[2]}), sa = sin(${v[2]}); float nx = pos.x*ca - pos.z*sa, nz = pos.x*sa + pos.z*ca; pos.x = nx; pos.z = nz; }`,
  },
  {
    id: 22, key: 'twist', name: 'Twist', wRule: W_UNCHANGED,
    params: [{ name: 'Twist', default: 30.0, min: -180.0, max: 180.0, step: 0.5, type: 'angle' }],
    // Rotate XY by an angle proportional to height (z): screws / spiral towers.
    // Volume-preserving (|Jacobian| = 1) but NOT a strict isometry — the analytic
    // r/|w| DE loosens at high twist (thin/banded artifacts), so keep it modest.
    wgsl: `
        let tw = radians(op.p0) * pos.z;
        let tc = cos(tw); let ts = sin(tw);
        let tx = pos.x * tc - pos.y * ts;
        let ty = pos.x * ts + pos.y * tc;
        pos.x = tx; pos.y = ty;`,
    glsl: (v) => `
    // twist — rotate XY by (rate * z). Volume-preserving; DE loosens at high twist.
    {
        float tw = ${v[0]} * pos.z;
        float tc = cos(tw), ts = sin(tw);
        float tx = pos.x * tc - pos.y * ts;
        float ty = pos.x * ts + pos.y * tc;
        pos.x = tx; pos.y = ty;
    }`,
  },
  {
    id: 23, key: 'quadratic', name: 'Quadratic (z²)', wRule: W_BULB,
    params: [],
    // Complex square in the XY plane (Mandelbrot family): (x,y) → (x²−y², 2xy),
    // z carried. ESCAPE-TIME like Mandelbulb Power — set DEoption 0 + AddC. w
    // tracks the analytic derivative dr = 2·|z|·dr + 1 for the escape-time DE.
    wgsl: `
        let qr = length(pos.xy);
        w = 2.0 * qr * w + 1.0;
        let qx = pos.x * pos.x - pos.y * pos.y;
        let qy = 2.0 * pos.x * pos.y;
        pos.x = qx; pos.y = qy;`,
    glsl: () => `
    // Quadratic z² (complex square in XY, Mandelbrot family) — escape-time.
    // Engine DEoption 0 drives the surface; w carries dr = 2·|z|·dr + 1.
    {
        float qr = length(pos.xy);
        w = 2.0 * qr * w + 1.0;
        float qx = pos.x * pos.x - pos.y * pos.y;
        float qy = 2.0 * pos.x * pos.y;
        pos.x = qx; pos.y = qy;
    }`,
  },
  {
    id: 24, key: 'icosaFold', name: 'Icosahedral Fold', wRule: W_UNCHANGED,
    params: [],
    // Reflect into the icosahedral fundamental domain (golden-ratio plane
    // normals) — Knighty-style kaleidoscopic IFS with 5-fold symmetry. Pure
    // reflections: |Jacobian| = 1, w untouched. Pair with scale + translate
    // (like Menger/Sierpinski) to grow the gasket.
    wgsl: `
        let i1 = vec3f(-0.809017, 0.309017, 0.5);
        let i2 = vec3f(0.5, -0.809017, 0.309017);
        let i3 = vec3f(0.309017, 0.5, -0.809017);
        pos = abs(pos);
        pos = pos - 2.0 * min(dot(pos, i1), 0.0) * i1;
        pos = pos - 2.0 * min(dot(pos, i2), 0.0) * i2;
        pos = pos - 2.0 * min(dot(pos, i3), 0.0) * i3;`,
    glsl: () => `
    // icosahedral fold — reflect into the fundamental domain (golden-ratio
    // normals). Reflections: |Jacobian| = 1, w untouched.
    {
        vec3 i1 = vec3(-0.809017, 0.309017, 0.5);
        vec3 i2 = vec3(0.5, -0.809017, 0.309017);
        vec3 i3 = vec3(0.309017, 0.5, -0.809017);
        pos = abs(pos);
        pos -= 2.0 * min(dot(pos, i1), 0.0) * i1;
        pos -= 2.0 * min(dot(pos, i2), 0.0) * i2;
        pos -= 2.0 * min(dot(pos, i3), 0.0) * i3;
    }`,
  },
  {
    id: 25, key: 'menger', name: 'Menger', wRule: W_UNCHANGED,
    params: [{ name: 'Smoothness', default: 0.005, min: -0.1, max: 0.1, step: 0.001, type: 'double' }],
    // Menger sponge fold (abs + descending sort + z-wrap). Smoothness rounds the
    // edges; its SIGN picks the rounding *type*:
    //   s = 0   sharp — exactly an octahedral fold + the Menger z-wrap.
    //   s > 0   sqrt rounding: |x|→sqrt(x²+s), min(t,0)→0.5(t−sqrt(t²+s)).
    //   s < 0   polynomial rounding (k=−s): |x| blended into a parabola within k
    //           of 0, IQ polynomial smin — a flatter, "linear→quadratic" edge.
    // Both smoothings are 1-Lipschitz (non-expanding) → |Jacobian| ≤ 1, so the
    // r/|w| DE stays a valid conservative bound and w is untouched. Pair with
    // Scale 3 + Translate(−2,−2,0) for the sponge. (Ports Luca GN's
    // MengerIFSsmooth incl. its s<0 alternate-smoothing mode. c = 1/3.)
    // Caveat: rounding compounds every iteration, so fine detail washes out fast.
    wgsl: `
        let s = op.p0;
        let c = 1.0 / 3.0;
        if (s >= 0.0) {
          pos = sqrt(pos * pos + s);
          var t = pos.x - pos.y; t = 0.5 * (t - sqrt(t * t + s)); pos.x -= t; pos.y += t;
          t = pos.x - pos.z; t = 0.5 * (t - sqrt(t * t + s)); pos.x -= t; pos.z += t;
          t = pos.y - pos.z; t = 0.5 * (t - sqrt(t * t + s)); pos.y -= t; pos.z += t;
          pos.z = c - sqrt((pos.z - c) * (pos.z - c) + s);
        } else {
          let k = -s;
          pos = vec3f(
            select(abs(pos.x), pos.x * pos.x / (2.0 * k) + 0.5 * k, abs(pos.x) < k),
            select(abs(pos.y), pos.y * pos.y / (2.0 * k) + 0.5 * k, abs(pos.y) < k),
            select(abs(pos.z), pos.z * pos.z / (2.0 * k) + 0.5 * k, abs(pos.z) < k));
          var t = pos.x - pos.y; var h = max(k - abs(t), 0.0) / k; t = min(t, 0.0) - h * h * k * 0.25; pos.x -= t; pos.y += t;
          t = pos.x - pos.z; h = max(k - abs(t), 0.0) / k; t = min(t, 0.0) - h * h * k * 0.25; pos.x -= t; pos.z += t;
          t = pos.y - pos.z; h = max(k - abs(t), 0.0) / k; t = min(t, 0.0) - h * h * k * 0.25; pos.y -= t; pos.z += t;
          let dz = pos.z - c; let adz = abs(dz);
          pos.z = c - select(adz, dz * dz / (2.0 * k) + 0.5 * k, adz < k);
        }`,
    glsl: (v) => `
    // menger fold — abs + sort + z-wrap. Smoothness sign picks the rounding type
    // (>0 sqrt, <0 polynomial, 0 sharp). 1-Lipschitz → w untouched.
    {
        float s = ${v[0]};
        float c = 1.0 / 3.0;
        if (s >= 0.0) {
            pos = sqrt(pos * pos + s);
            float t = pos.x - pos.y; t = 0.5 * (t - sqrt(t * t + s)); pos.x -= t; pos.y += t;
            t = pos.x - pos.z; t = 0.5 * (t - sqrt(t * t + s)); pos.x -= t; pos.z += t;
            t = pos.y - pos.z; t = 0.5 * (t - sqrt(t * t + s)); pos.y -= t; pos.z += t;
            pos.z = c - sqrt((pos.z - c) * (pos.z - c) + s);
        } else {
            float k = -s;
            pos.x = abs(pos.x) < k ? pos.x * pos.x / (2.0 * k) + 0.5 * k : abs(pos.x);
            pos.y = abs(pos.y) < k ? pos.y * pos.y / (2.0 * k) + 0.5 * k : abs(pos.y);
            pos.z = abs(pos.z) < k ? pos.z * pos.z / (2.0 * k) + 0.5 * k : abs(pos.z);
            float t, h;
            t = pos.x - pos.y; h = max(k - abs(t), 0.0) / k; t = min(t, 0.0) - h * h * k * 0.25; pos.x -= t; pos.y += t;
            t = pos.x - pos.z; h = max(k - abs(t), 0.0) / k; t = min(t, 0.0) - h * h * k * 0.25; pos.x -= t; pos.z += t;
            t = pos.y - pos.z; h = max(k - abs(t), 0.0) / k; t = min(t, 0.0) - h * h * k * 0.25; pos.y -= t; pos.z += t;
            float dz = pos.z - c; float adz = abs(dz);
            pos.z = c - (adz < k ? dz * dz / (2.0 * k) + 0.5 * k : adz);
        }
    }`,
  },
  // ── Phase 1 palette growth (DE-sound, data-only). Each folds a cluster of
  //    classic 3D-fractal transforms down onto one primitive. ──
  {
    id: 26, key: 'polyAngleFold', name: 'Poly Angle Fold', wRule: W_UNCHANGED,
    params: [
      { name: 'Symmetry', default: 6.0, min: 2.0, max: 16.0, step: 1.0, type: 'double' },
      { name: 'Angle',    default: 0.0, min: -180.0, max: 180.0, step: 0.5, type: 'angle' },
      { name: 'Mirror',   default: 0.0, min: 0.0, max: 1.0, step: 1.0, type: 'double' },
    ],
    // Knighty "PolyFold": ROTATE the XY angle to the nearest of N sectors (a
    // discrete rotation — NOT a reflection-into-wedge like Kaleidoscope). This is
    // the building block of the Apollonian / knot / polygon decorated IFS. Mirror
    // > 0 adds the final |angle| reflection (the "-sym" variant). Pure isometry:
    // |Jacobian| = 1, w untouched. Bound the radius with a box/sphere fold.
    wgsl: `
        let n = max(op.p0, 2.0);
        let wedge = 6.2831853 / n;
        let off = radians(op.p1);
        var ang = atan2(pos.y, pos.x) - off;
        ang = ang - wedge * floor(ang / wedge + 0.5);
        if (op.p2 > 0.5) { ang = abs(ang); }
        ang = ang + off;
        let rad = length(pos.xy);
        pos.x = cos(ang) * rad;
        pos.y = sin(ang) * rad;`,
    glsl: (v) => `
    // poly angle fold — snap the XY angle to one of N rotational sectors
    // (rotation, not reflection). Mirror>0 adds the -sym reflection. w untouched.
    {
        float n = max(${v[0]}, 2.0);
        float wedge = 6.2831853 / n;
        float off = ${v[1]};
        float ang = atan(pos.y, pos.x) - off;
        ang = ang - wedge * floor(ang / wedge + 0.5);
        if (${v[2]} > 0.5) ang = abs(ang);
        ang += off;
        float rad = length(pos.xy);
        pos.x = cos(ang) * rad;
        pos.y = sin(ang) * rad;
    }`,
  },
  {
    id: 27, key: 'cylinderFold', name: 'Cylinder Fold', wRule: W_MUL_K,
    params: [
      { name: 'MinRadius',   default: 0.5, min: 0.05, max: 2.0, step: 0.01, type: 'double' },
      { name: 'FixedRadius', default: 1.0, min: 0.1,  max: 3.0, step: 0.01, type: 'double' },
    ],
    // Sphere fold whose radius is measured in the XY plane only (the Z axis runs
    // free) — the Amazing-Surf "cylinder" ball-fold. Conformal scale by k tracked
    // onto w, same DE bookkeeping as the sphere fold.
    wgsl: `
        let minR2 = op.p0 * op.p0;
        let fixedR2 = op.p1 * op.p1;
        let r2 = pos.x * pos.x + pos.y * pos.y;
        var k = 1.0;
        if (r2 < minR2) { k = fixedR2 / minR2; }
        else if (r2 < fixedR2) { k = fixedR2 / r2; }
        pos = pos * k;
        w = w * k;`,
    glsl: (v) => `
    // cylinder fold — sphere fold with the radius taken in XY only (Z free).
    {
        float minR2 = ${v[0]} * ${v[0]};
        float fixedR2 = ${v[1]} * ${v[1]};
        float r2 = pos.x * pos.x + pos.y * pos.y;
        float k;
        if      (r2 < minR2)   k = fixedR2 / minR2;
        else if (r2 < fixedR2) k = fixedR2 / r2;
        else                   k = 1.0;
        pos  *= k;
        w    *= k;
        g_wq *= k;
    }`,
  },
  {
    id: 28, key: 'radialInvert', name: 'Inversion (shifted)', wRule: W_MUL_K,
    params: [
      { name: 'CenterX', default: 0.0, min: -2.0, max: 2.0, step: 0.01, type: 'double' },
      { name: 'CenterY', default: 0.0, min: -2.0, max: 2.0, step: 0.01, type: 'double' },
      { name: 'CenterZ', default: 0.0, min: -2.0, max: 2.0, step: 0.01, type: 'double' },
    ],
    // Unit sphere inversion about a SHIFTED center o:  p → (p−o)/|p−o|² + o.
    // Conformal (isotropic scale k = 1/|p−o|² → tracked onto w) — the Kleinian /
    // Poincaré generator. With center 0 it's the plain unit inversion (= Sphere
    // Inversion at radius 1); offsetting the center is what this adds. Unbounded
    // alone — pair with a box/sphere fold or the attractor escapes (blank sky).
    wgsl: `
        let o = vec3f(op.p0, op.p1, op.p2);
        let d = pos - o;
        let dd = max(dot(d, d), 1e-6);
        let k = 1.0 / dd;
        pos = d * k + o;
        w = w * k;`,
    glsl: (v) => `
    // shifted unit inversion — p → (p−o)/|p−o|² + o (conformal: k → w).
    {
        vec3 o = vec3(${v[0]}, ${v[1]}, ${v[2]});
        vec3 d = pos - o;
        float dd = max(dot(d, d), 1e-6);
        float k = 1.0 / dd;
        pos  = d * k + o;
        w    *= k;
        g_wq *= k;
    }`,
  },
  {
    id: 29, key: 'bulbAxis', name: 'Bulb Power (axis)', wRule: W_BULB,
    params: [
      { name: 'Power', default: 8.0, min: 2.0, max: 16.0, step: 0.1, type: 'double' },
      { name: 'Axis',  default: 0.0, min: 0.0, max: 2.0, step: 1.0, type: 'double' },
    ],
    // Mandelbulb spherical power z→zⁿ with a SELECTABLE polar axis (0 = z, the
    // White/Nylander default; 1 = y, the IQ convention; 2 = x). The convention
    // changes the bulb's shape, so this one op covers the IQ/PG/Ikenaga/… bulb
    // family. Escape-time like Mandelbulb Power: set DEoption 0 + AddC; w carries
    // the analytic derivative dr. Axis 0 reproduces Mandelbulb Power exactly.
    wgsl: `
        let bp = op.p0;
        let m = i32(op.p1 + 0.5);
        let br = length(pos);
        if (br > 1e-9) {
          var up = pos.z; var a = pos.x; var b = pos.y;
          if (m == 1) { up = pos.y; a = pos.z; b = pos.x; }
          else if (m == 2) { up = pos.x; a = pos.y; b = pos.z; }
          let bth = acos(clamp(up / br, -1.0, 1.0)) * bp;
          let bph = atan2(b, a) * bp;
          let brn = pow(br, bp);
          w = bp * brn / br * w + 1.0;
          let bst = sin(bth);
          let na = brn * bst * cos(bph);
          let nb = brn * bst * sin(bph);
          let nup = brn * cos(bth);
          if (m == 1) { pos = vec3f(nb, nup, na); }
          else if (m == 2) { pos = vec3f(nup, na, nb); }
          else { pos = vec3f(na, nb, nup); }
        }`,
    glsl: (v) => `
    // Mandelbulb power with a selectable polar axis (0 z · 1 y · 2 x) — escape-time.
    {
        float bp = ${v[0]};
        int m = int(${v[1]} + 0.5);
        float br = length(pos);
        if (br > 1e-9) {
            float up = pos.z, a = pos.x, b = pos.y;
            if (m == 1) { up = pos.y; a = pos.z; b = pos.x; }
            else if (m == 2) { up = pos.x; a = pos.y; b = pos.z; }
            float bth = acos(clamp(up / br, -1.0, 1.0)) * bp;
            float bph = atan(b, a) * bp;
            float brn = pow(br, bp);
            w = bp * brn / br * w + 1.0;
            float bst = sin(bth);
            float na = brn * bst * cos(bph);
            float nb = brn * bst * sin(bph);
            float nup = brn * cos(bth);
            if (m == 1) pos = vec3(nb, nup, na);
            else if (m == 2) pos = vec3(nup, na, nb);
            else pos = vec3(na, nb, nup);
        }
    }`,
  },
  {
    id: 30, key: 'hexFold', name: 'Hex Fold', wRule: W_UNCHANGED,
    params: [],
    // Hexagonal fold: reflect the XY plane into a 60° wedge (abs + one 60° plane
    // reflection → 6-fold symmetry); Z runs free. Pure reflections: |Jacobian| =
    // 1, w untouched. Pair with scale + translate for a hex-symmetric gasket.
    wgsl: `
        let kx = -0.8660254;
        let ky = 0.5;
        pos.x = abs(pos.x);
        pos.y = abs(pos.y);
        let d = min(kx * pos.x + ky * pos.y, 0.0);
        pos.x = pos.x - 2.0 * d * kx;
        pos.y = pos.y - 2.0 * d * ky;`,
    glsl: () => `
    // hex fold — reflect XY into a 60° wedge (6-fold symmetry), Z free.
    {
        float kx = -0.8660254;
        float ky = 0.5;
        pos.x = abs(pos.x);
        pos.y = abs(pos.y);
        float d = min(kx * pos.x + ky * pos.y, 0.0);
        pos.x -= 2.0 * d * kx;
        pos.y -= 2.0 * d * ky;
    }`,
  },
  {
    id: 31, key: 'absXYZ', name: 'Abs XYZ', wRule: W_UNCHANGED,
    params: [
      { name: 'AbsX', default: 1.0, min: 0.0, max: 1.0, step: 1.0, type: 'double' },
      { name: 'AbsY', default: 1.0, min: 0.0, max: 1.0, step: 1.0, type: 'double' },
      { name: 'AbsZ', default: 1.0, min: 0.0, max: 1.0, step: 1.0, type: 'double' },
    ],
    // Per-axis abs fold — independent "abs X / abs Y / abs Z" toggles. Each toggle
    // (param > 0.5 = on) reflects that axis into its positive half-space; folding
    // only some axes gives an ASYMMETRIC mirror (the gap absFold/absOffsetFold,
    // which always do all three, can't reach). Pure reflection: |Jacobian| = 1,
    // w untouched. All three on = plain Abs Fold.
    wgsl: `
        if (op.p0 > 0.5) { pos.x = abs(pos.x); }
        if (op.p1 > 0.5) { pos.y = abs(pos.y); }
        if (op.p2 > 0.5) { pos.z = abs(pos.z); }`,
    glsl: (v) => `
    // per-axis abs fold (abs X/Y/Z) — reflect only the enabled axes.
    if (${v[0]} > 0.5) pos.x = abs(pos.x);
    if (${v[1]} > 0.5) pos.y = abs(pos.y);
    if (${v[2]} > 0.5) pos.z = abs(pos.z);`,
  },
];

const _byId  = new Map(OPERATORS.map(o => [o.id, o]));
const _byKey = new Map(OPERATORS.map(o => [o.key, o]));
export const byId  = (id)  => _byId.get(id);
export const byKey = (key) => _byKey.get(key);

// A formula's DE is sound to the analytic `r/|w|` estimator as long as every
// op accounts for its own w (all our fold-family ops do). A power/escape-time
// op would flip the whole formula to a different DE family — flagged here so
// the UI can warn. For now the palette is all DE-sound.
// The ops that actually run: muted ops are kept in the list (so the user can
// toggle them back) but excluded from rendering, export, and DE classification.
export const activeOps = (formula) => formula.ops.filter(op => !op.muted);

export function isDeSound(formula) {
  return activeOps(formula).every(op => {
    const def = byKey(op.key);
    return def && [W_UNCHANGED, W_MUL_K, W_MUL_SCALE].includes(def.wRule);
  });
}

// True if the stack contains an escape-time op (Mandelbulb power). Such a
// formula must use the escape-time DE (deOption 0), not the IFS r/|w|. Deriving
// this from the ops means authoring a bulb from a blank slate "just works" — no
// hidden DE-family switch for the user to find.
export function isEscapeTime(formula) {
  return activeOps(formula).some(op => {
    const def = byKey(op.key);
    return def && def.wRule === W_BULB;
  });
}

// The DE family the preview + export should use: escape-time if the stack has a
// bulb op, otherwise the formula's stored deOption (2 = analytic IFS default).
export function effectiveDeOption(formula) {
  return isEscapeTime(formula) ? 0 : (formula.deOption ?? 2);
}

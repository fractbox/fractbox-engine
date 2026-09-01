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
//   category  palette grouping for op pickers: 'fold' | 'sphere' | 'symmetry' |
//           'move' | 'power' (+ 'warp' when Phase C lands). Presentation only —
//           the engine never reads it; both apps' pickers group by it.
//   wRule   how this op affects the running derivative `w` (the DE bookkeeping).
//           Folds/rotations are isometries (w ×1) → DE-free. Only sphere-fold
//           and scale move w. This is *the* reason primitives compose into a
//           correct distance estimate without any global re-derivation.
//   params  [{name, default, min, max, step, type}]  (type: 'double' | 'angle')
//           Drives the UI sliders AND the GPU op-buffer packing (positional).
//   seam    (#633) true = the op is discontinuous and reports the distance to
//           its tear set into the seam channel (WGSL `seamOut` / GLSL `g_seam`
//           / CPU `s.seam`) — see the seam-clamped DE contract above hingeFold
//           (id 67). Emission of the channel is keyed on this flag
//           (shader.js usesSeam), so seam-free formulas stay byte-identical.
//   wgsl    the interpreter switch-case body. Mutates `pos` / `w`; reads its
//           params from `op.p0`, `op.p1`, `op.p2`.
//   glsl(v) emits the native iterateJIT_ body fragment. `v` is the array of
//           local variable names the exporter assigned to this op's params.
//
// The WGSL and GLSL emitters are kept side-by-side on purpose: they are the
// same math in two shading languages, and a divergence between them is exactly
// the kind of bug a reviewer can catch by eye.
// ─────────────────────────────────────────────────────────────────────────

// The one import: the D0 shape-leaf registry (itself zero-dep), so the
// classifiers below (isApproxDE) can see approximate LEAVES as well as ops.
import { leafById } from "./leaves.js";
import { hybridSlots } from "./hybridmodel.js";

export const W_UNCHANGED = "unchanged"; // isometry — |Jacobian| = 1
export const W_MUL_K = "mul_k"; // sphere fold — multiplies w by fold factor k
export const W_MUL_SCALE = "mul_scale"; // conformal scale — multiplies w by |scale|
export const W_BULB = "bulb"; // escape-time power — tracks the analytic
// derivative dr; flips the DE family (not IFS)
export const W_BULB_NUMERIC = "bulb_numeric"; // escape-time map with NO analytic dr —
// routes the whole formula to the numeric
// (finite-difference) DE; w is ignored

export const OPERATORS = [
  {
    id: 0,
    key: "boxFold",
    name: "Box Fold",
    wRule: W_UNCHANGED,
    category: "fold",
    blurb:
      "Mirrors anything poking past a set wall back inward — the move that stamps crisp, repeating boxes into a Mandelbox.",
    params: [
      {
        name: "FoldLimit",
        default: 1.0,
        min: 0.1,
        max: 3.0,
        step: 0.01,
        type: "double",
        tip: "Where the walls sit. Lower packs tighter, busier boxes; higher gives fewer, roomier ones.",
      },
    ],
    wgsl: `
        let fold = op.p0;
        pos.x = abs(pos.x + fold) - abs(pos.x - fold) - pos.x;
        pos.y = abs(pos.y + fold) - abs(pos.y - fold) - pos.y;
        pos.z = abs(pos.z + fold) - abs(pos.z - fold) - pos.z;`,
    // perturbation twin (PERTURBATION_ZOOM_IMPL.md PR-2): the exact delta
    // map — mutates (ptD: vec3f, w: f32) given the reference records
    // pv0..pv3 (core/perturb.js layout). Plain f32 — no EFTs, no launder.
    wgslPt: `
        let fold = op.p0;
        for (var a = 0u; a < 3u; a = a + 1u) {
          let m1 = pv1[a]; let m2 = pv2[a]; let dx = ptD[a];
          var refB = 1; if (m1 < 0.0) { refB = 2; } else if (m2 < 0.0) { refB = 0; }
          var sB = 1; if (dx > m1) { sB = 2; } else if (dx < -m2) { sB = 0; }
          if (refB == 1 && sB == 1) { ptD[a] = dx; }
          else if (refB == sB) { ptD[a] = -dx; }
          else if (refB == 1 && sB == 2) { ptD[a] = 2.0 * m1 - dx; }
          else if (refB == 2 && sB == 1) { ptD[a] = dx - 2.0 * m1; }
          else if (refB == 1 && sB == 0) { ptD[a] = -2.0 * m2 - dx; }
          else if (refB == 0 && sB == 1) { ptD[a] = dx + 2.0 * m2; }
          else if (refB == 2 && sB == 0) { ptD[a] = -4.0 * fold - dx; }
          else { ptD[a] = 4.0 * fold - dx; }
        }`,
    // deep zoom P4 (DEEP_ZOOM_DF64.md): df64 twin — mutates (P: Df3, w).
    wgslDf: `
        let fold = op.p0;
        let bx = df3_get(P, 0); let by = df3_get(P, 1); let bz = df3_get(P, 2);
        let nx = df_sub(df_sub(df_abs(df_add_f32(bx, fold)), df_abs(df_add_f32(bx, -fold))), bx);
        let ny = df_sub(df_sub(df_abs(df_add_f32(by, fold)), df_abs(df_add_f32(by, -fold))), by);
        let nz = df_sub(df_sub(df_abs(df_add_f32(bz, fold)), df_abs(df_add_f32(bz, -fold))), bz);
        P = Df3(vec3f(nx.x, ny.x, nz.x), vec3f(nx.y, ny.y, nz.y));
`,
    glsl: (v) => `
    // box fold (reflection: |Jacobian| = 1, w untouched)
    pos.x = abs(pos.x + ${v[0]}) - abs(pos.x - ${v[0]}) - pos.x;
    pos.y = abs(pos.y + ${v[0]}) - abs(pos.y - ${v[0]}) - pos.y;
    pos.z = abs(pos.z + ${v[0]}) - abs(pos.z - ${v[0]}) - pos.z;`,
  },
  {
    id: 1,
    key: "sphereFold",
    name: "Sphere Fold",
    wRule: W_MUL_K,
    category: "sphere",
    blurb:
      "Inflates the space near the center outward like a magnifying bubble — the Mandelbox's rounding-and-bulging move.",
    params: [
      {
        name: "MinRadius",
        default: 0.5,
        min: 0.05,
        max: 2.0,
        step: 0.01,
        type: "double",
        tip: "Size of the inner core that gets pushed out the most. Smaller = a tighter, denser center.",
      },
      {
        name: "FixedRadius",
        default: 1.0,
        min: 0.1,
        max: 3.0,
        step: 0.01,
        type: "double",
        tip: "Size of the bubble doing the inflating. Bigger = more overall puff.",
      },
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
    // perturbation twin (PERTURBATION_ZOOM_IMPL.md PR-2): the exact delta
    // map — mutates (ptD: vec3f, w: f32) given the reference records
    // pv0..pv3 (core/perturb.js layout). Plain f32 — no EFTs, no launder.
    wgslPt: `
        let minR2 = op.p0 * op.p0;
        let fixedR2 = op.p1 * op.p1;
        let K0 = fixedR2 / minR2;
        let Z = pv0.xyz; let rr2 = pv1.w; let mr1 = pv2.w; let mr2 = pv3.x;
        let q = 2.0 * dot(Z, ptD) + dot(ptD, ptD);
        let rs2 = rr2 + q;
        var refB = 1; if (mr1 < 0.0) { refB = 0; } else if (mr2 <= 0.0) { refB = 2; }
        var sB = 1; if (mr1 + q < 0.0) { sB = 0; } else if (mr2 - q <= 0.0) { sB = 2; }
        var kr = pv0.w; var dk = 0.0;
        if (refB == sB) { if (refB == 1) { dk = -(fixedR2 * q) / (rs2 * rr2); } }
        else if (refB == 0 && sB == 1) { dk = -(fixedR2 * (mr1 + q)) / (rs2 * minR2); }
        else if (refB == 1 && sB == 0) { dk = (fixedR2 * mr1) / (minR2 * rr2); }
        else if (refB == 1 && sB == 2) { dk = -mr2 / rr2; }
        else if (refB == 2 && sB == 1) { dk = (mr2 - q) / rs2; }
        else if (refB == 0 && sB == 2) { dk = 1.0 - K0; }
        else { dk = K0 - 1.0; }
        ptD = kr * ptD + dk * (Z + ptD);
        w = w * (kr + dk);`,
    // deep zoom P4 (DEEP_ZOOM_DF64.md): df64 twin — mutates (P: Df3, w).
    wgslDf: `
        let minR2 = two_prod(op.p0, op.p0);
        let fixedR2 = two_prod(op.p1, op.p1);
        let r2 = df3_dot(P, P);
        var k = vec2f(1.0, 0.0);
        if (df_lt(r2, minR2)) { k = df_div(fixedR2, minR2); }
        else if (df_lt(r2, fixedR2)) { k = df_div(fixedR2, r2); }
        P = df3_mul(P, k);
        w = w * (k.x + k.y);
`,
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
    id: 2,
    key: "scale",
    name: "Scale",
    wRule: W_MUL_SCALE,
    category: "move",
    blurb:
      "Zooms the whole shape in or out each step — the engine that makes detail repeat at smaller and smaller sizes.",
    params: [
      {
        name: "Scale",
        default: 2.0,
        min: -4.0,
        max: 4.0,
        step: 0.01,
        type: "double",
        tip: "Zoom factor per step. Around 2 is classic; negative values flip and can add extra symmetry.",
      },
    ],
    wgsl: `
        let s = op.p0;
        pos = pos * s;
        w = w * abs(s);`,
    // perturbation twin (PERTURBATION_ZOOM_IMPL.md PR-2): the exact delta
    // map — mutates (ptD: vec3f, w: f32) given the reference records
    // pv0..pv3 (core/perturb.js layout). Plain f32 — no EFTs, no launder.
    wgslPt: `
        ptD = ptD * op.p0;
        w = w * abs(op.p0);`,
    // deep zoom P4 (DEEP_ZOOM_DF64.md): df64 twin — mutates (P: Df3, w).
    wgslDf: `
        let s = op.p0;
        P = df3_mul_f32(P, s);
        w = w * abs(s);
`,
    glsl: (v) => `
    // conformal scale (the expanding map → |scale| onto w)
    pos  *= ${v[0]};
    w    *= abs(${v[0]});
    g_wq *= abs(${v[0]});`,
  },
  {
    id: 3,
    key: "rotateXY",
    name: "Rotate XY",
    wRule: W_UNCHANGED,
    category: "move",
    blurb:
      "Spins the shape flat (around the depth axis) a fixed amount every step, for a swirl or pinwheel feel.",
    params: [
      {
        name: "AngleXY",
        default: 0.0,
        min: -180.0,
        max: 180.0,
        step: 0.5,
        type: "angle",
        tip: "Degrees of spin per step. Small angles twist gently; large ones churn the shape.",
      },
    ],
    wgsl: `
        let a = radians(op.p0);
        let ca = cos(a); let sa = sin(a);
        let nx = pos.x * ca - pos.y * sa;
        let ny = pos.x * sa + pos.y * ca;
        pos.x = nx; pos.y = ny;`,
    // perturbation twin (PERTURBATION_ZOOM_IMPL.md PR-2): the exact delta
    // map — mutates (ptD: vec3f, w: f32) given the reference records
    // pv0..pv3 (core/perturb.js layout). Plain f32 — no EFTs, no launder.
    wgslPt: `
        let a = radians(op.p0);
        let ca = cos(a); let sa = sin(a);
        let nx = ptD.x * ca - ptD.y * sa;
        let ny = ptD.x * sa + ptD.y * ca;
        ptD.x = nx; ptD.y = ny;`,
    // deep zoom P4 (DEEP_ZOOM_DF64.md): df64 twin — mutates (P: Df3, w).
    wgslDf: `
        let a = radians(op.p0);
        let ca = cos(a); let sa = sin(a);
        let rx = df3_get(P, 0); let ry = df3_get(P, 1);
        let nx = df_sub(df_mul_f32(rx, ca), df_mul_f32(ry, sa));
        let ny = df_add(df_mul_f32(rx, sa), df_mul_f32(ry, ca));
        P = Df3(vec3f(nx.x, ny.x, P.hi.z), vec3f(nx.y, ny.y, P.lo.z));
`,
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
    id: 4,
    key: "rotateYZ",
    name: "Rotate YZ",
    wRule: W_UNCHANGED,
    category: "move",
    blurb:
      "Tips the shape forward and back (around the side axis) each step, tilting the whole structure.",
    params: [
      {
        name: "AngleYZ",
        default: 0.0,
        min: -180.0,
        max: 180.0,
        step: 0.5,
        type: "angle",
        tip: "Degrees of tilt per step.",
      },
    ],
    wgsl: `
        let a = radians(op.p0);
        let ca = cos(a); let sa = sin(a);
        let ny = pos.y * ca - pos.z * sa;
        let nz = pos.y * sa + pos.z * ca;
        pos.y = ny; pos.z = nz;`,
    // perturbation twin (PERTURBATION_ZOOM_IMPL.md PR-2): the exact delta
    // map — mutates (ptD: vec3f, w: f32) given the reference records
    // pv0..pv3 (core/perturb.js layout). Plain f32 — no EFTs, no launder.
    wgslPt: `
        let a = radians(op.p0);
        let ca = cos(a); let sa = sin(a);
        let ny = ptD.y * ca - ptD.z * sa;
        let nz = ptD.y * sa + ptD.z * ca;
        ptD.y = ny; ptD.z = nz;`,
    // deep zoom P4 (DEEP_ZOOM_DF64.md): df64 twin — mutates (P: Df3, w).
    wgslDf: `
        let a = radians(op.p0);
        let ca = cos(a); let sa = sin(a);
        let ry = df3_get(P, 1); let rz = df3_get(P, 2);
        let ny = df_sub(df_mul_f32(ry, ca), df_mul_f32(rz, sa));
        let nz = df_add(df_mul_f32(ry, sa), df_mul_f32(rz, ca));
        P = Df3(vec3f(P.hi.x, ny.x, nz.x), vec3f(P.lo.x, ny.y, nz.y));
`,
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
    id: 5,
    key: "absFold",
    name: "Abs Fold",
    wRule: W_UNCHANGED,
    category: "fold",
    blurb:
      "Folds all three axes into one corner so everything mirrors into positive space — the simplest symmetry move.",
    params: [],
    wgsl: `        pos = abs(pos);`,
    // perturbation twin (PERTURBATION_ZOOM_IMPL.md PR-2): the exact delta
    // map — mutates (ptD: vec3f, w: f32) given the reference records
    // pv0..pv3 (core/perturb.js layout). Plain f32 — no EFTs, no launder.
    wgslPt: `
        for (var a = 0u; a < 3u; a = a + 1u) {
          let zx = pv0[a]; let dx = ptD[a];
          let refUp = zx >= 0.0; let sPos = dx >= -zx;
          if (refUp == sPos) { ptD[a] = select(-dx, dx, refUp); }
          else { ptD[a] = select(2.0 * zx + dx, -2.0 * zx - dx, refUp); }
        }`,
    // deep zoom P4 (DEEP_ZOOM_DF64.md): df64 twin — mutates (P: Df3, w).
    wgslDf: `
        let ax = df_abs(df3_get(P, 0));
        let ay = df_abs(df3_get(P, 1));
        let az = df_abs(df3_get(P, 2));
        P = Df3(vec3f(ax.x, ay.x, az.x), vec3f(ax.y, ay.y, az.y));
`,
    glsl: () => `
    // abs fold into the positive octant (reflection)
    pos = abs(pos);`,
  },
  {
    id: 6,
    key: "kaleido",
    name: "Kaleidoscope",
    wRule: W_UNCHANGED,
    category: "symmetry",
    blurb:
      "Wraps the shape into repeating pie-slice wedges around the center for kaleidoscope symmetry. Pair it with a box or sphere fold or the shape flies apart.",
    params: [
      {
        name: "Symmetry",
        default: 6.0,
        min: 2.0,
        max: 16.0,
        step: 1.0,
        type: "double",
        tip: "How many wedges go around. 6 gives six-fold; higher = more petals.",
      },
      {
        name: "Twist",
        default: 0.0,
        min: -180.0,
        max: 180.0,
        step: 0.5,
        type: "angle",
        tip: "Rotates the wedges, sweeping the petals around.",
      },
      {
        // Mirror OFF = pure rotational sector-snap (each wedge maps onto the
        // base wedge by rotation, no reflection) — the MB3D PolyFold/Koch
        // pre-step (PRIMITIVE_COVERAGE_PLAN.md, pre-step pass). Default ON is
        // the historic kaleidoscope; old 2-value links sanitize to ON.
        name: "Mirror",
        default: 1.0,
        min: 0.0,
        max: 1.0,
        step: 1.0,
        type: "double",
        tip: "On reflects each wedge (kaleidoscope); off repeats it by rotation only.",
      },
    ],
    wgsl: `
        let wedge = 6.2831853 / max(op.p0, 2.0);
        var ang = atan2(pos.y, pos.x);
        ang = ang - wedge * floor(ang / wedge + 0.5);
        if (op.p2 > 0.5) { ang = abs(ang); }
        ang = ang + radians(op.p1);
        let rad = length(pos.xy);
        pos.x = cos(ang) * rad;
        pos.y = sin(ang) * rad;`,
    glsl: (v) => `
    // N-fold kaleidoscope angle fold (reflection: w untouched).
    // NOTE: angle-folds bound DIRECTION only — pair with a box/sphere fold to
    // bound radius or the attractor escapes (renders blank sky).
    // #553: Mirror MUST be a real GLSL runtime branch on the v[2] reference
    // itself, not a JS-side ternary over it — v[2] is a variable-name string
    // ("uP[2]"/"p2"), never a number, so comparing it in JS is always false
    // (NaN). Mirrors the WGSL body's "if (op.p2 > 0.5) { ang = abs(ang); }"
    // and the polyFold precedent's "if (v[2] > 0.5) ang = abs(ang);".
    {
        float wedge = 6.2831853 / max(${v[0]}, 2.0);
        float ang = atan(pos.y, pos.x);
        ang = ang - wedge * floor(ang / wedge + 0.5);
        if (${v[2]} > 0.5) { ang = abs(ang); }
        ang = ang + ${v[1]};
        float rad = length(pos.xy);
        pos.x = cos(ang) * rad;
        pos.y = sin(ang) * rad;
    }`,
  },
  {
    id: 7,
    key: "translate",
    name: "Translate",
    wRule: W_UNCHANGED,
    category: "move",
    blurb:
      "Shifts the whole shape sideways, up/down, or in depth by a fixed offset.",
    // Range was ±2 until #538. The shipped "Cantor Rotations" preset (#234) uses
    // translate(-5.77, …) — that offset is MB3D CantorIFS's −(Scale−1)·CScale,
    // i.e. a value the formula's own algebra produces, not a stray. It sat
    // outside the slider range, so clamping op values to [min,max] would have
    // rewritten a parity preset. Widened to ±6 to cover it (the share codec
    // quantizes on a FIXED 0.01 grid, not on this range, so no link re-quantizes).
    params: [
      {
        name: "TransX",
        default: 0.0,
        min: -6.0,
        max: 6.0,
        step: 0.01,
        type: "double",
        tip: "Left/right shift.",
      },
      {
        name: "TransY",
        default: 0.0,
        min: -6.0,
        max: 6.0,
        step: 0.01,
        type: "double",
        tip: "Up/down shift.",
      },
      {
        name: "TransZ",
        default: 0.0,
        min: -6.0,
        max: 6.0,
        step: 0.01,
        type: "double",
        tip: "Nearer/farther shift.",
      },
    ],
    wgsl: `
        pos = pos + vec3f(op.p0, op.p1, op.p2);`,
    // perturbation twin (PERTURBATION_ZOOM_IMPL.md PR-2): the exact delta
    // map — mutates (ptD: vec3f, w: f32) given the reference records
    // pv0..pv3 (core/perturb.js layout). Plain f32 — no EFTs, no launder.
    wgslPt: `
        // constant-affine: the constant cancels exactly — δ unchanged`,
    // deep zoom P4 (DEEP_ZOOM_DF64.md): df64 twin — mutates (P: Df3, w).
    wgslDf: `
        P = df3_add_f32(P, vec3f(op.p0, op.p1, op.p2));
`,
    glsl: (v) => `
    // constant offset (translation: |Jacobian| = 1, w untouched)
    pos += vec3(${v[0]}, ${v[1]}, ${v[2]});`,
  },
  {
    id: 8,
    key: "rotateXZ",
    name: "Rotate XZ",
    wRule: W_UNCHANGED,
    category: "move",
    blurb:
      "Spins the shape around the up axis each step, turning it like a turntable.",
    params: [
      {
        name: "AngleXZ",
        default: 0.0,
        min: -180.0,
        max: 180.0,
        step: 0.5,
        type: "angle",
        tip: "Degrees of turn per step.",
      },
    ],
    wgsl: `
        let a = radians(op.p0);
        let ca = cos(a); let sa = sin(a);
        let nx = pos.x * ca - pos.z * sa;
        let nz = pos.x * sa + pos.z * ca;
        pos.x = nx; pos.z = nz;`,
    // perturbation twin (PERTURBATION_ZOOM_IMPL.md PR-2): the exact delta
    // map — mutates (ptD: vec3f, w: f32) given the reference records
    // pv0..pv3 (core/perturb.js layout). Plain f32 — no EFTs, no launder.
    wgslPt: `
        let a = radians(op.p0);
        let ca = cos(a); let sa = sin(a);
        let nx = ptD.x * ca - ptD.z * sa;
        let nz = ptD.x * sa + ptD.z * ca;
        ptD.x = nx; ptD.z = nz;`,
    // deep zoom P4 (DEEP_ZOOM_DF64.md): df64 twin — mutates (P: Df3, w).
    wgslDf: `
        let a = radians(op.p0);
        let ca = cos(a); let sa = sin(a);
        let rx = df3_get(P, 0); let rz = df3_get(P, 2);
        let nx = df_sub(df_mul_f32(rx, ca), df_mul_f32(rz, sa));
        let nz = df_add(df_mul_f32(rx, sa), df_mul_f32(rz, ca));
        P = Df3(vec3f(nx.x, P.hi.y, nz.x), vec3f(nx.y, P.lo.y, nz.y));
`,
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
    id: 9,
    key: "mengerFold",
    name: "Menger Fold",
    wRule: W_UNCHANGED,
    category: "symmetry",
    blurb:
      "Sorts the three axes so the biggest always comes first — the folding trick behind the Menger sponge lattice. Pair with an abs fold.",
    params: [],
    wgsl: `
        if (pos.x < pos.y) { let t = pos.x; pos.x = pos.y; pos.y = t; }
        if (pos.x < pos.z) { let t = pos.x; pos.x = pos.z; pos.z = t; }
        if (pos.y < pos.z) { let t = pos.y; pos.y = pos.z; pos.z = t; }`,
    // deep zoom P4 (DEEP_ZOOM_DF64.md): df64 twin — mutates (P: Df3, w).
    wgslDf: `
        var mx = df3_get(P, 0); var my = df3_get(P, 1); var mz = df3_get(P, 2);
        if (df_lt(mx, my)) { let t = mx; mx = my; my = t; }
        if (df_lt(mx, mz)) { let t = mx; mx = mz; mz = t; }
        if (df_lt(my, mz)) { let t = my; my = mz; mz = t; }
        P = Df3(vec3f(mx.x, my.x, mz.x), vec3f(mx.y, my.y, mz.y));
`,
    glsl: () => `
    // menger sort fold — order components descending (permutation, w untouched).
    // Pair with an abs fold to sort by magnitude (the Menger sponge recipe).
    if (pos.x < pos.y) { float t = pos.x; pos.x = pos.y; pos.y = t; }
    if (pos.x < pos.z) { float t = pos.x; pos.x = pos.z; pos.z = t; }
    if (pos.y < pos.z) { float t = pos.y; pos.y = pos.z; pos.z = t; }`,
  },
  {
    id: 10,
    key: "sierpinskiFold",
    name: "Sierpinski Fold",
    wRule: W_UNCHANGED,
    category: "symmetry",
    blurb:
      "Mirrors space across three diagonal planes — the fold that grows a Sierpinski tetrahedron. Pair with scale and translate.",
    params: [],
    wgsl: `
        if (pos.x + pos.y < 0.0) { let t = -pos.y; pos.y = -pos.x; pos.x = t; }
        if (pos.x + pos.z < 0.0) { let t = -pos.z; pos.z = -pos.x; pos.x = t; }
        if (pos.y + pos.z < 0.0) { let t = -pos.z; pos.z = -pos.y; pos.y = t; }`,
    // deep zoom P4 (DEEP_ZOOM_DF64.md): df64 twin — mutates (P: Df3, w).
    wgslDf: `
        var sx = df3_get(P, 0); var sy = df3_get(P, 1); var sz = df3_get(P, 2);
        if (df_lt(df_add(sx, sy), vec2f(0.0, 0.0))) { let t = df_neg(sy); sy = df_neg(sx); sx = t; }
        if (df_lt(df_add(sx, sz), vec2f(0.0, 0.0))) { let t = df_neg(sz); sz = df_neg(sx); sx = t; }
        if (df_lt(df_add(sy, sz), vec2f(0.0, 0.0))) { let t = df_neg(sz); sz = df_neg(sy); sy = t; }
        P = Df3(vec3f(sx.x, sy.x, sz.x), vec3f(sx.y, sy.y, sz.y));
`,
    glsl: () => `
    // sierpinski tetrahedral fold — reflect across x+y, x+z, y+z planes
    // (reflections: w untouched). Pair with scale ×2 + translate for the IFS.
    if (pos.x + pos.y < 0.0) { float t = -pos.y; pos.y = -pos.x; pos.x = t; }
    if (pos.x + pos.z < 0.0) { float t = -pos.z; pos.z = -pos.x; pos.x = t; }
    if (pos.y + pos.z < 0.0) { float t = -pos.z; pos.z = -pos.y; pos.y = t; }`,
  },
  {
    id: 11,
    key: "zFold",
    name: "Z Fold",
    wRule: W_UNCHANGED,
    category: "fold",
    blurb:
      "Snaps anything above a height back down by a fixed step — the vertical wrap that closes up a Menger sponge.",
    params: [
      {
        name: "Threshold",
        default: 1.0,
        min: -3.0,
        max: 3.0,
        step: 0.01,
        type: "double",
        tip: "Height where the fold kicks in.",
      },
      {
        name: "Shift",
        default: 2.0,
        min: -4.0,
        max: 4.0,
        step: 0.01,
        type: "double",
        tip: "How far it drops the folded material.",
      },
    ],
    wgsl: `
        if (pos.z > op.p0) { pos.z = pos.z - op.p1; }`,
    // deep zoom P4 (DEEP_ZOOM_DF64.md): df64 twin — mutates (P: Df3, w).
    // Strict compare matches the f32 body: pos.z > p0 ⟺ df_lt((p0,0), z).
    wgslDf: `
        let zz = df3_get(P, 2);
        if (df_lt(vec2f(op.p0, 0.0), zz)) {
          let nz = df_add_f32(zz, -op.p1);
          P = Df3(vec3f(P.hi.x, P.hi.y, nz.x), vec3f(P.lo.x, P.lo.y, nz.y));
        }
`,
    glsl: (v) => `
    // conditional z fold — the Menger sponge z-wrap (translation: w untouched)
    if (pos.z > ${v[0]}) pos.z -= ${v[1]};`,
  },
  {
    id: 12,
    key: "planeFold",
    name: "Plane Fold",
    wRule: W_UNCHANGED,
    category: "fold",
    blurb:
      "Mirrors everything on the back side of a tilted plane to the front — the general-purpose fold you can aim in any direction.",
    params: [
      {
        name: "NormalX",
        default: 1.0,
        min: -1.0,
        max: 1.0,
        step: 0.01,
        type: "double",
        tip: "Direction the mirror plane faces. These three together aim the fold.",
      },
      {
        name: "NormalY",
        default: 1.0,
        min: -1.0,
        max: 1.0,
        step: 0.01,
        type: "double",
        tip: "Direction the mirror plane faces. These three together aim the fold.",
      },
      {
        name: "NormalZ",
        default: 0.0,
        min: -1.0,
        max: 1.0,
        step: 0.01,
        type: "double",
        tip: "Direction the mirror plane faces. These three together aim the fold.",
      },
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
    // deep zoom P4 (DEEP_ZOOM_DF64.md): df64 twin — mutates (P: Df3, w).
    wgslDf: `
        var nv = vec3f(op.p0, op.p1, op.p2);
        if (dot(nv, nv) < 1e-12) { nv = vec3f(1.0, 0.0, 0.0); }
        let n = normalize(nv);
        let d = df3_dot(P, Df3(n, vec3f(0.0)));
        if (df_lt(d, vec2f(0.0, 0.0))) {
          let d2 = df_mul_f32(d, 2.0);
          let px = df_sub(df3_get(P, 0), df_mul_f32(d2, n.x));
          let py = df_sub(df3_get(P, 1), df_mul_f32(d2, n.y));
          let pz = df_sub(df3_get(P, 2), df_mul_f32(d2, n.z));
          P = Df3(vec3f(px.x, py.x, pz.x), vec3f(px.y, py.y, pz.y));
        }
`,
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
    id: 14,
    key: "mandelbulbPower",
    name: "Mandelbulb Power",
    wRule: W_BULB,
    category: "power",
    blurb:
      "Raises the point to a power in spherical coordinates — the classic Mandelbulb move that grows organic bulbs and whorls. Switches the formula into escape-time mode.",
    params: [
      {
        name: "Power",
        default: 8.0,
        min: 2.0,
        max: 16.0,
        step: 0.1,
        type: "double",
        tip: "How many lobes. 8 is the iconic Mandelbulb; higher = more, tighter bulbs.",
      },
    ],
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
    id: 13,
    key: "sphereInv",
    name: "Sphere Inversion",
    wRule: W_MUL_K,
    category: "sphere",
    blurb:
      "Turns space inside-out through a sphere, swapping near and far — the inversion at the heart of bubbly Kleinian shapes. Needs a bounding fold alongside it.",
    params: [
      {
        name: "Radius",
        default: 1.0,
        min: 0.1,
        max: 3.0,
        step: 0.01,
        type: "double",
        tip: "Size of the sphere everything reflects through.",
      },
    ],
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
    // perturbation twin (PERTURBATION_ZOOM_IMPL.md PR-2): the exact delta
    // map — mutates (ptD: vec3f, w: f32) given the reference records
    // pv0..pv3 (core/perturb.js layout). Plain f32 — no EFTs, no launder.
    wgslPt: `
        let num = op.p0 * op.p0;
        let KC = num * 1e6;
        let D = pv1.xyz; let R2 = pv1.w; let mc = pv2.w;
        let q = 2.0 * dot(D, ptD) + dot(ptD, ptD);
        let dds = R2 + q;
        let refCl = mc < 0.0; let sCl = mc + q < 0.0;
        var kr = pv0.w; var dk = 0.0;
        if (!refCl && !sCl) { dk = -(num * q) / (dds * R2); }
        else if (refCl && sCl) { kr = KC; }
        else if (!refCl && sCl) { dk = (num * mc) / (1e-6 * R2); }
        else { kr = KC; dk = -(num * (mc + q)) / (1e-6 * dds); }
        ptD = kr * ptD + dk * (D + ptD);
        w = w * (kr + dk);`,
    // NO wgslDf: the Tier B₁ df64 twin was dropped (PR #422 post-mortem).
    // The twin's algebra was verified correct; what failed is structural —
    // λ̂ = 0 demands full-loop df64 (K_STAR_MAX truncation = measured
    // corruption) at ~3× f32 frame cost with no renderpolicy governor
    // (= the GPU-crash report). Ineligibility (the pre-4b state) is the
    // shipped behavior until the 4b re-land issue lands both prerequisites.
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
    id: 15,
    key: "surfFold",
    name: "Surf Fold",
    wRule: W_UNCHANGED,
    category: "fold",
    blurb:
      "Box-folds only the width and depth and leaves height free, thinning the solid into curved sheets (the 'Amazing Surf' look).",
    params: [
      {
        // max was 3.0 until #538. The shipped "Surf Coral" preset (#116) ends on
        // surfFold(5) — outside its OWN slider range, so the preset could not be
        // reproduced by hand and, once sanitize started clamping op values to
        // [min,max], would have been silently rewritten to 3. The preset is the
        // evidence that the range was too narrow, so the range moved, not the art.
        name: "FoldLimit",
        default: 1.0,
        min: 0.1,
        max: 5.0,
        step: 0.01,
        type: "double",
        tip: "Where the two walls sit; lower = thinner, more sheets.",
      },
    ],
    // Amazing Surf box fold: fold X and Y only, leaving Z free. The unfolded Z
    // axis turns the Mandelbox's solid into thin sheets / surfaces.
    wgsl: `
        let sl = op.p0;
        pos.x = abs(pos.x + sl) - abs(pos.x - sl) - pos.x;
        pos.y = abs(pos.y + sl) - abs(pos.y - sl) - pos.y;`,
    // perturbation twin (PERTURBATION_ZOOM_IMPL.md PR-2): the exact delta
    // map — mutates (ptD: vec3f, w: f32) given the reference records
    // pv0..pv3 (core/perturb.js layout). Plain f32 — no EFTs, no launder.
    wgslPt: `
        let sl = op.p0;
        for (var a = 0u; a < 2u; a = a + 1u) {
          let m1 = pv1[a]; let m2 = pv2[a]; let dx = ptD[a];
          var refB = 1; if (m1 < 0.0) { refB = 2; } else if (m2 < 0.0) { refB = 0; }
          var sB = 1; if (dx > m1) { sB = 2; } else if (dx < -m2) { sB = 0; }
          if (refB == 1 && sB == 1) { ptD[a] = dx; }
          else if (refB == sB) { ptD[a] = -dx; }
          else if (refB == 1 && sB == 2) { ptD[a] = 2.0 * m1 - dx; }
          else if (refB == 2 && sB == 1) { ptD[a] = dx - 2.0 * m1; }
          else if (refB == 1 && sB == 0) { ptD[a] = -2.0 * m2 - dx; }
          else if (refB == 0 && sB == 1) { ptD[a] = dx + 2.0 * m2; }
          else if (refB == 2 && sB == 0) { ptD[a] = -4.0 * sl - dx; }
          else { ptD[a] = 4.0 * sl - dx; }
        }`,
    // deep zoom P4 (DEEP_ZOOM_DF64.md): df64 twin — mutates (P: Df3, w).
    wgslDf: `
        let sl = op.p0;
        let fx = df3_get(P, 0); let fy = df3_get(P, 1);
        let nx = df_sub(df_sub(df_abs(df_add_f32(fx, sl)), df_abs(df_add_f32(fx, -sl))), fx);
        let ny = df_sub(df_sub(df_abs(df_add_f32(fy, sl)), df_abs(df_add_f32(fy, -sl))), fy);
        P = Df3(vec3f(nx.x, ny.x, P.hi.z), vec3f(nx.y, ny.y, P.lo.z));
`,
    glsl: (v) => `
    // Amazing Surf fold — box fold on X,Y only (Z free): builds sheets/surfaces.
    pos.x = abs(pos.x + ${v[0]}) - abs(pos.x - ${v[0]}) - pos.x;
    pos.y = abs(pos.y + ${v[0]}) - abs(pos.y - ${v[0]}) - pos.y;`,
  },
  {
    id: 16,
    key: "octaFold",
    name: "Octahedral Fold",
    wRule: W_UNCHANGED,
    category: "fold",
    blurb:
      "Folds space into an eight-sided wedge for octahedral symmetry. Alone it just mirrors — pair with scale and translate to grow a gasket.",
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
    // deep zoom P4 (DEEP_ZOOM_DF64.md): df64 twin — mutates (P: Df3, w).
    wgslDf: `
        var ox = df_abs(df3_get(P, 0)); var oy = df_abs(df3_get(P, 1)); var oz = df_abs(df3_get(P, 2));
        if (df_lt(ox, oy)) { let t = ox; ox = oy; oy = t; }
        if (df_lt(ox, oz)) { let t = ox; ox = oz; oz = t; }
        if (df_lt(oy, oz)) { let t = oy; oy = oz; oz = t; }
        P = Df3(vec3f(ox.x, oy.x, oz.x), vec3f(ox.y, oy.y, oz.y));
`,
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
    id: 17,
    key: "modFold",
    name: "Mod Fold (Tile)",
    wRule: W_UNCHANGED,
    seam: true,
    category: "symmetry",
    blurb:
      "Tiles space into a repeating grid so one shape becomes an endless lattice of copies. Set a cell to 0 to leave that axis alone; small cells merge neighbouring copies.",
    params: [
      {
        name: "CellX",
        default: 4.0,
        min: 0.0,
        max: 8.0,
        step: 0.05,
        type: "double",
        tip: "Spacing of the copies along X (0 turns this axis off).",
      },
      {
        name: "CellY",
        default: 4.0,
        min: 0.0,
        max: 8.0,
        step: 0.05,
        type: "double",
        tip: "Spacing of the copies along Y (0 turns this axis off).",
      },
      {
        name: "CellZ",
        default: 0.0,
        min: 0.0,
        max: 8.0,
        step: 0.05,
        type: "double",
        tip: "Spacing of the copies along Z (0 turns this axis off).",
      },
    ],
    // Domain repetition: wrap each axis into a cell, tiling space into an
    // infinite lattice of copies (a cell size of 0 leaves that axis alone).
    // Per-cell translation — an isometry on each cell, but DISCONTINUOUS at the
    // cell walls, so this op is a seam declarer (#633; see the seam-clamped DE
    // contract above hingeFold, id 67).
    //
    // Seam term, DERIVED from this op's own wrap convention (NOT from any
    // external formulation): the wrap t' = t − c·floor(t/c + 0.5) writes
    // t = c·n + t' with t' ∈ [−c/2, c/2); the tear set is the wall lattice
    // t = c·(k + ½), and the walls of t's own cell sit at c·n ± c/2, i.e. at
    // distances c/2 − t' and c/2 + t' — so dist(t, walls) = c/2 − |t'|,
    // computable from the POST-wrap coordinate. Multi-axis: the tear set is
    // the union of the active axes' wall planes; the distance to a union is
    // the min. An axis with cell 0 is never wrapped and contributes no tear.
    // The pullback /max(|w|,1e-9) converts wall distance from this op's input
    // space to formula space (w is untouched by the fold itself: W_UNCHANGED).
    //
    // Under the seam clamp the marcher can no longer step THROUGH a wall, so
    // the old "keep cells bigger than the shape" caveat is gone: a body that
    // crosses its cell renders as the true union of copies (the neighbouring
    // copy is reached by marching across the wall, which the clamp paces).
    // What remains is the once-per-iteration density note (oplist.js ⚠ block):
    // re-folding every pass tiles the ITERATION, not the finished body.
    wgsl: `
        if (op.p0 > 0.0) { pos.x = pos.x - op.p0 * floor(pos.x / op.p0 + 0.5); }
        if (op.p1 > 0.0) { pos.y = pos.y - op.p1 * floor(pos.y / op.p1 + 0.5); }
        if (op.p2 > 0.0) { pos.z = pos.z - op.p2 * floor(pos.z / op.p2 + 0.5); }`,
    // #633 seam report — appended to `wgsl` ONLY in seam-armed variants
    // (buildWGSL splices `wgsl + wgslSeam` when seamUsed). A seam-free build
    // (the browse GENERAL variant with no seam formula on it) emits the fold
    // alone — the pre-#633 body byte for byte — so a mis-routed formula
    // degrades to the pre-#633 march, never to a dropped op.
    wgslSeam: `
        var mfs = 1e30;
        if (op.p0 > 0.0) { mfs = min(mfs, 0.5 * op.p0 - abs(pos.x)); }
        if (op.p1 > 0.0) { mfs = min(mfs, 0.5 * op.p1 - abs(pos.y)); }
        if (op.p2 > 0.0) { mfs = min(mfs, 0.5 * op.p2 - abs(pos.z)); }
        if (mfs < 1e30) { seamOut = min(seamOut, max(mfs, 0.0) / max(abs(w), 1e-9)); }`,
    glsl: (v) => `
    // mod fold — domain repetition / tiling (cell of 0 = axis off). Per-cell
    // translation; discontinuous at the walls, so it reports the wall distance
    // into the seam channel (#633) and the march step is clamped by it.
    if (${v[0]} > 0.0) pos.x -= ${v[0]} * floor(pos.x / ${v[0]} + 0.5);
    if (${v[1]} > 0.0) pos.y -= ${v[1]} * floor(pos.y / ${v[1]} + 0.5);
    if (${v[2]} > 0.0) pos.z -= ${v[2]} * floor(pos.z / ${v[2]} + 0.5);
    {
        float mfs = 1e30;
        if (${v[0]} > 0.0) mfs = min(mfs, 0.5 * ${v[0]} - abs(pos.x));
        if (${v[1]} > 0.0) mfs = min(mfs, 0.5 * ${v[1]} - abs(pos.y));
        if (${v[2]} > 0.0) mfs = min(mfs, 0.5 * ${v[2]} - abs(pos.z));
        if (mfs < 1e30) g_seam = min(g_seam, max(mfs, 0.0) / max(abs(w), 1e-9));
    }`,
    // Desktop iterateJIT_ ABI has no seam channel (and no g_seam global), so
    // the export emits the pre-#633 body byte-identically. The standing
    // desktop caveat therefore REMAINS "keep cells bigger than the shape"
    // there — the seam clamp is an engine-marcher feature, not part of the
    // exported formula text.
    desktopGlsl: (v) => `
    // mod fold — domain repetition / tiling (cell of 0 = axis off). Per-cell
    // translation; keep the cell larger than the body or copies overlap.
    if (${v[0]} > 0.0) pos.x -= ${v[0]} * floor(pos.x / ${v[0]} + 0.5);
    if (${v[1]} > 0.0) pos.y -= ${v[1]} * floor(pos.y / ${v[1]} + 0.5);
    if (${v[2]} > 0.0) pos.z -= ${v[2]} * floor(pos.z / ${v[2]} + 0.5);`,
  },
  {
    id: 18,
    key: "boxFoldXYZ",
    name: "Box Fold XYZ",
    wRule: W_UNCHANGED,
    category: "fold",
    blurb:
      "Box fold with a separate wall distance per axis, for stretched or slab-shaped boxes instead of a uniform cube.",
    params: [
      {
        name: "LimitX",
        default: 1.0,
        min: 0.1,
        max: 3.0,
        step: 0.01,
        type: "double",
        tip: "Wall distance on X; make the three unequal for stretched boxes.",
      },
      {
        name: "LimitY",
        default: 1.0,
        min: 0.1,
        max: 3.0,
        step: 0.01,
        type: "double",
        tip: "Wall distance on Y; make the three unequal for stretched boxes.",
      },
      {
        name: "LimitZ",
        default: 1.0,
        min: 0.1,
        max: 3.0,
        step: 0.01,
        type: "double",
        tip: "Wall distance on Z; make the three unequal for stretched boxes.",
      },
    ],
    // Box fold with an independent limit per axis → anisotropic (stretched /
    // slab) boxes instead of the uniform cube. Equals Box Fold when X=Y=Z.
    wgsl: `
        pos.x = abs(pos.x + op.p0) - abs(pos.x - op.p0) - pos.x;
        pos.y = abs(pos.y + op.p1) - abs(pos.y - op.p1) - pos.y;
        pos.z = abs(pos.z + op.p2) - abs(pos.z - op.p2) - pos.z;`,
    // perturbation twin (PERTURBATION_ZOOM_IMPL.md PR-2): the exact delta
    // map — mutates (ptD: vec3f, w: f32) given the reference records
    // pv0..pv3 (core/perturb.js layout). Plain f32 — no EFTs, no launder.
    wgslPt: `
        let fv = vec3f(op.p0, op.p1, op.p2);
        for (var a = 0u; a < 3u; a = a + 1u) {
          let m1 = pv1[a]; let m2 = pv2[a]; let dx = ptD[a];
          var refB = 1; if (m1 < 0.0) { refB = 2; } else if (m2 < 0.0) { refB = 0; }
          var sB = 1; if (dx > m1) { sB = 2; } else if (dx < -m2) { sB = 0; }
          if (refB == 1 && sB == 1) { ptD[a] = dx; }
          else if (refB == sB) { ptD[a] = -dx; }
          else if (refB == 1 && sB == 2) { ptD[a] = 2.0 * m1 - dx; }
          else if (refB == 2 && sB == 1) { ptD[a] = dx - 2.0 * m1; }
          else if (refB == 1 && sB == 0) { ptD[a] = -2.0 * m2 - dx; }
          else if (refB == 0 && sB == 1) { ptD[a] = dx + 2.0 * m2; }
          else if (refB == 2 && sB == 0) { ptD[a] = -4.0 * fv[a] - dx; }
          else { ptD[a] = 4.0 * fv[a] - dx; }
        }`,
    // deep zoom P4 (DEEP_ZOOM_DF64.md): df64 twin — mutates (P: Df3, w).
    wgslDf: `
        let bx = df3_get(P, 0); let by = df3_get(P, 1); let bz = df3_get(P, 2);
        let nx = df_sub(df_sub(df_abs(df_add_f32(bx, op.p0)), df_abs(df_add_f32(bx, -op.p0))), bx);
        let ny = df_sub(df_sub(df_abs(df_add_f32(by, op.p1)), df_abs(df_add_f32(by, -op.p1))), by);
        let nz = df_sub(df_sub(df_abs(df_add_f32(bz, op.p2)), df_abs(df_add_f32(bz, -op.p2))), bz);
        P = Df3(vec3f(nx.x, ny.x, nz.x), vec3f(nx.y, ny.y, nz.y));
`,
    glsl: (v) => `
    // per-axis box fold (anisotropic): independent fold limit on X, Y, Z.
    pos.x = abs(pos.x + ${v[0]}) - abs(pos.x - ${v[0]}) - pos.x;
    pos.y = abs(pos.y + ${v[1]}) - abs(pos.y - ${v[1]}) - pos.y;
    pos.z = abs(pos.z + ${v[2]}) - abs(pos.z - ${v[2]}) - pos.z;`,
  },
  {
    id: 19,
    key: "absOffsetFold",
    name: "Abs Fold (offset)",
    wRule: W_UNCHANGED,
    category: "fold",
    blurb:
      "Abs fold whose mirror is shifted off-center, breaking the tidy symmetry for lopsided, more organic folds.",
    params: [
      {
        name: "OffsetX",
        default: 0.0,
        min: -2.0,
        max: 2.0,
        step: 0.01,
        type: "double",
        tip: "How far to slide the mirror off-center on X.",
      },
      {
        name: "OffsetY",
        default: 0.0,
        min: -2.0,
        max: 2.0,
        step: 0.01,
        type: "double",
        tip: "How far to slide the mirror off-center on Y.",
      },
      {
        name: "OffsetZ",
        default: 0.0,
        min: -2.0,
        max: 2.0,
        step: 0.01,
        type: "double",
        tip: "How far to slide the mirror off-center on Z.",
      },
    ],
    // Abs fold across SHIFTED planes: reflect across x = -OffsetX (etc.) instead
    // of the origin. Off-centre mirrors break the symmetry of a plain abs fold.
    // Equals Abs Fold at offset 0. (Reflection + shift: |Jacobian| = 1.)
    wgsl: `
        pos = abs(pos + vec3f(op.p0, op.p1, op.p2)) - vec3f(op.p0, op.p1, op.p2);`,
    // deep zoom P4 (DEEP_ZOOM_DF64.md): df64 twin — mutates (P: Df3, w).
    wgslDf: `
        let off = vec3f(op.p0, op.p1, op.p2);
        let q = df3_add_f32(P, off);
        let qx = df_abs(df3_get(q, 0)); let qy = df_abs(df3_get(q, 1)); let qz = df_abs(df3_get(q, 2));
        P = df3_add_f32(Df3(vec3f(qx.x, qy.x, qz.x), vec3f(qx.y, qy.y, qz.y)), -off);
`,
    glsl: (v) => `
    // offset abs fold — mirror across planes shifted by the offset (reflection).
    pos = abs(pos + vec3(${v[0]}, ${v[1]}, ${v[2]})) - vec3(${v[0]}, ${v[1]}, ${v[2]});`,
  },
  {
    id: 20,
    key: "tentFold",
    name: "Tent Fold",
    wRule: W_UNCHANGED,
    category: "fold",
    blurb:
      "Mirrors space into repeating ridges along each axis, like an endless row of tents. Set a period to 0 to skip that axis.",
    params: [
      {
        name: "PeriodX",
        default: 0.0,
        min: 0.0,
        max: 8.0,
        step: 0.05,
        type: "double",
        tip: "Spacing of the ridges along X (0 turns this axis off).",
      },
      {
        name: "PeriodY",
        default: 0.0,
        min: 0.0,
        max: 8.0,
        step: 0.05,
        type: "double",
        tip: "Spacing of the ridges along Y (0 turns this axis off).",
      },
      {
        name: "PeriodZ",
        default: 0.0,
        min: 0.0,
        max: 8.0,
        step: 0.05,
        type: "double",
        tip: "Spacing of the ridges along Z (0 turns this axis off).",
      },
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
    id: 21,
    key: "rotateXYZ",
    name: "Rotate XYZ",
    wRule: W_UNCHANGED,
    category: "move",
    blurb:
      "Rotates the shape freely around all three axes at once, in one combined step.",
    params: [
      {
        name: "AngleXY",
        default: 0.0,
        min: -180.0,
        max: 180.0,
        step: 0.5,
        type: "angle",
        tip: "Flat spin amount.",
      },
      {
        name: "AngleYZ",
        default: 0.0,
        min: -180.0,
        max: 180.0,
        step: 0.5,
        type: "angle",
        tip: "Forward/back tilt amount.",
      },
      {
        name: "AngleXZ",
        default: 0.0,
        min: -180.0,
        max: 180.0,
        step: 0.5,
        type: "angle",
        tip: "Turntable turn amount.",
      },
    ],
    // One Euler rotation = the three plane rotations in sequence (about Z, then
    // X, then Y). Orthogonal: |Jacobian| = 1, w untouched.
    wgsl: `
        let ra = radians(op.p0); let rb = radians(op.p1); let rd = radians(op.p2);
        { let ca = cos(ra); let sa = sin(ra); let nx = pos.x*ca - pos.y*sa; let ny = pos.x*sa + pos.y*ca; pos.x = nx; pos.y = ny; }
        { let ca = cos(rb); let sa = sin(rb); let ny = pos.y*ca - pos.z*sa; let nz = pos.y*sa + pos.z*ca; pos.y = ny; pos.z = nz; }
        { let ca = cos(rd); let sa = sin(rd); let nx = pos.x*ca - pos.z*sa; let nz = pos.x*sa + pos.z*ca; pos.x = nx; pos.z = nz; }`,
    // perturbation twin (PERTURBATION_ZOOM_IMPL.md PR-2): the exact delta
    // map — mutates (ptD: vec3f, w: f32) given the reference records
    // pv0..pv3 (core/perturb.js layout). Plain f32 — no EFTs, no launder.
    wgslPt: `
        let ra = radians(op.p0); let rb = radians(op.p1); let rd = radians(op.p2);
        { let ca = cos(ra); let sa = sin(ra); let nx = ptD.x*ca - ptD.y*sa; let ny = ptD.x*sa + ptD.y*ca; ptD.x = nx; ptD.y = ny; }
        { let ca = cos(rb); let sa = sin(rb); let ny = ptD.y*ca - ptD.z*sa; let nz = ptD.y*sa + ptD.z*ca; ptD.y = ny; ptD.z = nz; }
        { let ca = cos(rd); let sa = sin(rd); let nx = ptD.x*ca - ptD.z*sa; let nz = ptD.x*sa + ptD.z*ca; ptD.x = nx; ptD.z = nz; }`,
    // deep zoom P4 (DEEP_ZOOM_DF64.md): df64 twin — mutates (P: Df3, w).
    wgslDf: `
        let ra = radians(op.p0); let rb = radians(op.p1); let rd = radians(op.p2);
        {
          let ca = cos(ra); let sa = sin(ra);
          let x = df3_get(P, 0); let y = df3_get(P, 1);
          let nx = df_sub(df_mul_f32(x, ca), df_mul_f32(y, sa));
          let ny = df_add(df_mul_f32(x, sa), df_mul_f32(y, ca));
          P = Df3(vec3f(nx.x, ny.x, P.hi.z), vec3f(nx.y, ny.y, P.lo.z));
        }
        {
          let ca = cos(rb); let sa = sin(rb);
          let y = df3_get(P, 1); let z = df3_get(P, 2);
          let ny = df_sub(df_mul_f32(y, ca), df_mul_f32(z, sa));
          let nz = df_add(df_mul_f32(y, sa), df_mul_f32(z, ca));
          P = Df3(vec3f(P.hi.x, ny.x, nz.x), vec3f(P.lo.x, ny.y, nz.y));
        }
        {
          let ca = cos(rd); let sa = sin(rd);
          let x = df3_get(P, 0); let z = df3_get(P, 2);
          let nx = df_sub(df_mul_f32(x, ca), df_mul_f32(z, sa));
          let nz = df_add(df_mul_f32(x, sa), df_mul_f32(z, ca));
          P = Df3(vec3f(nx.x, P.hi.y, nz.x), vec3f(nx.y, P.lo.y, nz.y));
        }
`,
    glsl: (v) => `
    // Euler rotation (XY, then YZ, then XZ) — orthogonal: w untouched.
    { float ca = cos(${v[0]}), sa = sin(${v[0]}); float nx = pos.x*ca - pos.y*sa, ny = pos.x*sa + pos.y*ca; pos.x = nx; pos.y = ny; }
    { float ca = cos(${v[1]}), sa = sin(${v[1]}); float ny = pos.y*ca - pos.z*sa, nz = pos.y*sa + pos.z*ca; pos.y = ny; pos.z = nz; }
    { float ca = cos(${v[2]}), sa = sin(${v[2]}); float nx = pos.x*ca - pos.z*sa, nz = pos.x*sa + pos.z*ca; pos.x = nx; pos.z = nz; }`,
  },
  {
    id: 22,
    key: "twist",
    name: "Twist",
    wRule: W_UNCHANGED,
    deApprox: true,
    category: "move",
    blurb:
      "Rotates the shape more and more the higher up you go, screwing it into a spiral tower. Keep it modest or the surface gets streaky.",
    params: [
      {
        name: "Twist",
        default: 30.0,
        min: -180.0,
        max: 180.0,
        step: 0.5,
        type: "angle",
        tip: "How hard it screws with height. Gentle = a soft lean; strong = a tight corkscrew.",
      },
    ],
    // Rotate XY by an angle proportional to height (z): screws / spiral towers.
    // Volume-preserving (|Jacobian| = 1) but NOT an isometry — and #643 makes
    // that magnitude explicit instead of a warning in prose.
    //
    // DE derivation (the #636 audit's breach). With a = radians(rate) and R(θ)
    // the XY rotation, f(p) = (R(a·z)·p_xy, z), so
    //   J = [[R, a·R·J₉₀·p_xy], [0, 1]],  J₉₀ = [[0,−1],[1,0]].
    // Left-multiplying by the orthogonal diag(Rᵀ, 1) leaves the norm alone, so
    // ‖J‖ = ‖I + s⊗ẑ‖ with the shear vector s = a·(−y, x), |s| = |a|·ρ, ρ the
    // (rotation-invariant) XY radius. That rank-1 shear's largest singular
    // value is the spiralVortex constant on a POSITION-DEPENDENT argument:
    //   L(b) = (b + √(b² + 4)) / 2,   ‖J‖ = L(|a|·ρ)
    // — exact (the audit's sampler attains it at 1.000) and UNBOUNDED in ρ,
    // which is precisely why log-spiral (constant L(a)) is safe where twist is
    // not. Measured breach of the old W_UNCHANGED declaration: 76.3×.
    //
    // WHY deApprox AND NOT `w *= L(|a|·ρ)` (#643, the decision the issue asked
    // for). Charging the true norm is sound and was implemented and RENDERED:
    // L ≈ 2.3 per iteration at the default 30° inside a Mandelbox core, so
    // over 12 iterations w gains ~10⁴ on top of the scale product and DE
    // collapses to ~1e-8. Every ray then exhausts its step budget while still
    // "hugging geometry" and the frame is uniform fog — not slow, UNUSABLE at
    // any budget. That is the case APPROX_DE.md §1 was written for: keep the
    // best-effort w-rule (here: untouched, the MB3D treatment) and tag the op,
    // so isDeSound() stops vouching for it and the policy layer tightens the
    // step ×0.5 with ×2 budget. varyScale (id 32) went the other way in the
    // same issue — its residual is a BOUNDED scalar it can afford to charge.
    // deaudit.test.mjs gates the derivation above as twist's approx-derived
    // ceiling, so the number stays honest even though w does not carry it.
    wgsl: `
        let tw = radians(op.p0) * pos.z;
        let tc = cos(tw); let ts = sin(tw);
        let tx = pos.x * tc - pos.y * ts;
        let ty = pos.x * ts + pos.y * tc;
        pos.x = tx; pos.y = ty;`,
    glsl: (v) => `
    // twist — rotate XY by (rate * z). Volume-preserving, but a z-shear of
    // magnitude |a|*rho: approximate DE (w untouched), so keep Step small.
    {
        float tw = ${v[0]} * pos.z;
        float tc = cos(tw), ts = sin(tw);
        float tx = pos.x * tc - pos.y * ts;
        float ty = pos.x * ts + pos.y * tc;
        pos.x = tx; pos.y = ty;
    }`,
  },
  {
    id: 23,
    key: "quadratic",
    name: "Quadratic (z²)",
    wRule: W_BULB,
    category: "power",
    blurb:
      "Squares the point like the flat Mandelbrot set, sweeping its curling bulbs out into 3D. Escape-time mode.",
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
    id: 24,
    key: "icosaFold",
    name: "Icosahedral Fold",
    wRule: W_UNCHANGED,
    category: "symmetry",
    blurb:
      "Folds space into 20-sided icosahedral symmetry with golden-ratio mirrors, for richly symmetric star gaskets. Pair with scale and translate.",
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
    // deep zoom P4 (DEEP_ZOOM_DF64.md): df64 twin — mutates (P: Df3, w).
    wgslDf: `
        var px = df_abs(df3_get(P, 0)); var py = df_abs(df3_get(P, 1)); var pz = df_abs(df3_get(P, 2));
        {
          let nx = -0.809017; let ny = 0.309017; let nz = 0.5;
          let d = df_add(df_add(df_mul_f32(px, nx), df_mul_f32(py, ny)), df_mul_f32(pz, nz));
          if (df_lt(d, vec2f(0.0, 0.0))) {
            let d2 = df_mul_f32(d, 2.0);
            px = df_sub(px, df_mul_f32(d2, nx)); py = df_sub(py, df_mul_f32(d2, ny)); pz = df_sub(pz, df_mul_f32(d2, nz));
          }
        }
        {
          let nx = 0.5; let ny = -0.809017; let nz = 0.309017;
          let d = df_add(df_add(df_mul_f32(px, nx), df_mul_f32(py, ny)), df_mul_f32(pz, nz));
          if (df_lt(d, vec2f(0.0, 0.0))) {
            let d2 = df_mul_f32(d, 2.0);
            px = df_sub(px, df_mul_f32(d2, nx)); py = df_sub(py, df_mul_f32(d2, ny)); pz = df_sub(pz, df_mul_f32(d2, nz));
          }
        }
        {
          let nx = 0.309017; let ny = 0.5; let nz = -0.809017;
          let d = df_add(df_add(df_mul_f32(px, nx), df_mul_f32(py, ny)), df_mul_f32(pz, nz));
          if (df_lt(d, vec2f(0.0, 0.0))) {
            let d2 = df_mul_f32(d, 2.0);
            px = df_sub(px, df_mul_f32(d2, nx)); py = df_sub(py, df_mul_f32(d2, ny)); pz = df_sub(pz, df_mul_f32(d2, nz));
          }
        }
        P = Df3(vec3f(px.x, py.x, pz.x), vec3f(px.y, py.y, pz.y));
`,
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
    id: 25,
    key: "menger",
    name: "Menger",
    wRule: W_UNCHANGED,
    category: "symmetry",
    blurb:
      "The whole Menger sponge fold in one move (mirror, sort, and wrap), with a smoothness dial to round the edges. Pair with scale 3 and translate.",
    params: [
      {
        name: "Smoothness",
        default: 0.005,
        min: -0.1,
        max: 0.1,
        step: 0.001,
        type: "double",
        tip: "Rounds the edges. 0 is razor-sharp; positive softens; negative rounds a different, flatter way.",
      },
    ],
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
    id: 26,
    key: "polyAngleFold",
    name: "Poly Angle Fold",
    wRule: W_UNCHANGED,
    category: "symmetry",
    blurb:
      "Snaps the shape angle to the nearest of N spokes, rotating rather than mirroring — the seed of polygon and knot patterns. Bound it with a box or sphere fold.",
    params: [
      {
        name: "Symmetry",
        default: 6.0,
        min: 2.0,
        max: 16.0,
        step: 1.0,
        type: "double",
        tip: "Number of spokes/sides.",
      },
      {
        name: "Angle",
        default: 0.0,
        min: -180.0,
        max: 180.0,
        step: 0.5,
        type: "angle",
        tip: "Rotates the whole spoke pattern.",
      },
      {
        name: "Mirror",
        default: 0.0,
        min: 0.0,
        max: 1.0,
        step: 1.0,
        type: "double",
        tip: "Turn on (1) to add a mirror for extra symmetry.",
      },
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
    id: 27,
    key: "cylinderFold",
    name: "Cylinder Fold",
    wRule: W_MUL_K,
    deApprox: true,
    category: "sphere",
    blurb:
      "Sphere fold measured only across width and depth, leaving height free — inflates the shape around a vertical tube.",
    params: [
      {
        name: "MinRadius",
        default: 0.5,
        min: 0.05,
        max: 2.0,
        step: 0.01,
        type: "double",
        tip: "Inner core size.",
      },
      {
        name: "FixedRadius",
        default: 1.0,
        min: 0.1,
        max: 3.0,
        step: 0.01,
        type: "double",
        tip: "Bubble size that does the inflating.",
      },
    ],
    // Sphere fold whose radius is measured in the XY plane only (the Z axis runs
    // free) — the Amazing-Surf "cylinder" ball-fold. w *= k tracks the XY-band
    // factor: the BEST-EFFORT part of the accounting (APPROX_DE.md §1), not the
    // whole of it, which is why this op is deApprox-tagged.
    //
    // DE derivation (#636's breach). Unlike sphereFold the mid band is NOT
    // conformal: the WHOLE vector (z included) is scaled by k(ρ_xy) = F²/ρ²,
    // so with ρ̂ the in-plane radial unit vector
    //   J = k·[I − 2·ρ̂⊗ρ̂ − (2z/ρ)·ẑ⊗ρ̂]
    // whose (ρ̂, ẑ) block is [[−1, 0], [−c, 1]], c = 2|z|/ρ (the θ̂ lane is
    // carried). That block's largest singular value is the shear norm
    // L(c) = (c + √(c²+4))/2, so the true operator norm is k·L(2|z|/ρ) — exact
    // (the audit attains it at 1.000) and UNBOUNDED in |z|/ρ, because this fold
    // is exactly the one that does not bound z. Measured breach of the plain-k
    // declaration: 21.8× at |z|/ρ ≈ 13. sphereFold measures r in 3D, which is
    // what makes ITS mid band a genuine conformal inversion — it stays sound.
    //
    // WHY deApprox AND NOT `w *= k·L(2|z|/ρ)` (#643): the same measurement that
    // settled twist. |z| is unbounded here by construction, so the exact charge
    // compounds without limit; charging it was implemented and rendered, and
    // Cylinder Surf came back as budget-exhausted fog instead of a surface. The
    // tag keeps the honest half of the accounting (k), stops isDeSound() from
    // vouching for the other half, and buys the ×0.5 step / ×2 budget the
    // policy layer applies. The derivation is gated as this op's
    // approx-derived ceiling in deaudit.test.mjs.
    wgsl: `
        let minR2 = op.p0 * op.p0;
        let fixedR2 = op.p1 * op.p1;
        let r2 = pos.x * pos.x + pos.y * pos.y;
        var k = 1.0;
        if (r2 < minR2) { k = fixedR2 / minR2; }
        else if (r2 < fixedR2) { k = fixedR2 / r2; }
        pos = pos * k;
        w = w * k;`,
    // perturbation twin (PERTURBATION_ZOOM_IMPL.md PR-2): the exact delta
    // map — mutates (ptD: vec3f, w: f32) given the reference records
    // pv0..pv3 (core/perturb.js layout). Plain f32 — no EFTs, no launder.
    wgslPt: `
        let minR2 = op.p0 * op.p0;
        let fixedR2 = op.p1 * op.p1;
        let K0 = fixedR2 / minR2;
        let Z = pv0.xyz; let rr2 = pv1.w; let mr1 = pv2.w; let mr2 = pv3.x;
        let q = 2.0 * (Z.x * ptD.x + Z.y * ptD.y) + (ptD.x * ptD.x + ptD.y * ptD.y);
        let rs2 = rr2 + q;
        var refB = 1; if (mr1 < 0.0) { refB = 0; } else if (mr2 <= 0.0) { refB = 2; }
        var sB = 1; if (mr1 + q < 0.0) { sB = 0; } else if (mr2 - q <= 0.0) { sB = 2; }
        var kr = pv0.w; var dk = 0.0;
        if (refB == sB) { if (refB == 1) { dk = -(fixedR2 * q) / (rs2 * rr2); } }
        else if (refB == 0 && sB == 1) { dk = -(fixedR2 * (mr1 + q)) / (rs2 * minR2); }
        else if (refB == 1 && sB == 0) { dk = (fixedR2 * mr1) / (minR2 * rr2); }
        else if (refB == 1 && sB == 2) { dk = -mr2 / rr2; }
        else if (refB == 2 && sB == 1) { dk = (mr2 - q) / rs2; }
        else if (refB == 0 && sB == 2) { dk = 1.0 - K0; }
        else { dk = K0 - 1.0; }
        ptD = kr * ptD + dk * (Z + ptD);
        w = w * (kr + dk);`,
    // deep zoom P4 (DEEP_ZOOM_DF64.md): df64 twin — mutates (P: Df3, w).
    wgslDf: `
        let minR2 = two_prod(op.p0, op.p0);
        let fixedR2 = two_prod(op.p1, op.p1);
        let cx = df3_get(P, 0); let cy = df3_get(P, 1);
        let r2 = df_add(df_mul(cx, cx), df_mul(cy, cy));
        var k = vec2f(1.0, 0.0);
        if (df_lt(r2, minR2)) { k = df_div(fixedR2, minR2); }
        else if (df_lt(r2, fixedR2)) { k = df_div(fixedR2, r2); }
        P = df3_mul(P, k);
        w = w * (k.x + k.y);
`,
    glsl: (v) => `
    // cylinder fold — sphere fold with the radius taken in XY only (Z free).
    // Mid band is NOT conformal (the z lane rides k), so w *= k is a
    // best-effort accounting: approximate DE — keep Step small.
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
    id: 28,
    key: "radialInvert",
    name: "Inversion (shifted)",
    wRule: W_MUL_K,
    category: "sphere",
    blurb:
      "Turns space inside-out through a sphere you can slide off-center — the Kleinian bubble generator. Needs a bounding fold.",
    params: [
      {
        name: "CenterX",
        default: 0.0,
        min: -2.0,
        max: 2.0,
        step: 0.01,
        type: "double",
        tip: "Where the inversion sphere sits; move it off-center for asymmetric bubbles.",
      },
      {
        name: "CenterY",
        default: 0.0,
        min: -2.0,
        max: 2.0,
        step: 0.01,
        type: "double",
        tip: "Where the inversion sphere sits; move it off-center for asymmetric bubbles.",
      },
      {
        name: "CenterZ",
        default: 0.0,
        min: -2.0,
        max: 2.0,
        step: 0.01,
        type: "double",
        tip: "Where the inversion sphere sits; move it off-center for asymmetric bubbles.",
      },
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
    // perturbation twin (PERTURBATION_ZOOM_IMPL.md PR-2): the exact delta
    // map — mutates (ptD: vec3f, w: f32) given the reference records
    // pv0..pv3 (core/perturb.js layout). Plain f32 — no EFTs, no launder.
    wgslPt: `
        let KC = 1e6;
        let D = pv1.xyz; let R2 = pv1.w; let mc = pv2.w;
        let q = 2.0 * dot(D, ptD) + dot(ptD, ptD);
        let dds = R2 + q;
        let refCl = mc < 0.0; let sCl = mc + q < 0.0;
        var kr = pv0.w; var dk = 0.0;
        if (!refCl && !sCl) { dk = -q / (dds * R2); }
        else if (refCl && sCl) { kr = KC; }
        else if (!refCl && sCl) { dk = mc / (1e-6 * R2); }
        else { kr = KC; dk = -(mc + q) / (1e-6 * dds); }
        ptD = kr * ptD + dk * (D + ptD);
        w = w * (kr + dk);`,
    // NO wgslDf: Tier B₁ twin dropped — see the sphereInv note (same
    // post-mortem; this op's λ̂ = 0 + K_STAR_MAX truncation was the measured
    // "corrupt render", and its ungoverned full-loop cost the GPU wedge).
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
    id: 29,
    key: "bulbAxis",
    name: "Bulb Power (axis)",
    wRule: W_BULB,
    category: "power",
    blurb:
      "Mandelbulb power with a choice of which axis the bulbs grow along, which trig flavor shapes them, and separate control of its two spin angles — several bulb styles in one move. Escape-time mode.",
    params: [
      {
        name: "Power",
        default: 8.0,
        min: 2.0,
        max: 16.0,
        step: 0.1,
        type: "double",
        tip: "Number of lobes (8 = classic).",
      },
      {
        name: "Axis",
        default: 0.0,
        min: 0.0,
        max: 2.0,
        step: 1.0,
        type: "double",
        tip: "Which axis is the pole: 0 up, 1 side, 2 front — each gives a different bulb shape.",
      },
      {
        name: "Convention",
        default: 0.0,
        min: 0.0,
        max: 2.0,
        step: 1.0,
        type: "double",
        tip: "Trig flavor of the bulb: 0 classic, 1 swapped (norm-style), 2 latitude (sine bulb) — each reshapes the lobes.",
      },
      {
        name: "ThetaMul",
        default: 1.0,
        min: -4.0,
        max: 4.0,
        step: 0.05,
        type: "double",
        tip: "Stretches the bulbs top-to-bottom, independently of the lobe count. 1 keeps them tied.",
      },
      {
        name: "PhiMul",
        default: 1.0,
        min: -4.0,
        max: 4.0,
        step: 0.05,
        type: "double",
        tip: "Winds the bulbs around, independently of the lobe count. 1 keeps them tied.",
      },
    ],
    // Mandelbulb spherical power z→zⁿ with a SELECTABLE polar axis (0 = z, the
    // White/Nylander default; 1 = y, the IQ convention; 2 = x), a TRIG
    // CONVENTION (0 = cos-polar, the classic; 1 = sin-polar, the NormBulb
    // flavor — swaps sin/cos on the multiplied acos angle; 2 = asin-latitude,
    // the "sine bulb") and, since the trig-bulb wave, INDEPENDENT PER-ANGLE
    // POWERS. Escape-time like Mandelbulb Power: set DEoption 0 + AddC.
    //
    // ── The split-power extension (parity wave 3, OP_PARAM_ENCODING.md PR-2) ──
    // TRIGBULB_SPIKE.md:33-36 wanted axis × convention × Power × ThetaMul ×
    // PhiMul and measured it at "5-6 slots. The ABI has 3." It shipped the
    // convention on the free third slot (:57) and recorded the cross —
    // "convention × split-power on one op" — as an ACCEPTED LOSS (:73-74),
    // with option B ("convention on 37 too") rejected on one line because
    // sphericalTwoStage has no free slot (:58). The opAux overflow lane
    // (OP_PARAM_ENCODING.md) removes the constraint that forced both
    // decisions, so the cross lands here: params 3-4 ride opAux[o].x/.y and
    // the author-facing syntax below is still just `op.p3` / `op.p4`.
    //
    // ENCODING. ThetaMul/PhiMul are MULTIPLIERS on the shared Power, exactly
    // as on sphericalTwoStage (37) and as ruckerBulb's AziPow (62):
    //   r' = r^Power,  θ' = Power·ThetaMul·θ,  φ' = Power·PhiMul·φ
    // That makes ThetaMul = PhiMul = 1 the exact tied-power degeneracy while
    // still reaching ANY absolute angle power. Note the three exponents are
    // then fully independent — the radial one is Power outright, so an
    // exemplar with a separate r-power (IQ-bulb old's Mode branch) is reached
    // by putting the r-power in Power and dividing the angle powers into the
    // multipliers. That is why this op needs NO RadialSel: unlike ruckerBulb,
    // whose radial exponent is tied to one of its two angle powers, nothing
    // here couples the radius to an angle. Five params, not six.
    //
    // Unlike ruckerBulb's AziPow the multipliers are SIGNED (±4, matching 37):
    // a negative multiplier can never reach the radial exponent here, so
    // pow(br, bp) cannot be handed a negative power and blow r→0 up to a
    // non-finite f32 — the reason 62 had to restrict its own.
    //
    // SUBSUMPTION (pinned in core/bulbaxis.test.mjs): at Axis 0 + Convention 0
    // this op is EXACTLY sphericalTwoStage(Power, ThetaMul, PhiMul), and at
    // ThetaMul = PhiMul = 1 it is EXACTLY the pre-wave bulbAxis(Power, Axis,
    // Convention) — hence, at Axis 0, EXACTLY mandelbulbPower(Power). Op 37 is
    // deliberately left alone rather than also growing a Convention: widening
    // ONE op to the full cross beats widening two (OP_PARAM_ENCODING.md R7 —
    // palette size is itself a measured compile cost).
    //
    // W-RULE. Unchanged, and unchanged for the same reason 37's is: an angle
    // multiplier is radius-preserving, so the radial map is still r→r^Power
    // and the analytic dr stays w = Power·r^(Power-1)·w + 1. Convention only
    // re-phases the angle (TRIGBULB_SPIKE.md:77-82). The op stays W_BULB with
    // a w update that does not read p3/p4 at all — asserted, not re-derived,
    // in the 3-emitter parity tests.
    wgsl: `
        let bp = op.p0;
        let m = i32(op.p1 + 0.5);
        let conv = i32(op.p2 + 0.5);
        let btm = op.p3;
        let bpm = op.p4;
        let br = length(pos);
        if (br > 1e-9) {
          var up = pos.z; var a = pos.x; var b = pos.y;
          if (m == 1) { up = pos.y; a = pos.z; b = pos.x; }
          else if (m == 2) { up = pos.x; a = pos.y; b = pos.z; }
          let bu = clamp(up / br, -1.0, 1.0);
          var bang = acos(bu);
          if (conv == 2) { bang = asin(bu); }
          let bth = bang * bp * btm;
          let bph = atan2(b, a) * bp * bpm;
          let brn = pow(br, bp);
          w = bp * brn / br * w + 1.0;
          var bst = sin(bth);
          var nup = brn * cos(bth);
          if (conv != 0) { bst = cos(bth); nup = brn * sin(bth); }
          let na = brn * bst * cos(bph);
          let nb = brn * bst * sin(bph);
          if (m == 1) { pos = vec3f(nb, nup, na); }
          else if (m == 2) { pos = vec3f(nup, na, nb); }
          else { pos = vec3f(na, nb, nup); }
        }`,
    glsl: (v) => `
    // Mandelbulb power, selectable polar axis (0 z · 1 y · 2 x) + trig
    // convention (0 cos · 1 sin-polar · 2 asin-latitude) + independent
    // per-angle power multipliers — escape-time.
    {
        float bp = ${v[0]};
        int m = int(${v[1]} + 0.5);
        int conv = int(${v[2]} + 0.5);
        float btm = ${v[3]};
        float bpm = ${v[4]};
        float br = length(pos);
        if (br > 1e-9) {
            float up = pos.z, a = pos.x, b = pos.y;
            if (m == 1) { up = pos.y; a = pos.z; b = pos.x; }
            else if (m == 2) { up = pos.x; a = pos.y; b = pos.z; }
            float bu = clamp(up / br, -1.0, 1.0);
            float bth = (conv == 2 ? asin(bu) : acos(bu)) * bp * btm;
            float bph = atan(b, a) * bp * bpm;
            float brn = pow(br, bp);
            w = bp * brn / br * w + 1.0;
            float bst = (conv != 0 ? cos(bth) : sin(bth));
            float nup = brn * (conv != 0 ? sin(bth) : cos(bth));
            float na = brn * bst * cos(bph);
            float nb = brn * bst * sin(bph);
            if (m == 1) pos = vec3(nb, nup, na);
            else if (m == 2) pos = vec3(nup, na, nb);
            else pos = vec3(na, nb, nup);
        }
    }`,
  },
  {
    id: 30,
    key: "hexFold",
    name: "Hex Fold",
    wRule: W_UNCHANGED,
    category: "symmetry",
    blurb:
      "Folds the width/depth plane into 60-degree wedges for six-fold honeycomb symmetry, height left free.",
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
    // deep zoom P4 (DEEP_ZOOM_DF64.md): df64 twin — mutates (P: Df3, w).
    wgslDf: `
        let kx = -0.8660254;
        let ky = 0.5;
        var hx = df_abs(df3_get(P, 0));
        var hy = df_abs(df3_get(P, 1));
        let hd = df_add(df_mul_f32(hx, kx), df_mul_f32(hy, ky));
        if (df_lt(hd, vec2f(0.0, 0.0))) {
          let d2 = df_mul_f32(hd, 2.0);
          hx = df_sub(hx, df_mul_f32(d2, kx));
          hy = df_sub(hy, df_mul_f32(d2, ky));
        }
        P = Df3(vec3f(hx.x, hy.x, P.hi.z), vec3f(hx.y, hy.y, P.lo.z));
`,
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
    id: 31,
    key: "absXYZ",
    name: "Abs XYZ",
    wRule: W_UNCHANGED,
    category: "fold",
    blurb:
      "Abs fold you can switch on per axis, so you can mirror just one or two axes for an asymmetric shape.",
    params: [
      {
        name: "AbsX",
        default: 1.0,
        min: 0.0,
        max: 1.0,
        step: 1.0,
        type: "double",
        tip: "Mirror this axis on (1) or off (0).",
      },
      {
        name: "AbsY",
        default: 1.0,
        min: 0.0,
        max: 1.0,
        step: 1.0,
        type: "double",
        tip: "Mirror this axis on (1) or off (0).",
      },
      {
        name: "AbsZ",
        default: 1.0,
        min: 0.0,
        max: 1.0,
        step: 1.0,
        type: "double",
        tip: "Mirror this axis on (1) or off (0).",
      },
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
    // perturbation twin (PERTURBATION_ZOOM_IMPL.md PR-2): the exact delta
    // map — mutates (ptD: vec3f, w: f32) given the reference records
    // pv0..pv3 (core/perturb.js layout). Plain f32 — no EFTs, no launder.
    wgslPt: `
        let gv = vec3f(op.p0, op.p1, op.p2);
        for (var a = 0u; a < 3u; a = a + 1u) {
          if (gv[a] > 0.5) {
            let zx = pv0[a]; let dx = ptD[a];
            let refUp = zx >= 0.0; let sPos = dx >= -zx;
            if (refUp == sPos) { ptD[a] = select(-dx, dx, refUp); }
            else { ptD[a] = select(2.0 * zx + dx, -2.0 * zx - dx, refUp); }
          }
        }`,
    // deep zoom P4 (DEEP_ZOOM_DF64.md): df64 twin — mutates (P: Df3, w).
    wgslDf: `
        if (op.p0 > 0.5) { let a = df_abs(df3_get(P, 0)); P = Df3(vec3f(a.x, P.hi.y, P.hi.z), vec3f(a.y, P.lo.y, P.lo.z)); }
        if (op.p1 > 0.5) { let a = df_abs(df3_get(P, 1)); P = Df3(vec3f(P.hi.x, a.x, P.hi.z), vec3f(P.lo.x, a.y, P.lo.z)); }
        if (op.p2 > 0.5) { let a = df_abs(df3_get(P, 2)); P = Df3(vec3f(P.hi.x, P.hi.y, a.x), vec3f(P.lo.x, P.lo.y, a.y)); }
`,
    glsl: (v) => `
    // per-axis abs fold (abs X/Y/Z) — reflect only the enabled axes.
    if (${v[0]} > 0.5) pos.x = abs(pos.x);
    if (${v[1]} > 0.5) pos.y = abs(pos.y);
    if (${v[2]} > 0.5) pos.z = abs(pos.z);`,
  },
  {
    id: 32,
    key: "varyScale",
    name: "Vary Scale Fold",
    wRule: W_MUL_K,
    category: "sphere",
    blurb:
      "Sphere fold whose reach changes with distance from the center, so detail grows denser or sparser as you move out.",
    params: [
      {
        name: "MinRadius",
        default: 0.5,
        min: 0.05,
        max: 2.0,
        step: 0.01,
        type: "double",
        tip: "Inner core size.",
      },
      {
        name: "FixedRadius",
        default: 1.0,
        min: 0.1,
        max: 3.0,
        step: 0.01,
        type: "double",
        tip: "Bubble size.",
      },
      {
        name: "RPower",
        default: 1.0,
        min: 0.2,
        max: 3.0,
        step: 0.05,
        type: "double",
        tip: "How much the reach drifts with distance. 1 is a plain sphere fold; away from 1 warps the density.",
      },
    ],
    // Radial-power sphere fold (MB3D ABoxVaryScale family): the ball-fold's
    // radial test runs on r²^RPower instead of r², so the fold radius — and
    // with it the apparent density — varies with distance from the origin.
    // RPower = 1 degenerates to the exact Sphere Fold. The max() guard keeps
    // log2 finite when an orbit lands on the origin.
    //
    // DE derivation (#636 found the breach, #643 pays it). The mid band is
    // f(p) = k(r)·p with k = F²/r^{2γ}, γ = RPower, so dk/dr = −2γ·k/r and
    //   J = k·I + p⊗∇k = k·[I − 2γ·(r̂⊗r̂)]
    // — RADIAL eigenvalue k·(1−2γ), tangential k. The mid band is therefore
    // conformal ONLY at γ = 1 (the Sphere Fold degeneracy): the true operator
    // norm is k·max(1, |1−2γ|), which for the shipped γ ∈ (1, 3] is
    // (2γ−1)·k > k (the audit measured 3.274 at γ = 2.139 against a theory
    // 3.278 — exact). γ ≤ 1 was always sound and its w is unchanged to the
    // bit; the other two branches are uniform scalings (norm = k) and carry no
    // correction either. Before #643 w charged k everywhere and the analytic
    // r/|w| DE over-stepped by up to (2γ−1) = 5× at the RPower ceiling.
    wgsl: `
        let minR2 = op.p0 * op.p0;
        let fixedR2 = op.p1 * op.p1;
        let rp = exp2(op.p2 * log2(max(dot(pos, pos), 1e-12)));
        var k = 1.0;
        var vsL = 1.0;
        if (rp < minR2) { k = fixedR2 / minR2; }
        else if (rp < fixedR2) {
            k = fixedR2 / rp;
            vsL = max(1.0, abs(1.0 - 2.0 * op.p2));
        }
        pos = pos * k;
        w = w * k * vsL;`,
    glsl: (v) => `
    // vary-scale fold — sphere fold with the radial test on r²^RPower. The mid
    // band stretches radially by (1-2*RPower), so w charges k * max(1,|1-2p|).
    {
        float minR2 = ${v[0]} * ${v[0]};
        float fixedR2 = ${v[1]} * ${v[1]};
        float rp = exp2(${v[2]} * log2(max(dot(pos, pos), 1e-12)));
        float k, vsL = 1.0;
        if      (rp < minR2)   k = fixedR2 / minR2;
        else if (rp < fixedR2) {
            k = fixedR2 / rp;
            vsL = max(1.0, abs(1.0 - 2.0 * ${v[2]}));
        }
        else                   k = 1.0;
        pos  *= k;
        w    *= k * vsL;
        g_wq *= k; // spatial scale only — the radial charge is DE bookkeeping
    }`,
  },
  {
    id: 33,
    key: "bristorBrot",
    name: "BristorBrot",
    wRule: W_BULB_NUMERIC,
    category: "power",
    blurb:
      "A twisted 3D square (the Bristorbrot) that mixes the axes into leafy, coral-like growths. Escape-time mode with numeric detail.",
    params: [
      {
        name: "XMul",
        default: 2.0,
        min: -4.0,
        max: 4.0,
        step: 0.1,
        type: "double",
        tip: "Axis-mixing strength; the defaults give the classic Bristorbrot.",
      },
      {
        name: "YMul",
        default: 1.0,
        min: -4.0,
        max: 4.0,
        step: 0.1,
        type: "double",
        tip: "Axis-mixing strength; the defaults give the classic Bristorbrot.",
      },
      {
        name: "ZMul",
        default: -1.0,
        min: -4.0,
        max: 4.0,
        step: 0.1,
        type: "double",
        tip: "Axis-mixing strength; the defaults give the classic Bristorbrot.",
      },
    ],
    // Bristorbrot triplex square (first W_BULB_NUMERIC op — COVERAGE_PLAN §3):
    //   x' = x² − y² − z²,  y' = y·(XMul·x + ZMul·z),  z' = z·(XMul·x + YMul·y)
    // (defaults 2/1/−1 give the classic Bristorbrot). A bilinear map with no
    // clean analytic dr — w is left untouched and the engine's numeric
    // finite-difference DE takes over (effectiveDeOption → 3). Escape-time:
    // pair with Add c like the bulbs.
    wgsl: `
        let bx = pos.x; let by = pos.y; let bz = pos.z;
        pos.x = bx * bx - by * by - bz * bz;
        pos.y = by * (op.p0 * bx + op.p2 * bz);
        pos.z = bz * (op.p0 * bx + op.p1 * by);`,
    glsl: (v) => `
    // Bristorbrot triplex square — numeric DE (no analytic dr; w untouched).
    {
        float bx = pos.x, by = pos.y, bz = pos.z;
        pos.x = bx * bx - by * by - bz * bz;
        pos.y = by * (${v[0]} * bx + ${v[2]} * bz);
        pos.z = bz * (${v[0]} * bx + ${v[1]} * by);
    }`,
  },
  {
    id: 34,
    key: "newtonTri2",
    name: "Newton Triplex z²",
    wRule: W_BULB_NUMERIC,
    category: "power",
    blurb:
      "A cross-mixing 3D square that grows knotted, interwoven lobes. Escape-time mode with numeric detail.",
    params: [],
    // NewtonTri quadratic triplex (MB3D NewtonTri2, from the readable C++ ref):
    //   x' = x² − 2yz,  y' = 2xy − z²,  z' = y² + 2xz
    // A cross-mixing polynomial square with no analytic dr → numeric DE
    // (w untouched; the desktop's crude w·=4 bound is ignored here). Escape-time:
    // pair with Add c.
    wgsl: `
        let nx = pos.x; let ny = pos.y; let nz = pos.z;
        pos.x = nx * nx - 2.0 * ny * nz;
        pos.y = 2.0 * nx * ny - nz * nz;
        pos.z = ny * ny + 2.0 * nx * nz;`,
    glsl: () => `
    // NewtonTri quadratic triplex — numeric DE (no analytic dr; w untouched).
    {
        float nx = pos.x, ny = pos.y, nz = pos.z;
        pos.x = nx * nx - 2.0 * ny * nz;
        pos.y = 2.0 * nx * ny - nz * nz;
        pos.z = ny * ny + 2.0 * nx * nz;
    }`,
  },
  {
    id: 35,
    key: "newtonTri3",
    name: "Newton Triplex z³",
    wRule: W_BULB_NUMERIC,
    category: "power",
    blurb:
      "A cross-mixing 3D cube — like Newton Triplex z² but a higher power, for busier interwoven forms. Escape-time mode with numeric detail.",
    params: [],
    // NewtonTri cubic triplex (MB3D NewtonTri3, from the readable C++ ref):
    //   x' = x³ − y³ + z³ − 6xyz,  y' = 3(x²y − y²z − xz²),  z' = 3(x²z + xy² − yz²)
    // Cubic cross-mixing polynomial → numeric DE (w untouched). Escape-time:
    // pair with Add c.
    wgsl: `
        let nx = pos.x; let ny = pos.y; let nz = pos.z;
        pos.x = nx * nx * nx - ny * ny * ny + nz * nz * nz - 6.0 * nx * ny * nz;
        pos.y = 3.0 * (nx * nx * ny - ny * ny * nz - nx * nz * nz);
        pos.z = 3.0 * (nx * nx * nz + nx * ny * ny - ny * nz * nz);`,
    glsl: () => `
    // NewtonTri cubic triplex — numeric DE (no analytic dr; w untouched).
    {
        float nx = pos.x, ny = pos.y, nz = pos.z;
        pos.x = nx * nx * nx - ny * ny * ny + nz * nz * nz - 6.0 * nx * ny * nz;
        pos.y = 3.0 * (nx * nx * ny - ny * ny * nz - nx * nz * nz);
        pos.z = 3.0 * (nx * nx * nz + nx * ny * ny - ny * nz * nz);
    }`,
  },
  {
    id: 36,
    key: "msltoeSym3",
    name: "Msltoe Sym z²",
    wRule: W_BULB,
    category: "power",
    blurb:
      "Msltoe's symmetric squared bulb — a smoother, more flowing take on the Mandelbulb. Escape-time mode.",
    params: [
      {
        name: "YMul",
        default: 1.0,
        min: -4.0,
        max: 4.0,
        step: 0.1,
        type: "double",
        tip: "Stretches the shape along one axis.",
      },
      {
        name: "Variant",
        default: 0.0,
        min: 0.0,
        max: 1.0,
        step: 1.0,
        type: "double",
        tip: "Switches between two closely related bulb versions (Sym3 / Sym2).",
      },
    ],
    // msltoe's symmetric quadratic bulb (MB3D MsltoeSym3/Sym2, published on
    // FractalForums). A sign-symmetrized (y,z) pair feeds a damped complex
    // square:  a=y, b=z;  sign rule (see Variant);  m = 1 − a²/r²;
    //   x' = (b² − x²)·m,  y' = 2·x·b·m·YMul,  z' = 2·a·√(x² + b²)
    // Variant 0 = Sym3 (b≥a negates BOTH a,b — verified vs a true MB3D
    // render); Variant 1 = Sym2 (negates b only; same decompiler conventions
    // the Sym3 ref adjudicated). Sym2 deviation: MB3D's body adds no c.z —
    // our Add c adds all three components. Old formulas carry one value and
    // sanitize pads Variant to 0 (Sym3), so shipped content is unchanged.
    // Degree-2 radial growth (|F| ≈ r² by design) — w tracks the quadratic
    // escape-time dr = 2·r·w + 1, like Quadratic/Mandelbulb Power. Escape-time:
    // pair with Add c.
    wgsl: `
        let mr2 = max(dot(pos, pos), 1e-12);
        w = 2.0 * sqrt(mr2) * w + 1.0;
        var ma = pos.y; var mb = pos.z;
        if (mb >= ma) { mb = -mb; if (op.p1 < 0.5) { ma = -ma; } }
        let mm = 1.0 - ma * ma / mr2;
        let mx = pos.x;
        pos.x = (mb * mb - mx * mx) * mm;
        pos.y = 2.0 * mx * mb * mm * op.p0;
        pos.z = 2.0 * ma * sqrt(mx * mx + mb * mb);`,
    glsl: (v) => `
    // Msltoe symmetric quadratic bulb — escape-time; w carries dr = 2·r·w + 1.
    // Variant 0 = Sym3 (negate both), 1 = Sym2 (negate b only).
    {
        float mr2 = max(dot(pos, pos), 1e-12);
        w = 2.0 * sqrt(mr2) * w + 1.0;
        float ma = pos.y, mb = pos.z;
        if (mb >= ma) { mb = -mb; if (${v[1]} < 0.5) ma = -ma; }
        float mm = 1.0 - ma * ma / mr2;
        float mx = pos.x;
        pos.x = (mb * mb - mx * mx) * mm;
        pos.y = 2.0 * mx * mb * mm * ${v[0]};
        pos.z = 2.0 * ma * sqrt(mx * mx + mb * mb);
    }`,
  },
  {
    id: 37,
    key: "sphericalTwoStage",
    name: "Bulb Power (two-angle)",
    wRule: W_BULB,
    category: "power",
    blurb:
      "Mandelbulb power with separate control over its two spin angles, for stretched or sheared bulbs. Escape-time mode.",
    params: [
      {
        name: "Power",
        default: 8.0,
        min: 2.0,
        max: 16.0,
        step: 0.1,
        type: "double",
        tip: "Number of lobes (8 = classic).",
      },
      {
        name: "ThetaMul",
        default: 1.0,
        min: -4.0,
        max: 4.0,
        step: 0.05,
        type: "double",
        tip: "Stretches the bulbs top-to-bottom.",
      },
      {
        name: "PhiMul",
        default: 1.0,
        min: -4.0,
        max: 4.0,
        step: 0.05,
        type: "double",
        tip: "Stretches the bulbs around.",
      },
    ],
    // Spherical power with INDEPENDENT angle multipliers (the published
    // generalized bulb — separate alpha/beta angle multipliers, the capability
    // the community "two-stage" bulb family exercises, see #83):
    //   θ' = n·ThetaMul·θ,  φ' = n·PhiMul·φ,  r' = rⁿ
    // Degeneracy anchor: ThetaMul = PhiMul = 1 is EXACTLY Mandelbulb Power —
    // the oracle verifies this op by matching that render. w carries the same
    // radial derivative dr = n·rⁿ⁻¹·dr + 1 (angle warps are radius-preserving).
    // Escape-time: pair with Add c.
    wgsl: `
        let bp = op.p0;
        let br = length(pos);
        if (br > 1e-9) {
          let bth = acos(clamp(pos.z / br, -1.0, 1.0)) * bp * op.p1;
          let bph = atan2(pos.y, pos.x) * bp * op.p2;
          let brn = pow(br, bp);
          w = bp * brn / br * w + 1.0;
          let bst = sin(bth);
          pos = brn * vec3f(bst * cos(bph), bst * sin(bph), cos(bth));
        }`,
    glsl: (v) => `
    // Two-angle spherical power (generalized bulb) — escape-time; w carries
    // the radial derivative like Mandelbulb Power.
    {
        float bp = ${v[0]};
        float br = length(pos);
        if (br > 1e-9) {
            float bth = acos(clamp(pos.z / br, -1.0, 1.0)) * bp * ${v[1]};
            float bph = atan(pos.y, pos.x) * bp * ${v[2]};
            float brn = pow(br, bp);
            w = bp * brn / br * w + 1.0;
            float bst = sin(bth);
            pos = brn * vec3(bst * cos(bph), bst * sin(bph), cos(bth));
        }
    }`,
  },
  {
    id: 38,
    key: "boxBulb",
    name: "Box Bulb",
    wRule: W_BULB,
    category: "power",
    blurb:
      "A boxy Mandelbulb built from a cube-like distance measure, giving angular bulbs with flatter facets. Escape-time mode.",
    params: [
      {
        name: "Power",
        default: 8.0,
        min: 2.0,
        max: 16.0,
        step: 0.1,
        type: "double",
        tip: "Number of lobes (8 = classic).",
      },
    ],
    // MB3D BoxBulb (from the readable C++ ref): the Mandelbulb recipe with the
    // L4 norm (x⁴+y⁴+z⁴)^¼ as radius and (x⁴+z⁴)^¼ in the polar angle, y as
    // the polar axis — the "boxy" bulb. w carries the analytic radial
    // derivative on the L4 radius, dr = n·r₄ⁿ⁻¹·dr + 1 (norm-equivalent to the
    // L2 bound). Escape-time: pair with Add c.
    wgsl: `
        let bp = op.p0;
        let q4 = pos * pos * pos * pos;
        let br = pow(q4.x + q4.y + q4.z, 0.25);
        if (br > 1e-9) {
          let bxz = pow(q4.x + q4.z, 0.25);
          let brn = pow(br, bp);
          let bth = atan2(bxz, pos.y) * bp;
          let bza = atan2(pos.x, pos.z) * bp;
          w = bp * brn / br * w + 1.0;
          let bst = sin(bth);
          pos = brn * vec3f(sin(bza) * bst, cos(bth), bst * cos(bza));
        }`,
    glsl: (v) => `
    // BoxBulb — L4-norm Mandelbulb, y-polar (readable-C++ ref). Escape-time.
    {
        float bp = ${v[0]};
        vec3 q4 = pos * pos * pos * pos;
        float br = pow(q4.x + q4.y + q4.z, 0.25);
        if (br > 1e-9) {
            float bxz = pow(q4.x + q4.z, 0.25);
            float brn = pow(br, bp);
            float bth = atan(bxz, pos.y) * bp;
            float bza = atan(pos.x, pos.z) * bp;
            w = bp * brn / br * w + 1.0;
            float bst = sin(bth);
            pos = brn * vec3(sin(bza) * bst, cos(bth), bst * cos(bza));
        }
    }`,
  },
  {
    id: 39,
    key: "slonoBrot2",
    name: "SlonoBrot",
    wRule: W_BULB,
    category: "power",
    blurb:
      "The SlonoBrot — a fold-and-square move that grows lumpy, brain-like surfaces. Escape-time mode.",
    params: [],
    // MB3D SlonoBrot2 (from the readable C++ ref): a z-folded quadratic —
    //   a = |z|;  x' = x² − y² + 2ax,  y' = 2y(x + a),  z' = |a² − y²|
    // Degree-2 radial growth; w carries the quadratic escape-time
    // dr = 2·r·w + 1 (the folds are isometries; the desktop's w×0.7 is a
    // hand fudge we don't need). Known deviation: MB3D's body adds |c.z| to
    // z; our engine's Add c adds plain c.z (identical wherever c.z ≥ 0).
    // Escape-time: pair with Add c.
    wgsl: `
        w = 2.0 * length(pos) * w + 1.0;
        let sa = abs(pos.z);
        let sx = pos.x; let sy = pos.y;
        pos.x = sx * sx - sy * sy + 2.0 * sa * sx;
        pos.y = 2.0 * sy * (sx + sa);
        pos.z = abs(sa * sa - sy * sy);`,
    glsl: () => `
    // SlonoBrot2 — z-folded quadratic; w carries dr = 2·r·w + 1.
    {
        w = 2.0 * length(pos) * w + 1.0;
        float sa = abs(pos.z);
        float sx = pos.x, sy = pos.y;
        pos.x = sx * sx - sy * sy + 2.0 * sa * sx;
        pos.y = 2.0 * sy * (sx + sa);
        pos.z = abs(sa * sa - sy * sy);
    }`,
  },
  {
    id: 40,
    key: "scaleDrift",
    name: "Scale Drift",
    wRule: W_MUL_SCALE,
    category: "move",
    blurb:
      "Zoom that slowly speeds up or eases off with each step, so the repeat rate drifts as detail gets finer (an Amazing-Surf trick).",
    params: [
      {
        name: "Scale",
        default: 2.0,
        min: -4.0,
        max: 4.0,
        step: 0.01,
        type: "double",
        tip: "Starting zoom factor per step (about 2 is classic).",
      },
      {
        name: "ScaleVary",
        default: 0.05,
        min: -0.5,
        max: 0.5,
        step: 0.005,
        type: "double",
        tip: "How much the zoom drifts each step. 0 is a plain scale; small values add gentle drift.",
      },
    ],
    // Amazing-Surf running-scale feedback (Scale = Scale + ScaleVary*(|Scale|-1)),
    // which is orbit-independent → the closed-form ramp
    // m = 1 + (Scale-1)*(1+ScaleVary)^(i+1). ScaleVary=0 ≡ the Scale op. Conformal
    // scale, so w tracks |m| like Scale; DE stays analytic. See docs/design/SCALE_VARY.md.
    // ScaleVary bounds keep (1+ScaleVary) ∈ [0.5,1.5] > 0 so pow() is well-defined.
    // Exponent is f32(i)+1 / s.i+1 (MB3D updates Scale BEFORE first use — §2 indexing).
    wgsl: `
        let m = clamp(1.0 + (op.p0 - 1.0) * pow(1.0 + op.p1, f32(i) + 1.0), -1.0e5, 1.0e5);
        pos = pos * m;
        w = w * abs(m);`,
    // deep zoom P4 (DEEP_ZOOM_DF64.md): df64 twin — mutates (P: Df3, w).
    wgslDf: `
        let m = clamp(1.0 + (op.p0 - 1.0) * pow(1.0 + op.p1, f32(i) + 1.0), -1.0e5, 1.0e5);
        P = df3_mul_f32(P, m);
        w = w * abs(m);
`,
    glsl: (v) => `
    // Scale Drift — closed-form per-iteration scale ramp (needs int i in scope).
    {
        float m = clamp(1.0 + (${v[0]} - 1.0) * pow(1.0 + ${v[1]}, float(i) + 1.0), -1.0e5, 1.0e5);
        pos  *= m;
        w    *= abs(m);
        g_wq *= abs(m);
    }`,
    // Desktop export ONLY (see SCALE_VARY.md §6.4): no iteration index in
    // iterateJIT_ → collapse the drift to its first-iteration factor S₁ (a plain
    // constant scale).
    desktopGlsl: (v) => `
    // Scale Drift — per-iteration drift NOT representable in the desktop JIT ABI;
    // collapsed to the first-iteration factor S1 = 1+(Scale-1)*(1+ScaleVary).
    {
        float m = 1.0 + (${v[0]} - 1.0) * (1.0 + ${v[1]});
        pos  *= m;
        w    *= abs(m);
        g_wq *= abs(m);
    }`,
  },
  {
    id: 41,
    key: "riemannBulb",
    name: "Riemann Bulb",
    wRule: W_BULB_NUMERIC,
    category: "power",
    blurb:
      "Spins the shape through the Riemann sphere — a stereographic power that grows tilted, planet-like shells. Escape-time mode with numeric detail.",
    params: [
      {
        name: "Power",
        default: 8.0,
        min: 2.0,
        max: 16.0,
        step: 0.1,
        type: "double",
        tip: "Angular power on the Riemann sphere (8 = the classic look).",
      },
    ],
    // Riemann-sphere stereographic power (B2 wave 4 — COVERAGE_PLAN §3):
    // project the unit direction stereographically from the +Y pole into two
    // tan-half-angle coordinates (a, b), raise each angle by P via
    // tan(P·2·atan(t)) (the atan2(2t, t²−1) double-angle form keeps the
    // branch), then invert the projection and scale by r^P. The bounded
    // combine — x,z by 1/(S+1), y by (S−1)/(S+1), S = ta²+tb² rescaled by
    // m = max(|ta|,|tb|,1) — cancels the tan poles so float32 never overflows;
    // d and cos+1 are clamped off their measure-zero poles. No clean analytic
    // dr: w is left untouched and the numeric finite-difference DE takes over
    // (same class as BristorBrot). Escape-time: pair with Add c.
    wgsl: `
        let rP = op.p0;
        let rR = dot(pos, pos);
        if (rR > 1e-18) {
          let invr = inverseSqrt(rR);
          let rd = min(pos.y * invr - 1.0, -1e-7);
          let ru = 1.0 / rd;
          let ra = pos.x * invr * ru;
          let rb = pos.z * invr * ru;
          let alphaP = atan2(ra + ra, ra * ra - 1.0) * rP;
          let betaP  = atan2(rb + rb, rb * rb - 1.0) * rP;
          let ta = sin(alphaP) / max(cos(alphaP) + 1.0, 1e-30);
          let tb = sin(betaP)  / max(cos(betaP)  + 1.0, 1e-30);
          let m  = max(max(abs(ta), abs(tb)), 1.0);
          let tA = ta / m; let tB = tb / m;
          let q  = tA * tA + tB * tB;
          let im = 1.0 / m;
          let den = m * q + im;
          let rp = pow(sqrt(rR), rP);
          pos = vec3f(2.0 * tA / den * rp,
                      (q - im * im) / (q + im * im) * rp,
                      2.0 * tB / den * rp);
        }`,
    glsl: (v) => `
    // Riemann-sphere stereographic power — numeric DE (no analytic dr; w untouched).
    {
        float rP = ${v[0]};
        float rR = dot(pos, pos);
        if (rR > 1e-18) {
            float invr = inversesqrt(rR);
            float rd = min(pos.y * invr - 1.0, -1e-7);
            float ru = 1.0 / rd;
            float ra = pos.x * invr * ru;
            float rb = pos.z * invr * ru;
            float alphaP = atan(ra + ra, ra * ra - 1.0) * rP;
            float betaP  = atan(rb + rb, rb * rb - 1.0) * rP;
            float ta = sin(alphaP) / max(cos(alphaP) + 1.0, 1e-30);
            float tb = sin(betaP)  / max(cos(betaP)  + 1.0, 1e-30);
            float m  = max(max(abs(ta), abs(tb)), 1.0);
            float tA = ta / m, tB = tb / m;
            float q  = tA * tA + tB * tB;
            float im = 1.0 / m;
            float den = m * q + im;
            float rp = pow(sqrt(rR), rP);
            pos = vec3(2.0 * tA / den * rp,
                       (q - im * im) / (q + im * im) * rp,
                       2.0 * tB / den * rp);
        }
    }`,
  },
  {
    id: 42,
    key: "kleinPolyMap",
    name: "Klein Poly Map",
    wRule: W_BULB_NUMERIC,
    category: "power",
    blurb:
      "Wraps space through the Riemann sphere and a Klein polyhedral map — braided, crystal-like symmetry knots. Escape-time mode with numeric detail.",
    params: [
      {
        name: "Log2Power",
        default: 0.0,
        min: 0.0,
        max: 7.0,
        step: 1.0,
        type: "double",
        tip: "How many warp passes (log scale). 0 and 1 both mean one pass.",
      },
      {
        name: "Log2MapPower",
        default: 0.0,
        min: 0.0,
        max: 7.0,
        step: 1.0,
        type: "double",
        tip: "How many map passes inside each warp (log scale).",
      },
      {
        name: "Variant",
        default: 0.0,
        min: 0.0,
        max: 3.0,
        step: 1.0,
        type: "double",
        tip: "0/1 tetrahedral, 2/3 dihedral; the odd ones flip the frame.",
      },
    ],
    // Klein polyhedral rational map (Tglad family — B2 wave 4): stereographic
    // Riemann-sphere warp in (tan-half-angle form), N inner passes of a Klein
    // group rational map on the mapped complex pair — tetrahedral
    // cubic-rational (Variant 0/1) or dihedral rational (Variant 2/3) — with
    // the carried radius squaring per pass, then warp back. Odd variants swap
    // the warp components (with a sign) and negate the final height — the
    // MB3D Tetra2/Dihed2 frames. Loop counts are max(1, Log2* & 7) with
    // bottom-tested loops (always ≥1 pass). Escape (≥1e10 on the outer r² or
    // the inner head) parks the orbit at (1e10,1e10,1e10); the source skips
    // the +c on that path — engine AddC still adds it here, which float32
    // absorbs (ulp(1e10) ≈ 1024 ≫ |c|). No analytic dr: w untouched, numeric
    // DE (same class as BristorBrot). Ported from the hand-crafted,
    // FPU-simulation-verified reconstruction (2400/2400 float64 matches).
    wgsl: `
        let ko = max(i32(op.p0 + 0.5) & 7, 1);
        let ki = max(i32(op.p1 + 0.5) & 7, 1);
        let kv = i32(op.p2 + 0.5);
        let kalt = (kv & 1) == 1;
        let kdih = kv >= 2;
        var x = pos.x; var y = pos.y; var z = pos.z;
        var esc = false;
        for (var io = 0; io < ko; io++) {
          let rho2 = x * x + y * y;
          let r2 = rho2 + z * z;
          if (r2 >= 1.0e10) { esc = true; break; }
          var rr = sqrt(r2);
          let zr = z / rr;
          let tang = (sqrt(rho2) - y) / x;
          let tsq = tang * tang;
          let qq = (1.0 / (tsq + 1.0)) * sqrt((1.0 - zr) / (1.0 + zr));
          if (kalt) { y = (qq + qq) * tang; x = (tsq - 1.0) * qq; }
          else      { x = (qq + qq) * tang; y = (1.0 - tsq) * qq; }
          for (var ii = 0; ii < ki; ii++) {
            if (kdih) {
              let j = x * x + y * y;
              if (j >= 1.0e10) { esc = true; break; }
              let ni = ((-y) * (j + 1.0)) * ((j - 2.0 * x) - 1.0) * ((j + 2.0 * x) - 1.0);
              let nr = (x * (j - 1.0)) * ((j * j + 2.0 * j) + 4.0 * y * y + 1.0);
              let cd = ((j - 2.0 * y) + 1.0) * ((j + 2.0 * y) + 1.0);
              let cm = 2.0 / (cd * cd);
              x = cm * nr; y = cm * ni;
            } else {
              let sRe = x * x * x - 3.0 * y * y * x;
              if (sRe >= 1.0e10) { esc = true; break; }
              let tIm = y * y * y - 3.0 * x * x * y;
              let bN = tIm * 3.181980515339464;
              let tSq = tIm * tIm;
              let sk = sRe + 0.3535533905932738;
              let aN = sk * (2.8284271247461903 - sRe) - tSq;
              let cm = (-0.3535533905932738) / (sk * sk + tSq);
              let xn = (aN * x - bN * y) * cm;
              let yn = (aN * y + bN * x) * cm;
              x = xn; y = yn;
            }
            rr = rr * rr;
          }
          if (esc) { break; }
          if (kalt) { let t = x; x = y; y = t; }
          let u = x * x + y * y;
          let su = sqrt(u);
          let tg = (su - y) / x;
          let tq = tg * tg;
          let inv = 1.0 / (tq + 1.0);
          let cosO = (1.0 - tq) * inv;
          let sinO = (tg + tg) * inv;
          let g = 2.0 / (1.0 + u);
          let sa = g * su;
          let xn = rr * sinO * sa;
          let yn = rr * cosO * sa;
          if (kalt) { z = (1.0 - g) * rr; } else { z = (g - 1.0) * rr; }
          x = xn; y = yn;
        }
        if (esc) { pos = vec3f(1.0e10, 1.0e10, 1.0e10); }
        else     { pos = vec3f(x, y, z); }`,
    glsl: (v) => `
    // Klein polyhedral rational map (Tglad family) — numeric DE (w untouched).
    {
        int ko = max(int(${v[0]} + 0.5) & 7, 1);
        int ki = max(int(${v[1]} + 0.5) & 7, 1);
        int kv = int(${v[2]} + 0.5);
        bool kalt = (kv & 1) == 1;
        bool kdih = kv >= 2;
        float x = pos.x, y = pos.y, z = pos.z;
        bool esc = false;
        for (int io = 0; io < ko; io++) {
            float rho2 = x * x + y * y;
            float r2 = rho2 + z * z;
            if (r2 >= 1.0e10) { esc = true; break; }
            float rr = sqrt(r2);
            float zr = z / rr;
            float tang = (sqrt(rho2) - y) / x;
            float tsq = tang * tang;
            float qq = (1.0 / (tsq + 1.0)) * sqrt((1.0 - zr) / (1.0 + zr));
            if (kalt) { y = (qq + qq) * tang; x = (tsq - 1.0) * qq; }
            else      { x = (qq + qq) * tang; y = (1.0 - tsq) * qq; }
            for (int ii = 0; ii < ki; ii++) {
                if (kdih) {
                    float j = x * x + y * y;
                    if (j >= 1.0e10) { esc = true; break; }
                    float ni = ((-y) * (j + 1.0)) * ((j - 2.0 * x) - 1.0) * ((j + 2.0 * x) - 1.0);
                    float nr = (x * (j - 1.0)) * ((j * j + 2.0 * j) + 4.0 * y * y + 1.0);
                    float cd = ((j - 2.0 * y) + 1.0) * ((j + 2.0 * y) + 1.0);
                    float cm = 2.0 / (cd * cd);
                    x = cm * nr; y = cm * ni;
                } else {
                    float sRe = x * x * x - 3.0 * y * y * x;
                    if (sRe >= 1.0e10) { esc = true; break; }
                    float tIm = y * y * y - 3.0 * x * x * y;
                    float bN = tIm * 3.181980515339464;
                    float tSq = tIm * tIm;
                    float sk = sRe + 0.3535533905932738;
                    float aN = sk * (2.8284271247461903 - sRe) - tSq;
                    float cm = (-0.3535533905932738) / (sk * sk + tSq);
                    float xn = (aN * x - bN * y) * cm;
                    float yn = (aN * y + bN * x) * cm;
                    x = xn; y = yn;
                }
                rr = rr * rr;
            }
            if (esc) { break; }
            if (kalt) { float t = x; x = y; y = t; }
            float u = x * x + y * y;
            float su = sqrt(u);
            float tg = (su - y) / x;
            float tq = tg * tg;
            float inv = 1.0 / (tq + 1.0);
            float cosO = (1.0 - tq) * inv;
            float sinO = (tg + tg) * inv;
            float g = 2.0 / (1.0 + u);
            float sa = g * su;
            float xn = rr * sinO * sa;
            float yn = rr * cosO * sa;
            z = (kalt ? (1.0 - g) : (g - 1.0)) * rr;
            x = xn; y = yn;
        }
        pos = esc ? vec3(1.0e10) : vec3(x, y, z);
    }`,
  },
  {
    id: 43,
    key: "magnetXYZ",
    name: "Magnet XYZ",
    wRule: W_BULB_NUMERIC,
    category: "power",
    blurb:
      "A magnet-style pull applied along each axis at once — dented, magnetized blobs with deep pockets. Escape-time mode with numeric detail.",
    params: [
      {
        name: "Power",
        default: 2.0,
        min: 1.0,
        max: 8.0,
        step: 0.1,
        type: "double",
        tip: "Strength of the magnet map (2 = classic).",
      },
      {
        name: "Axiom1",
        default: 1.57,
        min: 0.0,
        max: 3.14,
        step: 0.01,
        type: "double",
        tip: "Angle-source mix — shifts where the pull points.",
      },
      {
        name: "Axiom2",
        default: 0.79,
        min: 0.0,
        max: 3.14,
        step: 0.01,
        type: "double",
        tip: "Per-axis radius boost — reshapes the pockets.",
      },
    ],
    // Per-axis magnet-style rational map (MagVsXYZ family — B2 wave 4):
    //   a' = cos(P·atan2(r, |a|·P·A₁)) · (r² + A₂·a²)^(P/2)   for a ∈ {x,y,z}
    // with r the full radius, all three lanes computed from the same input
    // point. Implemented from the symmetric math, NOT the emitted GLSL: the
    // decompile's x-lane c-multiplier reads the wrong param slot (off-by-one)
    // and its exp2(mod)·pow radial pattern is the x87 f2xm1/fscale idiom for
    // an exact power (the hand-crafted Riemann source spells the same idiom
    // out as 2^e) — so the radial factor here is the smooth (r²+A₂a²)^(P/2).
    // CosShift and per-axis c multipliers are defaulted away (engine AddC adds
    // c uniformly — the plan's accepted loss). No analytic dr: w untouched,
    // numeric DE.
    wgsl: `
        let mP = op.p0;
        let mA1 = op.p1;
        let mA2 = op.p2;
        let mR2 = dot(pos, pos);
        if (mR2 > 1e-18) {
          let mr = sqrt(mR2);
          let pa1 = mP * mA1;
          let qx = pow(mR2 + mA2 * pos.x * pos.x, mP * 0.5);
          let qy = pow(mR2 + mA2 * pos.y * pos.y, mP * 0.5);
          let qz = pow(mR2 + mA2 * pos.z * pos.z, mP * 0.5);
          pos = vec3f(
            cos(mP * atan2(mr, abs(pos.x) * pa1)) * qx,
            cos(mP * atan2(mr, abs(pos.y) * pa1)) * qy,
            cos(mP * atan2(mr, abs(pos.z) * pa1)) * qz);
        }`,
    glsl: (v) => `
    // Magnet XYZ — per-axis magnet rational map; numeric DE (w untouched).
    {
        float mP = ${v[0]};
        float mA1 = ${v[1]};
        float mA2 = ${v[2]};
        float mR2 = dot(pos, pos);
        if (mR2 > 1e-18) {
            float mr = sqrt(mR2);
            float pa1 = mP * mA1;
            float qx = pow(mR2 + mA2 * pos.x * pos.x, mP * 0.5);
            float qy = pow(mR2 + mA2 * pos.y * pos.y, mP * 0.5);
            float qz = pow(mR2 + mA2 * pos.z * pos.z, mP * 0.5);
            pos = vec3(
                cos(mP * atan(mr, abs(pos.x) * pa1)) * qx,
                cos(mP * atan(mr, abs(pos.y) * pa1)) * qy,
                cos(mP * atan(mr, abs(pos.z) * pa1)) * qz);
        }
    }`,
  },
  {
    id: 44,
    key: "magnetXYZAbs",
    name: "Magnet XYZ (abs)",
    wRule: W_BULB_NUMERIC,
    category: "power",
    blurb:
      "The magnet pull folded into positive space — sharper, ship-like ridges instead of pockets. Escape-time mode with numeric detail.",
    params: [
      {
        name: "Power",
        default: 2.0,
        min: 1.0,
        max: 8.0,
        step: 0.1,
        type: "double",
        tip: "Strength of the magnet map (2 = classic).",
      },
      {
        name: "Axiom1",
        default: 1.57,
        min: 0.0,
        max: 3.14,
        step: 0.01,
        type: "double",
        tip: "Drives BOTH the pull direction and the cosine angle here.",
      },
      {
        name: "Axiom2",
        default: 1.57,
        min: 0.0,
        max: 3.14,
        step: 0.01,
        type: "double",
        tip: "Per-axis radius boost — reshapes the ridges.",
      },
    ],
    // MagVsXYZabs: the abs-folded magnet variant. Two deliberate deltas from
    // magnetXYZ, both faithful to the variant's design: outputs are folded
    // through abs() (Burning-Ship-style), and the cosine angle is driven by
    // A₁ instead of P (the variant's own angle-multiplier choice; its default
    // A₂ is also π/2, not π/4):
    //   a' = |cos(A₁·atan2(r, |a|·P·A₁)) · (r² + A₂·a²)^(P/2)|
    // Same symmetric-math ruling as magnetXYZ (decompile slot quirks not
    // reproduced). Numeric DE; w untouched.
    wgsl: `
        let mP = op.p0;
        let mA1 = op.p1;
        let mA2 = op.p2;
        let mR2 = dot(pos, pos);
        if (mR2 > 1e-18) {
          let mr = sqrt(mR2);
          let pa1 = mP * mA1;
          let qx = pow(mR2 + mA2 * pos.x * pos.x, mP * 0.5);
          let qy = pow(mR2 + mA2 * pos.y * pos.y, mP * 0.5);
          let qz = pow(mR2 + mA2 * pos.z * pos.z, mP * 0.5);
          pos = vec3f(
            abs(cos(mA1 * atan2(mr, abs(pos.x) * pa1)) * qx),
            abs(cos(mA1 * atan2(mr, abs(pos.y) * pa1)) * qy),
            abs(cos(mA1 * atan2(mr, abs(pos.z) * pa1)) * qz));
        }`,
    glsl: (v) => `
    // Magnet XYZ (abs) — abs-folded magnet variant; numeric DE (w untouched).
    {
        float mP = ${v[0]};
        float mA1 = ${v[1]};
        float mA2 = ${v[2]};
        float mR2 = dot(pos, pos);
        if (mR2 > 1e-18) {
            float mr = sqrt(mR2);
            float pa1 = mP * mA1;
            float qx = pow(mR2 + mA2 * pos.x * pos.x, mP * 0.5);
            float qy = pow(mR2 + mA2 * pos.y * pos.y, mP * 0.5);
            float qz = pow(mR2 + mA2 * pos.z * pos.z, mP * 0.5);
            pos = vec3(
                abs(cos(mA1 * atan(mr, abs(pos.x) * pa1)) * qx),
                abs(cos(mA1 * atan(mr, abs(pos.y) * pa1)) * qy),
                abs(cos(mA1 * atan(mr, abs(pos.z) * pa1)) * qz));
        }
    }`,
  },
  {
    id: 45,
    key: "makinTri",
    name: "Makin Triplex",
    wRule: W_BULB_NUMERIC,
    category: "power",
    blurb:
      "A pair of Makin's twisted 3D squares — swirled, horn-like growths depending on the flavor. Escape-time mode with numeric detail.",
    params: [
      {
        name: "Variant",
        default: 0.0,
        min: 0.0,
        max: 1.0,
        step: 1.0,
        type: "double",
        tip: "0 or 1 — two different axis-mixing flavors.",
      },
    ],
    // Makin triplex squares (B2 wave 4) — two param-free bilinear maps in one
    // op, enum-selected like kleinPolyMap's Variant:
    //   0 (Makin3D-1): x' = x²−y²−z²,  y' = 2xy,          z' = 2z(x−y)
    //   1 (Makin3D-2): x' = x²+2yz,    y' = −(y²+2zx),    z' = −z²+2xy
    // No analytic dr: w untouched, numeric DE. Escape-time: pair with Add c.
    wgsl: `
        let mv = i32(op.p0 + 0.5);
        let ax = pos.x; let ay = pos.y; let az = pos.z;
        if (mv == 1) {
          pos = vec3f(ax * ax + 2.0 * ay * az,
                      -(ay * ay + 2.0 * az * ax),
                      -(az * az) + 2.0 * ay * ax);
        } else {
          pos = vec3f(ax * ax - ay * ay - az * az,
                      2.0 * ax * ay,
                      2.0 * az * (ax - ay));
        }`,
    glsl: (v) => `
    // Makin triplex square (variant 0/1) — numeric DE (w untouched).
    {
        int mv = int(${v[0]} + 0.5);
        float ax = pos.x, ay = pos.y, az = pos.z;
        if (mv == 1) {
            pos = vec3(ax * ax + 2.0 * ay * az,
                       -(ay * ay + 2.0 * az * ax),
                       -(az * az) + 2.0 * ay * ax);
        } else {
            pos = vec3(ax * ax - ay * ay - az * az,
                       2.0 * ax * ay,
                       2.0 * az * (ax - ay));
        }
    }`,
  },
  {
    id: 46,
    key: "makinFuzzy",
    name: "Makin Fuzzy",
    wRule: W_BULB_NUMERIC,
    category: "power",
    blurb:
      "Makin's square with a fuzzy damping term — softened, smoke-like growths you can sharpen or sign-flip. Escape-time mode with numeric detail.",
    params: [
      {
        name: "FuzzyY",
        default: 0.0,
        min: 0.0,
        max: 1.0,
        step: 1.0,
        type: "double",
        tip: "1 makes the side damping signed — asymmetric growth.",
      },
      {
        name: "FuzzyZ",
        default: 0.0,
        min: 0.0,
        max: 1.0,
        step: 1.0,
        type: "double",
        tip: "1 makes the depth damping signed — asymmetric growth.",
      },
      {
        name: "Limiter",
        default: 0.01,
        min: 0.01,
        max: 1.0,
        step: 0.01,
        type: "double",
        tip: "Softens the damping poles; bigger = smoother, tamer shape.",
      },
    ],
    // Makin3D-3-4 (B2 wave 4): the Makin square with per-lane rational
    // damping. With mz = z² (or z·|z| when FuzzyY=1) and my = y² (or y·|y|
    // when FuzzyZ=1), L = |Limiter|:
    //   x' = x² − y² − z²
    //   y' = 2xy · (1 − mz/(x² + y² + L))
    //   z' = 2xz · (1 − my/(x² + z² + L))
    // (MB3D's Fuzzy_* params are tri-state selectors whose >0 and =0 branches
    // are both the plain square — only <0 differs, so they're booleans here:
    // 1 = signed square.) No analytic dr: w untouched, numeric DE.
    wgsl: `
        let fy = i32(op.p0 + 0.5);
        let fz = i32(op.p1 + 0.5);
        let fl = abs(op.p2);
        let ax = pos.x; let ay = pos.y; let az = pos.z;
        var mz = az * az;
        if (fy == 1 && az <= 0.0) { mz = -mz; }
        var my = ay * ay;
        if (fz == 1 && ay <= 0.0) { my = -my; }
        pos = vec3f(ax * ax - ay * ay - az * az,
                    2.0 * ax * ay * (1.0 - mz / (ax * ax + ay * ay + fl)),
                    2.0 * ax * az * (1.0 - my / (ax * ax + az * az + fl)));`,
    glsl: (v) => `
    // Makin fuzzy square (Makin3D-3-4) — numeric DE (w untouched).
    {
        int fy = int(${v[0]} + 0.5);
        int fz = int(${v[1]} + 0.5);
        float fl = abs(${v[2]});
        float ax = pos.x, ay = pos.y, az = pos.z;
        float mz = az * az;
        if (fy == 1 && az <= 0.0) mz = -mz;
        float my = ay * ay;
        if (fz == 1 && ay <= 0.0) my = -my;
        pos = vec3(ax * ax - ay * ay - az * az,
                   2.0 * ax * ay * (1.0 - mz / (ax * ax + ay * ay + fl)),
                   2.0 * ax * az * (1.0 - my / (ax * ax + az * az + fl)));
    }`,
  },
  {
    id: 47,
    key: "polygonFold",
    name: "Polygon Fold",
    wRule: W_MUL_K,
    deApprox: true,
    category: "warp",
    blurb:
      "Reshapes round cross-sections into polygon ones (or back) — hexagonal pipes, star columns. The surface is approximate: lower Detail if it looks foggy.",
    params: [
      {
        name: "Sides",
        default: 6.0,
        min: 3.0,
        max: 12.0,
        step: 1.0,
        type: "double",
        tip: "How many polygon sides (6 = hexagon).",
      },
      {
        name: "Strength",
        default: 1.0,
        min: -1.0,
        max: 1.0,
        step: 0.05,
        type: "double",
        tip: "+1 pulls the shape toward a circle cross-section, −1 pushes toward the polygon; 0 does nothing.",
      },
      {
        name: "Axis",
        default: 0.0,
        min: 0.0,
        max: 2.0,
        step: 1.0,
        type: "double",
        tip: "Which axis runs free: 0 up, 1 side, 2 front.",
      },
    ],
    // Polygon↔circle radial remap (Phase C, the FIRST deApprox op —
    // _BPolygonToCircle/FromCircle/_ngon family). Fold the plane angle into
    // one of N sectors (centered via round(), NOT the corpus decompiles'
    // linearized mod-loop — they carry the guard-vs-0 trap), then reshape the
    // RADIUS by the local cosine: a regular N-gon boundary sits at
    // R_in/cos(θ_local), so r·mix(1, cosθ, s) maps polygon→circle for s>0 and
    // r·mix(1, 1/cosθ, −s) maps circle→polygon for s<0. The ANGLE is kept —
    // this reshapes radius, unlike kaleido's reflection. w *= f is the
    // best-effort W_MUL_K accounting (the radial stretch), but f varies with
    // θ so the bound is approximate — hence deApprox: the render policy
    // tightens deScale (APPROX_DE.md) and the badge shows amber.
    wgsl: `
        let pfn = max(f32(i32(op.p0 + 0.5)), 3.0);
        let pfs = op.p1;
        let pfm = i32(op.p2 + 0.5);
        var u = pos.x; var v = pos.y;
        if (pfm == 1) { u = pos.z; v = pos.x; }
        else if (pfm == 2) { u = pos.y; v = pos.z; }
        let pr2 = u * u + v * v;
        if (pr2 > 1e-24) {
          let sector = 6.283185307179586 / pfn;
          let a = atan2(v, u);
          let th = a - sector * floor(a / sector + 0.5);
          let c = max(cos(th), 1e-6);
          var f = 1.0 + pfs * (c - 1.0);
          if (pfs < 0.0) { f = 1.0 + (-pfs) * (1.0 / c - 1.0); }
          u = u * f; v = v * f;
          w = w * abs(f);
          if (pfm == 1) { pos = vec3f(v, pos.y, u); }
          else if (pfm == 2) { pos = vec3f(pos.x, u, v); }
          else { pos = vec3f(u, v, pos.z); }
        }`,
    glsl: (v) => `
    // Polygon fold — polygon↔circle radial remap (approximate DE; w *= f best-effort).
    {
        float pfn = max(float(int(${v[0]} + 0.5)), 3.0);
        float pfs = ${v[1]};
        int pfm = int(${v[2]} + 0.5);
        float u = pos.x, vv = pos.y;
        if (pfm == 1) { u = pos.z; vv = pos.x; }
        else if (pfm == 2) { u = pos.y; vv = pos.z; }
        float pr2 = u * u + vv * vv;
        if (pr2 > 1e-24) {
            float sector = 6.283185307179586 / pfn;
            float a = atan(vv, u);
            float th = a - sector * floor(a / sector + 0.5);
            float c = max(cos(th), 1e-6);
            float f = pfs < 0.0 ? 1.0 + (-pfs) * (1.0 / c - 1.0)
                                : 1.0 + pfs * (c - 1.0);
            u *= f; vv *= f;
            w *= abs(f);
            if (pfm == 1) pos = vec3(vv, pos.y, u);
            else if (pfm == 2) pos = vec3(pos.x, u, vv);
            else pos = vec3(u, vv, pos.z);
        }
    }`,
  },
  {
    id: 48,
    key: "toCoord",
    name: "To Coordinates",
    wRule: W_UNCHANGED,
    deApprox: true,
    category: "warp",
    blurb:
      "Unrolls space into cylinder, sphere, torus or log-polar coordinates — folds applied after it wrap around the shape. Pair with From Coordinates.",
    params: [
      {
        name: "System",
        default: 0.0,
        min: 0.0,
        max: 4.0,
        step: 1.0,
        type: "double",
        tip: "0 cylinder, 1 sphere, 2 torus, 3 torus-tube, 4 log-polar.",
      },
      {
        name: "R",
        default: 2.0,
        min: 0.0,
        max: 4.0,
        step: 0.05,
        type: "double",
        tip: "Torus major radius (systems 2–3 only).",
      },
      {
        name: "Gamma",
        default: 0.0,
        min: -3.14,
        max: 3.14,
        step: 0.01,
        type: "double",
        tip: "Tube-angle phase (system 3); log-spiral pitch (system 4).",
      },
    ],
    // Curvilinear frame change TO (ρ/θ/z · r/θ/φ · torus) — the MB3D
    // _tocylindrical/_tospherical/_totorical/_totorical2 lane conventions
    // EXACTLY (round-trips with fromCoord must cancel). Angle lanes hold
    // radians, not lengths, so any op sandwiched between to/from acts in a
    // non-isometric frame: deApprox (the identity round-trip alone is exact).
    // w untouched, like the source transforms.
    //
    // System 4 (log-polar, parity wave 1) is NOT an MB3D lane — it is the
    // textbook complex logarithm w = log z written in the same lane layout:
    //   x ← ln ρ,  y ← θ − Γ·ln ρ,  z ← z.
    // Γ = 0 gives the plain conformal log-polar unwrap (self-similar radial
    // scaling by λ becomes a translation of ln λ along lane x, so a modFold
    // after it tiles a self-similar spiral). Γ ≠ 0 shears the angle lane by the
    // log-radius, which straightens the logarithmic spiral ρ = e^(θ/Γ) into a
    // line — the "log-spiral unwrap" pre-step (PRIMITIVE_COVERAGE_PLAN.md
    // pre-step pass; transLogSpIFS/translogsp4ifs). The shear is inverted
    // exactly by fromCoord's +Γ·x, so the round-trip still cancels.
    // ln ρ is floored at ρ = 1e-12 (→ −27.6) so the origin stays finite.
    wgsl: `
        let cs = i32(op.p0 + 0.5);
        let cR = op.p1;
        let cG = op.p2;
        let cx = pos.x; let cy = pos.y; let cz = pos.z;
        if (cs == 1) {
          pos = vec3f(sqrt(cx * cx + cy * cy + cz * cz),
                      atan2(sqrt(cx * cx + cy * cy), cz),
                      atan2(cy, cx));
        } else if (cs == 2) {
          pos = vec3f(cR - sqrt(cx * cx + cy * cy), atan2(cy, cx), cz);
        } else if (cs == 3) {
          let rho = sqrt(cx * cx + cy * cy) - cR;
          pos = vec3f(sqrt(rho * rho + cz * cz),
                      atan2(rho, cz) - cG - 1.5707963267948966,
                      atan2(cy, cx));
        } else if (cs == 4) {
          let lr = log(max(sqrt((cx * cx) + (cy * cy)), 1e-12));
          pos = vec3f(lr, atan2(cy, cx) - (cG * lr), cz);
        } else {
          pos = vec3f(sqrt(cx * cx + cy * cy), atan2(cy, cx), cz);
        }`,
    glsl: (v) => `
    // To coordinates (0 cyl / 1 sph / 2 tor / 3 tor2 / 4 log-polar) — approximate-DE frame change.
    {
        int cs = int(${v[0]} + 0.5);
        float cR = ${v[1]};
        float cG = ${v[2]};
        float cx = pos.x, cy = pos.y, cz = pos.z;
        if (cs == 1) {
            pos = vec3(sqrt(cx * cx + cy * cy + cz * cz),
                       atan(sqrt(cx * cx + cy * cy), cz),
                       atan(cy, cx));
        } else if (cs == 2) {
            pos = vec3(cR - sqrt(cx * cx + cy * cy), atan(cy, cx), cz);
        } else if (cs == 3) {
            float rho = sqrt(cx * cx + cy * cy) - cR;
            pos = vec3(sqrt(rho * rho + cz * cz),
                       atan(rho, cz) - cG - 1.5707963267948966,
                       atan(cy, cx));
        } else if (cs == 4) {
            float lr = log(max(sqrt((cx * cx) + (cy * cy)), 1e-12));
            pos = vec3(lr, atan(cy, cx) - (cG * lr), cz);
        } else {
            pos = vec3(sqrt(cx * cx + cy * cy), atan(cy, cx), cz);
        }
    }`,
  },
  {
    id: 49,
    key: "fromCoord",
    name: "From Coordinates",
    wRule: W_UNCHANGED,
    deApprox: true,
    category: "warp",
    blurb:
      "Rolls cylinder, sphere, torus or log-polar coordinates back into normal space — closes a To Coordinates sandwich.",
    params: [
      {
        name: "System",
        default: 0.0,
        min: 0.0,
        max: 4.0,
        step: 1.0,
        type: "double",
        tip: "0 cylinder, 1 sphere, 2 torus, 3 torus-tube, 4 log-polar — match the To move.",
      },
      {
        name: "R",
        default: 2.0,
        min: 0.0,
        max: 4.0,
        step: 0.05,
        type: "double",
        tip: "Torus major radius (systems 2–3 only).",
      },
      {
        name: "Gamma",
        default: 0.0,
        min: -3.14,
        max: 3.14,
        step: 0.01,
        type: "double",
        tip: "Tube-angle phase (system 3); log-spiral pitch (system 4).",
      },
    ],
    // Inverse frame change — MB3D _invcylindrical/_invspherical/_invtorical/
    // _invtorical2 lanes exactly. One deliberate rewrite: _invtorical's
    // q·tan(y) equals |R−x|·sin(y)·sign(cos y) algebraically — the tan form
    // goes 0·∞ = NaN exactly at |cos y| = 0, the sin form doesn't. (System 3
    // round-trips with a z-mirror; the source pair does the same.) deApprox;
    // w untouched.
    //
    // System 4 (log-polar) is the complex exponential z = e^w, the exact
    // inverse of toCoord's complex log: ρ = e^x, θ = y + Γ·x (the +Γ·x undoes
    // toCoord's −Γ·ln ρ shear, so to→from cancels for every Γ). The exponent
    // is capped at 60 (e^60 ≈ 1.1e26, far past any bailout) so a runaway x
    // lane saturates instead of returning inf — the cap must match toCoord's
    // ρ floor in all three emitters or the tiers disagree.
    wgsl: `
        let cs = i32(op.p0 + 0.5);
        let cR = op.p1;
        let cG = op.p2;
        let cx = pos.x; let cy = pos.y; let cz = pos.z;
        if (cs == 1) {
          pos = vec3f(cos(cz) * sin(cy) * cx,
                      sin(cz) * sin(cy) * cx,
                      cos(cy) * cx);
        } else if (cs == 2) {
          let q = abs(cR - cx);
          var sg = 1.0;
          if (cos(cy) < 0.0) { sg = -1.0; }
          pos = vec3f(q * abs(cos(cy)), q * sin(cy) * sg, cz);
        } else if (cs == 3) {
          let t = cx * cos(cy + cG) + cR;
          pos = vec3f(t * cos(cz), t * sin(cz), cx * sin(cy + cG));
        } else if (cs == 4) {
          let le = exp(min(cx, 60.0));
          let la = cy + (cG * cx);
          pos = vec3f(le * cos(la), le * sin(la), cz);
        } else {
          pos = vec3f(cos(cy) * cx, sin(cy) * cx, cz);
        }`,
    glsl: (v) => `
    // From coordinates (0 cyl / 1 sph / 2 tor / 3 tor2 / 4 log-polar) — approximate-DE frame change.
    {
        int cs = int(${v[0]} + 0.5);
        float cR = ${v[1]};
        float cG = ${v[2]};
        float cx = pos.x, cy = pos.y, cz = pos.z;
        if (cs == 1) {
            pos = vec3(cos(cz) * sin(cy) * cx,
                       sin(cz) * sin(cy) * cx,
                       cos(cy) * cx);
        } else if (cs == 2) {
            float q = abs(cR - cx);
            float sg = cos(cy) < 0.0 ? -1.0 : 1.0;
            pos = vec3(q * abs(cos(cy)), q * sin(cy) * sg, cz);
        } else if (cs == 3) {
            float t = cx * cos(cy + cG) + cR;
            pos = vec3(t * cos(cz), t * sin(cz), cx * sin(cy + cG));
        } else if (cs == 4) {
            float le = exp(min(cx, 60.0));
            float la = cy + (cG * cx);
            pos = vec3(le * cos(la), le * sin(la), cz);
        } else {
            pos = vec3(cos(cy) * cx, sin(cy) * cx, cz);
        }
    }`,
  },
  {
    id: 50,
    key: "gnarl2D",
    name: "Gnarl 2D",
    wRule: W_UNCHANGED,
    deApprox: true,
    category: "warp",
    blurb:
      "The classic gnarl warp in the ground plane — melts straight edges into twisted, root-like tangles. Approximate DE: keep Step small.",
    params: [
      {
        name: "Step",
        default: 0.1,
        min: -0.5,
        max: 0.5,
        step: 0.01,
        type: "double",
        tip: "Warp strength per iteration (0.1 is plenty; the DE loosens fast).",
      },
      {
        name: "Alpha",
        default: 3.0,
        min: 0.0,
        max: 6.0,
        step: 0.05,
        type: "double",
        tip: "Outer wave frequency.",
      },
      {
        name: "Beta",
        default: 3.0,
        min: 0.0,
        max: 6.0,
        step: 0.05,
        type: "double",
        tip: "Inner wave frequency.",
      },
    ],
    // Gnarl (Phase C): the classic nested-sine displacement
    //   g(b) = sin(sin((sin(b·Beta) + b) · Alpha) + b)
    // cross-coupled in the XY plane (x displaced by g(old y), y by g(old x)),
    // z free. Implemented from the published gnarl construction, NOT the
    // corpus decompiles (all carry #84 — lost trig range reduction).
    // Unbounded compounding derivative: THE canonical approximate-DE warp
    // (MB3D ships it DEscale 0). w untouched; the deApprox policy tightens
    // the march.
    wgsl: `
        let gs = op.p0;
        let ga = op.p1;
        let gb = op.p2;
        let gx = pos.x; let gy = pos.y;
        pos.x = gx - gs * sin(sin((sin(gy * gb) + gy) * ga) + gy);
        pos.y = gy - gs * sin(sin((sin(gx * gb) + gx) * ga) + gx);`,
    glsl: (v) => `
    // Gnarl 2D — nested-sine XY warp (approximate DE; w untouched).
    {
        float gs = ${v[0]};
        float ga = ${v[1]};
        float gb = ${v[2]};
        float gx = pos.x, gy = pos.y;
        pos.x = gx - gs * sin(sin((sin(gy * gb) + gy) * ga) + gy);
        pos.y = gy - gs * sin(sin((sin(gx * gb) + gx) * ga) + gx);
    }`,
  },
  {
    id: 51,
    key: "gnarl3D",
    name: "Gnarl 3D",
    wRule: W_UNCHANGED,
    deApprox: true,
    category: "warp",
    blurb:
      "The gnarl warp in all three axes (cyclic coupling) — fully melted, organic tangles. Approximate DE: keep Step small.",
    params: [
      {
        name: "Step",
        default: 0.1,
        min: -0.5,
        max: 0.5,
        step: 0.01,
        type: "double",
        tip: "Warp strength per iteration (0.1 is plenty; the DE loosens fast).",
      },
      {
        name: "Alpha",
        default: 3.0,
        min: 0.0,
        max: 6.0,
        step: 0.05,
        type: "double",
        tip: "Outer wave frequency.",
      },
      {
        name: "Beta",
        default: 3.0,
        min: 0.0,
        max: 6.0,
        step: 0.05,
        type: "double",
        tip: "Inner wave frequency.",
      },
    ],
    // Gnarl 3D: same g(b) as gnarl2D with the corpus's cyclic coupling
    // (x displaced by g(old z), y by g(old x), z by g(old y)). Same
    // published-math ruling and approximate-DE treatment.
    wgsl: `
        let gs = op.p0;
        let ga = op.p1;
        let gb = op.p2;
        let gx = pos.x; let gy = pos.y; let gz = pos.z;
        pos.x = gx - gs * sin(sin((sin(gz * gb) + gz) * ga) + gz);
        pos.y = gy - gs * sin(sin((sin(gx * gb) + gx) * ga) + gx);
        pos.z = gz - gs * sin(sin((sin(gy * gb) + gy) * ga) + gy);`,
    glsl: (v) => `
    // Gnarl 3D — cyclic nested-sine warp (approximate DE; w untouched).
    {
        float gs = ${v[0]};
        float ga = ${v[1]};
        float gb = ${v[2]};
        float gx = pos.x, gy = pos.y, gz = pos.z;
        pos.x = gx - gs * sin(sin((sin(gz * gb) + gz) * ga) + gz);
        pos.y = gy - gs * sin(sin((sin(gx * gb) + gx) * ga) + gx);
        pos.z = gz - gs * sin(sin((sin(gy * gb) + gy) * ga) + gy);
    }`,
  },
  {
    id: 52,
    key: "asinhWarp",
    name: "Asinh Warp",
    wRule: W_UNCHANGED,
    deApprox: true,
    category: "warp",
    blurb:
      "Compresses one axis on a gentle log curve (asinh) — squashes far structure while keeping the center intact. Approximate DE.",
    params: [
      {
        name: "Axis",
        default: 1.0,
        min: 0.0,
        max: 2.0,
        step: 1.0,
        type: "double",
        tip: "Which axis to warp: 0 up, 1 side, 2 front.",
      },
      {
        name: "Mul",
        default: -2.0,
        min: -4.0,
        max: 4.0,
        step: 0.05,
        type: "double",
        tip: "Input scale before the curve.",
      },
      {
        name: "Base",
        default: 0.37,
        min: -2.0,
        max: 2.0,
        step: 0.01,
        type: "double",
        tip: "Output scale after the curve.",
      },
    ],
    // Per-axis asinh warp (MB3D _SinhY/_SinhZ — note _SinhX's decompile is a
    // byte-copy of _LogX and is NOT a source): with t = Mul·a,
    //   a' = Base · log2(t + √(t²+1))
    // (the corpus's fixA/fixLg offsets default to 0 and are dropped — the
    // ≤3-param encoding). asinh's derivative is bounded, so this is the
    // mildest warp in the family, but still non-conformal: deApprox,
    // w untouched.
    wgsl: `
        let m = i32(op.p0 + 0.5);
        var a = pos.y;
        if (m == 0) { a = pos.z; } else if (m == 2) { a = pos.x; }
        let t = a * op.p1;
        let o = op.p2 * log2(t + sqrt(t * t + 1.0));
        if (m == 0) { pos.z = o; } else if (m == 2) { pos.x = o; } else { pos.y = o; }`,
    glsl: (v) => `
    // Asinh warp — per-axis log-curve compression (approximate DE; w untouched).
    {
        int m = int(${v[0]} + 0.5);
        float a = m == 0 ? pos.z : (m == 2 ? pos.x : pos.y);
        float t = a * ${v[1]};
        float o = ${v[2]} * log2(t + sqrt(t * t + 1.0));
        if (m == 0) pos.z = o; else if (m == 2) pos.x = o; else pos.y = o;
    }`,
  },
  {
    id: 53,
    key: "logWarp",
    name: "Log Warp",
    wRule: W_UNCHANGED,
    deApprox: true,
    category: "warp",
    blurb:
      "Crushes one axis onto a logarithmic scale — extreme flattening with a sharp crease at the center. Approximate DE.",
    params: [
      {
        name: "Axis",
        default: 0.0,
        min: 0.0,
        max: 2.0,
        step: 1.0,
        type: "double",
        tip: "Which axis to warp: 0 up, 1 side, 2 front.",
      },
      {
        name: "Mul",
        default: 1.0,
        min: -4.0,
        max: 4.0,
        step: 0.05,
        type: "double",
        tip: "Input scale before the log.",
      },
      {
        name: "Base",
        default: 0.37,
        min: -2.0,
        max: 2.0,
        step: 0.01,
        type: "double",
        tip: "Output scale after the log.",
      },
    ],
    // Per-axis log warp (MB3D _LogX/Y/Z): a' = Base · log2(|Mul·a| + 0.01).
    // The 0.01 is the corpus's fixLg log-guard at its default, baked; fixA
    // (default 0) is dropped — the ≤3-param encoding. The corpus decompiles
    // double-apply (Mul, fixA); this is the intended single application.
    // Unbounded derivative at the crease: deApprox, w untouched.
    wgsl: `
        let m = i32(op.p0 + 0.5);
        var a = pos.z;
        if (m == 1) { a = pos.y; } else if (m == 2) { a = pos.x; }
        let o = op.p2 * log2(abs(a * op.p1) + 0.01);
        if (m == 1) { pos.y = o; } else if (m == 2) { pos.x = o; } else { pos.z = o; }`,
    glsl: (v) => `
    // Log warp — per-axis logarithmic crush (approximate DE; w untouched).
    {
        int m = int(${v[0]} + 0.5);
        float a = m == 1 ? pos.y : (m == 2 ? pos.x : pos.z);
        float o = ${v[2]} * log2(abs(a * ${v[1]}) + 0.01);
        if (m == 1) pos.y = o; else if (m == 2) pos.x = o; else pos.z = o;
    }`,
  },
  {
    id: 54,
    key: "neoSqrWarp",
    name: "NeoSqr Warp",
    wRule: W_UNCHANGED,
    deApprox: true,
    category: "warp",
    blurb:
      "Bends one axis through a signed parabola — pinches the middle and flips the far ends. Approximate DE.",
    params: [
      {
        name: "Axis",
        default: 1.0,
        min: 0.0,
        max: 2.0,
        step: 1.0,
        type: "double",
        tip: "Which axis to warp: 0 up, 1 side, 2 front.",
      },
      {
        name: "FixSq",
        default: 1.0,
        min: -4.0,
        max: 4.0,
        step: 0.05,
        type: "double",
        tip: "Where the parabola peaks — the fold-back point.",
      },
      {
        name: "Mul",
        default: 1.0,
        min: -4.0,
        max: 4.0,
        step: 0.05,
        type: "double",
        tip: "Input scale before the curve.",
      },
    ],
    // Per-axis NeoSqr warp (MB3D _NeoSqrX/Y/Z): with t = Mul·a,
    //   a' = t≥0 ? t·(FixSq − t) : t·(t − FixSq)
    // (the corpus's fixA offset and Div divisor default to 0 and 1 and are
    // dropped — the ≤3-param encoding). Quadratic derivative: deApprox,
    // w untouched.
    wgsl: `
        let m = i32(op.p0 + 0.5);
        var a = pos.y;
        if (m == 0) { a = pos.z; } else if (m == 2) { a = pos.x; }
        let t = a * op.p2;
        var o = t * (t - op.p1);
        if (t >= 0.0) { o = t * (op.p1 - t); }
        if (m == 0) { pos.z = o; } else if (m == 2) { pos.x = o; } else { pos.y = o; }`,
    glsl: (v) => `
    // NeoSqr warp — per-axis signed parabola (approximate DE; w untouched).
    {
        int m = int(${v[0]} + 0.5);
        float a = m == 0 ? pos.z : (m == 2 ? pos.x : pos.y);
        float t = a * ${v[2]};
        float o = t >= 0.0 ? t * (${v[1]} - t) : t * (t - ${v[1]});
        if (m == 0) pos.z = o; else if (m == 2) pos.x = o; else pos.y = o;
    }`,
  },
  {
    id: 55,
    key: "sinShear",
    name: "Sine Shear",
    wRule: W_UNCHANGED,
    deApprox: true,
    category: "warp",
    blurb:
      "Slides one axis by a sine wave of another — ripples and wobbles without changing volume. Approximate DE (mild).",
    params: [
      {
        name: "Pair",
        default: 0.0,
        min: 0.0,
        max: 2.0,
        step: 1.0,
        type: "double",
        tip: "Which pair shears: 0 side←front, 1 up←side, 2 front←up.",
      },
      {
        name: "Amp",
        default: 1.0,
        min: -2.0,
        max: 2.0,
        step: 0.05,
        type: "double",
        tip: "Wave height (1 with Freq 1 = the classic).",
      },
      {
        name: "Freq",
        default: 1.0,
        min: 0.0,
        max: 6.0,
        step: 0.05,
        type: "double",
        tip: "Wave frequency.",
      },
    ],
    // Cross-axis sine shear (MB3D _YplusSinZ family): a += Amp·sin(Freq·b),
    // pair-selected (0: y += f(z) — the corpus default; 1: z += f(y);
    // 2: x += f(z))... pairs chosen so each axis can ripple. Volume-preserving
    // shear with |∂| ≤ Amp·Freq — the mildest warp here, still non-conformal:
    // deApprox, w untouched. Amp=1, Freq=1, Pair=0 is exactly _YplusSinZ.
    wgsl: `
        let m = i32(op.p0 + 0.5);
        if (m == 1) { pos.z = pos.z + op.p1 * sin(op.p2 * pos.y); }
        else if (m == 2) { pos.x = pos.x + op.p1 * sin(op.p2 * pos.z); }
        else { pos.y = pos.y + op.p1 * sin(op.p2 * pos.z); }`,
    glsl: (v) => `
    // Sine shear — cross-axis ripple (approximate DE; w untouched).
    {
        int m = int(${v[0]} + 0.5);
        if (m == 1) pos.z += ${v[1]} * sin(${v[2]} * pos.y);
        else if (m == 2) pos.x += ${v[1]} * sin(${v[2]} * pos.z);
        else pos.y += ${v[1]} * sin(${v[2]} * pos.z);
    }`,
  },
  {
    id: 56,
    key: "smoothBoxFold",
    name: "Smooth Box Fold",
    wRule: W_UNCHANGED,
    deApprox: true,
    category: "warp",
    blurb:
      "A box fold with rounded creases — the Mandelbox crinkle without the hard edges. Approximate DE.",
    params: [
      {
        name: "Fold",
        default: 1.0,
        min: 0.0,
        max: 2.0,
        step: 0.05,
        type: "double",
        tip: "Fold limit (1 = the classic box fold's).",
      },
      {
        name: "Sharpness",
        default: 6.0,
        min: 1.0,
        max: 12.0,
        step: 1.0,
        type: "double",
        tip: "How tight the rounded crease is (higher = closer to the hard fold).",
      },
      {
        name: "Fix",
        default: 1.0,
        min: 0.0,
        max: 4.0,
        step: 0.05,
        type: "double",
        tip: "Blend strength of the smoothing term.",
      },
    ],
    // Smooth box fold (MB3D ABoxSmoothFold's per-axis half, hand-crafted
    // readable source): per axis, with s = |a|^Sharpness · Fix,
    //   a' = (|a| + (2·Fold − |a|)·s) / (s + 1)
    // — a C¹ blend that maps into the positive octant (no sign restore, like
    // the source). The local derivative ≠ the piecewise fold factor: deApprox,
    // w untouched. ABoxSmoothFold ≈ this → smoothBallFold → scale (recipe).
    wgsl: `
        let sF = op.p0;
        let sSh = op.p1;
        let sFx = op.p2;
        let sfx = pow(abs(pos.x), sSh) * sFx;
        let sfy = pow(abs(pos.y), sSh) * sFx;
        let sfz = pow(abs(pos.z), sSh) * sFx;
        pos = vec3f(
          (abs(pos.x) + (sF + sF - abs(pos.x)) * sfx) / (sfx + 1.0),
          (abs(pos.y) + (sF + sF - abs(pos.y)) * sfy) / (sfy + 1.0),
          (abs(pos.z) + (sF + sF - abs(pos.z)) * sfz) / (sfz + 1.0));`,
    glsl: (v) => `
    // Smooth box fold — C¹ rounded crease (approximate DE; w untouched).
    {
        float sF = ${v[0]};
        float sSh = ${v[1]};
        float sFx = ${v[2]};
        float sfx = pow(abs(pos.x), sSh) * sFx;
        float sfy = pow(abs(pos.y), sSh) * sFx;
        float sfz = pow(abs(pos.z), sSh) * sFx;
        pos = vec3(
            (abs(pos.x) + (sF + sF - abs(pos.x)) * sfx) / (sfx + 1.0),
            (abs(pos.y) + (sF + sF - abs(pos.y)) * sfy) / (sfy + 1.0),
            (abs(pos.z) + (sF + sF - abs(pos.z)) * sfz) / (sfz + 1.0));
    }`,
  },
  {
    id: 57,
    key: "smoothBallFold",
    name: "Smooth Ball Fold",
    wRule: W_MUL_K,
    deApprox: true,
    category: "warp",
    blurb:
      "A sphere fold with a soft radial blend instead of the hard inversion switch — organic shells. Approximate DE.",
    params: [
      {
        name: "MinRsq",
        default: 0.25,
        min: 0.0,
        max: 0.99,
        step: 0.01,
        type: "double",
        tip: "Inner radius² of the blend (0.25 = the classic sphere fold's).",
      },
      {
        name: "Sharpness",
        default: 4.0,
        min: 1.0,
        max: 12.0,
        step: 1.0,
        type: "double",
        tip: "How tight the radial blend is.",
      },
      {
        name: "Fix",
        default: 0.3,
        min: 0.0,
        max: 4.0,
        step: 0.01,
        type: "double",
        tip: "Blend strength of the smoothing term.",
      },
    ],
    // Smooth ball fold (ABoxSmoothFold's radial half): with r² = |p|²,
    // center = (1+MinRsq)/2, half = (1−MinRsq)/2, n = |r²−center|/half and
    // b = √n^Sharpness · Fix, the blended radial substitute is
    //   m = center − half·(b + n)/(1 + b)      (m = 1 when MinRsq ≥ 0.99)
    // and the op multiplies pos AND w by 1/max(|m|, ε) — conformal per-point
    // (W_MUL_K accounting), approximate at the blend: deApprox. Compose with
    // a plain scale for the source's Scale/|m| — ABoxSmoothFold's recipe.
    wgsl: `
        let bMr = op.p0;
        let bSh = op.p1;
        let bFx = op.p2;
        let br2 = dot(pos, pos);
        let bc = (1.0 + bMr) * 0.5;
        let bh = max((1.0 - bMr) * 0.5, 1e-20);
        let bn = abs(br2 - bc) / bh;
        var bm = 1.0;
        if (bMr < 0.99) {
          let bb = pow(sqrt(bn), bSh) * bFx;
          bm = bc - bh * (bb + bn) / (1.0 + bb);
        }
        let bk = 1.0 / max(abs(bm), 1e-20);
        pos = pos * bk;
        w = w * bk;`,
    glsl: (v) => `
    // Smooth ball fold — soft radial blend (approximate DE; w *= k).
    {
        float bMr = ${v[0]};
        float bSh = ${v[1]};
        float bFx = ${v[2]};
        float br2 = dot(pos, pos);
        float bc = (1.0 + bMr) * 0.5;
        float bh = max((1.0 - bMr) * 0.5, 1e-20);
        float bn = abs(br2 - bc) / bh;
        float bm = 1.0;
        if (bMr < 0.99) {
            float bb = pow(sqrt(bn), bSh) * bFx;
            bm = bc - bh * (bb + bn) / (1.0 + bb);
        }
        float bk = 1.0 / max(abs(bm), 1e-20);
        pos *= bk;
        w   *= bk;
        g_wq *= bk;
    }`,
  },
  {
    id: 58,
    key: "torusInvert",
    name: "Torus Inversion",
    wRule: W_MUL_K,
    deApprox: true,
    category: "sphere",
    blurb:
      "Turns space inside-out through a torus instead of a sphere — bubbles thread a ring. Needs a bounding fold. Approximate DE.",
    params: [
      {
        name: "Radius",
        default: 1.0,
        min: 0.05,
        max: 4.0,
        step: 0.05,
        type: "double",
        tip: "Tube radius of the inverting torus — the circle the cross-section inverts through.",
      },
      {
        name: "R",
        default: 2.0,
        min: 0.0,
        max: 4.0,
        step: 0.05,
        type: "double",
        tip: "Torus major radius — the core circle the tube wraps around.",
      },
      {
        name: "Variant",
        default: 0.0,
        min: 0.0,
        max: 3.0,
        step: 1.0,
        type: "double",
        tip: "0 true tube inversion, 1–3 the pseudo forms (z divided / multiplied / unrooted).",
      },
    ],
    // Torus-space inversion — the _toruspinv family (PRIMITIVE_COVERAGE_PLAN.md
    // §3; the corpus ruling is explicit that radialInvert does NOT cover these,
    // correcting PRIMITIVE_GAPS §1 #2: op 28 is a pure SPHERE inversion).
    //
    // Every variant works in the meridian half-plane of the torus, in the same
    // lane convention toCoord system 2 already uses:
    //     ρ = |p.xy|,  u = R − ρ,  d² = u² + z²
    // so d is the distance from the point to the torus CORE CIRCLE of radius R.
    // Each variant produces a new (ρ′, z′) and the azimuth θ is preserved —
    // the map is a surface of revolution, so rebuilding p.xy from the unit
    // direction is exact (and is what keeps the ρ → 0 axis finite).
    //
    // Variant 0 — the real thing, and the default: inversion in the CIRCLE of
    // radius `Radius` centred on the core circle, i.e. classical 2D inversive
    // geometry applied to the meridian section (MathWorld "Inversion":
    // OP·OP′ = r², so (u,z) → (u,z)·Radius²/d²). Bounded only when paired with
    // a fold, exactly like radialInvert.
    //   ⚠ It is involutive only while the image stays at ρ′ ≥ 0. Inversion
    //   sends points near the core circle towards infinity, and "far" in the
    //   −u direction means ρ′ = R − u·k goes NEGATIVE — geometrically the
    //   correct point, at azimuth θ+π (which `dir * ρ′` reproduces exactly,
    //   continuously through ρ′ = 0). But re-applying the op recomputes ρ from
    //   a hypot, which is unsigned, so the second pass starts from R − |ρ′|
    //   and does not return. That is a property of the ρ ≥ 0 meridian CHART,
    //   not a defect in the map; do not "fix" it by clamping ρ′, which would
    //   tear the field. Defaults (Radius 1, R 2) sit well inside the involutive
    //   region; the test pins both the involution there and OP·OP′ = Radius²
    //   everywhere.
    // Variants 1–3 — the cheap "pseudo torus inversion" forms the _toruspinv1/
    // 2/3 bodies use, described in the corpus notes as
    //   d = √((R−ρ)² + z²);  xy ·= Radius/d;  z ·= d/Radius   (variant 1)
    // with variant 2 multiplying z by d·Radius instead of dividing, and
    // variant 3 skipping the square root (using d² wherever variant 1 uses d).
    // RULING: the notes write the xy/z updates as |x,y| and |z|. That reads
    // either as "the vector (x,y)" or as an abs() fold; we implement the
    // no-abs, pure-scaling reading, because the abs reading is recoverable
    // exactly as `absXYZ` + `torusInvert` while the converse is not. Recorded
    // here so a later oracle pass can flip it deliberately rather than by
    // accident.
    //
    // W-RULE. Variant 0 is conformal *within the meridian plane* (isotropic
    // factor k = Radius²/d²) but the azimuthal direction scales by ρ′/ρ, so
    // the 3-D map is NOT conformal and has no single exact k — Liouville's
    // theorem says only spheres and planes can manage that. Variants 1–3 are
    // frankly anisotropic. w therefore tracks the factor applied to the
    // MERIDIAN/radial lane — the inversion's own scale, and the one that sets
    // the radial geometry — exactly as polygonFold tracks its radial remap `f`
    // and leaves the rest to deApprox.
    //   ⚠ Do NOT "improve" this to max(k, ρ′/ρ). The azimuthal factor ρ′/ρ
    //   diverges on the rotation axis (ρ → 0 with ρ′ finite), so a max() makes
    //   w explode for any orbit that passes near the axis; DE = r/w collapses,
    //   the marcher stops on its first step, and every scene renders as a flat
    //   wall. That was measured, not theorised — it is why this comment exists.
    wgsl: `
        let tRad = max(abs(op.p0), 1e-6);
        let tR = op.p1;
        let tv = i32(op.p2 + 0.5);
        let trho = sqrt((pos.x * pos.x) + (pos.y * pos.y));
        let tu = tR - trho;
        let td2 = max((tu * tu) + (pos.z * pos.z), 1e-12);
        var trhoN = trho;
        var tzN = pos.z;
        var tk = 1.0;
        if (tv == 1) {
          let td = sqrt(td2);
          let ts = tRad / td;
          trhoN = trho * ts;
          tzN = pos.z * (td / tRad);
          tk = ts;
        } else if (tv == 2) {
          let td = sqrt(td2);
          let ts = tRad / td;
          trhoN = trho * ts;
          tzN = pos.z * (td * tRad);
          tk = ts;
        } else if (tv == 3) {
          let ts = tRad / td2;
          trhoN = trho * ts;
          tzN = pos.z * (td2 / tRad);
          tk = ts;
        } else {
          let tki = (tRad * tRad) / td2;
          trhoN = tR - (tu * tki);
          tzN = pos.z * tki;
          tk = tki;
        }
        var tdx = 1.0;
        var tdy = 0.0;
        if (trho > 1e-9) { tdx = pos.x / trho; tdy = pos.y / trho; }
        pos = vec3f(tdx * trhoN, tdy * trhoN, tzN);
        w = w * tk;`,
    glsl: (v) => `
    // Torus inversion — meridian-plane inversion about the core circle
    // (0 true / 1–3 pseudo); approximate DE, w *= largest singular value.
    {
        float tRad = max(abs(${v[0]}), 1e-6);
        float tR = ${v[1]};
        int tv = int(${v[2]} + 0.5);
        float trho = sqrt((pos.x * pos.x) + (pos.y * pos.y));
        float tu = tR - trho;
        float td2 = max((tu * tu) + (pos.z * pos.z), 1e-12);
        float trhoN = trho;
        float tzN = pos.z;
        float tk = 1.0;
        if (tv == 1) {
            float td = sqrt(td2);
            float ts = tRad / td;
            trhoN = trho * ts;
            tzN = pos.z * (td / tRad);
            tk = ts;
        } else if (tv == 2) {
            float td = sqrt(td2);
            float ts = tRad / td;
            trhoN = trho * ts;
            tzN = pos.z * (td * tRad);
            tk = ts;
        } else if (tv == 3) {
            float ts = tRad / td2;
            trhoN = trho * ts;
            tzN = pos.z * (td2 / tRad);
            tk = ts;
        } else {
            float tki = (tRad * tRad) / td2;
            trhoN = tR - (tu * tki);
            tzN = pos.z * tki;
            tk = tki;
        }
        float tdx = 1.0, tdy = 0.0;
        if (trho > 1e-9) { tdx = pos.x / trho; tdy = pos.y / trho; }
        pos = vec3(tdx * trhoN, tdy * trhoN, tzN);
        w    *= tk;
        g_wq *= tk;
    }`,
  },
  {
    id: 59,
    key: "mandalayFold",
    name: "Mandalay Fold",
    wRule: W_UNCHANGED,
    category: "fold",
    blurb:
      "Sorts space into one octahedral wedge, then folds it twice against an offset pair — the Mandalay tower fold. Exact DE.",
    params: [
      {
        name: "Fold",
        default: 0.5,
        min: 0.0,
        max: 2.0,
        step: 0.05,
        type: "double",
        tip: "Offset of the two fold planes — sets the tower spacing.",
      },
      {
        name: "Gap",
        default: 0.1,
        min: 0.0,
        max: 2.0,
        step: 0.05,
        type: "double",
        tip: "How far the diagonal fold may push before it clamps — widens the gap between towers.",
      },
      {
        name: "ZFold",
        default: 0.0,
        min: 0.0,
        max: 2.0,
        step: 0.05,
        type: "double",
        tip: "Optional independent fold on the short axis; 0 leaves it alone.",
      },
    ],
    // ── The Mandalay fold ────────────────────────────────────────────────
    // CLEAN-ROOM PROVENANCE. Derived from the authors' own PROSE + pseudocode
    // in the original Fractal Forums announcement thread, ""New" fractal type;
    // Mandalay (in KIFS/ non KIFS versions)" — DarkBeam, 2015-03-08, with
    // knighty co-developing in the same thread. No Mandelbulber2 / MB3D source
    // and no decompiled corpus body was read (see PRIMITIVE_COVERAGE_PLAN.md
    // §"Non-goals"). Step-by-step attribution:
    //   • abs → octant fold, and the 3-comparator DESCENDING sort that DarkBeam
    //     names the "Kifs Octahedral fold" — the sort is knighty's contribution
    //     (thread reply #17, "a slight modification to make it more symmetric").
    //   • the offset PAIR (fo, g) and the clamped diagonal fold are knighty's
    //     DBKNFold (reply #8), whose own comment calls the clamped step an
    //     "Odd fold (like the mandelbox fold)".
    //   • leaving the short axis alone by default is DarkBeam's own correction
    //     (reply #24, "leave z alone"); the optional ZFold plane is his
    //     independent-z remark (replies #31/#33, MB3D name "ZFold").
    //
    // WHY THIS FORMULATION AND NOT DARKBEAM'S BRANCHED KIFS ONE. Both are
    // isometries, but DarkBeam's branch structure is deliberately
    // DISCONTINUOUS — the thread is a running argument about the resulting
    // "cuts" ("Still cuts but almost never noticeable"). A tear is an infinite
    // Lipschitz constant, which is a real DE hazard: the marcher can step
    // through the seam. knighty's formulation reaches the same octahedral
    // tower family with every junction continuous by construction (each
    // conditional is written as a max(0,·)/min(g,·) blend that is a no-op ON
    // the switching plane), so it ships unflagged instead of needing deApprox.
    //
    // W-RULE = W_UNCHANGED, and it is EXACT, not best-effort. Every step is a
    // reflection, a transposition, or a translation:
    //   abs                        reflection in a coordinate plane
    //   t = max(0, b−a); a+=t; b−=t   identity, or reflection in the plane a=b
    //   a ← |a−fo| − fo            reflection + translation
    //   t = min(g, max(0,a−b))     identity | transposition | translation
    //   b ← fo − |b−fo|            reflection + translation
    // so |Jacobian| = 1 everywhere it is differentiable and w is untouched —
    // the same contract as kaleido/icosaFold (pinned in the test).
    //
    // ⚠ ORDER IS LOAD-BEARING. The lanes are updated IN PLACE and each step
    // reads the previous step's output. The pre-#24 versions in the source
    // thread had exactly the bug of computing two output lanes from the same
    // stale input, which is rank-deficient and NOT an isometry. Mirror this
    // sequence verbatim in all three emitters.
    wgsl: `
        let mfo = op.p0;
        let mg = op.p1;
        let mzf = op.p2;
        var mn = abs(pos);
        var mt = max(0.0, mn.y - mn.x);
        mn.x = mn.x + mt; mn.y = mn.y - mt;
        mt = max(0.0, mn.z - mn.y);
        mn.y = mn.y + mt; mn.z = mn.z - mt;
        mt = max(0.0, mn.y - mn.x);
        mn.x = mn.x + mt; mn.y = mn.y - mt;
        mn.x = abs(mn.x - mfo) - mfo;
        mt = min(mg, max(0.0, mn.x - mn.y));
        mn.x = mn.x - mt; mn.y = mn.y + mt;
        mn.y = mfo - abs(mn.y - mfo);
        if (mzf > 0.0) { mn.z = min(mn.z, (2.0 * mzf) - mn.z); }
        pos = mn;`,
    glsl: (v) => `
    // Mandalay fold — octahedral sort + offset-pair folds (exact DE, isometry).
    {
        float mfo = ${v[0]};
        float mg = ${v[1]};
        float mzf = ${v[2]};
        vec3 mn = abs(pos);
        float mt = max(0.0, mn.y - mn.x);
        mn.x += mt; mn.y -= mt;
        mt = max(0.0, mn.z - mn.y);
        mn.y += mt; mn.z -= mt;
        mt = max(0.0, mn.y - mn.x);
        mn.x += mt; mn.y -= mt;
        mn.x = abs(mn.x - mfo) - mfo;
        mt = min(mg, max(0.0, mn.x - mn.y));
        mn.x -= mt; mn.y += mt;
        mn.y = mfo - abs(mn.y - mfo);
        if (mzf > 0.0) { mn.z = min(mn.z, (2.0 * mzf) - mn.z); }
        pos = mn;
    }`,
  },
  {
    id: 60,
    key: "brickFold",
    name: "Brick Fold (Stagger)",
    wRule: W_UNCHANGED,
    category: "symmetry",
    blurb:
      "Tiles space like a brick wall — every row slides sideways by a half cell instead of stacking in a plain grid. Stagger 0 gives an ordinary tiling.",
    params: [
      {
        name: "CellX",
        default: 4.0,
        min: 0.0,
        max: 8.0,
        step: 0.05,
        type: "double",
        tip: "Brick width along X (0 turns this axis off — no bricks without it).",
      },
      {
        name: "CellY",
        default: 4.0,
        min: 0.0,
        max: 8.0,
        step: 0.05,
        type: "double",
        tip: "Course height along Y — the row spacing (0 turns the rows off).",
      },
      {
        name: "Stagger",
        default: 2.0,
        min: -8.0,
        max: 8.0,
        step: 0.05,
        type: "double",
        tip: "How far each row slides along X. Half of CellX is the classic running-bond brick; 0 is a plain grid.",
      },
    ],
    // Brick / running-bond domain repetition — the MB3D tilingbrick family
    // (PRIMITIVE_COVERAGE_PLAN.md; corpus_coverage_2026-07-13.json entries
    // `tilingbrickIFS` and `tilingbrick2IFS`, both classed needs_op with the
    // note "brick tiling = modFold + PER-ROW stagger offset ... Candidate:
    // stagger param on modFold").
    //
    // ⚠ WHY THIS IS A NEW OP AND NOT A 4th modFold PARAM. modFold (id 17)
    // already spends all three of its slots on CellX/CellY/CellZ, and three is
    // a hard ABI ceiling, not a style rule: `struct Op { opType: u32, p0: f32,
    // p1: f32, p2: f32 }` (shader.js), OP_STRIDE = 16 (renderer.js), and
    // validateOperators() fails any op with >3 params (invariants.js). The
    // kaleido Mirror precedent the plan cites was a 2→3 change, which fit the
    // existing budget; 3→4 is a GPU struct-layout change touching the
    // perturbation twins, the df64 twins and the tiled-export path. A sibling
    // op costs one palette slot and nothing else — and it is discoverable by
    // name, which a fourth slider on "Mod Fold (Tile)" would not be.
    //
    // THE MAP. Standard 2-D domain repetition with a per-row phase shift:
    //     row = round(y / CellY)                    (which course we are in)
    //     y  ←  y − CellY·row                       (fold Y into its course)
    //     x  ←  (x + Stagger·row) mod± CellX        (fold X, phase-shifted)
    // using the same round-to-nearest `t − c·floor(t/c + 0.5)` convention as
    // modFold, so each cell is centred on the origin rather than starting at
    // it. The row index is taken from the ORIGINAL y, before the Y fold — that
    // is the whole content of the op; take it after and every row reads 0.
    //
    // WHY A CONTINUOUS Stagger COVERS BOTH CORPUS FILES. The source pair
    // differs only in an "+offset gate" (tilingbrick2IFS is noted as "the
    // variant without the +offset gate"), i.e. one applies the shift on odd
    // rows only, the other accumulates it every row. The accumulating form
    // SUBSUMES the gated one: x is folded mod CellX afterwards, so the applied
    // phase is (Stagger·row) mod CellX, and at Stagger = CellX/2 that is the
    // alternating 0, CellX/2, 0, CellX/2 … of the odd-row gate — bit-identical
    // behaviour from the general law. Other values give the herringbone /
    // running-shear tilings the gated form cannot reach, so one continuous
    // param is strictly more general than an enum here (contrast kleinPolyMap,
    // where the variants are genuinely different rational maps). Pinned by
    // test: `brickFold(cx, cy, cx/2)` matches an explicit odd-row gate.
    //
    // Stagger = 0 degenerates to modFold(CellX, CellY, 0) EXACTLY — the corpus
    // note's "plain-mod sub-case (stagger 0) is rotate+modFold exact". Pinned.
    // Z is deliberately untouched: compose with modFold for a third axis, and
    // with the rotate ops to pick the brick plane (the corpus recipe is
    // "rotate + modFold", so the rotation is already expected to be separate).
    //
    // W-RULE. Inside a cell, row and the fold index are constant, so the map
    // is a pure TRANSLATION — Jacobian = I, |J| = 1 exactly. w is untouched
    // and the DE stays exact, identically to modFold, which ships unflagged on
    // the same reasoning. (Like every domain repetition it is discontinuous at
    // the cell walls and the DE is only sound while the body fits inside a
    // cell; that is the standing modFold caveat, not a new one.) No deApprox.
    wgsl: `
        let bcx = op.p0;
        let bcy = op.p1;
        var brow = 0.0;
        if (bcy > 0.0) {
          brow = floor((pos.y / bcy) + 0.5);
          pos.y = pos.y - (bcy * brow);
        }
        if (bcx > 0.0) {
          let bx = pos.x + (op.p2 * brow);
          pos.x = bx - (bcx * floor((bx / bcx) + 0.5));
        }`,
    glsl: (v) => `
    // brick fold — domain repetition with a per-row X stagger (running bond).
    // Row index comes from the pre-fold y; Stagger 0 == modFold. Pure per-cell
    // translation, so w is untouched and the DE is exact inside a cell.
    {
        float bcx = ${v[0]};
        float bcy = ${v[1]};
        float brow = 0.0;
        if (bcy > 0.0) {
            brow = floor((pos.y / bcy) + 0.5);
            pos.y -= bcy * brow;
        }
        if (bcx > 0.0) {
            float bx = pos.x + (${v[2]} * brow);
            pos.x = bx - (bcx * floor((bx / bcx) + 0.5));
        }
    }`,
  },
  {
    id: 61,
    key: "complexMap",
    name: "Complex Map (Möbius)",
    wRule: W_UNCHANGED,
    deApprox: true,
    category: "warp",
    blurb:
      "Reads the XY plane as a complex number and bends it through a Möbius map raised to a power — swirled, lens-like whorls. Z rides through untouched. Approximate DE.",
    // (params below: Order, C, Variant — see the derivation block after them)
    params: [
      {
        name: "Order",
        default: 2.0,
        min: 1.0,
        max: 8.0,
        step: 1.0,
        type: "double",
        tip: "The power the plane is raised to before the bend — how many whorls come out.",
      },
      {
        name: "C",
        default: 1.0,
        min: -2.0,
        max: 2.0,
        step: 0.05,
        type: "double",
        tip: "Strength of the bend. 0 leaves the plane alone in Murl; 1 is the classical Cayley transform.",
      },
      {
        name: "Variant",
        default: 0.0,
        min: 0.0,
        max: 2.0,
        step: 1.0,
        type: "double",
        tip: "0 Cayley (folds the plane into a disc), 1 Murl, 2 Murl2 (the two swirls agree at Order 2).",
      },
    ],
    // The murl / cayley complex-map family (parity wave 2) — MB3D corpus files
    // `murl`, `murl2_fast` and `cayley2IFS`, all classed needs_op in
    // corpus_coverage_2026-07-13.json.
    //
    // CLEAN-ROOM PROVENANCE. No decompiled corpus body and no GPL/LGPL source
    // (MB3D, Mandelbulber, flam3, JWildfire, Fractorium, Apophysis) was read
    // for this op. MB3D's own .m3f files are hex-encoded compiled x86 anyway —
    // there is no source there to read. Sources actually used:
    //   • The corpus PRE-STEP notes, which describe the maps STRUCTURALLY:
    //       murl       "complex-power Moebius composite (z^N, Cayley-style
    //                   rational, N-th root, z/Order)"
    //       murl2_fast "fast Order=2 specialization of murl; same gap"
    //       cayley2IFS "degree-4 complex rational map (Cayley transform of
    //                   z^2) in the rotated xy-plane; z kept, w *= |P7|"
    //   • For murl/murl2, TWO INDEPENDENT PERMISSIVELY-LICENSED
    //     implementations, cross-checked against each other and against a
    //     closed form re-derived here (agreement ~4e-15 over 4000 points):
    //       tsulej/GenerateMe  moire/Maps.pde        — the Unlicense
    //       generateme/fastmath src/fastmath/fields/m.clj — MIT
    //   • Draves & Reckase, "The Fractal Flame Algorithm" (flam3.com/
    //     flame_draves.pdf), Appendix — variation 39 `curl`.
    //   • Textbook complex analysis for the Cayley transform (Ahlfors/Rudin;
    //     Series, MA448 Hyperbolic Geometry notes, Warwick).
    // Attribution: murl/murl2 were invented by Zueuk (Peter Sdobnov) and
    // contributed to JWildfire by Nic Anderson (JWildfire V1.60 release
    // announcement). NO prose or published statement of either closed form
    // exists — the sidwellr CC-BY-SA variation catalogue (~1000 entries) does
    // not list murl at all. The forms below are therefore re-derived, not
    // transcribed.
    //
    // THE ONE CONSTRUCTION. Read the XY plane as a complex number ζ = x + iy
    // and leave the third coordinate alone. Every member is a Möbius-class
    // rational in ζ^N, i.e. a Möbius map pulled back through the power map
    // φ(ζ) = ζ^N — the corpus's "z^N → rational → root" pipeline. ζ^N is taken
    // by de Moivre in polar form (r^N, Nθ), θ = atan2(y, x), principal branch.
    //
    // Variant 0 — CAYLEY.   f(ζ) = (ζ^N − iC) / (ζ^N + iC).
    // C = 1 is the classical Cayley transform (z−i)/(z+i), the conformal map
    // carrying the upper half-plane onto the unit disc; C slides the reference
    // point along the imaginary axis. At N = 2 this is the corpus's "Cayley
    // transform of z²".
    //   ⚠ On "degree-4": the corpus calls C(z²) a "degree-4 rational map". As
    //   a rational map of the Riemann sphere its degree is max(deg num, deg
    //   den) = 2 (the numerator roots ±e^{iπ/4} and denominator roots
    //   ±e^{−iπ/4} are coprime, and a generic value has exactly 2 preimages).
    //   The note is counting numerator PLUS denominator degree, 2 + 2. Written
    //   down so a later pass does not "fix" this op by squaring something to
    //   chase the 4.
    //   ⚠ What MB3D's cayley2IFS actually computes could NOT be confirmed from
    //   any permissible source, so this variant is a principled reading of the
    //   corpus sentence, not a verified match. Flagged for a later oracle pass.
    //   (Aside, verified during research and worth knowing before anyone builds
    //   an escape-time preset on it: at exactly C = 1, N = 2 both critical
    //   orbits land on the repelling fixed point i, making C(z²) a Lattès map
    //   whose Julia set is the WHOLE sphere — uniform chaos, no basin structure
    //   to colour. Move C off 1 for anything escape-time.)
    //
    // Variant 1 — MURL.    c′ = C/(N−1) for N > 1, else C;
    //                      f(ζ) = (1 + c′)·ζ / (1 + c′·ζ^N).
    // A PLAIN reciprocal — no fractional root. Two details that are easy to
    // miss and are both load-bearing: the internal rescale of C by 1/(N−1),
    // and the (1 + c′) normalisation, which is exactly the factor making
    // f(ζ) = ζ on the unit circle where ζ^N = 1. Without it, moving the C
    // slider blows the figure up or collapses it instead of reshaping it.
    // At N = 1 this is ζ/(1 + Cζ) — the `curl` variation (flam3 V39,
    // z/(1 + c₁z + c₂z²)) at c₁ = 0, which is what makes the name read as
    // "Möbius curl". That equivalence is derived here, not cited: no published
    // sentence says murl generalises curl.
    //
    // Variant 2 — MURL2.   f(ζ) = |1 + C|^(2/N) · ζ / (1 + C·ζ^N)^(2/N).
    // The (2/N)-th-power reciprocal, on the UNRESCALED C, with the matching
    // normalisation. |1 + C| rather than (1 + C): for C < −1 and fractional
    // 2/N the unsigned form is NaN (one of the two reference implementations
    // has that bug; the other does not).
    //   ⚠ MURL AND MURL2 COINCIDE EXACTLY AT N = 2 (c′ = C/(2−1) = C and
    //   2/N = 1). That is not a coincidence to paper over — it is the
    //   independent confirmation that the corpus's `murl2_fast` really is
    //   "a fast Order=2 specialization of murl". Pinned by test.
    //
    // ORDER IS DELIBERATELY INTEGER (step 1). A non-integer N would put a
    // branch cut on ζ^N itself, tearing the field on the negative real axis
    // for every variant; integer N makes cos/sin of Nθ wrap coherently and
    // that inner cut disappears. Murl2 still carries an OUTER cut wherever
    // 1 + C·ζ^N crosses the negative reals (verified to occur even at integer
    // N ≠ 2); murl, being a plain reciprocal, has no cut at all.
    //
    // ALL THREE ARE BOUNDED AT INFINITY, which is what makes them usable in an
    // IFS without a bounding companion: Cayley sends ∞ → 1, and both swirls
    // decay (murl ~ A/(c′ζ^{N−1}), murl2 ~ A/(C^{2/N}ζ)). Each has poles —
    // Cayley at ζ^N = −iC, the swirls at 1 + cζ^N = 0, a ring of N points —
    // guarded by an epsilon on the denominator, plus an epsilon on the
    // normalisation for the degenerate c′ = −1 / C = −1.
    //
    // W-RULE — and this is the wave-1 torusInvert lesson applied on purpose.
    // f is holomorphic, so within the XY plane it is conformal with isotropic
    // factor |f′(ζ)|; but the third coordinate is untouched (factor 1), so the
    // 3-D map is NOT conformal and has no single exact k. The tempting move is
    // to push |f′| into w. DO NOT. |f′| DIVERGES at the poles of the rational
    // map, and because Z rides through untouched, each pole in the plane is a
    // whole VERTICAL LINE in space — the same axis-shaped divergence that made
    // torusInvert's azimuthal ratio explode w, collapse DE = r/w and render
    // every scene as a flat wall (see the ⚠ in id 58). w is therefore left
    // alone and the approximation is declared honestly with deApprox, which is
    // what the other nine warp-class ops (asinhWarp, logWarp, neoSqrWarp,
    // sinShear, gnarl2D/3D, toCoord/fromCoord, smoothBoxFold) already do.
    //   Second deApprox reason, inherent to the family: the fractional power
    //   d^(−1/N) carries a branch cut where d = 1 + C·ζ^N lands on the negative
    //   real axis. That is a genuine tear in the field for N ≥ 2 (N = 1 takes
    //   no root and is continuous). It belongs to the map, not to this
    //   implementation — the source has it too — and deApprox tightens the
    //   march step over it.
    wgsl: `
        let cmN = max(floor(op.p0 + 0.5), 1.0);
        let cmC = op.p1;
        let cmv = i32(op.p2 + 0.5);
        let cmr = sqrt((pos.x * pos.x) + (pos.y * pos.y));
        let cmth = atan2(pos.y, pos.x);
        let cmrn = pow(cmr, cmN);
        let cman = cmth * cmN;
        let cmur = cmrn * cos(cman);
        let cmui = cmrn * sin(cman);
        var cmx = pos.x;
        var cmy = pos.y;
        if (cmv == 1) {
          var cmc = cmC;
          if (cmN > 1.5) { cmc = cmC / (cmN - 1.0); }
          var cmA = 1.0 + cmc;
          if (abs(cmA) < 1e-6) { cmA = 1e-6; }
          let cmdr = 1.0 + (cmc * cmur);
          let cmdi = cmc * cmui;
          let cmden = max((cmdr * cmdr) + (cmdi * cmdi), 1e-12);
          cmx = (cmA * ((pos.x * cmdr) + (pos.y * cmdi))) / cmden;
          cmy = (cmA * ((pos.y * cmdr) - (pos.x * cmdi))) / cmden;
        } else if (cmv == 2) {
          let cme = 2.0 / cmN;
          let cmdr = 1.0 + (cmC * cmur);
          let cmdi = cmC * cmui;
          let cmdm = max(sqrt((cmdr * cmdr) + (cmdi * cmdi)), 1e-12);
          let cmpm = pow(cmdm, cme);
          let cmpa = atan2(cmdi, cmdr) * cme;
          let cmqr = cmpm * cos(cmpa);
          let cmqi = cmpm * sin(cmpa);
          let cmA = pow(max(abs(1.0 + cmC), 1e-6), cme);
          let cmden = max((cmqr * cmqr) + (cmqi * cmqi), 1e-12);
          cmx = (cmA * ((pos.x * cmqr) + (pos.y * cmqi))) / cmden;
          cmy = (cmA * ((pos.y * cmqr) - (pos.x * cmqi))) / cmden;
        } else {
          let cmni = cmui - cmC;
          let cmdi = cmui + cmC;
          let cmden = max((cmur * cmur) + (cmdi * cmdi), 1e-12);
          cmx = ((cmur * cmur) + (cmni * cmdi)) / cmden;
          cmy = ((cmni * cmur) - (cmur * cmdi)) / cmden;
        }
        pos = vec3f(cmx, cmy, pos.z);`,
    glsl: (v) => `
    // Complex map — a Möbius transform conjugated by ζ↦ζ^N on the XY plane
    // (0 Cayley, 1 Murl); Z untouched. Approximate DE, w deliberately NOT
    // tracked: |f'| diverges on the pole LINES (the torusInvert wall lesson).
    {
        float cmN = max(floor(${v[0]} + 0.5), 1.0);
        float cmC = ${v[1]};
        int cmv = int(${v[2]} + 0.5);
        float cmr = sqrt((pos.x * pos.x) + (pos.y * pos.y));
        float cmth = atan(pos.y, pos.x);
        float cmrn = pow(cmr, cmN);
        float cman = cmth * cmN;
        float cmur = cmrn * cos(cman);
        float cmui = cmrn * sin(cman);
        float cmx = pos.x;
        float cmy = pos.y;
        if (cmv == 1) {
            float cmc = (cmN > 1.5) ? (cmC / (cmN - 1.0)) : cmC;
            float cmA = 1.0 + cmc;
            if (abs(cmA) < 1e-6) cmA = 1e-6;
            float cmdr = 1.0 + (cmc * cmur);
            float cmdi = cmc * cmui;
            float cmden = max((cmdr * cmdr) + (cmdi * cmdi), 1e-12);
            cmx = (cmA * ((pos.x * cmdr) + (pos.y * cmdi))) / cmden;
            cmy = (cmA * ((pos.y * cmdr) - (pos.x * cmdi))) / cmden;
        } else if (cmv == 2) {
            float cme = 2.0 / cmN;
            float cmdr = 1.0 + (cmC * cmur);
            float cmdi = cmC * cmui;
            float cmdm = max(sqrt((cmdr * cmdr) + (cmdi * cmdi)), 1e-12);
            float cmpm = pow(cmdm, cme);
            float cmpa = atan(cmdi, cmdr) * cme;
            float cmqr = cmpm * cos(cmpa);
            float cmqi = cmpm * sin(cmpa);
            float cmA = pow(max(abs(1.0 + cmC), 1e-6), cme);
            float cmden = max((cmqr * cmqr) + (cmqi * cmqi), 1e-12);
            cmx = (cmA * ((pos.x * cmqr) + (pos.y * cmqi))) / cmden;
            cmy = (cmA * ((pos.y * cmqr) - (pos.x * cmqi))) / cmden;
        } else {
            float cmni = cmui - cmC;
            float cmdi = cmui + cmC;
            float cmden = max((cmur * cmur) + (cmdi * cmdi), 1e-12);
            cmx = ((cmur * cmur) + (cmni * cmdi)) / cmden;
            cmy = ((cmni * cmur) - (cmur * cmdi)) / cmden;
        }
        pos = vec3(cmx, cmy, pos.z);
    }`,
  },
  {
    id: 62,
    key: "ruckerBulb",
    name: "Rucker Bulb",
    wRule: W_BULB,
    category: "power",
    blurb:
      "A trig bulb with its two spin angles powered independently, a squashable pole and a choice of four angle flavors — the stretched, tilted cousin of the Mandelbulb. Escape-time mode.",
    params: [
      {
        name: "Power",
        default: 8.0,
        min: 2.0,
        max: 16.0,
        step: 0.1,
        type: "double",
        tip: "Number of lobes (8 = classic).",
      },
      {
        name: "AziPow",
        default: 1.0,
        min: 0.25,
        max: 4.0,
        step: 0.05,
        type: "double",
        tip: "Winds the bulbs around independently of the up-down count. 1 keeps them tied.",
      },
      {
        name: "ZMul",
        default: 1.0,
        min: -2.0,
        max: 2.0,
        step: 0.05,
        type: "double",
        tip: "Stretches or squashes the poles; negative flips them top-to-bottom.",
      },
      {
        name: "RadialSel",
        default: 0.0,
        min: 0.0,
        max: 1.0,
        step: 1.0,
        type: "double",
        tip: "Which power grows the body: 0 the lobe count, 1 the winding. No effect while AziPow is 1.",
      },
      {
        name: "Convention",
        default: 0.0,
        min: 0.0,
        max: 3.0,
        step: 1.0,
        type: "double",
        tip: "Trig flavor: 0 classic, 1 swapped, 2 latitude, 3 signed planar — each reshapes the lobes.",
      },
    ],
    // ── The Ruckerbulb family (parity wave 3, the >3-param guinea pig) ────────
    // Closes the MB3D corpus file `Ruckerbulb`, tagged trig_bulb_variant /
    // covered_op (i.e. approximated by bulbAxis at defaults only) in
    // corpus_coverage_2026-07-13.json. TRIGBULB_SPIKE.md:65-72 measured its dof
    // at ~5 and, because the ABI then had 3 slots, pinned a TRUNCATED encoding
    // — `ruckerBulb(Power, AziPow, ZMul)` — and recorded the radial-power
    // selector and the 4th angle flavor as ACCEPTED LOSSES. This op un-does
    // both: it is the first operator to spend the opAux overflow lane
    // (docs/planning/OP_PARAM_ENCODING.md), so params 3-4 ride opAux[o].x/.y
    // and the author-facing syntax below is still just `op.p3` / `op.p4`.
    //
    // CLEAN-ROOM PROVENANCE. No decompiled corpus body and no GPL/LGPL source
    // was read for this op. MB3D's .m3f files are hex-encoded compiled x86 —
    // there is no source in them to read. Sources actually used:
    //   • The corpus entry's own STRUCTURAL note: "azimuth atan(-x,z)*zAnglePow,
    //     polar atan(y,x)*Power, negated x/y outputs, separate angle powers
    //     (P=2 default)" — i.e. two independently-powered angles, one of them a
    //     SIGNED two-argument arctangent in a coordinate plane, and a sign flip
    //     on the outputs.
    //   • TRIGBULB_SPIKE.md's dof read (:65-72) and its ruling that axis and
    //     sign/argument-order conventions are ISOMETRIES (:39-45) — reachable
    //     by rotate/planeFold recipe cards, so they are deliberately NOT params
    //     here. That is why this op is z-polar with no Axis slot: conjugation
    //     by a rotation already covers every pole.
    //   • The shipped `bulbAxis` (id 29) conventions 0/1/2, which this op
    //     reproduces EXACTLY so the two ops agree wherever they overlap —
    //     pinned numerically in core/ruckerbulb.test.mjs.
    //   • Textbook spherical-power (White/Nylander) bulb construction, as
    //     already used by mandelbulbPower (id 14) and sphericalTwoStage (37).
    //
    // AziPow is a MULTIPLIER on the azimuth's power (sphericalTwoStage's
    // PhiMul precedent), not an absolute exponent: φ' = Power·AziPow·φ. That
    // makes AziPow = 1 the exact tied-power degeneracy while still reaching any
    // absolute azimuth power. It is POSITIVE-ONLY (unlike PhiMul's ±4) because
    // RadialSel can route it into the RADIAL exponent, where a negative power
    // would blow r→0 up to a non-finite f32. Negative windings are the mirror
    // isometry, i.e. a recipe card — exactly the spike's own ruling.
    //
    // RadialSel picks which of the two powers in play exponentiates the radius:
    // 0 = Power (tied — the classic r→rⁿ), 1 = the azimuth's effective power
    // Power·AziPow. It is INERT at AziPow = 1, when the two coincide.
    //
    // Convention 0/1/2 are bulbAxis's, verbatim: 0 cos-polar (classic),
    // 1 sin-polar (NormBulb — swaps sin/cos on the multiplied angle),
    // 2 asin-latitude (sine bulb). Convention 3 is the flavor the spike could
    // not afford: the polar angle is the SIGNED planar arctangent atan2(-x, z)
    // rather than the unsigned acos(z/r). Those are genuinely different maps —
    // acos lands in [0, π], the planar angle in (-π, π] — so once multiplied by
    // a power and fed to sin/cos they wrap differently. Any unsigned
    // reformulation (atan2(hypot(x,y), z), atan2(z, hypot(x,y))) is algebraically
    // just acos/asin again and would NOT be a distinct convention; the sign is
    // the whole content of the 4th flavor.
    //
    // W-RULE. The three angle conventions leave the radial map alone, so they
    // share one analytic dr (TRIGBULB_SPIKE.md:77-82). ZMul does NOT: it scales
    // one output component, so the map is D·B with D = diag(1,1,ZMul), giving
    // ‖J‖ <= max(1,|ZMul|)·‖J_B‖. The w update therefore carries that
    // conservative factor and stays a valid Lipschitz bound for every ZMul —
    // and degenerates to the plain W_BULB update EXACTLY at the ZMul = 1
    // default, so no precision is lost in the common case. The exponent tracked
    // is whichever RadialSel selected. Escape-time: pair with Add c.
    //
    // Degeneracy anchor (pinned): ruckerBulb(P, 1, 1, 0, 0) is EXACTLY
    // bulbAxis(P, 0, 0), which is EXACTLY mandelbulbPower(P).
    wgsl: `
        let bp = op.p0;
        let baz = op.p1;
        let bzm = op.p2;
        let rsel = i32(op.p3 + 0.5);
        let convRaw = i32(op.p4 + 0.5);
        let conv = select(0, convRaw, convRaw >= 0 && convRaw <= 3);
        let br = length(pos);
        if (br > 1e-9) {
          let bu = clamp(pos.z / br, -1.0, 1.0);
          var bth = acos(bu);
          if (conv == 2) { bth = asin(bu); }
          else if (conv == 3) { bth = atan2(-pos.x, pos.z); }
          bth = bth * bp;
          let bph = atan2(pos.y, pos.x) * bp * baz;
          let bq = select(bp, bp * baz, rsel == 1);
          let brn = pow(br, bq);
          w = bq * brn / br * max(1.0, abs(bzm)) * w + 1.0;
          var bst = sin(bth);
          var bpo = cos(bth);
          if (conv != 0) { bst = cos(bth); bpo = sin(bth); }
          pos = vec3f(brn * bst * cos(bph), brn * bst * sin(bph), brn * bpo * bzm);
        }`,
    glsl: (v) => `
    // Rucker bulb — independent azimuth power, z-output multiplier, radial-power
    // selector, 4 angle conventions (0 cos · 1 sin · 2 asin · 3 signed planar).
    // Escape-time; w carries the analytic radial derivative with the ZMul bound.
    {
        float bp = ${v[0]};
        float baz = ${v[1]};
        float bzm = ${v[2]};
        int rsel = int(${v[3]} + 0.5);
        int conv = int(${v[4]} + 0.5);
        if (conv < 0 || conv > 3) conv = 0;
        float br = length(pos);
        if (br > 1e-9) {
            float bu = clamp(pos.z / br, -1.0, 1.0);
            float bth = acos(bu);
            if (conv == 2) bth = asin(bu);
            else if (conv == 3) bth = atan(-pos.x, pos.z);
            bth *= bp;
            float bph = atan(pos.y, pos.x) * bp * baz;
            float bq = (rsel == 1 ? bp * baz : bp);
            float brn = pow(br, bq);
            w = bq * brn / br * max(1.0, abs(bzm)) * w + 1.0;
            float bst = (conv != 0 ? cos(bth) : sin(bth));
            float bpo = (conv != 0 ? sin(bth) : cos(bth));
            pos = vec3(brn * bst * cos(bph), brn * bst * sin(bph), brn * bpo * bzm);
        }
    }`,
  },
  // ── #635 Catoptron parity — Mirror Shells + Spiral Vortex ─────────────────
  {
    id: 63,
    key: "mirrorShells",
    name: "Mirror Shells",
    wRule: W_UNCHANGED,
    category: "sphere",
    blurb:
      "Folds space into nested mirrored shells around the center, turning one shape into an endless onion of concentric copies.",
    params: [
      {
        name: "Spacing",
        default: 1.5,
        min: 0.0,
        max: 8.0,
        step: 0.05,
        type: "double",
        tip: "Radial period of the shells (0 turns the fold off).",
      },
      {
        name: "Offset",
        default: 0.0,
        min: 0.0,
        max: 8.0,
        step: 0.05,
        type: "double",
        tip: "Inner radius where the shells start — everything closer stays untouched.",
      },
    ],
    // Radial reflection tiling (#635): the tent fold applied to RADIUS, with an
    // identity core below Offset. r' = r for r < R0, else R0 + tent_S(r − R0),
    // direction preserved (p *= r'/r).
    //
    // DE derivation (the #635 spec-review blocker, resolved): the map p ↦
    // (r'(r)/r)·p has singular values {|dr'/dr|, r'/r, r'/r}. The tent gives
    // |dr'/dr| = 1 a.e., and r' ≤ r EVERYWHERE — inside the first half-period
    // the map is the identity (r' = r), beyond it r ≥ R0 + S/2 ≥ r' — so every
    // singular value is ≤ 1: the fold is 1-Lipschitz (non-expanding), which
    // makes the child DE a valid lower bound with NO w change. This is the key
    // structural difference from polygonFold's deApprox case: its θ-dependent
    // stretch EXCEEDS 1, ours provably cannot. The tangential contraction
    // r'/r < 1 in outer shells makes the DE conservative there (slower march,
    // never an overshoot). Sampled operator-norm gate: mirrorvortex.test.mjs.
    //
    // Self-bounding: folded radius ≤ R0 + S/2 regardless of input, so this op
    // joins BOUNDING_FOLDS (stability.js) — the review's open question,
    // answered "yes" because the bound is unconditional, like sphereFold's cap.
    //
    // NO wgslDf / wgslPt twins: round() on the radius breaks the smooth-delta
    // assumption both deep-zoom tiers rely on (the same reason tentFold and
    // modFold ship without twins — a fold seam is exactly where a delta orbit
    // diverges from its reference).
    wgsl: `
        if (op.p0 > 0.0) {
            let msr = length(pos);
            let msm = msr - op.p1;
            if (msm > 0.0 && msr > 1e-12) {
                let msf = abs(msm - op.p0 * round(msm / op.p0));
                pos = pos * ((op.p1 + msf) / msr);
            }
        }`,
    glsl: (v) => `
    // mirror shells — radial tent fold past the offset radius (1-Lipschitz).
    if (${v[0]} > 0.0) {
        float msr = length(pos);
        float msm = msr - ${v[1]};
        if (msm > 0.0 && msr > 1e-12) {
            float msf = abs(msm - ${v[0]} * round(msm / ${v[0]}));
            pos = pos * ((${v[1]} + msf) / msr);
        }
    }`,
  },
  {
    id: 64,
    key: "spiralVortex",
    name: "Spiral Vortex",
    wRule: W_MUL_K,
    category: "move",
    blurb:
      "Winds space into a logarithmic spiral around an axis — the whorl of a galaxy or a snail shell, with an honest cost the marcher accounts for.",
    params: [
      {
        name: "Amount",
        default: 0.6,
        min: -2.0,
        max: 2.0,
        step: 0.01,
        type: "double",
        tip: "How hard it winds. Negative reverses the spiral's handedness.",
      },
      {
        name: "Axis",
        default: 1.0,
        min: 0.0,
        max: 2.0,
        step: 1.0,
        type: "double",
        tip: "Spiral axis: 0 X, 1 Y (upright whorl), 2 Z.",
      },
    ],
    // Log-spiral shear (#635): rotate the plane ⊥ Axis by a·ln(r), r = in-plane
    // radius. Distinct from twist (angle ∝ AXIAL distance, stretch grows with
    // radius) and from gnarl's compounding warps.
    //
    // DE derivation (the #635 spec-review blocker, resolved — and then
    // CORRECTED by its own gate): in in-plane polar coordinates the map is
    // (r, θ) ↦ (r, θ + a·ln r). A tangential step rotates isometrically; a
    // radial step dr ALSO rotates the image by a·dr/r — an arc of length
    // r·(a·dr/r) = a·dr, the log's 1/r cancelling the arc-length r. In
    // orthonormal components the differential is the SHEAR [[1,0],[a,1]] —
    // not a diagonal stretch — whose largest singular value is
    //   L(a) = (|a| + sqrt(a² + 4)) / 2
    // (naively reading the radial column as an independent (1,a) gives
    // sqrt(1+a²), which the sampled gate measured as an UNDERestimate: worst
    // ratio 1.3440095 at a = 0.6 — exactly L(0.6) to 7 digits). L is CONSTANT
    // over all r — precisely why log-spiral is safe where twist loosens
    // (twist's stretch grows with radius without bound). w *= L(a) is an
    // exact uniform Lipschitz accounting, applied unconditionally so the
    // bound holds through the near-axis guard branch. Sampled operator-norm
    // gate (which caught the shear error): mirrorvortex.test.mjs.
    //
    // Near the axis the total angle a·ln(r) grows without bound — the map
    // winds infinitely but never expands; the 1e-24 r² guard only skips the
    // numerically meaningless exact-axis points.
    //
    // NO wgslDf / wgslPt twins: ln/atan2 of a delta-displaced radius has no
    // stable split into reference + delta near the axis winding (same
    // exclusion reasoning as radialInvert's post-#422 note).
    wgsl: `
        let sva = i32(op.p1 + 0.5);
        var svu = pos.x; var svv = pos.z; // Axis 1 (Y): wind the XZ plane
        if (sva == 0) { svu = pos.y; svv = pos.z; }
        else if (sva == 2) { svu = pos.x; svv = pos.y; }
        let svr2 = svu * svu + svv * svv;
        if (svr2 > 1e-24) {
            let svt = op.p0 * 0.5 * log(svr2);
            let svc = cos(svt); let svs = sin(svt);
            let svx = svu * svc - svv * svs;
            let svy = svu * svs + svv * svc;
            if (sva == 0) { pos.y = svx; pos.z = svy; }
            else if (sva == 2) { pos.x = svx; pos.y = svy; }
            else { pos.x = svx; pos.z = svy; }
        }
        w *= (abs(op.p0) + sqrt(op.p0 * op.p0 + 4.0)) * 0.5;`,
    glsl: (v) => `
    // spiral vortex — rotate the plane ⊥ axis by a·ln(r); the differential is a
    // shear: w *= (|a| + sqrt(a²+4))/2, the exact uniform operator norm.
    {
        int sva = int(${v[1]} + 0.5);
        float svu = pos.x, svv = pos.z;
        if (sva == 0) { svu = pos.y; svv = pos.z; }
        else if (sva == 2) { svu = pos.x; svv = pos.y; }
        float svr2 = svu * svu + svv * svv;
        if (svr2 > 1e-24) {
            float svt = ${v[0]} * 0.5 * log(svr2);
            float svc = cos(svt), svs = sin(svt);
            float svx = svu * svc - svv * svs;
            float svy = svu * svs + svv * svc;
            if (sva == 0) { pos.y = svx; pos.z = svy; }
            else if (sva == 2) { pos.x = svx; pos.y = svy; }
            else { pos.x = svx; pos.z = svy; }
        }
        w *= (abs(${v[0]}) + sqrt(${v[0]} * ${v[0]} + 4.0)) * 0.5;
    }`,
  },
  // ── #632 Catoptron parity — crystallographic space-group folds ────────────
  //
  // CLEAN-ROOM PROVENANCE. Nothing here derives from Catoptron3D (unlicensed
  // upstream — see the issue's caveat): the group roster, the chambers and the
  // fold algorithms are derived independently from the crystallographic
  // literature (International Tables space-group facts + affine Coxeter
  // theory: a group generated by reflections folds space onto the closure of
  // one chamber, and the orbit of any point meets that closed chamber in
  // EXACTLY one point). The roster below even DIFFERS from the upstream menu —
  // the #632 spec review re-derived it (P-3m1 is NOT reflection-generated and
  // is replaced by P-6m2; centered, screw and glide groups are excluded until
  // the #633 seam contract exists) — which is itself clean-room evidence.
  //
  // WHY ONLY REFLECTION-GENERATED GROUPS: a conditional reflection ("fold")
  // is continuous and 1-Lipschitz — two points straddling the mirror map to
  // points at most as far apart (the chord argument). Any group operation
  // that is NOT a product of the group's own mirrors (a bare translation,
  // screw, glide, inversion or rotoinversion) forces a discontinuous
  // assignment somewhere, i.e. tears the DE at a seam. The five 3-D groups
  // and three wallpaper groups here are exactly the crystallographic groups
  // of their families that ARE generated by reflections, so their folds are
  // globally 1-Lipschitz with NO seam and NO w cost (W_UNCHANGED).
  //
  // THE FOLD ALGORITHM (shared by both ops): every step applied to the point
  // is an element of the target group — a mirror reflection applied only on
  // the far side (a fold), or a pure lattice translation (the sawtooth
  // "mod" steps). Group elements map orbits to themselves, so the CONVERGED
  // output is the unique orbit representative in the closed chamber, no
  // matter the path taken; the composite map is therefore continuous and
  // 1-Lipschitz even where an intermediate sawtooth is not. The rectangular
  // closed-form prefolds bound the point into one cell first, which makes the
  // remaining mirror cascade converge in a DISTANCE-FREE number of passes:
  // K = 3 is baked in everywhere (measured bound K = 2 over 300k samples out
  // to r = 1000·cell, +1 margin — core/spacegroup.test.mjs asserts the
  // fundamental-domain gate, the #632-review lesson that Lipschitz sampling
  // alone cannot see a fold that never reaches its cell).
  //
  // NO wgslDf / wgslPt twins, BY DESIGN (#632 review §4): a crystal lattice is
  // self-similar under TRANSLATION, not scale — deep zoom finds no new
  // detail — and the round() cell seams break the smooth-delta split both
  // deep tiers rely on (the tentFold/modFold exclusion). Formulas using these
  // ops are simply df64/pt-ineligible (stability.js derives eligibility from
  // twin presence — nothing else to wire).
  {
    id: 65,
    key: "wallpaperFold",
    name: "Wallpaper Fold",
    wRule: W_UNCHANGED,
    category: "symmetry",
    blurb:
      "Folds the horizontal plane into one repeating wallpaper tile — hexagonal *632, square *442 or triangular *333 — so any shape becomes an endless floor pattern.",
    params: [
      {
        name: "Tile",
        default: 0.0,
        min: 0.0,
        max: 2.0,
        step: 1.0,
        type: "double",
        tip: "Which wallpaper symmetry: 0 hexagonal *632, 1 square *442, 2 triangular *333.",
      },
      {
        name: "Cell",
        default: 1.5,
        min: 0.0,
        max: 8.0,
        step: 0.05,
        type: "double",
        tip: "Lattice spacing of the tiling (0 turns the fold off).",
      },
    ],
    // The 2-D cousin (#632): the three wallpaper groups generated by
    // reflections — the triangle groups *632 (p6m), *442 (p4m), *333 (p3m1) —
    // folding x,y and leaving z free (the induced 3-D symmetry is the
    // wallpaper group x the free z line, a prism). Chambers, with cell a:
    //   *632  right triangle   (0,0) (a/2,0) (a/2, a/(2√3))   — 0 ≤ y ≤ x/√3, x ≤ a/2
    //   *442  right triangle   (0,0) (a/2,0) (a/2, a/2)       — 0 ≤ y ≤ x,   x ≤ a/2
    //   *333  equilateral tri  (0,0) (a/2, a/(2√3)) (0, a/√3) — x ≥ 0, y ≥ x/√3, x+√3y ≤ a
    // Lattice: *442 square; *632/*333 hexagonal with a1 = (a,0),
    // a2 = (a/2, √3a/2). Mirror orientations are forced by the group: p6m has
    // mirrors every 30°; p3m1's mirrors are the three lines whose NORMALS lie
    // along the lattice vectors (30°/90°/150° lines — this is what makes it
    // *333 and not p31m, which has a naked 3-fold and is NOT a reflection
    // group). Every fold below reflects across a line of the group:
    //   tri(x,a)   — vertical mirrors x = m·a/2 (p6m: through the 6-fold and
    //                2-fold centers; p3m1: through all 3-fold centers)
    //   tri(y,√3a) — horizontal mirrors y = m·√3a/2 (p6m only — p3m1 has NO
    //                horizontal mirror, so its y step is the PURE-TRANSLATION
    //                sawtooth, legal by the orbit argument above)
    //   the conditional reflections — the 60°/30° lines through the origin
    //                (p6m) and the 30° line + far wall x+√3y = a (p3m1).
    // *442 is fully closed-form (tri, tri, one diagonal swap): 1 pass. The
    // hex kernels run the K = 3 bounded cascade (header note). Proof gates:
    // sampled operator norm ≤ 1 near AND far, fundamental domain, generator
    // invariance (incl. the #584 hex-lattice lesson: invariance under BOTH
    // lattice vectors, brute-force), orbit-enumeration ground truth, and
    // idempotence — core/spacegroup.test.mjs.
    wgsl: `
        let wpT0 = i32(op.p0 + 0.5);
        let wpT = select(0, wpT0, wpT0 >= 0 && wpT0 <= 2);
        if (op.p1 > 0.0) {
            let wpA = op.p1;
            var wpx = pos.x;
            var wpy = pos.y;
            if (wpT == 1) {
                wpx = abs(wpx - wpA * round(wpx / wpA));
                wpy = abs(wpy - wpA * round(wpy / wpA));
                if (wpy > wpx) { let t = wpx; wpx = wpy; wpy = t; }
            } else {
                let wpH = 1.7320508075688772 * wpA;
                for (var k = 0; k < 3; k++) {
                    wpx = abs(wpx - wpA * round(wpx / wpA));
                    if (wpT == 2) {
                        wpy = wpy - wpH * round(wpy / wpH);
                        var wd = 0.8660254037844386 * wpy - 0.5 * wpx;
                        if (wd < 0.0) { wpx += wd; wpy -= 1.7320508075688772 * wd; }
                        wd = 0.5 * wpx + 0.8660254037844386 * wpy - 0.5 * wpA;
                        if (wd > 0.0) { wpx -= wd; wpy -= 1.7320508075688772 * wd; }
                    } else {
                        wpy = abs(wpy - wpH * round(wpy / wpH));
                        var wd = 0.8660254037844386 * wpx - 0.5 * wpy;
                        if (wd < 0.0) { wpx -= 1.7320508075688772 * wd; wpy += wd; }
                        wd = 0.5 * wpx - 0.8660254037844386 * wpy;
                        if (wd < 0.0) { wpx -= wd; wpy += 1.7320508075688772 * wd; }
                    }
                }
            }
            pos.x = wpx;
            pos.y = wpy;
        }`,
    glsl: (v) => `
    // wallpaper fold — *632 / *442 / *333 reflection tile in x,y (z free).
    // Every step is a mirror of the group (or a lattice translation), so the
    // converged point is the chamber representative: 1-Lipschitz, w untouched.
    {
        int wpT = int(${v[0]} + 0.5);
        if (wpT < 0 || wpT > 2) wpT = 0;
        if (${v[1]} > 0.0) {
            float wpA = ${v[1]};
            float wpx = pos.x;
            float wpy = pos.y;
            if (wpT == 1) {
                wpx = abs(wpx - wpA * round(wpx / wpA));
                wpy = abs(wpy - wpA * round(wpy / wpA));
                if (wpy > wpx) { float t = wpx; wpx = wpy; wpy = t; }
            } else {
                float wpH = 1.7320508075688772 * wpA;
                for (int k = 0; k < 3; k++) {
                    wpx = abs(wpx - wpA * round(wpx / wpA));
                    if (wpT == 2) {
                        wpy = wpy - wpH * round(wpy / wpH);
                        float wd = 0.8660254037844386 * wpy - 0.5 * wpx;
                        if (wd < 0.0) { wpx += wd; wpy -= 1.7320508075688772 * wd; }
                        wd = 0.5 * wpx + 0.8660254037844386 * wpy - 0.5 * wpA;
                        if (wd > 0.0) { wpx -= wd; wpy -= 1.7320508075688772 * wd; }
                    } else {
                        wpy = abs(wpy - wpH * round(wpy / wpH));
                        float wd = 0.8660254037844386 * wpx - 0.5 * wpy;
                        if (wd < 0.0) { wpx -= 1.7320508075688772 * wd; wpy += wd; }
                        wd = 0.5 * wpx - 0.8660254037844386 * wpy;
                        if (wd < 0.0) { wpx -= wd; wpy += 1.7320508075688772 * wd; }
                    }
                }
            }
            pos.x = wpx;
            pos.y = wpy;
        }
    }`,
  },
  {
    id: 66,
    key: "spaceGroupFold",
    name: "Space Group",
    wRule: W_UNCHANGED,
    category: "symmetry",
    blurb:
      "Folds all of space into one crystallographic cell — pick a real space group and any shape becomes a crystal lattice: Pmmm, P4/mmm, P6/mmm, P-6m2 or Pm-3m.",
    params: [
      {
        name: "Group",
        default: 4.0,
        min: 0.0,
        max: 4.0,
        step: 1.0,
        type: "double",
        tip: "Space group: 0 Pmmm (orthorhombic), 1 P4/mmm (tetragonal), 2 P6/mmm (hexagonal), 3 P-6m2 (trigonal prism), 4 Pm-3m (cubic).",
      },
      {
        name: "CellA",
        default: 1.5,
        min: 0.0,
        max: 8.0,
        step: 0.05,
        type: "double",
        tip: "Lattice constant a (0 turns the fold off).",
      },
      {
        name: "CellC",
        default: 0.0,
        min: 0.0,
        max: 8.0,
        step: 0.05,
        type: "double",
        tip: "Lattice constant c for the tetragonal and hexagonal groups (0 = same as a; the cubic group ignores it).",
      },
      {
        name: "CellB",
        default: 0.0,
        min: 0.0,
        max: 8.0,
        step: 0.05,
        type: "double",
        tip: "Lattice constant b — read by Pmmm only (0 = same as a).",
      },
    ],
    // The 3-D op (#632): the five crystallographic space groups generated by
    // reflections, one per crystal family — each is a product of affine
    // Coxeter kernels, so each fold is the 2-D machinery plus a z tent:
    //   0 Pmmm   (#47)  = Ã1×Ã1×Ã1: tri(x,a) tri(y,b) tri(z,c).
    //                     Chamber [0,a/2]×[0,b/2]×[0,c/2].
    //   1 P4/mmm (#123) = *442 × Ã1: tri(x,a) tri(y,a), diagonal swap,
    //                     tri(z,c). Chamber 0 ≤ y ≤ x ≤ a/2, z ∈ [0,c/2].
    //   2 P6/mmm (#191) = *632 × Ã1 (the one group in the set with NO
    //                     composition path from shipped ops — the hexagonal
    //                     lattice fold). Chamber: the *632 triangle × [0,c/2].
    //   3 P-6m2  (#187) = *333 × Ã1. The vertical mirror planes' normals lie
    //                     along the a-axes (secondary H-M slot → "m"), the
    //                     in-plane C2 axes along the tertiary directions —
    //                     that is the -6m2 setting, and it is forced: the
    //                     other D3h setting over *333's lattice would need
    //                     p31m in-plane, which is not a reflection group.
    //                     Chamber: the *333 triangle × [0,c/2]. This is the
    //                     spec review's substitution for P-3m1, whose D3d
    //                     point group is NOT generated by its 3 mirrors
    //                     (rotoinversion required → torn → #633 tier).
    //   4 Pm-3m  (#221) = C̃3: tri each axis with a, then sort descending
    //                     (three conditional diagonal-mirror reflections).
    //                     Chamber: the orthoscheme a/2 ≥ x ≥ y ≥ z ≥ 0.
    //                     Exactly tentFold(a,a,a) + mengerFold — pinned in
    //                     core/spacegroup.test.mjs, the #632 review's own
    //                     composition, kept as a switch arm for the named
    //                     dropdown parity.
    // EXCLUDED (the torn tier, blocked on the #633 seam contract): P1, P-1,
    // P2/m, P-3m1, P2₁2₁2₁, P6₃/mmc (bare translations / inversions / screws
    // tear at seams), and the centered cubics Im-3m / Fm-3m — their centering
    // translation (a/2,a/2,a/2) resp. (a/2,a/2,0) is NOT a product of the
    // group's mirrors, so no reflection fold can produce it (the review's
    // derivation, kept here so nobody burns a session trying).
    // PARAM COLLAPSE RULE (review §3 ruling): CellA is the master switch
    // (0 = op off). Cubic reads a only; tetragonal/hexagonal read a and c
    // (c = 0 collapses to a); orthorhombic additionally reads b (p3 — the
    // opAux overflow lane, ruckerBulb precedent; b = 0 collapses to a).
    // Ignored dims are IGNORED, never misapplied, so a Vary roll of the Group
    // enum is always a correct crystal for whatever dims it lands on.
    // The hexagonal arms are textual twins of wallpaperFold's kernels (one
    // dimension up); cross-op agreement on the z = 0 plane is pinned in
    // core/spacegroup.test.mjs so the copies cannot drift.
    wgsl: `
        let sgG0 = i32(op.p0 + 0.5);
        let sgG = select(0, sgG0, sgG0 >= 0 && sgG0 <= 4);
        if (op.p1 > 0.0) {
            let sgA = op.p1;
            var sgC = op.p2;
            if (sgC <= 0.0) { sgC = sgA; }
            var sgx = pos.x;
            var sgy = pos.y;
            var sgz = pos.z;
            if (sgG == 2 || sgG == 3) {
                let sgH = 1.7320508075688772 * sgA;
                for (var k = 0; k < 3; k++) {
                    sgx = abs(sgx - sgA * round(sgx / sgA));
                    if (sgG == 3) {
                        sgy = sgy - sgH * round(sgy / sgH);
                        var sd = 0.8660254037844386 * sgy - 0.5 * sgx;
                        if (sd < 0.0) { sgx += sd; sgy -= 1.7320508075688772 * sd; }
                        sd = 0.5 * sgx + 0.8660254037844386 * sgy - 0.5 * sgA;
                        if (sd > 0.0) { sgx -= sd; sgy -= 1.7320508075688772 * sd; }
                    } else {
                        sgy = abs(sgy - sgH * round(sgy / sgH));
                        var sd = 0.8660254037844386 * sgx - 0.5 * sgy;
                        if (sd < 0.0) { sgx -= 1.7320508075688772 * sd; sgy += sd; }
                        sd = 0.5 * sgx - 0.8660254037844386 * sgy;
                        if (sd < 0.0) { sgx -= sd; sgy += 1.7320508075688772 * sd; }
                    }
                }
                sgz = abs(sgz - sgC * round(sgz / sgC));
            } else if (sgG == 1) {
                sgx = abs(sgx - sgA * round(sgx / sgA));
                sgy = abs(sgy - sgA * round(sgy / sgA));
                if (sgy > sgx) { let t = sgx; sgx = sgy; sgy = t; }
                sgz = abs(sgz - sgC * round(sgz / sgC));
            } else if (sgG == 4) {
                sgx = abs(sgx - sgA * round(sgx / sgA));
                sgy = abs(sgy - sgA * round(sgy / sgA));
                sgz = abs(sgz - sgA * round(sgz / sgA));
                if (sgx < sgy) { let t = sgx; sgx = sgy; sgy = t; }
                if (sgx < sgz) { let t = sgx; sgx = sgz; sgz = t; }
                if (sgy < sgz) { let t = sgy; sgy = sgz; sgz = t; }
            } else {
                var sgB = op.p3;
                if (sgB <= 0.0) { sgB = sgA; }
                sgx = abs(sgx - sgA * round(sgx / sgA));
                sgy = abs(sgy - sgB * round(sgy / sgB));
                sgz = abs(sgz - sgC * round(sgz / sgC));
            }
            pos = vec3f(sgx, sgy, sgz);
        }`,
    glsl: (v) => `
    // space-group fold — Pmmm / P4/mmm / P6/mmm / P-6m2 / Pm-3m reflection
    // fold into one crystallographic chamber. Every step is a mirror of the
    // group (or a lattice translation): 1-Lipschitz, w untouched.
    {
        int sgG = int(${v[0]} + 0.5);
        if (sgG < 0 || sgG > 4) sgG = 0;
        if (${v[1]} > 0.0) {
            float sgA = ${v[1]};
            float sgC = ${v[2]};
            if (sgC <= 0.0) sgC = sgA;
            float sgx = pos.x;
            float sgy = pos.y;
            float sgz = pos.z;
            if (sgG == 2 || sgG == 3) {
                float sgH = 1.7320508075688772 * sgA;
                for (int k = 0; k < 3; k++) {
                    sgx = abs(sgx - sgA * round(sgx / sgA));
                    if (sgG == 3) {
                        sgy = sgy - sgH * round(sgy / sgH);
                        float sd = 0.8660254037844386 * sgy - 0.5 * sgx;
                        if (sd < 0.0) { sgx += sd; sgy -= 1.7320508075688772 * sd; }
                        sd = 0.5 * sgx + 0.8660254037844386 * sgy - 0.5 * sgA;
                        if (sd > 0.0) { sgx -= sd; sgy -= 1.7320508075688772 * sd; }
                    } else {
                        sgy = abs(sgy - sgH * round(sgy / sgH));
                        float sd = 0.8660254037844386 * sgx - 0.5 * sgy;
                        if (sd < 0.0) { sgx -= 1.7320508075688772 * sd; sgy += sd; }
                        sd = 0.5 * sgx - 0.8660254037844386 * sgy;
                        if (sd < 0.0) { sgx -= sd; sgy += 1.7320508075688772 * sd; }
                    }
                }
                sgz = abs(sgz - sgC * round(sgz / sgC));
            } else if (sgG == 1) {
                sgx = abs(sgx - sgA * round(sgx / sgA));
                sgy = abs(sgy - sgA * round(sgy / sgA));
                if (sgy > sgx) { float t = sgx; sgx = sgy; sgy = t; }
                sgz = abs(sgz - sgC * round(sgz / sgC));
            } else if (sgG == 4) {
                sgx = abs(sgx - sgA * round(sgx / sgA));
                sgy = abs(sgy - sgA * round(sgy / sgA));
                sgz = abs(sgz - sgA * round(sgz / sgA));
                if (sgx < sgy) { float t = sgx; sgx = sgy; sgy = t; }
                if (sgx < sgz) { float t = sgx; sgx = sgz; sgz = t; }
                if (sgy < sgz) { float t = sgy; sgy = sgz; sgz = t; }
            } else {
                float sgB = ${v[3]};
                if (sgB <= 0.0) sgB = sgA;
                sgx = abs(sgx - sgA * round(sgx / sgA));
                sgy = abs(sgy - sgB * round(sgy / sgB));
                sgz = abs(sgz - sgC * round(sgz / sgC));
            }
            pos = vec3(sgx, sgy, sgz);
        }
    }`,
  },
  // ── #633 Catoptron parity — the seam-clamped DE contract + Hinge Fold ──────
  //
  // THE SEAM-CLAMPED DE CONTRACT (#633). An op marked `seam: true` is allowed
  // to be DISCONTINUOUS. Its "cells" are the connected components of the
  // complement of its tear set; ON EACH CELL the op must honour its declared
  // wRule exactly (both seam ops here are per-cell isometries, W_UNCHANGED).
  // In exchange, every emitter body must ALSO update the seam channel:
  //
  //     seamOut = min(seamOut, cutDist(pos) / max(abs(w), 1e-9))
  //     (WGSL: seamOut · GLSL: g_seam · CPU: s.seam)
  //
  // where cutDist(pos) is a LOWER bound on the distance from the op's INPUT
  // point to its tear set (measured in the op's input space) and w is the
  // running derivative AT THAT MOMENT (seam ops are w-neutral, so before/after
  // this op is the same value).
  //
  // The tear set may be ANY closed set with a computable distance lower
  // bound — the soundness argument below only uses "a ball of radius cutDist
  // around the input point misses the tear set". Cell-wall plane lattices
  // (modFold), a single cut plane (this op), the half-plane sheet trailing a
  // SCREW axis, or the line/point tear of a ROTOINVERSION (the #632 torn
  // space groups, explicitly in scope per its review) all qualify; each op
  // just reports its own geometry's distance. Conservative bounds are always
  // legal — they only shorten steps.
  //
  // SOUNDNESS — why the marcher may step min(DE, seamOut) from a sample p:
  // let g_i be the composite map from formula space to the i-th seam-op input
  // and L_i the accumulated Lipschitz bound carried in w. Take any q with
  // |q − p| < seamOut(p). By induction over the op applications: while both
  // orbits have taken the same branch everywhere, the composite so far is
  // L_i-Lipschitz, so at the i-th seam op |g_i(q) − g_i(p)| ≤ L_i·|q − p| <
  // L_i · cutDist_i/L_i = cutDist_i — and the ball of radius cutDist_i around
  // g_i(p) misses the tear set, so the segment to g_i(q) stays inside one
  // cell: q takes the SAME branch and the induction continues. The whole
  // composite is therefore L-Lipschitz on the ball of radius seamOut(p), so
  // |q − p| < DE(p) additionally gives |F(q) − F(p)| < dist(F(p), surface):
  // NO surface preimage lies within min(DE, seamOut) of p. A ray stepping
  // that much can never jump a seam — the exact guarantee the issue asks for.
  //
  // SCOPE (the review's "state the contract precisely" blocker): the guarantee
  // is EXACT when every active op's wRule ∈ {W_UNCHANGED, W_MUL_SCALE} and
  // none is deApprox (w is then a true global Lipschitz bound — constant
  // per-op factors). With W_MUL_K in the stack it holds at the ENGINE'S
  // STANDARD rigor: w is the same local-derivative bookkeeping the r/|w|
  // estimator itself is built on. Under W_BULB / W_BULB_NUMERIC / deApprox it
  // is best-effort only — but min() can only SHRINK a step, so the clamp
  // never makes an already-approximate formula worse, and those formulas keep
  // their existing APPROX_DESCALE_MUL tightening untouched (renderpolicy is
  // deliberately NOT consulted: seamOut is a geometric distance and is never
  // multiplied by deScale). Classified by seamGuarantee() below.
  //
  // CONSUMER CONTRACT (the review's 🔴 blocker, honoured): the seam bound
  // lives BESIDE the DE, never inside it. mapDE's return value is UNCHANGED
  // in all three tiers; only the primary march STEP is clamped
  // (shader.js / shader_gl.js / cpu.js), floored at a QUARTER of the DE —
  // the floor must be d-relative because a tear's fold-preimages are dense
  // near the attractor, so the running-min seam is ~0 along the whole
  // approach and any absolute floor makes seam formulas crawl (the Hinged
  // Bastion near-black incident: 3759 steps vs 3 plain). min(d, max(seam,
  // d/4)) caps march cost at 4x plain unconditionally. Hit test,
  // bisection refinement, normals, ∇DE/curvature diagnostics, soft shadows,
  // AO and the capture march all read the unclamped DE — a seam can never
  // register as a false surface hit, cast a shadow, or bend a normal.
  //
  // EMISSION is structurally pay-per-use (#125 / opAux discipline): the
  // channel, the clamped step and the membrane debug view are emitted ONLY
  // when the shader variant's op-set contains a seam op (shader.js usesSeam) —
  // every seam-free formula's WGSL is byte-identical to before.
  {
    id: 67,
    key: "hingeFold",
    name: "Hinge Fold",
    wRule: W_UNCHANGED,
    seam: true,
    category: "fold",
    blurb:
      "Cuts space along a flat plane and swings one side around a hinge line like a door — bent, architectural forms no mirror fold can make.",
    params: [
      {
        name: "FoldAngle",
        default: 30.0,
        min: -180.0,
        max: 180.0,
        step: 0.5,
        type: "angle",
        tip: "How far the cut side swings around the hinge. 0 leaves space uncut.",
      },
      {
        name: "CutAngle",
        default: 0.0,
        min: -180.0,
        max: 180.0,
        step: 0.5,
        type: "angle",
        tip: "Orientation of the cut plane around the hinge axis.",
      },
      {
        name: "Axis",
        default: 1.0,
        min: 0.0,
        max: 2.0,
        step: 1.0,
        type: "double",
        tip: "Hinge line direction: 0 X, 1 Y (upright door), 2 Z.",
      },
      {
        name: "Offset",
        default: 0.0,
        min: -4.0,
        max: 4.0,
        step: 0.05,
        type: "double",
        tip: "Slides the hinge line away from the center, moving the cut off the shape's spine.",
      },
    ],
    // Hinge fold (#633, clean-room): in the plane ⊥ Axis with in-plane coords
    // (u, v), the cut line has direction d̂ = (cos φ, sin φ) (φ = CutAngle) and
    // unit normal m̂ = (−sin φ, cos φ); the hinge point is c = Offset·m̂. The
    // half-plane {s > 0}, s = (u,v)·m̂ − Offset, rotates about c by θ =
    // FoldAngle; the other half stays. In 3D that is: rotate the half-SPACE on
    // one side of the cut plane (spanned by the axis direction and d̂) about
    // the hinge LINE (through c, along the axis) — a torn rigid motion.
    //
    // DE derivation (the #633 spec-review blocker — derived, not asserted):
    // each branch is a rigid motion (identity / rotation about a line), so
    // within a branch every singular value of the differential is exactly 1 —
    // per-cell 1-Lipschitz, w untouched: W_UNCHANGED is honest ON EACH CELL.
    // Globally the map is NOT Lipschitz at all — points straddling the cut
    // plane land ~2·sin(θ/2)·(distance to hinge line) apart — which is exactly
    // why W_UNCHANGED alone would be a lie and the seam declaration is
    // MANDATORY: cutDist = |s| is the exact 3D distance to the cut plane (the
    // tear set), and the seam-clamp contract above turns the per-cell
    // isometry back into a sound march. Both claims are proven in
    // core/seamclamp.test.mjs: the per-branch sampled operator norm gate, the
    // fundamental-domain gate (the rotating side lands rotated by exactly θ —
    // the #632-review lesson that Lipschitz sampling alone can pass a fold
    // that never reaches its cell), and a ground-truth gate showing the
    // unclamped DE overestimates across the seam while min(DE, seam) never
    // does.
    //
    // The tear set is the FULL cut plane (every off-hinge point of it moves
    // under the rotation), so cutDist needs no special-casing near the hinge
    // line: |s| is a valid lower bound everywhere, merely conservative ON the
    // line itself (where the map is continuous but not differentiable).
    //
    // Offset rides the opAux lane (op.p3 — first seam op past the inline 3;
    // spliceAux rewrites it, OP_PARAM_ENCODING.md §5.2). NO wgslDf / wgslPt
    // twins: the reference orbit and a delta sample can land on OPPOSITE sides
    // of the cut plane — the classic perturbation glitch the "with margins"
    // qualifier in perturb.js guards against; a seam-aware delta map is real
    // follow-up work, not a freebie (spec review §4).
    wgsl: `
        let hga = i32(op.p2 + 0.5);
        var hgu = pos.x; var hgv = pos.z; // Axis 1 (Y): hinge ⊥ plane is XZ
        if (hga == 0) { hgu = pos.y; hgv = pos.z; }
        else if (hga == 2) { hgu = pos.x; hgv = pos.y; }
        let hcp = radians(op.p1);
        let hmx = -sin(hcp); let hmy = cos(hcp);
        let hgs = hgu * hmx + hgv * hmy - op.p3;
        if (hgs > 0.0) {
            let hox = hmx * op.p3; let hoy = hmy * op.p3;
            let hfa = radians(op.p0);
            let hfc = cos(hfa); let hfs = sin(hfa);
            let hdu = hgu - hox; let hdv = hgv - hoy;
            let hru = hox + hdu * hfc - hdv * hfs;
            let hrv = hoy + hdu * hfs + hdv * hfc;
            if (hga == 0) { pos.y = hru; pos.z = hrv; }
            else if (hga == 2) { pos.x = hru; pos.y = hrv; }
            else { pos.x = hru; pos.z = hrv; }
        }`,
    // #633 seam report — same split contract as modFold's wgslSeam above.
    wgslSeam: `
        seamOut = min(seamOut, abs(hgs) / max(abs(w), 1e-9));`,
    glsl: (v) => `
    // hinge fold — swing the half-space past the cut plane around the hinge
    // line (per-cell isometry, w untouched); report the cut-plane distance
    // into the seam channel (#633).
    {
        int hga = int(${v[2]} + 0.5);
        float hgu = pos.x, hgv = pos.z;
        if (hga == 0) { hgu = pos.y; hgv = pos.z; }
        else if (hga == 2) { hgu = pos.x; hgv = pos.y; }
        float hmx = -sin(${v[1]}), hmy = cos(${v[1]});
        float hgs = hgu * hmx + hgv * hmy - ${v[3]};
        if (hgs > 0.0) {
            float hox = hmx * ${v[3]}, hoy = hmy * ${v[3]};
            float hfc = cos(${v[0]}), hfs = sin(${v[0]});
            float hdu = hgu - hox, hdv = hgv - hoy;
            float hru = hox + hdu * hfc - hdv * hfs;
            float hrv = hoy + hdu * hfs + hdv * hfc;
            if (hga == 0) { pos.y = hru; pos.z = hrv; }
            else if (hga == 2) { pos.x = hru; pos.y = hrv; }
            else { pos.x = hru; pos.z = hrv; }
        }
        g_seam = min(g_seam, abs(hgs) / max(abs(w), 1e-9));
    }`,
    // Desktop iterateJIT_ ABI has no seam channel — the fold itself exports
    // (same map, byte for byte minus the seam report), with the SAME standing
    // caveat modFold's desktop export carries: the desktop marcher can step
    // through the cut plane. The engine's own three tiers all clamp.
    desktopGlsl: (v) => `
    // hinge fold — swing the half-space past the cut plane around the hinge
    // line (per-cell isometry, w untouched). ⚠ no seam channel in this ABI:
    // the host marcher may step through the cut plane (see modFold's note).
    {
        int hga = int(${v[2]} + 0.5);
        float hgu = pos.x, hgv = pos.z;
        if (hga == 0) { hgu = pos.y; hgv = pos.z; }
        else if (hga == 2) { hgu = pos.x; hgv = pos.y; }
        float hmx = -sin(${v[1]}), hmy = cos(${v[1]});
        float hgs = hgu * hmx + hgv * hmy - ${v[3]};
        if (hgs > 0.0) {
            float hox = hmx * ${v[3]}, hoy = hmy * ${v[3]};
            float hfc = cos(${v[0]}), hfs = sin(${v[0]});
            float hdu = hgu - hox, hdv = hgv - hoy;
            float hru = hox + hdu * hfc - hdv * hfs;
            float hrv = hoy + hdu * hfs + hdv * hfc;
            if (hga == 0) { pos.y = hru; pos.z = hrv; }
            else if (hga == 2) { pos.x = hru; pos.y = hrv; }
            else { pos.x = hru; pos.z = hrv; }
        }
    }`,
  },
];

const _byId = new Map(OPERATORS.map((o) => [o.id, o]));
const _byKey = new Map(OPERATORS.map((o) => [o.key, o]));
export const byId = (id) => _byId.get(id);
export const byKey = (key) => _byKey.get(key);

// A formula's DE is sound to the analytic `r/|w|` estimator as long as every
// op accounts for its own w (all our fold-family ops do). A power/escape-time
// op would flip the whole formula to a different DE family — flagged here so
// the UI can warn. For now the palette is all DE-sound.
// The ops that actually run: muted ops are kept in the list (so the user can
// toggle them back) but excluded from rendering, export, and DE classification.
export const activeOps = (formula) => formula.ops.filter((op) => !op.muted);

// The wRule-membership half of soundness — every active op's w-accounting
// CLASS is analytic. Ignores the deApprox tag; use isDeSound for the full
// verdict. Exported for the health badge, which needs to distinguish "sound
// except for an approximate op" (amber) from a genuine DE-family mix (red).
export function deSoundExceptApprox(formula) {
  return activeOps(formula).every((op) => {
    const def = byKey(op.key);
    return def && [W_UNCHANGED, W_MUL_K, W_MUL_SCALE].includes(def.wRule);
  });
}

export function isDeSound(formula) {
  // A deApprox op's w-accounting is best-effort, not exact — it must not let
  // the formula vouch as sound even when its wRule is in the sound set
  // (APPROX_DE.md §1: the one subtle edit).
  return (
    deSoundExceptApprox(formula) &&
    !activeOps(formula).some((op) => byKey(op.key)?.deApprox)
  );
}

// True if ANY active op anywhere in the formula carries the deApprox tag —
// its analytic DE is not a true bound and the step policy tightens deScale
// (APPROX_DE.md §2). ⚠ RECURSIVE by design: walks flat ops + hybrid slot B +
// every scene object (modeled on stability's hybridDeFamily/measureScene —
// NOT on the flat isDeSound/isNumericDE above, which cannot see those
// structures). Muted ops don't count (the scene-mute precedent).
export function isApproxDE(formula) {
  const tagged = (ops) =>
    (ops || []).some((op) => !op.muted && byKey(op.key)?.deApprox);
  // A hybrid walks EVERY slot (A + B/C/D) through the one canonical accessor —
  // a deApprox op hiding in ANY slot (not just A/B) must still tighten the step,
  // so a future slot C/D can't smuggle an approximate op past the marcher.
  if (formula.hybrid) {
    if (hybridSlots(formula).slots.some((s) => tagged(s.ops))) return true;
  } else if (tagged(formula.ops)) return true;
  if (formula.objects)
    for (const o of formula.objects) {
      if (tagged(o.ops)) return true;
      // D0: a shape leaf whose own bound is approximate (leaves.js deApprox —
      // heightfield/Taubin D2 leaves) makes the object approximate too.
      if (leafById(o.shapeId)?.deApprox) return true;
    }
  return false;
}

// #633 — true if ANY active op anywhere in the formula declares a seam
// (discontinuous map + tear-distance report; the seam-clamped DE contract
// above hingeFold). ⚠ RECURSIVE like isApproxDE — walks flat ops, every
// hybrid slot, and every scene object's op chain, because the seam channel
// must be emitted (and the march step clamped) wherever a seam op can run.
// Muted ops don't count (they are excluded from rendering).
export function hasSeamOps(formula) {
  const tagged = (ops) =>
    (ops || []).some((op) => !op.muted && byKey(op.key)?.seam);
  if (formula.hybrid) {
    if (hybridSlots(formula).slots.some((s) => tagged(s.ops))) return true;
  } else if (tagged(formula.ops)) return true;
  if (formula.objects)
    for (const o of formula.objects) if (tagged(o.ops)) return true;
  return false;
}

// #633 — how strong the seam-clamp guarantee is for THIS formula (the
// review's "state the contract as valid iff …" blocker, as a classifier so
// the verdict is inspectable rather than implied):
//   "exact"       every active op is W_UNCHANGED / W_MUL_SCALE and none is
//                 deApprox — w is a true global Lipschitz bound (constant
//                 per-op factors), so the seam pullback is a THEOREM: a ray
//                 stepping min(DE, seam) can never cross a tear.
//   "standard"    W_MUL_K joins the mix (sphere-fold family) — w is the same
//                 local-derivative bookkeeping the r/|w| estimator itself
//                 rests on; the seam bound holds at exactly the engine's
//                 standard DE rigor, no better and no worse.
//   "best-effort" anything else (W_BULB, W_BULB_NUMERIC, or a deApprox op in
//                 the stack) — w is not a Lipschitz bound there, so the clamp
//                 is a strict tightening with no crossing guarantee. Those
//                 formulas keep their existing approx-DE step tightening
//                 (APPROX_DESCALE_MUL) untouched; min() only ever shrinks a
//                 step, so applying the clamp is always safe.
// Flat-formula classifier (the flat wRule walk, like deSoundExceptApprox);
// meaningful only when hasSeamOps(formula) — callers gate on that.
export function seamGuarantee(formula) {
  if (isApproxDE(formula)) return "best-effort";
  const rules = new Set(activeOps(formula).map((op) => byKey(op.key)?.wRule));
  if (rules.has(W_BULB) || rules.has(W_BULB_NUMERIC) || rules.has(undefined))
    return "best-effort";
  if (rules.has(W_MUL_K)) return "standard";
  return "exact";
}

// True if the stack contains an escape-time op (Mandelbulb power). Such a
// formula must use the escape-time DE (deOption 0), not the IFS r/|w|. Deriving
// this from the ops means authoring a bulb from a blank slate "just works" — no
// hidden DE-family switch for the user to find.
export function isEscapeTime(formula) {
  return activeOps(formula).some((op) => {
    const def = byKey(op.key);
    return def && def.wRule === W_BULB;
  });
}

// True if the stack contains an op with no analytic derivative — the whole
// formula must then use the numeric (finite-difference) DE, which ignores w
// entirely. Numeric tolerates any op mix (that's its point), so it wins over
// both the escape-time and IFS routings.
export function isNumericDE(formula) {
  return activeOps(formula).some(
    (op) => byKey(op.key)?.wRule === W_BULB_NUMERIC,
  );
}

// The DE family the preview + export should use: 3 = numeric finite-difference
// if the stack has a no-analytic-dr op, else escape-time (0) if it has a bulb
// op, otherwise the formula's stored deOption (2 = analytic IFS default).
export function effectiveDeOption(formula) {
  if (isNumericDE(formula)) return 3;
  return isEscapeTime(formula) ? 0 : (formula.deOption ?? 2);
}

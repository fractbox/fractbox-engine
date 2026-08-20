// Shape-leaf registry — the D0 single source of truth for scene-object leaves
// (docs/planning/PRIMITIVE_DIFS_D0.md §2.2). A leaf is the closed-form distance
// an object evaluates after (or, in iterated-shape mode, inside) its op chain:
//
//   d = leafDE(s, params) / max(|w|, ε)        final mode
//   d = min over iterations of the same        iterShape (D3)
//   shapeId 0 = no leaf: r/|w| (or the escape log-DE) — the classic IFS dust.
//
// Registry discipline mirrors operators.js: ids are CONTIGUOUS and APPEND-ONLY
// (the share codec stores them — check open sibling PRs before assigning), the
// JS evaluators live in cpu.js LEAF_FNS (same core-module split as ops), and
// every param's UI step must be a multiple of 0.001 (TAG.SHAPES encodes at
// fixed-point ×1000 — leaves.test.mjs asserts this statically).
//
// `deApprox: true` marks leaves whose distance is an approximate bound
// (heightfields, Taubin |f|/|∇f| numerics in Phase D2) — consumed by
// renderpolicy.sceneDeScale and the app health chip exactly like op-level
// deApprox. The 6 launch leaves are all exact SDFs (IQ distfunctions).
//
// WGSL emits `p`/`prm` (vec3f / vec4f), GLSL emits `p`/`prm` (vec3 / vec4) —
// each body is one `return` expression-set with those two names in scope.

export const LEAVES = [
  {
    id: 1,
    key: "box",
    name: "Box",
    glyph: "◻",
    params: [{ name: "Half-extent", def: 0.6, min: 0.05, max: 4, step: 0.01 }],
    wgsl: `let q = abs(p) - vec3f(prm.x);
  return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);`,
    glsl: `vec3 q = abs(p) - vec3(prm.x);
  return length(max(q, vec3(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);`,
  },
  {
    id: 2,
    key: "sphere",
    name: "Sphere",
    glyph: "●",
    params: [{ name: "Radius", def: 0.5, min: 0.05, max: 4, step: 0.01 }],
    wgsl: `return length(p) - prm.x;`,
    glsl: `return length(p) - prm.x;`,
  },
  {
    id: 3,
    key: "torus",
    name: "Torus",
    glyph: "◎",
    params: [
      { name: "Major R", def: 0.6, min: 0.05, max: 4, step: 0.01 },
      { name: "Minor r", def: 0.22, min: 0.02, max: 2, step: 0.01 },
      { name: "Boxy ring", def: 0, min: 0, max: 1, step: 1 },
      { name: "Boxy tube", def: 0, min: 0, max: 1, step: 1 },
    ],
    // Boxy variants (D2 torus widening — the MB3D TorusIFS family): swap the
    // Euclidean length for the Chebyshev max per stage. Params 3/4 default 0,
    // so every pre-D2 torus is bit-identical.
    wgsl: `let lr = select(length(p.xz), max(abs(p.x), abs(p.z)), prm.z > 0.5);
  let q = vec2f(lr - prm.x, p.y);
  let lq = select(length(q), max(abs(q.x), abs(q.y)), prm.w > 0.5);
  return lq - prm.y;`,
    glsl: `float lr = (prm.z > 0.5) ? max(abs(p.x), abs(p.z)) : length(p.xz);
  vec2 q = vec2(lr - prm.x, p.y);
  float lq = (prm.w > 0.5) ? max(abs(q.x), abs(q.y)) : length(q);
  return lq - prm.y;`,
  },
  {
    id: 4,
    key: "cylinder",
    name: "Cylinder",
    glyph: "▮",
    params: [
      { name: "Radius", def: 0.45, min: 0.05, max: 4, step: 0.01 },
      { name: "Height", def: 0.6, min: 0.05, max: 4, step: 0.01 },
    ],
    wgsl: `let d = vec2f(length(p.xz) - prm.x, abs(p.y) - prm.y);
  return min(max(d.x, d.y), 0.0) + length(max(d, vec2f(0.0)));`,
    glsl: `vec2 d = vec2(length(p.xz) - prm.x, abs(p.y) - prm.y);
  return min(max(d.x, d.y), 0.0) + length(max(d, vec2(0.0)));`,
  },
  {
    id: 5,
    key: "capsule",
    name: "Capsule",
    glyph: "▯",
    params: [
      { name: "Radius", def: 0.3, min: 0.05, max: 4, step: 0.01 },
      { name: "Height", def: 0.6, min: 0.05, max: 4, step: 0.01 },
    ],
    wgsl: `var q = p;
  q.y = q.y - clamp(q.y, -prm.y, prm.y);
  return length(q) - prm.x;`,
    glsl: `vec3 q = p; q.y -= clamp(q.y, -prm.y, prm.y);
  return length(q) - prm.x;`,
  },
  {
    id: 6,
    key: "plane",
    unbounded: true,
    name: "Slab",
    glyph: "▬",
    // Width/Depth (#353): 0 = old infinite-in-that-axis behavior (bit-
    // identical — every existing Slab has these unset → leaf default 0), a
    // positive value clips that one axis only, which a Box IFS clamp can't do
    // (it clamps both horizontal axes together).
    params: [
      { name: "Thickness", def: 0, min: 0, max: 2, step: 0.01 },
      { name: "Width", def: 0, min: 0, max: 4, step: 0.01 },
      { name: "Depth", def: 0, min: 0, max: 4, step: 0.01 },
    ],
    wgsl: `var d = abs(p.y) - prm.x;
  if (prm.y > 0.0) { d = max(d, abs(p.x) - prm.y); }
  if (prm.z > 0.0) { d = max(d, abs(p.z) - prm.z); }
  return d;`,
    glsl: `float d = abs(p.y) - prm.x;
  if (prm.y > 0.0) { d = max(d, abs(p.x) - prm.y); }
  if (prm.z > 0.0) { d = max(d, abs(p.z) - prm.z); }
  return d;`,
  },
  // ── D2 batch 1 — the first dIFS leaf wave (PRIMITIVE_DIFS_LEAVES.md) ──────
  // TPMS / implicit-surface leaves march on |f|/|∇f|-style bounds → deApprox
  // (the approx step-scale machinery from APPROX_DE handles them); the
  // lattice/prism constructions are exact per cell (fold isometries + convex
  // SDFs that never over-estimate).
  {
    id: 7,
    key: "gyroid",
    name: "Gyroid",
    glyph: "〰",
    deApprox: true,
    params: [
      { name: "Frequency", def: 3, min: 0.5, max: 12, step: 0.05 },
      { name: "Thickness", def: 0.06, min: 0.005, max: 0.5, step: 0.005 },
      { name: "Level", def: 0, min: -1.4, max: 1.4, step: 0.01 },
      { name: "Bound", def: 1.4, min: 0.2, max: 6, step: 0.01 },
    ],
    // Schoen gyroid implicit sin x cos y + sin y cos z + sin z cos x = c
    // (published TPMS math); |∇g| ≤ √3·f → d = |g−c|/(f√3) − t, sphere-bounded.
    wgsl: `let q = p * prm.x;
  let g = sin(q.x) * cos(q.y) + sin(q.y) * cos(q.z) + sin(q.z) * cos(q.x);
  let d = abs(g - prm.z) / (prm.x * 1.7320508) - prm.y;
  return max(d, length(p) - prm.w);`,
    glsl: `vec3 q = p * prm.x;
  float g = sin(q.x) * cos(q.y) + sin(q.y) * cos(q.z) + sin(q.z) * cos(q.x);
  float d = abs(g - prm.z) / (prm.x * 1.7320508) - prm.y;
  return max(d, length(p) - prm.w);`,
  },
  {
    id: 8,
    key: "schwarzP",
    name: "Schwarz P",
    glyph: "◇",
    deApprox: true,
    params: [
      { name: "Frequency", def: 3, min: 0.5, max: 12, step: 0.05 },
      { name: "Level", def: 0, min: -2.8, max: 2.8, step: 0.01 },
      { name: "Thickness", def: 0.06, min: 0.005, max: 0.5, step: 0.005 },
      { name: "Bound", def: 1.4, min: 0.2, max: 6, step: 0.01 },
    ],
    // Schwarz P implicit cos x + cos y + cos z = c (published TPMS math);
    // |∇| ≤ √3·f. The evaluation origin is shifted by a quarter period (+π/2 per
    // axis) so the c=0 surface passes THROUGH p=0 (a saddle node of the surface)
    // instead of the chamber centre where g=3 — otherwise the default view frames
    // the hollow interior and the camera can't dolly onto any surface (#280). The
    // shift is a rigid translation of the same periodic surface, so the √3·f
    // Lipschitz bound (∇ magnitude unchanged) still holds.
    wgsl: `let q = p * prm.x + vec3f(1.5707963);
  let g = cos(q.x) + cos(q.y) + cos(q.z);
  let d = abs(g - prm.y) / (prm.x * 1.7320508) - prm.z;
  return max(d, length(p) - prm.w);`,
    glsl: `vec3 q = p * prm.x + vec3(1.5707963);
  float g = cos(q.x) + cos(q.y) + cos(q.z);
  float d = abs(g - prm.y) / (prm.x * 1.7320508) - prm.z;
  return max(d, length(p) - prm.w);`,
  },
  {
    id: 9,
    key: "lidinoid",
    name: "Lidinoid",
    glyph: "❋",
    deApprox: true,
    params: [
      { name: "Frequency", def: 3, min: 0.5, max: 12, step: 0.05 },
      { name: "Level", def: 0, min: -1.5, max: 1.5, step: 0.01 },
      { name: "Thickness", def: 0.05, min: 0.005, max: 0.5, step: 0.005 },
      { name: "Bound", def: 1.4, min: 0.2, max: 6, step: 0.01 },
    ],
    // Lidin–Larsson lidinoid implicit (published TPMS literature):
    // ½Σ sin2u cos v sin w − ½Σ cos2u cos2v + 0.15 = c; conservative ∇ bound 3f.
    wgsl: `let q = p * prm.x;
  let g = 0.5 * (sin(2.0 * q.x) * cos(q.y) * sin(q.z)
       + sin(2.0 * q.y) * cos(q.z) * sin(q.x)
       + sin(2.0 * q.z) * cos(q.x) * sin(q.y))
       - 0.5 * (cos(2.0 * q.x) * cos(2.0 * q.y)
       + cos(2.0 * q.y) * cos(2.0 * q.z)
       + cos(2.0 * q.z) * cos(2.0 * q.x)) + 0.15;
  let d = abs(g - prm.y) / (prm.x * 3.0) - prm.z;
  return max(d, length(p) - prm.w);`,
    glsl: `vec3 q = p * prm.x;
  float g = 0.5 * (sin(2.0 * q.x) * cos(q.y) * sin(q.z)
      + sin(2.0 * q.y) * cos(q.z) * sin(q.x)
      + sin(2.0 * q.z) * cos(q.x) * sin(q.y))
      - 0.5 * (cos(2.0 * q.x) * cos(2.0 * q.y)
      + cos(2.0 * q.y) * cos(2.0 * q.z)
      + cos(2.0 * q.z) * cos(2.0 * q.x)) + 0.15;
  float d = abs(g - prm.y) / (prm.x * 3.0) - prm.z;
  return max(d, length(p) - prm.w);`,
  },
  {
    id: 10,
    key: "scherk",
    name: "Scherk Tower",
    glyph: "𝄢",
    deApprox: true,
    params: [
      { name: "Scale", def: 2.5, min: 0.5, max: 8, step: 0.05 },
      { name: "Thickness", def: 0.04, min: 0.005, max: 0.4, step: 0.005 },
      { name: "Bound", def: 1.4, min: 0.2, max: 6, step: 0.01 },
    ],
    // Scherk saddle-tower implicit sin z = sinh x sinh y (published minimal-
    // surface math); Taubin bound |f|/(s + |∇f|) since sinh is unbounded.
    wgsl: `let q = p * prm.x;
  let f = sin(q.z) - sinh(q.x) * sinh(q.y);
  let gx = cosh(q.x) * sinh(q.y);
  let gy = sinh(q.x) * cosh(q.y);
  let gr = prm.x * (1.0 + sqrt(gx * gx + gy * gy + cos(q.z) * cos(q.z)));
  return max(abs(f) / gr - prm.y, length(p) - prm.z);`,
    glsl: `vec3 q = p * prm.x;
  float f = sin(q.z) - sinh(q.x) * sinh(q.y);
  float gx = cosh(q.x) * sinh(q.y);
  float gy = sinh(q.x) * cosh(q.y);
  float gr = prm.x * (1.0 + sqrt(gx * gx + gy * gy + cos(q.z) * cos(q.z)));
  return max(abs(f) / gr - prm.y, length(p) - prm.z);`,
  },
  {
    id: 11,
    key: "hexGrid",
    unbounded: true,
    name: "Hex Grid",
    glyph: "⬡",
    params: [
      { name: "Cell size", def: 0.5, min: 0.05, max: 2, step: 0.01 },
      { name: "Z thickness", def: 0.3, min: 0.01, max: 4, step: 0.01 },
      { name: "Wall", def: 0.04, min: 0.005, max: 0.5, step: 0.005 },
      { name: "Cell radius ×", def: 0.9, min: 0.2, max: 1.2, step: 0.01 },
    ],
    // Honeycomb walls: dual-offset rectangular fold to the nearest hex center
    // (exact partition), IQ hexagon SDF ring |hd| − wall, prism-extruded in z.
    // #353 round 5 — the LATTICE the fold partitions was not a hex lattice.
    // sdHexagon below is flat-top: flat edges at y = ±r (r = apothem), vertices
    // on the x axis at ±2r/√3. The triangular lattice whose Voronoi cell is
    // that hexagon has nearest neighbours at (0, ±2r) and (±√3r, ±r), i.e. the
    // centred-rectangular period is (2√3·r, 2·r). With r = prm.x/2 at cellR 1
    // that is cs = (√3·s, s). The code used cs = (1.5·s, √3·s): cs.y was 2×
    // too big AND cs.x carried the OTHER convention's 1.5·R constant, so the
    // union of the two sublattices had a shortest-vector shell of multiplicity
    // 4 (rhombic), not 6 — no hexagonal lattice at all. Consequences measured
    // by brute force: the emitted distance OVERestimated the true distance to
    // the ring network by up to 3.2e−2 (a Lipschitz violation → raymarch
    // overshoot), and per-scanline wall coverage swung 23% → 96.6%, the
    // periodic smear bands reported as "never actually a clean hex grid".
    // No single apothem coefficient can fix it — cs.x needs cellR 0.866 and
    // cs.y needs cellR 1.732 simultaneously, which is why sweeping the
    // constant never worked. Fixed by correcting cs; the drawn hexagon size
    // (r = prm.x·0.5·prm.w) is unchanged, so "Cell size" now reads as the
    // cell's flat-to-flat width and cells tile exactly at Cell radius × = 1.
    // Post-fix the emitted value matches a brute-force point-to-polygon
    // distance over every neighbouring cell to float epsilon (see
    // d2leaves.test.mjs). The corner-aware sdHexagon fix below still stands:
    // #353 — the "IQ hexagon SDF" comment above was aspirational, not actual:
    // `max(q.x·0.866+q.y·0.5, q.y) − r` is only the max-of-two-half-planes
    // shortcut, which is exact along each face but UNDERESTIMATES distance
    // near the 6 corners (same defect as `max(q.x,q.y)` vs. the real box SDF)
    // — the honeycomb's walls never quite closed at a vertex, reading as an
    // "approximation" rather than a clean grid. Swapped in IQ's actual
    // corner-correct sdHexagon (fold across the edge normal, clamp, length) —
    // verified against a brute-force polygon distance (0 error) before
    // porting. Cell interior (hd < 0, no wall) is untouched either way.
    wgsl: `let cs = vec2f(prm.x * 1.7320508, prm.x);
  let a = (fract(p.xy / cs) - 0.5) * cs;
  let b = (fract(p.xy / cs + 0.5) - 0.5) * cs;
  var q = select(b, a, dot(a, a) < dot(b, b));
  q = abs(q);
  let r = prm.x * 0.5 * prm.w;
  let k = vec2f(-0.8660254, 0.5);
  q -= 2.0 * min(dot(k, q), 0.0) * k;
  let cx = clamp(q.x, -0.5773503 * r, 0.5773503 * r);
  let hd = length(q - vec2f(cx, r)) * sign(q.y - r);
  return max(abs(hd) - prm.z, abs(p.z) - prm.y);`,
    glsl: `vec2 cs = vec2(prm.x * 1.7320508, prm.x);
  vec2 a = (fract(p.xy / cs) - 0.5) * cs;
  vec2 b = (fract(p.xy / cs + 0.5) - 0.5) * cs;
  vec2 q = (dot(a, a) < dot(b, b)) ? a : b;
  q = abs(q);
  float r = prm.x * 0.5 * prm.w;
  vec2 k = vec2(-0.8660254, 0.5);
  q -= 2.0 * min(dot(k, q), 0.0) * k;
  float cx = clamp(q.x, -0.5773503 * r, 0.5773503 * r);
  float hd = length(q - vec2(cx, r)) * sign(q.y - r);
  return max(abs(hd) - prm.z, abs(p.z) - prm.y);`,
  },
  {
    id: 12,
    key: "triGrid",
    unbounded: true,
    name: "Tri Grid",
    glyph: "◬",
    params: [
      { name: "Cell size", def: 0.5, min: 0.05, max: 2, step: 0.01 },
      { name: "Z thickness", def: 0.3, min: 0.01, max: 4, step: 0.01 },
      { name: "Wall", def: 0.04, min: 0.005, max: 0.5, step: 0.005 },
    ],
    // Triangular lattice: three wall-plane families at 0°/60°/120° (exact
    // periodic planes, min-union), prism-extruded in z.
    wgsl: `let s = prm.x;
  let t0 = abs(fract(p.y / s + 0.5) - 0.5) * s;
  let d1 = dot(p.xy, vec2f(0.8660254, 0.5));
  let t1 = abs(fract(d1 / s + 0.5) - 0.5) * s;
  let d2 = dot(p.xy, vec2f(-0.8660254, 0.5));
  let t2 = abs(fract(d2 / s + 0.5) - 0.5) * s;
  return max(min(t0, min(t1, t2)) - prm.z, abs(p.z) - prm.y);`,
    glsl: `float s = prm.x;
  float t0 = abs(fract(p.y / s + 0.5) - 0.5) * s;
  float d1 = dot(p.xy, vec2(0.8660254, 0.5));
  float t1 = abs(fract(d1 / s + 0.5) - 0.5) * s;
  float d2 = dot(p.xy, vec2(-0.8660254, 0.5));
  float t2 = abs(fract(d2 / s + 0.5) - 0.5) * s;
  return max(min(t0, min(t1, t2)) - prm.z, abs(p.z) - prm.y);`,
  },
  {
    id: 13,
    key: "gear",
    name: "Gear",
    glyph: "⚙",
    deApprox: true,
    params: [
      { name: "Teeth", def: 12, min: 3, max: 48, step: 1 },
      { name: "Radius", def: 1, min: 0.1, max: 4, step: 0.01 },
      { name: "Z thickness", def: 0.12, min: 0.01, max: 2, step: 0.005 },
      { name: "Tooth depth", def: 0.18, min: 0.02, max: 1, step: 0.005 },
    ],
    // Ring + angular-repeated tooth boxes (IQ opRep + 2D box in polar-unrolled
    // coords — the unroll distorts arc length → approximate).
    wgsl: `let l = length(p.xz);
  let ring = max(l - prm.y, prm.y * 0.55 - l);
  let sector = 6.2831853 / prm.x;
  let ang = atan2(p.z, p.x);
  let am = (fract(ang / sector + 0.5) - 0.5) * sector;
  let tq = vec2f(l - prm.y - prm.w * 0.5, am * prm.y);
  let tb = abs(tq) - vec2f(prm.w * 0.5, sector * prm.y * 0.18);
  let tooth = length(max(tb, vec2f(0.0))) + min(max(tb.x, tb.y), 0.0);
  let d2 = min(ring, tooth);
  return max(d2, abs(p.y) - prm.z);`,
    glsl: `float l = length(p.xz);
  float ring = max(l - prm.y, prm.y * 0.55 - l);
  float sector = 6.2831853 / prm.x;
  float ang = atan(p.z, p.x);
  float am = (fract(ang / sector + 0.5) - 0.5) * sector;
  vec2 tq = vec2(l - prm.y - prm.w * 0.5, am * prm.y);
  vec2 tb = abs(tq) - vec2(prm.w * 0.5, sector * prm.y * 0.18);
  float tooth = length(max(tb, vec2(0.0))) + min(max(tb.x, tb.y), 0.0);
  float d2 = min(ring, tooth);
  return max(d2, abs(p.y) - prm.z);`,
  },
  {
    id: 14,
    key: "mengerPlate",
    name: "Menger Plate",
    glyph: "▦",
    params: [
      { name: "Size", def: 1, min: 0.1, max: 4, step: 0.01 },
      { name: "Thickness ×", def: 0.25, min: 0.02, max: 1, step: 0.01 },
      { name: "Detail", def: 4, min: 1, max: 6, step: 1 },
    ],
    // IQ menger sponge article (published SDF): box base (plate aspect via
    // thickness×), iterated cross intersection at ×3 scale.
    wgsl: `let sz = prm.x;
  let q0 = p / sz;
  let bq = abs(q0) - vec3f(1.0, 1.0, prm.y);
  var d = length(max(bq, vec3f(0.0))) + min(max(bq.x, max(bq.y, bq.z)), 0.0);
  var sc = 1.0;
  let it = u32(clamp(prm.z, 1.0, 6.0));
  for (var m: u32 = 0u; m < it; m = m + 1u) {
    let a = (fract(q0 * sc * 0.5) - 0.5) * 2.0;
    sc = sc * 3.0;
    let r = abs(1.0 - 3.0 * abs(a));
    let da = max(r.x, r.y);
    let db = max(r.y, r.z);
    let dc = max(r.z, r.x);
    let c = (min(da, min(db, dc)) - 1.0) / sc;
    d = max(d, c);
  }
  return d * sz;`,
    glsl: `float sz = prm.x;
  vec3 q0 = p / sz;
  vec3 bq = abs(q0) - vec3(1.0, 1.0, prm.y);
  float d = length(max(bq, vec3(0.0))) + min(max(bq.x, max(bq.y, bq.z)), 0.0);
  float sc = 1.0;
  int it = int(clamp(prm.z, 1.0, 6.0));
  for (int m = 0; m < 6; m++) {
    if (m >= it) break;
    vec3 a = (fract(q0 * sc * 0.5) - 0.5) * 2.0;
    sc *= 3.0;
    vec3 r = abs(1.0 - 3.0 * abs(a));
    float da = max(r.x, r.y);
    float db = max(r.y, r.z);
    float dc = max(r.z, r.x);
    float c = (min(da, min(db, dc)) - 1.0) / sc;
    d = max(d, c);
  }
  return d * sz;`,
  },
  {
    id: 15,
    key: "knotPQ",
    name: "Torus Knot",
    glyph: "♾",
    deApprox: true,
    params: [
      { name: "p (around)", def: 2, min: 1, max: 5, step: 1 },
      { name: "q (through)", def: 3, min: 1, max: 9, step: 1 },
      { name: "Ring R", def: 1, min: 0.2, max: 4, step: 0.01 },
      { name: "Tube r", def: 0.15, min: 0.02, max: 1, step: 0.005 },
    ],
    // (p,q)-torus-knot: polar unroll + min over the p strand crossings in the
    // meridian plane (standard published technique; the unroll is approximate).
    // The knot rides a carrier torus of minor radius 0.35·R.
    wgsl: `let l = length(p.xz);
  let theta = atan2(p.z, p.x);
  let np = clamp(prm.x, 1.0, 5.0);
  let rr = prm.z * 0.35;
  var d = 1.0e9;
  for (var k: f32 = 0.0; k < 5.0; k = k + 1.0) {
    if (k >= np) { break; }
    let ang = (theta + 6.2831853 * k) * prm.y / np;
    let c = vec2f(prm.z + rr * cos(ang), rr * sin(ang));
    d = min(d, length(vec2f(l, p.y) - c));
  }
  return d - prm.w;`,
    glsl: `float l = length(p.xz);
  float theta = atan(p.z, p.x);
  float np = clamp(prm.x, 1.0, 5.0);
  float rr = prm.z * 0.35;
  float d = 1.0e9;
  for (int k = 0; k < 5; k++) {
    if (float(k) >= np) break;
    float ang = (theta + 6.2831853 * float(k)) * prm.y / np;
    vec2 c = vec2(prm.z + rr * cos(ang), rr * sin(ang));
    d = min(d, length(vec2(l, p.y) - c));
  }
  return d - prm.w;`,
  },
  {
    id: 16,
    key: "loresVoxel",
    name: "Voxel Shape",
    glyph: "🀫",
    deApprox: true,
    params: [
      { name: "Base (0● 1◎ 2◼)", def: 0, min: 0, max: 2, step: 1 },
      { name: "Size", def: 1, min: 0.1, max: 3, step: 0.01 },
      { name: "Voxel size", def: 0.22, min: 0.04, max: 1, step: 0.005 },
      { name: "Rounding", def: 0.01, min: 0, max: 0.2, step: 0.005 },
    ],
    // Voxelized base shape (Aexion lores concept, clean-room): quantize to the
    // cell center, render the cell's cube when the base SDF says "inside",
    // else march by the base distance. Cross-cell nearest-neighbor error →
    // approximate.
    wgsl: `let res = prm.z;
  let c = (floor(p / res) + 0.5) * res;
  var db: f32;
  if (prm.x > 1.5) { let b = abs(c) - vec3f(prm.y); db = length(max(b, vec3f(0.0))) + min(max(b.x, max(b.y, b.z)), 0.0); }
  else if (prm.x > 0.5) { let t = vec2f(length(c.xz) - prm.y, c.y); db = length(t) - prm.y * 0.35; }
  else { db = length(c) - prm.y; }
  let bq = abs(p - c) - vec3f(res * 0.5 - prm.w);
  let cube = length(max(bq, vec3f(0.0))) + min(max(bq.x, max(bq.y, bq.z)), 0.0) - prm.w;
  return select(max(db, res * 0.25), cube, db < 0.0);`,
    glsl: `float res = prm.z;
  vec3 c = (floor(p / res) + 0.5) * res;
  float db;
  if (prm.x > 1.5) { vec3 b = abs(c) - vec3(prm.y); db = length(max(b, vec3(0.0))) + min(max(b.x, max(b.y, b.z)), 0.0); }
  else if (prm.x > 0.5) { vec2 t = vec2(length(c.xz) - prm.y, c.y); db = length(t) - prm.y * 0.35; }
  else { db = length(c) - prm.y; }
  vec3 bq = abs(p - c) - vec3(res * 0.5 - prm.w);
  float cube = length(max(bq, vec3(0.0))) + min(max(bq.x, max(bq.y, bq.z)), 0.0) - prm.w;
  return (db < 0.0) ? cube : max(db, res * 0.25);`,
  },
  // ── D2 batch 2 — cages, shells and frames (PRIMITIVE_DIFS_LEAVES.md) ──────
  {
    id: 17,
    key: "helix",
    unbounded: true,
    name: "Helix",
    glyph: "🌀",
    deApprox: true,
    params: [
      { name: "Pitch", def: 0.5, min: 0.05, max: 4, step: 0.01 },
      { name: "Strands", def: 2, min: 1, max: 6, step: 1 },
      { name: "String r", def: 0.1, min: 0.01, max: 1, step: 0.005 },
      { name: "Radius", def: 0.7, min: 0.05, max: 4, step: 0.01 },
    ],
    // Distance to a helix curve on a cylinder (radius prm.w, pitch prm.x per
    // turn, prm.y strands): nearest-turn point sampling (k−1, k, k+1) around
    // the unrolled parameter — the standard published approximation.
    wgsl: `let theta = atan2(p.z, p.x);
  let ns = clamp(prm.y, 1.0, 6.0);
  var d = 1.0e9;
  for (var s2: f32 = 0.0; s2 < 6.0; s2 = s2 + 1.0) {
    if (s2 >= ns) { break; }
    let ph = theta + s2 * 6.2831853 / ns;
    let k0 = round(p.y / prm.x - ph / 6.2831853);
    for (var dk: f32 = -1.0; dk <= 1.0; dk = dk + 1.0) {
      let t = ph + (k0 + dk) * 6.2831853;
      let cy = t * prm.x / 6.2831853;
      let cp = vec3f(prm.w * cos(t), cy, prm.w * sin(t));
      d = min(d, length(p - cp));
    }
  }
  return d - prm.z;`,
    glsl: `float theta = atan(p.z, p.x);
  float ns = clamp(prm.y, 1.0, 6.0);
  float d = 1.0e9;
  for (int s2 = 0; s2 < 6; s2++) {
    if (float(s2) >= ns) break;
    float ph = theta + float(s2) * 6.2831853 / ns;
    float k0 = floor(p.y / prm.x - ph / 6.2831853 + 0.5);
    for (int dk = -1; dk <= 1; dk++) {
      float t = ph + (k0 + float(dk)) * 6.2831853;
      float cy = t * prm.x / 6.2831853;
      vec3 cp = vec3(prm.w * cos(t), cy, prm.w * sin(t));
      d = min(d, length(p - cp));
    }
  }
  return d - prm.z;`,
  },
  {
    id: 18,
    key: "helixStairs",
    unbounded: true,
    name: "Spiral Stairs",
    glyph: "🪜",
    deApprox: true,
    params: [
      { name: "Steps/turn", def: 12, min: 4, max: 32, step: 1 },
      { name: "Step thick", def: 0.05, min: 0.01, max: 0.4, step: 0.005 },
      { name: "Width", def: 0.8, min: 0.1, max: 1, step: 0.01 },
      { name: "Pitch", def: 1.2, min: 0.1, max: 4, step: 0.01 },
    ],
    // #353 round 7 — each tread is now an EXACT annular-sector slab (a wedge
    // of the annulus [1-Width, 1], thickness 2*prm.y, at its spiral height),
    // and the leaf is the min-union over the three nearest sectors (k-1, k,
    // k+1 — the same 3-candidate pattern as the Helix leaf), each at its own
    // nearest turn. A min of exact SDFs is continuous everywhere, so the old
    // seam-blend + riser-notch construction (which fixed the round-2 seam
    // artifact but bent the tread profile — the reporter's "how did we get
    // this profile instead of rectangle") is gone entirely: flat tops,
    // straight risers, and a small edge rounding (30% of the half-thickness,
    // the reporter's "~10% corner" octagon ask). Width still spans inward
    // from the FIXED outer rim (radius 1, the earlier MB3D-matching fix).
    // Out-of-sector distance is exact too: past the wedge's end face the
    // nearest point is on that face's rectangle — hypot(face-plane box
    // distance, perpendicular) — valid for any sector half-angle < 90 deg
    // (steps >= 4 guarantees it).
    wgsl: `let l = length(p.xz);
  let v = atan2(p.z, p.x) / 6.2831853 * prm.x;
  let rr = min(0.3 * prm.y, 0.2 * prm.z);
  let r1 = 1.0 - rr;
  let r0 = 1.0 - prm.z + rr;
  var d = 1.0e9;
  for (var dk: f32 = -1.0; dk <= 1.0; dk = dk + 1.0) {
    let k = floor(v) + dk;
    let m = round(p.y / prm.w - k / prm.x);
    let dy = abs(p.y - (k / prm.x + m) * prm.w) - (prm.y - rr);
    let dv = abs(v - (k + 0.5));
    if (dv <= 0.5) {
      let dr = max(l - r1, r0 - l);
      d = min(d, length(max(vec2f(dr, dy), vec2f(0.0))) + min(max(dr, dy), 0.0));
    } else {
      let ang = (dv - 0.5) / prm.x * 6.2831853;
      let u2 = l * cos(ang);
      let dr = max(u2 - r1, r0 - u2);
      let face = length(max(vec2f(dr, dy), vec2f(0.0))) + min(max(dr, dy), 0.0);
      d = min(d, length(vec2f(max(face, 0.0), l * sin(ang))));
    }
  }
  return d - rr;`,
    glsl: `float l = length(p.xz);
  float v = atan(p.z, p.x) / 6.2831853 * prm.x;
  float rr = min(0.3 * prm.y, 0.2 * prm.z);
  float r1 = 1.0 - rr;
  float r0 = 1.0 - prm.z + rr;
  float d = 1.0e9;
  for (int dk = -1; dk <= 1; dk++) {
    float k = floor(v) + float(dk);
    float m = floor(p.y / prm.w - k / prm.x + 0.5);
    float dy = abs(p.y - (k / prm.x + m) * prm.w) - (prm.y - rr);
    float dv = abs(v - (k + 0.5));
    if (dv <= 0.5) {
      float dr = max(l - r1, r0 - l);
      d = min(d, length(max(vec2(dr, dy), vec2(0.0))) + min(max(dr, dy), 0.0));
    } else {
      float ang = (dv - 0.5) / prm.x * 6.2831853;
      float u2 = l * cos(ang);
      float dr = max(u2 - r1, r0 - u2);
      float face = length(max(vec2(dr, dy), vec2(0.0))) + min(max(dr, dy), 0.0);
      d = min(d, length(vec2(max(face, 0.0), l * sin(ang))));
    }
  }
  return d - rr;`,
  },
  {
    id: 19,
    key: "sphereCage",
    name: "Sphere Cage",
    glyph: "🕸",
    deApprox: true,
    params: [
      { name: "Radius", def: 1, min: 0.1, max: 4, step: 0.01 },
      { name: "Wire r", def: 0.035, min: 0.005, max: 0.4, step: 0.005 },
      { name: "Parallels", def: 6, min: 2, max: 16, step: 1 },
      { name: "Meridians", def: 8, min: 2, max: 24, step: 1 },
    ],
    // Lat/long wireframe: min-union of the nearest parallel ring and nearest
    // meridian great circle (quantized angles — derivable construction).
    wgsl: `let lxz = length(p.xz);
  let phi = atan2(lxz, p.y);
  let phq = clamp(round(phi * prm.z / 3.14159265) * 3.14159265 / prm.z, 0.0, 3.14159265);
  let dpar = length(vec2f(lxz - prm.x * sin(phq), p.y - prm.x * cos(phq)));
  let thq = round(atan2(p.z, p.x) * prm.w / 3.14159265) * 3.14159265 / prm.w;
  let u = p.x * cos(thq) + p.z * sin(thq);
  let v = -p.x * sin(thq) + p.z * cos(thq);
  let dmer = length(vec2f(length(vec2f(u, p.y)) - prm.x, v));
  return min(dpar, dmer) - prm.y;`,
    glsl: `float lxz = length(p.xz);
  float phi = atan(lxz, p.y);
  float phq = clamp(floor(phi * prm.z / 3.14159265 + 0.5) * 3.14159265 / prm.z, 0.0, 3.14159265);
  float dpar = length(vec2(lxz - prm.x * sin(phq), p.y - prm.x * cos(phq)));
  float thq = floor(atan(p.z, p.x) * prm.w / 3.14159265 + 0.5) * 3.14159265 / prm.w;
  float u = p.x * cos(thq) + p.z * sin(thq);
  float v = -p.x * sin(thq) + p.z * cos(thq);
  float dmer = length(vec2(length(vec2(u, p.y)) - prm.x, v));
  return min(dpar, dmer) - prm.y;`,
  },
  {
    id: 20,
    key: "sliceCage",
    name: "Slice Cage",
    glyph: "🍊",
    deApprox: true,
    params: [
      { name: "Radius", def: 1, min: 0.1, max: 4, step: 0.01 },
      { name: "Slices", def: 8, min: 2, max: 24, step: 1 },
      { name: "Thickness", def: 0.05, min: 0.005, max: 0.5, step: 0.005 },
      { name: "Swirl", def: 0, min: -3, max: 3, step: 0.01 },
    ],
    // Angular repeat of meridian slabs on a sphere shell (orange-slice cage);
    // swirl shears the slice angle with height.
    wgsl: `let shell = abs(length(p) - prm.x) - prm.z;
  let thq = round((atan2(p.z, p.x) + prm.w * p.y) * prm.y / 3.14159265) * 3.14159265 / prm.y - prm.w * p.y;
  let v = -p.x * sin(thq) + p.z * cos(thq);
  return max(shell, abs(v) - prm.z);`,
    glsl: `float shell = abs(length(p) - prm.x) - prm.z;
  float thq = floor(((atan(p.z, p.x) + prm.w * p.y) * prm.y / 3.14159265) + 0.5) * 3.14159265 / prm.y - prm.w * p.y;
  float v = -p.x * sin(thq) + p.z * cos(thq);
  return max(shell, abs(v) - prm.z);`,
  },
  {
    id: 21,
    key: "waveSurface",
    unbounded: true,
    name: "Wave Surface",
    glyph: "🌊",
    deApprox: true,
    params: [
      { name: "Frequency", def: 4, min: 0.2, max: 16, step: 0.05 },
      { name: "Amplitude", def: 0.25, min: 0.01, max: 2, step: 0.005 },
      { name: "Mode (0— 1◎ 2🌀)", def: 1, min: 0, max: 2, step: 1 },
      { name: "Thickness", def: 0.05, min: 0.005, max: 0.5, step: 0.005 },
    ],
    // Sine heightfield y = A sin(f·s) with s = x (linear) / r (circular) /
    // r+θ (spiral); Lipschitz-normalized by 1/√(1+(A f)²).
    wgsl: `let r = length(p.xz);
  var s2 = p.x;
  if (prm.z > 1.5) { s2 = r + atan2(p.z, p.x); }
  else if (prm.z > 0.5) { s2 = r; }
  let h = prm.y * sin(prm.x * s2);
  let lip = inverseSqrt(1.0 + prm.x * prm.x * prm.y * prm.y);
  return (abs(p.y - h)) * lip - prm.w;`,
    glsl: `float r = length(p.xz);
  float s2 = p.x;
  if (prm.z > 1.5) { s2 = r + atan(p.z, p.x); }
  else if (prm.z > 0.5) { s2 = r; }
  float h = prm.y * sin(prm.x * s2);
  float lip = inversesqrt(1.0 + prm.x * prm.x * prm.y * prm.y);
  return abs(p.y - h) * lip - prm.w;`,
  },
  {
    id: 22,
    key: "kleinBagel",
    name: "Klein Bagel",
    glyph: "🥯",
    deApprox: true,
    params: [
      { name: "Ring R", def: 1, min: 0.2, max: 4, step: 0.01 },
      { name: "Lobe r", def: 0.3, min: 0.05, max: 2, step: 0.01 },
      { name: "Twist", def: 1, min: 0, max: 3, step: 1 },
      { name: "Thickness", def: 0.04, min: 0.005, max: 0.4, step: 0.005 },
    ],
    // Klein-bottle figure-8 immersion (published parametric): a figure-8
    // cross-section (two tangent circles) rotating twist/2 turns per lap.
    wgsl: `let theta = atan2(p.z, p.x);
  let u0 = length(p.xz) - prm.x;
  let a = theta * 0.5 * prm.z;
  let u = u0 * cos(a) + p.y * sin(a);
  let v = -u0 * sin(a) + p.y * cos(a);
  let d8 = min(length(vec2f(u, v - prm.y * 0.5)), length(vec2f(u, v + prm.y * 0.5)));
  return abs(d8 - prm.y * 0.5) - prm.w;`,
    glsl: `float theta = atan(p.z, p.x);
  float u0 = length(p.xz) - prm.x;
  float a = theta * 0.5 * prm.z;
  float u = u0 * cos(a) + p.y * sin(a);
  float v = -u0 * sin(a) + p.y * cos(a);
  float d8 = min(length(vec2(u, v - prm.y * 0.5)), length(vec2(u, v + prm.y * 0.5)));
  return abs(d8 - prm.y * 0.5) - prm.w;`,
  },
  {
    id: 23,
    key: "seashell",
    name: "Seashell",
    glyph: "🐚",
    deApprox: true,
    params: [
      { name: "Tightness", def: 0.15, min: 0.02, max: 0.5, step: 0.005 },
      { name: "Tube ×", def: 0.35, min: 0.05, max: 1, step: 0.005 },
      { name: "Pitch", def: 0.12, min: 0, max: 1, step: 0.005 },
      { name: "Turns", def: 4, min: 1, max: 8, step: 1 },
    ],
    // Logarithmic conch spiral (standard shell math): tube centered on
    // r = e^(b·t), z = pitch·t, tube radius growing with r; nearest-turn
    // parameter sampling like the helix.
    wgsl: `let theta = atan2(p.z, p.x);
  let l = max(length(p.xz), 1e-4);
  let b = prm.x;
  let tGuess = log(l) / b;
  let k0 = round((tGuess - theta) / 6.2831853);
  var d = 1.0e9;
  let tmax = prm.w * 6.2831853;
  for (var dk: f32 = -1.0; dk <= 1.0; dk = dk + 1.0) {
    let t = clamp(theta + (k0 + dk) * 6.2831853, 0.0, tmax);
    let rc = exp(b * t);
    let cp = vec3f(rc * cos(t), prm.z * t, rc * sin(t));
    d = min(d, length(p - cp) - prm.y * rc * 0.5);
  }
  return d;`,
    glsl: `float theta = atan(p.z, p.x);
  float l = max(length(p.xz), 1e-4);
  float b = prm.x;
  float tGuess = log(l) / b;
  float k0 = floor((tGuess - theta) / 6.2831853 + 0.5);
  float d = 1.0e9;
  float tmax = prm.w * 6.2831853;
  for (int dk = -1; dk <= 1; dk++) {
    float t = clamp(theta + (k0 + float(dk)) * 6.2831853, 0.0, tmax);
    float rc = exp(b * t);
    vec3 cp = vec3(rc * cos(t), prm.z * t, rc * sin(t));
    d = min(d, length(p - cp) - prm.y * rc * 0.5);
  }
  return d;`,
  },
  {
    id: 24,
    key: "dini",
    name: "Dini Horn",
    glyph: "📯",
    deApprox: true,
    params: [
      { name: "Radius", def: 0.8, min: 0.1, max: 3, step: 0.01 },
      { name: "Taper", def: 0.6, min: 0.05, max: 2, step: 0.01 },
      { name: "Height", def: 2, min: 0.2, max: 6, step: 0.01 },
      { name: "Thickness", def: 0.04, min: 0.005, max: 0.4, step: 0.005 },
    ],
    // Twisted pseudosphere horn (Dini-style, derivable): exponentially
    // tapering revolved profile, height-clamped; shell of the profile.
    wgsl: `let yc = clamp(p.y, 0.0, prm.z);
  let rr = prm.x * exp(-prm.y * yc);
  let d2 = length(vec2f(length(p.xz) - rr, p.y - yc));
  return d2 - prm.w;`,
    glsl: `float yc = clamp(p.y, 0.0, prm.z);
  float rr = prm.x * exp(-prm.y * yc);
  float d2 = length(vec2(length(p.xz) - rr, p.y - yc));
  return d2 - prm.w;`,
  },
  {
    id: 25,
    key: "room",
    name: "Room",
    glyph: "🏠",
    params: [
      { name: "Width", def: 1, min: 0.1, max: 4, step: 0.01 },
      { name: "Height", def: 0.8, min: 0.1, max: 4, step: 0.01 },
      { name: "Depth", def: 1, min: 0.1, max: 4, step: 0.01 },
      { name: "Wall", def: 0.05, min: 0.005, max: 0.5, step: 0.005 },
    ],
    // #353: a closed 6-sided shell (|sdBox| − wall) is indistinguishable from
    // Round Box at any exterior camera angle — from outside, a hollow box
    // looks exactly like a solid one. Astiglic's own reference image (open
    // corner, floor + 2 back walls, seen from the missing front/top/+x side)
    // is what a "room" needs to actually read as one, and it needs no 5th
    // param: only 3 of the 6 faces are solid (floor at −y, walls at −z/−x),
    // each an exact sdBox plate of half-thickness Wall inset from that face —
    // the +x/+y/+z sides are simply absent, so the interior is always
    // visible without any reversed-normal/backface trick.
    wgsl: `let hw = max(prm.w, 0.001);
  let qf = abs(p - vec3f(0.0, -prm.y + hw, 0.0)) - vec3f(prm.x, hw, prm.z);
  let df = length(max(qf, vec3f(0.0))) + min(max(qf.x, max(qf.y, qf.z)), 0.0);
  let qb = abs(p - vec3f(0.0, 0.0, -prm.z + hw)) - vec3f(prm.x, prm.y, hw);
  let db = length(max(qb, vec3f(0.0))) + min(max(qb.x, max(qb.y, qb.z)), 0.0);
  let qs = abs(p - vec3f(-prm.x + hw, 0.0, 0.0)) - vec3f(hw, prm.y, prm.z);
  let ds = length(max(qs, vec3f(0.0))) + min(max(qs.x, max(qs.y, qs.z)), 0.0);
  return min(df, min(db, ds));`,
    glsl: `float hw = max(prm.w, 0.001);
  vec3 qf = abs(p - vec3(0.0, -prm.y + hw, 0.0)) - vec3(prm.x, hw, prm.z);
  float df = length(max(qf, vec3(0.0))) + min(max(qf.x, max(qf.y, qf.z)), 0.0);
  vec3 qb = abs(p - vec3(0.0, 0.0, -prm.z + hw)) - vec3(prm.x, prm.y, hw);
  float db = length(max(qb, vec3(0.0))) + min(max(qb.x, max(qb.y, qb.z)), 0.0);
  vec3 qs = abs(p - vec3(-prm.x + hw, 0.0, 0.0)) - vec3(hw, prm.y, prm.z);
  float ds = length(max(qs, vec3(0.0))) + min(max(qs.x, max(qs.y, qs.z)), 0.0);
  return min(df, min(db, ds));`,
  },
  {
    id: 26,
    key: "roundBox",
    name: "Round Box",
    glyph: "▢",
    params: [
      { name: "Width", def: 0.6, min: 0.05, max: 4, step: 0.01 },
      { name: "Height", def: 0.6, min: 0.05, max: 4, step: 0.01 },
      { name: "Depth", def: 0.6, min: 0.05, max: 4, step: 0.01 },
      { name: "Corner r", def: 0.1, min: 0, max: 1, step: 0.005 },
    ],
    // IQ sdRoundBox (published).
    wgsl: `let q = abs(p) - vec3f(prm.x, prm.y, prm.z) + vec3f(prm.w);
  return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - prm.w;`,
    glsl: `vec3 q = abs(p) - vec3(prm.x, prm.y, prm.z) + vec3(prm.w);
  return length(max(q, vec3(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - prm.w;`,
  },
  {
    id: 27,
    key: "boxFrame",
    name: "Box Frame",
    glyph: "⬚",
    params: [
      { name: "Size", def: 0.8, min: 0.05, max: 4, step: 0.01 },
      { name: "Edge", def: 0.06, min: 0.005, max: 1, step: 0.005 },
    ],
    // IQ sdBoxFrame (published).
    wgsl: `let pp = abs(p) - vec3f(prm.x);
  let q = abs(pp + vec3f(prm.y)) - vec3f(prm.y);
  let d1 = length(max(vec3f(pp.x, q.y, q.z), vec3f(0.0))) + min(max(pp.x, max(q.y, q.z)), 0.0);
  let d2 = length(max(vec3f(q.x, pp.y, q.z), vec3f(0.0))) + min(max(q.x, max(pp.y, q.z)), 0.0);
  let d3 = length(max(vec3f(q.x, q.y, pp.z), vec3f(0.0))) + min(max(q.x, max(q.y, pp.z)), 0.0);
  return min(d1, min(d2, d3));`,
    glsl: `vec3 pp = abs(p) - vec3(prm.x);
  vec3 q = abs(pp + vec3(prm.y)) - vec3(prm.y);
  float d1 = length(max(vec3(pp.x, q.y, q.z), vec3(0.0))) + min(max(pp.x, max(q.y, q.z)), 0.0);
  float d2 = length(max(vec3(q.x, pp.y, q.z), vec3(0.0))) + min(max(q.x, max(pp.y, q.z)), 0.0);
  float d3 = length(max(vec3(q.x, q.y, pp.z), vec3(0.0))) + min(max(q.x, max(q.y, pp.z)), 0.0);
  return min(d1, min(d2, d3));`,
  },
  // ── D2 Taubin wave — algebraic surfaces d ≈ f/|∇f| (signed, inside < 0) ───
  // One mechanism, one leaf per published implicit (sources in each entry;
  // never the decompiled corpus). Every gradient is analytic, derived by hand
  // from the cited equation; the Taubin quotient is an approximate bound →
  // all deApprox. Common shape: q = p/Scale (+ centering), d = Scale·f/|∇f|,
  // sphere-bounded so the algebraic far-field can't leak.
  {
    id: 28,
    key: "heartSurf",
    name: "Heart",
    glyph: "♥",
    deApprox: true,
    params: [
      { name: "Scale", def: 1, min: 0.1, max: 4, step: 0.01 },
      { name: "Bound", def: 1.6, min: 0.2, max: 6, step: 0.01 },
    ],
    // Taubin heart sextic (MathWorld HeartSurface, formula z = lobe/point axis):
    // (x² + 9/4·y² + z² − 1)³ − x²z³ − 9/80·y²z³ = 0.
    // #353 — the published axes (x=left/right, y=thin/depth, z=lobe/point)
    // ride world axes (x, z, y): thin axis on world Z (closest to the default
    // camera's forward vector), lobe/point axis UP world Y. The first remap
    // (#527) used -p.y here, which put the LOBES at world -Y — a heart
    // rendered upside down, whose inverted silhouette at the default orbit
    // angle is exactly the "featureless blob" reported for four rounds.
    // Verified by headless CPU-tier raymarch renders (front + default orbit):
    // lobes up, cleft resolved, point down — matches the reporter's spirulae
    // reference of the same Taubin sextic.
    wgsl: `let q = vec3f(p.x, p.z, p.y) / prm.x;
  let A = q.x * q.x + 2.25 * q.y * q.y + q.z * q.z - 1.0;
  let z3 = q.z * q.z * q.z;
  let f = A * A * A - q.x * q.x * z3 - 0.1125 * q.y * q.y * z3;
  let gx = 6.0 * q.x * A * A - 2.0 * q.x * z3;
  let gy = 13.5 * q.y * A * A - 0.225 * q.y * z3;
  let gz = 6.0 * q.z * A * A - (3.0 * q.x * q.x + 0.3375 * q.y * q.y) * q.z * q.z;
  let d = prm.x * f / (1e-6 + sqrt(gx * gx + gy * gy + gz * gz));
  return max(d, length(p) - prm.y);`,
    glsl: `vec3 q = vec3(p.x, p.z, p.y) / prm.x;
  float A = q.x * q.x + 2.25 * q.y * q.y + q.z * q.z - 1.0;
  float z3 = q.z * q.z * q.z;
  float f = A * A * A - q.x * q.x * z3 - 0.1125 * q.y * q.y * z3;
  float gx = 6.0 * q.x * A * A - 2.0 * q.x * z3;
  float gy = 13.5 * q.y * A * A - 0.225 * q.y * z3;
  float gz = 6.0 * q.z * A * A - (3.0 * q.x * q.x + 0.3375 * q.y * q.y) * q.z * q.z;
  float d = prm.x * f / (1e-6 + sqrt(gx * gx + gy * gy + gz * gz));
  return max(d, length(p) - prm.y);`,
  },
  {
    id: 29,
    key: "citrus",
    name: "Citrus",
    glyph: "🍋",
    deApprox: true,
    params: [
      { name: "Scale", def: 1.6, min: 0.1, max: 6, step: 0.01 },
      { name: "Bound", def: 1.6, min: 0.2, max: 6, step: 0.01 },
    ],
    // Hauser zitrus (imaginary.org): x² + z² = y³(1−y)³, y ∈ [0,1]
    // (centered here: y+½).
    wgsl: `let q = p / prm.x + vec3f(0.0, 0.5, 0.0);
  let u = q.y * (1.0 - q.y);
  let f = q.x * q.x + q.z * q.z - u * u * u;
  let gy = -3.0 * u * u * (1.0 - 2.0 * q.y);
  let d = prm.x * f / (1e-6 + sqrt(4.0 * q.x * q.x + 4.0 * q.z * q.z + gy * gy));
  return max(d, length(p) - prm.y);`,
    glsl: `vec3 q = p / prm.x + vec3(0.0, 0.5, 0.0);
  float u = q.y * (1.0 - q.y);
  float f = q.x * q.x + q.z * q.z - u * u * u;
  float gy = -3.0 * u * u * (1.0 - 2.0 * q.y);
  float d = prm.x * f / (1e-6 + sqrt(4.0 * q.x * q.x + 4.0 * q.z * q.z + gy * gy));
  return max(d, length(p) - prm.y);`,
  },
  {
    id: 30,
    key: "piriform",
    name: "Piriform",
    glyph: "🍐",
    deApprox: true,
    params: [
      { name: "Scale", def: 1.4, min: 0.1, max: 6, step: 0.01 },
      { name: "Bound", def: 1.6, min: 0.2, max: 6, step: 0.01 },
    ],
    // Piriform quartic of revolution (MathWorld): x⁴ − x³ + y² + z² = 0,
    // x ∈ [0,1] (centered: x+½), pear axis = x.
    wgsl: `let q = p / prm.x + vec3f(0.5, 0.0, 0.0);
  let f = q.x * q.x * q.x * (q.x - 1.0) + q.y * q.y + q.z * q.z;
  let gx = 4.0 * q.x * q.x * q.x - 3.0 * q.x * q.x;
  let d = prm.x * f / (1e-6 + sqrt(gx * gx + 4.0 * q.y * q.y + 4.0 * q.z * q.z));
  return max(d, length(p) - prm.y);`,
    glsl: `vec3 q = p / prm.x + vec3(0.5, 0.0, 0.0);
  float f = q.x * q.x * q.x * (q.x - 1.0) + q.y * q.y + q.z * q.z;
  float gx = 4.0 * q.x * q.x * q.x - 3.0 * q.x * q.x;
  float d = prm.x * f / (1e-6 + sqrt(gx * gx + 4.0 * q.y * q.y + 4.0 * q.z * q.z));
  return max(d, length(p) - prm.y);`,
  },
  {
    id: 31,
    key: "kissSurf",
    name: "Kiss",
    glyph: "💧",
    deApprox: true,
    params: [
      { name: "Scale", def: 1.2, min: 0.1, max: 6, step: 0.01 },
      { name: "Bound", def: 1.8, min: 0.2, max: 6, step: 0.01 },
    ],
    // Kiss quintic of revolution (MathWorld): x² + y² = (1 − z)z⁴, drop
    // axis = z (paired in MB3D with ding-dong — see dingDong).
    wgsl: `let q = p / prm.x;
  let f = q.x * q.x + q.y * q.y - (1.0 - q.z) * q.z * q.z * q.z * q.z;
  let gz = -4.0 * q.z * q.z * q.z + 5.0 * q.z * q.z * q.z * q.z;
  let d = prm.x * f / (1e-6 + sqrt(4.0 * q.x * q.x + 4.0 * q.y * q.y + gz * gz));
  return max(d, length(p) - prm.y);`,
    glsl: `vec3 q = p / prm.x;
  float f = q.x * q.x + q.y * q.y - (1.0 - q.z) * q.z * q.z * q.z * q.z;
  float gz = -4.0 * q.z * q.z * q.z + 5.0 * q.z * q.z * q.z * q.z;
  float d = prm.x * f / (1e-6 + sqrt(4.0 * q.x * q.x + 4.0 * q.y * q.y + gz * gz));
  return max(d, length(p) - prm.y);`,
  },
  {
    id: 32,
    key: "dingDong",
    name: "Ding Dong",
    glyph: "🔔",
    deApprox: true,
    params: [
      { name: "Scale", def: 1.2, min: 0.1, max: 6, step: 0.01 },
      { name: "Bound", def: 1.8, min: 0.2, max: 6, step: 0.01 },
    ],
    // Hauser ding-dong cubic (imaginary.org): x² + y² + z³ − z² = 0.
    wgsl: `let q = p / prm.x;
  let f = q.x * q.x + q.y * q.y + q.z * q.z * q.z - q.z * q.z;
  let gz = 3.0 * q.z * q.z - 2.0 * q.z;
  let d = prm.x * f / (1e-6 + sqrt(4.0 * q.x * q.x + 4.0 * q.y * q.y + gz * gz));
  return max(d, length(p) - prm.y);`,
    glsl: `vec3 q = p / prm.x;
  float f = q.x * q.x + q.y * q.y + q.z * q.z * q.z - q.z * q.z;
  float gz = 3.0 * q.z * q.z - 2.0 * q.z;
  float d = prm.x * f / (1e-6 + sqrt(4.0 * q.x * q.x + 4.0 * q.y * q.y + gz * gz));
  return max(d, length(p) - prm.y);`,
  },
  {
    id: 33,
    key: "devilSurf",
    name: "Devil",
    glyph: "🔱",
    deApprox: true,
    params: [
      { name: "a", def: 0.9, min: 0.1, max: 2, step: 0.01 },
      { name: "b", def: 1, min: 0.1, max: 2, step: 0.01 },
      { name: "Scale", def: 1, min: 0.1, max: 4, step: 0.01 },
      { name: "Bound", def: 1.8, min: 0.2, max: 6, step: 0.01 },
    ],
    // Devil's curve (MathWorld) y²(y²−a²) = x²(x²−b²), revolved x → √(x²+z²).
    wgsl: `let q = p / prm.z;
  let u = q.x * q.x + q.z * q.z;
  let f = q.y * q.y * (q.y * q.y - prm.x * prm.x) - u * (u - prm.y * prm.y);
  let gy = 4.0 * q.y * q.y * q.y - 2.0 * prm.x * prm.x * q.y;
  let gu = -2.0 * u + prm.y * prm.y;
  let gx = 2.0 * q.x * gu;
  let gz = 2.0 * q.z * gu;
  let d = prm.z * f / (1e-6 + sqrt(gx * gx + gy * gy + gz * gz));
  return max(d, length(p) - prm.w);`,
    glsl: `vec3 q = p / prm.z;
  float u = q.x * q.x + q.z * q.z;
  float f = q.y * q.y * (q.y * q.y - prm.x * prm.x) - u * (u - prm.y * prm.y);
  float gy = 4.0 * q.y * q.y * q.y - 2.0 * prm.x * prm.x * q.y;
  float gu = -2.0 * u + prm.y * prm.y;
  float gx = 2.0 * q.x * gu;
  float gz = 2.0 * q.z * gu;
  float d = prm.z * f / (1e-6 + sqrt(gx * gx + gy * gy + gz * gz));
  return max(d, length(p) - prm.w);`,
  },
  {
    id: 34,
    key: "trifoliumSurf",
    name: "Trifolium",
    glyph: "☘",
    deApprox: true,
    params: [
      { name: "Lobe a", def: 1, min: 0.2, max: 3, step: 0.01 },
      { name: "Scale", def: 1, min: 0.1, max: 4, step: 0.01 },
      { name: "Bound", def: 1.6, min: 0.2, max: 6, step: 0.01 },
    ],
    // Trifolium curve (MathWorld) (x²+y²)² = a(x³ − 3xy²), revolved
    // y → ρ = √(y²+z²) (three-lobed rose about the x axis).
    wgsl: `let q = p / prm.y;
  let r2 = q.y * q.y + q.z * q.z;
  let s2 = q.x * q.x + r2;
  let f = s2 * s2 - prm.x * (q.x * q.x * q.x - 3.0 * q.x * r2);
  let gx = 4.0 * q.x * s2 - prm.x * (3.0 * q.x * q.x - 3.0 * r2);
  let gr = 4.0 * s2 + 6.0 * prm.x * q.x;
  let gy = q.y * gr;
  let gz = q.z * gr;
  let d = prm.y * f / (1e-6 + sqrt(gx * gx + gy * gy + gz * gz));
  return max(d, length(p) - prm.z);`,
    glsl: `vec3 q = p / prm.y;
  float r2 = q.y * q.y + q.z * q.z;
  float s2 = q.x * q.x + r2;
  float f = s2 * s2 - prm.x * (q.x * q.x * q.x - 3.0 * q.x * r2);
  float gx = 4.0 * q.x * s2 - prm.x * (3.0 * q.x * q.x - 3.0 * r2);
  float gr = 4.0 * s2 + 6.0 * prm.x * q.x;
  float gy = q.y * gr;
  float gz = q.z * gr;
  float d = prm.y * f / (1e-6 + sqrt(gx * gx + gy * gy + gz * gz));
  return max(d, length(p) - prm.z);`,
  },
  {
    id: 35,
    key: "decoCube",
    name: "Decocube",
    glyph: "🎲",
    deApprox: true,
    params: [
      { name: "Cell c", def: 0.8, min: 0.25, max: 1.3, step: 0.01 },
      { name: "Level", def: 0.02, min: 0.001, max: 0.5, step: 0.001 },
      { name: "Scale", def: 1, min: 0.1, max: 4, step: 0.01 },
      { name: "Bound", def: 2.2, min: 0.2, max: 6, step: 0.01 },
    ],
    // 3D-XplorMath DecoCube (virtualmathmuseum.org ImplicitCompact):
    // ((x²+y²−c²)²+(z²−1)²)·((y²+z²−c²)²+(x²−1)²)·((z²+x²−c²)²+(y²−1)²) = ff.
    wgsl: `let q = p / prm.z;
  let x2 = q.x * q.x; let y2 = q.y * q.y; let z2 = q.z * q.z;
  let c2 = prm.x * prm.x;
  let a1 = x2 + y2 - c2; let b1 = z2 - 1.0;
  let a2 = y2 + z2 - c2; let b2 = x2 - 1.0;
  let a3 = z2 + x2 - c2; let b3 = y2 - 1.0;
  let P = a1 * a1 + b1 * b1; let Q = a2 * a2 + b2 * b2; let R = a3 * a3 + b3 * b3;
  let f = P * Q * R - prm.y;
  let Px = 4.0 * q.x * a1; let Py = 4.0 * q.y * a1; let Pz = 4.0 * q.z * b1;
  let Qx = 4.0 * q.x * b2; let Qy = 4.0 * q.y * a2; let Qz = 4.0 * q.z * a2;
  let Rx = 4.0 * q.x * a3; let Ry = 4.0 * q.y * b3; let Rz = 4.0 * q.z * a3;
  let gx = Px * Q * R + P * Qx * R + P * Q * Rx;
  let gy = Py * Q * R + P * Qy * R + P * Q * Ry;
  let gz = Pz * Q * R + P * Qz * R + P * Q * Rz;
  let d = prm.z * f / (1e-6 + sqrt(gx * gx + gy * gy + gz * gz));
  return max(d, length(p) - prm.w);`,
    glsl: `vec3 q = p / prm.z;
  float x2 = q.x * q.x; float y2 = q.y * q.y; float z2 = q.z * q.z;
  float c2 = prm.x * prm.x;
  float a1 = x2 + y2 - c2; float b1 = z2 - 1.0;
  float a2 = y2 + z2 - c2; float b2 = x2 - 1.0;
  float a3 = z2 + x2 - c2; float b3 = y2 - 1.0;
  float P = a1 * a1 + b1 * b1; float Q = a2 * a2 + b2 * b2; float R = a3 * a3 + b3 * b3;
  float f = P * Q * R - prm.y;
  float Px = 4.0 * q.x * a1; float Py = 4.0 * q.y * a1; float Pz = 4.0 * q.z * b1;
  float Qx = 4.0 * q.x * b2; float Qy = 4.0 * q.y * a2; float Qz = 4.0 * q.z * a2;
  float Rx = 4.0 * q.x * a3; float Ry = 4.0 * q.y * b3; float Rz = 4.0 * q.z * a3;
  float gx = Px * Q * R + P * Qx * R + P * Q * Rx;
  float gy = Py * Q * R + P * Qy * R + P * Q * Ry;
  float gz = Pz * Q * R + P * Qz * R + P * Q * Rz;
  float d = prm.z * f / (1e-6 + sqrt(gx * gx + gy * gy + gz * gz));
  return max(d, length(p) - prm.w);`,
  },
  {
    id: 36,
    key: "cayleyCubic",
    name: "Cayley",
    glyph: "✦",
    deApprox: true,
    params: [
      { name: "Scale", def: 1, min: 0.1, max: 4, step: 0.01 },
      { name: "Bound", def: 1.8, min: 0.2, max: 6, step: 0.01 },
    ],
    // Cayley cubic, tetrahedral-coordinate form (MathWorld CayleyCubic):
    // x² + y² − x²z + y²z + z² − 1 = 0.
    wgsl: `let q = p / prm.x;
  let f = q.x * q.x + q.y * q.y - q.x * q.x * q.z + q.y * q.y * q.z + q.z * q.z - 1.0;
  let gx = 2.0 * q.x * (1.0 - q.z);
  let gy = 2.0 * q.y * (1.0 + q.z);
  let gz = -q.x * q.x + q.y * q.y + 2.0 * q.z;
  let d = prm.x * f / (1e-6 + sqrt(gx * gx + gy * gy + gz * gz));
  return max(d, length(p) - prm.y);`,
    glsl: `vec3 q = p / prm.x;
  float f = q.x * q.x + q.y * q.y - q.x * q.x * q.z + q.y * q.y * q.z + q.z * q.z - 1.0;
  float gx = 2.0 * q.x * (1.0 - q.z);
  float gy = 2.0 * q.y * (1.0 + q.z);
  float gz = -q.x * q.x + q.y * q.y + 2.0 * q.z;
  float d = prm.x * f / (1e-6 + sqrt(gx * gx + gy * gy + gz * gz));
  return max(d, length(p) - prm.y);`,
  },
  {
    id: 37,
    key: "gumdropTorus",
    name: "Gumdrop Torus",
    glyph: "🍬",
    deApprox: true,
    params: [
      { name: "Scale", def: 0.6, min: 0.1, max: 4, step: 0.01 },
      { name: "Bound", def: 1.8, min: 0.2, max: 6, step: 0.01 },
    ],
    // Gumdrop torus (Paul Bourke, geometry/toroidal):
    // 4[x⁴ + (y²+z²)²] + 17x²(y²+z²) − 20(x²+y²+z²) + 17 = 0.
    wgsl: `let q = p / prm.x;
  let u = q.y * q.y + q.z * q.z;
  let x2 = q.x * q.x;
  let f = 4.0 * (x2 * x2 + u * u) + 17.0 * x2 * u - 20.0 * (x2 + u) + 17.0;
  let gx = 16.0 * x2 * q.x + 34.0 * q.x * u - 40.0 * q.x;
  let gc = 16.0 * u + 34.0 * x2 - 40.0;
  let gy = q.y * gc;
  let gz = q.z * gc;
  let d = prm.x * f / (1e-6 + sqrt(gx * gx + gy * gy + gz * gz));
  return max(d, length(p) - prm.y);`,
    glsl: `vec3 q = p / prm.x;
  float u = q.y * q.y + q.z * q.z;
  float x2 = q.x * q.x;
  float f = 4.0 * (x2 * x2 + u * u) + 17.0 * x2 * u - 20.0 * (x2 + u) + 17.0;
  float gx = 16.0 * x2 * q.x + 34.0 * q.x * u - 40.0 * q.x;
  float gc = 16.0 * u + 34.0 * x2 - 40.0;
  float gy = q.y * gc;
  float gz = q.z * gc;
  float d = prm.x * f / (1e-6 + sqrt(gx * gx + gy * gy + gz * gz));
  return max(d, length(p) - prm.y);`,
  },
  {
    id: 38,
    key: "quadricSurf",
    name: "Quadric",
    glyph: "◉",
    deApprox: true,
    params: [
      { name: "a (x²)", def: 1, min: -4, max: 4, step: 0.01 },
      { name: "b (y²)", def: 1, min: -4, max: 4, step: 0.01 },
      { name: "c (z²)", def: -1, min: -4, max: 4, step: 0.01 },
      { name: "Bound", def: 1.6, min: 0.2, max: 6, step: 0.01 },
    ],
    // General central quadric a·x² + b·y² + c·z² = 1 (ellipsoids,
    // hyperboloids, cones near 0) — the classic surface family.
    wgsl: `let f = prm.x * p.x * p.x + prm.y * p.y * p.y + prm.z * p.z * p.z - 1.0;
  let gx = 2.0 * prm.x * p.x;
  let gy = 2.0 * prm.y * p.y;
  let gz = 2.0 * prm.z * p.z;
  let d = f / (1e-6 + sqrt(gx * gx + gy * gy + gz * gz));
  return max(d, length(p) - prm.w);`,
    glsl: `float f = prm.x * p.x * p.x + prm.y * p.y * p.y + prm.z * p.z * p.z - 1.0;
  float gx = 2.0 * prm.x * p.x;
  float gy = 2.0 * prm.y * p.y;
  float gz = 2.0 * prm.z * p.z;
  float d = f / (1e-6 + sqrt(gx * gx + gy * gy + gz * gz));
  return max(d, length(p) - prm.w);`,
  },
  // ── D2 batch 3 — heightfields (all y-up: surface y = h(x,z), solid below;
  // signed gap foreshortened by a per-leaf Lipschitz bound; all deApprox — the
  // render policy tightens deScale). Clean-room sources per leaf comment.
  {
    id: 39,
    key: "gnarlyField",
    unbounded: true,
    name: "Gnarl Field",
    glyph: "〰",
    deApprox: true,
    params: [
      { name: "Step", def: 0.25, min: 0.01, max: 1, step: 0.005 },
      { name: "Warp", def: 3, min: 0, max: 8, step: 0.05 },
      { name: "Height", def: 0.3, min: 0.01, max: 1.5, step: 0.005 },
      { name: "Detail", def: 3, min: 1, max: 8, step: 1 },
    ],
    // Pickover "gnarl" 2D system (published; same recurrence family as the
    // gnarl2D op): iterate the plane point through the coupled sin warp, then
    // read a sine height off the wandered point. Foreshortening: the strict
    // N-step Jacobian power (1+step·(1+warp))^N is so conservative that
    // grazing rays starve for tens of units — use LINEAR growth instead
    // (typical field slope), and let the deApprox policy (halved deScale,
    // bigger budget) absorb the worst-case violations, as other approx
    // leaves do.
    wgsl: `var u = p.x;
  var v = p.z;
  let it = u32(clamp(prm.w, 1.0, 8.0));
  for (var i: u32 = 0u; i < it; i = i + 1u) {
    let t = u;
    u = u - prm.x * sin(v + sin(prm.y * v));
    v = v - prm.x * sin(t + sin(prm.y * t));
  }
  let h = prm.z * 0.5 * (sin(u) + sin(v));
  let lip = 1.0 / (1.0 + prm.z * (1.0 + prm.x * (1.0 + prm.y) * clamp(prm.w, 1.0, 8.0)));
  return abs(p.y - h) * lip;`,
    glsl: `float u = p.x;
  float v = p.z;
  int it = int(clamp(prm.w, 1.0, 8.0));
  for (int i = 0; i < 8; i++) {
    if (i >= it) break;
    float t = u;
    u = u - prm.x * sin(v + sin(prm.y * v));
    v = v - prm.x * sin(t + sin(prm.y * t));
  }
  float h = prm.z * 0.5 * (sin(u) + sin(v));
  float lip = 1.0 / (1.0 + prm.z * (1.0 + prm.x * (1.0 + prm.y) * clamp(prm.w, 1.0, 8.0)));
  return abs(p.y - h) * lip;`,
  },
  {
    id: 40,
    key: "ducksField",
    unbounded: true,
    name: "Ducks Field",
    glyph: "🦆",
    deApprox: true,
    params: [
      { name: "Height", def: 0.4, min: 0.01, max: 2, step: 0.005 },
      { name: "Detail", def: 6, min: 1, max: 24, step: 1 },
      { name: "Seed X", def: -0.6, min: -3, max: 3, step: 0.001 },
      { name: "Seed Y", def: -0.6, min: -3, max: 3, step: 0.001 },
    ],
    // "Ducks" fractal (Samuel Monnier, published 2011): z ← log(ẑ) + c with
    // ẑ = (Re z, |Im z|) and fixed c = seed (the Julia-style form); terrain
    // height = mean log-modulus of the orbit. log ẑ = (log|z|, arg ẑ).
    wgsl: `var wx = p.x;
  var wy = p.z;
  var acc = 0.0;
  let it = u32(clamp(prm.y, 1.0, 24.0));
  for (var i: u32 = 0u; i < it; i = i + 1u) {
    let ay = abs(wy);
    let l = 0.5 * log(max(wx * wx + ay * ay, 1e-6));
    let a = atan2(ay, wx);
    wx = l + prm.z;
    wy = a + prm.w;
    acc = acc + l;
  }
  let h = prm.x * acc / clamp(prm.y, 1.0, 24.0);
  let lip = 1.0 / (1.0 + 2.0 * prm.x);
  return abs(p.y - h) * lip;`,
    glsl: `float wx = p.x;
  float wy = p.z;
  float acc = 0.0;
  int it = int(clamp(prm.y, 1.0, 24.0));
  for (int i = 0; i < 24; i++) {
    if (i >= it) break;
    float ay = abs(wy);
    float l = 0.5 * log(max(wx * wx + ay * ay, 1e-6));
    float a = atan(ay, wx);
    wx = l + prm.z;
    wy = a + prm.w;
    acc = acc + l;
  }
  float h = prm.x * acc / clamp(prm.y, 1.0, 24.0);
  float lip = 1.0 / (1.0 + 2.0 * prm.x);
  return abs(p.y - h) * lip;`,
  },
  {
    id: 41,
    key: "mandelPlate",
    unbounded: true,
    name: "Mandel Plateau",
    glyph: "🏔",
    deApprox: true,
    params: [
      { name: "Detail", def: 24, min: 4, max: 96, step: 1 },
      { name: "Slope", def: 4, min: 0.1, max: 16, step: 0.05 },
      { name: "Depth", def: 0.4, min: 0.02, max: 2, step: 0.005 },
      { name: "Zoom", def: 1, min: 0.1, max: 50, step: 0.005 },
    ],
    // Distance-estimated Mandelbrot terrain (textbook math): interior =
    // plateau at Depth; the skirt falls off with the Milnor exterior
    // distance dM = |z|·ln|z|/|z'| and VANISHES at dM = 1/Slope — so the
    // horizontal guard max(de, dM − 1/Slope) is a valid bound (the solid is
    // empty beyond the skirt) and the contour bands hug the set boundary.
    // Center/pan via the op chain. #353: the solid is floored 10 units below
    // the baseline (was y = -inf — an endless column under the plateau); a
    // TWEAKABLE Height needs a 5th param slot (vec4 budget wall — same as
    // Room's doors/Kleinian's bounds), tracked separately.
    wgsl: `let cre = p.x / prm.w;
  let cim = p.z / prm.w;
  var zr = 0.0; var zi = 0.0;
  var dr = 1.0; var di = 0.0;
  var r2 = 0.0;
  var esc = false;
  let n = u32(clamp(prm.x, 4.0, 96.0));
  for (var i: u32 = 0u; i < n; i = i + 1u) {
    let ndr = 2.0 * (zr * dr - zi * di) + 1.0;
    let ndi = 2.0 * (zr * di + zi * dr);
    dr = ndr; di = ndi;
    let nzr = zr * zr - zi * zi + cre;
    let nzi = 2.0 * zr * zi + cim;
    zr = nzr; zi = nzi;
    r2 = zr * zr + zi * zi;
    if (r2 > 256.0) { esc = true; break; }
  }
  var dM = 0.0;
  if (esc) {
    let az = sqrt(max(r2, 1e-30));
    let adz = sqrt(max(dr * dr + di * di, 1e-30));
    dM = az * log(az) / adz / max(prm.w, 0.1);
  }
  let skirt = 1.0 / max(prm.y, 0.1);
  let h = prm.z * clamp(1.0 - dM / skirt, 0.0, 1.0);
  let de = (p.y - h) / sqrt(1.0 + prm.z * prm.z * prm.y * prm.y);
  return max(max(de, dM - skirt), -10.0 - p.y);`,
    glsl: `float cre = p.x / prm.w;
  float cim = p.z / prm.w;
  float zr = 0.0; float zi = 0.0;
  float dr = 1.0; float di = 0.0;
  float r2 = 0.0;
  bool esc = false;
  int n = int(clamp(prm.x, 4.0, 96.0));
  for (int i = 0; i < 96; i++) {
    if (i >= n) break;
    float ndr = 2.0 * (zr * dr - zi * di) + 1.0;
    float ndi = 2.0 * (zr * di + zi * dr);
    dr = ndr; di = ndi;
    float nzr = zr * zr - zi * zi + cre;
    float nzi = 2.0 * zr * zi + cim;
    zr = nzr; zi = nzi;
    r2 = zr * zr + zi * zi;
    if (r2 > 256.0) { esc = true; break; }
  }
  float dM = 0.0;
  if (esc) {
    float az = sqrt(max(r2, 1e-30));
    float adz = sqrt(max(dr * dr + di * di, 1e-30));
    dM = az * log(az) / adz / max(prm.w, 0.1);
  }
  float skirt = 1.0 / max(prm.y, 0.1);
  float h = prm.z * clamp(1.0 - dM / skirt, 0.0, 1.0);
  float de = (p.y - h) / sqrt(1.0 + prm.z * prm.z * prm.y * prm.y);
  return max(max(de, dM - skirt), -10.0 - p.y);`,
  },
  {
    id: 42,
    key: "checkerField",
    unbounded: true,
    name: "Checker Field",
    glyph: "🏁",
    deApprox: true,
    params: [
      { name: "Bump", def: 0.2, min: 0.01, max: 1, step: 0.005 },
      { name: "Cell", def: 0.5, min: 0.05, max: 4, step: 0.005 },
      { name: "Soft", def: 0.1, min: 0.01, max: 0.5, step: 0.005 },
    ],
    // Parity-of-cells board (trivial math): the smooth checker signal
    // sin(πu)·sin(πv) crosses zero exactly on the cell borders, so a
    // smoothstep on it raises alternating tiles with a Soft-width shoulder.
    wgsl: `let cw = sin(3.14159265 * p.x / prm.y) * sin(3.14159265 * p.z / prm.y);
  let h = prm.x * smoothstep(-prm.z, prm.z, cw);
  let lip = 1.0 / (1.0 + prm.x * 3.2 / (max(prm.z, 0.02) * prm.y));
  return (p.y - h) * lip;`,
    glsl: `float cw = sin(3.14159265 * p.x / prm.y) * sin(3.14159265 * p.z / prm.y);
  float h = prm.x * smoothstep(-prm.z, prm.z, cw);
  float lip = 1.0 / (1.0 + prm.x * 3.2 / (max(prm.z, 0.02) * prm.y));
  return (p.y - h) * lip;`,
  },
  {
    id: 43,
    key: "riemannSqrt",
    name: "Riemann Sheet",
    glyph: "🌀",
    deApprox: true,
    params: [
      { name: "Sheets", def: 1, min: 1, max: 6, step: 1 },
      { name: "Height", def: 0.6, min: 0.05, max: 2, step: 0.005 },
      { name: "Swirl", def: 0, min: -4, max: 4, step: 0.005 },
      { name: "Bound", def: 1.6, min: 0.2, max: 4, step: 0.005 },
    ],
    // Riemann surface of √z (math definition): |Re √(r·e^{iθ})| = √r·|cos θ/2|
    // as a height over the plane, mirrored to the two sheets; Sheets
    // multiplies the branch angle, Swirl adds a radial phase spiral.
    wgsl: `let r = length(p.xz);
  let sr = sqrt(max(r, 1e-4));
  let th = atan2(p.z, p.x);
  let h = prm.y * sr * abs(cos(0.5 * prm.x * th + prm.z * sr));
  let d = min(abs(p.y - h), abs(p.y + h));
  let lip = 1.0 / (1.0 + prm.y * (0.5 * prm.x / max(sr, 0.3) + abs(prm.z) + 0.6));
  return max(d * lip, length(p) - prm.w);`,
    glsl: `float r = length(p.xz);
  float sr = sqrt(max(r, 1e-4));
  float th = atan(p.z, p.x);
  float h = prm.y * sr * abs(cos(0.5 * prm.x * th + prm.z * sr));
  float d = min(abs(p.y - h), abs(p.y + h));
  float lip = 1.0 / (1.0 + prm.y * (0.5 * prm.x / max(sr, 0.3) + abs(prm.z) + 0.6));
  return max(d * lip, length(p) - prm.w);`,
  },
  // ── D2 batch 4 — the geometric tail: polyhedra, prisms, columns, spirals.
  // Clean-room sources per leaf comment (IQ articles, GDF plane folds,
  // textbook curve math); the deferred no-source algebraics stay deferred.
  {
    id: 44,
    key: "octahedron",
    name: "Octahedron",
    glyph: "⯁",
    params: [
      { name: "Size", def: 0.8, min: 0.05, max: 4, step: 0.01 },
      { name: "Round", def: 0, min: 0, max: 0.5, step: 0.005 },
    ],
    // IQ sdOctahedron (published exact SDF): plane distance in the face
    // region, edge distance elsewhere via the fold-to-octant closest point.
    wgsl: `let q = abs(p);
  let m = q.x + q.y + q.z - prm.x;
  var o = vec3f(0.0);
  if (3.0 * q.x < m) { o = q.xyz; }
  else if (3.0 * q.y < m) { o = q.yzx; }
  else if (3.0 * q.z < m) { o = q.zxy; }
  else { return m * 0.57735027 - prm.y; }
  let k = clamp(0.5 * (o.z - o.y + prm.x), 0.0, prm.x);
  return length(vec3f(o.x, o.y - prm.x + k, o.z - k)) - prm.y;`,
    glsl: `vec3 q = abs(p);
  float m = q.x + q.y + q.z - prm.x;
  vec3 o;
  if (3.0 * q.x < m) o = q.xyz;
  else if (3.0 * q.y < m) o = q.yzx;
  else if (3.0 * q.z < m) o = q.zxy;
  else return m * 0.57735027 - prm.y;
  float k = clamp(0.5 * (o.z - o.y + prm.x), 0.0, prm.x);
  return length(vec3(o.x, o.y - prm.x + k, o.z - k)) - prm.y;`,
  },
  {
    id: 45,
    key: "dodecahedron",
    name: "Dodecahedron",
    glyph: "⬟",
    params: [
      { name: "Size", def: 0.8, min: 0.05, max: 4, step: 0.01 },
      { name: "Round", def: 0, min: 0, max: 0.5, step: 0.005 },
      { name: "Icosa", def: 0, min: 0, max: 1, step: 1 },
    ],
    // GDF plane-fold polyhedron (published construction): max over
    // dot(|p|, n) - size, faces normal to the icosahedron vertex directions
    // (0, 1, φ)/|·| cyclic; Icosa flips to the dual set (1,1,1)/√3 +
    // (0, 1/φ, φ)/√3 cyclic.
    // #353: `d - size - round` (max-of-planes minus TWO scalars, both just
    // subtracted) was mathematically identical to `d - (size+round)` — Round
    // was 100% redundant with Size, not an edge round at all. Real rounding
    // needs the max() *itself* softened only near where two face-planes tie
    // (an edge/vertex); away from an edge one term dominates and must stay
    // exact (flat faces don't move, matching Round Box's own contract, and
    // Round=0 stays bit-identical to every existing preset). Chained
    // smooth-max (poly quadratic, the same form as shader.js sminP/smaxP —
    // inlined here since a leaf body is spliced into 3 separate contexts
    // that don't share that helper: WGSL live render, WebGL2 live render,
    // and the standalone/Shadertoy GLSL exporter).
    wgsl: `let q = abs(p);
  let k = prm.y;
  var d = 0.0;
  if (prm.z > 0.5) {
    d = dot(q, vec3f(0.57735027, 0.57735027, 0.57735027));
    let d1 = dot(q, vec3f(0.0, 0.35682209, 0.93417236));
    let d2 = dot(q, vec3f(0.35682209, 0.93417236, 0.0));
    let d3 = dot(q, vec3f(0.93417236, 0.0, 0.35682209));
    if (k > 0.0) {
      var h = clamp(0.5 + 0.5 * (d - d1) / k, 0.0, 1.0);
      d = mix(d1, d, h) + k * h * (1.0 - h);
      h = clamp(0.5 + 0.5 * (d - d2) / k, 0.0, 1.0);
      d = mix(d2, d, h) + k * h * (1.0 - h);
      h = clamp(0.5 + 0.5 * (d - d3) / k, 0.0, 1.0);
      d = mix(d3, d, h) + k * h * (1.0 - h);
    } else {
      d = max(d, max(d1, max(d2, d3)));
    }
  } else {
    d = dot(q, vec3f(0.0, 0.52573111, 0.85065081));
    let e1 = dot(q, vec3f(0.52573111, 0.85065081, 0.0));
    let e2 = dot(q, vec3f(0.85065081, 0.0, 0.52573111));
    if (k > 0.0) {
      var h = clamp(0.5 + 0.5 * (d - e1) / k, 0.0, 1.0);
      d = mix(e1, d, h) + k * h * (1.0 - h);
      h = clamp(0.5 + 0.5 * (d - e2) / k, 0.0, 1.0);
      d = mix(e2, d, h) + k * h * (1.0 - h);
    } else {
      d = max(d, max(e1, e2));
    }
  }
  return d - prm.x;`,
    glsl: `vec3 q = abs(p);
  float k = prm.y;
  float d;
  if (prm.z > 0.5) {
    d = dot(q, vec3(0.57735027));
    float d1 = dot(q, vec3(0.0, 0.35682209, 0.93417236));
    float d2 = dot(q, vec3(0.35682209, 0.93417236, 0.0));
    float d3 = dot(q, vec3(0.93417236, 0.0, 0.35682209));
    if (k > 0.0) {
      float h = clamp(0.5 + 0.5 * (d - d1) / k, 0.0, 1.0);
      d = mix(d1, d, h) + k * h * (1.0 - h);
      h = clamp(0.5 + 0.5 * (d - d2) / k, 0.0, 1.0);
      d = mix(d2, d, h) + k * h * (1.0 - h);
      h = clamp(0.5 + 0.5 * (d - d3) / k, 0.0, 1.0);
      d = mix(d3, d, h) + k * h * (1.0 - h);
    } else {
      d = max(d, max(d1, max(d2, d3)));
    }
  } else {
    d = dot(q, vec3(0.0, 0.52573111, 0.85065081));
    float e1 = dot(q, vec3(0.52573111, 0.85065081, 0.0));
    float e2 = dot(q, vec3(0.85065081, 0.0, 0.52573111));
    if (k > 0.0) {
      float h = clamp(0.5 + 0.5 * (d - e1) / k, 0.0, 1.0);
      d = mix(e1, d, h) + k * h * (1.0 - h);
      h = clamp(0.5 + 0.5 * (d - e2) / k, 0.0, 1.0);
      d = mix(e2, d, h) + k * h * (1.0 - h);
    } else {
      d = max(d, max(e1, e2));
    }
  }
  return d - prm.x;`,
  },
  {
    id: 46,
    key: "nPrism",
    name: "N-Prism",
    glyph: "⬢",
    params: [
      { name: "Sides", def: 6, min: 3, max: 16, step: 1 },
      { name: "Radius", def: 0.8, min: 0.05, max: 4, step: 0.01 },
      { name: "Height", def: 0.6, min: 0.05, max: 4, step: 0.01 },
      { name: "Round", def: 0, min: 0, max: 0.3, step: 0.005 },
    ],
    // IQ hex/tri prism generalized (derivable): polar sector fold, flat-face
    // apothem distance, slab cap. Bound (under near corners).
    // #353: `max(d2, cap) - round` was mathematically the exact same as
    // widening Radius/Height by round — Round just resized the prism (the
    // "changing radius" report), it never actually chamfered anything.
    // Blending the two terms with a smooth-max (only near the cap↔side rim,
    // where they tie) gives Round a real, distinct job: bevel that rim.
    // Round=0 keeps the exact old hard max (bit-identical to every existing
    // preset); the vertical edges between the N flat sides are a separate,
    // more involved fold-math change not attempted here.
    wgsl: `let n = clamp(prm.x, 3.0, 16.0);
  let sector = 6.2831853 / n;
  let am = (fract(atan2(p.z, p.x) / sector + 0.5) - 0.5) * sector;
  let d2 = length(p.xz) * cos(am) - prm.y;
  let cap = abs(p.y) - prm.z;
  let k = prm.w;
  if (k > 0.0) {
    let hh = clamp(0.5 + 0.5 * (d2 - cap) / k, 0.0, 1.0);
    return mix(cap, d2, hh) + k * hh * (1.0 - hh);
  }
  return max(d2, cap);`,
    glsl: `float n = clamp(prm.x, 3.0, 16.0);
  float sector = 6.2831853 / n;
  float am = (fract(atan(p.z, p.x) / sector + 0.5) - 0.5) * sector;
  float d2 = length(p.xz) * cos(am) - prm.y;
  float cap = abs(p.y) - prm.z;
  float k = prm.w;
  if (k > 0.0) {
    float hh = clamp(0.5 + 0.5 * (d2 - cap) / k, 0.0, 1.0);
    return mix(cap, d2, hh) + k * hh * (1.0 - hh);
  }
  return max(d2, cap);`,
  },
  {
    id: 47,
    key: "pyramid",
    name: "Pyramid",
    glyph: "🔺",
    params: [
      { name: "Sides", def: 4, min: 3, max: 16, step: 1 },
      { name: "Base", def: 1, min: 0.05, max: 4, step: 0.01 },
      { name: "Height", def: 1, min: 0.05, max: 4, step: 0.01 },
      { name: "Round", def: 0, min: 0, max: 0.3, step: 0.005 },
    ],
    // n-gon pyramid (derivable, IQ sdPyramid family): polar fold, then the
    // slanted side plane through the base edge (inradius r at y=-h/2) and
    // the apex (0, +h/2), capped by the base plane.
    // #353: `max(side, -yb) - round` was the same redundant-with-size trick
    // as N-Prism/Dodecahedron ("Round value is just changing radius").
    // Smooth-max the side/base pair instead — real bevel at the base rim
    // only, Round=0 bit-identical to the old hard max (note: the local `hh`
    // blend var is intentionally not `h` — that name is already the
    // pyramid's own height local below).
    wgsl: `let n = clamp(prm.x, 3.0, 16.0);
  let sector = 6.2831853 / n;
  let am = (fract(atan2(p.z, p.x) / sector + 0.5) - 0.5) * sector;
  let l = length(p.xz) * cos(am);
  let r = max(prm.y, 0.05);
  let h = max(prm.z, 0.05);
  let yb = p.y + 0.5 * h;
  let side = (l * h + yb * r - r * h) / sqrt(h * h + r * r);
  let base = -yb;
  let k = prm.w;
  if (k > 0.0) {
    let hh = clamp(0.5 + 0.5 * (side - base) / k, 0.0, 1.0);
    return mix(base, side, hh) + k * hh * (1.0 - hh);
  }
  return max(side, base);`,
    glsl: `float n = clamp(prm.x, 3.0, 16.0);
  float sector = 6.2831853 / n;
  float am = (fract(atan(p.z, p.x) / sector + 0.5) - 0.5) * sector;
  float l = length(p.xz) * cos(am);
  float r = max(prm.y, 0.05);
  float h = max(prm.z, 0.05);
  float yb = p.y + 0.5 * h;
  float side = (l * h + yb * r - r * h) / sqrt(h * h + r * r);
  float base = -yb;
  float k = prm.w;
  if (k > 0.0) {
    float hh = clamp(0.5 + 0.5 * (side - base) / k, 0.0, 1.0);
    return mix(base, side, hh) + k * hh * (1.0 - hh);
  }
  return max(side, base);`,
  },
  {
    id: 48,
    key: "greekCross",
    name: "Greek Cross",
    glyph: "✚",
    params: [
      { name: "Length", def: 1, min: 0.05, max: 4, step: 0.01 },
      { name: "Arm", def: 0.25, min: 0.02, max: 2, step: 0.005 },
      { name: "Round", def: 0, min: 0, max: 0.3, step: 0.005 },
    ],
    // Union of three orthogonal boxes (IQ sdBox, exact each): the 3D plus /
    // jack shape from the corpus GreekCrossIFS.
    wgsl: `let q = abs(p);
  let a1 = q - vec3f(prm.x, prm.y, prm.y);
  let d1 = length(max(a1, vec3f(0.0))) + min(max(a1.x, max(a1.y, a1.z)), 0.0);
  let a2 = q - vec3f(prm.y, prm.x, prm.y);
  let d2 = length(max(a2, vec3f(0.0))) + min(max(a2.x, max(a2.y, a2.z)), 0.0);
  let a3 = q - vec3f(prm.y, prm.y, prm.x);
  let d3 = length(max(a3, vec3f(0.0))) + min(max(a3.x, max(a3.y, a3.z)), 0.0);
  return min(d1, min(d2, d3)) - prm.z;`,
    glsl: `vec3 q = abs(p);
  vec3 a1 = q - vec3(prm.x, prm.y, prm.y);
  float d1 = length(max(a1, vec3(0.0))) + min(max(a1.x, max(a1.y, a1.z)), 0.0);
  vec3 a2 = q - vec3(prm.y, prm.x, prm.y);
  float d2 = length(max(a2, vec3(0.0))) + min(max(a2.x, max(a2.y, a2.z)), 0.0);
  vec3 a3 = q - vec3(prm.y, prm.y, prm.x);
  float d3 = length(max(a3, vec3(0.0))) + min(max(a3.x, max(a3.y, a3.z)), 0.0);
  return min(d1, min(d2, d3)) - prm.z;`,
  },
  {
    id: 49,
    key: "borg",
    name: "Borg Shell",
    glyph: "🧊",
    params: [
      { name: "Size", def: 0.9, min: 0.1, max: 4, step: 0.01 },
      { name: "Density", def: 6, min: 0.5, max: 24, step: 0.05 },
      { name: "Amp", def: 0.12, min: 0, max: 1, step: 0.005 },
      { name: "Thick", def: 0.06, min: 0.01, max: 0.5, step: 0.005 },
    ],
    // Box shell |sdBox| - t displaced by the separable sin product (trivial
    // math): where the wave exceeds the thickness the shell opens into a
    // porous cube. Valid bound via the displacement Lipschitz divisor
    // (|∇ sin·sin·sin| ≤ √3·density).
    wgsl: `let b = abs(p) - vec3f(prm.x, prm.x, prm.x);
  let box = length(max(b, vec3f(0.0))) + min(max(b.x, max(b.y, b.z)), 0.0);
  let disp = prm.z * sin(prm.y * p.x) * sin(prm.y * p.y) * sin(prm.y * p.z);
  return (abs(box) - prm.w + disp) / (1.0 + prm.z * prm.y * 1.8);`,
    glsl: `vec3 b = abs(p) - vec3(prm.x);
  float box = length(max(b, vec3(0.0))) + min(max(b.x, max(b.y, b.z)), 0.0);
  float disp = prm.z * sin(prm.y * p.x) * sin(prm.y * p.y) * sin(prm.y * p.z);
  return (abs(box) - prm.w + disp) / (1.0 + prm.z * prm.y * 1.8);`,
  },
  {
    id: 50,
    key: "tower",
    name: "Column",
    glyph: "🏛",
    unbounded: true,
    deApprox: true,
    params: [
      { name: "Radius", def: 0.5, min: 0.05, max: 2, step: 0.005 },
      { name: "Wave", def: 0.15, min: 0, max: 1, step: 0.005 },
      { name: "Freq", def: 3, min: 0.1, max: 12, step: 0.05 },
      { name: "Flutes", def: 8, min: 0, max: 24, step: 1 },
    ],
    // Sin-modulated column along y (derivable): radius breathes with height,
    // Flutes adds an angular cosine ripple. Gradient-bound divisor; the
    // angular term steepens near the axis → approximate there.
    wgsl: `let l = length(p.xz);
  let ang = atan2(p.z, p.x);
  let fl = step(0.5, prm.w) * 0.06 * prm.x * cos(prm.w * ang);
  let rr = prm.x * (1.0 + prm.y * sin(prm.z * p.y)) + fl;
  let gy = prm.x * prm.y * prm.z;
  let ga = 0.06 * prm.w;
  return (l - rr) / sqrt(1.0 + gy * gy + ga * ga);`,
    glsl: `float l = length(p.xz);
  float ang = atan(p.z, p.x);
  float fl = step(0.5, prm.w) * 0.06 * prm.x * cos(prm.w * ang);
  float rr = prm.x * (1.0 + prm.y * sin(prm.z * p.y)) + fl;
  float gy = prm.x * prm.y * prm.z;
  float ga = 0.06 * prm.w;
  return (l - rr) / sqrt(1.0 + gy * gy + ga * ga);`,
  },
  {
    id: 51,
    key: "gem",
    name: "Gem",
    glyph: "💎",
    params: [
      { name: "Facets", def: 8, min: 4, max: 24, step: 1 },
      { name: "Radius", def: 1, min: 0.05, max: 4, step: 0.01 },
      { name: "Crown", def: 0.35, min: 0.02, max: 2, step: 0.005 },
      { name: "Pavilion", def: 0.9, min: 0.05, max: 4, step: 0.005 },
    ],
    // Brilliant cut (derivable): polar facet fold, then the intersection of
    // the pavilion cone plane (girdle → culet), the crown plane (girdle →
    // table rim at half radius), and the flat table. Bound (under at edges).
    wgsl: `let n = clamp(prm.x, 4.0, 24.0);
  let sector = 6.2831853 / n;
  let am = (fract(atan2(p.z, p.x) / sector + 0.5) - 0.5) * sector;
  let l = length(p.xz) * cos(am);
  let R = max(prm.y, 0.05);
  let hc = max(prm.z, 0.02);
  let hp = max(prm.w, 0.05);
  let pav = (l * hp - p.y * R - R * hp) / sqrt(hp * hp + R * R);
  let cro = (l * hc + p.y * 0.5 * R - R * hc) / sqrt(hc * hc + 0.25 * R * R);
  return max(max(pav, cro), p.y - hc);`,
    glsl: `float n = clamp(prm.x, 4.0, 24.0);
  float sector = 6.2831853 / n;
  float am = (fract(atan(p.z, p.x) / sector + 0.5) - 0.5) * sector;
  float l = length(p.xz) * cos(am);
  float R = max(prm.y, 0.05);
  float hc = max(prm.z, 0.02);
  float hp = max(prm.w, 0.05);
  float pav = (l * hp - p.y * R - R * hp) / sqrt(hp * hp + R * R);
  float cro = (l * hc + p.y * 0.5 * R - R * hc) / sqrt(hc * hc + 0.25 * R * R);
  return max(max(pav, cro), p.y - hc);`,
  },
  {
    id: 52,
    key: "loxodrome",
    name: "Loxodrome",
    glyph: "🧭",
    deApprox: true,
    params: [
      { name: "Winding", def: 4, min: 0.5, max: 12, step: 0.05 },
      { name: "Radius", def: 1, min: 0.1, max: 3, step: 0.01 },
      { name: "Strands", def: 1, min: 1, max: 8, step: 1 },
      { name: "Thick", def: 0.08, min: 0.01, max: 0.5, step: 0.005 },
    ],
    // Rhumb line (textbook math): in Mercator coordinates (isometric
    // latitude ψ = artanh sin λ) the loxodrome is the straight line
    // φ = k·ψ; tube distance = radial gap ⊕ the wrapped chart deviation
    // scaled back by the conformal factor r·cos λ. Approximate (conformal
    // distortion), safety ×0.8.
    wgsl: `let r = length(p);
  let ir = max(r, 1e-4);
  let sl = clamp(p.y / ir, -0.9999, 0.9999);
  let psi = 0.5 * log((1.0 + sl) / (1.0 - sl));
  let sector = 6.2831853 / max(prm.z, 1.0);
  let u = atan2(p.z, p.x) - prm.x * psi;
  let w = (fract(u / sector + 0.5) - 0.5) * sector;
  let cl = sqrt(max(1.0 - sl * sl, 1e-4));
  let arc = w * ir * cl / sqrt(1.0 + prm.x * prm.x);
  return (length(vec2f(r - prm.y, arc)) - prm.w) * 0.8;`,
    glsl: `float r = length(p);
  float ir = max(r, 1e-4);
  float sl = clamp(p.y / ir, -0.9999, 0.9999);
  float psi = 0.5 * log((1.0 + sl) / (1.0 - sl));
  float sector = 6.2831853 / max(prm.z, 1.0);
  float u = atan(p.z, p.x) - prm.x * psi;
  float w = (fract(u / sector + 0.5) - 0.5) * sector;
  float cl = sqrt(max(1.0 - sl * sl, 1e-4));
  float arc = w * ir * cl / sqrt(1.0 + prm.x * prm.x);
  return (length(vec2(r - prm.y, arc)) - prm.w) * 0.8;`,
  },
  {
    id: 53,
    key: "logSpiral",
    name: "Spiral Walls",
    glyph: "🍥",
    unbounded: true,
    deApprox: true,
    params: [
      { name: "Tightness", def: 0.25, min: 0.02, max: 1, step: 0.005 },
      { name: "Arms", def: 2, min: 1, max: 8, step: 1 },
      { name: "Height", def: 0.8, min: 0.05, max: 4, step: 0.01 },
      { name: "Thick", def: 0.05, min: 0.01, max: 0.5, step: 0.005 },
    ],
    // Logarithmic spiral r = e^{bθ} (textbook math): in log-polar
    // coordinates the arms are the level sets of ln(r)/b − θ; wall distance
    // = wrapped level gap ÷ |∇| = gap·r·b/√(1+b²), extruded in y.
    // Approximate near the converging center, safety ×0.9.
    wgsl: `let r = max(length(p.xz), 1e-4);
  let b = max(prm.x, 0.02);
  let sector = 6.2831853 / max(prm.y, 1.0);
  let u = log(r) / b - atan2(p.z, p.x);
  let w = (fract(u / sector + 0.5) - 0.5) * sector;
  let d2 = abs(w) * r * b / sqrt(1.0 + b * b) - prm.w;
  return max(d2, abs(p.y) - prm.z) * 0.9;`,
    glsl: `float r = max(length(p.xz), 1e-4);
  float b = max(prm.x, 0.02);
  float sector = 6.2831853 / max(prm.y, 1.0);
  float u = log(r) / b - atan(p.z, p.x);
  float w = (fract(u / sector + 0.5) - 0.5) * sector;
  float d2 = abs(w) * r * b / sqrt(1.0 + b * b) - prm.w;
  return max(d2, abs(p.y) - prm.z) * 0.9;`,
  },
  {
    id: 54,
    key: "pseudoSphere",
    name: "Pseudosphere",
    glyph: "🎺",
    deApprox: true,
    params: [
      { name: "Radius", def: 0.8, min: 0.1, max: 3, step: 0.01 },
      { name: "Length", def: 3, min: 0.5, max: 6, step: 0.01 },
      { name: "Thick", def: 0.08, min: 0.01, max: 0.5, step: 0.005 },
    ],
    // Tractricoid (textbook math): the tractrix (sech u, u − tanh u) revolved
    // around y, mirrored to both horns. No closed-form distance → min over
    // 24 curve samples in the (radial, |y|) half-plane; sampled min
    // overestimates between samples, so safety ×0.7 + approx policy.
    wgsl: `let l = length(p.xz);
  let ay = abs(p.y);
  var best = 1e9;
  for (var i: u32 = 0u; i < 24u; i = i + 1u) {
    let u = prm.y * (f32(i) / 23.0);
    let e = exp(u);
    let sh = 2.0 / (e + 1.0 / e);
    let tn = (e - 1.0 / e) / (e + 1.0 / e);
    let cx = prm.x * sh;
    let cy = prm.x * (u - tn);
    best = min(best, length(vec2f(l - cx, ay - cy)));
  }
  return (best - prm.z) * 0.7;`,
    glsl: `float l = length(p.xz);
  float ay = abs(p.y);
  float best = 1e9;
  for (int i = 0; i < 24; i++) {
    float u = prm.y * (float(i) / 23.0);
    float e = exp(u);
    float sh = 2.0 / (e + 1.0 / e);
    float tn = (e - 1.0 / e) / (e + 1.0 / e);
    float cx = prm.x * sh;
    float cy = prm.x * (u - tn);
    best = min(best, length(vec2(l - cx, ay - cy)));
  }
  return (best - prm.z) * 0.7;`,
  },
  // ── D2 batch 5 — the last leaves this side of kleinian-limit.
  {
    id: 55,
    key: "randomCells",
    name: "Random Cells",
    glyph: "⚄",
    unbounded: true,
    params: [
      { name: "Cell", def: 0.6, min: 0.1, max: 2, step: 0.005 },
      { name: "Fill", def: 0.7, min: 0, max: 1, step: 0.005 },
      { name: "Seed", def: 1, min: 0, max: 1023, step: 1 },
      { name: "Slab", def: 1, min: 1, max: 9, step: 1 },
    ],
    // Hash-grid cell dispatch (IQ's grid technique, integer Wang-style hash
    // so all three tiers see the SAME layout — float sin-hashes diverge
    // between f32 GPU and f64 CPU). Per occupied cell one of four primitives
    // (sphere / cube / octahedron / cylinder) with hashed size; scanning the
    // 3×3×3 neighborhood + the half-cell clamp keeps the DE a true bound
    // (shapes never exceed 0.38·cell, so anything outside the block is
    // ≥ cell/2 away).
    wgsl: `let c = max(prm.x, 0.05);
  let seed = u32(clamp(prm.z, 0.0, 1023.0));
  let slab = i32(clamp(prm.w, 1.0, 9.0));
  let base = vec3i(floor(p / c));
  var d = 0.5 * c;
  for (var oi: i32 = -1; oi <= 1; oi = oi + 1) {
  for (var oj: i32 = -1; oj <= 1; oj = oj + 1) {
  for (var ok: i32 = -1; ok <= 1; ok = ok + 1) {
    let id = base + vec3i(oi, oj, ok);
    if (abs(id.y) >= slab) { continue; }
    var h = (bitcast<u32>(id.x) * 73856093u) ^ (bitcast<u32>(id.y) * 19349663u) ^ (bitcast<u32>(id.z) * 83492791u) ^ (seed * 2654435761u);
    h = h ^ (h >> 13u); h = h * 1274126177u; h = h ^ (h >> 16u);
    if (f32(h & 1023u) >= prm.y * 1024.0) { continue; }
    let r = c * (0.16 + 0.22 * f32((h >> 10u) & 255u) / 255.0);
    let l = p - (vec3f(id) + vec3f(0.5, 0.5, 0.5)) * c;
    let shape = (h >> 18u) & 3u;
    var sd = 0.0;
    if (shape == 0u) { sd = length(l) - r; }
    else if (shape == 1u) {
      let b = abs(l) - vec3f(r, r, r);
      sd = length(max(b, vec3f(0.0))) + min(max(b.x, max(b.y, b.z)), 0.0);
    }
    else if (shape == 2u) { sd = (abs(l.x) + abs(l.y) + abs(l.z) - 1.5 * r) * 0.57735027; }
    else { sd = max(length(l.xz) - r, abs(l.y) - r); }
    d = min(d, sd);
  }}}
  return d;`,
    glsl: `float c = max(prm.x, 0.05);
  uint seed = uint(clamp(prm.z, 0.0, 1023.0));
  int slab = int(clamp(prm.w, 1.0, 9.0));
  ivec3 base = ivec3(floor(p / c));
  float d = 0.5 * c;
  for (int oi = -1; oi <= 1; oi++) {
  for (int oj = -1; oj <= 1; oj++) {
  for (int ok = -1; ok <= 1; ok++) {
    ivec3 id = base + ivec3(oi, oj, ok);
    if (abs(id.y) >= slab) continue;
    uint h = uint(id.x) * 73856093u ^ uint(id.y) * 19349663u ^ uint(id.z) * 83492791u ^ (seed * 2654435761u);
    h = h ^ (h >> 13u); h = h * 1274126177u; h = h ^ (h >> 16u);
    if (float(h & 1023u) >= prm.y * 1024.0) continue;
    float r = c * (0.16 + 0.22 * float((h >> 10u) & 255u) / 255.0);
    vec3 l = p - (vec3(id) + vec3(0.5)) * c;
    uint shape = (h >> 18u) & 3u;
    float sd;
    if (shape == 0u) { sd = length(l) - r; }
    else if (shape == 1u) {
      vec3 b = abs(l) - vec3(r);
      sd = length(max(b, vec3(0.0))) + min(max(b.x, max(b.y, b.z)), 0.0);
    }
    else if (shape == 2u) { sd = (abs(l.x) + abs(l.y) + abs(l.z) - 1.5 * r) * 0.57735027; }
    else { sd = max(length(l.xz) - r, abs(l.y) - r); }
    d = min(d, sd);
  }}}
  return d;`,
  },
  {
    id: 56,
    key: "umbrella",
    name: "Umbrella",
    glyph: "☂",
    deApprox: true,
    params: [
      { name: "Size", def: 1, min: 0.1, max: 4, step: 0.01 },
      { name: "Thick", def: 0, min: 0, max: 0.5, step: 0.005 },
      { name: "Bound", def: 1.5, min: 0.2, max: 4, step: 0.01 },
    ],
    // Whitney umbrella x² = z²·y (published cubic, handle up the y axis):
    // Taubin quotient with the analytic gradient (2x, −z², −2zy); Thick > 0
    // shells the surface, 0 keeps the signed solid; sphere bound.
    wgsl: `let q = p / prm.x;
  let f = q.x * q.x - q.z * q.z * q.y;
  let gx = 2.0 * q.x;
  let gy = -q.z * q.z;
  let gz = -2.0 * q.z * q.y;
  var d = prm.x * f / (1e-6 + sqrt(gx * gx + gy * gy + gz * gz));
  if (prm.y > 0.0) { d = abs(d) - prm.y; }
  return max(d, length(p) - prm.z);`,
    glsl: `vec3 q = p / prm.x;
  float f = q.x * q.x - q.z * q.z * q.y;
  float gx = 2.0 * q.x;
  float gy = -q.z * q.z;
  float gz = -2.0 * q.z * q.y;
  float d = prm.x * f / (1e-6 + sqrt(gx * gx + gy * gy + gz * gz));
  if (prm.y > 0.0) d = abs(d) - prm.y;
  return max(d, length(p) - prm.z);`,
  },
  {
    id: 57,
    key: "kleinBottle",
    name: "Klein Bottle",
    glyph: "🍾",
    deApprox: true,
    params: [
      { name: "Size", def: 0.4, min: 0.05, max: 2, step: 0.005 },
      { name: "Thick", def: 0, min: 0, max: 0.5, step: 0.005 },
      { name: "Bound", def: 1.4, min: 0.2, max: 4, step: 0.01 },
    ],
    // The classic Klein-bottle sextic implicit (published, e.g. MathWorld):
    // f = A·C + 16xzB with S = |q|², A = S+2y−1, B = S−2y−1, C = B²−8z²;
    // Taubin quotient with the hand-derived gradient. Thick shells; the
    // origin sits inside (f(0) = −1). The KleinBotTIFS decompile is broken —
    // this is built from the math directly.
    wgsl: `let q = p / prm.x;
  let S = dot(q, q);
  let A = S + 2.0 * q.y - 1.0;
  let B = S - 2.0 * q.y - 1.0;
  let C = B * B - 8.0 * q.z * q.z;
  let f = A * C + 16.0 * q.x * q.z * B;
  let gx = 2.0 * q.x * C + 4.0 * A * B * q.x + 16.0 * q.z * B + 32.0 * q.x * q.x * q.z;
  let gy = (2.0 * q.y + 2.0) * C + 4.0 * A * B * (q.y - 1.0) + 32.0 * q.x * q.z * (q.y - 1.0);
  let gz = 2.0 * q.z * C + A * (4.0 * B * q.z - 16.0 * q.z) + 16.0 * q.x * B + 32.0 * q.x * q.z * q.z;
  var d = prm.x * f / (1e-6 + sqrt(gx * gx + gy * gy + gz * gz));
  if (prm.y > 0.0) { d = abs(d) - prm.y; }
  return max(d, length(p) - prm.z);`,
    glsl: `vec3 q = p / prm.x;
  float S = dot(q, q);
  float A = S + 2.0 * q.y - 1.0;
  float B = S - 2.0 * q.y - 1.0;
  float C = B * B - 8.0 * q.z * q.z;
  float f = A * C + 16.0 * q.x * q.z * B;
  float gx = 2.0 * q.x * C + 4.0 * A * B * q.x + 16.0 * q.z * B + 32.0 * q.x * q.x * q.z;
  float gy = (2.0 * q.y + 2.0) * C + 4.0 * A * B * (q.y - 1.0) + 32.0 * q.x * q.z * (q.y - 1.0);
  float gz = 2.0 * q.z * C + A * (4.0 * B * q.z - 16.0 * q.z) + 16.0 * q.x * B + 32.0 * q.x * q.z * q.z;
  float d = prm.x * f / (1e-6 + sqrt(gx * gx + gy * gy + gz * gz));
  if (prm.y > 0.0) d = abs(d) - prm.y;
  return max(d, length(p) - prm.z);`,
  },
  {
    id: 58,
    key: "kleinianLimit",
    name: "Kleinian",
    glyph: "🫧",
    unbounded: true,
    deApprox: true,
    params: [
      { name: "KleinR", def: 1.95, min: 1.4, max: 2.2, step: 0.005 },
      { name: "KleinI", def: 0.07, min: -0.5, max: 0.5, step: 0.005 },
      { name: "Bend", def: 0.12, min: 0, max: 0.8, step: 0.005 },
      { name: "Detail", def: 48, min: 4, max: 128, step: 1 },
    ],
    // Maskit-slice Kleinian limit set, built from Jos Leys' PUBLISHED 2017
    // algorithm + the textbook Poincaré extension and conformal-pullback DE
    // — derivation and provenance in docs/planning/KLEINIAN_LIMIT.md (the
    // GPL Fragmentarium shader was NOT used). Generators a(z) = it + 1/z,
    // b(z) = z + 2 with t = KleinR + i·KleinI; wrap x into the sheared
    // strip, apply a below the exponential separation line / A above,
    // escape when y leaves [0, u]; DE = |min(y, u−y)|/DF − 0.025 — a thin
    // SHELL around the limit-set surface rather than the signed solid: the
    // solid form is two infinite slabs, and an orbit camera that dips into
    // one sees a featureless interior (user-reported); the shell looks
    // identical from the corridor and stays visible from every side.
    wgsl: `let u = clamp(prm.x, 1.4, 2.2);
  let v = clamp(prm.y, -0.5, 0.5);
  let bend = clamp(prm.z, 0.0, 0.8);
  let it = u32(clamp(prm.w, 4.0, 128.0));
  var q = vec3f(p.x, p.y + 0.5 * u, p.z);
  var df = 1.0;
  for (var i: u32 = 0u; i < it; i = i + 1u) {
    if (q.y < 0.0 || q.y > u || df > 1e30) { break; }
    let s = v * q.y / u;
    q.x = fract((q.x + 1.0 + s) * 0.5) * 2.0 - 1.0 - s;
    let xx = q.x + 0.5 * v;
    let sep = 0.5 * u + sign(xx) * bend * u * (1.0 - exp(-3.0 * abs(xx)));
    if (q.y < sep) {
      let r2 = max(dot(q, q), 1e-12);
      q = vec3f(-v + q.x / r2, u - q.y / r2, q.z / r2);
      df = df / r2;
    } else {
      let dx = q.x + v;
      let dy = q.y - u;
      let r2 = max(dx * dx + dy * dy + q.z * q.z, 1e-12);
      q = vec3f(dx / r2, -dy / r2, q.z / r2);
      df = df / r2;
    }
  }
  return abs(min(q.y, u - q.y)) / df - 0.025;`,
    glsl: `float u = clamp(prm.x, 1.4, 2.2);
  float v = clamp(prm.y, -0.5, 0.5);
  float bend = clamp(prm.z, 0.0, 0.8);
  int it = int(clamp(prm.w, 4.0, 128.0));
  vec3 q = vec3(p.x, p.y + 0.5 * u, p.z);
  float df = 1.0;
  for (int i = 0; i < 128; i++) {
    if (i >= it || q.y < 0.0 || q.y > u || df > 1e30) break;
    float s = v * q.y / u;
    q.x = fract((q.x + 1.0 + s) * 0.5) * 2.0 - 1.0 - s;
    float xx = q.x + 0.5 * v;
    float sep = 0.5 * u + sign(xx) * bend * u * (1.0 - exp(-3.0 * abs(xx)));
    if (q.y < sep) {
      float r2 = max(dot(q, q), 1e-12);
      q = vec3(-v + q.x / r2, u - q.y / r2, q.z / r2);
      df /= r2;
    } else {
      float dx = q.x + v;
      float dy = q.y - u;
      float r2 = max(dx * dx + dy * dy + q.z * q.z, 1e-12);
      q = vec3(dx / r2, -dy / r2, q.z / r2);
      df /= r2;
    }
  }
  return abs(min(q.y, u - q.y)) / df - 0.025;`,
  },
];

// Linear scans, not prebuilt maps: the registry stays small (≤65 after all of
// Phase D) and live scans keep test fixtures free to push a synthetic leaf.
export const leafById = (id) => LEAVES.find((l) => l.id === Number(id)) ?? null;
export const leafByKey = (key) => LEAVES.find((l) => l.key === key) ?? null;

// Highest assigned leaf id — sanitize clamps shapeId to [0, MAX_LEAF_ID].
// A function (not a constant) for the same reason the lookups scan live.
export const maxLeafId = () => LEAVES.reduce((m, l) => Math.max(m, l.id), 0);
export const MAX_LEAF_ID = maxLeafId();

// Registry self-checks, mirrored by leaves.test.mjs (kept here so the no-build
// import smoke also exercises them cheaply at module load in dev).
export function validateLeaves() {
  const failures = [];
  const ids = LEAVES.map((l) => l.id).sort((a, b) => a - b);
  ids.forEach((id, i) => {
    if (id !== i + 1)
      failures.push(`leaf ids must be contiguous from 1 (saw ${id} at ${i})`);
  });
  for (const l of LEAVES) {
    if (!l.params || l.params.length < 1 || l.params.length > 4)
      failures.push(`${l.key}: leaves take 1-4 params`);
    for (const p of l.params || []) {
      // TAG.SHAPES fixed-point ×1000: any UI step must land on the grid.
      if (Math.round(p.step * 1000) !== p.step * 1000 || p.step <= 0)
        failures.push(
          `${l.key}/${p.name}: step must be a positive multiple of 0.001`,
        );
      if (!(p.min <= p.def && p.def <= p.max))
        failures.push(`${l.key}/${p.name}: def outside [min,max]`);
    }
    if (!l.wgsl || !l.glsl) failures.push(`${l.key}: missing an emitter body`);
  }
  return failures;
}

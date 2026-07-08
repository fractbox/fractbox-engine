// ─────────────────────────────────────────────────────────────────────────
// Code generators — turn the operator IR into runnable code.
// ─────────────────────────────────────────────────────────────────────────
// Two backends, one source of truth (operators.js):
//
//   buildWGSL()    assembles the IN-BROWSER interpreter. Every operator's
//                  `wgsl` body becomes a `case` in a switch the march loop
//                  runs per op per iteration. The switch is generated ONCE at
//                  startup; thereafter editing param VALUES or REORDERING ops
//                  is pure data (rewrite the op buffer) — no shader rebuild.
//                  Only adding a brand-new operator TYPE regenerates this.
//                  `numericDE` gates the deOption-3 finite-difference path:
//                  its orbitR helper inlines the full op-switch 4× into
//                  mapDE_single, and that register pressure taxes EVERY
//                  formula ~7-30% (measured, Mandelbox/Mandelbulb @2400²) —
//                  so the renderer compiles an analytic-only variant for the
//                  common case and swaps in the numeric one on demand.
//
//   exportGLSL()   emits a native iterateJIT_ body for the desktop app — the
//                  "design in browser → render on desktop" handoff. The op-list
//                  is the interchange format; this proves it round-trips.
// ─────────────────────────────────────────────────────────────────────────

import { OPERATORS, byKey, effectiveDeOption, activeOps } from "./operators.js";

// ── WGSL interpreter ───────────────────────────────────────────────────────
export function buildWGSL({ numericDE = true } = {}) {
  const cases = OPERATORS.map(
    (op) =>
      `      case ${op.id}u: {${op.wgsl}
      }`,
  ).join("\n");

  return `
struct Globals {
  res     : vec4f,   // x,y = resolution px ; z = fov(rad) ; w = tNear (deep zoom §5)
  camPos  : vec4f,   // xyz = eye ; w = tFar (deep zoom §5 — was unused padding)
  camFwd  : vec4f,
  camRight: vec4f,
  camUp   : vec4f,
  ctrl    : vec4u,   // iters, opCount, addC, maxSteps
  prm     : vec4f,   // bailout, epsilon, deScale, colorMode
  colA    : vec4f,   // low / base color (rgb) ; .w = deOption (0 escape, 2 IFS)
  colB    : vec4f,   // high color (rgb)
  bgc     : vec4f,   // background color (rgb)
  jc      : vec4f,   // Julia constant (xyz) ; .w > 0.5 = Julia mode on
  palA    : vec4f,   // cosine palette a (rgb) ; .w > 0.5 = palette on
  palB    : vec4f,   // cosine palette b (rgb)
  palC    : vec4f,   // cosine palette c (rgb, frequency)
  palD    : vec4f,   // cosine palette d (rgb, phase)
  light   : vec4f,   // light direction (xyz)
  lprm    : vec4f,   // x=ambient, y=rim, z=gloss, w=light intensity
  scene   : vec4u,   // x = objectCount (0 = single-object legacy path)
  hyb     : vec4u,   // Phase 1a spike (IDEAS ①, docs/design/HYBRID_ITERATION.md):
                      // x=aOpCount, y=bOpCount (both 0 = no hybrid, legacy path)
                      // z = (scheduleA & 0xFF) | (scheduleB & 0xFF) << 8
                      //     | (addC_A ? 1<<16 : 0) | (addC_B ? 1<<17 : 0)
                      // w = reserved
  offset  : vec4f,   // deep zoom §3 recenter — xyz = O (the f64 JS-side pan
                      // target, truncated to f32 here). camPos then carries the
                      // RESIDUAL ro_rel = eye−O, not the absolute eye, and every
                      // DE entry point reconstructs p_world = p_rel + offset as
                      // its first op. Scenes render with offset=(0,0,0) (§14,
                      // CSG recenter deferred) — an exact no-op, today's behavior.
  morphB  : vec4u,   // Formula morph go/no-go spike (VIDEO_EXPORT_DRAWER_V2 tier 2,
                      // WGSL-only prototype like hybrid Phase 1a was). Formula B
                      // rides the ops buffer concatenated after A (same trick as
                      // hybrid/CSG): x=bOpCount, y=bIters,
                      // z = flags (bit0 addC · bit1 julia · bits2-3 deOption),
                      // w = B's OWN orbit bailout as f32 bits (bitcast to read).
                      //     Doubles as the on-flag: 0 = morph off (a bailout is
                      //     never 0). B needs its own bound — sharing A's can
                      //     run a power-8 escape orbit to an IFS-sized 1e6,
                      //     overflowing dr in f32 (DE → 0/NaN → blank render).
  morphT  : vec4f,   // x = blend t (0 = pure A … 1 = pure B); yzw = B's Julia c
  morphX  : vec4f,   // x = swell: a dilation (world units) subtracted from the
                      // blended field to fatten the mid-morph "waist" (level-set
                      // erosion where the two fields don't overlap). d − s is
                      // still 1-Lipschitz (pure dilation) → DE-safe. JS shapes
                      // the bump (peak at t=0.5, 0 at the endpoints). yzw reserved.
  colorX  : vec4f,   // Coloring-mode crossfade (timeline transitions): enums
                      // can't lerp, shaded COLORS can. x = blend (0 = off,
                      // legacy shade only), y = the other view's color mode,
                      // z = the other view's palette-on (0/1), w reserved.
                      // The shade runs under both modes and mixes by x.
  post    : vec4f,   // RENDER_QUALITY P0 — post-pass controls (read by the post
                      // shader; the march shader ignores it): x=toneMode
                      // (0 classic pass-through, 1 filmic soft-shoulder),
                      // y=exposure bias (EV, whole frame incl. bg), z=dither
                      // amplitude (LSBs at 8-bit), w=vignette strength.
  lightC  : vec4f,   // RENDER_QUALITY P1 — key light color (rgb, sRGB); w reserved
  mat     : vec4f,   // P1 material/shadow/AO: x=metallic, y=penumbra k (smaller
                      // = softer), z=shadow on (0/1), w=AO strength (0 = off →
                      // the 5-tap march is skipped entirely)
  light2  : vec4f,   // P1 fill light: xyz=dir (derived JS-side from the key
                      // dir — opposite azimuth, flattened elevation), w=intensity
  light2c : vec4f,   // P1 fill color (rgb, sRGB); w reserved
  light3  : vec4f,   // P1 back light: xyz=dir (the key dir mirrored), w=intensity
  light3c : vec4f,   // P1 back color (rgb, sRGB); w reserved
  jitter  : vec4f,   // RENDER_QUALITY P2 — progressive accumulation: x,y =
                      // subpixel jitter (pixels, R2 sequence), z = accumulation
                      // weight 1/(N+1) (read by the accum pass, not the march),
                      // w reserved. writeGlobals zeroes it; writeJitter (a
                      // 16-byte partial upload) sets it between accum frames.
  env     : vec4f,   // RENDER_QUALITY P3 — sky/environment: x=skyBlend (0 =
                      // legacy screen-space bg, exact), y=sun glow, z=ground
                      // dim, w=IBL ambient tint amount. One UI macro ("Sky")
                      // drives x=y=w JS-side; z is fixed.
  fog     : vec4f,   // P3 atmosphere/bloom: x=fog amount (0 = legacy 0.6·t/tFar
                      // fade, exact), y=sun in-scatter, z=bloom strength
                      // (composited in the post pass; 0 skips the bloom passes
                      // entirely), w=bloom threshold (pre-tonemap HDR units).
  dof     : vec4f,   // RENDER_QUALITY P4 — thin-lens depth of field, riding the
                      // P2 accumulation substrate: x=lens radius (world units,
                      // 0 = pinhole/off), y=focus distance (autofocus = orbit
                      // distance), z,w = this sample's lens point (unit disk;
                      // (0,0) on the base frame → interaction stays sharp,
                      // bokeh converges while idle). zw written per sample by
                      // writeJitter alongside the subpixel jitter word.
};
struct Op { opType: u32, p0: f32, p1: f32, p2: f32 };

// CSG Phase 1b — one descriptor per scene object. FULL v1 layout: 80 bytes,
// 5 × vec4-sized words, every field 16-B aligned (all scalars → struct align 4,
// stride 80). Byte offsets:
//   word 0 (off  0, u32×4): opStart, opCount, iters, flags
//   word 1 (off 16, f32×4): ox, oy, oz, uscale          (origin + uniform scale)
//   word 2 (off 32, f32×4): qx, qy, qz, qw              (rotation quaternion, local→world)
//   word 3 (off 48, f32×4): jcx, jcy, jcz, blendK       (per-object Julia c + smin blend)
//   word 4 (off 64, f32×4): primParam, primParam2, pad1, pad2
//     primParam  = box halfExtent / sphere r / torus R / cylinder r / capsule r / plane thickness
//                  — REPURPOSED for IFS objects (objType 0) as the box-DE base half-extent
//                    when boxBase (bit11) is set (IFS objects don't otherwise use primParam).
//     primParam2 = torus minor r / cylinder half-height / capsule half-height (unused by box/sphere/plane)
// flags bits: bit0 addC · bits1-2 deOption · bit3 julia · bit4 looseDE ·
//             bits5-6 combineType (0 union · 1 smooth-union · 2 subtract · 3 intersect) ·
//             bits7-10 objType (0 IFS op-slice · 1 box · 2 sphere · 3 torus ·
//                               4 cylinder · 5 capsule · 6 plane) — 4 bits, 0-15 ·
//             bit11 boxBase (IFS deOption-2 only: finalize with boxDE(pos,he)/|w| → flat
//                            cube faces instead of round dust; primParam = he. No-op for
//                            escape/primitive objects).
//             (bits 12+ free.)
struct Obj {
  opStart   : u32,
  opCount   : u32,
  iters     : u32,
  flags     : u32,
  ox        : f32,
  oy        : f32,
  oz        : f32,
  uscale    : f32,
  qx        : f32,
  qy        : f32,
  qz        : f32,
  qw        : f32,
  jcx       : f32,
  jcy       : f32,
  jcz       : f32,
  blendK    : f32,
  primParam : f32,
  primParam2: f32,
  pad1      : f32,
  pad2      : f32,
  colR      : f32,   // word 5 (off 80): per-object albedo (sRGB) — see §3.8 per-object color
  colG      : f32,
  colB      : f32,
  pad3      : f32,
};

@group(0) @binding(0) var<uniform> G : Globals;
@group(0) @binding(1) var<storage, read> ops : array<Op>;
@group(0) @binding(2) var<storage, read> objects : array<Obj>;

${numericDE ? `
// Orbit escape radius only — the numeric-DE probe (COVERAGE_PLAN §3 B1). The
// same loop as mapDE_single with no DE finalize; sampled 4× (center + 3 axis
// offsets) for the finite-difference gradient. Emitted only in the numeric
// pipeline variant (see buildWGSL doc — its 4× inlining taxes analytic
// formulas via register pressure otherwise).
fn orbitR(p0: vec3f) -> f32 {
  var pos = p0;
  var w = 1.0;
  let c = select(p0, G.jc.xyz, G.jc.w > 0.5);
  let n = G.ctrl.y;
  for (var i: u32 = 0u; i < G.ctrl.x; i = i + 1u) {
    for (var o: u32 = 0u; o < n; o = o + 1u) {
      let op = ops[o];
      switch op.opType {
${cases}
        default: {}
      }
    }
    if (G.ctrl.z != 0u) { pos = pos + c; }
    if (dot(pos, pos) > G.prm.x) { break; }
  }
  return length(pos);
}` : ``}

// Single-object iteration body (today's exact map). Reads the global ctrl/jc/colA.
fn mapDE_single(p0: vec3f) -> f32 {
${numericDE ? `
  // DEoption 3 — numeric finite-difference DE (MB3D numDiff style, for
  // W_BULB_NUMERIC maps with no analytic dr): DE = R·ln(R)/(|∇R| + 0.06),
  // falling back to the conservative 0.5·√R heuristic where the gradient
  // vanishes. ln clamps at R=1 so interior (non-escaped) points read DE→0
  // rather than negative. v1 scope: flat single formulas only (sanitize
  // rejects numeric ops in hybrid/morph/scene stacks).
  if (G.colA.w >= 2.5) {
    let R = orbitR(p0);
    let eps = 1e-4 * max(1.0, length(p0));
    let g = vec3f(orbitR(p0 + vec3f(eps, 0.0, 0.0)) - R,
                  orbitR(p0 + vec3f(0.0, eps, 0.0)) - R,
                  orbitR(p0 + vec3f(0.0, 0.0, eps)) - R) / eps;
    let gl = length(g);
    let de = R * log(max(R, 1.0)) / (gl + 0.06);
    return select(0.5 * sqrt(max(R, 0.0)), de, gl > 1e-3);
  }` : ``}
  var pos = p0;
  var w = 1.0;
  let c = select(p0, G.jc.xyz, G.jc.w > 0.5);  // Julia: fixed c, else sample point
  let n = G.ctrl.y;
  for (var i: u32 = 0u; i < G.ctrl.x; i = i + 1u) {
    for (var o: u32 = 0u; o < n; o = o + 1u) {
      let op = ops[o];
      switch op.opType {
${cases}
        default: {}
      }
    }
    if (G.ctrl.z != 0u) { pos = pos + c; }
    if (dot(pos, pos) > G.prm.x) { break; }
  }
  let r = length(pos);
  // DEoption 0 — escape-time DE (Mandelbulb / power): 0.5·ln(r)·r / dr, with the
  // analytic derivative dr carried in w. DEoption 2 — analytic IFS r/|w|.
  if (G.colA.w < 1.0) { return 0.5 * log(max(r, 1e-9)) * r / max(abs(w), 1e-9); }
  return r / max(abs(w), 1e-9);
}

// Phase 1a go/no-go spike (IDEAS ①, docs/design/HYBRID_ITERATION.md §5 phase 1a).
// WGSL-only prototype: alternates op-slot A / slot B across OUTER iterations by
// a {schedA, schedB} repeating schedule, sharing one (pos, w) accumulator — the
// same op buffer concat/slice trick CSG's objIterDE uses, just re-sliced per
// iteration instead of per object. v1-scoped decision (the doc's §3.3): this is
// ONLY DE-safe for a same-family hybrid (both slots IFS, or both escape) — a
// mixed hybrid has no valid w bookkeeping (test case 2 below deliberately
// exercises the unsafe case to confirm it fails visibly, not silently).
fn mapDE_hybrid(p0: vec3f) -> f32 {
  var pos = p0;
  var w = 1.0;
  let c = select(p0, G.jc.xyz, G.jc.w > 0.5);
  let aCount = G.hyb.x;
  let bCount = G.hyb.y;
  let schedA = G.hyb.z & 0xFFu;
  let schedB = (G.hyb.z >> 8u) & 0xFFu;
  let addCA = (G.hyb.z & (1u << 16u)) != 0u;
  let addCB = (G.hyb.z & (1u << 17u)) != 0u;
  let period = max(schedA + schedB, 1u);
  for (var i: u32 = 0u; i < G.ctrl.x; i = i + 1u) {
    let useB = (i % period) >= schedA;
    let lo = select(0u, aCount, useB);
    let hi = select(aCount, aCount + bCount, useB);
    for (var o: u32 = lo; o < hi; o = o + 1u) {
      let op = ops[o];
      switch op.opType {
${cases}
        default: {}
      }
    }
    let addC = select(addCA, addCB, useB) || (G.jc.w > 0.5);
    if (addC) { pos = pos + c; }
    if (dot(pos, pos) > G.prm.x) { break; }
  }
  let r = length(pos);
  if (G.colA.w < 1.0) { return 0.5 * log(max(r, 1e-9)) * r / max(abs(w), 1e-9); }
  return r / max(abs(w), 1e-9);
}

// Formula morph (VIDEO_EXPORT_DRAWER_V2 tier-2 spike) — one formula's full
// orbit over an op slice [lo,hi), with its own iters/addC/c/deOption. The 4th
// copy of the op switch (mapDE_single/mapDE_hybrid/objIterDE have the others);
// deliberately NOT refactored into them — additive spike, zero legacy risk.
fn morphIter(p0: vec3f, lo: u32, hi: u32, itersN: u32, addC: bool, cIn: vec3f, deOpt: u32, bail: f32) -> f32 {
  var pos = p0;
  var w = 1.0;
  for (var i: u32 = 0u; i < itersN; i = i + 1u) {
    for (var o: u32 = lo; o < hi; o = o + 1u) {
      let op = ops[o];
      switch op.opType {
${cases}
        default: {}
      }
    }
    if (addC) { pos = pos + cIn; }
    if (dot(pos, pos) > bail) { break; }
  }
  let r = length(pos);
  if (deOpt == 0u) { return 0.5 * log(max(r, 1e-9)) * r / max(abs(w), 1e-9); }
  return r / max(abs(w), 1e-9);
}

// Morph-aware coloring metrics — the same two-slice walk as morphIter but
// returning the orbit-trap minimum / escape fraction instead of a distance.
// Without these, trap/band coloring reads only the DOMINANT formula's orbits
// and the surface pattern snaps at the midpoint slot-swap while the geometry
// glides (the "jarring flip"). mix(mA, mB, t) is swap-symmetric like the DE,
// so coloring is now continuous across the whole transition.
fn morphTrap(p0: vec3f, lo: u32, hi: u32, itersN: u32, addC: bool, cIn: vec3f, bail: f32) -> f32 {
  var pos = p0;
  var w = 1.0;
  var tr = 1.0e9;
  for (var i: u32 = 0u; i < itersN; i = i + 1u) {
    for (var o: u32 = lo; o < hi; o = o + 1u) {
      let op = ops[o];
      switch op.opType {
${cases}
        default: {}
      }
    }
    if (addC) { pos = pos + cIn; }
    tr = min(tr, length(pos));
    if (dot(pos, pos) > bail) { break; }
  }
  return tr;
}

fn morphEsc(p0: vec3f, lo: u32, hi: u32, itersN: u32, addC: bool, cIn: vec3f, bail: f32) -> f32 {
  var pos = p0;
  var w = 1.0;
  var esc: u32 = itersN;
  for (var i: u32 = 0u; i < itersN; i = i + 1u) {
    for (var o: u32 = lo; o < hi; o = o + 1u) {
      let op = ops[o];
      switch op.opType {
${cases}
        default: {}
      }
    }
    if (addC) { pos = pos + cIn; }
    if (dot(pos, pos) > bail) { esc = i; break; }
  }
  return f32(esc) / f32(max(itersN, 1u));
}

// d = mix(dA, dB, t): a convex blend of two 1-Lipschitz distance bounds is
// itself a valid bound, so sphere tracing stays safe — the intermediate field
// is a genuine 3D morph, not an image dissolve. Formula A lives in the legacy
// globals (ctrl/jc/colA/prm.x); B rides the morphB/morphT words + the
// concatenated op slice, with its OWN bailout (bitcast from morphB.w).
fn mapDE_morph(p0: vec3f) -> f32 {
  let cA = select(p0, G.jc.xyz, G.jc.w > 0.5);
  let dA = morphIter(p0, 0u, G.ctrl.y, G.ctrl.x, G.ctrl.z != 0u, cA, u32(G.colA.w), G.prm.x);
  let bCount = G.morphB.x;
  let juliaB = (G.morphB.z & 2u) != 0u;
  let addB = ((G.morphB.z & 1u) != 0u) || juliaB;
  let deB = (G.morphB.z >> 2u) & 3u;
  let cB = select(p0, G.morphT.yzw, juliaB);
  let bailB = bitcast<f32>(G.morphB.w);
  let dB = morphIter(p0, G.ctrl.y, G.ctrl.y + bCount, G.morphB.y, addB, cB, deB, bailB);
  return mix(dA, dB, clamp(G.morphT.x, 0.0, 1.0)) - G.morphX.x;
}

// Analytic box SDF (exact distance bound → ideal CSG child). Defined before
// objIterDE so the IFS box-DE base (flags bit11) can call it.
fn boxDE(p: vec3f, he: f32) -> f32 {
  let q = abs(p) - vec3f(he);
  return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

// CSG Phase 1b — one scene object's DE, driven by its Obj descriptor (op slice,
// iters, per-object addC + deOption + Julia). Julia (flags bit3): c = the object's
// own juliaC constant instead of the sample point.
fn objIterDE(p0: vec3f, ob: Obj) -> f32 {
  var pos = p0;
  var w = 1.0;
  let julia = (ob.flags & 8u) != 0u;
  let c = select(p0, vec3f(ob.jcx, ob.jcy, ob.jcz), julia);
  let addC = ((ob.flags & 1u) != 0u) || julia;
  let deOpt = (ob.flags >> 1u) & 3u;
  let lo = ob.opStart;
  let hi = ob.opStart + ob.opCount;
  for (var i: u32 = 0u; i < ob.iters; i = i + 1u) {
    for (var o: u32 = lo; o < hi; o = o + 1u) {
      let op = ops[o];
      switch op.opType {
${cases}
        default: {}
      }
    }
    if (addC) { pos = pos + c; }
    if (dot(pos, pos) > G.prm.x) { break; }
  }
  let r = length(pos);
  if (deOpt == 0u) { return 0.5 * log(max(r, 1e-9)) * r / max(abs(w), 1e-9); }
  // Box-DE base (flags bit11): finalize the IFS with an exact box SDF instead of
  // length(pos), so the satellites render as flat-faced cubes (the canonical DE
  // for cube-family fractals — Menger/Cantor). primParam = the box half-extent.
  // Box is exact and /|w| is the standard IFS scale accumulation → DE-safe, no
  // step-clamp. Escape (deOpt 0) keeps its point-base finalization above.
  if ((ob.flags & 2048u) != 0u) { return boxDE(pos, ob.primParam) / max(abs(w), 1e-9); }
  return r / max(abs(w), 1e-9);
}

// Analytic sphere SDF (exact distance bound).
fn sphereDE(p: vec3f, r: f32) -> f32 {
  return length(p) - r;
}

// Analytic torus SDF (exact). R = major radius (xz ring), r = minor (tube) radius.
fn torusDE(p: vec3f, R: f32, r: f32) -> f32 {
  let q = vec2f(length(p.xz) - R, p.y);
  return length(q) - r;
}

// Analytic capped-cylinder SDF (exact). r = radius, h = half-height (axis = y).
fn cylinderDE(p: vec3f, r: f32, h: f32) -> f32 {
  let d = vec2f(length(p.xz) - r, abs(p.y) - h);
  return min(max(d.x, d.y), 0.0) + length(max(d, vec2f(0.0)));
}

// Analytic vertical-capsule SDF (exact). r = radius, h = half-height (axis = y).
fn capsuleDE(p: vec3f, r: f32, h: f32) -> f32 {
  var q = p;
  q.y = q.y - clamp(q.y, -h, h);
  return length(q) - r;
}

// Slab/ground-plane SDF (exact, bounded → DE-safe for union). y = 0 plane with a
// half-thickness "thick"; thick = 0 → a zero-thickness plane abs(p.y).
fn planeDE(p: vec3f, thick: f32) -> f32 {
  return abs(p.y) - thick;
}

// Rotate a vector by a quaternion (q = local→world). v' = q v q*.
fn qrot(q: vec4f, v: vec3f) -> vec3f {
  let u = q.xyz;
  let s = q.w;
  return 2.0 * dot(u, v) * u + (s * s - dot(u, u)) * v + 2.0 * s * cross(u, v);
}

// Polynomial smooth-min (under-estimates by ≤ k/4 at the seam → DE-safe).
fn sminP(a: f32, b: f32, k: f32) -> f32 {
  if (k <= 0.0) { return min(a, b); }
  let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
  return mix(b, a, h) - k * h * (1.0 - h);
}
// Smooth max — the carve dual of sminP (k=0 → hard max). Rounds the seam where a
// subtract/intersect cuts, instead of a sharp edge. Over-estimates like max(), so
// carving scenes already march with the tighter CARVE_DESCALE (preview.js).
fn smaxP(a: f32, b: f32, k: f32) -> f32 { return -sminP(-a, -b, k); }

// Placed distance of one object at world point p0: rigid world→local transform
// (inverse-rotate by the conjugate quat, then /uscale) then the type dispatch.
// Shared by mapDE and sceneTint so the two never drift.
fn objDist(p0: vec3f, ob: Obj) -> f32 {
  let qinv = vec4f(-ob.qx, -ob.qy, -ob.qz, ob.qw);
  let pk = qrot(qinv, p0 - vec3f(ob.ox, ob.oy, ob.oz)) / ob.uscale;
  let objType = (ob.flags >> 7u) & 0xFu;
  if      (objType == 1u) { return boxDE(pk, ob.primParam) * ob.uscale; }
  else if (objType == 2u) { return sphereDE(pk, ob.primParam) * ob.uscale; }
  else if (objType == 3u) { return torusDE(pk, ob.primParam, ob.primParam2) * ob.uscale; }
  else if (objType == 4u) { return cylinderDE(pk, ob.primParam, ob.primParam2) * ob.uscale; }
  else if (objType == 5u) { return capsuleDE(pk, ob.primParam, ob.primParam2) * ob.uscale; }
  else if (objType == 6u) { return planeDE(pk, ob.primParam) * ob.uscale; }
  return objIterDE(pk, ob) * ob.uscale;
}

// Scene DE = combine over objects. objectCount==0 → today's exact single map.
fn mapDE(p_rel: vec3f) -> f32 {
  let p0 = p_rel + G.offset.xyz; // deep zoom §3.2 — the ONE reconstruction point
  if (G.morphB.w != 0u) { return mapDE_morph(p0); } // formula-morph spike
  if (G.hyb.x + G.hyb.y > 0u) { return mapDE_hybrid(p0); } // Phase 1a spike
  if (G.scene.x == 0u) { return mapDE_single(p0); }
  var d = 1.0e9;
  for (var k: u32 = 0u; k < G.scene.x; k = k + 1u) {
    let ob = objects[k];
    let dk = objDist(p0, ob);
    // Combine k into the accumulated d. NOTE: subtract/intersect use (smooth-)max,
    // which is NOT a valid distance bound (it over-estimates → the marcher
    // oversteps the carved walls). The scene compensates with a tighter global
    // deScale when any object carves (see preview.js sceneDeScale CARVE_DESCALE).
    let combine = (ob.flags >> 5u) & 3u;
    if      (combine == 1u) { d = sminP(d, dk, ob.blendK); }    // smooth-union
    else if (combine == 2u) { d = smaxP(d, -dk, ob.blendK); }   // subtract: carve k out (blendK rounds the cut)
    else if (combine == 3u) { d = smaxP(d, dk, ob.blendK); }    // intersect: keep overlap (blendK rounds the seam)
    else                    { d = min(d, dk); }                 // union
  }
  return d;
}

// Per-object color (§3.8): the albedo of the object whose surface the hit belongs
// to. Re-runs the combine tracking which object last "owned" the surface — union
// picks the nearest member; carve picks the cutter when its wall is the surface.
// Cheap: one extra scene pass, only at the final hit point (like orbitTrap).
fn sceneTint(p_rel: vec3f) -> vec3f {
  let p0 = p_rel + G.offset.xyz; // deep zoom §3.2 (no-op for scenes — offset is 0)
  var d = 1.0e9;
  var win: u32 = 0u;
  for (var k: u32 = 0u; k < G.scene.x; k = k + 1u) {
    let ob = objects[k];
    let dk = objDist(p0, ob);
    let combine = (ob.flags >> 5u) & 3u;
    if      (combine == 2u) { let nd = smaxP(d, -dk, ob.blendK); if (-dk > d) { win = k; } d = nd; }
    else if (combine == 3u) { let nd = smaxP(d,  dk, ob.blendK); if ( dk > d) { win = k; } d = nd; }
    else {
      if (dk < d) { win = k; }
      if (combine == 1u) { d = sminP(d, dk, ob.blendK); } else { d = min(d, dk); }
    }
  }
  let ob = objects[win];
  return vec3f(ob.colR, ob.colG, ob.colB);
}

// Orbit trap: the closest the iterated point came to the origin. Re-runs the
// iteration once (only at the final hit point, so it's cheap) to drive coloring.
fn orbitTrap(p_rel: vec3f) -> f32 {
  let p0 = p_rel + G.offset.xyz; // deep zoom §3.2
  // Formula morph: blend BOTH formulas' trap metrics (see morphTrap) — reading
  // only the dominant one snaps the surface pattern at the midpoint slot-swap.
  if (G.morphB.w != 0u) {
    let cA = select(p0, G.jc.xyz, G.jc.w > 0.5);
    let tA = morphTrap(p0, 0u, G.ctrl.y, G.ctrl.x, G.ctrl.z != 0u, cA, G.prm.x);
    let juliaB = (G.morphB.z & 2u) != 0u;
    let addB = ((G.morphB.z & 1u) != 0u) || juliaB;
    let cB = select(p0, G.morphT.yzw, juliaB);
    let tB = morphTrap(p0, G.ctrl.y, G.ctrl.y + G.morphB.x, G.morphB.y, addB, cB,
                       bitcast<f32>(G.morphB.w));
    return mix(tA, tB, clamp(G.morphT.x, 0.0, 1.0));
  }
  var pos = p0;
  var w = 1.0;
  let c = select(p0, G.jc.xyz, G.jc.w > 0.5);  // Julia: fixed c, else sample point
  let n = G.ctrl.y;
  // Hybrid iteration (§3.6) — the same schedule branch mapDE_hybrid uses, so
  // glow coloring works on a hybrid instead of only seeing slot A.
  let hybOn = G.hyb.x + G.hyb.y > 0u;
  let schedA = G.hyb.z & 0xFFu;
  let schedB = (G.hyb.z >> 8u) & 0xFFu;
  let addCA = (G.hyb.z & (1u << 16u)) != 0u;
  let addCB = (G.hyb.z & (1u << 17u)) != 0u;
  let period = max(schedA + schedB, 1u);
  var tr = 1.0e9;
  for (var i: u32 = 0u; i < G.ctrl.x; i = i + 1u) {
    let useB = hybOn && ((i % period) >= schedA);
    let lo = select(0u, G.hyb.x, useB);
    let hi = select(select(n, G.hyb.x, hybOn), G.hyb.x + G.hyb.y, useB);
    for (var o: u32 = lo; o < hi; o = o + 1u) {
      let op = ops[o];
      switch op.opType {
${cases}
        default: {}
      }
    }
    let addC = select(G.ctrl.z != 0u, select(addCA, addCB, useB) || (G.jc.w > 0.5), hybOn);
    if (addC) { pos = pos + c; }
    tr = min(tr, length(pos));
    if (dot(pos, pos) > G.prm.x) { break; }
  }
  return tr;
}

// Escape iteration fraction (for "bands" coloring): how many iterations until the
// point flies past the bailout, normalized 0..1. Re-runs the iteration once.
fn escapeIter(p_rel: vec3f) -> f32 {
  let p0 = p_rel + G.offset.xyz; // deep zoom §3.2
  // Formula morph: blend both formulas' escape fractions (mirrors orbitTrap).
  if (G.morphB.w != 0u) {
    let cA = select(p0, G.jc.xyz, G.jc.w > 0.5);
    let eA = morphEsc(p0, 0u, G.ctrl.y, G.ctrl.x, G.ctrl.z != 0u, cA, G.prm.x);
    let juliaB = (G.morphB.z & 2u) != 0u;
    let addB = ((G.morphB.z & 1u) != 0u) || juliaB;
    let cB = select(p0, G.morphT.yzw, juliaB);
    let eB = morphEsc(p0, G.ctrl.y, G.ctrl.y + G.morphB.x, G.morphB.y, addB, cB,
                      bitcast<f32>(G.morphB.w));
    return mix(eA, eB, clamp(G.morphT.x, 0.0, 1.0));
  }
  var pos = p0;
  var w = 1.0;
  let c = select(p0, G.jc.xyz, G.jc.w > 0.5);
  let n = G.ctrl.y;
  // Hybrid iteration (§3.6) — mirrors orbitTrap's schedule branch above.
  let hybOn = G.hyb.x + G.hyb.y > 0u;
  let schedA = G.hyb.z & 0xFFu;
  let schedB = (G.hyb.z >> 8u) & 0xFFu;
  let addCA = (G.hyb.z & (1u << 16u)) != 0u;
  let addCB = (G.hyb.z & (1u << 17u)) != 0u;
  let period = max(schedA + schedB, 1u);
  var esc: u32 = G.ctrl.x;
  for (var i: u32 = 0u; i < G.ctrl.x; i = i + 1u) {
    let useB = hybOn && ((i % period) >= schedA);
    let lo = select(0u, G.hyb.x, useB);
    let hi = select(select(n, G.hyb.x, hybOn), G.hyb.x + G.hyb.y, useB);
    for (var o: u32 = lo; o < hi; o = o + 1u) {
      let op = ops[o];
      switch op.opType {
${cases}
        default: {}
      }
    }
    let addC = select(G.ctrl.z != 0u, select(addCA, addCB, useB) || (G.jc.w > 0.5), hybOn);
    if (addC) { pos = pos + c; }
    if (dot(pos, pos) > G.prm.x) { esc = i; break; }
  }
  return f32(esc) / f32(max(G.ctrl.x, 1u));
}

// e scales with hit distance t (deep zoom §3.4/§5) — a fixed epsilon straddles
// unrelated geometry once the near/far range is no longer a fixed [0.02, 80].
// p_rel is the RESIDUAL hit point (deep zoom §3.4) — perturbing it here, in
// residual space, before each mapDE call reconstructs per sample, is what makes
// the 4-tap finite difference precision-correct: perturbing an already-
// reconstructed absolute coordinate instead would round the tiny e away once
// |offset| is large enough (the "naive mistake" §3.4 warns against). No other
// change needed here — mapDE does the one reconstruction per call (§3.2).
fn calcNormal(p_rel: vec3f, t: f32) -> vec3f {
  let e = vec2f(1.0, -1.0) * clamp(t * 3e-5, 1e-6, 6e-4);
  return normalize(
      e.xyy * mapDE(p_rel + e.xyy) +
      e.yyx * mapDE(p_rel + e.yyx) +
      e.yxy * mapDE(p_rel + e.yxy) +
      e.xxx * mapDE(p_rel + e.xxx));
}

// ── RENDER_QUALITY P1 shading helpers ──────────────────────────────────────
// All three ride the generic mapDE dispatcher, so they cover single/hybrid/
// morph formulas AND CSG scenes with no extra plumbing.

// Penumbra soft shadow (iq): march from the hit toward the light, tracking the
// narrowest angular clearance k·d/s. Uses the SAME deScale the primary march
// used (G.prm.z) so loose-DE formulas don't leak light, and starts at an
// eps·t-scaled offset to dodge self-shadow acne (deep-zoom safe, same
// reasoning as calcNormal).
fn softShadow(p0: vec3f, n: vec3f, ldir: vec3f, tHit: f32, k: f32) -> f32 {
  // Lift off along the NORMAL (not ldir): a light-direction offset hugs the
  // surface at grazing angles and self-shadows the whole object.
  let org = p0 + n * max(G.prm.y * tHit * 12.0, 2e-3);
  var sh = 1.0;
  var s = max(G.prm.y * tHit * 8.0, 1e-3);
  for (var i: u32 = 0u; i < 32u; i = i + 1u) {
    let d = mapDE(org + ldir * s) * G.prm.z;
    sh = min(sh, k * d / s);
    s = s + clamp(d, 0.01, 0.5);
    if (sh < 0.005 || s > 12.0) { break; }
  }
  return clamp(sh, 0.0, 1.0);
}

// 5-tap DE ambient occlusion (iq's fractal AO): walk the normal comparing
// expected vs actual clearance. Probe radius is t-scaled (zoom-invariant, same
// philosophy as the marcher's eps·t); occ/r keeps it dimensionless.
fn calcAO(p0: vec3f, n: vec3f, tHit: f32) -> f32 {
  let r = clamp(tHit * 0.3, 0.1, 1.5);
  var occ = 0.0;
  var sca = 1.0;
  for (var i: u32 = 1u; i <= 5u; i = i + 1u) {
    let h = r * (0.01 + 0.12 * f32(i) / 5.0);
    occ = occ + (h - mapDE(p0 + n * h)) * sca;
    sca = sca * 0.85;
  }
  return clamp(1.0 - 3.0 * G.mat.w * (occ / r), 0.0, 1.0);
}

// P3 environment: the bg gradient lifted into DIRECTION space + sun glow
// around the key light + ground dim below the horizon. It both IS the
// background (misses, fog target) and LIGHTS the object (IBL ambient tint,
// env reflections) — one world. Returns LINEAR color.
fn envColor(rd: vec3f) -> vec3f {
  let tg = clamp(rd.y * 0.5 + 0.5, 0.0, 1.0);
  var sky = s2l(mix(G.bgc.rgb * 0.35, G.bgc.rgb, tg));
  sky = mix(sky * (1.0 - G.env.z), sky, smoothstep(-0.35, 0.12, rd.y)); // ground dim
  let sunAmt = pow(max(dot(rd, normalize(G.light.xyz)), 0.0), 24.0);
  return sky + s2l(G.lightC.rgb) * (sunAmt * G.env.y);
}

// Cook-Torrance GGX specular (Schlick-GGX geometry, Schlick fresnel). Returns
// the full radiance factor for one light (the ×NoL is folded in: D·G·F/(4·NoV)).
fn ggxSpec(n: vec3f, v: vec3f, l: vec3f, rough: f32, f0: vec3f) -> vec3f {
  let h = normalize(v + l);
  let ndl = max(dot(n, l), 0.0);
  let ndv = max(dot(n, v), 1e-3);
  let ndh = max(dot(n, h), 0.0);
  let a = rough * rough;
  let a2 = a * a;
  let dd = ndh * ndh * (a2 - 1.0) + 1.0;
  let D = a2 / (3.14159265 * dd * dd);
  let kk = (rough + 1.0) * (rough + 1.0) / 8.0;
  let gv = ndv / (ndv * (1.0 - kk) + kk);
  let gl = ndl / (ndl * (1.0 - kk) + kk);
  let F = f0 + (vec3f(1.0) - f0) * pow(1.0 - max(dot(h, v), 0.0), 5.0);
  return D * gv * gl * F / max(4.0 * ndv, 1e-3);
}

struct VSOut { @builtin(position) clip: vec4f, @location(0) uv: vec2f };

@vertex fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  var tri = array<vec2f, 3>(vec2f(-1.0,-1.0), vec2f(3.0,-1.0), vec2f(-1.0,3.0));
  var o: VSOut;
  let xy = tri[vi];
  o.clip = vec4f(xy, 0.0, 1.0);
  o.uv = (xy + vec2f(1.0)) * 0.5;
  return o;
}

// sRGB → linear (exact piecewise EOTF). Albedo/picker/theme colors are authored
// in sRGB; linearize before lighting so the post pass's exact piecewise encode
// round-trips them to the picked color (issue #6). P0 upgraded both directions
// from pow-2.2 to the real curve TOGETHER — mixing a 2.2 decode with a piecewise
// encode would break the round trip near black (−64% at sRGB 0.05).
fn s2l(c: vec3f) -> vec3f {
  let cc = max(c, vec3f(0.0));
  let lo = cc / 12.92;
  let hi = pow((cc + vec3f(0.055)) / 1.055, vec3f(2.4));
  return select(hi, lo, cc <= vec3f(0.04045));
}

// The mixT source for a given color mode (0 surface / 1 orbit-trap / 2 bands)
// and the albedo for a given palette toggle — factored out so the coloring-
// mode crossfade (colorX) can shade under BOTH transition endpoints' modes.
fn mixTFor(mode: f32, p: vec3f, nz: f32) -> f32 {
  if (mode > 1.5) { return escapeIter(p); }
  if (mode > 0.5) { return clamp(orbitTrap(p) / 1.5, 0.0, 1.0); }
  return 0.5 + 0.5 * nz;
}
fn albedoFor(palOn: bool, mixT: f32) -> vec3f {
  if (palOn) {
    return clamp(G.palA.rgb + G.palB.rgb * cos(6.2831853 * (G.palC.rgb * mixT + G.palD.rgb)),
                 vec3f(0.0), vec3f(1.0));
  }
  return mix(G.colA.rgb, G.colB.rgb, mixT);
}

@fragment fn fs(@location(0) uv: vec2f) -> @location(0) vec4f {
  // P2: subpixel jitter (pixels → ndc is 2 units across the frame). Zero when
  // not accumulating — the expression is exact identity then.
  let ndc = uv * 2.0 - vec2f(1.0) + 2.0 * G.jitter.xy / G.res.xy;
  let aspect = G.res.x / G.res.y;
  let tanF = tan(0.5 * G.res.z);
  var rd = normalize(G.camFwd.xyz
      + (ndc.x * aspect * tanF) * G.camRight.xyz
      + (ndc.y * tanF) * G.camUp.xyz);
  // Deep zoom §3.1 — camPos carries the RESIDUAL ro_rel = eye−offset (small,
  // f32-friendly), not the absolute eye. Every p below is therefore p_rel;
  // mapDE/orbitTrap/escapeIter/sceneTint each reconstruct p_world = p_rel +
  // G.offset once, at their own entry (§3.2). For scenes offset is (0,0,0),
  // so ro is the absolute eye and this is exactly today's behavior.
  var ro = G.camPos.xyz;
  // P4 thin-lens DOF: offset the eye on the lens disk and re-aim at the focus
  // plane. Per accumulation sample (zw = the sample's lens point); the base
  // frame uses the lens center, so interactive frames stay sharp and the
  // bokeh converges while idle — and offline export samples make it clean.
  if (G.dof.x > 0.0) {
    let fp = ro + rd * G.dof.y;
    ro = ro + (G.camRight.xyz * G.dof.z + G.camUp.xyz * G.dof.w) * G.dof.x;
    rd = normalize(fp - ro);
  }

  let bg = s2l(mix(G.bgc.rgb * 0.35, G.bgc.rgb, clamp(uv.y, 0.0, 1.0)));

  var t = G.res.w;
  var steps: u32 = 0u;
  var hit = false;
  let maxSteps = G.ctrl.w;
  let eps = G.prm.y;
  let tFar = G.camPos.w;
  for (steps = 0u; steps < maxSteps; steps = steps + 1u) {
    let p = ro + rd * t;
    let d = mapDE(p) * G.prm.z;
    if (d < eps * t) { hit = true; break; }
    t = t + d;
    if (t > tFar) { break; }
  }
  // P3: sky blend — 0 keeps the legacy screen-space gradient EXACTLY; > 0
  // blends toward the directional environment (shared by misses, the fog
  // target, IBL ambient and env reflections below).
  var bgOut = bg;
  if (G.env.x > 0.0) { bgOut = mix(bg, envColor(rd), clamp(G.env.x, 0.0, 1.0)); }
  // Miss → background, displayed AS AUTHORED. Historically the single-pass
  // shader returned the linearized gradient raw (no encode) — a "double-dark"
  // quirk that showed every theme's bg far darker than its picked color ("
  // everything feels dark"). The post pass now encodes the linear gradient,
  // so the displayed bg matches the picker (the #6 round-trip, applied to the
  // background at last) — and matches the fade target, killing the old
  // miss/fade seam as well.
  if (!hit) { return vec4f(bgOut, 1.0); }

  let p = ro + rd * t;
  let nrm = calcNormal(p, t);
  // Step-count "cavity dust" — the pre-P1 AO heuristic, kept as a softened
  // multiplier (fractal artists like the look); real occlusion is calcAO below.
  let stepAO = 1.0 - f32(steps) / f32(maxSteps);
  let cav = 0.7 + 0.3 * stepAO;

  let lightDir = normalize(G.light.xyz);
  let diff = max(dot(nrm, lightDir), 0.0);
  let rim = pow(1.0 - max(dot(nrm, -rd), 0.0), 2.0);
  let amb = G.lprm.x;
  // P1: real AO + key-light soft shadow (each skipped entirely at strength 0 —
  // the smooth interactive tier flips these words, no recompile).
  var occ = 1.0;
  if (G.mat.w > 0.0) { occ = calcAO(p, nrm, t); }
  var sh = 1.0;
  if (G.mat.z > 0.5) { sh = 0.15 + 0.85 * softShadow(p, nrm, lightDir, t, G.mat.y); }

  // mixT source by mode: 0 surface (normal), 1 orbit-trap glow, 2 escape bands.
  let mixT = mixTFor(G.prm.w, p, nrm.z);

  // Albedo: scenes color per-object (§3.8); single objects use the cosine palette
  // (a + b·cos(2π(c·t + d))) or the plain colA→colB ramp. During a timeline
  // transition between views with DIFFERENT color modes / palette toggles
  // (colorX.x > 0), shade under the other view's mode too and crossfade —
  // enums can't lerp, shaded colors can (no more midpoint snap).
  var albedo: vec3f;
  if (G.scene.x > 0u) {
    albedo = sceneTint(p);
  } else {
    albedo = albedoFor(G.palA.w > 0.5, mixT);
    if (G.colorX.x > 0.0) {
      let mixB = mixTFor(G.colorX.y, p, nrm.z);
      albedo = mix(albedo, albedoFor(G.colorX.z > 0.5, mixB), clamp(G.colorX.x, 0.0, 1.0));
    }
  }
  albedo = s2l(albedo); // sRGB→linear (issue #6)

  // P1 GGX material. gloss remaps to 1−roughness so saved colorings keep their
  // meaning; specAmt gates the lobe by max(gloss, metallic) so the gloss-0,
  // metal-0 default stays spec-free exactly like the old Blinn-Phong path.
  let rough = clamp(1.0 - G.lprm.z, 0.05, 1.0);
  let metal = clamp(G.mat.x, 0.0, 1.0);
  let f0 = mix(vec3f(0.04), albedo, metal);
  let kd = 1.0 - metal;
  let specAmt = clamp(max(G.lprm.z, metal), 0.0, 1.0);
  let keyC = s2l(G.lightC.rgb);

  // Key light: shadowed diffuse + GGX; ambient rides real AO.
  let ambT = amb * (0.3 + 0.7 * occ); // AO eases ambient to a 30% floor, not black
  // P3 IBL: tint the ambient by the environment seen along the normal —
  // HUE-only (normalized by luminance) so overall brightness stays put and a
  // dark bg doesn't kill the ambient term.
  var ambC = vec3f(ambT);
  if (G.env.w > 0.0) {
    let e = envColor(nrm);
    let elum = max(dot(e, vec3f(0.2126, 0.7152, 0.0722)), 1e-3);
    ambC = ambT * mix(vec3f(1.0), e / elum, clamp(G.env.w, 0.0, 1.0));
  }
  // Key energy 1.25: pre-HDR the render could never exceed the albedo (the
  // brightest matte pixel was exactly 1.0 × paint at perfect incidence), which
  // is why everything read dark. The P0 shoulder rolls off the overs, so the
  // key can finally push like a real light.
  var col = albedo * kd * (ambC + 1.25 * (1.0 - amb) * diff * sh * keyC) * cav;
  col = col + ggxSpec(nrm, -rd, lightDir, rough, f0) * keyC * (1.25 * sh * occ * specAmt);
  // Fill + back lights (shadowless by design — that's what fill means).
  if (G.light2.w > 0.0) {
    let fd = normalize(G.light2.xyz);
    let fc = s2l(G.light2c.rgb) * G.light2.w;
    col = col + albedo * kd * max(dot(nrm, fd), 0.0) * fc * cav;
    col = col + ggxSpec(nrm, -rd, fd, rough, f0) * fc * (occ * specAmt);
  }
  if (G.light3.w > 0.0) {
    let bd = normalize(G.light3.xyz);
    let bc = s2l(G.light3c.rgb) * G.light3.w;
    col = col + albedo * kd * max(dot(nrm, bd), 0.0) * bc * cav;
    col = col + ggxSpec(nrm, -rd, bd, rough, f0) * bc * (occ * specAmt);
  }
  col = col + s2l(G.colB.rgb) * (rim * G.lprm.y * occ);
  // P3 env reflections: the GGX env term samples the sky along the reflection
  // vector, softened toward the sky's mean as roughness rises (an analytic env
  // needs no mips). Fresnel-weighted; metals finally reflect the world.
  if (G.env.x > 0.0 && specAmt > 0.0) {
    let er = mix(envColor(reflect(rd, nrm)), s2l(G.bgc.rgb) * 0.6, rough);
    let fenv = f0 + (vec3f(1.0) - f0) * pow(1.0 - max(dot(nrm, -rd), 0.0), 5.0);
    col = col + er * fenv * (G.env.x * specAmt * occ);
  }
  // Object intensity stays HERE (not in post): it has never brightened the
  // background, and moving it to a whole-frame exposure would shift every
  // saved look. The post pass owns tone map + encode + its own exposure bias.
  col = col * G.lprm.w;
  // P3 fog: 0 = the legacy 0.6·t/tFar fade EXACTLY; > 0 = exponential fog
  // toward the (sky-blended) background with a sun in-scatter boost — light
  // shafts feel without a volumetric march.
  if (G.fog.x > 0.0) {
    let f = 1.0 - exp(-3.0 * G.fog.x * t / tFar);
    let scat = pow(max(dot(rd, normalize(G.light.xyz)), 0.0), 8.0) * G.fog.y;
    col = mix(col, bgOut + s2l(G.lightC.rgb) * scat, f);
  } else {
    col = mix(col, bgOut, clamp(t / tFar, 0.0, 1.0) * 0.6); // legacy distance fade
  }
  return vec4f(max(col, vec3f(0.0)), 1.0);                // linear HDR out (P0)
}
`;
}

// ── Post pass (RENDER_QUALITY P0) ──────────────────────────────────────────
// Fullscreen pass from the rgba16float march target to the 8-bit swap chain:
// exposure bias → tone map → exact sRGB encode → dither. Binds the SAME
// Globals uniform buffer as the march pass, viewed as a raw vec4 array so this
// shader needs no struct mirror — word 24 is `post` (renderer.js gF[96..99]).
export const POST_WORD = 24; // vec4 index of `post` in the Globals buffer
export const JITTER_WORD = 31; // vec4 index of `jitter` (P2 accumulation)
export const FOG_WORD = 33; // vec4 index of `fog` (P3 — post reads z=bloom strength)
export const DOF_WORD = 34; // vec4 index of `dof` (P4 — zw = per-sample lens point)
export const GLOBALS_WORDS = 35; // total vec4 count (P3: env=32, fog=33; P4: dof=34)
export function buildPostWGSL() {
  return `
struct PG { w : array<vec4f, ${GLOBALS_WORDS}> };
@group(0) @binding(0) var<uniform> G : PG;
@group(0) @binding(1) var hdr : texture_2d<f32>;
@group(0) @binding(2) var bloomTex : texture_2d<f32>; // P3 half-res blurred brights (zero-init when off)
@group(0) @binding(3) var bloomSamp : sampler;

struct VSOut { @builtin(position) clip: vec4f };
@vertex fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  var tri = array<vec2f, 3>(vec2f(-1.0,-1.0), vec2f(3.0,-1.0), vec2f(-1.0,3.0));
  var o: VSOut;
  o.clip = vec4f(tri[vi], 0.0, 1.0);
  return o;
}

// Filmic soft-shoulder: EXACT identity below the shoulder (so lit albedos
// round-trip to the picker — the #6 invariant, satisfied by construction, not
// by a mid-gray anchor), tanh rolloff to a 1.0 asymptote above it. Highlights
// (spec/rim stacks) compress instead of clipping to white.
const TONE_SHOULDER : f32 = 0.75;
fn tone1(x: f32) -> f32 {
  if (x <= TONE_SHOULDER) { return x; }
  return TONE_SHOULDER + (1.0 - TONE_SHOULDER) * tanh((x - TONE_SHOULDER) / (1.0 - TONE_SHOULDER));
}
fn tone3(c: vec3f) -> vec3f { return vec3f(tone1(c.x), tone1(c.y), tone1(c.z)); }

// linear → sRGB (exact piecewise OETF — the true inverse of the march pass's
// s2l, correct near black where a pow(1/2.2) encode bands).
fn l2s(c: vec3f) -> vec3f {
  let cc = clamp(c, vec3f(0.0), vec3f(1.0));
  let lo = cc * 12.92;
  let hi = 1.055 * pow(cc, vec3f(1.0 / 2.4)) - vec3f(0.055);
  return select(hi, lo, cc <= vec3f(0.0031308));
}

// Interleaved-gradient noise (Jimenez) — per-pixel hash, no texture, no deps.
fn ign(px: vec2f) -> f32 {
  return fract(52.9829189 * fract(dot(px, vec2f(0.06711056, 0.00583715))));
}

@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let post = G.w[${POST_WORD}];
  var c = textureLoad(hdr, vec2i(pos.xy), 0).rgb;
  // P3 bloom composite (pre-tonemap, HDR): bilinear-upsample the half-res
  // blurred brights. Strength 0 → the bloom passes never ran; the texture is
  // zero-initialized (WebGPU), so this adds exactly nothing.
  let bloomStr = G.w[${FOG_WORD}].z;
  if (bloomStr > 0.0) {
    let uv = pos.xy / G.w[0].xy;
    c = c + textureSampleLevel(bloomTex, bloomSamp, uv, 0.0).rgb * bloomStr;
  }
  c = c * exp2(post.y);                                  // exposure bias (EV)
  if (post.w > 0.0) {                                    // vignette (default 0)
    let res = G.w[0].xy;
    let d = distance(pos.xy / res, vec2f(0.5)) * 1.41421;
    c = c * (1.0 - post.w * smoothstep(0.55, 1.0, d));
  }
  if (post.x > 0.5) { c = tone3(c); }                    // filmic soft-shoulder
  var o = l2s(c);
  o = o + vec3f((ign(pos.xy) - 0.5) * (post.z / 255.0)); // dither pre-quantize
  return vec4f(o, 1.0);
}
`;
}

// ── Accumulation pass (RENDER_QUALITY P2) ──────────────────────────────────
// Running-average update: accumNext = mix(accumPrev, hdrSample, jitter.z).
// The settled frame writes with weight 1 (replaces — no clear needed, and any
// invalidation simply re-runs the settled frame); jittered refinement frames
// write 1/(N+1). rgba32float targets keep the average exact to the cap.
export function buildAccumWGSL() {
  return `
struct PG { w : array<vec4f, ${JITTER_WORD + 1}> };
@group(0) @binding(0) var<uniform> G : PG;
@group(0) @binding(1) var cur : texture_2d<f32>;
@group(0) @binding(2) var prev : texture_2d<f32>;

struct VSOut { @builtin(position) clip: vec4f };
@vertex fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  var tri = array<vec2f, 3>(vec2f(-1.0,-1.0), vec2f(3.0,-1.0), vec2f(-1.0,3.0));
  var o: VSOut;
  o.clip = vec4f(tri[vi], 0.0, 1.0);
  return o;
}

@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let px = vec2i(pos.xy);
  let w = G.w[${JITTER_WORD}].z;
  let c = textureLoad(cur, px, 0);
  let p = textureLoad(prev, px, 0);
  return mix(p, c, clamp(w, 0.0, 1.0));
}
`;
}

// ── Bloom passes (RENDER_QUALITY P3) ───────────────────────────────────────
// Three tiny fullscreen passes at HALF resolution: fs_bright thresholds +
// downsamples the resolved HDR (2×2 box), fs_blurH / fs_blurV run a 9-tap
// separable Gaussian. The post pass composites the result pre-tonemap.
// All passes are skipped entirely when bloom strength is 0.
export function buildBloomWGSL() {
  return `
struct PG { w : array<vec4f, ${GLOBALS_WORDS}> };
@group(0) @binding(0) var<uniform> G : PG;
@group(0) @binding(1) var src : texture_2d<f32>;

struct VSOut { @builtin(position) clip: vec4f };
@vertex fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  var tri = array<vec2f, 3>(vec2f(-1.0,-1.0), vec2f(3.0,-1.0), vec2f(-1.0,3.0));
  var o: VSOut;
  o.clip = vec4f(tri[vi], 0.0, 1.0);
  return o;
}

@fragment fn fs_bright(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let thr = G.w[${FOG_WORD}].w;
  let p = vec2i(pos.xy) * 2;
  var c = (textureLoad(src, p, 0).rgb + textureLoad(src, p + vec2i(1, 0), 0).rgb
         + textureLoad(src, p + vec2i(0, 1), 0).rgb + textureLoad(src, p + vec2i(1, 1), 0).rgb) * 0.25;
  return vec4f(max(c - vec3f(thr), vec3f(0.0)), 1.0);
}

fn blur(px: vec2i, axis: vec2i) -> vec3f {
  // Local var (NOT module-scope const): runtime indexing of a const array is
  // a WGSL validation error that only surfaces when the pipeline is USED —
  // it silently killed every pass in the submit (black canvas).
  var w9 = array<f32, 5>(0.2270270270, 0.1945945946, 0.1216216216, 0.0540540541, 0.0162162162);
  var c = textureLoad(src, px, 0).rgb * w9[0];
  for (var i = 1; i <= 4; i = i + 1) {
    c = c + textureLoad(src, px + axis * i, 0).rgb * w9[i];
    c = c + textureLoad(src, px - axis * i, 0).rgb * w9[i];
  }
  return c;
}
@fragment fn fs_blurH(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  return vec4f(blur(vec2i(pos.xy), vec2i(1, 0)), 1.0);
}
@fragment fn fs_blurV(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  return vec4f(blur(vec2i(pos.xy), vec2i(0, 1)), 1.0);
}
`;
}

// ── Native GLSL export (desktop iterateJIT_ body) ──────────────────────────
export function exportGLSL(formula) {
  const names = []; // PARAM_NAMES
  const types = []; // PARAM_TYPES
  const defs = []; // DEFAULTS
  const ranges = []; // PARAM_RANGES (min:max:step — for desktop slider bounds)
  const decls = []; // local `float pN = getGenericParam(...)`
  const body = []; // op snippets

  let slot = 0;
  const seen = new Map();
  const ops = activeOps(formula); // muted ops are omitted from the export
  for (const op of ops) {
    const def = byKey(op.key);
    const vars = def.params.map((pm, i) => {
      // unique-ify duplicate param names across repeated ops
      let nm = pm.name;
      const n = (seen.get(nm) || 0) + 1;
      seen.set(nm, n);
      if (n > 1) nm = `${nm}${n}`;
      names.push(nm);
      types.push(pm.type === "angle" ? "DoubleAngle" : "Double");
      defs.push(op.values[i]);
      // Authored slider bounds (same units as DEFAULTS — degrees for angles).
      ranges.push(`${pm.min}:${pm.max}:${pm.step}`);

      const vn = `p${slot}`;
      const get = `getGenericParam(slot, ${slot})`;
      decls.push(
        `    float ${vn} = ${pm.type === "angle" ? `radians(${get})` : get};`,
      );
      slot++;
      return vn;
    });
    // Desktop export prefers `desktopGlsl` when an op provides one — for ops
    // whose live `glsl()` body references the iteration index `i` (absent in the
    // desktop iterateJIT_ ABI), this emits a static best-effort substitute.
    body.push((def.desktopGlsl ?? def.glsl)(vars));
  }

  // Julia mode: the iteration is f(z) + jc with jc FIXED, so we bake the
  // constant into the body and turn AddC OFF (else the engine would also re-add
  // the world seed c). For a non-Julia formula AddC is left as the preset set it.
  const julia = !!formula.julia;
  const jc = formula.juliaC || [0, 0, 0];
  if (julia)
    body.push(`
    // Julia constant (baked: this formula adds a FIXED c, not the world seed)
    pos += vec3(${jc[0]}, ${jc[1]}, ${jc[2]});`);
  const effAddC = julia ? false : formula.addC;

  const safe = formula.name.replace(/[^A-Za-z0-9_]/g, "_");
  return `// HAND_CRAFTED: generated by the web formula creator (op-list export).
// JIT formula: ${formula.name} (DEscale=0.0)
// JIT_VERSION: 2
// DEFAULTS: ${defs.join(",")}
// PARAM_NAMES: ${names.join(",")}
// PARAM_TYPES: ${types.join(",")}
// PARAM_RANGES: ${ranges.join(",")}
// AddC: ${effAddC ? "true" : "false"}
// DEoption: ${effectiveDeOption(formula)}
//
// Composed from ${ops.length} primitive(s):
//   ${ops.map((o) => o.key).join(" → ")}${effAddC ? "  (+c)" : ""}${julia ? `  (Julia c = ${jc.join(", ")})` : ""}
void iterateJIT_${safe}(int slot, vec3 c, inout vec3 pos, inout float w) {
${decls.join("\n")}
${body.join("\n")}
}
`;
}

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
//                  its orbitR helper adds another copy of the op switch to
//                  mapDE_single, and that register pressure taxes EVERY
//                  formula ~7-30% (measured, Mandelbox/Mandelbulb @2400²) —
//                  so the renderer compiles an analytic-only variant for the
//                  common case and swaps in the numeric one on demand. The
//                  FD gradient walks its 4 probes through ONE orbitR call
//                  site in an unroll-resistant loop (#218) — four call sites
//                  inlined the switch 4× and pushed the numeric pipeline
//                  compile to ~9 s.
//
//   exportGLSL()   emits a native iterateJIT_ body for the desktop app — the
//                  "design in browser → render on desktop" handoff. The op-list
//                  is the interchange format; this proves it round-trips.
// ─────────────────────────────────────────────────────────────────────────

import { OPERATORS, byKey, effectiveDeOption, activeOps } from "./operators.js";
import { LEAVES } from "./leaves.js";
import { glslNum } from "./glslfmt.js";

// Shape-leaf functions + the shapeId dispatch, generated from the registry
// (leaves.js is the single source of truth — D0 §2.2). Data-driven like the op
// switch: shape changes rewrite the object buffer, never the shader.
//
// `leaves=false` emits ONLY the dispatch stub (no per-leaf bodies) — the
// leaf-free BOOT variant. The 58 leaf SDFs are ~2/3 of the march WGSL, and a
// cold-start hero is almost always a plain fractal (shapeId 0, which never
// calls leafDist). The renderer compiles this small variant at boot and builds
// the leaves-included one lazily the first time a shape/scene formula renders
// (renderer.js — mirrors the numeric-DE variant). A leaf-free shader that ever
// saw a leaf id degrades to the radial fallback, it never crashes.
// `leaves`: true = every leaf SDF; an ARRAY = only those leaf ids (perf, lever
// #3 for leaves — a scene uses 1-2 leaves, not 58, and the leaf algebraics
// (TPMS / Taubin / knots) DOMINATE pipeline compile time on D3D12: an all-leaves
// variant measured 65 s on a GTX 1080 vs ~2.5 s for a leaf-free flat one).
// false/empty = none (the dispatch degrades to the radial fallback). The
// renderer keys each scene variant by the exact leaf-id set it wrote.
const leafFnsWGSL = (leaves = true) => {
  const use =
    leaves === true
      ? LEAVES
      : Array.isArray(leaves)
        ? LEAVES.filter((l) => leaves.includes(l.id))
        : [];
  return (
    use
      .map(
        (l) => `fn leaf_${l.key}(p: vec3f, prm: vec4f) -> f32 {
  ${l.wgsl}
}`,
      )
      .join("\n") +
    `
fn leafDist(id: u32, p: vec3f, prm: vec4f) -> f32 {
${use.map((l, i) => `  ${i ? "else " : ""}if (id == ${l.id}u) { return leaf_${l.key}(p, prm); }`).join("\n")}
  return length(p); // unknown/omitted leaf id → radial fallback (degrade, don't vanish)
}`
  );
};

// Feature-gate the op-switch-heavy blocks. Each block is wrapped in the WGSL
// with `//__GATE:name:START … //__GATE:name:END` comment markers (valid WGSL, a
// no-op when the flag is on). When a flag is OFF, the marked region is replaced
// by a signature-matching STUB, so the pipeline compiles a far smaller shader.
// A stub is only ever REACHED when its runtime guard is false (surface mode
// never calls the coloring fns; a non-scene frame never calls objIterDE; …), so
// this changes what compiles, never what renders — the renderer picks the
// variant whose flags match the frame's features (renderer.js activeMarch).
const GATE_STUBS = {
  hybrid: `fn mapDE_hybrid(p0: vec3f) -> f32 { return 1.0e9; }`,
  morph1: `fn morphIter(p0: vec3f, lo: u32, hi: u32, itersN: u32, addC: bool, cIn: vec3f, deOpt: u32, bail: f32) -> f32 { return 1.0e9; }
fn morphTrap(p0: vec3f, lo: u32, hi: u32, itersN: u32, addC: bool, cIn: vec3f, bail: f32) -> f32 { return 1.0e9; }
fn morphSilk(p0: vec3f, lo: u32, hi: u32, itersN: u32, addC: bool, cIn: vec3f, bail: f32) -> f32 { return 0.0; }
fn morphPin(p0: vec3f, lo: u32, hi: u32, itersN: u32, addC: bool, cIn: vec3f, bail: f32) -> f32 { return 0.0; }`,
  morph2: `fn morphEsc(p0: vec3f, lo: u32, hi: u32, itersN: u32, addC: bool, cIn: vec3f, bail: f32) -> f32 { return 0.0; }`,
  scene1: `fn objIterDE(p0: vec3f, ob: Obj) -> f32 { return 1.0e9; }`,
  scene2: `fn objOrbitSignal(pk: vec3f, ob: Obj, mode: f32) -> f32 { return 0.0; }`,
  coloring: `fn orbitTrap(p_rel: vec3f) -> f32 { return 0.0; }
fn orbitAddress(p_rel: vec3f) -> f32 { return 0.0; }
fn orbitPainter(p_rel: vec3f) -> vec3f { return vec3f(0.0); }
fn orbitSilk(p_rel: vec3f) -> f32 { return 0.0; }
fn orbitPin(p_rel: vec3f) -> f32 { return 0.0; }
fn orbitIrid(p_rel: vec3f) -> f32 { return 0.0; }
fn escapeIter(p_rel: vec3f) -> f32 { return 0.0; }`,
};
// name → which flag keeps it (true = full body kept).
const GATE_FLAG = {
  hybrid: "hybrid",
  morph1: "morph",
  morph2: "morph",
  scene1: "scene",
  scene2: "scene",
  coloring: "coloring",
};
function applyGates(src, flags) {
  let out = src;
  for (const name of Object.keys(GATE_STUBS)) {
    if (flags[GATE_FLAG[name]]) continue; // flag on → keep the full block
    const re = new RegExp(
      `//__GATE:${name}:START[\\s\\S]*?//__GATE:${name}:END`,
    );
    if (!re.test(out))
      throw new Error(`buildWGSL: gate marker ${name} not found`);
    out = out.replace(re, GATE_STUBS[name]);
  }
  return out;
}

// ── §S2 GPU capture uniform (CaptureU) ───────────────────────────────────────
// The per-view/per-layer capture params, packed into 5×vec4f (20 f32 words) in
// the exact order the capture fragment's `CaptureU` struct reads them. Pure +
// exported so Node pins the word offsets without a GPU. `originRel` is the
// ray-plane center MINUS the recenter origin O (the deep-zoom residual,
// computed in f64 JS before this f32 pack — §S2.5). Layout:
//   0  viewDir : d.xyz            layerIndex (w)
//   4  right   : rgt.xyz          radius     (w)   ← SCALE only (eps/AO), not the window
//   8  up      : up.xyz           eps        (w)
//   12 origin  : (planeCenter−O)  tmax       (w)
//   16 knobs   : deScale, aoStrength, maxSteps, layers
//   20 vol     : ext.xyz          kind       (w)   ← capture volume (§SHAPES)
//   24 win     : hu, hv, 0, 0                      ← this view's support window
//   28 rot0    : r0.xyz                 oriented? (w: 1 = yes, 0 = identity)
//   32 rot1    : r1.xyz                 0             ← r2 = cross(r0, r1)
// The last two rows are the CAPTURE_VOLUME_SHAPES.md bump: every one of the
// original 20 words was already assigned, so there was no slot to repurpose.
// `hu`/`hv` are computed CPU-side by viewBasis (one source for the support
// math); only `inside` is mirrored in WGSL below.
export const CAPTURE_U_WORDS = 36;
export function packCaptureUniform({
  d,
  rgt,
  up,
  originRel,
  radius,
  eps,
  tmax,
  ext,
  kind = 0,
  hu,
  hv,
  rot = null,
  layerIndex = 0,
  deScale = 1,
  aoStrength = 0,
  maxSteps = 200,
  layers = 2,
}) {
  const u = new Float32Array(CAPTURE_U_WORDS);
  u[0] = d[0];
  u[1] = d[1];
  u[2] = d[2];
  u[3] = layerIndex;
  u[4] = rgt[0];
  u[5] = rgt[1];
  u[6] = rgt[2];
  u[7] = radius;
  u[8] = up[0];
  u[9] = up[1];
  u[10] = up[2];
  u[11] = eps;
  u[12] = originRel[0];
  u[13] = originRel[1];
  u[14] = originRel[2];
  u[15] = tmax;
  u[16] = deScale;
  u[17] = aoStrength;
  u[18] = maxSteps;
  u[19] = layers;
  const e = ext ?? [radius, radius, radius]; // pre-cuboid callers / probe literals
  u[20] = e[0];
  u[21] = e[1];
  u[22] = e[2];
  u[23] = kind;
  u[24] = hu ?? radius;
  u[25] = hv ?? radius;
  u[26] = 0;
  u[27] = 0;
  // Orientation (CAPTURE_VOLUME_SHAPES): the volume's first two LOCAL axes in
  // world space, already orthonormal — volBasis does the Gram-Schmidt once per
  // frame so the fragment only crosses them. Absent ⇒ identity, flagged in w so
  // the shader skips the transform entirely.
  if (rot) {
    u[28] = rot[0][0];
    u[29] = rot[0][1];
    u[30] = rot[0][2];
    u[31] = 1;
    u[32] = rot[1][0];
    u[33] = rot[1][1];
    u[34] = rot[1][2];
    u[35] = 0;
  }
  return u;
}

// ── df64 micro-library (deep zoom Phase 4 — DEEP_ZOOM_DF64.md §4a-1) ───────
// Double-float arithmetic: each value is an unevaluated f32 (hi, lo) pair
// giving ~49 mantissa bits, via the classic error-free transformations
// (Møller/Knuth TwoSum; Dekker 1971 split/TwoProd; Hida–Li–Bailey add/mul).
// Scalars ride vec2f = (hi, lo); positions ride Df3 = (hi: vec3f, lo: vec3f).
// Emitted ONLY in the df64 pipeline variant (buildWGSL({df64:true})) — the
// default build is byte-identical without it.
//
// Numerical ground rules (df64.test.mjs pins each against an
// fround-emulated JS mirror — the semantic reference; df_launder is the
// identity in exact arithmetic, so the mirror simply frounds each op):
//   - EFTs assume each +,-,* is individually rounded-to-nearest f32 with NO
//     cross-op algebraic rewriting. Compilers don't grant that by default
//     (see the df_launder block below) — hence every f32 op goes through
//     the laundered df_fadd/df_fsub/df_fmul primitives.
//   - df_split's constant 4097 = 2^12+1 (f32 Veltkamp splitter). Overflow-safe
//     for |a| < ~2^115; orbit magnitudes here are O(1).
//   - df_abs: when hi == 0 the sign lives in lo (exact fold-boundary case) —
//     a hi-only sign test misclassifies there (spec §4a-1).
// Exported for the device probe (harness/df64.html) and tests — the probe
// must exercise the EXACT shipped library text, not a copy.
//
// ⚠ THE df_launder BARRIERS ARE LOAD-BEARING — measured, not theoretical.
// On Chrome/Metal (our PRIMARY platform, M-series Mac, probed 2026-07-23
// via harness/df64.html + a compute readback) the MSL toolchain applies
// REAL-number algebra across f32 ops: naked error-free transforms come back
// with error terms exactly 0 and the df64 round trip returns garbage
// (O.lo-scale constant error). Two weaker defenses measured and REJECTED:
// (1) ×df_one uniform-fed multiplies (the deck.gl/luma.gl fp64 trick) — the
// simplified algebra still cancels structurally when the runtime value is 1;
// (2) laundering only "critical" temporaries — the optimizer merges the
// remaining naked ops (e.g. ea+eb reassociated into (a+b)−…, re-rounding
// the sum). What works, verified by readback: TOTAL isolation — every f32
// add/sub/mul inside the EFTs goes through df_launder, a bitcast→u32-add→
// bitcast round trip with a uniform-fed 0u the compiler cannot see through
// (real-number algebra cannot cross the integer domain). Cost: one integer
// add per f32 op, well inside the phase's 4-6× budget.
//
// THE CALLER MUST SET df_lz FROM A UNIFORM (a 0-valued word) before any
// df64 math — the harness arms it from its own uniform; the marcher wires
// it in plan PR-3. Left at its literal 0u initializer, the compiler folds
// the bitcasts to identity and every barrier evaporates.
// ── Perturbation deep zoom (PERTURBATION_ZOOM_IMPL.md PR-2) ──────────────
// The reference-orbit record buffer + tiny helpers. Layout is
// core/perturb.js buildOrbit's: 4 vec4 per (iteration, op) —
// [Z, kr][A3, r2][B3, mA][mB,0,0,0] — plus one trailer slot
// [finalZ, escapeAt]. Everything here and in the wgslPt twins is PLAIN f32:
// the delta maps are exact identities evaluated as small×O(1) products, so
// there are no error-free transforms for a fast-math compiler to destroy —
// no df_launder, by construction (P1 spike: kernel correct to 1e-30 on
// Metal). Injected only when buildWGSL({perturb:true}); binding 6 sits
// clear of the capture block's 3-5, so the march AND capture pipelines can
// both carry the records (capture parity, plan PR-3/D9).
export const PT_LIB_WGSL = `
@group(0) @binding(6) var<storage, read> ptRecs : array<vec4f>;
const PT_TAU2 : f32 = 1e-4;  // τ² — impl plan D3 (any τ ≫ 1e-5 is safe)
fn pt_trailer() -> u32 { return G.ctrl.x * G.ctrl.y * 4u; }
fn pt_escapeAt() -> u32 { return u32(ptRecs[pt_trailer()].w); }
fn pt_final() -> vec3f { return ptRecs[pt_trailer()].xyz; }
`;

export const DF64_LIB_WGSL = `
// ── df64 (double-float) library — deep zoom §4a ─────────────────────────────
// df_lz MUST be assigned from a uniform at entry (see shader.js) — the
// df_launder int-domain round trip is a fast-math barrier, not math. Every
// f32 op below runs through df_fadd/df_fsub/df_fmul; do NOT "simplify" one
// back to a naked operator (measured: the Metal compiler then re-associates
// through it and the error terms collapse to zero).
var<private> df_lz : u32 = 0u;
fn df_launder(x: f32) -> f32 { return bitcast<f32>(bitcast<u32>(x) + df_lz); }
fn df_fadd(a: f32, b: f32) -> f32 { return df_launder(a + b); }
fn df_fsub(a: f32, b: f32) -> f32 { return df_launder(a - b); }
fn df_fmul(a: f32, b: f32) -> f32 { return df_launder(a * b); }
fn two_sum(a: f32, b: f32) -> vec2f {
  let s = df_fadd(a, b);
  let bb = df_fsub(s, a);
  let ea = df_fsub(a, df_fsub(s, bb));
  let eb = df_fsub(b, bb);
  return vec2f(s, df_fadd(ea, eb));
}
fn quick_two_sum(a: f32, b: f32) -> vec2f { // requires |a| >= |b|
  let s = df_fadd(a, b);
  return vec2f(s, df_fsub(b, df_fsub(s, a)));
}
fn df_split(a: f32) -> vec2f {
  let t = df_fmul(a, 4097.0);
  let hi = df_fsub(t, df_fsub(t, a));
  return vec2f(hi, df_fsub(a, hi));
}
fn two_prod(a: f32, b: f32) -> vec2f {
  let p = df_fmul(a, b);
  let aa = df_split(a);
  let bb = df_split(b);
  let e1 = df_fsub(df_fmul(aa.x, bb.x), p);
  let e2 = df_fadd(e1, df_fmul(aa.x, bb.y));
  let e3 = df_fadd(e2, df_fmul(aa.y, bb.x));
  return vec2f(p, df_fadd(e3, df_fmul(aa.y, bb.y)));
}
fn df_add(a: vec2f, b: vec2f) -> vec2f {
  let s = two_sum(a.x, b.x);
  let t = two_sum(a.y, b.y);
  let s2 = quick_two_sum(s.x, df_fadd(s.y, t.x));
  return quick_two_sum(s2.x, df_fadd(s2.y, t.y));
}
fn df_neg(a: vec2f) -> vec2f { return vec2f(-a.x, -a.y); }
fn df_sub(a: vec2f, b: vec2f) -> vec2f { return df_add(a, df_neg(b)); }
fn df_add_f32(a: vec2f, b: f32) -> vec2f {
  let s = two_sum(a.x, b);
  return quick_two_sum(s.x, df_fadd(s.y, a.y));
}
fn df_mul(a: vec2f, b: vec2f) -> vec2f {
  let p = two_prod(a.x, b.x);
  let c1 = df_fmul(a.x, b.y);
  let c2 = df_fmul(a.y, b.x);
  let e = df_fadd(p.y, df_fadd(c1, c2));
  return quick_two_sum(p.x, e);
}
fn df_mul_f32(a: vec2f, b: f32) -> vec2f {
  let p = two_prod(a.x, b);
  let e = df_fadd(p.y, df_fmul(a.y, b));
  return quick_two_sum(p.x, e);
}
fn df_abs(a: vec2f) -> vec2f {
  var neg = a.x < 0.0;
  if (a.x == 0.0) { neg = a.y < 0.0; } // hi==0: the sign lives in lo
  return select(a, df_neg(a), neg);
}
fn df_lt(a: vec2f, b: vec2f) -> bool { // fold decisions in df64 (spec §4a-4)
  let d = df_sub(a, b);
  return d.x < 0.0 || (d.x == 0.0 && d.y < 0.0);
}
fn df_div(a: vec2f, b: vec2f) -> vec2f {
  // Two-term long division (Hida-Li-Bailey). Needed because sphereFold's
  // fold factor k = fixedR²/r² is DATA-DEPENDENT — an f32-rounded k would
  // re-inject 2⁻²⁴-relative pixel noise into the df64 position during
  // exactly the iterations df64 exists to protect (unlike the rotates'
  // constant sin/cos, whose rounding is uniform across pixels).
  let q1 = df_launder(a.x / b.x);
  let r = df_sub(a, df_mul_f32(b, q1));
  let q2 = df_launder((r.x + r.y) / b.x);
  return quick_two_sum(q1, q2);
}
fn df_to_f32(a: vec2f) -> f32 { return a.x + a.y; }

// vec3 layer — positions as component-parallel (hi, lo) triples.
struct Df3 { hi: vec3f, lo: vec3f }
fn df3_get(a: Df3, i: i32) -> vec2f {
  return vec2f(a.hi[i], a.lo[i]);
}
fn df3_two_sum(a: vec3f, b: vec3f) -> Df3 {
  let x = two_sum(a.x, b.x);
  let y = two_sum(a.y, b.y);
  let z = two_sum(a.z, b.z);
  return Df3(vec3f(x.x, y.x, z.x), vec3f(x.y, y.y, z.y));
}
fn df3_add(a: Df3, b: Df3) -> Df3 {
  let x = df_add(df3_get(a, 0), df3_get(b, 0));
  let y = df_add(df3_get(a, 1), df3_get(b, 1));
  let z = df_add(df3_get(a, 2), df3_get(b, 2));
  return Df3(vec3f(x.x, y.x, z.x), vec3f(x.y, y.y, z.y));
}
fn df3_add_f32(a: Df3, b: vec3f) -> Df3 {
  let x = df_add_f32(df3_get(a, 0), b.x);
  let y = df_add_f32(df3_get(a, 1), b.y);
  let z = df_add_f32(df3_get(a, 2), b.z);
  return Df3(vec3f(x.x, y.x, z.x), vec3f(x.y, y.y, z.y));
}
fn df3_mul_f32(a: Df3, s: f32) -> Df3 {
  let x = df_mul_f32(df3_get(a, 0), s);
  let y = df_mul_f32(df3_get(a, 1), s);
  let z = df_mul_f32(df3_get(a, 2), s);
  return Df3(vec3f(x.x, y.x, z.x), vec3f(x.y, y.y, z.y));
}
fn df3_mul(a: Df3, s: vec2f) -> Df3 { // df64 scalar (sphereFold's df_div k)
  let x = df_mul(df3_get(a, 0), s);
  let y = df_mul(df3_get(a, 1), s);
  let z = df_mul(df3_get(a, 2), s);
  return Df3(vec3f(x.x, y.x, z.x), vec3f(x.y, y.y, z.y));
}
fn df3_dot(a: Df3, b: Df3) -> vec2f {
  let px = df_mul(df3_get(a, 0), df3_get(b, 0));
  let py = df_mul(df3_get(a, 1), df3_get(b, 1));
  let pz = df_mul(df3_get(a, 2), df3_get(b, 2));
  return df_add(df_add(px, py), pz);
}
fn df3_to_f32(a: Df3) -> vec3f { return a.hi + a.lo; }
`;

// ── N-slot hybrid schedule/slice walk — the ONE shared WGSL helper (§2.3) ─────
// Replaces the eight hand-copied schedule/slice walks (mapDE_hybrid + the seven
// coloring orbit fns); GL's hybStep() is the shape precedent. Decodes the packed
// G.hyb word (hybridmodel.packHyb is the JS mirror) and returns, for iteration i,
// the op-buffer slice [lo,hi) to run and whether +c fires. WGSL has no closures,
// so it returns a struct keyed by the iteration index. Emitted ungated (before
// both the hybrid and coloring gates) so either feature alone reaches it.
//   • slotCount==0 (flat, non-hybrid): the whole op buffer [0,G.ctrl.y) with the
//     flat +c gate G.ctrl.z (julia already folded in JS-side) — so the coloring
//     fns call this unconditionally.
//   • hybrid: prefix-sum per-slot op counts for the slice; the active slot is the
//     one whose schedule window holds i % period; +c = that slot's addC || julia.
export const HYB_WALK_WGSL = `struct HybWalk { lo: u32, hi: u32, addC: bool }
fn hybOpCount(s: u32) -> u32 {
  let word = select(G.hyb.x, G.hyb.w, s >= 4u); // opCounts[0..3] in x, [4..7] in w
  return (word >> ((s & 3u) * 8u)) & 0xFFu;
}
fn hybSchedCount(s: u32) -> u32 {
  return max((G.hyb.y >> (s * 4u)) & 0xFu, 1u); // nibble; floor at 1 like parseHybrid
}
fn hybWalk(i: u32) -> HybWalk {
  let slotCount = G.hyb.z & 0xFu;
  if (slotCount == 0u) { return HybWalk(0u, G.ctrl.y, G.ctrl.z != 0u); }
  var period: u32 = 0u;
  for (var s: u32 = 0u; s < slotCount; s = s + 1u) { period = period + hybSchedCount(s); }
  period = max(period, 1u);
  let phase = i % period;
  var acc: u32 = 0u; // schedule prefix sum
  var off: u32 = 0u; // op-buffer prefix sum
  var lo: u32 = 0u;
  var hi: u32 = 0u;
  var addC: bool = false;
  for (var s: u32 = 0u; s < slotCount; s = s + 1u) {
    let opc = hybOpCount(s);
    let cnt = hybSchedCount(s);
    if (phase >= acc && phase < acc + cnt) {
      lo = off;
      hi = off + opc;
      addC = ((G.hyb.z & (1u << (16u + s))) != 0u) || (G.jc.w > 0.5);
    }
    acc = acc + cnt;
    off = off + opc;
  }
  return HybWalk(lo, hi, addC);
}`;

// ── WGSL interpreter ───────────────────────────────────────────────────────
// Feature flags gate the op-switch-heavy functions (each ~21 KB — the switch is
// inlined ~16× and is 75% of the shader). When a flag is off, applyGates swaps
// the marked block (//__GATE:name) for a tiny stub, so the BOOT variant is small
// and fast to compile; richer variants are built lazily on first use (renderer
// .js — the numeric-variant pattern generalized). All default true so the GLSL
// export path, tests, and any other caller get the full shader unchanged.
export function buildWGSL({
  numericDE = true,
  leaves = true,
  coloring = true,
  scene = true,
  hybrid = true,
  morph = true,
  ops = null,
  capture = false,
  // Deep zoom Phase 4 (DEEP_ZOOM_DF64.md): emit the df64 (double-float)
  // micro-library. PR-1 state: library only, no call sites — the df64 loop
  // bodies and op twins arrive with the variant wiring (plan PR-3). Default
  // false so every existing build is byte-identical.
  df64 = false,
  // Perturbation deep zoom (PERTURBATION_ZOOM_IMPL.md PR-2): the delta-orbit
  // variant — mapDE_single iterates the residual against the reference-orbit
  // record buffer (binding 3, core/perturb.js buildOrbit layout) in PLAIN
  // f32. Default false: every existing build is byte-identical.
  perturb = false,
} = {}) {
  // The precision tiers are alternatives, not layers (impl plan D10).
  // perturb+capture is legal since PR-3: ptRecs lives at binding 6, clear
  // of the capture block's 3-5.
  if (perturb && df64)
    throw new Error("buildWGSL: perturb and df64 are exclusive");
  // Op-switch specialization (perf, lever #3): the ~58-case switch is inlined
  // across every DE/coloring entry point and is the bulk of the pipeline
  // COMPILE time (dominant on non-M-series D3D12/HLSL stacks — measured ~7 s for
  // the boot shader on a Brave/Windows box). A formula uses only a handful of
  // ops, so `ops` (a list of the op ids it actually contains) emits ONLY those
  // cases; the renderer keys each variant by its op-set and compiles it lazily.
  // An opType never in the buffer can't be dispatched, so the dropped cases are
  // dead — the `default: {}` is never reached for a real op. `ops = null` (the
  // default) keeps the full switch for the GLSL export, tests, and any caller
  // that doesn't specialize.
  const allow = ops ? new Set(ops) : null;
  const cases = OPERATORS.filter((op) => !allow || allow.has(op.id))
    .map(
      (op) =>
        `      case ${op.id}u: {${op.wgsl}
      }`,
    )
    .join("\n");

  // ── Deep zoom P4 (DEEP_ZOOM_DF64.md, plan D2/D3) — df64 codegen blocks ──
  // casesDf: the df64 op switch. wgslDf presence IS subset membership; the
  // eligibility gate (stability.js df64Eligible) guarantees a df64 frame
  // never dispatches an op without a twin, so its default: {} stays dead.
  const casesDf = df64
    ? OPERATORS.filter((op) => op.wgslDf && (!allow || allow.has(op.id)))
        .map(
          (op) =>
            `        case ${op.id}u: {${op.wgslDf}
        }`,
        )
        .join("\n")
    : "";
  // D2 reconstruction header: p0 rebuilt as a df64 pair P = (p_rel + hi) ⊕ lo,
  // plus the IMMUTABLE copy (P0h, P0l) — the non-Julia addC source (c is p0
  // ITSELF there — the exact quantity df64 protects; adding G.jc, which is 0
  // for non-Julia, would build a structurally different fractal — TOURBILLON).
  const dfReconBlock = `// df64 reconstruction (deep zoom P4)
  var P = df3_add_f32(df3_two_sum(p_rel, G.offset.xyz), G.offsetLo.xyz);
  let P0h = P.hi; let P0l = P.lo;
  let p0 = P.hi + P.lo;
  let kStar_ = u32(G.offsetLo.w);`;
  // Under perturb the sites keep their p0 recon line (signals + morph guards
  // read p0) and gain the delta state (PR-3); byte-identical otherwise.
  const dfRecon = (orig) =>
    df64 ? dfReconBlock : perturb ? orig + "\n  " + ptStateBlock : orig;
  // D2 switch wrapper: first kStar_ iterations run the df64 twins on (P, w)
  // and refresh the f32 mirror `pos`; afterwards the untouched f32 switch
  // continues from the rounded value. kStar_ = 0 degenerates to today's loop.
  const dfSwitch = df64
    ? `if (i < kStar_) {
        switch op.opType {
${casesDf}
          default: {}
        }
        pos = P.hi + P.lo;
      } else {
        switch op.opType {
${cases}
          default: {}
        }
      }`
    : `switch op.opType {
${cases}
        default: {}
      }`;
  // D2 addC companion: while in the df64 segment the per-iteration c add must
  // target (P) — Julia c is an exact f32 constant; non-Julia c is p0 (P0).
  const dfAddC = df64
    ? ` if (i < kStar_) {
      if (G.jc.w > 0.5) { P = df3_add_f32(P, G.jc.xyz); }
      else { P = df3_add(P, Df3(P0h, P0l)); }
      pos = P.hi + P.lo;
    }`
    : "";

  // ── Perturbation codegen blocks (PERTURBATION_ZOOM_IMPL.md PR-2) ────────
  // casesPt: the delta-op switch. wgslPt presence IS subset membership (the
  // wgslDf convention); the eligibility gate (ptEligible, plan PR-4)
  // guarantees a pt frame never dispatches an untwinned op.
  const casesPt = perturb
    ? OPERATORS.filter((op) => op.wgslPt && (!allow || allow.has(op.id)))
        .map(
          (op) =>
            `          case ${op.id}u: {${op.wgslPt}
          }`,
        )
        .join("\n")
    : "";
  // D4 header: p_rel IS the delta seed — no reconstruction while tracking.
  // p0 still exists for the numericDE block and the switched tail's c (the
  // production f32 sample point — exactly what the tail should add).
  // ptStateBlock is the per-function delta state, shared by mapDE_single and
  // the 7 orbit-signal sites (PR-3) — their own p0 recon lines stay as-is.
  const ptStateBlock = `var ptD = p_rel;
  let ptDc = p_rel;
  var ptTrk = true;
  var ptRi = 0u;`;
  const ptReconBlock = `// perturbation header (impl plan D4): the residual is the seed
  let p0 = p_rel + G.offset.xyz;
  ${ptStateBlock}`;
  // The per-op wrapper: while tracking, check the escape-forcing + τ
  // switchover, then run the delta twin on (ptD, w) against this slot's
  // records; after a switchover the untouched f32 switch continues on `pos`.
  const ptSwitch = `if (ptTrk) {
        let zt = ptRecs[ptRi * 4u].xyz;
        if (i >= pt_escapeAt() || dot(ptD, ptD) > PT_TAU2 * max(1.0, dot(zt, zt))) {
          pos = zt + ptD; ptTrk = false;
        }
      }
      if (ptTrk) {
        let pv0 = ptRecs[ptRi * 4u];
        let pv1 = ptRecs[ptRi * 4u + 1u];
        let pv2 = ptRecs[ptRi * 4u + 2u];
        let pv3 = ptRecs[ptRi * 4u + 3u];
        switch op.opType {
${casesPt}
          default: {}
        }
      } else {
        switch op.opType {
${cases}
          default: {}
        }
      }
      ptRi = ptRi + 1u;`;
  // ONE switch expression per loop site keeps the template line unchanged:
  // perturb → ptSwitch, else the df64/plain dfSwitch — byte-identical when
  // perturb is off.
  const iterSwitch = perturb ? ptSwitch : dfSwitch;
  // addC while tracking: Julia δc = 0 (the constant cancels exactly);
  // non-Julia δc = δ₀. The switched tail adds the production f32 c.
  const addCBody = perturb
    ? ` if (ptTrk) { if (G.jc.w <= 0.5) { ptD = ptD + ptDc; } }
      else { pos = pos + c; }`
    : ` pos = pos + c;${dfAddC} `;
  // PR-3, the orbit-signal sites: while tracking, refresh the f32 mirror
  // `pos` once per iteration from the post-add reference (slot (i+1, 0), or
  // the trailer after the last iteration — ptRi has already advanced), so
  // every per-iteration signal read (trap min, silk stripe, pin angle, …)
  // and the sites' untouched bailout line see the reconstructed sample.
  // O(1) magnitudes — f32-relative precision is exactly what signals need.
  const ptPosSync = perturb
    ? `
    if (ptTrk) { pos = ptRecs[ptRi * 4u].xyz + ptD; }`
    : "";
  // Bailout mirrors makeOrbit on the reconstructed position: ptRi has
  // advanced past this iteration's ops, so slot (i+1, 0) — or the trailer
  // on the last iteration — carries the Z the sample sits against.
  const bailLine = perturb
    ? `if (ptTrk) {
      let zn = ptRecs[ptRi * 4u].xyz;
      let pp = zn + ptD;
      if (dot(pp, pp) > G.prm.x) { pos = pp; ptTrk = false; break; }
    } else if (dot(pos, pos) > G.prm.x) { break; }`
    : `if (dot(pos, pos) > G.prm.x) { break; }`;
  const ptFinalize = perturb
    ? `if (ptTrk) { pos = pt_final() + ptD; }
  `
    : "";

  // §S2 GPU capture fragment (docs/planning/UE_SPLAT_S2_IMPL.md §2). Emitted
  // ONLY when capture:true (the live shader is byte-identical otherwise). It
  // re-implements captureView's march VERBATIM in WGSL (NOT the live fs march):
  // unsigned |dd| sphere-trace (S1c), capture eps (fixed, not t-scaled), the
  // formula's own deScale (C.knobs.x), 6-tap ∇DE normals, 5-tap AO (S1d),
  // surfaceAlbedo (§2.1), one peel layer per draw (§2.4). Marches in p_rel
  // space; posT stores the residual, readback adds fround(O) (§2.5). MRT: 3
  // targets = 32 B (rgba32f + rgba16f + rgba16f), the core limit (§2.6).
  const captureBlock = capture
    ? `
// ── §S2 GPU capture fragment ─────────────────────────────────────────────────
struct CaptureU {
  viewDir : vec4f,   // xyz = ray dir d (unit) ; w = layerIndex
  right   : vec4f,   // xyz = right basis      ; w = radius (SCALE only: eps/AO)
  up      : vec4f,   // xyz = up basis         ; w = eps (fixed capture eps)
  origin  : vec4f,   // xyz = planeCenter − O (residual) ; w = tmax
  knobs   : vec4f,   // deScale, aoStrength, maxSteps, layers
  vol     : vec4f,   // xyz = capture volume ext ; w = kind (0 = box)
  win     : vec4f,   // hu, hv (this view's support window), 0, 0
  rot0    : vec4f,   // volume's local x-axis in world space ; w = 1 when oriented
  rot1    : vec4f,   // volume's local y-axis  (local z = cross(rot0, rot1))
};
struct CaptureOut {
  @location(0) posT : vec4f,   // p_rel.xyz, t   (t < 0 ⇒ miss)
  @location(1) aux  : vec4f,   // normal.xyz, remaining budget  (normal 0 ⇒ drop)
  @location(2) alb  : vec4f,   // display-sRGB albedo.xyz, 1
};
@group(0) @binding(3) var<uniform> C : CaptureU;
@group(0) @binding(4) var prevPosT : texture_2d<f32>;   // layer ℓ−1 (dummy 1×1 at ℓ=0)
@group(0) @binding(5) var prevAux  : texture_2d<f32>;

// Is q (RELATIVE to the frame centre) inside the capture volume? The WGSL twin
// of capturevolume.js volInside — keep the two in step. Reserved kinds fall back
// to the box test, so an unrecognised kind can never silently capture nothing.
// Express a world vector in the volume's local frame — the WGSL twin of
// capturevolume.js volToLocal. Points and directions alike: the basis is
// orthonormal, so it preserves both lengths and the ray's t parameter.
fn volLocal(p: vec3f) -> vec3f {
  if (C.rot0.w < 0.5) { return p; }          // identity — the common case, free
  let r0 = C.rot0.xyz;
  let r1 = C.rot1.xyz;
  return vec3f(dot(p, r0), dot(p, r1), dot(p, cross(r0, r1)));
}

fn captureInside(qw: vec3f) -> bool {
  let q = volLocal(qw);
  let e = C.vol.xyz;
  let s = 1.0 + 1e-6;      // f32 slack: a hit exactly ON a face must survive
  if (C.vol.w > 0.5 && C.vol.w < 1.5) {          // ellipsoid
    let u = q / e;
    return dot(u, u) <= s;
  }
  if (C.vol.w > 1.5) {                            // cylinder (axis z)
    let u = q.xy / e.xy;
    return dot(u, u) <= s && abs(q.z) <= e.z * s;
  }
  return all(abs(q) <= e * s);                    // box
}

// The ray's inside-interval, the WGSL twin of capturevolume.js volRayInterval
// (#450). Returns (t0, t1); t1 < t0 means the ray misses the volume. Clipping
// the march to it is what keeps a ray from spending its whole step budget
// crawling 3·eps at a time through solid material that sits OUTSIDE the volume
// the user framed. Convex ⇒ a single interval, so nothing inside is skipped.
fn captureSpan(ow: vec3f, dw: vec3f) -> vec2f {
  let o = volLocal(ow);
  let d = volLocal(dw);
  let e = C.vol.xyz;
  let kind = C.vol.w;
  var t0 = -1e30;
  var t1 = 1e30;
  let miss = vec2f(1.0, -1.0);

  if (kind > 0.5) {                               // ellipsoid / cylinder body
    let n = select(2, 3, kind < 1.5);             // cylinder: x,y only
    var A = 0.0; var B = 0.0; var C2 = -1.0;
    for (var i = 0; i < n; i = i + 1) {
      let u = o[i] / e[i];
      let v = d[i] / e[i];
      A = A + v * v;
      B = B + 2.0 * u * v;
      C2 = C2 + u * u;
    }
    if (A < 1e-30) {
      if (C2 > 0.0) { return miss; }
    } else {
      let disc = B * B - 4.0 * A * C2;
      if (disc < 0.0) { return miss; }
      let s = sqrt(disc);
      t0 = max(t0, (-B - s) / (2.0 * A));
      t1 = min(t1, (-B + s) / (2.0 * A));
    }
    if (kind > 1.5) {                             // cylinder's z slab
      if (abs(d.z) < 1e-12) {
        if (abs(o.z) > e.z) { return miss; }
      } else {
        let a = (-e.z - o.z) / d.z;
        let b = ( e.z - o.z) / d.z;
        t0 = max(t0, min(a, b));
        t1 = min(t1, max(a, b));
      }
    }
  } else {                                        // box — three slabs
    for (var i = 0; i < 3; i = i + 1) {
      if (abs(d[i]) < 1e-12) {
        if (abs(o[i]) > e[i]) { return miss; }
      } else {
        let a = (-e[i] - o[i]) / d[i];
        let b = ( e[i] - o[i]) / d[i];
        t0 = max(t0, min(a, b));
        t1 = min(t1, max(a, b));
      }
    }
  }
  return vec2f(t0, t1);
}

// 5-tap DE ambient occlusion — the WGSL twin of splatcapture.js aoScale (S1d):
// probe outward along the normal, weight 2^−i, ao ∈ [1−aoStrength, 1]. mapDE
// reconstructs p0 = p_rel + G.offset internally, so p_rel taps are correct.
fn captureAO(p: vec3f, nrm: vec3f, eps: f32, radius: f32, aoStrength: f32) -> f32 {
  if (aoStrength <= 0.0) { return 1.0; }
  let h = max(4.0 * eps, 0.01 * radius);
  var occ = 0.0;
  var wsum = 0.0;
  for (var i = 1; i <= 5; i = i + 1) {
    let w = pow(2.0, -f32(i));
    wsum = wsum + w;
    let di = f32(i) * h;
    let dd = mapDE(p + nrm * di);
    occ = occ + w * max(0.0, 1.0 - dd / di);
  }
  return 1.0 - aoStrength * (occ / wsum);
}

@fragment fn fsCapture(@builtin(position) fragPos: vec4f, @location(0) uv: vec2f) -> CaptureOut {${df64 ? "\n  df_lz = bitcast<u32>(G.offset.w); // arm the df64 fast-math barrier (runtime 0)" : ""}
  let d = C.viewDir.xyz;
  let radius = C.right.w;
  let eps = C.up.w;
  let tmax = C.origin.w;
  let deScale = C.knobs.x;
  let aoStrength = C.knobs.y;
  let maxSteps = C.knobs.z;
  let layers = C.knobs.w;
  let layer = C.viewDir.w;

  // Ray origin = plane-center residual + screen offset. uv ∈ [0,1] (vs) →
  // ±radius, the same cell-centered grid captureView marches (splatcapture.js
  // :122-128). All coords stay in p_rel space (§2.5).
  let sx = (uv.x * 2.0 - 1.0) * C.win.x;
  let sy = (uv.y * 2.0 - 1.0) * C.win.y;
  let o = C.origin.xyz + C.right.xyz * sx + C.up.xyz * sy;

  var out : CaptureOut;
  out.posT = vec4f(0.0, 0.0, 0.0, -1.0);   // default: miss
  out.aux  = vec4f(0.0);
  out.alb  = vec4f(0.0);

  // Peel state (§2.4): layer 0 starts fresh; layer ℓ≥1 continues from ℓ−1's
  // texel (t + re-arm, carried budget). One shared budget = maxSteps·layers.
  var t = 0.0;
  var budget = maxSteps * layers;
  let px = vec2i(i32(fragPos.x), i32(fragPos.y));
  if (layer >= 0.5) {
    let prev = textureLoad(prevPosT, px, 0);
    if (prev.w < 0.0) { return out; }      // prev layer missed → miss (CPU: !hit break)
    t = prev.w + 3.0 * eps;                // S1c re-arm (splatcapture.js:179)
    budget = textureLoad(prevAux, px, 0).w;
  }

  // Unsigned |dd| sphere-trace to the next hit — crosses solid interiors (S1c).
  // Frame centre in p_rel space. planeCenter sits 1.5·hd behind it and
  // tmax = 3·hd, so the offset is exactly tmax/2 — no extra uniform needed.
  let cRel = C.origin.xyz + d * (tmax * 0.5);
  // Clip to the volume before marching a single step (#450) — see captureSpan.
  let span = captureSpan(o - cRel, d);
  let tEnter = max(0.0, span.x);
  let tEnd = min(tmax, span.y);
  if (tEnd <= tEnter) { return out; }   // this ray misses the volume entirely
  t = max(t, tEnter);
  var hit = false;
  var p = o;
  loop {
    if (budget <= 0.0 || t >= tEnd) { break; }
    budget = budget - 1.0;
    p = o + d * t;
    let dd = mapDE(p);
    if (abs(dd) < eps) {
      // A hit OUTSIDE the capture volume is transparent: re-arm past the
      // surface and keep marching in the SAME layer (splatcapture.js twin).
      if (captureInside(p - cRel)) { hit = true; break; }
      t = t + 3.0 * eps;
      continue;
    }
    t = t + max(abs(dd) * deScale, 0.5 * eps);
  }
  if (!hit) { return out; }   // miss — posT.w stays −1

  // 6-tap ∇DE central-difference normal at h = 2·eps (splatcapture.js:158-161).
  let h = 2.0 * eps;
  let gx = mapDE(p + vec3f(h, 0.0, 0.0)) - mapDE(p - vec3f(h, 0.0, 0.0));
  let gy = mapDE(p + vec3f(0.0, h, 0.0)) - mapDE(p - vec3f(0.0, h, 0.0));
  let gz = mapDE(p + vec3f(0.0, 0.0, h)) - mapDE(p - vec3f(0.0, 0.0, h));
  let g = vec3f(gx, gy, gz);
  if (length(g) < 1e-12) {
    // Degenerate normal: keep peeling (record t so ℓ+1 re-arms) but a zero
    // normal makes the readback drop this texel — the CPU skip-record-keep-peel.
    out.posT = vec4f(p, t);
    out.aux  = vec4f(0.0, 0.0, 0.0, budget);
    return out;
  }
  let nrm = g / length(g);
  let ao = captureAO(p, nrm, eps, radius, aoStrength);
  let alb = surfaceAlbedo(p, nrm, t) * ao;   // display-sRGB, pre-lighting

  out.posT = vec4f(p, t);           // p is p_rel; readback adds fround(O)
  out.aux  = vec4f(nrm, budget);
  out.alb  = vec4f(alb, 1.0);
  return out;
}
`
    : "";

  return applyGates(
    `
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
  hyb     : vec4u,   // N-slot hybrid packing (HYBRID_NSLOT_SPEC.md §2.3), laid out
                      // for the 8-slot engineered ceiling; hybridmodel.packHyb is
                      // the JS mirror, hybWalk the WGSL decode. slotCount==0 ⇒ no
                      // hybrid (flat/single path):
                      // x = opCounts[0..3] (8 bits each)
                      // y = schedule counts[0..7] (4-bit nibbles; counts 1..8)
                      // z = slotCount (bits0-3) | addC bits (bits16-23, one/slot)
                      // w = opCounts[4..7] (8 bits each; 0 until slots 5-8 exist)
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
  pstops  : array<vec4f, 8>,  // COLORING P0 — N-stop palette. Each stop is
                      // (xyz = OKLab L,a,b ; w = position ∈ [0,1]), sorted by
                      // position; up to 8. Read ONLY when pctl.x ≥ 2 (the
                      // legacy cosine/ramp path is otherwise byte-for-byte
                      // untouched). Interpolated in OKLab, decoded to sRGB
                      // once per hit — perceptually even, no muddy midpoint.
  pctl    : vec4f,   // x = stop count (0 = legacy path, no stops); y = cyclic
                      // (>0.5 wraps last→first); z = sigLo, w = sigSpan
                      // (COLORING P2 auto-levels; default 0,1 = identity).
  p3ctl   : vec4f,   // COLORING P3 — x = iridescence (Glow trap-XYZ modulation,
                      // 0 = off); y = palette phase (0..1 cyclic rotation);
                      // z,w reserved (S5 scene-orbit flags).
  offsetLo: vec4f,   // deep zoom Phase 4 (DEEP_ZOOM_DF64.md) — xyz = O's low
                      // f32 words (recenter.js splitHiLo: lo = O − fround(O)),
                      // the precision the plain offset word truncates away.
                      // w = k* (as f32; 0 = df64 disengaged — the engagement
                      // signal AND the iteration switchover count, plan D4).
                      // Unread by any non-df64 shader text: writing zeros here
                      // renders byte-identically to before the field existed.
  tile    : vec4f,   // TILED_EXPORT §2.1.3 — the off-axis sub-projection window
                      // (sx, sy, bx, by), applied as ×scale + bias on the ray-gen
                      // expression. Default (1,1,0,0): ×1.0 + 0.0 is exact in
                      // IEEE-754, so the untiled path is BIT-identical.
  tilepx  : vec4f,   // TILED_EXPORT §2.2.1(a) — (rx0, ry0, W, H) in full-frame
                      // framebuffer px, for the terms that are functions of
                      // ABSOLUTE screen position (background gradient, vignette,
                      // dither). w = 0 (default) means OFF, and the guarded
                      // blocks then change not one instruction's result.
};
struct Op { opType: u32, p0: f32, p1: f32, p2: f32 };

// CSG — one descriptor per scene object. D0 layout: 96 bytes / 24 words (u32×4
// + f32×20), unchanged stride from v1 (the leaf param block absorbed the two
// pad words — PRIMITIVE_DIFS_D0 §2.3). Byte offsets:
//   word 0 (off  0, u32×4): opStart, opCount, iters, flags
//   word 1 (off 16, f32×4): ox, oy, oz, uscale          (origin + uniform scale)
//   word 2 (off 32, f32×4): qx, qy, qz, qw              (rotation quaternion, local→world)
//   word 3 (off 48, f32×4): jcx, jcy, jcz, blendK       (per-object Julia c + smin blend)
//   word 4 (off 64, f32×4): sp0..sp3 — the leaf's shapeParams (leaves.js)
//   word 5 (off 80, f32×4): colR, colG, colB, pad       (per-object albedo, §3.8)
// flags bits: bit0 addC · bits1-2 deOption · bit3 julia · bit4 looseDE ·
//             bits5-6 combineType (0 union · 1 smooth-union · 2 subtract · 3 intersect) ·
//             bits7-10 legacy objType (WRITTEN for debugging, no longer read) ·
//             bit11 retired (was boxBase — now shapeId=1 + final mode) ·
//             bits12-19 shapeId (0 none · 1-6 launch leaves · 7+ D2) ·
//             bit20 iterShape (D3: leaf min inside the fold loop) ·
//             bits21-31 RESERVED for D2/E — spec amendment required to claim.
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
  sp0       : f32,
  sp1       : f32,
  sp2       : f32,
  sp3       : f32,
  colR      : f32,
  colG      : f32,
  colB      : f32,
  pad3      : f32,
};

@group(0) @binding(0) var<uniform> G : Globals;
@group(0) @binding(1) var<storage, read> ops : array<Op>;
@group(0) @binding(2) var<storage, read> objects : array<Obj>;
${df64 ? DF64_LIB_WGSL : ""}${perturb ? PT_LIB_WGSL : ""}
${
  numericDE
    ? `
// Orbit escape radius only — the numeric-DE probe (COVERAGE_PLAN §3 B1). The
// same loop as mapDE_single with no DE finalize; sampled 4× (center + 3 axis
// offsets) for the finite-difference gradient — through a SINGLE call site
// (see mapDE_single, #218). Emitted only in the numeric pipeline variant
// (see buildWGSL doc — the extra switch copy taxes analytic formulas via
// register pressure otherwise).
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
}`
    : ``
}

// Single-object iteration body (today's exact map). Reads the global ctrl/jc/colA.
fn mapDE_single(${df64 || perturb ? "p_rel" : "p0"}: vec3f) -> f32 {${df64 ? "\n  " + dfReconBlock : perturb ? "\n  " + ptReconBlock : ""}
${
  numericDE
    ? `
  // DEoption 3 — numeric finite-difference DE (MB3D numDiff style, for
  // W_BULB_NUMERIC maps with no analytic dr): DE = R·ln(R)/(|∇R| + 0.06),
  // falling back to the conservative 0.5·√R heuristic where the gradient
  // vanishes. ln clamps at R=1 so interior (non-escaped) points read DE→0
  // rather than negative. v1 scope: flat single formulas only (sanitize
  // rejects numeric ops in hybrid/morph/scene stacks).
  if (G.colA.w >= 2.5) {
    let eps = 1e-4 * max(1.0, length(p0));
    // ONE orbitR call site, probes walked by a loop (#218): shader backends
    // fully inline WGSL functions, so four separate call sites duplicate the
    // whole op switch 4x in the compiled kernel (~9 s pipeline compile at
    // 58 ops). The trip count is derived from a uniform — bailout is always
    // > 0, so nProbe == 4u at runtime — purely so the compiler cannot
    // constant-fold it and unroll the probes back into four inlined copies.
    var R = 0.0;
    var gv = vec3f(0.0);
    let nProbe = 3u + u32(G.prm.x > 0.0);
    for (var k: u32 = 0u; k < nProbe; k = k + 1u) {
      // Scalar/branch probe selection, not a runtime-indexed array: dynamic
      // local-array indexing forces the probes onto the stack and taxes the
      // hot march ~10% (measured, 1024² full-iteration draw).
      var off = vec3f(0.0);
      if (k == 1u) { off = vec3f(eps, 0.0, 0.0); }
      else if (k == 2u) { off = vec3f(0.0, eps, 0.0); }
      else if (k == 3u) { off = vec3f(0.0, 0.0, eps); }
      let r = orbitR(p0 + off);
      if (k == 0u) { R = r; }
      else if (k == 1u) { gv.x = r - R; }
      else if (k == 2u) { gv.y = r - R; }
      else { gv.z = r - R; }
    }
    let g = gv / eps;
    let gl = length(g);
    let de = R * log(max(R, 1.0)) / (gl + 0.06);
    return select(0.5 * sqrt(max(R, 0.0)), de, gl > 1e-3);
  }`
    : ``
}
  var pos = p0;
  var w = 1.0;
  let c = select(p0, G.jc.xyz, G.jc.w > 0.5);  // Julia: fixed c, else sample point
  let n = G.ctrl.y;
  for (var i: u32 = 0u; i < G.ctrl.x; i = i + 1u) {
    for (var o: u32 = 0u; o < n; o = o + 1u) {
      let op = ops[o];
      ${iterSwitch}
    }
    if (G.ctrl.z != 0u) {${addCBody}}
    ${bailLine}
  }
  ${ptFinalize}let r = length(pos);
  // DEoption 0 — escape-time DE (Mandelbulb / power): 0.5·ln(r)·r / dr, with the
  // analytic derivative dr carried in w. DEoption 2 — analytic IFS r/|w|.
  if (G.colA.w < 1.0) { return 0.5 * log(max(r, 1e-9)) * r / max(abs(w), 1e-9); }
  return r / max(abs(w), 1e-9);
}

// N-slot hybrid iteration (HYBRID_NSLOT_SPEC.md §2.3). WGSL-only interpreter (the
// GL tier bakes the equivalent): shares one (pos, w) accumulator across a
// repeating slot schedule, re-slicing the concatenated op buffer per iteration via
// the ONE shared hybWalk helper (above) — the same concat/slice trick CSG's
// objIterDE uses, just per iteration not per object. v1 DE-safety (spec §1/§3.3):
// only sound for a same-family hybrid (all slots IFS, or all escape); a mixed
// hybrid has no valid w bookkeeping (the unsafe case fails visibly, not silently).
${HYB_WALK_WGSL}
//__GATE:hybrid:START
fn mapDE_hybrid(p0: vec3f) -> f32 {
  var pos = p0;
  var w = 1.0;
  let c = select(p0, G.jc.xyz, G.jc.w > 0.5);
  for (var i: u32 = 0u; i < G.ctrl.x; i = i + 1u) {
    let hw = hybWalk(i);
    for (var o: u32 = hw.lo; o < hw.hi; o = o + 1u) {
      let op = ops[o];
      switch op.opType {
${cases}
        default: {}
      }
    }
    if (hw.addC) { pos = pos + c; }
    if (dot(pos, pos) > G.prm.x) { break; }
  }
  let r = length(pos);
  if (G.colA.w < 1.0) { return 0.5 * log(max(r, 1e-9)) * r / max(abs(w), 1e-9); }
  return r / max(abs(w), 1e-9);
}
//__GATE:hybrid:END

// Formula morph (VIDEO_EXPORT_DRAWER_V2 tier-2 spike) — one formula's full
// orbit over an op slice [lo,hi), with its own iters/addC/c/deOption. The 4th
// copy of the op switch (mapDE_single/mapDE_hybrid/objIterDE have the others);
// deliberately NOT refactored into them — additive spike, zero legacy risk.
//__GATE:morph1:START
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

// COLORING P0 — Silk (S2, stripe average coloring). The morph twin of orbitSilk:
// the mean over the orbit of 0.5 + 0.5·sin(k·θ), θ = the iterated point's angle
// in the XY plane, k = stripe frequency (colB.w). Result ∈ [0,1] — flowing
// bands that follow the fractal's internal dynamics.
fn morphSilk(p0: vec3f, lo: u32, hi: u32, itersN: u32, addC: bool, cIn: vec3f, bail: f32) -> f32 {
  let k = max(G.colB.w, 1.0);
  var pos = p0;
  var w = 1.0;
  var acc = 0.0;
  var cnt = 0.0;
  for (var i: u32 = 0u; i < itersN; i = i + 1u) {
    for (var o: u32 = lo; o < hi; o = o + 1u) {
      let op = ops[o];
      switch op.opType {
${cases}
        default: {}
      }
    }
    if (addC) { pos = pos + cIn; }
    acc = acc + 0.5 + 0.5 * sin(k * atan2(pos.y, pos.x));
    cnt = cnt + 1.0;
    if (dot(pos, pos) > bail) { break; }
  }
  return acc / max(cnt, 1.0);
}

// COLORING P2 — Pinwheel (S3, trap-angle). The morph twin of orbitPin: records
// θ = atan2(y, x) of the orbit point at the CLOSEST approach to the origin
// (argmin of the trap), mapped to [0,1). Cyclic by construction — pairs with a
// cyclic palette for seam-free sectors radiating from fold centers.
fn morphPin(p0: vec3f, lo: u32, hi: u32, itersN: u32, addC: bool, cIn: vec3f, bail: f32) -> f32 {
  var pos = p0;
  var w = 1.0;
  var tr = 1.0e9;
  var ang = 0.0;
  for (var i: u32 = 0u; i < itersN; i = i + 1u) {
    for (var o: u32 = lo; o < hi; o = o + 1u) {
      let op = ops[o];
      switch op.opType {
${cases}
        default: {}
      }
    }
    if (addC) { pos = pos + cIn; }
    let r2 = dot(pos, pos);
    if (r2 < tr) { tr = r2; ang = atan2(pos.y, pos.x); }
    if (r2 > bail) { break; }
  }
  return fract(ang * 0.15915494 + 0.5); // 1/(2π); [-π,π] → [0,1)
}
//__GATE:morph1:END

// COLORING P0 — smooth escape fraction (S1 bands, #239 D6). Adds a fractional
// offset to the integer escape count so the "bands" mode stops stair-stepping.
// It is a HEURISTIC: a fractbox formula is an arbitrary op-list with no single
// "power", so the classic smooth-iteration divisor log(power) has no value here
// — log2 is a CHOSEN constant. rEsc is |pos| at the escaping iteration, bailSq
// the squared-radius bailout (so rBail = sqrt). Guarded: needs rBail>1 AND
// rEsc>1, else 0 — a bailout < 1 flips the log-ratio sign to NaN, which clamp
// does NOT sanitize in WGSL (#239 finding 8). Range stays [0,1).
fn smoothEscFrac(rEsc: f32, bailSq: f32) -> f32 {
  let rBail = sqrt(bailSq);
  if (rBail <= 1.0 || rEsc <= 1.0) { return 0.0; }
  return clamp(1.0 - log2(log(rEsc) / log(rBail)), 0.0, 1.0);
}

//__GATE:morph2:START
fn morphEsc(p0: vec3f, lo: u32, hi: u32, itersN: u32, addC: bool, cIn: vec3f, bail: f32) -> f32 {
  var pos = p0;
  var w = 1.0;
  var esc: u32 = itersN;
  var rEsc = 0.0;
  for (var i: u32 = 0u; i < itersN; i = i + 1u) {
    for (var o: u32 = lo; o < hi; o = o + 1u) {
      let op = ops[o];
      switch op.opType {
${cases}
        default: {}
      }
    }
    if (addC) { pos = pos + cIn; }
    if (dot(pos, pos) > bail) { esc = i; rEsc = length(pos); break; }
  }
  return (f32(esc) + smoothEscFrac(rEsc, bail)) / f32(max(itersN, 1u));
}
//__GATE:morph2:END

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

// Shape-leaf SDFs + shapeId dispatch, generated from leaves.js (D0 §2.2).
${leafFnsWGSL(leaves)}

// CSG D0 — one scene object's DE: op chain + shape leaf, one unified path
// (PRIMITIVE_DIFS_D0 §2.1). Pure leaves arrive as a 1-iteration empty loop
// (normalizeSceneObject), so there is no separate primitive branch. Julia
// (flags bit3): c = the object's own juliaC constant instead of the sample
// point. iterShape (bit20, D3): sample the leaf after every full iteration
// (ops + addC, before the bail check) and keep the min — each term is the
// finalize bound at that iteration, and a min of lower bounds is a bound.
//__GATE:scene1:START
fn objIterDE(p0: vec3f, ob: Obj) -> f32 {
  var pos = p0;
  var w = 1.0;
  let julia = (ob.flags & 8u) != 0u;
  let c = select(p0, vec3f(ob.jcx, ob.jcy, ob.jcz), julia);
  let addC = ((ob.flags & 1u) != 0u) || julia;
  let deOpt = (ob.flags >> 1u) & 3u;
  let shapeId = (ob.flags >> 12u) & 0xFFu;
  let iterShape = (ob.flags & (1u << 20u)) != 0u;
  let prm = vec4f(ob.sp0, ob.sp1, ob.sp2, ob.sp3);
  var dmin = 1.0e9;
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
    if (iterShape) { dmin = min(dmin, leafDist(shapeId, pos, prm) / max(abs(w), 1e-9)); }
    if (dot(pos, pos) > G.prm.x) { break; }
  }
  let r = length(pos);
  // Finalize — exactly one applies (spec §2.2.1): escape keeps the log-DE and
  // ignores the leaf (the boxBase gate, generalized); otherwise the leaf (min
  // or final mode); otherwise the classic radial dust.
  if (deOpt == 0u) { return 0.5 * log(max(r, 1e-9)) * r / max(abs(w), 1e-9); }
  if (shapeId > 0u) {
    if (iterShape) { return dmin; }
    return leafDist(shapeId, pos, prm) / max(abs(w), 1e-9);
  }
  return r / max(abs(w), 1e-9);
}
//__GATE:scene1:END

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
// (inverse-rotate by the conjugate quat, then /uscale) then the ONE unified
// evaluator (D0 — no type dispatch left; pure leaves are a trivial loop).
// Shared by mapDE and sceneTint so the two never drift.
fn objDist(p0: vec3f, ob: Obj) -> f32 {
  let qinv = vec4f(-ob.qx, -ob.qy, -ob.qz, ob.qw);
  let pk = qrot(qinv, p0 - vec3f(ob.ox, ob.oy, ob.oz)) / ob.uscale;
  return objIterDE(pk, ob) * ob.uscale;
}

// Scene DE = combine over objects. objectCount==0 → today's exact single map.
fn mapDE(p_rel: vec3f) -> f32 {
  let p0 = p_rel + G.offset.xyz; // deep zoom §3.2 — the ONE reconstruction point
  if (G.morphB.w != 0u) { return mapDE_morph(p0); } // formula-morph spike
  if ((G.hyb.z & 0xFu) != 0u) { return mapDE_hybrid(p0); } // N-slot hybrid (slotCount>0)
  if (G.scene.x == 0u) { return mapDE_single(${df64 || perturb ? "p_rel" : "p0"}); }
  var d = 1.0e9;
  for (var k: u32 = 0u; k < G.scene.x; k = k + 1u) {
    let ob = objects[k];
    let dk = objDist(p0, ob);
    // Combine k into the accumulated d. NOTE: subtract/intersect use (smooth-)max,
    // which is NOT a valid distance bound (it over-estimates → the marcher
    // oversteps the carved walls). The scene compensates with a tighter global
    // deScale when any object carves (see preview.js sceneDeScale CARVE_DESCALE).
    // Object 0 is the BASE: subtract/intersect need something to act on, and
    // against the empty 1e9 accumulator they are degenerate — subtract keeps
    // the 1e9 (object vanishes) and a later intersect then keeps 1e9 FOREVER
    // (whole scene = empty sky; field report 2026-08-01, sphere⊖ reordered
    // first). Union/smooth on k==0 collapse to d = dk anyway, so forcing the
    // base is byte-identical for every previously-working scene.
    let combine = select((ob.flags >> 5u) & 3u, 0u, k == 0u);
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
    let combine = select((ob.flags >> 5u) & 3u, 0u, k == 0u); // k==0 = base (see mapDE)
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

// COLORING P3 S5 — per-object scene orbit. Runs the WINNING object's own orbit
// (in its local space) at the hit, tracking all three orbit signals in one loop:
// glow (min-radius trap), bands (escape fraction), silk (stripe average). Gives
// scenes a REAL orbit signal instead of the nz/radial stand-ins. Mirrors
// objIterDE's iteration exactly (op-slice + addC/julia), so it can't drift.
//__GATE:scene2:START
fn objOrbitSignal(pk: vec3f, ob: Obj, mode: f32) -> f32 {
  var pos = pk;
  var w = 1.0;
  let julia = (ob.flags & 8u) != 0u;
  let c = select(pk, vec3f(ob.jcx, ob.jcy, ob.jcz), julia);
  let addC = ((ob.flags & 1u) != 0u) || julia;
  let lo = ob.opStart;
  let hi = ob.opStart + ob.opCount;
  let sk = max(G.colB.w, 1.0); // silk stripe frequency (colB.w)
  var tr = 1.0e9;
  var esc: u32 = ob.iters;
  var rEsc = 0.0;
  var acc = 0.0;
  var cnt = 0.0;
  for (var i: u32 = 0u; i < ob.iters; i = i + 1u) {
    for (var o: u32 = lo; o < hi; o = o + 1u) {
      let op = ops[o];
      switch op.opType {
${cases}
        default: {}
      }
    }
    if (addC) { pos = pos + c; }
    tr = min(tr, length(pos));
    acc = acc + 0.5 + 0.5 * sin(sk * atan2(pos.y, pos.x));
    cnt = cnt + 1.0;
    let r2 = dot(pos, pos);
    if (r2 > G.prm.x) { esc = i; rEsc = sqrt(r2); break; }
  }
  // COLORING R S8 — address (sign-octant of the FINAL orbit point → 8 colors).
  if (mode > 6.5) {
    let oct = f32(u32(pos.x > 0.0) + 2u * u32(pos.y > 0.0) + 4u * u32(pos.z > 0.0));
    return (oct + 0.5) / 8.0;
  }
  if (mode > 2.5) { return acc / max(cnt, 1.0); }                                    // 3 silk
  if (mode > 1.5) { return (f32(esc) + smoothEscFrac(rEsc, G.prm.x)) / f32(max(ob.iters, 1u)); } // 2 bands
  return clamp(tr / 1.5, 0.0, 1.0);                                                  // 1 glow (trap)
}
//__GATE:scene2:END

// Scene-level orbit signal: find the owning object (same combine walk as
// sceneTint), transform the hit into its local space (rigid inverse, like
// objDist), and run its orbit. One extra scene pass at the hit, like sceneTint.
fn sceneOrbit(p_rel: vec3f, mode: f32) -> f32 {
  let p0 = p_rel + G.offset.xyz;
  var d = 1.0e9;
  var win: u32 = 0u;
  for (var k: u32 = 0u; k < G.scene.x; k = k + 1u) {
    let ob = objects[k];
    let dk = objDist(p0, ob);
    let combine = select((ob.flags >> 5u) & 3u, 0u, k == 0u); // k==0 = base (see mapDE)
    if      (combine == 2u) { let nd = smaxP(d, -dk, ob.blendK); if (-dk > d) { win = k; } d = nd; }
    else if (combine == 3u) { let nd = smaxP(d,  dk, ob.blendK); if ( dk > d) { win = k; } d = nd; }
    else {
      if (dk < d) { win = k; }
      if (combine == 1u) { d = sminP(d, dk, ob.blendK); } else { d = min(d, dk); }
    }
  }
  let ob = objects[win];
  let qinv = vec4f(-ob.qx, -ob.qy, -ob.qz, ob.qw);
  let pk = qrot(qinv, p0 - vec3f(ob.ox, ob.oy, ob.oz)) / ob.uscale;
  return objOrbitSignal(pk, ob, mode);
}

// Orbit trap: the closest the iterated point came to the origin. Re-runs the
// iteration once (only at the final hit point, so it's cheap) to drive coloring.
//__GATE:coloring:START
fn orbitTrap(p_rel: vec3f) -> f32 {
  ${dfRecon(`let p0 = p_rel + G.offset.xyz; // deep zoom §3.2`)}
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
  var tr = 1.0e9;
  for (var i: u32 = 0u; i < G.ctrl.x; i = i + 1u) {
    let hw = hybWalk(i);
    for (var o: u32 = hw.lo; o < hw.hi; o = o + 1u) {
      let op = ops[o];
      ${iterSwitch}
    }
    if (hw.addC) {${addCBody}}${ptPosSync}
    tr = min(tr, length(pos));
    if (dot(pos, pos) > G.prm.x) { break; }
  }
  return tr;
}

// COLORING R S8 — IFS address (cheap proxy): the sign-octant of the FINAL orbit
// point → one of 8 categorical colors (each fold sends limbs to a distinct
// octant, so this approximates "which branch"). Mirrors orbitTrap's hybrid loop;
// returns the octant band center in [0,1). The full per-op-branch address is a
// larger metadata wave (COLORING.md S8) — this is the "seen on real presets" spike.
fn orbitAddress(p_rel: vec3f) -> f32 {
  ${dfRecon(`let p0 = p_rel + G.offset.xyz;`)}
  var pos = p0;
  var w = 1.0;
  let c = select(p0, G.jc.xyz, G.jc.w > 0.5);
  for (var i: u32 = 0u; i < G.ctrl.x; i = i + 1u) {
    let hw = hybWalk(i);
    for (var o: u32 = hw.lo; o < hw.hi; o = o + 1u) {
      let op = ops[o];
      ${iterSwitch}
    }
    if (hw.addC) {${addCBody}}${ptPosSync}
    if (dot(pos, pos) > G.prm.x) { break; }
  }
  let oct = f32(u32(pos.x > 0.0) + 2u * u32(pos.y > 0.0) + 4u * u32(pos.z > 0.0));
  return (oct + 0.5) / 8.0;
}

// COLORING R S7 — "Painter" (direct orbit traps). Instead of ONE scalar → one
// palette read at the end, blend a palette color PER ITERATION, weighted by trap
// proximity (exp(−d) → close approaches dominate). The painterly UF/MB3D look:
// many hues woven across one surface. Returns sRGB albedo DIRECTLY (bypasses the
// mixT funnel + normSig). Hybrid-schedule aware; the rare morph case uses this
// same flat path. ⚠ cost: one palette read (albedoFor) per iteration.
fn orbitPainter(p_rel: vec3f) -> vec3f {
  ${dfRecon(`let p0 = p_rel + G.offset.xyz;`)}
  var pos = p0;
  var w = 1.0;
  let c = select(p0, G.jc.xyz, G.jc.w > 0.5);
  let palOn = G.palA.w > 0.5;
  var col = vec3f(0.0);
  var wsum = 0.0;
  for (var i: u32 = 0u; i < G.ctrl.x; i = i + 1u) {
    let hw = hybWalk(i);
    for (var o: u32 = hw.lo; o < hw.hi; o = o + 1u) {
      let op = ops[o];
      ${iterSwitch}
    }
    if (hw.addC) {${addCBody}}${ptPosSync}
    let d = length(pos);
    let ti = fract(d * 0.35 + f32(i) * 0.03); // per-iteration palette coordinate
    let wt = exp(-1.5 * d);                    // trap-proximity weight
    col = col + wt * albedoFor(palOn, ti);
    wsum = wsum + wt;
    if (dot(pos, pos) > G.prm.x) { break; }
  }
  return col / max(wsum, 1e-4);
}

// COLORING P0 — Silk (S2). Stripe average coloring: the mean over the orbit of
// 0.5 + 0.5·sin(k·atan2(pos.y, pos.x)), k = stripe frequency (colB.w). Mirrors
// orbitTrap's iteration exactly (hybrid schedule + formula-morph blend) so Silk
// works on hybrids and morphs, not just flat formulas. Result ∈ [0,1].
fn orbitSilk(p_rel: vec3f) -> f32 {
  ${dfRecon(`let p0 = p_rel + G.offset.xyz; // deep zoom §3.2`)}
  if (G.morphB.w != 0u) {
    let cA = select(p0, G.jc.xyz, G.jc.w > 0.5);
    let sA = morphSilk(p0, 0u, G.ctrl.y, G.ctrl.x, G.ctrl.z != 0u, cA, G.prm.x);
    let juliaB = (G.morphB.z & 2u) != 0u;
    let addB = ((G.morphB.z & 1u) != 0u) || juliaB;
    let cB = select(p0, G.morphT.yzw, juliaB);
    let sB = morphSilk(p0, G.ctrl.y, G.ctrl.y + G.morphB.x, G.morphB.y, addB, cB,
                       bitcast<f32>(G.morphB.w));
    return mix(sA, sB, clamp(G.morphT.x, 0.0, 1.0));
  }
  let k = max(G.colB.w, 1.0);
  var pos = p0;
  var w = 1.0;
  let c = select(p0, G.jc.xyz, G.jc.w > 0.5);
  var acc = 0.0;
  var cnt = 0.0;
  for (var i: u32 = 0u; i < G.ctrl.x; i = i + 1u) {
    let hw = hybWalk(i);
    for (var o: u32 = hw.lo; o < hw.hi; o = o + 1u) {
      let op = ops[o];
      ${iterSwitch}
    }
    if (hw.addC) {${addCBody}}${ptPosSync}
    acc = acc + 0.5 + 0.5 * sin(k * atan2(pos.y, pos.x));
    cnt = cnt + 1.0;
    if (dot(pos, pos) > G.prm.x) { break; }
  }
  return acc / max(cnt, 1.0);
}

// COLORING P2 — Pinwheel (S3). Trap-angle coloring: the angle atan2(pos.y,pos.x)
// of the orbit point at its closest approach to the origin, mapped to [0,1).
// Mirrors orbitTrap's iteration exactly (hybrid schedule + formula-morph blend)
// so it works on hybrids and morphs, not just flat formulas. Cyclic.
fn orbitPin(p_rel: vec3f) -> f32 {
  ${dfRecon(`let p0 = p_rel + G.offset.xyz; // deep zoom §3.2`)}
  if (G.morphB.w != 0u) {
    let cA = select(p0, G.jc.xyz, G.jc.w > 0.5);
    let aA = morphPin(p0, 0u, G.ctrl.y, G.ctrl.x, G.ctrl.z != 0u, cA, G.prm.x);
    let juliaB = (G.morphB.z & 2u) != 0u;
    let addB = ((G.morphB.z & 1u) != 0u) || juliaB;
    let cB = select(p0, G.morphT.yzw, juliaB);
    let aB = morphPin(p0, G.ctrl.y, G.ctrl.y + G.morphB.x, G.morphB.y, addB, cB,
                      bitcast<f32>(G.morphB.w));
    return mix(aA, aB, clamp(G.morphT.x, 0.0, 1.0));
  }
  var pos = p0;
  var w = 1.0;
  let c = select(p0, G.jc.xyz, G.jc.w > 0.5);
  var tr = 1.0e9;
  var ang = 0.0;
  for (var i: u32 = 0u; i < G.ctrl.x; i = i + 1u) {
    let hw = hybWalk(i);
    for (var o: u32 = hw.lo; o < hw.hi; o = o + 1u) {
      let op = ops[o];
      ${iterSwitch}
    }
    if (hw.addC) {${addCBody}}${ptPosSync}
    let r2 = dot(pos, pos);
    if (r2 < tr) { tr = r2; ang = atan2(pos.y, pos.x); }
    if (r2 > G.prm.x) { break; }
  }
  return fract(ang * 0.15915494 + 0.5); // 1/(2π); [-π,π] → [0,1)
}

// COLORING P3 — iridescence (S6). Fragmentarium-style trap-XYZ: the per-axis
// closest approach min|x|, min|y|, min|z| over the orbit. Returns the axis
// ASYMMETRY (min|x| − min|z|) — a signed modulator that shifts the Glow palette
// phase per pixel, painting multi-hue iridescence within one surface. Uses the
// hybrid schedule like orbitTrap; the rare morph case returns 0 (no shift).
fn orbitIrid(p_rel: vec3f) -> f32 {
  ${dfRecon(`let p0 = p_rel + G.offset.xyz;`)}
  if (G.morphB.w != 0u) { return 0.0; }
  var pos = p0;
  var w = 1.0;
  let c = select(p0, G.jc.xyz, G.jc.w > 0.5);
  var mn = vec3f(1.0e9, 1.0e9, 1.0e9);
  for (var i: u32 = 0u; i < G.ctrl.x; i = i + 1u) {
    let hw = hybWalk(i);
    for (var o: u32 = hw.lo; o < hw.hi; o = o + 1u) {
      let op = ops[o];
      ${iterSwitch}
    }
    if (hw.addC) {${addCBody}}${ptPosSync}
    mn = min(mn, abs(pos));
    if (dot(pos, pos) > G.prm.x) { break; }
  }
  // Scale-invariant asymmetry ∈ [-1,1] — raw min|x|−min|z| is unbounded for
  // escape-time orbits (blows the phase shift up); the ratio bounds it and works
  // across IFS and escape formulas alike.
  return (mn.x - mn.z) / (mn.x + mn.z + 1e-6);
}

// Escape iteration fraction (for "bands" coloring): how many iterations until the
// point flies past the bailout, normalized 0..1. Re-runs the iteration once.
fn escapeIter(p_rel: vec3f) -> f32 {
  ${dfRecon(`let p0 = p_rel + G.offset.xyz; // deep zoom §3.2`)}
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
  var esc: u32 = G.ctrl.x;
  var rEsc = 0.0;
  for (var i: u32 = 0u; i < G.ctrl.x; i = i + 1u) {
    let hw = hybWalk(i);
    for (var o: u32 = hw.lo; o < hw.hi; o = o + 1u) {
      let op = ops[o];
      ${iterSwitch}
    }
    if (hw.addC) {${addCBody}}${ptPosSync}
    if (dot(pos, pos) > G.prm.x) { esc = i; rEsc = length(pos); break; }
  }
  // S1 smooth bands: integer count + a fractional offset (see smoothEscFrac).
  return (f32(esc) + smoothEscFrac(rEsc, G.prm.x)) / f32(max(G.ctrl.x, 1u));
}
//__GATE:coloring:END

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
  let g = e.xyy * mapDE(p_rel + e.xyy) +
          e.yyx * mapDE(p_rel + e.yyx) +
          e.yxy * mapDE(p_rel + e.yxy) +
          e.xxx * mapDE(p_rel + e.xxx);
  // The four tetrahedron offsets sum to 0, so where the DE field is locally flat
  // (all four taps equal — e.g. the interior of a sphere-fold inversion, or a
  // flat DF face seen edge-on) the gradient is EXACTLY (0,0,0) and normalize()
  // is NaN. A NaN normal propagates through the lighting math and the final
  // max(col,0) collapses it to PURE BLACK — a shaded-but-colorless patch on the
  // surface. Fall back to a stable normal (radial, then world-up) so the patch
  // shades and takes palette color like the rest of the surface. length > 1e-20
  // also rejects a NaN/Inf gradient (the comparison is false for NaN).
  let L = length(g);
  if (L > 1e-20) { return g / L; }
  let pl = length(p_rel);
  return select(vec3f(0.0, 0.0, 1.0), p_rel / pl, pl > 1e-6);
}

// COLORING P2 — Curvature tint (S4). The discrete Laplacian of the DE field via
// the SAME tetrahedron taps as calcNormal (Σ dᵢ = 0 cancels the gradient, so the
// residual is pure second-order). Ridges (Δd < 0) vs valleys drive mixT. Because
// mapDE dispatches to single/hybrid/morph AND CSG scenes, this is the ONE signal
// that colors scenes for real (COLORING.md S4) — no orbit plumbing.
//   ⚠ eps is DELIBERATELY ~100× the normal's: curvature is an O(e²) quantity, so
//   at the normal's tiny eps it sinks below f32 precision (the 4 taps are ~1e-2,
//   the residual ~2e-4 at ce=1e-2 vs ~1e-9 f32 noise — coarser probe is load-
//   bearing). The gain 0.15 is a chosen heuristic constant (like smooth-bands'
//   log2); tanh bounds the result to [0,1] regardless of curvature magnitude.
fn curvatureAt(p_rel: vec3f, t: f32) -> f32 {
  let ce = clamp(t * 3e-3, 1e-4, 1e-2);
  let e = vec2f(1.0, -1.0) * ce;
  let lap = mapDE(p_rel + e.xyy) + mapDE(p_rel + e.yyx)
          + mapDE(p_rel + e.yxy) + mapDE(p_rel + e.xxx) - 4.0 * mapDE(p_rel);
  return clamp(0.5 + 0.5 * tanh(lap / (ce * ce) * 0.15), 0.0, 1.0);
}

// ── Debug surface-quality overlay (issue #370, SPIRULAE_LEARNINGS Plan C3) ──
// Dev-only diagnostic (WebGPU tier only): remap a per-pixel march metric to a
// heat ramp INSTEAD of shading, so discontinuity / precision hot-spots are
// visible the way spirulae paints them red. Rides the spare Globals word
// p3ctl.z (0 = off, byte-identical) → NO pipeline variant, no codegen change.
// Gated in the UI behind ?diag / the showDiag pref (never a public control).

// Classic blue→cyan→green→yellow→red "jet" heat ramp (returned in sRGB; the
// caller s2l()s it so the post pass's encode round-trips it back to these hues).
fn heatPalette(x: f32) -> vec3f {
  let v = clamp(x, 0.0, 1.0);
  return clamp(
    vec3f(1.5 - abs(4.0 * v - 3.0), 1.5 - abs(4.0 * v - 2.0), 1.5 - abs(4.0 * v - 1.0)),
    vec3f(0.0),
    vec3f(1.0),
  );
}

// |∇DE| via the SAME tetrahedron taps as calcNormal, but KEPT unnormalized. A
// true signed-distance field has unit gradient everywhere; deviation from 1
// flags a non-conservative / discontinuous field (the DE overshoots or folds
// non-Lipschitz → staircase silhouettes, marching artifacts). The 4 offsets
// sum to 0 so the constant term cancels; |g| ≈ 4·h·|∇d| for step h. Extra 4
// mapDE calls, but ONLY on debug-mode 3 rays (dev overlay), so cost is moot.
fn deGradMag(p_rel: vec3f, t: f32) -> f32 {
  let e = vec2f(1.0, -1.0) * clamp(t * 3e-5, 1e-6, 6e-4);
  let g = e.xyy * mapDE(p_rel + e.xyy) +
          e.yyx * mapDE(p_rel + e.yyx) +
          e.yxy * mapDE(p_rel + e.yxy) +
          e.xxx * mapDE(p_rel + e.xxx);
  return length(g) / max(4.0 * e.x, 1e-20);
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
// Up is world +Z — the frame lighting adopted in #160 (31e2253, camera
// worldUp = +Z) — so the horizon sits in the world XY plane.
fn envColor(rd: vec3f) -> vec3f {
  let tg = clamp(rd.z * 0.5 + 0.5, 0.0, 1.0);
  var sky = s2l(mix(G.bgc.rgb * 0.35, G.bgc.rgb, tg));
  sky = mix(sky * (1.0 - G.env.z), sky, smoothstep(-0.35, 0.12, rd.z)); // ground dim
  let sunAmt = pow(max(dot(rd, normalize(G.light.xyz)), 0.0), 24.0);
  // Neutral (white) sun glow, NOT tinted by the light albedo (#160 item 2): it's
  // a "where is the light" indicator, so an amber key shouldn't paint the sky
  // orange. The fog in-scatter below stays light-tinted (physical sun-through-fog).
  return sky + vec3f(1.0) * (sunAmt * G.env.y);
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
fn mixTFor(mode: f32, p: vec3f, nz: f32, t: f32) -> f32 {
  if (mode > 6.5) { return orbitAddress(p); }     // 7 address (IFS sign-octant)
  if (mode > 4.5) { return curvatureAt(p, t); }   // 5 curvature (geometry Laplacian)
  if (mode > 3.5) { return orbitPin(p); }         // 4 pinwheel (trap angle)
  if (mode > 2.5) { return orbitSilk(p); }        // 3 silk (stripe average)
  if (mode > 1.5) { return escapeIter(p); }       // 2 escape bands
  if (mode > 0.5) { return clamp(orbitTrap(p) / 1.5, 0.0, 1.0); } // 1 orbit-trap glow
  return 0.5 + 0.5 * nz;                           // 0 surface
}
// COLORING P2 — auto-levels: remap the raw signal by its per-formula range
// (pctl.z = lo, pctl.w = span) so the palette spans the actual signal. lo=0,
// span=1 (surface/pinwheel/off) is the identity. Guarded: a zero span (buffer
// pre-write) returns x unchanged rather than dividing by ~0.
fn normSig(x: f32) -> f32 {
  let s = G.pctl.w;
  if (s <= 1e-4) { return x; }
  return clamp((x - G.pctl.z) / s, 0.0, 1.0);
}
// OKLab (Björn Ottosson) → sRGB. The palette lerps in OKLab (perceptually even,
// straight-line, no muddy midpoint — COLORING.md); this decodes ONE blended
// stop back to the sRGB the rest of the shader expects (albedoFor's callers
// s2l() the result, exactly as they do colA/colB). Inverse of the JS
// srgbToOklab in core/oklab.js — the two MUST stay in sync (one round-trips
// the other; core/oklab.test.mjs pins it).
fn oklabToSrgb(lab: vec3f) -> vec3f {
  let l_ = lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z;
  let m_ = lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z;
  let s_ = lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z;
  let l = l_ * l_ * l_;
  let m = m_ * m_ * m_;
  let s = s_ * s_ * s_;
  let lin = vec3f(
     4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s);
  // linear → sRGB (exact piecewise OETF — mirrors post.js l2s / oklab.js).
  let cc = clamp(lin, vec3f(0.0), vec3f(1.0));
  let lo = cc * 12.92;
  let hi = 1.055 * pow(cc, vec3f(1.0 / 2.4)) - vec3f(0.055);
  return select(hi, lo, cc <= vec3f(0.0031308));
}

// Walk the ≤8 uniform-indexed stops (sorted by position), lerp the bracketing
// pair in OKLab, decode once. Cyclic (pctl.y) wraps last→first across the seam
// so a periodic signal (pinwheel angle, cycled bands) has no discontinuity.
fn albedoStops(t: f32) -> vec3f {
  let n = i32(G.pctl.x);
  let cyclic = G.pctl.y > 0.5;
  let tt = clamp(t, 0.0, 1.0);
  let first = G.pstops[0];
  let last = G.pstops[n - 1];
  if (tt <= first.w) {
    if (cyclic) {
      let pl = last.w - 1.0; // last stop, one period back
      let f = (tt - pl) / max(first.w - pl, 1e-6);
      return oklabToSrgb(mix(last.xyz, first.xyz, clamp(f, 0.0, 1.0)));
    }
    return oklabToSrgb(first.xyz);
  }
  if (tt >= last.w) {
    if (cyclic) {
      let ph = first.w + 1.0; // first stop, one period forward
      let f = (tt - last.w) / max(ph - last.w, 1e-6);
      return oklabToSrgb(mix(last.xyz, first.xyz, clamp(f, 0.0, 1.0)));
    }
    return oklabToSrgb(last.xyz);
  }
  for (var i = 0; i < n - 1; i = i + 1) {
    let a = G.pstops[i];
    let b = G.pstops[i + 1];
    if (tt >= a.w && tt <= b.w) {
      let f = (tt - a.w) / max(b.w - a.w, 1e-6);
      return oklabToSrgb(mix(a.xyz, b.xyz, f));
    }
  }
  return oklabToSrgb(last.xyz); // unreachable (interior is bracketed) — degrade
}

fn albedoFor(palOn: bool, mixT: f32) -> vec3f {
  // COLORING P3 — palette phase (p3ctl.y): rotate the palette lookup for color
  // cycling (keyframable on the timeline). Exact identity at phase 0 (no fract,
  // so mixT = 1 still maps to the last stop / colB); fract wraps for cyclic
  // palettes → seamless flow (a 2-color ramp shows a seam, as expected).
  var t = mixT;
  if (G.p3ctl.y != 0.0) { t = fract(mixT + G.p3ctl.y); }
  if (G.pctl.x > 1.5) { return albedoStops(t); } // N-stop path (count ≥ 2)
  if (palOn) {
    return clamp(G.palA.rgb + G.palB.rgb * cos(6.2831853 * (G.palC.rgb * t + G.palD.rgb)),
                 vec3f(0.0), vec3f(1.0));
  }
  return mix(G.colA.rgb, G.colB.rgb, t);
}

// §S2 — view-independent display-sRGB surface albedo: the pre-s2l, pre-lighting
// color (spec D1). The WGSL twin of cpu.js makePointAlbedo, shared by the live
// fs and the capture fs (§S2a) so the two can never drift. mixT is recomputed
// here (was a live-fs local) so the fn is self-sufficient: G/mixTFor are a
// uniform + global fn readable anywhere, exactly as normSig reads G.pctl.
fn surfaceAlbedo(p: vec3f, nrm: vec3f, t: f32) -> vec3f {
  let mixT = mixTFor(G.prm.w, p, nrm.z, t);
  var albedo: vec3f;
  if (G.scene.x > 0u) {
    // Scene coloring (docs/design/SCENES.md §Coloring): Surface (mode 0) paints
    // each object with its own albedo; Glow/Bands respond to the palette like a
    // flat formula, keyed by the signals a scene actually has — surface angle
    // (Glow) and radial distance bands (Bands); orbit data doesn't exist here.
    if (G.prm.w > 6.5) {
      // COLORING R S8 — Address: the winning object's own orbit sign-octant.
      albedo = albedoFor(G.palA.w > 0.5, sceneOrbit(p, 7.0));
    } else if (G.prm.w > 5.5) {
      // Painter (S7) has no per-object codegen on scenes yet → per-object Glow.
      albedo = albedoFor(G.palA.w > 0.5, normSig(sceneOrbit(p, 1.0)));
    } else if (G.prm.w > 4.5) {
      albedo = albedoFor(G.palA.w > 0.5, normSig(curvatureAt(p, t))); // Curvature — geometry-space, REAL on scenes
    } else if (G.prm.w > 3.5) {
      albedo = albedoFor(G.palA.w > 0.5, fract(atan2(p.y, p.x) * 0.15915494 + 0.5)); // Pinwheel → world azimuth (cyclic)
    } else if (G.prm.w > 0.5) {
      // COLORING P3 S5 — Glow/Bands/Silk now run the winning object's REAL orbit
      // (no more nz/radial stand-ins). sceneOrbit picks the signal by mode.
      albedo = albedoFor(G.palA.w > 0.5, normSig(sceneOrbit(p, G.prm.w)));
    } else {
      albedo = sceneTint(p);
    }
  } else if (G.prm.w > 5.5 && G.prm.w < 6.5) {
    // COLORING R S7 — "Painter": per-iteration palette blend, a direct color
    // (bypasses the mixT funnel). Its own arm because it returns albedo, not mixT.
    albedo = orbitPainter(p);
  } else {
    // P3 iridescence (S6): a Glow-only palette-phase modulator. After auto-levels
    // (normSig), shift the palette lookup per pixel by the orbit's axis asymmetry
    // (fract wraps → seam-free multi-hue shimmer). Flat formulas only — a scene
    // has no single orbit for orbitIrid. Off (p3ctl.x = 0) is byte-identical.
    var m = normSig(mixT);
    if (G.prm.w > 0.5 && G.prm.w < 1.5 && G.p3ctl.x > 0.001) {
      m = fract(m + G.p3ctl.x * orbitIrid(p));
    }
    albedo = albedoFor(G.palA.w > 0.5, m);
    if (G.colorX.x > 0.0) {
      let mixB = mixTFor(G.colorX.y, p, nrm.z, t);
      albedo = mix(albedo, albedoFor(G.colorX.z > 0.5, normSig(mixB)), clamp(G.colorX.x, 0.0, 1.0));
    }
  }
  return albedo;
}

@fragment fn fs(@builtin(position) pos: vec4f, @location(0) uv: vec2f) -> @location(0) vec4f {${df64 ? "\n  df_lz = bitcast<u32>(G.offset.w); // arm the df64 fast-math barrier (runtime 0)" : ""}
  // P2: subpixel jitter (pixels → ndc is 2 units across the frame). Zero when
  // not accumulating — the expression is exact identity then.
  let ndc = uv * 2.0 - vec2f(1.0) + 2.0 * G.jitter.xy / G.res.xy;
  let aspect = G.res.x / G.res.y;
  let tanF = tan(0.5 * G.res.z);
  // TILED_EXPORT §2.1.3 — the off-axis window, ×scale + bias on the plane
  // coordinate, shared by the perspective and ortho branches below (they differ
  // only in tanF vs orthoH). Default tile = (1,1,0,0) ⇒ wx = ndc.x*aspect and
  // wy = ndc.y, the pre-tiling expressions to the last bit.
  //
  // The jitter above needs NO tile correction: its contribution to the plane
  // coordinate is 2·j/rw · (rw/rh) · (rh/H) · tanF = 2·j·tanF/H, the same value
  // the full frame produces — the window's rh/H cancels the 2/res factor exactly
  // (§2.1.5). That cancellation is why there is no second resolution uniform.
  let wx = ndc.x * aspect * G.tile.x + G.tile.z;
  let wy = ndc.y * G.tile.y + G.tile.w;
  // #441 ORTHOGRAPHIC: camFwd.w > 0 is the ortho half-height. Perspective fans
  // directions from one origin; orthographic spreads the ORIGIN across the
  // image plane and holds the direction — the dual of the same two lines. Note
  // this also keeps DETAIL: the march tests d < eps*t, so faking ortho by
  // pushing the camera back (the "long lens") coarsens eps with distance and
  // dissolves the fractal; parallel rays start at today's depth instead.
  // Every product is parenthesised on purpose — an unparenthesised multiply has
  // blacked out every scene in this repo before. And NO BACKTICKS in comments
  // here: this WGSL is a JS template literal, so a stray backtick ends the
  // string and the rest evaluates as JS (buildWGSL returned NaN, not source).
  let orthoH = G.camFwd.w;
  var rd = normalize(G.camFwd.xyz
      + (wx * tanF) * G.camRight.xyz
      + (wy * tanF) * G.camUp.xyz);
  if (orthoH > 0.0) { rd = normalize(G.camFwd.xyz); }
  // Deep zoom §3.1 — camPos carries the RESIDUAL ro_rel = eye−offset (small,
  // f32-friendly), not the absolute eye. Every p below is therefore p_rel;
  // mapDE/orbitTrap/escapeIter/sceneTint each reconstruct p_world = p_rel +
  // G.offset once, at their own entry (§3.2). For scenes offset is (0,0,0),
  // so ro is the absolute eye and this is exactly today's behavior.
  var ro = G.camPos.xyz;
  // The ortho spread is a world-space displacement, so it adds to the RESIDUAL
  // ro_rel exactly as-is — no offset reconstruction changes (verified against
  // the df64 dfReconBlock, which rebuilds p0 = p_rel + G.offset downstream).
  if (orthoH > 0.0) {
    ro = ro
      + ((wx * orthoH) * G.camRight.xyz)
      + ((wy * orthoH) * G.camUp.xyz);
  }
  // P4 thin-lens DOF: offset the eye on the lens disk and re-aim at the focus
  // plane. Per accumulation sample (zw = the sample's lens point); the base
  // frame uses the lens center, so interactive frames stay sharp and the
  // bokeh converges while idle — and offline export samples make it clean.
  // DOF is a thin-LENS model — meaningless under a parallel projection, and
  // ortho is a transient inspection mode, so the two are mutually exclusive.
  if (G.dof.x > 0.0 && orthoH <= 0.0) {
    let fp = ro + rd * G.dof.y;
    ro = ro + (G.camRight.xyz * G.dof.z + G.camUp.xyz * G.dof.w) * G.dof.x;
    rd = normalize(fp - ro);
  }

  // TILED_EXPORT §2.2.1(a) — the gradient is a function of ABSOLUTE screen
  // height, so a tile left alone restarts it and produces the most visible seam
  // of all. tilepx.w = 0 (untiled) keeps bgY = uv.y exactly.
  var bgY = uv.y;
  if (G.tilepx.w > 0.0) { bgY = 1.0 - (G.tilepx.y + pos.y) / G.tilepx.w; }
  let bg = s2l(mix(G.bgc.rgb * 0.35, G.bgc.rgb, clamp(bgY, 0.0, 1.0)));

  var t = G.res.w;
  var tPrev = t;
  var steps: u32 = 0u;
  var hit = false;
  let maxSteps = G.ctrl.w;
  let eps = G.prm.y;
  let tFar = G.camPos.w;
  var lastD = 1e9;
  var overshoot = 0.0; // debug #370 — last-step span vs the eps·t hit shell
  for (steps = 0u; steps < maxSteps; steps = steps + 1u) {
    let p = ro + rd * t;
    let d = mapDE(p) * G.prm.z;
    lastD = d;
    if (d < eps * t) { hit = true; break; }
    tPrev = t;
    t = t + d;
    if (t > tFar) { break; }
  }
  // Hit refinement — the loop stops at the first SAMPLE inside the eps·t
  // shell, so the landed t overshoots the true crossing by up to a full march
  // step; neighboring rays cross on different steps, and at grazing incidence
  // that per-ray overshoot spans many pixels on screen → terraced "staircase"
  // silhouettes on smooth surfaces (worst zoomed into sheet-like geometry,
  // where whole regions graze). Bisect [tPrev, t] to the threshold crossing:
  // 8 halvings shrink the overshoot 256× (sub-pixel at any depth the f32
  // march itself can reach) for ≤8 extra mapDE calls per HIT ray. tPrev==t ⇔
  // hit on the very first sample (eye already inside the shell) — nothing to
  // bracket; the exhausted-ray softA pseudo-hit below never refines either
  // (its hit flag is set after this).
  if (hit && t > tPrev) {
    // Debug #370 — how far the landed sample overshot the true crossing, in
    // units of the eps·t hit shell (before bisection collapses the bracket).
    // Big values ⇒ grazing / thin geometry ⇒ staircase-prone silhouettes.
    overshoot = (t - tPrev) / max(eps * t, 1e-12);
    var lo = tPrev;
    var hi = t;
    for (var r = 0u; r < 8u; r = r + 1u) {
      let mid = 0.5 * (lo + hi);
      if (mapDE(ro + rd * mid) * G.prm.z < eps * mid) { hi = mid; } else { lo = mid; }
    }
    t = hi;
  }
  // Budget exhausted while still hugging geometry (grazing rays over an
  // unbounded field burn every step crossing dune crests): shade at the last
  // position instead of dropping to sky. CONFIDENCE-WEIGHTED: a ray that
  // merely PASSED NEAR an object (silhouette graze — d small but nonzero at
  // exhaustion) must not paint phantom surface, or the horizon bulges around
  // the object; softA fades the shade toward the sky by how close the ray
  // actually got. True sky rays leave via t > tFar and stay misses.
  var softA = 1.0;
  if (!hit && steps >= maxSteps && lastD < eps * t * 8.0) {
    hit = true;
    softA = 1.0 - clamp(lastD / (eps * t * 8.0), 0.0, 1.0);
    softA = softA * softA; // bias toward sky — only true huggers stay solid
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
  // With fog on, sky rays traverse the WHOLE fog column, so they get the
  // same sun in-scatter a fully-fogged surface converges to — without it the
  // hit/miss boundary draws a hard seam wherever fogged geometry meets empty
  // sky (a fractal-edged patch of raw sky against scatter-whitened haze —
  // exposed at deep zoom once the density compensation made f → 1 reachable;
  // field report 2026-07-31 "is the top left banding intended?"). The
  // saturating (1 − e^(−3·fog)) term equals a fully-fogged hit's mix weight
  // at t = tFar, so the two paths agree exactly where they meet; at low fog
  // it stays a subtle warm tint in the sun cone.
  if (!hit) {
    var skyOut = bgOut;
    if (G.fog.x > 0.0) {
      let scatM = pow(max(dot(rd, normalize(G.light.xyz)), 0.0), 8.0) * G.fog.y;
      skyOut = skyOut + s2l(G.lightC.rgb) * scatM * (1.0 - exp(-3.0 * G.fog.x));
    } else if (G.fog.x < 0.0) {
      // Log-depth mode: the sky column spans all decades out to tFar.
      let t0M = max(length(G.camPos.xyz), 1e-12);
      let decM = log2(1.0 + tFar / t0M) * 0.30103;
      let scatM = pow(max(dot(rd, normalize(G.light.xyz)), 0.0), 8.0) * G.fog.y;
      skyOut =
        skyOut +
        s2l(G.lightC.rgb) * scatM * (1.0 - exp(-2.3 * (-G.fog.x) * decM));
    }
    // Alpha 0 on a true miss (empty background/sky) — carried through the post
    // pass untouched. Harmless for every existing consumer: the live swap-chain
    // is configured alphaMode:"opaque" (ignores alpha entirely) and
    // renderToImage's readback forces alpha back to 255 unless the caller
    // opts in (issue #428 item 2 — transparent PNG export). Hit pixels below
    // stay 1.0 (opaque) unchanged, including the confidence-faded "soft hit"
    // ones — only genuine sky rays become transparent.
    return vec4f(skyOut, 0.0);
  }

  let p = ro + rd * t;
  let nrm = calcNormal(p, t);
  // Step-count "cavity dust" — the pre-P1 AO heuristic, kept as a softened
  // multiplier (fractal artists like the look); real occlusion is calcAO below.
  let stepAO = 1.0 - f32(steps) / f32(maxSteps);
  let cav = 0.7 + 0.3 * stepAO;

  // Debug surface-quality overlay (#370, Plan C3). p3ctl.z selects the metric
  // (0 = off ⇒ this whole block is skipped and shading is byte-identical). Paints
  // a per-pixel march diagnostic as heat instead of lighting the surface — misses
  // keep the background, so heat concentrates on the geometry like spirulae's red.
  let dbg = G.p3ctl.z;
  if (dbg > 0.5) {
    var hv = 0.0;
    if (dbg < 1.5) {
      // 1 — march step-count heat: fraction of the step budget this ray burned.
      hv = f32(steps) / f32(maxSteps);
    } else if (dbg < 2.5) {
      // 2 — overshoot / bisection bracket width (see the march block above).
      hv = clamp(overshoot / 32.0, 0.0, 1.0);
    } else {
      // 3 — ∇DE instability: |∇DE| deviation from 1 (unit gradient = clean SDF).
      hv = clamp(abs(deGradMag(p, t) - 1.0), 0.0, 1.0);
    }
    return vec4f(s2l(heatPalette(hv)), 1.0);
  }

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

  // Albedo: the view-independent surface color (§S2 surfaceAlbedo — the exact
  // block that used to live inline here, now shared with the capture fs). Scenes
  // color per-object; single objects use the palette / iridescence / colorX
  // crossfade. Everything after is lighting, which capture omits.
  var albedo = surfaceAlbedo(p, nrm, t);
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
  // Far fade — the last stretch before tFar blends fully into the sky, so
  // the finite render boundary (a SPHERE around the eye) never shows: an
  // unbounded field otherwise ends in a bent-horizon arc of budget shells.
  // Bounded formulas never render this deep (tFar = dist*13+), so no change.
  let farF = max(smoothstep(0.62 * tFar, 0.95 * tFar, t), 1.0 - softA);
  if (G.fog.x > 0.0) {
    let f = max(1.0 - exp(-3.0 * G.fog.x * t / tFar), farF);
    let scat = pow(max(dot(rd, normalize(G.light.xyz)), 0.0), 8.0) * G.fog.y;
    col = mix(col, bgOut + s2l(G.lightC.rgb) * scat, f);
  } else if (G.fog.x < 0.0) {
    // Deep-zoom LOG-DEPTH fog (fog.x < 0 is the mode flag; |x| = the raw
    // slider — see frameparams.js). A deep frame spans decades of depth, so
    // optical depth accumulates per DECADE beyond the near-field scale
    // (≈ |camPos| = the camera distance after recenter) — a linear density
    // either ignores the near field or erases the background vista.
    let t0 = max(length(G.camPos.xyz), 1e-12);
    let dec = log2(1.0 + t / t0) * 0.30103;
    let f = max(1.0 - exp(-2.3 * (-G.fog.x) * dec), farF);
    let scat = pow(max(dot(rd, normalize(G.light.xyz)), 0.0), 8.0) * G.fog.y;
    col = mix(col, bgOut + s2l(G.lightC.rgb) * scat, f);
  } else {
    // Legacy distance fade, driven to 100% across the far band.
    col = mix(col, bgOut, max(clamp(t / tFar, 0.0, 1.0) * 0.6, farF));
  }
  return vec4f(max(col, vec3f(0.0)), 1.0);                // linear HDR out (P0)
}
${captureBlock}`,
    { coloring, scene, hybrid, morph },
  );
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
export const PSTOPS_WORD = 35; // vec4 index of `pstops[0]` (COLORING P0 — 8 words, 35..42)
export const PCTL_WORD = 43; // vec4 index of `pctl` (COLORING P0 — stop count/cyclic)
export const P3CTL_WORD = 44; // vec4 index of `p3ctl` (COLORING P3 — iridescence/phase)
export const OFFSETLO_WORD = 45; // vec4 index of `offsetLo` (deep zoom P4 — O lo words + k*)
// TILED_EXPORT §2.2.1(a) — appended AFTER offsetLo on purpose: every index above
// stays untouched, which is the whole point of appending rather than inserting.
export const TILE_WORD = 46; // vec4 index of `tile`   = (sx, sy, bx, by), identity (1,1,0,0)
export const TILEPX_WORD = 47; // vec4 index of `tilepx` = (rx0, ry0, W, H), w=0 ⇒ off
export const GLOBALS_WORDS = 48; // total vec4 count (…offsetLo=45, tile=46, tilepx=47)
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
  let hdrSample = textureLoad(hdr, vec2i(pos.xy), 0);
  var c = hdrSample.rgb;
  // P3 bloom composite (pre-tonemap, HDR): bilinear-upsample the half-res
  // blurred brights. Strength 0 → the bloom passes never ran; the texture is
  // zero-initialized (WebGPU), so this adds exactly nothing.
  let bloomStr = G.w[${FOG_WORD}].z;
  if (bloomStr > 0.0) {
    let uv = pos.xy / G.w[0].xy;
    c = c + textureSampleLevel(bloomTex, bloomSamp, uv, 0.0).rgb * bloomStr;
  }
  // TILED_EXPORT §2.2.1(a) — vignette and dither are functions of ABSOLUTE
  // normalised / absolute screen position, so a tile must be told where it sits
  // in the parent frame or each tile grows its own vignette and reseeds its own
  // dither pattern. tilepx.w = 0 (untiled) leaves gpos/gres exactly the values
  // the two blocks below used before this existed. NOTE the bloom composite
  // above keeps pos.xy / G.w[0].xy — the bloom texture is TILE-sized, so
  // tile-local is the correct address there and must not be "fixed".
  var gpos = pos.xy;
  var gres = G.w[0].xy;
  if (G.w[${TILEPX_WORD}].w > 0.0) {
    gpos = pos.xy + G.w[${TILEPX_WORD}].xy;
    gres = G.w[${TILEPX_WORD}].zw;
  }
  c = c * exp2(post.y);                                  // exposure bias (EV)
  if (post.w > 0.0) {                                    // vignette (default 0)
    let d = distance(gpos / gres, vec2f(0.5)) * 1.41421;
    c = c * (1.0 - post.w * smoothstep(0.55, 1.0, d));
  }
  if (post.x > 0.5) { c = tone3(c); }                    // filmic soft-shoulder
  var o = l2s(c);
  o = o + vec3f((ign(gpos) - 0.5) * (post.z / 255.0));   // dither pre-quantize
  return vec4f(o, hdrSample.a); // straight (non-premultiplied) alpha — issue #428
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
// glslNum (GLSL float literal) now lives in glslfmt.js, shared with the
// standalone exporter. Imported at the top of this file.

// Shape formulas (a single leaf) and scenes (multiple CSG objects) keep their
// geometry in `objects[]`, NOT the top-level op-list — so the iterateJIT_ path
// below (which only walks `ops`) would emit an empty body for them. Emit the
// leaf's own DE for a single shape, and an explanatory stub for a scene.
function exportShapeDE(formula, obj, leaf) {
  const safe = formula.name.replace(/[^A-Za-z0-9_]/g, "_");
  const prm = (obj.shapeParams || []).slice(0, 4);
  while (prm.length < 4) prm.push(0);
  const t = obj.transform || {};
  const origin = t.origin || [0, 0, 0];
  const uscale = t.uscale ?? 1;
  const rot = t.rot || [0, 0, 0];
  const identity =
    origin.every((x) => x === 0) && uscale === 1 && rot.every((x) => x === 0);
  const notes = [];
  if (!identity)
    notes.push(
      `// NOTE: object transform (origin ${origin.join(",")}, uscale ${uscale}, rot ${rot.join(",")}°) is NOT baked — this DE is in the shape's local space.`,
    );
  if (obj.ops && obj.ops.length)
    notes.push(
      `// NOTE: ${obj.ops.length} pre-step op(s) on this shape are NOT baked (${obj.ops.map((o) => o.key).join(" → ")}).`,
    );
  const paramNames = (leaf.params || []).map((p) => p.name).join(", ");
  return `// HAND_CRAFTED: generated by the web formula creator (shape-DE export).
// SHAPE: ${formula.name} — leaf '${leaf.key}' (id ${leaf.id})
// This formula is a SHAPE (an implicit-surface distance estimator), so it
// exports as a standalone DE function — NOT the op-list iterateJIT_ format that
// plain fractals use. Feed it a point, get the signed distance.
// PARAMS (prm.x/y/z/w): ${paramNames}
// DEoption: ${effectiveDeOption(formula)}
${notes.join("\n")}${notes.length ? "\n" : ""}//
float shapeDE_${safe}(vec3 p) {
  vec4 prm = vec4(${prm.map(glslNum).join(", ")});
  ${leaf.glsl}
}
`;
}

function exportSceneStub(formula, objects) {
  const list = objects
    .map((o) => {
      const leaf = LEAVES.find((l) => l.id === o.shapeId);
      const nm = leaf ? `${leaf.key} (leaf ${leaf.id})` : `shape #${o.shapeId}`;
      return `//   • ${nm}  [combine ${o.combine ?? 0}]`;
    })
    .join("\n");
  return `// HAND_CRAFTED: generated by the web formula creator.
// SCENE: ${formula.name} — ${objects.length} objects (CSG composition).
//
// A scene combines several shape DEs with CSG operators (union / subtract /
// intersect / smooth). The iterateJIT_ op-list format represents ONE iterated
// formula, not a composition, so there is no single body to emit here.
//
// Objects in this scene:
${list}
//
// To export usable GLSL: open a single shape on its own, or use the standalone
// scene export (roadmap: standalone shader/scene export — issue #291).
`;
}

export function exportGLSL(formula) {
  // Shape / scene formulas carry geometry in objects[], not the op-list.
  const objects = formula.objects || [];
  if (objects.length === 1) {
    const leaf = LEAVES.find((l) => l.id === objects[0].shapeId && l.id !== 0);
    if (leaf) return exportShapeDE(formula, objects[0], leaf);
  }
  if (objects.length > 1) return exportSceneStub(formula, objects);

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

// Single source of truth for the engine's fixed capacity limits. Every renderer
// tier (WebGPU renderer.js, WebGL2 renderer_gl.js/shader_gl.js) and the importer
// (sanitize.js) sized these independently as scattered magic numbers, so the
// caps could — and did — disagree. Defining them ONCE means a change is made in
// one place and "parity by construction" replaces "MUST match" comments.
//
// CROSS-TIER OP-COUNT PARITY (docs/planning/REFACTORING.md item 2 — unified
// 2026-07-04): a FLAT formula is capped at MAX_FLAT_OPS on EVERY tier, and that
// cap is the SMALLER (WebGL2) value — so the same flat formula renders
// identically on WebGPU, WebGL2, and CPU (no more "65–192 ops draws differently
// per machine"). MAX_OPS_WEBGPU stays larger only as the WebGPU op-BUFFER
// capacity: it must hold a full CSG scene (MAX_OBJECTS × MAX_OPS_PER_OBJECT) and
// hybrid/morph slot concatenations, none of which is a single flat formula.
// Presets top out at ~8 ops, so nothing ships near the cap.

// Scene dimensions — all tiers agree on these.
export const MAX_OBJECTS = 8; // CSG scene object cap
export const MAX_OPS_PER_OBJECT = 24; // per-object op cap within a scene

// WebGPU op-BUFFER capacity — sized for a full scene + hybrid/morph concat, NOT
// the flat-formula cap (that's MAX_FLAT_OPS below). writeOps/writeScene throw if
// a concatenation exceeds it (never silently truncate).
export const MAX_OPS_WEBGPU = MAX_OBJECTS * MAX_OPS_PER_OBJECT; // 192 (8 × 24)
// WebGL2 op budget: ops are unrolled into generated GLSL, and uP[] must hold
// MAX_OP_PARAMS params for every one of them (see MAX_PARAMS below). This is the
// tighter tier, so it sets the shared flat cap.
export const MAX_OPS_WEBGL2 = 64;

// Flat op-list cap the importer applies — unified to the SMALLER (WebGL2) tier so
// a flat formula renders identically everywhere (exact cross-tier parity). Scene
// objects are capped separately at MAX_OPS_PER_OBJECT; this is flat formulas only.
export const MAX_FLAT_OPS = MAX_OPS_WEBGL2; // 64

// ── Per-op parameter budget (docs/planning/OP_PARAM_ENCODING.md) ────────────
// An op declares up to MAX_OP_PARAMS params. The first MAX_OP_PARAMS_INLINE ride
// `struct Op` on the WebGPU tier (16 B, UNCHANGED); any beyond that ride the
// parallel `opAux` overflow lane (one vec4f per op slot) and are read ONLY inside
// the fat op's own case body — so an op-set with no fat op emits a shader that
// neither declares nor reads the lane, and pays exactly nothing (#125's lesson:
// uniform-gated heavy code is NOT free, so the guarantee is made structural).
// The WebGL2 tier needs none of this — shader_gl.js's uP[] table has always been
// arity-driven — and the CPU tier takes an untyped array.
export const MAX_OP_PARAMS_INLINE = 3; // p0..p2 — struct Op, unchanged
export const OP_AUX_F32 = 3; // p3..p5 — one vec4 lane per op slot (.w reserved)
export const MAX_OP_PARAMS = MAX_OP_PARAMS_INLINE + OP_AUX_F32; // 6

// WebGL2 param uniform-array size (uP[]) = every flat op's full param budget.
// DERIVED, never hand-typed: this identity used to live in a comment ("192 = 64
// ops × 3 params") with nothing recomputing it and the flat packer silently
// TRUNCATING on overrun, so the two halves could — and did — desync unnoticed.
// Now a change to either factor propagates, and uniformPack.js throws instead.
//
// Sized off MAX_FLAT_OPS (not the larger WebGPU buffer cap) for the same reason
// MAX_FLAT_OPS itself takes the smaller tier: it preserves the property that NO
// legal flat formula can be rejected for params on one tier only. Any smaller
// number reintroduces a WebGPU-renders / WebGL2-throws split.
export const MAX_PARAMS = MAX_FLAT_OPS * MAX_OP_PARAMS; // 384

// A shape LEAF declares up to MAX_LEAF_PARAMS params (#627 — the objAux lane,
// mirroring the op lane above; docs/planning/OP_PARAM_ENCODING.md §"objAux").
// The first MAX_LEAF_PARAMS_INLINE ride `struct Obj`'s one 16-byte word
// (sp0..sp3, word 4 — 96-byte descriptor UNCHANGED); any beyond ride a
// parallel `objAux` storage buffer (one vec4f per OBJECT slot) that is
// declared and bound ONLY when the shader variant's leaf-id set contains a
// fat leaf — a scene of ≤4-param leaves emits byte-identical WGSL and pays
// exactly nothing (the same structural pay-per-use guarantee as opAux).
// WebGL2 mirrors with a second uniform vec4 array (uObjPrimP2), added to the
// Bulk block on the same predicate; the CPU tier takes an untyped array.
export const MAX_LEAF_PARAMS_INLINE = 4; // sp0..sp3 — struct Obj word 4, unchanged
export const LEAF_AUX_F32 = 4; // sp4..sp7 — one vec4 lane per object slot
export const MAX_LEAF_PARAMS = MAX_LEAF_PARAMS_INLINE + LEAF_AUX_F32; // 8

// Iteration cap — the GLSL literal loop bound (shader_gl.js interpolates this
// into every orbit loop); WGSL's bound is a uniform but preview.js clamps to the
// same value for cross-backend parity. Also the importer's upper bound on iters.
export const MAX_ITERS = 64;

// Bailout radii. Escape-time power maps need a small bailout (r=8 ⇒ r²=64) or
// rᵖᵒʷᵉʳ overflows fp32; IFS folds stay bounded so a huge radius is harmless.
// preview.js (uniforms), cpu.js (CPU orbit), and the scene marcher all key off
// these two values.
export const BAILOUT_ESCAPE = 64.0;
export const BAILOUT_IFS = 1.0e6;

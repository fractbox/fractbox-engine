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
// WebGL2 op budget: uP[192] params ÷ 3 params/op = 64, and ops are unrolled into
// generated GLSL. This is the tighter tier, so it sets the shared flat cap.
export const MAX_OPS_WEBGL2 = 64;

// WebGL2 param uniform-array size (uP[]). 192 = 64 ops × 3 params.
export const MAX_PARAMS = 192;

// Flat op-list cap the importer applies — unified to the SMALLER (WebGL2) tier so
// a flat formula renders identically everywhere (exact cross-tier parity). Scene
// objects are capped separately at MAX_OPS_PER_OBJECT; this is flat formulas only.
export const MAX_FLAT_OPS = MAX_OPS_WEBGL2; // 64

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

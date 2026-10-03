// Shared op-parameter packer — the single source of truth for the `uP[]` uniform
// contents. Extracted from renderer_gl.js's writeOps/writeHybrid so the standalone
// GLSL exporter (see docs/planning/EXPORT_M0_STANDALONE_GLSL.md) bakes byte-identical
// values instead of re-deriving the packing and drifting from the live render.
//
// Pure: (op-lists) → Float32Array(MAX_PARAMS). No GL, no renderer state.

import { byKey } from "./operators.js";
import { MAX_PARAMS } from "./limits.js";

// Pack one or more op-lists CONTIGUOUSLY from slot 0 into a fresh, zero-padded
// Float32Array(MAX_PARAMS). One list = the flat/single-object path; two lists =
// the hybrid A-then-B concat (packed into the SAME array, B continuing where A
// left off), matching writeHybrid's original loop. Callers pass op-lists already
// clamped to MAX_OPS (structure/codegen is the renderer's concern); this only
// walks def.params and honours the MAX_PARAMS slot ceiling.
//
// Throws on an unknown op key with the SAME message the engine has always used
// ("writeOps: unknown op key <k>") so the error contract is unchanged.
export function packOpParams(...opLists) {
  const out = new Float32Array(MAX_PARAMS);
  let slot = 0;
  for (const ops of opLists) {
    for (let i = 0; i < ops.length; i++) {
      const def = byKey(ops[i].key);
      // Unknown keys are programmer errors — throw like the WebGPU tier, never
      // silently skip the op (sanitize.js throws upstream for app flows; this
      // hardens non-sanitized callers).
      if (!def) throw new Error(`writeOps: unknown op key ${ops[i].key}`);
      // Overrunning uP[] used to SILENTLY TRUNCATE (`&& slot < MAX_PARAMS`),
      // so a flat formula past the pool rendered a different — but plausible —
      // fractal with no error anywhere. The scene path has always thrown
      // (renderer_gl.js writeScene); this makes the flat path agree. Correctness
      // fix, independent of the overflow lane (OP_PARAM_ENCODING.md §7).
      if (slot + def.params.length > MAX_PARAMS)
        throw new Error(
          `packOpParams: param slot ${slot + def.params.length} > cap ${MAX_PARAMS}`,
        );
      for (let k = 0; k < def.params.length; k++) {
        out[slot++] = ops[i].values[k] ?? 0;
      }
    }
  }
  return out;
}

// ── The WebGPU overflow lane (docs/planning/OP_PARAM_ENCODING.md §5.5) ────────
// Packs the OVERFLOW params (p3..p5) of one or more op-lists into a fresh,
// zero-filled Float32Array of one vec4f per op SLOT, concatenated in exactly the
// order the WebGPU op buffer packs them — so opAux[o] always lines up with
// ops[o]. Unlike packOpParams above this is indexed per-OP, not per-param: every
// slot gets a lane whether or not its op declares overflow params, because a
// thin op must never inherit a previous frame's p3..p5 from the same slot.
//
// ⚠ WHY THIS IS A SHARED PURE FUNCTION and not two inline loops. renderer.js's
// two packers do NOT use the same index: writeOps walks a single `i`, but
// writeScene's inner `for (let i = 0; ...)` RESTARTS at 0 for every object and
// only `cursor` is the global slot. Writing the lane with `i` there would alias
// every object after the first onto object 0's lanes — silently zeroing or
// swapping the overflow params of a fat op in any non-first scene object, with
// no compile error and no CI signal (WGSL is compiled nowhere in CI). Handing
// both callers ONE function that walks the concatenation itself removes the
// chance to get the index wrong, and makes the ordering unit-testable with no
// GPU (core/opaux.test.mjs).
export function packOpAuxLanes(...opLists) {
  let total = 0;
  for (const ops of opLists) total += ops.length;
  const out = new Float32Array(Math.max(total, 1) * 4);
  let slot = 0;
  for (const ops of opLists) {
    for (let i = 0; i < ops.length; i++) {
      const v = ops[i].values;
      const j = slot * 4;
      out[j + 0] = v?.[3] ?? 0;
      out[j + 1] = v?.[4] ?? 0;
      out[j + 2] = v?.[5] ?? 0;
      // out[j + 3] stays 0 — the reserved .w component.
      slot++;
    }
  }
  return out;
}

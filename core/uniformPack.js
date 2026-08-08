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
      for (let k = 0; k < def.params.length && slot < MAX_PARAMS; k++) {
        out[slot++] = ops[i].values[k] ?? 0;
      }
    }
  }
  return out;
}

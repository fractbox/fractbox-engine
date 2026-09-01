// Guards the engine capacity limits (core/limits.js) — the single source the
// renderers + sanitizer size their buffers from. These assert the values every
// tier historically hard-coded, so a well-meaning edit can't silently change a
// buffer size out from under one tier. Named *.test.mjs so sync skips it.
//
// Run: node --test core/limits.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";
import {
  MAX_OBJECTS,
  MAX_OPS_PER_OBJECT,
  MAX_OPS_WEBGPU,
  MAX_OPS_WEBGL2,
  MAX_PARAMS,
  MAX_FLAT_OPS,
  MAX_OP_PARAMS,
  MAX_OP_PARAMS_INLINE,
  OP_AUX_F32,
} from "./limits.js";
import { SCENE_CAPS } from "./sanitize.js";
import {
  MAX_PARAMS as GL_PARAMS,
  MAX_OBJECTS as GL_OBJECTS,
} from "./shader_gl.js";

test("caps hold the exact values every tier previously hard-coded", () => {
  assert.equal(MAX_OBJECTS, 8);
  assert.equal(MAX_OPS_PER_OBJECT, 24);
  assert.equal(MAX_OPS_WEBGPU, 192); // 8 × 24 — NOT the param cap, despite
  assert.equal(MAX_OPS_WEBGL2, 64); // having been the same number until 2026-08.
  assert.equal(MAX_PARAMS, 384); // was 192 at 3 params/op (OP_PARAM_ENCODING.md)
  assert.equal(MAX_FLAT_OPS, 64);
  assert.equal(MAX_OP_PARAMS_INLINE, 3); // struct Op p0..p2 — must never move
  assert.equal(OP_AUX_F32, 3); // opAux lane p3..p5 (.w reserved)
  assert.equal(MAX_OP_PARAMS, 6);
});

test("derived relationships stay consistent", () => {
  // WebGPU op buffer = scene dimensions (holds a full scene + hybrid/morph concat).
  assert.equal(MAX_OPS_WEBGPU, MAX_OBJECTS * MAX_OPS_PER_OBJECT);
  // Cross-tier parity (item 2): the flat cap is unified to the SMALLER (WebGL2)
  // tier, so a flat formula renders identically everywhere.
  assert.equal(MAX_FLAT_OPS, MAX_OPS_WEBGL2);
  // The WebGPU op BUFFER is still larger than the flat cap — it's sized for
  // scenes/concats, not single flat formulas.
  assert.ok(MAX_OPS_WEBGL2 < MAX_OPS_WEBGPU);
  // uP[] must hold EVERY flat op's FULL param budget. This identity used to be
  // prose ("192 = 64 ops × 3 params") with nothing recomputing it; pinning it
  // means neither factor can move without the other (OP_PARAM_ENCODING.md §5.1).
  // Sized off the flat cap, not the WebGPU buffer, so no legal flat formula can
  // be rejected for params on one tier only.
  assert.equal(MAX_PARAMS, MAX_FLAT_OPS * MAX_OP_PARAMS);
  assert.equal(MAX_OP_PARAMS, MAX_OP_PARAMS_INLINE + OP_AUX_F32);
  // The overflow lane is ONE vec4 per op slot — .w is reserved, so the lane can
  // carry at most 3 more f32. A 4th would need a second lane + a second binding.
  assert.ok(
    OP_AUX_F32 <= 3,
    "one vec4f lane holds at most 3 params + reserved",
  );
});

test("no registered operator exceeds the param budget", async () => {
  const { OPERATORS } = await import("./operators.js");
  for (const o of OPERATORS)
    assert.ok(
      o.params.length <= MAX_OP_PARAMS,
      `${o.key} has ${o.params.length} params > ${MAX_OP_PARAMS}`,
    );
  // A whole flat formula of the WIDEST op must still fit the pool — the
  // property MAX_PARAMS is derived to guarantee.
  const widest = Math.max(...OPERATORS.map((o) => o.params.length));
  assert.ok(MAX_FLAT_OPS * widest <= MAX_PARAMS);
});

test("consumers see the single-sourced values (no drift)", () => {
  assert.equal(SCENE_CAPS.MAX_OBJECTS, MAX_OBJECTS);
  assert.equal(SCENE_CAPS.MAX_OPS_PER, MAX_OPS_PER_OBJECT);
  assert.equal(GL_PARAMS, MAX_PARAMS);
  assert.equal(GL_OBJECTS, MAX_OBJECTS);
});

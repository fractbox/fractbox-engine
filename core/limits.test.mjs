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
} from "./limits.js";
import { SCENE_CAPS } from "./sanitize.js";
import { MAX_PARAMS as GL_PARAMS, MAX_OBJECTS as GL_OBJECTS } from "./shader_gl.js";

test("caps hold the exact values every tier previously hard-coded", () => {
  assert.equal(MAX_OBJECTS, 8);
  assert.equal(MAX_OPS_PER_OBJECT, 24);
  assert.equal(MAX_OPS_WEBGPU, 192);
  assert.equal(MAX_OPS_WEBGL2, 64);
  assert.equal(MAX_PARAMS, 192);
  assert.equal(MAX_FLAT_OPS, 64);
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
});

test("consumers see the single-sourced values (no drift)", () => {
  assert.equal(SCENE_CAPS.MAX_OBJECTS, MAX_OBJECTS);
  assert.equal(SCENE_CAPS.MAX_OPS_PER, MAX_OPS_PER_OBJECT);
  assert.equal(GL_PARAMS, MAX_PARAMS);
  assert.equal(GL_OBJECTS, MAX_OBJECTS);
});

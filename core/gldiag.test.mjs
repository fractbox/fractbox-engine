// Unit test for the pure WebGL2 health classifier (core/gldiag.js) — the
// verdict layer the GL fallback tier relies on. Named *.test.mjs so sync skips
// it. Run: node --test core/gldiag.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";
import { classifyGlHealth, glErrorName } from "./gldiag.js";

test("healthy / not-yet-observed signals → not dead", () => {
  assert.deepEqual(classifyGlHealth(), { dead: false, reason: null });
  assert.deepEqual(classifyGlHealth({}), { dead: false, reason: null });
  // A NO_ERROR (0) in the draw samples is not a fault.
  assert.deepEqual(classifyGlHealth({ drawErrors: [0, 0, 0] }), {
    dead: false,
    reason: null,
  });
});

test("a shader compile failure marks the tier dead", () => {
  const v = classifyGlHealth({ compileFailed: true });
  assert.equal(v.dead, true);
  assert.match(v.reason, /compile/);
});

test("a program link failure marks the tier dead", () => {
  const v = classifyGlHealth({ linkFailed: true });
  assert.equal(v.dead, true);
  assert.match(v.reason, /link/);
});

test("a nonzero GL error on the first draws → dead, reason names the code", () => {
  const v = classifyGlHealth({ drawErrors: [0, 0x0502] });
  assert.equal(v.dead, true);
  assert.match(v.reason, /INVALID_OPERATION/);
});

test("context loss and creation error each mark the tier dead", () => {
  assert.equal(classifyGlHealth({ contextLost: true }).dead, true);
  assert.equal(classifyGlHealth({ contextCreationError: true }).dead, true);
});

test("verdict is worst-first: a creation error wins over a later draw error", () => {
  const v = classifyGlHealth({
    contextCreationError: true,
    drawErrors: [0x0500],
  });
  assert.equal(v.reason, "webglcontextcreationerror");
});

test("compile beats link when both are flagged", () => {
  const v = classifyGlHealth({ compileFailed: true, linkFailed: true });
  assert.match(v.reason, /compile/);
});

test("glErrorName maps known codes and falls back to hex", () => {
  assert.equal(glErrorName(0x0502), "INVALID_OPERATION");
  assert.equal(glErrorName(0x9242), "CONTEXT_LOST_WEBGL");
  assert.equal(glErrorName(0x1234), "0x1234");
});

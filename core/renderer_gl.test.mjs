// #708 — behavioral tests for the WebGL2 renderer's fault handling, driven
// against a MOCK WebGL2 context (node has no GL; the mock implements exactly
// the calls createRendererGL makes). What real-GL probes can't cover in CI is
// covered here: the failed-build latch (a persistently failing compile is
// attempted ONCE, not per pumped frame), per-formula recovery (a later good
// link clears the compile/link latches so one bad formula no longer reads as
// a dead tier), the context-lost preventDefault + restore path, and the boot
// link deferral. Named *.test.mjs so sync skips it.
// Run: node --test core/renderer_gl.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";
import { createRendererGL } from "./renderer_gl.js";

// Minimal WebGL2 mock — just the surface renderer_gl.js touches. `state` is
// mutable so a test can flip compile failure on and off mid-run.
function makeMock() {
  const state = { failFrag: false, failLink: false };
  const calls = { vertCompiles: 0, fragCompiles: 0, links: 0 };
  const listeners = {};
  const gl = {
    VERTEX_SHADER: 0x8b31,
    FRAGMENT_SHADER: 0x8b30,
    COMPILE_STATUS: 0x8b81,
    LINK_STATUS: 0x8b82,
    UNIFORM_BUFFER: 0x8a11,
    DYNAMIC_DRAW: 0x88e8,
    INVALID_INDEX: 0xffffffff,
    TRIANGLES: 0x0004,
    FRAMEBUFFER: 0x8d40,
    createVertexArray: () => ({}),
    bindVertexArray: () => {},
    createShader: (type) => ({ type }),
    shaderSource: (sh, src) => {
      sh.src = src;
    },
    compileShader: (sh) => {
      if (sh.type === gl.FRAGMENT_SHADER) calls.fragCompiles++;
      else calls.vertCompiles++;
      sh.ok = sh.type === gl.FRAGMENT_SHADER ? !state.failFrag : true;
    },
    getShaderParameter: (sh) => sh.ok,
    getShaderInfoLog: () => "mock compile log",
    deleteShader: () => {},
    createProgram: () => ({}),
    attachShader: () => {},
    linkProgram: () => {
      calls.links++;
    },
    getProgramParameter: () => !state.failLink,
    getProgramInfoLog: () => "mock link log",
    deleteProgram: () => {},
    createBuffer: () => ({}),
    bindBuffer: () => {},
    bufferData: () => {},
    bufferSubData: () => {},
    getUniformBlockIndex: () => 0,
    uniformBlockBinding: () => {},
    bindBufferBase: () => {},
    getUniformLocation: () => null,
    useProgram: () => {},
    viewport: () => {},
    bindFramebuffer: () => {},
    drawArrays: () => {},
    getError: () => 0,
    finish: () => {},
  };
  const canvas = {
    width: 4,
    height: 4,
    addEventListener: (name, fn) => {
      (listeners[name] ||= []).push(fn);
    },
    getContext: () => gl,
  };
  const fire = (name, ev = {}) =>
    (listeners[name] || []).forEach((fn) => fn(ev));
  return { canvas, gl, calls, state, fire };
}

const OPS_A = [{ key: "boxFold", values: [1] }];
const OPS_B = [{ key: "scale", values: [1.8] }];

test("#708 boot: the first link is DEFERRED until globals exist", async () => {
  const m = makeMock();
  const r = await createRendererGL(m.canvas);
  assert.equal(m.calls.vertCompiles, 1, "boot compiles the vertex shader");
  // capturesettle ordering: write* lands BEFORE the first writeGlobals —
  // linking here would use the all-false want defaults and force an immediate
  // relink on the first applyUniforms (the boot double compile).
  r.writeOps(OPS_A);
  assert.equal(m.calls.fragCompiles, 0, "no frag compile before writeGlobals");
  r.writeGlobals({});
  r.writeOps(OPS_A); // same signature — the deferred link must still happen
  assert.equal(m.calls.fragCompiles, 1, "exactly one deferred link");
  assert.equal(r.glHealth().dead, false);
  assert.equal(r.glHealth().everLinked, true);
});

test("#708 storm guard: a failing compile runs ONCE per signature, then cheap-rethrows", async () => {
  const m = makeMock();
  const notes = [];
  const r = await createRendererGL(m.canvas, {
    onTrouble: (kind) => notes.push(kind),
  });
  r.writeGlobals({});
  m.state.failFrag = true;
  assert.throws(() => r.writeOps(OPS_A), /compile failed/);
  assert.equal(m.calls.fragCompiles, 1);
  // The pump re-calls write* every settled/interactive frame — each retry of
  // the SAME failing signature must be a cheap rethrow, not codegen+compile.
  for (let i = 0; i < 5; i++) assert.throws(() => r.writeOps(OPS_A));
  assert.equal(m.calls.fragCompiles, 1, "no per-frame recompile storm");
  assert.equal(
    notes.filter((k) => k === "gl-compile-fail").length,
    1,
    "one diag note per failing build, not one per frame",
  );
  assert.equal(r.glHealth().dead, true);
  assert.equal(r.glHealth().everLinked, false, "boot-path failure: tier-fatal");
  // A DIFFERENT signature is a fresh attempt.
  assert.throws(() => r.writeOps(OPS_B));
  assert.equal(m.calls.fragCompiles, 2);
});

test("#708 recovery: a later good link clears the per-formula fault latches", async () => {
  const m = makeMock();
  const r = await createRendererGL(m.canvas);
  r.writeGlobals({});
  r.writeOps(OPS_A); // good boot link
  assert.equal(r.glHealth().dead, false);
  m.state.failFrag = true;
  assert.throws(() => r.writeOps(OPS_B)); // ONE formula's GLSL fails
  assert.equal(r.glHealth().dead, true, "the failing formula reads as dead…");
  assert.equal(
    r.glHealth().everLinked,
    true,
    "…but everLinked marks it formula-local, not tier-fatal",
  );
  m.state.failFrag = false;
  r.writeOps(OPS_A); // next good formula
  assert.equal(
    r.glHealth().dead,
    false,
    "compileFailed must clear on the next successful link (was a one-way latch)",
  );
});

test("#708 context loss: preventDefault + restore recovers the tier", async () => {
  const m = makeMock();
  const r = await createRendererGL(m.canvas);
  r.writeGlobals({});
  r.writeOps(OPS_A);
  let prevented = false;
  m.fire("webglcontextlost", { preventDefault: () => (prevented = true) });
  assert.equal(
    prevented,
    true,
    "contextlost must preventDefault — the spec default never fires restored",
  );
  assert.equal(r.glHealth().dead, true);
  assert.match(r.glHealth().reason, /context lost/);
  const vertsBefore = m.calls.vertCompiles;
  m.fire("webglcontextrestored", {});
  assert.equal(
    m.calls.vertCompiles,
    vertsBefore + 1,
    "restore recompiles the vertex shader",
  );
  assert.equal(r.glHealth().dead, false, "restore clears the stale latches");
  const frags = m.calls.fragCompiles;
  r.writeOps(OPS_A); // same signature — opSig was reset, so this relinks
  assert.equal(m.calls.fragCompiles, frags + 1, "next write* relinks");
  assert.equal(r.glHealth().dead, false);
});

// Source tripwires for the bits a mock can't observe cheaply — the repo's
// established pattern (equirect.test.mjs, scenemute.test.mjs).
test("#708 source tripwires: U() memoizes null; scene rebuild resets progNeon", async () => {
  const { readFile } = await import("node:fs/promises");
  const src = await readFile(
    new URL("./renderer_gl.js", import.meta.url),
    "utf8",
  );
  assert.doesNotMatch(
    src,
    /loc\[name\] \?\?=/,
    "??= re-queries a memoized null location every frame",
  );
  assert.match(
    src,
    /if \(v === undefined\) loc\[name\] = v = gl\.getUniformLocation/,
    "U() must memoize the null (inactive-uniform) result too",
  );
  assert.match(
    src,
    /progNeon = false;[^]{0,400}progFilm = false;/,
    "the scene rebuild must reset BOTH flat-only prog latches",
  );
});

// ── #718 — overflow THROWS, matching the WebGPU twin's contract ─────────────
// renderer.js writeOps: "Overflow/unknown-key are programmer errors — throw…
// never silently truncate". The GL tier used to Math.min-clip at MAX_OPS, so
// a raw-core consumer feeding a 70-op formula got a plausible-looking but
// DIFFERENT fractal here while WebGPU threw loudly.
test("#718 writeOps/writeHybrid throw on op overflow instead of truncating", async () => {
  const m = makeMock();
  const r = await createRendererGL(m.canvas);
  const over = Array.from({ length: r.MAX_OPS + 1 }, () => ({
    key: "boxFold",
    values: [1],
  }));
  assert.throws(() => r.writeOps(over), /ops > cap/);
  assert.throws(() => r.writeHybrid([OPS_A, over], [1, 1], [0, 0]), /cap/);
  // at the cap exactly: accepted (the boundary must not throw)
  r.writeOps(over.slice(0, r.MAX_OPS));
});

test("#718 hybrid uP[] overflow throws an actionable budget error at codegen (not an opaque GLSL log)", async () => {
  const { buildFragGL } = await import("./shader_gl.js");
  const { MAX_PARAMS, MAX_OPS_WEBGL2 } = await import("./limits.js");
  const { OPERATORS } = await import("./operators.js");
  // The widest registered op, so per-slot counts stay legal (≤ 64) while the
  // CHAINED total exceeds the shared pool — the sceneParamLayout fixture rule.
  const widest = OPERATORS.reduce((a, b) =>
    b.params.length > a.params.length ? b : a,
  );
  const per = Math.ceil((MAX_PARAMS + 1) / (2 * widest.params.length));
  assert.ok(
    per <= MAX_OPS_WEBGL2,
    "fixture stays under the per-slot op cap while busting the param pool",
  );
  const slot = Array.from({ length: per }, () => ({
    key: widest.key,
    values: widest.params.map((p) => p.default),
  }));
  assert.throws(
    () => buildFragGL(slot, [{ ops: slot }]),
    /over the WebGL2 uniform budget/,
    "hybrid budget overflow must throw the scene path's actionable message",
  );
  // A 2-slot hybrid exactly ON the cap still builds.
  const perOk = Math.floor(MAX_PARAMS / (2 * widest.params.length));
  const slotOk = slot.slice(0, perOk);
  buildFragGL(slotOk, [{ ops: slotOk }]);
});

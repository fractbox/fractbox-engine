// #730 — runtime hygiene pins for createRenderer's diagnostics + write paths,
// against a stub WebGPU device (no GPU in CI — the streamlines.test.mjs
// precedent). Three contracts:
//
//   1. diag.compiles / diag.events are bounded rings (DIAG_CAP newest kept,
//      shed records counted in diag.dropped) — a browse-heavy tab left open
//      for days must not ratchet memory or grow getDiag() without bound.
//   2. timedPipelineAsync touches NO device error scopes (the scope stack is
//      device-GLOBAL and LIFO; two concurrent compiles holding scopes across
//      their awaits popped each other's, misattributing validation errors),
//      and a rejection is attributed to the compile that actually failed even
//      when concurrent builds settle out of order.
//   3. A writeScene that THROWS (op-cap overflow / unknown op) leaves the
//      frameLeafIds/frameOps variant latch describing the PREVIOUS scene —
//      the one still in the GPU buffers — not the rejected one.
import test from "node:test";
import assert from "node:assert/strict";
import { createRenderer, DIAG_CAP } from "./renderer.js";

// createRenderer reads the WebGPU usage-flag enums off the global; Node has
// none. Values are opaque to the code under test (OR'd into descriptors the
// stub ignores), so any distinct set will do.
globalThis.GPUBufferUsage ??= {
  STORAGE: 128,
  COPY_DST: 8,
  COPY_SRC: 4,
  UNIFORM: 64,
  MAP_READ: 1,
};
globalThis.GPUTextureUsage ??= {
  COPY_DST: 2,
  TEXTURE_BINDING: 4,
  RENDER_ATTACHMENT: 16,
  COPY_SRC: 1,
};
globalThis.GPUMapMode ??= { READ: 1 };

// A stub navigator.gpu provider, just deep enough for the headless-renderer
// boot path (canvas === null + opts.gpu — the EXPORT_P1 PR-A contract) plus
// the compile/draw machinery these pins exercise.
//   asyncMode "resolve" — createRenderPipelineAsync resolves (default)
//   asyncMode "reject"  — rejects with a numbered "forced-fail N" error
//   asyncMode "manual"  — parks; the test settles each build via s.asyncBuilds
function stubGpu(asyncMode = "resolve") {
  const scopeCalls = []; // "push" / "pop", in device-call order
  const asyncBuilds = []; // manual-mode deferred controls
  let rejectN = 0;
  const pipeline = () => ({ getBindGroupLayout: () => ({}) });
  const pass = {
    setPipeline() {},
    setBindGroup() {},
    setScissorRect() {},
    draw() {},
    end() {},
  };
  const device = {
    features: [],
    pushErrorScope: () => scopeCalls.push("push"),
    popErrorScope: () => {
      scopeCalls.push("pop");
      return Promise.resolve(null);
    },
    createShaderModule: (d) => ({ code: d?.code || "" }),
    createRenderPipeline: () => pipeline(),
    createRenderPipelineAsync() {
      if (asyncMode === "manual") {
        let res, rej;
        const p = new Promise((a, b) => ((res = a), (rej = b)));
        asyncBuilds.push({
          resolve: () => res(pipeline()),
          reject: (msg) => rej(new Error(msg)),
        });
        return p;
      }
      if (asyncMode === "reject")
        return Promise.reject(new Error("forced-fail " + ++rejectN));
      return Promise.resolve(pipeline());
    },
    createBuffer: (d) => ({
      size: d.size,
      mapAsync: async () => {},
      getMappedRange: () => new ArrayBuffer(d.size),
      unmap() {},
      destroy() {},
    }),
    createTexture: () => ({ createView: () => ({}), destroy() {} }),
    createSampler: () => ({}),
    createBindGroup: () => ({}),
    createCommandEncoder: () => ({
      beginRenderPass: () => pass,
      finish: () => ({}),
    }),
    queue: { writeBuffer() {}, submit() {}, async onSubmittedWorkDone() {} },
  };
  const adapter = {
    isFallbackAdapter: false,
    info: { vendor: "stub", architecture: "t", device: "d", description: "x" },
    requestDevice: async () => device,
  };
  return {
    gpu: {
      requestAdapter: async () => adapter,
      getPreferredCanvasFormat: () => "rgba8unorm",
    },
    device,
    scopeCalls,
    asyncBuilds,
  };
}

const feat = (over = {}) => ({
  numericDE: false,
  leaves: null,
  coloring: false,
  scene: false,
  hybrid: false,
  morph: false,
  df64: false,
  ops: [],
  ...over,
});

// Sync compiles record their diag entry from popErrorScope().then(...) — one
// microtask late by design. Let those land before asserting.
const settle = () => new Promise((r) => setImmediate(r));

test("#730 diag.compiles/diag.events are bounded rings that keep the newest records", async () => {
  const s = stubGpu("reject");
  const r = await createRenderer(null, { gpu: s.gpu });
  const N = DIAG_CAP + 25;
  // Every prewarm of the same (never-cached — the build always rejects) key
  // records one compile rec + one compile-error event.
  for (let i = 0; i < N; i++) await r.prewarmMarchFor(feat());
  const d = r.getDiag();
  assert.equal(d.compiles.length, DIAG_CAP, "compiles capped at DIAG_CAP");
  assert.equal(d.events.length, DIAG_CAP, "events capped at DIAG_CAP");
  assert.equal(d.dropped.compiles, 25, "shed compile records are counted");
  assert.equal(d.dropped.events, 25, "shed event records are counted");
  // The ring keeps the NEWEST end — the last record is the last failure.
  assert.equal(d.compiles[DIAG_CAP - 1].error, "forced-fail " + N);
  assert.equal(d.compiles[0].error, "forced-fail 26");
  assert.equal(d.events[DIAG_CAP - 1].kind, "compile-error");
  assert.equal(d.events[DIAG_CAP - 1].detail.error, "forced-fail " + N);
});

test("#730 concurrent async compiles: no device error scopes, rejections attributed to the failing compile", async () => {
  const s = stubGpu("manual");
  const r = await createRenderer(null, { gpu: s.gpu });
  const pA = r.prewarmMarchFor(feat({ ops: [1] })); // key 0:1:-
  const pB = r.prewarmMarchFor(feat({ ops: [2] })); // key 0:2:-
  assert.equal(s.asyncBuilds.length, 2, "both builds in flight concurrently");
  // Settle OUT of order: B (started second) succeeds first, then A fails.
  // Under the old device-global scope window this ordering popped A's scope
  // from B's await — the exact misattribution the fix removes.
  s.asyncBuilds[1].resolve();
  await pB;
  s.asyncBuilds[0].reject("bad A shader");
  await pA; // prewarm swallows the rejection after recording it
  const d = r.getDiag();
  const recA = d.compiles.find((c) => c.name === "prewarm:0:1:-");
  const recB = d.compiles.find((c) => c.name === "prewarm:0:2:-");
  assert.ok(recA && recB, "both compiles recorded");
  assert.equal(recB.ok, true, "the succeeding compile reports ok");
  assert.equal(recB.error, undefined, "…and carries no error");
  assert.equal(recA.ok, false, "the failing compile reports the failure");
  assert.match(recA.error, /bad A shader/, "…with ITS OWN message");
  assert.equal(
    s.scopeCalls.length,
    0,
    "the async compile path must not touch device error scopes at all " +
      "(createRenderPipelineAsync reports failure via rejection; a scope " +
      "held across the await interleaves with concurrent compiles)",
  );
});

test("#730 a throwing writeScene leaves the variant latch consistent with the GPU buffers", async () => {
  const s = stubGpu();
  const r = await createRenderer(null, { gpu: s.gpu });
  const tr = { origin: [0, 0, 0], uscale: 1, rot: [0, 0, 0] };
  r.writeScene([{ objType: 1, transform: tr }]); // box → leaf id 1
  const cam = {
    fov: 1,
    dist: 4,
    target: [0, 0, 0],
    basis: () => ({
      eye: [0, 0, 4],
      fwd: [0, 0, -1],
      right: [1, 0, 0],
      up: [0, 1, 0],
    }),
  };
  const payload = {
    res: [8, 8],
    cam,
    iters: 4,
    opCount: 0,
    addC: false,
    maxSteps: 16,
    bailout: 4,
    eps: 1e-3,
    objectCount: 1,
  };
  r.writeGlobals(payload); // latches activeFeat from frameLeafIds/frameOps
  const target = {
    getCurrentTexture: () => ({ width: 8, height: 8, createView: () => ({}) }),
  };
  r.drawTo(target); // sync-compiles the scene's specialized variant
  await settle();
  const variants = () =>
    r.getDiag().compiles.filter((c) => c.name.startsWith("variant:"));
  assert.equal(variants().length, 1);
  assert.match(
    variants()[0].name,
    /:1$/,
    "the variant key carries the box scene's leaf-id set",
  );

  // A scene with a DIFFERENT leaf (sphere → id 2) whose op concat overflows
  // MAX_OPS: writeScene must throw AND must not move the latch.
  const tooMany = Array.from({ length: r.MAX_OPS + 1 }, () => ({
    key: "boxFold",
    values: [1, 0, 0],
  }));
  assert.throws(
    () =>
      r.writeScene([
        { objType: 2, transform: tr },
        { objType: 0, ops: tooMany, transform: tr },
      ]),
    /cap/,
  );

  // Re-latch + redraw: activeFeat must still describe the BOX scene, so the
  // cached variant is reused — no new "variant:" compile. Before the fix,
  // frameLeafIds was assigned ([2]) BEFORE the throwing validation loop, so
  // this draw compiled (and marched) a sphere-leaf variant against buffers
  // still holding the box scene.
  r.writeGlobals(payload);
  r.drawTo(target);
  await settle();
  assert.equal(
    variants().length,
    1,
    "a rejected scene must not re-key the march variant",
  );
});

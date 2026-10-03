// Pins renderer.releaseBundles() — the ≤4096 still export's "peak GPU memory
// is export-size only" lever (beta black-PNG follow-up, 2026-09-07): drop the
// cached {hdr, accum} bundles before a save larger than the live canvas and
// again after it, so neither the live-size bundle sits beside the export nor
// the export-size bundle lingers after it.
import test from "node:test";
import assert from "node:assert/strict";
import { createRenderer } from "./renderer.js";

globalThis.GPUBufferUsage ??= {
  UNIFORM: 1, STORAGE: 2, COPY_DST: 4, COPY_SRC: 8, MAP_READ: 16, VERTEX: 32,
};
globalThis.GPUTextureUsage ??= {
  RENDER_ATTACHMENT: 1, TEXTURE_BINDING: 2, COPY_SRC: 4, COPY_DST: 8,
};
globalThis.GPUMapMode ??= { READ: 1 };

function stubGpu() {
  const textures = []; // every createTexture, with its destroyed flag
  const pipeline = () => ({ getBindGroupLayout: () => ({}) });
  const pass = { setPipeline() {}, setBindGroup() {}, setScissorRect() {}, draw() {}, end() {} };
  const device = {
    features: [],
    pushErrorScope() {},
    popErrorScope: () => Promise.resolve(null),
    createShaderModule: (d) => ({ code: d?.code || "" }),
    createRenderPipeline: () => pipeline(),
    createRenderPipelineAsync: () => Promise.resolve(pipeline()),
    createBuffer: (d) => ({
      size: d.size,
      mapAsync: async () => {},
      getMappedRange: () => new ArrayBuffer(d.size),
      unmap() {},
      destroy() {},
    }),
    createTexture: (d) => {
      const t = { size: d.size, format: d.format, destroyed: false, createView: () => ({}), destroy() { t.destroyed = true; } };
      textures.push(t);
      return t;
    },
    createSampler: () => ({}),
    createBindGroup: () => ({}),
    createCommandEncoder: () => ({ beginRenderPass: () => pass, copyTextureToBuffer() {}, finish: () => ({}) }),
    queue: { writeBuffer() {}, submit() {}, async onSubmittedWorkDone() {} },
  };
  const adapter = {
    isFallbackAdapter: false,
    info: { vendor: "stub", architecture: "t", device: "d", description: "x" },
    requestDevice: async () => device,
  };
  return { gpu: { requestAdapter: async () => adapter, getPreferredCanvasFormat: () => "rgba8unorm" }, textures };
}
const feat = () => ({ numericDE: false, leaves: null, coloring: false, scene: false, hybrid: false, morph: false, df64: false, ops: [] });

test("releaseBundles destroys every cached hdr/accum bundle and reports the count", async () => {
  const s = stubGpu();
  const r = await createRenderer(null, { gpu: s.gpu });
  assert.equal(typeof r.releaseBundles, "function", "exported on the renderer");
  assert.equal(r.releaseBundles(), 0, "nothing cached on a fresh renderer");
  await r.prewarmMarchFor(feat());
  // A settled accumulate into a caller-owned target populates one bundle at
  // that size (ensureHdr + ensureAccum) — the same route the export takes.
  const tgt = r.createTileTarget(8, 8);
  r.drawAccum({ target: tgt });
  const before = s.textures.filter((t) => !t.destroyed).length;
  const rgba32 = s.textures.filter((t) => t.format === "rgba32float" && !t.destroyed);
  assert.equal(rgba32.length, 2, "the accum ping-pong pair is live after a draw");
  assert.equal(r.releaseBundles(), 1, "one bundle destroyed");
  assert.ok(rgba32.every((t) => t.destroyed), "…including both accum textures");
  const hdr16 = s.textures.filter((t) => t.format === "rgba16float");
  assert.ok(hdr16.length >= 3 && hdr16.filter((t) => t.destroyed).length >= 3, "hdr + bloom pair destroyed");
  assert.ok(s.textures.filter((t) => !t.destroyed).length < before, "fewer live textures than before");
  assert.equal(r.releaseBundles(), 0, "idempotent");
  // The next draw re-ensures a fresh bundle — nothing else was needed.
  r.drawAccum({ target: tgt });
  assert.equal(s.textures.filter((t) => t.format === "rgba32float" && !t.destroyed).length, 2, "re-ensured on the next draw");
  assert.equal(r.releaseBundles(), 1);
});

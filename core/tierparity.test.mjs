// #729 — BEHAVIORAL cross-tier uniform-pack parity: WebGPU (renderer.js
// writeGlobals → Globals words) vs WebGL2 (renderer_gl.js applyUniforms →
// gl.uniform* + the Bulk UBO), driven through the REAL renderers against
// fakes (a mock GPU device / the recording GL mock — no GPU anywhere).
//
// The failure class this replaces a source-regex for: a shading field is
// packed into the Globals tail in renderer.js but the parallel packing in
// renderer_gl.js is missed (or vice versa) — the exact shape of the
// palettePhase/iridescence/sigLo/sigSpan bug (7494398), which rendered as
// silent no-ops on ONE tier for five weeks while every test stayed green.
// The old guard (core/uniformbag.test.mjs) regex-matched the call shape in
// renderer.js only and never touched renderer_gl.js.
//
// Method: for each labeled payload perturbation, write a baseline frame and a
// perturbed frame through BOTH tiers and assert each tier's packed state
// CHANGED (plus, where a distinctive magic value exists, that it landed
// somewhere in the packed words of both tiers — frameparams' charter is that
// both packers truncate the same f64 doubles to f32, so equal inputs give
// bit-equal uniforms). A tier that drops the field packs identical bytes for
// both frames and fails with the field's name attached.
//
// The meta-test at the bottom keeps the perturbation list honest: every key
// deriveFrameParams produces must be moved by at least one perturbation, or
// sit on the explicit constants ledger — so the NEXT derived field (the next
// palettePhase) fails the meta-test until a perturbation covers it, and the
// parity assertions then guard it on both tiers forever.
//
// Scope: the deriveFrameParams surface + the shared raw payload words both
// tiers pack (orthoH / planetK / equirectS). Renderer-owned word groups the
// frameparams charter assigns to ONE tier (WGSL hyb/morph/colorX/post/jitter/
// DOF) are out of scope by design; WebGPU-only fields (bloom, image textures,
// self-reflection — "backend-structural absence", RENDER_QUALITY.md) are
// marked webgpuOnly and assert the WGSL side only.
//
// Run: node --test core/tierparity.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { createRenderer } from "./renderer.js";
import { createRendererGL } from "./renderer_gl.js";
import { GLOBALS_WORDS_ALLOC } from "./shader.js";
import { deriveFrameParams } from "./frameparams.js";
import { makeCamera } from "./camera.js";
import { makeRecordingGL } from "./__fixtures__/glmock.mjs";

// ── mock WebGPU device — exactly the surface createRenderer touches ─────────
// Node has no WebGPU globals; install the spec's usage-flag enums (values per
// the WebGPU IDL) so renderer.js's buffer/texture descriptors evaluate.
globalThis.GPUBufferUsage ??= {
  MAP_READ: 0x0001,
  MAP_WRITE: 0x0002,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  INDEX: 0x0010,
  VERTEX: 0x0020,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
  INDIRECT: 0x0100,
  QUERY_RESOLVE: 0x0200,
};
globalThis.GPUTextureUsage ??= {
  COPY_SRC: 0x01,
  COPY_DST: 0x02,
  TEXTURE_BINDING: 0x04,
  STORAGE_BINDING: 0x08,
  RENDER_ATTACHMENT: 0x10,
};
globalThis.GPUMapMode ??= { READ: 1, WRITE: 2 };
globalThis.GPUShaderStage ??= { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };

function makeGpuMock() {
  const buffers = [];
  const device = {
    features: [],
    pushErrorScope() {},
    popErrorScope: () => Promise.resolve(null),
    createShaderModule: (d) => ({ code: d.code }),
    createRenderPipeline: () => ({ getBindGroupLayout: () => ({}) }),
    createSampler: () => ({}),
    createTexture: () => ({ createView: () => ({}), destroy() {} }),
    createBindGroup: () => ({}),
    createCommandEncoder: () => ({
      beginRenderPass: () => ({
        setPipeline() {},
        setBindGroup() {},
        draw() {},
        end() {},
      }),
      finish: () => ({}),
    }),
    createBuffer: (d) => {
      const b = { size: d.size, data: new Uint8Array(d.size), destroy() {} };
      buffers.push(b);
      return b;
    },
    queue: {
      writeBuffer(buf, bufOff, data, dataOff = 0, size) {
        let src;
        if (data instanceof ArrayBuffer) {
          src = new Uint8Array(data, dataOff, size ?? data.byteLength - dataOff);
        } else {
          const bpe = data.BYTES_PER_ELEMENT || 1;
          src = new Uint8Array(
            data.buffer,
            data.byteOffset + dataOff * bpe,
            (size ?? data.length - dataOff) * bpe,
          );
        }
        buf.data.set(src, bufOff);
      },
      writeTexture() {},
      copyExternalImageToTexture() {},
      submit() {},
      onSubmittedWorkDone: () => Promise.resolve(),
    },
  };
  const adapter = {
    isFallbackAdapter: false,
    info: { vendor: "mock", description: "tierparity mock" },
    requestDevice: async () => device,
  };
  return { gpu: { requestAdapter: async () => adapter }, buffers };
}

async function makeWebGPUDriver() {
  const m = makeGpuMock();
  const r = await createRenderer(null, { gpu: m.gpu });
  const globalsBufs = m.buffers.filter(
    (b) => b.size === GLOBALS_WORDS_ALLOC * 16,
  );
  assert.equal(
    globalsBufs.length,
    1,
    "exactly one Globals-sized buffer expected — the snapshot target",
  );
  const gbuf = globalsBufs[0];
  return {
    write: (payload) => r.writeGlobals(payload),
    // Copy of the packed Globals bytes (u32 + f32 words alike).
    snap: () => gbuf.data.slice(),
  };
}

const OPS = [{ key: "boxFold", values: [1] }];

async function makeGLDriver() {
  const m = makeRecordingGL();
  const r = await createRendererGL(m.canvas);
  r.writeOps(OPS); // link is deferred to the first applyUniforms (#708)
  return {
    draw: (payload) => {
      r.writeGlobals(payload);
      m.beginDraw();
      r.draw();
      return m.snapshot();
    },
    values: m.values,
  };
}

// The full realistic payload capturesettle hands writeGlobals every frame.
const basePayload = () => ({
  res: [640, 360],
  cam: makeCamera({ yawDeg: 35, pitchDeg: 22, fovDeg: 42, dist: 4 }),
  iters: 12,
  opCount: 1,
  addC: false,
  maxSteps: 128,
  bailout: 64,
  eps: 6e-4,
  deScale: 0.85,
  colA: [0.86, 0.46, 0.18],
  colB: [0.18, 0.62, 0.74],
  bg: [0.07, 0.09, 0.15],
  colorMode: 1,
  stripeFreq: 5,
  deOption: 2,
  julia: false,
  juliaC: [0, 0, 0],
  palette: { on: false },
  light: {},
  objectCount: 0,
  tNear: 0.02,
  tFar: 80,
  sigLo: 0,
  sigSpan: 1,
  iridescence: 0,
  palettePhase: 0,
});

// ── the perturbation table ──────────────────────────────────────────────────
// name       diagnostic label (failures carry it)
// set(p)     mutate a fresh base payload
// expect     magic values that must land VERBATIM in both tiers' packed words
// webgpuOnly true = the field is WebGPU-only by structural absence — assert
//            the WGSL side changed, skip the GL side
const PERTURBATIONS = [
  { name: "colA", set: (p) => (p.colA = [0.111, 0.222, 0.333]), expect: [0.111, 0.222, 0.333] },
  { name: "colB", set: (p) => (p.colB = [0.121, 0.232, 0.343]), expect: [0.121, 0.232, 0.343] },
  { name: "bg", set: (p) => (p.bg = [0.131, 0.242, 0.353]), expect: [0.131, 0.242, 0.353] },
  { name: "addC → addGate", set: (p) => (p.addC = true) },
  {
    name: "julia + juliaC",
    set: (p) => {
      p.julia = true;
      p.juliaC = [0.311, 0.322, 0.333];
    },
    expect: [0.311, 0.322, 0.333],
  },
  { name: "colorMode", set: (p) => (p.colorMode = 3) },
  { name: "stripeFreq", set: (p) => (p.stripeFreq = 7), expect: [7] },
  { name: "deScale", set: (p) => (p.deScale = 0.441), expect: [0.441] },
  { name: "deOption", set: (p) => (p.deOption = 1) },
  {
    name: "tNear/tFar",
    set: (p) => {
      p.tNear = 0.051;
      p.tFar = 44.5;
    },
    expect: [0.051, 44.5],
  },
  {
    name: "cosine palette",
    set: (p) =>
      (p.palette = {
        on: true,
        a: [0.41, 0.42, 0.43],
        b: [0.44, 0.45, 0.46],
        c: [1.1, 1.2, 1.3],
        d: [0.11, 0.21, 0.31],
      }),
    expect: [0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 1.1, 1.2, 1.3, 0.11, 0.21, 0.31],
  },
  {
    name: "N-stop palette + cyclic",
    set: (p) =>
      (p.palette = {
        on: true,
        cyclic: true,
        stops: [
          { c: [1, 0, 0], p: 0 },
          { c: [0, 1, 0], p: 0.5 },
          { c: [0, 0, 1], p: 1 },
        ],
      }),
  },
  {
    name: "auto-levels sigLo/sigSpan",
    set: (p) => {
      p.sigLo = 0.211;
      p.sigSpan = 0.441;
    },
    expect: [0.211, 0.441],
  },
  { name: "iridescence", set: (p) => (p.iridescence = 0.313), expect: [0.313] },
  { name: "palettePhase", set: (p) => (p.palettePhase = 0.421), expect: [0.421] },
  {
    name: "light dir (drives fill/back dirs too)",
    set: (p) => (p.light.dir = [0.211, 0.522, 0.841]),
    expect: [0.211, 0.522, 0.841],
  },
  {
    name: "ambient/rim/gloss/intensity",
    set: (p) => {
      p.light.ambient = 0.271;
      p.light.rim = 0.611;
      p.light.gloss = 0.421;
      p.light.intensity = 1.31;
    },
    expect: [0.271, 0.611, 0.421, 1.31],
  },
  {
    name: "keyColor + metallic",
    set: (p) => {
      p.light.keyColor = [0.91, 0.81, 0.71];
      p.light.metallic = 0.351;
    },
    expect: [0.91, 0.81, 0.71, 0.351],
  },
  { name: "shadow → penumbra k + on-flag", set: (p) => (p.light.shadow = 0) },
  { name: "ao", set: (p) => (p.light.ao = 0.311), expect: [0.311] },
  {
    name: "fill + back rigs",
    set: (p) => {
      p.light.fill = 0.411;
      p.light.fillColor = [0.61, 0.62, 0.63];
      p.light.back = 0.321;
      p.light.backColor = [0.71, 0.72, 0.73];
    },
    expect: [0.411, 0.61, 0.62, 0.63, 0.321, 0.71, 0.72, 0.73],
  },
  { name: "sky macro (sky/sunGlow/ibl)", set: (p) => (p.light.sky = 0.611), expect: [0.611] },
  { name: "fog macro (fog/inScatter)", set: (p) => (p.light.fog = 0.511) },
  { name: "exposure", set: (p) => (p.light.exposure = 1.51), expect: [1.51] },
  {
    name: "envx stars (+density/seed)",
    set: (p) => {
      p.light.stars = 0.611;
      p.light.starDensity = 0.811;
      p.light.starSeed = 0.311;
    },
    expect: [0.611],
  },
  {
    name: "envx band (+tilt → bandDir)",
    set: (p) => {
      p.light.band = 0.511;
      p.light.bandTilt = 0.911;
    },
    expect: [0.511],
  },
  {
    name: "envx zenith color",
    set: (p) => (p.light.zenith = [0.11, 0.22, 0.93]),
    expect: [0.11, 0.22, 0.93],
  },
  {
    name: "aurora (+nebula/hue → both colors)",
    set: (p) => {
      p.light.aurora = 0.511;
      p.light.nebula = 0.411;
      p.light.auroraHue = 0.71;
    },
    expect: [0.511, 0.411],
  },
  { name: "neon gain", set: (p) => (p.light.neon = 0.511) },
  { name: "thin film", set: (p) => (p.light.thinFilm = 0.611), expect: [0.611] },
  {
    name: "cine grade block",
    set: (p) =>
      (p.light.grade = {
        strength: 0.71,
        contrast: 0.41,
        saturation: 1.51,
        shadowDesat: 0.21,
        shadowTint: [0.61, 0.51, 0.41],
        splitAmt: 0.31,
        hiTint: [0.42, 0.52, 0.62],
        duoAmt: 0.22,
        vignette: 0.32,
      }),
    expect: [0.71, 0.41, 1.51, 0.21],
  },
  {
    name: "clip plane (on + offset)",
    set: (p) => {
      p.light.clipOn = true;
      p.light.clipOffset = 0.611;
    },
    expect: [0.611],
  },
  {
    name: "clip plane geometry (axis + flip)",
    set: (p) => {
      p.light.clipOn = true;
      p.light.clipAxis = 2;
      p.light.clipFlip = true;
      p.light.clipOffset = 0.611;
    },
  },
  {
    name: "clip jagged (amp + Lipschitz divisor)",
    set: (p) => {
      p.light.clipOn = true;
      p.light.clipJag = 0.511;
    },
  },
  // ── WebGPU-only by structural absence (bloom passes, image textures, the
  // marched mirror) — RENDER_QUALITY.md; the GL tier never reads these.
  { name: "glow → bloom words", set: (p) => (p.light.glow = 0.711), webgpuOnly: true },
  {
    name: "env-map image sliders",
    set: (p) => {
      p.light.envMapAmount = 0.511;
      p.light.envMapBright = 2.1;
      p.light.envMapRot = 0.251;
    },
    webgpuOnly: true,
    expect: [0.511, 2.1, 0.251],
  },
  {
    name: "surface-texture sliders",
    set: (p) => {
      p.light.surfTexAmount = 0.511;
      p.light.surfTexScale = 2.1;
    },
    webgpuOnly: true,
    expect: [0.511, 2.1],
  },
  {
    name: "self-reflection words",
    set: (p) => {
      p.light.reflBounces = 2;
      p.light.reflectivity = 0.51;
      p.light.reflFresnel = 0.71;
      p.light.reflTint = 0.61;
    },
    webgpuOnly: true,
    expect: [2, 0.51, 0.71, 0.61],
  },
  // ── shared RAW payload words outside deriveFrameParams that both tiers
  // still must pack (ray-gen controls).
  { name: "orthoH (raw payload word)", set: (p) => (p.orthoH = 0.811), expect: [0.811] },
  { name: "planetK (raw payload word)", set: (p) => (p.planetK = 0.511), expect: [0.511] },
  // equirect: the GL tier bakes the longitude scale into the program TEXT
  // (no uniform), so its packed-state change is the relink — the snapshot
  // includes the live program source for exactly this case.
  { name: "equirectS (raw payload word)", set: (p) => (p.equirectS = 1.21) },
];

const fr = Math.fround;

test("cross-tier pack parity: every shading change lands in BOTH tiers' packed state", async (t) => {
  for (const P of PERTURBATIONS) {
    await t.test(P.name, async () => {
      const pert = basePayload();
      P.set(pert);

      // WebGPU tier — the packed Globals bytes must change.
      const w = await makeWebGPUDriver();
      w.write(basePayload());
      const wA = w.snap();
      w.write(pert);
      const wB = w.snap();
      assert.ok(
        Buffer.compare(Buffer.from(wA), Buffer.from(wB)) !== 0,
        `WGSL Globals identical under "${P.name}" — the tier dropped the ` +
          "field (the 7494398 palettePhase/iridescence class)",
      );
      if (P.expect) {
        const words = new Float32Array(
          wB.buffer,
          wB.byteOffset,
          wB.byteLength / 4,
        );
        for (const v of P.expect)
          assert.ok(
            words.includes(fr(v)),
            `WGSL Globals missing packed value ${v} for "${P.name}"`,
          );
      }

      // WebGL2 tier — the recorded uniform uploads / Bulk UBO / live program
      // must change (unless the field is WebGPU-only by structural absence).
      if (P.webgpuOnly) return;
      const g = await makeGLDriver();
      const gA = g.draw(basePayload());
      const gB = g.draw(pert);
      assert.notEqual(
        gB,
        gA,
        `GL uniform state identical under "${P.name}" — the tier dropped ` +
          "the field (the class the old renderer.js-only regex could not see)",
      );
      if (P.expect) {
        const vals = g.values();
        for (const v of P.expect)
          assert.ok(
            vals.includes(fr(v)),
            `GL uploads missing packed value ${v} for "${P.name}"`,
          );
      }
    });
  }
});

// ── the honesty gate ────────────────────────────────────────────────────────
// Constants deriveFrameParams returns that NO payload can move — each is a
// deliberate one-source-of-truth word (documented at its derivation). Anything
// else left uncovered fails: a new derived field must gain a perturbation
// above before this suite goes green, which is what makes the parity guard
// self-maintaining.
const DERIVED_CONSTANTS = [
  "ground", // fixed ground dim (0.35)
  "clipShade", // reserved cut-face gain, no UI writer yet
  "auroraDrift", // reserved 0 — no per-frame writer yet
  "clipJagFreq", // CLIP_JAG_FREQ constant (one source, three tiers)
  "bloomThreshold", // fixed pre-tonemap HDR threshold (1.0)
];

test("meta: every deriveFrameParams key is moved by a perturbation (or is a documented constant)", () => {
  const d0 = deriveFrameParams(basePayload());
  const covered = new Set();
  for (const P of PERTURBATIONS) {
    const p = basePayload();
    P.set(p);
    const d1 = deriveFrameParams(p);
    for (const k of Object.keys(d0))
      if (JSON.stringify(d0[k]) !== JSON.stringify(d1[k])) covered.add(k);
  }
  const missing = Object.keys(d0).filter(
    (k) => !covered.has(k) && !DERIVED_CONSTANTS.includes(k),
  );
  assert.deepEqual(
    missing,
    [],
    `derived key(s) ${JSON.stringify(missing)} have NO parity perturbation — ` +
      "add one to PERTURBATIONS (or, for a genuine constant, the ledger) so " +
      "the next palettePhase cannot ship dark on one tier",
  );
  // …and the ledger itself stays honest: a constant that BECOMES payload-
  // driven must leave it.
  const stale = DERIVED_CONSTANTS.filter((k) => covered.has(k));
  assert.deepEqual(
    stale,
    [],
    `ledger key(s) ${JSON.stringify(stale)} are payload-driven now — remove from DERIVED_CONSTANTS`,
  );
});

// The two tiers pack from ONE derivation (frameparams charter): sanity-pin
// that the base payload derives identically across two calls — a derivation
// that consults hidden state would quietly invalidate every parity result.
test("meta: deriveFrameParams is pure on the parity payload", () => {
  assert.deepEqual(
    deriveFrameParams(basePayload()),
    deriveFrameParams(basePayload()),
  );
});

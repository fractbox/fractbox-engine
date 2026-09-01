// Field streamlines (lab) — static gates for the DE-reuse codegen
// (core/streamlines.js). No GPU in CI, so these pin the TEXT contracts:
//   1. extractDEWGSL slices the emitted march WGSL at the end of fn mapDE —
//      the whole DE cluster survives, no render entry point leaks through.
//   2. buildStreamSimWGSL assembles a compute module that carries the emitted
//      map function (the one hard requirement: reuse, never duplicate) and
//      only group-1 sim bindings of its own.
//   3. streamSimFeat forces every render-only codegen flag off, so the sim
//      variant key-space stays the {numeric, scene, hybrid, morph, ops,
//      leaves} slice of the march key.
//
// Run: node --test core/streamlines.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";
import { buildWGSL } from "./shader.js";
import { PRESETS } from "./oplist.js";
import { OPERATORS } from "./operators.js";
import {
  extractDEWGSL,
  buildStreamSimWGSL,
  buildStreamDrawWGSL,
  buildStreamBlitWGSL,
  streamSimFeat,
  streamSimKey,
  streamSeedFor,
  createStreamlines,
  STREAM_MAX,
  STREAM_PARTICLE_STRIDE,
  STREAM_PREROLL_STEPS,
  STREAM_PREROLL_DT,
} from "./streamlines.js";

// ── A stub WebGPU device, just deep enough to record what the controller
// SUBMITS. There is no GPU in CI, but the deterministic-export contract is a
// claim about the command/uniform stream — which is observable right here, and
// is the half a headless render probe can't isolate (a render tells you the
// frames matched, not that the seed, the clock and the pre-roll were the reason).
// ensureBase reads the WebGPU usage-flag enum off the global; Node has none.
// The values are opaque to the controller (it only ORs them into createBuffer
// descriptors the stub ignores), so any distinct set will do.
globalThis.GPUBufferUsage ??= {
  STORAGE: 128,
  COPY_DST: 8,
  COPY_SRC: 4,
  UNIFORM: 64,
  MAP_READ: 1,
};
function stubDevice() {
  const writes = []; // queue.writeBuffer payloads, in order
  const copies = []; // copyBufferToBuffer (the park/restore snapshots)
  const dispatches = []; // dispatchWorkgroups counts (one per advect pass)
  const pass = {
    setPipeline() {},
    setBindGroup() {},
    dispatchWorkgroups(n) {
      dispatches.push(n);
    },
    draw() {},
    end() {},
  };
  const enc = () => ({
    beginComputePass: () => pass,
    beginRenderPass: () => pass,
    copyBufferToBuffer: (src, so, dst, dof, size) => copies.push({ size }),
    copyTextureToTexture() {},
    finish: () => ({}),
  });
  const device = {
    createBuffer: (d) => ({ size: d.size, usage: d.usage, destroy() {} }),
    createShaderModule: () => ({
      getCompilationInfo: async () => ({ messages: [] }),
    }),
    createBindGroup: () => ({}),
    createComputePipelineAsync: async () => ({ getBindGroupLayout: () => ({}) }),
    createRenderPipelineAsync: async () => ({ getBindGroupLayout: () => ({}) }),
    createCommandEncoder: enc,
    queue: {
      writeBuffer(buf, off, data) {
        writes.push(data.slice ? data.slice() : data);
      },
      submit() {},
    },
  };
  return { device, writes, copies, dispatches };
}
const stubTex = () => ({ width: 8, height: 8, createView: () => ({}) });
// The sim uniform is the only 16-lane f32 write the controller makes (draw is 4).
const simWrites = (writes) =>
  writes.filter((w) => w instanceof Float32Array && w.length === 16);
const zeroFills = (writes) => writes.filter((w) => w instanceof Uint8Array);
// Build a warmed controller: the sim pipeline resolves through an async
// microtask, so encode once and let it land before measuring anything.
async function warmed(count = 128) {
  const s = stubDevice();
  const io = {
    format: "bgra8unorm",
    globalsBuf: {},
    opsBuf: {},
    objectsBuf: {},
    opAuxBuf: {},
    objAuxBuf: {},
  };
  const ctl = createStreamlines(s.device, io);
  ctl.set({ on: true, count });
  ctl.encodeOffline(s.device.createCommandEncoder(), stubTex(), {}, 1 / 30);
  for (let i = 0; i < 8; i++) await Promise.resolve();
  s.writes.length = 0;
  s.copies.length = 0;
  s.dispatches.length = 0;
  return { ctl, ...s };
}
// One export session: arm, render `frames` offline frames, disarm.
function exportSession(ctl, dev, key, frames = 3, dt = 1 / 30) {
  ctl.beginOffline(key);
  for (let i = 0; i < frames; i++)
    ctl.encodeOffline(dev.createCommandEncoder(), stubTex(), {}, dt);
  ctl.endOffline();
}

const braceBalance = (src) => {
  let depth = 0;
  for (const ch of src) {
    if (ch === "{") depth++;
    else if (ch === "}") depth--;
  }
  return depth;
};

// The march variant shapes the sim must mirror (renderer.js activeFeat slices).
const VARIANTS = [
  ["default", {}],
  ["analytic", { numericDE: false }],
  ["hybrid", { hybrid: true }],
  ["morph", { morph: true }],
  ["scene+leaves", { scene: true, leaves: [1, 2] }],
  ["specialized ops", { ops: [1, 2, 7] }],
  ["empty op-set", { ops: [] }],
];

test("extractDEWGSL: the DE cluster survives, render text does not", () => {
  for (const [name, opts] of VARIANTS) {
    const full = buildWGSL(opts);
    const de = extractDEWGSL(full);
    assert.ok(full.startsWith(de.slice(0, -1)), `${name}: a clean prefix`);
    assert.equal(braceBalance(de), 0, `${name}: balanced braces`);
    // The cluster: bindings + the scene walker + the single map + mapDE itself.
    for (const s of [
      "struct Globals",
      "@group(0) @binding(0) var<uniform> G : Globals;",
      "@group(0) @binding(1) var<storage, read> ops : array<Op>;",
      "@group(0) @binding(2) var<storage, read> objects : array<Obj>;",
      "fn mapDE_single(",
      "fn mapDE(p_rel: vec3f) -> f32 {",
    ]) {
      assert.ok(de.includes(s), `${name}: DE prefix must contain "${s}"`);
    }
    // Nothing render-only may leak into a compute module: entry points,
    // texture bindings, the coloring orbits, the post-mapDE scene tint.
    for (const s of ["@vertex", "@fragment", "fn fs(", "fn sceneTint("]) {
      assert.ok(!de.includes(s), `${name}: DE prefix must NOT contain "${s}"`);
    }
    // The slice ends AT mapDE: exactly one text occurrence of its declaration,
    // and it is the LAST function declared.
    const decl = de.lastIndexOf("\nfn ");
    assert.ok(
      de.slice(decl).startsWith("\nfn mapDE(p_rel: vec3f) -> f32 {"),
      `${name}: mapDE must be the final function in the slice`,
    );
  }
});

test("extractDEWGSL: op specialization carries through to the sim source", () => {
  const ids = [1, 7];
  const de = extractDEWGSL(buildWGSL({ ops: ids }));
  for (const id of ids)
    assert.ok(de.includes(`case ${id}u:`), `case ${id}u present`);
  // An op OUTSIDE the set must have no case — the specialization contract.
  const absent = OPERATORS.find((o) => !ids.includes(o.id));
  assert.ok(absent, "registry has an op outside the set");
  assert.ok(
    !de.includes(`case ${absent.id}u:`),
    `case ${absent.id}u (op "${absent.key}") must be dropped`,
  );
});

test("buildStreamSimWGSL: compute module = emitted DE + group-1 sim tail", () => {
  const sim = buildStreamSimWGSL(extractDEWGSL(buildWGSL({})));
  assert.equal(braceBalance(sim), 0, "balanced braces");
  for (const s of [
    "@compute @workgroup_size(64)",
    "fn advect(@builtin(global_invocation_id) gid: vec3u)",
    "@group(1) @binding(0) var<storage, read_write> streamPs",
    "@group(1) @binding(1) var<uniform> SS : StreamSimU;",
    "fn mapDE(p_rel: vec3f) -> f32 {", // THE reused map — not a copy
    "fn slSpawn(",
  ]) {
    assert.ok(sim.includes(s), `sim module must contain "${s}"`);
  }
  assert.ok(!sim.includes("@vertex"), "no render entry points");
  assert.ok(!sim.includes("@fragment"), "no render entry points");
  // The motion recipe: 7 field taps (the value + the 6-tap central-difference
  // stencil) + the occlusion march's single in-loop call site.
  const tail = sim.slice(sim.indexOf("fn advect("));
  assert.equal(
    (tail.match(/mapDE\(/g) || []).length,
    8,
    "advect samples mapDE at 8 sites: d + 6 stencil taps + occlusion trace",
  );
  // The sim never claims a group-0 binding of its own — group 0 stays the
  // march's binding plan (Globals/ops/objects/+aux), shared buffers unchanged.
  const simTail = sim.slice(sim.indexOf("Field streamlines sim"));
  assert.ok(
    !/@group\(0\)/.test(simTail),
    "sim tail must not declare group-0 bindings",
  );
});

test("R3 motion recipe + occlusion: the sim tail carries the probe-tuned upgrades", () => {
  const sim = buildStreamSimWGSL(extractDEWGSL(buildWGSL({})));
  // DE-scale-NORMALIZED curvature variation from the same six taps (the v0.1
  // un-normalized flat-gate measured dead on real box fields — CPU sweep).
  assert.match(sim, /abs\(dpx \+ dmx - 2\.0 \* d0\)/, "per-axis second diff");
  assert.match(sim, /let rel = d2 \/ \(gl \* h\);/, "normalized by gl*h");
  assert.match(sim, /let edge = clamp\(SS\.edgeK \* rel/, "edge emphasis");
  assert.ok(!sim.includes("exp(-0.3 * curv)"), "the dead flat-gate is gone");
  // Face-coherent current: shared cross(n, world-up) frame, +/- per particle,
  // blended by SS.align — kills the per-particle "rain" on flat faces.
  assert.match(sim, /let fd0 = cross\(n, upA\);/, "face-stable frame");
  assert.match(sim, /clamp\(SS\.align, 0\.0, 1\.0\)/, "align blend");
  // Divergence-free curl braid, surface-projected, ALWAYS blended (no flat
  // detector exists on box fields to gate it on).
  assert.ok(sim.includes("fn slCurlV("), "curl field present");
  assert.match(sim, /cv = cv - n \* dot\(cv, n\);/, "surface-projected braid");
  // Anti-congregation (R3 "dots congregate"): floored slowdown, faster aging
  // when slow, and a sustained-stagnation respawn counter in misc.z.
  assert.match(sim, /vt = vt \* max\(0\.55, 1\.0 - SS\.edgeAcc \* edge\);/);
  assert.match(sim, /let ageDt = dt \* \(1\.0 \+ 1\.5 \* \(1\.0 - srel\)\);/);
  assert.match(
    sim,
    /if \(slowN > 40\.0\) \{ streamPs\[i\] = slSpawn\(i\); return; \}/,
  );
  // Wide spawn shell (inner radius = half the outer), not an origin-heavy ball.
  assert.match(sim, /pow\(0\.125 \+ 0\.875 \* slRand\(b \+ 2u\), 0\.3333\)/);
  // World scale (the probe's Mandelbox find: presets span ~7× in world size;
  // a fixed sim world buried every Mandelbox particle inside the body). Knobs
  // stay unit-scale; the shader applies SS.wscale itself, so the preview's
  // auto radius probe can never clobber an explicitly tuned knob.
  assert.match(sim, /let shellW = SS\.shell \* SS\.wscale;/);
  assert.match(sim, /let swirlW = SS\.swirl \* SS\.wscale;/);
  assert.match(sim, /let boundW = SS\.boundR \* SS\.wscale;/);
  assert.match(
    sim,
    /SS\.curlFreq \/ SS\.wscale/,
    "braids-per-object frequency",
  );
  // Occlusion marches FROM THE PARTICLE TOWARD THE EYE (t = distance from the
  // particle — the standard soft-shadow geometry; dividing by distance-from-
  // eye faded the whole overlay to near-invisible: census 16292/16328 at
  // vis<0.1 on Mandelbox), with the conservative-DE gl correction and the
  // grazing-angle shell allowance (whole oblique faces faded without it).
  assert.match(sim, /let pv = ro - np;/, "particle->eye ray");
  assert.match(sim, /var t = margin;/, "starts a margin off the particle");
  assert.match(sim, /let occAmp = 1\.0 \/ max\(gl, 0\.2\);/, "gl correction");
  assert.match(sim, /let occEps = 0\.75 \* shellW;/, "shell allowance");
  assert.match(sim, /clamp\(1\.0 - t \/ \(24\.0 \* shellW\), 0\.0, 1\.0\)/);
  assert.match(sim, /vis = min\(vis, clamp\(12\.0 \* c \/ t, 0\.0, 1\.0\)\);/);
  assert.match(sim, /let margin = max\(2\.5 \* shellW, 0\.03 \* SS\.wscale\);/);
  // Converging spawn dust is dimmed until it reaches the shell.
  assert.match(
    sim,
    /vis = vis \/ \(1\.0 \+ \(6\.0 \/ SS\.wscale\) \* max\(0\.0, d0 - shellW\)\);/,
  );
  // The sim uniform carries the six tuning words.
  for (const f of ["curl", "curlFreq", "edgeAcc", "edgeK", "align", "wscale"])
    assert.ok(
      new RegExp(`${f}\\s*:\\s*f32`).test(sim),
      `StreamSimU carries ${f}`,
    );
});

test("particle struct: WGSL layout and JS stride agree (sim, draw, allocation)", () => {
  const decl = "struct StreamP { posAge: vec4f, axHue: vec4f, misc: vec4f };";
  assert.ok(buildStreamSimWGSL(extractDEWGSL(buildWGSL({}))).includes(decl));
  assert.ok(buildStreamDrawWGSL().includes(decl));
  assert.equal(STREAM_PARTICLE_STRIDE, 3 * 16, "3 vec4f rows");
  // Occlusion + edge signals actually reach the shading: visibility fades the
  // sprite, the crease signal brightens it, dead/occluded quads are skipped.
  const draw = buildStreamDrawWGSL();
  assert.match(draw, /o\.fade = clamp\(age, 0\.0, 1\.0\) \* P\.misc\.x;/);
  assert.match(draw, /1\.0 \+ 0\.6 \* P\.misc\.y/);
  assert.match(draw, /P\.misc\.x <= 0\.004/);
});

test("sim source builds for every march variant shape", () => {
  for (const [name, opts] of VARIANTS) {
    const sim = buildStreamSimWGSL(extractDEWGSL(buildWGSL(opts)));
    assert.equal(braceBalance(sim), 0, `${name}: balanced`);
    assert.ok(sim.includes("fn advect("), `${name}: entry point present`);
  }
});

test("sim source builds from every preset's op-set (specialized, like the renderer)", () => {
  let built = 0;
  for (const p of PRESETS) {
    if (p.objects || !Array.isArray(p.ops)) continue;
    const ids = [
      ...new Set(
        p.ops
          .map((o) => OPERATORS.find((d) => d.key === o.key)?.id)
          .filter((id) => id != null),
      ),
    ];
    const sim = buildStreamSimWGSL(extractDEWGSL(buildWGSL({ ops: ids })));
    assert.equal(braceBalance(sim), 0, `${p.name}: balanced`);
    assert.ok(sim.includes("fn advect("), `${p.name}: entry point`);
    built++;
  }
  assert.ok(built >= 50, `expected the preset catalog, built ${built}`);
});

test("streamSimFeat: render-only flags are forced off; the DE slice survives", () => {
  const f = streamSimFeat({
    numericDE: true,
    leaves: [3],
    coloring: true,
    scene: true,
    hybrid: true,
    morph: true,
    ops: [1, 2],
    df64: true,
    perturb: true,
    envx: true,
    envMap: true,
    surfTex: true,
    sreflect: true,
  });
  assert.deepEqual(
    {
      numericDE: f.numericDE,
      scene: f.scene,
      hybrid: f.hybrid,
      morph: f.morph,
      ops: f.ops,
      leaves: f.leaves,
    },
    {
      numericDE: true,
      scene: true,
      hybrid: true,
      morph: true,
      ops: [1, 2],
      leaves: [3],
    },
  );
  for (const k of [
    "coloring",
    "df64",
    "perturb",
    "envx",
    "envMap",
    "surfTex",
    "sreflect",
    "capture",
  ]) {
    assert.equal(f[k], false, `${k} must be forced off`);
  }
  // …and buildWGSL accepts the derived options (perturb/df64 exclusivity etc.).
  assert.doesNotThrow(() => buildWGSL(streamSimFeat({ df64: true })));
});

test("streamSimKey: keyed on the sim-relevant slice only", () => {
  const base = { numericDE: true, ops: [1, 2], leaves: null };
  const k = streamSimKey(base);
  // Render-only flag changes must NOT rebuild the sim pipeline…
  assert.equal(streamSimKey({ ...base, envx: true, sreflect: true }), k);
  assert.equal(streamSimKey({ ...base, df64: true }), k);
  // …but the DE-shaping inputs must.
  assert.notEqual(streamSimKey({ ...base, ops: [1, 2, 3] }), k);
  assert.notEqual(streamSimKey({ ...base, hybrid: true }), k);
  assert.notEqual(streamSimKey({ ...base, numericDE: false }), k);
  // ops: [] (specialized-to-zero) and ops: null (full switch) emit different
  // shaders — they must never share a pipeline (the renderer's #265 rule).
  assert.notEqual(
    streamSimKey({ ...base, ops: [] }),
    streamSimKey({ ...base, ops: null }),
  );
});

test("draw/blit modules: self-contained render passes, march camera contract", () => {
  const draw = buildStreamDrawWGSL();
  assert.equal(braceBalance(draw), 0);
  for (const s of [
    "@vertex fn vs(",
    "@fragment fn fs(",
    "instance_index",
    "G.camFwd.xyz", // projects through the march camera basis…
    "tan(0.5 * G.res.z)", // …with the fs ray-gen's exact fov term
    "G.palA.w > 0.5", // march cosine-palette gate, same words
  ]) {
    assert.ok(draw.includes(s), `draw module must contain "${s}"`);
  }
  assert.ok(!draw.includes("mapDE"), "the draw pass needs no DE");
  const blit = buildStreamBlitWGSL();
  assert.equal(braceBalance(blit), 0);
  assert.ok(blit.includes("textureLoad(streamSrc"));
  assert.ok(STREAM_MAX >= 16384, "v0 target: at least 16k particles");
});

test("off-path safety: extraction reads buildWGSL output, never mutates inputs", () => {
  const a = buildWGSL({});
  const before = a;
  extractDEWGSL(a);
  assert.equal(a, before);
  // And the default march build itself is what the extraction was pinned
  // against — a marker drift here means extractDEWGSL needs re-verifying.
  assert.ok(a.includes("\nfn mapDE(p_rel: vec3f) -> f32 {"));
});

// ── Deterministic offline exports (the PR #656 v1 tail) ──────────────────────

test("streamSeedFor: integer-only, stable, and key-sensitive", () => {
  // Pinned value — a change here re-rolls every export's particle field, which
  // is exactly the kind of silent drift the pin exists to make loud.
  assert.equal(streamSeedFor(""), 0x811c9dc5);
  assert.equal(streamSeedFor("a"), streamSeedFor("a"));
  assert.notEqual(streamSeedFor("Mandelbox#12"), streamSeedFor("Mandelbox#13"));
  // Always a u32 — the value goes into an f32 uniform lane and is cast back to
  // u32 in WGSL, so a negative or fractional result would be nonsense.
  for (const k of ["", "x", "Menger#8#boxFold:1,1,1", "ÿĀ"]) {
    const v = streamSeedFor(k);
    assert.ok(Number.isInteger(v) && v >= 0 && v <= 0xffffffff, k);
  }
  assert.ok(
    STREAM_PREROLL_STEPS >= 30 && Number.isInteger(STREAM_PREROLL_STEPS),
  );
  assert.ok(STREAM_PREROLL_DT > 0 && STREAM_PREROLL_DT <= 0.1);
});

test("offline export: the same key replays the same sim stream, byte for byte", async () => {
  const a = await warmed();
  exportSession(a.ctl, a.device, "FORMULA-KEY");
  const runA = simWrites(a.writes);

  const b = await warmed();
  exportSession(b.ctl, b.device, "FORMULA-KEY");
  const runB = simWrites(b.writes);

  assert.ok(
    runA.length > STREAM_PREROLL_STEPS,
    "pre-roll + frames were encoded",
  );
  assert.equal(runA.length, runB.length);
  for (let i = 0; i < runA.length; i++)
    assert.deepEqual([...runA[i]], [...runB[i]], `sim uniform step ${i}`);
});

test("offline export: two sessions in ONE controller replay identically too", async () => {
  // The stronger claim — the second export must not inherit anything from the
  // first (v0's whole failure mode was carried-over particle phase).
  const a = await warmed();
  exportSession(a.ctl, a.device, "KEY");
  const first = simWrites(a.writes).map((w) => [...w]);
  a.writes.length = 0;
  exportSession(a.ctl, a.device, "KEY");
  const second = simWrites(a.writes).map((w) => [...w]);
  assert.deepEqual(first, second);
});

test("offline export: the seed lane tracks the key AND the knobs", async () => {
  const seedLane = (writes) => simWrites(writes).map((w) => w[7]);
  const a = await warmed();
  exportSession(a.ctl, a.device, "ONE");
  const one = seedLane(a.writes);
  a.writes.length = 0;
  exportSession(a.ctl, a.device, "TWO");
  assert.notDeepEqual(
    seedLane(a.writes),
    one,
    "another formula, another stream",
  );

  // Same key, different knobs → a different picture, so a different stream.
  a.writes.length = 0;
  a.ctl.set({ swirl: 1.4 });
  exportSession(a.ctl, a.device, "ONE");
  assert.notDeepEqual(seedLane(a.writes), one);
});

test("offline export: arming reseeds, pre-rolls, and parks/restores the live field", async () => {
  const a = await warmed();
  const FRAMES = 3;
  exportSession(a.ctl, a.device, "KEY", FRAMES);

  // Reset: exactly one zero fill, sized to the LIVE particle count (not the
  // 65536 ceiling) — that is the whole init path, age <= 0 being the shader's
  // own respawn signal.
  const zeros = zeroFills(a.writes);
  assert.equal(zeros.length, 1);
  assert.equal(zeros[0].byteLength, 128 * STREAM_PARTICLE_STRIDE);
  assert.ok(zeros[0].every((b) => b === 0));

  // Pre-roll: STREAM_PREROLL_STEPS advect passes ahead of the FRAMES exported
  // ones, each on its own uniform write (batching them into one submit would
  // collapse every step onto the last write — see resetAndPreroll).
  assert.equal(a.dispatches.length, STREAM_PREROLL_STEPS + FRAMES);
  const sims = simWrites(a.writes);
  assert.equal(sims.length, STREAM_PREROLL_STEPS + FRAMES);
  // (fround: the uniform is an f32 lane, so the value that comes back is the
  // f32 image of 1/30, not the f64 constant.)
  const dtF32 = Math.fround(STREAM_PREROLL_DT);
  assert.ok(sims.slice(0, STREAM_PREROLL_STEPS).every((w) => w[0] === dtF32));
  // …and the clock advances monotonically across them, from 0.
  const times = sims.map((w) => w[1]);
  assert.ok(Math.abs(times[0] - STREAM_PREROLL_DT) < 1e-6);
  for (let i = 1; i < times.length; i++) assert.ok(times[i] > times[i - 1]);

  // Park + restore: one full-buffer copy each way.
  assert.equal(a.copies.length, 2);
  for (const c of a.copies)
    assert.equal(c.size, STREAM_MAX * STREAM_PARTICLE_STRIDE);
});

test("offline export: UNARMED captureFrame keeps the v0 shared-flow behavior", async () => {
  // A bare one-off capture (a thumbnail-grade frame, no export session) must
  // not pay a 90-step pre-roll or wipe the live field — WYSIWYG is the right
  // answer for a single frame.
  const a = await warmed();
  for (let i = 0; i < 3; i++)
    a.ctl.encodeOffline(a.device.createCommandEncoder(), stubTex(), {}, 1 / 30);
  assert.equal(a.dispatches.length, 3);
  assert.equal(zeroFills(a.writes).length, 0);
  assert.equal(a.copies.length, 0);
  assert.equal(a.ctl.offline, false);
});

test("offline export: the session flag is honest, and disarming is safe unpaired", async () => {
  const a = await warmed();
  assert.equal(a.ctl.offline, false);
  a.ctl.beginOffline("K");
  assert.equal(a.ctl.offline, true);
  assert.equal(a.ctl.info().offline, true);
  a.ctl.endOffline();
  assert.equal(a.ctl.offline, false);
  a.ctl.endOffline(); // unpaired — must not throw or copy anything back
  assert.equal(a.copies.length, 2);
  // Overlay off ⇒ nothing to arm (a splat capture / thumbnail run under the
  // same setOffline bracket must stay zero-cost).
  a.ctl.set({ on: false });
  assert.equal(a.ctl.beginOffline("K"), false);
});

test("deep zoom: the world multiplier accepts frustum-sized values", () => {
  // At x1000 the frustum half-height is far below the old 0.1 floor, which
  // pinned the spawn ball thousands of frustums wide and put every particle
  // off screen. Clamps still hold at both ends.
  const { device } = stubDevice();
  const ctl = createStreamlines(device, {
    format: "bgra8unorm",
    globalsBuf: {},
    opsBuf: {},
    objectsBuf: {},
    opAuxBuf: {},
    objAuxBuf: {},
  });
  ctl.set({ on: true, wscale: 0.0092 });
  assert.equal(ctl.info().wscale, 0.0092);
  ctl.set({ wscale: 1e-9 });
  assert.equal(ctl.info().wscale, 1e-4);
  ctl.set({ wscale: 1e6 });
  assert.equal(ctl.info().wscale, 64);
});

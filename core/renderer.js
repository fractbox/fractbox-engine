// WebGPU renderer. Owns the device, the pipeline (built once from the
// generated WGSL), and two GPU buffers:
//   - a 176-byte uniform "Globals" (camera + control scalars; 11 * vec4)
//   - a storage "ops" buffer (the op-list as packed Op structs)
// Editing param values or reordering ops only rewrites the ops buffer; the
// pipeline is never rebuilt at runtime.

import { buildWGSL, buildPostWGSL, buildAccumWGSL, buildBloomWGSL, JITTER_WORD, DOF_WORD } from "./shader.js";
import { byKey } from "./operators.js";
import { computeRecenter } from "./recenter.js";
import { MAX_OPS_WEBGPU, MAX_OBJECTS } from "./limits.js";
import { eulerToQuat } from "./quat.js";

// PoC (CSG Phase 1a): the ops buffer is shared across all objects in a scene, so
// it's sized for MAX_OBJECTS * MAX_OPS_PER_OBJECT (8 * 24 = 192) and the scene
// writer bounds-checks the concatenated total (throws, never silently truncates).
const MAX_OPS = MAX_OPS_WEBGPU; // op-buffer capacity (192 * 16 = 3 KiB)
const OP_STRIDE = 16; // bytes per Op (u32 + 3*f32)
const OBJ_STRIDE = 96; // bytes per Obj (24 words: 4 u32 + 20 f32 — see shader.js Obj)

const GLOBALS_BYTES = 560; // 35 * vec4 (16B each) — …, post (P0), lightC/… (P1), jitter (P2), env/fog (P3), dof (P4). Packed, cumulative, no anonymous padding.

// P4 lens samples — a SECOND low-discrepancy stream (golden-ratio pair,
// decorrelated from the subpixel R2) mapped to the unit disk.
export function lensSample(i) {
  const f = (x) => x - Math.floor(x);
  const r = Math.sqrt(f(0.5 + i * 0.6180339887498949));
  const th = 2 * Math.PI * f(0.5 + i * 0.38196601125010515);
  return [r * Math.cos(th), r * Math.sin(th)];
}
const ACCUM_FORMAT = "rgba32float"; // P2 running average — renderable (unblended) + textureLoad, core WebGPU

// R2 low-discrepancy sequence (Roberts) → subpixel jitter in [-0.5, 0.5)².
// Deterministic in the sample index — offline export replay stays bit-stable.
export function r2jitter(i) {
  const f = (x) => x - Math.floor(x);
  return [f(0.5 + i * 0.7548776662466927) - 0.5, f(0.5 + i * 0.5698402909980532) - 0.5];
}
const HDR_FORMAT = "rgba16float"; // P0 march target (core-renderable, no optional features)

export async function createRenderer(canvas) {
  if (!navigator.gpu) throw new Error("WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("no GPU adapter");
  const device = await adapter.requestDevice();
  const ctx = canvas.getContext("webgpu");
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: "opaque" });

  // March pipeline VARIANTS (perf): the numeric-DE path (deOption 3) inlines
  // its orbitR probe — a full op-switch — 4× into mapDE_single, and that
  // register pressure taxes every analytic formula ~7-30% (measured @2400²).
  // So the analytic-only variant is the eager default, and the numeric one is
  // compiled lazily the first time a frame actually carries deOption ≥ 2.5
  // (writeGlobals records it; the encode sites pick the active variant).
  async function makeMarchVariant(numericDE) {
    const module = device.createShaderModule({ code: buildWGSL({ numericDE }) });
    // Surface compile errors loudly (best-effort; not all browsers populate this).
    const info = await module.getCompilationInfo?.();
    if (info) {
      for (const m of info.messages) {
        const line = `WGSL ${m.type} @${m.lineNum}:${m.linePos} — ${m.message}`;
        if (m.type === "error") console.error(line);
        else console.warn(line);
      }
    }
    return module;
  }
  const module = await makeMarchVariant(false);

  // P0 two-pass spine: the march pass renders linear HDR into an rgba16float
  // intermediate; the post pass (tone map + exact sRGB encode + dither, see
  // shader.js buildPostWGSL) resolves it to the 8-bit target. All three draw
  // entry points (draw / drawTo / renderToImage) share the sequence.
  const marchPipe = (m) =>
    device.createRenderPipeline({
      layout: "auto",
      vertex: { module: m, entryPoint: "vs" },
      fragment: { module: m, entryPoint: "fs", targets: [{ format: HDR_FORMAT }] },
      primitive: { topology: "triangle-list" },
    });
  const pipeline = marchPipe(module);
  const postModule = device.createShaderModule({ code: buildPostWGSL() });
  const postPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: postModule, entryPoint: "vs" },
    fragment: { module: postModule, entryPoint: "fs", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });
  // P2 accumulation pass: mix(prev, cur, weight) into the other ping-pong half.
  const accumModule = device.createShaderModule({ code: buildAccumWGSL() });
  const accumPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: accumModule, entryPoint: "vs" },
    fragment: { module: accumModule, entryPoint: "fs", targets: [{ format: ACCUM_FORMAT }] },
    primitive: { topology: "triangle-list" },
  });
  // P3 bloom: bright(threshold+downsample) → blurH → blurV at half res.
  const bloomModule = device.createShaderModule({ code: buildBloomWGSL() });
  const bloomPipe = (entry) =>
    device.createRenderPipeline({
      layout: "auto",
      vertex: { module: bloomModule, entryPoint: "vs" },
      fragment: { module: bloomModule, entryPoint: entry, targets: [{ format: HDR_FORMAT }] },
      primitive: { topology: "triangle-list" },
    });
  const brightPipeline = bloomPipe("fs_bright");
  const blurHPipeline = bloomPipe("fs_blurH");
  const blurVPipeline = bloomPipe("fs_blurV");
  const bloomSampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });
  let bloomOn = false; // set by writeGlobals from the fog word — gates the passes

  const globalsBuf = device.createBuffer({
    size: GLOBALS_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const opsBuf = device.createBuffer({
    size: MAX_OPS * OP_STRIDE,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  // CSG Phase 1a: per-object descriptors. Bound at @binding(2); unused (but still
  // bound) on the single-object path so the additive invariant holds.
  const objectsBuf = device.createBuffer({
    size: MAX_OBJECTS * OBJ_STRIDE,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const marchBind = (pl) =>
    device.createBindGroup({
      layout: pl.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: globalsBuf } },
        { binding: 1, resource: { buffer: opsBuf } },
        { binding: 2, resource: { buffer: objectsBuf } },
      ],
    });
  const bind = marchBind(pipeline);
  // The numeric variant, built on first demand and cached for the session.
  let numericVariant = null; // { pl, bg }
  let numericActive = false; // latched by writeGlobals from the frame's deOption
  function activeMarch() {
    if (!numericActive) return { pl: pipeline, bg: bind };
    if (!numericVariant) {
      // Synchronous createShaderModule: the one-time hitch happens exactly when
      // the user first loads a numeric-DE formula, never on the analytic path.
      const m = device.createShaderModule({ code: buildWGSL({ numericDE: true }) });
      const pl = marchPipe(m);
      numericVariant = { pl, bg: marchBind(pl) };
    }
    return numericVariant;
  }

  const gBuf = new ArrayBuffer(GLOBALS_BYTES);
  const gF = new Float32Array(gBuf);
  const gU = new Uint32Array(gBuf);

  function writeGlobals({
    res,
    cam,
    iters,
    opCount,
    addC,
    maxSteps,
    bailout,
    eps,
    deScale,
    colA,
    colB,
    bg,
    colorMode,
    deOption,
    julia,
    juliaC,
    palette,
    light,
    objectCount,
    tNear,
    tFar,
    hybrid, // Phase 1a spike — {aOpCount,bOpCount,scheduleA,scheduleB,addCA,addCB}
    morph, // formula-morph spike — {bOpCount,bIters,bAddC,bJulia,bJuliaC,bDeOption,t}
    colorBlend, // coloring-mode crossfade — {t, modeB, palOnB}
    post, // P0 post pass — {tone: 0 classic|1 filmic, exposure EV, dither LSBs, vignette}
  }) {
    const b = cam.basis();
    const A = colA || [0.86, 0.46, 0.18],
      B = colB || [0.18, 0.62, 0.74],
      BG = bg || [0.07, 0.09, 0.15];
    const JC = juliaC || [0, 0, 0];
    // Julia mode folds into the add-gate: c is a fixed constant (jc.xyz) instead
    // of the sample point, and it's always added (regardless of the preset's AddC).
    const addGate = addC || julia ? 1 : 0;
    // Deep zoom §3.1 recenter. O = the pan target, kept in JS f64 (cam.target
    // already is — it's a plain number array). ro_rel = eye−O is computed HERE,
    // in f64, before it ever touches a Float32Array — only the small residual
    // and O separately get truncated to f32, not their (huge) sum. CSG scenes
    // are explicitly out of scope for v1 (§14, objDist's per-object placement
    // isn't recentered) — offset=(0,0,0) there makes the shader's reconstruction
    // an exact no-op, so scenes render byte-identically to before this change.
    const isScene = (objectCount || 0) > 0;
    const { O, roRel } = computeRecenter(b.eye, cam.target, isScene);
    gF[0] = res[0];
    gF[1] = res[1];
    gF[2] = cam.fov;
    gF[3] = tNear ?? 0.02; // res.w — deep zoom §5 (was unused padding)
    gF[4] = roRel[0]; // camPos.xyz — deep zoom §3.1: the RESIDUAL, not b.eye
    gF[5] = roRel[1];
    gF[6] = roRel[2];
    gF[7] = tFar ?? 80.0; // camPos.w — deep zoom §5 (was unused padding)
    gF[8] = b.fwd[0];
    gF[9] = b.fwd[1];
    gF[10] = b.fwd[2];
    gF[11] = 0; // camFwd
    gF[12] = b.right[0];
    gF[13] = b.right[1];
    gF[14] = b.right[2];
    gF[15] = 0; // camRight
    gF[16] = b.up[0];
    gF[17] = b.up[1];
    gF[18] = b.up[2];
    gF[19] = 0; // camUp
    gU[20] = iters;
    gU[21] = opCount;
    gU[22] = addGate;
    gU[23] = maxSteps; // ctrl
    gF[24] = bailout;
    gF[25] = eps;
    gF[26] = deScale;
    gF[27] = colorMode || 0; // prm
    gF[28] = A[0];
    gF[29] = A[1];
    gF[30] = A[2];
    gF[31] = deOption ?? 2; // colA.rgb + .w=deOption
    numericActive = (deOption ?? 2) >= 2.5; // pick the march pipeline variant
    gF[32] = B[0];
    gF[33] = B[1];
    gF[34] = B[2];
    gF[35] = 0; // colB
    gF[36] = BG[0];
    gF[37] = BG[1];
    gF[38] = BG[2];
    gF[39] = 0; // bgc
    gF[40] = JC[0];
    gF[41] = JC[1];
    gF[42] = JC[2];
    gF[43] = julia ? 1 : 0; // jc.xyz + .w=julia flag

    // Cosine palette + lighting (defaults reproduce the original look).
    const P = palette || {},
      L = light || {};
    const pa = P.a || [0.5, 0.5, 0.5],
      pb = P.b || [0.5, 0.5, 0.5];
    const pc = P.c || [1, 1, 1],
      pd = P.d || [0, 0.33, 0.67];
    const ld = L.dir || [0.45, -0.65, 0.75];
    gF[44] = pa[0];
    gF[45] = pa[1];
    gF[46] = pa[2];
    gF[47] = P.on ? 1 : 0; // palA.rgb + .w=paletteOn
    gF[48] = pb[0];
    gF[49] = pb[1];
    gF[50] = pb[2];
    gF[51] = 0; // palB
    gF[52] = pc[0];
    gF[53] = pc[1];
    gF[54] = pc[2];
    gF[55] = 0; // palC (freq)
    gF[56] = pd[0];
    gF[57] = pd[1];
    gF[58] = pd[2];
    gF[59] = 0; // palD (phase)
    gF[60] = ld[0];
    gF[61] = ld[1];
    gF[62] = ld[2];
    gF[63] = 0; // light dir
    gF[64] = L.ambient ?? 0.16;
    gF[65] = L.rim ?? 0.45;
    gF[66] = L.gloss ?? 0.0;
    gF[67] = L.intensity ?? 1.0; // lprm (w=intensity)
    gU[68] = objectCount || 0; // scene.x — 0 = single-object (legacy) path
    gU[69] = 0;
    gU[70] = 0;
    gU[71] = 0; // scene padding
    // Phase 1a spike (IDEAS ①, docs/design/HYBRID_ITERATION.md) — hyb word.
    // aOpCount+bOpCount==0 (default) ⇒ mapDE skips straight to the legacy path.
    const h = hybrid;
    gU[72] = h?.aOpCount ?? 0;
    gU[73] = h?.bOpCount ?? 0;
    gU[74] = h
      ? (h.scheduleA & 0xff) |
        ((h.scheduleB & 0xff) << 8) |
        (h.addCA ? 1 << 16 : 0) |
        (h.addCB ? 1 << 17 : 0)
      : 0;
    gU[75] = 0; // hyb.w reserved
    // Deep zoom §3.1 — offset word. O truncates to f32 here (fine on its own,
    // O(1) relative precision — see DEEP_ZOOM.md §4.1); it's the RECONSTRUCTION
    // ADD in the shader, not O's storage, that sets the precision ceiling (§4.2).
    gF[76] = O[0];
    gF[77] = O[1];
    gF[78] = O[2];
    gF[79] = 0; // offset.w reserved
    // Formula-morph spike words (VIDEO_EXPORT_DRAWER_V2 tier 2). morphB.w==0
    // (the default) ⇒ mapDE skips straight past the morph path — every legacy
    // frame is byte-identical to before this change. morphB.w carries B's OWN
    // orbit bailout as f32 bits (word 83 written via the FLOAT view — the
    // shader bitcasts it back; non-zero doubles as the on-flag). B must not
    // share A's bailout: a power-8 escape orbit run to an IFS-sized 1e6
    // overflows dr in f32 and blanks the render.
    const M = morph;
    const mjc = M?.bJuliaC || [0, 0, 0];
    gU[80] = M?.bOpCount ?? 0;
    gU[81] = M?.bIters ?? 0;
    gU[82] = M
      ? (M.bAddC ? 1 : 0) | (M.bJulia ? 2 : 0) | (((M.bDeOption ?? 2) & 3) << 2)
      : 0;
    gF[83] = M ? (M.bailB ?? 1.0e6) : 0; // morph on + B's bailout (f32 bits)
    gF[84] = M?.t ?? 0;
    gF[85] = mjc[0] ?? 0;
    gF[86] = mjc[1] ?? 0;
    gF[87] = mjc[2] ?? 0;
    gF[88] = M?.swell ?? 0; // morphX.x — mid-blend dilation (world units, pre-shaped by preview)
    gF[89] = 0;
    gF[90] = 0;
    gF[91] = 0; // morphX.yzw reserved
    // Coloring-mode crossfade word (colorX): 0 blend ⇒ the legacy shade only.
    const CB = colorBlend;
    gF[92] = CB?.t ?? 0;
    gF[93] = CB?.modeB ?? 0;
    gF[94] = CB?.palOnB ? 1 : 0;
    gF[95] = 0; // colorX.w reserved
    // P0 post word — defaults are the shipped look: filmic soft-shoulder tone
    // map on, no exposure bias, 1-LSB dither, no vignette. `tone: 0` gives the
    // pre-P0 straight encode ("Classic") for A/B comparisons and tests.
    gF[96] = post?.tone ?? 1;
    gF[97] = post?.exposure ?? L.exposure ?? 0; // whole-frame EV (user "Exposure" slider)
    gF[98] = post?.dither ?? 1;
    gF[99] = post?.vignette ?? 0;
    // P1 light rig + material words. Defaults ARE the shipped "defaults
    // upgrade": soft shadow on (softness 0.5 → k 17), AO on (0.55), no
    // metallic, white key, fills off — so old saved colorings (no fields)
    // render the upgraded default look, and these MUST match core/coloring.js
    // defaultColoring (the curated-defaults invariant the share codec's
    // guarded reads rely on).
    const kc = L.keyColor || [1, 1, 1];
    gF[100] = kc[0];
    gF[101] = kc[1];
    gF[102] = kc[2];
    gF[103] = 0; // lightC.w reserved
    const shadowAmt = L.shadow ?? 0.5; // 0 = off; 0..1 = penumbra size
    gF[104] = L.metallic ?? 0;
    gF[105] = 30 - 26 * shadowAmt; // penumbra k (30 hard … 4 very soft)
    gF[106] = shadowAmt > 0 ? 1 : 0;
    gF[107] = L.ao ?? 0.55;
    // Fill/back directions are DERIVED from the key dir (not stored): fill
    // from the opposite azimuth with flattened elevation, back mirrored fully.
    // Shader normalizes.
    gF[108] = -ld[0];
    gF[109] = ld[1] * 0.35;
    gF[110] = -ld[2];
    gF[111] = L.fill ?? 0; // light2.w = fill intensity
    const fc = L.fillColor || [1, 1, 1];
    gF[112] = fc[0];
    gF[113] = fc[1];
    gF[114] = fc[2];
    gF[115] = 0; // light2c.w reserved
    gF[116] = -ld[0];
    gF[117] = -ld[1];
    gF[118] = -ld[2];
    gF[119] = L.back ?? 0; // light3.w = back intensity
    const bc = L.backColor || [1, 1, 1];
    gF[120] = bc[0];
    gF[121] = bc[1];
    gF[122] = bc[2];
    gF[123] = 0; // light3c.w reserved
    // P2 jitter word — zero on every full write (the un-jittered base frame);
    // writeJitter() partial-updates it between accumulation frames.
    gF[124] = 0;
    gF[125] = 0;
    gF[126] = 0;
    gF[127] = 0;
    // P3 env/fog words. One UI macro each: L.sky drives blend + sun glow +
    // IBL amount; L.fog drives density + in-scatter; L.glow drives bloom
    // strength (threshold fixed at 1.0 — only above-white blooms). All default
    // 0 → byte-identical to the P2 pipeline (sky/fog/bloom fully opt-in;
    // defaults must match core/coloring.js defaultColoring).
    const sky = L.sky ?? 0;
    const fogAmt = L.fog ?? 0;
    const glow = L.glow ?? 0;
    gF[128] = sky;
    gF[129] = sky; // sun glow rides the sky macro
    gF[130] = 0.35; // ground dim (fixed)
    gF[131] = sky; // IBL ambient tint rides the sky macro
    gF[132] = fogAmt;
    gF[133] = fogAmt; // in-scatter rides the fog macro
    gF[134] = glow * 0.8; // bloom strength
    gF[135] = 1.0; // bloom threshold (pre-tonemap HDR)
    bloomOn = glow > 0;
    // P4 DOF word: slider 0..1 → lens radius scaled by the orbit distance
    // (quadratic feel — the top half of the slider does the drama); autofocus
    // = the orbit distance (zoom-to-surface glides the target onto the
    // surface, so this focuses on what you're looking at). zw = lens point,
    // zeroed here (base frame = lens center) and written per accumulation
    // sample by writeJitter.
    const ap = L.aperture ?? 0;
    gF[136] = ap > 0 ? ap * ap * (cam.dist ?? 4) * 0.06 : 0;
    gF[137] = cam.dist ?? 4;
    gF[138] = 0;
    gF[139] = 0;
    device.queue.writeBuffer(globalsBuf, 0, gBuf);
  }

  function writeOps(ops) {
    // Overflow/unknown-key are programmer errors — throw (like writeScene),
    // never silently truncate to MAX_OPS or write a garbage id from an undefined
    // def. Flat formulas are capped at MAX_FLAT_OPS (64) upstream in sanitize;
    // this buffer (MAX_OPS = 192) also holds hybrid/morph slot concatenations, so
    // the cap here is the physical buffer, not the flat cap.
    const n = ops.length;
    if (n === 0) return 0; // empty stack: nothing to upload (WebGPU rejects 0-byte writes)
    if (n > MAX_OPS)
      throw new Error(`writeOps: ${n} ops > cap ${MAX_OPS}`);
    const buf = new ArrayBuffer(n * OP_STRIDE);
    const u = new Uint32Array(buf);
    const f = new Float32Array(buf);
    for (let i = 0; i < n; i++) {
      const def = byKey(ops[i].key);
      if (!def) throw new Error(`writeOps: unknown op key ${ops[i].key}`);
      u[i * 4 + 0] = def.id;
      f[i * 4 + 1] = ops[i].values[0] ?? 0;
      f[i * 4 + 2] = ops[i].values[1] ?? 0;
      f[i * 4 + 3] = ops[i].values[2] ?? 0;
    }
    device.queue.writeBuffer(opsBuf, 0, buf);
    return n;
  }

  // Hybrid iteration (IDEAS ①, docs/design/HYBRID_ITERATION.md §3.5) — WGSL is
  // data-driven (a runtime switch over the op buffer), so unlike GLSL there's no
  // codegen split: just concatenate both slots onto the SAME shared ops buffer
  // writeOps already uses. The {aOpCount,bOpCount} split that tells mapDE_hybrid
  // where slot A ends and B begins rides the `hyb` globals word (writeGlobals),
  // not this call — this only needs to get the op DATA into the buffer.
  function writeHybrid(opsA, opsB) {
    return writeOps([...opsA, ...opsB]);
  }

  // Formula-morph spike — same concat trick as hybrid: A's ops then B's on the
  // shared buffer; the {opCount, morphB.x} split tells mapDE_morph where B
  // starts. Metadata rides writeGlobals' `morph` param, not this call.
  function writeMorph(opsA, opsB) {
    return writeOps([...opsA, ...opsB]);
  }

  // CSG Phase 1b — multi-object scene upload. Concatenates every object's ops
  // into the shared ops buffer (tracking per-object opStart/opCount) and packs an
  // Obj descriptor per object (full 80-B v1 layout; see shader.js). Bounds-checks
  // both caps and THROWS on overflow (never silently truncates). Returns the count.
  //
  // Each object (canonical ObjectSpec; flat-shape fallbacks kept for safety):
  //   { objType, ops?, iters?, addC?, deOption?, julia?, juliaC?, looseDE?,
  //     combine?, blendK?, primParam?, transform:{ origin:[x,y,z], uscale, rot } }
  //   objType 0 = IFS op-slice · 1 box · 2 sphere · 3 torus · 4 cylinder ·
  //     5 capsule · 6 plane. Multi-param prims (torus/cylinder/capsule) use
  //     primParam + primParam2 (see shader.js Obj word 4).
  //   transform.rot is Euler XYZ degrees (or a length-4 quaternion).
  function writeScene(objects) {
    if (!Array.isArray(objects) || objects.length === 0)
      throw new Error("writeScene: empty scene");
    if (objects.length > MAX_OBJECTS)
      throw new Error(
        `writeScene: ${objects.length} objects > cap ${MAX_OBJECTS}`,
      );

    // Pass 1 — concatenate ops, assign opStart/opCount, bounds-check the total.
    const opBuf = new ArrayBuffer(MAX_OPS * OP_STRIDE);
    const opU = new Uint32Array(opBuf);
    const opF = new Float32Array(opBuf);
    let cursor = 0;
    const slices = [];
    for (const o of objects) {
      // Per-op mute inside a scene object — filter EXACTLY like the CPU tier
      // (cpu.js makeSceneDE activeOps) and WebGL2 (shader_gl.js/renderer_gl.js);
      // 3-emitter mirror discipline, guarded by core/scenemute.test.mjs.
      const ops =
        Number(o.objType) > 0 ? [] : (o.ops || []).filter((op) => !op.muted);
      if (cursor + ops.length > MAX_OPS)
        throw new Error(
          `writeScene: concatenated ops ${cursor + ops.length} > cap ${MAX_OPS}`,
        );
      const start = cursor;
      for (let i = 0; i < ops.length; i++) {
        const def = byKey(ops[i].key);
        if (!def) throw new Error(`writeScene: unknown op "${ops[i].key}"`);
        const j = cursor * 4;
        opU[j + 0] = def.id;
        opF[j + 1] = ops[i].values?.[0] ?? 0;
        opF[j + 2] = ops[i].values?.[1] ?? 0;
        opF[j + 3] = ops[i].values?.[2] ?? 0;
        cursor++;
      }
      slices.push({ start, count: ops.length });
    }
    if (cursor > 0)
      device.queue.writeBuffer(opsBuf, 0, opBuf, 0, cursor * OP_STRIDE);

    // Pass 2 — pack the Obj descriptors (20 words = 80 B each).
    const objBuf = new ArrayBuffer(objects.length * OBJ_STRIDE);
    const oU = new Uint32Array(objBuf);
    const oF = new Float32Array(objBuf);
    objects.forEach((o, k) => {
      const base = k * 24;
      const objType = Number(o.objType) & 0xf; // 0 IFS·1 box·2 sphere·3 torus·4 cyl·5 capsule·6 plane
      const deOption = objType > 0 ? 0 : (o.deOption ?? 2);
      const tr = o.transform || {};
      const julia = objType === 0 && !!o.julia;
      let flags = 0;
      if (o.addC) flags |= 1 << 0; // bit0 addC
      flags |= (deOption & 3) << 1; // bits1-2 deOption
      if (julia) flags |= 1 << 3; // bit3 julia
      if (o.looseDE) flags |= 1 << 4; // bit4 looseDE
      flags |= ((o.combine ?? o.combineType ?? 0) & 3) << 5; // bits5-6 combineType
      flags |= (objType & 0xf) << 7; // bits7-10 objType (widened 2→4 bits for types 3-6)
      if (objType === 0 && o.boxBase) flags |= 1 << 11; // bit11 box-DE base (IFS only)
      oU[base + 0] = slices[k].start;
      oU[base + 1] = slices[k].count;
      oU[base + 2] = o.iters ?? 1;
      oU[base + 3] = flags;
      const org = tr.origin || o.origin || [0, 0, 0];
      oF[base + 4] = org[0] ?? 0;
      oF[base + 5] = org[1] ?? 0;
      oF[base + 6] = org[2] ?? 0;
      oF[base + 7] = tr.uscale ?? o.uscale ?? 1;
      const q = eulerToQuat(tr.rot ?? o.rot ?? [0, 0, 0]);
      oF[base + 8] = q[0];
      oF[base + 9] = q[1];
      oF[base + 10] = q[2];
      oF[base + 11] = q[3];
      const jc = o.juliaC || [0, 0, 0];
      oF[base + 12] = jc[0] ?? 0;
      oF[base + 13] = jc[1] ?? 0;
      oF[base + 14] = jc[2] ?? 0;
      oF[base + 15] = o.blendK ?? 0; // smin blend
      // primParam: box he / sphere r / torus R / cyl r / capsule r / plane thick.
      oF[base + 16] = o.primParam ?? o.halfExtent ?? o.radius ?? 1;
      // primParam2: torus minor r / cylinder + capsule half-height (else unused).
      oF[base + 17] = o.primParam2 ?? 0;
      oF[base + 18] = 0; // pad1
      oF[base + 19] = 0; // pad2
      // word 5: per-object albedo (sRGB; shader applies s2l). §3.8 per-object color.
      const col = o.color || [0.86, 0.46, 0.18];
      oF[base + 20] = col[0] ?? 0.86;
      oF[base + 21] = col[1] ?? 0.46;
      oF[base + 22] = col[2] ?? 0.18;
      oF[base + 23] = 0; // pad3
    });
    device.queue.writeBuffer(objectsBuf, 0, objBuf);
    return objects.length;
  }

  // Cached HDR intermediate for the interactive paths (draw/drawTo re-ensure it
  // on size change — resizes are rare; renderToImage builds its own per call).
  let hdr = null; // { tex, view, postBind, bloom*, w, h, destroy }
  function makeHdr(w, h) {
    const mk = (mw, mh) =>
      device.createTexture({
        size: [mw, mh],
        format: HDR_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
    const tex = mk(w, h);
    const view = tex.createView();
    // P3 bloom half-res pair. Zero-initialized (WebGPU) → composites nothing
    // until the bloom passes actually run; the post pass always binds bloomA.
    const bw = Math.max(1, Math.ceil(w / 2)),
      bh = Math.max(1, Math.ceil(h / 2));
    const bloomA = mk(bw, bh),
      bloomB = mk(bw, bh);
    const bloomAView = bloomA.createView(),
      bloomBView = bloomB.createView();
    // NOTE on layout:'auto': a binding lands in the layout only if the entry
    // point STATICALLY USES it. fs_bright reads G (threshold) + src; the blur
    // entry points read ONLY src — their layouts contain just binding 1, and
    // passing binding 0 anyway makes the bind group (and every submit that
    // touches it) invalid. Bit us: black canvas, no exception.
    const brightBindFor = (src) =>
      device.createBindGroup({
        layout: brightPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: globalsBuf } },
          { binding: 1, resource: src },
        ],
      });
    const blurBind = (pl, src) =>
      device.createBindGroup({
        layout: pl.getBindGroupLayout(0),
        entries: [{ binding: 1, resource: src }],
      });
    const postBindFor = (src) =>
      device.createBindGroup({
        layout: postPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: globalsBuf } },
          { binding: 1, resource: src },
          { binding: 2, resource: bloomAView },
          { binding: 3, resource: bloomSampler },
        ],
      });
    return {
      tex,
      view,
      w,
      h,
      postBind: postBindFor(view),
      postBindFor,
      bloomAView,
      bloomBView,
      brightBind: brightBindFor(view), // bright reads the plain HDR
      brightBindFor, // …or an accum half
      blurHBind: blurBind(blurHPipeline, bloomAView),
      blurVBind: blurBind(blurVPipeline, bloomBView),
      destroy() {
        tex.destroy();
        bloomA.destroy();
        bloomB.destroy();
      },
    };
  }
  function ensureHdr(w, h) {
    if (hdr && hdr.w === w && hdr.h === h) return hdr;
    hdr?.destroy();
    hdr = makeHdr(w, h);
    return hdr;
  }
  // P3: encode bright → blurH → blurV (half res). `brightBind` selects the
  // source (plain HDR or the just-written accumulation half).
  function encodeBloom(enc, h, brightBind) {
    encodeOnePass(enc, brightPipeline, brightBind, h.bloomAView);
    encodeOnePass(enc, blurHPipeline, h.blurHBind, h.bloomBView);
    encodeOnePass(enc, blurVPipeline, h.blurVBind, h.bloomAView);
  }
  // P2 accumulation ping-pong pair (rgba32float running average) + the bind
  // groups that read/write it. Cached like hdr; renderToImage builds its own.
  function makeAccum(h) {
    const mk = () =>
      device.createTexture({
        size: [h.w, h.h],
        format: ACCUM_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
    const texA = mk(),
      texB = mk();
    const viewA = texA.createView(),
      viewB = texB.createView();
    const accumBindFor = (prevView) =>
      device.createBindGroup({
        layout: accumPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: globalsBuf } },
          { binding: 1, resource: h.view },
          { binding: 2, resource: prevView },
        ],
      });
    return {
      texA,
      texB,
      viewA,
      viewB,
      w: h.w,
      h: h.h,
      flip: false, // false → next write lands in A (prev = B)
      bindToA: accumBindFor(viewB),
      bindToB: accumBindFor(viewA),
      // Post + bloom-bright binds come from the hdr object so they carry the
      // P3 bloom texture/sampler entries (post layout requires them).
      postA: h.postBindFor(viewA),
      postB: h.postBindFor(viewB),
      brightA: h.brightBindFor(viewA),
      brightB: h.brightBindFor(viewB),
      destroy() {
        texA.destroy();
        texB.destroy();
      },
    };
  }
  let accum = null;
  function ensureAccum(h) {
    // Keyed on the hdr OBJECT, not its size: the interactive tiers resize the
    // canvas (e.g. 900 → 720 → 900), each resize recreates the HDR texture,
    // and a size-keyed cache would revive an accum whose bind groups still
    // reference the DESTROYED old HDR — "destroyed texture used in a submit",
    // every post-interaction settled frame silently dropped (black canvas).
    if (accum && accum.src === h) return accum;
    accum?.destroy();
    accum = makeAccum(h);
    accum.src = h;
    return accum;
  }
  // Per-sample partial uploads: the jitter word (subpixel offset + accum
  // weight) and the dof word's zw (this sample's lens point — P4).
  function writeJitter(jx, jy, weight, lensX = 0, lensY = 0) {
    const j = new Float32Array([jx, jy, weight, 0]);
    device.queue.writeBuffer(globalsBuf, JITTER_WORD * 16, j);
    const l = new Float32Array([lensX, lensY]);
    device.queue.writeBuffer(globalsBuf, DOF_WORD * 16 + 8, l);
  }

  // Encode the two-pass sequence: march → HDR intermediate, post → target view.
  function encodeOnePass(enc, pl, bg, view) {
    const pass = enc.beginRenderPass({
      colorAttachments: [
        { view, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" },
      ],
    });
    pass.setPipeline(pl);
    pass.setBindGroup(0, bg);
    pass.draw(3);
    pass.end();
  }
  function encodePasses(enc, h, targetView) {
    encodeOnePass(enc, activeMarch().pl, activeMarch().bg, h.view); // march → HDR
    if (bloomOn) encodeBloom(enc, h, h.brightBind); // P3 (skipped at glow 0)
    encodeOnePass(enc, postPipeline, h.postBind, targetView); // post → target
  }
  // P2 accumulation frame: march → HDR, blend into the ping-pong average
  // (weight came in via writeJitter — 1 replaces, 1/(N+1) refines), post reads
  // the average → swap chain. The caller owns the sample counter.
  function drawAccum() {
    const tex = ctx.getCurrentTexture();
    const h = ensureHdr(tex.width, tex.height);
    const ac = ensureAccum(h);
    const enc = device.createCommandEncoder();
    encodeOnePass(enc, activeMarch().pl, activeMarch().bg, h.view);
    encodeOnePass(enc, accumPipeline, ac.flip ? ac.bindToB : ac.bindToA, ac.flip ? ac.viewB : ac.viewA);
    if (bloomOn) encodeBloom(enc, h, ac.flip ? ac.brightB : ac.brightA); // read the just-written half
    encodeOnePass(enc, postPipeline, ac.flip ? ac.postB : ac.postA, tex.createView());
    ac.flip = !ac.flip;
    device.queue.submit([enc.finish()]);
  }

  // Render into a target context (defaults to the main canvas). Thumbnails pass
  // their own offscreen context here, reusing the same pipelines + buffers.
  function drawTo(target) {
    const t = target || ctx;
    const tex = t.getCurrentTexture();
    const h = ensureHdr(tex.width, tex.height);
    const enc = device.createCommandEncoder();
    encodePasses(enc, h, tex.createView());
    device.queue.submit([enc.finish()]);
  }
  function draw() {
    drawTo(ctx);
  }

  // Configure an extra canvas's context to share this device/format so drawTo()
  // can render into it (used by the preset thumbnail gallery).
  function configureContext(targetCanvas) {
    const c = targetCanvas.getContext("webgpu");
    c.configure({ device, format, alphaMode: "opaque" });
    return c;
  }

  // Render the current globals/ops into an offscreen texture and read the pixels
  // straight back as ImageData. Unlike drawing to a canvas + drawImage(), this is
  // DETERMINISTIC — no canvas-presentation lag — so thumbnail captures never grab
  // a stale frame. Used by the preset gallery.
  async function renderToImage(W, H, samples = 1) {
    const tex = device.createTexture({
      size: [W, H],
      format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    // Own HDR intermediate per call (requested size ≠ canvas size); destroyed
    // below with the readback texture. Same two-pass sequence as draw/drawTo,
    // so exports match the screen exactly.
    const h = makeHdr(W, H);
    // P2 multi-sample export: N jittered march+accumulate rounds (one submit
    // each — writeJitter is a queue write, so per-sample values need per-sample
    // submits), then post reads the average. Sample 0 is unjittered with
    // weight 1, matching the live accumulation sequence exactly.
    let ac = null;
    if (samples > 1 && !Number.isNaN(samples)) {
      ac = makeAccum(h);
      for (let i = 0; i < samples; i++) {
        const [jx, jy] = i === 0 ? [0, 0] : r2jitter(i);
        const [lx, ly] = i === 0 ? [0, 0] : lensSample(i);
        writeJitter(jx, jy, 1 / (i + 1), lx, ly);
        const e = device.createCommandEncoder();
        encodeOnePass(e, activeMarch().pl, activeMarch().bg, h.view);
        encodeOnePass(e, accumPipeline, ac.flip ? ac.bindToB : ac.bindToA, ac.flip ? ac.viewB : ac.viewA);
        ac.flip = !ac.flip;
        device.queue.submit([e.finish()]);
      }
      writeJitter(0, 0, 0, 0, 0); // restore — live frames render unjittered, lens-centered
    }
    const bytesPerRow = Math.ceil((W * 4) / 256) * 256; // 256-byte row alignment
    const buf = device.createBuffer({
      size: bytesPerRow * H,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = device.createCommandEncoder();
    if (ac) {
      // Post resolves the accumulated average (the LAST-written half: flip
      // toggled after the final write, so read the half `flip` now points AWAY
      // from — texA when flip is true).
      if (bloomOn) encodeBloom(enc, h, ac.flip ? ac.brightA : ac.brightB);
      encodeOnePass(enc, postPipeline, ac.flip ? ac.postA : ac.postB, tex.createView());
    } else {
      encodePasses(enc, h, tex.createView());
    }
    enc.copyTextureToBuffer(
      { texture: tex },
      { buffer: buf, bytesPerRow },
      { width: W, height: H },
    );
    device.queue.submit([enc.finish()]);

    await buf.mapAsync(GPUMapMode.READ);
    const src = new Uint8Array(buf.getMappedRange());
    const out = new Uint8ClampedArray(W * H * 4);
    const bgra = format.startsWith("bgra"); // preferred format may be BGRA; ImageData is RGBA
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const s = y * bytesPerRow + x * 4,
          d = (y * W + x) * 4;
        out[d] = bgra ? src[s + 2] : src[s];
        out[d + 1] = src[s + 1];
        out[d + 2] = bgra ? src[s] : src[s + 2];
        out[d + 3] = src[s + 3];
      }
    }
    buf.unmap();
    buf.destroy();
    tex.destroy();
    h.destroy();
    ac?.destroy();
    return new ImageData(out, W, H);
  }

  return {
    device,
    format,
    writeGlobals,
    writeJitter,
    drawAccum,
    writeOps,
    writeHybrid,
    writeMorph,
    writeScene,
    draw,
    drawTo,
    configureContext,
    renderToImage,
    MAX_OPS,
    MAX_OBJECTS,
  };
}

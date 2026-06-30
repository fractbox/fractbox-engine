// WebGPU renderer. Owns the device, the pipeline (built once from the
// generated WGSL), and two GPU buffers:
//   - a 176-byte uniform "Globals" (camera + control scalars; 11 * vec4)
//   - a storage "ops" buffer (the op-list as packed Op structs)
// Editing param values or reordering ops only rewrites the ops buffer; the
// pipeline is never rebuilt at runtime.

import { buildWGSL } from "./shader.js";
import { byKey } from "./operators.js";

const MAX_OPS = 64; // op-buffer capacity (64 * 16 = 1 KiB)
const OP_STRIDE = 16; // bytes per Op (u32 + 3*f32)
const GLOBALS_BYTES = 272; // 17 * vec4 (16B each)

export async function createRenderer(canvas) {
  if (!navigator.gpu) throw new Error("WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("no GPU adapter");
  const device = await adapter.requestDevice();
  const ctx = canvas.getContext("webgpu");
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: "opaque" });

  const module = device.createShaderModule({ code: buildWGSL() });

  // Surface compile errors loudly (best-effort; not all browsers populate this).
  const info = await module.getCompilationInfo?.();
  if (info) {
    for (const m of info.messages) {
      const line = `WGSL ${m.type} @${m.lineNum}:${m.linePos} — ${m.message}`;
      if (m.type === "error") console.error(line);
      else console.warn(line);
    }
  }

  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module, entryPoint: "vs" },
    fragment: { module, entryPoint: "fs", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });

  const globalsBuf = device.createBuffer({
    size: GLOBALS_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const opsBuf = device.createBuffer({
    size: MAX_OPS * OP_STRIDE,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: globalsBuf } },
      { binding: 1, resource: { buffer: opsBuf } },
    ],
  });

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
  }) {
    const b = cam.basis();
    const A = colA || [0.86, 0.46, 0.18],
      B = colB || [0.18, 0.62, 0.74],
      BG = bg || [0.07, 0.09, 0.15];
    const JC = juliaC || [0, 0, 0];
    // Julia mode folds into the add-gate: c is a fixed constant (jc.xyz) instead
    // of the sample point, and it's always added (regardless of the preset's AddC).
    const addGate = addC || julia ? 1 : 0;
    gF[0] = res[0];
    gF[1] = res[1];
    gF[2] = cam.fov;
    gF[3] = 0; // res
    gF[4] = b.eye[0];
    gF[5] = b.eye[1];
    gF[6] = b.eye[2];
    gF[7] = 0; // camPos
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
    device.queue.writeBuffer(globalsBuf, 0, gBuf);
  }

  function writeOps(ops) {
    const n = Math.min(ops.length, MAX_OPS);
    if (n === 0) return 0; // empty stack: nothing to upload (WebGPU rejects 0-byte writes)
    const buf = new ArrayBuffer(n * OP_STRIDE);
    const u = new Uint32Array(buf);
    const f = new Float32Array(buf);
    for (let i = 0; i < n; i++) {
      const def = byKey(ops[i].key);
      u[i * 4 + 0] = def.id;
      f[i * 4 + 1] = ops[i].values[0] ?? 0;
      f[i * 4 + 2] = ops[i].values[1] ?? 0;
      f[i * 4 + 3] = ops[i].values[2] ?? 0;
    }
    device.queue.writeBuffer(opsBuf, 0, buf);
    return n;
  }

  // Render into a target context (defaults to the main canvas). Thumbnails pass
  // their own offscreen context here, reusing the same pipeline + buffers.
  function drawTo(target) {
    const t = target || ctx;
    const enc = device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [
        {
          view: t.getCurrentTexture().createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.draw(3);
    pass.end();
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
  async function renderToImage(W, H) {
    const tex = device.createTexture({
      size: [W, H],
      format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const bytesPerRow = Math.ceil((W * 4) / 256) * 256; // 256-byte row alignment
    const buf = device.createBuffer({
      size: bytesPerRow * H,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [
        {
          view: tex.createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.draw(3);
    pass.end();
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
    return new ImageData(out, W, H);
  }

  return {
    device,
    format,
    writeGlobals,
    writeOps,
    draw,
    drawTo,
    configureContext,
    renderToImage,
    MAX_OPS,
  };
}

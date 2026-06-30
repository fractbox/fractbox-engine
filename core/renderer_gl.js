// WebGL2 renderer — a drop-in twin of renderer.js (same writeOps / writeGlobals
// / draw / renderToImage interface), so preview.js drives WebGPU or WebGL2 with
// no controller changes. Used as the middle fallback tier: WebGPU → WebGL2 →
// CPU/ASCII. The fragment shader is regenerated only on an op-STRUCTURE change;
// param values + camera ride uniforms (no relink on a slider drag).

import { byKey } from "./operators.js";
import { VERT_GL, buildFragGL, MAX_PARAMS } from "./shader_gl.js";

const MAX_OPS = 64;

export async function createRendererGL(canvas) {
  const gl = canvas.getContext("webgl2", {
    antialias: false,
    preserveDrawingBuffer: true, // so canvas.toBlob (PNG export) sees the frame
    alpha: false,
  });
  if (!gl) throw new Error("WebGL2 unavailable");

  gl.bindVertexArray(gl.createVertexArray()); // attribute-less draw needs a bound VAO

  const vert = compileShader(gl, gl.VERTEX_SHADER, VERT_GL);
  let program = null;
  let opSig = null; // op-key signature the current program was built for
  let loc = {}; // uniform-location cache for the live program

  let G = null; // last writeGlobals payload
  const params = new Float32Array(MAX_PARAMS);
  let opCount = 0;

  function rebuild(ops) {
    const frag = compileShader(gl, gl.FRAGMENT_SHADER, buildFragGL(ops));
    const p = gl.createProgram();
    gl.attachShader(p, vert);
    gl.attachShader(p, frag);
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(p);
      gl.deleteProgram(p);
      gl.deleteShader(frag);
      throw new Error("WebGL2 link failed: " + log);
    }
    if (program) gl.deleteProgram(program);
    program = p;
    loc = {};
    gl.deleteShader(frag);
  }

  const U = (name) => (loc[name] ??= gl.getUniformLocation(program, name));

  // ── renderer.js-compatible surface ─────────────────────────────────────────
  function writeOps(ops) {
    const n = Math.min(ops.length, MAX_OPS);
    const sig = ops
      .slice(0, n)
      .map((o) => o.key)
      .join("|");
    if (sig !== opSig || !program) {
      rebuild(ops.slice(0, n));
      opSig = sig;
    }
    params.fill(0);
    let slot = 0;
    for (let i = 0; i < n; i++) {
      const def = byKey(ops[i].key);
      if (!def) continue;
      for (let k = 0; k < def.params.length && slot < MAX_PARAMS; k++) {
        params[slot++] = ops[i].values[k] ?? 0;
      }
    }
    opCount = n;
    return n;
  }

  function writeGlobals(payload) {
    G = payload;
  }

  function applyUniforms(res) {
    gl.useProgram(program);
    const b = G.cam.basis();
    const A = G.colA || [0.86, 0.46, 0.18];
    const B = G.colB || [0.18, 0.62, 0.74];
    const BG = G.bg || [0.07, 0.09, 0.15];
    const JC = G.juliaC || [0, 0, 0];
    const P = G.palette || {};
    const L = G.light || {};
    const pa = P.a || [0.5, 0.5, 0.5],
      pb = P.b || [0.5, 0.5, 0.5];
    const pc = P.c || [1, 1, 1],
      pd = P.d || [0, 0.33, 0.67];
    const ld = L.dir || [0.45, -0.65, 0.75];
    const addGate = G.addC || G.julia ? 1 : 0;

    gl.uniform2f(U("uRes"), res[0], res[1]);
    gl.uniform1f(U("uFov"), G.cam.fov);
    gl.uniform3f(U("uCamPos"), b.eye[0], b.eye[1], b.eye[2]);
    gl.uniform3f(U("uCamFwd"), b.fwd[0], b.fwd[1], b.fwd[2]);
    gl.uniform3f(U("uCamRight"), b.right[0], b.right[1], b.right[2]);
    gl.uniform3f(U("uCamUp"), b.up[0], b.up[1], b.up[2]);
    gl.uniform1i(U("uIters"), G.iters | 0);
    gl.uniform1i(U("uAddGate"), addGate);
    gl.uniform1i(U("uMaxSteps"), G.maxSteps | 0);
    gl.uniform1i(U("uColorMode"), G.colorMode || 0);
    gl.uniform1f(U("uBailout"), G.bailout);
    gl.uniform1f(U("uEps"), G.eps);
    gl.uniform1f(U("uDeScale"), G.deScale ?? 0.85);
    gl.uniform1f(U("uDeOption"), G.deOption ?? 2);
    gl.uniform3f(U("uColA"), A[0], A[1], A[2]);
    gl.uniform3f(U("uColB"), B[0], B[1], B[2]);
    gl.uniform3f(U("uBg"), BG[0], BG[1], BG[2]);
    gl.uniform3f(U("uJc"), JC[0], JC[1], JC[2]);
    gl.uniform1f(U("uJulia"), G.julia ? 1 : 0);
    gl.uniform3f(U("uPalA"), pa[0], pa[1], pa[2]);
    gl.uniform3f(U("uPalB"), pb[0], pb[1], pb[2]);
    gl.uniform3f(U("uPalC"), pc[0], pc[1], pc[2]);
    gl.uniform3f(U("uPalD"), pd[0], pd[1], pd[2]);
    gl.uniform1f(U("uPalOn"), P.on ? 1 : 0);
    gl.uniform3f(U("uLightDir"), ld[0], ld[1], ld[2]);
    gl.uniform1f(U("uAmbient"), L.ambient ?? 0.16);
    gl.uniform1f(U("uRim"), L.rim ?? 0.45);
    gl.uniform1f(U("uGloss"), L.gloss ?? 0.0);
    gl.uniform1f(U("uIntensity"), L.intensity ?? 1.0);
    gl.uniform1fv(U("uP"), params);
  }

  function draw() {
    if (!program || !G) return;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvas.width, canvas.height);
    applyUniforms([canvas.width, canvas.height]);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  // Offscreen render → ImageData (RGBA, top-left origin) for preset thumbnails.
  let fbo = null,
    ftex = null,
    fw = 0,
    fh = 0;
  function ensureFbo(W, H) {
    if (fbo && fw === W && fh === H) return;
    if (ftex) gl.deleteTexture(ftex);
    if (!fbo) fbo = gl.createFramebuffer();
    ftex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, ftex);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA,
      W,
      H,
      0,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      null,
    );
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D,
      ftex,
      0,
    );
    fw = W;
    fh = H;
  }
  async function renderToImage(W, H) {
    if (!program || !G) return new ImageData(W, H);
    ensureFbo(W, H);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.viewport(0, 0, W, H);
    applyUniforms([W, H]);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    const px = new Uint8Array(W * H * 4);
    gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, px);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    // GL origin is bottom-left; ImageData is top-left → flip rows.
    const out = new Uint8ClampedArray(W * H * 4);
    const row = W * 4;
    for (let y = 0; y < H; y++)
      out.set(px.subarray((H - 1 - y) * row, (H - y) * row), y * row);
    return new ImageData(out, W, H);
  }

  // Shim the bits of the WebGPU `device.queue` that preview.js awaits.
  const device = {
    queue: { onSubmittedWorkDone: () => (gl.finish(), Promise.resolve()) },
  };

  return {
    device,
    writeGlobals,
    writeOps,
    draw,
    renderToImage,
    MAX_OPS,
    backend: "webgl2",
  };
}

function compileShader(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh);
    gl.deleteShader(sh);
    throw new Error("WebGL2 shader compile failed: " + log);
  }
  return sh;
}

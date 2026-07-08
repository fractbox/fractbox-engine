// WebGL2 renderer — a drop-in twin of renderer.js (same writeOps / writeGlobals
// / draw / renderToImage interface), so preview.js drives WebGPU or WebGL2 with
// no controller changes. Used as the middle fallback tier: WebGPU → WebGL2 →
// CPU/ASCII. The fragment shader is regenerated only on an op-STRUCTURE change;
// param values + camera ride uniforms (no relink on a slider drag).

import { byKey } from "./operators.js";
import { computeRecenter } from "./recenter.js";
import {
  VERT_GL,
  buildFragGL,
  buildSceneFragGL,
  sceneParamLayout,
  activeSceneOps,
  MAX_PARAMS,
  MAX_OBJECTS,
} from "./shader_gl.js";
import { MAX_OPS_WEBGL2 } from "./limits.js";
import { eulerToQuat } from "./quat.js";

const MAX_OPS = MAX_OPS_WEBGL2; // 64 — lower than the WebGPU tier (see limits.js)

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

  // CSG scene state — per-object descriptors (uniform arrays) + the scene flag.
  let isScene = false;
  let sceneCount = 0;

  // Hybrid iteration state (IDEAS ①, docs/design/HYBRID_ITERATION.md §3.5).
  let isHybrid = false;
  let hybA = 1,
    hybB = 1,
    hybAddGateA = 0,
    hybAddGateB = 0;
  const objOrigin = new Float32Array(MAX_OBJECTS * 3);
  const objUscale = new Float32Array(MAX_OBJECTS).fill(1);
  const objQuat = new Float32Array(MAX_OBJECTS * 4);
  const objBlendK = new Float32Array(MAX_OBJECTS);
  const objJc = new Float32Array(MAX_OBJECTS * 3);
  const objPrim = new Float32Array(MAX_OBJECTS).fill(1);
  const objPrim2 = new Float32Array(MAX_OBJECTS);
  const objIters = new Int32Array(MAX_OBJECTS).fill(1);
  const objAddGate = new Int32Array(MAX_OBJECTS);
  const objJulia = new Float32Array(MAX_OBJECTS);
  const objDeOption = new Float32Array(MAX_OBJECTS).fill(2);
  const objColor = new Float32Array(MAX_OBJECTS * 3).fill(0.5); // per-object albedo (sRGB) — §3.8

  function linkProgram(fragSrc) {
    const frag = compileShader(gl, gl.FRAGMENT_SHADER, fragSrc);
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

  function rebuild(ops) {
    linkProgram(buildFragGL(ops));
  }

  const U = (name) => (loc[name] ??= gl.getUniformLocation(program, name));

  // ── renderer.js-compatible surface ─────────────────────────────────────────
  function writeOps(ops) {
    isScene = false;
    isHybrid = false;
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

  // Hybrid iteration (IDEAS ①, docs/design/HYBRID_ITERATION.md §3.5) — the GLSL
  // codegen split. Structural signature covers BOTH slots' op-key sequences (a
  // change to either rebuilds the program); the schedule counts + per-slot addC
  // ride uniforms (uHybA/uHybB/uAddGateA/uAddGateB), not codegen — editing the
  // ratio is a uniform upload, matching the plain single-object path's rule.
  function writeHybrid(opsA, opsB, schedule, addCA, addCB) {
    isScene = false;
    isHybrid = true;
    const nA = Math.min(opsA.length, MAX_OPS);
    const nB = Math.min(opsB.length, MAX_OPS);
    const sig =
      "hyb:" +
      opsA
        .slice(0, nA)
        .map((o) => o.key)
        .join(",") +
      "|" +
      opsB
        .slice(0, nB)
        .map((o) => o.key)
        .join(",");
    if (sig !== opSig || !program) {
      linkProgram(buildFragGL(opsA.slice(0, nA), { ops: opsB.slice(0, nB) }));
      opSig = sig;
    }
    params.fill(0);
    let slot = 0;
    for (let i = 0; i < nA; i++) {
      const def = byKey(opsA[i].key);
      if (!def) continue;
      for (let k = 0; k < def.params.length && slot < MAX_PARAMS; k++) {
        params[slot++] = opsA[i].values[k] ?? 0;
      }
    }
    for (let i = 0; i < nB; i++) {
      const def = byKey(opsB[i].key);
      if (!def) continue;
      for (let k = 0; k < def.params.length && slot < MAX_PARAMS; k++) {
        params[slot++] = opsB[i].values[k] ?? 0;
      }
    }
    opCount = nA + nB;
    hybA = schedule?.a ?? 1;
    hybB = schedule?.b ?? 1;
    hybAddGateA = addCA ? 1 : 0;
    hybAddGateB = addCB ? 1 : 0;
    return opCount;
  }

  // CSG Phase 1b/C — multi-object scene upload (mirrors renderer.js writeScene).
  // The fragment program is codegen'd per scene STRUCTURE (object count, each
  // object's objType/combine/op-key sequence) and rebuilt when that signature
  // changes; per-object SCALARS ride uniform arrays (no recompile on a value
  // tweak). Param values for IFF objects share one uP[] array, concatenated at
  // the bases from sceneParamLayout (kept in lockstep with the codegen).
  function writeScene(objects) {
    isHybrid = false;
    if (!Array.isArray(objects) || objects.length === 0)
      throw new Error("writeScene: empty scene");
    if (objects.length > MAX_OBJECTS)
      throw new Error(
        `writeScene: ${objects.length} objects > cap ${MAX_OBJECTS}`,
      );
    isScene = true;
    sceneCount = objects.length;

    // Structural signature → rebuild the program only when the STRUCTURE changes.
    const sig =
      "scene:" +
      objects
        .map((o) => {
          const t = Number(o.objType) & 0xf;
          const cmb = (o.combine ?? o.combineType ?? 0) & 3;
          // Active ops only (per-op scene mute) — muting/unmuting changes the
          // codegen'd program, so it must change the signature too.
          const keys =
            t === 0 ? activeSceneOps(o).map((x) => x.key).join(",") : "";
          // boxBase is baked into the IFS objDE codegen → part of the structure.
          const bb = t === 0 && o.boxBase ? "b" : "";
          return `${t}/${cmb}/${bb}/${keys}`;
        })
        .join("|");
    if (sig !== opSig || !program) {
      linkProgram(buildSceneFragGL(objects));
      opSig = sig;
    }

    // Param values into the shared uP[] (IFS objects only) + per-object scalars.
    params.fill(0);
    const layout = sceneParamLayout(objects);
    objects.forEach((o, k) => {
      const objType = Number(o.objType) & 0xf;
      if (objType === 0) {
        let slot = layout[k].slotBase;
        // Active ops only — must match sceneParamLayout/buildSceneFragGL's
        // slice or the uP[] values shift against the codegen'd indices.
        for (const op of activeSceneOps(o)) {
          const def = byKey(op.key);
          if (!def) continue;
          for (let p = 0; p < def.params.length; p++) {
            if (slot >= MAX_PARAMS)
              throw new Error("writeScene: scene params > cap " + MAX_PARAMS);
            params[slot++] = op.values?.[p] ?? 0;
          }
        }
      }
      const tr = o.transform || {};
      const org = tr.origin || o.origin || [0, 0, 0];
      objOrigin[k * 3 + 0] = org[0] ?? 0;
      objOrigin[k * 3 + 1] = org[1] ?? 0;
      objOrigin[k * 3 + 2] = org[2] ?? 0;
      objUscale[k] = tr.uscale ?? o.uscale ?? 1;
      const q = eulerToQuat(tr.rot ?? o.rot ?? [0, 0, 0]);
      objQuat[k * 4 + 0] = q[0];
      objQuat[k * 4 + 1] = q[1];
      objQuat[k * 4 + 2] = q[2];
      objQuat[k * 4 + 3] = q[3];
      objBlendK[k] = o.blendK ?? 0;
      const col = o.color || [0.86, 0.46, 0.18];
      objColor[k * 3 + 0] = col[0] ?? 0.86;
      objColor[k * 3 + 1] = col[1] ?? 0.46;
      objColor[k * 3 + 2] = col[2] ?? 0.18;
      const jc = o.juliaC || [0, 0, 0];
      objJc[k * 3 + 0] = jc[0] ?? 0;
      objJc[k * 3 + 1] = jc[1] ?? 0;
      objJc[k * 3 + 2] = jc[2] ?? 0;
      objPrim[k] = o.primParam ?? o.halfExtent ?? o.radius ?? 1;
      objPrim2[k] = o.primParam2 ?? 0;
      objIters[k] = o.iters ?? 1;
      const julia = objType === 0 && !!o.julia;
      objAddGate[k] = objType === 0 && (!!o.addC || julia) ? 1 : 0;
      objJulia[k] = julia ? 1 : 0;
      objDeOption[k] = objType === 0 ? (o.deOption ?? 2) : 0;
    });
    return sceneCount;
  }

  function writeGlobals(payload) {
    G = payload;
  }

  // Upload the per-object scene uniform arrays (only when a scene is live).
  function applySceneUniforms() {
    gl.uniform3fv(U("uObjOrigin"), objOrigin);
    gl.uniform1fv(U("uObjUscale"), objUscale);
    gl.uniform4fv(U("uObjQuat"), objQuat);
    gl.uniform1fv(U("uObjBlendK"), objBlendK);
    gl.uniform3fv(U("uObjColor"), objColor);
    gl.uniform3fv(U("uObjJc"), objJc);
    gl.uniform1fv(U("uObjPrim"), objPrim);
    gl.uniform1fv(U("uObjPrim2"), objPrim2);
    gl.uniform1iv(U("uObjIters"), objIters);
    gl.uniform1iv(U("uObjAddGate"), objAddGate);
    gl.uniform1fv(U("uObjJulia"), objJulia);
    gl.uniform1fv(U("uObjDeOption"), objDeOption);
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
    // Deep zoom §3.1 recenter — shares renderer.js's exact computation.
    const { O, roRel } = computeRecenter(b.eye, G.cam.target, isScene);

    gl.uniform2f(U("uRes"), res[0], res[1]);
    gl.uniform1f(U("uFov"), G.cam.fov);
    gl.uniform3f(U("uCamPos"), roRel[0], roRel[1], roRel[2]);
    gl.uniform3f(U("uOffset"), O[0], O[1], O[2]);
    gl.uniform3f(U("uCamFwd"), b.fwd[0], b.fwd[1], b.fwd[2]);
    gl.uniform3f(U("uCamRight"), b.right[0], b.right[1], b.right[2]);
    gl.uniform3f(U("uCamUp"), b.up[0], b.up[1], b.up[2]);
    gl.uniform1i(U("uIters"), G.iters | 0);
    if (isHybrid) {
      // Hybrid iteration (§3.5) — schedule + per-slot addC ride uniforms, not
      // codegen, so a ratio/addC tweak is a value upload, not a recompile.
      gl.uniform1i(U("uHybA"), hybA);
      gl.uniform1i(U("uHybB"), hybB);
      // Fold the formula-level Julia flag into both slot gates, matching the WGSL
      // (`… || (G.jc.w > 0.5)`) and CPU (`… || julia`) tiers — without this a
      // Julia hybrid rendered a different fractal (no +c added) on the WebGL2 tier.
      const jHyb = G.julia ? 1 : 0;
      gl.uniform1i(U("uAddGateA"), hybAddGateA || jHyb);
      gl.uniform1i(U("uAddGateB"), hybAddGateB || jHyb);
    } else {
      gl.uniform1i(U("uAddGate"), addGate);
    }
    gl.uniform1i(U("uMaxSteps"), G.maxSteps | 0);
    gl.uniform1i(U("uColorMode"), G.colorMode || 0);
    gl.uniform1f(U("uBailout"), G.bailout);
    gl.uniform1f(U("uEps"), G.eps);
    gl.uniform1f(U("uDeScale"), G.deScale ?? 0.85);
    gl.uniform1f(U("uNear"), G.tNear ?? 0.02); // deep zoom §5
    gl.uniform1f(U("uFar"), G.tFar ?? 80.0); // deep zoom §5
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
    // P1 light rig — mirrors renderer.js writeGlobals gF[100..123] exactly
    // (defaults must match core/coloring.js defaultColoring; fill/back dirs
    // are DERIVED from the key dir, not stored).
    const kc = L.keyColor || [1, 1, 1];
    const shadowAmt = L.shadow ?? 0.5;
    gl.uniform3f(U("uKeyC"), kc[0], kc[1], kc[2]);
    gl.uniform1f(U("uMetallic"), L.metallic ?? 0);
    gl.uniform1f(U("uShadowK"), 30 - 26 * shadowAmt);
    gl.uniform1f(U("uShadowOn"), shadowAmt > 0 ? 1 : 0);
    gl.uniform1f(U("uAoStr"), L.ao ?? 0.55);
    gl.uniform3f(U("uFillDir"), -ld[0], ld[1] * 0.35, -ld[2]);
    gl.uniform1f(U("uFill"), L.fill ?? 0);
    const fc = L.fillColor || [1, 1, 1];
    gl.uniform3f(U("uFillC"), fc[0], fc[1], fc[2]);
    gl.uniform3f(U("uBackDir"), -ld[0], -ld[1], -ld[2]);
    gl.uniform1f(U("uBack"), L.back ?? 0);
    const bc = L.backColor || [1, 1, 1];
    gl.uniform3f(U("uBackC"), bc[0], bc[1], bc[2]);
    // P3 env/fog macros — mirror renderer.js gF[128..133] (bloom is WebGPU-only).
    const sky = L.sky ?? 0;
    const fogAmt = L.fog ?? 0;
    gl.uniform1f(U("uSky"), sky);
    gl.uniform1f(U("uSunGlow"), sky);
    gl.uniform1f(U("uGround"), 0.35);
    gl.uniform1f(U("uIbl"), sky);
    gl.uniform1f(U("uFogAmt"), fogAmt);
    gl.uniform1f(U("uInScatter"), fogAmt);
    gl.uniform1f(U("uExposure"), L.exposure ?? 0); // whole-frame EV (mirrors post.y)
    gl.uniform1fv(U("uP"), params);
    if (isScene) applySceneUniforms();
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
    writeHybrid,
    writeScene,
    draw,
    renderToImage,
    MAX_OPS,
    MAX_OBJECTS,
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

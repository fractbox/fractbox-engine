// Recording WebGL2 mock for node tests (#729) — the renderer_gl.test.mjs mock
// grown a memory: it implements exactly the surface createRendererGL touches
// AND records what the renderer uploads, so behavioral tests can compare the
// GL tier's packed uniform state against the WebGPU tier's packed Globals
// words (core/tierparity.test.mjs) and cross-check the renderer's std140 Bulk
// allocation against the emitted shader's own Bulk declaration
// (core/glbulkparity.test.mjs) — all with no GL context anywhere in CI.
//
// Recorded per draw (cleared by `beginDraw()`):
//   rec.uniforms  Map<name, number[]> — every gl.uniform* call, f32-rounded
//                 (the real driver truncates to f32 at upload; Math.fround
//                 mirrors that so values compare bit-equal with a
//                 Float32Array-packed WebGPU snapshot).
//   rec.bulk      Float32Array — the std140 Bulk UBO upload (bufferSubData).
// Recorded per link:
//   rec.progSrc   the fragment source of the LIVE program (codegen variant
//                 changes — a relink — are part of the tier's packed state).
//   rec.bulkAlloc the byte size the renderer allocated for the Bulk UBO
//                 (setupBulk's bufferData) — renderer-side bulkLayout() args.

export function makeRecordingGL() {
  const rec = {
    uniforms: new Map(),
    bulk: null,
    progSrc: null,
    bulkAlloc: 0,
  };
  const f = Math.fround;
  const put = (loc, vals) => {
    if (loc && loc.name) rec.uniforms.set(loc.name, vals.map(f));
  };
  let pendingFrag = null; // last compiled fragment source, adopted on link
  let boundBuf = null;
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
      if (sh.type === gl.FRAGMENT_SHADER) pendingFrag = sh.src;
      sh.ok = true;
    },
    getShaderParameter: (sh) => sh.ok,
    getShaderInfoLog: () => "",
    deleteShader: () => {},
    createProgram: () => ({}),
    attachShader: () => {},
    linkProgram: () => {
      rec.progSrc = pendingFrag;
    },
    getProgramParameter: () => true,
    getProgramInfoLog: () => "",
    deleteProgram: () => {},
    createBuffer: () => ({}),
    bindBuffer: (target, buf) => {
      boundBuf = buf;
    },
    bufferData: (target, sizeOrData) => {
      if (target === gl.UNIFORM_BUFFER)
        rec.bulkAlloc =
          typeof sizeOrData === "number" ? sizeOrData : sizeOrData.byteLength;
    },
    bufferSubData: (target, off, data) => {
      if (target === gl.UNIFORM_BUFFER) rec.bulk = Float32Array.from(data);
    },
    getUniformBlockIndex: () => 0,
    uniformBlockBinding: () => {},
    bindBufferBase: () => {},
    // A location per NAME — the renderer memoizes per link (loc cache reset),
    // and the record keys off the name either way.
    getUniformLocation: (prog, name) => ({ name }),
    useProgram: () => {},
    viewport: () => {},
    bindFramebuffer: () => {},
    drawArrays: () => {},
    getError: () => 0,
    finish: () => {},
    uniform1f: (l, x) => put(l, [x]),
    uniform2f: (l, x, y) => put(l, [x, y]),
    uniform3f: (l, x, y, z) => put(l, [x, y, z]),
    uniform4f: (l, x, y, z, w) => put(l, [x, y, z, w]),
    uniform1i: (l, x) => put(l, [x]),
  };
  const listeners = {};
  const canvas = {
    width: 4,
    height: 4,
    addEventListener: (name, fn) => {
      (listeners[name] ||= []).push(fn);
    },
    getContext: () => gl,
  };
  const beginDraw = () => {
    rec.uniforms = new Map();
    rec.bulk = null;
  };
  // Stable serialization of everything a draw uploaded + the live program
  // text, for change-detection between two draws.
  const snapshot = () => {
    const u = [...rec.uniforms.entries()].sort((a, b) =>
      a[0] < b[0] ? -1 : 1,
    );
    return JSON.stringify({
      uniforms: u,
      bulk: rec.bulk ? [...rec.bulk] : null,
      progSrc: rec.progSrc,
    });
  };
  // Every f32 value a draw uploaded, flattened (uniform args + bulk words) —
  // the "did value X land ANYWHERE in this tier's packed state" probe.
  const values = () => {
    const out = [];
    for (const vals of rec.uniforms.values()) out.push(...vals);
    if (rec.bulk) out.push(...rec.bulk);
    return out;
  };
  return { canvas, gl, rec, beginDraw, snapshot, values };
}

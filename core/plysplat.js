// INRIA-convention 3D Gaussian Splat .ply writer (UE splat export §1.1 / §5).
// binary_little_endian, all fields float32, INRIA canonical property order:
//   x y z | nx ny nz | f_dc_0..2 | [f_rest_0..44] | opacity |
//   scale_0..2 | rot_0..3        (rot = w,x,y,z; opacity = pre-sigmoid logit;
//                                 scale = log-scale; f_dc = SH0, display-sRGB).
// Degree-0 by default: f_rest_* omitted (PLY is self-describing / name-keyed, so
// SuperSplat/PlayCanvas accept it and the file is ~3.6× smaller). `fRest: 45`
// emits all-zero f_rest_0..44 for a byte-exact INRIA header. No deps, DOM-free
// (TextEncoder + DataView exist in Node and browsers).

const BASE_PROPS = [
  "x",
  "y",
  "z",
  "nx",
  "ny",
  "nz",
  "f_dc_0",
  "f_dc_1",
  "f_dc_2",
];
const TAIL_PROPS = [
  "opacity",
  "scale_0",
  "scale_1",
  "scale_2",
  "rot_0",
  "rot_1",
  "rot_2",
  "rot_3",
];

function propNames(fRest) {
  const rest = [];
  for (let i = 0; i < fRest; i++) rest.push(`f_rest_${i}`);
  return [...BASE_PROPS, ...rest, ...TAIL_PROPS];
}

function header(count, fRest) {
  const lines = [
    "ply",
    "format binary_little_endian 1.0",
    `element vertex ${count}`,
    ...propNames(fRest).map((p) => `property float ${p}`),
    "end_header",
    "",
  ];
  return lines.join("\n");
}

// splats: { count, pos, normal, fdc, opacity, scaleLog, rot } (fitSplats output).
// opts: { fRest = 0 }.  Returns an ArrayBuffer.
// Throws RangeError on count === 0 — the §5.3a backstop; callers must abort on an
// empty capture, but the writer never emits a degenerate file.
export function writeSplatPly(splats, opts = {}) {
  const { fRest = 0 } = opts;
  const count = splats.count | 0;
  if (count <= 0)
    throw new RangeError("writeSplatPly: refusing to write 0 splats");

  const stride = 4 * (17 + fRest); // 17 base+tail floats + fRest zeros
  const headBytes = new TextEncoder().encode(header(count, fRest));
  const buf = new ArrayBuffer(headBytes.length + count * stride);
  new Uint8Array(buf).set(headBytes, 0);
  const dv = new DataView(buf);
  const LE = true;

  let off = headBytes.length;
  const put = (v) => {
    dv.setFloat32(off, v, LE);
    off += 4;
  };
  const { pos, normal, fdc, opacity, scaleLog, rot } = splats;
  for (let i = 0; i < count; i++) {
    const j = 3 * i,
      k = 4 * i;
    put(pos[j]);
    put(pos[j + 1]);
    put(pos[j + 2]);
    put(normal[j]);
    put(normal[j + 1]);
    put(normal[j + 2]);
    put(fdc[j]);
    put(fdc[j + 1]);
    put(fdc[j + 2]);
    for (let r = 0; r < fRest; r++) put(0);
    put(opacity[i]);
    put(scaleLog[j]);
    put(scaleLog[j + 1]);
    put(scaleLog[j + 2]);
    put(rot[k]);
    put(rot[k + 1]);
    put(rot[k + 2]);
    put(rot[k + 3]);
  }
  return buf;
}

// Minimal header + first-records reader for tests / CLI --verify. Returns
// { count, properties: [{name, type}], stride, vertices: Float32Array } where
// `vertices` holds the first `maxVerts` records flattened (props-per-vertex).
export function readSplatPlyHeader(buffer, maxVerts = 4) {
  const u8 = new Uint8Array(buffer);
  // find "end_header\n"
  const marker = new TextEncoder().encode("end_header\n");
  let hdrEnd = -1;
  outer: for (let i = 0; i <= u8.length - marker.length; i++) {
    for (let m = 0; m < marker.length; m++)
      if (u8[i + m] !== marker[m]) continue outer;
    hdrEnd = i + marker.length;
    break;
  }
  if (hdrEnd < 0) throw new Error("readSplatPlyHeader: no end_header");
  const text = new TextDecoder().decode(u8.subarray(0, hdrEnd));
  const lines = text.split("\n");
  const properties = [];
  let count = 0;
  for (const ln of lines) {
    if (ln.startsWith("element vertex ")) count = parseInt(ln.slice(15), 10);
    else if (ln.startsWith("property ")) {
      const [, type, name] = ln.split(/\s+/);
      properties.push({ name, type });
    }
  }
  const stride = properties.length * 4;
  const dv = new DataView(buffer);
  const nv = Math.min(maxVerts, count);
  const vertices = new Float32Array(nv * properties.length);
  for (let v = 0; v < nv; v++)
    for (let p = 0; p < properties.length; p++)
      vertices[v * properties.length + p] = dv.getFloat32(
        hdrEnd + v * stride + p * 4,
        true,
      );
  return { count, properties, stride, vertices };
}

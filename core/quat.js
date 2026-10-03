// Quaternion helpers — the single JS-side source for the CSG rotation convention.
// The WGSL/GLSL emitters carry their own mirror of this math in shader strings;
// this module is the one JS copy shared by renderer.js / renderer_gl.js / cpu.js.

// Euler (degrees, Three.js 'XYZ' intrinsic order) → unit quaternion [x,y,z,w].
// Steps B/C (GLSL/CPU mirrors) MUST use this exact convention. A length-4 input
// is treated as a quaternion already (normalized).
export function eulerToQuat(rot) {
  if (Array.isArray(rot) && rot.length === 4) {
    const [x, y, z, w] = rot;
    const n = Math.hypot(x, y, z, w) || 1;
    return [x / n, y / n, z / n, w / n];
  }
  const d2r = Math.PI / 180;
  const rx = (rot?.[0] || 0) * d2r,
    ry = (rot?.[1] || 0) * d2r,
    rz = (rot?.[2] || 0) * d2r;
  const c1 = Math.cos(rx / 2),
    s1 = Math.sin(rx / 2);
  const c2 = Math.cos(ry / 2),
    s2 = Math.sin(ry / 2);
  const c3 = Math.cos(rz / 2),
    s3 = Math.sin(rz / 2);
  return [
    s1 * c2 * c3 + c1 * s2 * s3, // x
    c1 * s2 * c3 - s1 * c2 * s3, // y
    c1 * c2 * s3 + s1 * s2 * c3, // z
    c1 * c2 * c3 - s1 * s2 * s3, // w
  ];
}

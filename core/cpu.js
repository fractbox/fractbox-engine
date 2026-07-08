// CPU fallback renderer — a plain-JS port of the engine's distance estimate
// (see shader.js `mapDE`) plus an ASCII raymarcher. No GPU, no DOM, no deps:
// pure ESM so it runs anywhere core/ runs (the no-WebGPU fallback, the OSS demo,
// headless tooling, the stacking games). Returns text; the app wraps it in a UI.
//
// ⚠ The per-op math below MIRRORS the WGSL in operators.js. The two are the same
// math in two languages — if you change an operator there, change it here too
// (cpu.test guards that every preset still produces a finite, non-empty render).

import { makeCamera } from "./camera.js";
import { isEscapeTime, effectiveDeOption, activeOps } from "./operators.js";
import { looseDE, hybridDeFamily } from "./stability.js";
import { eulerToQuat } from "./quat.js";

const D2R = Math.PI / 180;
const rad = (d) => d * D2R;

// ── CSG Phase 1b — scene math (mirrors shader.js qrot/boxDE/sphereDE/sminP) ────
// eulerToQuat is hoisted to ./quat.js (the one JS copy shared with the renderers).

// Rotate (vx,vy,vz) by quaternion q=[x,y,z,w]. Mirrors shader.js qrot():
// v' = 2·dot(u,v)·u + (s²−dot(u,u))·v + 2s·cross(u,v), u=q.xyz, s=q.w.
function qrot(q, vx, vy, vz) {
  const ux = q[0],
    uy = q[1],
    uz = q[2],
    s = q[3];
  const dotuv = ux * vx + uy * vy + uz * vz;
  const k = s * s - (ux * ux + uy * uy + uz * uz);
  const cx = uy * vz - uz * vy,
    cy = uz * vx - ux * vz,
    cz = ux * vy - uy * vx;
  return [
    2 * dotuv * ux + k * vx + 2 * s * cx,
    2 * dotuv * uy + k * vy + 2 * s * cy,
    2 * dotuv * uz + k * vz + 2 * s * cz,
  ];
}

// Analytic box SDF (exact). Mirrors shader.js boxDE().
function boxDE(x, y, z, he) {
  const qx = Math.abs(x) - he,
    qy = Math.abs(y) - he,
    qz = Math.abs(z) - he;
  const ox = Math.max(qx, 0),
    oy = Math.max(qy, 0),
    oz = Math.max(qz, 0);
  return Math.hypot(ox, oy, oz) + Math.min(Math.max(qx, Math.max(qy, qz)), 0);
}

// Analytic sphere SDF. Mirrors shader.js sphereDE().
function sphereDE(x, y, z, r) {
  return Math.hypot(x, y, z) - r;
}

// Analytic torus SDF. Mirrors shader.js torusDE() (R major, r minor; axis = y).
function torusDE(x, y, z, R, r) {
  return Math.hypot(Math.hypot(x, z) - R, y) - r;
}

// Capped-cylinder SDF. Mirrors shader.js cylinderDE() (r radius, h half-height).
function cylinderDE(x, y, z, r, h) {
  const dx = Math.hypot(x, z) - r,
    dy = Math.abs(y) - h;
  return (
    Math.min(Math.max(dx, dy), 0) + Math.hypot(Math.max(dx, 0), Math.max(dy, 0))
  );
}

// Vertical-capsule SDF. Mirrors shader.js capsuleDE() (r radius, h half-height).
function capsuleDE(x, y, z, r, h) {
  const qy = y - Math.max(-h, Math.min(h, y));
  return Math.hypot(x, qy, z) - r;
}

// Slab/ground-plane SDF. Mirrors shader.js planeDE() (thick half-thickness).
function planeDE(x, y, z, thick) {
  return Math.abs(y) - thick;
}

// Polynomial smooth-min. Mirrors shader.js sminP() (mix(b,a,h)=b+(a-b)·h).
function sminP(a, b, k) {
  if (k <= 0) return Math.min(a, b);
  const h = Math.max(0, Math.min(1, 0.5 + (0.5 * (b - a)) / k));
  return b + (a - b) * h - k * h * (1 - h);
}
// Smooth max — carve dual of sminP (k=0 → hard max). Mirrors shader.js smaxP().
function smaxP(a, b, k) {
  return -sminP(-a, -b, k);
}

// ── per-op apply: mutate state s = {x,y,z,w}; mirrors operators.js `wgsl` ──────
// The single source of the JS-side per-operator math. Pure — closes over no
// render state — so evaluate.js imports it instead of keeping a twin switch (the
// twin drifted once, on `menger`, and mis-graded presets). See REFACTORING.md #1.
export function applyOp(key, v, s) {
  let t, h;
  switch (key) {
    case "boxFold": {
      const f = v[0];
      s.x = Math.abs(s.x + f) - Math.abs(s.x - f) - s.x;
      s.y = Math.abs(s.y + f) - Math.abs(s.y - f) - s.y;
      s.z = Math.abs(s.z + f) - Math.abs(s.z - f) - s.z;
      break;
    }
    case "boxFoldXYZ":
      s.x = Math.abs(s.x + v[0]) - Math.abs(s.x - v[0]) - s.x;
      s.y = Math.abs(s.y + v[1]) - Math.abs(s.y - v[1]) - s.y;
      s.z = Math.abs(s.z + v[2]) - Math.abs(s.z - v[2]) - s.z;
      break;
    case "surfFold": {
      const f = v[0];
      s.x = Math.abs(s.x + f) - Math.abs(s.x - f) - s.x;
      s.y = Math.abs(s.y + f) - Math.abs(s.y - f) - s.y;
      break;
    }
    case "sphereFold": {
      const minR2 = v[0] * v[0],
        fixedR2 = v[1] * v[1],
        r2 = s.x * s.x + s.y * s.y + s.z * s.z;
      const k =
        r2 < minR2 ? fixedR2 / minR2 : r2 < fixedR2 ? fixedR2 / r2 : 1.0;
      s.x *= k;
      s.y *= k;
      s.z *= k;
      s.w *= k;
      break;
    }
    case "cylinderFold": {
      const minR2 = v[0] * v[0],
        fixedR2 = v[1] * v[1],
        r2 = s.x * s.x + s.y * s.y;
      const k =
        r2 < minR2 ? fixedR2 / minR2 : r2 < fixedR2 ? fixedR2 / r2 : 1.0;
      s.x *= k;
      s.y *= k;
      s.z *= k;
      s.w *= k;
      break;
    }
    case "sphereInv": {
      const r2 = v[0] * v[0],
        d = Math.max(s.x * s.x + s.y * s.y + s.z * s.z, 1e-6),
        k = r2 / d;
      s.x *= k;
      s.y *= k;
      s.z *= k;
      s.w *= k;
      break;
    }
    case "radialInvert": {
      const dx = s.x - v[0],
        dy = s.y - v[1],
        dz = s.z - v[2];
      const dd = Math.max(dx * dx + dy * dy + dz * dz, 1e-6),
        k = 1 / dd;
      s.x = dx * k + v[0];
      s.y = dy * k + v[1];
      s.z = dz * k + v[2];
      s.w *= k;
      break;
    }
    case "scale": {
      const k = v[0];
      s.x *= k;
      s.y *= k;
      s.z *= k;
      s.w *= Math.abs(k);
      break;
    }
    case "scaleDrift": {
      // Closed-form per-iteration scale ramp; reads the loop index s.i (set by
      // every applyOp consumer before its op loop — SCALE_VARY.md §6.5). The
      // `s.i ?? 0` default is a belt-and-suspenders floor if a caller forgets.
      const m = Math.max(
        -1e5,
        Math.min(1e5, 1 + (v[0] - 1) * Math.pow(1 + v[1], (s.i ?? 0) + 1)),
      );
      s.x *= m;
      s.y *= m;
      s.z *= m;
      s.w *= Math.abs(m);
      break;
    }
    case "translate":
      s.x += v[0];
      s.y += v[1];
      s.z += v[2];
      break;
    case "absFold":
      s.x = Math.abs(s.x);
      s.y = Math.abs(s.y);
      s.z = Math.abs(s.z);
      break;
    case "absOffsetFold":
      s.x = Math.abs(s.x + v[0]) - v[0];
      s.y = Math.abs(s.y + v[1]) - v[1];
      s.z = Math.abs(s.z + v[2]) - v[2];
      break;
    case "absXYZ":
      if (v[0] > 0.5) s.x = Math.abs(s.x);
      if (v[1] > 0.5) s.y = Math.abs(s.y);
      if (v[2] > 0.5) s.z = Math.abs(s.z);
      break;
    case "varyScale": {
      const minR2 = v[0] * v[0],
        fixedR2 = v[1] * v[1],
        r2 = s.x * s.x + s.y * s.y + s.z * s.z,
        rp = Math.pow(Math.max(r2, 1e-12), v[2]);
      const k =
        rp < minR2 ? fixedR2 / minR2 : rp < fixedR2 ? fixedR2 / rp : 1.0;
      s.x *= k;
      s.y *= k;
      s.z *= k;
      s.w *= k;
      break;
    }
    case "newtonTri2": {
      const nx = s.x,
        ny = s.y,
        nz = s.z;
      s.x = nx * nx - 2 * ny * nz;
      s.y = 2 * nx * ny - nz * nz;
      s.z = ny * ny + 2 * nx * nz;
      break;
    }
    case "newtonTri3": {
      const nx = s.x,
        ny = s.y,
        nz = s.z;
      s.x = nx * nx * nx - ny * ny * ny + nz * nz * nz - 6 * nx * ny * nz;
      s.y = 3 * (nx * nx * ny - ny * ny * nz - nx * nz * nz);
      s.z = 3 * (nx * nx * nz + nx * ny * ny - ny * nz * nz);
      break;
    }
    case "bristorBrot": {
      const bx = s.x,
        by = s.y,
        bz = s.z;
      s.x = bx * bx - by * by - bz * bz;
      s.y = by * (v[0] * bx + v[2] * bz);
      s.z = bz * (v[0] * bx + v[1] * by);
      break;
    }
    case "rotateXY": {
      const a = rad(v[0]),
        c = Math.cos(a),
        sn = Math.sin(a);
      const nx = s.x * c - s.y * sn,
        ny = s.x * sn + s.y * c;
      s.x = nx;
      s.y = ny;
      break;
    }
    case "rotateYZ": {
      const a = rad(v[0]),
        c = Math.cos(a),
        sn = Math.sin(a);
      const ny = s.y * c - s.z * sn,
        nz = s.y * sn + s.z * c;
      s.y = ny;
      s.z = nz;
      break;
    }
    case "rotateXZ": {
      const a = rad(v[0]),
        c = Math.cos(a),
        sn = Math.sin(a);
      const nx = s.x * c - s.z * sn,
        nz = s.x * sn + s.z * c;
      s.x = nx;
      s.z = nz;
      break;
    }
    case "rotateXYZ": {
      {
        const a = rad(v[0]),
          c = Math.cos(a),
          sn = Math.sin(a),
          nx = s.x * c - s.y * sn,
          ny = s.x * sn + s.y * c;
        s.x = nx;
        s.y = ny;
      }
      {
        const a = rad(v[1]),
          c = Math.cos(a),
          sn = Math.sin(a),
          ny = s.y * c - s.z * sn,
          nz = s.y * sn + s.z * c;
        s.y = ny;
        s.z = nz;
      }
      {
        const a = rad(v[2]),
          c = Math.cos(a),
          sn = Math.sin(a),
          nx = s.x * c - s.z * sn,
          nz = s.x * sn + s.z * c;
        s.x = nx;
        s.z = nz;
      }
      break;
    }
    case "twist": {
      const tw = rad(v[0]) * s.z,
        c = Math.cos(tw),
        sn = Math.sin(tw);
      const tx = s.x * c - s.y * sn,
        ty = s.x * sn + s.y * c;
      s.x = tx;
      s.y = ty;
      break;
    }
    case "kaleido": {
      const wedge = 6.2831853 / Math.max(v[0], 2);
      let ang = Math.atan2(s.y, s.x);
      ang = ang - wedge * Math.floor(ang / wedge + 0.5);
      ang = Math.abs(ang) + rad(v[1]);
      const r = Math.hypot(s.x, s.y);
      s.x = Math.cos(ang) * r;
      s.y = Math.sin(ang) * r;
      break;
    }
    case "polyAngleFold": {
      const n = Math.max(v[0], 2),
        wedge = 6.2831853 / n,
        off = rad(v[1]);
      let ang = Math.atan2(s.y, s.x) - off;
      ang = ang - wedge * Math.floor(ang / wedge + 0.5);
      if (v[2] > 0.5) ang = Math.abs(ang);
      ang += off;
      const r = Math.hypot(s.x, s.y);
      s.x = Math.cos(ang) * r;
      s.y = Math.sin(ang) * r;
      break;
    }
    case "hexFold": {
      const kx = -0.8660254,
        ky = 0.5;
      s.x = Math.abs(s.x);
      s.y = Math.abs(s.y);
      const d = Math.min(kx * s.x + ky * s.y, 0);
      s.x -= 2 * d * kx;
      s.y -= 2 * d * ky;
      break;
    }
    case "mengerFold":
      if (s.x < s.y) {
        t = s.x;
        s.x = s.y;
        s.y = t;
      }
      if (s.x < s.z) {
        t = s.x;
        s.x = s.z;
        s.z = t;
      }
      if (s.y < s.z) {
        t = s.y;
        s.y = s.z;
        s.z = t;
      }
      break;
    case "octaFold":
      s.x = Math.abs(s.x);
      s.y = Math.abs(s.y);
      s.z = Math.abs(s.z);
      if (s.x < s.y) {
        t = s.x;
        s.x = s.y;
        s.y = t;
      }
      if (s.x < s.z) {
        t = s.x;
        s.x = s.z;
        s.z = t;
      }
      if (s.y < s.z) {
        t = s.y;
        s.y = s.z;
        s.z = t;
      }
      break;
    case "sierpinskiFold":
      if (s.x + s.y < 0) {
        t = -s.y;
        s.y = -s.x;
        s.x = t;
      }
      if (s.x + s.z < 0) {
        t = -s.z;
        s.z = -s.x;
        s.x = t;
      }
      if (s.y + s.z < 0) {
        t = -s.z;
        s.z = -s.y;
        s.y = t;
      }
      break;
    case "zFold":
      if (s.z > v[0]) s.z -= v[1];
      break;
    case "modFold":
      if (v[0] > 0) s.x -= v[0] * Math.floor(s.x / v[0] + 0.5);
      if (v[1] > 0) s.y -= v[1] * Math.floor(s.y / v[1] + 0.5);
      if (v[2] > 0) s.z -= v[2] * Math.floor(s.z / v[2] + 0.5);
      break;
    case "tentFold":
      if (v[0] > 0) s.x = Math.abs(s.x - v[0] * Math.round(s.x / v[0]));
      if (v[1] > 0) s.y = Math.abs(s.y - v[1] * Math.round(s.y / v[1]));
      if (v[2] > 0) s.z = Math.abs(s.z - v[2] * Math.round(s.z / v[2]));
      break;
    case "planeFold": {
      let nx = v[0],
        ny = v[1],
        nz = v[2];
      if (nx * nx + ny * ny + nz * nz < 1e-12) {
        nx = 1;
        ny = 0;
        nz = 0;
      }
      const L = Math.hypot(nx, ny, nz);
      nx /= L;
      ny /= L;
      nz /= L;
      const d = s.x * nx + s.y * ny + s.z * nz;
      if (d < 0) {
        s.x -= 2 * d * nx;
        s.y -= 2 * d * ny;
        s.z -= 2 * d * nz;
      }
      break;
    }
    case "icosaFold": {
      const I = [
        [-0.809017, 0.309017, 0.5],
        [0.5, -0.809017, 0.309017],
        [0.309017, 0.5, -0.809017],
      ];
      s.x = Math.abs(s.x);
      s.y = Math.abs(s.y);
      s.z = Math.abs(s.z);
      for (const i of I) {
        const dd = Math.min(s.x * i[0] + s.y * i[1] + s.z * i[2], 0);
        s.x -= 2 * dd * i[0];
        s.y -= 2 * dd * i[1];
        s.z -= 2 * dd * i[2];
      }
      break;
    }
    case "menger": {
      const sm = v[0],
        c = 1 / 3;
      if (sm >= 0) {
        s.x = Math.sqrt(s.x * s.x + sm);
        s.y = Math.sqrt(s.y * s.y + sm);
        s.z = Math.sqrt(s.z * s.z + sm);
        t = s.x - s.y;
        t = 0.5 * (t - Math.sqrt(t * t + sm));
        s.x -= t;
        s.y += t;
        t = s.x - s.z;
        t = 0.5 * (t - Math.sqrt(t * t + sm));
        s.x -= t;
        s.z += t;
        t = s.y - s.z;
        t = 0.5 * (t - Math.sqrt(t * t + sm));
        s.y -= t;
        s.z += t;
        s.z = c - Math.sqrt((s.z - c) * (s.z - c) + sm);
      } else {
        const k = -sm;
        s.x =
          Math.abs(s.x) < k ? (s.x * s.x) / (2 * k) + 0.5 * k : Math.abs(s.x);
        s.y =
          Math.abs(s.y) < k ? (s.y * s.y) / (2 * k) + 0.5 * k : Math.abs(s.y);
        s.z =
          Math.abs(s.z) < k ? (s.z * s.z) / (2 * k) + 0.5 * k : Math.abs(s.z);
        t = s.x - s.y;
        h = Math.max(k - Math.abs(t), 0) / k;
        t = Math.min(t, 0) - h * h * k * 0.25;
        s.x -= t;
        s.y += t;
        t = s.x - s.z;
        h = Math.max(k - Math.abs(t), 0) / k;
        t = Math.min(t, 0) - h * h * k * 0.25;
        s.x -= t;
        s.z += t;
        t = s.y - s.z;
        h = Math.max(k - Math.abs(t), 0) / k;
        t = Math.min(t, 0) - h * h * k * 0.25;
        s.y -= t;
        s.z += t;
        const dz = s.z - c,
          adz = Math.abs(dz);
        s.z = c - (adz < k ? (dz * dz) / (2 * k) + 0.5 * k : adz);
      }
      break;
    }
    case "mandelbulbPower": {
      const bp = v[0],
        br = Math.hypot(s.x, s.y, s.z);
      if (br > 1e-9) {
        const bth = Math.acos(Math.max(-1, Math.min(1, s.z / br))) * bp;
        const bph = Math.atan2(s.y, s.x) * bp,
          brn = Math.pow(br, bp);
        s.w = ((bp * brn) / br) * s.w + 1;
        const bst = Math.sin(bth);
        s.x = brn * bst * Math.cos(bph);
        s.y = brn * bst * Math.sin(bph);
        s.z = brn * Math.cos(bth);
      }
      break;
    }
    case "sphericalTwoStage": {
      const bp = v[0],
        br = Math.hypot(s.x, s.y, s.z);
      if (br > 1e-9) {
        const bth = Math.acos(Math.max(-1, Math.min(1, s.z / br))) * bp * v[1];
        const bph = Math.atan2(s.y, s.x) * bp * v[2],
          brn = Math.pow(br, bp);
        s.w = ((bp * brn) / br) * s.w + 1;
        const bst = Math.sin(bth);
        s.x = brn * bst * Math.cos(bph);
        s.y = brn * bst * Math.sin(bph);
        s.z = brn * Math.cos(bth);
      }
      break;
    }
    case "boxBulb": {
      const bp = v[0],
        x4 = s.x ** 4,
        y4 = s.y ** 4,
        z4 = s.z ** 4;
      const br = Math.pow(x4 + y4 + z4, 0.25);
      if (br > 1e-9) {
        const bxz = Math.pow(x4 + z4, 0.25),
          brn = Math.pow(br, bp);
        const bth = Math.atan2(bxz, s.y) * bp,
          bza = Math.atan2(s.x, s.z) * bp;
        s.w = ((bp * brn) / br) * s.w + 1;
        const bst = Math.sin(bth);
        s.x = brn * Math.sin(bza) * bst;
        s.y = brn * Math.cos(bth);
        s.z = brn * bst * Math.cos(bza);
      }
      break;
    }
    case "slonoBrot2": {
      s.w = 2 * Math.hypot(s.x, s.y, s.z) * s.w + 1;
      const sa = Math.abs(s.z),
        sx = s.x,
        sy = s.y;
      s.x = sx * sx - sy * sy + 2 * sa * sx;
      s.y = 2 * sy * (sx + sa);
      s.z = Math.abs(sa * sa - sy * sy);
      break;
    }
    case "bulbAxis": {
      const bp = v[0],
        m = Math.round(v[1]),
        br = Math.hypot(s.x, s.y, s.z);
      if (br > 1e-9) {
        let up = s.z,
          a = s.x,
          b = s.y;
        if (m === 1) {
          up = s.y;
          a = s.z;
          b = s.x;
        } else if (m === 2) {
          up = s.x;
          a = s.y;
          b = s.z;
        }
        const bth = Math.acos(Math.max(-1, Math.min(1, up / br))) * bp;
        const bph = Math.atan2(b, a) * bp,
          brn = Math.pow(br, bp);
        s.w = ((bp * brn) / br) * s.w + 1;
        const bst = Math.sin(bth);
        const na = brn * bst * Math.cos(bph),
          nb = brn * bst * Math.sin(bph),
          nup = brn * Math.cos(bth);
        if (m === 1) {
          s.x = nb;
          s.y = nup;
          s.z = na;
        } else if (m === 2) {
          s.x = nup;
          s.y = na;
          s.z = nb;
        } else {
          s.x = na;
          s.y = nb;
          s.z = nup;
        }
      }
      break;
    }
    case "msltoeSym3": {
      const mr2 = Math.max(s.x * s.x + s.y * s.y + s.z * s.z, 1e-12);
      s.w = 2 * Math.sqrt(mr2) * s.w + 1;
      let ma = s.y,
        mb = s.z;
      if (mb >= ma) {
        mb = -mb;
        if ((v[1] ?? 0) < 0.5) ma = -ma;
      }
      const mm = 1 - (ma * ma) / mr2;
      const mx = s.x;
      s.x = (mb * mb - mx * mx) * mm;
      s.y = 2 * mx * mb * mm * v[0];
      s.z = 2 * ma * Math.sqrt(mx * mx + mb * mb);
      break;
    }
    case "quadratic": {
      const qr = Math.hypot(s.x, s.y);
      s.w = 2 * qr * s.w + 1;
      const qx = s.x * s.x - s.y * s.y,
        qy = 2 * s.x * s.y;
      s.x = qx;
      s.y = qy;
      break;
    }
    default:
      break; // unknown op → no-op (forward-compatible, like the WGSL default)
  }
}

// Build a distance-estimate function de(x,y,z) for a formula. Faithful to
// shader.js: per-iter run the op stack, optional +c (or fixed Julia c), bail on
// escape; DE is escape-time (0.5·ln r·r/|w|) for bulb formulas, else IFS r/|w|.
export function makeDE(formula) {
  // CSG Phase 1b — a scene (formula.objects[]) maps as a combine over objects.
  // No `objects` ⇒ the single-object closure below, unchanged (additive invariant).
  if (Array.isArray(formula?.objects) && formula.objects.length > 0) {
    return makeSceneDE(formula);
  }
  // Hybrid iteration (IDEAS ①, docs/design/HYBRID_ITERATION.md §3.5) — mutually
  // exclusive with objects[] (§3.8), checked next.
  if (formula?.hybrid) {
    return makeHybridDE(formula);
  }
  const ops = activeOps(formula).map((o) => ({
    key: o.key,
    v: o.values || [],
  }));
  const iters = formula.iters ?? 8;
  // +c gate mirrors the renderer (renderer.js addGate = addC || julia): Julia mode
  // forces the per-iteration add ON — c is then the fixed jc constant instead of the
  // sample point. Gating on addC alone left Julia invisible for addC-off presets (#16).
  const addC = !!formula.addC || !!formula.julia;
  const escape = isEscapeTime(formula);
  const deOpt = effectiveDeOption(formula); // 3 = numeric finite-difference
  const bail = escape || deOpt === 3 ? 64.0 : 1.0e6; // matches preview.js bailoutFor()
  const jc = formula.julia ? formula.juliaC || [0, 0, 0] : null;
  // One orbit run → {r, aw}. The analytic DEs use it once; the numeric DE
  // (deOpt 3, mirrors shader.js orbitR/mapDE_single) samples r at 4 points.
  const run = (px, py, pz) => {
    const s = { x: px, y: py, z: pz, w: 1.0 };
    const cx = jc ? jc[0] : px,
      cy = jc ? jc[1] : py,
      cz = jc ? jc[2] : pz;
    for (let i = 0; i < iters; i++) {
      s.i = i; // iteration index for scaleDrift (SCALE_VARY.md §6.5)
      for (let o = 0; o < ops.length; o++) applyOp(ops[o].key, ops[o].v, s);
      if (addC) {
        s.x += cx;
        s.y += cy;
        s.z += cz;
      }
      if (s.x * s.x + s.y * s.y + s.z * s.z > bail) break;
    }
    return {
      r: Math.hypot(s.x, s.y, s.z),
      aw: Math.max(Math.abs(s.w), 1e-9),
    };
  };
  if (deOpt === 3) {
    return function de(px, py, pz) {
      const R = run(px, py, pz).r;
      const eps = 1e-4 * Math.max(1, Math.hypot(px, py, pz));
      const gx = (run(px + eps, py, pz).r - R) / eps,
        gy = (run(px, py + eps, pz).r - R) / eps,
        gz = (run(px, py, pz + eps).r - R) / eps;
      const gl = Math.hypot(gx, gy, gz);
      if (gl <= 1e-3) return 0.5 * Math.sqrt(Math.max(R, 0));
      return (R * Math.log(Math.max(R, 1))) / (gl + 0.06);
    };
  }
  return function de(px, py, pz) {
    const { r, aw } = run(px, py, pz);
    return escape ? (0.5 * Math.log(Math.max(r, 1e-9)) * r) / aw : r / aw;
  };
}

// Hybrid iteration (IDEAS ①, docs/design/HYBRID_ITERATION.md §3.5) — the CPU
// mirror of shader.js mapDE_hybrid. Alternates slot A / slot B by a {a,b}
// repeating schedule; deOption/bailout key off hybridDeFamily (the UNION of
// both slots' ops — §3.3), same as the WGSL/GLSL tiers. Julia is formula-level
// (§3.8): one seed, ORed into whichever slot's own addC decided to fire.
function makeHybridDE(formula) {
  const opsA = activeOps(formula).map((o) => ({
    key: o.key,
    v: o.values || [],
  }));
  const bRaw = formula.hybrid.b?.ops || [];
  const opsB = bRaw
    .filter((o) => !o.muted)
    .map((o) => ({ key: o.key, v: o.values || [] }));
  const sched = formula.hybrid.schedule || {};
  const a = Math.max(1, sched.a ?? 1);
  const b = Math.max(1, sched.b ?? 1);
  const period = a + b;
  const iters = formula.iters ?? 8;
  const addCA = !!formula.addC;
  const addCB = !!formula.hybrid.b?.addC;
  const julia = !!formula.julia;
  const family = hybridDeFamily(formula);
  const escape = family !== "ifs"; // 'escape' or the unsafe 'mixed' case (§3.3)
  const bail = escape ? 64.0 : 1.0e6;
  const jc = julia ? formula.juliaC || [0, 0, 0] : null;
  return function de(px, py, pz) {
    const s = { x: px, y: py, z: pz, w: 1.0 };
    const cx = jc ? jc[0] : px,
      cy = jc ? jc[1] : py,
      cz = jc ? jc[2] : pz;
    for (let i = 0; i < iters; i++) {
      s.i = i; // iteration index for scaleDrift (SCALE_VARY.md §6.5)
      const useB = i % period >= a;
      const ops = useB ? opsB : opsA;
      for (let o = 0; o < ops.length; o++) applyOp(ops[o].key, ops[o].v, s);
      if ((useB ? addCB : addCA) || julia) {
        s.x += cx;
        s.y += cy;
        s.z += cz;
      }
      if (s.x * s.x + s.y * s.y + s.z * s.z > bail) break;
    }
    const r = Math.hypot(s.x, s.y, s.z),
      aw = Math.max(Math.abs(s.w), 1e-9);
    return escape ? (0.5 * Math.log(Math.max(r, 1e-9)) * r) / aw : r / aw;
  };
}

// CSG Phase 1b — scene DE: combine over objects. Mirrors shader.js mapDE() with
// G.scene.x > 0: d = +INF; for each object k: pk = qrot(conj(q), p−origin)/uscale;
// dk = objectDE(pk)·uscale; d = combine(d, dk). objType 0 = IFS op-slice (objIterDE),
// 1 box · 2 sphere · 3 torus · 4 cylinder · 5 capsule · 6 plane (analytic SDFs).
// The scene marcher bailout is 1e6 (preview.js scene path).
function makeSceneDE(formula) {
  const SCENE_BAIL = 1.0e6; // matches preview.js scene-path bailout (G.prm.x)
  const built = formula.objects.map((o) => {
    const objType = Number(o.objType) & 0xf; // 0 IFS·1 box·2 sphere·3 torus·4 cyl·5 capsule·6 plane
    const tr = o.transform || {};
    const origin = tr.origin || o.origin || [0, 0, 0];
    const uscale = tr.uscale ?? o.uscale ?? 1;
    const q = eulerToQuat(tr.rot ?? o.rot ?? [0, 0, 0]);
    const qinv = [-q[0], -q[1], -q[2], q[3]]; // conjugate: world → local rotate
    const combine = (o.combine ?? o.combineType ?? 0) & 3; // 0 union · 1 smooth-union
    const blendK = o.blendK ?? 0;
    let child;
    if (objType === 1) {
      const he = o.primParam ?? o.halfExtent ?? o.radius ?? 1;
      child = (x, y, z) => boxDE(x, y, z, he);
    } else if (objType === 2) {
      const r = o.primParam ?? o.radius ?? o.halfExtent ?? 1;
      child = (x, y, z) => sphereDE(x, y, z, r);
    } else if (objType === 3) {
      const R = o.primParam ?? 1,
        r = o.primParam2 ?? 0.25;
      child = (x, y, z) => torusDE(x, y, z, R, r);
    } else if (objType === 4) {
      const r = o.primParam ?? 0.5,
        h = o.primParam2 ?? 0.5;
      child = (x, y, z) => cylinderDE(x, y, z, r, h);
    } else if (objType === 5) {
      const r = o.primParam ?? 0.3,
        h = o.primParam2 ?? 0.5;
      child = (x, y, z) => capsuleDE(x, y, z, r, h);
    } else if (objType === 6) {
      const thick = o.primParam ?? 0;
      child = (x, y, z) => planeDE(x, y, z, thick);
    } else {
      // IFS op-slice — objIterDE: per-object ops, iters, addC, deOption, Julia.
      const ops = activeOps({ ops: o.ops || [] }).map((op) => ({
        key: op.key,
        v: op.values || [],
      }));
      const iters = o.iters ?? 1;
      const escape = (o.deOption ?? 2) === 0; // deOption 0 = escape-time, else IFS
      const julia = !!o.julia;
      const addC = !!o.addC || julia; // flags bit0 || julia (objIterDE)
      const jc = julia ? o.juliaC || [0, 0, 0] : null;
      // Box-DE base (flags bit11): finalize IFS deOption-2 with boxDE(pos,he)/|w|
      // (flat cube faces) instead of length(pos)/|w| (round dust). he = primParam
      // (repurposed for IFS). Mirrors shader.js objIterDE / shader_gl.js.
      const boxBase = !escape && !!o.boxBase;
      const he = o.primParam ?? 1;
      child = (px, py, pz) => {
        const s = { x: px, y: py, z: pz, w: 1.0 };
        const cx = jc ? jc[0] : px,
          cy = jc ? jc[1] : py,
          cz = jc ? jc[2] : pz;
        for (let i = 0; i < iters; i++) {
          s.i = i; // iteration index for scaleDrift (SCALE_VARY.md §6.5)
          for (let k = 0; k < ops.length; k++) applyOp(ops[k].key, ops[k].v, s);
          if (addC) {
            s.x += cx;
            s.y += cy;
            s.z += cz;
          }
          if (s.x * s.x + s.y * s.y + s.z * s.z > SCENE_BAIL) break;
        }
        const r = Math.hypot(s.x, s.y, s.z),
          aw = Math.max(Math.abs(s.w), 1e-9);
        if (escape) return (0.5 * Math.log(Math.max(r, 1e-9)) * r) / aw;
        if (boxBase) return boxDE(s.x, s.y, s.z, he) / aw;
        return r / aw;
      };
    }
    return { origin, uscale, qinv, combine, blendK, child };
  });
  return function de(px, py, pz) {
    let d = 1.0e9;
    for (const ob of built) {
      const r = qrot(
        ob.qinv,
        px - ob.origin[0],
        py - ob.origin[1],
        pz - ob.origin[2],
      );
      const inv = 1 / ob.uscale;
      const dk = ob.child(r[0] * inv, r[1] * inv, r[2] * inv) * ob.uscale;
      // combine: 0 union · 1 smooth-union · 2 subtract · 3 intersect. Mirrors
      // shader.js mapDE() — subtract/intersect use max() (over-estimating; the
      // scene marches tighter to compensate, see preview.js sceneDeScale).
      if (ob.combine === 1) d = sminP(d, dk, ob.blendK);
      else if (ob.combine === 2)
        d = smaxP(d, -dk, ob.blendK); // subtract (blendK rounds the cut)
      else if (ob.combine === 3)
        d = smaxP(d, dk, ob.blendK); // intersect (blendK rounds the seam)
      else d = Math.min(d, dk);
    }
    return d;
  };
}

// Curated character ramps (dark → bright). All are HTML-safe (no < > & ") so the
// colored renderer can drop them into <span>s unescaped. `classic` is the
// original; `fine` trades for smoother tone (more levels), `blocks` for solid
// ink, `contrast` for punch. The app can also build a font-accurate ramp at
// runtime with calibrateRamp() (proposal #3).
export const RAMPS = {
  classic: " .:-=+*#%@",
  fine: " .'`:;~-+=ilfjtrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW8%B@",
  blocks: " ░▒▓█",
  contrast: " .:oa8#@",
};
const RAMP = RAMPS.classic; // dark → bright (default)

// Build a perceptually-even ramp (proposal #3). Given candidate glyphs and a
// coverage(char) → [0,1] ink-fraction measurer (the APP supplies one from a
// canvas; core stays DOM-free), pick `n` glyphs at evenly-spaced coverage so the
// brightness steps look uniform. Returns dark → bright; falls back to the classic
// ramp if the coverage signal is degenerate.
export function calibrateRamp(chars, coverage, n = 10) {
  const scored = [...chars]
    .map((ch) => [ch, coverage(ch)])
    .filter(([, v]) => Number.isFinite(v))
    .sort((a, b) => a[1] - b[1]);
  if (scored.length < 2) return RAMP;
  const lo = scored[0][1],
    hi = scored[scored.length - 1][1];
  if (hi - lo < 1e-6) return RAMP;
  const out = [];
  for (let i = 0; i < n; i++) {
    const target = lo + ((hi - lo) * i) / (n - 1);
    let best = scored[0],
      bd = Infinity;
    for (const s of scored) {
      const d = Math.abs(s[1] - target);
      if (d < bd) {
        bd = d;
        best = s;
      }
    }
    out.push(best[0]);
  }
  return out.join("");
}

// Edge glyphs indexed by the gradient-angle bucket: 0 = horizontal gradient
// (→ vertical edge), 1 = 45°, 2 = vertical gradient (→ horizontal edge), 3 = 135°.
// Screen y points down, so the diagonals come out as drawn.
const EDGE_GLYPHS = ["|", "/", "_", "\\"];
// Faint counterparts for the interior structure isolines (proposal #4) — light
// punctuation so they read as texture under the bold silhouette EDGE_GLYPHS.
const STRUCT_GLYPHS = [":", ",", ".", "`"];

// 4×4 ordered (Bayer) dither, normalised to (−0.5, 0.5): perturbs the ramp index
// so smooth tonal gradients break up into stipple instead of flat banding
// (proposal #5). Off by default — at ss = 1 / dither off the index is unchanged.
const BAYER4 = [
  [0, 8, 2, 10],
  [12, 4, 14, 6],
  [3, 11, 1, 9],
  [15, 7, 13, 5],
].map((row) => row.map((v) => (v + 0.5) / 16 - 0.5));
const rampIndex = (inten, last, dither, r, c) => {
  const x = inten * last + (dither ? BAYER4[r & 3][c & 3] : 0);
  const i = Math.round(x);
  return i < 0 ? 0 : i > last ? last : i;
};

const norm3 = (x, y, z) => {
  const L = Math.hypot(x, y, z) || 1;
  return [x / L, y / L, z / L];
};
const DEFAULT_LIGHT = norm3(-0.4, 0.55, 0.75);

// Shared sphere-trace pass behind BOTH ASCII renderers. Marches `ss`×`ss` rays
// per character cell (ss = 1 → one centered ray → the original behaviour) and
// returns a flat per-cell G-buffer: coverage, lit intensity, surface normal,
// depth, view ray, AO, and the nearest-hit position (for colour + edges). One
// march, two consumers — folds the two duplicated raymarchers into one (one
// fewer place the per-op math can drift out of sync with operators.js).
function traceGrid(formula, opts) {
  const {
    cols = 88,
    rows = 44,
    cam = makeCamera(formula.camera),
    lightDir = DEFAULT_LIGHT,
    eps = 0.0012,
    aspect: aspectOpt,
    ss = 1,
  } = opts;
  // A LOOSE analytic IFS DE (scale < 2, see stability.looseDE) over-estimates
  // distance, so the marcher oversteps thin/far surfaces and they vanish — the
  // hollow-back artifact. Mirror the GPU's loose treatment (preview.js: deScale
  // 0.3 + more steps). Explicit opts still win, so cpu.test #14's overstep/resolve
  // cases (which pass deScale/maxSteps) are intact, and non-loose presets keep the
  // fast 0.85/110 defaults (byte-identical).
  const loose = looseDE(formula);
  const maxSteps = opts.maxSteps ?? (loose ? 260 : 110);
  const deScale = opts.deScale ?? (loose ? 0.3 : 0.85);
  // Deep zoom (§5) — near/far scale off cam.dist instead of the old fixed
  // [0.02, 80] (mirrors shader.js/shader_gl.js); REF_DIST=24 keeps every
  // existing formula byte-identical (cam.dist defaults to 24 in makeCamera's
  // typical framing). Explicit opts still win.
  const tNear = opts.tNear ?? cam.dist * (0.02 / 24);
  const tFar = opts.tFar ?? cam.dist * (80 / 24);
  const de = makeDE(formula);
  const { eye, fwd, right, up } = cam.basis();
  const tanF = Math.tan(0.5 * cam.fov);
  // Match the on-screen pixel aspect when the caller passes it (so the ASCII
  // framing lines up with the GPU render); else assume cells are 2× tall.
  const aspect = aspectOpt ?? cols / (2 * rows);
  const n = cols * rows;
  const cov = new Float32Array(n); // 0..1 fraction of subrays that hit
  const inten = new Float32Array(n); // lit intensity, misses count as 0 (AA tone)
  const nx = new Float32Array(n),
    ny = new Float32Array(n),
    nz = new Float32Array(n);
  const depth = new Float32Array(n).fill(Infinity);
  const rdx = new Float32Array(n),
    rdy = new Float32Array(n),
    rdz = new Float32Array(n);
  const ao = new Float32Array(n);
  const hx = new Float32Array(n),
    hy = new Float32Array(n),
    hz = new Float32Array(n);
  const nSub = ss * ss;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const i = r * cols + c;
      let intenSum = 0,
        hits = 0,
        nxAcc = 0,
        nyAcc = 0,
        nzAcc = 0,
        bestT = Infinity;
      for (let sy = 0; sy < ss; sy++) {
        const ndcY = 1 - (2 * (r + (sy + 0.5) / ss)) / rows;
        for (let sx = 0; sx < ss; sx++) {
          const ndcX = -1 + (2 * (c + (sx + 0.5) / ss)) / cols;
          let dx =
            fwd[0] + ndcX * aspect * tanF * right[0] + ndcY * tanF * up[0];
          let dy =
            fwd[1] + ndcX * aspect * tanF * right[1] + ndcY * tanF * up[1];
          let dz =
            fwd[2] + ndcX * aspect * tanF * right[2] + ndcY * tanF * up[2];
          const dl = Math.hypot(dx, dy, dz) || 1;
          dx /= dl;
          dy /= dl;
          dz /= dl;
          let t = tNear,
            hit = false,
            st = 0;
          for (; st < maxSteps; st++) {
            const d =
              de(eye[0] + dx * t, eye[1] + dy * t, eye[2] + dz * t) * deScale;
            if (d < eps * t) {
              hit = true;
              break;
            }
            t += d;
            if (t > tFar) break;
          }
          if (!hit) continue;
          const px = eye[0] + dx * t,
            py = eye[1] + dy * t,
            pz = eye[2] + dz * t;
          // e scales with hit distance t (deep zoom §3.4/§5) — a fixed epsilon
          // straddles unrelated geometry once near/far isn't a fixed [0.02, 80].
          const e = Math.max(1e-6, Math.min(6e-4, t * 3e-5));
          const gx = de(px + e, py, pz) - de(px - e, py, pz);
          const gy = de(px, py + e, pz) - de(px, py - e, pz);
          const gz = de(px, py, pz + e) - de(px, py, pz - e);
          const gl = Math.hypot(gx, gy, gz) || 1;
          const nX = gx / gl,
            nY = gy / gl,
            nZ = gz / gl;
          const diff = Math.max(
            nX * lightDir[0] + nY * lightDir[1] + nZ * lightDir[2],
            0,
          );
          const aoS = 1 - st / maxSteps;
          let it = (0.18 + 0.82 * diff) * (0.45 + 0.55 * aoS);
          it = it < 0 ? 0 : it > 1 ? 1 : it;
          intenSum += it;
          hits++;
          nxAcc += nX;
          nyAcc += nY;
          nzAcc += nZ;
          if (t < bestT) {
            // Keep the NEAREST subray's hit for colour + edges (silhouette-stable).
            bestT = t;
            depth[i] = t;
            ao[i] = aoS;
            rdx[i] = dx;
            rdy[i] = dy;
            rdz[i] = dz;
            hx[i] = px;
            hy[i] = py;
            hz[i] = pz;
          }
        }
      }
      cov[i] = hits / nSub;
      // Misses contribute 0 → a half-covered silhouette cell dims naturally
      // (anti-aliased tone). At ss = 1 this is hit→it / miss→0: byte-identical
      // to the old single-ray march.
      inten[i] = intenSum / nSub;
      if (hits) {
        const nl = Math.hypot(nxAcc, nyAcc, nzAcc) || 1;
        nx[i] = nxAcc / nl;
        ny[i] = nyAcc / nl;
        nz[i] = nzAcc / nl;
      }
    }
  }
  return {
    cols,
    rows,
    cov,
    inten,
    nx,
    ny,
    nz,
    depth,
    rdx,
    rdy,
    rdz,
    ao,
    hx,
    hy,
    hz,
  };
}

// Geometry-native edge layer (proposal #1). Screen-space ASCII shaders fake
// contours from a 2D luminance image with Difference-of-Gaussians + Sobel; we
// have the real G-buffer, so we read edges straight off it: a Sobel on the lit
// intensity (background = 0) catches silhouettes + shading ridges, a depth jump
// catches occluding contours where two lit surfaces overlap, and a normal
// discontinuity catches creases. The gradient angle picks the glyph (| / _ \).
// Returns Int8Array of glyph-bucket per cell, or -1 for "no edge".
const SOBEL_X = [
  [-1, 0, 1],
  [-2, 0, 2],
  [-1, 0, 1],
];
function detectEdges(g, opts) {
  const { cols, rows, inten, cov, depth, nx, ny, nz } = g;
  // Defaults tuned on the Mandelbox: strong contours + silhouettes, but not the
  // "every fractal fold is an edge" noise a low threshold gives. Lower them to
  // etch interior self-similar detail; raise for a bare outline.
  const { edgeThresh = 0.6, creaseDot = 0.0, depthCliff = 0.6 } = opts;
  const out = new Int8Array(cols * rows).fill(-1);
  const idx = (r, c) =>
    r < 0 || c < 0 || r >= rows || c >= cols ? -1 : r * cols + c;
  // Sobel on a field sampled by `pick` (out-of-bounds → 0).
  const sobel = (r, c, pick) => {
    let gx = 0,
      gy = 0;
    for (let dr = -1; dr <= 1; dr++)
      for (let dc = -1; dc <= 1; dc++) {
        const j = idx(r + dr, c + dc);
        const v = j < 0 ? 0 : pick(j);
        gx += SOBEL_X[dr + 1][dc + 1] * v;
        gy += SOBEL_X[dc + 1][dr + 1] * v; // transpose of Sobel-X = Sobel-Y
      }
    return [gx / 4, gy / 4];
  };
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const i = r * cols + c;
      if (cov[i] <= 0) continue; // draw the outline on foreground cells only
      let [gx, gy] = sobel(r, c, (j) => inten[j]);
      let mag = Math.hypot(gx, gy);
      let cliff = false,
        crease = false;
      for (const [dr, dc] of [
        [-1, 0],
        [1, 0],
        [0, -1],
        [0, 1],
      ]) {
        const j = idx(r + dr, c + dc);
        if (j < 0 || cov[j] <= 0) continue;
        if (
          Math.abs(depth[i] - depth[j]) >
          depthCliff * Math.max(depth[i], 1e-3)
        )
          cliff = true;
        if (nx[i] * nx[j] + ny[i] * ny[j] + nz[i] * nz[j] < creaseDot)
          crease = true;
      }
      if (mag < edgeThresh && !cliff && !crease) continue;
      // Pure silhouette/occlusion (flat shading across the edge): take the
      // direction from the depth field instead, over foreground neighbours only.
      if (mag < 1e-3) {
        [gx, gy] = sobel(r, c, (j) => (cov[j] > 0 ? depth[j] : depth[i]));
      }
      let a = Math.atan2(gy, gx);
      if (a < 0) a += Math.PI; // edge orientation is modulo π
      out[i] = Math.round(a / (Math.PI / 4)) % 4;
    }
  }
  return out;
}

// Per-cell orbit-trap value [0,1] over the hit surface — the scalar the structure
// layer contours. Uses the trap measure (defined for every formula), independent
// of the colour mode, so the interior bands are consistent.
function structField(formula, g) {
  const measure = makeIterMeasure(formula, "trap");
  const n = g.cols * g.rows;
  const s = new Float32Array(n);
  for (let i = 0; i < n; i++)
    if (g.cov[i] > 0)
      s[i] = Math.min(measure(g.hx[i], g.hy[i], g.hz[i]) / 1.5, 1);
  return s;
}

// Interior "structure" isolines (proposal #4) — the fractal-native feature a
// screen-space ASCII shader fundamentally cannot do, because the orbit-trap field
// isn't in the rendered image. Quantise the trap field into `structBands` bands
// and mark cells on a band boundary (an isoline); the field gradient picks the
// glyph direction. Drawn only INSIDE the silhouette, under the bolder edge layer.
function detectContours(s, cov, cols, rows, opts) {
  const { structBands = 7 } = opts;
  const out = new Int8Array(cols * rows).fill(-1);
  const idx = (r, c) =>
    r < 0 || c < 0 || r >= rows || c >= cols ? -1 : r * cols + c;
  const band = (i) => Math.floor(s[i] * structBands);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const i = r * cols + c;
      if (cov[i] <= 0) continue;
      const b = band(i);
      let isoline = false,
        gx = 0,
        gy = 0;
      for (let dr = -1; dr <= 1; dr++)
        for (let dc = -1; dc <= 1; dc++) {
          const j = idx(r + dr, c + dc);
          const v = j < 0 || cov[j] <= 0 ? s[i] : s[j];
          gx += SOBEL_X[dr + 1][dc + 1] * v;
          gy += SOBEL_X[dc + 1][dr + 1] * v;
          if (j >= 0 && cov[j] > 0 && band(j) !== b) isoline = true;
        }
      if (!isoline) continue;
      let a = Math.atan2(gy, gx);
      if (a < 0) a += Math.PI;
      out[i] = Math.round(a / (Math.PI / 4)) % 4;
    }
  }
  return out;
}

// Sphere-trace the formula into an ASCII string (rows joined by \n). `cam` is an
// optional makeCamera() instance (for live orbit); else built from formula.camera.
// `ss` supersamples each cell (anti-aliasing); `edges:true` overlays the
// geometry-native contour glyphs.
export function renderAscii(formula, opts = {}) {
  const {
    cols = 88,
    rows = 44,
    ramp = RAMP,
    light = DEFAULT_LIGHT,
    edges = false,
    structure = false,
    dither = false,
  } = opts;
  const g = traceGrid(formula, { ...opts, cols, rows, lightDir: light });
  const edge = edges ? detectEdges(g, opts) : null;
  const struct = structure
    ? detectContours(structField(formula, g), g.cov, cols, rows, opts)
    : null;
  const last = ramp.length - 1;
  const out = [];
  for (let r = 0; r < rows; r++) {
    let line = "";
    for (let c = 0; c < cols; c++) {
      const i = r * cols + c;
      line +=
        edge && edge[i] >= 0
          ? EDGE_GLYPHS[edge[i]]
          : struct && struct[i] >= 0
            ? STRUCT_GLYPHS[struct[i]]
            : ramp[rampIndex(g.inten[i], last, dither, r, c)];
    }
    out.push(line); // keep the full cols×rows rectangle so the shape stays centered
  }
  return out.join("\n");
}

// ── colored ASCII ────────────────────────────────────────────────────────────
// Orbit-trap / escape-iteration re-runs (for color modes 1 and 2), mirroring the
// WGSL orbitTrap()/escapeIter(). Only built when the active color mode needs them.
function makeIterMeasure(formula, kind) {
  // Hybrid iteration (§3.6) — glow/bands coloring "just works" on a hybrid
  // (unlike CSG scenes, which restrict to surface mode): thread the same
  // schedule branch makeHybridDE uses, kept in lockstep by construction (both
  // read formula.hybrid the same way). Objects-first tie-break, matching
  // makeDE above (Formula Outline Step 3 §4a): a malformed dual-set formula
  // (objects + hybrid both set) must color the same shape makeDE marched.
  if (
    formula?.hybrid &&
    !(Array.isArray(formula.objects) && formula.objects.length > 0)
  ) {
    const opsA = activeOps(formula).map((o) => ({
      key: o.key,
      v: o.values || [],
    }));
    const bRaw = formula.hybrid.b?.ops || [];
    const opsB = bRaw
      .filter((o) => !o.muted)
      .map((o) => ({ key: o.key, v: o.values || [] }));
    const sched = formula.hybrid.schedule || {};
    const a = Math.max(1, sched.a ?? 1);
    const b = Math.max(1, sched.b ?? 1);
    const period = a + b;
    const iters = formula.iters ?? 8;
    const addCA = !!formula.addC;
    const addCB = !!formula.hybrid.b?.addC;
    const julia = !!formula.julia;
    const escape = hybridDeFamily(formula) !== "ifs";
    const bail = escape ? 64.0 : 1.0e6;
    const jc = julia ? formula.juliaC || [0, 0, 0] : null;
    return function measure(px, py, pz) {
      const s = { x: px, y: py, z: pz, w: 1.0 };
      const cx = jc ? jc[0] : px,
        cy = jc ? jc[1] : py,
        cz = jc ? jc[2] : pz;
      let tr = 1e9,
        esc = iters;
      for (let i = 0; i < iters; i++) {
        s.i = i; // iteration index for scaleDrift (SCALE_VARY.md §6.5)
        const useB = i % period >= a;
        const ops = useB ? opsB : opsA;
        for (let o = 0; o < ops.length; o++) applyOp(ops[o].key, ops[o].v, s);
        if ((useB ? addCB : addCA) || julia) {
          s.x += cx;
          s.y += cy;
          s.z += cz;
        }
        const r2 = s.x * s.x + s.y * s.y + s.z * s.z;
        if (kind === "trap") tr = Math.min(tr, Math.sqrt(r2));
        if (r2 > bail) {
          esc = i;
          break;
        }
      }
      return kind === "trap" ? tr : esc / Math.max(iters, 1);
    };
  }
  const ops = activeOps(formula).map((o) => ({
    key: o.key,
    v: o.values || [],
  }));
  const iters = formula.iters ?? 8;
  // +c gate mirrors the renderer (addGate = addC || julia) — see makeDE above (#16).
  const addC = !!formula.addC || !!formula.julia;
  const escape = isEscapeTime(formula);
  const bail = escape ? 64.0 : 1.0e6;
  const jc = formula.julia ? formula.juliaC || [0, 0, 0] : null;
  return function measure(px, py, pz) {
    const s = { x: px, y: py, z: pz, w: 1.0 };
    const cx = jc ? jc[0] : px,
      cy = jc ? jc[1] : py,
      cz = jc ? jc[2] : pz;
    let tr = 1e9,
      esc = iters;
    for (let i = 0; i < iters; i++) {
      s.i = i; // iteration index for scaleDrift (SCALE_VARY.md §6.5)
      for (let o = 0; o < ops.length; o++) applyOp(ops[o].key, ops[o].v, s);
      if (addC) {
        s.x += cx;
        s.y += cy;
        s.z += cz;
      }
      const r2 = s.x * s.x + s.y * s.y + s.z * s.z;
      if (kind === "trap") tr = Math.min(tr, Math.sqrt(r2));
      if (r2 > bail) {
        esc = i;
        break;
      }
    }
    return kind === "trap" ? tr : esc / Math.max(iters, 1);
  };
}

const clamp01 = (v) => Math.max(0, Math.min(1, v));

// Shaded surface colour [0..255], mirroring the WGSL fragment's albedo + light:
// cosine palette OR colA→colB by mixT, then diffuse+ambient+AO and a rim glow,
// gamma-corrected. (Skips the WGSL distance-fade/spec — keeps the chars punchy.)
function shadeRGB(coloring, mixT, nx, ny, nz, dx, dy, dz, ao) {
  const L = coloring.light || {};
  const pal = coloring.palette || {};
  const amb = L.ambient ?? 0.16,
    rimAmt = L.rim ?? 0.45,
    intensity = L.intensity ?? 1.0;
  const ld = norm3(...(L.dir || [0.45, -0.65, 0.75]));
  const diff = Math.max(nx * ld[0] + ny * ld[1] + nz * ld[2], 0);
  const rim = Math.pow(1 - Math.max(-(nx * dx + ny * dy + nz * dz), 0), 2);
  const B = coloring.colB || [0.18, 0.62, 0.74];
  // sRGB → linear: colors are authored/picked in sRGB; linearize before lighting
  // so the 1/2.2 encode below round-trips them to the picked color (issue #6).
  const s2l = (x) => Math.pow(Math.max(x, 0), 2.2);
  const Blin = B.map(s2l);
  let alb;
  if (pal.on) {
    const a = pal.a || [0.5, 0.5, 0.5],
      b = pal.b || [0.5, 0.5, 0.5];
    const c = pal.c || [1, 1, 1],
      d = pal.d || [0, 0.33, 0.67];
    alb = [0, 1, 2].map((i) =>
      clamp01(a[i] + b[i] * Math.cos(6.2831853 * (c[i] * mixT + d[i]))),
    );
  } else {
    const A = coloring.colA || [0.86, 0.46, 0.18];
    alb = [0, 1, 2].map((i) => A[i] + (B[i] - A[i]) * mixT);
  }
  alb = alb.map(s2l); // sRGB→linear (issue #6)
  const sh = (amb + (1 - amb) * diff) * (0.35 + 0.65 * ao);
  return [0, 1, 2].map((i) =>
    Math.round(
      255 *
        Math.pow(
          Math.max(
            (alb[i] * sh + Blin[i] * (rim * rimAmt * ao)) * intensity,
            0,
          ),
          1 / 2.2,
        ),
    ),
  );
}

const q8 = (v) => Math.min(255, Math.round(v / 8) * 8); // quantize → longer runs
const hex2 = (v) => q8(v).toString(16).padStart(2, "0");

// Cell-fill highlight (Reddit tip): when the caller paints each cell's background
// with the surface colour (the "average colour under the char box"), the glyph
// ink would vanish into it, so lift the ink toward white — it reads as a light
// stipple over a solid colour field. Fixes the washed-out look sparse ink on a
// flat page gives: dark/low-coverage cells now show their true surface colour
// instead of leaking the muted page background through the gaps.
const CELL_LIFT = 0.4;
const lift = (px) => [
  px[0] + (255 - px[0]) * CELL_LIFT,
  px[1] + (255 - px[1]) * CELL_LIFT,
  px[2] + (255 - px[2]) * CELL_LIFT,
];

// Like renderAscii, but each character is tinted by the formula's coloring (same
// palette/mode/lighting the GPU uses). Returns { text, html }: `text` is the plain
// char grid (for copy-as-text), `html` is colour-run <span>s for display. The ramp
// chars (` .:-=+*#%@`) contain no HTML-special chars, so the runs need no escaping.
// Shared colour pass behind the HTML and ANSI renderers: trace + edge + structure
// layers, then per cell resolve the final glyph and its RGB (null = empty). The
// two public renderers below only differ in how they serialise these cells.
function shadeGrid(formula, opts) {
  const {
    cols = 88,
    rows = 44,
    ramp = RAMP,
    coloring = {},
    edges = false,
    structure = false,
    dither = false,
  } = opts;
  const mode = coloring.mode || 0;
  const trap = mode === 1 ? makeIterMeasure(formula, "trap") : null;
  const esc = mode === 2 ? makeIterMeasure(formula, "escape") : null;
  // CHAR intensity uses the SAME light the colour does (so the form reads the
  // way the palette is lit) — this is what the old per-cell `ld` did.
  const lightDir = norm3(
    ...((coloring.light && coloring.light.dir) || [0.45, -0.65, 0.75]),
  );
  const g = traceGrid(formula, { ...opts, cols, rows, lightDir });
  const edge = edges ? detectEdges(g, opts) : null;
  const struct = structure
    ? detectContours(structField(formula, g), g.cov, cols, rows, opts)
    : null;
  const last = ramp.length - 1;
  const n = cols * rows;
  const chars = new Array(n).fill(" ");
  const rgb = new Array(n).fill(null);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const i = r * cols + c;
      if (g.cov[i] <= 0) continue;
      const nx = g.nx[i],
        ny = g.ny[i],
        nz = g.nz[i];
      const mixT =
        mode === 2
          ? esc(g.hx[i], g.hy[i], g.hz[i])
          : mode === 1
            ? Math.min(trap(g.hx[i], g.hy[i], g.hz[i]) / 1.5, 1)
            : 0.5 + 0.5 * nz;
      // CHAR (density) from geometry lighting only — same as the mono ramp — so
      // the 3D form reads clearly. COLOR is the shaded albedo (hue), separately,
      // or the depth gets flattened by a bright palette. Edge > structure > fill:
      // a contour glyph swaps the density glyph but keeps the surface colour.
      chars[i] =
        edge && edge[i] >= 0
          ? EDGE_GLYPHS[edge[i]]
          : struct && struct[i] >= 0
            ? STRUCT_GLYPHS[struct[i]]
            : ramp[rampIndex(g.inten[i], last, dither, r, c)];
      rgb[i] = shadeRGB(
        coloring,
        mixT,
        nx,
        ny,
        nz,
        g.rdx[i],
        g.rdy[i],
        g.rdz[i],
        g.ao[i],
      );
    }
  }
  return { cols, rows, chars, rgb };
}

export function renderAsciiColored(formula, opts = {}) {
  const { cellBg = false } = opts;
  const { cols, rows, chars, rgb } = shadeGrid(formula, opts);
  const textRows = [],
    htmlRows = [];
  for (let r = 0; r < rows; r++) {
    let text = "",
      html = "",
      // Runs key on the SURFACE colour (both ink and fill derive from it, so a
      // run is uniform in both). null = empty cell (page background shows).
      runKey = undefined,
      runFg = null,
      runBg = null,
      runStr = "";
    const emit = () => {
      if (!runStr) return;
      html +=
        runFg == null
          ? runStr
          : `<span style="color:${runFg}${runBg ? `;background:${runBg}` : ""}">${runStr}</span>`;
      runStr = "";
    };
    for (let c = 0; c < cols; c++) {
      const i = r * cols + c;
      const px = rgb[i];
      const key = px ? `#${hex2(px[0])}${hex2(px[1])}${hex2(px[2])}` : null;
      if (key !== runKey) {
        emit();
        runKey = key;
        if (!px) {
          runFg = runBg = null;
        } else if (cellBg) {
          const l = lift(px);
          runFg = `#${hex2(l[0])}${hex2(l[1])}${hex2(l[2])}`;
          runBg = key; // surface colour fills the whole cell
        } else {
          runFg = key; // ink IS the surface colour (original look)
          runBg = null;
        }
      }
      text += chars[i];
      runStr += chars[i];
    }
    emit();
    textRows.push(text);
    htmlRows.push(html);
  }
  return { text: textRows.join("\n"), html: htmlRows.join("\n") };
}

// 24-bit-colour ANSI for terminals (proposal #5). Same glyphs as the HTML
// renderer; each run of one colour gets a single truecolor escape, reset at each
// line end. Returns { text, ansi }: `text` is the uncoloured grid (plain copy),
// `ansi` is the escape-coded string to print/paste into a truecolor terminal.
const ANSI_RESET = "[0m";
export function renderAsciiAnsi(formula, opts = {}) {
  const { cellBg = false } = opts;
  const { cols, rows, chars, rgb } = shadeGrid(formula, opts);
  const textRows = [],
    ansiRows = [];
  for (let r = 0; r < rows; r++) {
    let text = "",
      ansi = "",
      cur = null;
    for (let c = 0; c < cols; c++) {
      const i = r * cols + c;
      const px = rgb[i];
      const key = px ? `${px[0]};${px[1]};${px[2]}` : null;
      if (key !== cur) {
        if (!px) ansi += ANSI_RESET;
        else if (cellBg) {
          const l = lift(px);
          // lifted ink (38) over the surface colour as background (48)
          ansi += `[38;2;${q8(l[0])};${q8(l[1])};${q8(l[2])}m[48;2;${key}m`;
        } else ansi += `[38;2;${key}m`;
        cur = key;
      }
      ansi += chars[i];
      text += chars[i];
    }
    if (cur) ansi += ANSI_RESET;
    textRows.push(text);
    ansiRows.push(ansi);
  }
  return { text: textRows.join("\n"), ansi: ansiRows.join("\n") };
}

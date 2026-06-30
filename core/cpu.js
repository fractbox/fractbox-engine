// CPU fallback renderer — a plain-JS port of the engine's distance estimate
// (see shader.js `mapDE`) plus an ASCII raymarcher. No GPU, no DOM, no deps:
// pure ESM so it runs anywhere core/ runs (the no-WebGPU fallback, the OSS demo,
// headless tooling, the stacking games). Returns text; the app wraps it in a UI.
//
// ⚠ The per-op math below MIRRORS the WGSL in operators.js. The two are the same
// math in two languages — if you change an operator there, change it here too
// (cpu.test guards that every preset still produces a finite, non-empty render).

import { makeCamera } from "./camera.js";
import { isEscapeTime, activeOps } from "./operators.js";

const D2R = Math.PI / 180;
const rad = (d) => d * D2R;

// ── per-op apply: mutate state s = {x,y,z,w}; mirrors operators.js `wgsl` ──────
function applyOp(key, v, s) {
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
  const bail = escape ? 64.0 : 1.0e6; // matches preview.js bailoutFor()
  const jc = formula.julia ? formula.juliaC || [0, 0, 0] : null;
  return function de(px, py, pz) {
    const s = { x: px, y: py, z: pz, w: 1.0 };
    const cx = jc ? jc[0] : px,
      cy = jc ? jc[1] : py,
      cz = jc ? jc[2] : pz;
    for (let i = 0; i < iters; i++) {
      for (let o = 0; o < ops.length; o++) applyOp(ops[o].key, ops[o].v, s);
      if (addC) {
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

const RAMP = " .:-=+*#%@"; // dark → bright
const norm3 = (x, y, z) => {
  const L = Math.hypot(x, y, z) || 1;
  return [x / L, y / L, z / L];
};
const DEFAULT_LIGHT = norm3(-0.4, 0.55, 0.75);

// Sphere-trace the formula into an ASCII string (rows joined by \n). `cam` is an
// optional makeCamera() instance (for live orbit); else built from formula.camera.
export function renderAscii(formula, opts = {}) {
  const {
    cols = 88,
    rows = 44,
    cam = makeCamera(formula.camera),
    light = DEFAULT_LIGHT,
    maxSteps = 110,
    deScale = 0.85,
    eps = 0.0012,
    ramp = RAMP,
    aspect: aspectOpt,
  } = opts;
  const de = makeDE(formula);
  const { eye, fwd, right, up } = cam.basis();
  const tanF = Math.tan(0.5 * cam.fov);
  // Match the on-screen pixel aspect when the caller passes it (so the ASCII
  // framing lines up with the GPU render); else assume cells are 2× tall.
  const aspect = aspectOpt ?? cols / (2 * rows);
  const E = 0.0009; // normal finite-difference epsilon
  const out = [];
  for (let r = 0; r < rows; r++) {
    const ndcY = 1 - (2 * (r + 0.5)) / rows;
    let line = "";
    for (let c = 0; c < cols; c++) {
      const ndcX = -1 + (2 * (c + 0.5)) / cols;
      let dx = fwd[0] + ndcX * aspect * tanF * right[0] + ndcY * tanF * up[0];
      let dy = fwd[1] + ndcX * aspect * tanF * right[1] + ndcY * tanF * up[1];
      let dz = fwd[2] + ndcX * aspect * tanF * right[2] + ndcY * tanF * up[2];
      const dl = Math.hypot(dx, dy, dz) || 1;
      dx /= dl;
      dy /= dl;
      dz /= dl;
      let t = 0.02,
        hit = false,
        steps = 0;
      for (; steps < maxSteps; steps++) {
        const d =
          de(eye[0] + dx * t, eye[1] + dy * t, eye[2] + dz * t) * deScale;
        if (d < eps * t) {
          hit = true;
          break;
        }
        t += d;
        if (t > 80) break;
      }
      if (!hit) {
        line += " ";
        continue;
      }
      const hx = eye[0] + dx * t,
        hy = eye[1] + dy * t,
        hz = eye[2] + dz * t;
      const gx = de(hx + E, hy, hz) - de(hx - E, hy, hz);
      const gy = de(hx, hy + E, hz) - de(hx, hy - E, hz);
      const gz = de(hx, hy, hz + E) - de(hx, hy, hz - E);
      const gl = Math.hypot(gx, gy, gz) || 1;
      const diff = Math.max(
        (gx / gl) * light[0] + (gy / gl) * light[1] + (gz / gl) * light[2],
        0,
      );
      const ao = 1 - steps / maxSteps;
      let inten = (0.18 + 0.82 * diff) * (0.45 + 0.55 * ao);
      inten = Math.max(0, Math.min(1, inten));
      line += ramp[Math.round(inten * (ramp.length - 1))];
    }
    out.push(line); // keep the full cols×rows rectangle so the shape stays centered
  }
  return out.join("\n");
}

// ── colored ASCII ────────────────────────────────────────────────────────────
// Orbit-trap / escape-iteration re-runs (for color modes 1 and 2), mirroring the
// WGSL orbitTrap()/escapeIter(). Only built when the active color mode needs them.
function makeIterMeasure(formula, kind) {
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

const hex2 = (v) =>
  Math.min(255, Math.round(v / 8) * 8)
    .toString(16)
    .padStart(2, "0"); // quantize → longer runs

// Like renderAscii, but each character is tinted by the formula's coloring (same
// palette/mode/lighting the GPU uses). Returns { text, html }: `text` is the plain
// char grid (for copy-as-text), `html` is colour-run <span>s for display. The ramp
// chars (` .:-=+*#%@`) contain no HTML-special chars, so the runs need no escaping.
export function renderAsciiColored(formula, opts = {}) {
  const {
    cols = 88,
    rows = 44,
    cam = makeCamera(formula.camera),
    maxSteps = 110,
    deScale = 0.85,
    eps = 0.0012,
    ramp = RAMP,
    coloring = {},
    aspect: aspectOpt,
  } = opts;
  const de = makeDE(formula);
  const mode = coloring.mode || 0;
  const trap = mode === 1 ? makeIterMeasure(formula, "trap") : null;
  const esc = mode === 2 ? makeIterMeasure(formula, "escape") : null;
  const { eye, fwd, right, up } = cam.basis();
  const tanF = Math.tan(0.5 * cam.fov);
  const aspect = aspectOpt ?? cols / (2 * rows);
  const E = 0.0009;
  const textRows = [],
    htmlRows = [];
  for (let r = 0; r < rows; r++) {
    const ndcY = 1 - (2 * (r + 0.5)) / rows;
    let text = "",
      html = "",
      runColor = undefined,
      runStr = "";
    const emit = () => {
      if (!runStr) return;
      html +=
        runColor == null
          ? runStr
          : `<span style="color:${runColor}">${runStr}</span>`;
      runStr = "";
    };
    for (let c = 0; c < cols; c++) {
      const ndcX = -1 + (2 * (c + 0.5)) / cols;
      let dx = fwd[0] + ndcX * aspect * tanF * right[0] + ndcY * tanF * up[0];
      let dy = fwd[1] + ndcX * aspect * tanF * right[1] + ndcY * tanF * up[1];
      let dz = fwd[2] + ndcX * aspect * tanF * right[2] + ndcY * tanF * up[2];
      const dl = Math.hypot(dx, dy, dz) || 1;
      dx /= dl;
      dy /= dl;
      dz /= dl;
      let t = 0.02,
        hit = false,
        steps = 0;
      for (; steps < maxSteps; steps++) {
        const d =
          de(eye[0] + dx * t, eye[1] + dy * t, eye[2] + dz * t) * deScale;
        if (d < eps * t) {
          hit = true;
          break;
        }
        t += d;
        if (t > 80) break;
      }
      let ch = " ",
        color = null;
      if (hit) {
        const hx = eye[0] + dx * t,
          hy = eye[1] + dy * t,
          hz = eye[2] + dz * t;
        const gx = de(hx + E, hy, hz) - de(hx - E, hy, hz);
        const gy = de(hx, hy + E, hz) - de(hx, hy - E, hz);
        const gz = de(hx, hy, hz + E) - de(hx, hy, hz - E);
        const gl = Math.hypot(gx, gy, gz) || 1;
        const nx = gx / gl,
          ny = gy / gl,
          nz = gz / gl;
        const ao = 1 - steps / maxSteps;
        const mixT =
          mode === 2
            ? esc(hx, hy, hz)
            : mode === 1
              ? Math.min(trap(hx, hy, hz) / 1.5, 1)
              : 0.5 + 0.5 * nz;
        // CHAR (density) from geometry lighting only — same as the mono ramp — so
        // the 3D form reads clearly. COLOR is the shaded albedo (hue), separately,
        // or the depth gets flattened by a bright palette.
        const ld = norm3(
          ...((coloring.light && coloring.light.dir) || [0.45, -0.65, 0.75]),
        );
        const diff = Math.max(nx * ld[0] + ny * ld[1] + nz * ld[2], 0);
        const inten = clamp01((0.18 + 0.82 * diff) * (0.45 + 0.55 * ao));
        ch = ramp[Math.round(inten * (ramp.length - 1))];
        const rgb = shadeRGB(coloring, mixT, nx, ny, nz, dx, dy, dz, ao);
        color = `#${hex2(rgb[0])}${hex2(rgb[1])}${hex2(rgb[2])}`;
      }
      text += ch;
      if (color !== runColor) {
        emit();
        runColor = color;
      }
      runStr += ch;
    }
    emit();
    textRows.push(text);
    htmlRows.push(html);
  }
  return { text: textRows.join("\n"), html: htmlRows.join("\n") };
}

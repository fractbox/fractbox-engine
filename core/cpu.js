// CPU fallback renderer — a plain-JS port of the engine's distance estimate
// (see shader.js `mapDE`) plus an ASCII raymarcher. No GPU, no DOM, no deps:
// pure ESM so it runs anywhere core/ runs (the no-WebGPU fallback, the OSS demo,
// headless tooling, the stacking games). Returns text; the app wraps it in a UI.
//
// ⚠ The per-op math below MIRRORS the WGSL in operators.js. The two are the same
// math in two languages — if you change an operator there, change it here too
// (cpu.test guards that every preset still produces a finite, non-empty render).

import { makeCamera } from "./camera.js";
import {
  isEscapeTime,
  effectiveDeOption,
  activeOps,
  isApproxDE,
} from "./operators.js";
import { looseDE, hybridDeFamily } from "./stability.js";
import { normalizeSceneObject } from "./sceneobj.js";
import {
  TNEAR_K,
  TFAR_K,
  TFAR_MIN,
  TFAR_UNBOUNDED_MUL,
  STEPS_UNBOUNDED_MUL,
  unboundedScene,
} from "./renderpolicy.js";
import { BAILOUT_ESCAPE, BAILOUT_IFS } from "./limits.js";
import { sampleStops } from "./oklab.js";
// applyOp/parseHybrid/makeOrbit used to live here — split into cpuorbit.js
// (#266) so evaluate.js's generator oracle (measure(), on the app boot path
// via vary.js) can import the ~1000-line orbit engine WITHOUT dragging in the
// ~2400 lines of scene/leaf/render/ASCII code below. cpu.js still needs all
// three for its own scene (buildSceneObjs) and coloring (makeIterMeasure/
// makePainterMeasure) code — re-imported, not duplicated.
import { applyOp, parseHybrid, hybridSlotAt, makeOrbit } from "./cpuorbit.js";

// ── CSG Phase 1b — scene math (mirrors shader.js qrot/boxDE/sphereDE/sminP) ────
// Object normalization (fallback chains + eulerToQuat) is hoisted to
// ./sceneobj.js (the one JS copy shared with the renderers).

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

// Build a distance-estimate function de(x,y,z) for a formula. Faithful to
// shader.js: per-iter run the op stack, optional +c (or fixed Julia c), bail on
// escape; DE is escape-time (0.5·ln r·r/|w|) for bulb formulas, else IFS r/|w|.
// Hybrid formulas (mutually exclusive with objects[], §3.8) alternate slot A /
// slot B inside the runner — the CPU mirror of shader.js mapDE_hybrid.
// iters (optional) — the effective iteration count from the render policy
// (renderpolicy.js effectiveIters: auto-detail zoom boost + Detail-slider
// override). The ASCII tier threads it in so the CPU march resolves the same
// fine structure the GPU shader does once zoomed (#181). Undefined ⇒ the
// formula's own iters, exactly as before. Scenes carry per-object iters (§14),
// so the override applies to the single-orbit / hybrid path only.
export function makeDE(formula, iters) {
  // CSG Phase 1b — a scene (formula.objects[]) maps as a combine over objects.
  // No `objects` ⇒ the single-orbit closure below, unchanged (additive invariant).
  if (Array.isArray(formula?.objects) && formula.objects.length > 0) {
    return makeSceneDE(formula);
  }
  // One orbit run → {r, aw}. The analytic DEs use it once; the numeric DE
  // (deOpt 3, mirrors shader.js orbitR/mapDE_single) samples r at 4 points.
  const run = makeOrbit(formula, iters != null ? { iters } : {});
  // #418 — mirror the GPU DE-family selector. capturesettle.js writes
  // colA.w = effectiveDeOption(f) (flat) / the hybridDeFamily verdict (hybrid),
  // and shader.js mapDE_single/mapDE_hybrid pick log-DE iff colA.w < 1. The CPU
  // previously keyed the finalize off run.escape (= isEscapeTime), so a user-set
  // deOption 0 on a non-escape-time stack was a silent no-op here (GPU logged,
  // CPU stayed on r/|w|). run.escape still drives the ORBIT bailout, which the GPU
  // keys off isEscapeTime too (bailoutFor) — that stays correct and unchanged.
  const eff = effectiveDeOption(formula);
  const logFinalize = formula?.hybrid
    ? hybridDeFamily(formula) === "ifs"
      ? (formula.deOption ?? 2) === 0
      : true
    : eff === 0;
  if (!formula?.hybrid && eff === 3) {
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
    return logFinalize ? (0.5 * Math.log(Math.max(r, 1e-9)) * r) / aw : r / aw;
  };
}

// Shape-leaf dispatch table (leaves.js registry ids → the analytic SDFs above).
// The JS bodies live HERE, same core-module split as the op table (applyOp) —
// leaves.js carries only the WGSL/GLSL strings + metadata. D0 §2.2.
const fract = (v) => v - Math.floor(v);
const box2 = (bx, by, hx, hy) => {
  const qx = Math.abs(bx) - hx,
    qy = Math.abs(by) - hy;
  return (
    Math.hypot(Math.max(qx, 0), Math.max(qy, 0)) + Math.min(Math.max(qx, qy), 0)
  );
};
const box3 = (x, y, z, hx, hy, hz) => {
  const qx = Math.abs(x) - hx,
    qy = Math.abs(y) - hy,
    qz = Math.abs(z) - hz;
  return (
    Math.hypot(Math.max(qx, 0), Math.max(qy, 0), Math.max(qz, 0)) +
    Math.min(Math.max(qx, Math.max(qy, qz)), 0)
  );
};

const LEAF_FNS = {
  1: (x, y, z, p) => boxDE(x, y, z, p[0]),
  2: (x, y, z, p) => sphereDE(x, y, z, p[0]),
  3: (x, y, z, p) => {
    // Boxy variants (D2 torus widening) — params 2/3 default 0 = classic torus.
    const lr =
      p[2] > 0.5 ? Math.max(Math.abs(x), Math.abs(z)) : Math.hypot(x, z);
    const qx = lr - p[0];
    const lq =
      p[3] > 0.5 ? Math.max(Math.abs(qx), Math.abs(y)) : Math.hypot(qx, y);
    return lq - p[1];
  },
  4: (x, y, z, p) => cylinderDE(x, y, z, p[0], p[1]),
  5: (x, y, z, p) => capsuleDE(x, y, z, p[0], p[1]),
  6: (x, y, z, p) => {
    // Slab: Width/Depth (p[1]/p[2], #353) optionally clip one horizontal
    // axis at a time — 0 (the leaf default) keeps the old infinite plane.
    let d = planeDE(x, y, z, p[0]);
    if (p[1] > 0) d = Math.max(d, Math.abs(x) - p[1]);
    if (p[2] > 0) d = Math.max(d, Math.abs(z) - p[2]);
    return d;
  },
  // ── D2 batch 1 (leaves.js is the WGSL/GLSL source; these mirror it) ────────
  7: (x, y, z, p) => {
    // gyroid: |g−c|/(f√3) − t, sphere-bounded.
    const f = p[0];
    const g =
      Math.sin(x * f) * Math.cos(y * f) +
      Math.sin(y * f) * Math.cos(z * f) +
      Math.sin(z * f) * Math.cos(x * f);
    const d = Math.abs(g - p[2]) / (f * 1.7320508) - p[1];
    return Math.max(d, Math.hypot(x, y, z) - p[3]);
  },
  8: (x, y, z, p) => {
    // schwarz P: |cos+cos+cos − c|/(f√3) − t, sphere-bounded. +π/2-per-axis phase
    // shift centres the c=0 surface on the origin (mirrors leaves.js WGSL/GLSL, #280).
    const f = p[0],
      H = 1.5707963;
    const g = Math.cos(x * f + H) + Math.cos(y * f + H) + Math.cos(z * f + H);
    const d = Math.abs(g - p[1]) / (f * 1.7320508) - p[2];
    return Math.max(d, Math.hypot(x, y, z) - p[3]);
  },
  9: (x, y, z, p) => {
    // lidinoid (published TPMS implicit), conservative ∇ bound 3f.
    const f = p[0];
    const qx = x * f,
      qy = y * f,
      qz = z * f;
    const g =
      0.5 *
        (Math.sin(2 * qx) * Math.cos(qy) * Math.sin(qz) +
          Math.sin(2 * qy) * Math.cos(qz) * Math.sin(qx) +
          Math.sin(2 * qz) * Math.cos(qx) * Math.sin(qy)) -
      0.5 *
        (Math.cos(2 * qx) * Math.cos(2 * qy) +
          Math.cos(2 * qy) * Math.cos(2 * qz) +
          Math.cos(2 * qz) * Math.cos(2 * qx)) +
      0.15;
    const d = Math.abs(g - p[1]) / (f * 3) - p[2];
    return Math.max(d, Math.hypot(x, y, z) - p[3]);
  },
  10: (x, y, z, p) => {
    // scherk tower: Taubin |f|/(s + |∇f|) on sin z = sinh x sinh y.
    const s = p[0];
    const qx = x * s,
      qy = y * s,
      qz = z * s;
    const f = Math.sin(qz) - Math.sinh(qx) * Math.sinh(qy);
    const gx = Math.cosh(qx) * Math.sinh(qy);
    const gy = Math.sinh(qx) * Math.cosh(qy);
    const gr = s * (1 + Math.sqrt(gx * gx + gy * gy + Math.cos(qz) ** 2));
    return Math.max(Math.abs(f) / gr - p[1], Math.hypot(x, y, z) - p[2]);
  },
  // #353 — exact (corner-correct) hexagon SDF; the old max-of-two-half-planes
  // shortcut underestimated distance near the 6 corners (see leaves.js).
  11: (x, y, z, p) => {
    // hex grid: dual-offset honeycomb fold + hexagon ring, z-extruded.
    const csx = p[0] * 1.5,
      csy = p[0] * 1.7320508;
    const ax = (fract(x / csx) - 0.5) * csx,
      ay = (fract(y / csy) - 0.5) * csy;
    const bx = (fract(x / csx + 0.5) - 0.5) * csx,
      by = (fract(y / csy + 0.5) - 0.5) * csy;
    const useA = ax * ax + ay * ay < bx * bx + by * by;
    let qx = Math.abs(useA ? ax : bx),
      qy = Math.abs(useA ? ay : by);
    const r = p[0] * 0.5 * p[3];
    const kx = -0.8660254,
      ky = 0.5;
    const m = Math.min(kx * qx + ky * qy, 0);
    qx -= 2 * m * kx;
    qy -= 2 * m * ky;
    const cx = Math.max(-0.5773503 * r, Math.min(0.5773503 * r, qx));
    const hd = Math.hypot(qx - cx, qy - r) * Math.sign(qy - r);
    return Math.max(Math.abs(hd) - p[2], Math.abs(z) - p[1]);
  },
  12: (x, y, z, p) => {
    // tri grid: three wall-plane families at 0°/60°/120°, z-extruded.
    const s = p[0];
    const t0 = Math.abs(fract(y / s + 0.5) - 0.5) * s;
    const d1 = x * 0.8660254 + y * 0.5;
    const t1 = Math.abs(fract(d1 / s + 0.5) - 0.5) * s;
    const d2 = -x * 0.8660254 + y * 0.5;
    const t2 = Math.abs(fract(d2 / s + 0.5) - 0.5) * s;
    return Math.max(Math.min(t0, Math.min(t1, t2)) - p[2], Math.abs(z) - p[1]);
  },
  13: (x, y, z, p) => {
    // gear: ring + angular-repeated tooth boxes (polar unroll — approx).
    const l = Math.hypot(x, z);
    const ring = Math.max(l - p[1], p[1] * 0.55 - l);
    const sector = 6.2831853 / p[0];
    const ang = Math.atan2(z, x);
    const am = (fract(ang / sector + 0.5) - 0.5) * sector;
    const tooth = box2(
      l - p[1] - p[3] * 0.5,
      am * p[1],
      p[3] * 0.5,
      sector * p[1] * 0.18,
    );
    return Math.max(Math.min(ring, tooth), Math.abs(y) - p[2]);
  },
  14: (x, y, z, p) => {
    // menger plate: IQ menger sponge on a plate-aspect box base.
    const sz = p[0];
    const qx = x / sz,
      qy = y / sz,
      qz = z / sz;
    let d = box3(qx, qy, qz, 1, 1, p[1]);
    let sc = 1;
    const it = Math.max(1, Math.min(6, Math.round(p[2])));
    for (let m = 0; m < it; m++) {
      const ax = (fract(qx * sc * 0.5) - 0.5) * 2;
      const ay = (fract(qy * sc * 0.5) - 0.5) * 2;
      const az = (fract(qz * sc * 0.5) - 0.5) * 2;
      sc *= 3;
      const rx = Math.abs(1 - 3 * Math.abs(ax));
      const ry = Math.abs(1 - 3 * Math.abs(ay));
      const rz = Math.abs(1 - 3 * Math.abs(az));
      const da = Math.max(rx, ry),
        db = Math.max(ry, rz),
        dc = Math.max(rz, rx);
      d = Math.max(d, (Math.min(da, Math.min(db, dc)) - 1) / sc);
    }
    return d * sz;
  },
  15: (x, y, z, p) => {
    // (p,q)-torus knot: polar unroll, min over the p strand crossings.
    const l = Math.hypot(x, z);
    const theta = Math.atan2(z, x);
    const np = Math.max(1, Math.min(5, p[0]));
    const rr = p[2] * 0.35;
    let d = 1e9;
    for (let k = 0; k < 5; k++) {
      if (k >= np) break;
      const ang = ((theta + 6.2831853 * k) * p[1]) / np;
      const cx = p[2] + rr * Math.cos(ang),
        cy = rr * Math.sin(ang);
      d = Math.min(d, Math.hypot(l - cx, y - cy));
    }
    return d - p[3];
  },
  16: (x, y, z, p) => {
    // voxelized base shape (sphere/torus/box select) — approx across cells.
    const res = p[2];
    const cx = (Math.floor(x / res) + 0.5) * res;
    const cy = (Math.floor(y / res) + 0.5) * res;
    const cz = (Math.floor(z / res) + 0.5) * res;
    let db;
    if (p[0] > 1.5) db = box3(cx, cy, cz, p[1], p[1], p[1]);
    else if (p[0] > 0.5) {
      db = Math.hypot(Math.hypot(cx, cz) - p[1], cy) - p[1] * 0.35;
    } else db = Math.hypot(cx, cy, cz) - p[1];
    const cube =
      box3(
        x - cx,
        y - cy,
        z - cz,
        res * 0.5 - p[3],
        res * 0.5 - p[3],
        res * 0.5 - p[3],
      ) - p[3];
    return db < 0 ? cube : Math.max(db, res * 0.25);
  },
  // ── D2 batch 2 (leaves.js is the WGSL/GLSL source; these mirror it) ────────
  17: (x, y, z, p) => {
    // helix: nearest-turn point sampling per strand.
    const theta = Math.atan2(z, x);
    const ns = Math.max(1, Math.min(6, p[1]));
    let d = 1e9;
    for (let s = 0; s < 6; s++) {
      if (s >= ns) break;
      const ph = theta + (s * 6.2831853) / ns;
      const k0 = Math.round(y / p[0] - ph / 6.2831853);
      for (let dk = -1; dk <= 1; dk++) {
        const t = ph + (k0 + dk) * 6.2831853;
        const cy = (t * p[0]) / 6.2831853;
        d = Math.min(
          d,
          Math.hypot(x - p[3] * Math.cos(t), y - cy, z - p[3] * Math.sin(t)),
        );
      }
    }
    return d - p[2];
  },
  // #353 — Width is now the tread span inward from a FIXED outer rim (see
  // leaves.js for the full rationale), and the tread height blends into the
  // next sector's over the outer 40% of each sector to kill the seam-jump
  // marching noise. Mirrors the WGSL/GLSL exactly.
  18: (x, y, z, p) => {
    const theta = Math.atan2(z, x) / 6.2831853;
    const v = theta * p[0];
    const k0 = Math.floor(v);
    const f = v - k0;
    const t = Math.min(1, Math.max(0, (f - 0.6) / 0.4));
    const wgt = t * t * (3 - 2 * t);
    const turnA = Math.round(y / p[3] - k0 / p[0]);
    const slabA = Math.abs(y - (k0 / p[0] + turnA) * p[3]) - p[1];
    const k1 = k0 + 1;
    const turnB = Math.round(y / p[3] - k1 / p[0]);
    const slabB = Math.abs(y - (k1 / p[0] + turnB) * p[3]) - p[1];
    const slab = slabA + (slabB - slabA) * wgt;
    const l = Math.hypot(x, z);
    const band = Math.max(l - 1, 1 - p[2] - l);
    const am = ((0.5 - Math.min(f, 1 - f)) / p[0]) * 6.2831853 * Math.max(l, 0.05);
    return Math.max(Math.max(slab, band), am - 3.0 / p[0]);
  },
  19: (x, y, z, p) => {
    // sphere cage: nearest parallel ring ∪ nearest meridian circle.
    const lxz = Math.hypot(x, z);
    const phi = Math.atan2(lxz, y);
    const phq = Math.max(
      0,
      Math.min(Math.PI, (Math.round((phi * p[2]) / Math.PI) * Math.PI) / p[2]),
    );
    const dpar = Math.hypot(
      lxz - p[0] * Math.sin(phq),
      y - p[0] * Math.cos(phq),
    );
    const thq =
      (Math.round((Math.atan2(z, x) * p[3]) / Math.PI) * Math.PI) / p[3];
    const u = x * Math.cos(thq) + z * Math.sin(thq);
    const v = -x * Math.sin(thq) + z * Math.cos(thq);
    const dmer = Math.hypot(Math.hypot(u, y) - p[0], v);
    return Math.min(dpar, dmer) - p[1];
  },
  20: (x, y, z, p) => {
    // slice cage: sphere shell ∩ swirled meridian slabs.
    const shell = Math.abs(Math.hypot(x, y, z) - p[0]) - p[2];
    const thq =
      (Math.round(((Math.atan2(z, x) + p[3] * y) * p[1]) / Math.PI) * Math.PI) /
        p[1] -
      p[3] * y;
    const v = -x * Math.sin(thq) + z * Math.cos(thq);
    return Math.max(shell, Math.abs(v) - p[2]);
  },
  21: (x, y, z, p) => {
    // wave surface: sine heightfield, Lipschitz-normalized.
    const r = Math.hypot(x, z);
    const s = p[2] > 1.5 ? r + Math.atan2(z, x) : p[2] > 0.5 ? r : x;
    const h = p[1] * Math.sin(p[0] * s);
    const lip = 1 / Math.sqrt(1 + p[0] * p[0] * p[1] * p[1]);
    return Math.abs(y - h) * lip - p[3];
  },
  22: (x, y, z, p) => {
    // klein bagel: figure-8 cross-section rotating twist/2 per lap.
    const theta = Math.atan2(z, x);
    const u0 = Math.hypot(x, z) - p[0];
    const a = theta * 0.5 * p[2];
    const u = u0 * Math.cos(a) + y * Math.sin(a);
    const v = -u0 * Math.sin(a) + y * Math.cos(a);
    const d8 = Math.min(
      Math.hypot(u, v - p[1] * 0.5),
      Math.hypot(u, v + p[1] * 0.5),
    );
    return Math.abs(d8 - p[1] * 0.5) - p[3];
  },
  23: (x, y, z, p) => {
    // seashell: log-spiral tube, nearest-turn sampling, growing tube radius.
    const theta = Math.atan2(z, x);
    const l = Math.max(Math.hypot(x, z), 1e-4);
    const b = p[0];
    const k0 = Math.round((Math.log(l) / b - theta) / 6.2831853);
    let d = 1e9;
    const tmax = p[3] * 6.2831853;
    for (let dk = -1; dk <= 1; dk++) {
      const t = Math.max(0, Math.min(tmax, theta + (k0 + dk) * 6.2831853));
      const rc = Math.exp(b * t);
      d = Math.min(
        d,
        Math.hypot(x - rc * Math.cos(t), y - p[2] * t, z - rc * Math.sin(t)) -
          p[1] * rc * 0.5,
      );
    }
    return d;
  },
  24: (x, y, z, p) => {
    // dini horn: exponentially tapering revolved profile, height-clamped.
    const yc = Math.max(0, Math.min(p[2], y));
    const rr = p[0] * Math.exp(-p[1] * yc);
    return Math.hypot(Math.hypot(x, z) - rr, y - yc) - p[3];
  },
  25: (x, y, z, p) => {
    // room: open corner — floor (−y) + two back walls (−z, −x), each an
    // exact box plate of half-thickness Wall (#353: a closed 6-sided shell
    // was indistinguishable from Round Box from any outside camera angle).
    const hw = Math.max(p[3], 0.001);
    const floor = box3(x, y + p[1] - hw, z, p[0], hw, p[2]);
    const back = box3(x, y, z + p[2] - hw, p[0], p[1], hw);
    const side = box3(x + p[0] - hw, y, z, hw, p[1], p[2]);
    return Math.min(floor, Math.min(back, side));
  },
  26: (x, y, z, p) => {
    // IQ sdRoundBox.
    const qx = Math.abs(x) - p[0] + p[3],
      qy = Math.abs(y) - p[1] + p[3],
      qz = Math.abs(z) - p[2] + p[3];
    return (
      Math.hypot(Math.max(qx, 0), Math.max(qy, 0), Math.max(qz, 0)) +
      Math.min(Math.max(qx, Math.max(qy, qz)), 0) -
      p[3]
    );
  },
  27: (x, y, z, p) => {
    // IQ sdBoxFrame.
    const px = Math.abs(x) - p[0],
      py = Math.abs(y) - p[0],
      pz = Math.abs(z) - p[0];
    const qx = Math.abs(px + p[1]) - p[1],
      qy = Math.abs(py + p[1]) - p[1],
      qz = Math.abs(pz + p[1]) - p[1];
    const seg = (a, b, c) =>
      Math.hypot(Math.max(a, 0), Math.max(b, 0), Math.max(c, 0)) +
      Math.min(Math.max(a, Math.max(b, c)), 0);
    return Math.min(
      seg(px, qy, qz),
      Math.min(seg(qx, py, qz), seg(qx, qy, pz)),
    );
  },

  // ── D2 Taubin wave (signed f/|∇f|; leaves.js carries sources) ─────────────
  // #353 — axes remapped so the shape faces the fixed default camera; mirrors
  // the WGSL/GLSL in leaves.js (see its comment for why).
  28: (x, y, z, p) => {
    const s = p[0],
      qx = x / s,
      qy = z / s,
      qz = -y / s;
    const A = qx * qx + 2.25 * qy * qy + qz * qz - 1;
    const z3 = qz * qz * qz;
    const f = A * A * A - qx * qx * z3 - 0.1125 * qy * qy * z3;
    const gx = 6 * qx * A * A - 2 * qx * z3;
    const gy = 13.5 * qy * A * A - 0.225 * qy * z3;
    const gz = 6 * qz * A * A - (3 * qx * qx + 0.3375 * qy * qy) * qz * qz;
    return Math.max(
      (s * f) / (1e-6 + Math.hypot(gx, gy, gz)),
      Math.hypot(x, y, z) - p[1],
    );
  },
  29: (x, y, z, p) => {
    const s = p[0],
      qx = x / s,
      qy = y / s + 0.5,
      qz = z / s;
    const u = qy * (1 - qy);
    const f = qx * qx + qz * qz - u * u * u;
    const gy = -3 * u * u * (1 - 2 * qy);
    return Math.max(
      (s * f) / (1e-6 + Math.hypot(2 * qx, gy, 2 * qz)),
      Math.hypot(x, y, z) - p[1],
    );
  },
  30: (x, y, z, p) => {
    const s = p[0],
      qx = x / s + 0.5,
      qy = y / s,
      qz = z / s;
    const f = qx * qx * qx * (qx - 1) + qy * qy + qz * qz;
    const gx = 4 * qx * qx * qx - 3 * qx * qx;
    return Math.max(
      (s * f) / (1e-6 + Math.hypot(gx, 2 * qy, 2 * qz)),
      Math.hypot(x, y, z) - p[1],
    );
  },
  31: (x, y, z, p) => {
    const s = p[0],
      qx = x / s,
      qy = y / s,
      qz = z / s;
    const f = qx * qx + qy * qy - (1 - qz) * qz * qz * qz * qz;
    const gz = -4 * qz * qz * qz + 5 * qz * qz * qz * qz;
    return Math.max(
      (s * f) / (1e-6 + Math.hypot(2 * qx, 2 * qy, gz)),
      Math.hypot(x, y, z) - p[1],
    );
  },
  32: (x, y, z, p) => {
    const s = p[0],
      qx = x / s,
      qy = y / s,
      qz = z / s;
    const f = qx * qx + qy * qy + qz * qz * qz - qz * qz;
    const gz = 3 * qz * qz - 2 * qz;
    return Math.max(
      (s * f) / (1e-6 + Math.hypot(2 * qx, 2 * qy, gz)),
      Math.hypot(x, y, z) - p[1],
    );
  },
  33: (x, y, z, p) => {
    const s = p[2],
      qx = x / s,
      qy = y / s,
      qz = z / s;
    const u = qx * qx + qz * qz;
    const f = qy * qy * (qy * qy - p[0] * p[0]) - u * (u - p[1] * p[1]);
    const gy = 4 * qy * qy * qy - 2 * p[0] * p[0] * qy;
    const gu = -2 * u + p[1] * p[1];
    return Math.max(
      (s * f) / (1e-6 + Math.hypot(2 * qx * gu, gy, 2 * qz * gu)),
      Math.hypot(x, y, z) - p[3],
    );
  },
  34: (x, y, z, p) => {
    const s = p[1],
      qx = x / s,
      qy = y / s,
      qz = z / s;
    const r2 = qy * qy + qz * qz;
    const s2 = qx * qx + r2;
    const f = s2 * s2 - p[0] * (qx * qx * qx - 3 * qx * r2);
    const gx = 4 * qx * s2 - p[0] * (3 * qx * qx - 3 * r2);
    const gr = 4 * s2 + 6 * p[0] * qx;
    return Math.max(
      (s * f) / (1e-6 + Math.hypot(gx, qy * gr, qz * gr)),
      Math.hypot(x, y, z) - p[2],
    );
  },
  35: (x, y, z, p) => {
    const s = p[2],
      qx = x / s,
      qy = y / s,
      qz = z / s;
    const x2 = qx * qx,
      y2 = qy * qy,
      z2 = qz * qz,
      c2 = p[0] * p[0];
    const a1 = x2 + y2 - c2,
      b1 = z2 - 1;
    const a2 = y2 + z2 - c2,
      b2 = x2 - 1;
    const a3 = z2 + x2 - c2,
      b3 = y2 - 1;
    const P = a1 * a1 + b1 * b1,
      Q = a2 * a2 + b2 * b2,
      R = a3 * a3 + b3 * b3;
    const f = P * Q * R - p[1];
    const gx = 4 * qx * a1 * Q * R + P * 4 * qx * b2 * R + P * Q * 4 * qx * a3;
    const gy = 4 * qy * a1 * Q * R + P * 4 * qy * a2 * R + P * Q * 4 * qy * b3;
    const gz = 4 * qz * b1 * Q * R + P * 4 * qz * a2 * R + P * Q * 4 * qz * a3;
    return Math.max(
      (s * f) / (1e-6 + Math.hypot(gx, gy, gz)),
      Math.hypot(x, y, z) - p[3],
    );
  },
  36: (x, y, z, p) => {
    const s = p[0],
      qx = x / s,
      qy = y / s,
      qz = z / s;
    const f = qx * qx + qy * qy - qx * qx * qz + qy * qy * qz + qz * qz - 1;
    const gx = 2 * qx * (1 - qz);
    const gy = 2 * qy * (1 + qz);
    const gz = -qx * qx + qy * qy + 2 * qz;
    return Math.max(
      (s * f) / (1e-6 + Math.hypot(gx, gy, gz)),
      Math.hypot(x, y, z) - p[1],
    );
  },
  37: (x, y, z, p) => {
    const s = p[0],
      qx = x / s,
      qy = y / s,
      qz = z / s;
    const u = qy * qy + qz * qz;
    const x2 = qx * qx;
    const f = 4 * (x2 * x2 + u * u) + 17 * x2 * u - 20 * (x2 + u) + 17;
    const gx = 16 * x2 * qx + 34 * qx * u - 40 * qx;
    const gc = 16 * u + 34 * x2 - 40;
    return Math.max(
      (s * f) / (1e-6 + Math.hypot(gx, qy * gc, qz * gc)),
      Math.hypot(x, y, z) - p[1],
    );
  },
  38: (x, y, z, p) => {
    const f = p[0] * x * x + p[1] * y * y + p[2] * z * z - 1;
    const g = Math.hypot(2 * p[0] * x, 2 * p[1] * y, 2 * p[2] * z);
    return Math.max(f / (1e-6 + g), Math.hypot(x, y, z) - p[3]);
  },
  // ── D2 batch 3 — heightfields (y-up; leaves.js is the WGSL/GLSL source) ────
  39: (x, y, z, p) => {
    // gnarlyField: Pickover gnarl walk, sine height off the wandered point.
    let u = x,
      v = z;
    const it = Math.max(1, Math.min(8, Math.round(p[3])));
    for (let i = 0; i < it; i++) {
      const t = u;
      u -= p[0] * Math.sin(v + Math.sin(p[1] * v));
      v -= p[0] * Math.sin(t + Math.sin(p[1] * t));
    }
    const h = p[2] * 0.5 * (Math.sin(u) + Math.sin(v));
    const lip = 1 / (1 + p[2] * (1 + p[0] * (1 + p[1]) * it));
    return Math.abs(y - h) * lip;
  },
  40: (x, y, z, p) => {
    // ducksField: Monnier ducks orbit, height = mean log-modulus.
    let wx = x,
      wy = z,
      acc = 0;
    const it = Math.max(1, Math.min(24, Math.round(p[1])));
    for (let i = 0; i < it; i++) {
      const ay = Math.abs(wy);
      const l = 0.5 * Math.log(Math.max(wx * wx + ay * ay, 1e-6));
      const a = Math.atan2(ay, wx);
      wx = l + p[2];
      wy = a + p[3];
      acc += l;
    }
    const h = (p[0] * acc) / it;
    return (Math.abs(y - h) * 1) / (1 + 2 * p[0]);
  },
  41: (x, y, z, p) => {
    // mandelPlate: interior plateau + smooth-count skirt + Milnor rim guard.
    const zoom = Math.max(p[3], 0.1);
    const cre = x / zoom,
      cim = z / zoom;
    let zr = 0,
      zi = 0,
      dr = 1,
      di = 0,
      r2 = 0;
    let esc = false;
    const n = Math.max(4, Math.min(96, Math.round(p[0])));
    for (let i = 0; i < n; i++) {
      const ndr = 2 * (zr * dr - zi * di) + 1;
      const ndi = 2 * (zr * di + zi * dr);
      dr = ndr;
      di = ndi;
      const nzr = zr * zr - zi * zi + cre;
      const nzi = 2 * zr * zi + cim;
      zr = nzr;
      zi = nzi;
      r2 = zr * zr + zi * zi;
      if (r2 > 256) {
        esc = true;
        break;
      }
    }
    let dM = 0;
    if (esc) {
      const az = Math.sqrt(Math.max(r2, 1e-30));
      const adz = Math.sqrt(Math.max(dr * dr + di * di, 1e-30));
      dM = (az * Math.log(az)) / adz / zoom;
    }
    const skirt = 1 / Math.max(p[1], 0.1);
    const h = p[2] * Math.min(1, Math.max(0, 1 - dM / skirt));
    const de = (y - h) / Math.sqrt(1 + p[2] * p[2] * p[1] * p[1]);
    return Math.max(de, dM - skirt);
  },
  42: (x, y, z, p) => {
    // checkerField: smooth parity checker raised by Bump.
    const cw =
      Math.sin((3.14159265 * x) / p[1]) * Math.sin((3.14159265 * z) / p[1]);
    const t = Math.min(1, Math.max(0, (cw + p[2]) / (2 * p[2])));
    const h = p[0] * t * t * (3 - 2 * t);
    const lip = 1 / (1 + (p[0] * 3.2) / (Math.max(p[2], 0.02) * p[1]));
    return (y - h) * lip;
  },
  43: (x, y, z, p) => {
    // riemannSqrt: √r·|cos(sheets·θ/2 + swirl·√r)| height, two sheets.
    const r = Math.hypot(x, z);
    const sr = Math.sqrt(Math.max(r, 1e-4));
    const th = Math.atan2(z, x);
    const h = p[1] * sr * Math.abs(Math.cos(0.5 * p[0] * th + p[2] * sr));
    const d = Math.min(Math.abs(y - h), Math.abs(y + h));
    const lip =
      1 /
      (1 + p[1] * ((0.5 * p[0]) / Math.max(sr, 0.3) + Math.abs(p[2]) + 0.6));
    return Math.max(d * lip, Math.hypot(x, y, z) - p[3]);
  },
  // ── D2 batch 4 — geometric tail (leaves.js is the WGSL/GLSL source) ───────
  44: (x, y, z, p) => {
    // octahedron: IQ exact SDF.
    const q = [Math.abs(x), Math.abs(y), Math.abs(z)];
    const m = q[0] + q[1] + q[2] - p[0];
    let o;
    if (3 * q[0] < m) o = [q[0], q[1], q[2]];
    else if (3 * q[1] < m) o = [q[1], q[2], q[0]];
    else if (3 * q[2] < m) o = [q[2], q[0], q[1]];
    else return m * 0.57735027 - p[1];
    const k = Math.min(Math.max(0.5 * (o[2] - o[1] + p[0]), 0), p[0]);
    return Math.hypot(o[0], o[1] - p[0] + k, o[2] - k) - p[1];
  },
  45: (x, y, z, p) => {
    // dodecahedron / icosahedron: GDF plane fold, chained smaxP over face
    // normals (#353: `d - size - round` used to be identical to `d -
    // (size+round)` — Round was 100% redundant with Size, never an actual
    // edge round). smaxP falls back to Math.max at k<=0, so Round=0 stays
    // bit-identical to every existing preset; only ties (edges/vertices)
    // bend, flat faces don't move.
    const qx = Math.abs(x),
      qy = Math.abs(y),
      qz = Math.abs(z);
    const k = p[1];
    let d;
    if (p[2] > 0.5) {
      d = (qx + qy + qz) * 0.57735027;
      d = smaxP(d, 0.35682209 * qy + 0.93417236 * qz, k);
      d = smaxP(d, 0.35682209 * qx + 0.93417236 * qy, k);
      d = smaxP(d, 0.93417236 * qx + 0.35682209 * qz, k);
    } else {
      d = 0.52573111 * qy + 0.85065081 * qz;
      d = smaxP(d, 0.52573111 * qx + 0.85065081 * qy, k);
      d = smaxP(d, 0.85065081 * qx + 0.52573111 * qz, k);
    }
    return d - p[0];
  },
  46: (x, y, z, p) => {
    // nPrism: polar sector fold + apothem + slab, smaxP'd (#353: Round used
    // to just resize the prism — "changing radius" — instead of chamfering
    // the cap↔side rim; k<=0 is the exact old hard max).
    const n = Math.min(16, Math.max(3, p[0]));
    const sector = 6.2831853 / n;
    const u = Math.atan2(z, x) / sector + 0.5;
    const am = (u - Math.floor(u) - 0.5) * sector;
    const d2 = Math.hypot(x, z) * Math.cos(am) - p[1];
    return smaxP(d2, Math.abs(y) - p[2], p[3]);
  },
  47: (x, y, z, p) => {
    // pyramid: polar fold + side plane through base edge and apex, smaxP'd
    // against the base plane (#353: same redundant-with-size Round bug as
    // N-Prism/Dodecahedron — "Round value is just changing radius").
    const n = Math.min(16, Math.max(3, p[0]));
    const sector = 6.2831853 / n;
    const u = Math.atan2(z, x) / sector + 0.5;
    const am = (u - Math.floor(u) - 0.5) * sector;
    const l = Math.hypot(x, z) * Math.cos(am);
    const r = Math.max(p[1], 0.05);
    const h = Math.max(p[2], 0.05);
    const yb = y + 0.5 * h;
    const side = (l * h + yb * r - r * h) / Math.hypot(h, r);
    return smaxP(side, -yb, p[3]);
  },
  48: (x, y, z, p) => {
    // greekCross: union of three orthogonal exact boxes.
    const box = (hx, hy, hz) => {
      const bx = Math.abs(x) - hx,
        by = Math.abs(y) - hy,
        bz = Math.abs(z) - hz;
      return (
        Math.hypot(Math.max(bx, 0), Math.max(by, 0), Math.max(bz, 0)) +
        Math.min(Math.max(bx, Math.max(by, bz)), 0)
      );
    };
    return (
      Math.min(
        box(p[0], p[1], p[1]),
        box(p[1], p[0], p[1]),
        box(p[1], p[1], p[0]),
      ) - p[2]
    );
  },
  49: (x, y, z, p) => {
    // borg: box shell displaced by a separable sin product, lip-guarded.
    const bx = Math.abs(x) - p[0],
      by = Math.abs(y) - p[0],
      bz = Math.abs(z) - p[0];
    const box =
      Math.hypot(Math.max(bx, 0), Math.max(by, 0), Math.max(bz, 0)) +
      Math.min(Math.max(bx, Math.max(by, bz)), 0);
    const disp =
      p[2] * Math.sin(p[1] * x) * Math.sin(p[1] * y) * Math.sin(p[1] * z);
    return (Math.abs(box) - p[3] + disp) / (1 + p[2] * p[1] * 1.8);
  },
  50: (x, y, z, p) => {
    // tower: sin-breathing fluted column along y.
    const l = Math.hypot(x, z);
    const ang = Math.atan2(z, x);
    const fl = (p[3] > 0.5 ? 1 : 0) * 0.06 * p[0] * Math.cos(p[3] * ang);
    const rr = p[0] * (1 + p[1] * Math.sin(p[2] * y)) + fl;
    const gy = p[0] * p[1] * p[2];
    const ga = 0.06 * p[3];
    return (l - rr) / Math.sqrt(1 + gy * gy + ga * ga);
  },
  51: (x, y, z, p) => {
    // gem: facet fold + pavilion/crown planes + table.
    const n = Math.min(24, Math.max(4, p[0]));
    const sector = 6.2831853 / n;
    const u = Math.atan2(z, x) / sector + 0.5;
    const am = (u - Math.floor(u) - 0.5) * sector;
    const l = Math.hypot(x, z) * Math.cos(am);
    const R = Math.max(p[1], 0.05);
    const hc = Math.max(p[2], 0.02);
    const hp = Math.max(p[3], 0.05);
    const pav = (l * hp - y * R - R * hp) / Math.hypot(hp, R);
    const cro = (l * hc + y * 0.5 * R - R * hc) / Math.hypot(hc, 0.5 * R);
    return Math.max(Math.max(pav, cro), y - hc);
  },
  52: (x, y, z, p) => {
    // loxodrome: Mercator straight-line tube on the sphere shell.
    const r = Math.hypot(x, y, z);
    const ir = Math.max(r, 1e-4);
    const sl = Math.min(0.9999, Math.max(-0.9999, y / ir));
    const psi = 0.5 * Math.log((1 + sl) / (1 - sl));
    const sector = 6.2831853 / Math.max(p[2], 1);
    const uu = (Math.atan2(z, x) - p[0] * psi) / sector + 0.5;
    const w = (uu - Math.floor(uu) - 0.5) * sector;
    const cl = Math.sqrt(Math.max(1 - sl * sl, 1e-4));
    const arc = (w * ir * cl) / Math.sqrt(1 + p[0] * p[0]);
    return (Math.hypot(r - p[1], arc) - p[3]) * 0.8;
  },
  53: (x, y, z, p) => {
    // logSpiral: log-polar wall gap, extruded in y.
    const r = Math.max(Math.hypot(x, z), 1e-4);
    const b = Math.max(p[0], 0.02);
    const sector = 6.2831853 / Math.max(p[1], 1);
    const uu = (Math.log(r) / b - Math.atan2(z, x)) / sector + 0.5;
    const w = (uu - Math.floor(uu) - 0.5) * sector;
    const d2 = (Math.abs(w) * r * b) / Math.sqrt(1 + b * b) - p[3];
    return Math.max(d2, Math.abs(y) - p[2]) * 0.9;
  },
  54: (x, y, z, p) => {
    // pseudoSphere: sampled min distance to the tractrix, revolved.
    const l = Math.hypot(x, z);
    const ay = Math.abs(y);
    let best = 1e9;
    for (let i = 0; i < 24; i++) {
      const u = p[1] * (i / 23);
      const e = Math.exp(u);
      const sh = 2 / (e + 1 / e);
      const tn = (e - 1 / e) / (e + 1 / e);
      best = Math.min(best, Math.hypot(l - p[0] * sh, ay - p[0] * (u - tn)));
    }
    return (best - p[2]) * 0.7;
  },
  // ── D2 batch 5 ─────────────────────────────────────────────────────────────
  55: (x, y, z, p) => {
    // randomCells: integer-hash grid dispatch — Math.imul keeps the 32-bit
    // wrap bit-identical to the u32 math on the GPU tiers.
    const c = Math.max(p[0], 0.05);
    const seed = Math.min(1023, Math.max(0, Math.round(p[2])));
    const slab = Math.min(9, Math.max(1, Math.round(p[3])));
    const bx = Math.floor(x / c),
      by = Math.floor(y / c),
      bz = Math.floor(z / c);
    let d = 0.5 * c;
    for (let oi = -1; oi <= 1; oi++)
      for (let oj = -1; oj <= 1; oj++)
        for (let ok = -1; ok <= 1; ok++) {
          const ix = bx + oi,
            iy = by + oj,
            iz = bz + ok;
          if (Math.abs(iy) >= slab) continue;
          let h =
            (Math.imul(ix, 73856093) ^
              Math.imul(iy, 19349663) ^
              Math.imul(iz, 83492791) ^
              Math.imul(seed, 2654435761)) >>>
            0;
          h = (h ^ (h >>> 13)) >>> 0;
          h = Math.imul(h, 1274126177) >>> 0;
          h = (h ^ (h >>> 16)) >>> 0;
          if ((h & 1023) >= p[1] * 1024) continue;
          const r = c * (0.16 + (0.22 * ((h >>> 10) & 255)) / 255);
          const lx = x - (ix + 0.5) * c,
            ly = y - (iy + 0.5) * c,
            lz = z - (iz + 0.5) * c;
          const shape = (h >>> 18) & 3;
          let sd;
          if (shape === 0) sd = Math.hypot(lx, ly, lz) - r;
          else if (shape === 1) {
            const ax = Math.abs(lx) - r,
              ay = Math.abs(ly) - r,
              az = Math.abs(lz) - r;
            sd =
              Math.hypot(Math.max(ax, 0), Math.max(ay, 0), Math.max(az, 0)) +
              Math.min(Math.max(ax, Math.max(ay, az)), 0);
          } else if (shape === 2)
            sd =
              (Math.abs(lx) + Math.abs(ly) + Math.abs(lz) - 1.5 * r) *
              0.57735027;
          else sd = Math.max(Math.hypot(lx, lz) - r, Math.abs(ly) - r);
          d = Math.min(d, sd);
        }
    return d;
  },
  56: (x, y, z, p) => {
    // umbrella: Whitney umbrella x² = z²·y, Taubin quotient.
    const qx = x / p[0],
      qy = y / p[0],
      qz = z / p[0];
    const f = qx * qx - qz * qz * qy;
    const g = Math.hypot(2 * qx, qz * qz, 2 * qz * qy);
    let d = (p[0] * f) / (1e-6 + g);
    if (p[1] > 0) d = Math.abs(d) - p[1];
    return Math.max(d, Math.hypot(x, y, z) - p[2]);
  },
  57: (x, y, z, p) => {
    // kleinBottle: the published sextic implicit, Taubin quotient.
    const qx = x / p[0],
      qy = y / p[0],
      qz = z / p[0];
    const S = qx * qx + qy * qy + qz * qz;
    const A = S + 2 * qy - 1;
    const B = S - 2 * qy - 1;
    const C = B * B - 8 * qz * qz;
    const f = A * C + 16 * qx * qz * B;
    const gx = 2 * qx * C + 4 * A * B * qx + 16 * qz * B + 32 * qx * qx * qz;
    const gy =
      (2 * qy + 2) * C + 4 * A * B * (qy - 1) + 32 * qx * qz * (qy - 1);
    const gz =
      2 * qz * C + A * (4 * B * qz - 16 * qz) + 16 * qx * B + 32 * qx * qz * qz;
    let d = (p[0] * f) / (1e-6 + Math.hypot(gx, gy, gz));
    if (p[1] > 0) d = Math.abs(d) - p[1];
    return Math.max(d, Math.hypot(x, y, z) - p[2]);
  },
  58: (x, y, z, p) => {
    // kleinianLimit: Jos Leys' published Maskit algorithm + Poincaré
    // extension + conformal-pullback DE (docs/planning/KLEINIAN_LIMIT.md).
    const u = Math.min(2.2, Math.max(1.4, p[0]));
    const v = Math.min(0.5, Math.max(-0.5, p[1]));
    const bend = Math.min(0.8, Math.max(0, p[2]));
    const it = Math.min(128, Math.max(4, Math.round(p[3])));
    let qx = x,
      qy = y + 0.5 * u,
      qz = z,
      df = 1;
    for (let i = 0; i < it; i++) {
      if (qy < 0 || qy > u || df > 1e30) break;
      const s = (v * qy) / u;
      const w = (qx + 1 + s) / 2;
      qx = (w - Math.floor(w)) * 2 - 1 - s;
      const xx = qx + 0.5 * v;
      const sep =
        0.5 * u + Math.sign(xx) * bend * u * (1 - Math.exp(-3 * Math.abs(xx)));
      if (qy < sep) {
        const r2 = Math.max(qx * qx + qy * qy + qz * qz, 1e-12);
        const nx = -v + qx / r2;
        qy = u - qy / r2;
        qx = nx;
        qz = qz / r2;
        df /= r2;
      } else {
        const dx = qx + v;
        const dy = qy - u;
        const r2 = Math.max(dx * dx + dy * dy + qz * qz, 1e-12);
        qx = dx / r2;
        qy = -dy / r2;
        qz = qz / r2;
        df /= r2;
      }
    }
    return Math.abs(Math.min(qy, u - qy)) / df - 0.025;
  },
};

// CSG D0 — scene DE: combine over objects. Mirrors shader.js mapDE() with
// G.scene.x > 0: d = +INF; for each object k: pk = qrot(conj(q), p−origin)/uscale;
// dk = objIterDE(pk)·uscale; d = combine(d, dk). ONE unified per-object path
// (op chain + shape leaf — PRIMITIVE_DIFS_D0 §2.1): pure leaves arrive from
// normalizeSceneObject as a 1-iteration empty loop, so there is no separate
// primitive branch; iterShape (D3) samples the leaf after each full iteration
// (ops + addC, before the bail check) and keeps the min.
// The scene marcher bailout is 1e6 (preview.js scene path).
function makeSceneDE(formula) {
  const SCENE_BAIL = BAILOUT_IFS; // same value preview.js sends the scene path (G.prm.x)
  // Canonical per-object metadata (fallback chains, active-op slice, quat) —
  // shared with renderer.js/renderer_gl.js via sceneobj.js normalizeSceneObject;
  // 3-emitter mirror discipline, guarded by core/scenemute.test.mjs.
  const built = formula.objects.map((raw) => {
    const o = normalizeSceneObject(raw);
    const { origin, uscale, combine, blendK } = o;
    const q = o.quat;
    const qinv = [-q[0], -q[1], -q[2], q[3]]; // conjugate: world → local rotate
    const ops = o.ops.map((op) => ({
      key: op.key,
      v: op.values || [],
    }));
    const iters = o.iters;
    const escape = o.deOption === 0; // deOption 0 = escape-time, else IFS
    const julia = o.julia;
    const addC = o.addC || julia; // flags bit0 || julia (objIterDE)
    const jc = julia ? o.juliaC : null;
    // Unknown leaf id (a D2 link opened in an older build) → radial fallback,
    // mirroring WGSL leafDist()'s default (degrade, don't vanish).
    const leaf =
      o.shapeId > 0
        ? (LEAF_FNS[o.shapeId] ?? ((x, y, z) => Math.hypot(x, y, z)))
        : null;
    const prm = o.shapeParams;
    const iterShape = o.iterShape && leaf;
    const child = (px, py, pz) => {
      const s = { x: px, y: py, z: pz, w: 1.0 };
      const cx = jc ? jc[0] : px,
        cy = jc ? jc[1] : py,
        cz = jc ? jc[2] : pz;
      let dmin = 1.0e9;
      for (let i = 0; i < iters; i++) {
        s.i = i; // iteration index for scaleDrift (SCALE_VARY.md §6.5)
        for (let k = 0; k < ops.length; k++) applyOp(ops[k].key, ops[k].v, s);
        if (addC) {
          s.x += cx;
          s.y += cy;
          s.z += cz;
        }
        if (iterShape)
          dmin = Math.min(
            dmin,
            leaf(s.x, s.y, s.z, prm) / Math.max(Math.abs(s.w), 1e-9),
          );
        if (s.x * s.x + s.y * s.y + s.z * s.z > SCENE_BAIL) break;
      }
      const r = Math.hypot(s.x, s.y, s.z),
        aw = Math.max(Math.abs(s.w), 1e-9);
      if (escape) return (0.5 * Math.log(Math.max(r, 1e-9)) * r) / aw;
      if (leaf) return iterShape ? dmin : leaf(s.x, s.y, s.z, prm) / aw;
      return r / aw;
    };
    return { origin, uscale, qinv, combine, blendK, child };
  });
  return function de(px, py, pz) {
    let d = 1.0e9;
    let k = -1;
    for (const ob of built) {
      k++;
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
      // Object 0 is the BASE (combine forced to union) — mirrors shader.js:
      // a first-object carve against the empty accumulator blanked the scene.
      const combine = k === 0 ? 0 : ob.combine;
      if (combine === 1) d = sminP(d, dk, ob.blendK);
      else if (combine === 2)
        d = smaxP(d, -dk, ob.blendK); // subtract (blendK rounds the cut)
      else if (combine === 3)
        d = smaxP(d, dk, ob.blendK); // intersect (blendK rounds the seam)
      else d = Math.min(d, dk);
    }
    return d;
  };
}

// COLORING P3 S5 — per-object scene orbit signal (CPU/ASCII mirror of shader.js
// sceneOrbit). Finds the owning object (same combine walk as makeSceneDE, but
// tracking the winner), transforms the hit into its local space, and runs its
// orbit for glow(1)/bands(2)/silk(3). Real orbit coloring on scenes, no stand-in.
function buildSceneObjs(formula, mode, stripeFreq = 5) {
  const SCENE_BAIL = BAILOUT_IFS;
  const sk = Math.max(1, Math.min(16, stripeFreq ?? 5));
  return formula.objects.map((raw) => {
    const o = normalizeSceneObject(raw);
    const { origin, uscale, combine, blendK } = o;
    const q = o.quat;
    const qinv = [-q[0], -q[1], -q[2], q[3]];
    const ops = o.ops.map((op) => ({ key: op.key, v: op.values || [] }));
    const iters = o.iters;
    const escape = o.deOption === 0;
    const julia = o.julia;
    const addC = o.addC || julia;
    const jc = julia ? o.juliaC : null;
    const leaf =
      o.shapeId > 0
        ? (LEAF_FNS[o.shapeId] ?? ((x, y, z) => Math.hypot(x, y, z)))
        : null;
    const prm = o.shapeParams;
    const iterShape = o.iterShape && leaf;
    // Distance (for the combine walk — matches makeSceneDE's child).
    const dist = (px, py, pz) => {
      const s = { x: px, y: py, z: pz, w: 1.0 };
      const cx = jc ? jc[0] : px,
        cy = jc ? jc[1] : py,
        cz = jc ? jc[2] : pz;
      let dmin = 1.0e9;
      for (let i = 0; i < iters; i++) {
        s.i = i;
        for (let k = 0; k < ops.length; k++) applyOp(ops[k].key, ops[k].v, s);
        if (addC) {
          s.x += cx;
          s.y += cy;
          s.z += cz;
        }
        if (iterShape)
          dmin = Math.min(
            dmin,
            leaf(s.x, s.y, s.z, prm) / Math.max(Math.abs(s.w), 1e-9),
          );
        if (s.x * s.x + s.y * s.y + s.z * s.z > SCENE_BAIL) break;
      }
      const r = Math.hypot(s.x, s.y, s.z),
        aw = Math.max(Math.abs(s.w), 1e-9);
      if (escape) return (0.5 * Math.log(Math.max(r, 1e-9)) * r) / aw;
      if (leaf) return iterShape ? dmin : leaf(s.x, s.y, s.z, prm) / aw;
      return r / aw;
    };
    // Orbit signal (glow/bands/silk) for this object's own iteration.
    const orbit = (px, py, pz) => {
      const s = { x: px, y: py, z: pz, w: 1.0 };
      const cx = jc ? jc[0] : px,
        cy = jc ? jc[1] : py,
        cz = jc ? jc[2] : pz;
      let tr = 1e9,
        esc = iters,
        rEsc2 = 0,
        acc = 0,
        cnt = 0;
      for (let i = 0; i < iters; i++) {
        s.i = i;
        for (let k = 0; k < ops.length; k++) applyOp(ops[k].key, ops[k].v, s);
        if (addC) {
          s.x += cx;
          s.y += cy;
          s.z += cz;
        }
        const r2 = s.x * s.x + s.y * s.y + s.z * s.z;
        tr = Math.min(tr, Math.sqrt(r2));
        acc += 0.5 + 0.5 * Math.sin(sk * Math.atan2(s.y, s.x));
        cnt += 1;
        if (r2 > SCENE_BAIL) {
          esc = i;
          rEsc2 = r2;
          break;
        }
      }
      if (mode === 7) {
        // COLORING R S8 — sign-octant of the winning object's final orbit point.
        const oct = (s.x > 0 ? 1 : 0) + (s.y > 0 ? 2 : 0) + (s.z > 0 ? 4 : 0);
        return (oct + 0.5) / 8;
      }
      if (mode === 3) return acc / Math.max(cnt, 1);
      if (mode === 2)
        return (
          (esc + smoothEscFrac(Math.sqrt(rEsc2), SCENE_BAIL)) /
          Math.max(iters, 1)
        );
      return Math.min(tr / 1.5, 1); // glow
    };
    // color: the object's own tint (matches the GPU objects buffer, renderer.js:948);
    // used by makeSceneTint for scene surface color.
    const color = o.color ?? raw.color ?? [0.8, 0.8, 0.8];
    return { origin, uscale, qinv, combine, blendK, dist, orbit, color };
  });
}

// The winning object at a world point — the SAME combine walk makeSceneDE and
// the GPU sceneTint (shader.js:698) use (union=min, 1=smin, 2=subtract,
// 3=intersect). Returns the built-object that owns the surface there.
function sceneWinner(built, px, py, pz) {
  let d = 1.0e9,
    win = built[0],
    k = -1;
  for (const ob of built) {
    k++;
    const r = qrot(
      ob.qinv,
      px - ob.origin[0],
      py - ob.origin[1],
      pz - ob.origin[2],
    );
    const inv = 1 / ob.uscale;
    const dk = ob.dist(r[0] * inv, r[1] * inv, r[2] * inv) * ob.uscale;
    // Object 0 is the BASE (combine forced to union) — mirrors makeSceneDE.
    const combine = k === 0 ? 0 : ob.combine;
    if (combine === 2) {
      if (-dk > d) win = ob;
      d = smaxP(d, -dk, ob.blendK);
    } else if (combine === 3) {
      if (dk > d) win = ob;
      d = smaxP(d, dk, ob.blendK);
    } else {
      if (dk < d) win = ob;
      d = combine === 1 ? sminP(d, dk, ob.blendK) : Math.min(d, dk);
    }
  }
  return win;
}

// COLORING P3 S5 — the winning object's OWN orbit signal at a world point.
function makeSceneMeasure(formula, mode, stripeFreq = 5) {
  const built = buildSceneObjs(formula, mode, stripeFreq);
  return function measure(px, py, pz) {
    const win = sceneWinner(built, px, py, pz);
    const r = qrot(
      win.qinv,
      px - win.origin[0],
      py - win.origin[1],
      pz - win.origin[2],
    );
    const inv = 1 / win.uscale;
    return win.orbit(r[0] * inv, r[1] * inv, r[2] * inv);
  };
}

// Scene SURFACE color (mode 0): the winning object's own color, matching the GPU
// sceneTint (shader.js:698) — NOT the normal-tint palette fallback. Without this
// the CPU splat export dropped per-object scene colors (monochromatic exports).
function makeSceneTint(formula) {
  const built = buildSceneObjs(formula, 0);
  return function tint(px, py, pz) {
    return sceneWinner(built, px, py, pz).color;
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
    // #441 ORTHOGRAPHIC half-height; 0 = perspective. Mirrors the WGSL
    // (camFwd.w) and GL (uOrthoH) tiers — three hand-kept copies, none of which
    // CI compares against the others.
    orthoH = 0,
    // TILED_EXPORT §2.1.3 — the off-axis sub-projection window
    // { sx, sy, bx, by } from tilegrid.tileWindow(), mirroring the WGSL
    // G.tile. null (default) = the whole frame, and the ray-gen expression
    // below is then the pre-tiling one, unchanged arithmetic included.
    //
    // This mirror is cheap AND it is what makes the seam gate automatable
    // without a GPU (tileseam.test.mjs): WGSL is compiled nowhere in CI, so the
    // CPU tier is the only place the window's algebra can be executed by a test.
    //
    // ONE CALLER REQUIREMENT: pass the SQUARE-PIXEL aspect (cols/rows), not this
    // function's character-cell default cols/(2*rows). sy/by are
    // aspect-independent, but bx is expressed in units of `ndc.x * aspect` —
    // under the cell convention it would need halving. The GPU tiers always use
    // res.x/res.y, so this is a CPU-tier-only footgun and the test passes the
    // aspect explicitly on both sides of the comparison.
    tile = null,
  } = opts;
  // A LOOSE analytic IFS DE (scale < 2, see stability.looseDE) over-estimates
  // distance, so the marcher oversteps thin/far surfaces and they vanish — the
  // hollow-back artifact. Mirror the GPU's loose treatment (preview.js: deScale
  // 0.3 + more steps). Explicit opts still win, so cpu.test #14's overstep/resolve
  // cases (which pass deScale/maxSteps) are intact, and non-loose presets keep the
  // fast 0.85/110 defaults (byte-identical).
  const loose = looseDE(formula);
  // Approximate-DE (APPROX_DE.md §3): the ASCII/CPU tier is called DIRECTLY
  // by ascii.ts/cast.ts (renderpolicy's qualityParams is NOT in this chain),
  // so the deApprox tightening must live in this default too — halve the
  // step, double the budget, exactly like the GPU tiers. Explicit opts win.
  const approx = isApproxDE(formula);
  // Unbounded scenes mirror the GPU policy: bigger budget + further far cut.
  const unb = unboundedScene(formula);
  const maxSteps =
    opts.maxSteps ??
    Math.max(
      (loose ? 260 : 110) * (approx ? 2 : 1) * (unb ? STEPS_UNBOUNDED_MUL : 1),
      unb ? 512 : 0,
    );
  const deScale = opts.deScale ?? (loose ? 0.3 : 0.85) * (approx ? 0.5 : 1);
  // Deep zoom (§5) — near/far scale off cam.dist instead of the old fixed
  // [0.02, 80] (mirrors shader.js/shader_gl.js); the shared constants live in
  // renderpolicy.js (REF_DIST=24 keeps every existing formula byte-identical —
  // cam.dist defaults to 24 in makeCamera's typical framing). Explicit opts
  // still win.
  const tNear = opts.tNear ?? cam.dist * TNEAR_K;
  const tFar =
    opts.tFar ??
    Math.max(cam.dist * TFAR_K, TFAR_MIN) * (unb ? TFAR_UNBOUNDED_MUL : 1);
  // Auto-detail (#181): opts.iters is the render policy's effective iteration
  // count (zoom boost + Detail slider); undefined ⇒ formula.iters, unchanged.
  const de = makeDE(formula, opts.iters);
  const { eye, fwd, right, up } = cam.basis();
  const tanF = Math.tan(0.5 * cam.fov);
  // Match the on-screen pixel aspect when the caller passes it (so the ASCII
  // framing lines up with the GPU render); else assume cells are 2× tall.
  const aspect = aspectOpt ?? cols / (2 * rows);
  const n = cols * rows;
  // Optional caller-owned G-buffer scratch (opts.scratch = {}): interactive
  // callers (the app's ASCII view) redraw every frame, and 13 fresh
  // Float32Array planes per frame is ~1 MB of garbage at high density.
  // Reused planes are re-filled to the fresh-allocation state, so results
  // stay byte-identical; callers that omit scratch allocate exactly as before.
  const S = opts.scratch;
  const plane = (k, fill = 0) => {
    let a = S?.[k];
    const fresh = !a || a.length !== n;
    if (fresh) {
      a = new Float32Array(n);
      if (S) S[k] = a;
    }
    if (fill || !fresh) a.fill(fill);
    return a;
  };
  const cov = plane("cov"); // 0..1 fraction of subrays that hit
  const inten = plane("inten"); // lit intensity, misses count as 0 (AA tone)
  const nx = plane("nx"),
    ny = plane("ny"),
    nz = plane("nz");
  const depth = plane("depth", Infinity);
  const rdx = plane("rdx"),
    rdy = plane("rdy"),
    rdz = plane("rdz");
  const ao = plane("ao");
  const hx = plane("hx"),
    hy = plane("hy"),
    hz = plane("hz");
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
          // TILED_EXPORT §2.1.3 — plane coordinate with the off-axis window.
          // The untiled branch is the pre-tiling expression, so the ASCII tier's
          // output is unchanged to the last bit (no ×1 + 0 rewrite in the hot
          // loop, deliberately: a `-0` would survive the multiply but not the add).
          let wx, wy;
          if (tile) {
            wx = ndcX * aspect * tile.sx + tile.bx;
            wy = ndcY * tile.sy + tile.by;
          } else {
            wx = ndcX * aspect;
            wy = ndcY;
          }
          // Perspective fans directions from one origin; orthographic spreads
          // the ORIGIN and holds the direction (#441).
          let dx, dy, dz;
          let ox = eye[0],
            oy = eye[1],
            oz = eye[2];
          if (orthoH > 0) {
            dx = fwd[0];
            dy = fwd[1];
            dz = fwd[2];
            const sxw = wx * orthoH,
              syw = wy * orthoH;
            ox += right[0] * sxw + up[0] * syw;
            oy += right[1] * sxw + up[1] * syw;
            oz += right[2] * sxw + up[2] * syw;
          } else {
            dx = fwd[0] + wx * tanF * right[0] + wy * tanF * up[0];
            dy = fwd[1] + wx * tanF * right[1] + wy * tanF * up[1];
            dz = fwd[2] + wx * tanF * right[2] + wy * tanF * up[2];
          }
          const dl = Math.hypot(dx, dy, dz) || 1;
          dx /= dl;
          dy /= dl;
          dz /= dl;
          let t = tNear,
            tPrev = t,
            hit = false,
            st = 0,
            lastD = 1e9;
          for (; st < maxSteps; st++) {
            const d = de(ox + dx * t, oy + dy * t, oz + dz * t) * deScale;
            lastD = d;
            if (d < eps * t) {
              hit = true;
              break;
            }
            tPrev = t;
            t += d;
            if (t > tFar) break;
          }
          // Hit refinement (mirrors shader.js/shader_gl.js): the loop lands up
          // to a full march step past the eps·t crossing; the per-ray overshoot
          // terraces grazing silhouettes. Bisect [tPrev, t] 8× (÷256, sub-pixel
          // at any zoom) — skipped when the eye starts inside the shell
          // (tPrev===t) and never applied to the exhausted pseudo-hit below.
          if (hit && t > tPrev) {
            let lo = tPrev,
              hi = t;
            for (let r = 0; r < 8; r++) {
              const mid = 0.5 * (lo + hi);
              if (
                de(ox + dx * mid, oy + dy * mid, oz + dz * mid) * deScale <
                eps * mid
              ) {
                hi = mid;
              } else {
                lo = mid;
              }
            }
            t = hi;
          }
          // Budget exhausted while hugging geometry → shade the last position
          // (mirrors shader.js), but only for true huggers: silhouette-graze
          // rays (d small-but-nonzero at exhaustion) stay sky, or the horizon
          // bulges around objects. The stylized ASCII tier takes the hard cut
          // at half-cone instead of an alpha blend.
          if (!hit && st >= maxSteps && lastD < eps * t * 4) hit = true;
          if (!hit) continue;
          const px = ox + dx * t,
            py = oy + dy * t,
            pz = oz + dz * t;
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
function structField(formula, g, iters) {
  const measure = makeIterMeasure(formula, "trap", iters);
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
    ? detectContours(
        structField(formula, g, opts.iters),
        g.cov,
        cols,
        rows,
        opts,
      )
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
// itersOverride (optional, #181) — the render policy's effective iteration count,
// so the colour re-iteration (glow/bands) resolves the same depth the marched
// surface did once zoomed. Undefined ⇒ formula.iters, unchanged.
// COLORING P0 — smooth escape fraction (S1 bands, #239 D6). Mirror of the WGSL/
// GLSL smoothEscFrac: a fractional offset on the integer escape count so bands
// mode stops stair-stepping. HEURISTIC — an op-list has no single power, so log2
// is a chosen constant, not log(power). Guarded rBail>1 && rEsc>1 (a bailout<1
// or r≈1 would make log2 of a non-positive ratio → NaN).
export function smoothEscFrac(rEsc, bailSq) {
  const rBail = Math.sqrt(bailSq);
  if (rBail <= 1 || rEsc <= 1) return 0;
  const f = 1 - Math.log2(Math.log(rEsc) / Math.log(rBail));
  return f < 0 ? 0 : f > 1 ? 1 : f;
}

export function makeIterMeasure(formula, kind, itersOverride, stripeFreq = 5) {
  // COLORING S2 — Silk stripe frequency (mode "silk"), clamped to the slider
  // range; captured by the measure closures below (mirrors the shaders' k).
  const kStripe = Math.max(1, Math.min(16, stripeFreq));
  // Hybrid iteration (§3.6) — glow/bands coloring "just works" on a hybrid
  // (unlike CSG scenes, which restrict to surface mode): same parseHybrid
  // config and schedule branch the makeOrbit runner uses — shared by
  // construction. Objects-first tie-break, matching makeDE above (Formula
  // Outline Step 3 §4a): a malformed dual-set formula (objects + hybrid both
  // set) must color the same shape makeDE marched.
  if (
    formula?.hybrid &&
    !(Array.isArray(formula.objects) && formula.objects.length > 0)
  ) {
    const { slots: hslots, period } = parseHybrid(formula);
    const iters = itersOverride ?? formula.iters ?? 8;
    const julia = !!formula.julia;
    const escape = hybridDeFamily(formula) !== "ifs";
    const bail = escape ? BAILOUT_ESCAPE : BAILOUT_IFS;
    const jc = julia ? formula.juliaC || [0, 0, 0] : null;
    return function measure(px, py, pz) {
      const s = { x: px, y: py, z: pz, w: 1.0 };
      const cx = jc ? jc[0] : px,
        cy = jc ? jc[1] : py,
        cz = jc ? jc[2] : pz;
      let tr = 1e9,
        esc = iters,
        rEsc2 = 0,
        silkAcc = 0,
        silkCnt = 0,
        pinR2 = 1e18,
        pinAng = 0,
        mnx = 1e9,
        mnz = 1e9;
      for (let i = 0; i < iters; i++) {
        s.i = i; // iteration index for scaleDrift (SCALE_VARY.md §6.5)
        const slot = period === 1 ? hslots[0] : hybridSlotAt(hslots, period, i);
        const ops = slot.ops;
        for (let o = 0; o < ops.length; o++) applyOp(ops[o].key, ops[o].v, s);
        if (slot.addC || julia) {
          s.x += cx;
          s.y += cy;
          s.z += cz;
        }
        const r2 = s.x * s.x + s.y * s.y + s.z * s.z;
        if (kind === "trap") tr = Math.min(tr, Math.sqrt(r2));
        if (kind === "silk") {
          silkAcc += 0.5 + 0.5 * Math.sin(kStripe * Math.atan2(s.y, s.x));
          silkCnt += 1;
        }
        if (kind === "pin" && r2 < pinR2) {
          pinR2 = r2;
          pinAng = Math.atan2(s.y, s.x);
        }
        if (kind === "irid") {
          mnx = Math.min(mnx, Math.abs(s.x));
          mnz = Math.min(mnz, Math.abs(s.z));
        }
        if (r2 > bail) {
          esc = i;
          rEsc2 = r2;
          break;
        }
      }
      if (kind === "silk") return silkAcc / Math.max(silkCnt, 1);
      if (kind === "pin") return fract(pinAng * 0.15915494 + 0.5);
      if (kind === "irid") return (mnx - mnz) / (mnx + mnz + 1e-6); // COLORING P3 scale-invariant axis asymmetry
      if (kind === "address") {
        // COLORING R S8 — sign-octant of the final orbit point → 8 colors.
        const oct = (s.x > 0 ? 1 : 0) + (s.y > 0 ? 2 : 0) + (s.z > 0 ? 4 : 0);
        return (oct + 0.5) / 8;
      }
      return kind === "trap"
        ? tr
        : (esc + smoothEscFrac(Math.sqrt(rEsc2), bail)) / Math.max(iters, 1);
    };
  }
  const ops = activeOps(formula).map((o) => ({
    key: o.key,
    v: o.values || [],
  }));
  const iters = itersOverride ?? formula.iters ?? 8;
  // +c gate mirrors the renderer (addGate = addC || julia) — see makeDE above (#16).
  const addC = !!formula.addC || !!formula.julia;
  const escape = isEscapeTime(formula);
  const bail = escape ? BAILOUT_ESCAPE : BAILOUT_IFS;
  const jc = formula.julia ? formula.juliaC || [0, 0, 0] : null;
  return function measure(px, py, pz) {
    const s = { x: px, y: py, z: pz, w: 1.0 };
    const cx = jc ? jc[0] : px,
      cy = jc ? jc[1] : py,
      cz = jc ? jc[2] : pz;
    let tr = 1e9,
      esc = iters,
      rEsc2 = 0,
      silkAcc = 0,
      silkCnt = 0,
      pinR2 = 1e18,
      pinAng = 0,
      mnx = 1e9,
      mnz = 1e9;
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
      if (kind === "silk") {
        silkAcc += 0.5 + 0.5 * Math.sin(kStripe * Math.atan2(s.y, s.x));
        silkCnt += 1;
      }
      if (kind === "pin" && r2 < pinR2) {
        pinR2 = r2;
        pinAng = Math.atan2(s.y, s.x);
      }
      if (kind === "irid") {
        mnx = Math.min(mnx, Math.abs(s.x));
        mnz = Math.min(mnz, Math.abs(s.z));
      }
      if (r2 > bail) {
        esc = i;
        rEsc2 = r2;
        break;
      }
    }
    if (kind === "silk") return silkAcc / Math.max(silkCnt, 1);
    if (kind === "pin") return fract(pinAng * 0.15915494 + 0.5);
    if (kind === "irid") return (mnx - mnz) / (mnx + mnz + 1e-6); // COLORING P3 scale-invariant axis asymmetry
    if (kind === "address") {
      const oct = (s.x > 0 ? 1 : 0) + (s.y > 0 ? 2 : 0) + (s.z > 0 ? 4 : 0);
      return (oct + 0.5) / 8;
    }
    return kind === "trap"
      ? tr
      : (esc + smoothEscFrac(Math.sqrt(rEsc2), bail)) / Math.max(iters, 1);
  };
}

const clamp01 = (v) => Math.max(0, Math.min(1, v));

// COLORING S4 — Curvature tint (CPU/ASCII mirror of shader.js curvatureAt): the
// discrete Laplacian of the DE field via the same tetrahedron taps as the normal
// (gradient cancels; residual is second-order). `de` may be a scene DE, so this
// is the one signal that colors CSG scenes for real. A fixed object-space eps
// (the fallback tier isn't zoom-deep); f64 has ample precision headroom here.
export function curvatureMixT(de, x, y, z) {
  const ce = 6e-3;
  const lap =
    de(x + ce, y - ce, z - ce) +
    de(x - ce, y - ce, z + ce) +
    de(x - ce, y + ce, z - ce) +
    de(x + ce, y + ce, z + ce) -
    4 * de(x, y, z);
  return clamp01(0.5 + 0.5 * Math.tanh((lap / (ce * ce)) * 0.15));
}

// Shaded surface colour [0..255], mirroring the WGSL fragment's albedo + light:
// cosine palette OR colA→colB by mixT, then diffuse+ambient+AO and a rim glow,
// gamma-corrected. (Skips the WGSL distance-fade/spec — keeps the chars punchy.)
// COLORING R S7 — one palette read (stops / cosine / ramp) at coordinate t, in
// sRGB. Mirror of shader.js albedoFor + GL palLookupGL, applying palettePhase so
// the Painter's per-iteration reads match the GPU tiers. (shadeRGB's inline path
// keeps its own copy — phase is applied to its mixT upstream in shadeGrid.)
function palLookupCPU(coloring, t) {
  const pal = coloring.palette || {};
  const phase = coloring.palettePhase ?? 0;
  const tt = phase ? fract(t + phase) : t;
  if (Array.isArray(pal.stops) && pal.stops.length >= 2)
    return sampleStops(pal.stops, tt, pal.cyclic);
  if (pal.on) {
    const a = pal.a || [0.5, 0.5, 0.5],
      b = pal.b || [0.5, 0.5, 0.5],
      c = pal.c || [1, 1, 1],
      d = pal.d || [0, 0.33, 0.67];
    return [0, 1, 2].map((i) =>
      clamp01(a[i] + b[i] * Math.cos(6.2831853 * (c[i] * tt + d[i]))),
    );
  }
  const A = coloring.colA || [0.86, 0.46, 0.18],
    B = coloring.colB || [0.18, 0.62, 0.74];
  return [0, 1, 2].map((i) => A[i] + (B[i] - A[i]) * tt);
}

// COLORING R S7 — Painter measure: run the orbit and blend a palette color PER
// ITERATION, weighted by trap proximity (exp(−d)). Returns sRGB albedo. Mirror
// of shader.js orbitPainter (non-hybrid + hybrid via activeOps/effective iters).
export function makePainterMeasure(formula, coloring, itersOverride) {
  const ops = activeOps(formula).map((o) => ({
    key: o.key,
    v: o.values || [],
  }));
  const iters = itersOverride ?? formula.iters ?? 8;
  const addC = !!formula.addC || !!formula.julia;
  const escape = isEscapeTime(formula);
  const bail = escape ? BAILOUT_ESCAPE : BAILOUT_IFS;
  const jc = formula.julia ? formula.juliaC || [0, 0, 0] : null;
  return function painter(px, py, pz) {
    const s = { x: px, y: py, z: pz, w: 1.0 };
    const cx = jc ? jc[0] : px,
      cy = jc ? jc[1] : py,
      cz = jc ? jc[2] : pz;
    const col = [0, 0, 0];
    let wsum = 0;
    for (let i = 0; i < iters; i++) {
      s.i = i;
      for (let k = 0; k < ops.length; k++) applyOp(ops[k].key, ops[k].v, s);
      if (addC) {
        s.x += cx;
        s.y += cy;
        s.z += cz;
      }
      const d = Math.hypot(s.x, s.y, s.z);
      const ti = fract(d * 0.35 + i * 0.03);
      const wt = Math.exp(-1.5 * d);
      const a = palLookupCPU(coloring, ti);
      col[0] += wt * a[0];
      col[1] += wt * a[1];
      col[2] += wt * a[2];
      wsum += wt;
      if (d * d > bail) break;
    }
    const inv = 1 / Math.max(wsum, 1e-4);
    return [col[0] * inv, col[1] * inv, col[2] * inv];
  };
}

// The pre-lighting display-sRGB albedo for a coloring at signal `mixT` — the
// non-`albOverride` branches of shadeRGB, extracted verbatim so shadeRGB and the
// splat-export makePointAlbedo (below) can't drift. Returns sRGB floats in [0,1]
// (NOT linearized — the caller applies its own gamma if it needs light-space).
function paletteAlbedo(coloring, mixT) {
  const pal = coloring.palette || {};
  if (Array.isArray(pal.stops) && pal.stops.length >= 2)
    return sampleStops(pal.stops, mixT, pal.cyclic);
  if (pal.on) {
    const a = pal.a || [0.5, 0.5, 0.5],
      b = pal.b || [0.5, 0.5, 0.5];
    const c = pal.c || [1, 1, 1],
      d = pal.d || [0, 0.33, 0.67];
    return [0, 1, 2].map((i) =>
      clamp01(a[i] + b[i] * Math.cos(6.2831853 * (c[i] * mixT + d[i]))),
    );
  }
  const A = coloring.colA || [0.86, 0.46, 0.18];
  const B = coloring.colB || [0.18, 0.62, 0.74];
  return [0, 1, 2].map((i) => A[i] + (B[i] - A[i]) * mixT);
}

function shadeRGB(coloring, mixT, nx, ny, nz, dx, dy, dz, ao, albOverride) {
  const L = coloring.light || {};
  const amb = L.ambient ?? 0.16,
    rimAmt = L.rim ?? 0.45,
    intensity = L.intensity ?? 1.0;
  const ld = norm3(...(L.dir || [0.395, 0.657, 0.643])); // #160 Z-up default
  const diff = Math.max(nx * ld[0] + ny * ld[1] + nz * ld[2], 0);
  const rim = Math.pow(1 - Math.max(-(nx * dx + ny * dy + nz * dz), 0), 2);
  const B = coloring.colB || [0.18, 0.62, 0.74];
  // sRGB → linear: colors are authored/picked in sRGB; linearize before lighting
  // so the 1/2.2 encode below round-trips them to the picked color (issue #6).
  const s2l = (x) => Math.pow(Math.max(x, 0), 2.2);
  const Blin = B.map(s2l);
  // COLORING R S7 — the Painter blends its own sRGB albedo per iteration; use it
  // directly. Otherwise the mixT → palette read (paletteAlbedo, shared source).
  let alb = albOverride || paletteAlbedo(coloring, mixT);
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
// COLORING P2 — auto-levels. The signal modes that get range-normalized:
// glow(1), bands(2), silk(3), curvature(5). Surface(0) is already the [0,1]
// normal and Pinwheel(4) is cyclic (remapping would break its seam) — both stay
// identity. The shader applies (mixT-lo)/span unconditionally; non-normalizable
// modes just receive lo=0, span=1.
export const AUTOLEVEL_MODES = new Set([1, 2, 3, 5]);

// The raw color signal at a traced cell, BEFORE palette or level-normalization.
// Single source of truth shared by shadeGrid (the render) and sampleSignalMixT
// (the range sampler) so the two never drift. `m` holds the per-mode measure
// closures {trap, esc, silk, pin, curvDE}; mirrors the GPU tiers' mixTFor.
// The raw signal at a WORLD POINT (x,y,z) with world-normal-z `nz` — the
// extractable core of rawMixT. Shared by the grid render (rawMixT below) and the
// splat-export makePointAlbedo, so the two never drift.
function rawMixTAt(mode, x, y, z, nz, m, scene) {
  if (mode === 5) return curvatureMixT(m.curvDE, x, y, z);
  if (scene) {
    // COLORING P3 S5 — Glow/Bands/Silk + Address (7) run the winning object's
    // real orbit; Painter (6) falls to per-object Glow on scenes (S7).
    if (((mode >= 1 && mode <= 3) || mode === 6 || mode === 7) && m.sceneOrbit)
      return m.sceneOrbit(x, y, z);
    if (mode === 4) return fract(Math.atan2(y, x) * 0.15915494 + 0.5); // Pinwheel → world azimuth
    return 0.5 + 0.5 * nz; // fallback
  }
  if (mode === 7) return m.address(x, y, z); // S8 sign-octant
  if (mode === 4) return m.pin(x, y, z);
  if (mode === 3) return m.silk(x, y, z);
  if (mode === 2) return m.esc(x, y, z);
  if (mode === 1) return Math.min(m.trap(x, y, z) / 1.5, 1);
  return 0.5 + 0.5 * nz; // surface
}

function rawMixT(mode, g, i, m, scene) {
  return rawMixTAt(mode, g.hx[i], g.hy[i], g.hz[i], g.nz[i], m, scene);
}

// Build the per-mode measure closures for a coloring mode (shared setup).
// COLORING P3 S5 — a scene (formula.objects[]) drives glow/bands/silk from the
// winning object's OWN orbit (makeSceneMeasure), not the flat closures (which
// have no top-level ops on a scene). Curvature (5) stays geometry-space.
function signalClosures(formula, mode, coloring, iters) {
  const scene = Array.isArray(formula.objects) && formula.objects.length > 0;
  if (scene) {
    return {
      // Glow/Bands/Silk (1–3) + Address (7) run the winning object's orbit;
      // Painter (6) has no per-object codegen on scenes → per-object Glow.
      sceneOrbit:
        (mode >= 1 && mode <= 3) || mode === 6 || mode === 7
          ? makeSceneMeasure(
              formula,
              mode === 6 ? 1 : mode,
              coloring.stripeFreq,
            )
          : null,
      curvDE: mode === 5 ? makeDE(formula, iters) : null,
    };
  }
  return {
    trap: mode === 1 ? makeIterMeasure(formula, "trap", iters) : null,
    esc: mode === 2 ? makeIterMeasure(formula, "escape", iters) : null,
    silk:
      mode === 3
        ? makeIterMeasure(formula, "silk", iters, coloring.stripeFreq)
        : null,
    pin: mode === 4 ? makeIterMeasure(formula, "pin", iters) : null,
    curvDE: mode === 5 ? makeDE(formula, iters) : null,
    address: mode === 7 ? makeIterMeasure(formula, "address", iters) : null,
  };
}

// Per-world-point display-sRGB albedo — the splat-export twin of the live grid/
// GPU color chain (rawMixT → signalRange lo/span → iridescence → palettePhase →
// palette), with NO lighting/rim/AO baked (spec D1) and NO gamma applied (SH0
// stores display-sRGB as-is). Returns albedoAt(x, y, z, nz) → [r, g, b] in [0,1].
// `nz` = world-frame normal z (mode-0 Surface + the scene fallback need it).
// Uses signalRange — the memoized canonical-view range the GPU tiers key off
// (not shadeGrid's own-grid self-leveling) — so the export matches the live
// render exactly (spec §5.4 step 2). Mirrors shadeGrid's per-cell order verbatim.
export function makePointAlbedo(formula, coloring, iters) {
  // Survive a missing coloring (#432). The `?.` on mode below always intended
  // this — but `palettePhase` (and paletteAlbedo's `.palette`) were left raw
  // and threw, which is how an export job with no coloring crashed the CPU
  // capture + refine workers. `{}` rather than defaultColoring() on purpose:
  // every read downstream already has its own `|| default` fallback, and the
  // absent-field semantics matter (autoLevels is OFF when absent by design —
  // old saves predate it). Callers wanting the default LOOK merge it in.
  coloring = coloring || {};
  const mode = coloring.mode || 0;
  const scene = Array.isArray(formula.objects) && formula.objects.length > 0;
  const m = signalClosures(formula, mode, coloring, iters);
  const { lo, span } = signalRange(formula, coloring, iters); // identity when off
  const levels = lo !== 0 || span !== 1;
  const iridAmt = mode === 1 && !scene ? (coloring.iridescence ?? 0) : 0;
  const iridFn =
    iridAmt > 0.001 ? makeIterMeasure(formula, "irid", iters) : null;
  const phase = coloring.palettePhase ?? 0;
  const painter =
    mode === 6 && !scene ? makePainterMeasure(formula, coloring, iters) : null;
  // Scene SURFACE mode (0) → the winning object's own color (GPU sceneTint),
  // not the normal-tint palette fallback that made scene exports monochromatic.
  const sceneTint = scene && mode === 0 ? makeSceneTint(formula) : null;
  return function albedoAt(x, y, z, nz) {
    if (painter) return painter(x, y, z); // already sRGB
    if (sceneTint) return sceneTint(x, y, z); // already sRGB (object color)
    let mixT = rawMixTAt(mode, x, y, z, nz, m, scene);
    if (levels) mixT = clamp01((mixT - lo) / span);
    if (iridFn) mixT = fract(mixT + iridAmt * iridFn(x, y, z));
    if (phase) mixT = fract(mixT + phase);
    return paletteAlbedo(coloring, mixT);
  };
}

// Robust [lo, span] from a set of signal samples — 3rd/97th percentiles (reject
// outliers) with a floor so a near-constant signal isn't blown up into noise.
// span is (hi - lo); the shader does (mixT - lo) / span.
export function robustRange(vals, floor = 0.06) {
  if (!vals || vals.length < 8) return { lo: 0, span: 1 };
  const s = Float64Array.from(vals).sort(); // TypedArray sort is numeric
  const at = (q) =>
    s[Math.max(0, Math.min(s.length - 1, Math.round(q * (s.length - 1))))];
  const lo = at(0.03);
  const span = Math.max(at(0.97) - lo, floor);
  return { lo, span };
}

// Sample the raw color signal over a coarse canonical-view trace → the array of
// mixT values at covered cells. Camera-independent (traceGrid builds its own
// camera from formula.camera, not the live orbit) so the range is stable as the
// user rotates. Reused by signalRange (GPU tiers) below.
export function sampleSignalMixT(
  formula,
  coloring,
  { iters, cols = 40, rows = 26 } = {},
) {
  const mode = coloring.mode || 0;
  const m = signalClosures(formula, mode, coloring, iters);
  const g = traceGrid(formula, { cols, rows, iters });
  const scene = Array.isArray(formula.objects) && formula.objects.length > 0;
  const out = [];
  for (let i = 0; i < cols * rows; i++) {
    if (g.cov[i] <= 0) continue;
    out.push(rawMixT(mode, g, i, m, scene));
  }
  return out;
}

// Memoized signal range for the GPU tiers. Keyed on the formula OBJECT identity
// (a fresh formula from setFormula misses → recompute) + mode/stripeFreq/iters,
// so the ~1k-cell sample runs once per formula/mode change, not per frame
// (deriveFrameParams runs every frame). Returns identity for off / cyclic /
// surface so the caller can pass the result straight through.
const _sigRangeMemo = new WeakMap();
export function signalRange(formula, coloring, iters) {
  const mode = coloring?.mode || 0;
  if (!formula || !coloring?.autoLevels || !AUTOLEVEL_MODES.has(mode)) {
    return { lo: 0, span: 1 };
  }
  let sub = _sigRangeMemo.get(formula);
  if (!sub) {
    sub = new Map();
    _sigRangeMemo.set(formula, sub);
  }
  const key = `${mode}:${coloring.stripeFreq ?? 5}:${iters ?? 0}`;
  let r = sub.get(key);
  if (!r) {
    r = robustRange(sampleSignalMixT(formula, coloring, { iters }));
    sub.set(key, r);
  }
  return r;
}

// Exported for the tile-seam gate (TILED_EXPORT.md §3/PR-1,
// core/tileseam.test.mjs): this is the shared grid behind renderAsciiColored /
// renderAsciiAnsi, and its { chars, rgb } IS the CPU tier's image. The two
// public renderers wrap it in run-length HTML/ANSI, whose run BOUNDARIES shift
// when a row is assembled from tiles even where every pixel agrees — so they
// cannot express a byte-exact stitch comparison and this can.
//
// Tile-invariance caveat, load-bearing for that test: `coloring.autoLevels`
// normalises against a robustRange over the WHOLE grid, which is a global
// reduction and therefore NOT tile-invariant on this tier (unlike the GPU
// tiers, where signalRange samples formula space). Same for `edges` /
// `structure` (neighbourhood ops) and `dither` (a function of the LOCAL row/
// col). All four default off; the seam test leaves them off and says so.
export function shadeGrid(formula, opts) {
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
  // #181 — colour re-iteration uses the same effective iters (opts.iters) as the
  // marched surface, so glow/bands don't collapse a zoomed shape into plateaus.
  // The per-mode measure closures (trap/esc/silk/pin/curvDE) + the raw-signal
  // expression are shared with the sampler via signalClosures/rawMixT.
  const m = signalClosures(formula, mode, coloring, opts.iters);
  // CHAR intensity uses the SAME light the colour does (so the form reads the
  // way the palette is lit) — this is what the old per-cell `ld` did.
  const lightDir = norm3(
    ...((coloring.light && coloring.light.dir) || [0.395, 0.657, 0.643]), // #160 Z-up default
  );
  const g = traceGrid(formula, { ...opts, cols, rows, lightDir });
  const edge = edges ? detectEdges(g, opts) : null;
  const struct = structure
    ? detectContours(
        structField(formula, g, opts.iters),
        g.cov,
        cols,
        rows,
        opts,
      )
    : null;
  const last = ramp.length - 1;
  const n = cols * rows;
  const chars = new Array(n).fill(" ");
  const rgb = new Array(n).fill(null);
  const scene = Array.isArray(formula.objects) && formula.objects.length > 0;
  // COLORING P2 — auto-levels (CPU/ASCII tier self-levels from its OWN grid): a
  // first pass caches the raw signal per covered cell, then normalizes by the
  // grid's robust [lo, span] so the palette spans the actual signal range. Gated
  // on coloring.autoLevels and the normalizable modes (surface/pinwheel are
  // identity). Scenes have no orbit data (SCENES.md §Coloring): Bands keys by
  // radial distance, Glow/Silk/Surface by the normal — see rawMixT.
  const mixTArr = new Float64Array(n);
  const covered = [];
  for (let i = 0; i < n; i++) {
    if (g.cov[i] <= 0) continue;
    mixTArr[i] = rawMixT(mode, g, i, m, scene);
    covered.push(i);
  }
  let sigLo = 0,
    sigSpan = 1;
  if (coloring.autoLevels && AUTOLEVEL_MODES.has(mode)) {
    const vals = covered.map((i) => mixTArr[i]);
    ({ lo: sigLo, span: sigSpan } = robustRange(vals));
  }
  const levels = sigLo !== 0 || sigSpan !== 1;
  // COLORING P3 iridescence (S6) — Glow-only (mode 1), flat formulas only, applied
  // AFTER auto-levels as a per-pixel palette-phase shift (mirrors shader.js).
  const iridAmt = mode === 1 && !scene ? (coloring.iridescence ?? 0) : 0;
  const iridFn =
    iridAmt > 0.001 ? makeIterMeasure(formula, "irid", opts.iters) : null;
  // COLORING P3 — palette phase (rotate the palette lookup; 0 = identity).
  const phase = coloring.palettePhase ?? 0;
  // COLORING R S7 — Painter (mode 6, flat formulas): a per-iteration palette
  // blend, computed as a direct sRGB albedo (bypasses the mixT path). Scenes have
  // no single orbit → they fall to per-object Glow (see rawMixT / signalClosures).
  const painter =
    mode === 6 && !scene
      ? makePainterMeasure(formula, coloring, opts.iters)
      : null;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const i = r * cols + c;
      if (g.cov[i] <= 0) continue;
      const nx = g.nx[i],
        ny = g.ny[i],
        nz = g.nz[i];
      let mixT = levels ? clamp01((mixTArr[i] - sigLo) / sigSpan) : mixTArr[i];
      if (iridFn)
        mixT = fract(mixT + iridAmt * iridFn(g.hx[i], g.hy[i], g.hz[i]));
      if (phase) mixT = fract(mixT + phase);
      const albOv = painter ? painter(g.hx[i], g.hy[i], g.hz[i]) : undefined;
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
        albOv,
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

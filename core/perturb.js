// ─────────────────────────────────────────────────────────────────────────
// Perturbation deep zoom — reference orbit + f32 delta kernel (PR-1).
// docs/planning/PERTURBATION_ZOOM.md (assessment + spike evidence) and
// PERTURBATION_ZOOM_IMPL.md (this build's plan) are the governing docs.
// ─────────────────────────────────────────────────────────────────────────
// The idea: iterate ONE reference orbit of the zoom target T in extended
// precision on the CPU, record per-(iteration, op) values + auxiliaries
// (fold margins, r², k) rounded to f32, and let each render sample iterate
// only its small residual δ = p − T through exact per-op delta maps in
// plain f32. Cancellation is done algebraically (products of small×O(1),
// never differences of near-equal O(1)), so the GPU kernel needs no EFTs
// and no df_launder — measured on Metal (P1 spike, harness/perturb.html).
//
// The extended-precision engine is BigInt fixed point, 240 fractional bits
// (plan D1): margins are genuine cancellations f − Z that must be computed
// high-precision THEN rounded — a small float keeps full relative precision
// at any magnitude, so f32 storage is exact enough once the subtraction
// happened exactly. ~34 + 3.32·N bits are needed at ×10ᴺ (the §7 margin
// rule, P0-measured: a double-double reference cliffs at ~×10²²); 240 bits
// clear the ×10³⁰ δ-underflow wall outright.
//
// This module is pure math — no GPU, no renderer imports. Consumers:
//   · PR-2's WGSL kernel reads `packed` via a storage buffer (layout below)
//   · the deep zoom-to-surface probe evaluates deltaDE in JS (plan D8)
//   · core/perturb.test.mjs pins everything against the exact map
// Op semantics mirror cpuorbit.js applyOp / makeOrbit (the CPU tier truth);
// the addC gate mirrors makeOrbit: (addC || julia), c = julia ? juliaC : T.

import { byKey, activeOps } from "./operators.js";
import { BAILOUT_IFS } from "./limits.js";

const F = Math.fround;
const D2R = Math.PI / 180;

// ── the v1 op set (plan D7) ───────────────────────────────────────────────
// Presence here = delta-map membership (the wgslDf convention: no separate
// list to drift — ptEligible walks this via ptSupported). Every op below has
// an EXACT delta map: constant-affine, piecewise-affine with margins, or
// rational-radial with algebraic k-differences. The two inversions are the
// λ̂ = 0 family df64 cannot serve (Tier B₁) — perturbation is their first
// deep tier.
const PT_OPS = new Set([
  "boxFold",
  "boxFoldXYZ",
  "surfFold",
  "absFold",
  "absXYZ",
  "sphereFold",
  "cylinderFold",
  "sphereInv",
  "radialInvert",
  "scale",
  "translate",
  "rotateXY",
  "rotateYZ",
  "rotateXZ",
  "rotateXYZ",
]);
export const ptSupported = (key) => PT_OPS.has(key);

// τ-switchover threshold (plan D3): once |δ| > τ·max(1, |Z|) the sample is
// macroscopically separated from the reference and the plain-f32 tail is
// fully adequate. Any τ ≫ ~1e-5 is safe (the f32 tail's absolute error
// 2⁻²⁴·|Z| is invisible once |δ| clears it with contrast margin); 1e-2
// sits three decades above that floor. P0 pins the boundary.
export const PT_TAU = 1e-2;

// ── BigInt fixed point, 240 fractional bits ───────────────────────────────
const FB = 240n;
const FXONE = 1n << FB;

export function fxFromF64(v) {
  if (!Number.isFinite(v)) throw new Error("fxFromF64: " + v);
  const neg = v < 0;
  if (neg) v = -v;
  let r = BigInt(Math.floor(v)) << FB;
  let frac = v - Math.floor(v),
    sh = FB;
  while (frac > 0 && sh >= 24n) {
    frac *= 16777216; // 2^24 — exact (dyadic)
    const w = Math.floor(frac);
    frac -= w;
    sh -= 24n;
    r += BigInt(w) << sh;
  }
  return neg ? -r : r;
}

export function fxToF64(a) {
  const neg = a < 0n;
  if (neg) a = -a;
  if (a === 0n) return 0;
  const bits = a.toString(2).length;
  const sh = BigInt(Math.max(0, bits - 53));
  const v = Number(a >> sh) * 2 ** (Number(sh) - 240);
  return neg ? -v : v;
}

const fxMul = (a, b) => (a * b) >> FB;
const fxDiv = (a, b) => (a << FB) / b;
const fxAbs = (a) => (a < 0n ? -a : a);

// Target helpers (plan D6): the canonical deep-zoom target is a BigInt
// triple; re-pins nudge it by an f64 delta EXACTLY (each delta is dyadic).
export const targetToFx = (t) => [fxFromF64(t[0] ?? 0), fxFromF64(t[1] ?? 0), fxFromF64(t[2] ?? 0)];
export const fxTargetToF64 = (Tfx) => [fxToF64(Tfx[0]), fxToF64(Tfx[1]), fxToF64(Tfx[2])];
export const fxNudge = (Tfx, d) => [
  Tfx[0] + fxFromF64(d[0]),
  Tfx[1] + fxFromF64(d[1]),
  Tfx[2] + fxFromF64(d[2]),
];

// ── per-op constants, f64-rounded exactly as the engine computes them ─────
// Params fall back to the registry defaults (the lambdaHat convention) so an
// absent `values` matches what the shader would run.
function prepOps(formula) {
  return activeOps(formula)
    .filter((op) => PT_OPS.has(op.key))
    .map((op) => {
      const def = byKey(op.key);
      const v = (i) => op.values?.[i] ?? def?.params?.[i]?.default ?? 0;
      const c = { key: op.key, v: [v(0), v(1), v(2)] };
      switch (op.key) {
        case "sphereFold":
        case "cylinderFold":
          c.minR2 = c.v[0] * c.v[0];
          c.fixedR2 = c.v[1] * c.v[1];
          c.K0 = c.fixedR2 / c.minR2;
          break;
        case "sphereInv":
          c.r2p = c.v[0] * c.v[0];
          break;
        case "rotateXY":
        case "rotateYZ":
        case "rotateXZ": {
          const a = c.v[0] * D2R;
          c.rot = [[Math.cos(a), Math.sin(a)]];
          break;
        }
        case "rotateXYZ":
          c.rot = c.v.map((deg) => {
            const a = deg * D2R;
            return [Math.cos(a), Math.sin(a)];
          });
          break;
      }
      return c;
    });
}

// ── the exact map, in fx (branch form — identical values to applyOp) ──────
const CLAMP_INV = fxFromF64(1e-6); // both inversions clamp |D|² at 1e-6

function fxRotate(s, ax0, ax1, co, sn) {
  const n0 = fxMul(s[ax0], co) - fxMul(s[ax1], sn);
  const n1 = fxMul(s[ax0], sn) + fxMul(s[ax1], co);
  s[ax0] = n0;
  s[ax1] = n1;
}

function fxStep(c, s) {
  switch (c.key) {
    case "boxFold":
    case "boxFoldXYZ":
    case "surfFold": {
      const axes = c.key === "surfFold" ? ["x", "y"] : ["x", "y", "z"];
      axes.forEach((ax, i) => {
        const f = fxFromF64(c.key === "boxFoldXYZ" ? c.v[i] : c.v[0]);
        const x = s[ax];
        s[ax] = x > f ? 2n * f - x : x < -f ? -2n * f - x : x;
      });
      break;
    }
    case "absFold":
      for (const ax of ["x", "y", "z"]) s[ax] = fxAbs(s[ax]);
      break;
    case "absXYZ":
      ["x", "y", "z"].forEach((ax, i) => {
        if (c.v[i] > 0.5) s[ax] = fxAbs(s[ax]);
      });
      break;
    case "sphereFold":
    case "cylinderFold": {
      const xy = c.key === "cylinderFold";
      const minR2 = fxFromF64(c.minR2),
        fixedR2 = fxFromF64(c.fixedR2);
      const r2 =
        fxMul(s.x, s.x) + fxMul(s.y, s.y) + (xy ? 0n : fxMul(s.z, s.z));
      const k =
        r2 < minR2 ? fxDiv(fixedR2, minR2) : r2 < fixedR2 ? fxDiv(fixedR2, r2) : FXONE;
      s.x = fxMul(s.x, k);
      s.y = fxMul(s.y, k);
      s.z = fxMul(s.z, k);
      s.w = fxMul(s.w, k);
      break;
    }
    case "sphereInv": {
      const r2p = fxFromF64(c.r2p);
      let d = fxMul(s.x, s.x) + fxMul(s.y, s.y) + fxMul(s.z, s.z);
      if (d < CLAMP_INV) d = CLAMP_INV;
      const k = fxDiv(r2p, d);
      s.x = fxMul(s.x, k);
      s.y = fxMul(s.y, k);
      s.z = fxMul(s.z, k);
      s.w = fxMul(s.w, k);
      break;
    }
    case "radialInvert": {
      const o = c.v.map(fxFromF64);
      const dx = s.x - o[0],
        dy = s.y - o[1],
        dz = s.z - o[2];
      let dd = fxMul(dx, dx) + fxMul(dy, dy) + fxMul(dz, dz);
      if (dd < CLAMP_INV) dd = CLAMP_INV;
      const k = fxDiv(FXONE, dd);
      s.x = fxMul(dx, k) + o[0];
      s.y = fxMul(dy, k) + o[1];
      s.z = fxMul(dz, k) + o[2];
      s.w = fxMul(s.w, k);
      break;
    }
    case "scale": {
      const k = fxFromF64(c.v[0]);
      s.x = fxMul(s.x, k);
      s.y = fxMul(s.y, k);
      s.z = fxMul(s.z, k);
      s.w = fxMul(s.w, fxAbs(k));
      break;
    }
    case "translate":
      s.x += fxFromF64(c.v[0]);
      s.y += fxFromF64(c.v[1]);
      s.z += fxFromF64(c.v[2]);
      break;
    case "rotateXY":
      fxRotate(s, "x", "y", fxFromF64(c.rot[0][0]), fxFromF64(c.rot[0][1]));
      break;
    case "rotateYZ":
      fxRotate(s, "y", "z", fxFromF64(c.rot[0][0]), fxFromF64(c.rot[0][1]));
      break;
    case "rotateXZ":
      fxRotate(s, "x", "z", fxFromF64(c.rot[0][0]), fxFromF64(c.rot[0][1]));
      break;
    case "rotateXYZ":
      fxRotate(s, "x", "y", fxFromF64(c.rot[0][0]), fxFromF64(c.rot[0][1]));
      fxRotate(s, "y", "z", fxFromF64(c.rot[1][0]), fxFromF64(c.rot[1][1]));
      fxRotate(s, "x", "z", fxFromF64(c.rot[2][0]), fxFromF64(c.rot[2][1]));
      break;
    default:
      throw new Error("perturb fxStep: unsupported op " + c.key);
  }
}

// ── record layout (plan D2 — must match PR-2's WGSL) ─────────────────────
// Per (iteration, op) slot, 16 f32 = 4 vec4:
//   [ 0.. 2] Z (f32 of the fx value entering the op)   [ 3] kr
//   [ 4.. 6] A3  (fold m1 = f−Z per axis | inversion D = Z−o)
//   [ 7]     r2  (sphere/cylinder rr2 | inversion |D|²)
//   [ 8..10] B3  (fold m2 = Z+f per axis)
//   [11]     mA  (sphere mr1 = rr2−minR2 | inversion mc = |D|²−1e-6)
//   [12]     mB  (sphere mr2 = fixedR2−rr2)
//   [13..15] 0
// One trailer slot after the last op: [finalZ.xyz, escapeAt] — bailout and
// final reconstruction read Z_next + δ, and the last iteration's Z_next is
// the trailer. (The plan text said "per iteration"; slot (i+1, 0) already
// carries every non-final iteration's end-Z, so one global trailer is the
// whole need — recorded here as the as-built truth.)
export const PT_SLOT_F32 = 16;

function record(c, s, out, off) {
  const zf = [fxToF64(s.x), fxToF64(s.y), fxToF64(s.z)];
  out[off] = F(zf[0]);
  out[off + 1] = F(zf[1]);
  out[off + 2] = F(zf[2]);
  switch (c.key) {
    case "boxFold":
    case "boxFoldXYZ":
    case "surfFold": {
      const axes = c.key === "surfFold" ? 2 : 3;
      const comps = [s.x, s.y, s.z];
      for (let i = 0; i < axes; i++) {
        const f = fxFromF64(c.key === "boxFoldXYZ" ? c.v[i] : c.v[0]);
        out[off + 4 + i] = F(fxToF64(f - comps[i])); // m1
        out[off + 8 + i] = F(fxToF64(comps[i] + f)); // m2
      }
      break;
    }
    case "sphereFold":
    case "cylinderFold": {
      const xy = c.key === "cylinderFold";
      const r2 =
        fxMul(s.x, s.x) + fxMul(s.y, s.y) + (xy ? 0n : fxMul(s.z, s.z));
      const r2f = fxToF64(r2);
      out[off + 7] = F(r2f);
      out[off + 11] = F(fxToF64(r2 - fxFromF64(c.minR2)));
      out[off + 12] = F(fxToF64(fxFromF64(c.fixedR2) - r2));
      out[off + 3] = F(
        r2f < c.minR2 ? c.K0 : r2f < c.fixedR2 ? fxToF64(fxDiv(fxFromF64(c.fixedR2), r2)) : 1,
      );
      break;
    }
    case "sphereInv":
    case "radialInvert": {
      const o = c.key === "sphereInv" ? [0, 0, 0] : c.v;
      const num = c.key === "sphereInv" ? c.r2p : 1;
      const dx = s.x - fxFromF64(o[0]),
        dy = s.y - fxFromF64(o[1]),
        dz = s.z - fxFromF64(o[2]);
      let dd = fxMul(dx, dx) + fxMul(dy, dy) + fxMul(dz, dz);
      out[off + 4] = F(fxToF64(dx));
      out[off + 5] = F(fxToF64(dy));
      out[off + 6] = F(fxToF64(dz));
      out[off + 7] = F(fxToF64(dd));
      out[off + 11] = F(fxToF64(dd - CLAMP_INV)); // mc
      if (dd < CLAMP_INV) dd = CLAMP_INV;
      out[off + 3] = F(fxToF64(fxDiv(fxFromF64(num), dd)));
      break;
    }
    default:
      break; // affine ops need Z only
  }
}

// ── buildOrbit ────────────────────────────────────────────────────────────
// The reference orbit of the zoom target through the formula, packed for
// the GPU and consumable by the JS kernel below. `target` is a number[3]
// (shallow) or a BigInt[3] from targetToFx/fxNudge (deep — past ~×10¹⁶ the
// target exceeds f64, plan D6). Runs the FULL `iters` regardless of the
// reference's own bailout; `escapeAt` marks the first post-add escape and
// the kernel forces its switchover there (assessment §4: post-bailout
// magnitudes are O(bail), f32-relative is all a sample needs).
export function buildOrbit(formula, target, iters) {
  const ops = prepOps(formula);
  const all = activeOps(formula);
  if (ops.length !== all.length)
    throw new Error(
      "buildOrbit: unsupported op in formula (gate with ptSupported): " +
        all.map((o) => o.key).filter((k) => !PT_OPS.has(k)).join(","),
    );
  const julia = !!formula.julia;
  const jc = julia ? formula.juliaC || [0, 0, 0] : null;
  const addGate = !!formula.addC || julia;
  const Tfx = typeof target[0] === "bigint" ? target.slice(0, 3) : targetToFx(target);
  const T = fxTargetToF64(Tfx);
  const n = Math.max(1, Math.floor(iters ?? formula.iters ?? 8));
  const opCount = ops.length;
  const packed = new Float32Array((n * opCount + 1) * PT_SLOT_F32);
  const s = { x: Tfx[0], y: Tfx[1], z: Tfx[2], w: FXONE };
  const cAdd = julia ? targetToFx(jc) : [Tfx[0], Tfx[1], Tfx[2]];
  const bail = fxFromF64(BAILOUT_IFS);
  let escapeAt = n;
  let off = 0;
  for (let i = 0; i < n; i++) {
    for (const c of ops) {
      record(c, s, packed, off);
      off += PT_SLOT_F32;
      fxStep(c, s);
    }
    if (addGate) {
      s.x += cAdd[0];
      s.y += cAdd[1];
      s.z += cAdd[2];
    }
    if (escapeAt === n) {
      const r2 = fxMul(s.x, s.x) + fxMul(s.y, s.y) + fxMul(s.z, s.z);
      if (r2 > bail) escapeAt = i + 1;
    }
  }
  const finalZ = [F(fxToF64(s.x)), F(fxToF64(s.y)), F(fxToF64(s.z))];
  packed[off] = finalZ[0];
  packed[off + 1] = finalZ[1];
  packed[off + 2] = finalZ[2];
  packed[off + 3] = escapeAt;
  return {
    packed,
    iters: n,
    opCount,
    ops,
    addC: addGate,
    julia,
    jc,
    T,
    Tfx,
    finalZ,
    finalZfx: [s.x, s.y, s.z], // exact end frame — the test oracle's anchor
    escapeAt,
  };
}

// ── the exact truth (test oracle; also the future CI structure pin) ──────
// Runs the SAME map at full precision for the sample point T+δ0, mirroring
// makeOrbit's loop semantics (addC||julia gate, bailout after the add).
// Returns the exact end state as fx BigInts plus the escape iteration.
export function truthRun(formula, target, d0, iters) {
  const ops = prepOps(formula);
  const julia = !!formula.julia;
  const jc = julia ? formula.juliaC || [0, 0, 0] : null;
  const addGate = !!formula.addC || julia;
  const Tfx = typeof target[0] === "bigint" ? target : targetToFx(target);
  const s = {
    x: Tfx[0] + fxFromF64(d0[0]),
    y: Tfx[1] + fxFromF64(d0[1]),
    z: Tfx[2] + fxFromF64(d0[2]),
    w: FXONE,
  };
  const c = julia ? targetToFx(jc) : [s.x, s.y, s.z];
  const bail = fxFromF64(BAILOUT_IFS);
  const n = Math.max(1, Math.floor(iters ?? formula.iters ?? 8));
  let escapedAt = -1;
  for (let i = 0; i < n; i++) {
    for (const op of ops) fxStep(op, s);
    if (addGate) {
      s.x += c[0];
      s.y += c[1];
      s.z += c[2];
    }
    if (fxMul(s.x, s.x) + fxMul(s.y, s.y) + fxMul(s.z, s.z) > bail) {
      escapedAt = i;
      break;
    }
  }
  return { x: s.x, y: s.y, z: s.z, w: s.w, escapedAt };
}

// ── the f32 delta kernel (fround-mirrored — the truth for PR-2's WGSL) ───
// Every arithmetic result passes through Math.fround, so this IS the f32
// computation the GPU runs (plain arithmetic — no EFT identities for a
// compiler to simplify, so the mirror is faithful; the df64 self-agreement
// trap does not apply).

function deltaOp(c, rec, off, d, wRef) {
  const p = rec;
  switch (c.key) {
    case "boxFold":
    case "boxFoldXYZ":
    case "surfFold": {
      const axes = c.key === "surfFold" ? 2 : 3;
      for (let a = 0; a < axes; a++) {
        const f = c.key === "boxFoldXYZ" ? c.v[a] : c.v[0];
        const m1 = p[off + 4 + a],
          m2 = p[off + 8 + a],
          dx = d[a];
        const refB = m1 < 0 ? 2 : m2 < 0 ? 0 : 1; // 0 low, 1 mid, 2 up
        const sB = dx > m1 ? 2 : dx < -m2 ? 0 : 1;
        if (refB === 1 && sB === 1) d[a] = dx;
        else if (refB === sB) d[a] = -dx;
        else if (refB === 1 && sB === 2) d[a] = F(F(2 * m1) - dx);
        else if (refB === 2 && sB === 1) d[a] = F(dx - F(2 * m1));
        else if (refB === 1 && sB === 0) d[a] = F(-F(2 * m2) - dx);
        else if (refB === 0 && sB === 1) d[a] = F(dx + F(2 * m2));
        else if (refB === 2 && sB === 0) d[a] = F(-F(4 * f) - dx);
        else d[a] = F(F(4 * f) - dx); // low → up
      }
      return wRef;
    }
    case "absFold":
    case "absXYZ": {
      for (let a = 0; a < 3; a++) {
        if (c.key === "absXYZ" && !(c.v[a] > 0.5)) continue;
        const Zx = p[off + a],
          dx = d[a];
        // abs = a fold at 0: the margin IS the stored Z component (a
        // near-zero f32 keeps full relative precision — no extra aux)
        const refUp = Zx >= 0;
        const sPos = dx >= -Zx; // sign of Z+δ via the exact margin compare
        if (refUp && sPos) d[a] = dx;
        else if (!refUp && !sPos) d[a] = -dx;
        else if (refUp) d[a] = F(-F(2 * Zx) - dx); // ref +, sample −
        else d[a] = F(F(2 * Zx) + dx); // ref −, sample +
      }
      return wRef;
    }
    case "sphereFold":
    case "cylinderFold": {
      const xy = c.key === "cylinderFold";
      const Z0 = p[off],
        Z1 = p[off + 1],
        Z2 = p[off + 2];
      const rr2 = p[off + 7],
        mr1 = p[off + 11],
        mr2 = p[off + 12],
        kr0 = p[off + 3];
      const dotZ = xy
        ? F(F(Z0 * d[0]) + F(Z1 * d[1]))
        : F(F(F(Z0 * d[0]) + F(Z1 * d[1])) + F(Z2 * d[2]));
      const dotD = xy
        ? F(F(d[0] * d[0]) + F(d[1] * d[1]))
        : F(F(F(d[0] * d[0]) + F(d[1] * d[1])) + F(d[2] * d[2]));
      const q = F(F(2 * dotZ) + dotD);
      const rs2 = F(rr2 + q);
      const refB = mr1 < 0 ? 0 : mr2 <= 0 ? 2 : 1; // 0 low, 1 mid, 2 high
      const sB = F(mr1 + q) < 0 ? 0 : F(mr2 - q) <= 0 ? 2 : 1;
      let kr = kr0,
        dk = 0;
      if (refB === sB) {
        if (refB === 1) dk = F(-F(c.fixedR2 * q) / F(rs2 * rr2));
      } else if (refB === 0 && sB === 1) dk = F(-F(c.fixedR2 * F(mr1 + q)) / F(rs2 * F(c.minR2)));
      else if (refB === 1 && sB === 0) dk = F(F(c.fixedR2 * mr1) / F(F(c.minR2) * rr2));
      else if (refB === 1 && sB === 2) dk = F(-mr2 / rr2);
      else if (refB === 2 && sB === 1) dk = F(F(mr2 - q) / rs2);
      else if (refB === 0 && sB === 2) dk = F(1 - c.K0);
      else dk = F(c.K0 - 1); // high → low
      const ks = F(kr + dk);
      for (let a = 0; a < 3; a++) {
        const S = F(p[off + a] + d[a]);
        d[a] = F(F(kr * d[a]) + F(dk * S));
      }
      return F(wRef * ks);
    }
    case "sphereInv":
    case "radialInvert": {
      const num = c.key === "sphereInv" ? F(c.r2p) : 1;
      const KC = F(num / 1e-6);
      const R2 = p[off + 7],
        mc = p[off + 11],
        kr0 = p[off + 3];
      const q = F(
        F(2 * F(F(F(p[off + 4] * d[0]) + F(p[off + 5] * d[1])) + F(p[off + 6] * d[2]))) +
          F(F(F(d[0] * d[0]) + F(d[1] * d[1])) + F(d[2] * d[2])),
      );
      const dds = F(R2 + q);
      const refCl = mc < 0,
        sCl = F(mc + q) < 0;
      let kr = kr0,
        dk = 0;
      if (!refCl && !sCl) dk = F(-F(num * q) / F(dds * R2));
      else if (refCl && sCl) kr = KC;
      else if (!refCl && sCl) dk = F(F(num * mc) / F(1e-6 * R2));
      else {
        kr = KC;
        dk = F(-F(num * F(mc + q)) / F(1e-6 * dds));
      }
      const ks = F(kr + dk);
      for (let a = 0; a < 3; a++) {
        const Ds = F(p[off + 4 + a] + d[a]);
        d[a] = F(F(kr * d[a]) + F(dk * Ds));
      }
      return F(wRef * ks);
    }
    case "scale": {
      const k = F(c.v[0]);
      for (let a = 0; a < 3; a++) d[a] = F(k * d[a]);
      return F(wRef * Math.abs(k));
    }
    case "translate":
      return wRef;
    case "rotateXY":
      rotD(d, 0, 1, c.rot[0]);
      return wRef;
    case "rotateYZ":
      rotD(d, 1, 2, c.rot[0]);
      return wRef;
    case "rotateXZ":
      rotD(d, 0, 2, c.rot[0]);
      return wRef;
    case "rotateXYZ":
      rotD(d, 0, 1, c.rot[0]);
      rotD(d, 1, 2, c.rot[1]);
      rotD(d, 0, 2, c.rot[2]);
      return wRef;
    default:
      throw new Error("perturb deltaOp: " + c.key);
  }
}

function rotD(d, a0, a1, rc) {
  const co = F(rc[0]),
    sn = F(rc[1]);
  const n0 = F(F(d[a0] * co) - F(d[a1] * sn));
  const n1 = F(F(d[a0] * sn) + F(d[a1] * co));
  d[a0] = n0;
  d[a1] = n1;
}

// plain-f32 op bodies (branchless, production-shaped) — the switched tail
function f32Op(c, s) {
  switch (c.key) {
    case "boxFold":
    case "boxFoldXYZ":
    case "surfFold": {
      const axes = c.key === "surfFold" ? 2 : 3;
      const keys = ["x", "y", "z"];
      for (let i = 0; i < axes; i++) {
        const f = F(c.key === "boxFoldXYZ" ? c.v[i] : c.v[0]);
        const ax = keys[i];
        s[ax] = F(F(Math.abs(F(s[ax] + f)) - Math.abs(F(s[ax] - f))) - s[ax]);
      }
      break;
    }
    case "absFold":
      s.x = Math.abs(s.x);
      s.y = Math.abs(s.y);
      s.z = Math.abs(s.z);
      break;
    case "absXYZ":
      if (c.v[0] > 0.5) s.x = Math.abs(s.x);
      if (c.v[1] > 0.5) s.y = Math.abs(s.y);
      if (c.v[2] > 0.5) s.z = Math.abs(s.z);
      break;
    case "sphereFold":
    case "cylinderFold": {
      const xy = c.key === "cylinderFold";
      const minR2 = F(c.minR2),
        fixedR2 = F(c.fixedR2);
      const r2 = xy
        ? F(F(s.x * s.x) + F(s.y * s.y))
        : F(F(F(s.x * s.x) + F(s.y * s.y)) + F(s.z * s.z));
      const k = r2 < minR2 ? F(fixedR2 / minR2) : r2 < fixedR2 ? F(fixedR2 / r2) : 1;
      s.x = F(s.x * k);
      s.y = F(s.y * k);
      s.z = F(s.z * k);
      s.w = F(s.w * k);
      break;
    }
    case "sphereInv": {
      const d = Math.max(F(F(F(s.x * s.x) + F(s.y * s.y)) + F(s.z * s.z)), 1e-6);
      const k = F(F(c.r2p) / d);
      s.x = F(s.x * k);
      s.y = F(s.y * k);
      s.z = F(s.z * k);
      s.w = F(s.w * k);
      break;
    }
    case "radialInvert": {
      const dx = F(s.x - F(c.v[0])),
        dy = F(s.y - F(c.v[1])),
        dz = F(s.z - F(c.v[2]));
      const dd = Math.max(F(F(F(dx * dx) + F(dy * dy)) + F(dz * dz)), 1e-6);
      const k = F(1 / dd);
      s.x = F(F(dx * k) + F(c.v[0]));
      s.y = F(F(dy * k) + F(c.v[1]));
      s.z = F(F(dz * k) + F(c.v[2]));
      s.w = F(s.w * k);
      break;
    }
    case "scale": {
      const k = F(c.v[0]);
      s.x = F(s.x * k);
      s.y = F(s.y * k);
      s.z = F(s.z * k);
      s.w = F(s.w * Math.abs(k));
      break;
    }
    case "translate":
      s.x = F(s.x + F(c.v[0]));
      s.y = F(s.y + F(c.v[1]));
      s.z = F(s.z + F(c.v[2]));
      break;
    case "rotateXY":
      rotS(s, "x", "y", c.rot[0]);
      break;
    case "rotateYZ":
      rotS(s, "y", "z", c.rot[0]);
      break;
    case "rotateXZ":
      rotS(s, "x", "z", c.rot[0]);
      break;
    case "rotateXYZ":
      rotS(s, "x", "y", c.rot[0]);
      rotS(s, "y", "z", c.rot[1]);
      rotS(s, "x", "z", c.rot[2]);
      break;
    default:
      throw new Error("perturb f32Op: " + c.key);
  }
}

function rotS(s, a0, a1, rc) {
  const co = F(rc[0]),
    sn = F(rc[1]);
  const n0 = F(F(s[a0] * co) - F(s[a1] * sn));
  const n1 = F(F(s[a0] * sn) + F(s[a1] * co));
  s[a0] = n0;
  s[a1] = n1;
}

// Full delta run. d0 = the residual sample point (p − T, f32-scale).
// Returns { d, pos, w, switched, viaZ }:
//   viaZ=true  — the delta tracked to the end (or a sample bailout): `d`
//                is the residual IN THE REFERENCE FRAME (the precise
//                result — a tiny δ absorbs into nothing when added to the
//                O(1) frame, so consumers needing δ must read `d`, never
//                pos − finalZ); `pos` = finalZ + d is provided for O(1)
//                consumers (bailout radius, DE) where f32-relative is the
//                need.
//   viaZ=false — a τ/escape switchover ran the plain-f32 tail: `pos` is
//                the tail's state, `d` is null (the sample is
//                macroscopically separated; no small residual exists).
// `onIter(pos3, i)` — optional per-iteration observer, called after the
// c-add with the reconstructed sample position (Z_next + δ while tracking,
// the f32 tail state after a switchover) — the same value the GPU sites'
// per-iteration pos-sync sees. Powers the orbit-signal mirrors (pt-opdiff
// SIGNALS mode) and the D8 probe's future signal needs. Called BEFORE the
// bailout break, exactly like the sites read their signals.
export function deltaRun(orbit, d0, onIter = null) {
  const { packed: p, iters, opCount, ops, addC, jc, julia, T, escapeAt } = orbit;
  const d = [F(d0[0]), F(d0[1]), F(d0[2])];
  const dc = [d[0], d[1], d[2]];
  let w = 1;
  let off = 0;
  let switched = -1;
  const s = { x: 0, y: 0, z: 0, w: 1 };
  // the switched tail's c: the f32 sample point (Julia: the constant)
  const cT = julia
    ? [F(jc[0]), F(jc[1]), F(jc[2])]
    : [F(F(T[0]) + dc[0]), F(F(T[1]) + dc[1]), F(F(T[2]) + dc[2])];
  for (let i = 0; i < iters; i++) {
    for (let o = 0; o < opCount; o++) {
      if (switched < 0) {
        const Z0 = p[off],
          Z1 = p[off + 1],
          Z2 = p[off + 2];
        const zm = Math.max(1, Z0 * Z0 + Z1 * Z1 + Z2 * Z2);
        const dm = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        if (i >= escapeAt || dm > PT_TAU * PT_TAU * zm) {
          s.x = F(Z0 + d[0]);
          s.y = F(Z1 + d[1]);
          s.z = F(Z2 + d[2]);
          s.w = w;
          switched = i;
        }
      }
      if (switched >= 0) f32Op(ops[o], s);
      else w = deltaOp(ops[o], p, off, d, w);
      off += PT_SLOT_F32;
    }
    if (addC) {
      if (switched >= 0) {
        s.x = F(s.x + cT[0]);
        s.y = F(s.y + cT[1]);
        s.z = F(s.z + cT[2]);
      } else if (!julia) {
        d[0] = F(d[0] + dc[0]);
        d[1] = F(d[1] + dc[1]);
        d[2] = F(d[2] + dc[2]);
      }
      // Julia while tracking: δc = 0 — the constant cancels exactly.
    }
    // sample bailout, mirroring makeOrbit: after the add, on the
    // reconstructed position (Z entering the next iteration + δ)
    if (switched >= 0) {
      if (onIter) onIter([s.x, s.y, s.z], i);
      if (s.x * s.x + s.y * s.y + s.z * s.z > BAILOUT_IFS)
        return { d: null, pos: [s.x, s.y, s.z], w: s.w, switched, viaZ: false };
    } else {
      const zx = p[off],
        zy = p[off + 1],
        zz = p[off + 2]; // slot (i+1, 0) or the trailer
      const px = zx + d[0],
        py = zy + d[1],
        pz = zz + d[2];
      if (onIter) onIter([px, py, pz], i);
      if (px * px + py * py + pz * pz > BAILOUT_IFS)
        return { d: [d[0], d[1], d[2]], pos: [px, py, pz], w, switched: -1, viaZ: true };
    }
  }
  if (switched >= 0)
    return { d: null, pos: [s.x, s.y, s.z], w: s.w, switched, viaZ: false };
  const off0 = iters * opCount * PT_SLOT_F32;
  return {
    d: [d[0], d[1], d[2]],
    pos: [p[off0] + d[0], p[off0 + 1] + d[1], p[off0 + 2] + d[2]],
    w,
    switched: -1,
    viaZ: true,
  };
}

// The analytic IFS DE r/|w| at a residual point — the deep zoom-to-surface
// probe's evaluator (plan D8). Valid for deOption-2 formulas (the same line
// ptEligible draws; escape-time DEs are out of v1).
export function deltaDE(orbit, d0) {
  const r = deltaRun(orbit, d0);
  return (
    Math.hypot(r.pos[0], r.pos[1], r.pos[2]) / Math.max(Math.abs(r.w), 1e-9)
  );
}

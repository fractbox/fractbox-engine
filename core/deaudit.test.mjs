// #636 — Catoptron3D-parity Gate 2: the numeric DE validation audit.
//
// A sampled Lipschitz / operator-norm audit of EVERY registered operator,
// run against the CPU twins (cpuorbit.js applyOp — the testable leg of the
// 3-emitter WGSL/GLSL/CPU mirror). Generalizes the proven per-op gates in
// mirrorvortex.test.mjs (seeded finite-difference pairs + exact-attainment
// tightness + fundamental-domain checks) and mandalay.test.mjs (the original
// non-expansion gate) to the whole OPERATORS registry, registry-driven: a new
// op is picked up automatically and audited under its wRule's default
// contract; the AUDIT table below only carries overrides, each justified.
//
// ── Protocol (resolving the #636 spec-review blockers) ────────────────────
// THREE DE REGIMES, three contracts (blocker 1 — "op" is not homogeneous):
//   exact   W_UNCHANGED / W_MUL_K / W_MUL_SCALE, no deApprox: the declared
//           per-point w-factor is a TRUE local Lipschitz bound. Gate: sampled
//           operator norm ≤ declared·(1+tol) over seam-free pairs, AND
//           attained (worst ≥ 0.98·declared — a too-LOOSE declaration fails
//           too: the spiralVortex-constant lesson). W_UNCHANGED must leave w
//           byte-untouched; W_MUL_K/W_MUL_SCALE w must be linear in w_in.
//           An exact-tier op MAY also carry a `ceil` annotation — the hand
//           derivation of its norm, written independently of the shipped
//           code. Where present it is gated too, so the op cannot drift into
//           self-consistent nonsense (implementation and declaration are the
//           SAME number for W_MUL_K: only a third, independent copy of the
//           formula can catch them moving together). #643 left varyScale one.
//   approx  deApprox-tagged ops: the RAW bound is loose BY DESIGN — the gap
//           is absorbed at the policy layer (APPROX_DESCALE_MUL = 0.5,
//           APPROX_DE.md §3; the f9226cd gnarl stance). This audit therefore
//           gates the raw bound only where a ceiling is DERIVABLE
//           (approx-derived: measured ≤ analytic param-formula, + tightness)
//           and otherwise pins seeded percentile magnitudes (approx-report)
//           so drift in any emitter twin surfaces loudly. Pins are DRIFT
//           PINS, not soundness bounds (blocker 3 resolved: raw is audited,
//           policy absorbs the documented gap).
//   bulb    W_BULB: no static declared factor exists (the DE is the orbit
//           recurrence w ← G·w + 1, G read back per-point as w_out−1).
//           Gate: the recurrence must be exactly affine in w (G from w=1 and
//           w=2 agree; position independent of w), plus a seeded p99 pin of
//           ratio/(G+1) — the +1 is the recurrence's own dr floor, which
//           keeps the small-G region honest. Escape-time DE additionally has
//           the 0.5·r·ln(r) factor and its own step policy; these pins are
//           regression teeth, not proofs (blocker 1: "differently-shaped
//           check" for this regime).
//   numeric W_BULB_NUMERIC: opts out of w entirely (isNumericDE) — the DE is
//           finite-difference at march time, so NO norm contract exists to
//           audit. What IS auditable: applyOp must leave w byte-untouched
//           and positions must not read w (the #418 twin-divergence class),
//           plus a pinned non-finite sample rate.
//
// UNBOUNDED OPS (blocker 2): logWarp's crease is floored (+0.01 ⇒ derivative
// ≤ |Mul·Base|/(0.01·ln2) — large but finite, gated); neoSqrWarp's quadratic
// derivative is unbounded globally but the ceiling is a per-point FIELD
// (|Mul·(FixSq−2t)|), so every sampled pair is gated against its own local
// bound — no radius exemption needed. Genuinely unbounded-at-a-set ops
// (toCoord/fromCoord atan2 poles, complexMap poles, torusInvert axis,
// smoothBallFold blend center) are approx-report: percentile pins, with the
// singular set named in the table note. Nothing is skipped silently.
//
// SEAMS: piecewise-isometric ops with measure-zero discontinuities (cell
// wraps, sector snaps, conditional shifts) are 1-Lipschitz within each cell
// but a straddling pair measures the jump. The #632-review lesson: Lipschitz
// sampling alone cannot validate a fold — so for each seam op the excluded
// pairs are compensated by a FUNDAMENTAL-DOMAIN gate (output provably lands
// in the declared cell) in the dedicated tests below the audit loop.
//
// SAMPLING (spec review "protocol unspecified"): seeded mulberry32 (per-op
// seed from the op key, deterministic in CI); params = registry defaults +
// 6 uniform draws per param in [min,max] (step ≥ 1 snapped to the grid the
// UI can produce); positions log-uniform in radius — IFS ops r ∈ [10^-1.5,
// 30] (fold cells live at O(1), far field included: the Catoptron3D bug
// class), escape-time ops r ∈ [10^-1.5, 2.5] (the escape surface region,
// BAILOUT_ESCAPE = 64 on r²); pair separation ε = min(1e-4, r·1e-3) (stay
// local at tiny radii — a fixed step under a winding map measures a chord,
// mirrorvortex lesson), random independent direction.
//
// NON-GOALS: Gate 1 (shader-variant compile matrix — needs a GPU/naga in
// CI) and Gate 3 (marched conservativeness) are out of scope for #636.
// core/leaves.js D2 shape leaves (SDF math, own deApprox lane) are a
// follow-up, not audited here. stability.lambdaHat is the OPPOSITE bound
// direction (guaranteed MINIMUM |J| for the df64 law) and closed-form —
// machinery intentionally not shared.
//
// Named *.test.mjs so it stays out of the apps' served *.js surface.
// Run: node --test core/deaudit.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { applyOp } from "./cpuorbit.js";
import {
  OPERATORS,
  byKey,
  W_UNCHANGED,
  W_MUL_K,
  W_MUL_SCALE,
  W_BULB,
  W_BULB_NUMERIC,
} from "./operators.js";

// ── engine ────────────────────────────────────────────────────────────────
function mulberry32(a) {
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const seedOf = (key) => {
  let h = 0x811c9dc5;
  for (let i = 0; i < key.length; i++)
    h = Math.imul(h ^ key.charCodeAt(i), 0x01000193);
  return h >>> 0;
};
const rad = (d) => (d * Math.PI) / 180;
const L_SHEAR = (b) => (Math.abs(b) + Math.sqrt(b * b + 4)) / 2;

const runOp = (key, v, p, w, i = 0) => {
  const s = { x: p[0], y: p[1], z: p[2], w, i };
  applyOp(key, v, s);
  return s;
};

// Registry defaults + n seeded draws in [min,max]; step ≥ 1 snapped to the
// integer grid (selectors/enums — the only values the UI can produce).
function paramSets(def, rand, n = 6) {
  const sets = [def.params.map((p) => p.default)];
  for (let i = 0; i < n; i++)
    sets.push(
      def.params.map((p) => {
        let val = p.min + rand() * (p.max - p.min);
        if (p.step && p.step >= 1) val = Math.round(val);
        return val;
      }),
    );
  return sets;
}

const isBulbRule = (r) => r === W_BULB || r === W_BULB_NUMERIC;

// One measurement pass over an op. Returns seeded, deterministic stats.
function measure(def, ann) {
  const rand = mulberry32(seedOf(def.key));
  const rmax = isBulbRule(def.wRule) ? 2.5 : 30;
  const logSpan = Math.log10(rmax) + 1.5;
  const NPAIR = 2200;
  const rels = []; // ratio / declared (the DE-looseness measure)
  let worstRel = 0;
  let worstCeil = 0; // ratio / derived ceiling field (ann.ceil only)
  let nonfinite = 0,
    total = 0,
    seamSkips = 0,
    wViol = null;
  for (const v of paramSets(def, rand)) {
    for (let n = 0; n < NPAIR; n++) {
      total++;
      const r = 10 ** (rand() * logSpan - 1.5);
      const dir = [rand() - 0.5, rand() - 0.5, rand() - 0.5];
      const dn = Math.hypot(...dir) || 1;
      const p = dir.map((x) => (x / dn) * r);
      const eps = Math.min(1e-4, r * 1e-3) || 1e-7;
      const du = [rand() - 0.5, rand() - 0.5, rand() - 0.5];
      const un = Math.hypot(...du) || 1;
      const q = p.map((x, j) => x + (du[j] / un) * eps);
      const it = def.key === "scaleDrift" ? Math.floor(rand() * 8) : 0;
      const a = runOp(def.key, v, p, 1, it);
      const b = runOp(def.key, v, q, 1, it);
      if (![a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w].every(Number.isFinite)) {
        nonfinite++;
        continue;
      }
      // w-rule conformance — byte-exact where the rule says untouched, and
      // linear/affine in w_in (checked on a stride so the pass stays cheap).
      if (!wViol) {
        if (def.wRule === W_UNCHANGED && a.w !== 1)
          wViol = `W_UNCHANGED but w ${a.w} at ${JSON.stringify({ v, p })}`;
        if (def.wRule === W_BULB_NUMERIC && a.w !== 1)
          wViol = `W_BULB_NUMERIC must ignore w, got ${a.w}`;
        if (n % 16 === 0) {
          const a2 = runOp(def.key, v, p, 2, it);
          if (a2.x !== a.x || a2.y !== a.y || a2.z !== a.z)
            wViol = `position depends on w_in at ${JSON.stringify({ v, p })}`;
          else if (def.wRule === W_UNCHANGED || def.wRule === W_BULB_NUMERIC) {
            if (a2.w !== 2) wViol = `w not left untouched (w_in=2 → ${a2.w})`;
          } else if (def.wRule === W_MUL_K || def.wRule === W_MUL_SCALE) {
            if (Math.abs(a2.w - 2 * a.w) > 1e-12 * Math.abs(a2.w) + 1e-300)
              wViol = `w not linear in w_in: ${a.w} vs ${a2.w}`;
            else if (!(a.w > 0)) wViol = `w factor not positive: ${a.w}`;
          } else if (def.wRule === W_BULB) {
            // affine w ← G·w + 1 (G position-only, ≥ 0)
            const G = a.w - 1;
            if (G < -1e-12) wViol = `bulb G negative: ${G}`;
            else if (Math.abs(a2.w - (2 * G + 1)) > 1e-9 * (1 + Math.abs(a2.w)))
              wViol = `bulb w not affine: w1=${a.w} w2=${a2.w}`;
          }
        }
      }
      if (ann.seam && ann.seam(v, p, q)) {
        seamSkips++;
        continue;
      }
      const ratio = Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z) / eps;
      let declared;
      if (def.wRule === W_UNCHANGED) declared = 1;
      else if (def.wRule === W_MUL_K || def.wRule === W_MUL_SCALE)
        declared = Math.max(Math.abs(a.w), Math.abs(b.w), 1e-12);
      else if (def.wRule === W_BULB)
        declared = Math.max(a.w, b.w); // = G+1 ≥ 1, the dr-recurrence floor
      else declared = NaN; // numeric: no norm contract (see tier notes)
      if (Number.isFinite(declared)) {
        const rel = ratio / declared;
        rels.push(rel);
        if (rel > worstRel) worstRel = rel;
      }
      if (ann.ceil) {
        const c = Math.max(ann.ceil(v, p), ann.ceil(v, q), 1e-12);
        const rc = ratio / c;
        if (rc > worstCeil) worstCeil = rc;
      }
    }
  }
  rels.sort((x, y) => x - y);
  const p99 = rels.length ? rels[Math.floor(rels.length * 0.99)] : 0;
  return {
    worstRel,
    p99,
    worstCeil,
    nanRate: nonfinite / total,
    seamSkips,
    wViol,
  };
}

// ── the audit table — OVERRIDES only; every entry justified ───────────────
// Default (no entry): exact-tier contract for the op's wRule (see header).
// tier values: "approx-derived" | "approx-report" | "breach-pinned"
// (bulb/numeric tiers come from the wRule; deApprox ops default to
// "approx-report" unless a derived ceiling is given here).
const AUDIT = {
  // ── exact tier, seam predicates (fundamental-domain gates compensate) ──
  kaleido: {
    // Mirror OFF (p2 < 0.5) is a rotation-only sector snap: an isometry per
    // sector, discontinuous at sector boundaries. Mirror ON reflects — the
    // tent is globally continuous, no seam. Integer Symmetry (step 1) keeps
    // the atan2 cut seamless in both modes.
    seam: (v, p, q) => {
      if ((v[2] ?? 1) > 0.5) return false;
      const w = (2 * Math.PI) / Math.max(v[0], 2);
      const s = (pt) => Math.floor(Math.atan2(pt[1], pt[0]) / w + 0.5);
      return s(p) !== s(q);
    },
  },
  polyAngleFold: {
    // Same structure as kaleido with an offset-shifted sector lattice.
    seam: (v, p, q) => {
      if (v[2] > 0.5) return false;
      const w = (2 * Math.PI) / Math.max(v[0], 2);
      const off = rad(v[1]);
      const s = (pt) => Math.floor((Math.atan2(pt[1], pt[0]) - off) / w + 0.5);
      return s(p) !== s(q);
    },
  },
  modFold: {
    // Per-axis lattice wrap: a translation per cell (isometry), discontinuous
    // at half-cell boundaries. Domain gate: |out| ≤ cell/2.
    seam: (v, p, q) => {
      for (let i = 0; i < 3; i++)
        if (
          v[i] > 0 &&
          Math.floor(p[i] / v[i] + 0.5) !== Math.floor(q[i] / v[i] + 0.5)
        )
          return true;
      return false;
    },
  },
  brickFold: {
    // Row wrap + per-row phase-shifted column wrap; translation per brick.
    seam: (v, p, q) => {
      const row = (pt) => (v[1] > 0 ? Math.floor(pt[1] / v[1] + 0.5) : 0);
      const rp = row(p),
        rq = row(q);
      if (rp !== rq) return true;
      if (v[0] > 0) {
        const col = (pt) => Math.floor((pt[0] + v[2] * rp) / v[0] + 0.5);
        if (col(p) !== col(q)) return true;
      }
      return false;
    },
  },
  zFold: {
    // Conditional shift: identity vs translate-by-Shift, a jump across the
    // z = Threshold plane. Closed form pinned in its own gate below.
    seam: (v, p, q) => p[2] > v[0] !== q[2] > v[0],
  },
  hingeFold: {
    // #633 — the first op that is a seam declarer BY CONTRACT (operators.js
    // `seam: true` + the seam-clamped DE contract above id 67): a rigid
    // rotation of the half-space past the cut plane, torn AT the plane. Per
    // cell it is an isometry — exactly what the exact tier gates over
    // non-straddling pairs — and the straddling pairs this predicate
    // excludes are not "unvalidated": the op reports its cut distance into
    // the seam channel and the marcher clamps its step by it, so a ray can
    // never cross the tear (proven end-to-end against hand-computed ground
    // truth in core/seamclamp.test.mjs, alongside the fundamental-domain
    // gate — the rotating side lands rotated by EXACTLY FoldAngle about the
    // hinge line). A compact domain gate also lives below with the others.
    // v = [FoldAngle°, CutAngle°, Axis, Offset]; the predicate mirrors the
    // op's own branch test: signed cut-plane distance changes sign.
    seam: (v, p, q) => {
      const ax = Math.round(v[2] ?? 1);
      const uv = (pt) =>
        ax === 0 ? [pt[1], pt[2]] : ax === 2 ? [pt[0], pt[1]] : [pt[0], pt[2]];
      const mx = -Math.sin(rad(v[1] ?? 0)),
        my = Math.cos(rad(v[1] ?? 0));
      const side = (pt) => {
        const [u, w2] = uv(pt);
        return u * mx + w2 * my - (v[3] ?? 0) > 0;
      };
      return side(p) !== side(q);
    },
  },

  // ── #643 — the three ex-breach ops, resolved two different ways ─────────
  // #636 found all three DECLARING a factor their true Jacobian breached
  // (twist 76.3×, cylinderFold 21.8×, varyScale 4.5×) and parked them in a
  // `breach-pinned` tier whose gates were built to FAIL the moment an op was
  // fixed. #643 fixed them, and the split is the finding worth keeping:
  //
  //   varyScale  → EXACT. Its residual is a BOUNDED scalar, max(1, |1−2γ|)
  //                ≤ 5 over the shipped RPower range, and only in the mid
  //                band. It charges it (w *= k·max(1,|1−2γ|)) and audits in
  //                the exact tier — a genuine soundness fix, and the render
  //                visibly improves (the RPower-3 corner went from
  //                over-stepped noise to resolved surface).
  //   twist,     → deApprox (APPROX_DE.md §1). Both true norms are UNBOUNDED
  //   cylinder-    fields — L(|a|·ρ) in the XY radius, k·L(2|z|/ρ) in the
  //   Fold         height ratio — so the exact charge compounds ~10⁴ over a
  //                12-iteration stack and the DE collapses to ~1e-8. That was
  //                implemented and rendered before this decision: every ray
  //                exhausted its budget mid-air and the frame came back as
  //                uniform fog. A sound-but-unusable bound is worse than a
  //                loose one that the step policy tightens, which is exactly
  //                the trade APPROX_DE.md was written for. They keep their
  //                best-effort w-rule (untouched / k), lose isDeSound()'s
  //                vouch, and gain the ×0.5 step + ×2 budget.
  //
  // All three keep the hand-derived `ceil` — for the two approx ops it is the
  // approx-derived tier's proof-of-analysis, and for varyScale it is an
  // INDEPENDENT copy of the formula its w now charges (for a W_MUL_K op the
  // "declared" factor IS whatever applyOp wrote, so worstRel alone can only
  // prove self-consistency; the exact tier gates `ceil` too when present).
  twist: {
    tier: "approx-derived",
    // Rotate XY by angle ∝ z: a z-shear of magnitude |a|·ρ in disguise —
    // true norm L(|a|·ρ), which grows with the XY radius without bound (the
    // spiralVortex contrast: constant L(a) there, a field here).
    ceil: (v, p) => L_SHEAR(rad(Math.abs(v[0])) * Math.hypot(p[0], p[1])),
    attain: 0.8,
  },
  cylinderFold: {
    tier: "approx-derived",
    // The WHOLE vector (z included) is scaled by k(ρ_xy), so unlike
    // sphereFold the mid band is not conformal: the ρ–z block of J/k is
    // [[-1,0],[-c,1]], c = 2|z|/ρ, giving k·L(2|z|/ρ), unbounded in |z|/ρ.
    // (sphereFold measures r in 3D, which is what makes ITS mid band a
    // genuine conformal inversion — it stays exact-tier on a plain k.)
    ceil: (v, p) => {
      const minR2 = v[0] * v[0],
        fixedR2 = v[1] * v[1],
        rho2 = p[0] * p[0] + p[1] * p[1];
      if (rho2 < minR2) return fixedR2 / minR2;
      if (rho2 < fixedR2) {
        const rho = Math.sqrt(rho2);
        return (fixedR2 / rho2) * L_SHEAR((2 * Math.abs(p[2])) / rho);
      }
      return 1;
    },
    attain: 0.6, // the worst direction mixes ρ̂ and ẑ — random pairs get close
  },
  varyScale: {
    // f = k(r)·p with k = F²/r^{2γ} in the mid band: radial derivative is
    // (1−2γ)·k, so for RPower γ > 1 the true norm is (2γ−1)·k (measured
    // 3.274 at γ = 2.139 vs theory 3.278 — exact). γ ≤ 1 was always sound
    // and max(1, |1−2γ|) leaves that half of the range bit-identical.
    ceil: (v, p) => {
      const minR2 = v[0] * v[0],
        fixedR2 = v[1] * v[1],
        g = v[2];
      const rp = Math.max(p[0] * p[0] + p[1] * p[1] + p[2] * p[2], 1e-12) ** g;
      if (rp < minR2) return fixedR2 / minR2;
      if (rp < fixedR2)
        return (fixedR2 / rp) * Math.max(1, Math.abs(1 - 2 * g));
      return 1;
    },
    attain: 0.9,
  },

  // ── approx-derived: deApprox ops with an analytic per-point ceiling ─────
  // The gate proves the CPU twin implements exactly the map the derivation
  // describes (ceiling holds AND is approached); worstRel is the documented
  // DE-looseness the ×0.5 policy step absorbs.
  sinShear: {
    tier: "approx-derived",
    // Cross-axis shear x_dst += Amp·sin(Freq·x_src): shear magnitude
    // |Amp·Freq·cos| ≤ |Amp·Freq|, norm = L(|Amp·Freq|) (largest singular
    // value of the 2D shear — the spiralVortex formula).
    ceil: (v) => L_SHEAR(v[1] * v[2]),
    attain: 0.9,
  },
  asinhWarp: {
    tier: "approx-derived",
    // s[k] ← Base·log2(t+√(t²+1)), t = s[k]·Mul: derivative
    // Mul·Base/(ln2·√(t²+1)) — bounded, max at t = 0 (near-sound, as
    // APPROX_DE.md predicted).
    ceil: (v, p) => {
      const m = Math.round(v[0]);
      const k = m === 0 ? 2 : m === 2 ? 0 : 1;
      const t = p[k] * v[1];
      return Math.max(
        1,
        Math.abs(v[1] * v[2]) / (Math.LN2 * Math.sqrt(t * t + 1)),
      );
    },
    attain: 0.9,
  },
  logWarp: {
    tier: "approx-derived",
    // s[k] ← Base·log2(|s[k]·Mul| + 0.01): the crease derivative is FLOORED
    // by the +0.01 — |Mul·Base|/(ln2·(|t|+0.01)) ≤ |Mul·Base|/(0.01·ln2),
    // huge (≈ 2309 at the param corners) but finite, so it gates as a field
    // (the "unbounded at the crease" comment is conservative: the shipped
    // map is Lipschitz, just enormously so).
    ceil: (v, p) => {
      const m = Math.round(v[0]);
      const k = m === 1 ? 1 : m === 2 ? 0 : 2;
      const t = p[k] * v[1];
      return Math.max(
        1,
        Math.abs(v[1] * v[2]) / (Math.LN2 * (Math.abs(t) + 0.01)),
      );
    },
    attain: 0.5, // attaining needs |t| ≈ 0 AND an axis-aligned step at once
  },
  neoSqrWarp: {
    tier: "approx-derived",
    // s[k] ← ±t(FixSq ∓ t), t = s[k]·Mul: derivative Mul·(FixSq−2t) (t ≥ 0)
    // / Mul·(2t−FixSq) (t < 0) — unbounded GLOBALLY (quadratic op) but exact
    // as a per-point field, so every pair is gated locally; no domain cap.
    ceil: (v, p) => {
      const m = Math.round(v[0]);
      const k = m === 0 ? 2 : m === 2 ? 0 : 1;
      const t = p[k] * v[2];
      const gp = t >= 0 ? v[1] - 2 * t : 2 * t - v[1];
      return Math.max(1, Math.abs(v[2] * gp));
    },
    attain: 0.8,
  },
  gnarl2D: {
    tier: "approx-derived",
    // x' = x − S·g(y), y' = y − S·g(x), g = sin(sin((sin(B·b)+b)·A)+b):
    // |g'| ≤ A(B+1)+1 by the chain rule; the off-diagonal pattern is a
    // scaled permutation, so ‖J‖ ≤ 1 + S·(A(B+1)+1).
    ceil: (v) =>
      1 + Math.abs(v[0]) * (Math.abs(v[1]) * (Math.abs(v[2]) + 1) + 1),
    attain: 0.6,
  },
  gnarl3D: {
    tier: "approx-derived",
    // Cyclic 3-lane version of gnarl2D — same permutation-pattern bound.
    ceil: (v) =>
      1 + Math.abs(v[0]) * (Math.abs(v[1]) * (Math.abs(v[2]) + 1) + 1),
    attain: 0.6,
  },
  polygonFold: {
    tier: "approx-derived",
    // Radial remap r *= f(θ) in the selected plane: J = f·I + (u,v)⊗∇f, so
    // ‖J‖ ≤ f + |f′(θ)| — computed per point from the closed form (both
    // Strength signs). w *= f tracks only the radial part; the |f′| term is
    // the documented approximation (APPROX_DE.md's first tagged op).
    ceil: (v, p) => {
      const n = Math.max(Math.round(v[0]), 3);
      const st = v[1];
      const m = Math.round(v[2]);
      const [u, vv] =
        m === 1 ? [p[2], p[0]] : m === 2 ? [p[1], p[2]] : [p[0], p[1]];
      if (u * u + vv * vv <= 1e-24) return 1;
      const sector = (2 * Math.PI) / n;
      const a = Math.atan2(vv, u);
      const th = a - sector * Math.floor(a / sector + 0.5);
      const c = Math.max(Math.cos(th), 1e-6);
      let f, fp;
      if (st >= 0) {
        f = 1 + st * (c - 1);
        fp = st * Math.abs(Math.sin(th));
      } else {
        f = 1 - st * (1 / c - 1);
        fp = (-st * Math.abs(Math.sin(th))) / (c * c);
      }
      return Math.max(1, Math.abs(f) + fp);
    },
    attain: 0.8,
  },

  // ── approx-report: deApprox ops with a genuinely unbounded/intractable
  //    Jacobian — seeded percentile DRIFT PINS + the singular set named ────
  toCoord: {
    tier: "approx-report",
    note: "angle lanes: |∂θ/∂p| ~ 1/ρ near the axis, unbounded; log-polar ln ρ unbounded at 0; atan2 branch cuts are measure-zero chords — p99 pin.",
  },
  fromCoord: {
    tier: "approx-report",
    note: "radius·angle frames: derivative ∝ |r| far out (unbounded); complex-exp lane reaches e^60; cs=2 sign flip at cos(y)=0 is a jump seam — p99 pin.",
  },
  complexMap: {
    tier: "approx-report",
    note: "Möbius/rational maps: |f'| diverges at the (floored) denominator zeros — the pole lines the id-61 derivation names; w deliberately untracked — p99 pin.",
  },
  torusInvert: {
    tier: "approx-report",
    note: "w tracks the MERIDIAN factor only (the azimuthal ρ'/ρ diverges on the axis — the flat-wall lesson, operators.js id 58); pseudo variants 1–3 scale z on entirely different laws — p99 pin.",
  },
  smoothBallFold: {
    tier: "approx-report",
    note: "w *= 1/max(|bm|,1e-20): bm crosses 0 at the blend center, so k (and the untracked ∇k term) blow up there — p99 pin.",
  },
  smoothBoxFold: {
    tier: "approx-report",
    note: "C1 rational shoulder: slope overshoots 1 near the crease, growing with Sharpness (measured ≈ 6 at S=12); closed-form ceiling is messy — p99 pin.",
  },

  // ── bulb tier notes (pins live in BULB_PINS below) ──────────────────────
  // All W_BULB ops share: angle-multiplier lanes (ThetaMul/PhiMul/YMul/
  // AziPow) and non-integer powers stretch tangentially beyond the scalar
  // recurrence G = p·r^{p−1} (up to ~p·mul near the poles/equator), and
  // non-integer powers tear at the atan2 cut (measure-zero chords) — which
  // is exactly why the escape DE is heuristic. p99 of ratio/(G+1) is pinned.

  // ── numeric tier ────────────────────────────────────────────────────────
  kleinPolyMap: {
    // (√ρ²−y)/x tan-half-angle form NaNs near the x=0 half-plane — the WGSL
    // twin is byte-identically unguarded (verified operators.js id 42), the
    // numeric-DE marcher treats NaN as a miss. Pinned so a guard added to
    // one emitter but not the others surfaces here.
    nanPin: 0.4,
  },
};

// Empirical drift pins: p99 of ratio/declared under THE seeded protocol
// above (so exactly reproducible run to run). Values = measured at landing
// (2026-08-20, noted per line, see the #636 PR table) with ~1.3–3× headroom.
// NOT soundness bounds — regression teeth for emitter/param-range drift.
const REPORT_PINS = {
  toCoord: 90, // measured p99 31.6 (worst 410 at the log-polar origin)
  fromCoord: 40, // measured p99 14.0 (worst 26 in the far field, ∝ r)
  complexMap: 25, // measured p99 8.1 (worst 532 near a pole line)
  torusInvert: 30000, // measured p99 11461 — the pseudo variants' z·(d·Rad)
  //                     lane vs a meridian-only w is THE loosest shipped pair
  smoothBallFold: 20, // measured p99 5.8 (worst 257 at the blend center)
  smoothBoxFold: 7, // measured p99 2.9 (worst 5.6, grows with Sharpness)
  logWarp: 250, // measured p99 108.6 (crease floor |Mul·Base|/(0.01·ln2))
  neoSqrWarp: 300, // measured p99 126.9 (quadratic growth, r ≤ 30 domain)
  asinhWarp: 15, // measured p99 7.3 (bounded: |Mul·Base|/ln2)
  gnarl2D: 10, // measured p99 4.9 (bounded: 1+S(A(B+1)+1))
  gnarl3D: 10, // measured p99 4.9
  sinShear: 4, // measured p99 1.8 (bounded: L(|Amp·Freq|))
  polygonFold: 2.5, // measured p99 1.16 (mildest tagged op, as APPROX_DE.md
  //                   predicted: |f|+|f'| vs the tracked |f|)
  // #643 — twist/cylinderFold gained the deApprox tag, so their old
  // breach-magnitude pins become ordinary approx-tier looseness pins (same
  // seeded numbers, now measuring a documented gap instead of an undeclared
  // one). varyScale's entry is GONE: it charges its true norm and audits
  // exact at worstRel 1.000, where a drift pin would assert only 1 ≤ 1.
  twist: 70, // measured p99 33.1 (worst 76.3 at Twist 180°, ρ ≈ 27)
  cylinderFold: 5, // measured p99 2.22 (worst 21.8, mid band, |z|/ρ ≈ 13)
};
const BULB_PINS = {
  // p99 of ratio/(G+1); measured 2026-08-20, ~2× headroom. The worst-sample
  // column in the PR table is dominated by atan2-cut chords at non-integer
  // powers (measure-zero; the escape DE is heuristic there by construction).
  mandelbulbPower: 3, // p99 1.36 — tangential sin(pθ)/sinθ ≤ p on top of G
  quadratic: 1.2, // p99 0.86 — z-lane carried (ratio 1) vs G+1 ≥ 1; XY exact
  bulbAxis: 7, // p99 3.57 — ThetaMul/PhiMul lanes untracked by G
  msltoeSym3: 4, // p99 1.72 — |YMul| ≤ 4 lane stretch + the z≥y swap seam
  sphericalTwoStage: 6, // p99 3.10 — same multiplier class as bulbAxis
  boxBulb: 3, // p99 1.21 — 4-norm frame distortion on top of the scalar G
  slonoBrot2: 2, // p99 1.13 — mild tangential overshoot of the 2r recurrence
  ruckerBulb: 3.5, // p99 1.52 — AziPow azimuth + conv-3 atan2(−x,z) cut
};
const DEFAULT_BULB_PIN = 64; // a NEW bulb op lands here until annotated
const DEFAULT_NAN_PIN = 0.005;

const tierOf = (def) => {
  const ann = AUDIT[def.key] ?? {};
  if (ann.tier) return ann.tier;
  if (def.wRule === W_BULB) return "bulb";
  if (def.wRule === W_BULB_NUMERIC) return "numeric";
  return def.deApprox ? "approx-report" : "exact";
};

// One shared measurement pass (each op measured once; tests read from here).
const RESULTS = new Map();
for (const def of OPERATORS)
  RESULTS.set(def.key, measure(def, AUDIT[def.key] ?? {}));

test("audit table hygiene: overrides/pins name real ops; tiers partition the registry", () => {
  for (const key of [
    ...Object.keys(AUDIT),
    ...Object.keys(REPORT_PINS),
    ...Object.keys(BULB_PINS),
  ])
    assert.ok(
      byKey(key),
      `audit annotation for unknown op '${key}' — renamed/removed?`,
    );
  // deApprox ⇒ approx tier; sound IFS rules default to exact; every breach-
  // pinned op must justify itself with a derived ceiling AND a breach floor.
  for (const def of OPERATORS) {
    const tier = tierOf(def);
    if (def.deApprox)
      assert.ok(
        tier === "approx-derived" || tier === "approx-report",
        `${def.key}: deApprox op must audit in an approx tier (got ${tier})`,
      );
    if (tier === "breach-pinned") {
      const ann = AUDIT[def.key];
      assert.ok(
        ann.ceil && ann.breachFloor,
        `${def.key}: breach-pinned needs ceil + breachFloor`,
      );
      assert.ok(
        !def.deApprox,
        `${def.key}: a deApprox tag would resolve the breach — move to approx tier`,
      );
    }
  }
});

test("w-rule conformance: every op's w bookkeeping matches its declared rule", () => {
  for (const def of OPERATORS) {
    const m = RESULTS.get(def.key);
    assert.equal(m.wViol, null, `${def.key}: ${m.wViol}`);
  }
});

test("exact tier: declared w-factor is a TRUE bound, and it is attained", () => {
  for (const def of OPERATORS) {
    if (tierOf(def) !== "exact") continue;
    const m = RESULTS.get(def.key);
    assert.equal(
      m.nanRate,
      0,
      `${def.key}: non-finite outputs on the sound domain`,
    );
    const tol = def.wRule === W_UNCHANGED ? 1e-7 : 5e-3;
    assert.ok(
      m.worstRel <= 1 + tol,
      `${def.key}: measured operator norm ${m.worstRel.toFixed(6)}× the declared ` +
        `w-factor — the analytic DE r/|w| is NOT a bound for this op`,
    );
    assert.ok(
      m.worstRel >= 0.98,
      `${def.key}: declared bound never approached (worst ${m.worstRel.toFixed(4)}) — ` +
        `declaration too loose (the spiralVortex-constant class) or sampler blind`,
    );
    // …and where the op ALSO carries a hand-derived ceiling (#643 gave
    // varyScale one), the shipped charge must agree with the maths, not
    // merely with itself.
    const ann = AUDIT[def.key];
    if (ann?.ceil) {
      assert.ok(
        m.worstCeil <= 1.02,
        `${def.key}: measured ${m.worstCeil.toFixed(4)}× the DERIVED ceiling — ` +
          `w now charges something the derivation does not describe`,
      );
      assert.ok(
        m.worstCeil >= ann.attain,
        `${def.key}: derived ceiling not approached (${m.worstCeil.toFixed(3)} < ` +
          `${ann.attain}) — the op charges MORE than its true norm (DE over-tight)`,
      );
    }
  }
});

test("approx-derived tier: analytic ceiling holds, is approached, and the raw looseness is pinned", () => {
  for (const def of OPERATORS) {
    const tier = tierOf(def);
    if (tier !== "approx-derived") continue;
    const ann = AUDIT[def.key];
    const m = RESULTS.get(def.key);
    assert.equal(m.nanRate, 0, `${def.key}: non-finite outputs`);
    assert.ok(
      m.worstCeil <= 1.02,
      `${def.key}: measured ${m.worstCeil.toFixed(4)}× the DERIVED ceiling — ` +
        `the CPU twin does not implement the map the derivation describes`,
    );
    assert.ok(
      m.worstCeil >= ann.attain,
      `${def.key}: ceiling never approached (${m.worstCeil.toFixed(3)} < ${ann.attain}) — ` +
        `derived bound too loose, tighten the formula`,
    );
    const pin = REPORT_PINS[def.key];
    if (pin !== undefined)
      assert.ok(
        m.p99 <= pin,
        `${def.key}: p99 looseness ${m.p99.toFixed(2)} > drift pin ${pin}`,
      );
  }
});

test("approx-report tier: seeded percentile drift pins (justified in AUDIT notes)", () => {
  for (const def of OPERATORS) {
    if (tierOf(def) !== "approx-report") continue;
    const ann = AUDIT[def.key] ?? {};
    assert.ok(
      ann.note,
      `${def.key}: report-tier op without a justification note — never skip silently`,
    );
    const m = RESULTS.get(def.key);
    const pin = REPORT_PINS[def.key];
    assert.ok(
      pin !== undefined,
      `${def.key}: report tier needs a REPORT_PINS entry`,
    );
    assert.ok(
      m.p99 <= pin,
      `${def.key}: p99 ${m.p99.toFixed(2)} > drift pin ${pin} — emitter/param drift?`,
    );
    assert.ok(
      m.nanRate <= (ann.nanPin ?? DEFAULT_NAN_PIN),
      `${def.key}: non-finite rate ${m.nanRate}`,
    );
  }
});

// The breach-pinned tier is EMPTY as of #643: its three founding members were
// resolved (varyScale → exact, twist + cylinderFold → deApprox/approx-derived)
// and the pins below fired exactly as designed on the way out. The machinery
// stays — it is the lifecycle a future audit finding follows: pin the derived
// ceiling + the breach magnitude, ship the decision as its own issue, and let
// these gates fail the moment the op is fixed. An empty tier makes this test
// vacuous, which IS the reading to want: nothing shipped now declares a factor
// its Jacobian breaches without a deApprox tag to say so.
test("breach-pinned tier: derived ceiling proves the analysis; the breach itself is pinned", () => {
  for (const def of OPERATORS) {
    if (tierOf(def) !== "breach-pinned") continue;
    const ann = AUDIT[def.key];
    const m = RESULTS.get(def.key);
    assert.equal(m.nanRate, 0, `${def.key}: non-finite outputs`);
    // The hand-derived true-norm field is a real upper bound…
    assert.ok(
      m.worstCeil <= 1.02,
      `${def.key}: measured ${m.worstCeil.toFixed(4)}× the derived TRUE-norm ceiling — ` +
        `the breach derivation itself is wrong, re-derive before pinning`,
    );
    // …and tight enough to be THE norm, not just an inequality:
    assert.ok(
      m.worstCeil >= ann.attain,
      `${def.key}: derived ceiling not approached (${m.worstCeil.toFixed(3)})`,
    );
    // …and the DECLARED factor is provably breached. If this fails, the op
    // was fixed — congratulations: delete its breach-pinned entry and let it
    // audit exact (or approx, if it gained a deApprox tag).
    assert.ok(
      m.worstRel >= ann.breachFloor,
      `${def.key}: declared-contract breach no longer measured ` +
        `(worst ${m.worstRel.toFixed(2)} < ${ann.breachFloor}) — un-pin this op`,
    );
    const pin = REPORT_PINS[def.key];
    assert.ok(
      m.worstRel <= pin,
      `${def.key}: breach magnitude ${m.worstRel.toFixed(1)} > pin ${pin} — looseness grew`,
    );
  }
});

test("bulb tier: dr recurrence is exactly affine; ratio/(G+1) percentile pinned", () => {
  for (const def of OPERATORS) {
    if (tierOf(def) !== "bulb") continue;
    const m = RESULTS.get(def.key);
    const pin = BULB_PINS[def.key] ?? DEFAULT_BULB_PIN;
    assert.ok(
      m.p99 <= pin,
      `${def.key}: p99 ratio/(G+1) = ${m.p99.toFixed(2)} > pin ${pin}`,
    );
    assert.ok(
      m.nanRate <= DEFAULT_NAN_PIN,
      `${def.key}: non-finite rate ${m.nanRate}`,
    );
  }
});

test("numeric tier: w byte-untouched (the only cross-emitter contract that exists)", () => {
  for (const def of OPERATORS) {
    if (tierOf(def) !== "numeric") continue;
    const ann = AUDIT[def.key] ?? {};
    const m = RESULTS.get(def.key);
    // Norm gates are skipped ON PURPOSE: the numeric DE finite-differences
    // the composed map at march time, so no per-op factor exists to audit
    // (isNumericDE "ignores w entirely" — operators.js). The auditable
    // contract is w-neutrality (checked in the conformance test) + a pinned
    // non-finite rate.
    assert.ok(
      m.nanRate <= (ann.nanPin ?? DEFAULT_NAN_PIN),
      `${def.key}: non-finite rate ${(m.nanRate * 100).toFixed(1)}% ` +
        `(pin ${(ann.nanPin ?? DEFAULT_NAN_PIN) * 100}%)`,
    );
  }
});

// ── fundamental-domain gates for the seam ops ─────────────────────────────
// Lipschitz-over-non-seam-pairs alone cannot validate a fold (#632 lesson):
// each op whose seam pairs are excluded above must prove its output lands in
// the declared cell.

test("modFold domain: wrapped axes land in [−cell/2, cell/2]; cell 0 = untouched", () => {
  const rand = mulberry32(0x636a);
  for (let i = 0; i < 20000; i++) {
    const v = [rand() * 8, rand() * 8, rand() < 0.3 ? 0 : rand() * 8];
    const p = [1, 2, 3].map(() => (rand() - 0.5) * 60);
    const s = runOp("modFold", v, p, 1);
    [s.x, s.y, s.z].forEach((out, ax) => {
      if (v[ax] > 0)
        assert.ok(
          Math.abs(out) <= v[ax] / 2 + 1e-9,
          `axis ${ax}: ${out} cell ${v[ax]}`,
        );
      else assert.equal(out, p[ax], "cell 0 leaves the axis alone");
    });
  }
});

test("brickFold domain: course and (staggered) column land in their half-cells", () => {
  const rand = mulberry32(0x636b);
  for (let i = 0; i < 20000; i++) {
    const v = [rand() * 8, rand() * 8, (rand() - 0.5) * 16];
    if (rand() < 0.2) v[rand() < 0.5 ? 0 : 1] = 0;
    const p = [1, 2, 3].map(() => (rand() - 0.5) * 60);
    const s = runOp("brickFold", v, p, 1);
    if (v[1] > 0)
      assert.ok(Math.abs(s.y) <= v[1] / 2 + 1e-9, `y ${s.y} cell ${v[1]}`);
    else assert.equal(s.y, p[1]);
    if (v[0] > 0)
      assert.ok(Math.abs(s.x) <= v[0] / 2 + 1e-9, `x ${s.x} cell ${v[0]}`);
    else assert.equal(s.x, v[1] > 0 ? p[0] : p[0], "CellX 0 leaves x alone");
    assert.equal(s.z, p[2], "z never touched");
  }
});

test("kaleido/polyAngleFold domain: radius+z preserved exactly, folded angle in the wedge", () => {
  const rand = mulberry32(0x636c);
  const wrapPi = (a) => Math.atan2(Math.sin(a), Math.cos(a));
  for (let i = 0; i < 20000; i++) {
    const key = rand() < 0.5 ? "kaleido" : "polyAngleFold";
    const n = 2 + Math.floor(rand() * 14);
    const twist = (rand() - 0.5) * 360;
    const mirror = rand() < 0.5 ? 0 : 1;
    const v = [n, twist, mirror];
    const r = 10 ** (rand() * 3 - 1.5);
    const th = rand() * 2 * Math.PI;
    const p = [r * Math.cos(th), r * Math.sin(th), (rand() - 0.5) * 4];
    const s = runOp(key, v, p, 1);
    assert.ok(
      Math.abs(Math.hypot(s.x, s.y) - r) <= 1e-9 * (1 + r),
      `${key}: in-plane radius must be preserved (fold reaches its cell by rotation only)`,
    );
    assert.equal(s.z, p[2], `${key}: z untouched`);
    const wedge = (2 * Math.PI) / Math.max(n, 2);
    // kaleido applies its twist AFTER the fold; polyAngleFold folds about the
    // offset lattice. Either way the folded local angle lives in the cell.
    const local = wrapPi(Math.atan2(s.y, s.x) - rad(twist));
    if (mirror > 0.5)
      assert.ok(
        local >= -1e-9 && local <= wedge / 2 + 1e-9,
        `${key} mirror: local ${local} outside [0, ${wedge / 2}]`,
      );
    else
      assert.ok(
        Math.abs(local) <= wedge / 2 + 1e-9,
        `${key}: local ${local} outside ±${wedge / 2}`,
      );
  }
});

test("zFold closed form: identity below Threshold, translate-by-Shift above", () => {
  const rand = mulberry32(0x636d);
  for (let i = 0; i < 20000; i++) {
    const v = [(rand() - 0.5) * 6, (rand() - 0.5) * 8];
    const p = [1, 2, 3].map(() => (rand() - 0.5) * 12);
    const s = runOp("zFold", v, p, 1);
    assert.equal(s.x, p[0]);
    assert.equal(s.y, p[1]);
    assert.equal(
      s.z,
      p[2] > v[0] ? p[2] - v[1] : p[2],
      "piecewise translation exact",
    );
  }
});

test("tentFold domain: active axes land in [0, period/2]", () => {
  const rand = mulberry32(0x636e);
  for (let i = 0; i < 20000; i++) {
    const v = [rand() * 8, rand() * 8, rand() < 0.3 ? 0 : rand() * 8];
    const p = [1, 2, 3].map(() => (rand() - 0.5) * 60);
    const s = runOp("tentFold", v, p, 1);
    [s.x, s.y, s.z].forEach((out, ax) => {
      if (v[ax] > 0)
        assert.ok(
          out >= -1e-9 && out <= v[ax] / 2 + 1e-9,
          `axis ${ax}: ${out}`,
        );
      else assert.equal(out, p[ax]);
    });
  }
});

test("hingeFold domain: identity side untouched; cut side rotated rigidly about the hinge line", () => {
  // Compact twin of the full gate in core/seamclamp.test.mjs (which also
  // proves the seam-CHANNEL side of the contract against ground truth —
  // min(DE, seam) never exceeds the true distance across the tear). Here:
  // the two branch actions are exactly what the seam predicate above assumes.
  const rand = mulberry32(0x633d);
  for (let i = 0; i < 20000; i++) {
    const v = [
      rand() * 360 - 180, // FoldAngle°
      rand() * 360 - 180, // CutAngle°
      Math.floor(rand() * 3), // Axis
      rand() * 8 - 4, // Offset
    ];
    const p = [1, 2, 3].map(() => (rand() - 0.5) * 8);
    const s = runOp("hingeFold", v, p, 1);
    const ax = v[2];
    const uv = (x, y, z) => (ax === 0 ? [y, z] : ax === 2 ? [x, y] : [x, z]);
    const axc = (x, y, z) => (ax === 0 ? x : ax === 2 ? z : y);
    const mx = -Math.sin(rad(v[1])),
      my = Math.cos(rad(v[1]));
    const [pu, pv] = uv(...p);
    const side = pu * mx + pv * my - v[3];
    if (side <= 0) {
      assert.deepEqual([s.x, s.y, s.z], p, "identity side must not move");
    } else {
      const c = [mx * v[3], my * v[3]];
      const [su, sv] = uv(s.x, s.y, s.z);
      assert.ok(
        Math.abs(axc(s.x, s.y, s.z) - axc(...p)) < 1e-12,
        "hinge-axis coordinate untouched",
      );
      assert.ok(
        Math.abs(
          Math.hypot(su - c[0], sv - c[1]) - Math.hypot(pu - c[0], pv - c[1]),
        ) < 1e-9,
        "distance to the hinge line preserved (rigid rotation)",
      );
      const dAng =
        Math.atan2(sv - c[1], su - c[0]) - Math.atan2(pv - c[1], pu - c[0]);
      const wrap = ((dAng - rad(v[0]) + Math.PI * 3) % (Math.PI * 2)) - Math.PI;
      assert.ok(
        Math.abs(wrap) < 1e-9,
        `rotation exactly FoldAngle (Δ ${wrap})`,
      );
      // and the seam report is the exact cut distance (w_in = 1)
      assert.ok(Math.abs(s.seam - side) < 1e-12, "seam = |cut distance|");
    }
  }
});

// ── the findings table (printed for the PR / future audits) ───────────────
test("audit summary (informational)", () => {
  const rows = [
    ["op", "wRule", "tier", "worst/declared", "p99", "worst/ceiling", "nan%"],
  ];
  for (const def of OPERATORS) {
    const m = RESULTS.get(def.key);
    rows.push([
      def.key + (def.deApprox ? "*" : ""),
      def.wRule,
      tierOf(def),
      m.worstRel ? m.worstRel.toFixed(3) : "—",
      m.p99 ? m.p99.toFixed(3) : "—",
      m.worstCeil ? m.worstCeil.toFixed(3) : "—",
      m.nanRate ? (m.nanRate * 100).toFixed(1) : "0",
    ]);
  }
  const w = rows[0].map((_, c) =>
    Math.max(...rows.map((r) => String(r[c]).length)),
  );
  for (const r of rows)
    console.log(r.map((x, c) => String(x).padEnd(w[c])).join("  "));
  assert.ok(rows.length === OPERATORS.length + 1);
});

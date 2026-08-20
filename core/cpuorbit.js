// The per-operator DE math + orbit runner — split out of cpu.js (issue #266)
// so evaluate.js's generator oracle (measure()) doesn't have to statically
// import the whole CPU/ASCII render tier just to get makeOrbit. cpu.js still
// imports applyOp/parseHybrid/makeOrbit from HERE (its scene/leaf/render code
// needs them too) — this module has no reverse dependency on cpu.js, so it's
// the one piece of the CPU tier that's safe on the boot path.
//
// ⚠ The per-op math below MIRRORS the WGSL in operators.js. The two are the
// same math in two languages — if you change an operator there, change it
// here too (cpu.test guards that every preset still produces a finite,
// non-empty render).

import { isEscapeTime, effectiveDeOption, activeOps } from "./operators.js";
import { hybridDeFamily } from "./stability.js";
import { activeHybridSlots } from "./hybridmodel.js";
import { BAILOUT_ESCAPE, BAILOUT_IFS } from "./limits.js";

const D2R = Math.PI / 180;
const rad = (d) => d * D2R;

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
    case "mandalayFold": {
      // Mandalay fold — abs → 3-comparator descending sort → offset-pair folds
      // with a clamped diagonal ("odd") fold between them. Every step is a
      // reflection/transposition/translation, so |J| = 1 and w is untouched.
      // ⚠ In-place and order-dependent: mirror this sequence exactly.
      const mfo = v[0],
        mg = v[1],
        mzf = v[2];
      let mx = Math.abs(s.x),
        my = Math.abs(s.y),
        mz = Math.abs(s.z);
      let mt = Math.max(0, my - mx);
      mx += mt;
      my -= mt;
      mt = Math.max(0, mz - my);
      my += mt;
      mz -= mt;
      mt = Math.max(0, my - mx);
      mx += mt;
      my -= mt;
      mx = Math.abs(mx - mfo) - mfo;
      mt = Math.min(mg, Math.max(0, mx - my));
      mx -= mt;
      my += mt;
      my = mfo - Math.abs(my - mfo);
      if (mzf > 0) mz = Math.min(mz, 2 * mzf - mz);
      s.x = mx;
      s.y = my;
      s.z = mz;
      break;
    }
    case "torusInvert": {
      // Torus-space inversion (_toruspinv family) — meridian-plane inversion
      // about the core circle of radius R. Variant 0 is the true inversion
      // (Radius²/d²), 1–3 the pseudo forms. Approximate DE: w tracks the
      // MERIDIAN/radial factor only (polygonFold's precedent) — never max()
      // with the azimuthal ρ′/ρ, which diverges on the axis and flattens every
      // render to a wall. See the operators.js note.
      const tRad = Math.max(Math.abs(v[0]), 1e-6),
        tR = v[1],
        tv = Math.round(v[2]);
      const trho = Math.hypot(s.x, s.y),
        tu = tR - trho,
        td2 = Math.max(tu * tu + s.z * s.z, 1e-12);
      let trhoN, tzN, tk;
      if (tv === 1) {
        const td = Math.sqrt(td2),
          ts = tRad / td;
        trhoN = trho * ts;
        tzN = s.z * (td / tRad);
        tk = ts;
      } else if (tv === 2) {
        const td = Math.sqrt(td2),
          ts = tRad / td;
        trhoN = trho * ts;
        tzN = s.z * (td * tRad);
        tk = ts;
      } else if (tv === 3) {
        const ts = tRad / td2;
        trhoN = trho * ts;
        tzN = s.z * (td2 / tRad);
        tk = ts;
      } else {
        const tki = (tRad * tRad) / td2;
        trhoN = tR - tu * tki;
        tzN = s.z * tki;
        tk = tki;
      }
      let tdx = 1.0,
        tdy = 0.0;
      if (trho > 1e-9) {
        tdx = s.x / trho;
        tdy = s.y / trho;
      }
      s.x = tdx * trhoN;
      s.y = tdy * trhoN;
      s.z = tzN;
      s.w *= tk;
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
    case "riemannBulb": {
      // Riemann-sphere stereographic power — numeric DE (w untouched).
      const rP = v[0],
        rR = s.x * s.x + s.y * s.y + s.z * s.z;
      if (rR > 1e-18) {
        const invr = 1 / Math.sqrt(rR);
        const rd = Math.min(s.y * invr - 1, -1e-7);
        const ru = 1 / rd;
        const ra = s.x * invr * ru,
          rb = s.z * invr * ru;
        const alphaP = Math.atan2(ra + ra, ra * ra - 1) * rP;
        const betaP = Math.atan2(rb + rb, rb * rb - 1) * rP;
        const ta = Math.sin(alphaP) / Math.max(Math.cos(alphaP) + 1, 1e-30);
        const tb = Math.sin(betaP) / Math.max(Math.cos(betaP) + 1, 1e-30);
        const m = Math.max(Math.abs(ta), Math.abs(tb), 1);
        const tA = ta / m,
          tB = tb / m;
        const q = tA * tA + tB * tB;
        const im = 1 / m;
        const den = m * q + im;
        const rp = Math.pow(Math.sqrt(rR), rP);
        s.x = ((2 * tA) / den) * rp;
        s.y = ((q - im * im) / (q + im * im)) * rp;
        s.z = ((2 * tB) / den) * rp;
      }
      break;
    }
    case "makinTri": {
      // Makin triplex square (variant 0/1) — numeric DE (w untouched).
      const mv = Math.round(v[0]);
      const ax = s.x,
        ay = s.y,
        az = s.z;
      if (mv === 1) {
        s.x = ax * ax + 2 * ay * az;
        s.y = -(ay * ay + 2 * az * ax);
        s.z = -(az * az) + 2 * ay * ax;
      } else {
        s.x = ax * ax - ay * ay - az * az;
        s.y = 2 * ax * ay;
        s.z = 2 * az * (ax - ay);
      }
      break;
    }
    case "makinFuzzy": {
      // Makin fuzzy square (Makin3D-3-4) — numeric DE (w untouched).
      const fy = Math.round(v[0]),
        fz = Math.round(v[1]),
        fl = Math.abs(v[2]);
      const ax = s.x,
        ay = s.y,
        az = s.z;
      let mz = az * az;
      if (fy === 1 && az <= 0) mz = -mz;
      let my = ay * ay;
      if (fz === 1 && ay <= 0) my = -my;
      s.x = ax * ax - ay * ay - az * az;
      s.y = 2 * ax * ay * (1 - mz / (ax * ax + ay * ay + fl));
      s.z = 2 * ax * az * (1 - my / (ax * ax + az * az + fl));
      break;
    }
    case "smoothBoxFold": {
      // C1 rounded box fold — approximate DE (w untouched).
      const sF = v[0],
        sSh = v[1],
        sFx = v[2];
      const f = (a) => {
        const t = Math.pow(Math.abs(a), sSh) * sFx;
        return (Math.abs(a) + (sF + sF - Math.abs(a)) * t) / (t + 1);
      };
      s.x = f(s.x);
      s.y = f(s.y);
      s.z = f(s.z);
      break;
    }
    case "smoothBallFold": {
      // Soft radial blend — approximate DE (w *= k).
      const bMr = v[0],
        bSh = v[1],
        bFx = v[2];
      const br2 = s.x * s.x + s.y * s.y + s.z * s.z;
      const bc = (1 + bMr) * 0.5;
      const bh = Math.max((1 - bMr) * 0.5, 1e-20);
      const bn = Math.abs(br2 - bc) / bh;
      let bm = 1;
      if (bMr < 0.99) {
        const bb = Math.pow(Math.sqrt(bn), bSh) * bFx;
        bm = bc - (bh * (bb + bn)) / (1 + bb);
      }
      const bk = 1 / Math.max(Math.abs(bm), 1e-20);
      s.x *= bk;
      s.y *= bk;
      s.z *= bk;
      s.w *= bk;
      break;
    }
    case "asinhWarp": {
      // Per-axis asinh warp — approximate DE (w untouched).
      const m = Math.round(v[0]);
      const k = m === 0 ? "z" : m === 2 ? "x" : "y";
      const t = s[k] * v[1];
      s[k] = v[2] * Math.log2(t + Math.sqrt(t * t + 1));
      break;
    }
    case "logWarp": {
      // Per-axis log crush — approximate DE (w untouched).
      const m = Math.round(v[0]);
      const k = m === 1 ? "y" : m === 2 ? "x" : "z";
      s[k] = v[2] * Math.log2(Math.abs(s[k] * v[1]) + 0.01);
      break;
    }
    case "neoSqrWarp": {
      // Per-axis signed parabola — approximate DE (w untouched).
      const m = Math.round(v[0]);
      const k = m === 0 ? "z" : m === 2 ? "x" : "y";
      const t = s[k] * v[2];
      s[k] = t >= 0 ? t * (v[1] - t) : t * (t - v[1]);
      break;
    }
    case "sinShear": {
      // Cross-axis sine shear — approximate DE (w untouched).
      const m = Math.round(v[0]);
      if (m === 1) s.z += v[1] * Math.sin(v[2] * s.y);
      else if (m === 2) s.x += v[1] * Math.sin(v[2] * s.z);
      else s.y += v[1] * Math.sin(v[2] * s.z);
      break;
    }
    case "gnarl2D":
    case "gnarl3D": {
      // Nested-sine gnarl warp — approximate DE (w untouched).
      const gs = v[0],
        ga = v[1],
        gb = v[2];
      const g = (b) => Math.sin(Math.sin((Math.sin(b * gb) + b) * ga) + b);
      const gx = s.x,
        gy = s.y,
        gz = s.z;
      if (key === "gnarl3D") {
        s.x = gx - gs * g(gz);
        s.y = gy - gs * g(gx);
        s.z = gz - gs * g(gy);
      } else {
        s.x = gx - gs * g(gy);
        s.y = gy - gs * g(gx);
      }
      break;
    }
    case "toCoord": {
      // Curvilinear frame change TO — approximate DE (w untouched).
      const cs = Math.round(v[0]),
        cR = v[1],
        cG = v[2];
      const cx = s.x,
        cy = s.y,
        cz = s.z;
      if (cs === 1) {
        s.x = Math.hypot(cx, cy, cz);
        s.y = Math.atan2(Math.hypot(cx, cy), cz);
        s.z = Math.atan2(cy, cx);
      } else if (cs === 2) {
        s.x = cR - Math.hypot(cx, cy);
        s.y = Math.atan2(cy, cx);
      } else if (cs === 3) {
        const rho = Math.hypot(cx, cy) - cR;
        s.x = Math.hypot(rho, cz);
        s.y = Math.atan2(rho, cz) - cG - Math.PI / 2;
        s.z = Math.atan2(cy, cx);
      } else if (cs === 4) {
        // Log-polar (complex log): x ← ln ρ, y ← θ − Γ·ln ρ. ρ floored at
        // 1e-12 so the origin stays finite; Γ shears the angle lane by the
        // log-radius, straightening logarithmic spirals.
        const lr = Math.log(Math.max(Math.hypot(cx, cy), 1e-12));
        s.x = lr;
        s.y = Math.atan2(cy, cx) - cG * lr;
      } else {
        s.x = Math.hypot(cx, cy);
        s.y = Math.atan2(cy, cx);
      }
      break;
    }
    case "fromCoord": {
      // Inverse frame change — approximate DE (w untouched).
      const cs = Math.round(v[0]),
        cR = v[1],
        cG = v[2];
      const cx = s.x,
        cy = s.y,
        cz = s.z;
      if (cs === 1) {
        s.x = Math.cos(cz) * Math.sin(cy) * cx;
        s.y = Math.sin(cz) * Math.sin(cy) * cx;
        s.z = Math.cos(cy) * cx;
      } else if (cs === 2) {
        const q = Math.abs(cR - cx);
        const sg = Math.cos(cy) < 0 ? -1 : 1;
        s.x = q * Math.abs(Math.cos(cy));
        s.y = q * Math.sin(cy) * sg;
      } else if (cs === 3) {
        const t = cx * Math.cos(cy + cG) + cR;
        s.x = t * Math.cos(cz);
        s.y = t * Math.sin(cz);
        s.z = cx * Math.sin(cy + cG);
      } else if (cs === 4) {
        // Complex exponential — exact inverse of toCoord's log-polar. The
        // exponent cap (60) must match the shader tiers or they disagree.
        const le = Math.exp(Math.min(cx, 60));
        const la = cy + cG * cx;
        s.x = le * Math.cos(la);
        s.y = le * Math.sin(la);
      } else {
        s.x = Math.cos(cy) * cx;
        s.y = Math.sin(cy) * cx;
      }
      break;
    }
    case "polygonFold": {
      // Polygon↔circle radial remap — approximate DE (w *= f best-effort).
      const pfn = Math.max(Math.round(v[0]), 3);
      const pfs = v[1];
      const pfm = Math.round(v[2]);
      let u = s.x,
        vv = s.y;
      if (pfm === 1) {
        u = s.z;
        vv = s.x;
      } else if (pfm === 2) {
        u = s.y;
        vv = s.z;
      }
      if (u * u + vv * vv > 1e-24) {
        const sector = (2 * Math.PI) / pfn;
        const a = Math.atan2(vv, u);
        const th = a - sector * Math.floor(a / sector + 0.5);
        const c = Math.max(Math.cos(th), 1e-6);
        const f = pfs < 0 ? 1 + -pfs * (1 / c - 1) : 1 + pfs * (c - 1);
        u *= f;
        vv *= f;
        s.w *= Math.abs(f);
        if (pfm === 1) {
          s.z = u;
          s.x = vv;
        } else if (pfm === 2) {
          s.y = u;
          s.z = vv;
        } else {
          s.x = u;
          s.y = vv;
        }
      }
      break;
    }
    case "magnetXYZ":
    case "magnetXYZAbs": {
      // Per-axis magnet rational map — numeric DE (w untouched).
      const mP = v[0],
        mA1 = v[1],
        mA2 = v[2];
      const mx = s.x,
        my = s.y,
        mz = s.z;
      const mR2 = mx * mx + my * my + mz * mz;
      if (mR2 > 1e-18) {
        const mr = Math.sqrt(mR2);
        const pa1 = mP * mA1;
        const isAbs = key === "magnetXYZAbs";
        const ang = isAbs ? mA1 : mP;
        const lane = (a) => {
          const q = Math.pow(mR2 + mA2 * a * a, mP * 0.5);
          const out = Math.cos(ang * Math.atan2(mr, Math.abs(a) * pa1)) * q;
          return isAbs ? Math.abs(out) : out;
        };
        s.x = lane(mx);
        s.y = lane(my);
        s.z = lane(mz);
      }
      break;
    }
    case "kleinPolyMap": {
      // Klein polyhedral rational map (Tglad family) — numeric DE (w untouched).
      const ko = Math.max(Math.round(v[0]) & 7, 1);
      const ki = Math.max(Math.round(v[1]) & 7, 1);
      const kv = Math.round(v[2]);
      const kalt = (kv & 1) === 1;
      const kdih = kv >= 2;
      let x = s.x,
        y = s.y,
        z = s.z;
      let esc = false;
      for (let io = 0; io < ko; io++) {
        const rho2 = x * x + y * y;
        const r2 = rho2 + z * z;
        if (r2 >= 1e10) {
          esc = true;
          break;
        }
        let rr = Math.sqrt(r2);
        const zr = z / rr;
        const tang = (Math.sqrt(rho2) - y) / x;
        const tsq = tang * tang;
        const qq = (1 / (tsq + 1)) * Math.sqrt((1 - zr) / (1 + zr));
        if (kalt) {
          y = (qq + qq) * tang;
          x = (tsq - 1) * qq;
        } else {
          x = (qq + qq) * tang;
          y = (1 - tsq) * qq;
        }
        for (let ii = 0; ii < ki; ii++) {
          if (kdih) {
            const j = x * x + y * y;
            if (j >= 1e10) {
              esc = true;
              break;
            }
            const ni = -y * (j + 1) * (j - 2 * x - 1) * (j + 2 * x - 1);
            const nr = x * (j - 1) * (j * j + 2 * j + 4 * y * y + 1);
            const cd = (j - 2 * y + 1) * (j + 2 * y + 1);
            const cm = 2 / (cd * cd);
            x = cm * nr;
            y = cm * ni;
          } else {
            const sRe = x * x * x - 3 * y * y * x;
            if (sRe >= 1e10) {
              esc = true;
              break;
            }
            const tIm = y * y * y - 3 * x * x * y;
            const bN = tIm * 3.181980515339464;
            const tSq = tIm * tIm;
            const sk = sRe + 0.3535533905932738;
            const aN = sk * (2.8284271247461903 - sRe) - tSq;
            const cm = -0.3535533905932738 / (sk * sk + tSq);
            const xn = (aN * x - bN * y) * cm;
            const yn = (aN * y + bN * x) * cm;
            x = xn;
            y = yn;
          }
          rr = rr * rr;
        }
        if (esc) break;
        if (kalt) {
          const t = x;
          x = y;
          y = t;
        }
        const u = x * x + y * y;
        const su = Math.sqrt(u);
        const tg = (su - y) / x;
        const tq = tg * tg;
        const inv = 1 / (tq + 1);
        const cosO = (1 - tq) * inv;
        const sinO = (tg + tg) * inv;
        const g = 2 / (1 + u);
        const sa = g * su;
        const xn = rr * sinO * sa;
        const yn = rr * cosO * sa;
        z = (kalt ? 1 - g : g - 1) * rr;
        x = xn;
        y = yn;
      }
      if (esc) {
        s.x = 1e10;
        s.y = 1e10;
        s.z = 1e10;
      } else {
        s.x = x;
        s.y = y;
        s.z = z;
      }
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
      // Mirror off (v[2] 0) = pure rotational sector-snap; a missing 3rd
      // value (pre-mirror-param op lists) keeps the historic reflection.
      if ((v[2] ?? 1) > 0.5) ang = Math.abs(ang);
      ang = ang + rad(v[1]);
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
    // Complex map — a Möbius transform conjugated by ζ↦ζ^N on the XY plane
    // (variant 0 Cayley, 1 Murl); the Z lane rides through untouched. w is
    // deliberately NOT tracked: |f'| diverges on the pole LINES, which is the
    // torusInvert flat-wall lesson. See operators.js id 61 for the derivation.
    case "complexMap": {
      const cmN = Math.max(Math.round(v[0]), 1);
      const cmC = v[1];
      const cmv = Math.round(v[2]);
      const cmr = Math.hypot(s.x, s.y);
      const cmth = Math.atan2(s.y, s.x);
      const cmrn = Math.pow(cmr, cmN);
      const cman = cmth * cmN;
      const cmur = cmrn * Math.cos(cman);
      const cmui = cmrn * Math.sin(cman);
      if (cmv === 1) {
        const cmc = cmN > 1.5 ? cmC / (cmN - 1) : cmC;
        let cmA = 1 + cmc;
        if (Math.abs(cmA) < 1e-6) cmA = 1e-6;
        const cmdr = 1 + cmc * cmur;
        const cmdi = cmc * cmui;
        const cmden = Math.max(cmdr * cmdr + cmdi * cmdi, 1e-12);
        const nx = (cmA * (s.x * cmdr + s.y * cmdi)) / cmden;
        s.y = (cmA * (s.y * cmdr - s.x * cmdi)) / cmden;
        s.x = nx;
      } else if (cmv === 2) {
        const cme = 2 / cmN;
        const cmdr = 1 + cmC * cmur;
        const cmdi = cmC * cmui;
        const cmdm = Math.max(Math.hypot(cmdr, cmdi), 1e-12);
        const cmpm = Math.pow(cmdm, cme);
        const cmpa = Math.atan2(cmdi, cmdr) * cme;
        const cmqr = cmpm * Math.cos(cmpa);
        const cmqi = cmpm * Math.sin(cmpa);
        const cmA = Math.pow(Math.max(Math.abs(1 + cmC), 1e-6), cme);
        const cmden = Math.max(cmqr * cmqr + cmqi * cmqi, 1e-12);
        const nx = (cmA * (s.x * cmqr + s.y * cmqi)) / cmden;
        s.y = (cmA * (s.y * cmqr - s.x * cmqi)) / cmden;
        s.x = nx;
      } else {
        const cmni = cmui - cmC;
        const cmdi = cmui + cmC;
        const cmden = Math.max(cmur * cmur + cmdi * cmdi, 1e-12);
        const nx = (cmur * cmur + cmni * cmdi) / cmden;
        s.y = (cmni * cmur - cmur * cmdi) / cmden;
        s.x = nx;
      }
      break;
    }
    // Brick / running-bond tiling: fold Y into its course, then fold X with a
    // per-row phase shift. The row index MUST be read off the pre-fold y.
    // Stagger 0 is modFold(CellX, CellY, 0) exactly; per-cell translation, so
    // w is untouched (see operators.js id 60 for the derivation).
    case "brickFold": {
      let brow = 0;
      if (v[1] > 0) {
        brow = Math.floor(s.y / v[1] + 0.5);
        s.y -= v[1] * brow;
      }
      if (v[0] > 0) {
        const bx = s.x + v[2] * brow;
        s.x = bx - v[0] * Math.floor(bx / v[0] + 0.5);
      }
      break;
    }
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
    // Bulb Power (axis) — mirrors operators.js id 29. Params 3-4 (ThetaMul,
    // PhiMul) ride the opAux lane on the WebGPU tier; here `v` is an untyped
    // array, so the split powers are invisible plumbing-wise. `?? 1` keeps
    // every short payload — the pre-Convention 2-value form, the 3-value form
    // the Convention add shipped, and 4-value — reading as the TIED-power
    // degeneracy, which is the same arity-migration mechanism Convention's own
    // `?? 0` uses. `??` (not `||`) matters: ThetaMul 0 is a legal value.
    case "bulbAxis": {
      const bp = v[0],
        m = Math.round(v[1]),
        conv = Math.round(v[2] ?? 0),
        btm = v[3] ?? 1,
        bpm = v[4] ?? 1,
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
        const bu = Math.max(-1, Math.min(1, up / br));
        const bth = (conv === 2 ? Math.asin(bu) : Math.acos(bu)) * bp * btm;
        const bph = Math.atan2(b, a) * bp * bpm,
          brn = Math.pow(br, bp);
        s.w = ((bp * brn) / br) * s.w + 1;
        const bst = conv !== 0 ? Math.cos(bth) : Math.sin(bth);
        const na = brn * bst * Math.cos(bph),
          nb = brn * bst * Math.sin(bph),
          nup = brn * (conv !== 0 ? Math.sin(bth) : Math.cos(bth));
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
    // Rucker bulb — mirrors operators.js id 62 (see the derivation block there).
    // The first op whose params run past the 3 inline slots: on this tier that
    // is invisible (v is an untyped array), which is exactly why the CPU is the
    // oracle the opAux lane is checked against. `?? 0` keeps short payloads
    // reading as RadialSel/Convention 0 — the same arity-migration mechanism
    // bulbAxis's Convention uses.
    case "ruckerBulb": {
      const rbp = v[0],
        rbaz = v[1] ?? 1,
        rbzm = v[2] ?? 1,
        rsel = Math.round(v[3] ?? 0),
        rc0 = Math.round(v[4] ?? 0),
        // An out-of-range enum falls back to 0, like every other selector op
        // (complexMap's Variant). Without this, conv 9 would take the acos
        // angle of flavor 0 but flavor 1's swapped sin/cos — a hybrid that is
        // no declared convention at all.
        rconv = rc0 >= 0 && rc0 <= 3 ? rc0 : 0,
        rbr = Math.hypot(s.x, s.y, s.z);
      if (rbr > 1e-9) {
        const rbu = Math.max(-1, Math.min(1, s.z / rbr));
        let rbth =
          rconv === 2
            ? Math.asin(rbu)
            : rconv === 3
              ? Math.atan2(-s.x, s.z)
              : Math.acos(rbu);
        rbth *= rbp;
        const rbph = Math.atan2(s.y, s.x) * rbp * rbaz;
        const rbq = rsel === 1 ? rbp * rbaz : rbp;
        const rbrn = Math.pow(rbr, rbq);
        s.w = ((rbq * rbrn) / rbr) * Math.max(1, Math.abs(rbzm)) * s.w + 1;
        const rbst = rconv !== 0 ? Math.cos(rbth) : Math.sin(rbth);
        const rbpo = rconv !== 0 ? Math.sin(rbth) : Math.cos(rbth);
        s.x = rbrn * rbst * Math.cos(rbph);
        s.y = rbrn * rbst * Math.sin(rbph);
        s.z = rbrn * rbpo * rbzm;
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

// Hybrid iteration (IDEAS ①, docs/design/HYBRID_ITERATION.md §3.5) — parse the
// A/B slot config ONCE. Shared by makeOrbit (the DE orbit) and makeIterMeasure
// (trap/escape coloring); the two used to parse formula.hybrid verbatim-
// identically and were "kept in lockstep by construction" — now by sharing.
// Slot A = the formula's own active ops; slot B rides in formula.hybrid.b.
// Julia is formula-level (§3.8): one seed, ORed into whichever slot's own addC
// decided to fire (the caller applies that gate).
// Reads every slot through the ONE canonical accessor (hybridmodel.js) rather
// than hand-picking `.b.ops`/`.schedule` — so a legacy 2-slot save and an N-slot
// save parse through the same code. Returns the accessor-backed N-slot walk config
// `{ slots, period }` shared by makeOrbit (the DE orbit) and makeIterMeasure
// (coloring): slots[i] = { ops:[{key,v}], addC, count } with slot A at index 0
// and muted ops filtered; period = Σcount. The CPU mirror of the WGSL hybWalk /
// GL hybStep — the slot at iteration i comes from hybridSlotAt below.
export function parseHybrid(formula) {
  // activeHybridSlots is the ONE choke point that drops MUTED slots (the eye
  // toggle) — so a 3-slot hybrid with slot B muted parses byte-identically to the
  // 2-slot hybrid [A, C], and the CPU orbit skips the muted phase in lockstep
  // with the WGSL walk / GL codegen (both fed the same active-slot list).
  const { slots, counts } = activeHybridSlots(formula);
  const map = (ops) =>
    ops.filter((o) => !o.muted).map((o) => ({ key: o.key, v: o.values || [] }));
  const parsed = slots.map((s, i) => ({
    ops: map(s.ops),
    addC: !!s.addC,
    count: Math.max(1, counts[i] ?? 1),
  }));
  // Every slot muted → nothing to iterate. One degenerate empty slot (period 1)
  // keeps the orbit runner + hybridSlotAt total valid while rendering nothing —
  // the CPU twin of the legacy all-muted "renders nothing" rule.
  if (!parsed.length)
    return { slots: [{ ops: [], addC: false, count: 1 }], period: 1 };
  const period = parsed.reduce((n, s) => n + s.count, 0) || 1;
  return { slots: parsed, period };
}

// The active slot at outer iteration i — walk the schedule counts at i % period.
// ONE source for the orbit runner and the coloring measure loop (the CPU twin of
// hybWalk's schedule pick); returns a parseHybrid slot { ops, addC, count }.
export function hybridSlotAt(slots, period, i) {
  const phase = i % period;
  let acc = 0;
  for (let k = 0; k < slots.length; k++) {
    acc += slots[k].count;
    if (phase < acc) return slots[k];
  }
  return slots[slots.length - 1]; // unreachable when period === Σcount
}

// ── THE single orbit runner ───────────────────────────────────────────────────
// The one place the per-formula CPU orbit loop lives. Everything that needs
// "iterate this point and see where it lands" derives from it: the analytic and
// numeric DEs (makeDE), the hybrid A/B schedule (formerly a private makeHybridDE
// twin), and the headless evaluator (evaluate.js measure/probe/surface-march).
// Deriving — instead of re-rolling the loop — is what closed the vary.js gate
// blindspot: evaluate.js's private copies iterated slot A only, so a jitter that
// broke a hybrid's slot B sailed through measure() and shipped blank share links.
//
// makeOrbit(formula, opts) → orbit(x,y,z) → { r, aw, escaped, nan }
//   r        final radius |p| after the orbit
//   aw       accumulated |dw|, floored at 1e-9 (a safe DE denominator)
//   escaped  true iff the bailout tripped
//   nan      true iff the state went non-finite (only with opts.checkFinite)
// The closure carries `.escape` — the DE family verdict (escape-time vs IFS;
// hybrids key off hybridDeFamily, the UNION of both slots' ops, §3.3) — so
// callers can finalize a DE without re-deriving it.
//
// opts (all optional — defaults are the render-authoritative policy):
//   iters       override formula.iters (evaluate.js probes shallower)
//   bailout2    override the bailout threshold on r² (evaluate.js keeps its
//               historical radius-1e6 policy by passing bailout²)
//   checkFinite per-iteration isFinite guard → early {nan:true}. The render
//               path leaves it off and lets NaN propagate, like the GPU.
export function makeOrbit(formula, opts = {}) {
  const hybrid = !!formula?.hybrid; // callers route scenes elsewhere (§3.8)
  // +c gate mirrors the renderer (renderer.js addGate = addC || julia): Julia
  // mode forces the per-iteration add ON — c is then the fixed jc constant
  // instead of the sample point. Gating on addC alone left Julia invisible for
  // addC-off presets (#16).
  const julia = !!formula.julia;
  const jc = julia ? formula.juliaC || [0, 0, 0] : null;
  const iters = opts.iters ?? formula.iters ?? 8;
  const checkFinite = !!opts.checkFinite;
  // Flat = a degenerate schedule (period 1 ⇒ slot A every iteration), so ONE
  // loop below serves both shapes; schedule selection is integer-only and
  // leaves the flat float math bit-identical.
  const { slots: hslots, period } = hybrid
    ? parseHybrid(formula)
    : {
        slots: [
          {
            ops: activeOps(formula).map((o) => ({
              key: o.key,
              v: o.values || [],
            })),
            addC: !!formula.addC,
            count: 1,
          },
        ],
        period: 1,
      };
  const escape = hybrid
    ? hybridDeFamily(formula) !== "ifs" // 'escape' or the unsafe 'mixed' case (§3.3)
    : isEscapeTime(formula);
  // Bailout policy (same as preview.js bailoutFor()): escape-time needs the
  // small radius or rᵖᵒʷᵉʳ overflows; numeric DE (deOption 3, flat only) rides
  // the escape bailout too. Constants are r² thresholds (limits.js).
  const bail2 =
    opts.bailout2 ??
    (escape || (!hybrid && effectiveDeOption(formula) === 3)
      ? BAILOUT_ESCAPE
      : BAILOUT_IFS);
  const orbit = function orbit(px, py, pz) {
    const s = { x: px, y: py, z: pz, w: 1.0 };
    const cx = jc ? jc[0] : px,
      cy = jc ? jc[1] : py,
      cz = jc ? jc[2] : pz;
    let escaped = false;
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
      if (
        checkFinite &&
        !(isFinite(s.x) && isFinite(s.y) && isFinite(s.z) && isFinite(s.w))
      ) {
        return { r: NaN, aw: NaN, escaped, nan: true };
      }
      if (s.x * s.x + s.y * s.y + s.z * s.z > bail2) {
        escaped = true;
        break;
      }
    }
    return {
      r: Math.hypot(s.x, s.y, s.z),
      aw: Math.max(Math.abs(s.w), 1e-9),
      escaped,
      nan: false,
    };
  };
  orbit.escape = escape;
  return orbit;
}

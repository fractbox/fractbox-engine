// Splat FIT + coordinate transform (UE Gaussian-splat export §1.3 / spec §5.5).
// Pure array-in/array-out, zero formula knowledge, no deps, DOM-free — the
// reduced oriented-point cloud from splatcapture.js becomes the flat typed-array
// bundle plysplat.js writes. All the RH-Z-up → LH-Z-up (Unreal) handedness work
// lives here, and is done WITHOUT ever reflecting a quaternion: for an isotropic
// flat disc the in-plane gauge and the normal sign are free, so we mirror the
// position + normal in the data and build the orientation FRESH from the mirrored
// normal (spec §5.5). SH degree-0 color is stored as display-sRGB with NO
// de-gamma (S-pre spike: INRIA RGB2SH packs raw sRGB; viewers show it as-is).

// SH degree-0 (DC) encode/decode. `c` is display-sRGB in [0,1]; NOT linearized.
export const SH_C0 = 0.28209479177387814;
export const rgb2sh0 = (c) => (c - 0.5) / SH_C0;
export const sh02rgb = (f) => SH_C0 * f + 0.5;

// Standard sRGB → linear (opt-in via fitSplats `degamma`). The default export stores
// display-sRGB (SuperSplat-verified); degamma is ONLY for a viewer that re-applies the
// sRGB curve on read (double-gamma) — see the S2b swatch check.
export const srgb2lin = (c) =>
  c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);

// Shortest-arc unit quaternion rotating +Z onto unit vector `n`, as [w,x,y,z]
// (INRIA rot order). Standard half-vector construction: w = 1 + dot(+Z, n),
// (x,y,z) = cross(+Z, n). Antiparallel (n ≈ [0,0,-1]) collapses |q|→0 — return a
// 180° turn about X ([0,1,0,0]); any horizontal axis is valid since the disc is
// in-plane isotropic and the Gaussian is symmetric.
export function quatFromZTo(n) {
  const w = 1 + n[2];
  let x = -n[1],
    y = n[0],
    z = 0;
  let len = Math.hypot(w, x, y, z);
  if (len < 1e-8) return [0, 1, 0, 0]; // antiparallel: 180° about X
  const inv = 1 / len;
  return [w * inv, x * inv, y * inv, z * inv];
}

// Unit quaternion [w,x,y,z] for the rotation whose columns are the orthonormal
// frame (u,v,w) — i.e. R·x̂ = u, R·ŷ = v, R·ẑ = w. Standard Shepperd branch (picks
// the largest diagonal term for numerical stability). P2 anisotropy uses it to
// orient a splat with a chosen in-plane major axis (u = dir), unlike quatFromZTo
// which only pins the normal and leaves the in-plane gauge free.
export function quatFromFrame(ux, uy, uz, vx, vy, vz, wx, wy, wz) {
  // matrix columns: [u v w] → m[row][col]
  const m00 = ux,
    m01 = vx,
    m02 = wx,
    m10 = uy,
    m11 = vy,
    m12 = wy,
    m20 = uz,
    m21 = vz,
    m22 = wz;
  const tr = m00 + m11 + m22;
  let qw, qx, qy, qz;
  if (tr > 0) {
    const s = Math.sqrt(tr + 1) * 2;
    qw = 0.25 * s;
    qx = (m21 - m12) / s;
    qy = (m02 - m20) / s;
    qz = (m10 - m01) / s;
  } else if (m00 > m11 && m00 > m22) {
    const s = Math.sqrt(1 + m00 - m11 - m22) * 2;
    qw = (m21 - m12) / s;
    qx = 0.25 * s;
    qy = (m01 + m10) / s;
    qz = (m02 + m20) / s;
  } else if (m11 > m22) {
    const s = Math.sqrt(1 + m11 - m00 - m22) * 2;
    qw = (m02 - m20) / s;
    qx = (m01 + m10) / s;
    qy = 0.25 * s;
    qz = (m12 + m21) / s;
  } else {
    const s = Math.sqrt(1 + m22 - m00 - m11) * 2;
    qw = (m10 - m01) / s;
    qx = (m02 + m20) / s;
    qy = (m12 + m21) / s;
    qz = 0.25 * s;
  }
  const inv = 1 / Math.hypot(qw, qx, qy, qz);
  return [qw * inv, qx * inv, qy * inv, qz * inv];
}

// points: { count, pos, normal, albedo, radius?, alpha? }  (pos/normal/albedo =
//          Float32Array ×3N, normal unit, albedo display-sRGB in [0,1]).
//          radius/alpha (Float32Array ×N, §S1b auto-tune) are OPTIONAL per-splat
//          overrides for the global r0 / opacity; absent ⇒ exactly the S0
//          constants (bit-identical), so old callers are unaffected.
// opts: { convention = "ue" | "raw", cmScale = 1, r0, thinEps = 0.1,
//         opacity = 0.95, degamma = false }.  degamma → sRGB-linearize color first.
//   cmScale default is 1 (identity/unit size) — every real caller (app target-size,
//   CLI --size/--cmscale, swatch) passes it explicitly (#345).
//   "ue"  → p' = (−x, y, z)·cmScale, n' = (−nx, ny, nz) (one-axis mirror = the
//           det=−1 handedness flip), quat built fresh from n'.
//   "raw" → identity handedness (RH-Z-up / INRIA-COLMAP for web viewers). cmScale
//           still scales SIZE in both conventions (#345) — pass cmScale=1 for unit scale.
// Returns plysplat.writeSplatPly's input shape:
//   { count, pos, normal, fdc, opacity, scaleLog, rot }  (Float32Arrays).
export function fitSplats(points, opts = {}) {
  const {
    convention = "ue",
    cmScale = 1,
    r0,
    thinEps = 0.1,
    opacity = 0.95,
    degamma = false,
  } = opts;
  // thinEps floor (R1): below ~1e-2 flat splats enter the numerically-dangerous
  // zone the SOTA review documents (StopThePop clamps inverse-scale at 1e3; 2DGS
  // needs a dedicated low-pass standard viewers lack; edge-on discs degenerate to
  // missed screen-space lines). 0.01 is a FLOOR, not a tuning target — the
  // recommended band is 0.05–0.1. See docs/planning/SPLAT_SOTA_REVIEW.md §2.
  // (`!(x >= 0.01)` also rejects NaN/null; a missing/undefined key takes the 0.1
  // default and never reaches here, so the default path is unaffected.)
  if (!(thinEps >= 0.01))
    throw new RangeError(`fitSplats: thinEps must be ≥ 0.01 (got ${thinEps})`);
  // Color encode: display-sRGB by default; degamma linearizes first (S2b lever).
  // Clamp to [0,1] FIRST (P1.5 field-diff finding): the GPU capture's f16
  // albedo MRT delivers rare small negatives at shaded/AO'd grazing hits
  // (~0.24% of components on a Mandelbulb capture), which wrote f_dc below
  // rgb2sh0(0) — and with degamma, pow(negative, 2.4) is NaN in the .ply.
  // Viewers clamp at render so display was unaffected, but strict importers/
  // quantizers (SOG) read the raw values.
  const clamp01 = (c) => (c < 0 ? 0 : c > 1 ? 1 : c);
  const enc = degamma
    ? (c) => rgb2sh0(srgb2lin(clamp01(c)))
    : (c) => rgb2sh0(clamp01(c));
  const perRadius = points.radius && points.radius.length === points.count;
  const perAlpha = points.alpha && points.alpha.length === points.count;
  // P2 anisotropy: r2 (minor radius) + dir (major-axis) present ⇒ full-frame quat.
  // Per-splat, only when this splat's dir is nonzero (isotropic survivors keep 0).
  const perAniso = !!(points.r2 && points.dir);
  // r0 is still required as the global fallback when there's no per-splat radius.
  if (!perRadius && !(r0 > 0))
    throw new RangeError(`fitSplats: r0 must be > 0 (got ${r0})`);
  const ue = convention !== "raw";
  // Size (cmScale) and handedness (convention) are ORTHOGONAL: cmScale scales BOTH
  // conventions; `ue` only decides the one-axis mirror. (Was `ue ? cmScale : 1`, which
  // silently dropped any requested size for web/raw exports — a target-size in a web
  // viewer became a dimensionless cloud. #345. Pass cmScale=1 for identity/unit scale.)
  const scale = cmScale;
  const n = points.count | 0;

  const pos = new Float32Array(3 * n);
  const normal = new Float32Array(3 * n);
  const fdc = new Float32Array(3 * n);
  const rot = new Float32Array(4 * n);
  const scaleLog = new Float32Array(3 * n);
  const opac = new Float32Array(n);

  // Uniform log-scales (S0 global radius): in-plane r·scale, thin along the disc
  // normal. logit(0.95) ≈ 2.9444; renderers apply sigmoid/exp on read. Per-splat
  // radius/alpha (S1b) override these inside the loop when present.
  const lnR = perRadius ? 0 : Math.log(r0 * scale);
  const lnThin = perRadius ? 0 : Math.log(r0 * scale * thinEps);
  // Ceiling 0.9999 before logit: logit(1) is +Infinity, and an exact-1.0 alpha from
  // any caller (e.g. a sharp-end alphaBase) would write IEEE Inf into the .ply
  // opacity column. Same defensive discipline as the thinEps floor guard above.
  const opLogit = Math.log(
    Math.min(opacity, 0.9999) / (1 - Math.min(opacity, 0.9999)),
  );
  const logit = (a0) => {
    const a = Math.min(a0, 0.9999);
    return Math.log(a / (1 - a));
  };

  for (let i = 0; i < n; i++) {
    const j = 3 * i;
    const px = points.pos[j],
      py = points.pos[j + 1],
      pz = points.pos[j + 2];
    const nx = points.normal[j],
      ny = points.normal[j + 1],
      nz = points.normal[j + 2];

    // Mirror (UE) or identity (raw), in the DATA only.
    const pxp = ue ? -px : px;
    const nxp = ue ? -nx : nx;
    const nrm = [nxp, ny, nz];

    pos[j] = pxp * scale;
    pos[j + 1] = py * scale;
    pos[j + 2] = pz * scale;

    normal[j] = nrm[0];
    normal[j + 1] = nrm[1];
    normal[j + 2] = nrm[2];

    // P2 anisotropy: if this splat has a nonzero major-axis dir, orient from the
    // FULL frame (mirrored dir + normal) and use (r_max, r_min). Else the S0/S1b
    // path — orientation fresh from the (mirrored) normal only, no quaternion
    // reflection. rot = [w,x,y,z]; scaleLog = [major, minor, thin(∝r_min)].
    let q, rMajLog, rMinLog, rThinLog;
    let ax = 0,
      ay = 0,
      az = 0;
    if (perAniso) {
      const dx = points.dir[j],
        dy = points.dir[j + 1],
        dz = points.dir[j + 2];
      if (dx || dy || dz) {
        ax = ue ? -dx : dx; // mirror dir exactly like the normal
        ay = dy;
        az = dz;
      }
    }
    if (ax || ay || az) {
      // Right-handed frame from mirrored (dir', n'): u = dir'⊥w, w = n', v = w×u.
      const wx = nrm[0],
        wy = nrm[1],
        wz = nrm[2];
      let ux = ax,
        uy = ay,
        uz = az;
      const uw = ux * wx + uy * wy + uz * wz;
      ux -= uw * wx;
      uy -= uw * wy;
      uz -= uw * wz;
      const ul = Math.hypot(ux, uy, uz);
      if (ul < 1e-8) {
        q = quatFromZTo(nrm); // dir ∥ normal (degenerate) → normal-only
      } else {
        ux /= ul;
        uy /= ul;
        uz /= ul;
        const vx = wy * uz - wz * uy,
          vy = wz * ux - wx * uz,
          vz = wx * uy - wy * ux;
        q = quatFromFrame(ux, uy, uz, vx, vy, vz, wx, wy, wz);
      }
      const rMaj = points.radius[i],
        rMin = points.r2[i];
      rMajLog = Math.log(rMaj * scale);
      rMinLog = Math.log(rMin * scale);
      rThinLog = Math.log(rMin * scale * thinEps); // thin anchored to r_min (§3.1b)
    } else {
      q = quatFromZTo(nrm);
      const rIn = perRadius ? Math.log(points.radius[i] * scale) : lnR;
      rMajLog = rIn;
      rMinLog = rIn;
      rThinLog = perRadius
        ? Math.log(points.radius[i] * scale * thinEps)
        : lnThin;
    }
    rot[4 * i] = q[0];
    rot[4 * i + 1] = q[1];
    rot[4 * i + 2] = q[2];
    rot[4 * i + 3] = q[3];
    scaleLog[j] = rMajLog;
    scaleLog[j + 1] = rMinLog;
    scaleLog[j + 2] = rThinLog;

    fdc[j] = enc(points.albedo[j]);
    fdc[j + 1] = enc(points.albedo[j + 1]);
    fdc[j + 2] = enc(points.albedo[j + 2]);

    opac[i] = perAlpha ? logit(points.alpha[i]) : opLogit;
  }

  return { count: n, pos, normal, fdc, opacity: opac, scaleLog, rot };
}

// §S-6 σ calibration ("rings too large"): the Auto solver calibrates COVERING-DISC
// radii (hard-disc coverage/overdraw on the fidelity sample), but the .ply stores
// σ — and a Gaussian at near-opaque alpha renders visibly out to ~2σ, so every
// splat reads ~2× fatter than the solved disc. Measured (Tourbillon 600k, EWA
// viewer A/B 2026-07-23): σ=r mush → σ=r/1.5 clean detail gain, σ=r/2 crispest
// with only pinhole-scale silhouette gaps. The shrink must run AFTER the Auto φ
// solve (both solvers read scaleLog as the disc radius — shrinking first just
// inflates φ back by K).
// Sharpness-slider mapping: radiusScale 0.8 (sharpest)→2.0, 1.2 (default)→1.5,
// 2.4 (softest)→1.0 (the legacy look — haze is the point there).
export function sigmaShrinkFor(radiusScale) {
  const rs = Number.isFinite(radiusScale) ? radiusScale : 1.2;
  const k =
    rs <= 1.2 ? 1.5 + (1.2 - rs) * 1.25 : 1.5 - ((rs - 1.2) * 0.5) / 1.2;
  return Math.min(2, Math.max(1, k));
}

// Uniformly shrink the written Gaussian scales by k (all three axes — aniso
// ellipses shrink proportionally). Positions/opacity/rotation untouched.
export function shrinkSigma(splats, k) {
  if (!(k > 1.0001) || !splats?.scaleLog) return splats;
  const dl = Math.log(k);
  const s = splats.scaleLog;
  for (let i = 0; i < s.length; i++) s[i] -= dl;
  return splats;
}

// §S2b calibration swatch — a 3×3 grid of large flat +Z discs at known albedos,
// as a points bundle to run through the SHIPPING fitSplats + writeSplatPly (not a
// bypass). Pins the whole color chain: rgb2sh0(0.5)=0 exactly, so a mid-gray disc
// rendering non-mid in a viewer proves an importer-side transform (sRGB double-
// gamma). Colors: mid/dark/light gray, pure R/G/B, white, black, 0.5-sat cyan.
export function swatchPoints() {
  const colors = [
    [0.5, 0.5, 0.5],
    [0.25, 0.25, 0.25],
    [0.75, 0.75, 0.75],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 1],
    [0, 0, 0],
    [0.5, 1, 1], // 0.5-saturation cyan
  ];
  const n = colors.length;
  const pos = new Float32Array(3 * n);
  const normal = new Float32Array(3 * n);
  const albedo = new Float32Array(3 * n);
  const radius = new Float32Array(n);
  const alpha = new Float32Array(n);
  const spacing = 2, // world units between disc centers
    R = 0.7; // disc radius (→ ~70 cm at cmScale 100)
  for (let i = 0; i < n; i++) {
    const col = i % 3,
      row = (i / 3) | 0,
      j = 3 * i;
    pos[j] = (col - 1) * spacing; // left..right
    pos[j + 1] = (1 - row) * spacing; // top row highest (Y up)
    pos[j + 2] = 0;
    normal[j + 2] = 1; // face +Z
    albedo[j] = colors[i][0];
    albedo[j + 1] = colors[i][1];
    albedo[j + 2] = colors[i][2];
    radius[i] = R;
    alpha[i] = 0.98;
  }
  return { count: n, pos, normal, albedo, radius, alpha };
}

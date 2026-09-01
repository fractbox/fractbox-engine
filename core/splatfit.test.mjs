import { test } from "node:test";
import assert from "node:assert/strict";
import {
  SH_C0,
  rgb2sh0,
  sh02rgb,
  srgb2lin,
  quatFromZTo,
  quatFromFrame,
  fitSplats,
  swatchPoints,
} from "./splatfit.js";

// Rotation matrix from quaternion [w,x,y,z] (row-major 3×3 as number[9]).
function quatToMat3(q) {
  const [w, x, y, z] = q;
  return [
    1 - 2 * (y * y + z * z),
    2 * (x * y - w * z),
    2 * (x * z + w * y),
    2 * (x * y + w * z),
    1 - 2 * (x * x + z * z),
    2 * (y * z - w * x),
    2 * (x * z - w * y),
    2 * (y * z + w * x),
    1 - 2 * (x * x + y * y),
  ];
}
const mulZ = (R) => [R[2], R[5], R[8]]; // R·[0,0,1] = third column
function det3(R) {
  return (
    R[0] * (R[4] * R[8] - R[5] * R[7]) -
    R[1] * (R[3] * R[8] - R[5] * R[6]) +
    R[2] * (R[3] * R[7] - R[4] * R[6])
  );
}
const close = (a, b, e = 1e-6) => Math.abs(a - b) <= e;
const closeV = (a, b, e = 1e-6) => a.every((v, i) => close(v, b[i], e));

test("SH0 encode/decode round-trips; pins no-de-gamma direction", () => {
  for (const c of [0, 0.5, 1, 0.18, 0.73])
    assert.ok(close(sh02rgb(rgb2sh0(c)), c, 1e-9), `round-trip ${c}`);
  // These pin SH_C0 AND the sRGB-in (no linearize) direction — a de-gamma step
  // would move them.
  assert.equal(rgb2sh0(0.5), 0);
  assert.ok(close(rgb2sh0(1), 0.5 / SH_C0, 1e-9));
  assert.ok(close(rgb2sh0(1), 1.7724538509, 1e-6));
});

test("quatFromZTo maps +Z to n (unit, det +1) across generic + degenerate n", () => {
  const norm = (v) => {
    const l = Math.hypot(...v);
    return v.map((x) => x / l);
  };
  const cases = [
    [0, 0, 1], // identity
    [1, 0, 0],
    [0, 1, 0],
    norm([0.3, -0.7, 0.5]),
    norm([-0.9, 0.1, -0.2]),
    [0, 0, -1], // exact antiparallel (guard branch)
    norm([1e-6, 0, -1 + 1e-6]), // near-antiparallel, just outside |q|<1e-8 guard
  ];
  for (const n of cases) {
    const q = quatFromZTo(n);
    assert.ok(q.every(Number.isFinite), `finite quat for ${n}`);
    assert.ok(close(Math.hypot(...q), 1, 1e-6), `unit quat for ${n}`);
    const R = quatToMat3(q);
    assert.ok(close(det3(R), 1, 1e-5), `det +1 for ${n}`);
    // Antiparallel: any horizontal axis is valid; only assert R·ẑ = n there.
    assert.ok(closeV(mulZ(R), n, 1e-5), `R·ẑ ≈ n for ${n} (got ${mulZ(R)})`);
  }
});

test("quatFromFrame P2: rotation columns = [u,v,w] (R·x̂=u, R·ŷ=v, R·ẑ=w), unit, det +1", () => {
  const nrm = (v) => {
    const l = Math.hypot(...v);
    return v.map((x) => x / l);
  };
  const col = (R, c) => [R[c], R[c + 3], R[c + 6]]; // column c of row-major 3×3
  // build a right-handed frame from a generic normal: u ⊥ w, v = w × u
  const frame = (w) => {
    let u = [1, 0, 0];
    const uw = u[0] * w[0] + u[1] * w[1] + u[2] * w[2];
    u = nrm([u[0] - uw * w[0], u[1] - uw * w[1], u[2] - uw * w[2]]);
    const v = [
      w[1] * u[2] - w[2] * u[1],
      w[2] * u[0] - w[0] * u[2],
      w[0] * u[1] - w[1] * u[0],
    ];
    return [u, v, w];
  };
  const cases = [
    [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ], // identity
    [
      [0, 1, 0],
      [0, 0, 1],
      [1, 0, 0],
    ], // cyclic
    frame(nrm([0.3, -0.7, 0.5])),
    frame(nrm([-0.9, 0.1, -0.2])),
  ];
  for (const [u, v, w] of cases) {
    const q = quatFromFrame(
      u[0],
      u[1],
      u[2],
      v[0],
      v[1],
      v[2],
      w[0],
      w[1],
      w[2],
    );
    assert.ok(close(Math.hypot(...q), 1, 1e-6), "unit quat");
    const R = quatToMat3(q);
    assert.ok(close(det3(R), 1, 1e-5), "det +1");
    assert.ok(closeV(col(R, 0), u, 1e-5), "col x = u");
    assert.ok(closeV(col(R, 1), v, 1e-5), "col y = v");
    assert.ok(closeV(col(R, 2), w, 1e-5), "col z = w");
  }
});

test("fitSplats P2 aniso: r2+dir → full-frame quat (major=dir, thin ∝ r_min); mirror keeps det +1", () => {
  const col = (R, c) => [R[c], R[c + 3], R[c + 6]];
  // raw (no mirror): major axis +x, normal +z, r_max 0.1 / r_min 0.02
  const pts = {
    count: 1,
    pos: Float32Array.of(0, 0, 0),
    normal: Float32Array.of(0, 0, 1),
    albedo: Float32Array.of(0.5, 0.5, 0.5),
    radius: Float32Array.of(0.1),
    r2: Float32Array.of(0.02),
    dir: Float32Array.of(1, 0, 0),
  };
  const out = fitSplats(pts, { convention: "raw", r0: 0.1 });
  assert.ok(close(out.scaleLog[0], Math.log(0.1), 1e-6), "major = ln r_max");
  assert.ok(close(out.scaleLog[1], Math.log(0.02), 1e-6), "minor = ln r_min");
  assert.ok(close(out.scaleLog[2], Math.log(0.02 * 0.1), 1e-6), "thin ∝ r_min");
  let R = quatToMat3([out.rot[0], out.rot[1], out.rot[2], out.rot[3]]);
  assert.ok(closeV(col(R, 0), [1, 0, 0], 1e-5), "major axis → dir");
  assert.ok(closeV(col(R, 2), [0, 0, 1], 1e-5), "thin axis → normal");
  // zero dir ⇒ isotropic passthrough (per-radius path, in-plane axes equal)
  const iso = fitSplats(
    { ...pts, dir: Float32Array.of(0, 0, 0) },
    { convention: "raw", r0: 0.1 },
  );
  assert.ok(
    close(iso.scaleLog[0], iso.scaleLog[1], 1e-9),
    "iso: in-plane radii equal",
  );
  // UE mirror with nx≠0: frame built from mirrored (dir', n') stays right-handed (det +1)
  const norm3 = (v) => {
    const l = Math.hypot(...v);
    return v.map((x) => x / l);
  };
  const nrm = norm3([0.6, 0, 0.8]);
  const dir = norm3([-0.8, 0, 0.6]); // ⊥ normal
  const uePts = {
    count: 1,
    pos: Float32Array.of(0.1, 0.2, 0.3),
    normal: Float32Array.from(nrm),
    albedo: Float32Array.of(0.5, 0.5, 0.5),
    radius: Float32Array.of(0.1),
    r2: Float32Array.of(0.03),
    dir: Float32Array.from(dir),
  };
  const ue = fitSplats(uePts, { convention: "ue", cmScale: 100, r0: 0.1 });
  R = quatToMat3([ue.rot[0], ue.rot[1], ue.rot[2], ue.rot[3]]);
  assert.ok(
    close(det3(R), 1, 1e-5),
    "mirrored frame still det +1 (RH, no reflection)",
  );
  assert.ok(
    closeV(col(R, 2), [-nrm[0], nrm[1], nrm[2]], 1e-5),
    "thin axis → mirrored normal",
  );
  assert.ok(
    closeV(col(R, 0), [-dir[0], dir[1], dir[2]], 1e-5),
    "major axis → mirrored dir",
  );
});

test("fitSplats: chirality by numbers (R·ẑ = mirrored normal, det +1, pos mirrored)", () => {
  // pre-mirror normals with n.x ≠ 0 so a pre-mirror-normal bug fails.
  const n = [
    0.6,
    0.0,
    0.8, //
    -0.5,
    0.5,
    0.7071, //
    0.8,
    -0.6,
    0.0,
  ];
  const norm3 = (a, j) => {
    const l = Math.hypot(a[j], a[j + 1], a[j + 2]);
    return [a[j] / l, a[j + 1] / l, a[j + 2] / l];
  };
  const points = {
    count: 3,
    pos: Float32Array.from([1, 2, 3, -4, 5, -6, 0.1, 0.2, 0.3]),
    normal: Float32Array.from(n),
    albedo: Float32Array.from([0.2, 0.4, 0.6, 0.9, 0.1, 0.5, 0.5, 0.5, 0.5]),
  };
  const cm = 100;
  const out = fitSplats(points, { convention: "ue", cmScale: cm, r0: 0.05 });
  for (let i = 0; i < 3; i++) {
    const j = 3 * i;
    const nn = norm3(n, j);
    const nMirror = [-nn[0], nn[1], nn[2]]; // n' the UE transform produces
    const q = [
      out.rot[4 * i],
      out.rot[4 * i + 1],
      out.rot[4 * i + 2],
      out.rot[4 * i + 3],
    ];
    const R = quatToMat3(q);
    assert.ok(close(det3(R), 1, 1e-5), `det +1 splat ${i}`);
    assert.ok(closeV(mulZ(R), nMirror, 1e-5), `R·ẑ ≈ n′ splat ${i}`);
    // position mirrored on x and cm-scaled
    assert.ok(
      close(out.pos[j], -points.pos[j] * cm, 1e-3),
      `pos.x mirror ${i}`,
    );
    assert.ok(close(out.pos[j + 1], points.pos[j + 1] * cm, 1e-3));
    assert.ok(close(out.pos[j + 2], points.pos[j + 2] * cm, 1e-3));
  }
});

test("fitSplats: raw convention is identity handedness; cmScale scales size in both (#345); opacity logit", () => {
  const points = {
    count: 1,
    pos: Float32Array.from([1, 2, 3]),
    normal: Float32Array.from([0, 0, 1]),
    albedo: Float32Array.from([0.5, 0.5, 0.5]),
  };
  const raw = fitSplats(points, { convention: "raw", cmScale: 100, r0: 0.1 });
  assert.deepEqual([...raw.pos], [100, 200, 300]); // scaled by cmScale, NOT mirrored (#345)
  const rawUnit = fitSplats(points, { convention: "raw", cmScale: 1, r0: 0.1 });
  assert.deepEqual([...rawUnit.pos], [1, 2, 3]); // cmScale=1 → identity/unit scale
  const ue = fitSplats(points, { convention: "ue", cmScale: 100, r0: 0.1 });
  assert.deepEqual([...ue.pos], [-100, 200, 300]);
  // logit(0.95) ≈ 2.9444
  assert.ok(close(ue.opacity[0], Math.log(0.95 / 0.05), 1e-6));
  assert.ok(close(ue.opacity[0], 2.944438979, 1e-6));
});

test("fitSplats R1: thinEps hard floor ≥ 0.01 (below is the flat-splat danger zone)", () => {
  const p = {
    count: 1,
    pos: Float32Array.of(0, 0, 0),
    normal: Float32Array.of(0, 0, 1),
    albedo: Float32Array.of(0.5, 0.5, 0.5),
  };
  for (const bad of [0.005, 0, -0.1, NaN])
    assert.throws(
      () => fitSplats(p, { r0: 0.1, thinEps: bad }),
      RangeError,
      `thinEps ${bad} rejected`,
    );
  // 0.01 (the floor) and the default 0.1 (thinEps omitted) both pass
  assert.doesNotThrow(() => fitSplats(p, { r0: 0.1, thinEps: 0.01 }));
  const def = fitSplats(p, { convention: "raw", r0: 0.1 }); // thinEps default 0.1
  assert.ok(
    close(def.scaleLog[2], Math.log(0.1 * 0.1), 1e-6),
    "default thinEps 0.1 → lnThin unchanged",
  );
});

test("fitSplats: r0 must be positive", () => {
  const p = {
    count: 1,
    pos: Float32Array.of(0, 0, 0),
    normal: Float32Array.of(0, 0, 1),
    albedo: Float32Array.of(0.5, 0.5, 0.5),
  };
  assert.throws(() => fitSplats(p, { r0: 0 }), RangeError);
});

test("fitSplats S1b: per-splat radius/alpha → scaleLog = ln(r·scale), opacity = logit(alpha)", () => {
  const points = {
    count: 3,
    pos: Float32Array.from([1, 0, 0, 0, 1, 0, 0, 0, 1]),
    normal: Float32Array.from([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    albedo: Float32Array.from([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
    radius: Float32Array.from([0.02, 0.05, 0.1]),
    alpha: Float32Array.from([0.9, 0.6, 0.3]),
  };
  const cm = 100,
    thinEps = 0.1;
  const out = fitSplats(points, {
    convention: "ue",
    cmScale: cm,
    thinEps,
    r0: 0.01,
  });
  for (let i = 0; i < 3; i++) {
    const j = 3 * i,
      r = points.radius[i],
      a = points.alpha[i];
    assert.ok(
      close(out.scaleLog[j], Math.log(r * cm), 1e-6),
      `in-plane lnR splat ${i}`,
    );
    assert.ok(close(out.scaleLog[j + 1], Math.log(r * cm), 1e-6));
    assert.ok(
      close(out.scaleLog[j + 2], Math.log(r * cm * thinEps), 1e-6),
      `thin lnThin splat ${i}`,
    );
    assert.ok(
      close(out.opacity[i], Math.log(a / (1 - a)), 1e-6),
      `opacity logit splat ${i}`,
    );
  }
  // distinct per splat — not the shared global
  assert.ok(out.scaleLog[0] !== out.scaleLog[3], "radii differ per splat");
  assert.ok(out.opacity[0] !== out.opacity[1], "alphas differ per splat");
});

test("fitSplats S1b: per-splat radius makes r0 optional; absent arrays = S0 global path", () => {
  const base = {
    count: 1,
    pos: Float32Array.of(0, 0, 0),
    normal: Float32Array.of(0, 0, 1),
    albedo: Float32Array.of(0.5, 0.5, 0.5),
  };
  // radius present → no r0 required
  const withR = {
    ...base,
    radius: Float32Array.of(0.03),
    alpha: Float32Array.of(0.8),
  };
  assert.doesNotThrow(() => fitSplats(withR, {}));
  const o = fitSplats(withR, { convention: "raw" });
  assert.ok(
    close(o.scaleLog[0], Math.log(0.03), 1e-6),
    "per-splat radius honored w/o r0",
  );
  // no radius/alpha → exactly the S0 global constants (bit-identical path)
  const g = fitSplats(base, { convention: "raw", r0: 0.1, thinEps: 0.1 });
  assert.ok(close(g.scaleLog[0], Math.log(0.1), 1e-6), "global lnR"); // f32 stored
  assert.ok(close(g.scaleLog[2], Math.log(0.01), 1e-6), "global lnThin");
  assert.ok(
    close(g.opacity[0], Math.log(0.95 / 0.05), 1e-6),
    "global opacity logit",
  );
});

test("fitSplats degamma: off = display-sRGB (default), on = sRGB→linear before SH0", () => {
  const points = {
    count: 3,
    pos: Float32Array.from([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    normal: Float32Array.from([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    // mid-gray, white, black — endpoints pin the curve
    albedo: Float32Array.from([0.5, 0.5, 0.5, 1, 1, 1, 0, 0, 0]),
  };
  // default (no degamma) stores display-sRGB: mid-gray → f_dc 0 exactly
  const off = fitSplats(points, { convention: "raw", r0: 0.1 });
  assert.ok(close(off.fdc[0], 0, 1e-6), "sRGB mid-gray f_dc = 0");
  // degamma linearizes first: mid-gray 0.5 → srgb2lin(0.5) ≈ 0.214 → f_dc < 0
  const on = fitSplats(points, { convention: "raw", r0: 0.1, degamma: true });
  assert.ok(close(on.fdc[0], rgb2sh0(srgb2lin(0.5)), 1e-6), "degamma mid-gray");
  assert.ok(
    on.fdc[0] < -0.9 && on.fdc[0] > -1.1,
    "degamma pulls mid-gray negative",
  );
  // 0 and 1 are curve fixed points ⇒ unchanged by degamma (white/black pins hold)
  assert.ok(
    close(on.fdc[3], rgb2sh0(1), 1e-6),
    "white unchanged (srgb2lin(1)=1)",
  );
  assert.ok(
    close(on.fdc[6], rgb2sh0(0), 1e-6),
    "black unchanged (srgb2lin(0)=0)",
  );
});

test("swatchPoints → fitSplats: §S2b 9-disc calibration, known SH0 pins + round-trip", () => {
  const pts = swatchPoints();
  assert.equal(pts.count, 9);
  const out = fitSplats(pts, { convention: "ue", cmScale: 100, r0: 0.7 });
  assert.equal(out.count, 9);
  // The load-bearing pin: mid-gray (swatch 0) → f_dc 0 exactly. An importer that
  // re-applies sRGB (double gamma) would move it off 0.
  assert.ok(close(out.fdc[0], 0, 1e-6), "mid-gray f_dc = 0");
  assert.ok(close(out.fdc[6 * 3], rgb2sh0(1), 1e-6), "white f_dc = rgb2sh0(1)");
  assert.ok(close(out.fdc[7 * 3], rgb2sh0(0), 1e-6), "black f_dc = rgb2sh0(0)");
  // every swatch's f_dc decodes back to its exact albedo (writer is faithful)
  for (let i = 0; i < 9; i++)
    for (let c = 0; c < 3; c++)
      assert.ok(
        close(sh02rgb(out.fdc[3 * i + c]), pts.albedo[3 * i + c], 1e-6),
        `swatch ${i} chan ${c}`,
      );
});

test("sigmaShrinkFor: slider mapping — sharp 2.0, default 1.5, soft 1.0", async () => {
  const { sigmaShrinkFor, shrinkSigma } = await import("./splatfit.js");
  const close = (a, b, t = 1e-9) => Math.abs(a - b) <= t;
  assert.ok(close(sigmaShrinkFor(0.8), 2.0), "sharpest → 2.0");
  assert.ok(close(sigmaShrinkFor(1.2), 1.5), "default → 1.5");
  assert.ok(close(sigmaShrinkFor(2.4), 1.0), "softest → 1.0 (legacy σ=r)");
  assert.ok(close(sigmaShrinkFor(undefined), 1.5), "no rs → default");
  assert.ok(
    sigmaShrinkFor(0.1) <= 2 && sigmaShrinkFor(9) >= 1,
    "clamped [1,2]",
  );
  // shrinkSigma: uniform −ln k on every scaleLog word; everything else untouched
  const pts = swatchPoints();
  const out = fitSplats(pts, { convention: "ue", cmScale: 100, r0: 0.7 });
  const before = Float32Array.from(out.scaleLog);
  const pos0 = Float32Array.from(out.pos);
  shrinkSigma(out, 1.5);
  const dl = Math.log(1.5);
  for (let i = 0; i < before.length; i++)
    assert.ok(
      close(out.scaleLog[i], before[i] - dl, 1e-6),
      `scaleLog[${i}] −ln k`,
    );
  assert.deepEqual([...out.pos], [...pos0], "positions untouched");
  // k=1 (soft end) is the identity — legacy output byte-identical
  const same = Float32Array.from(out.scaleLog);
  shrinkSigma(out, 1.0);
  assert.deepEqual([...out.scaleLog], [...same], "k=1 identity");
});

test("fitSplats: out-of-range albedo clamps to [0,1] before SH0 (P1.5 finding — f16 MRT negatives; degamma NaN guard)", () => {
  const p = {
    count: 1,
    pos: Float32Array.from([0, 0, 0]),
    normal: Float32Array.from([0, 0, 1]),
    albedo: Float32Array.from([-0.3, 1.4, 0.5]),
  };
  const out = fitSplats(p, { convention: "raw", r0: 0.1 });
  assert.ok(
    close(out.fdc[0], rgb2sh0(0)),
    "negative albedo → rgb2sh0(0), not below",
  );
  assert.ok(close(out.fdc[1], rgb2sh0(1)), ">1 albedo → rgb2sh0(1)");
  assert.ok(close(out.fdc[2], rgb2sh0(0.5)), "in-range untouched");
  const dg = fitSplats(p, { convention: "raw", r0: 0.1, degamma: true });
  assert.ok(
    [...dg.fdc].every(Number.isFinite),
    "degamma of negative albedo must not write NaN",
  );
  assert.ok(
    close(dg.fdc[0], rgb2sh0(0)),
    "degamma(clamp(-0.3)) = linear black",
  );
});

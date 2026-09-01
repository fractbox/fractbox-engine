// Rung-1 fidelity harness for the splat export (SPLAT_GAP_IMPL.md §P0). Five
// geometric numbers, computable in Node from the capture G-buffer + the fitted
// splats, that stand in for SOTA R4 (live-vs-splat image SSIM) — there is no
// in-app splat renderer, so per the roadmap's Rung-1 note the CI-able substitute
// is geometric fidelity. These numbers make every later rung (anisotropy, flow
// resampling, compression) falsifiable without a viewer.
//
// PURE: no imports, no DOM, no deps. `de` / `albedoAt` / `aoScale` arrive as
// arguments (the cpu.js closures + the exported capture AO fn), so the module is
// trivially testable with synthetic closures (a unit-sphere `de`, a constant
// albedo). All definitions are FROZEN here — later phases compare against these
// numbers, so they must not drift (SPLAT_GAP_IMPL R-5).

// Central-difference gradient step: the capture eps rule (splatcapture.js
// `captureEps`), restated here — h = 2·eps. This module is PURE by contract
// (see the header) and splatcapture.js pulls in cpu/evaluate/camera/stability,
// so this stays a copy rather than an import; it MUST be kept in step with
// captureEps, which is pinned by a test ("splatmetrics epsFor mirrors
// captureEps"). A metric that measures a capture with a different eps than the
// capture itself used is measuring the wrong thing — so both floor terms are
// mirrored here: `epsFloor` (#496, the framing layer's absolute floor) and
// `epsMeasured` (#507, the DE's measured convergence floor), which is what
// keeps the harness grading a cloud at the epsilon it was actually captured at
// instead of a stale default. Frames carrying neither are unchanged.
export function epsFor(frame) {
  return Math.max(
    3e-4 * frame.radius,
    frame.epsFloor ?? 0,
    frame.epsMeasured ?? 0,
    1e-5,
  );
}

// Seeded 32-bit LCG (matches makeHitReservoir): reproducible center subsampling.
function lcg(seed) {
  let s = seed >>> 0 || 1;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296; // [0, 1)
  };
}

// Up to `maxCenters` center indices, seeded-uniform. All indices when n ≤ cap.
function centerIndices(n, maxCenters, seed) {
  if (n <= maxCenters) {
    const all = new Array(n);
    for (let i = 0; i < n; i++) all[i] = i;
    return all;
  }
  const rnd = lcg(seed);
  const out = new Array(maxCenters);
  for (let c = 0; c < maxCenters; c++) out[c] = Math.floor(rnd() * n);
  return out;
}

// 1. onSurface — fraction of splat centers with |de(center)| < k·r0. S2 pins 100%.
export function metricOnSurface(de, points, r0, k = 1) {
  const n = points.count;
  if (!n) return 0;
  const thr = k * r0;
  let hit = 0;
  for (let i = 0; i < n; i++) {
    const j = 3 * i;
    if (Math.abs(de(points.pos[j], points.pos[j + 1], points.pos[j + 2])) < thr)
      hit++;
  }
  return hit / n;
}

// Spatial hash of splat centers into a uniform grid at `cell` (Euclidean cover):
// with cell = max(radius), every splat that could cover a hit lies in the hit's
// 27-cell neighborhood. Local reimplementation — splatmetrics must not reach into
// reducePoints internals (R-5). Numeric packed key over a bounded bbox.
function centerGrid(points, cell) {
  const n = points.count;
  let xmin = Infinity,
    ymin = Infinity,
    zmin = Infinity,
    xmax = -Infinity,
    ymax = -Infinity,
    zmax = -Infinity;
  for (let i = 0; i < n; i++) {
    const j = 3 * i;
    const x = points.pos[j],
      y = points.pos[j + 1],
      z = points.pos[j + 2];
    if (x < xmin) xmin = x;
    if (y < ymin) ymin = y;
    if (z < zmin) zmin = z;
    if (x > xmax) xmax = x;
    if (y > ymax) ymax = y;
    if (z > zmax) zmax = z;
  }
  const inv = 1 / cell;
  const Wx = Math.max(1, Math.floor((xmax - xmin) * inv) + 1);
  const Wy = Math.max(1, Math.floor((ymax - ymin) * inv) + 1);
  const Wz = Math.max(1, Math.floor((zmax - zmin) * inv) + 1);
  const idx = (x, xmn, W) =>
    Math.min(W - 1, Math.max(0, Math.floor((x - xmn) * inv)));
  const cellOf = (x, y, z) => [
    idx(x, xmin, Wx),
    idx(y, ymin, Wy),
    idx(z, zmin, Wz),
  ];
  // Injective ONLY for in-range indices — the neighbor probe (coverScan) must
  // skip out-of-[0,W) cells, else negative/overflow offsets alias other buckets
  // and the same cell gets counted several times (overdraw inflation bug).
  const key = (ix, iy, iz) => ix + iy * Wx + iz * Wx * Wy;
  const map = new Map();
  for (let i = 0; i < n; i++) {
    const j = 3 * i;
    const [ix, iy, iz] = cellOf(
      points.pos[j],
      points.pos[j + 1],
      points.pos[j + 2],
    );
    const k = key(ix, iy, iz);
    let arr = map.get(k);
    if (!arr) map.set(k, (arr = []));
    arr.push(i);
  }
  return { map, cellOf, key, Wx, Wy, Wz };
}

// Shared scan: per sample hit, count splats (from the 27-cell neighborhood) whose
// per-splat radius covers it (Euclidean dist ≤ radius[i]). Returns coverage
// (fraction of hits covered by ≥1 splat) + overdraw (mean covering count).
function coverScan(sample, points) {
  if (!sample || !sample.count || !points.count)
    return { coverage: 0, overdraw: 0 };
  const rad = points.radius;
  if (!rad)
    throw new Error(
      "splatmetrics: coverage/overdraw need points.radius (S1b per-splat)",
    );
  let maxR = 0;
  for (let i = 0; i < points.count; i++) if (rad[i] > maxR) maxR = rad[i];
  const cell = maxR > 0 ? maxR : 1e-6;
  const { map, cellOf, key, Wx, Wy, Wz } = centerGrid(points, cell);
  let covered = 0,
    overSum = 0;
  const m = sample.count;
  for (let h = 0; h < m; h++) {
    const j = 3 * h;
    const hx = sample.pos[j],
      hy = sample.pos[j + 1],
      hz = sample.pos[j + 2];
    const [cx, cy, cz] = cellOf(hx, hy, hz);
    let count = 0;
    for (let dz = -1; dz <= 1; dz++) {
      const nz = cz + dz;
      if (nz < 0 || nz >= Wz) continue;
      for (let dy = -1; dy <= 1; dy++) {
        const ny = cy + dy;
        if (ny < 0 || ny >= Wy) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const nx = cx + dx;
          if (nx < 0 || nx >= Wx) continue;
          const arr = map.get(key(nx, ny, nz));
          if (!arr) continue;
          for (let t = 0; t < arr.length; t++) {
            const i = arr[t],
              o = 3 * i;
            const ddx = hx - points.pos[o],
              ddy = hy - points.pos[o + 1],
              ddz = hz - points.pos[o + 2];
            if (ddx * ddx + ddy * ddy + ddz * ddz <= rad[i] * rad[i]) count++;
          }
        }
      }
    }
    if (count > 0) covered++;
    overSum += count;
  }
  return { coverage: covered / m, overdraw: overSum / m };
}

// 2. coverage — fraction of sample hits within a splat's in-plane radius (holes ⇒ <1).
export function metricCoverage(sample, points) {
  return coverScan(sample, points).coverage;
}

// 3. overdraw — mean number of splats covering a hit (opacity-heuristic sanity).
export function metricOverdraw(sample, points) {
  return coverScan(sample, points).overdraw;
}

// 4. normalAgreement — mean dot(splat.normal, ∇DE/|∇DE|) at up to maxCenters centers.
export function metricNormalAgreement(
  de,
  points,
  frame,
  maxCenters = 100_000,
  seed = 1,
) {
  const n = points.count;
  if (!n) return 0;
  const eps = epsFor(frame);
  const h = 2 * eps;
  const idxs = centerIndices(n, maxCenters, seed);
  let sum = 0,
    used = 0;
  for (const i of idxs) {
    const j = 3 * i;
    // Probe the gradient slightly OFF-surface along the splat's own normal
    // (+h), not exactly at the center: S-2 snapping puts centers ON the zero
    // set, where the central-difference taps straddle a thinner-than-h wall
    // symmetrically and cancel (Menger measured ≈0 agreement for perfectly
    // placed splats — a probe artifact, not a normal error). An eps-scale
    // offset to the claimed outside restores a well-posed sample; for
    // off-surface (un-snapped) centers it changes nothing material.
    const x = points.pos[j] + points.normal[j] * h,
      y = points.pos[j + 1] + points.normal[j + 1] * h,
      z = points.pos[j + 2] + points.normal[j + 2] * h;
    const gx = de(x + h, y, z) - de(x - h, y, z);
    const gy = de(x, y + h, z) - de(x, y - h, z);
    const gz = de(x, y, z + h) - de(x, y, z - h);
    const gl = Math.hypot(gx, gy, gz);
    if (gl < 1e-20) continue; // undefined gradient — skip
    sum +=
      (points.normal[j] * gx +
        points.normal[j + 1] * gy +
        points.normal[j + 2] * gz) /
      gl;
    used++;
  }
  return used ? sum / used : 0;
}

// 5. colorDrift — mean per-center max-channel |albedo_splat − albedo_reeval|, where
// albedo_reeval = albedoAt(center) · aoScale(...) when aoStrength>0 — reproducing the
// capture chain so this measures DRIFT, not the deliberately-baked AO. (colorDrift is
// the only coloring-dependent metric: its magnitude tracks the coloring mode's spatial
// sensitivity, and reducePoints averages albedo per cell, so drift is nonzero by
// construction — SPLAT_GAP_IMPL R-3. Pins use defaultColoring for reproducibility.)
export function metricColorDrift(
  de,
  albedoAt,
  aoScale,
  points,
  frame,
  aoStrength = 0,
  maxCenters = 100_000,
  seed = 1,
) {
  const n = points.count;
  if (!n) return 0;
  const eps = epsFor(frame);
  const idxs = centerIndices(n, maxCenters, seed);
  const rad = points.radius;
  let sum = 0;
  for (const i of idxs) {
    const j = 3 * i;
    const x = points.pos[j],
      y = points.pos[j + 1],
      z = points.pos[j + 2];
    const a = albedoAt(x, y, z, points.normal[j + 2]);
    let ao = 1;
    if (aoStrength > 0) {
      const r = rad ? rad[i] : 0.01 * frame.radius;
      ao = aoScale(
        de,
        x,
        y,
        z,
        points.normal[j],
        points.normal[j + 1],
        points.normal[j + 2],
        eps,
        r,
        aoStrength,
      );
    }
    const dr = Math.abs(points.albedo[j] - a[0] * ao);
    const dg = Math.abs(points.albedo[j + 1] - a[1] * ao);
    const db = Math.abs(points.albedo[j + 2] - a[2] * ao);
    sum += Math.max(dr, dg, db);
  }
  return sum / idxs.length;
}

// #536: onSurface asks whether |DE(center)| is small next to r0 — but
// computeR0 (splatcapture.js) is `radiusScale · diag / √hits`, with no eps
// term, so a crop can be REDUCED to a pitch finer than the eps its hits were
// actually found at. #583 closed the common case of that (raising a crop's
// iteration count tightens the DE's measured convergence floor so eps stays
// ahead of r0) but not every case: the object's OWN scale-relative floor
// (splatcapture.js EPS_SCALE·radius) is a hard bound no iteration count can
// push eps below, while r0 keeps shrinking without limit as a capture volume
// keeps shrinking — so an aggressive-enough crop still crosses over. Measured
// on dev post-#583 (Menger corner crop, magnification sweep against the
// object frame): the crossover lands around ×150 (eps 3.30e-4 vs r0 3.27e-4,
// onSurface 0.9943) and onSurface keeps degrading well past it (×1500: eps/r0
// = 11.1, onSurface 0.5624) — the exact "eps > r0, metric unverifiable"
// failure this issue describes, just at a tighter close-up than #583's own
// regression case. Floor r0 at the eps THIS metric's frame actually promises
// (epsFor, mirroring captureEps) — never below it, since asking the metric to
// verify membership finer than the field's own resolution isn't a real
// signal. This floors only the METRIC's reference: computeR0/reducePoints
// (the reduce/dedup pitch that shapes the actual point cloud) are untouched,
// so no capture-behavior baseline moves — only what onSurface compares
// against for frames whose eps already exceeds r0.
function metricR0(r0, frame) {
  return Math.max(r0, epsFor(frame));
}

// All five, in one pass over the shared inputs. de/albedoAt = cpu.js closures;
// aoScale = the exported capture AO fn; sample = raw-hit subsample (coverage/
// overdraw); points = the reduced survivors (on-surface/normal/color); frame for
// the eps rule; r0 the dedup cell. opts: { k=1, aoStrength=0, maxCenters=1e5, seed=1 }.
export function splatMetrics(
  { de, albedoAt, aoScale, sample, points, r0, frame },
  opts = {},
) {
  const { k = 1, aoStrength = 0, maxCenters = 100_000, seed = 1 } = opts;
  const { coverage, overdraw } = coverScan(sample, points);
  return {
    onSurface: metricOnSurface(de, points, metricR0(r0, frame), k),
    coverage,
    overdraw,
    normalAgreement: metricNormalAgreement(de, points, frame, maxCenters, seed),
    colorDrift: metricColorDrift(
      de,
      albedoAt,
      aoScale,
      points,
      frame,
      aoStrength,
      maxCenters,
      seed,
    ),
    counts: { centers: points.count, sampled: sample ? sample.count : 0 },
  };
}

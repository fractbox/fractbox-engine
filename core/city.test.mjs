// #634 — City shape leaf: the DE-soundness PROOFS the spec review demanded.
// Its blocker: "a 3×3 scan is enough" was asserted, not shown — nothing
// committed footprints to stay inside their own cell. The shipped invariant
// (leaves.js): both tiers have horizontal half-extent ≤ fp = 0.5·c·(1−f) with
// f clamped ≥ 0.05, so out-of-window buildings sit ≥ c + s/2 > c away
// HORIZONTALLY regardless of hashed height, and the scan seeds with the cap
// d = c. These gates prove it through the SHIPPED CPU twin (cpu.js LEAF_FNS —
// the 3-emitter mirror discipline's testable leg) against an independent
// clamp-to-closest-point box reference (no SDF formula shared with the
// implementation), in the style of d2leaves.test.mjs's hexGrid brute force.
// Run: node --test core/city.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { leafById, leafByKey, LEAVES } from "./leaves.js";
import { makeDE } from "./cpu.js";
import { buildWGSL, usesObjAux } from "./shader.js";
import { buildSceneFragGL } from "./shader_gl.js";
import { MAX_LEAF_PARAMS_INLINE } from "./limits.js";

const leafDE = (shapeParams) =>
  makeDE({
    name: "t",
    ops: [],
    iters: 1,
    deOption: 2,
    addC: false,
    objects: [
      {
        shapeId: 59,
        shapeParams,
        ops: [],
        iters: 1,
        transform: { origin: [0, 0, 0], uscale: 1, rot: [0, 0, 0] },
        combine: 0,
        blendK: 0,
      },
    ],
  });

// ── independent reference ────────────────────────────────────────────────────
// The height hash IS the leaf's contract (a pure function of (i, k, seed) —
// determinism across tiers), so the reference re-states it; the GEOMETRY is
// where independence lives: buildings as min/max corner boxes, distance via
// clamp-to-closest-point — not the length(max(q,0))+interior SDF form the
// implementation uses.
function cellHash(ix, iz, seed) {
  let h =
    (Math.imul(ix, 73856093) ^
      Math.imul(iz, 83492791) ^
      Math.imul(seed, 2654435761)) >>>
    0;
  h = (h ^ (h >>> 13)) >>> 0;
  h = Math.imul(h, 1274126177) >>> 0;
  h = (h ^ (h >>> 16)) >>> 0;
  return h;
}

const clampP = (P) => ({
  c: Math.min(2, Math.max(0.1, P[0])),
  f: Math.min(0.9, Math.max(0.05, P[1])),
  hmax: Math.min(3, Math.max(0.05, P[2])),
  vary: Math.min(1, Math.max(0, P[3])),
  det: Math.min(1, Math.max(0, P[4] ?? 0)),
  seed: Math.min(1023, Math.max(0, Math.round(P[5] ?? 0))),
});

// The two tier boxes of cell (ix, iz), as [min, max] corners.
function cellBoxes(ix, iz, Q) {
  const { c, f, hmax, vary, det, seed } = Q;
  const fp = 0.5 * c * (1 - f);
  const h = cellHash(ix, iz, seed);
  const bh = hmax * Math.max(1 - (vary * (h & 1023)) / 1023, 0.02);
  const h1 = bh * (0.45 + (0.25 * ((h >>> 10) & 255)) / 255);
  const fp2 = fp * (1 - det * (0.15 + (0.45 * ((h >>> 18) & 255)) / 255));
  const cx = (ix + 0.5) * c,
    cz = (iz + 0.5) * c;
  return [
    { lo: [cx - fp, 0, cz - fp], hi: [cx + fp, h1, cz + fp] },
    { lo: [cx - fp2, 0, cz - fp2], hi: [cx + fp2, bh, cz + fp2] },
  ];
}

const distToBox = (p, b) => {
  let s = 0;
  for (let a = 0; a < 3; a++) {
    const q = Math.max(b.lo[a], Math.min(p[a], b.hi[a]));
    s += (p[a] - q) * (p[a] - q);
  }
  return Math.sqrt(s);
};
const insideBox = (p, b) => p.every((v, a) => v > b.lo[a] && v < b.hi[a]);

// True distance to the union over a window of ±R cells around p's cell, plus
// the (global, closed-form) ground slab y ∈ [−0.1c, 0]. 0 when p is inside a
// solid. Valid as the GLOBAL true distance whenever the result < (R−1)·c —
// any building outside the window is ≥ (R−1)·c + s/2 away horizontally (the
// same projection argument the leaf's cap rests on, two rings further out).
function truthDist(p, Q, R = 3) {
  const { c } = Q;
  let t;
  let inside = false;
  const gLo = -0.1 * c;
  if (p[1] >= gLo && p[1] <= 0) {
    inside = true;
    t = 0;
  } else t = p[1] > 0 ? p[1] : gLo - p[1];
  const bi = Math.floor(p[0] / c),
    bk = Math.floor(p[2] / c);
  for (let i = bi - R; i <= bi + R; i++)
    for (let k = bk - R; k <= bk + R; k++)
      for (const b of cellBoxes(i, k, Q)) {
        if (insideBox(p, b)) inside = true;
        t = Math.min(t, distToBox(p, b));
      }
  return inside ? 0 : t;
}

function mulberry32(a) {
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ── registry pins ────────────────────────────────────────────────────────────

test("#634 registry: id 59, unbounded, EXACT, 6 params riding the objAux lane", () => {
  const l = leafByKey("city");
  assert.equal(l.id, 59, "reserved id (#634 session partition: 59-60)");
  assert.equal(leafById(59), l);
  assert.equal(l.unbounded, true, "infinite lattice must declare unbounded");
  assert.ok(!l.deApprox, "city ships as an EXACT bound (proven below)");
  assert.equal(l.params.length, 6, "Cell/Street/Height/Vary + Detail/Seed");
  assert.ok(
    l.params.length > MAX_LEAF_PARAMS_INLINE,
    "fat leaf: params 5-6 ride prm2 (the #627 objAux lane)",
  );
  assert.match(l.wgsl, /\bprm2\b/);
  assert.match(l.glsl, /\bprm2\b/);
  assert.equal(usesObjAux([59]), true);
});

test("#634 WGSL/GLSL bodies carry the same construction constants", () => {
  // The only executable leg is the CPU twin; this pins the two GPU strings to
  // the same hash/tier constants so a one-sided edit can't desync them.
  const l = leafByKey("city");
  for (const k of [
    "73856093",
    "83492791",
    "2654435761",
    "1274126177",
    "1023",
    "255",
    "13",
    "16",
    "10",
    "18",
    "0.45",
    "0.25",
    "0.15",
    "0.02",
    "0.05",
    "0.9",
    "0.1",
    "2.0",
    "3.0",
  ]) {
    assert.ok(l.wgsl.includes(k), `WGSL lost constant ${k}`);
    assert.ok(l.glsl.includes(k), `GLSL lost constant ${k}`);
  }
});

test("#634 emitters: the fat-leaf lane engages end to end for a city scene", () => {
  const wgsl = buildWGSL({ leaves: [59], ops: [] });
  assert.match(
    wgsl,
    /fn leaf_city\(p: vec3f, prm: vec4f, prm2: vec4f\) -> f32/,
  );
  assert.match(
    wgsl,
    /@group\(0\) @binding\(8\) var<storage, read> objAux : array<vec4f>;/,
  );
  const gl = buildSceneFragGL([
    {
      shapeId: 59,
      shapeParams: [0.5, 0.25, 0.8, 0.6, 0.5, 1],
      ops: [],
      iters: 1,
      transform: { origin: [0, 0, 0], uscale: 1, rot: [0, 0, 0] },
      combine: 0,
      blendK: 0,
    },
  ]);
  assert.match(gl, /float leaf_city\(vec3 p, vec4 prm, vec4 prm2\)/);
  assert.match(gl, /uObjPrimP2/);
});

// ── the DE-soundness proof ───────────────────────────────────────────────────

test("#634 brute force: DE never overestimates, and is EXACT where the cap doesn't bind", () => {
  // Dense sampling over several parameter sets (varying every param, incl.
  // det = 0 plain boxes and det = 1 full setbacks). For each outside sample:
  //   de ≤ truth       — the never-overestimate contract (facade piercing);
  //   de = truth       — when truth < 0.95·c, i.e. wherever the cell-pitch
  //                      cap can't be the min, the DE is the true distance
  //                      (the "exact, not deApprox" classification's proof).
  // Inside samples must report ≤ 0. Sample heights stay ≤ 1.3·hmax + c so the
  // ±3-cell truth window is globally valid (see truthDist).
  const SETS = [
    [0.5, 0.25, 0.8, 0.6, 0.5, 1],
    [0.4, 0.05, 1.1, 1.0, 1.0, 7], // narrowest streets the clamp allows
    [0.7, 0.6, 0.5, 0.3, 0.0, 300], // det 0: plain single boxes
    [1.1, 0.9, 2.0, 0.9, 0.7, 1023], // widest streets, near-max variance
  ];
  for (const P of SETS) {
    const Q = clampP(P);
    const de = leafDE(P);
    const rand = mulberry32(0x634);
    let outside = 0,
      insideN = 0,
      exactChecked = 0,
      worstOver = -Infinity;
    for (let n = 0; n < 30000; n++) {
      const p = [
        (rand() * 5 - 2.5) * Q.c + 0.0013,
        rand() * (1.3 * Q.hmax + Q.c) - 0.2 * Q.c,
        (rand() * 5 - 2.5) * Q.c + 0.0007,
      ];
      const t = truthDist(p, Q);
      const d = de(p[0], p[1], p[2]);
      if (t === 0) {
        insideN++;
        assert.ok(d <= 1e-9, `inside point reports clearance ${d}`);
        continue;
      }
      outside++;
      worstOver = Math.max(worstOver, d - t);
      assert.ok(
        d <= t + 1e-9,
        `OVERESTIMATE at ${p}: de ${d} > true ${t} (params ${P})`,
      );
      if (t < 0.95 * Q.c) {
        exactChecked++;
        assert.ok(
          Math.abs(d - t) <= 1e-9,
          `inexact under the cap at ${p}: de ${d} vs true ${t} (params ${P})`,
        );
      }
    }
    // The sweep must actually exercise all three regimes.
    assert.ok(outside > 5000 && insideN > 200 && exactChecked > 3000);
    assert.ok(worstOver <= 1e-9);
  }
});

test("#634 neighborhood contract: a taller neighbor across the street is seen; a cell-local DE would not see it", () => {
  // The review's exact failure mode: p hovers above its own SHORT building's
  // roofline near the shared border — the nearest surface is the TALL
  // neighbor's facade s/2 away, which any single-cell estimator misses (it
  // would report the much larger min(own building, ground, cap) and pierce
  // the facade). det = 0 so buildings are plain full-height boxes.
  const P = [0.6, 0.3, 1.2, 1.0, 0.0, 1];
  const Q = clampP(P);
  const de = leafDE(P);
  const H = (i, k) =>
    Q.hmax *
    Math.max(1 - (Q.vary * (cellHash(i, k, Q.seed) & 1023)) / 1023, 0.02);
  // Deterministic search for the steepest x-adjacent height step.
  let best = null;
  for (let i = -6; i < 6; i++)
    for (let k = -6; k <= 6; k++) {
      const diff = H(i + 1, k) - H(i, k);
      if (!best || diff > best.diff) best = { i, k, diff };
    }
  assert.ok(
    best.diff > 0.4 * Q.hmax,
    `no steep pair found (best ${best.diff}) — widen the search`,
  );
  const hA = H(best.i, best.k),
    hB = H(best.i + 1, best.k);
  const s = Q.f * Q.c,
    eps = 0.002;
  const borderX = (best.i + 1) * Q.c;
  const p = [borderX - eps, hA + 0.25 * (hB - hA), (best.k + 0.5) * Q.c];
  // True distance: straight across to B's facade (y is inside B's height
  // range, z is on B's centerline) = s/2 + eps.
  const d = de(p[0], p[1], p[2]);
  assert.ok(
    d <= s / 2 + eps + 1e-9,
    `missed the neighbor: de ${d} > ${s / 2 + eps}`,
  );
  assert.ok(d > 0, "p is in open air");
  // What a cell-local (1×1) estimator would return: own building / ground /
  // cap — every candidate is decisively farther than the neighbor's facade.
  const local = Math.min(
    Math.hypot(s / 2 - eps, p[1] - hA), // own building's roof edge
    p[1], // ground
    Q.c, // the cap
  );
  assert.ok(
    local > 1.3 * (s / 2 + eps),
    `contract not load-bearing here: local ${local} vs true ${s / 2 + eps}`,
  );
});

// ── determinism ──────────────────────────────────────────────────────────────

test("#634 determinism: heights are the pure (i,k,seed) hash — rooflines pin it", () => {
  // Directly over a tower's centre the DE must equal the vertical gap to the
  // hash-predicted roof (nearest candidates: own sides ≥ fp2, neighbors
  // ≥ (c+s)/2, ground ≥ bh — all larger than t here). Proves the CPU leg
  // computes h(i,k,seed) exactly as specified, with no hidden state.
  const P = [0.5, 0.25, 0.8, 0.6, 0.5, 1];
  const Q = clampP(P);
  const de = leafDE(P);
  for (const [i, k] of [
    [0, 0],
    [3, -2],
    [-4, 5],
    [117, -63],
  ]) {
    const [, tower] = cellBoxes(i, k, Q);
    const bh = tower.hi[1];
    const fp2 = (tower.hi[0] - tower.lo[0]) / 2;
    const t = Math.min(0.5 * fp2, 0.1 * Q.c);
    const d = de((i + 0.5) * Q.c, bh + t, (k + 0.5) * Q.c);
    assert.ok(
      Math.abs(d - t) < 1e-12,
      `cell (${i},${k}): roof gap ${d} ≠ ${t}`,
    );
  }
});

test("#634 determinism: two evaluator instances agree; seeds actually differ", () => {
  const P = [0.5, 0.25, 0.8, 0.6, 0.5, 1];
  const a = leafDE(P),
    b = leafDE([...P]);
  const rand = mulberry32(0xc17);
  let diff = false;
  const other = leafDE([0.5, 0.25, 0.8, 0.6, 0.5, 2]);
  for (let n = 0; n < 2000; n++) {
    const x = rand() * 4 - 2,
      y = rand() * 1.2 - 0.1,
      z = rand() * 4 - 2;
    assert.equal(a(x, y, z), b(x, y, z), "same params, same field, bit-equal");
    if (Math.abs(a(x, y, z) - other(x, y, z)) > 1e-6) diff = true;
  }
  assert.ok(diff, "seed 2 must reshuffle the skyline");
});

test("#634 defaults: missing sp4/sp5 behave as 0 (the zeroed aux lane)", () => {
  // An old 4-slot shapeParams array must evaluate exactly like an explicit
  // det = 0 / seed = 0 — the same zeros the GPU's objAux lane carries for a
  // formula that predates the fat params.
  const a = leafDE([0.5, 0.25, 0.8, 0.6]);
  const b = leafDE([0.5, 0.25, 0.8, 0.6, 0, 0]);
  for (const [x, y, z] of [
    [0.3, 0.4, -0.2],
    [1.1, 0.05, 0.9],
    [-0.7, 0.9, 0.31],
  ])
    assert.equal(a(x, y, z), b(x, y, z));
});

// ── composition smoke ────────────────────────────────────────────────────────

test("#634 composes: op chain + iterShape + transform stay finite and sane", () => {
  const de = makeDE({
    name: "t",
    ops: [],
    iters: 1,
    deOption: 2,
    addC: false,
    objects: [
      {
        shapeId: 59,
        shapeParams: [0.5, 0.25, 0.8, 0.6, 0.5, 1],
        ops: [{ key: "rotateXYZ", values: [23, 12, 7] }],
        iters: 2,
        iterShape: true,
        transform: { origin: [0.1, 0, -0.2], uscale: 0.8, rot: [0, 20, 0] },
        combine: 0,
        blendK: 0,
      },
    ],
  });
  const rand = mulberry32(0xace);
  for (let n = 0; n < 500; n++) {
    const d = de(rand() * 4 - 2, rand() * 2 - 0.5, rand() * 4 - 2);
    assert.ok(Number.isFinite(d), "finite everywhere");
    assert.ok(d < 1e6, "never the +INF sentinel");
  }
});

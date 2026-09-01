// #632 — Wallpaper Fold + Space Group: the proof battery the spec review
// demanded. Both gates run the SHIPPED CPU twin (cpuorbit applyOp — the
// 3-emitter mirror discipline's testable leg). The review's central lesson,
// proved there with a fold that stayed perfectly 1-Lipschitz while leaving
// 82 879/100 000 points outside its cell: NON-EXPANSION AND CORRECTNESS ARE
// INDEPENDENT. So every group gets, beyond the mirrorvortex-style sampled
// operator norm (near AND far pairs):
//   • a fundamental-domain gate  — folded points land inside the chamber,
//     sampled out to r = 1000·cell (proves the baked K = 3 pass bound is
//     distance-free; measured need was K = 2, +1 margin),
//   • generator invariance       — f(g·p) == f(p) for a generating set,
//     including BOTH hex lattice vectors (the #584 rhombic-lattice lesson),
//     with NEGATIVE controls (operations NOT in the group must not be
//     invariant — this is what tells *333 apart from *632 and p31m),
//   • brute-force orbit ground truth (d2leaves style: an independent
//     enumeration sharing no code with the implementation),
//   • idempotence and chamber COVERAGE (the fold reaches its whole cell —
//     the exact failure mode Lipschitz sampling cannot see).
import { test } from "node:test";
import assert from "node:assert/strict";
import { applyOp } from "./cpuorbit.js";
import { byKey, W_UNCHANGED } from "./operators.js";
import { BOUNDING_FOLDS } from "./stability.js";
import { TOUCHY } from "./vary.js";

const SQRT3 = Math.sqrt(3);
const run = (key, v, p) => {
  const s = { x: p[0], y: p[1], z: p[2], w: 1 };
  applyOp(key, v, s);
  return s;
};
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);

function mulberry32(a) {
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Chamber membership per wallpaper tile (cell a, tolerance e) — restated from
// the operators.js derivation, in the test's own words.
const IN_TILE = [
  (x, y, a, e) => y >= -e && y <= x / SQRT3 + e && x <= a / 2 + e, // *632
  (x, y, a, e) => y >= -e && y <= x + e && x <= a / 2 + e, // *442
  (x, y, a, e) => x >= -e && y >= x / SQRT3 - e && x + SQRT3 * y <= a + e, // *333
];
// Signed distance to the chamber edge (positive = inside), for coverage bins.
const EDGE_TILE = [
  (x, y, a) => Math.min(y, x / 2 - (SQRT3 / 2) * y, a / 2 - x),
  (x, y, a) => Math.min(y, (x - y) / Math.SQRT2, a / 2 - x),
  (x, y, a) => Math.min(x, (SQRT3 / 2) * y - x / 2, (a - x - SQRT3 * y) / 2),
];

test("#632 registry: ids 65/66, isometry w-rules, symmetry category, lanes", () => {
  const wp = byKey("wallpaperFold");
  const sg = byKey("spaceGroupFold");
  assert.equal(wp.id, 65);
  assert.equal(sg.id, 66);
  assert.equal(wp.wRule, W_UNCHANGED);
  assert.equal(sg.wRule, W_UNCHANGED);
  assert.equal(wp.category, "symmetry");
  assert.equal(sg.category, "symmetry");
  // wallpaperFold fits the inline slots; spaceGroupFold spends the opAux lane
  // for CellB (ruckerBulb precedent — the CI warning at invariants.js:51 is
  // the deliberate, visible cost of the orthorhombic b axis).
  assert.ok(wp.params.length <= 3);
  assert.equal(sg.params.length, 4);
  // Both fold every touched coordinate into a bounded chamber — box-fold
  // family members for the pairing requirement (surfFold precedent for the
  // z-free wallpaper op).
  assert.ok(BOUNDING_FOLDS.includes("wallpaperFold"));
  assert.ok(BOUNDING_FOLDS.includes("spaceGroupFold"));
  // Absolute tile pitch — Vary keeps them on the tentFold leash.
  assert.ok(TOUCHY.has("wallpaperFold"));
  assert.ok(TOUCHY.has("spaceGroupFold"));
});

test("#632 3-emitter mirror: all params reach all three emitters", () => {
  // ruckerbulb.test.mjs precedent, including the #553 class-fence.
  const sg = byKey("spaceGroupFold");
  for (const p of ["op.p0", "op.p1", "op.p2", "op.p3"])
    assert.ok(sg.wgsl.includes(p), `wgsl body must read ${p}`);
  const emitted = sg.glsl(["A0", "A1", "A2", "A3"]);
  for (const v of ["A0", "A1", "A2", "A3"])
    assert.ok(emitted.includes(v), `glsl body must interpolate ${v}`);
  for (const def of [sg, byKey("wallpaperFold")])
    assert.equal(
      /\$\{[^}]*[<>][^}]*\}/.test(def.glsl.toString()),
      false,
      "a JS-side comparison on a param reference is always-false dead code (#553)",
    );
  // The CPU leg reads v[3]: Pmmm with b = 1 differs from b = 2 on a y probe.
  const p = [0.3, 0.9, 0.4];
  assert.notEqual(
    run("spaceGroupFold", [0, 2.0, 2.0, 1.0], p).y,
    run("spaceGroupFold", [0, 2.0, 2.0, 2.0], p).y,
  );
});

test("#632 sampled operator norm ≤ 1, near AND far pairs, w untouched", () => {
  // Near pairs at the mandalay gate's 1e-4 scale; far pairs because a PERIODIC
  // fold's failure modes live across cells, not within one (review §2d — the
  // local test alone is blind to a wrong lattice period).
  const rand = mulberry32(0x632);
  const CASES = [
    ["wallpaperFold", [0, 1.5]],
    ["wallpaperFold", [1, 0.7]],
    ["wallpaperFold", [2, 2.3]],
    ["spaceGroupFold", [0, 1.5, 0.8, 2.2]],
    ["spaceGroupFold", [1, 1.5, 0.6, 0]],
    ["spaceGroupFold", [2, 1.1, 1.9, 0]],
    ["spaceGroupFold", [3, 2.0, 0.9, 0]],
    ["spaceGroupFold", [4, 1.3, 0, 0]],
  ];
  for (const [key, v] of CASES) {
    let worstNear = 0;
    let worstFar = 0;
    for (let i = 0; i < 12000; i++) {
      const p = [rand() * 30 - 15, rand() * 30 - 15, rand() * 30 - 15];
      const eps = 1e-4;
      const dp = [rand() - 0.5, rand() - 0.5, rand() - 0.5];
      const dn = Math.hypot(...dp) || 1;
      const q = p.map((x, j) => x + (dp[j] / dn) * eps);
      const A = run(key, v, p);
      const B = run(key, v, q);
      worstNear = Math.max(worstNear, dist(A, B) / eps);
      assert.equal(A.w, 1, "W_UNCHANGED: the fold must not touch w");
      const p2 = [rand() * 30 - 15, rand() * 30 - 15, rand() * 30 - 15];
      const C = run(key, v, p2);
      const d0 = Math.hypot(p[0] - p2[0], p[1] - p2[1], p[2] - p2[2]);
      if (d0 > 1e-9) worstFar = Math.max(worstFar, dist(A, C) / d0);
    }
    // A crease-straddling pair measures the chord, ≤ both branch lengths, so
    // 1-Lipschitz survives sampling exactly; allow only fp noise.
    assert.ok(worstNear <= 1 + 1e-9, `${key} ${v}: near norm ${worstNear}`);
    assert.ok(worstFar <= 1 + 1e-9, `${key} ${v}: far norm ${worstFar}`);
  }
});

test("#632 fundamental domain: every folded point lands in its chamber (r to 1000·a)", () => {
  // The review's core lesson. Radii log-uniform out to 1000·cell prove the
  // baked K = 3 hex-cascade bound is distance-free (the closed-form lattice
  // prefold is what makes that possible; measured need was K = 2).
  const rand = mulberry32(0xfd0);
  const sph = (r) => {
    const th = rand() * 2 * Math.PI;
    const u = 2 * rand() - 1;
    const s = Math.sqrt(Math.max(0, 1 - u * u));
    return [r * s * Math.cos(th), r * s * Math.sin(th), r * u];
  };
  // wallpaper tiles: x,y in chamber, z EXACTLY untouched.
  for (const [tile, a] of [
    [0, 1.5],
    [1, 0.7],
    [2, 2.3],
  ]) {
    for (let i = 0; i < 15000; i++) {
      const p = sph(10 ** (rand() * 5 - 2) * a);
      const s = run("wallpaperFold", [tile, a], p);
      assert.ok(
        IN_TILE[tile](s.x, s.y, a, 1e-9 * a),
        `tile ${tile}: (${p}) → (${s.x},${s.y}) outside chamber`,
      );
      assert.equal(s.z, p[2], "wallpaper fold must not touch z");
    }
  }
  // space groups: full 3-D chamber per group (a, c, b as collapsed).
  const IN_SG = {
    0: (s, a, c, b) =>
      s.x >= -1e-9 &&
      s.x <= a / 2 + 1e-9 &&
      s.y >= -1e-9 &&
      s.y <= b / 2 + 1e-9 &&
      s.z >= -1e-9 &&
      s.z <= c / 2 + 1e-9,
    1: (s, a, c) =>
      s.y >= -1e-9 &&
      s.y <= s.x + 1e-9 &&
      s.x <= a / 2 + 1e-9 &&
      s.z >= -1e-9 &&
      s.z <= c / 2 + 1e-9,
    2: (s, a, c) =>
      IN_TILE[0](s.x, s.y, a, 1e-9 * a) && s.z >= -1e-9 && s.z <= c / 2 + 1e-9,
    3: (s, a, c) =>
      IN_TILE[2](s.x, s.y, a, 1e-9 * a) && s.z >= -1e-9 && s.z <= c / 2 + 1e-9,
    4: (s, a) =>
      s.x <= a / 2 + 1e-9 &&
      s.x >= s.y - 1e-9 &&
      s.y >= s.z - 1e-9 &&
      s.z >= -1e-9,
  };
  for (const [g, a, c, b] of [
    [0, 1.5, 0.8, 2.2],
    [1, 1.5, 0.6, 0],
    [2, 1.1, 1.9, 0],
    [3, 2.0, 0.9, 0],
    [4, 1.3, 0, 0],
  ]) {
    const ce = c > 0 ? c : a;
    const be = b > 0 ? b : a;
    for (let i = 0; i < 15000; i++) {
      const p = sph(10 ** (rand() * 5 - 2) * a);
      const s = run("spaceGroupFold", [g, a, c, b], p);
      assert.ok(
        IN_SG[g](s, a, ce, be),
        `group ${g}: (${p}) → (${s.x},${s.y},${s.z}) outside chamber`,
      );
    }
  }
});

test("#632 generator invariance: f(g·p) == f(p), with negative controls", () => {
  // The #584 lesson made executable: a hex fold that quietly built a rhombic
  // lattice passes every Lipschitz gate — only invariance under BOTH lattice
  // vectors catches it. Negative controls prove we built each group and not a
  // larger one (e.g. *333 must NOT be invariant under the *632-only mirror).
  const rand = mulberry32(0x8f4);
  const check = (key, v, gens, negs, label) => {
    let worst = 0;
    let negMiss = 0;
    let negTried = 0;
    for (let i = 0; i < 4000; i++) {
      const p = [rand() * 16 - 8, rand() * 16 - 8, rand() * 16 - 8];
      const f0 = run(key, v, p);
      for (const g of gens)
        worst = Math.max(worst, dist(f0, run(key, v, g(p))));
      for (const g of negs) {
        negTried++;
        if (dist(f0, run(key, v, g(p))) < 1e-6) negMiss++;
      }
    }
    assert.ok(worst < 1e-9, `${label}: generator deviation ${worst}`);
    // A sampled point can sit ON the extra-symmetry locus by chance; allow a
    // sliver, not a pattern.
    assert.ok(
      negMiss <= negTried / 100,
      `${label}: a non-symmetry looked invariant (${negMiss}/${negTried}) — built a LARGER group than declared`,
    );
  };
  const rotZ = (t) => (p) => [
    p[0] * Math.cos(t) - p[1] * Math.sin(t),
    p[0] * Math.sin(t) + p[1] * Math.cos(t),
    p[2],
  ];
  const A = 1.37;
  const C = 0.83;
  const B = 2.05;
  const hexT1 = (p) => [p[0] + A, p[1], p[2]];
  const hexT2 = (p) => [p[0] + A / 2, p[1] + (SQRT3 / 2) * A, p[2]];
  const mir30 = (p) => [
    p[0] * 0.5 + p[1] * (SQRT3 / 2),
    p[0] * (SQRT3 / 2) - p[1] * 0.5,
    p[2],
  ];
  const mx = (p) => [-p[0], p[1], p[2]];
  const my = (p) => [p[0], -p[1], p[2]];
  const mz = (p) => [p[0], p[1], -p[2]];
  // *632: full hex — every 30° mirror + both lattice vectors.
  check(
    "wallpaperFold",
    [0, A],
    [hexT1, hexT2, rotZ(Math.PI / 3), mx, my, mir30],
    [],
    "*632",
  );
  // *442: square lattice + diagonal.
  check(
    "wallpaperFold",
    [1, A],
    [
      (p) => [p[0] + A, p[1], p[2]],
      (p) => [p[0], p[1] + A, p[2]],
      rotZ(Math.PI / 2),
      mx,
      my,
      (p) => [p[1], p[0], p[2]],
    ],
    [rotZ(Math.PI / 3)],
    "*442",
  );
  // *333: C3v only — my (the 0° mirror) and R60 belong to *632, NOT *333.
  check(
    "wallpaperFold",
    [2, A],
    [hexT1, hexT2, rotZ((2 * Math.PI) / 3), mx, mir30],
    [my, rotZ(Math.PI / 3)],
    "*333",
  );
  // Pmmm (a≠b≠c): axis mirrors + all three translations; the diagonal swap
  // must NOT be a symmetry (that would be tetragonal).
  check(
    "spaceGroupFold",
    [0, A, C, B],
    [
      (p) => [p[0] + A, p[1], p[2]],
      (p) => [p[0], p[1] + B, p[2]],
      (p) => [p[0], p[1], p[2] + C],
      mx,
      my,
      mz,
    ],
    [(p) => [p[1], p[0], p[2]]],
    "Pmmm",
  );
  // P4/mmm (c≠a): C4 + diagonal; the x↔z swap must NOT hold (not cubic).
  check(
    "spaceGroupFold",
    [1, A, C, 0],
    [
      (p) => [p[0] + A, p[1], p[2]],
      (p) => [p[0], p[1] + A, p[2]],
      (p) => [p[0], p[1], p[2] + C],
      rotZ(Math.PI / 2),
      (p) => [p[1], p[0], p[2]],
      mz,
    ],
    [(p) => [p[2], p[1], p[0]]],
    "P4/mmm",
  );
  // P6/mmm: the full hex prism.
  check(
    "spaceGroupFold",
    [2, A, C, 0],
    [hexT1, hexT2, (p) => [p[0], p[1], p[2] + C], rotZ(Math.PI / 3), my, mz],
    [],
    "P6/mmm",
  );
  // P-6m2: D3h — σh yes, but NOT the 0° vertical mirror and NOT C6.
  check(
    "spaceGroupFold",
    [3, A, C, 0],
    [
      hexT1,
      hexT2,
      (p) => [p[0], p[1], p[2] + C],
      rotZ((2 * Math.PI) / 3),
      mx,
      mir30,
      mz,
    ],
    [my, rotZ(Math.PI / 3)],
    "P-6m2",
  );
  // Pm-3m: full Oh — cyclic axis permutation, C4, inversion, diagonal.
  check(
    "spaceGroupFold",
    [4, A, 0, 0],
    [
      (p) => [p[0] + A, p[1], p[2]],
      (p) => [p[0], p[1] + A, p[2]],
      (p) => [p[0], p[1], p[2] + A],
      (p) => [p[1], p[2], p[0]],
      rotZ(Math.PI / 2),
      (p) => [-p[0], -p[1], -p[2]],
      (p) => [p[1], p[0], p[2]],
      mx,
    ],
    [],
    "Pm-3m",
  );
});

test("#632 ground truth: hex folds equal the brute-force orbit representative", () => {
  // d2leaves precedent — an INDEPENDENT enumeration, no code shared with the
  // implementation: apply every point-group matrix and every nearby lattice
  // translation, find the orbit member inside the chamber, compare.
  const rand = mulberry32(0x67d);
  const rot = (t) => [
    [Math.cos(t), -Math.sin(t)],
    [Math.sin(t), Math.cos(t)],
  ];
  const refl = (t) => [
    [Math.cos(2 * t), Math.sin(2 * t)],
    [Math.sin(2 * t), -Math.cos(2 * t)],
  ];
  const D6 = [];
  for (let k = 0; k < 6; k++) {
    D6.push(rot((k * Math.PI) / 3));
    D6.push(refl((k * Math.PI) / 6));
  }
  const C3V = [];
  for (let k = 0; k < 3; k++) {
    C3V.push(rot((k * 2 * Math.PI) / 3));
    C3V.push(refl(Math.PI / 6 + (k * Math.PI) / 3));
  }
  const a = 1.37;
  for (const [tile, ptOps] of [
    [0, D6],
    [2, C3V],
  ]) {
    let worst = 0;
    for (let i = 0; i < 2500; i++) {
      const x = (rand() * 8 - 4) * a;
      const y = (rand() * 8 - 4) * a;
      const s = run("wallpaperFold", [tile, a], [x, y, 0]);
      let best = Infinity;
      for (const M of ptOps) {
        const px = M[0][0] * x + M[0][1] * y;
        const py = M[1][0] * x + M[1][1] * y;
        for (let m = -6; m <= 6; m++)
          for (let n = -6; n <= 6; n++) {
            const qx = px + m * a + (n * a) / 2;
            const qy = py + n * ((SQRT3 / 2) * a);
            if (IN_TILE[tile](qx, qy, a, 1e-7 * a))
              best = Math.min(best, Math.hypot(qx - s.x, qy - s.y));
          }
      }
      worst = Math.max(worst, best);
    }
    assert.ok(
      worst < 1e-9,
      `tile ${tile}: fold deviates from the orbit representative by ${worst}`,
    );
  }
});

test("#632 idempotence and coverage: the fold IS a retraction ONTO its cell", () => {
  const rand = mulberry32(0xc0e);
  // Idempotence: a folded point is already the chamber representative.
  const CASES = [
    ["wallpaperFold", [0, 1.5]],
    ["wallpaperFold", [1, 0.7]],
    ["wallpaperFold", [2, 2.3]],
    ["spaceGroupFold", [0, 1.5, 0.8, 2.2]],
    ["spaceGroupFold", [1, 1.5, 0.6, 0]],
    ["spaceGroupFold", [2, 1.1, 1.9, 0]],
    ["spaceGroupFold", [3, 2.0, 0.9, 0]],
    ["spaceGroupFold", [4, 1.3, 0, 0]],
  ];
  for (const [key, v] of CASES) {
    for (let i = 0; i < 3000; i++) {
      const p = [rand() * 20 - 10, rand() * 20 - 10, rand() * 20 - 10];
      const f1 = run(key, v, p);
      const f2 = run(key, v, [f1.x, f1.y, f1.z]);
      assert.ok(dist(f1, f2) < 1e-12, `${key} ${v}: not idempotent`);
    }
  }
  // Coverage (surjectivity ONTO the chamber): fold a fine grid of one full
  // cell and demand every interior bin of the chamber is hit — the flip side
  // of the domain gate, and the direct executable form of "a fold that never
  // reaches its cell".
  const a = 1.5;
  for (const tile of [0, 1, 2]) {
    const bx = tile === 2 ? a / 2 : a / 2; // chamber bounding box
    const by = tile === 0 ? a / (2 * SQRT3) : tile === 1 ? a / 2 : a / SQRT3;
    const N = 24;
    const hit = new Set();
    const G = 400;
    for (let i = 0; i < G; i++)
      for (let j = 0; j < G; j++) {
        const s = run(
          "wallpaperFold",
          [tile, a],
          [(i / G) * 2 * a - a, (j / G) * 2 * a - a, 0],
        );
        hit.add(`${Math.floor((s.x / bx) * N)},${Math.floor((s.y / by) * N)}`);
      }
    const diag = Math.hypot(bx / N, by / N);
    let want = 0;
    let miss = 0;
    for (let i = 0; i < N; i++)
      for (let j = 0; j < N; j++) {
        const cx = ((i + 0.5) / N) * bx;
        const cy = ((j + 0.5) / N) * by;
        if (EDGE_TILE[tile](cx, cy, a) > diag) {
          want++;
          if (!hit.has(`${i},${j}`)) miss++;
        }
      }
    assert.ok(want > 30, `tile ${tile}: degenerate bin layout (${want})`);
    assert.equal(
      miss,
      0,
      `tile ${tile}: ${miss}/${want} interior chamber bins never reached`,
    );
  }
});

test("#632 cross-op and composition pins", () => {
  const rand = mulberry32(0xace);
  for (let i = 0; i < 4000; i++) {
    const p = [rand() * 12 - 6, rand() * 12 - 6, rand() * 12 - 6];
    // The hexagonal spaceGroupFold arms are the wallpaper kernels one
    // dimension up — exact agreement on x,y (kernels are shared on the CPU
    // leg and textual twins in WGSL/GLSL; this pin holds all three together).
    const w0 = run("wallpaperFold", [0, 1.3], p);
    const s0 = run("spaceGroupFold", [2, 1.3, 0.7, 0], p);
    assert.ok(Math.hypot(w0.x - s0.x, w0.y - s0.y) < 1e-12, "P6/mmm vs *632");
    const w2 = run("wallpaperFold", [2, 1.3], p);
    const s2 = run("spaceGroupFold", [3, 1.3, 0.7, 0], p);
    assert.ok(Math.hypot(w2.x - s2.x, w2.y - s2.y) < 1e-12, "P-6m2 vs *333");
    // The review's verified compositions, kept as executable clean-room
    // anchors: Pm-3m ≡ tentFold(a,a,a) + mengerFold (exactly), and
    // P4/mmm ≡ tentFold(a,a,c) + kaleido(4, 0, Mirror) (up to trig fp).
    const a = 1.7;
    const c = 0.9;
    const cube = run("spaceGroupFold", [4, a, 0, 0], p);
    const t = { x: p[0], y: p[1], z: p[2], w: 1 };
    applyOp("tentFold", [a, a, a], t);
    applyOp("mengerFold", [], t);
    assert.deepEqual(
      [cube.x, cube.y, cube.z],
      [t.x, t.y, t.z],
      "Pm-3m must equal tentFold+mengerFold exactly",
    );
    const tet = run("spaceGroupFold", [1, a, c, 0], p);
    const k = { x: p[0], y: p[1], z: p[2], w: 1 };
    applyOp("tentFold", [a, a, c], k);
    applyOp("kaleido", [4, 0, 1], k);
    // 1e-6, not 1e-9: kaleido bakes the truncated 6.2831853 for 2π (wedge off
    // by ~2e-9 rad) and round-trips through atan2 — OUR op is the exact one.
    assert.ok(dist(tet, k) < 1e-6, "P4/mmm must equal tentFold+kaleido(4)");
    // Pmmm ≡ tentFold(a,b,c), the review's freebie.
    const orth = run("spaceGroupFold", [0, a, c, 2.2], p);
    const u = { x: p[0], y: p[1], z: p[2], w: 1 };
    applyOp("tentFold", [a, 2.2, c], u);
    assert.deepEqual([orth.x, orth.y, orth.z], [u.x, u.y, u.z], "Pmmm");
  }
});

test("#632 param collapse rule: ignored dims are ignored, 0 collapses to a", () => {
  const rand = mulberry32(0xbcd);
  for (let i = 0; i < 3000; i++) {
    const p = [rand() * 12 - 6, rand() * 12 - 6, rand() * 12 - 6];
    // cubic reads a only.
    assert.deepEqual(
      run("spaceGroupFold", [4, 1.3, 0.4, 2.0], p),
      run("spaceGroupFold", [4, 1.3, 0, 0], p),
      "Pm-3m must ignore CellC and CellB",
    );
    // tetragonal/hexagonal ignore b.
    for (const g of [1, 2, 3])
      assert.deepEqual(
        run("spaceGroupFold", [g, 1.3, 0.7, 3.0], p),
        run("spaceGroupFold", [g, 1.3, 0.7, 0], p),
        `group ${g} must ignore CellB`,
      );
    // c = 0 and b = 0 collapse to a.
    assert.deepEqual(
      run("spaceGroupFold", [1, 1.3, 0, 0], p),
      run("spaceGroupFold", [1, 1.3, 1.3, 0], p),
      "CellC=0 must mean CellC=a",
    );
    assert.deepEqual(
      run("spaceGroupFold", [0, 1.3, 0.7, 0], p),
      run("spaceGroupFold", [0, 1.3, 0.7, 1.3], p),
      "CellB=0 must mean CellB=a",
    );
    // CellA is the master switch; out-of-range Group falls back to 0 (the
    // selector-op convention, mirrored in all three emitters).
    assert.deepEqual(run("spaceGroupFold", [2, 0, 0.7, 0], p), {
      x: p[0],
      y: p[1],
      z: p[2],
      w: 1,
    });
    assert.deepEqual(run("wallpaperFold", [0, 0], p), {
      x: p[0],
      y: p[1],
      z: p[2],
      w: 1,
    });
    assert.deepEqual(
      run("spaceGroupFold", [9, 1.3, 0.7, 0], p),
      run("spaceGroupFold", [0, 1.3, 0.7, 0], p),
      "out-of-range Group must fall back to 0",
    );
  }
});

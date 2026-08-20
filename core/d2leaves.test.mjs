// D2 batch-1 leaf math pins — closed-form spot values for every new leaf,
// derived independently from the published formulas (TPMS implicits, IQ
// constructions, torus-knot unroll), evaluated through the CPU tier
// (cpu.js LEAF_FNS via a pure-leaf scene DE — the same numbers the WGSL/GLSL
// bodies mirror). Run: node --test core/d2leaves.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { makeDE } from "./cpu.js";

const close = (a, b, eps = 1e-9) => Math.abs(a - b) < eps;
const leafDE = (shapeId, shapeParams) =>
  makeDE({
    name: "t",
    ops: [],
    iters: 1,
    deOption: 2,
    addC: false,
    objects: [
      {
        shapeId,
        shapeParams,
        ops: [],
        iters: 1,
        transform: { origin: [0, 0, 0], uscale: 1, rot: [0, 0, 0] },
        combine: 0,
        blendK: 0,
      },
    ],
  });

test("gyroid: origin sits ON the level-0 surface; sphere bound caps the field", () => {
  const de = leafDE(7, [3, 0.06, 0, 1.4]);
  // g(0,0,0) = 0 → d = −thickness (inside the shell).
  assert.ok(close(de(0, 0, 0), -0.06));
  // Far outside the bound the sphere term wins: |p| − bound.
  assert.ok(close(de(4, 0, 0), 4 - 1.4));
  // At (π/(2f), 0, 0): g = sin(π/2) = 1 → d = 1/(f√3) − t.
  const x = Math.PI / 6; // f = 3
  assert.ok(close(de(x, 0, 0), 1 / (3 * Math.sqrt(3)) - 0.06, 1e-7));
});

test("schwarz P (#280): +π/2 phase shift puts the level-0 surface THROUGH the origin", () => {
  const de = leafDE(8, [3, 0, 0.06, 1.4]);
  // #280 regression: with the quarter-period shift, g(0) = cos(π/2)·3 = 0, so the
  // origin sits ON the shell (d = −thickness) — like the gyroid — instead of the
  // old g(0)=3 chamber centre that framed the hollow interior and blocked zoom.
  // This single assertion fails against the pre-fix (unshifted) body.
  assert.ok(close(de(0, 0, 0), -0.06, 1e-7));
  // The shift moves the chamber centre (g=±3) to p = π/(2f) per axis: there
  // q = π/2 + π/2 = π, cos π · 3 = −3 → d = 3/(f√3) − t, deep off-surface.
  const a = Math.PI / 6; // f = 3
  assert.ok(close(de(a, a, a), 3 / (3 * Math.sqrt(3)) - 0.06, 1e-7));
  // Far outside the bound the sphere term wins: |p| − bound.
  assert.ok(close(de(4, 0, 0), 4 - 1.4));
});

test("schwarz P (#280): a central-axis ray from the default framing hits the surface", () => {
  // The bug: the camera targets the origin, but the pre-fix surface never crossed
  // the central axis (min DE ≈ 0.13 > 0) — you could never dolly onto it. March
  // the DE down the +z axis toward the origin from the default cam distance and
  // assert it now reaches the surface (d ≤ 0), the minimal "not empty/tiny" check.
  const de = leafDE(8, [3, 0, 0.06, 1.4]);
  let z = 4.4,
    hit = false;
  for (let i = 0; i < 4000 && z > -4.4; i++) {
    const d = de(0, 0, z);
    if (d <= 0) {
      hit = true;
      break;
    }
    z -= Math.max(Math.abs(d), 1e-4) * 0.5;
  }
  assert.ok(
    hit,
    "central-axis ray must reach the schwarzP surface at default params",
  );
});

test("lidinoid: origin value matches the published implicit (+0.15 term)", () => {
  const de = leafDE(9, [2, 0, 0.05, 2]);
  // At the origin every sin term is 0, every cos2 product is 1 →
  // g = −1.5 + 0.15 = −1.35 → d = 1.35/(2·3) − 0.05.
  assert.ok(close(de(0, 0, 0), 1.35 / 6 - 0.05, 1e-7));
});

test("scherk: on-surface at the origin (sin 0 = sinh 0 sinh 0), Taubin-normalized", () => {
  const de = leafDE(10, [2.5, 0.04, 1.4]);
  // f(0)=0, |∇| term = s(1+√(0+0+1)) = 2s → d = −t.
  assert.ok(close(de(0, 0, 0), -0.04, 1e-9));
  assert.ok(close(de(3, 0, 0), 3 - 1.4)); // bound
});

test("hexGrid: cell center is (cellR·mod − wall) from the honeycomb wall", () => {
  const de = leafDE(11, [0.5, 0.3, 0.04, 0.9]);
  // Center of the (0,0) cell: hex SDF = −cellR (0.5·0.5·0.9) → |hd|−wall.
  assert.ok(close(de(0, 0, 0), 0.225 - 0.04, 1e-7));
  // Above the slab the z extrusion wins at cell center: |z| − zThick.
  assert.ok(close(de(0, 0, 1.0), 1.0 - 0.3, 1e-7));
});

test("hexGrid (#353): the exact corner-aware SDF no longer underestimates near a vertex", () => {
  // cellSize .5, zThick 50 (large, so the z prism never clamps the 2D value),
  // wall .04, cellR .5. Near a hex corner, the pre-fix
  // `max(q.x·0.866+q.y·0.5, q.y) − r` shortcut is only exact along each FACE —
  // it undershoots the true (vertex-aware) distance, the reported "walls never
  // quite close at a corner" bug. Pinned against an inline copy of the old
  // formula so a regression back to the shortcut fails loudly.
  const p = [0.5, 50, 0.04, 0.5];
  const de = leafDE(11, p);
  const r = 0.5 * p[0] * p[3]; // drawn apothem .125
  const R = (2 * r) / Math.sqrt(3); // circumradius .1443376 — a vertex on +x
  // (.28, 0) is still inside cell (0,0)'s Voronoi region (circumradius
  // 2·r0/√3 = .2886751 for tiling apothem r0 = .25), and its nearest point on
  // the ring network is exactly that vertex — so the true value is analytic.
  const x = 0.28;
  const newDe = de(x, 0, 0);
  // EPS 1e-7: the leaf ships √3/√3⁻¹ as 7-decimal float32-grade constants
  // (1.7320508 / 0.5773503) while this expectation uses exact arithmetic, so
  // ~5e-9 of disagreement is the constants themselves, not the geometry. The
  // defects these tests guard are O(1e-2) — five orders clear of the bound.
  const EPS = 1e-7;
  assert.ok(close(newDe, x - R - p[2], EPS), "exact corner-aware vertex distance");
  const fract = (v) => v - Math.floor(v);
  const oldHd = (px, py) => {
    const csx = p[0] * 1.7320508,
      csy = p[0];
    const ax = (fract(px / csx) - 0.5) * csx,
      ay = (fract(py / csy) - 0.5) * csy;
    const bx = (fract(px / csx + 0.5) - 0.5) * csx,
      by = (fract(py / csy + 0.5) - 0.5) * csy;
    const useA = ax * ax + ay * ay < bx * bx + by * by;
    const qx = Math.abs(useA ? ax : bx),
      qy = Math.abs(useA ? ay : by);
    return Math.max(qx * 0.8660254 + qy * 0.5, qy) - r;
  };
  const oldWallDist = Math.abs(oldHd(x, 0)) - p[2]; // what the pre-fix formula reported
  // The old shortcut reports a noticeably SMALLER (unsafe, underestimated —
  // "deeper inside" than reality) distance at the same point; that gap is
  // exactly the reported "walls never quite close at a corner" defect.
  assert.ok(newDe - oldWallDist > 0.004, "old formula underestimated by >0.004");
});

// #353 round 5 — the cell LATTICE, one level up from the per-cell SDF above.
// The dual-offset fold's two sublattices union to a centred-rectangular
// lattice, which is hexagonal ONLY when the rect period is (2√3·r, 2·r) for
// hexagon apothem r. The shipped code used (1.5·s, √3·s) against r = s/2 —
// a shortest-vector shell of multiplicity 4 (rhombic), no hex lattice at all,
// which is what produced the reported periodic smear bands. These three tests
// pin the lattice through the SHIPPED DE only (no re-implementation of the
// fold), so they fail on any recurrence.

// Flat-top hexagon of apothem r centred at (cx, cy): vertices at 0°,60°,…,300°
// on the circumradius 2r/√3. Independent ground truth — explicit polygon,
// point-to-segment distance, no SDF trickery shared with the implementation.
const hexRingDist = (px, py, cx, cy, r) => {
  const R = (2 * r) / Math.sqrt(3);
  let best = Infinity;
  for (let i = 0; i < 6; i++) {
    const a0 = (i * Math.PI) / 3,
      a1 = ((i + 1) * Math.PI) / 3;
    const ax = cx + R * Math.cos(a0),
      ay = cy + R * Math.sin(a0);
    const bx = cx + R * Math.cos(a1),
      by = cy + R * Math.sin(a1);
    const vx = bx - ax,
      vy = by - ay,
      wx = px - ax,
      wy = py - ay;
    const t = Math.max(0, Math.min(1, (wx * vx + wy * vy) / (vx * vx + vy * vy)));
    best = Math.min(best, Math.hypot(wx - t * vx, wy - t * vy));
  }
  return best;
};

test("hexGrid (#353 round 5): emitted distance matches a brute-force point-to-polygon field over every cell", () => {
  // The same bar PR #527 used for the per-cell SDF, now applied to the whole
  // lattice: densely sample several periods and compare the leaf's own value
  // against the true distance to the union of EVERY cell's hexagon ring.
  for (const cellR of [0.5, 0.9, 1.0]) {
    const s = 0.5,
      wall = 0.04;
    const p = [s, 50, wall, cellR]; // zThick 50 → the z prism never clamps
    const de = leafDE(11, p);
    const r = 0.5 * s * cellR;
    const csx = s * 1.7320508,
      csy = s;
    let samples = 0,
      mismatches = 0,
      worst = 0;
    const N = 90;
    for (let ix = 0; ix < N; ix++)
      for (let iy = 0; iy < N; iy++) {
        // offset off the exact grid so samples don't all land on symmetry axes
        const x = -csx + (2 * csx * ix) / (N - 1) + 0.0013;
        const y = -csy + (2 * csy * iy) / (N - 1) + 0.0007;
        samples++;
        let truth = Infinity;
        const i0 = Math.floor(x / csx),
          j0 = Math.floor(y / csy);
        for (let i = i0 - 2; i <= i0 + 2; i++)
          for (let j = j0 - 2; j <= j0 + 2; j++) {
            // sublattice A: ((i+.5)·P, (j+.5)·Q);  sublattice B: (i·P, j·Q)
            truth = Math.min(truth, hexRingDist(x, y, (i + 0.5) * csx, (j + 0.5) * csy, r));
            truth = Math.min(truth, hexRingDist(x, y, i * csx, j * csy, r));
          }
        // the leaf emits |hd| − wall, so |hd| = de + wall. Tolerance 1e-7 —
        // see the EPS note above: the ground truth is exact arithmetic, the
        // leaf ships 7-decimal constants, so ~5e-9 residual is expected. The
        // pre-fix lattice missed this field by up to 3.2e-2.
        const err = Math.abs(de(x, y, 0) + wall - truth);
        if (err > 1e-7) {
          mismatches++;
          worst = Math.max(worst, err);
        }
      }
    assert.equal(
      mismatches,
      0,
      `cellR ${cellR}: ${mismatches}/${samples} samples disagree with brute force (worst ${worst})`,
    );
    assert.ok(samples === N * N, "sampled the full grid");
  }
});

test("hexGrid (#353 round 5): the honeycomb has C6 symmetry about a cell centre", () => {
  // A true hexagonal grid is invariant under 60° rotation about a cell centre.
  // The pre-fix rhombic lattice is only C2 — this is the cheapest direct
  // discriminator between the two, and it goes through the shipped DE.
  const de = leafDE(11, [0.5, 50, 0.04, 0.9]);
  const c = Math.cos(Math.PI / 3),
    sn = Math.sin(Math.PI / 3);
  for (const [x, y] of [
    [0.19, 0.03],
    [0.07, 0.21],
    [0.24, 0.11],
    [0.33, 0.05],
    [0.12, 0.09],
  ]) {
    let px = x,
      py = y;
    const base = de(px, py, 0);
    for (let k = 1; k < 6; k++) {
      const nx = px * c - py * sn,
        ny = px * sn + py * c;
      px = nx;
      py = ny;
      assert.ok(close(de(px, py, 0), base, 1e-7), `C6: ${x},${y} rotated ${60 * k}°`);
    }
  }
});

test("hexGrid (#353 round 5): all six nearest-neighbour cell translations have equal length", () => {
  // What makes a centred-rect lattice HEXAGONAL rather than rhombic is that
  // its shortest-vector shell has multiplicity 6, all of length 2·r0 = s.
  // Assert the shipped field is invariant under all six of those, and (below)
  // that nothing shorter is also a period — which the rhombic version fails.
  const s = 0.5;
  const de = leafDE(11, [s, 50, 0.04, 0.9]);
  const nn = [
    [0, s],
    [0, -s],
    [(Math.sqrt(3) / 2) * s, s / 2],
    [(Math.sqrt(3) / 2) * s, -s / 2],
    [(-Math.sqrt(3) / 2) * s, s / 2],
    [(-Math.sqrt(3) / 2) * s, -s / 2],
  ];
  const pts = [
    [0.031, 0.017],
    [0.213, -0.087],
    [-0.144, 0.202],
  ];
  for (const [dx, dy] of nn) {
    // pure arithmetic on the exact vectors — this one really is exact
    assert.ok(close(Math.hypot(dx, dy), s, 1e-15), "neighbour vector length is s");
    for (const [x, y] of pts)
      assert.ok(close(de(x + dx, y + dy, 0), de(x, y, 0), 1e-7), `period (${dx},${dy})`);
  }
  // At cellR 1 the cells tile exactly: the midpoint of a shared flat edge sits
  // dead centre in a wall, so the ring distance is 0 → de = −wall.
  const tile = leafDE(11, [s, 50, 0.04, 1.0]);
  assert.ok(close(tile(0, s / 2, 0), -0.04, 1e-7), "exact tiling at cellR 1");
});

test("triGrid: on a 0° wall plane the distance is −wall (inside the wall)", () => {
  const de = leafDE(12, [0.5, 0.3, 0.04]);
  assert.ok(close(de(0.123, 0, 0), -0.04, 1e-7)); // y=0 is a t0 wall plane
  // A symmetric interior point equidistant from all three families (0, 0.3):
  // t0 = 0.2, t1 = t2 = 0.15 → d = 0.15 − wall.
  assert.ok(close(de(0, 0.3, 0), 0.15 - 0.04, 1e-7), "gap interior distance");
});

test("gear: ring band contains the mid-radius, tooth extends past the rim", () => {
  const de = leafDE(13, [12, 1, 0.12, 0.18]);
  assert.ok(de(0.8, 0, 0) < 0, "mid-ring is inside"); // 0.55 < 0.8 < 1
  assert.ok(close(de(0.2, 0, 0), 1 * 0.55 - 0.2, 1e-7)); // hub hole
  // On the +x axis a tooth is centered (am = 0): solid out to R + depth.
  assert.ok(de(1.08, 0, 0) < 0, "tooth tip region is solid");
  assert.ok(de(1.3, 0, 0) > 0, "past the tooth tip is outside");
});

test("mengerPlate: detail 1 carves the central cross out of the plate", () => {
  const de = leafDE(14, [1, 0.25, 1]);
  // Center of the plate is inside the first cross → carved (outside).
  assert.ok(de(0, 0, 0) > 0, "central cross carved");
  // A point inside the plate but off the cross arms stays solid:
  assert.ok(de(0.5, 0.5, 0) < 0, "corner cell solid");
  // detail 0 clamps to 1 (registry min).
});

test("knotPQ: distance to the (1,0) degenerate knot = a plain circle", () => {
  // p=1, q=0 → the strand angle is constant 0 → circle at radius R+0.35R in
  // the y=0 plane; distance from a point on the ring axis is analytic.
  const de = leafDE(15, [1, 0, 1, 0.15]);
  const ringR = 1 + 0.35; // R + rr·cos(0)
  assert.ok(close(de(ringR, 0, 0), -0.15, 1e-7)); // on the circle
  assert.ok(close(de(ringR + 1, 0, 0), 1 - 0.15, 1e-7));
});

test("loresVoxel: sphere-inside cell renders its cube; empty space marches by base", () => {
  const de = leafDE(16, [0, 1, 0.25, 0]);
  // Origin cell center (0.125,0.125,0.125) is inside the unit sphere → the
  // cube SDF governs: at the cell center d = −half = −0.125.
  assert.ok(close(de(0.125, 0.125, 0.125), -0.125, 1e-7));
  // Far outside, the base-distance branch (≥ res·0.25 floor) applies.
  assert.ok(de(3, 3, 3) > 1, "far field marches by the base distance");
});

test("torus boxy widening: params 2/3 = 0 is bit-identical to the classic torus", () => {
  const classic = leafDE(3, [0.8, 0.25, 0, 0]);
  for (const [x, y, z] of [
    [1.1, 0.2, 0.3],
    [0.5, -0.4, 0.7],
  ]) {
    const q = Math.hypot(Math.hypot(x, z) - 0.8, y) - 0.25;
    assert.ok(close(classic(x, y, z), q));
  }
  // Boxy ring: Chebyshev ring metric.
  const boxy = leafDE(3, [0.8, 0.25, 1, 0]);
  const lr = Math.max(Math.abs(1.1), Math.abs(0.3));
  assert.ok(close(boxy(1.1, 0.2, 0.3), Math.hypot(lr - 0.8, 0.2) - 0.25));
});

// ── D2 batch 2 pins ──────────────────────────────────────────────────────────

test("helix: a point ON the strand at t=0 is −stringR; pitch lifts the turn", () => {
  const de = leafDE(17, [0.5, 1, 0.1, 0.7]);
  assert.ok(close(de(0.7, 0, 0), -0.1, 1e-7)); // curve point t=0
  // One full turn up: same xz, y = pitch.
  assert.ok(close(de(0.7, 0.5, 0), -0.1, 1e-7));
  // Half a turn: the strand is at (−0.7, 0.25) — from (0.7, 0.25, 0) the
  // nearest candidate is a half-turn away in angle; distance ≥ strand gap.
  assert.ok(de(0.7, 0.25, 0) > 0.1);
});

test("sphereCage: pole and equator ring hits", () => {
  const de = leafDE(19, [1, 0.035, 6, 8]);
  // Equator parallel (phi = π/2 quantizes exactly at even counts): the ring
  // passes through (1, 0, 0) → on-wire.
  assert.ok(close(de(1, 0, 0), -0.035, 1e-7));
  // A meridian passes through the pole: (0, 1, 0) lies on every meridian.
  assert.ok(close(de(0, 1, 0), -0.035, 1e-7));
});

test("sliceCage: on-shell on a slice plane vs between slices", () => {
  const de = leafDE(20, [1, 8, 0.05, 0]);
  assert.ok(close(de(1, 0, 0), -0.05, 1e-7)); // θ=0 is a slice plane, on shell
  // Rotate half a sector (π/16) at the same radius: off the slab → positive.
  const a = Math.PI / 16;
  assert.ok(de(Math.cos(a), 0, Math.sin(a)) > 0);
});

test("waveSurface: linear mode crest height and Lipschitz factor", () => {
  const de = leafDE(21, [4, 0.25, 0, 0.05]);
  // At x = π/8 (sin = 1): surface height 0.25 → on-surface reads −thick.
  assert.ok(close(de(Math.PI / 8, 0.25, 0), -0.05, 1e-7));
  const lip = 1 / Math.sqrt(1 + 1);
  assert.ok(close(de(Math.PI / 8, 1.25, 0), 1 * lip - 0.05, 1e-7));
});

test("kleinBagel: twist 0 = two stacked tori (figure-8 shell)", () => {
  const de = leafDE(22, [1, 0.3, 0, 0.04]);
  // Upper lobe center circle: u=0, v=0.15 → d8=0 on (len−1, y−0.15)=0…
  // at (1, 0.15+0.15, 0): d8 = 0.15 → |0.15−0.15|−0.04 = −0.04 (on shell).
  assert.ok(close(de(1, 0.3, 0), -0.04, 1e-7));
  assert.ok(close(de(1, 0, 0), 0.15 - 0.15 - 0.04, 1e-6)); // tangent point
});

test("seashell: t=0 mouth point sits on the tube", () => {
  const de = leafDE(23, [0.15, 0.35, 0.12, 4]);
  // t=0: center (1, 0, 0), tube radius 0.35·0.5 = 0.175.
  assert.ok(close(de(1, 0, 0), -0.175, 1e-7));
});

test("dini horn: profile radius tapers exponentially with height", () => {
  const de = leafDE(24, [0.8, 0.6, 2, 0.04]);
  assert.ok(close(de(0.8, 0, 0), -0.04, 1e-7)); // base circle
  const r1 = 0.8 * Math.exp(-0.6);
  assert.ok(close(de(r1, 1, 0), -0.04, 1e-7)); // profile at y=1
});

test("room: open corner (floor + two back walls), not a closed shell (#353)", () => {
  // #353: a closed 6-sided shell rendered indistinguishably from Round Box
  // from any outside camera angle. Now only 3 of 6 faces are solid (floor at
  // −y, walls at −z/−x) so the interior is always visible.
  const de = leafDE(25, [1, 0.8, 1, 0.05]);
  // Interior point: 0.7 above the floor's inner (top) surface — the nearest
  // of the 3 plates (back/side walls are farther away from the origin here).
  assert.ok(close(de(0, 0, 0), 0.7, 1e-9));
  // Center of the floor plate (thickness 0.1, centered at y = -0.75).
  assert.ok(close(de(0, -0.75, 0), -0.05, 1e-9));
  // (1,0,0) used to sit IN the +x wall (old test: -0.05, inside solid). The
  // +x side is now open — this point is 0.7 above the floor instead.
  assert.ok(close(de(1, 0, 0), 0.7, 1e-9));
});

test("roundBox / boxFrame match the IQ closed forms", () => {
  const rb = leafDE(26, [0.6, 0.6, 0.6, 0.1]);
  assert.ok(close(rb(0.7, 0, 0), 0.1, 1e-7)); // 0.7 − 0.6 face distance
  assert.ok(close(rb(0, 0, 0), -0.6, 1e-7));
  const bf = leafDE(27, [0.8, 0.06, 0, 0]);
  assert.ok(close(bf(0.8, 0.8, 0), 0, 1e-7)); // the cube edge is ON the beam surface
  assert.ok(close(bf(0.74, 0.74, 0), -0.06, 1e-7)); // beam centerline (inset by edge)
  assert.ok(bf(0, 0, 0) > 0.5, "frame center is hollow");
});

// ── D2 Taubin wave pins (signed f/|∇f|; sources cited in leaves.js) ─────────

test("heartSurf (#353): lobes UP (+Y), cusp DOWN — not the round-3 inversion", () => {
  // Round 3 (#527) remapped the axes with q = (x, z, −y): asymmetry on world
  // Y (right idea) but with the LOBES at −Y — an upside-down heart, whose
  // silhouette at the default orbit angle was the still-reported blob.
  // Round 7 flips it: q = (x, z, y). The wide lobe side must be at +Y and
  // the cusp at −Y. At x=1, z=0 the lobes bulge toward the +z³ side of the
  // formula, so world +y is INSIDE (lobe material) and world −y OUTSIDE.
  const de = leafDE(28, [1, 1.6]);
  const above = de(1, 0.5, 0);
  const below = de(1, -0.5, 0);
  assert.ok(close(above, -0.18981382118013107, 1e-9));
  assert.ok(close(below, 0.12480743338154411, 1e-9));
  assert.ok(above < 0 && below > 0, "lobes up: inside at +y, outside at -y (was the reverse)");
});

test("Taubin wave: on-surface zeros and inside signs match the cited implicits", () => {
  const eps = 2e-3; // Taubin quotient at an exact root is 0 up to fp noise
  // heart (formula z = lobe/point axis, lobes at world +Y after the round-7
  // flip q=(x,z,y)): the on-axis roots sit at world (0,±1,0) — check the
  // cusp-side one — and the interior spans the origin's neighborhood.
  assert.ok(Math.abs(leafDE(28, [1, 1.6])(0, -1, 0)) < eps, "heart root");
  assert.ok(leafDE(28, [1, 1.6])(0, 0.3, 0) < 0, "heart inside (lobe side up)");
  // citrus: tips at y=±0.5 (surface y∈[0,1] centered): f(0, ±0.5, 0) = 0.
  assert.ok(Math.abs(leafDE(29, [1, 1.6])(0, 0.5, 0)) < eps, "citrus tip");
  assert.ok(leafDE(29, [1, 1.6])(0, 0, 0) < 0, "citrus inside");
  // piriform: x∈[0,1] centered → tips at x=±0.5.
  assert.ok(Math.abs(leafDE(30, [1, 1.6])(-0.5, 0, 0)) < eps, "piriform tip");
  assert.ok(leafDE(30, [1, 1.6])(0, 0, 0) < 0, "piriform inside");
  // kiss: x²+y²=(1−z)z⁴ → at z=1 the radius is 0 (tip).
  assert.ok(Math.abs(leafDE(31, [1, 1.8])(0, 0, 1)) < eps, "kiss tip");
  assert.ok(leafDE(31, [1, 1.8])(0, 0, 0.5) < 0, "kiss inside");
  // ding-dong: x²+y²+z³−z² = 0 → radius √(z²−z³); at z=1 radius 0.
  assert.ok(Math.abs(leafDE(32, [1, 1.8])(0, 0, 1)) < eps, "dingdong tip");
  assert.ok(leafDE(32, [1, 1.8])(0, 0, 0.5) < 0, "dingdong inside");
  // devil (a=0.9, b=1): on the y axis f = y²(y²−a²) → root at y=a.
  assert.ok(
    Math.abs(leafDE(33, [0.9, 1, 1, 1.8])(0, 0.9, 0)) < eps,
    "devil root",
  );
  assert.ok(
    leafDE(33, [0.9, 1, 1, 1.8])(0, 0.5, 0) < 0,
    "devil inside spindle",
  );
  // trifolium (a=1): on the +x axis (ρ=0): x⁴ = x³ → root at x=1.
  assert.ok(
    Math.abs(leafDE(34, [1, 1, 1.6])(1, 0, 0)) < eps,
    "trifolium lobe tip",
  );
  assert.ok(leafDE(34, [1, 1, 1.6])(0.5, 0, 0) < 0, "trifolium inside lobe");
  // decocube (c=0.8): near a face circle the product ≈ 0 < level → inside.
  assert.ok(
    leafDE(35, [0.8, 0.02, 1, 2.2])(0.8, 0, 1) < 0,
    "decocube on-circle",
  );
  assert.ok(
    leafDE(35, [0.8, 0.02, 1, 2.2])(0, 0, 0) > 0,
    "decocube center hollow",
  );
  // cayley: f(0,0,z)=z²−1 → root at z=1; origin −1 inside.
  assert.ok(Math.abs(leafDE(36, [1, 1.8])(0, 0, 1)) < eps, "cayley root");
  assert.ok(leafDE(36, [1, 1.8])(0, 0, 0) < 0, "cayley inside");
  // gumdrop torus: on the ring plane (x=0) f = 4u²−20u+17, u=r² → root at
  // u=(20−√128)/8; scale 1 for the pin.
  const r0 = Math.sqrt((20 - Math.sqrt(128)) / 8);
  assert.ok(Math.abs(leafDE(37, [1, 3])(0, r0, 0)) < eps, "gumdrop ring root");
  assert.ok(leafDE(37, [1, 3])(0, 1.5, 0) < 0, "gumdrop inside ring");
  // quadric (1,1,−1): hyperboloid x²+y²−z²=1 → root at (1,0,0); axis inside.
  assert.ok(
    Math.abs(leafDE(38, [1, 1, -1, 1.6])(1, 0, 0)) < eps,
    "quadric root",
  );
  assert.ok(leafDE(38, [1, 1, -1, 1.6])(0, 0, 0.5) < 0, "quadric inside");
});

test("Taubin quotient is a sane step size (≈ true distance for the quadric sphere)", () => {
  // a=b=c=1 → the unit sphere; f/|∇f| = (r²−1)/(2r) ≈ r−1 near the surface.
  const de = leafDE(38, [1, 1, 1, 3]);
  for (const r of [0.8, 0.95, 1.05, 1.3])
    assert.ok(Math.abs(de(r, 0, 0) - (r * r - 1) / (2 * r)) < 1e-6, `r=${r}`);
});

// ── D2 batch 3 — heightfields ────────────────────────────────────────────────

test("gnarlyField: the origin column is a fixed point → pure |y|·lip gap", () => {
  const [s, b, A, N] = [0.25, 3, 0.3, 3];
  const de = leafDE(39, [s, b, A, N]);
  // u=v=0 is a fixed point of the gnarl walk (sin(0+sin(0))=0), so h=0 and
  // the gap is |y| times the N-step Lipschitz foreshortening.
  const lip = 1 / (1 + A * (1 + s * (1 + b) * N));
  assert.ok(close(de(0, 0.5, 0), 0.5 * lip, 1e-9));
  assert.ok(close(de(0, -0.5, 0), 0.5 * lip, 1e-9)); // two-sided surface
});

test("ducksField: one step from (1,0) has log-modulus 0 → h=0", () => {
  const H = 0.4;
  const de = leafDE(40, [H, 1, -0.6, -0.6]);
  // w0=(1,0): |w|=1 → l=0, so the 1-iteration mean is 0 and h=0.
  assert.ok(close(de(1, 0.3, 0), 0.3 / (1 + 2 * H), 1e-9));
});

test("mandelPlate: interior plateau + Milnor skirt closed forms", () => {
  const [N, slope, depth] = [24, 4, 0.4];
  const de = leafDE(41, [N, slope, depth, 1]);
  const k = Math.sqrt(1 + depth * depth * slope * slope);
  // c = 0 is interior → dM = 0 → plateau at Depth, foreshortened.
  assert.ok(close(de(0, 1, 0), (1 - depth) / k, 1e-9));
  // c = (3,0) escapes with z=(147,0), z'=(169,0): dM = 147·ln147/169 ≈ 4.34
  // is beyond the 1/Slope skirt cutoff, so the horizontal guard wins.
  const dM = (147 * Math.log(147)) / 169;
  assert.ok(close(de(3, 0, 0), dM - 1 / slope, 1e-6));
  // The skirt is REAL terrain: just outside the cutoff the solid is empty,
  // just inside it the surface rises above the plane.
  assert.ok(de(3, 0.001, 0) > 0);
});

test("checkerField: tile centers sit at Bump / 0; borders at Bump/2", () => {
  const [B, C, S] = [0.2, 0.5, 0.1];
  const de = leafDE(42, [B, C, S, 0]);
  const lip = 1 / (1 + (B * 3.2) / (S * C));
  assert.ok(close(de(0.25, 0.5, 0.25), (0.5 - B) * lip, 1e-9)); // raised tile
  assert.ok(close(de(-0.25, 0.5, 0.25), 0.5 * lip, 1e-9)); // low tile
  assert.ok(close(de(0, 0.5, 0.25), (0.5 - B / 2) * lip, 1e-9)); // border
});

test("riemannSqrt: h = H·√r·|cos θ/2| — on-surface at θ=0, branch cut at θ=π", () => {
  const H = 0.6;
  const de = leafDE(43, [1, H, 0, 1.6]);
  // θ=0, r=1: h = H → the point (1, H, 0) lies on the sheet.
  assert.ok(close(de(1, H, 0), 0, 1e-9));
  // θ=π: cos(π/2)=0 → h=0; gap is |y| times the lip factor (sr=1).
  const lip = 1 / (1 + H * (0.5 + 0 + 0.6));
  assert.ok(close(de(-1, 0.3, 0), 0.3 * lip, 1e-9));
});

// ── D2 batch 4 — geometric tail ──────────────────────────────────────────────

test("octahedron: vertex distance exact, face point on-surface, inside plane", () => {
  const de = leafDE(44, [0.8, 0, 0, 0]);
  assert.ok(close(de(2, 0, 0), 1.2)); // to the +x vertex at (0.8,0,0)
  const a = 0.8 / 3;
  assert.ok(close(de(a, a, a), 0, 1e-9)); // face plane |x|+|y|+|z| = s
  assert.ok(close(de(0, 0, 0), -0.8 * 0.57735027, 1e-7));
});

test("dodecahedron: face-normal max; Icosa flips to the dual normal set", () => {
  const de = leafDE(45, [0.8, 0, 0, 0]);
  assert.ok(close(de(0, 0, 0), -0.8));
  assert.ok(close(de(0, 0, 1), 0.85065081 - 0.8, 1e-7)); // z hits (0,a,b)·q
  const ic = leafDE(45, [0.8, 0, 1, 0]);
  assert.ok(close(ic(1, 1, 1), 3 * 0.57735027 - 0.8, 1e-7)); // (1,1,1)/√3 face
});

test("nPrism: apothem face + slab cap", () => {
  const de = leafDE(46, [6, 0.8, 0.6, 0]);
  assert.ok(close(de(2, 0, 0), 1.2)); // face-normal direction
  assert.ok(close(de(0.8, 0, 0), 0, 1e-9)); // on the face
  assert.ok(close(de(0, 0, 0), -0.6)); // slab cap dominates inside
});

test("pyramid: apex, base plane and base edge all on-surface", () => {
  const de = leafDE(47, [4, 1, 1, 0]);
  assert.ok(close(de(0, 0.5, 0), 0, 1e-9)); // apex
  assert.ok(close(de(0, -0.5, 0), 0, 1e-9)); // base center
  assert.ok(close(de(1, -0.5, 0), 0, 1e-9)); // base edge midpoint
  assert.ok(close(de(0, 0, 0), -0.5 / Math.SQRT2, 1e-9)); // side plane inside
});

// ── #353 Round fix: Dodecahedron/N-Prism/Pyramid ────────────────────────────
// Before this fix, all three leaves computed `max(...) - round`, which is
// algebraically IDENTICAL to enlarging Size/Radius/Base by `round` — Round
// was 100% redundant with the size param, never an actual edge/corner round
// ("Round doesn't do anything", "Round value is just changing radius").
// The fix chains a smooth-max (smaxP) so only TIES between the two/three
// blended terms (i.e. edges/vertices) bend; a term that clearly dominates
// (a face far from any edge) stays exactly on the old hard-max value. Each
// test below (a) finds an exact tie of the old formula's terms, confirms the
// tie assumption, then (b) shows the new value is offset by exactly `k/4`
// (the smooth-max's known bump at an exact tie) — NOT by `-round` (the old
// bug) and not by 0 (no-op). A companion assertion shows a dominant-term
// point is completely unaffected by Round, i.e. faces don't move.

test("dodecahedron Round chamfers the tie between two faces, not a Size shift (#353)", () => {
  const size = 0.8,
    k = 0.2;
  // Icosa=0 faces: n0=(0,a,b), n1=(a,b,0); dot0=dot1 when qx=0, qz=(b-a)/b·qy.
  const a = 0.52573111,
    b = 0.85065081;
  const qy = 1,
    qz = (b - a) / b;
  const flat = leafDE(45, [size, 0, 0, 0]);
  const rounded = leafDE(45, [size, k, 0, 0]);
  const tie = b * qy - size; // dot1 (= dot0) minus size, i.e. the old hard-max result
  assert.ok(
    close(flat(0, qy, qz), tie, 1e-6),
    "point is an exact tie of dot0/dot1",
  );
  assert.ok(
    close(rounded(0, qy, qz), tie + k / 4, 1e-6),
    "edge bends by k/4, not -k",
  );
  // Deep on a single dominant face (existing pin point): Round has zero effect.
  assert.ok(close(rounded(0, 0, 1), flat(0, 0, 1), 1e-9));
});

test("nPrism Round chamfers the cap/side rim, not a Radius/Height shift (#353)", () => {
  const radius = 0.8,
    height = 0.6,
    k = 0.2;
  const flat = leafDE(46, [6, radius, height, 0]);
  const rounded = leafDE(46, [6, radius, height, k]);
  const x = radius - height; // makes d2 (x - radius) equal cap (0 - height)
  const tie = -height;
  assert.ok(close(flat(x, 0, 0), tie, 1e-9), "point is an exact d2/cap tie");
  assert.ok(
    close(rounded(x, 0, 0), tie + k / 4, 1e-9),
    "rim bends by k/4, not -k",
  );
  // Deep on the flat side face (existing pin point): Round has zero effect.
  assert.ok(close(rounded(2, 0, 0), flat(2, 0, 0), 1e-9));
});

test("pyramid Round chamfers the base rim, not a Base/Height shift (#353)", () => {
  const r = 1,
    h = 1,
    k = 0.4;
  const yb = 0.2; // pick any base-plane offset; solve l so side == base (-yb)
  const l = 1 - yb * (1 + Math.SQRT2);
  const x = l,
    y = yb - 0.5 * h;
  const flat = leafDE(47, [4, r, h, 0]);
  const rounded = leafDE(47, [4, r, h, k]);
  assert.ok(close(flat(x, y, 0), -yb, 1e-9), "point is an exact side/base tie");
  assert.ok(
    close(rounded(x, y, 0), -yb + k / 4, 1e-9),
    "rim bends by k/4, not -k",
  );
  // Deep inside near the apex (existing pin point): Round has zero effect.
  assert.ok(close(rounded(0, 0.5, 0), flat(0, 0.5, 0), 1e-9));
});

test("greekCross: min of the three arm boxes", () => {
  const de = leafDE(48, [1, 0.25, 0, 0]);
  assert.ok(close(de(2, 0, 0), 1)); // past the +x arm tip
  assert.ok(close(de(0, 0, 0), -0.25)); // arm half-width inside
  assert.ok(close(de(0.5, 0.25, 0), 0, 1e-9)); // on an arm side face
});

test("borg: amp 0 degenerates to the plain box shell", () => {
  const de = leafDE(49, [0.9, 6, 0, 0.06]);
  assert.ok(close(de(0, 0, 0), 0.9 - 0.06)); // hollow center
  assert.ok(close(de(0.9, 0, 0), -0.06)); // in the wall
  // sin-product displacement + Lipschitz divisor, mirrored closed form.
  const dA = leafDE(49, [0.9, 6, 0.12, 0.06]);
  const q = Math.PI / 12; // sin(6q) = 1 on all axes
  const box = q - 0.9;
  const want = (Math.abs(box) - 0.06 + 0.12) / (1 + 0.12 * 6 * 1.8);
  assert.ok(close(dA(q, q, q), want, 1e-9));
});

test("tower: wave 0 flutes 0 is a plain infinite column", () => {
  const de = leafDE(50, [0.5, 0, 3, 0]);
  assert.ok(close(de(2, 0, 0), 1.5));
  assert.ok(close(de(0, 5, 0), -0.5)); // still inside far up the axis
  // breathing radius at the sine crest, gradient-bound divisor.
  const dW = leafDE(50, [0.5, 0.2, 2, 0]);
  const y = Math.PI / 4; // sin(2y) = 1
  assert.ok(close(dW(1, y, 0), (1 - 0.6) / Math.sqrt(1 + 0.04), 1e-9));
});

test("gem: table, culet and girdle all on-surface", () => {
  const de = leafDE(51, [8, 1, 0.35, 0.9]);
  assert.ok(close(de(0, 0.35, 0), 0, 1e-9)); // table center
  assert.ok(close(de(0, -0.9, 0), 0, 1e-9)); // culet tip
  assert.ok(close(de(1, 0, 0), 0, 1e-9)); // girdle on a facet mid-line
});

test("loxodrome: the equator crossing of the curve is inside the tube", () => {
  const de = leafDE(52, [4, 1, 1, 0.08]);
  // λ=0 → ψ=0 → the strand passes through φ=0: (1,0,0) is dead center.
  assert.ok(close(de(1, 0, 0), -0.08 * 0.8, 1e-9));
  // opposite meridian: chart deviation π scaled by the conformal factor.
  const arc = Math.PI / Math.sqrt(1 + 16);
  assert.ok(close(de(-1, 0, 0), (arc - 0.08) * 0.8, 1e-7));
});

test("logSpiral: r=1 θ=0 sits on an arm; quarter turn is between arms", () => {
  const de = leafDE(53, [0.25, 2, 0.8, 0.05]);
  assert.ok(close(de(1, 0, 0), -0.05 * 0.9, 1e-9));
  const gap = ((Math.PI / 2) * 0.25) / Math.sqrt(1 + 0.0625);
  assert.ok(close(de(0, 0, 1), (gap - 0.05) * 0.9, 1e-7));
});

test("pseudoSphere: the u=0 rim sample is an exact hit", () => {
  const de = leafDE(54, [0.8, 3, 0.08, 0]);
  assert.ok(close(de(0.8, 0, 0), -0.08 * 0.7, 1e-9)); // on the rim circle
  assert.ok(close(de(3, 0, 0), (2.2 - 0.08) * 0.7, 1e-7)); // radially out
});

test("helixStairs (#353): Width scales the tread from a FIXED outer rim, not the middle", () => {
  // Pre-fix, Width WAS the outer radius directly — raising it grew the whole
  // silhouette outward from the axis (reporter: "scaling from the middle").
  // The outer rim is now pinned at radius 1 regardless of Width; only the
  // tread's inward span (and so the central void) changes.
  const thin = leafDE(18, [12, 0.05, 0.2, 1.2]); // narrow tread, big void
  const thick = leafDE(18, [12, 0.05, 0.9, 1.2]); // wide tread, small void
  // Just outside the rim, on a step (y=0, x-axis): identical for both Widths.
  assert.ok(close(thin(1.05, 0, 0), 0.05, 1e-9));
  assert.ok(close(thick(1.05, 0, 0), 0.05, 1e-9));
  // Mid-radius (0.5): Width still controls the tread span — thin's void
  // reaches out to 0.5 (outside, +0.3), thick's doesn't (inside, −0.05).
  assert.ok(close(thin(0.5, 0, 0), 0.3, 1e-9));
  assert.ok(close(thick(0.5, 0, 0), -0.05, 1e-9));
});

test("helixStairs (#353): the tread height no longer JUMPS in value at a sector seam", () => {
  // Pre-fix, each sector independently rounded p.y/Pitch to its own nearest
  // step — at a seam, the two neighboring sectors can round to treads that
  // differ by more than one Lipschitz-safe step, so the raw DE VALUE (not
  // just its slope) jumped there by ~Pitch/Steps. That's the raymarcher
  // overshoot-at-a-jump noise seen right at the risers. Straddling the exact
  // point that reproduced the jump (found by search) by 2e-6 radians:
  const p = [19, 0.06, 0.62, 1.01];
  const de = leafDE(18, p);
  const l = 0.4,
    y = -2.1120664092221695,
    thetaBoundary = -8.5 / 19, // the pre-fix formula's own seam angle here
    eps = 1e-6;
  const a1 = (thetaBoundary - eps) * 2 * Math.PI,
    a2 = (thetaBoundary + eps) * 2 * Math.PI;
  const d1 = de(l * Math.cos(a1), y, l * Math.sin(a1));
  const d2 = de(l * Math.cos(a2), y, l * Math.sin(a2));
  // Pre-fix this pair reads 0.326 vs 0.273 (Δ ≈ 0.053 ≈ Pitch/Steps) for a
  // 2e-6 rad step. The round-7 exact-SDF union is continuous with |∇d| ≤ 1,
  // so the difference must be bounded by the points' own separation (the
  // Lipschitz property that makes sphere-tracing sound) — it need not be
  // zero: an exact distance varies smoothly as the angle sweeps past a
  // tread corner, unlike the old locally-flat blend.
  const sep = Math.hypot(
    l * Math.cos(a1) - l * Math.cos(a2),
    l * Math.sin(a1) - l * Math.sin(a2),
  );
  assert.ok(
    Math.abs(d1 - d2) <= 1.5 * sep,
    `seam jump: ${d1} vs ${d2} over sep ${sep}`,
  );
});

// ── D2 batch 5 ───────────────────────────────────────────────────────────────

test("randomCells: fill 0 marches by half-cells; a full grid puts a shape in every cell", () => {
  const empty = leafDE(55, [0.6, 0, 1, 1]);
  assert.ok(close(empty(0.1, 0.2, -0.4), 0.3)); // no shapes anywhere → 0.5·cell
  const full = leafDE(55, [0.6, 1, 1, 1]);
  // every slab cell occupied → its own cell center is inside SOME shape
  // (smallest dispatch radius 0.16·cell > 0).
  assert.ok(full(0.3, 0.3, 0.3) < 0);
  // outside the slab (|cell.y| ≥ 1) there is nothing, even at fill 1.
  assert.ok(close(full(0.3, 3, 0.3), 0.3));
});

test("umbrella: x² = z²y — on-surface at (1,1,1), signed inside at (0,1,1)", () => {
  const de = leafDE(56, [1, 0, 4, 0]);
  assert.ok(close(de(1, 1, 1), 0, 1e-9));
  assert.ok(close(de(0, 1, 1), -1 / Math.sqrt(5), 1e-6)); // f=−1, ∇=(0,−1,−2)
});

test("kleinBottle: f(0)=−1 with |∇|=2; the x-axis unit point is a surface zero", () => {
  const s = 0.4;
  const de = leafDE(57, [s, 0, 4, 0]);
  assert.ok(close(de(0, 0, 0), -s / 2, 1e-6));
  assert.ok(close(de(s, 0, 0), 0, 1e-6)); // A=B=C=0 → sextic zero (singular ring)
});

test("kleinianLimit: closed-form orbit pins at t = 2 (fixed point, one-step escape)", () => {
  const de = leafDE(58, [2, 0, 0, 48]);
  // Shell field: |pullback| − skin (0.025). Strip mid-line is a fixed point
  // of A with df staying 1 → |1|/1 − 0.025.
  assert.ok(close(de(0, 0, 0), 0.975, 1e-9));
  // Below the strip entirely: escapes before any transform, df = 1.
  assert.ok(close(de(0, -6, 0), 4.975, 1e-9));
  // One a-step: (0, 0.1) → ρ²=0.01 → (0, −8), df=100 → |−8|/100 − skin.
  assert.ok(close(de(0, -0.9, 0), 0.055, 1e-9));
});

// ── #353 Slab Width/Depth (a launch leaf, id 6 — not a D2 batch leaf, but
// grouped here with the rest of the #353 fix set) ───────────────────────────

test("slab: Width/Depth optionally clip one horizontal axis at a time (#353)", () => {
  // astiglic: "Bound Width, Bound Length ... using it on one axis" — a Box
  // IFS clamp constrains x AND z together; this clips either independently.
  const infinite = leafDE(6, [0.1, 0, 0]);
  const boundedX = leafDE(6, [0.1, 1, 0]);
  const boundedBoth = leafDE(6, [0.1, 1, 0.5]);
  // Width=Depth=0 (the leaf default) is the old, unbounded-in-x-and-z plane —
  // bit-identical for any existing Slab (no saved formula sets these slots).
  assert.ok(close(infinite(50, 0, 50), -0.1));
  // Width=1 clips x but leaves z unbounded.
  assert.ok(close(boundedX(50, 0, 0), 50 - 1));
  assert.ok(close(boundedX(0, 0, 50), -0.1));
  // Both set: past either bound wins (an SDF intersection, still exact away
  // from the thickness term since it's a simple max-of-planes box).
  assert.ok(close(boundedBoth(50, 0, 0), 50 - 1));
  assert.ok(close(boundedBoth(0, 0, 50), 50 - 0.5));
  assert.ok(close(boundedBoth(0, 0, 0), -0.1));
});

// ── #353 round 7 — flat treads, floored plateau ──────────────────────────────

test("helixStairs (#353 r7): tread tops are FLAT — no profile bend in the old blend zone", () => {
  // The round-3 seam-blend tilted the tread surface over the outer 40% of
  // every sector (mix(slabA, slabB, wgt) with wgt rising 0.6→1.0) — the
  // reporter's "how did we get this profile instead of rectangle". With
  // exact sector-slab treads the top face is a plane: the DE at a fixed
  // height above a tread must read the SAME across the sector's whole
  // angular span, including the zone the blend used to warp.
  const p = [19, 0.06, 0.62, 1.01];
  const de = leafDE(18, p);
  const steps = p[0], pitch = p[3];
  const k = 3; // an arbitrary sector
  const h = (k / steps) * pitch; // its tread height (turn 0)
  const l = 0.8;
  // The SURFACE height is the flatness witness (the field itself may
  // legitimately read a neighboring tread's corner when that is closer —
  // an exact union distance does that; the old blend instead moved the
  // surface). The top face must sit at exactly h + thick at every angle
  // across the sector, including the old blend zone:
  const topAt = (fFrac) => {
    const th = ((k + fFrac) / steps) * 2 * Math.PI;
    const cx = l * Math.cos(th), cz = l * Math.sin(th);
    assert.ok(de(cx, h + p[1] + 1e-6, cz) > 0, `f=${fFrac}: just above the top is outside`);
    assert.ok(de(cx, h + p[1] - 1e-6, cz) < 0, `f=${fFrac}: just below the top is inside`);
  };
  topAt(0.2); // old flat zone
  topAt(0.8); // old blend zone — used to be tilted
  // 0.9 is the deepest probe outside the deliberate rr seam-overlap (adjacent
  // treads inflate by rr so they fuse edge-to-edge; within rr of the seam the
  // NEIGHBOR's joint legitimately owns the point).
  topAt(0.9);
});

test("helixStairs (#353 r7): edges are rounded by 30% of the half-thickness", () => {
  const p = [12, 0.1, 0.6, 1.2];
  const de = leafDE(18, p);
  // Straight out past the outer rim at tread height: the surface sits at
  // radius 1 (the rr inset + rounding inflate cancel on a face), so the DE
  // from radius 1.5 is 0.5 — but at the EDGE (tread height + half-thickness,
  // diagonal approach) the corner is rounded: distance from a point diagonal
  // to the corner is hypot(dr, dy) - rr against the shrunk box corner.
  const k = 0.5 / p[0]; // mid-sector angle (in turns)
  const th = k * 2 * Math.PI;
  const rr = Math.min(0.3 * p[1], 0.2 * p[2]);
  const d = de(1.5 * Math.cos(th), 0, 1.5 * Math.sin(th));
  assert.ok(close(d, 0.5, 1e-9), "flat face unmoved by the rounding");
  // corner probe: 0.2 out and 0.2 up from the tread's outer-top edge
  const dc = de(1.2 * Math.cos(th), p[1] + 0.2, 1.2 * Math.sin(th));
  const expect = Math.hypot(0.2 + rr, 0.2 + rr) - rr;
  assert.ok(close(dc, expect, 1e-9), "corner distance is the rounded-box form");
});

test("mandelPlate (#353): the solid is floored 10 units down, not infinite", () => {
  const de = leafDE(41, [24, 4, 0.4, 1]);
  // (0,·,0) is inside the Mandelbrot set (c=0), under the plateau top:
  assert.ok(de(0, -5, 0) < 0, "still solid at y=-5 (above the floor)");
  assert.ok(de(0, -9.9, 0) < 0, "still solid just above the floor");
  assert.ok(de(0, -10.1, 0) > 0, "OUTSIDE just below the floor (was solid to -inf)");
  assert.ok(close(de(0, -12, 0), 2, 1e-9), "below the floor the DE is the plain plane distance");
});

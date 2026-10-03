// Guards for brickFold (parity wave 2) — the MB3D tilingbrick family.
//
// The corpus gap this op exists to close: corpus_coverage_2026-07-13.json
// classes `tilingbrickIFS` and `tilingbrick2IFS` as needs_op, noting "brick
// tiling = modFold + PER-ROW stagger offset ... plain-mod sub-case (stagger 0)
// is rotate+modFold exact" and "same brick-stagger gap (variant without the
// +offset gate)". Both claims are pinned below as executable rulings: stagger
// 0 IS modFold exactly, and the continuous stagger SUBSUMES the odd-row gate
// at Stagger = CellX/2, which is why one param replaces the source's two
// variants.
//
// Named *.test.mjs so it stays out of the apps' served *.js surface.
// Run: node --test core/brickfold.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";
import { applyOp } from "./cpuorbit.js";
import { isApproxDE, isDeSound, byKey } from "./operators.js";

const run = (values, pt) => {
  const s = { ...pt };
  applyOp("brickFold", values, s);
  return s;
};
const mod = (values, pt) => {
  const s = { ...pt };
  applyOp("modFold", values, s);
  return s;
};
const V = [4.0, 4.0, 2.0]; // the shipped defaults — classic running bond
const close = (a, b, eps = 1e-12) => Math.abs(a - b) < eps;

// A spread of points that lands in several different courses, including
// negative rows and points sitting exactly on a cell wall.
const PTS = [
  { x: 0.5, y: 0.4, z: 0.7, w: 1.3 },
  { x: -1.7, y: 5.2, z: -0.3, w: 1.0 },
  { x: 3.9, y: -6.1, z: 2.2, w: 0.6 },
  { x: 0.0, y: 2.0, z: 0.0, w: 1.0 },
  { x: -7.25, y: 11.75, z: 4.0, w: 2.5 },
];

test("Stagger 0 is modFold EXACTLY — the corpus's plain-mod sub-case", () => {
  // corpus note: "plain-mod sub-case (stagger 0) is rotate+modFold exact".
  // Bit-for-bit, not approximately: same rounding convention, same folds, and
  // the Y fold happening first must not change the X result (independent axes).
  for (const p of PTS)
    for (const [cx, cy] of [
      [4.0, 4.0],
      [1.5, 3.25],
      [8.0, 0.05],
      [0.0, 2.0],
      [2.0, 0.0],
    ]) {
      const b = run([cx, cy, 0], p);
      const m = mod([cx, cy, 0], p);
      assert.equal(b.x, m.x, `x @ cell ${cx}×${cy} ${JSON.stringify(p)}`);
      assert.equal(b.y, m.y, `y @ cell ${cx}×${cy} ${JSON.stringify(p)}`);
      assert.equal(b.z, m.z, "z is untouched by both");
    }
});

test("Stagger = CellX/2 reproduces the odd-row gate (the +offset variant)", () => {
  // The two corpus files differ only in an "+offset gate": one shifts odd rows
  // by half a cell, the other accumulates the shift every row. Because X is
  // folded mod CellX afterwards, the accumulating law applies a phase of
  // (Stagger·row) mod CellX — which at Stagger = CellX/2 IS the alternating
  // 0, CellX/2 of the gate. One continuous param therefore covers both files.
  const gated = (cx, cy, p) => {
    const row = Math.floor(p.y / cy + 0.5);
    const off = (((row % 2) + 2) % 2) * (cx * 0.5); // odd rows only
    const bx = p.x + off;
    return {
      x: bx - cx * Math.floor(bx / cx + 0.5),
      y: p.y - cy * row,
    };
  };
  for (const p of PTS)
    for (const [cx, cy] of [
      [4.0, 4.0],
      [3.0, 1.5],
      [6.5, 2.25],
    ]) {
      const b = run([cx, cy, cx * 0.5], p);
      const g = gated(cx, cy, p);
      assert.ok(close(b.x, g.x), `x ${b.x} ≠ gated ${g.x} @ ${cx}×${cy}`);
      assert.ok(close(b.y, g.y), `y ${b.y} ≠ gated ${g.y}`);
    }
});

test("REGRESSION: the row index comes from the PRE-fold y", () => {
  // Read the row off the ALREADY-FOLDED y and it is always 0 (the folded y is
  // confined to ±CellY/2, so floor(y/cy + 0.5) collapses to 0) — the stagger
  // silently vanishes and the op degenerates into a plain modFold that merely
  // looks right. Pin it by demanding that two ADJACENT courses actually land
  // on different X phases.
  const cx = 4.0,
    cy = 4.0,
    st = 2.0;
  const a = run([cx, cy, st], { x: 0.5, y: 0.0, z: 0 }); // row 0
  const b = run([cx, cy, st], { x: 0.5, y: 4.0, z: 0 }); // row 1
  assert.ok(close(a.y, b.y), "both fold to the same position within a course");
  assert.ok(
    Math.abs(a.x - b.x) > 1e-9,
    `adjacent rows share the X phase (${a.x} vs ${b.x}) — stagger is dead`,
  );
  // and the offset is exactly the half cell, wrapped back into the cell
  assert.ok(close(a.x, 0.5), `row 0 unshifted, got ${a.x}`);
  assert.ok(close(b.x, -1.5), `row 1 shifted by +2 then wrapped, got ${b.x}`);
});

test("two courses up is the identity when Stagger = CellX/2 (period 2)", () => {
  // The running bond repeats every 2 courses: the accumulated phase 2·(cx/2)
  // = cx is a whole cell and folds away.
  const cx = 4.0,
    cy = 3.0;
  for (const p of PTS) {
    const a = run([cx, cy, cx * 0.5], p);
    const b = run([cx, cy, cx * 0.5], { ...p, y: p.y + 2 * cy });
    assert.ok(close(a.x, b.x), `x ${a.x} ≠ ${b.x} two courses up`);
    assert.ok(close(a.y, b.y), `y ${a.y} ≠ ${b.y} two courses up`);
  }
});

test("output always lands inside the cell, for any stagger", () => {
  const cx = 3.0,
    cy = 2.0;
  for (const st of [-8, -2.5, 0, 0.05, 1.5, 7.95])
    for (const p of PTS) {
      const q = run([cx, cy, st], p);
      assert.ok(
        Math.abs(q.x) <= cx * 0.5 + 1e-9,
        `x ${q.x} escaped the ±${cx / 2} cell (stagger ${st})`,
      );
      assert.ok(
        Math.abs(q.y) <= cy * 0.5 + 1e-9,
        `y ${q.y} escaped the ±${cy / 2} course (stagger ${st})`,
      );
    }
});

test("a cell of 0 turns that axis off, and Z is never touched", () => {
  const p = { x: 5.5, y: 7.5, z: -3.25, w: 1.0 };
  assert.equal(run([0, 4, 2], p).x, 5.5, "CellX 0 = X untouched");
  assert.equal(run([4, 0, 2], p).y, 7.5, "CellY 0 = Y untouched");
  // CellY 0 means there are no rows, so there is no stagger either
  assert.equal(run([4, 0, 2], p).x, mod([4, 0, 0], p).x, "no rows, no shift");
  for (const v of [V, [0, 0, 0], [4, 0, 2], [0, 4, 2]])
    assert.equal(run(v, p).z, -3.25, "brickFold leaves Z alone by design");
});

test("registry contract: an exact isometry — w untouched, DE stays sound", () => {
  const def = byKey("brickFold");
  assert.equal(def.wRule, "unchanged");
  assert.ok(
    !def.deApprox,
    "a per-cell translation must NOT be deApprox-flagged",
  );
  assert.equal(def.params.length, 3);
  assert.equal(def.category, "symmetry");
  // Every declared default must sit on its own step grid (share codec quantises
  // op params at 0.01, and vary snaps integer-stepped ones).
  for (const p of def.params) {
    assert.ok(p.step >= 0.01, `${p.name} step ${p.step} is finer than 0.01`);
    assert.ok(
      close(Math.round(p.default / p.step) * p.step, p.default, 1e-9),
      `${p.name} default ${p.default} is off its ${p.step} grid`,
    );
  }
  const f = { ops: [{ key: "brickFold", values: V }] };
  assert.equal(isApproxDE(f), false);
  assert.equal(isDeSound(f), true, "must not taint DE-soundness");
  // #426 contract: an isometry leaves w exactly alone.
  for (const p of PTS)
    assert.equal(run(V, p).w, p.w, "isometry must not touch w");
});

test("the map is a pure translation inside a cell (|Jacobian| = 1)", () => {
  // Sample the differential away from the cell walls: the three columns must
  // come out as the identity, which is the whole justification for W_UNCHANGED.
  const h = 1e-6;
  const base = { x: 0.3, y: 0.4, z: 0.1 };
  const f = (p) => run(V, p);
  const o = f(base);
  const cols = [
    f({ ...base, x: base.x + h }),
    f({ ...base, y: base.y + h }),
    f({ ...base, z: base.z + h }),
  ];
  const expect = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  cols.forEach((c, j) => {
    const d = [(c.x - o.x) / h, (c.y - o.y) / h, (c.z - o.z) / h];
    d.forEach((v, i) =>
      assert.ok(
        Math.abs(v - expect[j][i]) < 1e-6,
        `J[${i}][${j}] = ${v}, want ${expect[j][i]}`,
      ),
    );
  });
});

test("finite in ⇒ finite out, including the degenerate corners", () => {
  for (const p of [
    { x: 0, y: 0, z: 0 },
    { x: 1, y: 1, z: 1 },
    { x: -1, y: 1, z: -1 },
    { x: 2, y: 2, z: 2 }, // exactly on a cell wall at the defaults
  ])
    for (const params of [V, [0, 0, 0], [8, 8, -8], [0.05, 0.05, 0.05]]) {
      const o = run(params, { ...p, w: 1 });
      for (const k of ["x", "y", "z", "w"])
        assert.ok(
          Number.isFinite(o[k]),
          `${k} = ${o[k]} at ${JSON.stringify(p)} params ${params}`,
        );
    }
});

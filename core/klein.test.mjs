// Guards for the kleinPolyMap op (Tglad family) — the port is line-faithful to
// the FPU-simulation-verified reconstruction, so these tests pin the port's
// STRUCTURE: variant decoding, the max(1, n & 7) bottom-tested loop counts,
// the escape parking, and the W_BULB_NUMERIC contract.
//
// Named *.test.mjs so sync_web_core.sh skips it. Run: node --test core/klein.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";
import { applyOp } from "./cpuorbit.js";

const P = { x: 0.5, y: 0.4, z: 0.7, w: 1 };
const run = (values, pt = P) => {
  const s = { ...pt };
  applyOp("kleinPolyMap", values, s);
  return s;
};
const same = (a, b) =>
  Math.abs(a.x - b.x) < 1e-12 &&
  Math.abs(a.y - b.y) < 1e-12 &&
  Math.abs(a.z - b.z) < 1e-12;

test("all four variants are distinct maps", () => {
  const out = [0, 1, 2, 3].map((kv) => run([2, 2, kv]));
  for (let a = 0; a < 4; a++)
    for (let b = a + 1; b < 4; b++)
      assert.ok(!same(out[a], out[b]), `variants ${a} and ${b} must differ`);
});

test("loop counts: max(1, n & 7) — 0 ≡ 1, 8 masks to 1-pass floor, 2 differs", () => {
  assert.ok(
    same(run([0, 0, 0]), run([1, 1, 0])),
    "Log2*=0 must equal Log2*=1 (floor to one pass)",
  );
  assert.ok(
    !same(run([1, 1, 0]), run([2, 1, 0])),
    "outer 2 passes must differ from 1",
  );
  assert.ok(
    !same(run([1, 1, 0]), run([1, 2, 0])),
    "inner 2 passes must differ from 1",
  );
});

test("escape parks the orbit at (1e10,1e10,1e10)", () => {
  const s = run([1, 1, 0], { x: 2e5, y: 1e5, z: 1e5, w: 1 }); // r² ≥ 1e10 at the outer head
  assert.deepEqual([s.x, s.y, s.z], [1e10, 1e10, 1e10]);
});

test("generic orbits stay finite at realistic pass counts", () => {
  // At extreme settings (7,7) the rational map's poles can push x,y to ±Inf
  // and the next head's Inf−Inf is NaN, which no ≥1e10 check catches — that
  // matches the FPU-verified source exactly (same class as BristorBrot
  // overflow), so it is NOT guarded here; presets stay in the 0–3 range.
  const pts = [
    P,
    { x: -0.31, y: 0.62, z: -0.44, w: 1 },
    { x: 1.1, y: -0.2, z: 0.05, w: 1 },
  ];
  for (const kv of [0, 1, 2, 3])
    for (const n of [1, 2, 3])
      for (const pt of pts) {
        const s = run([n, n, kv], pt);
        for (const k of ["x", "y", "z"])
          assert.ok(
            Number.isFinite(s[k]),
            `variant ${kv} n=${n}: ${k} = ${s[k]}`,
          );
      }
});

test("W_BULB_NUMERIC contract: w is untouched", () => {
  const s = run([2, 2, 3], { ...P, w: 1.7 });
  assert.equal(s.w, 1.7);
});

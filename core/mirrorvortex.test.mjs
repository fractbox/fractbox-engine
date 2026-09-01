// #635 — Mirror Shells + Spiral Vortex: the DE-classification PROOFS the spec
// review demanded (its blocker: "exact DE" was asserted, not derived). Both
// gates run the SHIPPED CPU twin (cpuorbit applyOp — the 3-emitter mirror
// discipline's testable leg) against finite-difference operator-norm sampling,
// the same style as mandalay.test.mjs's non-expansion gate, plus the
// fundamental-domain check the #632 review showed Lipschitz sampling alone
// cannot provide (a fold can be 1-Lipschitz AND land outside its cell).
import { test } from "node:test";
import assert from "node:assert/strict";
import { applyOp } from "./cpuorbit.js";
import { byKey, W_UNCHANGED, W_MUL_K } from "./operators.js";
import { BOUNDING_FOLDS } from "./stability.js";

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

test("#635 registry: ids 63/64, declared w-rules, bounding-fold membership", () => {
  const ms = byKey("mirrorShells");
  const sv = byKey("spiralVortex");
  assert.equal(ms.id, 63);
  assert.equal(sv.id, 64);
  assert.equal(ms.wRule, W_UNCHANGED);
  assert.equal(sv.wRule, W_MUL_K);
  assert.ok(BOUNDING_FOLDS.includes("mirrorShells"));
  // params fit the inline slots — no opAux lane needed (review "minor" ask)
  assert.ok(ms.params.length <= 3 && sv.params.length <= 3);
});

test("#635 mirrorShells: sampled operator norm ≤ 1 (the 'exact' proof)", () => {
  // Pairs 1e-4 apart (the mandalay gate's scale) across radii spanning the
  // identity core, the first shells, deep shells, AND the offset>0 near-origin
  // region the naive tent-of-(r−offset) design would have blown up on.
  const rand = mulberry32(0x635);
  const CASES = [
    [1.5, 0],
    [0.4, 0],
    [1.0, 1.2], // offset > 0: the near-origin identity region must stay identity
    [3.0, 0.5],
  ];
  for (const v of CASES) {
    let worst = 0;
    for (let i = 0; i < 20000; i++) {
      const r = 10 ** (rand() * 3 - 1.5); // radii 0.03 .. ~30
      const dir = [rand() - 0.5, rand() - 0.5, rand() - 0.5];
      const n = Math.hypot(...dir) || 1;
      const p = dir.map((x) => (x / n) * r);
      const eps = 1e-4;
      const dp = [rand() - 0.5, rand() - 0.5, rand() - 0.5];
      const dn = Math.hypot(...dp) || 1;
      const q = p.map((x, j) => x + (dp[j] / dn) * eps);
      const a = run("mirrorShells", v, p);
      const b = run("mirrorShells", v, q);
      worst = Math.max(worst, dist(a, b) / eps);
      assert.equal(a.w, 1, "W_UNCHANGED: the fold must not touch w");
    }
    // A crease-straddling pair measures the chord, ≤ both branch lengths, so
    // 1-Lipschitz survives sampling exactly; allow only fp noise.
    assert.ok(
      worst <= 1 + 1e-9,
      `spacing=${v[0]} offset=${v[1]}: norm ${worst}`,
    );
  }
});

test("#635 mirrorShells: fundamental domain — folded radius lands in the cell", () => {
  // The #632-review lesson: Lipschitz alone can pass a fold that never reaches
  // its cell. Every folded point must satisfy R0 ≤ r' ≤ R0 + S/2 (+fp) for
  // r > R0, and r' === r below the offset (identity core).
  const rand = mulberry32(0xbee);
  for (const [S, R0] of [
    [1.5, 0],
    [0.7, 1.1],
  ]) {
    for (let i = 0; i < 20000; i++) {
      const r = 10 ** (rand() * 3 - 1.5);
      const th = rand() * Math.PI * 2;
      const ph = Math.acos(2 * rand() - 1);
      const p = [
        r * Math.sin(ph) * Math.cos(th),
        r * Math.sin(ph) * Math.sin(th),
        r * Math.cos(ph),
      ];
      const s = run("mirrorShells", [S, R0], p);
      const rp = Math.hypot(s.x, s.y, s.z);
      if (r <= R0) {
        assert.ok(Math.abs(rp - r) < 1e-12, "identity core untouched");
      } else {
        assert.ok(
          rp >= R0 - 1e-9 && rp <= R0 + S / 2 + 1e-9,
          `r=${r} → r'=${rp} outside [${R0}, ${R0 + S / 2}]`,
        );
      }
    }
  }
});

test("#635 spiralVortex: sampled operator norm ≤ (|a|+sqrt(a²+4))/2, exactly attained", () => {
  // The declared factor is the largest singular value of the SHEAR
  // [[1,0],[a,1]] — this gate originally ran against sqrt(1+a²) (the naive
  // per-column reading) and FAILED at 1.3440095 vs 1.1662, which is L(0.6)
  // to 7 digits: the test corrected its own op's derivation. Keep it exact.
  const rand = mulberry32(0x5417);
  for (const a of [0.6, -1.4, 2.0]) {
    const L = (Math.abs(a) + Math.sqrt(a * a + 4)) / 2;
    for (const axis of [0, 1, 2]) {
      let worst = 0;
      for (let i = 0; i < 12000; i++) {
        const r = 10 ** (rand() * 4 - 2.5); // includes the near-axis winding zone
        const dir = [rand() - 0.5, rand() - 0.5, rand() - 0.5];
        const n = Math.hypot(...dir) || 1;
        const p = dir.map((x) => (x / n) * r);
        const eps = Math.min(1e-4, r * 1e-3); // stay local at tiny radii — the
        // winding means a fixed-size step at r≪eps measures the chord of a
        // multi-turn arc, which underestimates; locality is what the operator
        // norm bounds.
        const dp = [rand() - 0.5, rand() - 0.5, rand() - 0.5];
        const dn = Math.hypot(...dp) || 1;
        const q = p.map((x, j) => x + (dp[j] / dn) * eps);
        const A = run("spiralVortex", [a, axis], p);
        const B = run("spiralVortex", [a, axis], q);
        worst = Math.max(worst, dist(A, B) / eps);
        assert.ok(
          Math.abs(A.w - L) < 1e-12,
          "w carries exactly (|a|+sqrt(a²+4))/2, unconditionally",
        );
      }
      assert.ok(
        worst <= L * (1 + 1e-6),
        `a=${a} axis=${axis}: ${worst} > ${L}`,
      );
      // The bound is EXACT, not conservative: radial steps attain it.
      assert.ok(
        worst >= L * 0.98,
        `a=${a} axis=${axis}: bound never approached (${worst} vs ${L}) — is the declared factor too loose?`,
      );
    }
  }
});

test("#635 spiralVortex: pure rotation at fixed radius (tangential isometry)", () => {
  // Points at the same in-plane radius map to the same radius — the op never
  // changes |in-plane component| or the axis coordinate.
  const rand = mulberry32(0x777);
  for (let i = 0; i < 5000; i++) {
    const p = [rand() * 4 - 2, rand() * 4 - 2, rand() * 4 - 2];
    const s = run("spiralVortex", [1.3, 1], p);
    assert.ok(
      Math.abs(Math.hypot(s.x, s.z) - Math.hypot(p[0], p[2])) < 1e-9,
      "in-plane radius preserved",
    );
    assert.equal(s.y, p[1], "axis coordinate untouched");
  }
});

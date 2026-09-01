// #418 — the CPU DE finalize must key off the SAME selector the GPU uses.
// The GPU (shader.js mapDE_single) picks log-DE vs IFS r/|w| from G.colA.w =
// effectiveDeOption(f) (written by capturesettle.js). The CPU (cpu.js makeDE)
// used to key off isEscapeTime, so a user-set deOption:0 on a non-escape-time
// stack was a silent no-op — the CPU/ASCII tier, splat CPU capture, the vary
// oracle and auto-levels all marched r/|w| while the GPU marched log-DE.
//
// Named *.test.mjs so it stays out of the apps' served *.js surface. Run: node --test core/deoption.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";
import { makeDE } from "./cpu.js";
import { makeOrbit } from "./cpuorbit.js";
import { effectiveDeOption, isEscapeTime } from "./operators.js";
import { hybridDeFamily } from "./stability.js";
import { measure } from "./evaluate.js";

// A non-escape-time IFS stack (the #32 reporter's class). isEscapeTime === false,
// so under the old code deOption was inert.
const IFS = {
  name: "deopt-ifs",
  addC: false,
  iters: 24,
  ops: [
    { key: "icosaFold", values: [] },
    { key: "kaleido", values: [5, 0, 1] },
    { key: "translate", values: [-1.17, -0.98, 0.05] },
    { key: "scale", values: [1.16] },
  ],
};
const PTS = [
  [0.2, 0.1, 0.3],
  [-0.4, 0.25, -0.1],
  [0.05, -0.6, 0.15],
];
const REL = (a, b) => Math.abs(a - b) / Math.max(1e-9, Math.abs(b));

// The GPU finalize, computed in JS from the SHARED orbit (r, aw). This is the
// exact WGSL branch shader.js mapDE_single emits for each effectiveDeOption.
function gpuDE(f, p) {
  const eff = effectiveDeOption(f);
  const { r, aw } = makeOrbit(f)(...p);
  if (eff === 0) return (0.5 * Math.log(Math.max(r, 1e-9)) * r) / aw;
  return r / aw; // eff 1 | 2
}

test("#418 non-escape stack: deOption 0 marches log-DE (was inert)", () => {
  assert.equal(isEscapeTime(IFS), false);
  const f = { ...IFS, deOption: 0 };
  assert.equal(effectiveDeOption(f), 0);
  const de = makeDE(f);
  for (const p of PTS) assert.ok(REL(de(...p), gpuDE(f, p)) < 1e-9);
});

test("#418 non-escape stack: deOption 1 and 2 march IFS r/|w|", () => {
  for (const deOption of [1, 2]) {
    const f = { ...IFS, deOption };
    const de = makeDE(f);
    for (const p of PTS) assert.ok(REL(de(...p), gpuDE(f, p)) < 1e-9);
  }
});

test("#418 deOption is no longer a no-op: 0 differs from 2 on the CPU", () => {
  const de0 = makeDE({ ...IFS, deOption: 0 });
  const de2 = makeDE({ ...IFS, deOption: 2 });
  // At least one probe must differ materially — the whole point of the fix.
  const anyDiff = PTS.some((p) => REL(de0(...p), de2(...p)) > 1e-3);
  assert.ok(anyDiff, "deOption 0 and 2 must produce different CPU DEs");
});

test("#418 escape-time stack: log-DE regardless of stored deOption", () => {
  const BULB = {
    name: "deopt-bulb",
    addC: true,
    iters: 8,
    ops: [{ key: "mandelbulbPower", values: [8] }],
  };
  assert.equal(isEscapeTime(BULB), true);
  for (const deOption of [0, 2]) {
    const f = { ...BULB, deOption };
    assert.equal(effectiveDeOption(f), 0); // escape wins the selector
    const de = makeDE(f);
    for (const p of PTS) assert.ok(REL(de(...p), gpuDE(f, p)) < 1e-9);
  }
});

test("#418 numeric-DE op forces deOption 3 (finite-difference branch)", () => {
  const NUM = {
    name: "deopt-num",
    addC: true,
    iters: 8,
    deOption: 2,
    ops: [{ key: "bristorBrot", values: [] }],
  };
  assert.equal(effectiveDeOption(NUM), 3); // isNumericDE overrides the stored 2
  // The deOption-3 finite-difference branch (makeDE flat path) is selected and
  // runs without throwing. (Its value can be NaN where the orbit diverges — that
  // is the numeric DE's own characteristic, matching the GPU numeric variant, not
  // a regression of this fix.)
  const de = makeDE(NUM);
  for (const p of PTS) assert.equal(typeof de(...p), "number");
});

test("#418 hybrid ifs-family stack honors deOption 0 (log-DE)", () => {
  const HYB = {
    name: "deopt-hyb",
    iters: 12,
    deOption: 0,
    addC: false,
    ops: [
      { key: "boxFold", values: [1.0] },
      { key: "scale", values: [1.6] },
    ],
    hybrid: {
      b: { ops: [{ key: "sphereFold", values: [] }], addC: false },
      schedule: { a: 1, b: 1 },
    },
  };
  assert.equal(hybridDeFamily(HYB), "ifs");
  const de = makeDE(HYB);
  const run = makeOrbit(HYB);
  for (const p of PTS) {
    const { r, aw } = run(...p);
    const logExp = (0.5 * Math.log(Math.max(r, 1e-9)) * r) / aw;
    assert.ok(REL(de(...p), logExp) < 1e-9); // log-DE, not r/|w|
  }
});

test("#418 the vary/Remix oracle (evaluate.measure) honors deOption", () => {
  // Regression for the SECOND copy of the selector bug (core/evaluate.js
  // makeProbes): before the fix, measure() marched r/|w| for BOTH deOptions on a
  // non-escape stack, so a deOption:0 formula reported hits=0 (blank) and
  // vary.js isSound rejected it. The oracle must now find the log-DE surface.
  const m0 = measure({ ...IFS, deOption: 0 });
  const m2 = measure({ ...IFS, deOption: 2 });
  assert.ok(m0.hits > 0, "deOption:0 must find a surface via the log-DE");
  assert.notEqual(m0.hits, m2.hits); // deOption is no longer inert in the oracle
});

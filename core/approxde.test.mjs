// Guards for the approximate-DE flag (APPROX_DE.md) + its first tagged op
// (polygonFold): the recursive classifier, the isDeSound exclusion, the step
// policy compositions, and the polygonFold closed form.
//
// Named *.test.mjs so sync_web_core.sh skips it. Run: node --test core/approxde.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";
import { isApproxDE, isDeSound, deSoundExceptApprox } from "./operators.js";
import {
  resolveDeScale,
  qualityParams,
  sceneDeScale,
  APPROX_DESCALE_MUL,
} from "./renderpolicy.js";
import { applyOp } from "./cpuorbit.js";

const PF = { key: "polygonFold", values: [6, 1, 0] };
const SOUND = { key: "boxFold", values: [1.0] };
const flat = (...ops) => ({ ops, iters: 12, deOption: 2 });

test("isApproxDE walks flat ops, hybrid slot B, and scene objects (recursive)", () => {
  assert.equal(isApproxDE(flat(SOUND)), false);
  assert.equal(isApproxDE(flat(SOUND, PF)), true);
  // muted tagged op does not count (scene-mute precedent)
  assert.equal(isApproxDE(flat(SOUND, { ...PF, muted: true })), false);
  // hybrid slot B — the flat isDeSound/isNumericDE precedent CANNOT see this
  const hybrid = {
    ops: [SOUND],
    hybrid: { b: { ops: [PF] }, schedule: { a: 1, b: 1 } },
  };
  assert.equal(isApproxDE(hybrid), true);
  // scene object
  const scene = {
    ops: [],
    objects: [
      { objType: 0, ops: [SOUND] },
      { objType: 0, ops: [PF] },
    ],
  };
  assert.equal(isApproxDE(scene), true);
});

test("isDeSound excludes tagged ops even though polygonFold's wRule is W_MUL_K", () => {
  const f = flat(SOUND, PF);
  assert.equal(deSoundExceptApprox(f), true, "wRule-wise the formula is sound");
  assert.equal(isDeSound(f), false, "the deApprox tag must veto soundness");
  assert.equal(isDeSound(flat(SOUND)), true, "untagged folds stay sound");
});

test("resolveDeScale: x0.5 for approx formulas, identity otherwise", () => {
  assert.equal(
    resolveDeScale(0.85, flat(SOUND, PF)),
    0.85 * APPROX_DESCALE_MUL,
  );
  assert.equal(resolveDeScale(0.85, flat(SOUND)), 0.85);
  assert.equal(resolveDeScale(0.5, null), 0.5);
});

test("qualityParams compositions: clean, approx, loose x approx, deep-zoom floor", () => {
  const clean = qualityParams(flat(SOUND), {});
  assert.equal(
    clean.deScale,
    0.5,
    "sound full tier untouched (degeneracy anchor)",
  );
  const approx = qualityParams(flat(SOUND, PF), {});
  assert.equal(approx.deScale, 0.25, "full 0.5 -> 0.25");
  assert.equal(approx.steps, clean.steps * 2, "march budget doubles");
  // loose (IFS scale < 2) x approx: 0.3 -> 0.15
  const loose = flat(
    { key: "boxFold", values: [1.0] },
    { key: "scale", values: [1.5] },
    PF,
  );
  assert.equal(qualityParams(loose, {}).deScale, 0.3 * APPROX_DESCALE_MUL);
  // deep zoom: the multiplier applies AFTER the sqrt-depth clamp
  // (APPROX_DE.md §3) — deep approx = deep clean x 0.5 at any depth...
  const deepClean = qualityParams(flat(SOUND), { dist: 1e-9 });
  const deep = qualityParams(flat(SOUND, PF), { dist: 1e-9 });
  assert.ok(
    Math.abs(deep.deScale - deepClean.deScale * APPROX_DESCALE_MUL) < 1e-12,
    `deep composition: ${deep.deScale} vs ${deepClean.deScale}`,
  );
  // ...and where the 0.25 depth floor actually binds (loose base 0.3 at
  // DEPTH_CAP), the approx floor is 0.125.
  const looseDeep = qualityParams(loose, { dist: 1e-9 });
  assert.ok(
    Math.abs(looseDeep.deScale - 0.125) < 1e-12,
    `approx depth floor: ${looseDeep.deScale}`,
  );
});

test("sceneDeScale: one approx object tightens the whole scene (carve-rule shape)", () => {
  const soundObj = { objType: 0, ops: [SOUND], combine: 0 };
  const approxObj = { objType: 0, ops: [PF], combine: 0 };
  assert.equal(sceneDeScale([soundObj, soundObj]), 0.5);
  assert.equal(sceneDeScale([soundObj, approxObj]), 0.5 * APPROX_DESCALE_MUL);
  // carve + approx stack: 0.25 x 0.5
  const carveApprox = { objType: 0, ops: [PF], combine: 2 };
  assert.equal(
    sceneDeScale([soundObj, carveApprox]),
    0.25 * APPROX_DESCALE_MUL,
  );
});

test("polygonFold closed form: radius remap, angle preserved, w *= f", () => {
  const run = (values, pt) => {
    const s = { ...pt };
    applyOp("polygonFold", values, s);
    return s;
  };
  const pt = { x: 0.5, y: 0.4, z: 0.7, w: 1.3 };
  const a = Math.atan2(pt.y, pt.x);
  const sector = (2 * Math.PI) / 6;
  const th = a - sector * Math.floor(a / sector + 0.5);
  const c = Math.cos(th);
  // s = +1: r *= cos(theta_local), toward the circle
  const to = run([6, 1, 0], pt);
  assert.ok(
    Math.abs(to.x - pt.x * c) < 1e-12 && Math.abs(to.y - pt.y * c) < 1e-12,
  );
  assert.ok(Math.abs(to.z - pt.z) < 1e-15, "free axis untouched");
  assert.ok(Math.abs(to.w - pt.w * c) < 1e-12, "w tracks the radial factor");
  assert.ok(
    Math.abs(Math.atan2(to.y, to.x) - a) < 1e-12,
    "angle preserved (not a kaleido)",
  );
  // s = -1: r *= 1/cos — the inverse map; round-trip is identity
  const back = run([6, -1, 0], to);
  for (const k of ["x", "y", "z", "w"])
    assert.ok(Math.abs(back[k] - pt[k]) < 1e-9, `to/from round-trip ${k}`);
  // s = 0: identity
  const id = run([6, 0, 0], pt);
  for (const k of ["x", "y", "z", "w"]) assert.equal(id[k], pt[k]);
});

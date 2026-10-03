import { test } from "node:test";
import assert from "node:assert/strict";
import { cruiseAdvance, CRUISE_K } from "./cruise.js";

test("closing in is asymptotic: fixed fraction per second, dt-slicing independent", () => {
  // 1 s in one step vs 60 steps must shrink dist identically (exp composes).
  const one = cruiseAdvance({ dir: 1, hasHit: true, dist: 10 }, 1000).distFactor;
  let sliced = 1;
  for (let i = 0; i < 60; i++)
    sliced *= cruiseAdvance({ dir: 1, hasHit: true, dist: 10 }, 1000 / 60).distFactor;
  assert.ok(Math.abs(one - sliced) < 1e-12);
  assert.ok(Math.abs(one - Math.exp(-CRUISE_K)) < 1e-12);
  assert.ok(one > 0, "never reaches or crosses the surface");
});

test("open-space drift is scale-invariant: k·dist per second, dist untouched", () => {
  const near = cruiseAdvance({ dir: 1, hasHit: false, dist: 0.001 }, 500);
  const far = cruiseAdvance({ dir: 1, hasHit: false, dist: 100 }, 500);
  assert.equal(near.distFactor, 1);
  assert.equal(far.distFactor, 1);
  // Drift ∝ dist ⇒ the same fraction of the current scale.
  assert.ok(Math.abs(near.drift / 0.001 - far.drift / 100) < 1e-12);
  assert.ok(Math.abs(far.drift - CRUISE_K * 100 * 0.5) < 1e-12);
});

test("reverse is a plain scale-invariant back-out and exactly inverts fly-in", () => {
  const out = cruiseAdvance({ dir: -1, hasHit: true, dist: 5 }, 700).distFactor;
  const inn = cruiseAdvance({ dir: 1, hasHit: true, dist: 5 }, 700).distFactor;
  assert.ok(out > 1);
  assert.ok(Math.abs(out * inn - 1) < 1e-12);
});

test("zero/negative dt is a no-op frame", () => {
  for (const dt of [0, -5]) {
    const s = cruiseAdvance({ dir: 1, hasHit: true, dist: 3 }, dt);
    assert.equal(s.distFactor, 1);
    assert.equal(s.drift, 0);
  }
});

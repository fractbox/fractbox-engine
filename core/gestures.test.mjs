import { test } from "node:test";
import assert from "node:assert/strict";
import {
  wheelZoomFactor,
  makeVelocityTracker,
  makeGlide,
  GLIDE_MIN_SPEED,
  twoFingerDelta,
} from "./gestures.js";

// ── wheelZoomFactor (CAMERA_CONTROLS.md §3.1) ────────────────────────────────

test("sign convention: deltaY > 0 zooms out (factor > 1), < 0 in, 0 is identity", () => {
  assert.ok(wheelZoomFactor(120) > 1);
  assert.ok(wheelZoomFactor(-120) < 1);
  assert.equal(wheelZoomFactor(0), 1);
});

test("proportional to delta: bigger scroll → stronger zoom (monotonic)", () => {
  let prev = 0;
  for (const d of [1, 5, 20, 60, 120]) {
    const f = wheelZoomFactor(d);
    assert.ok(f > prev, `factor(${d}) should exceed factor of smaller delta`);
    prev = f;
  }
});

test("in and out are exact inverses — a wheel round trip doesn't drift", () => {
  // The old fixed pair (×1.1 / ×0.9) lost 1% per in+out cycle.
  for (const d of [3, 40, 100]) {
    assert.ok(Math.abs(wheelZoomFactor(d) * wheelZoomFactor(-d) - 1) < 1e-12);
  }
});

test("equal deltas compose: f(a)·f(b) = f(a+b) — event coalescing is neutral", () => {
  const lhs = wheelZoomFactor(30) * wheelZoomFactor(50);
  assert.ok(Math.abs(lhs - wheelZoomFactor(80)) < 1e-12);
});

test("one classic wheel notch (~120 px) lands near the old ×1.1 step", () => {
  const f = wheelZoomFactor(120);
  assert.ok(f > 1.08 && f < 1.18, `notch factor ${f} should approximate the old feel`);
});

test("per-event clamp: a flicked wheel can't teleport (|factor−1| ≤ 0.25)", () => {
  assert.equal(wheelZoomFactor(1e6), 1.25);
  assert.equal(wheelZoomFactor(-1e6), 1 / 1.25);
});

test("DOM_DELTA_LINE mode scales by the line height (3 lines ≡ 48 px)", () => {
  assert.equal(wheelZoomFactor(3, 1), wheelZoomFactor(48, 0));
});

test("DOM_DELTA_PAGE mode scales by the given page height", () => {
  assert.equal(wheelZoomFactor(0.5, 2, 600), wheelZoomFactor(300, 0));
});

// ── makeVelocityTracker (§3.3) ───────────────────────────────────────────────

test("tracker converges to a constant drag velocity, at any frame rate", () => {
  for (const dt of [8, 16, 33]) {
    const v = makeVelocityTracker();
    // 100 °/s in x for 400 ms of samples.
    for (let t = 0; t < 400; t += dt) v.push((100 * dt) / 1000, 0, dt);
    const { vx, vy } = v.velocity();
    assert.ok(Math.abs(vx - 100) < 1, `vx=${vx} @ dt=${dt}`);
    assert.equal(vy, 0);
  }
});

test("tracker primes on the first sample (short flicks aren't under-read)", () => {
  const v = makeVelocityTracker();
  v.push(2, 0, 16); // one 16 ms sample at 125 °/s
  assert.ok(v.speed() > 100, `speed=${v.speed()} should read the flick immediately`);
});

test("tracker adapts to a direction reversal within ~the window", () => {
  const v = makeVelocityTracker(80);
  for (let i = 0; i < 20; i++) v.push(1.6, 0, 16); // +100 °/s
  for (let i = 0; i < 20; i++) v.push(-1.6, 0, 16); // −100 °/s for 320 ms ≫ 80 ms
  assert.ok(v.velocity().vx < -80, `vx=${v.velocity().vx} should have flipped`);
});

test("tracker ignores zero/negative dt and resets clean", () => {
  const v = makeVelocityTracker();
  v.push(5, 5, 0);
  v.push(5, 5, -3);
  assert.equal(v.speed(), 0);
  v.push(1.6, 0, 16);
  v.reset();
  assert.equal(v.speed(), 0);
});

// ── makeGlide (§3.3) ─────────────────────────────────────────────────────────

test("glide total travel is v·τ — and is dt-slicing independent", () => {
  const travel = (dts) => {
    const g = makeGlide(100, 0, { tau: 180, stop: 0.001 });
    let sum = 0;
    for (const dt of dts) sum += g.step(dt).dYawDeg;
    return sum;
  };
  const fine = travel(Array(2000).fill(1)); // 2 s in 1 ms steps
  const coarse = travel(Array(20).fill(100)); // 2 s in 100 ms steps
  const analytic = (100 * 180) / 1000; // v·τ = 18°
  assert.ok(Math.abs(fine - analytic) < 0.05, `fine=${fine}`);
  assert.ok(Math.abs(fine - coarse) < 1e-9, "same wall time ⇒ same travel, any slicing");
});

test("glide deactivates when the speed magnitude falls below stop", () => {
  const g = makeGlide(30, 40, { tau: 100, stop: 2 }); // |v| = 50 °/s
  let steps = 0;
  while (g.step(16).active) if (++steps > 1000) break;
  // e-folding: 50·e^(−t/100) < 2 ⇒ t ≈ 322 ms ≈ 20 steps of 16 ms.
  assert.ok(steps > 10 && steps < 40, `deactivated after ${steps} steps`);
});

test("glide preserves the flick direction (yaw/pitch ratio)", () => {
  const g = makeGlide(60, -30, { tau: 180, stop: 1 });
  const s = g.step(50);
  assert.ok(Math.abs(s.dYawDeg / s.dPitchDeg + 2) < 1e-9); // 60/−30 = −2
});

test("GLIDE_MIN_SPEED is a sane flick threshold", () => {
  assert.ok(GLIDE_MIN_SPEED > 5 && GLIDE_MIN_SPEED < 200);
});

// ── twoFingerDelta (§3.4) ────────────────────────────────────────────────────

test("pure translation: both fingers move together → pan only, zoom 1", () => {
  const g = twoFingerDelta(
    { x: 100, y: 100 },
    { x: 200, y: 100 },
    { x: 110, y: 95 },
    { x: 210, y: 95 },
  );
  assert.equal(g.panX, 10);
  assert.equal(g.panY, -5);
  assert.equal(g.zoom, 1);
});

test("pure pinch: fingers spread symmetrically → zoom in (<1), no pan", () => {
  const g = twoFingerDelta(
    { x: 140, y: 100 },
    { x: 160, y: 100 },
    { x: 120, y: 100 },
    { x: 180, y: 100 },
  );
  assert.equal(g.panX, 0);
  assert.equal(g.panY, 0);
  assert.ok(Math.abs(g.zoom - 20 / 60) < 1e-12); // old/new distance ratio
});

test("combined gesture decomposes into both components at once", () => {
  // Spread ×2 AND drift the centroid +30/+10.
  const g = twoFingerDelta(
    { x: 90, y: 100 },
    { x: 110, y: 100 },
    { x: 110, y: 110 },
    { x: 150, y: 110 },
  );
  assert.equal(g.panX, 30);
  assert.equal(g.panY, 10);
  assert.equal(g.zoom, 0.5);
});

test("degenerate (stacked pointers) never yields a 0/∞ zoom", () => {
  const p = { x: 100, y: 100 };
  assert.equal(twoFingerDelta(p, p, { x: 120, y: 100 }, { x: 80, y: 100 }).zoom, 1);
  assert.equal(twoFingerDelta({ x: 120, y: 100 }, { x: 80, y: 100 }, p, p).zoom, 1);
});

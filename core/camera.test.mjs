// Orbit camera basis — the pole fix for #463 (tesselava "Top view isn't
// really straight down"). tesselava's Top/Front/Side buttons used to pin
// pitchDeg to 89 instead of a genuine 90 because cam.basis() went degenerate
// exactly at the pole (cross(fwd, worldUp) = 0) and fell back to a FIXED
// [1,0,0] right vector — which is wrong for any yaw other than 0, and even
// at yaw=0 has the opposite handedness from every neighbouring pitch. These
// pins guard the replacement: a yaw-derived limit that stays continuous
// through the pole, so an exact 90° needs no epsilon fudge.
// Run: node --test core/camera.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";
import {
  makeCamera,
  clampDist,
  MAX_DIST,
  MIN_DIST,
  PT_MIN_DIST,
} from "./camera.js";
import { magnificationFor, REF_DIST } from "./renderpolicy.js";

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const mag = (v) => Math.hypot(v[0], v[1], v[2]);

test("#463: basis() stays orthonormal looking straight down at every yaw", () => {
  for (const yawDeg of [0, 45, 90, 180, -30, 359]) {
    const cam = makeCamera({ yawDeg, pitchDeg: 90, dist: 5 });
    const b = cam.basis();
    assert.ok(Math.abs(mag(b.right) - 1) < 1e-9, `right unit at yaw ${yawDeg}`);
    assert.ok(Math.abs(mag(b.up) - 1) < 1e-9, `up unit at yaw ${yawDeg}`);
    assert.ok(
      Math.abs(dot(b.right, b.fwd)) < 1e-9,
      `right⊥fwd at yaw ${yawDeg}`,
    );
    assert.ok(Math.abs(dot(b.right, b.up)) < 1e-9, `right⊥up at yaw ${yawDeg}`);
  }
});

test("#463: basis() at pitch 90 is the LIMIT of pitch→90, not a fixed axis", () => {
  // The old fallback hardcoded right=[1,0,0] regardless of yaw. The fix
  // derives it from yaw, so it must agree with what a pitch that merely
  // APPROACHES 90 (no degenerate branch involved) already gives — continuity
  // through the pole, not a jump onto an arbitrary axis.
  for (const yawDeg of [0, 45, 90, 137, -60]) {
    const near = makeCamera({ yawDeg, pitchDeg: 89.9999, dist: 5 }).basis();
    const pole = makeCamera({ yawDeg, pitchDeg: 90, dist: 5 }).basis();
    const err = mag([
      near.right[0] - pole.right[0],
      near.right[1] - pole.right[1],
      near.right[2] - pole.right[2],
    ]);
    assert.ok(
      err < 1e-4,
      `right continuous through the pole at yaw ${yawDeg} (err ${err})`,
    );
  }
});

// The concrete regression: at yaw=0 (tesselava's "Top" button) the OLD fixed
// fallback returned [1,0,0] — the opposite sign from the true limit, which
// this pins directly so a revert back to a hardcoded axis fails loudly.
test("#463: yaw=0 straight-down right vector is [-1,0,0], not the old [1,0,0]", () => {
  const b = makeCamera({ yawDeg: 0, pitchDeg: 90, dist: 5 }).basis();
  assert.ok(Math.abs(b.right[0] - -1) < 1e-9);
  assert.ok(Math.abs(b.right[1]) < 1e-9);
  assert.ok(Math.abs(b.right[2]) < 1e-9);
});

// ── clampDist — the dist invariant every write is funneled through ───────────
// (zoom-oob) The soft brake in preview.js only *approximates* this: it gates on
// PRE-zoom headroom, so one oversized step lands (or eases) past the wall.
// clampDist is the hard backstop, run at the point the value is stored.

test("clampDist rejects non-finite writes, keeping the last good value", () => {
  // A NaN/Inf factor must never stick — a NaN dist blacks out the render.
  assert.equal(clampDist(NaN, 5, null), 5);
  assert.equal(clampDist(Infinity, 5, null), 5);
  assert.equal(clampDist(-Infinity, 5, null), 5);
});

test("clampDist enforces [PT_MIN_DIST, MAX_DIST] when no floor is supplied", () => {
  // Absent a consumer floor, only the ABSOLUTE engine floor binds — so a bare
  // camera (unit tests, pre-install) can still reach any legitimate depth.
  assert.equal(clampDist(1e-40, 5, null), PT_MIN_DIST); // below the absolute floor
  assert.equal(clampDist(1e-20, 5, null), 1e-20); // deep but legal → untouched
  assert.equal(clampDist(1e9, 5, null), MAX_DIST); // absurd zoom-out
  assert.equal(clampDist(14, 5, null), 14); // in range → untouched
});

test("clampDist honors a supplied floor — a shallow wall OR a deep pt floor", () => {
  const wall = 2e-4;
  assert.equal(clampDist(1e-8, 5, wall), wall); // a past-the-wall write is lifted back
  assert.equal(clampDist(3e-4, 5, wall), 3e-4); // above the wall passes through
  // Perturbation tier: the floor can be far BELOW MIN_DIST — the clamp must
  // allow legitimate deep zoom, not force everything up to the df64-era floor.
  assert.equal(clampDist(1e-30, 5, PT_MIN_DIST), PT_MIN_DIST);
  assert.equal(clampDist(1e-25, 5, PT_MIN_DIST), 1e-25);
});

test("clampDist falls back to the absolute floor for a nonpositive / non-finite floor", () => {
  for (const bad of [0, -1, NaN, Infinity]) {
    assert.equal(clampDist(1e-40, 5, bad), PT_MIN_DIST, `floor=${bad}`);
  }
});

// ── cam.dist accessor — no writer, any path, can persist out-of-bounds ────────

test("cam.zoom can't cross the wall floor, however hard you zoom in", () => {
  const cam = makeCamera({ dist: 10 });
  const WALL = 1e-4;
  cam.distFloor = () => WALL;
  for (let i = 0; i < 300; i++) cam.zoom(0.5); // relentless zoom-in (pinch/wheel burst)
  assert.equal(cam.dist, WALL);
});

test("a NaN zoom factor leaves dist at its last good value (no black-out)", () => {
  const cam = makeCamera({ dist: 12 });
  cam.zoom(NaN); // e.g. a 0/0 gesture factor
  assert.equal(cam.dist, 12);
  cam.dist = NaN; // a direct bad write, too
  assert.equal(cam.dist, 12);
  cam.dist = Infinity;
  assert.equal(cam.dist, 12);
});

test("a direct out-of-bounds dist write (share-load / frameTo) clamps on store", () => {
  // decodeShare → frameTo assigns cam.dist directly; the accessor clamps it, so
  // an out-of-bounds SHARE self-heals on load without touching the wire format.
  const cam = makeCamera({ dist: 10 });
  cam.distFloor = () => 5e-4;
  cam.dist = 1e-9; // share lands past the wall
  assert.equal(cam.dist, 5e-4);
  cam.dist = 1e9; // share absurdly zoomed out
  assert.equal(cam.dist, MAX_DIST);
});

test("the accessor never blocks a legitimate perturbation-depth write", () => {
  // Regression guard: pt-eligible formulas zoom to ~1e-30. With the pt floor
  // installed the accessor must let dist reach it, not clamp up to MIN_DIST.
  const cam = makeCamera({ dist: 10 });
  cam.distFloor = () => PT_MIN_DIST;
  cam.dist = 1e-28;
  assert.equal(cam.dist, 1e-28);
  cam.zoom(1e-20, PT_MIN_DIST); // an enormous but legitimate deep zoom-in
  assert.ok(cam.dist >= PT_MIN_DIST && cam.dist < 1e-20);
});

test("an out-of-bounds INITIAL dist is clamped by makeCamera", () => {
  assert.equal(makeCamera({ dist: 1e9 }).dist, MAX_DIST);
  assert.equal(makeCamera({ dist: 1e-40 }).dist, PT_MIN_DIST); // below the absolute floor
  assert.equal(makeCamera({ dist: NaN }).dist, 14); // NaN → the safe default
});

test("distFloor is non-enumerable — it never leaks into {...cam} spreads", () => {
  // reactor.ts does `{ ...cam, dist: baseDist }`; a serialized/cloned function
  // key would be a footgun, so distFloor stays off the enumerable surface.
  const cam = makeCamera();
  cam.distFloor = () => 1;
  assert.ok(!Object.keys(cam).includes("distFloor"));
  assert.equal({ ...cam }.distFloor, undefined);
  assert.ok(Object.keys(cam).includes("dist")); // dist DOES round-trip (a value)
  assert.equal({ ...cam }.dist, cam.dist);
});

test("the wall-floor law halts zoom-in exactly at headroom == brakeStop", () => {
  // Mirror preview.js wallFloorDist: headroom is LINEAR in dist, so
  // floor = dist·STOP/headroom(dist) is dist-independent (= STOP/slope) and
  // halts the zoom right at the wall — the same law the soft brake stops at.
  const STOP = 3; // f32-only brakeStop
  const slope = 5000; // headroom = dist·slope for some formula/fov/quantum
  const headroom = (d) => d * slope;
  const cam = makeCamera({ dist: 8 });
  cam.distFloor = () => {
    const h = headroom(cam.dist);
    return h > 0 ? (cam.dist * STOP) / h : null;
  };
  const wall = STOP / slope;
  for (let i = 0; i < 500; i++) cam.zoom(0.7); // dive at the wall
  assert.ok(
    Math.abs(cam.dist - wall) < wall * 1e-9,
    `dist=${cam.dist} wall=${wall}`,
  );
  assert.ok(
    headroom(cam.dist) >= STOP - 1e-9,
    "never past the detail-limit wall",
  );
});

test("zoom-out is never blocked by the wall floor (only the MAX ceiling binds)", () => {
  const cam = makeCamera({ dist: 1 });
  cam.distFloor = () => 1e-4;
  for (let i = 0; i < 100; i++) cam.zoom(1.5); // back away
  assert.equal(cam.dist, MAX_DIST);
});

// ── the pt floor and the READOUT must agree ──────────────────────────────────
// A camera on the perturbation tier may descend to PT_MIN_DIST. That range is
// only usable if the number on screen keeps counting through it — the deep-zoom
// pin was exactly this pair coming apart: the clamp allowed 1e-27, the badge's
// own floor stopped at 1e-12, and the feature looked dead at ×2.4·10¹³.
test("a pt-floored descent stays legal AND readable all the way down", () => {
  const cam = makeCamera({ dist: 24 });
  cam.distFloor = () => PT_MIN_DIST; // pt-eligible formula, pt tier available
  for (let i = 0; i < 400; i++) cam.zoom(0.8, PT_MIN_DIST);
  assert.equal(cam.dist, PT_MIN_DIST, "descent must reach the pt floor");
  // the same camera, read through the shared law: no saturation on the way
  const seen = [];
  const c2 = makeCamera({ dist: 24 });
  c2.distFloor = () => PT_MIN_DIST;
  for (let i = 0; i < 200; i++) {
    c2.zoom(0.5, PT_MIN_DIST);
    seen.push(magnificationFor(c2.dist));
  }
  const deep = seen.filter((m) => m > 1e20);
  assert.ok(deep.length > 0, "the pt range must be reachable AND reportable");
  // strictly increasing until the floor binds — never a frozen plateau below it
  for (let i = 1; i < seen.length; i++) {
    if (seen[i - 1] >= REF_DIST / PT_MIN_DIST) break;
    assert.ok(seen[i] > seen[i - 1], `magnification froze at step ${i}`);
  }
});

test("MIN_DIST (the df64-era default) still bounds a caller that asks for it", () => {
  // Non-pt formulas must NOT deepen by accident: cam.zoom's default floor is
  // unchanged, so a df64-only descent still stops at its own tier's bound.
  const cam = makeCamera({ dist: 24 });
  for (let i = 0; i < 400; i++) cam.zoom(0.8); // no explicit minDist
  assert.equal(cam.dist, MIN_DIST);
});

// Capture volumes (CAPTURE_VOLUME_SHAPES.md) — the support/inside pair and the
// property that motivated them: the capture is BOUNDED BY THE VOLUME.
// Run: node --test core/capturevolume.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  volSupport,
  volInside,
  volExt,
  volRayInterval,
  volBasis,
  VOL_BOX,
  VOL_ELLIPSOID,
  VOL_CYLINDER,
} from "./capturevolume.js";
import { captureView, fibonacciDir } from "./splatcapture.js";

const unit = (v) => {
  const L = Math.hypot(...v);
  return v.map((c) => c / L);
};

test("volExt: a frame with no ext is the uniform volume ±radius", () => {
  assert.deepEqual(volExt({ radius: 2 }), [2, 2, 2]);
  assert.deepEqual(volExt({ radius: 2, ext: [1, 3, 5] }), [1, 3, 5]);
  // a degenerate ext must not silently produce a zero-width capture
  assert.deepEqual(volExt({ radius: 2, ext: [1, 0, 5] }), [2, 2, 2]);
  assert.deepEqual(volExt({ radius: 2, ext: [1, 2] }), [2, 2, 2]);
});

test("volSupport box: L1 along the axis, √3·e on the body diagonal", () => {
  const f = { radius: 1, ext: [1, 1, 1] };
  assert.ok(Math.abs(volSupport(f, [1, 0, 0]) - 1) < 1e-12, "axis → e");
  // THE fact that made 'a uniform ext reproduces the old window' false
  const diag = volSupport(f, unit([1, 1, 1]));
  assert.ok(
    Math.abs(diag - Math.sqrt(3)) < 1e-12,
    `body diagonal → √3 (${diag})`,
  );
  // a cuboid's support follows the axis it leans on
  const c = { radius: 4, ext: [4, 1, 1] };
  assert.ok(Math.abs(volSupport(c, [1, 0, 0]) - 4) < 1e-12);
  assert.ok(Math.abs(volSupport(c, [0, 1, 0]) - 1) < 1e-12);
});

test("volSupport: a sphere's support is direction-independent, a box's is not", () => {
  const s = { radius: 1, ext: [1, 1, 1], kind: VOL_ELLIPSOID };
  for (let k = 0; k < 8; k++)
    assert.ok(Math.abs(volSupport(s, fibonacciDir(k, 8)) - 1) < 1e-12);
  const cyl = { radius: 1, ext: [1, 1, 2], kind: VOL_CYLINDER };
  assert.ok(Math.abs(volSupport(cyl, [0, 0, 1]) - 2) < 1e-12, "along the axis");
  assert.ok(Math.abs(volSupport(cyl, [1, 0, 0]) - 1) < 1e-12, "across it");
});

test("volInside: box / ellipsoid / cylinder disagree exactly at the corner", () => {
  const e = [1, 1, 1];
  const corner = [0.9, 0.9, 0.9]; // inside the box, outside the unit sphere
  assert.equal(
    volInside({ radius: 1, ext: e, kind: VOL_BOX }, ...corner),
    true,
  );
  assert.equal(
    volInside({ radius: 1, ext: e, kind: VOL_ELLIPSOID }, ...corner),
    false,
  );
  // a face-exact hit must survive f32 rounding rather than punch a pinhole
  assert.equal(volInside({ radius: 1, ext: e }, 1, 0, 0), true);
  assert.equal(volInside({ radius: 1, ext: e }, 1.001, 0, 0), false);
  const cyl = { radius: 1, ext: [1, 1, 0.5], kind: VOL_CYLINDER };
  assert.equal(volInside(cyl, 0.5, 0.5, 0.4), true);
  assert.equal(volInside(cyl, 0.5, 0.5, 0.6), false, "outside the slab");
});

// A lattice of spheres: every ray hits something almost immediately from any
// direction, so the captured cloud's boundary is the FRAME's shape and nothing
// else. That is what makes these assertions about the volume, not the geometry.
const latticeDE = (x, y, z) => {
  const m = (v) => v - Math.floor(v) - 0.5;
  return Math.hypot(m(x), m(y), m(z)) - 0.25;
};
const stub = () => [0, 0, 0];

function capturedBBox(frame, { views = 24, res = 24, layers = 12 } = {}) {
  const out = { pos: [], normal: [], albedo: [] };
  for (let k = 0; k < views; k++)
    captureView(latticeDE, stub, frame, fibonacciDir(k, views), res, out, {
      layers,
      maxSteps: 200,
    });
  const n = out.pos.length / 3;
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i++)
    for (let c = 0; c < 3; c++) {
      const p = out.pos[3 * i + c];
      if (p < lo[c]) lo[c] = p;
      if (p > hi[c]) hi[c] = p;
    }
  return { n, lo, hi };
}

// THE regression this feature exists for. Before it, the capture marched a fixed
// ±radius window and kept every hit, so the cloud was a union of oblique slabs
// reaching ~1.74·r (guaranteed) to 2.06·r — the rounded boundary users saw as
// "spherical in the back". It must now stop at the box.
test("a uniform frame captures the BOX, not a ~2·radius blob", () => {
  const frame = {
    center: [0, 0, 0],
    ext: [3, 3, 3],
    radius: 3,
    diag: 6 * Math.sqrt(3),
  };
  const { n, lo, hi } = capturedBBox(frame);
  assert.ok(n > 2000, `lattice filled the volume (${n} hits)`);
  for (let c = 0; c < 3; c++) {
    assert.ok(hi[c] <= 3.01, `axis ${c} max ${hi[c].toFixed(3)} ≤ ext`);
    assert.ok(lo[c] >= -3.01, `axis ${c} min ${lo[c].toFixed(3)} ≥ −ext`);
  }
  // and it really does FILL the box — a clip that also lost the corners would
  // pass the bound above while quietly capturing a smaller volume. The lattice's
  // outermost surface sits at 2.75 (sphere centred 2.5, radius 0.25), so 2.7 is
  // "reaches the face"; anything materially lower means the clip ate the box.
  for (let c = 0; c < 3; c++) {
    assert.ok(hi[c] > 2.7, `axis ${c} reaches the +face (${hi[c].toFixed(3)})`);
    assert.ok(
      lo[c] < -2.7,
      `axis ${c} reaches the −face (${lo[c].toFixed(3)})`,
    );
  }
});

test("a cuboid frame captures a CUBOID (the point of the feature)", () => {
  const frame = {
    center: [0, 0, 0],
    ext: [4, 2, 1],
    radius: 4,
    diag: 2 * Math.hypot(4, 2, 1),
  };
  const { n, lo, hi } = capturedBBox(frame);
  assert.ok(n > 2000, `hits (${n})`);
  const e = [4, 2, 1];
  for (let c = 0; c < 3; c++) {
    assert.ok(
      hi[c] <= e[c] + 0.02,
      `axis ${c} max ${hi[c].toFixed(3)} ≤ ${e[c]}`,
    );
    assert.ok(
      lo[c] >= -e[c] - 0.02,
      `axis ${c} min ${lo[c].toFixed(3)} ≥ −${e[c]}`,
    );
    // lattice surfaces land a quarter-cell inside each extent (see above)
    assert.ok(
      hi[c] > e[c] - 0.3,
      `axis ${c} fills to its own extent (${hi[c].toFixed(3)})`,
    );
  }
  // the short axis is genuinely short — a uniform capture would blow past it
  assert.ok(hi[2] < 1.5, "Z stayed inside the thin axis, not the radius");
});

test("an ellipsoid volume rejects the box corners it would otherwise keep", () => {
  const base = {
    center: [0, 0, 0],
    ext: [3, 3, 3],
    radius: 3,
    diag: 6 * Math.sqrt(3),
  };
  const opt = { views: 16, res: 20, layers: 8 };
  // NOT a hit-count comparison: a tighter volume also gets a tighter window, so
  // at fixed res it is sampled DENSER and the totals come out similar. The
  // property that actually distinguishes the shapes is WHERE the hits are.
  const grab = (frame) => {
    const out = { pos: [], normal: [], albedo: [] };
    for (let k = 0; k < opt.views; k++)
      captureView(
        latticeDE,
        stub,
        frame,
        fibonacciDir(k, opt.views),
        opt.res,
        out,
        {
          layers: opt.layers,
        },
      );
    return out.pos;
  };
  const rad = (P, i) => Math.hypot(P[3 * i], P[3 * i + 1], P[3 * i + 2]);
  const boxP = grab(base);
  const ballP = grab({ ...base, kind: VOL_ELLIPSOID });

  let corners = 0;
  for (let i = 0; i < boxP.length / 3; i++) if (rad(boxP, i) > 3.06) corners++;
  assert.ok(corners > 100, `the box keeps its corners (${corners} beyond r=3)`);

  for (let i = 0; i < ballP.length / 3; i++)
    assert.ok(
      rad(ballP, i) <= 3.06,
      `ellipsoid hit ${rad(ballP, i).toFixed(3)} ≤ 3`,
    );
});

// ── volRayInterval (#450) ────────────────────────────────────────────────────

test("volRayInterval box: the interval is where inside() is true", () => {
  const f = { radius: 1, ext: [2, 1, 3] };
  // Straight down +X through the centre: enters at −2, exits at +2.
  assert.deepEqual(volRayInterval(f, [-10, 0, 0], [1, 0, 0]), [8, 12]);
  // Parallel to the slab and outside it — no t can bring the ray in.
  assert.equal(volRayInterval(f, [0, 5, 0], [1, 0, 0]), null);
  // Parallel and inside stays bounded by the OTHER axes.
  assert.deepEqual(volRayInterval(f, [0, 0.5, 0], [1, 0, 0]), [-2, 2]);
  // A ray that misses the box on the diagonal.
  assert.equal(volRayInterval(f, [-10, 10, 0], [1, 0, 0]), null);
});

test("volRayInterval ellipsoid/cylinder: matches inside() along the ray", () => {
  const cases = [
    { radius: 1, ext: [2, 1, 3], kind: VOL_ELLIPSOID },
    { radius: 1, ext: [2, 1, 3], kind: VOL_CYLINDER },
    { radius: 1, ext: [2, 1, 3], kind: VOL_BOX },
  ];
  // Sample directions and origins; the marched inside-set must equal [t0,t1].
  const dirs = Array.from({ length: 24 }, (_, k) => fibonacciDir(k, 24));
  for (const f of cases) {
    for (const d of dirs) {
      for (const off of [
        [0, 0, 0],
        [0.7, -0.3, 1.1],
        [-1.5, 0.6, -2.0],
      ]) {
        const o = [off[0] - d[0] * 8, off[1] - d[1] * 8, off[2] - d[2] * 8];
        const span = volRayInterval(f, o, d);
        // Brute-force the true inside-set along the ray.
        let lo = Infinity,
          hi = -Infinity;
        for (let i = 0; i <= 4000; i++) {
          const t = (i / 4000) * 16;
          if (volInside(f, o[0] + d[0] * t, o[1] + d[1] * t, o[2] + d[2] * t)) {
            lo = Math.min(lo, t);
            hi = Math.max(hi, t);
          }
        }
        if (lo === Infinity) {
          // No sample was inside: either a true miss, or a sliver the sampling
          // stepped over — an interval, if reported, must be sliver-thin.
          if (span) assert.ok(span[1] - span[0] < 0.01, `sliver ${span}`);
          continue;
        }
        assert.ok(span, `kind ${f.kind}: inside at t=${lo} but no interval`);
        // The analytic interval must contain every sampled inside point, and
        // must not overreach it by more than one sample step.
        assert.ok(span[0] <= lo + 1e-9, `t0 ${span[0]} > first inside ${lo}`);
        assert.ok(span[1] >= hi - 1e-9, `t1 ${span[1]} < last inside ${hi}`);
        assert.ok(span[0] > lo - 0.01, `t0 ${span[0]} far below ${lo}`);
        assert.ok(span[1] < hi + 0.01, `t1 ${span[1]} far above ${hi}`);
      }
    }
  }
});

// ── Orientation ─────────────────────────────────────────────────────────────

test("volBasis: absent/degenerate rot is identity, not a zero volume", () => {
  assert.equal(volBasis({ radius: 1 }), null);
  assert.equal(volBasis({ radius: 1, rot: [1, 2, 3] }), null); // wrong length
  assert.equal(volBasis({ radius: 1, rot: [0, 0, 0, 0, 1, 0] }), null); // zero row
  assert.equal(volBasis({ radius: 1, rot: [1, 0, 0, 2, 0, 0] }), null); // r1 ∥ r0
  // A skewed r1 is re-orthogonalised against r0 rather than shearing the volume.
  const b = volBasis({ radius: 1, rot: [1, 0, 0, 0.5, 1, 0] });
  assert.ok(Math.abs(b[0][0] * b[1][0] + b[0][1] * b[1][1] + b[0][2] * b[1][2]) < 1e-12);
  for (const r of b) assert.ok(Math.abs(Math.hypot(...r) - 1) < 1e-12);
});

test("an oriented cylinder is the same volume, turned (the UI's X/Y/Z presets)", () => {
  const ext = [1, 1, 3]; // radial 1, half-length 3 along the LOCAL z
  const zCyl = { radius: 3, ext, kind: VOL_CYLINDER }; // identity ⇒ axis = world Z
  const xCyl = { radius: 3, ext, kind: VOL_CYLINDER, rot: [0, 1, 0, 0, 0, 1] }; // local z → world X
  const yCyl = { radius: 3, ext, kind: VOL_CYLINDER, rot: [0, 0, 1, 1, 0, 0] }; // local z → world Y

  // Far along its own axis is inside; the same distance across it is not.
  assert.ok(volInside(zCyl, 0, 0, 2.5) && !volInside(zCyl, 2.5, 0, 0));
  assert.ok(volInside(xCyl, 2.5, 0, 0) && !volInside(xCyl, 0, 0, 2.5));
  assert.ok(volInside(yCyl, 0, 2.5, 0) && !volInside(yCyl, 2.5, 0, 0));

  // Support along a volume's own axis is its half-length; across it, the radius.
  assert.ok(Math.abs(volSupport(xCyl, [1, 0, 0]) - 3) < 1e-12);
  assert.ok(Math.abs(volSupport(xCyl, [0, 1, 0]) - 1) < 1e-12);
  assert.ok(Math.abs(volSupport(yCyl, [0, 1, 0]) - 3) < 1e-12);
});

test("rotation is a rigid motion: inside/support/rayInterval all follow it", () => {
  // A deliberately oblique basis — not an axis permutation.
  const rot = [0.6, 0.8, 0, -0.48, 0.36, 0.8];
  const plain = { radius: 3, ext: [2, 1, 3], kind: VOL_CYLINDER };
  const turned = { ...plain, rot };
  const B = volBasis(turned);
  // World point for a given LOCAL point q: p = Σ qᵢ·rᵢ.
  const toWorld = (q) => [0, 1, 2].map((k) => q[0] * B[0][k] + q[1] * B[1][k] + q[2] * B[2][k]);

  for (const q of [[0,0,0],[1.9,0,0],[2.1,0,0],[0,0.9,2.9],[0,0,3.1],[1,0.5,-2]]) {
    assert.equal(
      volInside(turned, ...toWorld(q)), volInside(plain, ...q),
      `inside disagreed at local ${q}`,
    );
  }
  // rayInterval must return the SAME t-interval for the rotated ray, since an
  // orthonormal basis leaves arc length (and so t) alone.
  for (const [o, d] of [
    [[-9, 0.3, 0.2], [1, 0, 0]],
    [[0.2, -9, 0.4], [0, 1, 0]],
    [[-5, -5, -5], [0.5774, 0.5774, 0.5774]],
  ]) {
    const a = volRayInterval(plain, o, d);
    const b = volRayInterval(turned, toWorld(o), toWorld(d));
    if (a === null) { assert.equal(b, null); continue; }
    assert.ok(b, `rotated ray missed where the plain one hit (${o})`);
    assert.ok(Math.abs(a[0] - b[0]) < 1e-9 && Math.abs(a[1] - b[1]) < 1e-9,
      `interval moved under rotation: ${a} vs ${b}`);
  }
});

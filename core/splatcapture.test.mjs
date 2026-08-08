import { test } from "node:test";
import assert from "node:assert/strict";
import {
  frameFormula,
  captureView,
  captureSplats,
  reducePoints,
  refineFrame,
  growFrameToSurface,
  growEpsToSurface,
  probeFrameHits,
  deScaleFor,
  fibonacciDir,
  makeHitReservoir,
  sampleHits,
  computeR0,
  capturedDiag,
  makeStreamingReduce,
  aoScale,
  snapPoints,
  densifySplats,
  viewFrame,
  cameraFrame,
  captureEps,
  objectEpsFloor,
  deConvergenceFloor,
  withCaptureEps,
  EPS_FLOOR_FACTOR,
  EPS_FLOOR_MIN_RAYS,
} from "./splatcapture.js";
import { volExt, volInside } from "./capturevolume.js";
import { splatMetrics, epsFor as metricsEpsFor } from "./splatmetrics.js";
import {
  makeDE,
  makePointAlbedo,
  makeIterMeasure,
  signalRange,
} from "./cpu.js";
import { PRESETS, BLANK } from "./oplist.js";
import { defaultColoring } from "./coloring.js";

// A leaf-under-ops scene object (objType 0), the shape used across scene tests.
const sceneObj = (shapeId, params, origin, combine = 0) => ({
  objType: 0,
  shapeId,
  shapeParams: params,
  ops: [],
  iters: 1,
  transform: { origin, uscale: 1, rot: [0, 0, 0] },
  combine,
  blendK: 0,
  color: [0.7, 0.5, 0.3],
});
const sceneWrap = (objects) => ({
  name: "t",
  ops: [],
  iters: 8,
  deOption: 2,
  camera: { yawDeg: 20, pitchDeg: 15, dist: 5, fovDeg: 42 },
  objects,
});

const clamp01 = (v) => Math.max(0, Math.min(1, v));

test("signal-parity: makePointAlbedo == the live signal→signalRange→palette chain", () => {
  const f = PRESETS.find((p) => p.name === "Mandelbulb");
  const iters = f.iters ?? 8;
  // mode 1 (Glow), autoLevels ON, palette OFF → the colA→colB lerp branch,
  // replicable here without the private paletteAlbedo.
  const col = {
    ...defaultColoring(),
    mode: 1,
    autoLevels: true,
    palette: { on: false },
  };
  const albedoAt = makePointAlbedo(f, col, iters);
  const { lo, span } = signalRange(f, col, iters);
  const trap = makeIterMeasure(f, "trap", iters);
  const A = col.colA,
    B = col.colB;
  for (const p of [
    [0.6, 0.2, 0.3],
    [-0.4, 0.5, -0.1],
    [0.05, -0.9, 0.2],
  ]) {
    const mixT = clamp01((Math.min(trap(...p) / 1.5, 1) - lo) / span); // §5.4 + mode-1 min clamp
    const want = [0, 1, 2].map((i) => A[i] + (B[i] - A[i]) * mixT);
    const got = albedoAt(p[0], p[1], p[2], 0); // nz unused for mode 1
    for (let i = 0; i < 3; i++)
      assert.ok(Math.abs(got[i] - want[i]) < 1e-9, `chan ${i} at ${p}`);
  }
});

test("signal-parity: mode 0 (surface, nz path) + autoLevels off = identity", () => {
  const f = PRESETS.find((p) => p.name === "Mandelbulb");
  const iters = f.iters ?? 8;
  const A = [0.86, 0.46, 0.18],
    B = [0.18, 0.62, 0.74];
  // mode 0 uses nz; not in AUTOLEVEL_MODES so signalRange is identity.
  const col = {
    ...defaultColoring(),
    mode: 0,
    colA: A,
    colB: B,
    palette: { on: false },
  };
  const albedoAt = makePointAlbedo(f, col, iters);
  for (const nz of [-1, -0.3, 0.5, 1]) {
    const mixT = 0.5 + 0.5 * nz;
    const want = [0, 1, 2].map((i) => A[i] + (B[i] - A[i]) * mixT);
    const got = albedoAt(0.1, 0.2, 0.3, nz);
    for (let i = 0; i < 3; i++) assert.ok(Math.abs(got[i] - want[i]) < 1e-9);
  }
});

// #432: an ExportJob's coloring is optional, and the CPU capture + S-2 refine
// workers pass it through verbatim — `makePointAlbedo(f, undefined, iters)`
// threw "Cannot read properties of undefined (reading 'palettePhase')" and
// failed the whole export. A missing look must degrade to the default gradient,
// not crash.
test("makePointAlbedo survives a missing coloring (#432)", () => {
  const f = PRESETS.find((p) => p.name === "Mandelbulb");
  const iters = f.iters ?? 8;
  for (const col of [undefined, null, {}]) {
    const albedoAt = makePointAlbedo(f, col, iters);
    const got = albedoAt(0.1, 0.2, 0.3, 0.5);
    assert.equal(got.length, 3, "returns an RGB triple");
    for (const c of got) assert.ok(Number.isFinite(c), `finite channel (${c})`);
  }
  // …and it degrades to the documented fallback gradient (mode 0, colA→colB
  // defaults), so a coloring-less export still looks like the default look.
  const A = [0.86, 0.46, 0.18],
    B = [0.18, 0.62, 0.74];
  const bare = makePointAlbedo(f, undefined, iters)(0.1, 0.2, 0.3, 0.5);
  const want = [0, 1, 2].map((i) => A[i] + (B[i] - A[i]) * 0.75); // mixT = 0.5+0.5·nz
  for (let i = 0; i < 3; i++) assert.ok(Math.abs(bare[i] - want[i]) < 1e-9);
});

test("captureView geometric truth: unit-sphere SDF → hits on |p|≈1, normal = p/|p|", () => {
  const de = (x, y, z) => Math.hypot(x, y, z) - 1; // unit sphere
  const albedoAt = () => [0.5, 0.5, 0.5];
  const frame = { center: [0, 0, 0], radius: 1.5 };
  const eps = Math.max(3e-4 * 1.5, 1e-5);
  const out = { pos: [], normal: [], albedo: [] };
  let total = 0;
  for (let k = 0; k < 12; k++)
    total += captureView(de, albedoAt, frame, fibonacciDir(k, 12), 32, out);
  assert.ok(total > 0, "some rays hit the sphere");
  const n = out.pos.length / 3;
  for (let i = 0; i < n; i++) {
    const j = 3 * i;
    const r = Math.hypot(out.pos[j], out.pos[j + 1], out.pos[j + 2]);
    assert.ok(Math.abs(r - 1) < 5 * eps, `hit radius ${r}`);
    // outward normal ≈ p/|p|
    const dot =
      (out.pos[j] * out.normal[j] +
        out.pos[j + 1] * out.normal[j + 1] +
        out.pos[j + 2] * out.normal[j + 2]) /
      r;
    assert.ok(dot > 0.99, `normal aligns with radial (dot ${dot})`);
  }
});

test("frameFormula: a real flat preset frames to a finite positive radius", () => {
  const f = PRESETS.find((p) => p.name === "Mandelbulb");
  const fr = frameFormula(f);
  assert.ok(fr, "found a surface");
  assert.ok(fr.radius > 0 && Number.isFinite(fr.radius));
  assert.ok(fr.center.every(Number.isFinite));
  assert.ok(fr.diag > 0);
});

// A synthetic raw cloud of N distinct hits (value i encoded in every channel).
function fakeRaw(n) {
  const pos = new Float32Array(3 * n),
    normal = new Float32Array(3 * n),
    albedo = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    pos[3 * i] = pos[3 * i + 1] = pos[3 * i + 2] = i;
    normal[3 * i] = normal[3 * i + 1] = normal[3 * i + 2] = i;
    albedo[3 * i] = albedo[3 * i + 1] = albedo[3 * i + 2] = i;
  }
  return { count: n, pos, normal, albedo };
}

test("makeHitReservoir/sampleHits: bounded, deterministic, cap-respecting (P0 harness)", () => {
  // cap ≥ hits ⇒ keep all, order preserved
  const all = sampleHits(fakeRaw(100), 1000, 7);
  assert.equal(all.count, 100);
  assert.equal(all.pos[3 * 42], 42, "kept all, in order");
  // cap < hits ⇒ exactly cap survivors, all triplets internally consistent
  const sub = sampleHits(fakeRaw(10_000), 500, 7);
  assert.equal(sub.count, 500);
  for (let i = 0; i < sub.count; i++) {
    const v = sub.pos[3 * i];
    assert.ok(v >= 0 && v < 10_000, "survivor is a real hit index");
    assert.equal(
      sub.normal[3 * i + 2],
      v,
      "triplet stays aligned (pos/normal/albedo)",
    );
    assert.equal(sub.albedo[3 * i + 1], v);
  }
  // determinism: same seed + input ⇒ identical bytes; different seed ⇒ differs
  const a = sampleHits(fakeRaw(10_000), 500, 7);
  const b = sampleHits(fakeRaw(10_000), 500, 7);
  assert.deepEqual([...a.pos], [...sub.pos], "same seed reproduces");
  assert.deepEqual([...b.pos], [...a.pos]);
  const c = sampleHits(fakeRaw(10_000), 500, 99);
  assert.notDeepEqual(
    [...c.pos],
    [...a.pos],
    "different seed → different sample",
  );
  // streaming addChunk over K chunks == one-shot over the concatenation isn't
  // required (different draw order), but seen must total the offered hits
  const r = makeHitReservoir(500, 7);
  r.addChunk(fakeRaw(300).pos, fakeRaw(300).normal, fakeRaw(300).albedo);
  r.addChunk(fakeRaw(400).pos, fakeRaw(400).normal, fakeRaw(400).albedo);
  assert.equal(r.seen, 700);
  assert.equal(r.result().count, 500, "filled capped at cap");
});

test("captureSplats sampleHits: default 0 = shape unchanged; >0 attaches sample + stats.iters", () => {
  const f = PRESETS.find((p) => p.name === "Mandelbulb");
  const base = captureSplats(f, defaultColoring(), { views: 4, res: 16 });
  assert.equal(base.sample, undefined, "no sample by default");
  assert.equal(base.stats.iters, undefined, "stats shape unchanged by default");
  const withS = captureSplats(f, defaultColoring(), {
    views: 4,
    res: 16,
    sampleHits: 200,
  });
  assert.ok(withS.sample && withS.sample.count > 0, "sample attached");
  assert.ok(withS.sample.count <= 200, "sample capped");
  assert.ok(withS.sample.count <= withS.stats.rawHits, "sample ≤ raw hits");
  assert.equal(
    typeof withS.stats.iters,
    "number",
    "stats.iters present when sampling",
  );
});

test("reducePoints P2 aniso: back-compat + cylinder dir along axis + flat/sparse isotropic", () => {
  const seeded = (s0) => {
    let s = s0 >>> 0 || 1;
    return () => (
      (s = (Math.imul(s, 1664525) + 1013904223) >>> 0),
      s / 4294967296
    );
  };
  // --- back-compat: aniso:0 deep-equals no-opts (byte-identical, no r2/dir/anisoStats) ---
  const rnd = seeded(7);
  const n = 3000;
  const pos = new Float32Array(3 * n),
    nr = new Float32Array(3 * n),
    al = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    const j = 3 * i;
    pos[j] = rnd() * 2 - 1;
    pos[j + 1] = rnd() * 2 - 1;
    pos[j + 2] = rnd() * 2 - 1;
    let x = rnd() - 0.5,
      y = rnd() - 0.5,
      z = rnd() - 0.5;
    const l = Math.hypot(x, y, z) || 1;
    nr[j] = x / l;
    nr[j + 1] = y / l;
    nr[j + 2] = z / l;
    al[j] = rnd();
    al[j + 1] = rnd();
    al[j + 2] = rnd();
  }
  const raw = { count: n, pos, normal: nr, albedo: al };
  const base = reducePoints(raw, 0.05, 5000, {});
  const off = reducePoints(raw, 0.05, 5000, { aniso: 0 });
  assert.deepEqual(
    [...off.points.radius],
    [...base.points.radius],
    "aniso:0 radius identical",
  );
  assert.deepEqual(
    [...off.points.pos],
    [...base.points.pos],
    "aniso:0 pos identical",
  );
  assert.equal(off.points.r2, undefined, "aniso:0 emits no r2");
  assert.equal(off.points.dir, undefined, "aniso:0 emits no dir");
  assert.equal(off.anisoStats, undefined, "aniso:0 emits no anisoStats");

  // --- cylinder wall (axis = z): major axis (low curvature) → ±z ---
  const R = 2,
    cp = [],
    cn = [],
    ca = [];
  for (let t = 0; t < 300; t++)
    for (let zi = 0; zi < 160; zi++) {
      const th = (t / 300) * 2 * Math.PI,
        z = -2 + (zi / 160) * 4;
      cp.push(R * Math.cos(th), R * Math.sin(th), z);
      cn.push(Math.cos(th), Math.sin(th), 0);
      ca.push(0.5, 0.5, 0.5);
    }
  const cyl = {
    count: cp.length / 3,
    pos: Float32Array.from(cp),
    normal: Float32Array.from(cn),
    albedo: Float32Array.from(ca),
  };
  const cres = reducePoints(
    cyl,
    computeR0(2 * Math.hypot(R, R, 2), cyl.count, 1.6),
    40000,
    {
      aniso: 1,
    },
  );
  assert.ok(cres.points.dir, "cylinder produced dir");
  assert.ok(
    cres.anisoStats.fitted > 100,
    "most cylinder survivors fit anisotropically",
  );
  let dotZ = 0,
    cnt = 0;
  for (let i = 0; i < cres.points.count; i++) {
    const d = cres.points.dir;
    if (
      Math.abs(cres.points.pos[3 * i + 2]) < 0.3 &&
      (d[3 * i] || d[3 * i + 1] || d[3 * i + 2])
    ) {
      dotZ += Math.abs(d[3 * i + 2]);
      cnt++;
    }
  }
  assert.ok(
    cnt > 50 && dotZ / cnt > 0.95,
    `major axis ≈ cylinder axis (mean |dir.z| ${(dotZ / cnt).toFixed(3)})`,
  );

  // --- flat sheet: both curvatures ≈ 0 ⇒ noise floor ⇒ isotropic (all dir 0) ---
  const fp = [],
    fn = [],
    fa = [];
  for (let x = 0; x < 60; x++)
    for (let y = 0; y < 60; y++) {
      fp.push((x / 60) * 2 - 1, (y / 60) * 2 - 1, 0);
      fn.push(0, 0, 1);
      fa.push(0.5, 0.5, 0.5);
    }
  const flat = {
    count: fp.length / 3,
    pos: Float32Array.from(fp),
    normal: Float32Array.from(fn),
    albedo: Float32Array.from(fa),
  };
  const fres = reducePoints(flat, computeR0(4, flat.count, 1.6), 20000, {
    aniso: 1,
  });
  let anyDir = 0;
  for (let i = 0; i < fres.points.count; i++)
    if (
      fres.points.dir[3 * i] ||
      fres.points.dir[3 * i + 1] ||
      fres.points.dir[3 * i + 2]
    )
      anyDir++;
  assert.equal(
    anyDir,
    0,
    "flat sheet ⇒ every survivor isotropic (noise floor)",
  );

  // --- sparse: <5 neighbors ⇒ passthrough (dir 0, r2 = radius) ---
  const sparse = {
    count: 3,
    pos: Float32Array.of(0, 0, 0, 5, 0, 0, 0, 5, 0),
    normal: Float32Array.of(0, 0, 1, 0, 0, 1, 0, 0, 1),
    albedo: Float32Array.of(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
  };
  const sres = reducePoints(sparse, 0.1, 100, { aniso: 1 });
  for (let i = 0; i < sres.points.count; i++) {
    assert.equal(sres.points.dir[3 * i], 0, "sparse ⇒ dir 0");
    assert.equal(
      sres.points.r2[i],
      sres.points.radius[i],
      "sparse ⇒ r2 = radius (isotropic)",
    );
  }
});

test("captureSplats: empty formula → null (§5.3a guard, never a degenerate file)", () => {
  const r = captureSplats(BLANK, defaultColoring(), { views: 4, res: 8 });
  assert.equal(r, null);
});

// ── S1a: scene + hybrid capture (replaces the S0 flat-only reject test) ──────

test("captureSplats scene: two-sphere union → hits on both surfaces, both sampled", () => {
  const scene = sceneWrap([
    sceneObj(2, [0.6, 0, 0, 0], [-1, 0, 0]),
    sceneObj(2, [0.6, 0, 0, 0], [1.2, 0, 0]),
  ]);
  const r = captureSplats(scene, defaultColoring(), { views: 24, res: 48 });
  assert.ok(r, "scene captured (guard removed)");
  // frame stayed bounded (refineFrame worked, no R=40 balloon)
  assert.ok(
    r.frame.radius > 0 && r.frame.radius < 6,
    `frame radius ${r.frame.radius}`,
  );
  const de = makeDE(scene, 8);
  const eps = Math.max(3e-4 * r.frame.radius, 1e-5);
  let nearA = 0,
    nearB = 0;
  for (let i = 0; i < r.points.count; i++) {
    const j = 3 * i;
    const p = [r.points.pos[j], r.points.pos[j + 1], r.points.pos[j + 2]];
    assert.ok(
      Math.abs(de(...p)) < 8 * eps,
      `hit on surface, |de|=${Math.abs(de(...p))}`,
    );
    if (p[0] < 0) nearA++;
    else nearB++;
  }
  assert.ok(
    nearA > 0 && nearB > 0,
    `both spheres sampled (A ${nearA}, B ${nearB})`,
  );
});

test("captureSplats scene carve: box−sphere → no hit deep in the void, cut face present", () => {
  const scene = sceneWrap([
    sceneObj(1, [0.9, 0.9, 0.9, 0], [0, 0, 0]), // box
    sceneObj(2, [0.7, 0, 0, 0], [0.9, 0, 0], 2), // − sphere (combine 2 = subtract)
  ]);
  assert.equal(
    deScaleFor(scene),
    0.25,
    "carving scene → tighter 0.25 march step",
  );
  const r = captureSplats(scene, defaultColoring(), { views: 24, res: 48 });
  assert.ok(r, "carve scene captured");
  const de = makeDE(scene, 8);
  let deepInVoid = 0,
    onCutFace = 0;
  for (let i = 0; i < r.points.count; i++) {
    const j = 3 * i;
    const p = [r.points.pos[j], r.points.pos[j + 1], r.points.pos[j + 2]];
    const dSph = Math.hypot(p[0] - 0.9, p[1], p[2]);
    if (dSph < 0.6) deepInVoid++; // well inside the carved-out sphere
    if (dSph > 0.65 && dSph < 0.75 && Math.abs(de(...p)) < 0.02) onCutFace++;
  }
  assert.equal(
    deepInVoid,
    0,
    "the deScale step never pierces into the carved void",
  );
  assert.ok(onCutFace > 0, "the concave cut face is captured");
});

test("captureSplats hybrid: a hybrid preset captures (guard removed)", () => {
  const h = PRESETS.find((p) => p.name === "Bulb Hybrid");
  assert.ok(h && h.hybrid, "found a hybrid preset");
  const fr = frameFormula(h);
  assert.ok(fr && Number.isFinite(fr.radius), "hybrid frames finitely");
  const r = captureSplats(h, defaultColoring(), { views: 16, res: 40 });
  assert.ok(r && r.stats.rawHits > 0, "hybrid produced hits");
});

test("scene albedo through makePointAlbedo: mode 0 = winning object's color (GPU sceneTint parity)", () => {
  // Scene surface mode must return the winning object's OWN color — matching the
  // GPU sceneTint (shader.js:698) — NOT the normal-tint palette. (Regression:
  // scene splat exports were monochromatic because per-object color was dropped.)
  const objA = {
    ...sceneObj(2, [0.6, 0, 0, 0], [-1, 0, 0]),
    color: [0.9, 0.52, 0.2],
  }; // orange
  const objB = {
    ...sceneObj(2, [0.6, 0, 0, 0], [1.2, 0, 0]),
    color: [0.3, 0.55, 0.85],
  }; // blue
  const scene = sceneWrap([objA, objB]);
  // palette/nz must NOT influence the result — surface color is the object color.
  const c0 = {
    ...defaultColoring(),
    mode: 0,
    colA: [0.9, 0.3, 0.2],
    colB: [0.1, 0.5, 0.8],
    palette: { on: false },
  };
  const alb0 = makePointAlbedo(scene, c0, 8);
  for (const nz of [-0.7, 0.3, 1]) {
    const nearA = alb0(-1, 0, 0, nz); // on object A → orange, any nz
    const nearB = alb0(1.2, 0, 0, nz); // on object B → blue, any nz
    for (let i = 0; i < 3; i++) {
      assert.ok(
        Math.abs(nearA[i] - objA.color[i]) < 1e-9,
        `objA color chan ${i} (nz ${nz})`,
      );
      assert.ok(
        Math.abs(nearB[i] - objB.color[i]) < 1e-9,
        `objB color chan ${i} (nz ${nz})`,
      );
    }
  }
  // mode 1 (Glow) on a scene → finite albedo in [0,1] (a trivial leaf orbit is
  // flat → colA everywhere; the point is that the scene signal path runs and
  // yields valid colors — richer scene coloring is covered by scenecolor.test).
  const alb1 = makePointAlbedo(scene, { ...defaultColoring(), mode: 1 }, 8);
  for (const p of [
    [0.6, 0.1, 0.2, 0.3],
    [-0.3, 0.5, -0.4, -0.2],
  ]) {
    const g = alb1(...p);
    for (const v of g)
      assert.ok(Number.isFinite(v) && v >= 0 && v <= 1, `glow in range: ${v}`);
  }
});

test("refineFrame: offset sphere → converges on true bbox; empty → camFrame fallback", () => {
  const off = [2, 1, -1];
  const sph = (x, y, z) => Math.hypot(x - off[0], y - off[1], z - off[2]) - 1;
  const camFrame = {
    center: [0, 0, 0],
    ext: [5, 5, 5],
    radius: 5,
    diag: 2 * Math.hypot(5, 5, 5),
  };
  const oversized = { center: [0, 0, 0], radius: 8 }; // off-center + too big
  const fr = refineFrame(sph, oversized, camFrame, { grid: 48 });
  for (let c = 0; c < 3; c++)
    assert.ok(Math.abs(fr.center[c] - off[c]) < 0.15, `center[${c}]`);
  for (let c = 0; c < 3; c++)
    assert.ok(Math.abs(fr.ext[c] - 1) < 0.2, `ext[${c}] ≈ 1`);
  // no rays hit ⇒ fall back to the human-vetted saved camera, not the input
  const fb = refineFrame(() => 1e9, oversized, camFrame, { grid: 16 });
  assert.equal(fb, camFrame, "empty capture → camFrame fallback");
});

test("refineFrame balloon-cap: a ballooned scene frame is bounded by camFrame, not the balloon", () => {
  // The New Problem #1 path: an infinite plane makes the provisional frame
  // balloon (non-null). refineFrame must cap the probe basis by camFrame so the
  // result stays near the saved view — NOT scale with the balloon.
  const plane = (x, y, z) => y + 1; // plane at y = −1
  const camFrame = {
    center: [0, 0, 0],
    ext: [3, 3, 3],
    radius: 3,
    diag: 2 * Math.hypot(3, 3, 3),
  };
  const ballooned = { center: [0, 0, 0], radius: 44 }; // R ran to the 40 cap
  const fr = refineFrame(plane, ballooned, camFrame, { grid: 32 });
  assert.ok(
    fr.radius < 10,
    `capped near saved view (${fr.radius}), not the 44 balloon`,
  );
  // a well-bounded provisional frame is used as its own basis (identical here)
  const fr2 = refineFrame(plane, { center: [0, 0, 0], radius: 3 }, camFrame, {
    grid: 32,
  });
  assert.ok(
    Math.abs(fr.radius - fr2.radius) < 1e-6,
    "cap makes ballooned ≡ bounded basis",
  );
});

// ── #351 "No surface found in the current framing" ──────────────────────────
// captureView's hit eps is `3e-4 · frame.radius` — it shrinks with a smaller
// frame. A severely loose analytic DE (measured on issue #351's "Drifting
// Sponge Octacale": ScaleDrift decay stacked with kaleido/octaFold) never
// converges much below a fixed absolute floor near the true surface, so a
// small, tightly-fitted frame sets eps BELOW that floor and captureView finds
// zero hits everywhere — even though the same geometry captures fine once the
// frame is large enough to loosen eps past the floor. This synthetic DE
// reproduces that shape deterministically: a sphere whose DE never reports
// less than FLOOR near its true surface, so only `3e-4 · radius ≥ FLOOR` (i.e.
// radius ≥ ~6.67 here) can ever register a hit.
function loosySphereDE(center, r0, floor) {
  return (x, y, z) => {
    const d = Math.hypot(x - center[0], y - center[1], z - center[2]) - r0;
    if (Math.abs(d) < floor) return d >= 0 ? floor : -floor;
    return d;
  };
}

test("growFrameToSurface #351: a too-small frame with a loose-DE floor grows until it clears eps, and does NOT re-tighten below the floor", () => {
  const FLOOR = 0.002; // needs radius ≥ FLOOR/3e-4 ≈ 6.67 for eps to clear it
  const de = loosySphereDE([0, 0, 0], 5, FLOOR);
  const tooSmall = { center: [0, 0, 0], radius: 1 }; // eps = 3e-4 → below FLOOR
  const grown = growFrameToSurface(de, tooSmall, { deScale: 1 });
  assert.ok(
    grown.radius > tooSmall.radius,
    `frame grew past the input (${grown.radius})`,
  );
  assert.ok(
    3e-4 * grown.radius >= FLOOR,
    `grown radius clears the eps floor (radius=${grown.radius}, eps=${3e-4 * grown.radius})`,
  );
  // The grown frame must actually capture real surface at production-like
  // multi-view resolution — the regression this test pins is a frame that
  // LOOKS successful during the cheap probe but gets re-tightened back below
  // the floor and fails again on the real capture.
  const out = { pos: [], normal: [], albedo: [] };
  let hits = 0;
  const views = 32,
    res = 32;
  for (let k = 0; k < views; k++)
    hits += captureView(
      de,
      () => [0, 0, 0],
      grown,
      fibonacciDir(k, views),
      res,
      out,
      { maxSteps: 200, deScale: 1 },
    );
  assert.ok(hits > 0, `grown frame captures real surface (${hits} hits)`);
});

test("growFrameToSurface #351: an already-correct frame is returned unchanged (no cost for the common case)", () => {
  const de = (x, y, z) => Math.hypot(x, y, z) - 1; // ordinary tight sphere DE
  const good = { center: [0, 0, 0], radius: 3 };
  const grown = growFrameToSurface(de, good, { deScale: 1 });
  assert.equal(grown, good, "first probe already succeeds → same reference");
});

// ── #518 "Very slim number of exported points" ──────────────────────────────
// The export's splat count falls with the FOURTH power of the capture volume's
// linear size: the per-view window is the volume's shadow (so hits ∝ 1/ext²)
// and the dedup pitch is radiusScale·diag/√hits (so cell ∝ ext²), and
// survivors ≈ area/cell². Two places let empty space into that volume, and both
// are fixed here: growFrameToSurface used to inflate `ext` purely to buy a
// bigger eps, and the pitch was measured over the FRAME rather than over what
// was captured in it.
test("growFrameToSurface #518: a surface INSIDE the volume is recovered by eps alone — the captured volume is not inflated", () => {
  const FLOOR = 0.002; // needs radius ≥ FLOOR/3e-4 ≈ 6.67 for eps to clear it
  const de = loosySphereDE([0, 0, 0], 5, FLOOR);
  // A frame that genuinely CONTAINS the sphere (ext 6 > r 5) but whose eps
  // (3e-4·6 = 1.8e-3) sits below the DE's floor — the #351 symptom without the
  // #351 excuse: nothing needs to be sampled that isn't already being sampled.
  const frame = {
    center: [0, 0, 0],
    ext: [6, 6, 6],
    radius: 6,
    diag: 2 * Math.hypot(6, 6, 6),
  };
  const grown = growFrameToSurface(de, frame, { deScale: 1 });
  assert.ok(
    3e-4 * grown.radius >= FLOOR,
    `eps cleared the loose-DE floor (radius ${grown.radius}, eps ${3e-4 * grown.radius})`,
  );
  assert.deepEqual(
    grown.ext,
    [6, 6, 6],
    "#518: the captured VOLUME is untouched — only the eps scalar grew",
  );
  assert.deepEqual(grown.center, [0, 0, 0], "#518: the centre is untouched");
  assert.equal(
    grown.diag,
    frame.diag,
    "#518: diag follows ext, so the reduce pitch is not coarsened either",
  );
});

test("growFrameToSurface #351/#518: a surface OUTSIDE the volume still grows the volume — eps can't find what isn't sampled", () => {
  const FLOOR = 0.002;
  const de = loosySphereDE([0, 0, 0], 5, FLOOR);
  // The #351 shape: ext 1 around a sphere of radius 5. No epsilon reaches it —
  // the volume itself has to grow, and that fallback must survive #518.
  const tooSmall = { center: [0, 0, 0], radius: 1 };
  const grown = growFrameToSurface(de, tooSmall, { deScale: 1 });
  assert.ok(
    volExt(grown)[0] > 1,
    `the volume grew to reach the surface (ext ${volExt(grown)[0]})`,
  );
  assert.ok(grown.radius > tooSmall.radius, "and the eps scalar grew with it");
});

test("capturedDiag #518: pitch measures the captured geometry, and degrades to the frame", () => {
  const frameDiag = 100;
  assert.equal(
    capturedDiag([-1, -1, -1], [1, 1, 1], frameDiag),
    Math.hypot(2, 2, 2),
    "a hit cloud smaller than the frame sets the pitch",
  );
  assert.equal(
    capturedDiag([-99, -99, -99], [99, 99, 99], frameDiag),
    frameDiag,
    "never wider than the frame (hits are clipped to the volume anyway)",
  );
  assert.equal(
    capturedDiag(
      [Infinity, Infinity, Infinity],
      [-Infinity, -Infinity, -Infinity],
      frameDiag,
    ),
    frameDiag,
    "no hits yet ⇒ the frame's own diagonal, never NaN",
  );
  assert.equal(
    capturedDiag([1, 2, 3], [1, 2, 3], frameDiag),
    frameDiag,
    "a single coincident hit ⇒ the frame, never a zero pitch",
  );
});

test("makeStreamingReduce #518: an oversized capture volume does not coarsen the pitch — same cloud, same survivors", () => {
  // One synthetic surface (a 20×20 grid on a unit-ish patch), reduced inside two
  // frames that differ ONLY in how much empty space they enclose. The captured
  // set is identical, so the exported cloud must be too.
  const n = 40;
  const pos = [],
    normal = [],
    albedo = [];
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      pos.push(-0.5 + i / (n - 1), -0.5 + j / (n - 1), 0);
      normal.push(0, 0, 1);
      albedo.push(0.5, 0.5, 0.5);
    }
  const reduceIn = (ext) => {
    const frame = {
      center: [0, 0, 0],
      ext: [ext, ext, ext],
      radius: ext,
      diag: 2 * Math.hypot(ext, ext, ext),
    };
    const r = makeStreamingReduce({ frame, cap: 1_000_000, views: 1 });
    r.addChunk(pos, normal, albedo, 1);
    return r.finalize();
  };
  const tight = reduceIn(0.5); // the volume the geometry fills
  const loose = reduceIn(4); // the same geometry, 8× the volume around it
  // 324 cells over the patch, both ways. Before #518 the loose frame's pitch
  // came from its own diagonal (8× the tight one's) and the identical cloud
  // reduced to 4 splats.
  assert.ok(
    tight.points.count > 300,
    `tight frame resolves the patch (${tight.points.count})`,
  );
  assert.equal(
    loose.points.count,
    tight.points.count,
    "#518: survivors are a property of the captured geometry, not of the framing",
  );
  assert.ok(
    Math.abs(loose.cell / tight.cell - 1) < 1e-9,
    `#518: identical pitch (${tight.cell} vs ${loose.cell})`,
  );
});

test("captureSplats #518: the #351 formula exports a resolved cloud, not a handful of splats", () => {
  // The exact #351 reporter formula, at the same cost as the #351 pin above.
  // Before #518 this whole-object capture grew its frame 4× to loosen eps and
  // came back with 64 survivors (batch) / 132 (streaming) — a "very slim number
  // of exported points" against a 50,000 cap. Same geometry, same eps, same
  // rays: the volume just stopped being inflated around it.
  const formula = {
    name: "Drifting Sponge Octacale",
    iters: 7,
    deOption: 2,
    ops: [
      { key: "menger", values: [-0.05] },
      { key: "scaleDrift", values: [2, 0.08] },
      { key: "translate", values: [-1.12, -1.48, -0.6] },
      { key: "kaleido", values: [5, 45.5, 1] },
      { key: "octaFold", values: [] },
      { key: "scaleDrift", values: [1.03, -0.33] },
      { key: "translate", values: [0.04, -0.06, -0.09] },
    ],
    camera: {
      yawDeg: 136.1,
      pitchDeg: 37,
      dist: 5.288056566997441,
      target: [0.05892588109143022, -0.09794680224278356, -0.14888802889158695],
    },
  };
  const r = captureSplats(formula, defaultColoring(), {
    views: 16,
    res: 32,
    cap: 50_000,
    stream: true,
  });
  assert.ok(r, "#518: the capture still succeeds (#351 stays fixed)");
  // Measured 10244 after the fix, 132 before — 5000 is a floor no amount of
  // run-to-run drift reaches from either side.
  assert.ok(
    r.points.count > 5000,
    `#518: the cloud is resolved, not starved (${r.points.count} splats)`,
  );
  assert.deepEqual(
    r.frame.ext.map((e) => +e.toFixed(6)),
    frameFormula(formula).ext.map((e) => +e.toFixed(6)),
    "#518: the captured volume is still the one the framing probe measured",
  );
});

// #496: a custom (user-placed) capture volume much smaller than the object
// hit the SAME #351 floor — eps = 3e-4·radius sat below the DE's fixed
// convergence floor — but growFrameToSurface's fix (growing `ext` too) would
// silently resize the box the user drew, which kit/splatexport.ts's
// exportFrame explicitly refuses to do ("honoured verbatim"). growEpsToSurface
// grows ONLY radius, so ext/center — the actual captured geometry — never
// move.
test("growEpsToSurface #496: grows radius past a loose-DE floor WITHOUT touching ext or center", () => {
  const FLOOR = 0.002; // needs radius ≥ FLOOR/3e-4 ≈ 6.67 for eps to clear it
  const de = loosySphereDE([0, 0, 0], 5, FLOOR);
  // A tiny custom box wrapped around a small patch of the sphere's surface —
  // ext/center are what the gizmo drew and must survive unchanged.
  const tinyBox = {
    center: [5, 0, 0],
    ext: [0.5, 0.5, 0.5],
    radius: 0.5,
    diag: 2 * Math.hypot(0.5, 0.5, 0.5),
  };
  const grown = growEpsToSurface(de, tinyBox, { deScale: 1 });
  assert.ok(
    grown.radius > tinyBox.radius,
    `radius grew past the input (${grown.radius})`,
  );
  assert.ok(
    3e-4 * grown.radius >= FLOOR,
    `grown radius clears the eps floor (radius=${grown.radius}, eps=${3e-4 * grown.radius})`,
  );
  assert.deepEqual(
    grown.ext,
    tinyBox.ext,
    "ext — the drawn box — is untouched",
  );
  assert.deepEqual(
    grown.center,
    tinyBox.center,
    "center — the drawn box's position — is untouched",
  );
  // Must actually capture real surface at production-like resolution, not just
  // clear the probe's own (looser) threshold.
  const out = { pos: [], normal: [], albedo: [] };
  let hits = 0;
  const views = 32,
    res = 32;
  for (let k = 0; k < views; k++)
    hits += captureView(
      de,
      () => [0, 0, 0],
      grown,
      fibonacciDir(k, views),
      res,
      out,
      {
        maxSteps: 200,
        deScale: 1,
      },
    );
  assert.ok(hits > 0, `grown frame captures real surface (${hits} hits)`);
});

test("growEpsToSurface #496: an already-correct radius is returned unchanged (no cost for the common case)", () => {
  const de = (x, y, z) => Math.hypot(x, y, z) - 1; // ordinary tight sphere DE
  const good = { center: [0, 0, 0], ext: [3, 3, 3], radius: 3 };
  const grown = growEpsToSurface(de, good, { deScale: 1 });
  assert.equal(grown, good, "first probe already succeeds → same reference");
});

// #496 follow-up: the reporter's continued "merging way too many points" (and
// possibly-reversed normals) after the initial starvation fix (above) traced to
// `minHits` being an ABSOLUTE count. kit/splatexport.ts's real capture probes
// at the job's OWN (much denser) views/res, not the cheap 16×32² default — and
// against a denser probe, minHits=16 is a far lower RELATIVE bar, so growth
// stopped as soon as it cleared a near-meaningless threshold: an eps still too
// tight for the real capture to find much of anything (checked directly: only
// a couple dozen raw hits from a multi-megapixel capture). minHits must scale
// with the probe's own ray budget (probeViews·probeRes²) to mean the same
// thing at any density — this is the sphere-in-large-frame case where hit
// count is a genuine, non-degenerate fraction of the probe's rays (unlike a
// frame that tightly bounds the whole DE, where crossing the eps floor jumps
// straight to near-total coverage and every threshold converges at the same
// step regardless of scaling).
test("growEpsToSurface #496: minHits scales with probe density — a caller probing at its real (dense) capture rate doesn't converge on a too-tight eps", () => {
  const FLOOR = 0.002;
  const de = loosySphereDE([0.3, 0, 0], 0.05, FLOOR); // small sphere well off-center
  const frame = { center: [0, 0, 0], ext: [3, 3, 3], radius: 3 };
  const probeViews = 32,
    probeRes = 64; // 8× the default (16×32²) probe's ray budget

  // Pre-fix behavior, reproduced by pinning the OLD absolute minHits=16: it
  // converges early, well before the real capture below can find much.
  const oldStyle = growEpsToSurface(de, frame, {
    probeViews,
    probeRes,
    minHits: 16,
  });
  // The fix: default minHits scales with probeViews/probeRes automatically.
  const fixed = growEpsToSurface(de, frame, { probeViews, probeRes });

  assert.ok(
    fixed.radius > oldStyle.radius,
    `the rate-scaled bar demands more growth than the old absolute count (old=${oldStyle.radius}, fixed=${fixed.radius})`,
  );
  assert.deepEqual(
    fixed.ext,
    frame.ext,
    "ext — the drawn box — stays untouched",
  );
  assert.deepEqual(
    fixed.center,
    frame.center,
    "center — the drawn box's position — stays untouched",
  );

  // The real, much denser capture the eps is actually for: the rate-scaled
  // frame must resolve substantially more of the surface than the
  // old-absolute-count frame did — never regressing on the very problem this
  // fix targets.
  const realOld = probeFrameHits(de, oldStyle, {
    probeViews: 128,
    probeRes: 256,
  });
  const realFixed = probeFrameHits(de, fixed, {
    probeViews: 128,
    probeRes: 256,
  });
  assert.ok(
    realFixed > realOld * 2,
    `fixed eps resolves markedly more of the real capture (old=${realOld}, fixed=${realFixed})`,
  );
});

// A caller that never overrides probe density (every existing call site except
// kit/splatexport.ts's custom-box path) must see BYTE-IDENTICAL behavior — the
// rate is calibrated so probeViews=16,probeRes=32 (the defaults) reproduce the
// original absolute minHits=16 exactly.
test("growEpsToSurface #496: default probe density is unaffected by the minHits rate-scaling (back-compat)", () => {
  const FLOOR = 0.002;
  const de = loosySphereDE([0, 0, 0], 5, FLOOR);
  const tinyBox = {
    center: [5, 0, 0],
    ext: [0.5, 0.5, 0.5],
    radius: 0.5,
    diag: 2 * Math.hypot(0.5, 0.5, 0.5),
  };
  const withDefaults = growEpsToSurface(de, tinyBox, { deScale: 1 });
  const withExplicitOldMinHits = growEpsToSurface(de, tinyBox, {
    deScale: 1,
    minHits: 16,
  });
  assert.equal(
    withDefaults.radius,
    withExplicitOldMinHits.radius,
    "unscaled default probe density resolves to the same growth as the original absolute minHits=16",
  );
});

// ── #496 round 4: the eps floor is a property of the FIELD, not the crop ────
// The three fixes above all worked inside the "eps = 3e-4 · radius" rule, so a
// small box could only ever buy a looser eps by GROWING (radius → probe →
// double → re-probe), and the growth's stopping bar is a hit RATE. Measured on
// the reporter's own Menger sponge, that bar stops far too early: growth halts
// at a 0.14% hit rate where the SAME formula's whole-object capture runs at
// 12.5%, i.e. 8972 splats against 71 for a box 1/8.7 the object — the issue
// title, exactly. The DE's convergence floor is absolute (the Menger DE bottoms
// out near 4e-4 everywhere and never goes negative at all), so the eps that
// clears it does not scale with the box either. `epsFloor` states that
// directly, and the framing layer sets it to the eps the WHOLE OBJECT would
// have used — a crop is then never resolved worse than the capture the user is
// comparing it against, and a frame at or above object scale (#518's oversized
// box) is unaffected because the floor is already below its own eps.
test("captureEps #496: epsFloor raises the hit eps, never lowers it, and is inert when absent", () => {
  const base = { center: [0, 0, 0], ext: [1, 1, 1], radius: 1, diag: 2 };
  assert.equal(captureEps(base), 3e-4, "no floor ⇒ the plain scale rule");
  assert.equal(
    captureEps({ ...base, epsFloor: 2e-3 }),
    2e-3,
    "a floor ABOVE the scale rule wins (the small-crop case)",
  );
  assert.equal(
    captureEps({ ...base, epsFloor: 1e-5 }),
    3e-4,
    "a floor BELOW the scale rule is inert — never TIGHTENS a frame's eps",
  );
  assert.equal(
    captureEps({ ...base, radius: 1e-9, epsFloor: 0 }),
    1e-5,
    "the 1e-5 hard floor still bounds a degenerate radius",
  );
});

test("splatmetrics epsFor mirrors captureEps (the metric must measure the eps the capture used)", () => {
  // splatmetrics.js is PURE by contract (no imports), so its eps rule is a COPY
  // of captureEps and has to be kept in step by hand — this is that pin. The
  // observable proxy: metricOnSurface's threshold is r0-based, but the gradient
  // step h = 2·eps is what drifts, so compare the rule numerically via a frame
  // whose floor dominates.
  const base = { center: [0, 0, 0], ext: [1, 1, 1], radius: 1, diag: 2 };
  const frames = [
    base,
    { ...base, epsFloor: 5e-3 }, // #496 framing floor
    { ...base, epsMeasured: 7e-3 }, // #507 measured floor
    { ...base, epsFloor: 5e-3, epsMeasured: 7e-3 }, // both — the max must agree
    { ...base, epsFloor: 7e-3, epsMeasured: 5e-3 },
    { ...base, radius: 1e-9 },
  ];
  for (const f of frames)
    assert.equal(
      metricsEpsFor(f),
      captureEps(f),
      `splatmetrics and splatcapture agree on eps for radius=${f.radius} floor=${f.epsFloor} measured=${f.epsMeasured}`,
    );
});

test("objectEpsFloor #496: measures the whole-object eps, and is 0 (inert) when the probe measures nothing", () => {
  const measurable = {
    name: "sphere-ish",
    ops: [{ key: "boxFold", values: [1] }],
    iters: 4,
  };
  const probed = frameFormula(measurable);
  const floor = objectEpsFloor(measurable);
  if (probed) {
    assert.equal(
      floor,
      3e-4 * probed.radius,
      "the floor is exactly the whole-object frame's own eps",
    );
    // Conservative by construction: frameFormula's radius is the UN-grown one,
    // so the floor can never exceed the eps the object path really marches with.
    assert.ok(
      floor <=
        3e-4 * growFrameToSurface(makeDE(measurable, 4), probed, {}).radius,
      "never looser than the (possibly grown) object frame's eps",
    );
  }
  assert.equal(
    objectEpsFloor({ name: "unprobeable", ops: [], iters: 1, camera: {} }) >= 0,
    true,
    "an unmeasurable formula yields a non-negative (inert-or-real) floor",
  );
});

test("#496: a sub-box of a loose-DE object captures proportionately with the object's eps floor — starves without it", () => {
  // The object: a loose sphere whose DE never reports below FLOOR near the
  // surface. Whole-object framing (radius 8) clears it: eps = 2.4e-3 > FLOOR.
  const FLOOR = 0.002;
  const de = loosySphereDE([0, 0, 0], 5, FLOOR);
  const albedo = () => [1, 1, 1];
  const objFrame = {
    center: [0, 0, 0],
    ext: [8, 8, 8],
    radius: 8,
    diag: 2 * Math.hypot(8, 8, 8),
  };
  const objEps = 3e-4 * objFrame.radius;
  assert.ok(objEps > FLOOR, "premise: the whole object clears the DE's floor");

  // A crop around a patch of the surface, 1/16 the object's linear size. Its own
  // eps (3e-4 · 0.5 = 1.5e-4) is 13× BELOW the floor ⇒ no ray can ever converge.
  const r = 0.5;
  const box = {
    center: [5, 0, 0],
    ext: [r, r, r],
    radius: r,
    diag: 2 * Math.hypot(r, r, r),
  };
  const sweep = (frame) => {
    const out = { pos: [], normal: [], albedo: [] };
    let hits = 0;
    for (let k = 0; k < 24; k++)
      hits += captureView(de, albedo, frame, fibonacciDir(k, 24), 24, out, {
        maxSteps: 200,
        deScale: 1,
      });
    return hits;
  };
  const starved = sweep(box);
  const floored = sweep({ ...box, epsFloor: objEps });
  assert.equal(starved, 0, "without the floor the crop registers nothing");
  assert.ok(
    floored > 100,
    `with the object's eps floor the same crop captures real surface (${floored} hits)`,
  );
  // And the promise the gizmo makes is kept: the floor moved the EPS, never the
  // captured volume. Every hit is still inside the box the user drew.
  const out = { pos: [], normal: [], albedo: [] };
  for (let k = 0; k < 24; k++)
    captureView(
      de,
      albedo,
      { ...box, epsFloor: objEps },
      fibonacciDir(k, 24),
      24,
      out,
      {
        maxSteps: 200,
        deScale: 1,
      },
    );
  for (let i = 0; i < out.pos.length; i += 3)
    assert.ok(
      volInside(
        box,
        out.pos[i] - box.center[0],
        out.pos[i + 1] - box.center[1],
        out.pos[i + 2] - box.center[2],
      ),
      "every hit lies inside the drawn box — ext/center are untouched",
    );
});

test("#496: the eps floor removes the need to GROW the crop's radius (so #502's over-growth can't trigger)", () => {
  // Before the floor, the only lever was growEpsToSurface doubling `radius`,
  // which is also the AO tap scale (aoScale: h = max(4·eps, 0.01·radius)) — on
  // a close-up a 10× radius coarsens every AO tap to a quarter of the box, the
  // "colors are smudged" half of the report. With the floor in place the first
  // probe already succeeds, so radius (and therefore AO) stays the box's own.
  const FLOOR = 0.002;
  const de = loosySphereDE([0, 0, 0], 5, FLOOR);
  const r = 0.5;
  const box = {
    center: [5, 0, 0],
    ext: [r, r, r],
    radius: r,
    diag: 2 * Math.hypot(r, r, r),
    epsFloor: 3e-4 * 8, // the whole object's eps
  };
  const grown = growEpsToSurface(de, box, { deScale: 1 });
  assert.equal(
    grown,
    box,
    "already clears the bar ⇒ same reference, no growth",
  );
  assert.equal(
    grown.radius,
    r,
    "radius — and so the AO tap scale — is unchanged",
  );
  assert.equal(grown.epsFloor, box.epsFloor, "the floor survives the helper");
});

test("#496: growEpsToSurface and growFrameToSurface carry epsFloor through when they DO grow", () => {
  // A floor that is itself too tight (an object whose own eps doesn't clear the
  // field's floor either — #351's shape) must still fall through to growth, and
  // the grown frame must not silently drop the floor.
  const de = loosySphereDE([0, 0, 0], 5, 0.002);
  const tiny = {
    center: [5, 0, 0],
    ext: [0.5, 0.5, 0.5],
    radius: 0.5,
    diag: 2 * Math.hypot(0.5, 0.5, 0.5),
    epsFloor: 1e-6, // far below the DE floor ⇒ inert, growth still required
  };
  const grown = growEpsToSurface(de, tiny, { deScale: 1 });
  assert.ok(grown.radius > tiny.radius, "still grows when the floor is inert");
  assert.equal(grown.epsFloor, 1e-6, "epsFloor is preserved across growth");
  const gf = growFrameToSurface(
    de,
    { ...tiny, epsFloor: 1e-6 },
    { deScale: 1 },
  );
  assert.equal(gf.epsFloor, 1e-6, "growFrameToSurface preserves it too");
});

test("captureSplats #351: 'Drifting Sponge Octacale' (ScaleDrift decay + kaleido/octaFold) captures instead of reporting no surface", () => {
  // The exact reporter formula from issue #351 — previously captureSplats()
  // returned null ("no surface found") for this formula's whole-object frame.
  const formula = {
    name: "Drifting Sponge Octacale",
    iters: 7,
    deOption: 2,
    ops: [
      { key: "menger", values: [-0.05] },
      { key: "scaleDrift", values: [2, 0.08] },
      { key: "translate", values: [-1.12, -1.48, -0.6] },
      { key: "kaleido", values: [5, 45.5, 1] },
      { key: "octaFold", values: [] },
      { key: "scaleDrift", values: [1.03, -0.33] },
      { key: "translate", values: [0.04, -0.06, -0.09] },
    ],
    camera: {
      yawDeg: 136.1,
      pitchDeg: 37,
      dist: 5.288056566997441,
      target: [0.05892588109143022, -0.09794680224278356, -0.14888802889158695],
    },
  };
  const r = captureSplats(formula, defaultColoring(), {
    views: 16,
    res: 32,
    cap: 50_000,
  });
  assert.ok(r, "#351: capture succeeds instead of returning null");
  assert.ok(r.stats.rawHits > 0, "#351: real raw hits, not an empty guard");
  assert.ok(r.points.count > 0, "#351: reduced cloud is non-empty");
});

test("#426: deOption-2 icosaFold/kaleido stack needs the frame-grow — fitted frame misses, grown frame captures (the GPU-frame path must grow, same as the CPU path)", () => {
  // The exact #426 repro. #426 was filed as a WGSL `w`-bookkeeping divergence
  // (GPU marches a uniformly-tiny DE); a numeric CPU↔WGSL chain compare
  // (app/scripts/chaindiff-426.mjs) DISPROVED that — the tiers agree on the r/|w|
  // DE to f32. The real cause is framing: this is a LOOSE r/|w| DE whose value
  // never converges below a fixed floor near the surface, so captureView's
  // eps = 3e-4·radius on a tightly-FITTED frame sits below that floor and every
  // ray misses. A GPU frame path that skipped growFrameToSurface (export-splat
  // --gpu) got zero hits while the CPU path — which grows — found the surface,
  // reading exactly like a per-tier DE divergence. This pins the mechanism: the
  // fitted frame genuinely fails and the grow is what recovers it.
  const formula = {
    name: "repro-426",
    iters: 24,
    addC: false,
    deOption: 2,
    camera: { dist: 6, target: [0, 0, 0] },
    ops: [
      { key: "icosaFold", values: [] },
      { key: "kaleido", values: [5, 0, 1] },
      { key: "translate", values: [-1.17, -0.98, 0.05] },
      { key: "scale", values: [1.16] },
    ],
  };
  const de = makeDE(formula, 24);
  const deScale = deScaleFor(formula);
  assert.equal(
    deScale,
    0.3,
    "#426: this stack is a loose IFS DE (deScale 0.3)",
  );

  const fitted = frameFormula(formula);
  assert.ok(fitted, "#426: frameFormula finds a provisional frame");

  // Fitted frame: eps below the loose-DE floor ⇒ (near) zero hits.
  const marchHits = (frame) => {
    const out = { pos: [], normal: [], albedo: [] };
    let hits = 0;
    for (let k = 0; k < 16; k++)
      hits += captureView(
        de,
        () => [0, 0, 0],
        frame,
        fibonacciDir(k, 16),
        32,
        out,
        {
          maxSteps: 200,
          deScale,
          layers: 2,
        },
      );
    return hits;
  };
  assert.equal(
    marchHits(fitted),
    0,
    "#426: the tightly-fitted frame misses the surface (eps below the loose-DE floor)",
  );

  // Growing the frame loosens eps past the floor ⇒ the surface appears.
  const grown = growFrameToSurface(de, fitted, { deScale });
  assert.ok(grown.radius > fitted.radius, "#426: the frame actually grew");
  assert.ok(
    marchHits(grown) > 0,
    "#426: the grown frame captures the surface — the grow is load-bearing",
  );

  // The CPU path (captureSplats) grows internally, so it must succeed as-is —
  // the parity the GPU frame path (export-splat --gpu) now matches.
  const r = captureSplats(formula, defaultColoring(), {
    views: 16,
    res: 32,
    cap: 50_000,
  });
  assert.ok(
    r && r.stats.rawHits > 0,
    "#426: captureSplats (auto-grow) finds the surface",
  );
});

test("deScaleFor: scenes / loose / tight select the right march step", () => {
  assert.equal(
    deScaleFor(PRESETS.find((p) => p.name === "Mandelbulb")),
    1,
    "tight analytic → 1",
  );
  assert.equal(
    deScaleFor(PRESETS.find((p) => p.name === "Abs Menger")),
    0.3,
    "loose IFS → 0.3",
  );
  const union = sceneWrap([
    sceneObj(2, [0.6, 0, 0, 0], [-1, 0, 0]),
    sceneObj(2, [0.6, 0, 0, 0], [1, 0, 0]),
  ]);
  assert.equal(deScaleFor(union), 0.5, "pure-union scene → 0.5 base");
});

test("reducePoints: dedups per cell, averages, drops canceling normals", () => {
  // two clusters (each 2 pts, same cell) + one opposing-normal pair that cancels.
  const pos = [
    0.0,
    0.0,
    0.0,
    0.05,
    0.0,
    0.0, // cluster A (cell 0)
    5.0,
    0.0,
    0.0,
    5.05,
    0.0,
    0.0, // cluster B (far cell)
    9.0,
    0.0,
    0.0,
    9.0,
    0.0,
    0.0, // pair with opposing normals
  ];
  const normal = [
    1,
    0,
    0,
    1,
    0,
    0, //
    0,
    1,
    0,
    0,
    1,
    0, //
    1,
    0,
    0,
    -1,
    0,
    0, // cancel → dropped
  ];
  const albedo = [
    0.2,
    0.2,
    0.2,
    0.4,
    0.4,
    0.4, // avg 0.3
    0.6,
    0.6,
    0.6,
    0.8,
    0.8,
    0.8, //
    0.1,
    0.1,
    0.1,
    0.9,
    0.9,
    0.9, //
  ];
  const raw = {
    count: 6,
    pos: Float32Array.from(pos),
    normal: Float32Array.from(normal),
    albedo: Float32Array.from(albedo),
  };
  const { points, kept, dropped } = reducePoints(raw, 1.0, 1_000_000);
  assert.equal(kept, 2, "two real clusters kept, canceling pair dropped");
  assert.equal(dropped, 4);
  // cluster A albedo averaged to 0.3
  const near0 = points.pos.indexOf(0) === 0 || Math.abs(points.pos[0]) < 0.1;
  assert.ok(near0);
  assert.ok(Math.abs(points.albedo[0] - 0.3) < 1e-6, "albedo averaged");
});

test("reducePoints: cap forces deterministic re-reduce, reports dropped", () => {
  // 64 well-separated points, cap 8 → must merge down to ≤ 8.
  const N = 64,
    pos = [],
    normal = [],
    albedo = [];
  for (let i = 0; i < N; i++) {
    pos.push(i * 2, 0, 0);
    normal.push(0, 0, 1);
    albedo.push(0.5, 0.5, 0.5);
  }
  const raw = {
    count: N,
    pos: Float32Array.from(pos),
    normal: Float32Array.from(normal),
    albedo: Float32Array.from(albedo),
  };
  const { kept, dropped } = reducePoints(raw, 0.5, 8);
  assert.ok(kept <= 8, `kept ${kept} ≤ cap`);
  assert.equal(kept + dropped, N);
});

// ── S1b: auto-tune (per-splat local radius + overlap-attenuated opacity) ──────

const gridRaw = (build) => {
  const pos = [],
    normal = [],
    albedo = [];
  build((x, y, z) => {
    pos.push(x, y, z);
    normal.push(0, 0, 1);
    albedo.push(0.5, 0.5, 0.5);
  });
  return {
    count: pos.length / 3,
    pos: Float32Array.from(pos),
    normal: Float32Array.from(normal),
    albedo: Float32Array.from(albedo),
  };
};

test("reducePoints S1b: uniform lattice → uniform radius; attenGamma 0 = uniform alpha", () => {
  // 5×5×5 grid spacing 1, cell = r0 = 1 → one point per cell, nearest at dist 1.
  const raw = gridRaw((p) => {
    for (let x = 0; x < 5; x++)
      for (let y = 0; y < 5; y++) for (let z = 0; z < 5; z++) p(x, y, z);
  });
  const { points } = reducePoints(raw, 1, 1_000_000, {
    radiusScale: 1.6,
    attenGamma: 0.5,
  });
  assert.equal(points.count, 125);
  for (const r of points.radius)
    assert.ok(Math.abs(r - 1.6) < 1e-4, `uniform radius ${r}`);
  for (const a of points.alpha)
    assert.ok(a >= 0.3 - 1e-9 && a <= 0.95 + 1e-9, `alpha in range ${a}`);
  // attenGamma 0 ⇒ every splat sits at alphaBase (the S0 uniform-opacity behavior)
  const flat = reducePoints(raw, 1, 1_000_000, { attenGamma: 0 }).points;
  for (const a of flat.alpha)
    assert.ok(Math.abs(a - 0.95) < 1e-6, `flat alpha ${a}`); // f32
});

test("reducePoints S1b: isolated point gets gap-cover radius; dense clump attenuates", () => {
  const raw = gridRaw((p) => {
    for (let x = 0; x < 3; x++)
      for (let y = 0; y < 3; y++)
        for (let z = 0; z < 3; z++) p(x * 0.25, y * 0.25, z * 0.25);
    p(10, 10, 10); // a loner, far from the clump
  });
  const cell = 0.2,
    rs = 1.6;
  const { points } = reducePoints(raw, cell, 1_000_000, {
    radiusScale: rs,
    attenGamma: 0.5,
  });
  let li = -1;
  for (let i = 0; i < points.count; i++) if (points.pos[3 * i] > 5) li = i;
  assert.ok(li >= 0, "loner survived");
  // no stencil neighbor ⇒ max gap-cover radius = rClampHi(3)·cell·radiusScale
  assert.ok(
    Math.abs(points.radius[li] - 3 * cell * rs) < 1e-4,
    `loner radius ${points.radius[li]}`,
  );
  assert.ok(
    Math.abs(points.alpha[li] - 0.95) < 1e-6,
    `loner alpha = base ${points.alpha[li]}`,
  );
  let clumpMinA = 1;
  for (let i = 0; i < points.count; i++)
    if (i !== li) clumpMinA = Math.min(clumpMinA, points.alpha[i]);
  assert.ok(
    clumpMinA < 0.95,
    `dense clump attenuated below base (${clumpMinA})`,
  );
  assert.ok(clumpMinA < points.alpha[li], "clump alpha < loner alpha");
});

test("reducePoints S1b: radius/alpha are deterministic across runs", () => {
  const raw = gridRaw((p) => {
    for (let x = 0; x < 4; x++)
      for (let y = 0; y < 4; y++) for (let z = 0; z < 2; z++) p(x, y, z);
    p(20, 20, 20);
  });
  const a = reducePoints(raw, 1, 1_000_000).points;
  const b = reducePoints(raw, 1, 1_000_000).points;
  assert.deepEqual([...a.radius], [...b.radius]);
  assert.deepEqual([...a.alpha], [...b.alpha]);
});

// ── S1c: depth-peel (layers) ─────────────────────────────────────────────────

const unitSphere = (x, y, z) => Math.hypot(x, y, z) - 1;
const grey = () => [0.6, 0.6, 0.6];

test("captureView S1c: layers>1 peels to the back surface a solid occludes", () => {
  const frame = { center: [0, 0, 0], radius: 1.5 };
  const back = (layers) => {
    const out = { pos: [], normal: [], albedo: [] };
    captureView(unitSphere, grey, frame, [1, 0, 0], 16, out, { layers });
    let f = 0,
      b = 0;
    for (let i = 0; i < out.pos.length / 3; i++) out.pos[3 * i] < 0 ? f++ : b++;
    return { f, b };
  };
  const one = back(1),
    two = back(2);
  assert.equal(one.b, 0, "layers 1 sees only the near hemisphere");
  assert.ok(one.f > 0, "layers 1 sees the near hemisphere");
  assert.ok(
    two.b > 0,
    "layers 2 reaches the far hemisphere the solid occludes",
  );
});

test("captureView S1c: two thin walls — layers 1 hits only the front, layers≥2 both", () => {
  // two thin solid walls at x=∓0.3 (each surface has a clean ∇DE)
  const t0 = 0.01,
    a = -0.3,
    b = 0.3;
  const walls = (x) => Math.min(Math.abs(x - a) - t0, Math.abs(x - b) - t0);
  const de = (x, y, z) => walls(x);
  const frame = { center: [0, 0, 0], radius: 1 };
  const cross = (layers) => {
    const out = { pos: [], normal: [], albedo: [] };
    captureView(de, grey, frame, [1, 0, 0], 2, out, { layers });
    let A = false,
      B = false;
    for (let i = 0; i < out.pos.length / 3; i++)
      out.pos[3 * i] < 0 ? (A = true) : (B = true);
    return { A, B };
  };
  assert.deepEqual(
    cross(1),
    { A: true, B: false },
    "one layer: only the front wall",
  );
  assert.deepEqual(cross(3), { A: true, B: true }, "three layers: both walls");
});

test("captureView S1c: thin-feature floor — well-separated walls both captured", () => {
  // Sweep wall separation. Both walls resolve while sep ≫ eps; below a few·eps
  // the re-arm nudge merges them (documented thin-feature resolution floor).
  const frame = { center: [0, 0, 0], radius: 1 };
  const eps = Math.max(3e-4 * 1, 1e-5);
  const captured = (sepEps) => {
    const aa = (-sepEps * eps) / 2,
      bb = (sepEps * eps) / 2,
      tw = 0.3 * eps;
    const de = (x) => Math.min(Math.abs(x - aa) - tw, Math.abs(x - bb) - tw);
    const out = { pos: [], normal: [], albedo: [] };
    captureView((x, y, z) => de(x), grey, frame, [1, 0, 0], 1, out, {
      layers: 3,
    });
    let A = false,
      B = false;
    for (let i = 0; i < out.pos.length / 3; i++)
      out.pos[3 * i] < 0 ? (A = true) : (B = true);
    return A && B;
  };
  for (const sep of [20, 10, 6])
    assert.ok(captured(sep), `both walls at sep ${sep}·eps`);
  // Floor: below ~5·eps resolution is unreliable (features merge). If this ever
  // starts passing at sep=1, the re-arm nudge changed — re-measure the floor.
  assert.equal(
    captured(1),
    false,
    "sub-eps-scale walls merge (documented floor ≈ a few·eps)",
  );
});

test("captureView S1c: one ray is bounded by ~layers·maxSteps march steps", () => {
  let calls = 0;
  const counted = (x, y, z) => {
    calls++;
    return unitSphere(x, y, z);
  };
  const out = { pos: [], normal: [], albedo: [] };
  const frame = { center: [0, 0, 0], radius: 1.5 };
  captureView(counted, grey, frame, [1, 0, 0], 1, out, {
    layers: 3,
    maxSteps: 200,
  });
  // march ≤ layers·maxSteps, plus 6 normal taps per hit (≤ layers) — never unbounded
  assert.ok(calls < 3 * 200 + 6 * 3 + 10, `bounded de calls (${calls})`);
});

test("captureView S1c/d: default opts ≡ {layers:1, aoStrength:0} (byte-identical S0 path)", () => {
  const frame = { center: [0, 0, 0], radius: 1.5 };
  const def = { pos: [], normal: [], albedo: [] };
  const s0 = { pos: [], normal: [], albedo: [] };
  for (let k = 0; k < 8; k++) {
    captureView(unitSphere, grey, frame, fibonacciDir(k, 8), 20, def, {});
    captureView(unitSphere, grey, frame, fibonacciDir(k, 8), 20, s0, {
      layers: 1,
      aoStrength: 0,
    });
  }
  assert.deepEqual([...def.pos], [...s0.pos]);
  assert.deepEqual([...def.albedo], [...s0.albedo]);
});

// ── S1d: baked DE ambient occlusion ──────────────────────────────────────────

test("captureView S1d: AO leaves a convex surface unchanged (occ ≈ 0)", () => {
  const frame = { center: [0, 0, 0], radius: 1.5 };
  const a0 = { pos: [], normal: [], albedo: [] };
  const a5 = { pos: [], normal: [], albedo: [] };
  captureView(unitSphere, grey, frame, [1, 0, 0], 16, a0, { aoStrength: 0 });
  captureView(unitSphere, grey, frame, [1, 0, 0], 16, a5, { aoStrength: 0.5 });
  let maxDiff = 0;
  for (let i = 0; i < a0.albedo.length; i++)
    maxDiff = Math.max(maxDiff, Math.abs(a0.albedo[i] - a5.albedo[i]));
  assert.ok(
    maxDiff < 1e-3,
    `convex sphere unchanged by AO (max diff ${maxDiff})`,
  );
});

test("captureView S1d: a concave edge darkens, bounded by [1−aoStrength, 1]", () => {
  // 90° wedge: air (de>0) in the x>0,y>0 quadrant, walls at x=0 and y=0.
  const wedge = (x, y, z) => Math.min(x, y);
  const aoStrength = 0.6;
  const frame = { center: [0.5, 0.5, 0], radius: 1 };
  const out = { pos: [], normal: [], albedo: [] };
  for (const v of [
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
  ])
    captureView(wedge, grey, frame, v, 40, out, { aoStrength });
  let globalMin = 1,
    darkestDist = Infinity;
  for (let i = 0; i < out.pos.length / 3; i++) {
    const ao = out.albedo[3 * i] / 0.6; // recover ao (albedo = 0.6·ao)
    assert.ok(
      ao >= 1 - aoStrength - 1e-6 && ao <= 1 + 1e-6,
      `ao in band: ${ao}`,
    );
    if (ao < globalMin) {
      globalMin = ao;
      darkestDist = Math.hypot(out.pos[3 * i], out.pos[3 * i + 1]); // dist to the concave edge
    }
  }
  assert.ok(
    globalMin < 0.99,
    `the concave edge is occluded (min ao ${globalMin})`,
  );
  assert.ok(darkestDist < 0.2, "the darkest point sits at the concave edge");
});

// ── Streaming reduce (SPLAT_STREAMING_REDUCE PR-1) ────────────────────────────
// stream:true keeps a bounded per-cell accumulator at a FIXED frame.center origin
// and coarsens by INTEGER factors, instead of materialising the whole raw cloud
// (the OOM fix). It therefore shifts slightly from the batch reduce (origin
// lattice offset + integer-factor cap landing vs the batch cbrt landing). These
// pins bound that shift and prove the memory bound.
const streamMetrics = (f, r) =>
  splatMetrics(
    {
      de: makeDE(f, r.stats.iters),
      albedoAt: makePointAlbedo(f, defaultColoring(), r.stats.iters),
      aoScale,
      sample: r.sample,
      points: r.points,
      r0: r.r0,
      frame: r.frame,
    },
    { aoStrength: 0.5 },
  );
// Bounded metrics ([0,1]/[-1,1]) hold tight; `overdraw` is an UNBOUNDED mean-
// overlap COUNT (~4-5) far more sensitive to borderline points flipping in/out
// of a neighbor radius under the origin+coarsen change — same rationale the
// splatmetrics S2 pins use for their ~10× looser overdraw tolerance. Both are
// still ~50× tighter than a real regression (which moves overdraw by units).
// #518 raised this from 1.2e-2. Until then BOTH paths sized their lattice from
// `frame.diag`, so under cap they landed on the same pitch to 4 decimal places
// (measured stream/batch cell ratio 1.0004) and any Δ here was pure lattice-
// ORIGIN offset. The pitch now follows the CAPTURED extent instead (see
// capturedDiag), which the batch path measures over the whole cloud while the
// streaming path must ESTIMATE from its first chunk — one view under-covers the
// object along its own axis, so the streaming lattice runs slightly fine
// (measured 0.951× batch's on Mandelbulb). A finer lattice averages fewer hits
// per cell, which RAISES normalAgreement (stream 0.7925 vs batch 0.7772) — the
// two paths are not diverging in quality, they are resolving differently by the
// width of an estimate that streaming cannot avoid making. onSurface (3.7e-4)
// and coverage (3.9e-5) — the pins a real regression trips — did not move, and
// the estimate itself is now pinned directly, and much more tightly, by the
// cell-ratio assertion in the under-cap test below.
const S_BOUNDED_TOL = 2e-2;
const S_OVERDRAW_TOL = 1.5e-1;
// colorDrift gets its own, slightly looser bound. It is a mean-COLOUR difference
// between two survivor SETS, so under aggressive coarsening (this file's cap-8000
// case reduces ~50k raw hits) it swings on which borderline cells each path keeps
// — batch anchors its lattice at the cloud's minx, streaming at frame.center, and
// they coarsen by different rules (cbrt loop vs integer factor). It is a proxy
// for the coarsening REGIME, not for capture quality.
//
// Measured on Carved Cube after the CAPTURE_VOLUME_SHAPES box clip: Δ 1.20e-2,
// against a pin of exactly 1.2e-2 — i.e. it now sits ON the old bound and fails
// on float rounding alone. 2e-2 restores headroom without pretending the metric
// is tighter than it is. For scale, changing ONLY the frame size (a 1.1× margin,
// no capture-path change at all) moved this same number 4.16e-3 → 1.77e-2, so
// the old pin was never as discriminating as it looked.
//
// Quality is NOT what moved: onSurface and coverage are 1.0000 on both paths and
// normalAgreement holds inside 1.4e-3, and those are the pins a real regression
// actually trips. The principled fix is to anchor both reduces on the same
// lattice origin, which would move every batch baseline pin — left as follow-up.
const S_COLORDRIFT_TOL = 2e-2;
const checkParity = (f, batch, strm) => {
  const mb = streamMetrics(f, batch),
    ms = streamMetrics(f, strm);
  for (const k of ["onSurface", "coverage", "normalAgreement", "colorDrift"]) {
    const tol = k === "colorDrift" ? S_COLORDRIFT_TOL : S_BOUNDED_TOL;
    assert.ok(
      Math.abs(mb[k] - ms[k]) <= tol,
      `${k}: batch ${mb[k].toFixed(4)} vs stream ${ms[k].toFixed(4)} (Δ ${Math.abs(mb[k] - ms[k]).toExponential(2)}, tol ${tol})`,
    );
  }
  assert.ok(
    Math.abs(mb.overdraw - ms.overdraw) <= S_OVERDRAW_TOL,
    `overdraw: batch ${mb.overdraw.toFixed(3)} vs stream ${ms.overdraw.toFixed(3)} (Δ ${Math.abs(mb.overdraw - ms.overdraw).toExponential(2)}, tol ${S_OVERDRAW_TOL})`,
  );
};

test("captureSplats streaming ≈ batch: cloud exceeds cap → coarsened, lands in §4 band", () => {
  const f = PRESETS.find((p) => p.name === "Carved Cube");
  const cap = 8000;
  const opts = {
    views: 12,
    res: 64,
    cap,
    layers: 2,
    aoStrength: 0.5,
    sampleHits: 120_000,
  };
  const batch = captureSplats(f, defaultColoring(), { ...opts });
  const strm = captureSplats(f, defaultColoring(), { ...opts, stream: true });
  assert.ok(batch && strm, "both captured");
  // This config's raw cloud far exceeds cap (batch is forced to exactly cap) and
  // the streaming accumulator had to coarsen (its grid peaked over cap) — the
  // interesting integer-coarsen regime.
  assert.ok(
    batch.stats.rawHits > 2 * cap,
    `cloud exceeds cap (rawHits ${batch.stats.rawHits})`,
  );
  // The huge cloud forces mid-stream coarsening: the recorded peak (post-coarsen)
  // now lands at/just under cap under the tightened MEM_CEIL (1.3·cap) — the grid
  // filled substantially then coarsened. (Was `> cap` under the old 2·cap ceiling,
  // which let the live grid balloon before finalize capped it.)
  assert.ok(
    strm.stats.maxAccSize > 0.7 * cap,
    `streaming accumulator filled + coarsened (peak ${strm.stats.maxAccSize})`,
  );
  // §4 achieved-cap band: survivors in ~[0.5·cap, cap] when the cloud exceeds cap.
  assert.ok(strm.stats.kept <= cap, `kept ${strm.stats.kept} ≤ cap`);
  assert.ok(
    strm.stats.kept >= 0.5 * cap,
    `kept ${strm.stats.kept} ≥ 0.5·cap (§4 band)`,
  );
  // Memory bound: the persistent accumulator never tops 2·cap.
  assert.ok(
    strm.stats.maxAccSize <= Math.ceil(1.3 * cap),
    `maxAcc ${strm.stats.maxAccSize} ≤ MEM_CEIL (1.3·cap)`,
  );
  checkParity(f, batch, strm);
});

test("captureSplats streaming ≈ batch: cloud under cap → no coarsening, counts equal-ish", () => {
  const f = PRESETS.find((p) => p.name === "Mandelbulb");
  const cap = 40_000;
  const opts = {
    views: 12,
    res: 64,
    cap,
    layers: 2,
    aoStrength: 0.5,
    sampleHits: 120_000,
  };
  const batch = captureSplats(f, defaultColoring(), { ...opts });
  const strm = captureSplats(f, defaultColoring(), { ...opts, stream: true });
  // Neither path hits cap here (batch < cap ⇒ no cbrt loop; stream ⇒ no coarsen),
  // so survivor counts match closely — they differ only by the origin lattice
  // shift (frame.center vs the batch's minx).
  assert.ok(
    batch.stats.kept < cap && strm.stats.kept < cap,
    "under cap, no coarsening",
  );
  assert.ok(strm.stats.maxAccSize <= 2 * cap);
  const rel = Math.abs(strm.stats.kept - batch.stats.kept) / batch.stats.kept;
  assert.ok(
    rel < 0.05,
    `kept within 5%: batch ${batch.stats.kept} vs stream ${strm.stats.kept} (${(rel * 100).toFixed(1)}%)`,
  );
  // #518: the streaming lattice is sized from an ESTIMATE of the captured
  // extent (its first chunk) where the batch path measures it over the whole
  // cloud. Pin that estimate directly — it is the one quantity that actually
  // differs, and this bounds it far more tightly (measured 0.951×) than reading
  // it off a downstream quality metric.
  const cellRatio = strm.cell / batch.cell;
  assert.ok(
    Math.abs(cellRatio - 1) <= 0.1,
    `streaming's first-chunk pitch estimate is within 10% of the measured one (${cellRatio.toFixed(3)}×)`,
  );
  checkParity(f, batch, strm);
});

test("captureSplats streaming: memory bounded ≤ 2·cap on a hit-count ≫ cap cloud; valid output + bbox", () => {
  const f = PRESETS.find((p) => p.name === "Mandelbulb");
  const cap = 8000;
  const strm = captureSplats(f, defaultColoring(), {
    views: 12,
    res: 64,
    cap,
    layers: 2,
    aoStrength: 0.5,
    stream: true,
  });
  assert.ok(strm, "captured");
  assert.ok(
    strm.stats.rawHits > 4 * cap,
    `raw hits ≫ cap (${strm.stats.rawHits}) — would balloon the batch out[]`,
  );
  assert.ok(
    strm.stats.maxAccSize > cap,
    `accumulator forced to coarsen (peak ${strm.stats.maxAccSize} > cap)`,
  );
  assert.ok(
    strm.stats.maxAccSize <= Math.ceil(1.3 * cap),
    `accumulator ≤ 2·cap (${strm.stats.maxAccSize})`,
  );
  assert.ok(
    strm.points.count > 0 && strm.points.count <= cap,
    `non-empty, ≤ cap points (${strm.points.count})`,
  );
  // Running world bbox is exposed (PR-2's worldBBox(raw) replacement) and sane.
  assert.ok(
    strm.bbox &&
      strm.bbox.min.every(Number.isFinite) &&
      strm.bbox.max.every(Number.isFinite),
    "bbox exposed + finite",
  );
  for (let c = 0; c < 3; c++)
    assert.ok(strm.bbox.max[c] >= strm.bbox.min[c], `bbox axis ${c} ordered`);
});

test("captureSplats stream:false = default (flag inert; batch return shape unchanged)", () => {
  const f = PRESETS.find((p) => p.name === "Mandelbulb");
  const opts = { views: 4, res: 16, cap: 40_000 };
  const def = captureSplats(f, defaultColoring(), { ...opts });
  const off = captureSplats(f, defaultColoring(), { ...opts, stream: false });
  assert.deepEqual(
    [...off.points.pos],
    [...def.points.pos],
    "stream:false pos byte-identical to default",
  );
  assert.deepEqual(
    [...off.points.radius],
    [...def.points.radius],
    "stream:false radius byte-identical",
  );
  assert.equal(off.stats.kept, def.stats.kept);
  assert.equal(off.bbox, undefined, "batch path exposes no bbox");
  assert.equal(
    off.stats.maxAccSize,
    undefined,
    "batch path exposes no maxAccSize",
  );
});

// ── S-2 snapPoints (SPLAT_SHARPNESS §S-2) ────────────────────────────────────
test("snapPoints: Newton-snaps off-surface centers onto an analytic sphere", () => {
  // Unit-sphere SDF; points jittered radially off the surface (the cell-average
  // error the reduce introduces). Snap must land them at |p| ≈ 1, point normals
  // radially, and resample albedo at the SNAPPED position.
  const de = (x, y, z) => Math.hypot(x, y, z) - 1;
  const albedoAt = (x) => [x > 0 ? 0.9 : 0.1, 0.5, 0.5]; // side-coded color
  const n = 64;
  const pos = new Float32Array(3 * n),
    nor = new Float32Array(3 * n),
    alb = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    const th = (i / n) * Math.PI * 2,
      ph = ((i % 8) / 8) * Math.PI - Math.PI / 2;
    const r = 1 + (i % 2 ? 0.04 : -0.04); // ±0.04 off-surface (≤ cell 0.1)
    pos[3 * i] = r * Math.cos(ph) * Math.cos(th);
    pos[3 * i + 1] = r * Math.cos(ph) * Math.sin(th);
    pos[3 * i + 2] = r * Math.sin(ph);
    // approximately-correct radial normals — the snap's contract (it marches
    // ALONG the stored normal; the reduce provides hit-averaged ones)
    nor[3 * i] = Math.cos(ph) * Math.cos(th);
    nor[3 * i + 1] = Math.cos(ph) * Math.sin(th);
    nor[3 * i + 2] = Math.sin(ph);
  }
  const points = { count: n, pos, normal: nor, albedo: alb };
  const stats = snapPoints(points, de, albedoAt, {
    cell: 0.1,
    eps: 1e-4,
    aoStrength: 0, // isolate the snap (no AO factor)
    resample: true, // exercise the opt-in ∇DE-normal + albedo resample too
  });
  assert.equal(stats.snapped, n, "every point snapped");
  assert.equal(stats.rejected, 0);
  for (let i = 0; i < n; i++) {
    const x = pos[3 * i],
      y = pos[3 * i + 1],
      z = pos[3 * i + 2];
    const rr = Math.hypot(x, y, z);
    assert.ok(Math.abs(rr - 1) < 2e-3, `|p| ≈ 1 (got ${rr})`);
    // normal ≈ radial (dot with p̂ ≈ 1)
    const d =
      (nor[3 * i] * x + nor[3 * i + 1] * y + nor[3 * i + 2] * z) / (rr || 1);
    assert.ok(d > 0.999, `normal radial (dot ${d})`);
    // albedo resampled at the snapped x (side-coded; f32 storage tolerance)
    assert.ok(
      Math.abs(alb[3 * i] - (x > 0 ? 0.9 : 0.1)) < 1e-6,
      "albedo from snapped pos",
    );
  }
  assert.ok(stats.meanAbsAfter < stats.meanAbsBefore / 10, "|DE| collapsed");
});

test("snapPoints: default is POSITION-ONLY (normal/albedo untouched)", () => {
  const de = (x, y, z) => Math.hypot(x, y, z) - 1;
  const albedoAt = () => [0.7, 0.7, 0.7];
  const pos = Float32Array.from([1.05, 0, 0]); // slightly off-surface
  const nor = Float32Array.from([1, 0, 0]); // the hit-averaged normal
  const alb = Float32Array.from([0.3, 0.3, 0.3]);
  const points = { count: 1, pos, normal: nor, albedo: alb };
  const stats = snapPoints(points, de, albedoAt, { cell: 0.1, eps: 1e-4 });
  assert.equal(stats.snapped, 1);
  assert.ok(
    Math.abs(Math.hypot(pos[0], pos[1], pos[2]) - 1) < 1e-3,
    "position snapped",
  );
  assert.deepEqual(
    [...nor],
    [1, 0, 0],
    "averaged normal KEPT (footprint average)",
  );
  assert.ok(Math.abs(alb[0] - 0.3) < 1e-6, "averaged albedo KEPT");
});

test("snapPoints: displacement beyond cell is rejected (average kept)", () => {
  const de = (x, y, z) => Math.hypot(x, y, z) - 1;
  const albedoAt = () => [0.5, 0.5, 0.5];
  // one point FAR off-surface (0.5 away, cell 0.1) → the directional walk exits
  // the cell budget → kept untouched (degenerate bucket)
  const pos = Float32Array.from([1.5, 0, 0]);
  const nor = Float32Array.from([1, 0, 0]);
  const alb = Float32Array.from([0.2, 0.2, 0.2]);
  const points = { count: 1, pos, normal: nor, albedo: alb };
  const stats = snapPoints(points, de, albedoAt, { cell: 0.1, eps: 1e-4 });
  assert.equal(stats.snapped, 0);
  assert.equal(stats.rejected + stats.degenerate, 1, "kept, not snapped");
  assert.deepEqual([...pos], [1.5, 0, 0], "position untouched");
  assert.deepEqual([...nor], [1, 0, 0], "normal untouched");
  assert.deepEqual(
    [...alb],
    Array.from(Float32Array.from([0.2, 0.2, 0.2])),
    "albedo untouched",
  );
});

test("snapPoints: reducePoints + captureSplats expose the cell pitch", () => {
  const f = PRESETS.find((p) => p.name === "Mandelbulb");
  const r = captureSplats(f, defaultColoring(), {
    views: 4,
    res: 16,
    cap: 40_000,
  });
  assert.ok(Number.isFinite(r.cell) && r.cell > 0, "batch result carries cell");
  const s = captureSplats(f, defaultColoring(), {
    views: 4,
    res: 16,
    cap: 40_000,
    stream: true,
  });
  assert.ok(
    Number.isFinite(s.cell) && s.cell > 0,
    "streaming result carries cell",
  );
  assert.ok(Math.abs(s.cell - r.cell) / r.cell < 0.5, "pitches comparable");
});

test("densifySplats: edge parents spawn snapped children along the crease axis", () => {
  const de = (x, y, z) => Math.hypot(x, y, z) - 1;
  // One edge splat on the sphere at (1,0,0), edge axis = the tangent (0,1,0),
  // low dispersion → densify spawns 2 children at ±cell/4 along y, snapped to |p|=1.
  const points = {
    count: 2,
    pos: Float32Array.from([1, 0, 0, 0, 0, 1.0]),
    normal: Float32Array.from([1, 0, 0, 0, 0, 1]),
    albedo: Float32Array.from([0.5, 0.5, 0.5, 0.6, 0.6, 0.6]),
    radius: Float32Array.from([0.05, 0.05]),
    alpha: Float32Array.from([0.9, 0.9]),
    r2: Float32Array.from([0.05, 0.05]),
    dir: Float32Array.from([0, 1, 0, 1, 0, 0]),
    dispersion: Float32Array.from([0.5, 0.99]), // only #0 is an edge (< 0.85)
  };
  const st = densifySplats(points, de, { cell: 0.1, eps: 1e-3, budget: 100 });
  assert.ok(st, "densify ran");
  assert.equal(st.parents, 1, "only the low-dispersion splat is a parent");
  assert.equal(st.added, 2, "2 children");
  assert.equal(points.count, 4);
  for (let i = 2; i < 4; i++) {
    const r = Math.hypot(
      points.pos[3 * i],
      points.pos[3 * i + 1],
      points.pos[3 * i + 2],
    );
    assert.ok(
      Math.abs(r - 1) < 3e-3,
      `child ${i} snapped to the sphere (|p|=${r})`,
    );
    assert.ok(
      Math.abs(points.pos[3 * i + 1]) > 1e-3,
      "offset along the edge axis",
    );
    assert.ok(
      Math.abs(points.radius[i] - 0.05 * 0.7) < 1e-6,
      "child radius 0.7×",
    );
    assert.ok(
      Math.abs(points.albedo[3 * i] - 0.5) < 1e-6,
      "child inherits albedo",
    );
  }
});

test("densifySplats: no-op without aniso channels or budget", () => {
  const de = (x, y, z) => Math.hypot(x, y, z) - 1;
  const bare = {
    count: 1,
    pos: Float32Array.from([1, 0, 0]),
    normal: Float32Array.from([1, 0, 0]),
    albedo: Float32Array.from([0.5, 0.5, 0.5]),
    radius: Float32Array.from([0.05]),
    alpha: Float32Array.from([0.9]),
    dispersion: Float32Array.from([0.5]),
  };
  assert.equal(
    densifySplats(bare, de, { cell: 0.1, eps: 1e-3, budget: 100 }),
    null,
    "no dir → null",
  );
  const full = {
    ...bare,
    r2: Float32Array.from([0.05]),
    dir: Float32Array.from([0, 1, 0]),
  };
  assert.equal(
    densifySplats(full, de, { cell: 0.1, eps: 1e-3, budget: 0 }),
    null,
    "no budget → null",
  );
});

test("finalizeReduce output carries per-splat dispersion; Pass 1c shapes crease cells", () => {
  // Two planes meeting at 90° (normals +x and +z): cells on the crease get low
  // dispersion + an edge-aligned dir; flat cells keep dispersion ≈ 1.
  // Seeded PRNG (mulberry32), NOT Math.random: the closing crease-count
  // assertion sits near its cutoff, and an unseeded fixture made it flake on
  // CI (#393 — same commit green locally, red on one Ubuntu run, green on
  // rerun). A pinned sequence makes the count a constant.
  let seed = 0x393;
  const rand = () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const N = 30000;
  const pos = new Float32Array(3 * N),
    nor = new Float32Array(3 * N),
    alb = new Float32Array(3 * N);
  for (let i = 0; i < N; i++) {
    const onA = i % 2 === 0,
      t = rand() - 0.5,
      w = rand() * 0.5;
    if (onA) {
      pos[3 * i] = -w;
      pos[3 * i + 1] = t;
      pos[3 * i + 2] = 0;
      nor[3 * i + 2] = 1;
    } else {
      pos[3 * i] = 0;
      pos[3 * i + 1] = t;
      pos[3 * i + 2] = -w;
      nor[3 * i] = 1;
    }
    alb[3 * i] = 0.5;
  }
  const raw = { count: N, pos, normal: nor, albedo: alb };
  const r = reducePoints(raw, 0.02, 50000, { aniso: 1 });
  const p = r.points;
  assert.ok(p.dispersion instanceof Float32Array, "dispersion emitted");
  // crease cells (x≈0, z≈0) low dispersion + dir ≈ ±y; flat cells ≈ 1
  let creaseChecked = 0;
  for (let i = 0; i < p.count; i++) {
    const x = p.pos[3 * i],
      z = p.pos[3 * i + 2];
    if (Math.abs(x) < 0.005 && Math.abs(z) < 0.005 && p.dispersion[i] < 0.85) {
      const dy = Math.abs(p.dir[3 * i + 1]);
      assert.ok(dy > 0.9, `crease cell dir along y (|dy|=${dy})`);
      creaseChecked++;
    }
    if (Math.abs(x) > 0.05 && Math.abs(z) < 1e-6)
      assert.ok(p.dispersion[i] > 0.95, "flat cell high dispersion");
  }
  assert.ok(
    creaseChecked > 3,
    `some crease cells found + edge-shaped (${creaseChecked})`,
  );
});

// ── S-5a viewFrame (SPLAT_VIEW_CAPTURE) ──────────────────────────────────────
test("viewFrame: sizes to on-screen extent (aspect + margin), floors tiny dist", () => {
  // vertical half = dist·tan(fov/2); radius = that · aspect · margin.
  const halfV = (d, fov) => d * Math.tan(((fov * Math.PI) / 180) * 0.5);
  const f = viewFrame(
    { dist: 4, target: [1, 2, 3], fovDeg: 42 },
    { aspect: 1.6 },
  );
  const want = halfV(4, 42) * 1.6 * 1.1;
  assert.ok(
    Math.abs(f.radius - want) < 1e-9,
    `radius = halfV·aspect·margin (${f.radius} vs ${want})`,
  );
  assert.deepEqual([...f.center], [1, 2, 3], "center = target");
  assert.ok(Math.abs(f.diag - 2 * f.radius) < 1e-12, "diag = 2r");
  // aspect < 1 clamps to 1 (never smaller than the vertical extent)
  const sq = viewFrame(
    { dist: 4, target: [0, 0, 0], fovDeg: 42 },
    { aspect: 0.5 },
  );
  assert.ok(
    Math.abs(sq.radius - halfV(4, 42) * 1.1) < 1e-9,
    "aspect clamped to ≥1",
  );
  // no fovDeg ⇒ legacy dist/3
  const legacy = viewFrame({ dist: 6, target: [0, 0, 0] });
  assert.ok(
    Math.abs(legacy.radius - (6 / 3) * 1.1) < 1e-9,
    "no fov → dist/3·margin",
  );
  // degenerate zoom clamp → radius floored, finite, non-zero (no r0/eps underflow)
  const tiny = viewFrame({ dist: 1e-9, target: [0, 0, 0], fovDeg: 42 });
  assert.ok(
    tiny.radius >= 1e-4 && Number.isFinite(tiny.radius),
    `radius floored (${tiny.radius})`,
  );
  assert.ok(tiny.diag > 0, "diag positive");
});

// #438: a zoomed VIEW frame is chosen by the camera, not fitted to geometry, so
// it can sit wholly inside empty space — a Menger sponge's centre is a carved
// void, which is exactly where the default orbit centre points. probeFrameHits
// is what lets the export tell that apart from a frame that really has surface.
test("probeFrameHits: sees nothing in a void, sees surface once the frame reaches it (#438)", () => {
  const f = PRESETS.find((p) => p.name === "Menger");
  const iters = f.iters ?? 8;
  const de = makeDE(f, iters);
  const deScale = deScaleFor(f);
  // The sponge's centre is empty: the nearest surface is DE(0,0,0) away.
  const gap = de(0, 0, 0);
  assert.ok(gap > 0, `Menger centre is a void (DE ${gap})`);
  // A frame comfortably inside that void sees nothing at all...
  const inVoid = { center: [0, 0, 0], radius: gap * 0.4 };
  assert.equal(probeFrameHits(de, inVoid, { deScale }), 0);
  // ...and growing it lands a frame that does (the export's #438 recovery).
  const grown = growFrameToSurface(de, inVoid, { deScale });
  assert.ok(grown.radius > inVoid.radius, "grew past the void");
  assert.ok(
    probeFrameHits(de, grown, { deScale }) > 0,
    "grown frame has surface",
  );
});

// #450: a beta tester drew a capture volume around a bar INSIDE a Menger sponge
// and the export refused it as "empty". The volume was full of surface — but
// every ray reached it only after crossing sponge material, and an out-of-volume
// hit is stepped over 3·eps at a time, one unit of the shared march budget each.
// Crossing one wall costs thousands of steps against a 200-step budget (the
// probe's, and the real capture's first attempt), so the rays died before
// arriving. The fix clips each ray to the volume, so the budget is only ever
// spent inside it. The invariant: a volume that CONTAINS surface must not be
// reported empty, however much material sits in front of it.
test("a capture volume behind material is not reported empty (#450)", () => {
  const f = PRESETS.find((p) => p.name === "Abs Menger");
  const de = makeDE(f, f.iters ?? 10);
  const deScale = deScaleFor(f);
  // A slab buried in the sponge's upper structure — surrounded by bars, so most
  // view directions must cross solid material to reach it.
  const buried = {
    center: [0.33, 0.55, 0],
    ext: [0.441, 0.0787, 0.441],
    radius: 0.441,
    diag: 2 * Math.hypot(0.441, 0.0787, 0.441),
  };
  assert.ok(
    probeFrameHits(de, buried, { deScale }) > 0,
    "a volume full of surface must not probe empty",
  );
  // And the verdict must not be an artefact of the budget: the capture at the
  // real 200-step budget must find essentially what an unbounded one finds.
  const at = (maxSteps) => {
    const out = { pos: [], normal: [], albedo: [] };
    for (let k = 0; k < 8; k++)
      captureView(de, () => [0, 0, 0], buried, fibonacciDir(k, 8), 24, out, {
        maxSteps,
        deScale,
      });
    return out.pos.length / 3;
  };
  const tight = at(200);
  const loose = at(20000);
  assert.ok(loose > 0, "the volume really does contain surface");
  assert.ok(
    tight >= 0.95 * loose,
    `budget-starved: ${tight} hits at 200 steps vs ${loose} unbounded`,
  );
});

// #457: frameFormula returns null for 35 of the 90 presets — the evaluate.js
// orbit probe only finalizes the IFS / escape-time DE, so it is blind to leaf
// shapes and to deOption-3 (numeric) stacks and reports hits=0 at EVERY region
// (2.5 -> 40). Callers then fell back to cameraFrame's UNIFORM CUBE.
//
// 22 of those 35 are SCENES, which already reached refineFrame through its
// `scene` gate and were fitted all along — the issue's "35 get a cube" count is
// really 13. Those 13 are FLAT formulas, for which refineFrame was unreachable,
// and they are what this fix rescues: 9 of them now frame to the real surface
// (the other 4 legitimately find no surface and keep the cube).
//
// BristorBrot is the clean witness: a flat deOption-3 "numeric" stack whose
// probe measures nothing, so before the fix it captured inside a 1.67 cube and
// now fits to ~[0.98, 0.94, 0.94] — a ~5.4x smaller capture volume over the
// same geometry. It must NOT be a scene, or the `scene` gate would mask the bug.
test("#457: a flat, probe-blind formula frames to the real surface, not the camera cube", () => {
  const f = PRESETS.find((p) => p.name === "BristorBrot");
  assert.ok(f, "fixture present");
  assert.ok(
    !(Array.isArray(f.objects) && f.objects.length > 0),
    "must be FLAT — a scene would reach refineFrame via the pre-existing gate",
  );
  // Precondition: this is the probe blindness the fix works around.
  assert.equal(frameFormula(f), null, "the probe measures no extent at all");

  // Go through captureSplats so the test exercises the WIRING (refineFrame was
  // gated on `scene`); asserting on refineFrame directly would pass either way.
  const r = captureSplats(f, defaultColoring(), {
    views: 4,
    res: 32,
    cap: 5_000,
    sampleHits: 1_000,
  });
  assert.ok(r && r.frame, "captured with a frame");
  const ext = r.frame.ext;
  assert.ok(
    Array.isArray(ext) && ext.length === 3,
    "frame carries per-axis ext",
  );
  // The cube fallback is cameraFrame's radius on every axis. A frame fitted to
  // the real surface is decisively smaller, so this fails on the cube.
  const cube = cameraFrame(f).radius;
  assert.ok(
    Math.max(...ext) < 0.8 * cube,
    `expected a fitted volume well inside the ${cube.toFixed(2)} camera cube, got ext=[${ext
      .map((v) => v.toFixed(3))
      .join(", ")}]`,
  );
});

// ── #507: the DE's convergence floor, and the colors that hang off it ────────
// Reported as "even with considerable density Menger colors are undefined" — a
// SuperSplat screenshot of a Menger export speckled salt-and-pepper in colA/colB
// where the render shows clean two-tone faces. The color was never the defect:
// the default coloring is mode 0 (Surface), whose signal IS the normal
// (mixT = 0.5 + 0.5·nz), and the normals were noise because the march stopped
// below the DE's convergence floor. See the deConvergenceFloor header.

// A DE that CANNOT descend below `floor` — the loose-analytic-IFS failure mode,
// synthesized so the probe is pinned against a known answer rather than a
// preset's incidental numbers. An exact sphere SDF outside; from `floor` above
// the surface inward it is pinned AT `floor`, which is the shape that matters:
// the real Menger's DE doesn't merely blur its zero crossing, it reads ~5e-3
// throughout the solid, so a ray that enters crawls one floor-sized step at a
// time until its budget is gone. (An earlier draft floored only a thin ±band —
// a ray crosses that in two steps and never sticks, which is exactly the
// distinction this models.)
const floorySphere = (r, floor) => (x, y, z) => {
  const d = Math.hypot(x, y, z) - r; // signed
  return d > floor ? d : floor;
};
const epsTestFrame = (radius = 1) => ({
  center: [0, 0, 0],
  ext: [radius, radius, radius],
  radius,
  diag: 2 * Math.hypot(radius, radius, radius),
});

test("#507 deConvergenceFloor: measures a floored DE, reads 0 for one that converges", () => {
  const frame = epsTestFrame(1);
  const FLOOR = 5e-3;
  const loose = deConvergenceFloor(floorySphere(0.6, FLOOR), frame);
  assert.ok(loose.rays > 1000, `probe cast rays (${loose.rays})`);
  // A ray hugging the surface can never close the last `floor`, so the |DE| it
  // dies at IS the floor.
  assert.ok(
    Math.abs(loose.floor - FLOOR) <= 0.25 * FLOOR,
    `measured floor ${loose.floor} ≈ ${FLOOR}`,
  );
  assert.ok(
    loose.stuck >= EPS_FLOOR_MIN_RAYS * loose.rays,
    `the loose DE sticks on a real population (${loose.stuck}/${loose.rays})`,
  );

  // The same sphere with no floor converges: no stuck rays, floor 0.
  const tight = deConvergenceFloor(
    (x, y, z) => Math.hypot(x, y, z) - 0.6,
    frame,
  );
  assert.equal(tight.floor, 0, "a converging DE has no floor");
  assert.ok(
    tight.stuck < EPS_FLOOR_MIN_RAYS * tight.rays,
    `converging DE barely sticks (${tight.stuck}/${tight.rays})`,
  );
});

test("#507 withCaptureEps: raises eps for a floored DE, IDENTITY for a converging one", () => {
  const frame = epsTestFrame(1);
  const FLOOR = 5e-3;
  const raised = withCaptureEps(floorySphere(0.6, FLOOR), frame);
  assert.notEqual(raised, frame, "a floored DE gets a new frame");
  assert.ok(
    captureEps(raised) >= EPS_FLOOR_FACTOR * 0.75 * FLOOR,
    `eps ${captureEps(frame).toExponential(2)} → ${captureEps(raised).toExponential(2)} clears the floor`,
  );
  // The march tolerance is the ONLY thing that moves. The captured VOLUME is
  // #518/#522's business and this fix must not touch it.
  for (const k of ["center", "ext", "radius", "diag"])
    assert.deepEqual(raised[k], frame[k], `${k} unchanged`);

  // The identity guarantee — what keeps the rest of the preset library
  // byte-identical (measured: Mandelbulb sticks on 0.0% of rays, Carved Cube
  // 0.5%, both far under the 2% bar; their S2 metric pins are untouched).
  const same = withCaptureEps((x, y, z) => Math.hypot(x, y, z) - 0.6, frame);
  assert.equal(same, frame, "a converging DE gets the identical frame back");
});

test("#507 captureEps: one rule for every tier, and frame.epsMeasured raises it", () => {
  const frame = epsTestFrame(2);
  assert.equal(captureEps(frame), 3e-4 * 2);
  assert.equal(captureEps({ ...frame, radius: 1e-9 }), 1e-5, "1e-5 hard floor");
  assert.equal(
    captureEps({ ...frame, epsMeasured: 0.07 }),
    0.07,
    "the stamp wins",
  );
});

// #496 + #507 RECONCILED. The two issues each added a lower bound to the same
// epsilon, independently; this pins that they compose as a MAXIMUM, so eps is
// only ever raised and neither mechanism can undercut the other. The failure
// mode being excluded is the one an override (`frame.eps ?? …`) would allow: a
// measured stamp silently discarding a framing floor that is larger.
test("captureEps #496+#507: the floors compose as a max — eps is only ever RAISED", () => {
  const frame = epsTestFrame(2); // scale-relative term = 6e-4
  const rel = 3e-4 * 2;
  // Each floor alone raises; the larger of the two wins when both are present.
  assert.equal(captureEps({ ...frame, epsFloor: 5e-3 }), 5e-3, "framing alone");
  assert.equal(
    captureEps({ ...frame, epsMeasured: 9e-3 }),
    9e-3,
    "measured alone",
  );
  assert.equal(
    captureEps({ ...frame, epsFloor: 5e-3, epsMeasured: 9e-3 }),
    9e-3,
    "both present ⇒ the binding (larger) floor wins",
  );
  assert.equal(
    captureEps({ ...frame, epsFloor: 9e-3, epsMeasured: 5e-3 }),
    9e-3,
    "…in either order — a measured stamp cannot discard a bigger framing floor",
  );
  // Neither floor can ever LOWER the eps below the scale-relative rule.
  assert.equal(
    captureEps({ ...frame, epsFloor: 1e-9, epsMeasured: 1e-9 }),
    rel,
    "floors below the scale-relative term are inert, never a reduction",
  );
});

// THE #496 × #507 RECONCILIATION PIN — the case that trips BOTH mechanisms at
// once, and the one that silently broke when they merged.
//
// objectEpsFloor's contract is "the eps the WHOLE-OBJECT capture would actually
// march with", which a sub-region inherits so it is never resolved worse than
// the capture the user compares it against. #496 implemented that as
// `3e-4 · objectRadius` — correct at the time. #507 then raised the object's own
// eps to 3× its measured convergence floor whenever the DE bottoms out, and the
// hand-typed term silently stopped being the object's eps: measured on the
// Menger, the object marches at 1.41e-2 while objectEpsFloor still said 3.3e-4,
// 43× tighter. A 1/8.7 corner crop then came back with 365 splats and 30.7%
// axis-aligned normals — #496's starvation AND #507's confetti, together, in the
// exact case each fix was written for. Deriving the floor through
// withCaptureEps → captureEps is what keeps the two in step.
test("#496×#507: objectEpsFloor carries the MEASURED floor, so a crop inherits the object's real eps", () => {
  const FLOOR = 0.004;
  const de = floorySphere(0.6, FLOOR);
  const objFrame = epsTestFrame(1); // relative term = 3e-4, far below the floor
  const stub = { name: "floored", ops: [], iters: 4 };
  const rel = 3e-4 * objFrame.radius;
  assert.ok(
    rel < FLOOR,
    "premise: the object's scale-relative eps is below it",
  );

  const floor = objectEpsFloor(stub, { frame: objFrame, de, deScale: 1 });
  // The object's TRUE marching eps — what withCaptureEps stamps, not the
  // scale-relative term the pre-reconciliation code returned.
  assert.equal(
    floor,
    captureEps(withCaptureEps(de, objFrame, { deScale: 1 })),
    "objectEpsFloor === the eps the whole-object capture actually uses",
  );
  assert.ok(
    floor > rel,
    `a floored DE lifts it above the scale-relative term (${floor.toExponential(2)} > ${rel.toExponential(2)})`,
  );
  assert.ok(
    floor >= EPS_FLOOR_FACTOR * 0.75 * FLOOR,
    `and it clears the DE's own floor (${floor.toExponential(2)} vs ${FLOOR})`,
  );

  // Handed to a crop 1/8 the object's size, the floor still dominates — that is
  // the whole point: the crop marches at the object's eps, not at 1/8 of it.
  const crop = epsTestFrame(0.125);
  assert.equal(
    captureEps({ ...crop, epsFloor: floor }),
    floor,
    "the crop inherits the object's eps rather than its own 8× tighter one",
  );
  // A converging DE is untouched — the common case stays byte-identical.
  const conv = objectEpsFloor(stub, {
    frame: objFrame,
    de: (x, y, z) => Math.hypot(x, y, z) - 0.6,
    deScale: 1,
  });
  assert.equal(conv, rel, "a DE that converges ⇒ the plain scale-relative eps");
});

// THE #507 REGRESSION PIN. The Menger's DE floors at ~5e-3 on a flat face, so
// the old fixed eps (3e-4·radius = 3.3e-4) could only ever stop at the sparse
// creases where the DE genuinely reaches 0 — and the ∇DE there reads the floor's
// own noise, not the surface. Measured before the fix: normals UNIFORM over
// nz ∈ [−1,1] (~25% axis-aligned) on a sponge whose every face is axis-aligned,
// i.e. the mode-0 albedo was a random lerp of colA↔colB. Note this is NOT the
// #351/#496 "too few hits" case those nets already cover — the old capture found
// plenty of hits (11.5% of rays); every one of them was simply unusable.
test("#507 Menger export: axis-aligned normals ⇒ a Surface color that isn't confetti", () => {
  const f = PRESETS.find((p) => p.name === "Menger");
  assert.ok(f, "Menger preset present");
  const coloring = defaultColoring();
  assert.equal(coloring.mode, 0, "the default look is Surface — color IS nz");
  const r = captureSplats(f, coloring, {
    views: 8,
    res: 48,
    cap: 60_000,
    layers: 1,
    aoStrength: 0,
  });
  assert.ok(r && r.points.count > 0, "captured something");
  assert.ok(
    captureEps(r.frame) > 3e-3,
    `captured above the ~5e-3 floor (eps ${captureEps(r.frame).toExponential(2)})`,
  );

  const n = r.points.normal;
  let axis = 0;
  for (let i = 0; i < r.points.count; i++) {
    const j = 3 * i;
    if (Math.max(Math.abs(n[j]), Math.abs(n[j + 1]), Math.abs(n[j + 2])) > 0.9)
      axis++;
  }
  const frac = axis / r.points.count;
  assert.ok(
    frac >= 0.5,
    `a sponge's faces are axis-aligned — got ${(100 * frac).toFixed(1)}% (pre-fix ~25%)`,
  );

  // …and therefore the color. Every channel finite and in gamut, the cloud not
  // flat, but the values CONCENTRATED — the render shows a few face tones, not a
  // continuum — so the top 16 of 512 bins must hold most of the cloud.
  const a = r.points.albedo;
  const bins = new Map();
  for (let i = 0; i < r.points.count; i++) {
    const j = 3 * i;
    for (let c = 0; c < 3; c++)
      assert.ok(
        Number.isFinite(a[j + c]) && a[j + c] >= 0 && a[j + c] <= 1,
        `albedo[${j + c}] = ${a[j + c]} is a finite [0,1] channel`,
      );
    const k =
      (Math.min(7, (a[j] * 8) | 0) << 6) |
      (Math.min(7, (a[j + 1] * 8) | 0) << 3) |
      Math.min(7, (a[j + 2] * 8) | 0);
    bins.set(k, (bins.get(k) || 0) + 1);
  }
  assert.ok(bins.size > 1, "not a single flat color");
  const head =
    [...bins.values()]
      .sort((x, y) => y - x)
      .slice(0, 16)
      .reduce((s, v) => s + v, 0) / r.points.count;
  assert.ok(
    head >= 0.75,
    `color concentrates on a few face tones — top-16 bins hold ${(100 * head).toFixed(1)}% (pre-fix 62%)`,
  );
});

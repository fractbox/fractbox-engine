// §S2 GPU-capture scaffold (PR-1): Node-testable pieces of the WebGPU capture
// path — no GPU here, so this pins the SHAPES the GPU compile depends on (the
// shared ray basis, the capture uniform layout, and the extracted surfaceAlbedo
// WGSL twin). GPU render correctness is the manual Mac gate.
// Run: node --test core/splatgpu.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { buildWGSL, packCaptureUniform, CAPTURE_U_WORDS } from "./shader.js";
import { viewBasis, fibonacciDir } from "./splatcapture.js";

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const len = (a) => Math.hypot(a[0], a[1], a[2]);

test("viewBasis: orthonormal basis, plane origin behind center, stable at d∥Z", () => {
  const frame = { center: [2, -1, 0.5], radius: 1.5 };
  for (let k = 0; k < 12; k++) {
    const { d, rgt, up, oc } = viewBasis(fibonacciDir(k, 12), frame);
    assert.ok(Math.abs(len(d) - 1) < 1e-9, "d unit");
    assert.ok(Math.abs(len(rgt) - 1) < 1e-9, "rgt unit");
    assert.ok(Math.abs(len(up) - 1) < 1e-9, "up unit");
    assert.ok(Math.abs(dot(d, rgt)) < 1e-9, "rgt ⊥ d");
    assert.ok(Math.abs(dot(d, up)) < 1e-9, "up ⊥ d");
    assert.ok(Math.abs(dot(rgt, up)) < 1e-9, "rgt ⊥ up");
    // oc = center − d·1.5·hd, where hd is the volume's support along d
    // (CAPTURE_VOLUME_SHAPES). This frame carries no `ext`, so the volume is
    // the uniform fallback ±radius and hd is its L1 support — which is ≥ radius
    // and equals it only for an axis-aligned view. It is NOT 1.5·radius.
    const { hu, hv, hd } = viewBasis(fibonacciDir(k, 12), frame);
    const l1 =
      frame.radius * (Math.abs(d[0]) + Math.abs(d[1]) + Math.abs(d[2]));
    assert.ok(Math.abs(hd - l1) < 1e-9, "hd = L1 support along d");
    assert.ok(
      hd >= frame.radius - 1e-9,
      "support never shrinks below the half-extent",
    );
    for (let c = 0; c < 3; c++)
      assert.ok(
        Math.abs(oc[c] - (frame.center[c] - d[c] * 1.5 * hd)) < 1e-9,
        `oc[${c}]`,
      );
    // the window is sized the same way, off rgt/up
    assert.ok(hu > 0 && hv > 0, "window half-extents positive");
  }
  // d ∥ +Z → cross(d,+Z) degenerate → stable right = [1,0,0]
  const zUp = viewBasis([0, 0, 1], { center: [0, 0, 0], radius: 1 });
  assert.deepEqual([...zUp.rgt], [1, 0, 0], "degenerate d∥Z → rgt = [1,0,0]");
});

test("packCaptureUniform: exact word layout the CaptureU struct reads", () => {
  const u = packCaptureUniform({
    d: [0.1, 0.2, 0.3],
    rgt: [0.4, 0.5, 0.6],
    up: [0.7, 0.8, 0.9],
    originRel: [1.1, 1.2, 1.3],
    radius: 1.5,
    eps: 0.001,
    tmax: 4.5,
    ext: [2, 3, 4],
    kind: 0,
    hu: 2.25,
    hv: 3.25,
    layerIndex: 1,
    deScale: 0.3,
    aoStrength: 0.5,
    maxSteps: 200,
    layers: 2,
  });
  assert.equal(CAPTURE_U_WORDS, 36);
  assert.equal(u.length, 36);
  assert.equal(u instanceof Float32Array, true);
  const f = (x) => Math.fround(x); // stored as f32
  // viewDir: d.xyz, layerIndex
  assert.deepEqual([u[0], u[1], u[2], u[3]], [f(0.1), f(0.2), f(0.3), 1]);
  // right: rgt.xyz, radius
  assert.deepEqual([u[4], u[5], u[6], u[7]], [f(0.4), f(0.5), f(0.6), f(1.5)]);
  // up: up.xyz, eps
  assert.deepEqual(
    [u[8], u[9], u[10], u[11]],
    [f(0.7), f(0.8), f(0.9), f(0.001)],
  );
  // origin: residual, tmax
  assert.deepEqual(
    [u[12], u[13], u[14], u[15]],
    [f(1.1), f(1.2), f(1.3), f(4.5)],
  );
  // knobs: deScale, aoStrength, maxSteps, layers
  assert.deepEqual([u[16], u[17], u[18], u[19]], [f(0.3), f(0.5), 200, 2]);
  // vol: capture volume ext.xyz, kind  (CAPTURE_VOLUME_SHAPES layout bump)
  assert.deepEqual([u[20], u[21], u[22], u[23]], [2, 3, 4, 0]);
  // win: this view's support window hu/hv, then two spare words
  assert.deepEqual([u[24], u[25], u[26], u[27]], [f(2.25), f(3.25), 0, 0]);
  // rot0/rot1: unoriented volume ⇒ all zero, and rot0.w = 0 is what tells the
  // fragment to skip the transform. Zeros here must mean IDENTITY, never a
  // degenerate basis that would collapse the volume.
  assert.deepEqual([u[28], u[29], u[30], u[31]], [0, 0, 0, 0]);
  assert.deepEqual([u[32], u[33], u[34], u[35]], [0, 0, 0, 0]);
});

// An ORIENTED volume: the two local axes ride in rot0/rot1 with rot0.w = 1, and
// the fragment crosses them for the third. Pinned because a silent layout drift
// here rotates every captured volume without any other test noticing.
test("packCaptureUniform: an oriented volume packs its basis (CAPTURE_VOLUME_SHAPES)", () => {
  const u = packCaptureUniform({
    d: [1, 0, 0],
    rgt: [0, 1, 0],
    up: [0, 0, 1],
    originRel: [0, 0, 0],
    radius: 1,
    eps: 0.001,
    tmax: 3,
    ext: [1, 1, 3],
    kind: 2,
    hu: 1,
    hv: 1,
    rot: [
      [0, 1, 0],
      [0, 0, 1],
    ],
  });
  assert.deepEqual([u[28], u[29], u[30], u[31]], [0, 1, 0, 1]);
  assert.deepEqual([u[32], u[33], u[34], u[35]], [0, 0, 1, 0]);
});

// A frame that predates cuboid support (or a hand-built {center,radius} probe)
// carries no ext/hu/hv — the pack must fall back to the uniform volume rather
// than writing zeros, which would make `inside` reject every hit and capture
// an empty cloud.
test("packCaptureUniform: pre-cuboid callers fall back to the uniform volume", () => {
  const u = packCaptureUniform({
    d: [1, 0, 0],
    rgt: [0, 1, 0],
    up: [0, 0, 1],
    originRel: [0, 0, 0],
    radius: 1.5,
    eps: 0.001,
    tmax: 4.5,
  });
  assert.deepEqual(
    [u[20], u[21], u[22], u[23]],
    [1.5, 1.5, 1.5, 0],
    "ext = ±radius, kind = box",
  );
  assert.deepEqual([u[24], u[25]], [1.5, 1.5], "window = radius");
});

test("packCaptureUniform: defaults for the optional knobs", () => {
  const u = packCaptureUniform({
    d: [1, 0, 0],
    rgt: [0, 1, 0],
    up: [0, 0, 1],
    originRel: [0, 0, 0],
    radius: 1,
    eps: 0.001,
    tmax: 3,
  });
  assert.deepEqual([u[3], u[16], u[17], u[18], u[19]], [0, 1, 0, 200, 2]); // layerIndex, deScale, ao, maxSteps, layers
});

test("surfaceAlbedo: extracted WGSL twin — defined, called by fs, self-sufficient", () => {
  const src = buildWGSL();
  assert.ok(
    /fn surfaceAlbedo\(p: vec3f, nrm: vec3f, t: f32\) -> vec3f/.test(src),
    "defined with 3 args",
  );
  assert.ok(
    /var albedo = surfaceAlbedo\(p, nrm, t\);/.test(src),
    "the live fs calls it",
  );
  const body = src.match(/fn surfaceAlbedo[\s\S]*?\n}/)[0];
  // recomputes mixT internally (was a live-fs local one line above the block)
  assert.ok(
    /let mixT = mixTFor\(G\.prm\.w, p, nrm\.z, t\);/.test(body),
    "recomputes mixT",
  );
  // NO lighting/linearization inside — s2l stays in the live fs after the call
  assert.ok(!/s2l\(/.test(body), "no s2l inside (pre-lighting, display-sRGB)");
  assert.ok(/return albedo;/.test(body), "returns the albedo");
  // the old inline block is gone from fs (no duplicate mixT+albedo tree)
  assert.ok(
    !/let mixT = mixTFor[\s\S]{0,40}\/\/ Albedo/.test(src),
    "inline block removed from fs",
  );
});

test("fsCapture: capture:true adds the fragment; capture:false is byte-identical live", () => {
  const live = buildWGSL();
  const cap = buildWGSL({ capture: true });
  // capture:false emits NONE of the capture surface (live shader untouched)
  for (const s of [
    "fsCapture",
    "CaptureU",
    "CaptureOut",
    "@binding(3)",
    "captureAO",
  ])
    assert.ok(!live.includes(s), `live build must not contain ${s}`);
  // capture:true adds the whole capture surface
  assert.ok(
    /@fragment fn fsCapture\(@builtin\(position\)[^)]*@location\(0\) uv: vec2f\) -> CaptureOut/.test(
      cap,
    ),
    "fsCapture entry",
  );
  assert.ok(
    /struct CaptureU/.test(cap) && /struct CaptureOut/.test(cap),
    "structs",
  );
  for (const b of [
    "@group(0) @binding(3) var<uniform> C : CaptureU",
    "@binding(4) var prevPosT",
    "@binding(5) var prevAux",
  ])
    assert.ok(cap.includes(b), `binding: ${b}`);
  // 3 MRT targets at distinct locations
  assert.ok(
    /@location\(0\) posT/.test(cap) &&
      /@location\(1\) aux/.test(cap) &&
      /@location\(2\) alb/.test(cap),
    "3 MRT targets",
  );
});

test("fsCapture: march mirrors captureView (unsigned |dd|, capture eps, peel, AO, no s2l)", () => {
  const cap = buildWGSL({ capture: true });
  const fc = cap.match(/@fragment fn fsCapture[\s\S]*?\n}\n/)[0];
  assert.ok(/abs\(dd\) < eps/.test(fc), "unsigned |dd| hit test (S1c)");
  assert.ok(
    /t = t \+ max\(abs\(dd\) \* deScale, 0\.5 \* eps\)/.test(fc),
    "unsigned sphere-trace step w/ deScale",
  );
  assert.ok(
    /budget = maxSteps \* layers/.test(fc),
    "shared layers·maxSteps budget",
  );
  assert.ok(
    /t = prev\.w \+ 3\.0 \* eps/.test(fc),
    "S1c re-arm from prev layer",
  );
  assert.ok(
    /mapDE\(p \+ vec3f\(h, 0\.0, 0\.0\)\)/.test(fc),
    "6-tap ∇DE normal",
  );
  assert.ok(/surfaceAlbedo\(p, nrm, t\) \* ao/.test(fc), "AO × surfaceAlbedo");
  assert.ok(
    /captureAO\(p, nrm, eps, radius, aoStrength\)/.test(fc),
    "S1d AO tap",
  );
  assert.ok(!/s2l\(/.test(fc), "NO s2l — display-sRGB out (view-independent)");
  assert.ok(
    /out\.posT = vec4f\(p, t\)/.test(fc),
    "posT = p_rel + t (readback adds fround(O))",
  );
  // ≥2 surfaceAlbedo call sites across the two fragments (live fs + fsCapture)
  const defsAndCalls = (cap.match(/surfaceAlbedo\(/g) || []).length;
  assert.ok(
    defsAndCalls >= 3,
    `surfaceAlbedo def + ≥2 calls (got ${defsAndCalls})`,
  );
});

test("fsCapture: the capture block is brace/paren-balanced and gate-composes", () => {
  // The whole WGSL has a fixed paren skew from comments (e.g. "(P0)"), so check
  // the DELTA the capture block adds — it must be self-balanced.
  const bal = (s, open, close) => {
    let n = 0;
    for (const ch of s) {
      if (ch === open) n++;
      else if (ch === close) n--;
    }
    return n;
  };
  const live = buildWGSL();
  const cap = buildWGSL({ capture: true });
  assert.equal(
    bal(cap, "{", "}") - bal(live, "{", "}"),
    0,
    "capture block braces balanced",
  );
  assert.equal(
    bal(cap, "(", ")") - bal(live, "(", ")"),
    0,
    "capture block parens balanced",
  );
  // a gated capture build (flat Surface: coloring/scene off) still emits fsCapture
  const flat = buildWGSL({ capture: true, coloring: false, scene: false });
  assert.ok(
    /@fragment fn fsCapture\(/.test(flat),
    "fsCapture present in gated build",
  );
  assert.equal(
    bal(flat, "{", "}") -
      bal(buildWGSL({ coloring: false, scene: false }), "{", "}"),
    0,
    "gated capture block braces balanced",
  );
});

test("surfaceAlbedo: survives the coloring/scene gate combinations (compile-safety)", () => {
  // capture builds carry the frame's own coloring/scene flags; surfaceAlbedo's
  // callees must exist (full body) or be stubbed (return 0.0) — never dangling.
  for (const flags of [
    { coloring: true, scene: true },
    { coloring: true, scene: false },
    { coloring: false, scene: false },
  ]) {
    const src = buildWGSL(flags);
    assert.ok(
      /fn surfaceAlbedo\(/.test(src),
      `surfaceAlbedo present (${JSON.stringify(flags)})`,
    );
    // its always-emitted callees are declared regardless of gates
    for (const fn of [
      "fn mixTFor(",
      "fn normSig(",
      "fn albedoFor(",
      "fn sceneTint(",
      "fn curvatureAt(",
    ])
      assert.ok(src.includes(fn), `${fn} present (${JSON.stringify(flags)})`);
  }
});

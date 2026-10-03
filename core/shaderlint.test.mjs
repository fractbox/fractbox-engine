// Reserved-identifier lint for the GENERATED GLSL (issue #206). GLSL ES
// reserves every identifier starting with `gl_` (and any containing `__`);
// ANGLE rejects the whole shader at compile time. CI has no GL context, so a
// reserved name in any op snippet or the shared shade lib blacks out the
// entire WebGL2 tier without failing a single gate — exactly what happened
// when the P1 GGX helper named a local `gl_` (black stage on every
// non-WebGPU browser, all the way to live, for 9 days). This test compiles
// nothing: it regex-scans the emitted source, which is enough to catch the
// whole class. Runs the flat builder over EVERY registered operator (each
// op's glsl snippet is spliced into the output), a hybrid build, and the
// CSG scene builder.
//
// Run: node --test core/shaderlint.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";
import { OPERATORS } from "./operators.js";
import { buildFragGL, buildSceneFragGL, iterBodyGL } from "./shader_gl.js";
import { buildWGSL } from "./shader.js";

// The only gl_* names user code may legitimately reference in a fragment
// shader (the vertex builtins live in the internal vertex source, which is
// not under test here — op snippets and the shade lib are fragment-side).
const ALLOWED = new Set(["gl_FragCoord", "gl_FragDepth"]);

function assertNoReserved(src, label) {
  for (const m of src.matchAll(/\bgl_[A-Za-z0-9_]*/g)) {
    assert.ok(
      ALLOWED.has(m[0]),
      `${label}: reserved GLSL identifier "${m[0]}" in generated source ` +
        `(gl_ is a reserved prefix — ANGLE rejects the whole shader; see #206)`,
    );
  }
  const dunder = src.match(/\b[A-Za-z0-9_]*__[A-Za-z0-9_]*/);
  assert.equal(
    dunder,
    null,
    `${label}: identifier "${dunder?.[0]}" contains "__" — reserved in GLSL ES`,
  );
}

// One value per DECLARED param, not a fixed 3 — codegen walks def.params (so a
// >3-param op emits its full uP[] run either way), but a stub that lies about
// arity invites the next reader to assume 3 is still the ceiling.
const allOps = OPERATORS.map((o) => ({
  key: o.key,
  values: o.params.map(() => 0),
}));

test("flat build over every operator emits no reserved GLSL identifiers", () => {
  assertNoReserved(buildFragGL(allOps), "buildFragGL(all ops)");
});

test("hybrid build emits no reserved GLSL identifiers", () => {
  const half = Math.ceil(allOps.length / 2);
  assertNoReserved(
    buildFragGL(allOps.slice(0, half), {
      ops: allOps.slice(half),
      addC: false,
    }),
    "buildFragGL(hybrid)",
  );
});

test("#218: numeric-DE FD gradient reaches the op switch via ONE orbitR call site", () => {
  const numeric = buildWGSL({ numericDE: true });
  const analytic = buildWGSL({ numericDE: false });
  const switches = (src) => (src.match(/switch op\.opType/g) || []).length;
  // orbitR costs exactly ONE extra op-switch copy over the analytic build.
  assert.equal(
    switches(numeric),
    switches(analytic) + 1,
    "numeric variant must add exactly one op-switch copy (orbitR)",
  );
  // The FD gradient walks its 4 probes through a single call site (decl + 1
  // call). Four call sites let the shader backend inline the op switch 4×,
  // which pushed the numeric pipeline compile to ~9 s (#218).
  assert.equal(
    (numeric.match(/\borbitR\(/g) || []).length,
    2,
    "orbitR must appear exactly twice: its declaration + one call site",
  );
  // The probe loop's trip count must not be a foldable literal, or the
  // backend unrolls it and re-inlines the switch 4× anyway.
  assert.match(numeric, /for \(var k: u32 = 0u; k < nProbe;/);
  assert.match(numeric, /let nProbe = 3u \+ u32\(G\.prm\.x > 0\.0\);/);
  // The analytic variant carries no numeric machinery at all (its 7-30%
  // register-pressure tax is why the variant split exists).
  assert.equal((analytic.match(/orbitR/g) || []).length, 0);
});

test("#370: debug surface-quality overlay is present, gated on p3ctl.z, and helper-backed", () => {
  const wgsl = buildWGSL({});
  // The overlay reads the spare Globals word p3ctl.z (0 = off ⇒ shading path
  // byte-identical), so it adds NO pipeline variant / codegen change.
  assert.match(wgsl, /let dbg = G\.p3ctl\.z;/, "overlay gated on p3ctl.z");
  assert.match(wgsl, /if \(dbg > 0\.5\)/, "overlay skipped entirely when off");
  // All three metrics are wired: step-count heat, overshoot, ∇DE instability.
  assert.match(
    wgsl,
    /f32\(steps\) \/ f32\(maxSteps\)/,
    "mode 1: step-count heat",
  );
  assert.match(wgsl, /overshoot \/ 32\.0/, "mode 2: overshoot metric");
  assert.match(wgsl, /deGradMag\(p, t\)/, "mode 3: ∇DE instability");
  // The overshoot term is captured in the march loop's bisection block.
  assert.match(
    wgsl,
    /overshoot = \(t - tPrev\)/,
    "overshoot captured pre-bisection",
  );
  // Helper functions exist (heat ramp + unnormalized gradient magnitude).
  assert.match(wgsl, /fn heatPalette\(/, "heat palette helper present");
  assert.match(wgsl, /fn deGradMag\(/, "gradient-magnitude helper present");
  // Off-by-default is enforced by the branch guard — no unconditional heat return.
  const heatReturns = (wgsl.match(/return vec4f\(s2l\(heatPalette/g) || [])
    .length;
  assert.equal(
    heatReturns,
    1,
    "exactly one heat return, inside the dbg branch",
  );
});

test("df64 flag emits the double-float library, and ONLY under the flag", () => {
  const df = buildWGSL({ df64: true });
  const plain = buildWGSL({});
  const off = buildWGSL({ df64: false });
  // The library is present under the flag…
  for (const fn of [
    "fn df_launder(", // the fast-math barrier — REQUIRED, see DF64_LIB_WGSL
    "fn two_sum(",
    "fn two_prod(",
    "fn df_add(",
    "fn df_mul(",
    "fn df_abs(",
    "fn df3_two_sum(",
    "fn df3_dot(",
    "struct Df3",
  ]) {
    assert.ok(df.includes(fn), `df64 build must contain "${fn}"`);
  }
  // …and completely absent otherwise (the default build stays byte-identical
  // — the same discipline as the numericDE / capture blocks).
  assert.equal(plain, off, "df64:false must equal the default build");
  assert.ok(
    !plain.includes("two_sum"),
    "default build must carry no df64 text",
  );
  assert.ok(!plain.includes("Df3"), "default build must carry no df64 text");
  // PR-3: the df64 variant wraps EXACTLY the 8 flat sites (mapDE_single + the
  // 7 coloring loops) with a second, twin-only switch — morph/hybrid/scene
  // loops stay single-switch (D1 flat-only gate).
  const switches = (src) => (src.match(/switch op\.opType/g) || []).length;
  assert.equal(switches(df), switches(plain) + 8);
  assert.equal((df.match(/df64 reconstruction/g) || []).length, 8);
  // the D2 essentials: kStar uniform read, the f32 mirror refresh, the
  // non-Julia addC fed from the immutable df64 p0 (TOURBILLON fix), and the
  // fast-math barrier armed from the always-zero offset.w at BOTH entries
  assert.match(df, /let kStar_ = u32\(G\.offsetLo\.w\);/);
  assert.match(df, /pos = P\.hi \+ P\.lo;/);
  assert.match(df, /P = df3_add\(P, Df3\(P0h, P0l\)\)/);
  assert.match(df, /df_lz = bitcast<u32>\(G\.offset\.w\)/);
  assert.match(df, /fn mapDE_single\(p_rel: vec3f\)/);
  assert.match(plain, /fn mapDE_single\(p0: vec3f\)/);
  // capture build arms the barrier in fsCapture too
  const dfCap = buildWGSL({ df64: true, capture: true });
  assert.equal(
    (dfCap.match(/df_lz = bitcast<u32>\(G\.offset\.w\)/g) || []).length,
    2,
  );
  // twin coverage: every wgslDf op has a case in the df64 switch region
  const twinned = OPERATORS.filter((o) => o.wgslDf);
  assert.ok(twinned.length >= 12, "expected the 12-key subset");
  for (const op of twinned)
    assert.ok(
      df.includes(`case ${op.id}u: {${op.wgslDf}`),
      `twin case for ${op.key}`,
    );
});

test("Globals carries offsetLo (deep zoom P4) in EVERY build", () => {
  // The struct is layout, not feature text: the field must exist in all
  // variants (the buffer is one size), while remaining unread outside the
  // df64 variant so zero-filled words render byte-identically (PR-2 is
  // inert by construction — the reconstruction still reads only G.offset).
  for (const opts of [{}, { df64: true }, { numericDE: false }]) {
    const src = buildWGSL(opts);
    assert.match(src, /offsetLo: vec4f/);
  }
  const plain = buildWGSL({});
  assert.ok(
    !/G\.offsetLo/.test(plain),
    "no non-df64 shader text may read offsetLo (PR-2 inertness)",
  );
});

test("scene build emits no reserved GLSL identifiers", () => {
  const obj = (ops, objType = 0) => ({
    objType,
    ops,
    iters: 8,
    transform: { origin: [0, 0, 0], uscale: 1, rot: [0, 0, 0] },
    combine: 0,
    blendK: 0,
  });
  assertNoReserved(
    buildSceneFragGL([obj(allOps.slice(0, 8)), obj([], 1)]),
    "buildSceneFragGL",
  );
});

test("perturb flag emits the delta variant, and ONLY under the flag (PR-2)", () => {
  const pt = buildWGSL({ perturb: true });
  const plain = buildWGSL({});
  const off = buildWGSL({ perturb: false });
  // The pt machinery is present under the flag…
  for (const s of [
    "var<storage, read> ptRecs",
    "fn pt_trailer(",
    "fn pt_escapeAt(",
    "fn pt_final(",
    "const PT_TAU2",
    "fn mapDE_single(p_rel", // review M2: the residual IS the seed
  ]) {
    assert.ok(pt.includes(s), `perturb build must contain "${s}"`);
  }
  // …the twin switch actually carries delta cases (spot-check two families)…
  assert.ok(
    pt.includes("ptD = kr * ptD + dk *"),
    "rational-radial delta cases present",
  );
  // …and it is completely absent otherwise (inert-by-default, the df64
  // discipline; the WGSL byte-identity gate proper is run against the dev
  // tip in the PR, this pin keeps it from silently regressing after).
  assert.equal(plain, off, "perturb:false must equal the default build");
  assert.ok(!plain.includes("ptRecs"), "default build must carry no pt text");
  assert.ok(!plain.includes("ptTrk"), "default build must carry no pt text");
  // the tiers are exclusive; perturb+capture is legal since PR-3 (ptRecs
  // moved to binding 6, clear of the capture block's 3-5)
  assert.throws(() => buildWGSL({ perturb: true, df64: true }), /exclusive/);
  assert.doesNotThrow(() => buildWGSL({ perturb: true, capture: true }));
});

test("perturb PR-3: the 7 orbit-signal sites carry the delta loop", () => {
  const pt = buildWGSL({ perturb: true });
  // one delta walk per site + mapDE_single = 8 (the df64 D2 site set)
  assert.equal(
    (pt.match(/ptRi = ptRi \+ 1u/g) || []).length,
    8,
    "8 pt loop sites",
  );
  // the per-iteration pos re-sync feeds every signal read (7 sites; mapDE
  // reconstructs at bailout/finalize instead)
  assert.equal(
    (pt.match(/pos = ptRecs\[ptRi \* 4u\]\.xyz \+ ptD/g) || []).length,
    7,
    "7 signal-site pos syncs",
  );
  // capture parity (D9): perturb+capture now emits BOTH blocks, collision-free
  const cap = buildWGSL({ perturb: true, capture: true });
  assert.ok(
    cap.includes("binding(6) var<storage, read> ptRecs"),
    "ptRecs at binding 6",
  );
  assert.ok(cap.includes("binding(3) var<uniform> C"), "capture block intact");
});

// ── #553: kaleido Mirror wired into the REAL WebGL2 pipeline ─────────────────
// core/opmath.test.mjs pins the bug at the operators.js template-function
// level; this pins it at the level buildFragGL/renderer_gl actually run —
// the shipped op body must contain a genuine runtime conditional on Mirror's
// own uP[] slot, not just "some string that happens to include p2".
test("#553: kaleido Mirror emits a real uP[] runtime conditional via the shipped pipeline", () => {
  const { body } = iterBodyGL([{ key: "kaleido", values: [6, 0, 1] }]);
  assert.match(
    body,
    /if \(uP\[2\] > 0\.5\) \{ ang = abs\(ang\); \}/,
    "iterBodyGL must wire kaleido's Mirror param (slot 2) to a real GLSL branch",
  );
});

// ── #553 class-fence ──────────────────────────────────────────────────────────
// The root cause wasn't kaleido-specific: `glsl: (v) => ...` templates get a v
// of GLSL variable-name STRINGS ("uP[2]", "p2", …), never numbers, so any
// template that resolves a branch by comparing v[i] in JAVASCRIPT (inside its
// OWN `${...}` interpolation) rather than embedding the comparison as GLSL
// TEXT for the shader to evaluate at runtime is the same class of dead code —
// always resolves to one branch, silently. The correct idiom, used by every
// other conditional op body in this registry (zFold's plane bound, modFold's
// per-axis wrap, tentFold's per-axis fold, polyAngleFold's own Mirror,
// absXYZ's per-axis abs, msltoeSym3's sign flip), keeps the comparison
// OUTSIDE the `${...}` — e.g. `if (${v[2]} > 0.5) ang = abs(ang);` — so
// `v[2]` is just textually spliced into real GLSL and the branch is
// evaluated on the GPU against whatever value the uniform actually holds.
//
// A `${...}` block containing a `<`/`>` comparison is exactly the
// tell — the fixed value never appears in this registry (verified: exactly
// one match, at kaleido, before this fix).
test("#553 class-fence: no op's glsl template resolves a branch by comparing v[i] in JavaScript", () => {
  const offenders = [];
  for (const op of OPERATORS) {
    if (typeof op.glsl !== "function") continue;
    const src = op.glsl.toString();
    for (const m of src.matchAll(/\$\{[^}]*[<>][^}]*\}/g)) {
      offenders.push(`${op.key}: ${m[0]}`);
    }
  }
  assert.deepEqual(
    offenders,
    [],
    `op(s) resolve a GLSL branch by comparing a param reference in JS, not in ` +
      `the emitted GLSL (the exact #553 kaleido-Mirror bug class):\n` +
      offenders.join("\n"),
  );
});

// Self-reflection (#630, Catoptron parity) — marched mirror bounces.
//
// The load-bearing invariant is the same one ENVX pins: the feature is
// CODEGEN-gated, not uniform-gated. A look with reflBounces 0 (every existing
// save) must emit shader text with NOT ONE bounce token — that byte-identity
// is the perf doctrine's "prove it's free" standard (the #125 lesson: a
// never-executing uniform branch still cost Mandelbulb +31%). These tests
// fence the gate, the spec-review contracts baked into the on-variant text
// (offset reuse, t continuation, degenerate-normal refusal, precision-tier
// eligibility), the shared derivation, the cheap-tier word flip, the variant
// key, and sanitize.
//
// Run: node --test core/sreflect.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { buildWGSL } from "./shader.js";
import { deriveFrameParams } from "./frameparams.js";
import { shadeLight } from "./renderpolicy.js";
import { frameFeaturesFor } from "./capturesettle.js";
import { keyFor, wgslOf } from "./renderer.js";
import { sanitizeColoring } from "./sanitize.js";
import { defaultColoring } from "./coloring.js";

// Every token the sreflect splices introduce — none may appear when off.
const SR_TOKENS = [
  "accC",
  "thruR",
  "bounceN",
  "bnc",
  "srKR",
  "srThruN",
  "srMore",
  "srDegen",
  "outA",
];

// ── The gate: off emits nothing, on emits everything ─────────────────────────
test("WGSL: default build carries no self-reflection token (the free-when-off fence)", () => {
  const src = buildWGSL();
  for (const tok of SR_TOKENS)
    assert.ok(!src.includes(tok), `default WGSL leaked ${tok}`);
  assert.equal(
    buildWGSL(),
    buildWGSL({ sreflect: false }),
    "explicit false must equal the default build byte-for-byte",
  );
});

test("WGSL: sreflect build carries the full bounce loop", () => {
  for (const opts of [
    { sreflect: true },
    { sreflect: true, ops: [1, 2] },
    { sreflect: true, scene: true, leaves: [1] },
    { sreflect: true, hybrid: true },
    { sreflect: true, envx: true },
  ]) {
    const src = buildWGSL(opts);
    for (const tok of SR_TOKENS)
      assert.ok(
        src.includes(tok),
        `sreflect WGSL (${JSON.stringify(opts)}) missing ${tok}`,
      );
  }
});

// ── The spec-review contracts, pinned in the emitted text ────────────────────
test("bounce count is a runtime word clamped 0..6 inside the on-variant", () => {
  const src = buildWGSL({ sreflect: true });
  assert.ok(src.includes("u32(clamp(G.lightC.w, 0.0, 6.0))"));
  assert.ok(src.includes("for (var bnc = 0u; bnc <= bounceN; bnc = bnc + 1u)"));
});

test("reflected ray reuses the tuned softShadow normal-lift offset (review 2c)", () => {
  // The lift expression is the shadow march's documented contract. It appears
  // exactly once in the off build (softShadow itself) and exactly twice in the
  // on build (softShadow + the bounce launch site) — same constants, no
  // reinvented epsilon.
  const LIFT = "max(G.prm.y * t * 12.0, 2e-3)";
  const liftT = "max(G.prm.y * tHit * 12.0, 2e-3)"; // softShadow's own spelling
  const off = buildWGSL();
  const on = buildWGSL({ sreflect: true });
  assert.ok(off.includes(liftT) && !off.includes(LIFT));
  assert.ok(on.includes(liftT) && on.includes(LIFT));
  // …and the launch back-projects the origin so t continues as accumulated
  // path length (review 2b): the march text itself is unchanged.
  assert.ok(on.includes("- (rd * t)"));
  assert.ok(on.includes("let p = ro + rd * t;"));
});

test("degenerate/back-facing normals refuse to bounce (review 2d/2e)", () => {
  const on = buildWGSL({ sreflect: true });
  assert.ok(on.includes("var<private> srDegen"));
  assert.ok(on.includes("srDegen = !(L > 1e-20);"));
  assert.ok(on.includes("!srDegen && (srCos > 1e-4)"));
  // The env-reflection term becomes the fallback: skipped only while a
  // marched bounce replaces it.
  assert.ok(on.includes("if (!srMore && G.env.x > 0.0 && specAmt > 0.0)"));
});

test("no new mapDE/calcNormal call sites (review 2h — the #218 lesson)", () => {
  const count = (s, needle) => s.split(needle).length - 1;
  const off = buildWGSL();
  const on = buildWGSL({ sreflect: true });
  assert.equal(count(on, "calcNormal("), count(off, "calcNormal("));
  assert.equal(count(on, "mapDE("), count(off, "mapDE("));
});

test("precision tiers are an eligibility domain: sreflect+df64/perturb throws (review 2f)", () => {
  assert.throws(() => buildWGSL({ sreflect: true, df64: true }));
  assert.throws(() => buildWGSL({ sreflect: true, perturb: true }));
});

// ── Shared derivation ────────────────────────────────────────────────────────
test("deriveFrameParams: default look derives sreflect false with inert words", () => {
  const d = deriveFrameParams({ light: defaultColoring().light });
  assert.equal(d.sreflect, false);
  assert.equal(d.reflBounces, 0);
  assert.equal(d.reflectivity, 0.8); // dormant until bounces > 0
  assert.equal(d.reflFresnel, 0.5);
  assert.equal(d.reflTint, 0);
});

test("deriveFrameParams: bounces > 0 latches sreflect on; reflectivity 0 keeps it off", () => {
  assert.equal(deriveFrameParams({ light: { reflBounces: 2 } }).sreflect, true);
  assert.equal(
    deriveFrameParams({ light: { reflBounces: 2, reflectivity: 0 } }).sreflect,
    false,
  );
  // Fractional counts round (0.4 is not a bounce), and the count clamps to 6.
  assert.equal(
    deriveFrameParams({ light: { reflBounces: 0.4 } }).sreflect,
    false,
  );
  assert.equal(
    deriveFrameParams({ light: { reflBounces: 99 } }).reflBounces,
    6,
  );
});

// ── Cheap-tier word flip (the shadows/AO precedent — review 2g option i) ─────
test("shadeLight zeroes reflBounces on cheap frames, passes settled through", () => {
  const light = { shadow: 0.5, ao: 0.55, reflBounces: 3, reflectivity: 0.8 };
  const cheap = shadeLight({ cheap: true }, light);
  assert.equal(cheap.reflBounces, 0);
  assert.equal(cheap.shadow, 0);
  const settled = shadeLight({ cheap: false }, light);
  assert.equal(settled.reflBounces, 3);
  // …and the derived latch follows: a cheap frame renders the PLAIN variant.
  assert.equal(deriveFrameParams({ light: cheap }).sreflect, false);
  assert.equal(deriveFrameParams({ light: settled }).sreflect, true);
});

// ── Frame-feature prediction (prewarm holds frames on this) ──────────────────
test("frameFeaturesFor mirrors the latch incl. the precision-tier exclusion", () => {
  const f = { ops: [{ key: "boxFold", values: [1] }], iters: 8 };
  const mirror = {
    ...defaultColoring(),
    light: { ...defaultColoring().light, reflBounces: 2 },
  };
  assert.equal(frameFeaturesFor(f, mirror, {}).sreflect, true);
  assert.equal(frameFeaturesFor(f, mirror, { df64: true }).sreflect, false);
  assert.equal(frameFeaturesFor(f, mirror, { perturb: true }).sreflect, false);
  assert.equal(frameFeaturesFor(f, defaultColoring(), {}).sreflect, false);
});

// ── Variant key ──────────────────────────────────────────────────────────────
test("keyFor: sreflect is bit 2048; existing keys are unchanged", () => {
  const feat = (o = {}) => ({
    numericDE: false,
    leaves: null,
    coloring: false,
    scene: false,
    hybrid: false,
    morph: false,
    ops: null,
    ...o,
  });
  assert.equal(keyFor(feat()), "0:*:-");
  // 512/1024 are reserved for the #631 env-map arc (renderer.js bit partition).
  assert.equal(keyFor(feat({ sreflect: true })), "2048:*:-");
  assert.equal(wgslOf(feat({ sreflect: true })).sreflect, true);
  assert.equal(wgslOf(feat()).sreflect, false);
});

// ── Sanitize (the hostile-wire fence) ────────────────────────────────────────
test("sanitize: refl fields clamp to their domains", () => {
  const out = sanitizeColoring({
    light: { reflBounces: 99, reflectivity: -2, reflFresnel: 7, reflTint: 0.5 },
  });
  assert.equal(out.light.reflBounces, 6);
  assert.equal(out.light.reflectivity, 0);
  assert.equal(out.light.reflFresnel, 1);
  assert.equal(out.light.reflTint, 0.5);
});

test("sanitize: absent refl fields stay absent (shape-preserving)", () => {
  const out = sanitizeColoring({ light: { ambient: 0.2 } });
  for (const k of ["reflBounces", "reflectivity", "reflFresnel", "reflTint"])
    assert.ok(!(k in out.light), `sanitize invented light.${k}`);
});

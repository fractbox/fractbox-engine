// #729 — meta-gate for the self-maintaining codegen feature matrix
// (core/__fixtures__/featurematrix.mjs). The fixture derives buildWGSL's flag
// list from the function's own source so every byte-identity off-sweep grows
// automatically with new flags; THIS file is what keeps that derivation
// honest — a parse that silently shrinks (or a flag that stops reaching
// codegen) fails here, not as a quietly narrower sweep in some feature gate.
//
// Run: node --test core/featurematrix.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { buildWGSL } from "./shader.js";
import { wgslFlags, wgslMatrix } from "./__fixtures__/featurematrix.mjs";

const sha = (s) => createHash("sha256").update(s).digest("hex");

test("the parse reflects buildWGSL: sentinel flags present, ops is the null lever", () => {
  const flags = wgslFlags();
  // Sentinels across the flag family's age range — the original levers, the
  // precision tiers, and the newest features. A missing sentinel means the
  // destructure moved and the parse is returning garbage.
  for (const s of [
    "numericDE",
    "leaves",
    "coloring",
    "scene",
    "hybrid",
    "morph",
    "capture",
    "df64",
    "perturb",
    "envx",
    "sreflect",
    "envMap",
    "surfTex",
    "neon",
    "aurora",
    "thinFilm",
    "planet",
    "clip",
    "clipJag",
    "equirect",
  ])
    assert.ok(flags.has(s), `flag ${s} missing from the parsed table`);
  assert.equal(flags.get("ops"), null, "ops must parse as the null lever");
  assert.equal(flags.get("numericDE"), true);
  assert.equal(flags.get("clip"), false);
});

test("every boolean flag is LIVE: flipping it (on its base) changes the emitted text", () => {
  // A flag in the signature that no longer reaches codegen would silently
  // bloat every off-sweep with a no-op entry — and more importantly would
  // mean the feature's on-variant no longer exists.
  const flags = wgslFlags();
  const BASES = { clipJag: { clip: true } }; // sub-variant flags ride a base
  for (const [k, v] of flags) {
    if (typeof v !== "boolean") continue;
    const base = BASES[k] || {};
    assert.notEqual(
      sha(buildWGSL({ ...base, [k]: !v })),
      sha(buildWGSL(base)),
      `flag ${k} flipped to ${!v} emitted identical text — dead flag, or a ` +
        "new dependent flag needs a base in featurematrix's DEPENDENT_BASES",
    );
  }
});

test("every boolean flag (and the ops lever) appears in the full matrix", () => {
  const names = new Set();
  for (const [, opts] of wgslMatrix()) for (const k of Object.keys(opts)) names.add(k);
  for (const [k, v] of wgslFlags())
    if (typeof v === "boolean")
      assert.ok(names.has(k), `flag ${k} never enters the off-sweep matrix`);
  const labels = wgslMatrix().map(([n]) => n);
  assert.ok(labels.includes("ops:[]"), "the empty op-set entry is missing");
  assert.ok(labels.includes("minimal"), "the minimal entry is missing");
});

test("every matrix entry actually BUILDS — no illegal combination ships in the sweep", () => {
  for (const [name, opts] of wgslMatrix())
    assert.ok(
      buildWGSL(opts).length > 0,
      `matrix entry ${name} failed to build`,
    );
});

test("`except` removes exactly the entries touching the excluded flags", () => {
  const full = wgslMatrix();
  const cut = wgslMatrix({ except: ["clip", "clipJag"] });
  const cutNames = new Set(cut.map(([n]) => n));
  for (const [name, opts] of full) {
    const touches = "clip" in opts || "clipJag" in opts;
    assert.equal(
      cutNames.has(name),
      !touches,
      `entry ${name}: except mishandled it`,
    );
  }
  // A gate's own off-sweep must still cover every OTHER feature.
  assert.ok(cut.length >= full.length - 3);
});

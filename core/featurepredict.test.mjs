// The pump PREDICTS the shader variant a frame will need (frameFeaturesFor)
// so it can prewarm it; the renderer KEYS the cache on the same flags
// (bitsFor). If the two disagree by even one flag, the pump warms the wrong
// variant, the draw finds no match, and activeMarch() falls through to its
// SYNCHRONOUS build — a multi-second main-thread freeze on a general shader.
//
// That is not hypothetical. `envx` was missing from the prediction, and the
// field report (2026-09-09, M-series Metal) shows exactly what it costs: a
// 14907 ms frozen tab, every slow compile in the diag ring arriving in pairs
// 256 apart — F_ENVX — with the second of each pair marked `sync`. The blocked
// thread then missed the 10 s fence watchdog twice, halving the settle scale
// to 0.25 and leaving the render soft.
//
// So the first test here is a COVERAGE test, derived from bitsFor's own source
// the way featurematrix.test.mjs derives from buildWGSL's: add a codegen flag
// to the renderer's key and forget to predict it, and this fails immediately
// rather than shipping as a freeze.
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { frameFeaturesFor } from "./capturesettle.js";

const here = dirname(fileURLToPath(import.meta.url));
const rendererSrc = readFileSync(join(here, "renderer.js"), "utf8");

/** The flags bitsFor reads — the actual cache key, straight from its source. */
function keyedFlags() {
  const m = rendererSrc.match(/const bitsFor = \(f\) =>([\s\S]*?);\n/);
  assert.ok(m, "bitsFor not found in renderer.js — this test needs updating");
  const flags = [...m[1].matchAll(/f\.([A-Za-z0-9_]+)/g)].map((x) => x[1]);
  assert.ok(flags.length > 10, `parsed too few flags (${flags.length})`);
  return [...new Set(flags)];
}

const FLAT = { name: "t", ops: [{ key: "boxFold", values: [1] }], iters: 8 };
const predict = (light = {}, over = {}) =>
  frameFeaturesFor(FLAT, { mode: "orbit", light }, over);

test("every flag the renderer KEYS on is PREDICTED by frameFeaturesFor", () => {
  const predicted = predict();
  const missing = keyedFlags().filter((f) => !(f in predicted));
  assert.deepEqual(
    missing,
    [],
    `these codegen flags key the variant cache but are never predicted, so the ` +
      `pump will prewarm the wrong shader and the draw will compile SYNCHRONOUSLY: ${missing.join(", ")}`,
  );
});

test("envx is off for a bare look — the off path must stay byte-identical", () => {
  assert.equal(predict().envx, false);
  assert.equal(predict({ stars: 0, band: 0 }).envx, false);
});

test("envx flips on each of its three sources, like deriveFrameParams", () => {
  assert.equal(predict({ stars: 0.3 }).envx, true, "stars");
  assert.equal(predict({ band: 0.5 }).envx, true, "Milky-Way band");
  assert.equal(predict({ zenith: [0.1, 0.2, 0.3] }).envx, true, "zenith tint");
});

test("envx mirrors deriveFrameParams' clamps, not a looser truthiness test", () => {
  // A negative slider clamps to 0 there, so it must read as OFF here too.
  assert.equal(predict({ stars: -1 }).envx, false, "negative stars clamp to 0");
  // zenith is array PRESENCE there (`Array.isArray(...) ? ... : null` → !!) —
  // an empty array counts, and a non-array does not.
  assert.equal(
    predict({ zenith: [] }).envx,
    true,
    "empty array is still a zenith",
  );
  assert.equal(predict({ zenith: "#fff" }).envx, false, "a non-array is not");
});

test("band tilt alone does NOT flip it — only stars, band and zenith are sources", () => {
  // bandTilt only aims the band; with band at 0 there is nothing to aim.
  assert.equal(predict({ bandTilt: 0.6 }).envx, false);
});

test("the other background latches still read independently of envx", () => {
  assert.equal(predict({ aurora: 0.4 }).envx, false, "aurora is its own bit");
  assert.equal(predict({ aurora: 0.4 }).aurora, true);
  assert.equal(predict({ stars: 0.4 }).aurora, false);
});

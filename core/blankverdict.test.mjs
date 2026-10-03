// Pins classifyBlank (blankverdict.js) — the shared blank-frame rule behind the
// thumbnail pruner and the still-export guard (the beta "save produced a black
// PNG" report: a device lost mid-export left an unpresented canvas that
// toBlob happily encoded, and the app toasted "Saved").
import { test } from "node:test";
import assert from "node:assert/strict";
import { classifyBlank } from "./blankverdict.js";

const fill = (n, rgba) => {
  const d = new Uint8ClampedArray(n * n * 4);
  for (let i = 0; i < d.length; i += 4) d.set(rgba, i);
  return d;
};

test("all-zero (never presented / dead device) is uniform + allZero + black", () => {
  const v = classifyBlank(fill(12, [0, 0, 0, 0]));
  assert.deepEqual(v, { uniform: true, allZero: true, black: true });
});

test("opaque uniform black is black but NOT allZero (alpha 255)", () => {
  const v = classifyBlank(fill(12, [0, 0, 0, 255]));
  assert.deepEqual(v, { uniform: true, allZero: false, black: true });
});

test("a flat non-black sky is uniform (thumbnail-blank) but not black", () => {
  const v = classifyBlank(fill(12, [40, 60, 90, 255]));
  assert.deepEqual(v, { uniform: true, allZero: false, black: false });
});

test("a picture with structure is none of the three", () => {
  const d = fill(12, [0, 0, 0, 255]);
  // 5 pixels well off the reference — above the <3 stray threshold
  for (let p = 0; p < 5; p++) d.set([200, 120, 30, 255], p * 4 * 13);
  const v = classifyBlank(d);
  assert.deepEqual(v, { uniform: false, allZero: false, black: false });
});

test("two stray pixels stay under the thumbnail threshold (matches the old rule)", () => {
  const d = fill(12, [0, 0, 0, 255]);
  d.set([255, 255, 255, 255], 0 + 4 * 50);
  d.set([255, 255, 255, 255], 0 + 4 * 100);
  assert.equal(classifyBlank(d).uniform, true);
});

test("empty input counts as blank", () => {
  assert.deepEqual(classifyBlank(new Uint8ClampedArray(0)), {
    uniform: true,
    allZero: true,
    black: true,
  });
});

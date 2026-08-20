// Deep zoom Phase 4 (DEEP_ZOOM_DF64.md, plan PR-2) — splitHiLo pins.
// The ONE split function writeGlobals and overrideCaptureOffset both call;
// an inconsistent hi/lo pair silently regresses deep renders/captures to the
// f32 wall, so the split's invariants are pinned here.
//
// Run: node --test core/splithilo.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";
import { splitHiLo } from "./recenter.js";

const f = Math.fround;

test("hi is exactly the f32 the old single-word path stored", () => {
  const O = [1e6 * Math.PI, -1.2345678901, 0.1];
  const { hi } = splitHiLo(O);
  for (let i = 0; i < 3; i++) assert.equal(hi[i], f(O[i]));
});

test("hi + lo reconstructs O to the df64 quantum; lo is sub-f32", () => {
  const targets = [
    [1.2345678901, -0.7853981633974483, 2.718281828459045],
    [123456.789012345, -0.0000012345678901, 1e6 * Math.PI],
    [0, 0, 0], // scenes / origin: an exact no-op split
  ];
  for (const O of targets) {
    const { hi, lo } = splitHiLo(O);
    for (let i = 0; i < 3; i++) {
      // a two-term f32 split carries ~48 bits (see df64.test.mjs — this IS
      // the df64 wall), so reconstruction is a bound, not an identity
      assert.ok(
        Math.abs(hi[i] + lo[i] - O[i]) <=
          Math.max(Math.abs(O[i]), 1e-20) * 2 ** -47,
        `component ${i} of [${O}]`,
      );
      // lo is genuinely the sub-f32 remainder, not a second copy of hi
      assert.ok(
        Math.abs(lo[i]) <= Math.abs(hi[i]) * 2 ** -23 || hi[i] === 0,
        `lo[${i}] out of range for [${O}]`,
      );
      // both halves are f32-representable (they go straight into the buffer)
      assert.equal(hi[i], f(hi[i]));
      assert.equal(lo[i], f(lo[i]));
    }
  }
});

test("f32-representable values split exactly, lo = 0", () => {
  const { hi, lo } = splitHiLo([1.5, -4096.03125, 0.25]);
  assert.deepEqual(hi, [1.5, -4096.03125, 0.25]);
  assert.deepEqual(lo, [0, 0, 0]);
});

// Zero-tooling guard for the export/share helpers (#75: pasting a new #f=…
// into an already-open tab was ignored — no hashchange listener). The fix adds
// a hashchange handler in formula-creator/src/main.js that compares
// shareHashMatch(location.hash) before/after the event; these tests pin down
// the pure extraction it depends on so that comparison stays correct.
//
// Run: node --test core/exporter.test.mjs   (*.test.mjs → sync skips it)
import assert from "node:assert/strict";
import { test } from "node:test";
import { shareHashMatch } from "./exporter.js";

test("#75: shareHashMatch reads the f= payload out of a plain share hash", () => {
  assert.equal(shareHashMatch("#f=abc123"), "abc123");
});

test("#75: shareHashMatch finds f= alongside other hash flags in either order", () => {
  assert.equal(shareHashMatch("#f=abc123&loop"), "abc123");
  assert.equal(shareHashMatch("#loop&f=abc123"), "abc123");
});

test("#75: shareHashMatch returns null when the hash has no f= (so a hashchange to a flag-only hash is a no-op, not a reload)", () => {
  assert.equal(shareHashMatch("#loop"), null);
  assert.equal(shareHashMatch(""), null);
  assert.equal(shareHashMatch(undefined), null);
});

test("#75: shareHashMatch changes value when the share payload changes, so a pasted-in-place link is detected as new", () => {
  const before = shareHashMatch("#f=oldPayload");
  const after = shareHashMatch("#f=newPayload");
  assert.notEqual(before, after);
});

test("#75: shareHashMatch is stable for an unchanged hash, so re-firing hashchange on the same link is a no-op", () => {
  const hash = "#f=samePayload";
  assert.equal(shareHashMatch(hash), shareHashMatch(hash));
});

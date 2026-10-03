// Pins loadLibrary's parse memo (library.js) — the gallery-open freeze.
// galleryItems() calls loadLibrary 2 + 2N times per open, and each call used to
// re-parse the WHOLE library string, so the cost was quadratic in library size
// (measured: 13 ms at 50 saves, 55 ms at 100, 223 ms at 200, on an M4, all of it
// blocking the click before a single card is built).
import test from "node:test";
import assert from "node:assert/strict";

// A localStorage stub, installed before the module reads it (it reads at call
// time, not import time).
const store = new Map();
globalThis.localStorage = {
  getItem: (k) => (store.has(k) ? store.get(k) : null),
  setItem: (k, v) => store.set(k, String(v)),
  removeItem: (k) => store.delete(k),
};
const { loadLibrary, saveLibrary, putFormula, deleteFormula } =
  await import("./library.js");

const K = "fractbox.lib.test";
const seed = (n) => {
  const lib = {};
  for (let i = 0; i < n; i++)
    lib[`F${i}`] = { name: `F${i}`, ops: [{ key: "boxFold", values: [1] }] };
  store.set(K, JSON.stringify(lib));
  return lib;
};

// Count real parses by wrapping JSON.parse for the duration of a call.
const countingParses = (fn) => {
  const real = JSON.parse;
  let n = 0;
  JSON.parse = (...a) => {
    n++;
    return real(...a);
  };
  try {
    fn();
  } finally {
    JSON.parse = real;
  }
  return n;
};

test("repeat reads of an unchanged library parse exactly once", () => {
  seed(20);
  loadLibrary(K); // prime
  const parses = countingParses(() => {
    for (let i = 0; i < 42; i++) loadLibrary(K); // the 2 + 2N shape at N=20
  });
  assert.equal(parses, 0, "every repeat read is served from the memo");
});

test("the first read after a change parses again (another tab's write is seen)", () => {
  seed(5);
  loadLibrary(K);
  store.set(K, JSON.stringify({ Solo: { name: "Solo", ops: [] } }));
  const got = loadLibrary(K);
  assert.deepEqual(
    Object.keys(got),
    ["Solo"],
    "the new content is read, not the memo",
  );
});

test("the caller never receives the cached object — mutating a result cannot poison it", () => {
  seed(3);
  const a = loadLibrary(K);
  a.Injected = { name: "Injected" };
  delete a.F0;
  const b = loadLibrary(K);
  assert.equal(
    b.Injected,
    undefined,
    "an added key does not leak into the next read",
  );
  assert.ok(b.F0, "a deleted key is still there on the next read");
  assert.notEqual(a, b, "each call returns its own object");
});

test("putFormula / deleteFormula still round-trip through the memo", () => {
  seed(2);
  putFormula(K, "New", { name: "New", ops: [] });
  assert.ok(loadLibrary(K).New, "a written entry is visible immediately");
  assert.ok(JSON.parse(store.get(K)).New, "…and actually reached storage");
  deleteFormula(K, "New");
  assert.equal(
    loadLibrary(K).New,
    undefined,
    "a deleted entry is gone immediately",
  );
  assert.equal(JSON.parse(store.get(K)).New, undefined);
});

test("a save invalidates rather than primes — what is stored is a SNAPSHOT", () => {
  // The whole point of storage: editing the object you saved must not rewrite
  // what was saved. Priming the memo from the caller's object broke exactly
  // this (app/test/gallery.test.ts "a mine source is a FROZEN snapshot" caught
  // it), because a saved entry is the live formula the caller goes on editing.
  seed(1);
  const live = { name: "Rug", ops: [{ key: "boxFold", values: [7] }] };
  putFormula(K, "Rug", live);
  live.ops[0].values[0] = -1; // the caller keeps editing after the save
  assert.deepEqual(
    loadLibrary(K).Rug.ops[0].values,
    [7],
    "the stored snapshot is unchanged by a later edit to the saved object",
  );
  saveLibrary(K, { Only: { name: "Only", ops: [] } });
  assert.deepEqual(
    Object.keys(loadLibrary(K)),
    ["Only"],
    "a direct save is visible immediately",
  );
});

test("missing or corrupt storage degrades to an empty library", () => {
  store.delete(K);
  assert.deepEqual(loadLibrary(K), {});
  store.set(K, "{not json");
  assert.deepEqual(loadLibrary(K), {});
});

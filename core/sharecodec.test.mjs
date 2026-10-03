// Zero-tooling guard for the compact share codec. The container/formula decoders
// read length/count varints straight off an untrusted URL payload, so their caps
// and forward-compat behavior are security-relevant. Only exercised from app/
// before now (doesn't travel with the raw-ESM engine).
//
// Run: node --test core/sharecodec.test.mjs   (*.test.mjs → sync skips it)
import assert from "node:assert/strict";
import { test } from "node:test";
import { existsSync, readFileSync } from "node:fs";
import {
  encodeFormula,
  decodeFormula,
  packContainer,
  unpackContainer,
  bytesToB64url,
  ByteWriter,
  ByteReader,
  CODEC_VERSION,
  TAG,
  MAX_DECODE_OPS,
  MAX_DECODE_PARAMS,
} from "./sharecodec.js";

const sample = {
  addC: true,
  julia: true,
  deOption: 2,
  iters: 12,
  camera: { yawDeg: 35, pitchDeg: 22, dist: 14, fovDeg: 42 },
  juliaC: [0.5, -0.25, 0.1],
  ops: [
    { key: "boxFold", values: [1] },
    { key: "sphereFold", values: [0.5, 1] },
    { key: "scale", values: [2] },
  ],
};

test("encode → decode round-trips flags, ops, and Julia seed", () => {
  const f = decodeFormula(encodeFormula(sample));
  assert.equal(f.addC, true);
  assert.equal(f.julia, true);
  assert.equal(f.iters, 12);
  assert.deepEqual(f.ops.map((o) => o.key), ["boxFold", "sphereFold", "scale"]);
  // params ride at ×100 fixed point — exact for these values
  assert.deepEqual(f.ops[1].values, [0.5, 1]);
  assert.deepEqual(f.juliaC.map((n) => Math.round(n * 1000) / 1000), [0.5, -0.25, 0.1]);
});

test("caps are exported and sane", () => {
  assert.ok(MAX_DECODE_OPS > 0 && MAX_DECODE_OPS <= 4096);
  assert.ok(MAX_DECODE_PARAMS > 0 && MAX_DECODE_PARAMS <= 256);
});

// ── Overflow-lane arity on the wire (OP_PARAM_ENCODING.md §6) ───────────────
// The codec stores a per-op paramCount and reads it back, so >3 params needed
// no CODEC_VERSION bump. That claim was argued but never pinned — PR-1 listed
// "a 5-value share round-trip" and shipped without it. Pinned here.

test("a >3-param op round-trips all of its values on the wire", () => {
  const wide = {
    ...sample,
    ops: [
      { key: "bulbAxis", values: [8, 2, 1, -1.5, 0.25] },
      { key: "scale", values: [2] },
      { key: "ruckerBulb", values: [8, 1.5, 1, 1, 3] },
    ],
  };
  const f = decodeFormula(encodeFormula(wide));
  assert.deepEqual(
    f.ops.map((o) => o.key),
    ["bulbAxis", "scale", "ruckerBulb"],
  );
  // Values live on the 0.01 quantisation grid, so these are exact.
  assert.deepEqual(f.ops[0].values, [8, 2, 1, -1.5, 0.25]);
  assert.deepEqual(f.ops[2].values, [8, 1.5, 1, 1, 3]);
  // A thin op between two fat ones must not inherit a neighbour's arity.
  assert.deepEqual(f.ops[1].values, [2]);
});

test("a legacy short payload for a widened op decodes short, then sanitizes to the defaults", async () => {
  // The real back-compat path for every param add: the WIRE keeps whatever the
  // old client wrote (bulbAxis shipped with 2, then 3 values), and sanitize —
  // not the codec — pads the rest from the registry. Byte-compat is therefore a
  // property of sanitize, and this pins both halves.
  const legacy = { ...sample, ops: [{ key: "bulbAxis", values: [8, 1, 0] }] };
  const decoded = decodeFormula(encodeFormula(legacy));
  assert.deepEqual(decoded.ops[0].values, [8, 1, 0], "the wire is not padded");

  const { sanitizeFormula } = await import("./sanitize.js");
  const { byKey } = await import("./operators.js");
  const clean = sanitizeFormula(decoded);
  const def = byKey("bulbAxis");
  assert.equal(clean.ops[0].values.length, def.params.length);
  assert.deepEqual(
    clean.ops[0].values,
    def.params.map((p, i) => (i < 3 ? [8, 1, 0][i] : p.default)),
    "the overflow slots pad from the registry defaults",
  );
});

test("a payload declaring a huge opCount is capped, not looped billions of times", () => {
  // Craft a formula payload whose flags/iters/camera are minimal, then a varint
  // opCount of ~4 billion. decodeFormula must bail at MAX_DECODE_OPS / end-of-
  // buffer rather than spinning the loop 4e9 times.
  const w = new ByteWriter();
  w.u8(0); // flags: no addC/julia, deOption 0
  w.varint(8); // iters
  w.zigzag(0).zigzag(0).zigzag(0).zigzag(0); // camera yaw/pitch/dist/fov
  w.varint(0xffffffff); // opCount = 2^32-1
  const t0 = Date.now();
  const f = decodeFormula(w.take());
  const ms = Date.now() - t0;
  assert.ok(f.ops.length <= MAX_DECODE_OPS, `ops ${f.ops.length} exceeds cap`);
  assert.ok(ms < 1000, `decode took ${ms}ms — cap/bailout not working`);
});

test("unknown opcode ids are skipped (forward-compat), not fatal", () => {
  const w = new ByteWriter();
  w.u8(0).varint(8).zigzag(0).zigzag(0).zigzag(0).zigzag(0);
  w.varint(1); // one op
  w.varint(9999); // opcode id that doesn't exist in this build
  w.varint(1).zigzag(100); // 1 param
  const f = decodeFormula(w.take());
  assert.deepEqual(f.ops, []); // unknown op dropped, decode still succeeds
});

// ── #715 — f64 must obey the past-end-is-zero contract, per SECTION ──────────
// unpackContainer hands each section over as a subarray of the whole container
// buffer. The old f64 sliced `bytes.buffer` by ABSOLUTE offset, so a short
// section read the NEXT section's 8 bytes as this section's float — the one
// primitive that violated the length framing. It must now bound the read to its
// own reader and treat a short tail as 0, like u8/varint/zigzag already do.
test("f64 reads a full 8-byte word exactly, ignoring trailing bytes", () => {
  const eight = new ByteWriter().f64(1.5).take(); // exactly 8 bytes
  // Embed the section in a larger parent buffer with trailing bytes, then hand
  // the reader ONLY the 8-byte section as a subarray (the unpackContainer shape).
  const parent = Uint8Array.from([...eight, 7, 7, 7, 7]);
  const section = parent.subarray(0, 8);
  const r = new ByteReader(section);
  assert.equal(r.f64(), 1.5);
  assert.ok(r.done, "reader consumed exactly its 8-byte section");
});

test("f64 on a SHORT section reads 0, never the next section's bytes", () => {
  // A 4-byte section (too short for an f64) followed, in the SAME parent buffer,
  // by another section's bytes. The old code returned a float built from those
  // next-section bytes; the fixed code returns 0 and marks the reader done.
  const parent = Uint8Array.from([1, 2, 3, 4, 0xde, 0xad, 0xbe, 0xef, 9, 9]);
  const shortSection = parent.subarray(0, 4);
  const r = new ByteReader(shortSection);
  assert.equal(r.f64(), 0, "short section must not leak the next section");
  assert.ok(r.done, "reader is exhausted at its own section boundary");
});

// ── #715 — the section-tag registry must have no colliding ids ──────────────
// SHAPES2 and HYBRID_N were BOTH cut at 0x0c (both ship on live), masked only by
// scenes and hybrids being mutually exclusive. SHAPES2 moved to a distinct id.
// (The full uniqueness meta-test over every tag is #727's job; this pins the
// one collision #715 resolved.)
test("SHAPES2 and HYBRID_N no longer share a wire id", () => {
  assert.notEqual(TAG.SHAPES2, TAG.HYBRID_N);
  const ids = Object.values(TAG);
  assert.equal(ids.length, new Set(ids).size, "no two section tags share an id");
});

// ── #727 — the allocation ledgers, as executable gates ───────────────────────
// TAG 0x0c was assigned TWICE (SHAPES2 and HYBRID_N), and the prose ledger not
// only failed to catch it — its own "next free id" comment pointed at an id
// that was already taken. Prose ledgers demonstrably don't gate anything, so
// every wire-id / bit registry the codec and its sibling ledgers maintain is
// checked here as a UNIQUENESS gate that fails loudly, naming the colliders.
// Every test iterates its registry programmatically (Object.entries or a
// source-text scan of the declaration block) — no hand-copied value list that
// can drift out of date as ids are allocated.

// value → [names…]; returns one "nameA, nameB = value" line per collision.
const collisions = (entries) => {
  const byValue = new Map();
  for (const [name, value] of entries) {
    if (!byValue.has(value)) byValue.set(value, []);
    byValue.get(value).push(name);
  }
  return [...byValue]
    .filter(([, names]) => names.length > 1)
    .map(
      ([value, names]) =>
        `${names.join(", ")} = ${Number.isInteger(value) ? "0x" + value.toString(16) : value}`,
    );
};

const coreSrc = (file) =>
  readFileSync(new URL(file, import.meta.url), "utf8");

test("#727 TAG registry: every section tag is a unique u8", () => {
  const entries = Object.entries(TAG);
  // Guard the scan itself: if TAG ever came back near-empty, the uniqueness
  // assert below would pass vacuously.
  assert.ok(entries.length >= 14, `TAG scan implausibly small (${entries.length})`);
  const dups = collisions(entries);
  assert.equal(dups.length, 0, `wire-tag collision: ${dups.join("; ")}`);
  for (const [name, v] of entries) {
    assert.ok(
      Number.isInteger(v) && v >= 0x01 && v <= 0xff,
      `TAG.${name} = ${v} is not a valid nonzero u8 wire tag`,
    );
  }
});

test("#727 the TAG ledger's 'next free id' claim points at a genuinely free id", () => {
  // The 0x0c collision shipped WITH a comment asserting "0x0d is the next free
  // id" while MODULATORS already held 0x0d. Make the prose claim executable:
  // any id the ledger calls free must not be in the registry.
  const claims = [
    ...coreSrc("./sharecodec.js").matchAll(/0x([0-9a-fA-F]+) is the next free id/g),
  ].map((m) => parseInt(m[1], 16));
  assert.ok(claims.length >= 1, "the TAG block lost its 'next free id' ledger line");
  const holder = new Map(Object.entries(TAG).map(([n, v]) => [v, n]));
  for (const id of claims) {
    assert.ok(
      !holder.has(id),
      `ledger claims 0x${id.toString(16)} is free, but TAG.${holder.get(id)} holds it`,
    );
  }
});

test("#727 unpackContainer keeps the FIRST of two same-tag sections (no silent overwrite)", () => {
  // A duplicate tag is corruption or a crafted payload — last-wins would let a
  // trailing section silently shadow the real one. First-wins, still lenient.
  const first = Uint8Array.from([0xaa, 0xab]);
  const second = Uint8Array.from([0xbb]);
  const w = new ByteWriter().u8(CODEC_VERSION);
  w.u8(TAG.HYBRID_N).varint(first.length).raw(first);
  w.u8(TAG.HYBRID_N).varint(second.length).raw(second);
  w.u8(TAG.THEME).varint(1).raw(Uint8Array.from([7]));
  const { sections } = unpackContainer(bytesToB64url(w.take()));
  assert.deepEqual([...sections.get(TAG.HYBRID_N)], [0xaa, 0xab], "first wins");
  assert.deepEqual([...sections.get(TAG.THEME)], [7], "later DISTINCT tags still land");
  assert.equal(sections.size, 2);
});

test("#727 march feature bits (renderer.js F_*): unique single-bit values", () => {
  // The F_* variant-key bits are module-local to renderer.js by design, so the
  // registry is scanned from source. Declarations are `F_NAME = <number>`;
  // usages never put `= <digit>` after the name, so the scan sees exactly the
  // declaration list — self-maintaining as bits are allocated.
  const decls = [
    ...coreSrc("./renderer.js").matchAll(/\bF_([A-Z0-9_]+)\s*=\s*(\d+)/g),
  ].map((m) => [`F_${m[1]}`, Number(m[2])]);
  assert.ok(decls.length >= 15, `F_* scan implausibly small (${decls.length}) — regex rot?`);
  const dups = collisions(decls);
  assert.equal(dups.length, 0, `march feature-bit collision: ${dups.join("; ")}`);
  for (const [name, v] of decls) {
    assert.ok(
      Number.isInteger(v) && v > 0 && (v & (v - 1)) === 0,
      `${name} = ${v} is not a single bit`,
    );
  }
});

test("#727 Globals tail-word registry (shader.js *_WORD): unique rows inside the alloc ceiling", async () => {
  // The vec4-row starts are exported constants — iterate the module's own
  // export surface so a newly appended row joins the gate automatically.
  const shader = await import("./shader.js");
  const entries = Object.entries(shader).filter(([k]) => /_WORD$/.test(k));
  assert.ok(entries.length >= 10, `*_WORD scan implausibly small (${entries.length})`);
  const dups = collisions(entries);
  assert.equal(dups.length, 0, `Globals word-row collision: ${dups.join("; ")}`);
  const alloc = shader.GLOBALS_WORDS_ALLOC;
  assert.ok(Number.isInteger(alloc) && alloc > 0, "GLOBALS_WORDS_ALLOC missing");
  for (const [name, v] of entries) {
    assert.ok(
      Number.isInteger(v) && v >= 0 && v < alloc,
      `${name} = ${v} lies outside the Globals alloc ceiling (${alloc})`,
    );
  }
});

test("#727 share-link P3 flag bytes (kit/share.ts): unique single bits per byte", (t) => {
  // The coloring tail's presence bits are inline literals in the p3flags /
  // p3flags2 OR-expressions (kit/share.ts) — the ledger IDEAS.md tracks by
  // prose. Scan each expression's `? <bit> :` terms. kit/ does not travel with
  // the fractbox-engine mirror, so absent-file is a skip there — in the
  // monorepo the file exists and the gate enforces.
  const url = new URL("../kit/share.ts", import.meta.url);
  if (!existsSync(url)) {
    t.skip("kit/share.ts not present (engine mirror checkout)");
    return;
  }
  const src = readFileSync(url, "utf8");
  for (const decl of ["p3flags", "p3flags2"]) {
    const m = src.match(new RegExp(`const ${decl} =([^;]*);`));
    assert.ok(m, `could not find the ${decl} bit expression — regex rot?`);
    const bits = [...m[1].matchAll(/\?\s*(\d+)\s*:/g)].map((x) => Number(x[1]));
    assert.ok(bits.length >= 2, `${decl} scan implausibly small (${bits.length})`);
    const dups = collisions(bits.map((v, i) => [`${decl} term ${i + 1}`, v]));
    assert.equal(dups.length, 0, `share-link flag-bit collision: ${dups.join("; ")}`);
    for (const v of bits) {
      assert.ok(v > 0 && (v & (v - 1)) === 0, `${decl} bit ${v} is not a single bit`);
    }
  }
});

// ── #727 — frozen wire fixture: old links must keep decoding ────────────────
// A container packed by the shipped encoder (FORMULA + an opaque OBJECTS body
// + the pre-#715 LEGACY 0x0c overflow section), frozen as the literal link
// payload. Byte-level container framing and formula decode are pinned against
// any future codec edit; kit/share.ts's decodeShare additionally maps a 0x0c
// section to SHAPES2 whenever OBJECTS is present (its own vitest covers that).
test("#727 a frozen pre-#715 share payload still unpacks and decodes identically", () => {
  const FIXTURE = "AQETBQrYBKsCwAygBgIAAcgBAgGrAggDAQIDDAIEBQ";
  const { version, sections } = unpackContainer(FIXTURE);
  assert.equal(version, CODEC_VERSION);
  assert.deepEqual(
    [...sections.keys()],
    [TAG.FORMULA, TAG.OBJECTS, 0x0c],
    "section order/tags changed on the wire",
  );
  assert.deepEqual([...sections.get(TAG.OBJECTS)], [1, 2, 3]);
  assert.deepEqual([...sections.get(0x0c)], [4, 5], "legacy 0x0c payload intact");
  assert.deepEqual(decodeFormula(sections.get(TAG.FORMULA)), {
    addC: true,
    julia: false,
    deOption: 1,
    iters: 10,
    camera: { yawDeg: 30, pitchDeg: -15, dist: 8, fovDeg: 40 },
    ops: [
      { key: "boxFold", values: [1] },
      { key: "scale", values: [-1.5] },
    ],
  });
  // And the encoder still emits those exact bytes for the same input.
  assert.equal(
    packContainer([
      { tag: TAG.FORMULA, bytes: encodeFormula(decodeFormula(sections.get(TAG.FORMULA))) },
      { tag: TAG.OBJECTS, bytes: Uint8Array.from([1, 2, 3]) },
      { tag: 0x0c, bytes: Uint8Array.from([4, 5]) },
    ]),
    FIXTURE,
    "re-encoding the decoded fixture must be byte-identical",
  );
});

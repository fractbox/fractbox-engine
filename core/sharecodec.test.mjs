// Zero-tooling guard for the compact share codec. The container/formula decoders
// read length/count varints straight off an untrusted URL payload, so their caps
// and forward-compat behavior are security-relevant. Only exercised from app/
// before now (doesn't travel with the raw-ESM engine).
//
// Run: node --test core/sharecodec.test.mjs   (*.test.mjs → sync skips it)
import assert from "node:assert/strict";
import { test } from "node:test";
import {
  encodeFormula,
  decodeFormula,
  ByteWriter,
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

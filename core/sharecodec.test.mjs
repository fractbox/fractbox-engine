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

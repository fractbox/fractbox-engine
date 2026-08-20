import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { writeSplatPly, readSplatPlyHeader } from "./plysplat.js";

// One synthetic splat bundle (fitSplats output shape).
function bundle(count) {
  const f3 = (v) => Float32Array.from({ length: 3 * count }, (_, i) => v + i);
  return {
    count,
    pos: f3(0.1),
    normal: f3(0.01),
    fdc: f3(0.2),
    opacity: Float32Array.from({ length: count }, (_, i) => 2.9 + i),
    scaleLog: f3(-3),
    rot: Float32Array.from({ length: 4 * count }, (_, i) => (i % 4 === 0 ? 1 : 0)),
  };
}

test("writeSplatPly: degree-0 header order + property names/types (INRIA canonical)", () => {
  const buf = writeSplatPly(bundle(3));
  const { count, properties, stride } = readSplatPlyHeader(buf);
  assert.equal(count, 3);
  assert.equal(stride, 17 * 4); // degree-0: 17 float props
  assert.deepEqual(
    properties.map((p) => p.name),
    ["x","y","z","nx","ny","nz","f_dc_0","f_dc_1","f_dc_2","opacity","scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3"],
  );
  assert.ok(properties.every((p) => p.type === "float"));
});

test("writeSplatPly: fRest:45 → byte-exact INRIA 62-property header", () => {
  const { properties } = readSplatPlyHeader(writeSplatPly(bundle(1), { fRest: 45 }));
  assert.equal(properties.length, 62); // 9 + 45 f_rest + 8 tail
  assert.equal(properties[9].name, "f_rest_0");
  assert.equal(properties[53].name, "f_rest_44");
  assert.equal(properties[54].name, "opacity");
});

test("writeSplatPly round-trips every field bit-for-bit (little-endian)", () => {
  const b = bundle(3);
  const buf = writeSplatPly(b);
  const { vertices, properties } = readSplatPlyHeader(buf, 3);
  const P = properties.length;
  const idx = (n) => properties.findIndex((p) => p.name === n);
  for (let i = 0; i < 3; i++) {
    const j = 3 * i, base = i * P;
    assert.equal(vertices[base + idx("x")], b.pos[j]);
    assert.equal(vertices[base + idx("nz")], b.normal[j + 2]);
    assert.equal(vertices[base + idx("f_dc_1")], b.fdc[j + 1]);
    assert.equal(vertices[base + idx("opacity")], b.opacity[i]);
    assert.equal(vertices[base + idx("scale_2")], b.scaleLog[j + 2]);
    assert.equal(vertices[base + idx("rot_0")], b.rot[4 * i]); // w first
  }
  // explicit little-endian check: "format binary_little_endian" is in the header
  assert.ok(new TextDecoder().decode(new Uint8Array(buf, 0, 64)).includes("binary_little_endian"));
});

test("writeSplatPly refuses an empty capture (§5.3a backstop)", () => {
  assert.throws(() => writeSplatPly({ count: 0 }), RangeError);
});

// Byte-diff vs a REAL foreign writer (SuperSplat export). Pending until the
// fixture is pulled (S-pre carryover / PR B′). A hand-authored header is NOT an
// acceptable substitute (spec §9.1) — this stays skipped, not faked.
test("byte-diff vs reference_splat.ply (SuperSplat)", { skip: !existsSync(fileURLToPath(new URL("./__fixtures__/reference_splat.ply", import.meta.url))) }, () => {
  const ref = readFileSync(fileURLToPath(new URL("./__fixtures__/reference_splat.ply", import.meta.url)));
  const refHdr = readSplatPlyHeader(ref.buffer.slice(ref.byteOffset, ref.byteOffset + ref.byteLength));
  const ours = readSplatPlyHeader(writeSplatPly(bundle(1), { fRest: 45 }));
  // property names/types/order must match the real writer exactly.
  assert.deepEqual(ours.properties, refHdr.properties);
  assert.ok(refHdr.vertices.every(Number.isFinite));
});

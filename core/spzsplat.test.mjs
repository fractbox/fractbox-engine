// Byte-level tests for the .spz v2 writer (issue #368 — compressed splat export).
// The quantization tables are the load-bearing thing: a real viewer (SuperSplat)
// reads these exact bytes, so we pin the absolute encoding constants (not just a
// self-consistent round-trip) AND check pack→unpack recovers every attribute
// within its quantization step.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  packSpz,
  unpackSpz,
  writeSplatSpz,
  readSplatSpz,
  SPZ_MAGIC,
  SPZ_VERSION,
} from "./spzsplat.js";
import { SH_C0, rgb2sh0, fitSplats } from "./splatfit.js";

// A tiny hand-built fit bundle with values chosen to hit known quantization cells.
function bundle() {
  return {
    count: 3,
    // splat 0 at origin; 1 and 2 offset (all within ±2 ⇒ fractionalBits stays 12).
    pos: new Float32Array([0, 0, 0, 1, -0.5, 0.25, -2, 1.5, 0.75]),
    // f_dc so that splat 0 = mid-gray (rgb2sh0(0.5) = 0 exactly → color byte 128).
    fdc: new Float32Array([
      rgb2sh0(0.5),
      rgb2sh0(0.5),
      rgb2sh0(0.5),
      rgb2sh0(1),
      rgb2sh0(0),
      rgb2sh0(0.25),
      rgb2sh0(0.75),
      rgb2sh0(0.2),
      rgb2sh0(0.9),
    ]),
    // opacity is the stored logit: 0 → sigmoid 0.5 → alpha byte 128.
    opacity: new Float32Array([0, Math.log(0.9 / 0.1), Math.log(0.8 / 0.2)]),
    // log-scale: −4 → (−4+10)·16 = 96 exactly.
    scaleLog: new Float32Array([-4, -4, -4, -3, -3, -5, -2, -6, -6]),
    // rotations [w,x,y,z]: identity, and two arbitrary unit quats.
    rot: new Float32Array([
      1, 0, 0, 0, 0.7071067811865476, 0.7071067811865476, 0, 0, 0.5, 0.5, 0.5,
      0.5,
    ]),
  };
}

test("#368: packSpz emits a conforming spz v2 header + exact byte length", () => {
  const b = bundle();
  const packed = packSpz(b);
  // 16-byte header + per-splat (9 pos + 1 alpha + 3 color + 3 scale + 3 rot) = 19.
  assert.equal(packed.length, 16 + b.count * 19);
  const dv = new DataView(packed.buffer);
  assert.equal(dv.getUint32(0, true), SPZ_MAGIC);
  // Magic spells "NGSP" in the file bytes.
  assert.deepEqual([...packed.slice(0, 4)], [0x4e, 0x47, 0x53, 0x50]);
  assert.equal(dv.getUint32(4, true), SPZ_VERSION);
  assert.equal(dv.getUint32(8, true), b.count);
  assert.equal(packed[12], 0); // shDegree 0 (SH0-by-fiat)
  assert.equal(packed[13], 12); // fractionalBits (small cloud → default 12)
  assert.equal(packed[14], 0); // flags
  assert.equal(packed[15], 0); // reserved
});

test("#368: quantization constants match the spz spec (absolute bytes)", () => {
  const packed = packSpz(bundle());
  const count = 3;
  const posOff = 16;
  const alphaOff = posOff + count * 9;
  const colorOff = alphaOff + count;
  const scaleOff = colorOff + count * 3;
  const rotOff = scaleOff + count * 3;

  // splat 0 at origin → all 9 position bytes zero.
  assert.deepEqual(
    [...packed.slice(posOff, posOff + 9)],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
  );
  // splat 1 x=1 with fractionalBits 12 → 4096 = 0x001000 little-endian.
  assert.deepEqual(
    [...packed.slice(posOff + 9, posOff + 12)],
    [0x00, 0x10, 0x00],
  );
  // alpha: logit 0 → sigmoid 0.5 → round(127.5) = 128.
  assert.equal(packed[alphaOff], 128);
  // color: mid-gray (f_dc = 0) → round(0·SH_C0·255 + 127.5) = 128 (pins the color chain).
  assert.equal(packed[colorOff], 128);
  assert.equal(packed[colorOff + 1], 128);
  assert.equal(packed[colorOff + 2], 128);
  // scale: log −4 → round((−4+10)·16) = 96.
  assert.equal(packed[scaleOff], 96);
  // rotation of identity [w=1,x=y=z=0], w≥0 → xyz = round(0·127.5 + 127.5) = 128.
  assert.deepEqual([...packed.slice(rotOff, rotOff + 3)], [128, 128, 128]);
});

test("#368: pack→unpack round-trips every attribute within its quantization step", () => {
  const b = bundle();
  const out = unpackSpz(packSpz(b));
  assert.equal(out.numPoints, b.count);
  assert.equal(out.shDegree, 0);
  const invFP = 1 / 4096;
  for (let i = 0; i < b.count * 3; i++)
    assert.ok(Math.abs(out.pos[i] - b.pos[i]) <= invFP + 1e-6, `pos[${i}]`);
  // color step ≈ (1/255)/SH_C0 ≈ 0.0139 in f_dc units.
  const colStep = 1 / 255 / SH_C0 + 1e-4;
  for (let i = 0; i < b.count * 3; i++)
    assert.ok(Math.abs(out.fdc[i] - b.fdc[i]) <= colStep, `fdc[${i}]`);
  // scale step = 1/16 in log units.
  for (let i = 0; i < b.count * 3; i++)
    assert.ok(
      Math.abs(out.scaleLog[i] - b.scaleLog[i]) <= 1 / 16 + 1e-4,
      `scale[${i}]`,
    );
  // alpha compared in probability space (1/255 + a touch).
  const sig = (x) => 1 / (1 + Math.exp(-x));
  for (let i = 0; i < b.count; i++)
    assert.ok(
      Math.abs(sig(out.opacity[i]) - sig(b.opacity[i])) <= 1 / 255 + 1e-4,
      `alpha[${i}]`,
    );
  // rotation: q and −q are the same rotation, so compare |dot| ≈ 1.
  for (let i = 0; i < b.count; i++) {
    const k = 4 * i;
    let d = 0;
    for (let c = 0; c < 4; c++) d += out.rot[k + c] * b.rot[k + c];
    assert.ok(Math.abs(d) >= 0.99, `rot[${i}] dot=${d}`);
  }
});

test("#368: fractionalBits steps down for a large cloud (no 24-bit overflow)", () => {
  // A coordinate of 5000 would overflow 24-bit fixed-point at 12 bits (5000·4096 >
  // 2^23), so chooseFractionalBits must reduce it and still recover the value.
  const big = {
    count: 1,
    pos: new Float32Array([5000, -3000, 100]),
    fdc: new Float32Array([0, 0, 0]),
    opacity: new Float32Array([0]),
    scaleLog: new Float32Array([-3, -3, -3]),
    rot: new Float32Array([1, 0, 0, 0]),
  };
  const packed = packSpz(big);
  assert.ok(packed[13] < 12, `fractionalBits stepped down (got ${packed[13]})`);
  const out = unpackSpz(packed);
  const step = Math.pow(2, -packed[13]);
  assert.ok(Math.abs(out.pos[0] - 5000) <= step, `x recovered (±${step})`);
  assert.ok(Math.abs(out.pos[1] + 3000) <= step, "y recovered");
});

test("#368: packSpz refuses an empty cloud", () => {
  assert.throws(
    () =>
      packSpz({
        count: 0,
        pos: new Float32Array(0),
        fdc: new Float32Array(0),
        opacity: new Float32Array(0),
        scaleLog: new Float32Array(0),
        rot: new Float32Array(0),
      }),
    /0 splats/,
  );
});

test("#368: writeSplatSpz gzip container round-trips via readSplatSpz + shrinks", async () => {
  // Feed a realistic bundle through the shipping fitSplats so the gzip path sees
  // representative data (and compresses — repetitive quantized bytes gzip well).
  const n = 400;
  const pos = new Float32Array(3 * n);
  const normal = new Float32Array(3 * n);
  const albedo = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    const t = (i / n) * Math.PI * 2;
    pos[3 * i] = Math.cos(t);
    pos[3 * i + 1] = Math.sin(t);
    pos[3 * i + 2] = 0.1 * (i % 5);
    normal[3 * i + 2] = 1;
    albedo[3 * i] = 0.6;
    albedo[3 * i + 1] = 0.4;
    albedo[3 * i + 2] = 0.2;
  }
  const fit = fitSplats(
    { count: n, pos, normal, albedo },
    { convention: "raw", cmScale: 1, r0: 0.05 },
  );
  const packedLen = packSpz(fit).length;
  const buf = await writeSplatSpz(fit);
  assert.ok(buf instanceof ArrayBuffer);
  assert.ok(
    buf.byteLength < packedLen,
    `gzip shrinks (${buf.byteLength} < ${packedLen})`,
  );
  const back = await readSplatSpz(buf);
  assert.equal(back.numPoints, n);
  assert.equal(back.version, SPZ_VERSION);
});

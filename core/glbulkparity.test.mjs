// #729 — renderer-vs-emitter Bulk agreement, behaviorally, for every shipped
// preset. The #627 field failure: buildSceneFragGL derived its fat-leaf
// predicate one way while renderer_gl's bulkLayout() call site derived it
// another, so the emitted text REFERENCED uObjPrimP2 that the renderer's UBO
// packing didn't shape (and in the original break, that the block didn't even
// declare) — every fat-leaf scene killed the WebGL2 tier at first bake. The
// GL probes (app/scripts/gl-link-check.mjs) used to hand-construct their own
// bulkLayout() args, so a renderer-vs-emitter drift was invisible to them by
// construction.
//
// This gate closes the seam IN CI with no GL anywhere: drive the REAL
// createRendererGL through every preset's write path against the recording
// mock, capture (a) the byte size the renderer allocates for the Bulk UBO
// (setupBulk's bufferData — the renderer's bulkLayout() args) and (b) the
// fragment text it linked (the emitter's own Bulk declaration), and assert
// the declaration's std140 size equals the allocation. Disagreeing args —
// the #627 leafAux class, or a future member predicate — change one side's
// byte count and fail with the preset named.
//
// Run: node --test core/glbulkparity.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { createRendererGL } from "./renderer_gl.js";
import { PRESETS } from "./oplist.js";
import { activeHybridSlots } from "./hybridmodel.js";
import { makeRecordingGL } from "./__fixtures__/glmock.mjs";

// std140 size of the Bulk block DECLARED in the emitted text. Every member is
// an array, and std140 rounds float/int/vec array strides up to 16 bytes —
// the same running-sum renderer_gl.setupBulk computes from bulkLayout().
function declaredBulkBytes(glsl) {
  const block = glsl.match(/layout\(std140\) uniform Bulk \{([\s\S]*?)\};/);
  assert.ok(block, "emitted fragment declares no Bulk block");
  let bytes = 0;
  for (const m of block[1].matchAll(
    /^\s*(?:float|int|vec[234])\s+u\w+\[(\d+)\];\s*$/gm,
  ))
    bytes += Number(m[1]) * 16;
  assert.ok(bytes > 0, "Bulk block parsed empty — declaration format drifted");
  return bytes;
}

test("every preset: the renderer's Bulk UBO allocation equals the emitted shader's own declaration", async () => {
  let flat = 0,
    hyb = 0,
    scene = 0,
    fatDeclared = 0;
  for (const p of PRESETS) {
    const m = makeRecordingGL();
    const r = await createRendererGL(m.canvas);
    r.writeGlobals({}); // arm the deferred boot link (#708)
    if (p.objects) {
      r.writeScene(p.objects);
      scene++;
    } else if (p.hybrid) {
      const { slots, counts } = activeHybridSlots(p);
      r.writeHybrid(
        slots.map((s) => s.ops),
        counts,
        slots.map((s) => !!s.addC),
      );
      hyb++;
    } else {
      r.writeOps(p.ops);
      flat++;
    }
    assert.ok(m.rec.progSrc, `"${p.name}": no program linked`);
    if (/\buObjPrimP2\b/.test(m.rec.progSrc)) fatDeclared++;
    assert.equal(
      declaredBulkBytes(m.rec.progSrc),
      m.rec.bulkAlloc,
      `"${p.name}": the renderer allocated a Bulk UBO that disagrees with ` +
        "the Bulk block its own linked shader declares — the #627 " +
        "renderer-vs-emitter predicate drift (a member the packer shapes " +
        "that the text lacks, or vice versa)",
    );
  }
  assert.equal(flat + hyb + scene, PRESETS.length, "a preset fell out of the sweep");
  // The historic break was the FAT-leaf shape specifically — prove the sweep
  // actually exercises it (City Blocks + Whorl City ship today).
  assert.ok(
    fatDeclared >= 2,
    `only ${fatDeclared} preset(s) exercised the uObjPrimP2 fat-leaf Bulk shape`,
  );
});

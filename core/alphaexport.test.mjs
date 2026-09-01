// #428 item 2 — transparent-background PNG export (WebGPU tier only). CI has
// no GPU context ("WGSL is compiled nowhere in CI"), so this is a source-level
// regression pinning the exact alpha contract instead of a rendered-pixel
// check: a true miss (empty sky) writes alpha 0, a hit stays fully opaque
// (including the confidence-faded "soft hit" grazing case, which shares the
// same hit-path return), the post pass carries the HDR alpha through instead
// of re-hardcoding 1.0, and the readback in renderer.js/preview.js forces
// alpha back to fully opaque for every caller that hasn't explicitly opted
// in — so this feature can't silently change any EXISTING captureFrame
// consumer (thumbnails, mp4 frames, splat capture, etc.).
//
// Run: node --test core/alphaexport.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { buildWGSL, buildPostWGSL } from "./shader.js";

test("#428: a true miss (sky ray) writes alpha 0 in the generated march shader", () => {
  const src = buildWGSL({});
  assert.match(
    src,
    /return vec4f\(skyOut, 0\.0\);/,
    "miss path must write alpha 0 so renderToImage can expose real transparency",
  );
});

test("#428: the hit-surface return is untouched — still fully opaque", () => {
  const src = buildWGSL({});
  assert.match(
    src,
    /return vec4f\(max\(col, vec3f\(0\.0\)\), 1\.0\);.*linear HDR out/,
    "hit pixels (including soft/grazing hits, which share this return) must stay alpha 1",
  );
});

test("#428: the post pass carries the HDR sample's alpha through, not a hardcoded 1.0", () => {
  const src = buildPostWGSL();
  assert.match(src, /let hdrSample = textureLoad\(hdr, vec2i\(pos\.xy\), 0\);/);
  assert.match(src, /return vec4f\(o, hdrSample\.a\);/);
  assert.doesNotMatch(
    src,
    /return vec4f\(o, 1\.0\);/,
    "post pass must not silently re-opaque the alpha the march pass computed",
  );
});

test("#428: renderToImage forces alpha back to opaque unless the caller opts in", () => {
  const src = readFileSync(
    fileURLToPath(new URL("./renderer.js", import.meta.url)),
    "utf8",
  );
  assert.match(
    src,
    // overlayDt (Field streamlines lab, PR #656) trails the original list; its
    // 0 default means every existing caller keeps a byte-identical frame.
    /async function renderToImage\(\s*W,\s*H,\s*samples = 1,\s*wantAlpha = false,\s*bakeDOF = true,[\s\S]{0,900}?overlayDt = 0,?\s*\)/,
    "wantAlpha must default to false — existing callers (0-arg/3-arg) stay opaque",
  );
  // The streamlines overlay must never leak into an alpha export (additive
  // glow has no coverage — it would silently vanish on re-composite).
  assert.match(
    src,
    /if \(overlayDt > 0 && !wantAlpha && stream\?\.on\)/,
    "the offline overlay composite must be gated on overlayDt AND !wantAlpha",
  );
  assert.match(
    src,
    /out\[d \+ 3\] = wantAlpha \? src\[s \+ 3\] : 255;/,
    "the readback must gate real alpha behind wantAlpha, forcing 255 otherwise",
  );
});

test("#428: preview.js's captureFrame threads opts.alpha into renderToImage", () => {
  const src = readFileSync(
    fileURLToPath(new URL("./preview.js", import.meta.url)),
    "utf8",
  );
  // The call grew the streamlines overlayDt tail arg (PR #656) — the alpha
  // threading is unchanged, and the overlay arg itself re-guards on
  // !opts.alpha so a transparent capture can never carry the overlay.
  assert.match(
    src,
    /renderer\.renderToImage\(\s*W,\s*H,\s*samples,\s*!!opts\.alpha,/,
  );
  assert.match(
    src,
    /streamOn && !opts\.alpha \? opts\.streamDt \|\| 1 \/ 30 : 0,/,
    "offline overlay frames must inherit the live toggle, on a fixed virtual dt",
  );
});

// #509 — Plain Save PNG snapshot: the flythrough/animate drawer's transparent
// export (#428/#482) was scoped to that drawer only; stillBlob (the plain
// single-frame Save) had no alpha option at all. Same source-level pinning
// strategy as the #428 tests above (no GPU in CI).
test("#509: renderToImage's bakeDOF param defaults true and gates the per-sample lens jitter", () => {
  const src = readFileSync(
    fileURLToPath(new URL("./renderer.js", import.meta.url)),
    "utf8",
  );
  assert.match(
    src,
    /const \[lx, ly\] = i === 0 \|\| !bakeDOF \? \[0, 0\] : lensSample\(i\);/,
    "a sample past the first must drop its lens offset when bakeDOF is false, " +
      "matching stillBlob's own DOF-convergence gate",
  );
});

test("#509: stillBlob's alpha branch reuses renderToImage instead of the presented-canvas readback", () => {
  const src = readFileSync(
    fileURLToPath(new URL("./preview.js", import.meta.url)),
    "utf8",
  );
  assert.match(
    src,
    /if \(opts\.alpha && renderer\.renderToImage\) \{/,
    "stillBlob must route opts.alpha through renderToImage — canvas.toBlob() on " +
      'the presented (alphaMode:"opaque") canvas can never carry real alpha',
  );
  assert.match(
    src,
    /renderer\.renderToImage\(\s*W,\s*H,\s*STILL_SAMPLES,\s*true,\s*bakeDOF,?\s*\)/,
    "the alpha path must pass its own bakeDOF-gated STILL_SAMPLES + wantAlpha:true",
  );
});

test("#509: stillBlobAlpha still embeds opts.metadata (the PNG must re-open like every other save)", () => {
  const src = readFileSync(
    fileURLToPath(new URL("./preview.js", import.meta.url)),
    "utf8",
  );
  assert.match(
    src,
    /return embedMetaIfAny\(blob, opts\.metadata\);/,
    "a transparent save must still carry the #c= formula/share-URL metadata",
  );
});

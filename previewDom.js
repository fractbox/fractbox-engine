// App-side DOM glue for the preview engine (../core/preview.js). Core builds
// NO app markup — PNG export renders to a Blob (stillBlob) and a thumbnail
// renders to a data URL (renderThumbTile); this module turns those into a
// download <a> and a clickable thumbnail grid (issue #77 / REFACTORING.md #3
// — "core owns no DOM"). Duplicated near-verbatim in each of this repo's
// frontends (formula-blocks, imposter, the flagship app/, the OSS demo) since
// they don't share a build system — see core/preview.js's header comment.
//
// Each helper takes a `preview` handle (createPreview's return value) and
// never imports core internals directly, so there's no cycle.

// Render the current view to a PNG and trigger a browser download. Returns true
// on success, false if the engine had nothing to render (no GPU / no formula).
// A null-blob capture propagates as a throw, matching the pre-split exportPNG.
export async function downloadStill(preview, filename, opts = {}) {
  const blob = await preview.stillBlob(opts);
  if (!blob) return false;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
  return true;
}

// Fill a grid element with clickable preset thumbnails. Builds every button up
// front (so the grid appears immediately), paints cache hits instantly, then
// renders only the misses on the GPU under one busy bracket. Markup contract
// (relied on by the flagship's Mine-tile decoration and the Imposter tap
// handler): each tile is
// `<button class="thumb" type="button"><img><div class="lbl">name</div></button>`.
export async function fillThumbnailGrid(preview, presets, gridEl, onPick, opts = {}) {
  if (!preview.hasGPU || !gridEl) return;
  // Tile resolution defaults to the preset-picker size; callers (e.g. the
  // Imposter game) can pass a larger W/H for crisp, non-grainy tiles.
  const W = opts.W || 168,
    H = opts.H || 112;
  gridEl.innerHTML = "";
  const entries = presets.map((p) => {
    const b = document.createElement("button");
    b.className = "thumb";
    b.type = "button";
    const img = document.createElement("img");
    const lbl = document.createElement("div");
    lbl.className = "lbl";
    lbl.textContent = p.name;
    b.append(img, lbl);
    b.addEventListener("click", () => onPick(p));
    gridEl.appendChild(b);
    return { p, img };
  });
  // Cache hits first (instant); only the misses touch the GPU.
  const misses = [];
  for (const e of entries) {
    const hit = preview.thumbTileCached(e.p);
    if (hit) e.img.src = hit;
    else misses.push(e);
  }
  if (!misses.length) return;
  preview.beginThumbs();
  try {
    for (const e of misses) e.img.src = await preview.renderThumbTile(e.p, W, H);
  } catch (err) {
    console.error("thumbnails:", err);
  } finally {
    preview.endThumbs();
  }
}

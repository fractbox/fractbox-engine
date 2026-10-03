// Blank-frame verdicts over a small RGBA downsample — the ONE rule the
// thumbnail pruner (preview.js tileBlankVerdict) and the still-export guard
// share, kept pure so node can pin it.
//
// `data` is the RGBA byte array of an n×n (or any) downsample. Three
// independent readings, because the callers mean different things by "blank":
//   uniform — fewer than `minDiffering` pixels stray more than `tol` (summed
//             |ΔR|+|ΔG|+|ΔB|) from the top-left sample. The thumbnail rule:
//             "this move makes nothing visible" — a flat sky IS blank there.
//   allZero — every byte, alpha included, is 0. The tiled export's PR-1 failure
//             signature (renderer.js createTileTarget.read): an opaque render
//             writes alpha 255, so an all-zero copy is a texture nobody drew
//             into — never a legitimate picture.
//   black   — uniform AND the reference pixel is (near-)black. What a dead or
//             never-presented WebGPU canvas reads back as through a 2D
//             drawImage; a legitimate opaque render of an empty view has a
//             shaded background and fails `uniform` long before it gets here.
export function classifyBlank(data, { tol = 24, minDiffering = 3 } = {}) {
  if (!data || data.length < 4) return { uniform: true, allZero: true, black: true };
  const br = data[0],
    bg = data[1],
    bb = data[2];
  let differing = 0;
  let allZero = true;
  for (let i = 0; i < data.length; i += 4) {
    if (allZero && (data[i] | data[i + 1] | data[i + 2] | data[i + 3]) !== 0)
      allZero = false;
    if (Math.abs(data[i] - br) + Math.abs(data[i + 1] - bg) + Math.abs(data[i + 2] - bb) > tol)
      differing++;
  }
  const uniform = differing < minDiffering;
  return { uniform, allZero, black: uniform && br + bg + bb <= tol };
}

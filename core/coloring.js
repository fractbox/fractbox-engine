// Preview coloring — app-level style, NOT part of the exported op-list (the
// final look is set in the desktop app). Shared by both web frontends.

export const defaultColoring = () => ({
  mode: 0, // 0 = surface (normal), 1 = orbit trap, 2 = escape bands
  colA: [0.86, 0.46, 0.18],
  colB: [0.18, 0.62, 0.74],
  bg: [0.07, 0.09, 0.15],
  // Optional cosine palette (IQ): albedo = a + b·cos(2π(c·t + d)). off → colA→colB.
  palette: {
    on: false,
    a: [0.5, 0.5, 0.5],
    b: [0.5, 0.5, 0.5],
    c: [1, 1, 1],
    d: [0.0, 0.33, 0.67],
  },
  // Key light + ambient/rim/gloss, plus the P1 rig (RENDER_QUALITY):
  // soft shadow + real AO default ON (the "defaults upgrade" decision),
  // metallic/fill/back off. These values MUST match the renderer-side
  // fallbacks (renderer.js / renderer_gl.js) — an old saved coloring with no
  // P1 fields renders identically to a fresh defaultColoring().
  light: {
    dir: [0.45, -0.65, 0.75],
    ambient: 0.16,
    rim: 0.45,
    gloss: 0.0,
    intensity: 1.0,
    keyColor: [1, 1, 1],
    metallic: 0.0,
    shadow: 0.5, // penumbra softness; 0 = shadow off
    ao: 0.55, // AO strength; 0 = off
    fill: 0.0, // fill-light intensity (dir derived from key)
    fillColor: [1, 1, 1],
    back: 0.0, // back-light intensity (key dir mirrored)
    backColor: [1, 1, 1],
    // P3 atmosphere macros — all OPT-IN (0): the sky blend changes the bg mood,
    // which saved looks own. sky drives blend+sun+IBL; fog drives density+
    // in-scatter; glow drives bloom (WebGPU-only).
    sky: 0.0,
    fog: 0.0,
    glow: 0.0,
    aperture: 0.0, // P4 depth of field (0 = pinhole); lens radius scales with orbit distance
    // Whole-frame EV bias (post pass; lifts background too) — the "gamma"
    // knob. NEW colorings start at +0.25 EV ("everything feels dark"
    // feedback); old saves without the field keep 0 — their era's look.
    exposure: 0.25,
  },
});

export const hexToRgb = (h) => {
  let hex = String(h).replace(/^#/, "");
  // Expand 3-digit shorthand (#abc → #aabbcc) so it doesn't parse as garbage.
  if (hex.length === 3) hex = hex.replace(/./g, (c) => c + c);
  const n = parseInt(hex, 16);
  if (!Number.isFinite(n)) return [0, 0, 0]; // malformed → safe default, not NaNs
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
};

export const rgbToHex = (c) =>
  "#" +
  c
    .map((x) =>
      Math.round(Math.max(0, Math.min(1, x)) * 255)
        .toString(16)
        .padStart(2, "0"),
    )
    .join("");

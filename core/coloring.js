// Preview coloring — app-level style, NOT part of the exported op-list (the
// final look is set in the desktop app). Shared by both web frontends.

// Highest valid coloring mode. THE emitters are the authority — shader_gl.js
// (`uColorMode == 7`), shader.js (the `> 6.5` threshold ladder) and cpu.js
// (`mode === 7`) all top out at Address — and MODE_LABELS in the app has the
// matching 8 entries. Single-sourced here because the tiers disagree on what an
// OUT-of-range mode does: WGSL compares thresholds, so mode 9 renders as
// Address, while GLSL/CPU compare equality and fall through to Surface. That
// divergence is why sanitize.js clamps to this bound instead of passing it on.
export const COLOR_MODE_MAX = 7;

export const defaultColoring = () => ({
  // 0 surface / 1 glow / 2 bands / 3 silk / 4 pinwheel / 5 curvature /
  // 6 painter / 7 address  (== COLOR_MODE_MAX; this list used to stop at 5)
  mode: 0,
  stripeFreq: 5, // COLORING S2 — Silk stripe frequency (mode 3); rides colB.w
  // COLORING P2 — auto-levels: normalize the signal by its per-formula range so
  // the palette spans the actual values (fixes "muddy" narrow/skewed signals,
  // esp. curvature). Applies to glow/bands/silk/curvature; surface/pinwheel are
  // untouched (cyclic). Default on for new formulas; absent (old saves) → off.
  autoLevels: true,
  // COLORING P3 S6 — iridescence: a Glow-only modulator that shifts the palette
  // phase per pixel by the orbit's axis asymmetry, painting multi-hue shimmer.
  // 0 = off (default). Range 0–1.
  iridescence: 0,
  // COLORING P3 — palette phase: a global 0–1 rotation of the palette lookup.
  // 0 = identity; animatable on the timeline (interp cyclic-lerps it) for color
  // flow / cycling in video export. Seamless on cyclic palettes.
  palettePhase: 0,
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
    dir: [0.395, 0.657, 0.643], // #160: az 31°, el 40° in the Z-up az/el frame (dirFromAzEl)
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
    // Sun glow is visible by default when Sky > 0 (#160: "nice to see it while
    // positioning, but I want to turn it off"); false hides the glow disc from
    // the sky background while leaving the sky's IBL/reflection lighting intact.
    sunGlow: true,
    // Light-position indicator (#391) — a SEPARATE on/off toggle from Sun Glow
    // above: it shows the same glow-disc at a fixed amount regardless of the
    // Sky slider (Sun Glow only matters once Sky > 0). Off by default so a
    // fresh scene / old saved coloring renders byte-identical to before this
    // feature; flip it on while aiming the light, off again once positioned.
    lightIndicator: false,
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

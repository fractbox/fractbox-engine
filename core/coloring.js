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
  // Directional light + ambient/rim/gloss. Defaults reproduce the original look.
  light: {
    dir: [0.45, -0.65, 0.75],
    ambient: 0.16,
    rim: 0.45,
    gloss: 0.0,
    intensity: 1.0,
  },
});

export const hexToRgb = (h) => {
  const n = parseInt(h.slice(1), 16);
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

// Shared per-frame parameter derivation — the pure math both live renderers
// used to hand-mirror (renderer.js writeGlobals ⟷ renderer_gl.js applyUniforms):
// color/palette defaults, the light rig (penumbra k from the shadow slider,
// fill/back directions DERIVED from the key dir), and the P3 sky/fog macro
// expansion. Derive ONCE here, pack twice — byte consistency between the WebGPU
// and WebGL2 tiers is the whole point (both packers truncate the same f64
// doubles to f32, so equal inputs give bit-equal uniforms).
//
// Backend-specific words (WGSL hyb/morph/colorX/post/jitter/DOF, the recenter
// offset, iteration/march counts) stay in each renderer — this module owns only
// what BOTH tiers must agree on. Defaults here MUST match core/coloring.js
// defaultColoring (the curated-defaults invariant the share codec's guarded
// reads rely on).
import { srgbToOklab } from "./oklab.js";
import { TNEAR_K, TFAR_K, TFAR_MIN } from "./renderpolicy.js";

// #391 — fixed glow-disc amount for the standalone light-position indicator,
// independent of the Sky/Atmosphere slider (see sunGlow below). Chosen to be
// clearly visible without blowing out the background.
const LIGHT_INDICATOR_AMT = 0.5;

export function deriveFrameParams(g = {}) {
  const P = g.palette || {};
  const L = g.light || {};
  // COLORING P0 — N-stop palette. `palette.stops` is the v2 shape
  // ([{c: sRGB[3], p: 0..1}]); ABSENT (or < 2 stops) → count 0 and the legacy
  // cosine/ramp path renders unchanged (defaultColoring stays cosine — #239).
  // Convert to OKLab ONCE here (both GPU tiers upload the same words), sorted
  // by position, capped at the shader's 8-slot array. Each packed stop is
  // [L, a, b, p].
  // `on` is the master palette switch (gates both cosine and stops), so
  // toggling the palette off falls straight back to the colA→colB ramp even
  // when stops are still stored on the object.
  const rawStops = P.on && Array.isArray(P.stops) ? P.stops : [];
  const palStops =
    rawStops.length >= 2
      ? rawStops
          .slice()
          .sort((a, b) => (a.p ?? 0) - (b.p ?? 0))
          .slice(0, 8)
          .map((s) => [...srgbToOklab(s.c || [0, 0, 0]), s.p ?? 0])
      : [];
  const palStopCount = palStops.length; // 0 → legacy path
  const ld = L.dir || [0.395, 0.657, 0.643]; // #160 Z-up default (az 31°, el 40°)
  const shadowAmt = L.shadow ?? 0.5; // 0 = off; 0..1 = penumbra size
  const sky = L.sky ?? 0;
  // Deep zoom — the shader normalizes fog by tFar (G.camPos.w), but TFAR_MIN
  // floors tFar at object scale (#364), so past dist ≈ TFAR_MIN/TFAR_K the
  // whole visible band sits at t ≪ tFar and the Fog slider goes numb
  // (measured: fog 0.8 ⇒ ≤1.5% haze on the deepest visible surface at
  // ×9·10³ — "how do I make depth pronounced?", 2026-07-31). Boosting the
  // DENSITY by the floor ratio is exactly equivalent to normalizing by the
  // un-floored visible range (dist·TFAR_K): identical whenever the floor
  // isn't binding, and at depth the slider spans the visible band again.
  // Lives HERE so both GPU tiers inherit it from the one shared derivation;
  // dist derives from tNear, which is exact dist·TNEAR_K (never floored).
  const fogRaw = L.fog ?? 0;
  const dist = (g.tNear ?? 0) / TNEAR_K;
  const fogComp =
    fogRaw > 0 && dist > 0 ? Math.max(1, TFAR_MIN / (dist * TFAR_K)) : 1;
  // Deep zoom fog mode (field round 2, 2026-07-31): a deep-zoom frame spans
  // DECADES of depth (near cliff t ≈ dist, background valley t ≈ O(1)), and
  // any single LINEAR density either ignores the near field (the original
  // numb slider) or erases everything beyond ~100× dist (density-boost
  // round 1: fog 0.01 blacked out a vista's background). Past comp > 4
  // (dist ≲ 0.6 — where legacy fog was numb anyway, so no saved look
  // changes) the fog word goes NEGATIVE: |value| = the raw slider, and the
  // shaders switch to log-depth fog that accumulates per DECADE beyond the
  // near field. comp ∈ (1, 4] keeps the mild linear boost; comp = 1 is
  // byte-identical legacy.
  const fog = fogComp > 4 ? -fogRaw : fogRaw * fogComp;
  const glow = L.glow ?? 0; // bloom macro (WebGPU-only passes; GL ignores)
  // ENVX (backgrounds P5) — starfield / Milky-Way band / zenith color. All
  // opt-in (0/absent); `envx` is the CODEGEN latch both GPU tiers key their
  // shader variant on (renderer.js activeFeat, renderer_gl.js relink), so the
  // off path emits byte-identical text. Density maps the 0..1 slider onto the
  // cell scale D (≈12..48 cells/unit — star count grows ~D²); the band normal
  // derives from ONE tilt slider (elevation of the galactic plane, azimuth
  // fixed — the camera orbits, so azimuth is free anyway).
  const stars = Math.max(0, Math.min(1, L.stars ?? 0));
  const band = Math.max(0, Math.min(1, L.band ?? 0));
  const bandA = Math.max(0, Math.min(1, L.bandTilt ?? 0.6)) * (Math.PI / 2);
  const zenith = Array.isArray(L.zenith) ? L.zenith : null;
  return {
    colA: g.colA || [0.86, 0.46, 0.18],
    colB: g.colB || [0.18, 0.62, 0.74],
    bg: g.bg || [0.07, 0.09, 0.15],
    juliaC: g.juliaC || [0, 0, 0],
    // Julia mode folds into the add-gate: c is a fixed constant instead of the
    // sample point, and it's always added (regardless of the preset's AddC).
    addGate: g.addC || g.julia ? 1 : 0,
    julia: g.julia ? 1 : 0,
    colorMode: g.colorMode || 0,
    // COLORING S2 — Silk stripe frequency (mode 3). Rides colB.w; default 5,
    // clamped to the slider range so a corrupt value can't alias to a blur.
    stripeFreq: Math.max(1, Math.min(16, g.stripeFreq ?? 5)),
    deScale: g.deScale ?? 0.85,
    deOption: g.deOption ?? 2,
    tNear: g.tNear ?? 0.02, // deep zoom §5
    tFar: g.tFar ?? 80.0,
    // Cosine palette (defaults reproduce the original look).
    palA: P.a || [0.5, 0.5, 0.5],
    palB: P.b || [0.5, 0.5, 0.5],
    palC: P.c || [1, 1, 1], // freq
    palD: P.d || [0, 0.33, 0.67], // phase
    palOn: P.on ? 1 : 0,
    // N-stop palette (OKLab, packed [L,a,b,p]); count 0 → legacy path above.
    palStops,
    palStopCount,
    palCyclic: P.cyclic ? 1 : 0,
    // COLORING P2 — auto-levels signal range (computed upstream where the
    // formula is known; identity 0,1 = no remap). See core/cpu.js signalRange.
    sigLo: g.sigLo ?? 0,
    sigSpan: g.sigSpan ?? 1,
    // COLORING P3 — iridescence (Glow trap-XYZ modulator) + palette phase.
    iridescence: Math.max(0, Math.min(1, g.iridescence ?? 0)),
    palettePhase: g.palettePhase ?? 0,
    // P1 light rig. Defaults ARE the shipped "defaults upgrade": soft shadow on
    // (0.5 → k 17), AO on (0.55), no metallic, white key, fills off — so old
    // saved colorings (no fields) render the upgraded default look.
    lightDir: ld,
    ambient: L.ambient ?? 0.16,
    rim: L.rim ?? 0.45,
    gloss: L.gloss ?? 0.0,
    intensity: L.intensity ?? 1.0,
    keyColor: L.keyColor || [1, 1, 1],
    metallic: L.metallic ?? 0,
    shadowK: 30 - 26 * shadowAmt, // penumbra k (30 hard … 4 very soft)
    shadowOn: shadowAmt > 0 ? 1 : 0,
    ao: L.ao ?? 0.55,
    // Fill/back directions are DERIVED from the key dir (not stored): fill from
    // the opposite azimuth with flattened elevation, back mirrored fully. The
    // light frame is world Z-up (elevation on +Z, azimuth in XY — see 31e2253),
    // so "flatten elevation" = negate x/y, scale z. Shaders normalize.
    fillDir: [-ld[0], -ld[1], ld[2] * 0.35],
    fill: L.fill ?? 0,
    fillColor: L.fillColor || [1, 1, 1],
    backDir: [-ld[0], -ld[1], -ld[2]],
    back: L.back ?? 0,
    backColor: L.backColor || [1, 1, 1],
    // P3 env/fog macros. One UI knob each: sky drives blend + sun glow + IBL
    // amount (sun glow hideable, #160); fog drives density + in-scatter; glow
    // drives bloom strength (threshold fixed at 1.0 — only above-white blooms).
    sky,
    // sunGlow packs TWO independent sources into one shader amount (both drive
    // the same envColor() glow-disc term, so one uniform covers both):
    //  - the Sky-linked glow (existing #160 "Sun glow" checkbox) — requires
    //    Sky > 0 to show anything, and is 0 whenever that checkbox is off.
    //  - the standalone light-position indicator (#391) — a fixed amount shown
    //    whenever `lightIndicator` is on, regardless of Sky (so users who never
    //    touch Atmosphere can still see where the light points). Off by default
    //    so existing looks render byte-identical to before this feature.
    sunGlow: Math.max(
      L.sunGlow === false ? 0 : sky,
      L.lightIndicator ? LIGHT_INDICATOR_AMT : 0,
    ),
    ground: 0.35, // ground dim (fixed)
    ibl: sky,
    fog,
    // In-scatter rides the BOOSTED density magnitude, deliberately: at depth
    // the pow⁸ sun-cone tail × the boost acts as an ambient LIT haze, and
    // that is what makes far structure read as "far" against dark
    // backgrounds (A/B'd 2026-07-31: slider-true and √boost-capped variants
    // both fade far surfaces into darkness — technically fog, visually
    // nothing). Always positive — the fog word's sign is the mode flag.
    inScatter: fogRaw * fogComp,
    bloomStrength: glow * 0.8,
    bloomThreshold: 1.0, // pre-tonemap HDR
    bloomOn: glow > 0,
    exposure: L.exposure ?? 0, // whole-frame EV (user "Exposure" slider)
    // ENVX (backgrounds P5) — see the derivation block above. Like the sky
    // macro itself, GPU-tier only (the CPU/ASCII fallback renders the legacy
    // screen-space gradient — the bloom precedent).
    stars,
    // 14..80 cells/unit (star count ~D²), and starField adds a second 1.7×-D
    // dust layer on top — max density is "mountain sky", not "a few dots".
    starDensity: 14 + 66 * Math.max(0, Math.min(1, L.starDensity ?? 0.5)),
    starSeed: Math.max(0, Math.min(1, L.starSeed ?? 0)) * 97,
    band,
    bandDir: [0, Math.sin(bandA), Math.cos(bandA)],
    zenith: zenith ? zenith.slice(0, 3) : [0, 0, 0],
    zenithOn: zenith ? 1 : 0,
    envx: stars > 0 || band > 0 || !!zenith,
  };
}

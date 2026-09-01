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

// CLIP (cross-section) — plane-offset domain, world units. Wide enough for
// the LARGEST formula the world-extents probe can measure (preview.js
// measureWorldExtents scans ±48; Mandelbox alone needs ±7 — the fixed ±2 the
// feature landed with couldn't even reach ITS far side, which is half of the
// "MRI sweep doesn't go through the whole fractal" defect). The World-pane
// slider keeps its own tighter −2..2 range for hand-framing near-unit
// presets; this is the WIRE/derivation fence, one source for frameparams,
// sanitize.js and cpu.js.
export const CLIP_OFFSET_LIMIT = 64;

// CLIP JAGGED — the noised/eroded cut ("the plane consumes the shape"). The
// UI slider is 0..1; these three constants are the one source all three
// tiers derive from (WGSL/GLSL read the derived clipS.yzw / uClipJ words;
// cpu.js and the app's sweep-amplitude helper import the constants):
//
// CLIP_JAG_WORLD — slider→world-units amplitude (max erosion depth 0.3).
// CLIP_JAG_FREQ  — the noise's base spatial frequency (chunky erosion;
//                  2 octaves at f and 2.03f — see shader.js clipJagNoise).
// CLIP_JAG_LIP   — worst-case Lipschitz bound of the UNIT-amplitude noise
//                  per unit frequency: one smoothstep-interpolated value
//                  octave is ≤ √3·(1.5·2) ≈ 5.2 (per-axis slope 1.5 × the
//                  [-1,1] corner span), and the (1, 0.5)-weighted 2-octave
//                  sum normalized by 1.5 is ≤ 5.2·2/1.5 ≈ 7. The march term
//                  divides by (1 + LIP·amp·freq), which restores the
//                  lower-bound property of the noised plane distance — the
//                  DE-safety argument, paid only near the cut and only when
//                  Jagged > 0 (worst case ≈ 1/7.3 step scale at slider 1).
export const CLIP_JAG_WORLD = 0.3;
export const CLIP_JAG_FREQ = 3.0;
export const CLIP_JAG_LIP = 7.0;

// AURORA (ENVX P6) — one hue (0..1) → an sRGB color at fixed value 1 and the
// given saturation. A tiny HSV slice — enough for a sky tint, not a general
// picker. Derived HERE (not in the shaders) so both GPU tiers upload
// identical words; a shader-side hue decode would be a second implementation
// to drift (the deriveFrameParams charter at the top of this file).
const hueRgb = (h, s) => {
  const f = (n) => {
    const k = (n + h * 6) % 6;
    return 1 - s * Math.max(0, Math.min(k, 4 - k, 1));
  };
  return [f(5), f(3), f(1)];
};

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
  // IMGTEX (#631) — user-image env map / triplanar surface sliders. Amounts
  // default to 1 ON PURPOSE: the codegen latch is renderer-side ("an image is
  // loaded AND amount > 0" — this module can't know about textures), so the
  // defaults are inert with no image loaded, and a freshly loaded image shows
  // immediately. WebGPU tier only (renderer_gl.js/cpu.js ignore these — the
  // bloom precedent); the words are written every frame regardless.
  const emapAmt = Math.max(0, Math.min(1, L.envMapAmount ?? 1));
  const triAmt = Math.max(0, Math.min(1, L.surfTexAmount ?? 1));
  // #630 self-reflection — marched mirror bounces (Catoptron parity). Same
  // opt-in/latch shape as ENVX above: `sreflect` is the CODEGEN latch the
  // WebGPU renderer keys its shader variant on; reflBounces 0 (the default,
  // and what shadeLight forces on cheap frames) derives false and the off
  // build emits byte-identical text. The count is an integer 0..6 (the shader
  // floors it via u32()); the three scalars are plain 0..1 words. WebGPU-only
  // by structural absence — the GL/CPU tiers never read these (the bloom
  // precedent, RENDER_QUALITY.md "backend-structural absence, no policing").
  const reflBounces = Math.max(0, Math.min(6, Math.round(L.reflBounces ?? 0)));
  const reflectivity = Math.max(0, Math.min(1, L.reflectivity ?? 0.8));
  // NEON (IDEAS 2026-08-21) — emissive surface glow. Same opt-in/latch shape
  // as ENVX/#630: `neon` is the CODEGEN latch both GPU tiers key a shader
  // variant on (renderer.js activeFeat, renderer_gl.js relink — flat formulas
  // only, the iridescence S6 precedent); the 0..1 slider maps to `neonGain`,
  // a pre-tonemap HDR emission gain (×8: full slider pushes a bright palette
  // stop to ~8× the fixed bloom threshold of 1.0). 0 (the default) derives
  // the latch false and the off build emits byte-identical text. CPU/ASCII
  // tier approximates with an LDR albedo brightening (cpu.js shadeRGB — no
  // bloom pipeline there, the RENDER_QUALITY structural-absence precedent).
  const neonAmt = Math.max(0, Math.min(1, L.neon ?? 0));
  // AURORA (ENVX P6, IDEAS 2026-08-21 "Aurora/nebula ENVX layer") — colored
  // fbm sky: curtain + nebula amounts, ONE hue control. Same opt-in/latch
  // shape as ENVX: `aurora` is the CODEGEN latch both GPU tiers key a shader
  // variant on; both amounts 0 (the default) derives false and the off build
  // emits byte-identical text. The hue derives BOTH colors (curtain floor +
  // tip/accent at hue+0.42 — the classic green→purple pairing at the default
  // 0.36); the dormant hue is inert until an amount is up (bandTilt's shape).
  const auroraAmt = Math.max(0, Math.min(1, L.aurora ?? 0));
  const nebulaAmt = Math.max(0, Math.min(1, L.nebula ?? 0));
  const auroraHue = Math.max(0, Math.min(1, L.auroraHue ?? 0.36));
  // THIN FILM (IDEAS 2026-08-21) — angle-driven interference material (the
  // soap-bubble / beetle-shell sheen). NOT the S6 iridescence slider (an
  // orbit-trap modulator on the Glow signal): this one is pure geometry, so
  // the two stack. Same opt-in/latch shape as NEON: `thinFilm` is the CODEGEN
  // latch both GPU tiers key a shader variant on (flat formulas only, the
  // neon/S6 precedent); the 0..1 slider IS the runtime word (the shaders
  // shape the angle weight themselves — no premap). 0 (the default) derives
  // the latch false and the off build emits byte-identical text. CPU/ASCII
  // tier mirrors the modulation in LDR (cpu.js shadeRGB — the ray dir is
  // already on hand there, unlike bloom's structural absence).
  const filmAmt = Math.max(0, Math.min(1, L.thinFilm ?? 0));
  // CLIP (IDEAS "Clipping-plane cross-section" + MRI sweep) — a world-space
  // plane cuts the fractal open; the cut face flat-shades the interior. Same
  // opt-in/latch shape as ENVX/planet: `clip` is the CODEGEN latch both GPU
  // tiers key a shader variant on, and it is an EXPLICIT boolean (clipOn) —
  // unlike the amount-driven features, offset 0 is a meaningful plane
  // position, so presence can't key off a scalar being nonzero. The UI model
  // is deliberately tiny (axis 0..2 + flip + offset); the plane the shaders
  // read is a general vec4 (unit normal + offset along it), derived HERE so
  // both GPU tiers upload identical words and a future arbitrary-normal
  // control needs no shader change. Flip negates the normal AND the plane
  // constant, so the plane stays at the same axis position and only the KEPT
  // half swaps. No scene guard (march geometry — the planet contract). The
  // CPU/ASCII tier mirrors the march in cpu.js traceGrid.
  const clipOn = L.clipOn === true;
  const clipAxis = Math.max(0, Math.min(2, Math.round(L.clipAxis ?? 0)));
  const clipFlip = !!L.clipFlip;
  const clipOffset = Math.max(
    -CLIP_OFFSET_LIMIT,
    Math.min(CLIP_OFFSET_LIMIT, L.clipOffset ?? 0),
  );
  // CLIP JAGGED — slider 0..1 → world amplitude + the Lipschitz divisor (see
  // the CLIP_JAG_* constants above). Amount-keyed presence (like envx): the
  // noised sub-variant exists only while clipOn AND the slider is nonzero,
  // so a flat cut never pays the per-step noise.
  const clipJagAmt = Math.max(0, Math.min(1, L.clipJag ?? 0));
  const clipJagAmp = CLIP_JAG_WORLD * clipJagAmt;
  const clipJagInv = 1 / (1 + CLIP_JAG_LIP * clipJagAmp * CLIP_JAG_FREQ);
  const clipSign = clipFlip ? -1 : 1;
  const clipN = [0, 0, 0];
  clipN[clipAxis] = clipSign;
  // CINE GRADE (IDEAS 2026-08-21 wave) — cinematic post looks. The engine is
  // deliberately look-blind: it receives ONE small numeric parameter block
  // (L.grade, derived app-side by app/src/looks.ts from the named look + the
  // strength slider + the active palette) and `gradeOn` is the POST-pipeline
  // latch both GPU tiers key gated codegen on (WebGPU: the graded post-pass
  // variant; WebGL2: the cineGrade fragment-tail splice). Absent/strength-0
  // derives the latch false and the off build emits byte-identical text
  // (core/gradegate.test.mjs). CPU/ASCII: structural absence, the bloom
  // precedent (RENDER_QUALITY "backend-structural absence, no policing").
  // Tone colors are LINEAR RGB (the grade runs in display-linear, post-
  // tonemap, pre-encode); looks.ts converts its sRGB definitions once.
  const GR = L.grade || null;
  const c01 = (v, def = 0) => Math.max(0, Math.min(1, v ?? def));
  const gradeStrength = c01(GR?.strength);
  const gradeTint = (t, def) =>
    Array.isArray(t) ? [c01(t[0]), c01(t[1]), c01(t[2])] : def;
  const gradeShadow = gradeTint(GR?.shadowTint, [0.5, 0.5, 0.5]);
  const gradeHi = gradeTint(GR?.hiTint, [0.5, 0.5, 0.5]);
  return {
    // CINE GRADE words — gradeA..gradeD (shader.js GRADE_WORD..+3), always
    // written (the ENVX tail-row contract), read only by graded post builds.
    gradeA: [
      gradeStrength,
      c01(GR?.contrast),
      Math.max(0, Math.min(2, GR?.saturation ?? 1)),
      c01(GR?.shadowDesat),
    ],
    gradeB: [...gradeShadow, c01(GR?.splitAmt)],
    gradeC: [...gradeHi, c01(GR?.duoAmt)],
    gradeD: [c01(GR?.vignette), 0, 0, 0],
    gradeOn: gradeStrength > 0,
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
    // NEON arms the bloom composite even at Glow 0 (the floor below), so the
    // emissive actually halos with one slider — the feature FEEDS the existing
    // P3 bloom, it does not build one. Glow keeps full authority above the
    // floor; neon 0 leaves every word exactly as before (byte-stable looks).
    bloomStrength: Math.max(glow, neonAmt * 0.5) * 0.8,
    bloomThreshold: 1.0, // pre-tonemap HDR
    bloomOn: glow > 0 || neonAmt > 0,
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
    // IMGTEX (#631) — see the derivation block above. The half-latches carry
    // the slider state; renderer.js ANDs each with its texture's presence to
    // form the codegen flag (envMap / surfTex).
    emapAmt,
    emapBright: Math.max(0, Math.min(3, L.envMapBright ?? 1)),
    emapRot: Math.max(0, Math.min(1, L.envMapRot ?? 0)),
    triAmt,
    triScale: Math.max(0.02, Math.min(3, L.surfTexScale ?? 1)),
    // #630 self-reflection words (see the derivation block above).
    reflBounces,
    reflectivity,
    reflFresnel: Math.max(0, Math.min(1, L.reflFresnel ?? 0.5)),
    reflTint: Math.max(0, Math.min(1, L.reflTint ?? 0)),
    sreflect: reflBounces > 0 && reflectivity > 0,
    // NEON words (see the derivation block above). neonGain is what the
    // shaders read (WGSL p3ctl.w / GLSL uNeon); `neon` is the codegen latch.
    neonGain: neonAmt * 8,
    neon: neonAmt > 0,
    // AURORA (ENVX P6) words (see the derivation block above). `aurora` is
    // the codegen latch; the colors are the derived floor/tip pair both GPU
    // tiers upload verbatim. Drift is a reserved 0 (no per-frame writer —
    // the starsU.w twinkle precedent); it exists so the standalone bake and
    // the GL uniform surface have one source when a writer arrives. Like the
    // sky macro itself, GPU-tier only (CPU/ASCII renders the legacy
    // screen-space gradient — the bloom precedent).
    auroraAmt,
    nebulaAmt,
    auroraColA: hueRgb(auroraHue, 0.85), // curtain floor (green at default hue)
    auroraColB: hueRgb((auroraHue + 0.42) % 1, 0.6), // tip / nebula accent
    auroraDrift: 0,
    aurora: auroraAmt > 0 || nebulaAmt > 0,
    // THIN FILM words (see the derivation block above). filmAmt is what the
    // shaders read (WGSL morphX.w / GLSL uFilm); `thinFilm` is the codegen
    // latch.
    filmAmt,
    thinFilm: filmAmt > 0,
    // CLIP words (see the derivation block above). clipN/clipW are what the
    // shaders read (WGSL clipU / GLSL uClip); `clip` is the codegen latch.
    // clipW = n·(offset·axis) — flip negates both, so the plane holds its
    // axis position and only the kept half swaps. clipShade is the reserved
    // cut-face brightness gain (clipS.x / uClipS): 1 today, no UI writer —
    // it exists so the word surface has one source when a writer arrives
    // (the auroraDrift precedent).
    clipN,
    clipW: clipSign * clipOffset,
    clipShade: 1,
    clip: clipOn,
    // CLIP JAGGED words (clipS.yzw / GL uClipJ) + the sub-variant latch.
    // Always derived (the tail contract: written every frame, read only by
    // jag variants); clipJagInv is exactly 1 whenever the slider is 0.
    clipJagAmp,
    clipJagFreq: CLIP_JAG_FREQ,
    clipJagInv,
    clipJag: clipOn && clipJagAmp > 0,
  };
}

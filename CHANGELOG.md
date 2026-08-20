# Changelog

All notable changes to the **Fractbox Engine** are documented here. The engine
is versioned [semver](https://semver.org)-style; the canonical version lives in
[`core/version.js`](core/version.js).

- **MAJOR** — breaking change to the op-list JSON, operator keys, or public API
- **MINOR** — new operators or render backends, backward-compatible
- **PATCH** — render-correctness / bug fixes, no API change

This is a one-way mirror; entries track the engine surface (`core/` + demo), not
the private app that builds on it.

## 0.6.0 — 2026-08-20

### Added

- **opAux overflow lane** (`core/uniformPack.js`) — an operator can now
  declare up to **six** numeric params: params beyond the packed inline slots
  ride a per-op auxiliary uniform lane, backward-compatible with every
  existing formula and share link.
- **`ruckerBulb`** (`core/operators.js`) — a trig bulb with its two spin
  angles powered independently, a squashable pole and a choice of four angle
  flavors; the first operator to spend the opAux lane. Operator count is now
  **63**.
- **`bulbAxis` split powers** — gains `ThetaMul`/`PhiMul` on the opAux lane,
  so one op now holds the trig-bulb family's whole axis × convention ×
  split-power cross.
- **ENVX backgrounds** (`core/coloring.js`, both shader tiers) — a starfield
  (`stars`/`starDensity`/`starSeed`), a Milky-Way-style band
  (`band`/`bandTilt`), and an optional `zenith` color replacing the fixed sky
  gradient's zenith. All continuous macros, so they animate.
- **Eight new presets** — the catalogue is now **102** (was 94), including a
  six-preset category-balance wave.

### Changed

- **Fallback-ladder resilience** (`core/renderpolicy.js`, `core/renderer.js`)
  — a lost WebGPU device now falls down the render ladder instead of
  freezing; watchdog-hung buffer maps time out and recover the way hung
  fences already did; the detail governor's entry clamp survives the
  post-load reframe.
- **Deep-zoom flythroughs reach the perturbation tier** (`core/preview.js`,
  `core/perturb.js`) — continuous camera pushes preserve the
  extended-precision reference orbit instead of resetting it.
- **Splat capture** (`core/splatcapture.js`) — close-up crops march at the
  crop's own iteration count, and `onSurface`'s r0 is floored against the
  capture epsilon, so deep captures verify.

### Fixed

- **GLSL parity: kaleido `Mirror` mode** was dead under WebGL2
  (string-vs-number param compare).
- **Hex grid leaf** — the cell lattice was rhombic, not hexagonal.
- **Heart / stairs leaves** — heart upright, flat stair treads, floored
  plateau.
- **Orthographic zoom** — wheel/pinch/keyboard zoom now actually zooms under
  an orthographic camera.

## 0.5.0 — 2026-08-07

### Added

- **Twenty-one new operators** (`core/operators.js`), all backward-compatible —
  operator count is now **62** (was 41): `riemannBulb`, `kleinPolyMap`,
  `magnetXYZ`, `magnetXYZAbs`, `makinTri`, `makinFuzzy`, `polygonFold`,
  `toCoord`/`fromCoord` (curvilinear frame-change pair, incl. log-polar),
  `gnarl2D`, `gnarl3D`, `asinhWarp`, `logWarp`, `neoSqrWarp`, `sinShear`,
  `smoothBoxFold`, `smoothBallFold`, `torusInvert`, `mandalayFold`,
  `brickFold`, `complexMap` (parametrized Möbius/Cayley-family map).
- **N-slot hybrids** (`core/hybridmodel.js`, new) — a hybrid formula can now
  interleave up to **four** op-list slots on a repeating schedule (was two).
  `hybridSlots(f)` is the single canonical reader for both the legacy 2-slot
  shape and the new `slots[]` shape, so every consumer (sanitize, the share
  codec, `vary.js`'s jitter, DE-family resolution) reads one implementation.
- **Tiled/off-axis export** (`core/tilegrid.js`, new) — pure camera-window math
  for splitting a render into seam-exact tiles at an arbitrary target
  resolution (poster/print-size stills), independent of any GPU or DOM so it's
  unit-testable without a renderer.
- **Perturbation deep zoom** (`core/perturb.js`, new) — an extended-precision
  (BigInt fixed-point) reference orbit plus an f32 delta kernel that lets the
  GPU march a small residual around the reference instead of losing precision
  directly, pushing usable zoom depth well past the existing df64 tier.
- **Formula JSON v1 — a normative interchange spec** (`docs/spec/FORMULA_JSON.md`,
  new). Every Export/Import surface reads and writes this documented shape
  (flat / scene / hybrid), including which conditions are hard errors versus
  silent, warned repairs (clamping, truncation, padding).
- **`core/validate.js`** (new) — the spec above in executable, zero-dependency
  form: `validate(doc)` never throws and never parses text, and predicts
  exactly what the importer (`sanitize.js`) will accept — asserted equivalent
  to it over the full preset catalogue plus a generated mutation corpus.

### Changed

- Import hardening: several caps moved from "validate then discard" to
  "discard then validate" (truncate-then-validate, applied uniformly across
  flat ops, scene objects/counts, and hybrid slots/counts) — content past a
  cap is now dropped unread rather than inspected first.
- WebGL2 fallback tier fitted to the GLES minimum uniform budget (std140 UBO
  packing) after regressing to non-functional on minimum-spec devices.

## 0.4.0 — 2026-07-06

### Added

- **Nine new operators** (`core/operators.js`), all backward-compatible —
  operator count is now **41** (was 32):
  - `varyScale` — radial-power sphere fold (Amazing-Surf-style scale variation).
  - `bristorBrot` — Bristorbrot triplex square, the first **numeric-DE** map.
  - `newtonTri2` / `newtonTri3` — Newton triplex z² / z³ iterations.
  - `msltoeSym3` — Msltoe Sym z² symmetric bulb (with a sign-rule variant param).
  - `sphericalTwoStage` — two-angle spherical bulb power.
  - `boxBulb` — box-fold-flavored bulb power.
  - `slonoBrot2` — SlonoBrot triplex map.
  - `scaleDrift` — per-iteration drifting scale (Amazing Surf `Scale_vary`).
- **Numeric finite-difference DE path** — escape-time maps with no analytic
  derivative (the triplex `bulb_numeric` family above) get their distance
  estimate from a finite-difference gradient instead of a tracked `w`.
  `operators.js` exports `isNumericDE(formula)` and `effectiveDeOption(formula)`
  (numeric formulas resolve to `deOption 3`); `shader.js buildWGSL({numericDE})`
  emits the finite-difference gradient only when a formula needs it. Present on
  every backend (WebGPU / WebGL2 / CPU).
- **Formula variation + a soundness oracle** (`core/vary.js`, new) — the engine
  side of "Surprise / Remix": `jitterParams(formula, {spread})` nudges every op
  within its declared operator range; `isSound(formula)` scores a candidate on
  the CPU (via `evaluate.measure`, over a probe-region ladder) and rejects the
  degenerate rolls that render blank (collapsed / space-filling / all-escaping
  IFS); `soundCandidate(make, fallback)` is the generate-and-test loop.
  `core/random.js`'s `randomFormula()` (Surprise) now draws across the full
  operator set through this gate. **`core/evaluate.js` now ships** (it was
  previously omitted as game-only) because the generator depends on its
  `measure` / `surfaceLean`.
- **Camera navigation** (`core/gestures.js`, `core/cruise.js`, both new; wired
  through `preview.js`): cursor-anchored delta-proportional **wheel zoom**,
  **orbit inertia** (flick to coast, exponential decay), **two-finger pan +
  pinch-zoom** on touch, and **DE-scaled cruise** — hold to fly into the
  fractal, the step size scaling with the distance-estimate so the approach
  never overshoots the surface.
- **Demo: live ASCII mode.** With neither WebGPU nor WebGL2 available the demo
  no longer dead-ends on an error card — it runs the same presets through the
  CPU backend (`core/cpu.js`) as live, spinning, drag-orbitable colored ASCII
  (the diagnostic note stays, as a corner card). Force it on any machine with
  `?ascii=1`. README gains a "render fractals as text" section (node ANSI
  one-liner + HTML embed).

### Changed

- **Engine core owns no DOM** (#77). `preview.js` is now a pure render
  controller — PNG export returns a `Blob` and thumbnails render to data URLs;
  the download `<a>` and the clickable thumbnail grid moved to
  `core/preview-dom.js`, a thin glue layer. Callers of `exportPNG` /
  `renderThumbnails` are unchanged.
- Flat-formula operator cap unified to **64** on every backend for cross-tier
  parity (`core/limits.js` single-sources the capacity caps).
- Operator count is now **41** (was 32).

## 0.3.0 — 2026-07-03

### Added

- **Offline frame capture** (`preview.captureFrame({w, h, quality})` →
  `ImageBitmap`, plus `preview.setOffline(on)`): renders one frame at a chosen
  size into an **offscreen texture** and reads it back deterministically
  (`renderer.renderToImage`) — never the presented swap-chain canvas, whose
  double-buffered readback alternates stale/uninitialized frames. `setOffline`
  suspends the live pump so an offline render loop owns the device. Powers the
  app's video/GIF export; useful for any headless frame capture.
- **Formula morph** (`preview.setMorph(target, t, swell)` — WebGPU only):
  blends two plain formulas' distance fields, `d = mix(dA, dB, t)` — a convex
  blend of two 1-Lipschitz bounds is itself a valid bound, so sphere tracing
  stays safe. Each orbit runs with its **own bailout** (sharing one overflows
  a power-8 escape orbit's derivative in f32). `swell` is a peak mid-blend
  dilation (still DE-safe) that counteracts level-set erosion where the two
  fields don't overlap. Orbit-trap and escape-band coloring blend the metrics
  of **both** formulas, so the surface pattern morphs continuously too.
  Backends without the morph writer (WebGL2/CPU) render the current formula
  unchanged — the fallback is structural, not policed.
- **Coloring-mode crossfade** (`preview.setColorBlend({t, modeB, palOnB})` —
  WebGPU only): shades under two color modes / palette toggles and mixes the
  albedos, for callers interpolating between looks whose mode enums can't
  lerp. Off (null) → the legacy shade path, byte-identical.

### Changed

- WGSL `Globals` uniform grew 320 → 384 bytes (four new vec4 words:
  `morphB`/`morphT`/`morphX`/`colorX`). Internal layout only; all zero ⇒
  every legacy path renders byte-identically.
- Security/correctness hardening pass across `sanitize.js`, `sharecodec.js`,
  `exporter.js`, `glslImport.js`, and `renderer_gl.js` (input validation and
  bounds discipline; no API change).

### Fixed

- Hybrid deScale tightened for escape-slot + fold-only-slot combinations that
  could overstep the surface (`stability.js`, with regression tests).

## 0.2.1 — 2026-07-02

### Fixed

- Dual-set `objects` + `hybrid` formulas now dispatch **objects-first on every
  tier**: `preview.js` `writeFrame` (feeding both GPU backends) flipped to
  match `sanitize.js`/`cpu.js`, and `cpu.js`'s hybrid coloring branch gained
  the same objects-first gate. Behavior change only for malformed input — the
  op-list format forbids carrying both. An `objects: []` + `hybrid` formula no
  longer falls through to `writeScene([])` (an uncaught throw in the render
  loop).

## 0.2.0 — 2026-06-29

### Added

- **Seven new operators** (`core/operators.js`), all backward-compatible:
  - `menger` — smoothed Menger fold with signed-smoothness modes (rounded/organic edges).
  - `polyAngleFold` — N-fold polar angle fold (symmetry / angle / mirror).
  - `cylinderFold` — radius-bounded cylindrical fold (DE-tracked, `w ×k`).
  - `radialInvert` — spherical inversion about a shiftable center.
  - `bulbAxis` — Mandelbulb power around a selectable axis.
  - `hexFold` — hexagonal plane fold.
  - `absXYZ` — per-axis absolute-value fold (independent X/Y/Z toggles).
- **WebGL2 renderer backend** (`renderer_gl.js` + `shader_gl.js`) — a full-parity
  fallback below WebGPU, so the engine renders where WebGPU is unavailable.
- **CPU / colored-ASCII renderer** (`cpu.js`) — a GPU-free last-resort backend
  (also opt-in for testing), aligned to the GPU camera/aspect.
- `core/version.js` — the engine version constant (this file's source of truth).

### Changed

- **BREAKING:** the `roundMenger` operator key was renamed to **`menger`**. Op-lists
  that serialized `roundMenger` must be updated to `menger`. (Pre-1.0; called out
  rather than forcing a major.)
- Operator count is now **32** (was 25).

### Fixed

- **Color correctness:** albedo is now linearized from sRGB before lighting, so the
  render matches the GUI swatches. ([#6])
- Octahedral fold ships a working **Octahedron** preset; documented that `octaFold`
  needs a following Scale + Translate to render. ([#7])
- `evaluate.js` now gates the `+c` term on `addC || julia`, matching the renderer.

## 0.1.0 — 2026-06-26

Initial public release of the mirror: the operator-IR engine (`core/`), the
standalone WebGPU demo, the guided tour, and the tutorial. 25 operators.

[#6]: https://github.com/fractbox/fractbox-engine/issues/6
[#7]: https://github.com/fractbox/fractbox-engine/issues/7

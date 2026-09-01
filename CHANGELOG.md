# Changelog

All notable changes to the **Fractbox Engine** are documented here. The engine
is versioned [semver](https://semver.org)-style; the canonical version lives in
[`core/version.js`](core/version.js).

- **MAJOR** — breaking change to the op-list JSON, operator keys, or public API
- **MINOR** — new operators or render backends, backward-compatible
- **PATCH** — render-correctness / bug fixes, no API change

This is a one-way mirror; entries track the engine surface (`core/` + demo), not
the private app that builds on it.

## 0.7.0 — 2026-09-01

Every render feature below is **codegen-gated**: its off state emits
byte-identical shader text (never a uniform-gated dead branch), so existing
formulas pay exactly nothing.

### Added

- **Five new operators** (`core/operators.js`) — operator count is now **68**
  (was 63): `mirrorShells` (nested concentric mirror shells — a bounding
  fold, like `sphereFold`), `spiralVortex` (logarithmic spiral wind around an
  axis, DE-charged), `wallpaperFold` / `spaceGroupFold` (crystallographic
  reflection chambers — a 2-D wallpaper tile in x,y and a full 3-D
  space-group cell), and `hingeFold` (an angled cut-plane fold that reports
  its tear through the new seam channel).
- **Seam channel** (`core/operators.js`, all three march tiers) — ops whose
  folds tear space (`hingeFold`, `modFold`) report a running-min tear
  distance and the march clamps its step against it — floored at d/4 and
  band-gated so it only brakes where the bound is actionable — so fold
  membranes render instead of being stepped through. Formulas without seam
  ops emit seam-free shader text, byte-identical to before.
- **`city` shape leaf** (`core/leaves.js`) — a procedural building lattice;
  the first leaf to spend the objAux lane below. Leaf count is now **59**.
- **objAux overflow lane** (`core/limits.js`, `core/sharecodec.js`) — a
  shape leaf can now declare up to **eight** numeric params (was 4),
  mirroring 0.6.0's opAux lane: overflow rides a pay-per-use storage lane
  (WGSL) / a second uniform array (GLSL) that fat-leaf-free scenes never
  declare. The share codec gains the sibling tag `SHAPES2`; old decoders
  skip it by container length, and ≤4-param scenes' links stay
  byte-identical.
- **Neon emissive** (`core/shader.js`, both shader tiers) — an HDR emissive
  term driven by the coloring signal: albedo × gain × signal², bright enough
  to cross the bloom threshold and halo through the existing bloom
  composite. Flat formulas in v1.
- **Aurora / nebula backgrounds** (ENVX P6, both shader tiers) — colored fbm
  sky behind the fractal: aurora curtains plus nebula clouds, composited
  behind the surface and animatable like the 0.6.0 ENVX macros.
- **Thin-film interference material** (both shader tiers) — angle-driven
  spectral sheen (phase ∝ cos(view, normal)), the soap-bubble / beetle-shell
  look; its bands sweep as the camera orbits. Flat formulas in v1.
- **CINE grade post-pass** (`core/shader.js buildPostWGSL({grade})`, GLSL
  export included) — a data-driven display-referred color transform
  (S-curve, shadow-weighted saturation, split-tone, duotone, per-look
  vignette) applied post-tonemap; looks are parameter sets, never emitter
  forks, and `grade: false` emits the previous post shader byte-for-byte.
- **Self-reflection** (`core/shader.js buildWGSL({sreflect})`, WebGPU) —
  marched mirror bounces: the fragment loop re-marches the reflected ray
  from each hit, Fresnel-weighted, with one shared shading body serving
  every bounce.
- **User environment map + triplanar surface texture** (`core/envmap.js`
  new, WebGPU) — a user equirect image replaces the procedural sky
  (background, IBL ambient, and reflections all see it), and a repeat-tiled
  image blends over the albedo via sharpened triplanar projection.
  `core/envmap.js` carries the pure math (size budget, blend weights),
  pinned against the emitted WGSL.
- **Tiny-planet + 360° equirect projections** (both shader tiers) — two new
  ray generators: an inverse-stereographic "little planet" that folds the
  sphere of directions into a disc, and a mono 360×180 lat-long emitter
  whose saved 2:1 frame is a 360 photo (tile-exact, so it composes with the
  tiled export). Mutually exclusive with each other and with orthographic;
  `preview.setPlanet(fovDeg)` / `preview.setEquirect(on)`.
- **Clipping plane + jagged cut** (all three march tiers) — a world-space
  plane cuts the fractal open as a CSG intersection inside the march itself
  (the plane term is exact, so the clipped march is precisely as DE-safe as
  the plain one); rays landing on the plane inside the solid shade a flat
  cross-section face from the interior coloring signal. The jagged variant
  erodes the cut with static world-space value noise under a Lipschitz
  correction that keeps the term a valid lower bound.
- **Field streamlines** (`core/streamlines.js`, new — WebGPU only) —
  luminous particles advected along the distance field's gradient by a
  compute pass that reuses the emitted march DE verbatim (the module text is
  sliced at `mapDE`, never hand-copied), composited additively over the
  frame. Deterministic offline exports reseed and pre-roll the field from a
  caller-supplied key. `preview.setStreamlines(o)` reports `false` on the
  other tiers.
- **`preview.setOnPresent(cb)`** — a synchronous same-task present hook,
  fired after each live presenting draw before any await: the one moment a
  WebGPU swap-chain canvas is readable from 2D `drawImage`. Offline /
  export / capture draws never fire it; unset cost is one null check.
- **Seven new presets** — the catalogue is now **109** (was 102), including
  a showcase wave for the new operators (Whorl Citadel, Hinged Bastion,
  City Blocks, Bravais Cage, Jewel Wallpaper, …).

### Changed

- **Idle refinement converges heavy scenes** (`core/preview.js`,
  `core/renderer.js`) — refine ticks now march in the settle's own scissored
  ~60 ms bands (no more single-submit compositor stalls), the refinement
  ceiling admits any settle the resolution policy allows to exist, and a
  governed view that has settled runs one ungoverned banded upgrade so idle
  frames land at the resolution the user actually sees — the perf governor
  no longer punishes quality time as a perf violation.
- **Per-op DE audit** (`core/deaudit.test.mjs`, new) — every operator's
  declared derivative rule is now gated against its true Jacobian
  (Lipschitz / operator-norm analysis, exact and approximate tiers), so a
  DE overstatement fails the suite instead of piercing surfaces at render
  time.
- **Standalone GLSL bake** (`core/exportStandalone.js`) — exports carry the
  neon / aurora / thin-film surfaces when the look uses them, uploading
  exactly what the live GL tier uploads.
- Internal: the `Globals` uniform grew per-variant tail rows (ENVX at 48,
  image textures, aurora, grade, clip — allocation ceiling 62 vec4s, with a
  written ledger in `core/shader.js`); variants not using a tail carry it as
  dormant padding, and every legacy variant's struct is unchanged.

### Fixed

- **Palette phase and iridescence were exact no-ops on WebGPU** — the
  renderer's uniform pack read a hand-retyped subset of its argument bag
  that omitted both names, so the values silently defaulted; WebGL2 passed
  the whole payload and was correct all along. The payload now flows whole,
  and the handoff shape is pinned by test (`core/uniformbag.test.mjs`).
- **Auto-levels revived on WebGPU** — the other casualty of the same
  dropped-payload bug: the flagship tier discarded the computed signal range
  and drew the raw color signal while WebGL2 and CPU normalized. The tiers
  now pack the identical range and agree.
- **`varyScale` / `twist` / `cylinderFold` DE declarations** — the audit
  above caught all three overstating their derivative: `varyScale` now
  charges its exact non-conformal Jacobian (Vary Box resolves at high
  RPower), and `twist` / `cylinderFold` are reclassified approximate-DE
  (×0.5 step, ×2 budget) — pierced-surface renders resolve on every tier.
- **WebGL2 fat-leaf scene compile** — the scene fragment emitter omitted
  the objAux uniform declaration its own leaf call sites referenced, so
  every scene carrying an 8-param leaf failed to compile on the GL tier
  (`core/shader_gl.js`, `core/renderer_gl.js`; uses-implies-declares is now
  pinned in the emitted text).
- **Thumbnail-coloring race** — two concurrent thumbnail renders could park
  a thumbnail's look as the session coloring (every later frame rendered in
  a random tile's colors until reload). The tile look is now threaded
  through the render call, never swapped through shared state.

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

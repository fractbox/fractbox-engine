# Fractbox Engine

A composable, distance-estimated **3D fractal engine** for the browser. You build
a fractal by ordering small **primitive operators** (box fold, sphere fold, scale,
rotate, kaleido, Mandelbulb power, …) into a formula; the engine packs that formula
into a GPU buffer and renders it live with **WebGPU**.

**No build step. No dependencies.** `core/` is pure ES modules — drop it on any
static host and `import` it. There is nothing to compile.

> ### ▶ [Live demo](https://fractbox.github.io/fractbox-engine/)
>
> Drag to orbit · scroll to zoom · pick a preset. Needs a WebGPU-capable browser
> (recent Chrome/Edge, or Safari Technology Preview) with a GPU. New here? Hit
> **▶ Guided tour** in the demo, or read the step-by-step
> [**TUTORIAL.md**](TUTORIAL.md).

> [!NOTE]
> **This is a read-only mirror.** The engine is developed in a private monorepo and
> published here one-way. Bug reports are very welcome via
> [Issues](https://github.com/fractbox/fractbox-engine/issues); pull requests are
> automatically closed because changes can't flow back upstream through this mirror.
> See [CONTRIBUTING.md](CONTRIBUTING.md).

## The idea that makes it work

Most fractal renderers hand-write one big distance-estimator shader per formula.
Fractbox instead treats a formula as **data** — an ordered op-list — and keeps a
single **operator IR** as the source of truth. Each operator declares its
parameters, its WGSL interpreter body, *and* a GLSL emitter, all in one place
(`core/operators.js`). Adding a new primitive means adding one entry.

The part worth stealing is the **distance-estimate bookkeeping**. A raymarched
fractal needs a valid distance estimate (DE), and naively composing transforms
breaks it. Each operator instead declares how it affects the running derivative
`w`:

| `wRule`       | meaning                              | examples                    |
| ------------- | ------------------------------------ | --------------------------- |
| `unchanged`   | isometry, `\|Jacobian\| = 1`         | box fold, rotations, abs    |
| `mul_scale`   | conformal scale, `w ×\|scale\|`      | scale                       |
| `mul_k`       | radius-bounded fold, `w ×k`          | sphere fold                 |
| `bulb`        | escape-time power, tracks analytic `dr` | Mandelbulb power         |

Because every primitive carries its own `w` rule, **arbitrary compositions stay a
correct distance estimate** with no global re-derivation. That's why you can stack
folds, scales, and rotations in any order and still get a crisp raymarch.

The same IR drives two emitters — the live **WGSL interpreter** and a standalone
**GLSL exporter** (`iterateJIT_`) — kept side by side so a divergence between them
is a bug you can catch by eye.

## Quick start

```html
<canvas id="view" style="width:100vw;height:100vh"></canvas>
<script type="module">
  import { createPreview } from "./core/preview.js";
  import { PRESETS, clone } from "./core/oplist.js";

  const preview = await createPreview(document.getElementById("view"), {
    camera: PRESETS[0].camera,
  });
  preview.setFormula(clone(PRESETS[0])); // a Mandelbox-family preset
  preview.setAutoRotate(true);
</script>
```

That's the entire integration. `createPreview` owns the renderer, camera,
orbit/zoom/pinch gestures, quality tiers, and PNG/thumbnail export.

### Composing a formula by hand

A formula is plain JSON — an ordered op-list:

```js
const myBox = {
  name: "My Box",
  addC: true, // re-add the seed point each iteration (escape-style)
  iters: 12,
  deOption: 2, // analytic IFS distance estimate, r/|w|
  ops: [
    { key: "boxFold", values: [1.0] },
    { key: "sphereFold", values: [0.5, 1.0] },
    { key: "scale", values: [2.0] },
    { key: "rotateXY", values: [14.0] }, // degrees
  ],
  camera: { yawDeg: 35, pitchDeg: 22, dist: 24, fovDeg: 42 },
};
preview.setFormula(myBox);
```

There are **62 operators**. List them at runtime:

```js
import { OPERATORS } from "./core/operators.js";
console.log(OPERATORS.map((o) => `${o.key}(${o.params.length})`));
```

### Exporting a standalone shader

```js
import { glslFor } from "./core/exporter.js";
console.log(glslFor(myBox)); // engine-conformant iterateJIT_ GLSL
```

### Formula JSON — the interchange format, plus a validator

A formula is portable JSON — every Export/Import button in the app reads and
writes exactly the shape above. [`docs/spec/FORMULA_JSON.md`](docs/spec/FORMULA_JSON.md)
specifies it normatively (flat / scene / hybrid shapes, ranges, caps, error vs.
warning conditions), and `core/validate.js` is that spec in executable,
zero-dependency form — hand it anything and it predicts, without throwing or
parsing, whether the importer will accept it:

```js
import { validate } from "./core/validate.js";

const report = validate(JSON.parse(text));
// → { ok: boolean, errors: [ { severity, code, message, where? } … ] }
```

### Hybrids and tiled export

A **hybrid** interleaves up to four op-list "slots" on a repeating schedule
(`core/hybridmodel.js`) — e.g. alternate a Menger fold with a Mandelbulb power
every other iteration. For output larger than one frame, `core/tilegrid.js`
computes the off-axis camera windows for a seam-exact tiled render, so a
formula can be exported at poster/print resolution one tile at a time.

### No GPU? Render fractals as text

The engine carries a full CPU backend (`core/cpu.js`) that traces the same
distance-estimated formulas into **colored ASCII** — no GPU, no DOM, no browser.
Print a fractal straight into a truecolor terminal from plain node:

```js
// node -e "$(cat this-snippet)"  — zero dependencies, zero build
import("./core/cpu.js").then(async ({ renderAsciiAnsi }) => {
  const { PRESETS } = await import("./core/oplist.js");
  const menger = PRESETS.find((p) => p.name === "Menger");
  console.log(renderAsciiAnsi(menger, { cols: 80, rows: 40 }).ansi);
});
```

The same module renders color-run HTML for embedding in a page with zero GPU
requirements (`renderAsciiColored(formula, opts).html` → drop into a `<pre>`),
plus plain-text output, silhouette-edge glyphs (`edges: true`), orbit-trap
structure isolines (`structure: true`), dithering, and a choice of calibrated
glyph ramps. The [live demo](https://fractbox.github.io/fractbox-engine/)
falls back to this backend automatically when neither WebGPU nor WebGL2 is
available — or force it with
[`?ascii=1`](https://fractbox.github.io/fractbox-engine/?ascii=1).

## What's in `core/`

| file            | role                                                          |
| --------------- | ------------------------------------------------------------- |
| `operators.js`  | the operator IR — opcodes, params, WGSL + GLSL emitters       |
| `oplist.js`     | the formula shape + a starter gallery of presets              |
| `shader.js`     | the DE/raymarch scaffolding and GLSL export                   |
| `renderer.js`   | the WebGPU device/pipeline + op-buffer packing                |
| `preview.js`    | high-level controller: camera, gestures, quality, export      |
| `camera.js`     | orbit camera math                                             |
| `coloring.js`   | preview shading (surface / orbit-trap / escape bands, palettes) |
| `exporter.js`   | op-list JSON + GLSL + share-link codecs                        |
| `renderer_gl.js`, `shader_gl.js` | WebGL2 fallback backend (full parity below WebGPU)   |
| `cpu.js`        | GPU-free CPU backend — colored ASCII as text, HTML, or 24-bit ANSI |
| `vary.js`, `evaluate.js` | formula variation + the CPU soundness oracle behind `randomFormula` |
| `gestures.js`, `cruise.js` | wheel-zoom/inertia/pinch math + DE-scaled hold-to-fly |
| `hybridmodel.js` | canonical reader for hybrid "slots" — 2-slot legacy shape + N-slot (up to 4) |
| `tilegrid.js`   | off-axis tile geometry for seam-exact stitched high-res export              |
| `perturb.js`    | perturbation deep zoom — extended-precision reference orbit + f32 delta kernel |
| `validate.js`   | Formula JSON validator — the executable half of `docs/spec/FORMULA_JSON.md` |
| `version.js`    | the engine version constant (`ENGINE_VERSION`)               |
| `invariants.js`, `sanitize.js`, `random.js`, `library.js`, `glslImport.js` | validation, random formulas, helpers |

## Running the demo locally

WebGPU needs a secure context, so serve over `http://localhost` (file:// won't
work). No build — any static server will do:

```sh
python3 -m http.server 8000
# then open http://localhost:8000
```

## Versioning

The engine is versioned semver-style; the canonical version is `ENGINE_VERSION`
in [`core/version.js`](core/version.js). Release notes live in
[CHANGELOG.md](CHANGELOG.md).

```js
import { ENGINE_VERSION } from "./core/version.js";
```

## License

[MIT](LICENSE) © 2026 Vladimir Weinstein.

The fractal math is standard, community-published distance-estimation technique.
With thanks to the people who worked it out and wrote it down: **Tom Lowe (Tglad)**
and **Rudy Rucker** (Mandelbox), **Daniel White** and **Paul Nylander** (Mandelbulb),
**Knighty** (pseudo-Kleinian / KIFS plane folds), and **Iñigo Quilez** (cosine
palettes, raymarching writeups). Naming them is courtesy, not obligation — the math
itself isn't anyone's property.

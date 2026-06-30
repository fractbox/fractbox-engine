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

There are **32 operators**. List them at runtime:

```js
import { OPERATORS } from "./core/operators.js";
console.log(OPERATORS.map((o) => `${o.key}(${o.params.length})`));
```

### Exporting a standalone shader

```js
import { glslFor } from "./core/exporter.js";
console.log(glslFor(myBox)); // engine-conformant iterateJIT_ GLSL
```

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
| `cpu.js`        | GPU-free CPU / colored-ASCII last-resort backend              |
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

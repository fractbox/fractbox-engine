# Tutorial: build a fractal from primitives

This walks you through composing 3D fractals with the Fractbox engine, from a
single fold to a full Mandelbox and beyond — and explains *why* each piece does
what it does. By the end you'll understand the op-list model well enough to
design your own formulas.

No build step is needed anywhere. Serve the folder over `http://localhost` (so
the secure-context requirement for WebGPU is met) and open it in a WebGPU
browser:

```sh
python3 -m http.server 8000   # then open http://localhost:8000
```

## 1. The mental model

A **formula** is an ordered list of **operators**. To shade a pixel, the engine
marches a ray into the scene; at each sample point it runs the point through the
operator list, **iterated** several times, and measures how the point grows.
That measurement is the *distance estimate* (DE) — how far the ray can safely
step. Raymarching + a good DE is what turns a piece of math into a surface.

A formula is just data:

```js
const formula = {
  name: "My Box",
  addC: true, // re-add the original point each iteration (escape-time style)
  iters: 12, // how many times the op-list runs per sample
  deOption: 2, // which distance estimate to use (2 = analytic IFS, r/|w|)
  ops: [
    /* the ordered operators go here */
  ],
  camera: { yawDeg: 35, pitchDeg: 22, dist: 24, fovDeg: 42 },
};
```

The two `deOption` values you'll use:

- **`2` — analytic IFS DE** (`r / |w|`): for fold/scale fractals (Mandelbox,
  Menger, Kleinian). Fast and crisp.
- **`0` — escape-time DE** (`0.5·ln(r)·r/dr`): for power fractals like the
  Mandelbulb.

## 2. Render an empty scene

Wire up the engine and hand it a formula. This is the entire integration:

```html
<canvas id="view" style="width: 100vw; height: 100vh"></canvas>
<script type="module">
  import { createPreview } from "./core/preview.js";

  const preview = await createPreview(document.getElementById("view"), {
    camera: { yawDeg: 35, pitchDeg: 22, dist: 24, fovDeg: 42 },
  });

  const formula = { name: "My Box", addC: true, iters: 12, deOption: 2, ops: [] };
  preview.setFormula(formula);
  preview.setAutoRotate(true);
</script>
```

With no operators, nothing renders — the point never folds back on itself, so
there's no attractor. Let's give it some moves.

## 3. The Mandelbox in three moves

The Mandelbox is the "hello world" of distance-estimated fractals. It's three
operators, applied in order, every iteration.

### Move 1 — Box Fold (reflect)

```js
ops: [{ key: "boxFold", values: [1.0] }]; // FoldLimit = 1.0
```

The box fold reflects any coordinate that strays past `±FoldLimit` back toward
the center: `x → 2·FoldLimit − x`. It's a pure reflection, so it doesn't change
volume — important for the DE (see §6). On its own it just tiles space; the
magic comes from combining it with the next move.

### Move 2 — Sphere Fold (bound)

```js
ops: [
  { key: "boxFold", values: [1.0] },
  { key: "sphereFold", values: [0.5, 1.0] }, // MinRadius, FixedRadius
];
```

The sphere fold is a radius-bounded inversion: points inside `MinRadius` get
pushed out, points between `MinRadius` and `FixedRadius` get scaled up. This is
what keeps the fractal **bounded** — without it, the scale in Move 3 would fling
everything off to infinity and you'd render blank sky.

### Move 3 — Scale (expand)

```js
ops: [
  { key: "boxFold", values: [1.0] },
  { key: "sphereFold", values: [0.5, 1.0] },
  { key: "scale", values: [2.0] }, // ×2
];
```

The scale blows the folded, bounded point up by 2×. Iterate that — fold, bound,
expand, fold, bound, expand — and the competition between expansion and the two
folds carves out the Mandelbox's infinite filigreed shell.

That's a complete fractal. Set it and spin:

```js
preview.setFormula({
  name: "Mandelbox",
  addC: true,
  iters: 12,
  deOption: 2,
  ops: [
    { key: "boxFold", values: [1.0] },
    { key: "sphereFold", values: [0.5, 1.0] },
    { key: "scale", values: [2.0] },
  ],
  camera: { yawDeg: 35, pitchDeg: 22, dist: 24, fovDeg: 42 },
});
```

Try nudging the values: `FoldLimit` 0.8–1.5 changes the cell size; `scale`
2.0–3.0 makes it finer and more filigreed; a negative scale (`-1.5`) gives a
"Mandelbox −" with a totally different character.

## 4. Add character with rotations and symmetry

Once you have the box core, extra moves restyle it.

A **rotation between iterations** twists the whole structure into a screw — this
is the "Tourbillon":

```js
ops: [
  { key: "boxFold", values: [1.0] },
  { key: "sphereFold", values: [0.5, 1.0] },
  { key: "scale", values: [2.0] },
  { key: "rotateXY", values: [14.0] }, // degrees, applied each iteration
  { key: "rotateYZ", values: [7.0] },
];
```

A **kaleidoscope fold** imposes N-fold mirror symmetry:

```js
{ key: "kaleido", values: [5.0, 0.0] } // 5-fold symmetry, no twist
```

Operator order matters: the same operators in a different sequence make a
different fractal. The op-list *is* the formula.

## 5. A different family — the Mandelbulb

Fold fractals are IFS (`deOption: 2`). Power fractals are escape-time
(`deOption: 0`) and use a single operator:

```js
preview.setFormula({
  name: "Mandelbulb",
  addC: true,
  iters: 8,
  deOption: 0, // escape-time DE
  ops: [{ key: "mandelbulbPower", values: [8.0] }], // z → z⁸ + c
  camera: { yawDeg: 35, pitchDeg: 12, dist: 5.0, fovDeg: 42 },
});
```

Turn it into a **Juliabulb** by fixing the seed instead of re-adding the sample
point:

```js
{ name: "Juliabulb", addC: true, iters: 9, deOption: 0,
  julia: true, juliaC: [0.35, 0.30, -0.20],
  ops: [{ key: "mandelbulbPower", values: [8.0] }], camera: { /* … */ } }
```

## 6. Why arbitrary stacks stay a valid fractal

Here's the idea that makes the engine composable. A correct DE needs to track
how much space has been stretched — the running derivative `w`. Every operator
declares its effect on `w`:

| operator kind          | effect on `w`     | why                              |
| ---------------------- | ----------------- | -------------------------------- |
| box fold, rotation, abs | unchanged         | reflections/rotations preserve volume |
| scale                  | `× \|scale\|`     | uniform expansion                |
| sphere fold            | `× k`             | the inversion's local factor     |
| Mandelbulb power       | tracks `dr`       | flips to the escape-time DE      |

Because each primitive carries its own rule, you can stack folds, scales, and
rotations **in any order** and the engine still produces a correct distance
estimate — no per-formula derivation. That's the whole reason the op-list model
works. (You can read each rule in `core/operators.js` as `wRule`.)

## 7. Coloring

Coloring is a preview concern, separate from the formula:

```js
import { defaultColoring } from "./core/coloring.js";

const c = defaultColoring();
c.mode = 2; // 0 = surface normal, 1 = orbit trap, 2 = escape bands
c.palette.on = true; // enable the IQ cosine palette
preview.setColoring(c);
```

## 8. Export a standalone shader

Any formula can be emitted as engine-conformant GLSL:

```js
import { glslFor } from "./core/exporter.js";
console.log(glslFor(formula)); // a self-contained iterateJIT_ body
```

## 9. Where to go next

- **All 25 operators**, with parameters, live in `core/operators.js`
  (`import { OPERATORS } from "./core/operators.js"`).
- **The starter gallery** in `core/oplist.js` (`PRESETS`) is 16 worked examples —
  read them as recipes.
- **Random formulas:** `import { randomFormula } from "./core/random.js"`.
- Want a full editor instead of code? The hosted app at
  [fractbox.com](https://fractbox.com) composes the same op-lists with a UI.

Now go fold some space.

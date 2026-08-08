# Fractbox Formula JSON — interchange format, version 1

**Status:** normative, v1 · **Implementation:** `core/validate.js` (validator),
`core/sanitize.js` (importer), `core/exporter.js` (`stripForExport`, the writer)
· **Licence:** MIT, like the engine it describes.

A Fractbox _formula_ is an ordered list of primitive operators applied to a
point once per iteration, plus the handful of settings needed to march and frame
the result. This document specifies the JSON encoding of that formula: the
document every "Export JSON" / "Copy as JSON" button writes, every Import reads,
and every LLM round-trip passes through.

The format is deliberately small, entirely declarative, and has no executable
content. A conforming document is a plain JSON object with no `$ref`, no
expressions, no code — which is what makes it safe to accept from a stranger, a
chat model, or a text field.

**This spec is written to be implementable by someone who has never seen the
Fractbox source.** Where the reference implementation's behaviour is surprising,
this document says so plainly rather than describing the format it would prefer
to have. Every behavioural claim below is asserted by a test in
`core/validate.test.mjs`; none of it is aspirational.

---

## 1. Scope and non-goals

### 1.1 What this is

The **interchange surface**: how a formula is written down for another program,
another person, or another version of Fractbox to read.

### 1.2 What this is not

- **Not the share-link wire format.** A Fractbox share URL (`#c=…`) carries a
  _binary_ payload — a tag-length-value stream tuned for URL length, encoded by
  `app/src/share.ts` and decoded by the same. It is versioned separately, is not
  human-readable, and is explicitly **not** covered here. JSON is the format you
  read, diff, and hand-edit; the wire format is the one you paste into chat.
  They encode overlapping but not identical state — notably the wire format
  carries the colour theme and view state, which §8.2 explains this format does
  not.
- **Not a rendering specification.** Two conforming consumers agree on what a
  document _means_ (which operators, in what order, with what parameters). Pixel
  output additionally depends on the renderer tier, quality settings, and
  hardware.
- **Not a stable ABI for operator opcodes.** Operators are identified in JSON by
  their string `key`, never by their internal numeric id. The numeric ids exist,
  are stable, and are irrelevant here.
- **Not a container for app state.** Preferences, timeline animations, export
  settings and window layout are not part of a formula.

### 1.3 Terminology

**MUST**, **SHOULD**, **MAY** carry their usual force. A **producer** writes
documents; a **consumer** reads them. The reference consumer is
`core/sanitize.js`, hereafter _the importer_.

---

## 2. Document model

A formula document is a JSON **object**. It is one of exactly three _shapes_,
distinguished by which fields are present:

| Shape      | Determined by                                    | §   |
| ---------- | ------------------------------------------------ | --- |
| **flat**   | neither of the below                             | 5   |
| **scene**  | a non-empty `objects` array                      | 6   |
| **hybrid** | an `objects`-less document with a `hybrid` field | 7   |

Shape detection is performed in exactly that order, and the two structural
fields are **mutually exclusive**: a document carrying both a non-empty
`objects` and a `hybrid` is read as a **scene**, and the `hybrid` field is
discarded entirely. Producers MUST NOT emit both.

> An **empty** `objects: []` does not make a scene. Such a document is read as
> flat and the empty array disappears. Producers SHOULD omit the field instead.

### 2.1 Common top-level fields

Every shape shares this core. Only `ops` is required (and §6 relaxes even that
for scenes).

| Field      | Type                | Default      | Meaning                                                              |
| ---------- | ------------------- | ------------ | -------------------------------------------------------------------- |
| `ops`      | array of Op (§3)    | **required** | The formula body, applied in order, once per iteration.              |
| `name`     | string, ≤ 60 chars  | `"Imported"` | Short label. Not an identifier — not unique, not a key.              |
| `note`     | string, ≤ 120 chars | `""`         | One-line description.                                                |
| `iters`    | integer 2 … 64      | `12`         | How many times the op-list is applied.                               |
| `addC`     | boolean             | `false`      | Re-add the starting point each iteration (Mandelbrot-style seeding). |
| `deOption` | integer 0 … 3       | `2`          | Distance-estimator family (§2.2).                                    |
| `camera`   | Camera (§8.1)       | see §8.1     | The view. Not part of the shape.                                     |
| `julia`    | boolean             | absent       | Lock the seed to a fixed point instead of the marched pixel.         |
| `juliaC`   | `[x, y, z]`         | `[0,0,0]`    | That fixed point. Only read when `julia` is true.                    |

`name` and `note` are user-facing text. Control characters (U+0000–U+001F,
U+007F, U+2028, U+2029) are each replaced with a space on import — `name`
reaches a GLSL comment in exported shader source, and a newline there would
escape the comment. Producers SHOULD NOT emit them.

`juliaC` without `julia: true` is silently dropped. Producers MUST emit both or
neither.

### 2.2 `deOption`

| Value | Meaning                                                          |
| ----- | ---------------------------------------------------------------- |
| `0`   | Escape-time log DE — `0.5·ln(r)·r/dr`. For power maps.           |
| `1`   | _Unassigned._ Accepted, but no engine distinguishes it from `2`. |
| `2`   | Analytic IFS DE — `r/\|w\|`. The default; for folds and scales.  |
| `3`   | Numeric finite-difference DE. Required by the operators in §3.4. |

Producers SHOULD NOT emit `1`.

`deOption` states the DE _family the op-list belongs to_, not a free choice: a
mandelbulb-power stack with `deOption: 2` renders wrong, not differently. When
in doubt, copy the value from a document whose op-list has the same shape.

---

## 3. `ops` — the formula body

```json
{ "key": "sphereFold", "values": [0.5, 1], "muted": false }
```

| Field    | Type             | Required | Meaning                                          |
| -------- | ---------------- | -------- | ------------------------------------------------ |
| `key`    | string           | yes      | Operator identifier, verbatim from the registry. |
| `values` | array of numbers | yes\*    | Parameters, **positional** (§3.2).               |
| `muted`  | boolean          | no       | When true, the op is skipped. Omit when false.   |

\* An operator that declares no parameters takes `"values": []`. The field may
be omitted entirely, but a producer SHOULD write it — an explicit empty array
distinguishes "no parameters" from "I forgot".

### 3.1 `key` is the whole contract

An unrecognised `key` is a **hard error**: the document does not load, and it
does not partially load. There is no fallback operator and no way to express an
operator that the consumer does not implement.

This is deliberate. Silently skipping an unknown operator would render a
_different fractal_ under the original's name, which for a format whose entire
payload is "which operations, in what order" is the one failure mode worth
refusing outright.

The registry is `core/operators.js`; it holds **62** operators as of this
revision. Operator keys are **append-only and never renamed**. A document
written today will still name real operators in five years.

### 3.2 `values` is positional

`values[i]` is the operator's _i_-th declared parameter. There are no parameter
names in the document — the registry supplies them, and the order is part of the
operator's stable definition.

- **Too few** values: the missing trailing parameters take their registry
  defaults.
- **Too many**: the extras are dropped.

Neither is an error, and neither is what the producer meant. Both are reported
as warnings (§9). Write exactly as many values as the operator declares.

No operator declares more than **3** parameters.

### 3.3 Parameter ranges are normative

Every registry parameter declares `min`, `max`, `default` and a UI `step`.

> **A conforming producer MUST keep every value within its declared
> `min` … `max`.**

The reference importer **enforces this by clamping**
([#538](https://github.com/fractbox/fractbox/issues/538), landed in
[#542](https://github.com/fractbox/fractbox/pull/542)): a value outside its
declared range is pulled to the nearest bound rather than rejected. So an
out-of-range document still _loads_ — it just does not load _what it said_.
The validator reports every one as a warning naming the clamped result.

Two consequences worth stating plainly:

- **Clamping is silent and lossy.** It is the reason ranges are worth respecting
  even though violating them is not fatal. Producers cannot detect the clamp
  from the load succeeding.
- **Values are clamped, never snapped to `step`.** `step` is a **UI** hint
  (slider granularity) and carries no normative weight — a value riding between
  steps is perfectly valid and is left alone.

A parameter whose registry entry lacks a finite `min`/`max` is not clamped; it
falls back to the finite-number coercion of §4.

Where a shipped preset and its declared range disagree, the resolution is a
judgement call, not a mechanical one. Two presets did (`Surf Coral`'s
`surfFold(5)` against a max of 3, `Cantor Rotations`' `translate(-5.77)` against
a min of -2), and #542 **widened the ranges to match the art** rather than
clamping the art away — Cantor's -5.77 is a value the formula's own algebra
produces. A `presets.test.mjs` gate now keeps presets and ranges in agreement,
and `core/validate.test.mjs` requires the exported catalogue to be entirely
warning-free, so a future disagreement has to be decided rather than absorbed.

### 3.4 Operators restricted to flat formulas

A small set of operators (those whose escape-time map has no analytic
derivative — `wRule: "bulb_numeric"` in the registry) route the whole formula
through a numeric finite-difference DE. Scene objects and hybrid slots evaluate
their own DE bodies, which do not implement that path.

Using one of these operators inside a **scene object** or a **hybrid slot** is a
**hard error**. They are legal only in a flat formula. The registry is
authoritative about which operators these are; a consumer MUST derive the list
from `wRule` rather than hard-coding it.

---

## 4. Value semantics

- All numbers are plain finite JSON numbers. `NaN`, `Infinity` and `1e999` are
  not JSON and MUST NOT be emitted; a value that is not finite is replaced by
  its default on import.
- Non-numeric values in numeric positions are **coerced**, not rejected: the
  importer applies `Number(v)` before testing finiteness, so `"2"` is 2, `true`
  is 1, `null` is 0, and `[]` is 0. Producers MUST NOT rely on this — it is
  documented because it is observable, not because it is supported.
- Angles are degrees unless a parameter's registry name says otherwise.
- Integer-typed fields (`iters`, `deOption`, `objType`, `shapeId`, schedule
  counts) are **rounded and then clamped**; float-typed fields are clamped only.
- Clamping is silent. A document is never rejected for an out-of-bounds scalar,
  only quietly repaired.

---

## 5. Flat formulas

The base shape: one op-list, iterated. Nothing beyond §2 and §3 applies.

At most **64** ops. The list is **truncated first and validated second**, so an
unrecognised operator at position 65 is never seen and never an error — it is
simply gone. Producers MUST NOT exceed the cap.

> **Truncate-then-validate is the rule everywhere**, not a quirk of flat
> formulas: it holds for scene objects (§6), scene object counts, hybrid slot
> counts and each slot's own op-list (§7). The consequence is uniform and worth
> internalising — **content past any cap is discarded without ever being
> looked at**, so a document whose only fault lies beyond a cap loads cleanly.
> A validator or converter that inspects the whole array will disagree with the
> importer about which documents are valid.

---

## 6. Scenes (CSG)

A document with a non-empty `objects` array is a scene: a set of distance-
estimated objects combined by constructive solid geometry.

```json
{
  "ops": [],
  "objects": [ { … }, { … } ]
}
```

- At most **8** objects; extras are dropped (truncate-then-validate, as §5).
- The top-level `ops` is **optional** for a scene, and conventionally empty —
  the geometry lives in the objects. When present it is still validated.
- The **first object's `combine` is ignored**: it is the base that the rest
  combine into.

### 6.1 Object fields

| Field            | Type            | Default        | Meaning                                                  |
| ---------------- | --------------- | -------------- | -------------------------------------------------------- |
| `ops`            | array of Op     | `[]`           | The object's own op chain. ≤ **24**.                     |
| `iters`          | integer 1 … 24  | `8`            | Iterations of that chain.                                |
| `addC`           | boolean         | `false`        | As §2.1, per object.                                     |
| `deOption`       | integer 0 … 3   | `2`            | As §2.2, per object.                                     |
| `transform`      | Transform       | identity       | Placement (§6.3).                                        |
| `combine`        | integer 0 … 3   | `0`            | 0 union · 1 smooth-union · 2 subtract · 3 intersect.     |
| `blendK`         | number 0 … 10   | `0`            | Smooth-union softness. Only meaningful for `combine: 1`. |
| `color`          | `[r,g,b]` 0 … 1 | cycled palette | Per-object albedo, sRGB.                                 |
| `looseDE`        | boolean         | `false`        | Marks the object's DE as a loose bound.                  |
| `name`           | string ≤ 40     | absent         | User label.                                              |
| `muted`          | boolean         | absent         | Skip this object. Omit when false.                       |
| `julia`/`juliaC` | as §2.1         | absent         | Per-object seed lock; op-chain objects only.             |

`combineType` is accepted as a legacy alias for `combine`.

### 6.2 Two object forms

An object is either a **shape-leaf object** or a **legacy primitive object**,
decided by one rule: **`shapeId` present and not null** selects the leaf form.

**Leaf form (preferred).** `shapeId` names a closed-form shape from the leaf
registry (`core/leaves.js`, currently **58** leaves, ids 1 … 58; `0` means "no
leaf — classic IFS distance"). `shapeParams` is a positional array of up to 4
numbers, clamped per the leaf's declared ranges.

Crucially, **the leaf form keeps its op chain**: `ops` runs _and then_ the leaf
distance is evaluated. Mixed op-chain-plus-shape objects are the reason this
form exists.

`iterShape: true` (leaf ids > 0 only) evaluates the leaf _inside_ the iteration
and takes the minimum, instead of once at the end.

Unknown `shapeId`s in 0 … 255 are **preserved, not clamped** — a document
written by a newer build must survive a round-trip through an older one and come
back intact, even though the older build renders it with a fallback shape.

**Legacy form.** No `shapeId`; `objType` selects a built-in primitive:

| `objType` | Shape                                             |
| --------- | ------------------------------------------------- |
| `0`       | IFS — the op chain, no leaf                       |
| `1`…`6`   | box · sphere · torus · cylinder · capsule · plane |

with sizes in `primParam` (and `primParam2` for torus / cylinder / capsule).

> **Trap.** In the legacy form, a non-zero `objType` **discards the object's
> `ops` entirely** — silently, and before they are validated. An object written
> as `{"objType": 2, "ops": [ … ]}` is a plain sphere; the chain is gone, and
> even an unrecognised operator inside it goes unreported. New producers SHOULD
> use the leaf form, which does not have this behaviour.

An object with a `shapeId` also emits `objType`, `primParam` and `primParam2` as
**derived compatibility aliases** on export, so that older consumers see
something reasonable. Producers SHOULD NOT set them by hand alongside
`shapeId`; consumers SHOULD prefer `shapeId`/`shapeParams` when both appear.

### 6.3 Transform

```json
"transform": { "origin": [0, 0, 0], "uscale": 1, "rot": [14, 20, 0] }
```

`origin` clamps to ±10⁴; `uscale` is a positive uniform scale (10⁻⁴ … 10⁴);
`rot` is either 3 numbers (Euler XYZ, **degrees**) or 4 (quaternion). Scaling is
uniform only — there is no non-uniform scale, because a distance estimate does
not survive one.

For backward compatibility the three fields are also read directly off the
object when `transform` is absent. Producers SHOULD write the nested form.

---

## 7. Hybrid formulas

A hybrid interleaves several op-lists ("slots") on a repeating schedule. **Slot
A is always the document's own top-level `ops` / `addC`** — it is not repeated
inside the `hybrid` object. The `hybrid` field holds only the _other_ slots.

There are **two stored shapes**, and which one is correct depends purely on the
slot count. This is the format's one genuine wart; it exists so that every
hybrid document written before three-slot support was added is still
byte-identical today.

### 7.1 Exactly two slots — the legacy shape

```json
"hybrid": {
  "b": { "ops": [ … ], "addC": true },
  "schedule": { "a": 1, "b": 1 }
}
```

`a` and `b` are each 1 … 8, and `a + b` ≤ 12: run slot A `a` times, then slot B
`b` times, repeat. If the sum exceeds 12 the **larger** count is reduced.

### 7.2 Three or more slots — the `slots` shape

```json
"hybrid": {
  "slots": [ { "ops": [ … ], "addC": true }, { "ops": [ … ], "addC": true } ],
  "schedule": { "counts": [2, 1, 3] }
}
```

`slots[]` holds the slots **after A**, so total slots = `slots.length + 1`, and
`counts[0]` is slot **A**'s count — `counts` is one longer than `slots`. Each
count is 1 … 8 and the total period is ≤ 16; over that, the largest counts are
trimmed first.

### 7.3 Choosing between them

A producer MUST use the legacy shape for exactly two slots and the `slots` shape
for three or more. A consumer MUST accept both. Writing `slots` with a single
entry is tolerated — it normalises to the legacy shape on import — but is not
conforming output.

### 7.4 Slot caps

At most **4** slots total (A + 3) — so `slots[]` carries at most **3** entries.
Each slot's own `ops` is capped at **64**, exactly like slot A's.

Both caps truncate-then-validate (§5): `slots[]` is cut to length _before_ any
slot is examined, and each surviving slot's `ops` is cut to 64 before those are
examined. A fourth extra slot, or a 65th op inside a slot, is therefore dropped
unread — including any unrecognised operator in it.

### 7.5 Muting

`hybrid.aMuted: true` skips slot A; `hybrid.b.muted` / `slots[i].muted` skip the
others. All are emit-only-when-true. If every slot is muted the formula renders
nothing.

### 7.6 The shared parameter budget

Every slot's parameters are packed into one uniform array. The sum of all
operator parameters across **all** slots MUST NOT exceed **192**. Exceeding it
is a **hard error** — one of only two caps in this format that refuses rather
than truncates.

> `"hybrid": []` — an array, not an object — is read **as a hybrid** by the
> reference importer, because its shape test is `typeof h === "object"`. The
> practical consequence is that it activates §3.4's flat-only-operator ban. This
> is a quirk of the implementation, is pinned by a regression test so it cannot
> change unnoticed, and MUST NOT be emitted.

---

## 8. Presentation: camera and colour

Neither field describes the _shape_. Both describe how it is shown.

### 8.1 `camera`

```json
"camera": { "yawDeg": 35, "pitchDeg": 22, "dist": 24, "fovDeg": 42 }
```

| Field      | Range       | Default | Notes                           |
| ---------- | ----------- | ------- | ------------------------------- |
| `yawDeg`   | unbounded   | `35`    | Orbit azimuth, degrees.         |
| `pitchDeg` | unbounded   | `22`    | Orbit elevation, degrees.       |
| `dist`     | 10⁻⁶ … 10⁹  | `14`    | Orbit radius.                   |
| `fovDeg`   | 1 … 179     | `42`    | Vertical field of view.         |
| `target`   | `[x, y, z]` | absent  | Orbit centre. Omitted ⇒ origin. |

The camera is a _suggestion about framing_, not part of the shape. A consumer
building a thumbnail grid, or one with its own navigation, MAY ignore it
entirely.

### 8.2 `coloring` — carried, but not specified

`coloring` is a large object (palette stops, shading mode, light rig, atmosphere
macros — see `core/coloring.js`) describing how the surface is painted.

Its treatment in v1 is genuinely uneven, and this spec describes that rather
than tidying it:

- On a **scene**, the importer keeps `coloring`, running it through
  `sanitizeColoring` (added in #542 — it used to be a raw passthrough straight to
  the uniform writers). That pass is **shape-preserving**: it clamps and repairs
  the keys that are present and invents nothing, because absence is meaningful
  downstream (`encodeColoring` keys its defaults off it). It never rejects.
- On a **flat or hybrid** document, `coloring` is **dropped**.
- `stripForExport` — the writer behind every JSON export — **does not emit
  `coloring` at all**, for any shape. So a document produced by Fractbox never
  carries one, even for a scene whose colours were carefully set. Colour travels
  in the share link and in saved library entries, not in exported JSON.

**`coloring` is therefore NOT part of the v1 interchange contract.** It is
documented here so that implementers know why they may encounter the field, why
it may vanish, and why they must not depend on it. Specifying it properly —
and making the export surface symmetric — is deferred to v2; see the "Known
asymmetries" note at the end of §11.

Consumers MUST tolerate its presence and MUST NOT fail on its contents.

---

## 9. Conformance

The format distinguishes two questions that are easy to conflate.

### 9.1 "Will it load?" — errors

A document is **valid** when it contains none of the following. These are
exactly the conditions the reference importer refuses, and there are only six:

| Condition                                                    | Code            |
| ------------------------------------------------------------ | --------------- |
| Not a JSON object                                            | `not-an-object` |
| No `ops` array (non-scene shapes)                            | `missing-ops`   |
| An operator `key` not in the registry                        | `unknown-op`    |
| A flat-only operator (§3.4) in a scene object or hybrid slot | `numeric-de`    |
| A scene object that is not an object                         | `bad-object`    |
| A hybrid whose slots pack more than 192 parameters           | `over-cap`      |

Everything else loads.

### 9.2 "Will it load unchanged?" — warnings

The importer is deliberately lenient: it clamps, pads, truncates and drops
rather than refusing. Each such repair is a place where what the producer wrote
is not what the consumer got — invisible unless something says so.

A **conforming producer emits documents with neither errors nor warnings.** A
conforming consumer accepts anything without errors.

The main lossy behaviours, all reported as warnings:

| Behaviour                                        | Where          |
| ------------------------------------------------ | -------------- |
| Unrecognised fields are dropped                  | anywhere (§10) |
| Out-of-range parameter values are clamped        | §3.3           |
| `values` arity mismatch                          | §3.2           |
| Over-cap ops / objects / slots truncated         | §5, §6, §7     |
| `name` / `note` cut, control characters scrubbed | §2.1           |
| `coloring` dropped on non-scenes                 | §8.2           |
| `hybrid` dropped when `objects` is present       | §2             |
| Legacy primitive object discarding its `ops`     | §6.2           |
| `juliaC` without `julia`                         | §2.1           |
| Non-numeric values coerced                       | §4             |

---

## 10. Versioning

**There is no version field, and adding one would not help.** An unrecognised
top-level key — `formatVersion` included — is dropped on import by every
existing consumer, so a version marker could never be read by the builds that
would most need it. Producers MUST NOT emit one; the format is versioned by this
document, and compatibility is maintained structurally instead:

1. **Additive evolution.** New fields and new operator keys may be added at any
   time. Existing documents stay valid; existing consumers keep working.
2. **Unknown fields are dropped, not preserved.** The importer rebuilds every
   node from scratch rather than copying the input, so a field a consumer does
   not know about does not survive a load-and-save round-trip. **This format is
   not extensible by third parties.** Application-specific data has nowhere to
   live, and putting it in a formula document will silently lose it. A scene's
   `coloring` (§8.2) is the nearest thing to an exception — its own keys survive
   — but it is sanitized key by key, so unrecognised keys inside it are dropped
   like any other, and §8.2 explains why you should not rely on it regardless.
3. **Unknown enum values degrade, unknown operators do not.** A `shapeId` from a
   newer build is preserved and rendered with a fallback (§6.2). An operator
   `key` from a newer build is a hard error (§3.1) — the difference is that a
   wrong shape is recognisably wrong, while a missing operator produces a
   plausible-looking different fractal.
4. **Keys and ids are append-only.** Operator keys and leaf ids are never
   renamed or renumbered.
5. **Deprecation is by aliasing, not removal.** `combineType`, `objType`,
   `primParam`, `halfExtent` and `radius` are all still read. A future revision
   may stop _writing_ a field; it will not stop _reading_ one.

A future v2 may specify `coloring` properly and add an extension namespace so
third-party data has somewhere legitimate to live. Neither breaks a v1 document.

---

## 11. The validator

`core/validate.js` is this specification in executable form. Zero dependencies,
raw ES module, MIT, ships with the engine:

```js
import { validate } from "./core/validate.js";

const report = validate(JSON.parse(text));
// → { ok: boolean, errors: [ { severity, code, message, where? } … ] }
```

- `ok` is true when nothing of severity `"error"` was found.
- `errors` carries **both** severities — errors first, then warnings, each in
  document order. Filter on `severity` to separate §9.1 from §9.2.
- `where` is a JSON path (`$.objects[1].ops[3].values[0]`).
- `code` is a stable machine-readable string; `message` is one human sentence.
- It never throws, whatever it is handed, and it never parses text — the caller
  keeps control of where syntax errors are reported.

**The load-bearing guarantee:**

> `validate(x).ok === true` **if and only if** `sanitizeFormula(x)` does not
> throw.

A validator that disagreed with the importer would be worse than none. That
equivalence is asserted in `core/validate.test.mjs` over the whole 94-preset
catalogue (raw and exported), a hand-written hostile corpus, and a seeded
4 000-document mutation fuzzer; it is additionally soaked over 400 000 generated
documents across two seed batches, ~46 % of which the importer rejects. Every
bug those soaks found is pinned as a named regression test.

The equivalence is a _relationship_, which means importer changes can break it
without touching the validator at all — and did: #542 moved several caps from
"validate then discard" to "discard then validate", flipping the verdict on
documents whose only fault sat past a cap. **Any change to `core/sanitize.js`'s
accept/reject behaviour is a change to this specification** and must be made in
`core/validate.js` and here in the same commit.

`validate` is a **pre-flight, not a gate**. `sanitize` remains the only thing
that decides a formula is safe to load; the validator predicts and explains that
verdict. The LLM paste path (`app/src/llm.ts`) uses it for exactly that — good
error messages first, sanitize's authority second.

**Known asymmetries** (documented, not yet fixed):

- `stripForExport` drops `coloring` even for scenes, where the importer would
  have preserved it (§8.2). Export is therefore lossier than import.

---

## 12. Worked examples

All three are real, complete, unedited output of the reference writer
(`stripForExport`) on a shipped preset.

### 12.1 Flat — "Mandelbox"

The canonical three-operator formula. `addC: true` re-seeds each iteration, so
this is the Mandelbrot-style set of the box/sphere fold map rather than its
attractor. `deOption: 2` because a fold-and-scale stack has an analytic IFS
derivative.

```json
{
  "name": "Mandelbox",
  "note": "the classic box-fold · sphere-fold · scale ×2",
  "addC": true,
  "iters": 12,
  "deOption": 2,
  "ops": [
    { "key": "boxFold", "values": [1] },
    { "key": "sphereFold", "values": [0.5, 1] },
    { "key": "scale", "values": [2] }
  ],
  "camera": { "yawDeg": 35, "pitchDeg": 22, "dist": 24, "fovDeg": 42 }
}
```

`sphereFold`'s two positional values are `MinRadius` then `FixedRadius`. The
document never says so, and does not need to: the registry does. That is the
whole trade — the document stays small and stable, and every consumer reads the
same registry to interpret it.

### 12.2 Scene — "Cube Cluster"

Two objects unioned. The first is a legacy primitive (`objType: 1`, a box) with
an empty op chain — note its `deOption: 0`, which the importer assigns to every
non-IFS primitive. The second is an IFS object (`objType: 0`) whose chain builds
a Sierpinski-like satellite, with `boxBase: true` giving it flat cube faces
instead of round dust.

Both use `combine: 0` (union); the first object's is ignored regardless, being
the base.

```json
{
  "name": "Cube Cluster",
  "note": "central cube ∪ flat-cube fractal satellites (CSG)",
  "addC": false,
  "iters": 8,
  "deOption": 2,
  "ops": [],
  "camera": { "yawDeg": 28, "pitchDeg": 18, "dist": 8, "fovDeg": 42 },
  "objects": [
    {
      "objType": 1,
      "ops": [],
      "iters": 8,
      "addC": false,
      "deOption": 0,
      "transform": { "origin": [0, 0, 0], "uscale": 1, "rot": [14, 20, 0] },
      "combine": 0,
      "blendK": 0,
      "looseDE": false,
      "color": [0.9, 0.52, 0.2],
      "primParam": 0.8
    },
    {
      "objType": 0,
      "ops": [
        { "key": "absFold", "values": [] },
        { "key": "scale", "values": [3] },
        { "key": "translate", "values": [-2, -2, -2] }
      ],
      "iters": 5,
      "addC": false,
      "deOption": 2,
      "transform": { "origin": [0, 0, 0], "uscale": 2, "rot": [0, 0, 0] },
      "combine": 0,
      "blendK": 0,
      "looseDE": false,
      "color": [0.3, 0.55, 0.85],
      "boxBase": true,
      "primParam": 1
    }
  ]
}
```

`absFold` takes no parameters, hence `"values": []`. The top-level `ops` is
empty, as it usually is for a scene — all the geometry is in the objects.

### 12.3 Hybrid — "Menger x Mandelbox"

Two slots, so the **legacy** shape (§7.1). Slot A is the top-level `ops` (a
Menger fold); slot B rides in `hybrid.b` (a Mandelbox fold). `schedule: {a: 1,
b: 1}` alternates them every iteration.

Note that `addC` differs per slot — the document's own `addC: false` is slot
A's, and `hybrid.b.addC: true` is slot B's. That asymmetry is the point of the
formula.

```json
{
  "name": "Menger x Mandelbox",
  "note": "alternates a Menger fold with a Mandelbox fold each iteration (hybrid, IFS×IFS)",
  "addC": false,
  "iters": 12,
  "deOption": 2,
  "ops": [
    { "key": "absFold", "values": [] },
    { "key": "mengerFold", "values": [] },
    { "key": "scale", "values": [3] },
    { "key": "translate", "values": [-2, -2, 0] }
  ],
  "camera": { "yawDeg": 14, "pitchDeg": 14, "dist": 9.2, "fovDeg": 42 },
  "hybrid": {
    "b": {
      "ops": [
        { "key": "boxFold", "values": [1] },
        { "key": "sphereFold", "values": [0.5, 1] },
        { "key": "scale", "values": [2] }
      ],
      "addC": true
    },
    "schedule": { "a": 1, "b": 1 }
  }
}
```

### 12.4 Hybrid, three slots — "Triune Bulb"

The same formula family in the **`slots`** shape (§7.2), for contrast. Three
slots: A is `mandelbulbPower 8` in the top-level `ops`; B and C ride in
`slots[]`. `counts: [2, 1, 3]` is one entry longer than `slots` because
`counts[0]` belongs to slot A — run A twice, B once, C three times, repeat.
Period 6, within the limit of 16.

```json
{
  "name": "Triune Bulb",
  "note": "power-8, power-2 and Y-axis power-5 bulbs on a 2:1:3 schedule (hybrid, three escape-time phases)",
  "addC": true,
  "iters": 10,
  "deOption": 0,
  "ops": [{ "key": "mandelbulbPower", "values": [8] }],
  "camera": { "yawDeg": 35, "pitchDeg": 14, "dist": 5, "fovDeg": 42 },
  "hybrid": {
    "slots": [
      { "ops": [{ "key": "mandelbulbPower", "values": [2] }], "addC": true },
      { "ops": [{ "key": "bulbAxis", "values": [5, 1, 0] }], "addC": true }
    ],
    "schedule": { "counts": [2, 1, 3] }
  }
}
```

`deOption: 0` here, not `2`: every slot is an escape-time power map, so the
whole formula belongs to the log-DE family (§2.2).

---

## 13. Getting the registry

Everything this document defers to the registry for — operator keys, parameter
names, orders, ranges and defaults; leaf ids and their parameters; the caps — is
machine-readable and ships with the engine:

| What                  | Where                                      |
| --------------------- | ------------------------------------------ |
| Operators             | `core/operators.js` → `OPERATORS`          |
| Shape leaves          | `core/leaves.js` → `LEAVES`                |
| Caps and limits       | `core/limits.js`                           |
| Hybrid slot cap       | `core/hybridmodel.js` → `HYBRID_MAX_SLOTS` |
| This spec, executable | `core/validate.js` → `validate()`          |

`app/src/llm.ts`'s `opsReference()` renders the whole registry as plain text
from those same sources — a ready-made reference to hand to a language model,
generated rather than hand-maintained, so it cannot drift.

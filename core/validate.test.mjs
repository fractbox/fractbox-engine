// Formula JSON validator gate — node --test core/validate.test.mjs
//
// The load-bearing test in this file is EQUIVALENCE: for any document at all,
//
//     validate(x).ok === true   ⟺   sanitizeFormula(x) does not throw
//
// core/validate.js only earns its place as "the spec, executable" if it predicts
// the importer's verdict rather than inventing a second, subtly different one.
// A validator that rejects what sanitize accepts turns the paste pre-flight into
// a liar; one that accepts what sanitize rejects makes every published document
// a coin flip. So the property is asserted three ways: over the whole shipped
// preset catalogue, over a hand-written hostile corpus, and over a seeded
// mutation fuzzer that keeps breaking real presets in structural ways.
//
// Everything else here pins the WARNING half — the silent repairs sanitize
// performs — because "it loaded" and "it loaded unchanged" are different
// promises and the spec makes both.

import assert from "node:assert/strict";
import { test } from "node:test";
import { existsSync, readFileSync } from "node:fs";
import { validate, MAX_FINDINGS, SPEC_VERSION } from "./validate.js";
import { sanitizeFormula } from "./sanitize.js";
import { stripForExport } from "./exporter.js";
import { PRESETS } from "./oplist.js";
import { OPERATORS } from "./operators.js";
import { LEAVES, MAX_LEAF_ID } from "./leaves.js";
import { MAX_FLAT_OPS, MAX_OBJECTS, MAX_PARAMS } from "./limits.js";
import { HYBRID_MAX_SLOTS } from "./hybridmodel.js";

// A 2-slot hybrid that JUST exceeds the shared uniform budget — DERIVED, never
// hand-typed. This used to be a literal 40 × translate(3) × 2 slots = 240 vs a
// hand-typed MAX_PARAMS of 192. When the budget became MAX_FLAT_OPS ×
// MAX_OP_PARAMS (OP_PARAM_ENCODING.md §5.1) that fixture silently stopped
// exceeding anything — the over-cap arm would have kept "passing" while testing
// nothing. Note a 3-param op can no longer bust the pool at all now: two slots
// of 64 ops × 3 lands exactly ON the cap, which is precisely the property the
// derivation buys. So the fixture uses the WIDEST registered op.
const WIDEST = OPERATORS.reduce((a, b) =>
  b.params.length > a.params.length ? b : a,
);
const overCap = () => {
  const per = Math.ceil((MAX_PARAMS + 1) / (2 * WIDEST.params.length));
  if (per > MAX_FLAT_OPS)
    throw new Error("no legal 2-slot hybrid can exceed MAX_PARAMS any more");
  const ops = Array.from({ length: per }, () => ({
    key: WIDEST.key,
    values: WIDEST.params.map((p) => p.default),
  }));
  return { ops, total: 2 * per * WIDEST.params.length };
};

// Structured-clone via JSON, but tolerant of the non-JSON values the hostile
// corpus deliberately includes (undefined, NaN, a cyclic object).
const clone = (x) => {
  try {
    const s = JSON.stringify(x);
    return s === undefined ? x : JSON.parse(s);
  } catch {
    return x;
  }
};
const label = (x) => {
  try {
    return JSON.stringify(x)?.slice(0, 90) ?? String(x);
  } catch {
    return "(uncloneable)";
  }
};
const errorsOf = (r) => r.errors.filter((e) => e.severity === "error");
const warningsOf = (r) => r.errors.filter((e) => e.severity === "warning");
const codes = (list) => list.map((e) => e.code);

/** Does the real importer accept this document? */
function sanitizeAccepts(doc) {
  try {
    sanitizeFormula(clone(doc));
    return true;
  } catch {
    return false;
  }
}

/** The one invariant this module exists to hold. */
function assertAgrees(doc, label) {
  const r = validate(clone(doc));
  const accepted = sanitizeAccepts(doc);
  assert.equal(
    r.ok,
    accepted,
    `${label}: validate said ok=${r.ok} but sanitize ${
      accepted ? "ACCEPTED" : "REJECTED"
    } it.\nfindings: ${JSON.stringify(r.errors, null, 1)}`,
  );
  // A rejection must always come with a reason — an empty error list next to
  // ok:false would be unactionable.
  if (!r.ok)
    assert.ok(
      errorsOf(r).length > 0,
      `${label}: ok:false with no error-severity finding`,
    );
  return r;
}

// ── the shipped catalogue ────────────────────────────────────────────────────

test("every shipped preset is a valid formula document", () => {
  assert.ok(PRESETS.length > 0, "preset catalogue is non-empty");
  for (const p of PRESETS) {
    const r = validate(p);
    assert.deepEqual(
      errorsOf(r),
      [],
      `preset "${p.name}" must validate clean, got ${JSON.stringify(errorsOf(r))}`,
    );
  }
});

test("the EXPORTED form of every preset round-trips warning-free", () => {
  // stripForExport ∘ sanitizeFormula is what "Export JSON" / "Copy as JSON"
  // actually write. That surface is the format's shop window: it should be
  // exemplary, not merely loadable — ZERO warnings, no exceptions.
  //
  // It briefly had two: Surf Coral's surfFold(5) and Cantor Rotations'
  // translate(-5.77) sat outside their declared ranges. #542 settled that at the
  // source by WIDENING the registry to match the shipped art (surfFold max 3→5,
  // translate ±2→±6) rather than clamping the art away, so the exception list is
  // empty and stays empty. A stray appearing here means either a preset drifted
  // out of range or a range narrowed underneath one — both worth failing over.
  const strays = new Map();
  for (const p of PRESETS) {
    const doc = clone(stripForExport(sanitizeFormula(clone(p))));
    const r = validate(doc);
    assert.deepEqual(errorsOf(r), [], `exported "${p.name}" must be valid`);
    const w = warningsOf(r);
    if (w.length) strays.set(p.name, codes(w));
  }
  assert.deepEqual(
    Object.fromEntries(strays),
    {},
    "the exported form of the catalogue must be warning-free",
  );
});

test("validate agrees with sanitize across the whole catalogue, raw and exported", () => {
  for (const p of PRESETS) {
    assertAgrees(p, `preset "${p.name}"`);
    assertAgrees(
      stripForExport(sanitizeFormula(clone(p))),
      `exported "${p.name}"`,
    );
  }
});

// ── the spec document itself ─────────────────────────────────────────────────

test("the spec's cited registry sizes match the registries", () => {
  // The spec quotes concrete counts because "consult the registry" is useless to
  // someone reading it cold. Concrete numbers rot: the operator count was
  // already stale by two after two parity waves landed between writing the spec
  // and this commit. Pin them so the doc fails loudly instead of lying quietly.
  const specUrl = new URL("../docs/spec/FORMULA_JSON.md", import.meta.url);
  if (!existsSync(specUrl)) return;
  const md = readFileSync(specUrl, "utf8");

  const cited = (re, what) => {
    const m = re.exec(md);
    assert.ok(m, `the spec no longer states the ${what} in the expected form`);
    return Number(m[1]);
  };
  assert.equal(
    cited(/it holds \*\*(\d+)\*\* operators/, "operator count"),
    OPERATORS.length,
    "spec operator count is stale",
  );
  assert.equal(
    cited(/currently \*\*(\d+)\*\* leaves/, "leaf count"),
    LEAVES.length,
    "spec leaf count is stale",
  );
  assert.equal(
    cited(/ids 1 … (\d+)/, "highest leaf id"),
    MAX_LEAF_ID,
    "spec leaf-id range is stale",
  );
  assert.equal(
    cited(/whole (\d+)-preset/, "preset count"),
    PRESETS.length,
    "spec preset count is stale",
  );
  // The caps the spec states in prose, against limits.js.
  assert.equal(
    cited(/At most \*\*(\d+)\*\* ops\./, "flat op cap"),
    MAX_FLAT_OPS,
  );
  assert.equal(
    cited(/At most \*\*(\d+)\*\* objects/, "object cap"),
    MAX_OBJECTS,
  );
  assert.equal(
    cited(/At most \*\*(\d+)\*\* slots total/, "slot cap"),
    HYBRID_MAX_SLOTS,
  );
  assert.equal(
    cited(/MUST NOT exceed \*\*(\d+)\*\*/, "param budget"),
    MAX_PARAMS,
  );
});

test("the spec's worked examples are real, valid, and current", () => {
  // A specification whose examples have drifted from the implementation is
  // actively harmful — it is the part people copy. Every complete document in
  // docs/spec/FORMULA_JSON.md must (a) validate clean and (b) be byte-identical
  // to what the reference writer emits for the preset it claims to show.
  //
  // Skipped rather than failed when the doc is absent: the OSS engine mirror
  // ships core/ on its own, and these tests travel with it.
  const specUrl = new URL("../docs/spec/FORMULA_JSON.md", import.meta.url);
  if (!existsSync(specUrl)) return;
  const md = readFileSync(specUrl, "utf8");

  const blocks = [...md.matchAll(/```json\n([\s\S]*?)```/g)].map((m) => m[1]);
  assert.ok(
    blocks.length >= 8,
    `expected the spec to carry examples, found ${blocks.length}`,
  );

  const complete = [];
  for (const b of blocks) {
    let v;
    try {
      v = JSON.parse(b);
    } catch {
      continue; // an illustrative fragment ("…"), not a document
    }
    if (
      v &&
      typeof v === "object" &&
      (Array.isArray(v.ops) || Array.isArray(v.objects))
    )
      complete.push(v);
  }
  assert.ok(
    complete.length >= 4,
    `expected 4+ complete examples, found ${complete.length}`,
  );

  for (const doc of complete) {
    const r = validate(doc);
    assert.deepEqual(
      errorsOf(r),
      [],
      `spec example "${doc.name}" must be valid`,
    );
    assert.deepEqual(
      warningsOf(r),
      [],
      `spec example "${doc.name}" must be exemplary, not merely valid: ${JSON.stringify(codes(warningsOf(r)))}`,
    );
    const preset = PRESETS.find((p) => p.name === doc.name);
    assert.ok(preset, `spec example "${doc.name}" must name a real preset`);
    assert.deepEqual(
      doc,
      clone(stripForExport(sanitizeFormula(clone(preset)))),
      `spec example "${doc.name}" has drifted from what stripForExport actually writes`,
    );
  }
});

// ── the hostile corpus ───────────────────────────────────────────────────────
//
// Each entry: a document that must FAIL, the code it must fail with, and a
// fragment its message must contain — because "invalid" alone does not help
// anybody fix anything, and the paste repair loop hands these straight to a
// model.

const HOSTILE = [
  [null, "not-an-object", "must be a JSON object"],
  [undefined, "not-an-object", "must be a JSON object"],
  [42, "not-an-object", "must be a JSON object"],
  ["{}", "not-an-object", "must be a JSON object"],
  [true, "not-an-object", "must be a JSON object"],
  [[1, 2, 3], "missing-ops", "not an array"],
  [{}, "missing-ops", '"ops"'],
  [{ name: "no ops here" }, "missing-ops", '"ops"'],
  [{ ops: "not an array" }, "missing-ops", '"ops"'],
  [{ ops: {} }, "missing-ops", '"ops"'],
  [{ ops: null }, "missing-ops", '"ops"'],
  [
    { ops: [{ key: "notAnOperator", values: [] }] },
    "unknown-op",
    "notAnOperator",
  ],
  [
    { ops: [{ key: "boxFold", values: [1] }, { key: "nope" }] },
    "unknown-op",
    "nope",
  ],
  [{ ops: [{ values: [1] }] }, "unknown-op", "Unknown operator"],
  [{ ops: [null] }, "unknown-op", "not an operator object"],
  [{ ops: [42] }, "unknown-op", "not an operator object"],
  [{ ops: [["boxFold"]] }, "unknown-op", "an array"],
  [{ ops: [{ key: 7, values: [] }] }, "unknown-op", "Unknown operator"],
  // Numeric-DE operators are flat-only — rejected in a hybrid slot…
  [
    {
      ops: [{ key: "bristorBrot", values: [] }],
      hybrid: { b: { ops: [] }, schedule: { a: 1, b: 1 } },
    },
    "numeric-de",
    "numeric distance estimator",
  ],
  [
    {
      ops: [],
      hybrid: {
        b: { ops: [{ key: "bristorBrot", values: [] }] },
        schedule: { a: 1, b: 1 },
      },
    },
    "numeric-de",
    "numeric distance estimator",
  ],
  // …and in a scene object (the shapeId form keeps its op chain, so it is seen).
  [
    {
      ops: [],
      objects: [{ shapeId: 0, ops: [{ key: "bristorBrot", values: [] }] }],
    },
    "numeric-de",
    "numeric distance estimator",
  ],
  [
    {
      ops: [],
      objects: [{ objType: 0, ops: [{ key: "bristorBrot", values: [] }] }],
    },
    "numeric-de",
    "numeric distance estimator",
  ],
  [{ ops: [], objects: [null] }, "bad-object", "not an object"],
  [{ ops: [], objects: ["a shape"] }, "bad-object", "not an object"],
  [
    { ops: [], objects: [{ ops: [{ key: "bogusOp" }] }] },
    "unknown-op",
    "bogusOp",
  ],
];

test("the hostile corpus is rejected, each with a usable reason", () => {
  for (const [doc, code, fragment] of HOSTILE) {
    const lbl = `hostile ${label(doc)}`;
    const r = validate(doc);
    assert.equal(r.ok, false, `${lbl} must be rejected`);
    const errs = errorsOf(r);
    assert.ok(
      codes(errs).includes(code),
      `${lbl} expected code "${code}", got ${JSON.stringify(codes(errs))}`,
    );
    const hit = errs.find((e) => e.code === code);
    assert.ok(
      hit.message.includes(fragment),
      `${lbl} message ${JSON.stringify(hit.message)} must mention ${JSON.stringify(fragment)}`,
    );
    assert.ok(
      hit.message.length > 15,
      `${lbl} message must be a sentence, not a token`,
    );
  }
});

test("every hostile document gets the same verdict from sanitize", () => {
  for (const [doc] of HOSTILE) assertAgrees(doc, `hostile ${label(doc)}`);
});

test("a rejection points at WHERE the problem is", () => {
  const r = validate({
    ops: [],
    objects: [
      { objType: 2 },
      { shapeId: 0, ops: [{ key: "boxFold", values: [1] }, { key: "ghost" }] },
    ],
  });
  const e = errorsOf(r).find((x) => x.code === "unknown-op");
  assert.ok(e, "expected the unknown operator to be reported");
  assert.equal(
    e.where,
    "$.objects[1].ops[1]",
    "the path must locate the exact op",
  );
});

test("validate never throws, whatever it is handed", () => {
  const nasty = [
    null,
    undefined,
    NaN,
    [],
    {},
    { ops: [] },
    { ops: [], objects: [{ transform: [] }] },
    { ops: [], hybrid: [] },
    { ops: [], hybrid: { slots: "no" } },
    { ops: [], hybrid: { slots: [null, 3, "x"] } },
    { ops: [], camera: [] },
    { ops: [{ key: "boxFold", values: {} }] },
    { ops: [{ key: "boxFold", values: [Infinity, -Infinity] }] },
    Object.create(null),
  ];
  for (const n of nasty)
    assert.doesNotThrow(() => validate(n), `threw on ${label(n)}`);
  // A self-referencing document must not spin forever either.
  const cyc = { ops: [] };
  cyc.self = cyc;
  assert.doesNotThrow(() => validate(cyc), "threw on a cyclic document");
});

// ── the mutation fuzzer ──────────────────────────────────────────────────────

/** Deterministic PRNG (mulberry32) — a failure here must be reproducible. */
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Structural break-ins, aimed at exactly the routing seams sanitize has:
 *  shape detection, the op-cap slice, the slots[] map-before-slice, and the
 *  legacy-vs-D0 object branch. */
const MUTATIONS = [
  (d) => delete d.ops,
  (d) => (d.ops = "gone"),
  (d) => (d.ops = null),
  (d) =>
    Array.isArray(d.ops) &&
    d.ops.push({ key: "definitelyNotAnOp", values: [] }),
  (d) =>
    Array.isArray(d.ops) && d.ops.unshift({ key: "bristorBrot", values: [] }),
  (d) => Array.isArray(d.ops) && d.ops.push(null),
  (d) => Array.isArray(d.ops) && d.ops.push({ key: "boxFold", values: "nope" }),
  // Past the flat cap: sanitize slices first, so a bad op here is invisible.
  (d) => {
    d.ops = [
      ...(Array.isArray(d.ops) ? d.ops : []),
      ...Array.from({ length: MAX_FLAT_OPS }, () => ({
        key: "boxFold",
        values: [1],
      })),
      { key: "invisibleGarbage", values: [] },
    ];
  },
  (d) => (d.iters = 9999),
  (d) => (d.iters = "many"),
  (d) => (d.deOption = -4),
  (d) => (d.camera = null),
  (d) => (d.camera = { dist: "far", fovDeg: 900 }),
  (d) => (d.name = 123),
  (d) => (d.note = "x".repeat(400)),
  (d) => (d.julia = true),
  (d) => (d.juliaC = [1]),
  (d) => (d.mysteryField = { deeply: { nested: [1, 2, 3] } }),
  (d) => (d.coloring = { mode: 3 }),
  // Turn it into a hybrid.
  (d) =>
    (d.hybrid = {
      b: { ops: [{ key: "scale", values: [2] }] },
      schedule: { a: 2, b: 1 },
    }),
  (d) =>
    (d.hybrid = {
      b: { ops: [{ key: "notReal", values: [] }] },
      schedule: { a: 1, b: 1 },
    }),
  (d) => (d.hybrid = {}),
  (d) => (d.hybrid = { slots: [] }),
  (d) =>
    (d.hybrid = {
      slots: [{ ops: [] }, { ops: [] }, { ops: [] }],
      schedule: { counts: [9, 9, 9, 9] },
    }),
  // slots[] is SLICED before it is mapped (#542) — garbage past the cap is
  // dropped unvalidated and must NOT be an error. This mutation asserted the
  // opposite before #542; it is kept, inverted, as the fuzzer's probe of that seam.
  (d) => {
    d.hybrid = {
      slots: [
        ...Array.from({ length: HYBRID_MAX_SLOTS }, () => ({ ops: [] })),
        { ops: [{ key: "pastTheCapGarbage", values: [] }] },
      ],
      schedule: { counts: [1, 1, 1, 1, 1] },
    };
  },
  // Same seam one level down: a slot's OWN op-list is now capped at
  // MAX_FLAT_OPS before sanitizing, so an unknown op past that is invisible too.
  (d) => {
    d.hybrid = {
      slots: [
        {
          ops: [
            ...Array.from({ length: MAX_FLAT_OPS }, () => ({
              key: "boxFold",
              values: [1],
            })),
            { key: "pastTheSlotOpCap", values: [] },
          ],
        },
        { ops: [] },
      ],
      schedule: { counts: [1, 1, 1] },
    };
  },
  // Blow the shared uniform budget.
  (d) => {
    const { ops: wide } = overCap();
    d.ops = wide;
    d.hybrid = { b: { ops: wide }, schedule: { a: 1, b: 1 } };
  },
  // Turn it into a scene.
  (d) => (d.objects = [{ objType: 2 }, { shapeId: 1 }]),
  (d) => (d.objects = []),
  (d) => (d.objects = [null]),
  (d) => (d.objects = [{ objType: 0, ops: [{ key: "nopeNope", values: [] }] }]),
  // objType > 0 discards the op chain, so the bad op is never validated.
  (d) => (d.objects = [{ objType: 3, ops: [{ key: "nopeNope", values: [] }] }]),
  (d) => (d.objects = [{ shapeId: 1, ops: [{ key: "nopeNope", values: [] }] }]),
  (d) => (d.objects = [{ shapeId: 999, shapeParams: [1, 2, 3, 4] }]),
  (d) =>
    (d.objects = Array.from({ length: MAX_OBJECTS + 4 }, () => ({
      objType: 1,
    }))),
  // Past the object cap — sliced before the map, so never seen.
  (d) => {
    d.objects = [
      ...Array.from({ length: MAX_OBJECTS }, () => ({ objType: 1 })),
      { ops: [{ key: "invisibleGarbage", values: [] }] },
    ];
  },
  (d) => (
    (d.objects = [{ objType: 1 }]),
    (d.hybrid = { b: { ops: [] }, schedule: { a: 1, b: 1 } })
  ),
  (d) =>
    Array.isArray(d.objects) &&
    d.objects[0] &&
    (d.objects[0].transform = "nope"),
  (d) =>
    Array.isArray(d.objects) && d.objects[0] && (d.objects[0].color = [5, -5]),
  (d) => d.hybrid?.b && (d.hybrid.b.ops = [{ key: "alsoNotReal", values: [] }]),
  (d) => d.hybrid && (d.hybrid.schedule = "nope"),
  (d) => d.hybrid && (d.hybrid.slots = [{ ops: [{ key: "stillNotReal" }] }]),
];

test("validate agrees with sanitize across 4000 mutated documents", () => {
  const rand = rng(0x5eed);
  let mutated = 0;
  let rejected = 0;
  for (let i = 0; i < 4000; i++) {
    const base = PRESETS[Math.floor(rand() * PRESETS.length)];
    const doc = clone(base);
    // One to three stacked mutations — single edits rarely reach the seams
    // where two shapes interact (a scene that is also a hybrid, say).
    const n = 1 + Math.floor(rand() * 3);
    const applied = [];
    for (let k = 0; k < n; k++) {
      const mi = Math.floor(rand() * MUTATIONS.length);
      applied.push(mi);
      try {
        MUTATIONS[mi](doc);
      } catch {
        // A mutation that cannot apply to this shape is fine; skip it.
      }
    }
    mutated++;
    const r = assertAgrees(
      doc,
      `seed 0x5eed #${i} (base "${base.name}", mutations ${applied})`,
    );
    if (!r.ok) rejected++;
  }
  // Guard the fuzzer itself: a corpus that never rejects anything would pass
  // the equivalence assertion while testing nothing.
  assert.equal(mutated, 4000);
  assert.ok(
    rejected > 400,
    `the corpus must exercise the reject path hard; only ${rejected}/4000 were rejected`,
  );
  assert.ok(
    rejected < 3600,
    `the corpus must also exercise the accept path; ${rejected}/4000 were rejected`,
  );
});

// ── regressions the 200k-document soak found ─────────────────────────────────
//
// Both of these were real validator bugs, caught only by a fuzzer wide enough to
// write arbitrary junk at arbitrary paths. They are pinned here because the soak
// itself does not run in CI — these two cases are its residue.

test("regression: a coercible non-number decides routing the way sanitize does", () => {
  // clampInt runs Number() FIRST, so objType:true is objType 1 — a primitive,
  // whose op chain is discarded. The garbage operator inside it is therefore
  // never looked at, and the document LOADS. Reading `true` as "not a number,
  // use the default 0" instead would have made it an IFS object, validated the
  // chain, and rejected a document the importer happily accepts.
  const doc = {
    ops: [],
    objects: [{ objType: true, ops: [{ key: "bristorBrot", values: [1] }] }],
  };
  assert.equal(
    sanitizeAccepts(doc),
    true,
    "sanitize accepts this (precondition)",
  );
  assertAgrees(doc, "objType:true with a flat-only op in its discarded chain");
  assert.equal(validate(doc).ok, true);
  // Same coercion, same conclusion, for the other integer fields.
  for (const v of [true, "3", null, [], [2]])
    assertAgrees(
      { ops: [], objects: [{ objType: v, ops: [{ key: "ghostOp" }] }] },
      `objType: ${JSON.stringify(v) ?? String(v)}`,
    );
});

test("regression: an ARRAY hybrid is still a hybrid", () => {
  // sanitizeFormula routes on `typeof obj.hybrid === "object"`, which an array
  // satisfies. So `"hybrid": []` turns a flat formula into a hybrid one — and a
  // numeric-DE operator that was perfectly legal a moment ago is now rejected.
  const doc = { ops: [{ key: "makinTri", values: [0] }], hybrid: [] };
  assert.equal(
    sanitizeAccepts(doc),
    false,
    "sanitize rejects this (precondition)",
  );
  const r = assertAgrees(doc, "array hybrid");
  assert.ok(codes(errorsOf(r)).includes("numeric-de"));
  // The same op is fine with no hybrid key at all.
  assert.equal(validate({ ops: [{ key: "makinTri", values: [0] }] }).ok, true);
  for (const h of [[], [[]], [{ ops: [] }]])
    assertAgrees(
      { ops: [{ key: "makinTri", values: [0] }], hybrid: h },
      `hybrid: ${JSON.stringify(h)}`,
    );
});

test("regression: content past a cap is dropped UNVALIDATED, never rejected", () => {
  // The #542/#544 merge collision, pinned. #542 moved every remaining
  // "validate then throw away" to "slice then validate", which flips the
  // verdict on any document whose only problem sits past a cap: sanitize never
  // looks at it, so the document LOADS. Three caps, one rule.
  //
  // 1. slots[] past the product cap (this is the case the 4000-doc fuzzer
  //    caught on dev: seed 0x5eed #77).
  const pastSlotCap = {
    ops: [],
    hybrid: {
      slots: [
        ...Array.from({ length: HYBRID_MAX_SLOTS }, () => ({ ops: [] })),
        { ops: [{ key: "definitelyNotAnOperator", values: [] }] },
      ],
      schedule: { counts: [1, 1, 1, 1, 1] },
    },
  };
  assert.equal(
    sanitizeAccepts(pastSlotCap),
    true,
    "sanitize accepts (precondition)",
  );
  const r1 = assertAgrees(
    pastSlotCap,
    "garbage in a slot past the product cap",
  );
  assert.equal(r1.ok, true);
  assert.ok(
    codes(warningsOf(r1)).includes("truncated"),
    "dropping slots silently would be worse than saying so",
  );

  // 2. A slot's own ops past MAX_FLAT_OPS — capped like slot A since #542.
  const pastSlotOps = {
    ops: [],
    hybrid: {
      b: {
        ops: [
          ...Array.from({ length: MAX_FLAT_OPS }, () => ({
            key: "boxFold",
            values: [1],
          })),
          { key: "definitelyNotAnOperator", values: [] },
        ],
      },
      schedule: { a: 1, b: 1 },
    },
  };
  assert.equal(
    sanitizeAccepts(pastSlotOps),
    true,
    "sanitize accepts (precondition)",
  );
  assert.equal(validate(pastSlotOps).ok, true);
  assertAgrees(pastSlotOps, "garbage past a slot's own op cap");

  // 3. And the two that were always this way — flat ops, and scene objects.
  assertAgrees(
    {
      ops: [
        ...Array.from({ length: MAX_FLAT_OPS }, () => ({
          key: "boxFold",
          values: [1],
        })),
        { key: "definitelyNotAnOperator", values: [] },
      ],
    },
    "garbage past the flat op cap",
  );
  assertAgrees(
    {
      ops: [],
      objects: [
        ...Array.from({ length: MAX_OBJECTS }, () => ({ objType: 1 })),
        { ops: [{ key: "definitelyNotAnOperator", values: [] }] },
      ],
    },
    "garbage in an object past the object cap",
  );
});

// ── the warning half: what loads, but not unchanged ──────────────────────────

const warnCase = (doc, code, where) => {
  const r = validate(doc);
  assert.equal(r.ok, true, `must still LOAD: ${JSON.stringify(r.errors)}`);
  const w = warningsOf(r);
  assert.ok(
    codes(w).includes(code),
    `expected warning "${code}", got ${JSON.stringify(w.map((x) => `${x.code}@${x.where}`))}`,
  );
  if (where) {
    const hit = w.find((x) => x.code === code);
    assert.equal(hit.where, where);
  }
  return w;
};

test("out-of-range op values warn, load, and are CLAMPED (post-#538)", () => {
  // Registry ranges became enforcement rather than advice in #542: the importer
  // now clamps to [min,max] instead of taking the value as written. So this
  // stays a WARNING — the document still loads, so ok must remain true — but it
  // is a lossy one, and the message has to say what the value actually becomes.
  const box = OPERATORS.find((o) => o.key === "boxFold").params[0];
  const over = box.max + 100;
  const w = warnCase(
    { ops: [{ key: "boxFold", values: [over] }] },
    "out-of-range",
    "$.ops[0].values[0]",
  );
  const hit = w.find((x) => x.code === "out-of-range");
  assert.match(
    hit.message,
    /clamped to 3/,
    "the message must state the value actually imported",
  );
  assert.match(
    hit.message,
    /0\.1\.\.3/,
    "the message must state the declared range",
  );

  // The clamp is real, and the warning is telling the truth about it.
  const landed = sanitizeFormula({ ops: [{ key: "boxFold", values: [over] }] })
    .ops[0].values[0];
  assert.equal(landed, box.max, "sanitize must clamp to the registry max");
  assert.equal(landed, 3);
  assert.equal(
    sanitizeFormula({ ops: [{ key: "boxFold", values: [-999] }] }).ops[0]
      .values[0],
    box.min,
  );
  // Clamping must never turn an out-of-range value into a REJECTION.
  assertAgrees(
    { ops: [{ key: "boxFold", values: [over] }] },
    "over-range boxFold",
  );
});

test("positional arity mismatches warn in both directions", () => {
  warnCase(
    { ops: [{ key: "sphereFold", values: [0.5] }] },
    "values-arity",
    "$.ops[0].values",
  );
  warnCase(
    { ops: [{ key: "boxFold", values: [1, 2, 3] }] },
    "values-arity",
    "$.ops[0].values",
  );
  warnCase({ ops: [{ key: "boxFold" }] }, "values-arity", "$.ops[0].values");
  // An operator that declares no params is clean with an empty array.
  const r = validate({ ops: [{ key: "mengerFold", values: [] }] });
  assert.deepEqual(warningsOf(r), []);
});

test("unrecognised fields warn — a change expressed through one never happens", () => {
  warnCase({ ops: [], iterations: 20 }, "unknown-field", "$");
  warnCase(
    { ops: [{ key: "boxFold", values: [1], colour: "red" }] },
    "unknown-field",
    "$.ops[0]",
  );
  warnCase(
    { ops: [], camera: { yawDeg: 0, zoom: 3 } },
    "unknown-field",
    "$.camera",
  );
  const hit = warningsOf(validate({ ops: [], iterations: 20 })).find(
    (w) => w.code === "unknown-field",
  );
  assert.match(
    hit.message,
    /"iterations"/,
    "the message must name the offending field",
  );
  assert.match(
    hit.message,
    /dropped/,
    "the message must say what happens to it",
  );
});

test("a document with no version field is not an error — the format has none", () => {
  // Stated as a test because it is a deliberate design choice, not an omission:
  // versioning is by additive evolution, so a `formatVersion` key is simply an
  // unknown field that gets dropped. Producers must not depend on one.
  const r = validate({ ops: [], formatVersion: 1 });
  assert.equal(r.ok, true);
  assert.deepEqual(codes(warningsOf(r)), ["unknown-field"]);
});

test("the silent drops each get a voice", () => {
  // coloring rides scenes only.
  warnCase({ ops: [], coloring: { mode: 2 } }, "dropped", "$.coloring");
  // objects wins over hybrid, and the hybrid vanishes entirely.
  warnCase(
    {
      ops: [],
      objects: [{ objType: 1 }],
      hybrid: { b: { ops: [] }, schedule: { a: 1, b: 1 } },
    },
    "dropped",
    "$.hybrid",
  );
  // An empty objects[] is not a scene.
  warnCase({ ops: [], objects: [] }, "dropped", "$.objects");
  // A legacy primitive object silently discards its op chain.
  warnCase(
    {
      ops: [],
      objects: [{ objType: 3, ops: [{ key: "boxFold", values: [1] }] }],
    },
    "dropped",
    "$.objects[0].ops",
  );
  // juliaC without julia never reaches the renderer.
  warnCase({ ops: [], juliaC: [1, 2, 3] }, "orphan-field", "$.juliaC");
});

test("over-cap truncation warns rather than failing", () => {
  warnCase(
    {
      ops: Array.from({ length: MAX_FLAT_OPS + 5 }, () => ({
        key: "boxFold",
        values: [1],
      })),
    },
    "truncated",
    "$.ops",
  );
  warnCase(
    {
      ops: [],
      objects: Array.from({ length: MAX_OBJECTS + 2 }, () => ({ objType: 1 })),
    },
    "truncated",
    "$.objects",
  );
  warnCase(
    {
      ops: [],
      hybrid: {
        slots: Array.from({ length: HYBRID_MAX_SLOTS + 2 }, () => ({
          ops: [],
        })),
        schedule: { counts: [1, 1, 1, 1, 1, 1, 1] },
      },
    },
    "truncated",
    "$.hybrid.slots",
  );
});

test("text that will be cut or scrubbed warns", () => {
  warnCase({ ops: [], name: "x".repeat(61) }, "too-long", "$.name");
  warnCase({ ops: [], note: "x".repeat(121) }, "too-long", "$.note");
  const withNewline = {
    ops: [],
    name: ["a", "b"].join(String.fromCharCode(10)),
  };
  warnCase(withNewline, "control-chars", "$.name");
});

test("schedule bounds warn on both hybrid shapes", () => {
  warnCase(
    { ops: [], hybrid: { b: { ops: [] }, schedule: { a: 9, b: 9 } } },
    "out-of-range",
    "$.hybrid.schedule.a",
  );
  warnCase(
    {
      ops: [],
      hybrid: {
        slots: [{ ops: [] }, { ops: [] }],
        schedule: { counts: [8, 8, 8] },
      },
    },
    "out-of-range",
    "$.hybrid.schedule.counts",
  );
  // counts on a two-slot hybrid is the wrong shape and is ignored.
  warnCase(
    { ops: [], hybrid: { b: { ops: [] }, schedule: { counts: [1, 2] } } },
    "wrong-shape",
    "$.hybrid.schedule.counts",
  );
});

test("the param budget is a hard error, with the number in the message", () => {
  const { ops: wide, total } = overCap();
  const r = validate({
    ops: wide,
    hybrid: { b: { ops: wide }, schedule: { a: 1, b: 1 } },
  });
  assert.equal(r.ok, false);
  const e = errorsOf(r).find((x) => x.code === "over-cap");
  assert.ok(e, `expected over-cap, got ${JSON.stringify(codes(errorsOf(r)))}`);
  assert.match(
    e.message,
    new RegExp(String(MAX_PARAMS)),
    "must state the budget",
  );
  assert.match(
    e.message,
    new RegExp(String(total)),
    "must state what the document actually packs",
  );
  assert.ok(total > MAX_PARAMS, "the fixture must actually exceed the budget");
});

// ── report shape ─────────────────────────────────────────────────────────────

test("the result shape is stable and self-describing", () => {
  const r = validate({ ops: [{ key: "boxFold", values: [99] }], mystery: 1 });
  assert.deepEqual(Object.keys(r).sort(), ["errors", "ok"]);
  assert.equal(typeof r.ok, "boolean");
  assert.ok(Array.isArray(r.errors));
  for (const e of r.errors) {
    assert.ok(
      ["error", "warning"].includes(e.severity),
      `bad severity ${e.severity}`,
    );
    assert.equal(typeof e.code, "string");
    assert.ok(e.code.length > 0);
    assert.equal(typeof e.message, "string");
    assert.ok(e.message.length > 0);
    if (e.where !== undefined)
      assert.match(e.where, /^\$/, "paths are rooted at $");
  }
  assert.equal(typeof SPEC_VERSION, "string");
});

test("a flood of findings truncates the report but never the verdict", () => {
  // Many warnings, one error hiding behind them: the error must still win.
  // translate declares three params: 3 out-of-range warnings + 1 unknown-field
  // per op, so 64 ops overshoot MAX_FINDINGS several times over.
  const ops = Array.from({ length: MAX_FLAT_OPS }, () => ({
    key: "translate",
    values: [999, 999, 999],
    bogusField: 1,
  }));
  ops[MAX_FLAT_OPS - 1] = { key: "thisOperatorDoesNotExist", values: [] };
  const r = validate({ ops });
  assert.equal(r.ok, false, "the error must survive report truncation");
  assert.ok(
    errorsOf(r).some((e) => e.code === "unknown-op"),
    "the blocking error must still be listed, not crowded out by warnings",
  );
  // The cap is per severity, plus the one note that says the report was cut.
  assert.ok(
    errorsOf(r).length <= MAX_FINDINGS,
    "the error list must stay bounded",
  );
  assert.ok(
    warningsOf(r).length <= MAX_FINDINGS + 1,
    "the warning list must stay bounded",
  );
  assert.ok(
    codes(r.errors).includes("report-truncated"),
    "a truncated report must say so",
  );
  assert.equal(
    r.errors[0].severity,
    "error",
    "errors are reported before warnings",
  );
  assertAgrees({ ops }, "flooded document");
});

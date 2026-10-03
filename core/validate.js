// Formula JSON validator — the executable half of docs/spec/FORMULA_JSON.md.
//
// `validate(json)` answers two DIFFERENT questions about one document, and
// keeping them apart is the whole point of this module:
//
//   1. "Will this load?"          → severity "error".   These mirror, exactly,
//      the conditions core/sanitize.js THROWS on. The invariant is pinned by
//      validate.test.mjs over the whole preset catalogue plus a generated
//      corpus:  validate(x).ok === true  ⟺  sanitizeFormula(x) does not throw.
//      sanitize remains THE gatekeeper — this module never decides that a
//      formula is safe, it only predicts (and explains) sanitize's verdict.
//
//   2. "Will it load UNCHANGED?"  → severity "warning". sanitize is deliberately
//      lenient: it clamps, pads, truncates and drops rather than refusing. Every
//      one of those silent repairs is a place where what a producer wrote is NOT
//      what a consumer gets, which for an interchange format is exactly the
//      thing worth reporting. A warning never makes `ok` false, because the
//      document still loads — it just does not survive intact.
//
// `ok` is computed from the errors alone, so a document that only warns is a
// valid-but-lossy document. A conforming PRODUCER should emit neither.
//
// Zero dependencies beyond the core registries, no DOM, no throw: hand it
// anything at all (null, a string, a 10 MB nest of arrays) and it returns a
// verdict. That is what makes it usable as a pre-flight in front of sanitize
// and as a lint pass in someone else's toolchain.

import { byKey, W_BULB_NUMERIC } from "./operators.js";
import { leafById, MAX_LEAF_ID } from "./leaves.js";
import {
  MAX_FLAT_OPS,
  MAX_OBJECTS,
  MAX_OPS_PER_OBJECT,
  MAX_ITERS,
  MAX_PARAMS,
} from "./limits.js";
import { HYBRID_MAX_SLOTS } from "./hybridmodel.js";

/** The spec revision this validator implements — docs/spec/FORMULA_JSON.md. */
export const SPEC_VERSION = "1";

/** Findings are capped, PER SEVERITY, so a hostile document cannot turn a
 *  validation pass into an allocation attack — and so a flood of warnings can
 *  never crowd out the error that made the document invalid. `ok` is tracked
 *  separately and stays exact even after the cap is hit: truncating the REPORT
 *  must never change the VERDICT. */
export const MAX_FINDINGS = 200;

// Text limits sanitize slices to (cleanText) — over these, characters are lost.
const MAX_NAME = 60;
const MAX_NOTE = 120;
const MAX_OBJ_NAME = 40;
// The control characters cleanText replaces with a space. Tested by code point
// rather than a regex literal so no escape sequence can rot in transit.
const hasControl = (s) => {
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    if (c < 0x20 || c === 0x7f || c === 0x2028 || c === 0x2029) return true;
  }
  return false;
};

// Fields each node type actually declares. Anything else is dropped on import
// (sanitize rebuilds every node from scratch rather than spreading the input),
// so an unrecognised key is a change that will silently not happen.
const F_FLAT = [
  "name",
  "note",
  "addC",
  "iters",
  "deOption",
  "ops",
  "camera",
  "julia",
  "juliaC",
];
const F_OP = ["key", "values", "muted"];
const F_CAMERA = ["yawDeg", "pitchDeg", "dist", "fovDeg", "target"];
const F_HYBRID = ["b", "slots", "schedule", "aMuted"];
const F_SLOT = ["ops", "addC", "muted"];
const F_SCHEDULE = ["a", "b", "counts"];
const F_TRANSFORM = ["origin", "uscale", "rot"];
// The scene-object union: both the legacy `objType` form and the D0 `shapeId`
// form, plus the flat transform aliases sanitize still reads off the object.
const F_OBJECT = [
  "objType",
  "shapeId",
  "shapeParams",
  "iterShape",
  "ops",
  "iters",
  "addC",
  "deOption",
  "transform",
  "origin",
  "uscale",
  "rot",
  "combine",
  "combineType",
  "blendK",
  "looseDE",
  "color",
  "name",
  "muted",
  "julia",
  "juliaC",
  "boxBase",
  "primParam",
  "primParam2",
  "halfExtent",
  "radius",
];

const isObj = (v) => !!v && typeof v === "object" && !Array.isArray(v);
const isNum = (v) => typeof v === "number" && Number.isFinite(v);

/**
 * A collector, so every check can just report and move on.
 *
 * Errors and warnings are gathered SEPARATELY and capped separately. A single
 * bucket would let a few hundred warnings crowd out the one error that made the
 * document invalid, and `ok: false` with nothing listed under it is worse than
 * useless — it is a rejection nobody can act on.
 */
function makeCtx() {
  return {
    errs: [],
    warns: [],
    sawError: false,
    truncated: false,
    push(bucket, severity, code, message, where) {
      if (bucket.length >= MAX_FINDINGS) {
        this.truncated = true;
        return;
      }
      bucket.push({ severity, code, message, ...(where ? { where } : {}) });
    },
    err(code, message, where) {
      // The verdict is set before the cap is consulted, so truncating the report
      // can never flip ok from false to true.
      this.sawError = true;
      this.push(this.errs, "error", code, message, where);
    },
    warn(code, message, where) {
      this.push(this.warns, "warning", code, message, where);
    },
  };
}

/** Report keys the format does not define. Sorted so the message is stable. */
function unknownFields(ctx, node, allowed, where, label) {
  const extra = Object.keys(node)
    .filter((k) => !allowed.includes(k))
    .sort();
  if (extra.length)
    ctx.warn(
      "unknown-field",
      `${label} has ${extra.length === 1 ? "an unrecognised field" : "unrecognised fields"} ${extra
        .map((k) => `"${k}"`)
        .join(", ")} — dropped on import, so any change expressed through ${
        extra.length === 1 ? "it" : "them"
      } will not happen.`,
      where,
    );
}

// sanitize's num/clampNum/clampInt all run the raw value through Number()
// BEFORE testing finiteness, so "2", true, null and [] are numbers to the
// importer (2, 1, 0, 0). Mirroring that coercion is not pedantry: `objType:
// true` really does mean objType 1, which discards the object's op chain and
// therefore decides whether an unknown operator inside it is ever seen. Get the
// coercion wrong and the verdict diverges from sanitize on exactly those seams.
const coerce = (v) => {
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
};
/** Exactly sanitize's clampInt. */
const clampIntLike = (v, lo, hi, d) => {
  const n = Math.round(Number(v));
  return Number.isFinite(n) ? Math.max(lo, Math.min(hi, n)) : d;
};

/** Report a value that is not a JSON number but which the importer coerces. */
function noteCoercion(ctx, v, n, where, label) {
  if (typeof v === "number") return;
  ctx.warn(
    "not-a-number",
    n === undefined
      ? `${label} is ${JSON.stringify(v) ?? String(v)}, which is not a number — replaced with the default on import.`
      : `${label} is ${JSON.stringify(v) ?? String(v)}, not a JSON number — coerced to ${n} on import.`,
    where,
  );
}

/** A scalar sanitize will clamp into [lo, hi] (or replace when non-numeric). */
function checkScalar(ctx, v, lo, hi, where, label) {
  if (v === undefined) return;
  const n = coerce(v);
  noteCoercion(ctx, v, n, where, label);
  if (n === undefined) return;
  if (n < lo || n > hi)
    ctx.warn(
      "out-of-range",
      `${label} is ${n}, outside ${lo}..${hi} — clamped on import.`,
      where,
    );
}

/** A scalar sanitize ROUNDS and then clamps (clampInt) — objType, iters, … */
function checkInt(ctx, v, lo, hi, where, label) {
  if (v === undefined) return;
  const n = coerce(v);
  noteCoercion(ctx, v, n, where, label);
  if (n === undefined) return;
  const settled = Math.max(lo, Math.min(hi, Math.round(n)));
  if (settled !== n)
    ctx.warn(
      "out-of-range",
      `${label} is ${n}; it must be a whole number in ${lo}..${hi} and is read as ${settled} on import.`,
      where,
    );
}

/** A fixed-length numeric vector (origin, color, juliaC, rot…). */
function checkVec(ctx, v, len, where, label, lo, hi) {
  if (v === undefined) return;
  if (!Array.isArray(v)) {
    ctx.warn(
      "not-an-array",
      `${label} is not an array — replaced with the default on import.`,
      where,
    );
    return;
  }
  if (v.length !== len)
    ctx.warn(
      "wrong-length",
      `${label} has ${v.length} entries, expected ${len} — the import pads with 0 or ignores the extras.`,
      where,
    );
  for (let i = 0; i < Math.min(v.length, len); i++)
    checkScalar(ctx, v[i], lo, hi, `${where}[${i}]`, `${label}[${i}]`);
}

function checkText(ctx, v, max, where, label) {
  if (v === undefined) return;
  if (typeof v !== "string") {
    ctx.warn(
      "not-a-string",
      `${label} is not a string — replaced with the default on import.`,
      where,
    );
    return;
  }
  if (hasControl(v))
    ctx.warn(
      "control-chars",
      `${label} contains control characters — each becomes a space on import.`,
      where,
    );
  if (v.length > max)
    ctx.warn(
      "too-long",
      `${label} is ${v.length} characters; it is cut to ${max} on import.`,
      where,
    );
}

/**
 * One op-list. `capped` is the number of ops sanitize will look at before
 * slicing. EVERY op-list slices first — flat, scene object, and (since #542)
 * hybrid slots too — so an unknown operator past the cap is never seen and
 * never an error, only a truncation warning.
 *
 * Returns the packed param count of the ops that survive, so the hybrid caller
 * can check it against MAX_PARAMS the way sanitize does.
 */
function checkOps(
  ctx,
  ops,
  where,
  { capped = Infinity, numericOk = true } = {},
) {
  if (ops === undefined) return 0;
  if (!Array.isArray(ops)) {
    ctx.warn(
      "not-an-array",
      `"ops" is not an array — treated as empty on import.`,
      where,
    );
    return 0;
  }
  if (Number.isFinite(capped) && ops.length > capped)
    ctx.warn(
      "truncated",
      `${ops.length} ops, but only the first ${capped} are kept — the rest are dropped on import.`,
      where,
    );
  const seen = Math.min(ops.length, capped);
  let params = 0;
  for (let i = 0; i < seen; i++) {
    const o = ops[i];
    const at = `${where}[${i}]`;
    if (!isObj(o)) {
      // sanitize calls byKey(o && o.key) — a non-object has no key, so this is
      // an unknown-operator ERROR, not a shape warning.
      ctx.err(
        "unknown-op",
        `Op #${i + 1} is ${Array.isArray(o) ? "an array" : JSON.stringify(o)}, not an operator object with a "key".`,
        at,
      );
      continue;
    }
    unknownFields(ctx, o, F_OP, at, `Op #${i + 1}`);
    const def = byKey(o.key);
    if (!def) {
      ctx.err(
        "unknown-op",
        `Unknown operator ${JSON.stringify(o.key)} — operator keys must come from the registry verbatim.`,
        at,
      );
      continue;
    }
    if (!numericOk && def.wRule === W_BULB_NUMERIC) {
      ctx.err(
        "numeric-de",
        `Operator "${def.key}" uses the numeric distance estimator, which is only supported in a flat formula — not in a hybrid slot or a scene object.`,
        at,
      );
      continue;
    }
    params += def.params.length;
    if (o.muted !== undefined && typeof o.muted !== "boolean")
      ctx.warn(
        "not-a-boolean",
        `Op #${i + 1} "muted" is ${JSON.stringify(o.muted)} — read as ${o.muted ? "true" : "false"} on import.`,
        `${at}.muted`,
      );
    const want = def.params.length;
    if (!Array.isArray(o.values)) {
      if (want)
        ctx.warn(
          "values-arity",
          `Op #${i + 1} ("${def.key}") has no "values" array but declares ${want} param${want === 1 ? "" : "s"} — every one falls back to its default.`,
          `${at}.values`,
        );
      continue;
    }
    if (o.values.length !== want)
      ctx.warn(
        "values-arity",
        `Op #${i + 1} ("${def.key}") has ${o.values.length} value${o.values.length === 1 ? "" : "s"} but declares ${want} param${want === 1 ? "" : "s"} — "values" is positional, so ${
          o.values.length > want
            ? "the extras are dropped"
            : "the missing ones fall back to defaults"
        }.`,
        `${at}.values`,
      );
    def.params.forEach((p, pi) => {
      const v = o.values[pi];
      if (v === undefined) return;
      const vw = `${at}.values[${pi}]`;
      const n = coerce(v);
      if (typeof v !== "number")
        ctx.warn(
          "not-a-number",
          n === undefined
            ? `"${def.key}".${p.name} is ${JSON.stringify(v) ?? String(v)}, which is not a number — it becomes ${p.default} (the declared default) on import.`
            : `"${def.key}".${p.name} is ${JSON.stringify(v) ?? String(v)}, not a JSON number — coerced to ${n} on import.`,
          vw,
        );
      if (n === undefined) return;
      // Registry ranges are NORMATIVE and, since #538/#542, ENFORCED: the
      // importer clamps to [min,max] rather than refusing. So this stays a
      // warning (the document loads) but it is now a LOSSY one — the value the
      // producer wrote is not the value the consumer gets.
      if (n < p.min || n > p.max)
        ctx.warn(
          "out-of-range",
          `"${def.key}".${p.name} is ${n}, outside its declared range ${p.min}..${p.max} — clamped to ${Math.max(p.min, Math.min(p.max, n))} on import.`,
          vw,
        );
    });
  }
  return params;
}

function checkCamera(ctx, cam) {
  if (cam === undefined) return;
  if (!isObj(cam)) {
    ctx.warn(
      "not-an-object",
      `"camera" is not an object — the default view is used instead.`,
      "$.camera",
    );
    return;
  }
  unknownFields(ctx, cam, F_CAMERA, "$.camera", "camera");
  // yaw/pitch are unbounded (any angle is meaningful); dist and fov are clamped.
  for (const k of ["yawDeg", "pitchDeg"])
    if (cam[k] !== undefined && !isNum(cam[k]))
      ctx.warn(
        "not-a-number",
        `camera.${k} is ${JSON.stringify(cam[k])}, not a finite number — replaced with the default on import.`,
        `$.camera.${k}`,
      );
  checkScalar(ctx, cam.dist, 1e-6, 1e9, "$.camera.dist", "camera.dist");
  checkScalar(ctx, cam.fovDeg, 1, 179, "$.camera.fovDeg", "camera.fovDeg");
  checkVec(
    ctx,
    cam.target,
    3,
    "$.camera.target",
    "camera.target",
    -Infinity,
    Infinity,
  );
}

/** The fields every shape shares — the flat formula body. */
function checkFlatBody(ctx, f, { requireOps }) {
  if (requireOps && !Array.isArray(f.ops)) {
    ctx.err(
      "missing-ops",
      `A formula needs an "ops" array (it may be empty). Found ${
        f.ops === undefined ? 'no "ops" field' : JSON.stringify(f.ops)
      }.`,
      "$.ops",
    );
  }
  checkText(ctx, f.name, MAX_NAME, "$.name", `"name"`);
  checkText(ctx, f.note, MAX_NOTE, "$.note", `"note"`);
  checkInt(ctx, f.iters, 2, MAX_ITERS, "$.iters", `"iters"`);
  checkInt(ctx, f.deOption, 0, 3, "$.deOption", `"deOption"`);
  checkCamera(ctx, f.camera);
  if (f.juliaC !== undefined && !f.julia)
    ctx.warn(
      "orphan-field",
      `"juliaC" is present but "julia" is not true — the seed is dropped on import.`,
      "$.juliaC",
    );
  if (f.julia)
    checkVec(ctx, f.juliaC, 3, "$.juliaC", `"juliaC"`, -Infinity, Infinity);
}

function checkHybrid(ctx, f) {
  const h = f.hybrid;
  if (Array.isArray(h))
    ctx.warn(
      "not-an-object",
      `"hybrid" is an array. The importer still treats the formula AS a hybrid (which bans the flat-only operators) but finds no slot B in it.`,
      "$.hybrid",
    );
  else unknownFields(ctx, h, F_HYBRID, "$.hybrid", "hybrid");
  const flatOps = Array.isArray(f.ops) ? f.ops.slice(0, MAX_FLAT_OPS) : [];
  // Slot A is the formula body itself; it is re-checked here for the numeric-DE
  // rule, which only applies once the formula is a hybrid.
  let total = checkOps(ctx, flatOps, "$.ops", { numericOk: false });

  const nSlot = Array.isArray(h.slots) && h.slots.length > 0;
  if (nSlot) {
    // Since #542, sanitize SLICES slots[] to the product cap BEFORE mapping, so
    // a slot past the cap is never looked at: its ops are not validated and an
    // unknown operator in one is NOT an error, merely dropped. (Before #542 the
    // slice came after the map and such a slot did throw — this validator
    // mirrored that, which is exactly what the #542/#544 merge collided on.)
    const kept = Math.min(h.slots.length, HYBRID_MAX_SLOTS - 1);
    if (h.slots.length > kept)
      ctx.warn(
        "truncated",
        `${h.slots.length + 1} slots (slot A plus ${h.slots.length}), but the cap is ${HYBRID_MAX_SLOTS} — the extras are dropped on import, unvalidated.`,
        "$.hybrid.slots",
      );
    h.slots.slice(0, kept).forEach((s, i) => {
      const at = `$.hybrid.slots[${i}]`;
      const letter = String.fromCharCode(66 + i);
      if (!isObj(s)) {
        ctx.warn(
          "not-an-object",
          `Slot ${letter} is not an object — imported as an empty slot.`,
          at,
        );
        return;
      }
      unknownFields(ctx, s, F_SLOT, at, `Slot ${letter}`);
      // Each slot's ops are capped exactly like slot A's (#542 item 2).
      total += checkOps(ctx, s.ops, `${at}.ops`, {
        capped: MAX_FLAT_OPS,
        numericOk: false,
      });
    });
    const sched = h.schedule;
    if (sched !== undefined && !isObj(sched))
      ctx.warn(
        "not-an-object",
        `"hybrid.schedule" is not an object — every slot runs once per period.`,
        "$.hybrid.schedule",
      );
    else if (isObj(sched)) {
      unknownFields(
        ctx,
        sched,
        F_SCHEDULE,
        "$.hybrid.schedule",
        "hybrid.schedule",
      );
      const want = kept + 1;
      if (sched.counts === undefined)
        ctx.warn(
          "missing-field",
          `A ${want}-slot hybrid needs "schedule.counts" (one count per slot, slot A first) — every slot defaults to 1.`,
          "$.hybrid.schedule",
        );
      else if (!Array.isArray(sched.counts))
        ctx.warn(
          "not-an-array",
          `"schedule.counts" is not an array — every slot defaults to 1.`,
          "$.hybrid.schedule.counts",
        );
      else {
        if (sched.counts.length !== want)
          ctx.warn(
            "wrong-length",
            `"schedule.counts" has ${sched.counts.length} entries but there are ${want} slots (counts[0] is slot A) — missing counts default to 1.`,
            "$.hybrid.schedule.counts",
          );
        sched.counts
          .slice(0, want)
          .forEach((c, i) =>
            checkInt(
              ctx,
              c,
              1,
              8,
              `$.hybrid.schedule.counts[${i}]`,
              `schedule.counts[${i}]`,
            ),
          );
        const period = sched.counts
          .slice(0, want)
          .reduce((n, c) => n + clampIntLike(c, 1, 8, 1), 0);
        if (period > 16)
          ctx.warn(
            "out-of-range",
            `The schedule period is ${period}; the cap is 16 — the largest counts are trimmed on import.`,
            "$.hybrid.schedule.counts",
          );
      }
    }
  } else {
    if (h.slots !== undefined && !Array.isArray(h.slots))
      ctx.warn(
        "not-an-array",
        `"hybrid.slots" is not an array — the legacy two-slot shape is used instead.`,
        "$.hybrid.slots",
      );
    const b = h.b;
    if (b !== undefined && !isObj(b))
      ctx.warn(
        "not-an-object",
        `"hybrid.b" is not an object — slot B imports empty.`,
        "$.hybrid.b",
      );
    if (b === undefined)
      ctx.warn(
        "missing-field",
        `A two-slot hybrid needs "hybrid.b" — slot B imports empty, which renders as a plain flat formula.`,
        "$.hybrid",
      );
    if (isObj(b)) {
      unknownFields(ctx, b, F_SLOT, "$.hybrid.b", "Slot B");
      total += checkOps(ctx, b.ops, "$.hybrid.b.ops", {
        capped: MAX_FLAT_OPS,
        numericOk: false,
      });
    }
    const sched = h.schedule;
    if (sched !== undefined && !isObj(sched))
      ctx.warn(
        "not-an-object",
        `"hybrid.schedule" is not an object — both slots default to 1.`,
        "$.hybrid.schedule",
      );
    else if (isObj(sched)) {
      unknownFields(
        ctx,
        sched,
        F_SCHEDULE,
        "$.hybrid.schedule",
        "hybrid.schedule",
      );
      if (sched.counts !== undefined)
        ctx.warn(
          "wrong-shape",
          `"schedule.counts" is the three-or-more-slot form; a two-slot hybrid uses "schedule": { "a": …, "b": … } and ignores counts.`,
          "$.hybrid.schedule.counts",
        );
      checkInt(ctx, sched.a, 1, 8, "$.hybrid.schedule.a", "schedule.a");
      checkInt(ctx, sched.b, 1, 8, "$.hybrid.schedule.b", "schedule.b");
      const a = clampIntLike(sched.a, 1, 8, 1);
      const bb = clampIntLike(sched.b, 1, 8, 1);
      if (a + bb > 12)
        ctx.warn(
          "out-of-range",
          `The schedule period is ${a + bb}; a two-slot hybrid caps at 12 — the larger count is reduced on import.`,
          "$.hybrid.schedule",
        );
    }
  }

  if (total > MAX_PARAMS)
    ctx.err(
      "over-cap",
      `The hybrid packs ${total} operator params across its slots; the shared uniform budget is ${MAX_PARAMS}. Shorten a slot.`,
      "$.hybrid",
    );
}

function checkObject(ctx, o, oi) {
  const at = `$.objects[${oi}]`;
  const label = `Object #${oi + 1}`;
  // sanitize's own guard: `!o || typeof o !== "object"`. An array slips through
  // that check, so it is NOT an error here either — it just imports as defaults.
  if (!o || typeof o !== "object") {
    ctx.err(
      "bad-object",
      `${label} is ${JSON.stringify(o)}, not an object.`,
      at,
    );
    return;
  }
  if (Array.isArray(o)) {
    ctx.warn(
      "not-an-object",
      `${label} is an array — it imports as a default object with nothing of its own.`,
      at,
    );
    return;
  }
  unknownFields(ctx, o, F_OBJECT, at, label);

  const isD0 = o.shapeId !== undefined && o.shapeId !== null;
  if (isD0) {
    checkInt(ctx, o.shapeId, 0, 255, `${at}.shapeId`, `${label} "shapeId"`);
    const id = clampIntLike(o.shapeId, 0, 255, 0);
    const leaf = leafById(id);
    if (id > MAX_LEAF_ID)
      ctx.warn(
        "unknown-leaf",
        `${label} uses shape leaf ${id}, beyond the highest one this build knows (${MAX_LEAF_ID}). It is preserved rather than clamped, but renders as a fallback here.`,
        `${at}.shapeId`,
      );
    if (leaf && Array.isArray(o.shapeParams))
      leaf.params.forEach((p, j) =>
        checkScalar(
          ctx,
          o.shapeParams[j],
          p.min,
          p.max,
          `${at}.shapeParams[${j}]`,
          `${label} ${leaf.key}.${p.name}`,
        ),
      );
    // The op chain is kept ALONGSIDE the leaf in the D0 form, and validated.
    checkOps(ctx, o.ops, `${at}.ops`, {
      capped: MAX_OPS_PER_OBJECT,
      numericOk: false,
    });
  } else {
    checkInt(ctx, o.objType, 0, 6, `${at}.objType`, `${label} "objType"`);
    const t = clampIntLike(o.objType, 0, 6, 0);
    if (t === 0) {
      checkOps(ctx, o.ops, `${at}.ops`, {
        capped: MAX_OPS_PER_OBJECT,
        numericOk: false,
      });
    } else if (Array.isArray(o.ops) && o.ops.length) {
      // A legacy primitive object never runs an op chain — sanitize replaces the
      // list with []. Silently. Worth saying out loud.
      ctx.warn(
        "dropped",
        `${label} is a primitive (objType ${t}), so its ${o.ops.length} op${o.ops.length === 1 ? "" : "s"} are dropped on import. Use the "shapeId" form to keep an op chain alongside a shape.`,
        `${at}.ops`,
      );
    }
  }

  checkInt(ctx, o.iters, 1, 24, `${at}.iters`, `${label} "iters"`);
  checkInt(ctx, o.deOption, 0, 3, `${at}.deOption`, `${label} "deOption"`);
  checkInt(
    ctx,
    o.combine ?? o.combineType,
    0,
    3,
    `${at}.combine`,
    `${label} "combine"`,
  );
  checkScalar(ctx, o.blendK, 0, 10, `${at}.blendK`, `${label} "blendK"`);
  checkVec(ctx, o.color, 3, `${at}.color`, `${label} "color"`, 0, 1);
  checkText(ctx, o.name, MAX_OBJ_NAME, `${at}.name`, `${label} "name"`);
  const tr = isObj(o.transform) ? o.transform : o;
  const trAt = isObj(o.transform) ? `${at}.transform` : at;
  if (o.transform !== undefined && !isObj(o.transform))
    ctx.warn(
      "not-an-object",
      `${label} "transform" is not an object — the identity transform is used.`,
      `${at}.transform`,
    );
  if (isObj(o.transform))
    unknownFields(ctx, o.transform, F_TRANSFORM, trAt, `${label} transform`);
  checkVec(ctx, tr.origin, 3, `${trAt}.origin`, `${label} origin`, -1e4, 1e4);
  checkScalar(ctx, tr.uscale, 1e-4, 1e4, `${trAt}.uscale`, `${label} uscale`);
  if (tr.rot !== undefined) {
    if (!Array.isArray(tr.rot))
      ctx.warn(
        "not-an-array",
        `${label} "rot" is not an array — no rotation is applied.`,
        `${trAt}.rot`,
      );
    else if (tr.rot.length !== 3 && tr.rot.length !== 4)
      ctx.warn(
        "wrong-length",
        `${label} "rot" has ${tr.rot.length} entries; it must be 3 (Euler XYZ degrees) or 4 (quaternion).`,
        `${trAt}.rot`,
      );
  }
}

function checkScene(ctx, f) {
  if (!Array.isArray(f.objects)) {
    ctx.err(
      "missing-objects",
      `A scene needs an "objects" array.`,
      "$.objects",
    );
    return;
  }
  if (f.objects.length > MAX_OBJECTS)
    ctx.warn(
      "truncated",
      `${f.objects.length} objects, but the cap is ${MAX_OBJECTS} — the extras are dropped on import.`,
      "$.objects",
    );
  // Only the objects that survive the slice are validated — mirroring sanitize,
  // which slices BEFORE it maps, so a bad object past the cap is never seen.
  f.objects.slice(0, MAX_OBJECTS).forEach((o, i) => checkObject(ctx, o, i));
  if (f.coloring !== undefined && !isObj(f.coloring))
    ctx.warn(
      "not-an-object",
      `"coloring" is not an object — dropped on import.`,
      "$.coloring",
    );
}

/**
 * Validate a parsed formula document against the Formula JSON spec v1.
 *
 * @param {unknown} json  a value already through JSON.parse — this never parses
 *   text, so a caller keeps control of where syntax errors are reported.
 * @returns {{ ok: boolean, errors: Array<{severity: "error"|"warning", code: string, message: string, where?: string}> }}
 *   `ok` is true when nothing of severity "error" was found, which is exactly
 *   when core/sanitize.js will accept the document. `errors` carries BOTH
 *   severities, newest checks last; filter on `severity` to separate "will not
 *   load" from "will load, but not unchanged".
 */
export function validate(json) {
  const ctx = makeCtx();

  if (!json || typeof json !== "object") {
    ctx.err(
      "not-an-object",
      `A formula must be a JSON object. Found ${json === null ? "null" : typeof json}.`,
      "$",
    );
    return { ok: false, errors: ctx.errs };
  }
  if (Array.isArray(json)) {
    // sanitize gets past its typeof guard and dies on the missing ops array.
    ctx.err(
      "missing-ops",
      `A formula must be a JSON object, not an array.`,
      "$",
    );
    return { ok: false, errors: ctx.errs };
  }

  // Shape routing, mirroring sanitizeFormula: a non-empty objects[] wins over
  // hybrid, and a formula with neither is flat.
  const isScene = Array.isArray(json.objects) && json.objects.length > 0;
  // sanitizeFormula routes on `obj.hybrid && typeof obj.hybrid === "object"`,
  // and an ARRAY satisfies that — `"hybrid": []` really is a hybrid to the
  // importer, which is enough to make its flat-only operator rule bite.
  const isHybrid = !isScene && !!json.hybrid && typeof json.hybrid === "object";

  const allowed = [...F_FLAT];
  if (isScene) allowed.push("objects", "coloring");
  if (isHybrid) allowed.push("hybrid");
  unknownFields(ctx, json, allowed, "$", "The formula");

  if (isScene && json.hybrid !== undefined)
    ctx.warn(
      "dropped",
      `"objects" and "hybrid" are mutually exclusive; a document carrying both imports as the SCENE and the hybrid is dropped entirely.`,
      "$.hybrid",
    );
  if (!isScene && json.coloring !== undefined)
    ctx.warn(
      "dropped",
      `"coloring" is only carried on a scene — on a flat or hybrid formula it is dropped on import.`,
      "$.coloring",
    );
  if (Array.isArray(json.objects) && json.objects.length === 0)
    ctx.warn(
      "dropped",
      `An empty "objects" array is not a scene — the document imports as a flat formula and "objects" disappears.`,
      "$.objects",
    );

  // A scene may omit the top-level "ops" entirely; every other shape requires it.
  checkFlatBody(ctx, json, { requireOps: !isScene });

  if (isScene) {
    checkOps(ctx, json.ops, "$.ops", { capped: MAX_FLAT_OPS });
    checkScene(ctx, json);
  } else if (isHybrid) {
    checkHybrid(ctx, json);
  } else {
    checkOps(ctx, json.ops, "$.ops", { capped: MAX_FLAT_OPS });
  }

  if (ctx.truncated)
    ctx.warns.push({
      severity: "warning",
      code: "report-truncated",
      message: `More than ${MAX_FINDINGS} findings of one severity; the rest are not listed. The ok/failed verdict is still exact.`,
      where: "$",
    });

  // Errors first, then warnings — each in document order. A consumer that only
  // renders the first few lines should see the blocking problems.
  return { ok: !ctx.sawError, errors: [...ctx.errs, ...ctx.warns] };
}

/** Convenience for CLIs and logs: one line per finding. */
export function formatFindings(errors) {
  return errors
    .map((e) => `${e.severity}: ${e.message}${e.where ? `  (${e.where})` : ""}`)
    .join("\n");
}

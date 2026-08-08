// Validate + coerce an arbitrary object (pasted JSON / saved library entry) into
// a safe formula. Shared importer for both frontends. Throws on unknown ops.

import { byKey, W_BULB_NUMERIC } from "./operators.js";
import { leafById, MAX_LEAF_ID } from "./leaves.js";
import {
  MAX_FLAT_OPS as LIMIT_MAX_FLAT_OPS,
  MAX_OBJECTS as LIMIT_MAX_OBJECTS,
  MAX_OPS_PER_OBJECT as LIMIT_MAX_OPS_PER_OBJECT,
  MAX_ITERS,
  MAX_PARAMS,
} from "./limits.js";
import { BLANK } from "./oplist.js";
import { HYBRID_MAX_SLOTS } from "./hybridmodel.js";
import { COLOR_MODE_MAX } from "./coloring.js";

const num = (v, d = 0) => {
  const n = Number(v);
  return Number.isFinite(n) ? n : d;
};
const clampNum = (v, lo, hi, d = 0) => {
  const n = Number(v);
  return Number.isFinite(n) ? Math.max(lo, Math.min(hi, n)) : d;
};
const clampInt = (v, lo, hi, d) => {
  const n = Math.round(Number(v));
  return Number.isFinite(n) ? Math.max(lo, Math.min(hi, n)) : d;
};
// User-facing text (name/note) is echoed into DOM (via textContent) and, for
// `name`, into the `// JIT formula: …` comment line of exported GLSL. Strip
// control chars — newlines especially would let a crafted name break out of the
// export comment into shader source — then length-slice.
const cleanText = (v, max, fallback) =>
  typeof v === "string"
    ? v.replace(/[\u0000-\u001F\u007F\u2028\u2029]/g, " ").slice(0, max)
    : fallback;

// Coerce an arbitrary camera object into finite numbers. A pasted/shared JSON
// with a missing or non-numeric field would otherwise reach makeCamera and yield
// NaN eye/fov → a permanently black render. Defaults come from BLANK (cloned).
const sanitizeCamera = (c) => {
  const d = BLANK.camera;
  const src = c && typeof c === "object" ? c : d;
  const cam = {
    yawDeg: num(src.yawDeg, d.yawDeg),
    pitchDeg: num(src.pitchDeg, d.pitchDeg),
    dist: clampNum(src.dist, 1e-6, 1e9, d.dist),
    fovDeg: clampNum(src.fovDeg, 1, 179, d.fovDeg),
  };
  if (Array.isArray(src.target))
    cam.target = [0, 1, 2].map((i) => num(src.target[i], 0));
  return cam;
};

// Cap on flat op-count. Single-sourced in limits.js — since the 2026-07-04
// unification it is the SMALLER (WebGL2) tier's value, so a flat formula renders
// identically on every tier (the WebGPU op buffer stays larger only to hold
// scene/hybrid/morph concatenations — see limits.js).
const MAX_FLAT_OPS = LIMIT_MAX_FLAT_OPS;

// Empty slate for the New button — single-sourced in oplist.js (this module had
// its own copy, and the two had already drifted on `note`). Re-exported so
// existing importers keep working.
export { BLANK };

// Validate one op-list (throws on an unknown operator); fills missing params.
// Preserves per-op `muted` (emit-only-when-true) — dropping it here silently
// un-muted ops on every reload of a scene object / hybrid slot A / slot B.
// v1 scope guard (COVERAGE_PLAN §3 B1): numeric-DE ops (W_BULB_NUMERIC — no
// analytic dr, whole-formula finite-difference routing) are only supported in
// flat single formulas. Hybrid slots and scene objects have their own DE
// bodies that don't implement the numeric path yet; reject loudly rather than
// render silently wrong.
function rejectNumericOps(ops, where) {
  for (const op of ops) {
    if (byKey(op.key)?.wRule === W_BULB_NUMERIC)
      throw new Error(
        `operator "${op.key}"${where} uses the numeric DE, which isn't supported in hybrid/scene formulas yet — use it in a flat formula`,
      );
  }
  return ops;
}

// One op param value, clamped to its operator-registry range (#538 item 4).
// Finite was not enough: the share codec quantizes params on a FIXED 0.01 grid
// with an unbounded zigzag varint (sharecodec.js PARAM), so a crafted or
// hand-edited link can carry scale=1e12 / power=-1e9 — finite, and enough to
// drive the marcher into inf/NaN or to hang a tab on a pathological orbit.
// The registry's [min,max] is the only declared per-param domain there is, so
// it is the clamp. Values are NOT rounded to `step` — quantizing here would
// move in-range values that ride between steps (share links quantize to 0.01
// independently) — and a param without a finite min/max keeps the old
// finite-only coercion rather than being clamped to garbage.
const clampParam = (v, p) =>
  Number.isFinite(p?.min) && Number.isFinite(p?.max)
    ? clampNum(v, p.min, p.max, p.default)
    : num(v, p.default);

function sanitizeOps(rawOps, where = "") {
  return rawOps.map((o, i) => {
    const def = byKey(o && o.key);
    if (!def)
      throw new Error(
        `unknown operator "${o && o.key}"${where} at op #${i + 1}`,
      );
    return {
      key: def.key,
      values: def.params.map((p, pi) =>
        clampParam(Array.isArray(o.values) ? o.values[pi] : undefined, p),
      ),
      ...(o.muted ? { muted: true } : {}),
    };
  });
}

// Flat (single-object) formula — today's exact shape, unchanged.
function sanitizeFlat(obj) {
  if (!obj || typeof obj !== "object") throw new Error("not an object");
  if (!Array.isArray(obj.ops)) throw new Error('missing "ops" array');
  const f = {
    name: cleanText(obj.name, 60, "Imported"),
    note: cleanText(obj.note, 120, ""),
    addC: !!obj.addC,
    // Base detail. Upper bound is the engine/GLSL iteration cap (limits.js
    // MAX_ITERS), not the old UI cap of 24 — auto-detail (preview.js §6) and its
    // 2–64 "Detail" slider can carry the count that high, so a saved/shared high
    // base must survive.
    iters: clampInt(obj.iters, 2, MAX_ITERS, 12),
    deOption: clampInt(obj.deOption, 0, 3, 2),
    ops: sanitizeOps(obj.ops.slice(0, MAX_FLAT_OPS)),
    camera: sanitizeCamera(obj.camera),
  };
  if (obj.julia) {
    f.julia = true;
    // Pad to exactly 3 — a short juliaC (e.g. [1]) would otherwise write
    // `undefined` into the renderer's Float32Array and NaN-poison the render.
    f.juliaC = [0, 1, 2].map((i) =>
      num(Array.isArray(obj.juliaC) ? obj.juliaC[i] : undefined),
    );
  }
  return f;
}

export function sanitizeFormula(obj) {
  // A formula carrying objects[] is a scene — validate it as such (§3.5). A flat
  // formula (no objects) goes through the single-object path unchanged.
  if (
    obj &&
    typeof obj === "object" &&
    Array.isArray(obj.objects) &&
    obj.objects.length
  ) {
    return sanitizeScene(obj);
  }
  // Hybrid iteration (IDEAS ①, docs/design/HYBRID_ITERATION.md §3.7) — mutually
  // exclusive with objects[] (§3.8); the objects[] check above already wins ties
  // on malformed input carrying both (conservative — existing behavior wins).
  if (
    obj &&
    typeof obj === "object" &&
    obj.hybrid &&
    typeof obj.hybrid === "object"
  ) {
    return sanitizeHybrid(obj);
  }
  return sanitizeFlat(obj);
}

// Hybrid iteration (docs/design/HYBRID_ITERATION.md §3.7) — validate + coerce a
// hybrid formula. Slot A is the base sanitizeFlat result (ops/addC/deOption/
// julia/juliaC/camera/iters all stay formula-level, §3.3/§3.8); slot B's ops
// are validated the same way. Schedule clamps to a,b ∈ 1..8 with a+b ≤ 12 —
// reduce the LARGER count first so both stay ≥ 1 and the sum never exceeds 12.
// DOES NOT enforce the same-family DE-safety rule here (that's hybridDeFamily,
// checked by the caller / health badge, §3.9) — sanitize only shapes the data;
// v1 UI is expected to reject/flag 'mixed' before it reaches here, but a
// hand-crafted import can still carry one, so mixed formulas sanitize cleanly
// and rely on the health badge + render-side caution, not a thrown error.
export function sanitizeHybrid(obj) {
  const f = sanitizeFlat(obj);
  const h = obj.hybrid || {};
  // N-slot (≥3) new stored shape: the extra slots ride in `hybrid.slots[]`. PR-2
  // wired every tier for N slots, so the truncation is lifted — ≥3-slot formulas
  // now flow to the renderers (sanitizeHybridN clamps to the policy cap and keeps
  // the slots[] shape; a 2-slot import there normalizes back to the legacy shape).
  if (Array.isArray(h.slots) && h.slots.length > 0)
    return sanitizeHybridN(f, h);
  rejectNumericOps(f.ops, " in hybrid slot A");
  // Slot B's op-list is capped exactly like slot A's (#538 item 2). Slot A is
  // f.ops, already sliced to MAX_FLAT_OPS by sanitizeFlat; slot B had no cap at
  // all, so a link declaring 512 ops (sharecodec MAX_DECODE_OPS) got all 512
  // validated — and zero-param ops slip past the MAX_PARAMS check below, so
  // nothing downstream stopped them either. Slice BEFORE sanitizing so the work
  // is bounded too, not just the result.
  const bOps = rejectNumericOps(
    sanitizeOps(
      Array.isArray(h.b?.ops) ? h.b.ops.slice(0, MAX_FLAT_OPS) : [],
      " in hybrid slot B",
    ),
    " in hybrid slot B",
  );
  const sched = h.schedule || {};
  let a = clampInt(sched.a, 1, 8, 1);
  let b = clampInt(sched.b, 1, 8, 1);
  if (a + b > 12) {
    if (a >= b) a = 12 - b;
    else b = 12 - a;
  }
  // Both slots share the WebGL2 tier's one uP[] uniform array (limits.js
  // MAX_PARAMS), so the packed total across A+B must fit it.
  const totalParams =
    f.ops.reduce((n, o) => n + o.values.length, 0) +
    bOps.reduce((n, o) => n + o.values.length, 0);
  if (totalParams > MAX_PARAMS)
    throw new Error(`hybrid formula params ${totalParams} > cap ${MAX_PARAMS}`);
  // Per-slot mute (Formula Outline PR 0.2): `b.muted` skips slot B at upload,
  // `aMuted` (on hybrid — slot A has no object of its own, it IS formula.ops)
  // skips slot A. Emit-only-when-true; stripped by the app's engineView, never
  // read by the engine itself.
  f.hybrid = {
    b: { ops: bOps, addC: !!h.b?.addC, ...(h.b?.muted ? { muted: true } : {}) },
    schedule: { a, b },
    ...(h.aMuted ? { aMuted: true } : {}),
  };
  return f;
}

// N-slot (≥3) hybrid sanitizer (HYBRID_NSLOT_SPEC.md §2.1/§3). Slot A is the
// formula body (f.ops / f.addC, already sanitized); the extra slots ride in
// `h.slots[]`. EVERY slot's ops are validated + numeric-DE-rejected (§2.6 — no
// slot escapes the check). The slot COUNT clamps to HYBRID_MAX_SLOTS; each
// schedule count clamps to 1..8 with total period ≤ 16; the packed params across
// ALL slots must fit the WebGL2 uP[] budget. The result keeps the slots[] stored
// shape for ≥3 slots, but a formula that reduces to exactly 2 slots re-emits the
// LEGACY {b, schedule:{a,b}} shape (§2.1 — byte-identical old saves, zero fixture
// churn; the accessor is the single reader of both shapes).
function sanitizeHybridN(f, h) {
  const label = (i) => ` in hybrid slot ${String.fromCharCode(66 + i)}`; // B, C…
  rejectNumericOps(f.ops, " in hybrid slot A");
  // Per-slot mute (the eye toggle) — slot A rides `hybrid.aMuted`, every extra
  // slot its own `slots[i].muted`. Carried on the working slot objects here and
  // re-emitted (emit-only-when-true) in whichever stored shape wins below.
  // Slice to the product cap BEFORE mapping (#538 item 1). The cap used to be
  // applied to the finished array, so every slot in an oversized h.slots[] was
  // fully sanitized (and its shape accepted) first — a link declaring 10k slots
  // did 10k slots' worth of op validation to then throw all but 3 away. Slot A
  // holds index 0, so at most HYBRID_MAX_SLOTS - 1 extras survive; that makes
  // the old post-slice exact and therefore unnecessary. Each slot's ops are
  // capped like slot A's (#538 item 2), again before sanitizing.
  const slots = [
    { ops: f.ops, addC: !!f.addC, muted: !!h.aMuted },
    ...h.slots.slice(0, HYBRID_MAX_SLOTS - 1).map((s, i) => ({
      ops: rejectNumericOps(
        sanitizeOps(
          Array.isArray(s?.ops) ? s.ops.slice(0, MAX_FLAT_OPS) : [],
          label(i),
        ),
        label(i),
      ),
      addC: !!s?.addC,
      muted: !!s?.muted,
    })),
  ];
  const rawCounts = Array.isArray(h.schedule?.counts) ? h.schedule.counts : [];
  const counts = slots.map((_, i) => clampInt(rawCounts[i], 1, 8, 1));
  // Total period ≤ 16 — trim the largest counts first, never below 1.
  let period = counts.reduce((n, c) => n + c, 0);
  while (period > 16) {
    let mi = 0;
    for (let i = 1; i < counts.length; i++) if (counts[i] > counts[mi]) mi = i;
    if (counts[mi] <= 1) break;
    counts[mi]--;
    period--;
  }
  // All slots share the WebGL2 tier's one uP[] uniform array (MAX_PARAMS).
  const totalParams = slots.reduce(
    (n, s) => n + s.ops.reduce((m, o) => m + o.values.length, 0),
    0,
  );
  if (totalParams > MAX_PARAMS)
    throw new Error(`hybrid formula params ${totalParams} > cap ${MAX_PARAMS}`);
  // Exactly 2 slots reduce to the LEGACY stored shape (slot A in f.ops/f.addC,
  // slot B in .b), re-applying the EXACT legacy schedule rule (1..8, a+b <= 12)
  // so a 2-slot new-shape import is byte-identical to a hand-authored legacy one.
  if (slots.length <= 2) {
    const bSlot = slots[1] || { ops: [], addC: false };
    let a = clampInt(counts[0], 1, 8, 1);
    let b = clampInt(counts[1] ?? 1, 1, 8, 1);
    if (a + b > 12) {
      if (a >= b) a = 12 - b;
      else b = 12 - a;
    }
    f.hybrid = {
      b: {
        ops: bSlot.ops,
        addC: !!bSlot.addC,
        ...(bSlot.muted ? { muted: true } : {}),
      },
      schedule: { a, b },
      ...(slots[0].muted ? { aMuted: true } : {}),
    };
    return f;
  }
  // >=3 slots keep the N-slot shape: slot A stays in f.ops/f.addC; the extras ride
  // in slots[]; schedule.counts is the FULL per-slot list (incl. A at index 0).
  // Per-slot mute rides slots[i].muted; slot A's rides hybrid.aMuted (emit-only-
  // when-true, so an unmuted hybrid is byte-identical to a pre-mute save).
  f.hybrid = {
    slots: slots.slice(1).map((s) => ({
      ops: s.ops,
      addC: s.addC,
      ...(s.muted ? { muted: true } : {}),
    })),
    schedule: { counts },
    ...(slots[0].muted ? { aMuted: true } : {}),
  };
  return f;
}

// ── Coloring ────────────────────────────────────────────────────────────────
// sanitizeScene used to do `f.coloring = obj.coloring` — a raw passthrough of
// untrusted data straight to the uniform writers (#538 item 3). The share
// codec's COLORING section is the hostile source: light scalars ride a
// zigzag/1000 fixed point (so ±2.1e6 decodes fine), `mode` is a full u8, and
// the stop count is a u8 the DECODER never caps even though the encoder and
// every consumer cap at 8. Downstream nothing rescues it — renderer_gl.js does
// `uniform3f(uColA, d.colA[0], d.colA[1], d.colA[2])` on whatever length array
// arrives (a short one writes NaN and blacks the frame), the shaders
// `normalize(uLightDir)` unguarded (a truncated payload decodes dir to [0,0,0]
// → NaN), and oklab.js `srgbToOklab` DESTRUCTURES a stop's `c`, so a stop
// missing it throws outright on the CPU/ASCII tier.
//
// SHAPE-PRESERVING BY CONSTRUCTION: only keys actually present on the input are
// emitted, and no sub-object is invented. That matters for byte stability — the
// 11 shipped themes carry just {mode, colA, colB, bg}, and encodeColoring keys
// its defaults off absence (`autoLevels` absent → off, `sunGlow` absent → true,
// `exposure` absent → 0, not the fresh-coloring 0.25). Materializing a full
// coloring here would rewrite every one of those on re-share.
const has = (o, k) => Object.prototype.hasOwnProperty.call(o, k);

// Exactly three finite channels in [lo,hi]. Length is forced because every
// uniform writer indexes [0],[1],[2] blind; a bad channel falls back per-channel
// rather than dropping the whole triple.
const rgb3 = (v, def, lo = 0, hi = 1) =>
  [0, 1, 2].map((i) => clampNum(v[i], lo, hi, def[i]));

const LIGHT_DIR_DEF = [0.395, 0.657, 0.643];
// Light scalars: [lo, hi, default]. Bounds are the app's own slider domains
// (index.html), widened where a slider is deliberately conservative rather than
// physical — these are the values the UI can produce, and the wire should not
// be able to produce more.
const LIGHT_SCALARS = {
  ambient: [0, 1, 0.16],
  rim: [0, 1, 0.45],
  gloss: [0, 1, 0],
  intensity: [0, 2.5, 1],
  metallic: [0, 1, 0],
  shadow: [0, 1, 0.5],
  ao: [0, 1, 0.55],
  fill: [0, 1, 0],
  back: [0, 1, 0],
  sky: [0, 1, 0],
  fog: [0, 1, 0],
  glow: [0, 1, 0],
  aperture: [0, 1, 0],
  exposure: [-1, 1, 0],
};
const LIGHT_COLORS = {
  keyColor: [1, 1, 1],
  fillColor: [1, 1, 1],
  backColor: [1, 1, 1],
};
const LIGHT_FLAGS = ["sunGlow", "lightIndicator"];

function sanitizeLight(l) {
  const out = {};
  if (Array.isArray(l.dir)) {
    const d = rgb3(l.dir, LIGHT_DIR_DEF, -1e3, 1e3);
    // normalize(vec3(0)) is undefined in both WGSL and GLSL, and [0,0,0] is
    // exactly what a truncated COLORING section decodes to (ByteReader reads
    // past the end as 0). Fall back to the default direction, not to NaN.
    out.dir = Math.hypot(d[0], d[1], d[2]) < 1e-6 ? LIGHT_DIR_DEF.slice() : d;
  }
  for (const [k, [lo, hi, def]] of Object.entries(LIGHT_SCALARS))
    if (has(l, k)) out[k] = clampNum(l[k], lo, hi, def);
  for (const [k, def] of Object.entries(LIGHT_COLORS))
    if (Array.isArray(l[k])) out[k] = rgb3(l[k], def);
  for (const k of LIGHT_FLAGS) if (has(l, k)) out[k] = !!l[k];
  return out;
}

// A palette stop is only kept if it is whole — `c` missing here is the
// srgbToOklab destructure crash, not a cosmetic defect.
const MAX_STOPS = 8; // app/src/color.ts MAX_STOPS; encoder + consumers agree
const PAL_DEF = { a: [0.5, 0.5, 0.5], b: [0.5, 0.5, 0.5], c: [1, 1, 1], d: [0, 0.33, 0.67] }; // prettier-ignore

function sanitizePalette(p) {
  const out = {};
  if (has(p, "on")) out.on = !!p.on;
  for (const k of ["a", "b", "c", "d"])
    // Cosine palette: `c` is a FREQUENCY and legitimately exceeds 1, so these
    // four clamp loosely (finite + sane) rather than to 0..1 like a color.
    if (Array.isArray(p[k])) out[k] = rgb3(p[k], PAL_DEF[k], -1e3, 1e3);
  if (Array.isArray(p.stops)) {
    const stops = p.stops
      .slice(0, MAX_STOPS)
      .filter((s) => s && typeof s === "object" && Array.isArray(s.c))
      .map((s) => ({ c: rgb3(s.c, [0, 0, 0]), p: clampNum(s.p, 0, 1, 0) }));
    // Under 2 stops is the codec's own "no stops" signal — emit nothing rather
    // than a 1-entry list the samplers would have to special-case.
    if (stops.length >= 2) out.stops = stops;
  }
  if (has(p, "cyclic")) out.cyclic = !!p.cyclic;
  return out;
}

// Validate + coerce an attached coloring. Returns undefined when there is
// nothing usable, so callers can keep "absent" distinct from "default".
export function sanitizeColoring(c) {
  if (!c || typeof c !== "object" || Array.isArray(c)) return undefined;
  const out = {};
  // The one true enum (see COLOR_MODE_MAX) — an out-of-range mode renders as a
  // DIFFERENT mode per tier, so this clamp is a cross-tier parity fix too.
  if (has(c, "mode")) out.mode = clampInt(c.mode, 0, COLOR_MODE_MAX, 0);
  if (has(c, "stripeFreq")) out.stripeFreq = clampInt(c.stripeFreq, 1, 16, 5);
  if (has(c, "autoLevels")) out.autoLevels = !!c.autoLevels;
  if (has(c, "iridescence")) out.iridescence = clampNum(c.iridescence, 0, 1, 0);
  if (has(c, "palettePhase"))
    out.palettePhase = clampNum(c.palettePhase, 0, 1, 0);
  const COL_DEF = { colA: [0.86, 0.46, 0.18], colB: [0.18, 0.62, 0.74], bg: [0.07, 0.09, 0.15] }; // prettier-ignore
  for (const k of ["colA", "colB", "bg"])
    if (Array.isArray(c[k])) out[k] = rgb3(c[k], COL_DEF[k]);
  if (c.palette && typeof c.palette === "object")
    out.palette = sanitizePalette(c.palette);
  if (c.light && typeof c.light === "object")
    out.light = sanitizeLight(c.light);
  return Object.keys(out).length ? out : undefined;
}

// Scene caps + ranges (spec §3.2/§3.3) — single-sourced in limits.js.
export const SCENE_CAPS = {
  MAX_OBJECTS: LIMIT_MAX_OBJECTS,
  MAX_OPS_PER: LIMIT_MAX_OPS_PER_OBJECT,
};

// Validate + coerce a multi-object scene. Builds on the flat sanitizer for the
// base formula fields, then validates each ObjectSpec: op-count cap, object cap,
// transform ranges, combine enum, objType enum, per-object Julia. Per §3.8 (B1)
// passes an attached `coloring` through unchanged (SCENES.md §Coloring).
// Distinct default albedos (sRGB) so a fresh scene's objects read apart instead of
// merging into one blob (§3.8). Cycled by object index; an explicit color wins.
const OBJ_PALETTE = [
  [0.86, 0.46, 0.18], // orange
  [0.3, 0.55, 0.85], // blue
  [0.45, 0.78, 0.4], // green
  [0.82, 0.36, 0.52], // pink
  [0.85, 0.74, 0.3], // gold
  [0.56, 0.45, 0.82], // violet
  [0.36, 0.78, 0.74], // teal
  [0.8, 0.52, 0.34], // clay
];
function sanitizeColor(c, idx) {
  const def = OBJ_PALETTE[idx % OBJ_PALETTE.length];
  return Array.isArray(c) && c.length >= 3
    ? [0, 1, 2].map((i) => clampNum(c[i], 0, 1, def[i]))
    : def.slice();
}

// D0 new-form object (PRIMITIVE_DIFS_D0 §2.4): an explicit `shapeId` means op
// chain + shape leaf. Registry-driven param clamps (no per-type literals), ops
// KEPT alongside the leaf (mixed objects are the new capability). The legacy
// branch below stays byte-for-byte — pinned share fixtures depend on it.
function sanitizeLeafObject(o, oi, caps) {
  // Ids BEYOND the registry are preserved (0-255), not clamped into it: a D2
  // link opened in an older build must degrade at render time (radial
  // fallback) and re-share intact — clamping would silently turn a future
  // leaf into a different shape. Unknown-leaf params clamp to engine bounds.
  const shapeId = clampInt(o.shapeId, 0, 255, 0);
  const leaf = leafById(shapeId);
  const srcOps = Array.isArray(o.ops) ? o.ops.slice(0, caps.MAX_OPS_PER) : [];
  const ops = rejectNumericOps(
    sanitizeOps(srcOps, ` in object #${oi + 1}`),
    ` in object #${oi + 1}`,
  );
  const tIn = o.transform && typeof o.transform === "object" ? o.transform : o;
  const org = Array.isArray(tIn.origin)
    ? tIn.origin
    : Array.isArray(o.origin)
      ? o.origin
      : [0, 0, 0];
  const origin = [0, 1, 2].map((i) => clampNum(org[i], -1e4, 1e4, 0));
  const uscale = clampNum(tIn.uscale ?? o.uscale, 1e-4, 1e4, 1);
  const rawRot = Array.isArray(tIn.rot)
    ? tIn.rot
    : Array.isArray(o.rot)
      ? o.rot
      : [0, 0, 0];
  const rot = (rawRot.length === 4 ? rawRot : rawRot.slice(0, 3)).map((x) =>
    num(x),
  );
  // shapeParams: each declared param clamps to engine-hard bounds keyed off the
  // registry metadata (positive-size params ≥ 1e-4, zero-floor params ≥ 0,
  // signed params ≥ -1e4); undeclared slots are 0.
  const sp = Array.isArray(o.shapeParams)
    ? o.shapeParams
    : [o.primParam, o.primParam2];
  const shapeParams = [0, 1, 2, 3].map((j) => {
    const p = leaf?.params?.[j];
    // Unknown (future-D2) leaf: engine-bound clamps on all 4 slots; known
    // leaf: registry-declared slots clamp per metadata, the rest are 0.
    if (!p) return shapeId > MAX_LEAF_ID ? clampNum(sp?.[j], -1e4, 1e4, 0) : 0;
    const lo = p.min > 0 ? 1e-4 : p.min < 0 ? -1e4 : 0;
    return clampNum(sp?.[j], lo, 1e4, p.def);
  });
  const iterShape = shapeId > 0 && !!o.iterShape;
  const isChain = ops.length > 0;
  const out = {
    shapeId,
    shapeParams,
    ...(iterShape ? { iterShape: true } : {}),
    ops,
    iters: clampInt(o.iters, 1, 24, 8),
    addC: isChain && !!o.addC,
    deOption: isChain ? clampInt(o.deOption, 0, 3, 2) : 2,
    transform: { origin, uscale, rot },
    combine: clampInt(o.combine ?? o.combineType, 0, 3, 0),
    blendK: clampNum(o.blendK, 0, 10, 0),
    looseDE: !!o.looseDE,
    color: sanitizeColor(o.color, oi),
    // Conservative legacy aliases (spec §2.4): only a true legacy-range pure
    // leaf presents its shapeId; evaluate.js-style consumers read these. The
    // boxBase alias marks the one chain+leaf pattern the v1 wire body can
    // express (share.ts encodes it via flags bit7 for old-decoder compat).
    objType:
      shapeId >= 1 && shapeId <= 6 && !isChain && !iterShape && !o.addC
        ? shapeId
        : 0,
    ...(shapeId > 0
      ? { primParam: shapeParams[0], primParam2: shapeParams[1] }
      : {}),
    ...(shapeId === 1 &&
    isChain &&
    !iterShape &&
    shapeParams[1] === 0 &&
    shapeParams[2] === 0 &&
    shapeParams[3] === 0
      ? { boxBase: true }
      : {}),
    ...(typeof o.name === "string" ? { name: o.name.slice(0, 40) } : {}),
    ...(o.muted ? { muted: true } : {}),
  };
  if (isChain && o.julia) {
    out.julia = true;
    out.juliaC = [0, 1, 2].map((i) =>
      num(Array.isArray(o.juliaC) ? o.juliaC[i] : undefined),
    );
  }
  return out;
}

export function sanitizeScene(obj) {
  if (!obj || typeof obj !== "object") throw new Error("not an object");
  if (!Array.isArray(obj.objects)) throw new Error('missing "objects" array');
  // Base formula fields (name/camera/iters/flat ops). Scenes commonly have ops:[].
  const f = sanitizeFlat({
    ...obj,
    ops: Array.isArray(obj.ops) ? obj.ops : [],
  });

  const { MAX_OBJECTS, MAX_OPS_PER } = SCENE_CAPS;
  const objects = obj.objects.slice(0, MAX_OBJECTS).map((o, oi) => {
    if (!o || typeof o !== "object")
      throw new Error(`object #${oi + 1} is not an object`);
    // D0 new form: an explicit shapeId routes to the registry-driven sanitizer
    // (mixed op-chain + leaf objects). Legacy inputs stay on the branch below.
    if (o.shapeId !== undefined && o.shapeId !== null)
      return sanitizeLeafObject(o, oi, { MAX_OPS_PER });
    // 0 IFS · 1 box · 2 sphere · 3 torus · 4 cylinder · 5 capsule · 6 plane.
    const objType = clampInt(o.objType, 0, 6, 0);
    const srcOps = Array.isArray(o.ops) ? o.ops.slice(0, MAX_OPS_PER) : [];
    const ops =
      objType === 0
        ? rejectNumericOps(
            sanitizeOps(srcOps, ` in object #${oi + 1}`),
            ` in object #${oi + 1}`,
          )
        : [];
    // Transform: origin (clamped), uniform scale (>0), Euler XYZ degrees (or quat).
    const tIn =
      o.transform && typeof o.transform === "object" ? o.transform : o;
    const org = Array.isArray(tIn.origin)
      ? tIn.origin
      : Array.isArray(o.origin)
        ? o.origin
        : [0, 0, 0];
    const origin = [0, 1, 2].map((i) => clampNum(org[i], -1e4, 1e4, 0));
    const uscale = clampNum(tIn.uscale ?? o.uscale, 1e-4, 1e4, 1);
    const rawRot = Array.isArray(tIn.rot)
      ? tIn.rot
      : Array.isArray(o.rot)
        ? o.rot
        : [0, 0, 0];
    const rot = (rawRot.length === 4 ? rawRot : rawRot.slice(0, 3)).map((x) =>
      num(x),
    );
    const out = {
      objType,
      ops,
      iters: clampInt(o.iters, 1, 24, 8),
      addC: !!o.addC,
      deOption: objType === 0 ? clampInt(o.deOption, 0, 3, 2) : 0,
      transform: { origin, uscale, rot },
      combine: clampInt(o.combine ?? o.combineType, 0, 3, 0), // 0 union·1 smooth·2 subtract·3 intersect
      blendK: clampNum(o.blendK, 0, 10, 0),
      looseDE: !!o.looseDE,
      color: sanitizeColor(o.color, oi), // per-object albedo (sRGB 0..1, §3.8)
      // Formula Outline PR 0.2/0.4 — user label + object-level mute (skipped at
      // upload via the app's engineView; still serialized, still editable).
      ...(typeof o.name === "string" ? { name: o.name.slice(0, 40) } : {}),
      ...(o.muted ? { muted: true } : {}),
    };
    if (objType === 0 && o.julia) {
      out.julia = true;
      out.juliaC = [0, 1, 2].map((i) =>
        num(Array.isArray(o.juliaC) ? o.juliaC[i] : undefined),
      );
    }
    // Box-DE base (IFS objects only): render flat-faced cubes instead of round
    // dust. Repurposes primParam as the box half-extent (default 1). flags bit11.
    if (objType === 0 && o.boxBase) {
      out.boxBase = true;
      out.primParam = clampNum(o.primParam ?? o.halfExtent, 1e-4, 1e4, 1);
    }
    if (objType > 0) {
      // plane (6) uses primParam as a half-thickness ≥ 0; others a positive size.
      out.primParam = clampNum(
        o.primParam ?? o.halfExtent ?? o.radius,
        objType === 6 ? 0 : 1e-4,
        1e4,
        objType === 6 ? 0 : 1,
      );
      // Second size param for multi-param prims: torus minor r / cyl + capsule h.
      if (objType === 3 || objType === 4 || objType === 5)
        out.primParam2 = clampNum(o.primParam2, 1e-4, 1e4, 0.25);
    }
    return out;
  });
  f.objects = objects;

  // Scene coloring (SCENES.md §Coloring, amends §3.8): Glow/Bands now render
  // on scenes via orbit-free signals (surface angle / radial bands), so the
  // old mode-0 forcing is retired — a shared scene keeps its chosen mode.
  // …but it is no longer passed through RAW (#538 item 3) — see sanitizeColoring.
  const coloring = sanitizeColoring(obj.coloring);
  if (coloring) f.coloring = coloring;
  return f;
}

// Validate + coerce an arbitrary object (pasted JSON / saved library entry) into
// a safe formula. Shared importer for both frontends. Throws on unknown ops.

import { byKey, W_BULB_NUMERIC } from "./operators.js";
import {
  MAX_FLAT_OPS as LIMIT_MAX_FLAT_OPS,
  MAX_OBJECTS as LIMIT_MAX_OBJECTS,
  MAX_OPS_PER_OBJECT as LIMIT_MAX_OPS_PER_OBJECT,
} from "./limits.js";

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

// Cap on flat op-count. Single-sourced in limits.js (matches the WebGPU tier);
// the smaller WebGL2 tier still caps itself lower — see the divergence note there.
const MAX_FLAT_OPS = LIMIT_MAX_FLAT_OPS;

// Empty slate for the New button — a bare formula, no ops.
export const BLANK = {
  name: "Untitled",
  note: "",
  addC: false,
  iters: 8,
  deOption: 2,
  ops: [],
  camera: { yawDeg: 35, pitchDeg: 22, dist: 14, fovDeg: 42 },
};

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
        num(Array.isArray(o.values) ? o.values[pi] : undefined, p.default),
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
    // Base detail. Upper bound is the engine/GLSL iteration cap (64), not the old
    // UI cap of 24 — auto-detail (preview.js §6) and its 2–64 "Detail" slider can
    // now carry the count that high, so a saved/shared high base must survive.
    iters: clampInt(obj.iters, 2, 64, 12),
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
  rejectNumericOps(f.ops, " in hybrid slot A");
  const bOps = rejectNumericOps(
    sanitizeOps(Array.isArray(h.b?.ops) ? h.b.ops : [], " in hybrid slot B"),
    " in hybrid slot B",
  );
  const sched = h.schedule || {};
  let a = clampInt(sched.a, 1, 8, 1);
  let b = clampInt(sched.b, 1, 8, 1);
  if (a + b > 12) {
    if (a >= b) a = 12 - b;
    else b = 12 - a;
  }
  const totalParams =
    f.ops.reduce((n, o) => n + o.values.length, 0) +
    bOps.reduce((n, o) => n + o.values.length, 0);
  if (totalParams > 192)
    throw new Error(`hybrid formula params ${totalParams} > cap 192`);
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

// Scene caps + ranges (spec §3.2/§3.3) — single-sourced in limits.js.
export const SCENE_CAPS = { MAX_OBJECTS: LIMIT_MAX_OBJECTS, MAX_OPS_PER: LIMIT_MAX_OPS_PER_OBJECT };

// Validate + coerce a multi-object scene. Builds on the flat sanitizer for the
// base formula fields, then validates each ObjectSpec: op-count cap, object cap,
// transform ranges, combine enum, objType enum, per-object Julia. Per §3.8 (B1)
// forces surface coloring mode 0 when >1 object (an attached `coloring`, if any).
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
    // 0 IFS · 1 box · 2 sphere · 3 torus · 4 cylinder · 5 capsule · 6 plane.
    const objType = clampInt(o.objType, 0, 6, 0);
    const srcOps = Array.isArray(o.ops) ? o.ops.slice(0, MAX_OPS_PER) : [];
    const ops =
      objType === 0 ? rejectNumericOps(sanitizeOps(srcOps, ` in object #${oi + 1}`), ` in object #${oi + 1}`) : [];
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

  // §3.8 (B1): multi-object scenes are surface-coloring only. If a coloring rides
  // along (e.g. a share decode hands one in), force mode 0 so Glow/Bands → garbage.
  if (objects.length > 1 && obj.coloring && typeof obj.coloring === "object") {
    f.coloring = { ...obj.coloring, mode: 0 };
  } else if (obj.coloring && typeof obj.coloring === "object") {
    f.coloring = obj.coloring;
  }
  return f;
}

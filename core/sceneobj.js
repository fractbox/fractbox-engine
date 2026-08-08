// Canonical CSG scene-object normalization — ONE place for the fallback chains
// every scene consumer used to hand-repeat (renderer.js writeScene, renderer_gl.js
// writeScene, cpu.js makeSceneDE). Each tier packs its own buffers/uniforms/
// closures FROM this canonical shape, so the "3-emitter mirror discipline" for
// object metadata is parity by construction instead of three copies of
// `tr.origin || o.origin`, `primParam ?? halfExtent ?? radius ?? 1`, …
//
// D0 (docs/planning/PRIMITIVE_DIFS_D0.md): every object is op chain + shape
// leaf. Canonical fields the emitters read:
//   shapeId     0 = no leaf (radial r/|w| dust) · 1-6 launch leaves (see
//               leaves.js) · 7+ D2 leaves. 8 bits.
//   shapeParams f32×4 leaf parameter block
//   iterShape   D3: evaluate the leaf INSIDE the fold loop, min across iters
//   ops         the ACTIVE op slice — muted ops dropped
//   iters/addC/deOption/julia/juliaC/looseDE — iteration knobs. For a PURE
//               leaf (shapeId>0, no ops) these are canonicalized to the
//               loop-is-identity form (addC false · julia false · deOption 2 ·
//               iters 1) so the unified loop reproduces the legacy
//               skip-the-loop primitive exactly (old links carry deOption 0
//               there, which must NOT select the escape finalize).
//   combine     0 union · 1 smooth · 2 subtract · 3 intersect (masked to 2 bits)
//   blendK      smooth-combine k
//   origin/uscale/quat — the object transform (quat from Euler XYZ degrees or
//               a length-4 quaternion, via eulerToQuat)
//   color       per-object albedo (sRGB), component defaults applied
//
// Legacy aliases KEPT on the output for incremental consumers (evaluate.js
// reach heuristics, UI summaries): objType / primParam / primParam2 / boxBase.
// The objType alias is CONSERVATIVE (spec §2.4): it equals shapeId only for a
// true legacy pure shape; mixed objects, iterated shapes, and new leaf ids all
// present as objType 0 so shape-specific heuristics never misclassify them.
//
// Legacy INPUT forms accepted (precedence — spec §2.4):
//   1. explicit `shapeId` (new form; ops kept — mixed objects allowed)
//   2. objType 1-6            → shapeId = objType, params from primParam/2
//   3. objType 0 + boxBase    → shapeId = 1 (box) + final mode, ops kept
//
// Zero deps beyond quat.js + leaves.js. sanitize.js remains the upstream gate
// for app flows — these fallbacks only matter for non-sanitized callers.

import { eulerToQuat } from "./quat.js";

// Default per-object albedo (sRGB) — the engine-level fallback when an object
// carries no color (sanitize.js assigns its own per-index palette upstream).
const DEFAULT_COLOR = [0.86, 0.46, 0.18];

const num = (v, d) => (Number.isFinite(Number(v)) ? Number(v) : d);

export function normalizeSceneObject(o) {
  const legacyType = Number(o.objType) & 0xf;

  // Resolve the leaf (input precedence above).
  let shapeId = 0;
  let sp = null;
  if (o.shapeId !== undefined && o.shapeId !== null) {
    shapeId = Math.max(0, Math.min(255, Math.round(num(o.shapeId, 0))));
    sp = Array.isArray(o.shapeParams) ? o.shapeParams : null;
  } else if (legacyType > 0) {
    shapeId = legacyType;
  } else if (o.boxBase) {
    shapeId = 1;
  }
  const p0 = num(sp ? sp[0] : (o.primParam ?? o.halfExtent ?? o.radius), 1);
  const p1 = num(sp ? sp[1] : o.primParam2, 0);
  const shapeParams = [p0, p1, num(sp?.[2], 0), num(sp?.[3], 0)];

  // The op chain. Legacy pure shapes (objType 1-6 without an explicit shapeId)
  // never carried ops; new-form objects keep theirs (mixed objects).
  const legacyPure = o.shapeId === undefined && legacyType > 0;
  const ops = legacyPure ? [] : (o.ops || []).filter((op) => !op.muted);

  const iterShape = shapeId > 0 && !!o.iterShape;
  // Pure leaf ⇒ the loop is the identity (see header). Chain objects keep
  // their knobs (including the 0-op objType-0 + addC oddity, unchanged).
  const pureLeaf = shapeId > 0 && ops.length === 0 && !iterShape;

  const tr = o.transform || {};
  const org = tr.origin || o.origin || [0, 0, 0];
  const jc = o.juliaC || [0, 0, 0];
  const col = o.color || DEFAULT_COLOR;
  const julia = !pureLeaf && !legacyPure && !!o.julia;
  const isChain = ops.length > 0;

  return {
    shapeId,
    shapeParams,
    iterShape,
    ops,
    iters: pureLeaf ? 1 : (o.iters ?? 1),
    addC: pureLeaf ? false : !!o.addC,
    julia,
    juliaC: [jc[0] ?? 0, jc[1] ?? 0, jc[2] ?? 0],
    deOption: pureLeaf ? 2 : legacyPure ? 2 : (o.deOption ?? 2),
    looseDE: !!o.looseDE,
    combine: (o.combine ?? o.combineType ?? 0) & 3,
    blendK: o.blendK ?? 0,
    origin: [org[0] ?? 0, org[1] ?? 0, org[2] ?? 0],
    uscale: tr.uscale ?? o.uscale ?? 1,
    // o.quat passthrough keeps normalize idempotent (a normalized object
    // carries quat, not rot — eulerToQuat takes a length-4 array as a quat).
    quat: eulerToQuat(tr.rot ?? o.rot ?? o.quat ?? [0, 0, 0]),
    color: [col[0] ?? DEFAULT_COLOR[0], col[1] ?? DEFAULT_COLOR[1], col[2] ?? DEFAULT_COLOR[2]],
    // ── legacy aliases (read-only; emitters use the fields above) ────────────
    // Conservative objType: shapeId only for a true legacy pure shape —
    // everything else is 0 so evaluate.js-style heuristics take their existing
    // conservative op-chain path (spec §2.4).
    objType:
      shapeId >= 1 && shapeId <= 6 && !isChain && !iterShape && !o.addC ? shapeId : 0,
    primParam: shapeParams[0],
    primParam2: shapeParams[1],
    // The boxBase pattern (box leaf finalizing a chain, params 1-3 unused) —
    // the codec's §2.5 redundancy rule detects it here.
    boxBase:
      shapeId === 1 &&
      isChain &&
      !iterShape &&
      shapeParams[1] === 0 &&
      shapeParams[2] === 0 &&
      shapeParams[3] === 0,
  };
}

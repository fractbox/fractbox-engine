// Shared formula invariants — the static safety net, GPU-free.
//
// Two entry points, because the checks split cleanly by scope:
//   validateOperators()      — once, over the OPERATORS palette itself
//   validateFormula(formula) — per op-list (preset OR Blockly-generated)
//
// Both return { failures, warnings }. A failure means the formula is malformed
// or won't round-trip into the desktop engine (gate export on it). A warning is
// a soft "this may render blank" hint (e.g. analytic-DE scale < 2) — surface it,
// don't gate. Consumed by check.mjs (CLI/CI), the card app, and the Blockly
// frontend so all three share one validator.

import { OPERATORS, byKey } from "./operators.js";
import { exportGLSL } from "./shader.js";
import { MAX_OP_PARAMS, MAX_OP_PARAMS_INLINE } from "./limits.js";

// Headers FormulaRegistry parses on the desktop side — every export must carry them.
const REQUIRED_HEADERS = [
  "// DEFAULTS:",
  "// PARAM_NAMES:",
  "// AddC:",
  "// DEoption:",
];
const RANGE_ENTRY = /^-?\d*\.?\d+:-?\d*\.?\d+:-?\d*\.?\d+$/; // min:max:step

// ── Palette-level (no formula): opcodes contiguous from 0, each op within the
// MAX_OP_PARAMS budget (params past MAX_OP_PARAMS_INLINE ride the opAux lane).
// These read the global OPERATORS array, so they have no `formula` argument and
// can't live inside validateFormula — run once at load/CI.
// Palette grouping tags the op pickers group by (#87). 'warp' is reserved for
// the Phase-C approximate-DE ops.
const OP_CATEGORIES = ["fold", "sphere", "symmetry", "warp", "move", "power"];

export function validateOperators() {
  const failures = [],
    warnings = [];
  const ids = OPERATORS.map((o) => o.id).sort((a, b) => a - b);
  ids.forEach((id, i) => {
    if (id !== i)
      failures.push(`operator ids not contiguous at ${i} (got ${id})`);
  });
  for (const o of OPERATORS) {
    if (o.params.length > MAX_OP_PARAMS)
      failures.push(
        `operator "${o.key}" has ${o.params.length} params (max ${MAX_OP_PARAMS})`,
      );
    // Arity creep is reviewable, not silent (OP_PARAM_ENCODING.md §5.6): params
    // past MAX_OP_PARAMS_INLINE ride the opAux overflow lane, which costs a
    // storage load per iteration on every op that uses it. Warn — don't fail —
    // so the count stays visible in CI and each widening is a deliberate call.
    else if (o.params.length > MAX_OP_PARAMS_INLINE)
      warnings.push(
        `operator "${o.key}" has ${o.params.length} params — ${o.params.length - MAX_OP_PARAMS_INLINE} ride the opAux overflow lane`,
      );
    if (!OP_CATEGORIES.includes(o.category))
      failures.push(
        `operator "${o.key}" has no picker category (want one of ${OP_CATEGORIES.join("/")})`,
      );
    // Every op must carry a beginner blurb (presentation metadata the picker/card
    // surface). Guarded here so it fires in BOTH the web check.mjs gate and the
    // app's invariants test — a description-less op can't ship (PRIMITIVE_ONBOARDING.md P0).
    if (typeof o.blurb !== "string" || !o.blurb.trim())
      failures.push(
        `operator "${o.key}" has no blurb (one-sentence beginner description)`,
      );
    // deApprox is the approximate-DE tag (APPROX_DE.md §1) — boolean when present.
    if ("deApprox" in o && typeof o.deApprox !== "boolean")
      failures.push(`operator "${o.key}" has a non-boolean deApprox tag`);
  }
  return { failures, warnings };
}

// ── Per-formula: keys resolve, the export is engine-conformant, the DE is sound,
// and Julia is baked correctly. Works on any op-list, not just the presets.
export function validateFormula(formula) {
  const failures = [],
    warnings = [];
  const name = formula.name || "(unnamed)";

  for (const op of formula.ops)
    if (!byKey(op.key))
      failures.push(`formula "${name}": unknown operator "${op.key}"`);

  // D0 scene-object leaf fields (PRIMITIVE_DIFS_D0 §2.4) — type checks only;
  // range/registry clamping is sanitize.js's job.
  for (const [oi, ob] of (formula.objects ?? []).entries()) {
    const at = `formula "${name}" object #${oi + 1}`;
    if (
      "shapeId" in ob &&
      (!Number.isInteger(ob.shapeId) || ob.shapeId < 0 || ob.shapeId > 255)
    )
      failures.push(`${at}: shapeId must be an integer 0-255`);
    if ("shapeParams" in ob) {
      const sp = ob.shapeParams;
      if (
        !Array.isArray(sp) ||
        sp.length > 4 ||
        sp.some((v) => typeof v !== "number" || !Number.isFinite(v))
      )
        failures.push(`${at}: shapeParams must be ≤4 finite numbers`);
    }
    if ("iterShape" in ob && typeof ob.iterShape !== "boolean")
      failures.push(`${at}: iterShape must be a boolean`);
  }

  let glsl = "";
  try {
    glsl = exportGLSL(formula);
  } catch (e) {
    failures.push(`formula "${name}": export threw — ${e.message}`);
    return { failures, warnings };
  }

  const safe = name.replace(/[^A-Za-z0-9_]/g, "_");
  // exportGLSL emits three shapes: a single-leaf formula → a standalone shape-DE
  // function; a multi-object scene → an explanatory stub; everything else → the
  // desktop iterateJIT_ op-list body. Detect which was produced (from the emitted
  // marker) and check the matching contract — the iterateJIT_/header/ranges
  // checks below only apply to the op-list form.
  const isShapeExport = glsl.includes("(shape-DE export)");
  const isSceneExport = /^\/\/ SCENE:/m.test(glsl);

  if (isShapeExport) {
    // Shape (single leaf) → a standalone DE, not iterateJIT_.
    if (!glsl.includes(`float shapeDE_${safe}(vec3 p)`))
      failures.push(
        `formula "${name}": shape export missing shapeDE_${safe}(vec3 p) signature`,
      );
  } else if (isSceneExport) {
    // Scene (multi-object CSG) → an intentional explanatory stub; there is no
    // single body to compile-check. Just confirm it names its object count.
    if (!/\bobjects \(CSG composition\)/.test(glsl))
      failures.push(
        `formula "${name}": scene export missing the CSG-composition stub`,
      );
  } else {
    if (
      !glsl.includes(
        `void iterateJIT_${safe}(int slot, vec3 c, inout vec3 pos, inout float w)`,
      )
    )
      failures.push(
        `formula "${name}": export missing iterateJIT_${safe} signature`,
      );
    for (const h of REQUIRED_HEADERS)
      if (!glsl.includes(h))
        failures.push(`formula "${name}": export missing "${h}" header`);

    // Regression guard (SCALE_VARY.md §6.4): the desktop iterateJIT_ ABI passes no
    // iteration index, so an op whose live `glsl()` body references the loop var `i`
    // (e.g. scaleDrift) must supply a `desktopGlsl` fallback. A bare `i` token that
    // leaks into the emitted body would be an undeclared-variable desktop compile
    // error that nothing else here catches (the signature string check above passes
    // regardless). Scan the code (comments stripped) for a bare `\bi\b`.
    const bodyMatch = glsl.match(
      /void iterateJIT_[A-Za-z0-9_]+\([^)]*\)\s*\{([\s\S]*)\}\s*$/,
    );
    const bodyCode = (bodyMatch ? bodyMatch[1] : "").replace(/\/\/[^\n]*/g, "");
    if (/\bi\b/.test(bodyCode))
      failures.push(
        `formula "${name}": desktop export leaks a bare 'i' iteration index — an op needs a desktopGlsl override (SCALE_VARY.md §6.4)`,
      );

    // PARAM_RANGES (min:max:step per slot) must be present + slot-aligned with
    // PARAM_NAMES — the desktop reads it for authored slider bounds (issue #501).
    const nm = (glsl.match(/^\/\/ PARAM_NAMES: (.*)$/m) || [])[1] || "";
    const rg = (glsl.match(/^\/\/ PARAM_RANGES: (.*)$/m) || [])[1];
    if (rg === undefined)
      failures.push(
        `formula "${name}": export missing "// PARAM_RANGES" header`,
      );
    else {
      const nNames = nm ? nm.split(",").length : 0;
      const nRanges = rg ? rg.split(",").length : 0;
      if (nNames !== nRanges)
        failures.push(
          `formula "${name}": PARAM_RANGES count ${nRanges} != PARAM_NAMES count ${nNames}`,
        );
      for (const r of rg ? rg.split(",") : [])
        if (!RANGE_ENTRY.test(r))
          failures.push(
            `formula "${name}": malformed PARAM_RANGES entry "${r}" (want min:max:step)`,
          );
    }
  }

  // DE soundness (WARN only): analytic IFS DE needs |scale| >= 2 or it goes loose.
  if (formula.deOption === 2)
    for (const op of formula.ops)
      if (op.key === "scale" && Math.abs(op.values[0]) < 2)
        warnings.push(
          `formula "${name}": scale ${op.values[0]} < 2 — analytic DE may render blank`,
        );

  // Julia: a 3-component constant, baked into the body with AddC forced off so
  // the engine doesn't also add the world seed.
  if (formula.julia) {
    if (!Array.isArray(formula.juliaC) || formula.juliaC.length !== 3)
      failures.push(
        `formula "${name}": julia is on but juliaC is not a 3-component array`,
      );
    if (!/\/\/ AddC: false/.test(glsl))
      failures.push(
        `formula "${name}": Julia export must force "// AddC: false"`,
      );
    if (!glsl.includes("Julia constant (baked"))
      failures.push(
        `formula "${name}": Julia export missing the baked constant add`,
      );
  }

  return { failures, warnings };
}

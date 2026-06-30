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

import { OPERATORS, byKey } from './operators.js';
import { exportGLSL } from './shader.js';

// Headers FormulaRegistry parses on the desktop side — every export must carry them.
const REQUIRED_HEADERS = ['// DEFAULTS:', '// PARAM_NAMES:', '// AddC:', '// DEoption:'];
const RANGE_ENTRY = /^-?\d*\.?\d+:-?\d*\.?\d+:-?\d*\.?\d+$/;   // min:max:step

// ── Palette-level (no formula): opcodes contiguous from 0, each op <= 3 params.
// These read the global OPERATORS array, so they have no `formula` argument and
// can't live inside validateFormula — run once at load/CI.
export function validateOperators() {
  const failures = [], warnings = [];
  const ids = OPERATORS.map(o => o.id).sort((a, b) => a - b);
  ids.forEach((id, i) => { if (id !== i) failures.push(`operator ids not contiguous at ${i} (got ${id})`); });
  for (const o of OPERATORS)
    if (o.params.length > 3) failures.push(`operator "${o.key}" has ${o.params.length} params (max 3)`);
  return { failures, warnings };
}

// ── Per-formula: keys resolve, the export is engine-conformant, the DE is sound,
// and Julia is baked correctly. Works on any op-list, not just the presets.
export function validateFormula(formula) {
  const failures = [], warnings = [];
  const name = formula.name || '(unnamed)';

  for (const op of formula.ops)
    if (!byKey(op.key)) failures.push(`formula "${name}": unknown operator "${op.key}"`);

  let glsl = '';
  try { glsl = exportGLSL(formula); }
  catch (e) { failures.push(`formula "${name}": export threw — ${e.message}`); return { failures, warnings }; }

  const safe = name.replace(/[^A-Za-z0-9_]/g, '_');
  if (!glsl.includes(`void iterateJIT_${safe}(int slot, vec3 c, inout vec3 pos, inout float w)`))
    failures.push(`formula "${name}": export missing iterateJIT_${safe} signature`);
  for (const h of REQUIRED_HEADERS)
    if (!glsl.includes(h)) failures.push(`formula "${name}": export missing "${h}" header`);

  // PARAM_RANGES (min:max:step per slot) must be present + slot-aligned with
  // PARAM_NAMES — the desktop reads it for authored slider bounds (issue #501).
  const nm = (glsl.match(/^\/\/ PARAM_NAMES: (.*)$/m) || [])[1] || '';
  const rg = (glsl.match(/^\/\/ PARAM_RANGES: (.*)$/m) || [])[1];
  if (rg === undefined) failures.push(`formula "${name}": export missing "// PARAM_RANGES" header`);
  else {
    const nNames = nm ? nm.split(',').length : 0;
    const nRanges = rg ? rg.split(',').length : 0;
    if (nNames !== nRanges)
      failures.push(`formula "${name}": PARAM_RANGES count ${nRanges} != PARAM_NAMES count ${nNames}`);
    for (const r of (rg ? rg.split(',') : []))
      if (!RANGE_ENTRY.test(r))
        failures.push(`formula "${name}": malformed PARAM_RANGES entry "${r}" (want min:max:step)`);
  }

  // DE soundness (WARN only): analytic IFS DE needs |scale| >= 2 or it goes loose.
  if (formula.deOption === 2)
    for (const op of formula.ops)
      if (op.key === 'scale' && Math.abs(op.values[0]) < 2)
        warnings.push(`formula "${name}": scale ${op.values[0]} < 2 — analytic DE may render blank`);

  // Julia: a 3-component constant, baked into the body with AddC forced off so
  // the engine doesn't also add the world seed.
  if (formula.julia) {
    if (!Array.isArray(formula.juliaC) || formula.juliaC.length !== 3)
      failures.push(`formula "${name}": julia is on but juliaC is not a 3-component array`);
    if (!/\/\/ AddC: false/.test(glsl))
      failures.push(`formula "${name}": Julia export must force "// AddC: false"`);
    if (!glsl.includes('Julia constant (baked'))
      failures.push(`formula "${name}": Julia export missing the baked constant add`);
  }

  return { failures, warnings };
}

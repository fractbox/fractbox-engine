// Import a formula from text — auto-detects a web-exported .glsl vs an op-list
// JSON. Shared by both frontends.
//
// A web-exported .glsl embeds everything needed to rebuild the op-list:
//   // DEFAULTS: <values,in,order>
//   // Composed from N primitive(s):
//   //   key1 → key2 → key3  (+c)  (Julia c = a, b, c)
// We read the op keys + slice DEFAULTS per each operator's param count. NOTE:
// the .glsl headers don't carry iterations / camera / colours (the desktop
// engine owns those), so those fall back to defaults — only the formula recipe
// round-trips. Decompiled / hand-written native .glsl has no "Composed from"
// line and can't be mapped to the operator set → a clear error.

import { OPERATORS } from './operators.js';
import { sanitizeFormula } from './sanitize.js';

const byKey = (k) => OPERATORS.find((o) => o.key === k);

export function looksLikeGlsl(text) {
  const t = text.trimStart();
  if (t.startsWith('{') || t.startsWith('[')) return false;          // JSON
  return /iterateJIT_|\/\/\s*Composed from|\/\/\s*DEFAULTS:/.test(text);
}

export function glslToFormula(text) {
  const m = text.match(/\/\/\s*Composed from \d+ primitive\(s\):\s*\r?\n\s*\/\/\s+(.+)/);
  if (!m) throw new Error('not a web-exported .glsl (no "Composed from" line). Decompiled/native .glsl can’t be edited here.');

  let line = m[1];
  const jm = /\(Julia c = ([^)]+)\)/.exec(line);
  const hasPlusC = /\(\+c\)/.test(line);
  line = line.replace(/\s*\(\+c\)/, '').replace(/\s*\(Julia c = [^)]+\)/, '').trim();

  const keys = line.split('→').map((s) => s.trim()).filter(Boolean);
  if (!keys.length) throw new Error('no operators found in the "Composed from" line');
  for (const k of keys) if (!byKey(k)) throw new Error(`unknown operator "${k}" — not a web-exported formula`);

  const defs = (text.match(/\/\/\s*DEFAULTS:\s*(.*)/) || [])[1] || '';
  const vals = defs.split(',').map((s) => parseFloat(s.trim())).filter((v) => Number.isFinite(v));

  const ops = [];
  let i = 0;
  for (const k of keys) {
    const n = byKey(k).params.length;
    ops.push({ key: k, values: vals.slice(i, i + n) });
    i += n;
  }
  if (i !== vals.length)
    throw new Error(`DEFAULTS has ${vals.length} values but the operators need ${i}`);

  const deOpt = (text.match(/\/\/\s*DEoption:\s*(\d+)/) || [])[1];
  const name = (text.match(/\/\/\s*JIT formula:\s*(.+?)\s*\(/) || [])[1];

  const f = {
    name: (name || 'Imported').trim().slice(0, 60),
    note: 'imported from .glsl',
    addC: hasPlusC,
    deOption: deOpt !== undefined ? parseInt(deOpt, 10) : 2,
    ops,
  };
  if (jm) { f.julia = true; f.juliaC = jm[1].split(',').map((s) => parseFloat(s.trim())).slice(0, 3); }
  return sanitizeFormula(f);   // fills iters/camera defaults, validates ops
}

// Dispatch: glsl → reconstruct; otherwise parse as op-list JSON.
export function importFormula(text) {
  const t = (text || '').trim();
  if (!t) throw new Error('nothing to import');
  if (looksLikeGlsl(t)) return glslToFormula(t);
  return sanitizeFormula(JSON.parse(t));
}

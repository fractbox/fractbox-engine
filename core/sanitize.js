// Validate + coerce an arbitrary object (pasted JSON / saved library entry) into
// a safe formula. Shared importer for both frontends. Throws on unknown ops.

import { OPERATORS } from './operators.js';

const num = (v, d = 0) => { const n = Number(v); return Number.isFinite(n) ? n : d; };
const clampInt = (v, lo, hi, d) => { const n = Math.round(Number(v)); return Number.isFinite(n) ? Math.max(lo, Math.min(hi, n)) : d; };
const byKey = (k) => OPERATORS.find((o) => o.key === k);

// Empty slate for the New button — a bare formula, no ops.
export const BLANK = {
  name: 'Untitled', note: '', addC: false, iters: 8, deOption: 2, ops: [],
  camera: { yawDeg: 35, pitchDeg: 22, dist: 14, fovDeg: 42 },
};

export function sanitizeFormula(obj) {
  if (!obj || typeof obj !== 'object') throw new Error('not an object');
  if (!Array.isArray(obj.ops)) throw new Error('missing "ops" array');
  const ops = obj.ops.map((o, i) => {
    const def = byKey(o && o.key);
    if (!def) throw new Error(`unknown operator "${o && o.key}" at op #${i + 1}`);
    return { key: def.key, values: def.params.map((p, pi) => num(Array.isArray(o.values) ? o.values[pi] : undefined, p.default)) };
  });
  const f = {
    name: typeof obj.name === 'string' ? obj.name.slice(0, 60) : 'Imported',
    note: typeof obj.note === 'string' ? obj.note.slice(0, 120) : '',
    addC: !!obj.addC,
    iters: clampInt(obj.iters, 2, 24, 12),
    deOption: obj.deOption ?? 2,
    ops,
    camera: (obj.camera && typeof obj.camera === 'object') ? obj.camera : BLANK.camera,
  };
  if (obj.julia) { f.julia = true; f.juliaC = Array.isArray(obj.juliaC) ? obj.juliaC.slice(0, 3).map((x) => num(x)) : [0, 0, 0]; }
  return f;
}

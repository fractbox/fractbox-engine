// Export / share / interop. The op-list IR is the interchange format; these
// match across both frontends and the desktop engine. Shared by both apps.

import { exportGLSL } from './shader.js';

export const safeName = (f) => (f.name || 'formula').replace(/[^A-Za-z0-9_]/g, '_');

// The shape the card app serializes (no julia/juliaC unless set).
export function stripForExport(f) {
  const o = {
    name: f.name, note: f.note, addC: f.addC, iters: f.iters, deOption: f.deOption,
    // Preserve per-op `muted` — dropping it re-activated muted ops on the next
    // import (JSON export / share-hash round-trip), diverging from exportGLSL.
    ops: f.ops.map((op) => ({ key: op.key, values: op.values, ...(op.muted ? { muted: true } : {}) })),
    camera: f.camera,
  };
  if (f.julia) { o.julia = true; o.juliaC = (f.juliaC || [0, 0, 0]).slice(0, 3); }
  // Scenes/hybrids carry extra structure — pass it through so a JSON export or
  // share hash of one round-trips instead of silently collapsing to a flat formula.
  if (f.objects) o.objects = f.objects;
  if (f.hybrid) o.hybrid = f.hybrid;
  return o;
}

export const opListJSON = (f) => JSON.stringify(stripForExport(f), null, 2);
export const glslFor = (f) => exportGLSL(f);

export function b64urlEncode(str) {
  let bin = '';
  for (const b of new TextEncoder().encode(str)) bin += String.fromCharCode(b);
  return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}
export function b64urlDecode(b64) {
  const bin = atob(b64.replace(/-/g, '+').replace(/_/g, '/'));
  return new TextDecoder().decode(Uint8Array.from(bin, (c) => c.charCodeAt(0)));
}

export const shareHash = (f) => '#f=' + b64urlEncode(JSON.stringify(stripForExport(f)));

// Pull the raw `f=` payload out of a URL hash (or `null` if absent). Shared by
// the initial-load reader and the hashchange listener (#75) so both agree on
// what counts as "the same share link" — a plain string compare of this value
// is what lets the hashchange handler tell a genuinely new #f= apart from an
// unrelated hash edit (e.g. a `#loop` flag toggle) or a no-op re-fire.
const SHARE_HASH_RE = /[#&]f=([^&]+)/;
export function shareHashMatch(hash) {
  const m = (hash || '').match(SHARE_HASH_RE);
  return m ? m[1] : null;
}

export function downloadText(filename, text, mime = 'text/plain') {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type: mime }));
  a.download = filename;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

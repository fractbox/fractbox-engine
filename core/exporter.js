// Export / share / interop. The op-list IR is the interchange format; these
// match across both frontends and the desktop engine. Shared by both apps.

import { exportGLSL } from './shader.js';

export const safeName = (f) => (f.name || 'formula').replace(/[^A-Za-z0-9_]/g, '_');

// The shape the card app serializes (no julia/juliaC unless set).
export function stripForExport(f) {
  const o = {
    name: f.name, note: f.note, addC: f.addC, iters: f.iters, deOption: f.deOption,
    ops: f.ops.map((op) => ({ key: op.key, values: op.values })), camera: f.camera,
  };
  if (f.julia) { o.julia = true; o.juliaC = (f.juliaC || [0, 0, 0]).slice(0, 3); }
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

export function downloadText(filename, text, mime = 'text/plain') {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type: mime }));
  a.download = filename;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

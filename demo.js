// Fractbox Engine — standalone demo.
//
// This file is the whole demo. It imports the engine straight from ./core/ as
// raw ES modules (no bundler, no build step) and drives it through the
// high-level preview controller: pick a preset, hand it to the engine, let it
// auto-rotate. Everything visual here is the engine; the demo is just glue.

import { createPreview } from './core/preview.js';
import { PRESETS, clone } from './core/oplist.js';
import { initTour } from './tour.js';

const canvas = document.getElementById('view');
const presetBar = document.getElementById('presets');
const fpsEl = document.getElementById('fps');
const tourBtn = document.getElementById('tour-btn');

let lastFpsAt = 0;
const preview = await createPreview(canvas, {
  camera: PRESETS[0].camera,
  onFrame(ms) {
    // Light-touch FPS readout — throttled so it doesn't thrash the DOM.
    const now = performance.now();
    if (now - lastFpsAt > 500) {
      lastFpsAt = now;
      fpsEl.textContent = ms > 0 ? `${Math.round(1000 / ms)} fps` : '';
    }
  },
});

if (!preview.hasGPU) {
  // Self-diagnose WHY WebGPU is unavailable and show it on-page, so a visitor
  // (or the maintainer) doesn't have to open devtools to find out.
  diagnoseWebGPU().then((reason) => {
    const box = document.getElementById('nogpu');
    const why = document.createElement('p');
    why.className = 'why';
    why.textContent = reason;
    box.appendChild(why);
    box.hidden = false;
  });
} else {
  // Show the first preset immediately, auto-rotating, so the page is alive on load.
  let active = null;
  function show(preset) {
    active = preset;
    preview.frameTo(preset.camera);
    preview.setFormula(clone(preset));
    // renderThumbnails builds each button with a `.lbl` holding the preset name;
    // mark the matching one current (purely a visual highlight).
    for (const b of presetBar.children) {
      const name = b.querySelector('.lbl')?.textContent;
      b.setAttribute('aria-current', name === preset.name ? 'true' : 'false');
    }
  }

  // Build the clickable preset strip. We render each preset to a thumbnail via
  // the engine's offscreen path so the gallery itself is engine output.
  preview.renderThumbnails(PRESETS, presetBar, show);
  // renderThumbnails wires the click handlers (onPick → show); mark the first.
  show(PRESETS[0]);
  preview.setAutoRotate(true);

  // Space toggles the spin; handy when you want to inspect a still.
  let spinning = true;
  addEventListener('keydown', (e) => {
    if (e.code === 'Space') {
      e.preventDefault();
      spinning = !spinning;
      preview.setAutoRotate(spinning);
    } else if (e.key >= '1' && e.key <= '9') {
      const i = Number(e.key) - 1;
      if (PRESETS[i]) show(PRESETS[i]);
    }
  });

  // Guided tour: builds a fractal step by step, explaining each move. Hands the
  // engine a complete formula per step via the same path `show` uses.
  initTour(
    tourBtn,
    (formula) => {
      preview.frameTo(formula.camera);
      preview.setFormula(clone(formula));
      for (const b of presetBar.children) b.setAttribute('aria-current', 'false');
    },
    () => show(active || PRESETS[0]), // on finish/skip, settle on a preset
  );
  tourBtn.hidden = false;

  // Keep `active` referenced (lint) and expose for console tinkering.
  globalThis.__fractbox = { preview, show, get active() { return active; } };
}

// Work out the precise reason WebGPU couldn't start, in plain language. Mirrors
// the steps the engine's renderer takes (navigator.gpu → adapter → device) so
// the message points at the actual failing stage.
async function diagnoseWebGPU() {
  if (!window.isSecureContext) {
    return 'Reason: not a secure context. WebGPU needs HTTPS or http://localhost.';
  }
  if (!('gpu' in navigator)) {
    return (
      'Reason: this browser has no WebGPU (navigator.gpu is undefined). ' +
      'Firefox and older Safari don’t enable it by default; try recent Chrome or Edge.'
    );
  }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      return (
        'Reason: WebGPU is present but no GPU adapter was returned — typically a ' +
        'headless/remote session, a blocklisted or software GPU, or a Linux setup ' +
        'without the Vulkan backend enabled.'
      );
    }
    await adapter.requestDevice(); // if this throws, fall through to the catch
    return (
      'Reason: a GPU adapter exists but the engine’s renderer still failed to ' +
      'start — see the browser console for the WebGPU/WGSL error.'
    );
  } catch (e) {
    return `Reason: requesting a GPU device failed — ${e?.message || e}`;
  }
}

// Fractbox Engine — standalone demo.
//
// This file is the whole demo. It imports the engine straight from ./core/ as
// raw ES modules (no bundler, no build step) and drives it through the
// high-level preview controller: pick a preset, hand it to the engine, let it
// auto-rotate. Everything visual here is the engine; the demo is just glue.

import { createPreview } from './core/preview.js';
import { fillThumbnailGrid } from './previewDom.js';
import { PRESETS, clone } from './core/oplist.js';
import { renderAsciiColored } from './core/cpu.js';
import { makeCamera } from './core/camera.js';
import { defaultColoring } from './core/coloring.js';
import { initTour } from './tour.js';

const canvas = document.getElementById('view');
const presetBar = document.getElementById('presets');
const fpsEl = document.getElementById('fps');
const tourBtn = document.getElementById('tour-btn');

// ?ascii=1 forces the CPU/ASCII path even on a GPU machine — the engine's
// text-mode backend is a feature, not just a fallback, and this is its demo.
const FORCE_ASCII = new URLSearchParams(location.search).has('ascii');

let lastFpsAt = 0;
const preview = FORCE_ASCII
  ? null
  : await createPreview(canvas, {
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

if (!preview || !preview.hasGPU) {
  // No WebGPU and no WebGL2 (or ?ascii=1): the engine still renders — its CPU
  // backend (core/cpu.js) traces the same distance-estimated formulas into
  // colored text. Run the whole demo in ASCII instead of dead-ending.
  startAsciiDemo();
  if (!FORCE_ASCII) {
    // Still self-diagnose WHY the GPU tiers are unavailable, as a compact note
    // over the ASCII render, so nobody has to open devtools to find out.
    diagnoseWebGPU().then((reason) => {
      const box = document.getElementById('nogpu');
      const why = document.createElement('p');
      why.className = 'why';
      why.textContent = reason;
      box.appendChild(why);
      box.hidden = false;
    });
  }
} else {
  // Show the first preset immediately, auto-rotating, so the page is alive on load.
  let active = null;
  function show(preset) {
    active = preset;
    preview.frameTo(preset.camera);
    preview.setFormula(clone(preset));
    // fillThumbnailGrid builds each button with a `.lbl` holding the preset name;
    // mark the matching one current (purely a visual highlight).
    for (const b of presetBar.children) {
      const name = b.querySelector('.lbl')?.textContent;
      b.setAttribute('aria-current', name === preset.name ? 'true' : 'false');
    }
  }

  // Build the clickable preset strip. We render each preset to a thumbnail via
  // the engine's offscreen path so the gallery itself is engine output.
  fillThumbnailGrid(preview, PRESETS, presetBar, show);
  // fillThumbnailGrid wires the click handlers (onPick → show); mark the first.
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

// The text-mode demo: the same presets, live, as colored ASCII. Everything
// visible is engine output — renderAsciiColored (core/cpu.js) traces the
// distance-estimated formula on the CPU and emits color-run <span>s; this
// function only owns the spin clock, the drag, and the preset chips.
function startAsciiDemo() {
  document.body.classList.add('ascii');
  canvas.hidden = true;
  const pre = document.getElementById('asciiview');
  const frame = document.getElementById('asciiframe'); // see index.html: keeps spans inline
  pre.hidden = false;

  // Size the character grid to the viewport. Terminal glyphs aren't square, so
  // measure the real advance width of one monospace char and pass the true
  // pixel aspect to the tracer — otherwise the fractal renders squashed.
  const fontPx = 13;
  pre.style.fontSize = `${fontPx}px`;
  const probe = document.createElement('span');
  probe.textContent = 'M'.repeat(100);
  probe.style.cssText = 'position:absolute;visibility:hidden;white-space:pre';
  pre.appendChild(probe);
  const charW = probe.getBoundingClientRect().width / 100 || fontPx * 0.6;
  probe.remove();
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  let cols, rows, aspect;
  function fit() {
    cols = clamp(Math.floor(innerWidth / charW) - 2, 48, 150);
    rows = clamp(Math.floor((innerHeight * 0.86) / fontPx) - 2, 24, 60);
    aspect = (cols * charW) / (rows * fontPx); // line-height is 1 → charH = fontPx
  }
  fit();
  addEventListener('resize', fit);

  const coloring = defaultColoring();
  let active = PRESETS[0];
  let cam = makeCamera(active.camera);

  // Preset chips — text buttons instead of the GPU-thumbnail grid (thumbnails
  // need an offscreen GPU render; names do the job in text mode).
  function show(preset) {
    active = preset;
    cam = makeCamera(preset.camera);
    for (const b of presetBar.children)
      b.setAttribute('aria-current', b.textContent === preset.name ? 'true' : 'false');
  }
  for (const p of PRESETS) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'ascii-chip';
    b.textContent = p.name;
    b.addEventListener('click', () => show(p));
    presetBar.appendChild(b);
  }
  show(PRESETS[0]);

  // Drag to orbit (Space toggles the spin, digits jump presets — same keys as
  // the GPU demo). While dragging, the auto-spin yields to the pointer.
  let spinning = true;
  let dragging = false;
  let lastX = 0;
  let lastY = 0;
  pre.style.touchAction = 'none';
  pre.addEventListener('pointerdown', (e) => {
    dragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
    pre.setPointerCapture(e.pointerId);
  });
  const D2R = Math.PI / 180;
  pre.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    // makeCamera's fields are RADIANS (yaw/pitch, not the spec's yawDeg).
    cam.yaw -= (e.clientX - lastX) * 0.45 * D2R;
    cam.pitch = clamp(cam.pitch + (e.clientY - lastY) * 0.3 * D2R, -85 * D2R, 85 * D2R);
    lastX = e.clientX;
    lastY = e.clientY;
  });
  pre.addEventListener('pointerup', () => (dragging = false));
  addEventListener('keydown', (e) => {
    if (e.code === 'Space') {
      e.preventDefault();
      spinning = !spinning;
    } else if (e.key >= '1' && e.key <= '9') {
      const i = Number(e.key) - 1;
      if (PRESETS[i]) show(PRESETS[i]);
    }
  });

  // ~11 fps is plenty for text and keeps the CPU cool; the spin advances by
  // wall-clock so the turn rate is frame-rate independent.
  const FRAME_MS = 90;
  let last = 0;
  function tick(now) {
    if (now - last >= FRAME_MS) {
      if (spinning && !dragging) cam.yaw += (now - last) * 0.02 * D2R;
      last = now;
      const t0 = performance.now();
      frame.innerHTML = renderAsciiColored(active, { cols, rows, cam, aspect, coloring }).html;
      const ms = performance.now() - t0;
      if (now - lastFpsAt > 500) {
        lastFpsAt = now;
        fpsEl.textContent = `ascii · ${cols}×${rows} · ${ms.toFixed(0)} ms/frame`;
      }
    }
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);

  globalThis.__fractbox = { show, get active() { return active; } };
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

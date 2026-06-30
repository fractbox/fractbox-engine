// Shared WebGPU preview controller — the whole render engine both frontends
// use, so they differ ONLY in their formula editor. Owns the renderer, camera,
// coloring, the busy-gated pump with interactive quality tiers, canvas
// orbit/zoom/pinch gestures, auto-rotate, PNG export, and preset thumbnails.
//
// The app feeds it a formula (setFormula) + coloring (setColoring); the editor
// is entirely the app's business. DOM-side chrome (badge text, hints) is left
// to the app via the onFrame / onFrameStart callbacks.

import { createRenderer } from "./renderer.js";
import { createRendererGL } from "./renderer_gl.js";
import { makeCamera } from "./camera.js";
import { isEscapeTime, effectiveDeOption, activeOps } from "./operators.js";
import { looseDE } from "./stability.js";
import { defaultColoring } from "./coloring.js";

// Escape-time power maps need a small bailout (r=8) or rᵖᵒʷᵉʳ overflows fp32;
// IFS folds stay bounded so a huge radius is harmless.
const bailoutFor = (f) => (isEscapeTime(f) ? 64.0 : 1.0e6);
const D2R = Math.PI / 180;

export async function createPreview(canvas, opts = {}) {
  const isTouch = navigator.maxTouchPoints > 0;
  const DPR_CAP = isTouch ? 1.0 : 2.0;
  const onFrame = opts.onFrame || (() => {}); // (ms) after each frame
  const onFrameStart = opts.onFrameStart || (() => {});

  let renderer = null,
    hasGPU = false,
    backend = "none";
  let formula = null;
  let coloring = defaultColoring();
  const cam = makeCamera(
    opts.camera || { yawDeg: 35, pitchDeg: 22, dist: 24, fovDeg: 42 },
  );
  let needsDraw = false,
    busy = false,
    scheduled = false;
  let quality = "full",
    autoRotate = false,
    settleTimer = null,
    spinSpeed = 0.7,
    spinTilt = 0; // 0° = turntable (spin around +Z) … 90° = tumble (around +X)
  // Spin axis from the tilt angle: tilts from vertical (+Z) toward +X.
  const spinAxis = () => [
    Math.sin((spinTilt * Math.PI) / 180),
    0,
    Math.cos((spinTilt * Math.PI) / 180),
  ];

  // Renderer fallback ladder: WebGPU → WebGL2 → (none → app shows the ASCII view).
  // opts.backend forces a tier for testing: 'webgl2' skips WebGPU; 'ascii'/'none'
  // skips both (so the app falls to the ASCII view even with a GPU present).
  const force = opts.backend;
  try {
    if (force === "webgl2" || force === "ascii" || force === "none")
      throw new Error("forced " + force);
    renderer = await createRenderer(canvas);
    hasGPU = true;
    backend = "webgpu";
  } catch (e) {
    if (force !== "ascii" && force !== "none") {
      try {
        renderer = await createRendererGL(canvas);
        hasGPU = true;
        backend = "webgl2";
      } catch (e2) {
        console.error("WebGL2 unavailable:", e2?.message || e2);
        hasGPU = false;
      }
    }
    if (!hasGPU && !force) console.warn("WebGPU unavailable:", e?.message || e);
  }

  function scheduleDraw() {
    if (!hasGPU) return;
    needsDraw = true;
    if (!scheduled && !busy) {
      scheduled = true;
      requestAnimationFrame(pump);
    }
  }
  // Coarse quality while interacting, settle to full once the user pauses.
  function bumpInteract() {
    quality = "low";
    clearTimeout(settleTimer);
    settleTimer = setTimeout(() => {
      quality = "full";
      scheduleDraw();
    }, 200);
    scheduleDraw();
  }
  function qualityParams(f) {
    // Interactive (while orbiting/zooming): coarse + cheap so the drag stays
    // smooth (it's expected to be a bit rough).
    if (quality === "low")
      return {
        scale: isTouch ? 0.6 : 0.7,
        steps: 48,
        eps: 0.003,
        deScale: 0.65,
      };
    // Settled (full): extra march steps + a tight step size (deScale 0.5) so the
    // marcher doesn't overstep THIN SURFACES (Amazing Surf etc.) that drop out at
    // grazing angles. Only runs once you stop moving, so the cost is fine.
    //
    // A LOOSE analytic DE (scale < 2, see stability.looseDE) over-estimates
    // distance, so deScale 0.5 still oversteps its thin surface — the ray flies
    // past into the background and the static pass renders blank, while the
    // coarse pass (looser hit eps) catches it: the "renders only when moving"
    // bug (#14). March those with a tighter step (0.3) + extra steps to keep up
    // so the surface resolves statically too. Only loose-DE formulas pay the cost.
    if (f && looseDE(f))
      return {
        scale: 1.0,
        steps: DPR_CAP < 2 ? 220 : 320,
        eps: 0.001,
        deScale: 0.3,
      };
    return {
      scale: 1.0,
      steps: DPR_CAP < 2 ? 140 : 200,
      eps: 0.001,
      deScale: 0.5,
    };
  }
  function sizeCanvas(scale) {
    const r = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, DPR_CAP) * scale;
    const w = Math.max(1, Math.floor(r.width * dpr)),
      h = Math.max(1, Math.floor(r.height * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
  }
  function writeFrame(f, q, res, c) {
    const oc = renderer.writeOps(activeOps(f));
    renderer.writeGlobals({
      res,
      cam: c || cam,
      iters: f.iters,
      opCount: oc,
      addC: f.addC,
      maxSteps: q.steps,
      bailout: bailoutFor(f),
      eps: q.eps,
      deScale: q.deScale ?? 0.85,
      colA: coloring.colA,
      colB: coloring.colB,
      bg: coloring.bg,
      colorMode: coloring.mode,
      deOption: effectiveDeOption(f),
      julia: f.julia,
      juliaC: f.juliaC,
      palette: coloring.palette,
      light: coloring.light,
    });
  }

  async function pump() {
    scheduled = false;
    if (autoRotate) {
      cam.spinAround(spinAxis(), spinSpeed);
      needsDraw = true;
    }
    if (!needsDraw || busy || !formula) return;
    needsDraw = false;
    busy = true;
    onFrameStart();
    const q = qualityParams(formula);
    sizeCanvas(q.scale);
    writeFrame(formula, q, [canvas.width, canvas.height]);
    const t0 = performance.now();
    try {
      renderer.draw();
      await renderer.device.queue.onSubmittedWorkDone();
    } catch (e) {
      console.error("draw:", e);
    }
    busy = false;
    onFrame(performance.now() - t0);
    if (needsDraw || autoRotate) {
      scheduled = true;
      requestAnimationFrame(pump);
    }
  }

  // ── Orbit / zoom / pinch gestures ─────────────────────────────────────────
  const ptrs = new Map();
  const pinchDist = () => {
    const [a, b] = [...ptrs.values()];
    return Math.hypot(a.x - b.x, a.y - b.y);
  };
  canvas.addEventListener("pointerdown", (e) => {
    ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY });
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointerup", (e) => {
    ptrs.delete(e.pointerId);
    canvas.releasePointerCapture?.(e.pointerId);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!ptrs.has(e.pointerId)) return;
    if (ptrs.size >= 2) {
      const oldD = pinchDist();
      ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY });
      const nd = pinchDist();
      if (oldD > 0 && nd > 0) cam.zoom(oldD / nd);
      bumpInteract();
      return;
    }
    const p = ptrs.get(e.pointerId);
    // Orbit speed scales with zoom (dist): finer when zoomed in, never sluggish.
    const s = 0.4 * Math.max(0.4, Math.min(1.3, cam.dist / 18));
    cam.orbit((e.clientX - p.x) * s, (e.clientY - p.y) * s);
    ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY });
    bumpInteract();
  });
  canvas.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      cam.zoom(e.deltaY > 0 ? 1.1 : 0.9);
      bumpInteract();
    },
    { passive: false },
  );
  window.addEventListener("resize", scheduleDraw);

  function frameTo(c) {
    if (!c) return;
    cam.yaw = (c.yawDeg ?? 35) * D2R;
    cam.pitch = (c.pitchDeg ?? 22) * D2R;
    cam.dist = c.dist ?? 24;
    cam.fov = (c.fovDeg ?? 42) * D2R;
  }
  const camObj = () => ({
    yawDeg: cam.yaw / D2R,
    pitchDeg: cam.pitch / D2R,
    dist: cam.dist,
    fovDeg: cam.fov / D2R,
  });

  async function exportPNG(filename) {
    if (!hasGPU || !formula) return false;
    const rect = canvas.getBoundingClientRect();
    const H = 900,
      W = Math.round(H * ((rect.width || 4) / (rect.height || 3)));
    const prevW = canvas.width,
      prevH = canvas.height;
    busy = true;
    try {
      canvas.width = W;
      canvas.height = H;
      writeFrame(formula, { steps: 220, eps: 0.0006, deScale: 0.5 }, [W, H]);
      renderer.draw();
      await renderer.device.queue.onSubmittedWorkDone();
      const blob = await new Promise((res) => canvas.toBlob(res, "image/png"));
      if (!blob) throw new Error("canvas capture returned null");
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = filename;
      a.click();
      URL.revokeObjectURL(a.href);
      return true;
    } finally {
      canvas.width = prevW;
      canvas.height = prevH;
      busy = false;
      scheduleDraw();
    }
  }

  // Thumbnail dataURL cache, keyed by what actually drives the render (ops +
  // camera + the escape/IFS knobs). Lets the SAME formula be tiled in more than
  // one place — e.g. a filtered strip AND an expand-to-grid sheet, or the same
  // preset across category switches — while paying the GPU cost only once.
  const thumbCache = new Map();
  const thumbKey = (p) =>
    JSON.stringify([
      p.ops,
      p.camera,
      p.iters,
      !!p.addC,
      !!p.julia,
      p.juliaC || null,
      p.deOption ?? null,
    ]);

  // Fill a grid element with clickable preset thumbnails. Each is rendered to an
  // offscreen texture and read straight back (renderToImage) — deterministic, so
  // a tile never shows a stale/other preset (the canvas-present race that made
  // e.g. Slab Box show the Pseudo-Kleinian picture). Cached by fingerprint, so
  // re-rendering the same set (other container, same category again) is free.
  async function renderThumbnails(presets, gridEl, onPick, opts = {}) {
    if (!hasGPU || !gridEl) return;
    // Tile resolution defaults to the preset-picker size; callers (e.g. the
    // Imposter game) can pass a larger W/H for crisp, non-grainy tiles.
    const W = opts.W || 168,
      H = opts.H || 112;
    const scratch = document.createElement("canvas");
    scratch.width = W;
    scratch.height = H;
    const s2d = scratch.getContext("2d");
    gridEl.innerHTML = "";
    const entries = presets.map((p) => {
      const b = document.createElement("button");
      b.className = "thumb";
      b.type = "button";
      const img = document.createElement("img");
      const lbl = document.createElement("div");
      lbl.className = "lbl";
      lbl.textContent = p.name;
      b.append(img, lbl);
      b.addEventListener("click", () => onPick(p));
      gridEl.appendChild(b);
      return { p, img, key: thumbKey(p) };
    });
    // Cache hits first (instant); only the misses touch the GPU.
    const misses = [];
    for (const e of entries) {
      const hit = thumbCache.get(e.key);
      if (hit) e.img.src = hit;
      else misses.push(e);
    }
    if (!misses.length) return;
    busy = true;
    try {
      for (const e of misses) {
        // Full quality (deScale 0.5) so thin/complex surfaces don't render holey.
        writeFrame(
          e.p,
          { steps: 200, eps: 0.001, deScale: 0.5 },
          [W, H],
          makeCamera(e.p.camera),
        );
        s2d.putImageData(await renderer.renderToImage(W, H), 0, 0);
        const url = scratch.toDataURL("image/png");
        thumbCache.set(e.key, url);
        e.img.src = url;
      }
    } catch (e) {
      console.error("thumbnails:", e);
    } finally {
      busy = false;
      scheduleDraw();
    }
  }

  return {
    hasGPU,
    backend,
    cam,
    isTouch,
    camObj,
    frameTo,
    bumpInteract,
    setFormula(f) {
      formula = f;
      scheduleDraw();
    },
    getFormula() {
      return formula;
    },
    setColoring(c) {
      coloring = c;
      scheduleDraw();
    },
    getColoring() {
      return coloring;
    },
    setAutoRotate(b) {
      autoRotate = b;
      if (b) scheduleDraw();
    },
    setSpinSpeed(degPerFrame) {
      spinSpeed = degPerFrame;
      if (autoRotate) scheduleDraw();
    },
    setSpinTilt(deg) {
      spinTilt = deg;
      if (autoRotate) scheduleDraw();
    },
    requestDraw: scheduleDraw,
    zoom(factor) {
      cam.zoom(factor);
      bumpInteract();
    },
    exportPNG,
    renderThumbnails,
  };
}

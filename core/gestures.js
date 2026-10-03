// Gesture math for the preview's camera controls (CAMERA_CONTROLS.md §3) —
// pure/explicit-state functions of positions and dt only, so they unit-test
// under plain node (the zoomsurface.js pattern). No DOM, no Date.now(), no
// rAF in here: preview.js (and the app's ASCII view) own the events and the
// clock and just call in.

// ── Wheel zoom (§3.1) ────────────────────────────────────────────────────────
// One wheel/trackpad event → a zoom factor PROPORTIONAL to the scroll delta.
// The old fixed ×1.1-per-event step made zoom speed depend on the EVENT RATE,
// not the gesture: a Mac trackpad emits a stream of small-delta events and
// rocketed, while a notched mouse wheel crawled. Exponential in pixels instead:
//
//   factor = exp(K_WHEEL · deltaPx)      K_WHEEL > 0
//
// so deltaY > 0 ⇒ factor > 1 ⇒ zoom OUT (the existing sign convention), equal
// deltas compose (f(a)·f(b) = f(a+b) — event coalescing changes nothing), and
// in/out are exact inverses (f(d)·f(−d) = 1 — the old 1.1/0.9 pair drifted by
// 1% per round trip). K_WHEEL is tuned so one classic wheel notch (~120 px)
// lands near the old ×1.1 step. The clamp bounds one EVENT, not the gesture —
// a flicked wheel still zooms far, just smoothly across its event stream.
const K_WHEEL = 1.1e-3;
const WHEEL_FACTOR_MAX = 1.25; // |factor − 1| ≤ 0.25 per event
const LINE_PX = 16; // DOM_DELTA_LINE → px (browsers vary; 16 ≈ a text line)

export function wheelZoomFactor(deltaY, deltaMode = 0, pagePx = 800) {
  const px =
    deltaMode === 1 ? deltaY * LINE_PX : deltaMode === 2 ? deltaY * pagePx : deltaY;
  const f = Math.exp(K_WHEEL * px);
  return Math.max(1 / WHEEL_FACTOR_MAX, Math.min(WHEEL_FACTOR_MAX, f));
}

// ── Orbit inertia (§3.3) ─────────────────────────────────────────────────────
// Velocity estimate during a drag: an EMA over the instantaneous per-move
// velocity, weighted by each sample's own dt so the estimate is frame-rate
// independent (α = min(dt/window, 1) — the spec's recurrence verbatim). The
// first sample primes the estimate directly (an EMA from 0 would need ~window
// ms to catch up, under-reporting short flicks).
export function makeVelocityTracker(windowMs = 80) {
  let vx = 0,
    vy = 0,
    primed = false;
  return {
    // dx/dy in DEGREES of orbit (post-sensitivity), dt in ms.
    push(dxDeg, dyDeg, dtMs) {
      if (!(dtMs > 0)) return; // duplicate/zero timestamps carry no velocity info
      const ix = (dxDeg / dtMs) * 1000; // °/s instantaneous
      const iy = (dyDeg / dtMs) * 1000;
      if (!primed) {
        vx = ix;
        vy = iy;
        primed = true;
        return;
      }
      const a = Math.min(dtMs / windowMs, 1);
      vx += (ix - vx) * a;
      vy += (iy - vy) * a;
    },
    speed: () => Math.hypot(vx, vy),
    velocity: () => ({ vx, vy }),
    reset() {
      vx = vy = 0;
      primed = false;
    },
  };
}

// Post-release glide: exponential decay with time constant tau (ms). NOT pure —
// it threads velocity state across frames; the caller owns the clock and passes
// dt (no Date.now/rAF here, per the testable-module rule). Each step integrates
// the decay analytically — Δ = v·(τ/1000)·(1 − e^(−dt/τ)) — so any dt slicing
// of the same wall time yields the same total travel (v·τ/1000 as dt→∞).
// `active` goes false once the velocity MAGNITUDE drops below `stop` (°/s).
export function makeGlide(vYawDegPerS, vPitchDegPerS, { tau = 180, stop = 2 } = {}) {
  let vy = vYawDegPerS,
    vp = vPitchDegPerS;
  return {
    step(dtMs) {
      const decay = Math.exp(-Math.max(0, dtMs) / tau);
      const gain = (tau / 1000) * (1 - decay);
      const out = { dYawDeg: vy * gain, dPitchDeg: vp * gain, active: true };
      vy *= decay;
      vp *= decay;
      out.active = Math.hypot(vy, vp) >= stop;
      return out;
    },
  };
}

// The release-speed gate (°/s): below this a drag just stops dead (a slow,
// deliberate placement shouldn't drift); above it the glide carries through.
export const GLIDE_MIN_SPEED = 30;

// ── Two-finger pan + zoom (§3.4) ─────────────────────────────────────────────
// Decompose one two-pointer move into simultaneous components — standard map
// behavior, no mode switching, no thresholding: centroid translation → pan
// (screen px, caller scales to world units), inter-pointer distance ratio →
// zoom factor (old/new — fingers spreading ⇒ < 1 ⇒ zoom in, matching the
// existing pinch semantics). a0/b0 are the two pointers before the move,
// a1/b1 after; degenerate distances (stacked pointers) yield zoom 1.
export function twoFingerDelta(a0, b0, a1, b1) {
  const d0 = Math.hypot(a0.x - b0.x, a0.y - b0.y);
  const d1 = Math.hypot(a1.x - b1.x, a1.y - b1.y);
  return {
    panX: (a1.x + b1.x - a0.x - b0.x) / 2,
    panY: (a1.y + b1.y - a0.y - b0.y) / 2,
    zoom: d0 > 0 && d1 > 0 ? d0 / d1 : 1,
  };
}

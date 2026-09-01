// DE-scaled cruise (CAMERA_CONTROLS.md §4) — the "fly into it" math. Speed is
// proportional to the distance estimate, so motion feels uniform at every
// scale and the camera can never crash: approaching a surface is asymptotic
// (dist shrinks by a fixed FRACTION per second, Zeno-style), and open-space
// drift moves at k·dist per second (scale-invariant). Pure dt-explicit
// functions only (the testable-module rule); the probe/repin wiring lives in
// preview.js / the app's ASCII view.

// Fraction of the remaining distance covered per second while closing in —
// also the open-space drift rate (× dist). ~0.7 ⇒ halve the gap every second.
export const CRUISE_K = 0.7;

// One cruise frame. dir +1 = fly in, −1 = back out (plain scale-invariant
// dolly). `hasHit` = the last surface probe (throttled by the caller) found
// something ahead; dist = current orbit distance.
// Returns { distFactor, drift }:
//   distFactor — multiply cam.dist by this (asymptotic approach / back-out;
//                also the no-DE degraded mode, matching zoomAtCenter's miss);
//   drift      — world units to translate the orbit target along fwd (open-
//                space cruise with a live DE: eye + pinned target ride
//                together, dist untouched, until something enters range).
export function cruiseAdvance({ dir, hasHit, dist }, dtMs, k = CRUISE_K) {
  const dt = Math.max(0, dtMs) / 1000;
  if (dir < 0) return { distFactor: Math.exp(k * dt), drift: 0 };
  if (hasHit) return { distFactor: Math.exp(-k * dt), drift: 0 };
  return { distFactor: 1, drift: k * dist * dt };
}

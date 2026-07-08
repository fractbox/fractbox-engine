// Zoom-to-surface probe (docs/design/DEEP_ZOOM.md §5 navigation). Pure and
// GPU-free — no WebGPU, no DOM — so it's unit-testable and shared by the
// preview's scroll/pinch zoom handlers.
//
// WHY: plain zoom shrinks cam.dist, sliding the eye toward the orbit target. For
// an unpanned camera the target is the object's CENTROID, so zooming in far
// enough drives the eye THROUGH the surface into the interior, where the frame
// washes out to a flat colour. Probing where the surface is straight ahead lets
// the caller re-pin the orbit target onto it, so zoom dollies toward the surface
// (asymptotically, never through it) instead of toward the centroid.

// March a distance estimator `de(x,y,z)` from `eye` along unit vector `fwd` to
// the first surface crossing. Returns the hit distance t (eye→surface), or null
// when there's nothing to retarget onto:
//   • de isn't a function (couldn't be built for this formula),
//   • the eye is already at/inside the surface (de(eye) ≤ near — retargeting
//     from inside would just pin the pivot to the wall we're buried in),
//   • the centre ray misses within [near, far] (looking at empty space).
// Loosely mirrors the shader marcher; exactness isn't needed since the caller
// re-probes on every zoom tick.
export function surfaceHitDist(de, eye, fwd, near, far) {
  if (typeof de !== "function") return null;
  if (!(de(eye[0], eye[1], eye[2]) > near)) return null; // inside / on / NaN
  let t = near;
  for (let i = 0; i < 192; i++) {
    const d = de(eye[0] + fwd[0] * t, eye[1] + fwd[1] * t, eye[2] + fwd[2] * t);
    if (!Number.isFinite(d)) return null;
    if (d < 1e-3 * t) return t; // hit (eps·t, matching the marcher's relative eps)
    t += Math.max(d * 0.7, 1e-6); // conservative step (deScale-ish); no zero-step stall
    if (t > far) return null; // ran past the far bound without hitting
  }
  return null;
}

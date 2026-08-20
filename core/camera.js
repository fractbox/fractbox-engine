// Orbit camera. Basis convention matches the native engine:
//   right = normalize(cross(fwd, worldUp))   with worldUp = +Z
//   up    = cross(right, fwd)
// so an op-list authored here frames the same way it will on the desktop.

const D2R = Math.PI / 180;

function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}
function norm(v) {
  const L = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / L, v[1] / L, v[2] / L];
}

// Deep zoom P4 — the global zoom floor (see cam.zoom). One constant so the
// camera and every preview.js zoom path clamp identically.
export const MIN_DIST = 1e-13;
// Perturbation tier floor (PERTURBATION_ZOOM_IMPL.md PR-4): the delta
// kernel is exact to the ~×10³⁰ δ-underflow wall, so pt-eligible formulas
// zoom to 1e-30. Callers pass it explicitly (cam.zoom's minDist override) —
// the default clamp stays MIN_DIST so nothing deepens by accident.
export const PT_MIN_DIST = 1e-30;
// Zoom-out ceiling (max orbit distance). Was inline in cam.zoom; named so the
// dist setter and any caller share one source of truth.
export const MAX_DIST = 200;

// The dist invariant, enforced on EVERY write to cam.dist (the accessor in
// makeCamera). It is the hard backstop the soft zoom brake (preview.js
// brakeZoomIn) can't be: the brake gates on PRE-zoom headroom, so one oversized
// step — a large anchored pinch from far out, a stale-pointer delta on resume,
// a NaN factor — sails through it and lands (or eases) past the precision wall.
// This clamp runs where the value is STORED, so no input path can persist an
// out-of-bounds or non-finite distance.
//   • non-finite (NaN / ±Infinity) → reject the write, keep the last good value
//     (a bad factor can no longer black out the render).
//   • floor: the deepest dist this formula may reach — the consumer supplies it
//     via cam.distFloor (the precision-wall distance, perturbation-aware, or the
//     numeric tier floor when there's no wall). The semantic floor is ALWAYS the
//     consumer's; absent one, this falls back only to the ABSOLUTE engine floor
//     PT_MIN_DIST (the deepest any tier allows) so it never blocks a legitimate
//     deep write — it exists purely to reject 0/negative/NaN, not to pick a tier.
//   • ceiling: MAX_DIST (fully zoomed out).
export function clampDist(v, prev, floor) {
  if (!Number.isFinite(v)) return prev;
  const lo = Number.isFinite(floor) && floor > 0 ? floor : PT_MIN_DIST;
  return Math.min(MAX_DIST, Math.max(lo, v));
}

// ── #551 — is a camera push a CONTINUATION, or a jump somewhere else? ──────
// preview.frameTo is the single entry point for "put the camera here", and it
// serves two callers with opposite needs:
//
//   · a RETARGET — a share link, a preset, a restored view. The camera lands
//     somewhere unrelated to where it was, so every piece of descent state
//     (the manual Detail override, the perturbation tier's exact target and
//     its reference orbit) describes the OLD place and must be dropped.
//
//   · a CONTINUOUS push — flythrough playback, Wander, timeline scrub, the
//     camera tween. These write a camera EVERY FRAME, each one a small step
//     from the last. Treating every one as a retarget resets the descent state
//     sixty times a second, which is #551: the perturbation reference orbit
//     never survives long enough to render against, and pressing Play silently
//     wipes a manual Detail setting.
//
// The test is the frustum, which is also the perturbation tier's OWN
// near-field argument (PERTURBATION_ZOOM.md §3: at zoom ×10^Z everything that
// resolves detail "lies within ~10⁻ᶻ·O(1) of T", which is why one reference
// orbit can serve a whole view). A push whose new pivot is still inside the
// view it is moving from continues that view; one that lands outside it is a
// jump to somewhere else. The radius is dist·tan(fov/2) — the half-height of
// the frustum at the target plane — taken at the SMALLER of the two distances,
// i.e. the pivot must be inside BOTH the view being left and the view being
// entered. Judging by the larger one instead lets a big zoom-out swallow a
// genuine retarget whole: loading a preset (dist 24, pivot at the origin) from
// a view parked at ×10¹⁶ opens a frustum nine units wide, which contains the
// old pivot, and the reset that load needs would never fire.
//
// Deliberately NOT part of the test:
//   · dist. The reference orbit is a function of (target, iters, formula) and
//     not of dist at all, so even a large dolly step is a continuation. (The
//     ≤2-decade re-pin cadence in PERTURBATION_ZOOM.md §9 governs where the
//     surface PROBE may land — a different question from orbit validity.)
//   · yaw / pitch / fov. Orbiting around a pivot cannot invalidate an orbit
//     built at that pivot.
//   · op values. Judged by the caller, which owns formula identity; a
//     param-morphing flight moves values every frame and that is continuous.
//
// A camera carrying sub-f64 target words (TAG.VIEW v2 `targetLo`) is ALWAYS a
// retarget: only a share/restore produces them, and their exact words must win
// over whatever the live exact target currently holds.
export function isContinuousPush(from, to) {
  if (!from || !to) return false;
  if (Array.isArray(to.targetLo) || Array.isArray(to.targetLo2)) return false;
  const dA = from.dist,
    dB = to.dist;
  if (!Number.isFinite(dA) || !Number.isFinite(dB) || dA <= 0 || dB <= 0)
    return false;
  const fov = Number.isFinite(to.fovDeg) ? to.fovDeg : from.fovDeg;
  if (!Number.isFinite(fov) || fov <= 0 || fov >= 180) return false;
  const tA = from.target ?? [0, 0, 0];
  const tB = Array.isArray(to.target) ? to.target : [0, 0, 0];
  for (let i = 0; i < 3; i++)
    if (!Number.isFinite(tA[i] ?? 0) || !Number.isFinite(tB[i] ?? 0))
      return false;
  const moved = Math.hypot(
    (tB[0] ?? 0) - (tA[0] ?? 0),
    (tB[1] ?? 0) - (tA[1] ?? 0),
    (tB[2] ?? 0) - (tA[2] ?? 0),
  );
  return moved <= Math.min(dA, dB) * Math.tan((fov * D2R) / 2);
}

export function makeCamera(init = {}) {
  // Default each field (BLANK.camera values) so a partial/absent camera can't
  // produce a NaN eye/fov (→ permanently black render). Honor an explicit
  // `target` — cameras round-tripped through camObj() or a panned/deep-zoomed
  // preset carry one, and dropping it framed those views wrongly.
  const fin = (v, d) => (Number.isFinite(v) ? v : d);
  const cam = {
    yaw: fin(init.yawDeg, 35) * D2R,
    pitch: fin(init.pitchDeg, 22) * D2R,
    fov: fin(init.fovDeg, 42) * D2R,
    target: Array.isArray(init.target)
      ? [0, 1, 2].map((i) => fin(init.target[i], 0))
      : [0, 0, 0],
  };
  // `dist` is an accessor so the clampDist invariant holds for EVERY writer —
  // present or future, every input path (wheel, pinch, cruise, the eased
  // zoom-chase, share-load via frameTo, a resume burst). The wall floor is
  // formula/df64/perturbation-dependent and known only to the consumer, which
  // installs `cam.distFloor` (a () => number). distFloor is non-enumerable so it
  // stays out of {...cam} spreads (reactor.ts).
  let _dist = clampDist(fin(init.dist, 14), 14, null);
  Object.defineProperty(cam, "distFloor", {
    value: null,
    writable: true,
    enumerable: false,
  });
  Object.defineProperty(cam, "dist", {
    enumerable: true,
    configurable: true,
    get() {
      return _dist;
    },
    set(v) {
      _dist = clampDist(v, _dist, cam.distFloor && cam.distFloor());
    },
  });

  // Spherical → cartesian forward (looking AT the target from the orbit point).
  cam.basis = function () {
    const cp = Math.cos(cam.pitch),
      sp = Math.sin(cam.pitch);
    const dir = [cp * Math.sin(cam.yaw), cp * Math.cos(cam.yaw), sp]; // points target→eye
    // The residual eye − target, computed DIRECTLY as dir·dist — f64-exact at
    // any depth. Forming eye = target + dir·dist and subtracting the target
    // back out absorbs dir·dist into the target's f64 grid and recovers it
    // only to ~eps·max(|target|,1) absolute — a hard residual ceiling around
    // ×10¹⁵⁻¹⁶ (PERTURBATION_ZOOM_IMPL.md D11). writeGlobals prefers this.
    const roRel = [dir[0] * cam.dist, dir[1] * cam.dist, dir[2] * cam.dist];
    const eye = [
      cam.target[0] + roRel[0],
      cam.target[1] + roRel[1],
      cam.target[2] + roRel[2],
    ];
    const fwd = norm([-dir[0], -dir[1], -dir[2]]); // eye → target
    let right = cross(fwd, [0, 0, 1]);
    // Looking (nearly) straight up/down: cross(fwd, worldUp) collapses toward
    // zero for EVERY yaw, but its DIRECTION has a well-defined limit as
    // pitch→±90° — cross(fwd,Z) = cp·[-cos(yaw), sin(yaw), 0] (cp=cos(pitch)),
    // so the limit is [-cos(yaw), sin(yaw), 0] regardless of which pole. Using
    // that instead of a fixed axis keeps `right` continuous through the pole
    // (#463 — a fixed [1,0,0] flips handedness relative to every nearby pitch
    // at yaw=0, where the true limit is [-1,0,0]), so an exact pitchDeg:90
    // "top" view needs no epsilon fudge to look straight down.
    if (Math.hypot(...right) < 1e-4)
      right = [-Math.cos(cam.yaw), Math.sin(cam.yaw), 0];
    right = norm(right);
    const up = norm(cross(right, fwd));
    return { eye, fwd, right, up, roRel };
  };

  cam.orbit = function (dxDeg, dyDeg) {
    cam.yaw += dxDeg * D2R;
    cam.pitch += dyDeg * D2R;
    const lim = 89 * D2R;
    cam.pitch = Math.max(-lim, Math.min(lim, cam.pitch));
  };

  // Auto-spin around an arbitrary world axis: rotate the orbit direction
  // (target→eye) around `axis` by `deg` (Rodrigues), then re-derive yaw/pitch.
  // Because pitch comes back via asin of a unit vector it stays in range, so a
  // tilted/vertical spin tumbles cleanly with no clamp jam (axis=+Z ⇒ turntable).
  cam.spinAround = function (axis, deg) {
    const aL = Math.hypot(axis[0], axis[1], axis[2]) || 1;
    const a = [axis[0] / aL, axis[1] / aL, axis[2] / aL];
    const cp = Math.cos(cam.pitch),
      sp = Math.sin(cam.pitch);
    const v = [cp * Math.sin(cam.yaw), cp * Math.cos(cam.yaw), sp];
    // negate so +deg about +Z advances yaw the same way orbit(+deg, 0) does
    const t = -deg * D2R,
      c = Math.cos(t),
      s = Math.sin(t);
    const kv = cross(a, v);
    const kd = a[0] * v[0] + a[1] * v[1] + a[2] * v[2];
    const r = norm([
      v[0] * c + kv[0] * s + a[0] * kd * (1 - c),
      v[1] * c + kv[1] * s + a[1] * kd * (1 - c),
      v[2] * c + kv[2] * s + a[2] * kd * (1 - c),
    ]);
    cam.pitch = Math.asin(Math.max(-1, Math.min(1, r[2])));
    cam.yaw = Math.atan2(r[0], r[1]);
  };
  // Deep zoom Phase 1 (docs/design/DEEP_ZOOM.md §5): unclamp from the old
  // [1.2, 40] so zoom can go far deeper (toward the origin, this stays crisp —
  // see the doc's precision analysis) or further out. The lower bound is a
  // numerical floor (avoid 0/negative dist), not a usability one.
  // Phase 4 lifted it 1e-9 → MIN_DIST: the old floor was an f32-era ceiling
  // (~×2.4·10¹⁰ from the default framing) that bound BEFORE the df64 wall
  // (~×10¹³); the new floor sits past the df64 wall with margin.
  // `minDist` (PERTURBATION_ZOOM_IMPL.md PR-4/M1): the perturbation tier's
  // floor is 1e-30 — but only its callers know eligibility, so the deeper
  // floor is an explicit override; the default stays the df64-era MIN_DIST.
  // Without the parameter this internal clamp silently wins over any deeper
  // caller-side floor and the max depth becomes gesture-dependent.
  cam.zoom = function (factor, minDist = MIN_DIST) {
    cam.dist = Math.max(minDist, Math.min(200, cam.dist * factor));
  };
  // Pan the orbit target in the current view plane. dx/dy are WORLD-space
  // deltas already scaled by the caller (mirrors orbit(dxDeg,dyDeg) — camera.js
  // stays canvas/fov-agnostic; preview.js computes the pixel→world factor from
  // fov/dist/canvas size so a drag tracks the surface under the cursor).
  // `target` stays a plain JS number (f64) — the deep-zoom offset (§4.1).
  cam.pan = function (dx, dy) {
    const b = cam.basis();
    cam.target[0] += b.right[0] * dx + b.up[0] * dy;
    cam.target[1] += b.right[1] * dx + b.up[1] * dy;
    cam.target[2] += b.right[2] * dx + b.up[2] * dy;
  };
  return cam;
}

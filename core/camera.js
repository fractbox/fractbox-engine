// Orbit camera. Basis convention matches the native engine:
//   right = normalize(cross(fwd, worldUp))   with worldUp = +Z
//   up    = cross(right, fwd)
// so an op-list authored here frames the same way it will on the desktop.

const D2R = Math.PI / 180;

function cross(a, b) {
  return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
}
function norm(v) {
  const L = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0]/L, v[1]/L, v[2]/L];
}

export function makeCamera(init) {
  const cam = {
    yaw: init.yawDeg * D2R,
    pitch: init.pitchDeg * D2R,
    dist: init.dist,
    fov: init.fovDeg * D2R,
    target: [0, 0, 0],
  };

  // Spherical → cartesian forward (looking AT the target from the orbit point).
  cam.basis = function () {
    const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
    const dir = [cp * Math.sin(cam.yaw), cp * Math.cos(cam.yaw), sp]; // points target→eye
    const eye = [
      cam.target[0] + dir[0] * cam.dist,
      cam.target[1] + dir[1] * cam.dist,
      cam.target[2] + dir[2] * cam.dist,
    ];
    const fwd = norm([-dir[0], -dir[1], -dir[2]]);     // eye → target
    let right = cross(fwd, [0, 0, 1]);
    if (Math.hypot(...right) < 1e-4) right = [1, 0, 0]; // degenerate (looking along Z)
    right = norm(right);
    const up = norm(cross(right, fwd));
    return { eye, fwd, right, up };
  };

  cam.orbit = function (dxDeg, dyDeg) {
    cam.yaw   += dxDeg * D2R;
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
    const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
    const v = [cp * Math.sin(cam.yaw), cp * Math.cos(cam.yaw), sp];
    // negate so +deg about +Z advances yaw the same way orbit(+deg, 0) does
    const t = -deg * D2R, c = Math.cos(t), s = Math.sin(t);
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
  cam.zoom = function (factor) {
    cam.dist = Math.max(1.2, Math.min(40, cam.dist * factor));
  };
  return cam;
}

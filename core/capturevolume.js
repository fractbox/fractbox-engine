// Convex capture volumes (CAPTURE_VOLUME_SHAPES.md) — the ONE definition of a
// capture frame's SHAPE, shared by the CPU `captureView` (splatcapture.js) and
// the WGSL capture fragment (shader.js). Two questions define any convex volume:
//
//   support(a) — half-extent of the volume's SHADOW along a unit axis `a`.
//                Sizes the per-view sampling window and the march depth, so
//                every view covers the whole volume.
//   inside(p)  — is p (RELATIVE to the frame centre) in the volume?
//                Rejects hits that land in the window but outside the volume.
//
// BOTH are required, including for a box. They do opposite jobs: `support`
// alone only ENLARGES the sampled region (a box's L1 support is up to √3·e for
// an oblique axis) with nothing to trim the excess; `inside` alone leaves
// oblique views under-covering the volume. Together the captured set is exactly
// the requested volume — which is what makes a "cube" actually a cube.
//
// Before this, neither capture path tested a hit at all: the window was a fixed
// ±radius square and every hit inside `tmax` was recorded, so the captured
// region was a union of oblique slabs reaching ~1.74·r (guaranteed) to 2.06·r
// (the √(2r²+(1.5r)²) ceiling) — the rounded, view-count-dependent boundary
// users saw as "spherical in the back" in a viewer.
//
// `frame.radius` is NOT retired: it stays the frame's SCALE scalar (capture eps,
// the AO probe step, r0/pitch), where it is already max-axis-driven
// (frameFormula sets radius = 1.1·maxExt) and where #351 shows a too-SMALL
// value silently returns zero hits. Only the GEOMETRY moves to `ext`.

export const VOL_BOX = 0; // cuboid
export const VOL_ELLIPSOID = 1; // support = L2, inside = Σ(p/e)² ≤ 1
export const VOL_CYLINDER = 2; // axis = LOCAL z (see volBasis — orientable)

// ── Orientation ─────────────────────────────────────────────────────────────
// A volume may be ROTATED. `frame.rot` carries its first two local axes in world
// space, [r0x,r0y,r0z, r1x,r1y,r1z]; the third is their cross product, so an
// orthonormal basis costs 6 numbers on the wire and 8 in the uniform rather than
// 9 and 12. Absent ⇒ identity, which is the overwhelmingly common case and pays
// nothing.
//
// This is what generalises the three questions instead of special-casing them:
// `support`, `inside` and `rayInterval` each become "express the argument in the
// volume's local frame, then run exactly the math they already ran". A cylinder
// along an arbitrary direction is then the SAME code as one along z, and the
// UI's X/Y/Z presets are just three stored bases — so adding drag-to-rotate
// later is UI work with no engine change.
//
// Why it is only a rotation: `support(a) = max_p⟨p,a⟩` over the body, and with
// p = Σ qᵢ·rᵢ that is `max_q Σ qᵢ⟨rᵢ,a⟩` = the local support of a expressed in
// the same basis. The basis being ORTHONORMAL is what keeps |d| = 1 and leaves
// the ray's `t` parameter untouched, so no interval has to be rescaled.
export function volBasis(frame) {
  const r = frame.rot;
  if (!Array.isArray(r) || r.length !== 6 || !r.every((v) => Number.isFinite(v)))
    return null; // identity
  const n = (v) => {
    const L = Math.hypot(v[0], v[1], v[2]);
    return L > 1e-12 ? [v[0] / L, v[1] / L, v[2] / L] : null;
  };
  const r0 = n([r[0], r[1], r[2]]);
  let r1 = n([r[3], r[4], r[5]]);
  if (!r0 || !r1) return null; // degenerate ⇒ identity rather than a zero volume
  // Gram-Schmidt: trust the axis, re-derive the rest, so a hand-written or
  // drag-accumulated basis can't slowly shear the volume.
  const dot = r0[0] * r1[0] + r0[1] * r1[1] + r0[2] * r1[2];
  r1 = n([r1[0] - dot * r0[0], r1[1] - dot * r0[1], r1[2] - dot * r0[2]]);
  if (!r1) return null; // r1 ∥ r0
  const r2 = [
    r0[1] * r1[2] - r0[2] * r1[1],
    r0[2] * r1[0] - r0[0] * r1[2],
    r0[0] * r1[1] - r0[1] * r1[0],
  ];
  return [r0, r1, r2];
}

// Express a world vector in the volume's local frame. Used for POINTS (offsets
// from the centre) and for DIRECTIONS alike — the basis is orthonormal, so the
// same transform serves both.
export function volToLocal(basis, x, y, z) {
  if (!basis) return [x, y, z];
  const [r0, r1, r2] = basis;
  return [
    x * r0[0] + y * r0[1] + z * r0[2],
    x * r1[0] + y * r1[1] + z * r1[2],
    x * r2[0] + y * r2[1] + z * r2[2],
  ];
}

// A frame's per-axis half-extents. Frames that predate cuboid support carry no
// `ext` (or a stale uniform one) — fall back to the scalar so an old job, a
// hand-built {center, radius} probe literal, or a share link still captures.
export function volExt(frame) {
  const e = frame.ext;
  return Array.isArray(e) && e.length === 3 && e.every((v) => v > 0)
    ? e
    : [frame.radius, frame.radius, frame.radius];
}

export function volKind(frame) {
  return frame.kind ?? VOL_BOX;
}

// Half-width of the volume's shadow along unit axis `a`. For a box this is the
// L1 norm Σ|eᵢ·aᵢ|, which equals e only when `a` is exactly axis-aligned and
// rises to √3·e on the body diagonal — the reason a uniform `ext` does NOT
// reproduce the old constant-radius window.
export function volSupport(frame, aWorld) {
  const e = volExt(frame);
  const a = volToLocal(volBasis(frame), aWorld[0], aWorld[1], aWorld[2]);
  switch (volKind(frame)) {
    case VOL_ELLIPSOID:
      return Math.hypot(e[0] * a[0], e[1] * a[1], e[2] * a[2]);
    case VOL_CYLINDER:
      return Math.hypot(e[0] * a[0], e[1] * a[1]) + Math.abs(e[2] * a[2]);
    default:
      return (
        Math.abs(e[0] * a[0]) + Math.abs(e[1] * a[1]) + Math.abs(e[2] * a[2])
      );
  }
}

// Is (x,y,z) — RELATIVE to the frame centre — inside the volume? A hair of
// slack (1 + 1e-6) keeps a hit that lands exactly ON a face from being dropped
// by f32 rounding, which would punch pinholes in the box's own walls.
export function volInside(frame, wx, wy, wz) {
  const e = volExt(frame);
  const s = 1 + 1e-6;
  const [x, y, z] = volToLocal(volBasis(frame), wx, wy, wz);
  switch (volKind(frame)) {
    case VOL_ELLIPSOID: {
      const u = x / e[0],
        v = y / e[1],
        w = z / e[2];
      return u * u + v * v + w * w <= s;
    }
    case VOL_CYLINDER: {
      const u = x / e[0],
        v = y / e[1];
      return u * u + v * v <= s && Math.abs(z) <= e[2] * s;
    }
    default:
      return (
        Math.abs(x) <= e[0] * s &&
        Math.abs(y) <= e[1] * s &&
        Math.abs(z) <= e[2] * s
      );
  }
}

// The ray's inside-interval [t0, t1], or null when it misses the volume — the
// third question a convex volume answers, and the one that makes the other two
// affordable. `o` is the ray origin RELATIVE to the frame centre, `d` a unit
// direction. Convexity is what makes this well-posed: the set of t with
// inside(o + t·d) is a SINGLE interval, so a march clipped to [t0, t1] can never
// skip inside geometry.
//
// Why it exists (#450): without it a ray starts 1.5·hd behind the volume and
// sphere-traces the whole 3·hd depth, so every surface it meets OUTSIDE the
// volume is a hit that must be stepped over at 3·eps a time — one march step,
// and one unit of the shared budget, per 3·eps of solid. Crossing a single wall
// costs thousands of steps against a 200-step budget, so a volume with anything
// in front of it (the common case: a box the user drew INSIDE a sponge) ran out
// of budget before reaching the geometry they framed and captured nothing.
// Clipping deletes that work rather than paying for it: the march now begins at
// the volume's own boundary, and `inside` degrades to a cheap f32 guard that a
// correctly-clipped ray never trips.
export function volRayInterval(frame, oWorld, dWorld) {
  const e = volExt(frame);
  const kind = volKind(frame);
  // Both into the volume's local frame; the basis is orthonormal, so `t` means
  // the same thing on both sides and the interval needs no rescaling.
  const basis = volBasis(frame);
  const o = volToLocal(basis, oWorld[0], oWorld[1], oWorld[2]);
  const d = volToLocal(basis, dWorld[0], dWorld[1], dWorld[2]);
  let t0 = -Infinity,
    t1 = Infinity;

  // Slab test along one axis: |o + t·d| ≤ e. Returns false when the ray is
  // parallel to the slab and outside it (no t can bring it in).
  const slab = (oi, di, ei) => {
    if (Math.abs(di) < 1e-12) return Math.abs(oi) <= ei;
    const a = (-ei - oi) / di,
      b = (ei - oi) / di;
    t0 = Math.max(t0, Math.min(a, b));
    t1 = Math.min(t1, Math.max(a, b));
    return true;
  };
  // Quadratic |u + t·v|² = 1 over the axes in `ax` (the ellipsoid's 3, the
  // cylinder's 2), in the space where the volume is the unit ball.
  const quad = (ax) => {
    let A = 0,
      B = 0,
      C = -1;
    for (const i of ax) {
      const u = o[i] / e[i],
        v = d[i] / e[i];
      A += v * v;
      B += 2 * u * v;
      C += u * u;
    }
    if (A < 1e-30) return C <= 0; // direction degenerate in these axes
    const disc = B * B - 4 * A * C;
    if (disc < 0) return false; // misses the (cylinder/ellipsoid) body
    const s = Math.sqrt(disc);
    t0 = Math.max(t0, (-B - s) / (2 * A));
    t1 = Math.min(t1, (-B + s) / (2 * A));
    return true;
  };

  let ok;
  switch (kind) {
    case VOL_ELLIPSOID:
      ok = quad([0, 1, 2]);
      break;
    case VOL_CYLINDER:
      ok = quad([0, 1]) && slab(o[2], d[2], e[2]);
      break;
    default:
      ok =
        slab(o[0], d[0], e[0]) &&
        slab(o[1], d[1], e[1]) &&
        slab(o[2], d[2], e[2]);
  }
  if (!ok || t1 < t0) return null;
  return [t0, t1];
}

// The guaranteed-captured reach, as a multiple of the frame's own half-extent:
// with support-sized windows every point of the volume is covered by every view,
// so the volume is captured EXACTLY and the honest clip threshold is its own
// surface. Kept as a named export so the UI's clip warning cites one rule
// rather than re-deriving a magic number (#443: the warning fired at 1.0·r when
// the old union-of-slabs capture did not actually lose anything until ~1.74·r).
export function volClipHalfExtents(frame) {
  return volExt(frame);
}

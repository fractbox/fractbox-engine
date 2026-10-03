// Multi-view first-hit DE capture → oriented, colored point cloud (UE splat
// export §1.2 / spec §5.1-5.2). CPU path (Option A): portable, headless,
// Node-testable — no GPU. The DE and albedo are the exact engine ones
// (cpu.js makeDE / makePointAlbedo), so capture is formula-faithful. Flat,
// scene AND hybrid formulas (S1a); scenes get a capture-truth frame refine +
// a march-step deScale matching the live renderer. No DOM.

import { makeDE, makePointAlbedo } from "./cpu.js";
import { surfaceLean } from "./evaluate.js";
import { makeCamera } from "./camera.js";
import { sceneDeScale, itersForMagnification } from "./renderpolicy.js";
import { looseDE, hybridLooseDE } from "./stability.js";
import {
  volSupport,
  volInside,
  volExt,
  volRayInterval,
} from "./capturevolume.js";

const cross3 = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const len3 = (a) => Math.hypot(a[0], a[1], a[2]);
const norm3v = (a) => {
  const l = len3(a) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};

// ── The capture's surface epsilon — ONE source of truth (#496 + #507) ────────
// `|DE| < eps ⇒ hit`, read by the CPU captureView below, by the WGSL capture
// pass (renderer.js packs it into CaptureU), by the S-2 refine and by the
// fidelity metrics (core/splatmetrics.js) — which must all agree exactly, or a
// metric stops describing the capture it is measuring. It was five hand-typed
// copies of the same expression (#507); a derived value can't drift.
//
// The rule is a MAXIMUM of lower bounds, never an override: eps is only ever
// RAISED. Each term states a reason this march cannot resolve anything finer,
// and the binding one is whichever reason bites hardest. #496 and #507 arrived
// independently with one term each; they compose, because a small crop of a
// floored DE suffers from both at once.
//
//   EPS_SCALE · radius — the scale-relative term. `radius` is the frame's SCALE
//     scalar (CAPTURE_VOLUME_SHAPES.md), so a whole-object capture gets an eps
//     proportional to the object. The historical rule, and still the default.
//
//   frame.epsFloor (#496) — an ABSOLUTE floor in world units, set by the
//     FRAMING layer (kit/splatexport.ts, objectEpsFloor below) to the eps the
//     WHOLE OBJECT would have used. How close a DE gets to zero near the true
//     surface is a property of the FIELD, not of the box you look through, so a
//     crop marched at an eps scaled to the CROP stops registering the same
//     surface: at 1/8.7 the object's size the Menger's hit rate fell
//     12.5% → 0.14%, i.e. 8972 splats → 71 — #496's entire symptom ("far fewer
//     splats than the whole-object capture"). The floor makes a sub-region
//     strictly no worse than the capture the user is comparing it to, and is
//     inert on a frame already at or above object scale (#518's oversized box).
//
//   frame.epsMeasured (#507) — the DE's OWN convergence floor: measured, not
//     assumed, by deConvergenceFloor below and stamped by withCaptureEps at
//     EPS_FLOOR_FACTOR × the median. Where #496's term says "this crop is
//     small", this one says "this DE bottoms out" — the sharp Menger reaches
//     ~5e-3 on a flat face at ANY march resolution. Kept a SEPARATE field from
//     epsFloor deliberately: one is a property of the framing and one of the
//     field, they are set by different layers, and collapsing them would make a
//     regression in either indistinguishable from the other.
//
// Deliberately NOT done by growing `radius`: that is also the AO tap scale
// (aoScale's `h = max(4·eps, 0.01·radius)`), and inflating it ~10× on a
// close-up coarsens every AO tap to a quarter of the box — visibly smudged
// colors on exactly the export these fixes exist to sharpen.
export const EPS_SCALE = 3e-4; // eps as a fraction of the frame's radius
export const EPS_MIN = 1e-5; // hard absolute floor (degenerate/zero radius)
export function captureEps(frame) {
  return Math.max(
    EPS_SCALE * frame.radius,
    frame.epsFloor ?? 0,
    frame.epsMeasured ?? 0,
    EPS_MIN,
  );
}

// The k-th of n golden-angle sphere directions (unit). Exported for the Worker
// (which captures a view subset) and the tests.
export function fibonacciDir(k, n) {
  const ga = Math.PI * (3 - Math.sqrt(5)); // golden angle
  const z = 1 - (2 * k + 1) / n; // cell-centered z ∈ (−1, 1)
  const r = Math.sqrt(Math.max(0, 1 - z * z));
  const th = ga * k;
  return [r * Math.cos(th), r * Math.sin(th), z];
}

// D2 framing: surfaceLean with an auto-expanding region. surfaceLean's lean/ext
// are region-normalized (÷R) — this multiplies back to WORLD units. Returns
// { center:[x,y,z], ext:[x,y,z], radius, diag, hits } or null when nothing is
// found at any region (caller falls back to formula.camera, §3 step 2).
export function frameFormula(formula) {
  let best = null;
  let prevMax = -1;
  for (let R = 2.5; ; R *= 2) {
    const l2 = Math.log2(R / 2.5);
    const sl = surfaceLean(formula, {
      region: R,
      leanGrid: Math.round(12 + 4 * l2), // §1.2: scale probe density with R
      marchSteps: Math.round(64 + 32 * l2),
    });
    if (sl === null) break; // unsupported ops
    if (sl.hits) {
      const ext = [sl.ext.x * R, sl.ext.y * R, sl.ext.z * R]; // → world
      const maxExt = Math.max(ext[0], ext[1], ext[2]);
      // `ext` is the capture VOLUME now (CAPTURE_VOLUME_SHAPES.md) and stays the
      // bare fit: padding it per-axis by the 1.10 below was measurably worse
      // (Gnarl Dunes normalAgreement 0.7615 → 0.7241, and stream/batch colorDrift
      // 1.2e-2 → 3.7e-2), because the extra shell is sparse outer structure that
      // coarsens the grid for the geometry that matters. It is also the number
      // the UI's Volume row reports, so a hidden 1.1× would be its own lie.
      // radius keeps the 1.10 margin as the SCALE scalar it always was (eps, AO
      // probe, r0) — unchanged from before this feature.
      best = {
        center: [sl.x * R, sl.y * R, sl.z * R],
        ext,
        radius: 1.1 * maxExt, // = max(ext)
        diag: 2 * len3(ext),
        hits: sl.hits,
      };
      // Converged: the world extent stopped growing (< 5% between doublings).
      if (prevMax > 0 && Math.abs(maxExt - prevMax) <= 0.05 * prevMax) break;
      prevMax = maxExt;
    }
    if (R >= 40) break;
  }
  return best && best.radius > 0 ? best : null;
}

// 5-tap DE ambient occlusion along the surface normal (§S1d), Iñigo-Quilez
// style: probe outward at i·h and see how much closer geometry sits than free
// space. Returns a scalar in [1−aoStrength, 1] that multiplies albedo before
// Reduce (so dedup averages already-darkened colors). NOTE: h uses the GLOBAL
// frame.radius — for a small object in a large scene bbox the taps can be too
// coarse to read its own crevices (S1d scene caveat); the deferred fix is a
// per-object h (S2 GPU depth) or defaulting aoStrength to 0 for scenes.
// Exported for the P0 fidelity harness (core/splatmetrics.js color-drift metric):
// it must reproduce this exact pre-multiplied AO chain, not re-derive it (§P0/R-3).
export function aoScale(de, px, py, pz, nX, nY, nZ, eps, radius, aoStrength) {
  const h = Math.max(4 * eps, 0.01 * radius); // scale-relative step
  let occ = 0,
    wsum = 0;
  for (let i = 1; i <= 5; i++) {
    const w = 1 / (1 << i); // 2^−i
    wsum += w;
    const di = i * h;
    const dd = de(px + nX * di, py + nY * di, pz + nZ * di);
    occ += w * Math.max(0, 1 - dd / di);
  }
  return 1 - aoStrength * (occ / wsum);
}

// The orthographic ray basis for one view — the ONE source of the ray-grid
// convention, shared by the CPU captureView here and (via §S2) the GPU capture
// pass (which packs rgt/up/oc into CaptureU). Returns unit direction `d`, the
// screen `rgt`/`up` axes (rgt ∥ cross(d,+Z), degenerate d∥Z → [1,0,0]), and the
// ray-plane center `oc` = center − d·1.5·radius. A pixel's ray origin is
// oc + rgt·sx + up·sy with sx,sy ∈ ±radius.
export function viewBasis(view, frame) {
  const { center } = frame;
  const d = norm3v(view);
  let rgt = cross3(d, [0, 0, 1]);
  if (len3(rgt) < 1e-4) rgt = [1, 0, 0]; // d ∥ Z — pick a stable right
  rgt = norm3v(rgt);
  const up = cross3(rgt, d); // unit (rgt ⊥ d, both unit)
  // Window/depth from the volume's own shadow (CAPTURE_VOLUME_SHAPES.md), so
  // every view covers the whole volume however oblique it is. Reduces to the
  // old constant ±radius only for an axis-aligned view of a uniform ext; the
  // 1.5×/3× margins are today's, now proportional to the per-view depth.
  const hu = volSupport(frame, rgt);
  const hv = volSupport(frame, up);
  const hd = volSupport(frame, d);
  const oc = [
    center[0] - d[0] * 1.5 * hd, // origin plane, 1.5·hd behind center
    center[1] - d[1] * 1.5 * hd,
    center[2] - d[2] * 1.5 * hd,
  ];
  return { d, rgt, up, oc, hu, hv, hd };
}

// ── #507: the DE's own convergence floor ─────────────────────────────────────
// A loose analytic IFS DE does not descend to 0 at its surface — it bottoms out.
// The sharp Menger (`abs · sort · scale ×3 · z-fold`, deOption 2) reaches ~5e-3
// on a flat face and no closer, at ANY march resolution: that IS its floor.
// The capture's default eps is 3e-4·radius = 3.3e-4 there — an order of
// magnitude BELOW the floor — with two consequences, both of them the bug in
// #507:
//   • face rays never satisfy |DE| < eps. They crawl at ~floor per step, spend
//     their whole budget inside the material and are dropped. Measured on the
//     Menger: 19% of probe rays exhausted, 11.5% hit.
//   • the hits that DO land are the sparse creases where the DE genuinely
//     reaches 0, and the 6-tap ∇DE at h = 2·eps there reads the floor's own
//     high-frequency structure rather than the surface. Measured normals were
//     UNIFORM over nz ∈ [−1, 1] on a sponge whose faces are axis-aligned; with
//     the default coloring (mode 0 Surface, mixT = 0.5 + 0.5·nz) that is a
//     direct read of the normal, so the export came out as salt-and-pepper
//     colA/colB confetti — the reporter's screenshot exactly.
// The live renderer never sees this because its hit test is `DE·deScale < eps·t`
// with eps 1.2e-3 and t ≈ the camera distance, i.e. ~1.1e-2 at the object —
// comfortably ABOVE the floor, on the smooth part of the DE's ramp where
// |∇DE| = 1.000 and the normal is exact. So this is a render-parity fix: the
// export should stop where the render stops.
//
// The floor is measured, not assumed, and it is measured off the one signal
// that means "this DE cannot get closer": a ray that exhausts its step budget
// while still hugging geometry. (The renderer already treats that ray as a hit
// — cpu.js "budget exhausted while hugging geometry"; the capture just drops
// it.) Their median |DE| is the floor.
export const EPS_FLOOR_FACTOR = 3; // eps target as a multiple of the floor
export const EPS_FLOOR_MIN_RAYS = 0.02; // fraction of probe rays that must stick
const EPS_FLOOR_HUG = 0.02; // |DE| < this·radius ⇒ "hugging", not lost in space

// Returns { floor, stuck, rays }. `floor` is 0 when the DE converges — which is
// the overwhelmingly common case (measured across the preset library: only the
// loose analytic stacks stick at all, and they stick on ≥11% of rays while
// every converging formula sticks on ≤0.9%, so EPS_FLOOR_MIN_RAYS has an order
// of magnitude of margin on both sides).
export function deConvergenceFloor(de, frame, opts = {}) {
  const views = opts.probeViews ?? 16;
  const res = opts.probeRes ?? 32;
  const deScale = opts.deScale ?? 1;
  const maxSteps = opts.maxSteps ?? 200;
  const eps = captureEps(frame);
  const { center, radius } = frame;
  const hug = EPS_FLOOR_HUG * radius;
  const stuck = [];
  let rays = 0;
  for (let k = 0; k < views; k++) {
    const { d, rgt, up, oc, hu, hv, hd } = viewBasis(
      fibonacciDir(k, views),
      frame,
    );
    const tmax = 3 * hd;
    for (let iy = 0; iy < res; iy++) {
      const sy = (((iy + 0.5) / res) * 2 - 1) * hv;
      for (let ix = 0; ix < res; ix++) {
        const sx = (((ix + 0.5) / res) * 2 - 1) * hu;
        const ox = oc[0] + rgt[0] * sx + up[0] * sy;
        const oy = oc[1] + rgt[1] * sx + up[1] * sy;
        const oz = oc[2] + rgt[2] * sx + up[2] * sy;
        // Same volume clip the capture uses (#450) — a ray that never enters
        // the captured volume tells us nothing about the surface inside it.
        const span = volRayInterval(
          frame,
          [ox - center[0], oy - center[1], oz - center[2]],
          d,
        );
        if (!span) continue;
        let t = Math.max(0, span[0]);
        const tEnd = Math.min(tmax, span[1]);
        if (tEnd <= t) continue;
        rays++;
        let lastD = Infinity,
          step = 0;
        for (; step < maxSteps && t < tEnd; step++) {
          lastD = Math.abs(de(ox + d[0] * t, oy + d[1] * t, oz + d[2] * t));
          if (lastD < eps) break; // a clean hit — this DE converges here
          t += Math.max(lastD * deScale, 0.5 * eps);
        }
        // Budget gone with the ray still pressed against geometry: the DE could
        // not close the last `lastD`, so `lastD` is a sample of its floor.
        if (step >= maxSteps && lastD < hug) stuck.push(lastD);
      }
    }
  }
  if (!rays || stuck.length < EPS_FLOOR_MIN_RAYS * rays)
    return { floor: 0, stuck: stuck.length, rays };
  stuck.sort((a, b) => a - b);
  return { floor: stuck[stuck.length >> 1], stuck: stuck.length, rays };
}

// Stamp the measured epsilon onto a settled frame. Returns the SAME object when
// the DE converges (so every converging formula stays byte-identical), a copy
// carrying `.epsMeasured` when it does not. Every tier reads it back through
// captureEps(frame) — including the WGSL pass, which takes eps as a uniform, so
// the GPU tier is fixed by this without touching a line of shader code.
//
// The comparison against captureEps(frame) is what keeps this a no-op when some
// OTHER term already dominates — e.g. a small crop whose #496 epsFloor is
// already above 3×floor. captureEps takes the max regardless, so the stamp can
// never lower the eps even if this guard were removed; the guard exists to keep
// the frame object identical in that case, not to enforce the invariant.
export function withCaptureEps(de, frame, opts = {}) {
  const { floor } = deConvergenceFloor(de, frame, opts);
  const epsMeasured = EPS_FLOOR_FACTOR * floor;
  return epsMeasured > captureEps(frame) ? { ...frame, epsMeasured } : frame;
}

// One view's orthographic ray grid — the Worker unit of work. Marches res×res
// first-hit rays along +view from a plane behind the frame, appending each hit's
// (pos, ∇DE normal, albedo) to `out` ({pos:[],normal:[],albedo:[]}). Albedo is
// computed in the FRACTBOX frame (before the §1.3 mirror). Returns the hit count.
// §S1c depth-peel: opts.layers (default 1) peels that many surfaces per ray so
// concave pockets/back-faces fill; §S1d: opts.aoStrength (default 0) bakes AO.
export function captureView(de, albedoAt, frame, view, res, out, opts = {}) {
  const { radius, center } = frame;
  const eps = captureEps(frame); // scale-relative ∨ #496 floor ∨ #507 measured
  const maxSteps = opts.maxSteps ?? 200;
  // March-step scale (§S1a Gap 2): carve scenes / loose analytic DEs over-
  // estimate distance, so the live renderer shortens the step — capture must
  // match or it steps over surfaces. 1 = tight (flat non-loose, S0-verified).
  const deScale = opts.deScale ?? 1;
  const layers = Math.max(1, opts.layers ?? 1);
  const aoStrength = opts.aoStrength ?? 0;
  const h = 2 * eps;

  const { d, rgt, up, oc, hu, hv, hd } = viewBasis(view, frame);
  const tmax = 3 * hd;

  let hits = 0;
  for (let iy = 0; iy < res; iy++) {
    const sy = (((iy + 0.5) / res) * 2 - 1) * hv;
    for (let ix = 0; ix < res; ix++) {
      const sx = (((ix + 0.5) / res) * 2 - 1) * hu;
      const ox = oc[0] + rgt[0] * sx + up[0] * sy;
      const oy = oc[1] + rgt[1] * sx + up[1] * sy;
      const oz = oc[2] + rgt[2] * sx + up[2] * sy;
      // Clip the ray to the volume it is sampling (#450). The volume is convex,
      // so this is one interval — everything before it is material the user did
      // not frame, and marching it only burns budget (see volRayInterval).
      const span = volRayInterval(
        frame,
        [ox - center[0], oy - center[1], oz - center[2]],
        d,
      );
      if (!span) continue; // this ray misses the volume entirely
      const tEnter = Math.max(0, span[0]);
      const tEnd = Math.min(tmax, span[1]);
      if (tEnd <= tEnter) continue;
      // Depth-peel: march to a hit, record it, nudge past the surface, repeat.
      // Marching by the UNSIGNED distance |dd| (sphere-tracing) is what lets a
      // ray traverse a solid interior to reach the next (back / pocket) surface:
      // a signed march stalls at min-step once dd<0 and never crosses a thick
      // solid within budget. |dd| takes geometric steps and lands ON the next
      // surface. For layer 0 (rays start OUTSIDE, dd>0) |dd|≡dd, so layers=1 is
      // byte-identical to S0. One shared budget bounds a ray to layers·maxSteps
      // march steps (Test: grazing budget bound).
      let t = tEnter,
        budget = maxSteps * layers;
      for (let layer = 0; layer < layers; layer++) {
        let hit = false,
          px = 0,
          py = 0,
          pz = 0;
        while (budget > 0 && t < tEnd) {
          budget--;
          px = ox + d[0] * t;
          py = oy + d[1] * t;
          pz = oz + d[2] * t;
          const dd = de(px, py, pz);
          if (Math.abs(dd) < eps) {
            // The ray is clipped to the volume, so this is a formality: only f32
            // slop at the very boundary can land a hit outside. Keep it as the
            // guard (a stray hit stays transparent — re-arm and keep marching in
            // the SAME layer, never consuming a peel layer) and keep it mirrored
            // verbatim by the WGSL fragment (shader.js).
            if (
              volInside(frame, px - center[0], py - center[1], pz - center[2])
            ) {
              hit = true;
              break;
            }
            t += 3 * eps;
            continue;
          }
          t += Math.max(Math.abs(dd) * deScale, 0.5 * eps);
        }
        if (!hit) break; // miss / budget or tmax exhausted → ray done
        // ∇DE central differences
        const gx = de(px + h, py, pz) - de(px - h, py, pz);
        const gy = de(px, py + h, pz) - de(px, py - h, pz);
        const gz = de(px, py, pz + h) - de(px, py, pz - h);
        const gl = Math.hypot(gx, gy, gz);
        if (gl >= 1e-12) {
          // record (a degenerate normal, §5.3, skips recording but still peels)
          const nX = gx / gl,
            nY = gy / gl,
            nZ = gz / gl;
          const ao =
            aoStrength > 0
              ? aoScale(de, px, py, pz, nX, nY, nZ, eps, radius, aoStrength)
              : 1;
          const alb = albedoAt(px, py, pz, nZ);
          out.pos.push(px, py, pz);
          out.normal.push(nX, nY, nZ);
          out.albedo.push(alb[0] * ao, alb[1] * ao, alb[2] * ao);
          hits++;
        }
        if (layer === layers - 1) break; // last layer — no need to re-arm
        // Nudge past the just-hit surface band so the next |dd| march doesn't
        // re-detect it. 3·eps is small (eps-scale) so it only merges features
        // closer than ~3·eps apart — the documented thin-feature floor.
        t += 3 * eps;
      }
    }
  }
  return hits;
}

// S-2 snap-to-surface refine (SPLAT_SHARPNESS §S-2) — the analytic core of the
// sharpness plan. The reduce averages each cell's hits, which pulls centers
// OFF-surface (inside concave patches; at an edge, two faces' hits average to a
// point inside the corner — exactly where fuzz is most visible). Trained 3DGS
// fixes this by gradient descent; we fix it with the thing training has to
// discover: the exact SDF. Per survivor:
//   • 2 damped Newton steps, gradient RECOMPUTED each step (at edges the
//     gradient turns fastest — a stale n̂ is exactly wrong there):
//     p −= deScale·DE(p)·n̂(p), n̂ = the SAME 6-tap central-difference gradient
//     captureView uses (h = 2·eps — CPU-parity pin, UE_SPLAT_S2_IMPL §Normal),
//     deScale = deScaleFor(formula) (the march's own approx-DE damping — no
//     new heuristic).
//   • Reject when total displacement exceeds `cell` (the reduce grid pitch):
//     a snap can't legally move a splat out of the cell its hits came from —
//     a bigger move means a pathological/approx DE, keep the average.
//   • On accept: normal := ∇DE(p′) (n̂ from the FINAL iteration's gradient),
//     albedo := albedoAt(p′)·aoScale(p′) — the exact point sample, mirroring
//     captureView's per-hit convention (AO stays baked, same aoStrength).
// Scope pin (spec review): positions/normals/albedos ONLY — per-splat
// radius/alpha/aniso from finalizeReduce are NOT recomputed (displacement is
// ≤ cell; re-running the CSR passes is a measured-not-assumed follow-up).
// Mutates `points` in place (worker-transfer friendly); returns stats.
export function snapPoints(points, de, albedoAt, opts = {}) {
  const {
    cell, // required: displacement bound (reduce grid pitch)
    eps, // required: the capture's own march tolerance (h/AO parity)
    // Convergence target. The capture's eps IS the raw hits' position noise (a
    // ray stops anywhere inside the |DE|<eps band — a speed compromise for
    // millions of rays). The snap refines only the ~1M survivors, so it can
    // afford 10× tighter: THIS is where the sharpening comes from — position
    // noise drops from ±eps to ±eps/10 (sub-cell).
    tol = (eps ?? 1e-4) * 0.1,
    radius = 1, // frame radius — aoScale's occlusion probe scale
    deScale = 1, // per-formula Newton damping (deScaleFor)
    aoStrength = 0.5, // capture parity: 0 disables the AO resample factor
    iterations = 8, // budget, not target: converges (|DE|<tol) in 2-3 for tight DEs
    // POSITION-ONLY by default (measured, Tourbillon 40v/192²). The reduce's
    // hit-AVERAGED normal is the true footprint average of ~n exact per-hit
    // gradients — the physically right disc orientation — and its averaged
    // albedo is anti-aliased color. Point-resampling both at the snapped
    // center replaces area averages with single aliased samples: normalAgree
    // cratered (0.97→0.14 micro-h, 0.57 footprint-h — finite differences
    // ALIAS on striated surfaces at any single h). resample:true keeps the
    // ∇DE-normal + albedo·AO experiment available for smooth formulas.
    resample = false,
    onProgress, // optional (done, total) → false aborts (worker cancel)
  } = opts;
  const n = points.count;
  const P = points.pos,
    N = points.normal,
    A = points.albedo;
  // TWO gradient scales (both measured on Tourbillon, 40v/192²):
  // • STEPPING h = 2·tol — the march's h=2·eps reaches a large fraction of a
  //   cell on a generous frame, so central differences span multiple fine
  //   features and the slope turns garbage → wild steps → 17.7% rejections;
  //   at 2·tol rejections drop to 0.4%. (f64 DE — no cancellation cost.)
  // • NORMAL hN = cell/2 — the disc's own footprint scale. A micro-scale
  //   normal (2·tol) reflects striations far smaller than the disc it
  //   orients: adjacent discs tilt chaotically and normalAgree craters
  //   (0.97 → 0.14 measured). The disc must be oriented by the surface it
  //   physically covers, so its normal is the gradient smoothed over ~its
  //   radius — which also matches the capture-normal scale (2·eps ≈ cell/2
  //   at typical density).
  const h = 2 * tol;
  const hN = 0.5 * cell;
  const maxD2 = cell * cell;
  let snapped = 0,
    rejected = 0,
    degenerate = 0,
    sumMove = 0,
    sumAbsBefore = 0,
    sumAbsAfter = 0;
  for (let i = 0; i < n; i++) {
    if (onProgress && (i & 0x3fff) === 0 && onProgress(i, n) === false)
      return null; // cancelled — caller keeps the un-refined points
    const j = 3 * i;
    const x0 = P[j],
      y0 = P[j + 1],
      z0 = P[j + 2];
    const d00 = de(x0, y0, z0);
    sumAbsBefore += Math.abs(d00);
    // DIRECTIONAL 1-D Newton ALONG THE STORED NORMAL, not a free 3-D descent.
    // The stored normal is the reduce's hit-average — it knows WHICH surface
    // this cell belongs to. A free gradient descent snaps to whichever zero
    // crossing is NEAREST, and on a thinner-than-cell wall (Menger everywhere)
    // that is often the OPPOSITE face — the splat lands with its normal
    // anti-aligned to the surface it sits on (measured: normalAgree −0.71).
    // Constraining the search to the normal's own line lands on the face the
    // normal belongs to; for ordinary on-face cells the gradient is parallel
    // to the normal anyway, so the paths coincide. Bonus: 3 DE evals per step
    // instead of 7. The directional slope is measured (not assumed 1), which
    // is the loose-DE slope correction; deScale damps approx-DE steps.
    const nx0 = N[j],
      ny0 = N[j + 1],
      nz0 = N[j + 2];
    let t = 0,
      ok = true,
      converged = false;
    for (let it = 0; it < iterations; it++) {
      const d0 = it === 0 ? d00 : de(x0 + t * nx0, y0 + t * ny0, z0 + t * nz0);
      if (Math.abs(d0) < tol) {
        converged = true;
        break;
      }
      const dp = de(x0 + (t + h) * nx0, y0 + (t + h) * ny0, z0 + (t + h) * nz0);
      const dm = de(x0 + (t - h) * nx0, y0 + (t - h) * ny0, z0 + (t - h) * nz0);
      const slope = (dp - dm) / (2 * h);
      if (Math.abs(slope) < 1e-9) {
        ok = false; // surface ∥ the normal line here — no reachable crossing
        break;
      }
      let mv = (-deScale * d0) / slope;
      if (Math.abs(mv) > cell) mv = Math.sign(mv) * cell; // step clamp
      t += mv;
      if (Math.abs(t) > cell) {
        ok = false; // walked out of the cell — the line missed the surface
        break;
      }
    }
    if (!ok) {
      degenerate++;
      continue;
    }
    const x = x0 + t * nx0,
      y = y0 + t * ny0,
      z = z0 + t * nz0;
    // Accept only a real improvement: converged, or measurably closer to the
    // surface than the average was (the crease tail improves without reaching
    // tol; anything else keeps the average).
    if (!converged && Math.abs(de(x, y, z)) >= Math.abs(d00)) {
      rejected++;
      continue;
    }
    const d2 = t * t;
    if (d2 > maxD2) {
      rejected++; // moved out of its own cell — pathological, keep the average
      continue;
    }
    if (resample) {
      // Opt-in: ∇DE normal at the disc-footprint scale + exact albedo·AO at
      // the snapped point (see the resample note above for why this is not
      // the default).
      const gx = de(x + hN, y, z) - de(x - hN, y, z);
      const gy = de(x, y + hN, z) - de(x, y - hN, z);
      const gz = de(x, y, z + hN) - de(x, y, z - hN);
      const gl = Math.hypot(gx, gy, gz);
      if (gl < 1e-12) {
        degenerate++;
        continue;
      }
      const nX = gx / gl,
        nY = gy / gl,
        nZ = gz / gl;
      N[j] = nX;
      N[j + 1] = nY;
      N[j + 2] = nZ;
      const ao =
        aoStrength > 0
          ? aoScale(de, x, y, z, nX, nY, nZ, eps, radius, aoStrength)
          : 1;
      const alb = albedoAt(x, y, z, nZ);
      A[j] = alb[0] * ao;
      A[j + 1] = alb[1] * ao;
      A[j + 2] = alb[2] * ao;
    }
    P[j] = x;
    P[j + 1] = y;
    P[j + 2] = z;
    snapped++;
    sumMove += Math.sqrt(d2);
    sumAbsAfter += Math.abs(de(x, y, z));
  }
  const den = snapped || 1;
  return {
    snapped,
    rejected,
    degenerate,
    avgMove: sumMove / den,
    meanAbsBefore: sumAbsBefore / (n || 1),
    meanAbsAfter: sumAbsAfter / den,
  };
}

// S-3 analytic densification (SPLAT_SHARPNESS §S-3) — what 3DGS training does by
// cloning splats where the image error is high, we do where the GEOMETRY says
// detail lives: edge cells (dispersion < τ). Each edge splat spawns 2 children
// offset ±cell/4 ALONG its edge axis (points.dir — the Pass-1c crease line, so
// children start ≈ on-surface) and Newton-snapped exactly onto it; a child whose
// snap leaves it > 3·eps off-surface is dropped (a real edge END, not a line).
// Children inherit the parent's area-averaged normal/albedo/alpha (the S-2
// position-only lesson) and 0.7× its radii — densifying without shrinking would
// only inflate overdraw. Parents keep their radius (coverage safety).
// Budget discipline (spec review): the caller passes the REMAINING headroom
// (min(0.25·cap, cap − count)); worst-dispersion parents first; zero silent
// truncation — stats report parents/added/dropped.
// Requires points.dispersion + points.dir (aniso path) — returns null when
// either is absent or the budget is 0 (caller keeps the original points).
export function densifySplats(points, de, opts = {}) {
  const {
    cell,
    eps,
    tol = (eps ?? 1e-4) * 0.1,
    deScale = 1,
    budget = 0,
    edgeTau = 0.85,
    iterations = 8,
  } = opts;
  const n = points.count;
  const disp = points.dispersion,
    dirA = points.dir;
  if (!disp || !dirA || budget < 2 || !cell) return null;
  // Worst-dispersion edge parents first, as many as the budget seats (2 kids each).
  const parents = [];
  for (let i = 0; i < n; i++) if (disp[i] < edgeTau) parents.push(i);
  if (parents.length === 0) return null;
  parents.sort((a, b) => disp[a] - disp[b]);
  const take = Math.min(parents.length, Math.floor(budget / 2));
  const extra = take * 2;
  const off = cell / 4;
  const h = 2 * tol;
  // Grow every channel the cloud carries (children inherit; snap fixes pos).
  const P = new Float32Array(3 * (n + extra));
  const N = new Float32Array(3 * (n + extra));
  const A = new Float32Array(3 * (n + extra));
  P.set(points.pos);
  N.set(points.normal);
  A.set(points.albedo);
  const R = new Float32Array(n + extra);
  const AL = new Float32Array(n + extra);
  const DI = new Float32Array(n + extra);
  R.set(points.radius);
  AL.set(points.alpha);
  DI.set(disp);
  const R2 = new Float32Array(n + extra);
  const DR = new Float32Array(3 * (n + extra));
  R2.set(points.r2);
  DR.set(dirA);
  let m = n,
    added = 0,
    droppedChildren = 0;
  for (let t = 0; t < take; t++) {
    const i = parents[t];
    const j = 3 * i;
    for (let sgn = -1; sgn <= 1; sgn += 2) {
      const cx = P[j] + sgn * off * DR[j],
        cy = P[j + 1] + sgn * off * DR[j + 1],
        cz = P[j + 2] + sgn * off * DR[j + 2];
      // Directional 1-D Newton along the PARENT's normal (same rationale as
      // snapPoints: the normal knows which face this is — a free descent can
      // land a thin-wall child on the opposite face). The child starts near
      // the surface (offset along the crease TANGENT), so t stays small.
      const nx = N[j],
        ny = N[j + 1],
        nz = N[j + 2];
      let t = 0;
      for (let it = 0; it < iterations; it++) {
        const d0 = de(cx + t * nx, cy + t * ny, cz + t * nz);
        if (Math.abs(d0) < tol) break;
        const dp = de(cx + (t + h) * nx, cy + (t + h) * ny, cz + (t + h) * nz);
        const dm = de(cx + (t - h) * nx, cy + (t - h) * ny, cz + (t - h) * nz);
        const slope = (dp - dm) / (2 * h);
        if (Math.abs(slope) < 1e-9) break;
        let mv = (-deScale * d0) / slope;
        if (Math.abs(mv) > cell) mv = Math.sign(mv) * cell;
        t += mv;
        if (Math.abs(t) > cell) break;
      }
      const x = cx + t * nx,
        y = cy + t * ny,
        z = cz + t * nz;
      if (Math.abs(t) > cell || Math.abs(de(x, y, z)) > 3 * eps) {
        droppedChildren++; // off the edge's end — no surface there
        continue;
      }
      const jm = 3 * m;
      P[jm] = x;
      P[jm + 1] = y;
      P[jm + 2] = z;
      N[jm] = N[j];
      N[jm + 1] = N[j + 1];
      N[jm + 2] = N[j + 2];
      A[jm] = A[j];
      A[jm + 1] = A[j + 1];
      A[jm + 2] = A[j + 2];
      R[m] = R[i] * 0.7;
      AL[m] = AL[i];
      DI[m] = DI[i];
      R2[m] = R2[i] * 0.7;
      DR[jm] = DR[j];
      DR[jm + 1] = DR[j + 1];
      DR[jm + 2] = DR[j + 2];
      m++;
      added++;
    }
  }
  if (added === 0) return null;
  // Trim to the actual count (dropped children leave a tail).
  const cut3 = (arr) => (m + 0 < n + extra ? arr.subarray(0, 3 * m) : arr);
  const cut1 = (arr) => (m + 0 < n + extra ? arr.subarray(0, m) : arr);
  points.count = m;
  points.pos = cut3(P);
  points.normal = cut3(N);
  points.albedo = cut3(A);
  points.radius = cut1(R);
  points.alpha = cut1(AL);
  points.dispersion = cut1(DI);
  points.r2 = cut1(R2);
  points.dir = cut3(DR);
  return { parents: take, added, droppedChildren };
}

// The saved-camera framing, as a frame object — the degenerate fallback when a
// surface probe finds nothing (§3 step 2) AND the `refineFrame` fallback for a
// ballooned scene frame. Factored so captureSplats and app/src/splatexport.ts
// build it identically (they must not drift).
export function cameraFrame(formula) {
  // The saved-camera FALLBACK frame (no-surface / ballooned-scene). Deliberately
  // the loose dist/3 box — NOT viewFrame's aspect-tight sizing: it's an
  // oversized safety net, and tightening it would shift every fall-back capture
  // (e.g. Gnarl Dunes, whose frameFormula probe returns null). viewFrame (S-5a)
  // is the separate, aspect-aware framer for the deliberate zoomed-VIEW path.
  const cam = makeCamera(formula.camera);
  const radius = cam.dist / 3;
  return {
    center: cam.target.slice(),
    ext: [radius, radius, radius],
    radius,
    diag: 2 * radius,
  };
}

// A frame covering what the camera SEES (SPLAT_VIEW_CAPTURE / S-5a) — center at
// the look-at target, radius sized to the on-screen extent at that plane. Zoomed
// in (small dist) ⇒ small frame ⇒ the fixed splat budget lands over the region
// ⇒ fine pitch (r0 ∝ frame.diag) for close-up detail — the lever that isn't
// bounded by the splat count / accumulator memory.
//   • `cam`: the live camObj() ({ dist, target, fovDeg, … }) or a makeCamera().
//   • radius = dist · tan(fov/2) · max(aspect,1) · margin. fovDeg is the VERTICAL
//     fov (preview.js pixelRay scales the HORIZONTAL half-angle by aspect), so we
//     size to the WIDER screen axis + a margin — else content near the left/right
//     edges is silently cropped (spec-review blocker). No fovDeg ⇒ the dist/3
//     legacy approximation (cameraFrame's whole-object fallback, grossly oversized
//     anyway). aspect defaults 1 (the CLI/saved-camera path passes none).
//   • radius floored at 1e-4: cam.dist clamps to 1e-9 at max zoom, and a sub-
//     nano radius underflows r0/eps → NaN grid indices in the reduce.
export function viewFrame(cam, opts = {}) {
  const dist = Math.max(cam.dist ?? 24, 1e-6);
  const halfV = cam.fovDeg
    ? dist * Math.tan(((cam.fovDeg * Math.PI) / 180) * 0.5)
    : dist / 3;
  const aspect = Math.max(opts.aspect ?? 1, 1);
  const margin = opts.margin ?? 1.1;
  const radius = Math.max(halfV * aspect * margin, 1e-4);
  const target = (cam.target ?? [0, 0, 0]).slice();
  return {
    center: target,
    ext: [radius, radius, radius],
    radius,
    diag: 2 * radius,
  };
}

// The `epsFloor` a SUB-REGION capture of `formula` must respect (#496): the hit
// eps the WHOLE-OBJECT capture of the same formula would actually march with.
// See captureEps for why an absolute floor is the right shape of fix.
//
// BOTH terms of captureEps that a whole-object frame would see are included —
// this is the #496/#507 reconciliation, and it is not optional. #496 shipped
// this as `3e-4 · objectRadius` alone, which WAS the object's eps at the time.
// #507 then raised the object's eps to the measured convergence floor whenever
// the DE bottoms out, and a crop inheriting only the scale-relative term is
// suddenly no longer inheriting "the eps the object used" — the promise this
// function exists to keep. Measured on the Menger after the two merged: the
// object marches at 1.41e-2 (3 × its 4.7e-3 floor) while this returned 3.3e-4,
// 43× tighter, and the 1/8.7 corner crop came back with 365 splats and visible
// #507 confetti (30.7% axis-aligned normals) — i.e. BOTH bugs returned in the
// one case that trips both. Deriving the floor through withCaptureEps/captureEps
// rather than re-typing the rule is what stops that drifting apart again.
//
// Still CONSERVATIVE in the two ways #496 documented:
//   • it is the UN-grown object radius, so the floor is never looser than the
//     eps the object path would actually march with (that path may grow the
//     radius further for #351);
//   • when the probe measures nothing (it returns null for the leaf-shape and
//     numeric-deOption stacks — #457) there is no cheap honest answer, so the
//     floor is 0 (inert) and the crop keeps today's behavior rather than
//     inheriting `cameraFrame`'s grossly oversized dist/3 box as a "floor",
//     which would loosen eps on formulas that never needed it.
//
// KNOWN LIMIT (not introduced here — exposed by it). A crop inheriting a large
// measured floor can be reduced to a pitch FINER than the eps its hits were
// found at: computeR0 is `radiusScale · diag / √hits`, with no eps term. On the
// Menger's 1/8.7 corner box, 16 views: 19.5k splats at r0 = 4.0e-3 against
// eps = 1.41e-2, and metricOnSurface — which asks whether |DE| is small next to
// a splat — falls to 0.098, because for a floored DE |DE| ≥ 4.7e-3 everywhere
// near the surface and no cloud can beat that at a 4.0e-3 pitch. Color and
// normals are unaffected and in fact much improved (colorDrift 1.88e-2 → 4.4e-4,
// axis-aligned normals 30.7% → 69.5%), so the export is better than before on
// every axis the two issues were about; it is the position CONFIDENCE that a
// sub-floor pitch cannot establish. Flooring r0 against eps is the principled
// follow-up and is deliberately NOT done here: r0 feeds every splat metric and
// a change moves dozens of unrelated baselines.
//
// Cost: `frameFormula`'s orbit probe (~30 ms) plus one deConvergenceFloor probe
// — measured 0.6 s on the Menger, 0.12–0.17 s on formulas that converge (where
// it returns 0). That is new next to #496's ~30 ms, and it buys the only signal
// that makes a crop of a floored DE usable at all; the seconds-long part of
// whole-object framing (refineFrame/growFrameToSurface) is still skipped. Pass
// `frame`/`de`/`deScale` when the caller already has them (the viewCam path in
// kit/splatexport.ts does) so only the probe is paid for.
export function objectEpsFloor(formula, opts = {}) {
  const f = opts.frame ?? frameFormula(formula);
  if (!f) return 0;
  const de = opts.de ?? makeDE(formula, opts.iters ?? formula.iters ?? 8);
  const deScale = opts.deScale ?? deScaleFor(formula);
  return captureEps(withCaptureEps(de, f, { deScale }));
}

// The iteration count a SUB-REGION capture must march at (#496, round 5).
//
// Framing a crop is a ZOOM, and renderpolicy.js's auto-detail law already says
// what a zoom costs in iterations: "a distance-estimated fractal has a FIXED
// finest scale for a given iteration count, so zooming past it just smooths out
// (the DE becomes a bound, not the surface)" — one extra iteration per octave.
// The live render has obeyed that law since #181; the SPLAT EXPORT obeyed it too
// (`iters: views.effectiveIters()`) until #415 removed the export from the app
// and took the wiring with it, leaving `app/src/views.ts effectiveIters()` a
// dead export. Since then every capture — whole object, S-5a view frame, or a
// user-drawn capture volume — has marched at the formula's flat base count.
//
// That is the unfixed half of #496. Rounds 1-4 all corrected the hit EPSILON so
// a crop would register hits at all, and they worked: the splat COUNT recovered.
// But eps was the only lever, and loosening it buys hits at the cost of where
// those hits are. Measured on the reporter's Menger corner box (16 views × 64²,
// layers 3), against core/splatmetrics.js metricOnSurface:
//
//   framing            iters   splats   eps/radius   onSurface
//   whole object         5     150062     1.28e-2      0.977
//   crop ×1/8.7          5     131180     1.23e-1      0.010   ← before
//   crop ×1/8.7          8     116267     2.87e-3      1.000   ← after
//
// i.e. the crop's count was already fine and 99% of it was NOT ON THE SURFACE —
// a shell 12% as thick as the box, which is the reporter's "colors are still
// smudged" and "too many points are gone and merged". The Menger's measured
// convergence floor tracks 3^-iters almost exactly (iters 5 → 4.7e-3, 6 →
// 1.8e-3, 7 → 5.3e-4, 8 → 1.9e-4), so the floor a crop inherits is not a
// property of "the field" at all — it is a property of the field AT THE OBJECT'S
// ITERATION COUNT. Raise the count and the floor falls, eps can be honest again,
// and the crop finally resolves 4.5× FINER than the whole object rather than 9.6×
// coarser. That is what a close-up is for, and it is the reporter's own question
// ("if they have same limitation, then this is pointless") answered.
//
// INERT unless the frame is a strict sub-region: `mag <= 1` returns `base`, so a
// whole-object export and #518's oversized box are untouched (verified for
// Menger / Mandelbox / Sierpinski / Cube Cluster). DEs that already converge are
// unaffected in fidelity — they were at onSurface 1.000 before and after — they
// just gain the finer structure the extra iterations expose.
//
// Cost is linear in iters (renderpolicy.js formulaCostScore), and it is the same
// cost the LIVE VIEW already pays at that zoom, so it buys detail the user has
// already seen on screen. Bounded by ITER_CEIL. NOTE the caller must recompute
// `epsFloor` (objectEpsFloor) at the returned count — the floor is the eps the
// object would use in the field this capture actually marches, and inheriting a
// base-iters floor pins eps right back where it was (measured: onSurface reaches
// only 0.65 if the count rises but the floor does not).
export function cropCaptureIters(base, objFrame, cropFrame) {
  if (!objFrame || !cropFrame) return base;
  const ro = objFrame.radius,
    rc = cropFrame.radius;
  if (!(ro > 0) || !(rc > 0) || rc >= ro) return base; // not a sub-region ⇒ inert
  return itersForMagnification(base, ro / rc);
}

// The march-step scale for this formula (§S1a Gap 2), matching the live
// renderer: scenes → sceneDeScale (0.5 base / 0.25 carving / ×approx); flat or
// hybrid LOOSE analytic DE → 0.3 (the same tighter step preview applies); tight
// flat/hybrid → 1 (S0-verified). Threaded into captureView as opts.deScale.
export function deScaleFor(formula) {
  if (Array.isArray(formula.objects) && formula.objects.length)
    return sceneDeScale(formula.objects);
  const loose = formula.hybrid ? hybridLooseDE(formula) : looseDE(formula);
  return loose ? 0.3 : 1;
}

// Refine a provisional frame against the REAL capture DE (§S1a Gap 1): the
// evaluate.js probe drops leaf shapeIds and caps plane reach at `region`, so a
// leaf-under-ops or plane-bearing scene frames wrong or balloons to the R=40
// cap. Re-frame by marching the 6 axis directions over a box 1.5× the
// provisional radius with the EXACT capture march (incl. deScale), and take the
// hit bbox. `< minHits` total ⇒ the surface isn't where the probe thought —
// fall back to `camFrame` (the human-vetted saved camera), never the ballooned
// input. Cost ≈ 6·grid²·maxSteps DE evals — noise next to capture.
export function refineFrame(de, frame, camFrame, opts = {}) {
  const grid = opts.grid ?? 24;
  const minHits = opts.minHits ?? 8;
  const deScale = opts.deScale ?? 1;
  const stubAlbedo = () => [0, 0, 0]; // framing needs geometry only
  // A ballooned provisional frame (a plane/unbounded scene runs R to the 40 cap,
  // evaluate.js:204) would make the probe box huge too — and a plane fills any
  // box, so refining it wouldn't bound anything. Cap the probe basis by the
  // saved-camera frame so we re-frame near the human-vetted view, not the
  // balloon; a well-bounded provisional frame is trusted as-is (it's tighter).
  const basis = frame.radius > 3 * camFrame.radius ? camFrame : frame;
  const probe = { center: basis.center, radius: 1.5 * basis.radius };
  const out = { pos: [], normal: [], albedo: [] };
  const axes = [
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
  ];
  for (const v of axes)
    captureView(de, stubAlbedo, probe, v, grid, out, {
      maxSteps: 200,
      deScale,
    });

  const hits = out.pos.length / 3;
  if (hits < minHits) return camFrame;
  let lo = [Infinity, Infinity, Infinity];
  let hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < hits; i++) {
    for (let c = 0; c < 3; c++) {
      const p = out.pos[3 * i + c];
      if (p < lo[c]) lo[c] = p;
      if (p > hi[c]) hi[c] = p;
    }
  }
  const ext = [(hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2, (hi[2] - lo[2]) / 2];
  const maxExt = Math.max(ext[0], ext[1], ext[2]);
  if (!(maxExt > 0)) return camFrame; // all hits coincident — degenerate
  // Bare fit, for the same measured reason as frameFormula above — `ext` is the
  // capture volume and padding it per-axis cost quality. radius keeps the 1.10.
  return {
    center: [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2],
    ext,
    radius: 1.1 * maxExt,
    diag: 2 * len3(ext),
  };
}

// Grow-on-miss whole-object frame (#351 "No surface found in the current
// framing"): a last-resort safety net for when frameFormula/cameraFrame (and,
// for scenes, refineFrame) STILL hand back a frame that doesn't contain the
// real surface, so the real capture below would burn its whole budget on
// every ray and report zero hits. Root cause measured on issue #351's
// "Drifting Sponge Octacale" (steep ScaleDrift decay stacked with
// kaleido/octaFold): captureView's hit epsilon is `3e-4 · frame.radius` — it
// SHRINKS with a smaller (more tightly-fitted, i.e. conventionally "better")
// frame. For an ordinary formula that's a harmless resolution knob, but a
// severely loose analytic DE (r/|w| with |w| decaying toward a small residual
// under compounding ScaleDrift) never converges much below a fixed absolute
// floor near the true surface — so a small, tightly-fitted frame sets an eps
// BELOW that floor and the march reports zero hits everywhere, while the
// SAME geometry captures at a healthy, stable hit rate once the frame is
// large enough to loosen eps past the floor (measured: 0% at radius ≤3,
// ~11% at radius ≥4.2, stable across 16-64 views). So: probe the CURRENT
// frame with a cheap multi-view sample and, if the hit count is too thin to
// trust, double the radius and retry (up to `maxGrow` times) — and critically
// keep the radius that worked rather than re-fitting a tighter box around
// the found points, which would shrink eps right back below the floor and
// silently reproduce the bug. A frame that already contains the surface pays
// for exactly one cheap probe and returns unchanged.
// The cheap multi-view probe growFrameToSurface grows against: "how much
// surface does THIS frame see?". Exported because a caller can need the answer
// without the growing — the export's zoomed VIEW frame is chosen by the camera,
// not fitted to geometry, so it can sit in empty space and must be checked
// before it's captured (#438).
export function probeFrameHits(de, frame, opts = {}) {
  const views = opts.probeViews ?? 16;
  const res = opts.probeRes ?? 32;
  const deScale = opts.deScale ?? 1;
  const stubAlbedo = () => [0, 0, 0]; // framing needs geometry only
  const out = { pos: [], normal: [], albedo: [] };
  for (let k = 0; k < views; k++)
    captureView(de, stubAlbedo, frame, fibonacciDir(k, views), res, out, {
      maxSteps: 200,
      deScale,
    });
  return out.pos.length / 3;
}

export function growFrameToSurface(de, frame, opts = {}) {
  const maxGrow = opts.maxGrow ?? 6; // 64× up from the input radius
  const minHits = opts.minHits ?? 16;

  // Already contains surface: one cheap probe and the SAME REFERENCE back (the
  // common case, and callers pin the identity).
  if (probeFrameHits(de, frame, opts) >= minHits) return frame;

  // #518 — loosen eps WITHOUT enlarging the capture volume, first.
  //
  // This function grows the frame for one reason: `eps = 3e-4 · frame.radius`,
  // and a loose analytic DE needs that above its convergence floor before any
  // ray registers a hit (see the header). `radius` is the eps scalar; `ext` is
  // the sampled VOLUME. Growing them together made sense when they were the
  // same number, but since CAPTURE_VOLUME_SHAPES they are not, and inflating
  // the volume to buy a bigger eps is enormously expensive: the splat count
  // falls with the FOURTH power of the volume's linear size (see capturedDiag),
  // so the ×4 growth #351's own formula needs cost ~45× the exported splats
  // (measured: 1481 → 67376 on 'Drifting Sponge Octacale' at 16 views × 64²).
  //
  // growEpsToSurface (#496) is exactly this loop with `ext`/`center` pinned, so
  // the eps-only attempt is a call to it, with growFrameToSurface's own
  // absolute minHits bar rather than #496's rate-derived one (this caller
  // probes at the default density, where the two agree by construction).
  const epsGrown = growEpsToSurface(de, frame, { ...opts, minHits, maxGrow });
  if (probeFrameHits(de, epsGrown, opts) >= minHits) return epsGrown;

  // Still nothing: the surface is genuinely OUTSIDE the volume (frameFormula
  // under-measured, or a probe literal carries a radius with no geometry around
  // it) — no epsilon can find what is not being sampled. Fall back to today's
  // whole-frame growth, which is the only thing that reaches it.
  //
  // Grow by a SCALE factor, keeping the frame's aspect: this runs on every
  // capture, so rebuilding a uniform ext here would silently turn a cuboid the
  // user drew (or frameFormula fitted) back into a cube.
  const ext0 = volExt(frame);
  const diag0 = frame.diag ?? 2 * len3(ext0); // probe literals carry no diag
  for (let g = 1, k = 2; ; g++, k *= 2) {
    const grown = {
      ...frame,
      ext: [ext0[0] * k, ext0[1] * k, ext0[2] * k],
      radius: frame.radius * k,
      diag: diag0 * k,
    };
    if (probeFrameHits(de, grown, opts) >= minHits) return grown;
    if (g === maxGrow) return frame; // exhausted — the original "no surface" path stands
  }
}

// Grow-on-miss for an EXPLICIT (user-placed) capture volume (#496 — a small
// custom box gave 78 splats where the whole object gave thousands). Same
// #351 mechanism as growFrameToSurface: captureView/renderer.js's hit eps is
// `3e-4 · frame.radius`, and a box far smaller than the object (routine —
// zooming a custom volume into local detail) sets that below a loose DE's
// fixed convergence floor, so nearly every ray misses however finely it's
// sampled. growFrameToSurface's fix is to grow `ext` too, which here would
// silently resize the box the user placed — precisely the lie
// kit/splatexport.ts's exportFrame short-circuit exists to prevent (an
// explicit box is "honoured verbatim"). `radius` is architecturally already
// split from the captured geometry (CAPTURE_VOLUME_SHAPES.md: it's the
// eps/AO-probe/r0 SCALE scalar, never the sampled window — that's `ext`), so
// this doubles radius ALONE: same probe-and-double loop, `ext`/`center`
// pinned throughout. A box that already hits `minHits` costs one probe, same
// as growFrameToSurface's already-converged case.
// #496 follow-up: a fixed `minHits` is an absolute count, but the probe's own
// ray budget (`probeViews·probeRes²`) varies by caller — the default (16×32²)
// is a cheap proxy, while kit/splatexport.ts's real capture runs at the job's
// OWN (much denser) views/res. Two failure modes came from that mismatch:
//   - probed at the cheap default: minHits=16 is a 0.1% hit rate there, but
//     clearing it only proves the eps is "not literally zero" — nowhere near
//     what the real, far denser capture needs, so growth kept doubling toward
//     `maxGrow` and the resulting eps could become large enough, relative to
//     a SMALL user-drawn box, to blur past thin/fine detail (many raw hits
//     collapsing into very few reduced survivors — the reporter's "merging
//     way too many points").
//   - probed at the real (dense) capture's own density with the SAME
//     absolute minHits=16: that count is now a vanishingly low bar (checked
//     directly — it converges to an eps so tight the real capture then finds
//     only a couple dozen raw hits total, the ORIGINAL #496 starvation).
// So minHits is a RATE, not a count: it scales with whatever probe density is
// requested, preserving the original 16-of-16384 calibration as the rate
// itself — a caller that doesn't override probe density sees byte-identical
// behavior; a caller that probes at its real capture's density gets a bar
// scaled to match, instead of silently inheriting the cheap default's.
const EPS_PROBE_HIT_RATE = 16 / (16 * 32 * 32);
export function growEpsToSurface(de, frame, opts = {}) {
  const maxGrow = opts.maxGrow ?? 6;
  const probeViews = opts.probeViews ?? 16;
  const probeRes = opts.probeRes ?? 32;
  const minHits =
    opts.minHits ??
    Math.max(
      16,
      Math.round(EPS_PROBE_HIT_RATE * probeViews * probeRes * probeRes),
    );
  const radius0 = frame.radius;
  for (let g = 0, k = 1; ; g++, k *= 2) {
    const grown = k === 1 ? frame : { ...frame, radius: radius0 * k };
    if (probeFrameHits(de, grown, { ...opts, probeViews, probeRes }) >= minHits)
      return grown;
    // Exhausted: a scaled minHits (see above) can set a bar a very sparse/thin
    // feature never clears within maxGrow doublings. Return the MOST-grown
    // attempt, not the pristine input — still ext/center-identical (nothing
    // captured is honoured any less verbatim), but the loosest eps tried is
    // strictly the best chance the caller's own follow-up empty-check has of
    // finding real geometry, instead of re-testing the same too-tight eps that
    // already justified growing in the first place.
    if (g === maxGrow) return grown;
  }
}

// The S0 global splat radius / dedup cell: radiusScale · mean sample spacing.
// Shared by captureSplats and the app so the 1.6 default lives in one place.
export function computeR0(diag, rawHits, radiusScale = 1.6) {
  return (radiusScale * diag) / Math.sqrt(rawHits);
}

// The diagonal computeR0 should measure spacing over: the extent of what the
// capture actually FOUND, never the frame it looked in (#518).
//
// computeR0's `diag/√hits` is "mean sample spacing", and that identity holds
// only while the frame is FITTED to the geometry — which it was, back when
// every frame came from frameFormula/refineFrame. It no longer is: the capture
// volume can legitimately be much larger than what is inside it (a box the user
// drew around a detail, a camera-chosen view frame, a frame grown to loosen
// eps). Empty space in the frame then costs the export QUARTICALLY, because it
// hits both factors at once:
//   • the per-view window is the volume's shadow, so the geometry occupies a
//     smaller fraction of every ray grid  ⇒ rawHits ∝ 1/ext²
//   • diag rises with the box                                ⇒ cell ∝ ext²
// and survivors ≈ area/cell², so survivors ∝ 1/ext⁴. A volume 3.7× the geometry
// (#518's reporter: a 3.27-unit box around a 0.88-unit shape) is a ~190× loss —
// 5845 splats against a 2,500,000 cap.
//
// Measuring the pitch over the HIT bbox instead makes it a property of the
// captured geometry, which is what the formula always meant. It is a no-op for
// a fitted frame (hits fill it, so the two diagonals agree) and only bites when
// the frame holds empty space. Degenerate/absent bbox (no hits yet, a single
// hit, a perfectly planar chunk on the first view) ⇒ the frame's own diagonal,
// so the estimate can never be zero or NaN.
export function capturedDiag(min, max, frameDiag) {
  const d = Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
  return Number.isFinite(d) && d > 0 && d < frameDiag ? d : frameDiag;
}

// Voxel-grid dedup at cell = r0 (average pos/normal/albedo per cell; renormalize
// normal; |n|<1e-6 after averaging ⇒ drop, isotropic fallback is S1). If the
// survivor count exceeds `cap`, LOOP a re-reduce (cell ×= (kept/cap)^(1/3)) up to
// 4 passes, then hard-cap by dropping the lowest-density cells — deterministic,
// never silently exceeds cap.
//
// S1b auto-tune (§S1b): from the survivor grid (≤1 point per occupied cell), a
// two-pass O(N)·27 scan derives a per-splat local radius (nearest-survivor
// spacing → cover sparse pockets) and an overlap-attenuated opacity (thin only
// denser-than-typical pileups). opts knobs: radiusScale, alphaBase, alphaMin,
// attenGamma (0 ⇒ uniform opacity = S0), rClampLo/rClampHi (cell units).
// Returns { points:{count,pos,normal,albedo,radius,alpha}, kept, dropped }.
export function reducePoints(raw, r0, cap, opts = {}) {
  const {
    radiusScale = 1.6,
    alphaBase = 0.95,
    alphaMin = 0.3,
    attenGamma = 0.5,
    rClampLo = 0.5,
    rClampHi = 3,
    // §parallel slab-reduce: [lo, hi] keeps only survivors with x ∈ [lo, hi) in
    // the OUTPUT — the margin survivors (present so boundary radius/alpha are
    // computed against a full neighborhood) are dropped, owned by the adjacent
    // slab. null (default) ⇒ keep all (behavior-identical to non-parallel).
    coreX = null,
    // P2 (Rung 2) anisotropic fit: aniso=0 (default) SKIPS the PCA pass entirely
    // (byte-identical isotropic output — back-compat pin). (0,1] scales elongation
    // toward the clamp. anisoMax = max r_max:r_min; anisoFloor = min curvature-ratio
    // to trust a direction (noise floor); epsFrac → sagitta budget ε = epsFrac·r0.
    aniso = 0,
    anisoMax = 3,
    anisoFloor = 1.5,
    epsFrac = 0.35,
  } = opts;

  let minx = Infinity,
    miny = Infinity,
    minz = Infinity,
    maxx = -Infinity,
    maxy = -Infinity,
    maxz = -Infinity;
  for (let i = 0; i < raw.count; i++) {
    const j = 3 * i;
    const x = raw.pos[j],
      y = raw.pos[j + 1],
      z = raw.pos[j + 2];
    if (x < minx) minx = x;
    if (y < miny) miny = y;
    if (z < minz) minz = z;
    if (x > maxx) maxx = x;
    if (y > maxy) maxy = y;
    if (z > maxz) maxz = z;
  }
  // Cell key: a packed NUMERIC key (ix + iy·Wx + iz·Wx·Wy) is ~2-3× faster than
  // the "ix,iy,iz" string over millions of points (no concat/hash). Collision-
  // free while the grid fits a safe integer; falls back to the string key for a
  // pathologically large/deep grid (behavior-identical either way).
  const gridKeyer = (cell) => {
    const Wx = Math.floor((maxx - minx) / cell) + 1;
    const Wy = Math.floor((maxy - miny) / cell) + 1;
    const Wz = Math.floor((maxz - minz) / cell) + 1;
    // NOTE: the numeric key wraps at grid edges (ix=−1 ≡ ix=Wx−1 of the prior
    // row), so callers that probe OUT-OF-RANGE cells (the stencil) must bound-
    // guard with Wx/Wy/Wz first. reduceAt only keys in-range points, so it's safe.
    const keyOf =
      Wx * Wy * Wz <= Number.MAX_SAFE_INTEGER
        ? (ix, iy, iz) => ix + iy * Wx + iz * (Wx * Wy)
        : (ix, iy, iz) => ix + "," + iy + "," + iz;
    return { keyOf, Wx, Wy, Wz };
  };
  const reduceAt = (cell) => {
    const map = new Map();
    const { keyOf } = gridKeyer(cell);
    for (let i = 0; i < raw.count; i++) {
      const j = 3 * i;
      const ix = Math.floor((raw.pos[j] - minx) / cell);
      const iy = Math.floor((raw.pos[j + 1] - miny) / cell);
      const iz = Math.floor((raw.pos[j + 2] - minz) / cell);
      const key = keyOf(ix, iy, iz);
      let e = map.get(key);
      if (!e) {
        e = {
          ix,
          iy,
          iz,
          px: 0,
          py: 0,
          pz: 0,
          nx: 0,
          ny: 0,
          nz: 0,
          ax: 0,
          ay: 0,
          az: 0,
          n: 0,
        };
        map.set(key, e);
      }
      e.px += raw.pos[j];
      e.py += raw.pos[j + 1];
      e.pz += raw.pos[j + 2];
      e.nx += raw.normal[j];
      e.ny += raw.normal[j + 1];
      e.nz += raw.normal[j + 2];
      e.ax += raw.albedo[j];
      e.ay += raw.albedo[j + 1];
      e.az += raw.albedo[j + 2];
      e.n++;
    }
    return map;
  };

  let cell = r0;
  let map = reduceAt(cell);
  for (let pass = 0; pass < 4 && map.size > cap; pass++) {
    cell *= Math.cbrt(map.size / cap);
    map = reduceAt(cell);
  }
  let entries = [...map.values()];
  if (entries.length > cap) {
    entries.sort((a, b) => b.n - a.n); // keep the densest cells
    entries = entries.slice(0, cap);
  }

  const fin = finalizeReduce(entries, cell, r0, cap, opts);
  const out = {
    points: fin.points,
    kept: fin.kept,
    dropped: raw.count - fin.kept,
    cell, // the ACTUAL final grid pitch (≥ r0 after cap re-reduces) — snapPoints'
    //      displacement bound (SPLAT_SHARPNESS S-2); r0 alone under-reports it.
  };
  if (fin.anisoStats) out.anisoStats = fin.anisoStats;
  return out;
}

// The shared "finalize" tail of the reduce — the survivor→output pipeline
// (cancel-drop → CSR neighbors → Pass-1 radius → Pass-1b aniso → Pass-2 alpha →
// output points) factored out so the batch path (reducePoints) and the
// streaming path (captureSplats stream:true) run the EXACT same code. `entries`
// is the per-cell running-sum list ({ix,iy,iz,px..az,n}), ALREADY capped to
// ≤ cap by the caller; `cell` is the final grid cell (drives the radius clamps),
// `r0` the reference radius (aniso sagitta budget). Returns { points, kept,
// anisoStats? } — the caller adds `dropped` (it alone knows the raw-hit total).
//
// ORIGIN-AGNOSTIC keyer (the streaming enabler): the spatial index is built from
// the ENTRIES' own ix/iy/iz index BOUNDS (offset by their min), NOT from a world
// min/max. A streaming accumulator keyed off an arbitrary fixed origin
// (frame.center, so its indices can be NEGATIVE) therefore reduces identically
// to the batch path — whose entries are already 0-based (floored from minx), so
// ixMin=0 and the derived Wx == reduceAt's Wx ⇒ byte-identical keys/output.
export function finalizeReduce(entries, cell, r0, cap, opts = {}) {
  const {
    radiusScale = 1.6,
    alphaBase = 0.95,
    alphaMin = 0.3,
    attenGamma = 0.5,
    rClampLo = 0.5,
    rClampHi = 3,
    coreX = null,
    aniso = 0,
    anisoMax = 3,
    anisoFloor = 1.5,
    epsFrac = 0.35,
  } = opts;

  // Survivors (one per surviving cell): averaged pos/normal/albedo + cell coords.
  // Drop cells whose opposing-view normals canceled (|n|<1e-6).
  // `disp` (S-3): normal DISPERSION ‖Σn‖/n ∈ (0,1] — captured HERE, before the
  // renormalization discards the magnitude. 1 = every hit's normal agrees
  // (flat/smooth cell); low = the cell straddles an edge/crease (normals fan
  // out). This is the edge signal the curvature PCA can't provide (a crease is
  // not a quadratic patch — Pass 1b fits 6% at a 90° crease vs 98.5% smooth).
  const surv = [];
  for (const e of entries) {
    const inv = 1 / e.n;
    let nx = e.nx * inv,
      ny = e.ny * inv,
      nz = e.nz * inv;
    const nl = Math.hypot(nx, ny, nz);
    if (nl < 1e-6) continue;
    surv.push({
      ix: e.ix,
      iy: e.iy,
      iz: e.iz,
      x: e.px * inv,
      y: e.py * inv,
      z: e.pz * inv,
      nx: nx / nl,
      ny: ny / nl,
      nz: nz / nl,
      ax: e.ax * inv,
      ay: e.ay * inv,
      az: e.az * inv,
      disp: Math.min(nl, 1),
    });
  }
  const kept = surv.length;

  // Spatial index: cell key → survivor index (≤1 survivor per cell). The keyer
  // is derived from the ENTRIES' integer index bounds (offset to their min), so
  // it works for a streaming accumulator with negative indices AND stays
  // byte-identical to reduceAt on the batch path (where ixMin=0 ⇒ Wx==reduceAt's
  // Wx and the offset is 0). The 27-neighbor stencil runs ~kept×26 lookups, so
  // the packed-numeric key matters here; string fallback for a huge/deep grid.
  // Neighbor probes are bound-guarded to [ixMin,ixMax] etc. so the numeric key
  // can't wrap onto a cell on the opposite grid face.
  let ixMin = Infinity,
    iyMin = Infinity,
    izMin = Infinity,
    ixMax = -Infinity,
    iyMax = -Infinity,
    izMax = -Infinity;
  for (const e of entries) {
    if (e.ix < ixMin) ixMin = e.ix;
    if (e.ix > ixMax) ixMax = e.ix;
    if (e.iy < iyMin) iyMin = e.iy;
    if (e.iy > iyMax) iyMax = e.iy;
    if (e.iz < izMin) izMin = e.iz;
    if (e.iz > izMax) izMax = e.iz;
  }
  const Wx = ixMax - ixMin + 1,
    Wy = iyMax - iyMin + 1,
    Wz = izMax - izMin + 1;
  const survKey =
    Wx * Wy * Wz <= Number.MAX_SAFE_INTEGER
      ? (ix, iy, iz) =>
          ix - ixMin + (iy - iyMin) * Wx + (iz - izMin) * (Wx * Wy)
      : (ix, iy, iz) => ix + "," + iy + "," + iz;
  const cellIndex = new Map();
  for (let i = 0; i < kept; i++)
    cellIndex.set(survKey(surv[i].ix, surv[i].iy, surv[i].iz), i);
  // Build each survivor's occupied-neighbor list ONCE into a flat CSR structure
  // (nbrStart offsets → nbrFlat indices). The 27-cell Map scan is the dominant
  // cost, so doing it a single time (not once per pass) roughly halves it, and
  // the inline flat loop below drops the per-neighbor closure + Map overhead
  // the old stencil(s, visit) callback carried (~40s → single-digit at 1.5M).
  const nbrStart = new Int32Array(kept + 1);
  const nbrFlat = [];
  for (let i = 0; i < kept; i++) {
    nbrStart[i] = nbrFlat.length;
    const s = surv[i];
    for (let dx = -1; dx <= 1; dx++) {
      const nx = s.ix + dx;
      if (nx < ixMin || nx > ixMax) continue;
      for (let dy = -1; dy <= 1; dy++) {
        const ny = s.iy + dy;
        if (ny < iyMin || ny > iyMax) continue;
        for (let dz = -1; dz <= 1; dz++) {
          if (dx === 0 && dy === 0 && dz === 0) continue;
          const nz = s.iz + dz;
          if (nz < izMin || nz > izMax) continue;
          const j = cellIndex.get(survKey(nx, ny, nz));
          if (j !== undefined) nbrFlat.push(j);
        }
      }
    }
  }
  nbrStart[kept] = nbrFlat.length;

  // Pass 1 — local radius from nearest-survivor spacing (isolated ⇒ gap-cover).
  const radius = new Float32Array(kept);
  for (let i = 0; i < kept; i++) {
    const s = surv[i];
    let dNN = Infinity;
    for (let k = nbrStart[i], e = nbrStart[i + 1]; k < e; k++) {
      const t = surv[nbrFlat[k]];
      const d = Math.hypot(s.x - t.x, s.y - t.y, s.z - t.z);
      if (d < dNN) dNN = d;
    }
    const spacing =
      dNN === Infinity
        ? rClampHi * cell
        : Math.min(Math.max(dNN, rClampLo * cell), rClampHi * cell);
    radius[i] = radiusScale * spacing;
  }

  // Pass 1b (P2/Rung 2) — anisotropic fit via tangent-plane curvature PCA on the
  // SAME CSR neighborhood. Gated on aniso>0 so the aniso=0 default path never runs
  // it (radius/r2/dir untouched ⇒ byte-identical output — back-compat pin). For each
  // survivor with ≥5 occupied neighbor cells: fit the 2nd fundamental form (all 3
  // coefficients — the cross-term κ_uv is mandatory, u0/v0 is an arbitrary basis),
  // eigen-decompose the 2×2 shape operator for principal curvatures + direction,
  // and set radii from the L1 sagitta law r=√(2ε/|κ|). A curvature-ratio noise floor
  // (anisoFloor) + the ≥5-cell gate keep noisy/ambiguous neighborhoods isotropic.
  let r2 = null,
    dir = null,
    anisoStats = null;
  if (aniso > 0) {
    r2 = new Float32Array(kept);
    dir = new Float32Array(3 * kept);
    const eps = epsFrac * r0;
    const loClamp = rClampLo * cell,
      hiClamp = rClampHi * cell;
    const clampR = (r) => Math.min(Math.max(r, loClamp), hiClamp);
    let fitted = 0,
      isoFallback = 0;
    for (let i = 0; i < kept; i++) {
      r2[i] = radius[i]; // default isotropic (dir stays 0)
      const kStart = nbrStart[i],
        kEnd = nbrStart[i + 1];
      if (kEnd - kStart < 5) {
        isoFallback++;
        continue;
      }
      const s = surv[i];
      const nx = s.nx,
        ny = s.ny,
        nz = s.nz;
      // tangent basis: u0 = n × (axis of smallest |n| component), v0 = n × u0
      const anx = Math.abs(nx),
        any = Math.abs(ny),
        anz = Math.abs(nz);
      let axx = 0,
        axy = 0,
        axz = 0;
      if (anx <= any && anx <= anz) axx = 1;
      else if (any <= anz) axy = 1;
      else axz = 1;
      let u0x = ny * axz - nz * axy,
        u0y = nz * axx - nx * axz,
        u0z = nx * axy - ny * axx;
      const ul = Math.hypot(u0x, u0y, u0z) || 1;
      u0x /= ul;
      u0y /= ul;
      u0z /= ul;
      const v0x = ny * u0z - nz * u0y,
        v0y = nz * u0x - nx * u0z,
        v0z = nx * u0y - ny * u0x;
      // normal equations for h ≈ ½κ_u a² + κ_uv ab + ½κ_v b² (cols [½a², ab, ½b²])
      let m00 = 0,
        m01 = 0,
        m02 = 0,
        m11 = 0,
        m12 = 0,
        m22 = 0,
        g0 = 0,
        g1 = 0,
        g2 = 0;
      for (let k = kStart; k < kEnd; k++) {
        const t = surv[nbrFlat[k]];
        const dx = t.x - s.x,
          dy = t.y - s.y,
          dz = t.z - s.z;
        const a = dx * u0x + dy * u0y + dz * u0z;
        const b = dx * v0x + dy * v0y + dz * v0z;
        const h = dx * nx + dy * ny + dz * nz; // sagitta
        const c0 = 0.5 * a * a,
          c1 = a * b,
          c2 = 0.5 * b * b;
        m00 += c0 * c0;
        m01 += c0 * c1;
        m02 += c0 * c2;
        m11 += c1 * c1;
        m12 += c1 * c2;
        m22 += c2 * c2;
        g0 += c0 * h;
        g1 += c1 * h;
        g2 += c2 * h;
      }
      const sol = solve3sym(m00, m01, m02, m11, m12, m22, g0, g1, g2);
      if (!sol) {
        isoFallback++;
        continue;
      }
      const ku = sol[0],
        kuv = sol[1],
        kv = sol[2];
      // symmetric 2×2 eigen: principal curvatures l1,l2
      const mid = (ku + kv) / 2,
        rad = Math.hypot((ku - kv) / 2, kuv);
      const l1 = mid + rad,
        l2 = mid - rad;
      const a1 = Math.abs(l1),
        a2 = Math.abs(l2);
      const maxA = Math.max(a1, a2),
        minA = Math.min(a1, a2);
      // noise floor: near-isotropic curvature ⇒ direction untrustworthy → isotropic
      const ratio = maxA < 1e-9 ? 1 : minA < 1e-12 ? Infinity : maxA / minA;
      if (ratio < anisoFloor) {
        isoFallback++;
        continue;
      }
      // sagitta radii: low curvature (minA) → big radius (major), high → small (minor)
      let rMax = clampR(Math.sqrt((2 * eps) / Math.max(minA, 1e-12)));
      let rMin = clampR(Math.sqrt((2 * eps) / Math.max(maxA, 1e-12)));
      if (rMax > anisoMax * rMin) rMin = rMax / anisoMax; // ratio clamp
      rMin += (1 - aniso) * (rMax - rMin); // aniso<1 = strength lerp toward isotropic
      // Cap by the Pass-1 spacing radius (coverage-driven): aniso must REDISTRIBUTE a
      // splat's extent (elongate), never inflate its size — else overdraw regresses
      // with no coverage gain (the P0-harness gate). Scale r_min with r_max to keep
      // the elongation ratio. Below the spacing radius, the sagitta law is free to act.
      const rSpace = radius[i];
      if (rMax > rSpace) {
        rMin *= rSpace / rMax;
        rMax = rSpace;
      }
      // major direction = eigenvector of the SMALL-|κ| eigenvalue, in world space
      const majLam = a1 <= a2 ? l1 : l2;
      const [ea, eb] = eigVec2(ku, kuv, kv, majLam);
      let dmx = ea * u0x + eb * v0x,
        dmy = ea * u0y + eb * v0y,
        dmz = ea * u0z + eb * v0z;
      const dl = Math.hypot(dmx, dmy, dmz) || 1;
      radius[i] = rMax; // major radius replaces the Pass-1 heuristic (Pass 2 uses it)
      r2[i] = rMin;
      dir[3 * i] = dmx / dl;
      dir[3 * i + 1] = dmy / dl;
      dir[3 * i + 2] = dmz / dl;
      fitted++;
    }
    anisoStats = { fitted, isotropicFallback: isoFallback, epsBudget: eps };
  }

  // Pass 1c (S-3) — EDGE-aware shape from normal dispersion. The curvature PCA
  // above is structurally wrong at creases (a fold is not a quadratic patch —
  // measured 6% fitted at a 90° crease vs 98.5% on a cylinder), yet edges are
  // where the eye judges sharpness. Dispersion identifies them for free
  // (‖Σn‖/n < τ), and the edge DIRECTION falls out of the neighbor normals:
  // for a crease between faces n1,n2 the mean m ∝ n1+n2 and the top spread
  // eigenvector s ∝ n1−n2, so e = m×s is the crease line itself. Shape: major
  // axis ALONG the edge (radius kept), minor ACROSS it (radius/anisoMax) —
  // a thin ellipse hugging the crease, the trained-3DGS signature. Precedence:
  // dispersion < τ REPLACES the Pass-1b fit (which we KNOW is garbage there).
  let edgeStats = null;
  if (aniso > 0 && r2) {
    const edgeTau = opts.edgeTau ?? 0.85;
    const loClamp = rClampLo * cell;
    let edges = 0;
    for (let i = 0; i < kept; i++) {
      const s = surv[i];
      if (s.disp >= edgeTau) continue;
      const kStart = nbrStart[i],
        kEnd = nbrStart[i + 1];
      if (kEnd - kStart < 3) continue; // too few neighbors to estimate a spread
      // Covariance of neighbor mean-normals about this cell's mean normal.
      let cxx = 0,
        cxy = 0,
        cxz = 0,
        cyy = 0,
        cyz = 0,
        czz = 0;
      for (let k = kStart; k < kEnd; k++) {
        const t = surv[nbrFlat[k]];
        const dx = t.nx - s.nx,
          dy = t.ny - s.ny,
          dz = t.nz - s.nz;
        cxx += dx * dx;
        cxy += dx * dy;
        cxz += dx * dz;
        cyy += dy * dy;
        cyz += dy * dz;
        czz += dz * dz;
      }
      // Top eigenvector by power iteration (symmetric PSD 3×3; 8 rounds is
      // plenty at this tolerance). Deterministic fixed seed.
      let vx = 1,
        vy = 0.5,
        vz = 0.25;
      for (let it = 0; it < 8; it++) {
        const wx = cxx * vx + cxy * vy + cxz * vz;
        const wy = cxy * vx + cyy * vy + cyz * vz;
        const wz = cxz * vx + cyz * vy + czz * vz;
        const wl = Math.hypot(wx, wy, wz);
        if (wl < 1e-20) break; // no spread — leave the Pass-1b result
        vx = wx / wl;
        vy = wy / wl;
        vz = wz / wl;
      }
      // Edge line = mean-normal × spread-direction.
      let ex = s.ny * vz - s.nz * vy,
        ey = s.nz * vx - s.nx * vz,
        ez = s.nx * vy - s.ny * vx;
      const el = Math.hypot(ex, ey, ez);
      if (el < 1e-6) continue; // degenerate (spread ∥ normal) — keep Pass-1b
      ex /= el;
      ey /= el;
      ez /= el;
      dir[3 * i] = ex;
      dir[3 * i + 1] = ey;
      dir[3 * i + 2] = ez;
      r2[i] = Math.max(radius[i] / anisoMax, loClamp); // thin ACROSS the crease
      edges++;
    }
    edgeStats = { edges, edgeTau };
  }

  // Pass 2 — overlap-attenuated opacity, normalized by the population median so
  // a uniform sheet's baseline overlap doesn't dim it; only pileups thin out.
  const overlaps = new Int32Array(kept);
  for (let i = 0; i < kept; i++) {
    const s = surv[i];
    const ri = radius[i];
    let m = 0;
    for (let k = nbrStart[i], e = nbrStart[i + 1]; k < e; k++) {
      const t = surv[nbrFlat[k]];
      if (Math.hypot(s.x - t.x, s.y - t.y, s.z - t.z) < ri + radius[nbrFlat[k]])
        m++;
    }
    overlaps[i] = m;
  }
  const mMed = median(overlaps);
  const alpha = new Float32Array(kept);
  for (let i = 0; i < kept; i++) {
    const a =
      attenGamma > 0
        ? alphaBase * Math.pow((mMed + 1) / (overlaps[i] + 1), attenGamma)
        : alphaBase;
    alpha[i] = Math.min(Math.max(a, alphaMin), alphaBase);
  }

  // Output only the CORE survivors (all, when coreX is null). Radius/alpha were
  // computed against the full slab+margin, so core boundary splats are correct.
  const keep = [];
  for (let i = 0; i < kept; i++)
    if (!coreX || (surv[i].x >= coreX[0] && surv[i].x < coreX[1])) keep.push(i);
  const outN = keep.length;
  const pos = new Float32Array(3 * outN);
  const normal = new Float32Array(3 * outN);
  const albedo = new Float32Array(3 * outN);
  const radiusOut = new Float32Array(outN);
  const alphaOut = new Float32Array(outN);
  // S-3: per-splat normal dispersion (edge signal) — consumed by densifySplats
  // and available to any downstream shaping. Always emitted (4 B/splat).
  const dispOut = new Float32Array(outN);
  // Aniso (P2): r2/dir emitted ONLY when the PCA pass ran (aniso>0) — absent keys
  // in the default path keep the output object shape byte-identical (back-compat).
  const r2Out = r2 ? new Float32Array(outN) : null;
  const dirOut = dir ? new Float32Array(3 * outN) : null;
  for (let o = 0; o < outN; o++) {
    const i = keep[o],
      s = surv[i],
      j = 3 * o;
    pos[j] = s.x;
    pos[j + 1] = s.y;
    pos[j + 2] = s.z;
    normal[j] = s.nx;
    normal[j + 1] = s.ny;
    normal[j + 2] = s.nz;
    albedo[j] = s.ax;
    albedo[j + 1] = s.ay;
    albedo[j + 2] = s.az;
    radiusOut[o] = radius[i];
    alphaOut[o] = alpha[i];
    dispOut[o] = s.disp;
    if (r2Out) {
      r2Out[o] = r2[i];
      dirOut[j] = dir[3 * i];
      dirOut[j + 1] = dir[3 * i + 1];
      dirOut[j + 2] = dir[3 * i + 2];
    }
  }
  const points = {
    count: outN,
    pos,
    normal,
    albedo,
    radius: radiusOut,
    alpha: alphaOut,
    dispersion: dispOut,
  };
  if (r2Out) {
    points.r2 = r2Out;
    points.dir = dirOut;
  }
  const out = { points, kept: outN };
  if (anisoStats) {
    out.anisoStats = anisoStats;
    if (edgeStats) out.anisoStats.edges = edgeStats.edges;
  }
  return out;
}

// Solve a symmetric 3×3 system M x = g (M given by upper triangle) via the
// adjugate; returns [x0,x1,x2] or null if near-singular (degenerate neighbor set).
function solve3sym(m00, m01, m02, m11, m12, m22, g0, g1, g2) {
  const A00 = m11 * m22 - m12 * m12,
    A01 = m02 * m12 - m01 * m22,
    A02 = m01 * m12 - m02 * m11;
  const det = m00 * A00 + m01 * A01 + m02 * A02;
  if (Math.abs(det) < 1e-20) return null;
  const A11 = m00 * m22 - m02 * m02,
    A12 = m02 * m01 - m00 * m12,
    A22 = m00 * m11 - m01 * m01;
  const inv = 1 / det;
  return [
    inv * (A00 * g0 + A01 * g1 + A02 * g2),
    inv * (A01 * g0 + A11 * g1 + A12 * g2),
    inv * (A02 * g0 + A12 * g1 + A22 * g2),
  ];
}

// Unit eigenvector (in the u0,v0 tangent basis) of the symmetric 2×2
// [[ku,kuv],[kuv,kv]] for eigenvalue `lam`. Falls back across the two rows so a
// diagonal matrix (kuv≈0) still resolves to the correct axis.
function eigVec2(ku, kuv, kv, lam) {
  let ex = kuv,
    ey = lam - ku;
  let l = Math.hypot(ex, ey);
  if (l < 1e-12) {
    ex = lam - kv;
    ey = kuv;
    l = Math.hypot(ex, ey);
  }
  if (l < 1e-12) return [1, 0]; // fully degenerate → arbitrary in-plane axis
  return [ex / l, ey / l];
}

// Median of an integer array (copy-sort; empty ⇒ 0). Used for the overlap
// baseline in reducePoints Pass 2.
function median(arr) {
  const n = arr.length;
  if (n === 0) return 0;
  const s = Array.from(arr).sort((a, b) => a - b);
  const mid = n >> 1;
  return n % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

// Full single-threaded pipeline (CLI + tests + non-Worker fallback).
// opts: { views=64, res=256, cap=1_500_000, radiusScale=1.6, iters,
//         onProgress?: (done,total)=>boolean /* false = cancel */ }.
// Returns { points, r0, frame, stats } or null (scope reject throws; zero hits
// → null, the §5.3a empty guard — never a degenerate result).
// Deterministic streaming reservoir (Knuth's Algorithm R) over raw hit triplets,
// for the P0 fidelity harness (§SPLAT_GAP_IMPL P0). A SEEDED 32-bit LCG — NOT
// Math.random — so metric pins are reproducible. Bounded memory: 9 floats × cap,
// fixed regardless of hit count (cap ≤ 500k by convention; ~18 MB at the cap).
// Streaming by design (addChunk per capture chunk) so a future chunked capture
// (S3b/P3) can feed it without ever holding the whole cloud; P0's two call sites
// use the one-shot sampleHits() form on the already-materialized cloud.
export function makeHitReservoir(cap = 500_000, seed = 1) {
  cap = Math.max(0, Math.floor(cap));
  const pos = new Float32Array(3 * cap);
  const normal = new Float32Array(3 * cap);
  const albedo = new Float32Array(3 * cap);
  let filled = 0; // slots used so far = min(seen, cap)
  let seen = 0; // total hits offered
  let s = seed >>> 0 || 1;
  const rnd = () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296; // [0, 1)
  };
  const put = (slot, P, N, A, o) => {
    const j = 3 * slot;
    pos[j] = P[o];
    pos[j + 1] = P[o + 1];
    pos[j + 2] = P[o + 2];
    normal[j] = N[o];
    normal[j + 1] = N[o + 1];
    normal[j + 2] = N[o + 2];
    albedo[j] = A[o];
    albedo[j + 1] = A[o + 1];
    albedo[j + 2] = A[o + 2];
  };
  return {
    // pos/normal/albedo: Float32Array|number[] of 3N floats — one capture chunk.
    addChunk(P, N, A) {
      const nHits = (P.length / 3) | 0;
      for (let h = 0; h < nHits; h++) {
        const o = 3 * h;
        if (filled < cap) put(filled++, P, N, A, o);
        else {
          const r = Math.floor(rnd() * (seen + 1)); // uniform in [0, seen]
          if (r < cap) put(r, P, N, A, o);
        }
        seen++;
      }
    },
    get seen() {
      return seen;
    },
    // { count, pos, normal, albedo } Float32Arrays (SplatPoints shape).
    result() {
      const n = filled;
      return {
        count: n,
        pos: pos.slice(0, 3 * n),
        normal: normal.slice(0, 3 * n),
        albedo: albedo.slice(0, 3 * n),
      };
    },
  };
}

// One-shot reservoir sample of an in-memory raw cloud (SplatPoints in → SplatPoints
// out, ≤ cap hits). The form P0's CLI and app paths both use on the materialized cloud.
export function sampleHits(raw, cap = 500_000, seed = 1) {
  const r = makeHitReservoir(cap, seed);
  r.addChunk(raw.pos, raw.normal, raw.albedo);
  return r.result();
}

// ── Streaming reduce (docs/planning/SPLAT_STREAMING_REDUCE.md) ────────────────
// The per-cell running sum ({ix,iy,iz,Σpos,Σnormal,Σalbedo,n}) is additive, so
// each capture view can be merged into a persistent grid Map and its raw hits
// freed at once — peak memory ≈ final splat count, not total hits. ALL
// coarsening is INTEGER-factor (⌊ix/f⌋): sharing the fixed origin, that equals
// reduceAt at cell·f exactly (the ⌊⌊x/c⌋/f⌋=⌊x/(c·f)⌋ floor-composition identity
// holds only for integer f — non-integer factors would silently diverge).

// A fresh per-cell running-sum entry.
function newCell(ix, iy, iz) {
  return {
    ix,
    iy,
    iz,
    px: 0,
    py: 0,
    pz: 0,
    nx: 0,
    ny: 0,
    nz: 0,
    ax: 0,
    ay: 0,
    az: 0,
    n: 0,
  };
}

// Merge one capture chunk's hits into `acc` at grid `cell`, keyed off a FIXED
// world `origin` ([x,y,z], typically frame.center — known before any hits so
// the lattice is stable across views/workers). This is reduceAt's inner body.
// A STRING key (not the batch path's packed numeric one): streaming indices are
// unbounded and can be negative (points on either side of the origin), so no
// packed key has a safe fixed width — correctness over speed (CPU capture is
// already minutes-scale). finalizeReduce re-keys survivors numerically anyway.
function mergeInto(acc, cell, P, N, A, origin) {
  const ox = origin[0],
    oy = origin[1],
    oz = origin[2];
  const nHits = (P.length / 3) | 0;
  for (let h = 0; h < nHits; h++) {
    const j = 3 * h;
    const ix = Math.floor((P[j] - ox) / cell);
    const iy = Math.floor((P[j + 1] - oy) / cell);
    const iz = Math.floor((P[j + 2] - oz) / cell);
    const key = ix + "," + iy + "," + iz;
    let e = acc.get(key);
    if (!e) {
      e = newCell(ix, iy, iz);
      acc.set(key, e);
    }
    e.px += P[j];
    e.py += P[j + 1];
    e.pz += P[j + 2];
    e.nx += N[j];
    e.ny += N[j + 1];
    e.nz += N[j + 2];
    e.ax += A[j];
    e.ay += A[j + 1];
    e.az += A[j + 2];
    e.n++;
  }
}

// Integer-factor coarsen: re-key each entry at cell·f (⌊ix/f⌋…, floor division —
// valid for negative indices) and SUM the colliding running sums. Returns the
// new { acc, cell }. f MUST be an integer ≥ 2.
function coarsen(acc, cell, f) {
  const acc2 = new Map();
  for (const e of acc.values()) {
    const ix = Math.floor(e.ix / f),
      iy = Math.floor(e.iy / f),
      iz = Math.floor(e.iz / f);
    const key = ix + "," + iy + "," + iz;
    let g = acc2.get(key);
    if (!g) {
      g = newCell(ix, iy, iz);
      acc2.set(key, g);
    }
    g.px += e.px;
    g.py += e.py;
    g.pz += e.pz;
    g.nx += e.nx;
    g.ny += e.ny;
    g.nz += e.nz;
    g.ax += e.ax;
    g.ay += e.ay;
    g.az += e.az;
    g.n += e.n;
  }
  return { acc: acc2, cell: cell * f };
}

// Distinct coarse-cell count for factor f WITHOUT building entry objects — lets
// chooseFactor pick before committing to a (destructive) coarsen.
function coarsenedSize(acc, f) {
  const seen = new Set();
  for (const e of acc.values())
    seen.add(
      Math.floor(e.ix / f) +
        "," +
        Math.floor(e.iy / f) +
        "," +
        Math.floor(e.iz / f),
    );
  return seen.size;
}

// §4 factor ladder (2,3,5,7,… — finer than powers of two so the achieved count
// lands closer to `target` from below). Pick the SMALLEST factor whose coarsened
// size ≤ target (⇒ the largest size ≤ target = tightest landing); if none
// reaches target, the largest ladder factor (biggest single-step drop; the
// caller's loop re-coarsens).
const COARSEN_LADDER = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31];
function chooseFactor(acc, target) {
  for (const f of COARSEN_LADDER) if (coarsenedSize(acc, f) <= target) return f;
  return COARSEN_LADDER[COARSEN_LADDER.length - 1];
}

// Reusable streaming reducer (SPLAT_STREAMING_REDUCE PR-2 §5/§6): a persistent
// bounded-memory accumulator that any capture source — the CPU march here, the
// GPU G-buffer (core/preview.js), or a capture Worker (app/src/splatworker.ts) —
// feeds one CHUNK at a time (addChunk), freeing each chunk's raw hits at once, so
// peak memory ≈ final splat count, NOT total hits (the OOM fix). This is the
// factory form of captureSplats' inline stream branch — all three call sites share
// this exact code so parity holds. `finalize()` runs the cap-enforce + slice +
// finalizeReduce tail and returns { points, r0, bbox, sample, stats } — bbox and
// sample replace the app's old worldBBox(raw) / takeMetricsSample(raw) passes,
// which streaming can't run (there is no whole raw cloud to scan).
//
// - `frame`   : the capture frame; frame.center is the FIXED grid origin (known
//               before any hit, so the lattice never shifts across chunks) and
//               frame.diag drives r0. GPU/CPU hits are BOTH world coords (the GPU
//               pass adds O back), so frame.center works for every source.
// - `views`   : total view count — for the §3 first-chunk cell estimate (bias
//               FINE: over-estimating hits costs only a mid-stream coarsen, while
//               too-coarse is unrecoverable once the raw hits are freed).
// - `sampleN` : >0 ⇒ keep a ≤N seeded reservoir subsample of the raw hits
//               (result.sample), for Auto-radius + the P0 harness.
// - the reduce knobs (radiusScale, aniso*, rClamp*, alpha*, attenGamma, epsFrac)
//               match reducePoints/finalizeReduce defaults, so an omitted knob
//               reduces identically to the batch path. `thinEps` is accepted for
//               API symmetry (it belongs to the downstream fit, not the reduce).
export function makeStreamingReduce(opts = {}) {
  const {
    frame,
    cap = 1_500_000,
    radiusScale = 1.6,
    alphaBase = 0.95,
    alphaMin = 0.3,
    attenGamma = 0.5,
    rClampLo = 0.5,
    rClampHi = 3,
    aniso = 0,
    anisoMax = 3,
    anisoFloor = 1.5,
    epsFrac = 0.35,
    thinEps, // accepted for API symmetry; used by the downstream fit, not the reduce
    sampleN = 0,
    seed = 1,
    views = 64,
  } = opts;
  void thinEps;
  const origin = frame.center;
  // Bound the live accumulator's MEMORY, not just its final count. Each cell is a
  // JS object + string key ≈ 333 bytes (measured 2026-07-22), so the OLD 2·cap
  // ceiling let a dense 6M-cap capture grow the grid to ~4 GB and crash the tab
  // (Chrome "Aw Snap" code 5). 1.3·cap coarsens a touch sooner (negligible
  // quality cost — coarsening averages colliding cells) and holds the peak to
  // 1.3·cap·333 B (≈1.1 GB at the 2.5M cap the presets now use). The FUNDAMENTAL
  // fix (SoA typed-array cells → ~150 B, restoring a higher cap) is a follow-up.
  const MEM_CEIL = Math.ceil(1.3 * cap);
  const reservoir = sampleN > 0 ? makeHitReservoir(sampleN, seed) : null;
  const bmin = [Infinity, Infinity, Infinity];
  const bmax = [-Infinity, -Infinity, -Infinity];
  const updateBBox = (P) => {
    for (let j = 0; j < P.length; j += 3) {
      if (P[j] < bmin[0]) bmin[0] = P[j];
      if (P[j] > bmax[0]) bmax[0] = P[j];
      if (P[j + 1] < bmin[1]) bmin[1] = P[j + 1];
      if (P[j + 1] > bmax[1]) bmax[1] = P[j + 1];
      if (P[j + 2] < bmin[2]) bmin[2] = P[j + 2];
      if (P[j + 2] > bmax[2]) bmax[2] = P[j + 2];
    }
  };
  let acc = new Map();
  let cell = null; // sized from the first non-empty chunk's hit count (§3)
  let seen = 0;
  let maxAccSize = 0; // peak PERSISTENT grid size (post-coarsen) — test/diag seam
  return {
    // Merge one capture chunk into the accumulator, then free the chunk (caller
    // drops its reference). P/N/A: Float32Array|number[] of 3·nHits floats.
    // `totalScale` converts THIS chunk's hit count into an estimate of the
    // whole capture's: hits × totalScale. It defaults to `views` (every view
    // yields alike), which is what a batched chunk and the old constant-window
    // capture both want. Since CAPTURE_VOLUME_SHAPES the per-view window is the
    // volume's support along that view, so window AREA — and with it a view's
    // hit count — varies by direction; the driver passes an area-weighted scale
    // instead. Left unweighted, the first view alone sized the grid and streamed
    // and batched survivor counts drifted apart (measured 6.8% on Mandelbulb,
    // against a 5% pin).
    addChunk(P, N, A, totalScale = views) {
      const nHits = (P.length / 3) | 0;
      if (nHits === 0) return; // empty view — no-op (never sizes the cell)
      seen += nHits;
      updateBBox(P);
      // §3 initial cell: extrapolate the total and size the grid from it. The
      // estimate's bias washes out in finalize(), which knows the true count —
      // but the initial cell fixes the LATTICE, so a bad estimate still shows.
      // #518: measure the spacing over THIS CHUNK'S OWN hit bbox, not the frame
      // (capturedDiag) — a frame holding empty space would lock in a lattice
      // coarser than the geometry by the square of how much (and coarsening is
      // one-way, so the raw hits are gone by the time finalize() knows better).
      // One view under-covers the object along its own axis, so the estimate is
      // biased FINE, which is the recoverable direction by design: a too-fine
      // grid costs a mid-stream coarsen, a too-coarse one costs the detail.
      if (cell === null)
        cell = computeR0(
          capturedDiag(bmin, bmax, frame.diag),
          Math.max(1, nHits * totalScale),
          radiusScale,
        );
      if (reservoir) reservoir.addChunk(P, N, A);
      mergeInto(acc, cell, P, N, A, origin);
      while (acc.size > MEM_CEIL) {
        const r = coarsen(acc, cell, chooseFactor(acc, cap));
        acc = r.acc;
        cell = r.cell;
      }
      if (acc.size > maxAccSize) maxAccSize = acc.size;
    },
    // Cap-enforce (integer coarsen) → hard-cap slice → finalizeReduce. Returns
    // null on zero hits (§5.3a empty guard), else the finalized survivor cloud.
    finalize() {
      if (seen === 0) return null;
      // The true total AND the true captured extent are now known → the real r0
      // (the first chunk's bias washes out). Cap-enforce by integer coarsening,
      // then hard-cap slice.
      const r0 = computeR0(
        capturedDiag(bmin, bmax, frame.diag),
        seen,
        radiusScale,
      );
      while (acc.size > cap) {
        const r = coarsen(acc, cell, chooseFactor(acc, cap));
        acc = r.acc;
        cell = r.cell;
      }
      let entries = [...acc.values()];
      if (entries.length > cap) {
        entries.sort((a, b) => b.n - a.n); // keep the densest cells
        entries = entries.slice(0, cap);
      }
      const fin = finalizeReduce(entries, cell, r0, cap, {
        radiusScale,
        alphaBase,
        alphaMin,
        attenGamma,
        rClampLo,
        rClampHi,
        aniso,
        anisoMax,
        anisoFloor,
        epsFrac,
      });
      const result = {
        points: fin.points,
        r0,
        cell, // actual post-coarsen grid pitch — snapPoints' displacement bound (S-2)
        bbox: { min: bmin, max: bmax },
        stats: {
          rawHits: seen,
          kept: fin.kept,
          dropped: seen - fin.kept,
          maxAccSize,
        },
      };
      if (fin.anisoStats) result.stats.anisoStats = fin.anisoStats;
      if (reservoir) result.sample = reservoir.result();
      return result;
    },
  };
}

export function captureSplats(formula, coloring, opts = {}) {
  const {
    views = 64,
    res = 256,
    cap = 1_500_000,
    radiusScale = 1.6,
    alphaBase = 0.95,
    alphaMin = 0.3,
    attenGamma = 0.5,
    rClampLo = 0.5,
    rClampHi = 3,
    layers = 2,
    aoStrength = 0.5,
    sampleHits: sampleN = 0, // >0 ⇒ attach a ≤N reservoir subsample of raw hits (P0 harness)
    aniso = 0, // P2: forwarded to reducePoints (0 = isotropic, default)
    anisoMax = 3,
    anisoFloor = 1.5,
    epsFrac = 0.35,
    // Bounded-memory streaming reduce (SPLAT_STREAMING_REDUCE PR-1): merge each
    // view into a persistent grid + free its raw hits, so peak memory ≈ final
    // splat count not Σ hits. Default false ⇒ the batch path (below) is byte-
    // identical and all batch pins hold. See the streaming branch after albedoAt.
    stream = false,
    onProgress,
  } = opts;
  const iters = opts.iters ?? formula.iters ?? 8;
  const scene = Array.isArray(formula.objects) && formula.objects.length > 0;
  const deScale = deScaleFor(formula);

  // The saved-camera frame is built UNCONDITIONALLY — it's both the no-surface
  // fallback and the refineFrame fallback for a ballooned (non-null) scene
  // frame, which never enters the `if (!frame)` branch.
  const camFrame = cameraFrame(formula);
  // S-5a: an explicit `frame` (the caller's zoomed camera view) overrides the
  // whole-object framing — captured as-is (no refine; the user chose this box).
  // `probed === null` ⇒ NO measured extent, so `frame` is the generic cube.
  const probed = opts.frame ? null : frameFormula(formula);
  let frame = opts.frame ?? probed ?? camFrame;

  const de = makeDE(formula, iters);
  // Scenes: the evaluate.js probe mis-frames leaves/planes — re-frame against
  // the real capture DE (needs `de`, hence built above). Skipped when the caller
  // pinned an explicit view frame.
  // #457: ALSO re-frame when the probe measured nothing at all. frameFormula
  // returns null for 35 of 90 presets — the evaluate.js orbit probe only
  // finalizes the IFS / escape-time DE, so it is blind to leaf shapes and to
  // deOption-3 (numeric) stacks and reports hits=0 at every region. The
  // fallback is cameraFrame's UNIFORM CUBE, and growFrameToSurface only scales
  // it (aspect-preserving), so the per-axis capture volume was inert for 39% of
  // the library — Gnarl Dunes really measures ext ~[1.88, 1.88, 0.18] but
  // captured as a cube. refineFrame marches the REAL capture DE, so it sees
  // what the probe can't; when it still finds < minHits it returns camFrame,
  // i.e. exactly the previous behavior.
  if (!opts.frame && (scene || !probed))
    frame = refineFrame(de, frame, camFrame, { deScale });
  // #351 safety net: even the refined/whole-object frame can still miss the
  // real surface (see growFrameToSurface) — verify with a cheap multi-view
  // probe and grow if needed, for scenes and flat/hybrid formulas alike.
  // Skipped when the caller pinned an explicit view frame (their choice of box).
  if (!opts.frame) frame = growFrameToSurface(de, frame, { deScale });
  // #507: the frame is settled — now measure the DE's convergence floor and,
  // when it sits above the scale-relative default, raise the march epsilon to
  // clear it. Runs for a caller-pinned frame too: the floor is a property of
  // the DE, not of the framing. Converging DEs get the same frame object back.
  frame = withCaptureEps(de, frame, { deScale });

  const albedoAt = makePointAlbedo(formula, coloring, iters);

  // ── Streaming path (the OOM fix) ────────────────────────────────────────────
  // Persistent bounded-memory accumulator at a FIXED origin (frame.center, known
  // before any hit arrives so the lattice never shifts). Merge each view, free
  // its raw hits, coarsen (integer-factor) if the grid tops MEM_CEIL. The batch
  // path below is left untouched.
  if (stream) {
    const reduce = makeStreamingReduce({
      frame,
      cap,
      radiusScale,
      alphaBase,
      alphaMin,
      attenGamma,
      rClampLo,
      rClampHi,
      aniso,
      anisoMax,
      anisoFloor,
      epsFrac,
      sampleN,
      views,
    });
    // Per-view hit yield, for the streaming grid's initial cell estimate. Every
    // view spends the same res² rays, so a WIDER window (the volume's support
    // along that view) just spreads them thinner: hits scale as 1/area, not with
    // it. Treating the object's projected area as roughly view-independent,
    // total ≈ hits_k · area_k · Σ_j 1/area_j — which collapses to the old
    // `hits · views` exactly when every window is the same size.
    const dirs = Array.from({ length: views }, (_, k) =>
      fibonacciDir(k, views),
    );
    const areas = dirs.map((v) => {
      const b = viewBasis(v, frame);
      return Math.max(b.hu * b.hv, 1e-30);
    });
    const invAreaSum = areas.reduce((a, x) => a + 1 / x, 0);
    let rays = 0;
    for (let k = 0; k < views; k++) {
      const chunk = { pos: [], normal: [], albedo: [] };
      captureView(de, albedoAt, frame, dirs[k], res, chunk, {
        maxSteps: 200,
        deScale,
        layers,
        aoStrength,
      });
      rays += res * res;
      reduce.addChunk(
        chunk.pos,
        chunk.normal,
        chunk.albedo,
        areas[k] * invAreaSum,
      );
      // (chunk drops out of scope next iteration → its raw hits are freed.)
      if (onProgress && onProgress(k + 1, views) === false) return null; // cancelled
    }
    const fin = reduce.finalize();
    if (!fin) return null; // empty guard (§5.3a)
    const result = {
      points: fin.points,
      r0: fin.r0,
      cell: fin.cell, // S-2: snapPoints' displacement bound
      frame,
      // Running world bbox (replaces the app's old worldBBox(raw) pass).
      bbox: fin.bbox,
      stats: {
        views,
        rays,
        rawHits: fin.stats.rawHits,
        kept: fin.stats.kept,
        dropped: fin.stats.dropped,
        maxAccSize: fin.stats.maxAccSize,
      },
    };
    if (fin.stats.anisoStats) result.stats.anisoStats = fin.stats.anisoStats;
    if (sampleN > 0) {
      result.sample = fin.sample;
      result.stats.iters = iters;
    }
    return result;
  }

  const out = { pos: [], normal: [], albedo: [] };
  let rays = 0;
  for (let k = 0; k < views; k++) {
    captureView(de, albedoAt, frame, fibonacciDir(k, views), res, out, {
      maxSteps: 200,
      deScale,
      layers,
      aoStrength,
    });
    rays += res * res;
    if (onProgress && onProgress(k + 1, views) === false) return null; // cancelled
  }

  const rawHits = out.pos.length / 3;
  if (rawHits === 0) return null; // empty guard (§5.3a)
  // #518: pitch over what was CAPTURED, not over the frame we looked in — the
  // same rule the streaming path applies, so the two stay in parity.
  const bmin = [Infinity, Infinity, Infinity];
  const bmax = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < out.pos.length; i += 3)
    for (let c = 0; c < 3; c++) {
      const v = out.pos[i + c];
      if (v < bmin[c]) bmin[c] = v;
      if (v > bmax[c]) bmax[c] = v;
    }
  const r0 = computeR0(
    capturedDiag(bmin, bmax, frame.diag),
    rawHits,
    radiusScale,
  );
  const raw = {
    count: rawHits,
    pos: Float32Array.from(out.pos),
    normal: Float32Array.from(out.normal),
    albedo: Float32Array.from(out.albedo),
  };
  const { points, kept, dropped, anisoStats, cell } = reducePoints(
    raw,
    r0,
    cap,
    {
      radiusScale,
      alphaBase,
      alphaMin,
      attenGamma,
      rClampLo,
      rClampHi,
      aniso,
      anisoMax,
      anisoFloor,
      epsFrac,
    },
  );
  const result = {
    points,
    r0,
    cell, // S-2: snapPoints' displacement bound
    frame,
    stats: { views, rays, rawHits, kept, dropped },
  };
  if (anisoStats) result.stats.anisoStats = anisoStats;
  // P0 harness hook: attach a raw-hit subsample + the iters the metrics need to
  // rebuild de/albedoAt. Gated on sampleN>0 so the default return shape is
  // byte-for-byte unchanged (back-compat pin).
  if (sampleN > 0) {
    result.sample = sampleHits(raw, sampleN);
    result.stats.iters = iters;
  }
  return result;
}

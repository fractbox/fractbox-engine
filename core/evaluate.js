// ─────────────────────────────────────────────────────────────────────────
// CPU evaluator + stability measure — graphics-hardware-independent.
// ─────────────────────────────────────────────────────────────────────────
// stability.js answers a binary "does it stand?". The games want a CONTINUUM:
// how much does the tower wobble, and which way does it lean? That needs the
// actual dynamics, not static flags — so this module ports the per-op math to
// plain JS and iterates a deterministic cloud of sample points on the CPU. No
// WebGPU, so it runs in Node (check.mjs / puzzle generation) and on any browser.
//
//   measure(formula) → {
//     wobble,    // 0 = rock-solid, 1 = toppling / renders blank   (the game knob)
//     lean,      // {x,y,z} unit-ish direction the fractal mass leans (0 = balanced)
//     leanMag,   // |lean| magnitude, 0..~1
//     coverage,  // fraction of sample points inside the fractal body
//     escaped, nan,  // diagnostic fractions
//     family, supported,
//   }
//
// This is a THIRD copy of the operator math (after wgsl + glsl). That's a real
// maintenance cost the project's doctrine warns about — but it's the price of
// being GPU-independent, and evaluate.test.mjs guards every supported key
// against the IR so a renamed op fails CI rather than silently mis-measuring.
// The honest framing: this is a coarse CPU proxy for what the renderer would
// draw — good enough to drive wobble/lean and to flag blank renders headlessly,
// NOT a pixel-exact render.
// ─────────────────────────────────────────────────────────────────────────

import { activeOps, OPERATORS, effectiveDeOption } from "./operators.js";
import { deFamily, hybridDeFamily } from "./stability.js";
import { hybridSlots } from "./hybridmodel.js";
// The orbit loop lives ONCE, in makeOrbit (also used by cpu.js's own render
// path — the render-authoritative tier). Deriving from it here — instead of
// keeping private twin loops — is what stops the two from drifting (the op
// math did once, on `menger`, and the private loops were blind to hybrid A/B
// schedules, so a broken hybrid slot B sailed through the vary.js ship-gate).
// See REFACTORING.md item 1. Imported from cpuorbit.js, NOT cpu.js (#266): the
// vary.js generator gate (measure(), on the app boot path) only needs this
// ~1000-line orbit engine, not the ~2400 lines of scene/leaf/render/ASCII code
// that live in cpu.js alongside it.
import { makeOrbit } from "./cpuorbit.js";

// The operator keys this evaluator implements. Derived from the IR registry —
// applyOp (in cpu.js) has a case for every operator, so the supported set IS the
// registry. Deriving it means a newly-added op is measurable the moment it's in
// the registry, and can never fall out of sync with a hand-maintained list
// (opmath.test.mjs asserts applyOp actually transforms each key, not a no-op).
export const SUPPORTED = Object.freeze(OPERATORS.map((o) => o.key));
const _SUPPORTED = new Set(SUPPORTED);

// ── deterministic low-discrepancy sampling (Halton) — reproducible, no RNG ──
function halton(i, base) {
  let f = 1,
    r = 0;
  while (i > 0) {
    f /= base;
    r += f * (i % base);
    i = Math.floor(i / base);
  }
  return r;
}

// Sampling + thresholds. A point "renders" if its DE sharpness S = ln(|w|/r)
// exceeds `sharpThresh` (the derivative meaningfully outgrew the radius, so the
// distance estimate collapsed toward a surface there). `coverageGood` is the
// fraction of space that must render for a rock-solid wobble 0. Tuned vs. known
// recipes — using the COVERAGE (how much of space renders), not the mean
// sharpness, so thin-but-valid fractals (Menger gaskets) aren't read as broken
// just because most of empty space escapes.
const DEFAULTS = {
  samples: 512,
  iters: 14,
  region: 2.5,
  bailout: 1e6,
  sharpThresh: 1.0,
  coverageGood: 0.18,
  // (a) sharpness tiebreaker — adds up to `tightnessCap` of wobble in the fully-
  // covered region so a crisp solid (sharp ≥ sharpFull) reads firmer than a
  // marginal one (sharp ≤ sharpLow). Capped so it can never alone declare a break.
  sharpFull: 12,
  sharpLow: 2,
  tightnessCap: 0.3,
  // (b) surface-march lean — orthographic sphere-march to find the rendered
  // surface's center of mass (a real lean even for space-filling solids).
  lean: "surface",
  leanGrid: 12,
  marchSteps: 64,
  surfaceEps: 0.02,
};

// Probe closures for one formula, derived from cpu.js makeOrbit — the SAME
// loop the CPU renderer marches (hybrid A/B schedule included), just with this
// module's shallower iters and historical radius bailout (the runner compares
// r², so the radius is squared going in). checkFinite gives the diagnostic
// nan/∞ semantics the sampler wants; the +c gate (addC||julia) is the runner's.
//   de(x,y,z)    — what a sphere-marcher reads. IFS: r/|w|; escape-time:
//                  0.5·ln(r)·r/|w|. NaN/divergence → +∞ (a miss).
//   probe(x,y,z) — DE "sharpness" S = ln(|w|/r): how hard the distance estimate
//                  collapsed toward a surface. S large → solid here; S ≲ 0 →
//                  blank here. Plus escape/nan flags.
function makeProbes(formula, o) {
  const orbit = makeOrbit(formula, {
    iters: o.iters,
    bailout2: o.bailout * o.bailout,
    checkFinite: true,
  });
  // #418 — mirror the GPU/render DE-family selector (see cpu.js makeDE). The
  // Surprise/Remix/Wander solidity oracle marches THIS de(); before the fix it
  // keyed off orbit.escape (= isEscapeTime), so a user deOption:0 on a non-escape
  // stack was probed with r/|w| here too — measure() then reported hits=0 (blank)
  // and vary.js isSound() rejected a formula the GPU renders solid. The bailout is
  // overridden above (bailout2), so only the finalize choice moves.
  const eff = effectiveDeOption(formula);
  const logFinalize = formula?.hybrid
    ? hybridDeFamily(formula) === "ifs"
      ? (formula.deOption ?? 2) === 0
      : true
    : eff === 0;
  return {
    de(x, y, z) {
      const res = orbit(x, y, z);
      if (res.nan) return Infinity;
      const de = logFinalize
        ? (0.5 * Math.log(Math.max(res.r, 1e-9)) * res.r) / res.aw
        : res.r / res.aw;
      return isFinite(de) ? de : Infinity;
    },
    probe(x, y, z) {
      const res = orbit(x, y, z);
      if (res.nan) return { nan: true };
      return {
        sharp: Math.log(res.aw / Math.max(res.r, 1e-9)), // sharp = -ln(DE)
        escaped: res.escaped,
      };
    },
  };
}

// Hybrid formulas (formula.hybrid, no objects[]) get honest treatment: family
// is the UNION verdict (hybridDeFamily, §3.3 — same as every render tier), and
// the supported-op check covers BOTH slots, so a broken/exotic slot B can no
// longer hide behind a healthy slot A.
const isHybrid = (f) => !!f?.hybrid && !isScene(f);
function familyOf(f) {
  return isHybrid(f) ? hybridDeFamily(f) : deFamily(f);
}
// Exported for the N-slot reader pin (HYBRID_NSLOT_SPEC.md §2.6) — asserts a
// bogus op in a late slot (C/D) is caught, not just slots A/B.
export function opsSupported(f) {
  // A hybrid checks EVERY slot (A + B/C/…) through the one accessor, so a
  // broken/exotic slot C/D can no longer hide behind a healthy slot A/B (§2.6).
  if (isHybrid(f)) {
    for (const slot of hybridSlots(f).slots) {
      const ops = slot.ops.filter((op) => !op.muted);
      if (!ops.every((op) => _SUPPORTED.has(op.key))) return false;
    }
    return true;
  }
  return activeOps(f).every((op) => _SUPPORTED.has(op.key));
}

// March one ray; return the first surface hit point or null (miss).
function marchRay(de, ox, oy, oz, dx, dy, dz, eps, tmax, maxSteps) {
  let t = 0;
  for (let k = 0; k < maxSteps; k++) {
    const d = de(ox + dx * t, oy + dy * t, oz + dz * t);
    if (!isFinite(d)) return null;
    if (d < eps) return { x: ox + dx * t, y: oy + dy * t, z: oz + dz * t };
    t += Math.max(d, eps * 0.5);
    if (t > tmax) return null;
  }
  return null;
}

// (b) Surface geometry — sphere-march six orthographic sweeps (±x/±y/±z) over
// the bounding cube and report two things about the rendered surface hits:
//   lean   = center of mass, normalized to ±1. For this palette the folds
//            re-center the attractor every iteration, so a STANDING fractal is
//            almost always centered (lean ~0) — a signed tilt is rarely a strong
//            signal here. Reported honestly anyway.
//   ext     = half-extent per axis (how far the body reaches along x/y/z), and
//   elong/axis = the anisotropy: which axis it stretches along and by how much
//            (surfFold/cylinderFold leave an axis free → elongated; a Mandelbox
//            is ~isotropic). THIS is the directional cue a "lean" animation
//            should lean on.
// CSG Phase 1b — an object spec viewed as a flat formula (for the per-object
// stability recursion below). Primitives carry no ops; only IFS objects recurse.
function objAsFormula(o) {
  return {
    ops: o.ops || [],
    iters: o.iters ?? 1,
    addC: !!o.addC,
    deOption: o.deOption ?? 2,
    julia: !!o.julia,
    juliaC: o.juliaC,
  };
}
const isScene = (f) => Array.isArray(f?.objects) && f.objects.length > 0;

// Analytic-primitive test + reach. objType is masked & 0xf everywhere else in
// core (renderer.js writeScene, renderer_gl.js, cpu.js makeSceneDE, sanitize.js)
// with types 0 IFS · 1 box · 2 sphere · 3 torus · 4 cylinder · 5 capsule ·
// 6 plane — a & 3 mask here mis-graded 3–6 (cylinder 4&3=0 → treated as an
// op-less IFS object → wobble 1 → false "toppling" reject in the vary.js gate).
const isPrimType = (t) => t >= 1 && t <= 6;
// Reach = approximate bounding radius in LOCAL units (caller applies uscale).
// Param meanings mirror cpu.js makeSceneDE (primParam/primParam2 per type);
// the intent is a cheap over-estimate of how far the surface extends from the
// object's origin for the scene lean/extent aggregate — not an exact SDF bound.
// Exported: build.ts's object-duplicate nudge (#430) reuses this same cheap
// estimate so a duplicate lands clear of the original regardless of how big
// the author has scaled the primitive — see the call site for the story.
export function primReach(ob, region) {
  const t = Number(ob.objType) & 0xf;
  const p = ob.primParam,
    p2 = ob.primParam2;
  switch (t) {
    case 1:
      return p ?? ob.halfExtent ?? ob.radius ?? 1; // box: half-extent
    case 2:
      return p ?? ob.radius ?? ob.halfExtent ?? 1; // sphere: radius
    case 3:
      return (p ?? 1) + (p2 ?? 0.25); // torus: major R + minor r
    case 4:
      return Math.hypot(p ?? 0.5, p2 ?? 0.5); // cylinder: √(r² + h²)
    case 5:
      return (p2 ?? 0.5) + (p ?? 0.3); // capsule: half-height h + cap r
    case 6:
      return region; // plane: unbounded — cap at the measured region
    default:
      return 1;
  }
}

// CSG Phase 1b — scene surface geometry (union semantics). The combined DE is a
// union (min over objects), so its surface is the union of the objects' surfaces:
// aggregate each object's world-placed reach (extent) and center of mass. Folds
// re-center an IFS body (lean ~0), so a primitive at an offset origin dominates
// the scene lean — which is the honest directional cue here.
function surfaceLeanScene(formula, opts = {}) {
  const o = { ...DEFAULTS, ...opts };
  const R = o.region;
  const empty = {
    x: 0,
    y: 0,
    z: 0,
    mag: 0,
    hits: 0,
    ext: { x: 0, y: 0, z: 0 },
    elong: 1,
    axis: null,
  };
  let sx = 0,
    sy = 0,
    sz = 0,
    hits = 0;
  let ex = 0,
    ey = 0,
    ez = 0; // world half-extents (max reach per axis)
  for (const ob of formula.objects) {
    const objType = Number(ob.objType) & 0xf;
    const tr = ob.transform || {};
    const org = tr.origin || ob.origin || [0, 0, 0];
    const uscale = tr.uscale ?? ob.uscale ?? 1;
    let reachX, reachY, reachZ, w; // reach = half-extent in world units; w = hit weight
    if (isPrimType(objType)) {
      const he = primReach(ob, R) * uscale;
      reachX = reachY = reachZ = he;
      w = 1;
    } else {
      const g = surfaceLean(objAsFormula(ob), o); // recurses into the flat path
      if (!g || !g.hits) continue;
      reachX = g.ext.x * R * uscale; // un-normalize (surfaceLean divides ext by R)
      reachY = g.ext.y * R * uscale;
      reachZ = g.ext.z * R * uscale;
      w = g.hits;
    }
    sx += org[0] * w;
    sy += org[1] * w;
    sz += org[2] * w;
    hits += w;
    ex = Math.max(ex, Math.abs(org[0]) + reachX);
    ey = Math.max(ey, Math.abs(org[1]) + reachY);
    ez = Math.max(ez, Math.abs(org[2]) + reachZ);
  }
  if (!hits) return empty;
  const x = sx / hits / R,
    y = sy / hits / R,
    z = sz / hits / R;
  const ext = { x: ex / R, y: ey / R, z: ez / R };
  const exts = [ext.x, ext.y, ext.z];
  const mx = Math.max(...exts),
    mn = Math.max(Math.min(...exts), 0.05);
  const elong = mx / mn;
  const mean = (exts[0] + exts[1] + exts[2]) / 3;
  const devs = exts.map((e) => Math.abs(e - mean));
  const axis =
    elong < 1.3 ? null : ["x", "y", "z"][devs.indexOf(Math.max(...devs))];
  return { x, y, z, mag: Math.hypot(x, y, z), hits, ext, elong, axis };
}

// CSG Phase 1b — scene stability (union semantics). A union stands if its most-
// solid member stands, so wobble = min over objects (analytic primitives are
// exact solids → wobble 0). Combines per-object measures; lean from the scene
// surface geometry. Unsupported iff some IFS object uses an unsupported op.
function measureScene(formula, opts = {}) {
  const o = { ...DEFAULTS, ...opts };
  let wobble = 1,
    coverage = 0,
    meanSharp = 0,
    escaped = 0,
    nan = 0;
  let any = false,
    allSupported = true;
  for (const ob of formula.objects) {
    const objType = Number(ob.objType) & 0xf;
    if (isPrimType(objType)) {
      // Analytic primitive: exact, bounded, solid → rock-solid, fully covered.
      wobble = Math.min(wobble, 0);
      coverage = Math.max(coverage, 1);
      meanSharp = Math.max(meanSharp, o.sharpFull);
      any = true;
      continue;
    }
    const m = measure(objAsFormula(ob), o);
    if (m.supported === false) {
      allSupported = false;
      continue;
    }
    any = true;
    wobble = Math.min(wobble, m.wobble);
    coverage = Math.max(coverage, m.coverage ?? 0);
    meanSharp = Math.max(meanSharp, m.meanSharp ?? 0);
    escaped = Math.max(escaped, m.escaped ?? 0);
    nan = Math.max(nan, m.nan ?? 0);
  }
  if (!any) {
    return {
      wobble: null,
      lean: null,
      leanMag: null,
      sharpness: null,
      escaped: null,
      nan: null,
      family: "scene",
      supported: false,
    };
  }
  const g =
    o.lean === false
      ? {
          x: 0,
          y: 0,
          z: 0,
          mag: 0,
          ext: { x: 0, y: 0, z: 0 },
          elong: 1,
          axis: null,
        }
      : surfaceLeanScene(formula, o);
  return {
    wobble,
    coverage,
    meanSharp,
    escaped,
    nan,
    family: "scene",
    supported: allSupported,
    lean: { x: g.x, y: g.y, z: g.z },
    leanMag: g.mag,
    ext: g.ext,
    elong: g.elong,
    axis: g.axis,
  };
}

export function surfaceLean(formula, opts = {}) {
  if (isScene(formula)) return surfaceLeanScene(formula, opts);
  const o = { ...DEFAULTS, ...opts };
  const family = familyOf(formula);
  const empty = {
    x: 0,
    y: 0,
    z: 0,
    mag: 0,
    hits: 0,
    ext: { x: 0, y: 0, z: 0 },
    elong: 1,
    axis: null,
  };
  if (family === "empty" || family === "mixed") return empty;
  if (!opsSupported(formula)) return null;
  const { de } = makeProbes(formula, o);
  const R = o.region,
    start = R + 0.5,
    tmax = 2 * start;
  let sx = 0,
    sy = 0,
    sz = 0,
    hits = 0;
  let ex = 0,
    ey = 0,
    ez = 0; // half-extents (max |coord| of surface hits)
  // Six sweeps — both faces of each axis — so near-side hits from opposite
  // directions cancel and a symmetric body's surface COM lands at ~0.
  for (let axis = 0; axis < 3; axis++) {
    const o1 = (axis + 1) % 3,
      o2 = (axis + 2) % 3;
    for (const sign of [1, -1]) {
      const dir = [0, 0, 0];
      dir[axis] = sign;
      for (let a = 0; a < o.leanGrid; a++) {
        for (let b = 0; b < o.leanGrid; b++) {
          const org = [0, 0, 0];
          org[axis] = -sign * start; // start on the face opposite the march
          org[o1] = (((a + 0.5) / o.leanGrid) * 2 - 1) * R;
          org[o2] = (((b + 0.5) / o.leanGrid) * 2 - 1) * R;
          const hit = marchRay(
            de,
            org[0],
            org[1],
            org[2],
            dir[0],
            dir[1],
            dir[2],
            o.surfaceEps,
            tmax,
            o.marchSteps,
          );
          if (hit) {
            sx += hit.x;
            sy += hit.y;
            sz += hit.z;
            hits++;
            ex = Math.max(ex, Math.abs(hit.x));
            ey = Math.max(ey, Math.abs(hit.y));
            ez = Math.max(ez, Math.abs(hit.z));
          }
        }
      }
    }
  }
  if (!hits) return empty;
  const x = sx / hits / R,
    y = sy / hits / R,
    z = sz / hits / R;
  const ext = { x: ex / R, y: ey / R, z: ez / R };
  const exts = [ext.x, ext.y, ext.z];
  // Elongation = longest/shortest half-extent, with the short axis floored so an
  // unbounded/free axis (ext ~ 0, e.g. surfFold's Z) gives a capped ratio, not ∞.
  const mx = Math.max(...exts),
    mn = Math.max(Math.min(...exts), 0.05);
  const elong = mx / mn;
  // The "odd" axis = the one whose extent deviates most from the mean (a free or
  // a stretched axis); only meaningful once the body is clearly anisotropic.
  const mean = (exts[0] + exts[1] + exts[2]) / 3;
  const devs = exts.map((e) => Math.abs(e - mean));
  const axis =
    elong < 1.3 ? null : ["x", "y", "z"][devs.indexOf(Math.max(...devs))];
  return { x, y, z, mag: Math.hypot(x, y, z), hits, ext, elong, axis };
}

// Full measure — the game-facing call. `opts` overrides DEFAULTS.
//   wobble  0 (rock-solid) → 1 (toppling/blank), smooth & monotone — the knob.
//   lean    direction the fractal's converged mass sits off-center (0 = balanced).
// Hybrid formulas are measured honestly: the probes run the real A/B schedule
// (slot A + slot B), the family verdict is the slot-union, and the supported
// check covers both slots — a jittered slot B that breaks the render now
// raises wobble instead of slipping past the vary.js ship-gate.
export function measure(formula, opts = {}) {
  if (isScene(formula)) return measureScene(formula, opts);
  const o = { ...DEFAULTS, ...opts };
  const family = familyOf(formula);

  // Exact short-circuits: nothing renders, or the DE families conflict → toppled.
  if (family === "empty" || family === "mixed") {
    return {
      wobble: 1,
      lean: { x: 0, y: 0, z: 0 },
      leanMag: 0,
      sharpness: -Infinity,
      escaped: 0,
      nan: 0,
      hits: 0,
      family,
      supported: true,
    };
  }
  if (!opsSupported(formula)) {
    return {
      wobble: null,
      lean: null,
      leanMag: null,
      sharpness: null,
      escaped: null,
      nan: null,
      hits: null,
      family,
      supported: false,
    };
  }

  const { probe } = makeProbes(formula, o);
  const R = o.region;
  let renderN = 0,
    escN = 0,
    nanN = 0,
    sharpSum = 0;

  for (let i = 1; i <= o.samples; i++) {
    const px = (halton(i, 2) * 2 - 1) * R;
    const py = (halton(i, 3) * 2 - 1) * R;
    const pz = (halton(i, 5) * 2 - 1) * R;
    const res = probe(px, py, pz);
    if (res.nan) {
      nanN++;
      continue;
    }
    if (res.escaped) escN++;
    if (res.sharp > o.sharpThresh) {
      renderN++;
      sharpSum += res.sharp;
    } // this point renders
  }

  const coverage = renderN / o.samples; // fraction of sampled space that renders a surface
  const nan = nanN / o.samples;

  // Wobble from coverage: solid (coverage ≥ coverageGood) → 0; nothing renders → 1.
  let wobble = Math.max(0, Math.min(1, 1 - coverage / o.coverageGood));
  // (a) sharpness tiebreaker — within the covered region, a crisp surface firms
  // up (low add) while a marginal one (DE barely sharpens) wobbles a little more.
  const meanSharp = renderN ? sharpSum / renderN : 0;
  const tightness =
    o.tightnessCap *
    Math.max(
      0,
      Math.min(1, (o.sharpFull - meanSharp) / (o.sharpFull - o.sharpLow)),
    );
  wobble = Math.max(wobble, renderN ? tightness : 0, nan);

  // (b) Surface geometry — sphere-marched lean + anisotropy (or fast/off per opts).
  const blank = {
    x: 0,
    y: 0,
    z: 0,
    mag: 0,
    hits: 0,
    ext: { x: 0, y: 0, z: 0 },
    elong: 1,
    axis: null,
  };
  const g = o.lean === false ? blank : (surfaceLean(formula, o) ?? blank);

  return {
    wobble,
    coverage,
    meanSharp,
    escaped: escN / o.samples,
    nan,
    // march-verified surface evidence (0 when opts.lean === false skipped the
    // sweep) — the sharp-probe coverage above can read "renders" at points no
    // real march reaches (broken-lattice class), hits cannot.
    hits: g.hits,
    family,
    supported: true,
    lean: { x: g.x, y: g.y, z: g.z },
    leanMag: g.mag,
    ext: g.ext,
    elong: g.elong,
    axis: g.axis, // anisotropy: stretch axis + ratio
  };
}

// ── Field streamlines (lab) ──────────────────────────────────────────────────
// Luminous particles advected along the fractal's distance-field gradient: a
// WebGPU compute pass moves N particles through the CURRENT formula's DE each
// frame (tangential swirl around the ∇DE direction + a spring onto a thin
// shell just off the surface), and a render pass composites them as additive
// glowing points over the finished frame.
//
// SCOPE (accepted divergence — IDEAS.md "Field streamlines (moonshot)"):
// WebGPU tier ONLY. There is deliberately no WebGL2/CPU mirror — the feature
// needs compute + storage buffers, and it is a lab toy, not a render mode. On
// the other tiers preview.setStreamlines() reports false and the app leaves
// the control dark.
//
// THE ONE HARD RULE — reuse the emitted DE, never duplicate it. The march
// shader's mapDE (core/shader.js buildWGSL) is the single source of truth for
// "what surface is on screen". extractDEWGSL() slices the emitted module text
// at the end of `fn mapDE(...)` — everything up to and including it is the
// self-contained DE cluster (Globals/ops/objects bindings, the op switch,
// hybrid/morph/scene walkers, leaf SDFs) — and buildStreamSimWGSL() appends a
// compute entry point in bind group 1 (sim state only; group 0 stays the
// march's own binding plan, so the SAME GPU buffers the render pass reads are
// bound unchanged). A hand-copied DE would rot on the first operator PR.
//
// OFF-PATH CONTRACT (the #125 doctrine): nothing here touches buildWGSL or the
// march pipelines. The controller is created lazily on first enable; while the
// feature is off there are zero allocations, zero passes and zero extra
// bindings — renderer.js's hooks are two `stream?.on` guards. The emitted
// march shader is byte-identical by construction (core/shader.js is not
// edited; streamlines.test.mjs pins the extraction against its output).
//
// Compositing: the renderer copies each presented frame into a private
// `composed` texture; particles draw additively on top of the swap chain. When
// the camera is idle (settled/accumulating), preview.js ticks an overlay-only
// frame — advect + blit(composed) + particles — so the swirl keeps moving with
// NO extra march work and the march pipeline is never stalled.

import { buildWGSL, usesOpAux, usesObjAux } from "./shader.js";

// Capacity ceiling: the particle buffer is allocated ONCE at enable (48 B ×
// 65536 = 3 MiB); the live count is a uniform, so the slider costs no realloc.
export const STREAM_MAX = 65536;
// vec4f posAge + vec4f axHue + vec4f misc (x = occlusion visibility,
// y = crease signal; zw reserved). Exported so the test can pin the WGSL
// struct and the JS allocation to the same number.
export const STREAM_PARTICLE_STRIDE = 48;
const WG_SIZE = 64;

// ── Deterministic offline exports (the PR #656 v1 tail) ──────────────────────
// v0 shared the sim state with the live view, so an export picked the flow up
// mid-swirl: the SAME flight rendered twice differed frame for frame. An export
// session now reseeds the particle field from a key the caller derives from the
// formula + settings, pre-rolls it a fixed number of steps, and puts the live
// flow back afterwards (see beginOffline/endOffline below).

/**
 * FNV-1a over the session key → the 32-bit spawn-stream base. Integer-only
 * (Math.imul), so no engine can disagree about it — the same rule (and the same
 * reason: this number decides a picture) as app/src/dailyseed.ts's hashSeed.
 */
export function streamSeedFor(key) {
  let h = 0x811c9dc5;
  const s = String(key ?? "");
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

// How far the freshly-seeded field is advanced before the FIRST exported frame,
// and at what step. 90 × 1/30 s = 3 virtual seconds — spawn dust has reached the
// shell and the braid has developed by then (R3 measured the live view
// re-converging in ~2 s after a world-scale change), while costing one modest
// burst of compute per EXPORT rather than per frame. Both are exported so
// streamlines.test.mjs can pin the numbers the seam promises.
export const STREAM_PREROLL_STEPS = 90;
export const STREAM_PREROLL_DT = 1 / 30;

// ── Pure codegen helpers (unit-tested in Node, no GPU) ───────────────────────

// Slice the emitted march WGSL at the end of `fn mapDE(...)`. Everything
// before it is the DE cluster and its bindings; everything after (coloring
// orbits, shading, vs/fs, capture) is render-only and must not reach the
// compute module. Brace-matching from the declaration is robust as long as no
// comment inside mapDE's body contains an unbalanced brace — pinned by
// streamlines.test.mjs against the real buildWGSL output.
export function extractDEWGSL(src) {
  const sig = "\nfn mapDE(p_rel: vec3f) -> f32 {";
  const at = src.indexOf(sig);
  if (at < 0) throw new Error("extractDEWGSL: mapDE declaration not found");
  let depth = 0;
  for (let i = src.indexOf("{", at + 1); i < src.length; i++) {
    const ch = src[i];
    if (ch === "{") depth++;
    else if (ch === "}") {
      depth--;
      if (depth === 0) return src.slice(0, i + 1) + "\n";
    }
  }
  throw new Error("extractDEWGSL: unbalanced braces in mapDE");
}

// The buildWGSL options for the SIM twin of a march feature descriptor
// (renderer.js activeFeat). Everything render-only is forced off — those flags
// either gate text AFTER mapDE (coloring/sreflect) or add texture bindings /
// Globals tail rows the sim never reads (envx/envMap/surfTex; the globals
// BUFFER is allocated at the ceiling, so a shorter struct binds fine). df64 /
// perturb are forced off too: particles don't need deep-zoom precision, and
// the f32 mapDE is the exact field the plain tier marches.
export function streamSimFeat(feat = {}) {
  return {
    numericDE: !!feat.numericDE,
    leaves:
      Array.isArray(feat.leaves) && feat.leaves.length ? feat.leaves : false,
    coloring: false,
    scene: !!feat.scene,
    hybrid: !!feat.hybrid,
    morph: !!feat.morph,
    ops: Array.isArray(feat.ops) ? feat.ops : null,
    df64: false,
    perturb: false,
    envx: false,
    sreflect: false,
    envMap: false,
    surfTex: false,
    capture: false,
  };
}

// Cache key for the sim pipeline — the sim-relevant slice of the variant key
// (renderer.js keyFor covers more bits; those are forced constant above).
export function streamSimKey(feat = {}) {
  const f = streamSimFeat(feat);
  const bits =
    (f.numericDE ? 1 : 0) |
    (f.scene ? 2 : 0) |
    (f.hybrid ? 4 : 0) |
    (f.morph ? 8 : 0);
  const ops = f.ops ? (f.ops.length ? f.ops.join(".") : "-") : "*";
  const leaves = Array.isArray(f.leaves) ? f.leaves.join(".") : "-";
  return `${bits}:${ops}:${leaves}`;
}

// The sim tail appended after the extracted DE cluster. Group 1 only — group 0
// belongs to the march bindings the prefix declares. All identifiers carry the
// sl/stream prefix so they can never collide with emitted op bodies.
const SIM_TAIL_WGSL = `
// ── Field streamlines sim (appended by buildStreamSimWGSL) ───────────────────
struct StreamP { posAge: vec4f, axHue: vec4f, misc: vec4f };
struct StreamSimU {
  dt      : f32,  // seconds (controller-clamped; fixed per frame offline)
  time    : f32,  // accumulated sim seconds (wall-clocked live, virtual offline)
  count   : f32,  // live particle count (threads past it exit)
  shell   : f32,  // target iso-offset off the surface (world units)
  swirl   : f32,  // tangential speed (world units / s)
  spring  : f32,  // shell attraction rate (1 / s)
  boundR  : f32,  // recycle when |p| leaves this sphere
  seed    : f32,  // per-frame respawn stream seed
  curl    : f32,  // curl-braid weight (0 = pure directed flow)
  curlFreq: f32,  // curl field spatial frequency (cycles per world unit-ish)
  edgeAcc : f32,  // tangential slowdown at edge emphasis (floored in advect)
  edgeK   : f32,  // edge signal gain on the normalized curvature variation
  align   : f32,  // 0 = per-particle streams .. 1 = one face-coherent current
  wscale  : f32,  // world scale (fractal bounding radius / 1.3 — preview probes it)
  pad1    : f32,
  pad2    : f32,
};
@group(1) @binding(0) var<storage, read_write> streamPs : array<StreamP>;
@group(1) @binding(1) var<uniform> SS : StreamSimU;

fn slPcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn slRand(seed: u32) -> f32 { return f32(slPcg(seed)) * 2.3283064e-10; }
fn slDir(r0: f32, r1: f32) -> vec3f {
  let z = r0 * 2.0 - 1.0;
  let a = r1 * 6.2831853;
  let r = sqrt(max(0.0, 1.0 - z * z));
  return vec3f(r * cos(a), r * sin(a), z);
}
// Divergence-free braid field: two octaves of ABC (Arnold–Beltrami–Childress)
// flow, slowly time-drifting. Each component depends only on the OTHER two
// coordinates, so divergence is analytically zero per octave — no sinks where
// particles could pile up artificially. Blended in only where the DE field is
// flat (see advect): on box faces cross(grad, axis) collapses to uniform
// parallel streams per particle, and this is what braids them.
fn slCurlV(p: vec3f) -> vec3f {
  let s = 0.35 * SS.time;
  let q = p * (SS.curlFreq / SS.wscale); // braids-per-object, world-size-free
  let a = vec3f(
    sin(q.z + s) + cos(q.y - s),
    sin(q.x - s) + cos(q.z + s),
    sin(q.y + s) + cos(q.x - s));
  let q2 = q * 2.17 + vec3f(5.2, 1.3, 8.4);
  let b = vec3f(
    sin(q2.z - s) + cos(q2.y + s),
    sin(q2.x + s) + cos(q2.z - s),
    sin(q2.y - s) + cos(q2.x + s));
  return a + 0.5 * b;
}
// Fresh particle: random point in a WIDE shell of the bounding ball (inner
// radius half the outer — an origin-heavy ball favored whatever face was
// nearest on boxy shapes; the shell distributes flow across the silhouette),
// random swirl axis, random hue, finite life so the flow keeps reseeding.
fn slSpawn(i: u32) -> StreamP {
  let b = slPcg(i ^ u32(SS.seed)) + i * 9781u;
  let dir = slDir(slRand(b), slRand(b + 1u));
  let rad = SS.boundR * SS.wscale * 0.85 * pow(0.125 + 0.875 * slRand(b + 2u), 0.3333);
  let axis = slDir(slRand(b + 3u), slRand(b + 4u));
  let age = 2.0 + 6.0 * slRand(b + 5u);
  return StreamP(
    vec4f(dir * rad, age),
    vec4f(axis, slRand(b + 6u)),
    vec4f(1.0, 0.0, 0.0, 0.0),
  );
}
@compute @workgroup_size(${WG_SIZE})
fn advect(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (f32(i) >= SS.count) { return; }
  var P = streamPs[i];
  // Knobs are UNIT-scale; the world multiplier is applied here, so tuning
  // values keep one meaning on every preset (and the preview's auto world
  // probe only ever writes wscale — it can never clobber an explicit knob).
  let shellW = SS.shell * SS.wscale;
  let swirlW = SS.swirl * SS.wscale;
  let boundW = SS.boundR * SS.wscale;
  if (P.posAge.w <= 0.0) { streamPs[i] = slSpawn(i); return; }
  let p = P.posAge.xyz;
  // 7 field taps: the value + the 6-tap central-difference stencil (the
  // splat-capture normal recipe). The SAME six taps also yield the per-axis
  // second differences — the free flat-vs-crease signal the box recipe uses.
  let d0 = mapDE(p);
  let h = max(2.0e-4 * SS.wscale, 0.15 * abs(d0));
  let dpx = mapDE(p + vec3f(h, 0.0, 0.0));
  let dmx = mapDE(p - vec3f(h, 0.0, 0.0));
  let dpy = mapDE(p + vec3f(0.0, h, 0.0));
  let dmy = mapDE(p - vec3f(0.0, h, 0.0));
  let dpz = mapDE(p + vec3f(0.0, 0.0, h));
  let dmz = mapDE(p - vec3f(0.0, 0.0, h));
  let g = vec3f(dpx - dmx, dpy - dmy, dpz - dmz) / (2.0 * h);
  let gl = length(g);
  // Degenerate gradient, numeric junk (NaN fails every compare), or out of
  // the bounding ball → recycle.
  if (!(abs(d0) < 1.0e8) || gl < 1.0e-5 || length(p) > boundW) {
    streamPs[i] = slSpawn(i);
    return;
  }
  let n = g / gl;
  var tang = cross(n, normalize(P.axHue.xyz));
  let tl = length(tang);
  if (tl < 1.0e-4) { streamPs[i] = slSpawn(i); return; }
  // DE-scale-normalized curvature variation (dimensionless): d2/(gl*h).
  // Measured on REAL fields (CPU stencil sweep, PR #656 R3): bulbs ~0.06,
  // Menger ~0.3, Mandelbox ~3 EVERYWHERE — the escape-time box field is
  // creased at every scale, so no reliable face-vs-edge detector exists at
  // stencil scale (the v0.1 flat-gate measured ≈0 across every box and the
  // curl braid never activated). rel is therefore an EDGE-EMPHASIS signal,
  // clamped per formula by edgeK, not a face classifier.
  let d2 = abs(dpx + dmx - 2.0 * d0) + abs(dpy + dmy - 2.0 * d0) +
           abs(dpz + dmz - 2.0 * d0);
  let rel = d2 / (gl * h);
  let edge = clamp(SS.edgeK * rel, 0.0, 1.0);
  // Face/world-stable tangent frame (review R3): a per-particle axis makes
  // every particle its own stream — uniform "rain" on a flat face. Blend the
  // per-particle direction toward ONE shared frame, cross(n, world-up), with
  // a per-particle +/- sign — a coherent two-way current along the face for
  // the curl braid to weave.
  let upA = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.y) > 0.9);
  let fd0 = cross(n, upA);
  let fl = length(fd0);
  let sgn = select(1.0, -1.0, P.axHue.w > 0.5);
  var base = tang / tl;
  if (fl > 1.0e-4) {
    let mixed = mix(base, (sgn / fl) * fd0, clamp(SS.align, 0.0, 1.0));
    let ml = length(mixed);
    if (ml > 1.0e-3) { base = mixed / ml; }
  }
  // Divergence-free curl braid — ALWAYS blended in (measured above: there is
  // no flat detector to gate it on, and the braid is what kills the rain
  // look; the probe confirms bulbs keep their character under it).
  var cv = slCurlV(p);
  cv = cv - n * dot(cv, n); // keep the braid on the surface
  var vt = swirlW *
    ((1.0 - 0.5 * clamp(SS.curl, 0.0, 1.0)) * base + SS.curl * cv);
  // Edge emphasis is a FLOORED slowdown (review R3: v0.1's unbounded
  // density ∝ 1/speed pileup at saturated edge signal read as congregating
  // static dots) …
  vt = vt * max(0.55, 1.0 - SS.edgeAcc * edge);
  var vel = vt - (SS.spring * (d0 - shellW)) * n;
  let vmax = max(swirlW * 4.0, SS.wscale);
  vel = vel * min(1.0, vmax / max(length(vel), 1.0e-6));
  let dt = min(SS.dt, 0.033);
  let np = p + vel * dt;
  // … and slow particles LIVE SHORTER — moving trains, not piles: below-par
  // tangential speed ages up to 2.5x faster, and SUSTAINED near-stagnation
  // (curl zeros, corner traps — divergence-free does not mean zero-free)
  // respawns outright via the misc.z frame counter.
  let srel = clamp(length(vt) / max(swirlW, 1.0e-4), 0.0, 1.0);
  let ageDt = dt * (1.0 + 1.5 * (1.0 - srel));
  var slowN = select(0.0, P.misc.z + 1.0, srel < 0.18);
  if (slowN > 40.0) { streamPs[i] = slSpawn(i); return; }
  // Occlusion: soft-shadow march FROM THE PARTICLE TOWARD THE EYE, through
  // the SAME mapDE — t is distance from the PARTICLE, so the penumbra factor
  // k*dd/t widens with distance exactly like a standard soft shadow. (v0.1
  // marched eye->particle and divided by distance-from-EYE: at a normal view
  // distance that capped every particle's visibility at ~k*shell/dist and
  // the whole overlay faded to near-invisible — the R3 probe's first find.)
  // Starts a margin out so the particle's own shell never self-occludes;
  // <=24 steps with a floor step keeps the cost bounded.
  var vis = 1.0;
  let ro = G.camPos.xyz;
  let pv = ro - np;
  let pd = length(pv);
  let margin = max(2.5 * shellW, 0.03 * SS.wscale);
  // Conservative-DE correction: a Mandelbox-class escape field under-reports
  // distance by ~|grad d| (probe-measured gl ~0.35 there vs ~1.0 on true
  // SDFs like Menger/Ring & Stones), which strangled the penumbra ratio and
  // faded 99.8% of Mandelbox particles (R3 census: 16292/16328 at vis<0.1).
  // Dividing the clearance by the LOCAL gl restores the true-distance scale;
  // the 0.2 floor caps the correction where the gradient is nearly flat.
  let occAmp = 1.0 / max(gl, 0.2);
  // Shell allowance (the Amazing-Box grazing fix): a particle on a face seen
  // at a glancing angle has an eye ray that HUGS its own surface at ~shell
  // clearance for a long stretch, and the raw penumbra read that hug as
  // occlusion — whole oblique faces faded out (probe: a round head-on-only
  // sparkle blob). Near the particle (a 24-shell window) clearances up to
  // ~3/4 shell are the particle's own surface and are forgiven; far from it
  // the allowance is zero and thin geometry still hard-blocks.
  let occEps = 0.75 * shellW;
  if (pd > margin + 1.0e-3) {
    let rd = pv / pd;
    var t = margin;
    for (var k: u32 = 0u; k < 24u; k = k + 1u) {
      if (t >= pd - 1.0e-2) { break; }
      let ddRaw = mapDE(np + rd * t);
      let allow = occEps * clamp(1.0 - t / (24.0 * shellW), 0.0, 1.0);
      let c = ddRaw * occAmp - allow;
      if (c > 0.0) {
        vis = min(vis, clamp(12.0 * c / t, 0.0, 1.0));
      } else if (allow <= 1.0e-6) {
        vis = 0.0; // truly blocked past the allowance window
      }
      if (vis < 0.02) { break; }
      t = t + max(ddRaw, pd * 0.04);
    }
  }
  // Converging spawn dust stays dim until it reaches the shell — without
  // this the in-flight cloud reads as noise sprinkled around the silhouette.
  vis = vis / (1.0 + (6.0 / SS.wscale) * max(0.0, d0 - shellW));
  streamPs[i] = StreamP(
    vec4f(np, P.posAge.w - ageDt),
    P.axHue,
    vec4f(vis, edge, slowN, 0.0),
  );
}
`;

// DE cluster (extracted from a buildWGSL() output) + the sim tail → the full
// compute module text. Pure — streamlines.test.mjs asserts the module carries
// the map function, the compute entry, and no render entry points.
export function buildStreamSimWGSL(deCore) {
  return deCore + SIM_TAIL_WGSL;
}

// Particle render module — formula-independent (no DE). Reads the SAME globals
// buffer the march writes (a leading-rows view of the Globals struct: layout
// is all vec4 rows, and binding size ≥ struct size always holds), projects
// particles through the march camera (perspective + ortho, matching the fs
// ray-gen exactly), and shades with the march's cosine palette words.
export function buildStreamDrawWGSL() {
  return `
struct StreamP { posAge: vec4f, axHue: vec4f, misc: vec4f };
struct StreamGlobals {
  res     : vec4f,   // x,y = resolution px ; z = fov(rad)
  camPos  : vec4f,   // xyz = eye (deep-zoom residual — particles live in p_rel)
  camFwd  : vec4f,   // w = ortho half-height (#441; 0 = perspective)
  camRight: vec4f,
  camUp   : vec4f,
  ctrl    : vec4u,
  prm     : vec4f,
  colA    : vec4f,
  colB    : vec4f,
  bgc     : vec4f,
  jc      : vec4f,
  palA    : vec4f,   // cosine palette a ; .w > 0.5 = palette on
  palB    : vec4f,
  palC    : vec4f,
  palD    : vec4f,
};
struct StreamDrawU { intensity: f32, sizePx: f32, count: f32, wscale: f32 };
@group(0) @binding(0) var<uniform> G : StreamGlobals;
@group(0) @binding(1) var<storage, read> streamPs : array<StreamP>;
@group(0) @binding(2) var<uniform> SD : StreamDrawU;

fn slPal(t: f32) -> vec3f {
  if (G.palA.w > 0.5) {
    return clamp(G.palA.xyz + G.palB.xyz * cos(6.2831853 * (G.palC.xyz * t + G.palD.xyz)),
                 vec3f(0.0), vec3f(1.0));
  }
  return clamp(mix(G.colA.xyz, G.colB.xyz, t), vec3f(0.0), vec3f(1.0));
}

struct SVOut {
  @builtin(position) clip : vec4f,
  @location(0) q    : vec2f,
  @location(1) col  : vec3f,
  @location(2) fade : f32,
};

@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> SVOut {
  var o: SVOut;
  o.clip = vec4f(0.0, 0.0, 2.0, 1.0); // default: clipped away
  o.q = vec2f(0.0);
  o.col = vec3f(0.0);
  o.fade = 0.0;
  if (f32(ii) >= SD.count) { return o; }
  let P = streamPs[ii];
  let age = P.posAge.w;
  // Dead, or fully occluded (misc.x = the sim's eye-ray visibility): skip the
  // quad entirely rather than rasterizing an invisible sprite.
  if (age <= 0.0 || P.misc.x <= 0.004) { return o; }
  let v = P.posAge.xyz - G.camPos.xyz;
  let z = dot(v, G.camFwd.xyz);
  let planetK = G.camRight.w;
  let equirectS = G.camUp.w;
  // TINY PLANET and 360 EQUIRECT rays reach BEHIND the eye, so the in-front
  // cull below is only right for the two flat projections; under either wide
  // map a particle at z <= 0 is legitimately on screen (equirect puts it at
  // the frame's left/right thirds, the planet in its sky ring).
  if (z <= 1.0e-4 && planetK <= 0.0 && equirectS <= 0.0) { return o; }
  let x = dot(v, G.camRight.xyz);
  let y = dot(v, G.camUp.xyz);
  let aspect = G.res.x / G.res.y;
  let orthoH = G.camFwd.w;
  var ndc = vec2f(0.0);
  if (orthoH > 0.0) {
    ndc = vec2f(x / (orthoH * aspect), y / orthoH);
  } else if (planetK > 0.0) {
    // FORWARD stereographic projection — the inverse of the ray-gen map in
    // core/shader.js. With d the unit direction to the particle and
    // (dx,dy,dz) its basis components, the plane point is
    //   (u,v) = (dx, dy) / (1 + dz)
    // (substitute the ray-gen formula to check: 1+dz = 2/(1+q) collapses it).
    // A uniform branch is fine HERE — this is the overlay's vertex shader,
    // once per particle quad, not the march (the perf doctrine's subject).
    let d = normalize(v);
    let dz = dot(d, G.camFwd.xyz);
    let den = 1.0 + dz;
    if (den <= 1.0e-4) { return o; } // the antipode: infinitely far out
    let pu = dot(d, G.camRight.xyz) / den;
    let pv = dot(d, G.camUp.xyz) / den;
    ndc = vec2f(pu / (planetK * aspect), pv / planetK);
  } else if (equirectS > 0.0) {
    // FORWARD equirectangular projection — the inverse of the lat-long
    // ray-gen in core/shader.js (rayGenEquirect): with d the unit direction
    // to the particle, lon = atan2(d·right, d·fwd), lat = asin(d·up), and
    // ndc.x = lon / (eqS·aspect) because the ray-gen reads lon = wx·eqS with
    // wx = ndc.x·aspect (the overlay never draws under a tile window). Same
    // uniform-branch licence as the planet arm above: this is the overlay's
    // vertex shader, not the march.
    let d = normalize(v);
    let lon = atan2(dot(d, G.camRight.xyz), dot(d, G.camFwd.xyz));
    let lat = asin(clamp(dot(d, G.camUp.xyz), -1.0, 1.0));
    ndc = vec2f(lon / (equirectS * aspect), lat / 1.5707963267948966);
  } else {
    let tanF = tan(0.5 * G.res.z);
    ndc = vec2f(x / (z * tanF * aspect), y / (z * tanF));
  }
  let corner = vec2f(f32(vi & 1u) * 2.0 - 1.0, f32((vi >> 1u) & 1u) * 2.0 - 1.0);
  // Sprite size tracks world-units-per-pixel: bigger up close, scaled by the
  // sim world. max(z, 0.05) already floors the divisor; under the planet z can
  // be negative (behind the eye) and the same floor keeps the sprite sane.
  //
  // The wscale term carried a 0.25 FLOOR while the controller's own wscale
  // clamp made anything smaller unreachable. Deep-zoom spawning removed that
  // clamp (the sim world now follows the frustum), and the stale floor then
  // divided a 0.25 world by a ~0.02 depth and pinned every sprite at the 2.5
  // ceiling — the probe frame came back a wash of glow instead of flow. With
  // the floor gone the ratio is ~1 at BOTH framings, so deep zoom reads at the
  // same density as a normal one; every preset's own camera sits above 0.25
  // anyway, so those frames are unchanged.
  let px = SD.sizePx * clamp((2.0 * SD.wscale) / max(z, 0.05), 0.35, 2.5);
  o.clip = vec4f(ndc + corner * px * 2.0 / G.res.xy, 0.0, 1.0);
  o.q = corner;
  // Palette color, slightly brightened on creases (misc.y — the box-family
  // edge signal) so the accumulated edge flow reads as glowing seams; faded
  // by remaining life AND the soft occlusion visibility (misc.x) so flow
  // disappearing behind geometry reads as natural occlusion.
  o.col = slPal(fract(P.axHue.w + 0.15 * age)) * SD.intensity *
          (1.0 + 0.6 * P.misc.y);
  o.fade = clamp(age, 0.0, 1.0) * P.misc.x;
  return o;
}

@fragment fn fs(o: SVOut) -> @location(0) vec4f {
  let r2 = dot(o.q, o.q);
  if (r2 > 1.0) { discard; }
  let glow = exp(-3.5 * r2) + 0.22 * exp(-1.3 * r2);
  return vec4f(o.col * (glow * o.fade), 0.0);
}
`;
}

// Fullscreen blit of the private composed texture back onto the swap chain —
// the idle tick's "repaint the last frame" pass (textureLoad, no sampler).
export function buildStreamBlitWGSL() {
  return `
@group(0) @binding(0) var streamSrc : texture_2d<f32>;
@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  var tri = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(tri[vi], 0.0, 1.0);
}
@fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
  return textureLoad(streamSrc, vec2i(p.xy), 0);
}
`;
}

// ── The controller (renderer-side; GPU objects live here) ────────────────────
// io: { format, globalsBuf, opsBuf, objectsBuf, opAuxBuf, objAuxBuf } — the
// renderer's OWN per-frame buffers, bound read-only by the sim so the field it
// advects through is exactly the field on screen, with zero re-upload.
export function createStreamlines(device, io) {
  const { format, globalsBuf, opsBuf, objectsBuf, opAuxBuf, objAuxBuf } = io;
  const cfg = {
    on: false,
    count: 24576,
    intensity: 0.6,
    sizePx: 2.75,
    swirl: 0.6,
    spring: 5.0,
    shell: 0.02,
    boundR: 2.3,
    // Motion recipe knobs (PR #656 R3 — tuned with scripts/streamlines-probe
    // on real WebGPU frames): always-on curl braid + its frequency, the
    // floored edge slowdown + its signal gain, and the face-coherent blend.
    curl: 0.5,
    curlFreq: 3.2,
    edgeAcc: 0.45,
    edgeK: 0.35,
    align: 0.55,
    wscale: 1, // world scale — preview's CPU-DE radius probe sets it per formula
  };

  // Base sim state — allocated on first enable, kept for the session.
  let partsBuf = null; // STREAM_MAX × 32 B storage (zero ages → all respawn)
  let simUBuf = null; // 8 × f32
  let drawUBuf = null; // 4 × f32

  // Offline-export session state (deterministic exports). All of it is inert —
  // and saveBuf unallocated — until a caller arms an export with beginOffline.
  let offSeed = 0; // spawn-stream base while offline (0 = live wall-clock flow)
  let offPending = false; // reset + pre-roll owed before the next offline frame
  let offSavedTime = null; // the live sim clock, parked for the export's duration
  let saveBuf = null; // GPU snapshot of the live particle field (lazy, 3 MiB)
  let zeroBytes = null; // reusable zero fill for the reset

  // Pipelines: draw + blit are formula-independent (built once, async); the
  // sim compute is keyed by the march variant it mirrors and rebuilt (async,
  // never blocking a frame) when the formula's op-set/features change.
  let drawPl = null,
    drawBind = null,
    drawBuilding = false;
  let blitPl = null,
    blitBind = null,
    blitBuilding = false;
  let simPl = null,
    simBind0 = null,
    simBind1 = null,
    simKey = null,
    simBuilding = false;

  // The composed copy of the last presented frame (idle-tick substrate).
  let composed = null,
    composedView = null,
    composedW = 0,
    composedH = 0,
    copied = false;

  // Failure surfacing (the probe workflow's first lesson: a silently-caught
  // createPipelineAsync rejection reads as "no particles, no errors"). Lab
  // feature → console.warn is appropriate, and info() lets the headless
  // probe / diag read the pipeline states directly.
  const errors = [];
  function fail(what, e) {
    const msg = `${what}: ${String(e?.message || e).slice(0, 500)}`;
    errors.push(msg);
    try {
      console.warn("[streamlines]", msg);
    } catch {
      /* consoles can be stubbed */
    }
  }

  let lastT = 0,
    simTime = 0;

  function ensureBase() {
    if (partsBuf) return;
    partsBuf = device.createBuffer({
      size: STREAM_MAX * STREAM_PARTICLE_STRIDE,
      // COPY_SRC: the probe census (readStats) copies the buffer out.
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    });
    simUBuf = device.createBuffer({
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    drawUBuf = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  function ensureComposed(w, h) {
    if (composed && composedW === w && composedH === h) return;
    composed?.destroy();
    composed = device.createTexture({
      size: [w, h],
      format,
      usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
    });
    composedView = composed.createView();
    composedW = w;
    composedH = h;
    copied = false;
    blitBind = blitPl
      ? device.createBindGroup({
          layout: blitPl.getBindGroupLayout(0),
          entries: [{ binding: 0, resource: composedView }],
        })
      : null;
  }

  function ensurePipelines(feat) {
    ensureBase();
    if (!drawPl && !drawBuilding && device.createRenderPipelineAsync) {
      drawBuilding = true;
      const m = device.createShaderModule({ code: buildStreamDrawWGSL() });
      device
        .createRenderPipelineAsync({
          layout: "auto",
          vertex: { module: m, entryPoint: "vs" },
          fragment: {
            module: m,
            entryPoint: "fs",
            targets: [
              {
                format,
                blend: {
                  color: {
                    srcFactor: "one",
                    dstFactor: "one",
                    operation: "add",
                  },
                  alpha: {
                    srcFactor: "one",
                    dstFactor: "one",
                    operation: "add",
                  },
                },
              },
            ],
          },
          primitive: { topology: "triangle-strip" },
        })
        .then(
          (pl) => {
            drawPl = pl;
            drawBind = device.createBindGroup({
              layout: pl.getBindGroupLayout(0),
              entries: [
                { binding: 0, resource: { buffer: globalsBuf } },
                { binding: 1, resource: { buffer: partsBuf } },
                { binding: 2, resource: { buffer: drawUBuf } },
              ],
            });
          },
          (e) => fail("draw pipeline", e), // overlay draw stays disabled
        )
        .finally(() => (drawBuilding = false));
    }
    if (!blitPl && !blitBuilding && device.createRenderPipelineAsync) {
      blitBuilding = true;
      const m = device.createShaderModule({ code: buildStreamBlitWGSL() });
      device
        .createRenderPipelineAsync({
          layout: "auto",
          vertex: { module: m, entryPoint: "vs" },
          fragment: { module: m, entryPoint: "fs", targets: [{ format }] },
          primitive: { topology: "triangle-list" },
        })
        .then(
          (pl) => {
            blitPl = pl;
            if (composedView)
              blitBind = device.createBindGroup({
                layout: pl.getBindGroupLayout(0),
                entries: [{ binding: 0, resource: composedView }],
              });
          },
          (e) => fail("blit pipeline", e),
        )
        .finally(() => (blitBuilding = false));
    }
    // Sim variant — mirror the frame's march variant (the DE reuse contract).
    const key = streamSimKey(feat);
    if (key !== simKey && !simBuilding && device.createComputePipelineAsync) {
      simBuilding = true;
      const o = streamSimFeat(feat);
      const src = buildStreamSimWGSL(extractDEWGSL(buildWGSL(o)));
      const m = device.createShaderModule({ code: src });
      // WGSL diagnostics land on the MODULE, not the pipeline promise — read
      // them explicitly or a bad sim shader is invisible (probe lesson).
      m.getCompilationInfo?.().then((ci) => {
        const errs = (ci?.messages || []).filter((x) => x.type === "error");
        if (errs.length)
          fail(
            `sim WGSL (${key})`,
            errs
              .map((x) => `${x.lineNum}:${x.linePos} ${x.message}`)
              .join(" | "),
          );
      });
      device
        .createComputePipelineAsync({
          layout: "auto",
          compute: { module: m, entryPoint: "advect" },
        })
        .then(
          (pl) => {
            simPl = pl;
            simKey = key;
            simBind0 = device.createBindGroup({
              layout: pl.getBindGroupLayout(0),
              entries: [
                { binding: 0, resource: { buffer: globalsBuf } },
                { binding: 1, resource: { buffer: opsBuf } },
                { binding: 2, resource: { buffer: objectsBuf } },
                // Same auto-layout-prunes-unread rule as renderer.marchBind:
                // bind the overflow lanes ONLY when this variant declared them.
                ...(usesOpAux(o.ops)
                  ? [{ binding: 7, resource: { buffer: opAuxBuf } }]
                  : []),
                ...(usesObjAux(o.leaves)
                  ? [{ binding: 8, resource: { buffer: objAuxBuf } }]
                  : []),
              ],
            });
            simBind1 = device.createBindGroup({
              layout: pl.getBindGroupLayout(1),
              entries: [
                { binding: 0, resource: { buffer: partsBuf } },
                { binding: 1, resource: { buffer: simUBuf } },
              ],
            });
          },
          (e) => {
            // Failed compile: latch the key so we don't grind rebuilds; the
            // overlay simply freezes for this formula (lab-grade degradation).
            simKey = key;
            simPl = null;
            fail(`sim pipeline (${key})`, e);
          },
        )
        .finally(() => (simBuilding = false));
    }
  }

  function stepClock() {
    const now =
      typeof performance !== "undefined" ? performance.now() : Date.now();
    const dt = lastT ? Math.min(0.05, Math.max(0, (now - lastT) / 1000)) : 0;
    lastT = now;
    simTime += dt;
    return dt;
  }

  function count() {
    return Math.max(0, Math.min(STREAM_MAX, Math.round(cfg.count)));
  }

  // The knob signature that rides the export seed. Two exports at DIFFERENT
  // settings are different pictures and must not share a spawn stream; two at
  // the same settings must. (wscale is in here on purpose — preview pins it to
  // the formula's object scale for the duration of an export, so it is a
  // property of the formula, not of wherever the live camera happened to be.)
  const cfgSig = () =>
    [
      count(),
      cfg.shell,
      cfg.swirl,
      cfg.spring,
      cfg.boundR,
      cfg.curl,
      cfg.curlFreq,
      cfg.edgeAcc,
      cfg.edgeK,
      cfg.align,
      cfg.wscale,
    ].join(",");

  function encodeAdvect(enc, dt) {
    if (!simPl || !simBind0 || !simBind1) return;
    const n = count();
    if (!n) return;
    device.queue.writeBuffer(
      simUBuf,
      0,
      new Float32Array([
        dt,
        simTime,
        n,
        cfg.shell,
        cfg.swirl,
        cfg.spring,
        cfg.boundR,
        // Respawn stream. Live, it rides the wall-clocked sim clock; under an
        // armed export it rides that clock reset to 0 PLUS the session's
        // formula-derived base, so the whole spawn sequence is a pure function
        // of the key and the frame index.
        (simTime * 977 + 1 + offSeed) % 4294967296,
        cfg.curl,
        cfg.curlFreq,
        cfg.edgeAcc,
        cfg.edgeK,
        cfg.align,
        cfg.wscale,
        0,
        0,
      ]),
    );
    const pass = enc.beginComputePass();
    pass.setPipeline(simPl);
    pass.setBindGroup(0, simBind0);
    pass.setBindGroup(1, simBind1);
    pass.dispatchWorkgroups(Math.ceil(n / WG_SIZE));
    pass.end();
  }

  function encodePoints(enc, view) {
    if (!drawPl || !drawBind) return;
    const n = count();
    if (!n) return;
    device.queue.writeBuffer(
      drawUBuf,
      0,
      new Float32Array([cfg.intensity, cfg.sizePx, n, cfg.wscale]),
    );
    const pass = enc.beginRenderPass({
      colorAttachments: [{ view, loadOp: "load", storeOp: "store" }],
    });
    pass.setPipeline(drawPl);
    pass.setBindGroup(0, drawBind);
    pass.draw(4, n);
    pass.end();
  }

  // Reset the field to the armed export's seed and pre-roll it. Runs on its OWN
  // encoders, one submit per step, and NOT into the caller's encoder — the sim
  // uniform is a single buffer written by queue.writeBuffer, so N advect passes
  // batched into one submit would all read the LAST write (every step would
  // share one clock, one seed, one dt) and the pre-roll would silently degrade
  // to a single step of N× the work. Queue order still puts all of it ahead of
  // the frame the caller is encoding.
  function resetAndPreroll() {
    const n = count();
    if (!n || !partsBuf) return;
    // Age <= 0 IS the shader's respawn signal, so a plain zero fill makes every
    // thread take slSpawn(i) with the export's seed on the very first step —
    // no separate init kernel, no second copy of the spawn recipe.
    const bytes = n * STREAM_PARTICLE_STRIDE;
    if (!zeroBytes || zeroBytes.byteLength < bytes) zeroBytes = new Uint8Array(bytes);
    device.queue.writeBuffer(partsBuf, 0, zeroBytes, 0, bytes);
    for (let i = 0; i < STREAM_PREROLL_STEPS; i++) {
      simTime += STREAM_PREROLL_DT;
      const e = device.createCommandEncoder();
      encodeAdvect(e, STREAM_PREROLL_DT);
      device.queue.submit([e.finish()]);
    }
  }

  return {
    get on() {
      return cfg.on;
    },
    // Probe-only particle census (async GPU readback — never on a frame
    // path): where ARE the particles? Answers the "enabled, no errors, no
    // pixels" class directly: alive counts, visibility histogram, radius
    // spread, stagnation counters.
    async readStats() {
      if (!partsBuf) return null;
      const n = count();
      const bytes = n * STREAM_PARTICLE_STRIDE;
      const rb = device.createBuffer({
        size: bytes,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(partsBuf, 0, rb, 0, bytes);
      device.queue.submit([enc.finish()]);
      await rb.mapAsync(GPUMapMode.READ);
      const ab = rb.getMappedRange(); // ONE range, two views (overlap is illegal)
      const f = new Float32Array(ab);
      // Exact state digest — FNV-1a over the raw particle words. The census
      // numbers below are rounded summaries and two DIFFERENT fields can share
      // them; this is the number the determinism probe compares between two
      // export runs, and only bit-identical state produces it twice.
      const u = new Uint32Array(ab);
      let dg = 0x811c9dc5;
      for (let i = 0; i < u.length; i++) dg = Math.imul(dg ^ u[i], 0x01000193);
      const s = {
        n,
        alive: 0,
        visLo: 0, // vis < 0.1 — effectively invisible
        visMid: 0,
        visHi: 0, // vis > 0.5
        rSum: 0,
        ageSum: 0,
        slowSum: 0,
        edgeSum: 0,
      };
      for (let i = 0; i < n; i++) {
        const o = i * 12;
        const age = f[o + 3];
        if (!(age > 0)) continue;
        s.alive++;
        s.rSum += Math.hypot(f[o], f[o + 1], f[o + 2]);
        s.ageSum += age;
        const vis = f[o + 8];
        if (vis < 0.1) s.visLo++;
        else if (vis > 0.5) s.visHi++;
        else s.visMid++;
        s.edgeSum += f[o + 9];
        s.slowSum += f[o + 10];
      }
      rb.unmap();
      rb.destroy();
      const a = Math.max(1, s.alive);
      return {
        n,
        digest: (dg >>> 0).toString(16),
        alive: s.alive,
        visLo: s.visLo,
        visMid: s.visMid,
        visHi: s.visHi,
        meanR: +(s.rSum / a).toFixed(3),
        meanAge: +(s.ageSum / a).toFixed(2),
        meanEdge: +(s.edgeSum / a).toFixed(3),
        meanSlow: +(s.slowSum / a).toFixed(1),
      };
    },
    // Pipeline states + captured failures — read by the diag panel and the
    // headless probe (a lab overlay that silently fails is indistinguishable
    // from "no particles" without this).
    info() {
      return {
        on: cfg.on,
        simReady: !!simPl,
        simKey,
        simBuilding,
        drawReady: !!drawPl,
        blitReady: !!blitPl,
        composed: copied ? [composedW, composedH] : null,
        wscale: cfg.wscale,
        offline: offSavedTime != null,
        errors: [...errors],
      };
    },
    // Merge config. Numbers are clamped defensively — this is app-facing.
    set(o = {}) {
      const wasOn = cfg.on;
      if (typeof o.on === "boolean") cfg.on = o.on;
      if (Number.isFinite(o.count))
        cfg.count = Math.max(0, Math.min(STREAM_MAX, Math.round(o.count)));
      if (Number.isFinite(o.intensity))
        cfg.intensity = Math.max(0, Math.min(4, o.intensity));
      if (Number.isFinite(o.sizePx))
        cfg.sizePx = Math.max(0.5, Math.min(12, o.sizePx));
      if (Number.isFinite(o.swirl))
        cfg.swirl = Math.max(0, Math.min(4, o.swirl));
      if (Number.isFinite(o.spring))
        cfg.spring = Math.max(0, Math.min(30, o.spring));
      if (Number.isFinite(o.shell))
        cfg.shell = Math.max(0, Math.min(0.5, o.shell));
      if (Number.isFinite(o.boundR))
        cfg.boundR = Math.max(0.5, Math.min(20, o.boundR));
      if (Number.isFinite(o.curl))
        cfg.curl = Math.max(0, Math.min(1.5, o.curl));
      if (Number.isFinite(o.curlFreq))
        cfg.curlFreq = Math.max(0.5, Math.min(8, o.curlFreq));
      if (Number.isFinite(o.edgeAcc))
        cfg.edgeAcc = Math.max(0, Math.min(0.95, o.edgeAcc));
      if (Number.isFinite(o.edgeK))
        cfg.edgeK = Math.max(0, Math.min(3, o.edgeK));
      if (Number.isFinite(o.align))
        cfg.align = Math.max(0, Math.min(1, o.align));
      // Floor at 1e-4, not 0.1: preview sizes the sim world to the SMALLER of
      // the formula's bounding radius and the camera frustum (the deep-zoom
      // spawning tail), and at ×10³ zoom that frustum is far below 0.1 world
      // units. A 0.1 floor pinned the spawn ball to a sphere ~thousands of
      // frustums wide and put every particle off screen — the reported bug.
      if (Number.isFinite(o.wscale))
        cfg.wscale = Math.max(1e-4, Math.min(64, o.wscale));
      if (cfg.on) {
        ensureBase();
        if (!wasOn) lastT = 0; // don't integrate the time the overlay spent off
      }
    },
    // Live-frame hook (renderer drawTo/drawAccum, live canvas only): advect,
    // snapshot the clean frame into `composed`, then draw the points on top —
    // in that order, so composed never contains particles (no ghost trails
    // when the idle tick re-blits it).
    encodeLive(enc, tex, feat) {
      if (!cfg.on) return;
      ensurePipelines(feat);
      ensureComposed(tex.width, tex.height);
      encodeAdvect(enc, stepClock());
      enc.copyTextureToTexture({ texture: tex }, { texture: composed }, [
        tex.width,
        tex.height,
      ]);
      copied = true;
      encodePoints(enc, tex.createView());
    },
    // Idle overlay frame is only safe once a live frame has populated the
    // composed snapshot at the CURRENT canvas size and the pipelines are warm.
    canIdle(w, h) {
      return (
        cfg.on &&
        copied &&
        composedW === w &&
        composedH === h &&
        !!blitPl &&
        !!blitBind &&
        !!drawPl
      );
    },
    // ── Offline export session (deterministic exports) ───────────────────────
    // Arm an export: park the live flow and switch the sim onto a seed derived
    // from `key` (the caller's formula + settings digest). The actual reset and
    // pre-roll are DEFERRED to the first offline frame — by then captureFrame's
    // writeFrame has published the export's own globals, so the pre-roll runs
    // against the frame the export is actually about rather than whatever the
    // live view was showing when Export was clicked. (Particle POSITIONS never
    // read the camera at all — only the visibility trace does — so this is
    // belt and braces, not the load-bearing part.)
    //
    // The live field is parked as a GPU-side buffer copy rather than re-seeded
    // afterwards: re-seeding is one line cheaper and pops the whole overlay
    // back to spawn dust in front of the viewer every time they export.
    // Returns false when there is nothing to arm (overlay off).
    beginOffline(key) {
      if (!cfg.on) return false;
      ensureBase();
      const bytes = STREAM_MAX * STREAM_PARTICLE_STRIDE;
      if (!saveBuf)
        saveBuf = device.createBuffer({
          size: bytes,
          usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(partsBuf, 0, saveBuf, 0, bytes);
      device.queue.submit([enc.finish()]);
      offSavedTime = simTime;
      offSeed = streamSeedFor(`${key ?? ""}|${cfgSig()}`);
      simTime = 0;
      offPending = true;
      return true;
    },
    // Disarm: restore the parked live field + clock. Safe to call unpaired (an
    // export that never rendered an overlay frame, or one that ran while the
    // overlay was off) — it is a no-op unless beginOffline actually armed.
    endOffline() {
      offSeed = 0;
      offPending = false;
      if (offSavedTime == null) return;
      if (saveBuf && partsBuf) {
        const enc = device.createCommandEncoder();
        enc.copyBufferToBuffer(
          saveBuf,
          0,
          partsBuf,
          0,
          STREAM_MAX * STREAM_PARTICLE_STRIDE,
        );
        device.queue.submit([enc.finish()]);
      }
      simTime = offSavedTime;
      offSavedTime = null;
      lastT = 0; // the export's wall time is not the live flow's elapsed time
    },
    // Is an export session armed? (Probe/diag readout — `info()` carries it.)
    get offline() {
      return offSavedTime != null;
    },
    // Offline export frame (renderToImage → captureFrame): advance the sim on
    // the export's VIRTUAL clock — a fixed dt per frame, deterministic
    // stepping, never wall time — and composite the points onto the offscreen
    // target exactly like encodeLive does for the live canvas.
    //
    // Under an armed session the FIRST frame additionally reseeds and pre-rolls
    // (resetAndPreroll), so the flight opens on developed flow and opens on the
    // SAME flow every run. Unarmed (a bare captureFrame — a thumbnail-grade
    // one-off), the v0 behavior stands: the export rides the live flow
    // mid-swirl, which is what WYSIWYG means for a single frame.
    encodeOffline(enc, tex, feat, dt) {
      if (!cfg.on) return;
      ensurePipelines(feat);
      if (offPending) {
        offPending = false;
        resetAndPreroll();
      }
      const d = Math.min(0.1, Math.max(0, Number(dt) || 0));
      simTime += d;
      encodeAdvect(enc, d);
      encodePoints(enc, tex.createView());
    },
    // Idle tick (no march): advect + repaint the last frame + points.
    encodeIdle(enc, tex, feat) {
      ensurePipelines(feat);
      encodeAdvect(enc, stepClock());
      const view = tex.createView();
      const pass = enc.beginRenderPass({
        colorAttachments: [
          {
            view,
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: "clear",
            storeOp: "store",
          },
        ],
      });
      pass.setPipeline(blitPl);
      pass.setBindGroup(0, blitBind);
      pass.draw(3);
      pass.end();
      encodePoints(enc, view);
    },
  };
}

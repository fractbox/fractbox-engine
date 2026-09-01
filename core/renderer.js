// WebGPU renderer. Owns the device, the pipeline (built once from the
// generated WGSL), and two GPU buffers:
//   - the uniform "Globals" (camera + control scalars; GLOBALS_WORDS × vec4 —
//     shader.js owns the layout, GLOBALS_BYTES below derives from it)
//   - a storage "ops" buffer (the op-list as packed Op structs)
// Editing param values or reordering ops only rewrites the ops buffer; the
// pipeline is never rebuilt at runtime.

import {
  buildWGSL,
  buildPostWGSL,
  buildAccumWGSL,
  buildBloomWGSL,
  POST_WORD,
  JITTER_WORD,
  DOF_WORD,
  PSTOPS_WORD,
  PCTL_WORD,
  P3CTL_WORD,
  OFFSETLO_WORD,
  TILE_WORD,
  TILEPX_WORD,
  ENVX_WORD,
  EMAP_WORD,
  AUR_WORD,
  GRADE_WORD,
  CLIP_WORD,
  GLOBALS_WORDS_ALLOC,
  packCaptureUniform,
  CAPTURE_U_WORDS,
  usesOpAux,
  usesObjAux,
  usesSeam,
} from "./shader.js";
import { createStreamlines } from "./streamlines.js";
import { bandRect, readbackBudgetMs } from "./renderpolicy.js";
import { readbackToRGBA } from "./tilegrid.js";
import { byKey } from "./operators.js";
import { viewBasis, captureEps } from "./splatcapture.js";
import { volExt, volKind, volBasis } from "./capturevolume.js";

// IEEE-754 half-float (rgba16float) → f32, for reading the capture G-buffer's
// normal/albedo targets back (§S2). posT is rgba32float (read as Float32Array).
function f16ToF32(h) {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  const sign = s ? -1 : 1;
  if (e === 0) return sign * Math.pow(2, -14) * (f / 1024);
  if (e === 0x1f) return f ? NaN : sign * Infinity;
  return sign * Math.pow(2, e - 15) * (1 + f / 1024);
}
import { computeRecenter, splitHiLo } from "./recenter.js";
import { MAX_OPS_WEBGPU, MAX_OBJECTS } from "./limits.js";
import { packOpAuxLanes } from "./uniformPack.js";
import { normalizeSceneObject } from "./sceneobj.js";
import { packHyb } from "./hybridmodel.js";
import { deriveFrameParams } from "./frameparams.js";

// PoC (CSG Phase 1a): the ops buffer is shared across all objects in a scene, so
// it's sized for MAX_OBJECTS * MAX_OPS_PER_OBJECT (8 * 24 = 192) and the scene
// writer bounds-checks the concatenated total (throws, never silently truncates).
const MAX_OPS = MAX_OPS_WEBGPU; // op-buffer capacity (192 * 16 = 3 KiB)
const OP_STRIDE = 16; // bytes per Op (u32 + 3*f32)
// Op-param overflow lane (OP_PARAM_ENCODING.md §5.5) — one vec4f per op SLOT,
// parallel to the ops buffer and addressed by the same index. Deliberately a
// SEPARATE buffer rather than a wider Op: the struct, its stride and both
// packers' existing index math stay byte-identical, so the corruption hazard of
// bumping OP_STRIDE (the `i * 4` writes are hand-unrolled and would silently
// keep the old layout) never arises.
const AUX_STRIDE = 16; // bytes per aux lane (4*f32: p3, p4, p5, reserved)
const OBJ_STRIDE = 96; // bytes per Obj (24 words: 4 u32 + 20 f32 — see shader.js Obj)

// Derived from shader.js's layout constants (was a hand-typed 560 — #239): the
// WGSL struct is the single source of truth for the layout, so the byte size
// follows it rather than drifting behind a second constant. Sized at the ALLOC
// ceiling (base struct + the ENVX tail rows): non-envx variants declare the
// smaller base struct against the same buffer, which WebGPU permits (binding
// size ≥ struct size), and the post shader's PG stays at the base size.
const GLOBALS_BYTES = GLOBALS_WORDS_ALLOC * 16; // vec4 = 16 B each

// P4 lens samples — a SECOND low-discrepancy stream (golden-ratio pair,
// decorrelated from the subpixel R2) mapped to the unit disk.
export function lensSample(i) {
  const f = (x) => x - Math.floor(x);
  const r = Math.sqrt(f(0.5 + i * 0.6180339887498949));
  const th = 2 * Math.PI * f(0.5 + i * 0.38196601125010515);
  return [r * Math.cos(th), r * Math.sin(th)];
}
const ACCUM_FORMAT = "rgba32float"; // P2 running average — renderable (unblended) + textureLoad, core WebGPU

// R2 low-discrepancy sequence (Roberts) → subpixel jitter in [-0.5, 0.5)².
// Deterministic in the sample index — offline export replay stays bit-stable.
export function r2jitter(i) {
  const f = (x) => x - Math.floor(x);
  return [
    f(0.5 + i * 0.7548776662466927) - 0.5,
    f(0.5 + i * 0.5698402909980532) - 0.5,
  ];
}
const HDR_FORMAT = "rgba16float"; // P0 march target (core-renderable, no optional features)

// ── Feature- + op-specialized march-variant descriptors ────────────────────
// PURE helpers (no device, no closure state) hoisted to module scope so the
// static gate can unit-test them without a GPU — the #265 bug lived here and
// was invisible to every existing test because these were private closures.
// `keyFor`/`wgslOf` are exported for core/marchvariant.test.mjs only.
//
// buildWGSL stubs the op-switch-heavy blocks a frame doesn't use (lever #1:
// coloring/scene/hybrid/morph/leaves/numeric) AND emits only the ops a formula
// actually contains (lever #3). The op switch is the bulk of the pipeline
// COMPILE cost and super-linear in case count (measured on a Brave/Windows box:
// 64 KB boot shader = 6.9 s; adding one 58-op switch copy → 88 KB hybrid =
// 15.4 s), so specializing to the ~handful of ops a formula uses is the biggest
// lever on slow shader compilers. Nothing compiles at boot; the first frame
// compiles the hero's own variant and the splash waits only on that.
//
// KEY SAFETY: activeMarch() always returns the variant matching the frame's
// CURRENT (flags, op-set) — sync-building it if the async warm hasn't landed —
// so correctness never depends on warm timing. The op-set is EXACT (frameOps,
// collected from what writeOps/writeScene actually put in the buffer), so a
// dropped case can never be dispatched — `default: {}` is dead for real ops.
const F_NUMERIC = 1,
  F_LEAVES = 2,
  F_COLORING = 4,
  F_SCENE = 8,
  F_HYBRID = 16,
  F_MORPH = 32,
  F_DF64 = 64, // deep zoom P4 — the df64 precision variant
  F_PERTURB = 128, // perturbation tier (PERTURBATION_ZOOM_IMPL.md PR-2)
  F_ENVX = 256, // ENVX backgrounds P5 — stars/band/zenith environment codegen
  F_ENVMAP = 512, // IMGTEX #631 — user-image environment map codegen
  F_SURFTEX = 1024, // IMGTEX #631 — triplanar surface-image codegen
  // 512 and 1024 are RESERVED for the #631 env-map arc (envMap / surfTex,
  // PR #645) — the catoptron-sweep bit partition, agreed so merge order
  // between the sibling sessions can't collide two features on one bit.
  F_SREFL = 2048, // #630 self-reflection — marched mirror-bounce codegen. ONE
  // boolean bit (the bounce COUNT is a runtime word inside the on-variant), so
  // the key space doubles instead of multiplying by 7 — the spec review's
  // bitsFor-is-boolean constraint.
  F_NEON = 4096, // NEON emissive glow — one boolean bit (the GAIN is a runtime
  // word, p3ctl.w, inside the on-variant — the #630 pattern).
  F_AURORA = 8192, // AURORA (ENVX P6) — fbm sky layer codegen. One boolean bit;
  // the amounts/colors are runtime words (the aurU/aurA/aurB tail) inside the
  // on-variant, exactly the envx shape. 8192 was double-claimed in flight
  // (TINY PLANET, PR #670, F_PLANET); aurora landed first, so #670 renumbers
  // to 32768+ (the #631 partition rule: merge order must not collide two
  // features on one bit).
  F_FILM = 16384, // THIN FILM interference material — one boolean bit (the
  // AMOUNT is a runtime word, morphX.w, inside the on-variant — the neon
  // pattern). Deliberately skipped past the once-contested 8192.
  F_PLANET = 32768, // TINY PLANET (PR #670) — stereographic ray generation.
  // One boolean bit on the same contract: the planet FOV is a runtime word
  // (camRight.w) INSIDE the on-variant, so the slider never multiplies the key
  // space. THE RENUMBER the aurora comment above calls for: this branch was
  // written against 8192, aurora landed there first, and thin film then took
  // 16384 — so the first genuinely free bit is 32768. Verified against dev at
  // merge time, not assumed: F_FILM (16384) is the highest bit in this list, and
  // the two allocations tiny planet holds outside it — camRight.w (gF[15], still
  // `= 0` on dev) and the CODEGEN gate itself — are untouched by aurora's tail
  // rows (53-55), CINE grade's (56-59) or thin film's morphX.w lane.
  //
  // CINE grade (#671) needs no bit here at all: it gates the POST pipeline
  // (buildPostWGSL({grade:true}), its own lazily-compiled pipeline), not a march
  // variant — so it never enters this key space.
  F_CLIP = 65536, // CLIP (cross-section + MRI sweep) — plane-clipped march with
  // a flat-shaded cut face. One boolean bit on the same contract: the plane
  // itself is runtime words (clipU/clipS, tail rows 60-61) INSIDE the
  // on-variant, so the offset slider — and the Loop Lab sweep driving it every
  // frame — never multiplies the key space. PRE-ASSIGNED under the #631
  // partition rule (the 2026-08-26 allocation ledger): clip takes 65536 and
  // tail rows 60-62 (61 used, 62 spare); the 360°/equirect arm took 131072
  // and camUp.w (below) — the ledger held, merge order did not collide them.
  F_CLIPJAG = 262144, // CLIP JAGGED — the noised/eroded cut (a SUB-VARIANT of
  // clip: only ever set alongside F_CLIP). Its own bit on the #125 rationale:
  // the value noise is evaluated per march step, so a flat cut must not
  // carry it. amp/freq/invD are runtime words in clipS.yzw — lanes already
  // inside clip's row-61 allocation, no new row, no new ledger claim beyond
  // this bit (taken as the next free after equirect's 131072).
  F_EQUIRECT = 131072, // 360° EQUIRECT — lat-long ray generation, the
  // second arm on the planet's ray-gen seam. One boolean bit on the planet's
  // contract: the longitude scale is a runtime word (camUp.w) INSIDE the
  // on-variant, so a resize never multiplies the key space. 131072 by
  // PRE-ASSIGNMENT, not discovery: 65536 is CLAIMED IN FLIGHT by the concurrent
  // clipping-plane branch (the 2026-08-26 allocation-ledger discipline — the
  // #631 partition rule applied BEFORE the collision this time, not renumbered
  // after it like F_PLANET above). Mutually exclusive with F_PLANET at the
  // LATCH level (the preview setters); buildWGSL additionally throws on the
  // pair so the invariant is loud for any other caller.
  F_SEAM = 524288; // #633 seam channel — the seam-clamped march + per-case
// seam reports. GENERAL-key only (see generalKey below): specialized keys
// already separate seam from seam-free formulas through their op-set
// signature, so bitsFor stays byte-stable; the bits-only general key is the
// one place two formulas differing only in a seam op would otherwise collide.
// Derived from the feature's own op-set (never a caller-supplied flag): a
// formula containing a seam op (modFold 17 / hingeFold 67) keys the
// seam-armed general; everything else keys the seam-FREE one — which is what
// lets the browse GENERAL variant drop the channel (buildWGSL's `seam:false`)
// so seam-free content stops paying the #125-class off-state tax the v1.4
// "not sharpening" field regression traced to.
const seamFor = (f) => usesSeam(f.ops ?? null); // ops:null (unknown) → armed, the safe default
const bitsFor = (f) =>
  (f.numericDE ? F_NUMERIC : 0) |
  (f.leaves ? F_LEAVES : 0) |
  (f.coloring ? F_COLORING : 0) |
  (f.scene ? F_SCENE : 0) |
  (f.hybrid ? F_HYBRID : 0) |
  (f.morph ? F_MORPH : 0) |
  (f.df64 ? F_DF64 : 0) |
  (f.perturb ? F_PERTURB : 0) |
  (f.envx ? F_ENVX : 0) |
  (f.envMap ? F_ENVMAP : 0) |
  (f.surfTex ? F_SURFTEX : 0) |
  (f.sreflect ? F_SREFL : 0) |
  (f.neon ? F_NEON : 0) |
  (f.aurora ? F_AURORA : 0) |
  (f.thinFilm ? F_FILM : 0) |
  (f.planet ? F_PLANET : 0) |
  (f.clip ? F_CLIP : 0) |
  (f.clipJag ? F_CLIPJAG : 0) |
  (f.equirect ? F_EQUIRECT : 0);
// Feature descriptor { ...flags, ops:[ids] } → cache key + buildWGSL options.
// f.leaves is either a non-empty array of leaf ids (specialize to those SDFs)
// or falsy (no leaves). It rides the key so two scenes with different leaves
// don't share a variant.
const leavesSig = (f) =>
  Array.isArray(f.leaves) && f.leaves.length ? f.leaves.join(".") : "-";
export const keyFor = (f) =>
  bitsFor(f) +
  ":" +
  // #265: "-" = the EMPTY op-set (specialized to zero cases) — unchanged from
  // before, so every existing key string (e.g. "10:-:1.2") still reads the same.
  // "*" = ops null, i.e. NOT specialized (the full switch). The two emit
  // different shaders, so they must never collide on one marchVariants entry.
  (f.ops ? (f.ops.length ? f.ops.join(".") : "-") : "*") +
  ":" +
  leavesSig(f);
export const wgslOf = (f) => ({
  numericDE: f.numericDE,
  df64: !!f.df64,
  perturb: !!f.perturb,
  envx: !!f.envx,
  envMap: !!f.envMap, // IMGTEX #631 — user-image env map (latched per frame)
  surfTex: !!f.surfTex, // IMGTEX #631 — triplanar surface image
  sreflect: !!f.sreflect, // #630 marched mirror bounces (same latch shape as envx)
  neon: !!f.neon, // NEON emissive glow (latched per frame, same shape as envx)
  aurora: !!f.aurora, // AURORA (ENVX P6) fbm sky (latched per frame, same shape as envx)
  thinFilm: !!f.thinFilm, // THIN FILM interference material (latched per frame, same shape)
  planet: !!f.planet, // TINY PLANET stereographic ray-gen (same latch shape)
  clip: !!f.clip, // CLIP cross-section plane (latched per frame, same shape)
  clipJag: !!f.clipJag, // CLIP JAGGED — noised cut sub-variant (amount-keyed)
  equirect: !!f.equirect, // 360° EQUIRECT lat-long ray-gen (same latch shape)
  leaves: Array.isArray(f.leaves) && f.leaves.length ? f.leaves : false,
  coloring: f.coloring,
  scene: f.scene,
  hybrid: f.hybrid,
  morph: f.morph,
  // #265 — an ARRAY (even an EMPTY one) is a specialization instruction:
  // `[]` means "this formula dispatches NO ops, emit zero cases". Only
  // null/undefined means "caller isn't specializing — emit the full switch".
  // The old `f.ops.length ? … : null` collapsed [] into the full-switch
  // sentinel, so a pure-leaf scene (23 of the 90 presets carry zero ops) got
  // the WORST-case 58-case shader: 116 KB / ~2.4 s to compile on an M-series
  // Mac, vs 49 KB / ~0.33 s specialized. See docs/rca/rca-issue-265-*.md.
  ops: Array.isArray(f.ops) ? f.ops : null,
});
// GENERAL variant (perf — the browse fix): the FULL op switch + ALL leaves for
// a given feature-bit set, so ONE shader renders ANY preset of that kind. On a
// slow shader compiler (Pascal/D3D12: 2-12 s flat, 20-55 s per scene) op/leaf
// SPECIALIZATION cut a single first-paint but multiplied compiles — every
// distinct preset became its own multi-second compile, so BROWSING the gallery
// was a cascade of freezes (measured ~450 s over 30 presets). The general
// variant is compiled ONCE per feature-bit set and reused, so browsing after it
// is warm triggers ZERO new compiles. Specialization is kept only for the boot
// hero's fast first paint (preview.js prewarms it before the general is armed).
// The general shader is a SUPERSET — the frame's ops/leaf ids dispatch at run
// time exactly as before — so switching hero→general is pixel-identical.
// Returns buildWGSL OPTIONS directly (NOT a keyFor/wgslOf feature object):
// `leaves: true` is buildWGSL's "emit every leaf SDF" flag, and `ops: null` is
// its "full op switch" flag. Do NOT pass this through wgslOf — wgslOf coerces a
// non-array `leaves` to `false` (it expects a specialized id list), which would
// strip every leaf and make leafDist fall back to length(p): a scene of leaves
// renders as a SPHERE. Feed it straight to buildWGSL.
const generalFeat = (f) => ({
  numericDE: f.numericDE,
  coloring: f.coloring,
  scene: f.scene,
  hybrid: f.hybrid,
  morph: f.morph,
  envx: !!f.envx, // ENVX rides the bits so a starred look browses warm too
  envMap: !!f.envMap, // IMGTEX #631 — same reason: an image look browses warm
  surfTex: !!f.surfTex,
  sreflect: !!f.sreflect, // #630 — a mirrored look browses warm too, same rule
  neon: !!f.neon, // NEON — a glowing look browses warm too, same rule
  aurora: !!f.aurora, // AURORA — an aurora look browses warm too, same rule
  thinFilm: !!f.thinFilm, // THIN FILM — a filmed look browses warm too, same rule
  planet: !!f.planet, // TINY PLANET — browse the gallery inside the projection
  clip: !!f.clip, // CLIP — browse with the cross-section held, same rule
  clipJag: !!f.clipJag, // CLIP JAGGED — a jagged look browses warm too
  equirect: !!f.equirect, // 360° EQUIRECT — same rule as planet
  leaves: f.scene ? true : false, // true = ALL leaf SDFs (scenes); none otherwise
  ops: null, // the full op switch
  // #633 — the ONE caller that overrides buildWGSL's conservative default:
  // ops:null alone would arm the seam channel for every browse shader, taxing
  // every seam-free formula with the clamped march (the v1.4 "not sharpening"
  // regression). F_SEAM rides generalKey, so seam formulas browse warm on
  // their own seam-armed general and never share this one.
  seam: seamFor(f),
});
// Keyed by feature bits alone (all same-kind presets share it). `g` prefix so
// it never collides with a specialized `bits:ops:leaves` key.
//
// ⚠ FLAT ONLY. Measured on a GTX 1080 (Pascal/D3D12): Chrome's pipeline
// disk-cache holds the small flat blob (7.3 s cold → 0.2 s cached across
// sessions) but NOT the huge ALL-leaves scene (57-108 s) or dual-op-switch
// hybrid (16-35 s) blobs — they exceed the cache entry limit and recompile
// EVERY session, so the general "superset" shape is the WORST case for them.
// Scene/hybrid/morph therefore keep per-formula op/leaf SPECIALIZATION
// (#271/#273): a far smaller blob that compiles in a fraction of the time and
// is small enough for Chrome to cache. null ⇒ "no general variant — the caller
// falls back to the specialized one".
// df64 also has NO general variant: a full-op-switch df64 shader would
// carry the double switch copy across every march fn — a huge blob for a
// variant only engaged past ×10⁵ on specific formulas. Always specialized.
// perturb likewise (same double-switch shape, same depth-gated engagement).
const generalKey = (f) =>
  f.scene || f.hybrid || f.morph || f.df64 || f.perturb
    ? null
    : // #633 — F_SEAM joins the general key HERE (not in bitsFor): the general
      // shader now comes seam-free (the common case — plain step, no channel)
      // or seam-armed (any formula carrying modFold/hingeFold), and the two
      // must never share one marchVariants entry.
      "g" + (bitsFor(f) | (seamFor(f) ? F_SEAM : 0));

export async function createRenderer(canvas, opts = {}) {
  // Headless mode (EXPORT_P1 PR-A): `canvas === null` + an injected
  // navigator.gpu-like provider (`opts.gpu` — Dawn-in-Node's `create()`) skips
  // the swapchain entirely. Only the offscreen paths (capture, buffers,
  // pipelines) are usable then — present/canvas-size paths assume a ctx and
  // are never called by the capture driver. With a canvas, behavior is
  // byte-identical to before.
  const gpuApi = opts.gpu ?? globalThis.navigator?.gpu;
  if (!gpuApi) throw new Error("WebGPU unavailable");
  const adapter = await gpuApi.requestAdapter();
  if (!adapter) throw new Error("no GPU adapter");
  const device = await adapter.requestDevice();
  const ctx = canvas ? canvas.getContext("webgpu") : null;
  const format = canvas ? gpuApi.getPreferredCanvasFormat() : "rgba8unorm"; // headless: no swapchain; format only feeds present pipelines
  if (ctx) ctx.configure({ device, format, alphaMode: "opaque" });

  // ── Diagnostics (crash triage on hard-to-reach devices, esp. Brave/Windows) ─
  // A visible-on-page report of the GPU stack + every pipeline compile + any
  // device-lost / validation error. `opts.onTrouble(kind, detail)` fires the
  // moment something goes wrong so the app can surface it BEFORE a crash takes
  // the tab (a Windows user can't open devtools in time otherwise).
  const diag = {
    backend: "webgpu",
    adapter: null,
    isFallback: !!adapter.isFallbackAdapter,
    features: [],
    compiles: [], // { name, ms, ok, error? }
    events: [], // { kind, detail, at }
  };
  const note = (kind, detail) => {
    diag.events.push({ kind, detail, at: Math.round(performance.now()) });
    try {
      opts.onTrouble?.(kind, detail);
    } catch {
      /* never let reporting break rendering */
    }
  };
  // ── Watchdog-safe readback (#460) ──────────────────────────────────────────
  // A GPU dispatch killed by the platform watchdog fires NO device.lost and NO
  // error — the pending promise simply never settles. #477 hardened every queue
  // fence in preview.js against exactly this; `buffer.mapAsync()` is the same
  // primitive with the same failure, and every readback below awaits one. Left
  // bare, a killed submit wedges the app identically: the tiled save, the alpha
  // still and the thumbnail tile all map while preview.js holds `busy` (pump
  // gated → skip:"busy", and scheduleDraw() won't even re-arm rAF), the video
  // frame and the splat capture while the exporters hold `offline` (same frozen
  // canvas via scheduleDraw's `if (!hasGPU || offline || deviceLost) return`).
  //
  // Every one of those callers ALREADY has the recovery written — a
  // try/catch/finally that releases its latch. It just never runs, because a
  // hang is not a throw. So turn the hang INTO a throw and the existing cleanup
  // does the work: `busy`/`offline` clear, the pump resumes, the export reports
  // a real error, and capturedrive.js's catch even falls back to the CPU march
  // (its "device lost / mapAsync reject ⇒ caller falls back to CPU" comment
  // describes precisely the case a watchdog kill does NOT produce).
  //
  // On a timeout the buffer is left unmapped and undestroyed. That is a
  // deliberate leak of one export's transient buffers on a once-per-session
  // catastrophic event — the same trade #477 made abandoning the fence promise
  // — and is not worth restructuring five call sites' cleanup to reclaim.
  async function mapGuarded(where, submits, maps) {
    const budgetMs = readbackBudgetMs(submits);
    let timer = 0;
    const timeout = new Promise((_res, rej) => {
      timer = setTimeout(() => {
        note("readback-timeout", { where, timeoutMs: budgetMs, submits });
        rej(
          new Error(
            `readback "${where}" timed out after ${budgetMs} ms (#460)`,
          ),
        );
      }, budgetMs);
    });
    try {
      // A REAL rejection still propagates unchanged — only the hang is new.
      await Promise.race([Promise.all(maps), timeout]);
    } finally {
      clearTimeout(timer);
    }
  }
  try {
    // adapter.info is sync on modern Dawn; requestAdapterInfo() is the older API.
    const info = adapter.info || (await adapter.requestAdapterInfo?.());
    if (info)
      diag.adapter = {
        vendor: info.vendor,
        architecture: info.architecture,
        device: info.device,
        description: info.description,
      };
  } catch {
    /* adapter info is best-effort */
  }
  try {
    diag.features = [...(device.features || [])];
  } catch {
    /* best-effort */
  }
  // A software adapter (WARP on Windows) runs the march shader catastrophically
  // slowly → a near-certain freeze/TDR; flag it loudly.
  if (
    diag.isFallback ||
    /warp|basic render|software|llvmpipe|swiftshader/i.test(
      `${diag.adapter?.description ?? ""} ${diag.adapter?.vendor ?? ""}`,
    )
  )
    note("software-adapter", diag.adapter);
  // Device loss (TDR, driver reset, GPU switch, OOM) — the signal that
  // distinguishes a real GPU-process death from a JS exception.
  device.lost?.then?.((info) =>
    note("device-lost", { reason: info?.reason, message: info?.message }),
  );

  // Wrap a pipeline compile with a validation error scope + timing, recording
  // the result into diag. Async form (preferred — off the main thread).
  async function timedPipelineAsync(name, desc) {
    const t0 = performance.now();
    device.pushErrorScope("validation");
    let pl = null,
      thrown = null;
    try {
      pl = await device.createRenderPipelineAsync(desc);
    } catch (e) {
      thrown = String(e?.message || e);
    }
    const err = await device.popErrorScope().catch(() => null);
    const rec = {
      name,
      ms: Math.round(performance.now() - t0),
      ok: !!pl && !err,
    };
    if (thrown) rec.error = thrown;
    else if (err) rec.error = err.message;
    diag.compiles.push(rec);
    if (rec.error)
      note("compile-error", { name, error: rec.error, async: true });
    if (rec.ms > 1500) note("slow-compile", { name, ms: rec.ms });
    if (thrown) throw new Error(thrown);
    return pl;
  }
  // Sync form (the fallback path activeMarch takes when a variant is needed
  // before its async warm lands — the compile that can freeze/TDR on Windows).
  function timedPipelineSync(name, module) {
    const t0 = performance.now();
    device.pushErrorScope("validation");
    let pl = null;
    try {
      pl = device.createRenderPipeline(marchPipeDesc(module));
    } finally {
      device.popErrorScope().then((err) => {
        const rec = {
          name,
          ms: Math.round(performance.now() - t0),
          ok: !!pl && !err,
          sync: true,
        };
        if (err) rec.error = err.message;
        diag.compiles.push(rec);
        if (err)
          note("compile-error", { name, error: err.message, sync: true });
        if (rec.ms > 1500)
          note("slow-compile", { name, ms: rec.ms, sync: true });
      });
    }
    return pl;
  }

  // March pipeline VARIANTS (perf, lever #1 + #3): NO march pipeline is compiled
  // at boot. Each variant is keyed by its feature flags AND the op-set of the
  // formula it renders (the op switch dominates compile time — ~7 s for the boot
  // shader on a Brave/Windows box, and it's super-linear: adding one 58-op switch
  // copy took boot 6.9 s → hybrid 15.4 s there). Compiling a full-op boot shader
  // that no real formula uses would just waste that time, so the FIRST frame
  // compiles the hero's own specialized variant (few ops → far smaller → far
  // faster) and the splash waits only on that. See the variant machinery below.

  // P0 two-pass spine: the march pass renders linear HDR into an rgba16float
  // intermediate; the post pass (tone map + exact sRGB encode + dither, see
  // shader.js buildPostWGSL) resolves it to the 8-bit target. All three draw
  // entry points (draw / drawTo / renderToImage) share the sequence.
  const marchPipeDesc = (m) => ({
    layout: "auto",
    vertex: { module: m, entryPoint: "vs" },
    fragment: {
      module: m,
      entryPoint: "fs",
      targets: [{ format: HDR_FORMAT }],
    },
    primitive: { topology: "triangle-list" },
  });
  const postModule = device.createShaderModule({ code: buildPostWGSL() });
  const postPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: postModule, entryPoint: "vs" },
    fragment: { module: postModule, entryPoint: "fs", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });
  // CINE GRADE — the graded post-pass variant, compiled LAZILY on the first
  // frame whose look derives gradeOn (a session that never touches the Look
  // row builds exactly the pipelines it built before this feature). Same
  // bindings as the base post pipeline (the grade only reads more uniform
  // words), but layout:'auto' makes each pipeline's layout its own object, so
  // bind groups are keyed per-pipeline (see postBindFor below).
  let gradePostPipeline = null;
  let gradeOn = false; // POST-pipeline latch — set per frame by writeGlobals
  function activePost() {
    if (!gradeOn) return postPipeline;
    if (!gradePostPipeline) {
      const m = device.createShaderModule({
        code: buildPostWGSL({ grade: true }),
      });
      gradePostPipeline = device.createRenderPipeline({
        layout: "auto",
        vertex: { module: m, entryPoint: "vs" },
        fragment: { module: m, entryPoint: "fs", targets: [{ format }] },
        primitive: { topology: "triangle-list" },
      });
    }
    return gradePostPipeline;
  }
  // P2 accumulation pass: mix(prev, cur, weight) into the other ping-pong half.
  const accumModule = device.createShaderModule({ code: buildAccumWGSL() });
  const accumPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: accumModule, entryPoint: "vs" },
    fragment: {
      module: accumModule,
      entryPoint: "fs",
      targets: [{ format: ACCUM_FORMAT }],
    },
    primitive: { topology: "triangle-list" },
  });
  // P3 bloom: bright(threshold+downsample) → blurH → blurV at half res.
  const bloomModule = device.createShaderModule({ code: buildBloomWGSL() });
  const bloomPipe = (entry) =>
    device.createRenderPipeline({
      layout: "auto",
      vertex: { module: bloomModule, entryPoint: "vs" },
      fragment: {
        module: bloomModule,
        entryPoint: entry,
        targets: [{ format: HDR_FORMAT }],
      },
      primitive: { topology: "triangle-list" },
    });
  const brightPipeline = bloomPipe("fs_bright");
  const blurHPipeline = bloomPipe("fs_blurH");
  const blurVPipeline = bloomPipe("fs_blurV");
  const bloomSampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
  });
  let bloomOn = false; // set by writeGlobals from the fog word — gates the passes

  const globalsBuf = device.createBuffer({
    size: GLOBALS_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const opsBuf = device.createBuffer({
    size: MAX_OPS * OP_STRIDE,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  // CSG Phase 1a: per-object descriptors. Bound at @binding(2); unused (but still
  // bound) on the single-object path so the additive invariant holds.
  const objectsBuf = device.createBuffer({
    size: MAX_OBJECTS * OBJ_STRIDE,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  // Overflow-lane buffer (OP_PARAM_ENCODING.md §5.5): allocated once alongside
  // opsBuf (3 KiB — 0.005% of the 128 MiB maxStorageBufferBindingSize floor) but
  // bound at @binding(7) ONLY on variants whose shader declares it. An `auto`
  // pipeline layout prunes a declared-but-unread storage binding, so binding it
  // unconditionally is a validation error — same conditional shape as ptBuf.
  const opAuxBuf = device.createBuffer({
    size: MAX_OPS * AUX_STRIDE,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  // Leaf-param overflow lane (#627 — the objAux mirror of opAuxBuf above):
  // sp4..sp7 per OBJECT slot (128 B total), bound at @binding(8) ONLY on
  // variants whose leaf-id set declares a fat leaf (usesObjAux — the same
  // auto-layout-prunes-unread-bindings rule as opAux).
  const objAuxBuf = device.createBuffer({
    size: MAX_OBJECTS * AUX_STRIDE,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  // Perturbation reference-orbit records (PERTURBATION_ZOOM_IMPL.md PR-2,
  // core/perturb.js buildOrbit layout): 64-iteration × 64-op worst case + the
  // trailer slot, 64 B each — allocated once (~262 KB), bound at @binding(3)
  // ONLY on perturb variants (their auto layout is the only one that declares
  // it; adding the entry to a non-perturb bind group would be a validation
  // error). Uploaded by writePerturbOrbit on re-pin, not per frame.
  const PT_RECS_BYTES = (64 * 64 + 1) * 64;
  const ptBuf = device.createBuffer({
    size: PT_RECS_BYTES,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  // ── Field streamlines (lab — core/streamlines.js) ──────────────────────────
  // Lazy: `stream` stays null until the first enable, so a session that never
  // touches the lab toggle allocates NOTHING and every draw path below costs
  // exactly one `stream?.on` check. The march/post pipelines are untouched
  // either way — the overlay is its own compute + render passes appended after
  // the post resolve on the LIVE canvas only (thumbnails / tile targets /
  // renderToImage / capture never see it).
  let stream = null;
  // The swap chain needs COPY_SRC while the overlay is on (each presented
  // frame is snapshotted into the overlay's composed texture so idle ticks can
  // repaint it without re-marching). Reconfigure flips it on/off; while off
  // the configuration is byte-identical to the boot call above.
  const configureLive = (copySrc) =>
    ctx?.configure({
      device,
      format,
      alphaMode: "opaque",
      ...(copySrc
        ? {
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
          }
        : {}),
    });
  // App-facing (via preview.setStreamlines). Returns false when the feature
  // can't run here (headless — no canvas to composite onto).
  function setStreamlines(o = {}) {
    if (!ctx) return false;
    if (!stream) {
      if (!o.on) return true; // off and never created — stay zero-cost
      stream = createStreamlines(device, {
        format,
        globalsBuf,
        opsBuf,
        objectsBuf,
        opAuxBuf,
        objAuxBuf,
      });
      configureLive(true);
    } else if (o.on === true && !stream.on) {
      configureLive(true);
    } else if (o.on === false && stream.on) {
      configureLive(false); // restore the exact boot configuration
    }
    stream.set(o);
    return true;
  }
  // Idle overlay frame (preview.js ticks this while the pump is settled):
  // advect + repaint the composed snapshot + points. No march, no accum — the
  // march pipeline is never stalled by the overlay. False = nothing drawn
  // (caller keeps the previous presented frame; a real draw will re-arm it).
  function tickStreamlines() {
    if (!stream?.on || !ctx) return false;
    if (!stream.canIdle(ctx.canvas.width, ctx.canvas.height)) return false;
    const tex = ctx.getCurrentTexture();
    const enc = device.createCommandEncoder();
    stream.encodeIdle(enc, tex, activeFeat);
    device.queue.submit([enc.finish()]);
    return true;
  }

  // ── IMGTEX (#631) — user-image textures (env map 9 / sampler 10 / surface
  // 11). ALL state is created on demand by setEnvImage — a session that never
  // loads an image allocates nothing here. `imgTexGen` bumps on every swap so
  // cached texture-carrying bind groups know they're stale (see freshBind).
  let envTexObj = null,
    envTexView = null,
    triTexObj = null,
    triTexView = null,
    imgSampler = null,
    imgFallbackTex = null,
    imgFallbackView = null,
    imgTexGen = 0;
  const ensureImgSampler = () => {
    // Repeat both axes: the equirect u wraps a full turn and the triplanar
    // taps tile. Linear filtering, no mips (single-level upload).
    if (!imgSampler)
      imgSampler = device.createSampler({
        magFilter: "linear",
        minFilter: "linear",
        addressModeU: "repeat",
        addressModeV: "repeat",
      });
    return imgSampler;
  };
  const ensureImgFallback = () => {
    // 1×1 mid-gray stand-in so an in-flight prewarm whose feat latched a
    // texture flag can still bind if the image was cleared mid-build. Never
    // shown: the latch drops the flag on the next writeGlobals, so a frame
    // can only sample this during that same-frame race window.
    if (!imgFallbackTex) {
      imgFallbackTex = device.createTexture({
        size: [1, 1],
        format: "rgba8unorm-srgb",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      device.queue.writeTexture(
        { texture: imgFallbackTex },
        new Uint8Array([128, 128, 128, 255]),
        { bytesPerRow: 4 },
        [1, 1],
      );
      imgFallbackView = imgFallbackTex.createView();
    }
    return imgFallbackView;
  };
  // Upload (or clear, with source = null) one image slot. `slot` is "env"
  // (the equirect environment) or "surface" (the triplanar tile); `source` is
  // anything copyExternalImageToTexture accepts (ImageBitmap, canvas) —
  // ALREADY decode-capped by the caller (envmap.js fitImageDims: ≤2048px,
  // ≤1024 on mobile-class — the #476 governor's memory lesson). Stored as
  // rgba8unorm-srgb so samples arrive linear in the shader.
  function setEnvImage(slot, source) {
    const isEnv = slot !== "surface";
    const old = isEnv ? envTexObj : triTexObj;
    let tex = null,
      view = null;
    if (source) {
      ensureImgSampler();
      const w = source.width,
        h = source.height;
      tex = device.createTexture({
        size: [w, h],
        format: "rgba8unorm-srgb",
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.RENDER_ATTACHMENT, // copyExternalImageToTexture requires it
      });
      device.queue.copyExternalImageToTexture({ source }, { texture: tex }, [
        w,
        h,
      ]);
      view = tex.createView();
    }
    if (isEnv) {
      envTexObj = tex;
      envTexView = view;
    } else {
      triTexObj = tex;
      triTexView = view;
    }
    old?.destroy();
    imgTexGen++;
  }
  // The latch's texture-presence half (preview.js mirrors the frame latch for
  // its prewarm prediction — a mismatch only costs an extra compile, but this
  // keeps the prediction exact).
  const hasImgTex = (slot) =>
    slot === "surface" ? !!triTexView : !!envTexView;

  const marchBind = (
    pl,
    perturb = false,
    aux = false,
    objAux = false,
    envMap = false,
    surfTex = false,
  ) =>
    device.createBindGroup({
      layout: pl.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: globalsBuf } },
        { binding: 1, resource: { buffer: opsBuf } },
        { binding: 2, resource: { buffer: objectsBuf } },
        ...(perturb ? [{ binding: 6, resource: { buffer: ptBuf } }] : []),
        // Only when the shader DECLARED the lane — see opAuxBuf above. The flag
        // must come from usesOpAux() over the SAME ops list handed to buildWGSL.
        ...(aux ? [{ binding: 7, resource: { buffer: opAuxBuf } }] : []),
        // #627: same rule for the leaf lane — usesObjAux() over the SAME
        // leaves list handed to buildWGSL.
        ...(objAux ? [{ binding: 8, resource: { buffer: objAuxBuf } }] : []),
        // IMGTEX (#631): only on variants whose shader declared the bindings
        // (same auto-layout-prunes-unread rule as the lanes above). The
        // fallback view covers the clear-mid-prewarm race — see above.
        ...(envMap
          ? [{ binding: 9, resource: envTexView || ensureImgFallback() }]
          : []),
        ...(envMap || surfTex
          ? [{ binding: 10, resource: ensureImgSampler() }]
          : []),
        ...(surfTex
          ? [{ binding: 11, resource: triTexView || ensureImgFallback() }]
          : []),
      ],
    });

  const marchVariants = new Map(); // key -> {pl,bg}, MRU-ordered for LRU eviction
  const marchWarms = new Map(); // key -> in-flight async build promise
  const VARIANT_CAP = 24; // bound GPU memory on weak devices; specialized shaders recompile cheaply
  let activeFeat = { ops: null }; // latched each frame by writeGlobals
  let frameLeafIds = null; // EXACT leaf-id set written this frame (writeScene), or null
  let frameOps = null; // EXACT op-id set written this frame (writeOps / writeScene)

  const lruTouch = (key) => {
    const v = marchVariants.get(key);
    if (v) {
      marchVariants.delete(key);
      marchVariants.set(key, v);
    } // re-insert → MRU
  };
  // Deep zoom P4: while df64 is engaged, the current formula's df64 variant
  // AND its f32 twin (the drag/smooth-tier fallback) are pinned against
  // eviction — an LRU evict mid-interaction would force a recompile freeze at
  // exactly the wrong moment (plan PR-3 item 3). Latched by writeGlobals.
  let pinnedKeys = null; // Set<key> | null
  const lruEvict = (keep) => {
    if (marchVariants.size <= VARIANT_CAP) return;
    for (const k of [...marchVariants.keys()]) {
      // oldest-first
      if (marchVariants.size <= VARIANT_CAP) break;
      if (k === keep || pinnedKeys?.has(k)) continue;
      marchVariants.delete(k);
    }
  };

  // One variant record for every build path below: pipeline + bind group +
  // the bind-group RECIPE (`mk`) and the imgTexGen it bound (`gen`). The
  // recipe exists for IMGTEX (#631): a texture-carrying bind group references
  // the texture VIEW, so setEnvImage swaps strand it — freshBind() rebinds
  // (cheap — createBindGroup, never a compile) when the generation moved.
  const variantRec = (pl, perturb, o) => {
    const mk = {
      perturb: !!perturb,
      aux: usesOpAux(o.ops),
      objAux: usesObjAux(o.leaves),
      envMap: !!o.envMap,
      surfTex: !!o.surfTex,
    };
    return {
      pl,
      bg: marchBind(pl, mk.perturb, mk.aux, mk.objAux, mk.envMap, mk.surfTex),
      mk,
      gen: imgTexGen,
    };
  };
  const freshBind = (v) => {
    if ((v.mk?.envMap || v.mk?.surfTex) && v.gen !== imgTexGen) {
      v.bg = marchBind(
        v.pl,
        v.mk.perturb,
        v.mk.aux,
        v.mk.objAux,
        v.mk.envMap,
        v.mk.surfTex,
      );
      v.gen = imgTexGen;
    }
    return v;
  };

  function buildMarchVariant(feat) {
    const key = keyFor(feat);
    const cached = marchVariants.get(key);
    if (cached) {
      lruTouch(key);
      return cached;
    }
    const o = wgslOf(feat);
    const m = device.createShaderModule({ code: buildWGSL(o) });
    const pl = timedPipelineSync("variant:" + key, m);
    const v = variantRec(pl, feat.perturb, o);
    marchVariants.set(key, v);
    lruEvict(key);
    return v;
  }
  function prewarmMarch(feat) {
    const key = keyFor(feat);
    if (marchVariants.has(key)) {
      lruTouch(key);
      return Promise.resolve();
    }
    if (marchWarms.has(key)) return marchWarms.get(key);
    if (!device.createRenderPipelineAsync) return null; // sync fallback only
    const o = wgslOf(feat);
    const m = device.createShaderModule({ code: buildWGSL(o) });
    const p = timedPipelineAsync("prewarm:" + key, marchPipeDesc(m))
      .then(
        (pl) => {
          if (pl && !marchVariants.has(key)) {
            marchVariants.set(key, variantRec(pl, feat.perturb, o));
            lruEvict(key);
          }
        },
        () => {}, // timedPipelineAsync already recorded the error into diag
      )
      .finally(() => marchWarms.delete(key));
    marchWarms.set(key, p);
    return p;
  }
  // General (browse) variant — same structure as buildMarchVariant/prewarmMarch
  // but the FULL-op / all-leaf shader keyed by feature bits (reused by every
  // same-kind preset).
  function buildGeneral(feat) {
    const gkey = generalKey(feat);
    const cached = marchVariants.get(gkey);
    if (cached) {
      lruTouch(gkey);
      return cached;
    }
    const o = generalFeat(feat);
    const m = device.createShaderModule({ code: buildWGSL(o) });
    const pl = timedPipelineSync("general:" + gkey, m);
    const v = variantRec(pl, false, o);
    marchVariants.set(gkey, v);
    lruEvict(gkey);
    return v;
  }
  function prewarmGeneral(feat) {
    const gkey = generalKey(feat);
    if (!gkey) return null; // scene/hybrid/morph → no general; caller uses specialized
    if (marchVariants.has(gkey)) {
      lruTouch(gkey);
      return Promise.resolve();
    }
    if (marchWarms.has(gkey)) return marchWarms.get(gkey);
    if (!device.createRenderPipelineAsync) return null; // sync fallback only
    const o = generalFeat(feat);
    const m = device.createShaderModule({ code: buildWGSL(o) });
    const p = timedPipelineAsync("general:" + gkey, marchPipeDesc(m))
      .then(
        (pl) => {
          if (pl && !marchVariants.has(gkey)) {
            marchVariants.set(gkey, variantRec(pl, false, o));
            lruEvict(gkey);
          }
        },
        () => {},
      )
      .finally(() => marchWarms.delete(gkey));
    marchWarms.set(gkey, p);
    return p;
  }
  const activeMarch = () => {
    // Prefer the GENERAL (browse) variant if warm — one shader for all same-kind
    // presets, so browsing never compiles per preset. Fall back to the boot
    // hero's specialized variant, then to a sync build. The sync fallback builds
    // the GENERAL (reusable) shader, not a throwaway specialized one, so even a
    // forced hitch happens at most once per feature-bit set.
    const gkey = generalKey(activeFeat);
    const g = gkey && marchVariants.get(gkey);
    if (g) {
      lruTouch(gkey);
      return freshBind(g);
    }
    const key = keyFor(activeFeat);
    const v = marchVariants.get(key);
    if (v) {
      lruTouch(key);
      return freshBind(v);
    }
    // Sync fallback: the reusable GENERAL shader for flat, the SPECIALIZED shader
    // for scene/hybrid/morph (no general variant — smaller, cacheable).
    return freshBind(
      gkey ? buildGeneral(activeFeat) : buildMarchVariant(activeFeat),
    );
  };
  // Public, feature-object API (preview.js predicts the frame's (flags, op-set)
  // and holds the frame until a matching variant is warm — general OR the boot
  // hero's specialized one).
  const marchReadyFor = (feat) => {
    const gk = generalKey(feat);
    return (!!gk && marchVariants.has(gk)) || marchVariants.has(keyFor(feat));
  };
  const prewarmMarchFor = (feat) => prewarmMarch(feat);
  const prewarmGeneralFor = (feat) => prewarmGeneral(feat);

  const gBuf = new ArrayBuffer(GLOBALS_BYTES);
  const gF = new Float32Array(gBuf);
  const gU = new Uint32Array(gBuf);

  // `payload` is destructured in the body rather than in the signature so the
  // WHOLE bag can reach deriveFrameParams below. It used to be handed a
  // retyped 14-field subset, and every field missing from that list silently
  // read back as its identity default — which is how palettePhase and
  // iridescence (both added to the uniform words by the S6 commit, 7494398,
  // without joining the argument list) rendered as no-ops on this tier from
  // the day they shipped, and auto-levels' sigLo/sigSpan with them. The GL
  // tier passes its whole payload (`deriveFrameParams(G)`) and was correct all
  // along; "ONE source with the WebGPU tier … the two tiers can't drift on
  // defaults" only holds if both hand it the same thing.
  function writeGlobals(payload) {
    const {
      res,
      cam,
      iters,
      kStar = 0, // deep zoom P4 — df64 switchover count; 0 = disengaged
      perturb = false, // perturbation tier (PR-3): select the delta variant
      // ORTHOGRAPHIC_VIEWS (#441) — the ortho half-height, riding camFwd.w.
      // 0 = perspective, and the shader's perspective path is then byte-identical
      // to before the field meant anything. This slot had no readers but WAS
      // zeroed here every frame, so it needed a parameter, not just a shader edit.
      orthoH = 0,
      // TINY PLANET — the stereographic plane radius at the frame's vertical
      // half-extent, tan(planetFov/4), riding camRight.w. 0 = off. Same trap
      // as orthoH above: this slot had no readers but WAS zeroed here every
      // frame, so it needs a parameter, not just a shader edit. It also DRIVES
      // the codegen latch below (planetK > 0 selects the planet variant) —
      // one number, one source of truth, exactly like kStar for df64.
      planetK = 0,
      // 360° EQUIRECT — the longitude scale of the FULL image, π·H/W, riding
      // camUp.w. 0 = off. The same trap as planetK/orthoH above (the slot had
      // no readers but WAS zeroed here every frame, so it needs a parameter),
      // and the same double duty: it DRIVES the codegen latch below
      // (equirectS > 0 selects the equirect variant) — one number, one source
      // of truth. It is π·H/W of the FULL image, not this render target's,
      // which is what makes a TILE render exact (core/preview.js derives it
      // once, from the export's dims).
      equirectS = 0,

      opCount,
      addC,
      maxSteps,
      bailout,
      eps,
      deScale,
      colA,
      colB,
      bg,
      colorMode,
      stripeFreq,
      deOption,
      julia,
      juliaC,
      palette,
      light,
      objectCount,
      tNear,
      tFar,
      hybrid, // Phase 1a spike — {aOpCount,bOpCount,scheduleA,scheduleB,addCA,addCB}
      morph, // formula-morph spike — {bOpCount,bIters,bAddC,bJulia,bJuliaC,bDeOption,t}
      colorBlend, // coloring-mode crossfade — {t, modeB, palOnB}
      post, // P0 post pass — {tone: 0 classic|1 filmic, exposure EV, dither LSBs, vignette}
      debugView = 0, // #370 debug overlay — 0 off | 1 step heat | 2 overshoot | 3 ∇DE (rides p3ctl.z)
    } = payload;
    const b = cam.basis();
    // Shared pure derivations (color/palette defaults, light rig, sky/fog
    // expansion) — ONE source with the WebGL2 tier (frameparams.js), packed
    // here into the uniform words. The WHOLE payload goes in (see the note on
    // the signature): deriveFrameParams reads only the keys it knows, so a
    // field added to the uniform words below can never again be silently
    // dropped on the way in.
    // Auto-levels (COLORING P2) rides this too, as of the revival PR: sigLo/
    // sigSpan were dropped by the same bug and had been dark here since P2
    // shipped, so the flagship tier was the ONLY one rendering the raw,
    // un-normalized signal — the WebGL2 tier has applied the identical
    // signalRange() all along. capturesettle computes that range every frame
    // regardless (the result was simply thrown away here), so this costs
    // nothing new; it only stops discarding it.
    const d = deriveFrameParams(payload);
    const A = d.colA,
      B = d.colB,
      BG = d.bg;
    const JC = d.juliaC;
    // Deep zoom §3.1 recenter. O = the pan target, kept in JS f64 (cam.target
    // already is — it's a plain number array). ro_rel = eye−O is computed HERE,
    // in f64, before it ever touches a Float32Array — only the small residual
    // and O separately get truncated to f32, not their (huge) sum. CSG scenes
    // are explicitly out of scope for v1 (§14, objDist's per-object placement
    // isn't recentered) — offset=(0,0,0) there makes the shader's reconstruction
    // an exact no-op, so scenes render byte-identically to before this change.
    const isScene = (objectCount || 0) > 0;
    const { O, roRel } = computeRecenter(b.eye, cam.target, isScene);
    // D11 (PERTURBATION_ZOOM_IMPL.md, review blocker B1): eye − O re-derives
    // dir·dist through an absorb-then-cancel f64 round trip whose error is
    // ~eps·max(|O|,1) ABSOLUTE — the residual is garbage past ~×10¹⁵⁻¹⁶ no
    // matter how precise any downstream tier is. cam.basis() exposes the
    // pre-add dir·dist exactly (b.roRel); prefer it whenever the offset
    // split is live. Shallow frames are fround-identical either way (the
    // two paths agree to ~1e-17 relative); capture paths using viewBasis
    // have no roRel and keep the classic subtraction.
    const roR = !isScene && b.roRel ? b.roRel : roRel;
    gF[0] = res[0];
    gF[1] = res[1];
    gF[2] = cam.fov;
    gF[3] = d.tNear; // res.w — deep zoom §5 (was unused padding)
    gF[4] = roR[0]; // camPos.xyz — deep zoom §3.1: the RESIDUAL, not b.eye
    gF[5] = roR[1];
    gF[6] = roR[2];
    gF[7] = d.tFar; // camPos.w — deep zoom §5 (was unused padding)
    gF[8] = b.fwd[0];
    gF[9] = b.fwd[1];
    gF[10] = b.fwd[2];
    gF[11] = orthoH; // camFwd.w — #441 ortho half-height (0 = perspective)
    gF[12] = b.right[0];
    gF[13] = b.right[1];
    gF[14] = b.right[2];
    gF[15] = planetK; // camRight.w — TINY PLANET tan(planetFov/4) (0 = off)
    gF[16] = b.up[0];
    gF[17] = b.up[1];
    gF[18] = b.up[2];
    gF[19] = equirectS; // camUp.w — 360° EQUIRECT π·H/W (full image; 0 = off)
    gU[20] = iters;
    gU[21] = opCount;
    gU[22] = d.addGate;
    gU[23] = maxSteps; // ctrl
    gF[24] = bailout;
    gF[25] = eps;
    gF[26] = d.deScale; // ?? 0.85 default now matches the GL tier (was raw)
    gF[27] = d.colorMode; // prm
    gF[28] = A[0];
    gF[29] = A[1];
    gF[30] = A[2];
    gF[31] = d.deOption; // colA.rgb + .w=deOption
    // Latch the march-variant feature bits for THIS frame (see activeMarch).
    // Flat coloring fns (orbitTrap/escape/silk/pin/address/painter/irid) are
    // reached only for a FLAT formula in a coloring mode — scenes discard mixT
    // and colour via sceneOrbit (scene flag); Surface (0) and Curvature (5) use
    // no op-switch coloring fn. Leaf set rides writeScene's frameLeafIds.
    const coloringMode = (m) => m > 0.5 && Math.round(m) !== 5;
    const wantColoring =
      !isScene &&
      (coloringMode(d.colorMode) ||
        (!!colorBlend && coloringMode(colorBlend.modeB)));
    // Latch the full variant descriptor (feature flags + the EXACT op-set the
    // write path just uploaded). Every write path (flat/hybrid/morph/scene) runs
    // its write* BEFORE writeGlobals in the frame, so frameOps is current here.
    activeFeat = {
      numericDE: d.deOption >= 2.5,
      // df64 rides the SAME latch: kStar > 0 IS the engagement signal (plan
      // D4) — preview.js only sends kStar on the flat path when its
      // eligibility + hysteresis + tier gates all pass.
      df64: kStar > 0,
      // perturbation tier (PERTURBATION_ZOOM_IMPL.md PR-3): the caller-level
      // engagement flag — the harness/diff tools today, preview.js's
      // ptEligible+hysteresis in plan PR-4. Selects the F_PERTURB variant;
      // the caller must have writePerturbOrbit'd a matching orbit.
      perturb: !!perturb,
      // Non-empty leaf-id array (specialize to those SDFs) or null. Only the
      // scene path sets frameLeafIds; guard on isScene so a flat frame never
      // inherits a stale set.
      leaves: isScene ? frameLeafIds : null,
      coloring: wantColoring,
      scene: isScene,
      hybrid: !!hybrid,
      morph: !!morph,
      // ENVX (backgrounds P5): the codegen latch — stars/band/zenith all
      // zero/absent derives false, and the emitted shader is then byte-
      // identical to pre-ENVX text (the doctrine's "prove it's free" diff).
      envx: d.envx,
      // IMGTEX (#631): the same contract — each texture flag latches only
      // while an image is LOADED (setEnvImage) AND its amount slider is up;
      // otherwise the emitted text is byte-identical to pre-IMGTEX.
      envMap: !!envTexView && d.emapAmt > 0,
      surfTex: !!triTexView && d.triAmt > 0,
      // #630 self-reflection: same latch shape — reflBounces > 0 derives it
      // (frameparams), shadeLight zeroes reflBounces on cheap frames so a
      // drag renders the plain (off) variant, and the precision tiers are an
      // eligibility domain (spec review 2f): a reflected ray leaves the
      // reference-orbit neighbourhood df64/perturb linearise around, so the
      // mirror is forced off whenever either is engaged (buildWGSL throws on
      // the combination — this guard is what keeps it unreachable).
      sreflect: d.sreflect && !(kStar > 0) && !perturb,
      // NEON emissive glow: same latch shape — neonGain > 0 derives it
      // (frameparams). FLAT formulas only in V1 (the iridescence S6
      // precedent — a scene's arms don't record the signal), so the scene
      // guard keeps every scene shader byte-identical.
      neon: d.neon && !isScene,
      // AURORA (ENVX P6): the same latch shape as envx — either amount > 0
      // derives it (frameparams); a background layer, so scenes carry it too
      // (exactly the envx contract, unlike neon's flat-only signal).
      aurora: d.aurora,
      // THIN FILM interference material: same latch shape — the slider > 0
      // derives it (frameparams). FLAT formulas only in V1 (the neon/S6
      // precedent), so the scene guard keeps every scene shader
      // byte-identical.
      thinFilm: d.thinFilm && !isScene,
      // TINY PLANET: the codegen latch IS the uniform — planetK > 0 selects
      // the stereographic ray-gen variant, and 0 emits the pre-tiny-planet
      // text byte for byte (core/tinyplanet.test.mjs). Unlike neon/thinFilm
      // there is no scene guard: ray generation happens before anything knows
      // whether the scene is flat or CSG, so every kind gets it — the envx/
      // aurora contract rather than the flat-only one.
      planet: planetK > 0,
      // CLIP cross-section: the same latch shape as envx — clipOn derives it
      // (frameparams; an explicit boolean, since offset 0 is a meaningful
      // plane position). No scene guard: the plane is march geometry, so
      // every kind gets it (the planet/aurora contract). Off emits byte-
      // identical text (core/clipplane.test.mjs). clipJag is the noised-cut
      // sub-variant (amount-keyed in frameparams: clipOn AND Jagged > 0).
      clip: d.clip,
      clipJag: d.clipJag,
      // 360° EQUIRECT: same contract as planet in every respect (the latch IS
      // the runtime word; no scene guard). The two are mutually exclusive at
      // the source (preview.js setters), so at most one of these is true.
      equirect: equirectS > 0,
      ops: frameOps,
    };
    pinnedKeys =
      activeFeat.df64 || activeFeat.perturb
        ? new Set([
            keyFor(activeFeat),
            keyFor({ ...activeFeat, df64: false, perturb: false }),
          ])
        : activeFeat.sreflect
          ? // #630 — the df64 pinned-twin pattern: while mirrors are on, pin
            // the on-variant AND its plain twin (the cheap-tier drag fallback)
            // so an LRU evict can't force a recompile freeze mid-interaction.
            new Set([
              keyFor(activeFeat),
              keyFor({ ...activeFeat, sreflect: false }),
            ])
          : null;
    gF[32] = B[0];
    gF[33] = B[1];
    gF[34] = B[2];
    gF[35] = d.stripeFreq; // colB.rgb + .w = Silk stripe frequency (COLORING S2)
    gF[36] = BG[0];
    gF[37] = BG[1];
    gF[38] = BG[2];
    gF[39] = 0; // bgc
    gF[40] = JC[0];
    gF[41] = JC[1];
    gF[42] = JC[2];
    gF[43] = d.julia; // jc.xyz + .w=julia flag

    // Cosine palette + lighting — derived once in frameparams.js.
    gF[44] = d.palA[0];
    gF[45] = d.palA[1];
    gF[46] = d.palA[2];
    gF[47] = d.palOn; // palA.rgb + .w=paletteOn
    gF[48] = d.palB[0];
    gF[49] = d.palB[1];
    gF[50] = d.palB[2];
    gF[51] = 0; // palB
    gF[52] = d.palC[0];
    gF[53] = d.palC[1];
    gF[54] = d.palC[2];
    gF[55] = 0; // palC (freq)
    gF[56] = d.palD[0];
    gF[57] = d.palD[1];
    gF[58] = d.palD[2];
    gF[59] = 0; // palD (phase)
    // COLORING P0 — N-stop palette (words 35..43). palStops are already OKLab
    // (frameparams converts sRGB→OKLab once); pctl.x=0 → the shader ignores
    // them and takes the legacy cosine/ramp path, so an absent palette here is
    // byte-identical to before this field existed.
    {
      const S = PSTOPS_WORD * 4;
      for (let i = 0; i < 8; i++) {
        const st = d.palStops?.[i];
        gF[S + i * 4] = st ? st[0] : 0;
        gF[S + i * 4 + 1] = st ? st[1] : 0;
        gF[S + i * 4 + 2] = st ? st[2] : 0;
        gF[S + i * 4 + 3] = st ? st[3] : 0;
      }
      const C = PCTL_WORD * 4;
      gF[C] = d.palStopCount || 0;
      gF[C + 1] = d.palCyclic || 0;
      // COLORING P2 — auto-levels signal range (identity 0,1 when off/cyclic).
      gF[C + 2] = d.sigLo ?? 0;
      gF[C + 3] = d.sigSpan ?? 1;
      // COLORING P3 — iridescence (x) + palette phase (y); zw reserved.
      const C3 = P3CTL_WORD * 4;
      gF[C3] = d.iridescence ?? 0;
      gF[C3 + 1] = d.palettePhase ?? 0;
      // #370 debug surface-quality overlay — 0 (default) is byte-identical.
      gF[C3 + 2] = debugView || 0;
      // NEON — p3ctl.w carries the emissive gain (was reserved). Read only by
      // neon variants; writing it under any other shader renders byte-
      // identically (the offsetLo "unread words" contract, the #630 colorX.w
      // precedent).
      gF[C3 + 3] = d.neonGain ?? 0;
    }
    gF[60] = d.lightDir[0];
    gF[61] = d.lightDir[1];
    gF[62] = d.lightDir[2];
    gF[63] = 0; // light dir
    gF[64] = d.ambient;
    gF[65] = d.rim;
    gF[66] = d.gloss;
    gF[67] = d.intensity; // lprm (w=intensity)
    gU[68] = objectCount || 0; // scene.x — 0 = single-object (legacy) path
    gU[69] = 0;
    gU[70] = 0;
    gU[71] = 0; // scene padding
    // N-slot hybrid (HYBRID_NSLOT_SPEC.md §2.3) — the `hyb` vec4u, packed for the
    // 8-slot engineered ceiling via the ONE JS pack helper (hybridmodel.packHyb),
    // the mirror of the WGSL hybWalk decode. slotCount==0 (default) ⇒ the mapDE
    // dispatch stays on the flat/single path and the coloring walk reads the whole
    // op buffer. `hybrid` = { opCounts[], counts[], addC[] } from the accessor.
    const hpk = hybrid ? packHyb(hybrid) : { x: 0, y: 0, z: 0, w: 0 };
    gU[72] = hpk.x; // opCounts[0..3]
    gU[73] = hpk.y; // schedule counts[0..7] (nibbles)
    gU[74] = hpk.z; // slotCount (bits0-3) | addC bits (bits16-23)
    gU[75] = hpk.w; // opCounts[4..7]
    // Deep zoom §3.1 + Phase 4 — offset as an f32 hi/lo PAIR (recenter.js
    // splitHiLo). `offset` gets hi (numerically what the raw f32 store always
    // held); `offsetLo` gets the truncated-away low words the df64 marcher
    // reconstructs with (DEEP_ZOOM_DF64.md). offsetLo.w carries k* — 0 until
    // the df64 variant engages (plan PR-3), which keeps every current frame
    // byte-identical to before the field existed.
    {
      const s = splitHiLo(O);
      gF[76] = s.hi[0];
      gF[77] = s.hi[1];
      gF[78] = s.hi[2];
      gF[79] = 0; // offset.w reserved (always 0 — PR-3 arms df_lz from it)
      const lo = OFFSETLO_WORD * 4;
      gF[lo] = s.lo[0];
      gF[lo + 1] = s.lo[1];
      gF[lo + 2] = s.lo[2];
      gF[lo + 3] = kStar; // k* — 0 = df64 disengaged (offsetLo.w)
    }
    // Formula-morph spike words (VIDEO_EXPORT_DRAWER_V2 tier 2). morphB.w==0
    // (the default) ⇒ mapDE skips straight past the morph path — every legacy
    // frame is byte-identical to before this change. morphB.w carries B's OWN
    // orbit bailout as f32 bits (word 83 written via the FLOAT view — the
    // shader bitcasts it back; non-zero doubles as the on-flag). B must not
    // share A's bailout: a power-8 escape orbit run to an IFS-sized 1e6
    // overflows dr in f32 and blanks the render.
    const M = morph;
    const mjc = M?.bJuliaC || [0, 0, 0];
    gU[80] = M?.bOpCount ?? 0;
    gU[81] = M?.bIters ?? 0;
    gU[82] = M
      ? (M.bAddC ? 1 : 0) | (M.bJulia ? 2 : 0) | (((M.bDeOption ?? 2) & 3) << 2)
      : 0;
    gF[83] = M ? (M.bailB ?? 1.0e6) : 0; // morph on + B's bailout (f32 bits)
    gF[84] = M?.t ?? 0;
    gF[85] = mjc[0] ?? 0;
    gF[86] = mjc[1] ?? 0;
    gF[87] = mjc[2] ?? 0;
    gF[88] = M?.swell ?? 0; // morphX.x — mid-blend dilation (world units, pre-shaped by preview)
    gF[89] = 0;
    gF[90] = 0; // morphX.yz reserved (the morph arc's natural growth lanes;
    // .z is also the eyed slot for a thin-film thickness control)
    // THIN FILM — morphX.w carries the interference amount (the LAST reserved
    // lane, the p3ctl.w precedent — verified free on dev and every in-flight
    // branch at the #669 merge). Read only by thinFilm variants; writing it
    // under any other shader renders byte-identically (the offsetLo "unread
    // words" contract).
    gF[91] = d.filmAmt;
    // Coloring-mode crossfade word (colorX): 0 blend ⇒ the legacy shade only.
    const CB = colorBlend;
    gF[92] = CB?.t ?? 0;
    gF[93] = CB?.modeB ?? 0;
    gF[94] = CB?.palOnB ? 1 : 0;
    // #630 — colorX.w carries the self-reflection metal tint (was reserved).
    // Read only by sreflect variants; writing it under any other shader
    // renders byte-identically (the offsetLo "unread words" contract).
    gF[95] = d.reflTint;
    // P0 post word — defaults are the shipped look: filmic soft-shoulder tone
    // map on, no exposure bias, 1-LSB dither, no vignette. `tone: 0` gives the
    // pre-P0 straight encode ("Classic") for A/B comparisons and tests.
    gF[96] = post?.tone ?? 1;
    gF[97] = post?.exposure ?? d.exposure; // whole-frame EV (user "Exposure" slider)
    gF[98] = post?.dither ?? 1;
    gF[99] = post?.vignette ?? 0;
    // P1 light rig + material words (defaults + the derived penumbra k and
    // fill/back directions live in frameparams.js, shared with the GL tier).
    gF[100] = d.keyColor[0];
    gF[101] = d.keyColor[1];
    gF[102] = d.keyColor[2];
    // #630 — lightC.w carries the self-reflection bounce count 0..6 (was
    // reserved). Already 0 on cheap frames (shadeLight zeroes reflBounces —
    // the shadow/AO word-flip pattern); unread outside sreflect variants.
    gF[103] = d.reflBounces;
    gF[104] = d.metallic;
    gF[105] = d.shadowK; // penumbra k (30 hard … 4 very soft)
    gF[106] = d.shadowOn;
    gF[107] = d.ao;
    gF[108] = d.fillDir[0];
    gF[109] = d.fillDir[1];
    gF[110] = d.fillDir[2];
    gF[111] = d.fill; // light2.w = fill intensity
    gF[112] = d.fillColor[0];
    gF[113] = d.fillColor[1];
    gF[114] = d.fillColor[2];
    gF[115] = d.reflectivity; // #630 — light2c.w: self-reflection base amount (was reserved)
    gF[116] = d.backDir[0];
    gF[117] = d.backDir[1];
    gF[118] = d.backDir[2];
    gF[119] = d.back; // light3.w = back intensity
    gF[120] = d.backColor[0];
    gF[121] = d.backColor[1];
    gF[122] = d.backColor[2];
    gF[123] = d.reflFresnel; // #630 — light3c.w: self-reflection Fresnel edge (was reserved)
    // P2 jitter word — zero on every full write (the un-jittered base frame);
    // writeJitter() partial-updates it between accumulation frames.
    gF[124] = 0;
    gF[125] = 0;
    gF[126] = 0;
    gF[127] = 0;
    // P3 env/fog words — macro expansion (sky → glow/IBL, fog → in-scatter,
    // glow → bloom) derived in frameparams.js; all default 0 → byte-identical
    // to the P2 pipeline (sky/fog/bloom fully opt-in).
    gF[128] = d.sky;
    gF[129] = d.sunGlow; // sun glow rides the sky macro, unless hidden (#160)
    gF[130] = d.ground; // ground dim (fixed)
    gF[131] = d.ibl; // IBL ambient tint rides the sky macro
    gF[132] = d.fog;
    gF[133] = d.inScatter; // in-scatter rides the fog macro
    gF[134] = d.bloomStrength;
    gF[135] = d.bloomThreshold; // bloom threshold (pre-tonemap HDR)
    bloomOn = d.bloomOn;
    // P4 DOF word: slider 0..1 → lens radius scaled by the orbit distance
    // (quadratic feel — the top half of the slider does the drama); autofocus
    // = the orbit distance (zoom-to-surface glides the target onto the
    // surface, so this focuses on what you're looking at). zw = lens point,
    // zeroed here (base frame = lens center) and written per accumulation
    // sample by writeJitter. WebGPU-only (needs cam.dist) — stays here.
    const ap = (light || {}).aperture ?? 0;
    gF[136] = ap > 0 ? ap * ap * (cam.dist ?? 4) * 0.06 : 0;
    gF[137] = cam.dist ?? 4;
    gF[138] = 0;
    gF[139] = 0;
    // TILED_EXPORT §2.1.3 / §2.2.1(a) — the tile window and the absolute-screen
    // rect. Every FULL write restores the identity: (1,1,0,0) is ×1.0 + 0.0 on
    // the ray-gen expression and tilepx.w = 0 switches the three guarded blocks
    // off, so a frame written here is bit-identical to one written before these
    // words existed. writeTile() partial-updates them per tile, exactly as
    // writeJitter() does for the accumulation words.
    const T = TILE_WORD * 4;
    gF[T] = 1;
    gF[T + 1] = 1;
    gF[T + 2] = 0;
    gF[T + 3] = 0;
    const TP = TILEPX_WORD * 4;
    gF[TP] = 0;
    gF[TP + 1] = 0;
    gF[TP + 2] = 0;
    gF[TP + 3] = 0;
    // ENVX (backgrounds P5) tail rows — ALWAYS written (the buffer is sized at
    // GLOBALS_WORDS_ALLOC), read only by envx variants: writing them under a
    // non-envx shader renders byte-identically (the offsetLo contract).
    const E = ENVX_WORD * 4;
    gF[E] = d.stars; // starsU: amount, density, seed, reserved (twinkle)
    gF[E + 1] = d.starDensity;
    gF[E + 2] = d.starSeed;
    gF[E + 3] = 0;
    gF[E + 4] = d.bandDir[0]; // bandU: plane normal + amount
    gF[E + 5] = d.bandDir[1];
    gF[E + 6] = d.bandDir[2];
    gF[E + 7] = d.band;
    gF[E + 8] = d.zenith[0]; // zenU: zenith color + on-blend
    gF[E + 9] = d.zenith[1];
    gF[E + 10] = d.zenith[2];
    gF[E + 11] = d.zenithOn;
    // IMGTEX (#631) tail rows — same contract as the ENVX rows above: always
    // written, read only by envMap/surfTex variants.
    const IM = EMAP_WORD * 4;
    gF[IM] = d.emapAmt; // emapU: amount, brightness, rotation, reserved
    gF[IM + 1] = d.emapBright;
    gF[IM + 2] = d.emapRot;
    gF[IM + 3] = 0;
    gF[IM + 4] = d.triAmt; // triU: amount, tile size, reserved ×2
    gF[IM + 5] = d.triScale;
    gF[IM + 6] = 0;
    gF[IM + 7] = 0;
    // AURORA (ENVX P6) tail rows — same contract as the ENVX/IMGTEX rows
    // above: always written, read only by aurora variants.
    const AU = AUR_WORD * 4;
    gF[AU] = d.auroraAmt; // aurU: aurora amt, nebula amt, reserved, drift
    gF[AU + 1] = d.nebulaAmt;
    gF[AU + 2] = 0;
    gF[AU + 3] = d.auroraDrift;
    gF[AU + 4] = d.auroraColA[0]; // aurA: curtain-floor color
    gF[AU + 5] = d.auroraColA[1];
    gF[AU + 6] = d.auroraColA[2];
    gF[AU + 7] = 0;
    gF[AU + 8] = d.auroraColB[0]; // aurB: curtain-tip / nebula-accent color
    gF[AU + 9] = d.auroraColB[1];
    gF[AU + 10] = d.auroraColB[2];
    gF[AU + 11] = 0;
    // CINE GRADE tail rows (gradeA..gradeD) — same contract as the ENVX/IMGTEX
    // rows above: always written, read ONLY by the graded post-pass variant
    // (see shader.js GRADE_WORD — rows 56..59, after the aurora rows).
    const GRW = GRADE_WORD * 4;
    for (let i = 0; i < 4; i++) {
      gF[GRW + i] = d.gradeA[i];
      gF[GRW + 4 + i] = d.gradeB[i];
      gF[GRW + 8 + i] = d.gradeC[i];
      gF[GRW + 12 + i] = d.gradeD[i];
    }
    gradeOn = d.gradeOn; // latch the post-pipeline pick for this frame
    // CLIP tail rows (clipU/clipS) — same contract as every tail above:
    // always written, read only by clip march variants (rows 60-61; 62 is
    // clip's spare, 63 is the equirect arm's reservation — see CLIP_WORD).
    const CL = CLIP_WORD * 4;
    gF[CL] = d.clipN[0]; // clipU: plane unit normal + offset along it
    gF[CL + 1] = d.clipN[1];
    gF[CL + 2] = d.clipN[2];
    gF[CL + 3] = d.clipW;
    gF[CL + 4] = d.clipShade; // clipS.x: cut-face gain (reserved, 1)
    gF[CL + 5] = d.clipJagAmp; // clipS.y: JAGGED noise amplitude (world units)
    gF[CL + 6] = d.clipJagFreq; // clipS.z: JAGGED noise frequency
    gF[CL + 7] = d.clipJagInv; // clipS.w: JAGGED Lipschitz divisor (1 = off)
    device.queue.writeBuffer(globalsBuf, 0, gBuf);
  }

  function writeOps(ops) {
    // Overflow/unknown-key are programmer errors — throw (like writeScene),
    // never silently truncate to MAX_OPS or write a garbage id from an undefined
    // def. Flat formulas are capped at MAX_FLAT_OPS (64) upstream in sanitize;
    // this buffer (MAX_OPS = 192) also holds hybrid/morph slot concatenations, so
    // the cap here is the physical buffer, not the flat cap.
    const n = ops.length;
    frameOps = []; // op-set for the variant (lever #3) — EXACT, from the ids below
    if (n === 0) return 0; // empty stack: nothing to upload (WebGPU rejects 0-byte writes)
    if (n > MAX_OPS) throw new Error(`writeOps: ${n} ops > cap ${MAX_OPS}`);
    const buf = new ArrayBuffer(n * OP_STRIDE);
    const u = new Uint32Array(buf);
    const f = new Float32Array(buf);
    // Overflow lane, parallel and same-indexed (§5.5) — built by the shared
    // pure packer so this path and writeScene's cannot disagree on ordering.
    const aux = packOpAuxLanes(ops);
    const seen = new Set();
    for (let i = 0; i < n; i++) {
      const def = byKey(ops[i].key);
      if (!def) throw new Error(`writeOps: unknown op key ${ops[i].key}`);
      u[i * 4 + 0] = def.id;
      f[i * 4 + 1] = ops[i].values[0] ?? 0;
      f[i * 4 + 2] = ops[i].values[1] ?? 0;
      f[i * 4 + 3] = ops[i].values[2] ?? 0;
      if (!seen.has(def.id)) {
        seen.add(def.id);
        frameOps.push(def.id);
      }
    }
    frameOps.sort((a, b) => a - b); // stable key regardless of op order
    device.queue.writeBuffer(opsBuf, 0, buf);
    device.queue.writeBuffer(opAuxBuf, 0, aux);
    return n;
  }

  // N-slot hybrid iteration (HYBRID_NSLOT_SPEC.md §2.3) — WGSL is data-driven
  // (the shared hybWalk helper slices the op buffer per iteration), so unlike
  // GLSL there's no codegen split: just concatenate EVERY slot's ops (A, B, C…)
  // in order onto the SAME shared ops buffer writeOps uses. The per-slot op
  // counts + schedule + addC that tell hybWalk where each slot's slice lives ride
  // the packed `hyb` globals word (writeGlobals → packHyb), not this call —
  // this only gets the op DATA into the buffer, so `counts`/`addC` are unused
  // here (they're the GL tier's concern; kept for the unified writeHybrid shape).
  function writeHybrid(slotOps, _counts, _addC) {
    return writeOps([].concat(...slotOps));
  }

  // Formula-morph spike — same concat trick as hybrid: A's ops then B's on the
  // shared buffer; the {opCount, morphB.x} split tells mapDE_morph where B
  // starts. Metadata rides writeGlobals' `morph` param, not this call.
  function writeMorph(opsA, opsB) {
    return writeOps([...opsA, ...opsB]);
  }

  // CSG Phase 1b — multi-object scene upload. Concatenates every object's ops
  // into the shared ops buffer (tracking per-object opStart/opCount) and packs an
  // Obj descriptor per object (full 80-B v1 layout; see shader.js). Bounds-checks
  // both caps and THROWS on overflow (never silently truncates). Returns the count.
  //
  // Each object (canonical ObjectSpec; flat-shape fallbacks kept for safety):
  //   { objType, ops?, iters?, addC?, deOption?, julia?, juliaC?, looseDE?,
  //     combine?, blendK?, primParam?, transform:{ origin:[x,y,z], uscale, rot } }
  //   objType 0 = IFS op-slice · 1 box · 2 sphere · 3 torus · 4 cylinder ·
  //     5 capsule · 6 plane. Multi-param prims (torus/cylinder/capsule) use
  //     primParam + primParam2 (see shader.js Obj word 4).
  //   transform.rot is Euler XYZ degrees (or a length-4 quaternion).
  function writeScene(objects) {
    if (!Array.isArray(objects) || objects.length === 0)
      throw new Error("writeScene: empty scene");
    if (objects.length > MAX_OBJECTS)
      throw new Error(
        `writeScene: ${objects.length} objects > cap ${MAX_OBJECTS}`,
      );

    // Canonical per-object metadata (fallback chains, active-op slice, quat) —
    // shared with renderer_gl.js/cpu.js via sceneobj.js normalizeSceneObject;
    // 3-emitter mirror discipline, guarded by core/scenemute.test.mjs.
    const norm = objects.map(normalizeSceneObject);
    // Leaf-variant latch (see activeMarch): the frame needs the leaf SDFs the
    // moment ANY object carries a leaf. A fractal-only scene (every shapeId 0)
    // skips them. writeGlobals runs AFTER this in the scene path and folds this
    // into activeFeat (it can't see shapeId itself).
    // EXACT leaf-id set the scene uses (lever #3 for leaves) — emit ONLY these
    // leaf SDFs, not all 58. A fractal-only scene (every shapeId 0) → null.
    const leafSet = new Set();
    for (const o of norm) {
      const id = o.shapeId & 0xff;
      if (id !== 0) leafSet.add(id);
    }
    frameLeafIds = leafSet.size ? [...leafSet].sort((a, b) => a - b) : null;

    // Pass 1 — concatenate ops, assign opStart/opCount, bounds-check the total.
    const opBuf = new ArrayBuffer(MAX_OPS * OP_STRIDE);
    const opU = new Uint32Array(opBuf);
    const opF = new Float32Array(opBuf);
    // The overflow lane for the WHOLE concatenation, built up front by the same
    // pure packer writeOps uses. Deliberately NOT written inside the loop below:
    // that loop's `i` restarts at 0 for every object (only `cursor` is the
    // global slot), so an inline `aux[i * 4 + …]` would alias every object after
    // the first onto object 0's lanes — the exact silent corruption
    // OP_PARAM_ENCODING.md §5.5 flags as this mechanism's likeliest bug. Pinned
    // by core/opaux.test.mjs with a fat op in a NON-FIRST object.
    const auxBuf = packOpAuxLanes(...norm.map((o) => o.ops));
    let cursor = 0;
    const slices = [];
    const opSeen = new Set(); // union of every object's ops → the variant op-set
    for (const o of norm) {
      const ops = o.ops;
      if (cursor + ops.length > MAX_OPS)
        throw new Error(
          `writeScene: concatenated ops ${cursor + ops.length} > cap ${MAX_OPS}`,
        );
      const start = cursor;
      for (let i = 0; i < ops.length; i++) {
        const def = byKey(ops[i].key);
        if (!def) throw new Error(`writeScene: unknown op "${ops[i].key}"`);
        const j = cursor * 4;
        opU[j + 0] = def.id;
        opF[j + 1] = ops[i].values?.[0] ?? 0;
        opF[j + 2] = ops[i].values?.[1] ?? 0;
        opF[j + 3] = ops[i].values?.[2] ?? 0;
        opSeen.add(def.id);
        cursor++;
      }
      slices.push({ start, count: ops.length });
    }
    frameOps = [...opSeen].sort((a, b) => a - b); // EXACT op-set for the variant (lever #3)
    if (cursor > 0) {
      device.queue.writeBuffer(opsBuf, 0, opBuf, 0, cursor * OP_STRIDE);
      device.queue.writeBuffer(opAuxBuf, 0, auxBuf, 0, cursor * 4);
    }

    // Pass 2 — pack the Obj descriptors (24 words = 96 B each) from the
    // canonical shape (bit layout in shader.js's Obj comment — D0 §2.3).
    const objBuf = new ArrayBuffer(objects.length * OBJ_STRIDE);
    const oU = new Uint32Array(objBuf);
    const oF = new Float32Array(objBuf);
    // #627 — full-capacity lane so stale slots from a previous (larger) scene
    // are zeroed on every write, like the descriptor buffer's own re-pack.
    const objAuxF = new Float32Array(MAX_OBJECTS * 4);
    norm.forEach((o, k) => {
      const base = k * 24;
      let flags = 0;
      if (o.addC) flags |= 1 << 0; // bit0 addC
      flags |= (o.deOption & 3) << 1; // bits1-2 deOption
      if (o.julia) flags |= 1 << 3; // bit3 julia
      if (o.looseDE) flags |= 1 << 4; // bit4 looseDE
      flags |= o.combine << 5; // bits5-6 combineType
      flags |= (o.objType & 0xf) << 7; // bits7-10 legacy objType (debug only — not read)
      // bit11 retired (was boxBase — the normalizer maps it to shapeId 1).
      flags |= (o.shapeId & 0xff) << 12; // bits12-19 shapeId (leaves.js)
      if (o.iterShape) flags |= 1 << 20; // bit20 D3 iterated-shape mode
      oU[base + 0] = slices[k].start;
      oU[base + 1] = slices[k].count;
      oU[base + 2] = o.iters;
      oU[base + 3] = flags;
      oF[base + 4] = o.origin[0];
      oF[base + 5] = o.origin[1];
      oF[base + 6] = o.origin[2];
      oF[base + 7] = o.uscale;
      oF[base + 8] = o.quat[0];
      oF[base + 9] = o.quat[1];
      oF[base + 10] = o.quat[2];
      oF[base + 11] = o.quat[3];
      oF[base + 12] = o.juliaC[0];
      oF[base + 13] = o.juliaC[1];
      oF[base + 14] = o.juliaC[2];
      oF[base + 15] = o.blendK; // smin blend
      // word 4: the leaf's shapeParams block (leaves.js param order).
      oF[base + 16] = o.shapeParams[0];
      oF[base + 17] = o.shapeParams[1];
      oF[base + 18] = o.shapeParams[2];
      oF[base + 19] = o.shapeParams[3];
      // word 5: per-object albedo (sRGB; shader applies s2l). §3.8 per-object color.
      oF[base + 20] = o.color[0];
      oF[base + 21] = o.color[1];
      oF[base + 22] = o.color[2];
      oF[base + 23] = 0; // pad3
      // #627 — the leaf-param overflow lane rides a PARALLEL buffer at the
      // same object index k (never a second cursor: the opAux §5.5 hazard has
      // no analogue here because this loop has exactly one index). Slots an
      // object doesn't declare stay 0.
      objAuxF[k * 4 + 0] = o.shapeParams[4] ?? 0;
      objAuxF[k * 4 + 1] = o.shapeParams[5] ?? 0;
      objAuxF[k * 4 + 2] = o.shapeParams[6] ?? 0;
      objAuxF[k * 4 + 3] = o.shapeParams[7] ?? 0;
    });
    device.queue.writeBuffer(objectsBuf, 0, objBuf);
    device.queue.writeBuffer(objAuxBuf, 0, objAuxF);
    return objects.length;
  }

  // Cached HDR intermediates for the interactive paths (draw/drawTo re-ensure
  // on size change; renderToImage builds its own per call) — held in the
  // two-slot BUNDLE cache below (see ensureHdr).
  function makeHdr(w, h) {
    const mk = (mw, mh) =>
      device.createTexture({
        size: [mw, mh],
        format: HDR_FORMAT,
        usage:
          GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
    const tex = mk(w, h);
    const view = tex.createView();
    // P3 bloom half-res pair. Zero-initialized (WebGPU) → composites nothing
    // until the bloom passes actually run; the post pass always binds bloomA.
    const bw = Math.max(1, Math.ceil(w / 2)),
      bh = Math.max(1, Math.ceil(h / 2));
    const bloomA = mk(bw, bh),
      bloomB = mk(bw, bh);
    const bloomAView = bloomA.createView(),
      bloomBView = bloomB.createView();
    // NOTE on layout:'auto': a binding lands in the layout only if the entry
    // point STATICALLY USES it. fs_bright reads G (threshold) + src; the blur
    // entry points read ONLY src — their layouts contain just binding 1, and
    // passing binding 0 anyway makes the bind group (and every submit that
    // touches it) invalid. Bit us: black canvas, no exception.
    const brightBindFor = (src) =>
      device.createBindGroup({
        layout: brightPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: globalsBuf } },
          { binding: 1, resource: src },
        ],
      });
    const blurBind = (pl, src) =>
      device.createBindGroup({
        layout: pl.getBindGroupLayout(0),
        entries: [{ binding: 1, resource: src }],
      });
    // CINE GRADE: the post bind is a GETTER keyed on the post pipeline in use
    // (base vs graded — layout:'auto' gives each its own layout object). The
    // base bind is built eagerly exactly as before; a grade bind materializes
    // only the first time a graded frame resolves through this hdr, and both
    // die with the hdr object (bundle eviction) like the old single bind did.
    const postBindFor = (src) => {
      const cache = new Map(); // pipeline → bind group
      return (pl) => {
        let bg = cache.get(pl);
        if (!bg) {
          bg = device.createBindGroup({
            layout: pl.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: globalsBuf } },
              { binding: 1, resource: src },
              { binding: 2, resource: bloomAView },
              { binding: 3, resource: bloomSampler },
            ],
          });
          cache.set(pl, bg);
        }
        return bg;
      };
    };
    return {
      tex,
      view,
      w,
      h,
      postBind: postBindFor(view),
      postBindFor,
      bloomAView,
      bloomBView,
      brightBind: brightBindFor(view), // bright reads the plain HDR
      brightBindFor, // …or an accum half
      blurHBind: blurBind(blurHPipeline, bloomAView),
      blurVBind: blurBind(blurVPipeline, bloomBView),
      destroy() {
        tex.destroy();
        bloomA.destroy();
        bloomB.destroy();
      },
    };
  }
  // Two-slot LRU of {hdr, accum} BUNDLES keyed by size. The interactive tier
  // flips the canvas between 0.8× (dragging) and 1.0× (settled) on every
  // gesture; the old single-slot caches destroyed and reallocated the HDR +
  // bloom pair AND the rgba32float accumulation ping-pong (~44 B/px together)
  // twice per drag — driver-allocation churn landing exactly on the settle
  // frame. Two slots keep both sizes alive across a gesture cycle. hdr and
  // accum live and die TOGETHER, preserving the invariant the old
  // hdr-object-keyed ensureAccum guarded: an accum's bind groups can never
  // outlive the hdr texture they reference (the destroyed-texture black-canvas
  // bug). The accum half is created lazily, so the interactive-size slot —
  // which never draws a settled frame — skips its two full-res rgba32float
  // allocations entirely.
  const BUNDLES_MAX = 2;
  const bundles = new Map(); // "w×h" → { hdr, accum: null until first settle }
  function ensureHdr(w, h) {
    const key = w + "x" + h;
    let b = bundles.get(key);
    if (b) {
      bundles.delete(key); // LRU refresh: re-insert as newest
    } else {
      b = { hdr: makeHdr(w, h), accum: null };
      while (bundles.size >= BUNDLES_MAX) {
        const oldest = bundles.keys().next().value;
        const old = bundles.get(oldest);
        bundles.delete(oldest);
        old.accum?.destroy();
        old.hdr.destroy();
      }
    }
    bundles.set(key, b);
    return b.hdr;
  }
  // P3: encode bright → blurH → blurV (half res). `brightBind` selects the
  // source (plain HDR or the just-written accumulation half).
  function encodeBloom(enc, h, brightBind) {
    encodeOnePass(enc, brightPipeline, brightBind, h.bloomAView);
    encodeOnePass(enc, blurHPipeline, h.blurHBind, h.bloomBView);
    encodeOnePass(enc, blurVPipeline, h.blurVBind, h.bloomAView);
  }
  // P2 accumulation ping-pong pair (rgba32float running average) + the bind
  // groups that read/write it. Cached like hdr; renderToImage builds its own.
  function makeAccum(h) {
    const mk = () =>
      device.createTexture({
        size: [h.w, h.h],
        format: ACCUM_FORMAT,
        usage:
          GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
    const texA = mk(),
      texB = mk();
    const viewA = texA.createView(),
      viewB = texB.createView();
    const accumBindFor = (prevView) =>
      device.createBindGroup({
        layout: accumPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: globalsBuf } },
          { binding: 1, resource: h.view },
          { binding: 2, resource: prevView },
        ],
      });
    return {
      texA,
      texB,
      viewA,
      viewB,
      w: h.w,
      h: h.h,
      flip: false, // false → next write lands in A (prev = B)
      bindToA: accumBindFor(viewB),
      bindToB: accumBindFor(viewA),
      // Post + bloom-bright binds come from the hdr object so they carry the
      // P3 bloom texture/sampler entries (post layout requires them).
      postA: h.postBindFor(viewA),
      postB: h.postBindFor(viewB),
      brightA: h.brightBindFor(viewA),
      brightB: h.brightBindFor(viewB),
      destroy() {
        texA.destroy();
        texB.destroy();
      },
    };
  }
  function ensureAccum(h) {
    // `h` is always the hdr ensureHdr just returned, so its bundle is live —
    // matching by OBJECT identity (not size) keeps the old guarantee that an
    // accum is never revived against a destroyed hdr.
    for (const b of bundles.values())
      if (b.hdr === h) {
        if (!b.accum) b.accum = makeAccum(h);
        return b.accum;
      }
    throw new Error("ensureAccum: hdr is not in the bundle cache");
  }
  // Per-sample partial uploads: the jitter word (subpixel offset + accum
  // weight) and the dof word's zw (this sample's lens point — P4).
  function writeJitter(jx, jy, weight, lensX = 0, lensY = 0) {
    const j = new Float32Array([jx, jy, weight, 0]);
    device.queue.writeBuffer(globalsBuf, JITTER_WORD * 16, j);
    const l = new Float32Array([lensX, lensY]);
    device.queue.writeBuffer(globalsBuf, DOF_WORD * 16 + 8, l);
  }

  // Per-tile partial upload (TILED_EXPORT §2.1.3 / §2.2.1(a)) — the off-axis
  // window and the tile's absolute rect, written as ONE 8-float run because
  // `tile` and `tilepx` are adjacent words. Same shape as writeJitter: the rest
  // of the globals (camera, quality, coloring) are decided once for the whole
  // export and must NOT be re-derived per tile (§2.2).
  //
  //   win = { sx, sy, bx, by }  from tilegrid.tileWindow()
  //   px  = [rx0, ry0, W, H]    the RENDERED origin and the FULL-frame size;
  //                             W = 0 turns the absolute-position blocks off.
  //
  // Call writeTile(TILE_WINDOW_IDENTITY, [0,0,0,0]) to restore — or just let the
  // next writeGlobals() do it, which it always does.
  // Partial upload of the P0 post word (tone / exposure / dither / vignette).
  // writeGlobals already writes it on every full write, but NOTHING in the app
  // supplies a `post` argument today — every frame ships the defaults, so
  // `vignette` in particular has no producer anywhere in the tree. That makes
  // the post pass's absolute-position fix (TILED_EXPORT §2.2.1(a)) impossible to
  // SEE without a way to set it: at the default 1-LSB dither the only other
  // consumer moves by ±0.5/255. Used by the dev tile probe's manual gate; the
  // shape mirrors writeJitter/writeTile.
  function writePost({
    tone = 1,
    exposure = 0,
    dither = 1,
    vignette = 0,
  } = {}) {
    device.queue.writeBuffer(
      globalsBuf,
      POST_WORD * 16,
      new Float32Array([tone, exposure, dither, vignette]),
    );
  }

  function writeTile(win, px) {
    const t = new Float32Array([
      win.sx,
      win.sy,
      win.bx,
      win.by,
      px[0],
      px[1],
      px[2],
      px[3],
    ]);
    device.queue.writeBuffer(globalsBuf, TILE_WORD * 16, t);
  }

  // Encode the two-pass sequence: march → HDR intermediate, post → target view.
  function encodeOnePass(enc, pl, bg, view) {
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
    pass.setPipeline(pl);
    pass.setBindGroup(0, bg);
    pass.draw(3);
    pass.end();
  }
  function encodePasses(enc, h, targetView, skipMarch) {
    if (!skipMarch)
      encodeOnePass(enc, activeMarch().pl, activeMarch().bg, h.view); // march → HDR
    if (bloomOn) encodeBloom(enc, h, h.brightBind); // P3 (skipped at glow 0)
    const pp = activePost(); // CINE GRADE — graded variant only when a look is on
    encodeOnePass(enc, pp, h.postBind(pp), targetView); // post → target
  }
  // Banded march (#212): render ONLY rows [i/n, (i+1)/n) of the march pass
  // into the HDR intermediate, one submit per band. A settled frame of a heavy
  // formula on a big canvas is otherwise ONE dispatch taking seconds of GPU —
  // which starves Chrome's compositor (shared by every tab): the whole browser
  // freezes. Bands bound the largest single submit; the caller (preview.js
  // pump) yields a rAF between bands and finishes with draw/drawAccum
  // ({ skipMarch: true }) so bloom/accum/post still run once over the full
  // frame. The fullscreen-triangle vertex shader is band-agnostic — the
  // scissor alone confines the fragment work.
  function drawMarchBand(i, n) {
    // Size from the canvas, NOT getCurrentTexture(): bands span several
    // animation frames, and acquiring a fresh swapchain texture per band
    // without presenting it just churns the swapchain. Only the resolve
    // (draw/drawAccum with skipMarch) touches the presentable texture.
    const h = ensureHdr(ctx.canvas.width, ctx.canvas.height);
    // Band geometry comes from renderpolicy.bandRect keyed on the HDR's OWN
    // height — the target actually being marched — so an explicit-size still
    // export (EXPORT_SIZE) bands over the export height, never a stale or
    // live-canvas one. Tiling is pinned in renderpolicy.test.mjs.
    const { y0, h: bandH } = bandRect(i, n, h.h);
    if (bandH <= 0) return;
    const enc = device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [
        {
          view: h.view,
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          // First band clears the whole target; later bands must PRESERVE the
          // rows already marched.
          loadOp: i === 0 ? "clear" : "load",
          storeOp: "store",
        },
      ],
    });
    pass.setPipeline(activeMarch().pl);
    pass.setBindGroup(0, activeMarch().bg);
    pass.setScissorRect(0, y0, h.w, bandH);
    pass.draw(3);
    pass.end();
    device.queue.submit([enc.finish()]);
  }
  // P2 accumulation frame: march → HDR, blend into the ping-pong average
  // (weight came in via writeJitter — 1 replaces, 1/(N+1) refines), post reads
  // the average → swap chain. The caller owns the sample counter.
  // opts.skipMarch: the HDR already holds a banded march (drawMarchBand) —
  // run only the blend/bloom/post resolve.
  // opts.target (TILED_EXPORT §2.4.3): resolve into a CALLER-OWNED texture
  // instead of the swap chain — `{ view, w, h }`, as produced by
  // createTileTarget() below. The swap chain is deliberately not touched at all
  // on that path: `getCurrentTexture()` acquires a presentable image whose
  // contents are gone again at the next frame boundary, which is precisely why
  // reading a tile back off the canvas is unreliable (see createTileTarget).
  function drawAccum(opts) {
    const tgt = opts?.target;
    const tex = tgt ? null : ctx.getCurrentTexture();
    const h = ensureHdr(tgt ? tgt.w : tex.width, tgt ? tgt.h : tex.height);
    const ac = ensureAccum(h);
    const enc = device.createCommandEncoder();
    if (!opts?.skipMarch)
      encodeOnePass(enc, activeMarch().pl, activeMarch().bg, h.view);
    encodeOnePass(
      enc,
      accumPipeline,
      ac.flip ? ac.bindToB : ac.bindToA,
      ac.flip ? ac.viewB : ac.viewA,
    );
    if (bloomOn) encodeBloom(enc, h, ac.flip ? ac.brightB : ac.brightA); // read the just-written half
    const pp = activePost(); // CINE GRADE — graded variant only when a look is on
    encodeOnePass(
      enc,
      pp,
      (ac.flip ? ac.postB : ac.postA)(pp),
      tgt ? tgt.view : tex.createView(),
    );
    // Field streamlines (lab): overlay on the LIVE canvas only — a tiled-export
    // target must stay clean. WYSIWYG: an accumulated still that resolves to
    // the live canvas (stillBlob) captures the overlay like the screen shows it.
    if (stream?.on && !tgt) stream.encodeLive(enc, tex, activeFeat);
    ac.flip = !ac.flip;
    device.queue.submit([enc.finish()]);
  }

  // ── Tiled-export render target (TILED_EXPORT.md §2.4.3) ────────────────────
  // One offscreen `format` texture at the uniform tile size, plus one aligned
  // readback buffer, both allocated ONCE for the whole export and reused by
  // every tile. The tile loop renders into `view` (via drawMarchBand into the
  // shared HDR + drawAccum({ target })) and pulls the committed sub-rect back
  // with copyTextureToBuffer.
  //
  // WHY NOT read the canvas. PR-1 measured it: `createImageBitmap(canvas)` on a
  // WebGPU canvas returned a fully TRANSPARENT bitmap for 1 tile in 4 at 2048²
  // and for EVERY tile at 512², and yielding a rAF first made it worse (after a
  // frame boundary the swap-chain image is gone, so every tile read blank) —
  // a silently missing rectangle, not an error. `canvas.toBlob` forces the
  // readback and works, but costs a full PNG encode per tile, which is exactly
  // the cost the streaming encoder exists to avoid. An owned texture removes
  // the whole class: nothing presents it, so nothing can recycle it out from
  // under us, and the copy is an ordinary queue command that completes with the
  // fence the tile loop already awaits.
  //
  // Memory: the texture is ≤ TILE_AREA_MAX (12 Mpx → 48 MB) and the buffer is
  // its aligned twin — both are small beside the ~44 B/px HDR+accum bundle the
  // same tile size already requires.
  function createTileTarget(w, h) {
    const tex = device.createTexture({
      size: [w, h],
      format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const view = tex.createView();
    // 256-byte row alignment (the renderToImage pattern). Sized from the FULL
    // tile width so one buffer serves every crop: a narrower copy is legal at
    // any bytesPerRow ≥ its own row length.
    const bytesPerRow = Math.ceil((w * 4) / 256) * 256;
    const buf = device.createBuffer({
      size: bytesPerRow * h,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const bgra = format.startsWith("bgra"); // preferred format may be BGRA
    let destroyed = false;

    // Copy [x, y, cw, ch] of the tile back as tightly-packed RGBA. `wantAlpha`
    // false forces opaque, matching renderToImage — the shader writes real
    // hit/miss alpha (#428) and every opaque consumer expects 255.
    //
    // Returns { data, blank }: `blank` means every byte of the RAW copy was
    // zero. On the opaque path that is the PR-1 failure signature and the
    // caller treats it as an error; on the alpha path a fully transparent black
    // region is legitimate, so the caller must not.
    async function read(x, y, cw, ch, wantAlpha) {
      if (destroyed) throw new Error("tile target: read after destroy");
      const size = bytesPerRow * ch;
      const enc = device.createCommandEncoder();
      enc.copyTextureToBuffer(
        { texture: tex, origin: { x, y } },
        { buffer: buf, bytesPerRow },
        { width: cw, height: ch },
      );
      device.queue.submit([enc.finish()]);
      // One submit (the copy above) — the tile's render submits were already
      // fenced by preview.js's own guarded fence before read() is called.
      await mapGuarded("tile-read", 1, [
        buf.mapAsync(GPUMapMode.READ, 0, size),
      ]);
      const src = new Uint8Array(buf.getMappedRange(0, size));
      // The pixel loop is pure and lives in tilegrid.js, where a Node test can
      // reach it — the "an opaque readback is never all-transparent" invariant
      // is otherwise unpinnable, since nothing in CI has a GPU.
      const r = readbackToRGBA(src, bytesPerRow, cw, ch, bgra, wantAlpha);
      buf.unmap();
      return r;
    }

    return {
      view,
      w,
      h,
      read,
      destroy() {
        if (destroyed) return;
        destroyed = true;
        buf.destroy();
        tex.destroy();
      },
    };
  }

  // Render into a target context (defaults to the main canvas). Thumbnails pass
  // their own offscreen context here, reusing the same pipelines + buffers.
  function drawTo(target, opts) {
    const t = target || ctx;
    const tex = t.getCurrentTexture();
    const h = ensureHdr(tex.width, tex.height);
    const enc = device.createCommandEncoder();
    encodePasses(enc, h, tex.createView(), opts?.skipMarch);
    // Field streamlines (lab): live canvas only — thumbnail contexts stay clean.
    if (stream?.on && t === ctx) stream.encodeLive(enc, tex, activeFeat);
    device.queue.submit([enc.finish()]);
  }
  function draw(opts) {
    drawTo(ctx, opts);
  }

  // Configure an extra canvas's context to share this device/format so drawTo()
  // can render into it (used by the preset thumbnail gallery).
  function configureContext(targetCanvas) {
    const c = targetCanvas.getContext("webgpu");
    c.configure({ device, format, alphaMode: "opaque" });
    return c;
  }

  // Render the current globals/ops into an offscreen texture and read the pixels
  // straight back as ImageData. Unlike drawing to a canvas + drawImage(), this is
  // DETERMINISTIC — no canvas-presentation lag — so thumbnail captures never grab
  // a stale frame. Used by the preset gallery.
  async function renderToImage(
    W,
    H,
    samples = 1,
    wantAlpha = false,
    bakeDOF = true,
    // Field streamlines (lab): > 0 composites the particle overlay into this
    // offscreen frame, advancing the sim by EXACTLY this many seconds — the
    // export's virtual clock, deterministic stepping, never wall time. 0 (the
    // default every existing caller keeps — thumbnails, splat framing) means
    // no overlay pass at all. Only preview.captureFrame passes it, and only
    // while the live toggle is on.
    overlayDt = 0,
  ) {
    const tex = device.createTexture({
      size: [W, H],
      format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    // Own HDR intermediate per call (requested size ≠ canvas size); destroyed
    // below with the readback texture. Same two-pass sequence as draw/drawTo,
    // so exports match the screen exactly.
    const h = makeHdr(W, H);
    // P2 multi-sample export: N jittered march+accumulate rounds (one submit
    // each — writeJitter is a queue write, so per-sample values need per-sample
    // submits), then post reads the average. Sample 0 is unjittered with
    // weight 1, matching the live accumulation sequence exactly.
    // bakeDOF (#509): gates the per-sample lens offset the SAME way
    // preview.js's own stillBlob accumulate loop does — a caller rendering a
    // heavy formula whose live view never converges DOF (accumCap()==0) must
    // not bake lens-jittered blur into the image; every EXISTING caller
    // (captureFrame, thumbnails) omits this arg and keeps today's unconditional
    // lens jitter, unchanged.
    let ac = null;
    if (samples > 1 && !Number.isNaN(samples)) {
      ac = makeAccum(h);
      for (let i = 0; i < samples; i++) {
        const [jx, jy] = i === 0 ? [0, 0] : r2jitter(i);
        const [lx, ly] = i === 0 || !bakeDOF ? [0, 0] : lensSample(i);
        writeJitter(jx, jy, 1 / (i + 1), lx, ly);
        const e = device.createCommandEncoder();
        encodeOnePass(e, activeMarch().pl, activeMarch().bg, h.view);
        encodeOnePass(
          e,
          accumPipeline,
          ac.flip ? ac.bindToB : ac.bindToA,
          ac.flip ? ac.viewB : ac.viewA,
        );
        ac.flip = !ac.flip;
        device.queue.submit([e.finish()]);
      }
      writeJitter(0, 0, 0, 0, 0); // restore — live frames render unjittered, lens-centered
    }
    const bytesPerRow = Math.ceil((W * 4) / 256) * 256; // 256-byte row alignment
    const buf = device.createBuffer({
      size: bytesPerRow * H,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = device.createCommandEncoder();
    if (ac) {
      // Post resolves the accumulated average (the LAST-written half: flip
      // toggled after the final write, so read the half `flip` now points AWAY
      // from — texA when flip is true).
      if (bloomOn) encodeBloom(enc, h, ac.flip ? ac.brightA : ac.brightB);
      const pp = activePost(); // CINE GRADE — exports resolve through the same variant
      encodeOnePass(
        enc,
        pp,
        (ac.flip ? ac.postA : ac.postB)(pp),
        tex.createView(),
      );
    } else {
      encodePasses(enc, h, tex.createView());
    }
    // Field streamlines (lab): offline composite, after the post resolve and
    // before the readback — the same ordering encodeLive uses on the live
    // canvas. Alpha exports stay clean (additive glow has no meaningful
    // coverage, so it would silently vanish on re-composite anyway).
    if (overlayDt > 0 && !wantAlpha && stream?.on)
      stream.encodeOffline(enc, tex, activeFeat, overlayDt);
    enc.copyTextureToBuffer(
      { texture: tex },
      { buffer: buf, bytesPerRow },
      { width: W, height: H },
    );
    device.queue.submit([enc.finish()]);

    // The map cannot resolve until every queued submit retires: the N
    // per-sample accumulate submits (when accumulating) plus this one.
    await mapGuarded("render-to-image", (ac ? samples : 0) + 1, [
      buf.mapAsync(GPUMapMode.READ),
    ]);
    const src = new Uint8Array(buf.getMappedRange());
    const out = new Uint8ClampedArray(W * H * 4);
    const bgra = format.startsWith("bgra"); // preferred format may be BGRA; ImageData is RGBA
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const s = y * bytesPerRow + x * 4,
          d = (y * W + x) * 4;
        out[d] = bgra ? src[s + 2] : src[s];
        out[d + 1] = src[s + 1];
        out[d + 2] = bgra ? src[s] : src[s + 2];
        // The shader now writes real hit/miss alpha (issue #428) — force back
        // to opaque unless the caller explicitly opted in, so every existing
        // consumer (thumbnails, mp4 frames, stillBlob-adjacent callers) keeps
        // its current byte-for-byte opaque output.
        out[d + 3] = wantAlpha ? src[s + 3] : 255;
      }
    }
    buf.unmap();
    buf.destroy();
    tex.destroy();
    h.destroy();
    ac?.destroy();
    return new ImageData(out, W, H);
  }

  // ── §S2 GPU splat capture ──────────────────────────────────────────────────
  // A self-contained capture session for one export (docs/planning/
  // UE_SPLAT_S2_IMPL.md §3.1). Compiles the capture pipeline SPECIALIZED to
  // `feat` (fsCapture + 3-target MRT), OUTSIDE marchVariants — different fragment
  // targets, must never evict/be-evicted. Owns its textures + readback buffers;
  // captureView() runs one Fibonacci view (all peel layers), reconstructs world
  // hits (p_rel + fround(O)); dispose() frees. Returns null when the capture
  // pipeline can't build (no async pipelines / compile error) ⇒ CPU fallback.
  async function createSplatCapture(feat, res) {
    let pl;
    try {
      const mod = device.createShaderModule({
        // TINY PLANET is forced OFF here: fsCapture marches CaptureU's own
        // orthographic view rays, never the live fs ray-gen, so carrying the
        // flag would only fork the capture blob onto a second key for a code
        // path it does not use. A splat capture is projection-independent by
        // construction (the #441 note in splatcapture.js says the same).
        // CLIP is forced OFF for the same reason a splat capture ignores the
        // projection: fsCapture marches its own volume rays and a capture
        // means the OBJECT, not the current inspection cut — a clipped splat
        // export is deferred (v1, documented in the clipplane row). The
        // jagged sub-variant goes with it, and EQUIRECT is forced off on the
        // same argument word for word.
        code: buildWGSL({
          ...wgslOf(feat),
          clip: false,
          clipJag: false,
          planet: false,
          equirect: false,
          capture: true,
        }),
      });
      pl = await timedPipelineAsync("capture:" + keyFor(feat), {
        layout: "auto",
        vertex: { module: mod, entryPoint: "vs" },
        fragment: {
          module: mod,
          entryPoint: "fsCapture",
          targets: [
            { format: "rgba32float" }, // posT: p_rel.xyz, t
            { format: "rgba16float" }, // aux: normal.xyz, budget
            { format: "rgba16float" }, // alb: albedo.xyz, 1
          ],
        },
        primitive: { topology: "triangle-list" },
      });
    } catch {
      return null; // no createRenderPipelineAsync / validation error ⇒ CPU path
    }
    if (!pl) return null;

    const TEX =
      GPUTextureUsage.RENDER_ATTACHMENT |
      GPUTextureUsage.COPY_SRC |
      GPUTextureUsage.TEXTURE_BINDING;
    const mk = (fmt, w = res, h = res, usage = TEX) =>
      device.createTexture({ size: [w, h], format: fmt, usage });
    // ping-pong pos+aux (layer ℓ writes set ℓ%2, reads set (ℓ−1)%2); single alb.
    const posT = [mk("rgba32float"), mk("rgba32float")];
    const auxT = [mk("rgba16float"), mk("rgba16float")];
    const albT = mk(
      "rgba16float",
      res,
      res,
      GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    );
    const dumPos = mk("rgba32float", 1, 1, GPUTextureUsage.TEXTURE_BINDING);
    const dumAux = mk("rgba16float", 1, 1, GPUTextureUsage.TEXTURE_BINDING);
    const capU = device.createBuffer({
      size: CAPTURE_U_WORDS * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const bind = (pPos, pAux) =>
      device.createBindGroup({
        layout: pl.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: globalsBuf } },
          { binding: 1, resource: { buffer: opsBuf } },
          { binding: 2, resource: { buffer: objectsBuf } },
          { binding: 3, resource: { buffer: capU } },
          { binding: 4, resource: pPos.createView() },
          { binding: 5, resource: pAux.createView() },
          // capture parity (plan D9): a perturb capture module declares the
          // reference-orbit records too — same buffer the march variant uses
          ...(feat.perturb
            ? [{ binding: 6, resource: { buffer: ptBuf } }]
            : []),
          // same for the overflow lane — the capture module is built from the
          // SAME wgslOf(feat).ops, so it declares opAux on the same predicate
          ...(usesOpAux(wgslOf(feat).ops)
            ? [{ binding: 7, resource: { buffer: opAuxBuf } }]
            : []),
          // #627: the leaf lane, same SAME-wgslOf(feat) predicate discipline
          ...(usesObjAux(wgslOf(feat).leaves)
            ? [{ binding: 8, resource: { buffer: objAuxBuf } }]
            : []),
        ],
      });
    const bg0 = bind(dumPos, dumAux); // layer 0: dummy inputs
    const bgReadSet = [bind(posT[0], auxT[0]), bind(posT[1], auxT[1])]; // read set 0 / 1
    // 256-byte-aligned readback rows (renderToImage pattern): 16 B/texel (pos), 8 B (half).
    const rowPos = Math.ceil((res * 16) / 256) * 256;
    const rowHalf = Math.ceil((res * 8) / 256) * 256;
    const rb = (row) =>
      device.createBuffer({
        size: row * res,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
    const rbPos = rb(rowPos);
    const rbAux = rb(rowHalf);
    const rbAlb = rb(rowHalf);

    async function captureView(viewDir, frame, O, r, opts = {}) {
      const { d, rgt, up, oc, hu, hv, hd } = viewBasis(viewDir, frame);
      const originRel = [oc[0] - O[0], oc[1] - O[1], oc[2] - O[2]]; // f64 residual
      const froundO = [Math.fround(O[0]), Math.fround(O[1]), Math.fround(O[2])];
      // radius stays the SCALE scalar (eps, AO probe); the capture VOLUME is
      // ext/kind and the per-view window is hu/hv (CAPTURE_VOLUME_SHAPES.md).
      // Shared with the CPU march — and it carries both floor terms (#496's
      // absolute framing floor, #507's measured convergence floor), which is
      // how the GPU tier gets both fixes without a line of WGSL changing
      // (fsCapture takes eps as a uniform).
      const eps = captureEps(frame);
      const tmax = 3 * hd;
      const layers = Math.max(1, opts.layers ?? 1);
      const out = { pos: [], normal: [], albedo: [] };
      const posStride = rowPos / 4; // f32 per row
      const halfStride = rowHalf / 2; // u16 per row

      for (let layer = 0; layer < layers; layer++) {
        device.queue.writeBuffer(
          capU,
          0,
          packCaptureUniform({
            d,
            rgt,
            up,
            originRel,
            radius: frame.radius,
            eps,
            tmax,
            ext: volExt(frame),
            kind: volKind(frame),
            rot: volBasis(frame),
            hu,
            hv,
            layerIndex: layer,
            deScale: opts.deScale ?? 1,
            aoStrength: opts.aoStrength ?? 0,
            maxSteps: opts.maxSteps ?? 200,
            layers,
          }),
        );
        const wi = layer % 2;
        const bg = layer === 0 ? bg0 : bgReadSet[(layer - 1) % 2];
        const enc = device.createCommandEncoder();
        const pass = enc.beginRenderPass({
          colorAttachments: [
            {
              view: posT[wi].createView(),
              loadOp: "clear",
              storeOp: "store",
              clearValue: { r: 0, g: 0, b: 0, a: -1 },
            },
            {
              view: auxT[wi].createView(),
              loadOp: "clear",
              storeOp: "store",
              clearValue: { r: 0, g: 0, b: 0, a: 0 },
            },
            {
              view: albT.createView(),
              loadOp: "clear",
              storeOp: "store",
              clearValue: { r: 0, g: 0, b: 0, a: 0 },
            },
          ],
        });
        pass.setPipeline(pl);
        pass.setBindGroup(0, bg);
        pass.draw(3);
        pass.end();
        enc.copyTextureToBuffer(
          { texture: posT[wi] },
          { buffer: rbPos, bytesPerRow: rowPos },
          { width: r, height: r },
        );
        enc.copyTextureToBuffer(
          { texture: auxT[wi] },
          { buffer: rbAux, bytesPerRow: rowHalf },
          { width: r, height: r },
        );
        enc.copyTextureToBuffer(
          { texture: albT },
          { buffer: rbAlb, bytesPerRow: rowHalf },
          { width: r, height: r },
        );
        device.queue.submit([enc.finish()]);
        // One submit per layer iteration. Promise.all made a hang strictly
        // worse — any ONE of the three not settling hung all of them, once per
        // view × layer (128 chances on a default splat export).
        await mapGuarded("splat-capture", 1, [
          rbPos.mapAsync(GPUMapMode.READ),
          rbAux.mapAsync(GPUMapMode.READ),
          rbAlb.mapAsync(GPUMapMode.READ),
        ]);
        const P = new Float32Array(rbPos.getMappedRange());
        const A = new Uint16Array(rbAux.getMappedRange());
        const C = new Uint16Array(rbAlb.getMappedRange());
        for (let y = 0; y < r; y++) {
          for (let x = 0; x < r; x++) {
            const pi = y * posStride + x * 4;
            if (P[pi + 3] < 0) continue; // miss (posT.w < 0)
            const ai = y * halfStride + x * 4;
            const nx = f16ToF32(A[ai]),
              ny = f16ToF32(A[ai + 1]),
              nz = f16ToF32(A[ai + 2]);
            if (nx === 0 && ny === 0 && nz === 0) continue; // degenerate/dropped
            out.pos.push(
              P[pi] + froundO[0],
              P[pi + 1] + froundO[1],
              P[pi + 2] + froundO[2],
            );
            out.normal.push(nx, ny, nz);
            out.albedo.push(
              f16ToF32(C[ai]),
              f16ToF32(C[ai + 1]),
              f16ToF32(C[ai + 2]),
            );
          }
        }
        rbPos.unmap();
        rbAux.unmap();
        rbAlb.unmap();
      }
      // Return compact Float32Arrays, not the plain JS arrays (2× the bytes/elem):
      // the caller merges many views, so keeping these typed halves resident memory
      // and lets it concat with set() instead of a giant push + Float32Array.from.
      return {
        pos: Float32Array.from(out.pos),
        normal: Float32Array.from(out.normal),
        albedo: Float32Array.from(out.albedo),
      };
    }

    function dispose() {
      for (const t of [...posT, ...auxT, albT, dumPos, dumAux]) t.destroy();
      for (const b of [capU, rbPos, rbAux, rbAlb]) b.destroy();
    }

    return { captureView, dispose };
  }

  // §S2.5 — force the deep-zoom offset for a capture, overriding what
  // writeGlobals derived from the camera. The next live frame's writeGlobals
  // restores it (capture runs under setOffline, so no live frame interleaves).
  // O must match the residual origin the capture marches in.
  // Phase 4: writes BOTH halves of the hi/lo pair (same splitHiLo as
  // writeGlobals) — patching only the hi word would leave offsetLo holding the
  // last live frame's DIFFERENT O, an internally inconsistent pair that
  // silently regresses deep captures to the f32 wall (spec §4a-5). kStar
  // stays 0 here until the capture path learns df64 (plan PR-3).
  function overrideCaptureOffset(O, kStar = 0) {
    const s = splitHiLo(O);
    device.queue.writeBuffer(
      globalsBuf,
      76 * 4,
      new Float32Array([s.hi[0], s.hi[1], s.hi[2], 0]),
    );
    device.queue.writeBuffer(
      globalsBuf,
      OFFSETLO_WORD * 16,
      new Float32Array([s.lo[0], s.lo[1], s.lo[2], kStar]),
    );
  }

  // Perturbation reference-orbit upload (PERTURBATION_ZOOM_IMPL.md PR-2).
  // `packed` = core/perturb.js buildOrbit().packed — (iters·opCount + 1)
  // slots of 16 f32. Uploaded on re-pin / iters change (plan D5), never per
  // frame. The pt shader derives every index from G.ctrl (iters, opCount),
  // so the caller must keep the orbit and the frame's ctrl words describing
  // the SAME formula+iters — the engagement wiring (plan PR-4) owns that;
  // until then only the diff tools call this.
  function writePerturbOrbit(packed) {
    if (packed.byteLength > PT_RECS_BYTES)
      throw new Error(
        `writePerturbOrbit: ${packed.byteLength} B exceeds the ${PT_RECS_BYTES} B buffer`,
      );
    device.queue.writeBuffer(ptBuf, 0, packed);
  }

  return {
    device,
    format,
    writeGlobals,
    writeJitter,
    writeTile,
    writePost,
    drawAccum,
    drawMarchBand,
    createTileTarget,
    marchReadyFor,
    prewarmMarchFor,
    prewarmGeneralFor,
    getDiag: () => diag,
    writeOps,
    writeHybrid,
    writeMorph,
    writeScene,
    draw,
    drawTo,
    configureContext,
    renderToImage,
    createSplatCapture,
    overrideCaptureOffset,
    writePerturbOrbit,
    setEnvImage, // IMGTEX #631 — upload/clear the env ("env") / surface image
    hasImgTex, // IMGTEX #631 — texture-presence half of the codegen latch
    setStreamlines, // Field streamlines (lab) — core/streamlines.js
    tickStreamlines, // idle overlay frame (advect + repaint, no march)
    // Deterministic offline exports: bracket a captureFrame loop so the overlay
    // reseeds from `key` + pre-rolls once, then hands the live flow back. No-ops
    // when the overlay was never enabled (no controller ⇒ nothing to park).
    beginStreamOffline: (key) => stream?.beginOffline(key) ?? false,
    endStreamOffline: () => stream?.endOffline(),
    streamlinesInfo: () => stream?.info() ?? null, // pipeline states + errors
    streamlinesStats: async () => (stream ? stream.readStats() : null), // probe census
    MAX_OPS,
    MAX_OBJECTS,
  };
}

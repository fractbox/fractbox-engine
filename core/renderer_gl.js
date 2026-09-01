// WebGL2 renderer — a drop-in twin of renderer.js (same writeOps / writeGlobals
// / draw / renderToImage interface), so preview.js drives WebGPU or WebGL2 with
// no controller changes. Used as the middle fallback tier: WebGPU → WebGL2 →
// CPU/ASCII. The fragment shader is regenerated only on an op-STRUCTURE change;
// param values + camera ride uniforms (no relink on a slider drag).

import { byKey } from "./operators.js";
import { computeRecenter } from "./recenter.js";
import {
  VERT_GL,
  buildFragGL,
  buildSceneFragGL,
  sceneParamLayout,
  bulkLayout,
  MAX_PARAMS,
  MAX_OBJECTS,
} from "./shader_gl.js";
import { MAX_OPS_WEBGL2 } from "./limits.js";
import { packOpParams } from "./uniformPack.js";
import { normalizeSceneObject } from "./sceneobj.js";
import { leafById } from "./leaves.js";
import { deriveFrameParams } from "./frameparams.js";
import { classifyGlHealth, glErrorName } from "./gldiag.js";

const MAX_OPS = MAX_OPS_WEBGL2; // 64 — lower than the WebGPU tier (see limits.js)

// Cap a shader/program info log so the (user-pasted) diag dump stays compact.
function trimLog(s) {
  const t = String(s || "").trim();
  return t.length > 300 ? t.slice(0, 300) + "…" : t;
}

// opts.onTrouble(kind, detail) — the SAME diag hook the WebGPU tier uses
// (renderer.js note() → preview.js → app diag.ts onGpuTrouble). This tier fails
// DARK (#206: a 9-day silent black draw; the iOS-15 field dump showed frames
// "advancing" over a black canvas), so every compile/link/context/draw fault is
// reported here. opts.glFail ('compile'|'link'|'draw') is a test hook that forces
// the respective failure so the telemetry + auto-fallback can be exercised on a
// healthy GPU. Both are optional — omitted, this is byte-for-byte the old path.
export async function createRendererGL(canvas, opts = {}) {
  const note = (kind, detail) => {
    try {
      opts.onTrouble?.(kind, detail);
    } catch {
      /* reporting must never break rendering */
    }
  };
  // Raw signals for the pure classifier (gldiag.js). The renderer only COLLECTS;
  // preview.js reads glHealth() and drives the ladder fallback to ASCII.
  const health = {
    contextCreationError: false,
    contextLost: false,
    compileFailed: false,
    linkFailed: false,
    drawErrors: [],
  };
  const glFail = opts.glFail || null;

  // Context-loss telemetry — never fires on the boot path; only if the GPU
  // process is evicted mid-session (the "exit NOT clean (possible crash)" field
  // signature). webglcontextcreationerror must be listened for BEFORE getContext.
  canvas.addEventListener?.("webglcontextcreationerror", (e) => {
    health.contextCreationError = true;
    note("webglcontextcreationerror", { message: e?.statusMessage || "" });
  });
  canvas.addEventListener?.("webglcontextlost", () => {
    health.contextLost = true;
    note("webglcontextlost", {});
  });
  canvas.addEventListener?.("webglcontextrestored", () => {
    note("webglcontextrestored", {});
  });

  const gl = canvas.getContext("webgl2", {
    antialias: false,
    preserveDrawingBuffer: true, // so canvas.toBlob (PNG export) sees the frame
    alpha: false,
  });
  if (!gl) throw new Error("WebGL2 unavailable");

  gl.bindVertexArray(gl.createVertexArray()); // attribute-less draw needs a bound VAO

  // Shader compile — inner so a failure reaches the diag hook + health signals.
  // It still THROWS on failure: a vert failure at boot falls through preview.js's
  // WebGL2 catch (→ ASCII); a frag failure (built lazily on the first writeOps)
  // is caught by the pump, and the glHealth()==dead verdict then drives fallback.
  function compileShader(type, src) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    const forced = glFail === "compile" && type === gl.FRAGMENT_SHADER;
    const ok = !forced && gl.getShaderParameter(sh, gl.COMPILE_STATUS);
    if (!ok) {
      const stage = type === gl.VERTEX_SHADER ? "vertex" : "fragment";
      const log = forced
        ? "[glfail=compile forced]"
        : gl.getShaderInfoLog(sh) || "";
      health.compileFailed = true;
      note("gl-compile-fail", { stage, log: trimLog(log) });
      gl.deleteShader(sh);
      throw new Error("WebGL2 " + stage + " shader compile failed: " + log);
    }
    return sh;
  }

  const vert = compileShader(gl.VERTEX_SHADER, VERT_GL);
  let program = null;
  let opSig = null; // op-key signature the current program was built for
  let loc = {}; // uniform-location cache for the live program
  // ENVX (backgrounds P5) — the COLORING, not the formula, decides this codegen
  // bit, and coloring edits arrive without a write* call, so the relink trigger
  // lives in applyUniforms (the WebGPU tier's activeFeat latch equivalent).
  // `rebuildWith` is the current program shape's relink closure, stashed by
  // whichever write* path built it; `progEnvx` is what the live program was
  // built with; `envxWant` is what the last frame's look derived.
  let progEnvx = false;
  let envxWant = false;
  // NEON emissive glow — the second look-driven codegen bit, same latch shape
  // as ENVX above (upward relink in applyUniforms, downgrade rides the next
  // structural rebuild). Flat/hybrid only: the scene builder never emits it
  // (V1 flat-only, the iridescence S6 precedent), so neonWant derives false
  // on scenes and the trigger can't relink-loop against a neon-free program.
  let progNeon = false;
  let neonWant = false;
  // AURORA (ENVX P6) — the third look-driven codegen bit, same latch shape as
  // ENVX above (upward relink in applyUniforms, downgrade rides the next
  // structural rebuild). A background layer, so ALL builders emit it (flat/
  // hybrid/scene — the envx contract, unlike neon's flat-only guard).
  let progAurora = false;
  let auroraWant = false;
  // CINE GRADE — the fourth look-driven codegen bit, same upward-relink latch
  // shape as ENVX/NEON above. Unlike neon it is emitted by ALL builders
  // (flat/hybrid/scene — the grade lives in the shared mainGL tail), so no
  // scene guard is needed: a scene look with a grade relinks the scene program
  // with the cineGrade splice. Downgrade (look → None) needs no relink at all:
  // neutral uniforms (strength 0) make cineGrade an exact identity, and the
  // next structural rebuild drops the splice for free.
  let progGrade = false;
  let gradeWant = false;
  // THIN FILM interference material — the fourth look-driven codegen bit, same
  // latch shape as NEON above (upward relink in applyUniforms, downgrade rides
  // the next structural rebuild). Flat/hybrid only: the scene builder never
  // emits it (V1 flat-only, the neon/S6 precedent), so filmWant derives false
  // on scenes and the trigger can't relink-loop against a film-free program.
  let progFilm = false;
  let filmWant = false;
  // TINY PLANET — the LAST look-driven codegen bit, and the ONE that must
  // relink in BOTH directions. ENVX/NEON degrade gracefully (their on-variant
  // with zeroed uniforms renders the off picture), so their trigger is upward
  // only. The planet variant does NOT: its ray-gen has no perspective arm, so
  // running it with uPlanetK = 0 collapses every pixel onto +fwd and the frame
  // becomes one flat colour. Turning the projection OFF therefore has to
  // relink, not just stop writing the uniform.
  let progPlanet = false;
  let planetWant = false;
  // CLIP cross-section — bidirectional like TINY PLANET, for a subtler reason:
  // the on-variant with a ZEROED plane ((0,0,0)·p − 0 = 0) turns the march
  // term into max(de, 0) — visually near-identical — but the cut-face test
  // (planeD >= de) then misclassifies every de <= 0 hit as a cut, so a stale
  // clip program with neutral uniforms can flat-shade random pixels. Turning
  // the plane OFF therefore relinks, not just stops writing the uniform.
  // Emitted by ALL builders (flat/hybrid/scene — march geometry, the planet
  // contract).
  let progClip = false;
  let clipWant = false;
  // CLIP JAGGED — the noised-cut sub-variant, bidirectional with clip for
  // symmetry with the WebGPU keying (and perf on the way DOWN: a jag program
  // with amp 0 renders the flat cut correctly but pays the per-step noise).
  let progClipJag = false;
  let clipJagWant = false;
  // 360° EQUIRECT — the planet's twin in every respect that matters here,
  // including the BIDIRECTIONAL relink: the equirect variant has no
  // perspective arm either, so it must relink on the way OFF as well as on.
  // (Unlike the planet it declares no uniform at all — the GL tier never
  // tiles, so the map's scale is a shader constant.)
  let progEquirect = false;
  let equirectWant = false;
  let rebuildWith = null;

  let G = null; // last writeGlobals payload
  const params = new Float32Array(MAX_PARAMS);
  let opCount = 0;

  // CSG scene state — per-object descriptors (uniform arrays) + the scene flag.
  let isScene = false;
  let sceneCount = 0;

  // Hybrid iteration state (IDEAS ①, docs/design/HYBRID_ITERATION.md §3.5).
  let isHybrid = false;
  // N-slot hybrid uniform state (HYBRID_NSLOT_SPEC.md §2.4) — per-slot schedule
  // counts + addC gates, uploaded to the uHyb[]/uAddGate[] arrays. Sized by the
  // live program's slot count (rebuilt on a structural slot change).
  let hybCounts = [1, 1];
  let hybAddGate = [0, 0];
  const objOrigin = new Float32Array(MAX_OBJECTS * 3);
  const objUscale = new Float32Array(MAX_OBJECTS).fill(1);
  const objQuat = new Float32Array(MAX_OBJECTS * 4);
  const objBlendK = new Float32Array(MAX_OBJECTS);
  const objJc = new Float32Array(MAX_OBJECTS * 3);
  const objPrimP = new Float32Array(MAX_OBJECTS * 4); // leaf shapeParams (vec4/object)
  const objPrimP2 = new Float32Array(MAX_OBJECTS * 4); // #627 overflow lane (sp4..sp7)
  const objIters = new Int32Array(MAX_OBJECTS).fill(1);
  const objAddGate = new Int32Array(MAX_OBJECTS);
  const objJulia = new Float32Array(MAX_OBJECTS);
  const objDeOption = new Float32Array(MAX_OBJECTS).fill(2);
  const objColor = new Float32Array(MAX_OBJECTS * 3).fill(0.5); // per-object albedo (sRGB) — §3.8
  // COLORING P0 — reused scratch for the 8-stop palette (packed OKLab [L,a,b,p]
  // × 8); refilled per frame, never reallocated. Declared here (with the other
  // upload sources) so the bulkSource map below can reference it.
  const palStopScratch = new Float32Array(32);

  // ── std140 bulk UBO (GLES-minimum uniform budget) ──────────────────────────
  // The fat arrays (uP, palette stops, per-object scene arrays, hybrid schedule)
  // ride a std140 uniform BUFFER, not the default uniform block: on a
  // GLES-3.0-minimum device (MAX_FRAGMENT_UNIFORM_VECTORS == 224) the default
  // block overflowed and the whole tier failed to LINK (field report, iOS 15.8).
  // bulkLayout() (shader_gl.js) is the single source of the member set + order;
  // every member is an array → 16-byte std140 stride, so offsets are a running
  // sum of count·16. One buffer, rebuilt per link, whole-buffer bufferSubData
  // per frame (fewer GL calls than the ~13 uniform*fv it replaces).
  const BULK_BINDING = 0;
  let ubo = null;
  let bulkMembers = null; // ordered std140 descriptor for the live program
  let bulkOffset = {}; // member name → element index (byteOffset >> 2)
  let bulkBuf = null; // backing ArrayBuffer
  let bulkF32 = null;
  let bulkI32 = null;
  // member name → tight (non-strided) source array; uHyb/uAddGate are handled
  // inline (they fold in the frame's Julia flag + come from plain JS arrays).
  const bulkSource = {
    uP: params,
    uPalStops: palStopScratch,
    uObjOrigin: objOrigin,
    uObjUscale: objUscale,
    uObjQuat: objQuat,
    uObjBlendK: objBlendK,
    uObjJc: objJc,
    uObjPrimP: objPrimP,
    uObjPrimP2: objPrimP2, // #627 — present in the layout only for fat-leaf scenes
    uObjIters: objIters,
    uObjAddGate: objAddGate,
    uObjJulia: objJulia,
    uObjDeOption: objDeOption,
    uObjColor: objColor,
  };

  function setupBulk(members) {
    bulkMembers = members;
    bulkOffset = {};
    let byteOff = 0;
    for (const m of members) {
      bulkOffset[m.name] = byteOff >> 2; // element (float/int) index
      byteOff += m.count * 16; // std140: every member is an array, 16-byte stride
    }
    bulkBuf = new ArrayBuffer(byteOff);
    bulkF32 = new Float32Array(bulkBuf);
    bulkI32 = new Int32Array(bulkBuf);
    if (!ubo) ubo = gl.createBuffer();
    gl.bindBuffer(gl.UNIFORM_BUFFER, ubo);
    gl.bufferData(gl.UNIFORM_BUFFER, byteOff, gl.DYNAMIC_DRAW);
    // layout(binding=) is ES 3.1+, so bind the block to a fixed point here.
    const idx = gl.getUniformBlockIndex(program, "Bulk");
    if (idx !== gl.INVALID_INDEX) {
      gl.uniformBlockBinding(program, idx, BULK_BINDING);
      gl.bindBufferBase(gl.UNIFORM_BUFFER, BULK_BINDING, ubo);
    }
  }

  // Re-stride the tight source arrays into the std140 buffer (per element: C
  // active components into a 16-byte/vec4 slot, rest padding), then upload once.
  function packBulk() {
    if (!bulkMembers || !bulkBuf) return;
    const jHyb = G && G.julia ? 1 : 0; // fold the formula Julia flag (hybrid)
    for (const m of bulkMembers) {
      const base = bulkOffset[m.name];
      if (m.name === "uHyb") {
        for (let k = 0; k < m.count; k++)
          bulkI32[base + k * 4] = hybCounts[k] | 0;
        continue;
      }
      if (m.name === "uAddGate") {
        for (let k = 0; k < m.count; k++)
          bulkI32[base + k * 4] = hybAddGate[k] || jHyb ? 1 : 0;
        continue;
      }
      const src = bulkSource[m.name];
      if (!src) continue;
      const view = m.baseType === "int" ? bulkI32 : bulkF32;
      const C = m.comps;
      for (let k = 0; k < m.count; k++)
        for (let c = 0; c < C; c++) view[base + k * 4 + c] = src[k * C + c];
    }
    gl.bindBuffer(gl.UNIFORM_BUFFER, ubo);
    gl.bufferSubData(gl.UNIFORM_BUFFER, 0, bulkF32);
  }

  function linkProgram(fragSrc, bulkMemberList) {
    const frag = compileShader(gl.FRAGMENT_SHADER, fragSrc);
    const p = gl.createProgram();
    gl.attachShader(p, vert);
    gl.attachShader(p, frag);
    gl.linkProgram(p);
    const forced = glFail === "link";
    const linked = !forced && gl.getProgramParameter(p, gl.LINK_STATUS);
    if (!linked) {
      const log = forced
        ? "[glfail=link forced]"
        : gl.getProgramInfoLog(p) || "";
      health.linkFailed = true;
      note("gl-link-fail", { log: trimLog(log) });
      gl.deleteProgram(p);
      gl.deleteShader(frag);
      throw new Error("WebGL2 link failed: " + log);
    }
    if (program) gl.deleteProgram(program);
    program = p;
    loc = {};
    gl.deleteShader(frag);
    setupBulk(bulkMemberList);
  }

  const U = (name) => (loc[name] ??= gl.getUniformLocation(program, name));

  // ── renderer.js-compatible surface ─────────────────────────────────────────
  function writeOps(ops) {
    isScene = false;
    isHybrid = false;
    const n = Math.min(ops.length, MAX_OPS);
    const clipped = ops.slice(0, n);
    const sig = clipped.map((o) => o.key).join("|");
    rebuildWith = (ex, ne, au, gr, tf, pl, cl, cj, eq) => {
      linkProgram(
        buildFragGL(clipped, undefined, undefined, {
          envx: ex,
          neon: ne,
          aurora: au,
          grade: gr,
          thinFilm: tf,
          planet: !!pl,
          clip: !!cl,
          clipJag: !!cj,
          equirect: !!eq,
        }),
        bulkLayout({}),
      );
      progEnvx = ex;
      progNeon = !!ne;
      progAurora = !!au;
      progGrade = !!gr;
      progFilm = !!tf;
      progPlanet = !!pl;
      progClip = !!cl;
      progClipJag = !!cj;
      progEquirect = !!eq;
    };
    // Structural changes rebuild with the CURRENT envx/neon/aurora/grade/
    // thinFilm wants (so a pending downgrade rides along free); a pure
    // upgrade is applyUniforms' job.
    if (sig !== opSig || !program) {
      rebuildWith(
        envxWant,
        neonWant,
        auroraWant,
        gradeWant,
        filmWant,
        planetWant,
        clipWant,
        clipJagWant,
        equirectWant,
      );
      opSig = sig;
    }
    // packOpParams returns a full MAX_PARAMS array, so .set() overwrites every
    // slot (the old params.fill(0) is subsumed). Keep params `const` — the scene
    // path mutates it in place and the uP upload reads it by identity.
    params.set(packOpParams(ops.slice(0, n)));
    opCount = n;
    return n;
  }

  // N-slot hybrid iteration (HYBRID_NSLOT_SPEC.md §2.4) — the GLSL codegen split.
  // `slotOps` = per-slot op lists ([A, B, C…]); `counts`/`addC` = the schedule +
  // per-slot addC. The structural signature covers EVERY slot's op-key sequence
  // (a change to any rebuilds the program); the schedule counts + per-slot addC
  // ride the uHyb[]/uAddGate[] uniform arrays, not codegen — editing a ratio is a
  // uniform upload, matching the plain single-object path's rule.
  function writeHybrid(slotOps, counts, addC) {
    isScene = false;
    isHybrid = true;
    const clipped = slotOps.map((ops) => ops.slice(0, MAX_OPS));
    const sig =
      "hyb:" + clipped.map((ops) => ops.map((o) => o.key).join(",")).join("|");
    rebuildWith = (ex, ne, au, gr, tf, pl, cl, cj, eq) => {
      linkProgram(
        buildFragGL(
          clipped[0],
          clipped.slice(1).map((ops) => ({ ops })),
          undefined,
          {
            envx: ex,
            neon: ne,
            aurora: au,
            grade: gr,
            thinFilm: tf,
            planet: !!pl,
            clip: !!cl,
            clipJag: !!cj,
            equirect: !!eq,
          },
        ),
        bulkLayout({ hybrid: clipped.length }),
      );
      progEnvx = ex;
      progNeon = !!ne;
      progAurora = !!au;
      progGrade = !!gr;
      progFilm = !!tf;
      progPlanet = !!pl;
      progClip = !!cl;
      progClipJag = !!cj;
      progEquirect = !!eq;
    };
    if (sig !== opSig || !program) {
      rebuildWith(
        envxWant,
        neonWant,
        auroraWant,
        gradeWant,
        filmWant,
        planetWant,
        clipWant,
        clipJagWant,
        equirectWant,
      );
      opSig = sig;
    }
    // Every slot packed contiguously into the same uP[] (each continues where the
    // previous ended); .set() overwrites all MAX_PARAMS slots.
    params.set(packOpParams(...clipped));
    opCount = clipped.reduce((n, ops) => n + ops.length, 0);
    hybCounts = counts.map((c) => c | 0);
    hybAddGate = addC.map((c) => (c ? 1 : 0));
    return opCount;
  }

  // CSG Phase 1b/C — multi-object scene upload (mirrors renderer.js writeScene).
  // The fragment program is codegen'd per scene STRUCTURE (object count, each
  // object's objType/combine/op-key sequence) and rebuilt when that signature
  // changes; per-object SCALARS ride uniform arrays (no recompile on a value
  // tweak). Param values for IFF objects share one uP[] array, concatenated at
  // the bases from sceneParamLayout (kept in lockstep with the codegen).
  function writeScene(objects) {
    isHybrid = false;
    if (!Array.isArray(objects) || objects.length === 0)
      throw new Error("writeScene: empty scene");
    if (objects.length > MAX_OBJECTS)
      throw new Error(
        `writeScene: ${objects.length} objects > cap ${MAX_OBJECTS}`,
      );
    isScene = true;
    sceneCount = objects.length;

    // Canonical per-object metadata (fallback chains, active-op slice, quat) —
    // shared with renderer.js/cpu.js via sceneobj.js normalizeSceneObject. Its
    // active-op slice is EXACTLY shader_gl.js activeSceneOps, so the codegen'd
    // uP[] indices (sceneParamLayout) and the values packed below stay in
    // lockstep; guarded by core/scenemute.test.mjs.
    const norm = objects.map(normalizeSceneObject);

    // Structural signature → rebuild the program only when the STRUCTURE changes.
    const sig =
      "scene:" +
      norm
        .map((o) => {
          // Active ops only (per-op scene mute) — muting/unmuting changes the
          // codegen'd program, so it must change the signature too.
          const keys = o.ops.map((x) => x.key).join(",");
          // shapeId + iterShape are baked into the objDE codegen (which leaf
          // fn is called, and where) → part of the structure (D0 §2.2.1).
          return `${o.shapeId}${o.iterShape ? "i" : ""}/${o.combine}/${keys}`;
        })
        .join("|");
    rebuildWith = (ex, _ne, au, gr, _tf, pl, cl, cj, eq) => {
      linkProgram(
        buildSceneFragGL(objects, {
          envx: ex,
          aurora: au,
          grade: gr,
          planet: !!pl,
          clip: !!cl,
          clipJag: !!cj,
          equirect: !!eq,
        }),
        // #627 — the Bulk block gains uObjPrimP2 on the SAME predicate the
        // scene codegen uses (a >4-param leaf in the object set); shapeId is
        // in the structural signature, so fat/thin transitions relink.
        // Derived from the NORMALIZED objects — buildSceneFragGL normalizes
        // before computing its fatLeaf, and a legacy-field object (shapeId
        // living under objType) must not make the two disagree (the class of
        // the 2026-08-31 uObjPrimP2-undeclared compile fail).
        bulkLayout({
          scene: true,
          leafAux: norm.some(
            (o) => ((leafById(o.shapeId) || {}).params?.length ?? 0) > 4,
          ),
        }),
      );
      progEnvx = ex;
      progFilm = false; // …nor THIN FILM (same V1 flat-only rule)
      progAurora = !!au; // …but it DOES emit AURORA (a background layer)
      progGrade = !!gr; // …and the grade (shared mainGL tail)
      progPlanet = !!pl; // …and TINY PLANET (ray-gen, before flat/CSG matters)
      progClip = !!cl; // …and CLIP (march geometry, the planet contract)
      progClipJag = !!cj; // …and its JAGGED sub-variant
      progEquirect = !!eq; // …and 360° EQUIRECT (same reasoning)
    };
    if (sig !== opSig || !program) {
      rebuildWith(
        envxWant,
        false,
        auroraWant,
        gradeWant,
        false,
        planetWant,
        clipWant,
        clipJagWant,
        equirectWant,
      );
      opSig = sig;
    }

    // Param values into the shared uP[] (IFS objects only) + per-object scalars.
    params.fill(0);
    const layout = sceneParamLayout(objects);
    norm.forEach((o, k) => {
      if (o.ops.length) {
        let slot = layout[k].slotBase;
        // Active ops only (o.ops IS the active slice) — must match
        // sceneParamLayout/buildSceneFragGL's slice or the uP[] values shift
        // against the codegen'd indices.
        for (const op of o.ops) {
          const def = byKey(op.key);
          // Unknown keys throw, matching the WebGPU tier (renderer.js writeScene).
          if (!def) throw new Error(`writeScene: unknown op "${op.key}"`);
          for (let p = 0; p < def.params.length; p++) {
            if (slot >= MAX_PARAMS)
              throw new Error("writeScene: scene params > cap " + MAX_PARAMS);
            params[slot++] = op.values?.[p] ?? 0;
          }
        }
      }
      objOrigin[k * 3 + 0] = o.origin[0];
      objOrigin[k * 3 + 1] = o.origin[1];
      objOrigin[k * 3 + 2] = o.origin[2];
      objUscale[k] = o.uscale;
      objQuat[k * 4 + 0] = o.quat[0];
      objQuat[k * 4 + 1] = o.quat[1];
      objQuat[k * 4 + 2] = o.quat[2];
      objQuat[k * 4 + 3] = o.quat[3];
      objBlendK[k] = o.blendK;
      objColor[k * 3 + 0] = o.color[0];
      objColor[k * 3 + 1] = o.color[1];
      objColor[k * 3 + 2] = o.color[2];
      objJc[k * 3 + 0] = o.juliaC[0];
      objJc[k * 3 + 1] = o.juliaC[1];
      objJc[k * 3 + 2] = o.juliaC[2];
      objPrimP[k * 4 + 0] = o.shapeParams[0];
      objPrimP[k * 4 + 1] = o.shapeParams[1];
      objPrimP[k * 4 + 2] = o.shapeParams[2];
      objPrimP[k * 4 + 3] = o.shapeParams[3];
      // #627 — overflow lane, same index k; undeclared slots stay 0.
      objPrimP2[k * 4 + 0] = o.shapeParams[4] ?? 0;
      objPrimP2[k * 4 + 1] = o.shapeParams[5] ?? 0;
      objPrimP2[k * 4 + 2] = o.shapeParams[6] ?? 0;
      objPrimP2[k * 4 + 3] = o.shapeParams[7] ?? 0;
      objIters[k] = o.iters;
      // Pure leaves normalize to addC false, so the gate stays 0 for them.
      objAddGate[k] = o.addC || o.julia ? 1 : 0;
      objJulia[k] = o.julia ? 1 : 0;
      objDeOption[k] = o.deOption;
    });
    return sceneCount;
  }

  function writeGlobals(payload) {
    G = payload;
  }

  function applyUniforms(res) {
    gl.useProgram(program);
    const b = G.cam.basis();
    // Shared pure derivations (color/palette defaults, light rig, sky/fog
    // expansion) — ONE source with the WebGPU tier (frameparams.js). Derive
    // once, pack twice: the two tiers can't drift on defaults or the derived
    // fill/back dirs / penumbra k any more.
    const d = deriveFrameParams(G);
    // Deep zoom §3.1 recenter — shares renderer.js's exact computation.
    const { O, roRel } = computeRecenter(b.eye, G.cam.target, isScene);

    // ENVX (backgrounds P5) — relink when the look turns the extension ON (the
    // WebGPU tier's activeFeat.envx equivalent; see the state block at the
    // top). UPWARD only, deliberately: an envx program with all-zero uniforms
    // renders the legacy environment identically, so relinking DOWN would be a
    // full GL compile for nothing (WebGPU swaps cached pipelines; GL has no
    // program cache). The downgrade happens free on the next structural
    // rebuild (write* passes the then-current envxWant).
    // useProgram again after: linkProgram swapped the program object.
    envxWant = !!d.envx;
    // NEON — flat/hybrid only (the scene builder never emits it; the !isScene
    // guard is what keeps the upward trigger from relink-looping on scenes).
    neonWant = !isScene && !!d.neon;
    // AURORA (ENVX P6) — all shader shapes carry it (background layer).
    auroraWant = !!d.aurora;
    // CINE GRADE — all builders emit it (shared mainGL tail), scenes included.
    gradeWant = !!d.gradeOn;
    // THIN FILM — flat/hybrid only, exactly the NEON rule above.
    filmWant = !isScene && !!d.thinFilm;
    // TINY PLANET — BIDIRECTIONAL (see the state block): `!==`, not the
    // "want && !have" the other four use. Every one of those degrades
    // gracefully when its uniforms go neutral, so a downgrade can wait for the
    // next structural rebuild; the planet variant has no perspective arm, so
    // leaving it linked with uPlanetK = 0 collapses every ray onto +fwd and
    // the frame becomes one flat colour.
    planetWant = (G.planetK ?? 0) > 0;
    // CLIP — BIDIRECTIONAL like planet (see the state block: a stale clip
    // program with neutral uniforms misclassifies de <= 0 hits as cut faces).
    // All shader shapes carry it (march geometry). The JAGGED sub-variant
    // rides the same posture (see the state block).
    clipWant = !!d.clip;
    clipJagWant = !!d.clipJag;
    // 360° EQUIRECT — BIDIRECTIONAL for the planet's exact reason: the
    // equirect variant has no perspective arm, so a downgrade cannot wait for
    // the next structural rebuild (there is no "neutral uniforms" state — the
    // map is baked into the program text).
    equirectWant = (G.equirectS ?? 0) > 0;
    if (
      ((envxWant && !progEnvx) ||
        (neonWant && !progNeon) ||
        (auroraWant && !progAurora) ||
        (gradeWant && !progGrade) ||
        (filmWant && !progFilm) ||
        planetWant !== progPlanet ||
        clipWant !== progClip ||
        clipJagWant !== progClipJag ||
        equirectWant !== progEquirect) &&
      rebuildWith
    ) {
      rebuildWith(
        envxWant,
        neonWant,
        auroraWant,
        gradeWant,
        filmWant,
        planetWant,
        clipWant,
        clipJagWant,
        equirectWant,
      );
      gl.useProgram(program);
    }

    gl.uniform2f(U("uRes"), res[0], res[1]);
    gl.uniform1f(U("uFov"), G.cam.fov);
    gl.uniform3f(U("uCamPos"), roRel[0], roRel[1], roRel[2]);
    gl.uniform3f(U("uOffset"), O[0], O[1], O[2]);
    gl.uniform3f(U("uCamFwd"), b.fwd[0], b.fwd[1], b.fwd[2]);
    gl.uniform3f(U("uCamRight"), b.right[0], b.right[1], b.right[2]);
    gl.uniform3f(U("uCamUp"), b.up[0], b.up[1], b.up[2]);
    gl.uniform1f(U("uOrthoH"), G.orthoH ?? 0); // #441; 0 = perspective
    // TINY PLANET — declared ONLY by the planet variant, so guard the lookup
    // (the progEnvx/progNeon pattern below: no null-location churn off-path).
    if (progPlanet) gl.uniform1f(U("uPlanetK"), G.planetK ?? 0);
    gl.uniform1i(U("uIters"), G.iters | 0);
    // Hybrid schedule (§3.5) rides the uHyb[]/uAddGate[] arrays in the Bulk UBO
    // (packBulk, below) — a ratio/addC tweak is a buffer upload, not a recompile.
    // The flat path's single addC gate stays a default-block scalar. The formula
    // Julia flag is folded into the hybrid gates inside packBulk (matching the
    // WGSL `… || (G.jc.w > 0.5)` and CPU `… || julia` tiers).
    if (!isHybrid) gl.uniform1i(U("uAddGate"), d.addGate);
    gl.uniform1i(U("uMaxSteps"), G.maxSteps | 0);
    gl.uniform1i(U("uColorMode"), d.colorMode);
    gl.uniform1f(U("uBailout"), G.bailout);
    gl.uniform1f(U("uEps"), G.eps);
    gl.uniform1f(U("uDeScale"), d.deScale);
    gl.uniform1f(U("uNear"), d.tNear); // deep zoom §5
    gl.uniform1f(U("uFar"), d.tFar); // deep zoom §5
    gl.uniform1f(U("uDeOption"), d.deOption);
    gl.uniform3f(U("uColA"), d.colA[0], d.colA[1], d.colA[2]);
    gl.uniform3f(U("uColB"), d.colB[0], d.colB[1], d.colB[2]);
    gl.uniform3f(U("uBg"), d.bg[0], d.bg[1], d.bg[2]);
    gl.uniform3f(U("uJc"), d.juliaC[0], d.juliaC[1], d.juliaC[2]);
    gl.uniform1f(U("uJulia"), d.julia);
    gl.uniform3f(U("uPalA"), d.palA[0], d.palA[1], d.palA[2]);
    gl.uniform3f(U("uPalB"), d.palB[0], d.palB[1], d.palB[2]);
    gl.uniform3f(U("uPalC"), d.palC[0], d.palC[1], d.palC[2]);
    gl.uniform3f(U("uPalD"), d.palD[0], d.palD[1], d.palD[2]);
    gl.uniform1f(U("uPalOn"), d.palOn);
    // N-stop palette (OKLab, flat path). palStopScratch stays zero-padded past
    // the active count; uPalCount 0 → the shader takes the legacy path and
    // never reads the stops. On the scene program these locations are inactive
    // (null) — uniform* on null is a spec no-op, so this is safe there too.
    palStopScratch.fill(0);
    if (d.palStops) {
      for (let i = 0; i < d.palStops.length && i < 8; i++) {
        palStopScratch[i * 4] = d.palStops[i][0];
        palStopScratch[i * 4 + 1] = d.palStops[i][1];
        palStopScratch[i * 4 + 2] = d.palStops[i][2];
        palStopScratch[i * 4 + 3] = d.palStops[i][3];
      }
    }
    // uPalStops rides the Bulk UBO (packBulk); palStopScratch is its source.
    gl.uniform1f(U("uPalCount"), d.palStopCount || 0);
    gl.uniform1f(U("uPalCyclic"), d.palCyclic || 0);
    gl.uniform1f(U("uStripeFreq"), d.stripeFreq); // COLORING S2 Silk (inactive on the scene program)
    // COLORING P2 — auto-levels signal range (identity 0,1 when off/cyclic).
    gl.uniform1f(U("uSigLo"), d.sigLo ?? 0);
    gl.uniform1f(U("uSigSpan"), d.sigSpan ?? 1);
    // COLORING P3 — iridescence (Glow trap-XYZ modulator; 0 = off).
    gl.uniform1f(U("uIridescence"), d.iridescence ?? 0);
    // COLORING P3 — palette phase (0 = identity; cyclic rotation for timeline).
    gl.uniform1f(U("uPalettePhase"), d.palettePhase ?? 0);
    gl.uniform3f(U("uLightDir"), d.lightDir[0], d.lightDir[1], d.lightDir[2]);
    gl.uniform1f(U("uAmbient"), d.ambient);
    gl.uniform1f(U("uRim"), d.rim);
    gl.uniform1f(U("uGloss"), d.gloss);
    gl.uniform1f(U("uIntensity"), d.intensity);
    // P1 light rig — same derived values as renderer.js gF[100..123] BY
    // CONSTRUCTION (frameparams.js owns the defaults, penumbra k, and the
    // derived fill/back directions).
    gl.uniform3f(U("uKeyC"), d.keyColor[0], d.keyColor[1], d.keyColor[2]);
    gl.uniform1f(U("uMetallic"), d.metallic);
    gl.uniform1f(U("uShadowK"), d.shadowK);
    gl.uniform1f(U("uShadowOn"), d.shadowOn);
    gl.uniform1f(U("uAoStr"), d.ao);
    gl.uniform3f(U("uFillDir"), d.fillDir[0], d.fillDir[1], d.fillDir[2]);
    gl.uniform1f(U("uFill"), d.fill);
    gl.uniform3f(U("uFillC"), d.fillColor[0], d.fillColor[1], d.fillColor[2]);
    gl.uniform3f(U("uBackDir"), d.backDir[0], d.backDir[1], d.backDir[2]);
    gl.uniform1f(U("uBack"), d.back);
    gl.uniform3f(U("uBackC"), d.backColor[0], d.backColor[1], d.backColor[2]);
    // P3 env/fog macros — same derived values as renderer.js gF[128..133]
    // (bloom is WebGPU-only; d's bloom fields are ignored here).
    gl.uniform1f(U("uSky"), d.sky);
    gl.uniform1f(U("uSunGlow"), d.sunGlow); // #160: hide the sky's sun-glow disc, keep IBL
    gl.uniform1f(U("uGround"), d.ground);
    gl.uniform1f(U("uIbl"), d.ibl);
    gl.uniform1f(U("uFogAmt"), d.fog);
    gl.uniform1f(U("uInScatter"), d.inScatter);
    // ENVX (backgrounds P5) — declared only by envx programs (gated on
    // progEnvx so no null-location lookups pile up on the common path).
    if (progEnvx) {
      gl.uniform1f(U("uStars"), d.stars);
      gl.uniform1f(U("uStarDensity"), d.starDensity);
      gl.uniform1f(U("uStarSeed"), d.starSeed);
      gl.uniform1f(U("uBand"), d.band);
      gl.uniform3f(U("uBandDir"), d.bandDir[0], d.bandDir[1], d.bandDir[2]);
      gl.uniform3f(U("uZenC"), d.zenith[0], d.zenith[1], d.zenith[2]);
      gl.uniform1f(U("uZenOn"), d.zenithOn);
    }
    // NEON — declared only by neon programs (gated like the ENVX block above).
    if (progNeon) gl.uniform1f(U("uNeon"), d.neonGain);
    // AURORA (ENVX P6) — declared only by aurora programs (same gating).
    if (progAurora) {
      gl.uniform1f(U("uAurora"), d.auroraAmt);
      gl.uniform1f(U("uNebula"), d.nebulaAmt);
      gl.uniform1f(U("uAurDrift"), d.auroraDrift);
      gl.uniform3f(
        U("uAurA"),
        d.auroraColA[0],
        d.auroraColA[1],
        d.auroraColA[2],
      );
      gl.uniform3f(
        U("uAurB"),
        d.auroraColB[0],
        d.auroraColB[1],
        d.auroraColB[2],
      );
    }
    // THIN FILM — declared only by thinFilm programs (same gating).
    if (progFilm) gl.uniform1f(U("uFilm"), d.filmAmt);
    // CLIP — declared only by clip programs (same gating). One vec4 (unit
    // normal + offset along it, frameparams-derived) + the reserved gain.
    if (progClip) {
      gl.uniform4f(U("uClip"), d.clipN[0], d.clipN[1], d.clipN[2], d.clipW);
      gl.uniform1f(U("uClipS"), d.clipShade);
      // CLIP JAGGED — declared only by jag programs (same gating): amp,
      // freq, and the Lipschitz divisor, frameparams-derived.
      if (progClipJag)
        gl.uniform3f(U("uClipJ"), d.clipJagAmp, d.clipJagFreq, d.clipJagInv);
    }
    // CINE GRADE — declared only by grade programs (same gating). A look
    // turned off under a still-graded program uploads the neutral words
    // (strength 0 → cineGrade is an exact identity) until the next structural
    // rebuild drops the splice.
    if (progGrade) {
      gl.uniform4f(U("uGradeA"), d.gradeA[0], d.gradeA[1], d.gradeA[2], d.gradeA[3]); // prettier-ignore
      gl.uniform4f(U("uGradeB"), d.gradeB[0], d.gradeB[1], d.gradeB[2], d.gradeB[3]); // prettier-ignore
      gl.uniform4f(U("uGradeC"), d.gradeC[0], d.gradeC[1], d.gradeC[2], d.gradeC[3]); // prettier-ignore
      gl.uniform4f(U("uGradeD"), d.gradeD[0], d.gradeD[1], d.gradeD[2], d.gradeD[3]); // prettier-ignore
    }
    gl.uniform1f(U("uExposure"), d.exposure); // whole-frame EV (mirrors post.y)
    // uP, uPalStops, the hybrid schedule, and the per-object scene arrays all
    // ride the std140 Bulk UBO — re-strided + uploaded in one call here.
    packBulk();
  }

  // Sample gl.getError() only on the FIRST few draws — a getError() every frame
  // forces a pipeline flush (perf). A nonzero code here means the draw is a
  // no-op: the black-canvas failure class. Recorded + fed to glHealth().
  let drawsSampled = 0;
  const DRAWS_TO_SAMPLE = 3;
  function sampleDrawError() {
    if (drawsSampled >= DRAWS_TO_SAMPLE) return;
    drawsSampled++;
    let code = gl.getError();
    // Test hook: force one INVALID_OPERATION on the first draw so the fallback
    // path can be exercised on a healthy GPU (only if the draw was itself clean).
    if (glFail === "draw" && drawsSampled === 1 && !code) code = 0x0502;
    if (code) {
      health.drawErrors.push(code);
      note("gl-error", { code: glErrorName(code), phase: "draw" });
    }
  }

  function draw() {
    if (!program || !G) return;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvas.width, canvas.height);
    applyUniforms([canvas.width, canvas.height]);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    sampleDrawError();
  }

  // Offscreen render → ImageData (RGBA, top-left origin) for preset thumbnails.
  let fbo = null,
    ftex = null,
    fw = 0,
    fh = 0;
  function ensureFbo(W, H) {
    if (fbo && fw === W && fh === H) return;
    if (ftex) gl.deleteTexture(ftex);
    if (!fbo) fbo = gl.createFramebuffer();
    ftex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, ftex);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA,
      W,
      H,
      0,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      null,
    );
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D,
      ftex,
      0,
    );
    fw = W;
    fh = H;
  }
  async function renderToImage(W, H) {
    if (!program || !G) return new ImageData(W, H);
    ensureFbo(W, H);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.viewport(0, 0, W, H);
    applyUniforms([W, H]);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    const px = new Uint8Array(W * H * 4);
    gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, px);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    // GL origin is bottom-left; ImageData is top-left → flip rows.
    const out = new Uint8ClampedArray(W * H * 4);
    const row = W * 4;
    for (let y = 0; y < H; y++)
      out.set(px.subarray((H - 1 - y) * row, (H - y) * row), y * row);
    return new ImageData(out, W, H);
  }

  // Shim the bits of the WebGPU `device.queue` that preview.js awaits.
  const device = {
    queue: { onSubmittedWorkDone: () => (gl.finish(), Promise.resolve()) },
  };

  return {
    device,
    writeGlobals,
    writeOps,
    writeHybrid,
    writeScene,
    draw,
    renderToImage,
    MAX_OPS,
    MAX_OBJECTS,
    backend: "webgl2",
    // Pure verdict over the collected fault signals — preview.js polls this on
    // the first few frames and falls to the ASCII view when the tier is dead.
    glHealth: () => classifyGlHealth(health),
  };
}

// ─────────────────────────────────────────────────────────────────────────
// Code generators — turn the operator IR into runnable code.
// ─────────────────────────────────────────────────────────────────────────
// Two backends, one source of truth (operators.js):
//
//   buildWGSL()    assembles the IN-BROWSER interpreter. Every operator's
//                  `wgsl` body becomes a `case` in a switch the march loop
//                  runs per op per iteration. The switch is generated ONCE at
//                  startup; thereafter editing param VALUES or REORDERING ops
//                  is pure data (rewrite the op buffer) — no shader rebuild.
//                  Only adding a brand-new operator TYPE regenerates this.
//
//   exportGLSL()   emits a native iterateJIT_ body for the desktop app — the
//                  "design in browser → render on desktop" handoff. The op-list
//                  is the interchange format; this proves it round-trips.
// ─────────────────────────────────────────────────────────────────────────

import { OPERATORS, byKey, effectiveDeOption, activeOps } from "./operators.js";

// ── WGSL interpreter ───────────────────────────────────────────────────────
export function buildWGSL() {
  const cases = OPERATORS.map(
    (op) =>
      `      case ${op.id}u: {${op.wgsl}
      }`,
  ).join("\n");

  return `
struct Globals {
  res     : vec4f,   // x,y = resolution px ; z = fov(rad) ; w = time
  camPos  : vec4f,
  camFwd  : vec4f,
  camRight: vec4f,
  camUp   : vec4f,
  ctrl    : vec4u,   // iters, opCount, addC, maxSteps
  prm     : vec4f,   // bailout, epsilon, deScale, colorMode
  colA    : vec4f,   // low / base color (rgb) ; .w = deOption (0 escape, 2 IFS)
  colB    : vec4f,   // high color (rgb)
  bgc     : vec4f,   // background color (rgb)
  jc      : vec4f,   // Julia constant (xyz) ; .w > 0.5 = Julia mode on
  palA    : vec4f,   // cosine palette a (rgb) ; .w > 0.5 = palette on
  palB    : vec4f,   // cosine palette b (rgb)
  palC    : vec4f,   // cosine palette c (rgb, frequency)
  palD    : vec4f,   // cosine palette d (rgb, phase)
  light   : vec4f,   // light direction (xyz)
  lprm    : vec4f,   // x=ambient, y=rim, z=gloss, w=light intensity
};
struct Op { opType: u32, p0: f32, p1: f32, p2: f32 };

@group(0) @binding(0) var<uniform> G : Globals;
@group(0) @binding(1) var<storage, read> ops : array<Op>;

// One iteration step = run every op in the stack, then (if AddC) re-add c.
fn mapDE(p0: vec3f) -> f32 {
  var pos = p0;
  var w = 1.0;
  let c = select(p0, G.jc.xyz, G.jc.w > 0.5);  // Julia: fixed c, else sample point
  let n = G.ctrl.y;
  for (var i: u32 = 0u; i < G.ctrl.x; i = i + 1u) {
    for (var o: u32 = 0u; o < n; o = o + 1u) {
      let op = ops[o];
      switch op.opType {
${cases}
        default: {}
      }
    }
    if (G.ctrl.z != 0u) { pos = pos + c; }
    if (dot(pos, pos) > G.prm.x) { break; }
  }
  let r = length(pos);
  // DEoption 0 — escape-time DE (Mandelbulb / power): 0.5·ln(r)·r / dr, with the
  // analytic derivative dr carried in w. DEoption 2 — analytic IFS r/|w|.
  if (G.colA.w < 1.0) { return 0.5 * log(max(r, 1e-9)) * r / max(abs(w), 1e-9); }
  return r / max(abs(w), 1e-9);
}

// Orbit trap: the closest the iterated point came to the origin. Re-runs the
// iteration once (only at the final hit point, so it's cheap) to drive coloring.
fn orbitTrap(p0: vec3f) -> f32 {
  var pos = p0;
  var w = 1.0;
  let c = select(p0, G.jc.xyz, G.jc.w > 0.5);  // Julia: fixed c, else sample point
  let n = G.ctrl.y;
  var tr = 1.0e9;
  for (var i: u32 = 0u; i < G.ctrl.x; i = i + 1u) {
    for (var o: u32 = 0u; o < n; o = o + 1u) {
      let op = ops[o];
      switch op.opType {
${cases}
        default: {}
      }
    }
    if (G.ctrl.z != 0u) { pos = pos + c; }
    tr = min(tr, length(pos));
    if (dot(pos, pos) > G.prm.x) { break; }
  }
  return tr;
}

// Escape iteration fraction (for "bands" coloring): how many iterations until the
// point flies past the bailout, normalized 0..1. Re-runs the iteration once.
fn escapeIter(p0: vec3f) -> f32 {
  var pos = p0;
  var w = 1.0;
  let c = select(p0, G.jc.xyz, G.jc.w > 0.5);
  let n = G.ctrl.y;
  var esc: u32 = G.ctrl.x;
  for (var i: u32 = 0u; i < G.ctrl.x; i = i + 1u) {
    for (var o: u32 = 0u; o < n; o = o + 1u) {
      let op = ops[o];
      switch op.opType {
${cases}
        default: {}
      }
    }
    if (G.ctrl.z != 0u) { pos = pos + c; }
    if (dot(pos, pos) > G.prm.x) { esc = i; break; }
  }
  return f32(esc) / f32(max(G.ctrl.x, 1u));
}

fn calcNormal(p: vec3f) -> vec3f {
  let e = vec2f(1.0, -1.0) * 0.0006;
  return normalize(
      e.xyy * mapDE(p + e.xyy) +
      e.yyx * mapDE(p + e.yyx) +
      e.yxy * mapDE(p + e.yxy) +
      e.xxx * mapDE(p + e.xxx));
}

struct VSOut { @builtin(position) clip: vec4f, @location(0) uv: vec2f };

@vertex fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  var tri = array<vec2f, 3>(vec2f(-1.0,-1.0), vec2f(3.0,-1.0), vec2f(-1.0,3.0));
  var o: VSOut;
  let xy = tri[vi];
  o.clip = vec4f(xy, 0.0, 1.0);
  o.uv = (xy + vec2f(1.0)) * 0.5;
  return o;
}

// sRGB → linear: albedo/picker/theme colors are authored in sRGB; linearize before
// lighting so the 1/2.2 encode at the end round-trips them to the picked color
// (issue #6 — render now matches the GUI). 2.2 inverts the final 1/2.2.
fn s2l(c: vec3f) -> vec3f { return pow(max(c, vec3f(0.0)), vec3f(2.2)); }

@fragment fn fs(@location(0) uv: vec2f) -> @location(0) vec4f {
  let ndc = uv * 2.0 - vec2f(1.0);
  let aspect = G.res.x / G.res.y;
  let tanF = tan(0.5 * G.res.z);
  let rd = normalize(G.camFwd.xyz
      + (ndc.x * aspect * tanF) * G.camRight.xyz
      + (ndc.y * tanF) * G.camUp.xyz);
  let ro = G.camPos.xyz;

  let bg = s2l(mix(G.bgc.rgb * 0.35, G.bgc.rgb, clamp(uv.y, 0.0, 1.0)));

  var t = 0.02;
  var steps: u32 = 0u;
  var hit = false;
  let maxSteps = G.ctrl.w;
  let eps = G.prm.y;
  for (steps = 0u; steps < maxSteps; steps = steps + 1u) {
    let p = ro + rd * t;
    let d = mapDE(p) * G.prm.z;
    if (d < eps * t) { hit = true; break; }
    t = t + d;
    if (t > 80.0) { break; }
  }
  if (!hit) { return vec4f(bg, 1.0); }

  let p = ro + rd * t;
  let nrm = calcNormal(p);
  let ao = 1.0 - f32(steps) / f32(maxSteps);

  let lightDir = normalize(G.light.xyz);
  let diff = max(dot(nrm, lightDir), 0.0);
  let rim = pow(1.0 - max(dot(nrm, -rd), 0.0), 2.0);
  let amb = G.lprm.x;

  // mixT source by mode: 0 surface (normal), 1 orbit-trap glow, 2 escape bands.
  var mixT: f32;
  if      (G.prm.w > 1.5) { mixT = escapeIter(p); }
  else if (G.prm.w > 0.5) { mixT = clamp(orbitTrap(p) / 1.5, 0.0, 1.0); }
  else                    { mixT = 0.5 + 0.5 * nrm.z; }

  // Albedo: cosine palette (a + b·cos(2π(c·t + d))) or the plain colA→colB ramp.
  var albedo: vec3f;
  if (G.palA.w > 0.5) {
    albedo = clamp(G.palA.rgb + G.palB.rgb * cos(6.2831853 * (G.palC.rgb * mixT + G.palD.rgb)),
                   vec3f(0.0), vec3f(1.0));
  } else {
    albedo = mix(G.colA.rgb, G.colB.rgb, mixT);
  }
  albedo = s2l(albedo); // sRGB→linear (issue #6)

  let halfv = normalize(lightDir - rd);                       // Blinn-Phong half-vector
  let spec = G.lprm.z * pow(max(dot(nrm, halfv), 0.0), 32.0);
  var col = albedo * (amb + (1.0 - amb) * diff) * (0.35 + 0.65 * ao);
  col = col + vec3f(spec) * ao;
  col = col + s2l(G.colB.rgb) * (rim * G.lprm.y * ao);
  col = col * G.lprm.w;                                   // light intensity (exposure)
  col = mix(col, bg, clamp(t / 80.0, 0.0, 1.0) * 0.6);   // distance fade
  col = pow(max(col, vec3f(0.0)), vec3f(1.0 / 2.2));      // gamma
  return vec4f(col, 1.0);
}
`;
}

// ── Native GLSL export (desktop iterateJIT_ body) ──────────────────────────
export function exportGLSL(formula) {
  const names = []; // PARAM_NAMES
  const types = []; // PARAM_TYPES
  const defs = []; // DEFAULTS
  const ranges = []; // PARAM_RANGES (min:max:step — for desktop slider bounds)
  const decls = []; // local `float pN = getGenericParam(...)`
  const body = []; // op snippets

  let slot = 0;
  const seen = new Map();
  const ops = activeOps(formula); // muted ops are omitted from the export
  for (const op of ops) {
    const def = byKey(op.key);
    const vars = def.params.map((pm, i) => {
      // unique-ify duplicate param names across repeated ops
      let nm = pm.name;
      const n = (seen.get(nm) || 0) + 1;
      seen.set(nm, n);
      if (n > 1) nm = `${nm}${n}`;
      names.push(nm);
      types.push(pm.type === "angle" ? "DoubleAngle" : "Double");
      defs.push(op.values[i]);
      // Authored slider bounds (same units as DEFAULTS — degrees for angles).
      ranges.push(`${pm.min}:${pm.max}:${pm.step}`);

      const vn = `p${slot}`;
      const get = `getGenericParam(slot, ${slot})`;
      decls.push(
        `    float ${vn} = ${pm.type === "angle" ? `radians(${get})` : get};`,
      );
      slot++;
      return vn;
    });
    body.push(def.glsl(vars));
  }

  // Julia mode: the iteration is f(z) + jc with jc FIXED, so we bake the
  // constant into the body and turn AddC OFF (else the engine would also re-add
  // the world seed c). For a non-Julia formula AddC is left as the preset set it.
  const julia = !!formula.julia;
  const jc = formula.juliaC || [0, 0, 0];
  if (julia)
    body.push(`
    // Julia constant (baked: this formula adds a FIXED c, not the world seed)
    pos += vec3(${jc[0]}, ${jc[1]}, ${jc[2]});`);
  const effAddC = julia ? false : formula.addC;

  const safe = formula.name.replace(/[^A-Za-z0-9_]/g, "_");
  return `// HAND_CRAFTED: generated by the web formula creator (op-list export).
// JIT formula: ${formula.name} (DEscale=0.0)
// JIT_VERSION: 2
// DEFAULTS: ${defs.join(",")}
// PARAM_NAMES: ${names.join(",")}
// PARAM_TYPES: ${types.join(",")}
// PARAM_RANGES: ${ranges.join(",")}
// AddC: ${effAddC ? "true" : "false"}
// DEoption: ${effectiveDeOption(formula)}
//
// Composed from ${ops.length} primitive(s):
//   ${ops.map((o) => o.key).join(" → ")}${effAddC ? "  (+c)" : ""}${julia ? `  (Julia c = ${jc.join(", ")})` : ""}
void iterateJIT_${safe}(int slot, vec3 c, inout vec3 pos, inout float w) {
${decls.join("\n")}
${body.join("\n")}
}
`;
}

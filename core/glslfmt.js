// GLSL literal formatting — shared by the desktop JIT export (shader.js) and the
// standalone GLSL exporter (exportStandalone.js). Lifted out of shader.js so both
// consumers share one definition; it's a leaf module (no core imports) so routing
// shader.js and shader_gl.js through it can't create an import cycle.

// GLSL float literal — GLSL needs a decimal point (`6.0`, not `6`).
export const glslNum = (v) => {
  const n = Number(v) || 0;
  return Number.isInteger(n) ? n.toFixed(1) : String(n);
};

// vecN(...) constructor literal, e.g. glslVec("vec3", [1,0,0]) → "vec3(1.0, 0.0, 0.0)".
// ALWAYS emits exactly N components for the type (pads missing with 0) — a GLSL
// constructor needs args, so `vec4()` from an empty/short value is a compile error
// ("constructor does not have any arguments"); e.g. a padded, palette-off stop.
const VEC_N = { vec2: 2, vec3: 3, vec4: 4 };
export const glslVec = (type, arr) => {
  const n = VEC_N[type] ?? 0;
  const src = Array.from(arr ?? []); // Array.from(0)→[]; guards a scalar/undefined pad
  return `${type}(${Array.from({ length: n }, (_, i) => glslNum(src[i] ?? 0)).join(", ")})`;
};

// Fixed-size array constructor literal. Pads (or truncates) to exactly `n`
// elements so the initializer count matches the declared size — a GLSL compile
// error otherwise (e.g. `vec4[8]()` with an empty palette). `each` formats one
// element (default glslNum for scalars; pass glslVec-bound for vec arrays).
export const glslArr = (type, n, arr, each = glslNum) => {
  const src = Array.from(arr);
  const elems = [];
  for (let i = 0; i < n; i++) elems.push(each(src[i] ?? 0));
  return `${type}[${n}](${elems.join(", ")})`;
};

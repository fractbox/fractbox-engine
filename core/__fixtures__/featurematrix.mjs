// #729 — the SELF-MAINTAINING codegen feature matrix.
//
// The byte-identity gates (clipplane / equirect / tinyplanet, and any future
// feature gate) used to each carry a hand-copied WGSL_MATRIX, and the copies
// had already drifted: newer flags were absent from older features'
// off-sweeps, so "feature X off is byte-identical across the matrix" was
// never checked against the features that landed after X's gate was written —
// the register-pressure class these gates exist for (the +31% Mandelbulb
// precedent, core/shader.js:12-21) could ship under a green byte-identity
// claim. The maintenance cost was quadratic (N features × N-1 hand lists).
//
// This module derives the flag list FROM buildWGSL ITSELF — the destructured
// parameter defaults in its own source text — so a new flag automatically
// enters every off-sweep the moment it exists, and no gate ever needs its
// matrix touched again. core/featurematrix.test.mjs is the meta-gate: it pins
// the parse against reality (sentinel flags, every flag live in codegen).
//
// Usage in a feature gate:
//   import { wgslMatrix } from "./__fixtures__/featurematrix.mjs";
//   const WGSL_MATRIX = wgslMatrix({ except: ["clip", "clipJag"] });
// `except` removes every entry that involves the gate's OWN flags (their
// on-entries would legitimately carry the tokens the off-sweep must reject).
import { buildWGSL } from "../shader.js";

// Flags whose "on" text only exists on top of a base flag — the entry carries
// the base so the flip is live (a bare sub-flag emits the default text).
const DEPENDENT_BASES = { clipJag: { clip: true } };

// Standing multi-feature entries — seams where several features splice next
// to each other are exactly where a regression hides from single-flag
// entries. planet and equirect never share an entry (exclusive projections —
// buildWGSL throws).
const COMBOS = [
  ["planet+aurora+thinFilm", { planet: true, aurora: true, thinFilm: true }],
  ["equirect+aurora+thinFilm", { equirect: true, aurora: true, thinFilm: true }],
  ["envx+envMap+surfTex", { envx: true, envMap: true, surfTex: true }],
];

/**
 * The buildWGSL flag table, parsed from the function's own source:
 * name → default (true | false | null). A new destructured flag appears here
 * with no further wiring.
 */
export function wgslFlags() {
  const src = buildWGSL.toString();
  const open = src.indexOf("({");
  const close = src.indexOf("} = {})");
  if (open < 0 || close < 0)
    throw new Error(
      "featurematrix: buildWGSL no longer destructures `({...} = {})` — update the parse",
    );
  const flags = new Map();
  for (const m of src
    .slice(open + 2, close)
    .matchAll(/^\s*([A-Za-z_$][\w$]*)\s*=\s*(true|false|null),?\s*$/gm))
    flags.set(m[1], m[2] === "true" ? true : m[2] === "false" ? false : null);
  if (flags.size < 15)
    throw new Error(
      `featurematrix: parsed only ${flags.size} buildWGSL flags — the parse ` +
        "broke, and a gate sweeping this matrix would silently shrink",
    );
  return flags;
}

/**
 * The off-sweep matrix: [name, buildWGSL-opts] pairs — the default build, the
 * minimal build (every default-true lever off at once), every flag flipped
 * from its default (dependent flags ride their base), the op-set lever, and
 * the standing combined entries. Entries touching a flag in `except` are
 * dropped (a gate never sweeps its own feature's on-entries).
 */
export function wgslMatrix({ except = [] } = {}) {
  const flags = wgslFlags();
  const skip = new Set(except);
  const touches = (opts) => Object.keys(opts).some((k) => skip.has(k));
  const entries = [["default", {}]];

  const minimal = {};
  for (const [k, v] of flags) if (v === true && !skip.has(k)) minimal[k] = false;
  entries.push(["minimal", minimal]);

  for (const [k, v] of flags) {
    if (typeof v !== "boolean") continue; // `ops` — handled below
    const opts = { ...(DEPENDENT_BASES[k] || {}), [k]: !v };
    if (touches(opts)) continue;
    entries.push([v === false ? k : `${k}:off`, opts]);
  }

  entries.push(["ops:[]", { ops: [] }]);
  entries.push(["ops:[1,2,3]", { ops: [1, 2, 3] }]);

  for (const [name, opts] of COMBOS)
    if (!touches(opts)) entries.push([name, opts]);

  return entries;
}

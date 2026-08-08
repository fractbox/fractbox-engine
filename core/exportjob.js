// ExportJob — the export kit's wire format (EXPORT_APP_RESEARCH §7 P0).
// A job is everything an export needs that ISN'T host machinery: the formula,
// optional coloring, optional iters/live-view snapshot, and the resolved flat
// knob set. Quality PRESETS stay app-side — a job carries RESOLVED opts, so any
// consumer (app dialog, CLI --job, a future export app) reads the same shape
// without preset tables. Host-side concerns are deliberately NOT here:
//  - forceCPU: a diagnostic of the RUNNING host's environment, meaningless in
//    a job replayed elsewhere;
//  - aspect: screen-derived (canvas w/h) — the host supplies it at call time
//    (headless consumers use 1, exactly like the CLI --view path today).
// normalizeExportJob validates STRUCTURE only — it does NOT check that
// formula.ops keys exist in this engine's op vocabulary. "Job normalized" ≠
// "engine-version compatible" (the §9 skew marker is a later, separate check).
//
// Defaults are the CONSERVATIVE set — a sparse job behaves like a bare CLI
// invocation (refine/degamma/viewFraming off, standard-tier capture
// geometry), NOT like the app dialog (which always writes every knob
// explicitly). TWO deliberate divergences, both picking the coherent quality
// default over reproducing a legacy inconsistency: radiusScale defaults to
// 1.2 (the modern app/auto/σ default — the legacy CLI's internal reduce
// fallback was 1.6, which its own auto/σ paths already ignored), and
// autoRadius defaults ON (#431: without the solve, the fixed spacing radius
// leaves ~3.3× overdraw — visibly fat "rings" — where the solver lands at the
// σ-calibrated ~2.3 target; coverage always wins over the floor, so auto can
// only shrink, never open holes).

export const EXPORTJOB_VERSION = 1;

const CONVENTIONS = ["ue", "raw"];
const SIZE_UNITS = ["mm", "cm", "m", "km"];
const FORMATS = ["ply", "spz"]; // #368: ply = uncompressed INRIA (UE); spz = compressed web/mobile

const OPT_DEFAULTS = {
  views: 64,
  res: 256,
  cap: 1_500_000,
  layers: 2,
  aoStrength: 0.5,
  radiusScale: 1.2,
  thinEps: 0.1,
  convention: "ue",
  sizeUnit: "m",
  format: "ply",
  degamma: false,
  fRest: 0,
  aniso: 0,
  anisoMax: 3,
  autoRadius: true, // #431 — see the divergence note above
  refine: false,
  viewFraming: false,
};

// [min, max, integer?] sanity ranges — structural guards, not tuning advice.
const OPT_RANGES = {
  views: [1, 4096, true],
  res: [16, 4096, true],
  cap: [1000, 50_000_000, true],
  layers: [1, 8, true],
  aoStrength: [0, 1],
  radiusScale: [0.1, 10],
  thinEps: [0.01, 1], // fitSplats throws below 0.01 — fail at the boundary instead
  aniso: [0, 1],
  anisoMax: [1, 32],
};

const fail = (msg) => {
  throw new TypeError(`exportjob: ${msg}`);
};

function checkNum(name, v, [lo, hi, int]) {
  if (typeof v !== "number" || !Number.isFinite(v))
    fail(`opts.${name} must be a finite number (got ${v})`);
  if (v < lo || v > hi) fail(`opts.${name} ${v} outside [${lo}, ${hi}]`);
  if (int && !Number.isInteger(v))
    fail(`opts.${name} must be an integer (got ${v})`);
}

// Validate + default-fill a raw job object. Returns a NEW normalized job
// (inputs are not mutated); throws TypeError on structural garbage.
export function normalizeExportJob(raw) {
  if (!raw || typeof raw !== "object") fail("job must be an object");
  if (raw.version != null && raw.version !== EXPORTJOB_VERSION)
    fail(
      `unsupported job version ${raw.version} (this engine: ${EXPORTJOB_VERSION})`,
    );

  const f = raw.formula;
  if (!f || typeof f !== "object") fail("job.formula is required");
  const scene = Array.isArray(f.objects) && f.objects.length > 0;
  if (!scene && !Array.isArray(f.ops))
    fail("job.formula needs ops[] (or objects[] for a scene)");

  if (raw.iters != null) {
    if (!Number.isInteger(raw.iters) || raw.iters < 1 || raw.iters > 10_000)
      fail(`job.iters must be an integer in [1, 10000] (got ${raw.iters})`);
  }

  // S-5a live-view snapshot — a REDUCED shape (dist/target/fovDeg), distinct
  // from the canonical formula.camera (yaw/pitch/dist/…), which stays untouched
  // inside formula. No aspect (host-supplied). Optional; only meaningful when
  // opts.viewFraming is on.
  let viewCamera;
  if (raw.viewCamera != null) {
    const c = raw.viewCamera;
    if (typeof c !== "object") fail("job.viewCamera must be an object");
    if (typeof c.dist !== "number" || !(c.dist > 0) || !Number.isFinite(c.dist))
      fail(`job.viewCamera.dist must be > 0 (got ${c.dist})`);
    if (c.target != null) {
      if (
        !Array.isArray(c.target) ||
        c.target.length !== 3 ||
        c.target.some((t) => typeof t !== "number" || !Number.isFinite(t))
      )
        fail("job.viewCamera.target must be [x, y, z] numbers");
    }
    if (c.fovDeg != null && !(c.fovDeg > 0 && c.fovDeg < 180))
      fail(`job.viewCamera.fovDeg must be in (0, 180) (got ${c.fovDeg})`);
    viewCamera = {
      dist: c.dist,
      ...(c.target != null ? { target: [...c.target] } : {}),
      ...(c.fovDeg != null ? { fovDeg: c.fovDeg } : {}),
    };
  }

  // An EXPLICIT capture volume (SPLAT_FRAMING_GIZMO P1) — the user placed and
  // sized this box themselves, so unlike viewCamera it is honoured VERBATIM:
  // exportFrame neither grows it nor caps it at the whole-object frame. It
  // outranks viewCamera when both are present.
  //
  // `ext` (optional) makes it a CUBOID: viewBasis now sizes each view's window
  // from the volume's support and captureView rejects hits outside it, so three
  // different half-extents capture as drawn (CAPTURE_VOLUME_SHAPES.md). `radius`
  // stays required and remains the scale scalar (eps, AO probe, r0); omitting
  // `ext` is the uniform box every pre-cuboid job and share link already sends.
  let captureBox;
  if (raw.captureBox != null) {
    const b = raw.captureBox;
    if (typeof b !== "object") fail("job.captureBox must be an object");
    if (
      !Array.isArray(b.center) ||
      b.center.length !== 3 ||
      b.center.some((t) => typeof t !== "number" || !Number.isFinite(t))
    )
      fail("job.captureBox.center must be [x, y, z] numbers");
    if (
      typeof b.radius !== "number" ||
      !(b.radius > 0) ||
      !Number.isFinite(b.radius)
    )
      fail(`job.captureBox.radius must be > 0 (got ${b.radius})`);
    let ext;
    if (b.ext != null) {
      if (
        !Array.isArray(b.ext) ||
        b.ext.length !== 3 ||
        b.ext.some(
          (t) => typeof t !== "number" || !Number.isFinite(t) || t <= 0,
        )
      )
        fail("job.captureBox.ext must be three half-extents > 0");
      ext = [...b.ext];
    }
    // Shape (CAPTURE_VOLUME_SHAPES): 0 = box, 1 = ellipsoid, 2 = cylinder (z).
    // Absent = box, which is what every job written before shapes existed means.
    let kind;
    if (b.kind != null) {
      if (!Number.isInteger(b.kind) || b.kind < 0 || b.kind > 2)
        fail(`job.captureBox.kind must be 0 (box), 1 (ellipsoid) or 2 (cylinder) (got ${b.kind})`);
      kind = b.kind;
    }
    // Orientation (CAPTURE_VOLUME_SHAPES): the volume's first two LOCAL axes in
    // world space; the third is their cross product. Absent = axis-aligned,
    // which is every job written before volumes could be turned. Carried as the
    // general basis rather than an axis enum so that drag-to-rotate needs no
    // wire-format change — the UI's X/Y/Z presets are just three stored bases.
    let rot;
    if (b.rot != null) {
      if (
        !Array.isArray(b.rot) ||
        b.rot.length !== 6 ||
        b.rot.some((t) => typeof t !== "number" || !Number.isFinite(t))
      )
        fail("job.captureBox.rot must be six finite numbers (two local axes)");
      rot = [...b.rot];
    }
    captureBox = {
      center: [...b.center],
      radius: b.radius,
      ...(ext ? { ext } : {}),
      ...(kind ? { kind } : {}),
      ...(rot ? { rot } : {}),
    };
  }

  const rawOpts = raw.opts ?? {};
  if (typeof rawOpts !== "object") fail("job.opts must be an object");
  const opts = { ...OPT_DEFAULTS };
  for (const k of Object.keys(rawOpts)) {
    const v = rawOpts[k];
    if (v == null) continue; // absent/null ⇒ default (lets a merged CLI layer pass undefined)
    if (!(k in OPT_DEFAULTS)) fail(`unknown opts key "${k}"`);
    if (k === "convention") {
      if (!CONVENTIONS.includes(v))
        fail(`opts.convention must be one of ${CONVENTIONS} (got ${v})`);
    } else if (k === "sizeUnit") {
      if (!SIZE_UNITS.includes(v))
        fail(`opts.sizeUnit must be one of ${SIZE_UNITS} (got ${v})`);
    } else if (k === "format") {
      if (!FORMATS.includes(v))
        fail(`opts.format must be one of ${FORMATS} (got ${v})`);
    } else if (k === "fRest") {
      if (v !== 0 && v !== 45) fail(`opts.fRest must be 0 or 45 (got ${v})`);
    } else if (typeof OPT_DEFAULTS[k] === "boolean") {
      if (typeof v !== "boolean")
        fail(`opts.${k} must be a boolean (got ${v})`);
    } else {
      checkNum(k, v, OPT_RANGES[k]);
    }
    opts[k] = v;
  }

  return {
    version: EXPORTJOB_VERSION,
    formula: f,
    ...(raw.coloring != null ? { coloring: raw.coloring } : {}),
    ...(raw.iters != null ? { iters: raw.iters } : {}),
    ...(viewCamera ? { viewCamera } : {}),
    ...(captureBox ? { captureBox } : {}),
    // The §9 op-skew marker (TESSELAVA_P2 PR-1): the PRODUCING engine's
    // version string, so a consumer built from a different tree can refuse
    // newer-than-known ops gracefully. OPTIONAL + advisory; absent means
    // same-repo (zero skew while both faces build from one monorepo commit).
    // EXPORTJOB_VERSION stays 1 — this is a pass-through, not a shape change.
    ...(typeof raw.engine === "string" && raw.engine
      ? { engine: raw.engine }
      : {}),
    opts,
  };
}

// JSON round-trip — the on-disk / cross-app form. fromJSON normalizes (so a
// consumer can trust every knob is present + sane after a single call).
export function exportJobToJSON(job) {
  return JSON.stringify(normalizeExportJob(job), null, 2);
}

export function exportJobFromJSON(text) {
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch (e) {
    fail(`invalid JSON (${e.message})`);
  }
  return normalizeExportJob(parsed);
}

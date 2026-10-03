import { test } from "node:test";
import assert from "node:assert/strict";
import {
  EXPORTJOB_VERSION,
  normalizeExportJob,
  exportJobToJSON,
  exportJobFromJSON,
} from "./exportjob.js";

const FORMULA = { ops: [{ key: "boxFold", values: [1, 0, 0] }], iters: 12 };

test("normalizeExportJob: sparse job gets conservative defaults + version stamp", () => {
  const j = normalizeExportJob({ formula: FORMULA });
  assert.equal(j.version, EXPORTJOB_VERSION);
  assert.equal(j.formula, FORMULA);
  // Conservative (CLI-compatible) defaults — NOT the app dialog's
  assert.equal(j.opts.views, 64);
  assert.equal(j.opts.res, 256);
  assert.equal(j.opts.cap, 1_500_000);
  assert.equal(j.opts.layers, 2);
  assert.equal(j.opts.convention, "ue");
  assert.equal(j.opts.sizeUnit, "m");
  assert.equal(j.opts.format, "ply"); // #368: uncompressed default

  assert.equal(j.opts.degamma, false);
  // #431: autoRadius defaults ON (divergence #2 — without the solve the fixed
  // spacing radius ships ~3.3× overdraw, the "rings too big" report).
  assert.equal(j.opts.autoRadius, true);
  assert.equal(j.opts.refine, false);
  assert.equal(j.opts.viewFraming, false);
  assert.equal(j.opts.radiusScale, 1.2);
  // host-side knobs must NOT leak into the wire format
  assert.ok(!("forceCPU" in j.opts), "forceCPU is host-side, not job");
  // absent optionals stay absent (not null-stuffed)
  assert.ok(!("iters" in j));
  assert.ok(!("viewCamera" in j));
  assert.ok(!("coloring" in j));
});

test("normalizeExportJob: explicit knobs survive; null/undefined fall to defaults", () => {
  const j = normalizeExportJob({
    formula: FORMULA,
    iters: 96,
    coloring: { mode: 2 },
    opts: {
      views: 192,
      res: 384,
      refine: true,
      degamma: true,
      radiusScale: undefined,
    },
  });
  assert.equal(j.opts.views, 192);
  assert.equal(j.opts.res, 384);
  assert.equal(j.opts.refine, true);
  assert.equal(j.opts.degamma, true);
  assert.equal(
    j.opts.radiusScale,
    1.2,
    "undefined ⇒ default (merged-layer contract)",
  );
  assert.equal(j.iters, 96);
  assert.deepEqual(j.coloring, { mode: 2 });
});

// #543: raw.coloring used to ride through unvalidated (`{ coloring: raw.coloring }`)
// — the export-job JSON intake (tesselava / CLI --job) bypassed the sanitizeColoring
// hardening #542 shipped for sanitizeScene. These pin that the job still loads
// (sanitizeColoring never throws) and that the coloring it carries is clamped to
// the model's ranges, not passed through raw.
test("normalizeExportJob: garbage coloring is sanitized, not passed through raw (#543)", () => {
  // Out-of-range mode + an oversized palette-stop array, same shape a hostile
  // hand-edited job JSON (or a tesselava intake carrying a bad share link) could
  // send straight through --job.
  const stops = Array.from({ length: 50 }, (_, i) => ({
    c: [i / 50, i / 50, i / 50],
    p: i / 50,
  }));
  const j = normalizeExportJob({
    formula: FORMULA,
    coloring: { mode: 999, palette: { on: true, stops } },
  });
  assert.equal(j.coloring.mode, 7, "mode clamped to COLOR_MODE_MAX");
  assert.equal(
    j.coloring.palette.stops.length,
    8,
    "stop count capped at the model's max, not the 50 the job declared",
  );
});

test("normalizeExportJob: unusable coloring is dropped, job still loads (#543)", () => {
  // A non-object coloring is exactly what sanitizeColoring treats as garbage —
  // it returns undefined rather than throwing, so the job loads without one.
  const j = normalizeExportJob({ formula: FORMULA, coloring: "havoc" });
  assert.ok(!("coloring" in j), "unusable coloring is dropped, not thrown on");
});

test("normalizeExportJob: viewCamera reduced shape — no aspect, no yaw/pitch", () => {
  const j = normalizeExportJob({
    formula: FORMULA,
    viewCamera: { dist: 0.4, target: [0.5, -0.1, 0.3], fovDeg: 42 },
    opts: { viewFraming: true },
  });
  assert.deepEqual(j.viewCamera, {
    dist: 0.4,
    target: [0.5, -0.1, 0.3],
    fovDeg: 42,
  });
  // dist-only is valid (target/fovDeg optional)
  const j2 = normalizeExportJob({ formula: FORMULA, viewCamera: { dist: 1 } });
  assert.deepEqual(j2.viewCamera, { dist: 1 });
});

test("normalizeExportJob: rejects structural garbage", () => {
  const bad = [
    [null, /job must be an object/],
    [{}, /formula is required/],
    [{ formula: { name: "x" } }, /needs ops/],
    [{ formula: FORMULA, version: 99 }, /unsupported job version/],
    [{ formula: FORMULA, iters: 0 }, /iters/],
    [{ formula: FORMULA, iters: 2.5 }, /iters/],
    [{ formula: FORMULA, opts: { nope: 1 } }, /unknown opts key/],
    [{ formula: FORMULA, opts: { views: 0 } }, /views/],
    [{ formula: FORMULA, opts: { res: NaN } }, /res/],
    [{ formula: FORMULA, opts: { thinEps: 0.001 } }, /thinEps/],
    [{ formula: FORMULA, opts: { convention: "unity" } }, /convention/],
    [{ formula: FORMULA, opts: { sizeUnit: "furlong" } }, /sizeUnit/],
    [{ formula: FORMULA, opts: { format: "obj" } }, /format/], // #368
    [{ formula: FORMULA, opts: { fRest: 9 } }, /fRest/],
    [{ formula: FORMULA, opts: { refine: 1 } }, /boolean/],
    [{ formula: FORMULA, viewCamera: { dist: -1 } }, /dist/],
    [{ formula: FORMULA, viewCamera: { dist: 1, target: [1, 2] } }, /target/],
    [{ formula: FORMULA, viewCamera: { dist: 1, fovDeg: 200 } }, /fovDeg/],
  ];
  for (const [raw, re] of bad)
    assert.throws(() => normalizeExportJob(raw), re, JSON.stringify(raw));
});

test("normalizeExportJob: scene formulas (objects[], no top-level ops) pass", () => {
  const j = normalizeExportJob({
    formula: { objects: [{ ops: [], objType: 1 }] },
  });
  assert.equal(j.version, EXPORTJOB_VERSION);
});

test("JSON round-trip is stable and normalizing", () => {
  const text = exportJobToJSON({ formula: FORMULA, opts: { cap: 2_500_000 } });
  const back = exportJobFromJSON(text);
  assert.equal(back.opts.cap, 2_500_000);
  assert.equal(back.opts.views, 64, "defaults filled on read");
  assert.equal(exportJobToJSON(back), text, "round-trip fixed point");
  assert.throws(() => exportJobFromJSON("{nope"), /invalid JSON/);
});

test("normalizeExportJob does not mutate its input", () => {
  const raw = { formula: FORMULA, opts: { views: 100 } };
  const frozen = JSON.stringify(raw);
  normalizeExportJob(raw);
  assert.equal(JSON.stringify(raw), frozen);
});

test("engine marker (P2 PR-1): optional pass-through string; absent = same-repo", () => {
  const base = { formula: { ops: [{ key: "scale", values: [2] }], iters: 8 } };
  assert.ok(!("engine" in normalizeExportJob(base)), "absent stays absent");
  const j = normalizeExportJob({ ...base, engine: "fractbox@1.1.0" });
  assert.equal(j.engine, "fractbox@1.1.0");
  const rt = exportJobFromJSON(
    exportJobToJSON({ ...base, engine: "fractbox@1.1.0" }),
  );
  assert.equal(rt.engine, "fractbox@1.1.0", "survives the JSON round-trip");
  assert.ok(
    !("engine" in normalizeExportJob({ ...base, engine: 42 })),
    "non-string dropped",
  );
});

// SPLAT_FRAMING_GIZMO P1 — the explicit, user-placed capture volume. The trap
// this pins: normalizeExportJob's return literal WHITELISTS top-level keys and
// does not spread `raw`, so a field that is validated but not returned is
// silently dropped on every round-trip.
// CAPTURE_VOLUME_SHAPES: ext + kind must survive normalize. normalizeExportJob
// returns a WHITELIST literal, so a field that is validated but not rebuilt is
// silently dropped — exactly how a cuboid box could reach the engine as a cube.
test("captureBox carries ext + kind through normalize (whitelist trap)", () => {
  const j = normalizeExportJob({
    formula: { ops: [] },
    captureBox: { center: [0, 0, 0], radius: 4, ext: [4, 2, 1], kind: 1 },
  });
  assert.deepEqual(j.captureBox, {
    center: [0, 0, 0],
    radius: 4,
    ext: [4, 2, 1],
    kind: 1,
  });
  const back = JSON.parse(exportJobToJSON(j));
  assert.deepEqual(back.captureBox.ext, [4, 2, 1]);
  assert.equal(back.captureBox.kind, 1);
});

test("captureBox rejects a nonsense shape or extent", () => {
  const bad = (captureBox) =>
    assert.throws(
      () => normalizeExportJob({ formula: { ops: [] }, captureBox }),
      TypeError,
    );
  bad({ center: [0, 0, 0], radius: 1, kind: 7 });
  bad({ center: [0, 0, 0], radius: 1, kind: 1.5 });
  bad({ center: [0, 0, 0], radius: 1, ext: [1, 0, 1] });
  bad({ center: [0, 0, 0], radius: 1, ext: [1, 1] });
  // absent kind/ext is the pre-shapes box and must still pass
  const ok = normalizeExportJob({
    formula: { ops: [] },
    captureBox: { center: [0, 0, 0], radius: 1 },
  });
  assert.equal(ok.captureBox.kind, undefined);
});

test("captureBox survives normalize + a JSON round-trip", () => {
  const j = normalizeExportJob({
    formula: { ops: [] },
    captureBox: { center: [1, -2, 0.5], radius: 3 },
  });
  assert.deepEqual(j.captureBox, { center: [1, -2, 0.5], radius: 3 });
  const back = exportJobFromJSON(exportJobToJSON(j));
  assert.deepEqual(back.captureBox, { center: [1, -2, 0.5], radius: 3 });
  // absent stays absent — an automatic volume must not gain a phantom box
  assert.equal("captureBox" in normalizeExportJob({ formula: { ops: [] } }), false);
});

test("captureBox is validated like viewCamera", () => {
  const bad = (b) => () =>
    normalizeExportJob({ formula: { ops: [] }, captureBox: b });
  assert.throws(bad({ center: [0, 0, 0], radius: 0 }), /radius must be > 0/);
  assert.throws(bad({ center: [0, 0, 0], radius: -1 }), /radius must be > 0/);
  assert.throws(bad({ center: [0, 0, 0], radius: NaN }), /radius must be > 0/);
  assert.throws(bad({ center: [0, 0], radius: 1 }), /center must be/);
  assert.throws(bad({ center: [0, 0, "x"], radius: 1 }), /center must be/);
  // the centre is COPIED, not aliased — a later mutation must not reach the job
  const src = [1, 2, 3];
  const j = normalizeExportJob({
    formula: { ops: [] },
    captureBox: { center: src, radius: 1 },
  });
  src[0] = 99;
  assert.equal(j.captureBox.center[0], 1);
});

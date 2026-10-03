// Contract pins for the capture drive (EXPORT_P1 PR-B). A mock renderer +
// session record what driveSplatCapture does; these assertions pin the return
// contract preview.js AND the headless CLI (PR-C) both rely on — the
// {noPipeline} sentinel vs bare-null cancel/zero-hit distinction (#365 honest-
// toast fix), the capK df64 decision reaching BOTH overrideCaptureOffset and
// the compiled feature signature, and the bare StreamResult shape (no frame,
// no stats.iters — callers hold those).
import { test } from "node:test";
import assert from "node:assert/strict";
import { driveSplatCapture } from "./capturedrive.js";
import { fibonacciDir } from "./splatcapture.js";
import { df64Eligible } from "./stability.js";
import { TOURBILLON } from "./oplist.js";
import { defaultColoring } from "./coloring.js";

const COL = defaultColoring();
const FRAME = { center: [0, 0, 0], diag: 3.46, radius: 1 };
const SCENE = { objects: [{ boxBase: true, ops: [] }] }; // df64-ineligible
const O = [0.1, 0.2, 0.3];

// One 1-hit view chunk; k-varied so views land in distinct cells.
const chunk = (k) => ({
  pos: new Float32Array([0.1 * k, 0.2, 0.3]),
  normal: new Float32Array([0, 0, 1]),
  albedo: new Float32Array([1, 0.5, 0]),
});

const mock = ({ session, pipeline = true } = {}) => {
  const calls = { offset: [], features: [], views: [], disposed: 0 };
  const ses = session ?? {
    captureView: (dir, frame, o, res, opts) => (
      calls.views.push({ dir, frame, o, res, opts }),
      chunk(calls.views.length)
    ),
    dispose: () => calls.disposed++,
  };
  return {
    calls,
    overrideCaptureOffset: (o, k) => calls.offset.push({ o, k }),
    createSplatCapture: async (features, res) => (
      calls.features.push({ features, res }),
      pipeline ? ses : null
    ),
  };
};
const OPTS = { iters: 24, frame: FRAME, O, views: 4, res: 64 };

test("renderer without createSplatCapture ⇒ null (unavailable, not sentinel)", async () => {
  assert.equal(await driveSplatCapture({}, SCENE, COL, OPTS), null);
});

test("pipeline build failure ⇒ {noPipeline} sentinel; offset already forced", async () => {
  const r = mock({ pipeline: false });
  const out = await driveSplatCapture(r, SCENE, COL, OPTS);
  assert.deepEqual(out, { noPipeline: true });
  assert.equal(r.calls.offset.length, 1);
  assert.deepEqual(r.calls.offset[0], { o: O, k: 0 }, "scene ⇒ capK 0");
  const { features, res } = r.calls.features[0];
  assert.equal(features.scene, true);
  assert.equal(features.df64, false, "capK 0 ⇒ f32 capture pipeline");
  assert.equal(res, 64);
});

test("view loop: per-view render → reduce; bare StreamResult; knobs threaded", async () => {
  const r = mock();
  const seen = [];
  const out = await driveSplatCapture(r, SCENE, COL, {
    ...OPTS,
    deScale: 0.4,
    layers: 3,
    aoStrength: 0.7,
    maxSteps: 500,
    onView: (k, n) => void seen.push([k, n]),
  });
  assert.equal(r.calls.views.length, 4, "one captureView per view");
  r.calls.views.forEach((v, k) => {
    assert.deepEqual(v.dir, fibonacciDir(k, 4), "fibonacci view directions");
    assert.equal(v.frame, FRAME);
    assert.equal(v.o, O);
    assert.equal(v.res, 64);
    assert.deepEqual(v.opts, {
      deScale: 0.4,
      layers: 3,
      aoStrength: 0.7,
      maxSteps: 500,
    });
  });
  assert.deepEqual(seen, [
    [1, 4],
    [2, 4],
    [3, 4],
    [4, 4],
  ]);
  assert.equal(r.calls.disposed, 1, "session disposed exactly once");
  assert.ok(out && out.points && out.stats, "finalized survivor cloud");
  assert.equal(out.stats.rawHits, 4, "all 4 hits reached the reduce");
  // BARE contract (plan review critical #1): callers hold frame/iters
  // themselves — the drive must not bolt them on.
  assert.ok(!("frame" in out), "no result.frame");
  assert.equal(out.stats.iters, undefined, "no result.stats.iters");
});

test("cancel (onView → false) ⇒ bare null + dispose, loop stops", async () => {
  const r = mock();
  const out = await driveSplatCapture(r, SCENE, COL, {
    ...OPTS,
    onView: () => false,
  });
  assert.equal(out, null);
  assert.equal(r.calls.views.length, 1, "stopped after the first view");
  assert.equal(r.calls.disposed, 1);
});

test("mid-capture device failure ⇒ dispose + rethrow (caller falls back)", async () => {
  const r = mock();
  const boom = new Error("device lost");
  r.createSplatCapture = async () => ({
    captureView: () => {
      throw boom;
    },
    dispose: () => r.calls.disposed++,
  });
  await assert.rejects(() => driveSplatCapture(r, SCENE, COL, OPTS), boom);
  assert.equal(r.calls.disposed, 1);
});

test("zero hits across all views ⇒ bare null (ran, found nothing)", async () => {
  const r = mock({
    session: {
      captureView: () => ({
        pos: new Float32Array(0),
        normal: new Float32Array(0),
        albedo: new Float32Array(0),
      }),
      dispose: () => {},
    },
  });
  assert.equal(await driveSplatCapture(r, SCENE, COL, OPTS), null);
});

test("df64 capture decision: deep capture geometry ⇒ capK > 0 AND df64 pipeline", async () => {
  assert.ok(df64Eligible(TOURBILLON), "premise: Tourbillon is df64-eligible");
  const r = mock();
  // Tiny frame radius ⇒ ortho pixel ≈ 3e-11 against |O| ≈ 1 ⇒ zoom mag ~3e10,
  // far past the f32 wall — kStarFor must demand df64 lead iterations.
  await driveSplatCapture(r, TOURBILLON, COL, {
    ...OPTS,
    O: [1, 0, 0],
    frame: { ...FRAME, radius: 1e-9 },
  });
  assert.ok(r.calls.offset[0].k > 0, "capK computed from capture geometry");
  assert.equal(r.calls.features[0].features.df64, true, "df64 twin compiled");
  // And the shallow default stays f32 for the same eligible formula.
  const r2 = mock();
  await driveSplatCapture(r2, TOURBILLON, COL, { ...OPTS, O: [1, 0, 0] });
  assert.equal(r2.calls.offset[0].k, 0);
  assert.equal(r2.calls.features[0].features.df64, false);
});

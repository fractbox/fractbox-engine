// #633 — the seam-clamped DE contract: proofs, not assertions (the spec
// review's blocker). Three layers of gate, all on SHIPPED code paths:
//
//   1. Op-level proofs on the CPU twin (cpuorbit applyOp — the 3-emitter
//      mirror's testable leg): per-branch operator-norm sampling for the
//      hinge fold, the fundamental-domain gate (the #632-review lesson:
//      Lipschitz sampling alone can pass a fold that never reaches its cell),
//      and the seam-term formulas derived in operators.js.
//   2. CONTRACT proofs with ground truth: 1-iteration formulas whose surface
//      preimage is computable by hand, showing (a) the unclamped DE really
//      does overestimate across a tear — the hazard is real, not asserted —
//      and (b) min(DE, seam) never does. Plus the review's required
//      scale-sandwich pin: seam divides by |w| AT THE OP, not the final w.
//   3. Codegen gates: the channel is structurally pay-per-use (the opAux/#125
//      discipline) — not one seam token in a seam-free variant, all of them
//      in a seam variant, and the march-step/membrane text exactly as the
//      contract states (step clamped, DE value untouched).
//
// Run: node --test core/seamclamp.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { applyOp } from "./cpuorbit.js";
import {
  byKey,
  W_UNCHANGED,
  hasSeamOps,
  seamGuarantee,
  isDeSound,
} from "./operators.js";
import { buildWGSL, usesSeam, exportGLSL } from "./shader.js";
import { buildFragGL, buildSceneFragGL } from "./shader_gl.js";
import { makeDE, renderAscii } from "./cpu.js";
import { PRESETS } from "./oplist.js";
import { TOUCHY } from "./vary.js";
import { validateOperators, validateFormula } from "./invariants.js";

const D2R = Math.PI / 180;
const run = (key, v, p, w = 1) => {
  const s = { x: p[0], y: p[1], z: p[2], w };
  applyOp(key, v, s);
  return s;
};
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);

function mulberry32(a) {
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// The hinge's signed cut-plane distance in the axis-2 (u,v) frame — the same
// derivation as operators.js id 67, kept independent here so a sign slip in
// the op body cannot self-certify.
function hingeSide(p, [foldDeg, cutDeg, axis, offset]) {
  const [u, v] =
    axis === 0 ? [p[1], p[2]] : axis === 2 ? [p[0], p[1]] : [p[0], p[2]];
  const mx = -Math.sin(cutDeg * D2R),
    my = Math.cos(cutDeg * D2R);
  return u * mx + v * my - offset;
}

test("#633 registry: hinge id 67, seam flags, w-rules, TOUCHY, palette invariants", () => {
  const hf = byKey("hingeFold");
  const mf = byKey("modFold");
  assert.equal(hf.id, 67, "hingeFold takes 67 (the id #632's 65-66 left free)");
  assert.equal(hf.wRule, W_UNCHANGED);
  assert.equal(hf.seam, true);
  assert.equal(hf.category, "fold");
  assert.equal(hf.params.length, 4, "Offset rides the opAux lane (op.p3)");
  assert.equal(mf.seam, true, "modFold is now a seam declarer (amended op)");
  assert.equal(mf.id, 17, "modFold keeps its id — amended, not re-minted");
  assert.ok(TOUCHY.has("hingeFold"), "absolute cut placement → short leash");
  // Palette invariants hold with the merged id tail (65-66 = #632's space
  // groups, 67 = this op; contiguous, so the plain check applies).
  const { failures } = validateOperators();
  assert.deepEqual(failures, []);
});

test("#633 hingeFold: per-branch sampled operator norm is exactly 1 (isometry proof)", () => {
  // Pairs restricted to the SAME branch (both sides of the cut checked
  // separately) must never expand — and rigid motions attain 1, so a norm
  // materially below 1 would also flag a broken map. Crossing pairs are the
  // seam channel's job, NOT w's — that split is the whole contract.
  const rand = mulberry32(0x633);
  const CASES = [
    [35, 0, 1, 1.1], // the shipped preset's params
    [90, 30, 2, 0.4],
    [-137, -60, 0, 0],
    [180, 10, 1, -0.7],
  ];
  for (const v of CASES) {
    let worst = 0;
    let attained = 0;
    let n = 0;
    for (let i = 0; i < 20000; i++) {
      const p = [rand() * 8 - 4, rand() * 8 - 4, rand() * 8 - 4];
      const eps = 1e-4;
      const dp = [rand() - 0.5, rand() - 0.5, rand() - 0.5];
      const dn = Math.hypot(...dp) || 1;
      const q = p.map((x, j) => x + (dp[j] / dn) * eps);
      // same-branch pairs only — the per-cell claim is what W_UNCHANGED makes
      if (Math.sign(hingeSide(p, v)) !== Math.sign(hingeSide(q, v))) continue;
      n++;
      const A = run("hingeFold", v, p);
      const B = run("hingeFold", v, q);
      const r = dist(A, B) / eps;
      worst = Math.max(worst, r);
      attained = Math.max(attained, r);
      assert.equal(A.w, 1, "W_UNCHANGED: the fold must not touch w");
    }
    assert.ok(n > 15000, "sampling actually covered both branches");
    assert.ok(worst <= 1 + 1e-9, `fold=${v}: same-branch norm ${worst} > 1`);
    assert.ok(
      attained >= 1 - 1e-6,
      `fold=${v}: norm never attained 1 — dead map?`,
    );
  }
});

test("#633 hingeFold: fundamental domain — the cut side really rotates by θ about the hinge line", () => {
  // The #632-review lesson: a Lipschitz gate alone can pass a fold that never
  // reaches its cell. Check the map's action directly: the rotating side
  // lands rotated by EXACTLY FoldAngle about the hinge line (distance to the
  // line preserved, angle advanced by θ, axis coordinate untouched); the
  // identity side does not move at all.
  const rand = mulberry32(0xbee5);
  const v = [73, 25, 1, 0.6]; // axis Y: in-plane coords (x, z)
  const phi = v[1] * D2R;
  const th = v[0] * D2R;
  const m = [-Math.sin(phi), Math.cos(phi)];
  const c = [m[0] * v[3], m[1] * v[3]]; // hinge point in (u, v)
  let rotated = 0,
    kept = 0;
  for (let i = 0; i < 20000; i++) {
    const p = [rand() * 6 - 3, rand() * 6 - 3, rand() * 6 - 3];
    const s = run("hingeFold", v, p);
    const side = hingeSide(p, v);
    if (side <= 0) {
      kept++;
      assert.deepEqual([s.x, s.y, s.z], p, "identity side untouched");
    } else {
      rotated++;
      assert.equal(s.y, p[1], "axis coordinate untouched");
      const du = p[0] - c[0],
        dv = p[2] - c[1];
      const su = s.x - c[0],
        sv = s.z - c[1];
      assert.ok(
        Math.abs(Math.hypot(su, sv) - Math.hypot(du, dv)) < 1e-9,
        "distance to the hinge line preserved",
      );
      const dAng = Math.atan2(sv, su) - Math.atan2(dv, du);
      const wrap = ((dAng - th + Math.PI * 3) % (Math.PI * 2)) - Math.PI;
      assert.ok(Math.abs(wrap) < 1e-9, `rotation is exactly θ (got Δ=${wrap})`);
      // and the seam report is exactly the cut-plane distance (w = 1 here)
      assert.ok(Math.abs(s.seam - Math.abs(side)) < 1e-12, "seam = |side|");
    }
  }
  assert.ok(rotated > 5000 && kept > 5000, "both cells reached");
});

test("#633 modFold: seam term = wall distance, derived from the wrap's own convention", () => {
  const rand = mulberry32(0x1701);
  for (const cells of [
    [2, 0, 0],
    [2, 1.5, 0],
    [0.8, 0.8, 0.8],
  ]) {
    for (let i = 0; i < 8000; i++) {
      const p = [rand() * 10 - 5, rand() * 10 - 5, rand() * 10 - 5];
      const s = run("modFold", cells, p);
      // ground truth: distance to the nearest wall lattice plane, per axis
      let want = Infinity;
      for (let a = 0; a < 3; a++) {
        const c = cells[a];
        if (!(c > 0)) continue;
        const x = p[a];
        // walls at c·(k + 1/2): distance = |x − c·(round(x/c − 1/2) + 1/2)|…
        // computed exhaustively from the two straddling walls.
        const k = Math.floor(x / c + 0.5);
        const lo = c * (k - 0.5),
          hi = c * (k + 0.5);
        want = Math.min(want, Math.abs(x - lo), Math.abs(x - hi));
      }
      assert.ok(
        Math.abs(s.seam - want) < 1e-9,
        `cells=${cells} p=${p}: seam ${s.seam} != wall dist ${want}`,
      );
    }
  }
  // all-axes-off: NO seam report (the op is inert, not a zero-width tear)
  const off = run("modFold", [0, 0, 0], [0.3, 9.9, -2]);
  assert.equal(off.seam, undefined, "inert modFold must not report a seam");
});

test("#633 contract (modFold ground truth): DE alone oversteps the wall; min(DE, seam) never does", () => {
  // 1-iteration formula with a hand-computable surface: modFold then
  // translate ⇒ zero set = the lattice {(bx + kx·cx, by + ky·cy, 0…)}. The
  // exact distance to that lattice is the ground truth the marched bound
  // must never exceed.
  const cells = [2, 1.5, 0];
  const b = [0.8, 0.55, 0];
  const formula = {
    name: "seam-gate",
    iters: 1,
    addC: false,
    deOption: 2,
    ops: [
      { key: "modFold", values: cells },
      { key: "translate", values: [-b[0], -b[1], -b[2]] },
    ],
  };
  const de = makeDE(formula);
  assert.equal(de.seamAware, true, "makeDE flags the seam channel");
  const trueDist = (p) => {
    let best = Infinity;
    for (let kx = -4; kx <= 4; kx++)
      for (let ky = -4; ky <= 4; ky++)
        best = Math.min(
          best,
          Math.hypot(
            p[0] - (b[0] + kx * cells[0]),
            p[1] - (b[1] + ky * cells[1]),
            p[2],
          ),
        );
    return best;
  };
  const rand = mulberry32(0x5ea);
  let overestimates = 0;
  for (let i = 0; i < 20000; i++) {
    const p = [rand() * 8 - 4, rand() * 8 - 4, rand() * 4 - 2];
    const d = de(p[0], p[1], p[2]);
    const seam = de.seam;
    const truth = trueDist(p);
    if (d > truth + 1e-9) overestimates++; // the unclamped hazard — expected!
    assert.ok(
      Math.min(d, seam) <= truth + 1e-9,
      `p=${p}: min(DE ${d}, seam ${seam}) > true ${truth}`,
    );
  }
  // Non-vacuous: the hazard the clamp fixes must actually occur in the sample
  // (bodies near walls read the in-cell copy while a neighbour copy is nearer).
  assert.ok(
    overestimates > 500,
    `unclamped DE never overestimated (${overestimates}) — gate is vacuous`,
  );
});

test("#633 contract (hingeFold ground truth): the tear is a real DE hazard; the seam clamp bounds it", () => {
  // Same construction for the hinge: 1 iteration of hingeFold + translate ⇒
  // surface preimage = {b if b is on the identity side} ∪ {R⁻¹(b−c)+c if that
  // lands on the rotating side}. Hand-computable, so the bound is checked
  // against TRUTH, not against the op's own map.
  const v = [90, 0, 2, 0.4]; // axis Z: (u,v) = (x,y); cut: y > 0.4 rotates 90°
  const b = [0.8, 0.1, 0.3];
  const th = v[0] * D2R;
  const c = [0, v[3]]; // φ=0 ⇒ m=(0,1), hinge point (0, offset) in (x,y)
  const cand = [];
  if (hingeSide(b, v) <= 0) cand.push(b);
  {
    const du = b[0] - c[0],
      dv = b[1] - c[1];
    const q = [
      c[0] + du * Math.cos(-th) - dv * Math.sin(-th),
      c[1] + du * Math.sin(-th) + dv * Math.cos(-th),
      b[2],
    ];
    if (hingeSide(q, v) > 0) cand.push(q);
  }
  assert.ok(cand.length > 0, "surface preimage exists — gate not vacuous");
  const trueDist = (p) =>
    Math.min(
      ...cand.map((q) => Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2])),
    );
  const formula = {
    name: "hinge-gate",
    iters: 1,
    addC: false,
    deOption: 2,
    ops: [
      { key: "hingeFold", values: v },
      { key: "translate", values: [-b[0], -b[1], -b[2]] },
    ],
  };
  const de = makeDE(formula);
  assert.equal(de.seamAware, true);
  const rand = mulberry32(0x416e);
  let overestimates = 0;
  for (let i = 0; i < 20000; i++) {
    const p = [rand() * 6 - 3, rand() * 6 - 3, rand() * 6 - 3];
    const d = de(p[0], p[1], p[2]);
    const truth = trueDist(p);
    if (d > truth + 1e-9) overestimates++;
    assert.ok(
      Math.min(d, de.seam) <= truth + 1e-9,
      `p=${p}: min(DE ${d}, seam ${de.seam}) > true ${truth}`,
    );
  }
  // The DERIVED claim, demonstrated: without its seam term the hinge's
  // W_UNCHANGED DE steps through the cut plane — the exact reason the seam
  // declaration is mandatory, measured rather than asserted.
  assert.ok(
    overestimates > 500,
    `unclamped hinge DE never overestimated (${overestimates}) — hazard claim unproven`,
  );
});

test("#633 contract: seam divides by |w| AT the op — the review's scale-sandwich pin", () => {
  // [scale 2, modFold, scale 3]: w at the op is 2, final w is 6. A seam
  // divided by the FINAL w (or none) gets this wrong in the loose direction —
  // exactly the invisible-in-a-screenshot failure the review flagged.
  const p = { x: 0.31, y: 0, z: 0, w: 1 };
  applyOp("scale", [2], p);
  applyOp("modFold", [2, 0, 0], p);
  const atOp = p.seam;
  applyOp("scale", [3], p);
  assert.equal(p.seam, atOp, "later scales must not rewrite a recorded seam");
  // wall distance at 2·0.31 = 0.62 in scaled space: 1 − 0.62 = 0.38; /w=2.
  assert.ok(Math.abs(atOp - 0.38 / 2) < 1e-12, `seam ${atOp} != 0.38/2`);
  // Order flipped: [modFold, scale 2] — w at the op is 1: no division.
  const q = { x: 0.31, y: 0, z: 0, w: 1 };
  applyOp("modFold", [2, 0, 0], q);
  applyOp("scale", [2], q);
  assert.ok(Math.abs(q.seam - (1 - 0.31)) < 1e-12, "pre-scale seam undivided");
  // And the sandwich bound is genuinely the smaller of the two conventions.
  assert.ok(atOp < 1 - 0.31, "sandwich produces the tighter bound");
});

test("#633 classifiers: hasSeamOps recursion + seamGuarantee tiers", () => {
  assert.equal(hasSeamOps({ ops: [{ key: "boxFold", values: [1] }] }), false);
  assert.equal(hasSeamOps({ ops: [{ key: "hingeFold", values: [] }] }), true);
  assert.equal(
    hasSeamOps({ ops: [{ key: "modFold", values: [], muted: true }] }),
    false,
    "muted seam ops don't arm the channel",
  );
  // hybrid slot B and scene objects must be seen (the isApproxDE shape)
  assert.equal(
    hasSeamOps({
      ops: [{ key: "boxFold", values: [1] }],
      hybrid: { b: { ops: [{ key: "modFold", values: [2, 0, 0] }] } },
    }),
    true,
  );
  assert.equal(
    hasSeamOps({
      ops: [],
      objects: [{ ops: [{ key: "hingeFold", values: [30, 0, 1, 0] }] }],
    }),
    true,
  );
  // guarantee tiers (see operators.js seamGuarantee doc)
  const f = (ks) => ({ ops: ks.map((key) => ({ key, values: [] })) });
  assert.equal(seamGuarantee(f(["modFold", "scale", "rotateXY"])), "exact");
  assert.equal(seamGuarantee(f(["modFold", "sphereFold"])), "standard");
  assert.equal(seamGuarantee(f(["modFold", "mandelbulb"])), "best-effort");
  assert.equal(seamGuarantee(f(["hingeFold", "polygonFold"])), "best-effort");
  // isDeSound is UNCHANGED by the seam flag: a seam formula still vouches
  // (marching soundness is delivered by the clamp, not by reclassification —
  // stated in the PR rather than smuggled into the classifier).
  assert.equal(isDeSound(f(["hingeFold", "scale"])), true);
});

test("#633 codegen: the seam channel is structurally pay-per-use (WGSL)", () => {
  const seamIds = [17, 67];
  // every preset WITHOUT a seam op emits not one seam token…
  let seamPresets = 0;
  for (const pr of PRESETS) {
    const ids = [...new Set((pr.ops || []).map((o) => byKey(o.key).id))];
    const armed = ids.some((id) => seamIds.includes(id));
    const w = buildWGSL({ ops: ids });
    if (armed) {
      seamPresets++;
      assert.match(w, /var<private> seamOut/, `${pr.name}: channel armed`);
    } else {
      assert.ok(
        !/seamOut|membraneT|g_seam/.test(w),
        `${pr.name}: seam token leaked into a seam-free variant`,
      );
      assert.match(
        w,
        /\n    t = t \+ d;\n/,
        `${pr.name}: march step unchanged`,
      );
    }
    assert.equal(usesSeam(ids), armed, "usesSeam mirrors the emission");
  }
  assert.ok(seamPresets >= 1, "the negative leg needs a shipped seam preset");
  // …and an armed variant carries the whole contract, textually:
  const w = buildWGSL({ ops: [byKey("hingeFold").id, byKey("scale").id] });
  assert.match(w, /var<private> seamOut : f32 = 1e30;/);
  assert.match(
    w,
    /t = t \+ select\(d, min\(d, seamOut\), seamOut >= 0\.25 \* d\);/,
    // BAND-GATED step (the v1.4 "not sharpening" field-regression fix): brake
    // to the seam only while the bound is ACTIONABLE (seamOut in [d/4, d) —
    // braking lands short of the tear). The old unconditional d/4 FLOOR below
    // that band bought no crossing guarantee (it stepped d/4 across the tear
    // regardless — the bound is degenerate there, fold-preimages are dense)
    // while quartering the whole march: measured on the two 2026-08 owner
    // repro links, 130-190 ms settles (accumCap 0 → idle refinement dead,
    // step budget exhausted into the softA wash) vs 72-82 ms band-gated with
    // refinement alive and <0.7% px off the unclamped reference. The eps-t
    // history still stands: an ABSOLUTE floor crawled (3759 steps vs 3 on
    // Hinged Bastion).
    "STEP braked by the seam only inside the actionable band",
  );
  assert.match(
    w,
    /if \(d < eps \* t\) \{ hit = true; break; \}/,
    "hit test UNclamped",
  );
  assert.match(w, /seamOut = 1e30; \/\/ #633/, "mapDE resets the accumulator");
  // membrane view (mode 4) rides the same gate, and only the gate:
  assert.match(w, /membraneT/, "membrane recording present when armed");
  assert.match(
    w,
    /G\.p3ctl\.z > 3\.5/,
    "miss-path membrane on the same uniform",
  );
  // #370's pin shape survives: exactly one heat return, inside the dbg branch
  assert.equal((w.match(/return vec4f\(s2l\(heatPalette/g) || []).length, 1);
  // capture/shadow/AO march is NOT clamped (their loops keep plain stepping)
  const cap = buildWGSL({ ops: [17], capture: true });
  assert.ok(
    !/max\(min\(dd, seamOut\)/.test(cap),
    "capture march must stay on the unclamped DE (review consumer table)",
  );
});

test("#633 fix: the browse GENERAL variant can drop the seam channel (seam:false)", () => {
  // The v1.4 "not sharpening" live regression: the general (browse) shader is
  // built with ops:null, and usesSeam(null) is true, so EVERY seam-free flat
  // formula browsed on it paid the clamped march (#125-class off-state tax).
  // The renderer now passes seam:false for seam-free content and routes seam
  // formulas to a seam-armed general by key (F_SEAM) — pin both emissions.
  const free = buildWGSL({ seam: false }); // ops:null — the full switch
  assert.ok(
    !/seamOut|membraneT/.test(free),
    "seam-free GENERAL emits not one seam token",
  );
  assert.match(free, /\n    t = t \+ d;\n/, "plain march step");
  // The seam-op CASES stay in the switch with their fold bodies (only the
  // seam REPORT is stripped): a mis-routed seam formula degrades to the
  // pre-#633 march — the v1.3 behavior — never to a silently dropped op.
  assert.match(free, /case 17u: \{/, "modFold case present");
  assert.match(free, /case 67u: \{/, "hingeFold case present");
  assert.match(
    free,
    /pos\.x = pos\.x - op\.p0 \* floor\(pos\.x \/ op\.p0 \+ 0\.5\)/,
    "modFold fold body intact",
  );
  // Default (no override) stays the conservative pre-fix emission: armed.
  const dflt = buildWGSL({});
  assert.match(dflt, /var<private> seamOut/, "ops:null default stays armed");
  // And the armed emission is the exact wgsl+wgslSeam concatenation — the
  // split refactor must reproduce the case bodies byte for byte.
  assert.match(
    dflt,
    /if \(mfs < 1e30\) \{ seamOut = min\(seamOut, max\(mfs, 0\.0\) \/ max\(abs\(w\), 1e-9\)\); \}/,
    "modFold seam report present when armed",
  );
});

test("#633 codegen: GL tier mirrors the gate (flat, hybrid slot B, scene)", () => {
  const free = buildFragGL([{ key: "boxFold", values: [1] }]);
  assert.ok(!free.includes("g_seam"), "seam-free GL program has no channel");
  const armed = buildFragGL([{ key: "modFold", values: [2, 2, 0] }]);
  assert.match(armed, /float g_seam;/);
  assert.match(armed, /g_seam = 1e30; \/\/ #633/);
  assert.match(
    armed,
    /t \+= \(g_seam >= 0\.25 \* d\) \? min\(d, g_seam\) : d;/,
    "GL march step mirrors the WGSL band-gate",
  );
  // a seam op hiding in hybrid slot B arms the shared march too
  const hyb = buildFragGL(
    [{ key: "boxFold", values: [1] }],
    [{ ops: [{ key: "hingeFold", values: [30, 0, 1, 0] }] }],
  );
  assert.match(hyb, /float g_seam;/);
  // scene: the object chain reports in LOCAL units; the walk rescales by uscale
  const scene = buildSceneFragGL([
    {
      ops: [{ key: "modFold", values: [2, 0, 0] }],
      iters: 6,
      transform: { origin: [0, 0, 0], uscale: 2, rot: [0, 0, 0] },
      combine: 0,
      blendK: 0,
    },
  ]);
  assert.match(scene, /g_seam = min\(sm0_0, g_seam \* uObjUscale\[0\]\);/);
  const sceneFree = buildSceneFragGL([
    {
      ops: [{ key: "boxFold", values: [1] }],
      iters: 6,
      transform: { origin: [0, 0, 0], uscale: 1, rot: [0, 0, 0] },
      combine: 0,
      blendK: 0,
    },
  ]);
  assert.ok(!sceneFree.includes("g_seam"));
});

test("#633 desktop export: no seam channel in the iterateJIT_ ABI (desktopGlsl twins)", () => {
  const f = {
    name: "hinge export",
    iters: 8,
    addC: true,
    deOption: 2,
    ops: [
      { key: "hingeFold", values: [30, 0, 1, 0.5] },
      { key: "modFold", values: [2, 0, 0] },
      { key: "scale", values: [2] },
    ],
  };
  const glsl = exportGLSL(f);
  assert.ok(
    !glsl.includes("g_seam"),
    "the desktop host declares no g_seam — exports must not reference it",
  );
  const { failures } = validateFormula(f);
  assert.deepEqual(failures, [], "hinge formula round-trips the export gate");
});

test("#633 CPU tier: scene seam rescales by uscale; seam marcher renders the preset", () => {
  // Scene: one object at uscale 2 containing a modFold — the world-space seam
  // must be 2× the local wall distance (distances scale like dk).
  const scene = {
    name: "seam scene",
    objects: [
      {
        ops: [{ key: "modFold", values: [2, 0, 0] }],
        iters: 1,
        transform: { origin: [0, 0, 0], uscale: 2, rot: [0, 0, 0] },
      },
    ],
  };
  const de = makeDE(scene);
  assert.equal(de.seamAware, true);
  de(0.5, 0.2, 0.1); // local x = 0.25 → wall dist 1 − 0.25 = 0.75 → ×2
  assert.ok(Math.abs(de.seam - 1.5) < 1e-9, `scene seam ${de.seam} != 1.5`);
  // And the shipped preset actually marches on the CPU tier (smoke: real
  // coverage, no NaN) — the seam clamp is in the grid marcher's hot loop.
  const preset = PRESETS.find((p) => p.name === "Hinged Bastion");
  assert.ok(preset, "Hinged Bastion ships");
  const art = renderAscii(preset, { cols: 48, rows: 24 });
  const filled = (art.match(/[^ .\n]/g) || []).length;
  assert.ok(filled > 100, `preset renders (filled=${filled})`);
});

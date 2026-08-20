// N-slot hybrid model — accessor pins + the refactored-reader N-slot pins.
// See docs/planning/HYBRID_NSLOT_SPEC.md (PR-1: inert model + accessor + codec).
//
// Named *.test.mjs so sync_web_core.sh skips it. Run:
//   node --test core/hybridmodel.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";
import {
  hybridSlots,
  activeHybridSlots,
  HYBRID_MAX_SLOTS,
  HYBRID_HW_SLOTS,
  packHyb,
  unpackHyb,
} from "./hybridmodel.js";
import { isApproxDE } from "./operators.js";
import { opsSupported } from "./evaluate.js";
import { jitterParams } from "./vary.js";
import { sanitizeHybrid } from "./sanitize.js";
import { parseHybrid, hybridSlotAt, makeOrbit } from "./cpuorbit.js";

// ── shared fixtures (all real operator keys) ──────────────────────────────────
const A = { key: "absFold", values: [] };
const SCALE = (k) => ({ key: "scale", values: [k] });
const BOX = { key: "boxFold", values: [1.0] };
const SPHERE = { key: "sphereFold", values: [0.5, 1.0] };
const PF = { key: "polygonFold", values: [6, 1, 0] }; // the deApprox op

// A well-formed N-slot STORED formula: slot A is the body; extras ride in
// hybrid.slots[]; schedule.counts is the FULL per-slot list (incl. A).
const nSlot = (extraSlots, counts, aOps = [A], aAddC = false) => ({
  ops: aOps,
  addC: aAddC,
  iters: 10,
  deOption: 2,
  hybrid: { slots: extraSlots, schedule: { counts } },
});

// ── accessor ──────────────────────────────────────────────────────────────────

test("HYBRID_MAX_SLOTS is the policy cap (4 = A + 3)", () => {
  assert.equal(HYBRID_MAX_SLOTS, 4);
});

test("hybridSlots: flat formula → one degenerate slot (period 1)", () => {
  const f = { ops: [A, BOX], addC: true };
  const v = hybridSlots(f);
  assert.equal(v.slots.length, 1);
  assert.deepEqual(v.slots[0].ops, [A, BOX]);
  assert.equal(v.slots[0].addC, true);
  assert.deepEqual(v.counts, [1]);
});

test("hybridSlots: legacy 2-slot shape → [A, B] + [a, b]", () => {
  const f = {
    ops: [A],
    addC: false,
    hybrid: { b: { ops: [BOX], addC: true }, schedule: { a: 2, b: 3 } },
  };
  const v = hybridSlots(f);
  assert.equal(v.slots.length, 2);
  assert.deepEqual(v.slots[0], { ops: [A], addC: false });
  assert.deepEqual(v.slots[1], { ops: [BOX], addC: true });
  assert.deepEqual(v.counts, [2, 3]);
});

test("hybridSlots: N-slot shape → [A, …extras] with A prepended", () => {
  const f = nSlot(
    [
      { ops: [BOX], addC: false },
      { ops: [SPHERE], addC: true },
    ],
    [1, 2, 3],
    [A],
    true,
  );
  const v = hybridSlots(f);
  assert.equal(v.slots.length, 3);
  assert.deepEqual(v.slots[0], { ops: [A], addC: true }); // slot A = body
  assert.deepEqual(v.slots[1], { ops: [BOX], addC: false });
  assert.deepEqual(v.slots[2], { ops: [SPHERE], addC: true });
  assert.deepEqual(v.counts, [1, 2, 3]);
});

test("hybridSlots: handles up to the 8-slot engineered ceiling generically", () => {
  const extras = Array.from({ length: 7 }, () => ({ ops: [BOX], addC: false }));
  const counts = [1, 2, 3, 4, 5, 6, 7, 8];
  const v = hybridSlots(nSlot(extras, counts));
  assert.equal(v.slots.length, 8);
  assert.deepEqual(v.counts, counts);
});

// ── per-slot mute (the eye toggle) ────────────────────────────────────────────

test("hybridSlots: legacy shape surfaces per-slot mute (aMuted / b.muted)", () => {
  const f = {
    ops: [A],
    addC: false,
    hybrid: {
      b: { ops: [BOX], addC: true, muted: true },
      schedule: { a: 1, b: 1 },
      aMuted: true,
    },
  };
  const v = hybridSlots(f);
  assert.equal(v.slots[0].muted, true); // slot A ← aMuted
  assert.equal(v.slots[1].muted, true); // slot B ← b.muted
});

test("hybridSlots: N-slot shape surfaces per-slot mute (aMuted / slots[i].muted)", () => {
  const f = nSlot(
    [
      { ops: [BOX], addC: false }, // B (unmuted)
      { ops: [SPHERE], addC: true, muted: true }, // C (muted)
    ],
    [1, 1, 1],
  );
  f.hybrid.aMuted = true; // slot A muted
  const v = hybridSlots(f);
  assert.equal(v.slots[0].muted, true); // A
  assert.equal(v.slots[1].muted, undefined); // B — emit-only-when-true (deepEqual pins hold)
  assert.equal(v.slots[2].muted, true); // C
  // Unmuted extras stay `{ ops, addC }` exactly (no `muted:false` leaking in).
  assert.deepEqual(v.slots[1], { ops: [BOX], addC: false });
});

test("activeHybridSlots: drops muted phases, keeps order + counts", () => {
  const f = nSlot(
    [
      { ops: [BOX], addC: false, muted: true }, // B muted
      { ops: [SPHERE], addC: true }, // C
    ],
    [2, 3, 4],
  );
  const v = activeHybridSlots(f);
  assert.equal(v.slots.length, 2); // A + C (B dropped)
  assert.deepEqual(v.slots[0], { ops: [A], addC: false }); // A
  assert.deepEqual(v.slots[1], { ops: [SPHERE], addC: true }); // C
  assert.deepEqual(v.counts, [2, 4]); // A's + C's counts (B's 3 dropped)
});

test("activeHybridSlots: every phase muted → empty schedule (renders nothing)", () => {
  const f = nSlot([{ ops: [BOX], addC: false, muted: true }], [1, 1]);
  f.hybrid.aMuted = true; // A muted too → all muted
  const v = activeHybridSlots(f);
  assert.deepEqual(v.slots, []);
  assert.deepEqual(v.counts, []);
});

// THE engine-equivalence pin (CPU tier): a 3-slot hybrid with slot B MUTED must
// produce byte-identical DE/orbit output to the equivalent 2-slot hybrid [A, C]
// — i.e. muting a phase truly removes it from the schedule, not just visually.
// All slots share the IFS family so muting can't shift the bailout policy.
test("CPU orbit: 3-slot with B muted === the 2-slot [A, C] schedule", () => {
  const B = { key: "boxFold", values: [1.0] };
  const C = { key: "scale", values: [1.6] };
  // 3-slot [A(2), B(3), C(4)] with B muted → active schedule [A(2), C(4)].
  const muted3 = nSlot(
    [
      { ops: [B], addC: false, muted: true },
      { ops: [C], addC: true },
    ],
    [2, 3, 4],
  );
  // The equivalent hybrid with B simply absent: 2-slot [A(2), C(4)].
  const twoAC = {
    ops: [A],
    addC: false,
    iters: 10,
    deOption: 2,
    hybrid: { b: { ops: [C], addC: true }, schedule: { a: 2, b: 4 } },
  };
  const o1 = makeOrbit(muted3);
  const o2 = makeOrbit(twoAC);
  assert.equal(o1.escape, o2.escape);
  const pts = [
    [0.3, -0.2, 0.5],
    [1.1, 0.7, -0.4],
    [-0.8, 0.05, 0.9],
    [0.02, 0.02, 0.02],
  ];
  for (const [x, y, z] of pts) {
    const a = o1(x, y, z);
    const b = o2(x, y, z);
    assert.equal(a.r, b.r, `r @ ${x},${y},${z}`);
    assert.equal(a.aw, b.aw, `aw @ ${x},${y},${z}`);
    assert.equal(a.escaped, b.escaped, `escaped @ ${x},${y},${z}`);
  }
});

test("CPU orbit: all phases muted === an empty flat formula (renders nothing)", () => {
  const B = { key: "boxFold", values: [1.0] };
  const allMuted = nSlot([{ ops: [B], addC: false, muted: true }], [1, 1]);
  allMuted.hybrid.aMuted = true; // A muted too
  const emptyFlat = { ops: [], addC: false, iters: 10, deOption: 2 };
  const o1 = makeOrbit(allMuted);
  const o2 = makeOrbit(emptyFlat);
  for (const [x, y, z] of [
    [0.3, -0.2, 0.5],
    [1.1, 0.7, -0.4],
  ]) {
    const a = o1(x, y, z);
    const b = o2(x, y, z);
    assert.equal(a.r, b.r);
    assert.equal(a.escaped, b.escaped);
  }
});

test("hybridSlots: short/missing counts floor at 1 per slot", () => {
  const v = hybridSlots(nSlot([{ ops: [BOX], addC: false }], [3]));
  assert.deepEqual(v.counts, [3, 1]); // A's count given, B's missing → 1
});

// ── refactored readers (§2.6 / §2.8): every slot, not just A/B ────────────────

test("isApproxDE detects a deApprox op hiding in slot C (SAFETY-critical)", () => {
  // A + B clean, slot C carries polygonFold → the marcher must still tighten.
  const withC = nSlot(
    [
      { ops: [BOX], addC: false },
      { ops: [PF], addC: false },
    ],
    [1, 1, 1],
  );
  assert.equal(isApproxDE(withC), true);
  // No deApprox op anywhere → false.
  const clean = nSlot(
    [
      { ops: [BOX], addC: false },
      { ops: [SPHERE], addC: false },
    ],
    [1, 1, 1],
  );
  assert.equal(isApproxDE(clean), false);
  // muted deApprox op in slot C does NOT count (scene-mute precedent).
  const mutedC = nSlot(
    [
      { ops: [BOX], addC: false },
      { ops: [{ ...PF, muted: true }], addC: false },
    ],
    [1, 1, 1],
  );
  assert.equal(isApproxDE(mutedC), false);
});

test("opsSupported fails on a bogus op in slot C (can't hide behind A/B)", () => {
  const good = nSlot(
    [
      { ops: [BOX], addC: false },
      { ops: [SPHERE], addC: false },
    ],
    [1, 1, 1],
  );
  assert.equal(opsSupported(good), true);
  const bogusC = nSlot(
    [
      { ops: [BOX], addC: false },
      { ops: [{ key: "__nope__", values: [] }], addC: false },
    ],
    [1, 1, 1],
  );
  assert.equal(opsSupported(bogusC), false);
});

test("jitterParams nudges EVERY slot's ops (whole formula or none, §2.8)", () => {
  const f = nSlot(
    [
      { ops: [BOX], addC: false },
      { ops: [SPHERE], addC: false },
    ],
    [1, 1, 1],
    [SCALE(2.0)],
  );
  const before = JSON.stringify(f);
  const out = jitterParams(f, { spread: 0.5, rng: () => 1 });
  // Input untouched (jitterParams clones).
  assert.equal(JSON.stringify(f), before);
  const v = hybridSlots(out);
  // Every slot changed — slot A (scale), slot B (boxFold), slot C (sphereFold).
  assert.notDeepEqual(v.slots[0].ops, [SCALE(2.0)]);
  assert.notDeepEqual(v.slots[1].ops, [BOX]);
  assert.notDeepEqual(v.slots[2].ops, [SPHERE]);
});

// ── sanitize (PR-2): ≥3 slots keep the slots[] shape; 2 slots reduce to legacy ──

function withWarnCapture(fn) {
  const orig = console.warn;
  const calls = [];
  console.warn = (...a) => calls.push(a.join(" "));
  try {
    const result = fn();
    return { result, warned: calls.length > 0, calls };
  } finally {
    console.warn = orig;
  }
}

test("sanitizeHybrid: a 3-slot formula KEEPS the N-slot slots[] shape (no truncation, no warn)", () => {
  const f = nSlot(
    [
      { ops: [BOX], addC: true },
      { ops: [SPHERE], addC: false },
    ],
    [1, 2, 3],
  );
  const { result, warned } = withWarnCapture(() => sanitizeHybrid(f));
  assert.equal(warned, false, "PR-2 lifts the truncation — no warning");
  // N-slot stored shape: slots[] present (2 extras: B, C), no legacy .b, counts kept.
  assert.equal(result.hybrid.b, undefined, "no legacy .b for a 3-slot hybrid");
  assert.equal(result.hybrid.slots.length, 2, "slot B + slot C survive");
  assert.equal(result.hybrid.slots[0].ops[0].key, "boxFold"); // slot B
  assert.equal(result.hybrid.slots[0].addC, true);
  assert.equal(result.hybrid.slots[1].ops[0].key, "sphereFold"); // slot C
  assert.deepEqual(result.hybrid.schedule.counts, [1, 2, 3]); // full per-slot list, incl. A
  // The accessor reads the round-tripped shape back to 3 slots.
  const view = hybridSlots(result);
  assert.equal(view.slots.length, 3);
  assert.deepEqual(view.counts, [1, 2, 3]);
});

test("sanitizeHybrid: >4 slots clamp to HYBRID_MAX_SLOTS and keep slots[]", () => {
  // 6 total slots (A + 5 extras) — past HYBRID_MAX_SLOTS (4).
  const extras = Array.from({ length: 5 }, () => ({ ops: [BOX], addC: false }));
  const f = nSlot(extras, [1, 1, 1, 1, 1, 1]);
  const { result, warned } = withWarnCapture(() => sanitizeHybrid(f));
  assert.equal(warned, false);
  // Clamped to A + 3 extras = 4 slots total.
  assert.equal(result.hybrid.slots.length, HYBRID_MAX_SLOTS - 1);
  assert.equal(result.hybrid.schedule.counts.length, HYBRID_MAX_SLOTS);
  // Total period ≤ 16 holds.
  assert.ok(result.hybrid.schedule.counts.reduce((n, c) => n + c, 0) <= 16);
});

test("sanitizeHybrid: a 2-slot new-shape import normalizes to legacy WITHOUT warning", () => {
  const f = nSlot([{ ops: [BOX], addC: true }], [2, 3]);
  const { result, warned } = withWarnCapture(() => sanitizeHybrid(f));
  assert.equal(warned, false, "2 slots is not a truncation — no warning");
  assert.ok(result.hybrid.b);
  assert.equal(result.hybrid.slots, undefined);
  assert.deepEqual(result.hybrid.schedule, { a: 2, b: 3 });
});

test("sanitizeHybrid: N-slot per-slot mute round-trips (aMuted + slots[i].muted)", () => {
  const f = nSlot(
    [
      { ops: [BOX], addC: true, muted: true }, // B muted
      { ops: [SPHERE], addC: false }, // C unmuted
    ],
    [1, 2, 3],
  );
  f.hybrid.aMuted = true; // A muted
  const result = sanitizeHybrid(f);
  assert.equal(result.hybrid.aMuted, true); // slot A mute survives
  assert.equal(result.hybrid.slots[0].muted, true); // B mute survives
  assert.equal(result.hybrid.slots[1].muted, undefined); // C stays unmuted (emit-when-true)
  // The accessor reads the mute back for every phase.
  const v = hybridSlots(result);
  assert.equal(v.slots[0].muted, true);
  assert.equal(v.slots[1].muted, true);
  assert.equal(v.slots[2].muted, undefined);
});

test("sanitizeHybrid: an unmuted N-slot hybrid is byte-identical to pre-mute (no flags)", () => {
  const f = nSlot(
    [
      { ops: [BOX], addC: true },
      { ops: [SPHERE], addC: false },
    ],
    [1, 2, 3],
  );
  const result = sanitizeHybrid(f);
  assert.equal("aMuted" in result.hybrid, false);
  assert.equal("muted" in result.hybrid.slots[0], false);
  assert.equal("muted" in result.hybrid.slots[1], false);
});

// ── PR-2: the `hyb` uniform pack/unpack + the CPU walk at the 8-slot ceiling ──
// packHyb/unpackHyb and the CPU schedule walk must handle slotCount=8 even though
// sanitize caps the PRODUCT at HYBRID_MAX_SLOTS=4 — the expansion foundation
// (§2.3, "built for 8 from v1") cannot silently rot.

test("packHyb/unpackHyb round-trips an 8-slot descriptor bit-exactly", () => {
  assert.equal(HYBRID_HW_SLOTS, 8);
  const d = {
    opCounts: [3, 5, 1, 7, 2, 8, 4, 6], // per-slot op counts (< 255)
    counts: [1, 2, 3, 4, 5, 6, 7, 8], // schedule counts, one nibble each
    addC: [true, false, true, true, false, true, false, true],
  };
  const packed = packHyb(d);
  const back = unpackHyb(packed);
  assert.equal(back.slotCount, 8);
  assert.deepEqual(back.opCounts, d.opCounts);
  assert.deepEqual(back.counts, d.counts);
  assert.deepEqual(back.addC, d.addC);
  // The packed words are unsigned 32-bit (a top-byte opCount / high addC bit must
  // not read negative through the Uint32Array store).
  for (const k of ["x", "y", "z", "w"])
    assert.ok(packed[k] >= 0 && packed[k] <= 0xffffffff, `${k} unsigned`);
  // Layout spot-checks against the spec: slotCount in z bits0-3, opCounts[4] in w.
  assert.equal(packed.z & 0xf, 8);
  assert.equal(packed.w & 0xff, 2);
});

test("packHyb matches the legacy 2-slot semantics via unpack", () => {
  // A 2-slot hybrid: slot A 3 ops / sched 2 / addC on; slot B 1 op / sched 1 / off.
  const packed = packHyb({
    opCounts: [3, 1],
    counts: [2, 1],
    addC: [true, false],
  });
  const back = unpackHyb(packed);
  assert.deepEqual(back, {
    slotCount: 2,
    opCounts: [3, 1],
    counts: [2, 1],
    addC: [true, false],
  });
});

test("parseHybrid + hybridSlotAt walk an 8-slot schedule (CPU foundation at the ceiling)", () => {
  // Build the N-slot STORED shape directly (sanitize caps at 4; the engine walk
  // must still handle 8). 7 extras + slot A = 8 slots, counts all 1 ⇒ period 8.
  const extras = Array.from({ length: 7 }, (_, i) => ({
    ops: [SCALE(i + 2)],
    addC: i % 2 === 0,
  }));
  const f = nSlot(extras, [1, 1, 1, 1, 1, 1, 1, 1], [BOX], true);
  const { slots, period } = parseHybrid(f);
  assert.equal(slots.length, 8);
  assert.equal(period, 8);
  // Slot A (index 0) fires at iteration 0, slot k at iteration k (period 8).
  assert.equal(hybridSlotAt(slots, period, 0).ops[0].key, "boxFold");
  assert.equal(hybridSlotAt(slots, period, 3).ops[0].key, "scale");
  // Wraps: iteration 8 is slot A again.
  assert.equal(hybridSlotAt(slots, period, 8), slots[0]);
  // addC follows the per-slot flag (A on; extras alternate on/off).
  assert.equal(hybridSlotAt(slots, period, 0).addC, true); // slot A
  assert.equal(hybridSlotAt(slots, period, 1).addC, true); // extra 0 (i=0 even)
  assert.equal(hybridSlotAt(slots, period, 2).addC, false); // extra 1 (i=1 odd)
});

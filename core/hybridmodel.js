// N-slot hybrid model — the ONE canonical reader for a hybrid formula's slots.
// See docs/planning/HYBRID_NSLOT_SPEC.md §2.1/§2.2.
//
// A hybrid formula interleaves several op-lists ("slots" A, B, C, …) on a
// repeating schedule. There are TWO stored shapes and this module is the single
// place that understands both:
//   • legacy 2-slot  `{ b: { addC, ops }, schedule: { a, b } }`  (byte-identical
//     old saves / shares — slot A is the formula body, slot B rides in `.b`);
//   • N-slot (≥3)     `{ slots: [{ addC, ops }…], schedule: { counts: […] } }`
//     — slot A is STILL the formula body; the EXTRA slots (B, C, …) ride in
//     `slots[]` and `schedule.counts` is the full per-slot count list (incl. A).
//
// `hybridSlots(f)` returns the canonical `{ slots, counts }` view for EITHER
// shape (slot A first, incl. A), so every reader — parseHybrid, sanitizeHybrid,
// hybridDeFamily/hybridLooseDE, isApproxDE, opsSupported, jitterParams, and the
// share codec — funnels through one implementation instead of hand-rolling slot
// logic. Handles up to the 8-slot engineered ceiling generically; the 4-slot
// PRODUCT cap is policy (HYBRID_MAX_SLOTS), enforced by sanitize + the Phases UI,
// NOT by this reader.
//
// Zero core imports on purpose: this sits at the bottom of the import graph so
// every hybrid reader above can depend on it without a cycle. Raw no-dep ESM.

// Product cap on total slots (A + 3). The engineered ceiling is 8 — the wire's
// u8 slotCount, this accessor's generic walk, and the (PR-2) uniform packing all
// handle 8. Raising the product cap later = flip this constant, extend the UI +
// tests, and pick a new period cap; no wire migration or shader change.
export const HYBRID_MAX_SLOTS = 4;

// Floor a raw schedule count at 1 (mirrors parseHybrid's `Math.max(1, … ?? 1)`).
// Kept lenient — sanitize applies the true 1..8 clamp; this only guards the loop.
const count1 = (v) => {
  const n = v ?? 1;
  return n >= 1 ? n : 1;
};

// Canonical slot view for a formula, across all three shapes (flat / legacy /
// N-slot). `slots[i] = { ops, addC }` with slot A at index 0; `counts[i]` is the
// schedule count for `slots[i]`. `ops` are the STORED op objects (references,
// muted flag intact) — callers filter muted / map to their own shape as needed.
//
// Per-slot mute (the eye toggle) rides `slots[i].muted` (emit-only-when-true, so
// an unmuted slot stays `{ ops, addC }` and the deep-equal pins hold). Slot A's
// mute lives on `hybrid.aMuted` in BOTH stored shapes; slot B rides `hybrid.b
// .muted` (legacy) or `hybrid.slots[0].muted` (N-slot); the rest ride their own
// `slots[i].muted`. This ACCESSOR only surfaces the flag — `activeHybridSlots`
// below is the ONE place the engines drop muted slots from the schedule.
export function hybridSlots(formula) {
  const slotA = { ops: formula?.ops || [], addC: !!formula?.addC };
  const h = formula?.hybrid;
  // Flat (non-hybrid): one degenerate slot — slot A every iteration (period 1).
  if (!h) return { slots: [slotA], counts: [1] };
  if (h.aMuted) slotA.muted = true;
  // N-slot (≥3) shape: the extra slots ride in `slots[]`; `schedule.counts` is
  // the full per-slot list including A at index 0.
  if (Array.isArray(h.slots)) {
    const slots = [
      slotA,
      ...h.slots.map((s) => {
        const slot = { ops: s?.ops || [], addC: !!s?.addC };
        if (s?.muted) slot.muted = true;
        return slot;
      }),
    ];
    const raw = Array.isArray(h.schedule?.counts) ? h.schedule.counts : [];
    return { slots, counts: slots.map((_, i) => count1(raw[i])) };
  }
  // Legacy 2-slot shape: slot B rides in `.b`; schedule is `{ a, b }`.
  const slotB = { ops: h.b?.ops || [], addC: !!h.b?.addC };
  if (h.b?.muted) slotB.muted = true;
  return {
    slots: [slotA, slotB],
    counts: [count1(h.schedule?.a), count1(h.schedule?.b)],
  };
}

// The engine SCHEDULE — every NON-muted slot, in order. THE one choke point the
// engines share so a muted phase is skipped identically in the WGSL walk, the GL
// codegen, the CPU orbit, and the capturesettle upload (each reads this instead
// of hand-rolling its own mute filter). Slot A being muted is handled naturally:
// it simply doesn't appear, and the next active slot becomes the schedule head.
// When EVERY slot is muted the schedule is empty — the caller renders nothing,
// mirroring the legacy both-slots-muted rule (app/src/formulair.ts engineView).
export function activeHybridSlots(formula) {
  const { slots, counts } = hybridSlots(formula);
  const outSlots = [];
  const outCounts = [];
  for (let i = 0; i < slots.length; i++) {
    if (slots[i].muted) continue;
    outSlots.push(slots[i]);
    outCounts.push(counts[i]);
  }
  return { slots: outSlots, counts: outCounts };
}

// ── The `hyb` uniform packing — ONE source of truth (spec §2.3) ───────────────
// The 4-word `G.hyb: vec4u` GPU uniform, laid out for the 8-slot ENGINEERED
// ceiling from day one so a later 4→8 product-cap bump never touches this pack,
// the WGSL walk (hybWalk), or the Globals layout again:
//   x = opCounts[0..3]  — 8 bits each (per-slot count ≪ MAX_OPS_WEBGPU=192)
//   w = opCounts[4..7]  — 8 bits each (zero until slots 5-8 exist)
//   y = schedule counts[0..7] — 4-bit nibbles (counts clamp 1..8 = one nibble)
//   z = slotCount (bits 0-3) | addC bits (bits 16-23, one per slot)
// packHyb/unpackHyb are the JS mirror of the WGSL hybWalk decode + the GL
// uniform apply; a unit pin exercises slotCount=8 even though sanitize caps at 4
// (HYBRID_MAX_SLOTS) so the expansion foundation cannot silently rot.
export const HYBRID_HW_SLOTS = 8; // engineered ceiling (product cap = HYBRID_MAX_SLOTS)

export function packHyb({ opCounts, counts, addC }) {
  let x = 0;
  let y = 0;
  let z = 0;
  let w = 0;
  const slotCount = opCounts.length;
  for (let s = 0; s < slotCount && s < HYBRID_HW_SLOTS; s++) {
    const oc = (opCounts[s] | 0) & 0xff;
    if (s < 4) x |= oc << (s * 8);
    else w |= oc << ((s - 4) * 8);
    y |= (count1(counts[s]) & 0xf) << (s * 4);
    if (addC?.[s]) z |= 1 << (16 + s);
  }
  z |= slotCount & 0xf;
  // >>> 0 keeps each an unsigned 32-bit value for the Uint32Array store (a byte
  // 7 in the top opCount slot, or an addC bit, would otherwise read negative).
  return { x: x >>> 0, y: y >>> 0, z: z >>> 0, w: w >>> 0 };
}

export function unpackHyb({ x, y, z, w }) {
  const slotCount = z & 0xf;
  const opCounts = [];
  const counts = [];
  const addC = [];
  for (let s = 0; s < slotCount && s < HYBRID_HW_SLOTS; s++) {
    const word = s < 4 ? x : w;
    opCounts.push((word >>> ((s & 3) * 8)) & 0xff);
    counts.push((y >>> (s * 4)) & 0xf);
    addC.push((z & (1 << (16 + s))) !== 0);
  }
  return { slotCount, opCounts, counts, addC };
}

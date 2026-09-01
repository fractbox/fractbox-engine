// Pins the WebGPU tier's frame-parameter HANDOFF — renderer.js's writeGlobals
// → deriveFrameParams — against the dropped-field bug.
//
// The failure this exists for is invisible by construction. writeGlobals packs
// `d.palettePhase`, `d.iridescence`, `d.sigLo`, `d.sigSpan` into the uniform
// words, but it used to build `d` from a hand-retyped subset of its argument
// bag, and those four names were never on the list. Every one of them read
// back as `undefined` → the `?? identity` fallbacks at the pack site →
// palette phase and iridescence rendered as no-ops on the flagship tier from
// the day they shipped (7494398, 2026-07-14), with nothing anywhere to notice:
// no error, no NaN, no diff in any test, just colour cycling that silently did
// nothing. The GL tier passes its whole payload (`deriveFrameParams(G)`) and
// was right all along, so the two tiers had quietly drifted on exactly the
// fields a comment there promises they cannot drift on.
//
// A GPU device is out of reach here (writeGlobals is a closure over one), so
// the invariant is pinned where it can be: the shape of the call in the source,
// plus the pure derivation's own behaviour on a realistic payload.
//
// Auto-levels (sigLo/sigSpan) was the fourth casualty and is revived in its own
// PR — see the last test.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { deriveFrameParams } from "./frameparams.js";

const src = readFileSync(new URL("./renderer.js", import.meta.url), "utf8");

// The payload capturesettle.js hands writeGlobals every frame, trimmed to the
// colour fields this test is about.
const payload = (over = {}) => ({
  colA: [0.86, 0.46, 0.18],
  colB: [0.18, 0.62, 0.74],
  bg: [0.07, 0.09, 0.15],
  colorMode: 1,
  stripeFreq: 5,
  iridescence: 0,
  palettePhase: 0,
  sigLo: 0,
  sigSpan: 1,
  palette: { on: false },
  light: {},
  deScale: 0.85,
  deOption: 2,
  tNear: 0.02,
  tFar: 80,
  ...over,
});

test("writeGlobals feeds deriveFrameParams the WHOLE payload, not a retyped subset", () => {
  const call = src.match(/const d = deriveFrameParams\(([\s\S]*?)\);/);
  assert.ok(
    call,
    "renderer.js must call deriveFrameParams once, in writeGlobals",
  );
  // Handing over the payload itself is the invariant: a field added to the
  // uniform words then arrives automatically. A hand-listed subset is what
  // broke, so an object literal of individual field names is exactly what must
  // not come back — `payload` bare, or `{ ...payload }` if a call site ever
  // genuinely needs to override one key, and nothing else.
  const arg = call[1].trim();
  assert.ok(
    arg === "payload" || /\.\.\.payload/.test(arg),
    `deriveFrameParams must receive the whole payload (got: ${arg.slice(0, 60)}) — ` +
      "a hand-copied field list silently drops anything not on it",
  );
});

test("the four uniform-word reads are all names deriveFrameParams knows", () => {
  // Every `d.<name>` the pack site reads must be a key the derivation actually
  // produces; otherwise it is another silent identity fallback.
  const produced = Object.keys(deriveFrameParams(payload()));
  for (const name of ["palettePhase", "iridescence", "sigLo", "sigSpan"]) {
    assert.ok(
      src.includes(`d.${name}`),
      `renderer.js should pack d.${name} into the uniform words`,
    );
    assert.ok(
      produced.includes(name),
      `deriveFrameParams must produce ${name} for the pack site to read`,
    );
  }
});

test("palette phase and iridescence survive the derivation (the two revived fields)", () => {
  const d = deriveFrameParams(
    payload({ palettePhase: 0.42, iridescence: 0.3 }),
  );
  assert.equal(d.palettePhase, 0.42);
  assert.equal(d.iridescence, 0.3);
});

test("a subset that omits them yields the identity — the bug, reproduced exactly", () => {
  // This is what the old call site did: name the fields it remembered, and
  // silently default the rest. Kept as an executable description of the bug.
  const full = payload({ palettePhase: 0.42, iridescence: 0.3 });
  const { palettePhase, iridescence, ...subset } = full;
  const d = deriveFrameParams(subset);
  assert.equal(d.palettePhase, 0, "the dropped field reads back as identity");
  assert.equal(d.iridescence, 0);
});

test("a look with everything at its defaults derives exactly as it always did", () => {
  // The blast radius, measured rather than asserted: with phase/iridescence at
  // 0 and auto-levels already identity (off, or a non-normalizable mode),
  // forwarding the whole payload derives EXACTLY what the old subset derived —
  // so a look that used none of the three renders the same pixels as before.
  const full = payload();
  const before = deriveFrameParams({
    colA: full.colA,
    colB: full.colB,
    bg: full.bg,
    juliaC: full.juliaC,
    addC: full.addC,
    julia: full.julia,
    colorMode: full.colorMode,
    stripeFreq: full.stripeFreq,
    deScale: full.deScale,
    deOption: full.deOption,
    tNear: full.tNear,
    tFar: full.tFar,
    palette: full.palette,
    light: full.light,
  });
  assert.deepEqual(deriveFrameParams(full), before);
});

test("auto-levels reaches the uniform — the tier no longer discards it", () => {
  // The revival: sigLo/sigSpan were dropped by the same bug, leaving this tier
  // the only one rendering the raw, un-normalized signal while the WebGL2 tier
  // applied the identical signalRange() all along. Nothing may pin them back to
  // identity at the call site.
  const d = deriveFrameParams(payload({ sigLo: 0.21, sigSpan: 0.44 }));
  assert.equal(d.sigLo, 0.21);
  assert.equal(d.sigSpan, 0.44);
  const call = src.match(/const d = deriveFrameParams\(([\s\S]*?)\);/);
  assert.ok(call, "the call must forward the payload");
  assert.doesNotMatch(
    call[1],
    /sigLo|sigSpan/,
    "sigLo/sigSpan must not be overridden at the call site — the range capturesettle computed is the one to render",
  );
});

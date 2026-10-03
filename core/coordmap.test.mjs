// Guards for the toCoord/fromCoord pair (Phase C) — lane conventions pinned
// against the MB3D _to*/_inv* bodies, round-trip cancellation per system, the
// NaN-free from-torical rewrite, and the deApprox classification.
//
// Named *.test.mjs so it stays out of the apps' served *.js surface. Run: node --test core/coordmap.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";
import { applyOp } from "./cpuorbit.js";
import { isApproxDE } from "./operators.js";

const run = (key, values, pt) => {
  const s = { ...pt };
  applyOp(key, values, s);
  return s;
};
const P = { x: 0.5, y: 0.4, z: 0.7, w: 1.3 };
const close = (a, b, eps = 1e-12) => Math.abs(a - b) < eps;

test("toCoord lane conventions match the MB3D bodies", () => {
  const cyl = run("toCoord", [0, 2, 0], P);
  assert.ok(
    close(cyl.x, Math.hypot(P.x, P.y)) &&
      close(cyl.y, Math.atan2(P.y, P.x)) &&
      cyl.z === P.z,
  );
  const sph = run("toCoord", [1, 2, 0], P);
  assert.ok(close(sph.x, Math.hypot(P.x, P.y, P.z)));
  assert.ok(
    close(sph.y, Math.atan2(Math.hypot(P.x, P.y), P.z)),
    "polar from +z in lane y",
  );
  assert.ok(close(sph.z, Math.atan2(P.y, P.x)), "azimuth in lane z");
  const tor = run("toCoord", [2, 2, 0], P);
  assert.ok(close(tor.x, 2 - Math.hypot(P.x, P.y)), "R − rho in lane x");
});

test("round-trips cancel: cyl and sph exactly, tor in the principal domain", () => {
  for (const sys of [0, 1, 2]) {
    const rt = run("fromCoord", [sys, 2, 0], run("toCoord", [sys, 2, 0], P));
    for (const k of ["x", "y", "z"])
      assert.ok(
        close(rt[k], P[k], 1e-9),
        `sys ${sys} ${k}: ${rt[k]} vs ${P[k]}`,
      );
  }
  // system 3 (torical2) round-trips with a z-mirror — the source pair's own
  // behavior (sin(α−π/2) = −cos α), pinned here so nobody "fixes" it blind.
  const rt3 = run("fromCoord", [3, 2, 0.3], run("toCoord", [3, 2, 0.3], P));
  assert.ok(
    close(rt3.x, P.x, 1e-9) &&
      close(rt3.y, P.y, 1e-9) &&
      close(rt3.z, -P.z, 1e-9),
  );
});

test("from-torical is NaN-free at |cos y| = 0 (the q·tan rewrite)", () => {
  const s = run("fromCoord", [2, 2, 0], {
    x: 1.2,
    y: Math.PI / 2,
    z: 0.5,
    w: 1,
  });
  for (const k of ["x", "y", "z"])
    assert.ok(Number.isFinite(s[k]), `${k} = ${s[k]}`);
  assert.ok(close(s.x, 0, 1e-9), "|R−x|·|cos(π/2)| = 0");
  assert.ok(
    close(s.y, Math.abs(2 - 1.2) * 1, 1e-9),
    "|R−x|·sin(π/2)·sign(cos)≈+0 side",
  );
});

test("both ops are deApprox-tagged and leave w untouched", () => {
  assert.equal(
    isApproxDE({ ops: [{ key: "toCoord", values: [0, 2, 0] }] }),
    true,
  );
  assert.equal(
    isApproxDE({ ops: [{ key: "fromCoord", values: [0, 2, 0] }] }),
    true,
  );
  assert.equal(run("toCoord", [1, 2, 0], P).w, P.w);
  assert.equal(run("fromCoord", [3, 2, 0.5], P).w, P.w);
});

// ── System 4: log-polar (parity wave 1) ────────────────────────────────────
// The corpus pre-step pass named "log-spiral unwrap ×2" (transLogSpIFS,
// translogsp4ifs) and ruled that toCoord had no log-polar mode. System 4 is
// the textbook complex logarithm written in the existing lane layout.

test("log-polar is the complex log: x = ln ρ, y = θ (Γ = 0)", () => {
  const lp = run("toCoord", [4, 2, 0], P);
  assert.ok(close(lp.x, Math.log(Math.hypot(P.x, P.y))), "lane x = ln ρ");
  assert.ok(close(lp.y, Math.atan2(P.y, P.x)), "lane y = θ");
  assert.equal(lp.z, P.z, "z lane passes through");
});

test("log-polar round-trips exactly, for every Γ (the shear is inverted)", () => {
  for (const g of [0, 0.7, -1.3, 3.14]) {
    const rt = run("fromCoord", [4, 2, g], run("toCoord", [4, 2, g], P));
    for (const k of ["x", "y", "z"])
      assert.ok(close(rt[k], P[k], 1e-9), `Γ=${g} ${k}: ${rt[k]} vs ${P[k]}`);
  }
});

test("Γ straightens the logarithmic spiral ρ = e^(θ/Γ) into a line", () => {
  // THE point of the mode: on a log spiral of pitch Γ the angle lane becomes
  // constant, so a modFold after it tiles the spiral into equal cells.
  const G = 2.0; // spiral ρ = e^(θ/G)
  const ys = [0.2, 0.9, 1.7, 2.6].map((th) => {
    const rho = Math.exp(th / G);
    return run("toCoord", [4, 2, G], {
      x: rho * Math.cos(th),
      y: rho * Math.sin(th),
      z: 0,
      w: 1,
    }).y;
  });
  for (const y of ys)
    assert.ok(close(y, ys[0], 1e-9), `angle lane not constant: ${ys}`);
});

test("log-polar is finite at the origin and saturates instead of overflowing", () => {
  const o = run("toCoord", [4, 2, 0], { x: 0, y: 0, z: 0.3, w: 1 });
  assert.ok(Number.isFinite(o.x) && o.x < -20, `ln ρ floored, got ${o.x}`);
  // fromCoord's exponent cap must match across all three tiers — pin it.
  const e = run("fromCoord", [4, 2, 0], { x: 500, y: 0, z: 0, w: 1 });
  assert.ok(Number.isFinite(e.x) && Number.isFinite(e.y), "exp cap holds");
  assert.ok(close(e.x, Math.exp(60), 1e-9 * Math.exp(60)), "capped at e^60");
});

test("systems 0–3 unmoved by the System-4 add (no fallthrough drift)", () => {
  // Adding an arm before the terminal `else` must not disturb the existing
  // lanes, and out-of-range values must still land on the cylinder default.
  assert.deepEqual(run("toCoord", [9, 2, 0], P), run("toCoord", [0, 2, 0], P));
  assert.deepEqual(
    run("fromCoord", [9, 2, 0], P),
    run("fromCoord", [0, 2, 0], P),
  );
  assert.deepEqual(run("toCoord", [-1, 2, 0], P), run("toCoord", [0, 2, 0], P));
});

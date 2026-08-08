// Guards for gnarl2D/gnarl3D (Phase C) — the nested-sine closed form (from the
// published construction; corpus bodies are #84-tainted and are NOT the
// oracle), the lane couplings, Step=0 identity, and the deApprox contract.
//
// Named *.test.mjs so sync_web_core.sh skips it. Run: node --test core/gnarl.test.mjs
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
const g = (b, ga, gb) => Math.sin(Math.sin((Math.sin(b * gb) + b) * ga) + b);
const close = (a, b) => Math.abs(a - b) < 1e-12;

test("gnarl2D: x↔y cross-coupled displacement from OLD components, z free", () => {
  const [gs, ga, gb] = [0.1, 3, 3];
  const s = run("gnarl2D", [gs, ga, gb], P);
  assert.ok(close(s.x, P.x - gs * g(P.y, ga, gb)));
  assert.ok(close(s.y, P.y - gs * g(P.x, ga, gb)), "y must use OLD x");
  assert.equal(s.z, P.z);
});

test("gnarl3D: cyclic coupling x←g(z), y←g(x), z←g(y), all from OLD components", () => {
  const [gs, ga, gb] = [0.15, 2.5, 4];
  const s = run("gnarl3D", [gs, ga, gb], P);
  assert.ok(close(s.x, P.x - gs * g(P.z, ga, gb)));
  assert.ok(close(s.y, P.y - gs * g(P.x, ga, gb)));
  assert.ok(close(s.z, P.z - gs * g(P.y, ga, gb)));
});

test("Step = 0 is the identity; both ops are deApprox and leave w untouched", () => {
  for (const key of ["gnarl2D", "gnarl3D"]) {
    const id = run(key, [0, 3, 3], P);
    for (const k of ["x", "y", "z", "w"]) assert.equal(id[k], P[k]);
    assert.equal(isApproxDE({ ops: [{ key, values: [0.1, 3, 3] }] }), true);
    assert.equal(run(key, [0.2, 3, 3], P).w, P.w);
  }
});

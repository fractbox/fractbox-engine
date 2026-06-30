# Changelog

All notable changes to the **Fractbox Engine** are documented here. The engine
is versioned [semver](https://semver.org)-style; the canonical version lives in
[`core/version.js`](core/version.js).

- **MAJOR** — breaking change to the op-list JSON, operator keys, or public API
- **MINOR** — new operators or render backends, backward-compatible
- **PATCH** — render-correctness / bug fixes, no API change

This is a one-way mirror; entries track the engine surface (`core/` + demo), not
the private app that builds on it.

## 0.2.0 — 2026-06-29

### Added

- **Seven new operators** (`core/operators.js`), all backward-compatible:
  - `menger` — smoothed Menger fold with signed-smoothness modes (rounded/organic edges).
  - `polyAngleFold` — N-fold polar angle fold (symmetry / angle / mirror).
  - `cylinderFold` — radius-bounded cylindrical fold (DE-tracked, `w ×k`).
  - `radialInvert` — spherical inversion about a shiftable center.
  - `bulbAxis` — Mandelbulb power around a selectable axis.
  - `hexFold` — hexagonal plane fold.
  - `absXYZ` — per-axis absolute-value fold (independent X/Y/Z toggles).
- **WebGL2 renderer backend** (`renderer_gl.js` + `shader_gl.js`) — a full-parity
  fallback below WebGPU, so the engine renders where WebGPU is unavailable.
- **CPU / colored-ASCII renderer** (`cpu.js`) — a GPU-free last-resort backend
  (also opt-in for testing), aligned to the GPU camera/aspect.
- `core/version.js` — the engine version constant (this file's source of truth).

### Changed

- **BREAKING:** the `roundMenger` operator key was renamed to **`menger`**. Op-lists
  that serialized `roundMenger` must be updated to `menger`. (Pre-1.0; called out
  rather than forcing a major.)
- Operator count is now **32** (was 25).

### Fixed

- **Color correctness:** albedo is now linearized from sRGB before lighting, so the
  render matches the GUI swatches. ([#6])
- Octahedral fold ships a working **Octahedron** preset; documented that `octaFold`
  needs a following Scale + Translate to render. ([#7])
- `evaluate.js` now gates the `+c` term on `addC || julia`, matching the renderer.

## 0.1.0 — 2026-06-26

Initial public release of the mirror: the operator-IR engine (`core/`), the
standalone WebGPU demo, the guided tour, and the tutorial. 25 operators.

[#6]: https://github.com/fractbox/fractbox-engine/issues/6
[#7]: https://github.com/fractbox/fractbox-engine/issues/7

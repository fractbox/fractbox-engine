// Guided tour — a step-by-step build-up that teaches the op-list model live in
// the demo. Each step hands the engine a complete, rendering formula and shows
// a card explaining what just changed. Pure glue over the same preview API the
// rest of the demo uses; no engine internals are touched.

// Small op-list helpers so the steps read like the recipes they are.
const box = (f) => ({ key: 'boxFold', values: [f] });
const sphere = (min, fix) => ({ key: 'sphereFold', values: [min, fix] });
const scale = (s) => ({ key: 'scale', values: [s] });
const rotXY = (d) => ({ key: 'rotateXY', values: [d] });
const rotYZ = (d) => ({ key: 'rotateYZ', values: [d] });
const kaleido = (n, twist) => ({ key: 'kaleido', values: [n, twist] });
const bulb = (p) => ({ key: 'mandelbulbPower', values: [p] });

const BOX_CAM = { yawDeg: 35, pitchDeg: 22, dist: 24, fovDeg: 42 };

// Every step's formula renders on its own — we build up by ADDING moves to a
// shape that's already alive, so there are no blank intermediate frames.
const STEPS = [
  {
    title: 'A formula is a list of moves',
    body: 'Each point in space is run through an ordered list of operators, iterated several times. The competition between the moves carves out a fractal surface. Here are the three that make the classic Mandelbox.',
    formula: {
      name: 'Mandelbox',
      addC: true,
      iters: 12,
      deOption: 2,
      ops: [box(1.0), sphere(0.5, 1.0), scale(2.0)],
      camera: BOX_CAM,
    },
  },
  {
    title: 'Move 1+2+3 · fold, bound, scale',
    body: 'Box fold reflects stray coordinates back to center (a volume-preserving mirror). Sphere fold bounds the radius so nothing escapes. Scale ×2 expands what’s left. Iterate that and the shell appears.',
    formula: {
      name: 'Mandelbox',
      addC: true,
      iters: 12,
      deOption: 2,
      ops: [box(1.0), sphere(0.5, 1.0), scale(2.0)],
      camera: BOX_CAM,
    },
  },
  {
    title: 'Add a twist between iterations',
    body: 'Insert a small rotation each iteration and the whole structure winds into a screw. Box + sphere + scale + two rotations is the “Tourbillon”. Operator order is the formula — moving the rotation changes everything.',
    formula: {
      name: 'Tourbillon',
      addC: true,
      iters: 12,
      deOption: 2,
      ops: [box(1.0), sphere(0.5, 1.0), scale(2.0), rotXY(14), rotYZ(7)],
      camera: BOX_CAM,
    },
  },
  {
    title: 'Add mirror symmetry',
    body: 'A kaleidoscope fold imposes N-fold symmetry on the box core. Here: 5-fold, with a YZ tilt. Same three base moves — restyled by what you stack around them.',
    formula: {
      name: 'Star Box',
      addC: true,
      iters: 11,
      deOption: 2,
      ops: [box(1.0), sphere(0.5, 1.0), scale(2.0), kaleido(5, 0), rotYZ(30)],
      camera: { yawDeg: 40, pitchDeg: 18, dist: 24, fovDeg: 42 },
    },
  },
  {
    title: 'A different family',
    body: 'Not everything is a fold. The Mandelbulb is a single power operator (z → z⁸ + c) using the escape-time distance estimate instead of the analytic one. One operator, a whole different look.',
    formula: {
      name: 'Mandelbulb',
      addC: true,
      iters: 8,
      deOption: 0,
      ops: [bulb(8.0)],
      camera: { yawDeg: 35, pitchDeg: 12, dist: 5.0, fovDeg: 42 },
    },
  },
  {
    title: 'Your turn',
    body: 'That’s the whole idea: order primitives into a list, iterate, render. Pick any preset below to explore, read TUTORIAL.md to build your own in code, or open fractbox.com for a full visual editor.',
    formula: null, // keep whatever's on screen
    last: true,
  },
];

// Wire up the launch button and build the card on demand. `apply(formula)` is
// the demo's setter (frameTo + setFormula); `onDone` restores normal browsing.
export function initTour(launchBtn, apply, onDone) {
  let card = null;
  let i = 0;

  function build() {
    card = document.createElement('div');
    card.id = 'tour';
    card.innerHTML = `
      <div class="tour-dots"></div>
      <h3 class="tour-title"></h3>
      <p class="tour-body"></p>
      <div class="tour-nav">
        <button type="button" class="tour-skip">Skip</button>
        <span class="tour-spacer"></span>
        <button type="button" class="tour-back">Back</button>
        <button type="button" class="tour-next"></button>
      </div>`;
    document.body.appendChild(card);
    card.querySelector('.tour-skip').addEventListener('click', end);
    card.querySelector('.tour-back').addEventListener('click', () => go(i - 1));
    card.querySelector('.tour-next').addEventListener('click', () => {
      if (STEPS[i].last) end();
      else go(i + 1);
    });
    const dots = card.querySelector('.tour-dots');
    STEPS.forEach((_, k) => {
      const d = document.createElement('button');
      d.type = 'button';
      d.className = 'tour-dot';
      d.addEventListener('click', () => go(k));
      dots.appendChild(d);
    });
  }

  function go(n) {
    i = Math.max(0, Math.min(STEPS.length - 1, n));
    const step = STEPS[i];
    if (step.formula) apply(step.formula);
    card.querySelector('.tour-title').textContent = step.title;
    card.querySelector('.tour-body').textContent = step.body;
    card.querySelector('.tour-back').disabled = i === 0;
    card.querySelector('.tour-next').textContent = step.last ? 'Done' : 'Next →';
    card.querySelectorAll('.tour-dot').forEach((d, k) =>
      d.setAttribute('aria-current', k === i ? 'true' : 'false'),
    );
  }

  function start() {
    if (!card) build();
    card.hidden = false;
    launchBtn.hidden = true;
    go(0);
  }

  function end() {
    if (card) card.hidden = true;
    launchBtn.hidden = false;
    onDone?.();
  }

  launchBtn.addEventListener('click', start);
  addEventListener('keydown', (e) => {
    if (card && !card.hidden) {
      if (e.key === 'Escape') end();
      else if (e.key === 'ArrowRight' && !STEPS[i].last) go(i + 1);
      else if (e.key === 'ArrowLeft') go(i - 1);
    }
  });
}

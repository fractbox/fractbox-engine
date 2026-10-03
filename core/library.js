// Named-formula library — pure localStorage CRUD (no DOM). Both apps wire their
// own name input / dropdown to these. Values are op-list JSON objects.

// Parsing the library string is the gallery's single largest synchronous cost.
// galleryItems() calls loadLibrary 2 + 2N times per open (mineFormulas twice,
// then mineColoring + mineMeta once EACH per saved formula), and every call
// re-parsed the WHOLE string — so the work is quadratic in the library size.
// Measured on an M4, before a single card is built: 13 ms at 50 saves, 55 ms at
// 100, 223 ms at 200. A tablet is several times slower, and the click is blocked
// for all of it.
//
// Memoize the parse against the RAW STRING: getItem is cheap, JSON.parse is not,
// and comparing the raw string self-invalidates on any write — including one
// from another tab, which a timestamp or a dirty flag would miss.
//
// The cached object is NEVER handed out. Every call returns a fresh shallow
// copy, so the top-level add/delete the mutators below perform (and the same
// pattern in formula-creator / formula-blocks) can never reach into the cache.
//
// ENTRY objects are shared by reference, so an entry you get from here is
// READ-ONLY: copy it (`{ ...entry }`) before changing anything. Every caller
// already does — checked across app/, formula-creator/ and formula-blocks/ —
// and saveLibrary drops the memo rather than priming it, so the copy a save
// hands over can never become the cache. Keep both halves true.
let memoKey = null,
  memoRaw = null,
  memoLib = null;

export function loadLibrary(key) {
  try {
    const raw = localStorage.getItem(key);
    if (memoLib && key === memoKey && raw === memoRaw) return { ...memoLib };
    const lib = JSON.parse(raw) || {};
    memoKey = key;
    memoRaw = raw;
    memoLib = lib;
    return { ...lib };
  } catch {
    return {};
  }
}

export function saveLibrary(key, lib) {
  try {
    localStorage.setItem(key, JSON.stringify(lib));
    // INVALIDATE, never prime. Priming from `lib` would seat the CALLER'S own
    // entry objects in the memo, and a save is exactly where a caller still
    // holds them: `putFormula(key, name, formulaJSON)` stores the live formula,
    // and an edit to it afterwards would then reach into the cache and rewrite
    // history. Storage is a snapshot — gallery.test.ts pins that, and pinned it
    // by failing when this primed. The next read re-parses once, which is a
    // per-save cost, not a per-open one.
    memoKey = memoRaw = memoLib = null;
    return true;
  } catch {
    return false;
  }
}

export const libraryNames = (key) => Object.keys(loadLibrary(key)).sort();

export function putFormula(key, name, formulaJSON) {
  const lib = loadLibrary(key);
  lib[name] = formulaJSON;
  return saveLibrary(key, lib) ? lib : null;
}

export function deleteFormula(key, name) {
  const lib = loadLibrary(key);
  delete lib[name];
  saveLibrary(key, lib);
  return lib;
}

// Rename in ONE transaction (load → move key → save once), so a crash between
// two writes can't leave both or neither name behind. Refuses to clobber an
// existing entry under `to` — collision policy is the caller's (#763 §6.5).
// `updatedAt` (ISO string) is stamped when given.
export function renameFormula(key, from, to, updatedAt) {
  const lib = loadLibrary(key);
  if (!(from in lib) || from === to || to in lib) return null;
  const entry = { ...lib[from], name: to };
  if (updatedAt) entry.updatedAt = updatedAt;
  delete lib[from];
  lib[to] = entry;
  return saveLibrary(key, lib) ? lib : null;
}

// Named-formula library — pure localStorage CRUD (no DOM). Both apps wire their
// own name input / dropdown to these. Values are op-list JSON objects.

export function loadLibrary(key) {
  try { return JSON.parse(localStorage.getItem(key)) || {}; }
  catch { return {}; }
}

export function saveLibrary(key, lib) {
  try { localStorage.setItem(key, JSON.stringify(lib)); return true; }
  catch { return false; }
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

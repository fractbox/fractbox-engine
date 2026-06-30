// Fractbox Engine — version (semver). Single source of truth.
//
// Bump this on every engine release and add a matching CHANGELOG.md entry
// (publish/engine/CHANGELOG.md). publish_engine.sh reads this constant to
// stamp the mirror's release commit + tag, and gates that the changelog's
// top entry matches.
//
//   MAJOR — breaking change to the op-list JSON / operator keys / public API
//   MINOR — new operators or backends, backward-compatible
//   PATCH — render-correctness / bug fixes, no API change
export const ENGINE_VERSION = '0.2.0';

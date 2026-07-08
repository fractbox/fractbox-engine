# Security Policy

Fractbox Engine is a client-side rendering library — it runs entirely in the
browser, has no network calls, no server, and no dependencies. The realistic
surface is small, but reports are welcome.

## Reporting

Please report suspected vulnerabilities privately via GitHub's
**[Report a vulnerability](https://github.com/fractbox/fractbox-engine/security/advisories/new)**
(Security → Advisories) rather than a public issue.

Useful things to include: the affected file, a minimal reproduction, the browser
and GPU involved, and the impact you observed.

## Scope notes

- The engine generates WGSL/GLSL shader source from a formula op-list. Formula data
  is treated as untrusted structure but produces only shader code, not arbitrary
  host execution. Reports about shader-generation edge cases (malformed op-lists,
  resource exhaustion) are in scope.
- This repository is a read-only mirror; fixes are made upstream and republished.

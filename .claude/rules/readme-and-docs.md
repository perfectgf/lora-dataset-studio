---
paths:
  - "README.md"
  - "docs/**"
---

# Documentation

- Update `docs/guide/settings-reference.md` when a setting changes meaning.
- At a release, check README statements affected by the wave. Correct obsolete
  claims; add a capability only if it changes what users can do. What's new is
  the changelog, so do not copy each fix into README.
- State real limits: ranking is not filtering, partial undo is not full undo.
- Documentation-only edits need a diff/link review. Run a documentation
  contract only when its behavior is affected; no application suite or rebuild.

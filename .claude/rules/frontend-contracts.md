---
paths:
  - "frontend/**"
---

# Frontend details

Use the verification policy in [CLAUDE.md](../../CLAUDE.md#scope-and-verification).
This file does not require a browser run or the full frontend suite.

## Shipped features

- `src/whatsNew.js`: prepend a benefit-first entry for a shipped visible feature
  or fix. No entry for internal refactors or fixes to unreleased work. Release
  notes use the entries added since the previous release, not their dates.
- `src/help/helpRegistry.js`: add a topic and real Guide anchor for a new
  setting, page or substantial control that needs explanation.
- Preserve stored identifiers or supply aliases.
- Commit changed `dist` separately after a necessary delivery build.

## Visual checks

A source-text test cannot establish how JSX renders. If appearance is uncertain,
inspect the affected view. Check other widths only when adaptation, width or
overflow is part of the change.

The responsive probe is available for a broader responsive change or a reproduced
layout defect; it is not a gate for every JSX/CSS edit:

```bash
# From frontend/, against a suitable dev instance:
npm run probe:responsive -- --url http://127.0.0.1:5173/#/bank
```

Select the affected route. The probe covers five device sizes and reports
overflow, chrome budget, fill, truncation, targets and overlap. Exit 0 means
clean, 1 violations, 2 unable to run; skipped states are not verified.

If editing the probe or marked surfaces, preserve the roles:
`data-probe-chrome` (fixed chrome), `data-probe-panel` (fill),
`data-probe-world` (pannable content), `data-probe-reading` (reading surface),
`data-probe-layer` (intentional overlay). Bank/Dataset priming relies on their
accessible Open labels. Do not loosen thresholds just to conceal a regression.

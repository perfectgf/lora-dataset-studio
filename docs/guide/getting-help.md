# Getting help & reporting problems

Stuck, found a bug, or missing a feature? Two doors, both watched:

- **Discord** — [discord.gg/j6hnJBFtXE](https://discord.gg/j6hnJBFtXE) — ask in
  **#help**; usually the fastest way to get unstuck. Feature ideas and votes
  live in **#roadmap**.
- **GitHub** — [Issues](https://github.com/perfectgf/lora-dataset-studio/issues) —
  best for reproducible bugs and feature requests; the templates walk you
  through what to include.

---

## What makes a report solvable

The difference between a five-minute fix and a week of guessing is almost
always the same four things:

1. **Version** — shown in Settings → Maintenance → Updates ("Current build").
2. **Environment** — OS, and whether you run API-only, full local, or Docker.
3. **What you did → what you expected → what happened** — three short lines
   beat three paragraphs.
4. **The log** — the last lines of the server log usually name the real error.
   Settings → Maintenance → 🪵 Server log → **Copy all**.

## Or let the app write it for you

The **diagnostic report** button below assembles all of that in one click:
version, OS, capability status, non-secret settings and the last log lines —
formatted, copied to your clipboard, ready to paste into Discord or a GitHub
issue.

What it deliberately **never** includes: your API keys or tokens (only
whether each one is set) and your folder paths (only whether each one is
configured). One caveat: the log tail can mention file names from your machine
— skim the paste before posting if that matters to you.

## Feature requests

Describe the **job you were doing when you missed the feature** — the problem
is more valuable than the proposed solution. Post it in Discord **#roadmap** or
open a GitHub issue with the *Feature request* template.

## Support the project

LoRA Dataset Studio is free, open source and built in the open. If it saves
you time and you want to help development, you can sponsor it on
[GitHub Sponsors](https://github.com/sponsors/perfectgf) — one-time or
monthly, and 100% of it goes to the project (GitHub charges no fees).
The best free ways to help are just as welcome: report bugs, share ideas on
Discord, and star the repo.

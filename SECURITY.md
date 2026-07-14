# Security Policy

## Supported versions

LoRA Dataset Studio ships as a rolling release. Only the **latest release** and the current **`main`** branch are supported — fixes land there, and the in-app update check nudges packaged installs toward the newest release. Please reproduce on an up-to-date checkout before reporting.

## The default threat model

By design the app runs **locally, for a single user, bound to `127.0.0.1`**. It has **no user accounts**: on loopback that's fine, because only you can reach it. Security reports are most useful when they fit that model — for example a way for a *local* attacker, a malicious dataset/image, or a crafted API response from a connected tool to do something it shouldn't (RCE, path traversal outside the data dir, leaking your API keys or config).

Exposing the app to other machines is **your** decision and **out of the default threat model**. If you flip *Available on the local network*, the README's [Exposing the app beyond localhost](README.md#exposing-the-app-beyond-localhost) section is the contract: turn on the access token (or front it with a VPN / authenticated reverse proxy). "I bound it to `0.0.0.0` with the token off and someone on my network reached it" is expected behavior, not a vulnerability. A bypass of the token gate *while it's enabled* absolutely is — please report that.

## Reporting a vulnerability

**Please don't open a public GitHub issue for a security problem.** Report it privately instead:

1. **Preferred — GitHub private vulnerability reporting.** Go to the repo's **[Security tab](https://github.com/perfectgf/lora-dataset-studio/security) → Report a vulnerability**. This opens a private advisory visible only to the maintainers, where we can discuss and coordinate a fix and, if warranted, a CVE.
2. **Alternative — Discord.** If you can't use GitHub advisories, send a **direct message to a moderator** on the [Discord server](https://discord.gg/j6hnJBFtXE) rather than posting details in a public channel.

Helpful things to include: affected version (from **Settings → Maintenance**, or the diagnostic report), how to reproduce, the impact you think it has, and any proof-of-concept. Please **don't put secrets, real API keys or personal data** in the report.

## What to expect

This is a volunteer project, so response is **best-effort** — we'll acknowledge as soon as we reasonably can (think days, not minutes) and keep you in the loop while a fix is worked out. Please give us a fair chance to ship a fix before disclosing publicly. Credit is offered to reporters who want it.

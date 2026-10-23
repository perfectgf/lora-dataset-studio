# Network access and privacy

[Documentation](../README.md) · [Security policy](../../SECURITY.md)

Use **Settings** for normal configuration. The complete defaults, `config.json` keys, model locations and environment overrides live in [docs/guide/settings-reference.md](settings-reference.md).

The server binds to `127.0.0.1` by default. Before enabling LAN access or publishing a port, read [SECURITY.md](../../SECURITY.md#the-default-threat-model) and configure the access-token/VPN/reverse-proxy boundary that fits your network. The whole interface also works on a phone or tablet on your own network, so checking a run or triaging a bank does not need the machine that is training.

**What leaves this machine.** Optional usage statistics stay off unless you explicitly choose to share them. When available, **Settings → Maintenance → Optional usage statistics** explains and controls sharing: a random installation ID, activity dates, coarse feature names, operation outcomes/error categories, failure stages, model families, captioning engines, local/cloud categories, version, OS and duration ranges are sent to the LDS maintainer through PostHog Cloud EU. The expanded categories require a fresh opt-in; earlier refusals stay saved. Images, prompts, captions, file names/paths, credentials, raw logs and session recordings are excluded. Turning sharing off clears the local outbox and identifier; it does not recall events already sent. Builds without a configured collector send no usage statistics. See [the usage statistics guide](settings-reference.md#usage-statistics).

The app also reaches the internet in these situations:

- **Update check** — on load and once an hour, it asks GitHub whether a newer version exists (a `git fetch` on a checkout, the releases API on a packaged install). It sends nothing about you, and there is currently **no setting to turn it off** — block the process at the firewall if you need it silent.
- **Model downloads you start** — Setup and the Install buttons stream weights from Hugging Face, Civitai, Ollama and pytorch.org. Two extras also fetch their own weights the first time you use them: the aesthetic head (~13 MB, from GitHub) and the NSFW classifier plus SigLIP 2 (Hugging Face).
- **API engines and cloud training you configure** — only the providers whose keys you entered, and only when you press the button. OpenRouter additionally receives this project's public name and repository URL as attribution headers.
- **The built-in scraper** — the sites you ask it to scan, and nothing else.

When the app is served on an address the public internet can reach — a rented pod's proxy hostname, a tunnel — set `LDS_PUBLIC=1`. That forces the access token on whatever the setting says, so the switch cannot be turned off into an open door, and generates a token at boot if none exists. It applies to non-loopback binds only, and `LDS_ALLOW_UNAUTHENTICATED=1` still overrides it for setups that authenticate elsewhere.

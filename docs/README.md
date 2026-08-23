# Documentation

This is the documentation index for LoRA Dataset Studio. The root [README](../README.md) stays focused on capabilities and installation; detailed workflows and reference material live here.

## Start here

| Guide | Use it when |
|---|---|
| [Getting started](guide/getting-started.md) | You have just launched the app and want the shortest path to a first dataset |
| [End-to-end workflow](guide/workflow.md) | You want the full source → curate → caption → clean → train → test → export walkthrough |
| [Using the app](guide/using-the-app.md) | You need task-level instructions for a specific workspace, Image Bank, Canvas or Studio action |
| [Dataset quality guide](DATASET_GUIDE.md) | You want guidance on image counts, variety, captions, masks and training-ready data |

## Install and configure

| Guide | Covers |
|---|---|
| [README — Setup & install](../README.md#setup--install) | Windows launcher, manual venv, API-only Docker and the beginner GPU Docker path |
| [Docker guide](guide/docker.md) | GPU-container CLI, storage, existing ComfyUI data, UID/GID, DNS, updates, resources and limits |
| [RunPod guide](guide/runpod.md) | Running the whole studio on a rented GPU pod: registry push, template fields, network volume, the forced access token and the limits |
| [Extensions guide](guide/extensions.md) | Optional local packages under `backend/extensions/`: the `register(app, csrf)` contract, the manifest, the trust model and why the folder can never ship |
| [Settings reference](guide/settings-reference.md) | Every UI setting, dependency, model location, environment override and `config.json` key |
| [Security policy](../SECURITY.md) | Threat model, safe network exposure and private vulnerability reporting |

## Diagnose and recover

| Guide | Covers |
|---|---|
| [Troubleshooting](guide/troubleshooting.md) | Symptom-first fixes for models, ComfyUI, Ollama, training, Docker-adjacent paths and platform issues |
| [Known limitations](guide/known-limitations.md) | Current product boundaries and environment-specific caveats |
| [Getting help](guide/getting-help.md) | Paste-safe diagnostic reports and what to include in a bug report |

## Project information

| Document | Covers |
|---|---|
| [Roadmap](../README.md#roadmap) | Directions currently planned, without date promises |
| [Changelog](../CHANGELOG.md) | Shipped improvements rolled out of the README |
| [Contributing](../CONTRIBUTING.md) | Development setup, tests and pull-request conventions |
| [Code of Conduct](../CODE_OF_CONDUCT.md) | Community expectations |
| [License](../LICENSE) | PolyForm Noncommercial License 1.0.0 |

Current news, support and roadmap discussion live on [Discord](https://discord.gg/j6hnJBFtXE).

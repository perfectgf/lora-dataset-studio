# Documentation

[Project overview](../README.md) · [Public plugins](../README.md#plugins)

## Start here

| Guide | Use it when |
|---|---|
| [Getting started](guide/getting-started.md) | You have just launched the app and want the shortest path to a first dataset |
| [End-to-end workflow](guide/workflow.md) | You want the full source → curate → caption → clean → train → test → export walkthrough |
| [Using the app](guide/using-the-app.md) | You need task-level instructions for a specific workspace, Image Bank, Canvas or Studio action |
| [Dataset quality guide](DATASET_GUIDE.md) | You want guidance on image counts, variety, captions, masks and training-ready data |
| [LoRA training FAQ](faq/README.md) | You have one question: how many images, which captions, which base model, which settings, how much VRAM, which checkpoint |
| [Feature reference](guide/features.md) | Detailed capabilities, screenshots and the limits of each workflow |
| [LoRA training tools compared](lora-training-tools-compared.md) | You are choosing between LDS, kohya_ss, OneTrainer, AI Toolkit and FluxGym |

## Install and configure

| Guide | Covers |
|---|---|
| [Installation](guide/installation.md) | Windows, manual Python, Pinokio, updates, external tools and API keys |
| [Requirements](guide/requirements.md) | Hardware, disk space and dependencies by feature |
| [Migrate to V2](guide/migrate-to-v2.md) | Upgrade a V1 Git installation while preserving data |
| [Docker guide](guide/docker.md) | GPU-container CLI, storage, existing ComfyUI data, UID/GID, DNS, updates, resources and limits |
| [RunPod guide](guide/runpod.md) | Running the whole studio on a rented GPU pod: registry push, template fields, network volume, the forced access token and the limits |
| [Extensions guide](guide/extensions.md) | Optional local packages under `backend/extensions/`: the `register(app, csrf)` contract, the manifest, the trust model and why the folder can never ship |
| [Settings reference](guide/settings-reference.md) | Every UI setting, dependency, model location, environment override and `config.json` key |
| [Network access and privacy](guide/network-access.md) | External connections, optional usage statistics and public-access configuration |
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
| [Releases](https://github.com/perfectgf/lora-dataset-studio/releases) and [changelog](../CHANGELOG.md) | Current release notes and historical improvements |
| [Plugin authoring](plugins/README.md) | SDK, package layout, compatibility and building plugins |
| [Patreon access](plugins/patreon-access.md) | Members-only plugins and early LDS versions, per installation type, with their limits |
| [Contributing](../CONTRIBUTING.md) | Development setup, tests and pull-request conventions |
| [Code of Conduct](../CODE_OF_CONDUCT.md) | Community expectations |
| [License](../LICENSE) | PolyForm Noncommercial License 1.0.0 |
| [Responsible use](responsible-use.md) | Consent, privacy, source rights, prohibited uses and warranty |

Current news, support and feature discussions live on [Discord](https://discord.gg/j6hnJBFtXE).

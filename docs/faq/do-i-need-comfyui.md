# Do I need to know ComfyUI to use LoRA Dataset Studio?
<!-- description: LDS needs a ComfyUI install for local generation and checkpoint tests, not its node editor. Import, captioning and ai-toolkit training work without it. -->

You need a ComfyUI install for local generation and checkpoint tests, but not its node editor: LoRA Dataset Studio (LDS) sends its own workflows for dataset shots, edits, upscaling and checkpoint tests, and Setup downloads the generation models. Importing, curating, captioning and ai-toolkit training on the official Hugging Face bases work without [ComfyUI](https://github.com/comfyanonymous/ComfyUI). With Docker, the GPU launcher can set up a fresh, isolated ComfyUI for you.

## What needs ComfyUI

| Feature | Needs ComfyUI |
|---|---|
| Local Klein and Krea 2 Edit generation, reference edits | Yes |
| Test Studio, deploying a trained LoRA into ComfyUI | Yes |
| Klein watermark cleaning; the SeedVR2, Klein Improve and Camera angles plugins | Yes |
| SDXL training, which uses a base checkpoint from ComfyUI's model folders | Yes |
| Import, scraping, curation, manual captions, export, backup | No |
| Auto-captioning with Ollama, LM Studio or JoyCaption | No |
| API image engines (Gemini, ChatGPT, OpenRouter) | No |
| ai-toolkit training on official Hugging Face bases, cloud training | No |

## Connect an existing ComfyUI

In **Settings → Local tools → ComfyUI**, set the **ComfyUI API URL** (default `http://127.0.0.1:8188`) and the **ComfyUI install directory**, the folder that contains `models/`, `input/` and `output/`. **Test** confirms the connection. LDS reads `extra_model_paths.yaml` the same way ComfyUI does, so models kept in other folders are found.

The Setup wizard checks that folder as you type. You can also continue without ComfyUI: Setup lists what turns off and what stays on, and remembers the choice until you enter a directory.

When a Test Studio family needs nodes your ComfyUI lacks, **Fix missing ComfyUI nodes** gives update steps for portable, Desktop and manual installs. LDS guides the update; it does not run ComfyUI's updater or restart it for you. Krea 2 Edit also needs its own custom node pack and model files, listed in the [settings reference](../guide/settings-reference.md#krea-2-edit-local).

## ComfyUI with Docker

- `start-docker.bat` runs LDS in Docker and connects to the ComfyUI already on your host; it asks for its folder once.
- `start-docker-gpu.bat` creates a fresh, isolated ComfyUI with its own model and data folders beside the checkout, and never touches an existing ComfyUI by default. Budget about 20 GB of disk before model downloads.
- The Docker lane without a GPU has no ComfyUI, so local generation, the Test Studio and local training are unavailable there.
- No Docker image includes local ai-toolkit training; connect ai-toolkit on the host or train in the cloud.

## Limits

- If ComfyUI runs in another container, in WSL or on another machine, its `input/` and `output/` folders must be visible to both sides at the same path, because LDS hands images to ComfyUI through the filesystem.
- ComfyUI's port is not protected by the LDS access token. Keep it on loopback, or behind a firewall, VPN or authenticated reverse proxy.
- ComfyUI paths are tested extensively against a mocked API, but not every model and custom-node combination has run on live installations. A failed preflight names the missing asset.
- Local generation typically needs about 16 GB of NVIDIA VRAM.

## Related

- [How do I upscale or restore images locally?](upscale-and-restore-images-locally.md)
- [External tools](../guide/installation.md#external-tools)
- [Docker guide](../guide/docker.md)
- [ComfyUI runs in another container](../guide/troubleshooting.md#comfyui-runs-in-another-container-or-wsl-or-another-machine-and-generation-fails)

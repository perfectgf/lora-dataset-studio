# LoRA Dataset Studio compared with kohya_ss, OneTrainer, AI Toolkit and FluxGym
<!-- description: How LoRA Dataset Studio compares with kohya_ss, OneTrainer, AI Toolkit and FluxGym for LoRA training: models, captioning, checkpoint tests, installs. -->

kohya_ss and OneTrainer fit people who want a dedicated trainer with many settings and model families, and FluxGym fits a FLUX.1 LoRA with the fewest choices on a 12 to 20 GB card. AI Toolkit has the longest model list of the four trainers, including Krea 2, Qwen-Image 2.1, FLUX.2 and video models, and it is the engine LoRA Dataset Studio (LDS) trains with. LDS adds the work around each run for seven image families, among them Krea 2, FLUX.1, FLUX.2 Klein and Qwen-Image 2.1: building, generating and captioning the dataset, launching guided runs and comparing checkpoints at a fixed seed through ComfyUI. Choose LDS when the dataset or the choice of checkpoint is the hard part, and a trainer on its own when you want full control of the training configuration or a model family LDS has no recipe for.

## At a glance

Checked against each project's README in November 2026.

*Not documented* means the README does not say; the feature may still exist elsewhere in that project. LDS entries come from the LDS documentation.

| | [LDS](../README.md) | [kohya_ss](https://github.com/bmaltais/kohya_ss) | [OneTrainer](https://github.com/Nerogar/OneTrainer) | [AI Toolkit](https://github.com/ostris/ai-toolkit) (the engine LDS uses) | [FluxGym](https://github.com/cocktailpeanut/fluxgym) |
|---|---|---|---|---|---|
| **What it is** | A studio for building datasets, captioning, running training and testing checkpoints, on top of AI Toolkit and ComfyUI | A GUI and CLI for training diffusion models with sd-scripts | A trainer for diffusion models, with a GUI and a CLI | A training suite for image, video and audio models, with a web UI or the CLI | A web UI for FLUX LoRA training on low VRAM, built on sd-scripts |
| **Training engine** | AI Toolkit, installed separately, on your machine or on rented GPUs through the Cloud training plugin | sd-scripts; the GUI generates the CLI commands | Its own (`scripts/train.py`) | Itself (`run.py` with a YAML config) | sd-scripts (`sd3` branch) |
| **Interface** | Web app in one browser tab, served on your machine by default, with a guided flow from dataset to checkpoint | Gradio web UI and CLI | GUI window and CLI scripts | Web UI to start, stop and monitor jobs, with an optional access token; CLI | Gradio web UI; a hidden Advanced tab exposes every sd-scripts flag |
| **Model families** | Krea 2, FLUX.1, FLUX.2 Klein, Qwen-Image 2.1, Z-Image, SDXL, Anima; video LoRAs in beta | SD 1.5 and 2.x, SDXL, SD3, FLUX.1, Lumina Image 2.0, Anima, HunyuanImage-2.1 | FLUX.1, FLUX.2 Dev and Klein, Qwen Image, Z-Image, Chroma, Ernie Image, SD 1.5 to 3.5, SDXL, Würstchen-v2, Stable Cascade, PixArt, Sana, Hunyuan Video, inpainting models | Over 30 image models (among them Krea 2, FLUX.1, FLUX.2, Qwen-Image 2.1, Z-Image, SDXL, SD 1.5, Anima, Chroma, HiDream), edit models, Wan, LTX and MiniMax-H3 video, audio models | FLUX.1 dev, dev2pro and schnell; other bases can be added in `models.yaml` |
| **Training types** | LoRA; a LoKr preset for Qwen-Image 2.1 characters; slider LoRAs (beta); full-model Krea 2 (experimental, cloud only) | LoRA, LoHa, LoKr, Dreambooth, fine-tuning, Textual Inversion, LECO | Full fine-tuning, LoRA, embeddings | LoRA and LoKr, with per-layer targeting; other methods not documented | FLUX LoRA |
| **Dataset building and missing shots** | Character, Style and Concept workflows; import of images, ZIPs and folders; a shot catalog per subject; missing shots generated locally with Klein or Krea 2 Edit, or through API engines (plugin); Image Bank, face similarity, coverage report | Not documented | Image augmentation, and masks (ClipSeg, Rembg) for masked training; generating new shots not documented | A folder of images with same-name `.txt` captions; downscaling and bucketing are automatic; generating new shots not documented | Images uploaded in the UI; generating new shots not documented |
| **Auto-captioning** | JoyCaption through AI Toolkit, or a local vision model through Ollama or LM Studio; prose or booru tags to match the family; trigger word added on export | WD14 tagger, named in the README's change history | BLIP, BLIP2, WD-1.4 | Not documented in the README; LDS runs JoyCaption through it | Not documented; captions are typed in the UI or uploaded as `.txt` files |
| **Samples and checkpoint comparison** | Test Studio: checkpoint × strength grids at a fixed seed, prompt batches, votes and rankings, multi-LoRA comparisons; needs ComfyUI | Sample images during training from a prompt file, with seed, size, CFG and steps per prompt; side-by-side comparison not documented | Sampling during training in the same app, a sampling script, TensorBoard; side-by-side comparison not documented | Checkpoints and sample images saved in the run folder; side-by-side comparison not documented | Optional sample images every N steps, with a fixed seed per prompt to follow the LoRA's progress; side-by-side comparison not documented |
| **VRAM figures** | NVIDIA: 16 GB+ for Z-Image and SDXL, 16–24 GB for FLUX.2 Klein 4B (32–48 GB for 9B), 24 GB for Krea 2 at 1024 px, about 24 GB for FLUX.1 and Anima | Not documented | No figures; EMA weights can stay in CPU memory to save VRAM | No figures; aims at consumer-grade hardware and ships a 24 GB FLUX example config | 12, 16 or 20 GB |
| **Install routes** | Windows ZIP (`start.bat` fetches Python if needed), Git checkout or manual venv, Docker (existing or fresh ComfyUI, or API-only without a GPU), Pinokio, RunPod pod | uv or pip, Docker, RunPod and Novita guides, a community Colab notebook | Git clone and `install.bat` or `install.sh`, or a manual venv (Python 3.11.1 to 3.13); also through Stability Matrix and Visions of Chaos | AI Toolkit Manager (experimental, recommended), manual venv, RunPod template, Modal | Pinokio one-click, manual venv with sd-scripts, Docker |
| **OS** | Windows 10/11 for the full local stack; Linux and macOS for the API-only mode with a manual venv | Windows and Linux; macOS compatibility may vary | Windows, Linux, macOS | Windows, Linux (x86-64 and ARM64), macOS on Apple Silicon (experimental); the manual install expects an NVIDIA GPU | Windows and Linux; macOS not documented |
| **License** | PolyForm Noncommercial 1.0.0: noncommercial use is free, commercial use needs permission | Apache-2.0 | AGPL-3.0 | MIT | MIT |

### Model families people ask about

*Yes* means the family is named in that project's README; for LDS, in its documentation.

| Family | LDS | kohya_ss | OneTrainer | AI Toolkit | FluxGym |
|---|---|---|---|---|---|
| Krea 2 | Yes: Raw, Turbo or a Krea 2 checkpoint on your disk | Not documented | Not documented | Yes: Raw and Turbo, plus edit training | Not documented |
| FLUX.1 | Yes: dev | Yes | Yes | Yes: dev, plus Kontext editing | Yes: dev, dev2pro, schnell |
| FLUX.2 | Yes: Klein 4B and 9B | Not documented | Yes: Dev and Klein | Yes: Dev, Klein 4B and 9B | Not documented |
| Qwen-Image | Yes: 2.1 | Not documented | Yes, version not stated | Yes: Qwen-Image, 2512, 2.1 and 2.1 Turbo, plus edit models | Not documented |
| Z-Image | Yes | Not documented | Yes | Yes: base, Turbo, De-Turbo, L2P | Not documented |
| SDXL | Yes, from a base checkpoint in ComfyUI's model folders | Yes | Yes | Yes | Not documented |
| Anima | Yes | Yes | Not documented | Yes | Not documented |
| SD 1.5 | No | Yes | Yes | Yes | Not documented |
| Video | Beta: Wan 2.1 and 2.2, LTX-2 and 2.3, MiniMax H3 | Not documented | Hunyuan Video | Wan 2.1 and 2.2, LTX-2, 2.3 and 2.5, MiniMax-H3 | Not documented |

## Choose LoRA Dataset Studio if…

- You start from one reference photo: LDS proposes the face, bust, body and back shots a character set needs, generates the missing ones locally with Klein or Krea 2 Edit, and face similarity helps you reject shots where the face drifts.
- You start from a large folder: the Image Bank scores images, groups duplicates and people, and builds balanced shortlists to promote into a dataset.
- You want captions in the style each family expects (prose for Krea 2, FLUX and Z-Image, booru tags for SDXL), with the trigger word added on export and per-trait rules for what the captions describe.
- You would rather pick a recipe than write a config: LDS writes the AI Toolkit config for its seven image families, scales steps to the dataset, queues runs and continues from any saved epoch.
- You choose checkpoints by looking: Test Studio renders the checkpoints you pick at several strengths with the same seed and prompts, then lets you vote and rank them.

Datasets export as standard ZIPs of images with same-name `.txt` captions, so a set built in LDS can feed kohya_ss, OneTrainer or another trainer, and captions written in another tool can be imported back.

## Choose kohya_ss if…

- You want a GUI over a wide range of sd-scripts parameters, with the CLI command generated for each run.
- You need LoHa, Dreambooth, Textual Inversion or LECO concept erasure, which LDS does not offer, or fine-tuning beyond the experimental Krea 2 full-model lane in LDS.
- You train SD 1.5, SD 2.x, SD3, Lumina Image 2.0 or HunyuanImage-2.1, which have no LDS recipe.
- You train locally on Linux, which its README supports through community-maintained instructions, while the full local LDS stack targets Windows 10/11. Its README also documents a GUI that works offline once installed.

## Choose OneTrainer if…

- You want full fine-tuning or embeddings as well as LoRA; LDS fine-tunes only Krea 2, experimentally and in the cloud.
- You train families LDS has no recipe for, such as Chroma, SD 3.x, Sana, PixArt, Stable Cascade, Ernie Image or Hunyuan Video.
- You want the training options its README lists in one place: masked training, EMA, aspect-ratio bucketing, multi-resolution training, noise-schedule rescaling, image augmentation, several prompts per image and automatic backups to continue a run.
- You want BLIP, BLIP2 or WD-1.4 captioning, mask generation and model format conversion built in, with install scripts for Windows, Linux and macOS.

## Choose AI Toolkit directly if…

AI Toolkit is the engine LDS drives, so this is a choice of interface, not of trainer.

- You need a model LDS has no recipe for. Its README lists over 30 image models, edit models such as Qwen-Image-Edit and FLUX.1 Kontext, Wan, LTX and MiniMax-H3 video models, and audio models; the LDS feature reference itself recommends AI Toolkit directly for unsupported architectures and experimental configuration keys.
- You want to edit the YAML config yourself, train only chosen layers or switch the network to LoKr.
- You run Linux (including ARM64), macOS on Apple Silicon or a headless server, where the experimental AI Toolkit Manager keeps its whole environment inside its own folder.
- Your dataset is already built and captioned and you only need the run.

LDS uses an existing AI Toolkit folder set in **Settings → Local tools**, so one install can serve both.

## Choose FluxGym if…

- You want a FLUX.1 LoRA in three steps: enter the LoRA details, upload and caption the images, start.
- Your card has 12, 16 or 20 GB of VRAM, the sizes its README targets.
- You prefer a Pinokio one-click or Docker install, with models downloaded when training starts.
- You want sample images every N steps, publishing to Hugging Face from the same UI, and an Advanced tab with every sd-scripts flag when the defaults are not enough.

Its README lists FLUX.1 bases only; the family table above shows which tools name Krea 2, FLUX.2 or Qwen-Image.

## LDS limits

- **Training needs AI Toolkit.** Local training uses a separate AI Toolkit install, set in **Settings → Local tools**; Qwen-Image 2.1 needs a recent toolkit with `qwen_image_2` support. LDS has recipes for seven image families, not for every model AI Toolkit supports.
- **Generation and testing need ComfyUI.** Generating dataset shots locally, Test Studio and Canvas generation need a reachable ComfyUI with matching models; Setup downloads the generation models. Training on official Hugging Face bases does not need ComfyUI, but SDXL training takes its base checkpoint from ComfyUI's model folders.
- **VRAM.** Importing, sorting and manual captioning need no GPU. Auto-captioning takes about 8 GB of VRAM and local generation about 16 GB, on NVIDIA cards; training needs the amounts in the table above, and Qwen-Image 2.1 has not been measured in LDS. 16 GB of system RAM is recommended, and the app does not check it.
- **Platform.** The full local stack targets Windows 10/11; Linux and macOS run the API-only mode with a manual venv. The Docker images and the RunPod image do not include AI Toolkit: the GPU Docker lanes use a host install or cloud training, and the no-GPU image and the RunPod pod have no local training.
- **Experimental parts.** Video LoRA training is beta, and only Wan 2.2 14B has completed a run so far; slider LoRAs are beta; full-model Krea 2 training is experimental, cloud-only and needs 80 GB of VRAM. Cloud training support varies by family, and FLUX.1 and Anima train locally only.
- **License and releases.** PolyForm Noncommercial 1.0.0: commercial use needs separate permission from the licensor. New versions reach Patreon members first, then everyone; exclusive plugins stay members-only, and the installed version keeps working.

The [known limitations](guide/known-limitations.md) page lists the current boundaries in detail.

## Related

- [Requirements: hardware, VRAM and dependencies by feature](guide/requirements.md)
- [Pick your model family first](DATASET_GUIDE.md#1-pick-your-model-family-first)
- [Which LoRA trainer should I use?](faq/which-lora-trainer.md)
- [How to pick the best LoRA checkpoint](faq/pick-the-best-lora-checkpoint.md)

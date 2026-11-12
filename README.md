# LoRA Dataset Studio V2

**From one reference photo to a trained, ranked LoRA, on your own PC.**

LDS builds the dataset for you: it generates the shots your photos are missing, sorts a big image folder, cleans watermarks and writes the captions. It then trains the LoRA with [ai-toolkit](https://github.com/ostris/ai-toolkit) and renders your checkpoints side by side at the same seed, with votes and a face similarity ranking, so you can pick the best one. Character, style and concept LoRAs for Krea 2, FLUX.1, FLUX.2 Klein, Qwen-Image 2.1, Z-Image, SDXL and Anima. Free and local, with no account, in one browser tab: [ComfyUI](https://github.com/comfyanonymous/ComfyUI) runs behind it for local generation and checkpoint testing, without a node graph to wire.

[Install](#setup--install) · [Which GPU do I need?](#minimum-requirements) · [What it does](#what-it-does) · [Plugins](#plugins) · [Member edition](#latest-member-updates) · [Documentation](docs/README.md) · [Releases](https://github.com/perfectgf/lora-dataset-studio/releases) · [Discord](https://discord.gg/j6hnJBFtXE)

[![Support us on Patreon](https://img.shields.io/badge/Patreon-Support%20us-FF424D?style=for-the-badge&logo=patreon&logoColor=white)](https://www.patreon.com/c/Loraperfectgf) [![Join our Discord](https://img.shields.io/badge/Discord-Join%20us-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/j6hnJBFtXE) [![Sign up on Vast.ai](https://img.shields.io/badge/Vast.ai-Sign%20up-F7B32B?style=for-the-badge)](https://cloud.vast.ai/?ref_id=683073)

<table>
  <tr>
    <td width="50%" align="center"><a href="docs/screenshots/generate/generate-variations.png"><img src="docs/screenshots/generate/generate-variations.png" alt="Generation engines for LoRA dataset images" width="100%"></a><br><sub>Generation engines</sub></td>
    <td width="50%" align="center"><a href="docs/screenshots/02-workspace.png"><img src="docs/screenshots/02-workspace.png" alt="LoRA dataset workspace" width="100%"></a><br><sub>Dataset workspace</sub></td>
  </tr>
</table>

A complete Character LoRA workflow in seven minutes:

https://github.com/user-attachments/assets/d51ff89c-34e9-41a9-b47d-08939a8c867b

People shown in the demo and screenshots are AI-generated.

> [!WARNING]
> **No affiliation with loradataset.com.** That website is not operated, endorsed, or supported by the LoRA Dataset Studio team. Any payments made to that service do not support this project. Use this repository and the links provided here to find our official downloads and community channels.

## Setup & install

1. Download `LoRA-Dataset-Studio-windows.zip` from the [latest release](https://github.com/perfectgf/lora-dataset-studio/releases/latest) and extract it into a new folder.
2. Double-click `start.bat`. It prepares Python and opens LDS in your browser.

Complete **Setup**, then create a dataset or install the plugins you need. Importing, organizing and manually captioning images require no GPU or API key. Local training also needs [ai-toolkit](https://github.com/ostris/ai-toolkit) and local generation needs [ComfyUI](https://github.com/comfyanonymous/ComfyUI): see [external tools](docs/guide/installation.md#external-tools).

| Installation | Instructions |
|---|---|
| Git checkout or manual Python environment | [Native installation](docs/guide/installation.md#windows) |
| Docker with an existing or fresh ComfyUI | [Docker guide](docs/guide/docker.md) |
| Docker without a GPU | [API-only setup](docs/guide/installation.md#docker-without-a-gpu) |
| Pinokio | [One-click installation](docs/guide/installation.md#pinokio) |
| Rented RunPod GPU | [RunPod guide](docs/guide/runpod.md) |

**Updates:** install core updates from **Settings → Maintenance → Updates** and plugin updates from **Plugins → Updates**. Member updates come through the linked Patreon service; public releases are available on GitHub. Git and Pinokio checkouts fast-forward to a verified release; ZIP installations swap in a verified archive and retain their datasets, media, settings and history. Docker installations from the Windows ZIP install member updates from the host with `update-docker.bat patreon`. Public features remain available without a membership; reserved features require an active **LDS Member** membership. `v1` is frozen and old `main` installations must [migrate to V2](docs/guide/migrate-to-v2.md). Pinokio and Docker keep their own [update procedures](docs/guide/installation.md).

### Minimum requirements

The core runs without a GPU. Each task has its own floor:

| What you want to do | Smallest NVIDIA card |
|---|---|
| Import, sort, crop, caption by hand, export | No GPU |
| Caption automatically | About 8 GB |
| Generate dataset shots locally | About 16 GB |
| Train Z-Image or SDXL | 16 GB |
| Train FLUX.2 Klein 4B | 16 to 24 GB (9B: 32 to 48 GB) |
| Train Krea 2 | 24 GB at 1024 px, 768 px only below 24 GB |
| Train FLUX.1 or Anima | About 24 GB |
| No suitable card | Train on a rented [vast.ai](https://cloud.vast.ai/?ref_id=683073) GPU with the [Cloud training](bundled/cloud_training/) plugin, or run LDS on a [RunPod GPU](docs/guide/runpod.md) |

These floors come from a 24 GB reference card; training below 16 GB has not been measured. The app needs Python 3.11–3.12 (3.13+ runs the core but not the local ML extras); the Windows launcher downloads Python 3.12 if needed. See the [hardware and dependency tables](docs/guide/requirements.md) before downloading models or renting a GPU.

## What it does

**81 tools, from a few pictures to a LoRA you can trust.** Each one works on its own, and they hand off to each other. The [feature reference](docs/guide/features.md) describes every one in detail.

[Build a dataset from one photo](#build-a-dataset-from-one-photo) (10) · [Sort a big folder with the Image Bank](#sort-a-big-folder-with-the-image-bank) (15) · [Curate, caption and clean](#curate-caption-and-clean) (8) · [Train without writing a config](#train-without-writing-a-config) (9) · [Pick the best checkpoint with evidence](#pick-the-best-checkpoint-with-evidence) (10) · [Keep control of your files](#keep-control-of-your-files) (7) · [Build video datasets (beta)](#build-video-datasets-beta) (22)

### Build a dataset from one photo

Give LDS one reference photo of your character: it proposes the shots a LoRA needs and generates the missing ones, locally or through an API engine. Your own photos, ZIPs and folders work too. The demo above shows the whole Character workflow.

- **Character, Concept and Style datasets**: each kind has its own captioning, masking, readiness checks and training policies.
- **Human, animal, creature, object and anime subjects**: identity wording and shot catalogs adapted to the subject; Anime keeps the illustrated rendering.
- **Five generation engines**: Nano Banana Pro (Gemini), ChatGPT (`gpt-image-2`), OpenRouter, or local Klein and Krea 2 Edit through ComfyUI.
- **Several engines in one batch**: split one shot list across engines; every result stays labelled with the engine that made it.
- **Generation queue**: generations, upscale batches and retries line up, and a dock shows what the GPU is doing and what is waiting.
- **Krea 2 Edit**: restage a single reference while preserving identity, without a character LoRA first.
- **Variation catalog**: balanced expression, angle, lighting, framing, outfit and background shots, with your own entries and JSON import and export.
- **Shot-type views**: sort or group the grid by shot type to see what you have too many or too few of.
- **Reference editing and exact retry**: edit the reference with any engine, compare before and after, then retry with the exact prompt.
- **Import or scrape**: drag in images, merge ZIP or folder datasets, search Reddit, Pexels or the open web, or scan a gallery URL.

### Sort a big folder with the Image Bank

Point the Bank at a folder and review it in place: it scores, groups and searches thousands of images, then builds a balanced shortlist to promote into a dataset.

<p align="center"><a href="docs/screenshots/bank/bank-overview.png"><img src="docs/screenshots/bank/bank-overview.png" alt="Image Bank: sort a large image folder before building a LoRA dataset" width="800"></a></p>

- **Folder or web scrape to bank**: inventory a live folder in place, or scrape straight into a bank.
- **One scoring pass, many answers**: one pass gives aesthetic, NSFW and style rankings, groups, and the embeddings later tools reuse.
- **Quality and similarity passes**: flag blur, noise, flat frames, small images and black bars; group duplicates, crops, recompressions and near-duplicates.
- **Auto-reject by flag**: turn any set of quality flags into one bulk rejection, and undo it as one decision.
- **People, framing and captions**: cluster faces without a reference, classify face, bust, body and back, and caption the bank for full-text search.
- **Medium and head angle**: split photographs, anime, 3D renders and illustrations, and filter by frontal, three-quarter, profile or back view.
- **Find and shortlist**: find by text, pick diverse or framing-balanced sets, find similar images, or describe the set you want in a sentence.
- **Coverage advice**: names what the pool never shows, such as no profile views or a single outfit. Advice only: nothing is rejected.
- **Fast review tools**: filter, sort, review one by one, rotate without rewriting the source, and re-run only the passes that apply.
- **Editable watermark masks**: detect marks, redraw the mask zones, then crop or repaint into a separate clean copy; the source file is never written to.
- **Crop and upscale in the bank**: reframe a shot, or re-render it at a higher resolution with Klein or SeedVR2, without leaving the bank.
- **Find text**: read burned-in lettering (speech bubbles, subtitles, Latin or CJK) into the same masks, on the CPU.
- **Dataset and bank round trip**: promote bank keepers into a dataset, or copy dataset keepers into a bank, with their metadata.
- **Safe bulk work**: undo the last bulk decision, tune thresholds where you work, move a bank without losing its analysis, or run the whole chain overnight.
- **One destructive action**: **Delete rejected** is the only Bank action that touches the source folder, behind a typed confirmation that says where the files go.

### Curate, caption and clean

Reject what would end up in the LoRA, remove watermarks and burned-in text, and caption in the style your model reads.

<p align="center"><a href="docs/screenshots/03-curate.png"><img src="docs/screenshots/03-curate.png" alt="LoRA dataset curation: keep, reject and crop training images" width="800"></a></p>

- **Curation grid**: keep or reject, crop, mirror, rotate, zoom, resize and multi-select, with non-destructive upscale candidates from Klein or SeedVR2.
- **Identity and composition checks**: InsightFace similarity, score-based auto-triage, framing badges and a live composition meter for characters.
- **Model-matched captions**: prose or booru tags chosen by target family, with leak checks for Concept and content-only rules for Style.
- **Appearance policy**: per trait, choose whether captions omit or describe hair, makeup and nails, facial hair and glasses.
- **Caption Lab and recovery**: find and replace, tag frequencies, targeted re-captioning, stoppable batches and recovery after a reload.
- **External caption round trip**: export image and `.txt` pairs, caption them in any tool, then re-import without duplicates or overwritten captions.
- **Dual long and short captions**: text-side augmentation for the local families that support it; both wordings stay editable.
- **Watermark review**: detect, review and edit masks, then crop, inpaint with LaMa or re-render with Klein; every edit keeps a backup of the original.

### Train without writing a config

Pick a recipe and start: LDS fills in the ai-toolkit settings and scales the steps with the dataset, on your GPU or a rented one.

<p align="center"><a href="docs/screenshots/training/runs-hub.png"><img src="docs/screenshots/training/runs-hub.png" alt="LoRA training runs with ai-toolkit" width="800"></a></p>

- **Guided local training**: ai-toolkit underneath, starters per family, adaptive step policies, launch guards, a queue and advanced controls.
- **Slider LoRA (beta)**: train a two-sided slider from positive and negative prompts, so strength moves the trait either way.
- **Cloud training**: rent a [vast.ai](https://cloud.vast.ai/?ref_id=683073) GPU from the same launch flow, stream progress and checkpoints home, and terminate pods automatically.
- **Parallel cloud runs**: several cloud trainings on one dataset at once to compare settings, each on its own pod.
- **Full-model training (Krea 2)**: train the whole transformer instead of an adapter, on a rented GPU only.
- **Merge a LoRA into a checkpoint**: fold one or more LoRAs into a base, each at its own weight, with a plan shown before anything is written.
- **Custom bases and continuation**: train compatible custom weights and continue from any saved epoch.
- **Runs**: local and cloud runs together, with progress, logs, stop, retry, continue and download.
- **Experiment lineage**: inspect, annotate and diff the tree of runs and the checkpoint each continuation resumed from.

### Pick the best checkpoint with evidence

Render every checkpoint with the same seed and prompts, vote, rank by face similarity, and keep everything you generated.

<p align="center"><a href="docs/screenshots/studio/studio-grid.png"><img src="docs/screenshots/studio/studio-grid.png" alt="Test Studio grid comparing LoRA checkpoints at a fixed seed" width="800"></a></p>

- **LoRA Canvas**: every dataset's lineage on one board: compare runs, generate from checkpoints and blend several of them.
- **Test Studio**: fixed-seed checkpoint and strength grids, multi-LoRA comparisons or combined stacks, votes, Wilson ranking and face ranking.
- **Prompt batch**: tick several prompts and one launch renders them all with the same checkpoints, settings and seed.
- **Scenes**: run a bank's or a dataset's captions in order as one batch of prompts.
- **Gallery**: one feed of every image the app generated, across datasets, with filters, downloads and upscale.
- **Camera angles**: re-photograph a generated or dataset image from other camera positions, locally with Qwen-Image-Edit.
- **Civitai top prompts**: browse Civitai's most-reacted images next to their prompts and reuse one or several, with a free Civitai API key.
- **Video Test Studio**: the same comparison for a video LoRA: one clip per start frame, with the same seed and prompt.
- **DLSS 5 Neural Rendering**: re-render a finished clip through NVIDIA's DLSS 5 model and compare both side by side. Windows and NVIDIA only; you supply the model file.
- **Studio shortcuts and recovery**: open Studio from a run, draw prompts from kept captions, and pause safely when ComfyUI drops.

### Keep control of your files

Your datasets stay yours: standard exports, one-file backups and storage you can move.

- **Training ZIP and sidecars**: kept images with same-name `.txt` captions, for ai-toolkit and Kohya-compatible tools.
- **Portable backup and restore**: datasets, decisions, captions, settings and run history in one file; API keys stay out.
- **Hugging Face publishing**: publish kept pairs to a dataset repository, private by default, after a rights confirmation.
- **Publish to Civitai**: push a checkpoint and the images you pick to a Civitai model page; checkpoints stay drafts by default.
- **ComfyUI deployment**: deploy LoRA checkpoints or downloaded cloud results into your ComfyUI LoRA folder.
- **Recoverable deletion**: deleted app data goes to the Trash, and destructive Bank actions state their destination first.
- **Storage you can see and move**: every folder the app writes to, with its size, and the option to move data to another drive.

### Build video datasets (beta)

Turn long videos into training clips: cut, measure, clean and caption the shots before you train. This lane is in beta.

<p align="center"><a href="docs/screenshots/video/video-studio-clips.png"><img src="docs/screenshots/video/video-studio-clips.png" alt="Video Test Studio comparing video LoRA clips" width="800"></a></p>

- **Folder to video bank**: point a bank at a folder of videos; it is referenced in place and never written to.
- **Create a video dataset directly**: add local files or selected web videos, encoded to the model, length and size you choose.
- **Automatic shot detection**: finds the cuts, so a long file becomes individually reviewable shots.
- **Review without waiting**: thumbnails first, and a click plays the shot from the source; nothing is encoded before you decide.
- **Target-aware cutting**: clip lengths are limited to the frame counts your model accepts (Wan, LTX, MiniMax H3).
- **Encode only what you keep**: re-encoding is paid once, at promotion, for the clips you kept.
- **Fix a bad cut instead of rejecting it**: trim either bound, split a shot at the playhead, or draw a shot the detector missed.
- **Measure every shot**: one pass scores stillness, blur, black moments and frozen stretches; nothing is auto-rejected.
- **Sound measured, not assumed**: every shot gets its share of silence and its level, for the models that keep audio.
- **Cap one source's share**: limit how many clips a single file contributes.
- **Trim the transition off both ends**: optional trim at export; a clip made too short is dropped, never exported short.
- **Train it without leaving the app**: a promoted set gets a Train button that runs it through your ai-toolkit.
- **Shots described in words**: a pass writes what happens in each shot, which becomes its caption; your edits are never overwritten.
- **Spot the shot you already have**: groups near-identical takes and keeps the sharpest one unflagged.
- **Spot the watermarked shots**: runs the watermark detector on each shot's sharpest frame.
- **See the bands and the subtitles**: measures letterbox bars and text that holds still, and shows the crop that would remove them.
- **Catch encoding damage**: duplicated frames, compression blocking and upscales that hide at thumbnail size.
- **Find a scene by typing a word**: ranks the shots by a phrase and tells you which second matched. A ranking, not a filter.
- **Triage one keystroke per shot**: keep, reject or skip from the keyboard.
- **Sort shots by what the camera did**: labels pans, tilts, push-ins, handheld and locked-off shots.
- **Find the shots that are secretly two shots**: flags the soft cuts that detection misses.
- **See which shots may be generated**: flags clips whose motion looks generated. A hint, not a verdict.

Dependencies vary by feature. Bank search ranks matches; it does not guarantee exclusions. Undo covers specific actions, and **Delete rejected** can remove source files after confirmation. Video and slider workflows are experimental. See the [feature reference](docs/guide/features.md), [requirements](docs/guide/requirements.md) and [known limitations](docs/guide/known-limitations.md) for the detailed behavior.

## Screenshots

Click a screenshot to view it at full size.

<table>
  <tr>
    <td width="33%" align="center"><a href="docs/screenshots/bank/bank-analyze-and-overview.png"><img src="docs/screenshots/bank/bank-analyze-and-overview.png" alt="Image Bank analysis and dataset coverage" width="100%"></a><br><sub>Analysis and coverage</sub></td>
    <td width="33%" align="center"><a href="docs/screenshots/bank/bank-launch-all.png"><img src="docs/screenshots/bank/bank-launch-all.png" alt="Batch image analysis in the Image Bank" width="100%"></a><br><sub>Batch analysis</sub></td>
    <td width="33%" align="center"><a href="docs/screenshots/bank/find-text-launch.png"><img src="docs/screenshots/bank/find-text-launch.png" alt="Text detection on training images" width="100%"></a><br><sub>Text detection</sub></td>
  </tr>
  <tr>
    <td width="33%" align="center"><a href="docs/screenshots/canvas/canvas-board.png"><img src="docs/screenshots/canvas/canvas-board.png" alt="LoRA Canvas: compare and blend LoRAs on a board" width="100%"></a><br><sub>LoRA Canvas</sub></td>
    <td width="33%" align="center"><a href="docs/screenshots/studio/civitai-prompt-browser.png"><img src="docs/screenshots/studio/civitai-prompt-browser.png" alt="Civitai prompt browser for LoRA tests" width="100%"></a><br><sub>Civitai prompts</sub></td>
    <td width="33%" align="center"><a href="docs/screenshots/release/camera-angles-picker.png"><img src="docs/screenshots/release/camera-angles-picker.png" alt="Camera angles: new viewpoints for a character dataset" width="100%"></a><br><sub>Camera angles</sub></td>
  </tr>
  <tr>
    <td width="33%" align="center"><a href="docs/screenshots/video/dlss5-compare.png"><img src="docs/screenshots/video/dlss5-compare.png" alt="DLSS 5 neural rendering before and after comparison" width="100%"></a><br><sub>DLSS 5 comparison</sub></td>
  </tr>
</table>

## Plugins

Install optional features from **Plugins → Store**, configure and prepare them in their own settings, and update them through **Plugins → Updates**. Several plugins can be installed together with one LDS restart. Each is independently installable; model downloads, hardware and provider credentials depend on the feature.

| Plugin | What it does | Main requirements |
|---|---|---|
| [API image engines](bundled/api_engines/) | Generate images with Gemini/Nano Banana, ChatGPT or OpenRouter; experimental ChatGPT subscription connection | Provider credentials; API usage is billed by the provider |
| [Camera angles](bundled/camera_angles/) | Create new viewpoints of Gallery or dataset images with controls for direction, height and distance | ComfyUI, Qwen-Image-Edit and Multiple-Angles LoRA |
| [Canvas](bundled/canvas/) | Arrange runs and checkpoints on a visual board, compare or blend LoRAs, generate images and export layouts | ComfyUI for generation; one model family per generation batch |
| [Cloud training](bundled/cloud_training/) | Train image or video models on rented GPUs, monitor runs, recover checkpoints and deliver full Krea 2 models | [vast.ai](https://cloud.vast.ai/?ref_id=683073) account and API key; Hugging Face access for applicable runs |
| [Dataset Forge](bundled/qwen_dataset/) | Generate dataset variations locally with Qwen-Image 2.1, using 1–10 references of the same subject and your selected shots | Recent ComfyUI and Qwen-Image 2.1 model files; preparation included |
| [DLSS 5 Neural Rendering](bundled/dlss5/) | Improve a finished video's lighting and materials, compare with the original and export the result | Windows x86-64, NVIDIA driver, compatible model supplied by you and the prepared bridge |
| [Klein Improve](bundled/image_upscale/) | Re-render image detail with instructions, LoRA presets and finishing controls | ComfyUI and Klein models; appearance and color can change |
| [Live channels](bundled/live/) | Continuously render scripted H3 scenes and watch them in a browser or VLC | Compatible local ComfyUI, H3 weights and stream encoder; experimental |
| [Model tools](bundled/model_tools/) | Quantize full models to fp8 or merge weighted LoRAs into a base checkpoint | Python with PyTorch and output disk space; CPU processing, no GPU required |
| [Publish to Civitai](bundled/civitai_publish/) | Upload checkpoints and generated images to a model page | Civitai API key; checkpoints default to drafts, image posts default to publication |
| [Publish to Hugging Face](bundled/hf_publish/) | Export kept images, captions, metadata and a dataset card to the Hub | Write-enabled Hugging Face token; private repository by default and rights confirmation |
| [Resource monitor](bundled/resource_monitor/) | Show CPU, GPU, RAM, VRAM and temperatures, with guarded memory release | No model download |
| [SeedVR2](bundled/seedvr2/) | Restore and upscale images with high-resolution tiling and finishing controls | ComfyUI, SeedVR2 nodes and models; optional TTP nodes for tiling |
| [Video lane](bundled/video/) | Build video datasets from local files or web imports, curate shots, train LoRAs and test H3 generation, continuation and interpolation | Dependencies vary by task: PyAV/ffmpeg, ai-toolkit, ComfyUI and model weights; Video Bank remains beta |
| [Web scraping](bundled/scrape/) | Import selected images or clips from searches and supported gallery URLs into datasets or banks | Source-dependent credentials and permissions; Pexels requires explicit dataset/ML authorization |

API providers apply their own billing and content policies. Model licenses also apply, including MiniMax H3's territory restrictions; check the [video limits](docs/guide/features.md#video-bank-beta--first-release-read-the-limits) before using it. Plugin authors can start with the [SDK and package guide](docs/plugins/README.md).

### Member plugins

These plugins are delivered through the member catalog and require an active, linked **LDS Member** membership. Their cards list the included features, dependencies and preparation steps.

| Plugin | Available version | What it adds |
|---|---|---|
| Dataset Pilot | **1.4.0** | Prepare independent face datasets from selected references with saved multi-engine recipes, identity and quality checks, framing balance, profile review, optional watermark repair and JoyCaption. Resume interrupted work and fill missing shots within the original budget. Completion rounds favor engines that produced usable images; near-identical compositions are set aside. Follow each identity's progress and choose which dataset to view. |
| LoRA Manager | **1.8.2** | Manage and deploy training saves, merge weighted LoRAs, or bake LoRA/LoKr adapters into compatible base checkpoints, including H3 Image. Manage merged files and ComfyUI copies; the merge engine runs on the CPU. |
| Civitai Studio | **1.1.2** | Prepare editable LoRA or checkpoint drafts with up to 20 matching Gallery examples, resume uploads, publish video checkpoints and stamp posted images with their LoRA name and strength. Example uploads omit embedded prompts, metadata and workflows. |
| PiD 4K Upscale | **1.0.1** | Decode a Krea 2 Test Studio result at 4× size, about 4096 × 4096 for a square image, and keep it beside the original. Prepares its additional models when needed. |
| Angle Variations | **1.0.2** | Generate new dataset viewpoints with Qwen-Image 2.1 Turbo, from three-quarter and profile views to back, high and low angles. |
| Multi-subject | **1.1.2** | Place 2–8 people in one Krea 2 Test Studio image, each with their own LoRA, strength, description and drawn region. Requires Krea2 Regional nodes. |
| Full Model Files | **1.0.0** | Prepare INT8 convrot and Turbo files automatically after a full Krea 2 model returns from cloud training. |
| H3 Video Decoders | **1.0.0** | Prepare and select the optional Sol-H3 video decoder in FP16 or INT8, with verified downloads and ComfyUI compatibility checks. |
| Qwen Watermark Cleaner | **1.0.0** | Clean watermarks in datasets and the Bank with Qwen-Image 2.1 or its 8-step Turbo engine, with model preparation included. |
| Creative Library | **1.0.8** | Browse visual prompt libraries, save favorites and compose prompts from outfits, scenes, poses and lighting. Batch prompts or individual parts in Test Studio, and fill multi-subject scenes or templates. Use the online library or import your own Sick Ollie packs. |

## Latest member updates

> [!IMPORTANT]
> **The public edition is free; member features require an active LDS Member subscription.** Updates reach [Patreon](https://www.patreon.com/c/Loraperfectgf) members first. Bug fixes also reach the public edition; exclusive features and plugins remain members-only. Link an active **LDS Member** membership in LDS to use them, including in an already installed version. Your files and datasets remain yours when membership ends.

The public core and public plugins are free under the [PolyForm Noncommercial license](LICENSE), and running them needs no account; optional usage statistics are off by default. External APIs and rented GPUs have their own charges. Public features keep working without a membership.

The [member edition](#latest-member-updates) also trains and tests **MOSS-Image** and **MiniMax H3-Image** LoRAs, with additional editing, refinement and workflow tools.

The following improvements are available in the **2026.11.12.21 member update**. Install the member core from **Settings → Maintenance → Updates** and member plugins from **Plugins → Store / Updates**. The public GitHub download has its own release schedule.

| Area | What's new |
|---|---|
| H3 Image workspace | **Create, Edit and Combine** have their own controls. Edit one source or combine up to **16 ordered references**, with a role for each image. Import originals from the **Gallery**, filter by the main or additional LoRAs recorded on each image, and keep your selection as you change filters. |
| H3 sampling and finishing | Use the native **6-step** recipe or prepare the optional pDMD adapter for **Turbo at 2, 3 or 4 steps**. Standard refinement supports one, two or three passes: **6, 12 or 18 total sampling steps**, with optional enlargement up to 2× within the size limit, sharpening and film grain. References and settings stay with each result. |
| Image Test Studio | Drop an image file or a previous result onto H3 references, the prompt or **Describe**. Earlier results stay in the grid when you change sampling settings, including tests without a LoRA. Each tile shows its settings; generation status sits directly above the bottom shortcuts. Compatible **Always LoRAs** also work without a selected dataset. |
| Image Bank | Clear **Analyze, Review and Create dataset** action groups. Right-click an image to find similar images, or the grid to select a diverse or balanced set, search by description and apply quality-based rejection. Each tool exposes its settings before running. |
| H3 Image training | Train image LoRAs locally or on a rented GPU. Start from the sourced AI Toolkit rank-16 recipe or the rank-32 adaptation of DiffSynth's image recipe for Character, Style and Concept datasets. The default quantized cloud recipe starts its GPU search at an estimated **48 GB VRAM**; larger settings can need more. |
| Trainer maintenance | Update the configured **AI Toolkit** and its dependencies from the dataset, follow the log and refresh readiness automatically. Local jobs are protected during the update. |
| Civitai prompts | Search by image tag name, choose a suggestion and keep that filter across further pages or a model's gallery, with the original prompts available. |

H3 Image training, its Studio workspace and automatic trainer updates are member features. H3 model terms and the hardware requirements still apply. The in-app **What's new** panel and Guide describe the controls available in your installed version.

The member edition also includes these previously released tools:

| Area | Member capabilities |
|---|---|
| MOSS-Image | Train locally or in the cloud on image-and-caption datasets, then generate and compare LoRAs in Test Studio. Use Base or compatible SFT models, with sampler, scheduler, second-pass and finishing controls. The optional SFT distillation LoRA offers 4 steps up to 1 MP, 6 around 1536 px and 8 at 2048 px; output is limited to 4 MP. Image-edit training pairs are not supported for this family. |
| Krea 2 image tools | Optional **Krea2T prompt enhancer** with adjustable strength, plus a **2.25× two-stage upscale**: a 1.5× latent pass followed by a 1.5× image upscale and light refinement. Model and node preparation depend on the selected tool. |
| Training recipes | Edit learning rates, layer exclusions, caching and preview settings together. Saved presets and run snapshots keep the recipe for local and cloud training, with model-specific safeguards. |
| Suggested crops and recovery | Ask a local vision model to suggest a crop around the visible subject, adjust it before applying, then undo new dataset crops or restore their saved original. This does not recover edits discarded before crop history was available. |
| Remake one image | Render one Test Studio result again with a new seed without restarting the whole test. |

## Documentation

- [Getting started](docs/guide/getting-started.md) and [end-to-end workflow](docs/guide/workflow.md)
- [Task instructions](docs/guide/using-the-app.md) and [dataset quality](docs/DATASET_GUIDE.md)
- [Settings, models and paths](docs/guide/settings-reference.md), [external tools and API keys](docs/guide/installation.md#external-tools)
- [Troubleshooting](docs/guide/troubleshooting.md) and [known limitations](docs/guide/known-limitations.md)

The [documentation index](docs/README.md) includes development and plugin references.

## LoRA training FAQ

These are the questions that come up most in LoRA training threads, with what LDS does about each. Each question opens its full answer, and the [dataset guide](docs/DATASET_GUIDE.md) has the details.

### [How do I train a consistent character LoRA from one or a few photos?](docs/faq/character-lora-from-one-photo.md)

Create a **Character** dataset and set a reference photo. LDS proposes a shot catalog for the subject (face, bust, body and back framings from several angles) and generates the missing shots locally with Klein or Krea 2 Edit, which restages the reference while preserving identity. InsightFace similarity and score-based auto-triage help you reject shots where the face drifts, and the [Camera angles](bundled/camera_angles/) plugin adds new viewpoints. Your own photos work too: import them, then let [coverage](docs/DATASET_GUIDE.md#9-coverage--what-your-set-never-showed) name what the set never shows, such as profile views or a second outfit.

### [How many images do I need to train a LoRA?](docs/faq/how-many-images-to-train-a-lora.md)

About 25 varied images make a balanced character LoRA, and they beat 60 near-duplicates. Minimums depend on the family: 12 for Z-Image, 15 for Krea 2, FLUX.1 and FLUX.2 Klein, 20 for SDXL. Balance face, bust, body and back shots, because a set of close-ups gives a LoRA that falls apart on full-body prompts; the composition meter on the dataset page tracks that balance. See [how many images, and which ones](docs/DATASET_GUIDE.md#2-how-many-images-and-which-ones).

### [Tags or natural language captions? Where does the trigger word go?](docs/faq/lora-captions-tags-or-natural-language.md)

Match the caption style to the family: natural-language prose for Krea 2, FLUX.1, FLUX.2 Klein and Z-Image, booru tags for SDXL, either for Anima. LDS writes captions in that style with JoyCaption or a local vision model and puts the trigger word at the start of every caption on export. Describe what should vary (place, light, outfit, pose) and leave out what should stick to the trigger: per-trait appearance policies decide whether hair, makeup or glasses are described. See [captions](docs/DATASET_GUIDE.md#3-captions--the-make-or-break-step).

### [Which base model should I train my LoRA on?](docs/faq/which-base-model-for-a-lora.md)

Choose the family before captioning, because it sets the caption style, the image count and the settings. In short: Krea 2 has the highest realism ceiling, FLUX.1 the largest LoRA ecosystem, FLUX.2 Klein 4B trains on mid-range GPUs, Z-Image iterates fastest, SDXL suits booru-native checkpoints and Anima targets anime. Qwen-Image 2.1 is supported too. See the [family comparison](docs/DATASET_GUIDE.md#1-pick-your-model-family-first).

### [What steps, rank and learning rate should I use?](docs/faq/lora-steps-rank-learning-rate.md)

Start from the guided recipe for your family: LDS fills in the ai-toolkit settings and scales steps with the dataset, about 120 per image for a character LoRA, kept between 1,500 and 3,500, because a fixed 3,000 overcooks small sets. Raise the rank (48 to 64) only for a hard identity and a bigger dataset; a high rank on 15 images memorizes them. Advanced options stay editable, and a run can continue from any saved epoch. See the [settings cheat-sheet](docs/DATASET_GUIDE.md#4-settings-cheat-sheet).

### [Which LoRA trainer should I use? Do I need the command line?](docs/faq/which-lora-trainer.md)

LDS trains with [ai-toolkit](https://github.com/ostris/ai-toolkit). Once ai-toolkit is installed and set in **Settings → Local tools**, LDS writes the training config: pick a recipe and start the run from the browser, with the queue and logs on the runs page. Datasets also export as standard ZIPs with `.txt` caption files, so the same set can feed kohya_ss, OneTrainer or another trainer.

### [How much VRAM do I need to train a LoRA?](docs/faq/lora-training-vram.md)

Importing, sorting and manual captioning need no GPU. Auto-captioning takes about 8 GB of VRAM and local generation about 16 GB. LoRA training needs 16 GB or more for Z-Image (quantized, low-VRAM mode) and SDXL, 16 to 24 GB for FLUX.2 Klein 4B (32 to 48 GB for 9B), 24 GB for Krea 2 at 1024 px (under 24 GB, train at 768 px only) and about 24 GB for FLUX.1 and Anima. Short on VRAM? The [Cloud training](bundled/cloud_training/) plugin trains supported families on rented [vast.ai](https://cloud.vast.ai/?ref_id=683073) GPUs, and LDS itself can run on a [RunPod GPU](docs/guide/runpod.md). See the [hardware tables](docs/guide/requirements.md).

### [How do I pick the best epoch or checkpoint?](docs/faq/pick-the-best-lora-checkpoint.md)

Open the run in Test Studio: it renders the checkpoints you pick at several strengths with the same seed and prompts, so the images differ only by checkpoint and strength. Vote and rank them, compare several LoRAs or stack them, and test several prompts in one launch. See [after training](docs/DATASET_GUIDE.md#6-after-training-pick-the-right-checkpoint).

### [Can I train a style or concept LoRA?](docs/faq/style-and-concept-loras.md)

Yes: **Style** and **Concept** datasets have their own workflows. Concept captions must be prose rather than booru tags, steps grow more slowly with the image count (about 475 × √n, from 2,000 to 12,000), and the guide explains how to [keep faces out of a concept](docs/DATASET_GUIDE.md#8-concept-loras-keeping-faces-out).

### [Can I train a video LoRA (Wan 2.2, LTX-2, MiniMax H3)?](docs/faq/video-lora-training.md)

Experimentally, with the [Video lane](bundled/video/) plugin (beta). It cuts clips to the frame counts each model accepts (4n+1 for Wan, 8n+1 for LTX), measures silence and audio level, shows burned-in subtitles and letterbox bars before training, and trains with ai-toolkit locally or on a rented GPU. Nine targets are offered across Wan 2.1, Wan 2.2, LTX-2, LTX-2.3 and MiniMax H3, but only Wan 2.2 14B has been through a finished run so far, filling 24 GB with CPU offload.

### [Do I need to know ComfyUI?](docs/faq/do-i-need-comfyui.md)

You need a ComfyUI install for local generation, not its node editor. LDS sends its own workflows for dataset shots, edits, upscaling and checkpoint tests, and Setup downloads the generation models. With Docker, the launcher can set up a fresh, isolated ComfyUI.

### [How do I upscale or restore images locally?](docs/faq/upscale-and-restore-images-locally.md)

Install [SeedVR2](bundled/seedvr2/) to restore and upscale with high-resolution tiling, or [Klein Improve](bundled/image_upscale/) to re-render detail from instructions (appearance and color can change). Both run on your local ComfyUI.

## Configuration & network access

Use **Settings** for normal configuration. Native installs bind to `127.0.0.1` by default. Read the [security policy](SECURITY.md#the-default-threat-model) before enabling network access; a protected connection also lets you use LDS from a phone or tablet.

Optional usage statistics are off by default. Update checks, requested model downloads, configured providers and scraping can contact external services. The [network and privacy guide](docs/guide/network-access.md) describes these connections, the data shared and public-access settings.

## Support the project

A [Patreon](https://www.patreon.com/c/Loraperfectgf) membership (LDS Member) funds development and includes new features, exclusive plugins and LDS updates. [GitHub Sponsors](https://github.com/sponsors/perfectgf) also supports development, API testing and rented test GPUs. Bug reports, contributions and sharing the project also help. For support, generate a diagnostic report under **Guide → Getting help**, then use [Discord](https://discord.gg/j6hnJBFtXE) or [GitHub issues](https://github.com/perfectgf/lora-dataset-studio/issues).

**Affiliate disclosure:** the project's [vast.ai links](https://cloud.vast.ai/?ref_id=683073) pay the project 3% of referred users' spending for the lifetime of their account, at no extra cost. Rentals use your own API key and are billed directly by vast.ai. This is separate from optional usage statistics.

## Legal & responsible use

Use material you have the rights and consent to train on. Non-consensual likeness use, impersonation, fraud, and sexual or exploitative content involving minors are prohibited. You remain responsible for datasets and outputs. See [responsible use](docs/responsible-use.md) for consent, privacy, copyright, platform terms and warranty details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development and pull requests, and the [Code of Conduct](CODE_OF_CONDUCT.md) for community rules. Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md).

## License

[PolyForm Noncommercial 1.0.0](LICENSE). Noncommercial use is permitted; commercial use requires separate permission from the licensor.

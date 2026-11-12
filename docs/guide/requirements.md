# Requirements

[Documentation](../README.md) · [Installation](installation.md) · [Settings](settings-reference.md)

## Hardware and platform

The app scales from "no GPU at all" to a full local training rig — each capability has its own floor, and missing pieces are hidden or guided through Setup.

| Mode / capability | GPU (NVIDIA) | Disk | Notes |
|---|---|---|---|
| **API-only** (Gemini/ChatGPT/OpenRouter generation, import/scrape, curate, manual captions, export/backup) | none | ~2 GB | Any machine with Python 3.11+ (3.13/3.14 run the core app fine — the 3.11–3.12 window is an ML-extras constraint); Docker image available |
| **Auto-captioning & framing** (Ollama vision, 8B model) | ~8 GB VRAM | ~7 GB | Runs alongside generation, not concurrently |
| **Local generation** (Klein 9B **KV** fp8 via ComfyUI) | ~16 GB VRAM | ~30 GB (model + text encoder + VAE) | Free, local and NSFW-capable; Setup downloads the models. The KV build is up to **2.5× faster on multi-reference edits** at the same quality. Available in Docker GPU mode |
| **LoRA training — Z-Image / SDXL** (ai-toolkit) | 16 GB+ recommended | 10 GB+ free enforced per run | Quantized (qfloat8) + low-VRAM mode |
| **LoRA training — Krea 2** (ai-toolkit) | **24 GB VRAM** at 1024 px (enforced warning) | ~24 GB base download (Raw), or none if you start from a Krea 2 checkpoint you already have, + 10 GB+ free | Under 24 GB, select **Resolution → 768 only** in Advanced options |
| **LoRA training — FLUX.2 Klein** (ai-toolkit) | 4B: **16–24 GB VRAM** · 9B: **32–48 GB** | base download + 10 GB+ free | Both bases are gated on Hugging Face; the cloud lane is practical for 9B |
| **LoRA training — FLUX.1 / Anima** (ai-toolkit) | ~24 GB VRAM (both are 12B-class families) | base download + 10 GB+ free | **Local only — neither has a cloud lane.** FLUX.1 is gated on Hugging Face; Anima's base is public and reads booru tags natively |
| **Full-model (dense) training — Krea 2** (experimental, cloud only) | **80 GB VRAM** — there is no local lane | 200 GB on the pod, plus a private Hugging Face repo for the ~26 GB master | Not the recommended path: Krea's own advice is a LoRA on Raw applied to Turbo. Needs a separate `HF_CLOUD_TOKEN` |
| **Face scoring / person masks / watermark inpaint** (ML extras) | none (CPU) | ~3 GB (+ CPU torch for LaMa) | Python **3.11–3.12 required** for wheels; installable per capability from Setup |

- **OS:** Windows 10/11 for the full local stack (`start.bat`). Linux/macOS work for API-only + manual venv; GPU Docker depends on host NVIDIA support.
- **Python:** 3.11–3.12, but not required up front: `start.bat` fetches a self-contained CPython 3.12 when none is installed. Python 3.13+ can run the core app but not the ML extras.
- **RAM:** 16 GB+ recommended for local training. Unlike VRAM and free disk, this one is a recommendation the app never measures — a run that dies for want of system memory has no guard-rail in front of it.
- **Dataset size:** a launch is gated on a per-family floor — 12 images for Z-Image, 15 for Krea 2 / FLUX.1 / FLUX.2 Klein, 20 for SDXL, 4 for a slider LoRA — with 20-30 recommended. Below the floor the app asks you to confirm and warns about overfitting rather than refusing outright.
- Reference development rig: RTX 4090 (24 GB); every number above was measured or enforced there.

## Dependencies by feature

Core tools are prepared in **Setup**. Install optional features from **Plugins → Store**, then use each plugin’s settings and preparation screen. Missing dependencies keep the affected feature unavailable.

| Feature | Requires |
|---|---|
| Nano Banana Pro generation | `GEMINI_API_KEY` |
| ChatGPT / `gpt-image-2` generation | `OPENAI_API_KEY`, or the separate experimental ChatGPT-subscription connection |
| OpenRouter generation | `OPENROUTER_API_KEY` plus an image-capable model slug; OpenRouter billing and the upstream provider's policy still apply |
| Klein generation / improvement | ComfyUI reachable + Klein model stack |
| SeedVR2 upscaling | Reachable ComfyUI, the `ComfyUI-SeedVR2_VideoUpscaler` node pack and two model files (~3.9 GB). **Prepare SeedVR2** installs the nodes, dependencies and models on supported Windows portable layouts; other layouts require upstream node preparation. Re-check readiness after restarting ComfyUI. Optional `Comfyui_TTP_Toolset` enables overlapping tiling, with `always`/`never` controls. See [preparation and tiling limits](../../bundled/seedvr2/README.md) |
| Krea 2 Edit generation | ComfyUI reachable + `comfyui-krea2edit`, a Krea 2 base, Identity Edit LoRA, Qwen3-VL encoder and Qwen Image VAE; [exact files](settings-reference.md#krea-2-edit-local) |
| Captioning | A local LLM — **Ollama or LM Studio** — **or** ai-toolkit (JoyCaption) |
| Dual long + short captions | ai-toolkit + local vision caption derivation; local training only, and unavailable for Krea 2 / Anima |
| Auto-framing / auto head-crop | A local LLM (Ollama or LM Studio) with a vision model |
| Face similarity / auto-triage | `backend/requirements-ml.txt` (InsightFace + ONNX Runtime) |
| Character person masks | `backend/requirements-ml.txt` (rembg); Concept/Style intentionally disable them |
| Image Bank scoring, crops and semantic tools | The Bank scoring extra provides CLIP and ✨ Score. Each Bank can instead select the optional pinned SigLIP 2 engine from Setup; it builds a separate index, while aesthetic/NSFW/style/medium remain on CLIP. Balanced picks also need Framing. Both ship **CPU-only PyTorch** on purpose; on a machine that already has a CUDA Python (ai-toolkit's, ComfyUI's) each can be pointed at it instead — checked package by package, never installed into, and separately for ✨ Score and for SigLIP 2. |
| Watermark detection | A local LLM (Ollama or LM Studio) with a vision model, **or** the dedicated detector (torch + transformers — the bank-scoring extra's environment is reused when present — plus ~0.9 GB of model downloads at first use) |
| Watermark inpainting | LaMa extra from `backend/requirements-ml.txt`, or ComfyUI + Klein, which erases the found zones and re-renders the whole photo; crop remains model-free |
| Scraping | `backend/requirements-scrape.txt`; Pexels also needs `PEXELS_API_KEY` and explicit authorization. Gallery/URL scanning goes through gallery-dl for any site it recognizes, whatever its bundled extractors cover; an unrecognized site returns "No images found" in the picker (the single item gallery-dl's yt-dlp fallback can still fetch is video-typed, so it never reaches the image list), and a listing of albums returns one cover per album unless **Scan full albums** is ticked. A scan that was cut short — by the time budget, a result cap, or a source that blocked or rate-limited it — now says so under the results ("this scan stopped before the end of the listing"), instead of presenting a partial list as the whole thing. Web image search needs no key — it queries a metasearch layer over several backends and asks for photos, but the filter is not honored uniformly, so some non-photo results can still come through; results are capped per search rather than guaranteed — a request for the 120 maximum routinely comes back with far fewer — come from third-party sites whose licence is your responsibility, and a few links — mainly stock-photo CDNs that redirect to the actual file — are refused by the hardened fetch that protects every import |
| Video Bank — reading and triaging | `backend/requirements-ml.txt` (PyAV). Shot detection additionally needs `transnetv2-pytorch` (weights bundled, nothing to download), which rides the bank-scoring environment because it pulls torch. The three pieces install and fail **apart**, and Setup reports them as three separate rows |
| Video Bank — cutting clips into a dataset | An ffmpeg binary: `imageio-ffmpeg` ships one, or any ffmpeg on PATH. Needed **only to promote** — without it you can still scan, detect shots, watch and triage a whole bank |
| Video Bank — shot captions and scene search | The Bank scoring extra's environment (torch + `transformers` ≥ 4.57) plus a Qwen3-VL checkpoint downloaded at first use; the model is a setting, and the same environment serves ✨ Score, SigLIP 2 and the watermark detector |
| Civitai scanning | `backend/requirements-scrape.txt`; without `CIVITAI_API_KEY` the scan runs but returns SFW results only |
| 🌐 Civitai top prompts (Studio/Canvas) | Browsing needs nothing; reading the prompts needs `CIVITAI_API_KEY` (free account) — the same key Civitai scanning uses |
| 📷 Camera angles | ComfyUI reachable + the Qwen-Image-Edit stack Setup's Camera card downloads (the VAE is shared with Krea 2 Edit) |
| 🔤 Find text (bank & dataset) | The same small CPU OCR package the Video Bank's text pass uses, installed from Setup |
| Local LoRA training: Z-Image / Krea 2 / FLUX.1 / FLUX.2 Klein / Anima / Qwen-Image 2.1 | ai-toolkit; Qwen-Image 2.1 requires a recent toolkit with `qwen_image_2` support. No ComfyUI is needed for official Hugging Face bases. Krea 2 can also start from a local checkpoint discovered through ComfyUI's model tree, including a delivered full model. Ordinary fp8 weights are up-cast with a precision warning; packed ComfyUI exports are refused because the trainer cannot load their decompression tables |
| Local SDXL training | ai-toolkit + a base checkpoint discoverable in ComfyUI's model tree |
| Cloud training | `VAST_API_KEY`; supported families are shown in the launch UI. Full-model Krea 2 also needs `HF_CLOUD_TOKEN` with Krea base read and repository write access; fine-grained is recommended, global `role=write` is accepted with a warning, and read-only is rejected. A finished full model (~26 GB, plus its ~10 GB fp8 twin) is downloaded **to your machine** and verified before the rented pod is released; a copy of the master is then pushed to a private Hugging Face repository as a backup, and it can be turned off. Either copy can seed a fresh pod when you continue the run: the Hugging Face one is minutes over a datacenter link, the one on your machine costs your upload speed — and the ▶ Continue dialog shows both durations and what each one costs in rented GPU time before you pick. So you need **room on the checkpoint drive** (checked before anything is rented), and Hugging Face room only for the backup; Settings ▸ Storage lists what is taking that space |
| Quantizing a model to fp8 (Settings ▸ Storage, or a full model's card) | A Python with `torch`; the interpreter is probed before the button is enabled, so a missing package is a refusal with its pip line, not a crash thirty seconds in. Runs on the CPU, one at a time, so it never takes VRAM from ComfyUI or a training run. Your source file is never modified or overwritten; an already-quantized file, or an adapter, is refused |
| Merging a LoRA into a base checkpoint (produces a full model) | A Python with `torch` (the same one fp8 quantization uses) and room for a second copy of the base — a 26 GB Krea 2 base takes about two minutes and writes 26 GB. Refused on an already-quantized base: merge into the full-precision file, then quantize. LoRAs must name their modules the way the base names its weights (the ai-toolkit/diffusion-model convention); kohya's flattened `lora_unet_…` SDXL exports do not, and are refused by name before anything is written. The result is a **merged** model, not a trained one, and its metadata says so |
| LoRA Canvas browsing, layout, notes and diffs | No external service; generating needs ComfyUI and same-family checkpoints, continuing needs the chosen local/cloud training lane |
| Test Studio | ComfyUI reachable + assets for a supported Studio family |
| Backup/restore and ZIP/folder merge | No external service |
| Hugging Face publishing | Write-enabled `HF_TOKEN`; repositories are private by default |

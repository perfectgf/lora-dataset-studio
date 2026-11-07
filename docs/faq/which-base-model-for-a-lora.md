# Which base model should I train my LoRA on?
<!-- description: Pick the LoRA base model before captioning: Krea 2 for realism, FLUX.1 for its ecosystem, FLUX.2 Klein 4B for mid-range GPUs, Z-Image for speed. -->

Choose the model family before you caption, because it sets the caption style, the image count and the training settings. In LoRA Dataset Studio (LDS), Krea 2 has the highest realism ceiling, FLUX.1 the largest LoRA ecosystem, FLUX.2 Klein 4B trains on mid-range GPUs, Z-Image iterates fastest, SDXL suits booru-native checkpoints and Anima targets anime. Qwen-Image 2.1 is supported too, with less of it measured in LDS so far.

## The families side by side

| Family | Training base | Captions | Images (minimum, good) | Training VRAM | Lanes |
|---|---|---|---|---|---|
| Z-Image | Z-Image-Turbo, or a converted custom merge | Prose | 12, 20+ | 16 GB+ | Local, cloud |
| SDXL | A checkpoint from your ComfyUI | Booru tags | 20, 30+ | 16 GB+ | Local only |
| Krea 2 | Krea-2-Raw (default), Turbo, or a Krea 2 checkpoint on your disk | Prose | 15, 20+ | 24 GB at 1024 px | Local, cloud |
| FLUX.1 | FLUX.1-dev (gated) | Prose | 15, 20+ | About 24 GB | Local only |
| FLUX.2 Klein | FLUX.2-klein-base 4B (default) or 9B, both gated | Prose | 15, 20+ | 4B: 16–24 GB, 9B: 32–48 GB | Local, cloud |
| Anima | Anima-Base-v1.0-Diffusers (public) | Prose or tags | 12, 20+ | About 24 GB listed | Local only |
| Qwen-Image 2.1 | Comfy-Org/Qwen-Image-2.1 | Prose | 12, 20+ | Not measured locally; cloud uses 32 GB+ | Local, cloud |

Z-Image is the default family for new runs. "Cloud" means the [Cloud training](../../bundled/cloud_training/) plugin, which rents a GPU for each run.

## Notes per family

- **Krea 2.** The default trains on Krea-2-Raw, following the official advice to train on Raw and validate on Turbo. Raw runs take hours by design, and the best checkpoint tends to land in the last third. Turbo is the faster fallback for drafts. The **Base** selector also lists Krea 2 checkpoints from your ComfyUI model folders; an entry tagged `· fp8 cast` trains from degraded weights, and one tagged `· packed export` cannot be loaded at all.
- **FLUX.1.** FLUX.1-dev is gated on Hugging Face: accept its license and set a Hugging Face token before the first run, which downloads about 24 GB. Drop the resolution to 768 to fit a smaller card.
- **FLUX.2 Klein.** The 4B and 9B sizes are picked next to the base selector, and both bases are gated. The 9B is best trained in the cloud.
- **Z-Image.** A distilled base that converges fast; watch the last fifth of a run for waxy skin and a frozen expression.
- **SDXL.** Trains on a base checkpoint that ComfyUI can see, captioned in booru tags. Its default alpha is half its rank, the family's half-strength convention.
- **Anima.** The one family with hybrid prompting: prose and booru tags are both native, so the caption-style check never flags it. Its base is public, with no gated download.
- **Qwen-Image 2.1.** Needs a current ai-toolkit with the dedicated `qwen_image_2` architecture; the older Qwen Image architecture is incompatible. The recipe uses rank 32, 768 and 1024 px resolutions and 40-step previews at CFG 3. Cloud runs use at least 32 GB of VRAM on an Ampere-or-newer GPU and 100 GB of disk, and show no duration or cost estimate yet.

## Changing your mind later

The target model can be changed after the dataset is created. Each family remembers its own base and variant on the dataset, so switching back restores your choice. Captions do not convert themselves: moving between a prose family and SDXL means re-captioning, or forcing the launch past the caption-style check.

Every family can be tested in the Test Studio, which needs ComfyUI. FLUX.1, Anima and Qwen-Image 2.1 need their generation models prepared first in **Settings → Image engines → Trained image models**.

## Limits

- Qwen-Image 2.1 GPU memory has not been measured in LDS, and reference-image editing and transparent (RGBA) datasets are not available for it.
- SDXL, FLUX.1 and Anima have no cloud lane.
- Full-model training, which trains the whole transformer instead of a LoRA, is Krea 2 only, experimental and cloud only, and needs 80 GB of VRAM.

## Related

- [Tags or natural language captions? Where does the trigger word go?](lora-captions-tags-or-natural-language.md)
- [How much VRAM do I need to train a LoRA?](lora-training-vram.md)
- [Pick your model family first (dataset guide)](../DATASET_GUIDE.md#1-pick-your-model-family-first)
- [Qwen-Image 2.1 training](../DATASET_GUIDE.md#qwen-image-21-training)

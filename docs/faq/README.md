# LoRA training FAQ
<!-- description: Answers to common LoRA training questions: image counts, captions, base models, steps, VRAM, trainers, checkpoint picking, style, concept and video LoRAs. -->

These pages answer the questions that come up most in LoRA training threads, with what LoRA Dataset Studio (LDS) does about each; the [dataset guide](../DATASET_GUIDE.md) has the full detail.

- [How do I train a consistent character LoRA from one or a few photos?](character-lora-from-one-photo.md) Generate about 25 varied shots from one reference photo with Klein or Krea 2 Edit on your ComfyUI, then reject the shots where the face drifts.
- [How many images do I need to train a LoRA?](how-many-images-to-train-a-lora.md) About 25 varied images make a balanced character LoRA; the minimum is 12 for Z-Image, 15 for Krea 2, FLUX.1 and FLUX.2 Klein, and 20 for SDXL.
- [Tags or natural language captions? Where does the trigger word go?](lora-captions-tags-or-natural-language.md) Prose for Krea 2, FLUX.1, FLUX.2 Klein, Z-Image and Qwen-Image 2.1, booru tags for SDXL, either for Anima, with the trigger word first, added by LDS.
- [Which base model should I train my LoRA on?](which-base-model-for-a-lora.md) Krea 2 has the highest realism ceiling, FLUX.1 the largest LoRA ecosystem, FLUX.2 Klein 4B fits mid-range GPUs, Z-Image iterates fastest, SDXL suits booru checkpoints and Anima targets anime.
- [What steps, rank and learning rate should I use?](lora-steps-rank-learning-rate.md) About 120 steps per image for a character, kept between 1,500 and 3,500, with rank 16 or 32 by family and a learning rate of 1e-4.
- [Which LoRA trainer should I use? Do I need the command line?](which-lora-trainer.md) LDS drives ai-toolkit and writes its config, so runs start from the browser once ai-toolkit is installed; exported ZIPs also feed other trainers.
- [How much VRAM do I need to train a LoRA?](lora-training-vram.md) 16 GB or more for Z-Image and SDXL, 16 to 24 GB for FLUX.2 Klein 4B and 24 GB for Krea 2 at 1024 px; sorting and manual captioning need no GPU.
- [How do I pick the best epoch or checkpoint?](pick-the-best-lora-checkpoint.md) Compare checkpoints at a fixed seed in the Test Studio and keep the earliest one that holds the identity.
- [Can I train a style or concept LoRA?](style-and-concept-loras.md) Yes: Style and Concept datasets have their own captioning rules and step policies, and concepts can mask faces out of the loss.
- [Can I train a video LoRA (Wan 2.2, LTX-2, MiniMax H3)?](video-lora-training.md) Experimentally, with the beta Video lane plugin; only Wan 2.2 14B has been through a finished run so far.
- [Do I need to know ComfyUI to use LoRA Dataset Studio?](do-i-need-comfyui.md) You need a ComfyUI install for local generation and checkpoint tests, but never its node editor.
- [How do I upscale or restore images locally?](upscale-and-restore-images-locally.md) Use the SeedVR2 plugin to restore with tiling, or Klein Improve to re-render detail, both on your local ComfyUI.

# Changelog

Older entries rolled out of the README's [Recent improvements](README.md#recent-improvements)
section, newest first. Everything below is live on main.

## Rolled 2026-07-18 (shipped mid-July 2026)

- **☁🎚 Slider LoRAs go cloud** — slider training is no longer local-only: the cloud lane now accepts `concept_slider` jobs, and slider settings are **snapshotted at launch** so a mid-run edit can never retarget a rented run. The first paid slider run is still unproven — treat cloud sliders as extra-Beta.
- **🖼 Flip through Test Studio results** — the result lightbox now navigates: swipe on touch, **‹ ›** buttons and **arrow keys** on desktop, with an *i / n* counter and wrap-around. Strength variants of the same render sit **adjacent** in the order, so flipping compares strengths directly.
- **🗂 Denser dataset library** — pick your tile density (**S/M/L**, where S is a one-line compact list), collapse whole family sections (persisted, and forced open while a search or filter is active), and filter by kind with **Character / Concept / Style** chips.
- **🎚 Slider LoRAs (Beta)** — a new training mode on any dataset that learns a single **bipolar** LoRA from a prompt pair (positive vs negative pole), so one adapter dials a trait up or down at inference. All five families are offered behind honest per-family experimental notes (Krea 2 is the reference), it runs **locally only** for now, and it's Beta — expect to iterate. The Test Studio can now sweep **negative strengths (−2.0 → 4.0)** to exercise both poles.
- **☁ Train on custom bases in the cloud** — Z-Image, Krea 2 and FLUX.2 Klein custom weights are no longer local-only: a one-time push uploads the base to a **private** repo on your own Hugging Face account (private enforced, cached by hash so it never re-uploads), then the pod pulls it with your token. Official-base cloud runs are bit-for-bit unchanged; SDXL and FLUX.1 keep their local-only path.
- **🖼 Test Studio, sharper** — export the checkpoint × strength grid as a single **labeled, shareable image** (title banner with model/CFG/steps/seed) to post on Civitai or Reddit, turn any dropped image into a test prompt with **🔎 Describe** (local Ollama vision), and sweep strengths up to **4.0** to find a LoRA's over-cook point.


- **📂 ComfyUI `extra_model_paths.yaml` support** — models that live outside `models/` (portable builds, Stability Matrix) are now resolved exactly as ComfyUI sees them, across Klein generation, Setup probes, the model picker, the installer's already-present check and Studio preflight. Without a yaml, behavior is unchanged.
- **🩺 Sturdier captioning** — vision calls re-encode WebP to a JPEG Ollama can actually decode, JoyCaption's first-run model download streams into the log with a visible "downloading model" stage, and the vision-model probe reads Ollama's tags however `/api/tags` reports them so a pulled model is never falsely "not installed".


- **📑 Fifteen researched training presets** — a **Built-in (researched)** group ships a Character, Style and Concept preset for each of the five families (Z-Image, SDXL, Krea 2, FLUX.1-dev, FLUX.2 Klein). Every value — rank/alpha, timestep, resolution, save cadence — is sourced from ai-toolkit's own defaults, vendor guidance or documented community consensus, each preset explains *why* in one line, and one click applies the whole recipe through the normal validation.
- **🎨 Research-backed Style recipes** — Style is explicitly always-on, with no activation trigger. Training combines five family-specific presets with content-only caption rules, family/variant-aware step limits, and launch guards for missing, trigger-only or identical captions.
- **✨ Klein improvement, single or bulk** — multi-select eligible images and queue separate **2 MP** Klein candidates in one pass, or improve one from its lightbox. Progress and failures are reported for the batch, existing candidates are skipped, every original stays untouched until review, and both flows share one instruction under **Settings → Scraping & sources**. Klein restoration is generative and may alter fine details, which is why improved images never enter training without explicit validation.
- **🛡 Safer training launches** — family, base and variant recipes are revalidated for local, queued, continued and cloud runs. The Runs hub can stop an identified local run as well as a cloud run while preserving checkpoints already written.
- **📝 Better caption editing** — expand any caption into a larger editor with a character count and **Ctrl/⌘ + Enter** save. Frequency tools and re-caption guidance now adapt to character, concept or style datasets and to prose vs booru captions.
- **🩹 Editable watermark corrections** — move/resize detected boxes or add missed zones in **Review flagged**. LaMa cleanup can use **Auto, GPU (CUDA), or CPU** from Settings.
- **📦 Independent checkpoint browser** — **📦 Checkpoints & LoRAs** is a separate workspace destination between Train and Studio, with selectors independent from the next training configuration.

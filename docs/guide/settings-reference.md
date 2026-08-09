# Settings reference

Every setting in LoRA Dataset Studio, explained: what it does, its default, when to change it, and the traps to avoid.

## How settings work

Open **Settings** from the top nav. Each rail entry on the left is a section (Overview, Image engines, Scraping & sources, Local tools, Captioning & quality, Training, Server & access, Maintenance); its little LED shows live health at a glance — **green** when the section is fully configured, **amber** when it's partly set up, **off** when nothing is configured yet.

A few things hold true everywhere:

- **Nothing saves until you say so.** Change any field and a floating **Unsaved changes** bar appears with **Save** and **Discard**. Navigate away with changes pending and they're kept in the bar, not written.
- **Where values live.** Ordinary settings are written to `config.json` (git-ignored, in your data directory). Secrets — API keys and tokens — go to a separate `.env` file and are never written to `config.json` or committed.
- **Secret fields are write-only.** An API-key box is always blank, even when a key is saved (a ✓ *Configured* badge tells you it's there). Typing a new value replaces the old one; **leaving a field blank never erases a saved key** — that would be too easy to do by accident. To actually remove a key, use its **Remove** button.
- **Test buttons probe what's saved, not what's typed.** Hitting **Test** first persists whatever you've typed, then tests the *saved* setting end-to-end. So a Test result always reflects the value the app will really use.
- **Server changes need a restart.** Host, port and the access token only take effect when the server process starts. Those fields show a **Running vs Saved** comparison and a **Save & restart to apply** button. Everything else — including scraping credentials — applies immediately, no restart.
- **Every field can go back to its shipped value.** Change a number, a path, a dropdown or the enabled-engine list and a small **↺ Reset to default** button appears right under it; it disappears again once the field is back where it started, so a button you can see is always a button that does something. The value it restores is read from the server, not from a copy inside the page — so if a shipped default changes in a later release, Reset gives you the *new* default rather than the one your version was built with. On the fields that mean "work it out yourself" when left blank (the engine model slugs, the Krea base model, the dataset images root), Reset empties the box rather than typing today's answer in, which is how you keep following future improvements instead of freezing one. Nothing is written until you **Save**, same as any other edit.
- **Search finds settings, not just sections.** The search box at the top of the page matches both section names *and* individual settings, so typing "budget" or "vision model" jumps you straight to the right field.

### Advanced: environment overrides

For containerized or scripted setups, a handful of environment variables override paths and binds before `config.json` is even read. You rarely need these — the UI covers the normal cases.

| Variable | Overrides |
|---|---|
| `LDS_DATA_DIR` | Runtime data directory (where `config.json`, datasets and trash live). |
| `LDS_CONFIG` | Path to `config.json`. |
| `LDS_ENV` | Path to the `.env` secrets file. |
| `LDS_HOST` | Bind host — takes priority over `server.host`. |
| `FLASK_DEBUG` | `1` enables Flask debug mode. |

## Overview

The Overview section has **no settings of its own** — it's the at-a-glance dashboard for the rest of the page. If nothing is configured yet, it opens with a *Let's get you set up* banner. Below that, a **Capabilities** grid marks each feature ✓ or ✗ depending on what the app can currently see (a key, a reachable tool, an installed extra).

Every row is a **link to the control that turns that capability on**, not just to the right screen: picking *OpenRouter* lands on the OpenRouter key field with it scrolled to and highlighted; picking *Person masks* opens the Setup wizard step that installs it. Use the grid as your first stop to answer "why is this feature greyed out?" — the answer is one click away on the row itself.

A row marked **◐ in amber** is not broken: the tool is installed, it just isn't running (typically *launch ComfyUI to enable* for Klein and the Test Studio). Those rows lead to the **ComfyUI API URL** field and its **Test** button rather than to an install you have already done. The counter at the top reads `X/11 ready` plus, when it applies, how many are waiting on a process.

If nothing on the grid tells you where to start, the line at the bottom opens the **Setup wizard**, which scans the machine and installs what it can.

## Image engines

This is where you connect the services that *generate* dataset images. The app has five engines: **Nano Banana** (Google Gemini), **ChatGPT** (OpenAI), **OpenRouter** (one account in front of many providers), and two that run locally through ComfyUI — **Klein** and **Krea 2 Edit**. Each of the three API engines lets you choose which model it asks for — see *Image models* below. ComfyUI itself is configured under **Local tools**; the API keys and the two local engines' own knobs are configured here.

### API keys

- **Gemini API key** — powers the Nano Banana engine. Paste it here and hit **Test** to confirm the key works. Get one from [aistudio.google.com](https://aistudio.google.com) → *Get API key*.
- **OpenAI API key** — powers the ChatGPT engine (`gpt-image-2`). **Test** confirms it. This key is **optional if you connect a ChatGPT subscription** below — the subscription lane can run the ChatGPT engine on your plan's image quota instead.
- **OpenRouter API key** — powers the OpenRouter engine. Get one from [openrouter.ai/keys](https://openrouter.ai/keys). One account and one balance in front of most providers, *including the same models the two engines above call directly* — so it is the way in if you would rather not open an account per provider. **Test** only checks that a key is saved: OpenRouter bills per request, so the app never spends a credit just to light up a checkmark.

All three are write-only secrets: blank once saved, replaced by typing a new value, cleared only via **Remove**.

### Image models

One field per API engine — you choose the model each one asks for:

- **Nano Banana (Gemini) model** → `engines.nanobanana_model`. Blank = **`gemini-3-pro-image`**, the model this engine has always used. Note that **no model choice changes Google's output filter** — see *What the Gemini engine will and will not do* below.
- **ChatGPT (OpenAI) image model** → `engines.chatgpt_image_model`. Blank = **`gpt-image-2`**, the model this engine has always used.
- **OpenRouter model slug** → `engines.openrouter_model`. Blank = **`google/gemini-3-pro-image`** — the same weights the Nano Banana engine calls, so switching engine changes who bills you, not what the pictures look like.

All three are **free text on purpose**: providers publish image models far faster than this app publishes releases, and a dropdown frozen into a build would be out of date the day it shipped and would lock you out of a model that works. Leaving a field blank keeps that engine's historical model, so a field appearing here changes nothing about your results.

### What the Gemini engine will and will not do

Two properties of Nano Banana that no setting on this page can change. They are here because both are easier to meet in advance than to diagnose afterwards.

**Google screens the image, and that screen has no switch.** Gemini checks the picture it has just produced. When that check trips, the API answers **HTTP 200 with no image** — a success envelope with nothing in it. LDS reports each one as a refusal on the tile, relays Google's own reason code (`IMAGE_SAFETY`, `PROHIBITED_CONTENT`…) when it gives one, and tells you at the end of a run how many were refused as opposed to how many genuinely failed. What it cannot do is stop them:

- the four adjustable safety categories in the Gemini API act on the **prompt**. Nothing — no threshold, no `BLOCK_NONE`, no `OFF` — turns off the screen applied to the **returned image**, and Google does not document it. There is no setting for LDS to offer;
- it has **many false positives**: everyday requests get refused, and the trip point is not something you can reason about from your prompt text;
- it is **not deterministic**: the same prompt can pass on one attempt and be refused on the next. This is why neither the app nor this page tells you to retry or reword — that would be selling a coin toss as a remedy.

A refusal never stops a batch: the remaining rows still get their attempt, and the count at the end is exact.

**Adult content is not allowed on this engine.** Google's usage policy forbids it, with consequences up to restriction of your Google account. LDS is fail-closed on this: NSFW variations are never sent to an API engine — they exist only on the local **Klein** path. Nothing here is a way around the filter; it is a statement of which engine does what.

**Every Gemini output carries SynthID.** Google applies its invisible provenance watermark to 100% of the images this engine returns. If your dataset is destined for training, that is a material property of your data and you should know it is there. What effect it has on trained LoRA weights is **unmeasured** — nobody has established that it degrades a LoRA, and nobody has established that it does not. LDS states its presence and makes no claim beyond that. Images from **Klein** and **Krea 2 Edit** (local ComfyUI) carry no SynthID.

**Where the value comes from**, in order:

1. what you type in this field;
2. the `NANOBANANA_MODEL` / `CHATGPT_IMAGE_MODEL` environment variable, if you had set one (these existed before the fields did — your choice is still honoured, and is only overridden when you actually type a slug here);
3. the built-in default above.

A model typed here applies to the **next generation** — no restart.

The model **must accept reference images** — the dataset generator always sends your reference photo(s) with the prompt, so a text-to-image-only model is not usable here. When a provider refuses one, the failed tile names the model and repeats the provider's own reason (unknown model, model that will not take image input, key rejected, organization not verified), and the run **stops** instead of asking the same refused question once per image. What no app can catch for you is a model that *accepts* the references and then ignores them: if generated faces stop resembling your subject right after a model change, change it back.

Two provider-specific traps:

- **OpenAI: `gpt-image-2` is the only current model usable without organization verification.** `gpt-image-1.5` and `chatgpt-image-latest` answer **403** until your OpenAI organization is verified — that is the *model* refusing, not your key. The failed tile says so and names the model to fall back to.
- The ChatGPT **subscription** lane ignores this field: it renders through OpenAI's own image tool on whatever model your plan serves. `engines.chatgpt_subscription_model` is a different setting again — the Codex *router* model of that lane, which decides nothing about the pixels.

How many references a model takes varies (roughly 1 to 16 depending on the provider); the app sends every reference you gave it and, if the model refuses the request, says so and mentions the count rather than quietly dropping references you expected to be used.

What OpenRouter does **not** change:

- **Not cheaper by itself.** You still pay per image, at that model's rate, out of your OpenRouter credits.
- **Not less restricted.** OpenRouter forwards to the same upstream providers, so the same content policies apply. NSFW variations still run on a local engine only.
- **No subscription lane.** OpenRouter is credit-based; there is no equivalent of the ChatGPT-plan option below.

When a generation fails, the tile names the cause in OpenRouter's own words — no key saved, key rejected, out of credits, unknown model, rate-limited. The four causes that would fail every remaining image identically (no key, rejected key, no credits, unknown model) **stop the rest of the batch** instead of asking the same refused question once per image. The app never falls back to another engine behind your back: if you picked OpenRouter, only OpenRouter is billed. A moderation block arrives *inside* a successful response rather than as an error code, so it is read out of the body and shown with the reasons the provider gave — a refused image costs that row, not the run.

### ChatGPT subscription (experimental)

If you have a ChatGPT Plus/Pro plan, you can run the ChatGPT engine on your subscription's image quota instead of a pay-per-use API key. This uses the same sign-in lane as OpenAI's Codex CLI — it is **not a documented API and may stop working at any time**; you connect your own account at your own risk.

- **Connect with ChatGPT** — starts an OAuth device-code sign-in; the badge then shows the connected account's email.
- **Import from Codex CLI** — appears only if the app detects an existing `codex login` on this machine, and reuses that session.
- **Disconnect** — signs out of the subscription lane.
- **ChatGPT engine auth** → `engines.chatgpt_auth`. Chooses which credential the ChatGPT engine uses. Default **`auto`**.

| Value | Behaviour |
|---|---|
| `auto` *(default)* | Use the subscription when connected, otherwise fall back to the API key. |
| `api` | API key only — ignore the subscription. |
| `subscription` | Subscription only — never touch the API key. |

Good to know: in subscription mode you get up to **5 reference images** per generation (versus 16 on the API), your plan's image cap applies, and when the quota runs out mid-batch the remaining rows fail with a clear message — **the app never silently switches to your paid API key**.

**When a generation fails on this lane**, the tile names the cause rather than showing a blank "empty response": a network drop or timeout, an OpenAI outage, a plan quota, a connection that needs reconnecting, a refusal by OpenAI's safety system — and, because this lane is undocumented, the case that matters most for it: *OpenAI is no longer serving image generation on this ChatGPT subscription*. That last one stops the run and points at API-key mode, which is the only way back. No message suggests retrying: whether the same call would pass a second time is exactly what the app cannot know. If OpenAI answers without an image **and** without a reason, the tile says so instead of picking a cause.

### Engines

- **Default engine** → `engines.default`. Which engine is preselected in the workspace. One of `nanobanana`, `chatgpt`, `openrouter`, `klein`, `krea`. Default **`chatgpt`**.
- **Enabled engines** → `engines.enabled`. Checkboxes deciding which engines appear as options at all. Default: **all five** enabled. Untick an engine you never use to declutter the generator picker. An engine added by a later update is offered here automatically, even on an install whose settings were saved long before it existed — while an engine you unticked on purpose stays unticked, because the app records which engines it was showing you at the moment you chose.

#### Using several engines in one batch

In the workspace the engine cards are **checkboxes**, not a one-of-five choice: tick as many as you want. Each engine has its own colour (Klein indigo, Krea 2 Edit violet, Nano Banana amber, ChatGPT sky, OpenRouter fuchsia) and every generated tile is labelled with the engine that made it, so a mixed batch stays readable.

From **two** engines on, a mode appears deciding what "several engines" means:

| Mode | What it does | Images produced |
|---|---|---|
| **Split across engines** *(default)* | Each selected shot goes to **one** engine, round-robin. | Same as one engine — 25 shots stay 25 images. |
| **All engines** | **Every** engine renders **every** shot. | Multiplied — 25 shots on 3 engines = 75 images. |

Split gives you a more varied dataset for the price you already pay; All engines is for **comparing** engines on identical shots before keeping the best. The image count and estimated cost update live as you tick, and the Generate button carries the real total.

Good to know:

- **The cost shown is the whole run's.** The two local engines contribute nothing (it's your GPU) and neither does ChatGPT on the subscription lane.
- **The local engines run last and in series.** The API batches start immediately in the background; the local GPU handles its own shots one at a time behind them. The Klein and Krea cards say so while a mixed run is being set up.
- **There is a cap per batch** (60 images in flight on one dataset). A run over it is refused *before* it starts, with the number named — switch to Split, untick an engine, or select fewer shots.
- **🔞 NSFW shots stay local-only.** The uncensored catalog unlocks only when **every** ticked engine is local (Klein, Krea 2 Edit, or both), because those shots must never reach a third-party API.
- **Regenerating one tile** (🔄) uses the **first** ticked engine, not all of them.

### Krea 2 Edit (local)

The second local engine. Where Klein *restages* your reference with a general instruction-edit model, **Krea 2 Identity Edit** is trained specifically to keep an identity: from a **single** reference photo it holds the face, the body and the permanent markings while changing the angle, framing, light, background and clothes — **with no character LoRA**. That is what makes it useful *before* a LoRA exists, which is the whole point of building a dataset.

It is not installed by the app. It needs, inside your own ComfyUI:

- the **[comfyui-krea2edit](https://github.com/lbouaraba/comfyui-krea2edit)** custom-node pack in `custom_nodes/` (no Python dependencies), then a ComfyUI restart;
- a **Krea 2 Raw or Turbo** base model under a `krea`-named folder in `models/diffusion_models` (or `models/unet`) — from [Comfy-Org/Krea-2 ▸ diffusion_models](https://huggingface.co/Comfy-Org/Krea-2/tree/main/diffusion_models) (public, no account needed; `krea2_turbo_fp8_scaled.safetensors` is the usual pick);
- the **Krea 2 Identity Edit LoRA** in `models/loras` — from [Civitai](https://civitai.com/models/2761113);
- the **Qwen3-VL 4B** text encoder in `models/text_encoders` and the **Qwen Image VAE** in `models/vae` — both from the same [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2) repo, under `text_encoders/` and `vae/`. Keep the filenames as published: `qwen3vl_4b_fp8_scaled.safetensors` and `qwen_image_vae.safetensors`. The other Qwen encoders (`qwen_2.5_vl_*`, `qwen_3_8b_*`) belong to different models and are deliberately never picked up here.

The engine card in the workspace names whichever of these is still missing, one actionable line at a time, and the app never guesses a download URL for weights it cannot verify. Every path above is found by *searching* your ComfyUI model roots — including any `extra_model_paths.yaml` roots — so a non-standard layout works untouched.

Settings. The **four calibration dials** below (reference grounding, sampler steps, reference pull, identity LoRA strength) are editable **both here and in the 🧬 Krea 2 Edit tuning panel** of the workspace's *Generate variations* screen — same key, same endpoint, one value: whichever slider you move, the other screen shows the new number and every future Krea run uses it. The two **file-path** fields are on this page only, because they are filled once at install rather than adjusted while judging an image.

- **Reference grounding** → `krea.grounding_px`. Range `512`–`1536`, default **`1024`**. **The** dial of this engine: the resolution your reference is shown to the model's vision encoder at. At the low end it follows the shot description (more variety in pose, outfit and scene, looser likeness); **higher** values favor reference likeness and can copy the very pose and outfit you asked it to change. **1024 is the identity-first profile** shipped since the v4 calibration, paired with reference pull `4.0` — the two always ship together. **Lower it to `512` when you want variety more than likeness**: that was the v3 default, and it keeps the prompt and the selected catalog card more firmly in charge. This is Krea-only: it does not change ChatGPT, Gemini/Nano Banana, OpenRouter, or Klein.
- **Sampler steps** → `krea.steps`. Range `1`–`50`, default **`12`**. This base is guidance-distilled, so more steps mostly buy waiting; `8` — the value the model's own reference workflow uses, and the v3 default — remains a perfectly good choice and renders faster. `12` ships because it is the value the v4 calibration was measured at, not because 8 was found wanting. *(This line said "default 10" for a while, then "8"; the shipped default is now 12.)*
- **Base model file** → `krea.base_model`. **This is the GENERATION setting only** — the checkpoint ComfyUI loads for Krea 2 Identity Edit. It has **nothing to do with LoRA training**, which never reads it: training pulls its base from Hugging Face and picks it from the **Krea 2 training base** dropdown in the training panel (**Raw**, the default and the official recommendation — you train on Raw and apply the LoRA on Turbo at inference). Nobody can accidentally train on Turbo by leaving this field alone. *(The naming confusion was raised by strouder, GitHub #19.)* Blank (default) = the app picks a Krea 2 **Turbo** then **Raw** build from your ComfyUI. Set it only if you own several. Checkpoints that merely carry "krea" in their name but are not Krea 2 bases are **skipped on purpose** — the identity LoRA renders pure noise on them, which looks like a broken app rather than a wrong file. The field is a **searchable list of the Krea bases actually on disk** — the same candidates the app itself would elect, so a checkpoint it refuses is never offered; you can still type a name for a file you have not downloaded yet. **A name that is not on disk stops the engine** rather than falling back to automatic election: this exact fallback once had a whole training run on a third-party finetune nobody chose, because the field showed one file and the graph loaded another. **When the field is blank, the line under it names the build your runs actually load** — with two Krea checkpoints whose filenames both read as "turbo", the tie-break used to pick one silently, and the only way to find out was reading a finished PNG's metadata, long after the quality judgements had been made on a model nobody chose. The name is resolved server-side by the same code the generation path calls, never re-ranked in the browser. Clear the field to go back to automatic detection.
- **Identity edit LoRA** → `krea.identity_lora`. A searchable list of the LoRAs on disk (typing a path still works). At its **shipped default** — the canonical download name, which nobody typed — nothing is there under that name means the app searches your LoRA folders for a `krea2_identity_edit` file, so a renamed download still works. **A name you chose yourself and that is not on disk stops the engine instead**, naming the file: the identity LoRA *is* the face transfer, and quietly substituting another one changes every image you generate. Clear the field to return to the automatic search.
- **Krea 2 Edit generation LoRA presets** → `krea.generation_lora_presets`. Named,
  ordered combinations of **your own** LoRA files, chained after the identity-edit
  LoRA when Krea 2 Edit generates dataset images. Max 8 LoRAs per preset, 12
  presets; inside a preset the row order **is** the chain order. Per run you pick
  one preset in the workspace's 🧬 Krea 2 Edit tuning panel — which opens on
  **Preset selected by default** → `krea.default_generation_lora_preset` (default:
  empty = *None*), a starting point you can override for a single run without
  rewriting the setting — the run sends only the preset's NAME and the app resolves
  the files from this list, so renaming a preset can never make a run load
  something you didn't configure. Strength runs to 6, and to **20** for utility
  LoRAs whose filename says `filter-bypass`: those have no measurable effect below
  about 10. Limits worth knowing: the LoRA must be trained for **Krea 2** (another
  architecture loads as a silent no-op — the picker badges it), only the **model**
  side is patched so a LoRA's text-encoder weights are ignored, a row whose file
  has moved is **skipped** with the rest of the chain still applied, and the 🔄
  single-image regenerate in the workspace does **not** carry a preset. A row
  pointing at the **same file as Identity edit LoRA** is skipped too — it would
  chain the identity LoRA a second time on top of itself, summing both strengths
  into one delta well past what the file was trained for (visible as blocky,
  posterized output, not a subtler quality loss) — **the preset editor flags that
  row as you write it** rather than leaving the only trace in the server log, and
  it compares paths the way the server does, so a different separator or
  capitalisation cannot dodge the warning. Empty by default. *(Preset
  mechanism by @waltm, Discord.)*

Two more Krea calibration dials, on this page **and** in the **🧬 Krea 2 Edit tuning** panel of the workspace's *Generate variations* screen — that panel is where you judge what they do, so it is where you should be able to turn them. They were reachable only from there for a while, and reference grounding only from here, which meant "where do I change this?" had a different answer per dial. They are settings either way: moving a slider on either screen saves the value globally and it applies to **every** Krea run from now on. The panel says so above the sliders.

- **Reference pull** → `krea.ref_boost`. Range `0`–`10`, default **`4.0`**. How hard the source latent is pushed back into the model at every denoising step. This is the lever for *"the subject doesn't look enough like my reference"* — raise it and likeness sharpens; raise it far and the model also recopies the composition, pose and outfit the shot card asked it to change. It ships **paired** with `grounding_px` and always has (v1 = 1024 / 4.0, v2 = 512 / 1.0, v3 = 512 / 0.25, **v4 = 1024 / 4.0**), so moving one dial alone puts you on a combination no shipped profile ever calibrated — this is the dial that brings the pair back into balance. **What the v4 default rests on, stated so you can judge it:** a benchmark on a single reference, four scored images per profile, where 1024 / 4.0 led by +0.17 face similarity on bust framing with no overlap between runs — and did so at a *lower* pull than the setup it beat, so the extra likeness was not bought by recopying. Face-framing cards were dominated by seed noise and measured nothing; body framings produced no face score at all. It is a deliberate choice for identity over variety, not a proven optimum.
- **Identity LoRA strength** → `krea.identity_lora_strength`. Range `0`–`1.5`, default **`1.0`**. The weight applied to the Krea 2 identity-edit LoRA itself — the piece that carries the face across. Below 1 softens the likeness and leaves more room for the prompt; `0` disables the face transfer entirely; above 1 is past the weight the file was trained for and can look waxy or posterized rather than simply more similar.

Both are re-clamped by the server on every run, so a hand-edited `config.json` out of range is corrected rather than obeyed.

Two behaviours worth knowing before you build a dataset with it:

- **The selected card's framing is honored.** Krea Fit v1.2 uses the selected catalog card's framing and aspect ratio (including its 1:1 / 3:4 shape) instead of copying the source photo's shape.
- **The dataset's extra reference images are never used by Krea** — not when generating variations, and not when editing. Identity comes from the primary reference alone. Klein and the API engines are the ones that read those extra angles to strengthen identity.
  **Krea's one spare slot lives in the ✦ Edit reference dialog instead.** Add an image there with **+** and it becomes the second input of that single edit. The edit model was trained on two-input edits where the second image is a *different* subject — another person, or a scene to place yours in ("scene first, subject second"). So use it to compose: *"put her in this room"*, *"next to this person"*. Another angle of the same face is off-label there and can come back duplicated, which is exactly why that slot is not wired to the dataset's angles.

Outfits and expressions are steered differently here than on the other engines: this model preserves anything it is not *positively* told to change, so the catalog's "a different outfit (not the one in the reference)" phrasing is rewritten at generation time into a concrete garment ("wearing a red knit sweater"), picked from the shot's own name — so outfits genuinely differ across the dataset while regenerating one shot reproduces its own.

### SeedVR2 upscaling (local)

*Requested by SurpassHR ([GitHub #32](https://github.com/perfectgf/lora-dataset-studio/issues/32)).*

The **fidelity** half of ✨ Upscale & improve. The two passes are a choice, not two qualities of the same thing:

| | what it does | when you want it |
| --- | --- | --- |
| **Klein** | re-renders detail and texture from a prompt | a genuinely soft or low-detail photo you are willing to see changed |
| **SeedVR2** | resolves detail at a higher resolution, content untouched | the frame is right and you only want it sharper — the exact skin tone, grain and colour are part of what you are training |

Both are **non-destructive**: they create a separate candidate and never touch the source file.

**What it needs** (Setup ▸ ComfyUI ▸ *SeedVR2 — optional fidelity upscaler* handles the models):

- the **[ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler)** node pack (Apache-2.0) in ComfyUI, then a ComfyUI restart. **The app does not install this one for you**, unlike the Krea pack: it pulls thirteen Python packages that have to land in ComfyUI's own environment, and a bare copy of the folder would fail to import. Install it from ComfyUI-Manager (search "SeedVR2"), which does the dependencies properly.
- two model files in `<ComfyUI>/models/SEEDVR2` — from [numz/SeedVR2_comfyUI](https://huggingface.co/numz/SeedVR2_comfyUI) (Apache-2.0, public, no account): `seedvr2_ema_3b_fp8_e4m3fn.safetensors` (3.4 GB) and `ema_vae_fp16.safetensors` (0.5 GB). The Setup card downloads both on a click; an `extra_model_paths.yaml` root for `SEEDVR2` works too.

Settings:

- **Default engine for ✨ Upscale & improve** → `improve.engine`. One of `klein`, `seedvr2`. Default **`klein`** — what every improve did before this setting existed. It governs the ✨ button on a single tile and ↻ Re-improve. **Bulk runs are never decided by it**: the selection toolbar shows one button per available engine and each states its trade-off, so a batch always says which pass it is about to run.
- **Model build** → `seedvr2.model`. Blank (default) = the app resolves it: the 3B FP8 build when present, else whatever is in the folder. Only builds **already on disk** are offered in the dropdown — the pack's loader node downloads an unknown name on first use, and a dropdown must never start a multi-gigabyte download. To use a 7B or a GGUF build, drop the file in `models/SEEDVR2` and it appears in the list. Guidance from the pack: 3B FP8 ≈ 8–12 GB VRAM, 3B FP16 ≈ 12–16, 7B FP8 ≈ 16–20, 7B FP16 ≈ 24+.
- **VAE build** → `seedvr2.vae`. Blank (default) = `ema_vae_fp16.safetensors` when it is there, else the first file in the folder whose **name** says VAE. The dropdown lists the whole `models/SEEDVR2` folder, VAE-looking files first and the rest in a second group: a pin is honoured against every file, because the automatic search already covers every install where the name is recognisable and the setting exists for the one where it is not. Picking a DiT build here fails inside the loader node.
- **Target resolution (short edge)** → `seedvr2.resolution`. Range `256`–`4096`, default **`1080`**. The **short** edge is scaled to this and the aspect ratio is kept, so 1080 on a 3:2 photo gives 1620×1080. LoRA training buckets rarely exceed 1024–1280, so higher mostly costs VRAM and time. Past what your GPU can hold in one pass the app either tiles the frame (if the tiling pack is installed) or warns you before starting — see below.
- **Maximum long edge** → `seedvr2.max_resolution`. `0` (default) = no limit. The VRAM safety valve on a wide crop: at a 1080 short edge a 4:1 panorama becomes 4320 px across.
- **Colour correction** → `seedvr2.color_correction`. One of `lab`, `wavelet`, `wavelet_adaptive`, `hsv`, `adain`, `none`. Default **`lab`** — the model's own default and the most conservative. `wavelet` holds broad tone better on heavily degraded sources; `none` shows the raw output. Colour fidelity is the reason this engine exists, so it is worth trying two modes on one image before a long batch.
- **Blocks offloaded to system RAM** → `seedvr2.blocks_to_swap`. Range `0`–`36`, default **`0`** (none, and fastest). Raise it to fit a bigger build on a smaller card: it trades speed for VRAM headroom and does not change the result.

#### Large upscales: the ceiling, and the optional tiling pack

*Contributed by [SurpassHR](https://github.com/perfectgf/lora-dataset-studio/issues/32), who hit this as a real CUDA out-of-memory on an 11.6 GB card and shipped the tiled workflow this is ported from.*

Upscaling a whole frame at once needs the whole frame in VRAM, so past a certain size it simply fails. Two things follow from that:

- **You are told the limit before a run, not after.** Setup ▸ ComfyUI shows roughly how many megapixels this GPU is good for in a single pass. It is a guide, not a gate — real headroom moves with the build, block swapping and whatever else holds VRAM, so LDS still runs what you ask for and says when it looks over budget. On a GPU it cannot see (no `nvidia-smi`, a remote ComfyUI) it says **nothing**, rather than inventing a number.
- **With the [Comfyui_TTP_Toolset](https://github.com/TTPlanetPig/Comfyui_TTP_Toolset) node pack installed (MIT), large frames are tiled**: the frame is cut into overlapping tiles (1024 px by default), each is upscaled, and the seams are blended back. Nothing *has* to be configured — the lane switches on the geometry — but the tile size and the crossover are settings when you need them (below). Install the pack in ComfyUI-Manager and restart ComfyUI; LDS detects the two node classes it needs.

Without the pack nothing breaks: upscales still run, they are just capped by the card.

- **High-resolution tiling** → `seedvr2.tiling`. One of `auto` (default), `always`, `never`. Tiling is **not only a memory trick**: SeedVR2's target resolution is the size the model actually works at, so a whole 4K frame spreads its capacity over four times the surface while a tile is upscaled in the range it is good at. SurpassHR's side-by-side on his own card showed the full-frame result losing detail and gaining artifacts where the tiled one did not — which also means the old rule (tile only when the frame would not fit) had it backwards: the bigger your GPU, the less often you got the better picture.
  - `auto` — tile once the target short edge is past ~1536 px, or when the frame would not fit anyway. This is the recommended setting and the reason the pack is worth installing.
  - `always` — tile any frame bigger than a single tile, including below that crossover.
  - `never` — always full-frame. Pick this if you ever see a seam; the VRAM warning still applies.
  On `auto` nothing is tiled **at or below** the crossover: the model is already at a comfortable size and a grid would only add seams. The crossover has to be *passed*, not just reached — a 1536 px target at the default 1024 px tile runs full-frame, and so does a 768 px one if you dropped the tile to 512. The panel says which lane your configured target will take, so that decision is never silent. The pre-#32 rule (tile *only* when the frame would not fit) is deliberately not offered — it is the default the side-by-side refuted.
- **Tile size** → `seedvr2.tile_px`. Range `512`–`2048` (snapped to a multiple of 64), default **`1024`** — the contributed value. **This is the memory dial of the engine**: a pass holds one tile at a time, so lowering it to 768 or 512 is what makes a large upscale finish on an 8 GB card, at the cost of more seams and more passes; raising it on a 24 GB card gives fewer seams and more context per tile. It also sizes the model's own **tiled VAE encode/decode**, which runs on the full-frame lane as well — so it lowers VRAM use even with no tiling pack installed. Try this before concluding a big upscale is impossible on your card.
- **Start tiling above** → `seedvr2.tile_threshold`. Short edge past which `auto` tiles, in pixels — strictly *above*, so a target equal to this value still runs full-frame. **`0`** (default) = derive it from the tile size (1.5×, i.e. the shipped 1536 px at a 1024 px tile) so the crossover follows the tile you chose. A positive value places it by hand: lower to tile sooner (safer on a small card), higher to keep more targets in a single fast pass. No effect on `always` or `never`.

 LDS ports only the tiling itself — the original workflow also chained two further node packs to do arithmetic (counting tiles, normalising a pixel count), one of them GPL-3.0, and that arithmetic is done in Python here instead.

**There is no batch-size setting, on purpose.** SeedVR2's `batch_size` is a *video* window whose frames share temporal attention to stay coherent — feeding it unrelated dataset photos would let them bleed into each other. Images are upscaled one per job; throughput comes from the normal generation queue and its fan-out cap.

### Klein model files (optional)

*Contributed by socrasteeze (GitHub).* Pin the exact files the Klein graph loads instead of relying on auto-detection. Every field accepts **a full absolute path or a ComfyUI-relative loader name**; empty fields keep the default behaviour (the canonical download filename first, then a narrow token scan of the ComfyUI model folders). Each field now **lists the files actually found in that ComfyUI folder** (`extra_model_paths.yaml` roots included), with a ↻ to rescan after you drop a new file in; free text stays available because an absolute path from outside every ComfyUI root is a legitimate value no scan can enumerate.

- **Diffusion model (UNET)** → `klein.unet`. A full path, or a name relative to a diffusion-model folder — e.g. `klein/flux-2-klein-9b-fp8.safetensors` under `models/unet`, or a bare filename for a file sitting at a folder root. This is also what lets you use a UNET that does **not** live in a `klein`-named subfolder, which the automatic scan would never find. Default **empty** (auto-detect).
- **Text encoder** → `klein.text_encoder`. Full path, or relative to `models/text_encoders` — e.g. `qwen_3_8b_fp8mixed.safetensors`. Default **empty**.
- **VAE** → `klein.vae`. Full path, or relative to `models/vae` — e.g. `flux2-vae.safetensors`. Default **empty**.
- **Consistency LoRA** → `klein.consistency_lora`. Full path, or relative to `models/loras`. The structure-anchoring LoRA chained onto the Klein edit graph — this is the same key that was previously config-only. Unlike the three above it has a shipped default, so **clearing it disables the LoRA** rather than turning on auto-detection. Default `klein/Flux2-Klein-9B-consistency-V2.safetensors` (the Setup download location).

How references resolve:

- A **full path under any of ComfyUI's model folders** — including folders registered in `extra_model_paths.yaml` (the app parses it exactly like ComfyUI does) — is converted automatically to the relative loader name ComfyUI's nodes need, and the field shows **Found**. Nothing is copied or moved.
- A **full path anywhere else** — Downloads, a Hugging Face cache, another drive — is **hardlinked (or symlinked) into `<ComfyUI models>/<type>/lds-pinned/`** so stock loader nodes can open it, and shows the same **Found**. Your config keeps the original absolute path; the link is created when the reference resolves, costs no extra disk, and is reused on later runs. Staged files are deliberately *not* put in a `klein`-named folder, so they never show up as a second copy in the Klein model picker.
- A reference that **cannot be resolved** — no such file, or the link could not be created (a read-only models folder, or another volume on an account without symlink rights) — is **refused rather than substituted**, with a badge here and the file named. It used to fall back to auto-detection and only show the badge: the run then went ahead on a *different* file from the one displayed, and the only symptom arrived with the images. **Clearing the field is the explicit way back to auto-detection.**
  - **A bare file name is valid here.** The model dropdown in the workspace writes this same key and stores names *without* their folder, so `flux-2-klein-9b-fp8.safetensors` and `klein\flux-2-klein-9b-fp8.safetensors` both name the same file and both resolve. They did not always: the bare form was read as "not on disk", which meant **picking a model from the app's own menu switched the engine off**, explaining that nothing was missing. Nothing you typed was ever wrong there.
  - **An unresolvable Diffusion model does not darken the whole engine** while usable Klein builds sit on disk. That slot has a picker in the run panel, so switching the engine off removed the very control that would have repaired it — a feature announced as *optional* cannot also be a feature one missing file takes hostage. The card asks for a model instead, and the **run** refuses only if it is started without one (never by quietly loading a neighbouring file). With no Klein build on disk at all there is nothing to ask, and the engine stays off.
  - The one exception is **Consistency LoRA left at its shipped default**, which nobody chose: at that value a missing file is still simply skipped, exactly as before.
- Native / **bf16 UNETs** (a filename without `fp8`) now load with `weight_dtype: default`; FP8 builds keep `fp8_e4m3fn`. Both shipped Klein graphs hardcoded `fp8_e4m3fn`, which quantized a full-precision pin on load without saying so. The canonical download carries `fp8` in its name, so a stock install renders exactly as before.
- Generation-LoRA **preset rows** accept full paths the same way.
- **Not cleaned up automatically.** Changing or clearing a pin leaves its link behind in `lds-pinned/`; the folder is safe to delete by hand when nothing points there any more.

Traps and good-to-knows:

- The dataset's own **Klein model (per dataset)** choice still wins over `klein.unet` for that dataset's runs — the pin is what auto-detection falls back to, not an override of an explicit per-run pick.
- **This is the fix for a model the app insists is "missing" while you are looking at it.** Auto-detection is deliberately narrow (a wrong model fails at sampling time with a cryptic shape error, which is worse than a missing one), so it *declines* any file it cannot confidently name — and a declined file is reported as missing. Pinning it by name removes that discretion: the file resolves, and the integrity check then tells you the truth about it, including **"present but unreadable"** for a corrupt or half-downloaded weight, with the delete-and-re-download action attached.
- Pinning the wrong *kind* of file (e.g. another family's text encoder) is **not** validated — that generate will fail at sampling time with a shape error. The narrow auto-detection exists precisely to avoid that; only pin files you know are Klein-compatible.

### Klein generation LoRA presets (optional)

*Idea from @waltm on Discord.* Named combinations of generation LoRAs that stack on top of the local Klein edit graph. Stored in `klein.generation_lora_presets` (default: empty — no presets).

Each preset has a **name** and an **ordered list of LoRAs**, and each LoRA row has:

- a **file** — a name relative to your ComfyUI `models/loras` folder (e.g. `klein/my-lora.safetensors`), exactly like the consistency LoRA. The field is a **searchable dropdown of the LoRAs actually on disk** (every folder, `extra_model_paths.yaml` included), with Klein-compatible files listed first and each one badged by architecture; free text still works for a file you haven't downloaded yet;
- a **strength** — `0`–`1.5`, default **`0.6`**.

Use **＋ New preset**, **Duplicate**, **Delete** and rename to manage them, and the up/down controls to set chain order. **Caps: 8 LoRAs per preset, 12 presets.**

How presets are used matters:

- A preset is **chosen per run** in the **🖥️ Klein tuning** panel of the workspace. The panel opens on **Preset selected by default** → `klein.default_generation_lora_preset` (default: empty = *None*, the behaviour every install had before this key existed). That is a **starting point, not a lock**: the run panel still offers *None* and every other preset for that run, and choosing there **never rewrites the setting**. Fail-closed like the rest of the chain — a default naming a preset you have since renamed or deleted falls back to *None*, and the Settings field says so instead of pretending it is empty. Each engine has its **own** default key (`klein.default_generation_lora_preset` / `krea.default_generation_lora_preset`), because `klein.generation_lora_presets` and `krea.generation_lora_presets` are independent lists where the same name can designate two different chains.
- Resolution happens **by name** on the server, and it's **fail-closed**: if a run references a preset name that no longer exists, it runs **with no extra LoRAs** rather than erroring.
- **Trap:** *renaming* a preset does **not** follow a run that referenced it by the old name — that run silently falls back to no extra LoRAs. Rename before you queue, or re-pick the preset on the run.
- There is deliberately **no automatic NSFW gating** on individual LoRAs — the preset you pick carries the intent. If you want an "NSFW full" stack, make it a preset.
- **Trap:** a row pointing at the **same file as the consistency LoRA** (`klein.consistency_lora`) is skipped — it would chain that LoRA a second time on top of itself, summing both strengths into one delta well past what the file was trained for (visible as blocky, posterized output). **The preset editor now flags that row as you write it**, on the row itself, instead of leaving the only trace in the server log — which is how a preset holding exactly one such row produced a run with no extra LoRA and nothing on screen to explain it. The check compares paths the way the server does (separators unified, case ignored), so `klein/x.safetensors` and `klein\X.safetensors` are both caught. It does **not** claim to catch an absolute path aliasing the same file — the server still drops that row, quietly.

### Klein generation quality

*Raised by ashish.sinha.* **Generation steps** → `klein.generation_steps` (1–50, default **5**). How many sampler steps the local **Klein** engine spends on each generated image — variations, regenerations and the automatic small-image rescue. The shipped workflow had this pinned at **5** with no way to change it; the default is that same 5, so nothing moves until you raise it. More steps usually render more cleanly and cost proportionally more time (10 steps ≈ twice the wait per image).

It is a **rendering** knob, not an anatomy fix: extra limbs, tails or wrong body parts come from the identity prompt describing the wrong kind of subject (see the subject-type note below), and no number of steps repairs that.

**Enhancement LoRA on edits** → `klein.edit_base_lora_strength` (0–2, default **0**). How much of the detail LoRA (`klein/realistic.safetensors`) Klein mixes into an **edit**: the ✦ reference edit, variations, regenerations and the small-image rescue. The shipped workflow carries that LoRA at **0.8** and nothing on these lanes ever turned it down — which stayed invisible while the file existed on no install (the node was skipped), and became real once Setup started downloading it: from then on every Klein edit ran with a style LoRA at 0.8 pulling the result away from the instruction you typed. The default **0** is the render every install had before that download existed; raise it to let the LoRA add detail on purpose. “Upscale & improve” is unaffected — it has its own `klein.improve_base_lora_strength`.

Separate from **Upscale & improve ▸ Steps** (`klein.improve_steps`), which drives the manual improve pass only.

### Klein model (per dataset)

Not in Settings — it lives on the **dataset**, in the 🖥️ *Klein tuning* block of the generation panel and next to the ✨ **Upscale & improve** action itself. One setting, deliberately: generation and improve drive the same loader in the same workflow, so two near-identical model dropdowns would only be a lasting source of confusion. If you ever need to improve with a heavier model than the one you generate with, say so — splitting one stored value into two is additive; merging two back into one would have to throw one of your answers away.

**Default is Auto**, and Auto is exactly what every dataset did before this setting existed: Studio resolves the model itself (the canonical download first, then the first loadable Klein file it finds). Choosing nothing changes nothing.

**The list is detected, never typed.** It comes from the same scan ComfyUI itself would do — `models/unet`, `models/diffusion_models`, every root declared in `extra_model_paths.yaml`, and a relocated models folder (`comfyui.models_dir`) — in a `klein`-named subfolder **or** loose at the root. The one real constraint is that the model must be *nameable* as Klein: either the file name or its folder name has to contain `klein`. See *Where the Klein model can live* in the README.

**What it applies to:** every piece of Klein work that dataset starts — the single ✨ improve, the 🔄✨ re-run, the whole improve batch, Klein generation (variations and regenerations), the **reference edit** on the Klein engine, the **rescue of scraped images under 768 px**, and the **🧽 watermark clean** on the Klein engine (bulk and per-image).

**The one exception is a bank**, which has no dataset and therefore nothing to inherit: its 🧽 Klein inpaint resolves the model automatically, and the panel says which one. Naming the model and choosing it are separate questions — a bank can be told, it just has nowhere to store an answer.

**When there is only one model**, the picker does not appear — there is no choice to make — but the line naming the model still does. Not knowing which model produced an image was the actual complaint; a dropdown with one option was never the answer.

**If the model you chose is later moved or deleted**, the run **refuses by name** and tells you which file is gone. It does not quietly fall back to another model: that swap produces a result that looks perfectly fine and is not the one you asked for.

**Coming from an older version:** the generation picker used to save to your **browser** (`editPage_flux2KleinModel_v1`), which improve never read — that is why improve had no model option anywhere. That browser value is still honoured for generation and is now offered, once, to be saved onto the dataset. Nothing is adopted behind your back: until you accept, improve keeps resolving Auto exactly as before.

### Identity & Klein prompts (advanced)

*Feature request by @bbsorry (雨田壹).* Every generated variation is prefixed by a hidden **identity lock** — a block of text that tells the engine to keep the subject's exact identity and take the pose and setting from the description, not the reference photo. These used to be baked in and invisible; now you can read and edit them. They are stored under `identity_prompts.*`.

**One set per subject type.** *(Reported by ashish.sinha.)* The three identity locks are scoped to the dataset's **subject type** — Human, Animal, Creature, Object, Other. Pick the type with the chips at the top of the card; a small dot marks every type you have already customised. A prompt you write for Animal applies to **animal datasets only** and never to a human one. Before this, the override was one global text: someone who adapted it for animals then saw their human variations come back with tails, extra limbs and odd footwear.

Storage follows that split, and nothing was renamed or migrated: the **Human** overrides stay on the original flat keys (`identity_prompts.face_single`, `.face_multi`, `.klein_identity`), which is where every override written before this change was stored, so yours keeps applying to your human datasets. The other types live under `identity_prompts.by_subject.<type>.<kind>` and never fall back to the flat key. `identity_prompts.klein_improve` and its toggle stay **global** on purpose — "add texture and detail" means the same thing for a person, a dog or a car. **It does not mean the same thing for a drawing**, and that is a known rough edge: the shipped text asks for photographic detail, so on an **Anime** dataset the improve pass works against the very art style every other prompt in the app protects. Until that is settled, edit the box (or turn it off) for drawn datasets — see *Troubleshooting → "Upscale & improve" makes my anime look realistic*.

> If you had adapted the identity prompt for a non-human subject before this change, your text is now sitting in the **Human** set (that is where it was saved). Open the card, check the Human boxes, and move the text to the right subject type — nothing was discarded.

**One box, already filled.** Each prompt is a **single editable box that already contains the exact text in use** — the built-in default when you have not overridden it. Put your cursor in it and change a word; there is no "load the default first" step, and no second read-only copy of the text below (the old two-box layout is gone).

**Nothing is stored while you match the default.** As long as the box still holds the built-in text (surrounding whitespace ignored), the setting is saved as **blank**, which is what makes *blank = use the shipped default* work. This is not cosmetic: if merely opening a prompt persisted a **copy** of it, you would be pinned to that wording forever and every later improvement to the built-in prompt would stop reaching you, silently. The line under the box always tells you which state you are in — *Following the built-in default* or *Custom override*. **Reset to default** appears only in the second case and clears the value back to blank.

**Reproducibility guarantee:** with nothing overridden, generation is **byte-identical** to before this setting existed — you only change behaviour if you deliberately edit the text away from the default.

**Edit the improve instruction from the ✨ button itself.** `identity_prompts.klein_improve` is the one prompt on this card you never have to come here for: the note under **✨ Upscale & improve** (dataset lightbox, bulk toolbar, generated-image lightbox) carries **✎ Edit this instruction here**, which opens the same box and the same on/off tick under the button. It edits **this setting** — there is no per-image and no per-run copy, so what you write there is what this card shows and what every later improve sends, in every dataset. The panel says that in as many words before you type, because a control sitting inside one dataset's screen otherwise reads as belonging to that dataset. The rest of the contract is identical to the box here: the field is pre-filled with the text actually in force, text matching the built-in default is stored as blank, **Reset to default** shows up only once you really have an override, and edits save as you type (the last keystroke is saved even if you close the lightbox on it). What is *not* duplicated there, on purpose: the per-subject identity locks, the other prompt parts and the four strength knobs — the note links here for those.

**Shortcut from the workspace.** The multi-reference instructions are also reachable from **Add images ▸ Extra refs ▸ ✎**, without opening Settings. That modal shows **both** of them — `face_multi` (Nano Banana / ChatGPT) and `klein_identity` (Klein) — because the engines do not share one text, and badges **every** prompt your ticked engines actually use (a Klein + ChatGPT run really uses both). It edits the prompts of the **open dataset's subject type**, and says which one in its title and intro; edits made there are the same settings as here, for that subject.

- **API engine — identity lock (single reference)** → `identity_prompts.face_single`. Prepended to Nano Banana / ChatGPT variations built from **one** reference photo.
- **API engine — identity lock (multiple references)** → `identity_prompts.face_multi`. The same, for variations built from **several** reference photos — it tells the model every reference is the same person and to use them together.
- **Klein — restage & face-identity block** → `identity_prompts.klein_identity`. The instruction block the local **Klein** engine uses to restage the shot (pose, framing, outfit, expression) while keeping the face identical.
- **Klein upscale & improve prompt** → `identity_prompts.klein_improve`, with an on/off toggle `identity_prompts.klein_improve_enabled` (default **on**). The fixed instruction the manual **Klein upscale & improve** action sends to add texture and detail. The shipped text is `add detailed texture, add sharp details, add candid shot, add soft focus effect` — read it before you blame the model: those four clauses describe a **photograph**, and they are applied to every dataset, drawn ones included. **Turn the toggle off** to run that action with **no prompt at all** — a pure upscale with no added styling. Both the text in force and these two levers are now quoted **and editable** from the ✨ **Upscale & improve** button itself (lightbox and bulk toolbar) — see *Edit the improve instruction from the ✨ button itself* above — so you no longer have to know this setting exists to find it, nor leave your images to fix it.
- **Upscale & improve — strength** → `klein.improve_megapixels`, `klein.improve_base_lora_strength`, `klein.improve_consistency_strength`, `klein.improve_steps`. The output resolution, and how much that pass is allowed to change the image. Until these were exposed the whole profile was hardcoded — **both LoRA strengths pinned to 0**, so the *enhancement* LoRA baked into the workflow never applied at all, and the size was fixed at 2 MP whatever the source was worth. Defaults are those same historical values (**2 MP / 0 / 0 / 4 steps**), so leaving them alone keeps today's result exactly. These are read at each run, so to try a new value on an image you already improved, use the **🔄✨** button on that tile: it re-runs the pass on the same source image with the settings as they are now, and replaces the result in place. (The ordinary 🔄 stays hidden there — it would restart from the dataset's reference photo and make an unrelated image.)
  - **Output size (MP)** (0.5–8, default **2**) — the source is rescaled to this pixel budget before sampling, so it *is* the result's resolution. This is the knob that makes "Upscale" actually upscale.
  - **Enhancement LoRA** (0–2, default **0**) — the workflow's own detail LoRA. At 0 it does nothing; try **0.5–0.8**. It needs its weights file (`klein/realistic.safetensors`): when that file is missing the node is skipped entirely and this value changes nothing. **Setup ▸ Install everything downloads it** with the other Klein assets (from [dx8152/Flux2-Klein-9B-Enhanced-Details](https://huggingface.co/dx8152/Flux2-Klein-9B-Enhanced-Details), Apache-2.0) — run it first if the slider seems inert.
  - **Consistency LoRA** (0–1.5, default **1**) — anchors the **composition and background**, not identity. Deliberately high here: an improve pass should add detail *without* redrawing the shot, so the resistance to editing that makes 0.8–1.0 a poor choice for restaging is exactly what you want. (Shipped briefly as `improve_character_lora_strength`, a misnomer; a value saved under that name is still honoured.)
  - **Steps** (1–50, default **4**) — more steps is slower and usually cleaner.
  - Out-of-range or malformed values are **clamped**, never rejected: a bad config weakens the pass instead of failing your click.
  - **A strength you raised is never silently dropped.** If a LoRA's weights file is missing while its strength is above 0, the pass reports the missing asset (which is what triggers its download) instead of running without it and leaving you to wonder why nothing changed. At strength 0 it stays a quiet skip — nothing is lost by not loading a LoRA you did not ask for.

Each field is a plain textarea; there's no Test button — you see the effect on your next generation. If an override ever makes results worse, hit **Restore default**.

### The rest of the prompt (Klein & Krea)

The identity lock is only **one of six** sources a local-edit prompt is assembled from. The other five shipped hardcoded and invisible until this wave — including the one that caused a live incident, where a hold order that *listed* what to preserve ("tattoos, scars, moles…") had the model painting tattoos on subjects who have none. All of them follow the same contract as the locks above: **blank means the shipped default**, non-blank wins, **Restore default** on every box.

Two of them follow the **subject type** chips, because their text genuinely differs per type:

- **Rendering tail (SFW)** → `identity_prompts.render_tail_sfw`. The last thing Klein and Krea read on a safe-for-work shot: the medium and the clamp. For photographic subjects this is `Professional realistic photograph, SFW.`; for **Anime** it asks the model to stay a drawing in the reference's art style.
- **Rendering tail (uncensored)** → `identity_prompts.render_tail_nsfw`. The same position on an uncensored shot: the SFW clamp is dropped and anatomically correct forms are requested. Only the **local** engines ever see it — the API engines refuse this content.
- **Shot detail per framing** → `identity_prompts.framing_face` / `.framing_bust` / `.framing_body` / `.framing_back`. Klein and Krea under-fill a short tag prompt and invent the rest, so each shot carries a concrete description of the framing — this is where "85mm portrait lens look" and "the ENTIRE body visible from head to toe" live. If your full-body shots keep coming back cropped, this is the box.

Non-human overrides for those six live under `identity_prompts.by_subject.<type>.<kind>` like the locks; Human keeps the flat key.

Four more are **global** — one text for every subject type, because they have no per-subject meaning (the two directives are only ever injected into human shots):

- **Hold the skin (Krea)** → `identity_prompts.markings_lock`. Sent with every Krea prompt: it forbids adding marks to the skin, and forbids redrawing, restyling, moving or removing the ones the reference already has. ⚠️ **This is the delicate one.** Naming a body feature in this box is enough to make the model paint it — that is exactly what the first version did. Describe what *not* to do, without naming a single feature.
- **Outfit directive** → `identity_prompts.outfit_vary`. Added to every human shot that does not already name a garment, so clothing comes from the description instead of being copied off the reference (which teaches the LoRA that the person owns one outfit). Note that **Krea replaces it** with a concrete garment from the palette below, so editing this text does not change what Krea sends.
- **Expression directive** → `identity_prompts.expression_neutral`. Added to every human shot that does not already name an expression, so the reference's smile does not ride on all 40 variations.
- **Concrete garments** → `identity_prompts.outfit_palette`, **one garment per line**. Krea preserves anything it is not positively ordered to change, so "a different outfit" is a no-op on it; each shot is handed a real garment from this list instead. ⚠️ The garment is chosen from the shot's name **by position in the list**, so **adding or removing a line reshuffles which garment every shot gets** — same shots, different clothes. Editing the wording of one line only affects that line. Clear the box entirely to go back to the shipped list (an empty list never produces a prompt with no outfit in it).

Both directives are baked into the shot catalog and **stored** with each variation, so the override is applied when the prompt is sent rather than when the shot is created — which means an edit reaches datasets you built **before** you made it, on their next generation.

### What actually gets sent

At the bottom of the card, a live preview of the **composed** prompt: pick an engine, a framing and SFW/uncensored, and it shows the full ~1000 characters a real catalog shot would be sent, assembled from every box on the card, **including edits you have not saved yet**. It is composed by the server through the same functions generation uses, so it cannot drift from reality — and it generates nothing: no model is loaded, no GPU is touched, nothing is billed.

It is also the fastest way to answer "why did it do *that*": read the prompt, find the sentence, edit the box it came from.

## Scraping & sources

Credentials for the built-in web scraper. **All of these apply immediately — no restart** — because sources read their key at request time.

### Source credentials

None of these has a Test button; you find out they work on your next scan.

- **Reddit client ID** → `REDDIT_CLIENT_ID` (secret). Optional. Reddit scans work out of the box using a shared public client ID, but that ID is rate-limited across everyone who uses it, so you can hit *"rate limiting requests, retry in Ns"* (429) before your first scan of the day. Your own free ID gives you a private quota and clears those. **Trap:** on reddit.com/prefs/apps, create the app as type **installed app** — a *web app* or *script* comes with a client secret, and Reddit then rejects the anonymous login this app uses (every scan fails with **401**). The field has a built-in step-by-step guide.
- **Civitai API key** → `CIVITAI_API_KEY` (secret). Optional. Without it, Civitai scans return **SFW results only**; add a key to reach adult content you're entitled to use.
- **Pexels API key (required for Pexels)** → `PEXELS_API_KEY` (secret). **Required** for any Pexels search — there's no shared fallback. The free quota is **200 requests/hour and 20,000/month**. [Create one here](https://www.pexels.com/api/key/). Note the standing warning: an API key alone does **not** authorize dataset or machine-learning use — configure this only if Pexels has explicitly authorized your use case.

### Klein rescue — small scraped images

- **Small-image rescue instruction** → `klein.small_image_prompt`. An optional free-text instruction for **one flow only**: the automatic Klein **rescue** of scraped images under 768 px. Default **empty** — and empty is intentional: with nothing here the app improves from the reference image alone rather than inventing a restoration prompt on your behalf. Unlike the identity prompts above, this field has **no built-in text behind it**, so it stays a plain empty box: there is nothing to pre-fill or reset to. Add an instruction only if you want to steer that rescue (e.g. "sharpen skin texture, keep natural tones"). The manual **"Klein upscale & improve"** action in the lightbox does **not** use this field — it has its own editable prompt under Settings ▸ Engines ▸ **Identity & Klein prompts** (`identity_prompts.klein_improve`), which can also be turned off for a pure upscale.

## Local tools

Where you point the app at the local programs that unlock the full pipeline: **ComfyUI** (Klein generation and Test Studio), **Ollama** (the vision model behind captioning and framing) and **ai-toolkit** (training and JoyCaption). Each card has a **Test** button that tells you immediately whether the app can see the tool.

### ComfyUI

- **ComfyUI API URL** → `comfyui.api_url`. The HTTP endpoint of your running ComfyUI. Default **`http://127.0.0.1:8188`**. **Test** confirms it answers.
- **ComfyUI install directory** → `comfyui.base_dir`. The folder that contains `models/`, `output/`, `input/`. Default **empty**. This is what lets the app scan your checkpoints and LoRAs — set the API URL alone and there's nothing to scan. If you point it at a `..._windows_portable` folder, the app auto-corrects to the `ComfyUI` sub-folder inside it. In the **Setup wizard** this field is checked as you type: a wrong, empty or missing folder gets a specific reason, and pointing at the launcher/parent folder offers the real ComfyUI inside it in one click. The wizard additionally checks that the app can actually **put a file in that install's `input/` folder** (honouring an `input_dir` override if you set one) — the half it used to certify without testing. A failure there is a **warning, never a blocker**: configuring the app before mounting your volumes is a perfectly normal order of operations.
- **Advanced: ComfyUI folder overrides** → `comfyui.output_dir`, `comfyui.input_dir`, `comfyui.models_dir`, `comfyui.loras_dir`. All default **empty**, and empty is what you want unless ComfyUI runs on folders of its own — a ComfyUI started with `--output-directory`, `--input-directory` or `--models-directory` does *not* keep its files under the install directory, so without an override here the app reads and writes in the wrong place. Each field **shows the folder it falls back to while empty**, computed with the very function the app uses at runtime, so the effective path is never something you have to work out; a path that isn't on disk is flagged in amber rather than failing silently mid-generation. If ComfyUI is running, the app asks it which folders it was launched with (it reports its own command line via `/system_stats`) and offers them in one click — nothing is ever guessed from a folder layout, so no suggestion appears when ComfyUI is unreachable, predates that field, or was started with no custom folders.

  Each field is also checked for **usability, not just existence**: the app *writes* into `input/` (and into the LoRA folder when it installs a trained LoRA), so a folder that is there but cannot be written to from the app's process is flagged in amber with the reason. This is the case that used to be invisible — a ComfyUI in a **separate container, in WSL, or on another machine** answers on its URL while its `input/`/`output/` folders are not shared, so everything looked configured and the first generation died on a detail-free `500` (reported on Discord by nofaceman). The app hands ComfyUI its source images **through the filesystem**, not over the API: `input/` and `output/` must be visible to both sides **at the same path**. See *Troubleshooting → ComfyUI runs in another container*.
- **ComfyUI response timeout** → `comfyui.object_info_timeout_s`. How long ComfyUI may take to answer the one heavy question the app asks it: *list every node class and every model file you have* (`/object_info`). Default **45 s**, clamped to **5-300**. That list grows with every custom-node pack and every weight you install, so this is one of the very few settings whose right value genuinely depends on your install: it was a hardcoded **8 s** until a user measured **~15 s** on his own ComfyUI (j_o_e_l., Discord) and found that Krea 2 generations were being refused with *"ComfyUI isn't running"* — on a ComfyUI that was running. Raise it if you ever see **"ComfyUI is answering too slowly"**; the app now says that instead of blaming a stopped server, and names the number. **A high value costs you nothing when ComfyUI is off**: the connection attempt and the answer have separate budgets, and a ComfyUI that is not listening fails the connection in about 3 seconds no matter what you set here — so background checks never sit waiting for a server that isn't there. If the enumeration does fail, the app remembers that for about 20 seconds instead of letting every screen re-ask for the same multi-megabyte answer; pressing **Refresh models** (or saving Settings) re-asks immediately, which is what you want right after starting ComfyUI.

- **Hugging Face token** → `HF_TOKEN` (secret, no Test button). Only needed to auto-download **license-gated** models — notably the Klein fp8 weights, and the gated training bases (Krea 2, FLUX.1-dev, FLUX.2 Klein). Read access is enough for accepted gated models. This token is handed to the local training subprocess explicitly, so what you save here is what training authenticates with. **If you leave it empty**, a login already on the machine (`hf auth login`, i.e. a token file under `~/.cache/huggingface`, `$XDG_CACHE_HOME` or `$HF_HOME`) is found and used instead — the *Hugging Face cache override* below relocates the cache without logging you out. **When a download is refused, read the status code, not the sentence**: `401` means Hugging Face saw no valid token (paste one here), `403` means the token is valid but that account has not accepted the model licence yet (open the model page and accept it). The raw Hugging Face message says "you must have access to it and be authenticated" in both cases; the failure block in the training panel separates them for you.

**Overrides and `extra_model_paths.yaml`.** These two mechanisms stack rather than compete. `comfyui.models_dir`/`comfyui.loras_dir` set the app's *default* model roots; `extra_model_paths.yaml` is read **in addition**, from `<comfyui.base_dir>/extra_model_paths.yaml` — the same place ComfyUI itself looks. The yaml is therefore always located from the install directory, never from a models override, so the two can't end up pointing at different trees. For models and LoRAs the yaml usually already does the job; the override is for the case where the models folder itself moved.

**Where a deployed LoRA lands.** Reading a LoRA can span every root; installing one has to pick exactly one folder, and the app picks it in this order: the **`comfyui.loras_dir` override** if you filled it (you said where your files go — the yaml cannot take that back), otherwise **the first LoRA root in ComfyUI's own priority order**, otherwise `<install>/models/loras`. In yaml terms that first root is a declared `loras:` folder **only when its profile carries `is_default: true`** — that flag is exactly how ComfyUI is told *look here first*, and a plain extra root stays a secondary location for ComfyUI, so it stays secondary here too. The **open LoRA folder** button opens that same folder, by construction. Reported by Geekswordsman (GitHub #25), whose deploys were landing in `<install>/models/loras` while his yaml declared another folder. LoRAs deployed **before** this changed are still listed, still loadable and still deletable where they are — nothing on disk is moved.

**Continuing without ComfyUI.** Leaving the install directory empty in the Setup wizard is a deliberate choice: it shows what turns off (local Klein generation including the NSFW lane, Klein watermark cleaning, the Test Studio, training on your own ComfyUI base models, and the on-disk LoRA preset picker) versus what stays on (scraping, curation, captioning, the API image engines, ai-toolkit/cloud training, Hugging Face publishing), then remembers the skip (`comfyui.setup_skipped`) so it stops nagging. Entering a directory at any point cancels the skip automatically and turns those features back on — the flag never hides a real problem with a ComfyUI you *have* configured.

**Models outside `models/`?** If your ComfyUI uses an `extra_model_paths.yaml` (portable builds and Stability Matrix installs commonly do), the app reads it the same way ComfyUI does, so bases that live elsewhere are found. This isn't a setting — it follows automatically from your install directory. Without such a file, nothing changes.

This now includes the **training** bases, which were the last exception: an SDXL checkpoint and a Z-Image merge declared in the yaml are listed in the base picker, accepted at launch, and handed to ai-toolkit as a real path. When the same file name exists in two roots, the one a running ComfyUI would load wins (`is_default` first), so you train on the same weights you generate with. Capitalisation of folders and file names doesn't have to match what the picker stored. And when a base genuinely cannot be found, the app says so **naming the file**, instead of passing a bare name down to ai-toolkit and letting it fail on a path you never typed.

### Ollama

The card shows Ollama's live state and, when the binary is installed but the server isn't running, a **▶ Start Ollama** button that launches it for you — no terminal needed.

- **Ollama URL** → `ollama.url`. Where Ollama is listening. Default **`http://127.0.0.1:11434`**.

**Sharing one Ollama with another tool.** A single local Ollama runs one model
at a time, so if another app has one loaded, LDS refuses to start its own rather
than evict work that isn't its. When that happens, ✨ Enhance and 🔎 Describe say
so and then **wait**: Ollama unloads an idle model by itself after a few minutes,
and your action restarts on its own the moment the model is free — no clicking,
nothing retyped. If you would rather not wait, **Unload it and continue** frees
the other model and resumes; that click is the only thing that ever unloads a
model LDS did not load. Pointing LDS at a *second* Ollama instance on another
port (or another machine) removes the contention entirely. A **remote** URL is
never probed or unloaded — it isn't sharing this machine's GPU.

In Docker, choose the deployment only from **Setup → Ollama**: `none` disables it, `host` uses the existing host service at the authoritative `http://host.docker.internal:11434`, and `docker` uses the isolated companion at the authoritative `http://ollama:11434`. The managed URL is read-only. Neither launcher downloads a model: use the explicit **Pull** button in LDS to see progress and cancel or resume the transfer.

- **Docker deployment mode** → `ollama.deployment_mode`: `none`, `host`, or `docker`. This setting is selected by Setup and applies only to Docker; native installs keep their normal URL-based behavior.
- **Ollama vision model** → `ollama.vision_model`. The vision model used for auto-captioning, framing auto-classify, head-crop and watermark detection. Default **`huihui_ai/qwen3-vl-abliterated:8b-instruct`** — the **abliterated** (uncensored) build, so it captions adult datasets instead of refusing them. **Trap:** keep the **`-instruct`** tag. The plain `:8b` tag is the *Thinking* variant, which reasons out loud instead of captioning and produces garbage here.

- **Images analysed at once** → `ollama.vision_concurrency`. How many images a bank pass sends to Ollama at the same time. Default **4**. The passes that read every image in a bank — watermark scan, framing, captions — spend most of each request waiting on the round-trip rather than on the GPU, so overlapping them roughly **halves** a long pass (measured 2.0× at 4). Going higher gains little: 6 and 8 buy single-digit percentages unless your Ollama is configured for more parallel requests (`OLLAMA_NUM_PARALLEL`), and they make **Stop** take a few seconds longer because it waits for the calls already in flight. Set it to **1** to get the old strictly-one-at-a-time behaviour back. Any value the app can't read falls back to 4, and anything above 16 is clamped — a bad value costs you speed, never the pass.

- **Keep the vision model warm** → `ollama.vision_keep_warm_seconds`. How long a *one-off* vision job may leave the model loaded once it's done. Default **120 s** (0 = off, capped at 600). Loading the model costs about **13 s**; describing an image once it's loaded costs about **0.5 s** — so a cold call is roughly **25×** a warm one, and the old behaviour (unload after every single image) made cropping five reference photos in a row pay that load five times. The catch is memory: the vision model really occupies about **7.5 GB**, and a loaded ComfyUI already sits near 19 GB of a 24 GB card, so they don't both fit — on Windows nothing errors out, the driver just pages silently and a vision pass measured **13.5× slower** in that state. Keeping it warm is therefore *conditional and revocable*: the app only leases it when neither a training run nor its own generation queue wants the card, and it hands the memory straight back the moment a generation is submitted or a training starts. If the app can't tell what's using the GPU, it unloads — the old behaviour. Bank passes (watermark / framing / captions) are unaffected: they already keep the model warm for their own duration and unload at the end. Set it to **Off** on a card that's tight on memory, or if you run generations from ComfyUI's own interface (work LDS never sees, so it can't revoke the lease for it — the exposure is bounded by this value). If you *want* ComfyUI and the vision model to genuinely coexist rather than take turns, the lever is on ComfyUI's side, not here: it accepts a `--reserve-vram <GB>` launch flag ("the amount of VRAM in GB to reserve for your OS/other software"), which defaults to a mere 0.7 GB on Windows — that default is exactly why a loaded ComfyUI leaves no room. Raising it caps ComfyUI and frees the headroom, at the cost of heavier video workflows. LDS never launches ComfyUI, so it can't set this for you.

**Test** checks end-to-end: that Ollama is reachable *and* the configured model is actually pulled.

### ai-toolkit

- **ai-toolkit directory** → `aitoolkit.dir`. The folder containing ai-toolkit's `run.py`. Default **empty**. **Test** validates it and unlocks training + JoyCaption captioning. **Test also runs `import torch` on the chosen interpreter** — a folder that looks right but is paired with a Python that has no training dependencies used to pass this check and then fail every run.
- ⚠️ **This is ai-toolkit's folder and ai-toolkit's Python, not the Studio's.** The app you are using has its own `.venv`; the ai-toolkit folder has its own venv, the one carrying `torch`. The Next.js UI shipped inside ai-toolkit (`ui/`, port 8675) is unrelated — this app never launches or reads it. See [What is running on your machine](getting-started.md#architecture).
- **Python interpreter (optional)** → `aitoolkit.python`. Default **empty = auto-detect** a `venv/` or `.venv/` next to `run.py`. Fill this with the full path to the interpreter ai-toolkit should run with whenever there is no venv folder for the app to find — **conda, uv, the system Python**, or a **portable / embedded build** that ships its own `python_embeded\python.exe` (several community install scripts do exactly that). Examples: `C:\miniconda3\envs\aitk\python.exe`, `C:\ai-toolkit\python_embeded\python.exe`. A venv is one way to give ai-toolkit a Python, not a requirement — when Setup finds a plausible interpreter inside the ai-toolkit folder, it offers to fill this in for you in one click. **Nothing ever fills this field on your behalf**: it is set by that one-click offer, or typed here. **A value here always wins over auto-detection**, so a wrong path silently shadows a perfectly good venv — which is why a training launch now refuses, naming the path, when the interpreter set here cannot `import torch`, and offers the working venv it found next to `run.py` instead (reported by strouder, GitHub #19). On Windows, watch out for `…\AppData\Local\Microsoft\WindowsApps\python.exe`: that is usually the Store alias, not a real Python. See also the [supported Python versions](getting-started.md#python-versions) — ai-toolkit wants 3.11.

Under **Advanced: ai-toolkit overrides**, three optional path overrides (all default empty → derived from the ai-toolkit directory):

- **Datasets directory override** → `aitoolkit.datasets_dir` (defaults to `<dir>/datasets`).
- **Output directory override** → `aitoolkit.output_dir` (defaults to `<dir>/output`).
- **Hugging Face cache override** → `aitoolkit.hf_home` (defaults to a cache under the ai-toolkit folder). Point this at an existing HF cache to avoid re-downloading base models. It moves the *cache* only: a `hf auth login` token stored in your default Hugging Face home stays in use, so relocating the cache never de-authenticates you on gated bases.

## Captioning & quality

Settings for how captions are produced and how the quality tools behave.

### Dataset import

What happens to a photo the **moment it enters a dataset**. The default now
keeps the source as the master file; a training launch creates its own disposable
working copies. This follows **Qeeyana (Reddit)** asking: *"Images added to
'dataset' are automatically normalized to 1024. Why? Let me choose not to."*

- **Stored encoding** → `dataset_import.encoding`. Default **`preserve`**.
  | Value | What is written |
  |---|---|
  | `preserve` *(default)* | An un-cropped JPG/JPEG, PNG, WebP or BMP is kept byte-for-byte with the matching extension. `max_side` is ignored. |
  | `standard` | Opt-in normalization to WebP quality 92, with the selected maximum side. |
  | `high` | Opt-in normalization to WebP quality 100, with the selected maximum side. Still lossy. |
  | `lossless` | Opt-in normalization to lossless WebP, with the selected maximum side. |
  `preserve` is for a dataset that is also your archive. It does not ask the
  trainer to consume an arbitrary source file: at training start LDS writes
  temporary PNG + caption pairs for ai-toolkit, then leaves the imported master
  untouched.

**Import safety limit — every mode:** Before preserve, WebP normalization, or
auto head-crop can decode the source, it must fit the **Image size budget**
below (shipped default **64 Mi-pixels** and **16384 px per side**). A larger
file is rejected, and the message names the setting. WebP normalization does
not bypass this admission limit.
- **Stored resolution** → `dataset_import.max_side`. Used only by the three WebP
normalization modes. Choose `1024`, `1536`, `2048`, `4096`, or `0` = original
size. The aspect ratio is always preserved (no square padding) and an image is
never enlarged. This output setting takes effect only after the source passes
the import safety limit above; normalized output also clamps the longest side to
**8192 px** — that ceiling bounds what is *written* by a WebP mode and is
deliberately independent of the input budget.

### Image size budget

How large a source image **any** part of the app may decode. Not an import-only
rule: dataset import, ZIP and scrape ingest, Bank scan and thumbnails, edits,
ComfyUI staging and Ollama vision captioning all read these two keys, so an
image you can import is an image you can look at.

- **Maximum total pixels** → `image_input.max_pixels`. Default
  **`67108864`** (64 Mi-pixels). `0` = no limit.
- **Maximum side** → `image_input.max_side`. Default **`16384`** px. `0` = no
  limit.

This is a **memory** budget, not an encoder limit: a decoded RGB pixel costs 3
bytes (RGBA 4), and an edit or analysis pass can hold a second copy at the same
time. So 64 Mi-pixels is about **192 MiB** for one decoded RGB buffer and
roughly **384 MiB** while a working copy exists. It admits every current phone
and 35 mm camera master (a 61 MP 9504×6336 frame is 57 Mi-pixels) and ordinary
stitched panoramas — the previous fixed 16 Mi-pixels / 8192 px refused both.

**What `0` disarms.** No limit means a corrupt or hostile file can be decoded
until it fills memory: a few hundred header bytes can claim billions of pixels,
and the app has no second guard behind this one — an unlimited budget also lifts
Pillow's own decompression-bomb threshold, so the choice is real in both
directions. It is offered because "let it through" is a legitimate answer for a
panorama you produced yourself.

**Inference workers get it handed down, one lane at a time.** Face, aesthetic and
NSFW scoring, semantic indexing and watermark inpainting run in a separate Python
interpreter that cannot import the app or its config, so their shared guard
(`backend/infer/bank_image_guard.py`) enforces its own fixed 16 Mi-pixel /
8192 px ceiling unless the lane that launched it passes this budget down as
environment.

**JoyCaption captioning does.** It was the lane where the split actually hurt: an
image this budget accepts imports, displays and trains normally, but the
captioner refused it one by one — a 89-image dataset of camera masters came back
with 37 captions and 52 refusals reading *"bank image rejects images above 8192 px
per side or 16777216 pixels"*. Captioning now enforces the budget above (the
model sees a 384 px copy either way), and a pass that still has to refuse an image
counts it and repeats the engine's own reason in its result line instead of just
returning a smaller number.

**The other lanes still keep the fixed ceiling.** Scoring, embedding and watermark
work hold a model and a full decode at once, and nobody has measured what a 64
Mi-pixel input costs them — so they keep the conservative number rather than
inherit an unmeasured one. An image above it imports, displays and trains
normally; those workers skip it.

**Auto head-crop is deliberately different.** It changes the picture into a
square head shot, so it creates a derived WebP even when `preserve` is selected.
The same is true of later edits such as crop, rotate and watermark clean: a
transformed image is not the original master any more.

**Changing this is not retroactive.** It applies to images imported *from now
on*, so a dataset can hold mixed formats and sizes. That is harmless for training
(every trainer buckets and downscales on its own). Existing WebPs cannot be
reconstructed into the source files that were discarded by older versions.

**What it does NOT touch**: generated images, the ≤2048 px copies handed to an
image API, and any image you have already curated — those lanes keep their own
fixed sizes on purpose.

### Captioning

- **Captioning backend** → `captioning.backend`. Which captioner writes your captions. Default **`auto`**.

| Value | Behaviour |
|---|---|
| `auto` *(default)* | Prefer JoyCaption (via ai-toolkit), fall back to the Ollama vision model. |
| `joycaption` | JoyCaption only. |
| `ollama` | Ollama vision model only. |
| `none` | No auto-captioning — you write them yourself. |

`auto` is a **chain**, not a coin toss: JoyCaption captions the images it can in one
batch, the Ollama vision model covers whatever is left, and on a **Concept** dataset
Ollama rewrites JoyCaption's drafts. The two engines write in different styles, so one
batch can come back in two voices. Every caption pass now reports which engine wrote
what — in the toast, and on a line under the caption buttons (Captions ▸ Generate
captions), e.g. *“8 by JoyCaption · 4 by Ollama”*. That line describes the last pass
of the current session only; nothing is stored per image. Pick a single value above
(or per dataset, in Captions ▸ ⚙️ Options) if you want one voice across a set.

### Watermark inpainting

- **Processing device** → `watermark.device`. Where LaMa inpainting runs. Default **`auto`**. Options: `auto` (GPU when available, otherwise CPU), `cuda` (force GPU — pauses ComfyUI while cleaning), `cpu` (keep the GPU free). This only affects the **LaMa** engine.
- **Allow automatic crop** → `watermark.allow_crop`. Default **on**. When on, a watermark sitting in an outer border band is **cropped off** (a pure pixel crop — it invents nothing). Turn it **off** and such a mark is **repainted instead** (with LaMa or Klein per the chosen engine) rather than cropped. The exact same preference is editable inline in the workspace's **Clean** bar — it's one shared value, so changing it in either place changes both.

**Honest note on engine choice.** The **LaMa (fast) vs Klein (quality)** engine is *not* a Settings toggle — it's a per-batch picker in the Clean bar and a per-image choice in the review lightbox. `watermark.device` above governs LaMa only; Klein cleaning runs through ComfyUI.

### Face similarity

Two thresholds on the 0–1 face-similarity score (InsightFace), which badge each image against your reference.

- **Face score — green threshold** → `face_scoring.green`. At or above this, the image is a **strong match** (green). Default **`0.50`**.
- **Face score — orange threshold** → `face_scoring.orange`. At or above this but below green, it's **borderline** (orange); below it, red. Default **`0.45`**.

Raise them for a stricter set, lower them if good shots are being flagged too harshly.

**These thresholds do nothing on an Anime dataset.** InsightFace is trained on
photographs and cannot read a drawn face, so face similarity is refused outright
when the dataset's **subject type** is *Anime* — the 🎭 Analyze faces button, 🎯
Auto-triage, Best epoch and the Test Studio's face scoring all say so instead of
producing numbers nobody could measure. There is no override, on purpose: the
subject type *is* the switch. If the dataset really is photographic, set it back to
*Human* and everything scores again — scores from an earlier pass are never
deleted. Head-cropping a reference is unaffected: it runs on the vision model
(Qwen3-VL), which reads a drawn head fine.

### Image bank triage

Thresholds for the **🗃️ Bank** quality flags. Every scanned image stores its
**raw scores**, and the flags are recomputed against these values on every
read — so changing a threshold re-sorts an already-scanned bank instantly,
with **no rescan**. (The two exceptions are noted below.)

> **The same twelve values are editable from the Bank itself** — open
> **🎚 Filter thresholds** above the grid, under the filter chips they decide.
> It is one setting seen in two places, not a copy: editing either one writes
> `config.bank.<key>` and therefore applies to **every** bank. The Bank panel
> additionally previews how many images a candidate value would flag before you
> save, and groups the controls by intent. See
> *Using the app → Tune the Bank filter thresholds*.

- **Sharpness minimum** → `bank.sharpness_min`. Variance of the Laplacian (the classic focus measure) under this = flagged **🌫 blurry**. Default **`100`**. Raise it to be stricter about focus, lower it if artistic soft shots get flagged.
- **Noise maximum** → `bank.noise_max`. High-frequency residual (RMS vs a Gaussian blur) over this = flagged **📺 noisy**. Default **`15`**. Heavily textured images (foliage, fabric) score high by nature — this is a flag to review, not a verdict.
- **Uniformity minimum** → `bank.uniformity_min`. Grayscale spread under this = flagged **⬜ flat** (solid colors, black frames, empty screenshots). Default **`12`**.
- **Minimum side (px)** → `bank.min_side`. Smaller image side under this = flagged **📐 small**. Default **`768`** — the same bar as the dataset import guard, because trainers only ever *downscale*.
- **Real-detail minimum** → `bank.detail_min`. Share of the stored size (0–1) that still carries real picture, under which an image is flagged **🫧 soft detail**. Default **`0.72`** — on a real 36 000-image bank that selects the softest ~3%, and it sits below the 10th percentile of images measured to be genuinely full-resolution, so a sharp photo does not trip it. **What it measures:** the scan shrinks the image and rebuilds it at a ladder of sizes; the smallest size that still reconstructs it is where the picture actually stops. An image enlarged from 512 to 2048 rebuilds perfectly from a quarter-size copy, so it reads ~0.5 and the bank says *"2048 px stored · ~512 px of real detail"*. **What it does NOT measure:** which of the possible causes it was. Motion blur, an out-of-focus background and aggressive denoising remove the same detail and read the same way — treat it exactly like the sharpness score, as a shortlist to look at, never as proof an image was enlarged. It also cannot see nearest-neighbour enlargement (blocky pixels are real detail) and it under-states large enlargements, so the pixel figure is a rank, not a measurement of the original file.
- **Black-bar maximum** → `bank.bars_max`. Share of the frame (0–1) that may be flat black letterbox/pillarbox before an image is flagged **🎞 black bars**. Default **`0.04`**; it caught ~4% of the reference bank (screenshots of videos, stills padded into a square). Those bars survive a training crop, so they are worth seeing.
- **Duplicate distance** → `bank.dup_distance`. How many of the 64 perceptual-hash bits two images may differ by and still be grouped as **≈ near-duplicates**. Default **`8`** (the same hash and distance the dataset import dedup uses). *Applies at the next quality scan* (groups are rebuilt then).
- **Same-person similarity** → `bank.face_threshold`. Cosine similarity at or above which two faces cluster as the same person in **👥 Group by person**. Default **`0.45`**. Raise it if different people get merged into one cluster; lower it if the same person splits into several. *Applies at the next face pass* (embeddings are cached, so re-clustering is fast).
- **Aesthetic minimum** → `bank.aesthetic_min`. LAION aesthetic score (~1–10) under which an image is flagged **💔 low aesthetic** — the "keep the nice ones" cut. Default **`5`**. Only images the **✨ Score** pass reached carry a score; an unscored image is never flagged. The score also drives "keep best" on duplicate groups (the nicest-looking copy wins).
- **NSFW maximum** → `bank.nsfw_max`. NSFW probability (0–1) over which an image is flagged **🔞 NSFW**, to split a mixed SFW/NSFW dump. Default **`0.5`**. Set by the **✨ Score** pass; a review flag, not a verdict.
- **Same-style similarity** → `bank.style_threshold`. Cosine similarity on the CLIP image embeddings at or above which two images share a visual **🎨 style** (screenshots/memes cluster apart from photoreal) in the **✨ Score** pass. Default **`0.6`**. *Applies at the next scoring pass* — the embeddings are cached, so re-clustering at another threshold costs **no inference at all**: the pass does not even load the model. It is not instant, though, and the cost is the grouping itself, which compares every image with every other: **~8 s over 5 000 images, ~3 min over 23 000** (measured). Stopping the pass during that phase leaves the previous grouping in place rather than writing half of a new one — the ids are one numbering of the whole bank, so half of them would collide with the other half. **A measured limit, on a big single-subject bank:** the grouping is *transitive* (A groups with B and B with C puts A with C, even if A and C look nothing alike), so on a bank whose images are all of one person a chain of near-neighbours can merge everything into one group. Measured on a 25 058-image bank at the default `0.6`: **one group holding 25 056 of them** — and raising the threshold does not open a middle ground so much as move the cliff (0.8 → one group of 24 735; 0.9 → one of 15 066; 0.95 → the grouping shatters into 19 716 groups, 17 137 of them single images). The end-of-pass line now states the size of the biggest group against the total, so this is visible without opening the database. If your bank is varied, the default behaves; if it is one subject shot over and over, expect the 🎨 style chip to be close to useless whatever the threshold, and use ✂ **Find crops & variants** (a much tighter, per-pair comparison) for the grouping you probably wanted.
- **Semantic duplicate similarity** → `bank.semantic_dup_threshold`. CLIP cosine similarity at or above which two images are grouped as a **✂ semantic near-duplicate** — a crop or re-compressed variant of the *same shot* that the perceptual-hash **≈ Duplicates** (stage 1) misses. Default **`0.96`**. With CLIP selected it needs **✨ Score** first. SigLIP 2 uses the separate conservative starting value `bank_semantic.siglip2_semantic_dup_threshold`; it deliberately does not inherit the CLIP cutoff, and should be reviewed/calibrated on your Banks. Re-running the grouping reuses the selected cache; no image inference.

- **Which Python runs ✨ Score** → `bank_scoring.python`. **Auto-managed:** leave it empty and Setup ▸ Quality tools builds a dedicated environment and fills it in. It carries **CPU-only PyTorch** on purpose (a first install stays small instead of pulling ~2.5 GB of CUDA wheels on machines with no card), which costs roughly **336 ms per image** instead of ~15 ms on a GPU. On a machine that already has a working CUDA PyTorch — ai-toolkit's venv, ComfyUI's, a conda env — you can point Score at it instead: open a bank and click **⚡ Use a GPU Python I already have** under the CPU warning. The picker checks each candidate *package by package* (`torch`, `open_clip`, `transformers`, `timm`, `numpy`, `Pillow`) and **refuses** any interpreter that can't run the whole pass — CUDA alone is not enough, and a missing `open_clip` would only surface an hour into a run. Nothing is ever installed into an environment the app did not build: a missing package is named with the exact command, for you to run. Reversible at any time (**Back to the app default**), and leaving it alone changes nothing — detection is an offer, never a prerequisite. The picker also accepts a path you type: an interpreter **or** the environment folder holding it (venv, conda/miniconda, uv, a portable bundle, the system Python, another disk), spaces and accents included. No torch or CUDA *version* is required — only that the modules import and `torch.cuda.is_available()` is true. On a machine with no NVIDIA card the picker says so and stops suggesting CUDA; it still lets you borrow an interpreter that already has the packages, to avoid installing them twice. The **Install / ↻ Reinstall** button in Setup ▸ Quality tools honours the same rule: while Score is pointed at a borrowed interpreter it installs nothing and prints the `pip install` command instead — clear the setting (**Back to the app default**) if you want the app to build and fill its own environment again. See *Using the app ▸ Make Score use a GPU Python you already have*.

**Not a setting, but it lives with them:** the **🎨 Pick diverse** popover in a
bank carries a **Skip the odd ones out** slider (0–100%, **default 50%**) next to
its *How many*. It is a per-click control — chosen where you use it, not stored
in `config.json` — because it is a property of the selection you are asking for,
not of the bank. "Most diverse" is farthest-point sampling, which by construction
favours the most *isolated* images; the slider discounts an image for being alone
in the bank so memes, screenshots and stray photos of someone else stop winning
the first picks, while anything as typical as the median of the bank is left
untouched (it cannot pull the selection towards look-alikes). **0 reproduces the
pure-coverage behaviour this button had before the slider existed** — the change
of default does change what a given bank returns. See *Using the app ▸ Curate down
to the right subset*.

- **Watermark detection** → `watermark_detect.backend`. Which engine **🧽 Find watermarks** uses, on datasets *and* banks. `auto` (default) takes the optional watermark-detector extra when it is installed — ~0.14 s per image and it returns a score — and the Ollama vision model otherwise; that is the behaviour that shipped, so leaving this alone changes nothing. `detector` and `vision` pin one engine. A pinned `detector` with the extra **not** installed does not fail the scan: the vision model runs it and the app says so, naming *Setup ▸ Quality tools ▸ Watermark detector*. The two engines disagree at the margins, and only the detector can flag an image *without* a position — those are counted apart and 🧽 Clean leaves them for 🔍 Review instead of marking them failed. *Applies at the next scan; images already dismissed as false positives are only re-judged by **⟲ Rescan incl. dismissed**.*
- **Watermark detector sensitivity** → `watermark_detect.threshold`. The score (0–1) at or above which **🚩 Find watermarks** flags an image, *when the optional watermark-detector extra is installed* (Setup ▸ Quality tools). Default **`0.94`**. *Applies at the next scan.*
  This number is **measured, not chosen**, and it is deliberately nowhere near the 0.5 a probability normally implies: the classifier's scores sit hard against 1. On a 110-image sample drawn from a real 29 759-image bank and labelled by eye (2026-08-03, CPU), a threshold of 0.5 would have flagged **52 of the 55 clean images**; 0.94 flagged **none** of them and still caught **54 of the 55** marked ones. The clean images topped out at 0.939 and the marked ones bottomed out at 0.929, so the two populations overlap by about 0.01 — there is no setting that is perfect, only a knee. Lower it towards 0.92 to catch the faintest marks and hand-check a few clean images; raise it towards 0.96 to miss a mark rather than crop anything by mistake.
- **Watermark detector interpreter** → `watermark_detect.python`. Written by the installer, not edited here. Empty means "use the **Bank scoring** environment", which already carries the two packages this extra needs.
- **Watermark detector weights** → `watermark_detect.models_root`. Empty means `data/models/watermark_detect` (~0.9 GB, downloaded once at install).
- **Watermark detector device** → `watermark_detect.device` (`auto` | `cuda` | `cpu`). The extra installs CPU-only torch, so this is CPU unless you point `watermark_detect.python` at an environment with a CUDA torch. The pass only takes the exclusive-GPU window (which unloads ComfyUI and blocks a training start) when it will actually use the card.
- **Locate the mark** → `watermark_detect.locate` (default on). Runs the second model on the images the first one flagged, to record **where** the mark sits. Off means images are flagged with no position, and neither **✂ Auto-crop** nor **🧽 Inpaint** can route on them — only worth turning off on a bank you intend to filter rather than clean.

The **✨ Score** pass (aesthetic · NSFW · style) needs the **Bank scoring** extra (Setup ▸ Quality tools). **🚩 Find watermarks** runs one of two ways: the **watermark detector** extra when it is installed (~0.14 s per image, and it does not need Ollama at all), otherwise the vision model from **Captioning** (~1.7 s per image). Installing the extra only ever adds the faster route — with nothing installed the pass behaves exactly as it always has. Both are detection-only: the bank never edits your source files.

## Training

Defaults for new runs, plus everything about the optional cloud training lane.

### Defaults

- **Default training family** → `training.default_family`. The model family preselected when you start a new run. One of `zimage`, `sdxl`, `krea`, `flux`, `flux2klein`, `anima`. Default **`zimage`**. Purely a starting point — you can switch family per run. `anima` trains the open [Anima](https://huggingface.co/circlestone-labs/Anima-Base-v1.0-Diffusers) anime model on its public base (no gated download); it is **local-only** for now (needs an up-to-date ai-toolkit + diffusers — cloud training arrives once the GPU pod image is verified). **Anima is the one family with hybrid prompting:** booru tags *and* natural language are both first-class on it, so the caption-style guard says nothing there — prose is merely the preselected default, and a booru-captioned Anima dataset trains without being flagged or forced. Every other family keeps its single expected form (SDXL = booru tags, the rest = prose).

### Training base & variant are per FAMILY

The **Base** and **variant** chosen in *Advanced training options* belong to the
model family they were chosen under, not to the dataset as a whole. Switching
**LoRA type** hands you that family's own base — the official one the first time
you pick it — and switching back restores what you had. Nothing is discarded:
each family's choice is remembered on the dataset.

This matters because the choices are not interchangeable. A Z-Image base is a
ComfyUI merge **name** that gets converted to the diffusers layout first; Krea 2,
FLUX.1 and FLUX.2 Klein take an **absolute path** to a `.safetensors` of their own
architecture. Before this, one dataset held one base for every family, so a base
picked for Z-Image stayed attached to a Krea 2 run — where it was silently ignored
in favour of the official base, while the panel's summary line and the cloud
dialog both went on advertising it.

A base that provably belongs to another family (found on datasets created before
this change) is reported as such in the panel, is not used, and is not offered for
upload to Hugging Face by the cloud dialog.

### Krea 2: the checkpoints on your disk are listed as bases

The **Base** dropdown under *LoRA type = Krea 2* offers the official base **and**
every Krea 2 checkpoint found in your ComfyUI `unet` / `diffusion_models` folders,
including the roots declared in `extra_model_paths.yaml` — the same scan the Test
Studio uses. That is how a full model one of your own runs delivered, or a
community Krea 2 build, becomes something you can keep training on.

The value stored is the **absolute path** of the file, not the ComfyUI folder name
the Studio uses, because the trainer identifies a custom base by its being an
absolute path and a cloud pod has to receive the actual file. A checkpoint the app
can list but cannot resolve to a file on disk is left out of the dropdown rather
than offered as a name a run would ignore.

Each entry says what its format costs before you pick it:

- **no tag** — full precision, nothing to report;
- **`· fp8 cast`** — the weights are stored in fp8 under the tensor names a
  full-precision file already had, with nothing extra. The trainer up-casts it as
  it loads, and selecting it shows how many of the file's tensors are quantized
  and how many significand bits that leaves. The lost precision does not come
  back. The tag reads the *packing*, not the architecture: a file can be tagged
  this way and still be refused at load for carrying a tensor the model family
  does not declare;
- **`· packed export`** — a ComfyUI scaled-fp8 / `comfy_quant` / int8 repack. It
  carries decompression tables as extra tensors that a trainer's strict load
  rejects, so the load fails outright. Selecting it is refused, and training is
  blocked until another base is picked. Use the bf16/fp16 master instead — a
  full-model run keeps it next to its fp8 twin and the Checkpoints panel lists it.

With no ComfyUI folder configured the list falls back to the official base alone,
and says so.

### Concept face masking

Used **only** by Concept datasets that switched **Mask faces** on in *Advanced
training options* (see the dataset guide, §8). It re-weights the training loss over
detected faces so the concept learns the act rather than the identities in your
photos — it never alters your images. Both knobs are exposed rather than frozen
because no published measurement exists for the right value; preview the effect on
your own images from the training panel.

- **Head coverage (face box ×)** → `face_mask.expand`. Face detection returns a box
  running from the eyes to the chin; this grows it around its centre (biased upward,
  to catch hair) into a head. Default **2.0**, clamped to **1.0–3.0**. Higher covers
  hair and jaw, lower stays tight on the face. 2.0 matches the only published default
  for this detector/mask combination.
- **Loss weight kept on faces** → `face_mask.min_weight`. How much the masked area
  still counts, from 0 (nothing) to 1 (unmasked). Default **0.1**, clamped to
  **0.05–1.0**. Lower pushes identity out harder. **It deliberately cannot reach
  zero**: an area worth nothing isn't ignored, it's *unpenalised* — the model may
  render anything there at no cost, degraded anatomy is reported right below this
  floor, and a fully masked close-up would divide by zero in the trainer's own mask
  normalisation and kill the run.

Changing these does **not** affect the person-masking used by Character datasets;
that keeps its own historical weight.

**It needs face detection installed, and that is optional.** The detector is
InsightFace, the same optional extra face-similarity scoring uses — it is filed
under the `face_scoring` key (hence `face_scoring.python` and
`face_scoring.models_root` in *Settings ▸ Local tools*), but nothing installs it
for you. When it is missing, the **Mask faces** option says so and offers a
one-click install right there (~400 MB, a few minutes); the rest of the app is
unaffected and works exactly as before. If you launch a run with **Mask faces**
still on and the detector absent, the pre-launch report warns that the run would
train *unmasked* and asks you to confirm — it never blocks, and never trains
unmasked without telling you first. On a Python outside **3.10–3.12** InsightFace
publishes no wheels: the option explains that instead of offering an install that
could only fail, and points at `face_scoring.python` so you can aim it at a
separate 3.10–3.12 interpreter.

### Cloud GPU (vast.ai)

- **vast.ai API key** → `VAST_API_KEY` (secret). Add it to unlock **☁️ Train in cloud**. **Test** validates it (and auto-saves it first). The card includes a step-by-step guide to getting the key from [cloud.vast.ai](https://cloud.vast.ai/).

### Cloud training

Guard-rails on cost and host quality for rented pods. The card also shows a live **Spent this month** line. Everything here has a sane default — you can leave it all alone and just add the key.

Full-model Krea 2 runs also need `HF_CLOUD_TOKEN`. A separate fine-grained token is strongly recommended: grant **`repo.content.read` exactly on the Krea 2 repository the run trains from** (`krea/Krea-2-Raw`, or `krea/Krea-2-Turbo` for a Turbo run — a scope on either is accepted, and the one this run needs is required), then **`repo.content.read` + `repo.write` on one dedicated HF user or organization namespace containing only LDS deliveries**. [Create a fine-grained token](https://huggingface.co/settings/tokens/new?tokenType=fineGrained). A classic/global token with `role=write` is also accepted, but LDS shows a broad-access warning because it can modify every repository the account can write to; [create a global write token](https://huggingface.co/settings/tokens/new?tokenType=write). Read-only tokens are rejected, and Settings validates the token as soon as it is saved.

| Setting | Key | Default | Range | What it does |
|---|---|---|---|---|
| **Max simultaneous cloud runs** | `cloud.max_concurrent_runs` | `1` | 1–10 | How many cloud pods may train at once. |
| **Max price per hour ($)** | `cloud.max_price_per_hour` | `0.80` | 0.1–5 | A safety cap on the hourly offer price; pricier hosts are skipped before launch. |
| **Monthly budget ($, 0 = unlimited)** | `cloud.monthly_budget_usd` | `0` | ≥0 | A hard spend ceiling for the month; new launches are **blocked** once you pass it. `0` means no limit. |
| **Stall timeout (minutes)** | `cloud.stall_timeout_minutes` | `30` | 5–240 | If no training step progresses for this long, the watchdog rescues the logs and kills the pod. |
| **First-step timeout (minutes)** | `cloud.first_step_timeout_minutes` | `45` | 5–240 | Idle budget for the phase *before* training starts, while the pod downloads its base model (tens of GB). The clock **restarts every time the pod reports more downloaded bytes**, so an honestly slow download is never cut — only a pod that reports no bytes and no step for this long is terminated. A frozen byte counter counts as silence even when the log line keeps redrawing. |
| **Base-model download ceiling (minutes, 0 = none)** | `cloud.first_step_download_budget_minutes` | `180` | 0 or 5–480 | The **absolute** ceiling on that same phase. Because advancing bytes restart the timeout above, a host too slow to ever finish would keep its download — and your rental — alive until the runtime cap; past this it is terminated regardless. `0` removes the ceiling and leaves the runtime cap as the only backstop. |
| **Max runtime (minutes)** | `cloud.max_runtime_minutes` | `480` | 30–1440 | Hard stop on the whole run, whatever it is doing: the newest checkpoint is rescued, then the pod is terminated. Enforced by the out-of-run supervisor too, so it holds even if the run's own supervision dies. |
| **Freeze watchdog (minutes, 0 = warn only)** | `cloud.freeze_watchdog_minutes` | `45` | 0 or 15–480 | Last-resort net for a run whose own supervision stopped answering (an app restart, a wedged connection to the pod): if a **training** run reports no progress at all for this long, the pod is terminated from outside — the stall timeout above can only act while the run is still being watched. Checkpoints already downloaded are kept. Set to `0` to only see the warning on the run card and never cut automatically. **"No progress" means the pod, not the app.** The clock is kept in the database and advances only when something actually moves on the host (the training step, the download's byte counter, a checkpoint landing), so restarting the app no longer sets it back to zero, and a run whose supervision keeps repeating the same sentence is no longer counted as alive. Phases that are silent by design (booting, downloading the result) are never judged on this value; they get a fixed 2 h floor and the runtime cap as backstop. The **dataset upload** used to be one of them and is not any more — it reports the bytes it pushes, so it has its own, much shorter setting below. |
| **Dataset upload stall (minutes, 0 = never cut)** | `cloud.upload_stall_minutes` | `25` | 0 or 5–480 | **Not a time limit on the upload.** A large dataset is allowed to take as long as it needs — the run card shows the files and gigabytes going across — and this value is how long the rented machine may sit with **no data at all** arriving before the run is given up and the pod released. It exists because that phase used to report nothing: a 24 GB dataset that never reached the pod billed for two hours before anyone could tell a slow upload from a dead one. Set `0` to never cut; turning the freeze watchdog off turns this off too. |
| **Unreachable grace (minutes)** | `cloud.unreachable_grace_minutes` | `6` | 1–60 | How long a running pod may stay unreachable (a vast.ai network blackout) before the run is given up and auto-retried on a fresh host. Raise it if healthy runs die with *pod unreachable*. It also bounds the **reconnection after an app restart**: a run whose job was already training is given this long to answer again — asked directly, not through the vast.ai listing — before it is given up, and the pod is never told to stop on a verdict reached without reaching it. |
| **Min host reliability** | `cloud.min_reliability` | `0.98` | 0.9–0.999 | vast.ai reliability floor. Lowering toward 0.95 surfaces cheaper hosts at a higher boot-failure risk. |
| **Verified hosts only** | `cloud.verified_only` | **on** | toggle | Restrict to vast.ai's verified hosts (the historical, safer behaviour). |
| **Secure Cloud only** | `cloud.secure_cloud_only` | **off** | toggle | Restrict to vast.ai's *datacenter* (Secure Cloud) tier — usually narrows the market and raises the price, so it's opt-in. |

### Advanced options (per run)

#### Full-model (dense) recipe

Full-model training keeps a **locked** recipe — batch 1, bf16, Adafactor and
gradient checkpointing are what make a 12B model fit on one 80 GB card, and the
panel now names each lock and its reason instead of just greying it out. The
values below are editable because they change the *result*, not whether the run
fits.

**Raw, Turbo, or a checkpoint of your own — with one warning and one real
refusal.** The full-model panel has its own base picker: the Raw/Turbo switch,
the Krea 2 checkpoints installed on this machine, and **Custom weights…** for a
local `.safetensors`. A custom base travels to the rented GPU through a private
repository on your Hugging Face account, exactly as it already did for LoRA
runs, and the launch verifies the pod's own credential can read it before
anything is rented.

*Turbo is allowed and unmeasured.* Turbo is speed-distilled, and a dense run
rewrites the weights that distillation lives in. On the distilled models where
this *has* been measured, the result stays a valid checkpoint and simply stops
being fast (back toward real CFG and 25-30 steps); for Krea 2 specifically
nobody has published the measurement. The panel says exactly that before you
spend anything — Krea's own recommendation is still to train a LoRA on Raw and
apply it to Turbo — and then lets you through. See the dataset guide, §10, for
what is known and for the published trick to get the speed back afterwards.

*A structured fp8/int8 export is refused, and that one is mechanical.* A
ComfyUI scaled-fp8 build (or any `comfy_quant`/int8 repack, including this app's
own fp8 twin) carries dequantization tensors the architecture never declares,
and ai-toolkit loads a base with `strict=True` — the load itself raises. Pick
the bf16/fp16 build of the same model. A **bare** fp8 cast, which adds no key of
its own, stays allowed and says what it costs.

| Setting | Key | Default | Notes |
|---|---|---|---|
| **Steps** | (per launch) | adaptive | ≥ 500 when set explicitly. |
| **Preview prompts** | `sample_prompts` | generic per dataset kind | Up to 8 lines, `{trigger}` marks the subject. Shared with the LoRA lane on purpose — same dataset, same subject. The defaults describe nobody, and these images are the only way to judge a run while it is still costing money. |
| **🎲 Use dataset captions** | (fills `sample_prompts`) | — | Replaces the field with up to 5 captions drawn at random from this dataset's **kept** images (the long caption, the one the run trains on). Click again for a new draw. Captions are pasted as they are: the trigger is added by the run when it is missing, and never for a style dataset. Disabled while the dataset has no captions yet. |
| **Learning rate** | `dense_lr` | `1e-6` | 1e-7 – 5e-6. |
| **Resolution** | `dense_resolution` | `1024` | `768` or `1024`. 768 trains faster and cheaper, at lower fidelity. |
| **Checkpoint every / keep** | `dense_save_every` / `dense_max_step_saves` | `250` / `1` | ≥ 100 steps; keep 1–3. Each kept checkpoint is ~26 GB of **private** Hugging Face storage — the panel states the total before launch and the pre-check uses the same number, never the shipped default. |
| **Keep the bf16 master** | `dense_keep_bf16` | on | Keeps the ~26 GB master next to the fp8 export. fp8 is a lossy, one-way export: without the master the model can never be continued, merged or re-quantized. |
| **fp8 export** | `dense_fp8_export` | on | Quantize the finished checkpoint on the pod and upload the ~10 GB ComfyUI file. A failed export never fails the run. |
| **Images per step** | `dense_grad_accum` | `1` | 1, 2, 4 or 8. Gradient accumulation: how many images are averaged into each optimizer step. Batch size stays 1, so this costs **no extra VRAM** — it costs time. 4 images per step ≈ a run 4× longer on a GPU billed by the hour. Checkpoint count, cadence and storage are unchanged. |
| **Learning-rate schedule** | `dense_lr_schedule` | `constant` | `constant`, `constant_with_warmup` or `cosine`. Warmup eases the first steps instead of hitting a 12B model at full rate from step 1; cosine fades the rate to zero by the last step. |
| **Warm up over** | `dense_warmup` | `100` | 10 – 1000 steps. Only applies to `constant_with_warmup` — the schedulers behind the other choices reject the value outright, so it is not sent with them. |
| **Noise schedule** | `dense_timestep_type` | `linear` | `linear`, `sigmoid` or `weighted`. Which noise levels the run trains on; `weighted` keeps the linear draw but weights the loss on a bell curve. |

**Two knobs are deliberately missing, and that is not an oversight.** Both exist
in AI Toolkit and both are *harmful on this specific model*, so the full-model
card does not offer them:

- **EMA** (weight averaging) keeps a second copy of every trained parameter on
  the GPU, and a third one whenever it saves. For a LoRA that is a few hundred
  MB; for a 12B full model it is roughly +26 GB, then +26 GB again at the first
  checkpoint, on top of an unquantized model and its gradients. The run would
  die at its first save. EMA remains available for **LoRA** training, where it
  is cheap.
- **min-SNR weighting** needs a signal-to-noise table that only diffusion models
  of the older kind carry. Krea 2 is flow-matching and has none, and the
  trainer's attempt to build one fails silently at startup — so the run does not
  refuse at launch, it crashes in the middle of the loss computation an hour
  into a paid pod. Refusing it up front is the cheaper failure.

`shift`-style noise schedules are absent for the same class of reason: AI Toolkit
computes their shift from an image-token count that assumes a field Krea 2's
denoiser names differently, so the value silently comes out four times too large.
It is offered for LoRA training (where the same flaw applies to Krea and is
documented in the code) but never for full-model runs.

**Which AI Toolkit these statements describe.** LoRA training runs the AI Toolkit
installed on *your* machine, which moves whenever you update it. Full-model
training is cloud-only and runs the AI Toolkit baked into the rented pod's image,
which is pinned (`cloud.image`). The two are different codebases at different
dates; everything above was verified against the pinned one, and a test fails if
that pin moves without the verdicts being re-checked. Each run now also records
the image the pod actually booted, so a run can say for itself which trainer
produced its weights.

**Getting the fp8 file in one click.** When a run has delivered its model, the
recipe card's *Quantize a model to fp8* block is already aimed at it: **✨
Quantize to fp8** does the whole chain with nothing to type — it fetches the
master out of your private Hugging Face repo, converts it, and leaves the fp8
file in ComfyUI's own models folder
(`models/diffusion_models` for a dense transformer, `models/checkpoints` for an
SDXL-style full checkpoint; with ComfyUI unconfigured it falls back to the app's
`data/models/…` and **says so**). Before it starts it states:

- **which checkpoint it takes.** A dense repo usually holds the final save *and*
  several ~26 GB step snapshots with nearly the same name. One rule decides —
  the final save wins, otherwise the highest step — and it is the same rule that
  stamped the file the card lists, so the two can never disagree;
- **where the file lands**, spelled out, before and after;
- **what it costs in disk.** What is still to download, the fp8 file's own
  ceiling and 2 GB of working headroom, compared with the free space of the
  volume that *really* holds that folder — `realpath` first, because a ComfyUI
  models folder is very often a junction onto another drive. Too little is a
  refusal that writes out every term and offers another folder, not a failure at
  90 %. **What this forecast accepts, the conversion does not then refuse**: the
  threshold used to be applied only when starting, so the button stayed enabled
  and the refusal arrived after the click.

The download is the long part (~26 GB) and it is a real job: progress in bytes,
a **Stop** button, and resumption from where it stopped — stopping never throws
away what already came down. Afterwards the master is **kept** by default (it is
the only file you can train from again, merge or re-quantize); deleting it is one
radio button away, with its size on it.

**The path field, for everything else.** The same block still takes a full path
to any full-precision `.safetensors` — a file nothing in the app points at — and
it pre-fills with your **custom training base** when there is one. The same tool
is in **Settings ▸ Storage** for a model that has nothing to do with a dataset,
where it is documented in full. It writes `<name>_fp8.safetensors` into the same
ComfyUI folder as the one-click path; the source is never modified, an existing
output is never silently overwritten, and the result is re-opened and verified
before it reports success. It runs on the **CPU** (one elementwise cast per
tensor — disk-bound, not GPU-bound), one at a time app-wide. It refuses a file
that is already quantized and refuses a LoRA/adapter. Note this is **not** the
`quantize` training option, which only shrinks the model in memory while it
trains and writes no file.

**Testing what it delivers.** The artifact is a **Raw (undistilled)** Krea 2
checkpoint. Turbo-style few-step settings render a blurry sketch on it; use
**CFG ~4 (3.5–5) and 20–30 steps** — the same settings the run previewed with.
The Test Studio pre-fills them when the selected base looks Raw / full / fp8.

**Quantized bases are refused.** Choosing a community fp8/int8 export as *Custom
weights* is rejected at selection with *"This is an inference-only quantized
export — training needs the bf16/fp16 version of this model."* The check reads a
few kilobytes of file header (quantization markers and tensor dtypes), so it
costs nothing; a header it cannot read is let through rather than guessed at.



These live under **⚙️ Advanced options** in a dataset's training panel — rank, resolution, save/sample cadence, optimizer, scheduler, EMA, LoKr and more. Each carries its own inline **Why/How** note, so they aren't repeated here. Two are worth calling out because of a caveat.

#### Krea 2 Raw · LoKr likeness — a reported community starting point

The built-in **Krea 2 Raw · LoKr likeness** preset is deliberately narrow: it is
shown only for a **Character** dataset on a compatible Krea 2 Base/Raw variant.
It turns a [reported Krea 2 Raw LoKr recipe from the Stable Diffusion
community](https://www.reddit.com/r/StableDiffusion/comments/1v2vsqm/almost_perfect_likeness_in_750_steps_krea_2_lokr/)
into a named, inspectable starting point — **not** into a promise that a different
person, image set, captioning style or checkpoint will match at the same step.

The post linked a full Pastebin configuration, but that Pastebin has since been
deleted. LDS therefore records only the values the post actually reports:
**LoKr factor 16**, **768 px**, **Automagic2** with initial learning rate
**`1e-4`**, **Sigmoid** timestep weighting, **Balanced** content/style mode,
**Differential Guidance** at scale **3**, and a checkpoint/preview cadence of
**250** steps. The Krea-only Expert controls show those values plainly, and the
run snapshot carries the factor, content/style mode and Differential Guidance so
you can compare an experiment later instead of trusting a remembered recipe.

The post does **not** publish the LoKr linear rank or alpha. LDS keeps its
existing Krea Character **32/32** choice rather than inventing a rank/alpha pair
and presenting it as sourced. Likewise, the reported **3000 total steps** are
not forced by the preset: LDS keeps its adaptive step policy so a small dataset
is not silently overcooked. To reproduce that target intentionally, type
**3000** into the **Steps** box for that run; leave it empty to use the adaptive
policy. Treat the intermediate saves — including the early ones — as candidates
to compare in Test Studio, not as proof that a specific step will be best.

**One rule applies to all of them: they are stored per DATASET, not per family.** Switching **LORA TYPE** keeps every advanced setting you had — which is what you want for rank, optimizer or resolution, and what you do **not** want for the two settings below, whose right value is different on every family. Those two are handled explicitly:

- **Memory saving carries over, and is now said out loud.** `quantize` / `quantize_te` / `low_vram` are a statement about *your card*, and your card doesn't change when the family does — so the values follow you. What changes is whether the card still suffices: switching them off on Anima or SDXL (2B, where **off** is the calibrated default) and then moving to Krea 2, FLUX.1, FLUX.2 Klein or Z-Image used to build an unquantised 12B run in complete silence. Both the panel and the **pre-launch check** now name which saver is off, what that family needs without it (see the estimates below) and what your card reports. It stays a **warning, never a blocker** — a big card legitimately runs unquantised — and it is deliberately **not** dropped on the cloud lane, where the mistake bills rented GPU-hours. The warning is also *provenance-blind*: unticking a box directly on Krea 2 with a 24 GB card gets the same sentence as inheriting it from Anima, because it is the same danger.
- **Timestep weighting is remembered per family instead.** `sigmoid` is Z-Image's and FLUX.1's canonical flow-matching schedule, `linear` is Krea 2's, `weighted` is FLUX.2 Klein's and Anima's — the value has no meaning outside a family, and carrying it over changed the LoRA that came out with nothing at all to observe afterwards. Each family now keeps its own choice: switching hands the incoming family its own value (or **Auto**, its canonical default, if you never set one there), and coming back finds yours exactly where you left it. Nothing is destroyed and nothing is asked. **Existing datasets are untouched** — a dataset that never changes family keeps every setting it has, byte for byte.
- **Resolution stays global on purpose.** 768 and 1024 mean the same thing on every family, so remembering it per family would mean silently raising your 768 back to 768+1024 on a switch — a new silent change to fix an old one. The one combination that costs you (1024 on a 12B with a small card) is a pre-launch row instead, and that row no longer tells you to "drop the resolution to 768" when you are already at 768.


- **Memory saving** — three switches (`quantize`, `quantize_te`, `low_vram`) that used to be hard-coded. **The defaults have not changed:** Z-Image, Krea 2, FLUX.1 and FLUX.2 Klein quantise the base model and the text encoder to `qfloat8` and stream blocks between CPU and GPU, which is what makes a 12B model train on a 24 GB card; Anima and SDXL are small enough to run without any of it. Turning them **off** trades VRAM for precision and speed — worth it only if your card is bigger than the target. As a rough order of magnitude with the savers off: **Z-Image ≈ 18 GB**, **FLUX.2 Klein 4B ≈ 14 GB**, **FLUX.2 Klein 9B ≈ 24 GB**, **Krea 2 / FLUX.1 ≈ 30 GB** (estimates: bf16 weights plus headroom, not a measurement on your exact card). The panel detects your GPU and says which side of that line you are on; if it can't (no NVIDIA card, `nvidia-smi` missing), it falls back to a generic note and blocks nothing. ⚠️ **The failure mode is slowness, not a crash.** On Windows there is no clean out-of-memory error: the driver silently pages to system RAM and the run creeps along for hours. If a run that used to take 40 minutes is still going after three, put the switches back. The setting also works the other way — a small card can turn quantisation **on** for Anima or SDXL. It's recorded in each run's snapshot and in the ⎘ Share config, so two runs can be compared honestly.

- **Masked training (background at 10%)** — **on by default**, and since the 28/07 wave it is stored **on the dataset** instead of in the browser you set it in. A person mask is generated for every image (rembg, CPU) and the background only weighs 10% of the loss, so the LoRA binds the identity to the subject rather than to the room. What changed and why it matters: the toggle used to be a `localStorage` preference (`trainMasked_v1`) that reached the server only as a launch parameter, so **the readiness badge could not warn about it**, opening the app **from a phone or another machine** silently reverted to the default, and the value appeared in **no run snapshot** — two runs differing only by masking looked identical when compared. All three are fixed: it is patched like any other advanced option, stamped into every run's snapshot (local **and** cloud), and the pre-flight badge now carries a **Masked training ready** row that says *rembg is not installed — this dataset is set to masked but the run trains unmasked* **before** you open the launch dialog. It is a **warning, never a blocker** (a run without masks is a valid run), and it is **not** filtered out on the cloud lane: the masks are generated locally and uploaded with the images, so rembg missing here means the **paid** run trains unmasked. **Concept and Style datasets force it off** (a person mask erases the recurring concept, and a style must be learned across the whole frame), and **slider mode** ignores masks entirely — the panel says so instead of hiding the control. **Existing datasets do not change behaviour:** an untouched dataset resolves to the historical default (on). A browser that had explicitly turned masking **off** is asked once, in the training panel, whether to carry that choice onto the dataset or keep masking on — nothing is written until you answer, in either direction.

- **Dual captions (long + short)** — off by default. When on, the run uses ai-toolkit's native `short_and_long_captions`: every image trains with **both** its full caption and a short one (text-side augmentation, so the LoRA leans less on any single wording). The short variant is **derived from the long caption** the next time you (re-)caption — text-only, via the local vision model, honouring the same kind rules (no trigger; the identity/concept/aesthetic stays omitted) — and you can edit it per image in the **⛶** caption editor. **Local training only for now:** the cloud pod's dataset upload doesn't carry the JSON caption file the short is read from, so cloud runs train on the long caption alone. **Not available on Krea 2 or Anima:** those families pre-cache their text embeddings and unload the text encoder, so no second caption can be encoded — the toggle is reported as ignored on the training panel and in the pre-launch check, and the run trains on the long caption alone (issue #22, reported by 1Tomber).

## Storage

Everything about the disk in one place: which folder holds what, how big it is,
and how to put any of it on another drive. Before this tab those answers were
scattered across three screens, and the two directories that actually fill a
drive — the cloud run staging and the trained checkpoints — were hard-coded and
invisible.

**Sizes are measured only when you ask.** Walking a hundred gigabytes of datasets
takes seconds, so **📏 Measure everything** is a button, never something the tab
does while you read it. Until you press it a folder reads *not measured*, which
is deliberately not the same as *empty*.

- **What lives where** — one row per category: dataset images, image banks, bank
  source images, cloud run staging, the checkpoint store, the trash, the run image
  archive, backups, the ai-toolkit install, the Hugging Face cache and the app's
  own build. Each row shows the **effective path**, what it holds, the free space
  on that drive, and a **movable** tag when it can be relocated from here.

### Moving a folder to another drive

Four roots can be pointed anywhere: **Dataset images root**
(`paths.dataset_images_root`), **Cloud run staging** (`paths.cloud_runs_dir`),
the **Checkpoint store** (`paths.checkpoints_dir`) and **Video datasets**
(`paths.video_datasets_dir`). All four default to **empty →
a folder inside the app's data directory**; the field's *Reset to default* gives
that implicit state back rather than writing today's path in.

Type a path, press **Check folder**, and the app proves it can write there — by
actually writing a probe file, because permission bits lie on Windows. A relative
path, an uncreatable folder, or a target *inside* the folder it would replace is
refused with the reason. Then **you choose**, and nothing happens until you do:

- **Move what is already there** — every file is copied to the new folder first,
  and the old one is only removed once the last byte has landed. A progress bar
  shows files and percentage. If the destination drive has less free space than
  the folder needs, the button is disabled and says both numbers.
- **Start using it empty** — the new folder is used from now on. **Nothing is
  copied and nothing is deleted**: the old folder keeps its files and the app
  simply stops looking at them. On a full C: this is often the only choice that
  fits.

The setting is saved **after** the files have moved, so an interrupted move never
leaves the app pointing at a half-filled folder.

- **Dataset images root** → `paths.dataset_images_root`. Where dataset images are stored. Default **empty → `<data dir>/datasets`**. This folder (and every dataset folder under it) is refused as an **image bank** source: a bank points at a live folder and can delete from it, so the two must never share files — see *Using the app → A bank and a dataset never share files*. Moving this root onto a folder an existing bank already uses is not blocked here, but that bank will say so the next time you open it, and its 🗑 Delete rejected will be refused.
- **Cloud run staging** → `paths.cloud_runs_dir`. The working area of cloud training runs: the exported dataset copy, the sample images and the mirrored log, one `run_<id>` folder per run. Default **empty → `<data dir>/cloud_runs`**. This is the folder that grows to tens of gigabytes, and the one a cleanup empties.
- **Checkpoint store** → `paths.checkpoints_dir`. Where the `.safetensors` your cloud runs produce are kept, one `run_<id>` folder per run. Default **empty → `<data dir>/checkpoints`**. **No cleanup in the app ever removes a file from here** — only you can, from the Checkpoints panel or by emptying the trash.
- **Video datasets** → `paths.video_datasets_dir`. The flat folders of `.mp4` clips (plus their homonym `.txt` captions) that a video bank produces when you promote a selection. Default **empty → `<data dir>/video_datasets`**. **This is where the video lane actually uses disk.** A video bank itself stores almost nothing — only timestamps and one small thumbnail per detected shot — because cutting a clip means re-encoding it, and that is paid once, at promotion, for the clips you kept.
- **Video banks (working data)** → no path setting; it follows the data directory. Holds the shot thumbnails only: never your source videos, which a bank references in place and never writes to, and never the clips, which do not exist until you promote them.

### Why checkpoints have their own folder

They did not, and it cost weights. "Clean finished runs" trashed the whole staging
directory while advertising *checkpoint duplicates already imported* — but a
checkpoint that had never been deployed to ComfyUI had no duplicate: staging was
its **only** copy. Emptying the trash afterwards destroyed it.

Trained files now land in the checkpoint store from the moment they are
downloaded, and no cleanup path can reach them. An install that trained before
this change is swept once at startup; **Move stray checkpoints into the store**
re-runs that sweep on demand and is safe to press at any time.

### Cloud run housekeeping

- **What a cleanup actually does.** Cleaning a finished run — from this tab or with
  the 🧹 button on the Runs hub — moves its **dataset copy, its sample images and
  its logs** to the trash. It never moves a `.safetensors`. Files go to the trash
  **on the same disk**, so the space only comes back when you **empty the trash**.
- **Which runs are spared.** A run that is still active, and a run whose pod was
  kept for manual recovery **while that recovery window is still open**
  (`cloud.max_runtime_minutes` after the run ended). Once the window has closed the
  pod is gone, so its staging is cleanable like any other — it used to stay frozen
  forever, holding tens of gigabytes for a pod that no longer existed.
- **Find unclaimed run folders** — `run_<id>` folders on disk that **no run in the
  database points at**, left behind by a restored backup, a deleted database or an
  interrupted cleanup. The cleanup used to answer *already clean* while 25 GB sat
  right there. They are now listed with their size, and any checkpoint still inside
  one is **rescued into the store** before the folder goes to the trash.

### Trash and archives

- **Trash** — **Open folder** and **Empty trash**. Everything the app deletes goes here first; emptying is the one destructive action, and it asks for confirmation. It lives on the same disk as your data, so a cleanup gives space back only once you empty it.
- **Run image archive** — its size, its ceiling, and **Clear archive**. When a training run is launched, a **deduplicated** copy of the images it trains on is kept so that comparing two runs can still *show* an image you have since deleted from its dataset. Copies are **content-addressed**: relaunching an unchanged dataset stores nothing the second time, and only images that were added or re-edited cost anything. Clearing it keeps your runs, their settings and their caption text — you only lose the ability to look at images that are no longer in their dataset. The ceiling is `provenance.archive_max_gb` (see *Config-file-only settings*); past it, nothing more is stored and the compare panel says the picture is unavailable instead of showing a wrong one.
- **Back up everything** — not on this page but on the **Datasets library**: one button archives every dataset, its **training history** and your settings into a single file (⬇ download or 📂 open folder), and the library's **Import backup** restores it — datasets come back under **Trained**, not "Not trained yet". Tick **Include trained LoRAs** to bundle the (large) trained `.safetensors` too. **API keys and tokens are never included** — re-enter them on the new install. See *Using the app → Back up everything*.

### Quantize an existing model to fp8

A full-precision `.safetensors` is roughly **2.5× the size ComfyUI needs** to
generate with it. **Quantize a model to fp8** takes any full-precision model — a
~26 GB one downloaded from Hugging Face, a checkpoint an earlier full-model run
delivered, a large finetune someone shared — and writes `<name>_fp8.safetensors`
into ComfyUI's own models folder, loadable with the standard *Load Diffusion
Model* node without moving anything by hand.

This is the **same tool** as the one on the full-model recipe card (*Training →
Full-model (dense) recipe*), reachable here **without a dataset**: it was only in
that card at first, which nobody who simply downloaded a model ever opens.

- **The source is never modified**, and an existing output is never silently
  overwritten. The result is re-opened and its scaled tensors counted before it
  reports success — a file that cannot be read back is reported as a failure, not
  as a smaller model.
- **It refuses before you click, not after.** The plan states the source size,
  the name and folder it will write, the expected size and how many matrices are
  quantized. A file that is **already quantized**, a **LoRA/adapter**, an output
  that already exists and a drive without room are refused *there*, with the
  reason, and the button stays dead. Reading the plan costs a few kilobytes of
  file header. Every condition that would stop the conversion is evaluated here:
  a refusal that only existed at start time left the button enabled and landed
  after the user had committed.
- **The disk budget is derived, not a flat number.** What is still to download +
  the fp8 file's own ceiling + 2 GB of working headroom, against the free space
  of the volume that really holds the destination (`realpath` first — model
  folders are often junctions onto another drive). A flat 30 GB floor used to
  refuse a 12.8 GB conversion on a drive with 17.6 GB free. When a drive really
  is too full, the refusal offers to write the file to another folder.
- **It runs on the CPU**, one conversion at a time app-wide, so it never competes
  with ComfyUI or a training run for VRAM. It is disk-bound (measured ~1.2 GB/s).
- **It runs in a separate Python**, the one that has `torch` — this app installs
  without it on purpose. See `quantize.python` in *Config-file-only settings*. An
  environment that cannot do the work is a refusal in the plan, naming what to
  install.
- **Nothing is memory-mapped.** The checkpoint is read one tensor at a time, so
  the size of the file has no bearing on whether it can be opened. Mapping a
  26 GB file used to reserve 26 GB before reading a single number, which failed
  outright — with a "paging file is too small" error — on any machine whose
  pagefile was not unusually large.
- **fp8 is a one-way, inference-only export.** A quantized file is refused as a
  training base, so keep the full-precision one if you may ever want to continue,
  merge or re-quantize that model. And this is **not** the `quantize` training
  option, which only shrinks a model in memory while it trains and writes no file.

The result lands in ComfyUI's own folder (`models/diffusion_models`, or
`models/checkpoints` for an SDXL-style bundle), so there is nothing to move.
With ComfyUI not configured it falls back to the app's `data/models/…` and
**says so** rather than pretending. **A model a full-model run delivered needs no
path at all**: inside a dataset, the same block is already aimed at it — see
*Training → Full-model (dense) recipe*.

### Hugging Face storage

Full-model (dense) cloud training now delivers its ~26 GB result **to this
computer first** — the checkpoint folder set above — and only then backs the
master up to a **private** Hugging Face repo. Custom training bases are still
cached there (one `lds-base-<hash>` repo per distinct base), so the **private
storage allowance** still matters; what changed is that nothing is pushed *while
the run trains*. A full allowance used to arrive as
`403 … private repository storage limit reached` at the end of a paid run and
end it (this happened at step 2750 of 3000). It can now cost only the backup —
and with it the ability to continue that model later.

The **Hugging Face storage** card is the answer to that. Nothing here runs on
page load; **Check storage** is an explicit click.

- **What it measures.** Hugging Face publishes **no quota endpoint**. The card
  therefore sums the `usedStorage` the Hub reports for each of your private
  repos (models and datasets) — the same number you can read on
  huggingface.co — and compares it with what one dense run needs: one checkpoint
  (sized from what your **past runs actually delivered**, else ~26 GB) × the
  saves kept, plus the **fp8 twin** the run also uploads, plus a margin. The card
  spells that sum out term by term, in the same words as the launch refusal, so
  the total and its breakdown can never tell two different stories.
- **What it cannot know.** The **ceiling**. The published plan table says 100 GB
  of private storage for a free account and 1 TB for PRO, but a real refusal has
  been observed well below the free figure. Everything the card says about
  "allowance" and "room left" is therefore an **estimate**, clearly labelled as
  one, and the launch refusal it produces is always confirmable — *Train anyway*
  exists on purpose. Storage is also billed over git history: a superseded
  revision keeps counting until a repo's history is squashed, which this
  estimate does not model.
- **One blind spot, on purpose.** The *launch* pre-check measures with the
  dedicated `HF_CLOUD_TOKEN` only — dense runs are deliberately cut off from the
  general `HF_TOKEN`, and reading it here to sharpen an estimate would widen
  that boundary. So a fine-grained cloud token too narrow to *list* its delivery
  namespace makes the forecast **unknown**, and unknown never blocks a launch.
  The card below uses `HF_TOKEN` and stays fully sighted, so **Check storage**
  shows the whole picture even when the launch forecast could not.
- **Custom-base caches.** Every `lds-base-*` repo is listed with its size, the
  local file it mirrors, and the last cloud run that used it. Each is a **cache**:
  the local file is the source of truth, so deleting one costs a single
  re-upload the next time you launch on that base — *unless* the local file is
  gone, in which case the card says **Only copy** and the confirmation says so
  too. **Delete all** sweeps them in one go and reports partial failures.

| Setting | Key | Default | Notes |
|---|---|---|---|
| **Full-model delivery** | `cloud.full_transformer.delivery` | `both` | Where a finished full model goes. `both` = download it here, verify it (byte count **and** a re-read of the safetensors header), release the pod, then upload the master to the private repo as a backup. `local` = skip the backup and save the quota — the run can then **not** be continued later, because a 26 GB checkpoint can only reach a fresh pod from the Hub. `hub` = the historical Hugging-Face-only delivery, with its mid-training pushes. Runs launched before this setting existed keep the `hub` behaviour for ever. |
| **Private storage allowance (GB)** | `cloud.full_transformer.private_storage_limit_gb` | `0` | What the pre-check compares against. `0` = infer from the plan documented by Hugging Face (100 GB free / 1 TB PRO) — a guess, as above. Put your account's real ceiling here to make the check exact. |

**Which forecast blocks, and which one only warns.** With `delivery = hub` the
repository is the artifact's only address, so a forecast that does not fit is a
**refusal** (confirmable — the ceiling is an estimate). With a delivery that also
brings the model home, the same forecast is a **warning shown at launch**: the
run is unaffected, but you are told before the GPU is rented that this model will
probably not be resumable. A second, separate check looks at **this machine's**
disk (`LOCAL_DISK_FULL:`) and refuses — also confirmably — a launch whose
delivery plainly will not fit in the checkpoint folder.

**When a transfer fails.** The pod is destroyed **only** after the local file is
proven. Anything that goes wrong before that (a truncated stream, a full drive, a
cancelled transfer, the runtime cap) closes the run as **error_pod_kept** with the
machine still alive, and the Runs page grows a **Fetch to this computer** button
that resumes the download from where it stopped — an interrupted transfer keeps
every byte it had already written. Cancelling is a second click on the same
button.

**When a run hits the Hugging Face wall anyway.** A `hub`-only dense run whose
checkpoint push is refused for storage says so: the run card reads *HF private
storage full — free space then resume from the kept pod*, and the run closes as
**error_pod_kept** — the paid pod is **kept**, not destroyed, for the recovery
window (`cloud.max_runtime_minutes` after the failure). Free space here, then use
**Verify HF delivery** on the Runs page once the push lands.

## Server & access

How the app binds and who can reach it. **These are the settings that need a restart** — the card shows a **Running vs Saved** banner and a **Save & restart to apply** button that does it in one click.

- **Port** → `server.port`. The port the app listens on. Default **`5050`**. Change it if something else owns the port (on macOS, port 5000 is taken by AirPlay Receiver).
- **Available on the local network** — a toggle that flips the bind host between `127.0.0.1` (this machine only, the default) and `0.0.0.0` (reachable from your LAN — phone, tablet, another PC). The token and phone controls below only appear once this is on.
- **Require an access token** → `server.require_token`. Default **off** — a home LAN is treated as trusted, so LAN access is open and there's no token to type on a phone. Turn it **on** to demand a token from remote devices; requests from localhost never need one.
- **Access token** → `server.access_token`. Shown only when the token gate is on: a read-only field with **Generate new token** and **Copy**. It's persisted, so it survives restarts. Open `http://<machine>:<port>/?token=<token>` once from the remote device and a signed session cookie takes over.
- **Open it on your phone** — a card with a scannable **QR code** and copyable URLs built from this machine's real LAN IP (and Tailscale IP, if present). No guessing which address to type.

**Trap:** if you launched via `start.bat` with `LDS_PORT` set, that variable can override the port in your config. The in-app **Save & restart** pins host and port for the relaunch, precisely so the restart lands on the port you chose rather than the one the script forced.

## Maintenance

Keeping the **app itself** healthy: updating it, and getting a bug report out of it. No setting lives here, only actions. Everything about the **disk** — the trash, the run image archive, the dataset root and the folders that fill a drive — moved to *Storage* above, where those questions are answered together.

- **Updates** — **Check for updates** and **Update & restart**, plus a *see what's in this update* compare link. **The button adapts to how you installed.** A **git checkout** fast-forwards to the latest commits. A **packaged (ZIP) install** announces the release and its size (*Update to vX — download ~XX MB*) and shows a **live progress bar** while it downloads and installs (a release ZIP is far larger than a git pull), then backs up the current files and swaps in the new ones — keeping `data/`, `config.json`, `.env` and your `.venv` untouched — and restarts. A mid-way failure rolls back automatically, so a broken download never leaves you with a half-updated install. If the app can't identify a downloadable release (no ZIP asset, or offline), the button steps aside and links to the releases page instead of promising an update it can't perform.
- **Back up everything** — not on this page but on the **Datasets library**: one button archives every dataset, its **training history** and your settings into a single file (⬇ download or 📂 open folder), and the library's **Import backup** restores it — datasets come back under **Trained**, not "Not trained yet". Tick **Include trained LoRAs** to bundle the (large) trained `.safetensors` too. **API keys and tokens are never included** — re-enter them on the new install. See *Using the app → Back up everything*.
- **Diagnostic report** — a one-click, **paste-safe** report for bug reports: it carries the version, capability status and a log tail, with **no secrets** and file paths reduced to booleans (present/absent). Safe to drop into Discord or a GitHub issue.
- **Server log** — a live tail of the server log, with **Copy all**, for when you need to see what just happened.

## Per-dataset settings

Separate from everything above: these live **per dataset**, in the **⚙ Dataset settings** modal you open from the workspace. They travel with that one dataset and don't touch the global Settings page.

- **Name** — the dataset's display name. **Display only** for Character and Concept datasets: it never appears in a file name, so changing it touches nothing on disk (the *trigger word* names produced files — see below). **On a Style dataset it means more**: a Style is always-on and has no visible trigger, so its name is its only editable identity — renaming it also renames the LoRAs, run folder and export it already produced (and is refused while a run is live, same as a trigger change).
- **Dataset kind** *(🧑 Character / 💡 Concept / 🎨 Style)* — the nature of the LoRA, chosen at creation but changeable here. It is the disruptive setting, so picking a different pill reveals a confirmation block that spells out **what changes** and **what is kept** before you save:
  - *What changes* — the **caption strategy** (Character leaves out identity; Concept leaves out the recurring concept; Style leaves out the aesthetic), which **panels show** (Reference photo, Generate variations and Face analysis are Character-only — they appear when you become a Character and are hidden otherwise), the **trigger's role** (Style has none; switching to Character/Concept brings the field back, prefilled), and Character-only settings such as **face/body fidelity**. Switching **to Concept** requires a concept description.
  - *What is kept* — **nothing is deleted.** Every image, its caption text, keep/reject status, face scores, watermark work and **training history** stay exactly as they are (past runs are named by the model family and trigger, never the kind). A concept description is remembered so switching back restores it.
  - Existing captions were written for the **old** kind and are **not** rewritten automatically — use **🔄 Re-caption** in the Captions section to apply the new strategy. The switch is refused while the dataset has work in progress (generation, captioning or a quality pass) — wait for it to finish.
- **Trigger word** — the word you put in prompts to summon this LoRA (Character and Concept datasets). Safe to change anytime — it's added at export, so existing captions don't need redoing. It is also **the name everything this dataset produces carries** (the deployed LoRA, the training run folder, the export, the job config), so changing it **renames all of them to match** and repoints the Test Studio history and cloud runs at the new names — a toast tells you how many files moved. Two guards: if the new trigger is already used on disk by another dataset, **nothing** is renamed (never half a set) and the old names are kept; and the change is **refused while a training run is live**, because that run folder is what training resumes from — stop it or let it finish first. **Style datasets don't have one**: Style is always-on, and the modal shows a note reminding you to control the effect with the LoRA weight instead.
- **Concept description** *(Concept datasets only)* — the thing the LoRA learns, i.e. exactly what captions must **omit**. Editing it rebuilds the caption avoid-list, so **re-caption** afterwards to apply the new list to images already captioned.
- **Subject type** *(Human / Animal / Creature / Object / Other / Anime)* — **what your reference actually is**, chosen right in the **🎬 Generate variations** panel (it travels with the dataset). It is **orthogonal to the dataset kind**: a specific dog is *Character + Animal*, "dogs in general" is *Concept + Animal*. Anything other than **Human** switches two things so the generated shots stop assuming a person: the **shot catalogue** (an animal gets head / half-body / full-body / rear shots, an object gets front / angle / detail / rear views — the group headers relabel to match) and the **identity lock** the engine is given (a dog keeps its breed, coat and markings; a product keeps its shape, material and logo — instead of "same face, jawline, skin tone"). Each type ships its own balanced preset. **Human is the default and existing datasets are unchanged.** *(NSFW body shots stay Human-only.)* First-draft prompt sets — refine per subject as needed. Inspired by a community request.

  **Anime** is the one type where the *rendering* is part of the subject, so it behaves differently from the other five on purpose:

  - Its **identity lock** protects a character *design* rather than a photographed body — hair colour, hairstyle and silhouette, eye shape and iris colour, the **signature outfit and its colours**, the accessories (ribbon, hairpin, glasses, ears, tail) and the distinctive marks — **plus the art style itself** (line work, cel shading, palette). It then does what no other lock does: it **explicitly forbids** turning the character into a photograph, a 3D render or a real person (no skin texture, no pores, no film grain). Every other type ends its prompt with *"professional realistic photograph"*; for a drawing that instruction destroys the subject, so for Anime the engine is asked for an **anime illustration** instead — in the opening command and in the closing style tag alike.
  - The **signature outfit counts as identity**. For a Human dataset the app deliberately varies clothing on every shot (so a jacket never binds to the person); for a character the costume is half of what makes them recognisable, so the shots keep it. Two explicit *alternate outfit* cards let you vary it when you want to.
  - Its **shot catalogue** (55 shots) uses the vocabulary of the medium — **bust-up**, **cowboy shot** (knee-up), full body, a full **expression sheet** (smile, laughing, angry, surprised, blushing, eyes closed) and a **front / side / back character-sheet turnaround** on a plain background that no other type offers. Four presets: *Balanced*, *Face & expressions*, *Full body focused* and *Character sheet*.
  - It pairs naturally with the **Anima** training family (see *Default training family* above), though the two settings are independent — you can build an anime dataset and train it on any family.
- **Prompt suffixes** *(collapsible — optional creative direction)* — free text appended to **generated** variations at generation time, to steer a global look without rewriting anything:
  - **All shots** → `prompt_suffix` — one global suffix (e.g. *"shot on 35mm film, warm tones"*), up to **300 characters**.
  - **Face / Bust / Body / Back shots** → `prompt_suffixes` — one suffix per framing, up to **300 characters** each. A framing suffix applies to that shot type first, then the global one.

  Key behaviours: these are **applied at generation time and never stored into a tile's own prompt** (so a regenerate can't double-apply them), the **identity lock always comes first** — a suffix can't override it, clearing a field removes that suffix, and existing images stay as they are until you **regenerate** to apply.

  You can also edit the very same suffixes **inline in the generation panel** (the collapsible *✨ Prompt suffixes* row under the shot picker), which is handy for tuning them **per batch** without opening this modal — both surfaces read and write the one dataset value, and an edit made there is saved the moment you press **Generate**.

## Config-file-only settings

These have no UI control — they're for advanced users editing `config.json` by hand (copy `config.example.json` to `config.json` first). Most people never touch them; the defaults are tuned. Values below are the shipped defaults.

*(The four ComfyUI folder overrides used to live here. They are now editable in **Settings → Local tools → ComfyUI → Advanced: ComfyUI folder overrides** — see that section above. Values set by hand in `config.json` are unaffected: the same keys, read the same way, now simply shown in the app.)*

**Engines:**

| Key | Default | Role |
|---|---|---|
| `engines.chatgpt_subscription_model` | `gpt-5.4-mini` | The Codex **router** model used by the subscription lane — not the image model. The subscription lane renders on whatever image model your plan serves; the API-key lane's image model is `engines.chatgpt_image_model`. |

**Shot detection (video bank):** the boundary detector that cuts a long source into
individual shots. No UI control yet — the defaults are the reference
implementation's, and neither of the two numbers below has been measured against
this app's material, so they are stated as adjustable rather than tuned.

| Key | Default | Role |
|---|---|---|
| `shot_detect.python` | `''` | Interpreter that runs the detector. Empty means **reuse the Bank scoring environment**, which already carries torch — a second copy would cost you ~2.5 GB for nothing. Written by the installer; you rarely set it by hand. |
| `shot_detect.threshold` | `0.5` | Cut probability at or above which a frame is treated as a shot boundary. Lower it to cut more finely on soft transitions; raise it if dissolves are being split into fragments. |
| `shot_detect.min_shot_frames` | `5` | Shots shorter than this are **dropped, not merged** into a neighbour. Merging would silently move that neighbour's boundary, and a boundary is the one thing this whole lane exists to get right. 5 rejects a stray flash cut while leaving real rapid montages intact. |
| `shot_detect.device` | `auto` | `auto` \| `cuda` \| `cpu`. The network runs on 48×27 frames and is never the bottleneck — decoding is. CPU is a perfectly reasonable choice, and it leaves the GPU free for captioning and training. |

**Video bank quality cuts:** the thresholds behind the video bank's amber flags.
The cuts that describe your *footage* default to **empty = no cut** — that is a
decision, not an omission: published thresholds measurably do not transfer between
collections, so the app never ships one. `watermark_max` is the exception and the
reason is worth knowing: it does not measure your footage, it reads a *classifier's*
probability, which is calibrated with the model rather than with your material — so
the image lane's measurement transfers where a motion floor does not. Set them from
**Video bank → 🎚 Quality cuts**, where **Preview** shows how many shots each value
would flag before you apply it. Raw scores stay stored, so changing any of these
re-sorts every bank instantly, without rescanning.

| Key | Default | Role |
|---|---|---|
| `video_bank.min_duration_s` | *(empty)* | Flags shots shorter than this, in seconds (`brief`). The only cut here that needs no measuring pass — it reads the shot bounds, so it works straight after detection. Not the same thing as the promotion's `too short` refusal, which is your target profile's own arithmetic and no setting moves it; this one only decides what gets flagged for your eyes. |
| `video_bank.motion_floor` | *(empty)* | Flags shots whose average motion falls below this (`still`). |
| `video_bank.motion_ceiling` | *(empty)* | Flags shots whose busiest moments exceed this (`agitated`). |
| `video_bank.luma_floor` | *(empty)* | Flags shots whose darkest frame falls below this brightness (`black`). |
| `video_bank.freeze_max` | *(empty)* | Flags shots where more than this share of frames do not move (`freeze`). |
| `video_bank.sharpness_floor` | *(empty)* | Flags shots whose sharpest stretch stays below this (`soft`). |
| `video_bank.watermark_max` | `0.94` | Flags shots whose watermark score exceeds this (`watermark`), after the **🔖 Watermarks** pass has scored them. This model's scores are compressed hard against 1, so 0.94 is the measured cut and not the 0.5 a probability normally implies — on a 110-image hand-labelled sample it flagged none of the 55 clean images and still caught 54 of the 55 marked ones. Lower it toward 0.92 to catch the faintest marks and hand-check a few clean shots. A shot the pass has not judged carries no score and is **never** flagged — that is "not evaluated", not "clean". Set it to empty to flag nothing. |
| `video_bank.duplicate_threshold` | `0.96` | Cosine similarity at or above which two shots are grouped as near-duplicates by the **✂ Duplicates** pass, comparing them at their closest pair of embedded frames. Not a read-time cut like the rows above: changing it means re-running that pass — which is instant and costs no GPU, since it re-reads the frame vectors **🔎 Find scenes** already cached. **Where the number comes from:** it is inherited from the image bank's `bank.semantic_dup_threshold`, measured over the *same* CLIP space, and no video-pair calibration exists yet. Comparing shots at their closest frame pair also reaches any given value more easily than a single-image comparison does, so **raise** it if your bank over-groups. |

**Imported shot catalogs** — written by the workspace, not meant to be hand-edited (see *Using the app → Your own shot catalog*), but this is where they live so you know what to back up:

| Key | Default | Role |
|---|---|---|
| `custom_shots` | `{}` | `{subject_type: [{id, label, prompt, framing, nsfw?}]}` — the shots you imported from JSON, one list per subject type. Kept here rather than in the browser so they survive a cache wipe, follow you to another device and ride along in the backup. Entries are re-checked on read: a bad `framing`, a missing field, or a `label` that already belongs to a built-in shot is dropped (two shots sharing a label would resolve to the wrong prompt when one is regenerated). |

**Cloud (vast.ai) internals** — knobs for after the real-world smoke test; the UI-exposed cloud settings above are the ones you'll normally want:

| Key | Default | Role |
|---|---|---|
| `cloud.template_hash` | `471ed5903d8cdb8e63b0d0e50f6cd519` | The official vast.ai "Ostris AI Toolkit" template. Clearing it falls back to a raw-image launch. |
| `cloud.ui_port` | `18675` | Container port the pod UI is proxied on. |
| `cloud.image` | `vastai/ostris-ai-toolkit:…` | Raw-image fallback (used only when the template is cleared). |
| `cloud.offer_scan_limit` | `100` | How many offers are fetched when listing GPU speed tiers. |
| `cloud.pod_overhead_minutes` | `35` | Boot + model download + quantize time built into cost estimates. |
| `cloud.min_inet_down_mbps` | `400` | Skip hosts too slow to pull the image. |
| `cloud.min_disk_bw_mbps` | `500` | Skip hosts too slow to extract it. |
| `cloud.host_blacklist_days` | `3` | How long to skip a host whose pod showed no sign of booting. |
| `cloud.slow_boot_blacklist_hours` | `6` | Shorter skip for a host that was still visibly booting when the boot ceiling cut it — slow, not broken. |
| `cloud.ready_timeout_minutes` | `25` | **Idle** boot budget: the clock restarts every time the pod shows a boot fact it had never shown before (a new vast status, the UI port getting published, a moving host progress line), so an honest multi-gigabyte image pull is never cut. Only a pod that shows nothing new for this long is terminated. |
| `cloud.boot_budget_minutes` | `90` | **Absolute** ceiling on the boot phase. Because progress restarts the timeout above, a host too slow to ever finish would keep your rental alive; past this it is terminated regardless (`0` = no ceiling). |
| `cloud.disk_gb` | `60` | Instance disk (base model + dataset + checkpoints). |
| `cloud.min_vram_gb` | `{zimage:24, sdxl:16, krea:24, flux2klein:32}` | Minimum VRAM **per family**. flux2klein uses 32 (the 9B is the cloud-first lane; a 32 GB pod also trains the 4B). |
| `cloud.onstart` | `''` | Optional startup command for the raw-image fallback. |
| `cloud.full_transformer.storage_margin_gb` | `20` | Headroom added on top of *checkpoint × saves kept* in the Hugging Face storage pre-check. |
| `cloud.full_transformer.checkpoint_size_gb` | `0` | Size of one dense checkpoint used by that pre-check. `0` = measure it from what past dense runs really delivered, else ~26 GB. |

**Quality-tool interpreters and models:**

| Key | Default | Role |
|---|---|---|
| `face_scoring.python` | `''` | Interpreter for the InsightFace subprocess (empty = current interpreter). |
| `face_scoring.models_root` | `''` | Where InsightFace weights are stored/downloaded. |
| `face_scoring.device` | `'auto'` | Device for the Image-bank face pass. `auto` uses the GPU when the face interpreter exposes CUDA (needs `onnxruntime-gpu` installed in it) and falls back to CPU otherwise; `cpu` forces CPU (never touches the GPU); `cuda` requests the GPU but still falls back to CPU when unavailable. A GPU run is serialized through the GPU-exclusive window so it never competes with a training/scoring pass. |
| `masks.python` | `''` | Interpreter for the rembg (person-mask) subprocess. |
| `bank_scoring.text_search_idle_minutes` | `10` | How long the selected 🔤 **Find by text** encoder stays warm after its last query (capped at `120`). Set `0` to unload after every query. Text caches are separated by engine/model key. |
| `bank_semantic.python` | `''` | Interpreter for SigLIP 2 image/text workers and capability probes. Setup installs SigLIP 2 into `data/envs/bank_scoring` and records that managed Python here without changing `bank_scoring.python` (so a borrowed GPU Score runtime stays selected and untouched). You can also point it at a CUDA interpreter you already have, from the Bank's **Semantic engine** panel (**GPU Python I already have**) - an EXECUTION choice only: Setup's Install/repair keeps targeting the managed environment and now preserves your pick instead of overwriting it. The check uses SigLIP 2's own dependency list (`torch`, a `transformers` carrying `Siglip2Model`, `numpy`, `Pillow`) - no `open_clip`, no `timm`. Empty keeps backward compatibility: use `bank_scoring.python`, then the app Python. |
| `bank_semantic.models_root` | `''` | SigLIP 2 model cache (empty = `data/models/bank_semantic`). Setup downloads only the pinned files after an explicit click; inference is local-files-only. |
| `bank_semantic.device` | `'auto'` | Device for the SigLIP 2 image index (`auto`, `cuda`, `cpu`). Resolved against `bank_semantic.python`: `auto` only reaches the GPU if THAT interpreter's torch sees a card, which is what the Semantic engine panel's device line reports. A GPU run uses the same exclusive window as other Bank ML work. |
| `bank_semantic.siglip2_semantic_dup_threshold` | `0.97` | SigLIP 2 cosine threshold for **✂ Find crops & variants**. Separate from CLIP because the two spaces are not numerically interchangeable. |
| `watermark.python` | `''` | Interpreter for the LaMa watermark subprocess. **Auto-managed:** leave it empty and the **Install inpainting** button builds a dedicated Python 3.10-3.12 environment for you (`simple-lama-inpainting` needs Pillow&lt;10, so it can't share the app's own Python) and fills this in automatically. Set it yourself only to point at an environment you already have — a manual value is always respected and never overwritten. |

**Klein consistency LoRA:**

| Key | Default | Role |
|---|---|---|
| `klein.consistency_lora` | `klein/Flux2-Klein-9B-consistency-V2.safetensors` | The structure-anchoring LoRA on the Klein edit graph, relative to ComfyUI's LoRA folder. |
| `klein.consistency_strength` | `0.5` | Its strength (0–1). Its own guide warns 0.8–1.0 can stop edits applying; `0` disables it entirely. |

**Z-Image text encoder & VAE:**

Both are **blank by default, and blank is the right value** — the app finds them itself. It scans every registered `vae` / `text_encoders` folder (including the ones your `extra_model_paths.yaml` adds), sub-folders included, ignoring capitalisation and separators: `z_ae`, `z ae`, `z-ae`, and ComfyUI's own `ae.safetensors` all resolve, and `qwen_3_4b.safetensors` is found whether it sits at the root of `text_encoders/` or inside a folder called `Z image`, `Z Image` or `z-image`. It never picks a `.gguf` (the loader nodes cannot open one) and never picks Krea's `qwen3vl_4b` or Klein's `qwen_3_8b`, which live in the same folder and would fail at sample time.

Set one of these **only** to override that search — for instance if your ComfyUI is shared with FLUX.1, whose VAE is also called `ae.safetensors`, and the app picked the wrong one. A value you set here is used exactly as written and is never second-guessed; if the file isn't there, the error names *your* file rather than silently substituting another.

| Key | Default | Role |
|---|---|---|
| `zimage.vae` | `''` | Pin the Z-Image VAE, relative to ComfyUI's VAE folder (e.g. `z_ae.safetensors`). Blank = auto-resolve. |
| `zimage.text_encoder` | `''` | Pin the Z-Image text encoder, relative to ComfyUI's text-encoders folder (e.g. `Z image/qwen_3_4b.safetensors`). Blank = auto-resolve. |

**Updates:**

| Key | Default | Role |
|---|---|---|
| `updates.repo` | `perfectgf/lora-dataset-studio` | The GitHub repo the update checker reads its release feed from. |

**Run provenance:**

| Key | Default | Role |
|---|---|---|
| `provenance.archive_max_gb` | `5` | Ceiling of the **run image archive** (Settings → Maintenance): the deduplicated copies that let a two-run comparison still show an image you deleted afterwards. Copies are content-addressed, so a whole training history usually costs well under a gigabyte — on a real 20-dataset, 1471-image library, every distinct version of every image ever trained came to about **0.5 GB**. Past the ceiling nothing more is stored and the compare panel says so. Set it to `0` to turn archiving off entirely; the run records, settings and caption text are kept either way. |

## config.json key reference (all keys)

A flat cheat-sheet of the main `config.json` keys, for quick lookup or hand-editing (copy `config.example.json` to `config.json` first — it's git-ignored, in your data directory). Every key here is documented in full, with defaults and traps, in the sections above; this table is the index. **Secrets** (`GEMINI_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `HF_TOKEN`, `VAST_API_KEY`, optional scraper credentials) live in `.env`, not here.

| Key | Meaning |
|---|---|
| `server.host` | Interface the Flask server binds to (default `127.0.0.1`, local-only). |
| `server.port` | Port the server listens on (default `5050`). |
| `server.require_token` | On a non-loopback bind, require remote clients to present an access token (default `false` — a trusted LAN needs none). Toggle and token also live in Settings → Server & access. |
| `paths.dataset_images_root` | Where dataset images are stored. Empty string defaults to `<data dir>/datasets`. |
| `paths.cloud_runs_dir` | Working area of cloud training runs (dataset copy, samples, logs). Empty string defaults to `<data dir>/cloud_runs`. |
| `paths.checkpoints_dir` | Durable store for the checkpoints cloud runs produce. Empty string defaults to `<data dir>/checkpoints`. No cleanup ever removes a file from it. |
| `paths.video_datasets_dir` | Where promoted video datasets are written — a flat folder of `.mp4` clips with homonym `.txt` captions per dataset. Empty string defaults to `<data dir>/video_datasets`. |
| `dataset_import.max_side` | Longest side for opt-in WebP normalization (default `1024`; `0` = original size). It is ignored by the default `preserve` mode; ratio is always preserved, never enlarged, and normalized paths clamp at 8192 px. Every source must still fit `image_input.*` below; a larger one is rejected and must be resized before import (or the budget raised). Not retroactive. Editable in Settings → Captioning & quality. |
| `dataset_import.encoding` | How an un-cropped imported image is written: `preserve` (default; original JPG/JPEG, PNG, WebP or BMP bytes with the matching extension), or the opt-in WebP modes `standard` (q92), `high` (q100), and `lossless`. Auto head-crop is always a derived WebP. The `image_input.*` budget applies to every mode. Editable in Settings → Captioning & quality. |
| `image_input.max_pixels` | Largest source image any lane may decode, in pixels (default `67108864` = 64 Mi-pixels; `0` = no limit). A memory budget: ~3 bytes per decoded RGB pixel, and an edit or analysis pass can hold a second copy. Read by dataset import, ZIP/scrape ingest, Bank scan and thumbnails, edits, ComfyUI staging and Ollama vision. Bank *inference workers* keep their own fixed 16 Mi-pixel guard. Editable in Settings → Captioning & quality. |
| `image_input.max_side` | Largest side of a source image, in px (default `16384`; `0` = no limit). Separate from `max_pixels` because a wide panorama can sit inside the pixel budget and still exceed a side limit. Editable in Settings → Captioning & quality. |
| `comfyui.api_url` | Base URL of your ComfyUI instance (default `http://127.0.0.1:8188`). |
| `comfyui.base_dir` | ComfyUI install directory, used to derive `output`/`input`/`models`/`loras` dirs if those aren't set explicitly. |
| `comfyui.output_dir` | Explicit override for ComfyUI's output folder. Set it when ComfyUI runs with `--output-directory`. Editable in Settings → Local tools. |
| `comfyui.input_dir` | Explicit override for ComfyUI's input folder. Set it when ComfyUI runs with `--input-directory`. Editable in Settings → Local tools. |
| `comfyui.models_dir` | Explicit override for ComfyUI's models folder (used to scan available checkpoints/UNETs). `extra_model_paths.yaml` is still read on top of it. Editable in Settings → Local tools. |
| `comfyui.object_info_timeout_s` | Seconds ComfyUI may take to enumerate its nodes and model files (default `45`, clamped to 5-300). Raise it on an install with many custom nodes; a ComfyUI that is simply stopped is still detected in ~3 s regardless. Editable in Settings → Local tools. |
| `comfyui.loras_dir` | Explicit override for ComfyUI's LoRA folder — where trained LoRAs are installed. Wins over `extra_model_paths.yaml`; left empty, the deploy folder follows the yaml's `is_default` LoRA root, else `<install>/models/loras`. Editable in Settings → Local tools. |
| `ollama.url` | Base URL of your Ollama instance (default `http://127.0.0.1:11434`). |
| `ollama.deployment_mode` | Docker-only deployment selected in LDS Setup: `none`, `host`, or `docker`. Host and companion modes force `http://host.docker.internal:11434` and `http://ollama:11434` respectively; their URL is not user-editable. |
| `ollama.vision_model` | Ollama vision model used for auto-classify and auto head-crop (default `huihui_ai/qwen3-vl-abliterated:8b-instruct`, the uncensored **abliterated** build — use the Instruct, not Thinking, variant). |
| `ollama.vision_concurrency` | How many images a bank vision pass (watermark / framing / captions) sends to Ollama at once (default `4`, clamped to 1-16). Higher overlaps more waiting; `1` restores the old one-at-a-time behaviour. |
| `ollama.vision_keep_warm_seconds` | How long a one-off vision job (auto head-crop, Describe) may leave the model loaded when nothing else wants the GPU (default `120`, `0` = always unload, capped at 600). The lease is revoked as soon as a generation or a training starts. |
| `aitoolkit.dir` | ai-toolkit install directory. |
| `aitoolkit.datasets_dir` | Override for ai-toolkit's datasets folder (defaults to `<aitoolkit.dir>/datasets`). |
| `aitoolkit.output_dir` | Override for ai-toolkit's output folder (defaults to `<aitoolkit.dir>/output`). |
| `aitoolkit.hf_home` | Override for the Hugging Face cache directory ai-toolkit uses. |
| `aitoolkit.python` | Full path to the Python interpreter to run ai-toolkit with. Empty = auto-detect a `venv/`/`.venv/` next to `run.py`; set it for conda/uv/system-Python and portable/embedded (`python_embeded`) installs that have no venv folder. |
| `engines.default` | Default image-generation engine selected in the UI (`nanobanana`, `chatgpt`, `openrouter`, or `klein`). |
| `engines.enabled` | List of engines shown as options in the UI. Doubles as the engine catalogue: an engine added by an update is merged into a stored list on read, so a new engine reaches installs that already have saved settings. An engine you removed yourself is never added back. |
| `engines.known` | Not a setting — the ledger of which engines the app was offering the last time this list was saved. It is what tells "this engine did not exist yet" apart from "I unticked it". Written automatically; `[]` (or absent) means the app assumes the pre-OpenRouter trio. Delete it to be re-offered every engine. |
| `engines.chatgpt_auth` | Which credential the ChatGPT engine uses: `auto` (subscription when connected, else API key), `api`, or `subscription`. |
| `engines.openrouter_model` | Image model slug the OpenRouter engine requests. Free text; blank = `google/gemini-3-pro-image`. Must accept reference images. |
| `engines.nanobanana_model` | Image model the Nano Banana engine requests. Free text; blank = the `NANOBANANA_MODEL` environment variable if set, else `gemini-3-pro-image`. Must accept reference images. |
| `engines.chatgpt_image_model` | Image model the ChatGPT engine requests on the **API-key** lane. Free text; blank = the `CHATGPT_IMAGE_MODEL` environment variable if set, else `gpt-image-2` (the only model that needs no OpenAI organization verification). Must accept reference images. The subscription lane ignores it. |
| `engines.chatgpt_subscription_model` | Codex **router** model for the subscription lane (default `gpt-5.4-mini`) — not an image model. |
| `captioning.backend` | Caption backend: `auto` (prefer JoyCaption, fall back to Ollama), `joycaption`, `ollama`, or `none`. |
| `training.default_family` | Default model family preselected for new training runs (`zimage`, `sdxl`, `krea`, `flux`, `flux2klein`, or `anima`). |
| `cloud.max_concurrent_runs` | Simultaneous cloud pods allowed (default `1`, 1–10). Also in Settings → Storage. |
| `cloud.max_price_per_hour` | Safety cap on the hourly offer price in $ (default `0.80`); pricier hosts are skipped before launch. |
| `cloud.monthly_budget_usd` | Hard monthly spend ceiling in $ (default `0` = unlimited); launches are blocked past it. |
| `cloud.stall_timeout_minutes` | Kill + rescue a cloud run after this many minutes without step progress (default `30`, 5–240). |
| `cloud.first_step_timeout_minutes` | Kill a run that reaches no training step **and** reports no new downloaded bytes for this long (default `45`, 5–240). Also in Settings → Storage. |
| `cloud.first_step_download_budget_minutes` | Absolute ceiling on the pre-training base-model download, even while it is progressing (default `180`; `0` = no ceiling). Also in Settings → Storage. |
| `cloud.max_runtime_minutes` | Hard stop on the whole run (default `480`, 30–1440); the newest checkpoint is rescued first. Enforced by the out-of-run supervisor too. Also in Settings → Storage. |
| `cloud.freeze_watchdog_minutes` | Terminate a training run whose **pod** shows no progress for this long (step, download bytes or a new checkpoint), from outside the run's own supervision; the clock is durable and survives an app restart (default `45`; `0` = warn on the card only). |
| `cloud.upload_stall_minutes` | Give up a run whose dataset upload has had **no byte at all** reach the pod for this long, and release the machine (default `25`; `0` = never cut). Not a ceiling on the transfer's duration — a slow upload that keeps moving is never cut. Also in Settings → Storage. |
| `cloud.min_reliability` | vast.ai host-reliability floor (default `0.98`, 0.9–0.999); lower surfaces cheaper, riskier hosts. |
| `cloud.verified_only` | Restrict to vast.ai verified hosts (default `true`). |
| `cloud.secure_cloud_only` | Restrict to vast.ai's Secure Cloud (datacenter) tier (default `false`; narrows the market, raises price). |
| `cloud.full_transformer.delivery` | Where a finished full model is delivered: `both` (default — this computer first, Hugging Face backup after), `local`, or `hub`. Also in Settings → Storage. |
| `cloud.full_transformer.local_disk_margin_gb` | Free space left on the checkpoint volume on top of the delivery itself, checked before a pod is rented (default `15`). |
| `cloud.full_transformer.hub_push_budget_seconds` | Ceiling on the pod-side upload of the master to Hugging Face (default `3600`). |
| `cloud.full_transformer.hub_fetch_budget_seconds` | Ceiling on the pod-side download of a checkpoint when continuing a full model (default `3600`). |
| `cloud.full_transformer.push_slice_bytes` | Size of one slice when a full model is pushed to a pod **from this computer** (default `2147483648`, i.e. 2 GiB). It is not a memory setting — the request is streamed, so a slice of any size costs a megabyte of RAM. It is what an interruption COSTS: a broken transfer resumes at the last whole slice, so a smaller value loses less on a flaky link and a larger one makes fewer round-trips. |
| `cloud.uplink_mbps` | Your upload speed in Mbit/s, used to forecast how long sending a checkpoint to a pod would take and what that costs in rented GPU time (default `0` = work it out). The app **measures** the real speed of every checkpoint it pushes and prefers what it measured over what you typed, so this only fills the gap until you have sent one. Dataset uploads do not count towards that measurement: thousands of small files measure per-request latency, not the throughput one continuous 26 GB file would see, and averaging the two would describe neither. |
| `cloud.full_transformer.private_storage_limit_gb` | Private Hugging Face allowance the dense pre-check compares against (default `0` = infer from the documented plan, an estimate). Also in Settings → Storage. |
| `cloud.full_transformer.storage_margin_gb` | Headroom added to that forecast (default `20`). |
| `cloud.full_transformer.checkpoint_size_gb` | Dense checkpoint size used by that forecast (default `0` = measured from past runs, else ~26 GB). |
| `cloud.full_transformer.fp8_export` | Produce the ~10 GB ComfyUI-loadable fp8 twin at the end of a successful dense run (default `true`). `false` disables it for every dataset — an install that would rather not spend the extra pod minutes. |
| `cloud.quantize.max_minutes` | Hard ceiling on a cloud quantization rental (default `60`, floor 5). The machine is destroyed when it is reached, whatever it reported. |
| `cloud.quantize.max_price_per_hour` | Price cap for that rental (default: the general `cloud.max_price_per_hour`). |
| `cloud.quantize.min_inet_down_mbps` | Downlink floor for the host (default `200`). The job is network-bound — this is the setting that decides the bill. |
| *(not a setting)* free-disk floor | Derived, never configurable: the pod must hold the master, its fp8 twin and the download cache (~86 GB for a 26 GB model). Offers with less free disk are excluded from the search, because vast refuses an ask larger than the machine's disk — and the cheapest listing is exactly where free disk runs out. |
| `cloud.quantize.export_budget_seconds` | Time budget for the conversion inside the pod (default `1800`). |
| `cloud.full_transformer.fp8_export_budget_seconds` | Time budget for that conversion on the pod (default `1800`). Exceeding it abandons the export; the bf16 master is already delivered, so the run stays a success. |
| `face_scoring.python` | Python interpreter used to run the InsightFace subprocess (empty = current interpreter). |
| `face_scoring.models_root` | Directory where InsightFace model weights are stored/downloaded. |
| `face_scoring.green` | Similarity score threshold (0–1) above which an image is flagged "green" (strong match). |
| `face_scoring.orange` | Similarity score threshold (0–1) above which an image is flagged "orange" (borderline match). |
| `masks.python` | Python interpreter used to run the rembg subprocess (empty = current interpreter). |
| `video_caption.style` | Which PROMPT writes the captions: `standard` (default, the shipped wording) or `plain`. Measured to matter **more than the checkpoint**: asked the standard way, even an uncensored model describes *around* explicit footage, while the base model asked plainly named things precisely and wrote the best action description of the four combinations tried. `plain` adds explicit permission to state what is visible and what occurs, and forbids the two evasive words the test caught models hiding behind. It matters because a caption that talks around its subject teaches the trained model to look away, and the captions read perfectly well either way. Anything unknown falls back to `standard` — never to `plain`. Also pickable per run, next to the **🗣 Describe shots** button; every caption records the style that produced it. |
| `video_caption.model` | Which model writes the 🗣 **Describe shots** captions (empty = the shipped default, `Qwen/Qwen3-VL-4B-Instruct`). Any checkpoint of the **same architecture** is a drop-in; a different architecture fails loudly at load rather than silently misbehaving. Worth changing when the default **talks around** what your footage shows — a caption that names things evasively teaches the trained model to do the same, and nothing in the output reveals it. Pointing this at a model the machine does not have is allowed and downloads it on the first run, but never in silence: the pass says so in its own progress line before captioning anything. Every caption records which model wrote it, so a bank captioned across a change stays readable. |
| `quantize.python` | Python interpreter that runs the **fp8 conversion** and the **LoRA→base merge** (empty = the one ✨ Score uses, then ai-toolkit's, then the app's own). Both need `torch`, which this app deliberately does **not** install — it is gigabytes and nothing else here needs it — so they run in a subprocess, like the scoring and masking passes. One setting governs both on purpose: "the Python on this machine that has torch" is one fact, and saying it twice is how the two drift apart. The chosen interpreter is probed while the *plan* is drawn: one that lacks the packages disables the button with the reason and the `pip install` line, instead of failing after the click (or after a 26 GB download). `torch` is the only module either of them needs: both read and write the safetensors format themselves rather than memory-mapping it, so an environment with torch alone is enough. |
| `bank_scoring.python` | Python interpreter that runs the ✨ Score pass (empty = the app's own). Auto-filled by Setup with a CPU-only environment; repointable at any CUDA interpreter already on the machine via the bank's **⚡ Use a GPU Python I already have** picker, which verifies every dependency first and never installs into an environment it did not create. |
| `bank_semantic.python` | Python interpreter that runs SigLIP 2. New installs record the LDS-managed Bank environment here independently of Score; repointable at any CUDA interpreter already on the machine from the Bank's **Semantic engine** panel, verified against SigLIP 2's own (shorter) dependency list and never installed into. Empty falls back to `bank_scoring.python` for older configs, then the app's own interpreter. |
| `watermark.python` | Python interpreter used to run the LaMa watermark-inpainting subprocess (empty = reuse `masks.python`, then the current interpreter). |
| `watermark.device` | LaMa processing device: `auto` (CUDA when available, otherwise CPU), `cuda`, or `cpu`. |
| `watermark.allow_crop` | When `true` (default), a border watermark is cropped off; when `false`, it is repainted instead. Also editable in the Clean bar. |
| `zimage.vae` | Pins the Z-Image VAE (blank = the app resolves it itself, any spelling, any sub-folder). |
| `zimage.text_encoder` | Pins the Z-Image text encoder (blank = the app resolves it itself). |
| `klein.consistency_lora` | Filename of the Klein consistency LoRA, relative to ComfyUI's LoRA folder. |
| `klein.consistency_strength` | Strength (0–1) applied to the Klein consistency LoRA. |
| `klein.generation_steps` | Sampler steps for Klein **generation** (variations, regenerate, small-image rescue). Default `5` = the value hardcoded in the shipped workflow; 1–50. Not the improve pass (`klein.improve_steps`). |
| `klein.edit_base_lora_strength` | Strength of the enhancement LoRA (`klein/realistic.safetensors`, node 139) on Klein **edits** — reference edit, variations, regenerate, small-image rescue. Default `0` = off, the render before that LoRA became a Setup download; 0–2. Not the improve pass (`klein.improve_base_lora_strength`). |
| `klein.generation_lora_presets` | Named generation-LoRA stacks (default empty) picked per run in Klein tuning; each has a name and up to 8 `{file, strength}` rows. Managed in Settings → Image engines. |
| `klein.default_generation_lora_preset` | Which of `klein.generation_lora_presets` the 🖥️ Klein tuning panel STARTS on. Default `''` = *None*, the behaviour before this key existed. A starting point only — the run panel still offers None and every other preset for that run, and picking there does not rewrite this. Fail-closed: a name matching no preset behaves as *None*. |
| `krea.default_generation_lora_preset` | The same, for `krea.generation_lora_presets` and the 🧬 Krea 2 Edit tuning panel. A SEPARATE key on purpose: the two preset lists are independent and one name can designate two different chains. Default `''`. |
| `identity_prompts.markings_lock` | Krea's “hold the skin” order — forbids inventing or redrawing marks. Blank = shipped default. Naming a body feature here summons it. |
| `identity_prompts.outfit_vary` | The outfit directive injected into every human shot with no named garment. Blank = shipped default. |
| `identity_prompts.expression_neutral` | The neutral-expression directive injected into every human shot with no named expression. Blank = shipped default. |
| `identity_prompts.outfit_palette` | Krea's concrete garments, **one per line**. Blank (or nothing but blank lines) = the shipped list. The list's LENGTH decides which shot gets which garment. |
| `identity_prompts.render_tail_sfw` / `.render_tail_nsfw` | The Klein/Krea rendering tail, SFW and uncensored. Per subject type (`by_subject.<type>.<kind>` for non-human). Blank = shipped default. |
| `identity_prompts.framing_face` / `.framing_bust` / `.framing_body` / `.framing_back` | The per-framing shot-detail block for Klein/Krea. Per subject type. Blank = shipped default. |
| `identity_prompts.by_subject.<type>.<kind>` | Identity-lock overrides for a **non-human** subject type (`animal`, `creature`, `object`, `other`) × kind (`face_single`, `face_multi`, `klein_identity`). Human overrides stay on the flat `identity_prompts.<kind>` keys. Blank/absent = the shipped default for that subject. |
| `klein.small_image_prompt` | Optional shared instruction for scraper rescue and single/bulk image improvement (empty = reference image only). |
| `updates.repo` | GitHub repo the update checker reads its release feed from (default `perfectgf/lora-dataset-studio`). |

Additional config-file-only keys (ComfyUI folder overrides, cloud internals, quality-tool interpreters, Klein consistency LoRA) are documented in [Config-file-only settings](#config-file-only-settings) above.

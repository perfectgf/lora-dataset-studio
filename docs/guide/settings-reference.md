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

Settings:

- **Reference grounding** → `krea.grounding_px`. Range `512`–`1536`, default **`512`**. **The** dial of this engine: the resolution your reference is shown to the model's vision encoder at. At the low end it follows the shot description (more variety in pose, outfit and scene, looser likeness); **higher** values favor reference likeness and can copy the very pose and outfit you asked it to change. **512 is the dataset-restaging balance**: it keeps the prompt and selected catalog card in charge while preserving identity. Raise it deliberately when keeping the reference more closely matters than changing its pose. This is Krea-only: it does not change ChatGPT, Gemini/Nano Banana, OpenRouter, or Klein.
- **Sampler steps** → `krea.steps`. Default **`10`**, the value the model's own reference workflow uses. More is slower and rarely better on this pipeline.
- **Base model file** → `krea.base_model`. **This is the GENERATION setting only** — the checkpoint ComfyUI loads for Krea 2 Identity Edit. It has **nothing to do with LoRA training**, which never reads it: training pulls its base from Hugging Face and picks it from the **Krea 2 training base** dropdown in the training panel (**Raw**, the default and the official recommendation — you train on Raw and apply the LoRA on Turbo at inference). Nobody can accidentally train on Turbo by leaving this field alone. *(The naming confusion was raised by strouder, GitHub #19.)* Blank (default) = the app picks a Krea 2 **Turbo** then **Raw** build from your ComfyUI. Set it only if you own several. Checkpoints that merely carry "krea" in their name but are not Krea 2 bases are **skipped on purpose** — the identity LoRA renders pure noise on them, which looks like a broken app rather than a wrong file.
- **Identity edit LoRA** → `krea.identity_lora`. Path relative to `models/loras`; if nothing is there under that name the app searches your LoRA folders for a `krea2_identity_edit` file, so a renamed download still works.
- **Krea 2 Edit generation LoRA presets** → `krea.generation_lora_presets`. Named,
  ordered combinations of **your own** LoRA files, chained after the identity-edit
  LoRA when Krea 2 Edit generates dataset images. Max 8 LoRAs per preset, 12
  presets; inside a preset the row order **is** the chain order. Per run you pick
  one preset (or None, the default on every visit) in the workspace's 🧬 Krea 2
  Edit tuning panel — the run sends only the preset's NAME and the app resolves
  the files from this list, so renaming a preset can never make a run load
  something you didn't configure. Strength runs to 6, and to **20** for utility
  LoRAs whose filename says `filter-bypass`: those have no measurable effect below
  about 10. Limits worth knowing: the LoRA must be trained for **Krea 2** (another
  architecture loads as a silent no-op — the picker badges it), only the **model**
  side is patched so a LoRA's text-encoder weights are ignored, a row whose file
  has moved is **skipped** with the rest of the chain still applied, and the 🔄
  single-image regenerate in the workspace does **not** carry a preset. Empty by
  default. *(Preset mechanism by @waltm, Discord.)*

The pipeline's reference boost is an internal Krea calibration, not a second user-facing likeness slider; use **Reference grounding** for that trade-off.

Two behaviours worth knowing before you build a dataset with it:

- **The selected card's framing is honored.** Krea Fit v1.2 uses the selected catalog card's framing and aspect ratio (including its 1:1 / 3:4 shape) instead of copying the source photo's shape.
- **Extra reference images are ignored.** Identity comes from the primary reference alone. Klein and the API engines still use your extra refs.

Outfits and expressions are steered differently here than on the other engines: this model preserves anything it is not *positively* told to change, so the catalog's "a different outfit (not the one in the reference)" phrasing is rewritten at generation time into a concrete garment ("wearing a red knit sweater"), picked from the shot's own name — so outfits genuinely differ across the dataset while regenerating one shot reproduces its own.

### Klein generation LoRA presets (optional)

*Idea from @waltm on Discord.* Named combinations of generation LoRAs that stack on top of the local Klein edit graph. Stored in `klein.generation_lora_presets` (default: empty — no presets).

Each preset has a **name** and an **ordered list of LoRAs**, and each LoRA row has:

- a **file** — a name relative to your ComfyUI `models/loras` folder (e.g. `klein/my-lora.safetensors`), exactly like the consistency LoRA. The field is a **searchable dropdown of the LoRAs actually on disk** (every folder, `extra_model_paths.yaml` included), with Klein-compatible files listed first and each one badged by architecture; free text still works for a file you haven't downloaded yet;
- a **strength** — `0`–`1.5`, default **`0.6`**.

Use **＋ New preset**, **Duplicate**, **Delete** and rename to manage them, and the up/down controls to set chain order. **Caps: 8 LoRAs per preset, 12 presets.**

How presets are used matters:

- A preset is **chosen per run** in the **🖥️ Klein tuning** panel of the workspace, and it **defaults to *None* every visit** — presets never apply on their own.
- Resolution happens **by name** on the server, and it's **fail-closed**: if a run references a preset name that no longer exists, it runs **with no extra LoRAs** rather than erroring.
- **Trap:** *renaming* a preset does **not** follow a run that referenced it by the old name — that run silently falls back to no extra LoRAs. Rename before you queue, or re-pick the preset on the run.
- There is deliberately **no automatic NSFW gating** on individual LoRAs — the preset you pick carries the intent. If you want an "NSFW full" stack, make it a preset.

### Klein generation quality

*Raised by ashish.sinha.* **Generation steps** → `klein.generation_steps` (1–50, default **5**). How many sampler steps the local **Klein** engine spends on each generated image — variations, regenerations and the automatic small-image rescue. The shipped workflow had this pinned at **5** with no way to change it; the default is that same 5, so nothing moves until you raise it. More steps usually render more cleanly and cost proportionally more time (10 steps ≈ twice the wait per image).

It is a **rendering** knob, not an anatomy fix: extra limbs, tails or wrong body parts come from the identity prompt describing the wrong kind of subject (see the subject-type note below), and no number of steps repairs that.

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

**Shortcut from the workspace.** The multi-reference instructions are also reachable from **Add images ▸ Extra refs ▸ ✎**, without opening Settings. That modal shows **both** of them — `face_multi` (Nano Banana / ChatGPT) and `klein_identity` (Klein) — because the engines do not share one text, and badges **every** prompt your ticked engines actually use (a Klein + ChatGPT run really uses both). It edits the prompts of the **open dataset's subject type**, and says which one in its title and intro; edits made there are the same settings as here, for that subject.

- **API engine — identity lock (single reference)** → `identity_prompts.face_single`. Prepended to Nano Banana / ChatGPT variations built from **one** reference photo.
- **API engine — identity lock (multiple references)** → `identity_prompts.face_multi`. The same, for variations built from **several** reference photos — it tells the model every reference is the same person and to use them together.
- **Klein — restage & face-identity block** → `identity_prompts.klein_identity`. The instruction block the local **Klein** engine uses to restage the shot (pose, framing, outfit, expression) while keeping the face identical.
- **Klein upscale & improve prompt** → `identity_prompts.klein_improve`, with an on/off toggle `identity_prompts.klein_improve_enabled` (default **on**). The fixed instruction the manual **Klein upscale & improve** action sends to add texture and detail. The shipped text is `add detailed texture, add sharp details, add candid shot, add soft focus effect` — read it before you blame the model: those four clauses describe a **photograph**, and they are applied to every dataset, drawn ones included. **Turn the toggle off** to run that action with **no prompt at all** — a pure upscale with no added styling. Both the text in force and these two levers are now quoted and linked from the ✨ **Upscale & improve** button itself (lightbox and bulk toolbar), so you no longer have to know this setting exists to find it.
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
auto head-crop can decode the source, it must be no larger than **16 Mi-pixels**
and **8192 px per side**. A larger file is rejected; convert or resize it before
importing. WebP normalization does not bypass this admission limit.
- **Stored resolution** → `dataset_import.max_side`. Used only by the three WebP
normalization modes. Choose `1024`, `1536`, `2048`, `4096`, or `0` = original
size. The aspect ratio is always preserved (no square padding) and an image is
never enlarged. This output setting takes effect only after the source passes
the import safety limit above; normalized output also clamps the longest side to
**8192 px**.

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
- **Same-style similarity** → `bank.style_threshold`. Cosine similarity on the CLIP image embeddings at or above which two images share a visual **🎨 style** (screenshots/memes cluster apart from photoreal) in the **✨ Score** pass. Default **`0.6`**. *Applies at the next scoring pass* (embeddings are cached, so re-clustering at another threshold is fast).
- **Semantic duplicate similarity** → `bank.semantic_dup_threshold`. Cosine similarity on the *same* CLIP embeddings at or above which two scored images are grouped as a **✂ semantic near-duplicate** — a crop or re-compressed variant of the *same shot* that the perceptual-hash **≈ Duplicates** (stage 1) misses. Default **`0.96`** (much higher than the style threshold: a crop is far closer than merely "same style"). Needs the **✨ Score** pass first (it reuses those embeddings — no extra GPU work). *Re-running at another threshold re-sorts instantly* from the cached embeddings, no re-scan.

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

The **✨ Score** pass (aesthetic · NSFW · style) needs the **Bank scoring** extra (Setup ▸ Quality tools); **🚩 Find watermarks** reuses the vision model from **Captioning**. Both are GPU passes, serialized against training and captioning, and detection-only — the bank never edits your source files.

## Training

Defaults for new runs, plus everything about the optional cloud training lane.

### Defaults

- **Default training family** → `training.default_family`. The model family preselected when you start a new run. One of `zimage`, `sdxl`, `krea`, `flux`, `flux2klein`, `anima`. Default **`zimage`**. Purely a starting point — you can switch family per run. `anima` trains the open [Anima](https://huggingface.co/circlestone-labs/Anima-Base-v1.0-Diffusers) anime model on its public base (no gated download); it is **local-only** for now (needs an up-to-date ai-toolkit + diffusers — cloud training arrives once the GPU pod image is verified).

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

| Setting | Key | Default | Range | What it does |
|---|---|---|---|---|
| **Max simultaneous cloud runs** | `cloud.max_concurrent_runs` | `1` | 1–10 | How many cloud pods may train at once. |
| **Max price per hour ($)** | `cloud.max_price_per_hour` | `0.80` | 0.1–5 | A safety cap on the hourly offer price; pricier hosts are skipped before launch. |
| **Monthly budget ($, 0 = unlimited)** | `cloud.monthly_budget_usd` | `0` | ≥0 | A hard spend ceiling for the month; new launches are **blocked** once you pass it. `0` means no limit. |
| **Stall timeout (minutes)** | `cloud.stall_timeout_minutes` | `30` | 5–240 | If no training step progresses for this long, the watchdog rescues the logs and kills the pod. |
| **First-step timeout (minutes)** | `cloud.first_step_timeout_minutes` | `45` | 5–240 | Idle budget for the phase *before* training starts, while the pod downloads its base model (tens of GB). The clock **restarts every time the pod reports more downloaded bytes**, so an honestly slow download is never cut — only a pod that reports no bytes and no step for this long is terminated. A frozen byte counter counts as silence even when the log line keeps redrawing. |
| **Base-model download ceiling (minutes, 0 = none)** | `cloud.first_step_download_budget_minutes` | `180` | 0 or 5–480 | The **absolute** ceiling on that same phase. Because advancing bytes restart the timeout above, a host too slow to ever finish would keep its download — and your rental — alive until the runtime cap; past this it is terminated regardless. `0` removes the ceiling and leaves the runtime cap as the only backstop. |
| **Max runtime (minutes)** | `cloud.max_runtime_minutes` | `480` | 30–1440 | Hard stop on the whole run, whatever it is doing: the newest checkpoint is rescued, then the pod is terminated. Enforced by the out-of-run supervisor too, so it holds even if the run's own supervision dies. |
| **Freeze watchdog (minutes, 0 = warn only)** | `cloud.freeze_watchdog_minutes` | `45` | 0 or 15–480 | Last-resort net for a run whose own supervision stopped answering (an app restart, a wedged connection to the pod): if a **training** run reports no progress at all for this long, the pod is terminated from outside — the stall timeout above can only act while the run is still being watched. Checkpoints already downloaded are kept. Set to `0` to only see the warning on the run card and never cut automatically. **"No progress" means the pod, not the app.** The clock is kept in the database and advances only when something actually moves on the host (the training step, the download's byte counter, a checkpoint landing), so restarting the app no longer sets it back to zero, and a run whose supervision keeps repeating the same sentence is no longer counted as alive. Phases that are silent by design (booting, uploading, downloading the result) are never judged on this value; they get a fixed 2 h floor and the runtime cap as backstop. |
| **Unreachable grace (minutes)** | `cloud.unreachable_grace_minutes` | `6` | 1–60 | How long a running pod may stay unreachable (a vast.ai network blackout) before the run is given up and auto-retried on a fresh host. Raise it if healthy runs die with *pod unreachable*. |
| **Min host reliability** | `cloud.min_reliability` | `0.98` | 0.9–0.999 | vast.ai reliability floor. Lowering toward 0.95 surfaces cheaper hosts at a higher boot-failure risk. |
| **Verified hosts only** | `cloud.verified_only` | **on** | toggle | Restrict to vast.ai's verified hosts (the historical, safer behaviour). |
| **Secure Cloud only** | `cloud.secure_cloud_only` | **off** | toggle | Restrict to vast.ai's *datacenter* (Secure Cloud) tier — usually narrows the market and raises the price, so it's opt-in. |

### Advanced options (per run)

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

## Server & access

How the app binds and who can reach it. **These are the settings that need a restart** — the card shows a **Running vs Saved** banner and a **Save & restart to apply** button that does it in one click.

- **Port** → `server.port`. The port the app listens on. Default **`5050`**. Change it if something else owns the port (on macOS, port 5000 is taken by AirPlay Receiver).
- **Available on the local network** — a toggle that flips the bind host between `127.0.0.1` (this machine only, the default) and `0.0.0.0` (reachable from your LAN — phone, tablet, another PC). The token and phone controls below only appear once this is on.
- **Require an access token** → `server.require_token`. Default **off** — a home LAN is treated as trusted, so LAN access is open and there's no token to type on a phone. Turn it **on** to demand a token from remote devices; requests from localhost never need one.
- **Access token** → `server.access_token`. Shown only when the token gate is on: a read-only field with **Generate new token** and **Copy**. It's persisted, so it survives restarts. Open `http://<machine>:<port>/?token=<token>` once from the remote device and a signed session cookie takes over.
- **Open it on your phone** — a card with a scannable **QR code** and copyable URLs built from this machine's real LAN IP (and Tailscale IP, if present). No guessing which address to type.

**Trap:** if you launched via `start.bat` with `LDS_PORT` set, that variable can override the port in your config. The in-app **Save & restart** pins host and port for the relaunch, precisely so the restart lands on the port you chose rather than the one the script forced.

## Maintenance

Housekeeping and diagnostics. Only one true setting lives here; the rest are actions.

- **Updates** — **Check for updates** and **Update & restart**, plus a *see what's in this update* compare link. **The button adapts to how you installed.** A **git checkout** fast-forwards to the latest commits. A **packaged (ZIP) install** announces the release and its size (*Update to vX — download ~XX MB*) and shows a **live progress bar** while it downloads and installs (a release ZIP is far larger than a git pull), then backs up the current files and swaps in the new ones — keeping `data/`, `config.json`, `.env` and your `.venv` untouched — and restarts. A mid-way failure rolls back automatically, so a broken download never leaves you with a half-updated install. If the app can't identify a downloadable release (no ZIP asset, or offline), the button steps aside and links to the releases page instead of promising an update it can't perform.
- **Trash** — **Open folder** and **Empty trash**. Everything the app deletes goes here first; emptying is the one destructive action, and it asks for confirmation.
- **Run image archive** — its size, its ceiling, and **Clear archive**. When a training run is launched, a **deduplicated** copy of the images it trains on is kept so that comparing two runs can still *show* an image you have since deleted from its dataset. Copies are **content-addressed**: relaunching an unchanged dataset stores nothing the second time, and only images that were added or re-edited cost anything. Clearing it keeps your runs, their settings and their caption text — you only lose the ability to look at images that are no longer in their dataset. The ceiling is `provenance.archive_max_gb` (see *Config-file-only settings*); past it, nothing more is stored and the compare panel says the picture is unavailable instead of showing a wrong one.
- **Back up everything** — not on this page but on the **Datasets library**: one button archives every dataset, its **training history** and your settings into a single file (⬇ download or 📂 open folder), and the library's **Import backup** restores it — datasets come back under **Trained**, not "Not trained yet". Tick **Include trained LoRAs** to bundle the (large) trained `.safetensors` too. **API keys and tokens are never included** — re-enter them on the new install. See *Using the app → Back up everything*.
- **Dataset images root** → `paths.dataset_images_root`. Where dataset images are stored. Default **empty → `<data dir>/datasets`**. Point it at a bigger or faster drive if your default data directory is tight on space. This folder (and every dataset folder under it) is refused as an **image bank** source: a bank points at a live folder and can delete from it, so the two must never share files — see *Using the app → A bank and a dataset never share files*. Moving this root onto a folder an existing bank already uses is not blocked here, but that bank will say so the next time you open it, and its 🗑 Delete rejected will be refused.
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

**Quality-tool interpreters and models:**

| Key | Default | Role |
|---|---|---|
| `face_scoring.python` | `''` | Interpreter for the InsightFace subprocess (empty = current interpreter). |
| `face_scoring.models_root` | `''` | Where InsightFace weights are stored/downloaded. |
| `face_scoring.device` | `'auto'` | Device for the Image-bank face pass. `auto` uses the GPU when the face interpreter exposes CUDA (needs `onnxruntime-gpu` installed in it) and falls back to CPU otherwise; `cpu` forces CPU (never touches the GPU); `cuda` requests the GPU but still falls back to CPU when unavailable. A GPU run is serialized through the GPU-exclusive window so it never competes with a training/scoring pass. |
| `masks.python` | `''` | Interpreter for the rembg (person-mask) subprocess. |
| `bank_scoring.text_search_idle_minutes` | How long the 🔤 **Find by text** encoder stays warm after its last query (default `10`, capped at `120`). Loading CLIP costs ~10 s on the CPU; encoding a phrase afterwards costs ~20 ms, so the worker is kept alive to make a refine-and-retry session instant. It holds roughly **2.4 GB of RAM** while it lives, and is released when you close the search panel or when the window elapses. Set to `0` to never keep it warm — every new phrase then pays the ~10 s load, which is the right trade on a memory-tight machine. Already-searched phrases are cached on disk and cost nothing either way. |
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
| `dataset_import.max_side` | Longest side for opt-in WebP normalization (default `1024`; `0` = original size). It is ignored by the default `preserve` mode; ratio is always preserved, never enlarged, and normalized paths clamp at 8192 px. Every source must still be at most 16 Mi-pixels and 8192 px per side; a larger one is rejected and must be converted or resized before import. Not retroactive. Editable in Settings → Captioning & quality. |
| `dataset_import.encoding` | How an un-cropped imported image is written: `preserve` (default; original JPG/JPEG, PNG, WebP or BMP bytes with the matching extension), or the opt-in WebP modes `standard` (q92), `high` (q100), and `lossless`. Auto head-crop is always a derived WebP. The 16 Mi-pixel / 8192 px-per-side input limit applies to every mode. Editable in Settings → Captioning & quality. |
| `comfyui.api_url` | Base URL of your ComfyUI instance (default `http://127.0.0.1:8188`). |
| `comfyui.base_dir` | ComfyUI install directory, used to derive `output`/`input`/`models`/`loras` dirs if those aren't set explicitly. |
| `comfyui.output_dir` | Explicit override for ComfyUI's output folder. Set it when ComfyUI runs with `--output-directory`. Editable in Settings → Local tools. |
| `comfyui.input_dir` | Explicit override for ComfyUI's input folder. Set it when ComfyUI runs with `--input-directory`. Editable in Settings → Local tools. |
| `comfyui.models_dir` | Explicit override for ComfyUI's models folder (used to scan available checkpoints/UNETs). `extra_model_paths.yaml` is still read on top of it. Editable in Settings → Local tools. |
| `comfyui.object_info_timeout_s` | Seconds ComfyUI may take to enumerate its nodes and model files (default `45`, clamped to 5-300). Raise it on an install with many custom nodes; a ComfyUI that is simply stopped is still detected in ~3 s regardless. Editable in Settings → Local tools. |
| `comfyui.loras_dir` | Explicit override for ComfyUI's LoRA folder — where trained LoRAs are installed. Wins over `extra_model_paths.yaml`; left empty, the deploy folder follows the yaml's `is_default` LoRA root, else `<install>/models/loras`. Editable in Settings → Local tools. |
| `ollama.url` | Base URL of your Ollama instance (default `http://127.0.0.1:11434`). |
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
| `cloud.max_concurrent_runs` | Simultaneous cloud pods allowed (default `1`, 1–10). Also in Settings → Training. |
| `cloud.max_price_per_hour` | Safety cap on the hourly offer price in $ (default `0.80`); pricier hosts are skipped before launch. |
| `cloud.monthly_budget_usd` | Hard monthly spend ceiling in $ (default `0` = unlimited); launches are blocked past it. |
| `cloud.stall_timeout_minutes` | Kill + rescue a cloud run after this many minutes without step progress (default `30`, 5–240). |
| `cloud.first_step_timeout_minutes` | Kill a run that reaches no training step **and** reports no new downloaded bytes for this long (default `45`, 5–240). Also in Settings → Training. |
| `cloud.first_step_download_budget_minutes` | Absolute ceiling on the pre-training base-model download, even while it is progressing (default `180`; `0` = no ceiling). Also in Settings → Training. |
| `cloud.max_runtime_minutes` | Hard stop on the whole run (default `480`, 30–1440); the newest checkpoint is rescued first. Enforced by the out-of-run supervisor too. Also in Settings → Training. |
| `cloud.freeze_watchdog_minutes` | Terminate a training run whose **pod** shows no progress for this long (step, download bytes or a new checkpoint), from outside the run's own supervision; the clock is durable and survives an app restart (default `45`; `0` = warn on the card only). |
| `cloud.min_reliability` | vast.ai host-reliability floor (default `0.98`, 0.9–0.999); lower surfaces cheaper, riskier hosts. |
| `cloud.verified_only` | Restrict to vast.ai verified hosts (default `true`). |
| `cloud.secure_cloud_only` | Restrict to vast.ai's Secure Cloud (datacenter) tier (default `false`; narrows the market, raises price). |
| `face_scoring.python` | Python interpreter used to run the InsightFace subprocess (empty = current interpreter). |
| `face_scoring.models_root` | Directory where InsightFace model weights are stored/downloaded. |
| `face_scoring.green` | Similarity score threshold (0–1) above which an image is flagged "green" (strong match). |
| `face_scoring.orange` | Similarity score threshold (0–1) above which an image is flagged "orange" (borderline match). |
| `masks.python` | Python interpreter used to run the rembg subprocess (empty = current interpreter). |
| `bank_scoring.python` | Python interpreter that runs the ✨ Score pass (empty = the app's own). Auto-filled by Setup with a CPU-only environment; repointable at any CUDA interpreter already on the machine via the bank's **⚡ Use a GPU Python I already have** picker, which verifies every dependency first and never installs into an environment it did not create. |
| `watermark.python` | Python interpreter used to run the LaMa watermark-inpainting subprocess (empty = reuse `masks.python`, then the current interpreter). |
| `watermark.device` | LaMa processing device: `auto` (CUDA when available, otherwise CPU), `cuda`, or `cpu`. |
| `watermark.allow_crop` | When `true` (default), a border watermark is cropped off; when `false`, it is repainted instead. Also editable in the Clean bar. |
| `zimage.vae` | Pins the Z-Image VAE (blank = the app resolves it itself, any spelling, any sub-folder). |
| `zimage.text_encoder` | Pins the Z-Image text encoder (blank = the app resolves it itself). |
| `klein.consistency_lora` | Filename of the Klein consistency LoRA, relative to ComfyUI's LoRA folder. |
| `klein.consistency_strength` | Strength (0–1) applied to the Klein consistency LoRA. |
| `klein.generation_steps` | Sampler steps for Klein **generation** (variations, regenerate, small-image rescue). Default `5` = the value hardcoded in the shipped workflow; 1–50. Not the improve pass (`klein.improve_steps`). |
| `klein.generation_lora_presets` | Named generation-LoRA stacks (default empty) picked per run in Klein tuning; each has a name and up to 8 `{file, strength}` rows. Managed in Settings → Image engines. |
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

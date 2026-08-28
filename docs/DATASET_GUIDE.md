# Building a good LoRA dataset

This guide condenses what actually moves the needle when training a character LoRA
with this app (ai-toolkit under the hood). Every number here matches what the app
enforces or defaults to — when in doubt, the app's warnings are this guide applied.

> **The one principle behind everything:** a LoRA learns whatever is **constant
> across your images and NOT described in the captions**. Keep the subject constant,
> vary everything else, and never describe the subject — that's the trigger word's job.

---

## 1. Pick your model family first

The family changes the caption style, the image count, and the settings — so decide
before you caption anything.

| | Z-Image | SDXL | Krea 2 | FLUX.1 | FLUX.2 Klein |
|---|---|---|---|---|---|
| **Caption style** | Prose sentences | Booru tags | Prose sentences | Prose sentences | Prose sentences |
| **Images (min → good)** | 12 → 20+ | 20 → 30+ | 15 → 20+ | 15 → 20+ | 15 → 20+ |
| **Training base** | Z-Image-Turbo (or a converted custom merge) | Your ComfyUI checkpoint (e.g. bigLove) | Krea-2-Raw (default), Turbo, or a Krea 2 checkpoint on your disk | FLUX.1-dev (gated HF) | FLUX.2-klein-base 4B (default) or 9B (gated HF) |
| **Preview quality** | Fast, distilled | Depends on checkpoint | Raw: slow but faithful | High, ~20 steps | Non-distilled, real CFG (~25 steps) |
| **Best for** | Fast iteration, prose-driven prompting | Booru-native checkpoints, NSFW ecosystems | Highest realism ceiling | The largest LoRA ecosystem, strong prompt fidelity | Modern FLUX.2 stack; 4B trains on mid-range GPUs |

**Krea note:** the default trains on **Krea-2-Raw** — the official recommendation is
*"train on Raw, validate on Turbo"*. Raw runs are long (hours); that's normal, not stuck.
The **Base** selector also lists every Krea 2 checkpoint sitting in your ComfyUI
`unet` / `diffusion_models` folders — a model one of your own full-model runs
delivered, or a community Krea 2 build — so you can keep training on top of one
instead of starting from the official weights every time. Entries carry a tag when
the file is quantized: `· fp8 cast` trains but starts from degraded weights,
`· packed export` cannot be loaded at all (see *Which quantized checkpoints can be
trained on* in section 10). Local runs use the file directly; a cloud run first
pushes it to your private Hugging Face repo, which the panel offers to do.

**FLUX.1 note:** trains on **FLUX.1-dev**, a *gated* Hugging Face model — accept its
license and set a HF token before the first run (the initial download is ~24 GB). It's
a 12B model like Krea 2, so **~24 GB VRAM** is the comfort zone (drop the resolution to
**768** to fit smaller cards). **Local training only for now**; in-app testing (Test
Studio) is coming — until then, test your Flux LoRA in your own ComfyUI.

**FLUX.2 Klein note:** two model sizes, picked next to the base selector — **4B**
(default) trains on a **16–24 GB** local GPU, **9B** needs **32–48 GB VRAM** and is
best trained via **☁️ Train in cloud** (both local and cloud runs are supported for
this family). Both bases are *gated* on Hugging Face: accept the license of
`FLUX.2-klein-base-4B` / `-9B` and set a HF token before the first run. In-app
testing (Test Studio) is coming — until then, test your Klein LoRA in your own
ComfyUI.

**Anima note (the one family that takes BOTH caption styles):** Anima is an anime
model with **hybrid prompting** — its model card documents *booru tags* and *natural
language* as equally supported, which its LLM text encoder is what makes possible. So
this is the family where the "match the style" rule below does **not** apply: caption
in prose, caption in booru tags, or keep an existing dataset as it is — the app will
not flag either as a mismatch, and you never have to force the launch. Prose is only
the preselected default. It trains on the open `Anima-Base-v1.0-Diffusers` (no gated
download) and is **local-only** for now.

---

## 2. How many images, and which ones

- **Target ~25 images** for a balanced character LoRA. More isn't automatically
  better — 25 varied images beat 60 near-duplicates every time.
- **Balance the framing.** The app tracks four buckets: **face / bust / body / back**.
  A dataset that is 100% face close-ups produces a LoRA that falls apart on
  full-body prompts — it has never seen the body.
- **Imported images may have no shot type yet.** Only images imported with the
  head-crop option on are tagged automatically; a plain drag-and-drop import (the
  default on body-fidelity datasets) leaves the shot type unknown, and unknown
  images count for nothing in the Composition bar — a whole import can leave it
  at 0. **📐 Classify framing (N)**, right under that bar in 📸 Add images, reads
  those images with the local vision model (Ollama) and sorts each into face /
  bust / body / back. It needs Ollama running with a vision model pulled
  (Settings ▸ Local tools); it uses the GPU and waits rather than competing with
  a training run. Nothing is deleted and images it cannot read stay unknown, so
  running it again only retries those.
- **A crop forgets the old shot type.** Cropping a body shot into a face (or a
  bust into a close-up) clears the stored framing, the same way a Bank crop
  does. Composition drops that image from its bucket until you run **📐 Classify
  framing** again — and the button only counts the ones that actually changed,
  not the whole set. Same vision model, same GPU wait.
- **Vary everything except the person:** location, lighting, outfit, pose,
  expression, camera angle. Whatever repeats across images gets baked into the
  LoRA — a repeated background wall becomes part of "the person".
- **Reject near-duplicates.** Two frames of the same shot teach nothing and
  overweight that look. The pre-flight check flags them; reject one of each pair.
- **Quality floor:** no motion blur, no heavy compression, the face readable.
  One bad image does more harm than one good image does good.

**Body fidelity mode** (Datasets → ⋯ More): use it when the body shape and body
marks (tattoos, scars) should bind to the trigger too. It shifts the composition
targets toward bust/body shots, imports full-frame by default, and extends the
caption rules below to body marks.

---

## 3. Captions — the make-or-break step

The model reads your captions during training and learns to attribute **whatever
the caption does NOT explain** to the trigger word.

**The golden rule: never describe what the person IS — describe everything else.**

- ❌ `myTrigger, a woman with long blonde hair and blue eyes, smiling` —
  the LoRA learns almost nothing: the caption already "explains" the appearance.
- ✅ `myTrigger, sitting at a café table, warm afternoon light, denim jacket,
  looking at the camera` — hair, face and skin are unexplained → they bind
  to `myTrigger`.

Concretely:

1. **Start every caption with the trigger word.** The app injects it on export.
2. **Never mention face, eyes or skin** — and, by default, hair. Those bind
   to the trigger. ⚙️ *Options* on the Captions panel has **Appearance in
   captions**: flip Hair, Makeup, Facial hair or Glasses to **Describe** when
   you want that look prompt-controllable (different hairstyles, no mascara in
   every gen). **Omit** keeps it bound to the trigger. Face, eye colour, skin,
   age, gender and ethnicity stay omitted. Extra instructions cannot reintroduce
   an omitted family — flip the row instead. The *identity-leak* check watches
   whatever is currently omitted.
3. **Describe scene, outfit, pose, lighting, framing** — and any appearance
   family you set to Describe. Those stay promptable *independently* of the
   identity.
4. **Vary the captions.** Identical captions across images teach nothing;
   captions under ~8 words are too weak to isolate the identity.
5. **Match the style to the family.** Prose for Z-Image and Krea; booru tags for
   SDXL booru-native checkpoints. The app blocks a mismatch for a reason —
   a prose-captioned SDXL LoRA produces disjointed images. **Anima is the
   exception:** it reads both forms natively, so neither is ever blocked there
   (see the Anima note above).

   ⚠️ **Concept datasets cannot be captioned in booru tags at all** (the concept
   captioner only writes prose). A Concept dataset on a booru-native SDXL
   checkpoint will therefore always be stopped by the caption-style check: train
   the concept on a prose family instead, or force the launch knowing the cost.

**Caption length.** ⚙️ *Options* on the Captions panel carries a **Caption length**
preset — *Standard* (the prompt untouched), *Concise* (aims for one short sentence,
~20–30 words) or *Detailed* (several sentences). It is a **target the vision model
follows loosely**, not a hard cap: expect a spread around it, not a word count. Pick
*Concise* when detailed captions keep describing the identity you want bound to the
trigger, *Detailed* when you want scene, outfit and lighting to stay independently
promptable.

What that looked like when measured — 18 real portrait photos, the shipped default
vision model (`huihui_ai/qwen3-vl-abliterated:8b-instruct`), the plain descriptive
prompt, one pass per preset:

| Preset | Median | Range |
|---|---|---|
| Concise | 24.5 words | 18–30 |
| Standard | 87.5 words | 65–112 |
| Detailed | 126 words | 106–152 |

Your numbers will differ — another vision model, JoyCaption, or a different kind of
image all move them. Treat the presets as *shorter / as-is / longer*, not as a
contract on a word count.

Two more things worth knowing:

- **Order.** The prompt is built as: the base prompt with its omission rules, then the
  vocabulary register, then the length preset, then your free **Extra instructions**
  last — so a hand-written steer that contradicts a preset is what the model reads
  most recently and wins. The identity/concept leak cleaners run after all of it
  regardless, so Extra instructions cannot reintroduce an omitted identity term.
  Flip **Appearance in captions** (Hair / Makeup / Facial hair / Glasses) when
  you *want* that look in the caption so it stays prompt-controllable.
- **Concise is not the "short" of long + short captions.** Dual captions derive a
  short variant *from* the stored long caption into its own field; the length preset
  changes the long caption itself. They are separate axes and compose freely.
- Concise stays **prose** on purpose (never a comma-separated tag list), so a Concise
  dataset still passes the caption-style check for prose-native families instead of
  being mistaken for booru tags at launch.

**Concept datasets** (training a *thing/style/act*, not a person) invert the rule:
describe everything **except the concept** — the concept is what must bind to the
trigger. Keep *person* masking **off** for concepts — a person mask would erase the
very thing you're training. Masking **faces** is the opposite polarity and is
available on purpose: see §8.

**Stopping a run.** Started a big caption pass and realized it's captioning badly,
or an option was mis-set? A **⏹ Stop** button sits in the captioning progress
banner. It finishes the image being written (an inference is never cut off
mid-way), then stops cleanly: every caption written so far is kept, the rest is
left untouched, and you get a *"stopped — X captioned"* summary. Nothing is killed
and nothing already done is lost — just fix the option and run again on what's left.

---

## 4. Settings cheat-sheet

The defaults below are the app's defaults (post-research). Change them from
⚙️ Advanced options on the training panel — each knob has its own why/how there.
That panel also has a **Presets** row: apply a shipped ★ recipe (*Krea
character*, *Concept*, *Style*), or save your tuned settings as a named preset to
reuse across datasets and share (import/export as JSON).

| Setting | Z-Image | SDXL | Krea 2 | FLUX.1 | FLUX.2 Klein | Why |
|---|---|---|---|---|---|---|
| **LoRA rank / alpha** | 16 / 16 | 32 / 16 | 32 / 32 | 16 / 16 | 16 / 16 | Capacity to memorize the identity. SDXL's alpha = rank ÷ 2 is that family's half-strength convention. |
| **Resolution** | 768 + 1024 | 768 + 1024 | 768 + 1024 | 768 + 1024 | 768 + 1024 | Multi-scale: holds up from close-up to full-body. |
| **Save checkpoint** | every 250 | every 250 | every 250 | every 250 | every 250 | More snapshots → better odds one is at the sweet spot. |
| **Steps** | auto | auto | auto | auto | auto | ~120 × images, clamped 1500–3500. A fixed 3000 overcooks small sets. |
| **Masked training** | ON | ON | ON | ON | ON | Background weighs only 10% of the loss → identity binds to the person, not the room. OFF for concepts — they have their own face masking instead (§8). |

Rules of thumb:

- **Raise rank (48–64)** only for a hard identity (distinctive features the
  default misses) *and* a bigger dataset — high rank on 15 images just memorizes them.
- **Don't chase steps.** More steps past the sweet spot = overfitting (plastic
  skin, same face angle everywhere, prompt deafness). Train with checkpoints
  every 250 and pick the best one instead.
- **Turbo variant (Krea)** is the VRAM/time-friendly fallback — fine for drafts,
  Raw for the final run.
- **GPU under 24 GB?** Resolution is the #1 memory lever: set it to **768 only**
  (Krea 2 especially — 1024 saturates a 24 GB card). You trade some fine detail
  for a run that actually fits and trains far faster.

### Steps — how many, and where "good results" start

The app sets the step count **automatically** for a character LoRA:
**≈ 120 × kept images, clamped to 1500–3500.** The *target is the same* for
Z-Image, SDXL, Krea 2, FLUX.1 and FLUX.2 Klein — the model family changes how *fast*
that target converges, not the number. (Concept/style datasets scale differently:
**475 · √n, clamped 2000–12000**, because they train on hundreds of images.)

So the character step count just follows your dataset size:

| Kept images | Auto steps |
|---|---|
| 12–15 | 1500 – 1800 |
| 20 | 2400 |
| 25 | 3000 |
| 30 and up | 3500 (capped) |

**"Good results" is a checkpoint you pick, not the finish line.** A snapshot is
saved every 250 steps, and the best one is almost never the last — later
checkpoints know the face better but obey prompts worse. *Where* the first
usable checkpoint appears depends on how fast the model converges:

| Model | Converges | Where the sweet spot tends to land |
|---|---|---|
| **Z-Image** | Fast (distilled) | Around the **middle** of the run; watch for overfit in the last ~20% (waxy skin, frozen expression) |
| **Krea 2 – Turbo** | Fast (distilled) | Like Z-Image — check early-to-middle checkpoints first |
| **SDXL** | Medium (base-dependent) | Middle of the run; booru-native checkpoints lock an identity quickly |
| **Krea 2 – Raw** | Slow (12B, non-distilled) | The **last third** — the run is long by design, let it finish the full count rather than stopping early |
| **FLUX.1-dev** | Medium (12B, guidance-distilled) | Middle of the run; a strong prompt-follower, so watch for waxy skin / frozen expression if you overshoot into the last ~20% |
| **FLUX.2 Klein (4B/9B)** | Medium (non-distilled base) | Middle of the run; previews run with real CFG so overfit shows honestly — pick the earliest checkpoint that holds the identity |

**Takeaway:** don't hand-tune the step number. Train the auto count, then use the
**Test Studio** to pick the *earliest* checkpoint that nails the identity — that's
the one with the most prompt flexibility left.

---

## 5. Pre-flight checklist

The app runs these checks when you hit Train — here's the list to self-check earlier:

- [ ] At least the family minimum kept (12 Z-Image / 20 SDXL / 15 Krea / 15 FLUX.1 / 15 FLUX.2 Klein) — 20–30 is the comfort zone
- [ ] Framing balanced — not 100% face shots (some bust/body/back)
- [ ] Every kept image captioned *(strongly recommended — a blank caption won't block the launch, it just asks you to confirm "train anyway")*
- [ ] **Zero identity leaks** (the leak badge shows 0 for whatever is currently omitted — face/eyes/skin, and by default hair)
- [ ] Captions varied, ≥ 8 words, style matches the family (prose vs booru — Anima takes either)
- [ ] Near-duplicate pairs resolved (keep one of each)
- [ ] Body fidelity: if ON, actual full-body shots exist

**Continue anyway.** When the readiness panel turns red over a *quality* blocker —
most often too few images for the family — a **Continue anyway** checkbox appears
under the list. Tick it and the Train button unlocks; the launch is recorded as
"acknowledged not-ready" in its saved config. It's meant for deliberate
experiments (you'll usually get an overfit LoRA), not for skipping the work. The
checkbox only ever covers quality guard-rails: genuine impossibilities that would
just crash the trainer — **zero kept images**, or a **slider with no prompt pair**
— are never offered the option, and the box un-ticks itself the moment the
blockers change.

**Stopping a training run.** The red **⏹ Stop training** button next to Train
ends the run in progress — it is not a housekeeping button. It kills the training
process, clears the pending local training queue, and hands the GPU back to
ComfyUI. What you keep: **every checkpoint already saved**, which stays testable
in the Studio and can be continued later with ▶ Continue. Because a run can be
hours long, the button asks for confirmation first. The same run can also be
stopped from the **Runs** hub ("Stop run"), which does exactly the same thing.

---

## 6. After training: pick the right checkpoint

Training produces a checkpoint every 250 steps — **the last one is often NOT the
best one**. Later checkpoints know the identity better but obey prompts worse.

1. Open the **Test Studio** from the dataset (the LoRA comes pre-selected).
2. Generate the same prompt grid across several checkpoints and strengths.
3. Pick the **earliest checkpoint that nails the identity** — it keeps the most
   prompt flexibility. Signs you've gone too far: waxy skin, identical
   expression/angle regardless of prompt, outfits from the dataset bleeding in.
4. Save the winning settings (★) — they're reused as the dataset's defaults.

### Test several prompts in one launch

Under the prompt box is the history of the prompts you have saved, with a
thumbnail of the image you liked best for each. Clicking a card loads it into the
field, as before. **Ticking its box adds it to a batch**: the panel counts what is
selected, the button says how many prompts it is about to run, and one launch
renders them all — same checkpoints, same settings, **same seed**, which is what
makes two prompts comparable rather than two unrelated pictures.

It is one run, not several: the images queue up and the GPU works through them by
itself. Tick nothing and the screen behaves exactly as it always has, running the
prompt in the field.

**There is no limit on how many you tick.** What there is instead is the price,
shown before you click: the panel counts every generation the run will queue and
estimates how long it takes **at the pace your machine has actually been running
at** — measured from your own recent test generations, not assumed. Past about an
hour it asks once whether you meant it. The queue is serial, so you can stop it at
any point and everything already generated is kept.

The same tick boxes are in **🎨 Generate from the board** on the ◉ LoRA Canvas,
because both screens show the same prompt history.

### Compare LoRAs — or blend them

Check two or more LoRAs and Studio asks what you want to do with them:

- **⚖ Compare** (the default) tests each LoRA **on its own**, one column per LoRA,
  swept across the strengths you picked. This is what you want to answer "which of
  these is better".
- **🧬 Blend** loads them **together in the same image**, each at its own weight,
  and injects **every trigger word** into the prompt for you. This is what you want
  to answer "do these two work together" — a character plus a style, or a character
  plus a concept.

> This mode was called **🧬 Combine** until August 2026. Only the name changed;
> the ◉ LoRA Canvas offers the very same thing from the board, and calling it two
> different things was a needless thing to learn twice.

**What blending two characters actually gives you** is a *hybrid* — one person who
is neither of the two, not both of them side by side in one shot. That is a real
and deliberate use, but if you expected "my two characters together", this is not
it. The reliable pairings are **character + style** and **character + concept**.

In Blend mode the strength sweep disappears: each LoRA already carries its own
weight, so the run is one configuration instead of a grid. Start both around
0.7-0.9 — two LoRAs at 1.0 usually fight each other, and the one you care about
most should be the heavier of the two. Result tiles from a stack carry a **🧬**
badge naming the exact weights that made them.

**Steps and CFG are set in the same panel, in both modes.** They are render
settings, not LoRA settings, so they stay available when the strength sweep
disappears in Blend — and like every other axis, ticking two values renders both
(the cell counter shows what that costs before you launch). SDXL also exposes its
second pass there.

**Trying several weights at once.** Under each LoRA's slider is a row of weight
boxes. Tick two on one LoRA and two on the other, and the launch renders **all
four combinations** in a single run — the search you would otherwise do by
launching, looking, moving a slider and launching again. Each image is labelled
with its own pair, and the stack view lines the combinations up side by side so
you can pick the one that works and save its weights with ★.

Tick nothing and the slider governs, exactly as before the boxes existed; the
slider is also how you use a weight that is not on the grid. Tick one box and you
get one configuration — one image — like any other blend.

The count is spelled out before you launch ("4 weight combinations → 4 images,
about 1 min"), and past 24 images it turns amber and says so. It never refuses:
the queue is serial and it is your machine. Two LoRAs at four weights each is 16
images — the multiplication is quick, which is exactly why the panel does it for
you.

**One family per run, always.** A Krea LoRA and an SDXL LoRA cannot be blended:
they need different base models and different workflows. The picker greys out the
other families as soon as you check one, and a run that somehow mixes them is
refused with both family names in the message.

### Enhance a short prompt

**✨ Enhance** rewrites what you typed into a fuller prompt using your local Ollama
model — it adds framing, pose, lighting, background and mood, and deliberately
leaves identity and trigger words alone (the LoRA supplies the identity, and Studio
injects the trigger itself at generation time).

By default it runs the same model your captions use. The **⚙️ next to the button**
picks any other pulled Ollama model instead — the choice applies immediately, is
remembered on that browser, and drives the same button on the Canvas run panel. A
vanilla model can refuse NSFW prompts; the abliterated captioning default is the
safe choice there.

It is a local feature: without Ollama installed, running, and with its model pulled,
the button is **greyed out and says which of the three is missing** rather than
failing when you press it. Install or start it from **Settings › Local tools**.
(With a ⚙️ model picked, the last check moves server-side: the refusal names the
picked model instead of greying the button on the default one.)

### Reuse a dataset caption in Studio

Press **🎲 Caption** for a realistic test prompt from work you already curated.
The first use asks which dataset to draw from; after that, each main-button click
inserts a random **nonblank caption from a kept image** in that dataset. Studio
remembers the chosen source in this browser's localStorage. Use **▾** beside the
button to change the source dataset.

The source needs at least one kept image with a nonblank caption. If you have
typed a prompt, Studio asks before replacing it.

### Borrow a prompt from Civitai's top images

**🌐 Civitai** (next to the prompt field, on every generation surface) browses
the most-reacted Civitai images of the day, week, month, year or all time —
each image shown side by side with the generation prompt it was posted with.
**⤵ Use prompt** drops it into your prompt field (asking first if you typed
something), **📋 Copy** puts it on the clipboard, and clicking the picture
opens it on Civitai.

Two honest limits:

- **Not every image publishes its prompt.** The browser keeps only the ones
  that do by default; untick *Only images with a prompt* to see the full top.
- **Reading prompts needs a Civitai API key** (free account) — the same key
  the scraper uses, stored once in **Settings › Scraping & sources**. Without
  it the top images still show, but Civitai refuses the prompt data.

The content-level select is a ceiling (*Safe* by default, up to *Everything*);
your filters are remembered in this browser's localStorage.

### Continue a run instead of starting over

If the best checkpoint is *almost* there — the identity nearly locked but a touch
undercooked — you don't have to retrain from scratch. The **▶ Continue training**
button (on the dataset's Checkpoints panel and on the **Runs** hub) opens a small
dialog:

- **Resume from** — which checkpoint to restart from. The default is the latest,
  but the whole point is that you can pick an **earlier, less-cooked epoch**: the
  classic case where step 750 held up better than the over-cooked 1000. Choosing
  an earlier step never destroys the run's later saves — they're set aside intact
  (on disk locally, in the run's staging for cloud) and the continuation writes
  its own.
- **Extra steps** — how many *more* steps to train; the dialog shows the target
  step you'll land on.
- **Adjust settings (optional)** — a resume can only safely change a handful of
  things: the **checkpoint/preview cadence**, the **preview prompts** and the
  **preview steps and CFG** (test images only — never the weights), and the
  **timestep weighting**. Everything structural
  (rank, base model, optimizer) is locked to the checkpoint you're continuing.
  The timestep knob enables a known **two-phase recipe**: train balanced first,
  then continue with a low-noise-leaning emphasis to polish fine texture.

- **Run it** — **💻 Local** or **☁ Cloud**. A checkpoint is just a
  file, so where a run trained doesn't decide where it can be finished: a run
  trained on your GPU can be continued on a rented one (the checkpoint is uploaded
  and training picks up from it, on a fresh pod, leaving every local save
  untouched), and a cloud epoch mirrored into your run folder can be finished
  locally. A lane you can't use right now — no vast.ai key, no ai-toolkit, a
  training already running here, a cloud limit reached — is disabled **with the
  reason**, never hidden. The same choice is offered by the **Runs** page's
  ▶ Continue, where the cloud reason is counted against *that run's* dataset —
  the page lists runs from all of them.

You can also click a checkpoint pill in the **◉ Graph** and pick *▶ Continue from
here*: the dialog opens already set on that step.

Continue works for both **local and cloud** runs from the Runs hub.

## 7. Dual captions (long + short)

An optional, **off-by-default** training technique, toggled under **⚙️ Advanced
options → Dual captions** on the training panel. When on, the run uses
ai-toolkit's native `short_and_long_captions`: **every image trains with both its
full caption and a short one.** It's a *text-side augmentation* — showing the
model two phrasings of the same image so the LoRA leans less on any single
wording and generalizes to prompts that don't match your caption style.

How the short caption is produced:

- It's **derived from the long caption**, automatically, the next time you
  (re-)caption — text-only, via the local vision model. Turning the toggle on
  doesn't rewrite anything by itself; **re-caption** to generate the shorts.
- It follows the **same kind rules** as the long one: no trigger word, and the
  identity / concept / aesthetic stays omitted (that's still the trigger's job).
- You can **edit it per image** in the **⛶** caption editor, next to the long one.

**Local training only for now.** The cloud pod's dataset upload doesn't carry the
JSON file the short caption is read from, so **cloud runs train on the long
caption alone** — turning the toggle on simply has no effect there yet.

**Not on Krea 2 or Anima.** Those two families pre-cache their text embeddings and
unload the text encoder to fit their DiT in VRAM. ai-toolkit caches exactly one
embedding per image — the long caption — and once the encoder is gone the training
loop reads those cached embeddings instead of the caption text, so a second caption
has nowhere to be encoded. Asking for both used to crash the run at the first step,
*after* the weights download and the whole caching pass (reported by **1Tomber**,
GitHub #22). The app now refuses the combination when it builds the training config:
the toggle says so, the pre-launch check warns, and the run trains on the long
caption alone — trigger word included, exactly like a normal run.

---

## 8. Concept LoRAs: keeping faces out

A Concept LoRA learns the one thing every image shares. If those images all show
people, it quietly learns **their faces too** — and when you later stack it with a
Character LoRA, the two pull against each other over whose face to render. This was
reported by **shivdbz2010 (GitHub)**.

Turn on **Mask faces** in *Advanced options* on a Concept dataset. Faces are
detected and **weighed down in the training loss**, so the concept binds to the act
instead of to the people in your photos.

**Your images are not touched.** Nothing is blurred, pixelated or painted over.
That distinction matters: a blurred face would *be* what the model is trained to
reproduce, and the LoRA would learn to render blurry faces. A loss mask says
"don't correct me here" instead, so nothing at all is learned in that area.

Before you rely on it:

- **Variety beats masking.** The people who maintain these trainers say dataset
  diversity matters more here. A concept demonstrated by ten different people
  already dilutes identity; with two, the faces are as constant as the concept and
  no mask fully compensates.
- **Preview it.** The training panel draws the mask on your own shots and shows how
  many images got no face at all. A *partly* masked set is the bad case: the faces
  left unmasked become the only ones the LoRA still learns faces from, so they end
  up over-represented.
- **You can stop the preview, and it resumes.** On a large set the pass takes a
  while, so **Stop** is next to it — and what it already found is kept. Start it
  again and it continues from where it stopped rather than from image 1. The
  button says what stopping costs at the moment you press it, because that
  changes: the face detector is loaded before the first image and that load is
  paid again on every start, so stopping *during* the load gives up only the
  load, while stopping *during* the analysis keeps every face found so far.
  Change your kept images and the saved work is dropped instead of reused —
  boxes detected on photos that left the set would describe a run that no longer
  exists.
- **If your concept lives on the face** — an expression, a mouth, a gaze — masking
  the head can erase what you're teaching. The app warns when your description says
  so; it doesn't stop you, because only you know your dataset.
- **Nobody has measured this.** There's no published before/after of a concept LoRA
  trained with and without face masking. This gives you the lever, not a promise.

Two knobs live in **Settings ▸ Training**: how far the detected face box is grown
into a head, and how much the masked area still counts. Neither is zero, on
purpose — see the settings reference.

---

## 9. Coverage — what your set never showed

Section 2 says "vary everything except the person". The Composition bar cannot
check that: it counts face / bust / body / back against a target, so a set of
twenty-five front-on studio portraits in one outfit reaches a **fully green
target** while having no profile, no daylight and no second outfit. The LoRA that
comes out reproduces that one look and nothing else.

**🔍 Coverage**, the collapsible panel right under the Composition bar, is that
second check. Open it and it reports, per axis, what your captions describe and
what they never mention:

| Axis | What a gap means |
|---|---|
| Camera view | frontal / three-quarter / profile — a character with no profile has a side nobody ever saw |
| Camera height | eye level / low / high / overhead — eye-level-only is the default trap |
| Lighting | daylight, indoor, golden hour, studio, night, backlit, overcast |
| Setting | indoor, outdoor, urban, plain backdrop, water, vehicle |
| Outfit | counts how many **distinct** outfit types appear — one outfit gets learned as part of the person |
| Expression | counts how many distinct expressions appear |

Which axes apply depends on the dataset kind. A **style** dataset is judged on
lighting, setting and view only — "one outfit" is not a defect when the outfit is
not what you are teaching. A **concept** dataset drops the expression axis.

### What it can and cannot see

This is deliberately a cheap check, not a second model. It reads **the words in
the captions you already generated** — nothing new runs, there is no GPU cost,
and the numbers appear instantly. That comes with real limits, and the panel
repeats them on screen rather than hiding them:

- **No captions, no reading.** With an uncaptioned dataset the panel says so
  instead of drawing empty bars. Run the caption pass first.
- **It sees descriptions, not pixels.** A profile shot the captioner described
  without the word "profile" is invisible here. An absence is strong evidence,
  not proof.
- **Negation is not parsed.** "not smiling" counts as a smile.
- **Under five captions it refuses to judge** — at that size everything looks
  missing for the wrong reason.
- **It never selects, keeps, rejects or changes anything.** It is advice.

### Clicking a chip shows you those images

A number tells you *profile 3*; it does not tell you **which** three, and hunting
for them by eye in a grid of two hundred is the part that made the panel easy to
read and hard to act on. **Click any chip that has a count** and the grid opens
filtered to exactly the images that chip counted, with `🔍 profile — camera view`
in the filter bar and the usual *clear all* next to it.

It stays advice: filtering changes which images you are *looking at*, never what
they are. Nothing is kept, rejected, recaptioned or reordered by the click, and
removing the chip brings the whole grid back.

Two things follow from the panel reading captions rather than pixels, and they
are worth knowing before you trust a filter:

- **The filter shows what the chip counted, no more.** Rejected and failed images
  are outside the panel's pool, so they stay outside its filter — the number and
  the images you get can never disagree.
- **A chip with a zero is not clickable**, because there is nothing to show. That
  is the gap the panel is pointing at, and the answer to it is generating or
  importing, not filtering.

Pair it with **Sort ▸ Shot type** on the grid and the two compose: filter to the
profiles, group what is left by shot type, and decide what to keep with like
sitting next to like.

The panel reads the same pool the Composition bar counts: everything that is not
rejected and not failed. It also tells you how many images have **no shot type
yet**, which is the one thing the bar above silently drops.

## 10. Full-model recipe — what you can change

Full-model (dense) training is a different animal from a LoRA: instead of a small
adapter, it rewrites all 12B weights of Krea 2 Raw. That only fits on one 80 GB
card under a specific geometry, so most of the recipe is locked — and the panel
now says which parts and why.

**Locked, and not negotiable**

| Locked | Why |
| --- | --- |
| Batch size 1, bf16 | The 80 GB budget has no room for more. |
| Adafactor | Adam-family optimizer states would not fit alongside the weights. |
| Gradient checkpointing, cached latents + text embeddings | Same reason — turn any of them off and the run dies out of memory, an hour in, on a rented GPU. |

**Editable, because these change the RESULT rather than whether it fits**

| Setting | Default | Range | Why you would move it |
| --- | --- | --- | --- |
| Steps | adaptive | ≥ 500 | Longer runs on larger sets. |
| Preview prompts | generic per kind | up to 8 lines | The defaults describe nobody. These images are the only way to judge a run *while it is still costing money* — make them look like your dataset. `{trigger}` marks where the subject goes. |
| Learning rate | 1e-6 | 1e-7 – 5e-6 | Lower if the model drifts off the base too fast; higher only with evidence. |
| Resolution | 1024 px | 768 or 1024 | 768 trains faster and cheaper, at lower fidelity. |
| Checkpoint every / keep | 250 steps / keep 1 | ≥ 100 steps / keep 1-3 | More kept checkpoints means more sweet-spot candidates — and each one is about 26 GB of PRIVATE Hugging Face storage. The panel states the total before you launch; the launch itself refuses (confirmably) when it plainly will not fit. |
| Images per step | 1 | 1, 2, 4, 8 | Batch size is locked at 1, so by default each step learns from a **single image** — over a set of several thousand, that is a very noisy estimate of the right direction. This averages several images into one update instead. It needs no extra VRAM (the images go through one at a time); it needs TIME. |
| Learning-rate schedule | constant | constant · warmup · cosine | Constant is what shipped. Warming up eases the first steps rather than hitting a 12B model at full rate from step 1. Cosine fades the rate to zero by the last step, which settles fine detail late instead of still shoving the weights around at the end. |
| Warm up over | 100 steps | 10 - 1000 | Only used by the warmup schedule. |
| Noise schedule | linear | linear · sigmoid · weighted | Which noise levels the run trains on. `sigmoid` concentrates on the middle of the range; `weighted` keeps the linear draw but weights the loss on a bell curve. There is no settled consensus for Krea 2 — linear is what every validated run so far used, so it stays the default. |

**"Images per step" is the one setting here that spends money.** Everything else
changes what the run produces at the same price. This one multiplies the run:
4 images per step means about 4× the wall-clock and about 4× the bill on a GPU
rented by the hour. The card prints the multiplier next to the control and turns
it amber above 1, so the number is visible *before* you launch rather than on an
invoice. What it does **not** change: the number of checkpoints, their cadence,
or the Hugging Face storage the run needs — `steps` counts optimizer steps, so
raising this changes how much each step learned from, not how many files land.

**Two settings you may expect, and why they are not offered**

Both exist in AI Toolkit. Both would break *this* model, so the card does not
show them:

- **EMA** (averaging the weights as training goes) keeps a second copy of every
  trained parameter on the GPU, plus a third whenever it saves. On a LoRA that is
  a few hundred megabytes. On a 12B full model it is roughly +26 GB, then +26 GB
  again at the first checkpoint, on top of an unquantized model and its
  gradients — the run would die at its first save. EMA is still available for
  **LoRA** training, where it costs almost nothing.
- **min-SNR weighting** needs a signal-to-noise table that flow-matching models
  like Krea 2 simply do not have. Worse, the trainer's attempt to build that
  table fails *silently* at startup, so the job does not refuse when you launch
  it — it crashes inside the loss computation an hour later, on a pod you are
  paying for. Refusing it up front is the cheaper failure.

The same reasoning removes `shift`-style noise schedules from the full-model
list: the trainer derives their shift from a token count that assumes a field
Krea 2's denoiser names differently, so the value silently comes out four times
too big. A mis-shifted schedule looks like a tuned run and is not one.

> **Which AI Toolkit is this about?** LoRA training uses the AI Toolkit installed
> on *your* machine — it changes whenever you update it. Full-model training is
> cloud-only and uses the AI Toolkit baked into the rented pod's image, which is
> pinned. They are different codebases at different dates. Every statement above
> was checked against the pinned one, and each run now records the image the pod
> actually booted, so a run can say for itself which trainer produced its weights.

### Where a finished run lands, and why in that order

A finished full model is brought **to this computer first** — into the checkpoint
folder (Settings ▸ Storage) — and the pod is destroyed **only** once that file is
proven: its byte count matches what the pod advertised, and its safetensors
header re-reads and declares tensors. Nothing is pushed to Hugging Face *while
the run trains*, which is the whole point: a full private quota used to arrive as
a `403` at step 2750 of 3000 and end a paid run. Once the local copy exists, the
master is uploaded to your private repository as a **backup**, and that upload is
allowed to fail — it costs the ability to *continue* this model later, nothing
more.

Three deliveries, in Settings ▸ Storage ▸ **Full-model delivery**:

| Delivery | What you get | What it costs |
| --- | --- | --- |
| **This computer, then a Hugging Face backup** (default) | The model here, plus a Hub copy that keeps the run resumable. | The Hub copy still needs private storage. |
| **This computer only** | Nothing touches your Hugging Face quota. | The run can **not** be continued later. |
| **Hugging Face only** | The behaviour of runs made before this existed. | A full quota can end the run itself. |

If anything interrupts the download — a cut stream, a full drive, a cancelled
transfer — the run ends as **error_pod_kept** with the machine alive, and the
Runs page offers **Fetch to this computer**, which resumes from the byte it
stopped at. A launch also refuses (confirmably) when the checkpoint drive plainly
has no room for what is coming.

### Continuing a full model

▶ Continue works on a full model: a fresh pod is handed the checkpoint, drops it
into its job folder, and ai-toolkit resumes from the step written in the file —
so a run that stopped at 3000 continues to 4000 instead of paying for the first
3000 again.

The interesting part is **how the 26 GB gets to the pod**, because there are two
roads and they are not interchangeable. The dialog shows both, with numbers:

- **☁ Hugging Face** — the pod downloads the checkpoint itself over a datacenter
  link. Minutes. It needs a Hub copy of the run to exist, and the weights pass
  through a third party on the way.
- **💻 This computer** — the file goes straight up from here. Nothing outside
  your machine is involved, and it costs your upload speed: usually hours.

**The number that actually decides it is neither speed nor privacy — it is the
GPU bill.** The pod is rented and charged from the moment it boots, including
every minute it spends waiting for its checkpoint. Three hours of upload at
$1.40/h is **$4.20 of graphics card computing nothing**. The dialog shows that
figure for each road before you click, alongside the file size and how long it
expects to take.

That estimate is honest about where it comes from. The app times the checkpoints
it pushes to pods, so once you have sent one the forecast says *"measured at N
Mbit/s on your last 3 transfers"*. Until then it says it is an estimate and names
the speed it assumed. (If you already know your uplink, `cloud.uplink_mbps` seeds
it — but a real measurement always wins over a typed one.)

**Dataset uploads deliberately do not count towards that number**, even though
they are also transfers to a pod. A dataset is thousands of small files sent
eight per request, so what it measures is dominated by per-request latency; a
checkpoint is one continuous stream. Mixing them would produce a figure that
describes neither, and it would be used to forecast the faster of the two. The
cost of that choice is stated rather than hidden: ten dataset uploads still leave
this forecast labelled an estimate.

**A long upload is interruptible without being lost.** The file is sent in
slices, and every slice that reached the pod stays there: if the link drops, the
app is closed, or the machine reboots, continuing that run again picks up at the
last whole slice instead of starting over.

When a road is unavailable the dialog says which one and why — a run delivered
to this computer only has no Hub copy to pull, and a run whose local file was
deleted has only the Hub.

**The Hugging Face road is checked, not remembered.** The app records that a
delivery succeeded, but that record is a minute of the past: a repository you
deleted last night still reads "delivered". So opening the dialog asks whether
the repository still answers, and a confirmed deletion closes that road with a
price of nothing rather than an ETA — renting a pod to fetch a file that is not
there would spend money on a download that cannot succeed. A check that could
not be made (offline, no token, an outage) is **not** treated as a deletion: the
road stays open, because refusing your fast road over a dropped Wi-Fi connection
would be worse than the problem being avoided. Keeping the default **"This computer + Hugging Face"**
delivery keeps the fast road open for every future run.

**And the Hub copy is yours to delete — Hugging Face will not tell the app when
you do.** Deleting one to free space is normal; being told weeks later that a
model is "available" when the link answers 404 is not. So the Checkpoints panel
and the Runs page **ask** whether the repository still answers when you open
them, and say which of three things they found: it is still there, it is not
there any more, or the check itself failed (no token, offline, an outage) —
which is never reported as a loss. Until an answer comes back they describe the
delivery in the past tense, dated, rather than claiming the model is there right
now.

A repository confirmed gone stops offering what can no longer work: the dead
links disappear, *Quantize to fp8* is disabled on that card with its reason
(downloading from that repository is the first thing it would do), and ▶ Continue
is disabled **only when neither road is open** — that is, when the repository is
gone *and* this computer no longer holds a full-precision file. With one on the
disk, the 💻 road above is exactly the way out, so the button stays.

Both halves of that are read fresh, which is the whole point: the repository is
asked over the network, and the local file is looked for on the disk every time
the page is drawn. Neither is a note taken when the run finished — deleting
either one by hand shows up immediately.

### The two files a finished run delivers

A dense run produces a ~26 GB **bf16 master**. Nobody generates with a file that
size, so the app quantizes it **on the pod** and delivers a **~10 GB fp8 export**
next to it:

- **the fp8 file is the one to download for ComfyUI.** It is a scaled fp8
  checkpoint (per-tensor `float8_e4m3fn` weights with their scales) and loads
  with the standard *Load Diffusion Model* node, no extra setup;
- **the bf16 master is the only one that can be trained again**, merged, or
  re-quantized differently. fp8 is a lossy, one-way export. *Keep the bf16
  master* is ON by default for exactly that reason — turning it off halves your
  storage and closes that door permanently.

If the export fails, the run is still a success: the master is delivered either
way, and the panel says so rather than reporting a failure. Only the master is
ever backed up to Hugging Face — the fp8 twin is regenerated from it in seconds,
and pushing both would eat the private quota twice as fast.

### Quantizing a model you already have

The same conversion is available by hand, in **⚙️ Full-model recipe → Quantize an
existing model to fp8**: give it the full path to any full-precision
`.safetensors` checkpoint on this machine — a 26 GB model you downloaded from
Hugging Face, a dense checkpoint from an earlier run — and it writes
`<name>_fp8.safetensors` **next to it**. The source is never modified, and an
existing output is never silently overwritten.

- It runs on the **CPU**, not the GPU: the work is an elementwise cast plus one
  reduction per tensor (measured ~1.2 GB/s here, so a 26 GB file is bound by your
  disk, not by arithmetic). Nothing competes with ComfyUI or a training run.
- It runs in a **separate Python** — the one that has `torch` (the app installs
  without it; torch is gigabytes). Whether that environment can actually do the
  work is checked *while the plan is drawn*: one that cannot disables the button
  and names what to install, rather than failing after the click or, worse,
  after the download.
- **The size of the model has no bearing on whether it opens.** It is read one
  tensor at a time. Mapping the whole file used to reserve its entire size
  up front, which is why a big checkpoint could fail with "the paging file is
  too small" on a machine with plenty of free memory and disk.
- One at a time, app-wide, and it checks free space before it reads a byte.
- It **refuses a file that is already quantized** — quantizing twice only loses
  more precision — and refuses a LoRA or adapter, which has nothing large enough
  to shrink.
- When it finishes it **re-opens the file it just wrote** and checks the marker,
  the per-tensor scales and the payload dtype, so a bad conversion is reported
  now rather than at generation time.

> **This is not ai-toolkit's `quantize`.** The `quantize` / memory options in
> Advanced training shrink the model *in memory while it loads*, so a smaller
> card can train something that would not otherwise fit. They write nothing: the
> saved checkpoint is still full precision. This feature produces the **file**.

### ✨ Quantize to fp8 — one click, no path to find

A run delivered before the automatic export existed leaves you with a 26 GB file
in a private repo and no fp8 twin, and until now this block could not help: it
asked for a path on your disk, and that master has none — the dense lane never
downloads it. So the block now aims at the model **your run delivered**, and
does the whole chain with nothing to type: fetch the master, convert it, and
leave the fp8 file in ComfyUI's own models folder, ready to load.

Click **✨ Quantize to fp8** once and it tells you what it is about to do; the
conversion only starts on the second click.

- **Which checkpoint it takes, by name.** A dense repo usually holds the final
  save *and* several ~26 GB step snapshots whose names differ by a number. One
  rule decides — the **final save** wins, and without one the **highest step**
  does — and it is the same rule that stamped the file this card lists, so what
  you read and what runs can never be two different files.
- **Where the file lands, spelled out.** `models/diffusion_models` for a dense
  transformer, `models/checkpoints` for an SDXL-style full checkpoint, honouring
  an `extra_model_paths.yaml` root exactly as a LoRA deploy does. With ComfyUI
  not configured it falls back to the app's own `data/models/…` and **says so** —
  it never pretends to have put the file where ComfyUI looks.
- **What it costs in disk, before it starts.** What is still to download, the
  fp8 file's own ceiling, and 2 GB of working headroom — compared against the
  free space of the volume that *really* holds that folder (a ComfyUI models
  folder is very often a junction onto another drive). Not enough is a refusal
  that writes out every term, and offers to write the file to another folder
  rather than ending there. Whatever this forecast accepts, the conversion does
  not then refuse.
- **It is a real job.** Progress in gigabytes while the master comes down, then
  per-tensor while it converts, a **Stop** button, and resumption from where it
  stopped — stopping keeps what already arrived. The job also survives leaving
  the page: come back and the card shows the same run.
- **Afterwards, the master is kept by default.** It is the only file you can
  train from again, merge, or re-quantize. Deleting it is one radio button away,
  with its size written on it, and it only ever happens *after* the fp8 file has
  been re-opened and verified.
- It refuses a file that is **already quantized**, refuses a LoRA/adapter, and
  **never overwrites** an existing output.

**The path field is still there, as the exception.** A file nothing in the app
points at — a checkpoint someone shared, a model you downloaded yourself — is
typed in as before, and takes the same route: same refusals, same disk check,
same destination, stated. When you have set **Custom weights…**, that path
pre-fills it, so there is nothing to type there either.

### Full-model training on Turbo: allowed, and unmeasured

Dense (full model) training accepts **Raw, Turbo, or a Krea 2 checkpoint of your
own** — the full-model panel has its own base picker for all three. Raw is still
the recommendation, and Turbo now comes with a warning shown before the GPU is
rented rather than a refusal. Here is what that warning is made of.

**What Turbo is.** A speed-distilled build: it draws an image in about 8 steps
instead of ~50, and that compression lives in the very weights a dense run
rewrites. So a dense run on Turbo does eat into the speed.

**What that actually costs, where anyone has measured it.** Not a broken file.
One distilled model has published results for this: **Z-Image-Turbo**. Full
fine-tuning leaves a model that still generates properly and simply **stops being
fast**. Both published sources give the same replacement recipe: give up the
acceleration settings and infer at **~30 steps, CFG ~2** instead of 8 steps and
CFG 1.

- [DiffSynth-Studio ▸ Z-Image](https://github.com/modelscope/DiffSynth-Studio/blob/main/docs/en/Model_Details/Z-Image.md)
  — "Direct training will quickly cause the model to lose its acceleration
  capability", and after it "the effect of inference with 'acceleration
  configuration' becomes worse, while the effect of inference with 'no
  acceleration configuration' becomes better".
- [Training strategies of Z-Image-Turbo](https://huggingface.co/blog/kelseye/training-strategies-of-z-image-turbo)
  (kelseye, 2025-12-16) — "Directly updating the model weights (such as **full
  fine-tuning** or standard LoRA) tends to disrupt the model's pre-trained
  acceleration trajectory". Its Scheme 1, "the most general fine-tuning method",
  degrades "significantly" at 8 steps / CFG 1 and is then run at
  `num_inference_steps=30`, `cfg_scale=2`. It is offered to people "insensitive
  to inference speed", which is the whole point: usable, not broken.

Two things those sources do **not** say, and this page used to:

- **They do not say the erosion is progressive.** DiffSynth says training
  loses the acceleration "quickly"; the paper that studies the problem head-on
  ([D-OPSD, arXiv 2605.05204](https://arxiv.org/abs/2605.05204), 2026-05-06)
  treats it as something fine-tuning "would compromise" outright, and exists to
  *prevent* it rather than to describe it fading. Expect to lose the few-step
  mode, not to watch it drift.
- **They say nothing about FLUX.2 Klein.** This page named it as a second model
  with published results; no source we have does. Black Forest Labs' own
  fine-tuning material for Klein is about LoRA, and points at the undistilled 9B
  Base for post-training — consistent with the advice below, but it is not a
  measurement of what dense training costs a distilled build.

**Nobody has published that measurement for Krea 2 in particular.** Everything
above is carried over from neighbouring models. So the honest word for
dense-on-Turbo here is **untested**, not impossible — and an untested lane is a
thing to warn about, not a thing to forbid. The app says all of the above in the
panel and in the rental dialog, then lets you launch. What it will not do is
pretend: the run trains the base you picked, the configuration sent to the pod
names it, and the run's provenance and model card name it too.

One thing the dense lane deliberately does **not** do on Turbo: load the
de-distillation adapter the LoRA lane loads. Nothing in the save path unmerges
it from dense weights, and a LoRA-shaped subtraction would miss the
normalisation and modulation tensors a dense run moves. Adding it would create
the defect the old refusal feared; leaving it out cannot.

**Krea's own recommendation is train on Raw** — the undistilled checkpoint they
publish for exactly this — then validate on Turbo. musubi-tuner, the other
public trainer with Krea 2 support, recommends the same. Nothing here changes
that advice.

**If you want the speed back afterwards, the published route is a transplant.**
Fine-tune on Raw, then merge the Turbo re-distillation LoRA published in the
`Comfy-Org/Krea-2` repo onto your result; authors report doing this around
strength 0.8-1.0 to get an 8-step build out of a Raw fine-tune. The same trick
exists for neighbouring models (Z-Image-Turbo distill patches, LCM-LoRA).
**We have not tested it ourselves** — treat it as a lead, not a supported
feature.

### Testing a full model: it is a RAW checkpoint

The artifact is **undistilled**. Krea 2 Turbo-style settings — CFG 1 and a
handful of steps — produce a blurry sketch on it, which reads as "the training
failed" when nothing failed at all. Use the same settings the run previewed
with: **CFG ~4 (3.5-5) and 20-30 steps**. The Test Studio now pre-fills those
automatically when the selected base looks like a Raw / full / fp8 checkpoint.

### Which quantized checkpoints can be trained on, and which cannot

**The format decides, not the number of bits.** "Quantized" covers two different
files, and only one of them is a wall:

- a **packed export** — ComfyUI's scaled fp8 and its newer `comfy_quant` form,
  every int8 repack, and the fp8 twin this app itself writes — stores its
  decompression tables as *extra tensors* (`scaled_fp8`, `<layer>.scale_weight`,
  `<layer>.comfy_quant`). A trainer loads a base strictly: those tensors are keys
  it does not know, so **the load fails immediately** — not mid-run, not at the
  first optimizer step. This one is refused, and the message names both the
  obstacle and the way out;
- a **plain fp8 cast** stores the weights in fp8 under the tensor names the
  full-precision file already had, adding nothing. There is no unknown key for the
  strict load to trip on: the trainer up-casts it to bf16 as it loads. This one is
  **allowed**. Several widely used Krea 2 checkpoints — including the Turbo file
  most people already have — are of this kind, and refusing them closed a path
  that works.

Allowed is not recommended. Picking a cast base shows a warning with the actual
numbers (how many of the file's tensors are stored in fp8, and how many
significand bits that leaves against bf16's 8): the precision the cast dropped
does not come back, so the run starts from an already-degraded base and the LoRA
it produces is worse than the same run on the full-precision file, for the same
GPU time. Train on it if that is the file you have — the point is that you know
what it costs, not that you should not.

**What this check does not answer.** It reads how the file is *packed*, not
whether the model family can accept its tensors. A checkpoint can pass here and
still be refused at load for carrying a tensor the architecture does not declare.
Real case, found while building this: a widely circulated fp8 conversion of Krea 2
Turbo carries two extra 6144×6144 tensors under weight-shaped names — its own
metadata describes them as an embedded image, not weights — and a strict load
rejects them. That failure also happens in the first seconds, before any GPU time
is spent, and it comes with the trainer's own message naming the keys.

**The way out of a refusal is a click, not a download.** A full-model run keeps
its bf16 master next to the fp8 twin, and the Checkpoints panel lists that master
by name — pick it there. If the only copy you have is a packed export, the
full-precision version has to come from wherever the model was published; there
is no way back from a packed file, which is why *Keep the bf16 master* is on by
default.

The check reads a few kilobytes of file header — the quantization markers and the
tensor dtypes — so it costs nothing and fires the moment you pick the file, not
an hour into a paid run. A file whose header cannot be read is let through: the
app refuses what it can prove, never what it merely suspects.

## 11. Preview quality — steps and CFG

The preview images a run writes every few hundred steps are the only thing you
can judge it by while it is still running, so they have to be *readable*. How
they are rendered is two numbers — how many **steps** each preview gets, and at
what **guidance (CFG)** — and both live in ⚙️ **Advanced options** under
*Preview quality*, next to the cadence and the prompts.

**Leave them empty and nothing changes.** The boxes show, as a placeholder, the
default your base resolves to; that default follows the model you picked, because
the right answer is a property of the base and not a preference:

| Base | Preview default | Why |
| --- | --- | --- |
| A **distilled** one (Krea 2 Turbo, Z-Image Turbo) | 8 steps, CFG 1 | Distillation is what buys the few-step sampling. Asking for 25 steps at CFG 4 wastes minutes per preview and does not look better. |
| An **undistilled** one (Krea 2 Raw, Z-Image, FLUX, SDXL) | 20-35 steps, CFG 4-6 | At a distilled model's 8 steps these come back as unfinished sketches — muddy, half-formed — and you cannot tell a bad run from a bad preview. |

You need the boxes when you train on a base the studio does not ship — a merge of
your own, a converted checkpoint — because then the default is a guess about a
model nobody measured. Symptoms worth acting on: previews that look like
sketches (raise the steps), or a preview that visibly costs more time than the
training it interrupts (lower them).

These are **preview settings only**: they change the picture, never the weights.
That is also why a **▶ Continue** can change them even in *full training state*
mode, where the cadence and the learning rate are locked — a resume is exactly
when you have already seen the previews and know they are unreadable.

*Suggested by charlesangus (GitHub #46).*

---

*Everything above is enforced or surfaced by the app itself (pre-flight checks,
leak badge, composition bar, coverage panel, advanced options). This page just
explains why.*

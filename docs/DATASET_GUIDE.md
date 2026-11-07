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

### Qwen-Image 2.1 training

Select **Qwen-Image 2.1** in the dataset's model-family selector to train a
LoRA from RGB images and prose captions. For local training, update AI Toolkit and install its current
requirements in its own Python environment first; the dedicated `qwen_image_2`
architecture is required. The older Qwen Image architecture is incompatible.

The recipe uses `Comfy-Org/Qwen-Image-2.1`, the official Qwen repository for
configuration and auxiliary components, convrot8 quantization, shifted flow
matching, rank 32, and 40-step previews at CFG 3. Resolution defaults to 768/1024;
GPU memory requirements have not been measured in LDS. The text encoder stays
loaded, matching AI Toolkit's model defaults. Checkpoints have their own run
suffix and deploy to `loras/qwenimage21`.

For a **Character** dataset, **Advanced options → Presets → Built-in (community
starting points)** offers **Qwen-Image 2.1 · Character LoKr Multires**. It applies
full-rank LoKr factor 4, Automagic3 at `1e-4`, weight decay `2e-4`, sigmoid/Balanced,
batch 1, cached captions and 512/768/1024 resolutions. The
[original report](https://www.reddit.com/r/StableDiffusion/comments/1worox0/my_first_qwen_21_character_loras/)
used 30–70 images; the
[author's follow-up](https://www.reddit.com/r/StableDiffusion/comments/1x0jmuw/my_workflow_for_realism_with_qwen_21/)
confirms full-rank factor 4 and reports convergence around 2,000–3,000 steps.
LDS supplies a 50-steps/image target clamped to that range, with previews and
checkpoints every 250 steps. Quantization and preview sampling retain the LDS
defaults above. This is a reported starting point: the published examples also
used extra inference LoRAs, and another user reported incomplete likeness with
the same training settings. Compare your own checkpoints. The preset is scoped
to Character datasets and does not appear for Style or Concept.

Cloud Training uses a dedicated September 23 AI Toolkit image, at least 32 GB
VRAM on an Ampere-or-newer GPU and 100 GB disk. Older saved cloud settings cannot
select a smaller machine. Live hourly prices are shown; duration and total-cost
estimates are unavailable until this family has a measured speed model.

Reference-image editing datasets and transparent RGBA training datasets are
not available for this family. Trained LoRAs can generate images and compare
checkpoints in **Test Image / Studio**. Prepare the matching Qwen-Image 2.1
base, Qwen3-VL 8B encoder and dedicated 2.1 VAE in **Settings → Image engines →
Trained image models**. This requires a current ComfyUI with native Qwen 2.1
support; older Qwen Image/Edit assets are different models.

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
**768** to fit smaller cards). **Local training only for now**; generated images
and checkpoint comparisons are available in Test Studio using a compatible
FLUX.1 base, T5 encoder, CLIP-L and VAE.

**FLUX.2 Klein note:** two model sizes, picked next to the base selector — **4B**
(default) trains on a **16–24 GB** local GPU, **9B** needs **32–48 GB VRAM** and is
best trained via **☁️ Train in cloud** (both local and cloud runs are supported for
this family). Both bases are *gated* on Hugging Face: accept the license of
`FLUX.2-klein-base-4B` / `-9B` and set a HF token before the first run. In-app
testing and checkpoint comparisons are available in Test Studio with matching
Klein generation models.

**Anima note (the one family that takes BOTH caption styles):** Anima is an anime
model with **hybrid prompting** — its model card documents *booru tags* and *natural
language* as equally supported, which its LLM text encoder is what makes possible. So
this is the family where the "match the style" rule below does **not** apply: caption
in prose, caption in booru tags, or keep an existing dataset as it is — the app will
not flag either as a mismatch, and you never have to force the launch. Prose is only
the preselected default. It trains on the open `Anima-Base-v1.0-Diffusers` (no gated
download) and is **local-only** for training. Its LoRAs can generate images and
compare checkpoints in Test Studio. Prepare Anima's base, Qwen3 0.6B encoder and
VAE under **Settings → Image engines → Trained image models**.

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

**Trying before you commit.** 🧪 **Caption Lab** in the **Captions** section runs up
to four caption configurations — engine, vision model, vocabulary register and length —
on one image you pick, and lays the results side by side with the caption already
stored. Nothing is written until you keep one, so an engine or a register can be
settled on a single image instead of on a pass over the whole set. The same bench is
a tab of the per-image caption editor in **Images**, on whichever tile you opened — and
of an image bank's 🏷️ Caption window, so a captioner can be settled on the bank before
anything is promoted.

On a dataset, the Lab starts with its saved engine, model, vocabulary, length and
extra instructions. It uses the dataset's character, style or concept base prompt,
including body fidelity and appearance rules. The format follows the training family:
booru for SDXL, prose otherwise. Expand **Prompt sent** under a result to read the
composed instructions used for that candidate. Editing a candidate clears its old
result; **Make default** saves its options, including extra instructions.

Concept previews show the first captioning call; the batch also refines captions
and removes concept terms afterwards. The Lab marks this distinction beside the
prompt. Bank previews keep their descriptive prompt, including visible identity.

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

**Full-model datasets.** Full-model training runs in the cloud only, so the
readiness panel checks such a dataset the way its cloud launch will: it reads
**Ready to launch in the cloud** when nothing blocks the run, instead of a red
"Not ready" whose only blocker was that this computer cannot run it.

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
because both screens show the same prompt history. The **🌐 Civitai** browser
feeds the same batch (a ☐ Batch box on every prompt-bearing card), so a run can
mix your own saved prompts with prompts borrowed from Civitai's top images.

### Compare LoRAs — or blend them

Check two or more LoRAs and Studio asks what you want to do with them:

- **⚖ Compare** (the default) tests each LoRA **on its own**, one column per LoRA,
  swept across the strengths you picked. This is what you want to answer "which of
  these is better".
- **🧬 Blend** loads them **together in the same image**, each at its own weight,
  and — while the **Trigger word** box next to the prompt is ticked — injects
  **every trigger word** into the prompt for you. This is what you want to answer
  "do these two work together" — a character plus a style, or a character plus a
  concept.

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
injects the trigger itself at generation time — while the **Trigger word** box is
ticked, see below).

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

### Send the prompt as written — the Trigger word box

Studio normally prefixes the dataset's trigger word to whatever you type, at
generation time — that is what activates the LoRA, and it is why you never have
to type the trigger yourself. The **Trigger word** box next to the prompt (on
the Test Studio, the Compare page and the Canvas run panel — one shared,
remembered preference) makes that explicit and optional:

- **Ticked** (the default) — the historical behaviour, unchanged.
- **Unticked** — the prompt is sent **exactly as written**. Useful when a render
  keeps typing the trigger back into the image (a speech bubble or a sign asked
  to "say" something will happily spell out the first token it finds), or for
  pure style and scene tests where the token only adds noise.

Images generated with the box unticked say **"no trigger"** in their details, so
two runs of the same prompt never look inexplicably different later. One honest
limit: with the box unticked and an **empty** prompt, the default test prompt is
used without the trigger — which usually means the LoRA's subject will not
appear; type a prompt when testing without the trigger.

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

**☐ Batch** on a card adds its prompt to the batch instead — one more pass of
the next run, the field untouched — and the browser stays open so you can tick
several before pressing **Done**. The count shows under the prompt field (and
on the 🌐 button); the next **Run test** replays every ticked Civitai prompt
alongside the saved prompts you ticked, one image set per prompt, same
checkpoints, same settings, same seed. A prompt ticked in both places counts
once. After the run the Civitai prompts are in your saved prompts like any
other.

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
  locally. A lane you can't use right now — no [vast.ai](https://cloud.vast.ai/?ref_id=683073) key, no ai-toolkit, a
  training already running here, a cloud limit reached — is disabled **with the
  reason**, never hidden. The same choice is offered by the **Runs** page's
  ▶ Continue, where the cloud reason is counted against *that run's* dataset —
  the page lists runs from all of them.
- **Cloud GPU** — when continuing in the cloud, choose an available GPU model.
  The list shows VRAM and live hourly prices under your configured price cap.
  Use **Refresh offers** if availability changes. This choice applies to the
  continuation, including when the original run used another card.

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

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
| **Training base** | Z-Image-Turbo (or a converted custom merge) | Your ComfyUI checkpoint (e.g. bigLove) | Krea-2-Raw (default) or Turbo | FLUX.1-dev (gated HF) | FLUX.2-klein-base 4B (default) or 9B (gated HF) |
| **Preview quality** | Fast, distilled | Depends on checkpoint | Raw: slow but faithful | High, ~20 steps | Non-distilled, real CFG (~25 steps) |
| **Best for** | Fast iteration, prose-driven prompting | Booru-native checkpoints, NSFW ecosystems | Highest realism ceiling | The largest LoRA ecosystem, strong prompt fidelity | Modern FLUX.2 stack; 4B trains on mid-range GPUs |

**Krea note:** the default trains on **Krea-2-Raw** — the official recommendation is
*"train on Raw, validate on Turbo"*. Raw runs are long (hours); that's normal, not stuck.

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

---

## 2. How many images, and which ones

- **Target ~25 images** for a balanced character LoRA. More isn't automatically
  better — 25 varied images beat 60 near-duplicates every time.
- **Balance the framing.** The app tracks four buckets: **face / bust / body / back**.
  A dataset that is 100% face close-ups produces a LoRA that falls apart on
  full-body prompts — it has never seen the body.
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
2. **Never mention hair, face, eyes or skin.** The app's *identity-leak* check
   flags captions that do — fix every flagged one before training.
3. **Describe scene, outfit, pose, lighting, framing.** Those are the things you
   want to stay promptable *independently* of the identity.
4. **Vary the captions.** Identical captions across images teach nothing;
   captions under ~8 words are too weak to isolate the identity.
5. **Match the style to the family.** Prose for Z-Image and Krea; booru tags for
   SDXL booru-native checkpoints. The app blocks a mismatch for a reason —
   a prose-captioned SDXL LoRA produces disjointed images.

**Concept datasets** (training a *thing/style/act*, not a person) invert the rule:
describe everything **except the concept** — the concept is what must bind to the
trigger. Keep masked training **off** for concepts (a person mask would erase the
very thing you're training).

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
| **Masked training** | ON | ON | ON | ON | ON | Background weighs only 10% of the loss → identity binds to the person, not the room. OFF for concepts. |

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
- [ ] **Zero identity leaks** (no hair/face/skin words — the leak badge shows 0)
- [ ] Captions varied, ≥ 8 words, style matches the family (prose vs booru)
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
  things: the **checkpoint/preview cadence**, the **preview prompts** (test images
  only — never the weights), and the **timestep weighting**. Everything structural
  (rank, base model, optimizer) is locked to the checkpoint you're continuing.
  The timestep knob enables a known **two-phase recipe**: train balanced first,
  then continue with a low-noise-leaning emphasis to polish fine texture.

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

---

*Everything above is enforced or surfaced by the app itself (pre-flight checks,
leak badge, composition bar, advanced options). This page just explains why.*

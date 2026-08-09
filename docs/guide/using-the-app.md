# Using the app

The workspace is a **guided flow**: each stage stays folded until the one
before it is done, and the progress rail on the left tells you where you are
and what's blocking the next step. You never have to guess what comes next —
this chapter just explains what each stage does and where the useful buttons
hide.

The walkthrough below follows a **character** dataset end to end because it
exercises the most stages; **concept**, **style** and the **image bank** each
get their own section after it. The flow is the same for all of them — only the
captioning rules and a few guards change with the dataset kind.

---

## The character walkthrough (reference photo → trained LoRA)

1. **Create the dataset** — Datasets → New. Pick **Character**, name it, set a
   **trigger word** (the token your prompts will use), and choose the **target
   model** (Z-Image / SDXL / Krea 2 / FLUX.1 / FLUX.2 Klein — changes the caption
   style; you can change it later).
2. **Upload the reference photo.** The app head-crops it automatically; use the
   crop editor (or *Reset to auto*) if the framing is off. Up to 3 extra angles
   can be added for multi-view consistency. **✦ Edit** retouches the reference
   itself from a prompt ("plain studio-grey background", "add glasses") through
   **any engine that generates** — and shows you a Before/After to Keep or
   Discard. **Klein** and **Krea 2 Edit** run it on your own ComfyUI: free,
   private, and the sane choice when you expect to retry the prompt a few times;
   they appear only when your ComfyUI can actually run them, and say what is
   missing when it nearly can. **Nano Banana Pro, ChatGPT and OpenRouter** charge
   per edit, on the model set in *Settings › Image engines*. The engines differ in
   what else they look at: the API ones take the dataset's extra references *and*
   images you drop into the dialog, Klein takes the dataset's extra references,
   and Krea takes none of them — it reads **one image you add in the dialog
   itself**, because its spare slot was trained for a *different* subject
   (another person, or a scene to place yours in) rather than another angle of
   the same face. So: extra angles go on the reference card and strengthen
   identity everywhere; Krea's second image goes in the dialog and composes that
   one edit. The dialog says all of this before you press Generate.
3. **Generate variations** — pick an engine (Nano Banana Pro, ChatGPT,
   OpenRouter, or local Klein) and fire the **variation catalog**: 53 shots
   across expression, angle, lighting, framing, outfit and background, each
   wrapped in an identity guard so the face stays the same person.
4. **Import** your own photos too (drag & drop) — each is auto-cropped to the
   face on the way in.
5. **Auto-classify framing.** A local vision model tags every image
   **face / bust / body / back**; the badges feed the composition meter.
6. **Curate** — keep / reject / crop, guided by the live meter targeting
   **12 face · 6 bust · 6 body · 1 back**. Watch the face-similarity badges
   (green = strong match, orange = review) to drop off-identity shots before
   they poison training.
7. **Caption** — one click captions the kept set (prose or booru tags,
   matched to the target model). The **identity-leak check** flags any caption
   that describes hair/face/skin — fix every flagged one. A find/replace +
   tag-frequency panel sweeps the whole set at once; its **💾 Write .txt
   files** button drops a kohya-style `<image>.txt` next to each kept image
   in the dataset folder (same format as the export ZIP) for external tools.
8. **Fix individual shots** — every generated tile has a ✏️ button: edit the
   exact prompt that made it and regenerate in place, without losing the rest.
9. **Train** — the pre-flight check runs the full checklist (count, balance,
   captions, leaks, duplicates). It no longer *blocks*: leaking captions and
   near-duplicates are editable right inside the confirm, and missing captions
   just ask you to **Start anyway** (captions stay strongly recommended). Steps
   are computed automatically; ⚙️ Advanced options exposes every knob (each with
   its own why/how) and a **Presets** row — apply a shipped ★ recipe (*Krea
   character*, *Concept*, *Style*) or save/import/export your own as a JSON.
   No GPU? **☁️ Train in cloud** rents one per run. Watch this run — and every
   other, cloud or local — from the **🏋️ Runs** tab, where you can retry a
   failed run (↻), continue a finished cloud run for more steps (▶), and download
   the LoRA.
10. **Pick the best checkpoint** — open the **Test Studio** from the dataset:
    grid-test checkpoint × strength, vote, rank by face similarity, and star ★
    the winning settings. The last checkpoint is almost never the best one.
11. **Export** — at any point, **Export ZIP** gives you the curated, captioned
    set as a standard ai-toolkit dataset. Nothing is locked in.

## Compare settings with two runs at once

A dataset that is already training in the cloud can launch a second run of the
same family — change the toolkit settings, press Launch again, and confirm the
dialog (each run rents its own pod and bills separately; the fleet ceiling in
Settings › Training still applies). The Training panel then shows one chip per
active run — click a chip to follow that run's progress, samples and
checkpoint. If the dataset was edited between the two launches, the chips carry
a ⚠: the runs are no longer training on the same data, so the comparison is no
longer only about settings. If you launch right after the first run started,
the request can take up to two minutes while it waits its turn — if the first
run's export is still running when that time is up you'll get a busy error
and can retry. Once the second run is underway, its own progress line may say
**Waiting for the dataset — another run is exporting…** while it waits (up to
an hour) for the first export to finish.

## Retry a reference edit

After an **✦ Edit** candidate appears, **Retry** repeats the exact prompt, selected
engine and temporary reference files used for that candidate. Use **Try another
prompt** only when you want to change the instruction. The candidate also names the
engine/API that actually returned it, so you can see which service produced the
image before you Keep or Discard it.

## Test a run straight from Runs

The **🏋️ Runs** hub is also a shortcut back to the right **Test Studio**. Every
active or recent run that still has a dataset shows **🧪 Test in Studio** beside
its actions. Click it to open Studio with that run’s dataset already selected —
there is no need to return to the library and find the dataset first. The button
is also available on a folded Recent dataset group, so you can start comparing
checkpoints without expanding its run history.

## Using a full model you trained

Training the **whole model** (rather than a LoRA adapter) produces something
different from a checkpoint, and **📦 Checkpoints & LoRAs** lists it in its own
**🧱 Full models** block for exactly that reason.

A delivered run leaves up to two files, and they are not interchangeable:

- the **full-precision master** (~26 GB). This is the only file you can train
  again or resume from. It is **never** sent to ComfyUI — 26 GB of a model folder
  to do a job the smaller file does better;
- the **fp8 twin** (~13 GB). This is the inference format: the file ComfyUI loads
  with **Load Diffusion Model**.

If the run has a master but no twin, **✨ Quantize to fp8** makes one. It works
whether the master is on this computer or only in the run's private Hugging Face
repository — in the second case it is downloaded first, with progress, and the
transfer can be stopped and resumed. Once the twin exists, **→ Send to ComfyUI**
puts it where ComfyUI looks. On the same drive that is a hard link: instant, and
it costs no extra disk space.

**🗑 Trash** moves one of those files to the app trash, so a mis-click on a file
that cost hours of GPU is recoverable.

**A run whose model is only on Hugging Face is not a lost run.** It shows
**☁ on Hugging Face** on the board and in its card, and the app refuses to remove
it: doing so would discard the only record of where that model is.

### Testing a full model

Once the fp8 twin is in ComfyUI, the **Test Studio** lists it as a base and
**🧪 Test in Studio** opens straight onto it, with its own sample settings filled
in. That matters: a full model trained here is **undistilled**, so it wants a
real CFG and a real step count (CFG 4 / 25 steps for Krea 2). The family's
few-step Turbo defaults render a blurry sketch on it that reads as a failed
training.

One limit worth knowing before you go looking for a button that is not there:
**the Test Studio is entered through a LoRA of the dataset.** A dataset trained
only as a full model has none, so it cannot open the Studio at all. If you have
any LoRA of that dataset deployed, pick it and set its **strength to 0** — no
LoRA node is added at 0, so you generate with the bare model.

## Merge a LoRA into a base checkpoint

This is the step between *"I trained a LoRA"* and *"I have a model to publish"*,
and it is how most of the community checkpoints you can download were actually
made. Of the Krea 2 checkpoints whose authors describe their method, the ones
that explain themselves describe a **merge**, not a training run: train a LoRA on
Raw, fold it into a base, quantize, upload.

You will find it in **📦 Checkpoints & LoRAs**, as **🧬 Merge a LoRA into a base
checkpoint**. Inside a full model's card the same tool appears with that model
already filled in as the base.

**Say what you are merging.** Pick a full-precision base, then add one or more
LoRAs, each with a weight. `1.0` applies a LoRA exactly as it was trained;
lower blends it in more gently; a negative weight subtracts it. Several LoRAs
stack — that is what "baked in LoRAs with balanced weights" means when you read
it on a model page.

**Nothing starts on the first click.** The plan is computed from the file headers
alone — no weight is read — and it tells you how many tensors change, exactly how
big the output is, which drive it lands on, roughly how long it takes, and what
happens if it fails. On a 26 GB Krea 2 base, a measured merge took **about two
minutes** and rewrote 256 of 430 tensors.

**Nothing is ever overwritten.** The result is written next to the base under a
new timestamped name, through a temporary file that is only renamed once the
merge finishes. A merge that fails, or that you stop, leaves the base, the LoRAs
and any earlier merge exactly as they were.

### It is a merged model, and it says so

The file's own metadata records that it came from a merge, which base it used,
which LoRAs at which weights, and when. That matters because **file names lie**,
and because on the model sites "finetune" is routinely used for exactly this
object — by authors who describe the merge themselves a sentence later. LDS does
not copy that vocabulary: what comes out of here is a base with LoRAs folded into
its weights, not a model that was trained as a whole, and the header keeps saying
so after the file is renamed or re-uploaded.

### Getting the speed back (the Turbo transplant)

A full-model run in this app targets **Raw**, which is undistilled and therefore
slow. Krea publishes a re-distillation LoRA for Turbo; merging it at **0.8-1.0**
into a model trained on Raw is the published route people use to get few-step
behaviour back, and it is how the same model ends up on the model sites in both a
Raw and a Turbo flavour.

**We have not tested this ourselves.** It is an approximation, not an identity —
generate a few comparisons before you publish anything on the strength of it.

### Merge first, quantize after

Merging into an **already quantized** file is refused, on purpose. It would
dequantize every weight, modify it and re-quantize it: lossy on the way in and
again on the way out, and the loss compounds each time somebody does it. Merge
into the full-precision (bf16) model, then quantize the merged result with the
fp8 tool — which is the order the refusal points you at.

### Two things it will tell you about, rather than hide

- **A LoRA that does not belong to the base** is refused before anything is
  written, naming the weights it expected to find. A LoRA trained for another
  model has nothing to merge into.
- **Tensors that are not part of the model** are reported, not dropped. Not every
  `.safetensors` contains only a model: one community Krea 2 file circulating
  today carries about 75 MB of an image in two tensors hiding under a legitimate
  name. Nothing we do not understand is modified — it is copied through, and the
  plan names it so you know it is there.

**What the merge needs:** the same Python that quantization uses — one with
`torch` available. If it is missing, the plan says so with the command to fix it,
before you click anything.

## Recover a paused Test Studio batch

If ComfyUI drops while Test Studio is processing a batch, the affected tile says
**paused** and shows its paste-safe reason. The queue deliberately stops there:
it does **not** submit or start a later job, so nothing else runs against a
recovered or different ComfyUI state.

First recover or restart ComfyUI. For a valid local portable install,
**Setup → ComfyUI → ▶ Start ComfyUI** uses the app's fixed local-safe profile.
It does not read, change or execute any `.bat` file; your existing launcher and
its settings stay untouched. Once ComfyUI is responding, **Cancel** the paused
batch and resume it from Studio. That makes the next prompt an explicit choice,
never an automatic continuation.

## Concept datasets (an object or action, not a person)

Pick **Concept** at creation and describe the concept in the required field —
the captioner needs to know exactly *what to omit*. What changes vs character:

- **No reference photo.** Images come from **import** or the built-in
  **scraper** (paste a gallery URL or run a Reddit keyword search, tick the
  frames you want, they land straight in the dataset — deduplicated and
  quality-filtered). Already have a kohya-style dataset on disk (images +
  same-name `.txt` captions)? **⋯ More → 📂 Import from folder…** merges it in
  from a pasted folder path — captions attach, duplicates are skipped (a ZIP
  works too, via **📦 Import dataset**). On gallery sites (PornPics), a category/tag/search scan
  shows **the same previews the listing page does** — one per gallery, the shot
  that actually matches your keyword. Tick **Scan full albums** to pull every
  photo of each matched gallery instead, or paste a single `/galleries/…` URL
  to get that whole album. Sex.com works the same way for keyword searches
  (`sex.com/en/pics?search=…`) — every pin **is** a single matching image, so
  there is no album option to worry about. Civitai searches return **SFW
  results only** unless you add a Civitai API key in **Settings → Scraping &
  sources**.

  > **Reddit says "wait N seconds" (429)?** By default Reddit scans share a
  > public client id (and its ~1000 requests / 10 min quota) with many other
  > people, so it can be exhausted before your first scan. Add your own free
  > client ID in **Settings → Scraping & sources** — a one-minute, step-by-step
  > guide is built into that page.
- **Captions invert**: they describe everything *except* the concept, so the
  concept is what binds to the trigger. The leak check watches for stray
  descriptions of it.
- **Person masking is off** (a person mask would erase the very thing you're
  teaching), and imports keep the full frame instead of head-cropping.
- **You can mask the faces instead** — the opposite polarity. *Advanced training
  options ▸ Mask faces* weighs the detected faces down in the loss so the concept
  learns the act, not the people demonstrating it, and you can preview exactly what
  it would cover before training. Off by default. See the dataset guide, §8.

## Style datasets (a global aesthetic)

Pick **Style** at creation. What changes:

- **No trigger word** — the style tints every image once the LoRA is loaded.
- **Captions describe content only** (never the rendering), and they're
  optional; caption dropout rises so the style generalizes.
- **Step count switches to a sublinear √n scale** built for the large sets
  (hundreds of images) style LoRAs want.

## Caption your images in another tool

You are not locked into the captioners shipped here. The round trip is:

1. **⬇ Export ZIP** from *Import & export*. The archive is a plain kohya layout —
   one folder of `image.png` + same-name `image.txt` pairs. If some kept images
   have no caption yet, the app asks before exporting instead of refusing:
   confirm and their `.txt` files come out empty, ready to be filled.
2. **Caption them wherever you like.** Any tool that writes a `<image>.txt`
   sidecar next to each image works — that is the convention this app reads,
   whatever the file names are and whatever folder depth you use.
3. **📦 Import dataset (ZIP)** (or **📂 Import from folder…**) with the same
   images and their new `.txt` files. Images already in the dataset are **not
   duplicated**: their caption lands on the row that already holds them, and the
   toast says how many were applied.

Two things worth knowing before you start:

- **A caption you already wrote here is never overwritten.** Re-importing only
  fills the empty ones; the toast reports the rest as *"kept the caption written
  here"*. Clear a caption in the app first if you want the external one to win.
- **Only the caption travels back.** Statuses, scores and framing stay as they
  are here — the returning archive is read as captions for images you already
  have, not as a replacement dataset.

**A Style dataset asks louder, on purpose.** A Style LoRA learns everything its
captions do *not* name, so an empty `.txt` teaches it nothing; the export
confirmation says so before letting you through. Cancelling takes you straight
to the captions instead.

*Requested by Qeeyana (Reddit).*

## Krea and the shape of your reference photo

**Krea 2 Edit now follows the framing of the selected shot card** during dataset
generation. The reference photo still anchors identity, but Krea's v1.2 Fit path
adapts it to the requested output: **1:1** for face cards and **3:4** for bust,
body and back cards. A square reference therefore no longer forces a full-body or
sitting card into a tight bust crop.

This is deliberately limited to Krea dataset variations. The separate **Edit
reference** action keeps the source layout for a free-form edit, while Klein and
the API engines keep their existing, separate generation paths.

You can still crop a reference when you want a different identity anchor or
composition, but you no longer need to crop it merely to give a selected body
card enough vertical room. Reference quality still matters for likeness; the
selected card now owns the output frame.

## Your own shot catalog (JSON import)

The workspace ships a built-in shot catalog per subject type (53 shots for a
human, ~59 for an animal, 55 for an anime character, and so on). If you want shots nobody wrote for you —
40 breed-specific poses for a dog, a product line's signature angles — you don't
have to type them one at a time. Open **📥 Shot catalog (JSON)** under the shot
grid.

**Export first.** The exported file is the format, and the example an LLM needs:

```json
{
  "format": "lds-shots/1",
  "subject_type": "animal",
  "shots": [
    {
      "label": "Dog, zoomies on the lawn",
      "framing": "body",
      "prompt": "full body photo of the animal running fast across a lawn, side view, sunny day"
    }
  ],
  "examples": []
}
```

Then ask a chat assistant for more shots *in that exact shape*, and import the
file it gives you.

Each shot needs three things:

- **`label`** — a short name, max 80 characters, shown on the card. It must be
  unique: not a built-in label (of *any* subject type), and not one of your
  existing shots. The app refuses a collision and tells you which label is at
  fault — two shots sharing a label would make it resolve the wrong prompt the
  day you regenerate one.
- **`framing`** — exactly one of `face`, `bust`, `body`, `back`. Anything else is
  refused; it is never quietly remapped.
- **`prompt`** — the text sent to the image engine, max 500 characters.

`nsfw: true` is optional and only has an effect when Klein is the only engine
checked. Everything under **`examples`** is ignored on import — that's how the
export can show you samples without them coming back as duplicates. Any other
field (including `aspect`) is ignored too, and the import summary says so: an
imported shot uses its framing's default aspect ratio.

**Nothing is written until you confirm.** The app reads the file, lists what
would land and what it refuses (naming the entry and the reason), and waits. A
40-shot file whose 37th entry is broken never leaves 36 shots half-imported.

Imported shots appear in their own **📥 Imported** group after the built-ins, one
set per subject type. They never replace a built-in, you can delete them one by
one or all at once, and they're stored with the app — not in the browser — so
they survive a cache wipe, show up on your phone and ride along in the backup.

### Keeping a shot you wrote by hand

The **✨ Custom shot** box below the grid is the quick way to add one shot: type a
prompt, pick a framing, Add. Those cards are stored **in your browser**, so
clearing its data takes them with it.

Any card you want to keep, press **Keep** on it. It moves into the 📥 Imported
group and is saved with the app, exactly like an imported shot — surviving a
cache wipe, following you to another device, included in the backup. The card
keeps its identity, so a shot preset that had it selected still works. If its
label happens to clash with a built-in shot or with one you already imported, the
app says which label and refuses rather than creating a duplicate; rename the
card (remove it and add it again) and press **Keep** once more.

*Feature requested by ashish.sinha (Discord).*

## Back up everything

The **💾 Back up everything** button on the Datasets library packs your whole
setup into a single file so you can move to a new machine — or recover from one
— without losing anything.

- **What's inside**: every dataset (all images, captions, statuses, face and
  watermark states, references), its **training history** (which runs produced
  which version, the settings each used), plus your **settings** — engine
  choices, training defaults, cloud tuning, watermark preferences. It's a
  *logical* backup, one entry per dataset, not a raw disk dump.
- **Include trained LoRAs** (checkbox next to the button): also bundle the
  trained `.safetensors` files themselves. These are large — hundreds of MB per
  checkpoint — so it's **off by default**; the light training history above is
  always included, so a dataset comes back under **Trained** either way. Tick it
  when you want the finished LoRAs to travel too.
- **What's never inside**: your **API keys, Hugging Face token and scraping
  credentials**. They are deliberately left out so the file is safe to copy
  around; re-enter them once on the new install.
- **How it runs**: in the background. A library can be gigabytes, so you get a
  live "X / N datasets" progress bar and can keep working. When it's done, use
  **⬇ Download** to save the archive, or **📂 Open folder** to find it on disk.
- **Restoring**: hand the master archive to the same **📦 Import backup** button.
  It restores your settings (without overwriting keys you've already entered),
  rebuilds each dataset **and its training history** — so it lands back under
  **Trained** instead of "Not trained yet", with its runs in the Runs hub.
  Bundled LoRA files are re-deployed to ComfyUI when it's configured on the new
  machine; if it isn't, they're reported as skipped and the **Trained** status
  still stands (the run is what marks it trained, not the file on disk). Nothing
  is ever overwritten — a dataset whose name already exists comes back with a
  `(restored)` suffix — and you get an honest final report of what was restored,
  renamed or skipped.

## The image bank (triage a big folder)

You exported 9 000 unsorted images from Telegram (or a scraper dumped a
mountain of files) and a dataset only needs the best 30–150 of them. The
**🗃️ Bank** tab is the triage funnel that gets you there — without ever
touching the folder itself:

1. **Create a bank** — give it a name and paste the folder path. The app
   inventories every image in place (subfolders included). Nothing is copied,
   nothing is modified; rejecting an image is a reversible status, never a file
   deletion. The folder stays LIVE: keep dropping images into it and they are
   picked up automatically the next time you OPEN the bank ("42 new image(s)
   found in the folder"), as undecided images ready for the next scan — your
   existing keep/reject decisions, scores and captions are never touched. The
   bank LIST does not re-check the folders by itself: on a big library that was
   a full inventory of every image on disk each time you walked past the page.
   It tells you how fresh its counts are, and **🔄 Rescan folders** checks them
   all on demand. A folder that went missing (unplugged drive, renamed folder)
   is still flagged from the list without any rescan. Files you removed from the folder are reported at the top
   of the bank, never deleted from it, so an unplugged drive can't wipe your
   triage. One bank holds up to **200,000 images**; past that the refresh adds
   as many as fit and tells you how many it left out, so nothing you already
   triaged stops working. That ceiling counts what is in the folder now — files
   you deleted from it don't count against it.
1bis. **🕸 Scrape the web into a bank** — you don't need a folder you prepared
   by hand. Unfold **🕸 Scrape the web into a bank** on the bank list, choose a
   destination (a **new bank**, or **add to an existing one**), then scan a
   gallery URL and pick images exactly as you would for a dataset. They are
   downloaded into that bank's own folder and inventoried on the spot.

   Two things are worth knowing, because they are the whole point:

   - **Nothing is filtered on the way in.** Scraping straight into a *dataset*
     applies training-grade gates (short side ≥ 768 px, ratio ≤ 3:1, perceptual
     de-duplication) *before* anything is stored. A bank is the step **before**
     that judgement: "too small", "near-duplicate" and "wrong framing" are
     verdicts its own passes produce, with thresholds you move. So the bank
     stores what it downloaded and lets you decide. If you already know what you
     are collecting, scraping straight into a dataset is still the shorter road.
   - **A second scrape resumes the same bank.** Pick *Add to an existing bank*
     and the new images join the pile — nothing is replaced, and no triage
     decision you already made is reset. Re-downloading the exact same file
     lands on the same name instead of piling up copies; that is file identity,
     not a duplicate verdict (the bank's own passes own that word).

   The rest of the funnel is unchanged: scan, cull, promote into a dataset.
2. **🔎 Scan quality** — a background pass (CPU only, a few minutes even on
   thousands of images) scores every file: sharpness, noise, flat/empty
   frames, resolution — and groups **near-duplicates**. The flags follow the
   thresholds in *Settings → Captioning & quality*; because the raw scores are
   stored, tuning a threshold re-sorts the bank instantly, no rescan. The same
   pass also answers two questions the file itself lies about — see
   *Is this image really what it says it is?* below.
3. **Cull** — use the filter chips (🌫 Blurry, 📺 Noisy, ⬜ Flat, 📐 Small,
   🧇 Soft detail, 🎞 Black bars, ≈ Duplicates) to review the worst
   offenders first. **🧹 Auto-reject
   flagged…** clears whole categories in one click (your manual ✓/✕ are never
   flipped). The number beside each checkbox is what *that click* would reject —
   still-undecided images only, which is why it is usually smaller than the
   count on the matching filter chip: the chip shows every image carrying the
   flag, including the ones a previous auto-reject already threw away — and it
   counts them **inside whatever else you have filtered**, so it always states
   the size of the page it opens. (Each chip is measured with your other filters
   applied and its own value lifted, so picking one never blanks its
   neighbours, and a chip stays on offer even when it holds nothing under the
   current filter. The auto-reject number stays whole-bank on purpose: that pass
   runs over the bank, not over the view.) Run it
   twice and the second run legitimately says **0 to reject**: there is nothing
   left it is allowed to touch. A flag also warns when its pass never ran, and
   the panel says how many images have **never been scanned** — those are
   invisible to every quality flag until 🔎 Scan measures them, which is not the
   same thing as being clean. In the Duplicates view, resolve every group at
   once with **keep best** (highest resolution, then sharpest) or **keep
   first**, or pick the keeper by eye.
4. **👥 Group by person** — the face pass (needs the Quality tools from Setup)
   detects the dominant face of every remaining image and clusters the bank by
   person, *no reference photo needed*. Click a person card to see only them,
   select all, keep or reject. Embeddings are cached, so re-running after a
   cull is much faster.
5. **🏷️ Caption & 🔍 search** — caption the bank with the same engines your
   datasets use (JoyCaption / Ollama vision, your *Settings*). Hit **🏷️ Caption
   all** to describe every not-yet-captioned image, or select some first to
   caption just those. It runs in the background, frees the GPU like the other
   passes, and is Stop-able mid-run. The captions are plain descriptions (no
   trigger word, nothing omitted) whose real job is **search**: type into the
   🔍 box — `red dress`, `sunset`, a file name — and the grid filters to
   matching images, combinable with every other filter. It's the fast way to
   find shots in a 9 000-image dump.
6. **⬆ Promote** — the kept images are **copied** into the dataset you choose
   through the normal import path: normalized to webp, near-duplicates already
   in the dataset skipped. Any bank caption **rides along**, so a captioned
   selection starts already captioned in the dataset. From there they get
   everything datasets have — captions, watermark cleaning, face scoring against
   a reference, training.

Work the funnel in that order: quality first (cheap, catches the trash), then
subject, then selection. A promoted image keeps its ⬆ badge in the bank so you
always know what's been used where.

**🎨 Curate down to the right subset.** Culling removes the bad shots; curation
picks the *good* subset — and it's most of what makes a LoRA good. Once **✨
Score** has run (the default CLIP semantic index), or the Bank's optional
**SigLIP 2 semantic index** is ready, the **Curate** row under the selection bar
offers two selectors that cost no extra inference:

- **🎨 Pick diverse** — enter a number and it selects the images that best
  *cover the variety* of what you're looking at (varied angles, outfits, scenes),
  instead of that many near-identical frames. It's the antidote to a dump of
  4 000 shots of the same pose: ask for 60 and you get 60 that actually differ.
  **Skip the odd ones out** (the slider under the number) is why they are the
  *right* 60. "Most varied" is computed as "farthest from everything already
  picked", and the image that is farthest from everything in a collected bank is
  usually not a nice unusual shot of your subject — it's the meme, the screenshot,
  the botched frame, the one photo of somebody else. The slider discounts an image
  for being *alone in the bank*: at the default **50%** an image that resembles
  nothing else has to be far more interesting than a normal one to earn a slot,
  and at **100%** it is all but excluded. It never works the other way round —
  anything as typical as the median of the bank is left completely alone, so this
  cannot turn your 60 into 60 look-alikes. Set it to **0** for the pure-coverage
  behaviour the button had before this setting existed. On a very large bank the
  first click takes a few seconds (it reads every image's neighbourhood once);
  the button says *Sampling…* while it does.
- **⚖️ Balanced pick** — see [Pick a balanced set](#pick-a-balanced-set) below: the
  same sampling, but spread evenly over your **framings** instead of taken off
  the top of one ranking.
- **🎯 Similar to selected** — select **one** image as a reference, and it ranks
  everything by how much it looks like that image and selects the closest N — the
  fast way to pull one person or one look out of a mixed export.

Both honour whatever filter and 🔍 search are active ("the 60 most diverse of
*this* subfolder"), and both just **select** — the images light up and you review
them with the same ✓ Keep / ✕ Reject / ⬆ Promote bar. Nothing is auto-kept or
deleted, so a selection you don't like costs one click to clear.

**📐 Classify framing** tags every non-rejected image by *shot type* — face
close-up, bust, full body or back view — using the same detector the datasets
use. The result becomes a row of **📐 Framing** filter chips (compose with every
other filter and search), so balancing a character set's angles is a couple of
clicks. It's a GPU vision pass; add it to **🚀 Launch all** to have it run
overnight with the rest.

**📊 Coverage advice** (idea by [@antonp](https://github.com/perfectgf/lora-dataset-studio))
is a read-only panel next to the Curate row. From what you've **kept** (or every
non-rejected image before you've kept anything), it says in plain sentences what
leans and what's thin for a good LoRA — *"70% face shots, add body/back"*,
*"person #1 is 60% of the set — one subject or a mix?"*, *"only 8 kept, most
families want 20+"*. It's **advice only** — nothing is kept or rejected — and
pure maths on data the passes already computed, so it costs no GPU. The
framing-balance line needs the 📐 Framing pass to have run; without it the panel
still covers person mix, style spread and resolution and hints to run framing.

Those are all **labels**, and labels have a blind spot: they cannot tell two
hundred near-identical shots from two hundred different ones, and they say
nothing about outfits, lighting or camera angle. Two things you may already have
on disk can, so the panel also reads them when they exist:

- **Visual spread**, from the Bank's selected semantic index. It reports
  the average similarity across the pool — *"91% average similarity — a set this
  repetitive teaches one look"*. The bands were calibrated by measuring real
  banks: an ordinary one sits near 65%, an image plus its nearest neighbours
  lands around 79-90% with CLIP. SigLIP 2 has its own score distribution, so LDS
  shows its measured similarity but deliberately gives it no *varied/alike* band
  until that engine has been calibrated on real Banks. Without the selected index
  it says **Not measured** — never "varied", because nothing looked.
- **Caption variety**, from the captions the 🏷️ pass wrote, read by the same
  lexicon the dataset Coverage panel uses. It reports which camera views,
  lightings, settings, outfits and expressions your captions mention and which
  they never do.

Both limits are on the panel, not just here. The caption read looks at **words,
not pixels**: a profile shot the captioner never called a profile is invisible,
and *"not smiling"* still counts as a smile. A bank has no character/concept/style
kind the way a dataset does, so it is judged as a **character source** — the same
assumption the framing target and the person-mix advice already make.

The advice becomes a gesture with **⚖️ Pick a balanced set…** at the bottom of
the panel — see [Pick a balanced set](#pick-a-balanced-set).

**🗑 Delete rejected from disk** (next to Promote) is the one exception to the
"your source folder is never modified" rule, and it's opt-in. Once you're happy
with your triage, it removes every image you marked ✕ rejected from its source
folder — the actual files, not just the status. It asks you to type **DELETE**
first, and tells you where the files will go before you confirm: your OS trash
when [`send2trash`](https://pypi.org/project/Send2Trash/) is installed, the
app's own Trash otherwise (recoverable until you empty it from Settings), and a
permanent delete only when neither can take the file. Kept and undecided images
are never touched, and a file it can't remove (locked, read-only) is reported
and left alone rather than aborting the batch.

It runs as a normal bank pass: the confirmation closes straight away and the
progress bar at the top of the bank counts the files as they go, with a **Stop**
that takes effect between files. Stopping is safe — whatever already left the
disk has left the bank too, and the rest are still marked ✕ for a second run.

⚠️ A bank doesn't own its folder, so two banks can point at nested folders and
list the **same files**. That's harmless while you triage — decisions live on
the bank — but deleting from disk in one bank removes those files from the other
too, along with every decision you made on them there. The app says so when you
create such a bank, and the confirmation names the other bank and how many of
its files are about to disappear.

**🚀 Launch all** does the whole funnel for you in one go. Tick which passes
run and how auto-reject behaves, hit Go, and walk away — it chains *scan →
auto-reject → score → find watermarks → group by person → classify framing →
(optional) caption* in that exact order. Two things make it safe to run overnight: a pass whose
tool isn't installed, or a moment when the GPU is busy with a training run, is
**skipped with a reason** instead of failing the whole run; and because
auto-reject runs *before* the heavy passes, scoring/watermarks/person only ever
process the survivors, never the images you just rejected. Captioning is the one
pass left **off by default** (it's the slowest GPU pass and a clean-up run
rarely needs a description on every shot). Stop it any time — and when you come
back, a saved report at the top of the bank tells you exactly what ran, what was
skipped and why, with the headline counts.

## Choosing where a bank pass runs

Every pass button in the bank ends in `…` and opens a **launch window** before
anything runs. The window is not a settings panel — it says three separate
things, and keeping them apart is the point.

**This run — where it applies, and how big that is.** Five lines, and each one
quotes the number of images *that pass* would actually walk:

| Line | What it means |
|---|---|
| Kept + undecided | What every pass has always run on. The default; picking it sends exactly the request the app sent before this window existed. |
| ✓ Kept only | The images you already decided to keep. |
| Undecided only | The ones you have not ruled on. |
| ✕ Unkept only (the bin) | Images you rejected. Nothing is deleted or un-rejected — but the run spends its time on shots you set aside, and the window says what that costs for this particular pass. |
| All three, the bin included | Everything. |

If you have images **selected**, that becomes the first line and wins by
default — the pass runs on your selection, narrowed by what it still has to do.
It says *"up to N"*, never a bare N, because the server intersects your selection
with the pass's own pool and the run can only ever be shorter.

Under those lines sits the **"do it again"** tick: *also re-measure images that
were already scanned*, *throw the cached embeddings away*, and so on. This is
where the old **Rescan all** and **Rescore all** buttons went. They were never
separate passes — they were this scope, wearing a button's clothes — so they now
sit next to the pool they re-run, unticked, with their price written next to
them.

**Settings this pass reads.** Only what the *calculation* consumes, with where
each value lives. 🔎 Scan quality, for instance, reads exactly one of the twelve
🎚 filter thresholds (`dup_distance`), and it reads it for the duplicate grouping
at the end — not for the measuring.

**Not decided here.** The knobs that only change how the grid is **sorted and
flagged**. Those re-apply the moment you save them, with no pass at all. The
sharpness, noise and aesthetic thresholds live here: nudging one costs you
nothing.

Three passes **refuse a partial scope**, and the window shows the option greyed
out with the reason rather than hiding it: **✨ Score**, **👥 Group by person**
and **✂ Find crops & variants** each produce one numbering of the *whole* bank,
recomputed from scratch on every run. Handed a slice, they would number that
slice from 1 and land those ids on top of unrelated groups already saved.

Two things the scope does **not** cover, stated in the windows that need it:
🔎 Scan's duplicate grouping always covers the whole bank (it works from stored
hashes and renumbers them together), and 🎨 Classify medium also runs chained
inside ✨ Score with the default scope.

A run with **nothing to do** is refused before it starts, with the reason and a
suggestion — not launched and then reported as a success.

**The two watermark cleaning levels take the same scope**, and they are the two
where it matters most: ✂ **Auto-crop** and 🧽 **Repaint** are the only actions on
this page that produce a new image file. Their windows list the same five lines,
with one difference — their pool is not a pile but *the flagged images carrying a
usable mark*, so a scope narrows that set and can never widen it. The count on
each line is the pool the level **walks**; ✂ then crops only the marks that sit
in a border band, which is the narrower number written on the button itself.
Both windows state what is reversible before you start: your own files are never
written to, the cleaned pixels live in the bank's own copy, and ↩ **Undo
cleaning** deletes those copies and re-flags the images. Undo is bank-wide rather
than per run, and two things are out of its reach — an image you already promoted
(that copy was written into the dataset) and an image whose source file changed
on disk since the clean.

## When a folder is already one person

Scraped material usually arrives sorted: one folder per person. **👤 Group by
person** does not know that, so it pays one face embedding per image to
rediscover what the folder name already said — thousands of inferences for an
answer you had before you started.

Scope the grid to a folder with the **Subfolder** picker and the panel under it
offers **👤 Single person here**. One click groups every image of that folder as
one person, instantly, with no pass at all — and the next 👤 Group by person run
**skips those images entirely**. That skip is the saving: on a bank of 9 000
images where 8 000 sit in asserted folders, the pass embeds 1 000.

It is a rule, not a stamp. It survives re-scans, and an image you drop into the
folder tomorrow joins the group the moment the bank sees it. It is also
reversible at any time — **↩ Not one person after all** dissolves the group and
puts the folder back in the way of normal clustering. Nothing is deleted either
way.

**Check a sample (15 images)** is the honest counterweight. It picks about
fifteen images spread across the whole folder (not the first fifteen — those are
usually one shoot), embeds *only those*, and compares them at the same
similarity threshold the clustering uses. You get either *sample consistent
(14/15 same person)* or *2 different faces in the sample — check this folder*.
Two limits, stated plainly: fifteen images cannot prove a folder is clean, only
that the sample looked one way; and whatever it finds, **your assertion stands**
until you revoke it. It informs, it never overrules you.

Images in the folder that the face machinery could not read — no face in frame,
a face too small or too turned — are listed as *worth a look*. They stay in the
group: "I could not see a face here" is not "this is someone else".

### The app asks the question for you

You should not have to guess which of your forty folders are worth declaring, so
the same sampling runs by itself and **suggests**. A folder it sampled and found
consistent gets a **👤?** next to its name in the Subfolder picker, and scoping
to it says *Looks like one person (15/15 of the 15 sampled) — assert?* next to
the button. A folder holding several people says so too, which is just as useful.

**It suggests. It never asserts.** Confirming is always the same single click it
always was. This is deliberate: a wrong assertion made silently would corrupt
your person grouping with something you never said, and you would have no reason
to go looking for it.

It runs in three places, and the difference is when you are asked:

- **as the preflight of 👤 Group by person** — the default path, described in the
  next section. You are asked at launch time, before the expensive pass runs.
- **automatically at the end of 👤 Group by person** — free. That pass has just
  cached an embedding for every image, so sampling every folder adds no
  inference at all and no GPU time. The pass's line then ends with *N folder(s)
  look like a single person*.
- **on demand, with 🔎 Scan folders** — a secondary path now, for asking well
  before you launch anything. This one pays about fifteen embeddings per folder,
  so it says how many folders it will cover before you click, and covers the
  twenty biggest first when there are more. It tells you what it did not reach
  rather than leaving you to assume the rest are not one person.

A suggestion expires when the folder changes. If images arrive or leave, the
verdict no longer describes what is in front of you, so it is dropped and the
folder goes back into the queue instead of advising you from stale evidence.

## Checking your folders before the person pass

Everything above used to be reachable only from the Subfolder panel — and the
first thing anyone does with a fresh bank is press **🚀 Launch all**, so they
never opened it and paid the full face pass over forty folders that each held
one person. A saving the default path walks past is not a saving.

So the sampling now runs **as the preamble of the pass itself**. Press **👥 Group
by person**, or **🚀 Launch all** with the person pass ticked, and before
anything expensive starts the bank samples about fifteen images in each
subfolder it has not been told about, then asks you once:

> **12 folders look like a single person** — treat each as one person and skip
> their full analysis.

Those twelve are **already ticked**. One click on **👤 Group 12 folders & analyze
the rest** confirms them and starts the pass you asked for; untick any you
disagree with; **👥 Analyze everything anyway** is right there and states its own
cost. It is still an offer, never a decision — a wrong grouping made silently is
one you would have no reason to go looking for.

Four things the dialog always tells you:

- **what the check costs, against what it saves** — *Checking 12 folders (~15
  images each — 180 in all, up to 720 where faces are hard to find), against the
  7 316 this pass would embed.*
- **what ticking the boxes spares** — *3 412 images are grouped instantly and
  skipped by the pass.*
- **why a folder is not offered** — *3 different faces in the sample — analyzed
  in full*. A doubtful folder is never quietly ticked.
- **what it did not reach.** The preflight covers up to 200 folders in one go.
  Beyond that it says *N folders were not checked (biggest first) — they get the
  full analysis*, because silence there would read as "the rest are not one
  person".

### When the sampled images have no face in them

Scraped folders are full of crops, backs, distant shots and blur. A sample of
fifteen can land entirely on those, and until recently that ended the folder's
story: *only 0 of 15 sampled images had a usable face — analyzed in full*. On a
3 546-image folder that meant fifteen embeddings spent for no answer at all, and
then the whole pass anyway — exactly the cost the check exists to avoid.

A draw that cannot be read is now **replaced**. The check keeps drawing new
images — never one it has already tried, still spread across the whole folder —
until it has about fifteen images with a usable face, or until it runs out of
**budget**. That budget is the point, because "keep drawing" without one is the
full pass by the back door. It is the smaller of two numbers, per folder:

- **at most 60 images** — fifteen usable faces at a hit rate of one in four,
  which is the worst rate still worth chasing;
- **at most a quarter of the folder** — so a small folder is never nearly
  analysed in full just to be described. Folders of 60 images or fewer keep the
  single draw they have always had.

That cap is also why the check can never quietly become expensive: a quarter of
a folder is a quarter of what analysing it would cost, and the dialog prints the
ceiling next to the typical cost before you start.

Three ways it can end, and each says which one it is:

- **enough usable faces** — the verdict you already know: *15/15 of 30 sampled
  images look like the same person.*
- **the budget ran out with a few** — *looks like one person, on thin evidence —
  only 6 usable faces in 60 images tried.* It is still offered and still
  pre-ticked, because the bar for an offer has always been two agreeing faces and
  six is more evidence than two, not less — but the row says what it rests on so
  you can weigh it.
- **almost nothing readable** — *no readable face in 60 images tried across the
  folder — crops, backs or blur.* This is not the check failing; it is what the
  folder is. **The full pass will not do better on those images**: the preflight,
  the folder check and the pass all drive the same detector at the same
  thresholds, and the check writes its answers into the pass's own embedding
  cache, so the pass reads them straight back rather than looking again. Grouping
  by face simply has little to grip in that folder, and much of it will stay
  ungrouped whatever you run.

If there is nothing to ask — a bank with no subfolders, or one whose folders you
have already declared — no dialog appears at all and the pass starts straight
away. And whatever you accept here is an **ordinary assertion**: it survives
re-scans, adopts images that land in the folder later, and **↩ Not one person
after all** undoes it exactly as if you had clicked it by hand.

While the check is running you can stop it with **👥 Analyze everything anyway**;
it lets the sampling go and launches the full pass.

## Pick a balanced set

Advice is only half the gesture, so **📊 Coverage advice** ends with **⚖️ Pick a
balanced set…** (the same button sits in the **Curate** row). It answers a
question no per-image score can ask: *does my set cover what I want to be able to
generate?*

Ask **🎨 Pick diverse** for 20 images out of a bank that is 47% full body, 35%
bust, 12% face and 6% back views, and you get roughly those proportions — on a
synthetic reproduction of exactly that shape it returned **0 face shots and 0
back views**. The LoRA then renders one shot type well and the rest badly, and
nothing ever said so. **⚖️ Balanced pick** returns **5 face, 5 bust, 5 body, 5
back** out of the same pool, each bucket filled with the *same* most-varied
sampling — and the same **Skip the odd ones out** guard — that 🎨 Pick diverse
uses.

- **Balance on** — **Framing** by default. It is the axis that carries real
  information: on a one-subject bank, person groups are sparse and split into
  many small, arbitrary clusters, so balancing on them spreads a selection over
  noise. **Framing × person** is there for a dump that genuinely holds several
  subjects.
- **When an axis can't be satisfied**, it says so instead of quietly filling the
  gap: *"Only 3 back images exist in this filter — an even split wanted 15"*. The
  freed picks go to the buckets that have room, so asking for 60 still gives you
  60 — the deficit is reported as a number, never hidden. If even that isn't
  enough, it says how many you actually got and why.
- **The result is always stated** — *"Selected 60 of 60 requested, spread over
  framing: 15 face, 15 bust, 15 body, 15 back"* — as text, per bucket, next to
  what each bucket had available. There is no chart you have to read.
- **An unlabelled bank is the normal state**, not an error. Nothing has a framing
  until the 📐 Framing pass has run, so the button says which pass is missing and
  how many images it would bring in, rather than returning an empty or misleading
  selection. 🎨 Pick diverse keeps working without it.

Like the other selectors it honours the current filter and search, and it only
**selects** — nothing is kept, rejected or deleted.

## Is this image really what it says it is?

Two things a file will happily lie about, both measured by the ordinary
**🔎 Scan quality** pass — plain CPU work, no extra install, no GPU.

**Its size.** An image enlarged from 512 px to 2048 px still *reports* 2048, so
it walks into a dataset as a high-resolution shot and the LoRA learns
interpolated mush. The scan measures how far real detail actually goes and says
it in pixels on the image's details line: *"2048 px stored · ~512 px of real
detail"*. The worst offenders sit behind the **🧇 Soft detail** filter chip,
and *Settings → Captioning & quality → Real-detail minimum* moves the bar.

Treat it exactly like the sharpness score: **a shortlist, not a verdict.** A
photo with motion blur, a portrait with the background thrown out of focus, and
a heavily denoised phone shot all genuinely lack fine detail and all read the
same way as an enlargement — which is fine for choosing training images (a LoRA
learns as little from either), but it is not proof the image was ever resized.
Look before you mass-reject. Two honest limits: a *nearest-neighbour* enlargement
is invisible to it (blocky pixels are real high-frequency detail), and large
enlargements are under-stated, so the pixel figure ranks images rather than
recovering the original file's size.

**Where it came from.** The scan reads the file's own metadata and sorts the
bank with the **🔎 Origin** chips:

- **🤖 AI** — the file still carries generation metadata: a ComfyUI workflow
  in the PNG, A1111-style `parameters`, or the C2PA/XMP "generated" marker the
  commercial generators write. Certain when present.
- **📷 Camera** — the file still carries camera EXIF (make, model, exposure).
  Strong evidence it was actually photographed.
- **❔ Unknown** — nothing left to read. **This is the normal answer**, not a
  failure: scrapers, chat apps and social networks strip metadata on sight (on a
  36 000-image Telegram export, *every single file* landed here). It is not
  evidence the image is a real photo, and it is not evidence it is AI — it is
  the absence of evidence, which is why it is its own answer instead of being
  quietly folded into "not AI".

On an image whose metadata is gone, the details line may add a *hint* when the
dimensions are a standard generator size (1024×1024, 832×1216, 896×1152…) and
there is no camera EXIF. It says it is a hint; plenty of crops and downloads
land on round numbers too.

Two smaller facts come free with the same pass: **🎞 Black bars** flags flat
letterbox/pillarbox padding (video screenshots, stills padded into a square,
which survive a training crop), and the **JPEG quality** of the last save is
shown as-is — a low figure means the file has been through a re-encoding
pipeline, but it is far too common to be worth a filter.

A bank you already scanned picks all of this up on its next **🔎 Scan** — the
pass re-visits the images that predate these measurements on its own. You do not
need a full rescan.

## Sort a bank by medium and by head angle

Two more ways to slice a big dump, both built on passes you have already paid
for.

### 🎨 Medium — what the picture is *made of*

**🎨 Classify medium** sorts every scored image into **📷 Photo**, **🅰 Anime**,
**🧊 3D render**, **🖌 Illustration** — or **❔ Unsure**. It reads the CLIP
embedding the **✨ Score** pass already computed, so it looks at no image twice,
downloads nothing, and never touches the GPU. On a 23 000-image bank it finishes
in seconds. An image ✨ Score has not reached has no embedding and stays
unclassified; the row says how many.

**You no longer have to ask for it.** Because it costs nothing beyond what
✨ Score already paid, it now runs **automatically at the end of every ✨ Score
pass**, and the pass's own line reports it (`· 🎨 Medium: 812 classified`). If
the CLIP text encoder is missing, the line says *skipped* and names the reason
rather than staying quiet. The **🎨 Classify medium** button is still there: it
is how you re-run the pass on its own, and how you re-classify images that
already carry a verdict — something the automatic run never does, so a verdict
you are looking at is never rewritten behind your back.

This is **not** the same question as **🔎 Origin** above. Origin reads the
*file's metadata* and answers "who made this file". Medium reads *the picture*
and answers "what does it look like". A photorealistic AI portrait is 🤖 AI and
📷 Photo at the same time; a scanned manga page is ❔ Unknown and 🅰 Anime.
Neither is evidence for the other.

**What it is worth, measured.** On a real 23 532-image bank, against 167 images
labelled by hand:

- photograph verdicts were right **90 out of 90** times;
- both real anime drawings in the sample were found;
- every 3D render and illustration in the sample came back **Unsure**.

That last line is the honest shape of this feature. The bar for a non-photo
verdict is deliberately six times higher than for a photograph, because the
model reads a picture's *subject* as much as its medium: a photo of somebody
**cosplaying** an anime character scores as anime. At a lower bar the "anime"
pile filled with cosplay photographs and the "3D render" pile with advertising
banners. So the pass answers **Unsure** rather than guessing, and the row prints
how big that pile is instead of hiding it. Sort by **🎨 Medium confidence ↑** to
put the images it nearly could not call in front of you.

### ⤢ Angle — where the head is pointing

The **🎭 Person groups** pass estimates a head pose while it works. The **⤢**
chips turn that into **😐 Frontal** (turned less than 20°), **◑ Three-quarter**
(20–60°), **👤 Profile** (more than 60°) and **🔙 From behind**.

Two limits worth knowing before you trust a count:

- **Profile is under-counted.** A head turned far enough that one eye disappears
  often defeats the face detector outright, and an image with no detected face
  has no angle at all. The profiles you see are the ones that were still
  detectable.
- **From behind needs two passes.** It is the crossing of "no face found" with
  "the **📐 Framing** pass called it a back view" — because *no face* on its own
  is also what a landscape with nobody in it looks like. Without the framing
  pass this bucket stays empty rather than claiming a person is there.

**If your bank was scanned before this shipped**, its faces have no angle: older
builds measured the pose, used it once and threw it away, and the number is not
recoverable from what was stored. The ⤢ row then offers to measure them, tells
you how many there are and roughly how long it will take on your machine, and
does nothing until you click. It re-runs the face detector on those images only,
writes nothing but the angle, and leaves your person groups exactly as they are.

## Set the bank filters from a sentence

At the top of **Triage**, **🗣 Describe the set you want** takes a plain request —
`an amateur photo set, least polished first` — and moves the bank's own controls:
medium, quality flags, resolution tier, sort. The chip counters below then say,
measured, how many images that lands on.

The model never looks at your images and never chooses any. It reads the sentence
and nothing else, so a wrong reading costs you one glance at chips you can edit,
not a silent selection you would have to trust. Everything it proposes lands in the
same filters a click would set, and clearing them is the same gesture as always.

It answers over what your bank has actually measured. The real per-value counts go
to the model with the request, so it cannot reach for a bucket that holds nothing.

**It says when it cannot.** Asking for what is *in* the pictures — `women
outdoors` — has nowhere to land while captions cover a small fraction of a bank
and framing almost none of it. That part of the request comes back as *not
expressible here* rather than as a filter that would return a few thousand
convincing, unrelated images.

**It will not turn an exclusion into a search.** The ranker returns *more* of a
negated thing, not less (`a woman without a bikini` measured 60% bikinis against a
10.1% baseline), so `without a watermark` is reported back to you instead of being
quietly sent. To guarantee an absence, use the word-exclude box.

## Choose CLIP or SigLIP 2 for Bank semantics

Each Bank has its own **Semantic engine** choice in **① Analyze**:

- **CLIP** is the compatible default. Its index is the embedding cache already
  produced by **✨ Score**, so every existing Bank behaves exactly as before.
- **SigLIP 2** is optional. Install the pinned model once in **Setup ▸ Quality
  tools**, select it on the Bank, then explicitly build that Bank's semantic
  index. Selecting it never starts a scan or downloads a model by itself.

The selected engine powers **Find by text**, **Similar to selected**, **Pick
diverse**, **Balanced pick**, visual spread/coverage and **Find crops &
variants**. The calibrated aesthetic head, NSFW score, visual-style groups and
**🎨 Medium** remain on CLIP regardless of this choice.

CLIP and SigLIP 2 use separate, model-versioned caches and separate **same-shot
group partitions**. Switching swaps the visible partition but keeps both, so
returning to an engine restores its grouping instead of erasing completed work.
Both partitions and their exact cache entries travel with the existing analysis
snapshot on Bank → Dataset, Dataset → Bank and Bank → Bank copies; a changed
image fails the fingerprint check and is re-indexed instead of receiving stale
analysis.

The SigLIP 2 index is resumable and stoppable like Score: completed entries are
written atomically, and a later launch pays only for missing, failed or changed
images. **Reindex SigLIP 2** rebuilds that cache only; it never touches Score.

## Find bank images by describing them

Under **Curate**, **🔤 Find by text…** ranks images by how close they are to a
phrase you type — `brunette outdoors, wide shot`, `red dress against a white
wall`, `close-up, harsh flash`. It reads the Bank's selected semantic index:
the existing **✨ Score** cache for CLIP, or the separate index you explicitly
built for SigLIP 2. A search itself performs no image inference; searching while
a LoRA trains is fine.

**It is a ranking, not a filter.** Every image scores *something* against every
phrase, so a result list always comes back full. The panel therefore reports the
similarity of the best and of the last result, and tells you how far apart they
are — *"all about equally close"*, *"the last ones are noticeably looser"*, or
*"the tail is much weaker than the top"*. That spread is the useful signal: it
says whether you can trust the bottom of the list.

**Do not read those numbers as percentages, and do not compare engines by their
raw values.** The following measurements are specifically for the default CLIP
ViT-L/14 `openai` space, on a real bank (48 images from 8 unrelated datasets):

| | Range |
|---|---|
| Top-1 results verified correct by eye | **0.177 – 0.233** |
| Guaranteed-unrelated image/phrase pairs | median **0.112**, up to **0.197** |

So 0.22 is not "22% of a match" — it is roughly as good as this model ever gets.

**And this is why there is no similarity slider.** Look at the two rows again:
the unrelated *ceiling* (0.197) is **higher** than two genuinely correct answers
(0.177 and 0.178). The distributions overlap, so no cut-off separates "relevant"
from "unrelated" — anything below ~0.20 lets false positives through, anything
above ~0.18 throws away true matches. A threshold control would be a knob on a
boundary that does not exist, so the app gives you a result *count* instead and
shows the ranking honestly.

The app never compares your scores against those figures either. It measures
what a *typical* image of **your** bank scores for **your** phrase, and describes
the results relative to that — which is the only version of the question that
survives a different bank.

**On a bank that is mostly one subject** — the normal case here — expect the
ranking to flatten. Images of the same person score 0.60–0.89 against *each
other*, far above any text score, and a query's ability to discriminate
compresses by 30–70%. The summary will say *"barely above what any image here
scores — the order is a hint at best"* when that happens. Believe it: at that
point the first result is not meaningfully better than the tenth.

It searches **inside the current filter**, exactly like 🎨 Pick diverse and 🎯
Similar to selected. So "wide shots, in this subfolder, among the undecided" is
just a filter plus a phrase; nothing needs a second search grammar. Results land
as a normal selection you review with ✓ Keep / ✕ Reject / ⬆ Promote — nothing is
kept or deleted for you. **Clear search** returns to the full grid.

**Images missing from the selected index cannot be found by any phrase.** Rather
than letting them vanish, the summary counts them. Run **✨ Score** for CLIP, or
complete the explicit **SigLIP 2 index**, to include them.

### What it is good at, and what it is not

The default CLIP engine reads a picture as a whole. It is reliable for **subjects, styles, framing,
setting, materials and colour**, and unreliable for three things in particular:

| Ask for | What you actually get | Measured |
|---|---|---|
| **Counting** — "two people" | Photos of people, any number. | On a two-person image, "two people" beat "one person" by **0.001** — pure noise. It separates "one" from "several" at best. |
| **Negation** — "without glasses" | *More* glasses, not fewer. | On a photo of an astronaut **wearing** a helmet: "with a helmet" **0.212**, "without a helmet" **0.217**, plain "an astronaut" **0.219**. The negation scored **higher** than the affirmation. |
| **Spatial relations** — "to the left of" | Both objects, in any arrangement. | — |

The negation case is the one to remember, because it fails *silently and
backwards*: CLIP does not penalise "without", it simply ignores the word. Someone
searching `woman without glasses` gets women **wearing** glasses and has no way
to tell the search misfired. The same measurement on a 7,316-image bank: `a
photo of a woman without a bikini` returned **60% bikinis**, against a 10%
base rate — the query did not miss, it inverted. See **Push down** below.

These are properties of the model, not bugs to report. Describe what *is* in the
frame rather than what is absent, check counting and left/right by eye — and for
the negation case, use the **Push down** field described next, because typing
"without" will never work.

### Push down what you do not want

The panel has a second field, **Push down**, for the trait you are trying to get
away from: `hat`, `sunglasses`, `blonde hair`. You can also write it inline in
the query with a leading dash — `a woman in a car -hat` means the same thing.
Typing a query that starts negating something ("a woman without a hat") offers
you the field instead, rather than letting the search fail quietly.

It does **not** filter. The excluded phrase is encoded exactly like the positive
one and *subtracted* from each image's score, so images carrying that trait sink
in the ranking. They are still in the pool and one can still surface if it is
otherwise the best answer. If you need a guaranteed absence, that is a tag
filter's job, not this one.

**How hard** offers Gentle / Normal / Strong. The default, Normal, was measured
over 7,316 real bank images that carry both a CLIP embedding and a written
description, across 19 query/exclusion pairs, counting the top 60:

| How hard | Top 60 still carrying the unwanted trait | Top 60 still on-topic |
|---|---|---|
| off | 23.0% | 89.7% |
| Gentle | 11.9% | 89.5% |
| **Normal** | **7.6%** | **87.7%** |
| Strong | 3.8% | 79.8% |

Pushing harder always removes more of the trait — what you pay for it is
relevance, and that stays essentially flat up to Normal (2 points) then drops
off a cliff (10 points at Strong, 25 past it). That is why Normal is the default
and why Strong is described as a trade rather than as "better".

**Some pairs cannot be separated at all,** and the app says so instead of
pretending. Excluding `a bikini` from `a woman at the beach` barely moved: at
every usable strength two thirds of the results still had a bikini, because in
this model's eyes a beach photo largely *is* a bikini photo — and by the strength
that finally bit, the beach was gone too. After each search the summary reports
what actually happened on *your* bank: how many results the push-down brought in
that would not have been there, and how strongly the returned set still matches
the unwanted phrase compared with a typical image of the bank. When it changed
nothing, it says that too.

One last caveat, seen in the same measurement: a result can be right on the broad
trait and wrong on the detail. A generic indoor query returned a genuinely indoor
shot that was not the *kind* of indoor scene the wording implied. Text search
brings the likeliest images to the front; the final call stays yours.

### Why the first search takes a moment

The text encoder is the other half of the selected image/text model. Loading the
default CLIP encoder costs about **ten seconds** on the CPU; SigLIP 2 also has a
one-time model load. The app keeps the chosen encoder warm after the first
search, then releases it when you close the panel or after the idle window.
Every phrase is cached under that engine's model key, so CLIP and SigLIP 2 text
vectors can never be mixed and re-typing one is free even after a restart.

On a memory-tight machine you can set `bank_scoring.text_search_idle_minutes` to
`0`: nothing is ever kept warm, and each new phrase pays the ten seconds instead.

## Choose who captions a bank, and which pile

The 🏷️ **Caption** pass in ① Analyze has its own **Caption options** row, and
every control on it applies to **that run only** — your Settings stay the
default and are never rewritten from here.

**Which pile gets captioned.** Three choices, and rejected images are in none of
them:

- **Kept + undecided** — the default, and exactly what the pass always did.
- **✓ Kept only** — caption what you have already chosen, and nothing else. This
  is the cheap one: on a 20 000-image dump where you kept 300, it is 300 vision
  calls instead of 20 000.
- **Undecided only** — the opposite errand. Captions feed the 🔍 search and the
  🏷️ tag chips, so captioning the undecided pile is how you get *tools* to
  triage it with.

Each option carries its own count, and the button quotes the number it is really
about to write. That number is **not** the size of the pile: images that already
have a caption are skipped, so a bank of 4 000 kept images can honestly offer
"Caption 12 kept". When everything in a pile already has a caption the button
says so and goes inert.

**A selection wins.** Select images first and the scope select greys out: the
pass captions your selection, and the button switches to counting it. The server
would otherwise *intersect* the two, and "Caption 12 selected" could quietly
write 4.

**Which engine, and which model.** Two more selects on the same row:

- **Caption engine** — *Auto* is a chain, not a coin flip: JoyCaption drafts and
  Ollama covers whatever it missed. Forcing *JoyCaption only* removes the Ollama
  half rather than picking one of two.
- **Caption vision model** — any Ollama model you have pulled. It is only used
  when the engine can reach Ollama, and it is greyed out otherwise. A model
  configured elsewhere stays selectable even if it is not in the live list.

This last one matters more than it looks. A captioner that describes plainly
visible things in evasive terms produces captions that are about something
slightly *other* than your images — and a LoRA trained on those learns to look
away too, with nothing in the output to reveal it. The captions read perfectly
well. That is the problem. If you caption NSFW material, pair the **Explicit**
register with an uncensored (abliterated) model; the app warns you when the
model it is about to use does not look like one.

You can change the model between runs on the same bank. 🏷️ **Caption** never
rewrites anything: it only fills images that have no caption yet, so a second run
with a different model captions the rest, not the ones already done. To redo the
ones already done, see the next section.

## Redo the captions of a bank with a different model

🏷️ **Caption** skips images that already have a caption — which is what you want
until the day it isn't. Once a bank is fully captioned that button reaches zero
images and goes inert, and on a bank you captioned with a model you have since
decided was a poor one, "nothing left to caption" is the wrong answer.

🔄 **Re-caption**, at the end of the **Caption options** row, is that answer. It
runs the same pass with the same engine, model, register and length you picked on
that row, on the pile the scope select names — and it **overwrites** the captions
that are already there.

**It keeps the captions you wrote yourself.** Every caption now records who wrote
it — JoyCaption, Ollama, or you. "You" means: typed or corrected in a dataset's
caption box, changed by a find/replace across a dataset, or brought back as `.txt`
sidecars from another tool. That record travels with the text through
**Import to bank**, bank-to-bank copies, promotion back to a dataset, and backup
restores, so a caption you wrote in a dataset three steps ago is still recognised
as yours here. Re-caption skips those rows, exactly as the person pass skips a
subfolder you declared to hold one person.

**It tells you three numbers before you click, and never merges two of them.**
The button quotes what it will rewrite (the pile, minus what it spares). The amber
line under the row breaks the rest apart: how many captions it *keeps* because you
wrote them, how many it overwrites **whose author was never recorded**, and how
many a model wrote. The confirmation repeats them. None is an estimate; they all
come from the same count the pass itself uses, so the figure on the button is the
number of images that change.

**"Origin never recorded" is the one to read carefully.** Captions written before
the app started keeping track carry no author, and there is no way to work one out
after the fact. Those are re-captioned — sparing them would make this button do
nothing at all on any bank that already exists — so if you hand-wrote captions in
an older version, they are in that count. It is stated separately from the
machine-written ones for exactly that reason.

**If you do want your own captions redone**, tick **"Also rewrite the N caption(s)
I wrote"** next to the button. It only appears when there is something to protect,
it is never pre-ticked, and the confirmation names it again.

**There is still no undo.** The bank's ↩ Undo covers keep/reject decisions only;
it has never covered captions, and this change does not add one.

**It works by pile, never on a selection.** With images selected the button goes
inert and says why: a selection can cover pages that were never loaded, so the
app cannot count how many of them already have a caption — and it will not run a
destructive pass on a number it cannot state. Clear the selection to re-caption a
pile. 🏷️ **Caption** still honours selections as it always did.

## Review a bank one image at a time

Filter chips and bulk actions clear the obvious trash, but the last call —
*is this shot good enough for the LoRA?* — is made one image at a time, and
squinting at a 140-pixel thumbnail is not how you make it. **▶ Review one by
one** (above the grid) opens the images of the **current filter** full size, one
after the other:

- **✓ Keep**, **✕ Reject**, **⏭ Skip** — each one saves and jumps straight to the
  next image. The keyboard is the point: **K** keep, **R** reject, **S** skip,
  **←/→** move without deciding, **Esc** to leave. A few hundred images go by in
  minutes.
- **⏭ Skip** decides nothing (the image stays undecided) but is not shown again
  in that run — it's "not now", not "no".
- **🎲 Random order** walks what's left in shuffled order instead of folder
  order. On a scraped dump of 3 000 photos, sequential order means 200
  near-identical frames in a row; random gives you a representative sample from
  the first click. Ticking or unticking it mid-run only re-orders what you have
  **not** seen yet — nothing you already judged comes back.
- Under the image, the facts the passes already computed (resolution, sharpness,
  aesthetic score, NSFW, quality flags, person and duplicate groups) so you can
  call it without leaving the lightbox.
- The counter is honest — *12 / 340* over the snapshot taken when you opened the
  review, so a decision that drops the image out of the filter can't make the
  run skip images or loop. Each decision is saved on the spot: close after fifty
  of them and all fifty are there.

The ▶ button on a tile starts the same review **at that image**. A plain click
on a tile still selects it for the bulk ✓/✕/⬆ bar, so both ways of working stay.

## Promote a shortlist into a new bank

**⬆ Promote…** has two destinations, and picking the right one saves you a mess.

- **📁 An existing dataset** — the end of the funnel. The images are normalized
  to webp, deduplicated against what the dataset already holds, and become
  training material.
- **🗃 A new image bank** — for when you are not there yet. A 9 000-image dump,
  200 candidates isolated out of it, and you want to keep working on those 200
  apart: give the new bank a name and the selection lands in it, **un-triaged**,
  with every bank tool available again (scan, dedup, framing, captions, review).
  Nothing is committed to training.

With images selected in the grid, those are the ones that go; with nothing
selected, every **kept** image does.

Either way this is a **copy**. Banks never share their files, deliberately: the
app rewrites images in place (a re-crop, a watermark clean), so two banks reading
one file would stop being two banks at the first edit. The dialog therefore
states, before you click, **how many megabytes** the copy costs — a measured
figure for that exact selection, not an average. For photographs it is usually a
footnote; the line is there for the day a bank holds something heavier.

Your source bank is untouched by all this. It keeps every image, now marked ⬆
promoted, and your original folder is never written to — the copies live in the
app's own data folder, and deleting the new bank takes them with it.

If the copy cannot be written — a full disk, a drive pulled out — the new bank is
**discarded** rather than left holding half the shortlist and looking finished.
You are told what happened and nothing has changed.
## Undo the last bulk decision

A bank lets you mark hundreds of images with one click: select the whole filter
and press ✕, apply an auto-reject at a threshold, collapse every duplicate group,
or run 🚀 Launch all. That is the point of a bank — and it is also the click you
most want back when the threshold was wrong or the filter was not the one you
thought.

After any of those, an **↩ Undo** bar appears above the grid saying what
happened and how many images it moved. Press it and every one of those images
goes back to exactly what it was: its previous ✓/✕/undecided state *and* the
reason it carried. Images the action never touched are not touched here either —
if you had already kept a photo by hand and the bulk reject flipped it, undo puts
it back to **kept**, not to undecided.

The bar does not disappear on a timer, and it survives a page reload: the
decision it takes back lives in the app's database, not in your browser tab. It
stays until you use it, dismiss it, or run another bulk action.

**Its limits, stated plainly.**

- **One step.** Only the most recent bulk action is remembered. Run a second one
  and it replaces the first — this is a net under the click you just made, not a
  history of your session.
- **Until the app restarts.** The memory is in the running app. Restart it and
  the offer is gone; the decisions themselves are safely saved, as always.
- **It never over-claims.** If some of the images have left the bank since (a
  re-scan noticed the files were gone), or if you changed some of them yourself
  in the meantime — in ▶ Review, or in another tab — those are *not* overwritten.
  The result tells you exactly how many it restored out of how many, how many
  are gone, and names the ones a newer decision now owns.

**What is deliberately NOT offered.** Two bank actions have no undo, because a
half-working one would be worse than none:

- **🗑 Delete rejected** sends your source files to the recycle bin and drops
  their rows with everything the passes had computed about them. Files in the
  recycle bin are yours to restore, from your file manager — the app cannot do
  it for you, and it will not pretend otherwise. This action also withdraws any
  pending ↩ offer, since the images it pointed at are the ones just removed.
- **⬆ Promote** copies images into a dataset (or a new bank) through the normal
  import path. Un-promoting would mean deleting images in a dataset you may have
  already captioned, cropped or trained on. Delete them there if you want them
  gone.

The 🔄 rotate button needs no undo entry: turn the other way and the image is
byte-for-byte the original again.
## Find more images like this one — by attribute, not by look

**Select an image** in a captioned bank and its tags are already there: beside
the gallery on desktop, or in the filter bar on a phone. Tick `woman`, `red`,
`dress` or `balcony` and the grid narrows to the images whose captions mention
them. No extra click, no badge to find.

**Select several and the row counts.** Each chip carries how many of your
selected images cite it — `red dress 7 / 12` means 7 of the 12 captioned images
you picked mention it. That is deliberately *not* an intersection: keeping only
the tags every single image shares would print 12 next to each survivor (a number
that says nothing) and usually leave you with one word. What you want to know is
that a tag describes over half of what you selected.

The row is honest about what it did **not** count, on its own lines:

- images in your selection with **no caption yet** — named, not folded into the
  denominator, so `7 / 12` always means 7 of 12 images that had something to say;
- images whose caption held **no word worth filtering on** (`a photo of her`) —
  a different problem with a different fix;
- a selection **too large to read in one request**, which says how many images it
  left out rather than quietly shrinking the total.

Tick a chip and the row **holds still** while the filter runs, even though
filtering clears the selection — it keeps showing the tags of the selection you
filtered *from*.

The 🏷️ **badge on a tile** is still there, in the bottom-right corner next to ▶
and ⛶ where the tile's actions live. It reads one image's tags *without*
selecting it. On an image with no caption — or a caption with no word worth
filtering on — the badge stays visible and greyed, and its tooltip says which of
the two it is: a feature that silently disappears is indistinguishable from one
that was never built.

This is the readable cousin of **🎯 Similar to selected**, and the difference is
worth knowing because they fail differently:

| | 🎯 Similar to selected | 🏷️ Tags of this image |
|---|---|---|
| Matches on | the whole look (the selected CLIP or SigLIP 2 index) | words *you* ticked |
| Works without captions | yes | no |
| Tells you *why* it matched | no | yes — the chips you ticked |

Details that decide what you get:

- **Several chips mean AND.** Ticking `red` and `dress` shows images mentioning
  both, so every extra chip narrows further. The line under the chips says so
  while the filter is active.
- **Chips are matched as whole words**, in captions *and* file names. `car` will
  not bring back `scarf`. (The 🚫 exclude box below is looser — it matches
  anywhere — because a word you type by hand is often a fragment on purpose.)
- **Booru captions keep their tags whole** (`red dress` stays one chip); prose
  captions are cut into words, so `golden hour` becomes two chips and ticking
  both means "captions with both words", not "captions about golden hour".
- **It only sees what a captioner wrote.** An attribute nobody put in words is
  invisible here, however plain it is in the picture. Caption more of the bank
  (🏷️ Caption all) and the chips get better.
- It composes with every other filter, and it travels with them — **Select all
  in filter**, **▶ Review** and the curation picks all work on what you can see.

## Hide images you have already handled

The bank's 🔍 search box narrows the grid *to* a word. Next to it, the 🚫
**Exclude words** box does the opposite: it hides every image whose **caption or
file name** contains what you type. That turns a captioned bank into a checklist
— *what have I not tagged yet?* — instead of a list you have to keep re-reading.

- **Several words at once**, comma-separated: `logo, watermark, screenshot` hides
  anything mentioning any of them.
- **It composes with everything else** — the search box included. Searching
  `dress` while excluding `red` gives you the dresses that are not red, and the
  filter chips, subfolder, resolution tier and framing all still apply.
- **It travels with the filter**: **Select all in filter**, **▶ Review one by
  one** and the curation picks (🎨 diverse, ⚖️ balanced, similar) all work on the
  visible set, so an image you hid is never handed back to you by a pick.

Two limits worth knowing:

- **It matches anywhere in the text**, like the search box — so `car` also hides
  `scarf`. Type the longer word when that matters.
- **Images with no caption are never hidden.** They have nothing to match, and
  hiding them would remove exactly the images a checklist is looking for.

Unlike the sort, the exclude box is **not remembered** between visits: an order
you can see in a menu is a habit, but images missing from a grid for a reason you
set last week reads as data loss.

## Sort a grid to review faster

Filters answer *which images*; sorting answers *which one first*. Both grids
have a **Sort** control, and it changes nothing but the order — the same images
match, the counts stay put, and every bulk action keeps operating on exactly
what the filters left.

In a **bank** (View ▸ Sort, next to the tile size) you can order by *anything the
passes measured*, either way. The menu is grouped by the pass that produces the
figure, so a greyed-out section also tells you which pass to run:

- **📁 File** — **Resolution ↓ / ↑** (megapixels, so a 900×900 outranks a wider
  1200×300) and **File size ↓ / ↑** (bytes on disk — the one figure no filter
  chip exposes).
- **✨ Score** — **Aesthetic ↓ / ↑** (the 1–10 rating; ↓ puts your keepers on the
  first page, ↑ puts the duds there, which is usually the faster way to prune)
  and **NSFW likelihood ↓ / ↑**.
- **🔎 Scan quality** — **Sharpness** (↑ brings the blurry misses to you),
  **Noise**, **Contrast** (↑ = the flattest, near-empty frames first), **Detail**
  (↑ = the enlargements pretending to be big images), **Letterbox bars** and
  **JPEG quality**.
- **🎭 Faces** — **Face confidence ↓ / ↑**, the detection score: ↑ surfaces the
  tiny, turned or half-hidden faces.

A chip and a sort answer different questions. A chip only ranks the images that
*cross* its threshold, so "the noisiest of the ones I am keeping" — all of them
below the threshold — is a question only the sort can answer, and no chip ranks
the other way round at all.

**The bank remembers the order you chose, per bank.** Reopen it tomorrow and it
opens the way you were reviewing it; other banks keep their own. Pick **Default**
to forget the preference.

In a **dataset** (above the grid, next to the decision chips): **Face similarity
↓ / ↑**, the ArcFace cosine against your reference photo computed by **🎭 Analyze
faces**. ↓ is "who looks most like my subject", ↑ is the shortlist to cut.

Two things worth knowing:

- **Images a pass never reached always go last**, in both directions. An
  un-analysed image has no score — putting it first would bury the very images
  you asked to see.
- **A sort you have no data for is greyed out** and says which pass to run,
  rather than pretending to reorder. Run the pass, and it lights up.

In a bank the ordering is done by the database over the *whole* filter, not just
the page you can see — so **Select all in filter** and **▶ Review one by one**
walk the same order you are looking at.

## Move through a dataset without closing the image

Open any dataset image full screen (the 🔍 on its tile) and you can walk the
whole grid from there: **⟨** and **⟩** on the left and right edges of the picture,
or the **←** and **→** keys. **Esc** closes, as before.

The badge next to the image's name — **12 / 340** — is the part worth reading. It
counts *the images the grid is showing you*, so:

- **The arrows follow your filters and your sort.** Chip the grid down to "34
  awaiting ✓/✕", sort by face similarity, and ⟩ walks those 34 in that order.
  Change a filter and the badge changes with it. They never step onto an image
  the grid is currently hiding — if they did, you would have no way to notice.
- **They cross pages.** A dataset over 500 images is paged, and ⟩ turns the page
  under the overlay: close the lightbox and you are on the page holding the
  image you were just looking at, not where you started.
- **They stop at the ends.** There is no wrap-around: on the first image ⟨ goes
  grey and says *"You are on the first of the 340 images shown here"*, and the
  same at the other end. On a wall of near-identical shots, a loop that silently
  restarts makes "have I seen everything?" unanswerable.

What does **not** travel with you: the 100 % zoom, an open **⧉ Compare with
original** pane, and an improvement running on the image you left. Each image is
inspected from a clean slate — a pane captioned *original* is always the parent
of the picture in front of you, never of the previous one.

Navigating is a *read*, so it keeps working while a generation, a captioning
pass or a watermark scan holds the dataset — the same rule as opening an image
and ticking a selection. Only the edits in the bar (crop, mirror, rotate,
improve) wait for the pass.

The rescue pairs in **Curation** are the one place with no arrows: there you are
judging one pair, not walking a list.

## Inspect an image on a phone

Below a phone-sized window the lightbox changes shape, and it is the same
picture, the same actions and the same keys — only their arrangement moves.

- **The image takes the screen.** In a side-by-side comparison, both panes do.
- **Every action moves behind one button**, the **☰ Actions** pill floating at
  the bottom of the picture: compare with the original, compare with the
  reference, crop, mirror, rotate left and right, improve, upscale, the Klein
  instruction with its editor and its model, and the links to Settings. Nothing
  is dropped and nothing is renamed — it is the same list, in the same order,
  in a panel instead of a strip.
- **The panel is a drawer, not a new screen.** It covers the bottom of the
  picture and leaves the top of it visible, so you can see what you are about to
  rotate. **Esc** peels one layer: it closes the panel first, and the lightbox
  only once the panel is closed. **Done** and the pill itself close it too.
- **Asking to compare closes the drawer**, because a comparison is a request to
  *look* at something. The edits (rotate, mirror, improve) leave it open — those
  get chained.
- **⟨ / ⟩ and the ← → keys still walk the grid**, and moving to another image
  closes the panel with the picture it belonged to.

Why it changed: at 400 px the old bar was not a bar. Crop, Mirror, two Rotates,
two Improve buttons and the Klein note each took a full-width row, and with the
Klein instruction editor unfolded the photo itself was left **96 px tall** —
about 11 % of the screen. Side-by-side comparison, where size is the entire
point, gave each pane **144 px**. Measured again after the change, on the same
screen: **538 px** for a single image whatever the editor is doing, and **354 px
per pane** in comparison.

On a desktop none of this applies: the actions stay in the bottom bar, or in the
side rail beside a portrait photo, which already spends width the image cannot
use.

## Compare an improved image with the original

Two things in the app never overwrite an image — they add a **candidate** next
to it, and leave the choice to you:

- **✨ Upscale & improve** in the dataset lightbox (a manual Klein pass, 2 MP by
  default);
- the automatic **small-image rescue** of scraped images under 768 px.

Open that candidate full screen and it now carries **⧉ Compare with original**.
The view splits in two named panes — *Original* and *Improved* (or *Klein
rescue*) — **side by side on a wide screen, stacked on a phone**, where width is
the scarce axis and two half-width thumbnails would prove nothing.

Both panes are the same size and both images are fitted inside them, so they are
shown at **the same scale and the same framing** even though the candidate has
more pixels. That matters: an improve pass rescales to a megapixel budget, and
two images displayed at different scales cannot be compared honestly.

**Zoom is off inside the comparison**, and the hint under the image says so. At
100 % a 2 MP result and a 0.5 MP original cover different parts of the subject —
that is not a comparison. Leave the comparison (⊟) and the usual click-for-100 %
inspection is back, on whichever image you are looking at.

When you **✓ Keep** a completed **✨ Upscale & improve** candidate, LDS keeps
both files but returns its original to **Undecided** automatically — so the
improved image is the one selected for training. This happens in the lightbox
and with bulk **✓ Keep**, even if you selected both tiles. Nothing is deleted:
you can still compare them, and can mark the original **Keep** again later if
you deliberately want to train on both.

If the original was deleted, rejected and purged, or simply never recorded (very
old rows), there is no button — a short amber note says why instead, so a
missing control can't be mistaken for a bug. Everything else in the lightbox —
✂ Crop, ⇄ Mirror, ✨ Upscale & improve — is unchanged and still acts on the
image you opened.

## Compare an image with the dataset reference photo

⧉ *Compare with original* only exists on the two kinds of candidate above. The
question you actually ask of an ordinary generated variation is a different one
— **is this still the same person?** — and its answer is the reference photo,
which lives in another panel and is therefore never on screen beside the image
you are judging.

Open any image in the dataset full screen and it now carries
**◐ Compare with reference**. Same split view, same named panes — *Reference*
and *This image* — side by side on a wide screen, stacked on a phone. It works
on **every** image, generated or imported, not only on improve candidates.

**Each pane fits its own image**, and that is the honest thing to do here: the
reference is a square head crop and the image beside it may be a full-body plan,
so there is no shared scale to promise. The hint under the panes says
*different framings* rather than *same scale* — that promise belongs to the
comparison against the original, where both images really are two renderings of
one shot.

The two comparisons are **exclusive**: pressing one leaves the other, because
two pairs of panes at once are four thumbnails and prove nothing. On an improved
image both buttons are there and you can flip between the two questions; on a
plain variation only ◐ *Compare with reference* is.

A dataset with **no reference photo yet** shows no button and no warning — the
reference panel already asks you for one, and a second nudge here would be noise
on a screen that cannot act on it. Zoom is off inside this comparison too; leave
it (⊟) for the usual click-for-100 % inspection.

## Tune the Bank filter thresholds

The filter chips (🌫 Blurry, 📐 Small, ≈ Duplicates…) are verdicts, and every
verdict comes from a number. Those numbers used to live only in
*Settings ▸ Captioning & quality*, three screens away from the bank you were
triaging. They are now also under the chips themselves: open **🎚 Filter
thresholds** above the grid.

It is the **same setting in both places** — one value, seen twice — so anything
you change here applies to **every bank**, and the panel says so at the top.

The twelve knobs are grouped by the question they answer: **Image quality**,
**Duplicates**, **Size & framing**, **Content**, **Style**. The first two are
open by default; the rest fold away, and a folded group tells you how many of
its values you have moved off the default.

Three things each control tells you that a bare number cannot:

- **Which way catches more.** "Stricter" is not a direction. *Duplicate
  distance* is a distance in hash bits — **raise** it to catch more
  near-duplicates. *Semantic duplicate similarity* is a similarity — **lower**
  it to catch more. They sit side by side and they move opposite ways, so each
  field spells its own direction out in a sentence next to the input.
- **When it takes effect.** Eight of them re-sort the bank the moment you save,
  because the scan stores raw measurements and the verdicts are recomputed on
  every read — no rescan, ever. The other four are baked into stored groups by a
  pass, so they carry a button that re-runs that pass on the spot. Re-grouping
  duplicates is cheap: it walks the stored hashes and decodes nothing.
- **How many images it would touch.** As you change a read-time value, the panel
  asks the server how many images that number *would* flag and shows
  `1 240 → 3 019 images flagged` before you save anything. Nothing is written
  until you press **Save**.

Every field has **↺ Reset to default** (it only appears when the value is not
the default), and the header carries **↺ Reset all to defaults**. The defaults
come from the server, so they are always the real shipped values.

### What editing an image costs it

Crop, ✂ Mirror, ↺ Rotate and the watermark cleaners **overwrite** the file the
trainer will later copy verbatim, so whatever they discard is discarded for good.
They all follow one rule: **keep the file's format and re-encode it without losing
pixels.** A PNG stays a PNG, a WebP is rewritten losslessly (crop it ten times and the tenth
is identical to the first), and the file keeps a name that matches what is inside
it. JPEG is the exception nobody can fix — it has no lossless mode — so a JPEG is
re-saved at the highest practical quality with no chroma subsampling rather than
converted to something heavier to protect pixels that were already lossy.

Two honest caveats:

- **A large crop still resamples.** A box longer than 1024 px is normalised *down*
  to a 1024 px long side, and only the *encoding* is lossless — that downscale
  never can be. A box at or under 1024 px is a pure cut, so it is lossless end to
  end, as is the watermark **✂ auto-crop**, which only cuts and never resizes.
- **Files get bigger.** A cropped photo that used to weigh ~200 KB now weighs
  ~950 KB. That is the price of not throwing pixels away. Thumbnails and the
  copies uploaded to a generation API are unaffected: they stay small on purpose.

### A crop is never enlarged

A crop used to be stretched *up* to a 1024 px long side as well: select 240×180
and the file stored was 1024×768. That enlargement invented no detail — shrinking
such a file back recovers the real crop almost exactly — and since the encoder
went lossless it cost roughly **6× the bytes** for nothing. A crop now keeps its
own size, and only comes *down* to 1024 px.

Two consequences worth stating plainly:

- **Your dataset can end up mixing image sizes.** That is fine — training buckets
  images by size — but a tile cropped out of a small area really does carry less
  detail than a native shot of the same framing, and it always did; it just used
  to look like 1024 px.
- **The composition meter says so.** The old ⚠ *Upscaled* line is now
  ⚠ *Under training resolution*. It fires on the same measurement and means the
  same thing it always meant: this framing bucket is filled by cropping far into
  photos rather than by native shots — add native shots for it. (Images imported
  with the automatic head-crop *are* still enlarged to 1024, so both shapes land
  under the same warning.)

Images cropped **before** this change keep the enlarged pixels they have.

Images you cropped **before** this changed keep the pixels they have — nothing is
re-processed retroactively, and re-cropping an already-degraded file cannot bring
back what the old encoder removed.

## Why a ↻ re-run button is greyed out

A bank runs **one pass at a time**. While a ✨ Score, a 🔎 Quality scan or a
🚀 Launch all is walking it, the ↻ buttons in this panel are disabled — and each
one says which pass is holding the bank and how far it has got, for example
*✨ Score pass is running on this bank — 137 / 412*. Wait for it to land, or
press **Stop** in the ⏳ progress bar at the top of the bank; the buttons come
back by themselves the moment the bank is free.

When a re-run does start, the button reports what the pass produced right where
you pressed it: **`Done — 12 duplicate groups · 34 images (was 9 · 26)`**. If
your new value groups exactly the same images it says so — *unchanged* — rather
than leaving you unable to tell a no-op from a pass that never ran.

## Rotate a sideways image

Scraped folders and phone exports are full of shots lying on their side. Both
places you meet an image can turn it a quarter turn, and neither charges you for
it. (Asked for by 1Tomber, GitHub issue #17.)

**In a dataset**, open the image (click its tile) and use **↺ Rotate left** /
**↻ Rotate right** in the bar under the picture, next to ⇄ Mirror. The file
keeps its name, its caption, its status and its format — a PNG stays a PNG, a
WEBP stays a WEBP. Four turns bring you back to exactly where you started:
measured on the shipped encoder, a PNG and a WEBP come back **byte-identical**
after going all the way round, so a mis-click costs nothing. The one exception
is a JPEG, which the format itself forces to be re-encoded on every save: at the
quality LDS writes (95, no chroma subsampling) that is around 46 dB PSNR — far
below anything visible, and it barely grows with more turns — but it is not
free, so it is worth knowing. Datasets normally hold WEBP, so this mostly
concerns files restored from an old backup.

Rotation is deliberately **not** part of ✂ Crop, even though that is where you
might look for it first. Cropping **resamples** the image — it rescales the box
you drew to a 1024 px long side — and resampling costs detail no matter how
carefully the result is then saved. A quarter turn resamples nothing at all: it
just moves existing pixels to new coordinates. Sending it through the crop lane
would make it pay a price it does not owe.

**In a bank**, your own folder is never written to — so a bank rotation does not
touch your files at all. The turn is remembered against the image and applied to
what the app shows you and to what it copies when you **⬆ Promote**; your
original keeps its exact bytes, whatever you do. Select the images and use
**↺ Rotate left** / **↻ Rotate right** in the selection bar to fix a whole
sideways batch at once, or turn one image without leaving **▶ Review** with the
↺ / ↻ buttons (keyboard: `[` and `]`). Rotating in Review never decides
anything — the image stays under your cursor so you can judge it once it is the
right way up.

One caveat worth stating: the analysis passes (👤 Subject, ✨ Score, 📐 Framing)
still read the original file, so turning an image does **not** re-run them. Turn
first, then run the passes if you want them to see it upright.

## Clean the watermarks a bank found

**🚩 Find watermarks** flags the images carrying an overlaid logo, URL or
@username. Removing them used to mean promoting the watermark into a dataset
first and cleaning it there; the bank now does it itself, in **two steps you
launch by hand** — cheapest and safest first:

1. **✂ Auto-crop** cuts off the marks sitting in a border strip. No model, no
   GPU, no invented pixel: it simply trims the band up to the mark, and only
   when the image stays big enough to train on. Anything it can't crop that way
   is left flagged, on purpose.
2. **🧽 Inpaint** repaints what's left. **LaMa** (fast, non-generative) handles
   small off-centre marks and leaves marks *on the subject* flagged; **Klein**
   (slower, via ComfyUI) also clears those. Each engine says what to install
   when it isn't ready, and the button stays off rather than failing mid-pass.

Each step shows how many images it still has to work on and how many it has
already handled, so you can see where the funnel stands. **Your source files are
never modified** — a cleaned image is a copy the app keeps beside the bank's
thumbnails. That copy is what the grid shows, and what **⬆ Promote** sends to
the dataset, so a cleaned bank produces a clean dataset. **↩ Undo cleaning**
just deletes those copies and flags the images again, and **👁 Before / after**
flips a sample between the cleaned version and your untouched original.

If a bank was scanned by an older version, its flagged images carry no recorded
mark position; the panel says so and one more **🚩 Find watermarks** run makes
them cleanable.

### Who decided an image is watermarked

**🚩 Find watermarks** can run two ways, and the panel says which one produced
the verdicts you are looking at ("Judged 1 240 by the detector, 300 by the vision
model") and which one a new run would use.

- **The vision model** — the way that has always worked. It asks the local vision
  model, in words, whether the picture carries a mark, once per image. About
  1.7 seconds each, so about fifteen hours on a 30 000-image bank.
- **The watermark detector** — an optional extra (Setup ▸ Quality tools). A small
  classifier scores each image in about **0.14 second**, and a second model marks
  where the logo sits so the two cleaning steps still have a box to work on. It
  needs no Ollama at all.

Install nothing and nothing changes. Install the extra and it takes over on its
own; there is no switch to flip. What it costs is ~0.9 GB of weights, downloaded
once into the same Python the **✨ Score** pass already uses.

**How good is it, measured.** On 110 images pulled from a real bank and labelled
by eye — half of them hard on purpose: faint corner logos, semi-transparent
handles across the subject, an `OnlyFans.com/…` line barely a few pixels tall, and
clean photos containing legitimate signage — the detector at its default setting
flagged **none of the 55 clean images** and **54 of the 55 marked ones**. The
vision model, on the exact same 110, flagged one clean image and missed one marked
one. So the detector is not a downgrade in judgement; the gain you actually buy is
the ten-fold speed-up. Neither is a verdict: both are a review flag, and both leave
your source files untouched.

The one image the detector missed was a `MET-ART.com` line in a bottom corner
scoring 0.929, just under the 0.94 cut — and the highest-scoring clean image sat
at 0.939. The two overlap by about a hundredth, which is why the cut is a
**setting** (Settings ▸ Captioning & quality ▸ *Watermark detector sensitivity*)
and not a constant.

Images flagged **without** a position — the detector was sure there is a mark but
could not place it — stay flagged and are counted separately in the pass's report.
Draw a zone on them with **🚩 Edit mask** below, or leave them as a filter.


## Fix a watermark mask in a bank

The detector draws **one** box, and it is a guess: it can miss a second logo,
swallow half the face, or land beside the mark. Open **▶ Review**, walk to a
flagged image and press **🚩 Edit mask** (shortcut `M`) — the same zone editor
the datasets use, on the bank image, right there.

- **+ Add zone**, then drag on the photo to draw a rectangle over the mark. Up
  to 32 zones; drag a zone to move it, its corners to resize.
- **Delete zone** removes the selected one, **Reset to detected** throws your
  zones away and puts the detector's box back.
- Every edit saves as you draw. If a save fails it says so and offers a retry —
  the zones on screen are never silently unsaved.

What the two cleaning steps then do with your mask:

- **🧽 Inpaint repaints exactly the zones you drew** — all of them, including a
  zone sitting on the subject, which is precisely what a hand mask is for.
- **✂ Auto-crop skips a hand-masked image.** A crop can only cut one border
  band; it cannot express several zones or a mark on the subject, so cropping
  the old box would remove pixels you did not point at.
- **An empty mask cleans nothing.** Delete every zone and you have said "there
  is nothing to repaint here": neither step touches that image, and the panel
  says how many are in that state instead of leaving them looking unhandled.

A flagged image an older scan left *without* a box becomes cleanable as soon as
you draw the zones yourself — that drawing is the missing information. And as
everywhere else in a bank, **your own file is never modified**: cleaning writes
a separate copy. A rotated image is shown unrotated here, because the whole
watermark lane works on your original file, which the ↻ turn never changed.


## Reject every flagged image at once

In a dataset, **🧽 Find watermarks** flags the kept images that carry an overlaid
mark. The recommended way through the pile is **🔍 Review flagged**, one image at
a time — the detector is a review flag, not a verdict, and it *does* flag clean
images sometimes. When you would rather drop the whole pile and move on,
**✕ Reject all flagged (N)** does exactly that.

Four things worth knowing before you click it:

- **The number is the number.** `N` is what the button will really reject, not
  how many are flagged. Small-image rescue pairs are excluded (the server refuses
  a batch containing one, so including them would reject *nothing*) and failed
  rows are excluded (the server skips them). If the two differ, the row says so
  in plain text rather than showing you the bigger figure.
- **Nothing is deleted.** Rejected images stay on disk and simply leave the
  training set. To bring any of them back: **Show ▸ Rejected** in the grid,
  select, then **✓ Keep**.
- **It clears the watermark flags.** That is the one thing rejecting destroys:
  after the click, 🔍 Review flagged is empty and nothing records which images
  had been flagged. Re-run 🧽 Find watermarks to flag them again.
- **Stop is available while a scan runs.** The ⏹ Stop button in the progress
  banner ends the scan at the next image; everything already judged is kept, and
  running 🧽 Find watermarks again finishes the rest.

Which engine does the flagging is a setting — **Settings ▸ Captioning & quality ▸
Watermark detection** — and it applies to datasets and banks alike. *Auto* uses
the optional watermark detector when it is installed and the vision model
otherwise, which is what the app has always done. Pin *Watermark detector*
without the extra installed and the scan still runs, on the vision model, and
says so with the link to install it. Only the detector can flag an image
**without a position**; those are counted apart, 🧽 Clean leaves them alone, and
you can draw the zone in 🔍 Review flagged. Images you dismissed as false
positives are skipped by every later scan — **⟲ Rescan incl. dismissed** is the
only way to have them judged again, which is what you want after changing engine.


## A bank and a dataset never share files

A dataset and an image bank can hand images to each other in both directions,
and both directions **copy**. That is not an implementation detail — it is the
rule the whole flow rests on:

The files generated for **ai-toolkit are not LDS's dataset registry**. At launch,
LDS freezes a disposable training export (kept images, captions and a freshly
generated job config) from its own Dataset rows. Bank/Dataset identity, analysis
history and comparisons stay in LDS's database plus its SHA-bound snapshot/cache
sidecars; they are not reconstructed from an old ai-toolkit config file.

- **Bank → dataset** (**⬆ Promote…**) writes new files into the dataset.
- **Dataset → bank** (**🗃 Import to bank**, on the dataset) copies the dataset's
  kept images into a folder of the bank's own. Both choices retain the
  Dataset-owned captions, keep/reject curation, framing, watermark and
  provenance. Its dialog defaults to **Reuse compatible final-file analysis**;
  **Start fresh analysis** skips only reuse of prior analysis, not that metadata.
  The AI **Face**, **Score** and **SigLIP 2 semantic** results are not reused after
  normalization or another transformation because they are no longer proved.

Neither ever *points* at the other's files. The reason is that the two containers
have opposite contracts. A dataset **owns** its images; a bank merely **points**
at a live folder it does not own — which is exactly why 🗑 **Delete rejected** is
allowed to remove files from it. Put a bank on a dataset's folder and that button
stops deleting your rejects and starts deleting the dataset's training images.

So the app refuses it. If you paste a dataset's image folder into **➕ Create
bank** — or into **📦 Move folder…** for an existing bank — you get a refusal
that names the dataset and points you at **🗃 Import to bank** instead. The check
looks through the disguises: a subfolder of the dataset, the folder *containing*
all datasets, a different letter case, forward slashes instead of backslashes,
and symlinks or Windows junctions that resolve to the same place.

**If you already have such a bank** (it was possible before this check existed),
nothing is repaired or deleted behind your back. Opening it shows a red banner
naming the dataset, and 🗑 Delete rejected is refused on that bank — everything
else keeps working, so you can finish triaging. When you are ready, either
**📦 Move folder…** to point the bank at a folder of its own, or remove the bank
(removing a bank never touches files).

The dataset's own folder is shown at the top of the dataset, with a **⧉ Copy**
button, so you never have to go hunting for it in a file manager — which is how
this trap was found in the first place.


## Move a bank folder to another disk

A bank points at a folder *in place*, but nothing it computes lives in that
folder: the quality scores, duplicate groups, face clusters, captions and every
keep/reject decision are stored against the image row, and each row remembers
its file *relative* to the bank's folder. So moving a 30 000-image bank to
another drive costs nothing — you just have to tell the app where it went.

You can do this in either order. **📦 Move folder…** sits in the bank's header
next to its path (and **📦** on the bank's card in the list), so you can open it
before touching anything to see what the app will ask for; it also appears inside
the warning shown once the app notices the folder is gone, if you moved first.
Paste or browse to the new folder
and press **🔍 Check this folder**. Nothing is written yet: the app walks the
candidate folder and tells you how many of *this bank's* images are in there and
how many are not. Paste it however you like — Windows' *Copy as path* wraps the
path in quotes, and a trailing `\` or forward slashes are equally fine; the field
then shows the folder the app actually resolved, so what you confirm is what it
will use.

- **All of them found** → confirm, and the bank is repointed with every score
  and decision intact.
- **Some found, some missing** → you can still confirm. Nothing is deleted:
  rows whose file didn't come along keep their analysis and simply read as
  missing until the file comes back.
- **None found** → refused. That folder is a *different* folder, not a moved
  one — the usual cause is picking the parent of the folder you moved.

The app never deletes a row on its own, and an analysis pass run while the files
are away no longer degrades them either: a file that is *absent* is not a file
that is *broken*, so the pass stops and tells you the folder appears to have
moved instead of marking thousands of images unusable.


## Make Score use a GPU Python you already have

The **✨ Score** pass (aesthetic · NSFW · style) runs in its own small Python
environment, and that environment deliberately carries **CPU-only PyTorch**: a
first install stays a few hundred megabytes instead of pulling ~2.5 GB of CUDA
wheels onto machines that may have no card at all.

On a machine that *does* have one, that default is expensive — CLIP measures
about **336 ms per image on the CPU against ~15 ms on a recent card**, so a
30 000-image bank is the difference between a coffee break and most of an
afternoon. The bank says so: when Score is about to run on the CPU on a machine
with an NVIDIA card, an amber note gives you the estimate and a button, **⚡ Use
a GPU Python I already have**.

That button is the point. If you train LoRAs or run ComfyUI, this machine
*already* has a PyTorch with working CUDA. Score can simply borrow it — no
download, no third environment to maintain.

The dialog lists the interpreters the app knows about (the environment it built
for scoring, ai-toolkit's, ComfyUI's, its own) and reports each one **package by
package**:

- **GPU ready** — everything the pass imports is there *and* PyTorch sees the
  card. Pick it and the next Score run is minutes instead of hours.
- **Missing packages** — the reason is named. The common one is an interpreter
  with a perfect CUDA PyTorch but no **OpenCLIP**: Score needs `open_clip` and
  `transformers`/`timm` too, so CUDA alone is not enough. Such an interpreter is
  **refused**, on purpose — accepting it would trade slow-but-working scoring for
  an import error an hour into the pass.
- **CPU only** — it can run the pass, it just has no usable CUDA.
- **No answer** — the path is not a working interpreter (moved venv, unplugged
  drive). Nothing changes.

**The app never installs anything into an environment it did not create.** Your
ai-toolkit venv runs your training and ComfyUI's runs your generation; a silent
`pip install` into either is not something a dataset tool gets to do. When a
package is missing the dialog shows you the exact command and leaves the choice
to you — run it in a terminal, then hit **↻ Check again** and the row updates.

**Not listed? That field is not a fallback.** Most machines have neither
ai-toolkit nor ComfyUI where the app looks — or at all — so entering a path
yourself is a first-class route, checked exactly the same way. Paste an
interpreter *or* the environment folder that contains it: a venv, a conda or
miniconda env, a uv venv, a portable bundle, the system Python, something on a
second disk. Spaces, accents and quotes around the path are fine ("Copy as path"
on Windows wraps it in quotes; that is handled). The layout is never assumed —
the app knocks on the shapes an environment can have and keeps whichever one
actually answers.

No version of PyTorch or CUDA is required. The only question asked is the one
that matters: do the packages import, and does PyTorch see a card. An old card
on cu118, a 50-series that only works on cu128, a nightly build — all fine.

**No NVIDIA card?** Then there is nothing to fix, and the app says so plainly
instead of suggesting a CUDA install you could not use. Borrowing an interpreter
is still offered, for one honest reason: if another Python here already has the
packages, you can skip installing them a second time. It will not be faster.

**Back to the app default** puts everything back exactly as it was. The choice is
reversible at any time, and the note under the passes always says which
interpreter is in use. If you never open this dialog, nothing changes: an install
that works today keeps working, untouched.


## Build the SigLIP 2 index on a GPU Python you already have

The **SigLIP 2** semantic engine is the same story with a different dependency
list. Its index is built by a worker that lives in the app's own environment —
the CPU-only one — so on a machine with a card the index crawls for the same
reason Score used to.

SigLIP 2 is the lighter of the two: **92.9 M parameters against 303 M for the
CLIP ViT-L/14 Score runs**, measured at about **105 ms per image on the CPU**
rather than 336. Lighter is not free: a 30 000-image bank is still the better
part of an hour.

The **Semantic engine** panel now tells you which device the index will actually
use, and when a card is sitting idle it offers the same button, **⚡ Use a GPU
Python I already have**. It is the same detector, the same dialog and the same
promise — with one difference that matters:

**The dependency list is SigLIP 2's, not Score's.** The semantic worker never
imports `open_clip` or `timm`. An interpreter Score refuses for a missing
OpenCLIP — the most common shape of a ComfyUI venv — can be perfectly good here,
and refusing it would be a lie about a worker that does not need it. What it
*does* need is a **Transformers recent enough to carry `Siglip2Model`** (4.49 or
newer). That one is checked by really looking for the class, not just for the
package: an older `transformers` imports fine and then dies at model load, an
hour into an index. Such an interpreter is refused, and the repair line the
dialog hands you carries the version floor.

**Borrowing an interpreter downloads nothing here.** The pinned SigLIP 2
checkpoint lives in the app's own data folder, not inside the interpreter, so a
borrowed Python needs no copy of it.

**Where the index runs is not where anything is installed.** Setup ▸ Quality
tools always installs SigLIP 2 into the environment the app built, whatever you
picked in this dialog — including when you later hit Install/repair, which now
*keeps* your choice instead of quietly putting the index back on the CPU.

Score and the semantic index are chosen separately. Pointing one at an
interpreter never moves the other, and **Back to the app default** undoes either
on its own.

## The video bank (turn a folder of rushes into shots)

Videos are a different kind of material and they get their own bank. On the
**🗃️ Bank** page the switch at the top right says which kind you are making —
**🖼 Images** or **🎬 Video**. This matters more than it looks: an image bank
skips every `.mp4` you drop into its folder **without a word**, so a folder of
video used to look like an empty bank.

A video bank triages **shots**, not files. One two-hour rush is not something you
can judge; the three hundred shots inside it are.

1. **Create it** — name it, point it at the folder. Every `.mp4`, `.mov`, `.mkv`,
   `.webm` and `.avi` under it (subfolders included) is inventoried in place.
   Nothing is copied and the folder is never modified.
2. **▶ Run everything** chains the three passes in the only order that works:
   **scan** reads what each file is (length, size, frame rate), **find shots**
   cuts it at its shot boundaries, and **make thumbnails** grabs one frame from
   the middle of each shot. Each pass is also available on its own, and the box
   above the buttons always names the one step to take next — run them out of
   order and each simply finds nothing to do and reports success.
3. **Triage** — the grid is thumbnails, and only thumbnails. Click one to watch
   exactly that shot, `←`/`→` to move, `K` to keep, `R` to reject. Filter by
   status, or click a file in the **Files** list to see only its shots.
4. **🎬 Build the dataset** encodes what you kept. This is the only step that
   writes video.

**Nothing is encoded while you triage.** A bank stores where each shot starts and
ends — no clip file exists until you promote — which is why a bank of hundreds of
shots costs no disk space, and why the player streams the original file rather
than a preview.

**A missing piece never disables the whole lane.** The video extra is three
independent things: reading files, finding shots, and encoding clips. The app
says which one is missing and what still works — with no ffmpeg, for example, you
can scan, cut, watch and triage an entire bank, and only the final build waits.

## Measure your shots, and choose your own cuts

**📊 Measure quality** reads every frame of every shot in one pass and scores the
four things that quietly ruin a video dataset: shots that barely move, shots that
are all blur, black moments, and frozen stretches. The pass stores raw numbers,
never verdicts — so changing a threshold later re-sorts the bank instantly, with
no rescan. Stopping is safe; a re-run picks up where it left off.

Flagged shots get an **amber ⚑ mark** in the grid. Amber, not red, because a flag
is a reason to *look* — nothing is ever rejected for you. Hover the mark to see
which cuts a shot tripped.

**There are deliberately no default thresholds.** The same number that flags 2 %
of one bank flags 12 % of another — a cut only means something against *your*
bank's own distribution. Open **🎚 Quality cuts**, type a value (leave a field
empty to disable that cut), and press **👁 Preview**: it answers with how many
shots each rule would flag, per rule, before anything is applied. If a draft
would flag most of the bank, the preview says so in as many words instead of
letting you apply it by accident.

**One cut needs no measuring at all: Minimum length.** Shot detection keeps very
short cuts on purpose — a real flash cut is a real shot, and a detector that
refuses to emit one also hides genuine boundaries. The cost is a grid peppered
with half-second shots you scroll past a hundred times. Type a value in seconds
and every shorter shot wears the flag, immediately after detection, with no
measuring pass — this cut reads the shot's own bounds rather than its pixels.

Do not confuse it with the *too short* refusal you may see at promotion. That one
is your target profile's arithmetic — so many frames at so many fps — and no
setting on this panel moves it: those shots were never going to land. **Minimum
length** only decides what gets flagged for your eyes, so you can see and sort
them *before* spending triage time on them.

Two touches you get for free once shots are measured: thumbnails move from the
middle-of-shot guess to the **sharpest measured frame**, and the freeze detector
catches the failure the averages never can — a shot that plays fine and then
hangs on a still image for a second. On a real 4.5-hour test bank that turned
out to be the most common defect of all.

**The sound is measured too, for the targets that keep it.** LTX and MiniMax H3
mux the source's audio into every clip; Wan has no audio at all and forces the
track off. So the pass also reports, per shot, **how much of it is silence** and
**its overall level in dBFS** — because a dataset of silent clips teaches the
model to be silent, and nothing about the file on disk reveals it: it is the
right length, the right sample rate, and mute. Two cuts go with them, **Silent
share** and **Loudness floor**, and they raise two different flags on purpose —
a quiet clip can be normalised, a silent one cannot be rescued.

Three states are kept apart here, and it matters:

- **no sound track** — a property of the file. Never flagged; a Wan dataset is
  supposed to look like this.
- **silent** — a track that is there and carries nothing. That is the defect.
- **not measured** — nobody has listened yet. Shots measured before this shipped
  carry no sound reading at all, and an audio cut will never flag them. **Run
  Measure again with re-measure** to fill them in; the pass otherwise skips
  everything it has already done.

## Retouch a cut: trim, split, or draw a shot by hand

Shot detection is good and it is not right. It cuts a slow dissolve a second
early, and it happily hands back a shot whose last second is a frozen frame.
Before this panel existed the only gesture available on either was **✕ Reject** —
throwing away eight good seconds to be rid of one bad one.

Open any shot and unfold **✂ Trim & split this shot**, under the player:

- **Nudge either bound** by 1 s or by one frame. One frame means one frame *of
  your source file*, at its own rate — a 25 fps rush steps by 0.040 s, a 59.94 fps
  one by 0.017 s. The frame counts your target model wants are a different thing
  entirely, decided at build time.
- **⇤ playhead** snaps a bound to wherever the video is paused. Scrub to the frame
  you want, click, save.
- **✂ Split here** cuts the shot in two at the playhead. The half you were looking
  at keeps its triage decision, and so does the new one: split a *kept* shot and
  both halves stay kept, so you never have to find them again among hundreds.
- **＋ New shot from here** draws a shot the detector missed entirely. The player
  is pointed at the whole rush, so you can scrub anywhere in the file — not only
  inside the shot you opened — and mark a boundary that was never found.

**For image-to-video targets, the first frame is the conditioning image.** The
trainer conditions an i2v sample on the clip's *first* frame, so moving a start is
not trimming: it is choosing the exact picture the model learns to animate from.
If the first second of a shot is a dissolve, an i2v LoRA trained on it learns to
animate dissolves. The panel repeats this line where the buttons are.

**A re-cut shot loses its thumbnail and its quality scores, on purpose.** They
were measurements *of the old bounds* — a thumbnail showing a frame the shot no
longer contains is not stale, it is wrong. The tile goes blank and the bank's
next-step line offers **🖼 Make thumbnails** again; run it once when you are done
cutting rather than after every edit.

**Limits worth knowing.** A shot must last at least 0.5 s, and both halves of a
split must too — the buttons say so rather than silently clamping. Retouching is
refused while a pass is running on the bank (a thumbnail pass mid-edit would
produce a picture of the old span marked as current), so stop the pass first.
Re-detecting a file deletes the shots the detector drew and **never** the ones you
cut by hand; those stay, and may overlap the fresh ones. And editing a shot that
is already in a built dataset is allowed: the dataset stored its own copy of the
bounds when it was encoded, so nothing already on disk changes.

## Find scenes in a video bank by typing a word

A folder of rushes is a haystack whose needles have no names. The quality cuts
tell you which shots are sharp and which move; they cannot tell you which one has
the red car in it. **🔎 Find scenes** does: type *a woman walking on a beach* and
the gallery is replaced by the shots that look most like it, best first.

**Run the pass once, search as often as you like.** The 🔎 Find scenes button
looks at a few frames of every shot and remembers what they look like. It is the
slow part — it needs the same environment as the image bank's ✨ Score (Setup ▸
Quality tools, or a Python you already have with torch and open_clip), and on a
CPU it is minutes rather than seconds. Every search afterwards is instant and
costs nothing.

**Several frames per shot, not one.** A shot is a span of time, and a thumbnail is
one instant of it. If a car only drives into view in the last second, a search
that had looked at the opening frame would never find that shot — and would give
you no hint that it had missed it. So each shot contributes a frame near its
start, its sharpest frame, and one near its end, and a shot's score is the best of
the three. Every result tells you **which second matched**, and opening it starts
the player right there.

**It is a ranking, not a filter.** Every shot scores something against every
phrase, so the results always come back full, however wrong the query. The line
above the gallery says how strong the top and bottom of the ranking are, and how
many shots could not be searched at all — a shot the pass has not reached cannot
be found by any phrase, and it would be easy to conclude the scene simply is not
in the bank.

**What it cannot do**, measured on the model this app uses:

- **“Without” is ignored, not honoured.** Ask for *a street without cars* and you
  get cars. Type `-cars` instead: that subtracts the unwanted thing from the score
  and pushes those shots down the ranking. It cannot promise their absence, and
  the panel says so rather than pretending otherwise.
- **It cannot count.** *Two people* barely outranks a picture of one.
- **It cannot hear, and it cannot see motion.** Only still frames are looked at,
  so *a door slamming* or *panning left* describe nothing it can use.
- **Left and right carry almost no meaning.**

Searching respects the triage filter you are on, so *keep only* plus a phrase
ranks what you already decided to keep. Changing the filter clears the search: a
ranking computed over one bucket has nothing to say about another.

## Describe your shots, and search what happens in them

🔎 Find scenes ranks by what a moment **looks like**. It cannot find an action —
"turns and walks away" is a fact about *time*, and no single frame carries it. The
**🗣 Describe shots** pass closes that gap: it watches eight frames spread across
each shot and writes one or two sentences about what happens in it.

That line does two jobs, and the second is the one nobody sees coming:

- **It is what the clip trains on.** At promotion each clip gets a `.txt` sidecar
  next to it, and that file *is* the prompt. Before this pass existed every
  promoted clip shipped with an **empty** one — which the trainer accepts in
  silence, training the clip on no prompt at all. The build dialog now tells you
  how many clips are about to go out uncaptioned, before it encodes anything.
- **It makes the search read words as well as pixels.** Once captions exist,
  typing a phrase ranks on both, and the panel says which halves are running so
  that "nothing found" can be read correctly.

**Captions are drafts.** Open any shot and edit the caption under the player; a
bulk re-run will never overwrite one you wrote. Clearing it puts the shot back in
the queue. Regenerating over your own words is possible, but you have to ask for
it by name.

**You can change how plainly they are written.** Next to the button there is a
**Caption wording** choice: *Standard* (the shipped wording) or *Plain*, which
gives the model explicit permission to name what is on screen instead of
describing around it. On adult footage that difference is not cosmetic — a
captioner asked the standard way produces captions that are *about something
other than the shot*, and a LoRA trained on those learns the evasion. It was
measured rather than assumed: the wording turned out to matter **more than the
model**, and the stock model asked plainly beat an uncensored one asked the old
way. Every caption records which wording produced it, and the choice is
remembered as `video_caption.style` if you set it in your config.

**You can change which model writes them.** The pass ships with one checkpoint
and uses it unless you say otherwise (`video_caption.model` — see *Settings
reference*). It is worth changing when the default **talks around** what your
footage shows: a caption that names things evasively is not a style choice, it
teaches the trained model to look away too, and the captions read perfectly well
while being about something slightly other than the shot. Any checkpoint of the
same architecture is a drop-in. If it is not on your machine, the first run
downloads it — and the pass says so in its progress line before captioning
anything, rather than sitting at 0 % while gigabytes arrive. Every caption
records which model wrote it, so a bank captioned across a change stays readable.

**It needs the same environment as ✨ Score** (torch + transformers) and it uses
the GPU when there is one — a 4B vision model on a CPU is minutes per shot. It
will not start while a training run owns the card, and stopping is safe: what is
captioned stays captioned and the next run picks up where it left off.

## Video training sets (and the two things to check before you cut one)

Promoting a video bank builds a flat folder of clips with a `.txt` caption next
to each one, and lists it in your library under **🎬 Video training sets**.

**You can cap how many clips one source contributes.** A 50-clip set that is
three videos over-represented looks exactly like a diverse one on disk, and that
imbalance is the kind that quietly overfits a source. **Max clips per source**
caps it; leave it empty for no cap. The cap trims dominance without punishing
scarcity — a file with fewer clips than the cap keeps all of them — and it is
**not a random sample**: each source keeps its earliest clips, so promoting the
same bank twice gives you the same dataset. When a finished set leans on one file
anyway, the result tells you the real share.

**You can trim the edges of every clip.** A shot boundary is where a cut just
happened, so the first and last frames of a shot are disproportionately
dissolves, fades and leftovers of a transition — and a dataset whose clips all
open on half a dissolve teaches the model to open on half a dissolve. **Trim each
end** takes a number of seconds off *both* bounds; 0.25 is the common figure, and
the default is 0 so an existing recipe exports exactly what it exported before.

The trim never shortens a clip. Frame counts are a property of the target's VAE,
so a clip that no longer supplies the count is **dropped, not exported short** —
ffmpeg would write the short file and exit 0, and ai-toolkit would train it as
repeated stills without a word. The dialog says how many clips the trim will cost
*before* you press the button, and those are counted separately from clips that
were never long enough: only the first kind is fixed by lowering the trim.

**The clip length is chosen in FRAMES, from a menu.** That is not pedantry: the
legal frame counts are a property of each model's VAE, not of video. 29 frames is
legal for Wan and illegal for LTX; MiniMax H3 wants counts of the form 17n+5. No
trainer refuses an illegal count — they round it down in latent space and say
nothing. So the menu offers only counts the target can actually ingest, with the
duration shown next to each at that model's own frame rate.

Two labels sit next to every target, and both are there to save a wasted week:

- **Not trainable yet** — the app knows the model's geometry perfectly and no
  LoRA trainer for it is known to exist. Exactly one target of the four currently
  clears that bar (Wan 2.1 / 2.2 14B). You can still cut a dataset for the
  others; just know that today nothing is known to train on it.
- **Licence limits** — MiniMax H3's Community Licence grants rights **only**
  inside an "Applicable Territory" that excludes the EU, the UK, South Korea and
  the USA. The restriction covers the **outputs**, not just the model, so keeping
  your training private is not a way around it. Check your territory before you
  build the set, not after.

Deleting a video dataset deletes the encoded clips and nothing else: the bank
keeps every shot and every decision, so you can re-cut at another length or for
another target without triaging again.

## Stopping Score, and what a relaunch costs

**✨ Score** always covers the whole bank — but it only *computes* what it does
not already have. Every image it scores is written to a cache next to the bank
(the CLIP embedding plus the aesthetic and NSFW numbers), and a relaunch reads
that cache and pays only for the rest. On a bank that is fully scored, the pass
does not even load the model: it goes straight to the grouping.

So **Stop is safe**, and it is now safe in the database too. When you stop a run,
the scores it had already computed are written to your images before the pass
ends — that work was paid for, and it used to reach the cache and never reach a
single row. The line at the end of the pass says exactly what happened: how many
images were scored, how many remain, and how many were reused instead of
recomputed.

One thing does *not* survive a stop: the **🎨 style groups**. Those ids are not a
per-image measurement, they are a single numbering of the whole bank, computed
from every embedding at once and renumbered on each pass. Half of one is not
partial progress — it would put a new group 1 next to an old group 1 and mix two
unrelated styles under the same chip. So a stopped pass leaves the previous
grouping alone and says so. Relaunch and it finishes: the scoring part is already
cached, and only the grouping is left. That grouping is the slow tail of the pass
— about **8 seconds over 5 000 images and 3 minutes over 23 000** — so on a big
bank it is worth letting it finish.

**Rescore all** is the last line of ✨ Score's launch window, unticked. It is the
opposite intent: throw the cache away and recompute everything, for a bank you
scored with a different setup or whose results you no longer trust. It costs a
full pass, which is why it is a deliberate tick and never a default — ✨ Score
itself has always meant "cover the whole bank", and it still does.

One more thing a relaunch fixes on its own: if the aesthetic head or the NSFW
model could not be downloaded during an earlier run, the images scored in that
window carry a hole. They are picked up again the next time you run Score, once
the missing piece is available — an image is never left permanently half-scored
because a download failed once.

## The LoRA Canvas (every run on one board)

**Canvas** in the top bar opens a single board holding the training history of
every dataset you have. Each dataset gets a lane; inside a lane, each run is a
card and each save it wrote is a small pill underneath it. When a run continued
from an earlier one, the line between them starts at the *exact* checkpoint it
resumed from — so "where did this LoRA come from" is a thing you read, not a
thing you reconstruct.

**Choosing what is on the board.** Everything is on it by default. Above the
board sits a single row of filter chips, about 40 px tall — it used to be a
fold-out panel, and unfolded on a library of fourteen datasets it stood 389 px
on a 720-px screen, more than half the window, directly above the thing you came
to look at.

- **Datasets** opens a menu with a search box, **Select all** / **Clear**, and
  one checkbox per dataset with its run count. The search matches the name *and*
  the model family, so typing `krea` brings up every Krea lane.
- **Models** and **Status** are the same idea for the model family and the run
  state (Active, Completed, Errors, Unknown).
- **Pinned** toggles the pinned images on and off. Turned off it goes amber:
  pinned pictures missing from the board with no visible cause is a bug report
  waiting to happen.
- The **search box** stays at full size in the row — it filters the *runs* on the
  board (dataset, run ID, model, variant), which is a different question from
  "find me a lane to tick".
- **Reset** puts everything back, and goes dim when there is nothing to reset.

Every chip carries its own count and lights up while it is narrowing something,
and the row ends with **N runs shown** — so a filter you set and forgot can never
empty your board without saying why. Your choices are remembered between visits.

**Saving an arrangement.** **💾 Layouts** in the board toolbar keeps where every
run card and every pinned picture sits, under a name, and puts it back later —
closed pictures included. Until this existed, the only way out of an arrangement
was **✦ Tidy up**, which throws it away. A run deleted since the layout was saved
simply is not restored, and the app tells you how many were missing rather than
leaving you to hunt for the card that did not come back.

**Exporting the board.** **📷 PNG** writes the whole canvas to one image file:
every pinned picture at full size, every run card with its checkpoints, and the
lines that join them. It is a redraw rather than a screenshot, so the buttons,
badges and hover highlights are not in it — and a picture whose file has been
cleaned off the disk comes out as a labelled placeholder rather than silently
missing.

**Deleting a picture from the board.** A pinned image carries **✕** and **🗑**,
and they are not the same thing. **✕** takes it off the board and remembers where
it was, so re-pinning it from its gallery puts it back at the same spot and size.
**🗑** deletes the image itself, through the same route (and the same
recoverable-or-not setting) the gallery's own delete uses; it arms on the first
press and deletes on the second, because a delete one tap away from ✕ on a small
control is a delete that happens by accident.

**Zoomed out.** Below 55 % zoom each run card carries its run number at a
constant, readable size, and below 30 % the dataset name comes with it. A board
of a dozen lanes is read at 30-40 %, where a card's own title is about four
pixels tall.

**Moving around.** Drag the background to pan, use the wheel (or two fingers) to
zoom, and **Fit** puts the whole board back in view. The board only fits itself
automatically until you first touch it — after that a dataset finishing its load
never yanks your view away.

**Moving something counts as touching it.** Zooming and panning are not the only
way to take the view over: the first time you drag a picture or a run card to a
new place, the board stops re-framing itself for good. Placing a render far from
its lane makes the board bigger, and an automatic fit at that moment zoomed the
whole plateau out the instant you let go — your framing thrown away by the very
act of tidying. **✦ Fit** is still one click away whenever you *do* want the
whole board back; it simply is not decided for you any more. A board you have
never arranged still opens fitted, as it always did.

**The reference face.** A character dataset's lane opens with its reference
image, next to the dataset name — the person the renders on that lane are meant
to be. Click it to open it full size against them. It is part of the lane label,
not a pinned picture: it cannot be moved, closed, grouped or exported. Concept
and style datasets show nothing there, because they are not built around a
reference face.

**Reading a run.** Click a run card to open **everything that run produced**:
its images grouped by the checkpoint that made them, most-trained step first, so
you can see where the LoRA stopped getting better without opening one pill at a
time. Underneath the images are the run's note, its per-checkpoint notes, and the
settings it trained with. **ⓘ Full details** opens the drawer where those notes
can be edited.

A run with many checkpoints opens with its three most-trained steps expanded and
the rest folded behind their image counts — tap a step to unfold it. When a run
holds more images than one panel should carry, the panel says so rather than
looking complete; the missing ones are still reachable from each checkpoint's own
pill and in the Test Studio.

Sometimes a step reads **Step unknown**. Those are older test images whose file
name identifies the run but not the checkpoint inside it, so they belong to the
run and to no pill. Images that identify nothing at all are still counted in the
footnote at the bottom of the panel — they live in the Test Studio.

**Shift-click two** run cards to compare their settings side by side, with the
differences highlighted — and because every dataset is on the same board, those
two runs no longer have to belong to the same dataset. Dragging a card to
rearrange the board never opens the panel.

**Arranging the board.** Drag a run card and it stays where you put it, across
reloads. On a phone, moving a card and scrolling the board are the same gesture,
so a card is picked up with a **long press** — rest your finger on it for a
moment and it lifts; a finger that slides straight away scrolls as usual.

Once you have moved anything in a lane, that whole lane stops rearranging itself:
a training run that finishes later lands in free space next to your layout
instead of pushing everything sideways, which is what would otherwise happen —
the automatic tree centres each run over its continuations, so one new branch
re-flows the lane around it. Lanes you have never touched keep following the
automatic tree, because there is no arrangement to protect there.

**✦ Tidy up** is the way back: it forgets every card you have moved on the lanes
currently shown, rebuilds the automatic tree, and brings every pinned picture
back beside the run that made it — including one you dragged clean off its lane.
Positions are only ever a display preference — moving a card or a picture never
changes which run continued which or which checkpoint made which image, and Tidy
up never deletes a run, a checkpoint, a note or a picture.

**Generating from the board.** Every checkpoint pill carries a small **✓** box.
Tick one and the run settings open beside the board: the prompt, the seed, the
format, the steps, the engine settings — the Test Studio's own panel, not a
lookalike, so anything the Test Studio can do the board can do too.

What the board adds is that your picks do not have to belong to the same
dataset. Tick a checkpoint in one lane and two in another and they run together
on one shared prompt and one shared seed, which is the only honest way to
compare LoRAs against each other.

Two things it will tell you rather than fail at:

- **A checkpoint that is not in ComfyUI yet** is still pickable. The button then
  says what it is about to do — *"Deploy 2 checkpoints, then generate"* — and
  waits for you. Nothing is copied into your ComfyUI folder by a button that did
  not announce it, and if a copy fails, nothing generates: half a comparison
  answers a different question than the one you asked.
- **Two different families in one selection** (say Krea and Z-Image) is refused,
  and it says which two. This is not a restriction we chose: those families do
  not share a base model or a workflow, so there is no single run that can render
  both. Unpick one family and the button comes back.

**⚖ Compare or 🧬 Blend.** From the second pick onwards the panel offers a
choice, and it defaults to what it always did:

- **⚖ Compare** — one pass per checkpoint, swept across the strengths. This is
  how you find out which LoRA, or which step, is better.
- **🧬 Blend** — *one* generation loads them **all**, each at its own weight, and
  every dataset's trigger word is added to the front of your prompt. The panel
  lists those words before you launch; nothing is injected silently. It is the
  Test Studio's Blend mode, driven from the board — the same toggle, the same
  engine. (The Test Studio called it **🧬 Combine** until August 2026; only the
  name changed.)

A blend is one configuration, not one per pick, so the strength sweep disappears
(each LoRA carries its own weight instead) and the image counter drops to one
picture per seed.

**Trying several weights at once.** Each picked checkpoint has a row of weight
boxes under its slider. Tick two on one and two on another, and the launch
renders **all four combinations** in a single run instead of making you launch,
look, move a slider and launch again. Every image is labelled with the pair that
produced it. Tick nothing and the slider governs, exactly as before; the slider
is also how you use a weight that is not on the grid.

The panel counts the cost before you commit — "4 weight combinations → 4 images,
about 1 min" — and turns amber past 24 images. It does not refuse: the queue is
serial and the machine is yours. Two checkpoints at four weights each is 16
images, which is exactly why the panel does the multiplication for you.

What blending actually does is worth saying plainly: **two identity LoRAs give
you a hybrid person** — someone who is neither of the two. That is a real use, on
purpose, but it is not "both people in one shot". The combination that usually
pays off is **identity + style**, or **identity + concept**. Weights are the dial:
below 1 the LoRA contributes less, above 1 it dominates (0 to 2, 1 by default),
and a weight you set survives un-ticking another pick or reloading the page.

Blend needs **at least two checkpoints of one family**; with a mixed selection
the toggle is greyed out with the reason, because the run underneath it could not
exist either. Picks that are not deployed yet are deployed first, all of them,
before anything is generated — a blend never loads a subset of what it announced.

**▶ Continue training from a checkpoint.** Clicking a pill's body opens its
actions — Download, Deploy, Details, Delete — and **▶ Continue from here**. It
opens the *same* launch dialog the Checkpoints panel and the Runs page open, on
*that exact save*: pick where it runs (**💻 Local** or **☁ Cloud**), how many
extra steps, and — folded under *Adjust settings* — the checkpoint cadence, the
preview prompts, the timestep weighting and the learning rate. Rank, base and
optimizer are locked to the checkpoint being continued; they are not things a
resume can change.

The dialog also names **what “resume” means**; it never silently guesses:

- **Full training state** is offered only for a local checkpoint carrying a
  complete, hash-verified state bundle. It restores the raw adapter parameters,
  optimizer, scheduler, scaler, EMA, Python/NumPy/Torch/CUDA random generators,
  dataloader order and cursor, bucket/crop geometry, the exact latent/text-cache
  bytes, and the exact next step. Exported image, caption and mask contents,
  dataset topology, base, network shape, training recipe, ai-toolkit revision,
  GPU identity and the complete installed Python-package map must still match.
  In this mode only the preview prompts can change. Save/preview cadence,
  learning rate and timestep weighting stay locked because changing any of them
  would change the trajectory the state belongs to.
- **LoRA weights only** is the explicit fallback and is available for legacy
  checkpoints. The chosen `.safetensors` is copied into a clean run folder;
  optimizer, scheduler, scaler, RNG and dataloader progress restart. The source
  run is renamed aside, not deleted, so all its saves remain recoverable.

Each checkpoint says why full state is unavailable when its bundle is missing,
incomplete, corrupt or incompatible. Cloud continuation currently supports
weights only; choosing full state there is refused before a pod is created.
State bundles are published atomically and the newest two are retained alongside
the public checkpoints, so a crash during capture cannot masquerade as a usable
exact save.

One deliberately conservative boundary remains: low-level Torch/CUDA backend
flags changed externally after LDS performs its runtime preflight are not yet
part of the compatibility fingerprint. Do not change deterministic/TF32/cuDNN
flags between the original process and an exact continuation.

Read the step field as **extra** steps, not a total: the line beside it spells
out where you land ("→ target step 3500") and so does the button. Resuming step
2500 of a run that ended at 3500 is the whole point of opening this from a pill
— a later epoch can be over-cooked, and the earlier one is often the better
LoRA.

The board is the only screen that offers **both lanes for both kinds of run**: a
checkpoint is just a file, so a run trained on your machine can be finished on a
rented GPU, and a cloud run's epoch can be finished here. What is *not* always
possible is stated rather than hidden — a lane you cannot use stays visible,
greyed, with its reason:

- *"Local training needs ai-toolkit"* / *"A training is already running on this
  machine"* — local training is single-flight for the whole machine.
- *"Cloud training needs a vast.ai API key"*, an active cloud run on the same
  dataset, or the concurrency limit.
- *"This save is no longer on this machine"* — the lanes that need the file are
  closed. For a **cloud** run the cloud lane still works, because a fresh pod is
  seeded from that run's own staging rather than from your disk. For a local run
  there is no copy anywhere, and both lanes say so.

If the save vanished between the board being drawn and the click, the launch is
refused with the steps that *are* available, named — never a silent failure.

**The gallery under a checkpoint.** Images pile up. A checkpoint that has
produced more than one shows a small **× N** badge; clicking it opens everything
that checkpoint ever made, newest first — from the board, from the Test Studio,
from a comparison run, it does not matter. Regenerating no longer replaces what
was there.

Which image belongs to which checkpoint is recorded when the image is generated.
Images made before that was recorded are matched back where the evidence allows
it (the run tag the deploy stamps into the LoRA's name); those that cannot be
traced are **counted and left out** rather than shown under a checkpoint they
might not belong to. The gallery says how many those are — they are still in the
Test Studio, they simply have no node to sit under.

**What a generated image was made with.** Open any image from a gallery and the
full-screen view lays its record out beside it: the three facts you look for
first (**step**, **seed**, **LoRA strength**) as chips, then the settings that
actually decided the picture — sampler, scheduler, CFG, sampling steps, the base
model, the LoRA file, any always-on LoRAs, the format, the face-similarity score
— and the prompt last. The prompt folds when it is long instead of pushing
everything else off the screen, and both the **seed** and the **prompt** copy in
one click. A run that predates a given setting simply shows no row for it: an
absent line is honest, a dash is not.

**📌 Pinning an image onto the board.** Comparing two checkpoints means looking
at their pictures *at the same time*, which a full-screen viewer cannot do. So
**📌** drops an image onto the board as a node of its own, joined to the
checkpoint that produced it by the same connector the board uses for "this run
continued from that checkpoint".

There are two ways in, and the first one is the one to remember: **every
thumbnail in a run or checkpoint gallery carries a 📌 in its bottom-right
corner** — one tap, no need to open the image at all. It is hidden while you are
in **Select** mode (that mode is for arming a delete, and a second target there
is a mis-tap waiting to happen). The same action is also in the full-screen
viewer, spelled out as **📌 Pin to canvas**, for when you have already opened a
picture and decide it belongs on the board.

- **Move it** by dragging (on a phone: a long press picks it up, exactly like a
  run card). **Resize it** from the corner handle. **Close it** with **✕**.
- **It goes wherever you want on the board — its lane is not a box.** Drag it
  above its own lane, into the margin to the left of everything, or across to sit
  beside another dataset's runs: nothing stops at the lane's corner any more, and
  the arrow keys reach the same places. **✦ Fit** grows to include it, so a
  picture parked well outside its lane is always one click from being back on
  screen. Two things stay true wherever you put it: the line to the checkpoint
  that made it follows it (that link is read off the image, never off its
  position, so a picture can never end up claiming a run it did not come from),
  and the picture still belongs to its own dataset — moving it over another
  lane's runs changes nothing but where it is drawn.
- The one thing to know before parking one far away: a lane's own position on the
  board depends on which datasets are ticked and how tall the lanes above it are.
  A picture is measured from **its own lane**, so it travels with the run it is
  evidence about — put it next to *another* dataset's lane and it will keep that
  spot relative to its own lane, not relative to its neighbour. **✦ Tidy up**
  brings everything home if a board gets away from you.
- Closing forgets nothing. Pin the same image again and it comes back **exactly
  where you left it, at exactly the size you left it** — that is the point of the
  feature, not a side effect. The geometry lives with your card positions, on
  your machine's LoRA Dataset Studio rather than in one browser, so it follows
  the dataset.
- **Keyboard:** focus a pinned image (Tab), then the arrow keys move it,
  Shift+arrows move it faster, **+** / **−** resize it and **Esc** closes it.
- If the image is later **deleted**, its node quietly leaves the board — a node
  showing a picture that no longer exists would be worse than no node. If the
  *checkpoint* is gone but the image is not, the picture stays and simply loses
  its connecting line.
- Unticking a dataset takes its lane off the board, pinned images included; they
  come back with the lane, untouched.
- **✦ Tidy up** does not throw pinned images away — it brings them **home**. Every
  picture on the visible board comes back beside the run that made it, into the
  same tidy band **📌 Pin all** uses, wherever you had dragged it to. That is the
  guaranteed way back from a picture parked far outside its lane, and it is why
  free placement is safe to play with. Pictures you have **closed** are not
  touched: their remembered spot is a promise, and Tidy up is not the place to
  break it.
- The **✕**, the **🔍** and the resize corner keep a finger-sized target **at
  every zoom level**: they are drawn at a constant size on screen rather than at
  the board's, so a board fitted to twenty runs is still one you can tap.

**🖼🖼 Fuse pinned images side by side.** Comparing two renders across a gap and
two frames is comparing two frames. **Drop one pinned image onto another and
they become a single node**, pictures edge to edge with nothing drawn between
them. There is **no limit**: drop a third, a tenth, they all join the strip.

- **Where it lands.** While you drag, the picture you are about to join lights up
  with a dashed outline, a bar marks the exact slot yours would take, and a label
  says how many pictures the group would then hold. Let go anywhere else and it
  is an ordinary move — nothing fuses by surprise.
- **Which side.** Drop on the left half of a picture to land before it, on the
  right half to land after it. The same gesture **re-orders** a group: drag a
  member out and back onto the slot you want.
- **Move the whole group** by its **title bar** (`⠿ N images`), which is also
  where its **✕** lives. That bar is the only thing that moves a group, on
  purpose: dragging a *picture* inside a group means something else entirely.
- **Take one back out** by dragging it **off the group**. That is the whole rule
  — while it is still over the strip nothing has happened, and letting go there
  puts it back. Once it is clear of the strip it becomes a node of its own again,
  **at the size it had before it joined**, wherever you dropped it. Joining a
  group never rewrites a picture's own size; the strip only borrows it.
- **The pictures that stay do not move.** Take the first one out and the strip
  keeps its place and its height; the rest simply close the gap. A group left
  with a single picture stops being a group.
- **Which ✕ am I about to press?** At rest a group is nothing but photographs.
  Hover (or Tab to) one and *that* picture lights up and shows its own step
  label, its 🔍 and its ✕ — the group's own ✕ is the one on the title bar, and it
  carries the count (`✕3`) precisely so the two can never be confused. Closing a
  group closes all of its pictures, undoes the group, and each one keeps its own
  remembered size; re-pinning one from its gallery brings back **that one**, not
  the strip.
- **Every picture in a strip is the same height**, each scaled to keep its own
  shape — that is what makes the band continuous instead of a row of letterboxed
  tiles. Resize the group from its corner and the whole strip scales.
- **A strip gets ONE link back to each checkpoint it came from**, not one per
  picture, and they all leave the band at the same point. A strip is one object
  to the eye and to every gesture, so eight connectors fanning out of it was
  eight times the ink for one fact — and now that a picture can be parked far
  from its run, those links are long. A strip whose pictures all come from the
  same checkpoint therefore draws a single line; one built from three epochs
  draws three, because collapsing them would quietly credit one epoch with the
  other two.
- **A strip has no width limit, and that is the honest consequence of "no
  limit".** Ten pictures side by side is ten times as wide as one; the board
  zooms and pans, so **✦ Fit** is the answer. It deliberately does *not* wrap
  onto a second row — a strip that quietly stopped being a strip at some
  invisible threshold would be worse than a wide one. On a phone, expect to zoom.
- **✦ Tidy up moves a strip, and never takes one apart.** It brings the whole
  band back beside the run that made its first picture, in one piece and in the
  same order — a strip is something you assembled on purpose, so tidying it means
  putting it away, not dismantling it. (It used to leave strips exactly where
  they were, which was fine while a strip could not leave its lane; now that one
  can be parked anywhere on the board, "leave it alone" would have meant leaving
  it lost.) The way *out* of a group is still the group's ✕, or dragging its
  pictures back off it.

**📌 Pin all — the whole lot in one gesture.** When a generation launched from
the board finishes, the green bar says how many images are ready and names the
checkpoints they joined. **📌 Pin all N to the board** puts every one of them on
the board without opening a single gallery.

- **Where they land.** In a band under the lane, **one column per checkpoint**,
  each column under the checkpoint that produced it — so a lot spanning four runs
  reads as four groups, and each picture still draws its own line back to its
  pill. The band starts below everything already on the lane, which is what makes
  the guarantee a real one: **nothing is ever placed on top of a run card, a
  checkpoint pill or a picture you positioned yourself.**
- **One strip per generation, always in training order.** The pictures of one
  run fuse into a single strip that reads left to right by step — 500, 1000,
  1500 — so the strip is an epoch axis. A **second** generation, even fired at
  the same checkpoint, gets its **own** strip: two runs stay two runs on the
  board, which is the only way to compare them. Pinning one picture at a time
  from a gallery follows the same rule — it joins the strip of the generation it
  came from, in its place in the order, never the end. Images generated before
  LDS recorded which launch made them fall back to grouping by checkpoint.
- **Big lots become a contact sheet.** A pair of renders lands full size; twenty
  or thirty land as thumbnails, which is the size you actually compare that many
  pictures at. Each one is still resizable afterwards like any other node.
- **What is already on the board is left alone.** An image you have already
  pinned is neither moved nor duplicated, and the button counts only what is
  left — once everything is up, the button is simply not there any more. An
  image you *closed* is offered again, and comes back where you closed it when
  that spot is free.
- **Nothing is stacked in silence.** One click places at most 40 pictures; if the
  run made more, the bar says how many were left out and where to get them
  (their checkpoint gallery). The count of what was actually pinned is announced
  for screen readers too.
- **↩ Undo** takes exactly the images that click added straight back off the
  board, and nothing else.

**Which checkpoints you can generate from, at a glance.** Every checkpoint pill
carries its deployment state on its **left edge**: a **solid sky bar** means the
checkpoint is deployed to ComfyUI and can be generated from right now; a **dashed
grey bar** means the file is on your disk but not deployed yet. Not deployed does
*not* mean missing — the save is there, it simply has no copy in ComfyUI, and
ticking it before **🎨 Generate** makes the launch deploy it for you. The shape
(solid versus dashed) carries as much of the message as the colour does, a legend
sits above the board, and hovering a pill spells it out in words.

The graph embedded in a dataset's *Checkpoints & LoRAs* panel is unchanged and
still holds the per-checkpoint actions (download, deploy, continue from here,
inline previews). The canvas is a second way in, not a replacement.

## Upscale a picture straight from the board

Click a pinned picture (🔍, or the picture itself) and the full-screen view now
carries **✨ Upscale & improve** next to **⬇ Download** — the same pass, and the
same choice of engine, as the one in the dataset lightbox.

The same button is on the **checkpoint and run galleries** — open a picture from
a pill's 🖼 gallery, or from a run card, and it is there too. That is where an
improvement is delivered, so it is where the gesture costs the fewest clicks:
you are already comparing a checkpoint's renders when you decide one of them
deserves a bigger pass. Both surfaces are the same action on the same picture:

- **✨ Improve via Klein** re-renders detail and texture. Sharper, but skin and
  colour can shift. The note under the button quotes the exact instruction it is
  about to send and links to where you can edit it or switch it off.
- **🔍 Upscale via SeedVR2** resolves detail at a higher resolution and keeps the
  original look. It appears once SeedVR2 is installed; until then Setup ▸ ComfyUI
  can download it for you, and pressing ✨ before that answers with the same
  offer to install it rather than a plain error.

**Where the result goes.** The picture you started from is never touched. The
improvement arrives as its **own image in that checkpoint's gallery**, right next
to the original — open the gallery from the checkpoint pill (🖼) and you can
compare the two, download either, or pin the improved one onto the board beside
its source. Nothing moves on its own, which is why the confirmation says where to
look. The pass takes minutes, and a gallery already open does not refresh by
itself: close it and open it again to find the new picture waiting at the top.

Two things it deliberately will not do. An **improvement cannot be improved
again** — running two passes over the same pixels is how a face turns to
plastic — and the **lane's reference face** has no ✨ at all, because it is a
photo you supplied, not something the app generated. If a pass fails, press ✨
again: that is the retry.

**It stays out of the Test Studio.** These upscales are not sweep cells, so they
never appear in the Test Studio grid, never count as a run in progress, and never
enter the 👍/👎 ranking of a checkpoint — a rating you give an *upscale* would
otherwise be read as a vote for the checkpoint that did not produce it.

## Tips that save runs

- Trust the composition meter over your instinct — a set that "looks varied"
  is usually still face-heavy.
- Fix every leak the badge reports before training; one "a woman with long
  blonde hair" caption quietly competes with your trigger.
- Don't chase steps. Train the auto count, then let the Test Studio find the
  *earliest* checkpoint that nails the identity — it keeps the most prompt
  flexibility.
- The next chapter — **Building a good dataset** — explains *why* behind every
  rule above. Read it once before your first serious run.

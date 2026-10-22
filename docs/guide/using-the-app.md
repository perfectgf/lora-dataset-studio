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
   that describes a trait currently set to Omit (face/eyes/skin, and by default
   hair). ⚙️ Options lets you Describe hair, makeup, facial hair or glasses so
   they stay prompt-controllable. Fix every flagged caption. A find/replace +
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
    The **Steps** picker includes 1–5 alongside the usual presets: three values
    are visible at a time, with **− / +** to browse. Browsing keeps your selections;
    **Selected** lists all step counts that will run, including those off-screen.
11. **Export** — at any point, **Export ZIP** gives you the curated, captioned
    set as a standard ai-toolkit dataset. Nothing is locked in.

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

## Testing a full model

Once the fp8 twin is in ComfyUI, the **Test Studio** lists it as a base and
**🧪 Test in Studio** opens straight onto it, with its own sample settings filled
in. That matters: a full model trained here is **undistilled**, so it wants a
real CFG and a real step count (CFG 4 / 25 steps for Krea 2). The family's
few-step Turbo defaults render a blurry sketch on it that reads as a failed
training.

Open **Test Studio → Base model only**, choose the **Model family** and its
**Base model**, then enter a prompt or select a prompt batch. No trained dataset
or LoRA is required. CFG, steps, format, seed and images per configuration work
as usual. Images are saved in a **Test Studio · … (base model)** dataset; the
latest run for each family reopens when you return, including Stop and Resume.
Select a LoRA above whenever you want to test its effect or compare checkpoints.

**MOSS-Image** is available in **Base model only**. It uses the SFT BF16 model,
Qwen3.5 9B INT8 text encoder and FLUX.2 VAE. Compatible files already present in
ComfyUI's model folders are reused. Download missing files from the Studio, and
use **Fix missing ComfyUI nodes → Install MOSS-Image loaders** when needed.
Let queued jobs finish, restart ComfyUI, then check the nodes again. A ComfyUI
version with Qwen3.5 support is required. The default is 50 steps, CFG 3,
Euler / sgm_uniform, shift 5 and negative-branch skipping at layer 22.
This integration generates images from text; MOSS LoRA testing and photo editing
are not exposed.

## The generation queue

Everything that renders locally goes through one queue: your ComfyUI runs a
single job at a time, whether it was asked for from a dataset, the Test Studio,
the Canvas or the Bank. So you do not have to wait for one thing to finish
before starting the next — launch an **✨ Upscale & improve** batch, then a
**⚡ Generate**, then a retry on a tile, and they line up and run in turn.

The dock in the bottom-left corner is that queue. It appears only when there is
something in it, and shows, top to bottom: what the GPU is working on right now,
then what is waiting behind it, in the order it will be taken. Each line names
where the job came from and which dataset it belongs to, so two datasets feeding
the same queue are never confused for one another.

Two buttons per line:

- **↑** sends a waiting job to the front. Only the wait can be re-ordered — a
  job already on the GPU has nothing left to re-order, and says so.
- **✕** cancels that one job. This is not **⏹ Stop generation**, which ends a
  whole batch: cancelling here drops a single job and leaves its tile marked
  failed, and **Retry** on that tile queues it again.

Some jobs cannot be cancelled from the dock, and say who owns them instead: a
watermark inpaint belongs to the 🧽 Clean watermarks pass, and a reference edit
to the ✦ Edit reference panel. Both are being waited on by the pass that started
them, and each has its own Stop where it lives. A **paused** line means ComfyUI
stopped answering — that one is resolved from the recovery banner at the top of
the screen, not from here.

Two things still take the GPU exclusively and are not queued behind anything:
a training run, and a vision pass (captioning, framing, face analysis). While
one of those is running, new generations wait for it and the app says so.

### When the queue waits for something that is not LDS

Ollama shares your graphics card with ComfyUI, and only one of them can have it.
So before every generation LDS checks that the local Ollama is not holding a
model — and if one is loaded that LDS did not load itself, it waits rather than
evict somebody else's work. The dock says exactly what is in the way and, once
the wait passes a minute, how long it has been standing there.

That wait usually ends on its own: Ollama drops an idle model after a few
minutes and the queue resumes with nobody touching anything. When it does not —
another app is captioning, a second LDS instance is running a batch, or the
runner in the Ollama slot never unloads at all — you have two answers, and
neither is "quit and come back":

- **Unload it and continue** evicts the other model. Right when you know what it
  is and that it is idle; it is never done automatically, because LDS cannot
  tell your live work from a leftover.
- **Run anyway** shares the card instead: LDS starts generating next to the other
  model. Nothing of yours is unloaded. The cost is real — two loaded models on
  one card do not crash on Windows, they page, and generation can get much
  slower — so it asks once, in those words, and the guard comes back on its own
  after fifteen minutes.

An Ollama URL the app cannot use at all (a typo, or an address with a path on the
end) is not treated as a busy card: captioning will tell you it cannot reach its
model, and image generation keeps running.

## The Gallery (every image you generated)

**🖼 Gallery** in the top bar is one feed of everything the app ever rendered —
Test Studio cells, Canvas previews, comparison runs and ✨ Upscale & improve
results — across every dataset at once, newest first. The per-checkpoint
galleries answer "what did this training produce"; this page answers "what did
I make".

Narrow it from the row above the grid: one dataset, **Renders** or
**✨ Improved** only, or **👍 Liked** — the images you rated up in the Test
Studio. The count always names what the grid is actually showing. The feed
loads itself as you scroll towards its end; the **Load more** button at the
bottom states how many are left and still works as a plain button.

Tap any image to open the viewer — the same one the Canvas uses, with
everything the picture was made from: seed (copyable), checkpoint, base model,
sampler, CFG, the always-on LoRAs it was generated with, and the full prompt.
The **‹ ›** buttons (or the ← → keys) walk the feed without closing it; tap
the picture to put the details away, double-tap to magnify.

From the viewer you can also:

- **⬇ Download** — the file lands under a name that still says which dataset,
  run, step and seed made it.
- **✨ Upscale & improve** — Klein (re-renders detail; sharper, but skin can
  shift) or SeedVR2 (upscales and keeps the look). **Klein opens a small
  window first**: the exact instruction it is about to send (editable in
  place, or switched off), the Klein model, a **LoRA preset** to chain and
  the **output size (MP)** — all app-wide, the same values Settings shows —
  then **✨ Generate** starts the pass and the finished picture appears right
  in that window. Close it early and nothing is lost: the result arrives at
  the top of this gallery as its own ✨ image. SeedVR2 has no dials, so it
  runs straight away. Either way the original is untouched.
- **↩ Use these improve settings** — on a ✨ result you like: the
  instruction, LoRA preset, strength, steps, output size and model that made
  THIS image become the app-wide improve settings again, so the next
  improves run the same way. Every new improvement records what it ran
  with; older images restore what they carry, and the toast names exactly
  which parts were applied.

**Select** at the bottom turns on selection mode: tap the misses, then
**🗑 Delete** (files go to the recycle bin or the app Trash — and the rows
leave the Test Studio too, which the confirmation says before anything is
armed), or **⬇ ZIP** to download the picked images as one archive under their
lineage names.

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
  renamed or skipped. There is no size limit: LDS only checks that the drives
  receiving the backup have room for it, and tells you how much space is missing
  when they don't.

## The image bank (triage a big folder)

You exported 9 000 unsorted images from Telegram (or a scraper dumped a
mountain of files) and a dataset only needs the best 30–150 of them. The
**🗃️ Bank** tab is the triage funnel that gets you there — without ever
touching the folder itself.

**Where things are on that screen.** A bank you open is laid out in three
parts, and knowing which is which saves reading the rest of this section twice:

- a **top bar** with the bank's name, its counters, and the four actions that
  change what leaves the bank — **⚙ Passes**, **🚀 Launch all…**, **⬆ Promote…**
  and **🗑 Delete rejected from disk**;
- a **filter rail** down the left: the search, the exclude box, the subfolder
  picker, the person and style strips, and the chips. The six measured axes
  (Score, Framing, Medium, Angle, Resolution, Origin) sit behind **🎛 More
  filters** so the everyday ones stay on one screen. On a narrow window the rail
  becomes a drawer you open with **☰ Filters**, and it remembers whether you
  keep it open. On a wide window the rail stays put as you scroll the grid, so
  the chips are still there ten thousand images down;
- the **grid** filling the rest, with the selection actions directly above it.

The analysis passes live **inside ⚙ Passes** rather than across the top of the
page: they are the step you run once per bank and then leave alone for days, and
they were taking up the third of the screen the images now use. All eight are
still there, and each still opens its own window with its own scope and counts —
only the door changed. On a bank with nothing scanned yet the panel opens by
itself, because there is nothing else to do first.

The funnel itself:

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
   → **🧪 Caption Lab**, in the same 🏷️ Caption window, benches up to four
   configurations — engine, vision model, vocabulary register and length — on ONE
   image you pick, side by side, before you spend a pass on thousands. Nothing is
   written until you keep a result; **⚙️ Use for the next run** loads the winning
   configuration into the dials above (a bank picks its caption method per run rather
   than storing one, which is what that button means here). A bank caption can also be
   edited by hand from that window — what you write is stamped as yours, so a forced
   🔄 Re-caption spares it unless you tick the opt-out.
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

## Read-only bank sources

Enable **Read-only source** when creating an Image bank, or use the same
checkbox beside an existing bank's name. This is useful for a NAS or a folder
of original photos: **Delete rejected from disk** and web imports into the
source folder are refused by the server, even from an old browser tab.
Banks whose source folders overlap a protected bank are also blocked from
deleting or adding source files.

You can still browse, scan, keep/reject, caption, crop or improve working copies,
and **Promote** images into datasets or new banks. These operations leave the
source files untouched. Removing a protected bank removes its LDS record and
working data; its source folder is preserved, including an imported bank's copy.
The protection setting belongs to that bank and is removed with its record.

Existing banks keep their current behavior until you enable protection.
Turning it off requires confirmation. Wait for any running bank operation to
finish before changing protection; the app refuses a change while an overlapping
bank is busy too. This setting controls LDS bank operations; it does not change
filesystem permissions or prevent other programs from modifying the folder.

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

## Compare the copies of a duplicate group

The **≈ Duplicates** and **✂ Same shot** filters replace the grid with one card
per unresolved group, and those cards used to offer three ways to settle a
group: *Resolve ALL — keep best*, *keep first*, or clicking one of the
thumbnails. The first two are verdicts you take on trust; the third asks you to
tell two copies of the same shot apart in a 96-pixel stamp. **⤢ Compare** — on
the group's card, or *⤢ Compare & pick* at the top for the whole list — opens
those same copies at a size where the choice can actually be made.

- **Side by side** puts every copy of the group on screen at once, each as big
  as the screen allows. This is the view that settles *framing* — which one is
  cropped, which one has the shoulder in it.
- **⛶ Full screen** (**F**) shows one copy filling the frame, and **← →** flips
  between them *in the same frame*. That is the view that settles *detail*: the
  difference lands on the same pixels instead of asking your eye to carry it
  across a gap.
- **Under each copy, the numbers that separate them** — resolution, sharpness,
  aesthetic score, file weight — with the group's best value **lit**. The copy
  with nothing lit is the one that loses on everything. When two copies have the
  same dimensions *and* the same weight they are marked **≡ same file as 2**:
  they are the identical file, so keeping either keeps the same pixels.
- The cursor **opens on the copy the app elected** (the BEST badge), so **K**
  is "yes, that one" and moving off it is a deliberate disagreement. The badge's
  tooltip says what it wins on — or admits that nothing measured separates the
  copies and the tie-break was import order.

The keyboard is the same grammar as ▶ Review, at the level this screen decides
at: **K** keeps the copy under the cursor and rejects the rest of its group,
**R** rejects only that copy and moves to the next (which is how a group of five
is worked down one obvious loser at a time), **S** skips the group without
deciding, **Esc** leaves. What is this screen's own: **← →** move between the
copies of a group, **1**-**9** jump straight to one, **⇧← ⇧→** move between
groups, **B** puts the cursor back on the app's pick, **F** switches the layout.

A skipped group stays unresolved and is not shown again *in that run* — it is
"not now", not "no". When the walk runs out, the run refills itself from what is
still unresolved, so a bank with 300 duplicate groups is worked through without
going back to the list. Every verdict is saved on the spot, and the losers are
**rejected**, never deleted: the ✕ Rejected filter brings any of them back.

## Say "these are not duplicates"

Both grouping passes answer a question about pixels, and both are sometimes
wrong in the one direction you could not correct: a burst of frames, a series
shot on a tripod, two crops that a threshold called one picture. Every verdict
on offer ended in a rejection — *keep best*, *keep first*, a manual pick — and
**⏭ Skip** writes nothing at all, so the group came back on the next run, and the
one after that. The only ways out were to reject a photo you wanted, or to keep
saying "not now" forever.

**≠ Not duplicates** (on the group's card, or **N** in ⤢ Compare) is the missing
answer. It keeps **every** copy, rejects nothing, and the group stops being
proposed.

- **It decides nothing about the images.** They keep whatever status they had,
  kept or undecided, and they stay in every other filter. The claim is about the
  *relation* between two pictures, not about either one of them.
- **It survives a re-group**, which is the whole reason it works. Both passes
  renumber the entire bank from scratch on every run, so a verdict remembered as
  "group #7" would quietly apply to a different set of images next time. What is
  stored is the **pairs**: *this photo and that photo are not the same shot*.
  That sentence means the same thing before and after any renumbering.
- So a re-group that **splits** the group leaves it answered, and one that
  **adds a copy** asks you again — with the new copy on screen. A new member is
  a new question, and you were never asked about it.
- One answer covers **both stages**: ≠ on a ≈ Duplicates group also settles the
  ✂ Same shot group holding the same images. It is a fact about the pictures,
  not about which algorithm found them.
- **The way back is a line above the list** — *≠ N groups marked not duplicates
  — Put them back* — and it stays visible when marking the last group has
  emptied the panel, because an undo that vanishes with the thing it undoes is
  not an undo. Re-running *Keep best* on a group by name also overrides it: naming
  a group is ruling on it again, and you are allowed to change your mind.

**One limit, stated plainly:** ≠ records a decision about every *pair* in the
group, so a group of 80 copies costs 3 160 of them. Above 80 it is refused, with
the reason — a group that size means the duplicate distance is too loose, and the
fix is the 🎚 threshold, not 3 000 stored verdicts.

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

In a **dataset** (above the grid, next to the decision chips) there are two
kinds of entry, and they answer different questions:

- **Face similarity ↓ / ↑** — the ArcFace cosine against your reference photo
  computed by **🎭 Analyze faces**. ↓ is "who looks most like my subject", ↑ is
  the shortlist to cut. This *ranks* the whole grid.
- **Shot type** — face, then bust, then body, then back, in the order the
  composition bar counts them. This *groups*: it ranks nothing, it puts every
  shot of one kind in a single run so you can compare like with like. A grid in
  arrival order interleaves the four kinds, which is the wrong arrangement for
  the question you are actually asking — *do I have too many of these, not
  enough of those, and which of these near-identical ones do I keep?* The shot
  type is the one the **📐 Classify framing** pass wrote (and the one the shot
  card carried, for a generated image).
- **Shot type, then face similarity ↓** — the same grouping, with the closest to
  your reference at the head of each kind. This is the order for curating: walk
  down a run and the ones to cut are at its end.

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

## Keep or reject a dataset image without leaving the picture

The full-screen view is where you can actually *see* whether a hand is right or
an eye is mush — so that is where the verdict belongs. The bar under the image
carries the same three buttons as the Bank's **▶ Review**, on the same keys:

- **✓ Keep** — `K`
- **✕ Reject** — `R`
- **⏭ Skip** — `S` (or **→**)

**Keep and Reject move you on** as soon as the verdict is saved, so a folder of
300 pictures is worked through with one hand on the keyboard and never a return
trip to the grid. **Skip is nothing but "next"**: the image keeps whatever it
already had, undecided included. **←** goes back the same way — navigation only,
it decides nothing.

It is the *same* verdict as the ✓ / ✕ on the tile behind the overlay, not a
second one: only kept images are captioned, exported and trained on, and the
grid, the counters and the ⬇ Export all read that one status. The chip beside
the image's name says which one it is carrying right now — **✓ kept**,
**✕ rejected** or **· undecided** — so you can tell a landed decision from a
missed keystroke.

Two things it deliberately does not do. **Nothing is deleted**: a reject is a
status, the file stays on disk and ✓ takes it back. And **the verdict is sent
before you move** — on a slow disk the buttons grey out for a moment rather than
walking on with a decision still in flight.

At the end of the list there is nowhere to advance to, so the picture stays in
front of you wearing its new chip; the ⟩ arrow already says which end that is.

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

## Crop and upscale inside a bank

A bank is where the filtering and the curation happen, but reframing or
upscaling a shot used to mean leaving it: promote into a dataset, edit there,
export into a **new** bank, and start curating again. Both edits now happen in
the bank itself, so the loop is *curate → edit → re-analyse → promote*, in one
place. (Asked for by nofaceman on Discord, backed by mr.arrow.)

**✂ Crop** is per image, in **▶ Review** — the only place a bank shows a picture
big enough to draw a box on. Open Review (or press ▶ on a tile), then click
**✂ Crop** or press `C`. Drag the box, or snap it to a ratio, and confirm.
Cropping decides nothing: the image stays under your cursor so you can judge it
once it is framed properly.

**Nothing is resampled here**, and that is the one real difference from the crop
inside a dataset. A dataset crop rescales the box you drew to a 1024 px long
side, because a dataset image is training material and that is its size. A bank
sits *upstream* of that choice — shrinking here would pick your training
resolution before you have even picked a dataset, and would do it silently. So a
bank crop is a pure cut: it keeps the pixels inside the box, and the dataset
still decides the size when it imports.

**✨ Upscale & improve** is a pass, on the **✂ Edits** panel (⚙ Passes). It takes
the same kept / undecided / unkept / selection scope as everything else, which
matters more here than anywhere: this one spends GPU-minutes **per image**. Pick
the engine on the panel — **Klein** re-renders detail from a prompt (sharper, and
skin and colour can shift) or **SeedVR2** resolves detail and leaves the original
look alone — then launch. It runs in the background with a progress bar, and ⏹
Stop ends it between two images, keeping everything already done. Unlike the
dataset version, there is no candidate to validate: a bank *is* the review, so
the result replaces what the bank shows.

**Your own files are never written to.** Both edits land in a copy the app keeps
next to the bank, exactly like the watermark cleaning. **↩ Revert** on the ✂
Edits panel throws those copies away — for the selection, or for the whole bank —
and gives you back the image it started from, including any rotation the edit had
absorbed. In ▶ Review, **Revert all edits** does it for the image on screen.

**↩ Undo last edit** goes back just one step: after a manual crop and an upscale,
it removes the upscale and keeps your crop. Repeat to undo earlier edits. The
Edits panel offers the same action for the selection or for each edited image.
In Review, **Compare before / after** shows the input of the last edit beside
the current result, including the cropped input of an upscale.

History is saved for new edits from this update onward; previous versions already
discarded by older releases cannot be recovered. Retained versions use disk space
until undone or fully reverted. Missing or changed previous files are reported
and left unchanged when undoing.

Two consequences worth knowing. First, an edit **clears every measurement taken
from the old pixels**, so ✨ Score, 📐 Framing and the rest pass over those images
again — which is the point: a sharpness score read off the shot before you cropped
it describes an image the bank no longer holds. Second, ✨ Upscale & improve does
not re-run on an image it has already improved; **↩ Undo last edit** lets you try
again without losing the crop underneath. **↩ Revert** also remains available.

## Repaint one detail without regenerating the image

Two people asked for this from opposite directions on the same week: one wanted
the watermark remover pointed at a necklace and some skin blemishes, the other
wanted to fix a small glitch in a fresh picture without regenerating the whole
thing. Same hole.

The app already had the hard part. **🧽 Clean** repaints exactly the box you draw
and leaves every pixel outside it **byte-identical** — but its instruction was
frozen on "reconstruct a clean, natural image", so it could only ever be aimed at
a watermark. **✦ Edit**, the other lane, takes any instruction but re-renders the
**whole** image, which drifts outside the area you cared about.

**✦ Repair** is the first lane with both. Open the image (click its tile) and press
**✦ Repair** in the action bar. Draw the zone, type what should be there —
*"remove the necklace"* — and press **✦ Repair** again. Only that zone is
repainted. Everything outside it comes back exactly as it was, to the byte.

**Two shapes, one button.** Inside that dialog you choose how to point at the
area:

- **▭ Box** — drag a rectangle. The app crops a square around it and works on
  that crop, so it is quick and its memory use does not depend on how large the
  photo is. Right for a mark in a corner.
- **🖌 Brush** — paint over the thing itself, with a size slider, an eraser and
  Clear. The model sees your paint plus a generous ring of context around it —
  a localized touch-up travels as a native-resolution crop, and only a paint
  job that spans most of the frame sends the whole (size-capped) picture.
  Right for jewelry, glasses, straps — anything a rectangle would only enclose
  by taking a lot of its surroundings with it. Pixels you did not paint are
  copied from your file either way.

Both work under a finger, so this is usable from a phone. The brush was
contributed by OneCodingDude on GitHub.

**The brush needs one small install.** The masked pass runs on **LanPaint**, a
training-free inpainting sampler (a ~1 MB ComfyUI node pack, no Python
dependencies): Klein is an edit model, not an inpaint-trained one, and
conditioning it like one is what used to hand back a smeary patch — reported by
charlesangus on GitHub, and exactly what LanPaint exists to fix. Setup ▸ the
**LanPaint sampler** row installs it; restart ComfyUI afterwards so it loads.
Your paint is also grown by a few pixels before the model sees it, so the
edges of the removed thing get rebuilt instead of leaving a halo — and the
best prompts describe **what should be behind** (*"bare skin"*, *"plain
wall"*) rather than naming what to remove.

The 🚩 button next to it opens the same editor from the other intention — you
spotted a watermark the scan missed. Same screen, same zones; what differs is
whether you press 🧽 Clean or ✦ Repair once you are there.

A few things worth knowing:

- **It says nothing about watermarks.** A repair does not flag, clear or stamp
  anything: the image keeps whatever watermark state it had. It is an edit you
  asked for, not a verdict.
- **Your original is preserved first.** The master is copied aside *before*
  anything is written, so a repair that fails costs you nothing — the file is
  left exactly as it was.
- **An empty description is refused**, on purpose. Falling back to the watermark
  sentence would repaint your zone with an intention you never expressed.
- **↩ Undo puts the previous image back**, one step deep, so trying another
  description costs nothing — which is the normal way to use this: look, not
  right, change the sentence, go again. The dialog stays open after a repair for
  exactly that. The undo is consumed once used, and it never reaches the
  write-once original kept for ↩ Undo cleaning — undoing a repair must not throw
  away a watermark clean you made earlier and still want.
- It runs on Klein through ComfyUI, one round-trip per repair.

**On a picture you just generated, too.** Open a generated image full size — on
the Canvas, or from a checkpoint gallery — and press **✦ Repair** next to ⬇ and
✨. Same gesture, same guarantee: a stray finger or an unwanted object no longer
means throwing away the render you liked and rolling the dice again.

## Clean the watermarks a bank found

**🚩 Find watermarks** flags the images carrying an overlaid logo, URL or
@username. Removing them used to mean promoting the watermark into a dataset
first and cleaning it there; the bank now does it itself, in **two steps you
launch by hand** — cheapest and safest first:

1. **✂ Auto-crop** cuts off the marks sitting in a border strip. No model, no
   GPU, no invented pixel: it simply trims the band up to the mark, and only
   when the image stays big enough to train on. Anything it can't crop that way
   is left flagged, on purpose.
2. **🧽 Inpaint** repaints what's left. **LaMa** (fast, non-generative) repaints
   the marked zones and leaves marks *on the subject* flagged. **Klein** (slower,
   via ComfyUI) works in two steps: the zones the scan found are **erased** on the
   photo, and then the **whole photo** is re-rendered with the instruction to
   remove the watermarks. The erasing is what stops the model handing the mark
   back; the whole-photo pass is what also clears the marks the scan *missed* — a
   mark tiled across the picture, one on the subject, one the detector boxed in
   the wrong place. The price is an image whose every pixel is regenerated. Each
   engine says what to install when it isn't ready, and the button stays off
   rather than failing mid-pass.

   **It is a generative pass, not a mask, so read the result.** Measured on the
   shipped settings: a photo tiled wall-to-wall with a mark came back with all
   twelve zones gone and looking clean — the case that was hopeless before,
   because there was no unmarked area to copy from. A photo carrying seven
   distinct logos came back with all seven gone. What can still survive is a mark
   **nobody found**: nothing erased that one, so the model is free to keep it.

   **Why the erasing matters, in one measured example.** Run without it, the same
   photo came back with a round logo *redrawn* as a plausible **moon in the sky**
   — the model reinterpreting a mark it could still see rather than deleting it.
   A re-run of 🚩 Find watermarks sees nothing wrong with an image like that (a
   moon is not a watermark), so it would stay marked *cleaned* and no later step
   would catch it. Erasing the zones first removed every trace of that. It is
   still worth a look at the picture.

   **Three dials, right there under the engine.** Picking Klein reveals the
   **prompt it is actually sent** (`remove watermark` by default, editable, with
   *Reset to default* to get it back), the **processing size** the photo travels
   at (1 – 4 MP, default 2 — higher means finer regenerated detail and more VRAM
   and time, and a photo already smaller than the setting is never enlarged), and
   **what size the cleaned file is written at**: back at your file's own
   dimensions, as before, or at the render's size — in which case **the file
   changes dimensions**. The dataset's Clean bar offers exactly the same three,
   and they are one stored choice, so setting them on either side arms both.
   Every clean also writes the prompt, size and write-back mode it used to
   🪵 Server log, so you can tell afterwards what actually ran.

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

### The 🚩 launch window: sample, threshold, and the result in place

**🚩 Find watermarks opens the same kind of window as 🔤 Find text now, on
both surfaces** — the dataset button used to fire straight from the click.
*Try on a sample first* judges only the first N images of the scope
(deterministic — a re-run re-judges the same ones), so on a huge bank you can
check the flags before paying for the whole scan. When the dedicated detector
is installed, the *Detector threshold* slider edits the stored score an image
needs to be flagged — lower flags fainter marks at the cost of false flags —
one value, both surfaces (the vision route carries no score, and the window
says so instead of showing a dead slider). And the flagged pages appear below
the dials with their boxes drawn on them, filling in live while the scan
runs. The strip shows the **watermark-family** pages; pages flagged by 🔤
Find text live in that pass's own window — the same page-level split **What
to clean** repaints by.

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

## Erase burned-in text — bubbles, subtitles, captions

A comic page carries its dialogue, a screencap its subtitle, a meme its
caption — and a LoRA trained on them learns the lettering along with the
subject. **🔤 Find text** reads that text and feeds the exact same cleaning
funnel as the watermarks: every block of text becomes a zone in the image's
mask, the image is flagged, and **🧽 Inpaint** repaints the zones. One funnel,
one ↩ Undo, one mask editor — a text zone behaves exactly like a zone you drew
by hand. **✂ Auto-crop never touches them**, on purpose: cropping a speech
bubble out of the middle of a page is not a thing.

The reading is done by the same OCR engine as the Video bank's **🔳 Safe
zone** pass (one Setup install serves both — *Burned-in text*, a small
Apache-2.0 package that works offline). It runs on the **CPU only**, never the
GPU, so it can scan a bank while a training run owns the card. Regular
lettering is found whatever the script — Latin, Korean, Japanese, Chinese
dialogue, subtitles and captions are all boxes to it. **Heavily stylised
lettering can escape it**: a calligraphic sound-effect with thick outlines is
drawn more than written, and the detector can miss it entirely (measured on a
real page — no threshold recovers it). Those get the hand mask in **🚩 Edit
mask**, like any zone the machine missed.

**How the repaint treats these zones.** A text zone is not handed to the
repaint model as a rectangle any more — that is what used to eat balloon
outlines. The clean now runs an outline-safe filler first: every letter is a
small closed ink shape *inside* the zone, so anything drawn **across** the
zone's edge (the balloon outline, the art) is preserved by construction; the
letters are then erased with the bubble's own background colour —
including the faint JPEG haze around them — or rebuilt by a local
inpaint when the background is graded. Only lettering sitting on busy art
still goes to the repaint model, and it gets letter-sized boxes, never the
whole rectangle. Pages cleaned before this shipped can be upgraded:
**↩ Undo cleaning**, then Clean again.

What it does *not* do, said plainly:

- it reads **positions, not words** — no transcript of your images is stored
  anywhere, the boxes are all that is kept;
- the mask holds at most **32 zones per image**; a text-heavy page that
  produces more keeps the 32 biggest blocks and the pass's report says how
  many were left out (draw those in **🚩 Edit mask** if they matter);
- images you **dismissed** stay dismissed — this pass never re-flags a row you
  already ruled on, exactly like a watermark re-scan;
- a **🚩 Find watermarks** run afterwards will not undo it: text zones survive
  the scan, and a watermark box found on the same image joins them.

**Try it on a sample before paying for the whole bank.** The launch window
carries two dials. *Try on a sample first* reads only the first N images of
the scope (deterministic — a re-read hits the same pages), so on a 9 000-page
bank you can judge the result on twenty before committing to the rest.
*Sensitivity* is the OCR confidence a line needs to become a zone — lower
catches fainter or more stylised lettering at the cost of false zones. It is
stored (one value, both surfaces), and the zones are always yours to edit
afterwards in **🚩 Edit mask**.

**The result shows up in the same window.** Launching does not close it: the
flagged pages appear below the dials with every zone drawn on them, filling
in live while the scan runs — on both surfaces (the strip shows the first
pages and says how many are flagged in total, and each tile opens the
full-size page).
Judge the zones, adjust the two dials, re-run — all without leaving the
window; a zone that landed wrong is fixed by hand in **▶ Review** /
**🚩 Edit mask** as before. Close it whenever you are done looking.

**Clean text and watermarks separately.** Once Find text has flagged
something, the repaint level grows a **What to clean** switch — *Both*,
*🔤 Text*, *🚩 Marks* — next to the LaMa/Klein engine toggle (the bank's
Watermarks panel and the dataset's Clean row both carry it, and the Clean
button's count follows the choice). The split is **by page**: a page carrying
both a watermark and text counts as text and is repainted whole — its zones
live in one mask, so one page is never split between two runs. With no
text-flagged page the switch stays hidden, because all three choices would
mean the same thing.

It works on both surfaces, at full parity — a bank's Watermarks panel
carries the **🔤 Find text** card next to 🚩 Find, and a dataset's curation
row carries the same button next to its watermark scan. Both open the same
launch window: the sample dial, the Sensitivity slider (one stored value,
whichever side you move it from), the measured count of what the run will
actually read, and the flagged-pages strip.

## Fix a watermark mask — or mark one the scan missed

The detector draws **one** box, and it is a guess: it can miss a second logo,
swallow half the face, or land beside the mark. Open **▶ Review**, walk to the
image and press **🚩 Edit mask** (shortcut `M`) — the same zone editor the
datasets use, on the bank image, right there.

It also opens on an image the scan flagged **nothing** on, where the button reads
**🚩 Mark a watermark** instead. This is the answer to a miss: the detector is a
classifier, and a mark tiled across a whole stock photo can score under any
sensitivity you set. **The zones you draw become the flag**, so the cleaning
steps below can act on an image the scan cleared. It works the same way in a
dataset, from the image viewer.

Drawing on an image you had **dismissed** as a false positive takes that ruling
back. The one image that refuses is one already **cleaned** — its pixels have
been replaced, so a zone drawn now would describe a picture that no longer
exists; use **↩ Undo cleaning** first.

- **+ Add zone**, then drag on the photo to draw a rectangle over the mark. Up
  to 32 zones; drag a zone to move it, its corners to resize.
- **Delete zone** removes the selected one, **Reset to detected** throws your
  zones away and puts the detector's box back.
- Every edit saves as you draw. If a save fails it says so and offers a retry —
  the zones on screen are never silently unsaved.

What the two cleaning steps then do with your mask:

- **🧽 Inpaint acts on the zones you drew** — all of them, including a zone
  sitting on the subject, which is precisely what a hand mask is for. **On
  LaMa** they are exactly what gets repainted, and nothing else is touched. **On
  Klein** they are erased from the photo first and then the whole picture is
  re-rendered, so your zones decide what is guaranteed to go, while the pass also
  clears marks you did not draw — and everything else is re-rendered with them.
  Drawing zones therefore buys precision on LaMa, and on Klein it buys certainty
  about the marks you pointed at.
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

Open a bank, then **⚙ Passes → Manage Score Python…**. The button stays available
when packages are missing, the interpreter uses the CPU, or CUDA is detected.
Changing Python is disabled while a pass is active.

The **✨ Score** pass (aesthetic · NSFW · style) normally uses the dedicated
environment prepared by Setup. It carries **CPU-only PyTorch** to keep the first
install small. You can select another Python that already has the required
packages, such as a ComfyUI, ai-toolkit or conda environment. An environment that
works for another app can still fail when running Score's calculations.

The dialog shows **Python used for Score** and the **Managed environment** as
separate paths. Check the first one when diagnosing an error: that is the
interpreter Score actually uses, including any fallback from an empty setting.

Discovery checks the interpreters the app knows about, package by package:

- **CUDA detected · calculation not tested** — the required modules import and
  PyTorch reports CUDA availability. This does not prove that cuDNN, attention
  or a model calculation works.
- **Missing packages** — a required import or class is unavailable. Score needs
  `torch`, `open_clip`, `transformers`, `timm`, `numpy` and Pillow. The row names
  the missing dependency and cannot be selected until it passes discovery.
- **CPU only** — the required imports are present, but CUDA is not available in
  that interpreter. A CPU calculation has not been verified by discovery either.
- **No answer** — the interpreter could not be checked, for example because its
  environment moved or did not respond.

**Test calculation** is a separate button on each usable interpreter. It runs a
small convolution and attention calculation on the detected device and reports
its result separately from discovery. It downloads no models and starts no Bank
pass or training. A failure reports a bounded diagnostic; if another task is
using the runtime, the check can be deferred. A passed check verifies those small
operations, not every model or a complete scoring pass. Opening the dialog,
checking imports or selecting a Python never starts this calculation automatically.

**Setup → Quality tools → Bank scoring → Reinstall** repairs the managed
environment even when Score uses an external Python. It preserves that external
selection and never installs into it. After a repair, click **↻ Check again** in
the picker, then **Use managed environment** if you want Score to use the
repaired environment. You do not need to clear the Python setting first.

**Back to the app default** clears the explicit selection. For Score, an empty
`bank_scoring.python` falls back to the Python running the app; it does not
explicitly select the managed environment. Use the dedicated managed button for
that choice.

**The app never installs packages into an external environment.** When an
external Python is missing a package, the dialog shows the command you can run
yourself. Afterwards, use **↻ Check again** to refresh discovery. Diagnostic
results are cleared when discovery refreshes so an old result does not describe
a repaired or changed environment.

**Not listed?** Enter an interpreter or its environment folder, then click
**Check it**. Venvs, conda/miniconda environments, portable Python bundles and
paths on other disks are supported, including spaces, accents and Windows
“Copy as path” quotes. Checking a path adds it to the list; it does not select it.

**No NVIDIA card?** CPU interpreters can still be selected to reuse packages
already installed. The dialog does not suggest a CUDA speed-up in that case.

## Build the SigLIP 2 index on a GPU Python you already have

Open **Bank → ⚙ Passes → Semantic engine → Manage SigLIP 2 Python…**. This
management action stays visible independently of CUDA detection, installed
packages and the selected semantic engine. During a pass or semantic operation,
it remains visible but disabled.

The dialog has the same effective-path display, **Use managed environment**
choice and explicit **Test calculation** as Score. CUDA detection alone does not
verify the runtime, and the small calculation does not build or verify a Bank's
semantic index. Building that index remains a separate action.

**The dependency list is SigLIP 2's, not Score's.** It checks PyTorch, NumPy,
Pillow and a Transformers build carrying `Siglip2Model`. It does not require
`open_clip` or `timm`. An interpreter missing a required import or class is
refused for selection, with the missing dependency named.

**Borrowing an interpreter downloads nothing.** The pinned SigLIP 2 checkpoint
lives in the app's data folder, not inside the interpreter. Selecting a different
Python does not move or duplicate those weights.

**Setup → Quality tools → SigLIP 2 → Reinstall** repairs the managed Bank
environment and preserves an existing external SigLIP 2 selection. Afterwards,
refresh discovery and choose **Use managed environment** explicitly to run the
index there. Repairing that environment does not change Score's Python selection.

**Back to the app default** clears `bank_semantic.python`, which then inherits
`bank_scoring.python` or the app Python. That inherited Python can be external;
clearing the setting is not the same as choosing the managed environment. Score
and SigLIP 2 have separate explicit settings, but this fallback links them while
the SigLIP 2 setting is empty.

## Run the watermark detector on a GPU Python you already have

The **🚩 Find** scan is the third pass with the same story. Installing the
watermark detector (Setup ▸ Quality tools) builds it a small environment with
**CPU-only PyTorch** — the same deliberate default as Score — and *pins* that
environment as the detector's interpreter. On a machine with a card the scan
therefore ran on the CPU, silently, however good the GPU sitting idle next to
it was.

Two things changed:

- **The Bank's 🚩 Watermarks panel now says it.** When the fast detector is
  installed but its Python cannot reach CUDA on a machine that has a card, an
  amber note names the situation and offers **⚡ Use a GPU Python I already
  have**, opening the shared interpreter picker. The pass summary also
  reports which device the scan *actually* ran on — "(detector on GPU, …)" or
  "(detector on CPU, …)" — read back from the scan itself, not from a guess.
- **The picker speaks the detector's own dependency list.** It never imports
  `open_clip`, `timm` or even NumPy, so the ComfyUI interpreter Score refuses
  is usually perfect here. What it *does* need is a **Transformers carrying
  both halves of the cascade** — the SigLIP classifier and the Grounding-DINO
  locator (4.40 or newer). Both classes are really looked for, not assumed
  from the package name, and an interpreter missing either is refused with the
  exact repair command.

**Borrowing an interpreter downloads nothing.** The detector's pinned weights
live under the app's models folder, not inside the interpreter. And as
everywhere in this dialog family, nothing is ever installed into an
environment the app did not build — **Back to the app default** reverts the
choice at any time, after which the scan falls back to Score's interpreter and
then the app's own, exactly as before.

Each pass has its own explicit interpreter setting. Clearing a setting with
**Back to the app default** restores its documented fallback; SigLIP 2 and the
watermark detector can therefore inherit Score’s interpreter. Use **Use managed
environment** to select the app-managed Python explicitly.

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

## Tips that save runs

- Trust the composition meter over your instinct — a set that "looks varied"
  is usually still face-heavy.
- Fix every leak the badge reports before training; one "a woman with long
  blonde hair" caption quietly competes with your trigger unless Hair is set
  to Describe in Captions ⚙️ Options.
- Don't chase steps. Train the auto count, then let the Test Studio find the
  *earliest* checkpoint that nails the identity — it keeps the most prompt
  flexibility.
- The next chapter — **Building a good dataset** — explains *why* behind every
  rule above. Read it once before your first serious run.

# Using the app

The workspace is a **guided flow**: each stage stays folded until the one
before it is done, and the progress rail on the left tells you where you are
and what's blocking the next step. You never have to guess what comes next —
this chapter just explains what each stage does and where the useful buttons
hide.

---

## The character walkthrough (reference photo → trained LoRA)

1. **Create the dataset** — Datasets → New. Pick **Character**, name it, set a
   **trigger word** (the token your prompts will use), and choose the **target
   model** (Z-Image / SDXL / Krea 2 — changes the caption style; you can change
   it later).
2. **Upload the reference photo.** The app head-crops it automatically; use the
   crop editor (or *Reset to auto*) if the framing is off. Up to 3 extra angles
   can be added for multi-view consistency.
3. **Generate variations** — pick an engine (Nano Banana, ChatGPT, or local
   Klein) and fire the **variation catalog**: 45 shots across expression,
   angle, lighting, framing, outfit and background, each wrapped in an identity
   guard so the face stays the same person.
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
   tag-frequency panel sweeps the whole set at once.
8. **Fix individual shots** — every generated tile has a ✏️ button: edit the
   exact prompt that made it and regenerate in place, without losing the rest.
9. **Train** — the pre-flight check runs the full checklist (count, balance,
   captions, leaks, duplicates); fix what it flags, then launch. Steps are
   computed automatically; ⚙️ Advanced options exposes the knobs, each with its
   own explanation. No GPU? **☁️ Train in cloud** rents one per run.
10. **Pick the best checkpoint** — open the **Test Studio** from the dataset:
    grid-test checkpoint × strength, vote, rank by face similarity, and star ★
    the winning settings. The last checkpoint is almost never the best one.
11. **Export** — at any point, **Export ZIP** gives you the curated, captioned
    set as a standard ai-toolkit dataset. Nothing is locked in.

## Concept datasets (an object or action, not a person)

Pick **Concept** at creation and describe the concept in the required field —
the captioner needs to know exactly *what to omit*. What changes vs character:

- **No reference photo.** Images come from **import** or the built-in
  **scraper** (paste a gallery URL or run a Reddit keyword search, tick the
  frames you want, they land straight in the dataset — deduplicated and
  quality-filtered). On gallery sites (PornPics), a category/tag/search scan
  shows **the same previews the listing page does** — one per gallery, the shot
  that actually matches your keyword. Tick **Scan full albums** to pull every
  photo of each matched gallery instead, or paste a single `/galleries/…` URL
  to get that whole album.

  > **Reddit says "wait N seconds" (429)?** By default Reddit scans share a
  > public client id (and its ~1000 requests / 10 min quota) with many other
  > people, so it can be exhausted before your first scan. Add your own free
  > client ID in **Settings → Scraping & sources** — a one-minute, step-by-step
  > guide is built into that page.
- **Captions invert**: they describe everything *except* the concept, so the
  concept is what binds to the trigger. The leak check watches for stray
  descriptions of it.
- **Masked training is off** (a person mask would erase the very thing you're
  teaching), and imports keep the full frame instead of head-cropping.

## Style datasets (a global aesthetic)

Pick **Style** at creation. What changes:

- **No trigger word** — the style tints every image once the LoRA is loaded.
- **Captions describe content only** (never the rendering), and they're
  optional; caption dropout rises so the style generalizes.
- **Step count switches to a sublinear √n scale** built for the large sets
  (hundreds of images) style LoRAs want.

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

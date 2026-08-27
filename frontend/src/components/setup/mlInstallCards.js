/** The optional ML helpers Setup offers as one-click installs — as DATA.
 *
 * Extracted from SetupPage.jsx for one reason: the capability strips around the
 * app end their "✗ missing" lines with "→ Install … from Setup", and nothing
 * used to guarantee Setup actually OFFERED that install. The video lane shipped
 * whole with both of its extras reachable only through the API — the strip
 * pointed users at a page with no button (found by the first real user, the
 * day the wave landed). JSX does not execute under the bare `node --test`
 * suite, so the promise is only testable if the list lives in a plain module;
 * mlInstallCards.test.js pins every strip hint to a card here.
 *
 * `action` is the backend install action (setup_installer.INSTALL_ACTIONS),
 * `cap` the /api/capabilities key that turns the card's badge green — or an
 * ARRAY of keys when one action installs several pieces that fail apart (the
 * video card ships PyAV *and* a bundled ffmpeg; see cardInstalled below).
 */
export const ML_INSTALL_CARDS = [
  { action: 'face_scoring', cap: 'face_scoring', icon: '🎭', title: 'Face-similarity scoring',
    body: 'Powers the "Analyze faces" pass: scores how closely each generated image resembles your reference photo, so you keep the ones that truly look like the person. It only ranks — it never deletes anything.' },
  { action: 'masks', cap: 'masks', icon: '🧍', title: 'Person masks',
    body: 'Isolates the subject from the background for masked training: the surroundings are weighted down so the LoRA binds the identity to the person, not the room. A training without masks is still valid.' },
  { action: 'watermark_inpaint', cap: 'watermark_inpaint', icon: '🧽', title: 'Watermark inpainting',
    body: 'Repaints small off-center watermarks (LaMa) during 🧽 Clean instead of only cropping border marks. It can use CUDA or CPU from Settings. Without it, off-center marks are skipped.' },
  { action: 'bank_scoring', cap: 'bank_scoring', icon: '✨', title: 'Bank scoring (aesthetic · NSFW · style)',
    body: "Powers the 🗃️ Bank's ✨ Score pass: rates images for aesthetics (1–10), flags NSFW and groups them by visual style with one CLIP pass — and makes 'keep best' prefer the nicest-looking duplicate. Installs into its own Python (CLIP + a small NSFW model). Without it, the Score button is disabled with this hint." },
  { action: 'bank_siglip2', cap: 'bank_siglip2', icon: '🧠',
    title: 'SigLIP 2 semantic engine (optional)',
    body: "Adds Google's general SigLIP2 Base engine as a per-Bank alternative for semantic search, similarity, diversity and crop/variant detection. CLIP remains installed and keeps doing aesthetic, NSFW, style and medium scoring; switching a Bank never deletes either cache. Downloads one pinned Apache-2.0 checkpoint (~1.5 GB) into an LDS-managed Python only when you click Install. If Score uses a GPU Python you already have, that borrowed environment and your Score selection are never changed." },
  { action: 'watermark_detect', cap: 'watermark_detect', icon: '🚩',
    title: 'Watermark detector (faster 🚩 Find)',
    body: "Makes the Bank's 🚩 Find watermarks pass roughly ten times faster and lets it run without Ollama: a small classifier scores each image (~0.14 s instead of ~1.7 s asking the vision model in words), and a second model marks where the logo sits so ✂ Crop and 🧽 Inpaint have something to work on. Adds ~0.9 GB of weights into the scoring Python it shares with ✨ Score. Without it nothing breaks — the vision model keeps doing the same job, slower." },
  { action: 'video', cap: ['video_decode', 'video_encode'], icon: '🎬',
    title: 'Video decoding (the 🎬 Video bank reads your files)',
    body: 'Lets the Video bank open your files at all: read their length, size and frame rate, grab thumbnails, measure quality and cut the clips you promote. Two small packages (PyAV + a bundled ffmpeg) into the app\'s own Python — no torch, no GPU. Without it a video bank can list files and nothing more, and every pass names this install as the missing piece.' },
  { action: 'shot_detect', cap: 'video_detect', icon: '🎞️',
    title: 'Shot detection (triage shots, not whole rushes)',
    body: 'Cuts each video at its shot boundaries (TransNetV2) so a two-hour file becomes hundreds of individually reviewable shots. Installs torch (CPU is fine — the network reads 48×27 frames) and one small package into the scoring Python it shares with ✨ Score. Without it you can still watch and triage whole files; you just cannot split them.' },
  { action: 'video_text', cap: 'video_text', icon: '🔳',
    title: 'Burned-in text (🔤 Find text + the 🔳 Safe zone pass)',
    body: 'One OCR engine, two jobs. On Banks and Datasets it powers 🔤 Find text: speech bubbles, subtitles, captions and sound effects become zones 🧽 Repaint can erase. On the Video bank it lets the 🔳 Safe zone pass find subtitles, chyrons and text watermarks and work out how much of each frame a crop would leave you. One small Apache-2.0 package (RapidOCR) into the app\'s own Python — CPU only, no torch, no GPU, and its ~16 MB of weights ride inside the wheel so it works with no internet. Without it the Safe zone pass still measures letterbox bands ("bands only"), and 🔤 Find text stays greyed with this card as the fix.' },
  // This card is the fix for the exact gap the others already closed: the
  // Concept Sources panel (Datasets ▸ Sources) offers the same install action
  // through its own amber banner, but Setup — the screen a new user actually
  // opens first — had no button for it at all, so the "install the scraper
  // extras" advice from that banner led back to a page with nothing to click.
  { action: 'scrape_extras', cap: 'scrape_deps', icon: '🔎',
    title: 'Scraping extras (gallery links & keyless web image search)',
    body: 'Installs curl_cffi, gallery-dl, cloudscraper, ddgs and yt-dlp — what gallery-URL scraping, the keyless web image search and the video sources use to enumerate and fetch media. Pexels enumeration works without it (its official API); fetching the actual images still needs curl_cffi. Without it, scraping is limited to sources with no anti-bot layer.' },
]

/** The capability keys a card's install is responsible for, always as a list. */
export function cardCaps(card) {
  return Array.isArray(card.cap) ? card.cap : [card.cap]
}

/** True only when EVERY piece the card installs is present.
 *
 *  One action, one badge — but not one probe. `video` installs PyAV and a
 *  bundled ffmpeg, and capabilities.probe_video() reports them apart because
 *  they fail apart: imageio-ffmpeg hands back a path whether or not its binary
 *  download finished, so decoding can be green on a machine with no encoder.
 *  Reading only the first key badged that machine "✓ Installed" while the Video
 *  bank could not cut a clip — and hid the ↻ Reinstall that fixes it behind a
 *  green tick.
 */
export function cardInstalled(card, caps) {
  const c = caps || {}
  return cardCaps(card).every((k) => !!c[k])
}

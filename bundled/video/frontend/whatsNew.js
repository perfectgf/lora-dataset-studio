// The video plugin owns its full update history, including formerly archived
// entries. Keep ids and published copy stable; new video updates belong here.
// Data only: release tooling reads this export without loading the application.
export const WHATS_NEW = [
  { id: '2026-09-26-video-publishing-help', date: '2026-09-26', title: 'Open the publishing guide from video checkpoint help',
    blurb: 'The help link for publishing video checkpoints now opens the available Civitai publishing guide instead of a missing section.', to: '/datasets' },
  { id: '2026-09-25-video-continue-references', date: '2026-09-25', title: 'Keep your selected references when continuing a clip',
    blurb: 'Continue keeps the references selected for the next shot, restoring the parent clip\'s references only when the selection is empty. Reuse restores prompt tags with the matching images. If a removed reference still appears in Motion, a direct action selects that mention for editing.', to: '/studio?lane=video' },
  { id: '2026-09-24-video-gallery-performance', date: '2026-09-24', title: 'Browse large image selections with less overhead',
    blurb: 'The Video Studio Bank and Gallery pickers keep only nearby image tiles mounted and use cached Gallery thumbnails. All loaded images stay reachable, including by keyboard. Finished videos no longer extract and add a last-frame image to the Gallery automatically; Continue and frame previews prepare it on demand.', to: '/studio?lane=video' },
  { id: '2026-09-24-video-clipproj', date: '2026-09-24', title: 'Use a smaller H3 prompt encoder',
    blurb: 'H3 generation now uses Qwen3-VL 4B FP8 with ClipProj v3.1 in place of the 32B prompt encoder. Setup prepares the encoder, projection and nodes, reusing existing files. Restart ComfyUI after installing the nodes. The smaller encoder does not represent the total memory needed to render a video.', to: '/studio?lane=video' },
  { id: '2026-09-23-video-official-int8-vae', date: '2026-09-23', title: 'Prepare the smaller official H3 INT8 VAE',
    blurb: 'INT8 preparation now downloads the official optimized 2.81 GB video VAE. The previous experimental file is detected and replaced only after the new download passes its size and SHA-256 checks. Update LDS before preparing it from the Render panel.', to: '/studio?lane=video' },
  { id: '2026-09-23-video-refmods-no-count-limit', date: '2026-09-23', title: 'Use all your identity RefMods',
    blurb: 'First-frame video accepts identity images without a fixed count limit, from uploads or Library. All selected references reach generation and Auto/Enrich. More references use more memory and processing time.', to: '/studio?lane=video' },
  { id: '2026-09-23-video-reference-lightbox', date: '2026-09-23', title: 'Inspect selected video references in a large preview',
    blurb: 'Click a reference photo or use Enlarge below a reference video to inspect the selected media. Photos offer original-size viewing; videos keep playback controls and the current position. First and last frame guides also open large. Close with Escape, the backdrop or the close button.', to: '/studio?lane=video' },
  { id: '2026-09-23-video-recover-saved-mp4', date: '2026-09-23', title: 'Recover saved clips after ComfyUI history is cleared',
    blurb: 'Old reference clips can now recover their saved MP4 from the configured ComfyUI output folder even after its history is cleared or ComfyUI stops. Recovery checks the embedded generation settings before restoring the player; no new render is needed.', to: '/studio?lane=video' },
  { id: '2026-09-23-video-performance-playback', date: '2026-09-23', title: 'Choose H3 performance controls and recover reference clips',
    blurb: 'Use an existing Fused Turbo model, H3 SageAttention, Spectrum, INT8 video decoding and fast MP4 recording. Prepare optional components from the render panel and restore the settings with Reuse. Old reference clips that saved an input filename recover their actual output on playback when ComfyUI still has its history and MP4. Auto and Enrich now separate identity from the opening scene when replacing a subject in a video.', to: '/studio?lane=video' },
  {"id": "2026-09-23-video-workload-choice", "date": "2026-09-23", "title": "Use the complete video selection", "blurb": "Long-video slicing keeps every complete clip. Compare all selected checkpoints and supply every training sample prompt you need; the sampling controls explain the added time and cloud cost."},
  {"id": "2026-09-23-video-import-selection", "date": "2026-09-23", "title": "Import your full video selection", "blurb": "Video dataset and Video Bank imports no longer reject a selection because it contains more than six videos. Progress, cancellation and duplicate detection remain available.", "to": "/datasets"},
  {
    id: '2026-09-23-create-video-dataset', date: '2026-09-23',
    title: 'Create a video dataset straight from the library',
    blurb: 'Choose Video in New dataset, then add files in Add videos. Clips are encoded to your chosen model, length and size; long videos can be split into consecutive clips. Progress, skipped files and cancellation stay in the dataset.',
    to: '/datasets',
  },
  { id: '2026-09-20-video-refmods-without-frames', date: '2026-09-20', title: 'Generate from identity RefMods without a start frame',
    blurb: 'A prompt and one or two identity images now suffice for an FL2VA clip. Start and end frames are optional; choose the video shape when no start frame is supplied.', to: '/studio?lane=video' },
  { id: '2026-09-20-video-auto-refmod-inputs', date: '2026-09-20', title: 'Use reference cards directly in first-frame video',
    blurb: 'First-frame video now shows the same image reference cards as References mode. Add one or two identity images and they are automatically encoded as RefMods, with no switch to enable. Start and end frames keep their separate roles.', to: '/studio?lane=video' },
  { id: '2026-09-19-video-chimera-refmods', date: '2026-09-19', title: 'Keep identities across first-frame continuations',
    blurb: 'Add one or two identity RefMods to H3 First frame clips. Continue restores them and Auto carries them through the take; TaoMate H3 (3 steps) and FastH3 v0.2 are available in Render.', to: '/studio?lane=video' },
{
    id: '2026-09-09-video-smooth-landscape', date: '2026-09-09',
    title: 'Smooth controls stay reachable on a landscape phone',
    blurb: 'The Smooth dialog now sits above the Studio action bar, so its Cancel and Smooth buttons remain visible and clickable on short screens.',
    to: '/studio?lane=video',
  },
{
    id: '2026-09-09-video-smooth-audio-count', date: '2026-09-09',
    title: 'Smooth clips with their sound and the right frame count',
    blurb: 'Smooth keeps source audio when present and counts the frames RIFE inserts between each pair: 56 frames become 111 at ×2. The dialog explains the slightly shorter saved duration. Original clips stay unchanged.',
    to: '/studio?lane=video',
  },
{
    id: '2026-09-09-video-queue-owner', date: '2026-09-09',
    title: 'Recognize Video clips in the generation queue',
    blurb: 'Clips queued by Video now appear under Video Test Studio in the global queue. Their priority, cancellation and completion still use the same queue and clip history.',
    to: '/studio?lane=video',
  },
{
    id: '2026-09-08-vdn-reference', date: '2026-09-08',
    title: 'Try VDN-H3 with your video references',
    blurb: 'Reference videos can now use OpenVDN hybrid attention through Saganaki22’s ComfyUI-VDN-H3 pack. Compare it with LightX using the same references; Sparse stays off and sampling steps remain adjustable. Available locally and on one rented ComfyUI GPU.',
    to: '/studio?lane=video',
  },
{
    id: '2026-09-07-zzzz-continue-a-reference-clip',
    date: '2026-09-07',
    title: 'Continue a References clip without losing its characters',
    blurb: '⏭ Continue on a clip made from references now keeps that cast: its last frame becomes the first frame guide, the same references and base are reloaded, and the render still lands joined behind it. And a continuation can no longer go missing — whichever mode you switch to afterwards says whether this Generate continues the clip or renders a new one.',
    to: '/studio?lane=video',
  },
{
  "id": "2026-09-07-video-live-separate",
  "date": "2026-09-07",
  "title": "Live is now installed separately",
  "blurb": "Video keeps its Bank, datasets, training and clip Test Studio. The Live channel now has its own complete plugin, with its player, controls, setup and help. Install Live from the Store to keep using that channel; existing H3 models are reused.",
  "to": "/plugins"
},
{
    // Same-day ids sort the feed (date, then id): 'zzzz' keeps this one above
    // the day's earlier entries, so the badge counts it (2026-09-06).
    id: '2026-09-06-zzzz-video-prompt-survives-reload',
    date: '2026-09-06',
    title: 'The motion you typed survives a page reload',
    blurb:
      'In the video studio, the Motion field comes back as you left it after a '
      + 'refresh or a trip to another page — tags included, in every mode — the way '
      + 'the references, their roles and the render dials already did. It is kept in '
      + 'this browser as you type; clearing the field clears it.',
    to: '/studio?lane=video',
  },
{
    // Same-day ids sort the feed (date, then id): 'zzz' keeps this one above
    // the day's earlier entries, so the badge counts it (review, 2026-09-06).
    id: '2026-09-06-zzz-video-reference-cut',
    date: '2026-09-06',
    title: 'Cut a reference video to the part you need',
    blurb:
      'Under every video reference in the References panel, Cut keeps an interval of '
      + 'the video (Start and Duration, 2–15 s) and swaps the shorter excerpt in place: '
      + 'same tag, same role, same format choice, and the card says what it was cut '
      + 'from. What it changes: the render reads a reference video up to the clip’s '
      + 'length, so only a cut shorter than the clip lightens the sequence — 107 '
      + 'reference frames instead of 175 for a 5 s cut behind a 7.3 s clip, about a '
      + 'sixth of the sequence tokens; the card shows the threshold. On a 24 GB card, '
      + 'that is one of the levers against a reference render that pages, with the '
      + 'resolution, the clip length and the picture count.',
    to: '/studio?lane=video',
    image: new URL('./assets/news/reference-cut.png', import.meta.url).href,
  },
{
    // Same-day ids sort the feed (date, then id): 'zz' keeps this one above
    // the day's earlier entries, or the badge would count nothing new for
    // anyone who had already opened the panel today (review, 2026-09-06).
    id: '2026-09-06-zz-video-joycaption-stills-and-take-lora',
    date: '2026-09-06',
    title: 'Video prompts read the picture as it is: JoyCaption for every frame, and Auto keeps the take’s LoRA',
    blurb:
      'Motion → ⚙ now lets JoyCaption read the start frame, the last frame, the '
      + 'frame guides and the reference pictures — not only reference videos — so '
      + 'an explicit frame reaches the writer described as it is, nothing softened '
      + 'or swapped. With the vision model, the still is read with its reasoning '
      + 'switched off and a sharper brief: a thinking model was handing the writer '
      + 'its own deliberation instead of the description. And an Auto continuation '
      + 'keeps the character LoRA of the clip it continues (or of the nearest part '
      + 'above it that carries one), whatever the panel shows — a take no longer '
      + 'drops its character because the panel was empty when Auto was pressed; '
      + 'the Auto panel names the LoRA it kept.',
    to: '/studio?lane=video',
  },
{
    id: '2026-09-06-video-vdn-h3-acceleration',
    date: '2026-09-06',
    title: 'VDN-H3: a fourth video acceleration, a different attention rather than another LoRA',
    blurb:
      'The Video Test Studio’s ⚡ Acceleration menu gains OpenVDN’s VDN-H3: a '
      + 'linear-attention branch laid over the same MiniMax H3 base, rendered in '
      + '8 steps on the base’s own schedule — near-lossless against the dense '
      + 'model by its authors, and an attention cost that grows with clip length '
      + 'instead of its square. Not faster than turbo on one card (about twice '
      + 'the time per clip, closer on long ones): pick it to compare quality, and '
      + 'for long clips. On a 24 GB card tick the Lighter base (W4A8) with it — '
      + 'the app says so and refuses the 21 GB base, which pages for minutes. '
      + 'Setup downloads the stage (5.5 GB, or a 2.3 GB INT8 branch); the '
      + 'ComfyUI-VDN-H3 node pack is named and linked from the install card.',
    to: '/studio?lane=video',
  },
{
    id: '2026-09-05-zzzz-reference-clip-continue',
    date: '2026-09-05',
    title: 'Continue a clip made from references — the next part is image-to-video',
    blurb: 'A clip rendered in References now offers ⏭ Continue like any other: its last frame becomes the next start frame, the studio switches to From an image, and the render lands joined behind it. ✨ Auto and ✨ Enrich read the reference clip’s movement — its detailed description, not its subject definitions — so the next part carries the take on.',
    to: '/studio?lane=video',
  },
{
    id: '2026-09-05-zzzz-reference-video-joycaption',
    date: '2026-09-05',
    title: 'Let JoyCaption read your reference videos, three stills at a time',
    blurb: 'Motion → ⚙ gains a choice for reference videos. Keep the vision model’s two-frame read, or let JoyCaption describe the first, middle and last frame separately — explicit footage included — and hand all three, timestamped, to the writer, which then writes the movement between them. Pictures are read as before; when JoyCaption is not installed, Auto and Enrich say so instead of quietly using the other method.',
    to: '/studio?lane=video',
  },
{
    id: '2026-09-05-zzz-reference-video-format',
    date: '2026-09-05',
    title: 'Keep the format of a reference video',
    blurb: 'Tick the format option on a reference video to generate with its proportions, including portrait and unusual ratios. Resolution stays adjustable, and Reload and Reuse remember the video you chose.',
    to: '/studio?lane=video',
    image: new URL('./assets/guide/video-studio/reference-video-format.png', import.meta.url).href,
  },
{
    id: '2026-09-05-zz-reference-roles-sparse',
    date: '2026-09-05',
    title: 'Keep reference characters distinct and choose Sparse attention',
    blurb: 'Group pictures with character 1 and character 2 roles so Auto and Enrich include both people, while a motion reference supplies their movement. Incomplete drafts get one automatic repair. Reference renders now offer Sparse attention, including with the dedicated 4-step and 8-step accelerations.',
    to: '/studio?lane=video',
  },
{
    id: '2026-09-05-z-reference-library',
    date: '2026-09-05',
    title: 'Pick video references directly from your library',
    blurb: 'Choose reference images from Bank, Gallery and datasets, and reference videos from Video Bank, video datasets or rendered clips. Select a video excerpt, borrow its audio track or use a first or last frame, including for temporal guides.',
    to: '/studio?lane=video',
    image: new URL('./assets/guide/video-studio/reference-library.png', import.meta.url).href,
  },
{
    id: '2026-09-05-video-graph-previews',
    date: '2026-09-05',
    title: 'Compare video checkpoints on the same motion',
    blurb: 'Select H3 saves in a video dataset’s run graph and render them with one prompt, seed and set of frames. Reopen the history or compare two clips with synchronized playback; training samples stay available too.',
    to: '/datasets',
    image: new URL('./assets/guide/video/checkpoint-previews.png', import.meta.url).href,
  },
{
    id: '2026-09-04-zzzzzzzzzz-video-civitai-best-settings',
    date: '2026-09-04',
    title: 'Share your video LoRA and keep the settings that worked',
    blurb: 'Link a video checkpoint to Civitai or prepare its model page from the list and run graph, with both Wan experts on one version and draft selected by default. Star a successful Studio render to save its settings, then apply them again without replacing your prompt or images.',
    to: '/studio?lane=video',
  },
{
    id: '2026-09-04-zzzzzzzzzz-video-auto-continuation',
    date: '2026-09-04',
    title: 'Let a video scene carry on by itself',
    blurb: 'Check Auto beside Continue to turn each last frame into the next clip, with a fresh motion prompt and your scene direction. Set a clip limit, stop after the current clip, or resume after an error.',
    to: '/studio?lane=video',
  },
{
    id: '2026-09-04-zzzzzzzzzz-h3-reference-video',
    date: '2026-09-04',
    title: 'Make a video from people, motion and sound references',
    blurb: 'The Video Studio’s References mode combines images, videos and audio in one clip. Give each reference a role, let Auto or Enrich write the motion, and compare dedicated 4- or 8-step acceleration with the dense model. Setup installs the reference base and acceleration you choose; Reuse restores the references with the render settings.',
    to: '/studio?lane=video',
  },
{
    id: '2026-09-04-zzzzzzzzz-video-training-controls',
    date: '2026-09-04',
    title: 'Give video training its preview prompts, rank and memory mode',
    blurb: 'Set up to four sample prompts for the run graph, choose the LoRA rank and control low VRAM from the video dataset. The same controls apply locally and in the cloud; retries and continuations keep the cloud run’s original settings.',
    to: '/datasets',

  },
{
    id: '2026-09-04-zzzzzz-writers-follow-the-take-and-the-last-frame',
    date: '2026-09-04',
    title: '✨ Auto and ✨ Enrich now follow the take — the previous clips’ motion, and the last frame you picked',
    blurb:
      'On a ⏭ Continue, the writers are handed the motion of the clip being '
      + 'continued and of the two before it, most recent first, and asked to '
      + 'write what happens NEXT — same people, wardrobe, setting and camera '
      + 'language, no restart, no repeat. And when a last frame is staged, they '
      + 'describe it as a second still and write a movement that lands on it '
      + '— ✨ Enrich on a text-only start too. The per-picture writer and ✨ Enrich at '
      + 'launch get the same.',
    to: '/studio?lane=video',
  },
{
    id: '2026-09-04-zzzz-start-frame-viewer-and-last-frame-tab',
    date: '2026-09-04',
    title: 'Pick the LAST frame too — H3 renders from your first frame to it — and open any frame in the Gallery’s viewer',
    blurb:
      'Click a start frame in the strip and it opens in the same viewer as the '
      + 'Gallery, with the same buttons: ↓ Download, ✨ Improve via Klein, 🔍 '
      + 'Upscale via SeedVR2, ✦ Repair, 📷 Camera angles, 📤 Civitai, and ‹ › '
      + 'to walk the strip. The frame gets a library row the first time it '
      + 'opens (once per picture — in the dataset the Studio was opened from, '
      + 'or in a dataset named “Video Test Studio · frames” on the plain '
      + 'Studio), the improve and camera results land in the '
      + 'Gallery next to it, and a ✦ Repair is staged back into its slot so the '
      + 'next clip starts from the repaired picture. Above the containers, '
      + 'Stage as: First frame / Last frame says which frame a pick becomes: '
      + 'MiniMax H3 is a first-last-to-video model, and a clip given both '
      + 'resolves from one picture onto the other (one last frame per launch, '
      + 'shared by every clip of a batch; in Text only the containers serve the '
      + 'last frame alone). A Rendered clip tab offers every finished clip’s '
      + 'last frame as a picture to pick — for either frame; ⏭ Continue on the '
      + 'card stays the way to join, and the card says “ends on a picture”. And every '
      + 'finished clip’s last frame is now collected into the Gallery on its '
      + 'own — one picture per clip, in that same “Video Test Studio · frames” '
      + 'dataset — so the Gallery tab, the Gallery page and the viewer’s '
      + 'Improve and Repair reach it like any picture. And the pictures the '
      + 'Studio stages for ComfyUI no longer pile up in its input folder: each '
      + 'clip keeps its own copy of the frames it was made from, so ↻ Reuse '
      + 'works on a clip of any age, and staged pictures older than 48 hours '
      + 'are cleared when the app starts.',
    to: '/studio?lane=video',
  },
{
    id: '2026-09-04-zzz-shots-with-timecodes',
    date: '2026-09-04',
    title: 'Cut a clip into shots — the writers put timecodes where you ask',
    blurb:
      'A Shots row now sits under the Motion field (1 to 6). It tells ✨ Auto, '
      + '✨ Enrich, the per-picture batch and the enrich-at-launch rewrite how '
      + 'many shots to write: 1 is one continuous take, which in H3’s format '
      + 'carries no timecode; 2 or more cut the clip at even timecodes — '
      + '“[Shot 2] At 00:05.000, the camera cuts to…”. The server always knew '
      + 'how; the panel never asked, so a fifteen-second clip enriched as one '
      + 'take and no timestamp ever appeared. Choices a short clip cannot hold '
      + 'are greyed (one shot per second at most under five seconds).',
    to: '/studio?lane=video',
  },
{
    id: '2026-09-04-zz-lighter-w4a8-base',
    date: '2026-09-04',
    title: 'A lighter MiniMax H3 base for the Video Test Studio — 12.5 GB, a little faster, 3.8 GB less VRAM once resident',
    blurb:
      'The Render panel gains 🪶 Lighter base (W4A8): the official H3 weights '
      + 'quantized to 4-bit by Winnougan (apache-2.0), 12.5 GB on disk instead of 21. '
      + 'Measured on the reference 24 GB card with the same image, prompt and '
      + 'seed, on ComfyUI’s own clock: 13.8 s a clip against 16.7 at the default '
      + '6-step acceleration, level at 20 dense steps; once the base is resident, '
      + 'peak VRAM 19.5 GB against 23.3 — on '
      + 'the first clip after a base change the two are level; and that first '
      + 'clip a few seconds shorter, there being less to read. Setup has a '
      + 'button for it, and the box says so when your ComfyUI is older than '
      + '0.31, the first version that reads the format. Its fast path needs a '
      + 'GPU of compute capability 8.0 or newer (RTX 30-series and up); older '
      + 'cards still render it, unaccelerated. One thing it does not keep is '
      + 'the seed: the two bases render two different clips from one seed, a '
      + 'third of a reseed to slightly more apart, so compare within a base. '
      + '⏭ Continue keeps the base of the clip it continues.',
    to: '/studio?lane=video',
  },
{
    id: '2026-09-04-z-motion-writer-dials-and-clear',
    date: '2026-09-04',
    title: 'The motion writer has dials now — two temperatures, length, sampling — and a ⌫ Clear',
    blurb:
      'The ⚙ beside ✨ Auto and ✨ Enrich used to pick only the model. It now '
      + 'sets how it writes: a temperature for Auto (which invents from a still '
      + 'and wants room) and a separate one for Enrich (which rewrites your text '
      + 'and must stay close to it), the length budget, top-p, top-k, min-p and '
      + 'the presence penalty. Every value is clamped to what your provider '
      + 'accepts and the window shows what was kept. ⌫ Clear empties the Motion '
      + 'field in one press — worth having because the ✨ buttons read what is '
      + 'there. And a fix on the way: a Written-per-picture batch was ignoring '
      + 'the text in the field when it proposed; it steers every picture now, as '
      + 'Auto does on one.',
    to: '/studio',
  },
{
    id: '2026-09-03-video-test-studio-prompts-per-picture',
    date: '2026-09-03',
    title: 'A batch of pictures, one prompt each',
    blurb:
      'With several start frames in the strip, the Video Test Studio now asks '
      + 'which prompt the batch runs: the same motion for every picture (as '
      + 'before, the comparison that says something about the LoRA), or one '
      + 'written by ✨ per picture — your motion enriched with each frame, or '
      + 'a proposal from the frame alone — written before anything is queued.',
    to: '/studio?lane=video',
  },
{
    id: '2026-09-03-video-test-studio-continue-from-last-frame',
    date: '2026-09-03',
    title: 'Continue a clip from its last frame — and get one video',
    blurb:
      'A ⏭ Continue button on every finished clip of the Video Test Studio: '
      + 'its last frame becomes the next start frame, you write the next '
      + 'motion, and the render lands joined behind the clip it continues — '
      + 'one video, that clip then the new one, the cut frame dropped and the '
      + 'sound kept in step (a smoothed clip has none: its side is padded with '
      + 'silence, the new part keeps its own). Continue the result again and '
      + 'the chain grows.',
    to: '/studio?lane=video',
  },
{
    id: '2026-09-03-video-accelerations',
    date: '2026-09-03',
    title: 'Pick your video acceleration among the arena’s top three',
    blurb:
      'The Video Test Studio’s Turbo box is now a choice: larryvrh’s Turbo v4 '
      + '(as before), Plaguekind’s Parasyte Turbo or silveroxides’ DARE-TIES '
      + 'merge — the first three rows of the MiniMax-H3 acceleration arena, '
      + 'statistical ties at six steps. Each runs with the settings the arena '
      + 'verified; Setup downloads the two new LoRAs, and a clip remembers '
      + 'which one made it.',
    to: '/studio?lane=video',
  },
{
    id: '2026-09-03-video-run-graph-and-previews',
    date: '2026-09-03',
    title: 'A video set draws its runs as a graph, with the training samples on each save',
    blurb:
      'The Checkpoints & LoRAs section of a video set now opens on the same run graph '
      + 'an image dataset has: one card per run, this PC or a rented pod, a pill per '
      + 'save, and a curve from the exact step a continuation resumed from. A save that '
      + 'training rendered a sample for shows its still — click it to play the clip, '
      + 'one prompt after another — and every verb (download, deploy, continue, delete) '
      + 'is one click away on the pill, the same as in the list below.',
    to: '/datasets',
  },
{
    id: '2026-09-03-video-smooth-rate',
    date: '2026-09-03',
    title: 'Smooth asks which rate you want',
    blurb:
      'The ↗ Smooth button of the Video Test Studio opens a small window before '
      + 'it runs: 48, 72 or 96 fps for a 24 fps clip — ×2, ×3 or ×4, because the '
      + 'interpolator works by whole factors — with the frame count and the '
      + 'relative cost of each. It used to go straight to 48.',
    to: '/studio?lane=video',
  },
{
    id: '2026-09-03-video-quick-prompts',
    date: '2026-09-03',
    title: 'Pick a video prompt instead of writing one',
    blurb:
      'A ⚡ row of preset chips now sits under the Video Test Studio’s Motion '
      + 'field: Scenarios, Multi-Shot, Timeline, Camera, Audio, Voice and Visual '
      + 'Style, written in H3’s own prompt format. They stack rather than '
      + 'replace — take a scenario, add a camera move, add an audio bed, and each '
      + 'one lands on its own line under what you already wrote. In a text-only '
      + 'clip the presets drop their reference to a start frame, because there '
      + 'is not one.',
    to: '/studio?lane=video',
  },{
    id: '2026-09-03-video-studio-render-time',
    date: '2026-09-03',
    title: 'Every clip in the Video Test Studio says how long it took to render',
    blurb:
      'The card of a finished clip now reads "rendered in 24 s" — or "5 min '
      + '48 s" — the time from the moment the queue took the job to the moment '
      + 'the clip landed, model loading included. It is the number that tells '
      + 'a good run from a machine that is swapping: the same clip, same card, '
      + 'same evening, took five minutes with one launch flag and twenty-five '
      + 'seconds with another.',
  },
{
    id: '2026-09-02-video-studio-start-frame-batch',
    date: '2026-09-02',
    title: 'Test a video LoRA on several start frames in one click',
    blurb:
      'The Test Studio’s start frame is now a strip: pick several pictures '
      + '(several files at once, or tiles from a bank, the Gallery or a training '
      + 'set) and Generate queues one clip per frame on the same seed and the '
      + 'same prompt — ✨ Enrich rewrites it once, for the first clip — so the '
      + 'clips differ by their picture and nothing else. Each frame has its ✕, '
      + 'the button says how many clips a click queues, and dropping a file '
      + 'onto the picker works again.',
    to: '/studio?lane=video',
  },
{
    id: '2026-09-02-video-studio-fast-disk',
    date: '2026-09-02',
    title: 'Video clips in seconds instead of minutes on machines whose RAM cannot hold the H3 weights',
    blurb:
      'The Video Test Studio loads about 43 GB of weights, and ComfyUI keeps a '
      + 'copy of everything it offloads in system RAM — on a 48 GB machine a '
      + '56-frame clip took five to six minutes, nearly all of it swapping '
      + 'models. ComfyUI started from the Setup screen now runs with '
      + '--fast-disk, which reads the weights from disk instead: the same clip '
      + 'takes 20 to 30 seconds. When ComfyUI was started some other way, the '
      + 'Studio says so and names the flag to add — or the one to drop, when a '
      + 'launcher still switches the dynamic loader off.',
    to: '/studio?lane=video',
  },{
    id: '2026-09-02-video-studio-preview-size',
    date: '2026-09-02',
    title: 'A Preview size slider in the Video Test Studio’s start frame picker',
    blurb:
      'The Bank and Gallery tabs showed their pictures at one small size, '
      + 'and a face in a tile that small is a smudge. A 🔍 slider '
      + 'above the grid now enlarges the tiles more than three times over — one size for '
      + 'the three tabs, remembered by your browser — so the frame is chosen '
      + 'by eye, not by file name.',
    to: '/studio?lane=video',
  },
{
    id: '2026-09-02-video-studio-active-state-painted',
    date: '2026-09-02',
    title: 'The Video Test Studio shows which tab, mode, LoRA and lane are active',
    blurb:
      'The start frame tabs, the image/text toggle, the chosen LoRA and '
      + 'presets, the selected take and the Images/Video switch were styled '
      + 'with a colour the theme never defined, so their active state never '
      + 'showed. They take the app’s amber now, like every other picker.',
    to: '/studio?lane=video',
  },
{
    id: '2026-09-02-video-studio-dataset-clip-posters',
    date: '2026-09-02',
    title: 'Dataset clip in the Video Test Studio shows its clips as pictures — and works',
    blurb:
      'The start frame picker’s Dataset clip tab listed the clips of a '
      + 'training set as a column of file names that a set of any size '
      + 'squashed into unreadable slivers. It is a grid of posters now, the '
      + 'same frame the training set’s own page shows for each clip, with '
      + 'the file name under it — pick the shot by eye, and the picked frame '
      + 'appears beside “Ready” instead of a blank icon.',
    to: '/studio?lane=video',
  },
{
    id: '2026-09-02-video-checkpoints-and-loras',
    date: '2026-09-02',
    title: 'A video set gets its Checkpoints & LoRAs section — deploy, continue, clear, step by step',
    blurb:
      'Every save a video training brought back — from this PC or from a rented '
      + 'pod — now has its own section in the video workspace, listed by step so '
      + 'both experts of a Wan 2.2 pair travel together. Each step offers what an '
      + 'image dataset’s does: ⬇ download, 📦 deploy into ComfyUI’s loras folder '
      + '(the Video Test Studio lists it as deployed right away), ⏏ undeploy, '
      + '▶ continue training from that exact step on a fresh pod, ⓘ the run’s '
      + 'details, and 🗑 delete — to the app’s Trash, recoverable. A Studio section '
      + 'opens the Video tab of the Test Studio next door.',
    to: '/datasets',
  },
{
    id: '2026-09-01-video-dataset-workspace',
    date: '2026-09-01',
    title: 'Your video training sets finally have a screen of their own',
    blurb:
      'A video set used to be a card at the bottom of the library: a list of file '
      + 'names, one caption box each, and nothing else. Opening one now opens a full '
      + 'workspace — a grid of every clip with the rush and timecode it came from, a '
      + 'player you can step through with the arrow keys, a search, and caption tools '
      + 'that rewrite the .txt files in bulk. Same shape as an image dataset, because '
      + 'it is the same job.',
    to: '/datasets',
  },
{
    id: '2026-09-01-remove-a-clip-from-a-video-set',
    date: '2026-09-01',
    title: 'Drop a bad clip without re-cutting the whole set',
    blurb:
      'The three-frame clip of somebody’s hand is only visible after the encode, and '
      + 'until now the only way out was deleting the dataset and promoting it again. '
      + 'Remove it from the set instead: its .mp4 and .txt go, and the bank keeps the '
      + 'shot, its bounds and every decision — so you can re-cut and promote it again '
      + 'with no triage to redo.',
    to: '/datasets',
  },
{
    id: '2026-09-02-start-frame-clip-tab-fixed',
    date: '2026-09-02',
    title: 'Picking a start frame from a training clip no longer takes the page down',
    blurb:
      'In the Video Test Studio, opening the “Dataset clip” tab and choosing a '
      + 'training set blanked the screen: the clip list was read from the wrong '
      + 'field and a count arrived where the clips were meant to be. It lists '
      + 'them properly now, and says so plainly when a set holds none.',
    to: '/datasets',
  },
{
    id: '2026-09-01-motion-auto-instructed',
    date: '2026-09-01',
    title: '✨ Auto obeys what you type, and you choose the model behind it',
    blurb:
      'Type what should happen — "make her jump twice", "slower" — and ✨ Auto '
      + 'follows it while keeping the people your start frame actually shows; '
      + 'leave the field empty and it proposes freely. ✨ Enrich picks the same '
      + 'two modes by itself. The ⚙ beside them opens the model window: the '
      + 'motion writer is its own setting, so tuning it never re-points your '
      + 'image passes.',
  },
{
    id: '2026-09-01-motion-auto-and-enrich',
    date: '2026-09-01',
    title: 'The Motion field writes itself, in H3’s own words and paced to your clip',
    blurb:
      '✨ Auto reads your start frame and proposes the movement; ✨ Enrich '
      + 'rewrites what you wrote with more of the detail a sampler can use — '
      + 'both leave the text yours to edit. And a toggle enriches at launch, '
      + 'recording on the clip the prompt that actually ran. All three write '
      + 'the official three-field prompt the model was trained on, paced to '
      + 'the length you set (a 1 s clip and a 15 s one are no longer given the '
      + 'same beat), and — when there is a start frame — name it so the subject '
      + 'stays who it is; a prompt you typed yourself gets that first-frame line '
      + 'at launch, once. They go through the local model you already run for '
      + 'the image passes, or another one if you pick it under ⚙.',
  },
{
    id: '2026-09-01-video-studio-smooth-vfi',
    date: '2026-09-01',
    title: 'Smooth a test clip to twice its frame rate',
    blurb:
      'Every finished clip gains a ↗ Smooth button: RIFE frame interpolation, '
      + 'the same recipe the image generator runs (rife49, ×2, ensemble), so a '
      + 'clip smoothed here is the clip smoothed there. It makes a NEW clip at '
      + 'double the rate and the same duration — the original stays, because '
      + 'comparing the two is the point.',
  },
{
    id: '2026-09-01-reuse-brings-the-start-frame-back',
    date: '2026-09-01',
    title: '↻ Reuse gives the start frame back, and any LoRA can be imported',
    blurb:
      'Reusing an image-to-video clip restored every dial — model, steps, '
      + 'length, seed — and left the start frame empty, so Generate stayed '
      + 'blocked. It comes back now. And the LoRA picker gained an import: give '
      + 'it a path on this machine or choose the file, and it lands in '
      + 'ComfyUI’s folder ready to test — no more moving files by hand.',
  },
{
    id: '2026-09-01-video-studio-length-to-15s',
    date: '2026-09-01',
    title: 'Clips up to 15 seconds, on a slider instead of a 21-row list',
    blurb:
      'The length list stopped at 209 frames (8.7s) because it was reading the '
      + 'TRAINING catalogue — the model renders to 15s and the server always '
      + 'accepted it. Every legal length from 0.88s to 15.04s is now on one '
      + 'slider that snaps to what the VAE accepts, with the seconds and the '
      + 'frame count above it and both ends of the range in view.',
  },
{
    id: '2026-09-01-video-studio-steps-dial',
    date: '2026-09-01',
    title: 'The sampling steps are a dial now, not a decision made for you',
    blurb:
      'The Video Test Studio ran 6 steps with Turbo and 20 without, and nothing '
      + 'on screen let you move that — the one number that plainly trades time '
      + 'for fidelity. There is now a Sampling steps slider (4 to 40) that says '
      + 'what auto resolves to, and an explicit count wins over Turbo’s own. '
      + '↻ Reuse replays the count a clip really ran.',
  },
{
    id: '2026-09-01-start-frame-from-the-gallery',
    date: '2026-09-01',
    title: 'Animate an image straight from the Gallery',
    blurb:
      'The Video Test Studio’s start frame took an upload, a Bank image or '
      + 'a dataset clip — but not the picture this app had just generated, which '
      + 'meant exporting it to disk to feed it back in. The Gallery is now a '
      + 'fourth source: pick any generated image and it is staged at full size.',
    to: '/datasets',
  },
{
    id: '2026-09-01-video-prep-in-one-button',
    date: '2026-09-01',
    title: '▶ Run everything now really runs everything',
    blurb:
      'The chain stopped after thumbnails while its own tooltip promised more. '
      + 'It now offers every preparation pass — measure, embeddings, duplicates '
      + 'and camera — in a launch window where you tick what you want, in the '
      + 'order each one needs the previous. Describe shots stays its own button: '
      + 'its wording changes what the captions say.',
  },
{
    id: '2026-09-01-slice-long-shots',
    date: '2026-09-01',
    title: 'Long shots can give several training clips instead of one',
    blurb:
      'A 15-second shot built at 209 frames used to train on its first 8.7 '
      + 'seconds and the rest was never used. Tick “Slice shots longer than one '
      + 'clip” when building a set and it gives whole clips end to end instead — '
      + 'up to 8 per shot. Each slice carries its shot’s caption, so the window '
      + 'says it plainly.',
  },
{
    id: '2026-09-01-max-shot-length-cut',
    date: '2026-09-01',
    title: 'A maximum shot length, next to the minimum',
    blurb:
      'The Quality cuts had a floor and no ceiling, so the shots your target '
      + 'will truncate were invisible. Set a maximum and they are flagged '
      + '“Longer than a clip” — filter them, cut them by hand, or slice them at '
      + 'build time. It flags and sorts; it never rejects anything.',
  },
{
    id: '2026-09-01-clip-length-suggestion',
    date: '2026-09-01',
    title: 'The clip-length picker tells you what each length costs',
    blurb:
      'Building a set now says how long your kept shots actually run and how '
      + 'many of them each length keeps whole — “141 frames keeps 87% of them”. '
      + 'It never changes your choice, it just stops the default from being a '
      + 'guess.',
  },
{
    id: '2026-09-01-video-bank-wears-the-bank-shell',
    date: '2026-09-01',
    title: 'The video bank now looks and works like the image bank',
    blurb:
      'Same shell, same gestures: filters live in a rail beside the shot grid '
      + '(a drawer on a phone), the analysis passes open on demand from the ⚙ '
      + 'button, the two decisive actions — ▶ run the pipeline and 🎬 build a '
      + 'training set — sit in the top bar next to the same stats strip, and '
      + 'every chip and button is the one the image bank already taught you. '
      + 'Nothing moved in what the passes do — only where you reach them.',
  },
{
    id: '2026-09-01-video-captions-follow-what-you-installed',
    date: '2026-09-01',
    title: 'Video captions now run on what your machine has — Ollama and LM Studio included',
    blurb:
      'No torch Python? If Ollama or LM Studio is running, 🗣 Describe shots '
      + 'captions through it — the same local server and vision model your image '
      + 'passes already use — instead of showing a dead ✗. LDS\u2019s own '
      + 'transformers worker stays the default when available (it feeds the '
      + 'model real frame timestamps and measures captions in the encoder\u2019s '
      + 'own tokens), the launch window says which engine will run, and every '
      + 'caption records which engine wrote it.',
  },
{
    id: '2026-09-01-video-captions-fit-the-encoder',
    date: '2026-09-01',
    title: 'Video captions that fit the model — measured in its own tokens',
    blurb:
      '🗣 Describe shots now ends each caption with a short structured tail '
      + '(Subject, Motion, Setting, Style) and, when umT5\'s tokenizer is on your '
      + 'machine, counts the caption in the Wan encoder\'s own tokens instead of '
      + 'guessing from words. Building a training set uses both: a prompt that '
      + 'would overrun the encoder window (512 tokens on Wan, which cuts in '
      + 'silence) is written in its short form instead of being truncated '
      + 'mid-sentence, and the export tells you how many.',
  },
{
    id: '2026-09-01-video-captions-work-under-transformers-5',
    date: '2026-09-01',
    title: 'Describe shots works again on machines whose Python carries transformers 5',
    blurb:
      'Every shot of a caption pass was failing there — with no reason shown '
      + 'anywhere. The pass now runs on transformers 4 and 5 alike, and when a '
      + 'shot is refused the reason lands in the log instead of vanishing.',
  },
{
    id: '2026-08-31-video-test-studio',
    date: '2026-08-31',
    title: 'Play your video LoRA back, without leaving the app (beta)',
    blurb:
      'Training a video LoRA gave you a file and a loss curve, and judging it '
      + 'meant wiring a graph in ComfyUI by hand. The Test Studio now has a '
      + 'Video tab: pick a LoRA you trained (it is copied into ComfyUI for you '
      + 'the first time), give it a start frame — uploaded, from a bank, or the '
      + 'first frame of a clip in a training set — or none at all for '
      + 'text-to-video, describe the motion, and get a clip. ⚡ Turbo renders in '
      + 'minutes instead of tens of minutes, sparse attention and the latent '
      + 'upscale trade a little fidelity for speed, and every clip keeps the '
      + 'settings that made it so Reuse can rerun the same seed with one dial '
      + 'moved. New here? Setup ▸ 🎬 Video Test Studio downloads the engine '
      + '(about 39.5 GB); the clip itself needs no ComfyUI add-on at all, and '
      + 'the three optional accelerators are named and linked for you to '
      + 'install on the ComfyUI side. Marked beta while the first clips come '
      + 'back from real machines.',
    to: '/studio?lane=video',
  },
{
    id: '2026-08-30-delete-video-runs',
    date: '2026-08-30',
    title: 'Delete old video training runs — and a Beta label that says so',
    blurb:
      'Every run a video dataset ever made stayed on its card forever — '
      + 'smoke tests, superseded step counts, all of it. Each checkpoint '
      + 'group now has a 🗑 that removes that run’s LoRA files and its '
      + 'history line (with a confirmation that counts the files; an active '
      + 'run must be stopped first, and the dataset itself is never touched). '
      + 'And the training block now wears a Beta chip — the rail is proven '
      + 'end to end, but it is days old, and the label says exactly that. '
      + 'The whole Video training sets section also folds away now, like the '
      + 'two dataset sections above it.',
    to: '/datasets',
  },
{
    id: '2026-08-30-caption-budget-and-audio',
    date: '2026-08-30',
    title: 'The video export counts your words, and tells the truth about sound',
    blurb:
      'Two silent failures now speak up at Build the dataset. If captions run '
      + 'past the target model’s own published prompt budget (Wan caps at '
      + '200 words, 100 for I2V), the export says how many and how long the '
      + 'longest is — because the trainer would cut them mid-sentence without '
      + 'a word. And for targets that keep their audio (MiniMax H3), each '
      + 'clip’s prompt gains a measured Audio line when the numbers prove one '
      + '— a missing track, or near-total silence. Audible audio gets no '
      + 'invented description: only what was measured gets written.',
    to: '/video-bank',
  },
{
    id: '2026-08-30-sota-video-captions',
    date: '2026-08-30',
    title: 'Video captions grow up: a full paragraph, built like the measurements say',
    blurb:
      'Describe shots now writes 150-200 words per shot instead of a sentence '
      + 'or two — the length the published ablations converge on, where the '
      + 'whole gain lands on MOTION, exactly what a video LoRA learns. It '
      + 'watches 16 frames instead of 8 so that motion is actually visible, '
      + 'and the token budget follows. The camera line is no longer asked of '
      + 'the caption model (none describes it reliably — that is measured): '
      + 'the 🎥 Camera pass’s own classifier writes it into the exported '
      + 'prompt, in words it can prove, labeled the way MiniMax H3’s own '
      + 'prompts label their blocks. Expect the pass to take longer per shot '
      + '— it is reading twice the frames and writing four times the words.',
    to: '/video-bank',
  },
{
    id: '2026-08-30-pass-info-dots',
    date: '2026-08-30',
    title: 'Every video pass button now explains itself',
    blurb:
      'A small ⓘ sits beside each pass of the video bank — Safe zone, Defects, '
      + 'AI check and the rest. It opens the guide’s own explanation right '
      + 'there, in a window, without leaving the page or losing your scroll: '
      + 'what the pass does, what it flags, and what to do with the result. '
      + 'Same text as the guide, so it can never drift out of date.',
    to: '/video-bank',
  },
{
    id: '2026-08-30-describe-shots-window',
    date: '2026-08-30',
    title: 'Describe shots now asks its questions before it runs',
    blurb:
      'The video bank’s 🗣 Describe button opens a launch window instead of '
      + 'firing blind. Pick the wording there — Standard, or Plain, which '
      + 'names explicit content instead of describing around it (measured on '
      + 'real adult footage: the prompt matters more than the model). Pick '
      + 'the model too: the proven 4B default, or Qwen3-VL 8B for better '
      + 'motion writing — each saying whether it is already on your machine '
      + 'or downloads first. And choose what it covers: only the shots still '
      + 'missing a caption, or a rewrite of the whole bank in the new wording '
      + '— captions you edited by hand are never touched unless you '
      + 'explicitly say so.',
    to: '/video-bank',
  },
{
    id: '2026-08-30-video-triage-exits',
    date: '2026-08-30',
    title: 'Un-decide video shots, and always reach the ✕',
    blurb:
      'Two dead ends gone from the video bank. Selected shots can now go '
      + '↩ back to triage — until now a mis-kept shot could only switch to the '
      + 'other verdict, never to “undecided”. And the shot player’s header '
      + 'stays pinned while you scroll, so the ✕ is always on screen — on a '
      + 'phone, where Esc does not exist, it was possible to scroll the only '
      + 'way out off the top of the page.',
    to: '/video-bank',
  },
{
    id: '2026-08-30-promote-window-knows-the-numbers',
    date: '2026-08-30',
    title: 'The video promote window now tells you what the numbers mean',
    blurb:
      'Building a first video training set means guessing a target, a size and '
      + 'a clip count — so the window stops making you guess. Each target '
      + 'carries a one-line hint (which one is proven locally, which trains in '
      + 'the cloud, which needs reference photos). The size menu says which '
      + 'sizes train exactly as cut and which of the model’s stated sizes get '
      + 'rescaled a little. And a line under the clip count tells you where '
      + 'your dataset sits: a dozen clips proves the pipeline, strong LoRAs '
      + 'are typically trained on 50–200. All of it measured, none of it '
      + 'blocking — every field stays yours to set.',
    to: '/video-bank',
  },
{
    id: '2026-08-30-h3-ref2va-training',
    date: '2026-08-30',
    title: 'Train MiniMax H3 Ref2V LoRAs — identity from reference images',
    blurb:
      'The Ref2V flavour of H3 generates from reference images of a subject, '
      + 'and now you can train for it: pick the MiniMax H3 Ref2V target when '
      + 'promoting clips, attach 1–4 reference images on the dataset card, and '
      + 'train locally or in the cloud with the same recipe H3 uses. The app '
      + 'refuses to launch without references on purpose — without them the '
      + 'trainer silently learns nothing of the identity, which is a paid run '
      + 'wasted. Local training needs an ai-toolkit from 2026-08-13 or newer; '
      + 'the app checks yours and says so instead of failing mid-run.',
    to: '/datasets',
  },
{
    id: '2026-08-30-video-trigger-word',
    date: '2026-08-30',
    title: 'Video datasets get a trigger word',
    blurb:
      'Set it once when promoting clips, and it is prepended to every clip’s '
      + 'caption file at export — exactly once, in one place. Your captions '
      + 'stay clean on screen, editing one never loses the trigger, and a '
      + 'caption that already starts with it is left alone: a doubled trigger '
      + 'measurably hurts prompt adherence, so the app makes doubling '
      + 'impossible. Optional — a style set legitimately has none.',
    to: '/video-bank',
  },
{
    id: '2026-08-30-h3-i2v-training',
    date: '2026-08-30',
    title: 'Train MiniMax H3 LoRAs for image-to-video',
    blurb:
      'If you animate still images, train the way you generate: one checkbox '
      + 'on the video training panels switches an H3 run to first-frame '
      + 'conditioning, so the LoRA learns under the same setup your i2v '
      + 'generations use. Works locally and in the cloud, on any ai-toolkit '
      + 'that trains H3 at all.',
    to: '/datasets',
  },
{
    id: '2026-08-30-h3-stills-training',
    date: '2026-08-30',
    title: 'Train an H3 video LoRA from your image datasets — no clips needed',
    blurb:
      'MiniMax H3 trains on still images too, and your image datasets already '
      + 'have everything that needs: curated pictures, edited captions, a '
      + 'trigger. One button in the Video training sets section turns an image '
      + 'dataset into a ready-to-train stills set — people have trained H3 '
      + 'character LoRAs this way on 12 GB cards. The promotion window also '
      + 'now counts clips coming from 48+ fps sources, which are often '
      + 'slow-motion footage that teaches floaty movement.',
    to: '/datasets',
  },
{
    id: '2026-08-30-video-steps-sized-to-the-dataset',
    date: '2026-08-30',
    title: 'Video training steps now start from your dataset, not a constant',
    blurb:
      'A 12-clip set and a 176-clip set used to get the same step count. The '
      + 'Steps field on a video dataset now starts from a suggestion sized to '
      + 'the clips it actually holds — about 28 steps per clip, taken from '
      + 'measured runs, never below the old default and never past what the '
      + 'measurements support. And there is exactly ONE such field per dataset: '
      + 'the training block asks for the settings once, and training on this PC '
      + 'or on a rented GPU is just the button you press — no run starts on a '
      + 'number you never saw. Type over it freely: what you enter is what trains.',
    to: '/datasets',
  },
{
    id: '2026-08-30-h3-trains-the-way-h3-is-trained',
    date: '2026-08-30',
    title: 'MiniMax H3 video LoRAs now train the way the model expects',
    blurb:
      'H3 ships guidance-distilled, and training a LoRA on it without accounting '
      + 'for that quietly degrades the result. ai-toolkit answered with a '
      + 'contrastive guidance loss and a small training adapter, and made the '
      + 'pair its default for H3 — video training here now uses both, wherever '
      + 'the ai-toolkit it is driving can actually run them: your installed copy '
      + 'is read for the capability, and a rented pod is judged on the image it '
      + 'boots, so an older setup quietly skips the recipe instead of failing. '
      + 'Alongside it, a clip now defaults to 39 frames instead of 107 — the '
      + 'length the trainer itself trains at, and about a third of the work per '
      + 'step — with every other length still on the menu, 22 included.',
    to: '/datasets',
  },
{
    id: '2026-08-19-video-burst-triage',
    date: '2026-08-19',
    title: 'Triage a video bank one keystroke per shot',
    blurb:
      'A two-hour rush becomes three hundred shots, and until now judging them meant three gestures each: click a tile, click ✓ or ✕, come back to the grid. The 🎬 Video bank has a new ⌨ Burst mode above the gallery. Turn it on and one tile carries a cursor — K keeps it, R rejects it, P puts it back to untriaged, S or → moves on without deciding, ← steps back. They are the same keys as the image bank\'s ▶ Review, so the reflex you already have works here. The cursor then jumps to the next shot you have NOT judged yet, which on a half-triaged bank is most of the speed; untick Auto-advance and it stays put so K then R corrects the same shot. It never wraps silently: when nothing untriaged is left ahead, the bar says how many are still behind you and Home goes back to the first. U undoes the last decision and moves the cursor onto that shot so you can see what it fixed, ten steps deep, always restoring what the shot actually was before — undoing a reject on a shot you had kept puts the keep back. The offer sits in the bar rather than in a toast, because at one keystroke a second a toast is replaced before it can be read. Your keys never wait for the network either: the tile flips at once and the decisions are sent behind you, one request at a time, with a run of identical verdicts going out as a single batch and a "saving N…" counter so a run that has ended is never mistaken for a run that is saved. Press ? for the full list, and nothing fires while you are typing in the search box or a threshold field.',
  },
{
    id: '2026-08-19-video-temporal-coherence',
    date: '2026-08-19',
    title: 'Find the shots that are secretly two shots',
    blurb:
      'Shot detection cuts on a change big enough to see. The ones it misses are the soft ones — a dissolve, a match cut, a new angle inside the same room — and each one leaves behind a “shot” that is really two scenes. It is the worst kind of training example, because it teaches the model a transition nobody asked for, and you cannot catch it by scrolling: the thumbnail is one of the two halves and looks perfectly fine. The 🎬 Video bank now checks every shot for this by itself, at the end of 🔎 Find scenes, and it costs nothing at all — no decoding, no model, no GPU, no button. It compares a shot’s first frame to its last using vectors that pass already cached, so a bank you embedded weeks ago gets its reading by clicking 🔎 Find scenes again. Each shot gains a scene coherence number (1.00 means its ends are the same picture), and 🎚 Quality cuts gains a Scene coherence floor that flags anything below it as “Cut inside the shot” — then ✂ Split here does the repair. Empty by default, and the Guide is blunt about why: this is a ranking, not a verdict. Measured against shots of the same length, a cut at 0.80 catches about a third of the double shots and flags about one honest shot in seven, so use it to choose what to look at first. Long takes score lower whether or not anything was cut, which the panel says out loud.',
    to: '/video-bank',
  },
{
    id: '2026-08-19-video-camera-motion',
    date: '2026-08-19',
    title: 'Sort your shots by what the camera did',
    blurb:
      'A video LoRA learns camera language along with everything else, and until now there was no way to see any of it: a bank of a thousand shots gave you no answer to “which of these are locked off” or “where are the handheld ones”. The 🎬 Video bank has a new 🎥 Camera pass. It tracks every frame of every shot and labels what the camera did — pan left, pan right, pan up, pan down, zoom in, zoom out, static shot, handheld shot — using the same words the video trainer itself uses, so a label here means the same thing there. Three more are ours: rolling, slideshow (a photograph panned across rather than filmed) and subject moves. The labels appear on each thumbnail and as a new 🎥 Camera row of filters above the gallery, which composes with the ⚑ flag chips — “shaky shots that also pan right” is one click each. Nothing is ever rejected: these are descriptions, not faults, because the wobble one person is filtering out is exactly what the next person is training on. If you do want to cut on it, 🎚 Quality cuts gains a Camera shake threshold, empty by default like the rest. The pass runs on the CPU at about fifteen times real time and needs only the video decode extra, which now installs OpenCV alongside PyAV — press Install on the video row in Setup if it shows a ✗. Honest limits, stated in the app too: a pivot and a slide look identical in a flat picture so both are called a pan, orbits are not detected at all, and a real pan across a wall or a horizon can read as a slideshow because it has no depth either.',
    to: '/video-bank',
  },
{
    id: '2026-08-19-video-ai-check',
    date: '2026-08-19',
    title: 'See which shots may have been generated rather than filmed',
    blurb:
      'A scrape in 2026 brings back generated video mixed in with the real thing, and it is invisible at thumbnail size — a clean, well-lit, well-framed synthetic clip passes every other check in the bank. It is worth finding: published curation work reports that even under a tenth of a corpus being synthetic measurably degrades what a model learns from it. The 🎬 Video bank has a new 🤖 AI check: it looks at two contiguous seconds of each shot and measures how erratically the motion changes, because real footage is full of small irregularities and generated footage tends to be smoother than the world. Shots that come out suspiciously smooth get a “May be AI-generated” chip, against a new cut in 🎚 Quality cuts that is empty by default and applied as you move it with nothing rescanned. Read as a hint and never as a verdict: on re-compressed material — which anything scraped is — the best blind-evaluated detectors in the field are right about three times in four, and this one has never been measured against 2025-and-later generators. Nothing is ever rejected or deleted on it. Runs on the CPU, so it can check a bank while a training owns your card, and it needs the same ✨ Score interpreter the look score already uses.',
    to: '/video-bank',
  },
{
    id: '2026-08-18-video-defect-sweep',
    date: '2026-08-18',
    title: 'Spot the footage that has already been through the mill',
    blurb:
      'A clip that was uploaded, re-encoded and re-uploaded three times carries damage a thumbnail cannot show you — and a LoRA learns it first, because it sits identically on every frame of every shot from that file. The 🎬 Video bank has a new 🩻 Defects pass: one sweep per source file finds frames that were simply delivered twice (what 24 fps material uploaded as 30 fps looks like), the macroblock grid showing through a hard squeeze, and edges that stay soft at FULL size. That last one is the important one — the sharpness floor measures a small analysis copy, where footage upscaled from 480p and the genuine 1080p are the same picture, so until now nothing in the app could tell them apart. Three new cuts in 🎚 Quality cuts, all empty by default, all applied as you move them with nothing rescanned. Each file card also shows how hard it was squeezed, in bits per pixel. Needs ffmpeg, which the video extra already installs.',
    to: '/video-bank',
  },
{
    id: '2026-08-18-video-safe-zone',
    date: '2026-08-18',
    title: 'See the black bars and the subtitles before your LoRA learns them',
    blurb:
      'A subtitle sits in the same rectangle of every frame of every clip from the same source, so it is among the first things a LoRA learns to draw — and at thumbnail size you cannot see it. Neither are the letterbox bars on a vertical video somebody padded into 16:9. The 🎬 Video bank has a new 🔳 Safe zone pass: it looks at three frames of each shot, measures the flat bands on all four sides, reads any text that HOLDS STILL across those frames (a passing shop sign is scene content and is left alone), and works out how much of the frame a crop would leave you. Three new cuts in the thresholds panel — letterbox share, burned-in text share, usable frame floor — all empty by default, all applied as you move them with nothing rescanned. Reading text needs one small CPU package from Setup; without it the pass still measures the bands and says so rather than pretending it found none.',
    to: '/video-bank',
  },
{
    id: '2026-08-18-scrape-videos-into-a-video-bank',
    date: '2026-08-18',
    title: 'Fill a video bank straight from the web',
    blurb:
      'The scraper could already SEE videos — RedGifs, Erome, Picazor, TikTok, X, Civitai all list them — and the picker threw every one of them away, so the only way to triage a clip you found online was to download it by hand, drop it in a folder and point a bank at that folder. 🎬 Video bank now has its own 🕸 Scrape the web panel: paste a link, pick the clips, and they land in a bank ready to be cut into shots. Nothing is judged on the way in — length, motion and sharpness stay for the bank’s own passes, exactly like the image side. Send them to a brand new bank, which gets a folder of its own, or to any bank you already have — including one pointed at your own footage, where the clips are simply added to that folder. The picker names the folder before you start.',
    to: '/video-bank',
  },
{
    id: '2026-08-18-video-look-score',
    date: '2026-08-18',
    title: 'Your shots now carry a look score — for free',
    blurb:
      'The video bank could tell you a shot was sharp, lit and moving; it could not tell you it was ugly. 🔎 Find scenes now also rates how each shot LOOKS, using the same LAION aesthetic model — and the same ~1–10 scale — the image Bank’s ✨ Score puts on a still. It costs nothing extra: the rating is read off the frame vectors that pass already caches, so no video is decoded twice and your GPU is never touched. Already embedded a bank? Click 🔎 Find scenes again and it rates the whole thing in seconds, without re-reading a single file. The new Aesthetic floor sits in 🎚 Quality cuts, empty by default — preview 4 against your own bank first; the published LAION cuts (4 casual, 4.75 strict) were set for filtering a web crawl, and real rushes sit well above them. A shot with no rating is never flagged.',
    to: '/video-bank',
  },
{
    id: '2026-08-08-shot-threshold-recut',
    date: '2026-08-08',
    title: 'Cut a rush again at another sensitivity, in seconds instead of minutes',
    blurb:
      'Shot detection never found cuts — it scored every frame, and the shot list was '
      + 'a threshold applied to that score. The threshold was 0.5, it comes from the '
      + 'detector paper where it is never justified, and disagreeing with it cost a '
      + 'full pass over the file. The scores are kept now, so the new 🎬 Find shots '
      + 'panel previews how many shots each threshold would give you on YOUR footage, '
      + 'and re-cuts a whole folder with no decoding and no GPU at all. Per bank, and '
      + 'per file for the folder that holds both a single take and a tight edit.',
  },
{
    id: '2026-08-08-single-shot-file',
    date: '2026-08-08',
    title: 'Tell the app a video has no cuts, and it will stop inventing them',
    blurb:
      'On footage that is one continuous take, the failure was never a missed cut — '
      + 'it was a file quietly chopped into six fragments that each trained on a third '
      + 'of a gesture. ▣ Single shot, on any file card, replaces its shots with one '
      + 'covering the whole file, and every bulk pass leaves that file alone '
      + 'afterwards. ↻ Re-detect this file is the way back.',
  },
{
    id: '2026-08-08-min-shot-seconds',
    date: '2026-08-08',
    title: 'The shortest-shot floor is a duration now, not a frame count',
    blurb:
      'The old floor was 5 frames, which is 0.2 s on a 25 fps rush and 0.08 s on a '
      + '60 fps one — nobody chose that, and it meant something different on every '
      + 'file in a mixed folder. It is 0.6 s by default now, converted through each '
      + 'file’s own rate. A short shot can also be glued onto its neighbour instead '
      + 'of dropped, which keeps the footage. If you had set the old key by hand, it '
      + 'still wins.',
  },
{
    id: '2026-08-07-video-install-checks-the-encoder',
    date: '2026-08-07',
    title: 'Installing the video extra no longer claims success when clips still cannot be cut',
    blurb:
      'The video extra delivers two things — reading your files, and encoding the clips you keep — and Setup only ever checked the first. So on a machine where the bundled ffmpeg never finished downloading (or an antivirus emptied it), the install said "✓ installed successfully" while the "Video bank — clip encoding" row stayed ✗ right underneath, behind the very same ↻ button: you reinstalled the half that already worked. That install now fails honestly and tells you which half is missing and what repairs it. The Setup row got stricter too: it runs ffmpeg instead of trusting that a file exists at the right path, so a truncated or quarantined binary is caught in Setup rather than in the middle of an export.',
  },
{
    id: '2026-08-06-setup-counts-the-video-pieces',
    date: '2026-08-06',
    title: 'Setup now counts the video pieces — and its repair menu can reach them',
    blurb:
      'The setup wizard could certify "12 of 12 capabilities ready" on a machine whose Video bank could not open a single file: the two video pieces were not counted, not listed in the Install-or-repair menu, and the wizard skipped its own install screen because everything it DID count was green. The summary now counts 14 — reading video files and shot detection included, each ✗ row clickable to where it installs — and 🎬 Video decoding and 🎞️ Shot detection sit in the Install or repair individually menu like every other component.',
  },
{
    id: '2026-08-06-shot-detection-installs-its-decoder',
    date: '2026-08-06',
    title: 'Shot detection no longer fails every file right after a clean install',
    blurb:
      'The 🎞️ Shot detection install put the model in place but not the decoder it reads files with — so the install reported success, the readiness badge turned green, and then every single file answered "failed shot detection". The install now carries PyAV into the same environment, the badge only turns green when the worker can actually open a file, and a contract test holds the three ends (worker, installer, probe) to the same list. If you hit this: Setup → 🎞️ Shot detection → ↻ Reinstall, then run Find shots again — your files were never the problem.',
  },
{
    id: '2026-08-06-video-extras-installable-from-setup',
    date: '2026-08-06',
    title: 'The video extras can now be installed where the app said they were',
    blurb:
      'The Video bank\'s banner told you "Install the video extra from Setup" — and Setup had no such button: both installs existed, but only for the API. Setup\'s optional-helpers step now carries the two missing cards: 🎬 Video decoding (PyAV + bundled ffmpeg into the app\'s own Python, no torch) and 🎞️ Shot detection (TransNetV2 into the scoring Python, CPU is fine). One click each, live progress, and the banner clears without a restart. A new test holds Setup to every install the video banners promise, so a pass can no longer point at a button that does not exist.',
  },
{
    id: '2026-08-06-video-watermark-flag',
    date: '2026-08-06',
    title: 'Spot the watermarked shots before they teach your LoRA a logo',
    blurb:
      'Rushes come off stock sites and other people\'s uploads, and a logo sitting in the same corner of every frame is the most consistent thing in your dataset — so it is the first thing a LoRA learns to draw. You cannot catch that by scrolling 90-pixel thumbnails. The new 🔖 Watermarks pass runs the same detector the image bank uses over each shot\'s sharpest frame and flags what it finds. Nothing is deleted: it is an amber flag you can filter on and act on. Needs the watermark detector from Setup; the cut sits in 🎚 Quality cuts at the measured 0.94, and a shot the pass has not judged is never called clean.',
  },
{
    id: '2026-08-06-video-duplicate-shots',
    date: '2026-08-06',
    title: 'Find the takes you already have, without watching them twice',
    blurb:
      'Ten near-identical takes of one gesture do not teach a model ten things — they teach it one thing ten times as loudly, and that is how a LoRA ends up unable to do anything else. The new ✂ Duplicates pass compares your shots to each other and groups the near-identical ones, keeping the sharpest of each pile unflagged so you know which one to keep. It costs no GPU and no waiting: it reuses the frame vectors 🔎 Find scenes already cached, so it is dot products over a file you already have. Flags only — nothing is rejected or deleted for you.',
  },
{
    id: '2026-08-06-video-flag-chips',
    date: '2026-08-06',
    title: 'Act on a quality flag instead of just reading it',
    blurb:
      'The amber flags in a video bank could be read one shot at a time and nothing more. There is now a row of chips above the gallery — "Barely moves (14)", "Same as another shot (31)" — and pressing one narrows the grid to exactly those shots, so you can select them and reject the lot in one gesture. The counts cover the shots currently loaded and the row says so when there are more to load.',
  },
{
    id: '2026-08-04-video-minimum-length',
    date: '2026-08-04',
    title: 'Half-second flash cuts stop cluttering your triage',
    blurb:
      'Shot detection deliberately keeps very short cuts — a real flash cut is a real shot, and a detector that hides them also hides genuine boundaries. The price was a grid full of half-second shots you scrolled past over and over, and that could never reach a dataset anyway. 🎚 Quality cuts now has a "Minimum length" field: type 1 second and every shorter shot wears an amber flag you can see and sort by. It is the one cut that works straight after detection — it reads the shot bounds, so you do not have to run the measuring pass first — and Preview tells you how many it would flag before you apply it. Nothing is deleted: it is a flag, like every other cut in that panel.',
  },
{
    id: '2026-08-04-video-train-local',
    date: '2026-08-04',
    title: 'Your video sets can now be trained here, without leaving the app',
    blurb:
      'A promoted video set now carries a ▶ Train this dataset button, and it hands the clips straight to the ai-toolkit already installed on your machine — no export, no copy, no config to write by hand. It shares the GPU with everything else honestly: a captioning pass or a ComfyUI render in flight refuses the launch instead of fighting over the card, and an image training already running blocks it exactly as another video run would. MiniMax H3 is wired in alongside Wan, with the quantisation, the noise schedule, the audio flags and the guidance its own trainer actually expects — a mismatch there does not crash, it just trains a slightly wrong model, which is why each value was read in the installed trainer rather than guessed. Two things it refuses to do quietly. H3 needs about 43 GB of weights: if they are not on your disk the button says so, names the repository and the size, and waits for a yes rather than turning into a silent overnight download. And a set re-promoted to a different target is refused rather than resumed, because the run folder still holds the previous model’s LoRA. Wan 2.2 is the one target a finished run has been through here; the card says plainly which of the others are wired but not yet proven.',
  },
{
    id: '2026-08-04-video-cloud-training',
    date: '2026-08-04',
    title: 'A promoted video dataset can now be trained on a rented GPU without leaving the app',
    blurb:
      'Open a video training set in your library and it has a Cloud training panel: pick a step count, click Train in the cloud, and watch the run — the GPU it landed on and what it costs per hour are on screen as soon as the pod has them. When it finishes, the checkpoints are listed by step and downloadable from the same panel. Wan 2.2 saves TWO files per step, the high-noise and low-noise experts, and either one alone is a LoRA nothing can load — so a step is offered as a step, with both of its files, never as a loose file list. A run that failed has a Retry, a run that finished has Train further, and continuing a Wan run sends both experts back to the new pod rather than resuming half of it. Before the job starts, the pod is asked to decode one of the clips you just uploaded with the same decoder the trainer uses; a pod that cannot read them fails in the first minute instead of billing you for hours of training on nothing.',
  },
{
    id: '2026-08-04-video-caption-wording',
    date: '2026-08-04',
    title: 'Captions can now speak plainly instead of describing around the subject',
    blurb:
      'Next to 🗣 Describe shots there is now a Caption wording choice. Standard is the wording that shipped and stays the default. Plain gives the model explicit permission to name what is actually on screen rather than reaching for vague stand-ins. This came out of a measurement, not a hunch: on real footage, four combinations were compared and the WORDING mattered more than the model did — the stock model asked plainly named things precisely and wrote the best action description of the four, while an uncensored model asked the old way still described around the subject. That matters because a caption that talks around its footage is a dataset defect you cannot see: the text reads perfectly well, the training set looks complete, and the LoRA learns the evasion. Every caption now records which wording produced it, so a bank captioned across a change is still one you can reason about. Pick it per run, or set it once in your config.',
  },
{
    id: '2026-08-04-video-caption-model-choice',
    date: '2026-08-04',
    title: 'Captions can now speak plainly — the model that writes them is yours to choose',
    blurb:
      'The 🗣 Describe shots pass had one model wired in. On a real corpus that turned out to be a dataset problem rather than a matter of taste: a captioner that describes what it sees in evasive terms produces captions that are about something slightly other than your footage — and a LoRA trained on those learns to look away too, with nothing in the output to reveal it. The captions read perfectly well; they are just not about the shot. So the checkpoint is now a setting, `video_caption.model`. Leave it empty and nothing changes: the same model as before, the same captions. Point it at any checkpoint of the same architecture and the pass uses that instead. Two things come with it. If the model is not on your machine yet, the pass SAYS so in its own progress line before it starts, because the first run downloads it and that should never be a silent twenty-minute wait. And every caption now records which model wrote it, so a bank captioned half before the change and half after is still a bank you can reason about.',
  },
{
    id: '2026-08-04-video-describe-shots',
    date: '2026-08-04',
    title: 'Your shots get described — so you can search for what HAPPENS, and so they train on words',
    blurb:
      'There is a new 🗣 Describe shots pass. It watches eight frames spread across each shot and writes what happens in it — “a woman turns and walks away”, not an inventory of objects — and that one line does two jobs. It becomes the clip’s .txt sidecar at promotion, which IS the prompt it trains on: until now every promoted clip shipped with an EMPTY prompt, and the trainer accepts that in silence. And it makes 🔎 Find scenes able to answer a question it structurally could not before. CLIP looks at frames, so it finds what a moment LOOKS like; an action is a fact about time and no single frame carries it. With captions the search reads both, and the panel says which halves are running so an empty result can be read correctly. Captions are drafts: open any shot and edit it, and a bulk re-run will not overwrite what you wrote. The promotion now also tells you how many clips are about to ship with no caption at all, before it encodes anything.',
  },
{
    id: '2026-08-04-video-source-cap-knob',
    date: '2026-08-04',
    title: 'The per-source cap you could already read about now has a knob',
    blurb:
      'The build dialog gains a “Max clips per source” field. The cap itself is not new — it has capped nothing so far because nothing could send it: the setting was implemented and reachable from neither the dialog nor the API. It matters because a 50-clip set that is three videos over-represented looks exactly like a diverse one on disk. Leave it empty for no cap. Each source keeps its EARLIEST clips, so promoting the same bank twice gives the same dataset, and a source with fewer clips than the cap keeps all of them — it trims dominance without punishing scarcity. And when a finished set turns out to lean on one file anyway, the result now says so with the real share instead of leaving you to notice.',
  },
{
    id: '2026-08-04-video-edge-trim',
    date: '2026-08-04',
    title: 'Trim the dissolve off both ends of every clip you export',
    blurb:
      'A shot boundary is where a cut just happened, so the first and last frames of a detected shot are disproportionately dissolves, fades and leftovers of a transition — and a dataset whose clips all open on half a dissolve teaches the model to open on half a dissolve. The build dialog now has a “Trim each end” field: a number of seconds taken off BOTH bounds of every clip. 0.25 is the common figure; the default is 0, so an existing recipe exports exactly what it exported before. What it will NOT do is hand you a short clip. Frame counts are a property of the target model’s VAE, and ffmpeg happily writes a 32-frame file and exits 0 when asked for 81 — so a clip that no longer supplies the count is dropped rather than exported short. The dialog tells you how many clips the trim will cost before you press the button, and counts them separately from clips that were never long enough: only the first kind is fixed by lowering the trim, and reporting them as one number is how a setting quietly halves a dataset while the material looks to blame.',
  },
{
    id: '2026-08-04-video-audio-metrics',
    date: '2026-08-04',
    title: 'Your shots are now listened to, not just looked at',
    blurb:
      'For LTX and MiniMax H3 the source’s audio is muxed into every clip you export — and until today nothing had ever listened to it. A shot whose track is a silent stretch, a dropout or a muted camera passed exactly like a shot with sound, because the file on disk is the right length, the right sample rate, and mute. A dataset of silent clips teaches the model to be silent. Measure now reports, per shot, how much of it is silence and its overall level in dBFS, with two new cuts to go with them — Silent share and Loudness floor — raising two different flags on purpose, since a quiet clip can be normalised and a silent one cannot be rescued. Three states are kept strictly apart, because collapsing any two of them makes the bank lie: a file with NO sound track is never flagged (Wan datasets are supposed to look like that), a track that is there and carries nothing is the actual defect, and shots measured before this shipped have no sound reading at all — an audio cut will never flag those, and Measure with re-measure is what fills them in.',
  },
{
    id: '2026-08-04-video-find-scenes',
    date: '2026-08-04',
    title: 'Type a word and find the scene, in a folder of rushes with no names',
    blurb:
      'A video bank is a haystack whose needles have no names: quality cuts tell you which shots are sharp and which move, and nothing tells you which one has the red car in it. There is now a 🔎 Find scenes box above the gallery. Run the pass once — it looks at a few frames of every shot — then type “a woman walking on a beach” and the gallery is replaced by the shots that look most like it, best first, instantly. Several frames per shot on purpose: a car that only drives into view in the last second would be invisible to a search that had looked at the opening frame, and you would get no hint it had been missed. So every shot contributes a frame near its start, its sharpest frame and one near its end — and every result tells you WHICH SECOND matched, with the player opening right there. Two things it says out loud rather than hiding. It is a ranking, not a filter: every shot scores something against every phrase, so the results always come back full and the line above the gallery tells you how strong the top and the tail really are, plus how many shots have not been looked at yet and could not be searched at all. And “without” does not work — ask for a street without cars and you get cars, because the model ignores the word rather than honouring it. Type “-cars” instead: that pushes them down the ranking, which is a promise the app can actually keep.',
  },
{
    id: '2026-08-04-video-clip-retouch',
    date: '2026-08-04',
    title: 'A badly cut shot is no longer a shot you have to throw away',
    blurb:
      'Until today, a shot the detector cut one second too early — or one holding a frozen tail — had exactly one available gesture: ✕ Reject, which threw away the eight good seconds to be rid of the bad one. Open any shot and there is now a ✂ Trim & split panel under the player. Nudge either bound by one second or by one frame (one frame OF YOUR FILE, at its own rate), snap a bound to wherever the playhead sits, split a shot in two at the playhead, or draw a shot the detector missed entirely — scrub anywhere in the rush and press ＋ New shot from here. Splitting keeps the decision you were making: split a kept shot and both halves stay kept, so you do not have to find them again among hundreds. One thing worth knowing before you trim: for image-to-video targets, the trainer conditions on the clip’s FIRST frame. Moving a start is therefore not trimming — it is choosing the exact image the model learns to animate from, and the panel says so where the buttons are. Re-cut shots lose their thumbnail and their quality scores on purpose: a thumbnail of a frame the shot no longer contains is not stale, it is wrong. Run Make thumbnails again when you are done cutting. And re-detecting a file no longer destroys the cuts you made by hand.',
  },
{
    id: '2026-08-04-video-quality-flags',
    date: '2026-08-04',
    title: 'Your video bank now measures every shot — and tells you which ones to look at',
    blurb:
      'One pass reads every frame of every shot and scores what quietly ruins a video dataset: shots that barely move, shots that are all blur, black moments, frozen stretches. Nothing is rejected for you — flagged shots get an amber mark in the grid, and the verdict stays yours. The cuts are yours too: there are deliberately NO default thresholds, because the same number that flags 2% of one bank flags 12% of another. Open 🎚 Quality cuts, set a value, and Preview shows exactly how many shots each cut would flag — per rule, before anything is applied. On a real 4.5-hour test bank the most valuable filter turned out to be the frozen-stretch one: 15% of shots carried a freeze the average could never see. Bonus: thumbnails now come from the SHARPEST measured frame instead of the middle guess.',
  },
{
    id: '2026-08-04-video-bank',
    date: '2026-08-04',
    title: 'Your folder of rushes is now a training set — and a .mp4 is no longer ignored in silence',
    blurb:
      'Drop a video into an image bank and until today it was skipped without a word: no row, no warning, nothing to click. Videos now get their own bank. Point it at a folder of rushes and it cuts every file at its shot boundaries, so you triage SHOTS instead of files — a two-hour rush becomes three hundred things you can judge in an afternoon. Click any shot to watch exactly that moment; the grid stays thumbnails, so a bank of hundreds of shots stays as light as a page of photos. Nothing is copied and nothing is re-encoded while you triage: a bank stores where each shot starts and ends, and only the ones you keep are ever encoded. When you build the set, the length menu offers only the frame counts your target model can actually ingest — 29 frames is legal for Wan and illegal for LTX, and no trainer tells you, they just quietly round it down. Two things are written next to the target you pick, because they are what costs a wasted week: whether a LoRA trainer for it is known to exist at all (exactly one of the four), and MiniMax H3’s licence, which grants no rights in the EU, the UK, South Korea or the USA — outputs included. And if a piece is missing, the app names which one: with no ffmpeg you can still scan, cut, watch and triage everything — only the final encode waits.',
  }
];

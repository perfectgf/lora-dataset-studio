/** FROZEN before the Ink redesign. Every entry is an interactive surface a
 * user can click or read as a tooltip, with HOW MANY TIMES it occurs across the
 * Bank tree. The contract test asserts each one still occurs at least that many
 * times, anywhere in the tree.
 *
 * The count is not decoration. Several screens share a label — "Cancel",
 * "Close", "Open →" — and a plain set cannot tell "one is left" from "there
 * were two". Deleting BankPage's Open button left the test green because the
 * video lane has one too. Counting is what closes that hole, while still
 * letting a button MOVE between files for free.
 *
 * Adding a surface is fine and needs no edit here. Regenerating this file to
 * make a red test go green defeats its purpose: a surface that legitimately
 * disappears is edited out here in the SAME commit that removes it, with the
 * reason in the commit message. */
export const BANK_SURFACES = [
  [
    "— what the chips above count as blurry, small, duplicate…",
    1
  ],
  [
    "·",
    2
  ],
  [
    "” down instead?",
    1
  ],
  [
    "{busy ? 'Saving…' : `Save$",
    1
  ],
  [
    "{busy ? 'Starting…' : `Launch$",
    1
  ],
  [
    "{c.cover_image_id != null && (",
    2
  ],
  [
    "{clip.status !== 'pending' && (",
    1
  ],
  [
    "{clip.thumb_state === 'ok' ? (",
    1
  ],
  [
    "←",
    1
  ],
  [
    "← Back to the last image",
    1
  ],
  [
    "← Banks",
    2
  ],
  [
    "← Prev",
    3
  ],
  [
    "↩ Not one person after all",
    1
  ],
  [
    "↩ Undo cleaning",
    1
  ],
  [
    "↺",
    3
  ],
  [
    "↺ Undecided",
    1
  ],
  [
    "↻",
    2
  ],
  [
    "↻ Check again",
    1
  ],
  [
    "↻ Rescan folder",
    1
  ],
  [
    "↻ Retry",
    1
  ],
  [
    "⇤ playhead",
    1
  ],
  [
    "+ Add zone",
    1
  ],
  [
    "＋ New shot from here",
    1
  ],
  [
    "⏭ Skip",
    1
  ],
  [
    "⏹ Stop",
    1
  ],
  [
    "⏹ Stop training",
    1
  ],
  [
    "▶",
    1
  ],
  [
    "▶ Train further",
    1
  ],
  [
    "☁ Train in the cloud",
    1
  ],
  [
    "⚖️ Balanced pick…",
    1
  ],
  [
    "⚖️ Pick a balanced set…",
    1
  ],
  [
    "✂ Cut a shot by hand",
    1
  ],
  [
    "✂ Split here",
    1
  ],
  [
    "✓ Keep",
    4
  ],
  [
    "✕",
    10
  ],
  [
    "✕ Close",
    1
  ],
  [
    "✕ Reject",
    4
  ],
  [
    "⤢",
    1
  ],
  [
    "⬆ Promote…",
    1
  ],
  [
    "🎞",
    1
  ],
  [
    "Pick diverse…",
    1
  ],
  [
    "🎬",
    1
  ],
  [
    "Similar to selected…",
    1
  ],
  [
    "👁 Preview what these cuts would flag",
    1
  ],
  [
    "👤 Single person here",
    1
  ],
  [
    "📊 Bank overview",
    1
  ],
  [
    "Coverage advice",
    1
  ],
  [
    "Move folder…",
    2
  ],
  [
    "🔍",
    1
  ],
  [
    "🔎 Find scenes",
    1
  ],
  [
    "Find by text…",
    1
  ],
  [
    "🕸",
    1
  ],
  [
    "Delete rejected from disk",
    1
  ],
  [
    "Auto-reject…",
    1
  ],
  [
    "Launch all…",
    1
  ],
  [
    "🚩 Watermarks",
    1
  ],
  [
    "Add a 5 s shot at the start of this file, then trim or split it in the player",
    1
  ],
  [
    "Already promoted into a dataset",
    1
  ],
  [
    "Also re-caption the captions I wrote by hand",
    1
  ],
  [
    "Apply cuts",
    1
  ],
  [
    "Back to the app default",
    1
  ],
  [
    "Back to the grid",
    1
  ],
  [
    "Bank that receives the images",
    1
  ],
  [
    "Build a video training set",
    1
  ],
  [
    "Bulk-reject the still-undecided images carrying the chosen quality flags",
    1
  ],
  [
    "Cancel",
    7
  ],
  [
    "Caption engine",
    1
  ],
  [
    "Caption length",
    1
  ],
  [
    "Caption vision model",
    1
  ],
  [
    "Caption vocabulary register",
    1
  ],
  [
    "Check folders before the person pass",
    1
  ],
  [
    "Check it",
    1
  ],
  [
    "Check the Pythons on this machine and point the SigLIP 2 index at one that reaches your GPU. They are read, never changed.",
    1
  ],
  [
    "Clear",
    3
  ],
  [
    "Clear search",
    2
  ],
  [
    "Clear the exclude filter",
    1
  ],
  [
    "Close",
    1
  ],
  [
    "Close (Esc)",
    1
  ],
  [
    "Close (the mask saves as you edit)",
    1
  ],
  [
    "Close review",
    1
  ],
  [
    "Close the mask editor",
    1
  ],
  [
    "Close the player",
    1
  ],
  [
    "Community idea by @antonp",
    1
  ],
  [
    "Crop it off",
    1
  ],
  [
    "Decide later (S) — stays undecided and is not shown again in this review",
    1
  ],
  [
    "Delete rejected from disk",
    1
  ],
  [
    "Delete zone",
    1
  ],
  [
    "Describe the scene to look for",
    1
  ],
  [
    "Discard changes",
    1
  ],
  [
    "Dismiss",
    2
  ],
  [
    "Dismiss the report",
    1
  ],
  [
    "Edit the watermark mask",
    1
  ],
  [
    "Exact / resized duplicate groups (perceptual hash) with their resolution panel",
    1
  ],
  [
    "fast",
    1
  ],
  [
    "Filter the grid to this resolution tier (megapixels = width×height)",
    1
  ],
  [
    "Filter the grid to this shot type (from the Framing pass)",
    1
  ],
  [
    "Filter thresholds",
    1
  ],
  [
    "Find them",
    1
  ],
  [
    "Draw the watermark zones on this image (M) — decides nothing. Works even when the scan found nothing: what you draw becomes the flag, and 🧽 Inpaint then repaints exactly that.",
    1
  ],
  [
    "Framing distribution of the set",
    1
  ],
  [
    "Go back to the environment the app set up for this pass",
    1
  ],
  [
    "Grid pages",
    1
  ],
  [
    "Hide coverage advice",
    1
  ],
  [
    "Hide images whose caption or file name contains these words",
    1
  ],
  [
    "Hides every image whose caption or file name contains one of these words (comma-separated). Matches anywhere in the text, so 'car' also hides 'scarf'. Images with no caption are never hidden.",
    1
  ],
  [
    "How captions name nude or sexual content. Explicit needs an uncensored (abliterated) Ollama vision model. Richer, more explicit captions also make the 🔍 search find more.",
    1
  ],
  [
    "How much the captioner writes. Concise aims for one short sentence, Detailed for several - a target the model follows loosely, not a hard cap. Standard leaves the prompt untouched. Longer captions give the search more to match on.",
    1
  ],
  [
    "Image tags",
    1
  ],
  [
    "In every group: keep the best (aesthetic score, then resolution/sharpness) member, reject the rest",
    1
  ],
  [
    "In every group: keep the first member (import order), reject the rest",
    1
  ],
  [
    "Keep best",
    1
  ],
  [
    "Keep first",
    1
  ],
  [
    "Keep the change and hide this",
    1
  ],
  [
    "Keep this image and move on (K)",
    1
  ],
  [
    "Kind of bank",
    1
  ],
  [
    "Klein",
    1
  ],
  [
    "LaMa",
    1
  ],
  [
    "LaMa: fast, non-generative repaint of small off-centre marks. Marks on the subject stay flagged.",
    1
  ],
  [
    "Last Launch-all run —",
    1
  ],
  [
    "Launch all",
    1
  ],
  [
    "Measure sharpness, noise, flatness, size and detail, hash every image and group the exact duplicates — CPU only. Opens the launch window.",
    1
  ],
  [
    "Measure the head angle of the images a previous build face-scanned without keeping it. Writes the angle and nothing else.",
    1
  ],
  [
    "Move this bank's folder",
    1
  ],
  [
    "Moved this folder to another disk? Point the bank at its new location.",
    1
  ],
  [
    "Moving this folder to another disk? Point the bank at its new location.",
    1
  ],
  [
    "Name of the new bank",
    1
  ],
  [
    "Next →",
    3
  ],
  [
    "no folder to prepare — the images land in a bank ready to triage",
    1
  ],
  [
    "Not decided here",
    1
  ],
  [
    "one file only ✕",
    1
  ],
  [
    "Open →",
    2
  ],
  [
    "Open this image full size, as the Bank shows it — your own file on disk is never modified",
    1
  ],
  [
    "Open this Bank's source folder in the system file explorer.",
    1
  ],
  [
    "Order the grid by anything the passes measured — resolution, size, aesthetic, NSFW, sharpness, noise, contrast, detail, bars, JPEG quality, face confidence. Images a pass never reached sink to the end. Remembered for this bank.",
    1
  ],
  [
    "passes ran",
    1
  ],
  [
    "Previous image (←) — navigation only, decides nothing",
    1
  ],
  [
    "Promote the selection",
    1
  ],
  [
    "Push “",
    1
  ],
  [
    "quality",
    1
  ],
  [
    "Re-caption",
    1
  ],
  [
    "Re-check every interpreter — use this right after installing a package",
    1
  ],
  [
    "Re-walk the folder and inventory anything new",
    1
  ],
  [
    "Reject all…",
    1
  ],
  [
    "Reject them",
    1
  ],
  [
    "Reject this image and move on (R) — reversible, nothing is deleted from disk",
    1
  ],
  [
    "Repaint what's left",
    1
  ],
  [
    "Reset",
    1
  ],
  [
    "Reset all to defaults",
    1
  ],
  [
    "Reset to detected",
    1
  ],
  [
    "Resolve ALL — keep best",
    1
  ],
  [
    "Resolve ALL — keep first",
    1
  ],
  [
    "Retry save",
    1
  ],
  [
    "Review from this image — full size, one at a time, with Keep/Reject/Skip",
    1
  ],
  [
    "Review the bank image by image",
    1
  ],
  [
    "Review the images of this filter one at a time, full size: ✓ Keep / ✕ Reject / ⏭ Skip (K/R/S) each move to the next. Optional random order.",
    1
  ],
  [
    "Rotate 90° left ([) — decides nothing. Your own file is never modified: the turn is stored and applied to what you see and to what gets promoted.",
    1
  ],
  [
    "Rotate 90° left (counter-clockwise). Your own files are never modified — the turn is stored and applied to what you see and to what gets promoted.",
    1
  ],
  [
    "Rotate 90° right (]) — decides nothing. Your own file is never modified: the turn is stored and applied to what you see and to what gets promoted.",
    1
  ],
  [
    "Rotate 90° right (clockwise). Your own files are never modified — the turn is stored and applied to what you see and to what gets promoted.",
    1
  ],
  [
    "Rotate left",
    2
  ],
  [
    "Rotate right",
    2
  ],
  [
    "Rotate this image 90 degrees left",
    1
  ],
  [
    "Rotate this image 90 degrees right",
    1
  ],
  [
    "s",
    1
  ],
  [
    "Save bounds",
    1
  ],
  [
    "Scrape the web into a bank",
    1
  ],
  [
    "Search the bank by caption or file name",
    1
  ],
  [
    "See what your kept set leans on and what's thin for a good LoRA — advice only, nothing is kept or rejected.",
    1
  ],
  [
    "Select all in filter",
    1
  ],
  [
    "Select page",
    1
  ],
  [
    "Semantic near-duplicates — same shot, different crop/compression — with their resolution panel",
    1
  ],
  [
    "Settings this pass reads",
    1
  ],
  [
    "Show a sample of the cleaned images so you can flip between the cleaned version and your original.",
    1
  ],
  [
    "show all ✕",
    1
  ],
  [
    "Sort the grid",
    1
  ],
  [
    "This run",
    1
  ],
  [
    "Throw away every cleaned version and flag those images again. Your original files were never modified, so nothing is lost.",
    1
  ],
  [
    "Throw away the hand-drawn zones and go back to the detected box",
    1
  ],
  [
    "Video tools status",
    1
  ],
  [
    "Walk every bank's source folder now and pick up the images added to it",
    1
  ],
  [
    "Walk what's left in random order instead of folder order — on a big dump that shows you a representative sample straight away instead of 200 near-identical shots. Nothing you have already seen comes back.",
    1
  ],
  [
    "What this run is about to caption",
    1
  ],
  [
    "Which engine writes this run's captions, without changing your Settings. Auto is a CHAIN, not a choice between two: JoyCaption drafts, then Ollama covers whatever it missed.",
    1
  ]
]

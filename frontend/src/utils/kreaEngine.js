/* Krea 2 Identity Edit — the pure, testable half of the engine's UI.
   PURE JS (no JSX) so `node --test` can import and exercise it directly, the
   same split as engineSelection.js.

   WHY THIS FILE EXISTS
   --------------------
   "Other people don't have my install." Krea needs FOUR weight files AND a
   community custom-node pack, and each of those can be missing for a different
   reason with a different fix. A card that just greys out with "not available"
   is an inert button; the user has no idea whether to install a node pack, drop
   a LoRA somewhere, or fix their ComfyUI URL. So the reason is computed here,
   from the capabilities payload, and it is unit-tested — one branch per real
   failure mode, in the order the user must fix them. */
import { brokenAssetReason, blockingInvalid } from './modelIntegrityWords.js';
import { comfyuiDownReason } from './comfyuiStatus.js';
import { pinnedModelGapReason } from './modelFileOptions.js';

/** Setup/capabilities asset keys -> the words a sentence uses. Mirrors
 *  krea_edit_helper.KREA_ASSETS; an unknown key falls back to itself rather
 *  than disappearing from the message. */
export const KREA_ASSET_LABELS = {
  krea_model: 'base model',
  krea_identity_lora: 'identity edit LoRA',
  krea_text_encoder: 'text encoder',
  krea_vae: 'VAE',
};

/** Stable-order words for the missing assets, so the sentence reads the same
 *  way every time regardless of how the server ordered the list. */
export function kreaMissingLabels(missing) {
  const set = new Set(Array.isArray(missing) ? missing : []);
  return Object.keys(KREA_ASSET_LABELS)
    .filter((k) => set.has(k))
    .map((k) => KREA_ASSET_LABELS[k]);
}


/** Why the Krea engine can't be picked, or null when it can.
 *  Ordered by what the user has to do FIRST: a disabled engine and an
 *  unreachable ComfyUI both make the asset lists meaningless, and the node pack
 *  comes before the weights because without it nothing can run even with every
 *  file in place. */
export function kreaUnavailableReason({
  enabledInSettings = true, comfyuiReachable = true,
  missingAssets = [], missingNodes = [], invalidAssets = [],
  nodePackInstalled = false, comfyui = null, pinGaps = [],
} = {}) {
  if (!enabledInSettings) return '⚠ Krea 2 Edit is disabled in Settings (engines)';
  // `comfyui` is the raw capabilities block. Passing it is what lets this say
  // "running but slow to list its nodes" instead of "configure ComfyUI" at a
  // ComfyUI that IS configured and IS running — the reported bug. Omitting it
  // keeps the old single sentence; see utils/comfyuiStatus.js.
  if (!comfyuiReachable) return comfyuiDownReason(comfyui || { reachable: false });
  // BEFORE the node pack and the weights: a pinned file that is absent makes
  // every list below lie by omission. `krea_missing` reports nothing (the
  // auto-detected files ARE there), so without this branch the engine would go
  // dark with the catch-all sentence and no way to guess which field to fix.
  const pinned = pinnedModelGapReason('Krea 2 Edit', pinGaps);
  if (pinned) return pinned;
  if (Array.isArray(missingNodes) && missingNodes.length) {
    // The pack is ON DISK but ComfyUI hasn't loaded it: ComfyUI registers custom
    // nodes at STARTUP only. Now that the app installs the pack itself, this is
    // the common state right after the install — and telling someone to install
    // what they just watched install is how a working feature reads as broken.
    if (nodePackInstalled) {
      return '⚠ The comfyui-krea2edit node pack is installed but ComfyUI has not loaded '
        + 'it yet — restart ComfyUI';
    }
    return '⚠ Install the comfyui-krea2edit node pack in ComfyUI, then restart it';
  }
  const words = kreaMissingLabels(missingAssets);
  // This line has now been wrong in BOTH directions, which is why the test next
  // to it pins the reason rather than the wording. It first said "see Setup"
  // while Setup covered Klein only and never said the word Krea -- a pointer to
  // a page about something else. It was then changed to name the Guide, correct
  // at the time: reading was all a user could do. Since the app installs these
  // files itself, the Guide is no longer the best answer -- Setup is, because it
  // now ACTS instead of explaining. The invariant is not "say Setup" or "say
  // Guide": it is that this message names a place that both exists and covers
  // Krea. Re-check that before rewording it again.
  if (words.length) {
    return `⚠ Krea ${words.join(' + ')} missing — Setup can download them for you`;
  }
  // Present but NOT weights: an interrupted, proxied or error-page download saves
  // HTML or a half file as .safetensors. The file exists, which is why "missing"
  // says nothing — and without this the only symptom was ComfyUI's raw
  // "Expecting value: line 1 column 1 (char 0)" at generate time.
  // Worded by the SHARED helper: Klein hit the same class of failure (a truncated
  // 9.5 GB UNET reported as "missing", zigzag4794 on Discord) and one corrupted
  // file must not be described two different ways two screens apart.
  const broken = blockingInvalid(invalidAssets);
  if (broken.length) return brokenAssetReason('Krea', broken[0], KREA_ASSET_LABELS);
  return null;
}

/** What the grounding dial currently means, in one short phrase. The number
 *  alone tells nobody anything; this is the whole point of exposing it. */
export function groundingDescription(px) {
  const n = Number(px);
  if (!Number.isFinite(n) || n <= 0) return 'default (512)';
  if (n < 512) return `${n}px · follows the prompt/card, looser likeness`;
  if (n === 512) return `${n}px · dataset-restaging balance (prompt/card adherence)`;
  if (n <= 768) return `${n}px · leans towards reference likeness`;
  if (n <= 1280) return `${n}px · favors reference likeness, may copy its pose`;
  return `${n}px · sticks to the reference, may ignore the card`;
}

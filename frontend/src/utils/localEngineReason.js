/* Why a LOCAL engine (Klein, Krea 2 Edit) can't run on THIS install — one
   answer, shared by every surface that offers those engines.
   PURE JS (no JSX) so `node --test` can exercise it directly.

   WHY THIS FILE EXISTS
   --------------------
   The generation panel had computed Klein's reason inline for a long time, and
   Krea's in utils/kreaEngine.js. That was fine while the generation panel was the
   only place you could pick a local engine. The ✦ Edit-reference modal now offers
   them too, and a second inline copy is how "⚠ Configure ComfyUI in Settings"
   ends up worded one way in one dialog and another way two clicks later — for the
   same missing file. So the reason moved HERE and both callers read it.

   Every branch answers one question: what is the ONE thing to do next? "Not
   available" is an inert button; the user cannot tell a missing weight from a
   stopped ComfyUI from a node pack that needs a restart. */
import { kreaUnavailableReason } from './kreaEngine.js';
import { comfyEnumUnavailableReason } from './comfyEnumSupport.js';
// The REQUIRED-weights vocabulary already exists, next to the Setup step that
// downloads them. Re-listing it here is how the picker and Setup end up naming
// different files for one gap; the recommended LoRA stays excluded there for the
// right reason (optional + auto-fetched, so naming it sends the user nowhere).
import { kleinMissingLabels, KLEIN_ASSET_LABELS, KLEIN_REQUIRED_ASSETS } from './kleinAssets.js';
import { brokenAssetReason, blockingInvalid } from './modelIntegrityWords.js';
import { pinnedModelGapReason } from './modelFileOptions.js';
import { comfyuiDownReason } from './comfyuiStatus.js';

/** Why Klein can't be picked, or null when it can. Ordered by what has to be
 *  fixed FIRST — a disabled engine and an unreachable ComfyUI both make the asset
 *  list meaningless.
 *
 *  The enum gap comes BEFORE the weights: a reachable ComfyUI with every file in
 *  place can still refuse a widget VALUE the graph pins (the `beta57` scheduler
 *  took Klein out for everyone without the RES4LYF pack while every other check
 *  went green). Naming the weights there sends the user to re-check a step that
 *  was already correct.
 *
 *  `invalidAssets` is the branch this function was MISSING, and its absence is
 *  what reported a 9.5 GB file on disk as "Klein model missing" (zigzag4794,
 *  Discord): a truncated download is present, so no weight is named, and the
 *  sentence fell through to the catch-all. Krea has had this branch since its
 *  own licence-gate incident — Klein now says the same thing, from the same
 *  helper. */
export function kleinUnavailableReason({
  enabledInSettings = true, comfyuiReachable = true,
  missingAssets = [], unsupportedEnums = [], invalidAssets = [], comfyui = null,
  pinGaps = [],
} = {}) {
  if (!enabledInSettings) return '⚠ Klein is disabled in Settings (engines)';
  // Same two-causes split as Krea's: a ComfyUI that is up and slow to enumerate
  // itself must not be reported as one to go and configure. See
  // utils/comfyuiStatus.js.
  if (!comfyuiReachable) return comfyuiDownReason(comfyui || { reachable: false });
  // Before the enum gap and the weights, for the same reason as Krea's: the
  // asset scan answers "everything is here" when a pin is broken, because
  // auto-detection DID find files — just not the ones that were asked for.
  const pinned = pinnedModelGapReason('Klein', pinGaps);
  if (pinned) return pinned;
  const enumHint = comfyEnumUnavailableReason(unsupportedEnums);
  if (enumHint) return enumHint;
  const words = kleinMissingLabels(missingAssets);
  // Name the exact missing weight(s) rather than always blaming the UNET: the old
  // text sent users to models/unet/klein/ even when the real gap was the text
  // encoder or the VAE.
  if (words.length) return `⚠ Klein ${words.join(' + ')} missing — download it in the Setup step`;
  // Present but NOT loadable. Only the REQUIRED trio gates the engine, so only
  // those are worth a sentence here.
  const broken = blockingInvalid(invalidAssets, KLEIN_REQUIRED_ASSETS);
  if (broken.length) return brokenAssetReason('Klein', broken[0], KLEIN_ASSET_LABELS);
  // Last resort. It used to claim "Klein model missing", which is a guess dressed
  // as a finding — and the wrong guess in every case that reaches this line, since
  // every nameable gap is handled above. Say what is actually known instead.
  return '⚠ Klein is not ready, and this install reports no missing or broken file — '
    + 'open Setup ▸ ComfyUI and use Copy diagnostic to see what it found';
}

/** Reason for EITHER local engine, read straight off the capabilities payload.
 *  Returns null for an engine that is available, and for anything that is not a
 *  local engine (an API engine's failures are only knowable from the provider's
 *  answer — nothing here can predict them, and pretending to would be worse).
 *
 *  `enabledEngines` is OPTIONAL: pass the Settings list to honour it, omit it
 *  (null) to skip that gate. The ✦ Edit modal omits it — it does not fetch
 *  /api/settings — which is a known gap, not an opinion: see the follow-up note
 *  in the test file. */
export function localEngineUnavailableReason(engine, caps, enabledEngines = null) {
  const engines = caps?.engines || {};
  const comfy = caps?.comfyui || {};
  const enabled = (e) => (Array.isArray(enabledEngines) ? enabledEngines.includes(e) : true);
  if (engine === 'klein') {
    if (engines.klein) return null;
    return kleinUnavailableReason({
      enabledInSettings: enabled('klein'),
      comfyuiReachable: !!comfy.reachable,
      comfyui: comfy,
      missingAssets: comfy.klein_missing,
      unsupportedEnums: comfy.klein_unsupported_enums,
      invalidAssets: comfy.klein_invalid,
      pinGaps: comfy.klein_pin_gaps,
    });
  }
  if (engine === 'krea') {
    if (engines.krea) return null;
    return kreaUnavailableReason({
      enabledInSettings: enabled('krea'),
      comfyuiReachable: !!comfy.reachable,
      comfyui: comfy,
      missingAssets: comfy.krea_missing,
      missingNodes: comfy.krea_nodes_missing,
      invalidAssets: comfy.krea_invalid,
      nodePackInstalled: comfy.krea_nodes_installed,
      pinGaps: comfy.krea_pin_gaps,
      unsupportedEnums: comfy.krea_unsupported_enums,
    });
  }
  return null;
}

/** THE readiness verdict for a local engine — one question, one answer, for every
 *  screen that asks it.
 *
 *  WHY THIS EXISTS (and why Setup must not compute its own)
 *  --------------------------------------------------------
 *  `engines.klein` / `engines.krea` are computed by the BACKEND, from five
 *  conditions for Klein (ComfyUI reachable, no unsupported widget value, the UNET
 *  + text encoder + VAE all resolvable, no blocking-invalid required weight) and
 *  four for Krea. The Setup wizard used to re-derive its own version of that from
 *  the raw ingredients, and the install menu judged each file on presence alone —
 *  so a truncated 9.5 GB UNET read "✓ Installed" on one screen while the
 *  generation page refused the engine on the next. Two screens, both sincere,
 *  contradicting each other, with the user sent to fix the one thing that was
 *  fine.
 *
 *  So: the backend verdict is the truth, this reads it, and everything else reads
 *  this. Nothing here recomputes readiness — `reason` only EXPLAINS a verdict it
 *  was given.
 *
 *  Returns:
 *    ready    — the engine can be used right now (the backend said so);
 *    verified — whether the checks that need a live ComfyUI could actually RUN.
 *               False means "not checked", which is not the same as "checked and
 *               fine" and must never render as a ✓. The unsupported-value probe
 *               fails OPEN by design (an unreachable ComfyUI reports no gap rather
 *               than inventing one); this flag is how a screen says so out loud
 *               instead of inheriting a silent all-clear.
 *    reason   — null when ready, else the ONE shared sentence naming the real gap. */
export function localEngineReadiness(engine, caps, enabledEngines = null) {
  const c = caps || {};
  const comfy = c.comfyui || {};
  const ready = (c.engines || {})[engine] === true;
  return {
    engine,
    ready,
    verified: !!comfy.reachable,
    reason: ready ? null : localEngineUnavailableReason(engine, c, enabledEngines),
  };
}

/** Whether this install has a ComfyUI at all — the line between "a gap you can
 *  fix from here" and "a product you haven't got". A configured-but-unreachable
 *  ComfyUI counts as HAVING one: it is running-or-not, which is fixable, and
 *  hiding the local engines while the user restarts it would look like the app
 *  lost a feature. Only a wholly unconfigured, unreachable install hides them. */
export function hasComfyui(caps) {
  const comfy = caps?.comfyui || {};
  return !!(comfy.dir_configured || comfy.reachable);
}

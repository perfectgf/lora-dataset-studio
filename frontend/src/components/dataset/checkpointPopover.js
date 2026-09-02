/* Pure brain of the checkpoint ACTIONS popover — the one that opens on a
   checkpoint pill and offers ⬇ Download, ▶ Continue from here, 📦 Deploy,
   ⏏ Undeploy, 🗑 Delete and ⓘ Details.

   JSX-free on purpose. `node --test` does not parse JSX, and these are exactly
   the decisions that must not be eyeballed in a popover: WHICH actions a given
   checkpoint may offer, and — when it may not — WHY. The rule this file exists
   to enforce is the app's own: an action is either live, or shown disabled with
   its real reason, or absent. Never a button that fails in silence.

   The geometry lives here too, for the same reason: a popover that leaves its
   frame is a bug you can only see at one window size, and the app is consulted
   on a 400-px phone where it is the normal case rather than the edge one. Two
   placements, because the two surfaces host it in two different spaces:

     • the graph embedded in a run card draws it INSIDE its <svg>, in world
       units, so it is clamped against the graph's own box;
     • the LoRA Canvas floats it above a zoomed, panned board, so it is placed
       in SCREEN pixels at a constant size — a popover that scaled with the
       board would be unreadable at the zoom levels the board is useful at. */

import { canContinueFromCheckpoint } from './lineageContinue.js';
import {
  checkpointDeleteTarget, checkpointDeployed, checkpointUndeployAction,
  lineageImportPayload,
} from './lineagePreview.js';

/* The popover's fixed box. Both surfaces size it from here, so the geometry
   below is clamping the thing that is actually drawn. 210 px wide fits a 400-px
   screen with margins to spare; the height is the tallest the column gets (every
   row present at once). */
export const POPOVER_W = 210;
// One row taller since 📤 Civitai joined the column (every row present at once).
export const POPOVER_H = 264;

/* WHY a checkpoint cannot be deployed into ComfyUI, or null when it can. The
   payload builder answers only yes/no (it returns a body or null), and "no" has
   two very different causes the user can act on differently. */
export function deployRefusal(node, pill) {
  if (!pill || !pill.filename) return 'This checkpoint has no file on this machine to deploy';
  if (pill.present === false) return 'This save is no longer on disk — nothing to deploy';
  if (node?.source === 'cloud' && node?.run_id == null) {
    return 'This cloud run is not linked on this machine, so its file cannot be resolved';
  }
  return null;
}

/* WHY a save cannot be downloaded, or null when it can. */
export function downloadRefusal(pill) {
  if (!pill) return null;
  if (pill.present === false) return 'This save is no longer on this machine';
  if (!pill.download_url) return 'Download unavailable for this save';
  return null;
}

/**
 * Everything the popover renders, decided in one place.
 *
 * `pill` may be null: clicking a run CARD opens the same popover with only the
 * run-level rows (ⓘ Details). That is deliberate — one popover, one mental
 * model, and the detail drawer stops opening on its own.
 *
 * Returns null for no node at all. Otherwise:
 *   { step, final, isRun, download: {url}|{reason}|null,
 *     continue: {ok:true}|{reason}|null, deployed, deploy: {payload,folder}|{reason}|null,
 *     undeploy: action|null, del: target|null }
 */
export function checkpointActionModel(node, pill, {
  continueSource = 'cloud', hasContinueHandler = false, continueReason = null,
  folderLabel = null,
} = {}) {
  if (!node) return null;
  if (!pill) {
    return { step: null, final: false, isRun: true, download: null, continue: null,
      deployed: false, deploy: null, undeploy: null, del: null };
  }
  const deployed = checkpointDeployed(pill);
  const dlWhy = downloadRefusal(pill);
  const target = checkpointDeleteTarget(node, pill);
  const deployWhy = deployRefusal(node, pill);
  const canContinue = canContinueFromCheckpoint(node, pill, {
    continueSource, hasHandler: hasContinueHandler,
  });
  return {
    step: pill.step ?? null,
    final: !!pill.final,
    isRun: false,
    download: dlWhy ? { reason: dlWhy } : { url: pill.download_url },
    // A host with no continue flow may still explain where the gesture lives
    // (`continueReason`); with neither handler nor reason the row is absent,
    // which is the other half of the rule — never a dead button.
    continue: canContinue ? { ok: true } : (continueReason ? { reason: continueReason } : null),
    deployed,
    deploy: deployed ? null
      : (deployWhy || !lineageImportPayload(node, pill))
        ? { reason: deployWhy || 'This checkpoint cannot be resolved to a file to deploy' }
        : { payload: lineageImportPayload(node, pill), folder: folderLabel },
    undeploy: checkpointUndeployAction(node, pill),
    // The 🗑 retreat row is reserved for the one destructive action — the run's
    // own save. A deployed pill's ComfyUI copy is handled above by ⏏ Undeploy.
    del: target && target.kind === 'save' ? target : null,
  };
}

/**
 * WORLD placement, for the popover drawn inside a lineage <svg>.
 * Flips above the pill when there is no room below, and is clamped horizontally,
 * so the scrolling panel never clips it.
 */
export function checkpointPopoverPlacement(pill, world, {
  width = POPOVER_W, height = POPOVER_H, gap = 4,
} = {}) {
  const px = Number(pill?.x) || 0;
  const py = Number(pill?.y) || 0;
  const ph = Number(pill?.h) || 0;
  const ww = Math.max(0, Number(world?.width) || 0);
  const wh = Math.max(0, Number(world?.height) || 0);
  const below = py + ph + gap;
  const y = below + height > wh ? Math.max(0, py - height - gap) : below;
  return { x: Math.max(0, Math.min(px, Math.max(0, ww - width))), y };
}

/**
 * SCREEN placement, for the popover floated above the canvas board.
 *
 * `anchor` is the point that was clicked, in viewport pixels. The popover is
 * centred under it, then pushed back inside the window — and, when it would
 * overflow the bottom, flipped above the anchor. It also NARROWS on a screen
 * too small to hold its natural width, because the alternative is a popover
 * that forces the page to scroll sideways, which the app forbids.
 */
export function clampPopoverToViewport(anchor, viewport, {
  width = POPOVER_W, height = POPOVER_H, margin = 8, gap = 12,
} = {}) {
  const vw = Math.max(0, Number(viewport?.width) || 0);
  const vh = Math.max(0, Number(viewport?.height) || 0);
  const ax = Number(anchor?.x) || 0;
  const ay = Number(anchor?.y) || 0;
  const w = Math.max(120, Math.min(width, Math.max(120, vw - margin * 2)));
  const left = Math.max(margin, Math.min(ax - w / 2, Math.max(margin, vw - w - margin)));
  const maxTop = Math.max(margin, vh - height - margin);
  let top = ay + gap;
  if (top > maxTop) top = ay - gap - height;          // no room below → flip above
  top = Math.max(margin, Math.min(top, maxTop));
  return { left, top, width: w };
}

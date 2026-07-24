/* Reference-photo editing: which engines can edit, what the default pick is, and
   the small guards the modal relies on. PURE JS (no JSX) so node --test can
   import and exercise it directly — same split as engineSelection.js.

   The ✦ Edit modal sends the reference + a prompt to ChatGPT or Nano Banana and
   gets an edited candidate back. Klein is deliberately out of scope for editing
   (this wave), so the edit engine set is NOT the generation engine set. */
import { primaryEngine, readEngines, API_ENGINES } from './engineSelection.js';

/** Engines that can edit the reference. Klein excluded on purpose — the same two
 *  API engines the /ref/edit route accepts. Order = toggle order in the modal. */
export const EDIT_ENGINES = ['chatgpt', 'nanobanana'];

/** The engine the modal opens on: the workspace's PRIMARY generation engine when
 *  it can also edit, else ChatGPT. So a profile generating with Nano Banana edits
 *  with Nano Banana; one generating with Klein (can't edit) falls back to ChatGPT
 *  rather than a dead selection. */
export function defaultEditEngine(storage) {
  const primary = primaryEngine(readEngines(storage));
  return EDIT_ENGINES.includes(primary) ? primary : 'chatgpt';
}

/** Why the "Generate edit" button is disabled, or null when it can run. An empty
 *  prompt is the only hard block: the edit is free-form, but it needs SOMETHING. */
export function editBlockedReason(prompt, engine) {
  if (!EDIT_ENGINES.includes(engine)) return 'Pick ChatGPT or Nano Banana';
  if (!prompt || !prompt.trim()) return 'Describe the edit first';
  return null;
}

/** The modal's phase, DERIVED from the server's `reference_edit` payload object
 *  (not local state) so it restores correctly after a tab sleep or reload:
 *  'idle' (no pending edit / form), 'running', 'ready' (Before/After), 'failed'. */
export function editPhase(referenceEdit) {
  const s = referenceEdit?.status;
  return (s === 'running' || s === 'ready' || s === 'failed') ? s : 'idle';
}

/** Advisory shown when a generation batch is live. A Keep is provably safe (the
 *  batch snapshotted the reference at launch), so this INFORMS, it does not block:
 *  the point is that editing changes only FUTURE batches. Returns null when no
 *  batch is running. `activity` is the live dataset-activity object (or null). */
export function batchLiveNote(activity) {
  return activity && activity.kind === 'generate'
    ? "A batch is running. Editing the reference won't change variations already "
      + 'generated or still in flight — only future batches use the edited photo.'
    : null;
}

// Re-exported so the modal imports one module for all edit constants.
export { API_ENGINES };

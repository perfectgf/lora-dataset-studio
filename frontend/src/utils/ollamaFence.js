/* What a surface says and does when the local Ollama fence refuses it.
 *
 * The fence itself is right: LDS must not unload a model another tool loaded,
 * because on a machine that also runs an image generator, that model is
 * somebody's real work. What was wrong is that the refusal ENDED there. It
 * named the remedy in prose ("unload it first"), then left the user to go find
 * an Ollama prompt, do it by hand, come back, and click the button again.
 *
 * Two things change that, and both live here:
 *
 *   1. Consent, not silence. The user can say "unload it and continue" in one
 *      click. That click is the whole difference — the default stays refuse.
 *   2. Patience. Ollama unloads an idle model by itself after a few minutes,
 *      so most of these blocks end on their own. Instead of failing, the
 *      surface waits, watches, and resumes the action it was asked to do. The
 *      commonest case then costs zero clicks.
 *
 * Pure functions here (a `node --test` file imports them directly); React in
 * useOllamaFence.js and OllamaFenceNotice.jsx.
 */

/** The structured code the backend puts on a fence 409 (`_map_error`). */
export const OLLAMA_FENCE_CODE = 'ollama_fence_blocked';

/** What a ✨ writer says when an answer comes back for a click the user has
 *  moved on from — a newer click, a changed frame, mode or length, "stop
 *  waiting". The request was in flight and could not be stopped; the answer
 *  is set aside and the field left as it is. One sentence, on the surfaces
 *  that write into a field the user can see (the Video Test Studio's ✨ Auto
 *  and ✨ Enrich, the Studio's ✨ Enhance). The two 🔎 Describe surfaces stay
 *  quiet: a set-aside reading changes nothing on screen, and the newer
 *  request answers for itself. */
export const SUPERSEDED_ANSWER_NOTICE =
  'The ✨ answer to an earlier click arrived after you moved on — set aside, nothing was written.';

/** Whether an answer is still wanted. `run` is the handle the guard hands
 *  its action (`useOllamaFence`): `current()` read at the moment of the
 *  write, never captured. An action run without one (the hook before it)
 *  keeps every answer. When the click has been moved on from, `tell` says
 *  so — only while there is still a surface to say it on. */
export function keepAnswer(run, tell) {
  if (!run?.current || run.current()) return true;
  if (tell && run.mounted()) tell();
  return false;
}

/** How long the surface keeps waiting before it gives the decision back.
 *  Past Ollama's own ~5 min idle unload with room to spare: if the model is
 *  still there after ten minutes, something is actively using it and waiting
 *  longer is just a spinner pretending to be progress. */
export const AUTO_RETRY_CAP_MS = 10 * 60 * 1000;

/**
 * Is this failure the fence, as opposed to any other reason an Ollama call can
 * fail? Only the code counts. Matching on the message text would break the
 * moment the sentence is reworded or translated, and would just as happily
 * match a genuine error that quoted it.
 */
export function isOllamaFenceError(err) {
  return err?.body?.code === OLLAMA_FENCE_CODE || err?.code === OLLAMA_FENCE_CODE;
}

/**
 * Delay before the next /api/ps probe, given how long we have been waiting.
 *
 * Backs off on purpose. The first half-minute is where a model being unloaded
 * elsewhere shows up, so it is worth checking often; after that this is a
 * background vigil that may run for ten minutes and has no business polling a
 * local daemon every two seconds for all of it.
 */
export function nextPollDelay(elapsedMs) {
  const elapsed = Number.isFinite(elapsedMs) ? Math.max(0, elapsedMs) : 0;
  if (elapsed < 30_000) return 2_000;
  if (elapsed < 120_000) return 5_000;
  return 15_000;
}

/** "12s" / "1m 05s" — the counter in the waiting line. */
export function waitedLabel(elapsedMs) {
  const total = Math.max(0, Math.round((Number.isFinite(elapsedMs) ? elapsedMs : 0) / 1000));
  if (total < 60) return `${total}s`;
  const minutes = Math.floor(total / 60);
  const seconds = total % 60;
  return `${minutes}m ${String(seconds).padStart(2, '0')}s`;
}

/** How to name the models in the way. Ollama tags are long; two is the most
 *  that reads as a sentence rather than a dump. */
export function blockingModelsLabel(models) {
  const names = (Array.isArray(models) ? models : []).filter((m) => typeof m === 'string' && m.trim());
  if (!names.length) return null;
  if (names.length === 1) return names[0];
  if (names.length === 2) return `${names[0]} and ${names[1]}`;
  return `${names[0]}, ${names[1]} and ${names.length - 2} more`;
}

/**
 * Everything the notice draws, from the guard's state. Returns null when there
 * is nothing to say — the surface renders nothing at all rather than an empty
 * box, which is the normal case.
 *
 * `phase` is one of:
 *   waiting  — the block is live and we are watching for it to clear
 *   retrying — it cleared (or was unloaded) and the original action is re-running
 *   unloading— the user clicked, the eviction is in flight
 *   gave-up  — the patience cap ran out; the decision goes back to the user
 */
export function fenceNoticeModel(state) {
  if (!state || !state.phase) return null;
  const models = blockingModelsLabel(state.models);
  // The server the user actually runs. Naming Ollama to someone on LM Studio sends
  // them to look at a daemon that is not holding anything.
  const server = state.provider === 'lmstudio' ? 'LM Studio' : 'Ollama';
  const held = models ? `Another tool is using ${models} in ${server}.` : state.message
    || `Another tool is using a model in ${server}.`;

  if (state.phase === 'unloading') {
    return { tone: 'busy', headline: 'Unloading the other model…', detail: held,
             canUnload: false, canCancel: false, busy: true };
  }
  if (state.phase === 'retrying') {
    return { tone: 'busy', headline: 'The model is free — picking up where you left off…',
             detail: held, canUnload: false, canCancel: false, busy: true };
  }
  if (state.phase === 'gave-up') {
    return {
      tone: 'idle',
      headline: 'Still in use after 10 minutes.',
      detail: `${held} LDS stopped waiting — unload it below, or free it in the other tool and try again.`,
      canUnload: true, canCancel: false, busy: false,
      unloadLabel: 'Unload it and continue',
    };
  }
  return {
    tone: 'waiting',
    headline: `Waiting for the model to be released… (${waitedLabel(state.elapsedMs)})`,
    detail: `${held} LDS will not unload it without your say-so — it will start on its own the moment it is free.`,
    canUnload: true, canCancel: true, busy: false,
    unloadLabel: 'Unload it and continue',
  };
}

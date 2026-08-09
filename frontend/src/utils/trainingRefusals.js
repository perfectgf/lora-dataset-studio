/* Confirmable launch refusals — the ONE definition shared by every surface that
   starts or resumes a training.

   The server prefixes a bypassable refusal with a marker; the window.confirm IS
   the user's answer and the retry carries the matching force flag. Several can
   fire in sequence (uncaptioned first, then mismatch), so call sites loop until
   the launch goes through, the user declines, or a non-confirmable error comes
   back.

   Lives in utils/ (JSX-free) because two mounts now need it: the dataset
   TrainingPanel and the Runs hub, whose ▶ Continue can resume a run on THIS
   machine — where a resume re-exports the current dataset and hits exactly these
   guards. Duplicating the marker list would let the two drift apart. */

export const CONFIRMABLE_REFUSALS = [
  ['MISMATCH_CAPTION: ', 'allow_caption_mismatch'],
  ['UNCAPTIONED: ', 'allow_uncaptioned'],
  ['CAPTION_QUALITY: ', 'allow_caption_quality'],
  // Custom-weights arch sniff couldn't positively verify the file → the
  // window.confirm IS the answer, retry carries allow_unverified_weights.
  ['CUSTOM_WEIGHTS_UNVERIFIED: ', 'allow_unverified_weights'],
  // Full-model cloud run whose delivery would not fit in the account's PRIVATE
  // Hugging Face storage. Confirmable ON PURPOSE: Hugging Face publishes no
  // quota endpoint, so the ceiling the server compared against is an estimate —
  // a wrong guess must never lock a user out of their own paid GPU.
  ['HF_STORAGE_FULL: ', 'allow_hf_storage'],
  // The same question about THIS machine's disk: a full model is delivered here
  // first, and the size it will take is an estimate too (a checkpoint that does
  // not exist yet). Confirmable for the same reason — and said before the GPU
  // is rented rather than discovered when the drive fills up at the very end.
  ['LOCAL_DISK_FULL: ', 'allow_local_disk'],
  // A second cloud run of the same family on the SAME dataset. Confirmable on
  // purpose: it rents a second pod (billed separately), so a double-click must
  // stay free — but a deliberate settings A/B must not wait hours behind the
  // first run.
  ['PARALLEL_RUN: ', 'allow_parallel_run'],
];

/* The readiness floor (NOT_READY:) is confirmable too, but the dataset panel
   answers it with its OWN "Continue anyway" checkbox in the preparation card, so
   it stays out of the shared list above — a surface that already asks must not
   ask twice.

   Lanes with no preparation panel have nowhere else to ask. The Runs hub's
   ↻ Retry is one: a run legitimately started with too few images (the checkbox
   ticked at Start) hits the floor again on retry, and with the floor absent from
   every confirm list the button could only ever dead-end. */
export const READINESS_REFUSAL = ['NOT_READY: ', 'allow_not_ready'];
export const RETRY_CONFIRMABLE_REFUSALS = [...CONFIRMABLE_REFUSALS, READINESS_REFUSAL];

/* The pure half: which refusal is this, if any? Separated from the dialog so a
   caller can know a refusal is confirmable WITHOUT asking — see the "already
   confirmed" rule in postWithConfirmations, which must not re-open a dialog it
   is about to give up on. */
export function matchConfirmableRefusal(error, refusals = CONFIRMABLE_REFUSALS) {
  const s = String(error || '');
  return refusals.find(([marker]) => s.includes(marker)) || null;
}

/* confirmableRetryFlag(error, actionLabel) -> flag | 'declined' | null.
   null = not a confirmable refusal (the caller surfaces it as a plain error). */
export function confirmableRetryFlag(error, actionLabel, refusals = CONFIRMABLE_REFUSALS) {
  const hit = matchConfirmableRefusal(error, refusals);
  if (!hit) return null;
  const [marker, flag] = hit;
  const s = String(error || '');
  return window.confirm(s.replace(marker, '') + `\n\n${actionLabel}?`) ? flag : 'declined';
}

/* The confirm-and-resubmit LOOP, once, for every lane that needs it.

   `post(body)` must reject on refusal (that is what api/fetchClient's postJson
   does — a 400 throws, and for a 400 it shows nothing on its own, so a caller
   without a catch is a button that visibly does nothing: GitHub #23, 1Tomber).

   Returns the successful response, or null when the user declined a confirm —
   the decline IS the answer, not an error to report.

   Two safety rules the hand-rolled loops did not have:
    * a flag is only ever added ONCE. If the same refusal comes back after we
      answered it, the server did not honour the flag (an endpoint that ignores
      it, a legacy row) — rethrow so the caller SAYS so. Re-confirming forever
      would be the same dead button wearing a dialog.
    * anything that is not a confirmable refusal is rethrown untouched, so the
      caller's catch can surface the real message. */
export async function postWithConfirmations(post, body, actionLabel,
                                            refusals = CONFIRMABLE_REFUSALS) {
  let payload = { ...(body || {}) };
  for (;;) {
    let res;
    let thrown = null;
    try {
      res = await post(payload);
    } catch (e) {
      thrown = e;
    }
    // Two client dialects speak here. api/fetchClient's postJson REJECTS on a
    // refusal; hooks/useDataset's postJson NEVER throws and resolves it as
    // {ok:false, error} instead. Listening for rejections only let a resolved
    // 409 sail through as a success: the dataset panel toasted "Cloud run
    // created" while the server had refused and created nothing.
    if (!thrown && !(res && res.ok === false)) return res;
    const message = thrown ? thrown.message : String(res.error || '');
    const hit = matchConfirmableRefusal(message, refusals);
    // Not confirmable, or we ALREADY carried this flag and the refusal came
    // back anyway. Asked before any dialog opens: being asked a question that
    // is about to be ignored is its own kind of dead button. Either way it
    // LEAVES AS A REJECTION, whatever dialect it arrived in — returning the
    // ok:false envelope would hand the caller a refusal wearing a success.
    if (!hit || payload[hit[1]]) {
      if (thrown) throw thrown;
      const err = new Error(message || 'Unexpected error');
      err.body = res;
      throw err;
    }
    const flag = confirmableRetryFlag(message, actionLabel, refusals);
    if (flag === 'declined') return null;
    payload = { ...payload, [flag]: true };
  }
}

export default confirmableRetryFlag;

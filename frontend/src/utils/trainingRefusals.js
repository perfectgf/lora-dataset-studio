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
];

/* confirmableRetryFlag(error, actionLabel) -> flag | 'declined' | null.
   null = not a confirmable refusal (the caller surfaces it as a plain error). */
export function confirmableRetryFlag(error, actionLabel) {
  const s = String(error || '');
  for (const [marker, flag] of CONFIRMABLE_REFUSALS) {
    if (s.includes(marker)) {
      return window.confirm(s.replace(marker, '') + `\n\n${actionLabel}?`) ? flag : 'declined';
    }
  }
  return null;
}

export default confirmableRetryFlag;

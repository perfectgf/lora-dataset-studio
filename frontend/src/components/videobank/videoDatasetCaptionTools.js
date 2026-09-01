/** ✍ Bulk caption edits on a video dataset, decided here and applied by the
 * caller one clip at a time.
 *
 * WHY A PLAN AND NOT A REQUEST. The image lane posts one bulk route and the
 * server rewrites every caption; the video lane has no such route, and inventing
 * one was not this wave's job. So the shape is: build the list of clips whose
 * text ACTUALLY changes, show the user that number before anything is sent, then
 * replay the existing per-clip endpoint over exactly that list. The important
 * property is the same either way — every clip touched gets its .txt sidecar
 * rewritten, because that file is what the trainer reads.
 *
 * The plan excludes no-ops on purpose. "Replace 'woman' with 'woman'" over 300
 * clips is 300 disk writes for nothing, and a progress bar that counts them is a
 * progress bar that lies about what it did.
 */
import { hasCaption } from './videoDatasetClips.js';

export const CAPTION_OPS = Object.freeze(['replace', 'prefix', 'suffix']);

/** Escape a user string for a RegExp — the find field takes literal text, and a
 * caption full of parentheses is ordinary rather than exotic. */
function literal(text) {
  return String(text || '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function nextCaption(caption, op) {
  const current = String(caption || '');
  if (op.kind === 'prefix') {
    const text = String(op.text || '').trim();
    if (!text) return current;
    // Never twice: running a prefix pass again after fixing three clips is the
    // normal way this is used, and the naive version doubles it everywhere else.
    if (current.toLowerCase().startsWith(text.toLowerCase())) return current;
    return current ? `${text}, ${current}` : text;
  }
  if (op.kind === 'suffix') {
    const text = String(op.text || '').trim();
    if (!text) return current;
    if (current.toLowerCase().endsWith(text.toLowerCase())) return current;
    return current ? `${current}, ${text}` : text;
  }
  const find = String(op.find || '');
  if (!find) return current;
  const flags = 'gi';
  const pattern = op.wholeWord
    ? new RegExp(`(^|[^\\p{L}\\p{N}])${literal(find)}(?=[^\\p{L}\\p{N}]|$)`, `${flags}u`)
    : new RegExp(literal(find), flags);
  const replaced = op.wholeWord
    ? current.replace(pattern, (_m, lead) => `${lead}${op.replace ?? ''}`)
    : current.replace(pattern, op.replace ?? '');
  // An empty replacement leaves ", , " behind, which trains as a comma. Tidy the
  // separators rather than asking the user to.
  return replaced.replace(/\s*,\s*,+/g, ', ').replace(/^\s*,\s*/, '').replace(/\s*,\s*$/, '')
    .replace(/[ \t]{2,}/g, ' ').trim();
}

/** The clips this operation really changes: [{id, filename, before, after}].
 *
 * `prefix` runs over EVERY clip (a trigger-style prefix belongs on the silent
 * ones too); `replace` and `suffix` only over clips that already say something,
 * because appending to nothing is how a set of empty captions quietly becomes a
 * set of identical ones. */
export function captionEditPlan(clips, op) {
  const list = Array.isArray(clips) ? clips : [];
  if (!op || !CAPTION_OPS.includes(op.kind)) return [];
  const scope = op.kind === 'prefix' ? list : list.filter(hasCaption);
  const out = [];
  for (const clip of scope) {
    const before = String(clip.caption || '');
    const after = nextCaption(before, op);
    if (after !== before) out.push({ id: clip.id, filename: clip.filename, before, after });
  }
  return out;
}

export function captionEditConfirmation(plan, op) {
  const n = Array.isArray(plan) ? plan.length : 0;
  if (!n) return null;
  const what = op?.kind === 'prefix' ? 'gain the prefix'
    : op?.kind === 'suffix' ? 'gain the suffix'
      : 'be rewritten';
  return `${n} caption${n === 1 ? '' : 's'} will ${what}, and ${n === 1 ? 'its' : 'their'} .txt file${n === 1 ? '' : 's'} rewritten on disk.\n\nThis cannot be undone from here — the previous text is not kept anywhere.`;
}

/** What the progress line says while the plan is replayed. Named here so the
 * wording cannot drift between the button and the line under it. */
export function captionEditProgressLabel(done, total) {
  return `Rewriting captions — ${done} of ${total}…`;
}

/** The tail report, and it names FAILURES rather than rounding them off: a
 * sidecar that could not be written is exactly the silent half-success this
 * whole lane is built to refuse.
 *
 * THREE outcomes, not two, and conflating them made the report say something
 * false. The server commits the row BEFORE it tries the sidecar
 * (set_dataset_clip_caption), so:
 *   · `changed`       — row and .txt both hold the new text;
 *   · `sidecarFailed` — the APP shows the new text, the .txt still has the old
 *                       one, and the trainer reads the .txt. The dangerous one;
 *   · `failed`        — the request threw, nothing moved anywhere.
 * The old wording ("the failed ones still hold their previous text") was true of
 * the third and a lie about the second. */
export function captionEditReport({ changed = 0, sidecarFailed = 0, failed = 0 } = {}) {
  const parts = [];
  if (changed) parts.push(`${changed} caption${changed === 1 ? '' : 's'} rewritten, .txt files included`);
  if (sidecarFailed) {
    parts.push(`${sidecarFailed} saved in the app but their .txt could NOT be written — training will read the previous text for those`);
  }
  if (failed) parts.push(`${failed} could not be saved at all and still hold their previous text`);
  if (!parts.length) return 'Nothing was changed.';
  return `${parts.join('; ')}.`;
}

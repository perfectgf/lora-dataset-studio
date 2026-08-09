import { pickDatasetCaptions, keptCaptions } from '../../utils/datasetCaptions';

/** 🎲 Fill the Preview-prompts textarea with real captions from this dataset.
 *
 * The generic defaults describe nobody, so the preview images a run renders
 * every N steps show a subject that has nothing to do with the one being
 * trained. This draws from the text the dataset already carries about its own
 * kept images — the closest thing to "what this LoRA is supposed to produce"
 * that exists without asking the user to write anything.
 *
 * Captions are pasted AS THEY ARE: LDS captions never contain the trigger, and
 * the run adds it when it is absent (concept and character) or deliberately
 * omits it (style) — so the stored text is already the right shape for all
 * three kinds.
 *
 * It replaces the textarea rather than appending: clicking again is a NEW draw,
 * which is what makes it usable as a re-roll instead of a one-shot that piles
 * lines up past the panel's own maximum.
 *
 * A top-level component, not inline JSX, for the reason spelled out above
 * DenseBasePicker: it is used from both the LoRA lane and the full-model lane,
 * and the full-model one is unreachable for a test (that mode is entered from
 * an effect, and effects do not run under renderToStaticMarkup). As its own
 * component both call sites render markup a test can actually execute. */
export function UseDatasetCaptionsButton({
  images = [], max = 5, onPick = null, disabled = false, className = '',
}) {
  const available = keptCaptions(images).length;
  const limit = Math.max(0, Math.min(5, Number(max) || 0));
  const blocked = disabled || available === 0 || limit === 0;
  const title = available === 0
    ? 'No captions yet — caption this dataset\'s kept images first.'
    : limit === 0
      ? 'This recipe accepts no preview prompts.'
      : `Fill with up to ${Math.min(limit, available)} random caption${
        Math.min(limit, available) === 1 ? '' : 's'} from this dataset — click again for a new draw.`;
  return (
    <button type="button" disabled={blocked} title={title}
      aria-label="Use dataset captions as preview prompts"
      onClick={() => {
        if (blocked) return;
        onPick?.(pickDatasetCaptions(images, limit).join('\n'));
      }}
      className={`self-start px-2 py-1 rounded-lg border border-border bg-surface text-content-muted text-[0.6875rem] hover:text-content hover:border-content-subtle disabled:opacity-40 disabled:cursor-not-allowed ${className}`}>
      🎲 Use dataset captions
    </button>
  );
}

/**
 * The clips this studio has produced, newest first — the lane's answer to the
 * image studio's grid.
 *
 * A grid is impossible here (a cell is minutes, not seconds), so comparison is
 * done in time instead of in space: every clip keeps the settings that made it
 * as a row of tags underneath, and two players sitting one above the other is
 * what "this strength is better" actually looks like.
 *
 * `loop` and `muted` on purpose: H3 renders audio, and a list that starts
 * talking the moment it loads is a list nobody leaves open. The controls are
 * there for whoever wants to hear it.
 */
import { Trash2, ThumbsDown, ThumbsUp, RotateCcw, Loader2, Waves, Sparkles } from 'lucide-react';
import { clipVideoUrl, isRunning, renderTimeLabel } from './videoStudioApi';
import { clipTags } from './videoClipTags';

const ACTION = 'flex items-center justify-center gap-1 rounded-lg border px-2 py-1 text-[0.6875rem] min-h-10 lg:min-h-0';

export default function VideoClipHistory({
  clips, onRate, onDelete, onReuse, onVfi, vfiBusy, onNeuralRender, nrBusy, onCompare,
  onJumpTo, hasMore = false, loadingMore = false, onLoadMore,
}) {
  if (!clips.length) {
    return (
      <p className="rounded-xl border border-dashed border-border bg-surface px-3 py-6 text-center text-sm text-content-subtle">
        No clip yet. Pick a LoRA, describe the motion, and generate one.
      </p>
    );
  }
  return (
    <section data-probe-panel="video-studio-clips" className="flex flex-col gap-2">
      {clips.map((clip) => {
        const running = isRunning(clip);
        return (
          <article key={clip.id} id={`video-clip-${clip.id}`} tabIndex={-1}
            className={`flex flex-col gap-2 rounded-xl border bg-surface p-2 sm:flex-row ${
              running ? 'border-amber-400/40' : clip.status === 'failed' ? 'border-red-500/30' : 'border-border'}`}>
            <div className="w-full shrink-0 sm:w-64">
              {clip.status === 'done' ? (
                <video src={clipVideoUrl(clip.id)} controls loop muted playsInline
                  className="w-full rounded-lg border border-border bg-black" />
              ) : (
                <div className={`flex aspect-video w-full flex-col items-center justify-center gap-1.5 rounded-lg border text-xs ${
                  clip.status === 'failed'
                    ? 'border-red-500/40 bg-red-500/5 text-red-300'
                    : 'border-dashed border-amber-400/40 bg-amber-400/5 text-amber-200'}`}>
                  {running && <Loader2 aria-hidden="true" className="h-4 w-4 animate-spin" />}
                  <span className="px-3 text-center">{running ? 'Rendering…' : (clip.error || 'Failed')}</span>
                </div>
              )}
            </div>
            <div className="flex min-w-0 flex-1 flex-col gap-1.5">
              <p className="line-clamp-2 break-words text-sm text-content">{clip.prompt}</p>
              {/* Where a render came from, as a link that scrolls to it: the
                  source is older than its render by construction, and a pair
                  that cannot be seen together reads as a deleted original. */}
              {(clip.nr_of || clip.vfi_of) && onJumpTo && (
                <p className="text-[0.6875rem] text-content-subtle">
                  {clip.nr_of ? 'neural render of' : 'smoothed from'}{' '}
                  <button type="button" onClick={() => onJumpTo(clip.nr_of || clip.vfi_of)}
                    className="underline decoration-dotted underline-offset-2 hover:text-content">
                    clip #{clip.nr_of || clip.vfi_of}
                  </button>
                </p>
              )}
              {/* The facts that made this clip, one pill each — comparing two
                  cards is reading which pill differs. */}
              <div className="flex flex-wrap gap-1">
                {clipTags(clip).map((t) => (
                  <span key={t}
                    className="max-w-full truncate rounded-full border border-border bg-surface-raised px-2 py-px text-[0.6875rem] text-content-muted">
                    {t}
                  </span>
                ))}
              </div>
              <p className="text-[0.6875rem] text-content-subtle">
                {clip.mode === 't2v' ? 'text-to-video' : 'image-to-video'}
                {clip.seconds ? ` · ${clip.seconds}s` : ''}
                {clip.megapixels ? ` · ${clip.megapixels} MP` : ''}
                {/* ⏱ What the user waited for, model loading included — the
                    number that tells a good run from a swapping machine. A clip
                    that died keeps its time too, under a verb that does not
                    claim a render it never produced. */}
                {renderTimeLabel(clip.render_seconds)
                  ? ` · ${clip.status === 'failed' ? 'failed after' : 'rendered in'} ${renderTimeLabel(clip.render_seconds)}`
                  : ''}
              </p>
              <div className="mt-auto flex flex-wrap items-center gap-1">
                <button type="button" onClick={() => onRate(clip, clip.rating === 1 ? 0 : 1)}
                  title="Keep this one" aria-pressed={clip.rating === 1}
                  className={`${ACTION} ${clip.rating === 1 ? 'border-primary bg-primary/10 text-content' : 'border-border text-content-muted hover:text-content'}`}>
                  <ThumbsUp aria-hidden="true" className="h-3.5 w-3.5" />
                </button>
                <button type="button" onClick={() => onRate(clip, clip.rating === -1 ? 0 : -1)}
                  title="Not this one" aria-pressed={clip.rating === -1}
                  className={`${ACTION} ${clip.rating === -1 ? 'border-red-500/50 bg-red-500/10 text-content' : 'border-border text-content-muted hover:text-content'}`}>
                  <ThumbsDown aria-hidden="true" className="h-3.5 w-3.5" />
                </button>
                <button type="button" onClick={() => onReuse(clip)}
                  title="Load these settings back into the panel"
                  className={`${ACTION} border-border text-content-muted hover:text-content`}>
                  <RotateCcw aria-hidden="true" className="h-3.5 w-3.5" />Reuse
                </button>
                {/* ↗ VFI — the same RIFE pass the image generator runs, on a
                    clip that has finished. Offered only there: interpolating a
                    file that does not exist yet is the one thing this button
                    cannot mean. It makes a NEW clip, so the pair can be
                    compared — which is what this whole screen is for. */}
                {clip.status === 'done' && !clip.vfi_of && onVfi && (
                  <button type="button" onClick={() => onVfi(clip)}
                    disabled={vfiBusy === clip.id}
                    title={`Smooth this clip — interpolate to ${Math.round((clip.fps || 24) * 2)} fps, as a new clip`}
                    className={`${ACTION} border-border text-content-muted hover:text-content disabled:opacity-40`}>
                    <Waves aria-hidden="true" className="h-3.5 w-3.5" />
                    {vfiBusy === clip.id ? '…' : 'Smooth'}
                  </button>
                )}
                {/* ✨ DLSS 5 Neural Rendering over a finished clip — a NEW clip,
                    same rule as Smooth: the studio compares, it never edits.
                    Offered on every finished clip, a render included: a second
                    pass with other dials is a legitimate comparison. */}
                {clip.status === 'done' && onNeuralRender && (
                  <button type="button" onClick={() => onNeuralRender(clip)}
                    disabled={nrBusy === clip.id}
                    title="Re-render this clip with DLSS 5 Neural Rendering, as a new clip"
                    className={`${ACTION} border-border text-content-muted hover:text-content disabled:opacity-40`}>
                    <Sparkles aria-hidden="true" className="h-3.5 w-3.5" />
                    {nrBusy === clip.id ? '…' : 'Neural'}
                  </button>
                )}
                {/* ⇔ A rendered clip against the clip it was rendered from, in
                    step — the comparison this whole screen exists for, on the
                    one pair where nothing but the render differs. */}
                {clip.status === 'done' && clip.nr_of && onCompare && (
                  <button type="button" onClick={() => onCompare(clip)}
                    title={`Play clip #${clip.nr_of} (the source) next to this render, in step`}
                    className={`${ACTION} border-border text-content-muted hover:text-content`}>
                    ⇔ Compare
                  </button>
                )}
                <button type="button" onClick={() => onDelete(clip)} title="Delete this clip"
                  className={`${ACTION} ml-auto border-border text-content-muted hover:border-red-500/50 hover:text-red-300`}>
                  <Trash2 aria-hidden="true" className="h-3.5 w-3.5" />
                </button>
              </div>
            </div>
          </article>
        );
      })}
      {/* The history is paged: 24 newest, then this. Newest first stays true
          after a load — the page merges below what is already there. */}
      {hasMore && onLoadMore && (
        <button type="button" onClick={onLoadMore} disabled={loadingMore}
          className="min-h-10 self-center rounded-lg border border-border bg-surface px-3 py-1 text-xs text-content-muted hover:text-content disabled:opacity-50 lg:min-h-0">
          {loadingMore ? 'Loading…' : 'Load older clips'}
        </button>
      )}
    </section>
  );
}

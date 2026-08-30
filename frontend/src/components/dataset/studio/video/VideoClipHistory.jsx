/**
 * The clips this studio has produced, newest first — the lane's answer to the
 * image studio's grid.
 *
 * A grid is impossible here (a cell is minutes, not seconds), so comparison is
 * done in time instead of in space: every clip keeps the settings that made it
 * on the line underneath, and two players sitting one above the other is what
 * "this strength is better" actually looks like.
 *
 * `loop` and `muted` on purpose: H3 renders audio, and a list that starts
 * talking the moment it loads is a list nobody leaves open. The controls are
 * there for whoever wants to hear it.
 */
import { Trash2, ThumbsDown, ThumbsUp, RotateCcw } from 'lucide-react';
import { clipSummary, clipVideoUrl, isRunning } from './videoStudioApi';

export default function VideoClipHistory({ clips, onRate, onDelete, onReuse }) {
  if (!clips.length) {
    return (
      <p className="rounded-xl border border-border bg-surface px-3 py-6 text-center text-sm text-content-subtle">
        No clip yet. Pick a LoRA, describe the motion, and generate one.
      </p>
    );
  }
  return (
    <section data-probe-panel="video-studio-clips" className="flex flex-col gap-2">
      {clips.map((clip) => (
        <article key={clip.id}
          className="flex flex-col gap-1.5 rounded-xl border border-border bg-surface p-2">
          <div className="flex flex-col gap-2 sm:flex-row">
            <div className="w-full shrink-0 sm:w-64">
              {clip.status === 'done' ? (
                <video src={clipVideoUrl(clip.id)} controls loop muted playsInline
                  className="w-full rounded-lg border border-border bg-black" />
              ) : (
                <div className={`flex aspect-video w-full items-center justify-center rounded-lg border text-xs ${
                  clip.status === 'failed'
                    ? 'border-red-500/40 bg-red-500/5 text-red-300'
                    : 'border-border bg-app text-content-subtle'}`}>
                  {isRunning(clip) ? 'Rendering…' : (clip.error || 'Failed')}
                </div>
              )}
            </div>
            <div className="flex min-w-0 flex-1 flex-col gap-1">
              <p className="line-clamp-2 break-words text-sm text-content">{clip.prompt}</p>
              <p className="break-words text-[0.6875rem] text-content-subtle">
                {clipSummary(clip)}
              </p>
              <p className="text-[0.6875rem] text-content-subtle">
                {clip.mode === 't2v' ? 'text-to-video' : 'image-to-video'}
                {clip.seconds ? ` · ${clip.seconds}s` : ''}
                {clip.megapixels ? ` · ${clip.megapixels} MP` : ''}
              </p>
              <div className="mt-auto flex flex-wrap items-center gap-1">
                <button type="button" onClick={() => onRate(clip, clip.rating === 1 ? 0 : 1)}
                  title="Keep this one" aria-pressed={clip.rating === 1}
                  className={`rounded-lg border px-2 py-1 min-h-10 lg:min-h-0 ${
                    clip.rating === 1 ? 'border-accent bg-accent/10 text-content' : 'border-border text-content-muted'}`}>
                  <ThumbsUp aria-hidden="true" className="h-3.5 w-3.5" />
                </button>
                <button type="button" onClick={() => onRate(clip, clip.rating === -1 ? 0 : -1)}
                  title="Not this one" aria-pressed={clip.rating === -1}
                  className={`rounded-lg border px-2 py-1 min-h-10 lg:min-h-0 ${
                    clip.rating === -1 ? 'border-red-500/50 bg-red-500/10 text-content' : 'border-border text-content-muted'}`}>
                  <ThumbsDown aria-hidden="true" className="h-3.5 w-3.5" />
                </button>
                <button type="button" onClick={() => onReuse(clip)}
                  title="Load these settings back into the panel"
                  className="flex items-center gap-1 rounded-lg border border-border px-2 py-1 text-[0.6875rem] text-content-muted hover:text-content min-h-10 lg:min-h-0">
                  <RotateCcw aria-hidden="true" className="h-3.5 w-3.5" />Reuse
                </button>
                <button type="button" onClick={() => onDelete(clip)} title="Delete this clip"
                  className="ml-auto rounded-lg border border-border px-2 py-1 text-content-muted hover:text-red-300 min-h-10 lg:min-h-0">
                  <Trash2 aria-hidden="true" className="h-3.5 w-3.5" />
                </button>
              </div>
            </div>
          </div>
        </article>
      ))}
    </section>
  );
}

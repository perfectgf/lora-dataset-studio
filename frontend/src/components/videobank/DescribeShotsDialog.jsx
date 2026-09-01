import { useEffect, useState } from 'react'
import { HelpBadge } from '../../help/HelpMode'

/** 🗣 The launch window of the Describe pass — the options, AT the button.
 *
 * The wording choice used to be a lone dropdown at the bottom of the page,
 * three screens away from the button it configured. On the page it read as an
 * app setting; at launch time it was invisible, so the pass people actually
 * started ran with whatever the dropdown happened to hold — measured on real
 * adult footage, that is the difference between captions that name the shot
 * and captions that talk around it (see CAPTION_STYLES in video_caption.py).
 * The image bank asks its caption questions in a launch window; this is the
 * same shape on the video side (maintainer, 2026-08-30).
 *
 * The redo choice exists because a wording switch is only useful if it can
 * REWRITE what the previous wording produced: a resume-only pass would skip
 * every already-captioned shot and the new wording would apply to nothing.
 * `recaption` is the server's word for that; captions a human edited by hand
 * stay untouched unless the third, separate opt-in says otherwise.
 */
export default function DescribeShotsDialog({ captionModel, initialStyle, onLaunch, onClose }) {
  const styles = captionModel?.styles || []
  const models = captionModel?.models || []
  const [style, setStyle] = useState(
    initialStyle || captionModel?.style || styles[0]?.key || 'standard')
  // The configured checkpoint is the default of the picker, whatever it is —
  // the window offers the vetted alternatives, it never silently re-points.
  const [model, setModel] = useState(captionModel?.model || models[0]?.key || '')
  const [recaption, setRecaption] = useState(false)
  const [includeEdited, setIncludeEdited] = useState(false)

  // The shared dialog mechanics (PassDialog's): Escape closes, and so does a
  // click on the backdrop — but never a drag that merely ends there.
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const submit = (e) => {
    e.preventDefault()
    onLaunch({
      style,
      ...(model ? { model } : {}),
      recaption,
      // Meaningless without recaption, so it is never sent without it — the
      // server treats them independently and this window promises otherwise.
      include_edited: recaption && includeEdited,
    })
  }

  return (
    <div role="dialog" aria-modal="true" aria-label="Describe shots" data-probe-layer
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-2 sm:p-4"
      onMouseDown={(e) => { if (e.target === e.currentTarget) onClose() }}>
      {/* The image lane's PassDialog chrome, worn here: a pinned header, a
          scrolling body, a pinned footer — the launch button can never scroll
          out of reach of the choices above it. */}
      <form onSubmit={submit}
        className="flex w-full max-w-lg max-h-[92vh] flex-col overflow-hidden rounded-xl border border-border bg-surface-overlay shadow-2xl">
        <header className="shrink-0 space-y-1 border-b border-border p-4">
        <h2 className="flex items-center gap-2 text-base font-bold text-content">
          🗣 Describe shots
          <HelpBadge topic="video-captions" />
        </h2>
        <p className="text-sm text-content-muted">
          Watches each shot and writes what happens in it — the caption a
          training run reads as the shot&rsquo;s prompt.
        </p>
        {/* WHICH ENGINE, before the click: this app also drives Ollama,
            LM Studio and JoyCaption for images, so without this line the fair
            guess is "it goes through Ollama" — and the wrong guess costs
            someone a wait on a server that has nothing to do with this pass. */}
        <p className="text-xs text-content-subtle">
          {captionModel?.runtime?.backend === 'local_llm'
            ? `Runs through ${captionModel.runtime.label} — the local server `
              + 'this machine already operates — using its vision model'
              + `${captionModel.runtime.model ? ` ${captionModel.runtime.model}` : ''}. `
              + 'Change either in Settings → Local LLM.'
            : 'Runs in LDS\u2019s own local captioning process (Hugging Face '
              + `Transformers${captionModel?.runtime?.device
                ? `, on the ${captionModel.runtime.device.toUpperCase()}` : ''}) `
              + '\u2014 not Ollama or LM\u00a0Studio. Models download from Hugging Face.'}
        </p>
        </header>

        <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-3 sm:p-4">
        {captionModel?.runtime?.backend !== 'local_llm' && models.length > 1 && (
          <fieldset>
            <legend className="text-sm font-medium text-content">Model</legend>
            <div className="mt-1 space-y-1.5">
              {models.map((m) => (
                <label key={m.key}
                  className="flex cursor-pointer items-start gap-2 rounded-md border border-border bg-surface-raised px-3 py-2 text-sm text-content has-[:checked]:border-sky-400/70 has-[:checked]:bg-sky-500/10">
                  <input type="radio" name="describe-model" value={m.key}
                    checked={model === m.key} onChange={() => setModel(m.key)}
                    className="mt-0.5" />
                  <span className="min-w-0">
                    <span className="font-semibold">{m.label}</span>
                    {/* Downloads are allowed but never silent — the same rule
                        as the job line's download notice, one screen earlier. */}
                    <span className={`ml-2 text-[0.6875rem] ${m.cached ? 'text-emerald-300' : 'text-amber-300'}`}>
                      {m.cached ? 'on this machine' : 'downloads on first run'}
                    </span>
                    {m.hint && (
                      <span className="block text-xs text-content-muted">{m.hint}</span>
                    )}
                  </span>
                </label>
              ))}
            </div>
          </fieldset>
        )}

        <fieldset>
          <legend className="flex items-center gap-2 text-sm font-medium text-content">
            Wording
            <HelpBadge topic="video-caption-wording" />
          </legend>
          <div className="mt-1 space-y-1.5">
            {styles.map((s) => (
              <label key={s.key}
                className="flex cursor-pointer items-start gap-2 rounded-md border border-border bg-surface-raised px-3 py-2 text-sm text-content has-[:checked]:border-sky-400/70 has-[:checked]:bg-sky-500/10">
                <input type="radio" name="describe-wording" value={s.key}
                  checked={style === s.key} onChange={() => setStyle(s.key)}
                  className="mt-0.5" />
                <span className="min-w-0">
                  <span className="font-semibold">{s.label}</span>
                  {s.hint && (
                    <span className="block text-xs text-content-muted">{s.hint}</span>
                  )}
                </span>
              </label>
            ))}
          </div>
        </fieldset>

        <fieldset>
          <legend className="text-sm font-medium text-content">Which shots</legend>
          <div className="mt-1 space-y-1.5">
            <label className="flex cursor-pointer items-start gap-2 rounded-md border border-border bg-surface-raised px-3 py-2 text-sm text-content has-[:checked]:border-sky-400/70 has-[:checked]:bg-sky-500/10">
              <input type="radio" name="describe-scope" checked={!recaption}
                onChange={() => setRecaption(false)} className="mt-0.5" />
              <span className="min-w-0">
                <span className="font-semibold">Only shots without a caption</span>
                <span className="block text-xs text-content-muted">
                  Resumes where the last pass stopped. Shots that already have a
                  caption keep it — including one written with another wording.
                </span>
              </span>
            </label>
            <label className="flex cursor-pointer items-start gap-2 rounded-md border border-border bg-surface-raised px-3 py-2 text-sm text-content has-[:checked]:border-sky-400/70 has-[:checked]:bg-sky-500/10">
              <input type="radio" name="describe-scope" checked={recaption}
                onChange={() => setRecaption(true)} className="mt-0.5" />
              <span className="min-w-0">
                <span className="font-semibold">Rewrite every caption with this wording</span>
                <span className="block text-xs text-content-muted">
                  Recaptions the whole bank. What you edited by hand is kept
                  unless you also tick the box below.
                </span>
              </span>
            </label>
            {recaption && (
              <label className="ml-6 flex cursor-pointer items-center gap-2 text-xs text-content-muted">
                <input type="checkbox" checked={includeEdited}
                  onChange={(e) => setIncludeEdited(e.target.checked)} />
                Also overwrite the captions I edited by hand
              </label>
            )}
          </div>
        </fieldset>

        </div>

        <div className="shrink-0 border-t border-border p-3">
          <div className="flex flex-wrap justify-end gap-2">
            <button type="button" onClick={onClose}
              className="min-h-10 rounded-md border border-border px-3 py-1.5 text-sm text-content-muted hover:bg-surface-raised hover:text-content lg:min-h-0">
              Cancel
            </button>
            <button type="submit"
              className="min-h-10 rounded-md bg-gradient-primary px-4 py-1.5 text-sm font-semibold text-gray-950 lg:min-h-0">
              🗣 Describe
            </button>
          </div>
        </div>
      </form>
    </div>
  )
}

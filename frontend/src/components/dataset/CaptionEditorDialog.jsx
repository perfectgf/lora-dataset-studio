import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { captionCharacterLabel, isCaptionSaveShortcut, isLikelyTruncatedCaption } from '../../utils/captionEditor';
import { attemptModalSubmit } from '../../utils/submitOutcome.js';
import CaptionLab from './CaptionLab';
// WHO wrote the sentence in the box — this is the surface where a caption is read
// in full, so it is the one that says it in words rather than as a chip.
import { captionOriginInfo } from '../../utils/captionOrigin.js';

/** The authorship line for a text box, or null.
 *
 *  IT DESCRIBES THE SAVED TEXT, NOT THE DRAFT. The stamp on the row is about the
 *  sentence the server holds; the moment the user types, the two are different
 *  things and claiming the old author for the new words would be exactly the kind
 *  of confident wrong label this whole feature exists to remove. So an edited draft
 *  says what it really is — unsaved, and yours once saved.
 */
function authorshipNote(draft, saved, origin) {
  const text = (draft || '');
  if (!text.trim()) return null;
  if (text !== (saved || '')) {
    return { text: 'Unsaved edit — saving records this caption as written by you.',
             title: 'A caption you save is stamped as yours, and a forced 🔄 Re-caption '
               + 'then skips it.' };
  }
  const info = captionOriginInfo(origin);
  return { text: info.short, title: info.title };
}

export default function CaptionEditorDialog({
  initialCaption, initialShortCaption, showShort = false, imageUrl, imageLabel, onClose, onSave,
  labSurface = null, captionOrigin = null, shortCaptionOrigin = null,
  initialMode = 'edit', onPickAnotherImage = null,
  // "without the face" is a DATASET rule (identity must bind to the trigger, not to the
  // text). A bank caption is the opposite: a plain description whose job is search. The
  // dialog is mounted on both now, so the instruction belongs to the host.
  captionPlaceholder = 'Caption (without the face)…',
}) {
  const [draft, setDraft] = useState(initialCaption || '');
  const [shortDraft, setShortDraft] = useState(initialShortCaption || '');
  // Collapsed by default; auto-open when a short already exists so it isn't hidden.
  const [shortOpen, setShortOpen] = useState(Boolean((initialShortCaption || '').trim()));
  // 'edit' (the default caption editor) | 'lab' (🧪 try several caption configs). The Lab
  // is one click away and only offered when we know which image to caption.
  //
  // `labAvailable` is read by the initialiser below, so it is declared BEFORE it:
  // a useState initialiser runs on the first render and would read `undefined`
  // from a const declared under it.
  // The bench is offered when the HOST could name a surface for it — which is the
  // Dataset (a row it owns) or the Bank (a bank image). A host that cannot say where a
  // candidate would run gets the plain editor, not a button that would 404.
  const labAvailable = labSurface != null;
  // The Captions section opens this dialog to reach the BENCH, not the textarea
  // — landing on 'edit' there would hide the Lab behind one more click, which is
  // the whole defect that entry point exists to fix.
  const [mode, setMode] = useState(labAvailable && initialMode === 'lab' ? 'lab' : 'edit');
  const textareaRef = useRef(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  /* ONE way out, and it is closed only while the save is in flight — never
     because the server said no. Escape, the backdrop, ✕ and Cancel all route
     here: a dismissal mid-POST would leave the write running with nothing on
     screen to report it. */
  const dismiss = () => { if (!busy) onClose(); };

  /* LEAVING FOR ANOTHER IMAGE IS NOT A DISMISSAL — it unmounts the dialog, so the draft
     goes with it. This whole component exists because a refused save once destroyed a
     caption the user had just written; a new button that throws one away silently would
     put the same hole back, one door further along. So a dirty draft is refused ONCE,
     in words, and the second press leaves anyway (the user has now been told). */
  const dirty = draft !== (initialCaption || '')
    || (showShort && shortDraft !== (initialShortCaption || ''));
  const [leaveArmed, setLeaveArmed] = useState(false);
  const leaveForAnotherImage = () => {
    if (busy) return;
    if (dirty && !leaveArmed) {
      setLeaveArmed(true);
      setError('You have an unsaved caption on this image. Save it first, or press '
        + '‹ Another image again to leave it behind.');
      return;
    }
    setLeaveArmed(false);
    onPickAnotherImage();
  };

  /* A refusal used to land on a caption editor that had already closed, so the
     long AND short captions the user had just written were gone — on the most
     used screen in the app. Now the dialog stays, the text is still in the
     textareas, and the reason is drawn under them. The card scrolls inside
     itself, so scroll to the end: the message and the button that produced it
     are the last two blocks (measured on ContinueDialog at 400 px —
     scrollIntoView({block:'nearest'}) left the message half-cut under the fold). */
  const panelRef = useRef(null);
  useEffect(() => {
    const panel = panelRef.current;
    if (error && panel) panel.scrollTop = panel.scrollHeight;
  }, [error]);

  useEffect(() => {
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    textareaRef.current?.focus();
    /* CAPTURE PHASE, and the event dies here: on the Bank this dialog is opened from
       inside the 🏷️ Caption launch window, which also closes on Escape. One key press
       was closing both. The topmost layer owns the key. */
    const closeOnEscape = (event) => {
      if (event.key !== 'Escape') return;
      event.stopImmediatePropagation();
      dismiss();
    };
    window.addEventListener('keydown', closeOnEscape, true);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener('keydown', closeOnEscape, true);
    };
  }, [onClose, busy]);  // eslint-disable-line react-hooks/exhaustive-deps

  // Pass the short only when the dataset uses dual captions, so a plain edit never writes one.
  const save = async () => {
    if (busy) return;
    setBusy(true);
    setError(null);
    let outcome;
    try {
      outcome = await attemptModalSubmit(
        () => onSave(draft, showShort ? shortDraft : undefined),
        { fallback: 'Could not save the caption' });
    } finally { setBusy(false); }
    if (outcome.close) onClose();
    else setError(outcome.error);
  };

  return createPortal(
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/75 p-3 sm:p-6"
      onMouseDown={(event) => { if (event.target === event.currentTarget) dismiss(); }}>
      {/* chrome + layer, the DatasetLightbox pattern: the responsive probe scans
          touch targets and truncation ONLY inside a [data-probe-chrome] subtree,
          so an unmarked dialog is never "clean" — it is unmeasured. `layer` keeps
          it out of the fold budget and the overlap pairing: covering the page is
          what a dialog is for. */}
      <section role="dialog" aria-modal="true" aria-labelledby="caption-editor-title"
        data-probe-chrome="caption-editor" data-probe-layer
        className="flex h-[min(92vh,50rem)] w-[min(96vw,72rem)] flex-col overflow-hidden rounded-2xl border border-border bg-app shadow-2xl">
        <header className="flex items-start justify-between gap-4 border-b border-border bg-surface px-4 py-3 sm:px-5">
          <div className="flex min-w-0 items-center gap-3">
            <div className="min-w-0">
              {/* Opened from the Captions section, the bench is about the SET, not
                  about the one image that happened to be picked — so the eyebrow
                  becomes the way back to the picker. `inline-flex items-center`
                  travels with the 40 px touch floor: an inline box ignores
                  min-height, which is how a "floored" control stays at 21 px. */}
              {onPickAnotherImage ? (
                <button type="button" onClick={leaveForAnotherImage} disabled={busy}
                  title={dirty
                    ? 'You have an unsaved caption on this image — save it or clear the edit first'
                    : 'Pick another image to bench captions on'}
                  className="m-0 inline-flex min-h-10 items-center text-[0.6875rem] font-semibold uppercase tracking-[0.18em] text-content-subtle hover:text-content disabled:opacity-40 lg:min-h-0">
                  ‹ Another image
                </button>
              ) : (
                <p className="m-0 text-[0.6875rem] font-semibold uppercase tracking-[0.18em] text-content-subtle">Dataset image</p>
              )}
              <h2 id="caption-editor-title" className="m-0 mt-0.5 text-lg font-semibold text-content">
                {mode === 'lab' ? 'Caption Lab' : 'Edit caption'}
              </h2>
            </div>
            {labAvailable && (
              <div className="ml-1 flex rounded-lg border border-border bg-app p-0.5" role="tablist" aria-label="Caption editor mode">
                {/* min-h-10 lg:min-h-0 with inline-flex: the two tabs measured 24 px
                    on a phone, and an inline box ignores min-height on its own. */}
                <button type="button" role="tab" aria-selected={mode === 'edit'} onClick={() => setMode('edit')}
                  className={`inline-flex min-h-10 items-center rounded-md px-2.5 py-1 text-xs font-semibold lg:min-h-0 ${mode === 'edit' ? 'bg-surface text-content shadow-sm' : 'text-content-muted hover:text-content'}`}>
                  Edit
                </button>
                <button type="button" role="tab" aria-selected={mode === 'lab'} onClick={() => setMode('lab')}
                  className={`inline-flex min-h-10 items-center rounded-md px-2.5 py-1 text-xs font-semibold lg:min-h-0 ${mode === 'lab' ? 'bg-surface text-content shadow-sm' : 'text-content-muted hover:text-content'}`}>
                  🧪 Caption Lab
                </button>
              </div>
            )}
          </div>
          <button type="button" onClick={dismiss} disabled={busy} aria-label="Close expanded caption editor"
            className="inline-flex min-h-10 items-center rounded-lg border border-border bg-app px-2.5 py-1.5 text-sm text-content-muted hover:text-content disabled:opacity-40 lg:min-h-0">
            ✕
          </button>
        </header>

        <div className="grid min-h-0 flex-1 grid-rows-[minmax(6.5rem,26%)_1fr] sm:grid-rows-[minmax(9rem,34%)_1fr] md:grid-cols-[minmax(18rem,42%)_1fr] md:grid-rows-1">
          <div className="flex min-h-0 items-center justify-center border-b border-border bg-black md:border-b-0 md:border-r">
            {imageUrl ? (
              <img src={imageUrl} alt={imageLabel || 'Dataset image'}
                className="h-full w-full object-contain" />
            ) : (
              <span className="text-sm text-content-subtle">Image unavailable</span>
            )}
          </div>

          {mode === 'lab' ? (
            <div className="flex min-h-0 flex-col p-4 sm:p-5">
              <CaptionLab surface={labSurface} currentCaption={draft}
                onKeep={(text) => { setDraft(text); setMode('edit'); textareaRef.current?.focus(); }} />
            </div>
          ) : (
          <div ref={panelRef} className="flex min-h-0 flex-col gap-3 overflow-y-auto p-4 sm:p-5">
            <div className="flex items-center justify-between gap-3">
              <label htmlFor="expanded-caption" className="text-sm font-semibold text-content">Caption text</label>
              <span className="font-mono text-[0.6875rem] text-content-subtle" aria-live="polite">
                {captionCharacterLabel(draft)}
              </span>
            </div>
            {authorshipNote(draft, initialCaption, captionOrigin) && (
              <p className="m-0 text-[0.6875rem] leading-relaxed text-content-subtle"
                title={authorshipNote(draft, initialCaption, captionOrigin).title}>
                {authorshipNote(draft, initialCaption, captionOrigin).text}
              </p>
            )}
            {isLikelyTruncatedCaption(initialCaption) && (
              <p className="m-0 rounded-lg border border-amber-400/40 bg-amber-500/10 px-3 py-2 text-[0.6875rem] leading-relaxed text-amber-200">
                This caption is exactly 800 characters and ends mid-sentence — an earlier
                version of the app capped captions there. The cut-off text can’t be recovered;
                re-caption this image to regenerate the full description.
              </p>
            )}
            <textarea id="expanded-caption" ref={textareaRef} value={draft}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={(event) => {
                if (isCaptionSaveShortcut(event)) {
                  event.preventDefault();
                  save();
                }
              }}
              placeholder={captionPlaceholder}
              /* min-h-[6rem], not min-h-0: the column is a flex box, so once the
                 refusal box and the short-caption section are both open at
                 400 px a flex-1 child is free to be squeezed to nothing. The
                 column scrolls instead (overflow-y-auto above). */
              className="min-h-[6rem] flex-1 resize-none rounded-xl border border-border bg-surface p-4 text-sm leading-6 text-content outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-400/25" />

            {showShort && (
              <div className="rounded-xl border border-border bg-surface">
                <button type="button" onClick={() => setShortOpen((open) => !open)}
                  aria-expanded={shortOpen}
                  className="flex w-full items-center justify-between gap-2 px-3 py-2 text-left text-[0.75rem] font-semibold text-content-muted hover:text-content">
                  <span>{shortOpen ? '▾' : '▸'} Short caption <span className="font-normal text-content-subtle">· dual-caption training</span></span>
                  <span className="font-mono text-[0.6875rem] text-content-subtle">{captionCharacterLabel(shortDraft)}</span>
                </button>
                {shortOpen && (
                  <div className="flex flex-col gap-1.5 px-3 pb-3">
                    <p className="m-0 text-[0.6875rem] leading-relaxed text-content-subtle">
                      A brief alternative, trained alongside the long caption. Same rules apply
                      (no trigger, keep the identity/concept/aesthetic out). Leave empty to reuse
                      the long caption; (re-)captioning regenerates it automatically.
                    </p>
                    {/* Its OWN author: the short caption has its own writers on both
                        sides (this box types it, the dual-caption pass derives it),
                        so the long caption's stamp cannot answer for it. */}
                    {authorshipNote(shortDraft, initialShortCaption, shortCaptionOrigin) && (
                      <p className="m-0 text-[0.6875rem] leading-relaxed text-content-subtle"
                        title={authorshipNote(shortDraft, initialShortCaption,
                                              shortCaptionOrigin).title}>
                        {authorshipNote(shortDraft, initialShortCaption,
                                        shortCaptionOrigin).text}
                      </p>
                    )}
                    <textarea value={shortDraft}
                      onChange={(event) => setShortDraft(event.target.value)}
                      onKeyDown={(event) => {
                        if (isCaptionSaveShortcut(event)) {
                          event.preventDefault();
                          save();
                        }
                      }}
                      rows={2} placeholder="Short caption (optional)…"
                      className="resize-none rounded-lg border border-border bg-app p-3 text-sm leading-6 text-content outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-400/25" />
                  </div>
                )}
              </div>
            )}

            {/* The LAST attempt's refusal, right above the button that produced
                it — and the caption it is about is still in the textarea.
                shrink-0 because this column is a flex box (a flex child is
                otherwise squashed to a sliver of clipped text at 400 px), and
                it scrolls in its own box rather than pushing Save off screen. */}
            {error && (
              <div role="alert"
                className="shrink-0 rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 max-h-28 overflow-y-auto">
                <span className="block whitespace-pre-wrap break-words text-[0.6875rem] leading-relaxed text-red-200">
                  {error}
                </span>
                <span className="mt-1 block text-[0.625rem] text-content-subtle">
                  Your caption is kept — adjust and try again.
                </span>
              </div>
            )}

            <div className="flex shrink-0 flex-col-reverse gap-2 border-t border-border pt-3 sm:flex-row sm:items-center sm:justify-between">
              <span className="text-[0.6875rem] text-content-subtle">Esc to close · Ctrl/⌘ + Enter to save</span>
              <div className="flex justify-end gap-2">
                <button type="button" onClick={dismiss} disabled={busy}
                  className="rounded-lg border border-border bg-surface px-3 py-2 text-sm font-medium text-content-muted hover:text-content disabled:opacity-40">
                  Cancel
                </button>
                <button type="button" onClick={save} disabled={busy}
                  className="rounded-lg bg-gradient-primary px-4 py-2 text-sm font-semibold text-gray-950 disabled:opacity-40">
                  {busy ? 'Saving…' : 'Save caption'}
                </button>
              </div>
            </div>
          </div>
          )}
        </div>
      </section>
    </div>,
    document.body,
  );
}

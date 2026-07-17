import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { captionCharacterLabel, isCaptionSaveShortcut, isLikelyTruncatedCaption } from '../../utils/captionEditor';

export default function CaptionEditorDialog({ initialCaption, imageUrl, imageLabel, onClose, onSave }) {
  const [draft, setDraft] = useState(initialCaption || '');
  const textareaRef = useRef(null);

  useEffect(() => {
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    textareaRef.current?.focus();
    const closeOnEscape = (event) => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', closeOnEscape);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener('keydown', closeOnEscape);
    };
  }, [onClose]);

  const save = () => onSave(draft);

  return createPortal(
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/75 p-3 sm:p-6"
      onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
      <section role="dialog" aria-modal="true" aria-labelledby="caption-editor-title"
        className="flex h-[min(92vh,50rem)] w-[min(96vw,72rem)] flex-col overflow-hidden rounded-2xl border border-border bg-app shadow-2xl">
        <header className="flex items-start justify-between gap-4 border-b border-border bg-surface px-4 py-3 sm:px-5">
          <div>
            <p className="m-0 text-[0.6875rem] font-semibold uppercase tracking-[0.18em] text-content-subtle">Dataset image</p>
            <h2 id="caption-editor-title" className="m-0 mt-0.5 text-lg font-semibold text-content">Edit caption</h2>
          </div>
          <button type="button" onClick={onClose} aria-label="Close expanded caption editor"
            className="rounded-lg border border-border bg-app px-2.5 py-1.5 text-sm text-content-muted hover:text-content">
            ✕
          </button>
        </header>

        <div className="grid min-h-0 flex-1 grid-rows-[minmax(9rem,34%)_1fr] md:grid-cols-[minmax(18rem,42%)_1fr] md:grid-rows-1">
          <div className="flex min-h-0 items-center justify-center border-b border-border bg-black md:border-b-0 md:border-r">
            {imageUrl ? (
              <img src={imageUrl} alt={imageLabel || 'Dataset image'}
                className="h-full w-full object-contain" />
            ) : (
              <span className="text-sm text-content-subtle">Image unavailable</span>
            )}
          </div>

          <div className="flex min-h-0 flex-col gap-3 p-4 sm:p-5">
            <div className="flex items-center justify-between gap-3">
              <label htmlFor="expanded-caption" className="text-sm font-semibold text-content">Caption text</label>
              <span className="font-mono text-[0.6875rem] text-content-subtle" aria-live="polite">
                {captionCharacterLabel(draft)}
              </span>
            </div>
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
              placeholder="Caption (without the face)…"
              className="min-h-0 flex-1 resize-none rounded-xl border border-border bg-surface p-4 text-sm leading-6 text-content outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-400/25" />
            <div className="flex flex-col-reverse gap-2 border-t border-border pt-3 sm:flex-row sm:items-center sm:justify-between">
              <span className="text-[0.6875rem] text-content-subtle">Esc to close · Ctrl/⌘ + Enter to save</span>
              <div className="flex justify-end gap-2">
                <button type="button" onClick={onClose}
                  className="rounded-lg border border-border bg-surface px-3 py-2 text-sm font-medium text-content-muted hover:text-content">
                  Cancel
                </button>
                <button type="button" onClick={save}
                  className="rounded-lg bg-gradient-primary px-4 py-2 text-sm font-semibold text-white">
                  Save caption
                </button>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>,
    document.body,
  );
}

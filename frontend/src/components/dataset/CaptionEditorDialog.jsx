import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { captionCharacterLabel, isCaptionSaveShortcut, isLikelyTruncatedCaption } from '../../utils/captionEditor';
import { useI18n } from '../../i18n/I18nContext';

export default function CaptionEditorDialog({
  initialCaption, initialShortCaption, showShort = false, imageUrl, imageLabel, onClose, onSave,
}) {
  const { t } = useI18n();
  const [draft, setDraft] = useState(initialCaption || '');
  const [shortDraft, setShortDraft] = useState(initialShortCaption || '');
  // Collapsed by default; auto-open when a short already exists so it isn't hidden.
  const [shortOpen, setShortOpen] = useState(Boolean((initialShortCaption || '').trim()));
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

  // Pass the short only when the dataset uses dual captions, so a plain edit never writes one.
  const save = () => onSave(draft, showShort ? shortDraft : undefined);

  return createPortal(
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/75 p-3 sm:p-6"
      onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
      <section role="dialog" aria-modal="true" aria-labelledby="caption-editor-title"
        className="flex h-[min(92vh,50rem)] w-[min(96vw,72rem)] flex-col overflow-hidden rounded-2xl border border-border bg-app shadow-2xl">
        <header className="flex items-start justify-between gap-4 border-b border-border bg-surface px-4 py-3 sm:px-5">
          <div>
            <p className="m-0 text-[0.6875rem] font-semibold uppercase tracking-[0.18em] text-content-subtle">{t('workspace.captionEditor.datasetImage')}</p>
            <h2 id="caption-editor-title" className="m-0 mt-0.5 text-lg font-semibold text-content">{t('workspace.captionEditor.title')}</h2>
          </div>
          <button type="button" onClick={onClose} aria-label={t('workspace.captionEditor.close')}
            className="rounded-lg border border-border bg-app px-2.5 py-1.5 text-sm text-content-muted hover:text-content">
            ✕
          </button>
        </header>

        <div className="grid min-h-0 flex-1 grid-rows-[minmax(9rem,34%)_1fr] md:grid-cols-[minmax(18rem,42%)_1fr] md:grid-rows-1">
          <div className="flex min-h-0 items-center justify-center border-b border-border bg-black md:border-b-0 md:border-r">
            {imageUrl ? (
              <img src={imageUrl} alt={imageLabel || t('workspace.captionEditor.datasetImage')}
                className="h-full w-full object-contain" />
            ) : (
              <span className="text-sm text-content-subtle">{t('workspace.captionEditor.imageUnavailable')}</span>
            )}
          </div>

          <div className="flex min-h-0 flex-col gap-3 p-4 sm:p-5">
            <div className="flex items-center justify-between gap-3">
              <label htmlFor="expanded-caption" className="text-sm font-semibold text-content">{t('workspace.captionEditor.captionText')}</label>
              <span className="font-mono text-[0.6875rem] text-content-subtle" aria-live="polite">
                {captionCharacterLabel(draft)}
              </span>
            </div>
            {isLikelyTruncatedCaption(initialCaption) && (
              <p className="m-0 rounded-lg border border-amber-400/40 bg-amber-500/10 px-3 py-2 text-[0.6875rem] leading-relaxed text-amber-200">
                {t('workspace.captionEditor.truncatedWarning')}
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
              placeholder={t('workspace.captionEditor.placeholder')}
              className="min-h-0 flex-1 resize-none rounded-xl border border-border bg-surface p-4 text-sm leading-6 text-content outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-400/25" />

            {showShort && (
              <div className="rounded-xl border border-border bg-surface">
                <button type="button" onClick={() => setShortOpen((open) => !open)}
                  aria-expanded={shortOpen}
                  className="flex w-full items-center justify-between gap-2 px-3 py-2 text-left text-[0.75rem] font-semibold text-content-muted hover:text-content">
                  <span>{shortOpen ? '▾' : '▸'} {t('workspace.captionEditor.shortCaption')}{' '}
                    <span className="font-normal text-content-subtle">· {t('workspace.captionEditor.dualTraining')}</span>
                  </span>
                  <span className="font-mono text-[0.6875rem] text-content-subtle">{captionCharacterLabel(shortDraft)}</span>
                </button>
                {shortOpen && (
                  <div className="flex flex-col gap-1.5 px-3 pb-3">
                    <p className="m-0 text-[0.6875rem] leading-relaxed text-content-subtle">
                      {t('workspace.captionEditor.shortDescription')}
                    </p>
                    <textarea value={shortDraft}
                      onChange={(event) => setShortDraft(event.target.value)}
                      onKeyDown={(event) => {
                        if (isCaptionSaveShortcut(event)) {
                          event.preventDefault();
                          save();
                        }
                      }}
                      rows={2} placeholder={t('workspace.captionEditor.shortPlaceholder')}
                      className="resize-none rounded-lg border border-border bg-app p-3 text-sm leading-6 text-content outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-400/25" />
                  </div>
                )}
              </div>
            )}

            <div className="flex flex-col-reverse gap-2 border-t border-border pt-3 sm:flex-row sm:items-center sm:justify-between">
              <span className="text-[0.6875rem] text-content-subtle">{t('workspace.captionEditor.shortcuts')}</span>
              <div className="flex justify-end gap-2">
                <button type="button" onClick={onClose}
                  className="rounded-lg border border-border bg-surface px-3 py-2 text-sm font-medium text-content-muted hover:text-content">
                  {t('common.close')}
                </button>
                <button type="button" onClick={save}
                  className="rounded-lg bg-gradient-primary px-4 py-2 text-sm font-semibold text-white">
                  {t('workspace.captionEditor.save')}
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

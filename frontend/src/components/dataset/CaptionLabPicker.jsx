import { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { X } from 'lucide-react';
import { imageDisplayName } from './captionLabSurface';

/* 🧪 Caption Lab — the image picker that gives the bench an address in the
   Captions section.
 *
 * WHY IT EXISTS. The Lab could only ever be reached from a dataset TILE: Images →
 * a kept tile → the ⤢ button → the 🧪 tab inside the caption editor. The Captions
 * section — the one screen whose whole subject is captions, and the one the app's
 * own help sends you to when you search "caption lab" (help/topics/workspaceSections.js
 * routes those keywords to ?section=captions) — offered no way in at all.
 *
 * The Lab benches caption configs on ONE image, so an entry point that speaks
 * about the whole set has to name a subject first. That is all this is: pick the
 * image, and the existing CaptionEditorDialog opens straight on the bench.
 *
 * The HOST picks the pile and how to draw a thumbnail, because the two surfaces do not
 * agree on either: a dataset offers its KEPT images and serves them by filename, while a
 * bank holds up to six figures of rows and pages over SQL — so it offers the page you
 * are looking at (or your selection) and serves thumbnails by row id. An image with no
 * caption yet is offered on both: the bench generates, it does not compare stored text.
 */
export default function CaptionLabPicker({ images, thumbUrl, onPick, onClose,
                                          nameOf = imageDisplayName,
                                          emptyNote = 'No image to bench yet.' }) {
  const [query, setQuery] = useState('');
  const searchRef = useRef(null);

  useEffect(() => {
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    searchRef.current?.focus();
    /* CAPTURE PHASE, and the event dies here. On the Bank this picker is opened from
       INSIDE the 🏷️ Caption launch window, and that window closes on Escape too — one
       key press was closing both, taking the run dials (and a config just loaded by
       ⚙️ Use for the next run) with it. The topmost layer owns the key. */
    const closeOnEscape = (event) => {
      if (event.key !== 'Escape') return;
      event.stopImmediatePropagation();
      onClose();
    };
    window.addEventListener('keydown', closeOnEscape, true);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener('keydown', closeOnEscape, true);
    };
  }, [onClose]);

  /* A manga or scraped set arrives in the hundreds, so the list is filterable
     rather than "scroll until you find it" — on caption text AND filename,
     which are the two things a user can actually remember about a row. */
  const shown = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return images;
    return images.filter((img) => `${img.caption || ''} ${nameOf(img)}`
      .toLowerCase().includes(needle));
  }, [images, query, nameOf]);

  return createPortal(
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/75 p-3 sm:p-6"
      onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
      {/* Marked chrome+layer, the DatasetLightbox pattern: the responsive probe
          only measures touch targets and truncation INSIDE a [data-probe-chrome]
          subtree, so an unmarked dialog is not "clean", it is unmeasured. */}
      <section role="dialog" aria-modal="true" aria-labelledby="caption-lab-picker-title"
        data-probe-chrome="caption-lab-picker" data-probe-layer
        className="flex h-[min(92vh,46rem)] w-[min(96vw,60rem)] flex-col overflow-hidden rounded-2xl border border-border bg-app shadow-2xl">
        <header className="flex items-start justify-between gap-3 border-b border-border bg-surface px-4 py-3 sm:px-5">
          <div className="min-w-0">
            <p className="m-0 text-[0.6875rem] font-semibold uppercase tracking-[0.18em] text-content-subtle">Captions</p>
            <h2 id="caption-lab-picker-title" className="m-0 mt-0.5 text-lg font-semibold text-content">🧪 Caption Lab</h2>
          </div>
          <button type="button" onClick={onClose} aria-label="Close the Caption Lab picker"
            className="inline-flex min-h-10 shrink-0 items-center rounded-lg border border-border bg-app px-2.5 text-sm text-content-muted hover:text-content lg:min-h-0 lg:py-1.5">
            <X aria-hidden="true" className="h-4 w-4" />
          </button>
        </header>

        <div className="flex flex-col gap-2 border-b border-border px-4 py-3 sm:px-5">
          <p className="m-0 text-[0.75rem] leading-relaxed text-content-muted">
            The bench runs up to four caption configs — engine, vision model, vocabulary
            register and length — on <strong className="text-content">one</strong> image and
            shows them side by side. Nothing is written until you keep a result, so it costs
            you nothing to try before you re-caption the whole set.
          </p>
          <label className="flex flex-col gap-1">
            <span className="text-[0.6875rem] uppercase tracking-wide text-content-subtle">Pick the image to bench</span>
            <input type="search" ref={searchRef} value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Filter by caption or filename…"
              className="min-h-10 w-full rounded-lg border border-border bg-surface px-3 py-1.5 text-sm text-content outline-none focus:border-indigo-400" />
          </label>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto p-3 sm:p-4">
          {shown.length === 0 ? (
            <p className="m-0 py-6 text-center text-sm text-content-subtle">
              {images.length === 0 ? emptyNote : `No image matches “${query.trim()}”.`}
            </p>
          ) : (
            <ul className="m-0 grid list-none grid-cols-2 gap-2 p-0 sm:grid-cols-3 lg:grid-cols-4">
              {shown.map((img) => (
                <li key={img.id}>
                  <button type="button" onClick={() => onPick(img)}
                    aria-label={`Bench captions on image ${nameOf(img)}`}
                    className="flex w-full flex-col gap-1.5 rounded-xl border border-border bg-surface p-1.5 text-left hover:border-indigo-400/60 hover:bg-surface-raised">
                    <img src={thumbUrl(img)} alt="" loading="lazy"
                      className="aspect-square w-full rounded-lg bg-black object-cover" />
                    {/* Three lines of the stored caption, clamped — enough to tell two
                        rows apart, and `title` carries the whole sentence so the probe's
                        truncation check reads it as a deliberate ellipsis, not a loss. */}
                    <span title={img.caption || undefined}
                      className={`line-clamp-3 px-0.5 text-[0.6875rem] leading-snug ${img.caption ? 'text-content-muted' : 'text-content-subtle italic'}`}>
                      {img.caption || 'No caption yet'}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      </section>
    </div>,
    document.body,
  );
}

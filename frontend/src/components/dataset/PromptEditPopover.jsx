/** Small bubble anchored on a generated tile: shows the core creative prompt,
 *  lets the user edit it, and regenerates that tile with the edit on OK. The
 *  identity guard is re-applied server-side, so this is only the creative half
 *  (expression / scene / lighting) — never the face lock. Presentational only:
 *  the parent wires onSubmit (which calls regenerate) and onClose. */
import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { attemptModalSubmit } from '../../utils/submitOutcome.js';

export default function PromptEditPopover({ initialPrompt = '', onSubmit, onClose }) {
  const [text, setText] = useState(initialPrompt);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const areaRef = useRef(null);
  // Focus the textarea on open and select all so a full rewrite is one keystroke away.
  useEffect(() => {
    const el = areaRef.current;
    if (el) { el.focus(); el.select(); }
  }, []);
  /* ONE way out, shut only while the regenerate is being posted — a dismissal
     mid-request would leave the job starting with nothing on screen. */
  const dismiss = () => { if (!busy) onClose(); };
  /* The bubble used to fire onSubmit WITHOUT awaiting it and close on the next
     line, so a refused regenerate (GPU busy, engine misconfigured, network
     blip) threw away a prompt the user had just rewritten by hand. Now it waits
     for the answer, and only a start closes it. */
  const submit = async () => {
    const t = text.trim();
    if (!t || busy) return;
    setBusy(true);
    setError(null);
    let outcome;
    try {
      outcome = await attemptModalSubmit(() => onSubmit(t),
        { fallback: 'Could not start the regeneration' });
    } finally { setBusy(false); }
    if (outcome.close) onClose();
    else setError(outcome.error);
  };
  /* Portalled to <body> rather than absolutely positioned inside the tile.
     MEASURED at 400 px: pinned inside an M tile the bubble is ~150 px tall, so
     the refusal box added under the textarea rendered as three clipped lines
     overlapping the caption row — the message that exists to save the user's
     prompt was the least readable thing on screen. The tile stays visible
     behind it; only the ceiling moved. */
  return createPortal(
    // Backdrop closes on outside click; stopPropagation keeps clicks from
    // reaching the tile underneath (which would trigger inspect/select).
    <div className="fixed inset-0 z-[9995] flex items-center justify-center bg-black/70 p-4"
      onClick={(e) => { e.stopPropagation(); if (e.target === e.currentTarget) dismiss(); }}>
      {/* bg-surface-overlay, not bg-surface: the latter is a 4 % tint meant to
          sit ON a solid surface. Pinned inside a tile it read fine over the
          image; over the page it was see-through (measured at 400 px — the grid
          headings showed straight through the refusal). */}
      <div role="dialog" aria-modal="true" aria-label="Edit prompt & regenerate"
        className="w-full max-w-[20rem] max-h-[85vh] overflow-y-auto rounded-lg border border-border bg-surface-overlay p-3 shadow-2xl flex flex-col gap-2"
        onClick={(e) => e.stopPropagation()}>
        <span className="text-[0.625rem] uppercase text-content-muted">Edit prompt &amp; regenerate</span>
        <textarea ref={areaRef} value={text} onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Escape') { e.preventDefault(); dismiss(); }
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); submit(); }
          }}
          rows={4} placeholder="describe the shot (the face is kept automatically)…"
          aria-label="Edit the generation prompt"
          className="text-[11px] bg-app/60 border border-border rounded p-1.5 text-content resize-none" />
        {/* shrink-0: this is a flex column with a max height, so the box would
            otherwise be squashed to a clipped sliver. */}
        {error && (
          <div role="alert"
            className="shrink-0 rounded border border-red-500/40 bg-red-500/10 px-2 py-1.5 max-h-24 overflow-y-auto">
            <span className="block whitespace-pre-wrap break-words text-[11px] leading-relaxed text-red-200">
              {error}
            </span>
            <span className="mt-0.5 block text-[10px] text-content-subtle">
              Your prompt is kept — adjust and try again.
            </span>
          </div>
        )}
        <div className="flex gap-1.5 justify-end">
          <button type="button" onClick={dismiss} disabled={busy}
            className="px-2 py-1 rounded text-[11px] bg-surface border border-border text-content-muted disabled:opacity-40">
            Cancel
          </button>
          <button type="button" onClick={submit} disabled={busy || !text.trim()}
            className="px-3 py-1 rounded text-[11px] bg-gradient-primary text-gray-950 font-semibold disabled:opacity-40">
            {busy ? '…' : 'OK'}
          </button>
        </div>
      </div>
    </div>,
    document.body,
  );
}

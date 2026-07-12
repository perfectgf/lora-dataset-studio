/** Small bubble anchored on a generated tile: shows the core creative prompt,
 *  lets the user edit it, and regenerates that tile with the edit on OK. The
 *  identity guard is re-applied server-side, so this is only the creative half
 *  (expression / scene / lighting) — never the face lock. Presentational only:
 *  the parent wires onSubmit (which calls regenerate) and onClose. */
import { useEffect, useRef, useState } from 'react';

export default function PromptEditPopover({ initialPrompt = '', onSubmit, onClose }) {
  const [text, setText] = useState(initialPrompt);
  const areaRef = useRef(null);
  // Focus the textarea on open and select all so a full rewrite is one keystroke away.
  useEffect(() => {
    const el = areaRef.current;
    if (el) { el.focus(); el.select(); }
  }, []);
  const submit = () => {
    const t = text.trim();
    if (t) { onSubmit(t); onClose(); }
  };
  return (
    // Backdrop closes on outside click; stopPropagation on the bubble keeps clicks
    // inside from bubbling to the tile (which would trigger inspect/select).
    <div className="absolute inset-0 z-30 flex items-center justify-center bg-black/50 p-2"
      onClick={(e) => { e.stopPropagation(); onClose(); }}>
      <div className="w-full max-w-[15rem] rounded-lg border border-border bg-surface p-2 shadow-xl flex flex-col gap-2"
        onClick={(e) => e.stopPropagation()}>
        <span className="text-[0.625rem] uppercase text-content-muted">Edit prompt &amp; regenerate</span>
        <textarea ref={areaRef} value={text} onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Escape') { e.preventDefault(); onClose(); }
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); submit(); }
          }}
          rows={4} placeholder="describe the shot (the face is kept automatically)…"
          aria-label="Edit the generation prompt"
          className="text-[11px] bg-app/60 border border-border rounded p-1.5 text-content resize-none" />
        <div className="flex gap-1.5 justify-end">
          <button type="button" onClick={onClose}
            className="px-2 py-1 rounded text-[11px] bg-surface border border-border text-content-muted">
            Cancel
          </button>
          <button type="button" onClick={submit} disabled={!text.trim()}
            className="px-3 py-1 rounded text-[11px] bg-gradient-primary text-white font-semibold disabled:opacity-40">
            OK
          </button>
        </div>
      </div>
    </div>
  );
}

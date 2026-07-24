/** ✦ Edit the reference photo with a prompt (+ optional extra reference images),
 * via ChatGPT or Nano Banana.
 *
 * The edit runs as a SERVER background job — slow (1-3 min) and PAID, so it must
 * not ride the client's fetch (a backgrounded mobile tab would kill it and lose
 * the paid result). This modal is DRIVEN BY SERVER STATE (`referenceEdit` from the
 * dataset payload): running → spinner, ready → Before/After, failed → error. That
 * makes it restore correctly after a tab sleep, a reload, or a reopen — there is
 * no long client request to lose.
 *
 * Keep promotes the candidate (atomic on the server); Discard deletes it; the ✕
 * just closes and LEAVES the job running (rediscovered on reopen). Modal idiom
 * mirrors CropModal: role=dialog, Escape closes, initial focus. */
import { useEffect, useRef, useState } from 'react';
import { EDIT_ENGINES, editBlockedReason, batchLiveNote, editPhase } from './referenceEdit';

const ENGINE_LABEL = { chatgpt: 'ChatGPT', nanobanana: 'Nano Banana' };
const MAX_EDIT_REFS = 3;

export default function ReferenceEditModal({ datasetId, refFilename, nonce = 0,
                                             defaultEngine = 'chatgpt', liveActivity = null,
                                             referenceEdit = null,
                                             onEdit, onKeep, onDiscard, onClose }) {
  const [prompt, setPrompt] = useState('');
  const [engine, setEngine] = useState(EDIT_ENGINES.includes(defaultEngine) ? defaultEngine : 'chatgpt');
  const [editRefs, setEditRefs] = useState([]);            // transient File[]
  const [starting, setStarting] = useState(false);         // bridges POST -> server 'running'
  const [busyAction, setBusyAction] = useState(false);     // keep/discard in flight
  const inpRef = useRef(null);
  const promptRef = useRef(null);

  const serverPhase = editPhase(referenceEdit);            // idle | running | ready | failed
  const phase = (starting && serverPhase === 'idle') ? 'running' : serverPhase;
  // Once the server reflects the job (running/ready/failed), drop the local bridge.
  useEffect(() => { if (serverPhase !== 'idle') setStarting(false); }, [serverPhase]);

  const imgUrl = (fn) => `/api/dataset/${datasetId}/img/${encodeURIComponent(fn)}${nonce ? `?v=${nonce}` : ''}`;
  const beforeUrl = imgUrl(refFilename);
  const afterUrl = referenceEdit?.candidate_filename
    ? `/api/dataset/${datasetId}/img/${encodeURIComponent(referenceEdit.candidate_filename)}`
    : null;

  const busy = starting || busyAction;
  // Escape / ✕ close the modal but NEVER discard — a running or ready job is left
  // on the server and rediscovered on reopen.
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape' && !busy) onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose, busy]);
  useEffect(() => { if (phase === 'idle') promptRef.current?.focus(); }, [phase]);

  const blocked = editBlockedReason(prompt, engine);
  const liveNote = batchLiveNote(liveActivity);

  const addRefs = (files) => {
    const list = Array.from(files || []).filter((f) => f && f.type.startsWith('image/'));
    setEditRefs((cur) => [...cur, ...list].slice(0, MAX_EDIT_REFS));
  };

  const runEdit = async () => {
    if (blocked) return;
    setStarting(true);
    const ok = await onEdit(prompt, engine, editRefs);
    if (!ok) setStarting(false);          // start failed → stay on the form
  };

  const keep = async () => {
    setBusyAction(true);
    const ok = await onKeep();
    if (ok) onClose(); else setBusyAction(false);
  };

  const discard = async (close) => {
    setBusyAction(true);
    await onDiscard();
    setBusyAction(false);
    if (close) onClose();
  };

  return (
    <div role="dialog" aria-modal="true" aria-label="Edit reference photo"
      className="fixed inset-0 z-[9995] bg-black/85 flex flex-col p-3 sm:p-4 overflow-y-auto">
      <div className="w-full max-w-3xl mx-auto flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <h2 className="text-content text-base font-semibold">✦ Edit reference</h2>
          <button type="button" onClick={onClose} disabled={busy}
            aria-label="Close" className="px-2 py-1 rounded-lg bg-surface text-content text-sm disabled:opacity-40">✕</button>
        </div>

        {liveNote && (
          <p className="text-[0.6875rem] text-amber-300 bg-amber-500/10 border border-amber-500/30 rounded-lg px-2.5 py-1.5">
            {liveNote}
          </p>
        )}

        {phase === 'running' ? (
          <div className="flex flex-col items-center gap-3 py-8">
            <span className="inline-block w-8 h-8 border-2 border-indigo-400/40 border-t-indigo-400 rounded-full animate-spin" aria-hidden />
            <p className="text-content text-sm">Editing the reference… a “high” render can take 1–3 minutes.</p>
            <p className="text-content-muted text-[0.6875rem] text-center">
              This runs on the server — you can close this tab and come back; the Before/After will be here.
            </p>
            <button type="button" onClick={() => discard(true)} disabled={busy}
              className="px-4 py-2 rounded-lg bg-surface text-content text-sm disabled:opacity-40">
              Cancel edit
            </button>
          </div>
        ) : phase === 'ready' ? (
          <>
            {/* Before / After — side by side on desktop, stacked on mobile. */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <figure className="flex flex-col gap-1">
                <figcaption className="text-content-subtle text-xs">Before</figcaption>
                <img src={beforeUrl} alt="current reference"
                  className="w-full rounded-lg bg-black object-contain max-h-[45vh]" />
              </figure>
              <figure className="flex flex-col gap-1">
                <figcaption className="text-sky-300 text-xs">After (candidate)</figcaption>
                {afterUrl && <img src={afterUrl} alt="edited candidate"
                  className="w-full rounded-lg bg-black object-contain max-h-[45vh]" />}
              </figure>
            </div>
            <p className="text-[0.6875rem] text-content-muted">
              Keep replaces the reference — this can’t be undone after you Keep it. It changes
              only future variations, not images already generated. Discard doesn’t refund the edit.
            </p>
            <div className="flex gap-2 justify-end flex-wrap">
              <button type="button" onClick={() => discard(false)} disabled={busy}
                className="mr-auto px-4 py-2 rounded-lg bg-surface text-content text-sm disabled:opacity-40">
                Try another prompt
              </button>
              <button type="button" onClick={() => discard(true)} disabled={busy}
                className="px-4 py-2 rounded-lg bg-surface text-content text-sm disabled:opacity-40">
                Discard
              </button>
              <button type="button" onClick={keep} disabled={busy}
                className="px-4 py-2 rounded-lg bg-gradient-primary text-white text-sm font-semibold disabled:opacity-40">
                {busyAction ? 'Keeping…' : 'Keep'}
              </button>
            </div>
          </>
        ) : (
          <>
            <div className="flex flex-col gap-1">
              <img src={beforeUrl} alt="current reference"
                className="w-32 h-32 rounded-lg bg-black object-cover self-start" />
            </div>

            <label className="flex flex-col gap-1">
              <span className="text-content-subtle text-xs">What should change?</span>
              <textarea ref={promptRef} value={prompt} onChange={(e) => setPrompt(e.target.value)}
                rows={3} disabled={busy}
                placeholder="e.g. plain studio-grey background, add glasses, warmer lighting"
                className="w-full rounded-lg bg-surface-raised border border-border text-content text-sm p-2 resize-y disabled:opacity-40" />
            </label>

            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-content-subtle text-xs">Engine</span>
              {EDIT_ENGINES.map((e) => (
                <button key={e} type="button" onClick={() => setEngine(e)} disabled={busy}
                  aria-pressed={engine === e}
                  className={`px-2.5 py-1 rounded-lg text-xs font-semibold disabled:opacity-40 ${engine === e
                    ? 'bg-indigo-500 text-white' : 'bg-surface-raised text-content-muted hover:bg-surface'}`}>
                  {ENGINE_LABEL[e]}
                </button>
              ))}
            </div>

            {/* Optional extra reference images — transient inputs to THIS edit only,
                never saved as the dataset's extra refs. */}
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-content-subtle text-xs">Add reference images (optional)</span>
              {editRefs.map((f, i) => (
                <div key={i} className="relative w-12 h-12 rounded-lg overflow-hidden bg-black shrink-0">
                  <img src={URL.createObjectURL(f)} alt="edit reference" className="w-full h-full object-cover" />
                  <button type="button" disabled={busy}
                    onClick={() => setEditRefs((cur) => cur.filter((_, j) => j !== i))}
                    aria-label="Remove this reference image"
                    className="absolute top-0 right-0 w-4 h-4 flex items-center justify-center rounded-bl bg-black/70 text-white text-[0.625rem] leading-none disabled:opacity-40">✕</button>
                </div>
              ))}
              {editRefs.length < MAX_EDIT_REFS && (
                <button type="button" onClick={() => inpRef.current?.click()} disabled={busy}
                  aria-label="Add a reference image for the edit"
                  className="w-12 h-12 rounded-lg border border-dashed border-border-strong text-content-muted text-lg leading-none disabled:opacity-40">+</button>
              )}
              <input ref={inpRef} type="file" accept="image/*" multiple className="hidden" disabled={busy}
                onChange={(e) => { addRefs(e.target.files); e.target.value = ''; }} />
            </div>

            {phase === 'failed' && referenceEdit?.error && (
              <p className="text-[0.6875rem] text-red-300 bg-red-500/10 border border-red-500/30 rounded-lg px-2.5 py-1.5">
                {referenceEdit.error}
              </p>
            )}
            <p className="text-[0.6875rem] text-content-muted">
              Each edit is a paid API call — Discard doesn’t refund it. A “high” render can take 1–3 minutes.
            </p>

            <div className="flex gap-2 justify-end">
              <button type="button" onClick={onClose} disabled={busy}
                className="px-4 py-2 rounded-lg bg-surface text-content text-sm disabled:opacity-40">Cancel</button>
              <button type="button" onClick={runEdit} disabled={busy || !!blocked}
                title={blocked || undefined}
                className="px-4 py-2 rounded-lg bg-gradient-primary text-white text-sm font-semibold disabled:opacity-40">
                {starting ? 'Starting…' : 'Generate edit'}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

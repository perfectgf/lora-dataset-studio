import { useEffect, useRef, useState } from 'react';
import { apiFetch, postJson } from '../../api/fetchClient';
import { useToast } from '../common/Toast';
import KleinImproveNote, {
  flushImproveSettings, whenImproveSettingsSettled,
} from '../dataset/KleinImproveNote';
import ImproveResultView from './ImproveResultView';

/* ✨ The improve MODAL — settings on demand, result in place.

   The Klein settings block (instruction, subject-type warning, model, LoRA
   preset, output size) used to sit INLINE under the improve buttons on every
   surface, permanently — a screenful of prose and dials read once and paid
   for on every open (user-reported, with a phone screenshot of the actions
   sheet mostly being Klein settings). Now the ✨ button opens THIS dialog:
   the same settings component, a Generate button, and the result shown right
   here when it lands — or in the surface's own gallery if you left, exactly
   as before (the dataset's grid already refreshes itself).

   ONE modal for every host — Gallery, Canvas, checkpoint galleries, the
   Studio viewer, the dataset lightbox. The hosts differ by ONE word: which
   table their image lives in (`host`), which picks the improve route and the
   status poll. Same lesson as the camera picker: reproducible logic, not one
   re-implementation per screen.

   Closing while generating abandons NOTHING: the job is queued server-side,
   the toast says where the result will land, and the row arrives through the
   surface's ordinary refresh. */

const POLL_MS = 4000;
const ROUTES = {
  library: {
    post: (id) => `/api/canvas/image/${id}/improve`,
    poll: (id) => `/api/canvas/image/${id}/status`,
    dest: 'the Gallery feed',
  },
  dataset: {
    post: (id) => `/api/dataset/image/${id}/improve`,
    poll: (id) => `/api/dataset/image/${id}/status`,
    dest: 'this dataset’s Images (as a pending candidate)',
  },
};

export default function ImproveModal({ img, host = 'library', datasetId = null,
  subjectType = '', onClose }) {
  const toast = useToast();
  const [phase, setPhase] = useState('settings');   // settings | generating | done | failed
  const [candidateId, setCandidateId] = useState(null);
  const [result, setResult] = useState(null);       // {id, url}
  const [error, setError] = useState(null);
  const closeRef = useRef(null);
  const routes = ROUTES[host] || ROUTES.library;

  // Focus moves INTO the dialog on open — the camera picker's lesson: without
  // it the keys stay with whatever host sits underneath.
  useEffect(() => { closeRef.current?.focus(); }, []);

  // The 4-second heartbeat, only while a queued candidate is unresolved.
  // Transient poll misses are not failures; unmount stops the poll and never
  // the render.
  useEffect(() => {
    if (phase !== 'generating' || candidateId == null) return undefined;
    let alive = true;
    const tick = async () => {
      try {
        const d = await apiFetch(routes.poll(candidateId));
        if (!alive) return;
        if (d.status === 'done' && d.url) {
          setResult({ id: d.id, url: d.url });
          setPhase('done');
        } else if (d.status === 'failed') {
          setError(d.error || 'The render failed.');
          setPhase('failed');
        }
      } catch { /* transient — the next tick answers */ }
    };
    const h = setInterval(tick, POLL_MS);
    tick();
    return () => { alive = false; clearInterval(h); };
  }, [phase, candidateId, routes]);

  const generate = async (e) => {
    e.stopPropagation();
    setError(null);
    /* A candidate from a FAILED attempt must not survive into the next one: the
       poll effect restarts on 'settings' → 'generating' and ticks immediately,
       so a stale id answers 'failed' (it always will) and wins the race against
       the POST — the dialog then shows an error for a run that was queued fine,
       and never reaches the result. */
    setCandidateId(null);
    setResult(null);
    setPhase('generating');
    try {
      /* The dials are read SERVER-SIDE at enqueue time (the chain is resolved
         from config.json), so a slider dropped a beat ago must have landed
         before this POST — otherwise the render uses the previous value while
         the panel shows the new one. Flush what is coalescing, wait for what is
         in flight; a settings write that FAILED does not block the run (the
         panel reports it on its own). */
      flushImproveSettings();
      await whenImproveSettingsSettled();
      const d = await postJson(routes.post(img.id), { engine: 'klein' });
      if (!d?.ok) throw new Error(d?.error || 'Could not queue the improve');
      setCandidateId(d.candidate_id);
    } catch (err) {
      setPhase('failed');
      setError(err?.message || 'Could not queue the improve');
    }
  };

  const close = (e) => {
    e?.stopPropagation?.();
    if (phase === 'generating') {
      // Leaving is fine and the sentence says why — the surface's own refresh
      // delivers the result; nothing is lost with the dialog.
      toast.success(`✨ Still rendering — the result lands in ${routes.dest}.`);
    }
    onClose?.();
  };

  return (
    <div data-probe-layer className="fixed inset-0 z-[9998] flex items-center justify-center bg-black/80 p-3 sm:p-6"
      role="dialog" aria-modal="true" aria-label="Upscale & improve"
      /* The picker's two keyboard lessons, applied to this layer: keys pressed
         inside belong to the dialog, Escape peels IT (one layer), and keys
         with focus elsewhere reach the hosts' own stand-down branches. */
      onClick={(e) => { e.stopPropagation(); if (e.target === e.currentTarget) close(e); }}
      onKeyDown={(e) => { e.stopPropagation(); if (e.key === 'Escape') close(e); }}>
      {/* In the result phase the dialog takes the height it can: the picture IS
          the content, and a body that scrolls under a `touch-none` frame hides
          the bottom of the render behind a gesture that was taken away. */}
      <div className={`flex w-full max-w-2xl flex-col overflow-hidden rounded-xl border border-white/10 bg-surface-overlay shadow-2xl ${phase === 'done' ? 'h-full max-h-full' : 'max-h-full'}`}>
        <header className="flex items-start gap-3 border-b border-white/10 px-4 py-3">
          <span aria-hidden className="mt-0.5">✨</span>
          <div className="min-w-0 flex-1">
            <h2 className="font-sans text-base font-semibold text-gray-100">Upscale &amp; improve via Klein</h2>
            <p className="mt-0.5 text-[0.78rem] leading-snug text-gray-400">
              {phase === 'done'
                ? 'Done — the result below also lives in ' + routes.dest + '.'
                : 'Check the instruction and the dials, then Generate. The result shows here — or in ' + routes.dest + ' if you leave.'}
            </p>
          </div>
          <button type="button" onClick={close} aria-label="Close" ref={closeRef}
            className="min-h-10 lg:min-h-0 -mr-1 rounded-lg px-2 text-gray-400 hover:bg-white/5 hover:text-gray-200">
            ✕
          </button>
        </header>

        <div className={`min-h-0 flex-1 p-4 ${phase === 'done' ? 'flex overflow-hidden' : 'overflow-y-auto'}`}>
          {phase === 'settings' && (
            <KleinImproveNote subjectType={subjectType} datasetId={datasetId} className="w-full" />
          )}
          {phase === 'generating' && (
            <div className="flex flex-col items-center gap-3 py-8 text-center">
              <span aria-hidden className="animate-pulse text-2xl">✨</span>
              <p className="m-0 max-w-md text-[0.85rem] text-gray-300" role="status">
                Generating… you can close this dialog — the render keeps going and
                the result lands in {routes.dest}.
              </p>
            </div>
          )}
          {/* Zoomable: an upscale is judged on detail that fit-to-dialog hides.
              Same engine as the lightbox — see ImproveResultView. */}
          {phase === 'done' && result && (
            <ImproveResultView url={result.url} />
          )}
          {phase === 'failed' && (
            <p role="alert" className="m-0 rounded-md border border-amber-400/40 bg-amber-500/10 px-3 py-2 text-[0.85rem] text-amber-100">
              {error}
            </p>
          )}
        </div>

        <footer className="flex flex-wrap items-center justify-end gap-3 border-t border-white/10 px-4 py-3">
          {phase === 'failed' && (
            <button type="button"
              onClick={(e) => {
                e.stopPropagation();
                // With the id left behind, the next Generate polls the OLD
                // candidate and flips straight back to this same error.
                setCandidateId(null); setResult(null); setError(null); setPhase('settings');
              }}
              className="min-h-10 lg:min-h-0 rounded-lg border border-white/10 px-3 py-1.5 text-[0.78rem] text-gray-300 hover:border-white/25">
              Back to settings
            </button>
          )}
          <button type="button" onClick={close}
            className="min-h-10 lg:min-h-0 rounded-lg border border-white/10 px-3 py-1.5 text-[0.78rem] text-gray-300 hover:border-white/25">
            {phase === 'done' ? 'Done' : 'Close'}
          </button>
          {phase === 'settings' && (
            <button type="button" data-testid="improve-modal-generate" onClick={generate}
              className="min-h-10 lg:min-h-0 rounded-lg bg-gradient-primary px-4 py-1.5 text-[0.8rem] font-semibold text-gray-950">
              ✨ Generate
            </button>
          )}
        </footer>
      </div>
    </div>
  );
}

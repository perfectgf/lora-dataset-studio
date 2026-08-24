/** ✦ Edit the reference photo with a prompt (+ optional extra reference images),
 * via any engine the install can run — the list is DERIVED from EDIT_ENGINES and
 * from `engineOptions`, never spelled out here.
 *
 * The edit runs as a SERVER background job — slow (1-3 min), and on an API engine
 * PAID, so it must not ride the client's fetch (a backgrounded mobile tab would
 * kill it and lose the paid result). This modal is DRIVEN BY SERVER STATE
 * (`referenceEdit` from the dataset payload): running → spinner, ready →
 * Before/After, failed → error. That makes it restore correctly after a tab
 * sleep, a reload, or a reopen — there is no long client request to lose.
 *
 * TWO LANES, ONE FORM. The local engines (Klein, Krea 2 Edit) render on the
 * user's own ComfyUI: free, private, and therefore the sane way to try five
 * prompts. Three things then differ and every one of them is said BEFORE the
 * click, not after the render: what it costs (editCostNote), which references it
 * consumes (editRefNote / acceptsExtraEditRefs), and whether this install can run
 * it at all (the `blocked` reason on each option).
 *
 * Keep promotes the candidate (atomic on the server); Discard deletes it; the ✕
 * just closes and LEAVES the job running (rediscovered on reopen). Modal idiom
 * mirrors CropModal: role=dialog, Escape closes, initial focus. */
import { useEffect, useMemo, useRef, useState } from 'react';
import { useFocusTrap } from '../../hooks/useFocusTrap';
import KleinModelSetting from '../shared/KleinModelSetting';
import {
  EDIT_ENGINES, LOCAL_ENGINES, batchLiveNote, editPhase,
  editEngineOptions, editCostNote, editKeepNote, editRefNote,
  acceptsExtraEditRefs, acceptsExtraEditRefsForBatch, editBatchBlockedReason,
  referenceEditCandidates,
  ENGINE_LABELS, maxEditRefsForBatch,
} from './referenceEdit';

export default function ReferenceEditModal({ datasetId, refFilename, nonce = 0,
                                             defaultEngine = 'chatgpt', liveActivity = null,
                                             referenceEdit = null, datasetExtraCount = 0,
                                             comfyuiConfigured = false, engineAvailable = null,
                                             engineReason = null,
                                             onEdit, onRetry = null, canRetry = false,
                                             onKeep, onDiscard, onClose }) {
  const [prompt, setPrompt] = useState('');
  const initialEngine = EDIT_ENGINES.includes(defaultEngine) ? defaultEngine : 'chatgpt';
  const [engines, setEngines] = useState([initialEngine]);
  const [editRefs, setEditRefs] = useState([]);            // transient File[]
  const [starting, setStarting] = useState(false);         // bridges POST -> server 'running'
  const [busyAction, setBusyAction] = useState(null);      // engine id | discard
  const inpRef = useRef(null);
  const promptRef = useRef(null);
  const dialogRef = useRef(null);
  useFocusTrap(dialogRef);

  const serverPhase = editPhase(referenceEdit);            // idle | running | ready | failed
  const phase = starting ? 'running' : serverPhase;
  // Once the server reflects the job (running/ready/failed), drop the local bridge.
  useEffect(() => { if (serverPhase !== 'idle') setStarting(false); },
    [serverPhase, referenceEdit?.started_at]);

  const imgUrl = (fn) => `/api/dataset/${datasetId}/img/${encodeURIComponent(fn)}${nonce ? `?v=${nonce}` : ''}`;
  const beforeUrl = imgUrl(refFilename);
  const candidates = referenceEditCandidates(referenceEdit);
  const readyCandidates = candidates.filter(
    (candidate) => candidate.status === 'ready' && candidate.candidate_filename);
  const failedCandidates = candidates.filter((candidate) => candidate.status === 'failed');

  const busy = starting || busyAction !== null;
  // Escape / ✕ close the modal but NEVER discard — a running or ready job is left
  // on the server and rediscovered on reopen.
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape' && !busy) onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose, busy]);
  useEffect(() => { if (phase === 'idle') promptRef.current?.focus(); }, [phase]);

  // Which engines this install can offer, and why not when it can't. Computed
  // from the SAME capabilities the generation panel reads, so a missing Krea node
  // pack is explained here in the words it is explained there.
  const options = useMemo(() => editEngineOptions({
    comfyuiConfigured, available: engineAvailable || {}, reasonFor: engineReason,
  }), [comfyuiConfigured, engineAvailable, engineReason]);
  // A stored default pointing at an engine this install dropped (ComfyUI removed,
  // Klein was the primary) must not leave the form on a phantom selection.
  useEffect(() => {
    if (phase === 'running' || phase === 'ready') return;
    const visible = engines.filter((engine) => options.some((o) => o.engine === engine));
    if (visible.length !== engines.length) {
      const fallback = options.find((o) => o.usable) || options[0];
      setEngines(visible.length ? visible : (fallback ? [fallback.engine] : []));
    }
  }, [engines, phase, options]);

  // A reopened failed batch should show the exact engines that produced its
  // errors. The transient files themselves remain session-only in useDataset.
  useEffect(() => {
    const recorded = Array.isArray(referenceEdit?.engines)
      ? referenceEdit.engines
      : [referenceEdit?.engine].filter(Boolean);
    if (recorded.length) setEngines([...new Set(recorded)]);
    // started_at = l'identite du LOT : re-seeder sur chaque mutation de champ
    // ecraserait la selection en cours de l'utilisateur.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [referenceEdit?.started_at]);

  const blocked = editBatchBlockedReason(prompt, engines, options);
  const selectedBlocked = options.filter(
    (option) => engines.includes(option.engine) && option.blocked);
  const liveNote = batchLiveNote(liveActivity);
  const selectedLocalEngines = engines.filter((engine) => LOCAL_ENGINES.includes(engine));
  const localRefNotes = selectedLocalEngines
    .map((engine) => editRefNote(engine, { datasetExtraCount }))
    .filter(Boolean);
  const canAddRefs = acceptsExtraEditRefsForBatch(engines);
  // The cap follows the SELECTION, because engines read a different number of
  // these: Krea has room for one, the API engines for three. Deselecting the
  // generous one has to drop what no longer fits, or the request would carry
  // images the server is about to refuse.
  const maxRefs = maxEditRefsForBatch(engines);
  useEffect(() => {
    setEditRefs((cur) => (cur.length <= maxRefs ? cur : cur.slice(0, maxRefs)));
  }, [maxRefs]);

  const addRefs = (files) => {
    const list = Array.from(files || []).filter((f) => f && f.type.startsWith('image/'));
    setEditRefs((cur) => [...cur, ...list].slice(0, maxRefs));
  };

  const runEdit = async () => {
    if (blocked) return;
    setStarting(true);
    const ok = await onEdit(prompt, engines, canAddRefs ? editRefs : []);
    if (!ok) setStarting(false);          // start failed → stay on the form
  };

  const toggleEngine = (engine) => {
    setEngines((current) => current.includes(engine)
      ? current.filter((selected) => selected !== engine)
      : [...current, engine]);
  };

  const canRetryExact = Boolean(canRetry && typeof onRetry === 'function');
  const retryEdit = async () => {
    if (!canRetryExact) return;
    setStarting(true);
    const ok = await onRetry();
    if (!ok) setStarting(false);
  };

  const keep = async (engine) => {
    setBusyAction(engine);
    const ok = await onKeep(engine, referenceEdit?.batch_id || null);
    if (ok) onClose(); else setBusyAction(null);
  };

  const discard = async (close) => {
    setBusyAction('discard');
    await onDiscard();
    setBusyAction(null);
    if (close) onClose();
  };

  return (
    <div ref={dialogRef} role="dialog" aria-modal="true" aria-label="Edit reference photo"
      className="fixed inset-0 z-[9995] bg-black/80 backdrop-blur-sm flex flex-col p-3 sm:p-4 overflow-y-auto">
      {/* Opaque card — DO NOT remove. The form has transparent gaps; laid straight
          on the dim overlay, the page reads through. bg-surface is only 4% alpha
          (--surface-alpha) — the opaque modal-panel token is bg-surface-overlay,
          the one every other modal here uses. (Regressed once when this file was
          rewritten; kept explicit so it survives the next rewrite.) */}
      <div className="w-full max-w-3xl mx-auto my-auto flex flex-col gap-3
                      bg-surface-overlay border border-border rounded-2xl shadow-2xl p-4 sm:p-5">
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
          <div className="flex flex-col items-center gap-3 py-6">
            <span className="inline-block w-8 h-8 border-2 border-indigo-400/40 border-t-indigo-400 rounded-full animate-spin" aria-hidden />
            <p className="text-content text-sm">Editing the reference with the selected engines…</p>
            <div className="w-full grid grid-cols-1 sm:grid-cols-2 gap-2" aria-live="polite">
              {candidates.length ? candidates.map((candidate) => {
                const label = ENGINE_LABELS[candidate.engine] || candidate.engine;
                return (
                  <div key={candidate.engine}
                    className="rounded-lg bg-surface-raised border border-border px-3 py-2">
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-content text-xs font-semibold">{label}</span>
                      <span className={`text-[0.6875rem] ${candidate.status === 'failed'
                        ? 'text-red-300'
                        : candidate.status === 'ready' ? 'text-emerald-300' : 'text-indigo-300'}`}>
                        {candidate.status === 'ready' ? 'Ready' : candidate.status === 'failed' ? 'Failed' : 'Running'}
                      </span>
                    </div>
                    {candidate.error && (
                      <p className="text-red-300 text-[0.6875rem] mt-1">{candidate.error}</p>
                    )}
                  </div>
                );
              }) : (
                <p className="text-content-muted text-xs text-center sm:col-span-2">Starting engines…</p>
              )}
            </div>
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
            {/* One Before plus every successful candidate. */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
              <figure className="flex flex-col gap-1">
                <figcaption className="text-content-subtle text-xs">Before</figcaption>
                <img src={beforeUrl} alt="current reference"
                  className="w-full rounded-lg bg-black object-contain max-h-[45vh]" />
              </figure>
              {readyCandidates.map((candidate) => {
                const label = ENGINE_LABELS[candidate.engine] || candidate.engine;
                const afterUrl = `/api/dataset/${datasetId}/img/${encodeURIComponent(candidate.candidate_filename)}`;
                return (
                  <figure key={candidate.engine}
                    className="flex flex-col gap-2 rounded-lg border border-border p-2 bg-surface-raised">
                    <figcaption className="flex items-center justify-between gap-2 text-xs">
                      <span className="text-sky-300">{label}</span>
                      <span className="text-emerald-300">Ready</span>
                    </figcaption>
                    <img src={afterUrl} alt={`edited candidate from ${label}`}
                      className="w-full rounded-lg bg-black object-contain max-h-[45vh]" />
                    <p className="text-[0.6875rem] text-content-muted">
                      {editKeepNote(candidate.engine)}
                    </p>
                    <button type="button" onClick={() => keep(candidate.engine)} disabled={busy}
                      className="mt-auto px-4 py-2 rounded-lg bg-gradient-primary text-white text-sm font-semibold disabled:opacity-40">
                      {busyAction === candidate.engine ? 'Keeping…' : `Keep ${label}`}
                    </button>
                  </figure>
                );
              })}
            </div>
            {failedCandidates.length > 0 && (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2" aria-label="Failed edit engines">
                {failedCandidates.map((candidate) => {
                  const label = ENGINE_LABELS[candidate.engine] || candidate.engine;
                  return (
                    <div key={candidate.engine}
                      className="rounded-lg bg-red-500/10 border border-red-500/30 px-3 py-2">
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-content text-xs font-semibold">{label}</span>
                        <span className="text-red-300 text-[0.6875rem]">Failed</span>
                      </div>
                      <p className="text-red-300 text-[0.6875rem] mt-1">
                        {candidate.error || 'This engine did not produce a candidate.'}
                      </p>
                    </div>
                  );
                })}
              </div>
            )}
            {!canRetryExact && (
              <p className="text-[0.6875rem] text-content-subtle">
                Retry keeps temporary reference files only for this browser session. After reopening
                this page, use Try another prompt to attach them again.
              </p>
            )}
            <div className="flex gap-2 justify-end flex-wrap">
              <button type="button" onClick={() => discard(false)} disabled={busy}
                className="mr-auto px-4 py-2 rounded-lg bg-surface text-content text-sm disabled:opacity-40">
                Try another prompt
              </button>
              <button type="button" onClick={retryEdit} disabled={busy || !canRetryExact}
                title={canRetryExact ? undefined : 'The original temporary references are no longer available'}
                className="px-4 py-2 rounded-lg bg-surface text-content text-sm disabled:opacity-40">
                ↻ Retry same edit
              </button>
              <button type="button" onClick={() => discard(true)} disabled={busy}
                className="px-4 py-2 rounded-lg bg-surface text-content text-sm disabled:opacity-40">
                Discard
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

            {/* Engine pills. `flex-wrap` + `min-w-0` is what keeps five of them on
                a 400px screen: they wrap onto a second and third row instead of
                stretching the modal past the viewport. */}
            <div role="group" aria-label="Edit engines"
              className="flex items-center gap-1.5 flex-wrap min-w-0">
              <span className="text-content-subtle text-xs w-full sm:w-auto">
                Engines (select one or more)
              </span>
              {options.map((o) => (
                <button key={o.engine} type="button" onClick={() => toggleEngine(o.engine)} disabled={busy}
                  aria-pressed={engines.includes(o.engine)}
                  title={o.blocked || undefined}
                  className={`px-2.5 py-1 rounded-lg text-xs font-semibold disabled:opacity-40 ${engines.includes(o.engine)
                    ? 'bg-indigo-500 text-white'
                    : o.usable
                      ? 'bg-surface-raised text-content-muted hover:bg-surface'
                      : 'bg-surface-raised text-content-subtle line-through decoration-1'}`}>
                  {o.label}
                </button>
              ))}
            </div>
            {engines.length === 0 && (
              <p role="alert"
                className="text-[0.6875rem] text-amber-300 bg-amber-500/10 border border-amber-500/30 rounded-lg px-2.5 py-1.5">
                Select at least one engine.
              </p>
            )}

            {/* An unavailable LOCAL engine is still selectable — picking it is how
                you find out WHY, and the reason names the one action that fixes
                it. Greying it out silently was the failure mode this replaces. */}
            {selectedBlocked.map((option) => (
              <p key={option.engine}
                className="text-[0.6875rem] text-amber-300 bg-amber-500/10 border border-amber-500/30 rounded-lg px-2.5 py-1.5">
                {option.label}: {option.blocked}
              </p>
            ))}

            {localRefNotes.map((note) => (
              <p key={note} className="text-[0.6875rem] text-content-muted">{note}</p>
            ))}
            {/* Only worth saying when the selection actually splits: one engine
                reads these bytes and another does not. Krea reads them now, so
                the old blanket "API engines only" would have been false. */}
            {canAddRefs && selectedLocalEngines.some((e) => !acceptsExtraEditRefs(e)) && (
              <p className="text-[0.6875rem] text-sky-300 bg-sky-500/10 border border-sky-500/30 rounded-lg px-2.5 py-1.5">
                Images added below go to the engines that read them. The rest use the
                reference support described above.
              </p>
            )}

            {/* Optional extra reference images — transient inputs to THIS edit only,
                never saved as the dataset's extra refs. Shown whenever a SELECTED
                engine reads them: every API engine, and Krea for its one `_b`
                slot (a different subject to compose with). Klein does not — it
                reads the dataset's angles from the reference card instead. The
                count is capped by the most generous consumer in the selection. */}
            {canAddRefs && (
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
              {editRefs.length < maxRefs && (
                <button type="button" onClick={() => inpRef.current?.click()} disabled={busy}
                  aria-label="Add a reference image for the edit"
                  className="w-12 h-12 rounded-lg border border-dashed border-border-strong text-content-muted text-lg leading-none disabled:opacity-40">+</button>
              )}
              <input ref={inpRef} type="file" accept="image/*" multiple className="hidden" disabled={busy}
                onChange={(e) => { addRefs(e.target.files); e.target.value = ''; }} />
            </div>
            )}

            {phase === 'failed' && (failedCandidates.length > 0 || referenceEdit?.error) && (
              <div className="flex flex-col gap-2 text-[0.6875rem] bg-red-500/10 border border-red-500/30 rounded-lg px-2.5 py-1.5">
                {failedCandidates.length > 0 ? failedCandidates.map((candidate) => {
                  const label = ENGINE_LABELS[candidate.engine] || candidate.engine;
                  return (
                    <div key={candidate.engine}>
                      <p className="text-content font-semibold">{label} — Failed</p>
                      <p className="text-red-300">
                        {candidate.error || 'This engine did not produce a candidate.'}
                      </p>
                    </div>
                  );
                }) : <p className="text-red-300">{referenceEdit.error}</p>}
              </div>
            )}
            <p className="text-[0.6875rem] text-content-muted">{editCostNote(engines)}</p>
            {/* The reference is what the whole dataset is anchored on, and this
                lane runs on the dataset's Klein model like every other one — so
                say which, and let it be changed from here. Klein only: the other
                local engine (Krea) resolves a global base model, and the API
                engines have no local model at all. */}
            {engines.includes('klein') && <KleinModelSetting datasetId={datasetId} />}

            <div className="flex gap-2 justify-end">
              <button type="button" onClick={onClose} disabled={busy}
                className="px-4 py-2 rounded-lg bg-surface text-content text-sm disabled:opacity-40">Cancel</button>
              {phase === 'failed' && (
                <button type="button" onClick={retryEdit} disabled={busy || !canRetryExact}
                  title={canRetryExact ? undefined : 'The original temporary references are no longer available'}
                  className="px-4 py-2 rounded-lg bg-surface text-content text-sm disabled:opacity-40">
                  ↻ Retry same edit
                </button>
              )}
              <button type="button" onClick={runEdit} disabled={busy || !!blocked}
                title={blocked || undefined}
                className="px-4 py-2 rounded-lg bg-gradient-primary text-white text-sm font-semibold disabled:opacity-40">
                {starting
                  ? 'Starting…'
                  : `Generate ${engines.length || 0} edit${engines.length === 1 ? '' : 's'}`}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

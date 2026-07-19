import { useEffect, useRef, useState } from 'react';
import { apiFetch, postJson } from '../../api/fetchClient';
import { useToast } from '../common/Toast';
import { ENGINE_OPTIONS, OLLAMA_RELEVANT, VOCABULARY_OPTIONS } from './CaptionOptionsPopover';

/* 🧪 Caption Lab — try several caption configs on THIS image and compare them side by
   side, WITHOUT touching the stored caption. A candidate is engine × Ollama model ×
   vocabulary register; each runs through the /caption/preview endpoint, which reuses the
   real caption bricks (descriptive by-path pass) and never writes. Candidates run
   SEQUENTIALLY — the GPU is single and serialized server-side, so firing them in parallel
   would just 409/503. Results live only while the modal is open (ephemeral bench).

   ✓ Keep this one → drops the caption into the editor's textarea (normal save applies it).
   ⚙️ Make default → writes the winning config to the dataset's caption options. */

// A candidate with captioning disabled makes no sense in a comparison bench.
const CANDIDATE_ENGINES = ENGINE_OPTIONS.filter((o) => o.id !== 'none');
const MAX_CANDIDATES = 4;

// Short labels for a candidate card header (the long option labels are for the picker).
const ENGINE_SHORT = { '': 'Default engine', auto: 'Auto', joycaption: 'JoyCaption', ollama: 'Ollama' };
const VOCAB_SHORT = { '': '', explicit: 'Explicit', clinical: 'Clinical', safe: 'Safe' };

let candidateSeq = 0;
const newCandidate = (over = {}) => ({
  id: ++candidateSeq, backend: '', ollamaModel: '', vocabulary: '',
  status: 'idle', caption: '', chars: 0, durationMs: 0, error: '', cancelled: false, ...over,
});

function configLabel(c) {
  const parts = [ENGINE_SHORT[c.backend] ?? c.backend];
  if (OLLAMA_RELEVANT.has(c.backend) && c.ollamaModel) parts.push(c.ollamaModel);
  if (c.vocabulary) parts.push(VOCAB_SHORT[c.vocabulary]);
  return parts.join(' · ');
}

export default function CaptionLab({ datasetId, imageId, currentCaption, onKeep }) {
  const toast = useToast();
  const [models, setModels] = useState([]);
  const [modelsReachable, setModelsReachable] = useState(true);
  const [candidates, setCandidates] = useState(() => [newCandidate()]);
  const [running, setRunning] = useState(false);
  const abortRef = useRef(false);

  useEffect(() => {
    let alive = true;
    apiFetch('/api/ollama/models').catch(() => ({ models: [], reachable: false }))
      .then((mdl) => {
        if (!alive) return;
        setModels(mdl.models || []);
        setModelsReachable(mdl.reachable !== false);
      });
    return () => { alive = false; };
  }, []);

  const patch = (id, over) => setCandidates((cs) => cs.map((c) => (c.id === id ? { ...c, ...over } : c)));
  const addCandidate = () => setCandidates((cs) => (cs.length >= MAX_CANDIDATES ? cs : [...cs, newCandidate()]));
  const removeCandidate = (id) => setCandidates((cs) => (cs.length <= 1 ? cs : cs.filter((c) => c.id !== id)));

  // Run every candidate one after another. Between (and before) each we honor an abort;
  // the in-flight request is stopped server-side by /caption/cancel (the existing Stop path).
  const generate = async () => {
    abortRef.current = false;
    setRunning(true);
    // Reset prior results so a re-run reads cleanly.
    setCandidates((cs) => cs.map((c) => ({ ...c, status: 'idle', caption: '', error: '', cancelled: false })));
    // Snapshot the ids/config now — state updates during the loop won't reorder the run.
    const snapshot = candidates.map((c) => ({ id: c.id, backend: c.backend, ollamaModel: c.ollamaModel, vocabulary: c.vocabulary }));
    for (const c of snapshot) {
      if (abortRef.current) { patch(c.id, { status: 'idle' }); continue; }
      patch(c.id, { status: 'running', caption: '', error: '', cancelled: false });
      try {
        const started = performance.now();
        const r = await postJson(`/api/dataset/${datasetId}/image/${imageId}/caption/preview`,
          { backend: c.backend, ollama_model: c.ollamaModel, vocabulary: c.vocabulary });
        const elapsed = Math.round(performance.now() - started);
        if (r.cancelled) {
          patch(c.id, { status: 'cancelled', cancelled: true, durationMs: r.duration_ms ?? elapsed });
        } else {
          patch(c.id, { status: 'done', caption: r.caption || '', chars: r.chars ?? (r.caption || '').length,
            durationMs: r.duration_ms ?? elapsed });
        }
      } catch (e) {
        patch(c.id, { status: 'error', error: e.message || 'Preview failed' });
        if (e.status === 503 || e.status === 409) {
          // GPU busy / a real batch owns it — stop the whole run, nothing else will land.
          toast.error(e.message || 'The GPU is busy — try again once captioning/training is idle.');
          break;
        }
      }
    }
    setRunning(false);
  };

  const stop = async () => {
    abortRef.current = true;
    // Ask the in-flight preview to stop at its image boundary (idempotent; 409 = nothing running).
    await postJson(`/api/dataset/${datasetId}/caption/cancel`, {}).catch(() => null);
  };

  const makeDefault = async (c) => {
    try {
      await postJson(`/api/dataset/${datasetId}/caption/options`,
        { backend: c.backend, ollama_model: c.ollamaModel, vocabulary: c.vocabulary });
      toast.success('Saved as this dataset’s default caption method');
    } catch (e) {
      toast.error(e.message || 'Could not save the default');
    }
  };

  const selectCls = 'w-full px-2 py-1 rounded-lg bg-app/60 border border-border text-content text-xs';

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3">
      <div className="flex items-center justify-between gap-3">
        <p className="m-0 text-[0.6875rem] leading-relaxed text-content-subtle">
          Try up to {MAX_CANDIDATES} caption configs on this image and compare them. Nothing is saved
          until you pick one — “Keep” drops it into the editor, “Make default” stores the config for the dataset.
        </p>
        <div className="flex shrink-0 gap-2">
          {running ? (
            <button type="button" onClick={stop}
              className="rounded-lg border border-amber-400/40 bg-amber-500/10 px-3 py-1.5 text-xs font-semibold text-amber-200">
              ■ Stop
            </button>
          ) : (
            <button type="button" onClick={generate}
              className="rounded-lg bg-gradient-primary px-3 py-1.5 text-xs font-semibold text-white">
              ✨ Generate
            </button>
          )}
        </div>
      </div>

      {!modelsReachable && (
        <p className="m-0 rounded-lg border border-amber-400/40 bg-amber-500/10 px-3 py-1.5 text-[0.6875rem] text-amber-200">
          Ollama isn’t reachable — start it from Settings to list vision models. Candidates can still run on JoyCaption.
        </p>
      )}

      {/* Reference: the caption currently on the image. */}
      <div className="rounded-xl border border-border bg-surface p-3">
        <p className="m-0 text-[0.625rem] font-semibold uppercase tracking-[0.16em] text-content-subtle">Current caption</p>
        <p className="m-0 mt-1 whitespace-pre-wrap text-xs leading-5 text-content-muted">
          {currentCaption?.trim() ? currentCaption : <span className="italic text-content-subtle">— no caption yet —</span>}
        </p>
      </div>

      <div className="grid min-h-0 flex-1 gap-3 overflow-y-auto sm:grid-cols-2">
        {candidates.map((c) => (
          <div key={c.id} className="flex min-h-0 flex-col gap-2 rounded-xl border border-border bg-surface p-3">
            {/* Config picker */}
            <div className="flex flex-col gap-1.5">
              <div className="flex items-center justify-between gap-2">
                <span className="text-[0.625rem] font-semibold uppercase tracking-[0.16em] text-content-subtle">
                  {configLabel(c)}
                </span>
                <button type="button" onClick={() => removeCandidate(c.id)} disabled={candidates.length <= 1}
                  aria-label="Remove candidate"
                  className="text-content-subtle hover:text-content disabled:opacity-30 text-sm leading-none">✕</button>
              </div>
              <div className="grid grid-cols-2 gap-1.5">
                <select aria-label="Caption engine" value={c.backend} disabled={running}
                  onChange={(e) => patch(c.id, { backend: e.target.value })} className={selectCls}>
                  {CANDIDATE_ENGINES.map((o) => <option key={o.id} value={o.id}>{o.label}</option>)}
                </select>
                <select aria-label="Vocabulary" value={c.vocabulary} disabled={running}
                  onChange={(e) => patch(c.id, { vocabulary: e.target.value })} className={selectCls}>
                  {VOCABULARY_OPTIONS.map((o) => <option key={o.id} value={o.id}>{o.label}</option>)}
                </select>
              </div>
              <select aria-label="Ollama vision model" value={c.ollamaModel} disabled={running || !OLLAMA_RELEVANT.has(c.backend)}
                onChange={(e) => patch(c.id, { ollamaModel: e.target.value })}
                className={`${selectCls} ${OLLAMA_RELEVANT.has(c.backend) ? '' : 'opacity-40'}`}>
                <option value="">Default vision model</option>
                {(c.ollamaModel && !models.includes(c.ollamaModel) ? [c.ollamaModel, ...models] : models)
                  .map((m) => <option key={m} value={m}>{m}</option>)}
              </select>
            </div>

            {/* Result */}
            <div className="flex min-h-[4rem] flex-1 flex-col rounded-lg border border-border bg-app/60 p-2">
              {c.status === 'running' && <span className="text-xs text-content-subtle">Generating…</span>}
              {c.status === 'idle' && <span className="text-xs italic text-content-subtle">Not generated yet</span>}
              {c.status === 'cancelled' && <span className="text-xs text-amber-300">Stopped before it finished</span>}
              {c.status === 'error' && <span className="text-xs text-rose-400">{c.error}</span>}
              {c.status === 'done' && (
                <>
                  <p className="m-0 flex-1 overflow-y-auto whitespace-pre-wrap text-xs leading-5 text-content">
                    {c.caption || <span className="italic text-content-subtle">(the model returned an empty caption)</span>}
                  </p>
                  <span className="mt-1 font-mono text-[0.625rem] text-content-subtle">
                    {c.chars} chars · {(c.durationMs / 1000).toFixed(1)}s
                  </span>
                </>
              )}
            </div>

            <div className="flex justify-end gap-2">
              <button type="button" onClick={() => makeDefault(c)} disabled={running}
                title="Store this config as the dataset's default caption method"
                className="rounded-lg border border-border bg-surface px-2.5 py-1 text-[0.6875rem] font-medium text-content-muted hover:text-content disabled:opacity-40">
                ⚙️ Make default
              </button>
              <button type="button" onClick={() => onKeep(c.caption)} disabled={c.status !== 'done' || !c.caption}
                className="rounded-lg bg-emerald-600/90 px-2.5 py-1 text-[0.6875rem] font-semibold text-white disabled:opacity-30">
                ✓ Keep this one
              </button>
            </div>
          </div>
        ))}

        {candidates.length < MAX_CANDIDATES && (
          <button type="button" onClick={addCandidate} disabled={running}
            className="flex min-h-[4rem] items-center justify-center rounded-xl border border-dashed border-border text-sm text-content-subtle hover:text-content disabled:opacity-40">
            + Add candidate
          </button>
        )}
      </div>
    </div>
  );
}

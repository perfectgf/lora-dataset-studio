import { useEffect, useRef, useState } from 'react';
import { apiFetch, postJson } from '../../api/fetchClient';
import { useToast } from '../common/Toast';

/* ⚙️ Caption method options (per-dataset). Lets the user override, for THIS dataset:
   - the caption engine (or leave it on the global default);
   - which pulled Ollama vision model runs — and pull a new one by name, with progress;
   - extra instructions APPENDED to the caption prompt (the kind omission rules and the
     output cleaners stay in force server-side, so this can't reintroduce a banned term).
   Saved to caption_options and picked up by the next caption / re-caption run (targeted
   and dual-short included). Modal so it never fights the workspace layout. */

const ENGINE_OPTIONS = [
  { id: '', label: 'Use default (Settings ▸ Captioning)' },
  { id: 'auto', label: 'Auto — JoyCaption, then Ollama' },
  { id: 'joycaption', label: 'JoyCaption only' },
  { id: 'ollama', label: 'Ollama vision only' },
  { id: 'none', label: 'None — captioning disabled' },
];

// The Ollama model + pull only bite when the resolved engine can use Ollama.
const OLLAMA_RELEVANT = new Set(['', 'auto', 'ollama']);

// Vocabulary register for nude/sexual content. '' = leave the model to its own wording.
// 'explicit' is the NSFW lane — pair it with an abliterated Ollama vision model.
const VOCABULARY_OPTIONS = [
  { id: '', label: 'Default — the model’s own wording' },
  { id: 'explicit', label: 'Explicit — crude, uncensored terms' },
  { id: 'clinical', label: 'Clinical — neutral anatomical terms' },
  { id: 'safe', label: 'Safe — non-explicit, no crude terms' },
];

export default function CaptionOptionsPopover({ datasetId, onClose, onSaved }) {
  const toast = useToast();
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [backend, setBackend] = useState('');
  const [ollamaModel, setOllamaModel] = useState('');
  const [vocabulary, setVocabulary] = useState('');
  const [instructions, setInstructions] = useState('');
  const [models, setModels] = useState([]);
  const [modelsReachable, setModelsReachable] = useState(true);
  const [pullName, setPullName] = useState('');
  const [pull, setPull] = useState(null); // {state, model, progress, log, error}
  const pollRef = useRef(null);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const [opt, mdl] = await Promise.all([
          apiFetch(`/api/dataset/${datasetId}/caption/options`),
          apiFetch('/api/ollama/models').catch(() => ({ models: [], reachable: false })),
        ]);
        if (!alive) return;
        const o = opt.options || {};
        setBackend(o.backend || '');
        setOllamaModel(o.ollama_model || '');
        setVocabulary(o.vocabulary || '');
        setInstructions(o.instructions || '');
        setModels(mdl.models || []);
        setModelsReachable(mdl.reachable !== false);
      } catch {
        if (alive) toast.error('Could not load caption options');
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => { alive = false; clearInterval(pollRef.current); };
  }, [datasetId, toast]);

  const refreshModels = async () => {
    const mdl = await apiFetch('/api/ollama/models').catch(() => null);
    if (mdl) { setModels(mdl.models || []); setModelsReachable(mdl.reachable !== false); }
    return mdl;
  };

  const poll = () => {
    clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      let s;
      try { s = await apiFetch('/api/ollama/pull'); } catch { clearInterval(pollRef.current); return; }
      setPull(s);
      if (s.state === 'running') return;
      clearInterval(pollRef.current);
      if (s.state === 'success') {
        toast.success(`Pulled ${s.model}`);
        await refreshModels();
        setOllamaModel(s.model);   // use what we just pulled
        setPullName('');
      } else if (s.state === 'error') {
        toast.error(s.error || 'The pull failed');
      }
    }, 1200);
  };

  const startPull = async () => {
    const name = pullName.trim();
    if (!name) return;
    const r = await postJson('/api/ollama/pull', { model: name }).catch(() => null);
    if (!r) return;
    if (!r.ok) { toast.error(r.error || 'Could not start the pull'); return; }
    setPull({ state: 'running', model: name, progress: r.progress ?? null, log: r.log || [], error: null });
    poll();
  };

  const save = async () => {
    setSaving(true);
    try {
      const r = await postJson(`/api/dataset/${datasetId}/caption/options`,
        { backend, ollama_model: ollamaModel, vocabulary, instructions });
      toast.success('Caption options saved');
      onSaved?.(r.options);
      onClose();
    } catch (e) {
      toast.error(e.message || 'Could not save the caption options');
    } finally {
      setSaving(false);
    }
  };

  const pulling = pull?.state === 'running';
  // Keep a model that isn't in the live list (pulled elsewhere) selectable.
  const modelChoices = ollamaModel && !models.includes(ollamaModel)
    ? [ollamaModel, ...models] : models;
  const inputCls = 'w-full px-2 py-1.5 rounded-lg bg-app/60 border border-border text-content text-sm';

  return (
    <div className="fixed inset-0 z-[9990] flex items-center justify-center bg-black/80 p-3"
      onClick={(e) => { e.stopPropagation(); onClose(); }}
      onKeyDown={(e) => { if (e.key === 'Escape') { e.preventDefault(); onClose(); } }}>
      <div role="dialog" aria-label="Caption method options"
        className="w-full max-w-md max-h-[92vh] overflow-y-auto rounded-xl border border-border bg-surface-overlay p-4 shadow-2xl flex flex-col gap-4"
        onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between">
          <h3 className="text-content font-semibold text-sm">⚙️ Caption method — this dataset</h3>
          <button type="button" onClick={onClose} aria-label="Close"
            className="text-content-subtle hover:text-content text-lg leading-none">×</button>
        </div>

        {loading ? (
          <p className="text-content-subtle text-sm py-6 text-center">Loading…</p>
        ) : (
          <>
            {/* Engine */}
            <div className="flex flex-col gap-1">
              <label htmlFor="cap-opt-engine" className="text-sm font-medium text-content">Caption engine</label>
              <select id="cap-opt-engine" value={backend} onChange={(e) => setBackend(e.target.value)}
                className={inputCls}>
                {ENGINE_OPTIONS.map((o) => <option key={o.id} value={o.id}>{o.label}</option>)}
              </select>
              <p className="text-xs text-content-subtle">
                Overrides the global default only for this dataset. Leave on “default” to follow Settings.
              </p>
            </div>

            {/* Ollama model + pull */}
            <div className={`flex flex-col gap-1 ${OLLAMA_RELEVANT.has(backend) ? '' : 'opacity-50'}`}>
              <label htmlFor="cap-opt-model" className="text-sm font-medium text-content">Ollama vision model</label>
              <select id="cap-opt-model" value={ollamaModel} onChange={(e) => setOllamaModel(e.target.value)}
                className={inputCls}>
                <option value="">Use default (Settings ▸ Captioning)</option>
                {modelChoices.map((m) => <option key={m} value={m}>{m}</option>)}
              </select>
              {!modelsReachable && (
                <p className="text-xs text-amber-400/90">
                  Ollama isn’t reachable — start it from Settings to list or pull models.
                </p>
              )}
              <p className="text-xs text-content-subtle">
                Only used when the engine is Auto or Ollama. Pull a new vision model by name:
              </p>
              <div className="flex items-center gap-2">
                <input value={pullName} onChange={(e) => setPullName(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); startPull(); } }}
                  placeholder="e.g. huihui_ai/qwen3-vl-abliterated:8b-instruct"
                  aria-label="Ollama model to pull"
                  className="flex-1 px-2 py-1.5 rounded-lg bg-app/60 border border-border text-content text-xs" />
                <button type="button" onClick={startPull} disabled={pulling || !pullName.trim() || !modelsReachable}
                  className="px-3 py-1.5 rounded-lg bg-surface-raised border border-border text-content text-xs font-semibold disabled:opacity-40 hover:bg-surface">
                  {pulling ? 'Pulling…' : '⇩ Pull'}
                </button>
              </div>
              {pull && (
                <p className={`text-xs ${pull.state === 'error' ? 'text-rose-400' : 'text-content-subtle'}`}>
                  {pull.state === 'running'
                    ? `Pulling ${pull.model}${pull.progress != null ? ` — ${pull.progress}%` : ''}${pull.log?.length ? ` (${pull.log[pull.log.length - 1]})` : ''}`
                    : pull.state === 'success' ? `Pulled ${pull.model}.`
                    : `Pull failed: ${pull.error || 'unknown error'}`}
                </p>
              )}
            </div>

            {/* Vocabulary preset (NSFW register) */}
            <div className="flex flex-col gap-1">
              <label htmlFor="cap-opt-vocab" className="text-sm font-medium text-content">Vocabulary</label>
              <select id="cap-opt-vocab" value={vocabulary} onChange={(e) => setVocabulary(e.target.value)}
                className={inputCls}>
                {VOCABULARY_OPTIONS.map((o) => <option key={o.id} value={o.id}>{o.label}</option>)}
              </select>
              <p className="text-xs text-content-subtle">
                How the model names nude or sexual content. “Explicit” needs an uncensored
                (abliterated) vision model — pull one above. The omission rules and leak
                cleaners still run, so this changes wording, not what binds to the trigger.
              </p>
            </div>

            {/* Extra instructions */}
            <div className="flex flex-col gap-1">
              <label htmlFor="cap-opt-instructions" className="text-sm font-medium text-content">Extra instructions</label>
              <textarea id="cap-opt-instructions" value={instructions} rows={3}
                onChange={(e) => setInstructions(e.target.value)}
                placeholder="e.g. Always name the visible clothing colors and the time of day."
                className="w-full px-2 py-1.5 rounded-lg bg-app/60 border border-border text-content text-sm resize-y" />
              <p className="text-xs text-content-subtle">
                Added to the end of the caption prompt (both engines). The identity / concept / style
                omission rules and the leak cleaners still apply, so this can’t reintroduce a banned term.
              </p>
            </div>

            <div className="flex justify-end gap-2 pt-1">
              <button type="button" onClick={onClose}
                className="px-3 py-1.5 rounded-lg bg-surface border border-border text-content-muted text-sm">
                Cancel
              </button>
              <button type="button" onClick={save} disabled={saving}
                className="px-4 py-1.5 rounded-lg bg-gradient-primary text-white text-sm font-semibold disabled:opacity-40">
                {saving ? 'Saving…' : 'Save'}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

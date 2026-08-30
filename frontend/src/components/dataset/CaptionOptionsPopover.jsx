import { useEffect, useRef, useState } from 'react';
import { apiFetch, postJson } from '../../api/fetchClient';
import { useToast } from '../common/Toast';
import { appearancePolicyChanged } from '../../utils/captionAppearancePolicy.js';
import { modelPickerCopy } from '../../utils/localLlm.js';

/* ⚙️ Caption method options (per-dataset). Lets the user override, for THIS dataset:
   - the caption engine (or leave it on the global default);
   - which pulled Ollama vision model runs — and pull a new one by name, with progress;
   - the vocabulary register and the caption LENGTH preset, two independent axes;
   - extra instructions APPENDED to the caption prompt (the kind omission rules and the
     output cleaners stay in force server-side, so this can't reintroduce a banned term);
   - for character datasets, which appearance families to Omit (bind to the trigger) vs
     Describe (stay prompt-controllable). Extra instructions cannot move a family.
   Saved to caption_options and picked up by the next caption / re-caption run (targeted
   and dual-short included). Modal so it never fights the workspace layout. */

// Shared with the 🧪 Caption Lab (CaptionLab.jsx) so the engine/vocabulary choices
// never drift between "set the dataset default" and "try a candidate".
export const ENGINE_OPTIONS = [
  { id: '', label: 'Use default (Settings ▸ Captioning)' },
  // The stored VALUES never change — 'ollama' lives in datasets people already
  // have, and renaming it would silently alter what their saved options mean. Only
  // the labels follow the fact that the local engine may now be either server; the
  // value 'ollama' reads as "the configured local provider" (settings-reference
  // says so). This is the alias path CLAUDE.md asks for, applied to the label side.
  { id: 'auto', label: 'Auto — JoyCaption, then the local LLM' },
  { id: 'joycaption', label: 'JoyCaption only' },
  { id: 'ollama', label: 'Local LLM only (Ollama / LM Studio)' },
  { id: 'none', label: 'None — captioning disabled' },
];

// The Ollama model + pull only bite when the resolved engine can use Ollama.
export const OLLAMA_RELEVANT = new Set(['', 'auto', 'ollama']);

// Optional Krea 2 companion preset. This is deliberately a per-dataset override:
// the global 8B abliterated default remains the safer NSFW captioning choice.
export const KREA_2_OLLAMA_MODEL = 'qwen3-vl:4b-instruct';

// Vocabulary register for nude/sexual content. '' = leave the model to its own wording.
// 'explicit' is the NSFW lane — pair it with an abliterated Ollama vision model.
export const VOCABULARY_OPTIONS = [
  { id: '', label: 'Default — the model’s own wording' },
  { id: 'explicit', label: 'Explicit — crude, uncensored terms' },
  { id: 'clinical', label: 'Clinical — neutral anatomical terms' },
  { id: 'safe', label: 'Safe — non-explicit, no crude terms' },
];

// Caption LENGTH preset — how MUCH the captioner writes. Orthogonal to the vocabulary
// register above (which decides the wording, not the size) and to the dual long+short
// captions (whose short half is derived from the long one, in its own field).
// '' = Standard: nothing appended, byte-identical to the prompt before this option existed.
// Idea by djpraxis (Reddit).
export const CAPTION_LENGTH_OPTIONS = [
  { id: '', label: 'Standard — the prompt as it is' },
  { id: 'concise', label: 'Concise — aims for one short sentence' },
  { id: 'detailed', label: 'Detailed — several sentences' },
];

// Character-only omit vs describe. Must match backend APPEARANCE_FAMILIES /
// APPEARANCE_DEFAULTS in face_variations.py — Extra instructions cannot move a
// family between columns; these toggles can.
export const APPEARANCE_FAMILIES = [
  { id: 'hair', label: 'Hair', hint: 'length, colour, style, how it falls' },
  { id: 'makeup', label: 'Makeup and nails', hint: 'mascara, lipstick, eyeshadow, blush, manicure' },
  { id: 'facial_hair', label: 'Facial hair', hint: 'beard, stubble, moustache, or clean-shaven' },
  { id: 'glasses', label: 'Glasses', hint: 'glasses and sunglasses' },
];
export const APPEARANCE_DEFAULTS = {
  hair: 'omit', makeup: 'describe', facial_hair: 'omit', glasses: 'describe',
};
const APPEARANCE_LOCKED = [
  'face shape', 'eye colour', 'skin', 'age', 'gender', 'ethnicity',
];

export default function CaptionOptionsPopover({ datasetId, trainType, kind, onClose, onSaved }) {
  const toast = useToast();
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [backend, setBackend] = useState('');
  const [ollamaModel, setOllamaModel] = useState('');
  const [vocabulary, setVocabulary] = useState('');
  const [length, setLength] = useState('');
  const [instructions, setInstructions] = useState('');
  const [appearance, setAppearance] = useState(null); // null = no policy (classic lock)
  const [appearanceDirty, setAppearanceDirty] = useState(false);
  // The policy as LOADED, to tell a real edit from a plain re-save on Save.
  const initialAppearanceRef = useRef(null);
  const [models, setModels] = useState([]);
  const [modelsReachable, setModelsReachable] = useState(true);
  // Pulling is an Ollama-only capability: LM Studio downloads models in its own
  // app. The routed endpoint has always returned `provider`; nothing read it, so
  // the ⇩ Pull button stayed enabled by the ACTIVE provider's reachability while
  // still talking to /api/ollama/pull — a button that answers about a daemon the
  // user is not running.
  const [modelsProvider, setModelsProvider] = useState('ollama');
  const picker = modelPickerCopy(modelsProvider);
  const canPull = picker.canPull;
  const [pullName, setPullName] = useState('');
  const [pull, setPull] = useState(null); // {state, model, progress, log, error}
  const pollRef = useRef(null);
  const mountedRef = useRef(false);
  const pollGenerationRef = useRef(0);
  const selectOllamaAfterPullRef = useRef(false);

  useEffect(() => {
    let alive = true;
    mountedRef.current = true;
    (async () => {
      try {
        const [opt, mdl] = await Promise.all([
          apiFetch(`/api/dataset/${datasetId}/caption/options`),
          apiFetch('/api/local-llm/models').catch(() => ({ models: [], reachable: false })),
        ]);
        if (!alive) return;
        const o = opt.options || {};
        setBackend(o.backend || '');
        setOllamaModel(o.ollama_model || '');
        setVocabulary(o.vocabulary || '');
        setLength(o.length || '');
        setInstructions(o.instructions || '');
        const stored = o.appearance && Object.keys(o.appearance).length ? o.appearance : null;
        setAppearance(stored);
        initialAppearanceRef.current = stored;
        setAppearanceDirty(false);
        setModels(mdl.models || []);
        setModelsReachable(mdl.reachable !== false);
        setModelsProvider(mdl.provider || 'ollama');
      } catch {
        if (alive) toast.error('Could not load caption options');
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => {
      alive = false;
      mountedRef.current = false;
      pollGenerationRef.current += 1;
      clearTimeout(pollRef.current);
      pollRef.current = null;
      selectOllamaAfterPullRef.current = false;
    };
  }, [datasetId, toast]);

  const refreshModels = async (isCurrent = () => mountedRef.current) => {
    const mdl = await apiFetch('/api/local-llm/models').catch(() => null);
    if (mdl && isCurrent()) {
      setModels(mdl.models || []);
      setModelsReachable(mdl.reachable !== false);
      setModelsProvider(mdl.provider || 'ollama');
    }
    return mdl;
  };

  const poll = () => {
    clearTimeout(pollRef.current);
    const generation = ++pollGenerationRef.current;
    const isCurrent = () => (
      mountedRef.current && pollGenerationRef.current === generation
    );

    const checkPull = async () => {
      pollRef.current = null;
      let s;
      try { s = await apiFetch('/api/ollama/pull', { background: true }); }
      catch (error) {
        if (!isCurrent()) return;
        const message = error?.message || 'Could not check Ollama pull status';
        selectOllamaAfterPullRef.current = false;
        setPull((current) => ({
          ...(current || {}),
          state: 'error',
          error: message,
        }));
        toast.error(message);
        return;
      }
      if (!isCurrent()) return;
      setPull(s);
      if (s.state === 'running') {
        // Schedule only after this GET settled: at most one status request is in flight.
        pollRef.current = setTimeout(checkPull, 1200);
        return;
      }
      if (s.state === 'success') {
        const selectOllamaBackend = selectOllamaAfterPullRef.current;
        selectOllamaAfterPullRef.current = false;
        await refreshModels(isCurrent);
        if (!isCurrent()) return;
        toast.success(`Pulled ${s.model}`);
        setOllamaModel(s.model);   // use what we just pulled
        if (selectOllamaBackend) setBackend('ollama');
        setPullName('');
      } else if (s.state === 'error') {
        selectOllamaAfterPullRef.current = false;
        toast.error(s.error || 'The pull failed');
      }
    };

    pollRef.current = setTimeout(checkPull, 1200);
  };

  const startPullByName = async (rawName, selectOllamaBackend = false) => {
    const name = rawName.trim();
    if (!name) return;
    const r = await postJson('/api/ollama/pull', { model: name }).catch(() => null);
    if (!mountedRef.current || !r) return;
    if (!r.ok) { toast.error(r.error || 'Could not start the pull'); return; }
    selectOllamaAfterPullRef.current = selectOllamaBackend;
    setPull({ state: 'running', model: name, progress: r.progress ?? null, log: r.log || [], error: null });
    poll();
  };

  const startPull = () => startPullByName(pullName);

  const save = async () => {
    setSaving(true);
    try {
      const payload = { backend, ollama_model: ollamaModel, vocabulary, length, instructions };
      // Saving vocabulary / length / Extra instructions must not invent a policy.
      if (appearanceDirty) {
        payload.appearance = { ...APPEARANCE_DEFAULTS, ...(appearance || {}) };
      }
      const r = await postJson(`/api/dataset/${datasetId}/caption/options`, payload);
      // A moved policy only steers the NEXT run: images already captioned keep the
      // rule they were written under, and the plain Caption button skips them (it
      // only fills empty captions). Say so, or the dataset silently ends up half
      // under each rule. Same 'future captions' nudge as a kind change.
      const policyMoved = appearancePolicyChanged(
        initialAppearanceRef.current, r.options?.appearance);
      if (policyMoved) {
        toast.success(
          'Appearance policy saved — re-caption to apply it to existing captions',
          10000);
      } else {
        toast.success('Caption options saved');
      }
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
  const kreaModelInstalled = models.includes(KREA_2_OLLAMA_MODEL);
  const kreaModelSelected = backend === 'ollama' && ollamaModel === KREA_2_OLLAMA_MODEL;
  const kreaModelPulling = pulling && pull?.model === KREA_2_OLLAMA_MODEL;
  const chooseKreaModel = () => {
    if (kreaModelInstalled) {
      setBackend('ollama');
      setOllamaModel(KREA_2_OLLAMA_MODEL);
      return;
    }
    startPullByName(KREA_2_OLLAMA_MODEL, true);
  };
  const inputCls = 'w-full px-2 py-1.5 rounded-lg bg-app/60 border border-border text-content text-sm';
  const showAppearance = kind !== 'concept' && kind !== 'style';
  const shownAppearance = { ...APPEARANCE_DEFAULTS, ...(appearance || {}) };
  const policyActive = Boolean(appearance && Object.keys(appearance).length);
  const flipAppearance = (id, value) => {
    setAppearance({ ...shownAppearance, [id]: value });
    setAppearanceDirty(true);
  };

  return (
    <div className="fixed inset-0 z-[9990] flex items-center justify-center bg-black/80 p-3"
      onClick={(e) => { e.stopPropagation(); onClose(); }}
      onKeyDown={(e) => { if (e.key === 'Escape') { e.preventDefault(); onClose(); } }}>
      <div role="dialog" aria-modal="true" aria-label="Caption method options"
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
              <label htmlFor="cap-opt-model" className="text-sm font-medium text-content">{picker.modelLabel}</label>
              <select id="cap-opt-model" value={ollamaModel} onChange={(e) => setOllamaModel(e.target.value)}
                className={inputCls}>
                <option value="">Use default (Settings ▸ Captioning)</option>
                {modelChoices.map((m) => <option key={m} value={m}>{m}</option>)}
              </select>
              {!modelsReachable && (
                <p className="text-xs text-amber-400/90">{picker.down}</p>
              )}
              {trainType === 'krea' && (
                <section aria-labelledby="krea-caption-model-title"
                  className="mt-2 rounded-lg border border-sky-400/30 bg-sky-500/10 p-3">
                  <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                    <div className="min-w-0 flex-1">
                      <h4 id="krea-caption-model-title" className="text-sm font-semibold text-content">
                        Krea 2 companion preset
                      </h4>
                      <p className="mt-0.5 break-all font-mono text-xs text-sky-300">
                        {KREA_2_OLLAMA_MODEL}
                      </p>
                    </div>
                    <button type="button" onClick={chooseKreaModel}
                      disabled={pulling || (!kreaModelInstalled && !modelsReachable)}
                      aria-describedby="krea-caption-model-note"
                      aria-pressed={kreaModelInstalled ? kreaModelSelected : undefined}
                      aria-busy={kreaModelPulling}
                      className="w-full shrink-0 rounded-lg border border-sky-400/40 bg-sky-500/20 px-3 py-2 text-xs font-semibold text-sky-100 hover:bg-sky-500/30 disabled:opacity-40 sm:w-auto">
                      {kreaModelSelected ? 'Selected'
                        : kreaModelPulling ? 'Pulling…'
                        : kreaModelInstalled ? 'Use'
                        : 'Pull & use'}
                    </button>
                  </div>
                  <p id="krea-caption-model-note" className="mt-2 text-xs leading-relaxed text-content-subtle">
                    Uses the same Qwen3-VL-4B Instruct base model family/checkpoint as Krea 2 Edit.
                    Ollama uses a different quantization and runtime, so captions are not bit-identical.
                    The vanilla model can refuse explicit content; the default 8B abliterated model
                    remains the better choice for NSFW.
                  </p>
                </section>
              )}
              {!canPull && (
                <p className="text-xs text-content-subtle">
                  Downloading models is done in the LM Studio app, which shows progress and
                  lets you cancel — this app cannot do that for it.
                </p>
              )}
              {canPull && (<>
              <p className="text-xs text-content-subtle">
                Only used when the engine is Auto or Ollama. Pull a new vision model by name:
              </p>
              <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
                <input value={pullName} onChange={(e) => setPullName(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); startPull(); } }}
                  placeholder="e.g. huihui_ai/qwen3-vl-abliterated:8b-instruct"
                  aria-label="Ollama model to pull"
                  className="min-w-0 w-full flex-1 px-2 py-1.5 rounded-lg bg-app/60 border border-border text-content text-xs" />
                <button type="button" onClick={startPull} disabled={pulling || !pullName.trim() || !modelsReachable}
                  className="w-full px-3 py-1.5 rounded-lg bg-surface-raised border border-border text-content text-xs font-semibold disabled:opacity-40 hover:bg-surface sm:w-auto">
                  {pulling ? 'Pulling…' : '⇩ Pull'}
                </button>
              </div>
              {pull && (
                <p role="status" aria-live="polite"
                  className={`break-words text-xs ${pull.state === 'error' ? 'text-rose-400' : 'text-content-subtle'}`}>
                  {pull.state === 'running'
                    ? `Pulling ${pull.model}${pull.progress != null ? ` — ${pull.progress}%` : ''}${pull.log?.length ? ` (${pull.log[pull.log.length - 1]})` : ''}`
                    : pull.state === 'success' ? `Pulled ${pull.model}.`
                    : `Pull failed: ${pull.error || 'unknown error'}`}
                </p>
              )}
              </>)}
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

            {/* Caption length preset */}
            <div className="flex flex-col gap-1">
              <label htmlFor="cap-opt-length" className="text-sm font-medium text-content">Caption length</label>
              <select id="cap-opt-length" value={length} onChange={(e) => setLength(e.target.value)}
                className={inputCls}>
                {CAPTION_LENGTH_OPTIONS.map((o) => <option key={o.id} value={o.id}>{o.label}</option>)}
              </select>
              <p className="text-xs text-content-subtle">
                How much the captioner writes. “Standard” leaves the prompt untouched.
                “Concise” aims for one short sentence, “Detailed” for several — a target the
                model follows loosely, not a hard cap. Measured on 18 photos with the default
                vision model: median 24 / 88 / 126 words for Concise / Standard / Detailed;
                another engine or model will land elsewhere. Both stay full prose, so a Concise
                dataset still trains on prose-native families without forcing past the
                caption-style check.
              </p>
              <p className="text-xs text-content-subtle">
                Different axis from the long + short dual captions: the short one is derived
                from the long caption and stored separately, while this changes the long
                caption itself. You can use both.
              </p>
            </div>

            {/* Appearance policy (character only): omit binds to the trigger,
                describe stays prompt-controllable. Extra instructions cannot
                move a family between columns — these toggles can. */}
            {showAppearance && (
            <div className="flex flex-col gap-2">
              <span className="text-sm font-medium text-content">Appearance in captions</span>
              <p className="text-xs text-content-subtle leading-relaxed">
                <strong className="font-medium text-content-muted">Omit</strong> binds that look
                to the trigger (do not caption it).{' '}
                <strong className="font-medium text-content-muted">Describe</strong> names
                what is clearly visible so it stays prompt-controllable (lipstick, a beard,
                glasses). Re-caption after changing this.
              </p>
              {!policyActive && !appearanceDirty && (
                <p className="text-xs text-content-subtle leading-relaxed">
                  Not applied until you change a row — the captioner currently uses the classic
                  lock (hair omitted, makeup neither asked nor forbidden).
                </p>
              )}
              <div className="flex flex-col gap-1.5">
                {APPEARANCE_FAMILIES.map((fam) => {
                  const value = shownAppearance[fam.id];
                  return (
                    <div key={fam.id}
                      className="flex items-center justify-between gap-2 rounded-lg border border-border bg-app/40 px-2 py-1.5">
                      <div className="min-w-0">
                        <div className="text-sm text-content">{fam.label}</div>
                        <div className="text-[0.6875rem] text-content-subtle">{fam.hint}</div>
                      </div>
                      <div className="flex shrink-0 rounded-md border border-border overflow-hidden">
                        {['omit', 'describe'].map((state) => (
                          <button key={state} type="button"
                            aria-pressed={value === state}
                            onClick={() => flipAppearance(fam.id, state)}
                            className={`px-2 py-1 text-[0.6875rem] font-semibold capitalize ${
                              value === state
                                ? 'bg-surface-raised text-content'
                                : 'bg-transparent text-content-muted hover:text-content'
                            }`}>
                            {state}
                          </button>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>
              <p className="text-xs text-content-subtle leading-relaxed">
                Always omitted (the trigger’s job): {APPEARANCE_LOCKED.join(', ')}.
              </p>
            </div>
            )}

            {/* Extra instructions */}
            <div className="flex flex-col gap-1">
              <label htmlFor="cap-opt-instructions" className="text-sm font-medium text-content">Extra instructions</label>
              <textarea id="cap-opt-instructions" value={instructions} rows={3}
                onChange={(e) => setInstructions(e.target.value)}
                placeholder="e.g. Always name the visible clothing colors and the time of day."
                className="w-full px-2 py-1.5 rounded-lg bg-app/60 border border-border text-content text-sm resize-y" />
              <p className="text-xs text-content-subtle">
                Added to the end of the caption prompt (both engines), <em>after</em> the vocabulary
                and length presets — so when yours contradicts one of them, yours is what the model
                reads last.{showAppearance
                  ? ' Extra instructions can only steer wording of a described appearance family — they still cannot reintroduce an omitted one (hair stays omitted until you flip it above). Identity leak cleaners still run.'
                  : ' The identity / concept / style omission rules and the leak cleaners still apply, so this can’t reintroduce a banned term.'}
              </p>
            </div>

            <div className="flex justify-end gap-2 pt-1">
              <button type="button" onClick={onClose}
                className="px-3 py-1.5 rounded-lg bg-surface border border-border text-content-muted text-sm">
                Cancel
              </button>
              <button type="button" onClick={save} disabled={saving}
                className="px-4 py-1.5 rounded-lg bg-gradient-primary text-gray-950 text-sm font-semibold disabled:opacity-40">
                {saving ? 'Saving…' : 'Save'}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

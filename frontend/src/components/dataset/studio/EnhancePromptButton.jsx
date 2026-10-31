// react-frontend/src/components/dataset/studio/EnhancePromptButton.jsx
/**
 * Enhance sends the typed prompt to LOCAL Ollama before generation, defaulting to the captioning
 * model without a second client. The adjacent settings picker offers already-pulled Ollama models.
 * Empty selection follows the default and OMITS the request key, matching Bank caption controls'
 * spread-if-set contract. One localStorage preference serves Test Studio and Canvas through
 * RunSetupPanel/PromptField. Without Ollama, disable the button and explain the exact reason:
 * absent installation, unreachable server or missing model. Reuse /api/capabilities caps.ollama,
 * already used by Bank and Settings; add no probe. An explicit model bypasses the
 * missing-default-model gate because that default is not used; the server validates the chosen
 * model and names it in any 409 response.
 */
import { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import { apiFetch, postJson } from '../../../api/fetchClient';
import { useCapabilities } from '../../../context/CapabilitiesContext';
import { useToast } from '../../common/Toast';
import useOllamaFence from '../../../hooks/useOllamaFence';
import { SUPERSEDED_ANSWER_NOTICE, keepAnswer } from '../../../utils/ollamaFence';
import { modelPickerCopy } from '../../../utils/localLlm.js';
import OllamaFenceNotice from '../../common/OllamaFenceNotice';
import SettingsLink from '../../common/SettingsLink';
import { enhanceBlocker } from './enhanceGate';
import { activeLocalLlm } from '../../../utils/localLlm'

/* One preference for the tool, wherever it is mounted — deliberately NOT keyed per
   dataset or per surface: the same enhance on the Canvas must not silently run a
   different model than the one picked in the Studio. */
const MODEL_KEY = 'studioEnhanceModel';
const readStoredModel = () => {
  try { return localStorage.getItem(MODEL_KEY) || ''; } catch { return ''; }
};

/* ⚙️ The model picker. A small centered modal, not an anchored panel: this toolbar
   flex-wraps on phones and an anchored popover would overflow it (the Captions ⚙️
   made the same call, for the same reason). Mounted only while open, so the models
   list is fetched on open — never as a side effect of rendering the toolbar. */
function EnhanceModelPopover({ model, onPick, onClose }) {
  const [models, setModels] = useState([]);
  const [reachable, setReachable] = useState(true);
  const [provider, setProvider] = useState('ollama');
  const picker = modelPickerCopy(provider);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    let alive = true;
    // Always-200 endpoint: an unreachable Ollama is an empty list, never an error.
    apiFetch('/api/local-llm/models').catch(() => ({ models: [], reachable: false }))
      .then((d) => {
        if (!alive) return;
        setModels(d?.models || []);
        setReachable(d?.reachable !== false);
        setProvider(d?.provider || 'ollama');
        setLoading(false);
      });
    return () => { alive = false; };
  }, []);
  // A model pulled elsewhere (or picked before Ollama went down) stays selectable —
  // silently dropping the user's choice is worse than offering an unconfirmed name.
  const choices = model && !models.includes(model) ? [model, ...models] : models;
  /*
   * Portal required: the Studio aside's lg:sticky or Canvas transform creates a parent stacking
   * context that caps inner z-index, while overflow-auto clips content. See
   * studioModalsArePortaled.contract.test.js.
   */
  return createPortal(
    <div className="fixed inset-0 z-[9990] flex items-center justify-center bg-black/80 p-3"
      onClick={(e) => { e.stopPropagation(); onClose(); }}
      onKeyDown={(e) => { if (e.key === 'Escape') { e.preventDefault(); onClose(); } }}>
      <div role="dialog" aria-modal="true" aria-label="Enhance options"
        className="flex w-full max-w-sm flex-col gap-3 rounded-xl border border-border bg-surface-overlay p-4 shadow-2xl"
        onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-content">⚙️ Enhance — which model</h3>
          <button type="button" onClick={onClose} aria-label="Close"
            className="text-lg leading-none text-content-subtle hover:text-content">×</button>
        </div>
        <div className="flex flex-col gap-1">
          <label htmlFor="enhance-model" className="text-sm font-medium text-content">
            {picker.label} model
          </label>
          <select id="enhance-model" value={model} disabled={loading}
            onChange={(e) => onPick(e.target.value)}
            className="w-full rounded-lg border border-border bg-app/60 px-2 py-1.5 text-sm text-content">
            <option value="">Use default — the captioning model (Settings ▸ Captioning)</option>
            {choices.map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
          {!reachable && (
            <p className="text-xs text-amber-400/90">{picker.down}</p>
          )}
          <p className="text-xs text-content-subtle">
            Applies immediately, and is remembered on this browser for both the Test
            Studio and the Canvas. Enhance is a text call, so any model you have works —
            but a vanilla model can refuse NSFW prompts; the abliterated captioning
            default is the safe choice there.{' '}
            {picker.canPull
              ? <>Download a model in{' '}<SettingsLink section="local-tools" focus="ollama-vision-model">Ollama model settings</SettingsLink>.</>
              : 'Load another model inside LM Studio itself — it shows progress and lets you cancel, which this app cannot do for it.'}
          </p>
        </div>
        <div className="flex justify-end">
          <button type="button" onClick={onClose}
            className="rounded-lg bg-gradient-primary px-4 py-1.5 text-sm font-semibold text-gray-950">
            Done
          </button>
        </div>
      </div>
    </div>
    ,
    document.body,
  );
}

export default function EnhancePromptButton({ prompt, onResult, className = '' }) {
  const toast = useToast();
  const { caps, loading } = useCapabilities();
  const [busy, setBusy] = useState(false);
  const [model, setModel] = useState(readStoredModel);
  const [optionsOpen, setOptionsOpen] = useState(false);
  // A replay happens long after the click, outside the try/catch below — so it
  // needs its own way to speak, or a failed automatic retry would say nothing.
  const { fence, runGuarded, unloadAndRetry, stopWaiting } = useOllamaFence({
    onError: (e) => toast.error(e?.message || 'Enhance failed'),
  });
  const localLlm = activeLocalLlm(caps);
  const blocked = enhanceBlocker(localLlm, { capsLoading: loading, customModel: model });
  const empty = !((prompt || '').trim());
  const title = blocked
    || (empty ? 'Write a prompt first'
      : model ? `Enrich this prompt with ${model} (Ollama)`
        : 'Enrich this prompt with the local model (Ollama)');

  const pick = (m) => {
    setModel(m);
    try { localStorage.setItem(MODEL_KEY, m); } catch { /* private mode — best effort */ }
  };

  /* The action, not the click: the guard keeps it and replays it verbatim when
     the model frees up, so `prompt` (and the picked model) are captured here on
     purpose. '' = default → the key stays OUT of the body, byte-identical to the
     request before the ⚙️ existed. */
  const enhance = async (run) => {
    const d = await postJson('/api/studio/enhance-prompt',
      { prompt, ...(model ? { ollama_model: model } : {}) });
    // A reply for a click the user has moved on from — a newer click took
    // over while this one was in flight — is set aside, not written over
    // the newer one's answer.
    if (!keepAnswer(run, () => toast.info(SUPERSEDED_ANSWER_NOTICE))) return;
    if (d?.ok && d.prompt) onResult(d.prompt);
    else toast.error(d?.error || 'The model returned nothing');
  };

  const run = async () => {
    if (blocked || empty || busy) return;
    setBusy(true);
    try {
      // A fence refusal is not an error to announce: the notice below takes
      // over and the enhance restarts by itself. Everything else still toasts.
      await runGuarded(enhance);
    } catch (e) {
      toast.error(e?.message || 'Enhance failed');
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <button type="button" onClick={run} disabled={!!blocked || empty || busy} title={title}
        aria-label="Enhance the prompt with the local model"
        className={`px-2 py-0.5 rounded border border-border bg-surface text-content-subtle text-[0.625rem] hover:text-content disabled:opacity-40 disabled:cursor-not-allowed ${className}`}>
        {busy ? '✨ …' : '✨ Enhance'}
      </button>
      {/* The ⚙️ stays clickable even while Enhance is blocked: seeing and changing
          the picked model is exactly what fixes a "model not pulled" block. */}
      <button type="button" onClick={() => setOptionsOpen(true)}
        aria-haspopup="dialog" aria-label="Enhance options — pick the model"
        title={model ? `Enhance model: ${model}` : 'Enhance options — pick the model (default: the captioning model)'}
        className={`px-1.5 py-0.5 rounded border border-border bg-surface text-[0.625rem] hover:text-content ${model ? 'text-content' : 'text-content-subtle'}`}>
        <span aria-hidden>⚙️</span>
      </button>
      {blocked && !loading && (
        <SettingsLink section="local-tools" tone="warning"
          focus={localLlm.provider === 'lmstudio'
            ? (localLlm.reachable ? 'lmstudio-vision-model' : 'lmstudio-url')
            : (localLlm.reachable ? 'ollama-vision-model' : 'ollama-url')}>
          {localLlm.provider === 'lmstudio' ? 'LM Studio' : 'Ollama'} settings
        </SettingsLink>
      )}
      {optionsOpen && (
        <EnhanceModelPopover model={model} onPick={pick} onClose={() => setOptionsOpen(false)} />
      )}
      {/* w-full so it claims its own line in the flex-wrap toolbars that host
          this button, instead of being crushed between two small controls. */}
      <OllamaFenceNotice fence={fence} onUnload={unloadAndRetry} onStop={stopWaiting}
        className="w-full" />
    </>
  );
}

import { useEffect, useRef, useState } from 'react';
import { apiFetch, postJson, putJson } from '../../api/fetchClient';
import { activeLocalLlm, localLlmLabel } from '../../utils/localLlm.js';
import { pullCopy, visionModelSetting } from '../../utils/watermarkEngine.js';

/* The vision route's model, IN the scan window — the three things the
 * maintainer asked for in one breath: the model that will actually run, named
 * in the open; the installed models to switch to; and a pull / download, so the
 * fix is this window and not a trip to Settings. ONE component, mounted under
 * the shared engine choice, so the dataset dialog and the bank panel cannot
 * drift apart.
 *
 * Saving IS arming: `{provider}.vision_model` is what the vision scan reads on
 * both surfaces and in Settings ▸ Local tools (write-through, exactly like the
 * engine and the threshold beside it). The pull rides the provider-routed
 * /api/local-llm/pull the Settings card uses — Ollama pulls, LM Studio downloads
 * inside itself, same {state, model, progress, error} shape — and a finished
 * pull SELECTS the model it fetched: pulling a model you then have to find in
 * a list is half a feature.
 */
export default function VisionModelPicker({ caps = {}, disabled = false, onModel }) {
  const llm = activeLocalLlm(caps);
  const provider = llm.provider;
  const server = localLlmLabel(caps);
  const copy = pullCopy(provider);
  const [model, setModel] = useState(llm.vision_model || '');
  const [models, setModels] = useState([]);
  const [reachable, setReachable] = useState(true);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [name, setName] = useState('');
  const [job, setJob] = useState(null);      // last pull payload, or null
  const [busy, setBusy] = useState(false);   // a POST in flight
  const timerRef = useRef(null);
  const aliveRef = useRef(true);

  const loadModels = async () => {
    // Always-200 endpoint: an unreachable server is an empty list, never an error.
    const d = await apiFetch('/api/local-llm/models').catch(() => ({ models: [], reachable: false }));
    if (!aliveRef.current) return;
    setModels(d?.models || []);
    setReachable(d?.reachable !== false);
    setLoading(false);
  };

  const save = async (next) => {
    const value = String(next || '').trim();
    if (!value) return;
    setModel(value);
    setSaving(true);
    try {
      await putJson('/api/settings', { config: visionModelSetting(provider, value) });
      onModel?.(value);
    } catch { /* the scan reads the stored value; the select shows intent */ }
    setSaving(false);
  };

  const stop = () => { clearTimeout(timerRef.current); timerRef.current = null; };
  const poll = async () => {
    let s = null;
    try { s = await apiFetch('/api/local-llm/pull', { background: true }); } catch { /* keep the last state */ }
    if (!aliveRef.current) return;
    if (s) setJob(s);
    if (s && s.state === 'running') {
      timerRef.current = setTimeout(poll, 1500);
    } else if (s && s.state === 'done') {
      await loadModels();
      if (s.model) await save(s.model);
    }
  };

  useEffect(() => {
    aliveRef.current = true;
    loadModels();
    // Re-attach: a pull started in Settings (or before a reload) is this
    // install's one pull, and it belongs on this screen too.
    apiFetch('/api/local-llm/pull', { background: true })
      .then((s) => {
        if (!aliveRef.current || !s || s.state === 'idle') return;
        setJob(s);
        if (s.state === 'running') timerRef.current = setTimeout(poll, 1500);
      })
      .catch(() => {});
    return () => { aliveRef.current = false; stop(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [provider]);

  const start = async () => {
    const wanted = name.trim();
    if (!wanted || busy) return;
    setBusy(true);
    try {
      const r = await postJson('/api/local-llm/pull', { model: wanted });
      setJob(r);
      if (r.state === 'running') { stop(); timerRef.current = setTimeout(poll, 1500); }
      else if (r.state === 'done') { await loadModels(); await save(r.model || wanted); }
    } catch (e) {
      setJob({ state: 'error', error: e.message || 'The pull could not start.' });
    } finally { setBusy(false); }
  };

  const running = job?.state === 'running';
  // A model picked before the server went down stays selectable — dropping the
  // user's choice on the floor is worse than offering an unconfirmed name.
  const choices = model && !models.includes(model) ? [model, ...models] : models;
  return (
    <div className="mt-2 space-y-1.5 rounded border border-border bg-app/40 p-2">
      <div>
        <span className="font-medium text-content">Vision model</span>
        {' — this scan runs '}
        <span className="font-mono text-content">{model || 'the loaded model'}</span>
        {` via ${server}. Stored: the other surface and Settings ▸ Local tools read the same value.`}
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <select value={model} disabled={disabled || saving || loading}
          aria-label="Watermark vision model"
          onChange={(e) => save(e.target.value)}
          className="max-w-full rounded border border-border bg-app px-1.5 py-0.5 text-content">
          {choices.length === 0 && (
            <option value="">{loading ? 'Loading models…' : 'No model installed yet'}</option>
          )}
          {choices.map((m) => <option key={m} value={m}>{m}</option>)}
        </select>
        {!reachable && !loading && (
          <span className="text-amber-300">{server} is not answering — the list fills once it does.</span>
        )}
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <input
          type="text" value={name} onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); start(); } }}
          placeholder={copy.placeholder} aria-label={copy.inputLabel}
          disabled={disabled || running}
          className="w-full min-w-[10rem] flex-1 rounded border border-border bg-app px-2 py-1 text-content"
        />
        <button type="button" onClick={start} disabled={disabled || busy || running || !name.trim()}
          className="min-h-10 rounded border border-border px-2.5 py-1 font-semibold text-content-muted hover:bg-surface-raised hover:text-content disabled:opacity-50 lg:min-h-0">
          {running ? `${copy.busy}…` : copy.button}
        </button>
      </div>
      {running && (
        <p role="status" aria-live="polite" className="text-content-muted">
          {copy.busy} <span className="font-mono">{job.model}</span>
          {Number.isFinite(job.progress) ? ` — ${job.progress}%` : '…'}
          {` It runs inside ${server}; leaving this window does not stop it.`}
        </p>
      )}
      {job?.state === 'error' && job.error && (
        <p className="text-rose-300">{job.error}</p>
      )}
      {job?.state === 'done' && job.model && (
        <p className="text-emerald-400">
          ✓ <span className="font-mono">{job.model}</span> is ready and selected for the next scan.
        </p>
      )}
    </div>
  );
}

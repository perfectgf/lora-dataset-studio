/**
 * Which LoRA the clip is rendered with — a trained run, a file already in
 * ComfyUI, or none at all.
 *
 * "None" is a first-class choice and sits at the top on purpose: the only way
 * to know whether a LoRA did anything is to have seen the same seed without it.
 *
 * Two lists rather than one, because they are two different actions. A LoRA in
 * ComfyUI's folder is one click from a clip; a trained checkpoint is a 300 MB
 * copy first, and hiding that behind Generate would make the first launch look
 * like a hang.
 */
import { useCallback, useEffect, useState } from 'react';
import { Download, FlaskConical, RefreshCw } from 'lucide-react';
import { apiFetch, postJson } from '../../../../api/fetchClient';
import { useToast } from '../../../common/Toast';
import { deployUrl, lorasUrl } from './videoStudioApi';

const shortName = (f) => String(f || '').replace(/\\/g, '/').split('/').pop()
  .replace(/\.safetensors$/i, '');

export default function VideoLoraPicker({ value, onChange, strength, onStrength }) {
  const toast = useToast();
  const [deployed, setDeployed] = useState([]);
  const [trained, setTrained] = useState([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const d = await apiFetch(lorasUrl());
      setDeployed(d.deployed || []);
      setTrained(d.trained || []);
    } catch {
      setDeployed([]); setTrained([]);
    } finally {
      setLoading(false);
    }
  }, []);
  useEffect(() => { load(); }, [load]);

  /* Deploying and selecting are ONE gesture. Splitting them would leave the
     user with a copied file and no idea that they still have to pick it. */
  const deployAndPick = async (row) => {
    setBusy(`${row.run_id}:${row.filename}`);
    try {
      const r = await postJson(deployUrl(), {
        run_id: row.run_id, filename: row.filename,
      });
      toast.success(`${shortName(row.filename)} is now loadable by ComfyUI.`);
      onChange({ lora: r.filename, runId: row.run_id, datasetId: row.dataset_id });
      load();
    } catch (e) {
      toast.error(e?.message || 'Could not copy that checkpoint into ComfyUI.');
    } finally {
      setBusy(null);
    }
  };

  const row = (key, label, sub, selected, onClick, trailing) => (
    <button key={key} type="button" onClick={onClick}
      className={`flex w-full items-center gap-2 rounded-lg border px-2 py-1.5 text-left min-h-10 lg:min-h-0 ${
        selected ? 'border-accent bg-accent/10' : 'border-border bg-surface hover:border-accent/50'}`}>
      <span className="min-w-0 flex-1">
        <span className="block truncate text-sm text-content">{label}</span>
        {sub && <span className="block truncate text-[0.6875rem] text-content-subtle">{sub}</span>}
      </span>
      {trailing}
    </button>
  );

  return (
    <section data-probe-panel="video-studio-lora"
      className="flex flex-col gap-1.5 rounded-xl border border-border bg-surface p-2">
      <header className="flex items-center gap-2">
        <h2 className="flex items-center gap-1.5 text-sm font-semibold text-content">
          <FlaskConical aria-hidden="true" className="h-4 w-4" />LoRA under test
        </h2>
        <button type="button" onClick={load} title="Refresh the list"
          className="ml-auto rounded-lg border border-border px-2 py-1 text-content-muted hover:text-content min-h-10 lg:min-h-0">
          <RefreshCw aria-hidden="true" className="h-3.5 w-3.5" />
        </button>
      </header>

      {row('none', 'No LoRA — the base model alone',
        'The comparison point: the same seed without your LoRA.',
        !value, () => onChange({ lora: null, runId: null, datasetId: null }))}

      {trained.length > 0 && (
        <>
          <p className="mt-1 text-[0.6875rem] font-semibold uppercase tracking-wide text-content-subtle">
            Trained here
          </p>
          {trained.map((t) => row(
            `t:${t.run_id}:${t.filename}`, t.label,
            `run #${t.run_id}${t.deployed_as ? '' : ' — not in ComfyUI yet'}`,
            value && value === t.deployed_as,
            () => (t.deployed_as
              ? onChange({ lora: t.deployed_as, runId: t.run_id, datasetId: t.dataset_id })
              : deployAndPick(t)),
            t.deployed_as ? null : (
              <span className="flex shrink-0 items-center gap-1 text-[0.6875rem] text-content-muted">
                <Download aria-hidden="true" className="h-3.5 w-3.5" />
                {busy === `${t.run_id}:${t.filename}` ? 'Copying…' : 'Deploy'}
              </span>
            ),
          ))}
        </>
      )}

      {deployed.length > 0 && (
        <>
          <p className="mt-1 text-[0.6875rem] font-semibold uppercase tracking-wide text-content-subtle">
            Already in ComfyUI
          </p>
          {deployed.map((d) => row(`d:${d.filename}`, d.label, d.filename,
            value === d.filename,
            () => onChange({ lora: d.filename, runId: null, datasetId: null })))}
        </>
      )}

      {!loading && !trained.length && !deployed.length && (
        <p className="text-xs text-content-subtle">
          No video LoRA yet — train one from a video training set, or drop a
          <code className="mx-1">.safetensors</code> into ComfyUI’s
          <code className="mx-1">models/loras/h3</code> folder.
        </p>
      )}

      {value && (
        <label className="mt-1 flex items-center gap-2 text-xs text-content-muted">
          Strength
          <input type="range" min="0" max="2" step="0.05" value={strength}
            onChange={(e) => onStrength(Number(e.target.value))}
            className="min-w-0 flex-1 accent-accent" />
          <span className="w-9 text-right tabular-nums text-content">{Number(strength).toFixed(2)}</span>
        </label>
      )}
      {value && (
        <p className="text-[0.6875rem] text-content-subtle">
          1.3 is where identity came through on the runs measured here; past 2 a
          rank-16 LoRA destroys the shot before it expresses anything.
        </p>
      )}
    </section>
  );
}

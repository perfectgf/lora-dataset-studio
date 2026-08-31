/**
 * Which LoRA the clip is rendered with — a trained run, a file already in
 * ComfyUI, or none at all.
 *
 * "None" is a first-class choice and sits at the top on purpose: the only way
 * to know whether a LoRA did anything is to have seen the same seed without it.
 *
 * THE PANEL IS A DECISION, NOT AN INVENTORY (redesign, 2026-08-31). The first
 * build listed every checkpoint of every run and every file in ComfyUI's h3
 * folder as identical full-width rows — twenty of them on the maintainer's
 * machine, half of them engine parts (turbo distillations, the camera adapter)
 * that the graph grafts by itself — and the controls that actually make a clip
 * sat a screen below. Now:
 *   • a trained RUN is one row and its checkpoints are pills inside it;
 *   • engine parts are folded under their own honest label;
 *   • once a LoRA is chosen the list folds to the choice, its strength and a
 *     "Change" — the picker takes the room of one row.
 *
 * Deploying and selecting stay ONE gesture: a trained checkpoint is a 300 MB
 * copy into ComfyUI first, and hiding that behind Generate would make the
 * first launch look like a hang.
 */
import { useCallback, useEffect, useState } from 'react';
import { ChevronDown, Download, FlaskConical, RefreshCw } from 'lucide-react';
import { apiFetch, postJson } from '../../../../api/fetchClient';
import { useToast } from '../../../common/Toast';
import { deployUrl, lorasUrl } from './videoStudioApi';
import { groupTrained, shortLoraName, splitDeployed } from './videoLoraGroups';

const ROW = 'flex w-full items-center gap-2 rounded-lg border px-2.5 py-1.5 text-left min-h-10 lg:min-h-0';
const ROW_IDLE = 'border-border bg-surface-raised hover:border-accent/50';
const ROW_ON = 'border-accent bg-accent/10';
const PILL = 'rounded-full border px-2 py-0.5 text-[0.6875rem] min-h-10 lg:min-h-0 lg:py-0.5';

export default function VideoLoraPicker({ value, onChange, strength, onStrength }) {
  const toast = useToast();
  const [deployed, setDeployed] = useState([]);
  const [trained, setTrained] = useState([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(null);
  // Folded to the choice once one is made; open again on "Change" — and open
  // from the start when nothing is chosen, since choosing is the panel's job.
  const [open, setOpen] = useState(!value);

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

  const pick = (next) => { onChange(next); setOpen(false); };

  const deployAndPick = async (ck, group) => {
    setBusy(`${group.run_id}:${ck.filename}`);
    try {
      const r = await postJson(deployUrl(), { run_id: group.run_id, filename: ck.filename });
      toast.success(`${shortLoraName(ck.filename)} is now loadable by ComfyUI.`);
      pick({ lora: r.filename, runId: group.run_id, datasetId: group.dataset_id });
      load();
    } catch (e) {
      toast.error(e?.message || 'Could not copy that checkpoint into ComfyUI.');
    } finally {
      setBusy(null);
    }
  };

  const groups = groupTrained(trained);
  const { candidates, parts } = splitDeployed(deployed);
  const selectedName = value ? shortLoraName(value) : 'No LoRA — the base model alone';
  const selectedGroup = value && groups.find((g) => g.checkpoints.some((c) => c.deployed_as === value));

  return (
    <section data-probe-panel="video-studio-lora"
      className="flex flex-col gap-2 rounded-xl border border-border bg-surface p-3">
      <header className="flex items-center gap-2">
        <h2 className="flex items-center gap-1.5 text-sm font-semibold text-content">
          <FlaskConical aria-hidden="true" className="h-4 w-4 text-content-muted" />LoRA under test
        </h2>
        {!open && (
          <button type="button" onClick={() => setOpen(true)}
            className="ml-auto flex items-center gap-1 rounded-lg border border-border px-2.5 py-1 text-xs text-content-muted hover:text-content min-h-10 lg:min-h-0">
            Change <ChevronDown aria-hidden="true" className="h-3.5 w-3.5" />
          </button>
        )}
        {open && (
          <button type="button" onClick={load} title="Refresh the list"
            className="ml-auto rounded-lg border border-border px-2 py-1 text-content-muted hover:text-content min-h-10 lg:min-h-0">
            <RefreshCw aria-hidden="true" className="h-3.5 w-3.5" />
          </button>
        )}
      </header>

      {/* The chosen one, on one row — the whole panel when the list is folded. */}
      {!open && (
        <div className={`${ROW} ${value ? ROW_ON : ROW_IDLE} cursor-default`}>
          <span className="min-w-0 flex-1">
            <span className="block truncate text-sm text-content" title={value || ''}>{selectedName}</span>
            <span className="block truncate text-[0.6875rem] text-content-subtle">
              {value
                ? (selectedGroup ? `trained here — run #${selectedGroup.run_id}` : 'from ComfyUI’s folder')
                : 'The comparison point: the same seed without your LoRA.'}
            </span>
          </span>
        </div>
      )}

      {open && (
        <div className="flex flex-col gap-1.5">
          <button type="button" onClick={() => pick({ lora: null, runId: null, datasetId: null })}
            className={`${ROW} ${!value ? ROW_ON : ROW_IDLE}`}>
            <span className="min-w-0 flex-1">
              <span className="block truncate text-sm text-content">No LoRA — the base model alone</span>
              <span className="block truncate text-[0.6875rem] text-content-subtle">
                The comparison point: the same seed without your LoRA.
              </span>
            </span>
          </button>

          {groups.length > 0 && (
            <p className="mt-1 font-mono text-[0.625rem] uppercase tracking-[0.18em] text-content-subtle">
              Trained here
            </p>
          )}
          {groups.map((g) => {
            const chosen = g.checkpoints.some((c) => c.deployed_as && c.deployed_as === value);
            return (
              <div key={g.run_id}
                className={`flex w-full flex-col gap-1.5 rounded-lg border px-2.5 py-2 ${chosen ? ROW_ON : 'border-border bg-surface-raised'}`}>
                <div className="flex min-w-0 items-baseline gap-2">
                  <span className="min-w-0 flex-1 truncate text-sm text-content">{g.name}</span>
                  <span className="shrink-0 text-[0.6875rem] text-content-subtle">run #{g.run_id}</span>
                </div>
                {/* One pill per checkpoint: the result first, the moments before
                    it after. A pill that is not in ComfyUI yet says so with the
                    icon and copies on click — one gesture, never two. */}
                <div className="flex flex-wrap gap-1.5">
                  {g.checkpoints.map((c) => {
                    const on = c.deployed_as && c.deployed_as === value;
                    const copying = busy === `${g.run_id}:${c.filename}`;
                    return (
                      <button key={c.filename} type="button" title={c.filename}
                        onClick={() => (c.deployed_as
                          ? pick({ lora: c.deployed_as, runId: g.run_id, datasetId: g.dataset_id })
                          : deployAndPick(c, g))}
                        className={`${PILL} flex items-center gap-1 ${
                          on ? 'border-accent bg-accent/15 text-content'
                            : 'border-border text-content-muted hover:border-accent/60 hover:text-content'}`}>
                        {c.final ? 'Final' : `Step ${c.step}`}
                        {!c.deployed_as && (
                          <Download aria-hidden="true" className="h-3 w-3 opacity-70" />
                        )}
                        {copying ? '…' : ''}
                      </button>
                    );
                  })}
                </div>
              </div>
            );
          })}

          {candidates.length > 0 && (
            <p className="mt-1 font-mono text-[0.625rem] uppercase tracking-[0.18em] text-content-subtle">
              Already in ComfyUI
            </p>
          )}
          {candidates.map((d) => (
            <button key={d.filename} type="button" title={d.filename}
              onClick={() => pick({ lora: d.filename, runId: null, datasetId: null })}
              className={`${ROW} ${value === d.filename ? ROW_ON : ROW_IDLE}`}>
              <span className="min-w-0 flex-1">
                <span className="block truncate text-sm text-content">{shortLoraName(d.filename)}</span>
                <span className="block truncate text-[0.6875rem] text-content-subtle">{d.filename}</span>
              </span>
            </button>
          ))}

          {/* Folded, and named for what they are: the graph grafts these itself
              when their option is on. Still selectable — the camera adapter is a
              legitimate thing to render with — but never mistaken for a
              candidate. */}
          {parts.length > 0 && (
            <details className="mt-1 rounded-lg border border-border">
              <summary className="cursor-pointer px-2.5 py-1.5 text-[0.6875rem] text-content-subtle min-h-10 lg:min-h-0 flex items-center">
                Engine parts in the folder ({parts.length}) — turbo, camera, ref2v: grafted by the options, not LoRAs to test
              </summary>
              <div className="flex flex-col gap-1 border-t border-border p-1.5">
                {parts.map((d) => (
                  <button key={d.filename} type="button" title={d.filename}
                    onClick={() => pick({ lora: d.filename, runId: null, datasetId: null })}
                    className={`${ROW} ${value === d.filename ? ROW_ON : 'border-transparent hover:border-border'}`}>
                    <span className="block min-w-0 flex-1 truncate text-xs text-content-muted">{d.filename}</span>
                  </button>
                ))}
              </div>
            </details>
          )}

          {/* No <code> spans in this sentence: they split one full-width paragraph
              into three small islands of ink, which measures — and reads — as an
              empty panel. */}
          {!loading && !groups.length && !deployed.length && (
            <p className="text-xs text-content-subtle">
              No video LoRA yet — train one from a video training set, or drop a
              .safetensors file into ComfyUI’s models/loras/h3 folder.
            </p>
          )}
        </div>
      )}

      {value && (
        <label className="flex items-center gap-2 text-xs text-content-muted">
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

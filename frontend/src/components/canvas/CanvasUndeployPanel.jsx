/**
 * ⏏ Undeploy — every LoRA this app put into ComfyUI, in one list, with a tick
 * box each.
 *
 * WHY IT EXISTS, in the maintainer's words: "a button that lists every deployed
 * LoRA and lets me tick the ones to undeploy, to make it faster." Until now the
 * only way out was a node's popover, one pill at a time, and nothing on any
 * screen said how many were deployed at all.
 *
 * WHAT IT WILL NEVER SHOW is a LoRA the user brought themselves. The list comes
 * from the server's own attribution (`lora_<trigger>` boundary + the cloud runs'
 * staging prefixes), never from a directory scan — because this list feeds a
 * DELETE, and offering someone their own Civitai download for removal, on a
 * screen labelled "undeploy what the app deployed", is the one failure it must
 * not have. `backend/tests/test_deployed_loras_bulk.py` pins that.
 *
 * The grouping, the request and the ledger wording live in the JSX-free
 * `deployedLoras.js` so `node --test` covers them; this file is the shell.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { apiFetch, postJson } from '../../api/fetchClient';
import { useFocusTrap } from '../../hooks/useFocusTrap';
import { useToast } from '../common/Toast';
import {
  deployedSummary, familyLabel, groupByDataset, rowKey, undeployButtonLabel,
  undeployConfirm, undeployItems, undeployOutcome,
} from './deployedLoras.js';

export default function CanvasUndeployPanel({ open, onClose, onChanged }) {
  const dialogRef = useRef(null);
  const requestRef = useRef(0);
  const toast = useToast();
  const [rows, setRows] = useState([]);
  const [keys, setKeys] = useState(() => new Set());
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  useFocusTrap(dialogRef, open);

  const load = useCallback(async () => {
    const request = ++requestRef.current;
    setLoading(true);
    setError(null);
    try {
      const d = await apiFetch('/api/deployed-loras');
      if (request !== requestRef.current) return;
      setRows(d?.deployed || []);
      // The selection is dropped on every (re)load on purpose: ticks made
      // against a list that has since changed would aim at rows that moved.
      setKeys(new Set());
    } catch (err) {
      if (request !== requestRef.current) return;
      setRows([]);
      setError(err?.message || 'Could not read what is deployed. Check your connection and try again.');
    } finally {
      if (request === requestRef.current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!open) return undefined;
    load();
    return () => { requestRef.current += 1; };
  }, [open, load]);

  const toggle = (key) => setKeys((prev) => {
    const next = new Set(prev);
    if (next.has(key)) next.delete(key); else next.add(key);
    return next;
  });
  const allKeys = () => new Set(rows.map(rowKey));

  const run = async () => {
    const items = undeployItems(rows, keys);
    if (!items.length) return;
    if (!window.confirm(undeployConfirm(items.length))) return;
    setBusy(true);
    try {
      const ledger = await postJson('/api/deployed-loras/undeploy', { items });
      const msg = undeployOutcome(ledger);
      toast[msg.type](msg.text);
      // The refusals are NAMED, not just counted: "1 refused" with no file name
      // leaves nothing to act on.
      for (const f of (ledger?.failed || []).slice(0, 3)) {
        toast.error(`${f.filename || 'a file'} — ${f.error || 'refused'}`);
      }
      await load();
      await onChanged?.();
    } catch (err) {
      toast.error(err?.message || 'Could not undeploy — nothing was changed.');
    } finally {
      setBusy(false);
    }
  };

  if (!open) return null;
  const groups = groupByDataset(rows);

  return (
    <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/70 p-3 sm:p-4"
      role="dialog" aria-modal="true" aria-labelledby="canvas-undeploy-title" ref={dialogRef}
      onClick={(event) => { if (event.target === event.currentTarget) onClose(); }}>
      <div className="flex max-h-[calc(100vh-1.5rem)] w-full max-w-xl flex-col gap-3 overflow-hidden rounded-2xl border border-border bg-surface-overlay p-4 shadow-xl sm:max-h-[calc(100vh-2rem)]">
        <div className="flex items-start justify-between gap-3">
          <div>
            <h2 id="canvas-undeploy-title" className="text-sm font-semibold text-content">
              ⏏ Undeploy LoRAs from ComfyUI
            </h2>
            <p className="mt-1 text-[0.6875rem] leading-snug text-content-subtle">
              {deployedSummary(rows, keys.size)}
            </p>
          </div>
          <button type="button" onClick={onClose} aria-label="Close the undeploy panel"
            className="h-8 w-8 shrink-0 rounded-lg border border-border bg-app text-content-muted hover:text-content">
            ×
          </button>
        </div>

        {/* Said once, at the top, because it is what makes ticking freely safe. */}
        <p className="m-0 rounded-lg border border-border bg-app/60 px-3 py-2 text-[0.6875rem] leading-snug text-content-subtle">
          Only LoRAs <b>this app deployed</b> are listed — anything you downloaded into the
          same folder is never touched. Your training saves are kept, so every one of these
          can be deployed again from its checkpoint.
        </p>

        {loading && (
          <p className="m-0 flex items-center gap-2 rounded-lg border border-border bg-app/60 px-3 py-2 text-[0.75rem] text-content-muted" role="status">
            <span className="inline-block h-4 w-4 rounded-full border-2 border-purple-400/40 border-t-purple-400 animate-spin" aria-hidden />
            Reading ComfyUI's loras folders…
          </p>
        )}

        {error && (
          <div className="rounded-lg border border-red-400/40 bg-red-500/10 px-3 py-2 text-[0.75rem] text-red-200" role="alert">
            <p className="m-0">{error}</p>
            <button type="button" onClick={load}
              className="mt-2 rounded border border-red-300/40 px-2 py-1 text-[0.6875rem] font-semibold hover:bg-red-500/10">
              Try again
            </button>
          </div>
        )}

        {!loading && !error && rows.length === 0 && (
          <p className="m-0 rounded-lg border border-amber-400/40 bg-amber-500/10 px-3 py-2 text-[0.75rem] text-amber-200" role="status">
            Nothing to undeploy — this app has no LoRA in ComfyUI's loras folders right now.
          </p>
        )}

        {!loading && !error && rows.length > 0 && (
          <>
            <div className="flex flex-wrap items-center gap-2">
              <button type="button" onClick={() => setKeys(allKeys())} disabled={busy}
                className="rounded-md border border-border px-2 py-1 text-[0.6875rem] font-medium text-content-muted hover:bg-surface-raised hover:text-content disabled:opacity-40">
                Select all ({rows.length})
              </button>
              <button type="button" onClick={() => setKeys(new Set())}
                disabled={busy || keys.size === 0}
                className="rounded-md border border-border px-2 py-1 text-[0.6875rem] font-medium text-content-muted hover:bg-surface-raised hover:text-content disabled:opacity-40">
                Clear
              </button>
            </div>

            <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto">
              {groups.map((group) => (
                <section key={String(group.datasetId)} className="flex flex-col gap-1.5">
                  <h3 className="m-0 text-[0.6875rem] font-semibold uppercase tracking-wide text-content-subtle">
                    {group.datasetName}
                    <span className="ml-1.5 font-normal normal-case tracking-normal opacity-70">
                      ({group.rows.length})
                    </span>
                  </h3>
                  {group.rows.map((row) => {
                    const key = rowKey(row);
                    return (
                      <label key={key}
                        className="flex min-w-0 cursor-pointer items-center gap-2.5 rounded-lg border border-border bg-app/60 px-3 py-2 hover:border-purple-400/50">
                        <input type="checkbox" checked={keys.has(key)} disabled={busy}
                          onChange={() => toggle(key)} className="shrink-0 accent-purple-400" />
                        <span className="min-w-0 flex-1">
                          <span className="block truncate text-[0.8125rem] text-content">
                            {row.label || row.filename}
                          </span>
                          <span className="block truncate text-[0.625rem] text-content-subtle">
                            {familyLabel(row.family)} · {row.filename}
                          </span>
                        </span>
                        {/* The retrofit badge the per-dataset list already shows:
                            a file whose real architecture contradicts its folder. */}
                        {row.arch_mismatch && (
                          <span className="shrink-0 rounded bg-amber-500/20 px-1.5 py-0.5 text-[0.625rem] font-semibold text-amber-200"
                            title="This file's architecture does not match the folder it sits in.">
                            ⚠ {row.arch_label || row.arch_mismatch}
                          </span>
                        )}
                      </label>
                    );
                  })}
                </section>
              ))}
            </div>
          </>
        )}

        <div className="flex flex-col-reverse gap-2 pt-1 sm:flex-row sm:justify-end">
          <button type="button" onClick={onClose} disabled={busy}
            className="rounded-lg border border-border bg-app px-3 py-1.5 text-[0.75rem] text-content-muted hover:text-content disabled:opacity-50">
            Cancel
          </button>
          <button type="button" onClick={run} disabled={busy || keys.size === 0}
            className="rounded-lg border border-amber-400/50 bg-amber-500/15 px-4 py-1.5 text-[0.75rem] font-semibold text-amber-100 disabled:opacity-40">
            {busy ? '⏏ Undeploying…' : undeployButtonLabel(keys.size)}
          </button>
        </div>
      </div>
    </div>
  );
}

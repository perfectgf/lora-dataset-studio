/** 🗃 Import from a bank — a third source next to "📥 Import real photos" and
 *  "🕸 Scrape images from the web".
 *
 *  This is deliberately a BANK picker, not a second image picker: promotion runs
 *  server-side through the normal import path (webp normalization + perceptual
 *  dedup against this dataset), and the bank page already owns the fine-grained
 *  selection. Pick a bank, see how many of its kept images would actually land
 *  HERE, start it. It's a background job (bank_jobs), so we follow the very
 *  snapshot the bank page follows — the one embedded in the banks payload. */
import { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { apiFetch, postJson } from '../../api/fetchClient';
import { useToast } from '../common/Toast';
import {
  bankImportOptions, promotableUrl, promoteUrl, promoteAllBody,
  isBankJobLive, bankActivity, promoteOutcome,
} from './bankImport.js';

const BANK_CURRENT_KEY = 'bankCurrentId';   // BankPage's stored "open this bank" handle

export default function BankImportPanel({ datasetId, onImported, disabled = false }) {
  const toast = useToast();
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const [banks, setBanks] = useState(null);
  const [counts, setCounts] = useState({});
  const [chosen, setChosen] = useState('');
  const [starting, setStarting] = useState(false);
  // The bank whose promote job we are following; null when nothing is in flight.
  const [watching, setWatching] = useState(null);
  const wasLive = useRef(false);

  const loadBanks = useCallback(async () => {
    try {
      const d = await apiFetch('/api/banks');
      setBanks(d.banks || []);
      return d.banks || [];
    } catch (e) {
      toast.error(e?.message || 'Could not load the banks.');
      setBanks([]);
      return [];
    }
  }, [toast]);

  // Counts are per-target, so they are fetched for THIS dataset and refetched
  // whenever the panel reopens (a triage pass on the bank changes them).
  const loadCounts = useCallback(async (rows) => {
    if (!datasetId) return;
    const pairs = await Promise.all((rows || []).map(async (b) => {
      try {
        const d = await apiFetch(promotableUrl(b.id, datasetId));
        return [b.id, Number(d.count || 0)];
      } catch {
        return null;   // a bank that vanished mid-load simply stays "Counting…"
      }
    }));
    setCounts(Object.fromEntries(pairs.filter(Boolean)));
  }, [datasetId]);

  useEffect(() => {
    if (!open) return;
    let live = true;
    setCounts({});
    loadBanks().then((rows) => { if (live) loadCounts(rows); });
    return () => { live = false; };
  }, [open, loadBanks, loadCounts]);

  // ---- follow the background job (same snapshot the bank page polls) --------
  useEffect(() => {
    if (watching == null) return undefined;
    let live = true;
    const tick = async () => {
      let rows;
      try {
        const d = await apiFetch('/api/banks');
        rows = d.banks || [];
      } catch {
        return;   // transient — the next tick retries
      }
      if (!live) return;
      setBanks(rows);
      const act = bankActivity(rows, watching);
      if (isBankJobLive(act)) { wasLive.current = true; return; }
      // Gone or finished: the job is over. Report it with the SERVER's own words.
      if (wasLive.current || act) {
        wasLive.current = false;
        const outcome = promoteOutcome(act);
        if (outcome) toast[outcome.kind]?.(outcome.text);
        onImported?.();
        loadCounts(rows);
      }
      setWatching(null);
    };
    const t = setInterval(tick, 2000);
    tick();
    return () => { live = false; clearInterval(t); };
  }, [watching, toast, onImported, loadCounts]);

  const rows = bankImportOptions(banks, counts);
  const current = rows.find((r) => String(r.id) === String(chosen)) || null;
  const busy = starting || watching != null;

  const start = async () => {
    if (busy || !current?.ready) return;
    setStarting(true);
    try {
      await postJson(promoteUrl(current.id), promoteAllBody(datasetId));
      toast.success('Import started — the images arrive as they are copied.');
      wasLive.current = false;
      setWatching(current.id);
    } catch (e) {
      // postJson THROWS on 400/409 (409 = another job already runs on that bank);
      // without this the click would look like it did nothing.
      toast.error(e?.message || 'Could not start the import from this bank.');
    } finally {
      setStarting(false);
    }
  };

  const openBank = (id) => {
    try { localStorage.setItem(BANK_CURRENT_KEY, String(id)); } catch { /* ignore */ }
    navigate('/bank');
  };

  if (!open) {
    return (
      <button type="button" onClick={() => setOpen(true)} disabled={disabled}
        className="flex w-full items-center gap-2 rounded-lg border border-border bg-surface px-3 py-2 text-left text-content-muted hover:text-content hover:bg-surface-raised transition-colors disabled:opacity-50">
        <span aria-hidden>🗃</span>
        <span className="text-sm font-medium">Import from a bank</span>
        <span className="text-content-subtle text-[0.6875rem]">copy the kept images of a triaged bank into this dataset</span>
        <span aria-hidden className="ml-auto text-content-subtle">→</span>
      </button>
    );
  }

  return (
    <div id="ds-add-bank-import" tabIndex={-1}
      className="scroll-mt-20 flex flex-col gap-2 rounded-lg border border-border bg-surface px-3 py-2">
      <div className="flex items-center gap-2">
        <span aria-hidden>🗃</span>
        <span className="text-sm font-medium text-content">Import from a bank</span>
        <button type="button" onClick={() => setOpen(false)}
          className="ml-auto text-content-subtle text-[0.6875rem] hover:text-content">
          close
        </button>
      </div>
      <p className="text-content-subtle text-[0.6875rem]">
        Only KEPT images are copied, and only those not already here — normalized to webp,
        near-duplicates skipped. The bank and its source folder are left untouched.
      </p>

      {banks == null ? (
        <p className="text-content-subtle text-xs">Loading banks…</p>
      ) : rows.length === 0 ? (
        <p className="text-content-subtle text-xs">
          No bank yet —{' '}
          <button type="button" onClick={() => navigate('/bank')} className="underline hover:text-content">
            create one on the Bank page
          </button>{' '}
          to triage a big folder first.
        </p>
      ) : (
        <>
          <div>
            <label htmlFor="ds-bank-import-select" className="block text-content-muted text-[0.6875rem]">
              Source bank
            </label>
            <select id="ds-bank-import-select" value={chosen} disabled={busy}
              onChange={(e) => setChosen(e.target.value)}
              className="mt-1 w-full rounded-md border border-border bg-surface-raised px-3 py-1.5 text-sm text-content disabled:opacity-50">
              <option value="">Choose a bank…</option>
              {rows.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.name} — {r.count == null ? 'counting…' : `${r.count} to import`} ({r.keep}/{r.total} kept)
                </option>
              ))}
            </select>
          </div>
          {current && (
            <p className={`text-[0.6875rem] ${current.ready ? 'text-emerald-300' : 'text-amber-300'}`}>
              {current.hint}
              {!current.ready && current.reason !== 'loading' && (
                <>
                  {' '}
                  <button type="button" onClick={() => openBank(current.id)}
                    className="underline hover:text-content">
                    Open this bank →
                  </button>
                </>
              )}
            </p>
          )}
          <div className="flex items-center gap-2">
            <button type="button" onClick={start} disabled={busy || !current?.ready}
              className="rounded-md bg-gradient-primary px-3 py-1.5 text-sm font-semibold text-white disabled:opacity-40">
              {watching != null ? 'Importing…' : starting ? 'Starting…' : 'Import'}
            </button>
            {watching != null && (
              <span className="text-content-subtle text-[0.6875rem]">
                running in the background — the grid fills in when it ends
              </span>
            )}
          </div>
        </>
      )}
    </div>
  );
}

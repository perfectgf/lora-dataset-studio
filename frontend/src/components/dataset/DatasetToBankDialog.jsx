import { useEffect, useRef, useState } from 'react';
import { useFocusTrap } from '../../hooks/useFocusTrap';
import { canStartDatasetToBank } from './datasetToBank';

/**
 * Copy a dataset's kept images into a new bank. This stays separate from the
 * workspace because a native prompt cannot explain the preservation choice,
 * retain a failed submission, or keep focus inside a mobile-sized dialog.
 */
export default function DatasetToBankDialog({ datasetName, keptCount, onClose, onStart }) {
  const [name, setName] = useState(datasetName || '');
  const [preserveAnalysis, setPreserveAnalysis] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const dialogRef = useRef(null);
  const nameRef = useRef(null);

  useFocusTrap(dialogRef, true);

  // useFocusTrap restores the invoking button on unmount; Escape and the
  // backdrop share the same busy guard so a live request always has somewhere
  // to report its result.
  useEffect(() => {
    nameRef.current?.select();
    const onKeyDown = (event) => {
      if (event.key === 'Escape' && !busy) onClose();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [busy, onClose]);

  const dismiss = () => { if (!busy) onClose(); };
  const canStart = canStartDatasetToBank({ name, busy });

  const submit = async (event) => {
    event.preventDefault();
    if (!canStart) return;
    setBusy(true);
    setError(null);
    try {
      await onStart({ name: name.trim(), preserveAnalysis });
      // Success navigates away from the workspace. If a host ever keeps this
      // dialog mounted, it can still close it from onStart.
    } catch (err) {
      setError(err?.message || 'Could not create the bank. Please try again.');
      setBusy(false);
    }
  };

  return (
    <div className="fixed inset-0 z-[9991] flex items-center justify-center bg-black/80 p-3 sm:p-4"
      onMouseDown={(event) => { if (event.target === event.currentTarget) dismiss(); }}>
      <form ref={dialogRef} role="dialog" aria-modal="true"
        aria-labelledby="dataset-to-bank-title" aria-describedby="dataset-to-bank-copy-note"
        onSubmit={submit}
        className="flex w-full max-w-md max-h-[90vh] flex-col gap-4 overflow-y-auto rounded-xl border border-border bg-surface-overlay p-4 shadow-2xl sm:p-5">
        <div className="flex items-start justify-between gap-3">
          <div>
            <h2 id="dataset-to-bank-title" className="m-0 text-base font-bold text-content">
              🗃️ Import kept images into a new bank
            </h2>
            <p id="dataset-to-bank-copy-note" className="m-0 mt-1 text-[0.75rem] leading-relaxed text-content-muted">
              {keptCount} kept image{keptCount === 1 ? '' : 's'} will be copied. This dataset stays unchanged.
            </p>
          </div>
          <button type="button" onClick={dismiss} disabled={busy} aria-label="Close import to bank"
            className="shrink-0 rounded-lg border border-border bg-surface px-2.5 py-1.5 text-sm text-content-muted hover:text-content disabled:opacity-40">
            ✕
          </button>
        </div>

        <label className="flex flex-col gap-1.5">
          <span className="text-sm font-medium text-content">Name of the new bank</span>
          <input ref={nameRef} value={name} onChange={(event) => setName(event.target.value)}
            disabled={busy} maxLength={120} placeholder="Candidates" autoComplete="off"
            className="w-full rounded-lg border border-border bg-surface-raised px-3 py-2 text-sm text-content outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-400/25 disabled:opacity-40" />
        </label>

        <fieldset className="flex flex-col gap-2">
          <legend className="text-sm font-medium text-content">Analysis for the copied bank</legend>
          <p className="m-0 text-[0.75rem] leading-relaxed text-content-muted">
            Both choices keep Dataset-owned metadata: captions, curation, framing, watermark and provenance.
          </p>
          <label className={`cursor-pointer rounded-lg border p-3 transition-colors ${preserveAnalysis
            ? 'border-indigo-400/70 bg-indigo-500/10'
            : 'border-border bg-surface-raised hover:bg-surface'}`}>
            <span className="flex items-start gap-2.5">
              <input type="radio" name="dataset-to-bank-analysis" checked={preserveAnalysis}
                disabled={busy} onChange={() => setPreserveAnalysis(true)}
                className="mt-0.5 h-4 w-4 shrink-0 accent-indigo-500" />
              <span>
                <span className="block text-sm font-semibold text-content">Reuse compatible final-file analysis</span>
                <span className="mt-0.5 block text-[0.75rem] leading-relaxed text-content-muted">
                  Restore compatible final-file technical analysis. Face and Score AI results are not reused after a transformed copy.
                </span>
              </span>
            </span>
          </label>
          <label className={`cursor-pointer rounded-lg border p-3 transition-colors ${!preserveAnalysis
            ? 'border-amber-400/70 bg-amber-500/10'
            : 'border-border bg-surface-raised hover:bg-surface'}`}>
            <span className="flex items-start gap-2.5">
              <input type="radio" name="dataset-to-bank-analysis" checked={!preserveAnalysis}
                disabled={busy} onChange={() => setPreserveAnalysis(false)}
                className="mt-0.5 h-4 w-4 shrink-0 accent-amber-500" />
              <span>
                <span className="block text-sm font-semibold text-content">Start fresh analysis</span>
                <span className="mt-0.5 block text-[0.75rem] leading-relaxed text-content-muted">
                  Keep the same Dataset metadata, but skip reuse of prior analysis. Run bank passes when you want fresh analysis.
                </span>
              </span>
            </span>
          </label>
        </fieldset>

        {error && (
          <div role="alert" className="rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-200">
            {error}
            <span className="mt-1 block text-[0.6875rem] text-content-subtle">
              Your name and choice are kept — adjust them and try again.
            </span>
          </div>
        )}

        <div className="flex flex-col-reverse gap-2 pt-1 sm:flex-row sm:justify-end">
          <button type="button" onClick={dismiss} disabled={busy}
            className="rounded-lg border border-border px-3 py-2 text-sm text-content hover:bg-surface-raised disabled:opacity-40">
            Cancel
          </button>
          <button type="submit" disabled={!canStart}
            className="rounded-lg bg-gradient-primary px-4 py-2 text-sm font-semibold text-gray-950 disabled:opacity-40">
            {busy ? 'Starting…' : 'Create bank'}
          </button>
        </div>
      </form>
    </div>
  );
}

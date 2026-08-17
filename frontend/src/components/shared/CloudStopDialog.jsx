/**
 * ⏹ Stop a cloud run — the confirmation, plus the one thing a `window.confirm`
 * could never carry: "and do not rent this machine again".
 *
 * It replaces a native confirm on the Runs hub rather than being added next to
 * it. Two prompts in a row for one decision is worse than the problem it solves,
 * and the consequence sentence — which is genuinely different for a full-model
 * run — deserves better than a `\n\n` inside a string.
 *
 * The wording lives in `pages/cloudStopDialog.js` so `node --test` covers it;
 * this file is the shell. (Ban asked for by mr.arrow on Discord.)
 */
import { useEffect, useRef, useState } from 'react';
import { useFocusTrap } from '../../hooks/useFocusTrap';
import {
  banHostLabel, canBanHost, stopConsequence, stopTitle,
} from '../../pages/cloudStopDialog.js';

export default function CloudStopDialog({ run, fullModel = false, onCancel, onConfirm }) {
  const dialogRef = useRef(null);
  const cancelRef = useRef(null);
  const [banHost, setBanHost] = useState(false);
  useFocusTrap(dialogRef, !!run);

  // Escape cancels, and the focus starts on Cancel — stopping a run costs the
  // remaining steps, so the safe choice is the one under the finger.
  useEffect(() => {
    if (!run) return undefined;
    const onKey = (e) => { if (e.key === 'Escape') onCancel(); };
    window.addEventListener('keydown', onKey);
    cancelRef.current?.focus();
    return () => window.removeEventListener('keydown', onKey);
  }, [run, onCancel]);

  // A fresh dialog must never inherit the previous run's tick.
  useEffect(() => { setBanHost(false); }, [run?.run_id]);

  if (!run) return null;
  const ban = banHostLabel(run);
  const offerBan = canBanHost(run);

  return (
    <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/70 p-3 sm:p-4"
      role="dialog" aria-modal="true" aria-labelledby="cloud-stop-title" ref={dialogRef}
      onClick={(event) => { if (event.target === event.currentTarget) onCancel(); }}>
      <div className="flex w-full max-w-md flex-col gap-3 rounded-2xl border border-border bg-surface-overlay p-4 shadow-xl">
        <h2 id="cloud-stop-title" className="m-0 text-sm font-semibold text-content">
          {stopTitle(run)}
        </h2>
        <p className="m-0 text-[0.75rem] leading-snug text-content-muted">
          {stopConsequence(fullModel)}
        </p>

        {offerBan && (
          <label className="flex cursor-pointer items-start gap-2.5 rounded-lg border border-border bg-app/60 px-3 py-2.5">
            <input type="checkbox" checked={banHost} onChange={() => setBanHost((v) => !v)}
              className="mt-0.5 shrink-0 accent-amber-400" />
            <span className="min-w-0">
              <span className="block text-[0.75rem] font-medium text-content">{ban.label}</span>
              <span className="mt-0.5 block text-[0.6875rem] leading-snug text-content-subtle">
                {ban.detail}
              </span>
            </span>
          </label>
        )}

        <div className="flex flex-col-reverse gap-2 pt-1 sm:flex-row sm:justify-end">
          <button type="button" ref={cancelRef} onClick={onCancel}
            className="rounded-lg border border-border bg-app px-3 py-1.5 text-[0.75rem] text-content-muted hover:text-content">
            Keep it running
          </button>
          <button type="button" onClick={() => onConfirm({ banHost: offerBan && banHost })}
            className="rounded-lg border border-red-400/50 bg-red-500/20 px-4 py-1.5 text-[0.75rem] font-semibold text-red-100">
            ⏹ Stop the run
          </button>
        </div>
      </div>
    </div>
  );
}

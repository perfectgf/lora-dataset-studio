import { useEffect, useRef, useState } from 'react';
import { AlertTriangle, CheckCircle2, XCircle } from 'lucide-react';
import { readinessSignature, overrideAck } from '../../utils/readinessOverride';
import { HelpBadge } from '../../help/HelpMode';
import { readinessRequest, reportFitsLane } from './preflightLane.js';

/*
 * Training-readiness badge mirrors server GET /train/preflight checks and verdict: ready, warnings
 * or blocked, with expandable checks. Failed checks targeting workspace sections offer Fix through
 * onJump. Debounce refetches when relevant curation/caption/leak counters change, not every poll,
 * because preflight reads images from disk for dHash. Render nothing before loading or when the
 * backend gates training with 409 because ai-toolkit is absent.
 */

const VERDICT = {
  ready: { icon: CheckCircle2, label: 'Ready to train', cls: 'border-emerald-400/40 bg-emerald-500/10', iconCls: 'text-emerald-400' },
  warnings: { icon: AlertTriangle, label: 'Almost ready', cls: 'border-amber-400/40 bg-amber-500/10', iconCls: 'text-amber-400' },
  blocked: { icon: XCircle, label: 'Not ready', cls: 'border-red-400/40 bg-red-500/10', iconCls: 'text-red-400' },
};
const ROW_ICON = { ok: '✓', warn: '⚠', fail: '✕' };
const ROW_CLS = { ok: 'text-emerald-400', warn: 'text-amber-300', fail: 'text-red-300' };

// `endpoint` (optional): the preflight URL to read. Default = the image lane's,
// byte for byte, so nothing that already mounts this card changes. The video
// workspace passes its own — the report has the same shape (`checks` +
// `verdict`), and one card rendering both is the parity rule made literal.
// `trainingMode` (optional): a full-model dataset is asked on the cloud lane
// (preflightLane.js readinessRequest); without it, the historical request.
export default function TrainingReadiness({ datasetId, trainType, variant, refreshKey, onJump,
                                            onOverrideChange, endpoint = null, trainingMode = null }) {
  const request = readinessRequest({ datasetId, trainType, variant, trainingMode, endpoint });
  const [data, setData] = useState(null);
  const [open, setOpen] = useState(false);
  // Continue anyway acknowledges a bypassable QUALITY blocker. Uncheck whenever the blocking-state
  // signature changes, preventing a stale acknowledgment from surviving a new blocker, including a
  // physical one.
  const [ack, setAck] = useState(false);
  const timer = useRef(null);
  useEffect(() => {
    let alive = true;
    // Debounce because counters update in bursts during captioning.
    clearTimeout(timer.current);
    timer.current = setTimeout(async () => {
      try {
        const r = await fetch(request.url, { credentials: 'include' });
        if (!r.ok) { if (alive) setData(null); return; }   // 409 AI Toolkit missing → no data
        const d = await r.json();
        if (alive && d.ok) setData(d);
      } catch { /* Transient failure: retry on the next counter change. */ }
    }, 400);
    return () => { alive = false; clearTimeout(timer.current); };
  }, [request.url, refreshKey]);

  // Reset the ack whenever the blocking state changes (a new blocker, a fixed one,
  // an override that just became unavailable) so a stale tick never rides forward.
  const sig = readinessSignature(data);
  useEffect(() => { setAck(false); }, [sig]);
  // Report the server-authoritative ack upward (false unless the override is both
  // offered AND ticked). Runs on null data too → parent clears its state.
  useEffect(() => { onOverrideChange?.(overrideAck(data, ack)); },
    [sig, ack, data, onOverrideChange]);

  if (!data || !(data.checks || []).length) return null;
  // Switching to full-model mode re-asks on the cloud lane; until it answers,
  // the local verdict still in hand is not this dataset's.
  if (!reportFitsLane(data, request.lane)) return null;
  const v = VERDICT[data.verdict] || VERDICT.warnings;
  const warns = data.checks.filter((c) => c.status === 'warn').length;
  const fails = data.checks.filter((c) => c.status === 'fail').length;
  const subtitle = data.verdict === 'ready'
    ? `${data.checks.length} checks passed`
    : [fails && `${fails} blocker(s)`, warns && `${warns} warning(s)`].filter(Boolean).join(' · ');

  return (
    <div className={`rounded-lg border ${v.cls}`}>
      <button type="button" onClick={() => setOpen((o) => !o)} aria-expanded={open}
        className="w-full flex items-center gap-2 px-3 py-2 text-left">
        <v.icon aria-hidden="true" className={`h-4 w-4 shrink-0 ${v.iconCls}`} />
        <span className="text-content text-sm font-semibold">
          {data.verdict === 'ready' ? request.readyLabel : v.label}
        </span>
        <span className="text-content-subtle text-[0.6875rem]">{subtitle}</span>
        <span aria-hidden className="ml-auto text-content-subtle text-xs">{open ? '▾' : '▸'}</span>
      </button>
      {open && (
        <ul className="m-0 px-3 pb-2.5 flex flex-col gap-1 list-none">
          {data.checks.map((c) => (
            <li key={c.id} className="flex items-start gap-2 text-[0.75rem]">
              <span aria-hidden className={`w-4 shrink-0 text-center font-bold ${ROW_CLS[c.status]}`}>
                {ROW_ICON[c.status]}
              </span>
              <span className="text-content">{c.label}</span>
              <span className="text-content-subtle">— {c.detail}</span>
              {c.status !== 'ok' && c.target && (
                <button type="button" onClick={() => onJump?.(c.target)}
                  className="ml-auto shrink-0 px-1.5 py-0.5 rounded border border-border text-content-muted hover:text-content hover:bg-surface-raised text-[0.6875rem]">
                  Fix →
                </button>
              )}
            </li>
          ))}
        </ul>
      )}
      {/*
       * Offer Continue anyway ONLY when every blocker is a bypassable QUALITY safeguard, using
       * data.can_override to mirror the server. Physical impossibilities such as zero images or a
       * missing slider prompt hide the checkbox and still prevent launch. Keep it outside
       * collapsed details while a bypassable blocker exists.
       */}
      {data.can_override && (
        <div className="flex items-start gap-1 px-3 pb-2.5 pt-1 text-[0.75rem]">
          <label className="flex items-start gap-2 cursor-pointer">
            <input type="checkbox" checked={ack} onChange={(e) => setAck(e.target.checked)}
              className="accent-red-400 w-3.5 h-3.5 mt-0.5 shrink-0" />
            <span className="flex flex-col gap-0.5">
              <span className="text-content font-medium">
                Continue anyway — train with these issues unresolved
              </span>
              {data.override_hint && (
                <span className="text-content-subtle">{data.override_hint}</span>
              )}
            </span>
          </label>
          <HelpBadge topic="training-continue-anyway" />
        </div>
      )}
    </div>
  );
}

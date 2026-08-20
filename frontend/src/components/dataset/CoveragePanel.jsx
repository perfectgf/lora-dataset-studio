import { useCallback, useEffect, useState } from 'react';
import { apiFetch } from '../../api/fetchClient';
import {
  axisRows, axisSummary, coverageFilterLabel, coverageReadiness, coverageScope,
  generateMoreHint,
} from './datasetCoverage.js';

/** 🔍 Coverage — the variety read that sits under the composition meter.
 *
 * Collapsed by default: it is a second opinion, not a gate, and the column it
 * lives in is already dense. Opening it fetches once; it refreshes when the
 * dataset payload changes (new images, a caption pass landing).
 *
 * Everything shown is read from work the app already did — the stored shot type
 * and the captions — so opening it costs one query and no GPU. The footer says
 * that out loud, because a panel that looks like analysis and is actually a
 * keyword scan will otherwise be trusted further than it deserves.
 */
const STATE_STYLE = {
  ok: 'border-green-500/40 bg-green-500/10 text-green-300',
  thin: 'border-amber-400/50 bg-amber-400/10 text-amber-300',
  gap: 'border-rose-400/50 bg-rose-400/10 text-rose-300',
  none: 'border-border bg-surface-raised text-content-subtle',
};

function Axis({ axis, onPick }) {
  const rows = axisRows(axis);
  return (
    <div className="flex flex-col gap-1">
      <div className="flex flex-wrap items-baseline gap-x-2">
        <span className="text-content-muted text-[0.6875rem] uppercase tracking-wide">{axis.label}</span>
        {axis.hint && <span className="text-content-subtle text-[0.6875rem]">{axis.hint}</span>}
      </div>
      {/* The sentence is the carrier for a screen reader, never the colour. The
          chips used to be pure decoration on top of it (`aria-hidden` on the row);
          the ones that can now be CLICKED carry their own label instead, because
          an action nobody can reach is not an action. The dead ones stay hidden. */}
      <span className="sr-only">{axisSummary(axis)}</span>
      <div className="flex flex-wrap gap-1">
        {rows.map((r) => {
          // Clickable only when the payload actually named the images: a count
          // with no list behind it (the bank panel, an older server) must stay a
          // plain chip rather than a button that would filter to nothing.
          const pickable = !!onPick && r.count > 0 && !!r.imageIds && r.imageIds.length > 0;
          const cls = `rounded-full border px-2 py-0.5 text-[0.6875rem] ${STATE_STYLE[r.state]}`;
          /* Every chip carries its number, including the zeros. A chip that
             showed the count only when it had one left the absences marked by
             colour alone — and the marker standing in for it read as a typo
             rather than as "none". */
          const body = <>{r.label}<span className="opacity-60"> {r.count}</span></>;
          if (!pickable) {
            return (
              <span key={r.id} aria-hidden
                title={r.count ? `${r.count} caption${r.count === 1 ? '' : 's'} mention this`
                  : (r.state === 'gap' ? 'No caption mentions this' : 'Not mentioned (optional)')}
                className={cls}>{body}</span>
            );
          }
          return (
            <button type="button" key={r.id}
              onClick={() => onPick({
                key: `${axis.id}:${r.id}`,
                label: coverageFilterLabel(axis, r),
                ids: r.imageIds,
              })}
              title={`Show these ${r.count} in the grid — the images whose captions mention this`}
              className={`${cls} hover:brightness-125 focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-400/70`}>
              {body}
            </button>
          );
        })}
      </div>
    </div>
  );
}

export default function CoveragePanel({ datasetId, refreshKey = 0, onPick = null }) {
  const [open, setOpen] = useState(false);
  const [coverage, setCoverage] = useState(null);
  const [error, setError] = useState('');

  const load = useCallback(async () => {
    if (!datasetId) return;
    try {
      setError('');
      setCoverage(await apiFetch(`/api/dataset/${datasetId}/coverage`));
    } catch (e) {
      setCoverage(null);
      setError(e && e.message ? e.message : 'Could not read coverage.');
    }
  }, [datasetId]);

  // Only ever fetched while open: a panel nobody opened should cost nothing.
  useEffect(() => { if (open) load(); }, [open, load, refreshKey]);
  // A different dataset must not show the previous one's numbers for a frame.
  useEffect(() => { setCoverage(null); }, [datasetId]);

  const readiness = coverageReadiness(coverage);
  const hint = generateMoreHint(coverage);

  return (
    <div className="flex flex-col gap-2 rounded-lg border border-border bg-surface px-3 py-2">
      <button type="button" onClick={() => setOpen((v) => !v)} aria-expanded={open}
        className="flex flex-wrap items-center gap-2 text-left">
        <span className="text-content-muted text-[0.6875rem] uppercase tracking-wide">🔍 Coverage</span>
        <span className="text-content-subtle text-[0.6875rem]">
          what the set never shows — read from your captions
        </span>
        <span aria-hidden className="ml-auto text-content-subtle text-[0.6875rem]">{open ? '▲' : '▼'}</span>
      </button>

      {open && (
        <div className="flex flex-col gap-2">
          {error && <p className="m-0 text-amber-300/90 text-[0.6875rem]">⚠ {error}</p>}
          {!error && !readiness.ready && (
            <p className="m-0 text-content-subtle text-[0.6875rem]">{readiness.reason}</p>
          )}
          {!error && readiness.ready && (
            <>
              <p className="m-0 text-content-subtle text-[0.6875rem]">{coverageScope(coverage)}</p>
              {(coverage.advice || []).map((a, i) => (
                <p key={i} className={`m-0 text-[0.6875rem] ${a.tone === 'warn' ? 'text-amber-300/90' : 'text-content-subtle'}`}>
                  {a.tone === 'warn' ? '⚠ ' : '· '}{a.text}
                </p>
              ))}
              {hint && (
                <p className="m-0 text-[0.6875rem] text-emerald-300/90">→ {hint}</p>
              )}
              <div className="flex flex-col gap-2 border-t border-border pt-2">
                {(coverage.axes || []).map((axis) => (
                  <Axis key={axis.id} axis={axis} onPick={onPick} />
                ))}
              </div>
            </>
          )}
          <p className="m-0 text-content-subtle text-[0.6875rem]">
            Reads the same images as the Composition bar above (everything except rejected
            and failed). Advice only — nothing is kept, rejected or changed. This reads the words in your
            captions, not the pixels: a shot the captioner never described is invisible here,
            and “not smiling” still counts as a smile.
            {onPick && ' Click any chip with a count to see exactly those images in the grid.'}
          </p>
        </div>
      )}
    </div>
  );
}

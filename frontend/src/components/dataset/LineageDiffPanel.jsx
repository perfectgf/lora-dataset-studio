import { useEffect, useMemo, useState } from 'react';
import { diffConfigs } from './lineageDetail.js';
import {
  captionWordDiff, datasetChangeChips, datasetIsIdentical, sortedEnvRows, sideLabel,
} from './runCompare.js';
import { apiFetch } from '../../api/fetchClient';

/* The Lab compare panel — opens when exactly TWO run cards are checked for
   compare in the ◉ Graph (or on the LoRA Canvas).

   It used to diff the settings snapshot and stop there, which meant it could
   report "3 captions edited" and never show what they said, call two runs
   identical while the pixels behind unchanged ids had been re-cropped, and blame
   the dataset for a gap that actually came from a `git pull` in ai-toolkit. It
   now asks the backend for the whole freeze of both launches and lays out four
   things:

     · the RECIPE          — the two-column settings table, unchanged rows folded
     · the DATASET         — images added / removed / re-edited, and every edited
                             caption with the words that changed highlighted
     · the MACHINE         — ai-toolkit revision, torch/CUDA, GPU, base-model file
     · what is NOT KNOWN   — a run recorded before snapshots existed says so

   Removed images stay LOOKABLE: their pixels were copied into the deduplicated
   run archive at launch, so the one change that used to be unanswerable ("which
   image did I delete?") shows a thumbnail.

   Full-width drawer below `sm` so it stays readable at 400 px; opaque
   (bg-surface-overlay) so the graph behind never bleeds through. */
export default function LineageDiffPanel({ a, b, onClose }) {
  const [showUnchanged, setShowUnchanged] = useState(false);
  const [open, setOpen] = useState('');       // which image list is expanded
  const [data, setData] = useState(null);
  const [state, setState] = useState('idle'); // idle | loading | ready | error

  const aId = a?.record_id;
  const bId = b?.record_id;

  useEffect(() => {
    if (!aId || !bId) return undefined;
    let alive = true;
    setState('loading');
    setOpen('');
    // apiFetch resolves with the PARSED body (and throws on a non-ok status) —
    // it is not a raw fetch Response.
    apiFetch(`/api/dataset/train/runs/compare?a=${aId}&b=${bId}`)
      .then((d) => { if (alive) { setData(d); setState('ready'); } })
      .catch(() => { if (alive) { setData(null); setState('error'); } });
    return () => { alive = false; };
  }, [aId, bId]);

  // The recipe table prefers the backend's merged config (the settings snapshot
  // PLUS the facts that live on the record row: steps, base model, masked,
  // dataset version). Before it arrives — or if the request fails — it falls
  // back to the node's own `config`, so the panel is never blank.
  const rows = useMemo(
    () => diffConfigs(data?.config?.a || a?.config, data?.config?.b || b?.config),
    [data, a?.config, b?.config],
  );
  if (!a || !b) return null;

  const changedCount = rows.filter((r) => r.changed).length;
  const unchangedCount = rows.length - changedCount;
  const visible = showUnchanged ? rows : rows.filter((r) => r.changed);
  const images = data?.images;
  const chips = datasetChangeChips(images);
  const envRows = sortedEnvRows(data?.env);

  const cell = (v, changed) => (
    v === null
      ? <span className="italic text-content-subtle">—</span>
      : <span className={changed ? 'font-semibold text-amber-100' : 'text-content'}>{v}</span>
  );

  const listFor = (key) => {
    if (key === 'added') return images?.added || [];
    if (key === 'removed') return images?.removed || [];
    if (key === 'caption_changed') return images?.caption_changed || [];
    return images?.content_changed || [];
  };
  const withheldFor = (key) => ((
    key === 'added' ? images?.added_withheld
      : key === 'removed' ? images?.removed_withheld
        : key === 'caption_changed' ? images?.caption_withheld
          : images?.content_withheld) || 0);

  return (
    /* Full screen up to `lg`, a 24-rem side drawer from there. It narrowed at
       `sm` before, which on a 768-px tablet left a 372-px sliver of board beside
       a two-column table — neither side usable. Desktop is unchanged. */
    <div className="fixed right-0 top-0 z-50 flex h-full w-full flex-col overflow-y-auto border-l border-border bg-surface-overlay p-4 shadow-xl lg:w-96">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-content">Compare runs</h3>
        <button type="button" onClick={onClose}
          className="-mr-2 -mt-2 flex h-11 w-11 shrink-0 items-center justify-center text-content-subtle hover:text-content lg:-mr-1 lg:-mt-1 lg:h-8 lg:w-8"
          aria-label="Close">✕</button>
      </div>

      <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
        <div className="rounded-md border border-border bg-app/50 px-2 py-1.5">
          <div className="text-[10px] font-semibold uppercase tracking-wide text-content-subtle">Run A</div>
          <div className="break-words font-mono text-content">
            {data?.a ? sideLabel(data.a) : `#${a.record_id}`}
          </div>
        </div>
        <div className="rounded-md border border-border bg-surface-raised px-2 py-1.5">
          <div className="text-[10px] font-semibold uppercase tracking-wide text-content-subtle">Run B</div>
          <div className="break-words font-mono text-content">
            {data?.b ? sideLabel(data.b) : `#${b.record_id}`}
          </div>
        </div>
      </div>

      {/* --- what this comparison cannot know ----------------------------- */}
      {(data?.notes || []).length > 0 && (
        <ul className="mt-3 space-y-1 rounded-md border border-amber-500/30 bg-amber-500/5 p-2 text-[11px] leading-snug text-amber-100/90">
          {data.notes.map((n) => <li key={n}>{n}</li>)}
        </ul>
      )}

      {/* --- the recipe --------------------------------------------------- */}
      <section className="mt-3">
        <SectionTitle>Training recipe</SectionTitle>
        {rows.length === 0 ? (
          <p className="mt-1 text-xs italic text-content-subtle">
            Neither run recorded its settings, so there's nothing to compare.
          </p>
        ) : (
          <>
            <div className="flex items-center justify-between gap-2">
              <div className="text-[10px] font-semibold uppercase tracking-wide text-content-subtle">
                {changedCount === 0 ? 'No differences' : `${changedCount} change${changedCount > 1 ? 's' : ''}`}
              </div>
              {unchangedCount > 0 && (
                <button type="button"
                  onClick={() => setShowUnchanged((v) => !v)}
                  className="shrink-0 text-[10px] text-content-subtle underline decoration-dotted hover:text-content">
                  {showUnchanged ? 'Hide' : 'Show'} {unchangedCount} unchanged
                </button>
              )}
            </div>
            <table className="mt-1.5 w-full border-collapse text-xs">
              <tbody>
                {visible.map((r) => (
                  <tr key={r.key} className={r.changed ? 'bg-amber-500/10' : 'opacity-60'}>
                    <td className="w-24 py-1 pl-1 pr-2 align-top text-content-subtle">{r.label}</td>
                    <td className="break-all py-1 pr-2 align-top tabular-nums">{cell(r.a, r.changed)}</td>
                    <td className="break-all py-1 pr-1 align-top tabular-nums">{cell(r.b, r.changed)}</td>
                  </tr>
                ))}
                {visible.length === 0 && (
                  <tr>
                    <td colSpan={3} className="py-2 text-center text-content-subtle">
                      These two runs trained with identical settings.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </>
        )}
      </section>

      {/* --- the dataset -------------------------------------------------- */}
      <section className="mt-4">
        <SectionTitle>Dataset</SectionTitle>
        {state === 'loading' && (
          <p className="mt-1 text-xs italic text-content-subtle">Comparing datasets…</p>
        )}
        {state === 'error' && (
          <p className="mt-1 text-xs italic text-content-subtle">
            The dataset comparison could not be loaded.
          </p>
        )}
        {state === 'ready' && images && (
          <>
            <p className="mt-1 text-[11px] text-content-subtle">
              {images.total_a} → {images.total_b} images · {images.kept} in both
            </p>
            {chips.length === 0 ? (
              <p className="mt-1 text-xs text-content-subtle">
                {datasetIsIdentical(data)
                  ? 'Both runs trained on exactly the same images and captions.'
                  : 'No image or caption difference could be established.'}
              </p>
            ) : (
              <div className="mt-1.5 flex flex-wrap gap-1.5">
                {chips.map((c) => (
                  <button key={c.key} type="button"
                    onClick={() => setOpen((v) => (v === c.key ? '' : c.key))}
                    aria-expanded={open === c.key}
                    className={`rounded-full border px-2 py-0.5 text-[11px] transition-colors ${
                      open === c.key
                        ? 'border-amber-400/60 bg-amber-500/20 text-amber-100'
                        : 'border-border bg-app/50 text-content hover:border-amber-400/40'}`}>
                    {c.label}
                  </button>
                ))}
              </div>
            )}
            {open && (
              <ImageList kind={open} items={listFor(open)} withheld={withheldFor(open)} />
            )}
          </>
        )}
      </section>

      {/* --- the machine -------------------------------------------------- */}
      {envRows.length > 0 && (
        <section className="mt-4">
          <SectionTitle>Machine at launch</SectionTitle>
          <table className="mt-1.5 w-full border-collapse text-xs">
            <tbody>
              {envRows.map((r) => (
                <tr key={r.key} className={r.changed ? 'bg-amber-500/10' : 'opacity-60'}>
                  <td className="w-24 py-1 pl-1 pr-2 align-top text-content-subtle">{r.label}</td>
                  <td className="break-all py-1 pr-2 align-top">{cell(r.a, r.changed)}</td>
                  <td className="break-all py-1 pr-1 align-top">{cell(r.b, r.changed)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </div>
  );
}

function SectionTitle({ children }) {
  return (
    <h4 className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-content-subtle">
      {children}
    </h4>
  );
}

/* One expanded list of images. Captions are shown WORD-DIFFED for the edited
   ones — the point of the whole feature — and every entry carries its picture,
   including images that have since been deleted (served from the run archive). */
function ImageList({ kind, items, withheld }) {
  if (!items.length) {
    return <p className="mt-2 text-xs italic text-content-subtle">Nothing to show.</p>;
  }
  return (
    <div className="mt-2 space-y-2">
      {items.map((it) => (
        <div key={it.id} className="rounded-md border border-border bg-app/40 p-2">
          <div className="flex gap-2">
            {it.thumb ? (
              <img src={it.thumb} alt="" loading="lazy"
                className="h-14 w-14 shrink-0 rounded object-cover" />
            ) : (
              <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded border border-dashed border-border p-1 text-center text-[9px] leading-tight text-content-subtle">
                no copy kept
              </div>
            )}
            <div className="min-w-0 flex-1">
              <div className="font-mono text-[10px] text-content-subtle">
                #{it.id}{it.engine ? ` · ${it.engine}` : ''}{it.origin ? ` · ${it.origin}` : ''}
              </div>
              {kind === 'caption_changed' ? (
                it.text_recorded ? (
                  <p className="mt-0.5 break-words text-[11px] leading-snug text-content">
                    {captionWordDiff(it.before, it.after).map((seg, i) => (
                      <span key={`${seg.type}-${i}`}
                        className={seg.type === 'removed'
                          ? 'text-rose-300/90 line-through decoration-rose-400/60'
                          : seg.type === 'added' ? 'font-medium text-emerald-300' : ''}>
                        {seg.text}{' '}
                      </span>
                    ))}
                  </p>
                ) : (
                  <p className="mt-0.5 text-[11px] italic text-content-subtle">
                    The caption changed, but neither run recorded its text.
                  </p>
                )
              ) : (
                it.caption && (
                  <p className="mt-0.5 line-clamp-3 break-words text-[11px] leading-snug text-content-subtle">
                    {it.caption}
                  </p>
                )
              )}
            </div>
          </div>
        </div>
      ))}
      {withheld > 0 && (
        <p className="text-[11px] italic text-content-subtle">
          …and {withheld} more not shown.
        </p>
      )}
    </div>
  );
}

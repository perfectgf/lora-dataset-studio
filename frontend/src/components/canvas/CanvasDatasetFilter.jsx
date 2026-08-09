import { useMemo, useState } from 'react';
import { selectionSummary } from '../../utils/canvasSelection';
import { familyLabel } from '../../utils/canvasFamilyFilter';
import { statusLabel, matchesDatasetQuery } from '../../utils/canvasFilterBar';
import CanvasFilterMenu from './CanvasFilterMenu';

/* Which datasets — and which runs of them — sit on the board.
 *
 * ── What this replaced, and why ─────────────────────────────────────────────
 * This was a PANEL: a fold-out card holding a search row, a status row, a model
 * row and a three-column checkbox list, one line per dataset. Measured on a
 * real library of fourteen datasets at 1280×720 it stood 389 px tall — 54 % of
 * the viewport — sitting directly above the board, on every load, for anyone
 * who had ever left it unfolded. The board, which is the entire point of the
 * page, got what was left. Folding it was possible and was the answer nobody
 * used, because a filter you have to fold away before you can work is a filter
 * that is in the way.
 *
 * It is now a BAR: one wrapping row of chips, ~40 px tall, each opening a
 * popover with the controls that used to be spread across the panel. Nothing
 * was dropped — multi-select, per-dataset run counts, Select all / Clear, the
 * board search, the status and model filters, the pinned-images toggle and
 * Reset are all still here, one click deeper and none of them costing height
 * while unused.
 *
 * ── The two rules a bar has to obey that a panel did not ────────────────────
 *   • A FILTER YOU CANNOT SEE MUST STILL ANNOUNCE ITSELF. Behind a dropdown,
 *     "the board is empty" with no visible cause is a real failure mode, and it
 *     is the one this redesign could have introduced. Every chip therefore
 *     carries its own count at rest and lights up while it is narrowing
 *     anything, and the bar keeps the "N runs shown" readout it always had.
 *   • THE POPOVERS MUST WIN over the board. The section owns a stacking context
 *     (`relative isolate z-20`) and the board frame owns its own — so nothing
 *     drawn on the board, at any z-index, can paint over an open filter.
 *
 * The `lds.canvasFilterOpen` key that remembered the old fold is deliberately
 * left alone rather than deleted: it is stored in real browsers, the helpers
 * that read it are still exported and tested, and a bar has no fold for it to
 * mean anything about. It simply stops being consulted.
 */

export default function CanvasDatasetFilter({
  datasets, selected, onToggle, onAll, onNone,
  families, selectedFamilies, onToggleFamily, onAllFamilies, onNoFamilies,
  query, onQueryChange, statuses, selectedStatuses, onToggleStatus,
  showPinned, onTogglePinned, onResetFilters, visibleRuns,
}) {
  /* The dataset popover's OWN search, which is not the board's.
     Two different questions that used to share one box: "show me only the runs
     that match this" (the board search, still in the bar) and "find the lane I
     want to tick in this list of thirty". Merging them meant scrolling a
     checkbox list to find a dataset while the board silently filtered itself. */
  const [pick, setPick] = useState('');

  const sel = new Set(selected || []);
  const familySel = new Set(selectedFamilies || []);
  const total = (datasets || []).length;
  const shown = useMemo(
    () => (datasets || []).filter((d) => matchesDatasetQuery(d, pick)), [datasets, pick]);

  const statusCount = (statuses || []).length;
  const statusSel = (selectedStatuses || []).length;
  const narrowing = {
    datasets: sel.size < total,
    families: familySel.size < (families || []).length,
    statuses: statusCount > 0 && statusSel < statusCount,
  };
  const anyNarrowing = narrowing.datasets || narrowing.families || narrowing.statuses
    || !showPinned || (query || '').trim().length > 0;

  const box = 'h-4 w-4 shrink-0 accent-indigo-500';
  const row = 'flex min-h-9 cursor-pointer items-center gap-2 rounded-md border '
    + 'border-transparent px-2 py-1 hover:border-border hover:bg-app/50';

  return (
    // `relative isolate z-20`: the bar owns a stacking context of its own and
    // sits above the board's. The frame is isolated too, so the two cannot argue.
    <section data-testid="canvas-dataset-filter"
      aria-label="Canvas filters"
      className="lds-canvas-filter relative isolate z-20 mb-2 flex flex-wrap items-center gap-1.5">

      {/* ── Datasets ─────────────────────────────────────────────────────── */}
      <CanvasFilterMenu label="Datasets" glyph="◧" testId="canvas-filter-datasets"
        summary={selectionSummary(sel.size, total)} active={narrowing.datasets}
        disabled={total === 0}>
        <label className="sr-only" htmlFor="canvas-dataset-pick">Find a dataset</label>
        <input id="canvas-dataset-pick" type="search" value={pick}
          onChange={(e) => setPick(e.target.value)}
          placeholder="Find a dataset…"
          className="mb-1.5 h-9 w-full rounded-md border border-border bg-app/60 px-2.5 text-content text-[0.75rem] placeholder:text-content-subtle focus:border-primary focus:outline-none" />
        <div className="mb-1.5 flex items-center gap-1.5">
          <button type="button" onClick={onAll}
            className="flex h-8 items-center rounded-md border border-border bg-app/60 px-2.5 text-content-muted text-[0.6875rem] hover:text-content">
            Select all
          </button>
          <button type="button" onClick={onNone}
            className="flex h-8 items-center rounded-md border border-border bg-app/60 px-2.5 text-content-muted text-[0.6875rem] hover:text-content">
            Clear
          </button>
          <span className="ml-auto text-content-subtle text-[0.625rem] tabular-nums">
            {sel.size}/{total}
          </span>
        </div>
        {/* The list scrolls INSIDE the popover: a library of thirty datasets
            must not produce a menu taller than the window. */}
        <ul className="m-0 flex max-h-64 list-none flex-col gap-0.5 overflow-y-auto overscroll-contain p-0">
          {shown.map((d) => (
            <li key={d.id}>
              <label className={row}>
                <input type="checkbox" checked={sel.has(d.id)}
                  onChange={() => onToggle(d.id)} className={box} />
                <span className="min-w-0 flex-1 truncate text-content text-[0.75rem]" title={d.name}>
                  {d.name}
                </span>
                {/* The per-dataset run count, kept exactly where it was: it is
                    how you tell a lane worth putting on the board from one
                    with a single run. */}
                <span className="shrink-0 text-content-subtle text-[0.625rem] tabular-nums"
                  title={(d.families || []).map(familyLabel).join(', ')}>
                  {d.runs}
                </span>
              </label>
            </li>
          ))}
        </ul>
        {total > 0 && shown.length === 0 && (
          <p className="m-0 px-2 py-1.5 text-content-subtle text-[0.6875rem]">
            No dataset matches “{pick}”.
          </p>
        )}
        {total === 0 && (
          <p className="m-0 px-2 py-1.5 text-content-subtle text-[0.6875rem]">
            No dataset has been trained yet — the canvas draws training runs, so it
            fills up after your first run finishes.
          </p>
        )}
      </CanvasFilterMenu>

      {/* ── Models ───────────────────────────────────────────────────────── */}
      <CanvasFilterMenu label="Models" glyph="◈" testId="canvas-filter-models"
        summary={`${familySel.size}/${(families || []).length}`}
        active={narrowing.families} disabled={!(families || []).length}>
        <ul className="m-0 flex list-none flex-col gap-0.5 p-0">
          {(families || []).map((family) => (
            <li key={family}>
              <label className={row}>
                <input type="checkbox" checked={familySel.has(family)}
                  onChange={() => onToggleFamily(family)} className={box} />
                <span className="min-w-0 flex-1 truncate text-content text-[0.75rem]">
                  {familyLabel(family)}
                </span>
              </label>
            </li>
          ))}
        </ul>
        <div className="mt-1.5 flex items-center gap-1.5 border-t border-border pt-1.5">
          <button type="button" onClick={onAllFamilies}
            className="h-8 rounded-md px-2 text-content-muted text-[0.6875rem] hover:text-content">
            All models
          </button>
          <button type="button" onClick={onNoFamilies}
            className="h-8 rounded-md px-2 text-content-muted text-[0.6875rem] hover:text-content">
            None
          </button>
        </div>
        {familySel.size === 0 && (
          <p className="m-0 mt-1 text-amber-200/80 text-[0.6875rem]">
            No model selected — your dataset choices are kept, but the board is empty.
          </p>
        )}
      </CanvasFilterMenu>

      {/* ── Status ───────────────────────────────────────────────────────── */}
      {statusCount > 0 && (
        <CanvasFilterMenu label="Status" glyph="◐" testId="canvas-filter-status"
          summary={`${statusSel}/${statusCount}`} active={narrowing.statuses}>
          <ul className="m-0 flex list-none flex-col gap-0.5 p-0">
            {statuses.map((status) => (
              <li key={status}>
                <label className={row}>
                  <input type="checkbox" checked={selectedStatuses.includes(status)}
                    onChange={() => onToggleStatus(status)} className={box} />
                  <span className="min-w-0 flex-1 truncate text-content text-[0.75rem]">
                    {statusLabel(status)}
                  </span>
                </label>
              </li>
            ))}
          </ul>
        </CanvasFilterMenu>
      )}

      {/* ── 🖼 Pinned images: one state, so a chip and not a menu ─────────── */}
      <button type="button" onClick={onTogglePinned}
        aria-pressed={showPinned}
        data-testid="canvas-filter-pinned"
        title={showPinned
          ? 'Pinned images are on the board — click to hide them'
          : 'Pinned images are HIDDEN — click to put them back on the board'}
        className={'flex h-10 items-center gap-1.5 rounded-md border px-2.5 text-[0.75rem] font-semibold lg:h-9 '
          + (showPinned
            ? 'border-border bg-app/60 text-content hover:border-indigo-400/50'
            // Hidden is the state worth shouting about: pinned pictures missing
            // from the board with no visible cause is a bug report.
            : 'border-amber-400/60 bg-amber-500/15 text-amber-100')}>
        <span aria-hidden>🖼</span> Pinned
        {!showPinned && <span className="font-normal">off</span>}
      </button>

      {/* ── The board search. Stays in the bar, at full size: it is the only
             control here that is TYPED, and burying a text field one click
             deep costs more than the 12 rem it occupies. ───────────────── */}
      <label className="sr-only" htmlFor="canvas-filter-search">Search canvas runs</label>
      <input id="canvas-filter-search" type="search" value={query}
        onChange={(e) => onQueryChange(e.target.value)}
        placeholder="Search runs — dataset, ID, model, variant…"
        className={'h-10 min-w-[9rem] flex-1 basis-48 rounded-md border bg-app/60 px-3 text-content text-[0.75rem] placeholder:text-content-subtle lg:h-9 '
          + ((query || '').trim() ? 'border-indigo-400/60' : 'border-border')} />

      <button type="button" onClick={() => { setPick(''); onResetFilters(); }}
        disabled={!anyNarrowing}
        data-testid="canvas-filter-reset"
        title={anyNarrowing ? 'Put every dataset, model and status back on the board'
          : 'Nothing is filtered out'}
        className="flex h-10 items-center rounded-md border border-border px-2.5 text-content-muted text-[0.75rem] hover:text-content disabled:opacity-40 lg:h-9">
        Reset
      </button>

      {/* The readout that makes the whole bar honest: whatever is set, this says
          how much of the library actually reached the board. */}
      <span className="ml-auto shrink-0 text-content-subtle text-[0.6875rem] tabular-nums">
        {visibleRuns} run{visibleRuns === 1 ? '' : 's'} shown
      </span>
    </section>
  );
}

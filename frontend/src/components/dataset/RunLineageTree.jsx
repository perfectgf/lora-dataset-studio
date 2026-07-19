import { useCallback, useState } from 'react';
import { buildLineageRows, resumeCaption } from '../../utils/lineageTree';
import { famLabel, StatusDot, SavesChip } from './lineageChrome';
import RunLineageGraph from './RunLineageGraph';

/* 🌳 A run's lineage — the runs linked by continuations (run → continue →
   re-continue, and forks). Two views of the same genealogy, toggled in the
   header and remembered per browser:
     ☰ List  — a compact indented tree (file-tree rails), dense and scannable.
     ◉ Graph — a left-to-right showcase tree with flowing bezier edges.
   Both stay inside the app's dark design system (surface/border/indigo tokens),
   share the run vocabulary from lineageChrome, and read at a glance in a single
   screenshot. Click any run to jump to its card. */

const INDENT_REM = 1.5;   // width of one connector column
const VIEW_KEY = 'lds.lineageView';   // 'graph' | 'list'
const readView = () => {
  try { return localStorage.getItem(VIEW_KEY) === 'list' ? 'list' : 'graph'; }
  catch { return 'graph'; }
};

/** The file-tree connector gutter for one row: a fixed-width cell per ancestor
 *  column (a continuing vertical rail where guides[i] is true), then the elbow
 *  cell joining this node to its parent (└ last child, ├ otherwise). Pure CSS
 *  hairlines in the border token, so it stays crisp and theme-aware. */
function Connectors({ depth, guides, isLast, dim }) {
  if (depth === 0) return null;
  const rail = dim ? 'bg-border' : 'bg-border-strong';
  return (
    <>
      {guides.slice(0, depth - 1).map((live, i) => (
        <span key={i} className="relative shrink-0" style={{ width: `${INDENT_REM}rem` }}>
          {live && <i aria-hidden className={`absolute top-0 bottom-0 left-1/2 w-px ${rail}`} />}
        </span>
      ))}
      <span className="relative shrink-0" style={{ width: `${INDENT_REM}rem` }}>
        {/* vertical: from the top down to the card's centre; continues past it unless last */}
        <i aria-hidden className={`absolute left-1/2 w-px ${rail}`}
          style={{ top: 0, height: isLast ? '50%' : '100%' }} />
        {/* horizontal elbow into the card */}
        <i aria-hidden className={`absolute top-1/2 h-px ${rail}`}
          style={{ left: '50%', right: '0.15rem' }} />
      </span>
    </>
  );
}

function LineageNode({ row, onSelect, index }) {
  const { node, depth, guides, isLast } = row;
  const cur = node.is_current;
  const dim = node.checkpoint_ready === false;   // a superseded/gone leaf reads quieter
  const clickable = typeof onSelect === 'function';
  return (
    <div className="lds-lineage-node flex items-stretch" style={{ animationDelay: `${Math.min(index, 8) * 35}ms` }}>
      <Connectors depth={depth} guides={guides} isLast={isLast} dim={dim} />
      <div
        role={clickable ? 'button' : undefined}
        tabIndex={clickable ? 0 : undefined}
        onClick={clickable ? () => onSelect(node) : undefined}
        onKeyDown={clickable ? (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelect(node); } } : undefined}
        title={clickable ? 'Jump to this run' : undefined}
        className={'group my-1 flex min-w-0 flex-1 flex-col gap-1 rounded-lg border px-2.5 py-1.5 transition-colors '
          + (cur
            ? 'border-indigo-400/70 bg-indigo-500/10 ring-1 ring-indigo-400/30 '
            : dim
              ? 'border-border bg-app/30 '
              : 'border-border bg-app/50 ')
          + (clickable ? 'cursor-pointer hover:border-indigo-400/60 hover:bg-app/70' : '')}>
        <div className="flex min-w-0 items-center gap-1.5">
          <StatusDot status={node.status} />
          <span className="shrink-0 font-mono text-content-muted text-[0.625rem]">
            <span aria-hidden>{node.source === 'cloud' ? '☁' : '💻'}</span>{' '}
            #{node.source === 'cloud' && node.run_id ? node.run_id : node.record_id}
          </span>
          <span className={`min-w-0 truncate text-[0.75rem] font-semibold ${dim ? 'text-content-muted' : 'text-content'}`}
            title={`${famLabel(node.train_type)}${node.variant ? ` · ${node.variant}` : ''}`}>
            {famLabel(node.train_type)}{node.variant ? <span className="font-normal text-content-muted"> · {node.variant}</span> : null}
          </span>
          {cur && (
            <span className="shrink-0 rounded-full bg-indigo-500/25 px-1.5 py-0.5 text-indigo-100 text-[0.5rem] font-bold uppercase tracking-wider">
              this run
            </span>
          )}
          <span className="ml-auto shrink-0"><SavesChip node={node} /></span>
        </div>
        <div className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-0.5 text-content-subtle text-[0.5625rem]">
          {node.version != null && (
            <span className="rounded bg-app/60 px-1 py-px font-medium text-content-muted">v{node.version}</span>
          )}
          {node.steps ? <span className="tabular-nums">{node.steps.toLocaleString()} steps</span> : null}
          {resumeCaption(node) && (
            <span className="inline-flex items-center gap-0.5 text-content-subtle">
              <span aria-hidden className="text-[0.625rem] leading-none">↳</span>{resumeCaption(node)}
            </span>
          )}
          {node.origin_unknown && (
            <span className="italic" title="This run resumed from an earlier checkpoint, but its source run predates lineage tracking">
              origin not recorded
            </span>
          )}
          {node.has_superseded_tail && (
            <span className="inline-flex items-center gap-0.5 text-amber-300/70"
              title="A later run resumed from an earlier step of this one — its subsequent saves were set aside on disk (kept, never deleted)">
              <span aria-hidden>⋯</span>set-aside saves
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

/** The indented-tree body (☰ List). */
function LineageList({ rows, onSelect }) {
  return (
    <div className="flex min-w-fit flex-col">
      {rows.map((row, i) => (
        <LineageNode key={row.node.record_id} row={row} onSelect={onSelect} index={i} />
      ))}
    </div>
  );
}

/** Segmented ☰ List / ◉ Graph switch. */
function ViewToggle({ view, onChange }) {
  const opt = (id, glyph, label) => (
    <button type="button" onClick={() => onChange(id)}
      aria-pressed={view === id} title={`${label} view`}
      className={'inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[0.5625rem] font-semibold transition-colors '
        + (view === id
          ? 'bg-indigo-500/20 text-indigo-100 '
          : 'text-content-subtle hover:text-content')}>
      <span aria-hidden>{glyph}</span>{label}
    </button>
  );
  return (
    <div className="flex items-center gap-0.5 rounded-lg border border-border bg-app/40 p-0.5">
      {opt('list', '☰', 'List')}
      {opt('graph', '◉', 'Graph')}
    </div>
  );
}

export default function RunLineageTree({ tree, loading, error, onSelect, onContinueCheckpoint }) {
  const [view, setView] = useState(readView);
  const changeView = useCallback((v) => {
    setView(v);
    try { localStorage.setItem(VIEW_KEY, v); } catch { /* private mode: keep it in memory */ }
  }, []);

  if (loading) {
    return (
      <div className="lds-lineage-in flex items-center gap-2 rounded-lg border border-border bg-surface px-3 py-2 text-content-subtle text-[0.6875rem]">
        <span aria-hidden className="h-3 w-3 animate-spin rounded-full border-2 border-border-strong border-t-indigo-400" />
        Resolving lineage…
      </div>
    );
  }
  if (error) return <p className="m-0 text-rose-300/80 text-[0.6875rem]">{error}</p>;
  const rows = buildLineageRows(tree);
  if (!rows.length) return null;
  return (
    <div className="lds-lineage-in overflow-x-auto rounded-xl border border-border bg-surface p-2.5">
      <div className="mb-1.5 flex items-center gap-2 px-0.5">
        <span aria-hidden className="text-[0.8125rem] leading-none">🌳</span>
        <span className="text-content text-[0.6875rem] font-semibold">Lineage</span>
        <span className="rounded-full bg-app/60 px-1.5 py-0.5 text-content-muted text-[0.5625rem] font-medium">
          {rows.length} run{rows.length > 1 ? 's' : ''}
        </span>
        <div className="ml-auto flex items-center gap-2">
          <span className="hidden items-center gap-2 text-content-subtle text-[0.5rem] sm:flex">
            <span className="inline-flex items-center gap-1">
              <span aria-hidden className="h-2 w-2 rounded-full border border-indigo-400/70 bg-indigo-500/20" />current
            </span>
            <span className="inline-flex items-center gap-1">
              <span aria-hidden>↳</span>continued from
            </span>
          </span>
          <ViewToggle view={view} onChange={changeView} />
        </div>
      </div>
      {view === 'graph'
        ? <RunLineageGraph tree={tree} onSelect={onSelect} onContinueCheckpoint={onContinueCheckpoint} />
        : <LineageList rows={rows} onSelect={onSelect} />}
    </div>
  );
}

import { buildLineageRows, resumeCaption } from '../../utils/lineageTree';

/* 🌳 Genealogy tree of a run's lineage — the runs linked by continuations
   (run → continue → re-continue, and forks). A light, polished indented tree
   (no graph library): file-tree connector rails draw who descends from whom,
   each run is a compact card, the current run wears a primary accent, and a
   branch resumed from an earlier step is greyed and marks its parent's set-aside
   saves. Designed to read at a glance in a single screenshot — no hover or
   scroll needed to understand it. Stays inside the app's dark design system
   (surface/border/indigo accent tokens). */

const FAMILY_LABEL = { zimage: 'Z-Image', krea: 'Krea 2', sdxl: 'SDXL', flux: 'FLUX.1', flux2klein: 'FLUX.2 Klein' };
const famLabel = (f) => FAMILY_LABEL[f] || f || 'LoRA';

const INDENT_REM = 1.5;   // width of one connector column

const STATUS_TONE = {
  done: 'bg-emerald-400',
  error: 'bg-rose-400',
  error_pod_kept: 'bg-amber-400',
};
function StatusDot({ status }) {
  const tone = STATUS_TONE[status] || (status ? 'bg-sky-400' : 'bg-content-subtle');
  return (
    <span aria-hidden title={status || 'no recorded status'}
      className={`h-2 w-2 shrink-0 rounded-full ${tone} ${status === 'done' ? 'shadow-[0_0_6px] shadow-emerald-400/50' : ''}`} />
  );
}

/** LoRA/checkpoint availability chip: on-disk vs gone (superseded aside or
 *  deleted). null availability (a scan we couldn't run) shows nothing. */
function SavesChip({ node }) {
  if (node.checkpoint_ready === true) {
    const n = node.saves;
    return (
      <span className="inline-flex items-center gap-1 rounded-full border border-emerald-400/40 bg-emerald-500/10 px-1.5 py-0.5 text-emerald-200 text-[0.5625rem] font-medium"
        title={n ? `${n} checkpoint${n > 1 ? 's' : ''} still on disk` : 'LoRA on disk'}>
        <span aria-hidden>💾</span>{n ? `${n} on disk` : 'on disk'}
      </span>
    );
  }
  if (node.checkpoint_ready === false) {
    return (
      <span className="inline-flex items-center rounded-full border border-border px-1.5 py-0.5 text-content-subtle text-[0.5625rem] font-medium"
        title="This run's checkpoint is no longer on disk (set aside by a later resume, or deleted)">
        gone
      </span>
    );
  }
  return null;
}

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

export default function RunLineageTree({ tree, loading, error, onSelect }) {
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
        <span className="ml-auto flex items-center gap-2 text-content-subtle text-[0.5rem]">
          <span className="inline-flex items-center gap-1">
            <span aria-hidden className="h-2 w-2 rounded-full border border-indigo-400/70 bg-indigo-500/20" />current
          </span>
          <span className="inline-flex items-center gap-1">
            <span aria-hidden>↳</span>continued from
          </span>
        </span>
      </div>
      <div className="flex min-w-fit flex-col">
        {rows.map((row, i) => (
          <LineageNode key={row.node.record_id} row={row} onSelect={onSelect} index={i} />
        ))}
      </div>
    </div>
  );
}

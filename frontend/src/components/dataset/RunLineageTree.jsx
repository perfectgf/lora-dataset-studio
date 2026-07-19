import { buildLineageRows, resumeCaption } from '../../utils/lineageTree';

/* 🌳 Genealogy tree of a run's lineage — the runs linked by continuations
   (run → continue → re-continue, and forks). Rendered as a light indented tree
   (no graph library): one row per run, children indented under the run they
   resumed from, the current run highlighted, and a discreet grey note where a
   parent has saves set aside by a resume-from-an-earlier-step (superseded). */

const FAMILY_LABEL = { zimage: 'Z-Image', krea: 'Krea 2', sdxl: 'SDXL', flux: 'FLUX.1', flux2klein: 'FLUX.2 Klein' };
const famLabel = (f) => FAMILY_LABEL[f] || f || 'LoRA';

function StatusDot({ status }) {
  const color = status === 'done' ? 'bg-emerald-400'
    : (status === 'error' || status === 'error_pod_kept') ? 'bg-rose-400'
    : status ? 'bg-sky-400' : 'bg-border-strong';
  return <span aria-hidden className={`h-1.5 w-1.5 shrink-0 rounded-full ${color}`}
    title={status || 'no recorded status'} />;
}

/** LoRA/checkpoint availability chip: on-disk vs gone (superseded aside or
 *  deleted). null availability (a disk scan we couldn't run) shows nothing —
 *  never a wrong claim. */
function SavesChip({ node }) {
  if (node.checkpoint_ready === true) {
    return (
      <span className="rounded border border-emerald-400/40 bg-emerald-500/10 px-1 py-0.5 text-emerald-200 text-[0.5625rem]"
        title={node.saves ? `${node.saves} checkpoint${node.saves > 1 ? 's' : ''} on disk` : 'LoRA on disk'}>
        💾 on disk{node.saves ? ` ·${node.saves}` : ''}
      </span>
    );
  }
  if (node.checkpoint_ready === false) {
    return (
      <span className="rounded border border-border bg-surface px-1 py-0.5 text-content-subtle text-[0.5625rem]"
        title="This run's checkpoint is no longer on disk (set aside by a later resume, or deleted)">
        gone
      </span>
    );
  }
  return null;
}

function LineageNode({ row, onSelect }) {
  const { node, depth } = row;
  const cur = node.is_current;
  const clickable = typeof onSelect === 'function';
  return (
    <div className="flex items-start gap-1.5" style={{ paddingLeft: `${depth * 1.1}rem` }}>
      {depth > 0 && (
        <span aria-hidden className="mt-1.5 select-none text-content-subtle text-[0.625rem] leading-none">↳</span>
      )}
      <div
        role={clickable ? 'button' : undefined}
        tabIndex={clickable ? 0 : undefined}
        onClick={clickable ? () => onSelect(node) : undefined}
        onKeyDown={clickable ? (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelect(node); } } : undefined}
        title={clickable ? 'Jump to this run' : undefined}
        className={'flex min-w-0 flex-1 flex-col gap-0.5 rounded-md border px-2 py-1 '
          + (cur
            ? 'border-indigo-400/60 bg-indigo-500/10 '
            : 'border-border bg-app/40 ')
          + (clickable ? 'cursor-pointer hover:border-indigo-400/50' : '')}>
        <div className="flex flex-wrap items-center gap-x-1.5 gap-y-0.5">
          <StatusDot status={node.status} />
          <span className="font-mono text-content-muted text-[0.625rem]">
            {node.source === 'cloud' ? '☁' : '💻'} #{node.source === 'cloud' && node.run_id ? node.run_id : node.record_id}
          </span>
          {cur && (
            <span className="rounded bg-indigo-500/20 px-1 py-0.5 text-indigo-200 text-[0.5625rem] font-semibold uppercase">
              this run
            </span>
          )}
          <span className="text-content text-[0.6875rem] font-semibold">
            {famLabel(node.train_type)}{node.variant ? ` · ${node.variant}` : ''}
          </span>
          {node.version != null && (
            <span className="text-content-subtle text-[0.5625rem]">v{node.version}</span>
          )}
          {node.steps ? <span className="tabular-nums text-content-muted text-[0.5625rem]">{node.steps} steps</span> : null}
          <SavesChip node={node} />
        </div>
        <div className="flex flex-wrap items-center gap-x-1.5 text-content-subtle text-[0.5625rem]">
          {resumeCaption(node) && <span>{resumeCaption(node)}</span>}
          {node.origin_unknown && (
            <span className="italic" title="This run resumed from an earlier checkpoint, but its source run predates lineage tracking">
              origin not recorded
            </span>
          )}
          {node.has_superseded_tail && (
            <span className="text-content-subtle/80" title="A later run resumed from an earlier step of this one — its subsequent saves were set aside (kept on disk, never deleted)">
              · has set-aside saves
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

export default function RunLineageTree({ tree, loading, error, onSelect }) {
  if (loading) return <p className="m-0 text-content-subtle text-[0.6875rem]">Loading lineage…</p>;
  if (error) return <p className="m-0 text-rose-300/80 text-[0.6875rem]">{error}</p>;
  const rows = buildLineageRows(tree);
  if (!rows.length) return null;
  return (
    <div className="flex flex-col gap-1 rounded-lg border border-border bg-surface p-2">
      <p className="m-0 text-content-subtle text-[0.5625rem] uppercase tracking-wide">
        Lineage · {rows.length} run{rows.length > 1 ? 's' : ''}
      </p>
      {rows.map((row) => (
        <LineageNode key={row.node.record_id} row={row} onSelect={onSelect} />
      ))}
    </div>
  );
}

/* Shared visual vocabulary for a lineage run — used by BOTH the ☰ List
   (RunLineageTree) and the ◉ Graph (RunLineageGraph) views so a run reads the
   same in either: its family label, its status dot, its on-disk/gone chip. Kept
   in one place so the two views can never drift apart. */

export const FAMILY_LABEL = {
  zimage: 'Z-Image', krea: 'Krea 2', sdxl: 'SDXL',
  flux: 'FLUX.1', flux2klein: 'FLUX.2 Klein', anima: 'Anima',
};
export const famLabel = (f) => FAMILY_LABEL[f] || f || 'LoRA';

const STATUS_TONE = {
  done: 'bg-emerald-400',
  error: 'bg-rose-400',
  error_pod_kept: 'bg-amber-400',
};

/** Run status as a small dot; a finished run gets a soft emerald halo. */
export function StatusDot({ status }) {
  const tone = STATUS_TONE[status] || (status ? 'bg-sky-400' : 'bg-content-subtle');
  return (
    <span aria-hidden title={status || 'no recorded status'}
      className={`h-2 w-2 shrink-0 rounded-full ${tone} ${status === 'done' ? 'shadow-[0_0_6px] shadow-emerald-400/50' : ''}`} />
  );
}

/** LoRA/checkpoint availability chip: on-disk vs gone (superseded aside or
 *  deleted). A null availability (a scan we couldn't run) shows nothing. */
export function SavesChip({ node }) {
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

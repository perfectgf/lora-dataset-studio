/* Two DELIBERATELY distinct badge families so runs and dataset versions are
   never confused again:

   • RunIdChip — a rounded-full PILL carrying a training run's stable id
     (☁ #49 for a cloud run, 💻 #12 for a local one). This is the identity the
     Runs page and the Checkpoints panel share so "this final" ties back to
     "the run that produced it".
   • DatasetVersionChip — a rounded-RECTANGLE tag for the dataset version the
     run trained on (v1/v2). Different shape + hue = a different notion.

   Both are pure presentational helpers reused by CloudRunsPage and
   TrainingPanel. The pure identity/deep-link helpers live in
   utils/runIdentity.js (framework-free, unit-tested). */

export function RunIdChip({ source, id, className = '' }) {
  if (id == null) return null;
  const cloud = source === 'cloud';
  return (
    <span
      title="Training run id — the run that produced this LoRA"
      className={'inline-flex items-center gap-1 rounded-full border px-1.5 py-0.5 '
        + 'text-[0.625rem] font-semibold tabular-nums '
        + (cloud
          ? 'border-sky-400/50 bg-sky-500/10 text-sky-200'
          : 'border-violet-400/50 bg-violet-500/10 text-violet-200')
        + (className ? ` ${className}` : '')}>
      <span aria-hidden>{cloud ? '☁' : '💻'}</span>#{id}
    </span>
  );
}

export function DatasetVersionChip({ version, className = '' }) {
  if (version == null) return null;
  return (
    <span
      title="Dataset version at training time"
      className={'inline-flex items-center rounded border border-border '
        + 'bg-surface-raised px-1.5 py-0.5 text-[0.625rem] text-content-subtle '
        + (className ? ` ${className}` : '')}>
      v{version}
    </span>
  );
}

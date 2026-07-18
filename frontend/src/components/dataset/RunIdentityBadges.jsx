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

/* BaseModelChip — the REAL base a run trained on: the family's official base
   spelled out (e.g. "Z-Image Turbo") or the custom checkpoint's filename/tag
   (e.g. "bigLove_zt3.safetensors"). A custom base is tinted amber to read as
   "not an official base at a glance"; the leaf name is truncated with the full
   value in the title. `label` is the object from runBaseModelLabel() — render
   nothing when it's null (a legacy run that never recorded its base). */
export function BaseModelChip({ label, className = '' }) {
  if (!label) return null;
  return (
    <span
      title={label.title}
      className={'inline-flex min-w-0 max-w-[11rem] items-center gap-1 rounded border '
        + 'px-1.5 py-0.5 text-[0.625rem] '
        + (label.custom
          ? 'border-amber-400/40 bg-amber-500/10 text-amber-200'
          : 'border-border bg-surface-raised text-content-subtle')
        + (className ? ` ${className}` : '')}>
      <span aria-hidden>🧩</span>
      <span className="truncate">{label.text}</span>
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

import { useEffect, useState } from 'react';
import { Lightbulb } from 'lucide-react';

const LS_KEY = 'guidedCardCollapsed';

const COPY = {
  reference: 'Add the reference photo — one clear face shot is enough to start.',
  generate: 'Generate ~25 varied shots from the reference (or import real photos).',
  curate: 'Review the grid: keep ✓ the good ones, reject ✕ the rest.',
  caption: 'Caption the kept images — required for training.',
  finish_export: 'Your dataset is ready — export the ZIP for training.',
  finish_train: 'Your dataset is ready — launch a LoRA training.',
  studio: 'Compare checkpoints in the Test Studio and pick the best one.',
};

export default function NextStepCard({ step, trainMode, busy, totalImages, onAction, actionLabel }) {
  const [collapsed, setCollapsed] = useState(() => localStorage.getItem(LS_KEY) === '1');
  const forceOpen = totalImages === 0;
  const open = forceOpen || !collapsed;
  useEffect(() => { if (!forceOpen) localStorage.setItem(LS_KEY, collapsed ? '1' : '0'); },
    [collapsed, forceOpen]);
  if (!step) return null;
  const key = step.id === 'finish' ? (trainMode ? 'finish_train' : 'finish_export') : step.id;

  /* ONE ROW: the step, what it asks, the button that does it, the fold. It was
     a two-storey card (title, then sentence and button) above every section —
     78 px before the section's own title. On a row the same words take 44 px;
     where there is no room for a row (a phone) the sentence and the button wrap
     under the step, which is the old card's shape. Folding keeps the step's
     name and drops the rest, as before. */
  return (
    <div role="status"
      className="flex flex-wrap items-center gap-x-3 gap-y-1.5 rounded-lg border border-border bg-surface-raised px-3 py-2">
      <span className="flex items-center gap-2">
        <Lightbulb aria-hidden="true" className="h-4 w-4 shrink-0 text-primary" />
        <span className="text-content text-sm font-semibold">Next step: {step.label}</span>
      </span>
      {open && <p className="m-0 min-w-0 flex-1 basis-64 text-sm text-content-muted">{COPY[key]}</p>}
      <span className="ml-auto flex items-center gap-2">
        {open && (
          <button type="button" onClick={onAction} disabled={busy}
            className="px-3 py-1.5 rounded-lg bg-gradient-primary text-gray-950 text-sm font-semibold disabled:opacity-40">
            {actionLabel}
          </button>
        )}
        {!forceOpen && (
          <button type="button" onClick={() => setCollapsed((v) => !v)} aria-expanded={open}
            className="text-content-subtle hover:text-content px-1"
            title={open ? 'Collapse the guide' : 'Expand the guide'}>
            <span aria-hidden>{open ? '▾' : '▸'}</span>
            <span className="sr-only">{open ? 'Collapse' : 'Expand'} next-step guide</span>
          </button>
        )}
      </span>
    </div>
  );
}

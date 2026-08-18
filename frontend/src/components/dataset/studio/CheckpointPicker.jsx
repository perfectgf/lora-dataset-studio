// Sélecteur des checkpoints à tester (cases à cocher multi-sélection).
// Mine = this dataset's trigger-matched epochs. Theirs = guest files from
// models/loras, compared as their own cells (same prompt/seed), not stacked.
import { useState } from 'react';
import { HelpBadge } from '../../../help/HelpMode';
import KleinLoraCombobox, { useKleinGenerationLoras } from '../../settings/KleinLoraCombobox';
import { FAMILY_LABELS } from './constants';
import { MAX_GUEST_CHECKPOINTS } from '../../../utils/studioGuestCheckpoints';

export default function CheckpointPicker({
  checkpoints, chosen, onToggle,
  guests = [], onToggleGuest, onAddGuest, onRemoveGuest,
  family = 'zimage',
}) {
  const { loras, loading, error, rescan, rescanning } = useKleinGenerationLoras(family);
  const [pickText, setPickText] = useState('');
  // Start open when guests are already on the form (persisted) so a launch
  // cannot silently include files the accordion was hiding. Empty stays folded.
  const [theirsOpen, setTheirsOpen] = useState(guests.length > 0);
  const atCap = guests.length >= MAX_GUEST_CHECKPOINTS;
  const engineLabel = FAMILY_LABELS[family] || family;
  const add = () => {
    const fn = pickText.trim();
    if (!fn || atCap) return;
    onAddGuest?.(fn);
    setPickText('');
  };

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-col gap-1">
        <span className="text-content-muted text-[0.625rem] uppercase">Checkpoints to test</span>
        <div className="flex flex-wrap gap-2">
          {checkpoints.map((c) => (
            <label key={c.filename} className="flex items-center gap-1.5 px-2 py-1 rounded-lg border border-border bg-surface cursor-pointer text-[0.75rem] text-content">
              <input type="checkbox" checked={chosen.includes(c.filename)}
                onChange={() => onToggle(c.filename)} aria-label={`Test ${c.label}`} />
              {c.label}
            </label>
          ))}
        </div>
      </div>

      <details
        className="rounded-lg border border-cyan-400/30 bg-cyan-500/5 px-2 py-1.5"
        open={theirsOpen}
        onToggle={(e) => setTheirsOpen(e.currentTarget.open)}
      >
        <summary className="flex cursor-pointer select-none items-center gap-1.5 text-[0.6875rem] font-medium text-cyan-200">
          Compare with other LoRAs
          {guests.length > 0 && (
            <span className="font-normal text-cyan-100/80">({guests.length})</span>
          )}
          <HelpBadge topic="studio-guest-checkpoints" />
        </summary>
        <div className="mt-1.5 flex flex-col gap-1.5">
          <span className="text-content-subtle text-[0.625rem]">
            Compare a LoRA you did not train here — same prompt and seed, its own row.
          </span>
          {guests.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {guests.map((g) => (
                <label key={g.filename}
                  className="flex items-center gap-1.5 px-2 py-1 rounded-lg border border-cyan-400/40 bg-cyan-500/10 cursor-pointer text-[0.75rem] text-cyan-100">
                  <input type="checkbox" checked={chosen.includes(g.filename)}
                    onChange={() => onToggleGuest(g.filename)}
                    aria-label={`Test ${g.label}`} />
                  <span className="truncate max-w-[12rem]" title={g.filename}>{g.label}</span>
                  <button type="button"
                    onClick={(e) => { e.preventDefault(); e.stopPropagation(); onRemoveGuest(g.filename); }}
                    title="Remove this LoRA from the comparison"
                    aria-label={`Remove ${g.label}`}
                    className="leading-none text-content-subtle hover:text-red-300">×</button>
                </label>
              ))}
            </div>
          )}
          <KleinLoraCombobox value={pickText} onChange={setPickText}
            ariaLabel="LoRA file trained elsewhere" loras={loras} loading={loading}
            error={error} rescan={rescan} rescanning={rescanning}
            engineLabel={engineLabel} placeholder="path/to/lora.safetensors" />
          <button type="button" onClick={add}
            disabled={!pickText.trim() || atCap}
            className="rounded-md border border-cyan-400/50 bg-cyan-500/10 px-2 py-1.5 text-[0.6875rem] font-semibold text-cyan-100 disabled:opacity-40">
            Add
          </button>
          {atCap && (
            <p className="m-0 text-[0.625rem] text-amber-300">
              Limit of {MAX_GUEST_CHECKPOINTS} reached — remove one to add another.
            </p>
          )}
        </div>
      </details>
    </div>
  );
}

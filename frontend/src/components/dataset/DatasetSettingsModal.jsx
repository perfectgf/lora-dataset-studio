/**
 * DatasetSettingsModal — edit a dataset's identity after creation.
 *
 * Name and trigger word (any dataset), plus the concept description for concept
 * datasets. Changing the trigger is safe (it's prepended at export — no re-caption).
 * Changing the concept description is what drives the caption avoid-list, so editing
 * it resets that list; the parent's toast nudges a re-caption for existing captions.
 */
import { useState } from 'react';

const FIELD =
  'px-3 py-1.5 rounded-lg bg-surface-raised border border-border text-content text-sm ' +
  'placeholder:text-content-subtle focus:border-indigo-500 outline-none';

export default function DatasetSettingsModal({ d, busy, onSave, onClose }) {
  const concept = d.kind === 'concept';
  const [name, setName] = useState(d.name || '');
  const [trigger, setTrigger] = useState(d.trigger_word || '');
  const [desc, setDesc] = useState(d.concept_desc || '');

  const canSave = name.trim() && trigger.trim() && (!concept || desc.trim());
  const save = async () => {
    if (!canSave || busy) return;
    const res = await onSave({
      name: name.trim(),
      trigger_word: trigger.trim(),
      concept_desc: concept ? desc.trim() : undefined,
    });
    if (res?.ok) onClose();
  };

  return (
    <div role="dialog" aria-modal="true" aria-label="Dataset settings"
      className="fixed inset-0 z-[9990] bg-black/80 flex items-center justify-center p-3"
      onClick={onClose}>
      <div className="w-full max-w-md rounded-xl border border-border bg-surface-overlay p-4 flex flex-col gap-3"
        onClick={(e) => e.stopPropagation()}>
        <h2 className="text-content font-semibold flex items-center gap-1.5">⚙️ Dataset settings</h2>

        <label className="flex flex-col gap-1">
          <span className="text-content-muted text-xs">Name</span>
          <input value={name} onChange={(e) => setName(e.target.value)} className={FIELD} />
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-content-muted text-xs">Trigger word</span>
          <input value={trigger} onChange={(e) => setTrigger(e.target.value)}
            placeholder="e.g. myTrigger" className={`${FIELD} font-mono`} />
          <span className="text-content-subtle text-[0.6875rem]">
            The word you put in prompts to summon this LoRA. Safe to change anytime —
            it&apos;s added at export, so existing captions don&apos;t need redoing.
          </span>
        </label>

        {concept && (
          <label className="flex flex-col gap-1">
            <span className="text-content-muted text-xs">Concept description — what captions must OMIT</span>
            <textarea value={desc} onChange={(e) => setDesc(e.target.value)} rows={2}
              placeholder="e.g. a mirror selfie / a specific pose / an art style"
              className={`${FIELD} resize-y`} />
            <span className="text-content-subtle text-[0.6875rem]">
              This is the thing the LoRA learns. Captions describe everything <b>except</b> this,
              so it binds to the trigger. Editing it rebuilds the auto avoid-list —
              <b> re-caption</b> to apply it to images already captioned.
            </span>
          </label>
        )}

        <div className="flex justify-end gap-2 pt-1">
          <button type="button" onClick={onClose}
            className="px-3 py-1.5 rounded-lg border border-border bg-surface text-content-muted hover:text-content text-sm">
            Cancel
          </button>
          <button type="button" onClick={save} disabled={!canSave || busy}
            className="px-3 py-1.5 rounded-lg bg-gradient-primary text-white text-sm font-semibold disabled:opacity-40">
            Save
          </button>
        </div>
      </div>
    </div>
  );
}

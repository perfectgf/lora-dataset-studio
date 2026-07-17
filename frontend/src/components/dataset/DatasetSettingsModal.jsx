/**
 * DatasetSettingsModal — edit a dataset's identity after creation.
 *
 * Name and trigger word for character/concept datasets, plus the concept
 * description for concept datasets. Style is explicitly always-on and never
 * exposes its internal run identifier as an activation trigger.
 * Changing the concept description is what drives the caption avoid-list, so editing
 * it resets that list; the parent's toast nudges a re-caption for existing captions.
 *
 * Prompt suffixes (collapsible, community feature request): a free creative
 * direction — one global text + one per framing (face/bust/body/back) — appended
 * to every GENERATED variation at generation time. Never stored into the
 * per-image prompt (a regenerate would double-apply it), never ahead of the
 * identity lock. The whole map is replaced on save; empty fields clear.
 */
import { useState } from 'react';

const FIELD =
  'px-3 py-1.5 rounded-lg bg-surface-raised border border-border text-content text-sm ' +
  'placeholder:text-content-subtle focus:border-indigo-500 outline-none';

const SUFFIX_FRAMINGS = [
  ['face', 'Face'], ['bust', 'Bust'], ['body', 'Body'], ['back', 'Back'],
];

export default function DatasetSettingsModal({ d, busy, onSave, onClose }) {
  const concept = d.kind === 'concept';
  const style = d.kind === 'style';
  const [name, setName] = useState(d.name || '');
  const [trigger, setTrigger] = useState(d.trigger_word || '');
  const [desc, setDesc] = useState(d.concept_desc || '');
  const stored = d.prompt_suffixes || {};
  const [gSuffix, setGSuffix] = useState(d.prompt_suffix || '');
  const [fSuffix, setFSuffix] = useState({
    face: stored.face || '', bust: stored.bust || '',
    body: stored.body || '', back: stored.back || '',
  });
  // Discreet by default: open only when a suffix is already set.
  const [suffixOpen, setSuffixOpen] = useState(Boolean(
    (d.prompt_suffix || '').trim()
    || Object.values(stored).some((v) => (v || '').trim())));

  const canSave = name.trim() && (style || trigger.trim()) && (!concept || desc.trim());
  const save = async () => {
    if (!canSave || busy) return;
    const res = await onSave({
      name: name.trim(),
      trigger_word: style ? (d.trigger_word || '') : trigger.trim(),
      concept_desc: concept ? desc.trim() : undefined,
      // Always sent: '' / {} clear on the server (the map is replaced whole).
      prompt_suffix: gSuffix.trim(),
      prompt_suffixes: Object.fromEntries(
        Object.entries(fSuffix)
          .map(([k, v]) => [k, (v || '').trim()])
          .filter(([, v]) => v)),
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

        {style ? (
          <div className="rounded-lg border border-cyan-400/30 bg-cyan-500/10 px-3 py-2 text-[0.75rem] text-cyan-100">
            <b>Always-on Style:</b> no activation trigger is written into captions or prompts.
            Control the effect with the LoRA weight; when combining with a character LoRA,
            tune the two weights independently.
          </div>
        ) : (
          <label className="flex flex-col gap-1">
            <span className="text-content-muted text-xs">Trigger word</span>
            <input value={trigger} onChange={(e) => setTrigger(e.target.value)}
              placeholder="e.g. myTrigger" className={`${FIELD} font-mono`} />
            <span className="text-content-subtle text-[0.6875rem]">
              The word you put in prompts to summon this LoRA. Safe to change anytime —
              it&apos;s added at export, so existing captions don&apos;t need redoing.
            </span>
          </label>
        )}

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

        <div className="flex flex-col gap-1">
          <button type="button" onClick={() => setSuffixOpen(!suffixOpen)}
            aria-expanded={suffixOpen}
            className="flex items-center gap-1.5 text-left text-content-muted hover:text-content text-xs font-medium">
            <span className={`transition-transform ${suffixOpen ? 'rotate-90' : ''}`}>▸</span>
            ✨ Prompt suffixes
            <span className="text-content-subtle font-normal">— optional creative direction</span>
          </button>
          {suffixOpen && (
            <div className="flex flex-col gap-2 rounded-lg border border-border bg-surface px-3 py-2.5">
              <label className="flex flex-col gap-1">
                <span className="text-content-muted text-xs">All shots</span>
                <input value={gSuffix} onChange={(e) => setGSuffix(e.target.value)}
                  maxLength={300} placeholder="e.g. shot on 35mm film, warm tones"
                  className={FIELD} />
              </label>
              <div className="grid grid-cols-2 gap-2">
                {SUFFIX_FRAMINGS.map(([key, label]) => (
                  <label key={key} className="flex flex-col gap-1">
                    <span className="text-content-muted text-xs">{label} shots</span>
                    <input value={fSuffix[key]} maxLength={300}
                      onChange={(e) => setFSuffix({ ...fSuffix, [key]: e.target.value })}
                      className={FIELD} />
                  </label>
                ))}
              </div>
              <span className="text-content-subtle text-[0.6875rem]">
                Free text added to every <b>generated</b> variation — the identity lock is
                untouched. A framing suffix applies to that shot type first, then the global
                one. Applied at generation time: safe to change anytime, existing images
                stay as they are (regenerate to apply).
              </span>
            </div>
          )}
        </div>

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

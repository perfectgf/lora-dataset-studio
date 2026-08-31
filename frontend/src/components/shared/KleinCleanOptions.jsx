/* The 🧽 Klein clean's three dials — ONE control, mounted in BOTH clean surfaces (the
   bank's Level 3 panel and the dataset's Clean bar), write-through persisted exactly
   like the scan dials beside them: `watermark_clean.*` is what the clean lane reads on
   every route, so saving IS arming, on both surfaces at once.

   WHY IT EXISTS. The clean had exactly one option — which Klein model — and everything
   else was a constant in the backend: the prompt ("remove watermark"), the 2 MP
   processing cap, and the resample back to the file's own size. The maintainer asked
   the question a user would ask ("we can't see the prompt? we can't choose the output
   MP?"), and there was no answer: a mark that survived left nothing to turn.

   WHY IT PERSISTS RATHER THAN POSTING PER RUN. The clean is launched from five places
   (bank batch, dataset batch, the review lightbox one image at a time, ⚖ compare, a
   retry) through four layers of service. A per-run argument would have to be threaded
   through all of them and would still leave the lightbox and the compare dialog
   running yesterday's prompt. One stored value read at the seam gives every route the
   same behaviour by construction — which is also what the parity rule asks for.

   Values and clamps live in utils/kleinCleanOptions.js (pure JS) so `node --test` can
   cover them without a JSX parser. */
import { useState } from 'react';
import { putJson } from '../../api/fetchClient';
import {
  CLEAN_OUTPUT_MODES, cleanPromptText, clampMaxMp, formatMp, maxMpChoices, mpNote,
  normalizeOutput, outputNote,
} from '../../utils/kleinCleanOptions.js';

export default function KleinCleanOptions({ caps = {}, disabled = false, className = '',
  onChanged }) {
  /* What WE have written this session wins over the published caps: the save already
     reached the backend, and whether the parent re-probes is not this component's
     business. Everything untouched follows caps, which is how the panel fills in when
     the probe lands after the first render. */
  const [saved, setSaved] = useState({});
  const prompt = cleanPromptText(saved.klein_prompt ?? caps.watermark_clean_prompt);
  const maxMp = clampMaxMp(saved.klein_max_mp ?? caps.watermark_clean_max_mp);
  const output = normalizeOutput(saved.klein_output ?? caps.watermark_clean_output);

  /* null = "the box shows the stored value". Keeping the draft OUT of state until the
     user types is what lets a late caps arrival fill the field without ever
     overwriting something somebody is in the middle of typing. */
  const [draft, setDraft] = useState(null);
  const [saving, setSaving] = useState(false);
  const [failed, setFailed] = useState(false);
  const text = draft === null ? prompt : draft;
  const dirty = draft !== null && cleanPromptText(draft) !== prompt;

  const save = async (patch) => {
    setSaving(true);
    setFailed(false);
    try {
      await putJson('/api/settings', { config: { watermark_clean: patch } });
      setSaved((s) => ({ ...s, ...patch }));
      onChanged?.(patch);
    } catch {
      // The pass still uses the STORED value, so a silent swallow would leave the box
      // showing an instruction that will not run. Say so instead.
      setFailed(true);
    }
    setSaving(false);
  };

  const savePrompt = async (value) => {
    const next = cleanPromptText(value);
    if (next === prompt) { setDraft(null); return; }
    await save({ klein_prompt: next });
    setDraft(null);                     // follow the stored value again
  };

  return (
    <div className={`min-w-0 space-y-2 text-[11px] text-content-subtle ${className}`}>
      <label className="block min-w-0">
        <span className="font-medium text-content">Prompt sent to Klein</span>
        {' — stored: the other surface reads the same value.'}
        {/* A textarea, because an instruction can run to a sentence and a single-line
            input hides its own end. Enter SAVES (this is one instruction, not a
            paragraph) and Shift+Enter still breaks a line — said out loud below,
            because an Enter that does nothing visible reads as a failed save. */}
        <textarea
          rows={2}
          value={text}
          disabled={disabled || saving}
          aria-label="Prompt sent to Klein for watermark cleaning"
          onChange={(e) => setDraft(e.target.value)}
          onBlur={(e) => savePrompt(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              e.currentTarget.blur();   // onBlur saves — one path, not two
            }
          }}
          className="mt-1 block w-full min-w-0 resize-y rounded border border-border bg-app
                     px-1.5 py-1 text-content disabled:opacity-50"
        />
        <span className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1">
          <span>
            Enter saves, Shift+Enter adds a line. Short beats long here — the photo is
            being cleaned, not described.
          </span>
          {/* min-h-10 lg:min-h-0: this panel renders inside the bank's passes chrome,
              where the responsive probe holds every button to a finger-sized target
              below `lg` — and the fix for a short target is the height, never an
              exemption (.claude/rules/frontend-contracts.md). */}
          <button type="button" disabled={disabled || saving || !dirty}
            onClick={() => savePrompt(text)}
            /* Keep the caret where it is: without this the textarea blurs first, saves,
               and the click then saves again — two writes for one intent, and for
               Reset the FIRST of them would briefly store the text being discarded. */
            onMouseDown={(e) => e.preventDefault()}
            className="min-h-10 lg:min-h-0 underline text-content-muted hover:text-content disabled:opacity-40">
            Save
          </button>
          <button type="button" disabled={disabled || saving}
            onClick={() => savePrompt('')}
            title="Back to the three words the app ships with — the short instruction Klein answers best on a whole photo."
            onMouseDown={(e) => e.preventDefault()}
            className="min-h-10 lg:min-h-0 underline text-content-muted hover:text-content disabled:opacity-40">
            Reset to default
          </button>
          {dirty && !saving && (
            <span className="text-amber-300">Not saved yet — a run now uses “{prompt}”.</span>
          )}
          {failed && (
            <span className="text-amber-300">Could not save — a run still uses “{prompt}”.</span>
          )}
        </span>
      </label>

      {/* flex-wrap + min-w-0 throughout: label and select stack rather than overflow at
          400 px, the width every delivery is checked at. */}
      <label className="block min-w-0">
        <span className="font-medium text-content">Processing size</span>
        <span className="mt-1 flex flex-wrap items-center gap-2">
          <select value={formatMp(maxMp)} disabled={disabled || saving}
            aria-label="Klein clean processing size in megapixels"
            onChange={(e) => save({ klein_max_mp: Number(e.target.value) })}
            className="min-w-0 rounded border border-border bg-app px-1.5 py-0.5 text-content">
            {maxMpChoices(maxMp).map((mp) => (
              <option key={mp} value={formatMp(mp)}>
                {formatMp(mp)} MP{mp === 2 ? ' (default)' : ''}
              </option>
            ))}
          </select>
        </span>
        <span className="mt-1 block leading-snug">{mpNote(maxMp)}</span>
      </label>

      <label className="block min-w-0">
        <span className="font-medium text-content">Write back</span>
        <span className="mt-1 flex flex-wrap items-center gap-2">
          <select value={output} disabled={disabled || saving}
            aria-label="What dimensions the cleaned file is written at"
            onChange={(e) => save({ klein_output: e.target.value })}
            className="min-w-0 max-w-full rounded border border-border bg-app px-1.5 py-0.5 text-content">
            {CLEAN_OUTPUT_MODES.map((m) => (
              <option key={m.id} value={m.id}>{m.label}</option>
            ))}
          </select>
        </span>
        <span className={`mt-1 block leading-snug ${output === 'render' ? 'text-amber-300' : ''}`}>
          {outputNote(output)}
        </span>
      </label>
    </div>
  );
}

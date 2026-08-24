/* The "here is what improve will ask for, and here is where to change it" block,
   rendered wherever ✨ Upscale & improve can be triggered (the dataset lightbox,
   the grid's bulk toolbar, the generated-image lightbox). Reasoning, and the
   report behind it, in kleinImproveHint.js.

   ONE fetch, shared: /api/settings is a full settings payload and this note can
   mount several times per screen (every lightbox open, the bulk toolbar). The
   module-level promise below is the whole cache — with a short TTL, because a
   user who goes to Settings, edits the instruction and comes straight back must
   not be shown the text they just replaced. A failed request stays loaded:false
   and the note degrades to "there IS an instruction, it is editable" rather than
   inventing its content.

   IT NOW SETS, NOT ONLY QUOTES. This file used to declare itself "deliberately
   NOT a second editor" — pointing at Settings instead. That was reversed on
   purpose: reading the sentence that ruined your image and then being told to go
   find it somewhere else is most of the original complaint, not a fix for it.
   The rule that survives the reversal is the one that mattered in the first
   place: THERE IS STILL ONLY ONE TRUTH. What the box below writes is the GLOBAL
   `identity_prompts.klein_improve` setting — the same value Settings edits, with
   the same "blank = follow the shipped default" contract (shared, not
   re-implemented: PromptOverrideField + promptOverride.js). No per-image
   override, no per-run copy. The panel states its own reach out loud
   (IMPROVE_SCOPE_NOTE), because a control inside a dataset screen reads as
   per-dataset until something says otherwise.

   AND THAT IS WHAT MAKES THE CACHE THE LOAD-BEARING PART. The 15 s TTL existed
   for a reader; a WRITER turns it into a correctness problem, because this note
   is mounted several times at once and the other copies would keep quoting the
   text the user just replaced — the exact staleness the TTL was added to
   prevent, now caused from inside. So a successful save does not merely
   invalidate the cache: it PUBLISHES the payload the server returned to every
   mounted instance (`subscribers`) and seeds the cache with it. One write, every
   copy correct, and the next mount does not even re-fetch.

   Settings keeps its link: the other identity prompts, the per-subject picker
   and the four strength knobs are not here, and never will be. */
import { useCallback, useEffect, useId, useRef, useState } from 'react';
import { apiFetch, putJson } from '../../api/fetchClient';
import SettingsLink from '../common/SettingsLink';
import PromptOverrideField from '../common/PromptOverrideField';
import KleinModelSetting from '../shared/KleinModelSetting';
import { improveInstructionLine, improveAnimeCaution } from './kleinImproveHint';
import {
  IMPROVE_OFF_NOTE, IMPROVE_SCOPE_NOTE, createImproveSaver, effectiveImprovePrompt,
  improveEditorState, improveSettingsPatch,
} from './kleinImproveEditor';

const TTL_MS = 15000;
/* `value` is the RESOLVED payload, kept beside the promise so a note mounting
   while the cache is warm can render loaded on its very first frame instead of
   flashing the "there IS an instruction" wording for a tick. That matters more
   now than it did: opening the editor on a note that has not settled yet would
   show a box the user cannot tell from "no instruction". */
let cache = { at: 0, promise: null, value: null };
/** Every mounted note. A save reaches all of them — see the header. */
const subscribers = new Set();

function loadSettings() {
  const now = Date.now();
  if (!cache.promise || now - cache.at > TTL_MS) {
    const entry = {
      at: now,
      // Resolve to null on failure: the caller renders the honest "unknown"
      // wording instead of a toast about a hint nobody asked for.
      promise: apiFetch('/api/settings', { background: true }).catch(() => null),
      value: null,
    };
    cache = entry;
    entry.promise.then((d) => { if (cache === entry) cache.value = d || null; });
  }
  return cache.promise;
}

/** Hand a freshly saved settings payload to the cache AND to every mounted note.
 *  Invalidating alone would only fix the NEXT mount; the copies already on
 *  screen are the ones showing the replaced text. */
function publishSettings(payload) {
  if (!payload || typeof payload !== 'object') return;
  cache = { at: Date.now(), promise: Promise.resolve(payload), value: payload };
  // A listener that throws must not stop the ones after it — the whole point is
  // that ALL copies converge.
  subscribers.forEach((fn) => { try { fn(payload); } catch { /* keep going */ } });
}

/** Exported for tests/dev only — drops the shared payload cache. */
export function _resetKleinImproveNoteCache() {
  cache = { at: 0, promise: null, value: null };
}

/** Tests/dev only — makes a note render as if /api/settings had already
 *  answered. `node --test` runs no effects and no network, so this is the only
 *  way to execute the LOADED branches (the editor among them) outside a
 *  browser. Same category as the reset above; never called by the app. */
export function _seedKleinImproveNoteCache(payload) {
  cache = { at: Date.now(), promise: Promise.resolve(payload), value: payload };
}

export default function KleinImproveNote({
  subjectType = '', className = '', datasetId = null,
  /* Users always start closed — the note is a line, not a form, and it mounts in
     a 17 rem rail and in a horizontal bar where an always-open textarea would
     dominate the actions it annotates. The prop exists so the static render test
     can execute the OPEN branch: mountJsx runs no effects and fires no events,
     so a branch no test asks for is a branch nothing ever renders (the lesson
     `dropHint === 'leaving'` shipped). */
  defaultEditorOpen = false,
}) {
  // Seeded from the warm cache so a second note does not flash the loading line.
  const [payload, setPayload] = useState(() => cache.value);
  const [open, setOpen] = useState(defaultEditorOpen);
  // Several copies of this note can be open at once (rail + bulk toolbar), and
  // a duplicated DOM id would point every <label htmlFor> at the first textarea.
  const fieldId = `klein-improve-inline${useId()}`;
  // Local edits in flight. null = "follow the server". Kept apart from `payload`
  // so a publish landing from another copy of this note cannot overwrite the
  // sentence being typed here.
  const [draft, setDraft] = useState(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const alive = useRef(true);

  useEffect(() => {
    let cancelled = false;
    const receive = (d) => { if (!cancelled) setPayload(d); };
    loadSettings().then(receive);
    subscribers.add(receive);
    return () => { cancelled = true; subscribers.delete(receive); };
  }, []);

  const commit = useCallback(async (patch) => {
    if (alive.current) { setSaving(true); setError(null); }
    try {
      // PUT returns the WHOLE settings payload, so what every note ends up
      // showing is what the server stored, not what this one hoped it sent.
      publishSettings(await putJson('/api/settings', improveSettingsPatch(patch)));
    } catch (e) {
      if (alive.current) setError(e?.message || 'Could not save the instruction.');
    } finally {
      if (alive.current) setSaving(false);
    }
  }, []);

  const saver = useRef(null);
  if (!saver.current) saver.current = createImproveSaver((patch) => commit(patch));

  useEffect(() => {
    alive.current = true;
    const s = saver.current;
    // Closing the lightbox mid-sentence is the case that loses work: flush the
    // pending keystroke instead of letting the timer die with the component.
    return () => { alive.current = false; s.flush(); };
  }, []);

  const server = improveEditorState(payload);
  const stored = draft?.stored ?? server.stored;
  const enabled = draft?.enabled ?? server.enabled;
  const loraPreset = draft?.loraPreset ?? server.loraPreset;
  const state = {
    loaded: server.loaded,
    enabled,
    prompt: effectiveImprovePrompt({ stored, shipped: server.shipped }),
  };
  const line = improveInstructionLine(state);
  const caution = improveAnimeCaution({ ...state, subjectType });

  const setStored = (v) => {
    setDraft((d) => ({ ...(d || {}), stored: v, enabled: d?.enabled ?? server.enabled }));
    saver.current.schedule('prompt', v);
  };
  const setEnabled = (v) => {
    setDraft((d) => ({ ...(d || {}), stored: d?.stored ?? server.stored, enabled: v }));
    // A checkbox is a discrete act, not a stream of keystrokes: send it now.
    // schedule-then-flush rather than a direct call so a half-typed sentence
    // rides along in the SAME request instead of being overtaken by it.
    saver.current.schedule('enabled', v);
    saver.current.flush();
  };
  const setLoraPreset = (v) => {
    setDraft((d) => ({ ...(d || {}), loraPreset: v }));
    // A pick is discrete, like the checkbox — and rides with any half-typed
    // sentence for the same reason.
    saver.current.schedule('loraPreset', v);
    saver.current.flush();
  };

  return (
    // min-w-0 + break-words: the quoted instruction is user text of arbitrary
    // length and this sits inside flex rows that would otherwise be widened past
    // a 400 px screen by a single long word.
    <div className={`min-w-0 space-y-1 text-[0.6875rem] leading-relaxed ${className}`}>
      <p className="text-content-subtle break-words">
        {line.text}
        {line.quote && (
          <>
            {' '}
            <span className="text-content-muted italic" title={line.full}>“{line.quote}”</span>
          </>
        )}
      </p>
      {caution && (
        <p className="text-amber-300 break-words">{caution}</p>
      )}
      {/* WHICH model executes the instruction — the other half of the same
          question, and the half nothing on this screen ever answered. Named
          even on a one-model install; the picker itself only appears when
          there is more than one thing to pick. */}
      <KleinModelSetting datasetId={datasetId} />
      {/* WHICH LoRA preset the pass chains (klein.improve_lora_preset) — the
          third half of "what will improve run with", and app-wide exactly like
          the instruction, which the label says out loud because this control
          sits inside dataset screens. Only drawn when there IS something to
          pick — presets are defined in Settings ▸ Engines — or when a stale
          pick is stored, which must stay visible so it can be cleared (the
          backend already resolves it fail-closed to "none"). Klein only:
          SeedVR2 is a restoration and chains nothing. */}
      {(server.loraPresets.length > 0 || loraPreset) && (
        <label className="flex flex-wrap items-center gap-x-2 gap-y-1 min-w-0">
          <span className="text-content-muted">LoRA preset (app-wide, Klein only)</span>
          <select
            data-testid="klein-improve-lora-preset"
            aria-label="Generation-LoRA preset chained by every Klein improve"
            disabled={saving || !server.loaded}
            value={loraPreset}
            onChange={(e) => setLoraPreset(e.target.value)}
            className="min-w-0 max-w-full flex-1 bg-white/[0.03] border border-white/10 rounded-md
                       px-2 py-1 text-[0.6875rem] text-content focus:outline-none
                       focus:border-primary/60 disabled:opacity-50"
          >
            <option value="" className="bg-surface-overlay">None</option>
            {server.loraPresets.map((name) => (
              <option key={name} value={name} className="bg-surface-overlay">{name}</option>
            ))}
            {loraPreset && !server.loraPresets.includes(loraPreset) && (
              <option value={loraPreset} className="bg-surface-overlay">
                {loraPreset} (missing — runs as None)
              </option>
            )}
          </select>
        </label>
      )}
      {/* Two targets because they are two different problems: the WORDS
          (why it turned realistic) and the AMOUNT (how far it moved).
          flex-wrap so they stack rather than overflow on a phone. */}
      <p className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <button
          type="button"
          data-testid="klein-improve-edit-toggle"
          aria-expanded={open}
          onClick={() => setOpen((v) => !v)}
          disabled={!server.loaded}
          className="min-h-10 lg:min-h-0 inline-flex items-center underline text-indigo-300 hover:text-indigo-200 disabled:opacity-40"
        >
          <span aria-hidden="true">✎ </span>
          {open ? 'Close the instruction editor' : 'Edit this instruction here'}
        </button>
        <SettingsLink section="engines" focus="identity-prompt-klein-improve" className="min-h-10 lg:min-h-0 inline-flex items-center">
          All Klein prompts in Settings
        </SettingsLink>
        <SettingsLink section="engines" focus="klein-improve-strength" className="min-h-10 lg:min-h-0 inline-flex items-center">
          Adjust improve strength
        </SettingsLink>
      </p>
      {open && server.loaded && (
        // min-w-0 + w-full: the panel is the width of its host — 17 rem in the
        // lightbox rail, the toolbar's width in the grid — and never wider. The
        // textarea inside PromptOverrideField is w-full, so it follows.
        <div data-testid="klein-improve-editor"
          className="min-w-0 w-full rounded-lg border border-indigo-400/30 bg-indigo-500/[0.06] p-2 space-y-2">
          {/* Stated first, unconditionally, and before the box it qualifies:
              this control sits in a dataset screen but does not belong to it. */}
          <p className="text-amber-200/90 break-words">
            <span aria-hidden="true">⚠ </span>{IMPROVE_SCOPE_NOTE}
          </p>
          {/* "Turn off this instruction" was half of what the old Settings link
              promised. Losing it here would have made the panel a downgrade for
              the anime case, which is the case that started all of this. */}
          <label className="flex items-start gap-2 min-w-0 cursor-pointer">
            <input type="checkbox" checked={enabled} disabled={saving}
              onChange={(e) => setEnabled(e.target.checked)}
              className="mt-0.5 accent-indigo-400" />
            <span className="min-w-0 break-words text-content-muted">
              Send this instruction with every improve
              {!enabled && <span className="block text-content-subtle">{IMPROVE_OFF_NOTE}</span>}
            </span>
          </label>
          {/* The SAME field Settings uses, on purpose: the box shows the text in
              force, an edit becomes an override, and text equal to the shipped
              default collapses back to '' so nobody freezes a copy of a prompt
              that may improve later. "Reset to default" is rendered by that
              component only when there IS an override — its presence is the
              marker that says "you changed this". */}
          <PromptOverrideField
            id={fieldId}
            label="Improve instruction"
            value={stored}
            defaultText={server.shipped}
            onChange={setStored}
            /* NOT disabled while saving. A save fires 600 ms after the last
               keystroke, so disabling on it would take the field away from a
               user who is still typing — losing the caret, and the words after
               it. The saver coalesces instead: whatever is typed during a
               request rides in the next one. The checkbox above IS gated,
               because two overlapping toggles could land out of order. */
            disabled={!enabled}
            rows={4}
          />
          <p className="text-content-subtle break-words">
            {saving ? 'Saving…' : 'Saved automatically as you type.'}
          </p>
          {error && (
            <p className="text-rose-400 break-words"><span aria-hidden="true">✗ </span>{error}</p>
          )}
        </div>
      )}
    </div>
  );
}

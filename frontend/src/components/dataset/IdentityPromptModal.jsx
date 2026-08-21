/* ✎ Identity instruction — edit the multi-reference identity lock without
   leaving the workspace, right next to the Extra refs you just added.

   WHICH PROMPT(S). "Extra refs (all engines)" is true of the REFERENCES, not of
   the prompt: face_variations.py routes them through two different texts —
   wrap_variation picks `face_multi` for the API engines as soon as ref_count > 1
   (Nano Banana / ChatGPT), while the LOCAL engines — Klein and Krea 2 Edit,
   which share one prompt assembly — always use `klein_identity`, whatever the
   reference count. A modal editing only `face_multi` would let a local-engine
   user carefully rewrite a text that has ZERO effect on their generations. So
   both are shown, each labelled with the engine FAMILY that consumes it, and
   every prompt at least one SELECTED engine consumes carries a "used by your
   selected engine(s)" badge — a run on Klein + an API engine really uses both
   texts, and badging only one would say otherwise.

   WHICH SUBJECT. The modal edits the prompts of THIS dataset's subject type and
   says so, loudly. It used to edit one global text whatever the dataset was: a
   user on an Animal dataset was shown the HUMAN lock ("preserve their facial
   identity… jawline, lips, skin tone"), adapted it to animals, and that text
   then rode on every human dataset — extra limbs, tails, odd footwear (reported
   by ashish.sinha on Discord). An editor that does not name its subject is the
   whole bug, so the subject is in the title, in the intro line, and on a badge.

   Same storage semantics as Settings (shared PromptOverrideField): the box holds
   the shipped default, editing it creates an override, and text equal to the
   default normalises back to '' so nobody silently freezes a copy of a prompt
   that may improve in a later version. These prompts still apply to EVERY
   dataset that shares this subject type — the modal says that too. */
import { useCallback, useEffect, useState } from 'react';
import { apiFetch, putJson } from '../../api/fetchClient';
import { useToast } from '../common/Toast';
import { HelpBadge } from '../../help/HelpMode';
import PromptOverrideField from '../common/PromptOverrideField';
import {
  identityPromptFields, EXTRA_REF_PROMPT_KEYS, activeExtraRefPromptKeys,
  readIdentityPrompt, writeIdentityPrompt, identityPromptPatch, identityDefaultsFor,
} from '../common/promptOverride.js';
import { normalizeSubjectType, SUBJECT_TYPE_LABELS } from './subjectTypes.js';
import { readEngines } from './engineSelection.js';

export default function IdentityPromptModal({ onClose, subjectType = 'human' }) {
  const toast = useToast();
  const st = normalizeSubjectType(subjectType);
  const stLabel = SUBJECT_TYPE_LABELS[st];
  const [prompts, setPrompts] = useState(null);     // whole identity_prompts node
  const [payload, setPayload] = useState(null);     // settings payload (for defaults)
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const defaults = identityDefaultsFor(payload, st);
  const fields = identityPromptFields(st)
    .filter((f) => EXTRA_REF_PROMPT_KEYS.includes(f.key))
    .sort((a, b) => EXTRA_REF_PROMPT_KEYS.indexOf(a.key) - EXTRA_REF_PROMPT_KEYS.indexOf(b.key));
  // Every prompt at least ONE selected engine consumes — a run on Klein + an API
  // engine really uses both texts. readEngines reads the same storage
  // VariationCatalog writes (and degrades to the legacy single-engine key), so
  // private mode / an old profile still get a correct badge, never a crash.
  const activeKeys = activeExtraRefPromptKeys(
    readEngines(typeof localStorage === 'undefined' ? null : localStorage));

  useEffect(() => {
    let cancelled = false;
    apiFetch('/api/settings')
      .then((d) => {
        if (cancelled) return;
        setPrompts(d.config?.identity_prompts || {});
        setPayload(d);
      })
      .catch((e) => { if (!cancelled) setError(e.message || 'Could not load the prompts.'); });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  const save = useCallback(async () => {
    if (!prompts) return;
    setSaving(true);
    try {
      // PARTIAL config: only the keys this modal owns, FOR THIS SUBJECT TYPE.
      // /api/settings deep-merges, so the rest of identity_prompts (every other
      // subject included) is untouched by a save made from the workspace.
      const patch = identityPromptPatch(st, EXTRA_REF_PROMPT_KEYS, prompts);
      await putJson('/api/settings', { config: { identity_prompts: patch } });
      toast.success(`${stLabel} identity instruction saved.`);
      onClose();
    } catch (e) {
      setError(e.message || 'Save failed.');
    } finally {
      setSaving(false);
    }
  }, [prompts, st, stLabel, toast, onClose]);

  return (
    <div role="dialog" aria-modal="true" aria-label={`${stLabel} identity instruction for multiple references`}
      className="fixed inset-0 z-[9990] bg-black/80 flex items-center justify-center p-3"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="w-full max-w-2xl max-h-[90vh] overflow-y-auto rounded-xl border border-indigo-400/40 bg-app p-4 flex flex-col gap-3">
        {/* flex-wrap + a min-w-0 title: on a phone the title, the subject badge,
            the help dot and ✕ wrap onto two lines instead of pushing ✕ off-screen. */}
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-indigo-300 font-semibold min-w-0">
            <span aria-hidden>✎</span> Identity instruction — multiple references
          </span>
          <span className="rounded-full border border-indigo-400/50 bg-indigo-500/15 px-2 py-0.5 text-[0.625rem] font-semibold text-indigo-200">
            {stLabel} subject
          </span>
          <HelpBadge topic="action-edit-identity-prompt" />
          <button type="button" onClick={onClose}
            className="ml-auto text-content-subtle hover:text-content" aria-label="Close">✕</button>
        </div>

        <p className="text-content-muted text-xs leading-relaxed">
          The instruction sent ahead of every variation built from several reference photos.
          You are editing the <strong className="text-content">{stLabel.toLowerCase()}</strong> prompts:
          they apply to every <strong className="text-content">{stLabel.toLowerCase()}</strong> dataset
          and to no other subject type. Each engine family reads its own text, so both are
          here: edit the one your engine actually uses.
        </p>

        {error && <p className="text-xs text-rose-400"><span aria-hidden="true">✗</span> {error}</p>}

        {prompts === null && !error && (
          <p className="text-content-subtle text-xs">Loading…</p>
        )}

        {prompts !== null && fields.map((f) => (
          <PromptOverrideField
            key={f.key}
            id={`modal-${f.id}`}
            label={f.label}
            desc={f.desc}
            rows={5}
            value={readIdentityPrompt(prompts, st, f.key)}
            defaultText={defaults[f.key]}
            onChange={(v) => setPrompts((p) => writeIdentityPrompt(p, st, f.key, v))}
            badge={activeKeys.includes(f.key) ? (
              <span className="rounded-full border border-indigo-400/50 bg-indigo-500/15 px-2 py-0.5 text-[0.625rem] font-semibold text-indigo-200">
                used by your selected engine{activeKeys.length > 1 ? 's' : ''}
              </span>
            ) : null}
          />
        ))}

        <div className="flex items-center gap-2 flex-wrap pt-1">
          <a href="#/settings/engines" className="text-indigo-300 hover:text-indigo-200 text-xs underline decoration-indigo-300/50">
            All identity &amp; Klein prompts →
          </a>
          <button type="button" onClick={onClose}
            className="ml-auto px-3 py-1.5 rounded-lg bg-surface text-content text-sm">Cancel</button>
          <button type="button" onClick={save} disabled={saving || prompts === null}
            className="px-3 py-1.5 rounded-lg bg-gradient-primary text-white text-sm font-semibold disabled:opacity-40">
            {saving ? 'Saving…' : 'Save'}
          </button>
        </div>
      </div>
    </div>
  );
}

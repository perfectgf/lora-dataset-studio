/* ✎ Identity instruction — edit the multi-reference identity lock without
   leaving the workspace, right next to the Extra refs you just added.

   WHICH PROMPT(S). "Extra refs (all engines)" is true of the REFERENCES, not of
   the prompt: face_variations.py routes them through two different texts —
   wrap_variation picks `face_multi` for the API engines as soon as ref_count > 1
   (Nano Banana / ChatGPT), while wrap_variation_klein always uses
   `klein_identity`, whatever the reference count. A modal editing only
   `face_multi` would let a Klein user carefully rewrite a text that has ZERO
   effect on their generations. So both are shown, each labelled with the engine
   that consumes it, and the one matching the workspace's currently selected
   engine carries a "used by your current engine" badge.

   Same storage semantics as Settings (shared PromptOverrideField): the box holds
   the shipped default, editing it creates an override, and text equal to the
   default normalises back to '' so nobody silently freezes a copy of a prompt
   that may improve in a later version. These are GLOBAL settings — the modal
   says so, because it is opened from a per-dataset screen. */
import { useCallback, useEffect, useState } from 'react';
import { apiFetch, putJson } from '../../api/fetchClient';
import { useToast } from '../common/Toast';
import { HelpBadge } from '../../help/HelpMode';
import PromptOverrideField from '../common/PromptOverrideField';
import {
  IDENTITY_PROMPT_FIELDS, EXTRA_REF_PROMPT_KEYS, activeExtraRefPromptKey,
} from '../common/promptOverride.js';

const FIELDS = EXTRA_REF_PROMPT_KEYS
  .map((k) => IDENTITY_PROMPT_FIELDS.find((f) => f.key === k))
  .filter(Boolean);

/** The engine the workspace is currently generating with — the SAME source
 *  VariationCatalog persists its card selection to. Unreadable storage (private
 *  mode) just means no badge, never a crash. */
function currentGenerator() {
  try { return localStorage.getItem('datasetGenerator') || ''; } catch { return ''; }
}

export default function IdentityPromptModal({ onClose }) {
  const toast = useToast();
  const [prompts, setPrompts] = useState(null);     // stored overrides
  const [defaults, setDefaults] = useState({});     // shipped defaults (read-only)
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const activeKey = activeExtraRefPromptKey(currentGenerator());

  useEffect(() => {
    let cancelled = false;
    apiFetch('/api/settings')
      .then((d) => {
        if (cancelled) return;
        setPrompts(d.config?.identity_prompts || {});
        setDefaults(d.identity_prompt_defaults || {});
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
      // PARTIAL config: only the keys this modal owns. /api/settings deep-merges,
      // so the rest of identity_prompts (and every other section) is untouched by
      // a save made from the workspace.
      const patch = {};
      for (const k of EXTRA_REF_PROMPT_KEYS) patch[k] = prompts[k] ?? '';
      await putJson('/api/settings', { config: { identity_prompts: patch } });
      toast.success('Identity instruction saved.');
      onClose();
    } catch (e) {
      setError(e.message || 'Save failed.');
    } finally {
      setSaving(false);
    }
  }, [prompts, toast, onClose]);

  return (
    <div role="dialog" aria-modal="true" aria-label="Identity instruction for multiple references"
      className="fixed inset-0 z-[9990] bg-black/80 flex items-center justify-center p-3"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="w-full max-w-2xl max-h-[90vh] overflow-y-auto rounded-xl border border-indigo-400/40 bg-app p-4 flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <span className="text-indigo-300 font-semibold">
            <span aria-hidden>✎</span> Identity instruction — multiple references
          </span>
          <HelpBadge topic="action-edit-identity-prompt" />
          <button type="button" onClick={onClose}
            className="ml-auto text-content-subtle hover:text-content" aria-label="Close">✕</button>
        </div>

        <p className="text-content-muted text-xs leading-relaxed">
          The instruction sent ahead of every variation built from several reference photos.
          It is a <strong className="text-content">global</strong> setting — it applies to every
          dataset, not just this one. Each engine family reads its own text, so both are here:
          edit the one your engine actually uses.
        </p>

        {error && <p className="text-xs text-rose-400"><span aria-hidden="true">✗</span> {error}</p>}

        {prompts === null && !error && (
          <p className="text-content-subtle text-xs">Loading…</p>
        )}

        {prompts !== null && FIELDS.map((f) => (
          <PromptOverrideField
            key={f.key}
            id={`modal-${f.id}`}
            label={f.label}
            desc={f.desc}
            rows={5}
            value={prompts[f.key]}
            defaultText={defaults[f.key]}
            onChange={(v) => setPrompts((p) => ({ ...p, [f.key]: v }))}
            badge={f.key === activeKey ? (
              <span className="rounded-full border border-indigo-400/50 bg-indigo-500/15 px-2 py-0.5 text-[0.625rem] font-semibold text-indigo-200">
                used by your current engine
              </span>
            ) : null}
          />
        ))}

        <div className="flex items-center gap-2 pt-1">
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

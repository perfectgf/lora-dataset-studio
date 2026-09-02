/**
 * ⚙ Which local model writes the motion.
 *
 * A window rather than a select on the panel: the list is only wanted at the
 * moment somebody wonders about it, and a dropdown of every model a machine has
 * pulled would sit there permanently next to two buttons that do not need it.
 *
 * The list is the PROVIDER's own (the same one every other picker in this app
 * reads), so a model pulled in Ollama a minute ago is simply there. An
 * unreachable server says so and keeps the current choice visible — a picker
 * that emptied itself would read as "you have no models".
 */
import { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import { apiFetch, putJson } from '../../../../api/fetchClient';
import { useToast } from '../../../common/Toast';
import { motionModelUrl, motionModelsUrl } from './videoStudioApi';

export default function MotionModelDialog({ onClose, onSaved }) {
  const toast = useToast();
  const [state, setState] = useState(null);
  const [picked, setPicked] = useState(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    apiFetch(motionModelsUrl())
      .then((d) => { setState(d); setPicked(d?.current || ''); })
      .catch(() => setState({ models: [], reachable: false, current: '' }));
  }, []);

  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape' && !busy) onClose?.(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [busy, onClose]);

  const save = async (e) => {
    e.preventDefault();
    setBusy(true);
    try {
      const r = await putJson(motionModelUrl(), { model: picked || '' });
      onSaved?.(r.model || '');
      toast.success(r.model
        ? `✨ Motion will be written by ${r.model}.`
        : '✨ Motion goes back to the provider’s own vision model.');
      onClose?.();
    } catch (err) {
      toast.error(err?.message || 'That choice could not be saved.');
    } finally {
      setBusy(false);
    }
  };

  const models = state?.models || [];

  /* PORTAILLÉE sur `document.body`, comme toute modale du Studio.
     Aujourd'hui elle est montée en SŒUR de l'`<aside lg:sticky>` du lane vidéo
     (VideoTestStudio) et non dedans, donc son `z-50` passe encore — mesuré. Le
     portail est posé quand même : un `sticky` ou un `transform` sur un ancêtre,
     ou trois lignes de déplacement du montage, la feraient basculer SANS qu'une
     seule suite rougisse (c'est exactement ce qui est arrivé au navigateur
     Civitai). La règle est tenue par studioModalsArePortaled.contract.test.js,
     qui recense CE dossier et ses sous-dossiers. */
  return createPortal(
    <div role="dialog" aria-modal="true" aria-label="Model that writes the motion"
      data-probe-layer
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-2 sm:p-4"
      onMouseDown={(e) => { if (e.target === e.currentTarget && !busy) onClose?.(); }}>
      <form onSubmit={save}
        className="flex w-full max-w-md max-h-[92vh] flex-col overflow-hidden rounded-xl border border-border bg-surface-overlay shadow-2xl">
        <header className="shrink-0 space-y-1 border-b border-border p-4">
          <h2 className="text-base font-bold text-content">⚙ Model that writes the motion</h2>
          <p className="text-sm text-content-muted">
            Used by ✨ Auto and ✨ Enrich. Its own setting: the image passes keep
            the model you gave them.
          </p>
          {state && (
            <p className="text-xs text-content-subtle">
              Through {state.label || 'your local LLM'}
              {state.reachable ? '' : ' — not reachable right now'}
            </p>
          )}
        </header>

        <div className="min-h-0 flex-1 space-y-1.5 overflow-y-auto p-3 sm:p-4">
          <label className="flex cursor-pointer items-start gap-2 rounded-lg border border-border bg-surface-raised px-3 py-2 text-sm text-content has-[:checked]:border-sky-400/70 has-[:checked]:bg-sky-500/10">
            <input type="radio" name="motion-model" className="mt-0.5"
              checked={!picked} onChange={() => setPicked('')} />
            <span className="min-w-0">
              <span className="font-semibold">The provider’s own vision model</span>
              <span className="block text-xs text-content-muted">
                Whatever the image passes use. Nothing extra to keep in step.
              </span>
            </span>
          </label>

          {models.map((m) => (
            <label key={m}
              className="flex cursor-pointer items-start gap-2 rounded-lg border border-border bg-surface-raised px-3 py-2 text-sm text-content has-[:checked]:border-sky-400/70 has-[:checked]:bg-sky-500/10">
              <input type="radio" name="motion-model" className="mt-0.5"
                checked={picked === m} onChange={() => setPicked(m)} />
              <span className="min-w-0 break-all">{m}</span>
            </label>
          ))}

          {state && !models.length && (
            <p className="rounded-lg border border-dashed border-border px-3 py-4 text-xs text-content-muted">
              {state.reachable
                ? 'No model listed — pull one in your local LLM and reopen this.'
                : 'The server did not answer, so its models could not be listed. Your current choice is kept.'}
            </p>
          )}
        </div>

        <div className="shrink-0 border-t border-border p-3">
          <div className="flex flex-wrap justify-end gap-2">
            <button type="button" onClick={onClose} disabled={busy}
              className="min-h-10 rounded-md border border-border px-3 py-1.5 text-sm text-content-muted hover:bg-surface-raised hover:text-content disabled:opacity-50 lg:min-h-0">
              Cancel
            </button>
            <button type="submit" disabled={busy}
              className="min-h-10 rounded-md bg-gradient-primary px-4 py-1.5 text-sm font-semibold text-gray-950 disabled:opacity-50 lg:min-h-0">
              {busy ? 'Saving…' : 'Use this model'}
            </button>
          </div>
        </div>
      </form>
    </div>,
    document.body,
  );
}

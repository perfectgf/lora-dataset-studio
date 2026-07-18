import { useEffect, useRef, useState } from 'react';
import { readinessSignature, overrideAck } from '../../utils/readinessOverride';
import { HelpBadge } from '../../help/HelpMode';

/* Pastille de préparation à l'entraînement — miroir du preflight serveur
   (GET /train/preflight, champs checks+verdict) : 🟢 ready / 🟡 warnings /
   🔴 blocked, avec la liste des contrôles dépliable. Chaque ligne en défaut
   qui cible une section du workspace porte un bouton « Fix → » (onJump).
   Re-fetch débouncé quand les compteurs pertinents changent (curation,
   captions, fuites) — pas à chaque poll (le preflight relit les images sur
   disque pour le dHash). Rendu nul tant que rien n'est chargé ou si le
   backend gate (ai-toolkit absent → 409). */

const VERDICT = {
  ready: { icon: '🟢', label: 'Ready to train', cls: 'border-emerald-400/40 bg-emerald-500/10' },
  warnings: { icon: '🟡', label: 'Almost ready', cls: 'border-amber-400/40 bg-amber-500/10' },
  blocked: { icon: '🔴', label: 'Not ready', cls: 'border-red-400/40 bg-red-500/10' },
};
const ROW_ICON = { ok: '✓', warn: '⚠', fail: '✕' };
const ROW_CLS = { ok: 'text-emerald-400', warn: 'text-amber-300', fail: 'text-red-300' };

export default function TrainingReadiness({ datasetId, trainType, variant, refreshKey, onJump,
                                            onOverrideChange }) {
  const [data, setData] = useState(null);
  const [open, setOpen] = useState(false);
  // « Continue anyway » : ack de l'utilisateur pour lever un blocker QUALITÉ. Se
  // DÉcoche à chaque changement de l'état bloquant (signature) — jamais d'ack
  // fantôme qui survivrait à un nouveau blocker (physique compris).
  const [ack, setAck] = useState(false);
  const timer = useRef(null);
  useEffect(() => {
    let alive = true;
    // Débounce : les compteurs bougent en rafale pendant une passe de caption.
    clearTimeout(timer.current);
    timer.current = setTimeout(async () => {
      try {
        const params = new URLSearchParams();
        if (trainType) params.set('train_type', trainType);
        if (variant) params.set('variant', variant);
        const qs = params.size ? `?${params.toString()}` : '';
        const r = await fetch(`/api/dataset/${datasetId}/train/preflight${qs}`, { credentials: 'include' });
        if (!r.ok) { if (alive) setData(null); return; }   // 409 ai-toolkit absent → rien
        const d = await r.json();
        if (alive && d.ok) setData(d);
      } catch { /* transient — le prochain changement de compteur retentera */ }
    }, 400);
    return () => { alive = false; clearTimeout(timer.current); };
  }, [datasetId, trainType, variant, refreshKey]);

  // Reset the ack whenever the blocking state changes (a new blocker, a fixed one,
  // an override that just became unavailable) so a stale tick never rides forward.
  const sig = readinessSignature(data);
  useEffect(() => { setAck(false); }, [sig]);
  // Report the server-authoritative ack upward (false unless the override is both
  // offered AND ticked). Runs on null data too → parent clears its state.
  useEffect(() => { onOverrideChange?.(overrideAck(data, ack)); },
    [sig, ack, data, onOverrideChange]);

  if (!data || !(data.checks || []).length) return null;
  const v = VERDICT[data.verdict] || VERDICT.warnings;
  const warns = data.checks.filter((c) => c.status === 'warn').length;
  const fails = data.checks.filter((c) => c.status === 'fail').length;
  const subtitle = data.verdict === 'ready'
    ? `${data.checks.length} checks passed`
    : [fails && `${fails} blocker(s)`, warns && `${warns} warning(s)`].filter(Boolean).join(' · ');

  return (
    <div className={`rounded-lg border ${v.cls}`}>
      <button type="button" onClick={() => setOpen((o) => !o)} aria-expanded={open}
        className="w-full flex items-center gap-2 px-3 py-2 text-left">
        <span aria-hidden>{v.icon}</span>
        <span className="text-content text-sm font-semibold">{v.label}</span>
        <span className="text-content-subtle text-[0.6875rem]">{subtitle}</span>
        <span aria-hidden className="ml-auto text-content-subtle text-xs">{open ? '▾' : '▸'}</span>
      </button>
      {open && (
        <ul className="m-0 px-3 pb-2.5 flex flex-col gap-1 list-none">
          {data.checks.map((c) => (
            <li key={c.id} className="flex items-start gap-2 text-[0.75rem]">
              <span aria-hidden className={`w-4 shrink-0 text-center font-bold ${ROW_CLS[c.status]}`}>
                {ROW_ICON[c.status]}
              </span>
              <span className="text-content">{c.label}</span>
              <span className="text-content-subtle">— {c.detail}</span>
              {c.status !== 'ok' && c.target && (
                <button type="button" onClick={() => onJump?.(c.target)}
                  className="ml-auto shrink-0 px-1.5 py-0.5 rounded border border-border text-content-muted hover:text-content hover:bg-surface-raised text-[0.6875rem]">
                  Fix →
                </button>
              )}
            </li>
          ))}
        </ul>
      )}
      {/* « Continue anyway » : offerte UNIQUEMENT quand tous les blockers sont des
          garde-fous QUALITÉ contournables (data.can_override, miroir du garde
          serveur). Une impossibilité physique (0 image, prompt slider absent) →
          la case ne s'affiche pas et le launch reste refusé. Toujours visible
          (hors dépliage) tant qu'il y a un blocker contournable. */}
      {data.can_override && (
        <div className="flex items-start gap-1 px-3 pb-2.5 pt-1 text-[0.75rem]">
          <label className="flex items-start gap-2 cursor-pointer">
            <input type="checkbox" checked={ack} onChange={(e) => setAck(e.target.checked)}
              className="accent-red-400 w-3.5 h-3.5 mt-0.5 shrink-0" />
            <span className="flex flex-col gap-0.5">
              <span className="text-content font-medium">
                Continue anyway — train with these issues unresolved
              </span>
              {data.override_hint && (
                <span className="text-content-subtle">{data.override_hint}</span>
              )}
            </span>
          </label>
          <HelpBadge topic="training-continue-anyway" />
        </div>
      )}
    </div>
  );
}

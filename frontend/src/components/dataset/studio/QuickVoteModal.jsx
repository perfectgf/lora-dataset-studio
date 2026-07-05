// react-frontend/src/components/dataset/studio/QuickVoteModal.jsx
/**
 * File de vote rapide (modal plein écran) : 👎 / passer / 👍, swipe tactile et
 * clavier ←/→/Échap. Extrait 1:1 du bloc `{voteQueue && (...)}` de l'ancien
 * LoraTestStudio (behavior-preserving). Le clavier (←/→/Échap) est géré par le
 * hook useQuickVote — on ne le re-implémente PAS ici. Espace n'est volontairement
 * pas intercepté (correctif a11y : laisse l'activation native du bouton focalisé).
 *
 * Le focus-trap garde Tab dans le dialog (useFocusTrap).
 */
import { useRef } from 'react';
import { useFocusTrap } from '../../../hooks/useFocusTrap';

export default function QuickVoteModal({ vote, datasetId, fmt }) {
  const ref = useRef(null);
  useFocusTrap(ref, !!vote.current);

  if (!vote.voteQueue || !vote.current) return null;
  const cur = vote.current;

  return (
    <div ref={ref}
      className="fixed inset-0 z-[9999] bg-black/95 flex flex-col items-center justify-center p-4 gap-3"
      onTouchStart={vote.onTouchStart} onTouchEnd={vote.onTouchEnd}
      role="dialog" aria-modal="true" aria-label="Quick vote">
      <button type="button" onClick={() => vote.close()} aria-label="Close vote"
        className="absolute top-4 right-4 w-10 h-10 rounded-full bg-white/10 border border-white/20 text-white text-lg z-10 hover:bg-white/20">×</button>
      {vote.voteTitle && (
        <div className="px-3 py-1 rounded-full bg-green-500/20 border border-green-400/50 text-green-200 text-xs font-semibold">
          {vote.voteTitle}
        </div>
      )}
      <div className="text-content-subtle text-xs tabular-nums text-center">
        {vote.voteIdx + 1} / {vote.voteQueue.length} · {cur.label} · strength {fmt(cur.strength)}
        {cur.aspect ? ` · ${cur.aspect}` : ''}
      </div>
      <img src={`/api/dataset/${datasetId}/img/${encodeURIComponent(cur.filename)}`}
        alt={cur.label}
        className="max-w-[92vw] max-h-[64vh] object-contain rounded-lg border border-white/15" />
      <div className="text-content-subtle text-[0.625rem]">← swipe/left arrow = 👎 · right = 👍 → · Esc = close · ("skip" button to pass)</div>
      <div className="flex items-center gap-4">
        <button type="button" onClick={() => vote.voteCurrent(-1)} aria-label="Dislike"
          className="px-7 py-3 rounded-2xl text-2xl border border-red-400/60 bg-red-500/20 text-red-200 hover:bg-red-500/30">👎</button>
        <button type="button" onClick={vote.advanceVote} aria-label="Skip"
          className="px-3 py-2 rounded-xl text-xs border border-border bg-surface text-content-muted hover:text-content">skip</button>
        <button type="button" onClick={() => vote.voteCurrent(1)} aria-label="Like"
          className="px-7 py-3 rounded-2xl text-2xl border border-green-400/60 bg-green-500/20 text-green-200 hover:bg-green-500/30">👍</button>
      </div>
    </div>
  );
}
